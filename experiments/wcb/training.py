from __future__ import annotations
import copy
import os
from pathlib import Path
import random
import time
import numpy as np
import pandas as pd
import torch
from .common import digest,read_json,write_json,seed_for
from .data import Prepared
from .masks import matched_masks,global_mask,sigmoid,limited_update,curriculum
from .model import PlainMAE,masked_mse
from .evaluation import evaluate_bank,linear_probe

def device_for(c):
    if not c['synthetic'] and (not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available()):
        raise RuntimeError('Real training/profile requires a Slurm allocation with CUDA; no CPU fallback')
    return torch.device(c['device'])

def initialize(c,seed):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(c['cpu_threads'])
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    model=PlainMAE(c['patch_samples'],c['model_size'])
    return model

def clone_cpu(obj):
    if isinstance(obj,torch.Tensor): return obj.detach().cpu().clone()
    if isinstance(obj,dict): return {k:clone_cpu(v) for k,v in obj.items()}
    if isinstance(obj,list): return [clone_cpu(v) for v in obj]
    if isinstance(obj,tuple): return tuple(clone_cpu(v) for v in obj)
    return copy.deepcopy(obj)

def rng_state():
    return dict(torch=torch.get_rng_state(),numpy=np.random.get_state(),python=random.getstate(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])

def set_rng(s):
    torch.set_rng_state(s['torch']); np.random.set_state(s['numpy']); random.setstate(s['python'])
    if s['cuda']: torch.cuda.set_rng_state_all(s['cuda'])

def capture(model,optimizer):
    return dict(model=clone_cpu(model.state_dict()),optimizer=clone_cpu(optimizer.state_dict()),rng=rng_state())

def restore(state,model,optimizer):
    model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer']); set_rng(state['rng'])

def save_checkpoint(path,state):
    tmp=path.with_suffix('.tmp'); torch.save(state,tmp); tmp.replace(path)

def lr_at(step,c):
    if c['lr_warmup'] and step<=c['lr_warmup']: return c['lr']*step/c['lr_warmup']
    fraction=np.clip((step-c['lr_warmup'])/max(1,c['lr_horizon']-c['lr_warmup']),0,1)
    return float(c['min_lr']+(c['lr']-c['min_lr'])*.5*(1+np.cos(np.pi*fraction)))

def block_data(prepared,seed,step,h,rates,method):
    c=prepared.c; q=2500//c['patch_samples']; fit=np.array(prepared.selection['fit'])
    batches=[]; all_masks=[[] for _ in rates]; stats=[[] for _ in rates]
    for offset in range(h):
        absolute=step+offset+1
        rng=np.random.default_rng(seed_for(seed,'data',absolute))
        ids=rng.choice(fit,c['microbatch']*c['accumulation'],replace=True)
        masks=[[] for _ in rates]
        for j,index in enumerate(ids):
            mask_seed=seed_for(seed,'mask',absolute,j)
            kind='global' if method in ('random','random_best2') else 'conditional' if method=='conditional' else 'structured'
            if method=='random_best2':
                answers=[matched_masks([],q,[r],seed_for(mask_seed,branch),kind='global')[0]
                         for branch,r in enumerate(rates)]
            else:
                groups=prepared.groups(int(index)) if kind!='global' else []
                answers=matched_masks(groups,q,rates,mask_seed,k=c['targets'],kind=kind,
                                      structured_fraction=c['structured_fraction'])
            for branch,(mask,info) in enumerate(answers):
                masks[branch].append(mask); stats[branch].append(info)
        batches.append(ids)
        for branch in range(len(rates)): all_masks[branch].append(np.array(masks[branch]))
    return batches,all_masks,stats

def train_block(model,optimizer,prepared,ids,masks,step,device):
    c=prepared.c; losses=[]
    for offset,(indices,mask) in enumerate(zip(ids,masks)):
        model.train(); optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups: group['lr']=lr_at(step+offset+1,c)
        value=0.
        for start in range(0,len(indices),c['microbatch']):
            xt=torch.tensor(np.asarray(prepared.x[indices[start:start+c['microbatch']]]),device=device)
            mt=torch.tensor(mask[start:start+c['microbatch']],device=device)
            prediction,target=model(xt,mt); loss=masked_mse(prediction,target,mt)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
            (loss/c['accumulation']).backward(); value+=float(loss.detach())/c['accumulation']
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),c['grad_clip'])
        if not torch.isfinite(norm): raise RuntimeError('Nonfinite gradients')
        optimizer.step(); losses.append(value)
    return float(np.mean(losses))

def policy_candidates(method,x,c,block,step,seed,replay=None):
    rng=np.random.default_rng(seed_for(seed,'policy',block))
    direction=np.zeros(3); z=None
    if method=='coordinate': direction[block%3]=1
    elif method=='spsa': direction=rng.choice([-1.,1.],3)
    if method in ('coordinate','spsa'):
        rates=[sigmoid(x+c['delta']*direction),sigmoid(x-c['delta']*direction)]
    elif method=='reinforce': z=x+rng.normal(0,c['sigma'],3); rates=[sigmoid(z)]
    elif method=='random_best2': rates=[np.full(3,.5),np.full(3,.5)]
    elif method=='curriculum': rates=[curriculum(step/c['updates'])]
    elif method=='replay': rates=[np.array(replay[block]['committed_rates'])]
    else: rates=[np.array(c['fixed_rates'])]
    return rates,direction,z

def summarize_masks(stats):
    structured=[s for s in stats if 'realized' in s]
    rates=[]
    for g in range(3):
        values=[s['realized'][g] for s in structured if s['realized'][g] is not None]
        rates.append(float(np.mean(values)) if values else None)
    return dict(inputs=len(stats),structured_inputs=len(structured),realized=rates,
        mean_targets=float(np.mean([s['targets'] for s in stats])),
        fallback_inputs=sum(s.get('fallback',False) for s in stats),
        no_annotation_inputs=sum(s.get('no_annotations',False) for s in stats),
        U_candidates_mean=float(np.mean([s['U_candidates'] for s in structured])) if structured else None,
        U_visible_mean=float(np.mean([s['U_visible'] for s in structured])) if structured else None)

def run(study,plan,method,seed,*,stop_after=None,replay=None,folder=None):
    c=plan['config']; folder=Path(folder) if folder else Path(study)/'runs'/method/f'seed{seed}'
    folder.mkdir(parents=True,exist_ok=True)
    contract=dict(plan_id=plan['plan_id'],method=method,seed=seed,replay=digest(replay) if replay else None,
                  prepared_id=read_json(Path(study)/'prepared/completed.json')['prepared_id'])
    if (folder/'completed.json').exists():
        if read_json(folder/'completed.json')['contract']!=contract: raise ValueError('Completed run mismatch')
        return read_json(folder/'completed.json')
    device=device_for(c); model=initialize(c,seed).to(device)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=c['lr'],
                               betas=(.9,.95),weight_decay=c['weight_decay'])
    mu=torch.nn.Parameter(torch.zeros(3)); policy_optimizer=torch.optim.Adam([mu],lr=c['policy_lr'])
    x=np.zeros(3); baseline=0.; step=0; block=0; history=[]; seconds=0.; attempted=0; saved_balanced=None
    prepared=Prepared(study,plan); checkpoint=folder/'checkpoint.pt'
    if checkpoint.exists():
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if state['contract']!=contract: raise ValueError('Resume contract mismatch')
        restore(state['training'],model,optimizer)
        x=np.array(state['x']); mu.data.copy_(torch.tensor(x,dtype=torch.float32))
        policy_optimizer.load_state_dict(state['policy_optimizer'])
        step,block,baseline,history,seconds,attempted=(state[k] for k in ['step','block','baseline','history','seconds','attempted'])
        saved_balanced=state['last_balanced']
    write_json(folder/'status.json',dict(status='running',contract=contract,step=step))
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
    evaluate=lambda name: evaluate_bank(model,prepared.x,prepared.root/f'{name}_control.npz',device,c['microbatch'])
    before=evaluate('random')['loss']
    balanced_start=evaluate('balanced')
    balanced_initial=balanced_start if step==0 else None
    before_reward=before if c['reward']=='random' else balanced_start['loss']
    last_balanced=balanced_start['loss'] if saved_balanced is None else saved_balanced
    limit=min(c['updates'],step+stop_after) if stop_after else c['updates']
    if replay and len(replay)*c['H']<limit: raise ValueError('Replay schedule too short')
    while step<limit and not (c['budget_seconds'] and seconds>=c['budget_seconds']):
        started=time.perf_counter(); h=min(c['H'],c['updates']-step)
        # Boundary-only checkpoints: stop_after never truncates an H block.
        rates,direction,z=policy_candidates(method,x,c,block,step,seed,replay)
        ids,masks,mask_stats=block_data(prepared,seed,step,h,rates,method)
        start_state=capture(model,optimizer) if len(rates)>1 else None
        scores=[]; reward_scores=[]; branch_balanced=[]; train_losses=[]; plus=None
        for branch in range(len(rates)):
            if branch: restore(start_state,model,optimizer)
            train_losses.append(train_block(model,optimizer,prepared,ids,masks[branch],step,device))
            scores.append(evaluate('random')['loss'])
            bal=evaluate('balanced') if c['reward']=='balanced' else None
            branch_balanced.append(bal)
            reward_scores.append(scores[-1] if bal is None else bal['loss'])
            attempted+=h
            if branch==0 and len(rates)>1: plus=capture(model,optimizer)
        chosen=int(np.argmin(reward_scores))
        if len(rates)>1 and reward_scores[0]==reward_scores[1]: chosen=int(seed_for(seed,block,'tie')%2)
        if plus is not None and chosen==0: restore(plus,model,optimizer)
        rewards=np.array([before_reward-s for s in reward_scores])
        random_rewards=[before-s for s in scores]
        old_x=x.copy(); advantage=None
        if method in ('coordinate','spsa'):
            gradient=(rewards[0]-rewards[1])/(2*c['delta'])*direction
            x=limited_update(x,x+c['policy_lr']*gradient,c['max_rate_change'])
        elif method=='reinforce':
            advantage=float(rewards[0]-baseline)
            policy_optimizer.zero_grad(set_to_none=True)
            sample=torch.tensor(z,dtype=torch.float32)
            loss=-advantage*torch.distributions.Normal(mu,c['sigma']).log_prob(sample).sum()
            loss.backward(); torch.nn.utils.clip_grad_norm_([mu],1.); policy_optimizer.step()
            x=limited_update(old_x,mu.detach().numpy(),c['max_rate_change'])
            mu.data.copy_(torch.tensor(x,dtype=torch.float32)); baseline=.9*baseline+.1*float(rewards[0])
        step+=h; block+=1; before=scores[chosen]; before_reward=reward_scores[chosen]
        balanced=branch_balanced[chosen] or (evaluate('balanced') if block%c['balanced_every']==0 or step>=c['updates'] else None)
        balanced_progress=None if balanced is None else last_balanced-balanced['loss']
        if balanced is not None: last_balanced=balanced['loss']
        record=dict(step=step,block=block,H=h,center_before=sigmoid(old_x).tolist(),
            candidate_rates=[r.tolist() for r in rates],committed_rates=rates[chosen].tolist(),
            center_after=sigmoid(x).tolist(),rewards=rewards.tolist(),random_rewards=random_rewards,chosen=chosen,
            reward_kind=c['reward'],balanced_progress_since_last_evaluation=balanced_progress,
            random_loss=before,balanced=balanced,balanced_initial=balanced_initial if block==1 else None,
            train_losses=train_losses,advantage=advantage,baseline=baseline,
            masks=[summarize_masks(s) for s in mask_stats],
            mask_hashes=[digest([np.packbits(m).tobytes().hex() for m in branch]) for branch in masks])
        if c['probe_every'] and step%c['probe_every']==0:
            record['linear_probe']=linear_probe(model,prepared,folder/f'probe_{step}',seed,device)
        if device.type=='cuda': torch.cuda.synchronize()
        seconds+=time.perf_counter()-started; record['elapsed_training_pipeline_seconds']=seconds
        history.append(record)
        state=dict(contract=contract,training=capture(model,optimizer),policy_optimizer=policy_optimizer.state_dict(),
            x=x.tolist(),baseline=baseline,step=step,block=block,history=history,seconds=seconds,attempted=attempted,
            last_balanced=last_balanced)
        save_checkpoint(checkpoint,state)
        write_json(folder/'history.json',history)
        write_json(folder/'schedule.json',[dict(step=r['step'],H=r['H'],committed_rates=r['committed_rates']) for r in history])
        write_json(folder/'status.json',dict(status='running',step=step,attempted_updates=attempted))
        print(f'{method} seed={seed} step={step} random={before:.6g} center={sigmoid(x)}',flush=True)
    finished=step>=c['updates'] or bool(c['budget_seconds'] and seconds>=c['budget_seconds'])
    if not finished:
        write_json(folder/'status.json',dict(status='stopped_at_boundary',step=step)); return dict(complete=False,step=step)
    mark=time.perf_counter()
    balanced=evaluate('balanced'); validation=evaluate_bank(model,prepared.val,prepared.root/'random_validation.npz',device,c['microbatch'])
    linear=linear_probe(model,prepared,folder,seed,device)
    result=dict(contract=contract,complete=True,step=step,attempted_updates=attempted,
        stop_reason='updates' if step>=c['updates'] else 'compute_budget',
        random_control_loss=before,balanced_control=balanced,validation_reconstruction=validation,
        linear_probe=linear,training_pipeline_seconds=seconds,final_evaluation_seconds=time.perf_counter()-mark,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0,
        synthetic=c['synthetic'],test_opened=False)
    write_json(folder/'completed.json',result); write_json(folder/'status.json',dict(status='completed',step=step))
    return result
