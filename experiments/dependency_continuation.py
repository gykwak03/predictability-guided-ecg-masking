"""Controlled loss/optimizer ablations from immutable Random-pretrained checkpoints."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from experiments.dependency_gain import (ARMS, choose_data, draw_contexts, evaluate_panel, gains,
    hidden_losses, optimize, optimizer_for, paired_effects, panel, training_batch)
from experiments.phase1.common import digest, file_sha, source_identity, write_json
from experiments.phase1.config import ROOT, semantic_config
from experiments.phase1.masks import make_bank, mask_digest, per_lead_masks, seed_for
from experiments.phase1.model import build_mae
from experiments.phase1.training import (atomic_checkpoint, device_for, learning_rate,
    load_checkpoint, preserve_rng, restore_rng, rng_state, runtime, seed_all, sync)
from experiments.run_dependency_continuation import cases_for

VERSION = 'dependency-continuation-v1'


def source_files():
    result = source_identity()
    for name in ('dependency_gain.py','run_dependency_gain.py',
                 'dependency_continuation.py','run_dependency_continuation.py'):
        path = ROOT/'experiments'/name
        result[f'experiments/{name}'] = file_sha(path)
    return result


def continuation_lr(base_update, step, cfg, source_contract, options):
    if options['lr_policy'] == 'constant':
        return options['lr']
    # Continue the ORIGINAL cosine horizon; do not restart warmup or LR decay.
    # After the original horizon, the existing scheduler clamps at min_lr.
    return float(learning_rate(base_update+step,max(source_contract['options']['checkpoints']),cfg,'mae'))


def continuation_batch(data, cfg, source_update, step, arm, radius):
    absolute = source_update+step
    x, indices = training_batch(data,cfg,absolute,'pretrain')
    targets, contexts = draw_contexts(len(x),seed_for(cfg['mask_seed'],cfg['seed'],
        'continuation-context',source_update,step),radius)
    if arm == 'mae-random':
        # Exact next Random-mask stream from the ORIGINAL pretraining loop.
        masks = per_lead_masks(len(x),50,.5,seed_for(cfg['mask_seed'],cfg['seed'],'pretrain',absolute))
    else:
        masks = contexts[ARMS[arm]]
    flow = dict(batch_sha256=digest(indices.tolist()),target_sha256=digest(targets.tolist()),
                mask_sha256=mask_digest(masks))
    return x, masks, targets, flow


def prepare_optimizer(model, state, cfg, mode):
    optimizer = optimizer_for(model,cfg,cfg['lr'])
    if mode == 'keep':
        # load_state_dict may retain CPU tensor references; never mutate the common source state.
        optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
    elif mode != 'reset':
        raise ValueError('Unknown optimizer mode')
    return optimizer


def random_bank(data, cfg, options):
    metadata, masks = make_bank(data['frames']['validation'],data['selected']['validation'],50,.5,
        options['validation_repeats'],cfg['mask_seed'],'continuation-validation-random',pairs=True)
    return metadata, masks


def evaluate_random(model, data, bank, cfg, device):
    metadata, masks = bank
    values, previous = [], model.training
    scale = torch.as_tensor(data['scale'],dtype=torch.float32,device=device)
    try:
        model.eval()
        with preserve_rng(),torch.no_grad():
            for start in range(0,len(metadata),cfg['microbatch']):
                stop = start+cfg['microbatch']
                ix = metadata.iloc[start:stop].artifact_index.to_numpy()
                x = torch.from_numpy(np.asarray(data['arrays']['validation'][ix],dtype=np.float32)).to(device)
                mask = torch.from_numpy(masks[start:stop]).to(device)
                values.extend(hidden_losses(model(x,mask)[0],x,mask,scale).cpu().tolist())
    finally:
        model.train(previous)
    if not np.isfinite(values).all():
        raise RuntimeError('Nonfinite Random validation loss')
    return metadata.assign(hidden_mse=values)


def random_effects(before,after):
    keys = ['mask_id','ecg_id','patient_id','mask_sha256']
    frame = before[keys+['hidden_mse']].merge(after[keys+['hidden_mse']],on=keys,
        validate='one_to_one',suffixes=('_before','_after'))
    if len(frame) != len(before):
        raise ValueError('Common validation bank changed')
    frame['improvement'] = frame.hidden_mse_before-frame.hidden_mse_after
    return frame


def validate_state(state, checkpoint, data, cfg, saved):
    if state.get('update') != checkpoint or len(state.get('history',[])) != checkpoint:
        raise ValueError('Pretrain checkpoint update/history mismatch')
    if not {'model','optimizer','rng'} <= set(state):
        raise ValueError('Saved optimizer/RNG/model is required; no silent fresh-state fallback')
    x, indices = training_batch(data,cfg,checkpoint,'pretrain')
    expected = per_lead_masks(len(x),50,.5,seed_for(cfg['mask_seed'],cfg['seed'],'pretrain',checkpoint))
    last = state['history'][-1]
    if last['batch_sha256'] != digest(indices.tolist()) or last['mask_sha256'] != mask_digest(expected):
        raise ValueError('Source checkpoint is not the declared seed/data/Random-mask trajectory')
    lr = float(learning_rate(checkpoint,max(saved['options']['checkpoints']),cfg,'mae'))
    if any(not np.isclose(g['lr'],lr,rtol=1e-10,atol=1e-12) for g in state['optimizer']['param_groups']):
        raise ValueError('Saved optimizer LR differs from the declared pretraining schedule')


def train_branch(folder,state,base_sha,case,mode,data,cfg,saved,options,bank,before,device):
    folder.mkdir(parents=True,exist_ok=True)
    model = build_mae(cfg,50).to(device)
    model.load_state_dict(state['model'])
    optimizer = prepare_optimizer(model,state,cfg,mode)
    restore_rng(state['rng'])
    step, history, curve = 0, [], []
    path = folder/'latest.pt'
    identity = dict(base_sha256=base_sha,case=case,optimizer_mode=mode)
    if path.exists():
        previous = load_checkpoint(path)
        if previous['identity'] != identity:
            raise ValueError('Continuation branch identity changed')
        model.load_state_dict(previous['model'])
        optimizer.load_state_dict(previous['optimizer'])
        restore_rng(previous['rng'])
        step, history, curve = previous['update'],previous['history'],previous['curve']
    else:
        optimizer_steps = [float(s['step']) for s in optimizer.state.values() if 'step' in s]
        write_json(folder/'initial_state.json',dict(**identity,
            optimizer_state_entries=len(optimizer.state),
            optimizer_step_min=min(optimizer_steps) if optimizer_steps else 0,
            optimizer_step_max=max(optimizer_steps) if optimizer_steps else 0,
            training_denominator='ones: original per-record normalized waveform MSE',
            lr_policy=options['lr_policy']))
    if not curve:
        mse = float(before.hidden_mse.mean())
        curve.append(dict(update=0,validation_mse=mse,validation_nrmse=float(np.sqrt(mse)),
                          improvement=0.,mean_patient_improvement=0.))
    for current in range(step+1,options['updates']+1):
        sync(device)
        started = time.perf_counter()
        x,mask,targets,flow = continuation_batch(data,cfg,state['update'],current,case['arm'],saved['options']['radius'])
        lr = continuation_lr(state['update'],current,cfg,saved,options)
        for group in optimizer.param_groups:
            group['lr'] = lr
        # Keep the ORIGINAL training denominator for BOTH losses; only target set changes.
        loss = optimize(model,optimizer,x,mask,targets,cfg,np.ones(12),device,case['loss'])
        sync(device)
        history.append(dict(update=current,absolute_update=state['update']+current,lr=lr,
            training_loss=loss,training_seconds=time.perf_counter()-started,**flow))
        evaluate_now = current % options['eval_every']==0 or current==options['updates']
        if evaluate_now:
            after = evaluate_random(model,data,bank,cfg,device)
            effects = random_effects(before,after)
            effects.to_csv(folder/f'random_effects_{current:06d}.csv',index=False)
            mse = float(after.hidden_mse.mean())
            curve.append(dict(update=current,validation_mse=mse,validation_nrmse=float(np.sqrt(mse)),
                improvement=float(effects.improvement.mean()),
                mean_patient_improvement=float(effects.groupby('patient_id').improvement.mean().mean())))
            pd.DataFrame(curve).to_csv(folder/'curve.csv',index=False)
            print(f'seed={cfg["seed"]} base={state["update"]} case={case["case"]} '
                f'arm={case["arm"]} loss={case["loss"]} optimizer={mode} '
                f'update={current}/{options["updates"]} train={loss:.6f} '
                f'common_val={mse:.6f} improvement={curve[-1]["improvement"]:+.6f}',flush=True)
        if evaluate_now or current % options['save_every']==0:
            atomic_checkpoint(path,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng_state(),
                update=current,history=history,curve=curve,identity=identity))
            pd.DataFrame(history).to_csv(folder/'updates.csv',index=False)
    return model,history,curve


def summarize(root):
    rows,conditions = [], []
    for marker in sorted(root.glob('seed_*/base_*/*/*/*/completed.json')):
        record = json.loads(marker.read_text(encoding='utf-8'))
        folder = marker.parent
        for row in pd.read_csv(folder/'curve.csv').to_dict('records'):
            rows.append(dict(seed=record['seed'],checkpoint=record['checkpoint'],case=record['case'],
                arm=record['arm'],loss=record['loss'],optimizer_mode=record['optimizer_mode'],**row))
        conditions.append(pd.read_csv(folder/'context_summary.csv').assign(seed=record['seed'],
            checkpoint=record['checkpoint'],case=record['case'],arm=record['arm'],loss=record['loss'],
            optimizer_mode=record['optimizer_mode']))
    if not rows:
        return
    frame = pd.DataFrame(rows).sort_values(['seed','checkpoint','case','optimizer_mode','update'])
    frame.to_csv(root/'learning_curves.csv',index=False)
    final = frame.sort_values('update').groupby(['seed','checkpoint','case','optimizer_mode'],sort=False).tail(1)
    final = final.sort_values(['seed','checkpoint','case','optimizer_mode'])
    final.to_csv(root/'summary.csv',index=False)
    pd.concat(conditions,ignore_index=True).to_csv(root/'context_summary.csv',index=False)
    contrasts = []
    for keys,group in final.groupby(['seed','checkpoint','optimizer_mode']):
        values = group.set_index('case').improvement
        for label,left,right in [('within_vs_random','E','D'),('cross_vs_random','F','D'),
            ('within_vs_paired_random','E','G'),('cross_vs_paired_random','F','G'),
            ('within_full_vs_anchor','E','A'),('cross_full_vs_anchor','F','B'),
            ('paired_random_full_vs_anchor','G','C')]:
            if left in values and right in values:
                contrasts.append(dict(seed=keys[0],checkpoint=keys[1],optimizer_mode=keys[2],
                    contrast=label,value=float(values[left]-values[right])))
    for keys,group in final.groupby(['seed','checkpoint','case']):
        values = group.set_index('optimizer_mode').improvement
        if 'keep' in values and 'reset' in values:
            contrasts.append(dict(seed=keys[0],checkpoint=keys[1],case=keys[2],
                contrast='optimizer_keep_vs_reset',value=float(values['keep']-values['reset'])))
    pd.DataFrame(contrasts).to_csv(root/'contrasts.csv',index=False)
    synthetic = json.loads((root/'contract.json').read_text(encoding='utf-8'))['config']['synthetic']
    text = ['# Checkpoint continuation experiment','',
        'SYNTHETIC SOFTWARE CHECK ONLY — NOT ECG EVIDENCE.' if synthetic else 'Exploratory real ECG reconstruction experiment.', '',
        'Positive improvement = lower MSE on the SAME unconditioned Random validation bank.',
        'Primary metric: all hidden patches, fixed train-only lead scale. No downstream results.',
        'Patient-mean improvement is a secondary weighting check; ECGs are not independent training seeds.',
        'Shared checkpoints within a seed are dependent. No pooled-row significance test is reported.','',
        final.to_csv(index=False),'',
        'D uses the exact original unconditioned Random mask stream. E/F alter local context around one anchor but train on all hidden patches.',
        'G is the anchor-conditioned Random control for the optional matched loss ablation.',
        'Old A/B/C results differ in optimizer/LR/batch flow; only matched NEW cases isolate loss/optimizer effects.',
        'Check context_summary.csv for loss tradeoffs across the five diagnostic contexts.',
        'Check learning_curves.csv for early improvements followed by degradation.',
        'This run reuses Random-pretrained weights/optimizer/RNG; no pretraining is performed.']
    (root/'REPORT.md').write_text('\n'.join(text),encoding='utf-8')


def run_continuation(source,root,cfg,saved,options,resume=False):
    source,root = Path(source).resolve(),Path(root).resolve()
    if root==source or source in root.parents or root in source.parents:
        raise ValueError('Output must be separate from immutable source run')
    torch.set_num_threads(cfg['cpu_threads'])
    device = device_for(cfg)
    print(f'SOURCE_RUN={source}\nRUN_DIR={root}\nPretraining updates: 0; verifying saved checkpoints and artifact.',flush=True)
    files = source_files()
    if any(files.get(k)!=v for k,v in saved['source'].items()):
        raise ValueError('Original producer source changed; restore the recorded producer code before reuse')
    checkpoint_hashes = {}
    for seed in options['seeds']:
        for checkpoint in options['checkpoints']:
            path = source/f'seed_{seed}'/'pretrain'/f'update_{checkpoint}.pt'
            checkpoint_hashes[path.relative_to(source).as_posix()] = file_sha(path)
    data = choose_data(cfg,saved['options'])
    if data['data_id'] != saved['data_id'] or data['selection_id'] != saved['selection_id']:
        raise ValueError('Original artifact/patient selection changed')
    pd.testing.assert_frame_equal(pd.read_csv(source/'ecg_selection.csv'),data['selection'],check_dtype=False)
    contract = dict(version=VERSION,source_contract_sha256=file_sha(source/'contract.json'),
        checkpoints=checkpoint_hashes,config=semantic_config(cfg),options=options,source=files,
        data_id=data['data_id'],selection_id=data['selection_id'])
    if root.exists():
        if not resume:
            raise FileExistsError(f'Output exists: {root}; explicitly --resume this continuation')
        if json.loads((root/'contract.json').read_text(encoding='utf-8')) != contract:
            raise ValueError('Continuation source/checkpoint/data/config changed')
    else:
        root.mkdir(parents=True)
        write_json(root/'contract.json',contract)
    write_json(root/'status.json',dict(status='running',synthetic=cfg['synthetic'],test_opened=False,pretraining_updates=0))
    write_json(root/'provenance.json',dict(source_run=str(source),source_contract=saved,
        checkpoint_sha256=checkpoint_hashes,denominator=data['scale'],test_opened=False,label_files_opened=False))
    data['selection'].to_csv(root/'ecg_selection.csv',index=False)
    started = time.perf_counter()
    try:
        bank = random_bank(data,cfg,options)
        bank[0].to_csv(root/'validation_random_bank.csv',index=False)
        np.savez_compressed(root/'validation_random_masks.npz',masks=bank[1])
        contexts = panel(data,'validation',saved['options'],cfg['mask_seed'])
        cases = cases_for(options['losses'])
        for seed in options['seeds']:
            local = dict(cfg,seed=seed)
            seed_all(seed)
            for checkpoint in options['checkpoints']:
                base = root/f'seed_{seed}'/f'base_{checkpoint}'
                if (base/'completed.json').exists():
                    continue
                base.mkdir(parents=True,exist_ok=True)
                write_json(base/'runtime.json',runtime(device,local))
                path = source/f'seed_{seed}'/'pretrain'/f'update_{checkpoint}.pt'
                state = load_checkpoint(path)
                validate_state(state,checkpoint,data,local,saved)
                print(f'seed={seed} base={checkpoint}: evaluating common Random and context baselines.',flush=True)
                base_sha = checkpoint_hashes[path.relative_to(source).as_posix()]
                model = build_mae(local,50).to(device)
                model.load_state_dict(state['model'])
                before = evaluate_random(model,data,bank,local,device)
                # Reload the exact source and evaluate again without any update.
                model.load_state_dict(state['model'])
                replay = evaluate_random(model,data,bank,local,device)
                difference = float(np.max(np.abs(before.hidden_mse-replay.hidden_mse)))
                if difference > 1e-7:
                    raise RuntimeError('No-update replay changed common validation loss')
                write_json(base/'no_update_control.json',dict(max_abs_mse_difference=difference,passed=True,base_sha256=base_sha))
                before.to_csv(base/'validation_random_before.csv',index=False)
                context_before = evaluate_panel(model,data,contexts,local,device)
                context_before.to_csv(base/'context_before.csv',index=False)
                probe = panel(data,'gain_probe',saved['options'],cfg['mask_seed'])
                gains(evaluate_panel(model,data,probe,local,device)).to_csv(base/'gains.csv',index=False)
                del model
                sequences = []
                for case in cases:
                    for mode in options['optimizer_modes']:
                        folder = base/case['loss']/mode/case['arm']
                        marker = folder/'completed.json'
                        if marker.exists():
                            histories = pd.read_csv(folder/'updates.csv').to_dict('records')
                            sequences.append((case,mode,histories))
                            continue
                        model,history,curve = train_branch(folder,state,base_sha,case,mode,data,local,saved,
                            options,bank,before,device)
                        after = evaluate_panel(model,data,contexts,local,device)
                        effects = paired_effects(context_before,after)
                        effects.to_csv(folder/'context_effects.csv',index=False)
                        effects.groupby('condition')[['anchor_improvement','hidden_improvement']].mean().reset_index().to_csv(
                            folder/'context_summary.csv',index=False)
                        write_json(folder/'resources.json',dict(training_seconds=sum(h['training_seconds'] for h in history),
                            updates=len(history),process_peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0))
                        write_json(marker,dict(seed=seed,checkpoint=checkpoint,case=case['case'],arm=case['arm'],
                            loss=case['loss'],optimizer_mode=mode,base_sha256=base_sha,final_update=options['updates']))
                        sequences.append((case,mode,history))
                        del model
                        summarize(root)
                # All branches share ECG order, anchor references, LR and update counts.
                expected = [(h['batch_sha256'],h['target_sha256']) for h in sequences[0][2]]
                expected_lr = [h['lr'] for h in sequences[0][2]]
                for case,mode,history in sequences:
                    if ([(h['batch_sha256'],h['target_sha256']) for h in history] != expected or
                        not np.allclose([h['lr'] for h in history],expected_lr,rtol=1e-12,atol=1e-15)):
                        raise RuntimeError('Continuation branches differ in batch/anchor/LR flow')
                    if case['case']=='D' and mode=='keep' and options['lr_policy']=='continue':
                        reference_path = source/f'seed_{seed}'/'pretrain'/'updates.csv'
                        reference = pd.read_csv(reference_path).set_index('update')
                        comparable = [h for h in history if h['absolute_update'] in reference.index]
                        differences = []
                        for h in comparable:
                            original = reference.loc[h['absolute_update']]
                            if (h['batch_sha256']!=original.batch_sha256 or h['mask_sha256']!=original.mask_sha256 or
                                not np.isclose(h['lr'],original.lr,rtol=1e-12,atol=1e-15)):
                                raise RuntimeError('D does not reproduce the original Random batch/mask/LR stream')
                            differences.append(abs(h['training_loss']-original.loss))
                        write_json(base/'random_replay_audit.json',dict(compared_updates=len(comparable),
                            batch_mask_lr_match=True,mean_abs_training_loss_difference=float(np.mean(differences)) if differences else None,
                            note='Loss differences are descriptive across GPU/software versions; updates beyond the original horizon have no recorded reference.'))
                write_json(base/'completed.json',dict(matched_batch_anchor_lr=True,branches=len(sequences)))
        summarize(root)
        training_seconds,updates = 0.,0
        for path in root.glob('seed_*/base_*/*/*/*/updates.csv'):
            history = pd.read_csv(path)
            training_seconds += float(history.training_seconds.sum())
            updates += len(history)
        write_json(root/'compute_cost.json',dict(pretraining_updates=0,continuation_updates=updates,
            training_seconds=training_seconds,current_session_wall_seconds=time.perf_counter()-started))
        write_json(root/'status.json',dict(status='completed',synthetic=cfg['synthetic'],test_opened=False,
            pretraining_updates=0,downstream_evaluated=False))
    except Exception as error:
        write_json(root/'status.json',dict(status='failed',error=str(error),synthetic=cfg['synthetic'],test_opened=False))
        raise
    return root
