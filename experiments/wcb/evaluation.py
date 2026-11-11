from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .common import read_json,write_json,seed_for
from .data import arrays

@torch.no_grad()
def evaluate_bank(model,signals,path,device,batch):
    bank=np.load(path,allow_pickle=False); scores=[]; model.eval()
    for start in range(0,len(bank['indices']),batch):
        ids=bank['indices'][start:start+batch]
        x=torch.tensor(np.asarray(signals[ids]),device=device)
        mask=torch.tensor(bank['masks'][start:start+batch],device=device)
        prediction,target=model(x,mask)
        errors=(prediction-target).square().mean(-1)
        selected=torch.tensor(bank['anchors'][start:start+batch],device=device) if 'anchors' in bank else mask
        scores.extend(((errors*selected).sum(-1)/selected.sum(-1)).cpu().tolist())
    scores=np.array(scores)
    if not np.isfinite(scores).all(): raise RuntimeError('Nonfinite probe score')
    if 'strata' in bank:
        strata={str(k):float(scores[bank['strata']==k].mean()) for k in np.unique(bank['strata'])}
        return dict(loss=float(np.mean(list(strata.values()))),strata=strata,cases=len(scores))
    return dict(loss=float(scores.mean()),cases=len(scores))

def metrics(y,logits):
    from sklearn.metrics import roc_auc_score,average_precision_score
    auc=[]; ap=[]
    for i in range(5):
        valid=len(np.unique(y[:,i]))==2
        auc.append(float(roc_auc_score(y[:,i],logits[:,i])) if valid else None)
        ap.append(float(average_precision_score(y[:,i],logits[:,i])) if valid else None)
    values=[x for x in auc if x is not None]
    return dict(macro_auroc=float(np.mean(values)) if values else None,
        macro_auprc=float(np.mean([x for x in ap if x is not None])) if values else None,
        auroc_per_class=auc,auprc_per_class=ap,valid_auroc_classes=len(values))

@torch.no_grad()
def features(model,x,indices,device,batch):
    model.eval(); result=[]
    for start in range(0,len(indices),batch):
        result.append(model.features(torch.tensor(np.asarray(x[indices[start:start+batch]]),device=device)).cpu())
    return torch.cat(result)

def linear_probe(model,prepared,folder,seed,device):
    """Frozen encoder. Train labels only for the linear head; fold 9 selects head."""
    c=prepared.c; folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    # Save/restore all RNG streams used by the ongoing MAE if this is an intermediate probe.
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count())) if device.type=='cuda' else []):
        torch.manual_seed(seed+712)
        _,_,train_y=arrays(prepared.plan,'train',labels=True)
        _,_,val_y=arrays(prepared.plan,'validation',labels=True)
        fit=np.array(prepared.selection['fit'])
        z=features(model,prepared.x,fit,device,c['microbatch'])
        zv=features(model,prepared.val,np.arange(len(prepared.val)),device,c['microbatch'])
        mean,std=z.mean(0),z.std(0,unbiased=False).clamp_min(1e-6)
        z=(z-mean)/std; zv=(zv-mean)/std
        y=torch.tensor(np.asarray(train_y[fit]),dtype=torch.float32)
        yv=np.asarray(val_y)
        head=nn.Linear(z.shape[1],5)
        optimizer=torch.optim.AdamW(head.parameters(),lr=c['linear_lr'],weight_decay=0.)
        counts=y.sum(0); pos_weight=((len(y)-counts)/counts.clamp_min(1)).clamp(.1,100)
        objective=nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        best_score=-float('inf'); best=None; records=[]
        rng=np.random.default_rng(seed_for(seed,'linear'))
        for step in range(1,c['linear_updates']+1):
            ids=rng.integers(0,len(z),size=min(c['linear_batch'],len(z)))
            optimizer.zero_grad(set_to_none=True)
            loss=objective(head(z[ids]),y[ids]); loss.backward(); optimizer.step()
            if step%100==0 or step==c['linear_updates']:
                with torch.no_grad(): score=metrics(yv,head(zv).numpy())
                records.append(dict(step=step,loss=float(loss.detach()),**score))
                if score['macro_auroc'] is not None and score['macro_auroc']>best_score:
                    best_score=score['macro_auroc']; best={k:v.detach().clone() for k,v in head.state_dict().items()}
        if best is None: raise ValueError('No validation class has both labels; cannot select a head')
        head.load_state_dict(best)
        with torch.no_grad(): result=metrics(yv,head(zv).numpy())
        torch.save(dict(head=best,mean=mean,std=std,width=z.shape[1]),folder/'linear_head.pt')
        write_json(folder/'linear_history.json',records)
        write_json(folder/'linear_probe.json',dict(**result,seed=seed,pooling='norm_then_mean',
            encoder_frozen=True,fit_ecgs=len(fit),validation_ecgs=len(yv),test_opened=False))
        return result

def final_test(model,prepared,folder,device):
    folder=Path(folder)
    if (folder/'test.json').exists(): return read_json(folder/'test.json')
    state=torch.load(folder/'linear_head.pt',map_location='cpu',weights_only=False)
    x,frame,y=arrays(prepared.plan,'test',labels=True)
    if set(frame.patient_id)&(set(prepared.frame.patient_id)|set(prepared.val_frame.patient_id)):
        raise ValueError('Test patient overlap')
    z=features(model,x,np.arange(len(x)),device,prepared.c['microbatch'])
    head=nn.Linear(state['width'],5); head.load_state_dict(state['head'])
    with torch.no_grad(): score=metrics(y,head((z-state['mean'])/state['std']).numpy())
    result=dict(**score,test_opened=True,test_ecgs=len(x),encoder_frozen=True)
    write_json(folder/'test.json',result); return result
