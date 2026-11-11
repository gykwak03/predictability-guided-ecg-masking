from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
from .common import digest,file_sha,read_json,write_json,seed_for
from .masks import wave_groups,matched_masks,global_mask

LEADS=['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']

def load_split(artifact,split,labels=False):
    if split not in ('train','validation','test'): raise ValueError('Unknown split')
    root=Path(artifact); manifest=read_json(root/'manifest.json'); c=manifest['config']
    if (c.get('target_sampling_rate')!=250 or c.get('bandpass_low')!=.67 or c.get('bandpass_high')!=50
            or c.get('normalization')!='per-record' or manifest.get('lead_order')!=LEADS):
        raise ValueError('Expected canonical 250Hz / .67-50Hz per-record artifact')
    cols=['artifact_index','ecg_id','patient_id','strat_fold','split']
    frame=pd.read_csv(root/f'metadata_{split}.csv',usecols=cols)
    x=np.load(root/f'signals_{split}.npy',mmap_mode='r',allow_pickle=False)
    allowed=range(1,9) if split=='train' else [9 if split=='validation' else 10]
    if (x.shape!=(len(frame),12,2500) or x.dtype!=np.float32 or frame[cols].isna().any().any()
            or not frame.ecg_id.is_unique or not np.array_equal(frame.artifact_index,np.arange(len(frame)))
            or not frame.strat_fold.isin(allowed).all() or not frame.split.eq(split).all()):
        raise ValueError('Split identity/shape/fold mismatch')
    y=None
    if labels:
        y=np.load(root/f'labels_{split}.npy',allow_pickle=False)
        if y.shape!=(len(frame),5) or not np.isin(y,[0,1]).all(): raise ValueError('Expected five binary labels')
    return x,frame,y

def synthetic_split(split):
    n=12 if split=='train' else 4
    offset=0 if split=='train' else (1000 if split=='validation' else 2000)
    rng=np.random.default_rng(offset+88); t=np.arange(2500)
    x=np.empty((n,12,2500),np.float32); annotations={}
    for i in range(n):
        intervals=[]; signal=np.zeros(2500)
        for beat,r in enumerate(range(240,2350,250)):
            for phase,center,width in [('P',r-45,10),('QRS',r,8),('T',r+65,20)]:
                signal+=(2 if phase=='QRS' else .5)*np.exp(-.5*((t-center)/width)**2)
                for lead in LEADS:
                    intervals.append(dict(ecg_id=i+offset,lead=lead,beat=beat,r_sample=r,phase=phase,
                        onset=center-width,offset=center+width,valid=True,reason='synthetic_known_bounds'))
        x[i]=signal[None,:]*(1+np.arange(12)[:,None]/20)+rng.normal(0,.02,(12,2500))
        x[i]=(x[i]-x[i].mean(-1,keepdims=True))/x[i].std(-1,keepdims=True)
        annotations[i+offset]=dict(ecg_id=i+offset,intervals=intervals,
            status=[dict(ecg_id=i+offset,lead=l,status='synthetic',reference_lead='II',r_peaks=9,error='') for l in LEADS],warnings=[])
    frame=pd.DataFrame(dict(artifact_index=range(n),ecg_id=np.arange(n)+offset,
        patient_id=np.arange(n)+offset,strat_fold=1 if split=='train' else 9,split=split))
    y=np.array([[(i+j)%2 for j in range(5)] for i in range(n)],np.float32)
    return x,frame,y,annotations

def arrays(plan,split,labels=False):
    if plan['config']['synthetic']: return synthetic_split(split)[:3]
    return load_split(plan['artifact_dir'],split,labels)

def partition(plan):
    c=plan['config']; _,frame,_=arrays(plan,'train')
    rng=np.random.default_rng(c['seed']); patients=rng.permutation(frame.patient_id.unique())
    if len(patients)<=c['control_ecgs']: raise ValueError('Too few patients for control holdout')
    held=patients[:c['control_ecgs']]
    controls=np.array([rng.choice(frame.index[frame.patient_id.eq(p)]) for p in held],int)
    fit=np.flatnonzero(~frame.patient_id.isin(held).to_numpy())
    if c['train_ecgs']:
        if c['train_ecgs']>len(fit): raise ValueError('train_ecgs exceeds available fit pool')
        fit=np.sort(rng.choice(fit,c['train_ecgs'],replace=False))
    _,validation,_=arrays(plan,'validation')
    if set(frame.patient_id)&set(validation.patient_id): raise ValueError('Train/validation patient overlap')
    if c['validation_ecgs']>len(validation): raise ValueError('Too few validation ECGs')
    val=np.sort(rng.choice(len(validation),c['validation_ecgs'],replace=False))
    return dict(fit=fit.tolist(),control=controls.tolist(),validation=val.tolist())

class Prepared:
    def __init__(self,study,plan):
        self.root=Path(study)/'prepared'; self.plan=plan; self.c=plan['config']
        completion=read_json(self.root/'completed.json')
        if completion['plan_id']!=plan['plan_id'] or completion['prepared_id']!=digest(completion['hashes']):
            raise ValueError('Wrong or edited prepared data')
        self.hashes=completion['hashes']
        for name,expected in self.hashes.items():
            if not name.startswith('annotations/') and file_sha(self.root/name)!=expected:
                raise ValueError(f'Prepared file changed: {name}')
        self.selection=read_json(self.root/'selection.json')
        self.x,self.frame,_=arrays(plan,'train')
        self.val,self.val_frame,_=arrays(plan,'validation')
    @lru_cache(maxsize=512)
    def groups(self,index):
        ecg=int(self.frame.iloc[index].ecg_id)
        path=self.root/'annotations'/f'{ecg}.json'
        if file_sha(path)!=self.hashes[f'annotations/{ecg}.json']: raise ValueError('Annotation changed')
        r=read_json(path)
        if r['ecg_id']!=ecg: raise ValueError('Annotation identity mismatch')
        return wave_groups(r,self.c['patch_samples'])

def _random_bank(indices,n,repeats,seed):
    rows=[]; masks=[]
    for index in indices:
        for repeat in range(repeats):
            m=global_mask(n,np.random.default_rng(seed_for(seed,index,repeat)))
            rows.extend([index,index]); masks.extend([m,~m])
    return dict(indices=np.array(rows,int),masks=np.array(masks,bool))

def prepare(study,plan,annotation_cache=None):
    """CPU-only preprocessing; resumable annotations, no model creation or labels."""
    from experiments.patch_audit import delineate_record
    c=plan['config']; root=Path(study)/'prepared'; root.mkdir(exist_ok=True)
    contract=root/'contract.json'
    if contract.exists() and read_json(contract)['plan_id']!=plan['plan_id']: raise ValueError('Prepared contract mismatch')
    write_json(contract,dict(plan_id=plan['plan_id']))
    if (root/'completed.json').exists():
        if read_json(root/'completed.json')['plan_id']!=plan['plan_id']: raise ValueError('Prepared completion mismatch')
        print('Prepared data already complete',flush=True); return
    selected=partition(plan); write_json(root/'selection.json',selected)
    x,frame,_=arrays(plan,'train'); annotation_dir=root/'annotations'; annotation_dir.mkdir(exist_ok=True)
    wanted=sorted(set(selected['fit']+selected['control']))
    syn=synthetic_split('train')[3] if c['synthetic'] else None
    if annotation_cache:
        cache=Path(annotation_cache); cm=read_json(cache/'manifest.json')
        if cm.get('manifest_sha256')!=plan['artifact_manifest'] or cm.get('source')!=str(Path(plan['artifact_dir']).resolve()):
            raise ValueError('Annotation cache is from another artifact')
        cache_selection=pd.read_csv(cache/'selection.csv',usecols=['ecg_id','patient_id','artifact_index']).set_index('ecg_id')
    missing=[]
    for index in wanted:
        ecg=int(frame.iloc[index].ecg_id); path=annotation_dir/f'{ecg}.json'
        if path.exists():
            if read_json(path)['ecg_id']!=ecg: raise ValueError('Annotation identity mismatch')
            continue
        if syn is not None: write_json(path,syn[ecg])
        elif annotation_cache and (Path(annotation_cache)/'delineations'/path.name).exists():
            if (ecg not in cache_selection.index or cache_selection.loc[ecg].patient_id!=frame.iloc[index].patient_id
                    or cache_selection.loc[ecg].artifact_index!=index):
                raise ValueError('Cached annotation patient/array identity mismatch')
            r=read_json(Path(annotation_cache)/'delineations'/path.name)
            if r['ecg_id']!=ecg: raise ValueError('Cached annotation mismatch')
            write_json(path,r)
        else: missing.append(index)
    if missing:
        with ProcessPoolExecutor(max_workers=c['prepare_workers']) as pool:
            for start in range(0,len(missing),max(4,c['prepare_workers']*4)):
                part=missing[start:start+c['prepare_workers']*4]
                jobs=[(int(frame.iloc[i].ecg_id),np.asarray(x[i])) for i in part]
                for result in pool.map(delineate_record,jobs):
                    write_json(annotation_dir/f"{result['ecg_id']}.json",result)
                print(f'Annotated {min(start+len(part),len(missing))}/{len(missing)} new ECGs',flush=True)
    n=12*2500//c['patch_samples']; q=n//12
    np.savez_compressed(root/'random_control.npz',**_random_bank(selected['control'],n,1,c['seed']))
    np.savez_compressed(root/'random_validation.npz',**_random_bank(selected['validation'],n,1,c['seed']+1))
    bank_indices=[]; bank_masks=[]; anchors=[]; strata=[]
    conditions=[[.8,.2,.2],[.2,.8,.2],[.2,.2,.8],[.2,.2,.2]]
    annotation_stats=[]
    for index in wanted:
        r=read_json(annotation_dir/f'{int(frame.iloc[index].ecg_id)}.json')
        groups=wave_groups(r,c['patch_samples'])
        annotation_stats.append(dict(ecg_id=r['ecg_id'],patient_id=float(frame.iloc[index].patient_id),
            pool='control' if index in selected['control'] else 'fit',
            P=sum(g['phase']=='P' for g in groups),QRS=sum(g['phase']=='QRS' for g in groups),
            T=sum(g['phase']=='T' for g in groups),rejected=sum(not v['valid'] for v in r['intervals'])))
    for index in selected['control'][:c['balanced_ecgs']]:
        result=read_json(annotation_dir/f'{int(frame.iloc[index].ecg_id)}.json')
        groups=wave_groups(result,c['patch_samples'])
        for phase in ['P','QRS','T']:
            candidates=[g for g in groups if g['phase']==phase]
            if not candidates: continue
            target=candidates[seed_for(c['seed'],index,phase)%len(candidates)]
            # Put the fixed target first; all policies use this SAME target.
            ordered=[target]+[g for g in groups if g is not target]
            from .masks import _solve
            rng=np.random.default_rng(seed_for(c['seed'],index,phase)); priority=rng.random(n); rounding=rng.random(3)
            for condition,rates in enumerate(conditions):
                solution=_solve(n,[target],ordered,q,rates,priority,rounding)
                if solution is None: raise ValueError('Infeasible balanced probe target')
                a=np.zeros(n,bool); a[target['ids']]=True
                bank_indices.append(index); bank_masks.append(solution[0]); anchors.append(a)
                strata.append(f'{phase}_{condition}')
    if not bank_masks: raise ValueError('Balanced bank is empty; inspect annotation coverage')
    np.savez_compressed(root/'balanced_control.npz',indices=np.array(bank_indices),masks=np.array(bank_masks),
                        anchors=np.array(anchors),strata=np.array(strata))
    pd.DataFrame(annotation_stats).to_csv(root/'annotation_coverage.csv',index=False)
    hashes={p.relative_to(root).as_posix():file_sha(p) for p in sorted(root.rglob('*'))
            if p.is_file() and p.name!='completed.json'}
    write_json(root/'completed.json',dict(plan_id=plan['plan_id'],prepared_id=digest(hashes),hashes=hashes,
        train_ecgs=len(selected['fit']),control_ecgs=len(selected['control']),balanced_cases=len(bank_masks),
        labels_opened=False,test_opened=False))
    print('Preparation complete',flush=True)
