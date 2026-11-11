"""From-scratch W/C/B study: plan, prepare, profile, run, replay, analyze, test."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.wcb.common import DEFAULTS,digest,file_sha,read_json,write_json,sources,read_plan,validate_config

def create_plan(args):
    c=json.loads(json.dumps(DEFAULTS))
    if args.config:
        provided=read_json(args.config)
        if set(provided)-set(c): raise ValueError(f'Unknown settings: {set(provided)-set(c)}')
        c.update(provided)
    if args.smoke:
        c.update(synthetic=True,device='cpu',model_size='smoke',patch_samples=25,updates=4,
            lr_horizon=4,H=2,seeds=[42],train_ecgs=6,control_ecgs=2,balanced_ecgs=1,
            validation_ecgs=2,microbatch=1,accumulation=1,targets=1,balanced_every=1,
            linear_updates=3,linear_batch=4,cpu_threads=2,prepare_workers=1,lr_warmup=1)
    if args.methods: c['methods']=args.methods
    if args.seeds: c['seeds']=args.seeds
    if args.updates is not None: c['updates']=args.updates
    validate_config(c)
    artifact=None if c['synthetic'] else str(Path(args.artifact_dir or os.environ.get('ECG_ARTIFACT_DIR','')).resolve())
    if not c['synthetic'] and not (Path(artifact)/'manifest.json').is_file():
        raise FileNotFoundError('Set --artifact-dir / ECG_ARTIFACT_DIR to reviewed 250Hz arrays')
    plan=dict(version='wcb-whole-wave-v1',config=c,artifact_dir=artifact,
        artifact_manifest=None if c['synthetic'] else file_sha(Path(artifact)/'manifest.json'),
        sources=sources(),from_scratch=True,random_warmup=False,test_opened=False)
    plan['plan_id']=digest(plan)
    args.study_dir.mkdir(parents=True,exist_ok=False)
    write_json(args.study_dir/'plan.json',plan)
    rows=[dict(index=i,method=m,seed=s) for i,(m,s) in enumerate((m,s) for m in c['methods'] for s in c['seeds'])]
    write_json(args.study_dir/'tasks.json',rows)
    print(f'STUDY_DIR={args.study_dir.resolve()}\nTASKS={len(rows)}\nPLAN_ID={plan["plan_id"]}')

def analyze(study,plan):
    import pandas as pd
    rows=[]
    for path in sorted((study/'runs').glob('*/seed*/completed.json')):
        r=read_json(path)
        if r['contract']['plan_id']!=plan['plan_id']: raise ValueError('Foreign completed run')
        rows.append(dict(method=path.parent.parent.name,seed=r['contract']['seed'],
            updates=r['step'],attempted_updates=r['attempted_updates'],
            validation_mse=r['validation_reconstruction']['loss'],
            random_control_mse=r['random_control_loss'],balanced_control_mse=r['balanced_control']['loss'],
            macro_auroc=r['linear_probe']['macro_auroc'],macro_auprc=r['linear_probe']['macro_auprc'],
            training_seconds=r['training_pipeline_seconds'],final_evaluation_seconds=r['final_evaluation_seconds']))
    if not rows: raise ValueError('No complete runs yet')
    out=study/'analysis'; out.mkdir(exist_ok=True); frame=pd.DataFrame(rows)
    frame.to_csv(out/'by_seed.csv',index=False)
    summary=frame.groupby('method').agg(seeds=('seed','count'),auroc_mean=('macro_auroc','mean'),
        auroc_std=('macro_auroc','std'),validation_mse_mean=('validation_mse','mean'),
        training_seconds_mean=('training_seconds','mean'))
    summary.to_csv(out/'summary.csv')
    contrasts=[]
    if 'random' in set(frame.method):
        baseline=frame[frame.method.eq('random')].set_index('seed')
        for r in frame.itertuples():
            if r.method!='random' and r.seed in baseline.index:
                b=baseline.loc[r.seed]
                contrasts.append(dict(method=r.method,seed=r.seed,delta_auroc=r.macro_auroc-b.macro_auroc,
                    delta_validation_mse=r.validation_mse-b.validation_mse,delta_seconds=r.training_seconds-b.training_seconds))
    pd.DataFrame(contrasts).to_csv(out/'paired_vs_random.csv',index=False)
    os.environ.setdefault('MPLCONFIGDIR',str((out/'matplotlib-cache').resolve()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,4))
    x=list(range(len(summary)))
    ax.bar(x,summary.auroc_mean,yerr=summary.auroc_std.fillna(0),capsize=3)
    ax.set(xticks=x,xticklabels=summary.index.tolist(),ylabel='Validation frozen macro AUROC')
    ax.tick_params(axis='x',labelrotation=30); fig.tight_layout(); fig.savefig(out/'auroc.png',dpi=140); plt.close(fig)
    for path in sorted((study/'runs').glob('*/seed*/history.json')):
        history=read_json(path)
        if not history: continue
        fig,axes=plt.subplots(1,2,figsize=(11,3.5))
        steps=[r['step'] for r in history]
        for j,label in enumerate(['W','C','B']):
            axes[0].plot(steps,[r['center_after'][j] for r in history],label=label)
            axes[0].plot(steps,[r['committed_rates'][j] for r in history],ls=':',alpha=.4)
        axes[0].set(xlabel='Committed updates',ylabel='Center (solid) / applied (dotted)',ylim=(0,1)); axes[0].legend()
        axes[1].plot(steps,[r['random_loss'] for r in history],label='Random')
        bal=[r for r in history if r['balanced'] is not None]
        axes[1].plot([r['step'] for r in bal],[r['balanced']['loss'] for r in bal],label='Balanced')
        axes[1].set(xlabel='Committed updates',ylabel='Separate probe MSE'); axes[1].legend()
        fig.suptitle(f'{path.parent.parent.name} {path.parent.name}'); fig.tight_layout()
        fig.savefig(out/f'{path.parent.parent.name}_{path.parent.name}_curves.png',dpi=140); plt.close(fig)
    expected=len(plan['config']['methods'])*len(plan['config']['seeds'])
    main_count=int(frame.method.isin(plan['config']['methods']).sum())
    report=['# W/C/B from-scratch study','',f'Completed {main_count}/{expected} planned runs; {len(frame)-main_count} additional runs. Synthetic={plan["config"]["synthetic"]}.',
        '', 'Seed-level results: by_seed.csv. Means/std: summary.csv. Paired differences: paired_vs_random.csv.',
        f'Policy reward={plan["config"]["reward"]} LP. Random and Balanced Control are reported separately, never mixed.',
        'Report both committed-update and compute costs; branch-selection and mask-family effects are not pure physiological causality.',
        'No test data are used by this analysis. Three seeds do not by themselves establish statistical significance.',
        '',summary.to_csv()]
    (out/'REPORT.md').write_text('\n'.join(report),encoding='utf-8')
    print(summary.to_string())

def profile(study,plan):
    import time
    import numpy as np
    import pandas as pd
    import torch
    from experiments.wcb.training import device_for,initialize
    from experiments.wcb.model import masked_mse
    from experiments.wcb.masks import global_mask
    c=plan['config']; device=device_for(c); model=initialize(c,c['seeds'][0]).to(device)
    n=model.n; x=torch.zeros((c['microbatch'],12,2500),device=device)
    mask=torch.tensor(np.array([global_mask(n,np.random.default_rng(j)) for j in range(len(x))]),device=device)
    def sync():
        if device.type=='cuda': torch.cuda.synchronize()
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
    # No optimizer exists here. This is timing/VRAM profiling, not a pretraining prefix.
    for _ in range(1):
        model.zero_grad(set_to_none=True); y,t=model(x,mask); masked_mse(y,t,mask).backward()
    sync(); start=time.perf_counter()
    for _ in range(2):
        model.zero_grad(set_to_none=True); y,t=model(x,mask); masked_mse(y,t,mask).backward()
    sync(); train_seconds=(time.perf_counter()-start)/2
    model.eval(); start=time.perf_counter()
    with torch.no_grad():
        for _ in range(2): model(x,mask)
    sync(); eval_seconds=(time.perf_counter()-start)/2
    start=time.perf_counter()
    with torch.no_grad(): model.features(x)
    sync(); feature_seconds=time.perf_counter()-start
    records=[]; blocks=int(np.ceil(c['updates']/c['H']))
    for method in c['methods']:
        branches=2 if method in ('coordinate','spsa','random_best2') else 1
        attempted=c['updates']*branches
        random_cases=(1+blocks*branches)*c['control_ecgs']*2
        balanced_cases=(2+int(np.ceil(blocks/c['balanced_every'])))*c['balanced_ecgs']*12
        estimated=attempted*c['accumulation']*train_seconds+(random_cases+balanced_cases)/c['microbatch']*eval_seconds
        records.append(dict(method=method,seeds=len(c['seeds']),committed_updates=c['updates'],
            attempted_updates=attempted,estimated_hours_per_seed=estimated/3600,
            estimated_hours_all_seeds=estimated*len(c['seeds'])/3600))
    pd.DataFrame(records).to_csv(study/'profile_budget.csv',index=False)
    write_json(study/'profile.json',dict(train_microbatch_seconds=train_seconds,eval_microbatch_seconds=eval_seconds,
        full_input_feature_seconds=feature_seconds,peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0,
        optimizer_updates=0,model_discarded=True,estimated_total_gpu_hours=sum(r['estimated_hours_all_seeds'] for r in records),
        estimate_excludes=['CPU mask MILP','optimizer/checkpoint overhead','linear probe features/head','final test'],
        estimate_is_not_runtime_guarantee=True))
    print(pd.DataFrame(records).to_string(index=False),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['plan','validate','prepare','run','profile','analyze','replay','test'])
    p.add_argument('--study-dir',type=Path,required=True)
    p.add_argument('--config',type=Path); p.add_argument('--artifact-dir'); p.add_argument('--smoke',action='store_true')
    p.add_argument('--methods',nargs='+'); p.add_argument('--seeds',type=int,nargs='+'); p.add_argument('--updates',type=int)
    p.add_argument('--annotation-cache',type=Path); p.add_argument('--task-index',type=int)
    p.add_argument('--method'); p.add_argument('--seed',type=int); p.add_argument('--stop-after',type=int)
    p.add_argument('--source-method'); p.add_argument('--source-seed',type=int); p.add_argument('--shuffle',action='store_true')
    args=p.parse_args()
    if args.command=='plan': return create_plan(args)
    plan=read_plan(args.study_dir)
    os.environ.setdefault('MPLCONFIGDIR',str((args.study_dir/'.matplotlib-cache').resolve()))
    if args.command=='validate': print(plan['plan_id']); return
    if args.command=='prepare':
        from experiments.wcb.data import prepare
        return prepare(args.study_dir,plan,args.annotation_cache)
    if args.command=='analyze': return analyze(args.study_dir,plan)
    if args.command=='profile': return profile(args.study_dir,plan)
    if args.task_index is not None:
        tasks=read_json(args.study_dir/'tasks.json')
        if not 0<=args.task_index<len(tasks): raise ValueError('Invalid array task index')
        args.method,args.seed=tasks[args.task_index]['method'],tasks[args.task_index]['seed']
    if args.seed not in plan['config']['seeds']: raise ValueError('Seed must be in the plan')
    from experiments.wcb.training import run
    if args.command=='replay':
        import numpy as np
        source=args.study_dir/'runs'/str(args.source_method)/f'seed{args.source_seed}'
        if read_json(source/'completed.json')['contract']['plan_id']!=plan['plan_id']: raise ValueError('Replay source mismatch')
        schedule=read_json(source/'schedule.json')
        if args.shuffle:
            rng=np.random.default_rng(args.seed+990); rng.shuffle(schedule)
        name=f'replay_{args.source_method}_{args.source_seed}'+('_shuffled' if args.shuffle else '')
        return run(args.study_dir,plan,'replay',args.seed,replay=schedule,
                   folder=args.study_dir/'runs'/name/f'seed{args.seed}',stop_after=args.stop_after)
    if args.method not in plan['config']['methods']: raise ValueError('Method must be in the plan')
    if args.command=='test':
        import torch
        from experiments.wcb.training import device_for,initialize
        from experiments.wcb.data import Prepared
        from experiments.wcb.evaluation import final_test
        folder=args.study_dir/'runs'/args.method/f'seed{args.seed}'
        if read_json(folder/'completed.json')['contract']['plan_id']!=plan['plan_id']: raise ValueError('Run is incomplete')
        device=device_for(plan['config']); model=initialize(plan['config'],args.seed).to(device)
        model.load_state_dict(torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)['training']['model'])
        print(final_test(model,Prepared(args.study_dir,plan),folder,device)); return
    return run(args.study_dir,plan,args.method,args.seed,stop_after=args.stop_after)

if __name__=='__main__':
    main()
