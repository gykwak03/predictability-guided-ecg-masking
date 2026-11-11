from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METHODS = ('random','conditional','fixed','curriculum','coordinate','reinforce','spsa','random_best2')
DEFAULTS = dict(patch_samples=10, updates=3000, lr_horizon=3000, H=50, seeds=[42,43,44],
    methods=list(METHODS), train_ecgs=0, control_ecgs=128, balanced_ecgs=32,
    validation_ecgs=128, microbatch=1, accumulation=32, lr=3e-4, min_lr=3e-5,
    lr_warmup=100, weight_decay=.05, grad_clip=1., targets=6, structured_fraction=.8,
    delta=.4054651081, policy_lr=.1, sigma=.4, max_rate_change=.05,
    fixed_rates=[.5,.5,.5], balanced_every=5, linear_updates=1500, linear_batch=256,
    linear_lr=.01, probe_every=0, cpu_threads=4, prepare_workers=4, model_size='small',
    device='cuda', budget_seconds=0, seed=42, synthetic=False, reward='random')

def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def write_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    temp.replace(path)

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def seed_for(*parts):
    return int(digest(parts)[:15],16)

def sources():
    files=[*sorted((ROOT/'experiments/wcb').glob('*.py')),ROOT/'experiments/run_wcb.py',ROOT/'experiments/patch_audit.py']
    return {p.relative_to(ROOT).as_posix():file_sha(p) for p in files}

def validate_config(c):
    if c['patch_samples'] not in (10,25) or c['model_size'] not in ('small','smoke'):
        raise ValueError('Expected P10/P25 and small/smoke model')
    for k in ('updates','lr_horizon','H','microbatch','accumulation','control_ecgs','balanced_ecgs',
              'validation_ecgs','targets','balanced_every','linear_updates','linear_batch',
              'cpu_threads','prepare_workers'):
        if type(c[k]) is not int or c[k]<1: raise ValueError(f'Invalid {k}')
    if c['targets']>6 or c['train_ecgs']<0 or c['probe_every']<0 or c['lr_warmup']<0:
        raise ValueError('Invalid target/data/evaluation settings')
    if c['lr']<=0 or not 0<c['min_lr']<=c['lr'] or c['policy_lr']<=0 or c['budget_seconds']<0:
        raise ValueError('Invalid optimizer/budget settings')
    if not 0<=c['structured_fraction']<=1 or not 0<c['max_rate_change']<1 or c['delta']<=0 or c['sigma']<=0:
        raise ValueError('Invalid policy settings')
    if len(c['fixed_rates'])!=3 or any(not 0<=r<=1 for r in c['fixed_rates']):
        raise ValueError('Expected three independent fixed ratios in [0,1]')
    if not c['seeds'] or len(set(c['seeds']))!=len(c['seeds']) or any(type(s) is not int or s<0 for s in c['seeds']):
        raise ValueError('Invalid seeds')
    if not c['methods'] or len(set(c['methods']))!=len(c['methods']) or set(c['methods'])-set(METHODS):
        raise ValueError('Unknown or duplicate method')
    if c['synthetic'] != (c['device']=='cpu' and c['model_size']=='smoke'):
        raise ValueError('CPU/smoke model requires explicit synthetic mode; real runs require CUDA')
    if c['reward'] not in ('random','balanced'): raise ValueError('Reward must be random or balanced, not a mixture')
    return c

def read_plan(study,check=True):
    p=read_json(Path(study)/'plan.json')
    if p['plan_id']!=digest({k:v for k,v in p.items() if k!='plan_id'}): raise ValueError('Edited plan')
    validate_config(p['config'])
    if check and p['sources']!=sources(): raise ValueError('Source changed; use original bundle or a new study')
    if not p['config']['synthetic'] and file_sha(Path(p['artifact_dir'])/'manifest.json')!=p['artifact_manifest']:
        raise ValueError('Artifact manifest changed')
    return p
