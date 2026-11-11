from __future__ import annotations

from pathlib import Path
import numpy as np

from experiments.wcb.common import digest, file_sha, read_json, write_json, seed_for

ROOT = Path(__file__).resolve().parents[2]
VERSION = 'ncf-annotation-free-v1'
RATE_ORDER = ('near', 'cross', 'far')
METHODS = ('random', 'conditional', 'fixed', 'random_policy',
           'coordinate', 'reinforce', 'spsa', 'conditional_best2')
ADAPTIVE = ('coordinate', 'reinforce', 'spsa')
TWO_BRANCH = (*ADAPTIVE, 'conditional_best2')
DEFAULTS = dict(
    patch_samples=10, target_ms=400, near_ms=400, target_leads=6,
    rate_low=.05, rate_high=.95, fixed_rates=[.5, .5, .5],
    probe_rates=[.2, .8], updates=3000, lr_horizon=3000, H=50,
    seeds=[42, 43, 44], methods=list(METHODS), train_ecgs=0,
    control_ecgs=128, balanced_ecgs=32, validation_ecgs=128,
    balanced_validation_ecgs=32, random_every=5,
    microbatch=1, accumulation=32, lr=3e-4, min_lr=3e-5,
    lr_warmup=100, weight_decay=.05, grad_clip=1.,
    delta=.4054651081, sigma=.4, policy_lr=.1, max_rate_change=.05,
    normalization_min_history=8, normalization_floor=1e-6,
    linear_updates=1500, linear_batch=256, linear_lr=.01, probe_every=0,
    cpu_threads=4, model_size='small', device='cuda', seed=42,
    synthetic=False, budget_seconds=0,
)


def geometry(c):
    p = c['patch_samples']
    sizes = []
    for key in ('target_ms', 'near_ms'):
        samples = c[key] * 250 / 1000
        if samples != int(samples) or int(samples) % p:
            raise ValueError(f'{key} must align to the fixed patch grid')
        sizes.append(int(samples) // p)
    m, flank = sizes
    q, leads = 2500 // p, c['target_leads']
    if m < 1 or flank < 1 or m + 2 * flank > q or not 1 <= leads <= 11:
        raise ValueError('Invalid target/context geometry')
    counts = dict(target=leads * m, near=leads * 2 * flank,
                  cross=(12 - leads) * m, far=leads * (q - m - 2 * flank),
                  background=(12 - leads) * (q - m))
    if any(v <= 0 for v in counts.values()):
        raise ValueError('All context groups must be nonempty')
    return dict(patches_per_lead=q, target_patches=m, flank_patches=flank,
                tokens=12 * q, visible=6 * q, counts=counts)


def validate_config(c):
    for key in ('patch_samples', 'target_ms', 'near_ms', 'target_leads',
                'updates', 'lr_horizon', 'H', 'control_ecgs', 'balanced_ecgs',
                'validation_ecgs', 'balanced_validation_ecgs', 'random_every',
                'microbatch', 'accumulation', 'linear_updates', 'linear_batch',
                'normalization_min_history', 'cpu_threads'):
        if type(c[key]) is not int or c[key] < 1:
            raise ValueError(f'Invalid positive integer: {key}')
    if c['patch_samples'] not in (10, 25) or c['model_size'] not in ('small', 'smoke'):
        raise ValueError('Expected P10/P25 and small/smoke model')
    expected = ('cpu', 'smoke') if c['synthetic'] else ('cuda', 'small')
    if (c['device'], c['model_size']) != expected:
        raise ValueError('CPU/smoke requires synthetic mode; real runs require CUDA')
    if c['device'] not in ('cuda', 'cpu'):
        raise ValueError('device must be cuda or cpu')
    for key in ('train_ecgs', 'probe_every', 'lr_warmup', 'seed'):
        if type(c[key]) is not int or c[key] < 0:
            raise ValueError(f'Invalid nonnegative integer: {key}')
    for key in ('lr', 'min_lr', 'policy_lr', 'delta', 'sigma', 'grad_clip',
                'linear_lr', 'normalization_floor'):
        if not np.isfinite(c[key]) or c[key] <= 0:
            raise ValueError(f'Invalid positive scalar: {key}')
    if not 0 < c['min_lr'] <= c['lr'] or not 0 < c['max_rate_change'] < 1:
        raise ValueError('Invalid learning/rate step bounds')
    if c['budget_seconds'] < 0 or not np.isfinite(c['budget_seconds']):
        raise ValueError('Invalid budget_seconds')
    lo, hi = c['rate_low'], c['rate_high']
    if not 0 < lo < .5 < hi < 1:
        raise ValueError('Rate bounds must contain 0.5 and lie inside (0,1)')
    for key, length in (('fixed_rates', 3), ('probe_rates', 2)):
        values = c[key]
        if len(values) != length or any(not np.isfinite(v) or not lo <= v <= hi for v in values):
            raise ValueError(f'{key} must lie within the independent rate bounds')
    if c['probe_rates'][0] >= c['probe_rates'][1]:
        raise ValueError('probe_rates must be increasing')
    if not c['seeds'] or len(set(c['seeds'])) != len(c['seeds']) or any(type(s) is not int or s < 0 for s in c['seeds']):
        raise ValueError('Expected distinct nonnegative seeds')
    if not c['methods'] or len(set(c['methods'])) != len(c['methods']) or set(c['methods']) - set(METHODS):
        raise ValueError('Unknown or duplicate method')
    if c['balanced_ecgs'] > c['control_ecgs'] or c['balanced_validation_ecgs'] > c['validation_ecgs']:
        raise ValueError('Balanced probe subset exceeds its patient pool')
    g = geometry(c)
    ns = np.array([g['counts'][key] for key in RATE_ORDER])
    # Prove feasibility for the full independent cube, including stochastic rounding.
    if (np.ceil(ns * hi).sum() > g['visible'] or
            np.floor(ns * lo).sum() + g['counts']['background'] < g['visible']):
        raise ValueError('Independent rate cube is infeasible for this visible budget/geometry')
    return c


def sources():
    files = [*sorted((ROOT / 'experiments/ncf').glob('*.py')), ROOT / 'experiments/run_ncf.py']
    # Shared, unmodified MAE, data readers, frozen head, RNG and optimizer helpers.
    files += [ROOT / 'experiments/wcb' / f'{name}.py'
              for name in ('common', 'data', 'masks', 'model', 'evaluation', 'training')]
    return {p.relative_to(ROOT).as_posix(): file_sha(p) for p in files}


def read_plan(study, check=True):
    plan = read_json(Path(study) / 'plan.json')
    if plan.get('version') != VERSION or plan.get('rate_order') != list(RATE_ORDER):
        raise ValueError('Expected an NCF annotation-free study; create a new study directory')
    if plan['plan_id'] != digest({k: v for k, v in plan.items() if k != 'plan_id'}):
        raise ValueError('Edited plan')
    validate_config(plan['config'])
    if check and plan['sources'] != sources():
        raise ValueError('Source changed; use original code or a new study')
    if not plan['config']['synthetic'] and file_sha(Path(plan['artifact_dir']) / 'manifest.json') != plan['artifact_manifest']:
        raise ValueError('Artifact manifest changed')
    return plan
