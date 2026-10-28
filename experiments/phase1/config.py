"""Portable configuration. No data discovery and no Torch import at planning time."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path, PureWindowsPath

ROOT = Path(__file__).resolve().parents[2]
PATH_ENV = {key: 'ECG_' + key.upper() for key in ('raw_dir', 'artifact_dir', 'output_root', 'cache_dir')}
DEFAULTS = dict(
    synthetic=False, seed=42, subset_seed=42, mask_seed=1729, cpu_threads=4,
    raw_dir=None, artifact_dir=None,
    output_root=str(ROOT / 'experiments/artifacts/phase1'),
    cache_dir=str(ROOT / 'experiments/artifacts/phase1/cache'),
    patch_samples=50, patches=[25, 50, 125], mask_ratio=0.5,
    train_ecgs=5000, fit_ecgs=1024, score_ecgs=256, validation_ecgs=128,
    diagnostic_ecgs=16, train_repeats=2, score_repeats=4, complement_pairs=True,
    updates=3000, microbatch=4, accumulation=8, eval_every=200, checkpoint_every=200,
    profile_updates=3, lr=3e-4, min_lr=3e-5, warmup=100,
    betas=[0.9, 0.95], weight_decay=0.05, grad_clip=1.0,
    model_size='stmem-small', precision='fp32', device='cuda',
    estimator_updates=2000, estimator_lr=3e-4, estimator_weight_decay=1e-4,
    cnn_width=64, transformer_width=128, transformer_depth=4, transformer_heads=4,
    ridge_alpha=1e-3, ridge_samples=128, ridge_lag_mode='fixed_seconds',
    ridge_lags_seconds=[-0.4, -0.2, -0.004, 0.0, 0.004, 0.2, 0.4],
)
SMOKE = dict(synthetic=True, model_size='smoke', device='cpu', train_ecgs=12,
             fit_ecgs=8, score_ecgs=2, validation_ecgs=4, diagnostic_ecgs=2,
             updates=2, estimator_updates=2, microbatch=2, accumulation=2,
             eval_every=1, checkpoint_every=1, warmup=1, train_repeats=1,
             score_repeats=1, ridge_samples=8, cnn_width=8,
             transformer_width=16, transformer_depth=1, transformer_heads=4,
             patches=[50], cpu_threads=2)


def absolute_path(value, base):
    if value is None:
        return None
    value = str(value)
    # Reject a config from the other OS instead of silently making C:\\... relative.
    if os.name != 'nt' and PureWindowsPath(value).drive:
        raise ValueError(f'Windows path is not usable on this OS: {value}')
    if os.name == 'nt' and value.startswith('/') and not value.startswith('//'):
        raise ValueError(f'Linux absolute path is not usable on Windows: {value}')
    path = Path(value).expanduser()
    return str((path if path.is_absolute() else Path(base) / path).resolve())


def resolve_config(config_path=None, overrides=None, *, smoke=False, environ=None, cwd=None):
    env = os.environ if environ is None else environ
    cwd = Path.cwd() if cwd is None else Path(cwd)
    cfg = copy.deepcopy(DEFAULTS)
    if smoke:
        cfg.update(SMOKE)
    saved, base = {}, ROOT
    if config_path:
        path = Path(absolute_path(config_path, cwd))
        saved = json.loads(path.read_text(encoding='utf-8'))
        if set(saved) - set(DEFAULTS):
            raise ValueError(f'Unknown config keys: {sorted(set(saved)-set(DEFAULTS))}')
        base = path.parent
        cfg.update(saved)
    overrides = {k: v for k, v in (overrides or {}).items() if v is not None}
    if set(overrides) - set(DEFAULTS):
        raise ValueError(f'Unknown overrides: {set(overrides)-set(DEFAULTS)}')
    cfg.update(overrides)
    for key, variable in PATH_ENV.items():
        if key in overrides:
            value, origin = overrides[key], cwd
        elif env.get(variable):
            value, origin = env[variable], cwd
        elif key in saved:
            value, origin = saved[key], base
        else:
            value, origin = cfg[key], ROOT
        cfg[key] = absolute_path(value, origin)
    if cfg['precision'] != 'fp32':
        raise ValueError('Phase 1 currently implements FP32 only')
    positive = ('cpu_threads', 'updates', 'estimator_updates', 'microbatch', 'accumulation',
                'eval_every', 'checkpoint_every', 'profile_updates', 'diagnostic_ecgs',
                'train_repeats', 'score_repeats', 'ridge_samples', 'warmup')
    if any(not isinstance(cfg[k], int) or cfg[k] < 1 for k in positive):
        raise ValueError(f'Positive integers required: {positive}')
    if any(not isinstance(cfg[k], int) or cfg[k] < 0 for k in
           ('train_ecgs', 'fit_ecgs', 'score_ecgs', 'validation_ecgs', 'seed', 'mask_seed', 'subset_seed')):
        raise ValueError('ECG caps and seeds must be nonnegative integers; ECG cap 0 means all')
    if not 0 < cfg['mask_ratio'] < 1 or not 0 < cfg['min_lr'] <= cfg['lr']:
        raise ValueError('Invalid mask ratio or learning rate')
    return cfg


def semantic_config(cfg):
    """Filesystem locations may change on transfer; the experiment must not."""
    return {k: v for k, v in cfg.items() if k not in PATH_ENV}
