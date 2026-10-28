"""Reuse the unchanged factor_010 class, not a second copy of its architecture."""
from __future__ import annotations

from functools import lru_cache
import importlib.util

from .config import ROOT


@lru_cache(maxsize=1)
def legacy_module():
    path = ROOT / 'archive/20261028_legacy/experiments/mae_cause_model.py'
    spec = importlib.util.spec_from_file_location('_phase1_legacy_mae', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_mae(cfg, patch):
    spec = dict(size=cfg['model_size'], initialization='independent', fixed_time=True,
                patch_norm=False, sep=False, leadwise_decoder=False, model_seed=cfg['seed'])
    return legacy_module().CauseMAE(spec, patch=patch)


def build_estimator(name, cfg):
    from experiments.predictability_estimators import LaggedRidge, SmallCNN, SmallTransformer
    if name == 'ridge':
        return LaggedRidge(cfg['patch_samples'], alpha=cfg['ridge_alpha'], lags=ridge_lags(cfg))
    if name == 'cnn':
        return SmallCNN(width=cfg['cnn_width'])
    if name == 'transformer':
        return SmallTransformer(cfg['patch_samples'], width=cfg['transformer_width'],
                                depth=cfg['transformer_depth'], heads=cfg['transformer_heads'])
    raise ValueError(f'Unknown estimator: {name}')


def ridge_lags(cfg):
    p = cfg['patch_samples']
    if cfg['ridge_lag_mode'] == 'legacy_patch_scaled':
        return [-2*p, -p, -1, 0, 1, p, 2*p]
    if cfg['ridge_lag_mode'] != 'fixed_seconds':
        raise ValueError('ridge_lag_mode must be fixed_seconds or legacy_patch_scaled')
    samples = [seconds*250 for seconds in cfg['ridge_lags_seconds']]
    if any(abs(round(v)-v) > 1e-9 for v in samples):
        raise ValueError('Ridge seconds must map to exact 250 Hz sample offsets')
    return [int(round(v)) for v in samples]
