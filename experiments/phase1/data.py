"""Train/validation artifact access only. Labels are never loaded."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.predictability_estimators import patient_pools, lead_scale
from .common import digest, file_sha

LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
PREPROCESSING = dict(source_sampling_rate=500, target_sampling_rate=250, bandpass_low=.67,
                     bandpass_high=50., filter_order=4, normalization='per-record', epsilon=1e-6)
ORDER = ['load-physical-mV', 'canonical-lead-reorder', 'polyphase-resample',
         'zero-phase-butterworth-bandpass', 'per-record']
NORMALIZATION = dict(formula='(x - mean_time(x)) / max(std_time(x, ddof=0), 1e-6)',
    scope='one complete 10-second ECG and one lead, before masking',
    includes_future_hidden_samples=True, units='dimensionless per-record z units',
    resampling='scipy.signal.resample_poly(up=1, down=2, axis=1, padtype=line)',
    filter='scipy.signal.butter(order=4, [0.67,50], fs=250, output=sos); sosfiltfilt(axis=1)',
    provenance_limit='Legacy schema 1 does not encode padtype/ddof; these follow the reviewed notebook implementation. Arrays alone cannot prove raw WFDB provenance.')


def load_artifact(directory):
    if directory is None:
        raise FileNotFoundError('Artifact path is unset. Supply --artifact-dir or ECG_ARTIFACT_DIR; expected a 250 Hz / 0.67-50 Hz artifact.')
    directory = Path(directory).resolve()
    manifest_path = directory / 'manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError(f'Missing {manifest_path}. Supply --artifact-dir / ECG_ARTIFACT_DIR or run prepare with --raw-dir and a new --artifact-dir.')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    differences = {k: (manifest.get('config', {}).get(k), v) for k, v in PREPROCESSING.items()
                   if manifest.get('config', {}).get(k) != v}
    if differences or manifest.get('lead_order') != LEAD_NAMES or manifest.get('preprocessing_order') != ORDER:
        raise ValueError(f'50 Hz artifact contract mismatch at {directory}: {differences}; check lead order and preprocessing_order. Never refilter 40 Hz arrays; create a separate artifact from raw WFDB.')
    arrays, frames, hashes = {}, {}, {'manifest': file_sha(manifest_path)}
    columns = ['artifact_index', 'ecg_id', 'patient_id', 'strat_fold', 'split']
    # Explicit allowlist. Never enumerate/open test waveforms, metadata, labels or label_valid.
    for split in ('train', 'validation'):
        meta_path, wave_path = directory / f'metadata_{split}.csv', directory / f'signals_{split}.npy'
        for path in (meta_path, wave_path):
            if not path.is_file():
                raise FileNotFoundError(f'Missing {path}; --artifact-dir must contain train and validation arrays and aligned metadata')
        frame = pd.read_csv(meta_path, usecols=columns, low_memory=False)
        x = np.load(wave_path, mmap_mode='r', allow_pickle=False)
        if frame.empty or x.shape != (len(frame), 12, 2500) or x.dtype != np.float32:
            raise ValueError(f'Invalid {split} shape/dtype or empty ECG pool')
        if (frame[columns].isna().any().any() or not frame.ecg_id.is_unique or
                not np.array_equal(frame.artifact_index, np.arange(len(frame))) or
                not frame.ecg_id.is_monotonic_increasing or not frame.split.eq(split).all()):
            raise ValueError(f'{split} identity, split or metadata alignment mismatch')
        if not pd.api.types.is_numeric_dtype(frame.ecg_id) or not np.equal(frame.ecg_id, np.floor(frame.ecg_id)).all():
            raise ValueError(f'{split} ECG IDs must be integers')
        allowed = frame.strat_fold.isin(range(1, 9)) if split == 'train' else frame.strat_fold.eq(9)
        if not allowed.all():
            raise ValueError(f'Unexpected {split} folds')
        entry = manifest.get('splits', {}).get(split, {})
        signal_entry = entry.get('signals', {})
        if (entry.get('count') != len(frame) or signal_entry.get('shape') != list(x.shape) or
                signal_entry.get('dtype') != 'float32' or signal_entry.get('file') != wave_path.name or
                entry.get('metadata') != meta_path.name or signal_entry.get('bytes') != wave_path.stat().st_size):
            raise ValueError(f'{split} manifest array/metadata contract mismatch')
        for start in range(0, len(x), 128):
            chunk = np.asarray(x[start:start+128])
            if not np.isfinite(chunk).all() or np.abs(chunk.mean(-1)).max() > 2e-4 or chunk.std(-1).max() > 1.0002:
                raise ValueError(f'{split} has nonfinite/non-normalized signals')
        arrays[split], frames[split] = x, frame
        hashes[split] = dict(metadata=file_sha(meta_path), signals=file_sha(wave_path))
    assert_disjoint(frames['train'], frames['validation'])
    return dict(arrays=arrays, frames=frames, data_id=digest(hashes), hashes=hashes,
                manifest=manifest, synthetic=False, test_opened=False)


def assert_disjoint(*frames):
    for i, first in enumerate(frames):
        for second in frames[i+1:]:
            if set(first.patient_id) & set(second.patient_id):
                raise ValueError('Patient overlap between fit/score/validation pools')
            if set(first.ecg_id) & set(second.ecg_id):
                raise ValueError('ECG overlap between pools')


def synthetic_data():
    """Small deterministic waveforms, explicitly not PTB-XL performance evidence."""
    arrays, frames = {}, {}
    rng = np.random.default_rng(729)
    t = np.arange(2500)/250
    for split, size, offset, fold in [('train', 20, 0, 1), ('validation', 6, 10000, 9)]:
        x = np.empty((size, 12, 2500), dtype=np.float32)
        for i in range(size):
            base = np.sin(2*np.pi*(1+.03*i)*t)
            x[i] = np.stack([base+.15*np.cos(2*np.pi*7*t+c)+.03*rng.normal(size=len(t)) for c in range(12)])
        mean, std = x.mean(-1, keepdims=True, dtype=np.float64), x.std(-1, keepdims=True, dtype=np.float64)
        arrays[split] = ((x-mean)/np.maximum(std, 1e-6)).astype(np.float32)
        frames[split] = pd.DataFrame(dict(artifact_index=np.arange(size), ecg_id=np.arange(size)+offset,
            patient_id=np.arange(size)//2+offset, strat_fold=fold, split=split))
    hashes = {s: dict(signals=__import__('hashlib').sha256(arrays[s].tobytes()).hexdigest(),
                      metadata=digest(frames[s].to_dict('records'))) for s in arrays}
    return dict(arrays=arrays, frames=frames, data_id=digest(hashes), hashes=hashes,
                synthetic=True, test_opened=False)


def select_data(data, cfg):
    train, validation = data['frames']['train'], data['frames']['validation']
    rng = np.random.default_rng(cfg['subset_seed'])
    def cap(indices, n):
        if n > len(indices):
            raise ValueError(f'Requested {n} ECGs exceeds available pool {len(indices)}')
        return np.sort(rng.choice(indices, n, replace=False)) if n else np.asarray(indices)
    pilot = cap(np.arange(len(train)), cfg['train_ecgs'])
    val = cap(np.arange(len(validation)), cfg['validation_ecgs'])
    fit, score = patient_pools(train, cfg['fit_ecgs'], cfg['score_ecgs'], cfg['subset_seed'])
    assert_disjoint(train.iloc[fit], train.iloc[score], validation.iloc[val])
    if cfg['microbatch']*cfg['accumulation'] > len(pilot):
        raise ValueError('MAE effective batch exceeds the pilot ECG pool')
    mean, scale = lead_scale(data['arrays']['train'], fit)
    selected = dict(pilot=pilot, fit=fit, score=score, validation=val)
    selection = pd.concat([data['frames']['validation' if role == 'validation' else 'train'].iloc[ix].assign(pool=role)
                           for role, ix in selected.items()], ignore_index=True)
    return dict(data, selected=selected, selection=selection, scale=scale, mean=mean,
                selection_id=digest(selection.to_dict('records')))


def get_data(cfg):
    return select_data(synthetic_data() if cfg['synthetic'] else load_artifact(cfg['artifact_dir']), cfg)
