"""Separate MAE pilot, estimator fit and score stages; partial results survive errors."""
from __future__ import annotations

import json
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd
import torch

from .common import digest, file_sha, finish, new_run, source_identity, write_json
from .config import semantic_config
from .data import get_data, NORMALIZATION
from .masks import make_bank, replay_bank
from .model import build_mae, build_estimator, ridge_lags
from .training import (atomic_checkpoint, device_for, evaluate, load_checkpoint,
                       runtime, seed_all, train_model)


def identity_for(cfg, data):
    return dict(config=semantic_config(cfg), source=source_identity(), data_id=data['data_id'],
                selection_id=data['selection_id'], denominator=digest(data['scale'].tolist()))


def begin(cfg, kind, data, run_id=None, resume=None):
    identity = identity_for(cfg, data)
    if resume:
        root = Path(resume).resolve()
        if root.suffix == '.pt':
            root = root.parent.parent
        saved = json.loads((root / 'identity.json').read_text(encoding='utf-8'))
        if saved != identity:
            raise ValueError('Run config/data/source identity mismatch; do not change a run in place')
        status = json.loads((root / 'status.json').read_text(encoding='utf-8'))
        if status['status'] == 'completed':
            raise FileExistsError(f'Completed run is immutable: {root}')
        finish(root, 'partial', stage=kind, resuming=True)
    else:
        root = new_run(cfg['output_root'], kind, run_id)
        write_json(root / 'config.json', cfg)
        write_json(root / 'identity.json', identity)
        data['selection'].to_csv(root / 'ecg_selection.csv', index=False)
        write_json(root / 'data.json', dict(data_id=data['data_id'], hashes=data['hashes'],
            synthetic=data['synthetic'], test_opened=False, label_files_opened=False,
            preprocessing=NORMALIZATION))
        write_json(root / 'denominator.json', dict(lead_std=data['scale'], lead_mean=data['mean'],
            formula='sqrt(mean_hidden(((prediction-target)/s_lead)**2))',
            fit_pool='estimator-fit only', ddof=0, units='dimensionless per-record z units',
            ecg_ids=data['frames']['train'].iloc[data['selected']['fit']].ecg_id.tolist(),
            patient_ids=sorted(data['frames']['train'].iloc[data['selected']['fit']].patient_id.unique().tolist())))
    print(f'RUN_DIR={root}', flush=True)
    return root, identity


def bank_at(folder, data, cfg, pool, *, pilot=False):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if pilot:
        role = 'pilot' if pool == 'train' else 'validation'
        indices = data['selected'][role]
        if pool == 'train':
            indices = indices[:cfg['diagnostic_ecgs']]
        repeats, pairs = 1, False
    else:
        role = pool
        indices = data['selected'][role]
        repeats = cfg['train_repeats'] if pool == 'fit' else cfg['score_repeats'] if pool == 'score' else 1
        pairs = pool != 'fit' and cfg['complement_pairs'] and cfg['mask_ratio'] == .5
    split = 'validation' if pool == 'validation' else 'train'
    bank, masks = make_bank(data['frames'][split], indices, cfg['patch_samples'], cfg['mask_ratio'],
                            repeats, cfg['mask_seed'], pool, pairs=pairs)
    path = folder / f'{pool}.csv'
    if path.exists():
        loaded = pd.read_csv(path, keep_default_na=False)
        pd.testing.assert_frame_equal(loaded, bank, check_dtype=False)
        if not np.array_equal(replay_bank(loaded), masks):
            raise ValueError(f'Mask bank replay mismatch: {path}')
    else:
        bank.to_csv(path, index=False)
    return bank, masks


def run_pilot(cfg, *, run_id=None, resume=None, stop_after=None, profile=False):
    if not profile and cfg['mask_ratio'] != .5 and len(cfg['patches']) != 1:
        raise ValueError('Sensitivity requires one explicitly selected --patches value at 60% or 70%; primary patch pilot uses 50%')
    started = time.perf_counter()
    torch.set_num_threads(cfg['cpu_threads'])
    device = device_for(cfg)
    data = get_data(cfg)
    load_seconds = time.perf_counter()-started
    root, identity = begin(cfg, 'profile' if profile else 'pilot', data, run_id, resume)
    outcomes, errors = [], []
    patches = [cfg['patch_samples']] if profile else cfg['patches']
    for patch in patches:
        condition = f'p{patch}_r{cfg["mask_ratio"]:g}'
        folder = root / condition
        if (folder / 'completed.json').exists():
            outcomes.append(json.loads((folder / 'completed.json').read_text(encoding='utf-8')))
            continue
        condition_cfg = dict(cfg, patch_samples=patch)
        try:
            banks = {pool: bank_at(folder / 'banks', data, condition_cfg, pool, pilot=True)
                     for pool in ('train', 'validation')}
            bank_identity = {pool: file_sha(folder / 'banks' / f'{pool}.csv') for pool in banks}
            seed_all(cfg['seed'])
            model = build_mae(cfg, patch)
            model, resource = train_model(model, 'mae', data, condition_cfg, folder,
                dict(identity, banks=bank_identity), banks, resume=bool(resume), stop_after=stop_after,
                budget=cfg['profile_updates'] if profile else None)
            outcome = dict(condition=condition, patch_samples=patch, patch_seconds=patch/250,
                           mask_ratio=cfg['mask_ratio'], seed=cfg['seed'], **resource)
            if resource['complete']:
                for pool, (bank, masks) in banks.items():
                    summary = evaluate(model, data['arrays']['validation' if pool == 'validation' else 'train'],
                        bank, masks, data['scale'], cfg['microbatch'], device, 'mae', folder / pool)
                    outcome[pool] = summary
                write_json(folder / 'completed.json', outcome)
                if (folder / 'error.json').exists():
                    previous = json.loads((folder / 'error.json').read_text(encoding='utf-8'))
                    write_json(folder / 'error.json', dict(previous, resolved=True, excluded=False))
            outcomes.append(outcome)
        except Exception as error:
            folder.mkdir(parents=True, exist_ok=True)
            value = dict(condition=condition, error=str(error), traceback=traceback.format_exc(), excluded=True)
            write_json(folder / 'error.json', value)
            errors.append(value)
            print(f'FAILED {condition}: {error}', flush=True)
        finally:
            if 'model' in locals():
                del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    write_json(root / 'results.json', outcomes)
    pd.json_normalize(outcomes).to_csv(root / 'summary.csv', index=False)
    complete = not errors and len(outcomes) == len(patches) and all(o['complete'] for o in outcomes)
    finish(root, 'completed' if complete else 'partial', stage='profile' if profile else 'pilot',
        synthetic=cfg['synthetic'], errors=errors, test_opened=False,
        load_validation_seconds=load_seconds, session_total_seconds=time.perf_counter()-started,
        optimal_patch=None, selection='Manual validation review: stability, amplitude, zero-baseline improvement and cost; MSE alone is insufficient')
    return root, not errors


def estimator_metadata(cfg):
    lags = ridge_lags(cfg)
    return dict(ridge=dict(lags_samples=lags, lags_seconds=[x/250 for x in lags],
        mode=cfg['ridge_lag_mode'], alpha=cfg['ridge_alpha'], time_samples_per_ecg_mask=cfg['ridge_samples'],
        input_features=12*len(lags)*2+1, uses_future_offsets=True, context='selected time offsets across all 12 leads; offline reconstruction'),
        cnn=dict(width=cfg['cnn_width'], input_channels=24, channels='12 visible values + 12 observed indicators',
                 receptive_field_samples=97, receptive_field_seconds=97/250,
                 context='all leads; 7 + 6*(1+2+4+8) = 97 samples, up to +/-48 samples'),
        transformer=dict(width=cfg['transformer_width'], depth=cfg['transformer_depth'], heads=cfg['transformer_heads'],
            input='per-token P visible samples + 1 observed indicator; learned lead and fixed sin/cos time',
            context='bidirectional attention over all lead-time tokens'),
        fairness='Same observed ECG values/indicator and hidden targets; receptive fields and fitting compute differ',
        difficulty='Raw NRMSE only. Lower is easier; no Easy/Medium/Hard policy or percentile is fitted.')


def run_fit(cfg, *, run_id=None, resume=None, stop_after=None):
    started = time.perf_counter()
    torch.set_num_threads(cfg['cpu_threads'])
    device = device_for(cfg)
    data = get_data(cfg)
    root, identity = begin(cfg, 'estimators', data, run_id, resume)
    banks = {pool: bank_at(root / 'banks', data, cfg, pool) for pool in ('fit', 'validation', 'score')}
    identity = dict(identity, banks={pool: file_sha(root / 'banks' / f'{pool}.csv') for pool in banks})
    metadata = estimator_metadata(cfg)
    if not (root / 'estimator_contract.json').exists():
        write_json(root / 'estimator_contract.json', metadata)
    errors, results = [], []
    for name in ('ridge', 'cnn', 'transformer'):
        folder = root / name
        folder.mkdir(exist_ok=True)
        if (folder / 'fit_complete.json').exists():
            results.append(json.loads((folder / 'fit_complete.json').read_text(encoding='utf-8')))
            continue
        try:
            seed_all(cfg['seed'])
            model = build_estimator(name, cfg)
            if name == 'ridge':
                mark = time.perf_counter()
                model.fit(data['arrays']['train'], *banks['fit'], data['scale'], samples_per_ecg=cfg['ridge_samples'],
                          batch_size=cfg['microbatch'], seed=cfg['seed'])
                atomic_checkpoint(folder / 'checkpoint.pt', dict(kind='ridge', identity=identity,
                    weights=model.weights, scale=model.scale, target_counts=model.target_counts,
                    config=semantic_config(cfg), lags=list(model.lags), update=1))
                resource = dict(complete=True, training_seconds=time.perf_counter()-mark,
                    parameters=int(model.weights.size), peak_vram_bytes=0, fitting_target_counts=model.target_counts.tolist(),
                    fitting_targets=int(model.target_counts.sum()), fitting_bank_sha256=identity['banks']['fit'],
                    runtime=runtime(torch.device('cpu'), cfg))
                write_json(folder / 'resources.json', resource)
            else:
                _, resource = train_model(model, name, data, cfg, folder, identity,
                    {'validation': banks['validation']}, fit_bank=banks['fit'][0], fit_masks=banks['fit'][1],
                    resume=bool(resume), stop_after=stop_after)
            result = dict(estimator=name, **resource)
            if resource['complete']:
                write_json(folder / 'fit_complete.json', result)
                if (folder / 'error.json').exists():
                    previous = json.loads((folder / 'error.json').read_text(encoding='utf-8'))
                    write_json(folder / 'error.json', dict(previous, resolved=True, excluded=False))
            results.append(result)
        except Exception as error:
            value = dict(estimator=name, error=str(error), excluded=True, traceback=traceback.format_exc())
            write_json(folder / 'error.json', value)
            errors.append(value)
            print(f'FAILED {name}: {error}', flush=True)
        finally:
            if 'model' in locals():
                del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    sequences = [r['train_sequence_sha256'] for r in results if r['estimator'] != 'ridge' and r['complete']]
    if len(sequences) == 2 and len(set(sequences)) != 1:
        errors.append(dict(error='Neural estimator data/mask sequence mismatch'))
    complete = not errors and len(results) == 3 and all(r['complete'] for r in results)
    write_json(root / 'fit_results.json', results)
    finish(root, 'fit_completed' if complete else 'partial', stage='estimator-fit', errors=errors,
           synthetic=cfg['synthetic'], session_total_seconds=time.perf_counter()-started, test_opened=False)
    return root, not errors


def run_score(cfg, run_dir):
    started = time.perf_counter()
    torch.set_num_threads(cfg['cpu_threads'])
    device = device_for(cfg)
    data = get_data(cfg)
    root = Path(run_dir).resolve()
    identity = identity_for(cfg, data)
    if json.loads((root / 'identity.json').read_text(encoding='utf-8')) != identity:
        raise ValueError('Score config/data/source identity mismatch')
    if json.loads((root / 'status.json').read_text(encoding='utf-8'))['status'] == 'completed':
        raise FileExistsError(f'Completed scoring is immutable: {root}')
    banks = {pool: bank_at(root / 'banks', data, cfg, pool) for pool in ('fit', 'validation', 'score')}
    identity = dict(identity, banks={pool: file_sha(root / 'banks' / f'{pool}.csv') for pool in banks})
    summary, errors = [], []
    for name in ('ridge', 'cnn', 'transformer'):
        folder = root / name
        if (folder / 'score_complete.json').exists():
            summary.extend(json.loads((folder / 'score_complete.json').read_text(encoding='utf-8')))
            continue
        try:
            if not (folder / 'fit_complete.json').exists():
                raise ValueError('Estimator fit incomplete; excluded from scoring')
            checkpoint = load_checkpoint(folder / 'checkpoint.pt')
            if (checkpoint['identity'] if name == 'ridge' else checkpoint['contract']['identity']) != identity:
                raise ValueError('Estimator checkpoint/data/mask/source identity mismatch')
            model = build_estimator(name, cfg)
            if name == 'ridge':
                model.weights, model.scale = checkpoint['weights'], checkpoint['scale']
            else:
                model.load_state_dict(checkpoint['model'])
                model.to(device)
            rows = []
            mark = time.perf_counter()
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            for pool in ('validation', 'score'):
                metrics = evaluate(model, data['arrays']['validation' if pool == 'validation' else 'train'],
                    *banks[pool], data['scale'], cfg['microbatch'], device, name, folder / pool)
                rows.append(dict(estimator=name, pool=pool, **metrics))
            write_json(folder / 'score_resources.json', dict(seconds=time.perf_counter()-mark,
                peak_vram_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0))
            write_json(folder / 'score_complete.json', rows)
            if (folder / 'score_error.json').exists():
                previous = json.loads((folder / 'score_error.json').read_text(encoding='utf-8'))
                write_json(folder / 'score_error.json', dict(previous, resolved=True, excluded=False))
            summary.extend(rows)
        except Exception as error:
            value = dict(estimator=name, error=str(error), excluded=True, traceback=traceback.format_exc())
            write_json(folder / 'score_error.json', value)
            errors.append(value)
        finally:
            if 'model' in locals():
                del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    pd.DataFrame(summary).to_csv(root / 'estimator_summary.csv', index=False)
    finish(root, 'completed' if not errors and len(summary) == 6 else 'partial', stage='estimator-score',
           errors=errors, synthetic=cfg['synthetic'], test_opened=False,
           score_session_seconds=time.perf_counter()-started,
           quality_flags=[r for r in summary if r['worse_than_zero'] or r['amplitude_collapse_flag']])
    return root, not errors
