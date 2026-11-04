"""Matched P50 pretraining and frozen-probe comparison; no paper-reproduction claim."""
from __future__ import annotations

import json
import os
from pathlib import Path
import traceback

from experiments.phase1.common import digest, file_sha, finish, source_identity, write_json as _write_json
from experiments.phase1.config import DEFAULTS, PATH_ENV, ROOT, absolute_path, resolve_config

BACKBONES = ('plain', 'stmem')
CLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
EXTRA = dict(seeds=[42, 43, 44], milestones=[0, 1000, 3000],
    st_mem_repo=str(ROOT.parent / 'ST-MEM'), probe_updates=1500,
    probe_checkpoints=[300, 1500], probe_batch=256, probe_lr=1e-3,
    probe_weight_decay=1e-4, probe_validation_ecgs=0,
    poolings=['norm_then_mean', 'mean_then_norm'])
UPSTREAM_FILES = ('models/encoder/vit.py', 'models/encoder/st_mem_vit.py', 'models/st_mem.py', 'LICENSE')


def write_json(path, value):
    """A timeout must not turn a partly written result into a completed marker."""
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    _write_json(temp, value)
    temp.replace(path)


def comparison_source():
    paths = ['experiments/backbone_comparison.py', 'experiments/backbone_models.py',
             'experiments/run_backbone_comparison.py',
             'archive/20261028_legacy/experiments/st_mem_adapter.py']
    return dict(source_identity(), **{p: file_sha(ROOT / p) for p in paths})


def upstream_source(repo):
    repo = Path(repo)
    for name in UPSTREAM_FILES:
        if not (repo / name).is_file():
            raise FileNotFoundError(f'Missing ST-MEM source: {repo / name}; set --st-mem-repo')
    return {name: file_sha(repo / name) for name in UPSTREAM_FILES}


def configuration(path=None, overrides=None, *, synthetic=False):
    saved = json.loads(Path(path).read_text(encoding='utf-8')) if path else {}
    unknown = set(saved) - set(DEFAULTS) - set(EXTRA)
    if unknown:
        raise ValueError(f'Unknown comparison config keys: {sorted(unknown)}')
    overrides = {k: v for k, v in (overrides or {}).items() if v is not None}
    phase = {k: v for k, v in saved.items() if k in DEFAULTS}
    if synthetic:
        # A separate reduced CPU run, never silently substituted for real ECG.
        phase.update(synthetic=True, device='cpu', model_size='smoke', train_ecgs=12,
                     fit_ecgs=8, score_ecgs=2, validation_ecgs=4, diagnostic_ecgs=2,
                     updates=2, microbatch=2, accumulation=2, eval_every=1,
                     checkpoint_every=1, warmup=1, cpu_threads=2)
    base = Path(path).resolve().parent if path else ROOT
    # Resolve config-relative locations before reusing the Phase 1 resolver.
    for key in ('raw_dir', 'artifact_dir', 'cache_dir', 'output_root'):
        if key in phase:
            if key not in overrides and os.environ.get(PATH_ENV[key]):
                phase.pop(key)  # preserve CLI > environment > config precedence
            else:
                phase[key] = absolute_path(phase[key], base)
    cfg = resolve_config(overrides=dict(phase, **{k: v for k, v in overrides.items() if k in DEFAULTS}))
    extra = dict(EXTRA, **{k: v for k, v in saved.items() if k in EXTRA})
    if synthetic:
        extra.update(seeds=[42], milestones=[0, 1, 2], probe_updates=2,
                     probe_checkpoints=[1, 2], probe_batch=4)
    extra.update({k: v for k, v in overrides.items() if k in EXTRA})
    repo = overrides.get('st_mem_repo') or os.environ.get('ECG_ST_MEM_REPO') or extra['st_mem_repo']
    origin = Path.cwd() if 'st_mem_repo' in overrides or os.environ.get('ECG_ST_MEM_REPO') else base
    extra['st_mem_repo'] = absolute_path(repo, origin)
    cfg.update(extra)
    if not cfg['synthetic'] and cfg['model_size'] != 'stmem-small':
        raise ValueError('Real comparison requires the enlarged Plain MAE and ST-MEM Small')
    if cfg['patch_samples'] != 50 or cfg['mask_ratio'] != .5:
        raise ValueError('This confirmation experiment fixes P50 and exactly 50% per lead')
    cfg['patches'] = [50]
    for key in ('seeds', 'milestones', 'probe_checkpoints'):
        values = cfg[key]
        if not values or any(type(v) is not int or v < 0 for v in values) or values != sorted(set(values)):
            raise ValueError(f'{key} must be sorted, unique, nonnegative integers')
    if cfg['milestones'][0] != 0 or cfg['milestones'][-1] != cfg['updates']:
        raise ValueError('Milestones must include the initial encoder and the final update')
    for key in ('probe_updates', 'probe_batch'):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f'{key} must be positive')
    if (cfg['probe_checkpoints'][0] < 1 or cfg['probe_checkpoints'][-1] != cfg['probe_updates']
            or type(cfg['probe_validation_ecgs']) is not int or cfg['probe_validation_ecgs'] < 0
            or not 0 < cfg['probe_lr'] or cfg['probe_weight_decay'] < 0):
        raise ValueError('Invalid probe budget, validation cap, or optimizer')
    if cfg['poolings'] != EXTRA['poolings']:
        raise ValueError('Keep both prespecified pooling orders; norm_then_mean is primary')
    return cfg


def semantic(cfg):
    return {k: v for k, v in cfg.items()
            if k not in ('raw_dir', 'artifact_dir', 'cache_dir', 'output_root', 'st_mem_repo')}


def create_study(cfg, directory):
    directory = Path(directory).resolve()
    plan = dict(schema=1, config=cfg, source=comparison_source(),
        upstream=upstream_source(cfg['st_mem_repo']), backbones=list(BACKBONES),
        design='Same ECG order, hidden masks, effective batch, updates, LR, loss and probe; architecture bundle differs',
        comparison_unit='optimizer updates (not epochs); 1000 is a prefix of the 3000-update LR trajectory',
        pretrained_weights=False, test_opened=False, primary_pooling='norm_then_mean',
        primary_metric='validation frozen-probe Macro AUROC at the final pretrain/probe update',
        equal_compute=False, synthetic=cfg['synthetic'])
    plan['plan_id'] = digest(plan)
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / 'plan.json', plan)
    print(f'STUDY_DIR={directory}', flush=True)
    print(f'{len(BACKBONES)} backbones x {len(cfg["seeds"])} seeds; '
          f'{cfg["updates"]} optimizer updates each; milestones={cfg["milestones"]}', flush=True)
    return directory


def read_study(directory, artifact_dir=None, st_mem_repo=None):
    directory = Path(directory).resolve()
    plan = json.loads((directory / 'plan.json').read_text(encoding='utf-8'))
    value = dict(plan)
    recorded = value.pop('plan_id')
    if digest(value) != recorded:
        raise ValueError('Plan was edited; create a new study instead')
    if plan['source'] != comparison_source():
        raise ValueError('Experiment source changed since planning; use the original code or a new study')
    cfg = dict(plan['config'])
    for key, replacement in [('artifact_dir', artifact_dir), ('st_mem_repo', st_mem_repo)]:
        if replacement is not None:
            cfg[key] = absolute_path(replacement, Path.cwd())
    if upstream_source(cfg['st_mem_repo']) != plan['upstream']:
        raise ValueError('ST-MEM source changed since planning')
    return plan, cfg


def probe_cohort(data, cfg):
    """Open diagnostic labels ONLY in the explicitly supervised probe stage."""
    import numpy as np
    import pandas as pd
    cohort, labels = {}, {}
    for split in ('train', 'validation'):
        frame = data['frames'][split]
        if cfg['synthetic']:
            # Deliberate balanced test fixture, not clinical performance evidence.
            y = ((np.arange(len(frame))[:, None] + np.arange(5)) % 2).astype(np.float32)
        else:
            if data['manifest'].get('class_order') != CLASSES:
                raise ValueError('Unexpected diagnostic class order')
            meta = pd.read_csv(Path(cfg['artifact_dir']) / f'metadata_{split}.csv',
                               usecols=['ecg_id', 'patient_id', *CLASSES])
            if not (np.array_equal(meta.ecg_id, frame.ecg_id)
                    and np.array_equal(meta.patient_id, frame.patient_id)):
                raise ValueError('Probe metadata/ECG alignment mismatch')
            y = meta[CLASSES].to_numpy(dtype=np.float32)
        if not np.isin(y, [0, 1]).all():
            raise ValueError('Nonbinary diagnostic labels')
        indices = data['selected']['pilot'].copy() if split == 'train' else np.arange(len(frame))
        cap = cfg['probe_validation_ecgs']
        if split == 'validation' and cap:
            if cap > len(indices):
                raise ValueError('Probe validation cap exceeds available ECGs')
            indices = np.sort(np.random.default_rng(cfg['subset_seed'] + 81).choice(indices, cap, replace=False))
        indices = indices[y[indices].any(1)]  # same label-valid rule in both arms
        if len(indices) < 2:
            raise ValueError('Probe needs at least two labeled ECGs per split')
        cohort[split], labels[split] = indices, y[indices]
    return cohort, labels


def classification_metrics(y, probabilities):
    import numpy as np
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
    per_class, auc, ap = {}, [], []
    for i, name in enumerate(CLASSES):
        supported = len(np.unique(y[:, i])) == 2
        per_class[name + '_auroc'] = float(roc_auc_score(y[:, i], probabilities[:, i])) if supported else None
        per_class[name + '_auprc'] = float(average_precision_score(y[:, i], probabilities[:, i])) if supported else None
        if supported:
            auc.append(per_class[name + '_auroc'])
            ap.append(per_class[name + '_auprc'])
    # Do not silently report a five-class macro average over fewer classes.
    return dict(macro_auroc=float(np.mean(auc)) if len(auc) == 5 else None,
        macro_auprc=float(np.mean(ap)) if len(ap) == 5 else None,
        macro_f1=float(f1_score(y, probabilities >= .5, average='macro', zero_division=0)),
        supported_classes=len(auc), threshold=.5, **per_class)


def frozen_probe(model, data, cfg, folder):
    import numpy as np
    import pandas as pd
    import torch
    from experiments.phase1.masks import batch_indices
    from experiments.phase1.training import preserve_rng, seed_all

    folder.mkdir(parents=True, exist_ok=True)
    cohort, labels = probe_cohort(data, cfg)
    train_labels = torch.from_numpy(labels['train'])
    cohort_frame = pd.concat([data['frames'][s].iloc[ix].assign(probe_split=s)
                             for s, ix in cohort.items()], ignore_index=True)
    cohort_frame.to_csv(folder / 'cohort.csv', index=False)
    label_id = digest({s: dict(indices=ix.tolist(), labels=labels[s].tolist()) for s, ix in cohort.items()})
    device = next(model.parameters()).device
    rows = []
    model.eval()
    with preserve_rng():
        for pooling in cfg['poolings']:
            features = {}
            with torch.no_grad():
                for split, indices in cohort.items():
                    chunks = []
                    for start in range(0, len(indices), cfg['microbatch']):
                        x = np.asarray(data['arrays'][split][indices[start:start+cfg['microbatch']]], dtype=np.float32)
                        chunks.append(model.features(torch.from_numpy(x).to(device), pooling).cpu())
                    features[split] = torch.cat(chunks)
                    if features[split].ndim != 2 or not torch.isfinite(features[split]).all():
                        raise ValueError('Nonfinite/invalid encoder features')
            ft, fv = features['train'], features['validation']
            mean, std = ft.mean(0), ft.std(0, correction=0).clamp_min(1e-6)
            ft, fv = (ft-mean)/std, (fv-mean)/std
            head_seed = cfg['seed'] + 100
            seed_all(head_seed)
            head = torch.nn.Linear(ft.shape[1], 5)  # CPU: no encoder graph or optimizer
            optimizer = torch.optim.AdamW(head.parameters(), lr=cfg['probe_lr'],
                betas=(.9, .999), weight_decay=cfg['probe_weight_decay'])
            history, sequence = [], []
            for step in range(1, cfg['probe_updates']+1):
                ix = batch_indices(len(ft), min(cfg['probe_batch'], len(ft)), step, head_seed)
                sequence.append(digest(ix.tolist()))
                loss = torch.nn.functional.binary_cross_entropy_with_logits(head(ft[ix]), train_labels[ix])
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite probe loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if step % 50 == 0 or step in cfg['probe_checkpoints']:
                    with torch.no_grad():
                        probabilities = head(fv).sigmoid().numpy()
                    metrics = classification_metrics(labels['validation'], probabilities)
                    history.append(dict(probe_update=step, train_bce=float(loss.detach()), **metrics))
                    if step in cfg['probe_checkpoints']:
                        rows.append(dict(pooling=pooling, probe_update=step, cohort_sha256=label_id,
                            head_stream_sha256=digest(sequence), input='full_unmasked',
                            train_ecgs=len(ft), validation_ecgs=len(fv), **metrics))
                        meta = data['frames']['validation'].iloc[cohort['validation']]
                        pd.DataFrame(probabilities, columns=CLASSES).assign(ecg_id=meta.ecg_id.to_numpy(),
                            patient_id=meta.patient_id.to_numpy()).to_csv(
                                folder / f'{pooling}_u{step}_probabilities.csv', index=False)
                        torch.save(dict(head=head.state_dict(), mean=mean, std=std,
                            classes=CLASSES, step=step), folder / f'{pooling}_u{step}_head.pt')
            pd.DataFrame(history).to_csv(folder / f'{pooling}_history.csv', index=False)
    return rows


def run_seed(directory, seed, *, artifact_dir=None, st_mem_repo=None, stop_after=None):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import numpy as np
    import pandas as pd
    import torch
    from experiments.backbone_models import build_backbone
    from experiments.phase1.data import get_data, NORMALIZATION
    from experiments.phase1.pipeline import bank_at
    from experiments.phase1.training import (device_for, evaluate, load_checkpoint, runtime, seed_all, train_model)
    from experiments.predictability_estimators import lead_scale

    directory = Path(directory).resolve()
    plan, cfg = read_study(directory, artifact_dir, st_mem_repo)
    if seed not in cfg['seeds']:
        raise ValueError('Seed is not in the frozen plan')
    if stop_after is not None and stop_after < 1:
        raise ValueError('stop-after must be positive')
    cfg['seed'] = seed
    torch.set_num_threads(cfg['cpu_threads'])
    device = device_for(cfg)
    # All threads/devices use FP32 math. Record and disable TF32 for both arms.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    data = get_data(cfg)
    data['mean'], data['scale'] = lead_scale(data['arrays']['train'], data['selected']['pilot'])
    root = directory / f'seed{seed}'
    root.mkdir(exist_ok=True)
    banks = {pool: bank_at(root / 'banks', data, cfg, pool, pilot=True)
             for pool in ('train', 'validation')}
    shared = dict(plan_id=plan['plan_id'], data_id=data['data_id'], selection_id=data['selection_id'],
        seed=seed, banks={pool: file_sha(root / 'banks' / f'{pool}.csv') for pool in banks},
        scale=data['scale'].tolist(), runtime=runtime(device, cfg))
    if (root / 'identity.json').exists():
        if json.loads((root / 'identity.json').read_text(encoding='utf-8')) != shared:
            raise ValueError('Data/selection/mask/scale/GPU/runtime mismatch; do not mix old and new runs')
    else:
        write_json(root / 'identity.json', shared)
        data['selection'].to_csv(root / 'ecg_selection.csv', index=False)
        write_json(root / 'data.json', dict(preprocessing=NORMALIZATION, hashes=data['hashes'],
            denominator_fit='same selected SSL training ECGs only', test_opened=False,
            synthetic=cfg['synthetic']))
    phase_cfg = {k: v for k, v in cfg.items() if k in DEFAULTS}
    errors, outcomes = [], []
    for name in BACKBONES:
        folder = root / name
        folder.mkdir(exist_ok=True)
        completed = folder / 'completed.json'
        if completed.exists():
            record = json.loads(completed.read_text(encoding='utf-8'))
            if record['shared_identity'] != shared:
                raise ValueError('Completed arm identity mismatch')
            outcomes.append(record)
            continue
        finish(folder, 'partial', backbone=name, seed=seed)
        try:
            seed_all(seed)
            model = build_backbone(name, phase_cfg, cfg['st_mem_repo'])
            _, resources = train_model(model, 'mae', data, phase_cfg, folder,
                dict(shared=shared, backbone=name), banks,
                resume=(folder / 'checkpoint.pt').exists(), stop_after=stop_after,
                milestone_steps=cfg['milestones'])
            rows, probes = [], []
            timings = pd.read_csv(folder / 'updates.csv')
            for step in cfg['milestones']:
                checkpoint = folder / 'milestones' / f'update_{step:06d}.pt'
                if not checkpoint.exists():
                    continue
                state = load_checkpoint(checkpoint)
                if (state['update'] != step or state['contract']['identity'] != dict(shared=shared, backbone=name)
                        or len(state['sequence']) != step):
                    raise ValueError('Milestone identity/update/stream mismatch')
                output = folder / 'evaluations' / f'update_{step:06d}'
                output.mkdir(parents=True, exist_ok=True)
                evaluation_id = dict(shared=shared, backbone=name, update=step,
                                     checkpoint_sha256=file_sha(checkpoint))
                saved = output / 'completed.json'
                if saved.exists():
                    value = json.loads(saved.read_text(encoding='utf-8'))
                    if value['evaluation_id'] != evaluation_id:
                        raise ValueError('Stored milestone evaluation is for a different checkpoint')
                else:
                    model.load_state_dict(state['model'])
                    model.to(device)
                    val, masks = banks['validation']
                    metrics = evaluate(model, data['arrays']['validation'], val, masks,
                        data['scale'], cfg['microbatch'], device, 'mae', output / 'reconstruction')
                    row = dict(seed=seed, backbone=name, pretrain_update=step,
                        pretrain_exposures=step*cfg['microbatch']*cfg['accumulation'],
                        equivalent_passes=step*cfg['microbatch']*cfg['accumulation']/len(data['selected']['pilot']),
                        sequence_sha256=digest(state['sequence']), shared_id=digest(shared),
                        gpu_name=shared['runtime']['gpu_name'], torch_version=shared['runtime']['torch'],
                        parameters=resources['parameters'], peak_vram_bytes=resources['peak_vram_bytes'],
                        training_seconds=float(timings.loc[timings['update'] <= step, 'training_seconds'].sum()),
                        **metrics)
                    probe = [dict(seed=seed, backbone=name, pretrain_update=step, shared_id=digest(shared), **p)
                             for p in frozen_probe(model, data, cfg, output / 'probe')]
                    value = dict(evaluation_id=evaluation_id, reconstruction=row, probes=probe)
                    write_json(saved, value)
                rows.append(value['reconstruction'])
                probes.extend(value['probes'])
                print(f'{name} seed={seed} pretrain_update={step}: '
                      f'NRMSE={rows[-1]["pooled_nrmse"]:.5f}; frozen probe saved', flush=True)
            if resources['complete'] and len(rows) != len(cfg['milestones']):
                raise RuntimeError('Missing prespecified milestone checkpoints; cannot mark this arm complete')
            pd.DataFrame(rows).to_csv(folder / 'reconstruction.csv', index=False)
            pd.DataFrame(probes).to_csv(folder / 'probes.csv', index=False)
            record = dict(shared_identity=shared, backbone=name, seed=seed, complete=resources['complete'],
                          reconstruction=rows, probes=probes, resources=resources)
            write_json(completed if resources['complete'] else folder / 'partial.json', record)
            if (folder / 'error.json').exists():
                previous = json.loads((folder / 'error.json').read_text(encoding='utf-8'))
                write_json(folder / 'error.json', dict(previous, resolved=True))
            finish(folder, 'completed' if resources['complete'] else 'partial', backbone=name, seed=seed,
                   optimizer_updates=resources['update'], test_opened=False, synthetic=cfg['synthetic'])
            outcomes.append(record)
        except Exception as error:
            detail = dict(backbone=name, seed=seed, error=str(error), traceback=traceback.format_exc())
            write_json(folder / 'error.json', detail)
            finish(folder, 'failed', **detail)
            errors.append(detail)
            print(f'FAILED {name} seed={seed}: {error}', flush=True)
        finally:
            if 'model' in locals():
                del model
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    complete = not errors and len(outcomes) == 2 and all(row['complete'] for row in outcomes)
    finish(root, 'completed' if complete else 'partial', seed=seed, errors=errors,
           test_opened=False, synthetic=cfg['synthetic'])
    return complete


def analyze(directory):
    """Only pair arms with identical observed streams; report partial studies honestly."""
    import numpy as np
    import pandas as pd

    directory = Path(directory).resolve()
    plan = json.loads((directory / 'plan.json').read_text(encoding='utf-8'))
    check = dict(plan)
    recorded_id = check.pop('plan_id')
    if digest(check) != recorded_id:
        raise ValueError('Plan was edited; cannot analyze an altered study')
    reconstruction, probes, missing = [], [], []
    for seed in plan['config']['seeds']:
        for name in BACKBONES:
            folder = directory / f'seed{seed}' / name
            if not (folder / 'completed.json').exists():
                missing.append(f'{name}/seed{seed}')
            for step in plan['config']['milestones']:
                path = folder / 'evaluations' / f'update_{step:06d}' / 'completed.json'
                if not path.exists():
                    if (folder / 'completed.json').exists():
                        missing.append(f'{name}/seed{seed}/update{step}')
                    continue
                value = json.loads(path.read_text(encoding='utf-8'))
                identity = value['evaluation_id']['shared']
                if identity['plan_id'] != plan['plan_id'] or identity['seed'] != seed:
                    raise ValueError('Evaluation belongs to a different study')
                row = value['reconstruction']
                if (row['shared_id'] != digest(identity) or row['seed'] != seed
                        or row['backbone'] != name or row['pretrain_update'] != step):
                    raise ValueError('Evaluation shared identity was modified')
                reconstruction.append(row)
                probes.extend(value['probes'])
    output = directory / 'analysis'
    output.mkdir(exist_ok=True)
    recon, probe = pd.DataFrame(reconstruction), pd.DataFrame(probes)
    recon.to_csv(output / 'reconstruction.csv', index=False)
    probe.to_csv(output / 'probes.csv', index=False)
    paired = []
    if not recon.empty:
        keys = ['seed', 'pretrain_update']
        pairs = recon[recon.backbone.eq('stmem')].merge(recon[recon.backbone.eq('plain')],
            on=keys, suffixes=('_stmem', '_plain'), validate='one_to_one')
        for field in ('sequence_sha256', 'shared_id'):
            if not pairs[field+'_stmem'].eq(pairs[field+'_plain']).all():
                raise ValueError(f'Unmatched pretraining arms: {field}')
        for row in pairs.to_dict('records'):
            paired.append(dict(seed=row['seed'], pretrain_update=row['pretrain_update'],
                metric='pooled_nrmse', delta_stmem_minus_plain=row['pooled_nrmse_stmem']-row['pooled_nrmse_plain']))
        recon.groupby(['backbone', 'pretrain_update'])[
            ['pooled_nrmse', 'masked_pearson', 'rms_ratio', 'training_seconds']].agg(
                ['mean', 'std', 'count']).to_csv(output / 'reconstruction_summary.csv')
    if not probe.empty:
        keys = ['seed', 'pretrain_update', 'pooling', 'probe_update']
        pairs = probe[probe.backbone.eq('stmem')].merge(probe[probe.backbone.eq('plain')],
            on=keys, suffixes=('_stmem', '_plain'), validate='one_to_one')
        for field in ('shared_id', 'cohort_sha256', 'head_stream_sha256'):
            if not pairs[field+'_stmem'].eq(pairs[field+'_plain']).all():
                raise ValueError(f'Unmatched probe arms: {field}')
        for row in pairs.to_dict('records'):
            a, b = row['macro_auroc_stmem'], row['macro_auroc_plain']
            paired.append(dict(**{k: row[k] for k in keys}, metric='macro_auroc',
                delta_stmem_minus_plain=a-b if pd.notna(a) and pd.notna(b) else None))
        probe.groupby(['backbone', 'pretrain_update', 'pooling', 'probe_update'])[
            ['macro_auroc', 'macro_auprc', 'macro_f1']].agg(['mean', 'std', 'count']).to_csv(output / 'probe_summary.csv')
    pd.DataFrame(paired).to_csv(output / 'paired_deltas.csv', index=False)
    write_json(output / 'audit.json', dict(plan_id=plan['plan_id'], missing_arms=missing,
        status='completed' if not missing else 'partial', synthetic=plan['synthetic'],
        expected_arms=2*len(plan['config']['seeds']), observed_checkpoints=len(recon),
        paired_streams_checked=True, test_opened=False, equal_compute=False,
        interpretation='Architecture bundle comparison; not a masking-policy effect or a paper reproduction'))
    if not recon.empty:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for name, frame in recon.groupby('backbone'):
            stats = frame.groupby('pretrain_update').pooled_nrmse.agg(['mean', 'std'])
            axes[0].errorbar(stats.index, stats['mean'], yerr=stats['std'].fillna(0), marker='o', label=name)
        if not probe.empty:
            selected = probe[(probe.pooling == 'norm_then_mean') &
                             (probe.probe_update == plan['config']['probe_updates'])]
            for name, frame in selected.groupby('backbone'):
                stats = frame.groupby('pretrain_update').macro_auroc.agg(['mean', 'std'])
                axes[1].errorbar(stats.index, stats['mean'], yerr=stats['std'].fillna(0), marker='o', label=name)
        for ax, label in zip(axes, ['Validation hidden NRMSE (lower better)', 'Frozen-probe Macro AUROC (higher better)']):
            ax.set(xlabel='Pretraining optimizer updates', ylabel=label)
            ax.grid(alpha=.2)
            ax.legend()
        fig.suptitle(('SYNTHETIC CPU CHECK — ' if plan['synthetic'] else '') +
                     ('PARTIAL — ' if missing else '') + 'P50 / 50% matched training; mean +/- seed SD')
        fig.savefig(output / 'comparison_curves.png', dpi=150)
        plt.close(fig)
    print(f'ANALYSIS={output}; missing arms={missing}', flush=True)
    return output
