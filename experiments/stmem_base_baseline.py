"""ST-MEM Base at the existing P50/r50, 3000-update comparison budget.

The Small/Plain producer files stay unchanged so old frozen studies can resume.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import traceback

from experiments import backbone_comparison as matched
from experiments.phase1.common import digest, file_sha
from experiments.phase1.config import DEFAULTS, ROOT

NAME = 'stmem_base'
CONFIG = ROOT / 'configs/stmem_base_baseline.json'
write_json = matched.write_json


def configuration(path=None, overrides=None, *, synthetic=False):
    # Reuse the existing validation/path rules; then explicitly label the Base
    # model. Phase 1's trainer accepts an already constructed model.
    cfg = matched.configuration(path or CONFIG, overrides, synthetic=synthetic)
    cfg['model_size'] = 'smoke' if cfg['synthetic'] else 'stmem-base'
    if not cfg['synthetic']:
        if cfg['updates'] != 3000 or cfg['milestones'] != [0, 1000, 3000]:
            raise ValueError('Real Base baseline requires 3000 updates and milestones [0,1000,3000]')
        if cfg['microbatch'] * cfg['accumulation'] != 32:
            raise ValueError('Keep effective batch 32: microbatch * accumulation must equal 32')
    return cfg


def source_identity():
    paths = ['experiments/stmem_base_baseline.py', 'experiments/run_stmem_base_baseline.py']
    return dict(matched.comparison_source(), **{p: file_sha(ROOT / p) for p in paths})


def create_study(cfg, directory):
    directory = Path(directory).resolve()
    plan = dict(schema=1, backbone=NAME, config=cfg, source=source_identity(),
        upstream=matched.upstream_source(cfg['st_mem_repo']), synthetic=cfg['synthetic'],
        pretrained_weights=False, test_opened=False, equal_compute=False,
        comparison_unit='optimizer updates (not epochs); 1000 is a prefix of the 3000-update trajectory',
        primary_pooling='norm_then_mean',
        primary_metric='validation frozen-probe Macro AUROC at final pretrain/probe update',
        physical_batch_note='Base defaults to 1x32; Small/Plain use 4x8. Same effective ECG/mask stream; floating-point reductions differ.',
        full_paper_reproduction=False)
    plan['plan_id'] = digest(plan)
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / 'plan.json', plan)
    print(f'STUDY_DIR={directory}\nST-MEM Base; seeds={cfg["seeds"]}; updates={cfg["updates"]}; '
          f'batch={cfg["microbatch"]}x{cfg["accumulation"]}', flush=True)
    return directory


def read_study(directory, artifact_dir=None, st_mem_repo=None, *, check_source=True):
    plan = json.loads((Path(directory) / 'plan.json').read_text(encoding='utf-8'))
    original = dict(plan); recorded = original.pop('plan_id')
    if digest(original) != recorded or plan.get('backbone') != NAME:
        raise ValueError('Base plan was edited or belongs to another experiment')
    if check_source and plan['source'] != source_identity():
        raise ValueError('Producer source changed; use the original code or create a new study')
    cfg = dict(plan['config'])
    for key, value in [('artifact_dir', artifact_dir), ('st_mem_repo', st_mem_repo)]:
        if value is not None:
            cfg[key] = matched.absolute_path(value, Path.cwd())
    if check_source and matched.upstream_source(cfg['st_mem_repo']) != plan['upstream']:
        raise ValueError('Official ST-MEM source changed since planning')
    return plan, cfg


def build_model(cfg):
    import torch
    from experiments.backbone_models import adapter_module

    class BaseSTMEM(adapter_module().STMEMAdapter):
        def features(self, x, pooling='norm_then_mean'):
            # Same full-input feature/pooling path as the existing Small arm.
            encoder = self.core.encoder
            tokens = encoder.to_patch_embedding(x)
            b, c, q, d = tokens.shape
            tokens = tokens + encoder.pos_embedding[:, 1:q+1].unsqueeze(1)
            sep = encoder.sep_embedding[None, None, None, :].expand(b, c, 1, d)
            left = sep + encoder.pos_embedding[:, :1].unsqueeze(1)
            right = sep + encoder.pos_embedding[:, -1:].unsqueeze(1)
            tokens = torch.cat([left, tokens, right], dim=2)
            lead = torch.stack(list(encoder.lead_embeddings))[None, :, None, :]
            tokens = encoder.dropout((tokens + lead).flatten(1, 2))
            for i in range(encoder.depth):
                tokens = getattr(encoder, f'block{i}')(tokens)
            tokens = tokens.reshape(b, c, q+2, d)[:, :, 1:-1].flatten(1, 2)
            if pooling == 'norm_then_mean':
                return encoder.norm(tokens).mean(1)
            if pooling == 'mean_then_norm':
                return encoder.norm(tokens.mean(1))
            raise ValueError(f'Unknown pooling: {pooling}')

    return BaseSTMEM(cfg['st_mem_repo'], patch_samples=50,
                     model_size='smoke' if cfg['synthetic'] else 'base', seq_len=2500)


def status(folder, value, **details):
    write_json(Path(folder) / 'status.json', dict(status=value, **details))


def run_seed(directory, seed, *, artifact_dir=None, st_mem_repo=None, stop_after=None):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import pandas as pd
    import torch
    from experiments.phase1.data import get_data, NORMALIZATION
    from experiments.phase1.pipeline import bank_at
    from experiments.phase1.training import device_for, evaluate, load_checkpoint, runtime, seed_all, train_model
    from experiments.predictability_estimators import lead_scale

    directory = Path(directory).resolve()
    plan, cfg = read_study(directory, artifact_dir, st_mem_repo)
    if seed not in cfg['seeds']:
        raise ValueError('Seed is not in the frozen Base plan')
    if stop_after is not None and stop_after < 1:
        raise ValueError('stop-after must be positive')
    cfg['seed'] = seed
    torch.set_num_threads(cfg['cpu_threads'])
    device = device_for(cfg)  # Guard real compute before reading any ECG data.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    root = directory / f'seed{seed}'
    root.mkdir(exist_ok=True)
    folder = root / NAME; folder.mkdir(exist_ok=True)
    base_details = dict(backbone=NAME, seed=seed, synthetic=cfg['synthetic'], test_opened=False)
    status(root, 'running', **base_details)
    status(folder, 'running', **base_details)
    model = None
    try:
        print('Loading/validating training and validation ECGs.', flush=True)
        data = get_data(cfg)
        data['mean'], data['scale'] = lead_scale(data['arrays']['train'], data['selected']['pilot'])
        banks = {pool: bank_at(root / 'banks', data, cfg, pool, pilot=True) for pool in ('train', 'validation')}
        shared = dict(plan_id=plan['plan_id'], data_id=data['data_id'], selection_id=data['selection_id'],
            seed=seed, banks={pool: file_sha(root / 'banks' / f'{pool}.csv') for pool in banks},
            scale=data['scale'].tolist(), runtime=runtime(device, cfg))
        identity_file = root / 'identity.json'
        if identity_file.exists():
            if json.loads(identity_file.read_text(encoding='utf-8')) != shared:
                raise ValueError('Data, masks, normalization or GPU/runtime changed; do not mix runs')
        else:
            write_json(identity_file, shared)
            data['selection'].to_csv(root / 'ecg_selection.csv', index=False)
            write_json(root / 'data.json', dict(preprocessing=NORMALIZATION, hashes=data['hashes'],
                denominator_fit='same selected SSL training ECGs only', test_opened=False, synthetic=cfg['synthetic']))
        completed = folder / 'completed.json'
        if completed.exists():
            record = json.loads(completed.read_text(encoding='utf-8'))
            if record['shared_identity'] != shared or not record['complete']:
                raise ValueError('Completed Base identity mismatch')
            status(root, 'completed', optimizer_updates=record['resources']['update'], **base_details)
            status(folder, 'completed', optimizer_updates=record['resources']['update'], **base_details)
            print('Base already completed; no retraining.', flush=True)
            return True
        phase_cfg = {k: v for k, v in cfg.items() if k in DEFAULTS}
        seed_all(seed)
        model = build_model(cfg)
        write_json(folder / 'model.json', dict(backbone=NAME, model_size=model.model_size,
            architecture=model.architecture, input=[12, 2500], patch_samples=50,
            mask_ratio=.5, norm_pix_loss=False, loss='hidden waveform MSE over all 300 hidden patches',
            parameters=sum(p.numel() for p in model.parameters()), full_paper_reproduction=False))
        print(f'{NAME}: {sum(p.numel() for p in model.parameters()):,} parameters; '
              f'P50 / 50%; effective batch={cfg["microbatch"]*cfg["accumulation"]}', flush=True)
        training_identity = dict(shared=shared, backbone=NAME)
        _, resources = train_model(model, 'mae', data, phase_cfg, folder, training_identity, banks,
            resume=(folder / 'checkpoint.pt').exists(), stop_after=stop_after, milestone_steps=cfg['milestones'])
        timings = pd.read_csv(folder / 'updates.csv')
        rows, probes = [], []
        for step in cfg['milestones']:
            checkpoint = folder / 'milestones' / f'update_{step:06d}.pt'
            if not checkpoint.exists():
                continue
            state = load_checkpoint(checkpoint)
            if (state['update'] != step or state['contract']['identity'] != training_identity
                    or len(state['sequence']) != step):
                raise ValueError('Base milestone identity/update/stream mismatch')
            output = folder / 'evaluations' / f'update_{step:06d}'
            output.mkdir(parents=True, exist_ok=True)
            evaluation_id = dict(shared=shared, backbone=NAME, update=step, checkpoint_sha256=file_sha(checkpoint))
            saved = output / 'completed.json'
            if saved.exists():
                value = json.loads(saved.read_text(encoding='utf-8'))
                if value['evaluation_id'] != evaluation_id:
                    raise ValueError('Stored Base evaluation is for another checkpoint')
            else:
                model.load_state_dict(state['model']); model.to(device)
                val, masks = banks['validation']
                metrics = evaluate(model, data['arrays']['validation'], val, masks, data['scale'],
                    cfg['microbatch'], device, 'mae', output / 'reconstruction')
                row = dict(seed=seed, backbone=NAME, pretrain_update=step,
                    pretrain_exposures=step*cfg['microbatch']*cfg['accumulation'],
                    equivalent_passes=step*cfg['microbatch']*cfg['accumulation']/len(data['selected']['pilot']),
                    sequence_sha256=digest(state['sequence']), shared_id=digest(shared),
                    gpu_name=shared['runtime']['gpu_name'], torch_version=shared['runtime']['torch'],
                    parameters=resources['parameters'], peak_vram_bytes=resources['peak_vram_bytes'],
                    training_seconds=float(timings.loc[timings['update'] <= step, 'training_seconds'].sum()), **metrics)
                print(f'Evaluating Base seed={seed} checkpoint={step}: reconstruction done; frozen probes next.', flush=True)
                probe = [dict(seed=seed, backbone=NAME, pretrain_update=step, shared_id=digest(shared), **p)
                         for p in matched.frozen_probe(model, data, cfg, output / 'probe')]
                value = dict(evaluation_id=evaluation_id, reconstruction=row, probes=probe)
                write_json(saved, value)
            rows.append(value['reconstruction']); probes.extend(value['probes'])
            print(f'{NAME} seed={seed} update={step}: NRMSE={rows[-1]["pooled_nrmse"]:.5f}; probe saved', flush=True)
        if resources['complete'] and len(rows) != len(cfg['milestones']):
            raise RuntimeError('Missing prespecified Base milestones')
        pd.DataFrame(rows).to_csv(folder / 'reconstruction.csv', index=False)
        pd.DataFrame(probes).to_csv(folder / 'probes.csv', index=False)
        record = dict(shared_identity=shared, backbone=NAME, seed=seed, complete=resources['complete'],
                      reconstruction=rows, probes=probes, resources=resources)
        write_json(completed if resources['complete'] else folder / 'partial.json', record)
        if (folder / 'error.json').exists():
            previous = json.loads((folder / 'error.json').read_text(encoding='utf-8'))
            write_json(folder / 'error.json', dict(previous, resolved=True))
        for target in (root, folder):
            status(target, 'completed' if resources['complete'] else 'partial',
                   optimizer_updates=resources['update'], **base_details)
        return resources['complete']
    except Exception as error:
        detail = dict(error=str(error), traceback=traceback.format_exc())
        write_json(folder / 'error.json', detail)
        for target in (root, folder):
            status(target, 'failed', **detail, **base_details)
        print(f'FAILED {NAME} seed={seed}: {error}', flush=True)
        return False
    finally:
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()


def analyze(directory):
    """CPU-only aggregation; never opens ECGs, labels, checkpoints or a GPU."""
    import pandas as pd
    directory = Path(directory).resolve()
    plan, cfg = read_study(directory, check_source=False)
    reconstruction, probes, missing = [], [], []
    for seed in cfg['seeds']:
        folder = directory / f'seed{seed}' / NAME
        if not (folder / 'completed.json').exists():
            missing.append(seed)
        for step in cfg['milestones']:
            path = folder / 'evaluations' / f'update_{step:06d}' / 'completed.json'
            if not path.exists():
                if seed not in missing:
                    missing.append(seed)
                continue
            value = json.loads(path.read_text(encoding='utf-8'))
            identity = value['evaluation_id']
            row = value['reconstruction']
            if (identity['shared']['plan_id'] != plan['plan_id'] or identity['shared']['seed'] != seed
                    or identity['backbone'] != NAME or identity['update'] != step
                    or row['shared_id'] != digest(identity['shared']) or row['seed'] != seed
                    or row['backbone'] != NAME or row['pretrain_update'] != step):
                raise ValueError('Base evaluation identity was changed')
            for probe in value['probes']:
                if (probe['seed'] != seed or probe['backbone'] != NAME
                        or probe['pretrain_update'] != step or probe['shared_id'] != row['shared_id']):
                    raise ValueError('Base probe identity was changed')
            reconstruction.append(row); probes.extend(value['probes'])
    output = directory / 'analysis'; output.mkdir(exist_ok=True)
    recon, probe = pd.DataFrame(reconstruction), pd.DataFrame(probes)
    recon.to_csv(output / 'reconstruction.csv', index=False)
    probe.to_csv(output / 'probes.csv', index=False)
    if not recon.empty:
        recon.groupby('pretrain_update')[['pooled_nrmse', 'masked_pearson', 'rms_ratio', 'training_seconds']].agg(
            ['mean', 'std', 'count']).to_csv(output / 'reconstruction_summary.csv')
    if not probe.empty:
        probe.groupby(['pretrain_update', 'pooling', 'probe_update'])[['macro_auroc','macro_auprc','macro_f1']].agg(
            ['mean', 'std', 'count']).to_csv(output / 'probe_summary.csv')
    write_json(output / 'audit.json', dict(status='partial' if missing else 'completed', plan_id=plan['plan_id'],
        missing_seeds=missing, observed_checkpoints=len(recon), synthetic=cfg['synthetic'], test_opened=False,
        full_paper_reproduction=False, equal_compute=False,
        interpretation='Base baseline only; compare Small/Plain only after checking ECG/mask/probe stream hashes'))
    print(f'ANALYSIS={output}; missing seeds={missing}', flush=True)
    return output
