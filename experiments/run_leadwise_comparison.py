"""Four lead-wise MAEs at matched SSL and frozen linear-probe budgets."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import backbone_comparison as matched
from experiments import stmem_base_baseline as base
from experiments.phase1.common import digest, file_sha
from experiments.phase1.config import DEFAULTS

MODELS = ('plain_small', 'plain_base', 'stmem_small', 'stmem_base')
STUDY = 'leadwise_four_model'
SEEDS = (42, 43, 44)
PRIMARY_POOLING = 'norm_then_mean'
METRICS = ('pooled_nrmse', 'macro_auroc', 'macro_auprc', 'macro_f1')
write_json = matched.write_json


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def source_identity():
    sources = ('experiments/leadwise_models.py', 'experiments/run_leadwise_comparison.py')
    return dict(base.source_identity(), **{name: file_sha(ROOT / name) for name in sources})


def create_study(directory, artifact_dir, st_mem_repo, *, synthetic=False):
    cfg = matched.configuration(base.CONFIG, dict(artifact_dir=artifact_dir, st_mem_repo=st_mem_repo,
        seeds=[42] if synthetic else list(SEEDS)), synthetic=synthetic)
    if not synthetic and cfg['artifact_dir'] is None:
        raise ValueError('Set --artifact-dir to the preprocessed PTB-XL artifact directory')
    plan = dict(study=STUDY, models=list(MODELS), config=cfg, source=source_identity(),
                upstream=matched.upstream_source(cfg['st_mem_repo']),
                primary_pooling=PRIMARY_POOLING, test_opened=False, full_paper_reproduction=False)
    plan['plan_id'] = digest(plan)
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / 'plan.json', plan)
    print(f'STUDY_DIR={directory.resolve()}\nmodels={",".join(MODELS)}; seeds={cfg["seeds"]}; '
          f'updates={cfg["updates"]}; batch={cfg["microbatch"]}x{cfg["accumulation"]}', flush=True)


def read_plan(directory, *, check_source=True):
    plan = read_json(directory / 'plan.json')
    if (plan['study'] != STUDY or plan['models'] != list(MODELS)
            or plan['plan_id'] != digest({k: v for k, v in plan.items() if k != 'plan_id'})):
        raise ValueError('Unexpected or edited comparison plan')
    if check_source:
        if plan['source'] != source_identity():
            raise ValueError('Producer source changed; use the original code or create a new study')
        if plan['upstream'] != matched.upstream_source(plan['config']['st_mem_repo']):
            raise ValueError('Official ST-MEM source changed since planning')
    return plan


def run_seed(directory, plan, name, seed, *, stop_after=None):
    if name not in MODELS or seed not in plan['config']['seeds']:
        raise ValueError('Model and seed must be in the frozen plan')
    folder = directory / name / f'seed{seed}'
    completed = folder / 'completed.json'
    if completed.exists():
        record = read_json(completed)
        if (record['backbone'], record['seed'], record['shared']['plan_id']) != (name, seed, plan['plan_id']):
            raise ValueError(f'Completed run belongs to a different study: {completed}')
        print(f'{name} seed={seed}: already complete', flush=True)
        return True
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import pandas as pd
    import torch
    from experiments.leadwise_models import build_model, count_parameters
    from experiments.phase1.data import get_data
    from experiments.phase1.pipeline import bank_at
    from experiments.phase1.training import (
        atomic_checkpoint, device_for, evaluate, load_checkpoint, runtime, seed_all, train_model)
    from experiments.predictability_estimators import lead_scale

    cfg = dict(plan['config'], seed=seed)
    device = device_for(cfg)  # Slurm CUDA guard before opening real ECGs.
    torch.set_num_threads(cfg['cpu_threads'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    folder.mkdir(parents=True, exist_ok=True)

    def status(value, **extra):
        write_json(folder / 'status.json', dict(status=value, backbone=name, seed=seed,
                   synthetic=cfg['synthetic'], test_opened=False, **extra))

    status('running')
    try:
        print(f'{name} seed={seed}: loading train and validation ECGs', flush=True)
        data = get_data(cfg)
        data['mean'], data['scale'] = lead_scale(data['arrays']['train'], data['selected']['pilot'])
        bank_dir = folder / 'banks'
        banks = {pool: bank_at(bank_dir, data, cfg, pool, pilot=True) for pool in ('train', 'validation')}
        shared = dict(plan_id=plan['plan_id'], seed=seed, data_id=data['data_id'],
                      selection_id=data['selection_id'], scale=data['scale'].tolist(),
                      banks={pool: file_sha(bank_dir / f'{pool}.csv') for pool in banks})
        # Different tasks can receive different GPU models. Record the hardware
        # separately from the data/mask pairing key and keep resume identity explicit.
        run_env = runtime(device, cfg)
        identity = dict(shared=shared, backbone=name, torch_version=run_env['torch'])
        seed_all(seed)
        model = build_model(name, cfg)
        total, trainable = count_parameters(model)
        write_json(folder / 'model.json', dict(backbone=name, model_size=model.model_size,
            architecture=model.architecture, parameters=total, trainable_parameters=trainable,
            decoder='leadwise_shared_weights', cross_lead_attention='encoder_only'))
        print(f'{name} seed={seed}: {total:,} parameters; P50/r50; batch 1x32' if not cfg['synthetic']
              else f'{name} seed={seed}: synthetic CPU check', flush=True)
        latest = folder / 'checkpoint.pt'
        if latest.exists():
            # The shared trainer saves the resumable checkpoint first. Recover
            # the milestone if the job stopped before writing its second copy.
            previous = load_checkpoint(latest)
            if previous['contract']['identity'] != identity:
                raise ValueError('Checkpoint model, data, mask or Torch identity mismatch')
            step = previous['update']
            mark = folder / 'milestones' / f'update_{step:06d}.pt'
            if step in cfg['milestones'] and not mark.exists():
                atomic_checkpoint(mark, {key: previous[key] for key in ('update','model','contract','sequence')})
            del previous
        _, resources = train_model(model, 'mae', data, {k: cfg[k] for k in DEFAULTS},
            folder, identity, banks, resume=latest.exists(), stop_after=stop_after,
            milestone_steps=cfg['milestones'])
        if not resources['complete']:
            status('partial', optimizer_updates=resources['update'])
            return False
        rows, probes = [], []
        for step in cfg['milestones']:
            checkpoint = folder / 'milestones' / f'update_{step:06d}.pt'
            output = folder / 'evaluations' / f'update_{step:06d}'
            output.mkdir(parents=True, exist_ok=True)
            saved = output / 'completed.json'
            eval_id = dict(training=identity, pretrain_update=step, checkpoint_sha256=file_sha(checkpoint))
            if saved.exists():
                value = read_json(saved)
                if value['evaluation_id'] != eval_id:
                    raise ValueError(f'Evaluation checkpoint mismatch: {saved}')
            else:
                state = load_checkpoint(checkpoint)
                if state['update'] != step or state['contract']['identity'] != identity or len(state['sequence']) != step:
                    raise ValueError(f'Invalid milestone: {checkpoint}')
                model.load_state_dict(state['model'])
                sequence_id = digest(state['sequence'])
                del state
                val, masks = banks['validation']
                print(f'{name} seed={seed} update={step}: validation reconstruction and frozen probes', flush=True)
                metrics = evaluate(model, data['arrays']['validation'], val, masks, data['scale'],
                    cfg['microbatch'], device, 'mae', output / 'reconstruction')
                row = dict(backbone=name, seed=seed, pretrain_update=step, parameters=total,
                           sequence_sha256=sequence_id, **metrics)
                probe = [dict(backbone=name, seed=seed, pretrain_update=step, **p)
                         for p in matched.frozen_probe(model, data, cfg, output / 'probe')]
                value = dict(evaluation_id=eval_id, reconstruction=row, probes=probe)
                write_json(saved, value)
            rows.append(value['reconstruction'])
            probes.extend(value['probes'])
        pd.DataFrame(rows).to_csv(folder / 'reconstruction.csv', index=False)
        pd.DataFrame(probes).to_csv(folder / 'probes.csv', index=False)
        write_json(completed, dict(backbone=name, seed=seed, shared=shared, resources=resources,
                                   reconstruction=rows, probes=probes, test_opened=False))
        status('completed', optimizer_updates=resources['update'])
        return True
    except Exception as error:
        status('failed', error=str(error))
        raise


def analyze(directory):
    import pandas as pd

    plan = read_plan(directory, check_source=False)
    cfg = plan['config']
    seen, streams, reconstruction, probes, missing = {}, {}, [], [], []
    for name in MODELS:
        for seed in cfg['seeds']:
            path = directory / name / f'seed{seed}' / 'completed.json'
            if not path.exists():
                missing.append(f'{name}/seed{seed}')
                continue
            record = read_json(path)
            if (record['backbone'], record['seed'], record['shared']['plan_id']) != (name, seed, plan['plan_id']):
                raise ValueError(f'Mixed study results: {path}')
            if seen.setdefault(seed, record['shared']) != record['shared']:
                raise ValueError(f'Data or validation mask mismatch at seed {seed}')
            expected_recon = {(step,) for step in cfg['milestones']}
            expected_probes = {(step, pooling, update) for step in cfg['milestones']
                               for pooling in cfg['poolings'] for update in cfg['probe_checkpoints']}
            for key, rows, keys, expected in (
                ('reconstruction', reconstruction, ('pretrain_update',), expected_recon),
                ('probes', probes, ('pretrain_update','pooling','probe_update'), expected_probes),
            ):
                observed = record[key]
                if len(observed) != len(expected) or {tuple(r[k] for k in keys) for r in observed} != expected:
                    raise ValueError(f'Missing or duplicate evaluation rows: {path}')
                for r in observed:
                    stream = (r['sequence_sha256'],) if key == 'reconstruction' else (
                        r['cohort_sha256'], r['head_stream_sha256'])
                    identity = (key, seed, *(r[k] for k in keys))
                    if streams.setdefault(identity, stream) != stream:
                        raise ValueError(f'Paired training/probe stream mismatch: {path}')
                rows.extend(observed)
    out = directory / 'analysis'
    out.mkdir(exist_ok=True)
    write_json(out / 'audit.json', dict(status='partial' if missing else 'completed', missing=missing,
                                       synthetic=cfg['synthetic'], test_opened=False))
    if missing:
        print(f'Incomplete: {len(missing)} run(s) missing: {", ".join(missing)}')
    if not reconstruction:
        return out
    recon, probe = pd.DataFrame(reconstruction), pd.DataFrame(probes)
    recon.to_csv(out / 'reconstruction.csv', index=False)
    probe.to_csv(out / 'probes.csv', index=False)
    keys = ['backbone','seed','pretrain_update']
    final = recon[recon.pretrain_update == cfg['updates']][keys + ['parameters','pooled_nrmse']].merge(
        probe[(probe.pretrain_update == cfg['updates']) & (probe.probe_update == cfg['probe_updates'])
              & (probe.pooling == PRIMARY_POOLING)][keys + list(METRICS[1:])],
        on=keys, validate='one_to_one')
    final.to_csv(out / 'final_by_seed.csv', index=False)
    summary = final.groupby(['backbone','parameters'], sort=False).agg(
        n_seeds=('seed','count'),
        **{f'{m}_{stat}': (m, stat) for m in METRICS for stat in ('mean','std')}).reset_index()
    summary.to_csv(out / 'final_summary.csv', index=False)
    comparisons = [('plain_small','stmem_small'), ('plain_base','stmem_base'),
                   ('plain_base','plain_small'), ('stmem_base','stmem_small')]
    pairs = []
    for a, b in comparisons:
        left = final[final.backbone == a].set_index('seed')
        right = final[final.backbone == b].set_index('seed')
        for seed in left.index.intersection(right.index):
            pairs.append(dict(a=a, b=b, seed=seed, **{m: left.at[seed,m] - right.at[seed,m] for m in METRICS}))
    pd.DataFrame(pairs, columns=['a','b','seed',*METRICS]).to_csv(out / 'paired_deltas_a_minus_b.csv', index=False)
    print(summary.to_string(index=False, float_format=lambda x: f'{x:.5f}'))
    print(f'ANALYSIS_DIR={out.resolve()}')
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('plan','validate','run','smoke','analyze'))
    parser.add_argument('--study-dir', type=Path, required=True)
    parser.add_argument('--artifact-dir')
    parser.add_argument('--st-mem-repo')
    parser.add_argument('--model', choices=MODELS)
    args = parser.parse_args(argv)
    directory = args.study_dir.resolve()
    if args.command in ('plan','smoke'):
        create_study(directory, args.artifact_dir, args.st_mem_repo, synthetic=args.command == 'smoke')
        if args.command == 'smoke':
            plan = read_plan(directory)
            for name in MODELS:
                run_seed(directory, plan, name, plan['config']['seeds'][0])
            analyze(directory)
    elif args.command == 'analyze':
        analyze(directory)
    else:
        plan = read_plan(directory)
        if args.command == 'validate':
            if plan['config']['synthetic']:
                parser.error('Submit only a real study; use smoke for the synthetic CPU check')
        else:
            if args.model is None:
                parser.error('run requires --model')
            for seed in plan['config']['seeds']:
                run_seed(directory, plan, args.model, seed)


if __name__ == '__main__':
    main()
