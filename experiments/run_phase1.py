"""Phase 1 commands. Real GPU jobs are submitted separately by the user."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time

# Absolute bootstrap makes `python /path/to/experiments/run_phase1.py` cwd-independent.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase1.config import DEFAULTS, PATH_ENV, resolve_config
from experiments.phase1.common import new_run, write_json, finish, source_identity
from experiments.phase1.masks import geometry


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['plan', 'dry-run', 'preflight', 'prepare', 'smoke', 'profile',
                                     'pilot', 'estimator-fit', 'estimator-score', 'analyze'])
    p.add_argument('--config', type=Path)
    for key in PATH_ENV:
        p.add_argument('--'+key.replace('_', '-'))
    for key in ('seed', 'subset_seed', 'mask_seed', 'cpu_threads', 'patch_samples', 'train_ecgs', 'fit_ecgs',
                'score_ecgs', 'validation_ecgs', 'diagnostic_ecgs', 'updates', 'estimator_updates', 'microbatch',
                'accumulation', 'eval_every', 'checkpoint_every', 'profile_updates', 'warmup', 'ridge_samples',
                'train_repeats', 'score_repeats', 'cnn_width', 'transformer_width', 'transformer_depth', 'transformer_heads'):
        p.add_argument('--'+key.replace('_', '-'), type=int)
    for key in ('mask_ratio', 'lr', 'min_lr', 'ridge_alpha', 'estimator_lr'):
        p.add_argument('--'+key.replace('_', '-'), type=float)
    p.add_argument('--patches', nargs='+', type=int)
    p.add_argument('--device', choices=['cpu', 'cuda'])
    p.add_argument('--model-size', choices=['stmem-small', 'smoke'])
    p.add_argument('--ridge-lag-mode', choices=['fixed_seconds', 'legacy_patch_scaled'])
    p.add_argument('--synthetic', action='store_true', default=None, help='Explicit code-only synthetic mode; never real ECG evidence')
    p.add_argument('--complement-pairs', action=argparse.BooleanOptionalAction, default=None)
    p.add_argument('--run-id')
    p.add_argument('--resume', '--checkpoint', dest='resume', type=Path,
                   help='Existing partial run directory, or its condition/estimator checkpoint.pt')
    p.add_argument('--run-dir', type=Path, help='Existing estimator fit run to score, or any run to analyze')
    p.add_argument('--stop-after', type=int, help='Save and return partial results after this many additional updates')
    p.add_argument('--limit-per-split', type=int, help='Prepare-only reduced artifact check, separate output path required')
    return p


def plan(cfg):
    from experiments.phase1.model import legacy_module, ridge_lags
    conditions = []
    for patch in cfg['patches']:
        q, count = geometry(patch, cfg['mask_ratio'])
        conditions.append(dict(patch_samples=patch, patch_seconds=patch/250, tokens=12*q,
                               hidden_per_lead=count, hidden_tokens=12*count, ratio=cfg['mask_ratio']))
    root = new_run(cfg['output_root'], 'plan')
    value = dict(config=cfg, conditions=conditions, backbone=dict(name='Plain ECG-MAE factor_010',
        **legacy_module().SIZES[cfg['model_size']], independent_initialization=True,
        fixed_encoder_decoder_time=True, learned_lead=True, patch_norm=False, sep=False, leadwise_decoder=False),
        source=source_identity(), ridge_lags_samples=ridge_lags(cfg), data_verified=False,
        effective_batch=cfg['microbatch']*cfg['accumulation'], test_opened=False,
        comparison='Same subset, ECG order and update budget; FLOPs/time are not equal',
        selection='No optimal patch/ratio until validation stability, amplitude, baseline and cost are reviewed')
    write_json(root / 'plan.json', value)
    finish(root, 'completed', stage='plan', training_started=False)
    print(json.dumps(value, indent=2), flush=True)
    print(f'PLAN={root / "plan.json"}', flush=True)
    return root


def preflight(cfg):
    import torch
    from experiments.phase1.data import get_data
    from experiments.phase1.training import runtime
    from experiments.phase1.model import build_mae, build_estimator
    for key in ('output_root', 'cache_dir'):
        directory = Path(cfg[key])
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=directory) as probe:
            probe.write(b'phase1-write-check')
    data = get_data(cfg)
    # Import/model availability without a full enlarged model allocation on login nodes.
    build_mae(dict(cfg, model_size='smoke'), cfg['patch_samples'])
    build_estimator('ridge', cfg)
    print(json.dumps(dict(preflight='passed', data_id=data['data_id'], synthetic=cfg['synthetic'],
        ecg_counts={k: len(v) for k, v in data['selected'].items()}, test_opened=False,
        runtime=runtime(torch.device('cpu'), cfg)), indent=2), flush=True)


def prepare(cfg, limit):
    from experiments.phase1.data import load_artifact
    if not cfg['artifact_dir']:
        raise ValueError('prepare requires --artifact-dir (new output artifact) or ECG_ARTIFACT_DIR')
    if Path(cfg['artifact_dir']).exists():
        load_artifact(cfg['artifact_dir'])
        print(f'Reusing validated 50 Hz artifact: {cfg["artifact_dir"]}', flush=True)
        return
    if not cfg['raw_dir'] or not Path(cfg['raw_dir']).is_dir():
        raise FileNotFoundError(f'Raw WFDB directory missing: {cfg["raw_dir"]}; use --raw-dir / ECG_RAW_DIR')
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('New real-data preprocessing must run inside a Slurm job; do not preprocess on the login node')
    from preprocessing.run_preprocessing import run
    run(data_dir=cfg['raw_dir'], output_dir=cfg['artifact_dir'], limit_per_split=limit,
        splits=('train', 'validation'))
    load_artifact(cfg['artifact_dir'])


def smoke(cfg, run_id=None):
    from experiments.phase1.pipeline import run_pilot, run_fit, run_score
    from experiments.phase1.analysis import analyze
    if not cfg['synthetic'] or cfg['device'] != 'cpu':
        raise ValueError('smoke needs a synthetic CPU config; use configs/phase1_smoke.json')
    root = new_run(cfg['output_root'], 'smoke', run_id)
    local = dict(cfg, output_root=str(root))
    start = time.perf_counter()
    pilot, ok_pilot = run_pilot(local)
    fit, ok_fit = run_fit(local)
    scored, ok_score = run_score(local, fit)
    reports = [str(analyze(pilot)), str(analyze(scored))]
    ok = ok_pilot and ok_fit and ok_score
    finish(root, 'completed' if ok else 'partial', synthetic=True, test_opened=False,
        real_ecg_performance=False, pilot=str(pilot), estimators=str(fit), reports=reports,
        total_seconds=time.perf_counter()-start)
    print(f'SMOKE={root}', flush=True)
    return root, ok


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    overrides = {k: v for k, v in vars(args).items() if k in DEFAULTS and v is not None}
    try:
        if args.command == 'analyze':
            if not args.run_dir:
                p.error('analyze requires --run-dir')
            from experiments.phase1.analysis import analyze
            return analyze(args.run_dir)
        cfg = resolve_config(args.config, overrides, smoke=args.command == 'smoke')
        if args.command in ('smoke', 'pilot', 'plan', 'dry-run'):
            for patch in cfg['patches']:
                geometry(patch, cfg['mask_ratio'])
            if len(set(cfg['patches'])) != len(cfg['patches']) or not cfg['patches']:
                raise ValueError('Patch grid must be nonempty and unique')
        geometry(cfg['patch_samples'], cfg['mask_ratio'])
        if args.stop_after is not None and args.stop_after < 1:
            p.error('--stop-after must be positive')
        print(json.dumps(dict(command=args.command, resolved_paths={key: cfg[key] for key in PATH_ENV},
                              synthetic=cfg['synthetic']), indent=2), flush=True)
        if args.command in ('plan', 'dry-run'):
            return plan(cfg)
        if args.command == 'preflight':
            return preflight(cfg)
        if args.command == 'prepare':
            return prepare(cfg, args.limit_per_split)
        if args.command == 'smoke':
            root, ok = smoke(cfg, args.run_id)
        else:
            from experiments.phase1.pipeline import run_pilot, run_fit, run_score
            if args.command in ('pilot', 'profile'):
                root, ok = run_pilot(cfg, run_id=args.run_id, resume=args.resume,
                    stop_after=args.stop_after, profile=args.command == 'profile')
            elif args.command == 'estimator-fit':
                root, ok = run_fit(cfg, run_id=args.run_id, resume=args.resume, stop_after=args.stop_after)
            else:
                if not args.run_dir:
                    p.error('estimator-score requires --run-dir')
                root, ok = run_score(cfg, args.run_dir)
        if not ok:
            raise SystemExit(f'Partial/failed comparisons; inspect {root}/status.json (other results are preserved)')
        return root
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        raise SystemExit(str(error)) from error


if __name__ == '__main__':
    main()
