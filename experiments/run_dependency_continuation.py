#!/usr/bin/env python3
"""Continue saved Random-MAE checkpoints; never repeat pretraining."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', required=True, help='Original dependency_gain_results directory')
    p.add_argument('--output', required=True, help='NEW directory outside the original run')
    p.add_argument('--artifact-dir', help='Relocate the SAME preprocessing artifact')
    p.add_argument('--seeds', nargs='+', type=int)
    p.add_argument('--checkpoints', nargs='+', type=int)
    p.add_argument('--updates', type=int, default=200)
    p.add_argument('--optimizer-modes', nargs='+', choices=['keep', 'reset'], default=['keep', 'reset'])
    p.add_argument('--losses', nargs='+', choices=['all-hidden', 'anchor'], default=['all-hidden'])
    p.add_argument('--lr-policy', choices=['continue', 'constant'], default='continue')
    p.add_argument('--lr', type=float, help='Required only with --lr-policy constant')
    p.add_argument('--eval-every', type=int, default=50)
    p.add_argument('--save-every', type=int, default=50)
    p.add_argument('--validation-repeats', type=int, default=2, help='Complement pairs per validation ECG')
    p.add_argument('--resume', action='store_true', help='Resume this NEW continuation run unchanged')
    p.add_argument('--smoke', action='store_true', help='Only accept a synthetic source run; CPU test')
    p.add_argument('--plan', action='store_true', help='Check source metadata/files without Torch or ECG access')
    return p


def cases_for(losses):
    cases = []
    if 'all-hidden' in losses:
        cases += [dict(case=c, arm=a, loss='all-hidden') for c,a in
                  [('D','mae-random'), ('E','within'), ('F','cross')]]
        if 'anchor' in losses:
            cases.append(dict(case='G', arm='random', loss='all-hidden'))
    if 'anchor' in losses:
        cases += [dict(case=c, arm=a, loss='anchor') for c,a in
                  [('A','within'), ('B','cross'), ('C','random')]]
    return cases


def resolve(args):
    # Intentionally no NumPy/Torch imports: safe for Slurm submission on login node.
    from experiments.phase1.config import resolve_config
    source = Path(args.source_run).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError('--output must be separate from the read-only source run')
    path = source/'contract.json'
    saved = json.loads(path.read_text(encoding='utf-8'))
    if saved.get('version') != 'paired-context-v1':
        raise ValueError('Expected an original paired-context-v1 run contract')
    if bool(saved['config']['synthetic']) != args.smoke:
        raise ValueError('Synthetic sources require --smoke; real sources must not use --smoke')
    relocation = args.artifact_dir
    if not args.smoke and relocation is None:
        # Original contracts omit filesystem paths; the deployment's known path is a default only.
        import os
        relocation = os.environ.get('ECG_ARTIFACT_DIR', '/data/gykwak03/capstone2/ptbxl_250hz_bp067_50')
    overrides = dict(saved['config'])
    if relocation is not None:
        overrides['artifact_dir'] = relocation
    cfg = resolve_config(overrides=overrides, environ={})
    if cfg['patch_samples'] != 50 or cfg['mask_ratio'] != .5:
        raise ValueError('P50 / exact per-lead 50% is required')
    options = dict(seeds=args.seeds if args.seeds is not None else saved['options']['seeds'],
        checkpoints=args.checkpoints if args.checkpoints is not None else saved['options']['checkpoints'][-2:],
        updates=args.updates, optimizer_modes=args.optimizer_modes, losses=args.losses,
        lr_policy=args.lr_policy, lr=args.lr, eval_every=args.eval_every,
        save_every=args.save_every, validation_repeats=args.validation_repeats)
    for name in ('seeds', 'checkpoints'):
        value = options[name]
        if not value or value != sorted(set(value)) or not set(value) <= set(saved['options'][name]):
            raise ValueError(f'{name} must be an increasing subset of the original source run')
    for name in ('losses', 'optimizer_modes'):
        if len(options[name]) != len(set(options[name])):
            raise ValueError(f'Duplicate {name}')
        options[name] = sorted(options[name])
    if any(options[k] < 1 for k in ('updates','eval_every','save_every','validation_repeats')):
        raise ValueError('Positive update/interval/repeat counts required')
    if options['lr_policy'] == 'constant':
        if options['lr'] is None or not math.isfinite(options['lr']) or options['lr'] <= 0:
            raise ValueError('--lr-policy constant requires a finite positive --lr')
    elif options['lr'] is not None:
        raise ValueError('--lr is only allowed with --lr-policy constant')
    for seed in options['seeds']:
        for checkpoint in options['checkpoints']:
            checkpoint_path = source/f'seed_{seed}'/'pretrain'/f'update_{checkpoint}.pt'
            if not checkpoint_path.is_file() or checkpoint_path.stat().st_size == 0:
                raise FileNotFoundError(f'Missing saved PRETRAIN checkpoint: {checkpoint_path}; select completed seeds/checkpoints')
    if args.smoke:
        # Keep model/data identity untouched; only shorten the NEW continuation.
        options.update(updates=min(2,args.updates), eval_every=1, save_every=1, validation_repeats=1)
    return source, output, cfg, saved, options


def main(argv=None):
    args = parser().parse_args(argv)
    source, output, cfg, saved, options = resolve(args)
    if args.plan:
        cases = cases_for(options['losses'])
        print(json.dumps(dict(source=str(source), output=str(output), options=options, cases=cases,
            pretraining_updates=0, continuation_updates=len(options['seeds'])*len(options['checkpoints'])*
                len(cases)*len(options['optimizer_modes'])*options['updates']), ensure_ascii=False, indent=2))
        return 0
    from experiments.dependency_continuation import run_continuation
    print('RUN_DIR='+str(run_continuation(source,output,cfg,saved,options,resume=args.resume)),flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
