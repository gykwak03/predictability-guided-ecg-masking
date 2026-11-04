#!/usr/bin/env python3
"""Teacher-free paired Within/Cross mechanism pilot. Real compute requires Slurm."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config')
    p.add_argument('--artifact-dir')
    p.add_argument('--output',required=True)
    p.add_argument('--smoke',action='store_true',help='Synthetic CPU software check only')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    p.add_argument('--checkpoints',type=int,nargs='+',default=[500,1500,3000])
    p.add_argument('--fork-updates',type=int,default=200)
    p.add_argument('--fork-lr',type=float,default=1e-4)
    p.add_argument('--fork-loss',choices=['anchor','all-hidden'],default='anchor')
    p.add_argument('--probe-ecgs',type=int,default=128)
    p.add_argument('--panel-repeats',type=int,default=4)
    p.add_argument('--radius',type=int,default=2)
    p.add_argument('--save-every',type=int,default=50)
    p.add_argument('--plan',action='store_true',help='Resolve and print the plan without importing Torch or reading ECG arrays')
    return p


def resolve(args):
    from experiments.phase1.config import resolve_config
    cfg = resolve_config(args.config,dict(artifact_dir=args.artifact_dir),smoke=args.smoke)
    options = {k:getattr(args,k) for k in ('seeds','checkpoints','fork_updates','fork_lr','fork_loss',
        'probe_ecgs','panel_repeats','radius','save_every')}
    if args.smoke:
        if args.config:
            raise ValueError('--smoke must not use a real-data config')
        options.update(seeds=[42],checkpoints=[1,2],fork_updates=1,probe_ecgs=2,panel_repeats=1,save_every=1)
        cfg.update(validation_ecgs=2)
    if cfg['synthetic'] != args.smoke:
        raise ValueError('Synthetic mode must be explicit via --smoke')
    if cfg['patch_samples'] != 50 or cfg['mask_ratio'] != .5:
        raise ValueError('This matched intervention requires P50 and mask ratio 50%')
    if cfg['train_ecgs'] < cfg['microbatch']*cfg['accumulation'] or cfg['validation_ecgs'] < 1:
        raise ValueError('Need a full effective training batch and a positive validation cap')
    if (not options['seeds'] or any(s<0 for s in options['seeds']) or
        len(set(options['seeds'])) != len(options['seeds'])):
        raise ValueError('Unique nonnegative seeds required')
    if options['checkpoints'] != sorted(set(options['checkpoints'])) or min(options['checkpoints']) < 1:
        raise ValueError('Checkpoints must be unique, positive, increasing')
    if any(options[k]<1 for k in ('fork_updates','probe_ecgs','panel_repeats','save_every')) or not 1<=options['radius']<=12:
        raise ValueError('Positive sizes and radius in [1,12] required')
    import math
    if not math.isfinite(options['fork_lr']) or options['fork_lr'] <= 0:
        raise ValueError('Finite positive fork LR required')
    return cfg, options


def main(argv=None):
    args = parser().parse_args(argv)
    cfg, options = resolve(args)
    if args.plan:
        import json
        print(json.dumps(dict(config=cfg,experiment=options,output=str(Path(args.output).resolve()),
            total_optimizer_updates=len(options['seeds'])*(max(options['checkpoints'])+
                3*len(options['checkpoints'])*options['fork_updates']),
            fork_objective='identical single anchor' if options['fork_loss']=='anchor' else 'all hidden patches; target sets vary with context'),
            ensure_ascii=False,indent=2))
        return 0
    from experiments.dependency_gain import run_experiment
    print('RUN_DIR='+str(run_experiment(cfg,options,args.output,args.resume)),flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
