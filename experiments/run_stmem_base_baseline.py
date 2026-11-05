"""Plan/run/resume the P50/r50 3000-update ST-MEM Base baseline."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan','run','smoke','analyze','seed-count'])
    parser.add_argument('--config', type=Path)
    parser.add_argument('--study-dir', type=Path)
    parser.add_argument('--artifact-dir')
    parser.add_argument('--st-mem-repo')
    parser.add_argument('--output-root')
    parser.add_argument('--microbatch', type=int)
    parser.add_argument('--accumulation', type=int)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--seed', type=int)
    group.add_argument('--seed-index', type=int)
    parser.add_argument('--stop-after', type=int, help='Additional updates before saving a resumable partial run')
    args = parser.parse_args(argv)
    from experiments.stmem_base_baseline import analyze, configuration, create_study, read_study, run_seed
    try:
        if args.command in ('plan','smoke'):
            if args.seed is not None or args.seed_index is not None or args.stop_after is not None:
                parser.error('Set seeds in the planning config; seed selection/stop-after are run options')
            cfg = configuration(args.config, dict(artifact_dir=args.artifact_dir, st_mem_repo=args.st_mem_repo,
                output_root=args.output_root, microbatch=args.microbatch, accumulation=args.accumulation),
                synthetic=args.command=='smoke')
            directory = args.study_dir or Path(cfg['output_root']) / (
                ('smoke_' if cfg['synthetic'] else 'base_') + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
            create_study(cfg, directory)
            if args.command=='smoke':
                if not run_seed(directory, cfg['seeds'][0]):
                    raise RuntimeError(f'Synthetic CPU check failed: {directory}')
                analyze(directory)
            return directory
        if args.study_dir is None:
            parser.error('--study-dir is required')
        if any(x is not None for x in (args.config,args.output_root,args.microbatch,args.accumulation)):
            parser.error('config/output-root/microbatch/accumulation are planning options; do not alter a frozen run')
        if args.command=='analyze':
            return analyze(args.study_dir)
        plan, _ = read_study(args.study_dir, args.artifact_dir, args.st_mem_repo)
        if args.command=='seed-count':
            print(len(plan['config']['seeds']))
            return
        if args.seed_index is not None:
            if not 0 <= args.seed_index < len(plan['config']['seeds']):
                parser.error('--seed-index is outside the frozen plan')
            seed = plan['config']['seeds'][args.seed_index]
        else:
            seed = args.seed if args.seed is not None else plan['config']['seeds'][0]
        if not run_seed(args.study_dir,seed,artifact_dir=args.artifact_dir,
                        st_mem_repo=args.st_mem_repo,stop_after=args.stop_after):
            raise SystemExit('Partial/failed Base run; inspect seed*/status.json and resubmit the same study to resume')
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        raise SystemExit(str(error)) from error


if __name__=='__main__':
    main()
