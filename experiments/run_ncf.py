"""Annotation-free Near / Cross-lead / Far masking study, trained from scratch."""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.ncf.common import (DEFAULTS, VERSION, RATE_ORDER, TWO_BRANCH, digest, file_sha,
                                    read_json, write_json, sources, read_plan, validate_config, geometry)


def create_plan(args):
    c = copy.deepcopy(DEFAULTS)
    if args.config:
        provided = read_json(args.config)
        unknown = set(provided) - set(c)
        if unknown:
            raise ValueError(f'Unknown NCF settings: {unknown}')
        c.update(provided)
    if args.smoke:
        c.update(synthetic=True, device='cpu', model_size='smoke', patch_samples=25,
                 updates=4, lr_horizon=4, H=2, seeds=[42], train_ecgs=6,
                 control_ecgs=2, balanced_ecgs=1, validation_ecgs=2, balanced_validation_ecgs=1,
                 microbatch=1, accumulation=1, cpu_threads=1, linear_updates=3, linear_batch=4,
                 lr_warmup=1, random_every=1)
    for name in ('methods', 'seeds', 'updates'):
        value = getattr(args, name, None)
        if value is not None:
            c[name] = value
    validate_config(c)
    artifact = None
    if not c['synthetic']:
        path = args.artifact_dir or os.environ.get('ECG_ARTIFACT_DIR')
        if not path or not (Path(path) / 'manifest.json').is_file():
            raise FileNotFoundError('Set --artifact-dir / ECG_ARTIFACT_DIR to the reviewed 250Hz artifact')
        artifact = str(Path(path).resolve())
    plan = dict(version=VERSION, config=c, artifact_dir=artifact,
                artifact_manifest=None if artifact is None else file_sha(Path(artifact) / 'manifest.json'),
                sources=sources(), rate_order=list(RATE_ORDER), geometry=geometry(c),
                annotation_free=True, from_scratch=True, random_warmup=False,
                reward='context_balanced_all_hidden_lp', test_opened=False)
    plan['plan_id'] = digest(plan)
    args.study_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.study_dir / 'plan.json', plan)
    tasks = [dict(index=i, method=m, seed=s) for i, (m, s) in
             enumerate((m, s) for m in c['methods'] for s in c['seeds'])]
    write_json(args.study_dir / 'tasks.json', tasks)
    print(f'STUDY_DIR={args.study_dir.resolve()}\nTASKS={len(tasks)}\nPLAN_ID={plan["plan_id"]}')
    print(f'RATE_ORDER=Near,Cross-lead,Far\nCOUNTS={plan["geometry"]["counts"]}')


def profile(study, plan):
    import time
    import numpy as np
    import pandas as pd
    import torch
    from experiments.ncf.training import device_for, initialize
    from experiments.wcb.model import masked_mse
    from experiments.ncf.data import Prepared
    from experiments.ncf.masks import matched_masks
    c = plan['config']
    prepared = Prepared(study, plan)
    device = device_for(c)
    model = initialize(c, c['seeds'][0]).to(device)
    x = torch.zeros((c['microbatch'], 12, 2500), device=device)
    mask = torch.tensor(np.array([matched_masks(c, [[.5, .5, .5]], j)[0][0] for j in range(len(x))]), device=device)
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    model.zero_grad(set_to_none=True)
    pred, target = model(x, mask)
    masked_mse(pred, target, mask).backward()
    sync()
    start = time.perf_counter()
    for _ in range(2):
        model.zero_grad(set_to_none=True)
        pred, target = model(x, mask)
        masked_mse(pred, target, mask).backward()
    sync()
    train_seconds = (time.perf_counter() - start) / 2
    model.eval()
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(2):
            model(x, mask)
    sync()
    eval_seconds = (time.perf_counter() - start) / 2
    start = time.perf_counter()
    with torch.no_grad():
        model.features(x)
    sync()
    feature_seconds = time.perf_counter() - start
    bc, rc = c['balanced_ecgs'] * 8, c['control_ecgs'] * 2
    bv, rv = c['balanced_validation_ecgs'] * 8, c['validation_ecgs'] * 2
    blocks = int(np.ceil(c['updates'] / c['H']))
    feature_batches = int(np.ceil(len(prepared.selection['fit']) / c['microbatch']) +
                          np.ceil(len(prepared.val) / c['microbatch']))
    rows = []
    for method in c['methods']:
        branches = 2 if method in TWO_BRANCH else 1
        attempted = c['updates'] * branches
        cases = bc + rc + blocks * branches * bc + int(np.ceil(blocks / c['random_every'])) * rc + bc + rc + bv + rv
        estimate = attempted * c['accumulation'] * train_seconds + cases / c['microbatch'] * eval_seconds + feature_batches * feature_seconds
        rows.append(dict(method=method, seeds=len(c['seeds']), branches_per_decision=branches,
                         committed_updates=c['updates'], attempted_updates=attempted, probe_forward_cases=cases,
                         feature_batches=feature_batches, estimated_hours_per_seed=estimate / 3600,
                         estimated_hours_all_seeds=estimate * len(c['seeds']) / 3600))
    pd.DataFrame(rows).to_csv(study / 'profile_budget.csv', index=False)
    state_bytes = sum(p.numel() * p.element_size() * 2 for p in model.parameters() if p.requires_grad)
    write_json(study / 'profile.json', dict(
        train_microbatch_seconds=train_seconds, eval_microbatch_seconds=eval_seconds,
        full_input_feature_seconds=feature_seconds,
        measured_peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0,
        adam_moment_storage_estimate_bytes=state_bytes, optimizer_updates=0, model_discarded=True,
        estimated_total_gpu_hours=sum(r['estimated_hours_all_seeds'] for r in rows),
        estimate_excludes=['CPU mask generation', 'optimizer steps', 'branch snapshot/restore',
                          'checkpoint/file writes', 'linear head training', 'final test'],
        peak_memory_excludes_optimizer_states=True, estimate_is_not_runtime_guarantee=True))
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


def analyze(study, plan):
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = []
    for path in sorted((study / 'runs').glob('*/seed*/completed.json')):
        result = read_json(path)
        if result['contract']['plan_id'] != plan['plan_id']:
            raise ValueError('Foreign completed run')
        rows.append(dict(method=path.parent.parent.name, seed=result['contract']['seed'],
                         updates=result['step'], attempted_updates=result['attempted_updates'],
                         random_validation_mse=result['random_validation']['loss'],
                         context_validation_mse=result['context_balanced_validation']['loss'],
                         target_validation_mse=result['context_balanced_validation']['target_loss'],
                         random_control_mse=result['random_control']['loss'],
                         context_control_mse=result['context_balanced_control']['loss'],
                         macro_auroc=result['linear_probe']['macro_auroc'], macro_auprc=result['linear_probe']['macro_auprc'],
                         training_seconds=result['training_pipeline_seconds'],
                         final_evaluation_seconds=result['final_evaluation_seconds']))
    if not rows:
        raise ValueError('No complete runs yet')
    out = study / 'analysis'
    out.mkdir(exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'by_seed.csv', index=False)
    summary = frame.groupby('method').agg(
        seeds=('seed', 'count'), auroc_mean=('macro_auroc', 'mean'), auroc_std=('macro_auroc', 'std'),
        random_mse_mean=('random_validation_mse', 'mean'), context_mse_mean=('context_validation_mse', 'mean'),
        training_seconds_mean=('training_seconds', 'mean'), attempted_updates_mean=('attempted_updates', 'mean'))
    summary.to_csv(out / 'summary.csv')
    contrasts = []
    if 'random' in set(frame.method):
        baseline = frame[frame.method.eq('random')].set_index('seed')
        for row in frame.itertuples():
            if row.method != 'random' and row.seed in baseline.index:
                base = baseline.loc[row.seed]
                contrasts.append(dict(method=row.method, seed=row.seed, delta_auroc=row.macro_auroc - base.macro_auroc,
                                      delta_random_mse=row.random_validation_mse - base.random_validation_mse,
                                      delta_context_mse=row.context_validation_mse - base.context_validation_mse))
    pd.DataFrame(contrasts).to_csv(out / 'paired_vs_random.csv', index=False)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(range(len(summary)), summary.auroc_mean, yerr=summary.auroc_std.fillna(0), capsize=3)
    ax.set(xticks=range(len(summary)), xticklabels=summary.index, ylabel='Validation frozen macro AUROC')
    ax.tick_params(axis='x', labelrotation=30)
    fig.tight_layout()
    fig.savefig(out / 'auroc.png', dpi=140)
    plt.close(fig)
    policy_rows = []
    for path in sorted((study / 'runs').glob('*/seed*/history.json')):
        history = read_json(path)
        if not history:
            continue
        method, seed_name = path.parent.parent.name, path.parent.name
        steps = [r['step'] for r in history]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for j, label in enumerate(('Near', 'Cross-lead', 'Far')):
            axes[0].plot(steps, [r['center_after'][j] for r in history], label=label)
            axes[0].plot(steps, [r['masks'][r['chosen']]['realized'][RATE_ORDER[j]] for r in history], ls=':', alpha=.5)
        axes[0].plot(steps, [r['masks'][r['chosen']]['realized']['background'] for r in history], ls='--', color='gray', label='Background')
        axes[0].set(xlabel='Committed updates', ylabel='Reveal rate: center (solid), realized (dotted)', ylim=(0, 1))
        axes[0].legend()
        axes[1].plot(steps, [r['context_balanced']['loss'] for r in history], label='Context-Balanced (control)')
        random_points = [r for r in history if r['random_control'] is not None]
        axes[1].plot([r['step'] for r in random_points], [r['random_control']['loss'] for r in random_points], label='Random (control)')
        axes[1].set(xlabel='Committed updates', ylabel='Separate all-hidden MSE')
        axes[1].legend()
        fig.suptitle(f'{method} {seed_name}')
        fig.tight_layout()
        fig.savefig(out / f'{method}_{seed_name}_curves.png', dpi=140)
        plt.close(fig)
        for r in history:
            actual = r['masks'][r['chosen']]['realized']
            policy_rows.append(dict(method=method, seed=seed_name, step=r['step'],
                                    **{f'{name}_center': r['center_after'][i] for i, name in enumerate(RATE_ORDER)},
                                    **{f'{name}_realized': actual[name] for name in (*RATE_ORDER, 'background')},
                                    reward_scale=r['reward_scale'], chosen=r['chosen']))
    pd.DataFrame(policy_rows).to_csv(out / 'policy_rates.csv', index=False)
    report = ['# Near / Cross-lead / Far study', '',
              f"Complete planned runs: {int(frame.method.isin(plan['config']['methods']).sum())}/{len(plan['config']['methods']) * len(plan['config']['seeds'])}.",
              f"Synthetic-only training: {plan['config']['synthetic']}; annotation-free: True.",
              'Reward: Context-Balanced all-hidden MSE learning progress; Random is a separate metric.',
              'Validation probes use different patients from the policy-control bank. No test data used here.',
              'Compare committed updates AND costs. A three-seed pilot does not establish significance.',
              'Training timer excludes checkpoint/JSON writes and initial model/data setup; use Slurm elapsed time for full resource accounting.',
              '', summary.to_csv()]
    (out / 'REPORT.md').write_text('\n'.join(report), encoding='utf-8')
    print(summary.to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'validate', 'prepare', 'preview', 'profile', 'run', 'replay', 'analyze', 'test'])
    parser.add_argument('--study-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--artifact-dir')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--methods', nargs='+')
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--updates', type=int)
    parser.add_argument('--task-index', type=int)
    parser.add_argument('--method')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--stop-after', type=int)
    parser.add_argument('--source-method')
    parser.add_argument('--source-seed', type=int)
    parser.add_argument('--shuffle', action='store_true')
    args = parser.parse_args()
    if args.command == 'plan':
        return create_plan(args)
    plan = read_plan(args.study_dir)
    os.environ.setdefault('MPLCONFIGDIR', str((args.study_dir / '.matplotlib-cache').resolve()))
    if args.command == 'validate':
        print(plan['plan_id'])
        return
    if args.command in ('prepare', 'preview'):
        from experiments.ncf.data import prepare, preview
        if args.command == 'prepare':
            prepare(args.study_dir, plan)
        return preview(args.study_dir, plan)
    if args.command == 'analyze':
        return analyze(args.study_dir, plan)
    if args.command == 'profile':
        return profile(args.study_dir, plan)
    if args.task_index is not None:
        tasks = read_json(args.study_dir / 'tasks.json')
        if not 0 <= args.task_index < len(tasks):
            raise ValueError('Invalid task index')
        args.method, args.seed = tasks[args.task_index]['method'], tasks[args.task_index]['seed']
    if args.seed not in plan['config']['seeds']:
        raise ValueError('Seed must be in the plan')
    from experiments.ncf.training import run
    if args.command == 'replay':
        import numpy as np
        source = args.study_dir / 'runs' / str(args.source_method) / f'seed{args.source_seed}'
        completed = read_json(source / 'completed.json')
        if completed['contract']['plan_id'] != plan['plan_id'] or completed['step'] != plan['config']['updates']:
            raise ValueError('Replay requires a complete source horizon in this study')
        schedule = read_json(source / 'schedule.json')
        if args.shuffle:
            # Keep a final short block in place; do not pair a rate with a different duration.
            regular = [r for r in schedule if r['H'] == plan['config']['H']]
            np.random.default_rng(args.seed + 990).shuffle(regular)
            schedule = regular + [r for r in schedule if r['H'] != plan['config']['H']]
        name = f'replay_{args.source_method}_{args.source_seed}' + ('_shuffled' if args.shuffle else '')
        return run(args.study_dir, plan, 'replay', args.seed, replay=schedule,
                   folder=args.study_dir / 'runs' / name / f'seed{args.seed}', stop_after=args.stop_after)
    if args.method not in plan['config']['methods']:
        raise ValueError('Method must be in the plan')
    if args.command == 'test':
        import torch
        from experiments.ncf.training import device_for, initialize
        from experiments.ncf.data import Prepared
        from experiments.ncf.evaluation import final_test
        folder = args.study_dir / 'runs' / args.method / f'seed{args.seed}'
        result = read_json(folder / 'completed.json')
        if result['contract']['plan_id'] != plan['plan_id']:
            raise ValueError('Incomplete or foreign run')
        state = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=False)
        if state['contract'] != result['contract']:
            raise ValueError('Checkpoint contract mismatch')
        model = initialize(plan['config'], args.seed)
        device = device_for(plan['config'])
        model.load_state_dict(state['training']['model'])
        model.to(device)
        print(final_test(model, Prepared(args.study_dir, plan), folder, device))
        return
    return run(args.study_dir, plan, args.method, args.seed, stop_after=args.stop_after)


if __name__ == '__main__':
    main()
