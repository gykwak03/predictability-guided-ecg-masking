"""Read-only NCF checkpoint audit on existing validation probe banks.

Lives outside frozen experiment sources so existing plans remain valid.
Never reads test data or modifies checkpoints, probe banks or training results.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from experiments.ncf.common import read_json, read_plan, write_json


class Moments:
    def __init__(self):
        self.n = 0
        self.sums = np.zeros(5, np.float64)

    def add(self, prediction, target):
        p, y = np.asarray(prediction, dtype=np.float64).ravel(), np.asarray(target, dtype=np.float64).ravel()
        if p.shape != y.shape or not len(p) or not np.isfinite(p).all() or not np.isfinite(y).all():
            raise ValueError('Invalid hidden prediction/target samples')
        self.n += len(p)
        self.sums += [p.sum(), y.sum(), p @ p, y @ y, p @ y]

    def result(self):
        if not self.n:
            raise ValueError('No hidden samples')
        pm, ym, pp, yy, py = self.sums / self.n
        pv, yv = max(0., pp - pm * pm), max(0., yy - ym * ym)
        mse = max(0., pp + yy - 2 * py)
        corr = None if pv * yv <= 1e-24 else float(np.clip((py - pm * ym) / np.sqrt(pv * yv), -1, 1))
        return dict(hidden_samples=self.n, model_mse=float(mse), zero_mse=float(yy),
                    improvement_vs_zero=None if yy == 0 else float(1 - mse / yy),
                    prediction_mean=float(pm), prediction_std=float(np.sqrt(pv)),
                    target_mean=float(ym), target_std=float(np.sqrt(yv)),
                    prediction_target_correlation=corr,
                    prediction_target_std_ratio=None if yv == 0 else float(np.sqrt(pv / yv)))


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_times(study, plan, seed):
    rows = []
    for method in plan['config']['methods']:
        folder = study / 'runs' / method / f'seed{seed}'
        path = folder / 'completed.json'
        if not path.exists():
            rows.append(dict(method=method, seed=seed, completed=False))
            continue
        result = read_json(path)
        if result['contract']['plan_id'] != plan['plan_id'] or result['contract']['method'] != method or result['contract']['seed'] != seed:
            raise ValueError(f'Foreign completed result: {path}')
        train, final = result['training_pipeline_seconds'], result['final_evaluation_seconds']
        rows.append(dict(method=method, seed=seed, completed=True, step=result['step'],
                         training_seconds=train, final_evaluation_seconds=final,
                         recorded_total_seconds=train + final, recorded_total_hours=(train + final) / 3600,
                         macro_auroc=result['linear_probe']['macro_auroc']))
    return rows


def plot_case(path, signal, prediction, mask, p, anchor=None):
    import matplotlib.pyplot as plt
    from experiments.wcb.data import LEADS
    time = np.arange(2500) / 250
    hidden = np.repeat(mask.reshape(12, -1), p, axis=1)
    anchors = None if anchor is None else np.repeat(anchor.reshape(12, -1), p, axis=1)
    left = 0.
    if anchors is not None and anchors.any():
        center = np.flatnonzero(anchors.any(axis=0)).mean() / 250
        left = float(np.clip(center - 1.5, 0, 7))
    fig, axes = plt.subplots(3, 2, figsize=(14, 8))
    for row, lead in enumerate((1, 6, 10)):
        for col, ax in enumerate(axes[row]):
            ax.plot(time, signal[lead], color='#222222', lw=.8, label='Original')
            ax.plot(time, np.where(~hidden[lead], signal[lead], np.nan), color='#0072B2', lw=1.1, label='Visible input')
            ax.plot(time, np.where(hidden[lead], prediction[lead], np.nan), color='#D55E00', lw=1., label='Prediction (hidden only)')
            ax.axhline(0, color='#009E73', ls='--', lw=.8, label='Zero baseline')
            ax.fill_between(time, 0, 1, where=hidden[lead], transform=ax.get_xaxis_transform(), color='gray', alpha=.10)
            if anchors is not None:
                ax.fill_between(time, 0, 1, where=anchors[lead], transform=ax.get_xaxis_transform(), color='purple', alpha=.12)
            ax.set_xlim((0, 10) if col == 0 else (left, left + 3))
            ax.set_ylabel(f'{LEADS[lead]} (z-score)')
            # Same y limits for original and prediction; never rescale prediction separately.
            amplitude = max(1., float(np.max(np.abs(signal[lead]))), float(np.max(np.abs(prediction[lead]))))
            ax.set_ylim(-amplitude * 1.05, amplitude * 1.05)
    axes[0, 0].legend(loc='upper right', fontsize=7)
    axes[0, 0].set_title('Full 10 seconds; gray = hidden')
    axes[0, 1].set_title('3-second zoom; purple = target, when present')
    axes[-1, 0].set_xlabel('Seconds')
    axes[-1, 1].set_xlabel('Seconds')
    fig.suptitle(path.stem, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def audit_bank(model, signals, frame, bank_path, device, batch, output, name, examples):
    import torch
    aggregate = Moments()
    strata = {}
    rows, example_ids = [], set()
    with np.load(bank_path, allow_pickle=False) as bank:
        model.eval()
        with torch.no_grad():
            for start in range(0, len(bank['indices']), batch):
                ids = bank['indices'][start:start + batch]
                x = torch.tensor(np.asarray(signals[ids]), device=device)
                masks = np.asarray(bank['masks'][start:start + batch])
                prediction, target = model(x, torch.tensor(masks, device=device))
                predictions, targets = prediction.cpu().numpy(), target.cpu().numpy()
                for j, index in enumerate(ids):
                    mask = masks[j]
                    value = Moments()
                    value.add(predictions[j][mask], targets[j][mask])
                    aggregate.add(predictions[j][mask], targets[j][mask])
                    stratum = str(bank['strata'][start + j]) if 'strata' in bank else 'random'
                    strata.setdefault(stratum, Moments()).add(predictions[j][mask], targets[j][mask])
                    rows.append(dict(case=start + j, artifact_index=int(index), ecg_id=int(frame.iloc[int(index)].ecg_id),
                                     stratum=stratum, **value.result()))
                    if len(example_ids) < examples and int(index) not in example_ids:
                        example_ids.add(int(index))
                        anchor = bank['anchors'][start + j] if 'anchors' in bank else None
                        plot_case(output / f'{name}_ecg{rows[-1]["ecg_id"]}.png',
                                  targets[j].reshape(12, 2500), predictions[j].reshape(12, 2500),
                                  mask, model.p, anchor)
    write_csv(output / f'{name}_cases.csv', rows)
    stratum_rows = [dict(stratum=k, **v.result()) for k, v in strata.items()]
    write_csv(output / f'{name}_strata.csv', stratum_rows)
    result = aggregate.result()
    # Match production balanced metric: equal mean of stratum means.
    result.update(cases=len(rows), equal_stratum_model_mse=float(np.mean([r['model_mse'] for r in stratum_rows])),
                  equal_stratum_zero_mse=float(np.mean([r['zero_mse'] for r in stratum_rows])))
    return result


def diagnose(args):
    study = args.study_dir.resolve()
    plan = read_plan(study)
    c = plan['config']
    seed = c['seeds'][0] if args.seed is None else args.seed
    if seed not in c['seeds']:
        raise ValueError('Seed must be present in the original plan')
    timings = run_times(study, plan, seed)
    print('Recorded pipeline times (not Slurm wall time; concurrent runs must not be summed as elapsed time):')
    for row in timings:
        print(json.dumps(row, sort_keys=True))
    if args.timing_only:
        return
    import matplotlib
    matplotlib.use('Agg')
    import torch
    from experiments.ncf.data import Prepared
    from experiments.ncf.training import device_for, initialize
    from experiments.ncf.evaluation import linear_probe
    output = args.output_dir or study / 'diagnostics' / datetime.now(timezone.utc).strftime('reconstruction_%Y%m%dT%H%M%S%fZ')
    output = output.resolve()
    # Separate, new directory: do not overwrite any experiment or previous audit.
    if output == study or study in output.parents and output.relative_to(study).parts[0] in ('runs', 'prepared'):
        raise ValueError('Output must not be a training or prepared-data directory')
    output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault('MPLCONFIGDIR', str(output / '.matplotlib-cache'))
    device = device_for(c)
    prepared = Prepared(study, plan)
    write_csv(output / 'runtime.csv', timings)
    methods = args.methods or c['methods']
    if set(methods) - set(c['methods']):
        raise ValueError('Requested method is absent from the original plan')
    model = initialize(c, seed).to(device)
    summaries = []
    untrained = None
    for method in ['untrained', *methods]:
        completed = None
        if method != 'untrained':
            folder = study / 'runs' / method / f'seed{seed}'
            completed = read_json(folder / 'completed.json')
            state = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=False)
            expected = dict(plan_id=plan['plan_id'], method=method, seed=seed, replay=None,
                            prepared_id=read_json(prepared.root / 'completed.json')['prepared_id'])
            if state['contract'] != expected or completed['contract'] != expected or state['step'] != completed['step']:
                raise ValueError(f'Checkpoint/completion contract mismatch: {method}')
            model.load_state_dict(state['training']['model'], strict=True)
            del state
        for bank in ('random', 'balanced'):
            result = audit_bank(model, prepared.val, prepared.val_frame,
                                prepared.root / f'{bank}_validation.npz', device, c['microbatch'],
                                output, f'{method}_{bank}', args.examples)
            key = 'random_validation' if bank == 'random' else 'context_balanced_validation'
            original = None if completed is None else completed[key]['loss']
            row = dict(method=method, seed=seed, bank=bank, **result,
                       saved_validation_mse=original,
                       difference_from_saved_mse=None if original is None else result['equal_stratum_model_mse'] - original)
            summaries.append(row)
            write_csv(output / 'reconstruction_summary.csv', summaries)
            print(json.dumps(row, sort_keys=True), flush=True)
        if method == 'untrained' and args.untrained_linear_probe:
            untrained = linear_probe(model, prepared, output / 'untrained_linear_probe', seed, device)
            print('Untrained encoder linear probe:', json.dumps(untrained), flush=True)
    write_json(output / 'report.json', dict(plan_id=plan['plan_id'], seed=seed, timings=timings,
                                          reconstruction=summaries, untrained_linear_probe=untrained,
                                          checkpoints_modified=False, test_opened=False))
    report = ['# NCF reconstruction audit', '',
              'Existing validation banks only. All hidden samples; no test data. Examples are first distinct bank ECGs, not selected by model score.',
              'Positive improvement_vs_zero means lower MSE than zero prediction. Std ratio and correlation measure hidden predictions only.',
              'A low std ratio alone does not prove a bug. Compare zero baseline, correlation, plots and initialized-model results together.',
              'Saved metric differences should be near floating-point tolerance; large differences require investigation.',
              'runtime.csv sums recorded training and final-evaluation timers. It excludes startup and some I/O; Slurm elapsed is separate.',
              'Runs execute concurrently, so their sum is GPU task time, not study elapsed time.', '',
              'method | bank | model MSE | zero MSE | improvement vs zero | std ratio | correlation',
              '--- | --- | --- | --- | --- | --- | ---']
    for r in summaries:
        report.append(f"{r['method']} | {r['bank']} | {r['model_mse']:.8f} | {r['zero_mse']:.8f} | {r['improvement_vs_zero']} | {r['prediction_target_std_ratio']} | {r['prediction_target_correlation']}")
    if untrained is not None:
        report += ['', f"Untrained encoder validation macro-AUROC: {untrained['macro_auroc']:.8f}.",
                   'The same trained linear-head protocol and validation selection are used; this is not test-set evidence.']
    (output / 'REPORT.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print(f'DIAGNOSTICS_DIR={output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--methods', nargs='+')
    parser.add_argument('--examples', type=int, default=2)
    parser.add_argument('--timing-only', action='store_true')
    parser.add_argument('--untrained-linear-probe', action='store_true')
    args = parser.parse_args()
    if args.examples < 0:
        parser.error('--examples must be nonnegative')
    diagnose(args)


if __name__ == '__main__':
    main()
