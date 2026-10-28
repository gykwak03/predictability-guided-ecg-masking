"""Render stored evidence, including partial/failed conditions; never choose by MSE."""
from __future__ import annotations

from html import escape
import json
from pathlib import Path
import uuid

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import write_json


def analyze(run_dir):
    root = Path(run_dir).resolve()
    status = json.loads((root / 'status.json').read_text(encoding='utf-8'))
    output = root / ('analysis_' + uuid.uuid4().hex[:8])
    output.mkdir()
    blocks = [f'<h1>Phase 1 evidence</h1><p>Run: {escape(str(root))}</p>',
        f'<pre>{escape(json.dumps(status, ensure_ascii=False, indent=2))}</pre>',
        '<p>Synthetic smoke checks code only. Lower NRMSE means easier reconstruction. '
        'Review validation stability, zero-baseline skill, prediction/target RMS, correlation and cost together. '
        'No patch or ratio is automatically selected. Missing observations remain NaN.</p>']
    image_count = 0
    for csv in sorted(root.rglob('history.csv')):
        if any(p.startswith('analysis_') for p in csv.relative_to(root).parts):
            continue
        frame = pd.read_csv(csv)
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), constrained_layout=True)
        for pool, group in frame.groupby('pool'):
            axes[0].plot(group['update'], group.pooled_nrmse, label=pool)
            axes[1].plot(group['update'], group.skill_vs_zero, label=pool)
            axes[2].plot(group['update'], group.rms_ratio, label=pool)
        for ax, title in zip(axes, ['Hidden NRMSE', 'Skill versus zero', 'Prediction / target RMS']):
            ax.set(xlabel='Optimizer update', title=title)
            ax.grid(alpha=.2)
            ax.legend()
        filename = f'{image_count}_learning_curves.png'
        fig.savefig(output / filename, dpi=125)
        plt.close(fig)
        blocks.append(f'<h2>{escape(str(csv.relative_to(root)))}</h2><img src="{filename}">')
        image_count += 1
    for csv in sorted(root.rglob('aggregate_map.csv')):
        if any(p.startswith('analysis_') for p in csv.relative_to(root).parts):
            continue
        table = pd.read_csv(csv)
        q = len(table)//12
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        for ax, column, title in zip(axes, ['nrmse', 'hidden_observations'],
                                    ['Aggregate hidden NRMSE', 'Hidden observations (pairs correlated)']):
            image = ax.imshow(table[column].to_numpy().reshape(12, q), aspect='auto', extent=[0, 10, 11.5, -.5])
            ax.set(xlabel='Time (s)', ylabel='Canonical lead index', title=title)
            fig.colorbar(image, ax=ax)
        filename = f'{image_count}_map.png'
        fig.savefig(output / filename, dpi=125)
        plt.close(fig)
        blocks.append(f'<h2>{escape(str(csv.relative_to(root)))}</h2><img src="{filename}">')
        image_count += 1
        ecg = np.load(csv.parent / 'ecg_maps.npz', allow_pickle=False)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        for ax, key in zip(axes, ['nrmse', 'hidden_observations']):
            image = ax.imshow(ecg[key][0], aspect='auto', extent=[0, 10, 11.5, -.5])
            ax.set(title=f'ECG {ecg["ecg_id"][0]}: {key}', xlabel='Time (s)', ylabel='Lead index')
            fig.colorbar(image, ax=ax)
        filename = f'{image_count}_ecg_map.png'
        fig.savefig(output / filename, dpi=125)
        plt.close(fig)
        blocks.append(f'<img src="{filename}">')
        image_count += 1
    for path in sorted(root.rglob('examples.npz')):
        if any(p.startswith('analysis_') for p in path.relative_to(root).parts):
            continue
        examples = np.load(path, allow_pickle=False)
        target, prediction, mask = examples['target'], examples['prediction'], examples['mask']
        fig, axes = plt.subplots(len(target), 2, figsize=(14, 2.7*len(target)), squeeze=False, constrained_layout=True)
        patch = 2500//(mask.shape[1]//12)
        hidden = np.repeat(mask.reshape(len(target), 12, -1), patch, axis=2)
        t = np.arange(2500)/250
        for i in range(len(target)):
            for ax, lead, name in zip(axes[i], [1, 7], ['II', 'V2']):
                ax.plot(t, target[i, lead], color='#334155', lw=.7, label='Target')
                ax.plot(t, np.where(hidden[i, lead], prediction[i, lead], np.nan), color='#d94801', lw=.8, label='Hidden prediction')
                ax.fill_between(t, 0, 1, where=hidden[i, lead], transform=ax.get_xaxis_transform(), color='#fbbf24', alpha=.15)
                ax.set(xlabel='Time (s)', ylabel='Per-record z units', title=f'{examples["mask_id"][i]} / {name}')
                ax.legend(fontsize=7)
        filename = f'{image_count}_waveform.png'
        fig.savefig(output / filename, dpi=125)
        plt.close(fig)
        blocks.append(f'<h2>{escape(str(path.relative_to(root)))}</h2><img src="{filename}">')
        image_count += 1
    for filename in ('summary.csv', 'estimator_summary.csv'):
        if (root / filename).exists():
            table = pd.read_csv(root / filename)
            blocks.insert(2, table.to_html(index=False, float_format=lambda x: f'{x:.5g}'))
    for path in sorted(root.rglob('*error.json')):
        blocks.append(f'<h2>Recorded error: {escape(str(path.relative_to(root)))}</h2><pre>{escape(path.read_text(encoding="utf-8"))}</pre>')
    (output / 'REPORT.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:15px system-ui;margin:32px;max-width:1400px}img{max-width:100%}table{display:block;overflow:auto}pre{white-space:pre-wrap}</style>'+'\n'.join(blocks), encoding='utf-8')
    write_json(output / 'analysis.json', dict(source_status=status['status'], images=image_count,
        optimal_patch=None, optimal_ratio=None, test_opened=False,
        note='All ECG maps and counts are in each ecg_maps.npz; the report previews the first ECG. No downstream utility is inferred.'))
    print(f'REPORT={output / "REPORT.html"}', flush=True)
    return output
