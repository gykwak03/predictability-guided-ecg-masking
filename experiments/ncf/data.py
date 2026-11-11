from __future__ import annotations

from itertools import product
from pathlib import Path
import numpy as np

from experiments.wcb.data import arrays, partition
from .common import RATE_ORDER, digest, file_sha, read_json, write_json, seed_for, geometry
from .masks import global_mask, matched_masks, context_sets


def random_bank(indices, n, seed):
    rows, masks = [], []
    for index in indices:
        mask = global_mask(n, np.random.default_rng(seed_for(seed, 'random-probe', index)))
        rows.extend([index, index])
        masks.extend([mask, ~mask])
    return dict(indices=np.array(rows, int), masks=np.array(masks, bool))


def balanced_bank(indices, c, seed):
    rows, masks, anchors, strata, starts, leads = [], [], [], [], [], []
    conditions = list(product(c['probe_rates'], repeat=3))
    for index in indices:
        index = int(index)
        key = seed_for(seed, 'context-probe', index)
        groups, target = context_sets(c, key)
        answers = matched_masks(c, conditions, key)
        anchor = np.zeros(geometry(c)['tokens'], bool)
        anchor[groups['target']] = True
        for rates, (mask, _) in zip(conditions, answers):
            rows.append(index)
            masks.append(mask)
            anchors.append(anchor)
            strata.append('_'.join(f'{name}{value:g}' for name, value in zip(('N', 'C', 'F'), rates)))
            starts.append(target['start_patch'])
            leads.append(target['leads'])
    return dict(indices=np.array(rows, int), masks=np.array(masks, bool), anchors=np.array(anchors, bool),
                strata=np.array(strata), target_starts=np.array(starts, int), target_leads=np.array(leads, int))


def prepare(study, plan):
    """Build fixed probe masks from metadata and geometry; no signal delineation."""
    root = Path(study) / 'prepared'
    root.mkdir(exist_ok=True)
    if (root / 'completed.json').exists():
        Prepared(study, plan)
        print('Prepared probes already complete; no annotation step', flush=True)
        return
    c = plan['config']
    selected = partition(plan)
    write_json(root / 'selection.json', selected)
    write_json(root / 'geometry.json', dict(**geometry(c), rate_order=list(RATE_ORDER),
               annotations_used=False, r_peaks_used=False, period_estimation_used=False))
    n = geometry(c)['tokens']
    for split, indices, balanced_count in (
        ('control', selected['control'], c['balanced_ecgs']),
        ('validation', selected['validation'], c['balanced_validation_ecgs']),
    ):
        key = seed_for(c['seed'], 'bank', split)
        np.savez_compressed(root / f'random_{split}.npz', **random_bank(indices, n, key))
        # A seeded subset, not the first patients/records in a sorted table.
        balanced_indices = np.random.default_rng(key).choice(indices, balanced_count, replace=False)
        np.savez_compressed(root / f'balanced_{split}.npz', **balanced_bank(balanced_indices, c, key))
    hashes = {p.name: file_sha(p) for p in sorted(root.iterdir()) if p.is_file() and p.name != 'completed.json'}
    completion = dict(plan_id=plan['plan_id'], prepared_id=digest(hashes), hashes=hashes,
                      train_ecgs=len(selected['fit']), control_ecgs=len(selected['control']),
                      balanced_control_cases=c['balanced_ecgs'] * 8,
                      balanced_validation_cases=c['balanced_validation_ecgs'] * 8,
                      annotations_used=False, labels_opened=False, test_opened=False)
    write_json(root / 'completed.json', completion)
    print(f"Preparation complete: train={len(selected['fit'])}, control={len(selected['control'])}; "
          f"Near/Cross-lead/Far={geometry(c)['counts']}; no annotation or NeuroKit2", flush=True)


class Prepared:
    def __init__(self, study, plan):
        self.root, self.plan, self.c = Path(study) / 'prepared', plan, plan['config']
        completion = read_json(self.root / 'completed.json')
        if completion['plan_id'] != plan['plan_id'] or completion['prepared_id'] != digest(completion['hashes']):
            raise ValueError('Wrong prepared-data identity')
        for name, expected in completion['hashes'].items():
            if file_sha(self.root / name) != expected:
                raise ValueError(f'Prepared probe data changed: {name}')
        self.selection = read_json(self.root / 'selection.json')
        self.x, self.frame, _ = arrays(plan, 'train')
        self.val, self.val_frame, _ = arrays(plan, 'validation')


def preview(study, plan):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    c = plan['config']
    g = geometry(c)
    groups, target = context_sets(c, 42)
    coded = np.zeros(g['tokens'], int)
    labels = ['Background', 'Near', 'Cross-lead', 'Far', 'Target']
    for code, key in enumerate(('background', *RATE_ORDER, 'target')):
        coded[groups[key]] = code
    conditions = [[.8, .2, .2], [.2, .8, .2], [.2, .2, .8]]
    conditions = [np.clip(r, c['rate_low'], c['rate_high']) for r in conditions]
    answers = matched_masks(c, conditions, 42)
    fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True)
    colors = ['#DDDDDD', '#009E73', '#0072B2', '#D55E00', '#222222']
    axes[0].imshow(coded.reshape(12, -1), aspect='auto', interpolation='nearest',
                   cmap=ListedColormap(colors), extent=[0, 10, 12, 0], vmin=0, vmax=4)
    axes[0].set_title('Context regions (fixed 400ms target; Near is 400ms on each side)')
    from matplotlib.patches import Patch
    axes[0].legend(handles=[Patch(color=col, label=name) for col, name in zip(colors, labels)],
                   loc='lower center', bbox_to_anchor=(.5, 1.18), ncol=5)
    for ax, rates, (mask, info) in zip(axes[1:], conditions, answers):
        ax.imshow(mask.reshape(12, -1), aspect='auto', interpolation='nearest',
                  cmap=ListedColormap(['white', '#333333']), extent=[0, 10, 12, 0], vmin=0, vmax=1)
        ax.set_title(f"Near/Cross-lead/Far={rates.tolist()}; white=visible, dark=hidden; "
                     f"background reveal={info['realized']['background']:.3f}")
    for ax in axes:
        ax.set_ylabel('Lead')
        ax.set_yticks(np.arange(12) + .5,
                      ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'], fontsize=7)
    axes[-1].set_xlabel('Time (seconds)')
    fig.tight_layout()
    fig.savefig(Path(study) / 'mask_preview.png', dpi=140)
    plt.close(fig)
    write_json(Path(study) / 'mask_preview.json', dict(target=target, geometry=g,
               masks=[info for _, info in answers], annotations_used=False))
