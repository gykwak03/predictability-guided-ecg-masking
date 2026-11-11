from __future__ import annotations

import numpy as np

from .common import RATE_ORDER, geometry, seed_for


def rates_from_logits(x, c):
    x = np.asarray(x, float)
    return c['rate_low'] + (c['rate_high'] - c['rate_low']) / (1 + np.exp(-np.clip(x, -60, 60)))


def logits_from_rates(r, c):
    value = (np.asarray(r, float) - c['rate_low']) / (c['rate_high'] - c['rate_low'])
    value = np.clip(value, 1e-10, 1 - 1e-10)
    return np.log(value / (1 - value))


def limited_update(old, proposed, c):
    before = rates_from_logits(old, c)
    after = rates_from_logits(proposed, c)
    return logits_from_rates(np.clip(after, before - c['max_rate_change'], before + c['max_rate_change']), c)


def context_sets(c, seed):
    g = geometry(c)
    q, m, flank = g['patches_per_lead'], g['target_patches'], g['flank_patches']
    rng = np.random.default_rng(seed_for(seed, 'target'))
    selected = np.sort(rng.choice(12, c['target_leads'], replace=False))
    start = int(rng.integers(flank, q - m - flank + 1))
    local = np.isin(np.arange(12), selected)[:, None]
    t = np.arange(q)[None, :]
    target_time = (t >= start) & (t < start + m)
    near_time = ((t >= start - flank) & (t < start)) | ((t >= start + m) & (t < start + m + flank))
    groups = dict(target=np.flatnonzero(local & target_time),
                  near=np.flatnonzero(local & near_time),
                  cross=np.flatnonzero(~local & target_time),
                  far=np.flatnonzero(local & ~(target_time | near_time)),
                  background=np.flatnonzero(~local & ~target_time))
    return groups, dict(leads=selected.tolist(), start_patch=start, end_patch=start + m)


def global_mask(n, rng):
    mask = np.ones(n, bool)
    mask[rng.permutation(n)[:n // 2]] = False
    return mask


def describe(mask, groups, rates, mode, target):
    counts = {key: len(ids) for key, ids in groups.items()}
    visible = {key: int((~mask[ids]).sum()) for key, ids in groups.items()}
    return dict(mode=mode, target=target, requested=None if rates is None else dict(zip(RATE_ORDER, map(float, rates))),
                realized={key: visible[key] / counts[key] for key in (*RATE_ORDER, 'background')},
                candidates=counts, visible=visible, target_hidden=int(mask[groups['target']].sum()))


def matched_masks(c, rates, seed, kind='structured'):
    """True=hidden. Shared targets, priorities and rounding for paired policies."""
    g = geometry(c)
    groups, target = context_sets(c, seed)
    rng = np.random.default_rng(seed_for(seed, 'selection'))
    priority = rng.random(g['tokens'])
    rounding = rng.random(3)
    ordered = {key: ids[np.argsort(priority[ids], kind='stable')] for key, ids in groups.items()}
    answers = []
    for branch, requested in enumerate(rates):
        mask = np.ones(g['tokens'], bool)
        if kind == 'global':
            mask[np.argsort(priority, kind='stable')[:g['visible']]] = False
        elif kind in ('conditional', 'conditional_best2'):
            eligible = np.concatenate([ids for key, ids in groups.items() if key != 'target'])
            if kind == 'conditional_best2':
                branch_rng = np.random.default_rng(seed_for(seed, 'selection', branch))
                ids = branch_rng.permutation(eligible)
            else:
                ids = eligible[np.argsort(priority[eligible], kind='stable')]
            mask[ids[:g['visible']]] = False
        elif kind == 'structured':
            requested = np.asarray(requested, float)
            if requested.shape != (3,) or not np.isfinite(requested).all() or np.any(requested < c['rate_low']) or np.any(requested > c['rate_high']):
                raise ValueError('Expected independent Near/Cross-lead/Far rates inside configured bounds')
            total = 0
            for j, key in enumerate(RATE_ORDER):
                value = len(ordered[key]) * requested[j]
                count = int(np.floor(value) + (rounding[j] < value - np.floor(value)))
                mask[ordered[key][:count]] = False
                total += count
            remaining = g['visible'] - total
            if not 0 <= remaining <= len(ordered['background']):
                raise ValueError('Infeasible mask request; no silent rate projection or fallback')
            mask[ordered['background'][:remaining]] = False
        else:
            raise ValueError('Unknown mask family')
        if (~mask).sum() != g['visible'] or (kind != 'global' and not mask[groups['target']].all()):
            raise AssertionError('Visible budget or target mask invariant failed')
        actual_rates = requested if kind == 'structured' else None
        answers.append((mask, describe(mask, groups, actual_rates, kind, target)))
    return answers
