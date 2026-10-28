"""Lead-major tokens. True = hidden; token = lead * Q + time_patch."""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json

import numpy as np
import pandas as pd

FS, LENGTH, LEADS = 250, 2500, 12


def geometry(patch, ratio):
    if not isinstance(patch, int) or patch < 1 or LENGTH % patch:
        raise ValueError('patch_samples must divide 2500')
    q = LENGTH // patch
    count = Fraction(str(ratio)) * q
    if count.denominator != 1 or not 0 < count < q:
        raise ValueError(f'Exact per-lead ratio {ratio} impossible at P{patch} ({q} patches/lead)')
    return q, int(count)


def patch_location(index, patch):
    q = LENGTH // patch
    if not 0 <= index < LEADS*q:
        raise ValueError('Token index out of range')
    lead, time = divmod(index, q)
    return dict(lead_index=lead, time_patch=time, start_sample=time*patch,
                stop_sample=(time+1)*patch, start_seconds=time*patch/FS, stop_seconds=(time+1)*patch/FS)


def patchify(x, patch):
    if tuple(x.shape[1:]) != (LEADS, LENGTH) or LENGTH % patch:
        raise ValueError('Expected (B,12,2500) and a divisible patch size')
    return x.reshape(len(x), LEADS*LENGTH//patch, patch)


def unpatchify(x):
    if x.ndim != 3 or x.shape[1]*x.shape[2] != LEADS*LENGTH:
        raise ValueError('Invalid patch tensor')
    return x.reshape(len(x), LEADS, LENGTH)


def seed_for(*parts):
    return int.from_bytes(hashlib.sha256(':'.join(map(str, parts)).encode()).digest()[:8], 'little') & ((1 << 63)-1)


def mask_digest(mask):
    return hashlib.sha256(np.packbits(mask).tobytes()).hexdigest()


def per_lead_masks(batch, patch, ratio, seed):
    q, count = geometry(patch, ratio)
    rng = np.random.default_rng(seed)
    order = np.argsort(rng.random((batch, LEADS, q)), axis=-1)
    masks = np.zeros((batch, LEADS, q), dtype=bool)
    np.put_along_axis(masks, order[..., :count], True, axis=-1)
    return masks.reshape(batch, LEADS*q)


def make_bank(frame, indices, patch, ratio, repeats, seed, pool, *, pairs=False):
    q, count = geometry(patch, ratio)
    if repeats < 1 or (pairs and ratio != 0.5):
        raise ValueError('Positive repeats required; complement pairs require exactly 50%')
    rows, masks = [], []
    for index in indices:
        row = frame.iloc[int(index)]
        for repeat in range(repeats):
            pair_id = f'{pool}-e{int(row.ecg_id)}-p{patch}-r{ratio:g}-s{seed}-m{repeat}'
            mask_seed = seed_for(seed, pair_id)
            original = per_lead_masks(1, patch, ratio, mask_seed)[0]
            for member in range(2 if pairs else 1):
                mask = ~original if member else original
                rows.append(dict(ecg_id=int(row.ecg_id), patient_id=row.patient_id,
                    split=str(row.split), pool=pool, artifact_index=int(index),
                    mask_id=f'{pair_id}-c{member}', seed=seed, mask_seed=mask_seed, repeat=repeat,
                    patch_samples=patch, patch_seconds=patch/FS, length=LENGTH,
                    requested_mask_ratio=ratio, realized_mask_ratio=count/q,
                    mask_policy='per_lead_exact_independent', lead_hidden_count=count,
                    lead_hidden_counts=json.dumps([count]*LEADS), hidden_tokens=LEADS*count,
                    mask_sha256=mask_digest(mask), pair_id=pair_id if pairs else '',
                    pair_member=member, paired_complement=pairs))
                masks.append(mask)
    if not rows:
        raise ValueError(f'No evaluable masks in {pool}; check ECG selection')
    return pd.DataFrame(rows), np.stack(masks)


def replay_bank(bank):
    if bank.empty or bank.mask_id.duplicated().any():
        raise ValueError('Nonempty unique mask IDs required')
    masks = []
    for row in bank.itertuples(index=False):
        q, count = geometry(int(row.patch_samples), row.requested_mask_ratio)
        if row.mask_policy != 'per_lead_exact_independent' or row.length != LENGTH:
            raise ValueError('Unsupported mask geometry/policy')
        mask = per_lead_masks(1, int(row.patch_samples), row.requested_mask_ratio, int(row.mask_seed))[0]
        if int(row.pair_member) == 1:
            if row.requested_mask_ratio != 0.5:
                raise ValueError('Invalid complement')
            mask = ~mask
        if (mask_digest(mask) != row.mask_sha256 or row.lead_hidden_count != count or
                row.hidden_tokens != LEADS*count or row.realized_mask_ratio != count/q or
                row.patch_seconds != row.patch_samples/FS or json.loads(row.lead_hidden_counts) != [count]*LEADS):
            raise ValueError(f'Mask digest/metadata mismatch: {row.mask_id}')
        masks.append(mask)
    return np.stack(masks)


def batch_indices(size, batch, update, seed):
    """Stateless shuffled epochs, identical across architectures and patch sizes."""
    if size < batch or update < 1:
        raise ValueError('Effective batch must fit the selected pool and update must be positive')
    positions = np.arange((update-1)*batch, update*batch)
    result = np.empty(batch, dtype=np.int64)
    for epoch in np.unique(positions // size):
        use = positions // size == epoch
        result[use] = np.random.default_rng(seed_for(seed, 'order', int(epoch))).permutation(size)[positions[use] % size]
    return result
