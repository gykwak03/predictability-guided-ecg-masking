from __future__ import annotations

import numpy as np
import torch

from experiments.wcb.evaluation import linear_probe, final_test


@torch.no_grad()
def evaluate_bank(model, signals, path, device, batch):
    """Reward is ALL-hidden MSE. Anchor-only MSE is a separate diagnostic."""
    losses, target_losses = [], []
    model.eval()
    with np.load(path, allow_pickle=False) as bank:
        for start in range(0, len(bank['indices']), batch):
            ids = bank['indices'][start:start + batch]
            x = torch.tensor(np.asarray(signals[ids]), device=device)
            mask = torch.tensor(bank['masks'][start:start + batch], device=device)
            prediction, target = model(x, mask)
            errors = (prediction - target).square().mean(-1)
            losses.extend(((errors * mask).sum(-1) / mask.sum(-1)).cpu().tolist())
            if 'anchors' in bank:
                anchor = torch.tensor(bank['anchors'][start:start + batch], device=device)
                if not bool(mask[anchor].all()):
                    raise ValueError('Probe target must be hidden')
                target_losses.extend(((errors * anchor).sum(-1) / anchor.sum(-1)).cpu().tolist())
        scores = np.asarray(losses)
        if not len(scores) or not np.isfinite(scores).all():
            raise ValueError('Empty or nonfinite probe')
        result = dict(loss=float(scores.mean()), cases=len(scores), loss_scope='all_hidden_patches')
        if 'strata' in bank:
            keys = np.unique(bank['strata'])
            strata = {str(k): float(scores[bank['strata'] == k].mean()) for k in keys}
            result.update(loss=float(np.mean(list(strata.values()))), strata=strata)
            ts = np.asarray(target_losses)
            result.update(target_loss=float(ts.mean()), target_strata={str(k): float(ts[bank['strata'] == k].mean()) for k in keys})
    return result
