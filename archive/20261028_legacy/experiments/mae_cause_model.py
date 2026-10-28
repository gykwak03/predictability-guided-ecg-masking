"""Controlled interventions on the actual notebook MAE, without editing old runs.

The unchanged notebook class supplies all original weights/initialization. The
all-off intervention is numerically identical to that class, not a new baseline.
"""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
SIZES = {
    "baseline": dict(enc_dim=192, enc_depth=6, enc_heads=6,
                     dec_dim=128, dec_depth=2, dec_heads=4),
    "stmem-small": dict(enc_dim=384, enc_depth=12, enc_heads=6,
                        dec_dim=256, dec_depth=4, dec_heads=4),
    "smoke": dict(enc_dim=32, enc_depth=2, enc_heads=4,
                  dec_dim=32, dec_depth=2, dec_heads=4),
}


@lru_cache(maxsize=1)
def notebook_class():
    book = json.loads((HERE / "Masking_pilot.ipynb").read_text(encoding="utf-8"))
    source = next("".join(c["source"]) for c in book["cells"]
                  if c["cell_type"] == "code" and "class PlainECGMAE" in "".join(c["source"]))
    ns = dict(TRAINING_AVAILABLE=True, torch=torch, nn=nn, np=np,
              random=random, N_LEADS=12, SIGNAL_LENGTH=2500)
    exec(compile(source, "Masking_pilot.ipynb:model", "exec"), ns)
    return ns["PlainECGMAE"]


def sinusoidal(length, width):
    if width % 2:
        raise ValueError("Sinusoidal embedding needs even width")
    phase = torch.arange(length, dtype=torch.float32)[:, None] * (
        1 / 10000 ** (torch.arange(width // 2, dtype=torch.float32) / (width / 2)))[None]
    return torch.cat([phase.sin(), phase.cos()], dim=1)


class CauseMAE(notebook_class()):
    def __init__(self, spec, patch=50):
        cfg = dict(SIZES[spec["size"]], initialization=spec["initialization"])
        super().__init__(patch, cfg)
        self.spec = dict(spec)
        e, d = cfg["enc_dim"], cfg["dec_dim"]
        self.patch_pre = nn.LayerNorm(patch) if spec["patch_norm"] else nn.Identity()
        self.patch_post = nn.LayerNorm(e) if spec["patch_norm"] else nn.Identity()
        if spec["fixed_time"]:
            with torch.no_grad():
                self.enc_time.copy_(sinusoidal(self.q, e)[None, None])
                self.dec_time.copy_(sinusoidal(self.q, d)[None, None])
            self.enc_time.requires_grad_(False)
            self.dec_time.requires_grad_(False)
        self.sep = None
        if spec["sep"]:
            # A local generator keeps all existing weights/RNG streams unchanged.
            g = torch.Generator().manual_seed(int(spec.get("model_seed", 0)) + 991)
            self.sep = nn.Parameter(torch.randn(1, 1, 1, e, generator=g) * .02)

    def embedded_patches(self, x):
        patches = x.reshape(len(x), self.n, self.p)
        tokens = self.patch_post(self.patch_embed(self.patch_pre(patches)))
        return patches, tokens + (self.enc_lead + self.enc_time).reshape(1, self.n, -1)

    def _layout(self, x, mask):
        patches, tokens = self.embedded_patches(x)
        if mask.dtype != torch.bool or mask.shape != (len(x), self.n):
            raise ValueError("Expected boolean hidden mask (B,12*Q)")
        if not ((~mask).sum(1) == (~mask[0]).sum()).all() or mask.all() or not mask.any():
            raise ValueError("Equal nonzero visible counts per ECG are required")
        dec_pos = (self.dec_lead + self.dec_time).reshape(1, self.n, -1)
        if self.sep is not None:
            tokens = tokens.reshape(len(x), 12, self.q, -1)
            sep = (self.sep + self.enc_lead).expand(len(x), -1, -1, -1)
            tokens = torch.cat([sep, tokens, sep], dim=2).flatten(1, 2)
            z = torch.zeros(len(x), 12, 1, dtype=torch.bool, device=x.device)
            mask = torch.cat([z, mask.reshape(len(x), 12, self.q), z], 2).flatten(1, 2)
            # Encoded SEP already carries lead context. Zero time position, as in ST-MEM.
            pos = dec_pos.reshape(1, 12, self.q, -1)
            zpos = torch.zeros_like(pos[:, :, :1])
            dec_pos = torch.cat([zpos, pos, zpos], 2).flatten(1, 2)
        return patches, tokens, mask, dec_pos

    def forward(self, x, mask):
        patches, tokens, augmented_mask, dec_pos = self._layout(x, mask)
        b, n, width = tokens.shape
        ids = torch.arange(n, device=x.device).expand(b, -1)
        visible_ids = ids[~augmented_mask].reshape(b, -1)
        visible = tokens.gather(1, visible_ids[..., None].expand(-1, -1, width))
        encoded = self.encoder(visible)
        projected = self.to_decoder(encoded)
        full = self.mask_token.expand(b, n, -1).clone()
        full.scatter_(1, visible_ids[..., None].expand(-1, -1, full.shape[-1]), projected)
        full = full + dec_pos
        if self.spec["leadwise_decoder"]:
            full = full.reshape(b * 12, n // 12, -1)
        prediction = self.reconstruction(self.decoder(full))
        prediction = prediction.reshape(b, 12, n // 12, self.p)
        if self.sep is not None:
            prediction = prediction[:, :, 1:-1]
        return prediction.reshape(b, self.n, self.p), patches

    def features(self, x, pooling="norm_then_mean", visible_mask=None):
        _, tokens = self.embedded_patches(x)
        keep = torch.ones(tokens.shape[:2], dtype=torch.bool, device=x.device)
        if visible_mask is not None:
            keep = ~visible_mask
        is_patch = torch.ones_like(keep)
        if self.sep is not None:
            sep = (self.sep + self.enc_lead).expand(len(x), -1, -1, -1)
            tokens = torch.cat([sep, tokens.reshape(len(x), 12, self.q, -1), sep], 2).flatten(1, 2)
            ones = torch.ones(len(x), 12, 1, dtype=torch.bool, device=x.device)
            keep = torch.cat([ones, keep.reshape(len(x), 12, self.q), ones], 2).flatten(1, 2)
            is_patch = torch.cat([~ones, is_patch.reshape(len(x), 12, self.q), ~ones], 2).flatten(1, 2)
        if not (keep.sum(1) == keep[0].sum()).all():
            raise ValueError("Feature mask counts must match across ECGs")
        patch_flags = is_patch[keep].reshape(len(x), -1)
        tokens = tokens[keep].reshape(len(x), -1, tokens.shape[-1])
        for block in self.encoder.layers:
            tokens = block(tokens)
        if pooling == "norm_then_mean":
            tokens = self.encoder.norm(tokens)
            return (tokens * patch_flags[..., None]).sum(1) / patch_flags.sum(1, keepdim=True)
        if pooling == "mean_then_norm":
            pooled = (tokens * patch_flags[..., None]).sum(1) / patch_flags.sum(1, keepdim=True)
            return self.encoder.norm(pooled)
        raise ValueError("Unknown pooling")


def mask_batch(batch, n_tokens, ratio, seed, policy, leads=12):
    q = n_tokens // leads
    hidden = round(n_tokens * ratio)
    if n_tokens % leads or not 0 < hidden < n_tokens:
        raise ValueError("Invalid lead-time mask")
    rng = np.random.default_rng(seed)
    if policy == "global":
        order = np.argsort(rng.random((batch, n_tokens)), axis=1)
        mask = np.zeros((batch, n_tokens), dtype=bool)
        np.put_along_axis(mask, order[:, :hidden], True, axis=1)
    elif policy == "per_lead":
        # No floor-induced ratio change: reject non-comparable counts explicitly.
        if hidden % leads:
            raise ValueError("Exact per-lead ratio impossible for this patch length; use 50% or 76% at P50")
        order = np.argsort(rng.random((batch, leads, q)), axis=2)
        mask = np.zeros((batch, leads, q), dtype=bool)
        np.put_along_axis(mask, order[:, :, :hidden // leads], True, axis=2)
        mask = mask.reshape(batch, n_tokens)
    else:
        raise ValueError("Unknown mask policy")
    return mask


def learning_rate(update, *, horizon, warmup=100, base=3e-4):
    """Clamp after horizon: extension preserves the complete original LR prefix."""
    import math
    index = update - 1  # Match LambdaLR's initial step=0 in the historical notebook.
    if index < warmup:
        return base * (index + 1) / warmup
    fraction = min(1., (index - warmup) / max(1, horizon - warmup))
    return base * (.1 + .9 * .5 * (1 + math.cos(math.pi * fraction)))
