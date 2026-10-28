"""Connect an unmodified, separately cloned ST-MEM to our ECG pilot interface.

Upstream: https://github.com/vuno/ST-MEM (Na et al., ICLR 2024).
The upstream checkout and its CC BY-NC 4.0 license remain separate. This adapter
changes mask injection only; it calls the official encoder, decoder and pooling.
No pretrained weights are downloaded or loaded automatically.
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
import types
from decimal import Decimal
from pathlib import Path

import numpy as np
import torch
from torch import nn

_CACHE = {}
SOURCE_FILES = ("models/encoder/vit.py", "models/encoder/st_mem_vit.py", "models/st_mem.py")


def source_metadata(repo):
    repo = Path(repo).resolve()
    missing = [name for name in (*SOURCE_FILES, "LICENSE") if not (repo / name).is_file()]
    if missing:
        raise FileNotFoundError(f"ST-MEM checkout missing {missing}. Pass --st-mem-repo /path/to/ST-MEM")
    return {"repository": str(repo), "upstream": "https://github.com/vuno/ST-MEM",
            "sha256": {name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
                       for name in (*SOURCE_FILES, "LICENSE")}}


def load_official_class(repo):
    """Load just three model files, not upstream training/data dependencies.

Upstream uses absolute `models.*` imports. Temporarily provide these aliases,
then restore all previous entries (also on failure); do not edit the checkout
or leave a generic models package shadowing the user's environment.
"""
    meta = source_metadata(repo)
    key = (meta["repository"], tuple(meta["sha256"].items()))
    if key in _CACHE:
        return _CACHE[key]
    if importlib.util.find_spec("einops") is None:
        raise ImportError("Install the model dependency: python -m pip install einops==0.6.0")
    aliases = ["models", "models.encoder", "models.encoder.vit",
               "models.encoder.st_mem_vit", "models.st_mem"]
    old = {name: sys.modules.get(name) for name in aliases}
    try:
        for name, subdir in [("models", "models"), ("models.encoder", "models/encoder")]:
            package = types.ModuleType(name)
            package.__path__ = [str(Path(repo).resolve() / subdir)]
            sys.modules[name] = package
        for file in SOURCE_FILES:
            name = file[:-3].replace("/", ".")
            spec = importlib.util.spec_from_file_location(name, Path(repo).resolve() / file)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        official = sys.modules["models.st_mem"].ST_MEM
    finally:
        for name, module in old.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    _CACHE[key] = official
    return official


def mask_layout(n_tokens, ratio, num_leads=12):
    if not 0 < ratio < 1 or n_tokens % num_leads:
        raise ValueError("Require 0 < mask ratio < 1 and a complete lead-time grid")
    q = n_tokens // num_leads
    # Floor the intended decimal count, without binary float underflow at
    # integer boundaries (e.g. 10 * (1 - 0.90) must retain one patch).
    # Fractional counts still round down: 50 * (1 - 0.75) retains 12.
    keep = int(Decimal(q) * (Decimal(1) - Decimal(str(ratio))))
    if not 0 < keep < q:
        raise ValueError("Each lead must retain at least one visible and one hidden patch")
    return q, keep


def random_lead_mask(batch, n_tokens, ratio, seed, num_leads=12):
    q, keep = mask_layout(n_tokens, ratio, num_leads)
    order = np.argsort(np.random.default_rng(seed).random((batch, num_leads, q)), axis=-1)
    mask = np.ones((batch, num_leads, q), dtype=bool)
    np.put_along_axis(mask, order[..., :keep], False, axis=-1)
    return mask.reshape(batch, n_tokens)


def fixed_lead_mask(ecg_ids, n_tokens, ratio, seed, repeat=0):
    seeds = np.asarray([seed + 100000 + repeat * 1000000 + int(i) for i in ecg_ids], dtype=np.int64)
    masks = [random_lead_mask(1, n_tokens, ratio, int(s))[0] for s in seeds]
    return np.stack(masks), seeds


class STMEMAdapter(nn.Module):
    """(B,12,2500), boolean hidden mask -> flat predictions and targets.

Per-lead visible counts must be equal, as required by official ST-MEM's
decoder restore operation. Arbitrary global masks are explicitly rejected.
"""
    def __init__(self, repo, patch_samples=50, model_size="small", seq_len=2500):
        super().__init__()
        if seq_len < 1 or patch_samples < 1 or seq_len % patch_samples:
            raise ValueError("Sequence length must be divisible by patch length")
        sizes = {
            "smoke": dict(embed_dim=32, depth=1, num_heads=2, decoder_embed_dim=32,
                          decoder_depth=1, decoder_num_heads=2),
            "small": dict(embed_dim=384, depth=12, num_heads=6, decoder_embed_dim=256,
                          decoder_depth=4, decoder_num_heads=4),
            "base": dict(embed_dim=768, depth=12, num_heads=12, decoder_embed_dim=256,
                         decoder_depth=4, decoder_num_heads=4),
        }
        if model_size not in sizes:
            raise ValueError("Model size must be smoke, small or base")
        official = load_official_class(repo)

        class ExternalMaskSTMEM(official):
            _external_mask = None

            def random_masking(self, x, mask_ratio):
                if self._external_mask is None:
                    return super().random_masking(x, mask_ratio)
                b, c, q, width = x.shape
                mask = self._external_mask.reshape(b, c, q)
                # Visible-first stable ordering; positional embeddings already
                # identify time. Transformer attention is permutation equivariant.
                shuffle = torch.argsort(mask.to(torch.int64), dim=-1, stable=True)
                restore = torch.argsort(shuffle, dim=-1)
                keep = int((~mask[0, 0]).sum())
                visible = torch.gather(x, 2, shuffle[..., :keep, None].expand(-1, -1, -1, width))
                return visible, mask.to(x.dtype), restore

        self.core = ExternalMaskSTMEM(seq_len=seq_len, patch_size=patch_samples,
                                     num_leads=12, norm_pix_loss=False,
                                     norm_layer=lambda width: nn.LayerNorm(width, eps=1e-6),
                                     **sizes[model_size])
        self.p, self.n, self.seq_len = patch_samples, 12 * seq_len // patch_samples, seq_len
        self.ratio = 0.75
        self.model_size, self.architecture = model_size, sizes[model_size]

    def forward(self, x, mask):
        if x.ndim != 3 or len(x) == 0 or x.shape[1:] != (12, self.seq_len) or mask.shape != (len(x), self.n):
            raise ValueError("Expected ECG (B,12,L) and lead-major hidden mask (B,12*L/P)")
        if mask.dtype != torch.bool:
            raise ValueError("Mask must be boolean: True means hidden")
        counts = (~mask.reshape(len(x), 12, -1)).sum(dim=-1)
        if not bool((counts == counts[0, 0]).all()) or not 0 < int(counts[0, 0]) < self.n // 12:
            raise ValueError("ST-MEM requires equal nonzero visible counts for every lead/sample")
        self.core._external_mask = mask
        try:
            # Positive ratio enables the official masking branch; the supplied
            # mask controls the actual ratio. Use official encoder/decoder unchanged.
            latent, _, restore = self.core.forward_encoder(x, mask_ratio=self.ratio)
            pred = self.core.forward_decoder(latent, restore)
        finally:
            self.core._external_mask = None
        return pred.reshape(len(x), self.n, self.p), self.core.patchify(x).reshape(len(x), self.n, self.p)

    def features(self, x):
        # Official encoder: remove SEP, average lead-time tokens, then final norm.
        return self.core.encoder.forward_encoding(x)
