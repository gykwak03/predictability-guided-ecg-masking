"""Plain Transformer blocks with the same lead-wise interface as ST-MEM."""
from __future__ import annotations

import torch
from torch import nn

from experiments.phase1.model import legacy_module

legacy = legacy_module()
SIZES = {
    'smoke': (32, 1, 2, 32, 1, 2),
    'small': (384, 12, 6, 256, 4, 4),
    'base': (768, 12, 12, 256, 4, 4),
}
# (Total, trainable), including the fixed time positions in the total.
PARAMETERS = {'small': (24_624_278, 24_590_998), 'base': (88_529_558, 88_476_310)}


class PlainLeadwiseMAE(legacy.CauseMAE):
    def __init__(self, size, patch=50):
        e, depth, heads, d, dec_depth, dec_heads = SIZES[size]
        # Same Plain blocks/initialization as factor_010, without its preset size limit.
        legacy.notebook_class().__init__(self, patch, dict(
            enc_dim=e, enc_depth=depth, enc_heads=heads, dec_dim=d,
            dec_depth=dec_depth, dec_heads=dec_heads, initialization='independent'))
        self.architecture = dict(embed_dim=e, depth=depth, num_heads=heads,
                                 decoder_embed_dim=d, decoder_depth=dec_depth, decoder_num_heads=dec_heads)
        self.model_size = size
        self.spec = {'leadwise_decoder': True}
        self.patch_pre, self.patch_post = nn.LayerNorm(patch), nn.LayerNorm(e)
        self.sep = nn.Parameter(torch.randn(1, 1, 1, e) * .02)
        # Lead identity reaches the decoder via encoded patches and SEP; no extra
        # decoder lead embedding. The fixed tables include both SEP positions.
        del self.dec_lead
        self.enc_time = self._positions(e)
        self.dec_time = self._positions(d)

    def _positions(self, width):
        time = legacy.sinusoidal(self.q, width)
        return nn.Parameter(nn.functional.pad(time, (0, 0, 1, 1))[None, None], requires_grad=False)

    def embedded_patches(self, x):
        patches = x.reshape(len(x), self.n, self.p)
        tokens = self.patch_post(self.patch_embed(self.patch_pre(patches)))
        pos = self.enc_lead + self.enc_time[:, :, 1:-1]
        return patches, tokens + pos.reshape(1, self.n, -1)

    def _layout(self, x, mask):
        if x.ndim != 3 or not len(x) or x.shape[1:] != (12, self.q * self.p):
            raise ValueError('Expected ECG (B,12,L)')
        if mask.dtype != torch.bool or mask.shape != (len(x), self.n):
            raise ValueError('Expected a boolean lead-major hidden mask (B,12*Q)')
        mask = mask.reshape(len(x), 12, self.q)
        counts = (~mask).sum(-1)
        if not bool((counts == counts[0, 0]).all()) or not 0 < int(counts[0, 0]) < self.q:
            raise ValueError('Expected equal nonzero visible counts per lead and ECG')
        patches, tokens = self.embedded_patches(x)
        sep = (self.sep + self.enc_lead).expand(len(x), -1, -1, -1)
        tokens = torch.cat((sep + self.enc_time[:, :, :1],
                            tokens.reshape(len(x), 12, self.q, -1),
                            sep + self.enc_time[:, :, -1:]), dim=2)
        # CauseMAE gathers visible tokens across all leads for the encoder and
        # folds leads into the batch axis only for the shared decoder.
        return (patches, tokens.flatten(1, 2),
                nn.functional.pad(mask, (1, 1), value=False).flatten(1, 2),
                self.dec_time.expand(1, 12, -1, -1).flatten(1, 2))


def count_parameters(model):
    return (sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad))


def build_model(name, cfg):
    family, size = name.split('_')
    if family == 'plain':
        model = PlainLeadwiseMAE('smoke' if cfg['synthetic'] else size, cfg['patch_samples'])
    elif name == 'stmem_base':
        from experiments.stmem_base_baseline import build_model as base
        model = base(cfg)
    elif name == 'stmem_small':
        from experiments.backbone_models import build_backbone
        model = build_backbone('stmem', cfg, cfg['st_mem_repo'])
    else:
        raise ValueError(f'Unknown model: {name}')
    if not cfg['synthetic'] and count_parameters(model) != PARAMETERS[size]:
        raise ValueError(f'{name} parameter contract: expected {PARAMETERS[size]}, got {count_parameters(model)}')
    return model
