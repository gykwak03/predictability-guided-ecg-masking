"""Official ST-MEM with external masks, and the previously selected factor_010.

No upstream file is modified. Both feature pooling orders exclude SEP tokens.
The official ST-MEM decoder is lead-wise with SHARED decoder weights.
"""
from __future__ import annotations

from functools import lru_cache
import importlib.util

from experiments.phase1.config import ROOT
from experiments.phase1.model import build_mae


@lru_cache(maxsize=1)
def adapter_module():
    path = ROOT / 'archive/20261028_legacy/experiments/st_mem_adapter.py'
    spec = importlib.util.spec_from_file_location('_matched_stmem_adapter', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stmem_source(repo):
    return adapter_module().source_metadata(repo)['sha256']


def build_backbone(name, cfg, repo):
    if name == 'plain':
        return build_mae(cfg, cfg['patch_samples'])
    if name != 'stmem':
        raise ValueError(f'Unknown backbone: {name}')
    import torch

    class MatchedSTMEM(adapter_module().STMEMAdapter):
        def features(self, x, pooling='norm_then_mean'):
            # Same full-input computation as upstream forward_encoding, but
            # expose BOTH pooling orders for a matched evaluation protocol.
            encoder = self.core.encoder
            tokens = encoder.to_patch_embedding(x)
            b, c, q, d = tokens.shape
            tokens = tokens + encoder.pos_embedding[:, 1:q+1].unsqueeze(1)
            sep = encoder.sep_embedding[None, None, None, :].expand(b, c, 1, d)
            left = sep + encoder.pos_embedding[:, :1].unsqueeze(1)
            right = sep + encoder.pos_embedding[:, -1:].unsqueeze(1)
            tokens = torch.cat([left, tokens, right], dim=2)
            lead = torch.stack(list(encoder.lead_embeddings))[None, :, None, :]
            tokens = encoder.dropout((tokens + lead).flatten(1, 2))
            for i in range(encoder.depth):
                tokens = getattr(encoder, f'block{i}')(tokens)
            tokens = tokens.reshape(b, c, q+2, d)[:, :, 1:-1].flatten(1, 2)
            if pooling == 'norm_then_mean':
                return encoder.norm(tokens).mean(1)
            if pooling == 'mean_then_norm':
                return encoder.norm(tokens.mean(1))
            raise ValueError(f'Unknown pooling: {pooling}')

    return MatchedSTMEM(repo, patch_samples=cfg['patch_samples'],
                       model_size='smoke' if cfg['model_size'] == 'smoke' else 'small')
