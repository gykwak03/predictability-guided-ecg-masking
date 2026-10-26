"""Sehun Kim ECG-JEPA: official networks, explicit per-ECG lead-time masks.

The upstream checkout stays unmodified. CroPA, encoder/predictor blocks,
initialization and latent SmoothL1 objective come from that checkout. Only token
selection/reassembly is generalized to independent lead masks. Default 12 leads
is an adaptation; released encoder weights require 8 leads and P50.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import importlib.util
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SOURCE_COMMIT = "d937ad2c2c8a1e22856ce7e4a23a30f84a71217c"
SOURCE_FILES = ("ecg_jepa.py", "pos_encoding.py", "models.py", "pretrain_ECG_JEPA.py", "license")
LEADS8 = [0, 1, 6, 7, 8, 9, 10, 11]
_CACHE = {}


def load_upstream(repo):
    """Load model and actual timm utility leaves, without vision training imports.

    timm.__init__ imports torchvision even for time-series-only DropPath. Load
    its unmodified drop/grid/weight_init files under a private package instead;
    restore upstream's absolute import aliases immediately afterward.
    """
    repo = Path(repo).resolve()
    hashes = {f: hashlib.sha256((repo/f).read_bytes()).hexdigest() for f in SOURCE_FILES}
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    if commit != SOURCE_COMMIT or dirty:
        raise ValueError(f"Require clean pinned ECG_JEPA {SOURCE_COMMIT}; existing checkout left unchanged")
    distribution = importlib.metadata.distribution("timm")
    if distribution.version != "1.0.8":
        raise ValueError("Use timm==1.0.8, matching the official ECG-JEPA requirements")
    leaf_root = Path(distribution.locate_file("timm/layers"))
    key = (str(repo), tuple(hashes.items()), str(leaf_root))
    if key not in _CACHE:
        package_name = "_ecgjepa_timm_utilities"
        package = types.ModuleType(package_name)
        package.__path__ = [str(leaf_root)]
        sys.modules[package_name] = package
        for leaf in ("grid", "drop", "weight_init"):
            spec = importlib.util.spec_from_file_location(f"{package_name}.{leaf}", leaf_root/f"{leaf}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
        aliases = ("pos_encoding", "timm", "timm.models", "timm.models.layers")
        previous = {name: sys.modules.get(name) for name in aliases}
        try:
            for name in aliases[1:]:
                module = types.ModuleType(name)
                module.__path__ = []
                sys.modules[name] = module
            layer_module = sys.modules["timm.models.layers"]
            layer_module.DropPath = sys.modules[f"{package_name}.drop"].DropPath
            layer_module.trunc_normal_ = sys.modules[f"{package_name}.weight_init"].trunc_normal_
            spec = importlib.util.spec_from_file_location("pos_encoding", repo/"pos_encoding.py")
            positions = importlib.util.module_from_spec(spec)
            sys.modules["pos_encoding"] = positions
            spec.loader.exec_module(positions)
            spec = importlib.util.spec_from_file_location("_sehun_ecg_jepa", repo/"ecg_jepa.py")
            official = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(official)
        finally:
            for name, old in previous.items():
                if old is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = old
        _CACHE[key] = official, positions
    meta = dict(repository=str(repo), url="https://github.com/sehunfromdaegu/ECG_JEPA",
                commit=commit, sha256=hashes, timm_version=distribution.version,
                timm_utilities_sha256={f: hashlib.sha256((leaf_root/f).read_bytes()).hexdigest()
                                     for f in ("grid.py", "drop.py", "weight_init.py")})
    return *_CACHE[key], meta


def sample_masks(batch, leads, patches, ratio, seed, topology="lead-time"):
    """Exact ratio, equal hidden counts per lead; different masks per ECG/update.

    Equal counts constrain the mask family and enable the official single-lead
    predictor to run in a batch. This is not uniform sampling of all N choose K.
    """
    count = round(patches*ratio)
    if not 0 < count < patches or abs(count-patches*ratio) > 1e-8:
        raise ValueError("Mask ratio must hide an integer, nonzero, non-full count on every lead")
    if topology not in ("lead-time", "time-aligned"):
        raise ValueError("Unknown mask topology")
    rng = np.random.default_rng(seed)
    shape = (batch, leads if topology == "lead-time" else 1, patches)
    order = rng.random(shape).argsort(axis=-1)
    masks = np.zeros(shape, dtype=bool)
    np.put_along_axis(masks, order[..., :count], True, axis=-1)
    return np.broadcast_to(masks, (batch, leads, patches)).copy()


class ECGJEPAAdapter(nn.Module):
    def __init__(self, repo, patch_samples=50, leads=12, smoke=False):
        super().__init__()
        if patch_samples not in (25, 50, 125) or leads not in (8, 12):
            raise ValueError("Use P25/P50/P125 and 8 or 12 leads")
        official, positions, self.source = load_upstream(repo)
        self.lead_indices = LEADS8 if leads == 8 else list(range(12))
        self.p, self.c, self.n = patch_samples, leads, 2500//patch_samples
        self.architecture = dict(encoder_embed_dim=32 if smoke else 768,
            encoder_depth=1 if smoke else 12, encoder_num_heads=4 if smoke else 16,
            predictor_embed_dim=16 if smoke else 384, predictor_depth=1 if smoke else 6,
            predictor_num_heads=4 if smoke else 12, drop_path_rate=0. if smoke else .1)
        # Initialize the P50 reference first: all shared trainable tensors have
        # identical starting values across patch-length conditions at one seed.
        self.core = official.ecg_jepa(**self.architecture, c=leads, p=50, t=50,
            mask_type="random", mask_scale=(.5, .5), pos_type="sincos")
        self.dim = self.architecture["encoder_embed_dim"]
        self.core.p, self.core.t = self.n, self.p
        for encoder in (self.core.encoder, self.core.target_encoder):
            encoder.p, encoder.t = self.n, self.p
        self.core.predictor.p, self.core.predictor.t = self.n, self.p
        if self.p != 50:
            projection = nn.Linear(self.p, self.dim)
            self.core.encoder._init_weights(projection)
            self.core.encoder.W_P = projection
            self.core.target_encoder.W_P = copy.deepcopy(projection)
            pos = torch.from_numpy(positions.get_2d_sincos_pos_embed(self.dim, leads, self.n)).float()
            for encoder in (self.core.encoder, self.core.target_encoder):
                encoder.pos_embed = nn.Parameter(pos.clone(), requires_grad=False)
            pdim = self.architecture["predictor_embed_dim"]
            self.core.predictor.pos_embed = nn.Parameter(torch.from_numpy(
                positions.get_2d_sincos_pos_embed(pdim, 1, self.n)).float(), requires_grad=False)
        self.core.set_target_encoder()
        self.register_buffer("cross_attention", self.core.encoder._cross_attention_mask().bool(), persistent=False)

    def train(self, mode=True):
        super().train(mode)
        # Deterministic teacher targets: explicit adaptation from upstream
        # model.train(), which also activates teacher stochastic depth.
        self.core.target_encoder.eval()
        return self

    def _input(self, x):
        if x.ndim != 3 or x.shape[1:] != (12, 2500):
            raise ValueError("Input must use artifact lead order with shape (batch,12,2500)")
        return x[:, self.lead_indices].reshape(len(x), self.c*self.n, self.p)

    def _encode(self, encoder, patches, hidden=None):
        tokens = encoder.W_P(patches)
        pos = encoder.pos_embed.unsqueeze(0).expand(len(tokens), -1, -1)
        attention = self.cross_attention
        if hidden is not None:
            counts = hidden.sum(-1)
            if not torch.all(counts == counts.flatten()[0]) or not 0 < int(counts.flatten()[0]) < self.n:
                raise ValueError("Require the same nonzero hidden count per lead in every ECG")
            visible = ~hidden.flatten(1)
            size = int(visible[0].sum())
            indices = visible.nonzero(as_tuple=True)[1].reshape(len(tokens), size)
            tokens = tokens.gather(1, indices[..., None].expand(-1, -1, self.dim))
            pos = pos.gather(1, indices[..., None].expand(-1, -1, self.dim))
            attention = attention[indices[:, :, None], indices[:, None, :]][:, None]
        return encoder.norm(encoder.encoder_blocks(tokens, pos, attention))

    def predict_masked(self, x, hidden):
        if hidden.dtype != torch.bool or hidden.shape != (len(x), self.c, self.n):
            raise ValueError("hidden must be bool (batch,leads,time_patches)")
        patches = self._input(x)
        context = self._encode(self.core.encoder, patches, hidden)
        predictor = self.core.predictor
        embedded = predictor.predictor_embed(context).reshape(len(x)*self.c, -1,
                                                             predictor.predictor_embed.out_features)
        h = int(hidden[0, 0].sum())
        flat = hidden.reshape(len(x)*self.c, self.n)
        visible_indices = (~flat).nonzero(as_tuple=True)[1].reshape(len(flat), self.n-h)
        hidden_indices = flat.nonzero(as_tuple=True)[1].reshape(len(flat), h)
        indices = torch.cat((visible_indices, hidden_indices), dim=1)
        tokens = torch.cat((embedded, predictor.mask_token.expand(len(flat), h, -1)), dim=1)
        pos = predictor.pos_embed[indices]
        predicted = predictor.predictor_proj(predictor.predictor_norm(predictor.predictor_blocks(tokens, pos)))
        return predicted[:, -h:].reshape(len(x), self.c*h, self.dim)

    def ssl_loss(self, x, hidden):
        predicted = self.predict_masked(x, hidden)
        with torch.no_grad():
            target = self._encode(self.core.target_encoder, self._input(x))
            target = F.layer_norm(target.float(), (self.dim,))
            target = target[hidden.flatten(1)].reshape_as(predicted)
        return F.smooth_l1_loss(predicted.float(), target.float())

    def features(self, x):
        return self._encode(self.core.encoder, self._input(x)).mean(1)

    @torch.no_grad()
    def update_teacher(self, momentum):
        for student, teacher in zip(self.core.encoder.parameters(), self.core.target_encoder.parameters()):
            if student.requires_grad:
                teacher.mul_(momentum).add_(student.detach(), alpha=1-momentum)

    def load_released_encoder(self, checkpoint):
        if self.c != 8 or self.p != 50 or self.dim != 768:
            raise ValueError("Released checkpoint requires full-size 8-lead P50; use scratch for the patch/12-lead sweep")
        checkpoint = Path(checkpoint)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or "encoder" not in payload:
            raise ValueError("Expected official {'encoder': state_dict} checkpoint")
        self.core.encoder.load_state_dict(payload["encoder"], strict=True)
        self.core.target_encoder.load_state_dict(self.core.encoder.state_dict(), strict=True)
        return dict(path=str(checkpoint.resolve()), sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    epoch=payload.get("epoch"), predictor="random initialization; encoder-only checkpoint")

    def initial_backbone_digest(self):
        digest = hashlib.sha256()
        for name, value in self.core.encoder.state_dict().items():
            if name.startswith(("W_P.", "pos_embed")):
                continue
            digest.update(name.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
        return digest.hexdigest()
