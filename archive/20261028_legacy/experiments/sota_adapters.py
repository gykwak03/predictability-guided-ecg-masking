"""Thin adapters around separately cloned upstream model implementations.

This is a PTB-XL resource-constrained comparison, not a reproduction of the
papers' large-data pretrained checkpoints. Never silently substitute old CPC
or a generic Transformer for one of these upstream models.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import torch
from torch import nn

from st_mem_adapter import STMEMAdapter


def reference_cauchy(v, z, w):
    """Exact complex conjugate-pair sum; slower fallback without CUDA extensions."""
    return ((v[..., None]/(z-w[..., None])) +
            (v.conj()[..., None]/(z-w.conj()[..., None]))).sum(dim=-2)


def checkout_info(repo, files):
    repo = Path(repo).resolve()
    for name in files:
        if not (repo / name).is_file():
            raise FileNotFoundError(f"Missing {repo / name}; clone the official repository first")
    result = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True)
    return dict(path=str(repo), commit=result.stdout.strip() if result.returncode == 0 else None,
                tracked_changes=dirty.stdout.strip() if dirty.returncode == 0 else "unknown",
                sha256={name: hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in files})


def patch_class(repo):
    """Import just the official ViT and enums; restore generic src aliases."""
    aliases = {"src": "src", "src.core": "src/core", "src.core.constants": "src/core/constants"}
    names = [*aliases, "src.core.constants.definitions", "src.core.models.vit_1d"]
    old = {name: sys.modules.get(name) for name in names}
    try:
        for name, folder in aliases.items():
            module = types.ModuleType(name)
            module.__path__ = [str(Path(repo)/folder)]
            sys.modules[name] = module
        for name, relative in [("src.core.constants.definitions", "src/core/constants/definitions.py"),
                               ("src.core.models.vit_1d", "src/core/models/vit_1d.py")]:
            spec = importlib.util.spec_from_file_location(name, Path(repo)/relative)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        return sys.modules[names[-1]].ViT, sys.modules["src.core.constants.definitions"]
    finally:
        for name, previous in old.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


class PatchECGAdapter(nn.Module):
    def __init__(self, repo, patch_samples=50, smoke=False):
        super().__init__()
        self.source = checkout_info(repo, ["src/core/models/vit_1d.py", "src/core/constants/definitions.py",
            "src/scripts/multi-gpu/pretrain/transformer_comparison/small_patch_ecg.sh"])
        ViT, enums = patch_class(repo)
        # Official small_patch_ecg.sh: d_model=128, depth=6, heads=8, MLP=512.
        self.architecture = dict(dim=32 if smoke else 128, depth=1 if smoke else 6,
                                 heads=2 if smoke else 8, mlp_dim=64 if smoke else 512,
                                 dim_head=64, dropout=0., emb_dropout=0.)
        self.core = ViT(seq_len=12*2500, patch_size=patch_samples, stride=patch_samples,
                        num_classes=patch_samples, channels=1, mode=enums.Mode.PRETRAIN,
                        **self.architecture)
        self.p = patch_samples
        self.dim = self.architecture["dim"]
        self.augmentation = enums.DataAugmentation.none

    def forward(self, x, hidden):
        target = x.reshape(len(x), -1, self.p)
        visible = (~hidden).to(x.dtype)
        # Upstream's mask=True means visible, opposite to our common interface.
        pred = self.core(target, visible, hidden.to(x.dtype), None, self.augmentation)
        return pred, target

    def features(self, x):
        target = x.reshape(len(x), -1, self.p)
        visible = x.new_ones(target.shape[:2])
        encoded = self.core.all_leads(target, visible, 1-visible)
        return self.core.mlp_head.layer_norm(encoded[:, 0])


class STMEMBenchmarkAdapter(STMEMAdapter):
    def __init__(self, repo, patch_samples=50, smoke=False):
        super().__init__(repo, patch_samples, "smoke" if smoke else "small")
        self.source = checkout_info(repo, ["models/st_mem.py", "models/encoder/st_mem_vit.py",
                                           "models/encoder/vit.py"])


class ECGCPCAdapter(nn.Module):
    """Official CNN + causal S4 + prediction head and official CPC objective.

Uses a common 250 Hz input (native paper: 240 Hz), and mean CPC loss over
prediction positions (upstream returns their sum). These adaptations are
recorded. CPC has no waveform mask ratio or reconstruction loss.
"""
    def __init__(self, repo, smoke=False):
        super().__init__()
        files = ["code/clinical_ts/ts/encoder.py", "code/clinical_ts/ts/s4.py",
                 "code/clinical_ts/ts/head.py", "code/clinical_ts/loss/selfsupervised.py",
                 "code/clinical_ts/ts/s4_modules/s42.py", "code/conf/config_cpc_ecg_s4_heedb.yaml"]
        self.source = checkout_info(repo, files)
        code = str(Path(repo).resolve()/"code")
        current = sys.modules.get("clinical_ts")
        paths = [] if current is None else list(getattr(current, "__path__", []))
        if current is not None and str(Path(code)/"clinical_ts") not in [str(Path(p).resolve()) for p in paths]:
            raise RuntimeError("A different clinical_ts is already loaded; run each model in a fresh process")
        sys.path.insert(0, code)
        try:
            from clinical_ts.template_modules import ShapeConfig, StaticStats
            from clinical_ts.ts.encoder import RNNEncoder, RNNEncoderConfig
            from clinical_ts.ts.s4 import S4Predictor, S4PredictorConfig
            from clinical_ts.ts.head import MLPHead, MLPHeadConfig
            from clinical_ts.loss.selfsupervised import CPCLoss, CPCLossConfig
        except ImportError as exc:
            raise ImportError("ECG-CPC needs opt_einsum and pytorch-lightning. "
                              "Use the separate ecg-sota environment; never substitute LSTM-CPC.") from exc
        kernel = importlib.import_module("clinical_ts.ts.s4_modules.s42")
        original_cauchy = kernel.cauchy_conj
        if not getattr(original_cauchy, "_cpu_reference_dispatch", False):
            has_keops = importlib.util.find_spec("pykeops") is not None
            def dispatch(v, z, w):
                return reference_cauchy(v, z, w) if v.device.type == "cpu" or not has_keops else original_cauchy(v, z, w)
            dispatch._cpu_reference_dispatch = True
            kernel.cauchy_conj = dispatch
        width = 32 if smoke else 512
        self.architecture = dict(conv_features=[width]*4, conv_kernels=[3, 1, 1, 1],
            conv_strides=[2, 1, 1, 1], model_dim=width, state_dim=8, layers=1 if smoke else 4,
            causal=True, input_fs=250, native_fs=240, steps_predicted=14, negatives=128,
            loss_reduction="mean over prediction positions", mask_ratio=None)
        self.architecture["kernel"] = "upstream CUDA/PyKeOps if installed; exact PyTorch conjugate-pair sum otherwise"
        if not kernel.has_cauchy_extension and importlib.util.find_spec("pykeops") is None:
            print("ECG-CPC: exact PyTorch Cauchy backend active; optional CUDA/PyKeOps warnings above are not a failure.", flush=True)
        shape = ShapeConfig(channels=12, length=2500, sequence_last=True)
        self.encoder = RNNEncoder(RNNEncoderConfig(features=[width]*4, kss=[3, 1, 1, 1],
            strides=[2, 1, 1, 1]), shape, StaticStats())
        shape = self.encoder.get_output_shape()
        self.predictor = S4Predictor(S4PredictorConfig(model_dim=width, state_dim=8,
            layers=1 if smoke else 4, causal=True), shape)
        shape = self.predictor.get_output_shape()
        self.head = MLPHead(MLPHeadConfig(multi_prediction=True, bias=False), shape, width)
        self.criterion = CPCLoss(CPCLossConfig(steps_predicted=14, negatives_from_same_seq_only=True))
        self.dim = width

    def encoded_and_context(self, x):
        encoded = self.encoder(seq=x)["seq"]
        context = self.predictor(seq=encoded)["seq"]
        return encoded, context

    def ssl_loss(self, x):
        encoded, context = self.encoded_and_context(x)
        prediction = self.head(seq=context)["seq"]
        stats = self.criterion(prediction, encoded)
        return stats["loss"]/(encoded.shape[1]-14)

    def features(self, x):
        return self.encoded_and_context(x)[1].mean(dim=1)


def create_model(name, external_root, patch_samples=50, smoke=False):
    external_root = Path(external_root)
    if name == "stmem":
        return STMEMBenchmarkAdapter(external_root/"ST-MEM", patch_samples, smoke)
    if name == "patchecg":
        return PatchECGAdapter(external_root/"PatchECG", patch_samples, smoke)
    if name == "ecgcpc":
        return ECGCPCAdapter(external_root/"ecg-fm-benchmarking", smoke)
    raise ValueError(f"Unknown model: {name}")
