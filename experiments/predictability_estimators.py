"""Mask-conditioned ECG reconstruction and held-out mask difficulty analysis.

Inputs are preprocessed ECGs. Hidden values are removed inside every estimator.
NRMSE uses one fixed standard deviation per lead, fitted on estimator-fit ECGs.
"""
from __future__ import annotations

import hashlib
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch
from torch import nn

ESTIMATORS = ("ridge", "cnn", "transformer")


def patient_pools(metadata, fit_n, score_n, seed):
    """Reserve 20% of train patients for scoring; caps never mix patient pools."""
    if metadata.patient_id.isna().any() or metadata.ecg_id.duplicated().any():
        raise ValueError("Unique ECG IDs and nonmissing patient IDs are required")
    if fit_n < 0 or score_n < 0:
        raise ValueError("ECG counts must be nonnegative (0 means all)")
    patients = np.unique(metadata.patient_id.to_numpy())
    if len(patients) < 2:
        raise ValueError("At least two train patients are required")
    rng = np.random.default_rng(seed)
    patients = rng.permutation(patients)
    split = min(len(patients)-1, max(1, int(len(patients)*0.8)))
    fit = rng.permutation(np.flatnonzero(metadata.patient_id.isin(patients[:split])))
    score = rng.permutation(np.flatnonzero(metadata.patient_id.isin(patients[split:])))
    if (fit_n and fit_n > len(fit)) or (score_n and score_n > len(score)):
        raise ValueError("Requested ECG count exceeds its reserved patient pool")
    return np.sort(fit[:fit_n] if fit_n else fit), np.sort(score[:score_n] if score_n else score)


def lead_scale(signals, indices, batch_size=32):
    """Population std in input units; streaming moments, estimator-fit ECGs only."""
    if not len(indices):
        raise ValueError("Empty estimator-fit subset")
    channels = signals.shape[1]
    total, square, count = np.zeros(channels), np.zeros(channels), 0
    for start in range(0, len(indices), batch_size):
        x = np.asarray(signals[indices[start:start+batch_size]], dtype=np.float64)
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite fitting signals")
        total += x.sum(axis=(0, 2))
        square += np.square(x).sum(axis=(0, 2))
        count += x.shape[0]*x.shape[2]
    mean = total/count
    std = np.sqrt(np.maximum(0, square/count-mean**2))
    if (std < 1e-8).any():
        raise ValueError("A train lead has zero/near-zero std; inspect preprocessing")
    return mean, std


def mask_bank(metadata, indices, patch_samples, length, ratio, repeats, seed, role):
    """Persist identities/seeds/digests; masks can be regenerated in any row order."""
    if length % patch_samples or repeats < 1 or not 0 < ratio < 1:
        raise ValueError("Invalid patch length, repeat count or masking ratio")
    channels = 12
    tokens = channels*(length//patch_samples)
    count = int(round(tokens*ratio))
    if not 0 < count < tokens:
        raise ValueError("Mask needs hidden and visible patches")
    rows, masks = [], []
    for index in indices:
        ecg_id = int(metadata.iloc[index].ecg_id)
        for repeat in range(repeats):
            mask_id = f"{role}-e{ecg_id}-m{repeat}"
            mask_seed = int.from_bytes(hashlib.sha256(f"{seed}:{mask_id}".encode()).digest()[:8], "little") & ((1 << 63)-1)
            hidden = np.zeros(tokens, dtype=bool)
            hidden[np.random.default_rng(mask_seed).choice(tokens, count, replace=False)] = True
            rows.append(dict(mask_id=mask_id, ecg_id=ecg_id, patient_id=metadata.iloc[index].patient_id,
                             artifact_index=int(index), repeat=repeat, mask_seed=mask_seed,
                             hidden_tokens=count, realized_mask_ratio=count/tokens,
                             mask_sha256=hashlib.sha256(np.packbits(hidden).tobytes()).hexdigest()))
            masks.append(hidden)
    if not rows:
        raise ValueError("Empty mask bank")
    return pd.DataFrame(rows), np.stack(masks)


def sample_hidden(mask, shape):
    batch, channels, length = shape
    if mask.ndim != 2 or mask.shape[0] != batch or mask.shape[1] % channels:
        raise ValueError("Mask does not match batch/lead dimensions")
    patches = mask.shape[1]//channels
    if not patches or length % patches:
        raise ValueError("Mask does not match time dimension")
    return np.repeat(np.asarray(mask, bool).reshape(batch, channels, patches), length//patches, axis=2)


def masks_from_bank(bank, n_tokens):
    """Load a compact exported CSV and verify every regenerated mask digest."""
    masks = []
    if bank.mask_id.duplicated().any():
        raise ValueError("Duplicate mask IDs")
    for row in bank.itertuples():
        if not 0 < row.hidden_tokens < n_tokens:
            raise ValueError("Invalid hidden token count")
        hidden = np.zeros(n_tokens, dtype=bool)
        hidden[np.random.default_rng(int(row.mask_seed)).choice(n_tokens, int(row.hidden_tokens), replace=False)] = True
        digest = hashlib.sha256(np.packbits(hidden).tobytes()).hexdigest()
        if digest != row.mask_sha256:
            raise ValueError(f"Mask digest mismatch: {row.mask_id}")
        masks.append(hidden)
    if not masks:
        raise ValueError("Empty exported mask bank")
    return np.stack(masks)


class LaggedRidge:
    """Linear waveform+availability features at fixed lags, fitted masked-only.

    Shared across time, separate weights per output lead. Other leads at t and
    adjacent visible patches at t +/- P, 2P supply spatial and temporal context.
    Objective: mean target-lead squared error + alpha*||w||^2; intercept unpenalized.
    Availability flags distinguish missing values from a real zero voltage.
    """
    def __init__(self, patch_samples, alpha=1e-3, lags=None):
        if alpha <= 0:
            raise ValueError("Positive ridge alpha required")
        self.lags = tuple(lags if lags is not None else (-2*patch_samples, -patch_samples, -1, 0, 1, patch_samples, 2*patch_samples))
        self.alpha = alpha
        self.weights = None

    def features(self, signals, mask, times, scale):
        hidden = sample_hidden(mask, signals.shape)
        visible = np.where(hidden, 0, signals)/scale[None, :, None]
        observed = ~hidden
        positions = np.asarray(times)[:, :, None]+np.asarray(self.lags)[None, None, :]
        valid = (positions >= 0) & (positions < signals.shape[2])
        positions = np.clip(positions, 0, signals.shape[2]-1)
        b = np.arange(len(signals))[:, None, None, None]
        c = np.arange(signals.shape[1])[None, None, :, None]
        available = observed[b, c, positions[:, :, None, :]] & valid[:, :, None, :]
        values = visible[b, c, positions[:, :, None, :]]
        values = np.where(available, values, 0)
        shape = (len(signals)*times.shape[1], -1)
        features = np.concatenate([values.reshape(shape), available.reshape(shape),
                                   np.ones((len(signals)*times.shape[1], 1))], axis=1)
        return features.astype(np.float64)

    def fit(self, signals, bank, masks, scale, *, samples_per_ecg=32, batch_size=16, seed=42):
        if samples_per_ecg < 1:
            raise ValueError("Positive time sample count required")
        channels, length = signals.shape[1:]
        dimension = channels*len(self.lags)*2+1
        gram = np.zeros((channels, dimension, dimension))
        rhs, counts = np.zeros((channels, dimension)), np.zeros(channels, dtype=np.int64)
        rng = np.random.default_rng(seed)
        progress_interval = max(batch_size, (len(bank)//10//batch_size)*batch_size)
        for start in range(0, len(bank), batch_size):
            rows = bank.iloc[start:start+batch_size]
            x = np.asarray(signals[rows.artifact_index.to_numpy()], dtype=np.float32)
            m = masks[start:start+len(rows)]
            times = rng.integers(0, length, size=(len(rows), samples_per_ecg))
            features = self.features(x, m, times, scale)
            b = np.arange(len(x))[:, None]
            targets = x[b, :, times].reshape(-1, channels)/scale
            hidden = sample_hidden(m, x.shape)[b, :, times].reshape(-1, channels)
            for lead in range(channels):
                f = features[hidden[:, lead]]
                y = targets[hidden[:, lead], lead]
                gram[lead] += f.T @ f
                rhs[lead] += f.T @ y
                counts[lead] += len(f)
            if (start+len(rows)) % progress_interval == 0 or start+len(rows) == len(bank):
                print(f"Ridge fit: {start+len(rows)}/{len(bank)} common train masks", flush=True)
        if not counts.all():
            raise ValueError("No ridge training targets for a lead")
        penalty = np.eye(dimension)*self.alpha
        penalty[-1, -1] = 0
        self.weights = np.stack([np.linalg.solve(gram[c]/counts[c]+penalty, rhs[c]/counts[c])
                                 for c in range(channels)])
        self.scale, self.target_counts = np.asarray(scale).copy(), counts
        return self

    def predict(self, signals, mask):
        if self.weights is None:
            raise RuntimeError("Fit ridge before prediction")
        times = np.broadcast_to(np.arange(signals.shape[2]), (len(signals), signals.shape[2]))
        features = self.features(signals, mask, times, self.scale)
        output = (features @ self.weights.T)*self.scale
        return output.reshape(len(signals), signals.shape[2], signals.shape[1]).transpose(0, 2, 1).astype(np.float32)


def torch_visible(x, mask):
    if x.ndim != 3 or mask.ndim != 2 or mask.dtype != torch.bool or mask.shape[0] != len(x):
        raise ValueError("Invalid bool patch mask")
    patches = mask.shape[1]//x.shape[1]
    if not patches or mask.shape[1] % x.shape[1] or x.shape[2] % patches:
        raise ValueError("Invalid bool patch mask")
    hidden = mask.reshape(len(x), x.shape[1], patches).repeat_interleave(x.shape[2]//patches, dim=2)
    return x.masked_fill(hidden, 0), (~hidden).to(x.dtype), hidden


class ConvBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(width, width, 7, padding=3*dilation, dilation=dilation),
                                 nn.SiLU(), nn.Conv1d(width, width, 1))

    def forward(self, x):
        return torch.nn.functional.silu(x+self.net(x))


class SmallCNN(nn.Module):
    def __init__(self, channels=12, width=64):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(channels*2, width, 7, padding=3), nn.SiLU(),
                                 *(ConvBlock(width, d) for d in (1, 2, 4, 8)), nn.Conv1d(width, channels, 1))

    def forward(self, x, mask):
        visible, observed, _ = torch_visible(x, mask)
        return self.net(torch.cat([visible, observed], dim=1))


class SmallTransformer(nn.Module):
    def __init__(self, patch_samples=50, length=2500, channels=12, width=128, depth=4, heads=4):
        super().__init__()
        if length % patch_samples or width % heads or width % 2:
            raise ValueError("Invalid Transformer dimensions")
        self.patch_samples, self.channels = patch_samples, channels
        self.projection = nn.Linear(patch_samples+1, width)
        self.lead_embedding = nn.Parameter(torch.randn(1, channels, 1, width)*0.02)
        q = length//patch_samples
        positions = torch.arange(q)[:, None]
        frequencies = torch.exp(torch.arange(0, width, 2)*(-np.log(10000.0)/width))
        encoding = torch.zeros(q, width)
        encoding[:, 0::2], encoding[:, 1::2] = torch.sin(positions*frequencies), torch.cos(positions*frequencies)
        self.register_buffer("time_encoding", encoding[None, None])
        # Construct separately: no cloned, identically initialized blocks.
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(width, heads, width*4, dropout=0,
                                   activation="gelu", batch_first=True, norm_first=True) for _ in range(depth)])
        self.output = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, patch_samples))

    def forward(self, x, mask):
        visible, _, _ = torch_visible(x, mask)
        b, c, length = x.shape
        q, p = length//self.patch_samples, self.patch_samples
        if c != self.channels or q != self.time_encoding.shape[2]:
            raise ValueError("Input does not match model dimensions")
        chunks = visible.reshape(b, c, q, p)
        observed = (~mask).reshape(b, c, q, 1).to(x.dtype)
        z = self.projection(torch.cat([chunks, observed], dim=-1))+self.lead_embedding+self.time_encoding
        z = z.reshape(b, c*q, -1)
        for block in self.blocks:
            z = block(z)
        return self.output(z).reshape(b, c, length)


def normalized_masked_loss(prediction, target, mask, scale):
    _, _, hidden = torch_visible(target, mask)
    residual = (prediction-target)/scale[None, :, None]
    return residual.square()[hidden].mean()


def reconstruction_metrics(prediction, target, mask, scale):
    """Per-case pooled standardized NRMSE; per-hidden-patch NRMSE for maps."""
    if prediction.shape != target.shape or not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("Invalid reconstruction arrays")
    scale = np.asarray(scale)
    if scale.shape != (target.shape[1],) or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Fixed train lead std required")
    hidden = sample_hidden(mask, target.shape)
    error = (np.asarray(prediction, np.float64)-target)/scale[None, :, None]
    q = mask.shape[1]//target.shape[1]
    patch = np.sqrt(np.square(error).reshape(len(target), target.shape[1], q, -1).mean(axis=-1))
    patch = np.where(mask.reshape(patch.shape), patch, np.nan)
    rows = []
    for b in range(len(target)):
        y = target[b][hidden[b]]/np.broadcast_to(scale[:, None], target.shape[1:])[hidden[b]]
        pred = prediction[b][hidden[b]]/np.broadcast_to(scale[:, None], target.shape[1:])[hidden[b]]
        if not len(y):
            raise ValueError("No hidden target samples")
        mse = np.mean((pred-y)**2)
        centered_y, centered_pred = y-y.mean(), pred-pred.mean()
        energy, pred_energy = float(centered_y @ centered_y), float(centered_pred @ centered_pred)
        zero_mse = float(np.mean(y**2))
        rows.append(dict(nrmse=float(np.sqrt(mse)), zero_nrmse=float(np.sqrt(zero_mse)),
                         skill_vs_zero=1-mse/zero_mse if zero_mse > 1e-12 else np.nan,
                         masked_r2=1-float(np.sum((pred-y)**2))/energy if energy > 1e-12 else np.nan,
                         masked_pearson=float(np.clip(centered_y @ centered_pred/np.sqrt(energy*pred_energy), -1, 1))
                         if energy > 1e-12 and pred_energy > 1e-12 else np.nan,
                         masked_samples=int(hidden[b].sum())))
    return pd.DataFrame(rows), patch


def predictability_percentile(errors):
    """Midrank empirical percentile: 100*(N-rank+0.5)/N; ties share a rank."""
    x = np.asarray(errors, dtype=np.float64)
    if not len(x) or not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("Finite nonnegative NRMSEs required for every mask")
    return 100*(len(x)-pd.Series(x).rank(method="average").to_numpy()+0.5)/len(x)


def difficulty_table(case_metrics):
    """Strict three-estimator join; no partial consensus or imputed errors."""
    required = {"mask_id", "estimator", "nrmse"}
    if not required.issubset(case_metrics.columns) or case_metrics.duplicated(["mask_id", "estimator"]).any():
        raise ValueError("One metric per mask/estimator required")
    errors = case_metrics.pivot(index="mask_id", columns="estimator", values="nrmse")
    if set(errors.columns) != set(ESTIMATORS) or errors.isna().any().any() or len(errors) < 3:
        raise ValueError("All three estimators and at least three complete masks required")
    result = errors.rename(columns={name: f"{name}_nrmse" for name in ESTIMATORS}).copy()
    for name in ESTIMATORS:
        result[f"{name}_predictability"] = predictability_percentile(errors[name])
    columns = [f"{name}_predictability" for name in ESTIMATORS]
    values = result[columns].to_numpy()
    result["consensus_predictability"] = values.mean(axis=1)
    result["disagreement_std"] = values.std(axis=1, ddof=0)
    result["disagreement_range"] = np.ptp(values, axis=1)
    result = result.reset_index().sort_values(["consensus_predictability", "mask_id"], ascending=[False, True])
    result["consensus_rank"] = np.arange(1, len(result)+1)
    result["difficulty"] = ""
    for group, positions in zip(("Easy", "Medium", "Hard"), np.array_split(np.arange(len(result)), 3)):
        result.iloc[positions, result.columns.get_loc("difficulty")] = group
    correlations = []
    for first, second in combinations(ESTIMATORS, 2):
        x, y = errors[first].to_numpy(), errors[second].to_numpy()
        constant = np.ptp(x) == 0 or np.ptp(y) == 0
        rho = np.nan if constant else float(spearmanr(x, y).statistic)
        correlations.append(dict(estimator_a=first, estimator_b=second, spearman_rho=rho,
                                 masks=len(errors), undefined_constant_input=constant))
    return result.reset_index(drop=True), pd.DataFrame(correlations)
