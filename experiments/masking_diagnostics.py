"""Read-only MAE diagnostics; model/training definitions remain in the notebook.

All reconstruction statistics use hidden samples only, never the visible-copy
composite. Baseline covariance must be fitted on the SSL training subset only.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def write_json(path, value):
    """Write standards-compliant JSON: undefined correlations become null."""
    def clean(item):
        if isinstance(item, dict):
            return {str(k): clean(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(v) for v in item]
        if isinstance(item, np.ndarray):
            return clean(item.tolist())
        if isinstance(item, np.generic):
            return clean(item.item())
        if isinstance(item, float) and not np.isfinite(item):
            return None
        return item

    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False,
                                     indent=2, allow_nan=False), encoding="utf-8")


@contextmanager
def preserve_rng(device):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def fixed_mask(ecg_ids, n_tokens, ratio, seed, repeat=0):
    count = int(round(n_tokens * ratio))
    if not 0 < count < n_tokens:
        raise ValueError("A mask must contain both hidden and visible tokens.")
    masks, seeds = [], []
    for ecg_id in ecg_ids:
        mask_seed = seed + 100000 + repeat * 1000000 + int(ecg_id)
        rng = np.random.default_rng(mask_seed)
        mask = np.zeros(n_tokens, dtype=bool)
        # Match random_mask_np(1, n_tokens, ratio, mask_seed) in the notebook.
        mask[np.argsort(rng.random((1, n_tokens)), axis=1)[0, :count]] = True
        masks.append(mask)
        seeds.append(mask_seed)
    return np.stack(masks), np.asarray(seeds, dtype=np.int64)


def _moments(prediction, target, sample_mask, axes):
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask = np.asarray(sample_mask, dtype=bool)
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise ValueError("prediction/target/sample_mask shapes must match")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("Reconstruction contains NaN/Inf")
    return {
        "count": mask.sum(axis=axes),
        "sse": np.where(mask, (prediction-target)**2, 0).sum(axis=axes),
        "pred_sum": np.where(mask, prediction, 0).sum(axis=axes),
        "target_sum": np.where(mask, target, 0).sum(axis=axes),
        "pred_sq": np.where(mask, prediction**2, 0).sum(axis=axes),
        "target_sq": np.where(mask, target**2, 0).sum(axis=axes),
        "cross": np.where(mask, prediction*target, 0).sum(axis=axes),
    }


def _metrics(moments):
    count = int(moments["count"])
    if count == 0:
        return dict(masked_samples=0, **{key: np.nan for key in (
            "masked_mse", "zero_masked_mse", "skill_vs_zero", "pred_rms", "target_rms",
            "rms_ratio", "pred_std", "target_std", "mean_bias", "masked_correlation")})
    pmean, tmean = moments["pred_sum"]/count, moments["target_sum"]/count
    pvar = max(0.0, moments["pred_sq"]/count-pmean**2)
    tvar = max(0.0, moments["target_sq"]/count-tmean**2)
    prms, trms = np.sqrt(moments["pred_sq"]/count), np.sqrt(moments["target_sq"]/count)
    mse, zero = moments["sse"]/count, moments["target_sq"]/count
    correlation = ((moments["cross"]/count-pmean*tmean)/np.sqrt(pvar*tvar)
                   if pvar > 1e-12 and tvar > 1e-12 else np.nan)
    return dict(masked_samples=count, masked_mse=float(mse), zero_masked_mse=float(zero),
                skill_vs_zero=float(1-mse/zero) if zero > 0 else np.nan,
                pred_rms=float(prms), target_rms=float(trms),
                rms_ratio=float(prms/trms) if trms > 0 else np.nan,
                pred_std=float(np.sqrt(pvar)), target_std=float(np.sqrt(tvar)),
                mean_bias=float(pmean-tmean),
                masked_correlation=float(np.clip(correlation, -1, 1)))


def masked_statistics(prediction, target, sample_mask):
    """Pooling by number of hidden samples, not by mean of batch/lead means."""
    return _metrics(_moments(prediction, target, sample_mask, axes=(0, 1, 2)))


def fit_cross_lead_baseline(signals, indices, batch_size=32, ridge=1e-3):
    """Fit same-time lead covariance without labels or validation signals."""
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) == 0 or ridge <= 0:
        raise ValueError("Training indices and positive ridge are required")
    channels = signals.shape[1]
    total, second, count = np.zeros(channels), np.zeros((channels, channels)), 0
    for start in range(0, len(indices), batch_size):
        x = np.asarray(signals[indices[start:start+batch_size]], dtype=np.float64)
        if not np.isfinite(x).all():
            raise ValueError("Training covariance input contains NaN/Inf")
        total += x.sum(axis=(0, 2))
        second += np.einsum("bct,bdt->cd", x, x, optimize=True)
        count += x.shape[0] * x.shape[2]
    mean = total/count
    covariance = second/count - np.outer(mean, mean)
    return dict(mean=mean, covariance=covariance, ridge=float(ridge), train_indices=indices.copy())


def cross_lead_predict(signal, patch_mask, patch_length, baseline):
    """Hidden leads use ONLY observed leads at the same time and train covariance.

    With no visible lead at a time patch, fall back to the train lead mean.
    This baseline deliberately exploits lead redundancy; it is not a learned MAE.
    """
    batch, channels, length = signal.shape
    q = length // patch_length
    if length % patch_length or patch_mask.shape != (batch, channels*q):
        raise ValueError("Invalid lead-time mask shape")
    hidden = patch_mask.reshape(batch, channels, q).transpose(0, 2, 1).reshape(-1, channels)
    chunks = signal.reshape(batch, channels, q, patch_length).transpose(0, 2, 1, 3).reshape(-1, channels, patch_length)
    mean, covariance, ridge = baseline["mean"], baseline["covariance"], baseline["ridge"]
    output = np.broadcast_to(mean[None, :, None], chunks.shape).copy()
    for pattern in np.unique(hidden, axis=0):
        rows = np.flatnonzero((hidden == pattern).all(axis=1))
        missing, observed = np.flatnonzero(pattern), np.flatnonzero(~pattern)
        if len(missing) == 0 or len(observed) == 0:
            continue
        weights = np.linalg.solve(covariance[np.ix_(observed, observed)] + ridge*np.eye(len(observed)),
                                  covariance[np.ix_(observed, missing)]).T
        visible = chunks[rows][:, observed, :] - mean[observed][None, :, None]
        estimates = mean[missing][None, :, None] + np.einsum("ho,bop->bhp", weights, visible)
        output[rows[:, None], missing[None, :], :] = estimates
    return output.reshape(batch, q, channels, patch_length).transpose(0, 2, 1, 3).reshape(signal.shape).astype(np.float32)


@torch.no_grad()
def evaluate_reconstruction(model, signals, ecg_ids, indices, batch_size, seed,
                            *, repeats=3, examples=3, lead_order=None, fs=250,
                            linear_baseline=None, mask_factory=None):
    """Repeat the original validation-mask seed formula; restore train/eval mode."""
    if not len(indices) or repeats < 1 or batch_size < 1 or examples < 0:
        raise ValueError("Non-empty indices and positive evaluation sizes required")
    device = next(model.parameters()).device
    channels, length = signals.shape[1:]
    leads = lead_order if lead_order is not None else [str(i) for i in range(channels)]
    if len(leads) != channels:
        raise ValueError("Lead order does not match signal shape")
    aggregate, linear_aggregate, per_ecg, samples = None, None, [], []
    previous_mode = model.training
    model.eval()
    try:
        for repeat in range(repeats):
            for start in range(0, len(indices), batch_size):
                selected = np.asarray(indices[start:start+batch_size], dtype=np.int64)
                ids = np.asarray(ecg_ids)[selected]
                mask_np, seeds = (mask_factory or fixed_mask)(ids, model.n, model.ratio, seed, repeat)
                raw = np.asarray(signals[selected], dtype=np.float32).copy()
                x = torch.from_numpy(raw).to(device)
                mask = torch.from_numpy(mask_np).to(device)
                pred, _ = model(x, mask)
                prediction = pred.detach().float().cpu().numpy().reshape(len(raw), channels, length)
                sample_mask = np.repeat(mask_np, model.p, axis=1).reshape(raw.shape)
                moments = _moments(prediction, raw, sample_mask, axes=(0, 2))
                if aggregate is None:
                    aggregate = {k: v.copy() for k, v in moments.items()}
                else:
                    for key in aggregate:
                        aggregate[key] += moments[key]
                record_moments = _moments(prediction, raw, sample_mask, axes=(1, 2))
                lead_time = mask_np.reshape(len(raw), channels, length//model.p)
                for i, ecg_id in enumerate(ids):
                    stats = _metrics({k: v[i] for k, v in record_moments.items()})
                    per_ecg.append(dict(ecg_id=int(ecg_id), artifact_index=int(selected[i]),
                                        repeat=repeat, mask_seed=int(seeds[i]),
                                        entirely_hidden_leads=int(lead_time[i].all(axis=1).sum()),
                                        time_patches_without_visible_lead=int(lead_time[i].all(axis=0).sum()),
                                        **stats))
                if linear_baseline is not None:
                    estimate = cross_lead_predict(raw, mask_np, model.p, linear_baseline)
                    baseline_moments = _moments(estimate, raw, sample_mask, axes=(0, 2))
                    if linear_aggregate is None:
                        linear_aggregate = {k: v.copy() for k, v in baseline_moments.items()}
                    else:
                        for key in linear_aggregate:
                            linear_aggregate[key] += baseline_moments[key]
                if repeat == 0 and start < examples:
                    keep = min(len(raw), examples-start)
                    samples.append(dict(target=raw[:keep], prediction=prediction[:keep],
                                        patch_mask=mask_np[:keep], sample_mask=sample_mask[:keep],
                                        ecg_ids=ids[:keep], artifact_indices=selected[:keep], mask_seeds=seeds[:keep]))
    finally:
        model.train(previous_mode)
    metrics = _metrics({k: v.sum() for k, v in aggregate.items()})
    per_lead = pd.DataFrame([dict(lead=name, **_metrics({k: v[i] for k, v in aggregate.items()}))
                             for i, name in enumerate(leads)])
    if linear_aggregate is not None:
        linear_metrics = _metrics({k: v.sum() for k, v in linear_aggregate.items()})
        metrics.update(linear_masked_mse=linear_metrics["masked_mse"],
                       linear_skill_vs_zero=linear_metrics["skill_vs_zero"])
        per_lead["linear_masked_mse"] = [_metrics({k: v[i] for k, v in linear_aggregate.items()})["masked_mse"]
                                        for i in range(channels)]
    example_arrays = {}
    if samples:
        example_arrays = {key: np.concatenate([sample[key] for sample in samples]) for key in samples[0]}
        example_arrays["composite"] = np.where(example_arrays["sample_mask"], example_arrays["prediction"], example_arrays["target"])
        example_arrays.update(patch_samples=np.asarray(model.p), mask_ratio=np.asarray(model.ratio),
                              lead_order=np.asarray(leads), fs=np.asarray(fs), repeat=np.asarray(0),
                              units=np.asarray("preprocessed per-ECG per-lead z-score; not mV"))
    return dict(metrics=metrics, per_lead=per_lead, per_ecg=pd.DataFrame(per_ecg), samples=example_arrays)


def save_reconstruction(bundle, directory, prefix="final"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / f"{prefix}_reconstruction_metrics.json", bundle["metrics"])
    bundle["per_lead"].to_csv(directory / f"{prefix}_reconstruction_per_lead.csv", index=False)
    bundle["per_ecg"].to_csv(directory / f"{prefix}_reconstruction_per_ecg.csv", index=False)
    samples = bundle["samples"]
    if not samples:
        return
    np.savez_compressed(directory / f"{prefix}_reconstruction_samples.npz", **samples)
    time_axis = np.arange(samples["target"].shape[2]) / float(samples["fs"])
    for i, ecg_id in enumerate(samples["ecg_ids"]):
        fig, axes = plt.subplots(len(samples["lead_order"]), 1, figsize=(12, 16), sharex=True,
                                 squeeze=False, constrained_layout=True)
        try:
            for j, name in enumerate(samples["lead_order"]):
                ax = axes[j, 0]
                target, pred, hidden = samples["target"][i, j], samples["prediction"][i, j], samples["sample_mask"][i, j]
                ax.plot(time_axis, target, color="C0", lw=0.8, label="Target (preprocessed)")
                ax.plot(time_axis, np.where(hidden, pred, np.nan), color="C3", lw=0.9,
                        label="Prediction on hidden samples only")
                ax.fill_between(time_axis, 0, 1, where=hidden, color="0.6", alpha=0.15,
                                transform=ax.get_xaxis_transform(), step="mid", label="Hidden region")
                ax.set_ylabel(str(name))
                ax.grid(alpha=0.15)
            axes[0, 0].legend(loc="upper right", fontsize=7)
            axes[-1, 0].set_xlabel("Time (seconds)")
            fig.suptitle(f"{prefix} | ECG {int(ecg_id)} | patch {int(samples['patch_samples'])} | "
                         f"mask {float(samples['mask_ratio']):.0%}\n"
                         "Amplitude: z-score, NOT mV. Visible input is not counted as reconstructed.", fontsize=11)
            fig.savefig(directory / f"{prefix}_reconstruction_ecg{int(ecg_id)}.png", dpi=120)
        finally:
            plt.close(fig)


def initialization_report(model):
    def equality(stack):
        if stack is None or not hasattr(stack, "layers"):
            return None
        layers = list(stack.layers)
        if len(layers) < 2:
            return None
        first = layers[0].state_dict()
        return all(all(torch.equal(first[key], value) for key, value in layer.state_dict().items())
                   for layer in layers[1:])

    return dict(initialization=getattr(model, "initialization", "upstream"),
                encoder_layers_identical_at_initialization=equality(getattr(model, "encoder", None)),
                decoder_layers_identical_at_initialization=equality(getattr(model, "decoder", None)),
                note="Identical initial values do not imply shared parameters or establish a cause.")


def _group(name):
    # External ST-MEM adapter: separate official encoder blocks and fixed positions.
    if name.startswith("core."):
        parts = name.split(".")
        if len(parts) > 2 and parts[1] == "encoder":
            return ".".join(parts[1:3])
        if len(parts) > 2 and parts[1] == "decoder_blocks":
            return ".".join(parts[1:3])
        return parts[1]
    if name in {"enc_lead", "enc_time"}:
        return "encoder_position"
    if name in {"dec_lead", "dec_time"}:
        return "decoder_position"
    if name.startswith(("encoder.layers.", "decoder.layers.")):
        return ".".join(name.split(".")[:3])
    return name.split(".")[0]


@torch.no_grad()
def gradient_snapshot(model, step):
    groups, before = {}, {}
    for name, parameter in model.named_parameters():
        groups.setdefault(_group(name), []).append((name, parameter))
        before[name] = parameter.detach().clone()
    rows = []
    for group, entries in groups.items():
        gradients = [p.grad.detach().norm() for _, p in entries if p.grad is not None]
        weight_norm = torch.stack([p.detach().norm() for _, p in entries]).norm()
        rows.append(dict(step=step, group=group,
                         grad_norm_before_clip=float(torch.stack(gradients).norm()) if gradients else 0.0,
                         parameters_without_grad=sum(p.grad is None for _, p in entries),
                         parameter_tensors=len(entries), weight_norm_before=float(weight_norm)))
    return rows, before


@torch.no_grad()
def complete_update_snapshot(model, rows, before):
    changes = {}
    for name, parameter in model.named_parameters():
        changes.setdefault(_group(name), []).append((parameter.detach()-before[name]).norm())
    for row in rows:
        update_norm = float(torch.stack(changes[row["group"]]).norm())
        row.update(update_norm=update_norm,
                   relative_update_norm=update_norm/max(row["weight_norm_before"], 1e-12))
    return rows


def feature_statistics(features):
    x = features.detach().float().cpu().numpy().astype(np.float64)
    centered = x-x.mean(axis=0, keepdims=True)
    std = centered.std(axis=0)
    eigenvalues = np.maximum(np.linalg.eigvalsh(centered.T@centered/max(1, len(x))), 0)
    probabilities = eigenvalues[eigenvalues > 0] / max(eigenvalues.sum(), 1e-30)
    effective_rank = float(np.exp(-(probabilities*np.log(probabilities)).sum())) if len(probabilities) else 0.0
    normalized = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    mean_cosine = ((normalized.sum(axis=0)**2).sum()-(normalized**2).sum()) / max(1, len(x)*(len(x)-1))
    return dict(ecgs=len(x), feature_dim=x.shape[1], std_min=float(std.min()), std_mean=float(std.mean()),
                near_constant_dimensions=int((std < 1e-6).sum()), effective_rank=effective_rank,
                mean_off_diagonal_cosine=float(mean_cosine),
                note="Effective rank is based on centered covariance eigenvalues; no universal collapse cutoff.")


def save_diagnostic_progress(evaluations, gradient_rows, directory, *, plot=True):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    history = pd.DataFrame(evaluations)
    history.to_csv(directory / "evaluation_history.csv", index=False)
    gradients = pd.DataFrame(gradient_rows)
    if not gradients.empty:
        gradients.to_csv(directory / "gradient_updates.csv", index=False)
    if not plot:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    try:
        for split, group in history.groupby("split", sort=False):
            axes[0, 0].plot(group.step, group.masked_mse, marker="o", label=split)
            axes[0, 0].plot(group.step, group.zero_masked_mse, linestyle="--", alpha=0.6, label=f"{split}: zero baseline")
            axes[0, 1].plot(group.step, group.skill_vs_zero, marker="o", label=split)
            axes[1, 0].plot(group.step, group.rms_ratio, marker="o", label=split)
            axes[1, 1].plot(group.step, group.masked_correlation, marker="o", label=split)
        titles = ["Fixed-mask masked MSE", "Skill = 1 - MSE / zero MSE", "Masked prediction / target RMS", "Hidden-sample Pearson correlation"]
        for ax, title in zip(axes.flat, titles):
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Training update (0 = before training)")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
        axes[0, 1].axhline(0, color="0.5", lw=0.8)
        axes[1, 0].axhline(1, color="0.5", lw=0.8)
        fig.suptitle("Fixed train/validation diagnostic subsets; not checkpoint selection")
        fig.savefig(directory / "diagnostic_curves.png", dpi=140)
    finally:
        plt.close(fig)
    if gradients.empty:
        return
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    try:
        for group, rows in gradients.groupby("group", sort=False):
            axes[0].plot(rows.step, np.maximum(rows.grad_norm_before_clip, 1e-15), label=group)
            axes[1].plot(rows.step, np.maximum(rows.relative_update_norm, 1e-15), label=group)
        for ax in axes:
            ax.set_yscale("log")
            ax.grid(alpha=0.2)
        axes[0].set_ylabel("Gradient norm before clipping")
        axes[1].set_ylabel("Actual update / previous weight norm")
        axes[1].set_xlabel("Training update")
        axes[0].legend(fontsize=7, ncol=3)
        fig.suptitle("Module/layer diagnostics (zeros plotted at 1e-15; exact values in CSV)")
        fig.savefig(directory / "gradient_updates.png", dpi=140)
    finally:
        plt.close(fig)


def diagnostic_summary(final_metrics, history, random_probe, trained_probe, evaluations):
    observations = []
    if final_metrics["skill_vs_zero"] < 0.05:
        observations.append("Less than 5% MSE reduction versus zero prediction: reconstruction learning is weak.")
    if final_metrics["rms_ratio"] < 0.2:
        observations.append("Masked prediction RMS is below 20% of target: check near-zero or amplitude-shrunken outputs.")
    if final_metrics.get("linear_skill_vs_zero", -np.inf) > final_metrics["skill_vs_zero"] + 0.1:
        observations.append("Train-fitted same-time cross-lead baseline is substantially better; inspect lead/time alignment and model learning capacity. This is not proof of a specific cause.")
    gain = None if random_probe is None else trained_probe["macro_auroc"]-random_probe["macro_auroc"]
    if gain is not None and gain < 0.01:
        observations.append("AUROC gain over the matched random encoder is below 0.01; SSL benefit is not established by this one-seed run.")
    return dict(final_validation=final_metrics, random_encoder_probe=random_probe,
                trained_encoder_probe=trained_probe, auroc_gain_vs_random=gain,
                clipping_fraction=float(history.clipped.mean()),
                first_fixed_evaluation=evaluations[:2], last_fixed_evaluation=evaluations[-2:],
                observations=observations,
                caveat="Thresholds are debugging heuristics, not causal findings. Diagnose with curves, waveforms, tiny overfit and controlled longer/multi-seed runs.")


def run_tiny_overfit(model, signals, ecg_ids, indices, steps, seed, directory,
                     *, lead_order, fs=250, mask_factory=None):
    """A separate fresh model memorizes <=2 train ECGs with one fixed mask.

    No validation, test labels, probe or model selection. Constant LR is
    intentional: this is an optimizer/capacity sanity check, not the pilot.
    """
    if steps < 1 or not 0 < len(indices) <= 2:
        raise ValueError("Tiny overfit requires positive steps and 1-2 training ECGs")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    device = next(model.parameters()).device
    x = torch.from_numpy(np.asarray(signals[indices], dtype=np.float32).copy()).to(device)
    mask_np, _ = (mask_factory or fixed_mask)(np.asarray(ecg_ids)[indices], model.n, model.ratio, seed)
    mask = torch.from_numpy(mask_np).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.05)
    write_json(directory / "initialization.json", initialization_report(model))
    before = evaluate_reconstruction(model, signals, ecg_ids, indices, len(indices), seed,
                                     repeats=1, examples=2, lead_order=lead_order, fs=fs,
                                     mask_factory=mask_factory)
    save_reconstruction(before, directory, prefix="initial")
    write_json(directory / "run_config.json", dict(kind="tiny-overfit-not-pilot", steps=steps, seed=seed,
               patch_samples=model.p, mask_ratio=model.ratio, ecg_ids=np.asarray(ecg_ids)[indices],
               artifact_indices=np.asarray(indices), learning_rate=3e-4, schedule="constant", mask="fixed",
               note="Training ECG memorization only. No generalization claim or automatic pass/fail cutoff."))
    history = []
    model.train()
    for step in range(steps):
        pred, target = model(x, mask)
        loss = ((pred-target).square().mean(dim=-1)*mask).sum()/mask.sum()
        if not torch.isfinite(loss):
            raise RuntimeError("Tiny-overfit loss NaN/Inf")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        history.append(dict(step=step+1, masked_mse=float(loss.detach()), grad_norm=float(norm), lr=3e-4))
        if (step+1) % max(1, steps//5) == 0:
            pd.DataFrame(history).to_csv(directory / "pretrain_history.csv", index=False)
            print(f"Tiny overfit: {step+1}/{steps}, MSE={history[-1]['masked_mse']:.5f}", flush=True)
    after = evaluate_reconstruction(model, signals, ecg_ids, indices, len(indices), seed,
                                    repeats=1, examples=2, lead_order=lead_order, fs=fs,
                                    mask_factory=mask_factory)
    save_reconstruction(after, directory, prefix="final")
    write_json(directory / "summary.json", dict(initial=before["metrics"], final=after["metrics"],
               mse_reduction_fraction=1-after["metrics"]["masked_mse"]/max(before["metrics"]["masked_mse"], 1e-12),
               note="Low fixed-training loss verifies memorization, not validation or downstream benefit."))
    torch.save(dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                    patch_length=model.p, mask_ratio=model.ratio, kind="tiny-overfit-not-pilot"),
               directory / "checkpoint.pt")
    return pd.DataFrame(history)
