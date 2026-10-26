"""Shared physical-unit reconstruction and frozen-probe audits."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import masking_diagnostics as diagnostics
from mae_cause_data import CLASSES, LEADS, normalize
from mae_cause_model import mask_batch


def masks_for_ids(ids, plan, bank, repeat=0):
    return np.concatenate([mask_batch(1, 600, plan["mask_ratio"],
        plan["eval_seed"] + 100000 + repeat * 1000000 + int(i), bank) for i in ids])


@torch.no_grad()
def evaluate(model, data, spec, plan, device, *, split, indices, bank, repeats=1, baseline=None, example=None, fixed_masks=None):
    previous = model.training
    model.eval()
    rows, moments, example_data = [], None, None
    mask_hash = hashlib.sha256()
    std = np.asarray(data["stats"]["lead_std"])[None, :, None]
    try:
        for repeat in range(repeats):
            for start in range(0, len(indices), plan["microbatch"]):
                ix = indices[start:start + plan["microbatch"]]
                frame = data["frames"][split].iloc[ix]
                raw = np.asarray(data["arrays"][split][ix], dtype=np.float32)
                mask = (masks_for_ids(frame.ecg_id.to_numpy(), plan, bank, repeat) if fixed_masks is None
                        else fixed_masks[start:start + len(ix)])
                mask_hash.update(mask.tobytes())
                x, mean, scale = normalize(raw, spec["normalization"], data["stats"], mask=mask)
                pred, _ = model(torch.from_numpy(x).to(device), torch.from_numpy(mask).to(device))
                pz = pred.float().cpu().numpy().reshape(raw.shape)
                pmv = (pz * scale + mean).astype(np.float32)
                hidden = np.repeat(mask, 50, axis=1).reshape(raw.shape)
                linear = diagnostics.cross_lead_predict(raw, mask, 50, baseline) if baseline is not None else None
                for j, row in enumerate(frame.itertuples()):
                    metrics = diagnostics.masked_statistics(pmv[j:j+1], raw[j:j+1], hidden[j:j+1])
                    scaled_error = (((pmv[j:j+1] - raw[j:j+1]) / std)**2)[hidden[j:j+1]]
                    metrics.update(standardized_mse=float(scaled_error.mean()),
                        normalized_masked_mse=float(((pz[j] - x[j])**2)[hidden[j]].mean()),
                        normalized_zero_mse=float((x[j]**2)[hidden[j]].mean()))
                    if linear is not None:
                        metrics["linear_standardized_mse"] = float((((linear[j:j+1] - raw[j:j+1]) / std)**2)[hidden[j:j+1]].mean())
                    rows.append(dict(ecg_id=int(row.ecg_id), patient_id=int(row.patient_id),
                                     repeat=repeat, split=split, bank=bank, **metrics))
                batch_moments = diagnostics._moments(pmv, raw, hidden, axes=(0, 1, 2))
                if moments is None:
                    moments = batch_moments
                else:
                    for key in moments:
                        moments[key] += batch_moments[key]
                if example is not None and example_data is None:
                    example_data = dict(target=raw[0], prediction=pmv[0], hidden=hidden[0],
                                        ecg_id=int(frame.ecg_id.iloc[0]), units="mV")
    finally:
        model.train(previous)
    frame = pd.DataFrame(rows)
    metrics = diagnostics._metrics(moments)
    metrics.update(standardized_mse=float(frame.standardized_mse.mean()),
                   standardized_nrmse=float(np.sqrt(frame.standardized_mse.mean())),
                   normalized_masked_mse=float(frame.normalized_masked_mse.mean()),
                   normalized_zero_mse=float(frame.normalized_zero_mse.mean()), mask_sha256=mask_hash.hexdigest())
    if baseline is not None:
        metrics["linear_standardized_mse"] = float(frame.linear_standardized_mse.mean())
    if example_data is not None:
        save_waveform(example_data, Path(example))
    return metrics, frame


def save_waveform(sample, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    np.savez_compressed(path.with_suffix(".npz"), **sample)
    fig, axes = plt.subplots(12, 1, figsize=(13, 15), sharex=True, constrained_layout=True)
    t = np.arange(2500) / 250
    try:
        for i, ax in enumerate(axes):
            ax.plot(t, sample["target"][i], lw=.7, label="Target")
            ax.plot(t, np.where(sample["hidden"][i], sample["prediction"][i], np.nan), lw=.8, label="Prediction (hidden only)")
            ax.fill_between(t, 0, 1, where=sample["hidden"][i], transform=ax.get_xaxis_transform(), alpha=.08, color="black")
            ax.set_ylabel(LEADS[i] + "\nmV")
        axes[0].legend()
        axes[-1].set_xlabel("Seconds")
        fig.suptitle(f"ECG {sample['ecg_id']} | visible samples are not counted as reconstruction")
        fig.savefig(path.with_suffix(".png"), dpi=100)
    finally:
        plt.close(fig)


def class_metrics(y, probabilities):
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
    supported = [i for i in range(5) if len(np.unique(y[:, i])) == 2]
    return dict(macro_auroc=float(np.mean([roc_auc_score(y[:, i], probabilities[:, i]) for i in supported])) if supported else None,
                macro_auprc=float(np.mean([average_precision_score(y[:, i], probabilities[:, i]) for i in supported])) if supported else None,
                macro_f1=float(f1_score(y, probabilities >= .5, average="macro", zero_division=0)),
                supported_classes=len(supported), all_five_supported=len(supported) == 5)


@torch.no_grad()
def extract_features(model, data, spec, plan, device, split, indices, pooling, visible):
    chunks = []
    previous = model.training
    model.eval()
    try:
        for start in range(0, len(indices), plan["microbatch"]):
            ix = indices[start:start + plan["microbatch"]]
            raw = np.asarray(data["arrays"][split][ix], dtype=np.float32)
            ids = data["frames"][split].iloc[ix].ecg_id.to_numpy()
            bank = "per_lead" if "per_lead" in plan["evaluation_banks"] else "global"
            mask = masks_for_ids(ids, plan, bank) if visible else np.zeros((len(ix), 600), dtype=bool)
            x, _, _ = normalize(raw, spec["normalization"], data["stats"], mask=mask)
            x = torch.from_numpy(x).to(device)
            if spec["backend"] == "stmem":
                if pooling != "official" or visible:
                    raise ValueError("Official anchor probe supports only full-input official pooling")
                value = model.features(x)
            else:
                value = model.features(x, pooling, torch.from_numpy(mask).to(device) if visible else None)
            chunks.append(value.float().cpu())
    finally:
        model.train(previous)
    features = torch.cat(chunks)
    if features.ndim != 2 or not torch.isfinite(features).all():
        raise ValueError("Invalid/nonfinite frozen encoder features")
    return features


def probe_audit(model, initial, data, spec, plan, device, directory, make_model, *, expanded=False):
    """Fixed classifier checkpoints; no best-epoch selection on the reported set.

    Head SGD stream and initialization are shared across initial/trained encoders.
    Probe budgets are a single continuous trajectory (300 is a prefix of 1500).
    """
    directory = Path(directory)
    supported_inputs = [("official", False)] if spec["backend"] == "stmem" else (
        [(p, v) for p in ("norm_then_mean", "mean_then_norm") for v in (False, True)] if expanded else [("norm_then_mean", False)])
    rows = []
    train = np.flatnonzero(data["frames"]["train"][CLASSES].to_numpy().any(1))
    val = np.flatnonzero(data["frames"]["validation"][CLASSES].to_numpy().any(1))
    if not len(train) or not len(val):
        raise ValueError("Probe requires labeled train and validation ECGs")
    yt = torch.from_numpy(data["frames"]["train"].iloc[train][CLASSES].to_numpy(dtype=np.float32).copy())
    yv = data["frames"]["validation"].iloc[val][CLASSES].to_numpy()
    for tag in ("random", "trained"):
        encoder = model
        if tag == "random":
            encoder = make_model().to(device)
            encoder.load_state_dict(initial)
        for pooling, visible in supported_inputs:
            key = f"{tag}_{pooling}_{'visible' if visible else 'full'}"
            ft = extract_features(encoder, data, spec, plan, device, "train", train, pooling, visible)
            fv = extract_features(encoder, data, spec, plan, device, "validation", val, pooling, visible)
            stats = diagnostics.feature_statistics(ft)
            diagnostics.write_json(directory / f"{key}_features.json", stats)
            mean, scale = ft.mean(0), ft.std(0, correction=0).clamp_min(1e-6)
            ft, fv = (ft - mean) / scale, (fv - mean) / scale
            torch.manual_seed(spec["model_seed"] + 100)
            head = torch.nn.Linear(ft.shape[1], 5)
            optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
            rng = np.random.default_rng(spec["model_seed"] + 100)
            history = []
            for update in range(1, max(plan["probe_checkpoints"]) + 1):
                ix = rng.choice(len(ft), min(256, len(ft)), replace=False)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(head(ft[ix]), yt[ix])
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite linear-probe loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if update % 50 == 0 or update in plan["probe_checkpoints"]:
                    with torch.no_grad():
                        probabilities = head(fv).sigmoid().numpy()
                        train_loss = float(torch.nn.functional.binary_cross_entropy_with_logits(head(ft), yt))
                    metrics = class_metrics(yv, probabilities)
                    history.append(dict(update=update, train_loss=train_loss, **metrics))
                    if update in plan["probe_checkpoints"]:
                        rows.append(dict(tag=tag, pooling=pooling, input="visible" if visible else "full", update=update, **metrics))
                        pd.DataFrame(probabilities, columns=CLASSES).assign(
                            ecg_id=data["frames"]["validation"].iloc[val].ecg_id.to_numpy(),
                            patient_id=data["frames"]["validation"].iloc[val].patient_id.to_numpy()).to_csv(directory / f"{key}_{update}_probabilities.csv", index=False)
                        torch.save(dict(head=head.state_dict(), feature_mean=mean, feature_std=scale), directory / f"{key}_{update}_head.pt")
            pd.DataFrame(history).to_csv(directory / f"{key}_probe_history.csv", index=False)
        if tag == "random":
            del encoder
    result = pd.DataFrame(rows)
    random = result[result.tag.eq("random")].drop(columns="tag")
    trained = result[result.tag.eq("trained")].drop(columns="tag")
    paired = trained.merge(random, on=["pooling", "input", "update"], suffixes=("_trained", "_random"), validate="one_to_one")
    paired["auroc_gain_vs_random"] = paired.macro_auroc_trained - paired.macro_auroc_random
    paired.to_csv(directory / "probe_comparison.csv", index=False)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, constrained_layout=True)
    try:
        for path in directory.glob("*_probe_history.csv"):
            history = pd.read_csv(path)
            label = path.name.removesuffix("_probe_history.csv")
            axes[0].plot(history["update"], history.train_loss, label=label)
            axes[1].plot(history["update"], history.macro_auroc, label=label)
        axes[0].set_ylabel("Probe train BCE")
        axes[1].set_ylabel("Validation macro AUROC")
        axes[1].set_xlabel("Classifier update (encoder frozen)")
        axes[0].legend(fontsize=7, ncol=2)
        for ax in axes:
            ax.grid(alpha=.2)
        fig.savefig(directory / "probe_curves.png", dpi=120)
    finally:
        plt.close(fig)
    return paired.to_dict("records")
