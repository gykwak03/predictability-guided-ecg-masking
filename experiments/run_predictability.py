"""Fit Ridge/CNN/Transformer on train-only ECGs and score a common held-out mask bank.

CPU smoke checks the pipeline; GPU pilot generates exploratory difficulty groups.
No test arrays are opened. All default outputs live inside this Git repository.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time
import uuid

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import torch

from io_utils import write_json
from predictability_estimators import (ESTIMATORS, LaggedRidge, SmallCNN, SmallTransformer,
                                      difficulty_table, lead_scale, mask_bank, normalized_masked_loss,
                                      patient_pools, predictability_percentile, reconstruction_metrics)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "experiments/artifacts/predictability"
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def load_data(directory):
    """Read train/validation only, enforcing the existing preprocessing contract."""
    directory = Path(directory)
    manifest_bytes = (directory / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest["lead_order"] != LEADS or manifest["config"]["target_sampling_rate"] != 250:
        raise ValueError("Expected canonical 12 leads at 250 Hz")
    arrays, frames = {}, {}
    for split in ("train", "validation"):
        frames[split] = pd.read_csv(directory / f"metadata_{split}.csv", low_memory=False)
        arrays[split] = np.load(directory / f"signals_{split}.npy", mmap_mode="r")
        frame, x = frames[split], arrays[split]
        if x.shape != (len(frame), 12, 2500) or x.dtype != np.float32:
            raise ValueError(f"Invalid {split} signal contract")
        if not np.array_equal(frame.artifact_index.to_numpy(), np.arange(len(frame))):
            raise ValueError("Metadata rows must align with artifact_index")
        if frame.ecg_id.duplicated().any() or frame.patient_id.isna().any():
            raise ValueError("Unique ECG IDs and nonmissing patients required")
        allowed = set(range(1, 9)) if split == "train" else {9}
        if not set(frame.strat_fold).issubset(allowed):
            raise ValueError(f"Unexpected strat_fold in {split}")
    if set(frames["train"].patient_id) & set(frames["validation"].patient_id):
        raise ValueError("Train/validation patient leakage")
    if set(frames["train"].ecg_id) & set(frames["validation"].ecg_id):
        raise ValueError("Train/validation ECG leakage")
    return arrays, frames, manifest, hashlib.sha256(manifest_bytes).hexdigest()


def evaluate(estimator, signals, bank, masks, scale, batch_size, device, neural=False):
    rows, patches = [], []
    if neural:
        estimator.eval()
    with torch.no_grad():
        for start in range(0, len(bank), batch_size):
            meta = bank.iloc[start:start+batch_size]
            x = np.asarray(signals[meta.artifact_index.to_numpy()], dtype=np.float32)
            m = masks[start:start+len(meta)]
            if neural:
                prediction = estimator(torch.from_numpy(x).to(device), torch.from_numpy(m).to(device)).cpu().numpy()
            else:
                prediction = estimator.predict(x, m)
            metrics, patch = reconstruction_metrics(prediction, x, m, scale)
            metrics.insert(0, "mask_id", meta.mask_id.to_numpy())
            rows.append(metrics)
            patches.append(patch.astype(np.float32))
    return pd.concat(rows, ignore_index=True), np.concatenate(patches)


def aggregate_metrics(metrics):
    """Hidden-sample-weighted RMS; do not average case RMS values for pooled loss."""
    weights = metrics.masked_samples.to_numpy()
    mse = np.average(np.square(metrics.nrmse), weights=weights)
    zero = np.average(np.square(metrics.zero_nrmse), weights=weights)
    return dict(masks=len(metrics), pooled_nrmse=float(np.sqrt(mse)), zero_pooled_nrmse=float(np.sqrt(zero)),
                skill_vs_zero=float(1-mse/zero) if zero > 0 else np.nan,
                mean_case_nrmse=float(metrics.nrmse.mean()),
                mean_case_pearson=float(metrics.masked_pearson.mean()),
                mean_case_r2=float(metrics.masked_r2.mean()))


def train_neural(name, model, arrays, banks, masks, scale, profile, directory, seed, device):
    """CNN and Transformer sample identical train mask IDs in identical batch order."""
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed+7000)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=profile["lr"], weight_decay=1e-4)
    scale_tensor = torch.as_tensor(scale, dtype=torch.float32, device=device)
    history, first_batch_digest = [], None
    sequence_digest = hashlib.sha256()
    started = time.monotonic()

    def snapshot(step, train_loss=np.nan):
        val, _ = evaluate(model, arrays["validation"], banks["validation"], masks["validation"],
                          scale, profile["batch_size"], device, neural=True)
        history.append(dict(estimator=name, step=step, train_normalized_mse=train_loss,
                            elapsed_seconds=time.monotonic()-started, **aggregate_metrics(val)))
        pd.DataFrame(history).to_csv(directory / f"{name}_learning_history.csv", index=False)
        print(f"{name} {step}/{profile['steps']}: validation NRMSE={history[-1]['pooled_nrmse']:.5f}, "
              f"skill={history[-1]['skill_vs_zero']:.2%}", flush=True)

    snapshot(0)
    train = banks["fit"]
    interval = max(1, profile["steps"]//10)
    for step in range(1, profile["steps"]+1):
        rows = rng.integers(0, len(train), size=profile["batch_size"])
        sequence_digest.update(np.asarray(rows, dtype="<i8").tobytes())
        if first_batch_digest is None:
            first_batch_digest = hashlib.sha256("\n".join(train.iloc[rows].mask_id).encode()).hexdigest()
        x = torch.from_numpy(np.asarray(arrays["train"][train.iloc[rows].artifact_index.to_numpy()], dtype=np.float32)).to(device)
        m = torch.from_numpy(masks["fit"][rows]).to(device)
        # Fixed training budget and schedule; validation never selects a checkpoint.
        warmup = max(1, min(100, profile["steps"]//10))
        factor = step/warmup if step <= warmup else 0.1+0.9*(1+np.cos(np.pi*(step-warmup)/max(1, profile["steps"]-warmup)))/2
        optimizer.param_groups[0]["lr"] = profile["lr"]*factor
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x, m)
        loss = normalized_masked_loss(prediction, x, m, scale_tensor)
        if not torch.isfinite(loss):
            raise RuntimeError(f"{name}: nonfinite training loss at {step}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(norm):
            raise RuntimeError(f"{name}: nonfinite gradients at {step}")
        optimizer.step()
        if step % interval == 0 or step == profile["steps"]:
            snapshot(step, float(loss.detach().cpu()))
    torch.save(dict(model=model.state_dict(), estimator=name, profile=profile, fixed_train_std=scale,
                    first_train_batch_sha256=first_batch_digest, train_sequence_sha256=sequence_digest.hexdigest()),
               directory / f"{name}.pt")
    return pd.DataFrame(history), dict(first_batch=first_batch_digest, full_sequence=sequence_digest.hexdigest())


def make_maps(patch_errors, bank, directory, patch_samples):
    """Mean patch predictability conditional on that lead/time patch being hidden.

    Patch percentiles are calibrated over all hidden patch observations per
    estimator, separately from whole-mask percentiles used for difficulty groups.
    No visible copied value contributes to a map.
    """
    q = 2500//patch_samples
    maps, examples, rows = [], [], []
    finite_pattern = np.isfinite(patch_errors["ridge"])
    coverage = finite_pattern.sum(axis=0)
    for name in ESTIMATORS:
        errors = patch_errors[name]
        if errors.shape != finite_pattern.shape or not np.array_equal(np.isfinite(errors), finite_pattern):
            raise ValueError("Estimator patch coverage mismatch")
        percentiles = np.full(errors.shape, np.nan)
        percentiles[finite_pattern] = predictability_percentile(errors[finite_pattern])
        total = np.nansum(percentiles, axis=0)
        means = np.divide(total, coverage, out=np.full((12, q), np.nan), where=coverage > 0)
        nrmse = np.divide(np.nansum(errors, axis=0), coverage, out=np.full((12, q), np.nan), where=coverage > 0)
        maps.append(means)
        examples.append(percentiles[0])
        for c, lead in enumerate(LEADS):
            for t in range(q):
                rows.append(dict(estimator=name, lead=lead, patch_index=t,
                                 start_seconds=t*patch_samples/250, end_seconds=(t+1)*patch_samples/250,
                                 hidden_observations=int(coverage[c, t]), mean_patch_nrmse=nrmse[c, t],
                                 mean_patch_predictability=means[c, t]))
    # Here consensus is a map summary, not the case-level difficulty statistic.
    consensus = np.mean(np.stack(maps), axis=0)
    maps.append(consensus)
    examples.append(np.mean(np.stack(examples), axis=0))
    for c, lead in enumerate(LEADS):
        for t in range(q):
            rows.append(dict(estimator="consensus", lead=lead, patch_index=t,
                             start_seconds=t*patch_samples/250, end_seconds=(t+1)*patch_samples/250,
                             hidden_observations=int(coverage[c, t]), mean_patch_nrmse=np.nan,
                             mean_patch_predictability=consensus[c, t]))
    pd.DataFrame(rows).to_csv(directory / "lead_time_predictability.csv", index=False)
    for values, filename, title in (
        (maps, "lead_time_predictability_curves.png", "Held-out train masks: mean patch predictability (higher = easier)"),
        (examples, "example_predictability_curves.png", f"Mask {bank.iloc[0].mask_id}: hidden patches only"),
    ):
        fig, axes = plt.subplots(4, 1, figsize=(12, 12), constrained_layout=True)
        for ax, matrix, name in zip(axes, values, (*ESTIMATORS, "consensus")):
            im = ax.imshow(matrix, aspect="auto", extent=(0, 10, 11.5, -0.5), vmin=0, vmax=100, cmap="viridis")
            ax.set_yticks(range(12), LEADS)
            ax.set_title(name)
            ax.set_xlabel("Time (s)")
        fig.colorbar(im, ax=axes, label="Predictability percentile")
        fig.suptitle(title)
        fig.savefig(directory / filename, dpi=130)
        plt.close(fig)


def save_difficulty(case_metrics, bank, directory, frames, patch_errors, masks, patch_samples):
    difficulty, correlations = difficulty_table(case_metrics)
    difficulty = difficulty.merge(bank, on="mask_id", validate="one_to_one")
    # Labels are descriptive audits only: never features, fitting targets or ranks.
    classes = [c for c in ("NORM", "MI", "STTC", "CD", "HYP") if c in frames["train"]]
    labels = frames["train"][["ecg_id", *classes]]
    difficulty = difficulty.merge(labels, on="ecg_id", validate="many_to_one")
    difficulty.to_csv(directory / "mask_difficulty.csv", index=False)
    correlations.to_csv(directory / "spearman_correlations.csv", index=False)
    case_metrics.to_csv(directory / "mask_metrics.csv", index=False)
    group_rows = []
    for name in ("Easy", "Medium", "Hard"):
        group = difficulty[difficulty.difficulty == name]
        unique = group.drop_duplicates("ecg_id")
        record = dict(difficulty=name, masks=len(group), ecgs=group.ecg_id.nunique(),
                      patients=group.patient_id.nunique(), mean_consensus=group.consensus_predictability.mean(),
                      min_consensus=group.consensus_predictability.min(), max_consensus=group.consensus_predictability.max(),
                      mean_disagreement=group.disagreement_std.mean())
        for c in classes:
            record[f"{c}_mask_prevalence"] = group[c].mean()
            record[f"{c}_ecg_prevalence"] = unique[c].mean()
        group_rows.append(record)
        group[["mask_id", "ecg_id", "artifact_index", "repeat", "mask_seed", "mask_sha256",
               "consensus_predictability", "difficulty"]].to_csv(directory / f"{name.lower()}_masks.csv", index=False)
    pd.DataFrame(group_rows).to_csv(directory / "difficulty_groups.csv", index=False)
    counts = pd.crosstab(difficulty.ecg_id, difficulty.difficulty).reindex(columns=["Easy", "Medium", "Hard"], fill_value=0)
    counts["present_in_all_groups"] = (counts > 0).all(axis=1)
    counts.to_csv(directory / "ecg_group_coverage.csv")
    lead_rows = []
    for group in ("Easy", "Medium", "Hard"):
        ids = set(difficulty.loc[difficulty.difficulty == group, "mask_id"])
        selected = np.array([m in ids for m in bank.mask_id])
        counts_hidden = masks[selected].reshape(-1, 12, 2500//patch_samples).sum(axis=(0, 2))
        for lead, count in zip(LEADS, counts_hidden):
            lead_rows.append(dict(difficulty=group, lead=lead, hidden_patches=int(count)))
    pd.DataFrame(lead_rows).to_csv(directory / "difficulty_lead_coverage.csv", index=False)
    write_json(directory / "difficulty_definition.json", dict(
        ranking_population="held-out train ECG-mask cases only", high_predictability="easier",
        percentile="100 * (N - average_ascending_NRMSE_rank + 0.5) / N",
        consensus="equal-weight mean of ridge/cnn/transformer predictability percentiles",
        disagreement_std="population std of the three percentiles; units: percentile points",
        disagreement_range="max minus min of the three percentiles",
        groups="sort consensus descending, mask_id ascending; array_split into Easy/Medium/Hard",
        tied_consensus_masks=int(difficulty.consensus_predictability.duplicated(keep=False).sum()),
        tie_policy="deterministic mask_id order; tied cases may straddle boundaries, with no evidence of different difficulty",
        ecgs_present_in_all_groups=int(counts.present_in_all_groups.sum()), total_score_ecgs=len(counts),
        sampling_warning="Global tertiles do not guarantee the same ECG/class/lead distribution in all groups.",
        spearman_inference="Descriptive rho only: masks from one ECG are dependent; no naive per-mask p-values.",
        status="exploratory single-seed groups; inspect estimator learning/zero baseline before downstream MAE comparison"))
    make_maps(patch_errors, bank, directory, patch_samples)
    return difficulty


def run(args):
    profile = dict(fit_ecgs=8, score_ecgs=6, validation_ecgs=6, train_masks=2, score_masks=2,
                   steps=10, batch_size=2, cnn_width=16, transformer_width=32, transformer_depth=1,
                   lr=1e-3, ridge_samples=16) if args.mode == "smoke" else dict(
                       fit_ecgs=5000, score_ecgs=512, validation_ecgs=128, train_masks=4, score_masks=8,
                       steps=3000, batch_size=16, cnn_width=64, transformer_width=128, transformer_depth=4,
                       lr=3e-4, ridge_samples=32)
    for key in profile:
        value = getattr(args, key, None)
        if value is not None:
            profile[key] = value
    if args.seed < 0 or args.subset_seed < 0 or args.mask_seed < 0:
        raise ValueError("Seeds must be nonnegative")
    for key, value in profile.items():
        if value < 0 or (value == 0 and key not in ("fit_ecgs", "score_ecgs", "validation_ecgs")):
            raise ValueError(f"Invalid {key}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.mode == "pilot" and device.type != "cuda":
        raise RuntimeError("GPU pilot requires a Slurm GPU allocation. Use --mode smoke on CPU.")
    torch.set_num_threads(args.cpu_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    arrays, frames, manifest, source_hash = load_data(args.artifact_dir)
    fit, score = patient_pools(frames["train"], profile["fit_ecgs"], profile["score_ecgs"], args.subset_seed)
    validation = np.random.default_rng(args.subset_seed+9000).permutation(len(frames["validation"]))
    if profile["validation_ecgs"] > len(validation):
        raise ValueError("Requested validation count exceeds split")
    validation = np.sort(validation[:profile["validation_ecgs"]] if profile["validation_ecgs"] else validation)
    mean, scale = lead_scale(arrays["train"], fit)
    banks, masks = {}, {}
    for role, indices, repeats, split in (("fit", fit, profile["train_masks"], "train"),
                                         ("score", score, profile["score_masks"], "train"),
                                         ("validation", validation, 2, "validation")):
        banks[role], masks[role] = mask_bank(frames[split], indices, args.patch_samples, 2500,
                                            args.mask_ratio, repeats, args.mask_seed, role)
    output = Path(args.output_root).resolve() / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
              + f"_{args.mode}_seed{args.seed}_" + uuid.uuid4().hex[:8])
    output.mkdir(parents=True)
    for role in banks:
        banks[role].to_csv(output / f"{role}_mask_bank.csv", index=False)
    selection = []
    for role, indices, split in (("fit", fit, "train"), ("score", score, "train"), ("validation", validation, "validation")):
        f = frames[split].iloc[indices][["artifact_index", "ecg_id", "patient_id", "strat_fold"]].copy()
        f["role"], f["source_split"] = role, split
        selection.append(f)
    pd.concat(selection).to_csv(output / "ecg_selection.csv", index=False)
    write_json(output / "run_config.json", dict(
        schema_version=1, mode=args.mode, seed=args.seed, subset_seed=args.subset_seed, mask_seed=args.mask_seed,
        patch_samples=args.patch_samples, patch_seconds=args.patch_samples/250, mask_ratio=args.mask_ratio,
        realized_mask_ratio=float(masks["score"].sum(axis=1)[0]/masks["score"].shape[1]), profile=profile,
        artifact_dir=str(Path(args.artifact_dir).resolve()), source_manifest_sha256=source_hash,
        preprocessing=manifest["config"], lead_order=LEADS, fixed_train_lead_mean=mean, fixed_train_lead_std=scale,
        normalization_units="preprocessed input units (per-record lead z scores in current artifacts); not mV",
        split_policy="80/20 patient split within original train folds 1-8; estimator-fit and score patients disjoint",
        std_fit_policy="only selected estimator-fit ECGs; population std over ECG x time for each lead",
        test_opened=False, torch_version=torch.__version__, numpy_version=np.__version__,
        pandas_version=pd.__version__, scipy_version=scipy.__version__, device=str(device),
        default_condition_reason="P50/r50: ST-MEM MSE 0.3913/AUROC 0.8068; original MAE AUROC 0.7752, near its grid maximum 0.7799. Exploratory choice, not a proven optimum.",
        score_bank_policy="finite reproducible random lead-time masks; uniform global exact hidden-token count",
        training_policy="same fit ECG-mask bank and neural batch order; Ridge subsamples time targets and fits linear least squares",
        ridge_alpha=args.ridge_alpha, ridge_objective="mean masked target-lead MSE + alpha*sum(non-intercept weights squared)",
        transformer_policy="global bidirectional lead-time attention, raw zero-filled patch + visibility flag, fixed sinusoidal time and learned lead embeddings",
        cnn_policy="12 leads as channels, zero-filled values + sample visibility, four residual dilated temporal convolution blocks",
        evaluation_policy="final checkpoint only; validation is diagnostics, excluded from difficulty ranking"))
    print(f"Output: {output}\nCondition: P={args.patch_samples}, mask={args.mask_ratio:.0%} | {device}\n"
          f"Estimator-fit ECGs={len(fit)}, held-out train ECGs={len(score)}, score masks={len(banks['score'])}", flush=True)
    started = time.monotonic()
    results, summaries, patch_errors, histories = [], [], {}, []
    ridge = LaggedRidge(args.patch_samples, args.ridge_alpha)
    print("Fitting masked-only temporal/cross-lead Ridge", flush=True)
    ridge.fit(arrays["train"], banks["fit"], masks["fit"], scale,
              samples_per_ecg=profile["ridge_samples"], batch_size=profile["batch_size"], seed=args.seed)
    np.savez_compressed(output / "ridge_weights.npz", weights=ridge.weights, scale=scale, lags=ridge.lags)
    for split, source in (("score", "train"), ("validation", "validation")):
        metrics, patch = evaluate(ridge, arrays[source], banks[split], masks[split], scale, profile["batch_size"], device)
        summaries.append(dict(estimator="ridge", split=split, parameters=int(ridge.weights.size), **aggregate_metrics(metrics)))
        metrics["estimator"] = "ridge"
        if split == "score":
            results.append(metrics)
            patch_errors["ridge"] = patch
        else:
            metrics.to_csv(output / "ridge_validation_metrics.csv", index=False)
    digests = {}
    for name in ("cnn", "transformer"):
        torch.manual_seed(args.seed)
        model = SmallCNN(width=profile["cnn_width"]) if name == "cnn" else SmallTransformer(
            patch_samples=args.patch_samples, width=profile["transformer_width"], depth=profile["transformer_depth"])
        params = sum(p.numel() for p in model.parameters())
        print(f"Training {name}: {params:,} parameters", flush=True)
        history, digests[name] = train_neural(name, model, arrays, banks, masks, scale, profile, output, args.seed, device)
        histories.append(history)
        metrics, patch_errors[name] = evaluate(model, arrays["train"], banks["score"], masks["score"], scale,
                                              profile["batch_size"], device, neural=True)
        summaries.append(dict(estimator=name, split="score", parameters=params, **aggregate_metrics(metrics)))
        summaries.append(dict(estimator=name, split="validation", parameters=params,
                              **{k: history.iloc[-1][k] for k in aggregate_metrics(metrics)}))
        metrics["estimator"] = name
        results.append(metrics)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if digests["cnn"] != digests["transformer"]:
        raise RuntimeError("Neural training batch contract mismatch")
    np.savez_compressed(output / "score_patch_nrmse.npz", **patch_errors)
    np.savez_compressed(output / "mask_bank_bits.npz", **{k: np.packbits(v, axis=1) for k, v in masks.items()})
    difficulty = save_difficulty(pd.concat(results, ignore_index=True), banks["score"], output, frames,
                                 patch_errors, masks["score"], args.patch_samples)
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "estimator_summary.csv", index=False)
    learning = pd.concat(histories, ignore_index=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for name, group in learning.groupby("estimator"):
        axes[0].plot(group.step, group.pooled_nrmse, label=name)
        axes[1].plot(group.step, group.skill_vs_zero*100, label=name)
    axes[0].set_ylabel("Validation NRMSE")
    axes[1].set_ylabel("Validation improvement vs zero (%)")
    for ax in axes:
        ax.set_xlabel("Optimizer update")
        ax.legend()
        ax.grid(alpha=0.2)
    fig.savefig(output / "estimator_learning_curves.png", dpi=140)
    plt.close(fig)
    write_json(output / "completion.json", dict(status="completed", seconds=time.monotonic()-started,
        score_masks=len(difficulty), difficulty_counts=difficulty.difficulty.value_counts().to_dict(),
        neural_first_batch_sha256={k: v["first_batch"] for k, v in digests.items()},
        neural_train_sequence_sha256={k: v["full_sequence"] for k, v in digests.items()}, test_opened=False,
        estimator_quality=summary[summary.split == "score"].to_dict("records"),
        interpretation="Smoke groups check code only. Pilot groups require estimator learning review and seed/budget robustness checks before research claims."))
    print(summary.to_string(index=False), flush=True)
    print(f"Completed predictability. Output: {output}", flush=True)
    return output


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["smoke", "pilot"], default="smoke")
    p.add_argument("--train", action="store_true", help="Accepted for submit_pilot.sh; this pipeline always fits all three estimators")
    p.add_argument("--artifact-dir", type=Path, default=ROOT / "preprocessing/artifacts/ptbxl_250hz")
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--patch-samples", type=int, choices=[50, 125, 250], default=50)
    p.add_argument("--mask-ratio", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--subset-seed", type=int, default=42)
    p.add_argument("--mask-seed", type=int, default=42)
    for option in ("fit-ecgs", "score-ecgs", "validation-ecgs", "train-masks", "score-masks", "steps", "batch-size", "ridge-samples"):
        p.add_argument("--"+option, type=int, help="Override mode profile; ECG counts: 0 = all reserved pool ECGs")
    p.add_argument("--lr", type=float)
    p.add_argument("--ridge-alpha", type=float, default=1e-3)
    p.add_argument("--cpu-threads", type=int, default=4)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    if not 0 < args.mask_ratio < 1 or args.cpu_threads < 1:
        raise SystemExit("Need 0 < mask-ratio < 1 and positive cpu-threads")
    run(args)
