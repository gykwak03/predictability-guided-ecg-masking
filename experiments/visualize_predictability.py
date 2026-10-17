"""Visualize completed predictability scores on actual ECG waveforms.

Uses original per-patch cache when available. Otherwise deterministically replays
only the linear Ridge fit and verifies case NRMSE against the server CSV. Never
substitutes replayed Ridge errors for missing CNN/Transformer patch errors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import torch

from masking_diagnostics import write_json
from predictability_estimators import ESTIMATORS, LaggedRidge, masks_from_bank, predictability_percentile
from run_predictability import ROOT, LEADS, evaluate, load_data


def display_values(values, decimals=3):
    a = np.asarray(values)
    if a.ndim > 1:
        return [display_values(row, decimals) for row in a]
    return [round(float(x), decimals) if np.isfinite(x) else None for x in a]


def build(run, artifact_dir, output):
    run, output = Path(run).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((run / "run_config.json").read_text())
    condition_hash = hashlib.sha256((run / "run_config.json").read_bytes()).hexdigest()
    arrays, frames, _, source_hash = load_data(artifact_dir)
    if source_hash != config["source_manifest_sha256"]:
        raise ValueError("Local preprocessing manifest differs from the scored run")
    bank = pd.read_csv(run / "score_mask_bank.csv")
    difficulty = pd.read_csv(run / "mask_difficulty.csv").set_index("mask_id")
    metrics = pd.read_csv(run / "mask_metrics.csv")
    p = config["patch_samples"]
    q, tokens = 2500//p, 12*(2500//p)
    masks = masks_from_bank(bank, tokens)
    scale = np.asarray(config["fixed_train_lead_std"])
    weights_path = output / "replayed_ridge_weights.npz"
    ridge = LaggedRidge(p, config["ridge_alpha"])
    original_weights = run / "ridge_weights.npz"
    if original_weights.exists():
        with np.load(original_weights) as saved:
            ridge.weights, ridge.scale = saved["weights"].copy(), saved["scale"].copy()
            ridge.lags = tuple(saved["lags"])
        provenance = "original server ridge weights"
    elif weights_path.exists():
        with np.load(weights_path) as saved:
            if str(saved["condition_hash"]) != condition_hash:
                raise ValueError("Cached replay belongs to a different source run")
            ridge.weights, ridge.scale = saved["weights"].copy(), saved["scale"].copy()
            ridge.lags = tuple(saved["lags"])
        provenance = "cached deterministic replay of original ridge protocol"
    else:
        print("Replaying train-only Ridge; neural estimators are not retrained.", flush=True)
        fit = pd.read_csv(run / "fit_mask_bank.csv")
        fit_masks = masks_from_bank(fit, tokens)
        ridge.fit(arrays["train"], fit, fit_masks, scale,
                  samples_per_ecg=config["profile"]["ridge_samples"],
                  batch_size=config["profile"]["batch_size"], seed=config["seed"])
        np.savez_compressed(weights_path, weights=ridge.weights, scale=ridge.scale,
                            lags=ridge.lags, condition_hash=condition_hash)
        provenance = "deterministic replay of original ridge protocol"
    print("Scoring original 4,096 masks to verify Ridge replay.", flush=True)
    case_metrics, ridge_patch = evaluate(ridge, arrays["train"], bank, masks, scale, 16, torch.device("cpu"))
    expected = metrics[metrics.estimator == "ridge"].set_index("mask_id").loc[bank.mask_id]
    difference = float(np.abs(expected.nrmse.to_numpy()-case_metrics.nrmse.to_numpy()).max())
    if difference > 1e-5:
        raise ValueError(f"Ridge replay does not reproduce original case NRMSE: max error {difference}")
    cache_path = run / "score_patch_nrmse.npz"
    errors = {"ridge": ridge_patch}
    if cache_path.exists():
        with np.load(cache_path) as saved:
            if set(saved.files) != set(ESTIMATORS):
                raise ValueError("Original cache needs all three estimator arrays")
            for name in ESTIMATORS:
                a = saved[name].copy()
                if a.shape != (len(bank), 12, q) or not np.array_equal(np.isfinite(a), masks.reshape(len(bank), 12, q)):
                    raise ValueError("Original patch cache coverage/order mismatch")
                pooled = np.sqrt(np.nanmean(np.square(a.astype(np.float64)), axis=(1, 2)))
                recorded = metrics[metrics.estimator == name].set_index("mask_id").loc[bank.mask_id].nrmse.to_numpy()
                if np.max(np.abs(pooled-recorded)) > 1e-5:
                    raise ValueError(f"Original {name} patch cache does not match recorded case scores")
                errors[name] = a
    percentiles = {}
    for name, a in errors.items():
        valid = np.isfinite(a)
        ranks = np.full(a.shape, np.nan, np.float32)
        ranks[valid] = predictability_percentile(a[valid])
        percentiles[name] = ranks
    # Named extremes are examples, not a representative sample of the population.
    selected = [difficulty.consensus_predictability.idxmax(), difficulty.consensus_predictability.idxmin()]
    if "score-e71-m0" in difficulty.index:
        selected.insert(0, "score-e71-m0")
        harder_same_ecg = difficulty[difficulty.ecg_id == 71].consensus_predictability.idxmin()
        selected.insert(1, harder_same_ecg)
    selected = list(dict.fromkeys(selected))
    positions = {m: i for i, m in enumerate(bank.mask_id)}
    case_data, wave_data, patch_rows = [], {}, []
    for mask_id in selected:
        i = positions[mask_id]
        row = difficulty.loc[mask_id]
        ecg_id, index = int(row.ecg_id), int(row.artifact_index)
        x = np.asarray(arrays["train"][[index]], dtype=np.float32)
        pred = ridge.predict(x, masks[[i]])[0]
        target_rms = np.sqrt(np.square(x[0].astype(np.float64)/scale[:, None]).reshape(12, q, p).mean(axis=-1))
        wave_data[str(ecg_id)] = display_values(x[0, :, ::2])
        hidden_samples = np.repeat(masks[i].reshape(12, q), p, axis=1)
        prediction = np.where(hidden_samples, pred, np.nan)
        case = dict(mask_id=mask_id, ecg_id=ecg_id, difficulty=row.difficulty,
                    consensus=round(float(row.consensus_predictability), 2),
                    labels=[c for c in ("NORM", "MI", "STTC", "CD", "HYP") if row[c] == 1],
                    case_nrmse={name: float(row[f"{name}_nrmse"]) for name in ESTIMATORS},
                    ridge_prediction=display_values(prediction[:, ::2]),
                    patch_nrmse={name: display_values(a[i], 4) for name, a in errors.items()},
                    predictability={name: display_values(a[i], 1) for name, a in percentiles.items()},
                    target_rms=display_values(target_rms, 4))
        case_data.append(case)
        for name, values in errors.items():
            for c, lead in enumerate(LEADS):
                for t in np.flatnonzero(masks[i].reshape(12, q)[c]):
                    patch_rows.append(dict(mask_id=mask_id, ecg_id=ecg_id, estimator=name, lead=lead,
                                           patch_index=int(t), start_seconds=t*p/250, end_seconds=(t+1)*p/250,
                                           patch_nrmse=float(values[i, c, t]), predictability=float(percentiles[name][i, c, t]),
                                           target_rms=float(target_rms[c, t]),
                                           same_time_visible_other_leads=int((~masks[i].reshape(12, q)[:, t]).sum()),
                                           left_same_lead_visible=bool(t > 0 and not masks[i].reshape(12, q)[c, t-1]),
                                           right_same_lead_visible=bool(t+1 < q and not masks[i].reshape(12, q)[c, t+1])))
    patch_table = pd.DataFrame(patch_rows)
    patch_table.to_csv(output / "selected_patch_details.csv", index=False)
    difficulty.loc[selected].reset_index().to_csv(output / "selected_case_scores.csv", index=False)
    write_json(output / "ridge_replay_audit.json", dict(
        source_run=str(run), source_manifest_match=True, ridge_provenance=provenance,
        max_case_nrmse_difference_vs_server=difference, tolerance=1e-5, original_neural_patch_cache_available=cache_path.exists(),
        available_patch_estimators=list(errors), original_sample_rate_hz=250, display_sample_rate_hz=125,
        selected_mask_ids=selected, selection_policy="saved example ECG 71 and its relatively harder mask; global highest/lowest consensus masks",
        percentile_population="all hidden patch observations per estimator, not whole-mask ranks",
        scientific_limit="ECG labels are dataset annotations; no diagnostic inference or causal attribution to a waveform feature"))
    data = dict(leads=LEADS, patch_seconds=p/250, display_fs=125, sample_rate=250, cases=case_data,
                waves=wave_data, available_models=list(errors), score_ecgs=512, score_masks=len(bank))
    write_json(output / "explorer_data.json", data)

    # Export the global metric interpretation as a scientific figure.
    summary = pd.read_csv(run / "estimator_summary.csv").query("split == 'score'").set_index("estimator")
    labels = ["Zero prediction", "Ridge", "1D CNN", "Small Transformer"]
    residual = [100, *[100*(1-summary.loc[name, "skill_vs_zero"]) for name in ESTIMATORS]]
    fig, ax = plt.subplots(figsize=(8, 3.3), constrained_layout=True)
    ax.barh(labels[::-1], residual[::-1], color=["#d47a45", "#228e88", "#5473a7", "#aaaaaa"])
    for label, value in zip(labels, residual):
        ax.text(value+1, label, f"{value:.1f}", va="center")
    ax.set_xlim(0, 113)
    ax.set_xlabel("Remaining masked MSE (zero prediction = 100)")
    ax.set_title("Unseen ECGs: how much squared reconstruction error remains?")
    ax.grid(axis="x", alpha=.2)
    fig.savefig(output / "remaining_error_curves.png", dpi=160)
    plt.close(fig)

    # All three estimator maps can be plotted only when the original cache exists.
    for case in case_data:
        fig, axes = plt.subplots(len(errors), 1, figsize=(11, 3.2*len(errors)), squeeze=False, constrained_layout=True)
        for ax, name in zip(axes.flat, errors):
            matrix = np.array(case["predictability"][name], dtype=float)
            im = ax.imshow(matrix, aspect="auto", extent=(0, 10, 11.5, -.5), cmap="RdYlGn", vmin=0, vmax=100)
            ax.set_yticks(range(12), LEADS)
            ax.set_xlabel("Time (s)")
            ax.set_title(f"{name}: ECG {case['ecg_id']} / {case['mask_id']} / whole-mask group {case['difficulty']}")
        fig.colorbar(im, ax=list(axes.flat), label="Patch predictability: red = relatively hard; green = relatively easy")
        fig.savefig(output / f"{case['mask_id']}_patch_curves.png", dpi=150)
        plt.close(fig)

    # Same ECG and same mask: compare genuinely easy/hard target patches.
    reference = "score-e71-m0" if "score-e71-m0" in selected else selected[0]
    cells = patch_table[(patch_table.mask_id == reference)&(patch_table.estimator == "ridge")]
    hard = cells.sort_values("patch_nrmse", ascending=False).iloc[0]
    candidates = cells[cells.target_rms >= .2]
    easy = (candidates if len(candidates) else cells).sort_values("patch_nrmse").iloc[0]
    examples = pd.DataFrame([easy, hard])
    examples.to_csv(output / "ridge_waveform_examples.csv", index=False)
    i = positions[reference]
    original = np.asarray(arrays["train"][[int(bank.iloc[i].artifact_index)]], dtype=np.float32)
    prediction = ridge.predict(original, masks[[i]])[0]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), constrained_layout=True)
    for ax, (_, example), title, color in zip(axes, examples.iterrows(), ("Relatively easy target patch", "Relatively hard target patch"), ("#279567", "#c45f48")):
        c = LEADS.index(example.lead)
        start, end = example.start_seconds, example.end_seconds
        left, right = max(0, start-.8), min(10, end+.8)
        times = np.arange(2500)/250
        selected_time = (times >= left)&(times < right)
        ax.plot(times[selected_time], original[0, c, selected_time], color="#333333", linewidth=1.3, label="Original target waveform")
        samples = np.repeat(masks[i].reshape(12, q)[c], p)
        y = np.where(samples, prediction[c], np.nan)
        ax.plot(times[selected_time], y[selected_time], color="#3476b5", linewidth=1.1, label="Ridge reconstruction (hidden samples only)")
        ax.axvspan(start, end, color=color, alpha=.2)
        ax.set_title(f"{title}: ECG {int(example.ecg_id)}, lead {example.lead}, {start:.1f}-{end:.1f}s | NRMSE {example.patch_nrmse:.3f}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Preprocessed amplitude (z score)")
        ax.legend(loc="upper right")
        ax.grid(alpha=.2)
    fig.savefig(output / "ridge_local_waveforms_curves.png", dpi=160)
    plt.close(fig)
    print(examples.to_string(index=False), flush=True)
    print("Selected cases:", flush=True)
    print(difficulty.loc[selected, ["ecg_id", "difficulty", "consensus_predictability", "ridge_nrmse", "cnn_nrmse", "transformer_nrmse"]].to_string(), flush=True)
    print(f"Ridge max error vs server = {difference:.3g}; original cache available = {cache_path.exists()}", flush=True)
    print(f"Output: {output}", flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "preprocessing/artifacts/ptbxl_250hz")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    out = args.output_dir or ROOT / "experiments/artifacts/predictability_visuals" / args.run.name
    with threadpool_limits(limits=4):
        build(args.run, args.artifact_dir, out)
