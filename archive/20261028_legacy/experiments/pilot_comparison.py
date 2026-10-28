"""Summarize comparable runs; seed standard deviations are not confidence intervals."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = ["macro_auroc", "macro_auprc", "macro_f1",
           "random_encoder_macro_auroc", "random_encoder_macro_auprc",
           "auroc_gain_vs_random", "auprc_gain_vs_random",
           "validation_masked_mse", "validation_zero_masked_mse",
           "validation_skill_vs_zero", "validation_rms_ratio",
           "validation_masked_correlation", "train_seconds", "gpu_peak_allocated_mb"]


def comparison_signature(config):
    # Subsets and evaluation masks must be identical; model/training RNG may differ.
    keys = ["mode", "profile", "class_order", "lead_order", "source_manifest_sha256",
            "train_ecg_ids", "validation_ecg_ids", "reconstruction_validation_ecg_ids",
            "pretrain_optimizer", "probe_optimizer", "diagnostics_config", "policy", "pooling", "threshold"]
    signature = {key: config.get(key) for key in keys}
    signature["evaluation_mask_seed"] = config.get("evaluation_mask_seed", config["seed"])
    signature["diagnostic_source_sha256"] = config.get("diagnostic_source_sha256")
    return signature


def summarize_runs(run_dirs, output_dir, expected_seeds=None):
    frames, signature = [], None
    for directory in map(Path, run_dirs):
        config = json.loads((directory / "run_config.json").read_text(encoding="utf-8"))
        current = comparison_signature(config)
        if signature is not None and current != signature:
            raise ValueError("Runs have different subsets, evaluation masks or training/evaluation settings. Do not pool them.")
        signature = current
        frame = pd.read_csv(directory / "results.csv")
        if not frame.seed.eq(config["seed"]).all() or not frame.steps.eq(config["profile"]["steps"]).all():
            raise ValueError(f"Results disagree with run config: {directory}")
        frame["run_dir"] = str(directory.resolve())
        frames.append(frame)
    if not frames:
        raise ValueError("At least one completed run is required")
    combined = pd.concat(frames, ignore_index=True)
    if combined.duplicated(["patch_samples", "mask_ratio", "seed"]).any():
        raise ValueError("Duplicate condition/seed results; repeated runs are not independent seeds")
    metrics = [key for key in METRICS if key in combined]
    summary = combined.groupby(["patch_samples", "mask_ratio"])[metrics].agg(["count", "mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary = summary.reset_index()
    requested = sorted(set(expected_seeds if expected_seeds is not None else combined.seed.tolist()))
    groups = combined.groupby(["patch_samples", "mask_ratio"])
    summary["seed_count"] = [groups.get_group((row.patch_samples, row.mask_ratio)).seed.nunique()
                             for row in summary.itertuples()]
    summary["complete_requested_seeds"] = [set(groups.get_group((row.patch_samples, row.mask_ratio)).seed) == set(requested)
                                           for row in summary.itertuples()]
    summary = summary.sort_values("macro_auroc_mean", ascending=False)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_dir / "seed_results.csv", index=False)
    summary.to_csv(output_dir / "condition_summary.csv", index=False)
    (output_dir / "comparison_config.json").write_text(json.dumps(dict(
        expected_seeds=requested, completed_runs=[str(Path(p).resolve()) for p in run_dirs],
        comparison_settings=signature,
        note="Sample SD across seeds, not CI. No automatic best-condition or significance decision."
    ), ensure_ascii=False, indent=2), encoding="utf-8")
    panels = [("macro_auroc", "Trained encoder AUROC"),
              ("auroc_gain_vs_random", "AUROC gain over matched random encoder"),
              ("validation_skill_vs_zero", "MSE reduction vs zero (fraction)")]
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 9), constrained_layout=True)
    try:
        x = np.arange(len(summary))
        labels = [f"P{row.patch_samples}\nmask {row.mask_ratio:.0%}\nn={row.seed_count}"
                  for row in summary.itertuples()]
        for ax, (metric, title) in zip(axes, panels):
            if f"{metric}_mean" in summary:
                means = summary[f"{metric}_mean"].to_numpy(dtype=float)
                stds = summary[f"{metric}_std"].to_numpy(dtype=float)
                ax.plot(x, means, "o", color="C0")
                finite = np.isfinite(means) & np.isfinite(stds)
                ax.errorbar(x[finite], means[finite], yerr=stds[finite], fmt="none", capsize=4, color="C0")
                for i, row in enumerate(summary.itertuples()):
                    group = groups.get_group((row.patch_samples, row.mask_ratio))
                    ax.scatter(np.full(len(group), i), group[metric], s=12, alpha=0.35, color="C0")
            else:
                ax.text(0.5, 0.5, "Not recorded in these runs", transform=ax.transAxes, ha="center")
            if metric != "macro_auroc":
                ax.axhline(0, color="0.5", lw=0.8)
            ax.set_xticks(x, labels)
            ax.set_title(title, fontsize=10)
            ax.grid(axis="y", alpha=0.2)
        fig.suptitle("Condition comparison | points: seeds; error bars: sample SD (not CI)")
        fig.savefig(output_dir / "condition_comparison.png", dpi=150)
    finally:
        plt.close(fig)
    print(summary[["patch_samples", "mask_ratio", "seed_count", "complete_requested_seeds",
                   "macro_auroc_mean", "macro_auroc_std"]].to_string(index=False), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summarize_runs(args.run_dirs, args.output_dir)


if __name__ == "__main__":
    main()
