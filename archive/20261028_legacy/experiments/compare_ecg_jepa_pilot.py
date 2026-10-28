"""Rank matched ECG-JEPA pilots on validation linear probes, never SSL loss."""
import argparse
import json
from pathlib import Path

import pandas as pd


def compare(folders, output, required_seeds=3):
    rows, configs, summaries = [], [], []
    for folder in map(Path, folders):
        c = json.loads((folder/"run_config.json").read_text(encoding="utf-8"))
        s = json.loads((folder/"summary.json").read_text(encoding="utf-8"))
        if c["mode"] != "pilot" or s["status"] != "completed" or not s["eligible_for_condition_selection"]:
            raise ValueError(f"Not an eligible completed from-scratch GPU pilot: {folder}")
        if c["initialization"] != "scratch" or c["test_opened"] or s["test_opened"]:
            raise ValueError("Do not mix continued-pretraining or test-selected conditions")
        if c["model"] != "sehun_ecg_jepa" or not s["trained_probe"]["all_five_classes_supported"]:
            raise ValueError("Need this model and all five validation classes")
        rows.append(dict(patch_samples=c["patch_samples"], patch_seconds=c["patch_seconds"],
            mask_ratio=c["mask_ratio"], topology=c["topology"], seed=c["seed"], run=str(folder),
            validation_auroc=s["trained_probe"]["macro_auroc"],
            validation_auprc=s["trained_probe"]["macro_auprc_ap"],
            initial_auroc=s["initial_probe"]["macro_auroc"],
            auroc_gain=s["trained_probe"]["macro_auroc"]-s["initial_probe"]["macro_auroc"],
            pretrain_seconds=s["pretrain_seconds"], peak_bytes=s["gpu_peak_allocated_bytes"]))
        configs.append(c)
        summaries.append(s)
    if not rows:
        raise ValueError("No runs")
    common = ("model", "source", "source_manifest_sha256", "architecture", "initialization", "leads",
              "lead_indices", "steps", "effective_batch", "microbatch", "accum_steps", "train_ecg_ids",
              "validation_ecg_ids", "subset_seed", "optimizer", "teacher", "probe", "precision", "gpu")
    for field in common:
        if any(c[field] != configs[0][field] for c in configs[1:]):
            raise ValueError(f"Unmatched field: {field}")
    table = pd.DataFrame(rows)
    conditions = ["patch_samples", "mask_ratio", "topology"]
    if table.duplicated(conditions+["seed"]).any():
        raise ValueError("Duplicate condition/seed; choose one run explicitly")
    seed_sets = [set(g.seed) for _, g in table.groupby(conditions)]
    if any(seeds != seed_sets[0] for seeds in seed_sets) or len(seed_sets[0]) < required_seeds:
        raise ValueError(f"Require the same set of at least {required_seeds} seeds per condition")
    for seed in table.seed.unique():
        pairs = [(c, s) for c, s in zip(configs, summaries) if c["seed"] == seed]
        if len({s["batch_sequence_sha256"] for c, s in pairs}) != 1:
            raise ValueError("Unmatched ECG batch sequence within seed")
        if len({c["initial_backbone_sha256"] for c, s in pairs}) != 1:
            raise ValueError("Shared encoder weights did not start identically within seed")
    ranked = table.groupby(conditions, as_index=False).agg(
        seeds=("seed", "nunique"), mean_validation_auroc=("validation_auroc", "mean"),
        std_validation_auroc=("validation_auroc", "std"), mean_auroc_gain=("auroc_gain", "mean"),
        mean_validation_auprc=("validation_auprc", "mean"),
        mean_pretrain_seconds=("pretrain_seconds", "mean"), peak_bytes=("peak_bytes", "max"))
    ranked = ranked.sort_values("mean_validation_auroc", ascending=False)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output/"runs.csv", index=False)
    ranked.to_csv(output/"validation_ranking.csv", index=False)
    (output/"interpretation.txt").write_text(
        "Exploratory validation ranking, not a test result or significance test. "
        "Compare the 50% lead-time conditions first; report topology controls separately. "
        "Matched updates/ECG exposures do not mean matched FLOPs. "
        "Do not select on latent loss; inspect collapse, AUROC gain, seed variability and resource cost.\n",
        encoding="utf-8")
    return ranked


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(compare(args.runs, args.output).to_string(index=False))
