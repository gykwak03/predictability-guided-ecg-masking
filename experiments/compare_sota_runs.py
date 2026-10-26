"""Compare completed, matched pilot runs; never rank smoke/profile as science."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def compare(runs, output):
    records, configs, summaries = [], [], []
    for folder in map(Path, runs):
        config = json.loads((folder/"run_config.json").read_text(encoding="utf-8"))
        summary = json.loads((folder/"summary.json").read_text(encoding="utf-8"))
        if summary.get("status") != "completed" or not summary.get("eligible_for_model_selection"):
            raise ValueError(f"Not a completed five-class pilot: {folder}")
        if config.get("test_opened") or summary.get("test_opened") or config.get("pretrained_weights"):
            raise ValueError("Do not mix test-selected or external-pretrained runs with this from-scratch comparison")
        trained, random = summary["trained_probe"], summary["random_probe"]
        records.append(dict(model=config["model"], seed=config["seed"], run=str(folder),
            validation_auroc=trained["macro_auroc"], random_encoder_auroc=random["macro_auroc"],
            pretraining_gain=trained["macro_auroc"]-random["macro_auroc"],
            validation_auprc=trained["macro_auprc_ap"], validation_f1=trained["macro_f1"],
            parameters=config["parameters"], pretrain_seconds=summary["pretrain_seconds"],
            gpu_peak_allocated_bytes=summary["gpu_peak_allocated_bytes"],
            mask_sampler_compatible=summary["mask_sampler_compatible"]))
        configs.append(config)
        summaries.append(summary)
    table = pd.DataFrame(records)
    if table.duplicated(["model", "seed"]).any():
        raise ValueError("Duplicate model/seed: explicitly choose one run per condition")
    common = ["source_manifest_sha256", "steps", "effective_batch", "microbatch", "accum_steps",
              "optimizer", "train_ecg_ids", "validation_ecg_ids", "probe", "subset_seed"]
    for field in common:
        if any(c[field] != configs[0][field] for c in configs[1:]):
            raise ValueError(f"Unmatched comparison field: {field}")
    for seed in table.seed.unique():
        group = [s for c, s in zip(configs, summaries) if c["seed"] == seed]
        if len({s["batch_sequence_sha256"] for s in group}) > 1:
            raise ValueError("Training ECG batch order differs for the same seed")
        masked = [s for s in group if s["mask_sampler_compatible"]]
        if len({s["mask_sequence_sha256"] for s in masked}) > 1:
            raise ValueError("Reconstruction models saw different training masks")
    seed_sets = {model: set(g.seed) for model, g in table.groupby("model")}
    if any(seeds != next(iter(seed_sets.values())) for seeds in seed_sets.values()):
        raise ValueError("Different seed sets across models")
    ranking = table.groupby("model").agg(
        seeds=("seed", "count"), validation_auroc_mean=("validation_auroc", "mean"),
        validation_auroc_std=("validation_auroc", "std"), gain_mean=("pretraining_gain", "mean"),
        mask_sampler_compatible=("mask_sampler_compatible", "all"),
        mean_pretrain_seconds=("pretrain_seconds", "mean")
    ).sort_values("validation_auroc_mean", ascending=False)
    compatible = ranking.loc[ranking.mask_sampler_compatible & ranking.gain_mean.gt(0)]
    complete = set(table.model) == {"stmem", "patchecg", "ecgcpc"}
    status = "provisional_validation_ranking"
    if complete and table.groupby("model").seed.nunique().min() >= 3:
        status = "three_seed_validation_comparison"
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output/"runs.csv", index=False)
    ranking.to_csv(output/"validation_ranking.csv")
    report = dict(status=status, all_three_models_present=complete, test_opened=False,
                  provisional_mask_backbone=compatible.index[0] if len(compatible) else None,
                  scope="Validation ranking at matched PTB-XL data/update budget, not universal SOTA",
                  selection_note="Confirm uncertainty/seed stability before final selection; CPC is a comparator, not a waveform-mask backbone")
    (output/"comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.runs, args.output), indent=2))
