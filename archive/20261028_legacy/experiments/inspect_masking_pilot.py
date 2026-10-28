"""Analyze saved final checkpoints without MAE retraining or modifying old results.

Uses the current notebook's model/probe definitions and the original run's ECG
IDs, model profile and mask seed. Legacy random initialization is regenerated
and explicitly marked; it is not an archived original initialization.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

import numpy as np
import pandas as pd

from run_masking_pilot import run_notebook


def indices_for_ids(frame, ids):
    ids = np.asarray(ids, dtype=np.int64)
    if len(ids) == 0 or len(np.unique(ids)) != len(ids):
        raise ValueError("Saved subset ECG IDs must be non-empty and unique")
    positions = pd.Index(frame.ecg_id).get_indexer(ids)
    if (positions < 0).any():
        raise ValueError("Some saved ECG IDs are missing in the supplied artifacts")
    return positions


def inspect_run(run_dir, *, artifact_dir=None, output_dir=None, conditions=None,
                probe=True, examples=3, batch_size=None, device="auto"):
    run_dir = Path(run_dir).resolve()
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    original_results = pd.read_csv(run_dir / "results.csv")
    if config["mode"] not in {"smoke", "pilot"}:
        raise ValueError("Only smoke/pilot checkpoints may be inspected")
    expected_probe = dict(name="AdamW", lr=1e-3, weight_decay=1e-4, loss="BCEWithLogitsLoss")
    if probe and (config.get("probe_optimizer") != expected_probe
                  or config.get("pooling") != "mean-all-lead-time-tokens"
                  or config.get("threshold") != 0.5):
        raise ValueError("Saved probe settings differ from the current implementation; cannot claim a matched comparison")
    if examples < 0 or (batch_size is not None and batch_size < 1):
        raise ValueError("examples >= 0 and batch size > 0 required")
    directories = sorted(run_dir.glob("p*_r*/checkpoint.pt"))
    if conditions:
        missing = set(conditions) - {path.parent.name for path in directories}
        if missing:
            raise ValueError(f"Conditions have no checkpoint: {sorted(missing)}")
        directories = [path for path in directories if path.parent.name in conditions]
    if not directories:
        raise ValueError("No saved condition checkpoints found")
    # Initialize dependencies/contracts cheaply, then restore saved IDs/profile.
    ns = run_notebook(mode="smoke", train=False, seed=config["seed"],
                      artifact_dir=artifact_dir, quiet=True)
    if not ns["TRAINING_AVAILABLE"]:
        raise RuntimeError("PyTorch and scikit-learn are required")
    if ns["SOURCE_MANIFEST_SHA256"] != config["source_manifest_sha256"]:
        raise ValueError("Artifact manifest hash differs from the original run")
    if ns["LEADS"] != config["lead_order"] or ns["CLASSES"] != config["class_order"]:
        raise ValueError("Lead/class order differs from saved run")
    torch, diagnostics = ns["torch"], ns["diagnostics"]
    if device != "auto":
        ns["DEVICE"] = torch.device(device)
    if ns["DEVICE"].type == "cpu":
        torch.set_num_threads(4)
    ns["PROFILE"] = dict(config["profile"])
    if batch_size is not None:
        ns["PROFILE"]["batch"] = batch_size  # inference only; probe update/batch settings unchanged
    ns["EVAL_SEED"] = config.get("evaluation_mask_seed", config["seed"])
    for split, key in [("train", "TRAIN_IDX"), ("validation", "VAL_IDX")]:
        ns[key] = indices_for_ids(ns["frames"][split], config[f"{split}_ecg_ids"])
    ns["PROBE_TRAIN_IDX"] = ns["TRAIN_IDX"][ns["arrays"]["train"]["valid"][ns["TRAIN_IDX"]]]
    ns["PROBE_VAL_IDX"] = ns["VAL_IDX"][ns["arrays"]["validation"]["valid"][ns["VAL_IDX"]]]
    eval_ids = config.get("reconstruction_validation_ecg_ids")
    eval_indices = (indices_for_ids(ns["frames"]["validation"], eval_ids) if eval_ids is not None
                    else ns["VAL_IDX"][:config["profile"]["eval_n"]])
    baseline_ids = config.get("linear_baseline_train_ecg_ids")
    fit_indices = (indices_for_ids(ns["frames"]["train"], baseline_ids) if baseline_ids else
                   ns["TRAIN_IDX"][ns["fixed_subset"](len(ns["TRAIN_IDX"]), 256,
                                     config.get("subset_seed", config["seed"]) + 4)])
    baseline = diagnostics.fit_cross_lead_baseline(ns["arrays"]["train"]["x"], fit_indices)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(output_dir) if output_dir else run_dir / f"{stamp}_inspection_{uuid.uuid4().hex[:8]}"
    output_dir = output_dir.resolve()
    if output_dir == run_dir or output_dir in [path.parent for path in directories]:
        raise ValueError("Choose a separate output directory to preserve old results")
    output_dir.mkdir(parents=True, exist_ok=False)
    report_config = dict(config, source_run=str(run_dir), kind="checkpoint-inspection-not-training",
                         inference_batch=ns["PROFILE"]["batch"], probe_recomputed=probe,
                         device=str(ns["DEVICE"]), current_torch_version=torch.__version__)
    diagnostics.write_json(output_dir / "run_config.json", report_config)
    rows = []
    for path in directories:
        print(f"Inspecting {path.parent.name} on {ns['DEVICE']}", flush=True)
        patch, percent = path.parent.name.split("_")
        patch, percent = int(patch[1:]), int(percent[1:])
        matched = original_results[(original_results.patch_samples == patch) &
                                   (np.round(original_results.mask_ratio*100) == percent)]
        if len(matched) != 1:
            raise ValueError("Expected exactly one original result per condition")
        ratio = float(matched.iloc[0].mask_ratio)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if (checkpoint["patch_length"] != patch or checkpoint["profile"] != config["profile"]
                or checkpoint["source_manifest_sha256"] != ns["SOURCE_MANIFEST_SHA256"]
                or checkpoint.get("seed", config["seed"]) != config["seed"]
                or checkpoint.get("mask_ratio", ratio) != ratio):
            raise ValueError(f"Checkpoint metadata does not match run: {path}")
        ns["seed_everything"](config["seed"])
        model = ns["PlainECGMAE"](patch, ns["PROFILE"]).to(ns["DEVICE"])
        initial_path = path.with_name("initial_checkpoint.pt")
        initial = (torch.load(initial_path, map_location="cpu", weights_only=True)["model"]
                   if initial_path.exists() else {key: value.detach().cpu().clone()
                                                 for key, value in model.state_dict().items()})
        model.load_state_dict(checkpoint["model"], strict=True)
        model.ratio = ratio
        bundle = diagnostics.evaluate_reconstruction(
            model, ns["arrays"]["validation"]["x"], ns["frames"]["validation"].ecg_id.to_numpy(),
            eval_indices, ns["PROFILE"]["batch"], ns["EVAL_SEED"], repeats=3,
            examples=examples, lead_order=ns["LEADS"], fs=ns["FS"], linear_baseline=baseline)
        destination = output_dir / path.parent.name
        diagnostics.save_reconstruction(bundle, destination)
        row = matched.iloc[0].to_dict()
        row["original_validation_masked_mse"] = row["validation_masked_mse"]
        for key, value in bundle["metrics"].items():
            row[f"validation_{key}"] = value
        trained_metrics = random_metrics = None
        if probe:
            trained_metrics, detail, probabilities, state = ns["fit_linear_probe"](model)
            detail.to_csv(destination / "class_metrics.csv", index=False)
            np.save(destination / "validation_probabilities.npy", probabilities)
            ns["seed_everything"](config["seed"])
            random_model = ns["PlainECGMAE"](patch, ns["PROFILE"]).to(ns["DEVICE"])
            random_model.load_state_dict(initial)
            random_metrics, random_detail, random_probabilities, _ = ns["fit_linear_probe"](random_model)
            random_detail.to_csv(destination / "random_encoder_class_metrics.csv", index=False)
            np.save(destination / "random_encoder_probabilities.npy", random_probabilities)
            for metric, short in [("macro_auroc", "auroc"), ("macro_auprc", "auprc"), ("macro_f1", "f1")]:
                row[f"original_{metric}"] = row[metric]
                row[metric] = trained_metrics[metric]
                row[f"random_encoder_{metric}"] = random_metrics[metric]
                row[f"{short}_gain_vs_random"] = trained_metrics[metric]-random_metrics[metric]
            del random_model, state
        row["random_initialization_source"] = "archived" if initial_path.exists() else "regenerated-current-model"
        diagnostics.write_json(destination / "probe_comparison.json", dict(
            trained=trained_metrics, random=random_metrics,
            initialization_source=row["random_initialization_source"],
            caveat="Legacy initialization recreated from seed/current model; exact historical identity is unverified. Both probes recomputed with current settings; no MAE retraining."))
        rows.append(row)
        pd.DataFrame(rows).to_csv(output_dir / "results.csv", index=False)
        print(f"MSE={bundle['metrics']['masked_mse']:.6f}, zero={bundle['metrics']['zero_masked_mse']:.6f}, "
              f"reduction={bundle['metrics']['skill_vs_zero']:.2%}", flush=True)
        del model, initial, checkpoint, bundle
        if ns["DEVICE"].type == "cuda":
            torch.cuda.empty_cache()
    print("Inspection output:", output_dir, flush=True)
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--conditions", nargs="+", help="e.g. p50_r75 p50_r90; default: all saved conditions")
    parser.add_argument("--no-probe", action="store_true", help="Reconstruction only; skip both linear probes")
    parser.add_argument("--examples", type=int, default=3)
    parser.add_argument("--batch-size", type=int, help="Inference batch only")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    inspect_run(args.run_dir, artifact_dir=args.artifact_dir, output_dir=args.output_dir,
                conditions=args.conditions, probe=not args.no_probe, examples=args.examples,
                batch_size=args.batch_size, device=args.device)


if __name__ == "__main__":
    main()
