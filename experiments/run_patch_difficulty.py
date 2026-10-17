"""Re-evaluate saved predictability estimators for patch difficulty at 50% masking.

Inference only. Uses held-out train ECGs from the source run, independent discovery
and confirmation mask banks, and patient-level summaries. Never opens test arrays.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import uuid

import numpy as np
import pandas as pd
import torch

from patch_difficulty import (BANKS, ESTIMATORS, LEADS, analyze_ecg, complementary_bank,
                              replay_bank, summarize_ecgs)
from predictability_estimators import LaggedRidge, SmallCNN, SmallTransformer, reconstruction_metrics

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple, np.ndarray)):
            return [clean(v) for v in x]
        if isinstance(x, np.generic):
            return clean(x.item())
        return None if isinstance(x, float) and not np.isfinite(x) else x
    Path(path).write_text(json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def load_source(source, artifact_dir, count):
    source, artifact_dir = Path(source), Path(artifact_dir)
    config = json.loads((source / "run_config.json").read_text(encoding="utf-8"))
    if config.get("mask_ratio") != .5 or config.get("realized_mask_ratio") != .5:
        raise ValueError("Source estimators must have been fitted at exact 50% masking")
    manifest_bytes = (artifact_dir / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if hashlib.sha256(manifest_bytes).hexdigest() != config["source_manifest_sha256"]:
        raise ValueError("Source preprocessing manifest hash mismatch")
    if manifest["lead_order"] != list(LEADS) or manifest["config"]["target_sampling_rate"] != 250:
        raise ValueError("Expected canonical 12 leads at 250 Hz")
    selected = pd.read_csv(source / "ecg_selection.csv")
    if selected.role.isna().any() or not set(selected.role).issubset({"fit", "score", "validation"}):
        raise ValueError("Unexpected source roles")
    fit = selected[selected.role == "fit"]
    score = selected[selected.role == "score"].copy()
    if fit.empty or score.empty or score.ecg_id.duplicated().any() or selected.patient_id.isna().any():
        raise ValueError("Invalid source fit/score selection")
    if set(fit.patient_id) & set(score.patient_id) or set(fit.ecg_id) & set(score.ecg_id):
        raise ValueError("Estimator-fit/score patient or ECG leakage")
    if not (score.source_split == "train").all() or not score.strat_fold.isin(range(1, 9)).all():
        raise ValueError("Only held-out train folds 1-8 may be scored")
    metadata = pd.read_csv(artifact_dir / "metadata_train.csv", low_memory=False)
    if not np.array_equal(metadata.artifact_index, np.arange(len(metadata))):
        raise ValueError("Train metadata artifact indices are misaligned")
    if count < 0 or count > len(score):
        raise ValueError("Requested score ECGs exceed source pool (0 means all)")
    # Label-blind deterministic subset. Retains source artifact identity.
    score = score.sort_values("ecg_id").head(count if count else len(score)).reset_index(drop=True)
    if score.artifact_index.min() < 0 or score.artifact_index.max() >= len(metadata):
        raise ValueError("Invalid source artifact indices")
    actual = metadata.iloc[score.artifact_index.to_numpy()]
    for col in ("ecg_id", "patient_id", "strat_fold"):
        if not np.array_equal(actual[col].to_numpy(), score[col].to_numpy()):
            raise ValueError(f"Source {col} does not match train metadata")
    if not actual.strat_fold.isin(range(1, 9)).all():
        raise ValueError("Unexpected train folds")
    x = np.load(artifact_dir / "signals_train.npy", mmap_mode="r")
    if x.shape != (len(metadata), 12, 2500) or x.dtype != np.float32:
        raise ValueError("Invalid train signal contract")
    return config, score, metadata, x


def load_estimators(source, config, device):
    """Load only user-supplied source-run checkpoints; never fit or update weights."""
    scale = np.asarray(config["fixed_train_lead_std"], dtype=float)
    if scale.shape != (12,) or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Invalid fixed train lead standard deviations")
    models = {}
    with np.load(Path(source) / "ridge_weights.npz", allow_pickle=False) as w:
        ridge = LaggedRidge(config["patch_samples"], config["ridge_alpha"], lags=w["lags"].tolist())
        ridge.weights, ridge.scale = w["weights"].copy(), w["scale"].copy()
        if not np.allclose(ridge.scale, scale, rtol=0, atol=1e-12):
            raise ValueError("Ridge/source scale mismatch")
        if ridge.weights.shape != (12, 12*len(ridge.lags)*2+1) or not np.isfinite(ridge.weights).all():
            raise ValueError("Invalid Ridge checkpoint")
        models["ridge"] = ridge
    for name in ("cnn", "transformer"):
        path = Path(source) / f"{name}.pt"
        # Existing checkpoints include NumPy train std. Allow only the exact NumPy
        # reconstruction types required by that field; do not enable arbitrary pickle.
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as error:
            if "Weights only load failed" not in str(error):
                raise
            from numpy.core.multiarray import _reconstruct
            with torch.serialization.safe_globals([_reconstruct, np.ndarray, np.dtype, type(np.dtype("float64"))]):
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint["estimator"] != name or checkpoint["profile"] != config["profile"]:
            raise ValueError("Checkpoint estimator/profile mismatch")
        if not np.allclose(checkpoint["fixed_train_std"], scale, rtol=0, atol=1e-12):
            raise ValueError("Neural/source scale mismatch")
        p = config["profile"]
        model = SmallCNN(width=p["cnn_width"]) if name == "cnn" else SmallTransformer(
            patch_samples=config["patch_samples"], width=p["transformer_width"], depth=p["transformer_depth"])
        model.load_state_dict(checkpoint["model"], strict=True)
        models[name] = model.to(device).eval()
    return models, scale


def evaluate_patch_contexts(model, x, masks, scale, batch_size, device, neural):
    patches = []
    with torch.inference_mode():
        for start in range(0, len(masks), batch_size):
            m = masks[start:start+batch_size]
            batch = np.repeat(x[None], len(m), axis=0)
            if neural:
                prediction = model(torch.from_numpy(batch).to(device), torch.from_numpy(m).to(device)).cpu().numpy()
            else:
                prediction = model.predict(batch, m)
            _, patch = reconstruction_metrics(prediction, batch, m, scale)
            patches.append(patch.astype(np.float32))
    return np.concatenate(patches)


def run(args):
    pairs = args.pairs_per_bank if args.pairs_per_bank is not None else (4 if args.mode == "smoke" else 32)
    count = args.score_ecgs if args.score_ecgs is not None else (2 if args.mode == "smoke" else 64)
    if (pairs < 2 or args.batch_size < 1 or args.cpu_threads < 1 or args.seed < 0
            or args.bootstrap_repeats < 1 or args.shuffle_repeats < 1):
        raise ValueError("Invalid pairs, batch size, CPU threads or seed")
    config, score, metadata, signals = load_source(args.source_run, args.artifact_dir, count)
    patch_samples = int(config["patch_samples"])
    if patch_samples not in (50, 125, 250) or 2500 % patch_samples:
        raise ValueError("Unsupported patch geometry")
    required_files = ("ridge_weights.npz", "cnn.pt", "transformer.pt")
    missing = [f for f in required_files if not (args.source_run / f).is_file()]
    if missing:
        raise FileNotFoundError(f"Source inference needs server checkpoints: {', '.join(missing)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.mode == "pilot" and device.type != "cuda":
        raise RuntimeError("Pilot requires a Slurm GPU allocation; CPU supports --mode smoke")
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    models, scale = load_estimators(args.source_run, config, device)
    tokens = 12*(2500//patch_samples)
    bank, masks = complementary_bank(score, tokens, pairs, args.seed)
    if not np.array_equal(replay_bank(bank, tokens), masks):
        raise RuntimeError("Mask replay contract failed")
    output = Path(args.output_root).resolve() / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
             + f"_{args.mode}_seed{args.seed}_" + uuid.uuid4().hex[:8])
    output.mkdir(parents=True)
    bank.to_csv(output / "context_mask_bank.csv", index=False)
    score.to_csv(output / "ecg_selection.csv", index=False)
    hashes = {f: hashlib.sha256((args.source_run / f).read_bytes()).hexdigest() for f in
              (*required_files, "run_config.json", "ecg_selection.csv")}
    write_json(output / "run_config.json", dict(mode=args.mode, source_run=str(args.source_run.resolve()),
        source_files_sha256=hashes, source_manifest_sha256=config["source_manifest_sha256"],
        fixed_train_lead_std=scale, patch_samples=patch_samples, mask_ratio=.5, seed=args.seed,
        pairs_per_bank=pairs, contexts_per_ecg=4*pairs, hidden_observations_per_patch_per_bank=pairs,
        score_ecgs=len(score), device=str(device), test_opened=False, training_performed=False,
        target="Conditional hidden-patch reconstruction error under uniform exact-50% contexts",
        split_policy="Same held-out train ECGs in discovery/confirmation, independently generated mask contexts",
        rank_policy="Within ECG, per-estimator pooled NRMSE ascending percentiles, equal-weight consensus; high=hard",
        inference_unit="patient; complementary halves and patches are not independent subjects"))
    started = time.monotonic()
    ecg_rows = []
    for index, row in enumerate(score.itertuples(index=False)):
        selected = bank.ecg_id.to_numpy() == row.ecg_id
        local = bank.loc[selected]
        local_masks = masks[selected]
        x = np.asarray(signals[row.artifact_index], dtype=np.float32)
        zero = np.sqrt(np.mean((x.astype(float)/scale[:, None]).reshape(12, -1, patch_samples)**2, axis=-1))
        errors = {name: {} for name in ESTIMATORS}
        for name in ESTIMATORS:
            for split in BANKS:
                errors[name][split] = evaluate_patch_contexts(models[name], x,
                    local_masks[local.bank.to_numpy() == split], scale, args.batch_size, device, name != "ridge")
        identity = dict(ecg_id=int(row.ecg_id), patient_id=row.patient_id, artifact_index=int(row.artifact_index))
        patches, stats = analyze_ecg(errors, zero, identity, patch_samples,
                                     shuffle_repeats=args.shuffle_repeats, seed=args.seed)
        stats["min_hidden_contexts"] = min(int(patches[f"{a}_{b}_count"].min()) for a in ESTIMATORS for b in BANKS)
        # Labels are only descriptive output, added AFTER scoring and frozen groups.
        for cls in ("NORM", "MI", "STTC", "CD", "HYP"):
            if cls in metadata:
                stats[cls] = int(metadata.iloc[row.artifact_index][cls])
        ecg_rows.append(stats)
        patches.to_csv(output / "patch_scores.csv", mode="w" if index == 0 else "a", header=index == 0, index=False)
        pd.DataFrame(ecg_rows).to_csv(output / "ecg_validation.csv", index=False)
        if args.save_patch_cache:
            np.savez_compressed(output / f"patch_errors_ecg{row.ecg_id}.npz", zero_nrmse=zero,
                                **{f"{a}_{b}": errors[a][b] for a in ESTIMATORS for b in BANKS})
        print(f"ECG {index+1}/{len(score)} {row.ecg_id}: confirmation rho={stats['consensus_spearman']:.3f}", flush=True)
    frame = pd.DataFrame(ecg_rows)
    summary = summarize_ecgs(frame, args.bootstrap_repeats, args.seed, args.mode)
    summary.update(seconds=time.monotonic()-started, test_opened=False, training_performed=False)
    write_json(output / "summary.json", summary)
    write_json(output / "completion.json", dict(status="completed", scientific_status=summary["status"],
                                                ecgs=len(frame), test_opened=False, training_performed=False))
    print(f"Output: {output}\nScientific status: {summary['status']}", flush=True)
    return output


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-run", type=Path, required=True, help="Completed 50%-mask predictability run with all checkpoints")
    p.add_argument("--artifact-dir", type=Path, default=ROOT / "preprocessing/artifacts/ptbxl_250hz")
    p.add_argument("--output-root", type=Path, default=ROOT / "experiments/artifacts/patch_difficulty")
    p.add_argument("--mode", choices=("smoke", "pilot"), default="smoke")
    p.add_argument("--score-ecgs", type=int, help="0 = entire source score pool; default smoke 2 / pilot 64")
    p.add_argument("--pairs-per-bank", type=int, help="Default smoke 4 / pilot 32; each pair contains complementary 50% masks")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--bootstrap-repeats", type=int, default=2000)
    p.add_argument("--shuffle-repeats", type=int, default=32)
    p.add_argument("--save-patch-cache", action="store_true", help="Save per-ECG hidden-only errors; NPZ remains Git-ignored")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
