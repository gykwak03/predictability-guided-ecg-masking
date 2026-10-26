"""Train/validation-only mV cache and normalization interventions.

Never reverse Z-normalization or refilter an old normalized array. Rebuild the
filtered physical signal from the raw 500 Hz WFDB record, using the source
artifact's filter settings, and verify its per-record Z-score against that artifact.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def source_selection(folder, train_n, val_n, subset_seed):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    config = manifest["config"]
    if (config["source_sampling_rate"], config["target_sampling_rate"], config["normalization"]) != (500, 250, "per-record"):
        raise ValueError("Expected 500->250 Hz source artifact with per-record normalization")
    if config.get("bandpass_low") != .67 or config.get("bandpass_high") not in (40., 50.):
        raise ValueError("Use a reviewed 0.67-40 or 0.67-50 Hz source artifact")
    if manifest["lead_order"] != LEADS or manifest["class_order"] != CLASSES:
        raise ValueError("Lead/class order mismatch")
    frames = {}
    # Do not enumerate test metadata, waveform or labels.
    for split, limit, offset in [("train", train_n, 0), ("validation", val_n, 1)]:
        frame = pd.read_csv(folder / f"metadata_{split}.csv", low_memory=False)
        if not frame.ecg_id.is_unique or frame.patient_id.isna().any():
            raise ValueError("Invalid ECG/patient identities")
        if len(frame) < 2 or not np.isin(frame[CLASSES].to_numpy(), [0, 1]).all():
            raise ValueError("Need valid binary labels and at least two ECGs per split")
        if not np.array_equal(frame.artifact_index, np.arange(len(frame))):
            raise ValueError("Artifact row mismatch")
        if not frame.split.eq(split).all() or not (frame.strat_fold.between(1, 8).all() if split == "train" else frame.strat_fold.eq(9).all()):
            raise ValueError("Unexpected folds")
        if split == "train" and limit > len(frame):
            raise ValueError("Train subset too large")
        if limit and limit < len(frame):
            ix = np.sort(np.random.default_rng(subset_seed + offset).choice(len(frame), limit, replace=False))
            frame = frame.iloc[ix].copy()
        frames[split] = frame
    if set(frames["train"].patient_id) & set(frames["validation"].patient_id):
        raise ValueError("Patient overlap")
    return manifest, frames


def fit_statistics(signals):
    count, total, square = 0, np.zeros(12), np.zeros(12)
    for start in range(0, len(signals), 32):
        x = np.asarray(signals[start:start + 32], dtype=np.float64)
        count += x.shape[0] * x.shape[-1]
        total += x.sum((0, 2))
        square += (x * x).sum((0, 2))
    mean = total / count
    std = np.maximum(np.sqrt(np.maximum(square / count - mean**2, 0)), 1e-6)
    return dict(lead_mean=mean.tolist(), lead_std=std.tolist(),
                common_rms=float(max(np.sqrt(square.sum() / (count * 12)), 1e-6)),
                samples_per_lead=count, fit_split="selected SSL train only")


def normalize(raw, mode, stats, patch=50, mask=None):
    """Returns normalized waveform and inverse transform (B,C,1).

    visible_record computes moments on visible samples only; target values may
    use those moments, but can never change the visible input's normalization.
    """
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 3 or raw.shape[1] != 12 or not np.isfinite(raw).all():
        raise ValueError("Expected finite (B,12,L) physical mV")
    if mode == "per_record":
        mean = raw.mean(-1, keepdims=True, dtype=np.float64)
        scale = raw.std(-1, keepdims=True, dtype=np.float64)
    elif mode == "train_lead":
        mean = np.broadcast_to(np.asarray(stats["lead_mean"])[None, :, None], (len(raw), 12, 1))
        scale = np.broadcast_to(np.asarray(stats["lead_std"])[None, :, None], mean.shape)
    elif mode == "train_common":
        # No centering; one scalar preserves homogeneous limb-lead identities.
        mean = np.zeros((len(raw), 12, 1))
        scale = np.full_like(mean, stats["common_rms"])
    elif mode == "visible_record":
        if mask is None or raw.shape[-1] % patch:
            raise ValueError("visible_record requires a patch mask")
        visible = ~np.repeat(mask, patch, axis=1).reshape(raw.shape)
        count = visible.sum(-1, keepdims=True)
        # If a global mask hides an entire lead, use train-only fallback stats.
        fallback_m = np.asarray(stats["lead_mean"])[None, :, None]
        fallback_s = np.asarray(stats["lead_std"])[None, :, None]
        mean = np.where(count > 0, np.where(visible, raw, 0).sum(-1, keepdims=True, dtype=np.float64) / np.maximum(count, 1), fallback_m)
        variance = np.where(visible, (raw - mean)**2, 0).sum(-1, keepdims=True) / np.maximum(count, 1)
        scale = np.where(count > 0, np.sqrt(variance), fallback_s)
    else:
        raise ValueError("Unknown normalization")
    scale = np.maximum(scale, 1e-6)
    return ((raw - mean) / scale).astype(np.float32), mean, scale


def prepare_cache(plan, raw_dir, output):
    import wfdb
    from scipy.signal import butter, resample_poly, sosfiltfilt

    if plan["smoke"]:
        raise ValueError("Synthetic smoke plans do not use a scientific cache")
    raw_dir, output = Path(raw_dir).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Use a new cache folder; existing data is never overwritten")
    folder = Path(plan["artifact_dir"])
    if file_sha(folder / "manifest.json") != plan["source_manifest_sha256"]:
        raise ValueError("Source manifest changed after plan creation")
    manifest, frames = source_selection(folder, plan["train_ecgs"], plan["validation_ecgs"], plan["subset_seed"])
    config = manifest["config"]
    if {s: f.ecg_id.astype(int).tolist() for s, f in frames.items()} != plan["ecg_ids"]:
        raise ValueError("Metadata selection changed after plan creation")
    output.mkdir(parents=True)
    sos = butter(config["filter_order"], [config["bandpass_low"], config["bandpass_high"]], fs=250, btype="bandpass", output="sos")
    hashes, max_error = {}, 0.
    for split, frame in frames.items():
        reference = np.load(folder / f"signals_{split}.npy", mmap_mode="r")
        array = np.lib.format.open_memmap(output / f"mv_{split}.npy", mode="w+", dtype=np.float32, shape=(len(frame), 12, 2500))
        raw_hash = hashlib.sha256()
        for n, row in enumerate(frame.itertuples(index=False)):
            record = (raw_dir / row.source_record).resolve()
            if not record.is_relative_to(raw_dir):
                raise ValueError("Source record escapes raw dataset directory")
            signal, header = wfdb.rdsamp(str(record))
            names = [s.strip().upper() for s in header["sig_name"]]
            if (header["fs"] != 500 or signal.shape != (5000, 12) or len(set(names)) != 12 or
                    set(names) != set(s.upper() for s in LEADS) or any(u.lower() != "mv" for u in header["units"])):
                raise ValueError(f"Unexpected source waveform: {row.ecg_id}")
            raw = signal[:, [names.index(s.upper()) for s in LEADS]].T.astype(np.float32)
            raw_hash.update(raw.tobytes())
            x = sosfiltfilt(sos, resample_poly(raw, 1, 2, axis=1, padtype="line").astype(np.float32), axis=1).astype(np.float32)
            if not np.isfinite(x).all():
                raise ValueError("Nonfinite filtered ECG")
            z, _, _ = normalize(x[None], "per_record", {})
            error = float(np.max(np.abs(z[0] - reference[int(row.artifact_index)])))
            max_error = max(max_error, error)
            if error > 1e-4:
                raise ValueError(f"Preprocessing replay mismatch for ECG {row.ecg_id}: {error}; partial cache is invalid")
            array[n] = x
            if (n + 1) % 250 == 0:
                print(f"Cache {split}: {n + 1}/{len(frame)}", flush=True)
        array.flush()
        del array
        frame[["artifact_index", "ecg_id", "patient_id", "strat_fold", *CLASSES]].to_csv(output / f"metadata_{split}.csv", index=False)
        hashes[split] = dict(raw_signal_sha256=raw_hash.hexdigest(), filtered_sha256=file_sha(output / f"mv_{split}.npy"), metadata_sha256=file_sha(output / f"metadata_{split}.csv"))
    stats = fit_statistics(np.load(output / "mv_train.npy", mmap_mode="r"))
    cache_manifest = dict(schema=1, source_manifest_sha256=plan["source_manifest_sha256"],
                          selection_sha256=digest(plan["ecg_ids"]), source_config=config,
                          hashes=hashes, statistics=stats, replay_max_abs_error=max_error,
                          units="mV after source resample/filter, before normalization", test_opened=False)
    cache_manifest["cache_id"] = digest(cache_manifest)
    # Manifest is the completion marker. A partial cache cannot be loaded.
    write_json(output / "manifest.json", cache_manifest)
    return cache_manifest


def load_cache(plan, folder, verify=True):
    folder = Path(folder)
    m = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    recorded = dict(m)
    identity = recorded.pop("cache_id")
    if digest(recorded) != identity:
        raise ValueError("Cache manifest identity mismatch")
    if m["source_manifest_sha256"] != plan["source_manifest_sha256"] or m["selection_sha256"] != digest(plan["ecg_ids"]):
        raise ValueError("Cache belongs to a different source artifact or ECG selection")
    arrays, frames = {}, {}
    for split in ("train", "validation"):
        arrays[split] = np.load(folder / f"mv_{split}.npy", mmap_mode="r")
        frames[split] = pd.read_csv(folder / f"metadata_{split}.csv")
        if frames[split].ecg_id.astype(int).tolist() != plan["ecg_ids"][split] or arrays[split].shape != (len(frames[split]), 12, 2500):
            raise ValueError("Cache rows differ from plan")
        if verify and (file_sha(folder / f"mv_{split}.npy") != m["hashes"][split]["filtered_sha256"] or file_sha(folder / f"metadata_{split}.csv") != m["hashes"][split]["metadata_sha256"]):
            raise ValueError("Cache content was modified")
    return dict(arrays=arrays, frames=frames, stats=m["statistics"], cache_id=m["cache_id"])
