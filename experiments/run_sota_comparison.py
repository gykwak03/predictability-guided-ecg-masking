"""One model per GPU job: matched PTB-XL / 50 Hz / finite-budget comparison.

Profile precedes pilot; only completed pilot runs with all five validation
classes may be compared. Test arrays are never opened. Official upstream
model implementations are imported, not replaced by generic local models.
ECG-CPC keeps its CPC objective and is an external comparator, not a candidate
for arbitrary hidden-patch waveform reconstruction.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from sota_adapters import create_model
from st_mem_adapter import random_lead_mask, fixed_lead_mask

HERE = Path(__file__).resolve().parent
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", *[f"V{i}" for i in range(1, 7)]]


def json_write(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def load_data(folder, train_n, seed, validation_n=0):
    folder = Path(folder)
    manifest_file = folder/"manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    c = manifest["config"]
    if (c.get("target_sampling_rate"), c.get("bandpass_low"), c.get("bandpass_high"),
        c.get("normalization")) != (250, .67, 50., "per-record"):
        raise ValueError("This experiment requires the NEW 250 Hz / 0.67–50 Hz / per-record dataset. "
                         "Do not point it at the previous 40 Hz artifact.")
    if manifest["lead_order"] != LEADS or manifest["class_order"] != CLASSES:
        raise ValueError("Lead/class order mismatch")
    arrays, frames, labels, valid = {}, {}, {}, {}
    # Deliberately enumerate only these splits: no test waveform or labels.
    for split in ("train", "validation"):
        frames[split] = pd.read_csv(folder/f"metadata_{split}.csv", low_memory=False)
        arrays[split] = np.load(folder/f"signals_{split}.npy", mmap_mode="r")
        labels[split] = np.load(folder/f"labels_{split}.npy", mmap_mode="r")
        valid[split] = np.load(folder/f"label_valid_{split}.npy", mmap_mode="r")
        f = frames[split]
        if arrays[split].shape != (len(f), 12, 2500) or labels[split].shape != (len(f), 5):
            raise ValueError("Artifact shape mismatch")
        if not np.array_equal(f.artifact_index, np.arange(len(f))) or not f.ecg_id.is_unique:
            raise ValueError("Artifact metadata is not aligned")
        if not np.array_equal(labels[split], f[CLASSES].to_numpy(dtype=np.uint8)):
            raise ValueError("Artifact labels are not aligned")
        if not np.array_equal(valid[split], labels[split].any(axis=1)):
            raise ValueError("Missing-label mask mismatch")
        if f.patient_id.isna().any():
            raise ValueError("Missing patient ID")
        if not (f.strat_fold.between(1, 8).all() if split == "train" else f.strat_fold.eq(9).all()):
            raise ValueError("Unexpected official folds")
    if set(frames["train"].patient_id) & set(frames["validation"].patient_id):
        raise ValueError("Patient leakage between train and validation")
    if train_n > len(frames["train"]):
        raise ValueError("Requested training subset exceeds dataset")
    ti = np.sort(np.random.default_rng(seed).choice(len(frames["train"]), train_n, replace=False))
    vi = np.flatnonzero(valid["validation"])
    if validation_n:
        vi = vi[:validation_n]
    pi = ti[np.asarray(valid["train"])[ti]]
    if len(pi) < 2 or len(vi) < 2:
        raise ValueError("Need at least two labeled probe records in each split")
    return dict(manifest=manifest, manifest_sha256=hashlib.sha256(manifest_file.read_bytes()).hexdigest(),
                arrays=arrays, frames=frames, labels=labels, train=ti, probe_train=pi, validation=vi)


def get_batch(data, split, indices, device):
    x = np.asarray(data["arrays"][split][indices], dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite input waveform")
    return torch.from_numpy(x).to(device)


def class_metrics(y, probs):
    rows = []
    for i, name in enumerate(CLASSES):
        supported = len(np.unique(y[:, i])) == 2
        rows.append(dict(class_name=name, positives=int(y[:, i].sum()), records=len(y), supported=supported,
            auroc=float(roc_auc_score(y[:, i], probs[:, i])) if supported else None,
            average_precision=float(average_precision_score(y[:, i], probs[:, i])) if supported else None,
            f1_threshold_05=float(f1_score(y[:, i], probs[:, i] >= .5, zero_division=0))))
    supported_rows = [r for r in rows if r["supported"]]
    return dict(macro_auroc=float(np.mean([r["auroc"] for r in supported_rows])) if supported_rows else None,
                macro_auprc_ap=float(np.mean([r["average_precision"] for r in supported_rows])) if supported_rows else None,
                macro_f1=float(np.mean([r["f1_threshold_05"] for r in rows])),
                all_five_classes_supported=len(supported_rows) == 5, per_class=rows)


def features(model, data, split, indices, device, batch_size):
    model.eval()
    chunks = []
    # S4 caches frequency-domain tensors; no_grad keeps those caches usable by
    # subsequent training, unlike tensors first created in inference_mode.
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            chunks.append(model.features(get_batch(data, split, indices[start:start+batch_size], device)).float().cpu())
    result = torch.cat(chunks)
    if result.ndim != 2 or not torch.isfinite(result).all():
        raise ValueError("Nonfinite or invalid encoder features")
    return result


def probe(model, data, device, args, output, tag):
    """Frozen encoder, train-only feature scaling, exactly one linear layer."""
    ft = features(model, data, "train", data["probe_train"], device, args.microbatch)
    fv = features(model, data, "validation", data["validation"], device, args.microbatch)
    mean, std = ft.mean(0), ft.std(0, correction=0).clamp_min(1e-6)
    ft, fv = (ft-mean)/std, (fv-mean)/std
    yt = torch.from_numpy(np.asarray(data["labels"]["train"][data["probe_train"]], dtype=np.float32))
    yv = np.asarray(data["labels"]["validation"][data["validation"]])
    torch.manual_seed(args.seed+100)
    head = torch.nn.Linear(ft.shape[1], 5)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = torch.Generator().manual_seed(args.seed+200)
    history, best, best_score, best_state, best_probs = [], None, -math.inf, None, None
    for epoch in range(1, args.probe_epochs+1):
        head.train()
        order = torch.randperm(len(ft), generator=rng)
        for start in range(0, len(ft), 128):
            rows = order[start:start+128]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(head(ft[rows]), yt[rows])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        head.eval()
        with torch.inference_mode():
            probs = torch.sigmoid(head(fv)).numpy()
        metrics = class_metrics(yv, probs)
        value = metrics["macro_auroc"]
        history.append(dict(epoch=epoch, macro_auroc=value, macro_auprc_ap=metrics["macro_auprc_ap"]))
        if value is not None and value > best_score:
            best_score, best = value, dict(metrics, selected_probe_epoch=epoch)
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
            best_probs = probs.copy()
    if best is None:
        raise ValueError("No probe validation class has positives and negatives")
    pd.DataFrame(history).to_csv(output/f"{tag}_probe_history.csv", index=False)
    pd.DataFrame(best_probs, columns=CLASSES).assign(
        ecg_id=data["frames"]["validation"].iloc[data["validation"]].ecg_id.to_numpy()
    ).to_csv(output/f"{tag}_validation_probabilities.csv", index=False)
    torch.save(dict(head=best_state, feature_mean=mean, feature_std=std), output/f"{tag}_probe.pt")
    return best


def optimizer_groups(model, lr, weight_decay):
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        # Preserve official S4 state-parameter optimization hints, especially wd=0.
        options = dict(getattr(parameter, "_optim", {}))
        options.setdefault("lr", lr)
        options.setdefault("weight_decay", 0. if parameter.ndim <= 1 or "norm" in name.lower() else weight_decay)
        key = tuple(sorted(options.items()))
        groups.setdefault(key, dict(params=[], **options))["params"].append(parameter)
    return list(groups.values())


def evaluate_reconstruction(model, data, device, args):
    if args.model == "ecgcpc":
        return None
    model.eval()
    ids = data["validation"][:min(64, len(data["validation"]))]
    errors, zeros = [], []
    with torch.no_grad():
        for start in range(0, len(ids), args.microbatch):
            ix = ids[start:start+args.microbatch]
            ecgs = data["frames"]["validation"].iloc[ix].ecg_id.to_numpy()
            hidden, _ = fixed_lead_mask(ecgs, 12*2500//args.patch_samples, .5, args.eval_mask_seed)
            hidden = torch.from_numpy(hidden).to(device)
            pred, target = model(get_batch(data, "validation", ix, device), hidden)
            errors.append(((pred-target).square().mean(-1)[hidden]).float().cpu().numpy())
            zeros.append((target.square().mean(-1)[hidden]).float().cpu().numpy())
    mse, zero = float(np.concatenate(errors).mean()), float(np.concatenate(zeros).mean())
    return dict(masked_mse=mse, zero_mse=zero, skill_vs_zero=1-mse/zero)


def run(args):
    if args.mode != "smoke" and not torch.cuda.is_available():
        raise RuntimeError("Profile/pilot require an allocated CUDA GPU; do not run on the login node")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    if args.mode != "smoke" and device.type != "cuda":
        raise ValueError("CPU is only supported for reduced smoke checks")
    train_n = args.train_ecgs or (32 if args.mode == "smoke" else 64 if args.mode == "profile" else 5000)
    data = load_data(args.artifact_dir, train_n, args.subset_seed, 64 if args.mode != "pilot" else 0)
    torch.manual_seed(args.seed)
    model = create_model(args.model, args.external_root, args.patch_samples, args.mode == "smoke").to(device)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output_root/f"{stamp}_{args.mode}_{args.model}_seed{args.seed}_{uuid.uuid4().hex[:8]}"
    output.mkdir(parents=True, exist_ok=False)
    steps = args.steps or (2 if args.mode == "smoke" else 100 if args.mode == "profile" else 3000)
    effective = args.microbatch*args.accum_steps
    if train_n < effective:
        raise ValueError("Training subset must contain at least one effective batch")
    config = dict(vars(args), artifact_dir=str(args.artifact_dir.resolve()), external_root=str(args.external_root.resolve()),
        output_root=str(args.output_root.resolve()), steps=steps, effective_batch=effective, train_ecgs=train_n,
        train_ecg_ids=data["frames"]["train"].iloc[data["train"]].ecg_id.astype(int).tolist(),
        validation_ecg_ids=data["frames"]["validation"].iloc[data["validation"]].ecg_id.astype(int).tolist(),
        source_manifest_sha256=data["manifest_sha256"], preprocessing=data["manifest"]["config"],
        parameters=sum(p.numel() for p in model.parameters()), upstream=model.source,
        architecture=getattr(model, "architecture", {}), device=str(device), test_opened=False,
        pretrained_weights=False, classification="5-superclass multi-label", precision="fp32" if args.model == "ecgcpc" else "fp16-amp" if device.type == "cuda" else "fp32",
        masking="none (native CPC objective)" if args.model == "ecgcpc" else "per-lead exact 25-of-50 hidden; shared mask seeds",
        optimizer=dict(name="AdamW", lr=args.lr, betas=[.9, .95], weight_decay=.05, clip_norm=1.,
                       warmup_updates=max(1, round(.05*steps)), min_lr=.1*args.lr),
        probe=dict(epochs=args.probe_epochs, batch=128, lr=1e-3, weight_decay=1e-4,
                   loss="unweighted BCEWithLogitsLoss", feature_scaling="train-only", threshold=.5,
                   encoder="frozen + eval mode; BN statistics frozen", selection="validation macro AUROC"),
        training_sampling="effective batch without replacement, fresh random sample each update; not exact epochs",
        comparison_scope="PTB-XL adaptation at common data/update budget, not paper-SOTA reproduction or equal FLOPs",
        code_sha256={f:hashlib.sha256((HERE/f).read_bytes()).hexdigest() for f in
                     ["run_sota_comparison.py", "sota_adapters.py", "st_mem_adapter.py"]})
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        config["gpu"] = dict(name=props.name, total_memory_bytes=props.total_memory)
    json_write(output/"run_config.json", config)
    selection = data["frames"]["train"].iloc[data["train"]][["ecg_id", "patient_id", "artifact_index"]]
    selection.to_csv(output/"train_ecg_selection.csv", index=False)
    print(f"OUTPUT={output}\nMODEL={args.model}; {config['parameters']:,} parameters; {effective} ECG/update", flush=True)
    if args.mode == "profile":
        random_probe = None
    else:
        print("Random-initialization linear probe", flush=True)
        random_probe = probe(model, data, device, args, output, "random")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
    opt = torch.optim.AdamW(optimizer_groups(model, args.lr, .05), betas=(.9, .95))
    use_amp = device.type == "cuda" and args.model != "ecgcpc"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = np.random.default_rng(args.seed+7000)
    mask_rng = np.random.default_rng(args.seed+8000)
    batch_digest, mask_digest = hashlib.sha256(), hashlib.sha256()
    history, started = [], time.perf_counter()
    warmup = config["optimizer"]["warmup_updates"]
    for step in range(1, steps+1):
        chosen = rng.choice(data["train"], effective, replace=False)
        batch_digest.update(np.asarray(chosen, dtype="<i8").tobytes())
        mask = random_lead_mask(effective, 12*2500//args.patch_samples, .5,
                                int(mask_rng.integers(0, 2**63-1)))
        if args.model != "ecgcpc":
            mask_digest.update(np.packbits(mask).tobytes())
        factor = step/warmup if step <= warmup else .1+.9*.5*(1+math.cos(math.pi*(step-warmup)/max(1, steps-warmup)))
        # Keep S4 per-parameter LR ratios instead of overwriting them.
        for group in opt.param_groups:
            group.setdefault("initial_lr", group["lr"])
            group["lr"] = group["initial_lr"]*factor
        model.train()
        opt.zero_grad(set_to_none=True)
        loss_value = 0.
        if device.type == "cuda":
            torch.cuda.synchronize()
        step_started = time.perf_counter()
        for micro in range(args.accum_steps):
            a, b = micro*args.microbatch, (micro+1)*args.microbatch
            x = get_batch(data, "train", chosen[a:b], device)
            with torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext():
                if args.model == "ecgcpc":
                    loss = model.ssl_loss(x)
                else:
                    hidden = torch.from_numpy(mask[a:b]).to(device)
                    pred, target = model(x, hidden)
                    loss = (pred.float()-target.float()).square().mean(-1)[hidden].mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite {args.model} loss at update {step}")
            loss_value += float(loss.detach())/args.accum_steps
            scaler.scale(loss/args.accum_steps).backward()
        scaler.unscale_(opt)
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(grad):
            raise RuntimeError(f"Nonfinite gradients at update {step}")
        scaler.step(opt)
        scaler.update()
        if device.type == "cuda":
            torch.cuda.synchronize()
        history.append(dict(step=step, loss=loss_value, grad_norm_before_clip=float(grad),
                            lr=args.lr*factor, update_seconds=time.perf_counter()-step_started))
        if step == 1 or step % max(1, steps//10) == 0:
            pd.DataFrame(history).to_csv(output/"learning_history.csv", index=False)
            print(f"{args.model} update {step}/{steps}; loss={loss_value:.6f}; seconds={history[-1]['update_seconds']:.3f}", flush=True)
    pretrain_seconds = time.perf_counter()-started
    peak = torch.cuda.max_memory_allocated() if device.type == "cuda" else None
    torch.save(dict(model=model.state_dict(), config=config), output/"final.pt")
    trained_probe = None if args.mode == "profile" else probe(model, data, device, args, output, "trained")
    result = dict(status="completed", mode=args.mode, model=args.model, parameters=config["parameters"],
        pretrained_weights=False, mask_sampler_compatible=args.model != "ecgcpc", random_probe=random_probe,
        trained_probe=trained_probe, reconstruction=evaluate_reconstruction(model, data, device, args),
        pretrain_seconds=pretrain_seconds, gpu_peak_allocated_bytes=peak,
        median_update_seconds=float(np.median([r["update_seconds"] for r in history[5:] or history])),
        batch_sequence_sha256=batch_digest.hexdigest(),
        mask_sequence_sha256=mask_digest.hexdigest() if args.model != "ecgcpc" else None,
        test_opened=False, eligible_for_model_selection=args.mode == "pilot" and trained_probe["all_five_classes_supported"])
    json_write(output/"summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return output


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, choices=["stmem", "patchecg", "ecgcpc"])
    p.add_argument("--mode", choices=["smoke", "profile", "pilot"], default="profile")
    p.add_argument("--artifact-dir", type=Path, required=True)
    p.add_argument("--external-root", type=Path, default=HERE.parents[1]/"external")
    p.add_argument("--output-root", type=Path, default=HERE/"artifacts/sota_comparison_bp067_50")
    p.add_argument("--device", choices=["auto", "cpu"], default="auto")
    p.add_argument("--steps", type=int)
    p.add_argument("--train-ecgs", type=int)
    p.add_argument("--microbatch", type=int, default=4)
    p.add_argument("--accum-steps", type=int, default=8)
    p.add_argument("--probe-epochs", type=int, default=100)
    p.add_argument("--patch-samples", type=int, choices=[50], default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--subset-seed", type=int, default=42)
    p.add_argument("--eval-mask-seed", type=int, default=42)
    a = p.parse_args(argv)
    for field in ["steps", "train_ecgs", "microbatch", "accum_steps", "probe_epochs"]:
        if getattr(a, field) is not None and getattr(a, field) < 1:
            p.error(f"--{field.replace('_','-')} must be positive")
    if a.lr <= 0 or min(a.seed, a.subset_seed, a.eval_mask_seed) < 0:
        p.error("Positive LR and nonnegative seeds required")
    return a


if __name__ == "__main__":
    run(arguments())
