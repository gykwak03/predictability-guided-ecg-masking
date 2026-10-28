"""ECG-JEPA mask-unit pilot: train folds 1–8, select on fold 9, never open fold 10.

This is a from-scratch PTB-XL pilot of the Sehun Kim implementation, not a
reproduction of its CODE-15/Chapman/Ningbo large-data performance.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
import uuid

import numpy as np
import pandas as pd
import torch

from ecg_jepa_adapter import ECGJEPAAdapter, sample_masks
from run_sota_comparison import load_data, get_batch, json_write, probe

HERE = Path(__file__).resolve().parent


def pilot_plan():
    conditions = []
    for patch in (25, 50, 125):
        for ratio in (.5, .6, .7):
            conditions.append(dict(stage="primary" if ratio == .5 else "sensitivity",
                patch_samples=patch, patch_seconds=patch/250, mask_ratio=ratio,
                leads=12, topology="lead-time", total_tokens=12*2500//patch,
                hidden_tokens=round(12*2500//patch*ratio)))
    # One paired topology anchor at our existing P50 / 50% condition. This
    # retains 12 leads, so it isolates topology, not faithful paper reproduction.
    conditions.append(dict(stage="topology_anchor", patch_samples=50, patch_seconds=.2,
        mask_ratio=.5, leads=12, topology="time-aligned", total_tokens=600, hidden_tokens=300))
    return dict(model="Sehun Kim ECG-JEPA / official ViT-Base", initialization="scratch",
        selection_metric="fold-9 frozen plain linear-probe macro AUROC; not latent loss",
        test_opened=False, seeds=[42, 43, 44], conditions=conditions,
        budget=dict(train_ecgs=5000, updates=1000, effective_batch=128, probe_epochs=100),
        caveats=["12 leads and independent lead-time masks adapt the published 8-lead model",
                 "All conditions have equal updates and ECG exposures, not equal FLOPs or wall time",
                 "Equal hidden counts per lead constrain the mask family",
                 "No GPU runtime or scientific outcome has been measured by writing this plan"])


def run(args):
    if args.mode == "plan":
        plan = pilot_plan()
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return plan
    torch.set_num_threads(4)
    cuda = torch.cuda.is_available() and args.device != "cpu"
    if args.mode != "smoke" and not cuda:
        raise RuntimeError("Profile/pilot require an allocated CUDA GPU. Only reduced smoke may run on CPU.")
    device = torch.device("cuda" if cuda else "cpu")
    micro = args.microbatch or (2 if args.mode == "smoke" else 4)
    args.microbatch = micro
    accumulation = args.accum_steps or (32 if args.mode == "pilot" else 1)
    effective = micro*accumulation
    train_n = args.train_ecgs or (5000 if args.mode == "pilot" else 64 if args.mode == "profile" else 32)
    updates = args.steps or (1000 if args.mode == "pilot" else 100 if args.mode == "profile" else 2)
    if effective > train_n:
        raise ValueError("Effective batch exceeds train subset; reduce microbatch/accumulation")
    # Reuse the checked 50 Hz manifest loader. It enumerates train/validation
    # only, checks patient separation, metadata, labels and official folds.
    data = load_data(args.artifact_dir, train_n, args.subset_seed, 64 if args.mode != "pilot" else 0)
    torch.manual_seed(args.seed)
    model = ECGJEPAAdapter(args.jepa_repo, args.patch_samples, args.leads, args.mode == "smoke")
    initialized_from = model.load_released_encoder(args.checkpoint) if args.checkpoint else None
    initial_digest = model.initial_backbone_digest()
    model.to(device)
    # Validate mask counts before starting expensive probes/training.
    sample_masks(1, model.c, model.n, args.mask_ratio, 0, args.topology)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output_root/f"{stamp}_{args.mode}_P{model.p}_r{args.mask_ratio:g}_{args.topology}_s{args.seed}_{uuid.uuid4().hex[:6]}"
    output.mkdir(parents=True, exist_ok=False)
    warmup = max(1, math.ceil(updates*.05))
    amp = cuda and args.precision != "fp32"
    dtype = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() and args.precision == "auto" else torch.float16
    precision = str(dtype) if amp else "torch.float32"
    train_ids = data["frames"]["train"].iloc[data["train"]].ecg_id.tolist()
    val_ids = data["frames"]["validation"].iloc[data["validation"]].ecg_id.tolist()
    config = dict(model="sehun_ecg_jepa", mode=args.mode, seed=args.seed, subset_seed=args.subset_seed,
        source=model.source, source_manifest_sha256=data["manifest_sha256"], architecture=model.architecture,
        initialization="continued_pretraining" if initialized_from else "scratch", checkpoint=initialized_from,
        initial_backbone_sha256=initial_digest, patch_samples=model.p, patch_seconds=model.p/250,
        lead_indices=model.lead_indices, leads=model.c, total_tokens=model.c*model.n,
        mask_ratio=args.mask_ratio, topology=args.topology, mask_distribution="equal hidden count per lead",
        steps=updates, effective_batch=effective, microbatch=micro, accum_steps=accumulation,
        train_ecg_ids=train_ids, validation_ecg_ids=val_ids, precision=precision,
        parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        optimizer=dict(name="AdamW", lr=args.lr, betas=[.9, .999], weight_decay=.05,
            warmup_updates=warmup, final_lr=1e-6, gradient_clip=1.),
        teacher=dict(ema_start=.996, ema_end=1., eval_mode=True),
        probe=dict(epochs=args.probe_epochs, head="one linear layer", encoder="student frozen",
            lr=.001, weight_decay=.0001, batch=128, selection="validation macro AUROC"),
        objective="masked latent SmoothL1 against layer-normalized EMA teacher; not waveform NRMSE",
        test_opened=False, gpu=dict(name=torch.cuda.get_device_name(0),
            total_memory_bytes=torch.cuda.get_device_properties(0).total_memory) if cuda else None)
    json_write(output/"run_config.json", config)
    data["frames"]["train"].iloc[data["train"]][["ecg_id", "patient_id", "artifact_index"]].to_csv(
        output/"train_ecg_selection.csv", index=False)
    print(f"OUTPUT={output}\n{config['parameters']:,} trainable parameters; {effective} ECG/update; {precision}", flush=True)
    random_probe = None if args.mode == "profile" else probe(model, data, device, args, output, "initial")
    # Probe training changes torch RNG: restore it for paired SSL conditions.
    torch.manual_seed(args.seed+300)
    if cuda:
        torch.cuda.manual_seed_all(args.seed+300)
        torch.cuda.reset_peak_memory_stats()
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (no_decay if p.ndim <= 1 or name.endswith("bias") else decay).append(p)
    optimizer = torch.optim.AdamW([dict(params=decay, weight_decay=.05),
        dict(params=no_decay, weight_decay=0.)], lr=args.lr, betas=(.9, .999))
    scaler = torch.amp.GradScaler("cuda", enabled=amp and dtype == torch.float16)
    batch_rng = np.random.default_rng(args.seed+7000)
    mask_rng = np.random.default_rng(args.seed+8000)
    batch_digest, mask_digest = hashlib.sha256(), hashlib.sha256()
    history = []
    started = time.perf_counter()
    for step in range(1, updates+1):
        if cuda:
            torch.cuda.synchronize()
        step_started = time.perf_counter()
        chosen = batch_rng.choice(data["train"], effective, replace=False)
        mask = sample_masks(effective, model.c, model.n, args.mask_ratio,
                            int(mask_rng.integers(0, 2**63-1)), args.topology)
        batch_digest.update(np.asarray(chosen, dtype="<i8").tobytes())
        mask_digest.update(np.packbits(mask).tobytes())
        progress = (step-warmup)/max(1, updates-warmup)
        lr = args.lr*step/warmup if step <= warmup else 1e-6+(args.lr-1e-6)*.5*(1+math.cos(math.pi*progress))
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.
        for part in range(accumulation):
            a, b = part*micro, (part+1)*micro
            x = get_batch(data, "train", chosen[a:b], device)
            hidden = torch.from_numpy(mask[a:b]).to(device)
            with torch.autocast("cuda", dtype=dtype) if amp else nullcontext():
                loss = model.ssl_loss(x, hidden)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite latent loss at update {step}")
            total_loss += float(loss.detach())/accumulation
            scaler.scale(loss/accumulation).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(norm):
            raise RuntimeError(f"Nonfinite gradients at update {step}")
        scaler.step(optimizer)
        scaler.update()
        momentum = .996+.004*step/updates
        model.update_teacher(momentum)
        if cuda:
            torch.cuda.synchronize()
        history.append(dict(update=step, latent_loss=total_loss, lr=lr, teacher_momentum=momentum,
                            gradient_norm=float(norm), update_seconds=time.perf_counter()-step_started))
        if step == 1 or step % max(1, updates//10) == 0:
            pd.DataFrame(history).to_csv(output/"learning_history.csv", index=False)
            print(f"update {step}/{updates}; latent_loss={total_loss:.6f}; seconds={history[-1]['update_seconds']:.3f}", flush=True)
    pretrain_seconds = time.perf_counter()-started
    peak_memory = torch.cuda.max_memory_allocated() if cuda else None
    # Weights for evaluation; this script does not claim exact interrupted-run resume.
    torch.save(dict(model=model.state_dict(), config=config), output/"final.pt")
    trained = None if args.mode == "profile" else probe(model, data, device, args, output, "trained")
    model.eval()
    with torch.no_grad():
        # Small diagnostic only, no test data. Collapsed latent features can
        # have low SSL loss, so record their across-ECG variability as well.
        rows = data["validation"][:32]
        chunks = [model.features(get_batch(data, "validation", rows[i:i+micro], device)).float().cpu()
                  for i in range(0, len(rows), micro)]
        feature_std = float(torch.cat(chunks).std(0, correction=0).mean())
    summary = dict(status="completed", mode=args.mode, pretrain_seconds=pretrain_seconds,
        median_update_seconds=float(np.median([r["update_seconds"] for r in history[5:] or history])),
        gpu_peak_allocated_bytes=peak_memory, initial_probe=random_probe, trained_probe=trained,
        mean_feature_std_across_validation_ecgs=feature_std,
        batch_sequence_sha256=batch_digest.hexdigest(), mask_sequence_sha256=mask_digest.hexdigest(),
        test_opened=False, eligible_for_condition_selection=args.mode == "pilot" and not initialized_from
            and trained["all_five_classes_supported"] and feature_std > 1e-6)
    json_write(output/"summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return output


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["plan", "smoke", "profile", "pilot"], default="plan")
    p.add_argument("--artifact-dir", type=Path, default=HERE.parent/"preprocessing/artifacts/ptbxl_250hz_bp067_50")
    p.add_argument("--jepa-repo", type=Path, default=HERE.parents[1]/"external/ECG_JEPA")
    p.add_argument("--output-root", type=Path, default=HERE/"artifacts/ecg_jepa_pilot_bp067_50")
    p.add_argument("--patch-samples", type=int, choices=[25, 50, 125], default=50)
    p.add_argument("--mask-ratio", type=float, choices=[.5, .6, .7], default=.5)
    p.add_argument("--topology", choices=["lead-time", "time-aligned"], default="lead-time")
    p.add_argument("--leads", type=int, choices=[8, 12], default=12)
    p.add_argument("--checkpoint", type=Path, help="Explicit encoder-only continued-pretraining diagnostic; 8 leads/P50 only")
    p.add_argument("--microbatch", type=int)
    p.add_argument("--accum-steps", type=int)
    p.add_argument("--steps", type=int)
    p.add_argument("--train-ecgs", type=int)
    p.add_argument("--probe-epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--subset-seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=2.5e-5)
    p.add_argument("--precision", choices=["auto", "fp32"], default="auto")
    p.add_argument("--device", choices=["auto", "cpu"], default="auto")
    a = p.parse_args(argv)
    for field in ("microbatch", "accum_steps", "steps", "train_ecgs", "probe_epochs"):
        if getattr(a, field) is not None and getattr(a, field) < 1:
            p.error(f"{field} must be positive")
    if a.lr <= 1e-6 or min(a.seed, a.subset_seed) < 0:
        p.error("LR must exceed the 1e-6 decay floor; seeds must be nonnegative")
    return a


if __name__ == "__main__":
    run(arguments())
