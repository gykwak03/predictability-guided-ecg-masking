"""Plan, prepare, run and audit controlled explanations of the failed plain MAE.

No GPU training is launched by planning/preparation. Scientific runs require CUDA;
the separate synthetic smoke mode is never treated as experimental evidence.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import pandas as pd
import torch

import masking_diagnostics as diagnostics
from mae_cause_data import CLASSES, digest, file_sha, fit_statistics, load_cache, normalize, prepare_cache, write_json
from mae_cause_eval import evaluate, masks_for_ids, probe_audit
from mae_cause_model import CauseMAE, learning_rate, mask_batch
from mae_cause_plan import make_plan, validate_plan
from st_mem_adapter import STMEMAdapter, source_metadata

HERE = Path(__file__).resolve().parent
SOURCE_FILES = ["run_mae_cause_study.py", "mae_cause_model.py", "mae_cause_data.py", "mae_cause_plan.py",
                "mae_cause_eval.py", "analyze_mae_cause_study.py", "Masking_pilot.ipynb", "masking_diagnostics.py", "st_mem_adapter.py"]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def state_digest(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def runtime_signature(device):
    return dict(torch=str(torch.__version__), numpy=np.__version__, python=sys.version.split()[0],
        cuda=torch.version.cuda, device=device.type, matmul_precision=torch.get_float32_matmul_precision(),
        allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        gpu_name=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        compute_capability=list(torch.cuda.get_device_capability(device)) if device.type == "cuda" else None)


def synthetic_data(plan):
    rng = np.random.default_rng(1729)
    arrays, frames = {}, {}
    t = np.arange(2500) / 250
    for split in ("train", "validation"):
        ids = plan["ecg_ids"][split]
        x = np.empty((len(ids), 12, 2500), dtype=np.float32)
        for i in range(len(ids)):
            wave = np.sin(2 * np.pi * (1 + .01 * i) * t)
            x[i] = np.stack([(1 + c / 10) * wave + .1 * np.cos(2 * np.pi * 8 * t + c) for c in range(12)])
        y = rng.integers(0, 2, (len(ids), 5))
        y[0], y[1] = 0, 1
        frames[split] = pd.DataFrame(dict(ecg_id=ids, patient_id=np.asarray(ids) // 2,
            **{c: y[:, i] for i, c in enumerate(CLASSES)}))
        arrays[split] = x
    return dict(arrays=arrays, frames=frames, stats=fit_statistics(arrays["train"]), cache_id="synthetic-smoke")


def build_model(spec, upstream):
    seed_all(spec["model_seed"])
    if spec["backend"] == "plain":
        return CauseMAE(spec)
    # Reduced official model is only for smoke; native Small is the scientific anchor.
    return STMEMAdapter(upstream, patch_samples=50, model_size="smoke" if spec.get("smoke") else "small")


def contract_audit(model, data, spec, plan, device):
    """Tests direct model isolation; separately reports preprocessing dependence."""
    raw = np.asarray(data["arrays"]["train"][:2], dtype=np.float32).copy()
    mask = mask_batch(len(raw), 600, plan["mask_ratio"], 123, spec["train_policy"])
    x, _, _ = normalize(raw, spec["normalization"], data["stats"], mask=mask)
    hidden = np.repeat(mask, 50, axis=1).reshape(raw.shape)
    previous = model.training
    with diagnostics.preserve_rng(device):
        model.eval()
        xt = torch.from_numpy(x).to(device)
        mt = torch.from_numpy(mask).to(device)
        with torch.no_grad():
            pred, target = model(xt, mt)
            changed = xt.clone()
            changed[torch.from_numpy(hidden).to(device)] += 100
            other, _ = model(changed, mt)
            if not torch.allclose(pred, other, atol=2e-5, rtol=2e-5):
                raise AssertionError("Masked input entered model predictions")
            if not torch.equal(target, xt.reshape(len(xt), 600, 50)):
                raise AssertionError("Patch lead/time order changed")
        model.train()
        pred, target = model(xt, mt)
        loss = ((pred - target.detach()).square().mean(-1) * mt).sum() / mt.sum()
        loss.backward()
        trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        missing = [n for n, p in trainable if p.grad is None]
        nonfinite = [n for n, p in trainable if p.grad is not None and not torch.isfinite(p.grad).all()]
        if missing or nonfinite:
            raise AssertionError(f"Gradient audit failed: missing={missing}, nonfinite={nonfinite}")
        model.zero_grad(set_to_none=True)
        model.train(previous)
    changed_raw = raw.copy()
    changed_raw[hidden] += 5
    preprocessing = []
    for mode in ("per_record", "train_lead", "train_common", "visible_record"):
        a = normalize(raw, mode, data["stats"], mask=mask)[0]
        b = normalize(changed_raw, mode, data["stats"], mask=mask)[0]
        preprocessing.append(dict(normalization=mode, visible_max_change=float(np.abs(a - b)[~hidden].max()),
            expected_hidden_stat_dependence=mode == "per_record"))
    return dict(direct_hidden_input_isolated=True, patch_order_correct=True, finite_gradients=True,
                initialization=diagnostics.initialization_report(model), preprocessing=preprocessing,
                note="Perturbation is AFTER filtering. This does not claim raw hidden samples cannot influence a zero-phase filter.")


def gradient_snapshot(model):
    groups, before = defaultdict(list), {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        groups[diagnostics._group(name)].append((name, p))
        before[name] = p.detach().clone()
    rows = []
    for group, values in groups.items():
        rows.append(dict(group=group, missing_gradients=sum(p.grad is None for _, p in values),
            grad_norm=float(torch.stack([p.grad.norm() for _, p in values if p.grad is not None]).norm()),
            weight_norm=float(torch.stack([p.detach().norm() for _, p in values]).norm())))
    return rows, before


def save_checkpoint(path, value):
    temp = path.with_suffix(".tmp")
    torch.save(value, temp)
    temp.replace(path)


def train_run(plan, data, name, seed, args):
    spec = dict(plan["conditions"][name], model_seed=seed, smoke=plan["smoke"])
    kind = "synthetic-smoke" if plan["smoke"] else "tiny-fixed-mask" if args.tiny_overfit else "scientific"
    device = torch.device(args.device)
    if not plan["smoke"] and device.type != "cuda":
        raise RuntimeError("Scientific and real-data tiny runs require an allocated CUDA GPU; CPU smoke is separate")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; request a GPU node")
    runtime = runtime_signature(device)
    root = args.output_root / ("tiny" if args.tiny_overfit else "runs") / name / f"seed{seed}"
    hashes = {f: file_sha(HERE / f) for f in SOURCE_FILES}
    upstream = source_metadata(args.st_mem_repo) if spec["backend"] == "stmem" else None
    fingerprint = digest(dict(plan=plan["plan_id"], cache=data["cache_id"], spec=spec, sources=hashes,
                              upstream=upstream, kind=kind, probe=not args.no_probe, expanded=args.probe_audit, runtime=runtime,
                              tiny_steps=args.tiny_steps if args.tiny_overfit else None))
    if (root / "summary.json").exists():
        summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        if summary["fingerprint"] != fingerprint:
            raise ValueError(f"Existing run differs from code/settings: {root}; choose a new output root")
        print("Already complete:", root, flush=True)
        return root
    if root.exists() and not args.resume:
        raise FileExistsError(f"Incomplete run: {root}; use --resume with its own checkpoint or a new output root")
    root.mkdir(parents=True, exist_ok=True)
    make_model = lambda: build_model(spec, args.st_mem_repo)
    model = make_model().to(device)
    model.ratio = plan["mask_ratio"]
    config = dict(fingerprint=fingerprint, plan_id=plan["plan_id"], cache_id=data["cache_id"], condition=name,
        seed=seed, spec=spec, kind=kind, source_sha256=hashes, upstream=upstream,
        parameters=sum(p.numel() for p in model.parameters()), precision="fp32",
        test_opened=False, torch_version=str(torch.__version__), numpy_version=np.__version__, runtime=runtime,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        gpu=dict(name=torch.cuda.get_device_name(device), memory_bytes=torch.cuda.get_device_properties(device).total_memory) if device.type == "cuda" else None,
        effective_batch=plan["effective_batch"], microbatch=plan["microbatch"], accumulation=plan["accumulation"])
    if (root / "run_config.json").exists():
        old = json.loads((root / "run_config.json").read_text(encoding="utf-8"))
        if old["fingerprint"] != fingerprint:
            raise ValueError("Cannot resume under changed source/settings")
    else:
        write_json(root / "run_config.json", config)
    diagnostics.write_json(root / "contract_audit.json", contract_audit(model, data, spec, plan, device))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
        lr=plan["optimizer"]["lr"], betas=tuple(plan["optimizer"]["betas"]), weight_decay=plan["optimizer"]["weight_decay"])
    rng = np.random.default_rng(seed)
    start, history, evaluation_rows, gradients = 0, [], [], []
    streams = dict(batch="", mask="", lr="")
    milestones, train_seconds = {}, 0.
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if args.resume and (root / "latest.pt").exists():
        checkpoint = torch.load(root / "latest.pt", map_location=device, weights_only=False)
        if checkpoint["fingerprint"] != fingerprint:
            raise ValueError("Resume checkpoint mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start, history, evaluation_rows, gradients = (checkpoint[k] for k in ("update", "history", "evaluations", "gradients"))
        rng.bit_generator.state = checkpoint["batch_rng"]
        streams, milestones, train_seconds = (checkpoint[k] for k in ("streams", "milestones", "train_seconds"))
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint["cuda_rng"]])
        initial = torch.load(root / "initial.pt", map_location="cpu", weights_only=True)
    elif (root / "initial.pt").exists():
        raise RuntimeError("Run has no resumable checkpoint; use a new output root")
    else:
        torch.save(initial, root / "initial.pt")
    training_indices = np.arange(len(data["frames"]["train"]))
    val_indices = np.arange(len(data["frames"]["validation"]))
    # Preserve the old pilot's first ECGs; arrays/IDs are pinned by the plan/cache.
    diagnostic_train = training_indices[:min(plan["diagnostic_ecgs"], len(training_indices))]
    diagnostic_val = val_indices[:min(plan["diagnostic_ecgs"], len(val_indices))]
    linear = diagnostics.fit_cross_lead_baseline(data["arrays"]["train"], training_indices[:256])
    banks = [b for b in plan["evaluation_banks"] if spec["backend"] != "stmem" or b == "per_lead"]
    total = args.tiny_steps if args.tiny_overfit else spec["steps"]
    if args.tiny_overfit:
        training_indices = training_indices[:2]
        diagnostic_train = training_indices
    interval = min(plan["eval_every"], total)

    def periodic(update):
        for split, indices in [("train", diagnostic_train), ("validation", diagnostic_val)]:
            for bank in banks:
                metrics, _ = evaluate(model, data, spec, plan, device, split=split, indices=indices, bank=bank)
                evaluation_rows.append(dict(update=update, split=split, bank=bank, **metrics))
        latest = evaluation_rows[-1]
        print(f"{name} seed={seed} {update}/{total}: fixed-val NRMSE={latest['standardized_nrmse']:.4f}, r={latest['masked_correlation']}", flush=True)

    if not start:
        periodic(0)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for update in range(start + 1, total + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        tick = time.perf_counter()
        model.train()
        if args.tiny_overfit:
            chosen = training_indices
            mask_seed = seed + 1000
        else:
            chosen = rng.choice(training_indices, plan["effective_batch"], replace=False)
            mask_seed = seed + 1000 + update - 1
        mask = mask_batch(len(chosen), 600, plan["mask_ratio"], mask_seed, spec["train_policy"])
        lr = plan["optimizer"]["lr"] if args.tiny_overfit else learning_rate(update,
            horizon=spec["schedule_horizon"], warmup=plan["optimizer"]["warmup"], base=plan["optimizer"]["lr"])
        for group in optimizer.param_groups:
            group["lr"] = lr
        for key, value in [("batch", np.asarray(chosen, dtype="<i8").tobytes()), ("mask", mask.tobytes()), ("lr", np.asarray(lr, dtype="<f8").tobytes())]:
            streams[key] = hashlib.sha256(streams[key].encode() + value).hexdigest()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.
        for offset in range(0, len(chosen), plan["microbatch"]):
            ix = chosen[offset:offset + plan["microbatch"]]
            hidden_np = mask[offset:offset + len(ix)]
            raw = np.asarray(data["arrays"]["train"][ix], dtype=np.float32)
            x, _, _ = normalize(raw, spec["normalization"], data["stats"], mask=hidden_np)
            hidden = torch.from_numpy(hidden_np).to(device)
            pred, target = model(torch.from_numpy(x).to(device), hidden)
            loss = ((pred - target).square().mean(-1) * hidden).sum() / hidden.sum()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite reconstruction loss at update {update}")
            (loss * len(ix) / len(chosen)).backward()
            loss_sum += float(loss.detach()) * len(ix) / len(chosen)
        sample_grad = update == 1 or update % interval == 0 or update == total
        if sample_grad:
            grad_rows, before = gradient_snapshot(model)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), plan["optimizer"]["grad_clip"], error_if_nonfinite=True)
        optimizer.step()
        if sample_grad:
            changes = defaultdict(list)
            for param_name, p in model.named_parameters():
                if p.requires_grad:
                    changes[diagnostics._group(param_name)].append((p.detach() - before[param_name]).norm())
            for row in grad_rows:
                row.update(update=update, update_norm=float(torch.stack(changes[row["group"]]).norm()))
                gradients.append(row)
            del before
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = time.perf_counter() - tick
        train_seconds += seconds
        history.append(dict(update=update, normalized_train_mse=loss_sum, lr=lr,
                            grad_norm=float(norm), seconds=seconds, train_seconds=train_seconds))
        if update % interval == 0 or update in {total, spec["schedule_horizon"]}:
            periodic(update)
            milestones[str(update)] = dict(streams=streams.copy(), state_sha256=state_digest(model.state_dict()))
            pd.DataFrame(history).to_csv(root / "training_history.csv", index=False)
            pd.DataFrame(evaluation_rows).to_csv(root / "evaluation_history.csv", index=False)
            pd.DataFrame(gradients).to_csv(root / "gradient_updates.csv", index=False)
            save_checkpoint(root / "latest.pt", dict(fingerprint=fingerprint, model=model.state_dict(), optimizer=optimizer.state_dict(),
                update=update, history=history, evaluations=evaluation_rows, gradients=gradients,
                batch_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                streams=streams.copy(), milestones=milestones, train_seconds=train_seconds))
    final_rows, final = [], {}
    for bank in banks:
        metrics, per_ecg = evaluate(model, data, spec, plan, device, split="validation",
            indices=val_indices[:plan["final_ecgs"]] if plan["final_ecgs"] else val_indices,
            bank=bank, repeats=plan["repeats"], baseline=linear,
            example=root / f"waveform_{bank}")
        final[bank] = metrics
        final_rows.append(per_ecg)
    pd.concat(final_rows, ignore_index=True).to_csv(root / "final_per_ecg.csv", index=False)
    if args.tiny_overfit:
        fixed = mask_batch(2, 600, plan["mask_ratio"], seed + 1000, spec["train_policy"])
        tiny_metrics, tiny_rows = evaluate(model, data, spec, plan, device, split="train",
            indices=training_indices, bank=spec["train_policy"], fixed_masks=fixed,
            example=root / "tiny_training_fixed_mask")
        tiny_rows.to_csv(root / "tiny_training_per_ecg.csv", index=False)
        diagnostics.write_json(root / "tiny_training_metrics.json", tiny_metrics)
    probes = []
    if not args.no_probe and not args.tiny_overfit:
        with diagnostics.preserve_rng(device):
            probes = probe_audit(model, initial, data, spec, plan, device, root, make_model, expanded=args.probe_audit)
    summary = dict(fingerprint=fingerprint, plan_id=plan["plan_id"], cache_id=data["cache_id"],
        condition=name, seed=seed, kind=kind, spec=spec, final=final, probe=probes,
        milestones=milestones, train_seconds=train_seconds, parameters=config["parameters"],
        gpu_peak_allocated_mb=torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
        timing_scope="Sum of training updates including gradient audits; excludes evaluation/probe/checkpoint I/O",
        memory_scope="Current process peak including probes; previous preempted segments excluded",
        unsupported_banks=sorted(set(plan["evaluation_banks"]) - set(banks)), completed=True)
    diagnostics.write_json(root / "summary.json", summary)
    print("Completed:", root, flush=True)
    return root


def probe_saved_run(plan, data, name, seed, args):
    """Audit a completed encoder without spending another pretraining run."""
    folder = args.output_root / "runs" / name / f"seed{seed}"
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    config = json.loads((folder / "run_config.json").read_text(encoding="utf-8"))
    if summary["plan_id"] != plan["plan_id"] or summary["cache_id"] != data["cache_id"]:
        raise ValueError("Probe data/plan differ from the encoder run")
    if config["source_sha256"] != {f: file_sha(HERE / f) for f in SOURCE_FILES}:
        raise ValueError("Encoder code has changed since pretraining; inspect before probing")
    device = torch.device(args.device)
    if not plan["smoke"] and (device.type != "cuda" or not torch.cuda.is_available()):
        raise RuntimeError("Real-data probes require allocated CUDA; CPU smoke is separate")
    if config["runtime"] != runtime_signature(device):
        raise ValueError("Runtime differs from encoder run; use the same versions/device/precision for this paired probe audit")
    spec = summary["spec"]
    if spec["backend"] == "stmem" and config["upstream"] != source_metadata(args.st_mem_repo):
        raise ValueError("Official source changed")
    make_model = lambda: build_model(spec, args.st_mem_repo)
    model = make_model().to(device)
    checkpoint = torch.load(folder / "latest.pt", map_location=device, weights_only=False)
    if checkpoint["fingerprint"] != summary["fingerprint"] or checkpoint["update"] != spec["steps"]:
        raise ValueError("Not the completed encoder checkpoint")
    model.load_state_dict(checkpoint["model"])
    initial = torch.load(folder / "initial.pt", map_location="cpu", weights_only=True)
    output = folder / ("probe_audit_expanded" if args.probe_audit else "probe_audit_standard")
    if output.exists():
        raise FileExistsError("Probe audit already exists; keep it as evidence instead of overwriting")
    output.mkdir()
    with diagnostics.preserve_rng(device):
        rows = probe_audit(model, initial, data, spec, plan, device, output, make_model, expanded=args.probe_audit)
    diagnostics.write_json(output / "audit.json", dict(parent_fingerprint=summary["fingerprint"],
        kind=summary["kind"], plan_id=plan["plan_id"], cache_id=data["cache_id"], results=rows,
        selection="fixed classifier updates; no checkpoint selection", test_opened=False))
    return output


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    p = subs.add_parser("plan")
    p.add_argument("--artifact-dir", type=Path, default=HERE.parent / "preprocessing/artifacts/ptbxl_250hz_bp067_50")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stages", nargs="+", choices=["controls", "architecture", "normalization"], default=["controls", "architecture", "normalization"])
    p.add_argument("--size", choices=["baseline", "stmem-small"], default="stmem-small")
    p.add_argument("--ratio", type=float, default=.5)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--train-ecgs", type=int, default=5000)
    p.add_argument("--validation-ecgs", type=int, default=0)
    p.add_argument("--subset-seed", type=int, default=42)
    p.add_argument("--eval-seed", type=int, default=42)
    p.add_argument("--short", type=int, default=1000)
    p.add_argument("--long", type=int, default=3000)
    p.add_argument("--microbatch", type=int, default=4)
    p.add_argument("--accumulation", type=int, default=8)
    p.add_argument("--diagnostic-ecgs", type=int, default=32)
    p.add_argument("--final-ecgs", type=int, default=128, help="0 = all validation ECGs; default reproduces the pilot evaluation size")
    p.add_argument("--smoke", action="store_true")
    p = subs.add_parser("prepare")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--raw-dir", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p = subs.add_parser("run")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--cache", type=Path)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--conditions", nargs="+")
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--st-mem-repo", type=Path, default=HERE.parents[1] / "ST-MEM")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-probe", action="store_true")
    p.add_argument("--probe-audit", action="store_true", help="Also compare pooling and full/visible input using frozen identical weights")
    p.add_argument("--tiny-overfit", action="store_true", help="Separate two-ECG fixed-mask diagnostic; never pooled with real comparisons")
    p.add_argument("--tiny-steps", type=int, default=200)
    p = subs.add_parser("probe")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--cache", type=Path)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--conditions", nargs="+")
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--st-mem-repo", type=Path, default=HERE.parents[1] / "ST-MEM")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--probe-audit", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    torch.set_num_threads(4)
    if args.command == "plan":
        options = vars(args).copy()
        output = options.pop("output")
        options.pop("command")
        if args.smoke:
            options.update(train_ecgs=32, validation_ecgs=16, microbatch=2, accumulation=1, short=2, long=4)
        plan = make_plan(**options)
        if plan["smoke"]:
            plan.pop("plan_id")
            plan.update(eval_every=2, diagnostic_ecgs=2, final_ecgs=4, repeats=1, probe_checkpoints=[2, 4])
            plan["plan_id"] = digest(plan)
        if output.exists():
            raise FileExistsError("Plan already exists; use a new filename")
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json(output, plan)
        print(f"{len(plan['conditions'])} conditions x {len(plan['seeds'])} seeds; {sum(c['steps'] for c in plan['conditions'].values()) * len(plan['seeds']):,} optimizer updates. Plan only: {output}")
        return plan
    plan = validate_plan(json.loads(args.plan.read_text(encoding="utf-8")))
    if args.command == "prepare":
        return prepare_cache(plan, args.raw_dir, args.cache)
    if args.command == "run" and args.tiny_steps < 1:
        raise ValueError("Tiny steps must be positive")
    if not plan["smoke"] and args.cache is None:
        raise ValueError("Scientific run requires --cache prepared from raw physical signals")
    names, seeds = args.conditions or list(plan["conditions"]), args.seeds or plan["seeds"]
    if not set(names) <= set(plan["conditions"]) or not set(seeds) <= set(plan["seeds"]):
        raise ValueError("Conditions/seeds must be predeclared in the plan")
    data = synthetic_data(plan) if plan["smoke"] else load_cache(plan, args.cache)
    args.output_root.mkdir(parents=True, exist_ok=True)
    existing = args.output_root / "plan.json"
    if existing.exists() and json.loads(existing.read_text(encoding="utf-8"))["plan_id"] != plan["plan_id"]:
        raise ValueError("Output root belongs to another plan")
    if not existing.exists():
        write_json(existing, plan)
    for name in names:
        for seed in seeds:
            if args.command == "probe":
                probe_saved_run(plan, data, name, seed, args)
            else:
                train_run(plan, data, name, seed, args)


if __name__ == "__main__":
    main()
