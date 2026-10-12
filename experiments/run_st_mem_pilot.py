"""ST-MEM + existing PTB-XL artifacts; one condition, no automatic grid search.

Use ST_MEM_pilot.ipynb for explanations, or this runner on a Slurm GPU node.
Official model code is imported from --st-mem-repo, never copied or edited.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch

import masking_diagnostics as diagnostics
from st_mem_adapter import STMEMAdapter, fixed_lead_mask, mask_layout, random_lead_mask, source_metadata

HERE = Path(__file__).resolve().parent
LEGACY_NOTEBOOK = HERE / "Masking_pilot.ipynb"


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["smoke", "pilot"], default="smoke")
    p.add_argument("--train", action="store_true", help="Without this flag only data/model contracts run")
    p.add_argument("--st-mem-repo", type=Path, default=HERE.parents[1] / "ST-MEM")
    p.add_argument("--artifact-dir", type=Path)
    p.add_argument("--output-root", type=Path, default=HERE / "artifacts/st_mem_pilot")
    p.add_argument("--model-size", choices=["smoke", "small", "base"])
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--patch-samples", type=int, choices=[50, 125, 250], default=50)
    p.add_argument("--mask-ratio", type=float, default=0.75)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--subset-seed", type=int, default=42)
    p.add_argument("--eval-mask-seed", type=int, default=42)
    p.add_argument("--steps", type=int)
    p.add_argument("--batch-size", type=int, help="Microbatch size; default pilot=4, smoke=2")
    p.add_argument("--accum-steps", type=int, help="Default pilot=8 (effective batch 32), smoke=1")
    p.add_argument("--eval-every", type=int)
    p.add_argument("--amp", action="store_true", help="Optional CUDA float16 autocast; default FP32")
    p.add_argument("--no-probe", action="store_true", help="Reconstruction-only run; do not claim downstream benefit")
    p.add_argument("--tiny-overfit", action="store_true", help="Separate fixed two-ECG memorization diagnostic")
    p.add_argument("--tiny-steps", type=int, default=200)
    a = p.parse_args(argv)
    for name in ["steps", "batch_size", "accum_steps", "eval_every", "tiny_steps"]:
        if getattr(a, name) is not None and getattr(a, name) < 1:
            p.error(f"--{name.replace('_', '-')} must be positive")
    for name in ["seed", "subset_seed", "eval_mask_seed"]:
        if getattr(a, name) < 0:
            p.error(f"--{name.replace('_', '-')} must be non-negative")
    try:
        mask_layout(12 * 2500 // a.patch_samples, a.mask_ratio)
    except ValueError as e:
        p.error(str(e))
    if a.tiny_overfit and a.train:
        p.error("--tiny-overfit is separate from --train")
    a.model_size = a.model_size or ("smoke" if a.mode == "smoke" else "small")
    if a.mode == "pilot" and a.model_size == "smoke":
        p.error("The reduced smoke model is not a pilot backbone")
    return a


def prepare_experiment(a):
    """Reuse existing contracts, subsets and probe functions, not the custom MAE.

Execute only the legacy notebook's imports/config/data/definitions cells.
Do not execute its model check, training, plotting grid or selection advice.
"""
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if a.device == "auto" else a.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Request a GPU node, not the login/master node.")
    if a.mode == "pilot" and device.type != "cuda":
        raise RuntimeError("Pilot requires CUDA. Use --mode smoke for local CPU checks.")
    if a.amp and device.type != "cuda":
        raise ValueError("--amp is only supported on CUDA")
    meta = source_metadata(a.st_mem_repo)
    if device.type == "cpu":
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    notebook = json.loads(LEGACY_NOTEBOOK.read_text(encoding="utf-8"))
    ns = {"__name__": "__st_mem_pilot__", "DEVICE": device}
    old_cwd = Path.cwd()
    try:
        os.chdir(HERE.parent)
        from IPython.utils.capture import capture_output
        for number in [4, 6, 8, 10, 14, 18]:
            cell = notebook["cells"][number-1]
            if cell["cell_type"] != "code":
                raise RuntimeError(f"Legacy notebook cell {number} changed; review shared interface")
            source = "".join(cell["source"])
            if number == 6:
                replacements = {'MODE = "smoke"': f"MODE = {a.mode!r}", "SEED = 42": f"SEED = {a.seed!r}",
                    "SUBSET_SEED = None": f"SUBSET_SEED = {a.subset_seed!r}",
                    "EVAL_MASK_SEED = None": f"EVAL_MASK_SEED = {a.eval_mask_seed!r}",
                    "ARTIFACT_DIR_OVERRIDE = None": f"ARTIFACT_DIR_OVERRIDE = {str(a.artifact_dir.resolve())!r}"
                        if a.artifact_dir else "ARTIFACT_DIR_OVERRIDE = None"}
                for old, new in replacements.items():
                    if old not in source:
                        raise RuntimeError(f"Legacy config interface changed: {old}")
                    source = source.replace(old, new, 1)
            with capture_output():
                exec(compile(source, f"{LEGACY_NOTEBOOK.name}:cell{number}", "exec"), ns)
            if number == 6:
                ns["PROFILE"]["batch"] = a.batch_size or (2 if a.mode == "smoke" else 4)
                ns["PROFILE"]["steps"] = a.steps or (8 if a.mode == "smoke" else 1000)
                ns["DIAGNOSTICS"]["eval_every"] = a.eval_every or (4 if a.mode == "smoke" else 200)
                ns["DIAGNOSTICS"]["random_encoder_probe"] = not a.no_probe
                if a.mode == "smoke":
                    ns["DIAGNOSTICS"].update(eval_n=4, grad_every=4, cross_lead_fit_n=8, examples=1)
                ns["GRID"] = [(a.patch_samples, a.mask_ratio)]
        if not ns["TRAINING_AVAILABLE"]:
            raise ImportError("Install torch and scikit-learn into the active environment")
    finally:
        os.chdir(old_cwd)
    ns["seed_everything"](a.seed)
    model = STMEMAdapter(a.st_mem_repo, a.patch_samples, a.model_size).to(device)
    model.ratio = a.mask_ratio
    # Remove custom-model dimensions from the reused profile: actual ST-MEM
    # architecture is recorded separately, never mislabeled as the old model.
    for name in ["enc_dim", "enc_depth", "enc_heads", "dec_dim", "dec_depth", "dec_heads"]:
        ns["PROFILE"].pop(name, None)
    accum = a.accum_steps or (1 if a.mode == "smoke" else 8)
    q, keep = mask_layout(model.n, a.mask_ratio)
    config = dict(backend="st-mem-official", mode=a.mode, model_size=a.model_size,
        seed=a.seed, subset_seed=a.subset_seed, evaluation_mask_seed=a.eval_mask_seed,
        device=str(device), profile=dict(ns["PROFILE"]), architecture=model.architecture,
        upstream=meta, pretrained_weights=False, seq_len=2500, fs=250, patch_samples=model.p,
        nominal_mask_ratio=a.mask_ratio, realized_mask_ratio=1-keep/q,
        visible_patches_per_lead=keep, total_visible_patches=12*keep,
        policy="per-lead-uniform-decimal-floor-visible-count-v2", norm_pix_loss=False,
        target="existing preprocessed waveform; no per-patch target normalization",
        pooling="official-ST-MEM-mean-patch-tokens-before-final-norm-excluding-SEP",
        microbatch_size=ns["PROFILE"]["batch"], accumulation_steps=accum,
        effective_batch_size=min(len(ns["TRAIN_IDX"]), ns["PROFILE"]["batch"]*accum),
        amp=a.amp, probe_enabled=not a.no_probe,
        parameters=sum(p.numel() for p in model.parameters()),
        source_manifest_sha256=ns["SOURCE_MANIFEST_SHA256"], source_preprocessing=ns["manifest"]["config"],
        class_order=ns["CLASSES"], lead_order=ns["LEADS"],
        train_ecg_ids=ns["frames"]["train"].iloc[ns["TRAIN_IDX"]].ecg_id.astype(int).tolist(),
        validation_ecg_ids=ns["frames"]["validation"].iloc[ns["VAL_IDX"]].ecg_id.astype(int).tolist(),
        reconstruction_validation_ecg_ids=ns["frames"]["validation"].iloc[ns["VAL_IDX"][:ns["PROFILE"]["eval_n"]]].ecg_id.astype(int).tolist(),
        pretrain_optimizer=dict(name="AdamW", lr=3e-4, betas=[0.9, 0.95], weight_decay=0.05,
            grad_clip=1., warmup_steps=min(100, max(1, ns["PROFILE"]["steps"]//10)),
            schedule="warmup + cosine to 0.1*base_lr"),
        probe_optimizer=dict(name="AdamW", lr=1e-3, weight_decay=1e-4, loss="BCEWithLogitsLoss",
            steps=ns["PROFILE"]["probe_steps"], threshold=0.5, seed=a.seed+100),
        diagnostics_config=ns["DIAGNOSTICS"], versions=dict(torch=str(torch.__version__), numpy=np.__version__, pandas=pd.__version__),
        implementation_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__), HERE/"st_mem_adapter.py", HERE/"masking_diagnostics.py", LEGACY_NOTEBOOK]},
        note="PTB-XL adaptation, NOT the original paper recipe or an architecture-only ablation. No test split is opened. Fixed final checkpoint; no automatic model/mask selection.")
    print(f"ST-MEM {a.model_size}: {config['parameters']:,} parameters | {device}", flush=True)
    print(f"Patch={model.p}, nominal mask={a.mask_ratio:.0%}, realized mask={config['realized_mask_ratio']:.2%}; "
          f"microbatch={config['microbatch_size']} x accumulation={accum} => batch={config['effective_batch_size']}", flush=True)
    return dict(args=a, namespace=ns, model=model, config=config)


def check_model(ctx):
    ns, model = ctx["namespace"], ctx["model"]
    with diagnostics.preserve_rng(ns["DEVICE"]):
        x = ns["read_batch"]("train", ns["TRAIN_IDX"][:1], ns["DEVICE"])
        mask = torch.from_numpy(random_lead_mask(len(x), model.n, model.ratio, 42)).to(x.device)
        model.eval()
        with torch.no_grad():
            pred, target = model(x, mask)
            changed = x.clone().reshape(len(x), model.n, model.p)
            changed[mask] += 1000
            other, _ = model(changed.reshape_as(x), mask)
            if not torch.allclose(pred, other, atol=2e-5, rtol=2e-5):
                raise AssertionError("Hidden input changes predictions")
            if not torch.allclose(model.core.unpatchify(model.core.patchify(x)), x):
                raise AssertionError("Lead/time patch order changed")
        model.train()
        pred, target = model(x, mask)
        loss = ns["masked_mse"](pred, target, mask)
        loss.backward()
        if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise AssertionError("ST-MEM forward/backward NaN/Inf")
        model.zero_grad(set_to_none=True)
    print("PASS: ST-MEM patch order / hidden-input isolation / finite forward-backward", flush=True)


def _checkpoint(model, ctx, step, probe=None):
    return dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                config=ctx["config"], step=step, probe=probe,
                note="Weights for inference/inspection. Not an exact optimizer/RNG resume checkpoint.")


def train_experiment(ctx):
    a, ns, model, config = (ctx[k] for k in ["args", "namespace", "model", "config"])
    root = a.output_root.resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = root / f"{stamp}_stmem_{a.mode}_{a.model_size}_seed{a.seed}_{uuid.uuid4().hex[:8]}"
    condition = run / f"p{model.p}_r{int(round(model.ratio*100)):02d}"
    device, profile = ns["DEVICE"], ns["PROFILE"]
    if a.tiny_overfit:
        history = diagnostics.run_tiny_overfit(model, ns["arrays"]["train"]["x"],
            ns["frames"]["train"].ecg_id.to_numpy(), ns["TRAIN_IDX"][:2], a.tiny_steps,
            a.eval_mask_seed, condition, lead_order=ns["LEADS"], mask_factory=fixed_lead_mask)
        config.update(kind="tiny-overfit-not-pilot", steps=a.tiny_steps, schedule="constant", learning_rate=3e-4)
        diagnostics.write_json(run / "run_config.json", config)
        torch.save(_checkpoint(model, ctx, a.tiny_steps), condition / "checkpoint.pt")
        ns["save_learning_curves"](history, condition, model.p, model.ratio)
        print("Completed ST-MEM tiny overfit. Output:", run, flush=True)
        return dict(RUN_DIR=run, RESULTS=pd.DataFrame())
    if not a.train:
        return dict(RUN_DIR=None, RESULTS=pd.DataFrame())
    condition.mkdir(parents=True, exist_ok=False)
    diagnostics.write_json(run / "run_config.json", config)
    torch.save(_checkpoint(model, ctx, 0), condition / "initial_checkpoint.pt")
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    ids = {s: ns["frames"][s].ecg_id.to_numpy() for s in ["train", "validation"]}
    baseline = diagnostics.fit_cross_lead_baseline(ns["arrays"]["train"]["x"], ns["LINEAR_FIT_IDX"],
                                                ridge=ns["DIAGNOSTICS"]["cross_lead_ridge"])
    np.savez_compressed(condition / "cross_lead_baseline.npz", **baseline)
    evaluations, gradient_rows = [], []

    def evaluate(split, indices, *, repeats=1, examples=0, linear=None):
        return diagnostics.evaluate_reconstruction(model, ns["arrays"][split]["x"], ids[split],
            indices, profile["batch"], a.eval_mask_seed, repeats=repeats, examples=examples,
            lead_order=ns["LEADS"], fs=ns["FS"], linear_baseline=linear, mask_factory=fixed_lead_mask)

    def periodic(step):
        for split, indices in [("train", ns["TRAIN_DIAGNOSTIC_IDX"]), ("validation", ns["VAL_DIAGNOSTIC_IDX"])]:
            bundle = evaluate(split, indices, repeats=ns["DIAGNOSTICS"]["repeats"],
                              examples=ns["DIAGNOSTICS"]["examples"] if step==0 and split=="validation" else 0)
            evaluations.append(dict(step=step, split=split, **bundle["metrics"]))
            if step==0 and split=="validation":
                diagnostics.save_reconstruction(bundle, condition, "initial")
        if step:
            print(f"Update {step}: train MSE={evaluations[-2]['masked_mse']:.5f}, "
                  f"val MSE={evaluations[-1]['masked_mse']:.5f}, "
                  f"val skill={evaluations[-1]['skill_vs_zero']:.2%}", flush=True)

    print("Output:", run, flush=True)
    periodic(0)
    diagnostics.save_diagnostic_progress(evaluations, gradient_rows, condition, plot=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(.9,.95), weight_decay=.05)
    steps, warmup = profile["steps"], config["pretrain_optimizer"]["warmup_steps"]

    def schedule(step):
        if step < warmup:
            return (step+1)/warmup
        return .1 + .9 * .5 * (1+math.cos(math.pi*(step-warmup)/max(1,steps-warmup)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = torch.amp.GradScaler("cuda", enabled=a.amp)
    rng, history = np.random.default_rng(a.seed), []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    diagnostic_seconds = 0.
    for step in range(1, steps+1):
        model.train()
        chosen = rng.choice(ns["TRAIN_IDX"], config["effective_batch_size"], replace=False)
        optimizer.zero_grad(set_to_none=True)
        batch_loss, zero_mse = 0., 0.
        for micro, start in enumerate(range(0, len(chosen), profile["batch"])):
            x = ns["read_batch"]("train", chosen[start:start+profile["batch"]], device)
            mask = torch.from_numpy(random_lead_mask(len(x), model.n, model.ratio,
                        a.seed+1000+(step-1)*config["accumulation_steps"]+micro)).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=a.amp):
                pred, target = model(x, mask)
                loss = ns["masked_mse"](pred.float(), target.float(), mask)
            if not torch.isfinite(loss):
                raise RuntimeError("ST-MEM loss NaN/Inf; inspect AMP/LR/input")
            weight = len(x)/len(chosen)
            scaler.scale(loss * weight).backward()
            batch_loss += float(loss.detach())*weight
            zero_mse += float((target.detach().float().square().mean(-1)*mask).sum()/mask.sum())*weight
        scaler.unscale_(optimizer)
        sample_grad = step==1 or step % ns["DIAGNOSTICS"]["grad_every"]==0 or step==steps
        if sample_grad:
            grad_rows, before = diagnostics.gradient_snapshot(model, step)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        lr_used = optimizer.param_groups[0]["lr"]
        scaler.step(optimizer)
        scaler.update()
        if sample_grad:
            gradient_rows.extend(diagnostics.complete_update_snapshot(model, grad_rows, before))
            del before, grad_rows
        scheduler.step()
        history.append(dict(step=step, masked_mse=batch_loss, zero_masked_mse=zero_mse,
            skill_vs_zero=1-batch_loss/zero_mse if zero_mse>0 else None,
            grad_norm=float(norm), clipped=float(norm)>1., lr=optimizer.param_groups[0]["lr"], lr_used=lr_used))
        if step % ns["DIAGNOSTICS"]["eval_every"]==0 or step==steps:
            t = time.perf_counter()
            periodic(step)
            pd.DataFrame(history).to_csv(condition / "pretrain_history.csv", index=False)
            diagnostics.save_diagnostic_progress(evaluations, gradient_rows, condition, plot=False)
            if step < steps:
                torch.save(_checkpoint(model, ctx, step), condition / "latest_checkpoint.pt")
            diagnostic_seconds += time.perf_counter()-t
        if step % max(1, steps//5)==0:
            print(f"ST-MEM {step}/{steps}: MSE={batch_loss:.5f}, "
                  f"elapsed={time.perf_counter()-started:.1f}s", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    wall = time.perf_counter()-started
    torch.save(_checkpoint(model, ctx, steps), condition / "checkpoint.pt")
    del optimizer, scheduler, scaler, x, pred, target, loss
    final = evaluate("validation", ns["VAL_IDX"][:profile["eval_n"]], repeats=3,
                     examples=ns["DIAGNOSTICS"]["examples"], linear=baseline)
    diagnostics.save_reconstruction(final, condition, "final")
    diagnostics.save_diagnostic_progress(evaluations, gradient_rows, condition)
    ns["save_learning_curves"](pd.DataFrame(history), condition, model.p, model.ratio)
    result = dict(backend="st-mem-official", model_size=a.model_size, seed=a.seed, mode=a.mode,
        patch_samples=model.p, mask_ratio=model.ratio, realized_mask_ratio=config["realized_mask_ratio"],
        steps=steps, effective_batch_size=config["effective_batch_size"], parameters=config["parameters"],
        train_seconds=wall-diagnostic_seconds, pretrain_wall_seconds=wall,
        train_ecgs=len(ns["TRAIN_IDX"]), labeled_train_ecgs=len(ns["PROBE_TRAIN_IDX"]),
        validation_ecgs=len(ns["PROBE_VAL_IDX"]), validation_masked_mse=final["metrics"]["masked_mse"])
    result.update({"validation_"+k: v for k,v in final["metrics"].items()})
    if not a.no_probe:
        with diagnostics.preserve_rng(device):
            random_model = STMEMAdapter(a.st_mem_repo, model.p, a.model_size).to(device)
            random_model.load_state_dict(initial)
            rm, rd, rp, rs = ns["fit_linear_probe"](random_model)
            diagnostics.write_json(condition / "random_encoder_probe_metrics.json", rm)
            rd.to_csv(condition / "random_encoder_class_metrics.csv", index=False)
            np.save(condition / "random_encoder_probabilities.npy", rp)
            torch.save(rs, condition / "random_encoder_probe.pt")
            del random_model
        metrics, detail, probabilities, probe = ns["fit_linear_probe"](model)
        result.update(metrics)
        for metric in ["macro_auroc", "macro_auprc", "macro_f1"]:
            result["random_encoder_"+metric] = rm[metric]
            result[metric.removeprefix("macro_")+"_gain_vs_random"] = metrics[metric]-rm[metric]
        detail.to_csv(condition / "class_metrics.csv", index=False)
        np.save(condition / "validation_probabilities.npy", probabilities)
        np.save(condition / "validation_ecg_ids.npy", ids["validation"][ns["PROBE_VAL_IDX"]])
        diagnostics.write_json(condition / "feature_statistics.json", dict(trained=probe["feature_diagnostics"], random=rs["feature_diagnostics"]))
        torch.save(_checkpoint(model, ctx, steps, probe), condition / "checkpoint.pt")
    else:
        result["probe_enabled"] = False
    result["gpu_peak_allocated_mb"] = torch.cuda.max_memory_allocated()/2**20 if device.type=="cuda" else None
    results = pd.DataFrame([result])
    results.to_csv(run / "results.csv", index=False)
    print(results.to_string(index=False), flush=True)
    print("Completed ST-MEM. Output:", run, flush=True)
    return dict(RUN_DIR=run, RESULTS=results)


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    a = arguments(argv)
    ctx = prepare_experiment(a)
    check_model(ctx)
    if a.train or a.tiny_overfit:
        return train_experiment(ctx)
    print("Contracts only; add --train to save a new run. No training results generated.")
    return ctx


if __name__ == "__main__":
    main()
