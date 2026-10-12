"""Execute the notebook without Jupyter (e.g. on a GPU node).

Default: visualization/contracts only. --train explicitly enables training.
This runner keeps the notebook as the single source of model/training code.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


NOTEBOOK = Path(__file__).with_name("Masking_pilot.ipynb")


def run_notebook(*, mode="smoke", train=False, seed=42,
                 artifact_dir=None, output_root=None, quiet=False, steps=None,
                 patch_samples=None, mask_ratio=None, diagnostics_enabled=True,
                 random_probe=True, subset_seed=None, eval_mask_seed=None, mask_ratios=None):
    if steps is not None and (not isinstance(steps, int) or steps < 1):
        raise ValueError("steps must be a positive integer")
    if mask_ratios is not None:
        if patch_samples not in [50, 125, 250] or mask_ratio is not None or not mask_ratios:
            raise ValueError("mask_ratios requires patch_samples and cannot be combined with mask_ratio")
        if any(not 0 < ratio < 1 for ratio in mask_ratios) or len(set(mask_ratios)) != len(mask_ratios):
            raise ValueError("mask_ratios must be distinct values between 0 and 1")
        if len({int(round(ratio*100)) for ratio in mask_ratios}) != len(mask_ratios):
            raise ValueError("mask_ratios must have distinct rounded percentage directory names")
    elif (patch_samples is None) != (mask_ratio is None):
        raise ValueError("Pass both patch_samples and mask_ratio for a single condition")
    if mask_ratios is None and patch_samples is not None and (patch_samples not in [50, 125, 250] or not 0 < mask_ratio < 1):
        raise ValueError("patch_samples must be 50/125/250 and 0 < mask_ratio < 1")
    import matplotlib
    matplotlib.use("Agg")
    from IPython.utils.capture import capture_output

    namespace = {"__name__": "__masking_pilot__"}
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    original_cwd = Path.cwd()
    try:
        os.chdir(NOTEBOOK.parent.parent)
        for number, cell in enumerate(notebook["cells"], 1):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            if 'MODE = "smoke"' in source:
                source = source.replace('MODE = "smoke"', f"MODE = {mode!r}", 1)
                source = source.replace("RUN_TRAINING = False", f"RUN_TRAINING = {train!r}", 1)
                source = source.replace("SEED = 42", f"SEED = {seed!r}", 1)
                source = source.replace("SUBSET_SEED = None", f"SUBSET_SEED = {subset_seed!r}", 1)
                source = source.replace("EVAL_MASK_SEED = None", f"EVAL_MASK_SEED = {eval_mask_seed!r}", 1)
                source = source.replace("STEPS_OVERRIDE = None", f"STEPS_OVERRIDE = {steps!r}", 1)
                source = source.replace("ARTIFACT_DIR_OVERRIDE = None",
                                        f"ARTIFACT_DIR_OVERRIDE = {str(artifact_dir)!r}" if artifact_dir else
                                        "ARTIFACT_DIR_OVERRIDE = None", 1)
                source = source.replace("OUTPUT_ROOT_OVERRIDE = None",
                                        f"OUTPUT_ROOT_OVERRIDE = {str(output_root)!r}" if output_root else
                                        "OUTPUT_ROOT_OVERRIDE = None", 1)
                # Overrides are explicit and persisted in run_config.json.
                source += f"\nDIAGNOSTICS['enabled'] = {diagnostics_enabled!r}\n"
                source += f"DIAGNOSTICS['random_encoder_probe'] = {random_probe!r}\n"
                if mask_ratios is not None:
                    source += f"GRID = {[(patch_samples, ratio) for ratio in mask_ratios]!r}\n"
                elif patch_samples is not None:
                    source += f"GRID = [({patch_samples!r}, {mask_ratio!r})]\n"
            if not quiet:
                print(f"Executing notebook cell {number}", flush=True)
            # Suppress rich tables outside Jupyter but show progress text immediately.
            with capture_output(stdout=quiet, stderr=quiet, display=True):
                exec(compile(source, f"{NOTEBOOK.name}:cell{number}", "exec"), namespace)
        return namespace
    finally:
        os.chdir(original_cwd)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "pilot"], default="smoke")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", help="Run paired repeats, e.g. 42 43 44; requires --train")
    parser.add_argument("--subset-seed", type=int, help="Fix ECG subsets independently of model seed")
    parser.add_argument("--eval-mask-seed", type=int, help="Fix validation masks independently of model seed")
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--steps", type=int, help="Override MAE updates; recorded in the run config")
    parser.add_argument("--patch-samples", type=int, choices=[50, 125, 250])
    parser.add_argument("--mask-ratio", type=float, help="Together with --patch-samples, run one condition")
    parser.add_argument("--mask-ratios", type=float, nargs="+", help="Compare selected ratios for --patch-samples")
    parser.add_argument("--no-diagnostics", action="store_true", help="Keep the original final-only evaluation")
    parser.add_argument("--no-random-probe", action="store_true", help="Skip the additional random-encoder probe")
    parser.add_argument("--tiny-overfit", action="store_true", help="Separate two-ECG fixed-mask memorization check")
    parser.add_argument("--tiny-steps", type=int, default=200)
    args = parser.parse_args()
    if args.seeds is not None:
        if not args.train or args.tiny_overfit or args.no_diagnostics or args.no_random_probe:
            parser.error("--seeds requires --train with diagnostics/random probe enabled, without --tiny-overfit")
        if len(set(args.seeds)) != len(args.seeds):
            parser.error("--seeds must not contain duplicates")
    if args.mask_ratios is not None:
        if args.patch_samples is None or args.mask_ratio is not None or args.tiny_overfit:
            parser.error("--mask-ratios requires --patch-samples, without --mask-ratio/--tiny-overfit")
        if any(not 0 < ratio < 1 for ratio in args.mask_ratios) or len(set(args.mask_ratios)) != len(args.mask_ratios):
            parser.error("--mask-ratios must be distinct values between 0 and 1")
        if len({int(round(ratio*100)) for ratio in args.mask_ratios}) != len(args.mask_ratios):
            parser.error("--mask-ratios have colliding rounded percentage directory names")
    elif (args.patch_samples is None) != (args.mask_ratio is None):
        parser.error("Use --patch-samples and --mask-ratio together")
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be positive")
    if args.mask_ratio is not None and not 0 < args.mask_ratio < 1:
        parser.error("--mask-ratio must be between 0 and 1")
    if args.tiny_steps < 1:
        parser.error("--tiny-steps must be positive")
    if args.tiny_overfit and args.train:
        parser.error("--tiny-overfit is separate from --train; do not combine them")
    if args.seeds is not None:
        from datetime import datetime, timezone
        import uuid
        from pilot_comparison import summarize_runs
        root = (args.output_root or NOTEBOOK.parent / "artifacts/masking_pilot").resolve()
        suite = root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                        + "_multiseed_" + uuid.uuid4().hex[:8])
        suite.mkdir(parents=True)
        subset_seed = args.subset_seed if args.subset_seed is not None else args.seeds[0]
        eval_seed = args.eval_mask_seed if args.eval_mask_seed is not None else args.seeds[0]
        runs = []
        for seed in args.seeds:
            result = run_notebook(mode=args.mode, train=True, seed=seed,
                                  artifact_dir=args.artifact_dir.resolve() if args.artifact_dir else None,
                                  output_root=suite, steps=args.steps,
                                  patch_samples=args.patch_samples, mask_ratio=args.mask_ratio,
                                  subset_seed=subset_seed, eval_mask_seed=eval_seed, mask_ratios=args.mask_ratios)
            runs.append(result["RUN_DIR"])
            summarize_runs(runs, suite, expected_seeds=args.seeds)
        print("Completed seed comparison. Output:", suite, flush=True)
        return
    result = run_notebook(mode=args.mode, train=args.train, seed=args.seed,
                          artifact_dir=args.artifact_dir.resolve() if args.artifact_dir else None,
                          output_root=args.output_root.resolve() if args.output_root else None,
                          steps=args.steps, patch_samples=args.patch_samples, mask_ratio=args.mask_ratio,
                          diagnostics_enabled=not args.no_diagnostics, random_probe=not args.no_random_probe,
                          subset_seed=args.subset_seed, eval_mask_seed=args.eval_mask_seed, mask_ratios=args.mask_ratios)
    if args.tiny_overfit:
        if not result["TRAINING_AVAILABLE"]:
            parser.error("Install PyTorch and scikit-learn before tiny overfit")
        if args.mode == "pilot" and result["DEVICE"].type != "cuda":
            parser.error("Pilot-sized tiny overfit requires an allocated CUDA GPU; use --mode smoke on CPU")
        p, ratio = args.patch_samples or 125, args.mask_ratio if args.mask_ratio is not None else 0.75
        result["seed_everything"](args.seed)
        model = result["PlainECGMAE"](p, result["PROFILE"]).to(result["DEVICE"])
        model.ratio = ratio
        root = result["OUTPUT_ROOT"]
        root.mkdir(parents=True, exist_ok=True)
        stamp = result["datetime"].now(result["timezone"].utc).strftime("%Y%m%dT%H%M%SZ")
        directory = root / f"{stamp}_tiny_overfit_{args.mode}_seed{args.seed}_{result['uuid'].uuid4().hex[:8]}"
        history = result["diagnostics"].run_tiny_overfit(
            model, result["arrays"]["train"]["x"], result["frames"]["train"].ecg_id.to_numpy(),
            result["TRAIN_IDX"][:2], args.tiny_steps, args.seed, directory,
            lead_order=result["LEADS"], fs=result["FS"])
        result["save_learning_curves"](history, directory, p, ratio)
        config_path = directory / "run_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config.update(profile=result["PROFILE"], source_manifest_sha256=result["SOURCE_MANIFEST_SHA256"])
        result["diagnostics"].write_json(config_path, config)
        print("Completed tiny overfit. Output:", directory, flush=True)
        return
    print("Completed. Output:", result.get("RUN_DIR"), flush=True)


if __name__ == "__main__":
    main()
