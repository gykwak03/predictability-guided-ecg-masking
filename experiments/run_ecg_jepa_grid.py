"""Print or sequentially execute a declared ECG-JEPA pilot stage on one GPU."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
import uuid

from run_ecg_jepa_pilot import HERE, pilot_plan


def commands(args):
    seeds = args.seeds or ([42] if args.mode == "profile" else [42, 43, 44])
    if len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError("Use distinct nonnegative seeds")
    result = []
    for condition in pilot_plan()["conditions"]:
        if args.stage != "all" and condition["stage"] != args.stage:
            continue
        for seed in seeds:
            result.append([sys.executable, "-u", str(HERE/"run_ecg_jepa_pilot.py"),
                "--mode", args.mode, "--artifact-dir", str(args.artifact_dir),
                "--jepa-repo", str(args.jepa_repo), "--output-root", str(args.output_root),
                "--patch-samples", str(condition["patch_samples"]), "--mask-ratio", str(condition["mask_ratio"]),
                "--topology", condition["topology"], "--leads", "12", "--seed", str(seed),
                "--microbatch", str(args.microbatch), "--accum-steps", str(args.accum_steps),
                "--steps", str(args.steps), "--train-ecgs", str(args.train_ecgs),
                "--probe-epochs", str(args.probe_epochs)])
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["primary", "sensitivity", "topology_anchor", "all"], default="primary")
    p.add_argument("--mode", choices=["profile", "pilot"], default="profile")
    p.add_argument("--execute", action="store_true", help="Actually run the displayed jobs, sequentially")
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--artifact-dir", type=Path, default=HERE.parent/"preprocessing/artifacts/ptbxl_250hz_bp067_50")
    p.add_argument("--jepa-repo", type=Path, default=HERE.parents[1]/"external/ECG_JEPA")
    p.add_argument("--output-root", type=Path, default=HERE/"artifacts/ecg_jepa_pilot_bp067_50")
    p.add_argument("--microbatch", type=int, default=4)
    p.add_argument("--accum-steps", type=int)
    p.add_argument("--steps", type=int)
    p.add_argument("--train-ecgs", type=int)
    p.add_argument("--probe-epochs", type=int, default=100)
    args = p.parse_args()
    for name in ("microbatch", "accum_steps", "steps", "train_ecgs", "probe_epochs"):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            p.error(f"{name} must be positive")
    args.accum_steps = args.accum_steps or (32 if args.mode == "pilot" else 1)
    args.steps = args.steps or (1000 if args.mode == "pilot" else 100)
    args.train_ecgs = args.train_ecgs or (5000 if args.mode == "pilot" else 64)
    jobs = commands(args)
    for job in jobs:
        print(shlex.join(job), flush=True)
    if args.execute:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("--execute requires a CUDA GPU; plan only was printed")
        tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid.uuid4().hex[:6]
        logs = args.output_root/f"grid_{tag}"
        logs.mkdir(parents=True, exist_ok=False)
        (logs/"commands.json").write_text(json.dumps(jobs, indent=2), encoding="utf-8")
        for index, job in enumerate(jobs, 1):
            with (logs/f"job_{index:02d}.log").open("w", encoding="utf-8") as stream:
                process = subprocess.Popen(job, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, encoding="utf-8", errors="replace")
                for line in process.stdout:
                    print(line, end="", flush=True)
                    stream.write(line)
                    stream.flush()
                if process.wait():
                    raise RuntimeError(f"Job {index} failed; remaining jobs not started. See {logs}")
