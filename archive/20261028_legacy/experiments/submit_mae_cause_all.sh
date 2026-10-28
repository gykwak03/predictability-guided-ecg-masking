#!/usr/bin/env bash
# Submit all 24 scientific conditions, three seeds, and expanded frozen probes.
# No packages are installed and no model checkpoints are downloaded by this file.
set -euo pipefail

if [[ "${1:-}" != "--worker" ]]; then
  if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--dry-run" ) ]]; then
    printf 'Usage: MAE_PARTITION=<allowed batch partition> bash experiments/submit_mae_cause_all.sh [--dry-run]\n' >&2
    exit 2
  fi
  : "${MAE_PARTITION:?Set MAE_PARTITION to a batch partition allowed by show-assoc/show-qos.}"
  here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
  export MAE_REPO_DIR=$(cd -- "$here/.." && pwd -P)
  server_root=$(cd -- "$MAE_REPO_DIR/.." && pwd -P)
  log_dir=${MAE_LOG_DIR:-$server_root/logs}
  # Pass an absolute repository path: Slurm runs a spool copy of this script.
  command=(sbatch --export=ALL --account="${MAE_ACCOUNT:-ugrad_advisor_x}"
    --partition="$MAE_PARTITION" --nodes=1 --ntasks=1 --gres="${MAE_GRES:-gpu:normal:1}"
    --cpus-per-task=4 --mem=16G --time="${MAE_TIME:-04:00:00}"
    --job-name=mae-cause-all --chdir="$MAE_REPO_DIR"
    --output="$log_dir/mae-cause-all_%j.log" --error="$log_dir/mae-cause-all_%j.log")
  if [[ -n "${MAE_QOS:-}" ]]; then command+=(--qos="$MAE_QOS"); fi
  command+=("$here/submit_mae_cause_all.sh" --worker)
  if [[ "${1:-}" == "--dry-run" ]]; then
    printf 'Full study: 24 conditions x seeds 42,43,44; expanded probe audit enabled.\n'
    printf 'Study directory: %s\n' "${MAE_WORK_DIR:-$MAE_REPO_DIR/experiments/artifacts/mae_cause_all}"
    printf '%q ' "${command[@]}"; printf '\n'
    exit 0
  fi
  mkdir -p "$log_dir"
  exec "${command[@]}"
fi

if [[ $# -ne 1 || -z "${SLURM_JOB_ID:-}" || -z "${MAE_REPO_DIR:-}" ]]; then
  printf 'Worker mode requires a Slurm allocation and exported MAE_REPO_DIR.\n' >&2
  exit 2
fi
repo=$MAE_REPO_DIR
server_root=$(cd -- "$repo/.." && pwd -P)
py=${MAE_PYTHON:-$server_root/../envs/ecg-mask-pilot/bin/python}
work=${MAE_WORK_DIR:-$repo/experiments/artifacts/mae_cause_all}
artifact=${MAE_ARTIFACT_DIR:-$server_root/ptbxl_250hz_bp067_50}
if [[ -z "${MAE_ARTIFACT_DIR:-}" && ! -f "$artifact/manifest.json" ]]; then
  artifact=$repo/preprocessing/artifacts/ptbxl_250hz_bp067_50
fi
raw=${MAE_RAW_DIR:-$server_root/ptb-xl/ptb-xl}
if [[ -z "${MAE_RAW_DIR:-}" && ! -d "$raw/records500" ]]; then raw=$server_root/ptb-xl; fi
upstream=${MAE_STMEM_REPO:-$server_root/ST-MEM}
if [[ -z "${MAE_STMEM_REPO:-}" && ! -f "$upstream/models/st_mem.py" ]]; then
  upstream=$server_root/external/ST-MEM
fi
[[ -x "$py" ]] || { printf 'Python missing: %s\n' "$py" >&2; exit 2; }
[[ -f "$artifact/manifest.json" ]] || { printf '50 Hz artifact missing: %s\n' "$artifact" >&2; exit 2; }
[[ -d "$raw/records500" ]] || { printf 'Raw PTB-XL missing: %s; set MAE_RAW_DIR.\n' "$raw" >&2; exit 2; }
[[ -f "$upstream/models/st_mem.py" ]] || { printf 'Official ST-MEM missing: %s; set MAE_STMEM_REPO.\n' "$upstream" >&2; exit 2; }
cd -- "$repo"
mkdir -p "$work"
# Serialize jobs using this study directory, including jobs on different nodes.
command -v flock >/dev/null || { printf 'flock is required to prevent concurrent writes.\n' >&2; exit 2; }
exec 9>"$work/.study.lock"
flock -n 9 || { printf 'Another job is using this study directory: %s\n' "$work" >&2; exit 2; }
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
printf 'Repository: %s\nPython: %s\nArtifact: %s\nRaw: %s\nST-MEM: %s\nStudy: %s\n' \
  "$repo" "$py" "$artifact" "$raw" "$upstream" "$work"

# Check the real allocated GPU, imports, official implementation and full plan.
srun "$py" - "$repo" "$artifact" "$work" "$upstream" <<'PY'
import json
import sys
from pathlib import Path
import torch, numpy, pandas, scipy, wfdb, sklearn, matplotlib, einops

repo, artifact, work, upstream = map(Path, sys.argv[1:])
sys.path.insert(0, str(repo / "experiments"))
from mae_cause_plan import make_plan, validate_plan
from mae_cause_data import write_json
from st_mem_adapter import source_metadata, load_official_class

if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable in the Slurm allocation")
config = json.loads((artifact / "manifest.json").read_text())["config"]
if config["bandpass_low"] != .67 or config["bandpass_high"] != 50:
    raise ValueError("Full study requires the 0.67-50 Hz artifact")
source_metadata(upstream)
load_official_class(upstream)
expected = make_plan(artifact)
assert len(expected["conditions"]) == 24 and expected["seeds"] == [42, 43, 44]
path = work / "plan.json"
if path.exists():
    existing = validate_plan(json.loads(path.read_text()))
    if existing["plan_id"] != expected["plan_id"]:
        raise ValueError("Existing plan differs from the full default study; use a new MAE_WORK_DIR")
else:
    write_json(path, expected)
print("GPU:", torch.cuda.get_device_name(0))
print("Versions:", {"torch": str(torch.__version__), "numpy": numpy.__version__,
                    "pandas": pandas.__version__, "wfdb": wfdb.__version__})
print("24 conditions x 3 seeds = 72 pretraining runs; expanded probes enabled")
print("Pretraining optimizer updates:", sum(c["steps"] for c in expected["conditions"].values()) * 3)
PY

if [[ ! -e "$work/mv_cache" ]]; then
  srun "$py" -u experiments/run_mae_cause_study.py prepare \
    --plan "$work/plan.json" --raw-dir "$raw" --cache "$work/mv_cache"
elif [[ ! -f "$work/mv_cache/manifest.json" ]]; then
  printf 'Incomplete cache: %s/mv_cache. Preserve it and use a new MAE_WORK_DIR.\n' "$work" >&2
  exit 2
fi

# No --conditions/--seeds restriction and no --no-probe: run the full plan.
# Keep flags and versions identical when resubmitting this job after a timeout.
status=0
srun "$py" -u experiments/run_mae_cause_study.py run \
  --plan "$work/plan.json" --cache "$work/mv_cache" --output-root "$work/study" \
  --st-mem-repo "$upstream" --probe-audit --resume || status=$?

# Also summarize completed runs after an ordinary run error; time-limit kills
# may prevent this step. Resubmission skips completed matching runs.
if [[ -f "$work/study/plan.json" ]]; then
  analysis_status=0
  srun "$py" -u experiments/analyze_mae_cause_study.py \
    --root "$work/study" --output "$work/analysis-job${SLURM_JOB_ID}" || analysis_status=$?
  if [[ $status -eq 0 ]]; then status=$analysis_status; fi
fi
exit "$status"
