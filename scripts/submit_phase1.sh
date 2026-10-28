#!/usr/bin/env bash
# Run explicitly on Seraph. This script is never invoked by local CPU smoke.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export ECG_REPO="${ECG_REPO:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
: "${ECG_PYTHON:?Set ECG_PYTHON; do not use a Windows .venv on Linux}"
ECG_LOG_DIR="${ECG_LOG_DIR:-/data/gykwak03/capstone2/logs}"
partition="${ECG_PARTITION:-batch_ugrad_advisor_x}"
hours="${ECG_JOB_HOURS:-24}"
if ! [[ "$hours" =~ ^[0-9]+$ ]] || (( hours < 1 || hours > 24 )); then
  printf 'ECG_JOB_HOURS must be an integer in [1,24]\n' >&2
  exit 2
fi
if [[ "$partition" == debug_* ]] && (( hours > 4 )); then
  printf 'Debug partition allows at most 4 hours; set ECG_JOB_HOURS=4\n' >&2
  exit 2
fi
if (( $# == 0 )); then
  printf 'Usage: bash scripts/submit_phase1.sh {prepare|smoke|profile|pilot|estimator-fit|estimator-score} [runner options]\n' >&2
  exit 2
fi
case "$1" in prepare|smoke|profile|pilot|estimator-fit|estimator-score) ;; *) printf 'Unsupported job command: %s\n' "$1" >&2; exit 2;; esac
[[ -x "$ECG_PYTHON" ]] || { printf 'Missing executable: %s\n' "$ECG_PYTHON" >&2; exit 2; }
[[ -f "$ECG_REPO/experiments/run_phase1.py" ]] || { printf 'Invalid ECG_REPO: %s\n' "$ECG_REPO" >&2; exit 2; }
cd "$ECG_REPO"
# Use the runner's resolver even before submission: do not inject shell defaults
# that would accidentally take precedence over paths in a selected config.
resolved_paths="$("$ECG_PYTHON" -c '
import sys
from experiments.run_phase1 import parser
from experiments.phase1.config import DEFAULTS, resolve_config
args = parser().parse_args(sys.argv[1:])
overrides = {k:v for k,v in vars(args).items() if k in DEFAULTS and v is not None}
cfg = resolve_config(args.config, overrides, smoke=args.command == "smoke")
print(cfg["output_root"])
print(cfg["cache_dir"])
' "$@")"
mapfile -t resolved_array <<< "$resolved_paths"
export ECG_OUTPUT_ROOT="${resolved_array[0]}"
export ECG_CACHE_DIR="${resolved_array[1]}"
mkdir -p "$ECG_LOG_DIR" "$ECG_OUTPUT_ROOT" "$ECG_CACHE_DIR"
# Export explicit resolved path values to the worker. --export=ALL also carries
# optional ECG_RAW_DIR / ECG_ARTIFACT_DIR and the site's required environment.
# Arguments stay separate array elements, preserving paths containing spaces.
sbatch --parsable --job-name="ecg-phase1-$1" \
  --account="${ECG_ACCOUNT:-ugrad_advisor_x}" --qos="${ECG_QOS:-ugrad}" \
  --partition="$partition" --gres=gpu:normal:1 --ntasks=1 --cpus-per-task="${ECG_CPUS:-4}" \
  --mem="${ECG_MEMORY:-32G}" --time="${hours}:00:00" \
  --chdir="$ECG_REPO" --export=ALL \
  --output="$ECG_LOG_DIR/phase1-%x-%j.out" --error="$ECG_LOG_DIR/phase1-%x-%j.err" \
  "$ECG_REPO/scripts/phase1_worker.sh" "$@"
