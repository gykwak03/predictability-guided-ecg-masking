#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export ECG_REPO="${ECG_REPO:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
export ECG_PYTHON="${ECG_PYTHON:-/data/gykwak03/envs/ecg-mask-pilot/bin/python}"
if (( $# != 1 )); then
  printf 'Usage: bash scripts/submit_stmem_base_baseline.sh /absolute/path/to/Base-study\n' >&2; exit 2
fi
[[ -x "$ECG_PYTHON" ]] || { printf 'Missing Python: %s\n' "$ECG_PYTHON" >&2; exit 2; }
cd "$ECG_REPO"
study_dir="$(cd -- "$1" && pwd)"
partition="${ECG_PARTITION:-batch_ugrad_advisor_x}"
hours="${ECG_JOB_HOURS:-24}"
if ! [[ "$hours" =~ ^[0-9]+$ ]] || (( hours < 1 || hours > 24 )); then
  printf 'ECG_JOB_HOURS must be an integer in [1,24]\n' >&2; exit 2
fi
if [[ "$partition" == debug_* ]] && (( hours > 4 )); then
  printf 'Debug partition requires ECG_JOB_HOURS<=4\n' >&2; exit 2
fi
# Checks the frozen plan/source only; does not import Torch or open ECG arrays.
seed_count="$("$ECG_PYTHON" experiments/run_stmem_base_baseline.py seed-count --study-dir "$study_dir")"
if ! [[ "$seed_count" =~ ^[1-9][0-9]*$ ]] || (( seed_count > 10 )); then
  printf 'Expected 1-10 seeds; split larger studies\n' >&2; exit 2
fi
log_dir="${ECG_LOG_DIR:-/data/gykwak03/capstone2/logs}"
mkdir -p "$log_dir"
sbatch --parsable --job-name=stmem-base-3000 \
  --account="${ECG_ACCOUNT:-ugrad_advisor_x}" --qos="${ECG_QOS:-ugrad}" \
  --partition="$partition" --gres=gpu:normal:1 --ntasks=1 \
  --cpus-per-task="${ECG_CPUS:-4}" --mem="${ECG_MEMORY:-32G}" --time="${hours}:00:00" \
  --array="0-$((seed_count-1))%1" --chdir="$ECG_REPO" --export=ALL \
  --output="$log_dir/stmem-base-%A_%a.out" --error="$log_dir/stmem-base-%A_%a.err" \
  "$ECG_REPO/scripts/stmem_base_baseline_worker.sh" "$study_dir"
