#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export ECG_REPO="${ECG_REPO:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
export ECG_PYTHON="${ECG_PYTHON:-/data/gykwak03/envs/ecg-mask-pilot/bin/python}"
if (( $# < 1 || $# > 2 )); then
  printf 'Usage: bash scripts/submit_leadwise_comparison.sh STUDY_DIR [MODEL_INDEX_0_TO_3]\n' >&2
  exit 2
fi
array='0-3%4'
if (( $# == 2 )); then
  [[ "$2" =~ ^[0-3]$ ]] || { printf 'Model index must be 0-3\n' >&2; exit 2; }
  array="$2"
fi
study_dir="$(cd -- "$1" && pwd)"
cd "$ECG_REPO"
[[ -x "$ECG_PYTHON" ]] || { printf 'Missing Python: %s\n' "$ECG_PYTHON" >&2; exit 2; }
partition="${ECG_PARTITION:-batch_ugrad_advisor_x}"
hours="${ECG_JOB_HOURS:-24}"
if ! [[ "$hours" =~ ^[1-9][0-9]{0,2}$ ]] || (( hours > 144 )); then
  printf 'ECG_JOB_HOURS must be an integer in [1,144]\n' >&2; exit 2
fi
if [[ "$partition" == debug_* ]] && (( hours > 4 )); then
  printf 'Debug partition requires ECG_JOB_HOURS<=4\n' >&2; exit 2
fi
"$ECG_PYTHON" experiments/run_leadwise_comparison.py validate --study-dir "$study_dir"
log_dir="${ECG_LOG_DIR:-/data/gykwak03/capstone2/logs}"
mkdir -p "$log_dir"
sbatch_args=(
  --parsable --job-name=ecg-leadwise
  --account="${ECG_ACCOUNT:-ugrad_advisor_x}" --qos="${ECG_QOS:-qos_gykwak03_2026_2}"
  --partition="$partition" --gres=gpu:normal:1 --ntasks=1
  --cpus-per-task="${ECG_CPUS:-4}" --mem="${ECG_MEMORY:-32G}" --time="${hours}:00:00"
  --array="$array" --chdir="$ECG_REPO" --export=ALL
  --output="$log_dir/leadwise-%A_%a.out" --error="$log_dir/leadwise-%A_%a.err"
)
sbatch "${sbatch_args[@]}" "$ECG_REPO/scripts/leadwise_comparison_worker.sh" "$study_dir"
