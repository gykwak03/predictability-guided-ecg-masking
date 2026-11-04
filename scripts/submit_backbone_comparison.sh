#!/usr/bin/env bash
# One seed per job, two backbones serially on the SAME GPU. No job is auto-submitted.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export ECG_REPO="${ECG_REPO:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
: "${ECG_PYTHON:?Set ECG_PYTHON to the Seraph Python executable}"
if (( $# != 1 )); then
  printf 'Usage: bash scripts/submit_backbone_comparison.sh /absolute/path/to/study\n' >&2
  exit 2
fi
[[ -x "$ECG_PYTHON" ]] || { printf 'Missing Python: %s\n' "$ECG_PYTHON" >&2; exit 2; }
cd "$ECG_REPO"
study_dir="$(cd -- "$1" && pwd)"
[[ -f "$study_dir/plan.json" ]] || { printf 'Missing frozen plan.json\n' >&2; exit 2; }
partition="${ECG_PARTITION:-batch_ugrad_advisor_x}"
hours="${ECG_JOB_HOURS:-24}"
if ! [[ "$hours" =~ ^[0-9]+$ ]] || (( hours < 1 || hours > 24 )); then
  printf 'ECG_JOB_HOURS must be an integer in [1,24]\n' >&2; exit 2
fi
if [[ "$partition" == debug_* ]] && (( hours > 4 )); then
  printf 'Debug partition requires ECG_JOB_HOURS <= 4\n' >&2; exit 2
fi
seed_count="$("$ECG_PYTHON" experiments/run_backbone_comparison.py seed-count --study-dir "$study_dir")"
if ! [[ "$seed_count" =~ ^[1-9][0-9]*$ ]] || (( seed_count > 10 )); then
  printf 'Expected 1-10 seeds; split larger studies to respect submission limits\n' >&2; exit 2
fi
ECG_LOG_DIR="${ECG_LOG_DIR:-/data/gykwak03/capstone2/logs}"
mkdir -p "$ECG_LOG_DIR"
sbatch --parsable --job-name=ecg-backbone-matched \
  --account="${ECG_ACCOUNT:-ugrad_advisor_x}" --qos="${ECG_QOS:-ugrad}" \
  --partition="$partition" --gres=gpu:normal:1 --ntasks=1 \
  --cpus-per-task="${ECG_CPUS:-4}" --mem="${ECG_MEMORY:-32G}" --time="${hours}:00:00" \
  --array="0-$((seed_count-1))%1" --chdir="$ECG_REPO" --export=ALL \
  --output="$ECG_LOG_DIR/backbone-%A_%a.out" --error="$ECG_LOG_DIR/backbone-%A_%a.err" \
  "$ECG_REPO/scripts/backbone_comparison_worker.sh" "$study_dir"
