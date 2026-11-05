#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export ECG_REPO="${ECG_REPO:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
export ECG_PYTHON="${ECG_PYTHON:-/data/gykwak03/envs/ecg-mask-pilot/bin/python}"
log_dir="${ECG_LOG_DIR:-/data/gykwak03/capstone2/logs}"
hours="${ECG_JOB_HOURS:-2}"
partition="${ECG_PARTITION:-batch_ugrad_advisor_x}"
if ! [[ "$hours" =~ ^[0-9]+$ ]] || (( hours < 1 || hours > 24 )); then
  printf 'ECG_JOB_HOURS must be an integer in [1,24]\n' >&2
  exit 2
fi
if [[ "$partition" == debug_* ]] && (( hours > 4 )); then
  printf 'Debug partition requires ECG_JOB_HOURS<=4\n' >&2
  exit 2
fi
[[ -x "$ECG_PYTHON" ]] || { printf 'Missing Python: %s\n' "$ECG_PYTHON" >&2; exit 2; }
cd "$ECG_REPO"
# The login-node check imports no Torch and opens no ECG arrays.
"$ECG_PYTHON" experiments/profile_stmem_base.py --plan "$@" >/dev/null
mkdir -p "$log_dir"
sbatch --parsable --job-name=stmem-resource \
  --account="${ECG_ACCOUNT:-ugrad_advisor_x}" --qos="${ECG_QOS:-ugrad}" \
  --partition="$partition" --gres=gpu:normal:1 --ntasks=1 \
  --cpus-per-task="${ECG_CPUS:-4}" --mem="${ECG_MEMORY:-32G}" --time="${hours}:00:00" \
  --chdir="$ECG_REPO" --export=ALL \
  --output="$log_dir/stmem-resource-%j.out" --error="$log_dir/stmem-resource-%j.err" \
  "$ECG_REPO/scripts/stmem_resource_worker.sh" "$@"
