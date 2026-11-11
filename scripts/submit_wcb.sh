#!/usr/bin/env bash
set -euo pipefail
if (( $# < 2 || $# > 4 )); then
  echo 'Usage: bash scripts/submit_wcb.sh STUDY_DIR prepare|profile|run|test [DEPENDENCY_JOB_ID|-] [TASK_INDEX]' >&2
  exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export ECG_REPO="${ECG_REPO:-$(cd -- "$script_dir/.." && pwd)}"
export ECG_PYTHON="${ECG_PYTHON:-/data/gykwak03/envs/ecg-mask-pilot/bin/python}"
study="$(cd -- "$1" && pwd)"
stage="$2"
[[ "$stage" =~ ^(prepare|profile|run|test)$ ]] || exit 2
[[ -x "$ECG_PYTHON" ]] || { echo "Missing Python: $ECG_PYTHON" >&2; exit 2; }
cd "$ECG_REPO"
"$ECG_PYTHON" experiments/run_wcb.py validate --study-dir "$study" >&2
log_dir="${ECG_LOG_DIR:-/data/gykwak03/capstone2/logs}"
mkdir -p "$log_dir"
hours="${ECG_JOB_HOURS:-24}"
[[ "$hours" =~ ^[1-9][0-9]{0,2}$ ]] && (( hours <= 144 )) || exit 2
args=(--parsable --job-name="wcb-$stage" --account="${ECG_ACCOUNT:-ugrad_advisor_x}"
  --qos="${ECG_QOS:-qos_gykwak03_2026_2}" --partition="${ECG_PARTITION:-batch_ugrad_advisor_x}"
  --ntasks=1 --cpus-per-task="${ECG_CPUS:-4}" --mem="${ECG_MEMORY:-32G}"
  --time="${hours}:00:00" --chdir="$ECG_REPO" --export=ALL
  --output="$log_dir/wcb-$stage-%A_%a.out" --error="$log_dir/wcb-$stage-%A_%a.err")
if [[ "$stage" != prepare ]]; then args+=(--gres="${ECG_GRES:-gpu:normal:1}"); fi
if (( $# >= 3 )) && [[ "$3" != '-' ]]; then
  [[ "$3" =~ ^[0-9]+$ ]] || { echo 'Expected dependency job ID or -' >&2; exit 2; }
  args+=(--dependency="afterok:$3")
fi
if [[ "$stage" == run || "$stage" == test ]]; then
  count="$("$ECG_PYTHON" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))))' "$study/tasks.json")"
  parallel="${ECG_MAX_PARALLEL:-2}"
  [[ "$parallel" =~ ^[1-9][0-9]*$ ]] || exit 2
  array="0-$((count-1))%$parallel"
  if (( $# == 4 )); then
    [[ "$4" =~ ^[0-9]+$ ]] && (( $4 < count )) || exit 2
    array="$4"
  fi
  args+=(--array="$array")
fi
sbatch "${args[@]}" "$ECG_REPO/scripts/wcb_worker.sh" "$study" "$stage"
