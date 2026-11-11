#!/usr/bin/env bash
set -euo pipefail
: "${ECG_REPO:?}" "${ECG_PYTHON:?}" "${SLURM_JOB_ID:?}"
study="$1"
stage="$2"
cd "$ECG_REPO"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
args=(-u experiments/run_ncf.py "$stage" --study-dir "$study")
key="$stage"
if [[ "$stage" == run || "$stage" == test ]]; then
  : "${SLURM_ARRAY_TASK_ID:?}"
  args+=(--task-index "$SLURM_ARRAY_TASK_ID")
  key="task-${SLURM_ARRAY_TASK_ID}"
fi
if [[ "$stage" != prepare ]]; then nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv; fi
printf 'job=%s stage=%s study=%s\n' "$SLURM_JOB_ID" "$stage" "$study"
exec flock -n -E 75 "$study/.ncf-$key.lock" "$ECG_PYTHON" "${args[@]}"
