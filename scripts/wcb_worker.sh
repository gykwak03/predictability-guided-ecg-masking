#!/usr/bin/env bash
set -euo pipefail
: "${ECG_REPO:?}" "${ECG_PYTHON:?}" "${SLURM_JOB_ID:?}"
study="$1"
stage="$2"
cd "$ECG_REPO"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
args=(-u experiments/run_wcb.py "$stage" --study-dir "$study")
key="$stage"
if [[ "$stage" == run || "$stage" == test ]]; then
  : "${SLURM_ARRAY_TASK_ID:?}"
  args+=(--task-index "$SLURM_ARRAY_TASK_ID")
  key="task-${SLURM_ARRAY_TASK_ID}"
fi
if [[ "$stage" == prepare && -n "${ECG_ANNOTATION_CACHE:-}" ]]; then args+=(--annotation-cache "$ECG_ANNOTATION_CACHE"); fi
if [[ "$stage" != prepare ]]; then nvidia-smi --query-gpu=name,memory.total --format=csv; fi
printf 'job=%s stage=%s key=%s study=%s\n' "$SLURM_JOB_ID" "$stage" "$key" "$study"
exec flock -n -E 75 "$study/.$key.lock" "$ECG_PYTHON" "${args[@]}"
