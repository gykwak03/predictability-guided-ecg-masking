#!/usr/bin/env bash
set -euo pipefail
: "${ECG_REPO:?}" "${ECG_PYTHON:?}" "${SLURM_JOB_ID:?}" "${SLURM_ARRAY_TASK_ID:?}"
cd "$ECG_REPO"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
printf 'job=%s task=%s repo=%s python=%s\n' "$SLURM_JOB_ID" "$SLURM_ARRAY_TASK_ID" "$ECG_REPO" "$ECG_PYTHON"
nvidia-smi --query-gpu=name,memory.total --format=csv
exec "$ECG_PYTHON" -u experiments/run_stmem_base_baseline.py run \
  --study-dir "$1" --seed-index "$SLURM_ARRAY_TASK_ID"
