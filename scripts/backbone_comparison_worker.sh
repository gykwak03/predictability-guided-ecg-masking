#!/usr/bin/env bash
set -euo pipefail
: "${ECG_REPO:?}" "${ECG_PYTHON:?}" "${SLURM_JOB_ID:?}" "${SLURM_ARRAY_TASK_ID:?}"
cd "$ECG_REPO"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
printf 'job=%s task=%s repo=%s\n' "$SLURM_JOB_ID" "$SLURM_ARRAY_TASK_ID" "$ECG_REPO"
nvidia-smi --query-gpu=name,memory.total --format=csv
"$ECG_PYTHON" experiments/run_backbone_comparison.py run \
  --study-dir "$1" --seed-index "$SLURM_ARRAY_TASK_ID"
