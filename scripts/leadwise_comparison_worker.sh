#!/usr/bin/env bash
set -euo pipefail
: "${ECG_REPO:?}" "${ECG_PYTHON:?}" "${SLURM_JOB_ID:?}" "${SLURM_ARRAY_TASK_ID:?}"
models=(plain_small plain_base stmem_small stmem_base)
[[ "$SLURM_ARRAY_TASK_ID" =~ ^[0-3]$ ]] || exit 2
model="${models[$SLURM_ARRAY_TASK_ID]}"
cd "$ECG_REPO"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
printf 'job=%s task=%s model=%s\n' "$SLURM_JOB_ID" "$SLURM_ARRAY_TASK_ID" "$model"
nvidia-smi --query-gpu=name,memory.total --format=csv
# Keep a duplicate submission from writing to the same model checkpoints.
exec flock -n -E 75 "$1/.$model.lock" "$ECG_PYTHON" -u experiments/run_leadwise_comparison.py run +  --study-dir "$1" --model "$model"
