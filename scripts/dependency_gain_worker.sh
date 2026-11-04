#!/usr/bin/env bash
set -euo pipefail
: "${ECG_REPO:?Set ECG_REPO}"
: "${ECG_PYTHON:?Set ECG_PYTHON}"
: "${SLURM_JOB_ID:?Worker must run in a Slurm allocation}"
cd "$ECG_REPO"
printf 'job=%s repo=%s python=%s\n' "$SLURM_JOB_ID" "$ECG_REPO" "$ECG_PYTHON"
nvidia-smi --query-gpu=name,memory.total --format=csv
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
"$ECG_PYTHON" -u experiments/run_dependency_gain.py "$@"
