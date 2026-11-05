#!/usr/bin/env bash
set -euo pipefail
: "${ECG_REPO:?Set ECG_REPO}"
: "${ECG_PYTHON:?Set ECG_PYTHON}"
: "${SLURM_JOB_ID:?Run the worker inside Slurm}"
cd "$ECG_REPO"
printf 'job=%s repo=%s python=%s\n' "$SLURM_JOB_ID" "$ECG_REPO" "$ECG_PYTHON"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
exec "$ECG_PYTHON" -u experiments/profile_stmem_base.py "$@"
