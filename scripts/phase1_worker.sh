#!/usr/bin/env bash
set -euo pipefail
: "${ECG_REPO:?Set ECG_REPO to the checked-out repository}"
: "${ECG_PYTHON:?Set ECG_PYTHON to the server Python executable}"
: "${SLURM_JOB_ID:?Real-data worker must run inside Slurm}"
cd "$ECG_REPO"
printf 'job=%s repo=%s python=%s\n' "$SLURM_JOB_ID" "$ECG_REPO" "$ECG_PYTHON"
"$ECG_PYTHON" "$ECG_REPO/experiments/run_phase1.py" "$@"
