#!/usr/bin/env bash
set -euo pipefail
: "${ECG_REPO:?}" "${ECG_PYTHON:?}" "${NCF_STUDY:?}"
cd "$ECG_REPO"
"$ECG_PYTHON" experiments/run_ncf.py validate --study-dir "$NCF_STUDY" >&2
export NCF_DIAG_DIR="${NCF_DIAG_DIR:-$NCF_STUDY/diagnostics/reconstruction_$(date -u +%Y%m%dT%H%M%SZ)}"
test ! -e "$NCF_DIAG_DIR"
log_dir="${ECG_LOG_DIR:-/data/gykwak03/capstone2/logs}"
mkdir -p "$log_dir"
sbatch --parsable --job-name=ncf-diag \
  --account="${ECG_ACCOUNT:-ugrad_advisor_x}" \
  --qos="${ECG_QOS:-qos_gykwak03_2026_2}" \
  --partition="${ECG_PARTITION:-batch_ugrad_advisor_x}" \
  --gres="${ECG_GRES:-gpu:normal:1}" \
  --ntasks=1 --cpus-per-task=4 --mem=16G --time=01:30:00 \
  --chdir="$ECG_REPO" --export=ALL \
  --output="$log_dir/ncf-diag-%j.out" --error="$log_dir/ncf-diag-%j.err" \
  --wrap='set -euo pipefail; export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv; exec flock -n -E 75 "$NCF_STUDY/.ncf-diagnostics.lock" "$ECG_PYTHON" -u scripts/diagnose_ncf_reconstruction.py --study-dir "$NCF_STUDY" --output-dir "$NCF_DIAG_DIR" --untrained-linear-probe'
