#!/bin/bash
# Seraph: extract the reviewed local PTB-XL 1.0.3 ZIP, build 50 Hz data, run all.
# Upload /data/gykwak03/capstone2/ptb-xl.zip before submission.
#SBATCH --job-name=mae-prepare-all
#SBATCH --account=ugrad_advisor_x
#SBATCH --qos=ugrad
#SBATCH --partition=batch_ugrad_advisor_x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:normal:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --chdir=/data/gykwak03/capstone2/predictability-guided-ecg-masking
#SBATCH --output=/data/gykwak03/capstone2/logs/mae-prepare-all_%j.log
#SBATCH --error=/data/gykwak03/capstone2/logs/mae-prepare-all_%j.log
set -euo pipefail
base=/data/gykwak03/capstone2
py=/data/gykwak03/envs/ecg-mask-pilot/bin/python
prefix=ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3
export MAE_REPO_DIR=$base/predictability-guided-ecg-masking
export MAE_ARTIFACT_DIR=$base/ptbxl_250hz_bp067_50
export MAE_RAW_DIR=$base/ptbxl_raw/$prefix
export MAE_PYTHON=$py
export MAE_WORK_DIR=$MAE_REPO_DIR/experiments/artifacts/mae_cause_all
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
cd -- "$MAE_REPO_DIR"

# This is an import check only; never install packages in a running study.
srun "$py" -c 'import torch,numpy,pandas,scipy,wfdb,sklearn,matplotlib,einops,IPython; assert torch.cuda.is_available(); print("GPU:", torch.cuda.get_device_name(0)); print("WFDB:", wfdb.__version__)'

if [[ ! -f "$MAE_RAW_DIR/.records500-extracted" ]]; then
  [[ -f "$base/ptb-xl.zip" ]] || { printf 'Upload %s/ptb-xl.zip first.\n' "$base" >&2; exit 2; }
  mkdir -p "$base/ptbxl_raw"
  printf 'Extracting 500 Hz records and metadata from the reviewed PTB-XL ZIP.\n'
  # This new extraction directory is owned by this preparation job. If a prior
  # extraction was interrupted, re-extract to repair partially written records.
  # The ZIP prefix is from the local archive, not an assumed extraction layout.
  srun unzip -o -q "$base/ptb-xl.zip" \
    "$prefix/records500/*" "$prefix/ptbxl_database.csv" \
    "$prefix/scp_statements.csv" "$prefix/LICENSE.txt" -d "$base/ptbxl_raw"
  touch "$MAE_RAW_DIR/.records500-extracted"
fi

if [[ ! -f "$MAE_ARTIFACT_DIR/manifest.json" ]]; then
  printf 'Building a separate 0.67-50 Hz artifact from original 500 Hz WFDB.\n'
  srun "$py" -u preprocessing/run_preprocessing.py \
    --data-dir "$MAE_RAW_DIR" --output-dir "$MAE_ARTIFACT_DIR"
fi

# Continue in THIS allocation; do not request a second GPU or submit a child job.
bash experiments/submit_mae_cause_all.sh --worker
