#!/usr/bin/env bash
# One GPU job, with current Seraph account/partition and no stale node binding.
set -euo pipefail
if [[ $# -lt 1 ]]; then
  printf 'Usage: bash experiments/submit_sota.sh {all|stmem|patchecg|ecgcpc} [--mode profile|pilot] [runner options]\n' >&2
  exit 2
fi
model=$1
shift
case "$model" in
  stmem|patchecg|ecgcpc) models=("$model") ;;
  all) models=(stmem patchecg ecgcpc) ;;
  *) printf 'Unknown model\n' >&2; exit 2 ;;
esac
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
root=$(cd -- "$here/../.." && pwd -P)
python_bin=${SOTA_PYTHON:-$root/../envs/ecg-sota/bin/python}
# Same parent convention as the existing /data/gykwak03/envs/ecg-mask-pilot env.
artifact_dir=${SOTA_ARTIFACT_DIR:-$root/ptbxl_250hz_bp067_50}
external_root=${SOTA_EXTERNAL_ROOT:-$root/external}
output_root=$here/artifacts/sota_comparison_bp067_50
log_dir=$root/logs
mkdir -p "$log_dir"
if [[ ! -x $python_bin ]]; then printf 'Missing Python: %s\n' "$python_bin" >&2; exit 2; fi
if [[ ! -f $artifact_dir/manifest.json ]]; then printf 'Missing 50 Hz dataset: %s\n' "$artifact_dir" >&2; exit 2; fi
for arg in "$@"; do
  case "$arg" in --model|--model=*|--output-root|--output-root=*|--artifact-dir|--artifact-dir=*|--external-root|--external-root=*) printf 'Model/output/data paths are owned by this submitter; use SOTA_* environment overrides\n' >&2; exit 2 ;; esac
done
{
  printf '#!/bin/bash\nset -euo pipefail\n'
  printf 'export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"\n'
  # Sequential commands share ONE allocated GPU, never three concurrent jobs.
  for current_model in "${models[@]}"; do
    printf '%q ' srun "$python_bin" -u "$here/run_sota_comparison.py" --model "$current_model" \
      --mode profile --artifact-dir "$artifact_dir" --external-root "$external_root" \
      --output-root "$output_root" "$@"
    printf '\n'
  done
} | sbatch --account="${SOTA_ACCOUNT:-ugrad_advisor_x}" \
  --partition="${SOTA_PARTITION:-debug_ugrad_advisor_x}" \
  --nodes=1 --ntasks=1 --gres=gpu:normal:1 --cpus-per-task=4 --mem=16G \
  --time="${SOTA_TIME:-04:00:00}" --job-name="sota-$model" --chdir="$here" \
  --output="$log_dir/sota-${model}_%j.log" --error="$log_dir/sota-${model}_%j.log"
