#!/usr/bin/env bash
# Reuse the working server allocation settings; save results inside this clone.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  printf 'Usage: bash experiments/submit_pilot.sh {mae|stmem|predictability} [runner options]\n' >&2
  exit 2
fi
backend=$1
shift
experiment_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_dir=$(cd -- "$experiment_dir/.." && pwd -P)
server_root=$(cd -- "$repo_dir/.." && pwd -P)
model_size=baseline
previous=''
for argument in "$@"; do
  if [[ $argument == --output-root || $argument == --output-root=* ]]; then
    printf 'This submitter saves inside experiments/artifacts; omit --output-root.\n' >&2
    exit 2
  fi
  if [[ $previous == --model-size ]]; then model_size=$argument; fi
  if [[ $argument == --model-size=* ]]; then model_size=${argument#*=}; fi
  previous=$argument
done
case "$backend" in
  mae)
    runner=$experiment_dir/run_masking_pilot.py
    family=masking_pilot
    if [[ $model_size == stmem-small ]]; then family=mae_stmem_size_pilot; fi
    ;;
  stmem)
    runner=$experiment_dir/run_st_mem_pilot.py
    family=st_mem_pilot
    ;;
  predictability)
    runner=$experiment_dir/run_predictability.py
    family=predictability
    ;;
  *) printf 'Choose backend mae, stmem or predictability.\n' >&2; exit 2 ;;
esac
template=${PILOT_SBATCH_TEMPLATE:-$server_root/run_stmem.sbatch}
python_bin=${PILOT_PYTHON:-/data/gykwak03/envs/ecg-mask-pilot/bin/python}
if [[ ! -r $template ]]; then
  printf 'Cannot read allocation template: %s\nSet PILOT_SBATCH_TEMPLATE to your working sbatch file.\n' "$template" >&2
  exit 2
fi
if ! grep -Eq '^#SBATCH[[:space:]].*(--gres|--gpus)' "$template"; then
  printf 'Allocation template must explicitly request a GPU: %s\n' "$template" >&2
  exit 2
fi
output_root=$experiment_dir/artifacts/$family
log_dir=$server_root/logs
mkdir -p "$log_dir"
printf 'Results: %s\n' "$output_root"
{
  printf '#!/bin/bash\n'
  sed -n '/^#SBATCH/p' "$template"
  printf 'set -euo pipefail\n'
  printf '%q ' srun "$python_bin" -u "$runner" --mode pilot --train \
    --artifact-dir "$server_root/ptbxl_250hz" "$@" --output-root "$output_root"
  printf '\n'
} | sbatch --job-name="ecg-$backend" --chdir="$experiment_dir" \
  --output="$log_dir/${backend}_%j.log" --error="$log_dir/${backend}_%j.log"
