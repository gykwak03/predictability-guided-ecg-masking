#!/usr/bin/env bash
# Bootstrap reviewed sources/environment, then submit ONE sequential profile job.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
root=$(cd -- "$here/../.." && pwd -P)
base_python=${SOTA_BASE_PYTHON:-$root/../envs/ecg-mask-pilot/bin/python}
env_dir=${SOTA_ENV_DIR:-$root/../envs/ecg-sota}
artifact_dir=${SOTA_ARTIFACT_DIR:-$root/ptbxl_250hz_bp067_50}
external_root=${SOTA_EXTERNAL_ROOT:-$root/external}
if [[ ! -f $artifact_dir/manifest.json ]]; then
  printf 'Upload the validated 50 Hz dataset first: %s\n' "$artifact_dir" >&2
  exit 2
fi
"$base_python" "$here/setup_sota_repos.py" --external-root "$external_root"
bash "$here/setup_sota_env.sh"
"$env_dir/bin/python" -c 'import json,sys; from pathlib import Path; import numpy as np; p=Path(sys.argv[1]); m=json.loads((p/"manifest.json").read_text()); c=m["config"]; assert (c["target_sampling_rate"],c["bandpass_low"],c["bandpass_high"],c["normalization"])==(250,.67,50.,"per-record"); [(np.load(p/f"signals_{s}.npy",mmap_mode="r").shape==tuple(m["splits"][s]["signals"]["shape"]) or sys.exit("Shape mismatch")) for s in ("train","validation")]; print("50 Hz artifact header validation OK")' "$artifact_dir"
export SOTA_PYTHON="$env_dir/bin/python"
export SOTA_ARTIFACT_DIR="$artifact_dir"
export SOTA_EXTERNAL_ROOT="$external_root"
bash "$here/submit_sota.sh" all --mode profile --steps 20 --microbatch 4 --accum-steps 8
