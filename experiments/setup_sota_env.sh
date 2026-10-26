#!/usr/bin/env bash
# Separate environment; never install incompatible upstream requirements into base.
set -euo pipefail
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
root=$(cd -- "$here/../.." && pwd -P)
base_python=${SOTA_BASE_PYTHON:-$root/../envs/ecg-mask-pilot/bin/python}
env_dir=${SOTA_ENV_DIR:-$root/../envs/ecg-sota}
if [[ ! -x $base_python ]]; then printf 'Missing base Python: %s\n' "$base_python" >&2; exit 2; fi
if [[ -e $env_dir && ! -f $env_dir/.ecg-sota-managed ]]; then
  printf 'Existing unmanaged environment: %s; choose SOTA_ENV_DIR\n' "$env_dir" >&2
  exit 2
fi
if [[ ! -d $env_dir ]]; then
  "$base_python" -m venv "$env_dir"
  touch "$env_dir/.ecg-sota-managed"
fi
"$env_dir/bin/python" -m pip install --upgrade pip
"$env_dir/bin/python" -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
"$env_dir/bin/python" -m pip install -r "$here/requirements-sota.txt"
"$env_dir/bin/python" -m pip check
printf 'READY: %s/bin/python\n' "$env_dir"
