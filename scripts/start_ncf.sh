#!/usr/bin/env bash
# Creates a NEW study and submits CPU preparation followed by GPU profiling only.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export ECG_REPO="${ECG_REPO:-$(cd -- "$script_dir/.." && pwd)}"
export ECG_PYTHON="${ECG_PYTHON:-/data/gykwak03/envs/ecg-mask-pilot/bin/python}"
export ECG_ARTIFACT_DIR="${ECG_ARTIFACT_DIR:-/data/gykwak03/capstone2/ptbxl_250hz_bp067_50}"
export ECG_LOG_DIR="${ECG_LOG_DIR:-/data/gykwak03/capstone2/logs}"
export ECG_ACCOUNT="${ECG_ACCOUNT:-ugrad_advisor_x}"
export ECG_QOS="${ECG_QOS:-qos_gykwak03_2026_2}"
export ECG_PARTITION="${ECG_PARTITION:-batch_ugrad_advisor_x}"
export ECG_MAX_PARALLEL="${ECG_MAX_PARALLEL:-1}"
export ECG_JOB_HOURS="${ECG_JOB_HOURS:-24}"
export NCF_STUDY="${NCF_STUDY:-/data/gykwak03/capstone2/results/ncf40_$(date -u +%Y%m%dT%H%M%SZ)}"
cd "$ECG_REPO"
mkdir -p "$ECG_LOG_DIR"
test -x "$ECG_PYTHON"
test -f "$ECG_ARTIFACT_DIR/manifest.json"
"$ECG_PYTHON" experiments/run_ncf.py plan --config "${NCF_CONFIG:-configs/ncf_seraph.json}" \
  --artifact-dir "$ECG_ARTIFACT_DIR" --study-dir "$NCF_STUDY"
for name in ECG_REPO ECG_PYTHON ECG_ARTIFACT_DIR ECG_LOG_DIR ECG_ACCOUNT ECG_QOS \
            ECG_PARTITION ECG_MAX_PARALLEL ECG_JOB_HOURS NCF_STUDY; do
  printf 'export %s=%q\n' "$name" "${!name}"
done > "$NCF_STUDY/session.env"
NCF_PREP=$(bash scripts/submit_ncf.sh "$NCF_STUDY" prepare)
NCF_PREP=${NCF_PREP%%;*}
[[ "$NCF_PREP" =~ ^[0-9]+$ ]]
printf 'export NCF_PREP=%q\n' "$NCF_PREP" >> "$NCF_STUDY/session.env"
NCF_PROFILE=$(bash scripts/submit_ncf.sh "$NCF_STUDY" profile "$NCF_PREP")
NCF_PROFILE=${NCF_PROFILE%%;*}
[[ "$NCF_PROFILE" =~ ^[0-9]+$ ]]
printf 'export NCF_PROFILE=%q\n' "$NCF_PROFILE" >> "$NCF_STUDY/session.env"
printf '\nNCF_PREP=%s\nNCF_PROFILE=%s\n' "$NCF_PREP" "$NCF_PROFILE"
printf 'Restore session: source %q\n' "$NCF_STUDY/session.env"
printf 'After profile completes: cat %q/profile_budget.csv\n' "$NCF_STUDY"
squeue -j "$NCF_PREP,$NCF_PROFILE"
