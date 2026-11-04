#!/usr/bin/env bash
# No cluster access: every scheduler/submission/Python call is a local stub.
set -euo pipefail
repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$repo/.pytest_cache"
scratch="$(mktemp -d "$repo/.pytest_cache/seraph-tutorial.XXXXXX")"
helper="$repo/scripts/seraph_phase1_tutorial.sh"
export ECG_REPO="$scratch/fake repo" ECG_OUTPUT_ROOT="$scratch/results" ECG_CACHE_DIR="$scratch/cache"
export ECG_LOG_DIR="$scratch/logs" ECG_PYTHON="$scratch/bin/python" MOCK_TRACE="$scratch/calls"
export PATH="$scratch/bin:$PATH"
unset ECG_RUN_TAG ECG_PHASE1_CONFIG ECG_JOB_HOURS
mkdir -p "$scratch/bin" "$ECG_REPO/scripts" "$ECG_REPO/configs" "$ECG_REPO/experiments"
touch "$ECG_REPO/configs/phase1_seraph.json" "$ECG_REPO/experiments/run_phase1.py"
cat > "$ECG_PYTHON" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == -c && "$2" == *'import json'* ]]; then
  file="${@: -1}/status.json"
  if [[ -f "$file" ]]; then sed -nE 's/.*"status"[[:space:]]*:[[:space:]]*"([^"]+)".*/\1/p' "$file"
  else printf 'not_started\n'; fi
else printf 'python'; printf ' <%s>' "$@"; printf '\n'; fi
EOF
cat > "$scratch/bin/squeue" <<'EOF'
#!/usr/bin/env bash
printf '%s' "${MOCK_ACTIVE:-}"
EOF
cat > "$scratch/bin/srun" <<'EOF'
#!/usr/bin/env bash
printf 'srun'; printf ' <%s>' "$@"; printf '\n'
EOF
cat > "$scratch/bin/tail" <<'EOF'
#!/usr/bin/env bash
if [[ "${1:-}" == -n && "${2:-}" == 1 ]]; then /usr/bin/tail "$@"
else printf 'tail'; printf ' <%s>' "$@"; printf '\n'; fi
EOF
cat > "$ECG_REPO/scripts/submit_phase1.sh" <<'EOF'
#!/usr/bin/env bash
printf 'submit' >> "$MOCK_TRACE"; printf ' <%s>' "$@" >> "$MOCK_TRACE"; printf '\n' >> "$MOCK_TRACE"
printf 'MOANA: Job submitted\n123456;moana\n'
EOF
chmod +x "$scratch/bin/"*
run() { bash "$helper" "$@" > "$scratch/last-output" 2>&1; }
reject() { if run "$@"; then printf 'Expected rejection: %s\n' "$*"; exit 1; fi; }
status() { mkdir -p "$1"; printf '{"status":"%s"}\n' "$2" > "$1/status.json"; }
count() { if [[ -f "$MOCK_TRACE" ]]; then wc -l < "$MOCK_TRACE"; else printf 0; fi; }

run help
[[ ! -e "$ECG_CACHE_DIR" ]]
run --dry-run profile
run --dry-run fit 125
grep -q -- '--patch-samples 125' "$scratch/last-output"
run --dry-run sensitivity 50 0.7
[[ ! -e "$ECG_CACHE_DIR" && ! -e "$MOCK_TRACE" ]]
run init qa-session
[[ "$(cat "$ECG_CACHE_DIR/tutorial/current-tag")" == qa-session ]]
run env
grep -q 'TAG=qa-session' "$scratch/last-output"
reject pilot 25
run check
[[ -f "$ECG_CACHE_DIR/tutorial/qa-session/preflight.ready" ]]
reject profile
[[ "$(count)" == 0 ]]
run smoke
[[ "$(cat "$ECG_CACHE_DIR/tutorial/qa-session/smoke.job")" == 123456 ]]
status "$ECG_OUTPUT_ROOT/smoke_qa-session" completed
export MOCK_ACTIVE=123456
reject profile
[[ "$(count)" == 1 ]]
unset MOCK_ACTIVE
run profile
status "$ECG_OUTPUT_ROOT/profile_qa-session-profile" partial
reject pilot 25
run resume profile
/usr/bin/tail -n 1 "$MOCK_TRACE" | grep -q -- '<--resume>'
status "$ECG_OUTPUT_ROOT/profile_qa-session-profile" completed
run pilot 25
grep -q '<--patches> <25>' "$MOCK_TRACE"
for p in 25 50 125; do status "$ECG_OUTPUT_ROOT/pilot_qa-session-p$p" completed; done
run sensitivity 125 0.6
/usr/bin/tail -n 1 "$MOCK_TRACE" | grep -q '<--patches> <125> <--patch-samples> <125> <--mask-ratio> <0.6> <--run-id> <qa-session-p125-r60>'
run fit 125
[[ "$(cat "$ECG_CACHE_DIR/tutorial/qa-session/fit-patch")" == 125 ]]
reject fit 50
fit="$ECG_OUTPUT_ROOT/estimators_qa-session-fit"
status "$fit" partial
run resume fit
/usr/bin/tail -n 1 "$MOCK_TRACE" | grep -q -- '<--patch-samples> <125> <--resume>'
reject score
for model in ridge cnn transformer; do mkdir -p "$fit/$model"; touch "$fit/$model/fit_complete.json"; done
status "$fit" fit_completed
run score
/usr/bin/tail -n 1 "$MOCK_TRACE" | grep -q '<estimator-score>'
status "$fit" partial
run score
status "$fit" completed
reject score
reject resume fit
reject resume smoke
reject sensitivity 50 0.65
run logs fit
grep -q 'phase1-ecg-phase1-estimator-fit-123456.err' "$scratch/last-output"
run analyze p25
grep -q '<analyze>' "$scratch/last-output"
printf 'local report\n' > "$fit/REPORT.html"
printf 'weights must be excluded\n' > "$fit/model.pt"
printf 'arrays must be excluded\n' > "$fit/ecg_maps.npz"
run pack
bundle="$(sed -n 's/^REPORT_BUNDLE=//p' "$scratch/last-output")"
[[ -f "$bundle" ]]
tar -tzf "$bundle" > "$scratch/bundle-list"
grep -q 'REPORT.html' "$scratch/bundle-list"
if grep -Eq '\.(pt|npz)$' "$scratch/bundle-list"; then printf 'Large files leaked into report bundle\n'; exit 1; fi
run status
printf 'PASS: dry-run, persistent tag, prerequisites, active-job guard, job ID/log paths, chosen patch, sensitivity, resume, score retry, completed-run protection, compact bundle exclusions.\n'
printf 'All calls used local mocks. Scratch: %s\n' "$scratch"
