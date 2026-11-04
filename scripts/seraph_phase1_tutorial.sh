#!/usr/bin/env bash
# Step-by-step Seraph helper. No training starts without an explicit stage command.
set -euo pipefail

help() {
  cat <<'EOF'
Usage: bash scripts/seraph_phase1_tutorial.sh [--dry-run] COMMAND [ARG]

  init [TAG]          Create/reuse a persistent session tag; no training
  env                 Print effective paths and tag
  plan                Print/save the experiment plan; no training
  check               Run imports + real-data preflight inside srun (foreground)
  smoke               Submit synthetic CPU smoke through Slurm
  profile             Submit the full-model P25, 3-update resource check
  pilot 25|50|125     Submit one 50%, 3000-update MAE pilot
  fit 25|50|125       Submit Ridge/CNN/Transformer fitting at the chosen patch
  score               Submit scoring for this session's fitted estimators
  sensitivity P R    Optional pilot at patch P, ratio 0.6 or 0.7
  resume STAGE        Resume profile, p25/p50/p125, fit, or pPr60/pPr70
  status              Show queue, run status and latest recorded job IDs
  logs STAGE          Tail stdout + stderr for the latest job of that stage
  analyze STAGE|all   Generate reports from saved results (also partial results)
  pack                Pack compact reports/CSVs/JSONs; exclude weights/arrays

STAGE: smoke, profile, p25, p50, p125, fit, score, p50r60, ...
Run stages one at a time and inspect status/profile before proceeding.
Ctrl+C while tailing logs does not cancel a job. No git pull/install is automatic.
--dry-run prints commands only: it never calls srun/sbatch or writes session files.
EOF
}
die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
DRY=0
if [[ "${1:-}" == --dry-run ]]; then DRY=1; shift; fi
action="${1:-help}"
(( $# == 0 )) || shift
case "$action" in help|-h|--help) help; exit 0;; esac

here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export ECG_REPO="${ECG_REPO:-$(cd -- "$here/.." && pwd)}"
export ECG_PYTHON="${ECG_PYTHON:-/data/gykwak03/envs/ecg-mask-pilot/bin/python}"
export ECG_RAW_DIR="${ECG_RAW_DIR:-/data/gykwak03/capstone2/ptbxl_raw/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3}"
export ECG_ARTIFACT_DIR="${ECG_ARTIFACT_DIR:-/data/gykwak03/capstone2/ptbxl_250hz_bp067_50}"
export ECG_OUTPUT_ROOT="${ECG_OUTPUT_ROOT:-$ECG_REPO/experiments/artifacts/phase1}"
export ECG_CACHE_DIR="${ECG_CACHE_DIR:-$ECG_OUTPUT_ROOT/cache}"
export ECG_LOG_DIR="${ECG_LOG_DIR:-/data/gykwak03/capstone2/logs}"
export ECG_PARTITION="${ECG_PARTITION:-batch_ugrad_advisor_x}"
config="${ECG_PHASE1_CONFIG:-$ECG_REPO/configs/phase1_seraph.json}"
runner="$ECG_REPO/experiments/run_phase1.py"
submit="$ECG_REPO/scripts/submit_phase1.sh"
session_root="$ECG_CACHE_DIR/tutorial"
current="$session_root/current-tag"
tag="${ECG_RUN_TAG:-}"
if [[ -z "$tag" && -f "$current" ]]; then IFS= read -r tag < "$current"; fi
print_cmd() { printf 'COMMAND:'; printf ' %q' "$@"; printf '\n'; }
invoke() { print_cmd "$@"; if (( ! DRY )); then "$@"; fi; }
valid_patch() { case "$1" in 25|50|125) ;; *) die 'Patch must be 25, 50 or 125';; esac; }
if [[ "$action" == init ]]; then
  (( $# <= 1 )) || die 'init accepts at most one TAG'
  tag="${1:-${tag:-$(date -u +%Y%m%dT%H%M%SZ)-$$}}"
fi
if [[ -z "$tag" ]]; then
  if (( DRY )); then tag=DRYRUN; else die 'Run init first (or set ECG_RUN_TAG to your existing tag)'; fi
fi
[[ "$tag" =~ ^[a-zA-Z0-9][a-zA-Z0-9_.-]*$ ]] || die 'Invalid TAG'
session="$session_root/$tag"
if [[ "$action" == init ]]; then
  if (( ! DRY )); then
    mkdir -p "$session" "$ECG_OUTPUT_ROOT" "$ECG_LOG_DIR"
    printf '%s\n' "$tag" > "$current"
  fi
  printf 'TAG=%s\nSession=%s\nReuse this tag after reconnecting; start with plan, then check.\n' "$tag" "$session"
  exit 0
fi
if [[ "$action" == env ]]; then
  printf '%s\n' "TAG=$tag" "ECG_REPO=$ECG_REPO" "ECG_PYTHON=$ECG_PYTHON" \
    "ECG_ARTIFACT_DIR=$ECG_ARTIFACT_DIR" "ECG_OUTPUT_ROOT=$ECG_OUTPUT_ROOT" \
    "ECG_CACHE_DIR=$ECG_CACHE_DIR" "ECG_LOG_DIR=$ECG_LOG_DIR" "CONFIG=$config"
  exit 0
fi
if (( ! DRY )); then
  [[ -x "$ECG_PYTHON" ]] || die "Missing Python: $ECG_PYTHON"
  [[ -f "$runner" && -f "$submit" && -f "$config" ]] || die 'Missing Phase 1 code/config; check your Git checkout'
  [[ -d "$session" ]] || die 'Run init for this tag first'
fi

fit_patch() {
  if [[ -f "$session/fit-patch" ]]; then
    IFS= read -r patch < "$session/fit-patch"
  elif (( DRY )); then patch=50
  else die 'Run fit PATCH first; no patch has been selected'; fi
  valid_patch "$patch"
}
stage_info() {
  stage="$1"; args=(--config "$config")
  case "$stage" in
    smoke) command=smoke; run="$ECG_OUTPUT_ROOT/smoke_$tag"; id="$tag"
      args=(--config "$ECG_REPO/configs/phase1_smoke.json");;
    profile) command=profile; run="$ECG_OUTPUT_ROOT/profile_$tag-profile"; id="$tag-profile"
      args+=(--patch-samples 25);;
    p25|p50|p125) patch="${stage#p}"; command=pilot; id="$tag-$stage"
      run="$ECG_OUTPUT_ROOT/pilot_$id"; args+=(--patches "$patch");;
    fit|score) fit_patch; id="$tag-fit"; run="$ECG_OUTPUT_ROOT/estimators_$id"
      command=estimator-fit; [[ "$stage" != score ]] || command=estimator-score
      args+=(--patch-samples "$patch");;
    *) if [[ "$stage" =~ ^p(25|50|125)r(60|70)$ ]]; then
        patch="${BASH_REMATCH[1]}"; ratio="0.${BASH_REMATCH[2]:0:1}"
        command=pilot; id="$tag-p$patch-r${BASH_REMATCH[2]}"; run="$ECG_OUTPUT_ROOT/pilot_$id"
        args+=(--patches "$patch" --patch-samples "$patch" --mask-ratio "$ratio")
      else die "Unknown stage: $stage"; fi;;
  esac
}
read_status() {
  "$ECG_PYTHON" -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1])/"status.json"; print(json.loads(p.read_text())["status"] if p.exists() else "not_started")' "$1"
}
require_status() {
  if (( DRY )); then return; fi
  local actual
  actual="$(read_status "$1")"
  [[ "$actual" == "$2" ]] || die "Expected $2, got $actual: $1/status.json"
}
no_live_session_job() {
  if (( DRY )); then return; fi
  local file job active
  active="$(squeue -h -u "${USER:-gykwak03}" -o '%A')" || die 'Cannot verify queue state; inspect squeue first'
  for file in "$session"/*.job; do
    [[ -f "$file" ]] || continue
    IFS= read -r job < "$file"
    [[ $'\n'"$active"$'\n' != *$'\n'"$job"$'\n'* ]] || die "Job $job is still queued/running. Finish/check it before submitting another stage."
  done
}
submit_stage() {
  local resume="${1:-0}" response job hours
  no_live_session_job
  if [[ "$stage" == score ]]; then
    if (( ! DRY )); then
      state="$(read_status "$run")"
      [[ "$state" != completed ]] || die "Already scored: $run"
      for name in ridge cnn transformer; do
        [[ -f "$run/$name/fit_complete.json" ]] || die "Fit is not complete for $name: $run"
      done
      [[ "$state" == fit_completed || "$state" == partial ]] || die "Cannot score status=$state: $run"
    fi
    args+=(--run-dir "$run")
  elif (( resume )); then
    if (( ! DRY )); then
      [[ -f "$run/status.json" ]] || die "No saved run: $run"
      [[ "$(read_status "$run")" != completed ]] || die "Already completed: $run"
    fi
    args+=(--resume "$run")
  else
    if (( ! DRY )); then [[ ! -e "$run" ]] || die "Run exists. Use resume $stage, or a new TAG: $run"; fi
    args+=(--run-id "$id")
  fi
  hours="${ECG_JOB_HOURS:-24}"
  if [[ -z "${ECG_JOB_HOURS:-}" && ( "$stage" == smoke || "$stage" == profile ) ]]; then hours=1; fi
  print_cmd env "ECG_JOB_HOURS=$hours" bash "$submit" "$command" "${args[@]}"
  if (( DRY )); then return; fi
  response="$(ECG_JOB_HOURS="$hours" bash "$submit" "$command" "${args[@]}")"
  printf '%s\n' "$response"
  job="$(printf '%s\n' "$response" | sed -nE 's/^([0-9]+)(;[^[:space:]]+)?$/\1/p' | tail -n 1)"
  [[ "$job" =~ ^[0-9]+$ ]] || die 'Submission returned no parsable job ID; inspect squeue before retrying'
  printf '%s\n' "$job" > "$session/$stage.job"
  printf '%s\t%s\t%s\n' "$stage" "$job" "$run" >> "$session/jobs.tsv"
  printf 'Submitted %s: job=%s\nRun=%s\nNext: status, or logs %s\n' "$stage" "$job" "$run" "$stage"
}

case "$action" in
  plan) invoke "$ECG_PYTHON" "$runner" plan --config "$config";;
  check)
    no_live_session_job
    invoke "$ECG_PYTHON" -c 'import sys,torch,numpy,pandas,scipy,matplotlib,wfdb; print(sys.executable); print("Torch/CUDA",torch.__version__,torch.version.cuda); print("NumPy/pandas/WFDB",numpy.__version__,pandas.__version__,wfdb.__version__)'
    invoke srun --account="${ECG_ACCOUNT:-ugrad_advisor_x}" --qos="${ECG_QOS:-ugrad}" \
      --partition="$ECG_PARTITION" --gres=gpu:normal:1 --ntasks=1 --cpus-per-task="${ECG_CPUS:-4}" \
      --mem="${ECG_MEMORY:-32G}" --time=00:30:00 \
      "$ECG_PYTHON" "$runner" preflight --config "$config"
    if (( ! DRY )); then printf 'passed\n' > "$session/preflight.ready"; fi;;
  smoke) stage_info smoke; submit_stage;;
  profile|pilot|fit|sensitivity)
    if (( ! DRY )); then [[ -f "$session/preflight.ready" ]] || die 'Run check successfully first'; fi
    require_status "$ECG_OUTPUT_ROOT/smoke_$tag" completed
    if [[ "$action" != profile ]]; then require_status "$ECG_OUTPUT_ROOT/profile_$tag-profile" completed; fi
    case "$action" in
      profile) stage_info profile;;
      pilot) (( $# == 1 )) || die 'Usage: pilot PATCH'; valid_patch "$1"; stage_info "p$1";;
      fit)
        (( $# == 1 )) || die 'Usage: fit PATCH (choose after reviewing the pilot reports)'; valid_patch "$1"
        for p in 25 50 125; do require_status "$ECG_OUTPUT_ROOT/pilot_$tag-p$p" completed; done
        if (( ! DRY )); then
          if [[ -f "$session/fit-patch" ]]; then
            IFS= read -r selected < "$session/fit-patch"
            [[ "$selected" == "$1" ]] || die 'This session already chose another patch; use a new TAG'
          fi
          printf '%s\n' "$1" > "$session/fit-patch"
        fi
        # Dry runs must also reflect the explicitly selected patch.
        stage_info fit; args=(--config "$config" --patch-samples "$1");;
      sensitivity)
        (( $# == 2 )) || die 'Usage: sensitivity PATCH 0.6|0.7'; valid_patch "$1"
        case "$2" in 0.6) suffix=60;; 0.7) suffix=70;; *) die 'Ratio must be 0.6 or 0.7';; esac
        require_status "$ECG_OUTPUT_ROOT/pilot_$tag-p$1" completed
        stage_info "p$1"r"$suffix";;
    esac
    submit_stage;;
  score) stage_info score; submit_stage;;
  resume)
    (( $# == 1 )) || die 'Usage: resume STAGE'
    case "$1" in smoke|score) die 'Smoke has no resume; scoring uses score after a complete fit';; esac
    stage_info "$1"; submit_stage 1;;
  status)
    invoke squeue -u "${USER:-gykwak03}"
    if [[ -f "$session/jobs.tsv" ]]; then cat "$session/jobs.tsv"; fi
    for stage in smoke profile p25 p50 p125 fit score p25r60 p25r70 p50r60 p50r70 p125r60 p125r70; do
      if [[ ( "$stage" == fit || "$stage" == score ) && ! -f "$session/fit-patch" ]]; then continue; fi
      stage_info "$stage"
      if [[ -f "$run/status.json" ]]; then printf '\n%s: %s\n' "$stage" "$run"; cat "$run/status.json"; fi
    done;;
  logs)
    (( $# == 1 )) || die 'Usage: logs STAGE'; stage_info "$1"
    [[ -f "$session/$stage.job" ]] || die "No recorded job for $stage"
    IFS= read -r job < "$session/$stage.job"
    printf 'Ctrl+C stops tail only; it does not cancel the job.\n'
    invoke tail -n 60 -F "$ECG_LOG_DIR/phase1-ecg-phase1-$command-$job.out" \
      "$ECG_LOG_DIR/phase1-ecg-phase1-$command-$job.err";;
  analyze)
    (( $# == 1 )) || die 'Usage: analyze STAGE|all'
    stages=("$1"); [[ "$1" != all ]] || stages=(profile p25 p50 p125 fit)
    for name in "${stages[@]}"; do
      if [[ "$name" == fit && ! -f "$session/fit-patch" && "$1" == all ]]; then continue; fi
      stage_info "$name"
      if (( ! DRY )) && [[ ! -f "$run/status.json" ]]; then printf 'Skip missing run: %s\n' "$run"; continue; fi
      invoke "$ECG_PYTHON" "$runner" analyze --run-dir "$run"
    done;;
  pack)
    names=()
    for name in smoke profile p25 p50 p125 fit; do
      if [[ "$name" == fit && ! -f "$session/fit-patch" ]]; then continue; fi
      stage_info "$name"; if [[ -d "$run" ]]; then names+=("${run##*/}"); fi
    done
    (( ${#names[@]} > 0 )) || die 'No results to pack'
    archive="$ECG_OUTPUT_ROOT/$tag-reports-$(date -u +%Y%m%dT%H%M%SZ).tar.gz"
    invoke tar -czf "$archive" --exclude='*.pt' --exclude='*.npy' --exclude='*.npz' \
      -C "$ECG_OUTPUT_ROOT" "${names[@]}"
    printf 'REPORT_BUNDLE=%s\nNPZ maps/weights are excluded; transfer those separately if needed.\n' "$archive";;
  *) help; die "Unknown command: $action";;
esac
