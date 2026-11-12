# Saved NCF checkpoint diagnostics

This audit uses the original validation mask banks and completed checkpoints. It
does not train the MAE, update the policy, change the frozen experiment sources,
modify existing results, or open the test split. Diagnosis labels are used only
by the optional initialized-encoder frozen linear probe, with the same protocol
as the completed runs.

`scripts/diagnose_ncf_reconstruction.py` reports:

- Exact zero-prediction MSE on the same hidden samples as each model.
- Trained and initialized-model MSE, pooled hidden prediction/target standard
  deviations, amplitude ratio, Pearson correlation and improvement versus zero.
- Equal-stratum MSE, original saved validation MSE and recomputation difference.
- First two distinct ECG examples from each validation bank, with original,
  visible input, hidden predictions, zero baseline and target shading. All plots
  use common amplitude scales and break prediction lines across visible samples.
- Recorded training-pipeline time, final-evaluation time and their sum.
- Optionally the initialized encoder's validation macro-AUROC after training
  only a linear head. No performance claim is made from this audit alone.

An improvement-versus-zero value of zero means equal MSE to predicting zero;
positive is better. A small prediction/target standard-deviation ratio suggests
small output amplitude but does not by itself prove an implementation bug.
The baseline and correlation are measured on hidden waveform samples only.
Per-case and per-condition CSVs permit inspecting heterogeneity rather than
relying only on the pooled correlation.

## Seraph

Fetch only the two standalone scripts into the existing checkout so an original
plan remains valid. Use the commit containing these scripts as `DIAG_REF` when
pinning the download; the original model and experiment modules must remain at
the original study version.

```bash
source "$(cat /data/gykwak03/capstone2/ncf_pilot_latest_session.txt)"
cd "$ECG_REPO"
git fetch origin main
git checkout FETCH_HEAD -- scripts/diagnose_ncf_reconstruction.py scripts/submit_ncf_diagnostics.sh

# CPU-only, immediate reading of completed.json timer fields:
"$ECG_PYTHON" scripts/diagnose_ncf_reconstruction.py \
  --study-dir "$NCF_STUDY" --timing-only

# One GPU; initialized encoder linear probe is included.
export NCF_DIAG_DIR="$NCF_STUDY/diagnostics/reconstruction_$(date -u +%Y%m%dT%H%M%SZ)"
NCF_DIAG=$(bash scripts/submit_ncf_diagnostics.sh)
NCF_DIAG=${NCF_DIAG%%;*}
[[ "$NCF_DIAG" =~ ^[0-9]+$ ]]
printf 'export NCF_DIAG=%q\nexport NCF_DIAG_DIR=%q\n' \
  "$NCF_DIAG" "$NCF_DIAG_DIR" >> "$NCF_STUDY/session.env"
squeue -j "$NCF_DIAG"
```

The reconstruction comparisons involve around 2,560 forward cases for the
four-method pilot (initialized plus four trained models, two 256-case banks).
The initialized-encoder linear probe also requires a full train/validation
feature extraction. The supplied A2000 profile suggests approximately 3 minutes
of reconstruction forwards plus 30 minutes of feature extraction, before plots,
initialization, CPU linear-head fitting and I/O. The job requests 90 minutes.
These estimates exclude queueing and are not guarantees.

```bash
source "$(cat /data/gykwak03/capstone2/ncf_pilot_latest_session.txt)"
tail -n 20 "$ECG_LOG_DIR/ncf-diag-$NCF_DIAG.out"
tail -n 20 "$ECG_LOG_DIR/ncf-diag-$NCF_DIAG.err"
# Once the diagnostic finishes:
cat "$NCF_DIAG_DIR/REPORT.md"
cat "$NCF_DIAG_DIR/runtime.csv"
cat "$NCF_DIAG_DIR/untrained_linear_probe/linear_probe.json"
```

## Slurm elapsed versus internal timers

`runtime.csv` sums `training_pipeline_seconds` and
`final_evaluation_seconds`. The former excludes checkpoint/JSON writes; the sum
also excludes process startup and setup. It is useful measured pipeline time,
not exact job elapsed time. Because the four runs execute concurrently, summing
their durations measures GPU task hours, not the duration of the study.

When the accounting database is reachable:

```bash
sacct -X -j "$NCF_RUN" --format=JobID%20,State,Start,End,Elapsed,ExitCode
```

If accounting still times out, inspect recent task records in the controller:

```bash
for task in 0 1 2 3; do
  scontrol show job "${NCF_RUN}_${task}"
done
```

`RunTime`, `StartTime` and `EndTime` give actual allocated elapsed time for each
task. Controller records can expire. Overall study execution span is the last
task end minus the earliest task start; submission-to-completion additionally
includes queueing. Slurm elapsed does not require rerunning training.
