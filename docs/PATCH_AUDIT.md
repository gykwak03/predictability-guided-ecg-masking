# Whole-wave group masking audit: 100 ms versus 40 ms

This analysis keeps the fixed token grid, but one accepted P/QRS/T interval is
one target group: all patches intersecting its half-open [onset, offset) are
masked together. There is no purity threshold or patch-center selection.
No model, optimizer or checkpoint is created. Only TRAIN waveforms/identities
are opened; labels, validation and test are not read.

## Run on fresh delineations

From the repository root, with NumPy, pandas, SciPy, NeuroKit2 0.3.0 and matplotlib:

```powershell
python experiments/run_patch_audit.py --artifact-dir preprocessing/artifacts/ptbxl_250hz_bp067_50 --output ../outputs/patch_group_audit_new --ecgs 128 --workers 4
```

## Reuse earlier detected boundaries

To isolate the masking change, reuse the old audit's annotations and the same
seed/128-patient selection. Supply the audit RUN DIRECTORY, not its delineations
subdirectory. Source-manifest hash, source path, train ECG/patient identities,
lead list and per-record identity are checked. Input annotation hashes are saved.
A new output directory is required; original purity results remain untouched.

Verified local Windows setup (base Python plus isolated NeuroKit dependency):

```powershell
$env:PYTHONPATH = (Resolve-Path ../tmp/patch-audit-deps).Path
& 'C:/Users/gykwa/anaconda3/python.exe' experiments/run_patch_audit.py --artifact-dir preprocessing/artifacts/ptbxl_250hz_bp067_50 --output ../outputs/patch_group_audit_new --ecgs 128 --delineation-cache ../outputs/patch_audit_128_20261112_final
```

The old workspace venv references an unavailable Store Python. The local
NeuroKit2 installation is only in ../tmp/patch-audit-deps; base packages were
not changed. Elsewhere install the dependencies into your own environment.

## Definitions

For patch length p samples and detected wave [a,b):

- First covered index = floor(a/p), exclusive last index = ceil(b/p).
- m = number of covered patches; cover = [first*p, last*p).
- Left extra = a - first*p; right extra = last*p - b.
- Total extra = m*p - (b-a), converted to milliseconds at 250 Hz.
- Extra time touching another accepted wave or rejected interval is reported
  separately. These margins are NOT used to reject targets.
- B = union of complete covers of the same phase in other valid beats on the
  same lead. Drop and count an entire B group if it shares target patches.
  Deduplicate patch indices; B patch count is not the B wave-group count.
- For Q patches per lead: W = Q-m-|B|; C = 11*m; U = 11*(Q-m).
  Target/W/B/C/U partition all 12*Q tokens for ONE target wave.
- These counts do not establish joint feasibility for multiple target waves.

Both sizes use identical accepted wave intervals, so their target-wave counts
must match. They differ in patch count, extra masked time and context token
counts. Compare masking margins, not purity-selected patch counts, against the
old audit. A waveform is a target even when no individual patch is pure.

## Detection limitations

Fresh annotation uses per-lead NeuroKit DWT on cleaned z-normalized ECG with
shared R peaks (II, fallback V2 then I). Existing missing/invalid/overlapping
boundaries remain rejected. Removing purity does NOT repair those boundaries.
Failed leads and zero-target ECGs remain in coverage denominators. Unassigned
samples may contain baseline OR missed waves. The annotations are not expert
truth and must be visually checked.

## Outputs

- REPORT.md: main results, rules and limitations.
- comparison.csv: six rows (two sizes x three phases), waveform-weighted
  mean/median/p95/max of cover size, margins and W/B/C/U counts, plus all-patient
  coverage and rejection rates. Candidate counts refer to whole wave groups.
- target_candidates.csv: one row per accepted wave and patch size, with beat,
  boundaries, target patch IDs, margin metrics and context counts.
- lead_phase_summary.csv: all ECG/lead/phase combinations, including zeros.
- patch_overlap.csv: descriptive overlaps only; no threshold filters.
- detection_status.csv, wave_intervals.csv, delineations/*.json: annotation audit.
- selection.csv, manifest.json: train sample identities and provenance.
- compute_cost.csv: analytical forward MACs, not measured GPU runtime or peak memory.
- Example PNGs: original detected waves colored; extra masked margins yellow;
  rejected intervals hatched, R peaks dotted; both patch grids shown.

Cost assumes encoder width/depth/heads 384/12/6, global decoder 256/4/4,
FF ratio 4, batch 1 and FP32. Input projection covers all tokens before gathering
visible tokens. No backward or optimizer is estimated. Dense attention storage
is a theoretical sum across layers, not peak GPU memory; fused kernels may
avoid materialization. Group masking preserves the theoretical 50% signal
budget for this cost comparison; no actual training mask is generated here.

Tests: python -m pytest tests/test_patch_audit.py -q
