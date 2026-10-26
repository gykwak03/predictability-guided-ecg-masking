"""Predeclared comparisons: one-factor edges, factorial interactions, and anchors."""
from __future__ import annotations

from itertools import product
from pathlib import Path

from mae_cause_data import digest, file_sha, source_selection

FACTORS = ["patch_norm", "fixed_time", "leadwise_decoder"]
INTERVENTIONS = FACTORS + ["sep", "train_policy"]


def make_plan(artifact_dir, *, stages=("controls", "architecture", "normalization"),
              size="stmem-small", ratio=.5, seeds=(42, 43, 44), train_ecgs=5000,
              validation_ecgs=0, subset_seed=42, eval_seed=42, short=1000, long=3000,
              microbatch=4, accumulation=8, diagnostic_ecgs=32, final_ecgs=128, smoke=False):
    if not 0 < ratio < 1 or not 0 < short < long or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Invalid ratio, budgets or seeds")
    if min(microbatch, accumulation, train_ecgs) < 1 or microbatch * accumulation > train_ecgs:
        raise ValueError("Effective batch must fit the train subset")
    if validation_ecgs < 0 or min([subset_seed, eval_seed, *seeds]) < 0:
        raise ValueError("Validation limit and seeds must be nonnegative")
    if diagnostic_ecgs < 1 or final_ecgs < 0:
        raise ValueError("Diagnostic count must be positive; final 0 means all validation ECGs")
    if abs(round(600 * ratio) / 600 - ratio) > 1e-12:
        raise ValueError("Mask ratio must correspond to an exact P50 token count")
    if not set(stages) <= {"controls", "architecture", "normalization"}:
        raise ValueError("Unknown stage")
    # Every model can be evaluated on both banks at an identical exact ratio.
    exact_per_lead = round(600 * ratio) % 12 == 0
    if not exact_per_lead and "architecture" in stages:
        raise ValueError("Architecture stage needs an exact per-lead ratio (e.g. .50 or .76). For historical .75 use controls/normalization only")
    if smoke:
        source_hash, ids, config = "synthetic-smoke", {"train": list(range(train_ecgs)), "validation": list(range(100000, 100000 + (validation_ecgs or 16)))}, {}
        size = "smoke"
    else:
        manifest, frames = source_selection(artifact_dir, train_ecgs, validation_ecgs, subset_seed)
        source_hash = file_sha(Path(artifact_dir) / "manifest.json")
        ids = {s: f.ecg_id.astype(int).tolist() for s, f in frames.items()}
        config = manifest["config"]
    base = dict(backend="plain", size=size, initialization="independent", steps=long,
                schedule_horizon=long, patch_norm=False, fixed_time=False,
                leadwise_decoder=False, sep=False, train_policy="global", normalization="per_record")
    conditions, contrasts = {}, []

    def add(name, **changes):
        conditions[name] = dict(base, **changes)
        return name

    def edge(a, b, factor, kind="one_factor"):
        contrasts.append(dict(reference=a, treatment=b, factor=factor, kind=kind))

    add("independent_long")
    if "controls" in stages:
        add("legacy_short", initialization="legacy", steps=short, schedule_horizon=short)
        add("independent_short", steps=short, schedule_horizon=short)
        add("independent_extend", schedule_horizon=short)
        edge("legacy_short", "independent_short", "initialization")
        edge("independent_short", "independent_extend", "steps", "prefix_extension")
        edge("independent_extend", "independent_long", "schedule_horizon")
        if size != "baseline" and not smoke:
            add("capacity_small_legacy", size="baseline", initialization="legacy", steps=short, schedule_horizon=short)
            edge("capacity_small_legacy", "legacy_short", "size")
    if "architecture" in stages:
        for bits in product([False, True], repeat=3):
            name = "independent_long" if not any(bits) else "factor_" + "".join(str(int(b)) for b in bits)
            add(name, **dict(zip(FACTORS, bits)))
        # All edges of the 2^3 design, not just single changes against the baseline.
        for bits in product([0, 1], repeat=3):
            for i, factor in enumerate(FACTORS):
                if bits[i]:
                    continue
                changed = list(bits)
                changed[i] = 1
                a = "independent_long" if not any(bits) else "factor_" + "".join(map(str, bits))
                b = "factor_" + "".join(map(str, changed))
                edge(a, b, factor)
        add("sep_only", sep=True)
        add("per_lead_only", train_policy="per_lead")
        edge("independent_long", "sep_only", "sep")
        edge("independent_long", "per_lead_only", "train_policy")
        all_on = dict(patch_norm=True, fixed_time=True, leadwise_decoder=True, sep=True, train_policy="per_lead")
        add("all_on", **all_on)
        for factor in INTERVENTIONS:
            off = dict(all_on)
            off[factor] = "global" if factor == "train_policy" else False
            name = "all_without_" + factor
            add(name, **off)
            edge(name, "all_on", factor, "reversal")
        # A bridge, never an architecture-only causal contrast.
        add("official_stmem", **dict(all_on, backend="stmem", size="small", initialization="official"))
        edge("all_on", "official_stmem", "official_implementation_bundle", "anchor")
    if "normalization" in stages:
        for mode in ("train_lead", "train_common", "visible_record"):
            name = "normalization_" + mode
            add(name, normalization=mode)
            edge("independent_long", name, "normalization")
    plan = dict(schema=1, smoke=smoke, artifact_dir=str(Path(artifact_dir).resolve()),
                source_manifest_sha256=source_hash, source_config=config, ecg_ids=ids,
                train_ecgs=train_ecgs, validation_ecgs=validation_ecgs, subset_seed=subset_seed,
                eval_seed=eval_seed, seeds=list(seeds), patch_samples=50, mask_ratio=ratio,
                masked_tokens=round(600 * ratio), visible_tokens=600-round(600 * ratio),
                microbatch=microbatch, accumulation=accumulation, effective_batch=microbatch * accumulation,
                optimizer=dict(lr=3e-4, betas=[.9, .95], weight_decay=.05,
                               warmup=min(100, max(1, short // 10)), grad_clip=1.),
                precision="fp32", evaluation_banks=["global", "per_lead"] if exact_per_lead else ["global"],
                diagnostic_ecgs=diagnostic_ecgs, final_ecgs=final_ecgs, repeats=3, eval_every=200,
                probe_checkpoints=[300, 1500], conditions=conditions, contrasts=contrasts,
                inference=dict(primary="fixed-train-std normalized physical mV squared error on hidden samples",
                               minimum_seeds=3, bootstrap=2000, minimum_relative_mse_reduction=.10,
                               practical_floor=.02, seed_consistency=1., patient_resampling="paired within-seed patient clusters; seed also resampled",
                               caveat="Exploratory, correlated/multiple comparisons; independent confirmation required. No unique universal cause claim."),
                test_opened=False)
    plan["plan_id"] = digest(plan)
    validate_plan(plan)
    return plan


def validate_plan(plan):
    value = dict(plan)
    expected = value.pop("plan_id")
    if digest(value) != expected:
        raise ValueError("Plan modified; create a new plan instead of changing recorded settings")
    for contrast in plan["contrasts"]:
        a, b = (plan["conditions"][contrast[k]] for k in ("reference", "treatment"))
        changed = {k for k in a if a[k] != b[k]}
        if contrast["kind"] != "anchor" and changed != {contrast["factor"]}:
            raise ValueError(f"Confounded contrast: {contrast}, changed={changed}")
    return plan
