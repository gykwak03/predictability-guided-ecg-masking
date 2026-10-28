"""Paired, patient-clustered evidence, not an automatic universal cause claim."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mae_cause_data import digest, write_json
from mae_cause_plan import FACTORS, validate_plan


def crossed_bootstrap(matrix, repeats=2000, seed=7301):
    """Same patients recur across seeds: resample both crossed axes, not ECG rows.

    Matrix shape is seed x patient, positive = error reduction. Identical patient
    draws are shared across resampled seeds. Patients receive equal weight.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or not matrix.size or not np.isfinite(matrix).all():
        raise ValueError("Need a finite seed x patient effect matrix")
    rng = np.random.default_rng(seed)
    values = np.empty(repeats)
    s, p = matrix.shape
    for i in range(repeats):
        si = rng.integers(s, size=s)
        pi = rng.integers(p, size=p)
        values[i] = matrix[np.ix_(si, pi)].mean()
    return dict(mean=float(matrix.mean()), ci_low=float(np.quantile(values, .025)),
                ci_high=float(np.quantile(values, .975)), seeds=s, patients=p,
                positive_seed_fraction=float((matrix.mean(1) > 0).mean()))


def load_run(root, name, seed):
    folder = root / "runs" / name / f"seed{seed}"
    if not (folder / "summary.json").exists():
        return None
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    config = json.loads((folder / "run_config.json").read_text(encoding="utf-8"))
    if not summary.get("completed") or summary["fingerprint"] != config["fingerprint"]:
        raise ValueError(f"Unconfirmed or mismatched run: {folder}")
    if summary.get("seed") != seed or config.get("seed") != seed or summary["spec"].get("model_seed") != seed:
        raise ValueError("Seed identity mismatch; copied runs are not independent repeats")
    if summary["kind"] != "scientific":
        return None
    frame = pd.read_csv(folder / "final_per_ecg.csv")
    if frame.duplicated(["bank", "ecg_id", "repeat"]).any():
        raise ValueError("Duplicate reconstruction observations")
    return dict(summary=summary, config=config, frame=frame, folder=folder)


def assert_pair(a, b, plan, contrast, bank):
    for key in ("plan_id", "cache_id"):
        if a["summary"][key] != b["summary"][key]:
            raise ValueError(f"Unpaired {key}")
    if a["config"]["source_sha256"] != b["config"]["source_sha256"]:
        raise ValueError("Code versions differ; do not attribute this to one factor")
    if a["config"].get("runtime") != b["config"].get("runtime"):
        raise ValueError("Runtime/device/precision differ between paired conditions")
    sa, sb = a["summary"], b["summary"]
    if sa["plan_id"] != plan["plan_id"]:
        raise ValueError("Run belongs to another plan")
    for label, run in [("reference", a), ("treatment", b)]:
        actual = {k: v for k, v in run["summary"]["spec"].items() if k not in {"model_seed", "smoke"}}
        if actual != plan["conditions"][contrast[label]]:
            raise ValueError("Run spec does not match preregistered condition")
    if sa["final"][bank]["mask_sha256"] != sb["final"][bank]["mask_sha256"]:
        raise ValueError("Evaluation masks differ")
    factor = contrast["factor"]
    horizon = min(sa["spec"]["steps"], sb["spec"]["steps"])
    pa, pb = sa["milestones"].get(str(horizon)), sb["milestones"].get(str(horizon))
    if pa is None or pb is None:
        raise ValueError("Missing comparable training-prefix digest")
    if pa["streams"]["batch"] != pb["streams"]["batch"]:
        raise ValueError("Training ECG streams differ")
    if factor != "train_policy" and contrast["kind"] != "anchor" and pa["streams"]["mask"] != pb["streams"]["mask"]:
        raise ValueError("Unexpected training-mask stream difference")
    if factor not in {"schedule_horizon", "official_implementation_bundle"} and pa["streams"]["lr"] != pb["streams"]["lr"]:
        raise ValueError("Unexpected LR-prefix difference")
    if contrast["kind"] == "prefix_extension" and pa["state_sha256"] != pb["state_sha256"]:
        raise ValueError("Long run does not share the identical short-run checkpoint")


def paired_patients(a, b, bank):
    keys = ["ecg_id", "patient_id", "repeat"]
    left = a["frame"].loc[a["frame"].bank.eq(bank)]
    right = b["frame"].loc[b["frame"].bank.eq(bank)]
    merged = left.merge(right, on=keys, suffixes=("_a", "_b"), validate="one_to_one", how="outer", indicator=True)
    if not merged._merge.eq("both").all():
        raise ValueError("Patient/ECG/repeat sets differ")
    merged["gain"] = merged.standardized_mse_a - merged.standardized_mse_b
    return merged.groupby("patient_id")[["gain", "standardized_mse_a", "standardized_mse_b"]].mean().sort_index()


def analyze(root, output, *, bootstrap=None):
    root, output = Path(root), Path(output)
    plan = validate_plan(json.loads((root / "plan.json").read_text(encoding="utf-8")))
    repeats = bootstrap or plan["inference"]["bootstrap"]
    loaded = {(name, seed): load_run(root, name, seed) for name in plan["conditions"] for seed in plan["seeds"]}
    scientific = [run for run in loaded.values() if run is not None]
    if len({digest(r["config"]["source_sha256"]) for r in scientific}) > 1:
        raise ValueError("Code versions differ across seeds/conditions; create a coherent study")
    if len({digest(r["config"].get("runtime")) for r in scientific}) > 1:
        raise ValueError("Runtime/device/precision differ across study seeds")
    official_sources = [r["config"]["upstream"]["sha256"] for r in scientific if r["summary"]["spec"]["backend"] == "stmem"]
    if len({digest(value) for value in official_sources}) > 1:
        raise ValueError("Official ST-MEM code differs across seeds")
    output.mkdir(parents=True, exist_ok=True)
    rows, patient_rows, effects = [], [], []
    for contrast in plan["contrasts"]:
        for bank in plan["evaluation_banks"]:
            arrays, base, included, patient_ids = [], [], [], None
            for seed in plan["seeds"]:
                a, b = (loaded[(contrast[k], seed)] for k in ("reference", "treatment"))
                if a is None or b is None or bank not in a["summary"]["final"] or bank not in b["summary"]["final"]:
                    continue
                assert_pair(a, b, plan, contrast, bank)
                paired = paired_patients(a, b, bank)
                if patient_ids is not None and not np.array_equal(paired.index.to_numpy(), patient_ids):
                    raise ValueError("Patient populations differ between seeds")
                patient_ids = paired.index.to_numpy()
                arrays.append(paired.gain.to_numpy())
                base.append(paired.standardized_mse_a.to_numpy())
                included.append(seed)
                for pid, row in paired.iterrows():
                    patient_rows.append(dict(**contrast, bank=bank, seed=seed, patient_id=pid, **row.to_dict()))
            record = dict(**contrast, bank=bank, seeds=",".join(map(str, included)), seed_count=len(included))
            if not arrays:
                rows.append(dict(record, status="missing scientific runs"))
                continue
            matrix = np.stack(arrays)
            stats = crossed_bootstrap(matrix, repeats)
            relative = stats["mean"] / max(float(np.mean(base)), 1e-12)
            minimum = plan["inference"]["minimum_seeds"]
            complete = set(included) == set(plan["seeds"]) and len(included) >= minimum
            meaningful = (complete and stats["ci_low"] > plan["inference"]["practical_floor"] and
                          relative >= plan["inference"]["minimum_relative_mse_reduction"] and
                          stats["positive_seed_fraction"] >= plan["inference"]["seed_consistency"])
            status = ("anchor: several implementation differences" if contrast["kind"] == "anchor" else
                      "consistent candidate; independent confirmation needed" if meaningful else
                      "exploratory: incomplete seeds" if not complete else
                      "no predeclared practical benefit established")
            rows.append(dict(record, **{k: v for k, v in stats.items() if k != "seeds"},
                             relative_reduction=relative, status=status))
    comparisons = pd.DataFrame(rows)
    comparisons.to_csv(output / "paired_effects.csv", index=False)
    pd.DataFrame(patient_rows).to_csv(output / "paired_patient_effects.csv", index=False)
    # Difference-in-differences over the complete 2^3 cube, within every patient/seed.
    cube = {bits: "independent_long" if not any(bits) else "factor_" + "".join(map(str, bits))
            for bits in itertools.product([0, 1], repeat=3)}
    if set(cube.values()) <= set(plan["conditions"]):
        for bank in plan["evaluation_banks"]:
            for factor_pair in itertools.combinations(range(3), 2):
                matrices, conditional = [], {0: [], 1: []}
                kept, patient_index = [], None
                other = next(i for i in range(3) if i not in factor_pair)
                for seed in plan["seeds"]:
                    runs = {bits: loaded[(name, seed)] for bits, name in cube.items()}
                    if any(run is None or bank not in run["summary"]["final"] for run in runs.values()):
                        continue
                    reference = runs[(0, 0, 0)]
                    mse = {}
                    for bits, run in runs.items():
                        assert_pair(reference, run, plan, dict(reference="independent_long", treatment=cube[bits], factor="factorial", kind="anchor"), bank)
                        mse[bits] = paired_patients(reference, run, bank).standardized_mse_b
                    if len({tuple(v.index) for v in mse.values()}) != 1:
                        raise ValueError("Factorial patients mismatch")
                    interaction = 0
                    for level in (0, 1):
                        vectors = {}
                        for a, b in itertools.product([0, 1], repeat=2):
                            bits = [0, 0, 0]
                            bits[other], bits[factor_pair[0]], bits[factor_pair[1]] = level, a, b
                            vectors[(a, b)] = mse[tuple(bits)]
                        # Positive: combined reduction exceeds the sum of individual reductions.
                        level_effect = vectors[(1, 0)] + vectors[(0, 1)] - vectors[(1, 1)] - vectors[(0, 0)]
                        conditional[level].append(level_effect.to_numpy())
                        interaction = interaction + level_effect / 2
                    if patient_index is not None and not np.array_equal(interaction.index.to_numpy(), patient_index):
                        raise ValueError("Factorial populations differ across seeds")
                    patient_index = interaction.index.to_numpy()
                    matrices.append(interaction.to_numpy())
                    kept.append(seed)
                if matrices:
                    effects.append(dict(factors=" x ".join(FACTORS[i] for i in factor_pair), bank=bank,
                        included_seeds=kept, **crossed_bootstrap(np.stack(matrices), repeats),
                        note="Exploratory two-factor interaction averaged over the third factor; positive = beneficial synergy"))
                    for level in (0, 1):
                        effects.append(dict(factors=" x ".join(FACTORS[i] for i in factor_pair), bank=bank,
                            conditioning={FACTORS[other]: level}, included_seeds=kept,
                            **crossed_bootstrap(np.stack(conditional[level]), repeats),
                            note="Interaction at a fixed third-factor level; positive = beneficial synergy"))
                    effects.append(dict(factors="three_way_via_" + "_".join(FACTORS[i] for i in factor_pair), bank=bank,
                        included_seeds=kept, **crossed_bootstrap(np.stack(conditional[1]) - np.stack(conditional[0]), repeats),
                        note="Change in two-factor synergy when the third factor is enabled; three equivalent views of the same three-way interaction"))
    write_json(output / "interactions.json", effects)
    summarize_probes(loaded, output)
    plot_curves(loaded, output)
    lines = ["# Plain ECG-MAE 원인 탐색 결과", "", f"Plan: `{plan['plan_id']}`", "",
             "이 보고서는 실행이 완료된 실제 데이터 실험만 분석합니다. Synthetic smoke와 tiny-overfit은 원인 판단에서 제외합니다.", "",
             "양수 효과는 reference 대비 treatment의 복원 오차 감소입니다. 고정된 train lead 표준편차로 mV 오차를 나눈 뒤, 환자별 평균을 비교합니다.",
             "95% 구간은 seed와 환자를 교차 재표집한 탐색적 bootstrap입니다. 다중 비교 보정이나 독립 데이터 확인을 대체하지 않습니다.", "",
             "사전 지정 탐색 기준: 3 seeds 이상/계획 seed 완료, 모든 seed에서 개선, 상대 오차 10% 이상 감소, 절대 개선 CI 하한 0.02 초과. 보편적인 원인 판정 기준은 아닙니다.", "",
             "| Reference → Treatment | 평가 mask | Seed 수 | 평균 오차 감소 | 95% 구간 | 판단 |", "|---|---|---:|---:|---|---|"]
    for row in rows:
        value = f"{row['mean']:.5f}" if "mean" in row else "—"
        ci = f"[{row['ci_low']:.5f}, {row['ci_high']:.5f}]" if "ci_low" in row else "—"
        lines.append(f"| {row['reference']} → {row['treatment']} | {row['bank']} | {row['seed_count']} | {value} | {ci} | {row['status']} |")
    lines += ["", "결과 해석:", "",
              "- 초기화 대비는 학습량·스케줄·구조가 동일한 비교입니다.",
              "- 학습 연장은 짧은 실행과 모델 상태·학습률·데이터·mask prefix가 일치해야만 분석합니다.",
              "- architecture edge는 한 요소만 바뀝니다. Reversal은 결합 모델에서 해당 요소를 제거했을 때의 효과입니다.",
              "- 공식 ST-MEM anchor는 여러 구현 차이가 함께 남아 있으므로 단일 원인의 근거로 쓰지 않습니다.",
              "- 복원 개선과 linear-probe 개선은 별개입니다. probe_comparison.csv의 random encoder 대비 이득과 300→1500 update 곡선을 함께 확인하세요.",
              "- 하나의 seed/전처리/학습량에서 찾은 원인이 다른 조건에서도 동일하다고 단정할 수 없습니다.",
              "- 필요한 비교가 누락되거나 개선이 없으면 원인 미확정 상태입니다."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(output / "analysis_audit.json", dict(plan_id=plan["plan_id"], bootstrap_repeats=repeats,
        scientific_runs=sum(v is not None for v in loaded.values()), excluded_synthetic=plan["smoke"],
        inference="exploratory crossed seed/patient bootstrap", test_opened=False))
    return comparisons


def summarize_probes(loaded, output):
    frames = []
    for (name, seed), run in loaded.items():
        if run is None:
            continue
        for folder in [run["folder"], run["folder"] / "probe_audit_standard", run["folder"] / "probe_audit_expanded"]:
            path = folder / "probe_comparison.csv"
            if not path.exists():
                continue
            if folder != run["folder"]:
                audit = json.loads((folder / "audit.json").read_text(encoding="utf-8"))
                if audit["parent_fingerprint"] != run["summary"]["fingerprint"]:
                    raise ValueError("Posthoc probe does not belong to the saved encoder")
            frame = pd.read_csv(path)
            frame = frame.assign(condition=name, seed=seed, source="integrated" if folder == run["folder"] else folder.name)
            frames.append(frame)
    if not frames:
        return
    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(output / "probe_seed_results.csv", index=False)
    metrics = ["macro_auroc_trained", "macro_auroc_random", "auroc_gain_vs_random", "macro_auprc_trained", "macro_f1_trained"]
    supported = frame.all_five_supported_trained & frame.all_five_supported_random
    frame.loc[~supported, metrics] = np.nan
    summary = frame.groupby(["condition", "pooling", "input", "update", "source"])[metrics].agg(["count", "mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.reset_index().to_csv(output / "probe_summary.csv", index=False)


def plot_curves(loaded, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    frames = []
    for (name, seed), run in loaded.items():
        if run is not None:
            h = pd.read_csv(run["folder"] / "evaluation_history.csv")
            bank = "per_lead" if h.bank.eq("per_lead").any() else "global"
            frames.append(h[h.bank.eq(bank)].assign(condition=name, seed=seed))
    data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    panels = {
        "reconstruction_curves": ["legacy_short", "independent_short", "independent_extend", "independent_long", "all_on", "official_stmem"],
        "factorial_curves": ["independent_long", *["factor_" + "".join(map(str, b)) for b in itertools.product([0, 1], repeat=3) if any(b)]],
        "normalization_curves": ["independent_long", "normalization_train_lead", "normalization_train_common", "normalization_visible_record"],
    }
    metrics = ["standardized_nrmse", "rms_ratio", "masked_correlation"]
    for filename, names in panels.items():
        fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True, constrained_layout=True)
        try:
            selected = data[data.condition.isin(names)] if not data.empty else data
            if not selected.empty:
                for color, (name, group) in enumerate(selected.groupby("condition", sort=False)):
                    for split, split_rows in group.groupby("split"):
                        means = split_rows.groupby("update")[metrics].mean()
                        stds = split_rows.groupby("update")[metrics].std().fillna(0)
                        for ax, metric in zip(axes, metrics):
                            x, y, sd = means.index.to_numpy(), means[metric].to_numpy(), stds[metric].to_numpy()
                            ax.plot(x, y, "--" if split == "train" else "-", color=f"C{color % 10}", label=f"{name}/{split}")
                            ax.fill_between(x, y-sd, y+sd, color=f"C{color % 10}", alpha=.06)
                axes[0].legend(fontsize=7, ncol=2)
            else:
                axes[0].text(.5, .5, "No completed scientific runs for this panel", ha="center", transform=axes[0].transAxes)
            for ax, label in zip(axes, ["Physical mV / fixed train std: masked NRMSE", "Prediction / target RMS", "Hidden-sample correlation"]):
                ax.set_ylabel(label)
                ax.grid(alpha=.2)
            axes[-1].set_xlabel("Optimizer update")
            fig.suptitle("Fixed masks | mean over seeds, shading: sample SD (not CI)")
            fig.savefig(output / f"{filename}.png", dpi=120)
        finally:
            plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bootstrap", type=int)
    args = p.parse_args()
    if args.bootstrap is not None and args.bootstrap < 100:
        p.error("Use at least 100 bootstrap replicates")
    analyze(args.root, args.output, bootstrap=args.bootstrap)


if __name__ == "__main__":
    main()
