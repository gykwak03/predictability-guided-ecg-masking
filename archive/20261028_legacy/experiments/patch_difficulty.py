"""Within-ECG patch difficulty under exact 50% mask contexts.

Discovery assigns groups. Independent confirmation contexts evaluate those frozen
groups. No labels enter scoring; patches and masks are not independent patients.
"""
from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd

ESTIMATORS = ("ridge", "cnn", "transformer")
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
BANKS = ("discovery", "confirmation")
GROUPS = ("Easy", "Medium", "Hard")


def complementary_bank(metadata, tokens, pairs, seed):
    """Each independent pair is (M, ~M); each patch is hidden once per pair.

    Every M has exactly tokens/2 hidden positions. Discovery and confirmation use
    distinct seed namespaces. Identity, not row order, determines generated masks.
    """
    if tokens < 6 or tokens % 2 or pairs < 2 or seed < 0:
        raise ValueError("Need even tokens >= 6, pairs >= 2 and nonnegative seed")
    required = {"ecg_id", "patient_id", "artifact_index"}
    if not required.issubset(metadata) or metadata.empty or metadata.ecg_id.duplicated().any():
        raise ValueError("Unique nonempty ECG metadata required")
    if metadata[list(required)].isna().any().any():
        raise ValueError("Missing ECG/patient/artifact identity")
    rows, masks = [], []
    for row in metadata.itertuples(index=False):
        for bank in BANKS:
            for pair in range(pairs):
                key = f"patch50:{seed}:{tokens}:{bank}:{int(row.ecg_id)}:{pair}"
                draw_seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little") & ((1 << 63)-1)
                m = np.zeros(tokens, dtype=bool)
                m[np.random.default_rng(draw_seed).choice(tokens, tokens//2, replace=False)] = True
                for half, hidden in enumerate((m, ~m)):
                    rows.append(dict(mask_id=f"{bank}-e{int(row.ecg_id)}-p{pair}-h{half}",
                        ecg_id=int(row.ecg_id), patient_id=row.patient_id,
                        artifact_index=int(row.artifact_index), bank=bank, pair=pair, half=half,
                        mask_seed=draw_seed, hidden_tokens=tokens//2, realized_mask_ratio=.5,
                        mask_sha256=hashlib.sha256(np.packbits(hidden).tobytes()).hexdigest()))
                    masks.append(hidden.copy())
    return pd.DataFrame(rows), np.stack(masks)


def replay_bank(bank, tokens):
    """Reproduce exported complement masks, including half=1, and check hashes."""
    if bank.mask_id.duplicated().any() or bank.empty:
        raise ValueError("Unique nonempty mask bank required")
    masks = []
    for row in bank.itertuples(index=False):
        if row.half not in (0, 1) or row.hidden_tokens*2 != tokens or row.realized_mask_ratio != .5:
            raise ValueError("Only exact complementary 50% masks supported")
        m = np.zeros(tokens, bool)
        m[np.random.default_rng(int(row.mask_seed)).choice(tokens, tokens//2, replace=False)] = True
        if row.half:
            m = ~m
        if hashlib.sha256(np.packbits(m).tobytes()).hexdigest() != row.mask_sha256:
            raise ValueError(f"Mask digest mismatch: {row.mask_id}")
        masks.append(m)
    return np.stack(masks)


def rank_percentile(values):
    """High percentile = hard. Ties retain identical scores, including constant=50."""
    values = np.asarray(values, dtype=float)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("Finite patch scores required")
    return 100*(pd.Series(values).rank(method="average").to_numpy()-.5)/len(values)


def rank_correlation(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3 or any(np.ptp(v) <= 1e-12*max(1., np.max(np.abs(v))) for v in (a[valid], b[valid])):
        return float("nan")
    return float(np.corrcoef(pd.Series(a[valid]).rank(), pd.Series(b[valid]).rank())[0, 1])


def patch_statistics(errors):
    """Pool squared normalized residuals across hidden contexts before sqrt."""
    x = np.asarray(errors, dtype=float)
    if x.ndim != 3 or np.isinf(x).any() or (x[np.isfinite(x)] < 0).any():
        raise ValueError("Expected [contexts, leads, time-patches] nonnegative NRMSE or NaN")
    count = np.isfinite(x).sum(axis=0)
    if (count < 2).any():
        raise ValueError("Every patch needs >= 2 hidden observations in EACH bank")
    mean = np.nansum(x, axis=0)/count
    mse = np.nansum(x*x, axis=0)/count
    variance = np.maximum(0, np.nansum((x-mean)**2, axis=0)/(count-1))
    return dict(count=count.ravel(), nrmse=np.sqrt(mse).ravel(),
                context_mean_nrmse=mean.ravel(), context_std_nrmse=np.sqrt(variance).ravel())


def frozen_groups(scores):
    order = np.argsort(scores, kind="stable")
    groups = np.empty(len(scores), dtype=object)
    for name, indices in zip(GROUPS, np.array_split(order, 3)):
        groups[indices] = name
    boundary_ties = sum(scores[a[-1]] == scores[b[0]]
                        for a, b in zip(np.array_split(order, 3)[:-1], np.array_split(order, 3)[1:]))
    return groups, int(boundary_ties)


def analyze_ecg(errors, zero_patch_nrmse, identity, patch_samples=50,
                sampling_rate=250, shuffle_repeats=32, seed=42):
    """Return patch rows and per-ECG confirmation evidence for a fixed estimator set."""
    if set(errors) != set(ESTIMATORS) or shuffle_repeats < 1:
        raise ValueError("Need all three estimators and positive shuffle_repeats")
    zero = np.asarray(zero_patch_nrmse, float)
    if zero.ndim != 2 or zero.shape[0] != 12 or not np.isfinite(zero).all() or (zero < 0).any():
        raise ValueError("Expected finite nonnegative [12, time-patches] zero baseline")
    tokens = zero.size
    all_stats, score, patterns, inclusion = {}, {}, {}, {}
    for name in ESTIMATORS:
        if set(errors[name]) != set(BANKS):
            raise ValueError("Need discovery and confirmation for each estimator")
        all_stats[name], score[name], inclusion[name] = {}, {}, {}
        for bank in BANKS:
            x = np.asarray(errors[name][bank])
            if x.shape[1:] != zero.shape:
                raise ValueError("Patch geometry mismatch")
            pattern = np.isfinite(x)
            if (pattern.reshape(len(x), -1).sum(axis=1)*2 != tokens).any():
                raise ValueError("Each context must hide exactly 50% of patches")
            if bank in patterns and not np.array_equal(pattern, patterns[bank]):
                raise ValueError("Estimator hidden coverage mismatch")
            patterns[bank] = pattern
            stats = patch_statistics(x)
            all_stats[name][bank] = stats
            score[name][bank] = rank_percentile(stats["nrmse"])
            # Whole-task association, distinct from this patch's own error.
            # Inclusion changes context AND target composition, so is not causal.
            flat = pattern.reshape(len(x), -1)
            whole_error = np.sqrt(np.nansum(x*x, axis=(1, 2))/(tokens//2))
            inclusion[name][bank] = ((whole_error[:, None]*flat).sum(axis=0)/flat.sum(axis=0)
                -(whole_error[:, None]*(~flat)).sum(axis=0)/(~flat).sum(axis=0))
    consensus = {b: np.mean([score[a][b] for a in ESTIMATORS], axis=0) for b in BANKS}
    inclusion_consensus = {b: np.mean([rank_percentile(inclusion[a][b]) for a in ESTIMATORS], axis=0) for b in BANKS}
    groups, boundary_ties = frozen_groups(consensus["discovery"])
    inclusion_groups, inclusion_ties = frozen_groups(inclusion_consensus["discovery"])
    confirm_groups, _ = frozen_groups(consensus["confirmation"])
    flat_zero = zero.ravel()
    energetic = flat_zero**2 > 1e-8
    result = dict(identity, group_boundary_ties=boundary_ties,
                  consensus_spearman=rank_correlation(consensus["discovery"], consensus["confirmation"]),
                  hard_overlap=float(np.mean(confirm_groups[groups == "Hard"] == "Hard")),
                  chance_hard_overlap=float(np.mean(confirm_groups == "Hard")),
                  energy_spearman=rank_correlation(consensus["discovery"], flat_zero),
                  energetic_patches=int(energetic.sum()))
    result.update(inclusion_consensus_spearman=rank_correlation(
                      inclusion_consensus["discovery"], inclusion_consensus["confirmation"]),
                  inclusion_group_boundary_ties=inclusion_ties,
                  tie_boundary_fraction=float(boundary_ties > 0))
    rng = np.random.default_rng(seed + int(identity["ecg_id"]))
    # Same shuffled labels for all estimators. Only discovery groups are shuffled.
    permutations = [rng.permutation(groups) for _ in range(shuffle_repeats)]
    patch_rows = []
    for token in range(tokens):
        lead, pos = divmod(token, zero.shape[1])
        record = dict(identity, token_index=token, patch_number=token+1, lead=LEADS[lead],
                      time_patch_index=pos, start_seconds=pos*patch_samples/sampling_rate,
                      end_seconds=(pos+1)*patch_samples/sampling_rate,
                      discovery_group=groups[token], zero_nrmse=flat_zero[token],
                      inclusion_discovery_group=inclusion_groups[token],
                      discovery_consensus=consensus["discovery"][token],
                      confirmation_consensus=consensus["confirmation"][token],
                      discovery_disagreement=np.std([score[a]["discovery"][token] for a in ESTIMATORS]))
        for name in ESTIMATORS:
            for bank in BANKS:
                for key, values in all_stats[name][bank].items():
                    record[f"{name}_{bank}_{key}"] = values[token]
                record[f"{name}_{bank}_whole_mask_inclusion_gap"] = inclusion[name][bank][token]
        patch_rows.append(record)
    for name in ESTIMATORS:
        d = all_stats[name]["discovery"]["nrmse"]
        c = all_stats[name]["confirmation"]["nrmse"]
        means = {g: float(c[groups == g].mean()) for g in GROUPS}
        gap = means["Hard"] - means["Easy"]
        null_gaps = [float(c[p == "Hard"].mean()-c[p == "Easy"].mean()) for p in permutations]
        # Skill vs fixed zero predictor is an energy diagnostic, not clinical utility.
        dskill = np.divide(d*d, flat_zero**2, out=np.full(tokens, np.nan), where=energetic)
        cskill = np.divide(c*c, flat_zero**2, out=np.full(tokens, np.nan), where=energetic)
        energy_groups, _ = frozen_groups(rank_percentile(flat_zero))
        result.update({f"{name}_spearman": rank_correlation(d, c),
                       f"{name}_energy_adjusted_spearman": rank_correlation(dskill, cskill),
                       f"{name}_inclusion_spearman": rank_correlation(inclusion[name]["discovery"], inclusion[name]["confirmation"]),
                       f"{name}_inclusion_group_gap": float(
                           inclusion[name]["confirmation"][inclusion_groups == "Hard"].mean()
                           -inclusion[name]["confirmation"][inclusion_groups == "Easy"].mean()),
                       f"{name}_gap": gap,
                       f"{name}_hard_easy_ratio": means["Hard"]/means["Easy"] if means["Easy"] > 0 else np.nan,
                       f"{name}_ordered": float(means["Easy"] < means["Medium"] < means["Hard"]),
                       f"{name}_null_gap_mean": float(np.mean(null_gaps)),
                       f"{name}_gap_above_null": gap-float(np.mean(null_gaps)),
                       f"{name}_energy_only_gap": float(c[energy_groups == "Hard"].mean()-c[energy_groups == "Easy"].mean()),
                       **{f"{name}_confirmation_{g.lower()}_nrmse": value for g, value in means.items()}})
    result["mean_estimator_gap_above_null"] = float(np.mean([result[f"{a}_gap_above_null"] for a in ESTIMATORS]))
    result["all_estimators_ordered"] = float(all(result[f"{a}_ordered"] for a in ESTIMATORS))
    result["mean_inclusion_group_gap"] = float(np.mean([result[f"{a}_inclusion_group_gap"] for a in ESTIMATORS]))
    return pd.DataFrame(patch_rows), result


def summarize_ecgs(frame, bootstrap_repeats=2000, seed=42, mode="pilot"):
    """Equal-patient summaries; bootstrap patients, never individual patches."""
    if frame.empty or frame.ecg_id.duplicated().any() or frame.patient_id.isna().any() or bootstrap_repeats < 1:
        raise ValueError("Unique nonempty ECG rows, patient IDs and positive bootstrap count required")
    metrics = ["consensus_spearman", "hard_overlap", "chance_hard_overlap", "all_estimators_ordered",
               "mean_estimator_gap_above_null", "energy_spearman", "tie_boundary_fraction",
               "inclusion_consensus_spearman", "mean_inclusion_group_gap"]
    metrics += [f"{a}_{m}" for a in ESTIMATORS for m in
                ("spearman", "energy_adjusted_spearman", "gap", "hard_easy_ratio", "energy_only_gap", "inclusion_spearman")]
    patients = frame.groupby("patient_id")[metrics].mean()
    rng = np.random.default_rng(seed)
    # Shared patient draws across metrics retain their dependence.
    draws = rng.integers(0, len(patients), size=(bootstrap_repeats, len(patients)))
    summaries = {}
    for metric in metrics:
        values = patients[metric].to_numpy()
        valid = np.isfinite(values)
        sampled = values[draws]
        counts = np.isfinite(sampled).sum(axis=1)
        boot = np.divide(np.nansum(sampled, axis=1), counts,
                         out=np.full(bootstrap_repeats, np.nan), where=counts > 0)
        finite_boot = boot[np.isfinite(boot)]
        summaries[metric] = dict(mean=float(np.mean(values[valid])) if valid.any() else np.nan,
                                valid_patients=int(valid.sum()),
                                ci95_low=float(np.quantile(finite_boot, .025)) if len(finite_boot) else np.nan,
                                ci95_high=float(np.quantile(finite_boot, .975)) if len(finite_boot) else np.nan)
    # Predeclared pilot gates, not universal scientific cutoffs or downstream proof.
    def mean(name):
        return summaries[name]["mean"]

    gates = dict(enough_patients=len(patients) >= 10,
                 enough_valid_rank_patients=bool(all(summaries[m]["valid_patients"] >= 10 for m in
                     ["consensus_spearman", *[f"{a}_spearman" for a in ESTIMATORS]])),
                 sufficient_contexts=bool((frame.min_hidden_contexts >= 16).all()),
                 few_forced_tie_boundaries=bool(mean("tie_boundary_fraction") <= .1),
                 stable_consensus=bool(mean("consensus_spearman") >= .6),
                 hard_overlap_above_chance=bool(mean("hard_overlap") >= .6),
                 three_estimator_ordering=bool(mean("all_estimators_ordered") >= .8),
                 gap_above_shuffle=bool(summaries["mean_estimator_gap_above_null"]["ci95_low"] > 0),
                 estimator_rank_replication=bool(all(mean(f"{a}_spearman") >= .5 for a in ESTIMATORS)))
    passed = all(gates.values()) and mode == "pilot"
    inclusion_gates = dict(enough_patients=gates["enough_patients"], sufficient_contexts=gates["sufficient_contexts"],
                           enough_valid_rank_patients=summaries["inclusion_consensus_spearman"]["valid_patients"] >= 10,
                           stable_inclusion_ranking=bool(mean("inclusion_consensus_spearman") >= .6),
                           confirmation_group_gap=bool(summaries["mean_inclusion_group_gap"]["ci95_low"] > 0))
    return dict(status="supports_conditional_patch_ranking" if passed else
                ("smoke_only" if mode == "smoke" else "inconclusive"),
                whole_mask_inclusion_status="supports_inclusion_association" if all(inclusion_gates.values()) and mode == "pilot"
                                           else ("smoke_only" if mode == "smoke" else "inconclusive"),
                whole_mask_inclusion_gates=inclusion_gates,
                ecgs=len(frame), patients=len(patients), gates=gates, metrics=summaries,
                thresholds="Exploratory predeclared gates; do not tune after viewing confirmation results.",
                interpretation="Supports repeatable within-ECG patch ranking under this 50% random-context distribution and fixed checkpoints only. Does not prove a causal effect of hiding a patch, a useful masking policy, architecture independence, or downstream improvement.",
                energy_check="Inspect energy-adjusted rank replication and energy-only gap; stable raw error can reflect waveform energy rather than contextual predictability.")
