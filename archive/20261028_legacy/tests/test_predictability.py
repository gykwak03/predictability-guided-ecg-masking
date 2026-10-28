"""Scientific contracts: hidden isolation, held-out data, errors and mask ranking."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from predictability_estimators import (LaggedRidge, SmallCNN, SmallTransformer, difficulty_table,
                                      lead_scale, mask_bank, masks_from_bank, normalized_masked_loss, patient_pools,
                                      predictability_percentile, reconstruction_metrics, sample_hidden)
from run_predictability import aggregate_metrics, load_data, parser, run


def metadata(n=30):
    return pd.DataFrame(dict(ecg_id=np.arange(100, 100+n), patient_id=np.arange(n)//2,
                             artifact_index=np.arange(n), strat_fold=np.ones(n, dtype=int)))


def test_patient_split_no_leakage_and_seed_replay():
    meta = metadata()
    fit, score = patient_pools(meta, 8, 4, 42)
    again = patient_pools(meta, 8, 4, 42)
    assert np.array_equal(fit, again[0]) and np.array_equal(score, again[1])
    assert not set(meta.iloc[fit].patient_id) & set(meta.iloc[score].patient_id)
    assert len(fit) == 8 and len(score) == 4
    with pytest.raises(ValueError, match="exceeds"):
        patient_pools(meta, 100, 4, 42)


def test_mask_bank_identity_exact_count_and_row_order_independence():
    meta = metadata()
    first, masks = mask_bank(meta, [0, 3], 50, 2500, 0.5, 3, 42, "score")
    second, reversed_masks = mask_bank(meta, [3, 0], 50, 2500, 0.5, 3, 42, "score")
    order = second.set_index("mask_id").loc[first.mask_id].index
    replay = dict(zip(second.mask_id, reversed_masks))
    assert np.array_equal(masks, np.stack([replay[key] for key in order]))
    assert (masks.sum(axis=1) == 300).all()
    assert len(set(first.mask_sha256)) == 6
    for row, mask in zip(first.itertuples(), masks):
        expected = np.zeros(600, bool)
        expected[np.random.default_rng(row.mask_seed).choice(600, 300, replace=False)] = True
        assert np.array_equal(expected, mask)


def test_fixed_lead_std_fits_only_given_train_ecgs():
    x = np.broadcast_to(np.array([[-2., 2.], [-4., 4.]]), (4, 2, 2)).copy()
    x[3] *= 100
    mean, scale = lead_scale(x, np.array([0, 1, 2]), batch_size=2)
    assert np.allclose(mean, 0) and np.allclose(scale, [2, 4])


def test_exported_bank_csv_preserves_large_seed_and_mask_digest(tmp_path):
    bank, masks = mask_bank(metadata(3), [0, 1, 2], 50, 2500, 0.5, 3, 42, "score")
    path = tmp_path / "bank.csv"
    bank.to_csv(path, index=False)
    loaded = pd.read_csv(path)
    assert np.array_equal(masks_from_bank(loaded, 600), masks)
    loaded.loc[0, "mask_seed"] += 1
    with pytest.raises(ValueError, match="digest mismatch"):
        masks_from_bank(loaded, 600)


def test_nrmse_sample_pooling_and_no_visible_copy_credit():
    target = np.tile([2., -2., 4., -4.], (1, 2, 1))
    pred = target.copy()
    mask = np.array([[True, False, False, True]])
    hidden = sample_hidden(mask, target.shape)
    pred[hidden] += np.broadcast_to(np.array([2., 4.])[None, :, None], target.shape)[hidden]
    pred[~hidden] = 9999  # visible predictions have no effect
    stats, patches = reconstruction_metrics(pred, target, mask, [2, 4])
    assert stats.nrmse.iloc[0] == pytest.approx(1)
    assert stats.masked_samples.iloc[0] == 4
    assert np.allclose(patches[np.isfinite(patches)], 1)
    assert np.isnan(patches[0, 0, 1]) and np.isnan(patches[0, 1, 0])
    loss = normalized_masked_loss(torch.tensor(pred), torch.tensor(target), torch.tensor(mask), torch.tensor([2., 4.]))
    assert loss.item() == pytest.approx(1)


def test_flat_targets_have_undefined_r2_and_correlation():
    stats, _ = reconstruction_metrics(np.zeros((1, 12, 100)), np.ones((1, 12, 100)),
                                      np.tile([True, False], (1, 12)), np.ones(12))
    assert np.isnan(stats.masked_r2.iloc[0]) and np.isnan(stats.masked_pearson.iloc[0])


@pytest.mark.parametrize("kind", ["ridge", "cnn", "transformer"])
def test_estimators_isolate_hidden_values_and_use_visible_context(kind):
    torch.set_num_threads(2)
    rng = np.random.default_rng(8)
    x = rng.normal(size=(4, 12, 100)).astype(np.float32)
    # Construct strongly correlated leads; tests ridge is an actual useful linear fit.
    x[:, 1:] = x[:, :1]*np.arange(1, 12)[None, :, None]
    bank, mask = mask_bank(metadata(4), [0, 1, 2, 3], 10, 100, 0.5, 1, 42, "fit")
    hidden = sample_hidden(mask, x.shape)
    changed = np.where(hidden, x+10000, x).astype(np.float32)
    if kind == "ridge":
        model = LaggedRidge(10, alpha=1e-3).fit(x, bank, mask, np.ones(12), samples_per_ecg=100)
        predict = lambda value: model.predict(value, mask)
    else:
        torch.manual_seed(42)
        model = SmallCNN(width=8) if kind == "cnn" else SmallTransformer(10, 100, width=16, depth=2)
        model.eval()
        def predict(value):
            with torch.no_grad():
                return model(torch.tensor(value), torch.tensor(mask)).numpy()
    initial = predict(x)
    assert np.array_equal(initial, predict(changed))
    assert np.max(np.abs(initial-predict(np.where(hidden, x, x+2).astype(np.float32)))) > 1e-5


def test_ridge_recovers_known_cross_lead_linear_relationship():
    rng = np.random.default_rng(19)
    x = rng.normal(size=(20, 12, 100)).astype(np.float32)
    x[:, :6] = 2*x[:, 6:]-1
    mask = np.zeros((20, 12, 10), bool)
    mask[:, :6] = True
    mask = mask.reshape(20, -1)
    # Fit both directions so every output lead has hidden training targets.
    bank = pd.concat([metadata(20), metadata(20)], ignore_index=True)
    fit_masks = np.concatenate([mask, ~mask])
    ridge = LaggedRidge(10, lags=(0,)).fit(x, bank, fit_masks, np.ones(12), samples_per_ecg=100)
    stats, _ = reconstruction_metrics(ridge.predict(x, mask), x, mask, np.ones(12))
    assert aggregate_metrics(stats)["skill_vs_zero"] > 0.999


def test_ridge_temporal_context_recovers_without_same_time_visible_lead():
    # Constant per-record amplitudes, varying across ECGs. Entire time patch hidden
    # in every lead: adjacent visible patches are the only available information.
    values = np.linspace(-2, 2, 20)
    x = np.broadcast_to(values[:, None, None], (20, 12, 30)).copy().astype(np.float32)
    mask = np.zeros((20, 36), bool).reshape(20, 12, 3)
    mask[:, :, 1] = True
    mask = mask.reshape(20, -1)
    bank = metadata(20)
    ridge = LaggedRidge(10).fit(x, bank, mask, np.ones(12), samples_per_ecg=100)
    stats, _ = reconstruction_metrics(ridge.predict(x, mask), x, mask, np.ones(12))
    assert aggregate_metrics(stats)["pooled_nrmse"] < 0.02


@pytest.mark.parametrize("kind", ["cnn", "transformer"])
def test_neural_fixed_mask_can_learn_small_signal(kind):
    torch.set_num_threads(2)
    torch.manual_seed(7)
    time_axis = np.linspace(0, 4*np.pi, 100)
    x = np.broadcast_to(np.sin(time_axis)[None, None], (2, 12, 100)).copy().astype(np.float32)
    _, masks = mask_bank(metadata(2), [0, 1], 10, 100, 0.5, 1, 42, "fit")
    x, masks = torch.tensor(x), torch.tensor(masks)
    model = SmallCNN(width=8) if kind == "cnn" else SmallTransformer(10, 100, width=16, depth=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    losses = []
    for _ in range(50):
        optimizer.zero_grad()
        loss = normalized_masked_loss(model(x, masks), x, masks, torch.ones(12))
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    assert losses[-1] < losses[0]*0.2


def metric_cases(n=7, reverse=False):
    rows = []
    for name in ("ridge", "cnn", "transformer"):
        for i in range(n):
            rows.append(dict(mask_id=f"mask-{i}", estimator=name,
                             nrmse=float(n-i if reverse and name == "transformer" else i+1)))
    return pd.DataFrame(rows)


def test_percentiles_direction_ties_and_global_tertiles():
    assert np.allclose(predictability_percentile([3, 3, 3]), 50)
    assert np.allclose(predictability_percentile([1, 2, 3]), [100*2.5/3, 50, 100*.5/3])
    result, rho = difficulty_table(metric_cases())
    assert result.iloc[0].mask_id == "mask-0" and result.iloc[-1].mask_id == "mask-6"
    assert result.difficulty.value_counts().to_dict() == {"Easy": 3, "Medium": 2, "Hard": 2}
    assert np.allclose(result.disagreement_std, 0, atol=1e-12) and np.allclose(rho.spearman_rho, 1)
    shuffled, _ = difficulty_table(metric_cases().sample(frac=1, random_state=42))
    pd.testing.assert_frame_equal(result, shuffled)
    disagreed, negative = difficulty_table(metric_cases(reverse=True))
    assert disagreed.disagreement_range.max() > 50
    assert negative.spearman_rho.min() == pytest.approx(-1)


def test_missing_estimator_cannot_silently_change_consensus():
    with pytest.raises(ValueError, match="All three"):
        difficulty_table(metric_cases().iloc[:-1])
    with pytest.raises(ValueError, match="One metric"):
        difficulty_table(pd.concat([metric_cases(), metric_cases().iloc[:1]]))
    invalid = metric_cases()
    invalid.loc[0, "nrmse"] = np.inf
    with pytest.raises(ValueError, match="Finite"):
        difficulty_table(invalid)


def test_constant_spearman_and_balanced_tie_policy():
    cases = metric_cases(6)
    cases["nrmse"] = 1
    table, correlations = difficulty_table(cases)
    assert np.allclose(table.consensus_predictability, 50)
    assert correlations.spearman_rho.isna().all()
    assert (table.difficulty.value_counts() == 2).all()


def test_smoke_end_to_end_without_test_files(tmp_path):
    import json
    data = tmp_path / "data"
    data.mkdir()
    from run_predictability import LEADS
    (data / "manifest.json").write_text(json.dumps(dict(lead_order=LEADS, config=dict(target_sampling_rate=250,
                                              normalization="per-record"))), encoding="utf-8")
    rng = np.random.default_rng(42)
    for split, n, offset, fold in (("train", 30, 0, 1), ("validation", 6, 1000, 9)):
        frame = metadata(n)
        frame.ecg_id += offset
        frame.patient_id += offset
        frame["strat_fold"] = fold
        frame.to_csv(data / f"metadata_{split}.csv", index=False)
        x = rng.normal(size=(n, 12, 2500)).astype(np.float32)
        np.save(data / f"signals_{split}.npy", x)
    args = parser().parse_args(["--artifact-dir", str(data), "--output-root", str(tmp_path / "results"),
                                "--steps", "1", "--score-masks", "1", "--ridge-samples", "4", "--cpu-threads", "2"])
    output = run(args)
    table = pd.read_csv(output / "mask_difficulty.csv")
    assert len(table) == 6 and (table.difficulty.value_counts() == 2).all()
    assert (output / "lead_time_predictability_curves.png").stat().st_size > 1000
    maps = pd.read_csv(output / "lead_time_predictability.csv")
    assert len(maps) == 4*12*50
    assert (maps.groupby("estimator").hidden_observations.sum() == 6*300).all()
    assert (output / "completion.json").is_file()
    summary = pd.read_csv(output / "estimator_summary.csv")
    assert len(summary) == 6 and np.isfinite(summary.pooled_nrmse).all()
    config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
    assert config["test_opened"] is False and config["patch_samples"] == 50 and config["mask_ratio"] == 0.5
    selection = pd.read_csv(output / "ecg_selection.csv")
    assert not set(selection.loc[selection.role == "fit", "patient_id"]) & set(selection.loc[selection.role == "score", "patient_id"])
    completion = json.loads((output / "completion.json").read_text(encoding="utf-8"))
    assert len(set(completion["neural_first_batch_sha256"].values())) == 1
    assert len(set(completion["neural_train_sequence_sha256"].values())) == 1


def test_loader_rejects_patient_leakage_before_training(tmp_path):
    import json
    from run_predictability import LEADS
    (tmp_path / "manifest.json").write_text(json.dumps(dict(lead_order=LEADS, config=dict(target_sampling_rate=250))))
    for split, fold in (("train", 1), ("validation", 9)):
        frame = metadata(2)
        frame["strat_fold"] = fold
        frame.to_csv(tmp_path / f"metadata_{split}.csv", index=False)
        np.save(tmp_path / f"signals_{split}.npy", np.zeros((2, 12, 2500), np.float32))
    with pytest.raises(ValueError, match="patient leakage"):
        load_data(tmp_path)
