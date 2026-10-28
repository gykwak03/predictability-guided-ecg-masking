"""Scientific controls: exact masks, independent confirmation, null and energy."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from patch_difficulty import (BANKS, ESTIMATORS, LEADS, analyze_ecg, complementary_bank,
                              patch_statistics, rank_correlation, replay_bank, summarize_ecgs)
from predictability_estimators import SmallCNN, SmallTransformer
from run_patch_difficulty import load_source, parser, run


def meta(n=2):
    return pd.DataFrame(dict(ecg_id=np.arange(100, 100+n), patient_id=np.arange(n), artifact_index=np.arange(n)))


def synthetic(n=12, pairs=24, kind="stable"):
    metadata = meta(n)
    bank, masks = complementary_bank(metadata, 120, pairs, 42)
    outputs, patch_rows = [], []
    rng = np.random.default_rng(400)
    for row in metadata.itertuples(index=False):
        errors = {a: {} for a in ESTIMATORS}
        zero = np.ones((12, 10))
        if kind == "energy":
            zero = np.linspace(.1, 3, 120).reshape(12, 10)
        for a in ESTIMATORS:
            for b in BANKS:
                m = masks[(bank.ecg_id == row.ecg_id).to_numpy() & (bank.bank == b).to_numpy()].reshape(-1, 12, 10)
                if kind == "stable":
                    values = np.linspace(.1, 2, 120).reshape(12, 10) + rng.normal(0, .01, m.shape)
                elif kind == "energy":
                    values = np.broadcast_to(zero*.5, m.shape).copy()
                elif kind == "null":
                    values = rng.uniform(.1, 2, m.shape)
                else:
                    raise AssertionError(kind)
                errors[a][b] = np.where(m, values, np.nan)
        patches, stats = analyze_ecg(errors, zero, row._asdict(), shuffle_repeats=8)
        stats["min_hidden_contexts"] = pairs
        outputs.append(stats)
        patch_rows.append(patches)
    return pd.DataFrame(outputs), pd.concat(patch_rows, ignore_index=True)


def test_complement_bank_exact_50_balanced_reproducible_and_csv_replay(tmp_path):
    bank, masks = complementary_bank(meta(), 600, 16, 42)
    assert (masks.sum(axis=1) == 300).all()
    for _, part in bank.groupby(["ecg_id", "bank"]):
        selected = masks[part.index]
        assert (selected.sum(axis=0) == 16).all()
        assert np.array_equal(selected[::2], ~selected[1::2])
    path = tmp_path / "bank.csv"
    bank.to_csv(path, index=False)
    loaded = pd.read_csv(path)
    assert np.array_equal(masks, replay_bank(loaded, 600))
    reverse, reverse_masks = complementary_bank(meta().iloc[::-1], 600, 16, 42)
    lookup = dict(zip(reverse.mask_id, reverse_masks))
    assert np.array_equal(masks, np.stack([lookup[i] for i in bank.mask_id]))
    assert not set(bank.loc[bank.bank == "discovery", "mask_sha256"]) & set(bank.loc[bank.bank == "confirmation", "mask_sha256"])
    loaded.loc[0, "half"] = 1
    with pytest.raises(ValueError, match="digest"):
        replay_bank(loaded, 600)


def test_metrics_pool_squared_error_and_reject_missing_patch():
    x = np.array([[[1., 3.]], [[3., 1.]]])
    assert np.allclose(patch_statistics(x)["nrmse"], np.sqrt(5))
    assert np.allclose(patch_statistics(x)["context_mean_nrmse"], 2)
    with pytest.raises(ValueError, match="observations"):
        patch_statistics(np.array([[[np.nan, 1]], [[np.nan, 2]]]))


def test_stable_patch_signal_replicates_on_independent_contexts():
    frame, patches = synthetic()
    summary = summarize_ecgs(frame, bootstrap_repeats=100)
    assert summary["status"] == "supports_conditional_patch_ranking"
    assert frame.consensus_spearman.min() > .98
    assert frame.hard_overlap.min() > .95
    assert frame.mean_estimator_gap_above_null.min() > .5
    assert set(patches.discovery_group) == {"Easy", "Medium", "Hard"}
    assert patches.groupby(["ecg_id", "discovery_group"]).size().eq(40).all()
    assert patches.token_index.max() == 119 and patches.patch_number.max() == 120
    assert all(patches[f"{a}_{b}_count"].eq(24).all() for a in ESTIMATORS for b in BANKS)


def test_random_context_errors_do_not_manufacture_feasibility():
    frame, _ = synthetic(kind="null")
    summary = summarize_ecgs(frame, bootstrap_repeats=100)
    assert summary["status"] == "inconclusive"
    assert abs(summary["metrics"]["consensus_spearman"]["mean"]) < .2
    assert summary["metrics"]["hard_overlap"]["mean"] < .5
    assert summary["whole_mask_inclusion_status"] == "inconclusive"


def test_energy_only_signal_is_exposed_not_called_contextual_improvement():
    frame, _ = synthetic(kind="energy")
    assert frame.energy_spearman.min() > .99
    for a in ESTIMATORS:
        assert frame[f"{a}_energy_adjusted_spearman"].isna().all()
        assert frame[f"{a}_energy_only_gap"].min() > .5


def test_constant_error_does_not_become_stable_rank_via_roundoff():
    assert np.isnan(rank_correlation(np.ones(20), np.arange(20)))
    assert np.isnan(rank_correlation(np.ones(20)+np.arange(20)*1e-16, np.arange(20)))


def test_patient_not_patch_is_bootstrap_unit_and_smoke_cannot_pass():
    frame, _ = synthetic(n=3)
    frame["patient_id"] = [1, 1, 2]
    summary = summarize_ecgs(frame, bootstrap_repeats=100, mode="smoke")
    assert summary["patients"] == 2
    assert summary["metrics"]["consensus_spearman"]["valid_patients"] == 2
    assert summary["status"] == "smoke_only"
    assert summary["whole_mask_inclusion_status"] == "smoke_only"


@pytest.fixture
def source_fixture(tmp_path):
    artifact, source = tmp_path / "data", tmp_path / "source"
    artifact.mkdir()
    source.mkdir()
    manifest = dict(lead_order=list(LEADS), config=dict(target_sampling_rate=250))
    manifest_path = artifact / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    metadata = meta(3)
    metadata["strat_fold"] = 1
    metadata["NORM"] = [1, 0, 1]
    metadata.to_csv(artifact / "metadata_train.csv", index=False)
    np.save(artifact / "signals_train.npy", np.random.default_rng(1).normal(size=(3, 12, 2500)).astype(np.float32))
    # No test/validation files exist: touching them would fail the integration test.
    selection = metadata.drop(columns="NORM").copy()
    selection["role"] = ["fit", "score", "score"]
    selection["source_split"] = "train"
    selection.to_csv(source / "ecg_selection.csv", index=False)
    profile = dict(cnn_width=4, transformer_width=16, transformer_depth=1)
    config = dict(mask_ratio=.5, realized_mask_ratio=.5, patch_samples=50, ridge_alpha=.001,
                  fixed_train_lead_std=[1.]*12, profile=profile,
                  source_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest())
    (source / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
    lags = [-100, -50, -1, 0, 1, 50, 100]
    np.savez(source / "ridge_weights.npz", weights=np.zeros((12, 169)), scale=np.ones(12), lags=lags)
    for name, model in (("cnn", SmallCNN(width=4)), ("transformer", SmallTransformer(width=16, depth=1))):
        torch.save(dict(estimator=name, model=model.state_dict(), profile=profile,
                        fixed_train_std=np.ones(12)), source / f"{name}.pt")
    return artifact, source


def test_inference_smoke_end_to_end_no_train_no_test_and_no_source_mutation(source_fixture, tmp_path):
    artifact, source = source_fixture
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
    args = parser().parse_args(["--source-run", str(source), "--artifact-dir", str(artifact),
        "--output-root", str(tmp_path / "output"), "--mode", "smoke", "--score-ecgs", "1",
        "--pairs-per-bank", "2", "--batch-size", "2", "--cpu-threads", "2", "--bootstrap-repeats", "20",
        "--shuffle-repeats", "2", "--save-patch-cache"])
    output = run(args)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["status"] == "smoke_only" and summary["training_performed"] is False
    assert summary["test_opened"] is False
    patches = pd.read_csv(output / "patch_scores.csv")
    assert len(patches) == 600
    assert patches.ridge_confirmation_count.eq(2).all()
    assert len(pd.read_csv(output / "context_mask_bank.csv")) == 8
    assert list(output.glob("patch_errors_ecg*.npz"))
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}


def test_source_rejects_patient_leakage_and_changed_preprocessing(source_fixture):
    artifact, source = source_fixture
    selection_path = source / "ecg_selection.csv"
    selection = pd.read_csv(selection_path)
    selection.loc[1, "patient_id"] = selection.loc[0, "patient_id"]
    selection.to_csv(selection_path, index=False)
    with pytest.raises(ValueError, match="leakage"):
        load_source(source, artifact, 1)
    (artifact / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_source(source, artifact, 1)


def test_source_rejects_non_50_percent_training_condition(source_fixture):
    artifact, source = source_fixture
    path = source / "run_config.json"
    config = json.loads(path.read_text())
    config["mask_ratio"] = .75
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="50%"):
        load_source(source, artifact, 1)


def test_few_valid_rank_patients_cannot_pass_using_many_undefined_records():
    frame, _ = synthetic()
    frame.loc[1:, "consensus_spearman"] = np.nan
    summary = summarize_ecgs(frame, bootstrap_repeats=20)
    assert summary["gates"]["enough_valid_rank_patients"] is False
    assert summary["status"] == "inconclusive"
