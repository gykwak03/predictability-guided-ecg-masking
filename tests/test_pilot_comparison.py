"""Checks for fair seed aggregation and read-only checkpoint replay."""
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "experiments"))
    return importlib.import_module("pilot_comparison"), importlib.import_module("inspect_masking_pilot")


def saved_run(root, seed, auroc, *, eval_seed=42):
    directory = root / f"seed{seed}"
    directory.mkdir()
    config = dict(mode="pilot", seed=seed, evaluation_mask_seed=eval_seed,
                  profile=dict(steps=1000), train_ecg_ids=[1, 2], validation_ecg_ids=[3, 4],
                  source_manifest_sha256="same-data")
    (directory / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
    pd.DataFrame([dict(seed=seed, steps=1000, patch_samples=50, mask_ratio=0.75,
                       macro_auroc=auroc, random_encoder_macro_auroc=0.7,
                       auroc_gain_vs_random=auroc-0.7, validation_skill_vs_zero=0.02)]).to_csv(
                           directory / "results.csv", index=False)
    return directory


def test_paired_seed_mean_std_and_missing_seed(modules, tmp_path):
    comparison, _ = modules
    runs = [saved_run(tmp_path, 42, 0.78), saved_run(tmp_path, 43, 0.80)]
    output = tmp_path / "summary"
    summary = comparison.summarize_runs(runs, output, expected_seeds=[42, 43, 44])
    row = summary.iloc[0]
    assert row.seed_count == 2 and not row.complete_requested_seeds
    assert row.macro_auroc_mean == pytest.approx(0.79)
    assert row.macro_auroc_std == pytest.approx(np.sqrt(0.0002))
    assert row.auroc_gain_vs_random_mean == pytest.approx(0.09)
    assert (output / "condition_comparison.png").stat().st_size > 1000


def test_incompatible_masks_and_duplicate_seeds_rejected(modules, tmp_path):
    comparison, _ = modules
    first = saved_run(tmp_path, 42, 0.78)
    other_masks = saved_run(tmp_path, 43, 0.80, eval_seed=43)
    with pytest.raises(ValueError, match="different subsets, evaluation masks"):
        comparison.summarize_runs([first, other_masks], tmp_path / "bad")
    with pytest.raises(ValueError, match="Duplicate"):
        comparison.summarize_runs([first, first], tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_different_initialization_policies_cannot_be_pooled(modules, tmp_path):
    comparison, _ = modules
    runs = [saved_run(tmp_path, 42, 0.78), saved_run(tmp_path, 43, 0.80)]
    for run, policy in zip(runs, ["legacy", "independent"]):
        path = run / "run_config.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        config["profile"]["initialization"] = policy
        path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="different subsets, evaluation masks"):
        comparison.summarize_runs(runs, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_saved_ecg_id_order_restored_and_missing_ids_rejected(modules):
    _, inspector = modules
    frame = pd.DataFrame(dict(ecg_id=[50, 20, 30]))
    np.testing.assert_array_equal(inspector.indices_for_ids(frame, [30, 50]), [2, 0])
    for ids in ([10], [20, 20], []):
        with pytest.raises(ValueError):
            inspector.indices_for_ids(frame, ids)


def test_checkpoint_replay_matches_saved_metric_without_retraining(modules, tmp_path):
    _, inspector = modules
    ns = inspector.run_notebook(mode="smoke", train=True, steps=2,
                                output_root=tmp_path / "original", quiet=True)
    run = ns["RUN_DIR"]
    original_bytes = {path.relative_to(run): path.read_bytes()
                      for path in run.rglob("*") if path.is_file()}
    report = inspector.inspect_run(run, output_dir=tmp_path / "inspection", examples=1, probe=True)
    before, after = pd.read_csv(run / "results.csv").iloc[0], pd.read_csv(report / "results.csv").iloc[0]
    assert after.validation_masked_mse == pytest.approx(before.validation_masked_mse, abs=2e-7)
    assert after.macro_auroc == pytest.approx(before.macro_auroc, abs=1e-10)
    assert after.auroc_gain_vs_random == pytest.approx(before.auroc_gain_vs_random, abs=1e-10)
    assert after.random_initialization_source == "archived"
    samples = np.load(report / "p125_r75/final_reconstruction_samples.npz")
    assert samples["target"].shape == samples["prediction"].shape == samples["sample_mask"].shape
    # The report must preserve all pre-existing files byte-for-byte.
    assert original_bytes == {path.relative_to(run): path.read_bytes()
                              for path in run.rglob("*") if path.is_file()}


def test_runner_can_fix_subsets_and_masks_across_model_seeds(modules):
    _, inspector = modules
    states = [inspector.run_notebook(mode="smoke", seed=seed, subset_seed=42,
                                    eval_mask_seed=100, patch_samples=50, mask_ratios=[0.75, 0.90],
                                    quiet=True) for seed in (42, 43)]
    np.testing.assert_array_equal(states[0]["TRAIN_IDX"], states[1]["TRAIN_IDX"])
    np.testing.assert_array_equal(states[0]["VAL_IDX"], states[1]["VAL_IDX"])
    np.testing.assert_array_equal(states[0]["TRAIN_DIAGNOSTIC_IDX"], states[1]["TRAIN_DIAGNOSTIC_IDX"])
    assert states[0]["EVAL_SEED"] == states[1]["EVAL_SEED"] == 100
    assert states[0]["SEED"] != states[1]["SEED"]
    assert states[0]["GRID"] == states[1]["GRID"] == [(50, 0.75), (50, 0.9)]
