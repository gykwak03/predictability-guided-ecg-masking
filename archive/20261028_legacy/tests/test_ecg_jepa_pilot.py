"""Scientific contracts: official equivalence, hidden-value isolation, EMA, pairing."""
from pathlib import Path
import json
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"experiments"))
from ecg_jepa_adapter import ECGJEPAAdapter, sample_masks
from run_ecg_jepa_pilot import pilot_plan
from compare_ecg_jepa_pilot import compare


@pytest.fixture
def model():
    repo = ROOT.parent/"external/ECG_JEPA"
    if not repo.exists():
        pytest.skip("Pinned official ECG_JEPA checkout required")
    torch.set_num_threads(2)
    torch.manual_seed(42)
    return ECGJEPAAdapter(repo, smoke=True)


def test_masks_exact_and_topology():
    for n in (20, 50, 100):
        for r in (.5, .6, .7):
            independent = sample_masks(3, 12, n, r, 42)
            aligned = sample_masks(3, 12, n, r, 42, "time-aligned")
            assert (independent.sum(-1) == n*r).all()
            assert np.array_equal(aligned[:, :1], aligned[:, 7:8])
            assert not np.array_equal(independent[:, 0], independent[:, 1])
            assert np.array_equal(independent, sample_masks(3, 12, n, r, 42))
            assert not np.array_equal(independent, sample_masks(3, 12, n, r, 43))
    with pytest.raises(ValueError, match="integer"):
        sample_masks(2, 12, 25, .5, 42)


def test_hidden_values_cannot_affect_student_prediction(model):
    model.eval()
    x = torch.randn(2, 12, 2500)
    mask = torch.from_numpy(sample_masks(2, 12, 50, .5, 7))
    changed = x.clone().reshape(2, 12, 50, 50)
    changed[mask] = 100*torch.randn_like(changed[mask])
    with torch.no_grad():
        a = model.predict_masked(x, mask)
        b = model.predict_masked(changed.reshape_as(x), mask)
    torch.testing.assert_close(a, b, atol=1e-6, rtol=0)


def test_matches_official_forward_for_shared_time_mask(model):
    model.eval()
    x = torch.randn(2, 12, 2500)
    time_mask = torch.zeros(50, dtype=torch.bool)
    time_mask[::2] = True
    model.core.target_encoder._make_rand_mask = lambda scale: time_mask
    mask = time_mask[None, None].expand(2, 12, -1)
    with torch.no_grad():
        expected = model.core(x)
        actual = model.ssl_loss(x, mask)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)


def test_teacher_frozen_and_ema(model):
    model.train()
    assert not model.core.target_encoder.training
    old = model.core.target_encoder.W_P.weight.detach().clone()
    x = torch.randn(2, 12, 2500)
    mask = torch.from_numpy(sample_masks(2, 12, 50, .5, 5))
    loss = model.ssl_loss(x, mask)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None for p in model.core.target_encoder.parameters())
    assert model.core.encoder.W_P.weight.grad is not None
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001)
    opt.step()
    assert torch.equal(old, model.core.target_encoder.W_P.weight)
    new = model.core.encoder.W_P.weight.detach().clone()
    model.update_teacher(.9)
    torch.testing.assert_close(model.core.target_encoder.W_P.weight, .9*old+.1*new)


def test_patch_sweep_keeps_shared_initial_backbone(model):
    digest = model.initial_backbone_digest()
    for patch in (25, 125):
        torch.manual_seed(42)
        other = ECGJEPAAdapter(ROOT.parent/"external/ECG_JEPA", patch, smoke=True)
        assert other.initial_backbone_digest() == digest
        mask = torch.from_numpy(sample_masks(1, 12, 2500//patch, .5, 2))
        assert torch.isfinite(other.ssl_loss(torch.randn(1, 12, 2500), mask))
        assert other.features(torch.randn(2, 12, 2500)).shape == (2, 32)
        with pytest.raises(ValueError, match="checkpoint requires"):
            other.load_released_encoder("not-read.pth")


def test_eight_lead_input_order_and_native_equivalence(model):
    other = ECGJEPAAdapter(ROOT.parent/"external/ECG_JEPA", leads=8, smoke=True).eval()
    x = torch.randn(2, 12, 2500)
    aligned = torch.from_numpy(sample_masks(1, 8, 50, .6, 42, "time-aligned"))
    time_mask = aligned[0, 0]
    other.core.target_encoder._make_rand_mask = lambda scale: time_mask
    with torch.no_grad():
        expected = other.core(x[:, [0, 1, 6, 7, 8, 9, 10, 11]])
        actual = other.ssl_loss(x, aligned.expand(2, -1, -1))
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)


def test_plan_primary_is_50_percent():
    plan = pilot_plan()
    primary = [r for r in plan["conditions"] if r["stage"] == "primary"]
    assert len(primary) == 3
    assert {r["patch_seconds"] for r in primary} == {.1, .2, .5}
    assert all(r["mask_ratio"] == .5 for r in primary)


@pytest.fixture
def fake_runs(tmp_path):
    # Synthetic comparison metadata only, never reported as experimental data.
    folders = []
    for patch in (25, 50):
        for seed in (42, 43, 44):
            c = dict(model="sehun_ecg_jepa", mode="pilot", source={}, source_manifest_sha256="fixture",
                architecture={}, initialization="scratch", leads=12, lead_indices=list(range(12)),
                steps=1000, effective_batch=128, microbatch=4, accum_steps=32,
                train_ecg_ids=[1, 2], validation_ecg_ids=[3], subset_seed=42, optimizer={}, teacher={},
                probe={}, precision="fp32", gpu={}, seed=seed, patch_samples=patch, patch_seconds=patch/250,
                mask_ratio=.5, topology="lead-time", test_opened=False, initial_backbone_sha256=str(seed))
            s = dict(status="completed", eligible_for_condition_selection=True, test_opened=False,
                trained_probe=dict(all_five_classes_supported=True, macro_auroc=.9, macro_auprc_ap=.8),
                initial_probe=dict(macro_auroc=.5), pretrain_seconds=1., gpu_peak_allocated_bytes=100,
                batch_sequence_sha256=str(seed))
            folder = tmp_path/f"P{patch}_s{seed}"
            folder.mkdir()
            (folder/"run_config.json").write_text(json.dumps(c))
            (folder/"summary.json").write_text(json.dumps(s))
            folders.append(folder)
    return folders


def test_comparer_requires_paired_seeds_and_budget(fake_runs, tmp_path):
    result = compare(fake_runs, tmp_path/"ok")
    assert len(result) == 2 and (result.seeds == 3).all()
    with pytest.raises(ValueError, match="same set"):
        compare(fake_runs[:-1], tmp_path/"no")
    p = fake_runs[-1]/"run_config.json"
    c = json.loads(p.read_text())
    c["steps"] += 1
    p.write_text(json.dumps(c))
    with pytest.raises(ValueError, match="Unmatched field: steps"):
        compare(fake_runs, tmp_path/"no")


def test_comparer_rejects_continuation_and_mismatched_batches(fake_runs, tmp_path):
    p = fake_runs[-1]/"run_config.json"
    c = json.loads(p.read_text())
    c["initialization"] = "continued_pretraining"
    p.write_text(json.dumps(c))
    with pytest.raises(ValueError, match="continued-pretraining"):
        compare(fake_runs, tmp_path/"no")
    c["initialization"] = "scratch"
    p.write_text(json.dumps(c))
    p = fake_runs[-1]/"summary.json"
    s = json.loads(p.read_text())
    s["batch_sequence_sha256"] = "different"
    p.write_text(json.dumps(s))
    with pytest.raises(ValueError, match="ECG batch"):
        compare(fake_runs, tmp_path/"no")
