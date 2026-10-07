"""Integration tests require local PTB-XL artifacts and the notebook dependencies."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def pilot():
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    path = ROOT / "experiments/run_masking_pilot.py"
    spec = importlib.util.spec_from_file_location("pilot_runner", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    try:
        return runner.run_notebook(quiet=True)
    except FileNotFoundError:
        pytest.skip("PTB-XL artifacts are not available on this host")


def test_invalid_patch_and_masks(pilot):
    x = np.zeros((2, 12, 2500), dtype=np.float32)
    for bad in [0, -1, 64, 1.5]:
        with pytest.raises(ValueError):
            pilot["patchify_np"](x, bad)
    for ratio in [0, 1, -0.1, 1.1]:
        with pytest.raises(ValueError):
            pilot["random_mask_np"](2, 240, ratio, 42)


@pytest.mark.parametrize("patch_length", [50, 125, 250])
@pytest.mark.parametrize("ratio", [0.5, 0.75, 0.9])
def test_all_conditions_forward_and_hidden_isolation(pilot, patch_length, ratio):
    torch = pilot["torch"]
    pilot["seed_everything"](42)
    model = pilot["PlainECGMAE"](patch_length, pilot["PROFILE"]).cpu().eval()
    x = torch.randn(2, 12, 2500)
    mask = torch.from_numpy(pilot["random_mask_np"](2, model.n, ratio, 42))
    pred, target = model(x, mask)
    assert pred.shape == target.shape == (2, model.n, patch_length)
    loss = pilot["masked_mse"](pred, target, mask)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    with torch.no_grad():
        changed = x.clone()
        changed.reshape(2, model.n, patch_length)[mask] += 20
        alternative, _ = model(changed, mask)
        assert torch.allclose(pred, alternative, rtol=1e-5, atol=1e-5)
        visible_altered = pred.detach().clone()
        visible_altered[~mask] += 100
        assert torch.allclose(pilot["masked_mse"](visible_altered, target, mask), loss.detach())


def test_degenerate_label_metrics_are_explicit(pilot):
    y = np.zeros((4, 5), dtype=np.uint8)
    y[:, 0] = [0, 1, 0, 1]
    probabilities = np.full((4, 5), 0.2)
    metrics, detail = pilot["classification_metrics"](y, probabilities)
    assert metrics["valid_metric_classes"] == 1
    assert metrics["macro_auroc"] == 0.5
    assert detail.auroc.isna().sum() == 4


def test_actual_cpu_training_and_linear_probe(pilot):
    # 8 real-signal MAE updates + frozen linear probe; no artifacts overwritten.
    model, result, history, detail, probabilities, state = pilot["train_condition"](125, 0.75)
    assert len(history) == pilot["PROFILE"]["steps"] == 8
    assert np.isfinite(history.masked_mse).all()
    assert np.isfinite(result["validation_masked_mse"])
    assert np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all()
    assert probabilities.shape == (len(pilot["PROBE_VAL_IDX"]), 5)
    assert not any(parameter.requires_grad for parameter in model.parameters())
    assert state["feature_mean"].shape == (1, pilot["PROFILE"]["enc_dim"])
    assert len(detail) == 5
