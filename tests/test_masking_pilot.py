"""Integration tests require local PTB-XL artifacts and the notebook dependencies."""
import importlib.util
import json
import os
from pathlib import Path
import random

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def plain_model_definitions():
    """Load the real model without ECG files, to verify historical compatibility."""
    torch = pytest.importorskip("torch")
    notebook = json.loads((ROOT / "experiments/Masking_pilot.ipynb").read_text(encoding="utf-8"))
    source = next("".join(cell["source"]) for cell in notebook["cells"]
                  if "class PlainECGMAE" in "".join(cell["source"]))
    namespace = dict(TRAINING_AVAILABLE=True, torch=torch, nn=torch.nn,
                     np=np, random=random, N_LEADS=12, SIGNAL_LENGTH=2500)
    exec(compile(source, "notebook_model_definitions", "exec"), namespace)
    return namespace


def test_initialization_preserves_legacy_and_only_changes_transformer_blocks(plain_model_definitions):
    ns, torch = plain_model_definitions, plain_model_definitions["torch"]
    cfg = dict(enc_dim=32, enc_depth=2, enc_heads=4, dec_dim=16, dec_depth=2, dec_heads=4)
    ns["seed_everything"](42)
    historical = ns["PlainECGMAE"](50, cfg)
    ns["seed_everything"](42)
    explicit_legacy = ns["PlainECGMAE"](50, dict(cfg, initialization="legacy"))
    ns["seed_everything"](42)
    independent = ns["PlainECGMAE"](50, dict(cfg, initialization="independent"))
    ns["seed_everything"](42)
    repeated = ns["PlainECGMAE"](50, dict(cfg, initialization="independent"))
    assert sum(p.numel() for p in historical.parameters()) == sum(p.numel() for p in independent.parameters())
    for key, value in historical.state_dict().items():
        assert torch.equal(value, explicit_legacy.state_dict()[key])
        assert value.shape == independent.state_dict()[key].shape
        assert torch.equal(independent.state_dict()[key], repeated.state_dict()[key])
        if not key.startswith(("encoder.layers.", "decoder.layers.")):
            assert torch.equal(value, independent.state_dict()[key]), key
    for name in ("encoder", "decoder"):
        old_layers, new_layers = getattr(historical, name).layers, getattr(independent, name).layers
        assert torch.equal(old_layers[0].self_attn.in_proj_weight, old_layers[1].self_attn.in_proj_weight)
        assert not torch.equal(new_layers[0].self_attn.in_proj_weight, new_layers[1].self_attn.in_proj_weight)
        assert old_layers[0].self_attn.in_proj_weight.data_ptr() != old_layers[1].self_attn.in_proj_weight.data_ptr()
    with pytest.raises(ValueError, match="initialization"):
        ns["PlainECGMAE"](50, dict(cfg, initialization="unknown"))


@pytest.fixture
def curve_plotter():
    """Load the notebook's plotting implementation without data or PyTorch."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    notebook = json.loads((ROOT / "experiments/Masking_pilot.ipynb").read_text(encoding="utf-8"))
    source = next("".join(cell["source"]) for cell in notebook["cells"]
                  if cell["cell_type"] == "code"
                  and "def save_learning_curves(" in "".join(cell["source"]))
    namespace = dict(TRAINING_AVAILABLE=False, np=np, pd=pd, plt=plt, Path=Path)
    exec(compile(source, "notebook_learning_curves", "exec"), namespace)
    return namespace


@pytest.mark.parametrize("updates", [1, 8, 100])
def test_learning_curve_png_values_and_no_state_changes(curve_plotter, tmp_path, monkeypatch, updates):
    from matplotlib.figure import Figure
    pd, plt = curve_plotter["pd"], curve_plotter["plt"]
    history = pd.DataFrame(dict(step=np.arange(1, updates+1),
                                masked_mse=np.linspace(1.3, 0.9, updates),
                                grad_norm=np.linspace(2.0, 0.5, updates),
                                lr=np.linspace(3e-4, 3e-5, updates)))
    original = history.copy(deep=True)
    numpy_state, python_state = np.random.get_state(), random.getstate()
    figures_before = plt.get_fignums()
    captured = []
    savefig = Figure.savefig

    def inspect_savefig(fig, *args, **kwargs):
        captured.append(fig)
        return savefig(fig, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", inspect_savefig)
    destination = curve_plotter["save_learning_curves"](history, tmp_path / "p50_r75", 50, 0.75)
    assert destination.name == "learning_curves.png"
    assert destination.is_file() and destination.stat().st_size > 1000
    assert plt.imread(destination).shape[:2] == (1440, 1600)
    axes = captured[0].axes
    assert len(axes) == 3
    # The actual plot uses the raw CSV values; smoothing is trailing, not centered.
    np.testing.assert_array_equal(axes[0].lines[0].get_ydata(), history.masked_mse)
    np.testing.assert_allclose(axes[0].lines[1].get_ydata(),
                               history.masked_mse.rolling(min(50, updates), min_periods=1).mean())
    np.testing.assert_array_equal(axes[1].lines[0].get_ydata(), history.grad_norm)
    np.testing.assert_array_equal(axes[1].lines[1].get_ydata(), [1.0, 1.0])
    np.testing.assert_array_equal(axes[2].lines[0].get_ydata(), history.lr)
    assert "not fixed-mask" in axes[0].get_title()
    pd.testing.assert_frame_equal(history, original)
    assert plt.get_fignums() == figures_before
    current_state = np.random.get_state()
    assert current_state[0] == numpy_state[0] and current_state[2:] == numpy_state[2:]
    np.testing.assert_array_equal(current_state[1], numpy_state[1])
    assert random.getstate() == python_state


@pytest.mark.parametrize("invalid", ["empty", "nonfinite", "window"])
def test_learning_curve_rejects_invalid_history(curve_plotter, tmp_path, invalid):
    history = curve_plotter["pd"].DataFrame(dict(step=[1], masked_mse=[1.0], grad_norm=[0.5], lr=[1e-4]))
    if invalid == "empty":
        history = history.iloc[:0]
    elif invalid == "nonfinite":
        history.loc[0, "masked_mse"] = np.nan
    before = curve_plotter["plt"].get_fignums()
    with pytest.raises(ValueError):
        curve_plotter["save_learning_curves"](history, tmp_path, 50, 0.75,
                                               window=0 if invalid == "window" else 50)
    assert not (tmp_path / "learning_curves.png").exists()
    assert curve_plotter["plt"].get_fignums() == before


def test_learning_curve_closes_figure_when_save_fails(curve_plotter, tmp_path, monkeypatch):
    from matplotlib.figure import Figure
    history = curve_plotter["pd"].DataFrame(dict(step=[1], masked_mse=[1.0], grad_norm=[0.5], lr=[1e-4]))

    def failed_save(*args, **kwargs):
        raise OSError("Test save failure")

    monkeypatch.setattr(Figure, "savefig", failed_save)
    before = curve_plotter["plt"].get_fignums()
    with pytest.raises(OSError, match="Test save failure"):
        curve_plotter["save_learning_curves"](history, tmp_path, 50, 0.75)
    assert curve_plotter["plt"].get_fignums() == before


@pytest.mark.parametrize("leak_hidden", [False, True])
def test_notebook_isolation_check_uses_same_execution_mode(leak_hidden):
    """A train/eval numerical difference must not masquerade as hidden-input leakage."""
    torch = pytest.importorskip("torch")
    calls = []

    class ModeSensitiveModel(torch.nn.Module):
        def __init__(self, patch_length, cfg):
            super().__init__()
            self.p = patch_length
            self.n = 12 * 2500 // patch_length
            self.weight = torch.nn.Parameter(torch.tensor(0.1))

        def forward(self, x, mask):
            calls.append((self.training, torch.is_grad_enabled()))
            target = x.reshape(len(x), self.n, self.p)
            # Deliberately larger than allclose tolerance: only execution mode differs.
            prediction = self.weight + (0.01 if self.training else 0.0)
            if leak_hidden:
                prediction = prediction + target[mask].mean()
            return prediction.expand_as(target), target

    def make_mask(batch_size, n_tokens, ratio, seed):
        mask = np.zeros((batch_size, n_tokens), dtype=bool)
        mask[:, :round(n_tokens * ratio)] = True
        return mask

    def loss_fn(pred, target, mask):
        return ((pred-target).square().mean(dim=-1) * mask).sum() / mask.sum()

    # Exercise the real notebook check, not a duplicated implementation of it.
    notebook = json.loads((ROOT / "experiments/Masking_pilot.ipynb").read_text(encoding="utf-8"))
    source = next("".join(cell["source"]) for cell in notebook["cells"]
                  if cell["cell_type"] == "code"
                  and "DEVICE = None" in "".join(cell["source"]))
    namespace = dict(TRAINING_AVAILABLE=True, RUN_TRAINING=False, torch=torch,
                     os=os, np=np, SEED=42, PROFILE={}, seed_everything=lambda seed: None,
                     PlainECGMAE=ModeSensitiveModel, random_mask_np=make_mask,
                     masked_mse=loss_fn, TRAIN_IDX=np.arange(2),
                     arrays={"train": {"x": np.zeros((2, 12, 2500), dtype=np.float32)}})
    if leak_hidden:
        with pytest.raises(AssertionError, match="Hidden-input isolation failed"):
            exec(compile(source, "notebook_model_check", "exec"), namespace)
    else:
        exec(compile(source, "notebook_model_check", "exec"), namespace)
    assert calls == [(True, True), (False, False), (False, False)]


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


def test_notebook_execution_automatically_saves_curves(pilot, tmp_path):
    """Run the real save/execution cell so the notebook and CLI share the hook."""
    notebook = json.loads((ROOT / "experiments/Masking_pilot.ipynb").read_text(encoding="utf-8"))
    source = next("".join(cell["source"]) for cell in notebook["cells"]
                  if cell["cell_type"] == "code"
                  and "RESULTS = pd.DataFrame()" in "".join(cell["source"])
                  and "if RUN_TRAINING:" in "".join(cell["source"]))
    namespace = dict(pilot, RUN_TRAINING=True, OUTPUT_ROOT=tmp_path)
    exec(compile(source, "notebook_save_artifacts", "exec"), namespace)
    run_dir = namespace["RUN_DIR"]
    assert run_dir.parent == tmp_path
    assert len(namespace["RESULTS"]) == 1
    condition = run_dir / "p125_r75"
    assert (condition / "learning_curves.png").is_file()
    assert (condition / "checkpoint.pt").is_file()
    assert (run_dir / "results.csv").is_file()
    history = pilot["pd"].read_csv(condition / "pretrain_history.csv")
    assert list(history.columns[:4]) == ["step", "masked_mse", "grad_norm", "lr"]
    assert {"zero_masked_mse", "skill_vs_zero", "pred_rms_masked", "lr_used", "clipped"} <= set(history.columns)
    assert len(history) == pilot["PROFILE"]["steps"]
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    assert config["history_logging"]["moving_average_window"] == 50
    assert config["history_logging"]["validation"] == "periodic fixed diagnostic subset + final 3-repeat metric"
    for name in ["evaluation_history.csv", "diagnostic_curves.png", "gradient_updates.csv",
                 "gradient_updates.png", "initial_checkpoint.pt", "initialization.json",
                 "cross_lead_baseline.npz", "feature_statistics.json", "diagnostic_summary.json",
                 "initial_reconstruction_samples.npz", "final_reconstruction_samples.npz",
                 "random_encoder_probe_metrics.json", "random_encoder_probe.pt"]:
        assert (condition / name).is_file(), name
    evaluations = pilot["pd"].read_csv(condition / "evaluation_history.csv")
    assert set(evaluations.step) == {0, 8}
    assert set(evaluations.split) == {"train", "validation"}
    samples = np.load(condition / "final_reconstruction_samples.npz", allow_pickle=False)
    assert samples["target"].shape == samples["prediction"].shape == samples["sample_mask"].shape == (3, 12, 2500)
    assert samples["patch_mask"].shape == (3, 240)
    assert "not mV" in str(samples["units"])
    np.testing.assert_array_equal(samples["composite"][~samples["sample_mask"]], samples["target"][~samples["sample_mask"]])
    torch = pilot["torch"]
    checkpoint = torch.load(condition / "checkpoint.pt", map_location="cpu", weights_only=True)
    restored = pilot["PlainECGMAE"](checkpoint["patch_length"], checkpoint["profile"]).cpu().eval()
    restored.load_state_dict(checkpoint["model"])
    restored.ratio = checkpoint["mask_ratio"]
    with torch.no_grad():
        pred, _ = restored(torch.from_numpy(samples["target"]), torch.from_numpy(samples["patch_mask"]))
    np.testing.assert_allclose(pred.numpy().reshape(samples["target"].shape), samples["prediction"], atol=2e-6, rtol=1e-5)
    summary = json.loads((condition / "diagnostic_summary.json").read_text(encoding="utf-8"))
    assert summary["final_validation"]["masked_samples"] > 0
    assert np.isfinite(summary["final_validation"]["linear_masked_mse"])


@pytest.fixture(scope="module")
def diagnostic_module():
    pytest.importorskip("torch")
    path = ROOT / "experiments/masking_diagnostics.py"
    spec = importlib.util.spec_from_file_location("diagnostic_tests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hidden_statistics_zero_perfect_and_visible_exclusion(diagnostic_module):
    target = np.array([[[1., -1., 2., -2.]]])
    mask = np.array([[[True, True, False, False]]])
    prediction = target.copy()
    prediction[~mask] = 1e6
    perfect = diagnostic_module.masked_statistics(prediction, target, mask)
    assert perfect["masked_mse"] == 0 and perfect["skill_vs_zero"] == 1
    assert perfect["rms_ratio"] == perfect["masked_correlation"] == 1
    zero = diagnostic_module.masked_statistics(np.zeros_like(target), target, mask)
    assert zero["masked_mse"] == zero["zero_masked_mse"] == 1
    assert zero["skill_vs_zero"] == zero["rms_ratio"] == 0
    assert np.isnan(zero["masked_correlation"])
    reversed_wave = diagnostic_module.masked_statistics(-target, target, mask)
    assert reversed_wave["masked_mse"] == 4 and reversed_wave["masked_correlation"] == -1


def test_diagnostic_masks_exactly_match_previous_validation(pilot):
    for p in (50, 125, 250):
        for ratio in (0.5, 0.75, 0.9):
            ids = np.array([1, 102, 21999])
            mask, seeds = pilot["diagnostics"].fixed_mask(ids, 12*2500//p, ratio, 42, repeat=2)
            for i, ecg_id in enumerate(ids):
                seed = 42+100000+2*1000000+ecg_id
                np.testing.assert_array_equal(mask[i], pilot["random_mask_np"](1, mask.shape[1], ratio, seed)[0])
                assert seeds[i] == seed


def test_cross_lead_baseline_never_reads_hidden_inputs_or_validation_fit(diagnostic_module):
    rng = np.random.default_rng(1)
    base = rng.normal(size=(4, 1, 20)).astype(np.float32)
    train = np.concatenate([base, base*2], axis=1)
    indices = np.array([0, 1])
    baseline = diagnostic_module.fit_cross_lead_baseline(train, indices, ridge=1e-5)
    changed_train = train.copy()
    changed_train[2:] = 1e6  # excluded records must not affect the fitted baseline
    alternative = diagnostic_module.fit_cross_lead_baseline(changed_train, indices, ridge=1e-5)
    np.testing.assert_array_equal(baseline["covariance"], alternative["covariance"])
    mask = np.array([[False, False, True, True]])  # p=10; lead 1 visible, lead 2 hidden
    prediction = diagnostic_module.cross_lead_predict(train[:1], mask, 10, baseline)
    changed = train[:1].copy()
    changed[:, 1] += 10000
    np.testing.assert_array_equal(prediction, diagnostic_module.cross_lead_predict(changed, mask, 10, baseline))
    np.testing.assert_allclose(prediction[:, 1], train[:1, 1], atol=1e-4)
    all_hidden = np.ones_like(mask)
    fallback = diagnostic_module.cross_lead_predict(train[:1], all_hidden, 10, baseline)
    np.testing.assert_allclose(fallback, np.broadcast_to(baseline["mean"][None, :, None], fallback.shape), atol=1e-7)


def test_evaluation_restores_mode_and_matches_original_metric(pilot):
    torch = pilot["torch"]
    model = pilot["PlainECGMAE"](125, pilot["PROFILE"]).to(pilot["DEVICE"])
    model.ratio = 0.75
    model.train()
    indices = pilot["VAL_IDX"][:5]
    state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    bundle = pilot["diagnostics"].evaluate_reconstruction(
        model, pilot["arrays"]["validation"]["x"], pilot["frames"]["validation"].ecg_id.to_numpy(),
        indices, 2, 42, repeats=3, examples=2, lead_order=pilot["LEADS"])
    assert model.training
    assert all(torch.equal(state[k], v) for k, v in model.state_dict().items())
    original = pilot["reconstruction_eval"](model, indices, 2)
    assert bundle["metrics"]["masked_mse"] == pytest.approx(original, abs=2e-7)
    samples = bundle["samples"]
    assert samples["target"].shape[0] == 2
    assert bundle["metrics"]["masked_samples"] == 5*3*round(model.n*model.ratio)*model.p
    per_ecg = bundle["per_ecg"]
    assert len(per_ecg) == 15 and len(bundle["per_lead"]) == 12
    weighted = (per_ecg.masked_mse*per_ecg.masked_samples).sum()/per_ecg.masked_samples.sum()
    assert weighted == pytest.approx(bundle["metrics"]["masked_mse"], abs=1e-10)


def test_preserve_rng_restores_torch_numpy_python_even_after_exception(diagnostic_module):
    torch = diagnostic_module.torch
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    with pytest.raises(RuntimeError, match="test interruption"):
        with diagnostic_module.preserve_rng(torch.device("cpu")):
            random.seed(100)
            np.random.seed(100)
            torch.manual_seed(100)
            torch.randn(10)
            raise RuntimeError("test interruption")
    assert random.getstate() == python_state
    restored = np.random.get_state()
    np.testing.assert_array_equal(restored[1], numpy_state[1])
    assert restored[0] == numpy_state[0] and restored[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_gradient_snapshot_measures_actual_update_and_handles_missing_gradient(diagnostic_module):
    torch = diagnostic_module.torch
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    model[0].weight.grad = torch.ones_like(model[0].weight)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    rows, before = diagnostic_module.gradient_snapshot(model, 1)
    optimizer.step()
    rows = diagnostic_module.complete_update_snapshot(model, rows, before)
    assert rows[0]["grad_norm_before_clip"] == 2
    assert rows[0]["update_norm"] == pytest.approx(0.2, abs=1e-7)
    assert rows[1]["parameters_without_grad"] == 2
    assert rows[1]["update_norm"] == 0


def test_feature_collapse_and_strict_json(diagnostic_module, tmp_path):
    torch = diagnostic_module.torch
    collapsed = diagnostic_module.feature_statistics(torch.ones(8, 4))
    assert collapsed["near_constant_dimensions"] == 4 and collapsed["effective_rank"] == 0
    destination = tmp_path / "stats.json"
    diagnostic_module.write_json(destination, {"correlation": np.nan, "values": np.array([1., np.inf])})
    assert json.loads(destination.read_text()) == {"correlation": None, "values": [1., None]}


def test_diagnostic_observation_does_not_change_training(pilot):
    enabled = pilot["DIAGNOSTICS"]["enabled"]
    try:
        pilot["DIAGNOSTICS"]["enabled"] = True
        observed_model, _, observed_history, _, observed_probabilities, _ = pilot["train_condition"](125, 0.75)
        pilot["DIAGNOSTICS"]["enabled"] = False
        plain_model, _, plain_history, _, plain_probabilities, _ = pilot["train_condition"](125, 0.75)
        for name, parameter in observed_model.state_dict().items():
            pilot["torch"].testing.assert_close(parameter, plain_model.state_dict()[name], rtol=0, atol=0)
        np.testing.assert_array_equal(observed_history.masked_mse, plain_history.masked_mse)
        np.testing.assert_array_equal(observed_probabilities, plain_probabilities)
    finally:
        pilot["DIAGNOSTICS"]["enabled"] = enabled


def test_tiny_overfit_runs_separate_fixed_training_task(pilot, tmp_path):
    pilot["seed_everything"](42)
    model = pilot["PlainECGMAE"](125, pilot["PROFILE"]).cpu()
    model.ratio = 0.75
    directory = tmp_path / "tiny"
    history = pilot["diagnostics"].run_tiny_overfit(
        model, pilot["arrays"]["train"]["x"], pilot["frames"]["train"].ecg_id.to_numpy(),
        pilot["TRAIN_IDX"][:2], 8, 42, directory, lead_order=pilot["LEADS"])
    assert len(history) == 8
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    assert summary["mse_reduction_fraction"] > 0
    assert (directory / "final_reconstruction_samples.npz").is_file()
    config = json.loads((directory / "run_config.json").read_text(encoding="utf-8"))
    assert config["kind"] == "tiny-overfit-not-pilot"
    assert config["artifact_indices"] == pilot["TRAIN_IDX"][:2].tolist()


def test_runner_overrides_validation_before_training():
    path = ROOT / "experiments/run_masking_pilot.py"
    spec = importlib.util.spec_from_file_location("runner_options", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    for kwargs in [dict(steps=0), dict(patch_samples=50), dict(mask_ratio=0.5),
                   dict(patch_samples=64, mask_ratio=0.5), dict(patch_samples=50, mask_ratio=1),
                   dict(initialization="unknown")]:
        with pytest.raises(ValueError):
            runner.run_notebook(**kwargs)
