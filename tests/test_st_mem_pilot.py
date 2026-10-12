"""Official source integration; GPU memory/performance are not tested here."""
import io
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("einops")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import st_mem_adapter as adapter
import run_st_mem_pilot as runner
import masking_diagnostics as diagnostics

UPSTREAM = ROOT.parent / "ST-MEM"


@pytest.fixture
def model():
    if not (UPSTREAM / "models/st_mem.py").is_file():
        pytest.skip("Requires the separately cloned official ST-MEM")
    torch.set_num_threads(2)
    torch.manual_seed(42)
    return adapter.STMEMAdapter(UPSTREAM, seq_len=500, model_size="smoke")


@pytest.mark.parametrize("patch", [50, 125, 250])
@pytest.mark.parametrize("ratio", [.50, .75, .90])
def test_per_lead_mask_counts_and_seed(patch, ratio):
    n = 12 * 2500 // patch
    q, keep = adapter.mask_layout(n, ratio)
    assert keep == q // {.50: 2, .75: 4, .90: 10}[ratio]
    a = adapter.random_lead_mask(3, n, ratio, 42)
    b = adapter.random_lead_mask(3, n, ratio, 42)
    np.testing.assert_array_equal(a, b)
    assert a.dtype == bool and a.shape == (3, n)
    np.testing.assert_array_equal((~a.reshape(3, 12, q)).sum(-1), np.full((3, 12), keep))
    assert not np.array_equal(a, adapter.random_lead_mask(3, n, ratio, 43))


def test_actual_ratio_floors_fractional_counts_without_float_underflow():
    q, keep = adapter.mask_layout(600, .75)
    assert keep == 12 and 1-keep/q == .76
    assert adapter.mask_layout(600, .90) == (50, 5)
    assert adapter.mask_layout(240, .90) == (20, 2)
    assert adapter.mask_layout(120, .90) == (10, 1)
    assert adapter.mask_layout(120, .75) == (10, 2)
    # A genuinely fractional count below one remains invalid, not clamped.
    with pytest.raises(ValueError, match="at least one visible"):
        adapter.mask_layout(120, .91)


@pytest.mark.parametrize("patch", [50, 125, 250])
@pytest.mark.parametrize("ratio", [.50, .75, .90])
def test_all_nine_grid_conditions_have_finite_forward_backward(patch, ratio):
    if not (UPSTREAM / "models/st_mem.py").is_file():
        pytest.skip("Requires the separately cloned official ST-MEM")
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = adapter.STMEMAdapter(UPSTREAM, patch_samples=patch, model_size="smoke")
    model.ratio = ratio
    x = torch.randn(1, 12, 2500)
    mask = torch.from_numpy(adapter.random_lead_mask(1, model.n, ratio, 42))
    pred, target = model(x, mask)
    assert pred.shape == target.shape == (1, model.n, patch)
    loss = ((pred-target).square().mean(-1)*mask).sum()/mask.sum()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert any(p.grad is not None and bool(p.grad.abs().sum() > 0)
               for p in model.core.encoder.parameters())


def test_fixed_masks_are_ecg_keyed_and_do_not_change_global_rng():
    before = np.random.get_state()
    a, seeds = adapter.fixed_lead_mask([3, 9], 600, .75, 42, repeat=2)
    b, _ = adapter.fixed_lead_mask([9, 3], 600, .75, 42, repeat=2)
    np.testing.assert_array_equal(a, b[::-1])
    np.testing.assert_array_equal(seeds, [2100045, 2100051])
    now = np.random.get_state()
    assert before[0] == now[0] and before[2:] == now[2:]
    np.testing.assert_array_equal(before[1], now[1])


@pytest.mark.parametrize("n,ratio", [(600, 0), (600, 1), (600, .999), (601, .75)])
def test_invalid_mask_layout(n, ratio):
    with pytest.raises(ValueError):
        adapter.mask_layout(n, ratio)


def test_loader_does_not_shadow_existing_models_package(model, monkeypatch):
    before = adapter.source_metadata(UPSTREAM)
    names = ["models", "models.encoder", "models.encoder.vit", "models.encoder.st_mem_vit", "models.st_mem"]
    sentinels = {n: types.ModuleType(n) for n in names}
    for n, module in sentinels.items():
        monkeypatch.setitem(sys.modules, n, module)
    monkeypatch.setattr(adapter, "_CACHE", {})
    loaded = adapter.load_official_class(UPSTREAM)
    assert loaded.__name__ == "ST_MEM"
    assert all(sys.modules[n] is m for n, m in sentinels.items())
    assert adapter.source_metadata(UPSTREAM) == before


def test_adapter_predictions_match_official_forward_with_same_mask(model):
    model.eval()
    x = torch.randn(2, 12, 500)
    with torch.no_grad():
        official = model.core(x, mask_ratio=.75)
        mask = official["mask"].reshape(2, model.n).bool()
        pred, target = model(x, mask)
        torch.testing.assert_close(pred, official["pred"].reshape_as(pred), atol=2e-5, rtol=2e-5)
        assert torch.equal(target.reshape_as(x), x)
        loss = ((pred-target).square().mean(-1)*mask).sum()/mask.sum()
        torch.testing.assert_close(loss, official["loss"], atol=2e-5, rtol=2e-5)
    assert model.core._external_mask is None and model.core.norm_pix_loss is False


def test_hidden_input_isolation_and_masked_only_backward(model):
    model.eval()
    x = torch.randn(2, 12, 500, requires_grad=True)
    mask = torch.from_numpy(adapter.random_lead_mask(2, model.n, .75, 42))
    pred, target = model(x, mask)
    changed = x.detach().clone().reshape(2, model.n, model.p)
    changed[mask] += 1000
    other, _ = model(changed.reshape_as(x), mask)
    torch.testing.assert_close(pred, other, atol=2e-5, rtol=2e-5)
    pred.sum().backward()
    assert torch.equal(x.grad.reshape(2, model.n, model.p)[mask], torch.zeros_like(target[mask]))
    assert model.core.encoder.pos_embedding.grad is None
    assert model.core.encoder.pos_embedding.requires_grad is False
    model.zero_grad(set_to_none=True)
    pred, target = model(x.detach(), mask)
    loss = ((pred-target).square().mean(-1)*mask).sum()/mask.sum()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.core.encoder.parameters())


def test_arbitrary_global_or_nonboolean_masks_are_rejected(model):
    x = torch.randn(2, 12, 500)
    mask = torch.from_numpy(adapter.random_lead_mask(2, model.n, .75, 42))
    with pytest.raises(ValueError, match="boolean"):
        model(x, mask.float())
    unequal = mask.clone()
    visible = torch.where(~unequal[0, :10])[0][0]
    unequal[0, visible] = True
    with pytest.raises(ValueError, match="equal"):
        model(x, unequal)
    with pytest.raises(ValueError):
        model(x[:0], mask[:0])


def test_features_and_weights_only_checkpoint_roundtrip(model):
    model.eval()
    x = torch.randn(1, 12, 500)
    mask = torch.from_numpy(adapter.random_lead_mask(1, model.n, .75, 42))
    with torch.no_grad():
        torch.testing.assert_close(model.features(x), model.core.encoder.forward_encoding(x))
        pred, _ = model(x, mask)
    stream = io.BytesIO()
    torch.save(runner._checkpoint(model, {"config": {"backend": "st-mem-official"}}, 3), stream)
    stream.seek(0)
    checkpoint = torch.load(stream, weights_only=True)
    clone = adapter.STMEMAdapter(UPSTREAM, seq_len=500, model_size="smoke").eval()
    clone.load_state_dict(checkpoint["model"])
    with torch.no_grad():
        torch.testing.assert_close(clone(x, mask)[0], pred, atol=0, rtol=0)
    assert checkpoint["step"] == 3


@pytest.mark.parametrize("argv", [
    ["--mode", "pilot", "--model-size", "smoke"],
    ["--train", "--tiny-overfit"], ["--steps", "0"],
    ["--batch-size", "0"], ["--accum-steps", "0"], ["--seed", "-1"],
    ["--mask-ratio", "1"],
])
def test_cli_rejects_invalid_experiments(argv):
    with pytest.raises(SystemExit):
        runner.arguments(argv)


def test_cli_defaults_and_cpu_pilot_guard(monkeypatch):
    assert runner.arguments([]).model_size == "smoke"
    assert runner.arguments(["--mode", "pilot"]).model_size == "small"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="Pilot requires CUDA"):
        runner.prepare_experiment(runner.arguments(["--mode", "pilot"]))


def test_notebook_is_valid_and_has_no_stale_outputs():
    path = ROOT / "experiments/ST_MEM_pilot.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"ST_MEM_pilot:cell{i+1}", "exec")
            assert cell["outputs"] == [] and cell["execution_count"] is None


def test_accumulated_training_saves_diagnostics_and_safe_checkpoints(tmp_path, monkeypatch, model):
    """Two updates on actual smoke data; same accumulated gradients as full batch."""
    if not (ROOT / "preprocessing/artifacts/ptbxl_250hz/manifest.json").is_file():
        pytest.skip("Requires local PTB-XL preprocessing artifacts")
    a = runner.arguments(["--mode", "smoke", "--train", "--steps", "2", "--eval-every", "1",
                          "--batch-size", "2", "--accum-steps", "3", "--no-probe",
                          "--st-mem-repo", str(UPSTREAM), "--output-root", str(tmp_path)])
    ctx = runner.prepare_experiment(a)
    ns = ctx["namespace"]
    initial = {k: v.clone() for k, v in ctx["model"].state_dict().items()}
    grads = {}
    clip = torch.nn.utils.clip_grad_norm_

    def capture_grad(params, *args, **kwargs):
        if not grads:
            grads.update({n: p.grad.detach().clone() for n, p in ctx["model"].named_parameters() if p.grad is not None})
        return clip(params, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", capture_grad)
    result = runner.train_experiment(ctx)
    reference = adapter.STMEMAdapter(UPSTREAM, model_size="smoke")
    reference.load_state_dict(initial)
    chosen = np.random.default_rng(42).choice(ns["TRAIN_IDX"], 6, replace=False)
    x = ns["read_batch"]("train", chosen, torch.device("cpu"))
    masks = np.concatenate([adapter.random_lead_mask(2, reference.n, .75, 1042+i) for i in range(3)])
    mask = torch.from_numpy(masks)
    pred, target = reference(x, mask)
    loss = ns["masked_mse"](pred, target, mask)
    loss.backward()
    for name, p in reference.named_parameters():
        if p.grad is not None:
            torch.testing.assert_close(grads[name], p.grad, atol=2e-6, rtol=2e-4)
    run, condition = result["RUN_DIR"], result["RUN_DIR"] / "p50_r75"
    config = json.loads((run / "run_config.json").read_text(encoding="utf-8"))
    assert config["effective_batch_size"] == 6 and config["norm_pix_loss"] is False
    assert config["realized_mask_ratio"] == .76 and config["probe_enabled"] is False
    for filename, step in [("initial_checkpoint.pt", 0), ("latest_checkpoint.pt", 1), ("checkpoint.pt", 2)]:
        saved = torch.load(condition / filename, weights_only=True)
        assert saved["step"] == step and saved["config"]["upstream"]["sha256"]
    for filename in ["pretrain_history.csv", "gradient_updates.csv", "learning_curves.png",
                     "diagnostic_curves.png", "final_reconstruction_samples.npz", "final_reconstruction_metrics.json"]:
        assert (condition / filename).is_file(), filename
    assert len(ns["pd"].read_csv(condition / "pretrain_history.csv")) == 2
    metric = result["RESULTS"].iloc[0]
    assert np.isfinite(metric.validation_masked_mse)
    assert metric.validation_masked_samples == ns["PROFILE"]["eval_n"] * 3 * 12 * 38 * 50
    assert "macro_auroc" not in result["RESULTS"]


def test_trained_and_random_probes_are_saved_in_smoke(tmp_path, model):
    if not (ROOT / "preprocessing/artifacts/ptbxl_250hz/manifest.json").is_file():
        pytest.skip("Requires local PTB-XL preprocessing artifacts")
    ctx = runner.prepare_experiment(runner.arguments([
        "--mode", "smoke", "--train", "--steps", "1", "--st-mem-repo", str(UPSTREAM),
        "--output-root", str(tmp_path)]))
    result = runner.train_experiment(ctx)
    condition = result["RUN_DIR"] / "p50_r75"
    saved = torch.load(condition / "checkpoint.pt", weights_only=True)
    random_probe = torch.load(condition / "random_encoder_probe.pt", weights_only=True)
    assert saved["step"] == 1 and saved["probe"]["head"]["weight"].shape == (5, 32)
    assert random_probe["head"]["weight"].shape == (5, 32)
    assert not any(p.requires_grad for p in ctx["model"].parameters())
    trained = np.load(condition / "validation_probabilities.npy")
    untrained = np.load(condition / "random_encoder_probabilities.npy")
    ids = np.load(condition / "validation_ecg_ids.npy")
    assert trained.shape == untrained.shape == (len(ids), 5)
    assert np.isfinite(trained).all() and ((trained >= 0) & (trained <= 1)).all()
    row = result["RESULTS"].iloc[0]
    assert row.auroc_gain_vs_random == pytest.approx(row.macro_auroc-row.random_encoder_macro_auroc)
    assert (condition / "feature_statistics.json").is_file()
