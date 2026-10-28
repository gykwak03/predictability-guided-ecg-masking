"""Behavioral tests for isolation, paired learning streams and evidence boundaries."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from mae_cause_data import CLASSES, LEADS, digest, fit_statistics, normalize, write_json, prepare_cache, load_cache
from mae_cause_model import CauseMAE, learning_rate, mask_batch, notebook_class, SIZES
from mae_cause_plan import make_plan, validate_plan
from mae_cause_eval import probe_audit
import run_mae_cause_study as runner
from analyze_mae_cause_study import analyze, crossed_bootstrap


@pytest.fixture
def plan():
    torch.set_num_threads(2)
    p = make_plan(".", smoke=True, train_ecgs=32, validation_ecgs=16,
                  microbatch=2, accumulation=1, short=2, long=4, seeds=[42])
    p.pop("plan_id")
    p.update(eval_every=2, diagnostic_ecgs=2, final_ecgs=4, repeats=1, probe_checkpoints=[2, 4])
    p["plan_id"] = digest(p)
    return p


@pytest.mark.parametrize("initialization", ["legacy", "independent"])
def test_all_off_is_actual_legacy_notebook_including_features(plan, initialization):
    spec = dict(plan["conditions"]["legacy_short"], initialization=initialization)
    torch.manual_seed(42)
    old = notebook_class()(50, dict(SIZES["smoke"], initialization=initialization)).eval()
    torch.manual_seed(42)
    new = CauseMAE(spec).eval()
    assert old.state_dict().keys() == new.state_dict().keys()
    for k, v in old.state_dict().items():
        assert torch.equal(v, new.state_dict()[k]), k
    x = torch.randn(2, 12, 2500)
    mask = torch.from_numpy(mask_batch(2, 600, .5, 44, "global"))
    with torch.no_grad():
        torch.testing.assert_close(old(x, mask)[0], new(x, mask)[0], rtol=0, atol=0)
        torch.testing.assert_close(old.features(x), new.features(x), rtol=1e-5, atol=2e-7)


@pytest.mark.parametrize("name", ["factor_100", "factor_010", "factor_001", "sep_only", "all_on"])
def test_interventions_have_no_direct_target_leak_and_encoder_learns(plan, name):
    spec = dict(plan["conditions"][name], model_seed=42)
    model = runner.build_model(spec, ROOT.parent / "ST-MEM").eval()
    x = torch.randn(2, 12, 2500, requires_grad=True)
    mask = torch.from_numpy(mask_batch(2, 600, .5, 42, spec["train_policy"]))
    prediction, target = model(x, mask)
    loss = ((prediction - target.detach()).square().mean(-1) * mask).sum() / mask.sum()
    loss.backward()
    hidden = mask.repeat_interleave(50, 1).reshape_as(x)
    assert not x.grad[hidden].any()
    assert x.grad[~hidden].abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
    with torch.no_grad():
        original = model(x.detach(), mask)[0]
        changed = x.detach().clone()
        changed[hidden] += 100
        torch.testing.assert_close(original, model(changed, mask)[0], rtol=0, atol=0)
        torch.testing.assert_close(target, x.reshape(2, 600, 50), rtol=0, atol=0)


def test_each_planned_edge_changes_one_factor_and_historical75_is_not_rounded(plan):
    validate_plan(plan)
    for contrast in plan["contrasts"]:
        if contrast["kind"] != "anchor":
            a, b = [plan["conditions"][contrast[k]] for k in ["reference", "treatment"]]
            assert {k for k in a if a[k] != b[k]} == {contrast["factor"]}
    p = make_plan(".", smoke=True, stages=["controls"], ratio=.75,
                  train_ecgs=32, microbatch=2, accumulation=1)
    assert p["evaluation_banks"] == ["global"]
    with pytest.raises(ValueError, match="exact per-lead"):
        make_plan(".", smoke=True, ratio=.75, train_ecgs=32, microbatch=2, accumulation=1)
    bad = copy.deepcopy(plan)
    bad["conditions"]["factor_100"]["steps"] += 1
    with pytest.raises(ValueError, match="modified"):
        validate_plan(bad)


def test_mask_count_and_microbatch_do_not_change_full_mask(plan):
    for policy in ["global", "per_lead"]:
        mask = mask_batch(32, 600, .5, 42, policy)
        assert np.all(mask.sum(1) == 300)
        assert np.array_equal(np.concatenate([mask[s:s+4] for s in range(0, 32, 4)]), mask)
        if policy == "per_lead":
            assert np.all(mask.reshape(32, 12, 50).sum(-1) == 25)
    with pytest.raises(ValueError, match="Exact per-lead"):
        mask_batch(2, 600, .75, 42, "per_lead")


def test_lr_matches_old_notebook_and_extension_does_not_restart_cosine():
    def historical(index):
        if index < 100:
            return 3e-4 * (index + 1) / 100
        return 3e-4 * (.1 + .9 * .5 * (1 + np.cos(np.pi * (index - 100) / 900)))
    for update in [1, 50, 100, 101, 500, 1000]:
        assert learning_rate(update, horizon=1000) == pytest.approx(historical(update - 1), abs=1e-15)
    assert learning_rate(1001, horizon=1000) == pytest.approx(3e-5)
    assert learning_rate(3000, horizon=1000) == pytest.approx(3e-5)


def test_normalization_reversible_and_hidden_statistics_dependency_is_explicit():
    rng = np.random.default_rng(23)
    raw = rng.normal(size=(4, 12, 2500)).astype(np.float32)
    raw[:, 1] = raw[:, 0] + raw[:, 2]
    stats = fit_statistics(raw[:2])
    mask = mask_batch(4, 600, .5, 42, "per_lead")
    hidden = np.repeat(mask, 50, 1).reshape(raw.shape)
    changed = raw.copy()
    changed[hidden] += 10
    for mode in ["per_record", "train_lead", "train_common", "visible_record"]:
        x, mean, scale = normalize(raw, mode, stats, mask=mask)
        np.testing.assert_allclose(x * scale + mean, raw, atol=8e-7)
        other = normalize(changed, mode, stats, mask=mask)[0]
        if mode == "per_record":
            assert np.max(np.abs(x[~hidden] - other[~hidden])) > .1
        else:
            np.testing.assert_array_equal(x[~hidden], other[~hidden])
    common = normalize(raw, "train_common", stats)[0]
    np.testing.assert_allclose(common[:, 1], common[:, 0] + common[:, 2], atol=5e-7)
    # Parameters use train only: changing validation does not mutate fitted stats.
    assert stats == fit_statistics(raw[:2])


def test_crossed_patient_seed_bootstrap_detects_known_effect_and_null():
    null = crossed_bootstrap(np.zeros((3, 20)), repeats=100)
    assert null["mean"] == null["ci_low"] == null["ci_high"] == 0
    effect = crossed_bootstrap(np.full((3, 20), .2), repeats=100)
    assert effect["ci_low"] == pytest.approx(.2)
    assert effect["positive_seed_fraction"] == 1
    with pytest.raises(ValueError):
        crossed_bootstrap([[np.nan]])


@pytest.mark.parametrize("size", ["baseline", "stmem-small"])
def test_real_model_sizes_keep_original_predictions(plan, size):
    spec = dict(plan["conditions"]["legacy_short"], size=size)
    torch.manual_seed(42)
    old = notebook_class()(50, dict(SIZES[size], initialization="legacy")).eval()
    torch.manual_seed(42)
    model = CauseMAE(spec).eval()
    x = torch.randn(1, 12, 2500)
    mask = torch.from_numpy(mask_batch(1, 600, .5, 47, "global"))
    with torch.no_grad():
        torch.testing.assert_close(old(x, mask)[0], model(x, mask)[0], rtol=0, atol=0)
    assert sum(p.numel() for p in model.parameters()) == (3127282 if size == "baseline" else 24624818)


def test_raw_cache_replays_filter_keeps_test_closed_and_rejects_corruption(tmp_path):
    wfdb = pytest.importorskip("wfdb")
    from scipy.signal import butter, resample_poly, sosfiltfilt
    rawdir, source = tmp_path / "raw", tmp_path / "source"
    rawdir.mkdir()
    source.mkdir()
    config = dict(source_sampling_rate=500, target_sampling_rate=250, bandpass_low=.67,
                  bandpass_high=50., filter_order=4, normalization="per-record")
    write_json(source / "manifest.json", dict(config=config, lead_order=LEADS, class_order=CLASSES))
    t = np.arange(5000) / 500
    sos = butter(4, [.67, 50], fs=250, btype="bandpass", output="sos")
    for split, offset, fold in [("train", 0, 1), ("validation", 100, 9)]:
        frames, arrays = [], []
        for i in range(2):
            name = f"{split}{i}"
            signal = np.stack([.3 * np.sin(2*np.pi*8*t + c/12) + .1*np.sin(2*np.pi*45*t) for c in range(12)], 1)
            wfdb.wrsamp(name, fs=500, units=["mV"]*12, sig_name=LEADS, p_signal=signal,
                        fmt=["16"]*12, write_dir=str(rawdir))
            raw = wfdb.rdsamp(str(rawdir / name))[0].T.astype(np.float32)
            mv = sosfiltfilt(sos, resample_poly(raw, 1, 2, axis=1, padtype="line").astype(np.float32), axis=1).astype(np.float32)
            arrays.append(normalize(mv[None], "per_record", {})[0][0])
            frames.append(dict(artifact_index=i, ecg_id=offset+i, patient_id=offset+i, split=split,
                               strat_fold=fold, source_record=name, **{c: int(c == "NORM") for c in CLASSES}))
        pd.DataFrame(frames).to_csv(source / f"metadata_{split}.csv", index=False)
        np.save(source / f"signals_{split}.npy", np.stack(arrays))
    # There are deliberately no test files in either source or rawdir.
    plan = make_plan(source, stages=["controls"], train_ecgs=2, microbatch=1, accumulation=1)
    manifest = prepare_cache(plan, rawdir, tmp_path / "cache")
    assert manifest["replay_max_abs_error"] == 0
    assert manifest["test_opened"] is False
    loaded = load_cache(plan, tmp_path / "cache")
    assert loaded["stats"] == fit_statistics(loaded["arrays"]["train"])
    with pytest.raises(FileExistsError):
        prepare_cache(plan, rawdir, tmp_path / "cache")
    changed = np.load(tmp_path / "cache/mv_validation.npy", mmap_mode="r+")
    changed[0, 0, 0] += 1
    changed.flush()
    del changed
    with pytest.raises(ValueError, match="modified"):
        load_cache(plan, tmp_path / "cache")


def test_smoke_prefix_identity_probe_and_analysis_exclusion(plan, tmp_path, monkeypatch):
    data = runner.synthetic_data(plan)
    args = runner.arguments(["run", "--plan", str(tmp_path / "plan.json"),
        "--output-root", str(tmp_path / "study"), "--device", "cpu", "--no-probe"])
    write_json(args.output_root.parent / "plan.json", plan)
    args.output_root.mkdir()
    write_json(args.output_root / "plan.json", plan)
    a = runner.train_run(plan, data, "independent_short", 42, args)
    b = runner.train_run(plan, data, "independent_extend", 42, args)
    sa, sb = [json.loads((p / "summary.json").read_text(encoding="utf-8")) for p in [a, b]]
    assert sa["milestones"]["2"] == sb["milestones"]["2"]
    # Resume from the saved mid-run state, and compare the actual final weights.
    full = torch.load(b / "latest.pt", map_location="cpu", weights_only=False)
    assert full["update"] == 4
    interrupted_args = copy.copy(args)
    interrupted_args.output_root = tmp_path / "interrupted"
    original_save = runner.save_checkpoint

    def interrupt_after_checkpoint(path, checkpoint):
        original_save(path, checkpoint)
        if checkpoint["update"] == 2:
            raise RuntimeError("simulated preemption after a safe checkpoint")

    monkeypatch.setattr(runner, "save_checkpoint", interrupt_after_checkpoint)
    with pytest.raises(RuntimeError, match="simulated preemption"):
        runner.train_run(plan, data, "independent_extend", 42, interrupted_args)
    monkeypatch.setattr(runner, "save_checkpoint", original_save)
    interrupted_args.resume = True
    resumed = runner.train_run(plan, data, "independent_extend", 42, interrupted_args)
    resumed_checkpoint = torch.load(resumed / "latest.pt", map_location="cpu", weights_only=False)
    assert runner.state_digest(resumed_checkpoint["model"]) == runner.state_digest(full["model"])
    assert resumed_checkpoint["streams"] == full["streams"]
    frame = analyze(args.output_root, tmp_path / "analysis", bootstrap=100)
    assert frame.status.eq("missing scientific runs").all()
    audit = json.loads((tmp_path / "analysis/analysis_audit.json").read_text())
    assert audit["scientific_runs"] == 0 and audit["excluded_synthetic"]
    # Probe audit uses frozen weights, identical classifier prefixes and both pooling/input conditions.
    spec = dict(plan["conditions"]["independent_short"], model_seed=42, smoke=True)
    model = runner.build_model(spec, ROOT.parent / "ST-MEM")
    initial = copy.deepcopy(model.state_dict())
    out = tmp_path / "probe"
    out.mkdir()
    rows = probe_audit(model, initial, data, spec, plan, torch.device("cpu"), out,
                       lambda: runner.build_model(spec, ROOT.parent / "ST-MEM"), expanded=True)
    assert len(rows) == 8
    assert all(abs(row["auroc_gain_vs_random"]) < 1e-12 for row in rows)
    for k, v in model.state_dict().items():
        assert torch.equal(v, initial[k])


def test_analyzer_detects_injected_effect_and_rejects_changed_code(plan, tmp_path):
    """Artificial result files exercise statistics/guards; not scientific evidence."""
    p = copy.deepcopy(plan)
    p.pop("plan_id")
    p.update(smoke=False, seeds=[42, 43, 44])
    p["plan_id"] = digest(p)
    root = tmp_path / "fake-result-fixture"
    root.mkdir()
    write_json(root / "plan.json", p)
    import itertools
    cube = []
    for ln, pos, dec in itertools.product([0, 1], repeat=3):
        name = "independent_long" if not any([ln, pos, dec]) else f"factor_{ln}{pos}{dec}"
        cube.append((name, 1 - .10*ln - .05*pos - .02*dec - .20*ln*pos))
    for name, error in [("legacy_short", 1.), ("independent_short", .7), *cube]:
        for seed in p["seeds"]:
            folder = root / "runs" / name / f"seed{seed}"
            folder.mkdir(parents=True)
            fingerprint = digest([name, seed])
            spec = dict(p["conditions"][name], model_seed=seed, smoke=False)
            write_json(folder / "run_config.json", dict(fingerprint=fingerprint, seed=seed, source_sha256={"unit-test-fixture": "same"}))
            write_json(folder / "summary.json", dict(fingerprint=fingerprint, completed=True,
                kind="scientific", seed=seed, plan_id=p["plan_id"], cache_id="artificial-unit-fixture", spec=spec,
                milestones={str(u): dict(streams=dict(batch="same", mask="same", lr="same"), state_sha256=name) for u in [2, spec["steps"]]},
                final={bank: dict(mask_sha256="same") for bank in p["evaluation_banks"]}))
            pd.DataFrame([dict(ecg_id=e, patient_id=e // 2, repeat=r, bank=bank,
                               standardized_mse=error, masked_mse=error)
                for bank in p["evaluation_banks"] for e in range(8) for r in range(3)]).to_csv(folder / "final_per_ecg.csv", index=False)
            pd.DataFrame([dict(update=u, split=s, bank=bank, standardized_nrmse=np.sqrt(error), rms_ratio=.5, masked_correlation=.5)
                for bank in p["evaluation_banks"] for u in [0, 2] for s in ["train", "validation"]]).to_csv(folder / "evaluation_history.csv", index=False)
    result = analyze(root, tmp_path / "analysis", bootstrap=100)
    rows = result[result.factor.eq("initialization")]
    assert len(rows) == 2
    assert rows.seed_count.eq(3).all()
    np.testing.assert_allclose(rows["mean"], .3)
    assert rows.status.str.startswith("consistent candidate").all()
    interactions = json.loads((tmp_path / "analysis/interactions.json").read_text())
    ln_pos = [r for r in interactions if r["factors"] == "patch_norm x fixed_time" and "conditioning" not in r]
    assert len(ln_pos) == 2
    assert all(r["mean"] == pytest.approx(.20) for r in ln_pos)
    threeway = [r for r in interactions if r["factors"].startswith("three_way")]
    assert all(abs(r["mean"]) < 1e-12 for r in threeway)
    bad = root / "runs/independent_short/seed42/run_config.json"
    config = json.loads(bad.read_text())
    config["source_sha256"]["unit-test-fixture"] = "changed"
    write_json(bad, config)
    with pytest.raises(ValueError, match="Code versions differ"):
        analyze(root, tmp_path / "bad_analysis", bootstrap=100)
