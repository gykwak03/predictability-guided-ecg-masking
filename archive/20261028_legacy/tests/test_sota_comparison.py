"""Scientific contracts of the shared 50 Hz pilot, using real upstream models."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"experiments"))
from run_sota_comparison import CLASSES, LEADS, load_data, class_metrics, optimizer_groups
from sota_adapters import create_model, reference_cauchy
from st_mem_adapter import random_lead_mask
from compare_sota_runs import compare


@pytest.fixture
def fixture_data(tmp_path):
    manifest = dict(config=dict(target_sampling_rate=250, bandpass_low=.67, bandpass_high=50.,
                                normalization="per-record"), lead_order=LEADS, class_order=CLASSES)
    (tmp_path/"manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for offset, split in [(0, "train"), (1000, "validation")]:
        n = 32
        y = np.stack([((np.arange(n)+i)%3 == 0) for i in range(5)], axis=1).astype(np.uint8)
        frame = pd.DataFrame(dict(artifact_index=np.arange(n), ecg_id=np.arange(n)+offset,
            patient_id=np.arange(n)+offset, strat_fold=5 if split == "train" else 9))
        for i, cls in enumerate(CLASSES): frame[cls] = y[:, i]
        frame.to_csv(tmp_path/f"metadata_{split}.csv", index=False)
        x = np.random.default_rng(offset).normal(size=(n,12,2500)).astype(np.float32)
        np.save(tmp_path/f"signals_{split}.npy", x)
        np.save(tmp_path/f"labels_{split}.npy", y)
        np.save(tmp_path/f"label_valid_{split}.npy", y.any(axis=1))
    return tmp_path


def test_new_preprocessing_and_no_test_access(fixture_data):
    a = load_data(fixture_data, 8, 42)
    b = load_data(fixture_data, 8, 42)
    assert np.array_equal(a["train"], b["train"])
    assert len(a["probe_train"]) > 0
    assert not (fixture_data/"signals_test.npy").exists()
    m = json.loads((fixture_data/"manifest.json").read_text())
    m["config"]["bandpass_high"] = 40.
    (fixture_data/"manifest.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="NEW"):
        load_data(fixture_data, 8, 42)


def test_leakage_and_label_alignment_rejected(fixture_data):
    p = fixture_data/"metadata_validation.csv"
    f = pd.read_csv(p)
    f.loc[0,"patient_id"] = 0
    f.to_csv(p, index=False)
    with pytest.raises(ValueError, match="leakage"):
        load_data(fixture_data, 8, 42)


@pytest.mark.parametrize("name", ["stmem", "patchecg"])
def test_real_upstream_model_hidden_values_cannot_leak(name):
    external = ROOT.parent/"external"
    if not external.exists(): pytest.skip("Official external checkouts required")
    torch.set_num_threads(2)
    model = create_model(name, external, smoke=True).eval()
    x = torch.randn(2,12,2500)
    mask = torch.from_numpy(random_lead_mask(2,600,.5,42))
    changed = x.clone().reshape(2,600,50)
    changed[mask] = torch.randn_like(changed[mask])*50
    with torch.no_grad():
        a,t = model(x,mask)
        b,_ = model(changed.reshape_as(x),mask)
        f = model.features(x)
    assert a.shape == t.shape == (2,600,50)
    assert torch.allclose(a,b,rtol=0,atol=1e-6)
    assert f.shape[0] == 2 and torch.isfinite(f).all()
    model.train()
    opt = torch.optim.AdamW(optimizer_groups(model, 3e-4,.05), lr=3e-4)
    parameter = next(p for p in model.parameters() if p.requires_grad)
    before = parameter.detach().clone()
    pred,target = model(x,mask)
    loss = (pred-target).square().mean(-1)[mask].mean()
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)
    assert not torch.equal(before,parameter)


def test_reference_cauchy_matches_explicit_full_conjugates():
    torch.manual_seed(42)
    v = torch.randn(2,3,4,dtype=torch.cdouble,requires_grad=True)
    w = torch.randn(3,4,dtype=torch.cdouble)
    z = torch.randn(11,dtype=torch.cdouble)
    full_v, full_w = torch.cat([v,v.conj()],-1), torch.cat([w,w.conj()],-1)
    expected = (full_v[...,None]/(z-full_w[...,None])).sum(-2)
    actual = reference_cauchy(v,z,w)
    assert torch.allclose(actual,expected,atol=1e-12,rtol=1e-12)
    actual.abs().sum().backward()
    assert torch.isfinite(v.grad).all()


def test_cpc_is_official_ssm_and_native_objective():
    pytest.importorskip("pytorch_lightning")
    external = ROOT.parent/"external"
    if not (external/"ecg-fm-benchmarking").exists(): pytest.skip("Official CPC checkout required")
    torch.set_num_threads(2)
    model = create_model("ecgcpc", external, smoke=True)
    model2 = create_model("ecgcpc", external, smoke=True)  # namespace package may be reused
    assert model.architecture["state_dim"] == 8 and model.architecture["mask_ratio"] is None
    assert type(model.predictor).__module__ == "clinical_ts.ts.s4"
    x = torch.randn(2,12,64)
    # Reproduce the initial frozen-feature probe before training.
    model.eval()
    with torch.no_grad(): assert model.features(x).shape == (2,32)
    model.train()
    loss = model.ssl_loss(x)
    assert torch.isfinite(loss) and loss > 0
    p = next(model.encoder.parameters())
    before = p.detach().clone()
    opt = torch.optim.AdamW(optimizer_groups(model,3e-4,.05),lr=3e-4)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.predictor.parameters())
    opt.step()
    assert not torch.equal(before,p)
    model.eval()
    with torch.no_grad(): assert model.features(x).shape == (2,32)


def test_missing_classes_explicitly_marked():
    m = class_metrics(np.zeros((4,5)), np.full((4,5), .5))
    assert m["macro_auroc"] is None and not m["all_five_classes_supported"]


def test_comparison_refuses_unmatched_data_or_profile(tmp_path):
    folders = []
    for i,model in enumerate(["stmem","patchecg","ecgcpc"]):
        folder = tmp_path/model
        folder.mkdir()
        c = dict(model=model, seed=42, source_manifest_sha256="new-50hz", steps=3000,
                 effective_batch=32, microbatch=4, accum_steps=8, optimizer={},
                 train_ecg_ids=[1,2], validation_ecg_ids=[3,4],
                 subset_seed=42, probe={}, parameters=10, test_opened=False, pretrained_weights=False)
        s = dict(status="completed",eligible_for_model_selection=True,test_opened=False,
                 trained_probe=dict(macro_auroc=.8+i*.01,macro_auprc_ap=.5,macro_f1=.5),
                 random_probe=dict(macro_auroc=.7),pretrain_seconds=1,gpu_peak_allocated_bytes=10,
                 mask_sampler_compatible=model!="ecgcpc",batch_sequence_sha256="same",
                 mask_sequence_sha256="same" if model!="ecgcpc" else None)
        (folder/"run_config.json").write_text(json.dumps(c))
        (folder/"summary.json").write_text(json.dumps(s))
        folders.append(folder)
    result = compare(folders,tmp_path/"report")
    assert result["provisional_mask_backbone"] == "patchecg"
    assert result["status"] == "provisional_validation_ranking"
    c["source_manifest_sha256"] = "old-40hz"
    (folders[-1]/"run_config.json").write_text(json.dumps(c))
    with pytest.raises(ValueError, match="source_manifest"):
        compare(folders,tmp_path/"bad")
    s["eligible_for_model_selection"] = False
    (folders[-1]/"summary.json").write_text(json.dumps(s))
    with pytest.raises(ValueError, match="five-class pilot"):
        compare(folders,tmp_path/"profile")
