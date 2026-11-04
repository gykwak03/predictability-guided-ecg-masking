"""Scientific controls for the context/gain pilot; no real ECG claims from CPU tests."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.dependency_gain import (CONDITIONS, choose_data, context_masks, draw_contexts,
    evaluate_panel, gains, panel, run_experiment, target_losses, paired_effects)
from experiments.phase1.config import resolve_config
from experiments.phase1.model import build_mae


def settings(tmp_path):
    cfg = resolve_config(smoke=True,environ={})
    cfg.update(validation_ecgs=2)
    options = dict(seeds=[42],checkpoints=[1,2],fork_updates=1,fork_lr=1e-4,fork_loss='anchor',
                   probe_ecgs=2,panel_repeats=1,radius=2,save_every=1)
    torch.set_num_threads(2)
    return cfg, options


@pytest.mark.parametrize('radius',[1,2,12])
def test_identical_anchors_fixed_budget_and_context_constraints(radius):
    targets, masks = draw_contexts(24,173,radius)
    for condition,mask in masks.items():
        grid = mask.reshape(-1,12,50)
        assert mask.dtype == bool
        assert np.all(grid.sum(-1)==25)
        for b,(lead,t) in enumerate(targets):
            assert grid[b,lead,t]
            if condition != 'random':
                w,c = int(condition[1]),int(condition[4])
                neighbours = np.r_[t-radius:t,t+1:t+radius+1]
                assert np.all(grid[b,lead,neighbours] == (not w))
                assert np.all(grid[b,np.arange(12)!=lead,t] == (not c))
    again = draw_contexts(24,173,radius)[1]
    assert all(np.array_equal(masks[c],again[c]) for c in CONDITIONS)


def test_wrong_geometry_and_edge_targets_rejected():
    with pytest.raises(ValueError,match='P50'):
        context_masks(np.array([[0,4]]),np.zeros((1,12,20)))
    with pytest.raises(ValueError,match='neighbourhood'):
        context_masks(np.array([[0,0]]),np.zeros((1,12,50)))


def test_loss_uses_identical_anchor_only():
    prediction = torch.ones((2,600,50),requires_grad=True)
    x = torch.zeros((2,12,2500))
    targets = torch.tensor([[0,4],[11,45]])
    losses = target_losses(prediction,x,targets,torch.ones(12))
    assert torch.equal(losses,torch.ones(2))
    losses.mean().backward()
    grad = prediction.grad.reshape(2,12,50,50)
    assert torch.count_nonzero(grad)==100
    assert torch.count_nonzero(grad[0,0,4])==50
    assert torch.count_nonzero(grad[1,11,45])==50


def test_factorial_gains_signed_and_contrast_not_independent():
    frame = pd.DataFrame([dict(ecg_id=1,patient_id=2,repeat=0,lead=0,time=4,condition=c,anchor_mse=v)
        for c,v in zip(CONDITIONS,[9.,4.,6.,2.,5.])])
    row = gains(frame).iloc[0]
    assert row.gain_within==4.5 and row.gain_cross==2.5
    assert row.gain_contrast==row.w0_c1-row.w1_c0==2.
    assert row.interaction==-1.
    negative = frame.copy()
    negative.loc[negative.condition=='w1_c1','anchor_mse']=30.
    assert gains(negative).iloc[0].gain_within<0


def test_patient_separation_and_eval_does_not_change_rng_or_model_mode(tmp_path):
    cfg, options = settings(tmp_path)
    data = choose_data(cfg,options)
    train = data['frames']['train']
    assert not set(train.iloc[data['selected']['train']].patient_id) & set(train.iloc[data['selected']['gain_probe']].patient_id)
    model = build_mae(cfg,50).train()
    bank = panel(data,'gain_probe',options,cfg['mask_seed'])
    before = torch.get_rng_state().clone()
    result = evaluate_panel(model,data,bank,cfg,torch.device('cpu'))
    assert model.training
    assert torch.equal(before,torch.get_rng_state())
    assert len(result)==2*len(CONDITIONS)
    effects = paired_effects(result,result)
    assert not effects.anchor_improvement.any()


def test_end_to_end_resume_controls_and_synthetic_labelling(tmp_path):
    cfg, options = settings(tmp_path)
    root = run_experiment(cfg,options,tmp_path/'experiment')
    status = json.loads((root/'status.json').read_text())
    assert status['status']=='completed' and status['synthetic'] and not status['test_opened']
    table = pd.read_csv(root/'hypothesis_units.csv')
    assert len(table)==2 and np.isfinite(table.effect_contrast).all()
    association = json.loads((root/'association.json').read_text())
    assert not association['real_ecg_performance'] and not association['downstream_evaluated']
    for base in (root/'seed_42').glob('base_*'):
        flows = [pd.read_csv(base/a/'updates.csv') for a in ('within','cross','random')]
        assert len({f.batch_sha256.iloc[0] for f in flows})==1
        assert len({f.target_sha256.iloc[0] for f in flows})==1
        assert len({f.mask_sha256.iloc[0] for f in flows})==3
    previous = (root/'hypothesis_units.csv').read_bytes()
    run_experiment(cfg,options,root,resume=True)
    assert previous==(root/'hypothesis_units.csv').read_bytes()
    with pytest.raises(FileExistsError):
        run_experiment(cfg,options,root)
    with pytest.raises(ValueError,match='changed'):
        run_experiment(cfg,dict(options,fork_updates=2),root,resume=True)


def test_plan_does_not_require_torch_or_existing_data(tmp_path):
    import subprocess
    code = ('import sys; sys.modules["torch"]=None; '
            'from experiments.run_dependency_gain import main; '
            'main(["--output",sys.argv[1],"--artifact-dir",sys.argv[2],"--plan"])')
    result = subprocess.run([sys.executable,'-c',code,str(tmp_path/'out'),str(tmp_path/'missing')],
                            capture_output=True,text=True,check=True)
    plan = json.loads(result.stdout)
    assert plan['total_optimizer_updates']==14400
    assert not (tmp_path/'out').exists()


def test_interrupted_fork_resumes_to_identical_weights(tmp_path,monkeypatch):
    import experiments.dependency_gain as module
    from experiments.phase1.training import load_checkpoint
    cfg, options = settings(tmp_path)
    options.update(checkpoints=[1],fork_updates=2)
    original, calls = module.optimize, []
    def interrupt(*args,**kwargs):
        calls.append(None)
        if len(calls)==3:
            raise RuntimeError('simulated interruption after saved fork update')
        return original(*args,**kwargs)
    monkeypatch.setattr(module,'optimize',interrupt)
    with pytest.raises(RuntimeError,match='simulated interruption'):
        module.run_experiment(cfg,options,tmp_path/'interrupted')
    monkeypatch.setattr(module,'optimize',original)
    resumed = module.run_experiment(cfg,options,tmp_path/'interrupted',resume=True)
    complete = module.run_experiment(cfg,options,tmp_path/'uninterrupted')
    pd.testing.assert_frame_equal(pd.read_csv(resumed/'hypothesis_units.csv'),pd.read_csv(complete/'hypothesis_units.csv'))
    for arm in ('within','cross','random'):
        first = load_checkpoint(resumed/'seed_42'/'base_1'/arm/'latest.pt')['model']
        second = load_checkpoint(complete/'seed_42'/'base_1'/arm/'latest.pt')['model']
        assert all(torch.equal(first[k],second[k]) for k in first)
