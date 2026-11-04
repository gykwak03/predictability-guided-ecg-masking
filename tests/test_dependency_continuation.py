"""Checkpoint reuse, exact Random continuation, factorial controls and resume."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import experiments.dependency_continuation as module
from experiments.dependency_gain import choose_data, run_experiment
from experiments.phase1.config import resolve_config
from experiments.phase1.masks import per_lead_masks, seed_for
from experiments.phase1.model import build_mae
from experiments.phase1.training import load_checkpoint, restore_rng
from experiments.run_dependency_continuation import cases_for, parser, resolve


@pytest.fixture(scope='module')
def source(tmp_path_factory):
    torch.set_num_threads(2)
    root = tmp_path_factory.mktemp('continuation-source')/'original'
    cfg = resolve_config(smoke=True,environ={})
    cfg.update(validation_ecgs=2)
    options = dict(seeds=[42],checkpoints=[1,2],fork_updates=1,fork_lr=1e-4,fork_loss='anchor',
                   probe_ecgs=2,panel_repeats=1,radius=2,save_every=1)
    run_experiment(cfg,options,root)
    return root


def resolved(source,output,extra=()):
    return resolve(parser().parse_args(['--source-run',str(source),'--output',str(output),
        '--smoke','--checkpoints','1','--updates','1',*extra]))


def test_primary_and_full_factorial_cases():
    assert [c['case'] for c in cases_for(['all-hidden'])]==['D','E','F']
    cases = cases_for(['all-hidden','anchor'])
    assert {c['case'] for c in cases}==set('ABCDEFG')
    assert not any(c['arm']=='mae-random' and c['loss']=='anchor' for c in cases)
    assert len(cases)*2==14


def test_plan_never_imports_torch_or_pretrains(source,tmp_path):
    code = ('import sys; sys.modules["torch"]=None; '
        'from experiments.run_dependency_continuation import main; main(sys.argv[1:])')
    result = subprocess.run([sys.executable,'-c',code,'--source-run',str(source),'--output',str(tmp_path/'out'),
        '--smoke','--plan'],capture_output=True,text=True,check=True,cwd=ROOT)
    plan = json.loads(result.stdout)
    assert plan['pretraining_updates']==0
    assert plan['continuation_updates']==24  # 2 source checkpoints x 3 cases x 2 modes x 2 updates
    assert not (tmp_path/'out').exists()


def test_source_guards(source,tmp_path):
    with pytest.raises(ValueError,match='separate'):
        resolved(source,source/'new')
    with pytest.raises(ValueError,match='subset'):
        resolved(source,tmp_path/'out',['--checkpoints','100'])
    with pytest.raises(ValueError,match='constant'):
        resolved(source,tmp_path/'out',['--lr','0.001'])
    with pytest.raises(ValueError,match='requires'):
        resolved(source,tmp_path/'out',['--lr-policy','constant'])
    with pytest.raises(ValueError,match='Synthetic'):
        resolve(parser().parse_args(['--source-run',str(source),'--output',str(tmp_path/'out')]))


def test_exact_random_masks_optimizer_and_loss_match_original_training(source,tmp_path):
    src,out,cfg,saved,options = resolved(source,tmp_path/'out')
    data = choose_data(cfg,saved['options'])
    local = dict(cfg,seed=42)
    state = load_checkpoint(source/'seed_42/pretrain/update_1.pt')
    expected = load_checkpoint(source/'seed_42/pretrain/update_2.pt')
    before_state = copy.deepcopy(state['optimizer'])
    x,mask,targets,flow = module.continuation_batch(data,local,1,1,'mae-random',2)
    assert np.array_equal(mask,per_lead_masks(len(x),50,.5,seed_for(local['mask_seed'],42,'pretrain',2)))
    model = build_mae(local,50)
    model.load_state_dict(state['model'])
    optimizer = module.prepare_optimizer(model,state,local,'keep')
    assert len(optimizer.state)>0
    lr = module.continuation_lr(1,1,local,saved,options)
    assert lr==local['min_lr']  # Original horizon ends at 2, no warmup restart.
    for g in optimizer.param_groups:
        g['lr']=lr
    restore_rng(state['rng'])
    module.optimize(model,optimizer,x,mask,targets,local,np.ones(12),torch.device('cpu'),'all-hidden')
    assert all(torch.equal(v,expected['model'][k]) for k,v in model.state_dict().items())
    # Loading/updating a keep branch must not mutate the shared base optimizer tensors.
    for key,value in before_state['state'].items():
        for name,tensor in value.items():
            if isinstance(tensor,torch.Tensor):
                assert torch.equal(tensor,state['optimizer']['state'][key][name])
    assert not module.prepare_optimizer(model,state,local,'reset').state


def test_full_loss_backpropagates_all_hidden_patches():
    from experiments.dependency_gain import hidden_losses
    x = torch.zeros((1,12,2500))
    prediction = torch.ones((1,600,50),requires_grad=True)
    mask = torch.from_numpy(per_lead_masks(1,50,.5,7))
    hidden_losses(prediction,x,mask,torch.ones(12)).mean().backward()
    assert torch.count_nonzero(prediction.grad)==300*50
    assert not prediction.grad[~mask].any()


def test_factorial_end_to_end_does_not_change_source_and_resumes(source,tmp_path):
    src,out,cfg,saved,options = resolved(source,tmp_path/'full',['--losses','all-hidden','anchor'])
    hashes = {str(p.relative_to(source)):module.file_sha(p) for p in source.rglob('*') if p.is_file()}
    root = module.run_continuation(src,out,cfg,saved,options)
    summary = pd.read_csv(root/'summary.csv')
    assert len(summary)==14
    assert set(summary.case)==set('ABCDEFG')
    assert np.isfinite(summary.improvement).all()
    base = root/'seed_42/base_1'
    assert json.loads((base/'no_update_control.json').read_text())['max_abs_mse_difference']==0
    audit = json.loads((base/'random_replay_audit.json').read_text())
    assert audit['compared_updates']==1 and audit['mean_abs_training_loss_difference']==0
    initial_keep = json.loads((base/'all-hidden/keep/mae-random/initial_state.json').read_text())
    initial_reset = json.loads((base/'all-hidden/reset/mae-random/initial_state.json').read_text())
    assert initial_keep['optimizer_step_min']==1 and initial_reset['optimizer_state_entries']==0
    assert json.loads((root/'compute_cost.json').read_text())['pretraining_updates']==0
    contexts = pd.read_csv(root/'context_summary.csv')
    assert set(contexts.condition)=={'w0_c0','w1_c0','w0_c1','w1_c1','random'}
    contrasts = pd.read_csv(root/'contrasts.csv')
    assert 'paired_random_full_vs_anchor' in set(contrasts.contrast)
    previous = (root/'summary.csv').read_bytes()
    module.run_continuation(src,out,cfg,saved,options,resume=True)
    assert previous==(root/'summary.csv').read_bytes()
    assert hashes=={str(p.relative_to(source)):module.file_sha(p) for p in source.rglob('*') if p.is_file()}
    with pytest.raises(FileExistsError):
        module.run_continuation(src,out,cfg,saved,options)
    with pytest.raises(ValueError,match='changed'):
        module.run_continuation(src,out,cfg,saved,dict(options,updates=3),resume=True)


def test_interrupted_continuation_matches_uninterrupted(source,tmp_path,monkeypatch):
    src,out,cfg,saved,options = resolved(source,tmp_path/'interrupted',['--optimizer-modes','keep'])
    options.update(updates=2,eval_every=1,save_every=1)
    original,calls = module.optimize,[]
    def interrupt(*args,**kwargs):
        calls.append(None)
        if len(calls)==2:
            raise RuntimeError('interrupt after first saved continuation update')
        return original(*args,**kwargs)
    monkeypatch.setattr(module,'optimize',interrupt)
    with pytest.raises(RuntimeError,match='interrupt'):
        module.run_continuation(src,out,cfg,saved,options)
    monkeypatch.setattr(module,'optimize',original)
    module.run_continuation(src,out,cfg,saved,options,resume=True)
    clean = module.run_continuation(src,tmp_path/'clean',cfg,saved,options)
    pd.testing.assert_frame_equal(pd.read_csv(out/'summary.csv'),pd.read_csv(clean/'summary.csv'))
    for arm in ('mae-random','within','cross'):
        first = load_checkpoint(out/f'seed_42/base_1/all-hidden/keep/{arm}/latest.pt')['model']
        second = load_checkpoint(clean/f'seed_42/base_1/all-hidden/keep/{arm}/latest.pt')['model']
        assert all(torch.equal(first[k],second[k]) for k in first)


def test_learning_rate_does_not_restart_and_constant_is_explicit(source,tmp_path):
    src,out,cfg,saved,options = resolved(source,tmp_path/'lr')
    assert module.continuation_lr(2,100,cfg,saved,options)==cfg['min_lr']
    assert module.continuation_lr(2,100,cfg,saved,dict(options,lr_policy='constant',lr=1e-4))==1e-4


def test_changed_artifact_identity_is_rejected_before_training(source,tmp_path,monkeypatch):
    src,out,cfg,saved,options = resolved(source,tmp_path/'bad-data')
    original = module.choose_data
    def changed(*args):
        return dict(original(*args),data_id='different-artifact')
    monkeypatch.setattr(module,'choose_data',changed)
    with pytest.raises(ValueError,match='artifact/patient'):
        module.run_continuation(src,out,cfg,saved,options)
    assert not out.exists()
