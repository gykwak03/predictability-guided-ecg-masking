from argparse import Namespace
import json
from pathlib import Path
import numpy as np
import pytest
import torch

from experiments.wcb.common import METHODS,read_plan,read_json
from experiments.wcb.data import prepare,synthetic_split,Prepared
from experiments.wcb.masks import wave_groups,matched_masks,context_sets,limited_update,sigmoid
from experiments.wcb.training import run
from experiments.run_wcb import create_plan


@pytest.fixture(scope='module')
def study(tmp_path_factory):
    root=tmp_path_factory.mktemp('wcb')/'study'
    create_plan(Namespace(study_dir=root,config=None,smoke=True,methods=None,seeds=None,updates=2,artifact_dir=None))
    plan=read_plan(root); prepare(root,plan)
    return root,plan


@pytest.mark.parametrize('p',[10,25])
def test_matched_multi_target_masks_and_boundaries(p):
    record=synthetic_split('train')[3][0]; groups=wave_groups(record,p); q=2500//p
    rates=[np.array([.9,.1,.8]),np.array([.1,.9,.2])]
    answers=matched_masks(groups,q,rates,123,k=6,structured_fraction=1)
    assert answers[0][1]['targets']==answers[1][1]['targets']
    for m,s in answers:
        assert m.sum()==len(m)//2 and s['target_tokens']>0
        assert all(v is None or 0<=v<=1 for v in s['realized'])
    repeated=matched_masks(groups,q,rates,123,k=6,structured_fraction=1)
    for a,b in zip(answers,repeated): np.testing.assert_array_equal(a[0],b[0])


def test_single_wave_groups_disjoint_and_C_is_eleven_times_target():
    groups=wave_groups(synthetic_split('train')[3][0],10); target=groups[1]
    w,c,b=context_sets(target,groups,250)
    assert len(c)==11*len(target['ids'])
    assert len(w)+len(b)+len(target['ids'])==250
    assert not set(w)&set(b) and not set(w)&set(c) and not set(b)&set(c)


def test_rate_bound_and_no_annotation_fallback():
    old=np.array([-10.,0.,10.]); new=limited_update(old,np.array([20.,20.,-20.]),.05)
    assert np.max(np.abs(sigmoid(new)-sigmoid(old)))<=.050001
    m,s=matched_masks([],100,[np.ones(3)],9,structured_fraction=1)[0]
    assert m.sum()==600 and s['no_annotations']


def test_conditional_control_has_same_random_mixture_rule():
    groups=wave_groups(synthetic_split('train')[3][0],25)
    mask,stats=matched_masks(groups,100,[np.full(3,.5)],5,kind='conditional',structured_fraction=0)[0]
    assert stats['mode']=='random' and mask.sum()==600


def test_real_training_never_falls_back_to_cpu(monkeypatch):
    from experiments.wcb.training import device_for
    monkeypatch.delenv('SLURM_JOB_ID',raising=False)
    with pytest.raises(RuntimeError,match='Slurm'): device_for(dict(synthetic=False,device='cuda'))


def test_source_guard_rejects_drift(study,monkeypatch):
    import experiments.wcb.common as common
    monkeypatch.setattr(common,'sources',lambda:{'changed':'source'})
    with pytest.raises(ValueError,match='Source changed'): common.read_plan(study[0])


def test_holdout_and_separate_probe_banks(study):
    root,plan=study; d=Prepared(root,plan)
    fit=set(d.frame.iloc[d.selection['fit']].patient_id)
    control=set(d.frame.iloc[d.selection['control']].patient_id)
    assert not fit&control and not fit&set(d.val_frame.patient_id)
    random=np.load(root/'prepared/random_control.npz')
    for i in range(0,len(random['masks']),2):
        np.testing.assert_array_equal(random['masks'][i],~random['masks'][i+1])
    balanced=np.load(root/'prepared/balanced_control.npz')
    assert (balanced['masks'][balanced['anchors']]).all()
    assert len(np.unique(balanced['strata']))==12
    assert read_json(root/'prepared/completed.json')['labels_opened'] is False


@pytest.mark.parametrize('method',METHODS)
def test_end_to_end_all_methods(study,method):
    root,plan=study
    result=run(root,plan,method,42)
    assert result['complete'] and result['step']==2
    assert result['attempted_updates']==(4 if method in ('coordinate','spsa','random_best2') else 2)
    assert np.isfinite(result['validation_reconstruction']['loss'])
    assert 0<=result['linear_probe']['macro_auroc']<=1
    history=read_json(root/'runs'/method/'seed42/history.json')
    assert history[0]['step']==2  # first block already uses method policy, no random prefix
    assert history[0]['rewards']==history[0]['random_rewards']
    assert history[0]['balanced'] is not None
    if method=='reinforce': assert history[0]['center_before']!=history[0]['center_after']


def test_resume_matches_uninterrupted(study,tmp_path):
    root,original=study
    # A separate frozen plan with H=2, updates=4 makes two real blocks.
    other=tmp_path/'study'
    create_plan(Namespace(study_dir=other,config=None,smoke=True,methods=['spsa'],seeds=None,updates=4,artifact_dir=None))
    plan=read_plan(other); prepare(other,plan)
    full=other/'runs/spsa/seed42'
    run(other,plan,'spsa',42)
    resumed=other/'resumed'
    assert run(other,plan,'spsa',42,stop_after=2,folder=resumed)['complete'] is False
    run(other,plan,'spsa',42,folder=resumed)
    a=torch.load(full/'checkpoint.pt',map_location='cpu',weights_only=False)
    b=torch.load(resumed/'checkpoint.pt',map_location='cpu',weights_only=False)
    for key in a['training']['model']:
        torch.testing.assert_close(a['training']['model'][key],b['training']['model'][key],rtol=0,atol=0)
    assert a['x']==b['x'] and a['attempted']==b['attempted']
    assert [r['mask_hashes'] for r in a['history']]==[r['mask_hashes'] for r in b['history']]


def test_balanced_reward_is_separate_ablation(tmp_path):
    config=tmp_path/'balanced.json'; config.write_text(json.dumps({'reward':'balanced'}),encoding='utf-8')
    root=tmp_path/'balanced_study'
    create_plan(Namespace(study_dir=root,config=config,smoke=True,methods=['spsa'],seeds=None,updates=2,artifact_dir=None))
    plan=read_plan(root); prepare(root,plan); run(root,plan,'spsa',42)
    record=read_json(root/'runs/spsa/seed42/history.json')[0]
    assert record['reward_kind']=='balanced' and record['balanced'] is not None
    assert record['rewards']!=record['random_rewards']
