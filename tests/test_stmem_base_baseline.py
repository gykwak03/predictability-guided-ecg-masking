"""Base capacity, shared data/mask flow, complete evaluations and exact resume."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch

from experiments import backbone_comparison as matched
from experiments.stmem_base_baseline import (CONFIG, NAME, analyze, build_model,
    configuration, create_study, read_study, run_seed)
from experiments.phase1.config import ROOT
from experiments.phase1.data import get_data
from experiments.phase1.masks import per_lead_masks
from experiments.phase1.training import batch_for, load_checkpoint, seed_all


@pytest.fixture
def cfg(tmp_path):
    torch.set_num_threads(2)
    repo = Path(os.environ.get('ECG_ST_MEM_REPO', str(ROOT.parent/'ST-MEM')))
    if not (repo/'models/st_mem.py').exists():
        pytest.skip('Official ST-MEM checkout needed')
    return configuration(overrides=dict(output_root=str(tmp_path/'out'),st_mem_repo=str(repo)),synthetic=True)


def test_real_defaults_match_small_plain_except_capacity_and_microbatch():
    base = configuration()
    reference = matched.configuration(ROOT/'configs/backbone_matched.json')
    differences = {key for key in base if base[key] != reference[key]}
    assert differences == {'model_size','microbatch','accumulation','output_root'}
    assert base['model_size']=='stmem-base'
    assert (base['microbatch'],base['accumulation'])==(1,32)
    assert base['updates']==3000 and base['milestones']==[0,1000,3000]
    with pytest.raises(ValueError,match='effective batch 32'):
        configuration(overrides=dict(microbatch=1,accumulation=8))


def test_microbatch_does_not_change_ecg_mask_stream(cfg):
    smaller = dict(cfg,microbatch=1,accumulation=4)
    a,b = get_data(cfg),get_data(smaller)
    assert a['selection_id']==b['selection_id']
    for step in (1,2,4):
        xa,ma,ia = batch_for(a,cfg,'mae',step)
        xb,mb,ib = batch_for(b,smaller,'mae',step)
        assert ia==ib
        np.testing.assert_array_equal(xa,xb)
        np.testing.assert_array_equal(ma,mb)
        assert np.all(ma.reshape(len(ma),12,50).sum(-1)==25)


def test_base_is_base_capacity(cfg):
    real = dict(cfg,synthetic=False,model_size='stmem-base')
    model = build_model(real)
    assert model.model_size=='base'
    assert model.architecture==dict(embed_dim=768,depth=12,num_heads=12,
        decoder_embed_dim=256,decoder_depth=4,decoder_num_heads=4)
    assert 80_000_000 < sum(p.numel() for p in model.parameters()) < 100_000_000
    assert model.core.norm_pix_loss is False
    assert model.seq_len==2500 and model.p==50 and model.n==600


def test_hidden_isolation_and_both_full_input_poolings(cfg):
    seed_all(42)
    model = build_model(cfg).eval()
    data = get_data(cfg)
    x = torch.from_numpy(data['arrays']['train'][:2])
    mask = torch.from_numpy(per_lead_masks(2,50,.5,123))
    changed = x.clone()
    changed[mask.repeat_interleave(50,1).reshape_as(x)] += 10000
    with torch.no_grad():
        assert torch.equal(model(x,mask)[0],model(changed,mask)[0])
        assert torch.allclose(model.features(x,'mean_then_norm'),
                              model.core.encoder.forward_encoding(x),atol=2e-6,rtol=2e-6)
        assert model.features(x,'norm_then_mean').shape==(2,32)


def test_complete_cpu_pipeline_exact_resume_and_idempotence(cfg,tmp_path):
    full = create_study(cfg,tmp_path/'full')
    assert run_seed(full,42)
    split = create_study(cfg,tmp_path/'split')
    assert not run_seed(split,42,stop_after=1)
    assert not (split/'seed42'/NAME/'completed.json').exists()
    assert run_seed(split,42)
    for field in ('sequence','next_batch'):
        assert load_checkpoint(full/'seed42'/NAME/'checkpoint.pt')[field] == load_checkpoint(split/'seed42'/NAME/'checkpoint.pt')[field]
    first = load_checkpoint(full/'seed42'/NAME/'checkpoint.pt')
    second = load_checkpoint(split/'seed42'/NAME/'checkpoint.pt')
    for key in first['model']:
        assert torch.equal(first['model'][key],second['model'][key]),key
    checkpoint=split/'seed42'/NAME/'checkpoint.pt'
    mtime=checkpoint.stat().st_mtime_ns
    assert run_seed(split,42) and checkpoint.stat().st_mtime_ns==mtime
    out=analyze(split)
    audit=json.loads((out/'audit.json').read_text())
    assert audit['status']=='completed' and audit['observed_checkpoints']==3
    assert audit['synthetic'] and not audit['test_opened']
    model_info=json.loads((split/'seed42'/NAME/'model.json').read_text())
    assert model_info['model_size']=='smoke' and not model_info['full_paper_reproduction']
    plan_path=split/'plan.json'
    plan=json.loads(plan_path.read_text()); plan['config']['updates']=3000
    plan_path.write_text(json.dumps(plan))
    with pytest.raises(ValueError,match='plan was edited'):
        read_study(split)


def test_plan_is_lightweight_and_does_not_train_or_load_ecgs(cfg,tmp_path):
    code="""
import runpy, sys
class BlockHeavy:
 def find_spec(self,fullname,*args):
  if fullname.split('.')[0] in ('torch','pandas'):
   raise RuntimeError('Heavy import during plan: '+fullname)
sys.meta_path.insert(0,BlockHeavy())
sys.argv=[sys.argv[1],'plan','--study-dir',sys.argv[2],'--st-mem-repo',sys.argv[3]]
runpy.run_path(sys.argv[0],run_name='__main__')
"""
    target=tmp_path/'plan-only'
    subprocess.run([sys.executable,'-c',code,str(ROOT/'experiments/run_stmem_base_baseline.py'),
        str(target),cfg['st_mem_repo']],check=True,capture_output=True,text=True)
    plan=json.loads((target/'plan.json').read_text())
    assert plan['backbone']==NAME and plan['config']['model_size']=='stmem-base'
    assert plan['config']['updates']==3000
    assert not list(target.glob('seed*'))


def test_real_run_guard_precedes_data_loading(cfg,tmp_path,monkeypatch):
    real = dict(cfg,synthetic=False,device='cuda',model_size='stmem-base')
    study=create_study(real,tmp_path/'guard')
    monkeypatch.delenv('SLURM_JOB_ID',raising=False)
    with pytest.raises(RuntimeError,match='allocated CUDA GPU|Slurm allocation'):
        run_seed(study,42)
    assert not (study/'seed42').exists()


def test_sbatch_single_base_serial_array_and_spaces(tmp_path):
    bash = str(Path('C:/Program Files/Git/bin/bash.exe')) if os.name=='nt' else shutil.which('bash')
    if not bash or not Path(bash).exists():
        pytest.skip('Bash unavailable')
    for script in ('submit_stmem_base_baseline.sh','stmem_base_baseline_worker.sh'):
        subprocess.run([bash,'-n',str(ROOT/'scripts'/script)],check=True)
    stub=tmp_path/'bin'; stub.mkdir()
    python=stub/'fake-python.sh'; python.write_text('#!/usr/bin/env bash\nprintf "3\\n"\n',newline='\n'); python.chmod(0o755)
    sbatch=stub/'sbatch'; sbatch.write_text('#!/usr/bin/env bash\nprintf "ARG:%s\\n" "$@"\n',newline='\n'); sbatch.chmod(0o755)
    study=tmp_path/'study with spaces'; study.mkdir(); (study/'plan.json').write_text('{}')
    env=dict(os.environ,ECG_REPO=ROOT.as_posix(),ECG_PYTHON=python.as_posix(),
        ECG_LOG_DIR=(tmp_path/'logs').as_posix(),ECG_JOB_HOURS='24',ECG_PARTITION='batch_ugrad_advisor_x')
    stub_path=('/'+stub.drive[0].lower()+stub.as_posix()[2:]) if os.name=='nt' else stub.as_posix()
    command='export PATH="'+stub_path+':/usr/bin:/bin:$PATH"\nbash "$1" "$2"'
    result=subprocess.run([bash,'-c',command,'test',str(ROOT/'scripts/submit_stmem_base_baseline.sh'),study.as_posix()],
        env=env,text=True,capture_output=True,check=True)
    assert 'ARG:--array=0-2%1' in result.stdout and 'ARG:--gres=gpu:normal:1' in result.stdout
    assert result.stdout.rstrip().endswith('/study with spaces')
