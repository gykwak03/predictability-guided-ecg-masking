"""Scientific matching, official pooling parity, and interrupted-training checks."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
import torch

from experiments.backbone_comparison import (analyze, configuration, create_study,
    probe_cohort, read_study, run_seed)
from experiments.backbone_models import build_backbone
from experiments.phase1.config import ROOT
from experiments.phase1.data import get_data, synthetic_data
from experiments.phase1.masks import per_lead_masks
from experiments.phase1.pipeline import bank_at
from experiments.phase1.training import load_checkpoint, seed_all, train_model


@pytest.fixture
def cfg(tmp_path):
    torch.set_num_threads(2)
    return configuration(overrides=dict(output_root=str(tmp_path / 'out'),
        st_mem_repo=os.environ.get('ECG_ST_MEM_REPO', str(ROOT.parent / 'ST-MEM'))), synthetic=True)


def require_upstream(cfg):
    if not (Path(cfg['st_mem_repo']) / 'models/st_mem.py').is_file():
        pytest.skip('Official ST-MEM checkout not available; set ECG_ST_MEM_REPO')


def test_immutable_plan_and_update_units(cfg, tmp_path):
    require_upstream(cfg)
    root = create_study(cfg, tmp_path / 'study')
    plan, loaded = read_study(root)
    assert loaded['milestones'] == [0, 1, 2]
    assert plan['comparison_unit'].startswith('optimizer updates (not epochs)')
    assert not plan['pretrained_weights'] and not plan['equal_compute']
    plan['config']['updates'] = 3000
    (root / 'plan.json').write_text(json.dumps(plan))
    with pytest.raises(ValueError, match='Plan was edited'):
        read_study(root)
    with pytest.raises(ValueError, match='P50'):
        configuration(overrides=dict(patch_samples=125), synthetic=True)


def test_stmem_hidden_isolation_and_official_pooling_parity(cfg):
    require_upstream(cfg)
    seed_all(42)
    model = build_backbone('stmem', cfg, cfg['st_mem_repo']).eval()
    x = torch.from_numpy(synthetic_data()['arrays']['train'][:2])
    mask = torch.from_numpy(per_lead_masks(2, 50, .5, 99))
    changed = x.clone()
    hidden = mask.repeat_interleave(50, 1).reshape_as(x)
    changed[hidden] += 10000
    with torch.no_grad():
        assert torch.equal(model(x, mask)[0], model(changed, mask)[0])
        assert torch.allclose(model.features(x, 'mean_then_norm'),
                              model.core.encoder.forward_encoding(x), atol=2e-6, rtol=2e-6)
        assert model.features(x, 'norm_then_mean').shape == (2, 32)
    invalid = mask.clone()
    invalid[0, :50] = False
    with pytest.raises(ValueError, match='equal nonzero visible counts'):
        model(x, invalid)


def test_shared_lr_prefix_and_exact_resume(cfg, tmp_path):
    cfg = dict(cfg, updates=3, checkpoint_every=1, eval_every=1)
    data = get_data(cfg)
    banks = {'validation': bank_at(tmp_path / 'banks', data, cfg, 'validation', pilot=True)}
    kwargs = dict(milestone_steps=[0, 1, 3])
    seed_all(5)
    train_model(build_backbone('plain', cfg, cfg['st_mem_repo']), 'mae', data, cfg,
                tmp_path / 'full', {'match': True}, banks, **kwargs)
    seed_all(5)
    train_model(build_backbone('plain', cfg, cfg['st_mem_repo']), 'mae', data, cfg,
                tmp_path / 'split', {'match': True}, banks, stop_after=1, **kwargs)
    first = load_checkpoint(tmp_path / 'full/milestones/update_000001.pt')
    split = load_checkpoint(tmp_path / 'split/checkpoint.pt')
    assert first['contract']['horizon'] == split['contract']['horizon'] == 3
    assert first['sequence'] == split['sequence']
    for key in first['model']:
        assert torch.equal(first['model'][key], split['model'][key]), key
    seed_all(999)
    train_model(build_backbone('plain', cfg, cfg['st_mem_repo']), 'mae', data, cfg,
                tmp_path / 'split', {'match': True}, banks, resume=True, **kwargs)
    full = load_checkpoint(tmp_path / 'full/checkpoint.pt')
    resumed = load_checkpoint(tmp_path / 'split/checkpoint.pt')
    assert full['sequence'] == resumed['sequence'] and full['next_batch'] == resumed['next_batch']
    for key in full['model']:
        assert torch.equal(full['model'][key], resumed['model'][key]), key


def test_paired_pipeline_and_reject_unmatched_results(cfg, tmp_path):
    require_upstream(cfg)
    root = create_study(cfg, tmp_path / 'study')
    # Both arms first stop, then continue from their optimizer/RNG checkpoint.
    assert not run_seed(root, 42, stop_after=1)
    assert not (root / 'seed42/plain/completed.json').exists()
    assert run_seed(root, 42)
    output = analyze(root)
    audit = json.loads((output / 'audit.json').read_text())
    assert audit['status'] == 'completed' and audit['observed_checkpoints'] == 6
    assert audit['synthetic'] and not audit['test_opened']
    # Completed work is not retrained on another invocation.
    path = root / 'seed42/plain/checkpoint.pt'
    before = path.stat().st_mtime_ns
    assert run_seed(root, 42)
    assert path.stat().st_mtime_ns == before
    value_path = root / 'seed42/stmem/evaluations/update_000002/completed.json'
    value = json.loads(value_path.read_text())
    value['reconstruction']['sequence_sha256'] = 'wrong-mask-or-ECG-stream'
    value_path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='Unmatched pretraining arms'):
        analyze(root)


def test_probe_labels_are_separate_and_aligned(cfg):
    data = get_data(cfg)
    cohort, labels = probe_cohort(data, cfg)
    assert set(cohort['train']) <= set(data['selected']['pilot'])
    assert len(cohort['validation']) == len(data['frames']['validation'])
    assert labels['train'].shape[1] == labels['validation'].shape[1] == 5


def test_sbatch_serial_array_and_paths_with_spaces(tmp_path):
    bash = shutil.which('bash')
    git_bash = Path('C:/Program Files/Git/bin/bash.exe')
    if os.name == 'nt' and git_bash.exists():
        bash = str(git_bash)
    if bash is None:
        pytest.skip('Bash unavailable')
    subprocess.run([bash, '-n', str(ROOT / 'scripts/submit_backbone_comparison.sh')], check=True)
    subprocess.run([bash, '-n', str(ROOT / 'scripts/backbone_comparison_worker.sh')], check=True)
    # A fake Python and fake sbatch record arguments; no scheduler/network call.
    stub = tmp_path / 'bin'
    stub.mkdir()
    python = stub / 'fake-python.sh'
    python.write_text('#!/usr/bin/env bash\nprintf "3\\n"\n', newline='\n')
    python.chmod(0o755)
    sbatch = stub / 'sbatch'
    sbatch.write_text('#!/usr/bin/env bash\nprintf "ARG:%s\\n" "$@"\n', newline='\n')
    sbatch.chmod(0o755)
    study = tmp_path / 'study with spaces'
    study.mkdir()
    (study / 'plan.json').write_text('{}')
    env = dict(os.environ, ECG_REPO=ROOT.as_posix(), ECG_PYTHON=python.as_posix(),
        ECG_LOG_DIR=(tmp_path / 'logs').as_posix(), ECG_JOB_HOURS='24',
        ECG_PARTITION='batch_ugrad_advisor_x')
    # A Windows drive colon cannot be inserted directly into POSIX PATH. Keep
    # Git's bash ahead of WindowsApps/bash.exe (a different, WSL launcher).
    stub_path = ('/' + stub.drive[0].lower() + stub.as_posix()[2:]) if os.name == 'nt' else stub.as_posix()
    script = 'export PATH="'+stub_path+':/usr/bin:/bin:$PATH"\nbash "$1" "$2"'
    result = subprocess.run([bash, '-c', script, 'test',
        (ROOT / 'scripts/submit_backbone_comparison.sh').as_posix(), study.as_posix()],
        env=env, text=True, capture_output=True, check=True)
    assert 'ARG:--array=0-2%1' in result.stdout
    assert 'ARG:--gres=gpu:normal:1' in result.stdout
    assert '--nodelist' not in result.stdout
    assert result.stdout.rstrip().endswith('/study with spaces')
