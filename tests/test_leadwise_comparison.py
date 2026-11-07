"""Capacity, decoder information flow, and paired evaluation checks."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
import torch

from experiments.leadwise_models import PlainLeadwiseMAE, build_model, count_parameters
from experiments.phase1.config import ROOT
from experiments.phase1.masks import per_lead_masks
from experiments.run_leadwise_comparison import MODELS, analyze, create_study, read_plan, run_seed


@pytest.fixture
def upstream():
    path = ROOT.parent / 'ST-MEM'
    if not (path / 'models/st_mem.py').exists():
        pytest.skip('Official ST-MEM checkout needed')
    torch.set_num_threads(2)
    return path


@pytest.mark.parametrize('size,total,trainable', [
    ('small', 24_624_278, 24_590_998),
    ('base', 88_529_558, 88_476_310),
])
def test_real_capacity_matches_official(upstream, size, total, trainable):
    cfg = dict(synthetic=False, model_size='stmem-small', patch_samples=50, st_mem_repo=upstream)
    plain = build_model(f'plain_{size}', cfg)
    assert count_parameters(plain) == (total, trainable)
    architecture = plain.architecture
    del plain
    stmem = build_model(f'stmem_{size}', cfg)
    assert count_parameters(stmem) == (total, trainable)
    assert architecture == stmem.architecture


def test_plain_masks_hidden_values_and_only_encoder_mixes_leads():
    torch.set_num_threads(2)
    model = PlainLeadwiseMAE('smoke').eval()
    x = torch.randn(2, 12, 2500)
    mask = torch.from_numpy(per_lead_masks(2, 50, .5, 173))
    changed = x.clone()
    changed[mask.repeat_interleave(50, dim=1).reshape_as(x)] += 1000
    with torch.no_grad():
        before = model(x, mask)[0].reshape(2, 12, 50, 50)
        assert torch.equal(before, model(changed, mask)[0].reshape_as(before))
        # Perturb the first lead *after* encoder attention. It must affect only
        # that lead's decoding, even though the decoder weights are shared.
        def perturb(_layer, _inputs, output):
            y = output.clone()
            y[:, :27, 0] += 5  # 25 visible patches plus two SEP tokens.
            return y

        hook = model.encoder.register_forward_hook(perturb)
        after = model(x, mask)[0].reshape_as(before)
        hook.remove()
        assert not torch.allclose(before[:, 0], after[:, 0])
        assert torch.equal(before[:, 1:], after[:, 1:])
        # Altering visible input *before* attention can affect other leads.
        changed = x.clone()
        changed[:, 0] = changed[:, 0].roll(13, dims=-1)
        assert not torch.allclose(before[:, 1:], model(changed, mask)[0].reshape_as(before)[:, 1:])


def test_four_arm_smoke_resumes_and_audits_pairing(upstream, tmp_path):
    import pandas as pd

    study = tmp_path / 'study'
    create_study(study, None, upstream, synthetic=True)
    plan = read_plan(study)
    assert not run_seed(study, plan, MODELS[0], 42, stop_after=1)
    assert json.loads((analyze(study) / 'audit.json').read_text())['status'] == 'partial'
    for name in MODELS:
        assert run_seed(study, plan, name, 42)
    out = analyze(study)
    assert json.loads((out / 'audit.json').read_text()) == dict(
        status='completed', missing=[], synthetic=True, test_opened=False)
    final = pd.read_csv(out / 'final_by_seed.csv')
    assert final.backbone.tolist() == list(MODELS)
    assert np.isfinite(final[['pooled_nrmse','macro_auroc']].to_numpy()).all()
    assert len(pd.read_csv(out / 'paired_deltas_a_minus_b.csv')) == 4
    marker = study / MODELS[-1] / 'seed42' / 'completed.json'
    record = json.loads(marker.read_text())
    record['probes'][0]['head_stream_sha256'] = 'different_probe_batches'
    marker.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='Paired training/probe stream mismatch'):
        analyze(study)


def test_slurm_submission_requests_four_single_gpu_tasks(tmp_path):
    bash = 'C:/Program Files/Git/bin/bash.exe' if os.name == 'nt' else shutil.which('bash')
    if not bash or not Path(bash).exists():
        pytest.skip('Bash unavailable')
    submit = ROOT / 'scripts/submit_leadwise_comparison.sh'
    worker = ROOT / 'scripts/leadwise_comparison_worker.sh'
    for script in (submit, worker):
        subprocess.run([bash, '-n', script.as_posix()], check=True)
    python = tmp_path / 'fake-python.sh'
    python.write_text('''#!/usr/bin/env bash
if [[ "$2" == validate ]]; then
  exit 0
fi
printf '%s\\n' "$@"
''', newline='\n')
    python.chmod(0o755)
    sbatch = tmp_path / 'sbatch'
    sbatch.write_text('''#!/usr/bin/env bash
printf '%s\\n' "$@"
while [[ "$1" == --* ]]; do
  shift
done
[[ $# == 2 && -f "$1" && -d "$2" ]]
''', newline='\n')
    sbatch.chmod(0o755)
    study = tmp_path / 'study with spaces'
    study.mkdir()
    env = dict(os.environ, ECG_REPO=ROOT.as_posix(), ECG_PYTHON=python.as_posix(),
               ECG_LOG_DIR=(tmp_path / 'logs').as_posix(), ECG_JOB_HOURS='24',
               ECG_QOS='qos_gykwak03_2026_2')
    cmd = 'export PATH="$(cd -- "$1" && pwd):$PATH"; shift; bash "$@"'
    launch = [bash, '-c', cmd, 'test', tmp_path.as_posix(), submit.as_posix(), study.as_posix()]
    result = subprocess.run(launch, env=env, capture_output=True, text=True, check=True).stdout
    assert {'--array=0-3%4', '--gres=gpu:normal:1', '--qos=qos_gykwak03_2026_2'} <= set(result.splitlines())
    assert result.rstrip().endswith('/study with spaces')
    single = subprocess.run(launch + ['3'], env=env, capture_output=True, text=True, check=True)
    assert '--array=3' in single.stdout.splitlines()

    for filename, body in (('nvidia-smi', 'exit 0'), ('flock', 'shift 4\nexec "$@"')):
        stub = tmp_path / filename
        stub.write_text(f'#!/usr/bin/env bash\n{body}\n', newline='\n')
        stub.chmod(0o755)
    for task, model in enumerate(MODELS):
        launch_worker = [bash, '-c', cmd, 'test', tmp_path.as_posix(), worker.as_posix(), study.as_posix()]
        run_env = dict(env, SLURM_JOB_ID='1234', SLURM_ARRAY_TASK_ID=str(task))
        result = subprocess.run(launch_worker, env=run_env, capture_output=True, text=True, check=True)
        assert result.stdout.splitlines() == [
            f'job=1234 task={task} model={model}',
            '-u', 'experiments/run_leadwise_comparison.py', 'run',
            '--study-dir', study.as_posix(), '--model', model,
        ]
