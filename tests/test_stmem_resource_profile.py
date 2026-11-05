import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timedelta

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('stmem_profile', ROOT/'experiments/profile_stmem_base.py')
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


def test_real_batch_search_respects_oom_and_headroom():
    called = []
    def probe(size):
        called.append(size)
        return dict(oom=size>=10, peak_reserved_bytes=size*100 if size<10 else None)
    chosen, rows = profile.search_batch(probe, 32, 850)
    assert chosen == 8
    assert any(row['oom'] for row in rows)
    assert not next(row for row in rows if row['batch_size']==9)['safe']
    assert max(called)<=32
    assert all(row['safe']==(row['batch_size']<=8) for row in rows)
    assert profile.search_batch(probe, 3, 850)[0] == 3
    with pytest.raises(RuntimeError, match='batch=1'):
        profile.search_batch(probe, 32, 50)


def test_telemetry_preserves_low_samples_and_fixed_interval(tmp_path):
    begin = datetime(2026,11,5,12)
    lines=['timestamp, name, utilization.gpu [%], memory.used [MiB]']
    for i in range(14):
        when=(begin+timedelta(seconds=i)).strftime('%Y/%m/%d %H:%M:%S.%f')[:-3]
        lines.append(f'{when}, NVIDIA RTX A2000, {0 if i==5 else 99} %, 5000 MiB')
    lines.insert(8,lines[0])  # Repeated nvidia-smi headers remain in raw CSV.
    path=tmp_path/'raw.csv'; path.write_text('\n'.join(lines),encoding='utf-8')
    rows=profile.parse_telemetry(path,6000,begin+timedelta(seconds=1),begin+timedelta(seconds=12))
    assert len(rows)==12
    stats=profile.telemetry_statistics(rows)
    assert stats['at_least_10_seconds']
    assert stats['span_seconds']==11
    assert stats['gpu_utilization_percent']['minimum']==0
    assert stats['gpu_utilization_percent']['mean']<99
    assert stats['vram_percent']['mean']==pytest.approx(100*5000/6000)
    assert not profile.telemetry_statistics(rows[:10])['at_least_10_seconds']
    assert not profile.telemetry_statistics([])['at_least_10_seconds']


def test_plan_does_not_import_torch_or_open_data(tmp_path):
    code="""
import runpy, sys
class BlockTorch:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in ('torch', 'numpy', 'pandas'):
            raise RuntimeError('Heavy import in login-node plan: '+fullname)
sys.meta_path.insert(0, BlockTorch())
sys.argv=[sys.argv[1], '--output', sys.argv[2], '--artifact-dir', 'missing-data', '--plan']
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    out=tmp_path/'not-created'
    result=subprocess.run([sys.executable,'-c',code,str(ROOT/'experiments/profile_stmem_base.py'),str(out)],
                          check=True,capture_output=True,text=True)
    plan=json.loads(result.stdout)
    assert plan['architecture']['depth']==12
    assert plan['architecture']['embed_dim']==768
    assert plan['architecture']['patch_size']==75
    assert not out.exists()


def test_assigned_gpu_uses_allocated_uuid(monkeypatch):
    import torch
    from types import SimpleNamespace
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','1')
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda _: SimpleNamespace(uuid='GPU-allocated'))
    monkeypatch.setattr(profile.subprocess,'run',lambda *a,**kw: SimpleNamespace(
        stdout='0, GPU-other, GPU zero, 12000\n1, GPU-allocated, GPU one, 6000\n'))
    assert profile.assigned_gpu('cuda')['uuid']=='GPU-allocated'
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0,1')
    with pytest.raises(RuntimeError,match='exactly one'):
        profile.assigned_gpu('cuda')


def test_real_profile_rejects_login_node_before_creating_results(tmp_path,monkeypatch):
    monkeypatch.delenv('SLURM_JOB_ID',raising=False)
    cfg=profile.resolve(profile.parser().parse_args(['--output',str(tmp_path/'real')]))
    with pytest.raises(RuntimeError,match='Slurm CUDA'):
        profile.run(cfg)
    assert not Path(cfg['output']).exists()


def test_synthetic_cpu_training_is_not_application_evidence(tmp_path):
    import torch
    upstream=ROOT.parent/'ST-MEM'
    if not (upstream/'models/st_mem.py').exists():
        pytest.skip('Official upstream checkout required for raw ST_MEM class')
    cfg=profile.resolve(profile.parser().parse_args([
        '--output',str(tmp_path/'smoke'),'--st-mem-repo',str(upstream),'--smoke']))
    root=profile.run(cfg)
    status=json.loads((root/'status.json').read_text())
    assert status['status']=='completed'
    assert status['synthetic'] and not status['real_gpu_measurement']
    assert not status['application_threshold_evaluated']
    assert not (root/'gpu_raw.csv').exists()
    assert not (root/'resource_capture.png').exists()
    checkpoint=torch.load(root/'checkpoint.pt',weights_only=False,map_location='cpu')
    assert checkpoint['updates']==2
    assert checkpoint['optimizer']['state']
    assert checkpoint['metadata']['full_paper_reproduction'] is False
    assert checkpoint['metadata']['realized_mask_ratio']==pytest.approx(23/30)
    resources=json.loads((root/'resources.json').read_text())
    assert resources['successful_optimizer_updates']==2
