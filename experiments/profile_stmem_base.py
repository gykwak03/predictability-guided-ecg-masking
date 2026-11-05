#!/usr/bin/env python3
"""Real ST-MEM Base training resource profile and unmodified nvidia-smi evidence."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--artifact-dir',default='/data/gykwak03/capstone2/ptbxl_250hz_bp067_50')
    p.add_argument('--st-mem-repo',default=str(ROOT.parent/'ST-MEM'))
    p.add_argument('--batch-size',default='auto',help='auto or a positive physical batch size')
    p.add_argument('--max-batch-size',type=int,default=128)
    p.add_argument('--headroom-mib',type=int,default=512)
    p.add_argument('--precision',choices=['amp','fp32'],default='amp')
    p.add_argument('--accumulation',type=int,default=1)
    p.add_argument('--train-ecgs',type=int,default=5000)
    p.add_argument('--measure-seconds',type=float,default=90)
    p.add_argument('--min-updates',type=int,default=20)
    p.add_argument('--warmup-updates',type=int,default=3)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--cpu-threads',type=int,default=4)
    p.add_argument('--smoke',action='store_true',help='Tiny synthetic CPU test; NEVER GPU application evidence')
    p.add_argument('--plan',action='store_true',help='Print plan without importing Torch or opening data')
    return p


def resolve(args):
    import math
    cfg = vars(args).copy()
    for key in ('output','artifact_dir','st_mem_repo'):
        cfg[key] = str(Path(cfg[key]).expanduser().resolve())
    for key in ('max_batch_size','accumulation','train_ecgs','min_updates','warmup_updates','cpu_threads'):
        if cfg[key]<1:
            raise ValueError(f'{key} must be positive')
    if cfg['headroom_mib']<0 or cfg['seed']<0 or not math.isfinite(cfg['measure_seconds']) or cfg['measure_seconds']<10:
        raise ValueError('Nonnegative headroom/seed and at least 10 finite measurement seconds required')
    if cfg['batch_size']!='auto':
        cfg['batch_size']=int(cfg['batch_size'])
        if cfg['batch_size']<1 or cfg['batch_size']>cfg['train_ecgs']:
            raise ValueError('Physical batch must fit the selected ECG pool')
    if cfg['smoke']:
        cfg.update(batch_size=2,train_ecgs=4,precision='fp32',warmup_updates=1,
                   measure_seconds=0,min_updates=2,cpu_threads=2)
    return cfg


def architecture(smoke=False):
    size = dict(embed_dim=32,depth=1,num_heads=2,decoder_embed_dim=32,decoder_depth=1,decoder_num_heads=2) if smoke else dict(
        embed_dim=768,depth=12,num_heads=12,decoder_embed_dim=256,decoder_depth=4,decoder_num_heads=4)
    return dict(seq_len=2250,patch_size=75,num_leads=12,norm_pix_loss=True,**size)


def search_batch(probe, maximum, budget_bytes):
    """Maximize REAL ECG batch size within the measured allocator budget; no padding."""
    trials, good, bad = [], 0, maximum+1
    candidate = 1
    while candidate<=maximum:
        row = dict(batch_size=candidate,**probe(candidate))
        row['safe'] = not row['oom'] and row.get('peak_reserved_bytes',float('inf'))<=budget_bytes
        trials.append(row)
        if row['safe']:
            good=candidate
            if candidate==maximum:
                break
            candidate=min(maximum,candidate*2)
        else:
            bad=candidate
            break
    while bad-good>1 and good<maximum:
        candidate=(good+bad)//2
        row=dict(batch_size=candidate,**probe(candidate))
        row['safe']=not row['oom'] and row.get('peak_reserved_bytes',float('inf'))<=budget_bytes
        trials.append(row)
        if row['safe']:
            good=candidate
        else:
            bad=candidate
    if not good:
        raise RuntimeError('Even batch=1 does not fit the real workload and configured headroom')
    return good,trials


def parse_telemetry(path,total_mib,start=None,end=None):
    """Retain ALL samples in the declared interval, including low utilization."""
    rows=[]
    for values in csv.reader(Path(path).read_text(encoding='utf-8').splitlines()):
        if len(values)!=4:
            continue
        try:
            timestamp=datetime.strptime(values[0].strip(),'%Y/%m/%d %H:%M:%S.%f')
            utilization=float(values[2].strip().rstrip('%').strip())
            memory=float(values[3].strip().replace('MiB','').strip())
        except ValueError:
            continue  # Header rows, preserved in the original CSV.
        if (start is not None and timestamp<start) or (end is not None and timestamp>end):
            continue
        rows.append(dict(timestamp=timestamp.isoformat(timespec='milliseconds'),gpu_name=values[1].strip(),
            gpu_utilization_percent=utilization,memory_used_mib=memory,memory_total_mib=total_mib,
            vram_percent=100*memory/total_mib))
    return rows


def telemetry_statistics(rows):
    import numpy as np
    if not rows:
        return dict(samples=0,span_seconds=0,at_least_10_seconds=False)
    span=(datetime.fromisoformat(rows[-1]['timestamp'])-datetime.fromisoformat(rows[0]['timestamp'])).total_seconds()
    result=dict(samples=len(rows),span_seconds=span,at_least_10_seconds=len(rows)>=11 and span>=10)
    for key in ('gpu_utilization_percent','memory_used_mib','vram_percent'):
        values=[r[key] for r in rows]
        result[key]=dict(mean=float(np.mean(values)),median=float(np.median(values)),minimum=min(values),maximum=max(values))
    return result


def assigned_gpu(device):
    import os
    import torch
    command=['nvidia-smi','--query-gpu=index,uuid,name,memory.total','--format=csv,noheader,nounits']
    output=subprocess.run(command,check=True,capture_output=True,text=True).stdout
    records=[dict(index=r[0].strip(),uuid=r[1].strip(),name=r[2].strip(),total_mib=float(r[3]))
             for r in csv.reader(output.splitlines()) if len(r)==4]
    reported=str(getattr(torch.cuda.get_device_properties(device),'uuid',''))
    selected=[r for r in records if r['uuid']==reported]
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    if len(visible)!=1 or not visible[0]:
        raise RuntimeError('This profile requires exactly one explicitly allocated CUDA_VISIBLE_DEVICES entry')
    if not selected:
        selected=[r for r in records if r['uuid']==visible[0] or r['index']==visible[0]]
    if not selected and len(records)==1:
        selected=records
    if len(selected)!=1:
        raise RuntimeError('Cannot unambiguously map the allocated Torch device to nvidia-smi; no node-wide capture')
    return selected[0]


def build_model(repo,smoke):
    import torch
    from functools import partial
    from experiments.backbone_models import adapter_module
    official=adapter_module().load_official_class(repo)
    return official(norm_layer=partial(torch.nn.LayerNorm,eps=1e-6),**architecture(smoke))


def batch(signals,size,seed,device):
    import numpy as np
    import torch
    rng=np.random.default_rng(seed)
    indices=rng.choice(len(signals),size,replace=False)
    starts=rng.integers(0,signals.shape[-1]-2250+1,size=size)
    x=np.stack([signals[i,:,s:s+2250] for i,s in zip(indices,starts)]).astype(np.float32)
    tensor=torch.from_numpy(x)
    if device.type=='cuda':
        tensor=tensor.pin_memory()
    return tensor.to(device,non_blocking=True)


def make_optimizer(model,physical,cfg):
    import torch
    # Paper Table 6: LR .0012 at effective batch 2048. Official example uses blr .00015 / 256.
    lr=.0012*(physical*cfg['accumulation'])/2048
    return torch.optim.AdamW(model.parameters(),lr=lr,betas=(.9,.95),weight_decay=.01)


def train_update(model,optimizer,scaler,signals,physical,cfg,device,step):
    import torch
    from experiments.phase1.masks import seed_for
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses=[]
    for micro in range(cfg['accumulation']):
        x=batch(signals,physical,seed_for(cfg['seed'],'profile',step,micro),device)
        with torch.autocast(device_type=device.type,dtype=torch.float16,
                            enabled=cfg['precision']=='amp' and device.type=='cuda'):
            output=model(x,mask_ratio=.75)
            loss=output['loss']
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite reconstruction loss')
        scaler.scale(loss/cfg['accumulation']).backward()
        losses.append(float(loss.detach()))
    old_scale=scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    return sum(losses)/len(losses),scaler.get_scale()>=old_scale


def render_capture(root,metadata,rows,stats):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    fig=plt.figure(figsize=(13,12))
    grid=GridSpec(4,1,figure=fig,height_ratios=[1.35,1.7,1.7,3],
                  left=.08,right=.98,bottom=.04,top=.96,hspace=.4)
    ax=fig.add_subplot(grid[0]); ax.axis('off')
    ax.text(0,1,'ST-MEM Base: measured training resource profile',fontsize=18,weight='bold',va='top')
    gpu=metadata['gpu']
    text=(f"GPU: {gpu['name']} | {gpu['total_mib']:.0f} MiB | Slurm job: {metadata['slurm_job_id']}\n"
        f"Parameters: {metadata['parameters']:,} | batch: {metadata['physical_batch']} x {metadata['accumulation']} | precision: {metadata['precision']}\n"
        f"Real PTB-XL | input 12 x 2250 | P75 | requested mask 75%, realized 76.67%\n"
        f"Steady-training interval: {stats['span_seconds']:.1f}s, {stats['samples']} samples; ALL interval samples included\n"
        'Short resource reproduction; existing 0.67-50Hz / per-lead z artifact. Not full-paper accuracy reproduction.')
    ax.text(0,.72,text,fontsize=10,va='top',linespacing=1.5)
    times=[(datetime.fromisoformat(r['timestamp'])-datetime.fromisoformat(rows[0]['timestamp'])).total_seconds() for r in rows]
    for i,key,label in [(1,'gpu_utilization_percent','GPU utilization (%)'),(2,'vram_percent','VRAM used / total (%)')]:
        ax=fig.add_subplot(grid[i]); ax.plot(times,[r[key] for r in rows],linewidth=1.5)
        ax.set(ylim=(0,105),ylabel=label,xlabel='Seconds from first steady-training sample')
        ax.grid(alpha=.25)
        s=stats[key]
        ax.set_title(f"Mean {s['mean']:.1f}% | median {s['median']:.1f}% | min {s['minimum']:.1f}% | max {s['maximum']:.1f}%",fontsize=10)
    ax=fig.add_subplot(grid[3]); ax.axis('off')
    excerpt=rows[:15]  # First samples, never the highest-utilization samples.
    cells=[[r['timestamp'].replace('T',' '),r['gpu_name'],f"{r['gpu_utilization_percent']:.0f}%",
        f"{r['memory_used_mib']:.0f} MiB"] for r in excerpt]
    table=ax.table(cellText=cells,colLabels=['Timestamp (node local)','GPU','GPU util','VRAM used'],
                   bbox=[0,0,1,.9],cellLoc='left')
    table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1,1.2)
    ax.set_title('First 15 steady-training nvidia-smi records; source: gpu_raw.csv',fontsize=10,pad=12)
    fig.savefig(root/'resource_capture.png',dpi=160)
    plt.close(fig)


def run(cfg):
    import os
    import numpy as np
    import pandas as pd
    import torch
    from experiments.phase1.common import file_sha,write_json
    from experiments.phase1.data import load_artifact,synthetic_data,NORMALIZATION
    from experiments.phase1.training import atomic_checkpoint,seed_all,sync
    from experiments.backbone_models import adapter_module

    device=torch.device('cpu' if cfg['smoke'] else 'cuda')
    if not cfg['smoke'] and (not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available()):
        raise RuntimeError('Real profiling requires a Slurm CUDA allocation; never run on login node')
    torch.set_num_threads(cfg['cpu_threads'])
    root=Path(cfg['output']); root.mkdir(parents=True,exist_ok=False)
    monitor=raw=monitor_error=None
    write_json(root/'status.json',dict(status='running',synthetic=cfg['smoke'],test_opened=False))
    def terminate(_signum,_frame):
        raise KeyboardInterrupt('Slurm termination requested')
    old_handler=signal.signal(signal.SIGTERM,terminate)
    try:
        print(f'RUN_DIR={root}\nLoading real training artifact.' if not cfg['smoke'] else f'RUN_DIR={root}\nSYNTHETIC CPU CHECK ONLY',flush=True)
        data=synthetic_data() if cfg['smoke'] else load_artifact(cfg['artifact_dir'])
        if cfg['train_ecgs']>len(data['arrays']['train']):
            raise ValueError('Requested ECG cohort exceeds the training pool')
        indices=np.random.default_rng(cfg['seed']).choice(len(data['arrays']['train']),cfg['train_ecgs'],replace=False)
        signals=np.asarray(data['arrays']['train'][indices],dtype=np.float32)  # RAM preload avoids NAS stalls.
        data['frames']['train'].iloc[indices].to_csv(root/'training_ecgs.csv',index=False)
        seed_all(cfg['seed'])
        model=build_model(cfg['st_mem_repo'],cfg['smoke']).to(device)
        initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        gpu=assigned_gpu(device) if not cfg['smoke'] else None
        def scaler_for():
            return torch.amp.GradScaler('cuda',enabled=device.type=='cuda' and cfg['precision']=='amp',init_scale=1024.)
        def probe(size):
            print(f'Calibrating real training batch={size}',flush=True)
            model.load_state_dict(initial)
            seed_all(cfg['seed'])
            # A failed trial can release its last tensors only after unwinding.
            # Discard that cached allocation before measuring the next batch.
            torch.cuda.empty_cache()
            optimizer=make_optimizer(model,size,cfg); scaler=scaler_for()
            torch.cuda.reset_peak_memory_stats(device)
            started=time.perf_counter(); succeeded=0
            try:
                for attempt in range(10):
                    _,updated=train_update(model,optimizer,scaler,signals,size,cfg,device,attempt)
                    succeeded+=int(updated)
                    if succeeded>=2:
                        break
                if succeeded<2:
                    raise RuntimeError('AMP did not complete two optimizer steps during calibration')
                sync(device)
                return dict(oom=False,peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(device),seconds=time.perf_counter()-started)
            except torch.cuda.OutOfMemoryError:
                return dict(oom=True,peak_allocated_bytes=None,peak_reserved_bytes=None,seconds=time.perf_counter()-started)
            finally:
                optimizer.zero_grad(set_to_none=True)
                del optimizer,scaler
                torch.cuda.empty_cache()
        if cfg['batch_size']=='auto':
            free,total=torch.cuda.mem_get_info(device)
            other=total-free-torch.cuda.memory_reserved(device)
            budget=total-other-cfg['headroom_mib']*1024**2
            physical,trials=search_batch(probe,min(cfg['max_batch_size'],len(signals)),budget)
            pd.DataFrame(trials).to_csv(root/'batch_calibration.csv',index=False)
        else:
            physical=cfg['batch_size']; trials=[]
        model.load_state_dict(initial); seed_all(cfg['seed']); del initial
        optimizer=make_optimizer(model,physical,cfg); scaler=scaler_for()
        metadata=dict(config=cfg,model='synthetic-tiny-ST-MEM' if cfg['smoke'] else 'official-ST-MEM-Base',
            architecture=architecture(cfg['smoke']),parameters=sum(p.numel() for p in model.parameters()),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            physical_batch=physical,accumulation=cfg['accumulation'],effective_batch=physical*cfg['accumulation'],
            precision=cfg['precision'],lr=optimizer.param_groups[0]['lr'],gpu=gpu,
            slurm_job_id=os.environ.get('SLURM_JOB_ID'),python=sys.version,torch=str(torch.__version__),
            cuda=torch.version.cuda,upstream=adapter_module().source_metadata(cfg['st_mem_repo']),
            source_sha256=file_sha(__file__),data_id=data['data_id'],preprocessing=NORMALIZATION,
            requested_mask_ratio=.75,realized_mask_ratio=1-int(30*.25)/30,
            full_paper_reproduction=False,synthetic=cfg['smoke'],test_opened=False,
            paper_reference='https://arxiv.org/pdf/2402.09450v3#page=14',
            paper_gpu='NVIDIA GeForce RTX 3080 Ti',paper_epochs=800,paper_effective_batch=2048,
            paper_gpu_count=None,paper_wall_time_hours=None)
        write_json(root/'metadata.json',metadata)
        print(f"model={metadata['model']} parameters={metadata['parameters']:,} batch={physical} precision={cfg['precision']}",flush=True)
        if not cfg['smoke']:
            command=['nvidia-smi','--id='+gpu['uuid'],
                '--query-gpu=timestamp,name,utilization.gpu,memory.used','--format=csv','-l','1']
            write_json(root/'monitor_command.json',command)
            raw=(root/'gpu_raw.csv').open('w',encoding='utf-8')
            monitor_error=(root/'gpu_monitor.err').open('w',encoding='utf-8')
            monitor=subprocess.Popen(command,stdout=raw,stderr=monitor_error)
        for i in range(cfg['warmup_updates']):
            train_update(model,optimizer,scaler,signals,physical,cfg,device,-i-1)
        sync(device)
        if device.type=='cuda':
            torch.cuda.reset_peak_memory_stats(device)
        steady_start=datetime.now(); started=time.perf_counter(); updates=[]; successful=0; skipped=0
        print(f'Steady training started; capture at least {cfg["measure_seconds"]} seconds.',flush=True)
        while successful<cfg['min_updates'] or time.perf_counter()-started<cfg['measure_seconds']:
            mark=time.perf_counter()
            loss,updated=train_update(model,optimizer,scaler,signals,physical,cfg,device,len(updates))
            sync(device)
            successful+=int(updated)
            skipped=0 if updated else skipped+1
            if skipped>=10:
                raise RuntimeError('AMP skipped 10 consecutive optimizer steps; refusing a non-learning profile')
            updates.append(dict(update=len(updates)+1,loss=loss,seconds=time.perf_counter()-mark,optimizer_updated=updated))
            if len(updates)%10==0:
                print(f'update={len(updates)} loss={loss:.6f} elapsed={time.perf_counter()-started:.1f}s',flush=True)
            if monitor is not None and monitor.poll() is not None:
                raise RuntimeError('nvidia-smi monitor exited; inspect gpu_monitor.err')
        steady_end=datetime.now(); elapsed=time.perf_counter()-started
        if monitor is not None:
            monitor.terminate(); monitor.wait(timeout=10); monitor=None
            raw.close(); raw=None; monitor_error.close(); monitor_error=None
        pd.DataFrame(updates).to_csv(root/'updates.csv',index=False)
        write_json(root/'measurement_interval.json',dict(start_node_local=steady_start.isoformat(),
            end_node_local=steady_end.isoformat(),measured_utc=datetime.now(timezone.utc).isoformat(),
            warmup_updates=cfg['warmup_updates'],seconds=elapsed,calibration_included=False))
        resources=dict(measured_seconds=elapsed,updates=len(updates),successful_optimizer_updates=sum(r['optimizer_updated'] for r in updates),
            mean_seconds_per_update=float(np.mean([r['seconds'] for r in updates])),
            ecgs_per_second=len(updates)*physical*cfg['accumulation']/elapsed,
            torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0,
            torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else 0)
        write_json(root/'resources.json',resources)
        if not cfg['smoke']:
            rows=parse_telemetry(root/'gpu_raw.csv',gpu['total_mib'],steady_start,steady_end)
            stats=telemetry_statistics(rows)
            write_json(root/'gpu_summary.json',stats)
            if not stats['at_least_10_seconds']:
                raise RuntimeError('Fewer than 10 seconds of valid steady-training telemetry; raw logs retained')
            pd.DataFrame(rows).to_csv(root/'gpu_steady.csv',index=False)
            render_capture(root,metadata,rows,stats)
            print(json.dumps(stats,indent=2),flush=True)
        atomic_checkpoint(root/'checkpoint.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
            scaler=scaler.state_dict(),updates=len(updates),metadata=metadata))
        write_json(root/'status.json',dict(status='completed',synthetic=cfg['smoke'],
            real_gpu_measurement=not cfg['smoke'],application_threshold_evaluated=False,
            full_paper_reproduction=False,test_opened=False))
    except BaseException as error:
        write_json(root/'status.json',dict(status='failed',error=str(error),synthetic=cfg['smoke']))
        raise
    finally:
        if monitor is not None:
            monitor.terminate()
            try:
                monitor.wait(timeout=10)
            except subprocess.TimeoutExpired:
                monitor.kill(); monitor.wait()
        if raw is not None:
            raw.close()
        if monitor_error is not None:
            monitor_error.close()
        signal.signal(signal.SIGTERM,old_handler)
    return root


def main(argv=None):
    cfg=resolve(parser().parse_args(argv))
    if cfg['plan']:
        print(json.dumps(dict(config=cfg,architecture=architecture(cfg['smoke']),
            purpose='Real training workload profile; utilization is measured, never fabricated',
            reproduction='Architecture only; not the full 800-epoch paper experiment'),indent=2))
        return 0
    print('RUN_DIR='+str(run(cfg)),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
