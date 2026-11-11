"""Compare 100/40 ms patches on TRAIN patients only, without model training."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,help='New empty directory')
    parser.add_argument('--ecgs',type=int,default=128)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--delineation-cache',type=Path,
                        help='Earlier audit run directory; verify source and patient identities before reusing boundaries')
    parser.add_argument('--examples',type=int,default=4)
    args=parser.parse_args()
    if args.ecgs<1 or args.workers<1 or args.examples<0:
        parser.error('Positive ecgs/workers and nonnegative examples required')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output directory is not empty; choose a new directory')
    import numpy as np
    import pandas as pd
    from experiments.patch_audit import (delineate_record,analyze_record,write_overview,
                                         cost_table,plot_example,LEADS)
    manifest_path=args.artifact_dir/'manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    config=manifest['config']
    if (config.get('target_sampling_rate')!=250 or config.get('bandpass_low')!=.67
            or config.get('bandpass_high')!=50 or config.get('normalization')!='per-record'
            or manifest.get('lead_order')!=LEADS):
        raise ValueError('Expected canonical 250 Hz / 0.67-50 Hz / per-record artifact')
    # Deliberate allowlist: do not read labels, validation, or test files.
    columns=['artifact_index','ecg_id','patient_id','strat_fold','split']
    meta=pd.read_csv(args.artifact_dir/'metadata_train.csv',usecols=columns)
    signals=np.load(args.artifact_dir/'signals_train.npy',mmap_mode='r',allow_pickle=False)
    if (signals.shape!=(len(meta),12,2500) or signals.dtype!=np.float32
            or meta[columns].isna().any().any() or not meta.ecg_id.is_unique
            or not np.array_equal(meta.artifact_index,np.arange(len(meta)))
            or not meta.strat_fold.isin(range(1,9)).all() or not meta.split.eq('train').all()):
        raise ValueError('Invalid train array/metadata identity or train folds')
    rng=np.random.default_rng(args.seed)
    patients=meta.patient_id.unique()
    if args.ecgs>len(patients):
        raise ValueError('Requested more ECGs than unique train patients')
    chosen=rng.choice(patients,args.ecgs,replace=False)
    selection=meta.loc[[rng.choice(meta.index[meta.patient_id.eq(p)]) for p in chosen]].copy()
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'delineations').mkdir()
    (args.output/'matplotlib-cache').mkdir()
    os.environ['MPLCONFIGDIR']=str((args.output/'matplotlib-cache').resolve())
    selection.to_csv(args.output/'selection.csv',index=False)
    cached_results=[]
    cache_hashes={}
    if args.delineation_cache:
        cached_manifest=json.loads((args.delineation_cache/'manifest.json').read_text(encoding='utf-8'))
        if (cached_manifest.get('manifest_sha256')!=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
                or cached_manifest.get('source')!=str(args.artifact_dir.resolve())
                or cached_manifest.get('synthetic') is not False):
            raise ValueError('Delineation cache source mismatch')
        cached_selection=pd.read_csv(args.delineation_cache/'selection.csv',usecols=columns)
        for row in selection.itertuples():
            identity=cached_selection[cached_selection.ecg_id.eq(row.ecg_id)]
            if (len(identity)!=1 or int(identity.iloc[0].artifact_index)!=row.artifact_index
                    or identity.iloc[0].patient_id!=row.patient_id):
                raise ValueError('Delineation cache patient/ECG identity mismatch')
            cache_path=args.delineation_cache/'delineations'/f'{int(row.ecg_id)}.json'
            result=json.loads(cache_path.read_text(encoding='utf-8'))
            if (result['ecg_id']!=int(row.ecg_id)
                    or sorted(s['lead'] for s in result['status'])!=sorted(LEADS)
                    or any(r['ecg_id']!=int(row.ecg_id) or r['lead'] not in LEADS for r in result['intervals'])):
                raise ValueError('Malformed delineation cache identity')
            cache_hashes[cache_path.name]=hashlib.sha256(cache_path.read_bytes()).hexdigest()
            cached_results.append(result)
    provenance=dict(schema=2,analysis='whole-wave fixed-grid cover; no purity selection',
        training_executed=False,test_opened=False,validation_opened=False,
        synthetic=False,source=str(args.artifact_dir.resolve()),seed=args.seed,
        ecgs=args.ecgs,selection='one random ECG per sampled train patient',
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        selection_sha256=hashlib.sha256((args.output/'selection.csv').read_bytes()).hexdigest(),
        patches=[25,10],purity_filter=False,
        delineation_cache=str(args.delineation_cache.resolve()) if args.delineation_cache else None,
        annotation_sha256=cache_hashes,
        packages={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','neurokit2']},
        code_sha256={p:hashlib.sha256((Path(__file__).parent/p).read_bytes()).hexdigest()
                     for p in ['patch_audit.py','run_patch_audit.py']},
        delineation='NeuroKit DWT on cleaned per-lead z signals; shared R peaks from II, fallback V2/I',
        B_rule='union of full same-phase covers in other valid beats; exclude entire groups sharing target tokens; deduplicated',
        caveats=['Detected intervals are estimates, not expert annotations.',
                 'Target counts include every valid detected wave, regardless of its patch overlap fractions.',
                 'Padding can contain another waveform or rejected boundaries; these are reported, not silently filtered.',
                 'Unassigned samples include true background AND missing detections.',
                 'Counts describe one anchor at a time, not multi-anchor mask feasibility.',
                 'Cost is analytical dense global-decoder forward MAC estimate, not measured GPU time or peak memory.'])
    (args.output/'manifest.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    cost_table().to_csv(args.output/'compute_cost.csv',index=False)
    for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
        os.environ[key]='1'
    jobs=[(int(row.ecg_id),np.asarray(signals[int(row.artifact_index)])) for row in selection.itertuples()]
    if not all(np.isfinite(x).all() for _,x in jobs):
        raise ValueError('Nonfinite selected signals')
    start=time.perf_counter()
    summaries,statuses,target_frames=[],[],[]
    pool=ProcessPoolExecutor(max_workers=args.workers) if args.workers>1 and not args.delineation_cache else None
    try:
        results=iter(cached_results) if args.delineation_cache else (
            pool.map(delineate_record,jobs) if pool else map(delineate_record,jobs))
        for i,(job,result) in enumerate(zip(jobs,results)):
            (args.output/'delineations'/f"{result['ecg_id']}.json").write_text(
                json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
            grids,targets,summary=analyze_record(result)
            for name,frame in [('patch_overlap',grids),('target_candidates',targets)]:
                if not frame.empty:
                    path=args.output/f'{name}.csv'
                    frame.to_csv(path,mode='a',header=not path.exists(),index=False)
            summaries.append(summary)
            if not targets.empty:
                target_frames.append(targets)
            statuses.extend(result['status'])
            if i<args.examples:
                plot_example(job[1],result,args.output)
            print(f"[{i+1}/{len(jobs)}] ECG {result['ecg_id']}; {time.perf_counter()-start:.1f}s",flush=True)
    finally:
        if pool:
            pool.shutdown()
    summary=pd.concat(summaries,ignore_index=True)
    summary.to_csv(args.output/'lead_phase_summary.csv',index=False)
    pd.DataFrame(statuses).to_csv(args.output/'detection_status.csv',index=False)
    intervals=[r for result_path in (args.output/'delineations').glob('*.json')
               for r in json.loads(result_path.read_text(encoding='utf-8'))['intervals']]
    if intervals:
        pd.DataFrame(intervals).to_csv(args.output/'wave_intervals.csv',index=False)
    all_targets=pd.concat(target_frames,ignore_index=True) if target_frames else pd.DataFrame()
    overview=write_overview(summary,all_targets,args.output)
    costs=cost_table()
    report=['# Whole-wave group masking: 100 ms vs 40 ms','',
        f'{len(jobs)} real PTB-XL train ECGs / unique patients. No model training. No validation/test opened.',
        '', '## Interpretation', '', *['- '+x for x in provenance['caveats']], '',
        'Each accepted P/QRS/T interval is one target. Cover [onset, offset) by all intersecting fixed patches. No purity threshold.',
        'Extra time = rounded cover duration minus detected waveform duration. Left and right margins are measured separately.',
        'Other valid waves in the margins and rejected intervals in the margins are counted separately.',
        'B contains whole corresponding waves in other beats, rather than one representative patch. Conflicting whole groups are excluded and counted.',
        'For Q patches per lead, target size m and B union b: W=Q-m-b; C=11m; U=11(Q-m).',
        'Zero-target ECGs and failed leads remain in denominators. B=0 is measured separately from having no targets.',
        '', '## Compute estimate','',costs.to_csv(index=False),
        'Assumed encoder width/depth/heads=384/12/6; global decoder=256/4/4; FF expansion=4; batch=1; FP32.',
        'MAC excludes normalization, nonlinearities, softmax, backward and optimizer. Attention storage is a theoretical sum, NOT peak memory; fused kernels may avoid materialization.',
        '', '## Main comparison','',
        overview[['patch_ms','phase','targets','N_target_mean','extra_ms_mean','N_W_mean','N_B_mean','N_C_mean','no_B_fraction']].to_markdown(index=False)
            if __import__('importlib').util.find_spec('tabulate') else overview[['patch_ms','phase','targets','N_target_mean','extra_ms_mean','N_W_mean','N_B_mean','N_C_mean']].to_csv(index=False),
        '', 'Full distributions (median, p95, max) and coverage: comparison.csv.',
        'Patch size remains undecided. Inspect phase coverage, detection failures and example PNGs before choosing.']
    (args.output/'REPORT.md').write_text('\n'.join(report),encoding='utf-8')
    (args.output/'status.json').write_text(json.dumps(dict(status='completed',
        elapsed_seconds=time.perf_counter()-start,training_executed=False)),encoding='utf-8')
    print(overview.to_string(index=False),flush=True)


if __name__=='__main__':
    main()
