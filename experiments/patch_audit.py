"""Training-free fixed-grid waveform audit. No Torch, optimizer, labels, or test data."""
from __future__ import annotations

import json
import math
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

FS, LENGTH = 250, 2500
LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
PHASES = ['P', 'QRS', 'T']


def geometry(p):
    if p not in (10, 25):
        raise ValueError('This audit compares exactly P10 (40 ms) and P25 (100 ms)')
    return LENGTH // p


def cost_table(width=384, depth=12, heads=6, decoder_width=256, decoder_depth=4,
               decoder_heads=4, ff_ratio=4):
    """Analytical forward MACs; global decoder, batch 1, dense attention. Not timing."""
    rows = []
    for p in (25, 10):
        q = geometry(p)
        n, v = 12*q, 6*q
        block = lambda t, d: (4+2*ff_ratio)*t*d*d + 2*t*t*d
        macs = (n*p*width + depth*block(v, width) + v*width*decoder_width
                + decoder_depth*block(n, decoder_width) + n*decoder_width*p)
        attention_elements = depth*heads*v*v + decoder_depth*decoder_heads*n*n
        rows.append(dict(patch_samples=p, patch_ms=1000*p/FS, patches_per_lead=q,
            total_tokens=n, visible_tokens=v, hidden_samples=15000,
            forward_gmacs_estimate=macs/1e9,
            attention_elements_all_layers=attention_elements,
            fp32_attention_mib_if_all_materialized=attention_elements*4/2**20))
    frame = pd.DataFrame(rows)
    frame['forward_macs_ratio_to_100ms'] = frame.forward_gmacs_estimate / frame.forward_gmacs_estimate.iloc[0]
    frame['attention_ratio_to_100ms'] = frame.attention_elements_all_layers / frame.attention_elements_all_layers.iloc[0]
    return frame


def delineate_record(job):
    """Common R peaks across leads; lead-specific wave boundaries. Failures retained."""
    ecg_id, x = job
    import neurokit2 as nk
    intervals, status = [], []
    caught = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        rpeaks, reference = None, None
        reference_errors = []
        for lead in (1, 7, 0):
            try:
                clean = nk.ecg_clean(x[lead], sampling_rate=FS)
                _, info = nk.ecg_peaks(clean, sampling_rate=FS)
                peaks = np.asarray(info['ECG_R_Peaks'], dtype=int)
                if len(peaks) < 3:
                    raise ValueError('Fewer than 3 detected R peaks')
                rpeaks, reference = peaks, LEADS[lead]
                break
            except Exception as exc:
                reference_errors.append(f'{LEADS[lead]}: {type(exc).__name__}: {exc}')
        for lead, name in enumerate(LEADS):
            if rpeaks is None:
                status.append(dict(ecg_id=ecg_id, lead=name, reference_lead='', r_peaks=0,
                    status='r_detection_failed', error='; '.join(reference_errors)))
                continue
            try:
                clean = nk.ecg_clean(x[lead], sampling_rate=FS)
                _, waves = nk.ecg_delineate(clean, rpeaks=rpeaks, sampling_rate=FS,
                                            method='dwt', check=False)
                for beat, r in enumerate(rpeaks):
                    current = []
                    for phase, key in [('P','P'), ('QRS','R'), ('T','T')]:
                        def at(suffix):
                            values = waves.get(f'ECG_{key}_{suffix}', [])
                            return float(values[beat]) if beat < len(values) else math.nan
                        onset, offset = at('Onsets'), at('Offsets')
                        valid = (np.isfinite(onset) and np.isfinite(offset)
                                 and 0 <= onset < offset <= LENGTH)
                        if valid and phase == 'P':
                            valid = offset <= r
                        elif valid and phase == 'QRS':
                            valid = onset <= r < offset
                        elif valid and phase == 'T':
                            valid = onset >= r
                        current.append(dict(ecg_id=ecg_id, lead=name, beat=beat,
                            r_sample=int(r), phase=phase,
                            onset=int(onset) if np.isfinite(onset) else None,
                            offset=int(offset) if np.isfinite(offset) else None,
                            valid=bool(valid), reason='' if valid else 'missing_or_invalid_boundaries'))
                    # Contradictory phase or neighbouring-beat intervals are not silently trusted.
                    intervals.extend(current)
                status.append(dict(ecg_id=ecg_id, lead=name, reference_lead=reference,
                    r_peaks=len(rpeaks), status='delineated', error=''))
            except Exception as exc:
                status.append(dict(ecg_id=ecg_id, lead=name, reference_lead=reference,
                    r_peaks=len(rpeaks), status='delineation_failed',
                    error=f'{type(exc).__name__}: {exc}'))
    for lead in LEADS:
        rows = sorted([r for r in intervals if r['lead'] == lead and r['valid']], key=lambda r:r['onset'])
        for a, b in zip(rows, rows[1:]):
            if a['offset'] > b['onset']:
                for row in (a,b):
                    row.update(valid=False, reason='overlapping_detected_intervals')
    return dict(ecg_id=ecg_id, intervals=intervals, status=status,
                warnings=sorted(set(str(w.message) for w in caught)))


def overlap_grid(intervals, p):
    """Half-open sample intervals; unknown is distinct from annotated background."""
    q = geometry(p)
    occupancy = np.zeros((3, LENGTH), dtype=bool)
    invalid = np.zeros(LENGTH, dtype=bool)
    valid = [r for r in intervals if r['valid']]
    for r in valid:
        a, b = int(r['onset']), int(r['offset'])
        if not 0 <= a < b <= LENGTH:
            raise ValueError('Invalid interval passed to overlap_grid')
        occupancy[PHASES.index(r['phase']), a:b] = True
    for r in intervals:
        if not r['valid'] and r['onset'] is not None and r['offset'] is not None:
            a,b = max(0,r['onset']), min(LENGTH,r['offset'])
            if a < b:
                invalid[a:b] = True
    fractions = occupancy.reshape(3,q,p).mean(-1).T
    ambiguous = occupancy.sum(0).reshape(q,p).max(-1) > 1
    ambiguous |= invalid.reshape(q,p).any(-1)
    counts = (fractions > 0).sum(-1)
    return fractions, counts, ambiguous


def covering_patches(onset, offset, p):
    """Smallest fixed-grid cover of half-open [onset, offset)."""
    geometry(p)
    if not (0 <= onset < offset <= LENGTH) or int(onset) != onset or int(offset) != offset:
        raise ValueError('Expected integer sample boundaries inside the recording')
    return set(range(int(onset)//p, (int(offset)+p-1)//p))


def target_candidates(intervals, p):
    """One entire valid wave is one target group. No purity or center filter."""
    q = geometry(p)
    valid = [r for r in intervals if r['valid']]
    groups = [(r, covering_patches(r['onset'], r['offset'], p)) for r in valid]
    rows = []
    for source, target in groups:
        other_groups = [(r, ids) for r, ids in groups
                        if r['phase'] == source['phase'] and r['beat'] != source['beat']]
        conflicts = sum(bool(ids & target) for _, ids in other_groups)
        # Drop an entire conflicting B group; never silently expose part of that wave.
        eligible = [ids for _, ids in other_groups if not ids & target]
        b = set().union(*eligible) if eligible else set()
        start, end = min(target)*p, (max(target)+1)*p
        duration = source['offset']-source['onset']
        left, right = source['onset']-start, end-source['offset']
        extras = np.ones(end-start, dtype=bool)
        extras[source['onset']-start:source['offset']-start] = False
        other_wave = np.zeros(end-start, dtype=bool)
        rejected = np.zeros(end-start, dtype=bool)
        for r in intervals:
            if r is source or r['onset'] is None or r['offset'] is None:
                continue
            a,z = max(start,r['onset']), min(end,r['offset'])
            if a < z:
                (other_wave if r['valid'] else rejected)[a-start:z-start] = True
        m,nb = len(target),len(b)
        row = dict(phase=source['phase'],beat=source['beat'],onset=source['onset'],
            offset=source['offset'],target_patch_ids=';'.join(map(str,sorted(target))),
            N_target=m,wave_ms=duration*1000/FS,covered_ms=m*p*1000/FS,
            left_extra_ms=left*1000/FS,right_extra_ms=right*1000/FS,
            extra_ms=(left+right)*1000/FS,extra_to_wave_ratio=(left+right)/duration,
            extra_other_wave_ms=int((extras & other_wave).sum())*1000/FS,
            extra_rejected_interval_ms=int((extras & rejected).sum())*1000/FS,
            B_wave_groups=len(eligible),B_conflicting_groups=conflicts,
            B_unique_groups=len({tuple(sorted(ids)) for ids in eligible}),
            N_W=q-m-nb,N_B=nb,N_C=11*m,N_U=11*(q-m),no_B=not b)
        assert m+row['N_W']+nb+row['N_C']+row['N_U']==12*q
        rows.append(row)
    return rows


GROUP_METRICS = ['N_target','wave_ms','covered_ms','left_extra_ms','right_extra_ms',
    'extra_ms','extra_to_wave_ratio','extra_other_wave_ms','extra_rejected_interval_ms',
    'N_W','N_B','N_C','N_U','B_wave_groups','B_conflicting_groups']


def analyze_record(result, patches=(25,10)):
    grids, targets, summaries = [], [], []
    for lead in LEADS:
        intervals = [r for r in result['intervals'] if r['lead'] == lead]
        status = next(s for s in result['status'] if s['lead'] == lead)
        for p in patches:
            fractions, counts, ambiguous = overlap_grid(intervals,p)
            q = geometry(p)
            for k in range(q):
                grids.append(dict(ecg_id=result['ecg_id'],lead=lead,patch_samples=p,
                    time_patch=k,P=fractions[k,0],QRS=fractions[k,1],T=fractions[k,2],
                    mixed=bool(counts[k]>1),ambiguous=bool(ambiguous[k]),
                    unassigned=bool(counts[k]==0),delineation_status=status['status']))
            candidates=target_candidates(intervals,p)
            targets.extend(dict(ecg_id=result['ecg_id'],lead=lead,patch_samples=p,**r) for r in candidates)
            for phase in PHASES:
                selected=[r for r in candidates if r['phase']==phase]
                item=dict(ecg_id=result['ecg_id'],lead=lead,patch_samples=p,phase=phase,
                    targets=len(selected),total_patches=q,
                    detected_intervals=sum(r['phase']==phase for r in intervals),
                    rejected_intervals=sum(r['phase']==phase and not r['valid'] for r in intervals),
                    delineation_status=status['status'],
                    no_B_fraction=float(np.mean([r['no_B'] for r in selected])) if selected else None)
                for metric in GROUP_METRICS:
                    values=[r[metric] for r in selected]
                    item[metric+'_mean']=float(np.mean(values)) if values else None
                summaries.append(item)
    return pd.DataFrame(grids),pd.DataFrame(targets),pd.DataFrame(summaries)


def write_overview(summary, targets, output):
    """Target-level distributions and all-patient coverage, including failed detections."""
    rows=[]
    for (p,phase),group in summary.groupby(['patch_samples','phase'],sort=True):
        per_ecg=group.groupby('ecg_id').targets.sum()
        selected=targets[(targets.patch_samples==p)&(targets.phase==phase)] if not targets.empty else targets
        item=dict(patch_samples=p,patch_ms=p*1000/FS,phase=phase,ecgs=len(per_ecg),
            targets=int(group.targets.sum()),targets_per_ecg_median=float(per_ecg.median()),
            zero_target_ecg_fraction=float((per_ecg==0).mean()),
            zero_target_lead_fraction=float((group.targets==0).mean()),
            rejected_wave_fraction=float(group.rejected_intervals.sum()/group.detected_intervals.sum())
                if group.detected_intervals.sum() else None,
            no_B_fraction=float(selected.no_B.mean()) if len(selected) else None)
        for metric in GROUP_METRICS:
            values=selected[metric] if len(selected) else pd.Series(dtype=float)
            for stat,fn in [('mean',np.mean),('median',np.median),('p95',lambda x:np.percentile(x,95)),('max',np.max)]:
                item[metric+'_'+stat]=float(fn(values)) if len(values) else None
        rows.append(item)
    frame=pd.DataFrame(rows)
    frame.to_csv(output/'comparison.csv',index=False)
    return frame


def plot_example(x, result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(14,6),sharex=True)
    colors={'P':'#009E73','QRS':'#D55E00','T':'#0072B2'}
    for ax,p in zip(axes,(25,10)):
        ax.plot(np.arange(LENGTH)/FS,x[1],color='#333333',lw=.7)
        for r in result['intervals']:
            if r['lead']=='II' and r['valid']:
                ax.axvspan(r['onset']/FS,r['offset']/FS,color=colors[r['phase']],alpha=.25)
                ids=covering_patches(r['onset'],r['offset'],p)
                ax.axvspan(min(ids)*p/FS,r['onset']/FS,color='#E6AB02',alpha=.3)
                ax.axvspan(r['offset']/FS,(max(ids)+1)*p/FS,color='#E6AB02',alpha=.3)
            elif (r['lead']=='II' and r['onset'] is not None and r['offset'] is not None
                  and r['onset'] < r['offset']):
                ax.axvspan(r['onset']/FS,r['offset']/FS,facecolor='none',
                           edgecolor='gray',hatch='///',alpha=.3)
        for r in sorted(set(r['r_sample'] for r in result['intervals'] if r['lead']=='II')):
            ax.axvline(r/FS,color='purple',ls=':',lw=.7)
        for t in np.arange(0,2.001,p/FS):
            ax.axvline(t,color='gray',lw=.4,alpha=.6)
        ax.set_ylabel(f'{p/FS*1000:.0f} ms / z units')
    axes[0].set_title(f"ECG {result['ecg_id']}, lead II: P green / QRS orange / T blue; extra margins yellow; rejected hatched")
    axes[-1].set(xlim=(0,2),xlabel='Seconds')
    fig.tight_layout()
    fig.savefig(output/f"ecg_{result['ecg_id']}_lead_II.png",dpi=140)
    plt.close(fig)
