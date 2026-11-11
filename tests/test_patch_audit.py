import numpy as np
import pandas as pd
import pytest

from experiments.patch_audit import (LEADS, analyze_record, cost_table, covering_patches,
    overlap_grid, target_candidates, write_overview)


def wave(phase, beat, onset, offset, valid=True):
    return dict(phase=phase,beat=beat,onset=onset,offset=offset,valid=valid,lead='I')


def test_half_open_cover_and_record_end():
    assert covering_patches(100,200,25)=={4,5,6,7}
    assert covering_patches(2490,2500,10)=={249}
    assert covering_patches(0,1,25)=={0}
    with pytest.raises(ValueError):
        covering_patches(2500,2501,10)
    with pytest.raises(ValueError):
        covering_patches(3,3,10)


def test_grouping_does_not_require_purity_and_measures_other_wave_margin():
    intervals=[wave('P',0,0,10),wave('QRS',0,20,30)]
    fractions,counts,_=overlap_grid(intervals,25)
    np.testing.assert_allclose(fractions[0],[.4,.2,0])
    assert counts[0]==2
    rows=target_candidates(intervals,25)
    p=next(r for r in rows if r['phase']=='P')
    assert p['N_target']==1 and p['extra_ms']==60
    assert p['extra_other_wave_ms']==20
    assert len(rows)==2  # no center or purity selection


@pytest.mark.parametrize('p,expected',[(10,10),(25,4)])
def test_whole_corresponding_wave_and_disjoint_partition(p,expected):
    intervals=[wave('QRS',0,100,200),wave('QRS',1,400,500),wave('T',1,550,650)]
    rows=target_candidates(intervals,p)
    source=next(r for r in rows if r['phase']=='QRS' and r['beat']==0)
    assert source['N_B']==expected
    assert source['N_target']==expected and source['N_C']==11*expected
    assert source['extra_ms']==0
    for r in rows:
        assert r['N_W']+r['N_B']+r['N_target']==2500//p
        assert r['N_target']+r['N_W']+r['N_B']+r['N_C']+r['N_U']==12*2500//p
    assert all(r['no_B'] for r in rows if r['phase']=='T')


@pytest.mark.parametrize('p,n,extra',[(10,3,40),(25,2,120)])
def test_unaligned_wave_can_need_more_than_duration_over_patch(p,n,extra):
    row=target_candidates([wave('QRS',0,84,104)],p)[0]
    assert row['wave_ms']==80 and row['N_target']==n and row['extra_ms']==extra
    assert row['left_extra_ms']+row['right_extra_ms']==extra
    assert row['covered_ms']==row['wave_ms']+extra


def test_conflicting_B_group_is_excluded_whole():
    rows=target_candidates([wave('QRS',0,10,30),wave('QRS',1,40,60)],25)
    assert rows[0]['B_conflicting_groups']==1
    assert rows[0]['N_B']==0  # do not count just the non-overlapping tail


def test_B_union_deduplicates_shared_tokens():
    rows=target_candidates([wave('P',0,0,10),wave('P',1,100,120),wave('P',2,120,140)],25)
    assert rows[0]['N_B']==2 and rows[0]['B_wave_groups']==2


def test_invalid_intervals_are_not_targets_but_padding_contact_is_reported():
    rows=target_candidates([wave('QRS',0,84,104),wave('P',0,80,84,False)],10)
    assert len(rows)==1 and rows[0]['extra_rejected_interval_ms']==16


def test_failures_stay_in_denominators(tmp_path):
    result=dict(ecg_id=1,intervals=[],status=[dict(lead=l,status='r_detection_failed') for l in LEADS])
    _,targets,summary=analyze_record(result)
    assert targets.empty and summary.targets.sum()==0
    overview=write_overview(summary,targets,tmp_path)
    assert len(overview)==6 and overview.zero_target_ecg_fraction.eq(1).all()
    assert overview.zero_target_lead_fraction.eq(1).all()
    assert overview.no_B_fraction.isna().all()


def test_same_waves_are_paired_across_patch_sizes(tmp_path):
    result=dict(ecg_id=1,intervals=[wave('P',0,84,104),wave('P',1,334,354)],
        status=[dict(lead=l,status='delineated') for l in LEADS])
    _,targets,summary=analyze_record(result)
    assert targets.groupby('patch_samples').size().to_dict()=={10:2,25:2}
    overview=write_overview(summary,targets,tmp_path)
    assert overview[overview.phase.eq('P')].targets.eq(2).all()
    assert overview[overview.phase.eq('QRS')].targets.eq(0).all()


def test_cost_preserves_signal_budget_and_attention_scaling():
    costs=cost_table()
    assert costs.total_tokens.tolist()==[1200,3000]
    assert costs.visible_tokens.tolist()==[600,1500]
    assert costs.hidden_samples.eq(15000).all()
    assert costs.attention_ratio_to_100ms.tolist()==[1,6.25]
