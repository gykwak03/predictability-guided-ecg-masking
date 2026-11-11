from argparse import Namespace
import copy
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from experiments.ncf.common import (DEFAULTS, METHODS, TWO_BRANCH, RATE_ORDER, geometry,
                                    read_plan, read_json, validate_config, seed_for)
from experiments.ncf.masks import (context_sets, matched_masks, rates_from_logits,
                                   logits_from_rates, limited_update)
from experiments.ncf.data import prepare, Prepared
from experiments.ncf.evaluation import evaluate_bank
from experiments.ncf.training import run, policy_candidates, policy_gradient, reward_scale
from experiments.run_ncf import create_plan, analyze


def make_plan(root, methods=None, updates=2, overrides=None):
    config = None
    if overrides:
        config = root.parent / f'{root.name}_config.json'
        config.write_text(json.dumps(overrides), encoding='utf-8')
    create_plan(Namespace(study_dir=root, config=config, smoke=True, methods=methods,
                          seeds=None, updates=updates, artifact_dir=None))
    return read_plan(root)


@pytest.mark.parametrize('p,counts', [
    (10, dict(target=60, near=120, cross=60, far=1320, background=1440)),
    (25, dict(target=24, near=48, cross=24, far=528, background=576)),
])
def test_geometry_is_disjoint_exhaustive_and_all_rate_corners_feasible(p, counts):
    c = copy.deepcopy(DEFAULTS)
    c['patch_samples'] = p
    validate_config(c)
    for seed in range(4):
        groups, target = context_sets(c, seed)
        assert {k: len(v) for k, v in groups.items()} == counts
        np.testing.assert_array_equal(np.sort(np.concatenate(list(groups.values()))), np.arange(12 * 2500 // p))
        assert len(target['leads']) == len(set(target['leads'])) == 6
        for rates in product([c['rate_low'], c['rate_high']], repeat=3):
            mask, info = matched_masks(c, [rates], seed)[0]
            assert mask.sum() == len(mask) // 2
            assert mask[groups['target']].all()
            for key, rate in zip(RATE_ORDER, rates):
                assert abs(info['realized'][key] - rate) <= 1 / counts[key] + 1e-12


def test_changing_near_does_not_change_cross_or_far_selection():
    c = copy.deepcopy(DEFAULTS)
    seed = 88
    groups, _ = context_sets(c, seed)
    a, b = matched_masks(c, [[.2, .6, .4], [.8, .6, .4]], seed)
    for key in ('cross', 'far', 'target'):
        np.testing.assert_array_equal(a[0][groups[key]], b[0][groups[key]])
    assert set(groups['near'][~a[0][groups['near']]]) <= set(groups['near'][~b[0][groups['near']]])
    assert a[1]['target'] == b[1]['target']
    assert b[1]['visible']['near'] - a[1]['visible']['near'] == a[1]['visible']['background'] - b[1]['visible']['background']


def test_infeasible_cube_is_rejected_instead_of_projecting_rates():
    c = copy.deepcopy(DEFAULTS)
    c['target_leads'] = 10
    with pytest.raises(ValueError, match='infeasible'):
        validate_config(c)
    with pytest.raises(ValueError, match='bounds'):
        matched_masks(DEFAULTS, [[1., .5, .5]], 42)


def test_antithetic_conditional_best_two_uses_same_target_but_different_masks():
    a, b = matched_masks(DEFAULTS, [[.5] * 3] * 2, 99, kind='conditional_best2')
    assert a[1]['target'] == b[1]['target']
    assert not np.array_equal(a[0], b[0])
    assert a[1]['target_hidden'] == b[1]['target_hidden'] == 60


def test_logits_steps_and_antithetic_reinforce_score_gradient():
    c = DEFAULTS
    x = np.array([-3., 0., 2.])
    updated = limited_update(x, -x + 20, c)
    assert np.max(np.abs(rates_from_logits(updated, c) - rates_from_logits(x, c))) <= .05000001
    rates, direction, noise = policy_candidates('reinforce', x, c, 0, 42)
    np.testing.assert_allclose((logits_from_rates(rates[0], c) + logits_from_rates(rates[1], c)) / 2, x, atol=1e-12)
    # Analytic mean score estimator with the pair-mean reward baseline.
    reward = np.array([.03, -.02])
    baseline = reward.mean()
    expected = ((reward[0] - baseline) * noise / c['sigma'] +
                (reward[1] - baseline) * -noise / c['sigma']) / 2
    np.testing.assert_allclose(policy_gradient('reinforce', reward, direction, noise, c), expected)
    assert reward_scale([], 2., c) == (2., 'initial_probe_loss')
    history = list(range(10))
    scale, _ = reward_scale(history, 2., c)
    assert scale == pytest.approx(5.4) and history == list(range(10))


@pytest.fixture(scope='module')
def study(tmp_path_factory):
    root = tmp_path_factory.mktemp('ncf') / 'study'
    plan = make_plan(root)
    prepare(root, plan)
    return root, plan


def test_prepare_never_uses_annotations_and_keeps_control_validation_separate(study, monkeypatch):
    root, plan = study
    monkeypatch.setitem(sys.modules, 'neurokit2', None)
    import scipy.optimize
    monkeypatch.setattr(scipy.optimize, 'milp', lambda *a, **k: pytest.fail('MILP must not run'))
    prepare(root, plan)
    d = Prepared(root, plan)
    fit = set(d.frame.iloc[d.selection['fit']].patient_id)
    control = set(d.frame.iloc[d.selection['control']].patient_id)
    assert not fit & control and not control & set(d.val_frame.patient_id)
    assert not (root / 'prepared/annotations').exists()
    assert read_json(root / 'prepared/completed.json')['labels_opened'] is False
    for split in ('control', 'validation'):
        with np.load(root / f'prepared/random_{split}.npz') as bank:
            for i in range(0, len(bank['masks']), 2):
                np.testing.assert_array_equal(bank['masks'][i], ~bank['masks'][i + 1])
        with np.load(root / f'prepared/balanced_{split}.npz') as bank:
            assert len(np.unique(bank['strata'])) == 8
            assert bank['masks'][bank['anchors']].all()
            for start in range(0, len(bank['indices']), 8):
                assert (bank['anchors'][start:start + 8] == bank['anchors'][start]).all()


def test_balanced_reward_uses_all_hidden_not_just_targets(tmp_path):
    # Large errors occur outside the anchor. Reward must see those errors.
    class FakeModel:
        def eval(self):
            return self
        def __call__(self, x, mask):
            target = torch.zeros((len(x), 4, 1))
            prediction = torch.tensor([[[1.], [3.], [0.], [0.]]]).repeat(len(x), 1, 1)
            return prediction, target
    path = tmp_path / 'probe.npz'
    np.savez(path, indices=[0], masks=[[True, True, False, False]],
             anchors=[[True, False, False, False]], strata=['N0.2_C0.2_F0.2'])
    result = evaluate_bank(FakeModel(), np.zeros((1, 12, 2500), np.float32), path, torch.device('cpu'), 1)
    assert result['loss'] == 5. and result['target_loss'] == 1.


@pytest.mark.parametrize('method', METHODS)
def test_all_eight_methods_from_scratch_and_separate_validation(study, method):
    root, plan = study
    result = run(root, plan, method, 42)
    assert result['complete'] and result['step'] == 2
    assert result['attempted_updates'] == (4 if method in TWO_BRANCH else 2)
    assert result['annotations_used'] is False and result['test_opened'] is False
    for key in ('random_validation', 'context_balanced_validation'):
        assert np.isfinite(result[key]['loss'])
        assert result[key]['loss_scope'] == 'all_hidden_patches'
    assert 0 <= result['linear_probe']['macro_auroc'] <= 1
    record = read_json(root / f'runs/{method}/seed42/history.json')[0]
    assert record['reward_kind'] == 'context_balanced_all_hidden_lp'
    assert record['rate_order'] == ['near', 'cross', 'far']
    assert len(record['candidate_rates']) == (2 if method in TWO_BRANCH else 1)
    for r, outcome in zip(record['raw_rewards'], record['candidate_context_balanced']):
        assert r == pytest.approx(record['context_before_loss'] - outcome['loss'])
    assert record['reward_scale_kind'] == 'initial_probe_loss'
    if method != 'random':
        assert all(s['target_hidden_mean'] == 24 for s in record['masks'])


@pytest.mark.parametrize('method', ['spsa', 'reinforce', 'random_policy'])
def test_resume_preserves_model_optimizer_rng_and_policy(tmp_path, method):
    root = tmp_path / 'study'
    plan = make_plan(root, [method], updates=4,
                     overrides={'normalization_min_history': 1 if method == 'random_policy' else 2})
    prepare(root, plan)
    run(root, plan, method, 42)
    resumed = root / 'resumed'
    assert run(root, plan, method, 42, folder=resumed, stop_after=2)['complete'] is False
    run(root, plan, method, 42, folder=resumed)
    a = torch.load(root / f'runs/{method}/seed42/checkpoint.pt', map_location='cpu', weights_only=False)
    b = torch.load(resumed / 'checkpoint.pt', map_location='cpu', weights_only=False)
    for key in a['training']['model']:
        torch.testing.assert_close(a['training']['model'][key], b['training']['model'][key], rtol=0, atol=0)
    for key in a['training']['optimizer']['state']:
        for name, tensor in a['training']['optimizer']['state'][key].items():
            torch.testing.assert_close(tensor, b['training']['optimizer']['state'][key][name], rtol=0, atol=0)
    assert a['x'] == b['x'] and a['reward_history'] == b['reward_history']
    assert a['history'][1]['reward_scale_kind'] == 'historical_p80_minus_p20'
    assert [r['mask_hashes'] for r in a['history']] == [r['mask_hashes'] for r in b['history']]
    torch.testing.assert_close(a['training']['rng']['torch'], b['training']['rng']['torch'])


def test_sources_and_probe_tampering_rejected(study, monkeypatch, tmp_path):
    root, _ = study
    import experiments.ncf.common as common
    monkeypatch.setattr(common, 'sources', lambda: {'changed': 'file'})
    with pytest.raises(ValueError, match='Source changed'):
        read_plan(root)
    import shutil
    copied = tmp_path / 'copied'
    shutil.copytree(root / 'prepared', copied / 'prepared')
    path = copied / 'prepared/random_control.npz'
    path.write_bytes(path.read_bytes() + b'tampered')
    with pytest.raises(ValueError, match='Prepared probe data changed'):
        Prepared(copied, study[1])


@pytest.mark.parametrize('device,size', [('cpu', 'small'), ('cuda', 'smoke')])
def test_real_runs_require_full_model_and_cuda(device, size):
    c = copy.deepcopy(DEFAULTS)
    c.update(device=device, model_size=size)
    with pytest.raises(ValueError, match='real runs require CUDA'):
        validate_config(c)


def test_tiny_time_budget_still_saves_a_complete_block_checkpoint(tmp_path):
    root = tmp_path / 'budget'
    plan = make_plan(root, ['fixed'], updates=4, overrides={'budget_seconds': 1e-12})
    prepare(root, plan)
    result = run(root, plan, 'fixed', 42)
    assert result['complete'] and result['step'] == 2 and result['stop_reason'] == 'compute_budget'
    assert (root / 'runs/fixed/seed42/checkpoint.pt').is_file()
