from argparse import Namespace
import json

import numpy as np
import pytest

from experiments.ncf.common import file_sha, read_json, read_plan
from experiments.ncf.data import prepare
from experiments.ncf.training import run
from experiments.run_ncf import create_plan
from scripts.diagnose_ncf_reconstruction import Moments, diagnose, run_times


def test_hidden_zero_baseline_and_constant_prediction_correlation():
    moments = Moments()
    moments.add([0, 0], [-1, 1])
    r = moments.result()
    assert r['model_mse'] == r['zero_mse'] == 1
    assert r['improvement_vs_zero'] == 0
    assert r['prediction_target_std_ratio'] == 0
    assert r['prediction_target_correlation'] is None


def test_pooled_moments_detect_correct_inverted_and_biased_waveforms():
    y = np.array([-2., -1., 1., 2.])
    for prediction, mse, correlation in ((y, 0, 1), (-y, 10, -1), (y + 2, 4, 1)):
        m = Moments()
        # Streaming aggregation must match the complete hidden waveform.
        m.add(prediction[:1], y[:1])
        m.add(prediction[1:], y[1:])
        result = m.result()
        assert result['model_mse'] == pytest.approx(mse)
        assert result['zero_mse'] == pytest.approx(np.mean(y ** 2))
        assert result['prediction_target_correlation'] == pytest.approx(correlation)
        assert result['prediction_target_std_ratio'] == pytest.approx(1)


def test_saved_checkpoint_audit_matches_original_metrics_and_preserves_files(tmp_path):
    study = tmp_path / 'study'
    create_plan(Namespace(study_dir=study, config=None, smoke=True, methods=['random', 'spsa'],
                          seeds=None, updates=2, artifact_dir=None))
    plan = read_plan(study)
    prepare(study, plan)
    for method in plan['config']['methods']:
        run(study, plan, method, 42)
    original = {str(p): file_sha(p) for p in study.rglob('*') if p.is_file()}
    output = study / 'diagnostics' / 'audit'
    diagnose(Namespace(study_dir=study, output_dir=output, seed=42, timing_only=False,
                       methods=None, examples=1, untrained_linear_probe=True))
    report = read_json(output / 'report.json')
    assert report['test_opened'] is False
    assert report['checkpoints_modified'] is False
    assert len(report['reconstruction']) == 6
    assert report['untrained_linear_probe']['macro_auroc'] is not None
    for row in report['reconstruction']:
        assert row['cases'] > 0
        assert np.isfinite(row['zero_mse'])
        if row['method'] != 'untrained':
            assert abs(row['difference_from_saved_mse']) < 2e-6
    assert len(list(output.glob('*.png'))) == 6
    assert all(file_sha(path) == value for path, value in original.items())
    read_plan(study)  # Diagnostic files must not invalidate frozen source hashes.
    for row in report['timings']:
        assert row['recorded_total_seconds'] == row['training_seconds'] + row['final_evaluation_seconds']
    completed = study / 'runs' / 'random' / 'seed42' / 'completed.json'
    altered = read_json(completed)
    altered['contract']['plan_id'] = 'foreign'
    completed.write_text(json.dumps(altered), encoding='utf-8')
    with pytest.raises(ValueError, match='Foreign'):
        run_times(study, plan, 42)
