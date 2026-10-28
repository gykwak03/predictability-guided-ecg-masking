"""Reuse the reviewed NRMSE and add physical-input-unit amplitude diagnostics."""
from __future__ import annotations

import numpy as np

from experiments.predictability_estimators import reconstruction_metrics, sample_hidden


def measure(prediction, target, mask, scale):
    frame, patches = reconstruction_metrics(prediction, target, mask, scale)
    hidden = sample_hidden(mask, target.shape)
    for b in range(len(target)):
        y, pred = target[b][hidden[b]].astype(float), prediction[b][hidden[b]].astype(float)
        rms_y, rms_pred = np.sqrt(np.mean(y*y)), np.sqrt(np.mean(pred*pred))
        frame.loc[b, 'hidden_mse'] = np.mean((y-pred)**2)
        frame.loc[b, 'zero_hidden_mse'] = np.mean(y*y)
        frame.loc[b, 'prediction_rms'] = rms_pred
        frame.loc[b, 'target_rms'] = rms_y
        frame.loc[b, 'rms_ratio'] = rms_pred/rms_y if rms_y > 1e-12 else np.nan
    return frame, patches


def aggregate(frame):
    if frame.empty or not np.isfinite(frame.nrmse).all() or (frame.masked_samples <= 0).any():
        raise ValueError('No finite evaluable masks; results cannot be marked completed')
    w = frame.masked_samples.to_numpy()
    mse = np.average(frame.nrmse**2, weights=w)
    zero = np.average(frame.zero_nrmse**2, weights=w)
    rms = np.sqrt(np.average(frame.prediction_rms**2, weights=w))
    target_rms = np.sqrt(np.average(frame.target_rms**2, weights=w))
    skill = 1-mse/zero if zero > 1e-12 else np.nan
    ratio = rms/target_rms if target_rms > 1e-12 else np.nan
    return dict(masks=len(frame), pooled_nrmse=float(np.sqrt(mse)),
                hidden_mse=float(np.average(frame.hidden_mse, weights=w)),
                zero_nrmse=float(np.sqrt(zero)), skill_vs_zero=float(skill),
                masked_pearson=float(frame.masked_pearson.mean()),
                prediction_rms=float(rms), target_rms=float(target_rms), rms_ratio=float(ratio),
                worse_than_zero=bool(skill < 0), amplitude_collapse_flag=bool(ratio < .1),
                amplitude_flag_threshold=.1, lower_nrmse_means='easier reconstruction',
                quality='undefined_zero_baseline' if not np.isfinite(skill) else
                        'worse_than_zero' if skill < 0 else 'review_amplitude' if ratio < .1 else 'review_learning_curve')
