"""Rebuild a chronological cohort without future-fitted quality thresholds."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'three_vehicle_audit_20261116'))
import run_three as previous
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from diagnose_capacity_spikes import features, segment_points
from six_vehicle_temperature_pipeline import aggregate_current_only, hampel_flags

OUT = ROOT / 'output/四车严格时间边界复跑_20261116'
VEHICLES = ['LK6ADCE15MB210690', 'LK6ADAE28LB200991_1',
            'LK6CCAE31LE554709_1', 'LK6CCAE36LE550543_1']


def cache_path(vehicle: str) -> Path:
    name = ('vehicle_0690_capacity_20261111' if vehicle.endswith('0690')
            else 'six_vehicle_capacity_20261111_final')
    return ROOT / 'output' / name / vehicle / 'charging_points.npz'


def clean_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    """Thresholds are frozen after30 initial segments, before any origin64+."""
    frame = frame.copy().reset_index(drop=True)
    reference = float(frame.capacity_ah.iloc[:30].median())
    # These exact window maxima are recomputed from telemetry at this reference.
    inconsistent = frame.strict_max60_excess_pp.gt(2.).to_numpy()
    blocks = pd.to_datetime(frame.end_time).diff().dt.total_seconds().div(86400).fillna(0).gt(30).cumsum()
    spike = np.zeros(len(frame), dtype=bool)
    for _, part in frame.groupby(blocks, sort=False):
        ids = part.index.to_numpy()
        signal = ids[~inconsistent[ids]]
        spike[signal] = hampel_flags(frame.capacity_ah.to_numpy()[signal], window=21,
                                    sigma=3., scale_floor=.01*reference, minimum=10, causal=True)
    frame['strict_soc_inconsistent'] = inconsistent
    frame['strict_causal_spike'] = spike
    frame['strict_keep'] = ~inconsistent & ~spike
    return frame


def load_frame(vehicle: str) -> pd.DataFrame:
    frame = pd.read_csv(OUT / vehicle / 'strict_inputs.csv', float_precision='round_trip')
    frame['end_time'] = pd.to_datetime(frame.end_time)
    return frame


def prefix_sign_diagnostic(frame: pd.DataFrame) -> dict:
    """Unconstrained empirical association; never used to choose score labels."""
    x = np.c_[frame.elapsed_days/365.25, frame.avg_temp-30.,
              frame.avg_pack_current-frame.avg_pack_current.median()]
    mean, scale = x.mean(0), x.std(0)
    scale = np.where(scale > 1e-8, scale, 1.)
    model = HuberRegressor(alpha=.01, epsilon=1.35, max_iter=3000).fit((x-mean)/scale, frame.capacity_ah)
    coefficient = model.coef_/scale
    return {'age_ah_per_year': float(coefficient[0]), 'temperature_ah_per_c': float(coefficient[1]),
            'current_ah_per_a': float(coefficient[2]), 'calibration_samples': len(frame),
            'condition_number': float(np.linalg.cond(np.c_[np.ones(len(x)), (x-mean)/scale])),
            'role': 'diagnostic only; not a replacement calibration or independent physical truth'}


def prepare(vehicle: str) -> None:
    folder = OUT / vehicle
    folder.mkdir(parents=True, exist_ok=True)
    source, cache = previous.source_path(vehicle), cache_path(vehicle)
    original = pd.read_csv(source, float_precision='round_trip')
    reference = float(original.capacity_ah.iloc[:30].median())
    with np.load(cache) as bundle:
        raw = pd.DataFrame({key: bundle[key] for key in bundle.files})
    points, _ = aggregate_current_only(raw, 1.)
    del raw
    electrical = []
    for _, row in original.iterrows():
        values = features(segment_points(points, row), reference)
        electrical.append(values['max_60s_excess_vs_local_reference_pp'])
    del points
    original['strict_max60_excess_pp'] = electrical
    cleaned = clean_candidates(original)
    cleaned.to_csv(folder / 'candidate_decisions.csv', index=False, encoding='utf-8-sig', float_format='%.17g')
    frame = cleaned.loc[cleaned.strict_keep].copy().reset_index(drop=True)
    frame['end_time'] = pd.to_datetime(frame.end_time)
    frame['elapsed_days'] = (frame.end_time-frame.end_time.iloc[0]).dt.total_seconds()/86400
    frame['block'] = frame.end_time.diff().dt.total_seconds().div(86400).fillna(0).gt(30).cumsum().astype(int)
    frame['avg_pack_current'] = frame.mean_charge_current_a.abs()
    n = len(frame)
    test_start, train_end, cal_end = int(n*390/563), int(n*(390/563-.20)), int(n*240/563)
    assert cal_end < train_end < test_start
    calibration = previous.calibrate(frame.iloc[:cal_end])
    frame['reference_capacity_ah'] = frame.capacity_ah/previous.factor(frame, calibration)
    frame.to_csv(folder / 'strict_inputs.csv', index=False, encoding='utf-8-sig', float_format='%.17g')
    previous.dump(folder / 'calibration.json', calibration)
    audit = {'vehicle': vehicle, 'candidate_rows': len(original), 'kept_rows': n,
             'reference_quality_capacity_ah': reference, 'quality_reference_source_rows': [0, 29],
             'training_end_exclusive': train_end, 'validation_end_exclusive_test_start': test_start,
             'calibration_end_exclusive': cal_end, 'calibration_end_time': str(frame.end_time.iloc[cal_end-1]),
             'training_last_time': str(frame.end_time.iloc[train_end-1]),
             'test_first_time': str(frame.end_time.iloc[test_start]),
             'previous_quality_reference_last_time': str(original.loc[original.calibration_prefix, 'end_time'].max()),
             'quality_decisions_different_from_previous': int((cleaned.strict_keep != cleaned.use_for_causal_export).sum()),
             'centered_smoothing_in_model_inputs': False, 'synthetic_rows_in_paired_audit': 0,
             'source_sha256': previous.sha(source), 'telemetry_cache_sha256': previous.sha(cache),
             'empirical_ti_sign_diagnostic': prefix_sign_diagnostic(frame.iloc[:cal_end]),
             'prefix_decision_checks': []}
    for boundary in [int(len(original)*.5), int(len(original)*.7), int(len(original)*.85)]:
        altered = original.copy()
        for column in ['capacity_ah', 'avg_temp', 'mean_charge_current_a', 'strict_max60_excess_pp']:
            altered.loc[boundary:, column] = 1000 + altered.loc[boundary:, column]*3
        repeated = clean_candidates(altered)
        assert np.array_equal(cleaned.strict_keep.iloc[:boundary], repeated.strict_keep.iloc[:boundary])
        # A literal prefix yields the same decisions too (no hidden full-length dependency).
        truncated = clean_candidates(original.iloc[:boundary])
        assert np.array_equal(cleaned.strict_keep.iloc[:boundary], truncated.strict_keep)
        audit['prefix_decision_checks'].append({'first_hidden_candidate': boundary,
                                               'future_mutation_identical': True, 'truncation_identical': True})
    old_end = pd.Timestamp(audit['previous_quality_reference_last_time'])
    audit['old_quality_reference_after_new_development_training_end'] = bool(old_end > frame.end_time.iloc[train_end-1])
    audit['old_quality_reference_after_new_test_start'] = bool(old_end >= frame.end_time.iloc[test_start])
    previous.dump(folder / 'DATA_AUDIT.json', audit)
    print(f'PREPARED {vehicle[:17][-4:]}: {len(original)} -> {n}; train/val/test={train_end}/{test_start}/{n}', flush=True)


if __name__ == '__main__':
    for vehicle in VEHICLES:
        prepare(vehicle)
