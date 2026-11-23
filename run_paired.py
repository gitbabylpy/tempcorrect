"""Prespecified equal-budget correction pairs; validation chooses common epochs."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from data_audit import ROOT, OUT, VEHICLES, load_frame, previous as old
import numpy as np
import pandas as pd
import torch

ARCHITECTURES = ['VMD_PE_TCN', 'TCN', 'LSTM']
DEV_SEEDS = [42, 123, 2025]
SEEDS = [42, 123, 2025, 2026, 2027]
TASKS = {'future31_reference': 31, 'next1_reference': 1}
MAX_EPOCHS = 60


def protocol() -> dict:
    value = {'vehicles': VEHICLES, 'tasks': TASKS, 'seeds': SEEDS, 'development_seeds': DEV_SEEDS,
             'requested_gap_pp': [1., 2.], 'maximum_development_epochs': MAX_EPOCHS,
             'input_information': 'capacity, causal EMA, capacity increment and component history only',
             'pairing': 'same architecture, width, learning rate, seeds and selected epoch; only input correction differs',
             'calibration': 'existing constrained joint model; fit before development training end; no new target fitting by prediction score',
             'epoch_rule': 'minimum mean validation RMSE over both correction arms and3 development seeds; epochs1..60',
             'test_rule': 'one frozen suffix; no selection by test ranking or requested gap',
             'primary_task': 'future31_reference (same prior task; no centered smoothed inputs)',
             'alternative_task': 'next1_reference; prespecified unsmoothed single-point reference-capacity prediction',
             'future_label_policy': 'all training labels end before training cutoff; all validation labels end before test cutoff',
             'rolling_feedback': 'previous observed capacities become available; no online weight updates',
             'raw_data_policy': 'never add noise, downsample or tune a weaker baseline to attain requested gap',
             'cohort': 'strict first30-reference quality decisions; causal Hampel; gap-reset; observed covariates required',
             'previous_result_policy': 'preserve previous frozen six-model settings and artifacts; this is a separate matched diagnostic',
             'scoring': 'NRMSE=100*RMSE/median(first30 retained raw capacities), identical denominator for all6 methods',
             'retrospective_status': 'suffix has been inspected in previous work; an exploratory rerun, not a fresh external test',
             'source_files': {p.name: old.sha(p) for p in [HERE/'data_audit.py', Path(__file__)]},
             'input_files': {v: old.sha(OUT/v/'strict_inputs.csv') for v in VEHICLES}}
    old.freeze(OUT / 'PROTOCOL.json', value)
    return value


def spec_for(architecture: str, corrected: bool) -> dict:
    base = old.backend.profiles('TCN', 'pe_frozen')['w8_h32_lr0.003']
    spec = copy.deepcopy(base)
    spec.update(kind='LSTM' if architecture == 'LSTM' else 'TCN',
                decomposition='pe_frozen' if architecture == 'VMD_PE_TCN' else 'none',
                groups=3 if architecture == 'VMD_PE_TCN' else 1,
                hidden=8 if architecture == 'VMD_PE_TCN' else 16,
                component_weight=.1 if architecture == 'VMD_PE_TCN' else 0.,
                normalization='TI' if corrected else 'none', max_epochs=MAX_EPOCHS, patience=MAX_EPOCHS)
    return spec


def common_features(frame, q):
    return old.common_features(frame, q)[:, :3]


def assemble(frame, q, features, origin, state, cache):
    groups = old.groups_at(frame, q, origin, state, cache)
    anchor = groups[-31:].mean(axis=0)
    common = features[origin-32:origin].copy()
    common[:, :2] -= anchor.sum()
    component = groups[-32:] - anchor
    x = np.stack([np.column_stack([common, component[:, g]]).T for g in range(state['spec']['groups'])])
    return x, anchor


def build(frame, fit, calibration, spec, task, caches):
    state = old.state_for(frame, fit, calibration, spec)
    state['feature_count'] = 4
    state['feature_names'] = ['capacity', 'causal_ema', 'capacity_increment', 'component_history']
    raw = frame.capacity_ah.to_numpy()
    reference = raw / old.factor(frame, calibration)
    q = reference if spec['normalization'] == 'TI' else raw
    horizon = TASKS[task]
    ids = old.valid_origins(frame, horizon)
    features = common_features(frame, q)
    values = {'ids': ids, 'label_end': ids+horizon-1, 'x': [], 'anchors': [], 'component_targets': [], 'truth': []}
    cache = caches.setdefault((spec['normalization'], spec['decomposition']), {})
    for origin in ids:
        x, anchor = assemble(frame, q, features, int(origin), state, cache)
        truth = float(reference[origin:origin+horizon].mean())
        values['x'].append(x); values['anchors'].append(anchor); values['truth'].append(truth)
        if origin+horizon <= fit:
            key = (task, spec['normalization'], spec['decomposition'], int(origin+horizon))
            target_cache = caches.setdefault('targets', {})
            if key not in target_cache:
                end = int(origin+horizon)
                block = frame.block.iloc[origin]
                start = int(frame.index[frame.block.eq(block)][0])
                gg = old.decompose_history(reference[max(start, end-128):end], state)
                tg = gg[-horizon:].mean(axis=0)
                tg[-1] += truth-tg.sum()
                target_cache[key] = tg
            values['component_targets'].append(target_cache[key])
        else:
            values['component_targets'].append(np.full(spec['groups'], np.nan))
    data = {key: np.asarray(value) for key, value in values.items()}
    mask = data['label_end'] < fit
    assert mask.sum() >= 20
    np.testing.assert_allclose(data['component_targets'][mask].sum(1), data['truth'][mask], atol=1e-9)
    return state, data


def prefix_prediction(model, state, frame, origin):
    observed = frame.iloc[:origin].copy()
    raw = observed.capacity_ah.to_numpy()
    q = raw / old.factor(observed, state['calibration']) if state['spec']['normalization'] == 'TI' else raw
    x, anchor = assemble(observed, q, common_features(observed, q), origin, state, {})
    return float(old.backend.predict_array(model, state, {'x': x[None], 'anchors': anchor[None]}, torch.device('cpu'))[0])


def train_development(vehicle, task, frame, audit, calibration):
    folder = OUT/vehicle/task
    fit, stop = audit['training_end_exclusive'], audit['validation_end_exclusive_test_start']
    # Test rows are not even passed to development dataset creation.
    development = frame.iloc[:stop].copy()
    cache = {}
    for architecture in ARCHITECTURES:
        traces = []
        for corrected in [True, False]:
            name = ('TI_' if corrected else 'Raw_') + architecture
            state, data = build(development, fit, calibration, spec_for(architecture, corrected), task, cache)
            val = data['ids'] >= fit
            assert val.any() and data['label_end'][val].max() < stop
            for seed in DEV_SEEDS:
                run = folder/'development'/name/f'seed{seed}'
                if not (run/'state.json').exists():
                    old.backend.train(copy.deepcopy(state), data, seed, run, epochs=None)
                trace = pd.read_csv(run/'training_log.csv')
                assert trace.epoch.max() == MAX_EPOCHS
                trace['method'] = name; trace['seed'] = seed
                traces.append(trace)
            print(f'{vehicle[:17][-4:]} {task} DEV {name}:3x{MAX_EPOCHS} epochs', flush=True)
        combined = pd.concat(traces, ignore_index=True)
        means = combined.loc[combined.epoch.gt(0)].groupby('epoch').validation_rmse_ah.mean()
        epoch = int(means.idxmin())
        selection = {'architecture': architecture, 'selected_common_epoch': epoch,
                     'mean_validation_rmse_ah': float(means.loc[epoch]),
                     'arms_and_seeds_in_selection': 6, 'test_data_used': False}
        combined.to_csv(folder/f'{architecture}_development_traces.csv', index=False)
        old.freeze(folder/f'{architecture}_SELECTION.json', selection)
        print(f'{vehicle[:17][-4:]} {task} {architecture} FROZEN epoch={epoch}', flush=True)


def fit_and_evaluate(vehicle, task, frame, audit, calibration):
    folder = OUT/vehicle/task
    fit = audit['validation_end_exclusive_test_start']
    caches = {}
    for architecture in ARCHITECTURES:
        selection = json.loads((folder/f'{architecture}_SELECTION.json').read_text(encoding='utf-8'))
        epoch = selection['selected_common_epoch']
        arm_predictions = []
        for corrected in [True, False]:
            name = ('TI_' if corrected else 'Raw_')+architecture
            state, data = build(frame, fit, calibration, spec_for(architecture, corrected), task, caches)
            keep = data['ids'] >= fit
            test = {key: value[keep] for key, value in data.items()}
            for seed in SEEDS:
                run = folder/'evaluation'/name/f'seed{seed}'
                if not (run/'COMPLETE.json').exists():
                    model, fitted = old.backend.train(copy.deepcopy(state), data, seed, run, epochs=epoch)
                    prediction = old.backend.predict_array(model, fitted, test, torch.device('cpu'))
                    report = pd.DataFrame({'origin_index': test['ids'], 'label_last_index': test['label_end'],
                                           'truth_ah': test['truth'], 'prediction_ah': prediction})
                    report.to_csv(run/'predictions.csv', index=False, float_format='%.17g')
                    old.dump(run/'metrics.json', old.scores(test['truth'], prediction, fitted['reference_capacity_ah']))
                    loaded, saved = old.backend.load_model(run/'model.pt', torch.device('cpu'))
                    replay = old.backend.predict_array(loaded, saved, test, torch.device('cpu'))
                    replay_diff = float(np.max(np.abs(prediction-replay)))
                    assert replay_diff < 1e-7
                    checks = []
                    for ix in sorted(set([0, len(prediction)//2, len(prediction)-1])):
                        origin = int(test['ids'][ix])
                        a = prefix_prediction(loaded, saved, frame, origin)
                        changed = frame.copy()
                        for column in changed.select_dtypes(include='number').columns:
                            changed.loc[origin:, column] = np.nan
                        b = prefix_prediction(loaded, saved, changed, origin)
                        assert abs(a-prediction[ix]) < 1e-5 and a == b
                        checks.append({'origin': origin, 'future_mask_difference_ah': float(abs(a-b))})
                    old.dump(run/'COMPLETE.json', {'replay_difference_ah': replay_diff, 'future_checks': checks,
                              'last_training_label_index': fitted['last_training_label_source_index'],
                              'fit_end_exclusive': fit, 'epochs': epoch,
                              'files': {p.name: old.sha(p) for p in run.iterdir() if p.is_file()}})
                else:
                    receipt = json.loads((run/'COMPLETE.json').read_text(encoding='utf-8'))
                    for name_file, digest in receipt['files'].items():
                        assert old.sha(run/name_file) == digest
                pred = pd.read_csv(run/'predictions.csv', float_precision='round_trip')
                pred['method'] = name; pred['seed'] = seed
                arm_predictions.append(pred)
            print(f'{vehicle[:17][-4:]} {task} EVAL {name}:5 seeds epoch={epoch}', flush=True)
        pd.concat(arm_predictions, ignore_index=True).to_csv(folder/f'{architecture}_predictions.csv', index=False, float_format='%.17g')


def run_vehicle(vehicle, task):
    frame = load_frame(vehicle)
    audit = json.loads((OUT/vehicle/'DATA_AUDIT.json').read_text(encoding='utf-8'))
    calibration = json.loads((OUT/vehicle/'calibration.json').read_text(encoding='utf-8'))
    assert pd.Timestamp(calibration['calibration_end_time']) <= frame.end_time.iloc[audit['training_end_exclusive']-1]
    train_development(vehicle, task, frame, audit, calibration)
    fit_and_evaluate(vehicle, task, frame, audit, calibration)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=list(TASKS), default='future31_reference')
    parser.add_argument('--vehicle', nargs='*', choices=VEHICLES, default=VEHICLES)
    args = parser.parse_args()
    torch.set_num_threads(3)
    protocol()
    for vehicle in args.vehicle:
        run_vehicle(vehicle, args.task)
