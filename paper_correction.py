"""Energy2025 equations2-5/26 and literal equation4 PSO calibration."""
from __future__ import annotations

import numpy as np
import pandas as pd


def correction_factor(temperature, signed_current, parameters, reference_temperature, reference_current):
    t=np.asarray(temperature,dtype=float)
    current=np.asarray(signed_current,dtype=float)
    a,b,c,d,e=np.asarray(parameters,dtype=float)
    if b<=0 or d<=0 or reference_current>=0 or reference_temperature<=-273.15:
        raise ValueError('Invalid paper parameters/reference; charging current must be negative')
    if not np.isfinite(t).all() or not np.isfinite(current).all() or (current>0).any():
        raise ValueError('Use finite Celsius temperatures and nonpositive signed charging currents')
    with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
        return a*np.exp((t+273.15)/(b*(reference_temperature+273.15)))+c*np.exp(-current/(d*reference_current))+e


def literal_objective(errors:np.ndarray)->np.ndarray:
    errors=np.asarray(errors,float)
    return np.sum(errors,axis=-1)**2/errors.shape[-1]


def factors(frame:pd.DataFrame,state:dict)->np.ndarray:
    result=correction_factor(frame.avg_temp.to_numpy(float),-frame.avg_pack_current.to_numpy(float),
                             state['parameters'],state['reference_temperature_c'],state['reference_signed_current_a'])
    if not np.isfinite(result).all() or (result<=state['minimum_positive_factor']).any():
        raise ValueError('Paper correction factor is nonpositive/nonfinite; no targets may be dropped')
    return result


def published(study:dict)->dict:
    paper=study['paper']
    return {'mode':'published_equation26','parameters':paper['published_parameters'],
            'reference_temperature_c':paper['reference_temperature_c'],
            'reference_signed_current_a':paper['reference_signed_current_a'],
            'minimum_positive_factor':study['pso']['minimum_positive_factor'],
            'fitted_on_A':False,'source':'paper printed page7 equation26'}


def calibrate(calibration:pd.DataFrame,reference:pd.DataFrame,study:dict,seed:int)->tuple[dict,pd.DataFrame]:
    settings=study['pso']
    ref=study['a_reference']
    if len(calibration)<5 or len(reference)<1:
        raise ValueError('Insufficient calibration/reference observations')
    t=calibration.avg_temp.to_numpy(float)
    current=-calibration.avg_pack_current.to_numpy(float)
    capacity=calibration.capacity_ah.to_numpy(float)
    cbase=float(reference.capacity_ah.mean())
    lower,upper=np.array(settings['lower']),np.array(settings['upper'])
    if np.any(lower>upper):
        raise ValueError('Unordered PSO bounds; record the printed e reversal explicitly')
    rng=np.random.default_rng(seed)
    position=rng.uniform(lower,upper,size=(settings['particles'],5))
    velocity=np.zeros_like(position)

    def evaluate(particles):
        scores=[]
        mse=[]
        for parameter in particles:
            psi=correction_factor(t,current,parameter,ref['temperature_c'],ref['signed_current_a'])
            if not np.isfinite(psi).all() or np.any(psi<=settings['minimum_positive_factor']):
                scores.append(np.inf); mse.append(np.inf)
            else:
                error=capacity/psi-cbase
                scores.append(float(literal_objective(error)))
                mse.append(float(np.mean(error**2)))
        return np.asarray(scores),np.asarray(mse)

    personal=position.copy()
    personal_score,personal_mse=evaluate(position)
    index=int(np.argmin(personal_score))
    best,best_score,best_mse=personal[index].copy(),float(personal_score[index]),float(personal_mse[index])
    trace=[{'iteration':0,'best_literal_objective_ah2':best_score,'diagnostic_mse_ah2':best_mse}]
    for iteration in range(1,settings['iterations']+1):
        r1,r2=rng.random(position.shape),rng.random(position.shape)
        velocity=settings['inertia']*velocity+settings['cognitive']*r1*(personal-position)+settings['social']*r2*(best-position)
        position=np.clip(position+velocity,lower,upper)
        score,mse=evaluate(position)
        improved=score<personal_score
        personal[improved]=position[improved]
        personal_score[improved]=score[improved]
        personal_mse[improved]=mse[improved]
        index=int(np.argmin(personal_score))
        if personal_score[index]<best_score:
            best,best_score,best_mse=personal[index].copy(),float(personal_score[index]),float(personal_mse[index])
        trace.append({'iteration':iteration,'best_literal_objective_ah2':best_score,'diagnostic_mse_ah2':best_mse})
    if not np.isfinite(best_score):
        raise ValueError('PSO did not find a physically admissible correction')
    state={'mode':'same_paper_function_A_reference_PSO','seed':seed,'parameters':best.tolist(),
           'reference_temperature_c':ref['temperature_c'],'reference_signed_current_a':ref['signed_current_a'],
           'minimum_positive_factor':settings['minimum_positive_factor'],'fitted_on_A':True,
           'calibration_rows':len(calibration),'calibration_source_rows':calibration.source_row.astype(int).tolist(),
           'reference_rows':len(reference),'reference_source_rows':reference.source_row.astype(int).tolist(),
           'reference_mean_capacity_ah':cbase,'literal_objective_ah2':best_score,'diagnostic_mse_ah2':best_mse,
           'pso_settings':settings,'e_bounds_ordered_from_printed_reversed_endpoints':True}
    return state,pd.DataFrame(trace)
