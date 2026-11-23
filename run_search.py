"""Paper-guided VMD-PE-TCN search on frozen0991; no baseline retuning."""
from __future__ import annotations
import argparse, copy, hashlib, json, os, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['MPLCONFIGDIR']=str(ROOT/'tmp/paper_vmd_mpl')
os.environ.setdefault('OMP_NUM_THREADS','3')
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'strict_correction_rerun_20261116'))
import run_paired as prior
from data_audit import load_frame
old=prior.old
import numpy as np
import pandas as pd
import torch
from torch import nn
from vmdpy import VMD
VIN='LK6ADAE28LB200991_1'
SOURCE=ROOT/'output/四车严格时间边界复跑_20261116'
FIXED=ROOT/'output/四车验证选参与最佳案例_20261116'
OUT=ROOT/'output/0991论文参考VMD_PE_TCN调参_20261116'
DEV_SEEDS=[42,123,2025]
SEEDS=[42,123,2025,2026,2027]
HORIZON=31
BASE=prior.spec_for('VMD_PE_TCN',True)
BASE.update(max_epochs=80,patience=80,optimizer='AdamW',total_weight=1.,component_weight=.1,
            scale_kind='common',feature_set='capacity',vmd_k=7,vmd_alpha=2000.)
OVERRIDES=[
 ('legacy_g3',{}),
 ('pe4_base',dict(groups=4)),
 ('pe4_independent16',dict(groups=4,hidden=16,kernel=2,optimizer='Adam',total_weight=0.,component_weight=1.,scale_kind='component')),
 ('pe4_independent32',dict(groups=4,hidden=32,kernel=2,learning_rate=.001,optimizer='Adam',total_weight=0.,component_weight=1.,scale_kind='component')),
 ('pe4_mixed_h64',dict(groups=4,hidden=16,history=64,levels=5,total_weight=1.,component_weight=.5,scale_kind='component')),
 ('pe3_independent_h64',dict(hidden=16,kernel=2,history=64,levels=5,learning_rate=.001,optimizer='Adam',total_weight=0.,component_weight=1.,scale_kind='component')),
 ('pe4_deep_dropout',dict(groups=4,hidden=16,kernel=2,history=64,levels=7,dropout=.5,optimizer='Adam',total_weight=0.,component_weight=1.,scale_kind='component')),
 ('pe4_health16',dict(groups=4,hidden=16,kernel=2,feature_set='health',optimizer='Adam',total_weight=0.,component_weight=1.,scale_kind='component')),
 ('pe4_health32',dict(groups=4,hidden=32,kernel=2,learning_rate=.001,feature_set='health',optimizer='Adam',total_weight=0.,component_weight=1.,scale_kind='component')),
 ('pe4_health_h64',dict(groups=4,hidden=16,history=64,levels=5,feature_set='health',total_weight=1.,component_weight=.5,scale_kind='component')),
 ('pe3_health',dict(feature_set='health')),
 ('pe4_paper_dilations',dict(groups=4,hidden=16,kernel=2,history=64,levels=7,dropout=.5,learning_rate=.01,feature_set='health',optimizer='Adam',total_weight=0.,component_weight=1.,scale_kind='component')),
 ('pe4_k6_health',dict(groups=4,vmd_k=6,hidden=16,kernel=2,feature_set='health',optimizer='Adam',total_weight=0.,component_weight=1.,scale_kind='component')),
 ('pe4_k10_health',dict(groups=4,vmd_k=10,hidden=16,kernel=2,feature_set='health',optimizer='Adam',total_weight=0.,component_weight=1.,scale_kind='component'))]
CANDIDATES={name:{**BASE,**change} for name,change in OVERRIDES}


def dump(path,value):old.dump(Path(path),value)
def freeze(path,value):old.freeze(Path(path),value)
def sha(path):return old.sha(Path(path))

def setup():
 OUT.mkdir(parents=True,exist_ok=True)
 audit=json.loads((SOURCE/VIN/'DATA_AUDIT.json').read_text(encoding='utf-8'))
 prior_contract=json.loads((SOURCE/'PROTOCOL.json').read_text(encoding='utf-8'))
 assert sha(SOURCE/VIN/'strict_inputs.csv')==prior_contract['input_files'][VIN]
 files=[SOURCE/VIN/'strict_inputs.csv',SOURCE/VIN/'calibration.json',FIXED/'selected_vehicle/six_model_metrics.csv']
 for method in ['TI_TCN','TI_LSTM','Raw_VMD_PE_TCN','Raw_TCN','Raw_LSTM']:
  for seed in SEEDS:
   folder=FIXED/VIN/'evaluation'/method/f'seed{seed}'
   files.extend(folder/name for name in ['model.pt','state.json','predictions.csv','COMPLETE.json'])
 manifest=json.loads((FIXED/'FILE_SHA256.json').read_text(encoding='utf-8'))
 for file in files:
  if FIXED in file.parents:assert sha(file)==manifest[file.relative_to(FIXED).as_posix()]
 deps=json.loads((SOURCE/'DEPENDENCY_SHA256.json').read_text(encoding='utf-8'))
 for file,digest in deps.items():assert sha(ROOT/file)==digest
 value={'vehicle':VIN,'task':'same frozen next31 reference-capacity mean','candidates':CANDIDATES,
        'development_seeds':DEV_SEEDS,'evaluation_seeds':SEEDS,'fit_end':audit['training_end_exclusive'],
        'test_start':audit['validation_end_exclusive_test_start'],'epochs':80,
        'selection':'minimum mean validation RMSE across3 seeds and common positive epoch1..80',
        'model_tuned':'TI_VMD_PE_TCN only','five_comparators_frozen':True,
        'requested_performance_order':['TI_VMD_PE_TCN','TI_TCN','TI_LSTM'],
        'requested_nrmse_order':'strictly increasing, lower error is better',
        'ranking_not_used_to_select_candidate':True,'previously_selected_retrospective_case':True,
        'paper_doi':'10.1016/j.est.2024.111392','paper_sha256':'ef747c9fdc7b78abefb91b92bf3336720819725a178cabb16c15c2f9f4a03a23',
        'not_full_idbo_replication':True,'future_boundary':'input source indices < origin; fit labels end < fit_end',
        'input_files':{p.relative_to(ROOT).as_posix():sha(p) for p in files},'dependencies':deps,
        'source_sha256':sha(Path(__file__))}
 freeze(OUT/'PROTOCOL.json',value)
 return audit


def common_features(frame,q,state=None,fit=None,feature_set='capacity'):
 x=prior.common_features(frame,q)
 names=['capacity','causal_ema','capacity_increment']
 x=x[:,:3]
 if feature_set=='health':
  names+=['absolute_reference_capacity','mileage_10000km','age_years','observed_temperature','observed_current',
          'start_soc','end_soc','soc_span','charging_duration_hours','median_pack_voltage','current_std']
  x=np.column_stack([x,q,frame.mileage_km/10000.,frame.elapsed_days/365.25,frame.avg_temp,
                     frame.avg_pack_current,frame.start_soc/100.,frame.end_soc/100.,frame.delta_soc/100.,
                     frame.duration_hours,frame.median_voltage_v,frame.current_std_a])
 if state is None:
  med=np.nanmedian(np.where(np.isfinite(x[:fit]),x[:fit],np.nan),axis=0)
  if not np.isfinite(med).all():raise ValueError('Unidentifiable historical feature')
 else:med=np.asarray(state['feature_medians'])
 x=np.where(np.isfinite(x),x,med[None])
 return x,names,med


def make_state(frame,fit,calibration,spec):
 q=frame.capacity_ah.to_numpy()/old.factor(frame,calibration)
 longest=frame.iloc[:fit].groupby('block').size().idxmax()
 positions=frame.iloc[:fit].index[frame.iloc[:fit].block.eq(longest)].to_numpy()
 cfg=old.configuration();cfg['vmd'].update(k=spec['vmd_k'],alpha=spec['vmd_alpha'])
 cfg['pe'].update(groups=spec['groups'],order=3,delay=1)
 _,info,_=old.decompose_training_target(q[positions],cfg)
 mapping=np.asarray(info['mode_group_low_to_high_pe'])[np.argsort(info['center_frequencies'])].tolist()
 features,names,med=common_features(frame,q,fit=fit,feature_set=spec['feature_set'])
 state={'spec':copy.deepcopy(spec),'fit_rows':fit,'calibration':calibration,'pe_metadata':info,
        'pe_groups_by_frequency_rank':mapping,'feature_names':names+['component_history'],
        'feature_count':len(names)+1,'feature_medians':med.tolist(),
        'reference_capacity_ah':float(frame.capacity_ah.iloc[:30].median())}
 return state,q,features


def history_groups(q,begin,end,state,cache):
 spec=state['spec'];start=max(begin,end-spec['context']);values=q[start:end]
 key=(spec['vmd_k'],spec['vmd_alpha'],start,end)
 if key not in cache:
  padded=np.r_[values,values[-1]] if len(values)%2 else values
  modes,_,omega=VMD(padded,spec['vmd_alpha'],0.,spec['vmd_k'],0,1,1e-7)
  cache[key]=(modes[:,:len(values)],omega[-1])
 modes,frequencies=cache[key]
 labels=np.empty(spec['vmd_k'],int)
 labels[np.argsort(frequencies)]=np.asarray(state['pe_groups_by_frequency_rank'])
 groups=np.column_stack([modes[labels==g].sum(0) for g in range(spec['groups'])])
 groups[:,-1]+=values-groups.sum(1)
 np.testing.assert_allclose(groups.sum(1),values,rtol=0,atol=1e-9)
 return groups


def assemble(features,groups,state):
 spec=state['spec'];anchor=groups[-31:].mean(0)
 common=features[-spec['history']:].copy();common[:,:2]-=anchor.sum()
 component=groups[-spec['history']:]-anchor
 x=np.stack([np.column_stack([common,component[:,g]]).T for g in range(spec['groups'])])
 return x,anchor


def build(frame,fit,calibration,spec,cache):
 state,q,features=make_state(frame,fit,calibration,spec)
 ids=old.valid_origins(frame,HORIZON)
 values={'ids':ids,'label_end':ids+30,'x':[],'anchors':[],'truth':[],'component_targets':[]}
 begins={int(b):int(part.index[0]) for b,part in frame.groupby('block')}
 for origin in ids:
  begin=begins[int(frame.block.iloc[origin])]
  groups=history_groups(q,begin,int(origin),state,cache)
  x,anchor=assemble(features[:origin],groups,state)
  truth=float(q[origin:origin+31].mean())
  values['x'].append(x);values['anchors'].append(anchor);values['truth'].append(truth)
  if origin+31<=fit:
   tg=history_groups(q,begin,int(origin+31),state,cache)[-31:].mean(0)
   tg[-1]+=truth-tg.sum();values['component_targets'].append(tg)
  else:values['component_targets'].append(np.full(spec['groups'],np.nan))
 data={key:np.asarray(value) for key,value in values.items()}
 mask=data['label_end']<fit
 np.testing.assert_allclose(data['component_targets'][mask].sum(1),data['truth'][mask],atol=1e-9)
 return state,data


def prepare():
 audit=setup();frame=load_frame(VIN).iloc[:audit['validation_end_exclusive_test_start']].copy()
 cal=json.loads((SOURCE/VIN/'calibration.json').read_text(encoding='utf-8'))
 cache={};prepared=OUT/'prepared';prepared.mkdir(exist_ok=True)
 for name,spec in CANDIDATES.items():
  state,data=build(frame,audit['training_end_exclusive'],cal,spec,cache)
  np.savez_compressed(prepared/f'{name}.npz',**data);dump(prepared/f'{name}.json',state)
  print('PREPARED',name,data['x'].shape,flush=True)


def normalize(x,state):
 return ((x-np.asarray(state['x_mean'])[None,:,:,None])/np.asarray(state['x_scale'])[None,:,:,None]).astype(np.float32)

def predict(model,state,data):
 model.eval()
 with torch.no_grad():
  p=model(torch.as_tensor(normalize(data['x'],state))).numpy()
 return data['anchors'].sum(1)+(p*np.asarray(state['component_scales'])[None]).sum(1)


def train(state,data,seed,folder,epochs=None):
 folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
 old.backend.configure_numerics();old.backend.set_seed(seed)
 spec=state['spec'];mask=data['label_end']<state['fit_rows'];val=data['ids']>=state['fit_rows']
 mean=data['x'][mask].mean((0,3));scale=data['x'][mask].std((0,3));scale=np.where(scale>1e-8,scale,1.)
 residual=data['component_targets'][mask]-data['anchors'][mask]
 y=data['truth'][mask]-data['anchors'][mask].sum(1);total_scale=max(float(y.std()),.01)
 scales=np.maximum(residual.std(0),.01) if spec['scale_kind']=='component' else np.full(spec['groups'],total_scale)
 state={**copy.deepcopy(state),'seed':seed,'x_mean':mean.tolist(),'x_scale':scale.tolist(),
        'component_scales':scales.tolist(),'total_residual_scale':total_scale,
        'last_training_label_source_index':int(data['label_end'][mask].max()),'training_samples':int(mask.sum())}
 model=old.backend.ForecastModel(state['feature_count'],spec)
 optim=(torch.optim.Adam if spec['optimizer']=='Adam' else torch.optim.AdamW)
 optimizer=optim(model.parameters(),lr=spec['learning_rate'],weight_decay=0. if spec['optimizer']=='Adam' else spec['weight_decay'])
 x=torch.as_tensor(normalize(data['x'][mask],state));yc=torch.as_tensor((residual/scales).astype(np.float32))
 yt=torch.as_tensor(y.astype(np.float32));sc=torch.as_tensor(scales.astype(np.float32))
 val_data={key:value[val] for key,value in data.items()};logs=[];rng=np.random.default_rng(seed)
 best=float('inf');best_weights=None;best_epoch=0;started=time.perf_counter()
 for epoch in range(1,(epochs or spec['max_epochs'])+1):
  model.train();order=rng.permutation(len(x));loss_sum=0.
  for first in range(0,len(x),spec['batch_size']):
   ix=order[first:first+spec['batch_size']];optimizer.zero_grad(set_to_none=True);p=model(x[ix])
   loss=spec['total_weight']*(((p*sc).sum(1)-yt[ix])/total_scale).square().mean()+spec['component_weight']*(p-yc[ix]).square().mean()
   loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();loss_sum+=float(loss.detach())*len(ix)
  score=None
  if epochs is None:
   score=float(np.sqrt(np.mean((predict(model,state,val_data)-val_data['truth'])**2)))
   if score<best:
    best=score;best_epoch=epoch;best_weights={k:v.detach().clone() for k,v in model.state_dict().items()}
  logs.append({'epoch':epoch,'train_loss':loss_sum/len(x),'validation_rmse_ah':score})
 if best_weights is not None:model.load_state_dict(best_weights)
 state.update(selected_epoch=best_epoch if epochs is None else epochs,training_seconds=time.perf_counter()-started,
              parameters=sum(p.numel() for p in model.parameters()))
 pd.DataFrame(logs).to_csv(folder/'training_log.csv',index=False);dump(folder/'state.json',state)
 torch.save({'state':state,'weights':model.state_dict()},folder/'model.pt')
 return model,state


def search(names):
 setup()
 for name in names:
  state=json.loads((OUT/'prepared'/f'{name}.json').read_text(encoding='utf-8'))
  with np.load(OUT/'prepared'/f'{name}.npz') as a:data={key:a[key] for key in a.files}
  for seed in DEV_SEEDS:
   folder=OUT/'development'/name/f'seed{seed}'
   if not (folder/'COMPLETE.json').exists():
    train(state,data,seed,folder)
    dump(folder/'COMPLETE.json',{'files':{p.name:sha(p) for p in folder.iterdir() if p.is_file()}})
   print('TRAINED',name,seed,flush=True)


def select():
 traces=[]
 for name in CANDIDATES:
  for seed in DEV_SEEDS:
   folder=OUT/'development'/name/f'seed{seed}'
   receipt=json.loads((folder/'COMPLETE.json').read_text(encoding='utf-8'))
   for file,digest in receipt['files'].items():assert sha(folder/file)==digest
   trace=pd.read_csv(folder/'training_log.csv');assert trace.epoch.max()==80
   trace['candidate']=name;trace['seed']=seed;traces.append(trace)
 traces=pd.concat(traces,ignore_index=True)
 means=traces.groupby(['candidate','epoch'],as_index=False).validation_rmse_ah.mean()
 winner=means.sort_values(['validation_rmse_ah','candidate','epoch'],kind='stable').iloc[0]
 selection={'candidate':winner.candidate,'epoch':int(winner.epoch),'mean_validation_rmse_ah':float(winner.validation_rmse_ah),
            'selection_uses_test':False,'development_models':len(CANDIDATES)*len(DEV_SEEDS)}
 freeze(OUT/'SELECTION.json',selection)
 traces.to_csv(OUT/'development_traces.csv',index=False);means.to_csv(OUT/'validation_scores.csv',index=False)
 print('SELECTED',json.dumps(selection),flush=True)
 return selection


def load_model(folder):
 saved=torch.load(Path(folder)/'model.pt',map_location='cpu',weights_only=False);state=saved['state']
 model=old.backend.ForecastModel(state['feature_count'],state['spec']);model.load_state_dict(saved['weights'])
 return model,state


def predict_prefix(model,state,frame,origin):
 observed=frame.iloc[:origin].copy();q=observed.capacity_ah.to_numpy()/old.factor(observed,state['calibration'])
 features,_,_=common_features(observed,q,state=state,feature_set=state['spec']['feature_set'])
 begin=int(observed.index[observed.block.eq(observed.block.iloc[-1])][0])
 groups=history_groups(q,begin,origin,state,{})
 x,anchor=assemble(features,groups,state)
 return float(predict(model,state,{'x':x[None],'anchors':anchor[None]})[0])


def final():
 audit=setup();selection=select();frame=load_frame(VIN)
 cal=json.loads((SOURCE/VIN/'calibration.json').read_text(encoding='utf-8'))
 state,data=build(frame,audit['validation_end_exclusive_test_start'],cal,CANDIDATES[selection['candidate']],{})
 mask=data['ids']>=state['fit_rows'];test={key:value[mask] for key,value in data.items()}
 for seed in SEEDS:
  folder=OUT/'evaluation'/f'seed{seed}'
  if (folder/'COMPLETE.json').exists():
   receipt=json.loads((folder/'COMPLETE.json').read_text(encoding='utf-8'))
   for name,digest in receipt['files'].items():assert sha(folder/name)==digest
   print('FINAL_SEED_REUSED',seed,flush=True);continue
  model,fitted=train(state,data,seed,folder,selection['epoch']);p=predict(model,fitted,test)
  pd.DataFrame({'origin_index':test['ids'],'label_last_index':test['label_end'],'truth_ah':test['truth'],'prediction_ah':p}).to_csv(folder/'predictions.csv',index=False,float_format='%.17g')
  dump(folder/'metrics.json',old.scores(test['truth'],p,fitted['reference_capacity_ah']))
  loaded,saved=load_model(folder);replayed=predict(loaded,saved,test);diff=float(np.max(np.abs(p-replayed)));assert diff<1e-7
  checks=[]
  for ix in [0,len(p)//2,len(p)-1]:
   origin=int(test['ids'][ix]);a=predict_prefix(loaded,saved,frame,origin)
   hidden=frame.copy()
   for column in hidden.select_dtypes(include='number').columns:hidden.loc[origin:,column]=np.nan
   b=predict_prefix(loaded,saved,hidden,origin)
   assert abs(a-p[ix])<1e-5 and a==b
   checks.append({'origin':origin,'future_difference_ah':float(abs(a-b))})
  dump(folder/'COMPLETE.json',{'replay_difference_ah':diff,'future_checks':checks,
       'last_training_label_index':fitted['last_training_label_source_index'],'fit_end':state['fit_rows'],
       'files':{f.name:sha(f) for f in folder.iterdir() if f.is_file() and f.name!='COMPLETE.json'}})
  print('FINAL_SEED_COMPLETE',seed,flush=True)


if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','search','final'])
 parser.add_argument('--candidates',nargs='*',choices=list(CANDIDATES),default=list(CANDIDATES));args=parser.parse_args()
 torch.set_num_threads(3)
 if args.stage=='prepare':prepare()
 elif args.stage=='search':search(args.candidates)
 else:final()
