"""Verify the selected main model and redraw six frozen-comparator results."""
from pathlib import Path
import json,shutil,platform
import run_search as r
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from summarize import bootstrap_gap

ORDER=['TI_VMD_PE_TCN','TI_LSTM','TI_TCN','Raw_VMD_PE_TCN','Raw_TCN','Raw_LSTM']
LABELS=['温电校正\nVMD-PE-TCN','温电校正\nLSTM','温电校正\nTCN','未温电校正\nVMD-PE-TCN','未温电校正\nTCN','未温电校正\nLSTM']

def main():
 protocol=json.loads((r.OUT/'PROTOCOL.json').read_text(encoding='utf-8'))
 selection=json.loads((r.OUT/'SELECTION.json').read_text(encoding='utf-8'))
 assert r.sha(Path(r.__file__))==protocol['source_sha256']
 for name,digest in protocol['input_files'].items():assert r.sha(r.ROOT/name)==digest
 for name,digest in protocol['dependencies'].items():assert r.sha(r.ROOT/name)==digest
 old_metrics=pd.read_csv(r.FIXED/'selected_vehicle/six_model_metrics.csv').set_index('method')
 frame=r.load_frame(r.VIN);reference=float(frame.capacity_ah.iloc[:30].median())
 predictions=[];seed_rows=[];mask_count=0;weights_count=0
 for method in ORDER:
  for seed in r.SEEDS:
   folder=r.OUT/'evaluation'/f'seed{seed}' if method=='TI_VMD_PE_TCN' else r.FIXED/r.VIN/'evaluation'/method/f'seed{seed}'
   receipt=json.loads((folder/'COMPLETE.json').read_text(encoding='utf-8'))
   for name,digest in receipt['files'].items():assert r.sha(folder/name)==digest
   if method=='TI_VMD_PE_TCN':
    fitted=json.loads((folder/'state.json').read_text(encoding='utf-8'))
    assert fitted['spec']==r.CANDIDATES[selection['candidate']] and fitted['selected_epoch']==selection['epoch']
    assert receipt['last_training_label_index']<receipt['fit_end']==protocol['test_start']
    assert receipt['replay_difference_ah']<1e-7
    assert all(c['future_difference_ah']==0 for c in receipt['future_checks'])
    mask_count+=len(receipt['future_checks']);weights_count+=1
   pred=pd.read_csv(folder/'predictions.csv',float_precision='round_trip')
   expected=np.array([frame.reference_capacity_ah.iloc[o:o+31].mean() for o in pred.origin_index])
   np.testing.assert_allclose(pred.truth_ah,expected,rtol=0,atol=1e-10)
   seed_rows.append({'method':method,'seed':seed,**r.old.scores(pred.truth_ah,pred.prediction_ah,reference)})
   pred['method']=method;pred['seed']=seed;predictions.append(pred)
 all_predictions=pd.concat(predictions,ignore_index=True)
 ensemble=all_predictions.groupby(['method','origin_index'],as_index=False).agg(truth_ah=('truth_ah','first'),prediction_ah=('prediction_ah','mean'))
 results=[];truth_table=None
 for method in ORDER:
  one=ensemble.loc[ensemble.method.eq(method)].sort_values('origin_index')
  if truth_table is None:truth_table=one[['origin_index','truth_ah']].to_numpy()
  else:np.testing.assert_array_equal(truth_table,one[['origin_index','truth_ah']].to_numpy())
  score=r.old.scores(one.truth_ah,one.prediction_ah,reference)
  if method!='TI_VMD_PE_TCN':
   for key in ['nrmse_initial_percent','rmse_ah','mae_ah','r2']:
    assert abs(score[key]-old_metrics.loc[method,key])<1e-10
  results.append({'method':method,**score})
 metrics=pd.DataFrame(results).set_index('method').loc[ORDER]
 new=float(metrics.loc['TI_VMD_PE_TCN','nrmse_initial_percent'])
 tcn=float(metrics.loc['TI_TCN','nrmse_initial_percent']);lstm=float(metrics.loc['TI_LSTM','nrmse_initial_percent'])
 previous=float(old_metrics.loc['TI_VMD_PE_TCN','nrmse_initial_percent'])
 pipeline_rows=[]
 for architecture in ['VMD_PE_TCN','TCN','LSTM']:
  ti=float(metrics.loc['TI_'+architecture,'nrmse_initial_percent']);raw=float(metrics.loc['Raw_'+architecture,'nrmse_initial_percent'])
  pipeline_rows.append({'architecture':architecture,'corrected_pipeline_nrmse_percent':ti,'raw_pipeline_nrmse_percent':raw,'difference_pp':raw-ti})
 mean_pipeline_gap=float(np.mean([row['difference_pp'] for row in pipeline_rows]))
 pd.DataFrame(pipeline_rows).to_csv(r.OUT/'pipeline_comparison_gains.csv',index=False,encoding='utf-8-sig')
 order_ok=bool(new<tcn<lstm)
 spec=r.CANDIDATES[selection['candidate']]
 main_pred=ensemble.loc[ensemble.method.eq('TI_VMD_PE_TCN')].sort_values('origin_index')
 tcn_pred=ensemble.loc[ensemble.method.eq('TI_TCN')].sort_values('origin_index')
 low,high=bootstrap_gap(main_pred.truth_ah.to_numpy(),main_pred.prediction_ah.to_numpy(),tcn_pred.prediction_ah.to_numpy(),reference)
 seed_table=pd.DataFrame(seed_rows)
 seed_order=[]
 for seed in r.SEEDS:
  subset=seed_table.loc[seed_table.seed.eq(seed)].set_index('method').nrmse_initial_percent
  seed_order.append({'seed':seed,'main_nrmse_percent':float(subset['TI_VMD_PE_TCN']),
                     'tcn_nrmse_percent':float(subset['TI_TCN']),'lstm_nrmse_percent':float(subset['TI_LSTM']),
                     'requested_order_observed':bool(subset['TI_VMD_PE_TCN']<subset['TI_TCN']<subset['TI_LSTM'])})
 summary={'vehicle':r.VIN[:17],'selection':selection,'spec':spec,'requested_order_observed':order_ok,
          'new_main_nrmse_percent':new,'old_main_nrmse_percent':previous,'tcn_nrmse_percent':tcn,'lstm_nrmse_percent':lstm,
          'main_improvement_over_old_pp':previous-new,'main_advantage_over_tcn_pp':tcn-new,
          'main_vs_tcn_block_bootstrap95_pp':[low,high],
          'main_r2':float(metrics.loc['TI_VMD_PE_TCN','r2']),'mean_pipeline_difference_pp':mean_pipeline_gap,
          'individual_seed_orders':seed_order,'five_comparators_numerically_unchanged':True,
          'scope':'previously selected0991 retrospective case; no test-based candidate or epoch selection'}
 r.dump(r.OUT/'RESULT.json',summary)
 metrics.to_csv(r.OUT/'six_model_metrics.csv',encoding='utf-8-sig')
 ensemble.to_csv(r.OUT/'six_model_ensemble_predictions.csv',index=False,encoding='utf-8-sig',float_format='%.17g')
 seed_table.to_csv(r.OUT/'individual_seed_metrics.csv',index=False,encoding='utf-8-sig')
 pd.DataFrame(seed_order).to_csv(r.OUT/'individual_seed_order.csv',index=False,encoding='utf-8-sig')
 plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':12,'axes.unicode_minus':False,
                      'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
 fig,ax=plt.subplots(figsize=(14.5,5.5));values=metrics.nrmse_initial_percent.to_numpy();x=np.arange(6)
 ax.bar(x,values,width=.65,color=['#17826e','#376d91','#376d91','#b97332','#b97332','#b97332'])
 for xx,yy in zip(x,values):ax.annotate(f'{yy:.4f}%',(xx,yy),xytext=(0,6),textcoords='offset points',ha='center')
 ax.set_xticks(x,LABELS);ax.set_ylabel('NRMSE / %');ax.set_ylim(0,values.max()*1.2)
 ax.set_title(f'{r.VIN[:17]}｜论文参考主模型调参后对比',pad=15)
 fig.text(.5,.012,'其余五模型冻结；主模型仅按验证集选参；94个共同起点；5种子平均预测',ha='center',fontsize=9,color='#555555')
 fig.tight_layout(rect=[0,.045,1,1])
 for ext in ['png','svg']:fig.savefig(r.OUT/f'six_model_nrmse.{ext}',dpi=240,facecolor='white')
 plt.close(fig)
 fig,ax=plt.subplots(figsize=(11,4.8),constrained_layout=True)
 ax.plot(main_pred.origin_index,main_pred.truth_ah,color='#8796a3',lw=1.6,label='共同参考容量标签')
 for method,color,label in [('TI_VMD_PE_TCN','#17826e','VMD-PE-TCN（调参后）'),('TI_TCN','#376d91','TCN（冻结）'),('TI_LSTM','#b97332','LSTM（冻结）')]:
  one=ensemble.loc[ensemble.method.eq(method)].sort_values('origin_index')
  ax.plot(one.origin_index,one.prediction_ah,color=color,lw=1.4,label=label)
 ax.set(xlabel='保留片段中的预测起点',ylabel='容量 / Ah',title=f'{r.VIN[:17]}｜校正后三模型预测');ax.legend(fontsize=9);ax.grid(alpha=.12)
 for ext in ['png','svg']:fig.savefig(r.OUT/f'prediction_comparison.{ext}',dpi=240,facecolor='white')
 plt.close(fig)
 means=pd.read_csv(r.OUT/'validation_scores.csv')
 top=means.sort_values('validation_rmse_ah').groupby('candidate',sort=False).first().reset_index().sort_values('validation_rmse_ah')
 top.to_csv(r.OUT/'validation_candidate_ranking.csv',index=False,encoding='utf-8-sig')
 text=f'''0991论文参考VMD-PE-TCN主模型调整结果

主模型NRMSE：{new:.6f}%；原主模型：{previous:.6f}%。
冻结TCN：{tcn:.6f}%；冻结LSTM：{lstm:.6f}%。
性能顺序VMD-PE-TCN > TCN > LSTM是否观察到：{order_ok}。
对应NRMSE顺序应为VMD-PE-TCN < TCN < LSTM。
相对冻结TCN降低：{tcn-new:+.6f}个百分点。
主模型R²：{metrics.loc['TI_VMD_PE_TCN','r2']:.6f}。
校正组与未校正组三种模型的NRMSE平均差值：{mean_pipeline_gap:.6f}个百分点。
平均差值不表示每个对应架构均降低同一数值，主模型还包含额外选参及特征变化。

验证选择：{selection['candidate']}，共同训练轮数{selection['epoch']}。
三个开发种子平均验证RMSE：{selection['mean_validation_rmse_ah']:.6f} Ah。
最终五个种子共同评价，未挑选最好种子，也没有按照测试排名选择候选或轮数。
选定配置：K={spec['vmd_k']}，PE={spec['groups']}组，历史{spec['history']}，通道{spec['hidden']}，
时间卷积核宽{spec['kernel']}，膨胀层{spec['levels']}，Dropout={spec['dropout']}，学习率{spec['learning_rate']}，
优化器{spec['optimizer']}，分量尺度{spec['scale_kind']}，特征版本{spec['feature_set']}。

论文依据：Fu等，Journal of Energy Storage87(2024)111392，DOI10.1016/j.est.2024.111392。
第5-9页：VMD-PE重组、分量TCN独立预测、分量MSE寻参、分量预测叠加；
第7、9页：B5示例K7重构4组；第8页：Adam、Dropout0.2/0.5、膨胀率1到64。
PE分组在0991训练前缀重新估计，没有直接套用B5的分组编号。
本文只借鉴这些结构与选参思路，使用明确记录的14组候选验证搜索，没有完整复现IDBO，
因此结果不命名为IDBO-TCN。论文表4中卷积核数/尺寸的接口含义与本项目不完全一致，未静默颠倒。

与原任务保持一致：0991、403个严格筛选片段、训练/验证/测试边界、温电标定参数、
未来31段参考容量均值标签、评价起点、NRMSE分母和另外五模型全部保持原样。
历史工况版本使用过去的充电时长、SOC跨度、温度电流、里程等可得特征；
没有虚构论文中的单体等压升时间和放电等压降时间。
若选中health版本，收益包含历史工况输入及选参的综合变化，不能单独归因于VMD或温电校正。
本轮只给主模型增加搜索预算；排名描述冻结基线下的结果，不代表各架构在相同搜索预算下的最优比较。

时间边界：输入VMD每次只使用预测起点之前的观测；PE分组及缺失值填充值只在训练前缀拟合。
训练标签在训练截止前成熟，验证标签在测试开始前成熟。五个最终模型均经过权重回放和未来遮罩检查。
0991是之前从四车筛选出的回顾性案例，测试后段此前已经查看，不是全新外部独立测试。
参考容量仍为BMS数据形成的标准工况容量代理量，不是独立RPT容量真值。
若R²为负，表示尚不能据此宣称已取得高精度趋势预测；微小排名差异不等同于统计显著优势。

复现命令（从原项目根目录运行，沿用原科学计算环境）：
python -m unittest discover -s paper_guided_vmd_tuning_20261116 -p test_invariants.py -v
python paper_guided_vmd_tuning_20261116/run_search.py prepare
python paper_guided_vmd_tuning_20261116/run_search.py search
python paper_guided_vmd_tuning_20261116/run_search.py final
python paper_guided_vmd_tuning_20261116/make_results.py
'''
 (r.OUT/'调参结果与论文参考说明.txt').write_text(text,encoding='utf-8-sig')
 checks={'development_models_verified':42,'final_models_verified':weights_count,'future_mask_checks_passed':mask_count,
         'scientific_invariant_tests_passed':5,'baseline_models_frozen':25,'baseline_predictions_numerically_unchanged':True,
         'requested_performance_order_observed':order_ok,'selection_uses_validation_only':True}
 assert weights_count==5 and mask_count==15
 r.dump(r.OUT/'FINAL_VERIFICATION.json',checks)
 r.dump(r.OUT/'ENVIRONMENT.json',{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'torch':r.torch.__version__,'matplotlib':matplotlib.__version__,'device':'cpu','threads':3})
 code=r.OUT/'reproduction_code';code.mkdir(exist_ok=True)
 for p in Path(__file__).parent.iterdir():
  if p.is_file() and p.suffix in ['.py','.txt']:shutil.copy2(p,code/p.name)
 r.dump(r.OUT/'FILE_SHA256.json',{p.relative_to(r.OUT).as_posix():r.sha(p) for p in sorted(r.OUT.rglob('*')) if p.is_file() and p.name!='FILE_SHA256.json'})
 print(metrics[['nrmse_initial_percent','rmse_ah','r2']].to_string());print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
