"""Local retrospective cloud estimator; statistical prediction, not intervention.

Train once into a new directory; CloudEstimator then serves read-only inference.
No weather-token model files or training results are modified.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import copy
import csv
import hashlib
import json
import math
import platform
import sys
import time

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from revision.weather_tokens.data import load_raw, _indices

DEFAULT_ARTIFACT_DIR = ROOT.parent/'revision_outputs/cloud_simulator_v1'
MODES = ('wind_humidity','context')
HYPERPARAMETERS = dict(max_iter=120,max_leaf_nodes=15,l2_regularization=1.,
                       min_samples_leaf=100,learning_rate=.08,early_stopping=False,random_state=17)
THREADS = 8
ALPHA = .2
INPUT_NAMES = ('u_mps','v_mps','humidity_kgkg')
CONTEXT_NAMES = INPUT_NAMES+('region_id_category','month_sin','month_cos')


def _now(): return datetime.now(timezone.utc).isoformat()


def _sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):digest.update(block)
    return digest.hexdigest()


def _array_receipt(array):
    x=np.ascontiguousarray(array)
    return {'shape':list(x.shape),'dtype':str(x.dtype),'sha256':hashlib.sha256(x.tobytes()).hexdigest()}


def _save_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')


def feature_matrix(physical,region_ids,months,mode):
    """physical columns are u m/s, v m/s, q kg/kg; month is 1..12."""
    physical=np.asarray(physical,dtype=np.float64)
    if physical.ndim!=2 or physical.shape[1]!=3 or not np.isfinite(physical).all():
        raise ValueError('Expected finite N by 3 physical input')
    if mode=='wind_humidity':return physical.copy()
    if mode!='context':raise ValueError('Unknown model mode')
    region_ids=np.asarray(region_ids,dtype=np.int64);months=np.asarray(months,dtype=np.int64)
    if len(region_ids)!=len(physical) or len(months)!=len(physical) or ((months<1)|(months>12)).any():
        raise ValueError('Context lengths/months are invalid')
    angle=2*np.pi*(months-1)/12
    return np.column_stack((physical,region_ids,np.sin(angle),np.cos(angle)))


def conformal_expansion(y,lower,upper,alpha=ALPHA):
    """Nonnegative CQR expansion, exact finite-sample order statistic.

    This numerical calibration formula does not assert exchangeability of
    spatially and temporally dependent climate observations.
    """
    y,lower,upper=(np.asarray(v,dtype=np.float64) for v in (y,lower,upper))
    if y.ndim!=1 or lower.shape!=y.shape or upper.shape!=y.shape:
        raise ValueError('Calibration y/lower/upper must be matching one-dimensional arrays')
    if not len(y) or not 0<alpha<1 or not all(np.isfinite(v).all() for v in (y,lower,upper)):
        raise ValueError('Invalid or nonfinite calibration input')
    lower,upper=np.minimum(lower,upper),np.maximum(lower,upper)
    scores=np.maximum.reduce((lower-y,y-upper,np.zeros(len(y))))
    rank=int(math.ceil((len(scores)+1)*(1-alpha)))
    if rank>len(scores):raise ValueError('Too few calibration observations for the requested finite order statistic')
    correction=float(np.partition(scores,rank-1)[rank-1])
    return correction,{'n':len(scores),'rank_one_based':rank,'alpha':alpha,
                       'score_definition':'max(min(q10,q90)-y, y-max(q10,q90), 0)',
                       'score_sha256':_array_receipt(scores)['sha256'],'nonnegative_expansion':correction}


def _model_prediction(models,X,correction,return_raw=False):
    point=np.clip(models['mean'].predict(X),0,1)
    q10=models['lower'].predict(X);q90=models['upper'].predict(X)
    lower=np.clip(np.minimum(q10,q90)-correction,0,1)
    upper=np.clip(np.maximum(q10,q90)+correction,0,1)
    answer=(point,lower,upper,float(np.mean(q10>q90)))
    return answer+(q10,q90) if return_raw else answer


def _metric(y,point,lower=None,upper=None):
    error=np.asarray(point)-np.asarray(y)
    return {'n':len(y),'mae_pct':float(np.mean(np.abs(error))*100),
            'rmse_pct':float(np.sqrt(np.mean(error**2))*100),
            'coverage':None if lower is None else float(np.mean((y>=lower)&(y<=upper))),
            'width_pct':None if lower is None else float(np.mean(upper-lower)*100)}


def _masks(times,lat):
    early=times<np.datetime64('2023-01-11');late=times>=np.datetime64('2023-01-13')
    north=lat>0;south=lat<0
    return {'all':np.ones(len(times),dtype=bool),'early':early,'late':late,'nh':north,'sh':south,
            'early_nh':early&north,'early_sh':early&south,'late_nh':late&north,'late_sh':late&south}


def _support_checks(physical,minimum,maximum):
    values=np.asarray(physical,dtype=float)
    return np.any((values<minimum)|(values>maximum),axis=-1)


class CloudEstimator:
    """Read trusted locally generated artifacts and estimate one input setting."""
    def __init__(self,artifact_dir=DEFAULT_ARTIFACT_DIR):
        self.artifact_dir=Path(artifact_dir)
        self._manifest=json.loads((self.artifact_dir/'manifest.json').read_text(encoding='utf8'))
        if self._manifest.get('status')!='completed':raise ValueError('估计器训练尚未完成。')
        model_path=self.artifact_dir/'models.joblib'
        if _sha(model_path)!=self._manifest['outputs']['models.joblib']['sha256']:
            raise ValueError('本地模型文件校验失败。')
        self._metadata=json.loads((self.artifact_dir/'metadata.json').read_text(encoding='utf8'))
        self._bundle=joblib.load(model_path)

    def metadata(self):
        return copy.deepcopy(self._metadata)

    def predict(self,payload):
        if not isinstance(payload,dict):raise ValueError('输入必须是一个对象。')
        def number(name):
            if name not in payload or isinstance(payload[name],bool):raise ValueError(f'请填写{name}。')
            try:value=float(payload[name])
            except (ValueError,TypeError):raise ValueError(f'{name}必须是有限数值。') from None
            if not math.isfinite(value):raise ValueError(f'{name}必须是有限数值。')
            return value
        u,v,q=number('u_mps'),number('v_mps'),number('humidity_gkg')
        if not 0<=q<=1000:raise ValueError('比湿须在0至1000 g/kg之间；超出训练范围仍会另行提示。')
        mode=payload.get('mode','wind_humidity')
        if mode not in MODES:raise ValueError('mode须为wind_humidity或context。')
        requested=payload.get('horizon_hours',self._metadata['trained_horizons_hours'][0])
        if isinstance(requested,bool):raise ValueError('horizon_hours须为已训练的整数小时。')
        try:horizon=int(requested)
        except (TypeError,ValueError):raise ValueError('horizon_hours须为已训练的整数小时。') from None
        if float(requested)!=horizon or horizon not in self._metadata['trained_horizons_hours']:
            raise ValueError('该时间目标尚未训练。')
        region=month=None
        if payload.get('region_id') is not None:
            value=number('region_id');region=int(value)
            if value!=region or region not in self._bundle['node_ids'].tolist():raise ValueError('请选择已提供的12个区域之一。')
        if payload.get('month') is not None:
            value=number('month');month=int(value)
            if value!=month or not 1<=month<=12:raise ValueError('月份须为1至12的整数。')
        if mode=='context' and (region is None or month is None):raise ValueError('情境模式需要区域和月份。')
        physical=np.array([[u,v,q/1000]],dtype=np.float64)
        X=feature_matrix(physical,[region or 0],[month or 1],mode)
        item=self._bundle['estimators'][f'{mode}_h{horizon}']
        with threadpool_limits(limits=THREADS):point,lower,upper,_=_model_prediction(item['models'],X,item['correction'])
        minimum=self._bundle['support_min'];maximum=self._bundle['support_max']
        outside=bool(_support_checks(physical,minimum,maximum)[0])
        warnings=[];checks={}
        for j,name in enumerate(('u_mps','v_mps','humidity_gkg')):
            factor=1000 if j==2 else 1
            lo,hi=float(minimum[j]*factor),float(maximum[j]*factor);value=float(physical[0,j]*factor)
            flag=value<lo or value>hi
            checks[name]={'value':value,'training_min':lo,'training_max':hi,'outside':flag,'unit':'g/kg' if j==2 else 'm/s'}
            if flag:warnings.append(f'{name}={value:.4g}超出训练范围[{lo:.4g}, {hi:.4g}]，此处属于外推。')
        if not outside:warnings.append('各项数值在训练最小—最大范围内；这不保证该组合常见或估计准确。')
        if region is None or month is None:warnings.append('未同时提供地区和月份，因此不显示地区×月份基线。')
        warnings.append('这是统计条件估计；改变风或比湿输入不能解释为其对云量的因果作用。')
        warnings.append('区间名义覆盖率80%；时间、空间相关及来源变化下不保证个别输入的覆盖。')
        context_baseline=None
        if region is not None and month is not None:
            r=int(np.flatnonzero(self._bundle['node_ids']==region)[0])
            context_baseline=float(self._bundle['baselines'][str(horizon)]['region_month'][r,month-1]*100)
        return {'point':float(point[0]*100),'lower':float(lower[0]*100),'upper':float(upper[0]*100),
                'unit':'percent','mode':mode,'horizon_hours':horizon,
                'support':{'outside_training_range':outside,'warnings':warnings,'checks':checks,
                           'scope':'逐变量训练范围检查；未进行联合密度、最近邻或物理可实现性认证。'},
                'training_global_baseline_pct':float(self._bundle['baselines'][str(horizon)]['global']*100),
                'context_baseline_pct':context_baseline,'region_id':region,'month':month,
                'target':'同一时刻总云量' if horizon==0 else f'{horizon}小时后总云量'}


def train(artifact_dir=DEFAULT_ARTIFACT_DIR,horizons=(0,)):
    output=Path(artifact_dir)
    # The root agent authorized preserving the separately owned pre-training
    # design audit. Any existing training artifact still blocks a new run.
    if output.exists() and (not output.is_dir() or any(p.name!='independent_design_audit.json' for p in output.iterdir())):
        raise FileExistsError(f'Refusing to overwrite existing training artifacts: {output}')
    horizons=tuple(int(h) for h in horizons)
    if not horizons or len(set(horizons))!=len(horizons) or any(h not in (0,6,24) for h in horizons):
        raise ValueError('Horizons must be a unique nonempty selection from 0, 6, 24')
    started=time.perf_counter();rawdict=load_raw();raw=rawdict['raw'];times=rawdict['timestamps']
    lat,lon,nodes=rawdict['lat'],rawdict['lon'],rawdict['node_ids']
    all_regions=np.arange(len(nodes))
    indices,excluded=_indices(times,lat,all_regions,all_regions)
    rows={};fingerprints={}
    for split,ix in indices.items():
        t,r=ix.T;months=(times[t].astype('datetime64[M]').astype(np.int64)%12+1).astype(np.int64)
        physical=raw[t,r,:3].astype(np.float64)
        rows[split]={'physical':physical,'time_idx':t,'region_idx':r,'node_ids':nodes[r],
                     'month':months,'times':times[t],'lat':lat[r],
                     'X':{mode:feature_matrix(physical,nodes[r],months,mode) for mode in MODES}}
        fingerprints[split]={'indices':_array_receipt(ix),
                             'features':{mode:_array_receipt(rows[split]['X'][mode]) for mode in MODES},
                             'targets':{str(h):_array_receipt(raw[t+h//6,r,5].astype(np.float64)) for h in horizons}}
    minimum=rows['train']['physical'].min(0);maximum=rows['train']['physical'].max(0)
    protocol={'schema_version':1,'frozen_utc':_now(),'engine_sha256':_sha(__file__),
              'data_helper_sha256':_sha(ROOT/'revision/weather_tokens/data.py'),
              'source_hashes':rawdict['source_hashes'],'source_notes':rawdict['source_notes'],
              'horizons_hours':list(horizons),'target':'Regional raw total cloud cover fraction at t+h; horizon0 is same-time reconstruction, not future forecasting.',
              'feature_modes':{'wind_humidity':list(INPUT_NAMES),'context':list(CONTEXT_NAMES)},
              'feature_units':['m/s','m/s','kg/kg'],'ui_humidity_unit':'g/kg; divide by 1000 once at inference',
              'categorical_feature':'context column3 original region ID (12 categories); no ordinal numeric split on region',
              'models':{'mean':{'loss':'squared_error'},'lower':{'loss':'quantile','quantile':.1},'upper':{'loss':'quantile','quantile':.9}},
              'fixed_hyperparameters':HYPERPARAMETERS,'threadpool_limit':THREADS,
              'training_period':['1979-01-01','2014-12-31'],'calibration_period':['2015-01-01','2018-12-31'],
              'test_period':['2019-01-01','2025-12-31'],'selection':'No hyperparameter search, no early stopping, no test-based selection; validation only calibrates intervals.',
              'sample_rule':'Reuse pooled weather-token origin indices: 00UTC, t>=8, t+4 stays in split, t-8..t+4 cannot cross 2019-01-01 or 2023-01-11. Even horizon0 uses this conservative common sample set; model features use current values only.',
              'excluded_and_retained':excluded,'node_ids':nodes.tolist(),'lat':lat.tolist(),'lon':lon.tolist(),
              'fingerprints':fingerprints,
              'cqr':'Reorder q10/q90; validation score=max(lower-y,y-upper,0); k=ceil((n+1)*0.8), correction=sorted_score[k-1]; reject too-small calibration sets with k>n; expand equally, then clip to [0,1]. No guarantee under dependent or shifted data.',
              'baselines':'Training global target mean and training original-region by origin-calendar-month target mean, separate for each horizon.',
              'evaluation':'MAE/RMSE in percentage points; observed interval coverage and mean width. Separate early/late and NH/SH plus their intersections.',
              'version':{'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__,'joblib':joblib.__version__},
              'limitations':['Observational conditional estimation, not intervention or physical simulation.','Context gains cannot be attributed to wind/humidity alone.','Same-time estimation is distinct from the existing future token classification task; scores are not directly comparable.','Pressure levels may lie below terrain; source processing and coarse regional representation limitations persist.']}
    if output.exists() and any(p.name!='independent_design_audit.json' for p in output.iterdir()):
        raise FileExistsError('Artifact directory changed during input preparation')
    output.mkdir(parents=True,exist_ok=True)
    _save_json(output/'protocol.json',protocol)
    _save_json(output/'status.json',{'status':'training','started_utc':_now(),'models_complete':0})
    np.savez_compressed(output/'split_indices.npz',**{f'{k}_indices':v for k,v in indices.items()},
                        timestamps=times,node_ids=nodes,lat=lat,lon=lon)
    bundle={'schema_version':1,'node_ids':nodes.copy(),'estimators':{},'baselines':{},
            'support_min':minimum,'support_max':maximum}
    prediction_arrays={};metric_rows=[];model_records={};calibration={};fit_count=0
    try:
        for horizon in horizons:
            y={split:raw[row['time_idx']+horizon//6,row['region_idx'],5].astype(np.float64) for split,row in rows.items()}
            assert all(np.isfinite(v).all() and v.min()>=0 and v.max()<=1 for v in y.values())
            labels=rows['train']['region_idx']*12+rows['train']['month']-1
            counts=np.bincount(labels,minlength=len(nodes)*12)
            if (counts==0).any():raise ValueError('Missing training region-month baseline stratum')
            baseline=(np.bincount(labels,weights=y['train'],minlength=len(nodes)*12)/counts).reshape(len(nodes),12)
            bundle['baselines'][str(horizon)]={'global':float(y['train'].mean()),'region_month':baseline,'region_month_counts':counts.reshape(len(nodes),12)}
            for split in ('val','test'):
                prediction_arrays[f'observed_{split}_h{horizon}']=y[split]
                prediction_arrays[f'global_mean_{split}_h{horizon}']=np.full(len(y[split]),y['train'].mean())
                prediction_arrays[f'region_month_mean_{split}_h{horizon}']=baseline[rows[split]['region_idx'],rows[split]['month']-1]
                masks={'all':np.ones(len(y[split]),bool)} if split=='val' else _masks(rows[split]['times'],rows[split]['lat'])
                for name in ('global_mean','region_month_mean'):
                    pred=prediction_arrays[f'{name}_{split}_h{horizon}']
                    for subset,mask in masks.items():metric_rows.append({'mode':name,'horizon_hours':horizon,'split':split,'subset':subset,**_metric(y[split][mask],pred[mask])})
            for mode in MODES:
                key=f'{mode}_h{horizon}';fitted={};fit_started=time.perf_counter()
                for label,extra in [('mean',{'loss':'squared_error'}),('lower',{'loss':'quantile','quantile':.1}),('upper',{'loss':'quantile','quantile':.9})]:
                    estimator=HistGradientBoostingRegressor(**HYPERPARAMETERS,**extra,
                        categorical_features=[False,False,False,True,False,False] if mode=='context' else None)
                    with threadpool_limits(limits=THREADS):estimator.fit(rows['train']['X'][mode],y['train'])
                    fitted[label]=estimator;fit_count+=1
                    _save_json(output/'status.json',{'status':'training','updated_utc':_now(),'models_complete':fit_count,'last_model':key+'_'+label})
                    print('fitted',key,label,flush=True)
                with threadpool_limits(limits=THREADS):
                    val_low=fitted['lower'].predict(rows['val']['X'][mode]);val_high=fitted['upper'].predict(rows['val']['X'][mode])
                correction,receipt=conformal_expansion(y['val'],val_low,val_high)
                calibration[key]=receipt
                bundle['estimators'][key]={'models':fitted,'correction':correction,'feature_names':list(INPUT_NAMES if mode=='wind_humidity' else CONTEXT_NAMES)}
                model_records[key]={'n_features':int(fitted['mean'].n_features_in_),
                                    'nonconstant_training_features':int(np.count_nonzero(np.ptp(rows['train']['X'][mode],axis=0)>0)),
                                    'categorical_feature_mask':np.asarray(fitted['mean'].is_categorical_,dtype=bool).tolist() if fitted['mean'].is_categorical_ is not None else [False]*3,
                                    'iterations':{k:int(v.n_iter_) for k,v in fitted.items()},'elapsed_seconds':time.perf_counter()-fit_started,
                                    'fit_fingerprint':fingerprints['train']['features'][mode],
                                    'target_fingerprint':fingerprints['train']['targets'][str(horizon)]}
                for split in ('val','test'):
                    with threadpool_limits(limits=THREADS):point,lower,upper,crossing,q10,q90=_model_prediction(fitted,rows[split]['X'][mode],correction,return_raw=True)
                    for name,pred in [('point',point),('lower',lower),('upper',upper)]:prediction_arrays[f'{key}_{name}_{split}']=pred
                    prediction_arrays[f'{key}_raw_q10_{split}']=q10
                    prediction_arrays[f'{key}_raw_q90_{split}']=q90
                    masks={'all':np.ones(len(y[split]),bool)} if split=='val' else _masks(rows[split]['times'],rows[split]['lat'])
                    for subset,mask in masks.items():metric_rows.append({'mode':mode,'horizon_hours':horizon,'split':split,'subset':subset,**_metric(y[split][mask],point[mask],lower[mask],upper[mask])})
                    model_records[key][split+'_raw_quantile_crossing_fraction']=crossing
        joblib.dump(bundle,output/'models.joblib',compress=3)
        np.savez_compressed(output/'predictions.npz',**prediction_arrays)
        _save_json(output/'metrics.json',metric_rows)
        with (output/'metrics.csv').open('w',encoding='utf8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(metric_rows[0]));writer.writeheader();writer.writerows(metric_rows)
        examples=[]
        for stamp,node in [('1985-01-15',24),('1996-07-15',46),('2010-10-15',19),('2014-04-15',57)]:
            t=int(np.flatnonzero(times==np.datetime64(stamp,'ns'))[0]);r=int(np.flatnonzero(nodes==node)[0])
            assert times[t]<np.datetime64('2015-01-01')
            examples.append({'label':f'{stamp} · 区域{node}（训练期真实记录）','timestamp':str(times[t]),
                             'observed_cloud_pct':float(raw[t+horizons[0]//6,r,5]*100),
                             'payload':{'u_mps':float(raw[t,r,0]),'v_mps':float(raw[t,r,1]),'humidity_gkg':float(raw[t,r,2]*1000),
                                        'region_id':node,'month':int(stamp[5:7]),'mode':'context','horizon_hours':horizons[0]},
                             'notice':'用于演示输入；这是训练期记录，展示预测不代表泛化能力。'})
        support={split:{'n':len(row['physical']),'outside_any_training_feature_range_fraction':float(_support_checks(row['physical'],minimum,maximum).mean())} for split,row in rows.items()}
        metadata={'schema_version':1,'title':'本地风湿—总云量统计估计器','trained_horizons_hours':list(horizons),
                  'modes':list(MODES),'mode_labels':{'wind_humidity':'仅风和比湿','context':'风、比湿及地区/月度情境'},
                  'regions':[{'id':int(n),'lat':float(a),'lon':float(o)} for n,a,o in zip(nodes,lat,lon)],
                  'input_units':{'u_mps':'850 hPa向东风分量，m/s','v_mps':'850 hPa向北风分量，m/s','humidity_gkg':'850 hPa比湿，g/kg','region_id':'原始区域编号','month':'日历月份1–12'},
                  'prediction_unit':'percent','interval_nominal_coverage':.8,'examples':examples,
                  'test_metrics':[r for r in metric_rows if r['split']=='test' and r['mode'] in MODES],
                  'baseline_metrics':[r for r in metric_rows if r['split']=='test' and r['mode'] not in MODES],
                  'baseline_labels':{'global_mean':'训练全局平均云量','region_month_mean':'训练地区×月份平均云量'},
                  'training_ranges':{name:{'minimum':float(minimum[j]*(1000 if j==2 else 1)),'maximum':float(maximum[j]*(1000 if j==2 else 1)),'unit':'g/kg' if j==2 else 'm/s'} for j,name in enumerate(('u_mps','v_mps','humidity_gkg'))},
                  'training_support_summary':support,'split_counts':{k:len(v) for k,v in indices.items()},
                  'model_records':model_records,'calibration':calibration,'protocol_sha256':_sha(output/'protocol.json'),
                  'method_notes':['horizon=0估计同一时刻总云量，不是未来天气预报。','当前输入只有u/v风分量和比湿；情境模式另加入区域类别及月份周期。','2015–2018年只用于区间校准；模型超参数事先固定，未依据测试数据调节。','测试来自2019–2025年再分析资料；early与late分开报告，不能代表业务实时预报。','MAE、RMSE和区间宽度以百分点表示；coverage是0–1覆盖比例。','区间使用验证期非负CQR扩展，名义80%，不保证时空相关/分布变化下的覆盖。','观察性估计不是因果干预；不施加湿度或风的单调因果关系。','训练期示例只展示交互功能，不能作为泛化效果证据。','与原来的未来云量变化词元分类任务不同，不能直接比较原有分数。'],
                  'source_notes':rawdict['source_notes']}
        _save_json(output/'metadata.json',metadata)
        manifest={'status':'completed','completed_utc':_now(),'elapsed_seconds':time.perf_counter()-started,
                  'trained_horizons_hours':list(horizons),'fitted_models':fit_count,'protocol_sha256':_sha(output/'protocol.json'),
                  'outputs':{name:{'bytes':(output/name).stat().st_size,'sha256':_sha(output/name)} for name in ('protocol.json','split_indices.npz','models.joblib','predictions.npz','metrics.json','metrics.csv','metadata.json')},
                  'no_existing_results_modified':True}
        _save_json(output/'manifest.json',manifest)
        _save_json(output/'status.json',{'status':'completed','completed_utc':_now(),'models_complete':fit_count})
        print(json.dumps({'status':'completed','elapsed_seconds':manifest['elapsed_seconds'],'models':fit_count,'artifact_dir':str(output)}),flush=True)
        return manifest
    except Exception as error:
        _save_json(output/'status.json',{'status':'failed','updated_utc':_now(),'models_complete':fit_count,'error_type':type(error).__name__,'message':str(error)})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument('--horizons',nargs='+',type=int,default=[0],choices=[0,6,24])
    args=parser.parse_args();train(args.output,args.horizons)


if __name__=='__main__':main()
