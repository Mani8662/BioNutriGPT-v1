"""Checkpointed, one-family-at-a-time comparison on the frozen repaired splits.

Default action is preflight only. --run-model NAME performs an independent study run.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import gc
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import random
import threading
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import psutil
from scipy.special import expit, logit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss, confusion_matrix
from threadpoolctl import threadpool_limits
import torch
from torch import nn

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'Data/stage1_repaired_v2'
PROTOCOL_PATH=ROOT/'stage1_comparison_protocol_v2.json'
RESULTS=ROOT/'Saved Model/stage1_comparison_v2'
NEURAL={'mlp','bionutrigpt','ft_transformer'}
TARGETS=['TARGET_DIABETES','TARGET_OBESITY','TARGET_CVD']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    tmp.replace(path)


def save_joblib(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');joblib.dump(value,tmp);tmp.replace(path)


def save_torch(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');torch.save(value,tmp);tmp.replace(path)


def probabilities(p):
    p=np.asarray(p,dtype=float)
    if not np.isfinite(p).all() or np.any(p<0) or np.any(p>1):
        raise ValueError('Nonfinite or out-of-range probabilities.')
    return np.clip(p,1e-7,1-1e-7)


def validate_partitions(parts,features):
    seen=set()
    for name,df in parts.items():
        if list(df.columns)!=['SEQN','NHANES_CYCLE']+TARGETS+features:
            raise ValueError(f'{name}: exact schema/order mismatch')
        ids=set(df.SEQN)
        if df.SEQN.isna().any() or len(ids)!=len(df) or seen&ids:
            raise ValueError(f'{name}: duplicate, missing or overlapping participant IDs')
        seen |= ids
        if not np.isfinite(df[features].to_numpy()).all():raise ValueError('Nonfinite features')
        for t in TARGETS:
            if set(df[t].unique())!={0,1}:raise ValueError(f'{name}: invalid or single-class target {t}')


def load_data():
    report=json.loads((DATA/'repair_report.json').read_text())
    if report['version']!='stage1-foundation-v2':raise ValueError('Wrong data version')
    for name,h in report['output_hashes'].items():
        if sha(DATA/name)!=h:raise ValueError(f'Frozen data hash mismatch: {name}')
    features=report['feature_order']
    if len(features)!=87 or len(set(features))!=87:raise ValueError('Feature contract mismatch')
    from train_bionutrigpt_stage1 import FEATURE_COLS
    if features!=FEATURE_COLS:raise ValueError('Architecture feature order differs from frozen data')
    parts={k:pd.read_csv(DATA/f'{k}_ml_ready.csv') for k in ['training','validation','calibration','temporal']}
    validate_partitions(parts,features)
    manifest=pd.read_csv(DATA/'split_membership_LOCAL.csv')
    for k,df in parts.items():
        if set(df.SEQN)!=set(manifest.loc[manifest.partition==k,'SEQN']):raise ValueError('Split manifest mismatch')
    return {k:{'x':df[features].to_numpy(dtype=np.float32),'y':df[TARGETS].to_numpy(dtype=np.float32),
               'ids':df.SEQN.to_numpy(dtype=np.int64)} for k,df in parts.items()},features


def verify_frozen_inputs():
    report=json.loads((DATA/'repair_report.json').read_text())
    for name,h in report['output_hashes'].items():
        if sha(DATA/name)!=h:raise RuntimeError(f'Frozen input changed during execution: {name}')


def selection_score(y,p,indices=(0,2)):
    p=probabilities(p)
    return float(np.mean([roc_auc_score(y[:,i],p[:,i]) for i in indices]))


def fit_calibrators(y,p):
    """Fit ONLY to reserved calibration labels; caller never supplies temporal labels."""
    p=probabilities(p);out=[]
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        for i in range(3):
            c=LogisticRegression(C=np.inf,solver='lbfgs',max_iter=2000,tol=1e-10)
            c.fit(logit(p[:,i]).reshape(-1,1),y[:,i]);out.append(c)
    return out


def apply_calibrators(calibrators,p):
    p=probabilities(p)
    return np.column_stack([c.predict_proba(logit(p[:,i]).reshape(-1,1))[:,1] for i,c in enumerate(calibrators)])


def metrics(y,p):
    p=probabilities(p);rows=[];curves=[]
    for i,target in enumerate(TARGETS):
        z=logit(p[:,i]).reshape(-1,1)
        # Diagnostic fit on evaluation data, never applied to evaluation predictions.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',ConvergenceWarning)
            diagnostic=LogisticRegression(C=np.inf,max_iter=2000,tol=1e-10).fit(z,y[:,i])
        tn,fp,fn,tp=confusion_matrix(y[:,i],p[:,i]>=.5,labels=[0,1]).ravel()
        divide=lambda a,b:float(a/b) if b else None
        rows.append({'outcome':target,'n':len(y),'positive':int(y[:,i].sum()),
            'auroc':float(roc_auc_score(y[:,i],p[:,i])),'average_precision':float(average_precision_score(y[:,i],p[:,i])),
            'brier':float(brier_score_loss(y[:,i],p[:,i])),'log_loss':float(log_loss(y[:,i],p[:,i])),
            'calibration_intercept':float(diagnostic.intercept_[0]),'calibration_slope':float(diagnostic.coef_[0,0]),
            'calibration_diagnostic_convergence_warning':any(issubclass(w.category,ConvergenceWarning) for w in caught),
            'threshold':.5,'threshold_scope':'fixed descriptive threshold; not clinically validated',
            'sensitivity':divide(tp,tp+fn),'specificity':divide(tn,tn+fp),'ppv':divide(tp,tp+fp),'npv':divide(tn,tn+fn)})
        # Equal-width bins avoid splitting identical predicted values across bins.
        bins=np.minimum((p[:,i]*10).astype(int),9)
        for b in range(10):
            mask=bins==b
            curves.append({'outcome':target,'bin':b,'n':int(mask.sum()),
                'mean_probability':float(p[mask,i].mean()) if mask.any() else None,
                'observed_fraction':float(y[mask,i].mean()) if mask.any() else None})
    return rows,curves


class MLP(nn.Module):
    def __init__(self,cfg):
        super().__init__();layers=[];prev=87
        for h in cfg['hidden']:
            layers += [nn.Linear(prev,h),nn.LayerNorm(h),nn.ReLU(),nn.Dropout(cfg['dropout'])];prev=h
        self.net=nn.Sequential(*layers,nn.Linear(prev,3))
    def forward(self,x):return self.net(x)


def make_neural(name,cfg):
    if name=='mlp':return MLP(cfg)
    if name=='bionutrigpt':
        from train_bionutrigpt_stage1 import BioNutriGPTStage1
        return BioNutriGPTStage1()
    if name=='ft_transformer':
        from rtdl_revisiting_models import FTTransformer
        return FTTransformer(n_cont_features=87,cat_cardinalities=[],d_out=3,**FTTransformer.get_default_kwargs(cfg['n_blocks']))
    raise ValueError(name)


def forward(name,model,x):
    if name=='bionutrigpt':
        heads,aux,_=model(x)
        return torch.stack([heads[k] for k in ['diabetes','obesity','cvd']],dim=1),aux
    return (model(x,None) if name=='ft_transformer' else model(x)),x.new_zeros(())


def predict(name,model,x,device,batch=128):
    if name not in NEURAL:
        return probabilities(np.column_stack([m.predict_proba(x)[:,1] for m in model]))
    model.eval();result=[]
    with torch.inference_mode():
        for start in range(0,len(x),batch):
            logits,_=forward(name,model,torch.as_tensor(x[start:start+batch],device=device))
            result.append(torch.sigmoid(logits).cpu().numpy())
    return probabilities(np.concatenate(result))


@contextmanager
def memory_monitor():
    process=psutil.Process();stop=threading.Event();stats={'peak_process_rss_bytes':process.memory_info().rss}
    def poll():
        while not stop.wait(.1):stats['peak_process_rss_bytes']=max(stats['peak_process_rss_bytes'],process.memory_info().rss)
    thread=threading.Thread(target=poll,daemon=True);thread.start()
    try:yield stats
    finally:stop.set();thread.join()


def train_trial(name,cfg,train,val,protocol,folder,device):
    seed=protocol['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if device=='cuda':torch.cuda.manual_seed_all(seed);torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter();epochs=[]
    with memory_monitor() as memory:
        if name in NEURAL:
            model=make_neural(name,cfg).to(device)
            opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
            best=-np.inf;stale=0;settings=protocol['neural']
            x=torch.as_tensor(train['x'],device=device);y=torch.as_tensor(train['y'],device=device)
            for epoch in range(settings['max_epochs']):
                model.train();order=torch.randperm(len(x),device=device);total=0.
                for start_idx in range(0,len(x),settings['batch_size']):
                    idx=order[start_idx:start_idx+settings['batch_size']];opt.zero_grad(set_to_none=True)
                    logits,aux=forward(name,model,x[idx]);loss=nn.functional.binary_cross_entropy_with_logits(logits,y[idx])
                    loss=loss+settings['moe_aux_coefficient']*aux
                    if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                    loss.backward();nn.utils.clip_grad_norm_(model.parameters(),settings['gradient_clip']);opt.step()
                    total+=float(loss.detach())*len(idx)
                vp=predict(name,model,val['x'],device);score=selection_score(val['y'],vp)
                row={'epoch':epoch+1,'training_loss':total/len(x),'validation_selection_auroc':score}
                epochs.append(row)
                with (folder/'epochs.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
                print(f'{name} {folder.name} epoch {epoch+1}: validation AUROC={score:.5f}',flush=True)
                if score>best:
                    best=score;stale=0
                    save_torch(folder/'model.pt',{k:v.detach().cpu() for k,v in model.state_dict().items()})
                else:stale+=1
                if stale>=settings['patience']:break
            model.load_state_dict(torch.load(folder/'model.pt',map_location=device,weights_only=True))
            del x,y,opt
        else:
            model=[]
            for i in range(3):
                if name=='logistic_regression':m=LogisticRegression(**cfg,max_iter=3000,tol=1e-6,solver='lbfgs',random_state=seed)
                elif name=='hist_gradient_boosting':m=HistGradientBoostingClassifier(**cfg,early_stopping=False,random_state=seed)
                elif name=='catboost':
                    from catboost import CatBoostClassifier
                    m=CatBoostClassifier(**cfg,task_type='GPU',devices='0',thread_count=os.cpu_count() or 1,
                        random_seed=seed,loss_function='Logloss',eval_metric='AUC',allow_writing_files=False,verbose=False)
                else:raise ValueError(name)
                with warnings.catch_warnings():
                    warnings.simplefilter('error',ConvergenceWarning)
                    if name=='catboost':m.fit(train['x'],train['y'][:,i],eval_set=(val['x'],val['y'][:,i]),early_stopping_rounds=40,use_best_model=True)
                    else:m.fit(train['x'],train['y'][:,i])
                model.append(m)
            save_joblib(folder/'model.joblib',model)
        vp=predict(name,model,val['x'],device)
    elapsed=time.perf_counter()-start
    summary={'configuration':cfg,'seed':seed,'selection_score':selection_score(val['y'],vp),
        'training_seconds_including_validation_and_checkpointing':elapsed,'epochs':epochs,**memory,
        'cuda_peak_allocated_bytes':int(torch.cuda.max_memory_allocated()) if name in NEURAL and device=='cuda' else None,
        'cuda_memory_scope':'PyTorch allocator only; CatBoost allocation is not measured',
        'parameter_count':sum(p.numel() for p in model.parameters()) if name in NEURAL else None}
    return model,vp,summary


def load_trial(name,cfg,folder,device):
    if name in NEURAL:
        model=make_neural(name,cfg).to(device)
        model.load_state_dict(torch.load(folder/'model.pt',map_location=device,weights_only=True));return model
    return joblib.load(folder/'model.joblib')


def verify_completion(folder,filename):
    record=json.loads((folder/filename).read_text())
    for name,h in record['artifact_hashes'].items():
        if sha(folder/name)!=h:raise ValueError(f'Completed artifact changed: {folder/name}')
    return record


def latency(name,model,x,device):
    rows=[]
    for batch in [1,128]:
        sample=x[:batch]
        for _ in range(3):predict(name,model,sample,device)
        timings=[]
        for _ in range(20):
            if name in NEURAL and device=='cuda':torch.cuda.synchronize()
            start=time.perf_counter();predict(name,model,sample,device)
            if name in NEURAL and device=='cuda':torch.cuda.synchronize()
            timings.append((time.perf_counter()-start)*1000)
        rows.append({'batch_size':len(sample),'median_ms':float(np.median(timings)),'p95_ms':float(np.percentile(timings,95)),
            'inference_device':'cpu' if name not in NEURAL else device,'repeats':20})
    return rows


def environment():
    return {'python':platform.python_version(),'cpu_logical_cores':os.cpu_count(),
        'cuda_available':torch.cuda.is_available(),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'versions':{p:version(p) for p in ['torch','numpy','pandas','scikit-learn','catboost','rtdl-revisiting-models','joblib']}}


def binding(protocol):
    return {'protocol_sha256':sha(PROTOCOL_PATH),'data_report_sha256':sha(DATA/'repair_report.json'),
        'runner_sha256':sha(Path(__file__)),'architecture_sha256':sha(ROOT/'train_bionutrigpt_stage1.py'),
        'protocol_version':protocol['version'],'environment':environment()}


def run_model(name,data,features,protocol,root):
    device='cuda' if name in NEURAL or name=='catboost' else 'cpu'
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA required; no silent CPU fallback.')
    folder=root/name;folder.mkdir(parents=True,exist_ok=True)
    current=binding(protocol)
    if (folder/'run_binding.json').exists():
        if json.loads((folder/'run_binding.json').read_text())!=current:raise ValueError('Run source/data/environment binding changed; create a new study version.')
    else:write_json(folder/'run_binding.json',current)
    if (folder/'completed.json').exists():
        verify_completion(folder,'completed.json')
        for i in range(len(protocol['models'][name])):verify_completion(folder/f'trial_{i:02d}','completed.json')
        print(f'{name}: already complete and verified.');return
    trials=[]
    for i,cfg in enumerate(protocol['models'][name]):
        trial=folder/f'trial_{i:02d}';trial.mkdir(exist_ok=True)
        if (trial/'completed.json').exists():summary=verify_completion(trial,'completed.json')
        else:
            # Explicitly restart only an interrupted trial; no partial logs masquerade as one run.
            if (trial/'epochs.jsonl').exists():
                (trial/'epochs.jsonl').rename(trial/f'epochs_interrupted_{time.time_ns()}.jsonl')
            model,vp,summary=train_trial(name,cfg,data['training'],data['validation'],protocol,trial,device)
            np.save(trial/'validation_probabilities.npy',vp)
            model_file='model.pt' if name in NEURAL else 'model.joblib'
            summary['artifact_hashes']={p:sha(trial/p) for p in [model_file,'validation_probabilities.npy']}
            write_json(trial/'completed.json',summary)
            del model;gc.collect()
            if device=='cuda':torch.cuda.empty_cache()
        trials.append(summary)
    best=max(range(len(trials)),key=lambda i:trials[i]['selection_score'])
    write_json(folder/'selection.json',{'selected_trial':best,'criterion':protocol['selection'],'validation_scores':[t['selection_score'] for t in trials],
        'selected_before_calibration_and_temporal_prediction':True})
    model=load_trial(name,protocol['models'][name][best],folder/f'trial_{best:02d}',device)
    # No calibration or temporal labels are passed into trial training or selection.
    cp=predict(name,model,data['calibration']['x'],device)
    calibrators=fit_calibrators(data['calibration']['y'],cp)
    save_joblib(folder/'calibrators.joblib',calibrators)
    raw=predict(name,model,data['temporal']['x'],device)
    calibrated=apply_calibrators(calibrators,raw)
    prediction_files=[]
    for part,pred in [('validation',np.load(folder/f'trial_{best:02d}'/'validation_probabilities.npy')),('calibration',cp),('temporal',raw)]:
        df=pd.DataFrame({'SEQN':data[part]['ids']})
        adj=apply_calibrators(calibrators,pred)
        for i,t in enumerate(TARGETS):
            df[t]=data[part]['y'][:,i].astype(int);df[t+'_raw']=pred[:,i];df[t+'_calibrated']=adj[:,i]
        filename=f'{part}_predictions_LOCAL.csv';df.to_csv(folder/filename,index=False);prediction_files.append(filename)
    all_metrics=[];all_curves=[]
    for variant,pred in [('raw',raw),('calibrated',calibrated)]:
        rows,curves=metrics(data['temporal']['y'],pred)
        all_metrics += [{'model':name,'variant':variant,**row} for row in rows]
        all_curves += [{'model':name,'variant':variant,**row} for row in curves]
    pd.DataFrame(all_metrics).to_csv(folder/'temporal_metrics.csv',index=False)
    pd.DataFrame(all_curves).to_csv(folder/'calibration_curve.csv',index=False)
    result={'model':name,'selected_trial':best,'selected_configuration':protocol['models'][name][best],
        'calibration':[{'outcome':TARGETS[i],'intercept':float(c.intercept_[0]),'logit_slope':float(c.coef_[0,0])} for i,c in enumerate(calibrators)],
        'metrics':all_metrics,'latency':latency(name,model,data['temporal']['x'],device),
        'training_seconds_all_configurations':sum(t['training_seconds_including_validation_and_checkpointing'] for t in trials),
        'training_device':device,'limits':['Single seed; modest four-configuration search','Unweighted selected-sample analysis',
            'No confidence intervals or paired superiority inference yet','Already-inspected historical temporal period','No deployment update'],
        'data_version':protocol['data_version'],'n':{k:len(v['ids']) for k,v in data.items()},
        'artifact_hashes':{p:sha(folder/p) for p in prediction_files+['run_binding.json','selection.json','calibrators.joblib','temporal_metrics.csv','calibration_curve.csv']}}
    # Bind completed trials as well as final exports.
    result['artifact_hashes'].update({f'trial_{i:02d}/completed.json':sha(folder/f'trial_{i:02d}'/'completed.json') for i in range(len(trials))})
    if binding(protocol)!=current:raise RuntimeError('Source binding changed during run; completion requires review.')
    verify_frozen_inputs()
    write_json(folder/'completed.json',result)
    print(f'{name}: completed, checkpointed and exported to {folder}',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-model',choices=['logistic_regression','hist_gradient_boosting','catboost','mlp','bionutrigpt','ft_transformer'])
    parser.add_argument('--output',type=Path,default=RESULTS)
    args=parser.parse_args()
    protocol=json.loads(PROTOCOL_PATH.read_text());data,features=load_data()
    if not args.run_model:
        print(json.dumps({'status':'preflight only; no training executed','environment':environment(),
            'partitions':{k:len(v['ids']) for k,v in data.items()},'features':len(features),
            'configurations':{k:len(v) for k,v in protocol['models'].items()},'binding':binding(protocol)},indent=2));return
    cores=os.cpu_count() or 1;torch.set_num_threads(cores)
    with threadpool_limits(limits=cores):run_model(args.run_model,data,features,protocol,args.output.resolve())


if __name__=='__main__':main()
