"""Paired participant bootstrap, conditional on frozen fitted models."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import hashlib
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

ROOT = Path(__file__).resolve().parent
NAMES = ['logistic_regression', 'hist_gradient_boosting', 'catboost', 'mlp', 'bionutrigpt', 'ft_transformer']
OUTCOMES = ['TARGET_DIABETES', 'TARGET_OBESITY', 'TARGET_CVD']
DEST = ROOT / 'Research Paper/Stage1_Paired_Uncertainty_v2'
PROTOCOL = dict(seed=20261119, replicates=20000,
    resampling='Ordinary unweighted participant bootstrap; same row multiplicities across all models and outcomes',
    contrasts='BioNutriGPT minus each of five comparators; no best-comparator selection',
    primary='Diabetes and CVD: raw AUROC and calibrated Brier; 20 contrasts',
    intervals='Percentile 95% pointwise; Bonferroni 99.75% individual intervals for the 20 primary contrasts (approximate bootstrap family-wise coverage)',
    secondary='Obesity contrasts are exploratory; raw Brier contrasts descriptive only',
    limitations='Post-results analysis, not preregistered. Conditional on fitted models; no training/selection/calibrator uncertainty. Not complex-survey inference. No clinical equivalence margin.',
    compute='Up to 12 independent CPU workers, one BLAS thread each; GPU not used for rank/bootstrap operations')

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def load():
    frames = []; hashes = {}
    truth = pd.read_csv(ROOT/'Data/stage1_repaired_v2/temporal_ml_ready.csv').sort_values('SEQN')
    for name in NAMES:
        folder = ROOT/'Saved Model/stage1_comparison_v2'/name
        completion = folder/'completed.json'
        rec = json.loads(completion.read_text())
        verification = json.loads((ROOT/f'Research Paper/Stage1_{name}_v2_Verification.json').read_text())
        assert verification['status'] == 'passed' and verification['completed_record_sha256'] == sha(completion)
        path = folder/'temporal_predictions_LOCAL.csv'
        assert sha(path) == rec['artifact_hashes'][path.name]
        frame = pd.read_csv(path).sort_values('SEQN')
        assert frame.SEQN.is_unique
        np.testing.assert_array_equal(frame.SEQN.to_numpy(), truth.SEQN.to_numpy())
        np.testing.assert_array_equal(frame[OUTCOMES].to_numpy(), truth[OUTCOMES].to_numpy())
        for o in OUTCOMES:
            for v in ['raw', 'calibrated']:
                p = frame[o+'_'+v].to_numpy()
                assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        hashes[name] = dict(completion=sha(completion), predictions=sha(path))
        frames.append(frame)
    return frames, hashes

def prepare(frames):
    specs=[]; labels=[]
    for name, f in zip(NAMES, frames):
        for o in OUTCOMES:
            y=f[o].to_numpy(dtype=float)
            p=f[o+'_raw'].to_numpy()
            _, groups=np.unique(p, return_inverse=True)
            specs.append(('auc', y, groups, int(groups.max()+1)))
            labels.append((name,o,'raw_auroc'))
            for v in ['raw','calibrated']:
                specs.append(('mean',(f[o+'_'+v].to_numpy()-y)**2,None,None))
                labels.append((name,o,v+'_brier'))
    return specs,labels

def evaluate(w,specs):
    vals=[]; n=w.sum()
    for kind, a, groups, size in specs:
        if kind=='mean': vals.append(np.dot(w,a)/n)
        else:
            pos=np.bincount(groups,weights=w*a,minlength=size)
            neg=np.bincount(groups,weights=w*(1-a),minlength=size)
            denom=pos.sum()*neg.sum()
            vals.append(np.dot(pos,np.cumsum(neg)-0.5*neg)/denom if denom else np.nan)
    return np.array(vals)

SPECS=None
def initialize(specs):
    global SPECS
    SPECS=specs

def chunk(args):
    start,stop,n,seed=args
    result=[]
    for i in range(start,stop):
        rng=np.random.default_rng(np.random.SeedSequence([seed,i]))
        weights=np.bincount(rng.integers(n,size=n),minlength=n)
        result.append(evaluate(weights,SPECS))
    return np.array(result)

def check():
    # Tie-aware weighted ranks must match explicit repeated-row sklearn metrics.
    y=np.array([0.,1.,0.,1.,1.,0.]); p=np.array([.1,.4,.4,.4,.9,.8])
    _,g=np.unique(p,return_inverse=True)
    specs=[('auc',y,g,len(np.unique(p))),('mean',(p-y)**2,None,None)]
    for seed in range(40):
        ix=np.random.default_rng(seed).integers(len(y),size=24)
        if len(np.unique(y[ix]))<2: continue
        got=evaluate(np.bincount(ix,minlength=len(y)),specs)
        np.testing.assert_allclose(got,[roc_auc_score(y[ix],p[ix]),brier_score_loss(y[ix],p[ix])],atol=1e-12)
    identical=evaluate(np.ones(len(y)),specs+specs)
    np.testing.assert_array_equal(identical[:2]-identical[2:],np.zeros(2))
    print('Weighted bootstrap metric checks passed (ties, repeated rows, identical models).',flush=True)

def main(run=False):
    check()
    frames,hashes=load(); specs,labels=prepare(frames); n=len(frames[0]); point=evaluate(np.ones(n),specs)
    for i,(name,o,metric) in enumerate(labels):
        f=frames[NAMES.index(name)]; y=f[o]
        ref=roc_auc_score(y,f[o+'_raw']) if metric=='raw_auroc' else brier_score_loss(y,f[o+'_'+metric.split('_')[0]])
        np.testing.assert_allclose(point[i],ref,atol=1e-12)
    print('All six verified inputs aligned; all 54 point metrics match sklearn.',flush=True)
    if not run:return
    DEST.mkdir(exist_ok=True)
    if (DEST/'results.json').exists():raise RuntimeError('Completed analysis exists; refusing overwrite')
    (DEST/'protocol.json').write_text(json.dumps(PROTOCOL,indent=2)+'\n')
    workers=min(12,os.cpu_count() or 1); b=PROTOCOL['replicates']
    batches=[(i,min(i+250,b),n,PROTOCOL['seed']) for i in range(0,b,250)]
    arrays=[]
    with ProcessPoolExecutor(max_workers=workers,initializer=initialize,initargs=(specs,)) as pool:
        for j,arr in enumerate(pool.map(chunk,batches)):
            arrays.append(arr)
            if (j+1)%8==0:print(f'Bootstrap: {min((j+1)*250,b)}/{b}',flush=True)
    draws=np.concatenate(arrays); assert np.isfinite(draws).all()
    np.savez_compressed(DEST/'bootstrap_metrics.npz',metrics=draws,labels=np.array(labels))
    metrics=[]; contrasts=[]
    for i,(name,o,metric) in enumerate(labels):
        lo,hi=np.quantile(draws[:,i],[.025,.975])
        metrics.append(dict(model=name,outcome=o,metric=metric,estimate=point[i],ci95_low=lo,ci95_high=hi))
    for o in OUTCOMES:
        for metric in ['raw_auroc','raw_brier','calibrated_brier']:
            a=labels.index(('bionutrigpt',o,metric))
            for name in NAMES:
                if name=='bionutrigpt':continue
                c=labels.index((name,o,metric)); d=draws[:,a]-draws[:,c]
                lo,hi=np.quantile(d,[.025,.975]); primary=o!='TARGET_OBESITY' and metric!='raw_brier'
                entry=dict(comparator=name,outcome=o,metric=metric,difference=point[a]-point[c],ci95_low=lo,ci95_high=hi,primary=primary)
                if primary:
                    low,high=np.quantile(d,[.05/(2*20),1-.05/(2*20)])
                    entry.update(family_adjusted_low=low,family_adjusted_high=high)
                contrasts.append(entry)
    pd.DataFrame(metrics).to_csv(DEST/'metric_intervals.csv',index=False)
    pd.DataFrame(contrasts).to_csv(DEST/'paired_contrasts.csv',index=False)
    result=dict(protocol=PROTOCOL,n=n,workers=workers,input_hashes=hashes,script_sha256=sha(Path(__file__)),metrics=metrics,contrasts=contrasts,
        invalid_replicates=0,checks=['Matched unique participant IDs and labels against frozen temporal data','Completion and prediction hashes match independently verified artifacts','Tie-aware repeated-row metrics match sklearn','All 54 original-sample metrics match sklearn','All bootstrap values finite'],
        artifacts={name:sha(DEST/name) for name in ['protocol.json','bootstrap_metrics.npz','metric_intervals.csv','paired_contrasts.csv']})
    (DEST/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Completed paired uncertainty analysis.',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',action='store_true');main(ap.parse_args().run)
