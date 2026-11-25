"""Read-only numerical audit; writes a separate audit receipt after success."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.optimize import root
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'Saved Model/stage1_subgroups_v1'

def weighted_auc(y, p, counts):
    order = np.argsort(p)
    starts = np.r_[0, np.flatnonzero(np.diff(p[order])) + 1]
    positive = np.add.reduceat(counts[:, order] * y[order], starts, axis=1)
    negative = np.add.reduceat(counts[:, order] * (1-y[order]), starts, axis=1)
    denom = positive.sum(1) * negative.sum(1)
    numerator = (positive * (np.cumsum(negative, axis=1) - negative / 2)).sum(1)
    return np.divide(numerator, denom, out=np.full(len(counts), np.nan), where=denom>0)

def main():
    part = pd.read_csv(OUT/'participant_subgroups_LOCAL.csv').sort_values('SEQN')
    source = pd.read_csv(ROOT/'Data/stage1_repaired_v2/survey_metadata_LOCAL.csv').set_index('SEQN').loc[part.SEQN]
    for col in ['RIAGENDR', 'RIDAGEYR', 'RIDRETH1']:
        np.testing.assert_array_equal(part[col], source[col])
    np.testing.assert_array_equal(part.sex_group, part.RIAGENDR.map({1:'Male',2:'Female'}))
    np.testing.assert_array_equal(part.age_group, np.where(part.RIDAGEYR<=40,'20-40',np.where(part.RIDAGEYR<=65,'41-65','>65')))
    np.testing.assert_array_equal(part.race_group, part.RIDRETH1.map({1:'Mexican American',2:'Other Hispanic',3:'Non-Hispanic White',4:'Non-Hispanic Black',5:'Other Race - Including Multi-Racial'}))
    metrics = pd.read_csv(OUT/'subgroup_metrics.csv')
    intervals = pd.read_csv(OUT/'subgroup_bootstrap_intervals.csv')
    diffs = pd.read_csv(OUT/'subgroup_paired_differences.csv')
    preds = {m:pd.read_csv(ROOT/'Saved Model/stage1_comparison_v2'/m/'temporal_predictions_LOCAL.csv').sort_values('SEQN') for m in metrics.model.unique()}
    truth = pd.read_csv(ROOT/'Data/stage1_repaired_v2/temporal_ml_ready.csv').sort_values('SEQN')
    for frame in preds.values():
        np.testing.assert_array_equal(frame.SEQN,part.SEQN)
        for t in metrics.outcome.unique():
            np.testing.assert_array_equal(frame[t],truth[t])
            for variant in ['raw','calibrated']:
                p=frame[f'{t}_{variant}'].values
                assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    columns={'Sex':'sex_group','Age':'age_group','Race_Ethnicity':'race_group'}
    # Independently solve logistic score equations for calibration diagnostics.
    for row in metrics.itertuples():
        mask=(part[columns[row.dimension]]==row.subgroup).values
        y=truth[row.outcome].values[mask]
        z=logit(np.clip(preds[row.model][f'{row.outcome}_{row.variant}'].values[mask],1e-7,1-1e-7))
        fit=root(lambda b: np.array([(expit(b[0]+b[1]*z)-y).sum(),((expit(b[0]+b[1]*z)-y)*z).sum()]),[0.,1.])
        assert fit.success and not row.calibration_convergence_warning
        np.testing.assert_allclose(fit.x,[row.joint_calibration_intercept,row.joint_calibration_slope],atol=2e-4,rtol=0)
        assert abs(np.mean(expit(row.calibration_in_the_large+z)-y))<1e-9
    # Check independent weighted-tie AUROC implementation against expanded samples.
    yy=np.array([0,1,0,1]); pp=np.array([.2,.2,.8,.9]); cc=np.array([[2,1,3,2],[1,0,1,0]])
    assert np.isclose(weighted_auc(yy,pp,cc)[0],roc_auc_score(np.repeat(yy,cc[0]),np.repeat(pp,cc[0])))
    assert np.isnan(weighted_auc(yy,pp,cc)[1])
    rng=np.random.default_rng(20261120)
    for dim, col in columns.items():
        for group in metrics.loc[metrics.dimension==dim,'subgroup'].drop_duplicates():
            mask=(part[col]==group).values; n=int(mask.sum())
            idx=rng.integers(0,n,size=(2000,n))
            counts=np.zeros((2000,n),dtype=np.int32)
            for k in range(2000): counts[k]=np.bincount(idx[k],minlength=n)
            for target in metrics.outcome.unique():
                y=truth[target].values[mask]; results={}
                for model,frame in preds.items():
                    p=frame[f'{target}_calibrated'].values[mask]
                    results[model]={'auroc':weighted_auc(y,p,counts),'brier':counts@((y-p)**2)/n}
                for row in intervals[(intervals.dimension==dim)&(intervals.subgroup==group)&(intervals.outcome==target)].itertuples():
                    values=results[row.model][row.metric]
                    assert np.isfinite(values).sum()==row.valid_replicates
                    np.testing.assert_allclose(np.nanpercentile(values,[2.5,97.5]),[row.ci_lower_95,row.ci_upper_95],atol=1e-12)
                for row in diffs[(diffs.dimension==dim)&(diffs.subgroup==group)&(diffs.outcome==target)].itertuples():
                    values=results['bionutrigpt'][row.metric]-results[row.comparator][row.metric]
                    assert np.isfinite(values).sum()==row.valid_replicates
                    np.testing.assert_allclose(np.nanpercentile(values,[2.5,97.5]),[row.ci_lower_95,row.ci_upper_95],atol=1e-12)
            print(f'PASS: {dim} / {group}',flush=True)
    files=['subgroup_metrics.csv','subgroup_counts.csv','subgroup_bootstrap_intervals.csv','subgroup_paired_differences.csv']
    receipt={'verified':True,'calibration_rows':len(metrics),'bootstrap_intervals_reproduced':len(intervals),'paired_intervals_reproduced':len(diffs),'replicates':2000,'scope':'Unweighted, conditional on frozen models; no clinical fairness or survey-design claim','sha256':{f:hashlib.sha256((OUT/f).read_bytes()).hexdigest() for f in files}}
    (OUT/'extended_verification.json').write_text(json.dumps(receipt,indent=2))
    print('Extended verification passed.',flush=True)

if __name__=='__main__': main()
