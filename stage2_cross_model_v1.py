"""Prepare/run frozen-model rescoring of existing Stage 2 proposals. No optimization."""
import argparse,ast,copy,gc,hashlib,json,os
from pathlib import Path
import joblib,numpy as np,pandas as pd,torch
from threadpoolctl import threadpool_limits
import run_stage1_comparison_v2 as comp
import stage2_counterfactual_engine as s2
import stage1_deployment as deployment

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'Saved Model/stage2_cross_model_v1'
SOURCE=ROOT/'Saved Model/stage2_ablation_v2'
MODELS=['bionutrigpt','catboost','ft_transformer']
DISEASES=['diabetes','obesity','cvd']
class ValidationError(ValueError):pass
def need(ok,msg):
    if not ok:raise ValidationError(msg)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8');q.replace(p)
def boot(x):
    x=np.asarray(x,float)
    if not len(x):return {'n':0,'mean':None,'ci95':None}
    rng=np.random.default_rng(20261124);means=x[rng.integers(len(x),size=(10000,len(x)))].mean(axis=1)
    return {'n':len(x),'mean':float(x.mean()),'ci95':np.percentile(means,[2.5,97.5]).tolist()}
def prepare():
    need(not (OUT/'input_manifest.json').exists(),'Already frozen; do not overwrite')
    paths=set()
    for name,h in read(SOURCE/'input_manifest.json').items():
        p=ROOT/name;need(sha(p)==h,'Changed ablation dependency: '+name);paths.add(p)
    receipt=read(SOURCE/'full/verification.json');need(receipt['records']==600 and receipt['full_cohort'],'Full ablation not verified')
    for name,h in receipt['checkpoint_hashes'].items():
        p=SOURCE/'full/checkpoints'/name;need(sha(p)==h,'Changed proposal: '+name);paths.add(p)
    paths.update([SOURCE/'full/verification.json',SOURCE/'full/independent_statistics_verification.json',Path(__file__),ROOT/'verify_stage2_cross_model_v1.py'])
    selected={}
    for name in MODELS:
        folder=comp.RESULTS/name;comp.verify_completion(folder,'completed.json')
        trial=read(folder/'selection.json')['selected_trial'];selected[name]=trial
        trialdir=folder/f'trial_{trial:02d}';comp.verify_completion(trialdir,'completed.json')
        if name=='bionutrigpt':need(sha(trialdir/'model.pt')==sha(deployment.CHECKPOINT),'Reference checkpoint differs from deployed source model')
        paths.update([folder/'selection.json',folder/'calibrators.joblib',folder/'temporal_predictions_LOCAL.csv',folder/'completed.json',trialdir/'completed.json',trialdir/('model.pt' if name in comp.NEURAL else 'model.joblib')])
    paths.add(comp.PROTOCOL_PATH)
    # Bind statically imported local code. No model inference in preparation.
    pending=list(paths);seen=set()
    while pending:
        p=pending.pop()
        if p.suffix!='.py' or p in seen:continue
        seen.add(p)
        for node in ast.walk(ast.parse(p.read_text(encoding='utf-8-sig'))):
            names=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) and node.module else []
            for name in names:
                q=ROOT/(name.replace('.','/')+'.py')
                if q.exists() and q not in paths:paths.add(q);pending.append(q)
    OUT.mkdir(exist_ok=True)
    protocol={'status':'prepared_no_inference','date':'2026-11-24','models':MODELS,'selected_trials':selected,'participants':100,'variants':6,'unique_rows_per_model':700,
      'primary':'Reference-variant nutrient changes rescored on all 100 participants. Assess per-outcome calibrated score changes and all-outcome nonincrease gate (delta<=0.0015).',
      'secondary':'All six variants; all outputs retained including numerically invalid proposals, with separate numerical-pass and source-accepted strata. Compare full versus no-model-objective within each frozen scorer.',
      'statistics':'Paired participant bootstrap seed 20261124, 10000 draws. Exploratory pointwise intervals, no p-values or clinical claims. Positive delta means score increase. Gates are engineering diagnostics, not clinical thresholds.',
      'controls':'Reproduce historical baseline predictions; reproduce source-model candidate predictions; exact zero-change control; repeat selected inference in independent verifier.',
      'limitations':'Independent architectures, not independent training data or clinical validation. Off-manifold changes possible. Obesity/BMI linkage may confound score interpretation. No outcomes under dietary intervention observed.',
      'execution':'FT-Transformer and reference neural inference CUDA; CatBoost CPU with available CPU parallelism. No fitting, recalibration, target regeneration or counterfactual optimization.'}
    write(OUT/'protocol.json',protocol);paths.add(OUT/'protocol.json')
    write(OUT/'input_manifest.json',{str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)})
def check():
    for name,h in read(OUT/'input_manifest.json').items():need(sha(ROOT/name)==h,'Bound input changed: '+name)
def build_inputs():
    cases=read(SOURCE/'evaluation_participants_LOCAL.json');variants=read(SOURCE/'variant_manifest.json')
    need(len(cases)==100 and len({c['SEQN'] for c in cases})==100,'Cohort mismatch')
    art=s2._load_preprocessing_artifacts(deployment.PREPROCESSOR)
    rows=[];matrix=[];records=[]
    for c in cases:
        raw,_=s2.canonicalize_dietary_ratios(c['vector']);rawt=torch.tensor(raw,dtype=torch.float32)
        matrix.append(s2._normalize_torch(rawt.unsqueeze(0),art).detach().numpy()[0]);rows.append({'SEQN':c['SEQN'],'variant':'baseline'})
    for variant in sorted(variants):
        for c in cases:
            p=SOURCE/'full/checkpoints'/f"case_{c['case']:03d}_{c['SEQN']}_{variant}.json";record=read(p)
            need(record['SEQN']==c['SEQN'] and record['variant_id']==variant,'Proposal identity mismatch')
            raw,_=s2.canonicalize_dietary_ratios(c['vector']);rawt=torch.tensor(raw,dtype=torch.float32);target=torch.tensor(record['vectors']['final_target'],dtype=torch.float32)
            candidate=s2._construct_candidate_88(rawt,target)
            end=len(s2.BIO_COLS)+len(s2.DIET_RAW_COLS)+len(s2.DIET_RATIO_COLS)
            need(torch.equal(candidate[0,:len(s2.BIO_COLS)],rawt[:len(s2.BIO_COLS)]) and torch.equal(candidate[0,end:],rawt[end:]),'Immutable features changed')
            matrix.append(s2._normalize_torch(candidate,art).detach().numpy()[0]);rows.append({'SEQN':c['SEQN'],'variant':variant,'numerical_pass':record['evaluation']['numerical_pass'],'source_accepted':record['evaluation']['overall_accepted']});records.append(record)
    x=np.asarray(matrix,dtype=np.float32);need(x.shape==(700,len(s2.FEATURE_COLS)) and np.isfinite(x).all(),'Bad input matrix')
    return rows,x,records
def load_model(name):
    folder=comp.RESULTS/name;selected=read(OUT/'protocol.json')['selected_trials'][name];cfg=read(comp.PROTOCOL_PATH)['models'][name][selected]
    device='cuda' if name in comp.NEURAL else 'cpu'
    if device=='cuda':need(torch.cuda.is_available(),'CUDA required; no silent fallback')
    model=comp.load_trial(name,cfg,folder/f'trial_{selected:02d}',device)
    return model,joblib.load(folder/'calibrators.joblib'),device
def score(name,model,cal,x,device):
    with threadpool_limits(limits=os.cpu_count() or 1):p=comp.predict(name,model,x,device)
    return p,comp.apply_calibrators(cal,p)
def summarize(rows,probs):
    variants=sorted({x['variant'] for x in rows}-{'baseline'});result={}
    deltas={name:p[100:]-np.tile(p[:100],(6,1)) for name,p in probs.items()}
    for name,d in deltas.items():
        result[name]={}
        for i,variant in enumerate(variants):
            a=d[i*100:(i+1)*100];meta=rows[100+i*100:200+i*100];strata={}
            for label,mask in [('all',np.ones(100,bool)),('numerically_valid',np.array([x['numerical_pass'] for x in meta])),('source_accepted',np.array([x['source_accepted'] for x in meta]))]:
                sub=a[mask];strata[label]={'n':int(mask.sum()),'gate_pass':int(np.all(sub<=.0015,axis=1).sum()),'outcomes':{disease:{**boot(sub[:,j]),'decrease_count':int((sub[:,j]<-1e-6).sum()),'increase_count':int((sub[:,j]>1e-6).sum())} for j,disease in enumerate(DISEASES)}}
            result[name][variant]=strata
        ref=variants.index('variant_0_reference');unguided=variants.index('variant_1_no_model_objective')
        diff=d[ref*100:(ref+1)*100]-d[unguided*100:(unguided+1)*100]
        result[name]['full_minus_no_model_objective']={disease:boot(diff[:,j]) for j,disease in enumerate(DISEASES)}
    return result
def run():
    check();need(not (OUT/'completed.json').exists(),'Already completed; use verifier, not overwrite')
    torch.set_num_threads(min(12,os.cpu_count() or 1));rows,x,records=build_inputs();write(OUT/'rows_LOCAL.json',rows)
    np.save(OUT/'normalized_inputs_LOCAL.npy',x);probs={};controls={}
    for name in MODELS:
        model,cal,device=load_model(name);raw,p=score(name,model,cal,x,device)
        saved=pd.read_csv(comp.RESULTS/name/'temporal_predictions_LOCAL.csv').set_index('SEQN').loc[[r['SEQN'] for r in rows[:100]]]
        baseline=saved[[k+'_calibrated' for k in comp.TARGETS]].to_numpy()
        discrepancy=float(np.max(np.abs(p[:100]-baseline)));need(discrepancy<=1e-5,'Historical baseline mismatch for '+name+': '+str(discrepancy))
        zero=score(name,model,cal,x[:5].copy(),device)[1];need(np.allclose(zero,p[:5],rtol=0,atol=1e-6),'Repeated zero-change control failed')
        if name=='bionutrigpt':
            source=np.array([[rec['evaluation']['target_probabilities'][k] for k in DISEASES] for rec in records]);gap=float(np.max(np.abs(source-p[100:])))
            need(gap<=1e-5,'Source-model candidate parity failed: '+str(gap))
        controls[name]={'baseline_max_abs_error':discrepancy,'zero_change_max_abs_error':float(np.max(abs(zero-p[:5]))),'device':device}
        if name=='bionutrigpt':controls[name]['source_candidate_max_abs_error']=gap
        np.savez_compressed(OUT/(name+'_predictions_LOCAL.npz'),raw=raw,calibrated=p);probs[name]=p
        print(name,'scored and parity checked',flush=True);del model;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
    write(OUT/'analysis.json',summarize(rows,probs));write(OUT/'controls.json',controls);check()
    write(OUT/'completed.json',{'status':'executed_pending_independent_verification','participants':100,'proposals':600,'models':3,'training_calls':0,'optimization_calls':0,'output_hashes':{p.name:sha(p) for p in OUT.iterdir() if p.suffix in ['.json','.npy','.npz'] and p.name!='completed.json'}})
def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group();g.add_argument('--prepare',action='store_true');g.add_argument('--run',action='store_true');a=p.parse_args()
    if a.prepare:prepare()
    elif a.run:run()
    else:print('Prepared code only. Gemini: --prepare, then --run, then independent verifier.')
if __name__=='__main__':main()
