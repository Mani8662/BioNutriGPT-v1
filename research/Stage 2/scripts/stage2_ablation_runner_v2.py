"""Versioned Stage 2 ablation execution; dry run by default, full study explicit."""
import argparse, copy, hashlib, json, math, os, platform, shutil, time
from pathlib import Path
import numpy as np
import torch
from scipy.stats import binomtest
import stage1_deployment as deployment
import stage2_counterfactual_engine as s2
import stage2_ablation_engine_v1 as engine
import verify_stage2_ablation_v1 as oldverify

ROOT=Path(__file__).resolve().parent
DIR=ROOT/'Saved Model/stage2_ablation_v2'
OLD=ROOT/'Saved Model/stage2_ablation_v1'
class ValidationError(ValueError):pass
def need(ok,msg):
    if not ok:raise ValidationError(msg)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8');q.replace(p)
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,allow_nan=False).encode()).hexdigest()
def check_manifest():
    for name,h in read(DIR/'input_manifest.json').items():need((ROOT/name).exists() and sha(ROOT/name)==h,'Bound file changed: '+name)
def prepare():
    need(not (DIR/'input_manifest.json').exists(),'Frozen v2 exists; use a new version for changes')
    prior=read(OLD/'input_manifest.json')['files']
    for name,h in prior.items():need(sha(ROOT/name)==h,'Original input changed: '+name)
    DIR.mkdir(exist_ok=True)
    for name in ['development_participants_LOCAL.json','evaluation_participants_LOCAL.json','variant_manifest.json','component_map.md']:
        shutil.copy2(OLD/name,DIR/name)
    dev=read(DIR/'development_participants_LOCAL.json');ev=read(DIR/'evaluation_participants_LOCAL.json')
    need(len(dev)==3 and len(ev)==100,'Wrong sample size')
    need(len({x['SEQN'] for x in ev})==100 and not({x['SEQN'] for x in dev}&{x['SEQN'] for x in ev}),'Cohort identity overlap')
    p=read(OLD/'protocol.json');p['protocol_version']='stage2_ablation_v2'
    p['repair']={'status':'prepared_full_unrun','design':'Same six variants and frozen participants; execution/verification repair after pilot inspection.',
      'order':'Rotate sorted six variants by participant position modulo six; counts per position differ at most one.',
      'timing':'Sequential CUDA-synchronized wrapper time excludes model loading, untimed forward warmup and verification; CPU threads=min(12,logical CPUs).',
      'statistics':'Primary overall-acceptance exact tests: Holm family 5. Secondary numerical/score-gate exact tests: Holm family 10. Continuous paired bootstrap intervals exploratory, unadjusted; missing execution outputs halt the study, not zero-imputed.',
      'limitations':'Filters 1-3 ablated jointly; does not establish each individual filter effect. No universal redundancy claim. Shared preprocessing and numerical validator, separate statistics formulas.',
      'device_required':'cuda for planned run; CPU fallback requires explicit protocol revision'}
    p.pop('reproducibility',None);write(DIR/'protocol.json',p)
    paths=set(ROOT/k for k in prior)
    paths.update([ROOT/'stage2_ablation_engine_v1.py',ROOT/'verify_stage2_ablation_v1.py',Path(__file__),ROOT/'verify_stage2_ablation_v2.py',ROOT/'test_stage2_ablation_v2.py'])
    # Bind local statically imported modules recursively as well as model assets.
    import ast
    pending=list(paths);seen=set()
    while pending:
        path=pending.pop()
        if path in seen or path.suffix!='.py':continue
        seen.add(path)
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            names=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) and node.module else []
            for name in names:
                local=ROOT/(name.replace('.','/')+'.py')
                if local.exists() and local not in paths:paths.add(local);pending.append(local)
    paths.update(DIR/n for n in ['protocol.json','variant_manifest.json','development_participants_LOCAL.json','evaluation_participants_LOCAL.json'])
    write(DIR/'input_manifest.json',{str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)})
def load(pilot):
    return read(DIR/('development_participants_LOCAL.json' if pilot else 'evaluation_participants_LOCAL.json')),read(DIR/'variant_manifest.json')
def model_context():
    need(torch.cuda.is_available(),'CUDA unavailable; no silent timing-protocol change')
    torch.set_num_threads(min(12,os.cpu_count() or 1));torch.manual_seed(20261124)
    dev=torch.device('cuda');art=s2._load_preprocessing_artifacts(deployment.PREPROCESSOR);model=s2._load_stage1_model(deployment.CHECKPOINT,art,dev)
    return model,art,dev
def binding(case,config):return digest({'case':case,'config':config,'manifest':sha(DIR/'input_manifest.json')})
def checkpoint(folder,case,variant):return folder/f"case_{case['case']:03d}_{case['SEQN']}_{variant}.json"
def validate(record,case,variant,config,context):
    need(record.get('case')==case['case'] and record.get('SEQN')==case['SEQN'] and record.get('variant_id')==variant,'Identity mismatch')
    need(record.get('variant_config')==config,'Variant switch mismatch')
    need(record.get('binding')==binding(case,config),'Binding mismatch')
    need(record.get('execution_status')=='completed','Incomplete/exception record requires review; no retry')
    model,art,device=context
    try:oldverify.audit_checkpoint_record(record,{case['SEQN']:case},model,art,device)
    except oldverify.ExpectedValidationError as exc:raise ValidationError(str(exc)) from exc
    raw,_=s2.canonicalize_dietary_ratios(case['vector']);n=len(s2.BIO_COLS);k=len(s2.DIET_RAW_COLS)
    obs=np.asarray(raw[n:n+k]);v=record['vectors'];target=np.asarray(v['final_target'],float)
    need(target.shape==obs.shape and np.isfinite(target).all(),'Invalid target')
    need(np.allclose(v['observed_nutrients'],obs,rtol=0,atol=1e-6),'Observed intake changed')
    scales=np.array([s2.DAILY_REFERENCE_SCALES[x] for x in s2.DIET_RAW_COLS]);diff=(target-obs)/scales
    for key,val in [('distance_from_observed_l1_norm',np.abs(diff).sum()),('distance_from_observed_l2_norm',np.linalg.norm(diff))]:
        need(math.isclose(record['evaluation'][key],float(val),rel_tol=0,abs_tol=1e-6),'Distance mismatch')
    for disease,p in record['evaluation']['target_probabilities'].items():
        need(abs(record['evaluation']['score_deltas'][disease]-(p-record['baseline_probabilities'][disease]))<1e-6,'Score delta mismatch')
    need(math.isfinite(record['wall_ms']) and record['wall_ms']>=0,'Invalid timing')
    if not config['include_joint_projection']:need(np.array_equal(v['pre_projection_adam_clipped'],v['post_projection']),'Projection bypass mismatch')
    if not config['include_sequential_filters']:need(np.array_equal(v['post_projection'],v['final_target']),'Filter bypass mismatch')
def execute(folder,case,variant,config,context,order):
    path=checkpoint(folder,case,variant)
    if path.exists():
        rec=read(path);validate(rec,case,variant,config,context);return rec
    need(not path.with_suffix('.quarantine').exists(),'Quarantined result requires review')
    model,art,dev=context;torch.cuda.synchronize();start=time.perf_counter()
    try:rec=engine.run_ablation_variant(case=case,variant_config=config,model=model,artifacts=art,device=dev)
    except Exception as exc:
        write(path,{'case':case['case'],'SEQN':case['SEQN'],'variant_id':variant,'variant_config':config,'binding':binding(case,config),'execution_status':'exception','error':repr(exc)})
        raise
    torch.cuda.synchronize();rec.update(wall_ms=(time.perf_counter()-start)*1000,binding=binding(case,config),execution_status='completed',device=str(dev),variant_order=order)
    try:validate(rec,case,variant,config,context)
    except Exception:write(path.with_suffix('.quarantine'),rec);raise
    write(path,rec);return rec
def boot(values):
    x=np.asarray(values,float);rng=np.random.default_rng(20261124);means=x[rng.integers(len(x),size=(10000,len(x)))].mean(axis=1)
    return {'n':len(x),'mean':float(x.mean()),'ci95':np.percentile(means,[2.5,97.5]).tolist()}
def analyze(records,cases,variants):
    idx={(x['SEQN'],x['variant_id']):x for x in records}
    need(len(idx)==len(records)==len(cases)*len(variants),'Missing/duplicate records')
    need(set(idx)=={(c['SEQN'],v) for c in cases for v in variants},'Wrong cohort/variant membership')
    result={};families={'primary':[],'secondary':[]}
    for variant in sorted(variants):
        if variant=='variant_0_reference':continue
        pairs=[(idx[c['SEQN'],'variant_0_reference'],idx[c['SEQN'],variant]) for c in cases]
        endpoints={}
        for ep in ['overall_accepted','numerical_pass','score_gate_pass']:
            bits=[(a['evaluation'][ep],b['evaluation'][ep]) for a,b in pairs];w=sum(b and not a for a,b in bits);l=sum(a and not b for a,b in bits)
            endpoints[ep]={**boot([100*(int(b)-int(a)) for a,b in bits]),'wins':w,'losses':l,'p':float(binomtest(w,w+l).pvalue) if w+l else 1.}
            families['primary' if ep=='overall_accepted' else 'secondary'].append(endpoints[ep])
        for ep in ['distance_from_observed_l1_norm','distance_from_observed_l2_norm']:
            endpoints[ep]=boot([b['evaluation'][ep]-a['evaluation'][ep] for a,b in pairs])
        endpoints['wall_ms']=boot([b['wall_ms']-a['wall_ms'] for a,b in pairs])
        for disease in ['diabetes','obesity','cvd']:
            endpoints['delta_'+disease]=boot([b['evaluation']['score_deltas'][disease]-a['evaluation']['score_deltas'][disease] for a,b in pairs])
        result[variant]=endpoints
    for family in families.values():
        maximum=0.
        for i,e in enumerate(sorted(family,key=lambda x:x['p'])):maximum=max(maximum,min(1.,e['p']*(len(family)-i)));e['holm_p']=maximum
    return result
def verify(pilot,context=None):
    check_manifest();cases,variants=load(pilot);folder=DIR/('pilot' if pilot else 'full')/'checkpoints'
    expected={checkpoint(folder,c,v) for c in cases for v in variants}
    need(set(folder.glob('*.json'))==expected,'Missing/extra checkpoint files')
    context=context or model_context();records=[]
    for c in cases:
        for v,config in variants.items():
            rec=read(checkpoint(folder,c,v));validate(rec,c,v,config,context);records.append(rec)
    stats=analyze(records,cases,variants);p=folder.parent/'analysis.json'
    if p.exists():need(read(p)==stats,'Saved analysis differs from reconstructed statistics')
    else:write(p,stats)
    check_manifest();write(folder.parent/'verification.json',{'status':'passed','records':len(records),'participants':len(cases),'full_cohort':not pilot,'checkpoint_hashes':{p.name:sha(p) for p in sorted(expected)},'analysis_sha256':sha(p),'scope':'Model replay and distance/gate reconstruction; shared deployed preprocessing and numerical checks.'})
    return records,stats
def main():
    parser=argparse.ArgumentParser();g=parser.add_mutually_exclusive_group();g.add_argument('--prepare',action='store_true');g.add_argument('--run-pilot',action='store_true');g.add_argument('--run-full',action='store_true');args=parser.parse_args()
    if args.prepare:prepare();return
    check_manifest();pilot=args.run_pilot;cases,variants=load(pilot)
    if not(args.run_pilot or args.run_full):print(f'Dry run: {len(cases)} participants, {len(cases)*len(variants)} tasks; no execution');return
    folder=DIR/('pilot' if pilot else 'full')/'checkpoints';folder.mkdir(parents=True,exist_ok=True)
    context=model_context();model,art,dev=context
    raw,_=s2.canonicalize_dietary_ratios(cases[0]['vector'])
    with torch.no_grad():s2._predict_probabilities(model,s2._normalize_torch(torch.tensor(raw,dtype=torch.float32,device=dev).unsqueeze(0),art),no_grad=True)
    torch.cuda.synchronize();start=time.perf_counter();order0=sorted(variants)
    for i,c in enumerate(cases):
        shift=i%len(order0);order=order0[shift:]+order0[:shift]
        for v in order:execute(folder,c,v,variants[v],context,order)
        print(f'{i+1}/{len(cases)} participants completed',flush=True)
    elapsed=time.perf_counter()-start;verify(pilot,context)
    write(folder.parent/'run_receipt.json',{'status':'passed','session_wall_seconds_including_checkpoint_checks':elapsed,'participants':len(cases),'tasks':len(cases)*len(variants),'device':'cuda','cpu_threads':torch.get_num_threads(),'platform':platform.platform(),'torch':torch.__version__,'timing_note':'Session duration includes resumes; individual wall_ms is original measured execution. No model loading included.'})
    print('Execution and verification passed')
if __name__=='__main__':main()
