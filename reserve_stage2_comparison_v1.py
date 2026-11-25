"""Lock a new Stage 2 evaluation sample; no model scoring or optimization."""
from pathlib import Path
from dataclasses import fields
import json,hashlib,shutil
import pandas as pd
import joblib
from prepare_stage1_repair_v2 import load

R=Path(__file__).resolve().parent
O=R/'Saved Model/stage2_comparison_design_v1'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    if O.exists():raise RuntimeError('Reserved design already exists; do not overwrite.')
    pp=load('comparison_design_preprocessing','Data/preprocess_for_training.py')
    data=R/'Data/stage1_repaired_v2/temporal_clean.csv';clean=pd.read_csv(data)
    rawpath=R/'Data/NHANES_2017_2020_PrePandemic_Raw.csv'
    raw=pd.read_csv(rawpath,usecols=['SEQN','BMXWT','BMXHT','RIDAGEYR','RIAGENDR']).set_index('SEQN')
    assert raw.index.is_unique and clean.SEQN.is_unique
    inputs=[c for c in clean if c not in set(pp.META_COLS+pp.TARGET_COLS) and c not in pp.MODEL_EXCLUDED_RAW_FEATURES]
    flags=pp.compute_missingness_indicators(clean,inputs)
    joined=clean[['SEQN']].join(raw,on='SEQN',validate='one_to_one')
    eligible=(flags.sum(axis=1)==0)&joined.BMXWT.between(30,180)&joined.BMXHT.between(120,210)&(joined.RIDAGEYR>=20)&joined.RIAGENDR.isin([1,2])
    repairpath=R/'Saved Model/stage2_constraint_audit_repaired_v1/cases_LOCAL.json'
    repair_ids={c['SEQN'] for c in json.loads(repairpath.read_text())}
    previouspath=R/'Saved Model/stage1_downstream_study_v1/selected_participants_LOCAL.csv'
    previous=pd.read_csv(previouspath);prior_ids=set(previous.SEQN.astype(int))
    pilotpath=R/'Saved Model/stage2_research_adapter_pilot_v1/envelopes_LOCAL.json'
    pilot_ids={int(c['SEQN']) for c in json.loads(pilotpath.read_text())}
    excluded=repair_ids|prior_ids|pilot_ids
    pool=clean[eligible & ~clean.SEQN.isin(excluded)].sort_values('SEQN')
    chosen=pool.sample(n=100,random_state=20261123).sort_values('SEQN')
    ids=chosen.SEQN.astype(int).tolist();assert not set(ids)&excluded
    sourcepaths=[data,rawpath,repairpath,previouspath,pilotpath,Path(__file__),R/'stage2_counterfactual_engine.py',R/'stage1_deployment.py',R/'nutrient_rule_policy.py',R/'Saved Model/deployment/repaired_stage1_v1/manifest.json']
    protocol={'version':'stage2_comparison_design_v1','status':'sample and scope locked; comparison not executed',
        'n':100,'selection_seed':20261123,'eligible_complete_records':int(eligible.sum()),'eligible_after_exclusions':len(pool),
        'exclusions':{'repair_profiles':len(repair_ids),'prior_downstream_study':len(prior_ids),'prior_adapter_pilot':len(pilot_ids),'unique_excluded':len(excluded)},
        'selection':'Random sample without Stage 2 outcome inspection; disjoint from named prior studies, not globally untouched NHANES data',
        'activity':'sedentary fixed engineering assumption in every arm','model':'repaired_v1 BioNutriGPT calibrated','routing_thresholds':dict(diabetes=.5,obesity=.5,cvd=.5),
        'comparison_scope':'Common per-participant envelope, calories, routing, ratio limits, acceptance gates and solver settings; prediction routing value itself not the question',
        'planned_arms':['deterministic rules-only nutrient targets','minimum-change joint constrained projection without model-score objective','50-step model-guided search plus the identical joint projection'],
        'primary_control':'Full model-guided versus minimum-change; isolates model-score objective',
        'required_before_execution':['Implement and freeze exact rules-only target construction without meaningless midpoints of million-unit search caps','Test runners on development cases only','Freeze common downstream optimizer objective and all solver limits','Version the executable protocol and analysis code before exposing the 100 reserved profiles to outcomes'],
        'required_outputs':['joint numerical feasibility','score-gate acceptance separately','normalized dietary-change magnitude with declared fixed scales','solver convergence versus feasible fallback','runtime and failures','paired Stage 3 acceptance if downstream run is included'],
        'uncertainty':'Paired participant bootstrap; count all failures; no model or weight tuning on evaluation results',
        'limits':'Not external Stage 1 validation, a clinical trial, or a journal-tier determination. The 32 repaired cases are development evidence.',
        'hashes':{str(p.relative_to(R)):sha(p) for p in sourcepaths}}
    O.mkdir(parents=True)
    (O/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    (O/'selected_participants_LOCAL.json').write_text(json.dumps(ids,indent=2))
    freeze=O/'frozen_sources';freeze.mkdir()
    for name in ['stage2_counterfactual_engine.py','stage1_deployment.py','nutrient_rule_policy.py']:shutil.copy2(R/name,freeze/name)
    (O/'verification.json').write_text(json.dumps({'passed':True,'n':100,'unique_ids':len(set(ids)),'overlap_repair':len(set(ids)&repair_ids),'overlap_previous_study':len(set(ids)&prior_ids),'overlap_previous_pilot':len(set(ids)&pilot_ids),'selected_ids_sha256':sha(O/'selected_participants_LOCAL.json'),'no_comparison_run':True},indent=2))
    print(json.dumps({k:protocol[k] for k in ['status','n','eligible_after_exclusions','exclusions']},indent=2))

if __name__=='__main__':main()
