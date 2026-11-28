"""Current-code rule/metadata audit. No model training or clinical validation."""
import hashlib,itertools,json
from pathlib import Path
import numpy as np
import stage2_counterfactual_engine as s2
import nutrient_rule_policy as policy
ROOT=Path(__file__).resolve().parent
DEST=ROOT/'Saved Model/stage2_current_audit_v1'

def bounds(**changes):
    args=dict(artifacts={},t_kcal=2000.,weight_kg=70.,is_female=False,age_years=42.,p_diab_0=.1,p_obes_0=.1,p_cvd_0=.1)
    args.update(changes)
    return s2._build_nutrient_bounds(**args)

def main():
    DEST.mkdir(exist_ok=False)
    def write(name,obj): (DEST/name).write_text(json.dumps(obj,indent=2,default=str),encoding='utf-8')
    write('source_hashes.json',{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['stage2_counterfactual_engine.py','nutrient_rule_policy.py','audit_stage2_rules.py','audit_stage2_current_v1.py']})
    lower,upper=bounds()
    rules=[policy.reviewed_rule_metadata(code,float(lower[i]),float(upper[i])) for i,code in enumerate(s2.DIET_RAW_COLS)]
    write('current_rule_table.json',rules)
    grid=[]
    for age,female,weight,energy in itertools.product([18,19,30,31,50,51,70,71,90],[False,True],[25,40,70,120],[1200,2000,3000]):
        low,high=bounds(age_years=age,is_female=female,weight_kg=weight,t_kcal=energy)
        protein=float(low[s2.NUTRIENT_INDEX['DR1TPROT']])
        # Necessary consistency condition between envelope minimum and filter 2 maximum.
        conflict=protein>2*weight+1e-6
        grid.append(dict(age=age,female=female,weight=weight,energy=energy,finite_ordered=bool(np.isfinite(low).all() and np.isfinite(high).all() and (low<=high).all()),protein_min=protein,filter_protein_max=2*weight,protein_filter_conflict=conflict))
    write('boundary_grid.json',grid)
    jumps=[]
    for key,cutoff in [('p_diab_0',s2.CUTOFF_DIABETES),('p_obes_0',s2.CUTOFF_OBESITY),('p_cvd_0',s2.CUTOFF_CVD)]:
        lo,hi=bounds(**{key:cutoff-1e-6});lo2,hi2=bounds(**{key:cutoff+1e-6})
        changes=[dict(nutrient=code,before=[float(lo[i]),float(hi[i])],after=[float(lo2[i]),float(hi2[i])]) for i,code in enumerate(s2.DIET_RAW_COLS) if lo[i]!=lo2[i] or hi[i]!=hi2[i]]
        jumps.append(dict(probability=key,cutoff=cutoff,probability_difference=2e-6,changed_bounds=changes))
    # Exact production energy formula, held-fixed example inputs; not a generated prescription.
    bmr=10*70+6.25*170-5*42+5
    tdee=bmr*s2.PAL_FACTORS['sedentary']
    write('threshold_sensitivity.json',dict(bound_jumps_at_fixed_2000_kcal=jumps,energy_formula_example=dict(weight=70,height=170,age=42,female=False,activity='sedentary',below_obesity_cutoff=max(tdee,1200),above_obesity_cutoff=max(tdee-500,1200))))
    vector=np.zeros(len(s2.FEATURE_COLS))
    base=dict(weight_kg=70,height_cm=170,age_years=42,activity_level='sedentary',is_female=False)
    cases=[('missing_weight',{k:v for k,v in base.items() if k!='weight_kg'}),('minor_age',{**base,'age_years':12}),('zero_age',{**base,'age_years':0}),('negative_weight',{**base,'weight_kg':-1}),('nonfinite_height',{**base,'height_cm':float('nan')}),('unknown_activity',{**base,'activity_level':'unknown'}),('sex_inferred',{k:v for k,v in base.items() if k!='is_female'}),('declared_renal',{**base,'renal_disease':True}),('declared_pregnancy',{**base,'pregnant':True})]
    checks=[]
    for label,metadata in cases:
        try: result=s2._validate_patient_metadata(metadata,vector);checks.append(dict(case=label,accepted=True,result=result))
        except Exception as exc:checks.append(dict(case=label,accepted=False,error=str(exc)))
    write('metadata_checks.json',checks)
    overrides=[]
    for label,value in [('contradictory',(20,10)),('negative',(-5,10)),('nonfinite',(0,float('nan')))]:
        try:
            bounds(artifacts={'nutrient_bounds':{'DR1TVB12':value}})
            overrides.append(dict(case=label,accepted=True))
        except Exception as exc:overrides.append(dict(case=label,accepted=False,error=str(exc)))
    write('override_checks.json',overrides)
    old=json.loads((ROOT/'Saved Model/stage2_rule_audit.json').read_text())
    summary=dict(scope='Current function-level computational audit; no model inference, counterfactual search, clinical source revalidation or food feasibility claims.',ruleset=policy.RULESET_VERSION,historical_report_ruleset=old['ruleset_version'],rule_count=len(rules),grid_cases=len(grid),finite_ordered_cases=sum(x['finite_ordered'] for x in grid),protein_filter_conflicts=sum(x['protein_filter_conflict'] for x in grid),metadata_checks=checks,override_checks=overrides,production_checkpoint=str(s2.DEFAULT_CHECKPOINT))
    write('summary.json',summary)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
