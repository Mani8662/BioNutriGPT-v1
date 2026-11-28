"""Shared solver/prompt eligibility, canonical rules and frozen-input provenance."""
import copy
import hashlib
import importlib.metadata
import json
import math
import platform
from pathlib import Path
import sys


def eligible_foods(catalog, active_codes, mode):
    required=set(active_codes)
    return [f for f in catalog
            if not (mode=='pure_vegetarian' and f.dietary_type=='non_vegetarian')
            and not required.intersection(f.unknown_nutrients)
            and all(isinstance(f.nutrients_per_serving.get(c),(int,float))
                    and math.isfinite(f.nutrients_per_serving[c]) for c in required)]


def prompt_post_checks(dossier,catalog,s3):
    import numpy as np
    stage2=dossier['stage2_counterfactual_prescription']
    stage=s3._parse_stage2_input(stage2)
    basket=dossier['stage3_discrete_meal_synthesis']['solved_food_basket']
    quantities={r['food_id']:r['quantity'] for r in basket}
    out=s3._build_output_from_quantities(np.array([quantities.get(f.food_id,0) for f in catalog]),catalog,stage,{})
    check=out.clinical_verification
    return {'calories_within_pct_of_target':check['calories']['acceptance_threshold_pct'],
            'target_calories':stage.energy_prescription['target_calories'],
            **check['macronutrient_split']['screening_ranges_pct'],
            'sodium_to_potassium_ratio_max':stage.baselines['DR1TSODI']/(stage.baselines['DR1TPOTA']+1),
            'sodium_to_potassium_formula':'sodium_mg / (potassium_mg + 1)',
            'fiber_g_min':stage.lower_bounds['DR1TFIBE'],
            'fiber_g_per_1000_kcal_min':check['fiber_adequacy']['minimum_g_per_1000_kcal']}


def load_frozen_scenarios(root,catalog,s3,build_stage2):
    import numpy as np
    path=root/'Saved Model/stage4_lasi_v3_frozen_scenarios.json'
    manifest=json.loads(path.read_text(encoding='utf-8'))
    patients={p['lasi_prim_key']:p for p in json.loads((root/'Data/lasi_w1_indian_cohort_100.json').read_text(encoding='utf-8'))['patients']}
    scenarios=copy.deepcopy(manifest['scenarios']);dossiers={}
    for sc in scenarios:
        stage2=build_stage2(patients[sc['lasi_prim_key']]);mode=sc['mode']
        byid={f.food_id:f for f in catalog}
        if any(r['food_id'] not in byid or r.get('catalog_version')!=byid[r['food_id']].catalog_version for r in sc['initial_v1_basket']):
            raise ValueError('Frozen initial basket uses a different catalog.')
        q={r['food_id']:int(r['quantity']) for r in sc['initial_v1_basket']}
        out=s3._build_output_from_quantities(np.array([q.get(f.food_id,0) for f in catalog]),catalog,s3._parse_stage2_input(stage2),{})
        certificate=s3.validate_candidate(out,catalog,stage2,dietary_mode=mode)
        if not certificate['accepted']:raise ValueError('Frozen initial plan fails current validator; do not silently replace it.')
        out.accepted=True;st3=out.to_dict();st3['revision']={'dietary_mode':mode,'excluded_food_ids':[],'status':'initial'}
        sc['dossier_key']=sc['id'];dossiers[sc['id']]={'patient_metadata':stage2['patient_metadata'],
            'stage1_diagnostic_assessment':{'probabilities':{},'model_metadata':{'model_version':'LASI status-routing adapter; Stage 1 not run'}},
            'stage2_counterfactual_prescription':stage2,'stage3_discrete_meal_synthesis':st3,'stage3_meal_plan':st3}
    manifest['benchmark_version']='stage4-lasi-indian-conversational-v4'
    manifest['source_v3_frozen_manifest_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    manifest['stage2_input_snapshot']={key:d['stage2_counterfactual_prescription'] for key,d in dossiers.items()}
    return scenarios,dossiers,manifest


def snapshot(root,scenario_path):
    names=['benchmark_stage4_lasi_conversational_v4.py','stage4_benchmark_protocol.py','stage4_benchmark_analysis.py',
           'backend/chat_service.py','backend/plan_store.py','stage3_milp_solver.py','stage2_counterfactual_engine.py',
           'nutrient_rule_policy.py','benchmark_lasi_stage3_indian_cohort.py','infer_bionutrigpt_pipeline_Stage_1_to_4.py',
           'benchmark_stage4_conversational_revisions.py','food_composition_cooked_sourced.json',
           'Data/lasi_w1_indian_cohort_100.json',str(scenario_path.relative_to(root))]
    versions={}
    for package in ['torch','transformers','bitsandbytes','numpy','scipy']:
        try:versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:versions[package]=None
    return {'hashes':{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names},
            'python':sys.version,'platform':platform.platform(),'packages':versions,
            'generation_config':{'model':'Qwen/Qwen2.5-3B-Instruct','max_new_tokens':512,'do_sample':False,'repetition_penalty':1.03},
            'status':'Captured before execution; model revision and actual generation details are logged per LLM turn.'}
