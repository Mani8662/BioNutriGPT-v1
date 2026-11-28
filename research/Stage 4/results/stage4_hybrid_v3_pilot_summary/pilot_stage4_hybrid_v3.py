"""Hybrid chat smoke pilot. Synthetic plans; revision solver mocked, real Qwen optional."""
import argparse
import dataclasses
import json
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch
from verify_stage3_production import fixture
import stage3_milp_solver as s3
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'backend'))
import chat_service as core
import conversation_service as chat
import chat_interpreter as interpreter
import plan_store

SCENARIOS=[
    ('How much protein in Test rice?', 'nutrient', '20.00 g'),
    ('How much energy in one serving of Test rice?', 'nutrient', '200.00 kcal'),
    ('Do not remove Test rice', 'clarification', None),
    ('Replace Test rice with 2 servings of Test dal', 'revision', None),
    ('I am tired of Test rice; please leave it out.', 'clarification', 'Check this interpreted request'),
    ('Could you describe what makes Test rice a good choice?', 'clarification', None),
    ('I would rather have Test dal instead of Test rice, two servings please.', 'clarification', 'Check this interpreted request'),
    ('I do not want you to leave Test rice out.', 'clarification', None),
]


def main(run):
    if not run:
        print(json.dumps({'requests':SCENARIOS,'scope':'Real model only on unresolved requests; synthetic plan and mocked solver; not clinical or meal-feasibility evidence'},indent=2));return
    path=ROOT/'Saved Model/stage4_hybrid_v3_pilot.jsonl'
    if path.exists():raise RuntimeError('Output exists; refusing overwrite')
    foods,stage2,q=fixture()
    foods=[dataclasses.replace(f,name='Test rice' if i==0 else 'Test dal' if i==1 else f.name) for i,f in enumerate(foods)]
    plan=s3._build_output_from_quantities(q,foods,s3._parse_stage2_input(stage2),{}).to_dict();plan['accepted']=True
    dossier={'stage2_counterfactual_prescription':stage2,'stage3_discrete_meal_synthesis':plan}
    real_interpret=interpreter.interpret
    for message,expected_intent,expected_text in SCENARIOS:
        model_calls=[]
        def observed(*args,**kwargs):
            result=real_interpret(*args,**kwargs,diagnostics=True)
            record=dict(result)
            if 'request' in record:record['request']=record['request'].model_dump()
            model_calls.append(record);return result
        with tempfile.TemporaryDirectory() as temp, patch.object(core.ps.s3,'load_food_database',return_value=foods), \
             patch.object(core,'tool_propose_revision',return_value=(None,None,{'summary':'Mocked solver: pilot does not assess feasibility.'})) as solver, \
             patch.object(interpreter,'interpret',side_effect=observed):
            sid=plan_store.create_session(dossier,store_dir=Path(temp))['session_id']
            start=time.perf_counter()
            result=chat.answer(message,dossier,session_id=sid,store_dir=Path(temp),use_llm_interpretation=True)
            row=dict(message=message,response=result,expected_intent=expected_intent,expected_text=expected_text,
                elapsed_ms=(time.perf_counter()-start)*1000,model_calls=model_calls,solver_calls=solver.call_count,
                intent_matches=result['intent']==expected_intent,
                content_matches=expected_text is None or expected_text in result['answer'] or expected_text in result.get('clarification_options',[]))
            with path.open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',action='store_true');main(ap.parse_args().run)
