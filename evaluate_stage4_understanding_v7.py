"""Real-catalog dialogue engineering evaluation, with real Qwen on unresolved turns."""
import copy, hashlib, json, sys, tempfile, time
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'backend'))
import conversation_service as chat
import chat_service as core
import chat_interpreter as interpreter
import plan_store
import food_groups
import stage3_milp_solver as s3
from benchmark_stage4_conversational_revisions import _validate_raw_basket_dict


def main():
    dest=ROOT/'Saved Model/stage4_understanding_v7'
    dest.mkdir(exist_ok=False)
    def write(name,obj): (dest/name).write_text(json.dumps(obj,indent=2,default=str),encoding='utf-8')
    stage2=json.loads((ROOT/'Saved Model/stage3_live_sample_stage2.json').read_text())
    catalog=s3.load_food_database()
    write('binding.json',{str(p):hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in [
        'backend/conversation_service.py','backend/chat_interpreter.py','backend/chat_service.py','backend/compound_requests.py',
        'backend/plan_store.py','food_group_preferences.json','food_composition_cooked_sourced.json',
        'Saved Model/stage3_live_sample_stage2.json','evaluate_stage4_understanding_v7.py']})
    initial=s3.solve_meal_plan(stage2,catalog,time_limit_sec=20,revision_spec={'dietary_mode':'veg_nonveg_combo'})
    write('initial.json',initial.to_dict())
    if not initial.accepted: raise RuntimeError('Initial plan rejected')
    plan=initial.to_dict(); plan['revision']={'dietary_mode':'veg_nonveg_combo','status':'initial'}
    dossier={'stage2_counterfactual_prescription':stage2,'stage3_discrete_meal_synthesis':plan}
    fish=set(food_groups.excluded_ids(['fish'],catalog))
    selected=sorted([x for x in plan['solved_food_basket'] if x['food_id'] not in fish],key=lambda x:x['food_id'])[:2]
    a,b=selected; name=a['name']; qty=a['quantity']
    cases=[
      ('description_local',['Please tell me about the first dish.',name],False),
      ('negated_omission',[f'Please never omit {name}.','Check this interpreted request'],False),
      ('description_model',[f'Give me a brief factual overview of {name}.'],False),
      ('information_model',[f'What details can you share regarding {name}?'],False),
      ('natural_removal',[f'Kindly omit {name} from my daily basket.','Check this interpreted request'],True),
    ]
    write('protocol.json',{'scope':'One saved Stage 2 case; engineering evaluation, not held-out/general accuracy or clinical evidence.',
      'selection':'First two non-fish initial basket items by food_id; retain quantities. No retries or alternate food selection.',
      'cases':cases,'initial_solver_seconds':20,'revision_solver_seconds':2,
      'scoring':'Report model interpretation, completed revisions, independent validity and unchanged-before-acceptance separately. Clarification does not count as completed revision.'})
    real_interpret=interpreter.interpret
    rows=[]
    for label,messages,allows_revision in cases:
      with tempfile.TemporaryDirectory() as folder:
        store=Path(folder); sid=plan_store.create_session(dossier,store_dir=store)['session_id']
        for question in messages:
          row={'case':label,'question':question,'model_calls':[]}
          def observe(*args,**kwargs):
            result=real_interpret(*args,**kwargs,diagnostics=True)
            logged=dict(result)
            if 'request' in logged:logged['request']=logged['request'].model_dump()
            row['model_calls'].append(logged)
            return result
          before=plan_store.get_session(sid,store_dir=store)
          start=time.perf_counter()
          try:
            with patch.object(interpreter,'interpret',side_effect=observe):
              response=chat.answer(question,dossier,session_id=sid,store_dir=store,use_llm_interpretation=True)
            row['response']=response
            after=plan_store.get_session(sid,store_dir=store)
            def basket(record):return plan_store.get_current_version_entry(record)['stage3_plan']['solved_food_basket']
            row['unchanged_before_acceptance']=basket(before)==basket(after) and before['current_version']==after['current_version']
            assert row['unchanged_before_acceptance']
            proposal=response.get('proposed_meal_plan')
            row['proposal_returned']=bool(proposal)
            if proposal:
              assert allows_revision,'Unexpected revision proposal'
              rev=proposal['revision']
              valid=_validate_raw_basket_dict(proposal['solved_food_basket'],catalog,stage2,excluded_ids=rev['excluded_food_ids'],locked_quantities=rev['locked_food_quantities'],dietary_mode=rev['dietary_mode'])
              row['independent_validation']=valid
              assert valid['accepted']
              if label in ('group_exclusion','compound'):assert not fish.intersection(x['food_id'] for x in proposal['solved_food_basket'])
              requirements={a['food_id']:'Dinner'} if label=='compound' else {a['food_id']:'Lunch',b['food_id']:'Dinner'} if label=='two_placements' else {}
              if label=='natural_removal':assert a['food_id'] not in [x['food_id'] for x in proposal['solved_food_basket']]
              schedule=core.build_requested_schedule(proposal['solved_food_basket'],rev.get('meal_requirements',{}))
              assert core._meal_requirements_met(schedule,requirements)
              tampered=copy.deepcopy(proposal);tampered['solved_food_basket'][0]['quantity']+=1
              try:core.accept(dossier,tampered,session_id=sid,store_dir=store);row['tamper_rejected']=False
              except ValueError:row['tamper_rejected']=True
              assert row['tamper_rejected']
              core.accept(dossier,proposal,session_id=sid,store_dir=store)
              assert basket(plan_store.get_session(sid,store_dir=store))==proposal['solved_food_basket']
              row['acceptance_verified']=True
              core.rollback(sid,before['current_version'],store_dir=store)
              row['rollback_verified']=basket(plan_store.get_session(sid,store_dir=store))==basket(before)
              assert row['rollback_verified']
          except Exception as exc:row['error']=repr(exc)
          row['elapsed_seconds']=time.perf_counter()-start
          rows.append(row)
          with (dest/'turns.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,default=str)+'\n')
          print(json.dumps({'case':label,'seconds':round(row['elapsed_seconds'],2),'proposal':row.get('proposal_returned'),'error':row.get('error'),'model_calls':len(row['model_calls'])}),flush=True)
    write('summary.json',{'turns':len(rows),'cases':len(cases),'proposals':sum(r.get('proposal_returned',False) for r in rows),'errors':[r for r in rows if r.get('error')],
      'unchanged_before_acceptance':sum(r.get('unchanged_before_acceptance',False) for r in rows),
      'model_calls':sum(len(r['model_calls']) for r in rows),'scope':'Engineering check on one saved profile. Inspect raw turns for interpretation quality; no general accuracy claim.'})

if __name__=='__main__':main()
