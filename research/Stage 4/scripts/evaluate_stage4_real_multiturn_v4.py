"""One saved Stage 2 case, real catalog/solver, isolated conversation sessions.

Default prepares a protocol only. --run requires separate user authorization.
No national/clinical/general chatbot accuracy claim follows from this check.
"""
import argparse
import copy
import dataclasses
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'backend'))
import conversation_service as chat
import chat_service as core
import chat_interpreter
import plan_store
import stage3_milp_solver as s3
from benchmark_stage4_conversational_revisions import _validate_raw_basket_dict

DEST=ROOT/'Saved Model/stage4_real_multiturn_v4'
SOURCE=ROOT/'Saved Model/stage3_live_sample_stage2.json'
FILES=['backend/conversation_service.py','backend/chat_interpreter.py','backend/chat_service.py',
       'backend/plan_store.py','stage3_milp_solver.py','evaluate_stage4_real_multiturn_v4.py']
PROTOCOL={
 'scope':'One existing saved Stage 2 sample; real catalog and solver. Not new patients or clinical validation.',
 'selection':'First active food by food_id with an alternative of the same category; first complete alternative by food_id. No feasibility searching or pair replacement.',
 'initial_solve_seconds':20,'revision_solve_seconds':2,
 'scenarios':[
  ['How much protein is in {source}?','And how many calories does it contain?'],
  ['Could I have {target} in place of {source}?','2 servings'],
  ["I'd rather have {target} instead of {source}, two servings please."],
  ['Do not replace {source} with {target}','Replace {source} with 2 servings of {target} at dinner'],
  ['I am fed up with {source}; take it off my plan.','Check this interpreted request'],
 ],
 'acceptance':'For every returned proposal: independently validate, verify pre-acceptance basket unchanged, reject tampered proposal, accept original, check version and requested exclusion/quantity, rollback and compare original basket.',
 'failure_policy':'Persist all responses, failures and exceptions. No proposal means revision not completed; do not count refusal as feasibility success. No automatic retry or alternate food search.',
 'limitations':'Small engineering evaluation, includes regression wording; not a held-out research benchmark. Saved Stage 2 provenance remains its original scope. No feasibility oracle for absent proposals.',
}

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,obj):
    (DEST/name).write_text(json.dumps(obj,indent=2,default=str)+'\n',encoding='utf-8')
def log(obj):
    with (DEST/'turns.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(obj,default=str)+'\n')
def basket(record):return plan_store.get_current_version_entry(record)['stage3_plan']['solved_food_basket']

def main(run):
    catalog=s3.load_food_database()
    stage2=json.loads(SOURCE.read_text())
    binding={'stage2_sha256':digest(SOURCE),'source_sha256':{p:digest(ROOT/p) for p in FILES},
             'catalog_sha256':hashlib.sha256(json.dumps([dataclasses.asdict(f) for f in catalog],sort_keys=True,default=str).encode()).hexdigest()}
    DEST.mkdir(exist_ok=True)
    if not run:
        if (DEST/'turns.jsonl').exists() or (DEST/'initial.json').exists():raise RuntimeError('Run already exists')
        write('protocol.json',PROTOCOL);write('binding.json',binding)
        print(json.dumps({'status':'prepared','catalog_count':len(catalog),'scenarios':PROTOCOL['scenarios'],'scope':PROTOCOL['scope']},indent=2));return
    if json.loads((DEST/'binding.json').read_text())!=binding:raise RuntimeError('Inputs changed since preparation')
    if json.loads((DEST/'protocol.json').read_text())!=PROTOCOL:raise RuntimeError('Protocol differs')
    if (DEST/'initial.json').exists():raise RuntimeError('Existing run; refusing overwrite')
    out=s3.solve_meal_plan(stage2,catalog,time_limit_sec=20,revision_spec={'dietary_mode':'veg_nonveg_combo'})
    write('initial.json',out.to_dict())
    if not out.accepted:
        write('summary.json',{'status':'blocked_initial_plan','failure':out.to_dict().get('failure'),'scope':PROTOCOL['scope']});return
    valid=s3.validate_candidate(out,catalog,stage2,dietary_mode='veg_nonveg_combo')
    write('initial_validation.json',valid)
    if not valid['accepted']:raise RuntimeError('Initial independent validation failed')
    initial=out.to_dict();initial['revision']={'dietary_mode':'veg_nonveg_combo','status':'initial'}
    dossier={'stage2_counterfactual_prescription':stage2,'stage3_discrete_meal_synthesis':initial}
    lookup={f.food_id:f for f in catalog};pair=None
    for x in sorted(initial['solved_food_basket'],key=lambda x:x['food_id']):
        alternatives=sorted([f for f in catalog if f.category==lookup[x['food_id']].category and f.food_id!=x['food_id'] and not f.unknown_nutrients],key=lambda f:f.food_id)
        if alternatives:pair=(x,alternatives[0]);break
    if pair is None:raise RuntimeError('No eligible fixed pair')
    source,target=pair
    write('frozen_pair.json',{'source':source,'target':dataclasses.asdict(target)})
    results=[]
    for si,templates in enumerate(PROTOCOL['scenarios']):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);sid=plan_store.create_session(dossier,store_dir=folder)['session_id']
            for ti,template in enumerate(templates):
                question=template.format(source=source['name'],target=target.name)
                before=plan_store.get_session(sid,store_dir=folder);start=time.perf_counter()
                row={'scenario':si+1,'turn':ti+1,'question':question}
                try:
                    response=chat.answer(question,dossier,session_id=sid,store_dir=folder,use_llm_interpretation=True)
                    after=plan_store.get_session(sid,store_dir=folder)
                    row.update(response=response,elapsed_ms=(time.perf_counter()-start)*1000,
                        unchanged_before_acceptance=basket(before)==basket(after) and before['current_version']==after['current_version'])
                    proposal=response.get('proposed_meal_plan')
                    if proposal:
                        rev=proposal['revision']
                        validation=_validate_raw_basket_dict(proposal['solved_food_basket'],catalog,stage2,
                            excluded_ids=rev['excluded_food_ids'],locked_quantities=rev['locked_food_quantities'],dietary_mode=rev['dietary_mode'])
                        row['independent_validation']=validation
                        if not validation['accepted']:raise RuntimeError('Proposal failed independent validation')
                        tampered=copy.deepcopy(proposal);tampered['solved_food_basket'][0]['quantity']+=1
                        try:core.accept(dossier,tampered,session_id=sid,store_dir=folder);row['tamper_rejected']=False
                        except ValueError:row['tamper_rejected']=True
                        if not row['tamper_rejected']:raise RuntimeError('Tampered proposal accepted')
                        core.accept(dossier,proposal,session_id=sid,store_dir=folder)
                        committed=plan_store.get_session(sid,store_dir=folder)
                        row['accepted_version_advanced']=committed['current_version']>before['current_version']
                        row['accepted_basket_matches']=basket(committed)==proposal['solved_food_basket']
                        core.rollback(sid,before['current_version'],store_dir=folder)
                        row['rollback_restored']=basket(plan_store.get_session(sid,store_dir=folder))==basket(before)
                    else:row['revision_completion']='no proposal; feasibility not established'
                except Exception as exc:
                    row.update(error_type=type(exc).__name__,error=str(exc))
                log(row);results.append(row);print(json.dumps(row,default=str),flush=True)
    write('summary.json',{'status':'completed','turns':len(results),'proposals':sum('independent_validation' in r for r in results),
        'exceptions':sum('error' in r for r in results),'unchanged_before_acceptance':sum(r.get('unchanged_before_acceptance') is True for r in results),
        'scope':PROTOCOL['scope'],'limitations':PROTOCOL['limitations'],'review_required':'Read every turn; no-proposal turns are not automatically passes.'})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',action='store_true');main(ap.parse_args().run)
