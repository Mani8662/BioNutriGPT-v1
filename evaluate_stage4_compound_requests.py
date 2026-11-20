"""Small real-solver engineering check; not a held-out clinical benchmark."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'backend'))
import conversation_service as chat
import chat_service as core
import food_groups
import plan_store
import stage3_milp_solver as s3
from benchmark_stage4_conversational_revisions import _validate_raw_basket_dict


def main():
    destination = ROOT / 'Saved Model/stage4_compound_integration_20261120.json'
    if destination.exists():
        raise RuntimeError('Refusing to overwrite existing evidence')
    source = ROOT / 'Saved Model/stage3_live_sample_stage2.json'
    stage2 = json.loads(source.read_text())
    catalog = s3.load_food_database()
    evidence = {'scope': 'One saved Stage 2 profile, real catalog and solver; engineering regression only.',
                'selection': 'First non-fish initial basket item by food_id; retain its daily quantity and request Dinner. No alternate selection or retries.',
                'source_hashes': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in [
                    'backend/compound_requests.py', 'backend/conversation_service.py', 'backend/chat_service.py',
                    'food_group_preferences.json', 'food_composition_cooked_sourced.json',
                    'Saved Model/stage3_live_sample_stage2.json']}}
    def save():
        destination.write_text(json.dumps(evidence, indent=2, default=str), encoding='utf-8')
    save()
    initial = s3.solve_meal_plan(stage2, catalog, time_limit_sec=20,
                                 revision_spec={'dietary_mode': 'veg_nonveg_combo'})
    evidence['initial'] = initial.to_dict()
    save()
    if not initial.accepted:
        return
    plan = initial.to_dict()
    plan['revision'] = {'dietary_mode': 'veg_nonveg_combo', 'status': 'initial'}
    dossier = {'stage2_counterfactual_prescription': stage2, 'stage3_discrete_meal_synthesis': plan}
    forbidden = set(food_groups.excluded_ids(['fish'], catalog))
    item = next(x for x in sorted(plan['solved_food_basket'], key=lambda x: x['food_id']) if x['food_id'] not in forbidden)
    question = f"Remove fish and include {item['quantity']} servings of {item['name']} at Dinner"
    evidence['question'] = question
    with tempfile.TemporaryDirectory() as folder:
        store = Path(folder)
        record = plan_store.create_session(dossier, store_dir=store)
        sid = record['session_id']
        evidence['interpretation'] = chat.answer(question, dossier, session_id=sid, store_dir=store)
        response = chat.answer('Check this interpreted request', dossier, session_id=sid, store_dir=store)
        evidence['response'] = response
        unchanged = plan_store.get_session(sid, store_dir=store)
        evidence['unchanged_before_acceptance'] = unchanged['current_version'] == record['current_version']
        proposal = response.get('proposed_meal_plan')
        if proposal:
            rev = proposal['revision']
            evidence['independent_validation'] = _validate_raw_basket_dict(
                proposal['solved_food_basket'], catalog, stage2, excluded_ids=rev['excluded_food_ids'],
                locked_quantities=rev['locked_food_quantities'], dietary_mode=rev['dietary_mode'])
            assert evidence['independent_validation']['accepted']
            assert not forbidden.intersection(x['food_id'] for x in proposal['solved_food_basket'])
            schedule = core.build_requested_schedule(proposal['solved_food_basket'], rev['meal_requirements'])
            assert core._meal_requirements_met(schedule, {item['food_id']: 'Dinner'})
            core.accept(dossier, proposal, session_id=sid, store_dir=store)
            accepted = plan_store.get_session(sid, store_dir=store)
            evidence['accepted_preferences'] = accepted['preferences']
            assert 'fish' in accepted['preferences']['excluded_food_groups']
            assert accepted['preferences']['meal_requirements'][item['food_id']] == 'Dinner'
            core.rollback(sid, record['current_version'], store_dir=store)
            restored = plan_store.get_current_version_entry(plan_store.get_session(sid, store_dir=store))
            evidence['rollback_restored'] = restored['stage3_plan']['solved_food_basket'] == plan['solved_food_basket']
            assert evidence['rollback_restored']
        else:
            evidence['limitation'] = 'No proposal found within revision solver limit; this is not proof of infeasibility.'
        save()
    print(json.dumps({'proposal': bool(proposal), 'evidence': str(destination)}))


if __name__ == '__main__':
    main()
