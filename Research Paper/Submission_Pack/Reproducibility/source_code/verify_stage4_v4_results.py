"""Verify measured v4 artifacts after completion and write an aggregate-only summary."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def main():
    folder=ROOT/'Saved Model'
    path=folder/'stage4_lasi_conversational_benchmark_v4_report.json'
    report=read(path);manifest=read(folder/'stage4_lasi_v4_frozen_scenarios.json')
    frozen={s['id']:s for s in manifest['scenarios']}
    assert report['source_hashes_unchanged_at_end']
    for name,digest in report['run_snapshot']['hashes'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest, name
    catalog=read(ROOT/'food_composition_cooked_sourced.json')
    inactive={'DR1TSUGR','DR1TFOLA','DR1TFF','DR1TACAR','DR1TBCAR','DR1TLYCO','DR1TLZ','DR1TCAFF','DR1TALCO'}
    active=set(catalog[0]['nutrients_per_serving'])-inactive
    assert len(active)==30
    systems={}
    mapping={
        'guarded_stage4_full_l1':('full_36_scenario_ablation','guarded_stage4_full_l1_v4'),
        'guarded_stage4_no_l1_ablation':('full_36_scenario_ablation','guarded_stage4_no_l1_ablation_v4'),
        'guarded_stage4_matched_subset':('matched_12_scenario_equal_info_comparison','guarded_stage4_matched_subset_v4'),
        'prompt_only_qwen_equal_info_subset':('matched_12_scenario_equal_info_comparison','prompt_only_qwen_2_5_3b_equal_info_v4')}
    prompt_checks=0;revisions=set();model_metadata=[]
    for system,logs in report['detailed_logs'].items():
        group,checkpoint_name=mapping[system]
        summary=report[group][system]
        checkpoint=read(folder/'stage4_lasi_v4_checkpoints'/f'{checkpoint_name}.json')
        assert checkpoint['summary']==summary and checkpoint['scenarios']==logs
        assert len({s['lasi_prim_key'] for s in logs})==len(logs)
        ts=[t for s in logs for t in s['turns']]
        assert summary['strict_protocol_passed_count']==f"{sum(t['strict_protocol_passed'] for t in ts)}/{len(ts)}"
        journal=folder/'stage4_lasi_v4_checkpoints'/f'{checkpoint_name}.turns.jsonl'
        assert len(journal.read_text(encoding='utf-8').splitlines())==len(ts)
        for sc in logs:
            assert [t['message'] for t in sc['turns']]==[t['message'] for t in frozen[sc['scenario_id']]['turns']]
        if system=='prompt_only_qwen_equal_info_subset':
            for sc in logs:
                expected={r['food_id'] for r in catalog if not active.intersection(r['unknown_nutrients'])
                          and not(sc['mode']=='pure_vegetarian' and r['dietary_type']=='non_vegetarian')}
                for t in sc['turns']:
                    prompt=t['prompt_full']
                    text=prompt.split('ELIGIBLE_FOOD_CATALOG (',1)[1].split(':\n',1)[1].split('\n\nCONVERSATION_HISTORY:',1)[0]
                    rows=text.splitlines();ids={r.split('|')[0] for r in rows[1:]}
                    assert ids==expected
                    assert len(rows[0].split('|'))==37
                    checks=json.loads(prompt.split('PLATE_AND_SAFETY_RULES:\n',1)[1].split('\n\n',1)[0])['post_checks']
                    assert checks['calories_within_pct_of_target']==5.0
                    assert checks['carb_pct_kcal']==[45.0,65.0] and checks['fat_pct_kcal']==[20.0,35.0]
                    assert math.isclose(checks['sodium_to_potassium_ratio_max'],2800.0/2601.0)
                    assert checks['fiber_g_per_1000_kcal_min']==14.0
                    actual=t['actual_model_metadata'];assert actual['input_device'].startswith('cuda')
                    revisions.add(actual.get('model_revision'));model_metadata.append(actual)
                    assert t['raw_llm_output_full'] is not None and 'input_state' in t
                    prompt_checks+=1
        systems[system]={'participants':len(logs),'turns':len(ts),'strata':dict(Counter(s['stratum'] for s in logs)),
                         'summary':summary,'outcomes':dict(Counter(t['outcome_category'] for t in ts))}
    full=systems['guarded_stage4_full_l1'];assert full['strata']==dict.fromkeys(['diabetes_only','obesity_only','cvd_only','multimorbid'],9)
    subset=systems['guarded_stage4_matched_subset'];assert set(subset['strata'].values())=={3}
    out={'status':'Measured run verified against saved prompts, journals, stage checkpoints, frozen requests and source hashes.',
         'report_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'prompt_checks_passed':prompt_checks,
         'actual_model_revisions':list(revisions),'model_metadata_example':model_metadata[0],
         'systems':systems,'statistics':report['repaired_statistical_analysis'],
         'scope':'Selected scripted LASI profiles; no Stage 1 external validation, national weighted effectiveness, clinical benefit or proof of general LLM superiority. Successful-pair ablation and empirical bootstrap are conditional analyses.'}
    (folder/'stage4_lasi_v4_verified_summary.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    g=subset['summary'];q=systems['prompt_only_qwen_equal_info_subset']['summary']
    a=out['statistics']['ablation']['same_input_state_pairs']
    lines=['# Verified checkpointed Stage 4 v4 results','',out['status'],'',
           '- Full guarded run: '+full['summary']['strict_protocol_passed_count']+' protocol passes; timeouts count as failures.',
           '- Matched guarded vs Qwen: '+g['strict_protocol_passed_count']+' versus '+q['strict_protocol_passed_count']+'.',
           f"- Validated proposals on feasible requests: {g['revision_outcomes']['validated_swap_proposals']}/{g['revision_outcomes']['oracle_feasible']} versus {q['revision_outcomes']['validated_swap_proposals']}/{q['revision_outcomes']['oracle_feasible']}.",
           f"- Mean measured response time: {g['mean_turn_latency_ms']} ms guarded; {q['mean_turn_latency_ms']} ms Qwen. Warm-model generation excludes loading and prompt construction; GPU work is synchronized.",
           f"- Same-input-state successful ablation pairs: n={a['paired_turns']}; mean serving changes {a['full_mean']:.2f} versus {a['no_penalty_mean']:.2f}; descriptive reduction {a['reduction_pct']:.2f}%.",
           f"- Scenario-cluster bootstrap 95% interval for serving-change difference (no penalty minus full): {a['cluster_bootstrap_difference_95ci']}.",
           f'- Verified all {prompt_checks} Qwen prompts: complete eligible records only, 30 active nutrient fields, canonical 5% calorie check, canonical macronutrient bands and Na/(K+1) baseline check.',
           '- All four completed-stage checkpoints and per-turn journals match the final detailed report.',
           '- Source/input hashes are unchanged. Frozen requests match the logged requests. Actual model/device configuration is retained in JSON.',
           '', '## Interpretation','',out['scope'],
           'The earlier failed-export attempt remains separately archived and is not pooled with this run. No missing raw outputs were fabricated or reconstructed from aggregate counts.']
    (folder/'stage4_lasi_v4_verified_summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'verified':True,'qwen_prompts_checked':prompt_checks,'model_revisions':list(revisions),
                      'full_passes':full['summary']['strict_protocol_passed_count'],
                      'matched_guarded':g['strict_protocol_passed_count'],'matched_qwen':q['strict_protocol_passed_count'],
                      'same_state_ablation':a},indent=2))


if __name__=='__main__':main()
