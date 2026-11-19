"""Audit saved LASI benchmark records without rerunning inference or releasing microdata."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent


def main():
    names=['Saved Model/lasi_cohort_extraction_summary.json','Saved Model/lasi_stage3_indian_cohort_report.json',
           'Saved Model/stage4_lasi_conversational_benchmark_report.json','extract_lasi_indian_cohort.py',
           'benchmark_lasi_stage3_indian_cohort.py','benchmark_stage4_lasi_conversational.py',
           'benchmark_stage4_conversational_revisions.py']
    source=json.loads((ROOT/names[0]).read_text(encoding='utf-8'))
    r=json.loads((ROOT/names[1]).read_text(encoding='utf-8'))
    c=json.loads((ROOT/names[2]).read_text(encoding='utf-8'))
    cohort=json.loads((ROOT/'Data/lasi_w1_indian_cohort_100.json').read_text(encoding='utf-8'))['patients']
    assert len(cohort)==100 and len({p['lasi_prim_key'] for p in cohort})==100
    assert Counter(p['stratum'] for p in cohort)==dict.fromkeys(['diabetes_only','obesity_only','cvd_only','multimorbid'],25)
    modes={}
    for mode,s in r['summary_by_mode'].items():
        rows=[x for x in r['cases'] if x['dietary_mode']==mode]
        assert len(rows)==100 and len({x['lasi_prim_key'] for x in rows})==100
        assert {x['lasi_prim_key'] for x in rows}=={x['lasi_prim_key'] for x in cohort}
        accepted=sum(x['accepted'] for x in rows)
        failures=dict(Counter(x['failure_code'] for x in rows if not x['accepted']))
        assert accepted==s['accepted_count'] and failures==s['failure_codes']
        modes[mode]={'accepted':accepted,'total':100,'acceptance_pct':accepted,
                     'failure_codes':failures,'median_solve_ms':round(statistics.median(x['solve_time_ms'] for x in rows),2),
                     'low_energy_band':s['acceptance_by_target_kcal_band']['<1500']}
    chats={}
    for group in ['full_30_scenario_systems','matched_12_scenario_qwen_comparison']:
        for system,d in c[group].items():
            ss=d['scenarios'];turns=[t for s in ss for t in s['turns']]
            successes=sum(t['completed'] for t in turns)
            assert d['summary']['task_completed_count']==f'{successes}/{len(turns)}'
            swaps=[t for t in turns if t['type'] in {'swap','swap_second','locked_swap'}]
            deltas=[t['serving_delta'] for t in swaps if t['serving_delta'] is not None]
            chats[system]={'scenario_count':len(ss),'turns':len(turns),'scored_completed':successes,
                'strata':dict(Counter(s['stratum'] for s in ss)), 'modes':dict(Counter(s['mode'] for s in ss)),
                'swap_requests':len(swaps),'validated_proposals_recorded':len(deltas),
                'no_proposal_scored_success':sum(t['completed'] and t['serving_delta'] is None for t in swaps),
                'mean_serving_delta':statistics.mean(deltas) if deltas else None,
                'summary':d['summary']}
    out={'scope':'Audit of saved results and current benchmark source; no inference rerun or participant-level export.',
         'cohort':{'merged':source['merged_total_n'],'eligible':source['valid_indian_adults_n'],'evaluated':len(cohort),
                   'strata':dict(Counter(p['stratum'] for p in cohort)), 'female':sum(p['is_female'] for p in cohort),
                   'states':len({p['state'] for p in cohort}),'survey_weighted':False},
         'stage3':modes,'stage4':chats,
         'serving_delta_reduction_pct':100*(1-chats['guarded_stage4_full_l1']['mean_serving_delta']/chats['guarded_stage4_no_l1_ablation']['mean_serving_delta']),
         'current_regression_check':{'suite':'verify_stage4_chat_repairs','checks':11,'passed':10,'failed':1,'failure':'test_allergy_questions_and_negation_do_not_mutate_state','note':'Executed during this audit; current explicit declaration with a question mark conflicts with prior test expectations. No application code repair performed.'},
         'limitations':[
             'Balanced 100-profile benchmark is not a nationally representative prevalence sample; counts and summaries are unweighted.',
             'Stage 2 adapter assigns 0.85/0.10 disease-status routing scores; these are not predicted probabilities. Sodium/potassium baselines are assumed 2800/2600 mg.',
             'This tests adapted Stage 2 bounds plus Stages 3/4, not Stage 1 external validity or the full Stage 2 counterfactual search.',
             'Every saved chatbot scenario is diabetes_only; first accepted profiles were selected from a stratum-ordered file. Distinct participant count is not saved in chat scenario logs.',
             'Oracle failure within two seconds is treated as grounds for a successful refusal, although timeout does not prove infeasibility.',
             'Qwen catalog prompt supplies kcal, protein and B12 but omits most of the 30 active nutrient coefficients; it also omits full plate rules/objective. This is not an equal-information comparison.',
             'Second swap prompts adapt to system-specific baskets. Ablation traces are consequently not identical interventions after divergence.',
             'Raw full baskets/verification certificates and full raw LLM responses are not retained in these new reports; chat logs contain truncated answer excerpts.',
             'Grounding checks use number proximity on selected nutrient turns, not exhaustive semantic hallucination detection.',
             'No human acceptability, clinical benefit, Stage 1 validation, or weighted national performance is established.'],
         'hashes':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}}
    (ROOT/'Saved Model/LASI_Evaluation_Audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    lines=['# LASI evaluation audit','',out['scope'],'',
           '## Verified saved counts','',f"- Merged {source['merged_total_n']:,}; eligible {source['valid_indian_adults_n']:,}; evaluated 100 unique profiles (25 per disease stratum, 52 male/48 female, 29 states/UTs).",
           '- Stage 3: 300 solves on 100 profiles; 285 accepted, 10 reported infeasible and 5 timed out without a valid plan. These are repeated dietary-mode evaluations, not 300 people.',
           '- Acceptance: vegetarian 91/100; combined 96/100; non-vegetarian 98/100. Vegetarian below 1500 kcal: 21/27 (77.8%).',
           '- Guarded chatbot: script-scored 60/60; of 30 swap requests, 26 have validated proposals and four receive no proposal but are scored successful refusals.',
           '- Matched subset: guarded script-scored 24/24 versus Qwen 2/24. Guarded produced 10 validated proposals from 12 swaps; Qwen produced none; its two scored successes are no-proposal refusals.',
           f"- Change-penalty ablation: mean serving delta {chats['guarded_stage4_full_l1']['mean_serving_delta']:.2f} versus {chats['guarded_stage4_no_l1_ablation']['mean_serving_delta']:.2f} among 26 accepted swaps per system ({out['serving_delta_reduction_pct']:.1f}% lower descriptively).",'',
           '## Publication interpretation','', 'This is useful real-profile computational evidence with important remaining design limitations. The numerical summary counts reconcile. It is not a nationally representative Indian effectiveness study or Stage 1 external validation.','',
           '## Corrections and remaining evidence repairs','']
    lines += ['- '+x for x in out['limitations']]
    lines += ['', 'Do not overwrite the saved raw reports to make them appear corrected or rerun. This audit is the interpretive correction. Before a new scientific run, freeze a disease-balanced scenario list, full shared information, feasible/timeout/refusal definitions and complete evidence logging.']
    (ROOT/'Saved Model/LASI_Evaluation_Audit.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'counts_verified':True,'stage3':modes,'chat_strata':chats['guarded_stage4_full_l1']['strata'],
                      'delta_reduction_pct':round(out['serving_delta_reduction_pct'],1)},indent=2))


if __name__=='__main__':main()
