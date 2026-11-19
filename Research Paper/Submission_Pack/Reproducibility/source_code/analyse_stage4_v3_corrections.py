"""Reanalyse existing v3 logs; never label these as repaired-prompt v4 results."""
import hashlib
import json
from pathlib import Path

from stage4_benchmark_analysis import revision_accounting, protocol_uncertainty, paired_ablation, paired_protocol_difference

ROOT=Path(__file__).resolve().parent


def main():
    src=ROOT/'Saved Model/stage4_lasi_conversational_benchmark_v3_report.json'
    report=json.loads(src.read_text(encoding='utf-8'));logs=report['detailed_logs']
    data={'scope':'Post-hoc reanalysis of unchanged v3 logs. Does not repair or rerun the old Qwen prompt.',
          'source_v3_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),
          'accounting':{k:revision_accounting(v) for k,v in logs.items()},
          'uncertainty':{k:protocol_uncertainty(v) for k,v in logs.items()},
          'ablation':paired_ablation(logs['guarded_stage4_full_l1'],logs['guarded_stage4_no_l1_ablation']),
          'matched_difference':paired_protocol_difference(logs['guarded_stage4_matched_subset'],logs['prompt_only_qwen_equal_info_subset']),
          'v4_inference_executed':False,
          'historical_limitations':['Six incomplete active-nutrient records were presented as numeric zeros in the old prompt.',
              'Old prompt post-checks differ from validator: calories 10% vs 5%, carbohydrate 35–65% vs 45–65%, fat 15–35% vs 20–35%; sodium/potassium formula and threshold also differ.',
              'v3 model/config/source hashes were not captured at original execution; new hashes cannot establish the old run configuration.',
              'Same-state post-hoc pairs are a selected successful subset, not a new randomized comparison.']}
    out=ROOT/'Saved Model/stage4_lasi_v3_posthoc_corrections.json'
    out.write_text(json.dumps(data,indent=2),encoding='utf-8')
    a=data['ablation']['same_input_state_pairs'];q=data['accounting']['prompt_only_qwen_equal_info_subset']
    lines=['# Stage 4 benchmark corrections','',data['scope'],'',
           '## Corrections calculated from saved logs','',
           f"- Qwen truly feasible requests with no proposal, excluding timeouts: {q['missed_feasible_other']} (old ambiguous count was 9). Two invalid proposals occurred on feasible requests.",
           f"- Separate diagnostics: {q['malformed_json_no_proposal']} malformed no-proposal outputs; {q['unexplained_null_on_proven_infeasible']} unexplained null on proven-infeasible request. Diagnostic counts may overlap feasibility categories.",
           f"- Same preceding-basket/frozen-request pairs: {a['paired_turns']} of 29 successful trajectory pairs. Mean serving changes {a['full_mean']:.2f} vs {a['no_penalty_mean']:.2f}, descriptive reduction {a['reduction_pct']:.2f}%.",
           '- Added participant/scenario-clustered bootstrap intervals (1,000 draws, seed 42), paired comparison intervals, and Wilson bounds on whole-scenario success. These quantify selected-case uncertainty, not national-population performance.',
           '- A 100% observed pass rate yields a degenerate empirical bootstrap interval; Wilson bounds are provided under an independent-scenario assumption instead of implying no uncertainty.',
           '', '## Corrected future runner','',
           '- benchmark_stage4_lasi_conversational_v4.py retains the v3 roster, frozen messages and initial baskets. It validates those initial plans; it does not silently reselect cases.',
           '- Incomplete foods are excluded using the same active-nutrient completeness rule as Stage 3. Known zeros remain legitimate; unknowns never become prompt zeros.',
           '- Prompt post-checks come directly from canonical validator output, with coefficient precision and the full selection/energy/change objective retained.',
           '- Exact input state and full model prompts are recorded per turn. CUDA execution is required for the LLM run; synchronized GPU timing and context-limit checks are used.',
           '- Source/catalog/scenario hashes and environment/config records are captured before execution; source changes are checked at the end. Actual model revision/device metadata are recorded when the model is loaded.',
           '- Separate same-input-state and trajectory-level ablation summaries and uncertainty are saved in the machine-readable v4 report.',
           '', '## Status','',
           'v3 files remain unchanged. No corrected-prompt v4 performance result exists yet. A new GPU run is necessary before replacing the v3 head-to-head figures in the manuscript. An already existing v4 frozen manifest is not silently overwritten.']
    (ROOT/'Saved Model/Stage4_Benchmark_Corrections.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'same_state_ablation':a,'qwen_revision_accounting':q,'new_inference':False},indent=2))


if __name__=='__main__':main()
