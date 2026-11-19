"""Update working publication sources from verified aggregates; no scientific rerun."""
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parent.parent


def main():
    verified=json.loads((ROOT/'Saved Model/stage4_lasi_v4_verified_summary.json').read_text(encoding='utf-8'))
    assert verified['prompt_checks_passed']==24
    p=ROOT/'Research Paper/BioNutriGPT_Manuscript_v2.md'
    for src in [p,ROOT/'BioNutriGPT_Research_Manuscript.md',ROOT/'Current_Model_Explanation.txt']:
        backup=src.with_name(src.stem+'.pre_v4_document_sync'+src.suffix)
        if not backup.exists():shutil.copy2(src,backup)
    t=p.read_text(encoding='utf-8')
    start=t.index('## Abstract');end=t.index('## 1. Introduction')
    t=t[:start]+'''## Abstract

**Background:** Conversational meal planning requires persistent preferences and reliable numerical verification as users request changes.

**Objective:** To develop and evaluate an auditable pipeline linking cardiometabolic screening, nutritional rules, discrete meal optimization and controlled conversational revision.

**Methods:** Stage 1 uses an 87-feature Transformer evaluated on a 7,046-participant NHANES temporal holdout. Downstream planning uses a 49-entry food catalog and 30 active nutrient fields. A separate LASI-derived evaluation includes 100 selected Indian profiles across three dietary modes. The corrected conversational study uses 36 distinct participants and 78 turns per guarded variant, with a matched 12-participant, 24-turn Qwen 2.5 3B comparison. Requests are frozen; incomplete foods are excluded consistently; proposals undergo numerical validation. Uncertainty is estimated by scenario-level resampling.

**Results:** Saved Stage 1 AUROCs were 0.7999 for diabetes, 0.7518 for obesity and 0.8260 for CVD. LASI optimizer acceptance was 91/100, 96/100 and 98/100 across dietary modes. The full guarded conversational system passed 76/78 turns, including 29 validated revisions and 11 justified refusals; two timeout-related failures remained. In the matched comparison, protocol passes were 24/24 versus 2/24, and valid proposals on feasible requests were 9/9 versus 0/9. Among 25 successful pairs with identical input state, mean serving changes were 7.72 with the change penalty versus 16.56 without it, a descriptive reduction of 53.38%.

**Conclusions:** The tested workflow supports auditable, less disruptive meal revision under declared computational rules. These selected-case results do not establish clinical benefit, patient acceptability, national performance or Stage 1 validity in India.

**Keywords:** nutrition decision support; conversational systems; mixed-integer optimization; provenance; tabular Transformer; Indian meals; reproducibility.

'''+t[end:]
    a=t.index('Stage 3 applies three dietary modes',t.index('### 2.9'));b=t.index('## 3. Results',a)
    t=t[:a]+'''Stage 3 applies three dietary modes to each of the same 100 profiles, yielding 300 solves with two-second limits. The final conversational study, v4, uses 36 distinct participants: nine per disease stratum and 12 per dietary mode, with three participants in every mode-by-stratum cell. Each guarded variant is evaluated on 78 turns. A 12-participant subset contains one participant per cell and 24 turns. Participants are selected from profiles with an accepted starting plan; this conditions the conversational evaluation on initial feasibility. The same frozen participants, starting baskets and request sequences are retained across comparisons. The earlier diabetes-only and adaptive-prompt LASI experiments are retained as historical development runs, not pooled into the final results.

The v4 Qwen prompt supplies the same complete-data food eligibility, all 30 active nutrient coefficients at round-trip precision, bounds, objective weights, cumulative restrictions and canonical post-checks. Six catalog entries with missing active values are excluded; prompt catalogs contain 37 foods for vegetarian mode and 43 for the other modes. Food selection costs are engineering parameters, not verified retail prices. The post-checks use 5% energy tolerance, protein 10–35%, carbohydrate 45–65% and fat 20–35% of energy, the recorded baseline sodium/(potassium+1) ratio, and both fiber checks. These are the implementation's acceptance rules, not a claim that every threshold is clinically validated.

The assistant solves revisions under a two-second limit; a separate three-second oracle distinguishes feasible requests, reported proven infeasibility and unresolved search. Timeouts count as protocol failures. Protocol success includes valid proposals, justified refusals, grounded answers and exact restoration as applicable; it is reported separately from proposal production. Nutrient-answer checks use number proximity within 5% relative or 0.3 absolute tolerance and are not exhaustive semantic hallucination detection. The comparison tests a deterministic tool-backed assistant against a specified solver-free LLM configuration, not two otherwise identical language agents.

The change-penalty study reports both 29 trajectory-level successful pairs and a restricted analysis of 25 successful pairs with identical preceding basket and recorded input restrictions. Restricting to successful pairs is a conditional analysis, not a randomized causal estimate. Exploratory 95% intervals use 1,000 participant/scenario-level bootstrap resamples with seed 42; paired comparisons resample the same scenario IDs. Whole-scenario Wilson bounds are also provided under a working independent-binomial model because all-success empirical bootstrap intervals can degenerate. These intervals do not support survey-weighted national inference.

Actual GPU execution used an RTX 3050 with 4-bit NF4 Qwen 2.5 3B, model revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`, greedy decoding and a 512-token output cap. Full prompts, raw outputs, validation certificates, input states, per-turn journals and completed-stage checkpoints are retained locally. Fourteen source/input hashes were unchanged across execution. Warm-model response timing excludes model loading and Qwen prompt construction; GPU work is synchronized. The final checkpointed run took 1,548.32 seconds including benchmark work. A prior attempt failed at report export; its incomplete summaries are archived separately and excluded from the measured final dataset.

'''+t[b:]
    a=t.index('The guarded conversational system was script-scored complete in 60/60');b=t.index('The audit also reran',a)
    t=t[:a]+'''The corrected full guarded run passed 76/78 turns (97.4%). Of 42 revision requests, 29 produced validated proposals, 11 received justified refusals and two ended in timeout-related failure; one of these was an oracle-feasible request. Valid proposals were produced for 29/30 oracle-feasible revisions. No invalid proposals were recorded. Thirty-four of 36 scenarios passed every turn. The turn-weighted scenario-bootstrap 95% interval for protocol success was 93.67–100%; whole-scenario success was 34/36 with Wilson bounds of 81.86–98.46% under the stated working model.

| Matched v4 endpoint | Guarded assistant | Prompt-only Qwen 2.5 3B |
|---|---:|---:|
| Protocol passes | 24/24 | 2/24 |
| Valid proposals on oracle-feasible requests | 9/9 | 0/9 |
| Invalid revision proposals | 0 | 8 |
| Targeted nutrient-answer checks passed | 3/3 | 2/3 |
| Mean response time | 664.14 ms | 46,111.91 ms |

Qwen's two successful turns were targeted nutrient answers; they were not valid meal revisions. Its 22/24 parseable outputs and absence of unknown food IDs did not establish nutritional compliance. Conversely, the guarded 24/24 protocol rate includes three justified refusals, rather than 12 completed swaps. All 12 guarded scenarios passed every turn; the empirical bootstrap interval is degenerate at 100%, whereas the working whole-scenario Wilson lower bound is 75.75%. The comparison supports the tested system/configuration and tasks, not a general ranking of all LLMs.

For the 29 paired successful trajectories, mean serving change was 8.34 with the penalty and 16.76 without it, a 50.21% descriptive reduction. Four later pairs began from different recorded states after earlier outputs diverged. In the 25 identical-input-state pairs, means were 7.72 and 16.56: 53.38% fewer serving changes, with a mean paired difference of 8.84 servings and scenario-bootstrap 95% interval 6.72–11.12. This is evidence of reduced numerical disruption within the selected successful pairs; acceptability and clinical outcomes were not measured.

All 24 saved Qwen prompts were checked for eligibility parity and canonical rules; journals and four completed-stage checkpoints match the final report. Frozen request equality, participant uniqueness, disease balance and source hashes were verified. The final report and aggregate verification summary are the authoritative v4 evidence; historical summaries are not combined with them.

'''+t[b:]
    t=t.replace('The new LASI pilot adds real-profile evidence but does not resolve the current comparator-information and scoring limitations.',
        'The final v4 run repairs the earlier information-parity and timeout-scoring problems, but remains a small, selected, English-language scripted benchmark using one quantized LLM configuration and initially feasible profiles.')
    t=t.replace('The next study should repair the identified LASI pilot limitations and freeze the catalog, code hashes, rules, full prompts, scenarios and logging before execution. Cover all four disease strata in the conversational suite and distinguish proven infeasibility from timeout or unresolved feasibility.',
        'The v4 study now covers all four disease strata and records frozen inputs, full logs and disaggregated feasibility outcomes. The next study should use an independently reserved dialogue set, broaden paraphrases and clinical scenarios, and add stronger tool-using language-model comparators where such claims are intended.')
    t=t.replace('Ablations should remove persistent memory, independent validation, and the revision change penalty individually.',
        'The revision change-penalty ablation is completed in v4. Additional ablations should remove persistent memory and independent acceptance validation separately.')
    t=t.replace('Equal-information comparisons, broader dialogue coverage, reliable refusal scoring and an entirely passing regression suite remain necessary before a strong effectiveness or novelty claim can be made.',
        'The corrected comparison and conditional same-state ablation strengthen the engineering evidence. Independent dialogue testing, stronger comparators, repaired Stage 1 analyses and a reconciled application regression suite remain necessary for broader claims.')
    t=t.replace('Current software verification comprises 11 targeted chatbot regression checks and 13 existing production checks.',
        'The earlier application verification comprises 11 chatbot checks and 13 production checks; a later rerun found one failing allergy-declaration expectation. Separately, eight benchmark-correction checks passed before the final v4 execution, including checkpoint persistence and simulated end-to-end report export.')
    p.write_text(t,encoding='utf-8');(ROOT/'BioNutriGPT_Research_Manuscript.md').write_text(t,encoding='utf-8')
    print('Synchronized root and working manuscripts with verified v4 results; historical reports preserved.')


if __name__=='__main__':main()
