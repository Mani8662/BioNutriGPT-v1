"""Integrate audited saved LASI results without changing original experiment outputs."""
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parent.parent
P=ROOT/'Research Paper/BioNutriGPT_Manuscript_v2.md'


def main():
    t=P.read_text(encoding='utf-8')
    if '### 2.9 LASI real-profile computational evaluation' in t:
        print('LASI manuscript update already applied.');return
    t=t.replace('Eleven targeted chatbot regression checks and 13 existing production checks passed.',
        'An earlier repair verification passed 24 software checks; the current audit reran 11 chatbot checks and found one failing allergy-declaration expectation. On 100 LASI-derived Indian profiles evaluated in three dietary modes, saved optimizer acceptance was 91%, 96%, and 98%.')
    t=t.replace('Comparative evaluation on the frozen current catalog, independent acceptability assessment, and geographic validation are required',
        'Broader, equal-information comparative evaluation, independent acceptability assessment, and geographic prediction validation are required')
    methods='''### 2.9 LASI real-profile computational evaluation

The new evaluation uses LASI Wave 1 anthropometry, demographics and reported diagnoses [11]. The extraction summary reports 66,859 merged biomarker/individual records and 65,368 eligible records after age, anthropometric, BMI and diagnosis-completeness filters. These are archive-specific analytic counts, not the official total LASI recruitment count. A deterministic sample of 100 unique profiles contains 25 diabetes-only, 25 obesity-only, 25 CVD-only and 25 multimorbid profiles, with 52 male and 48 female profiles across 29 states/UTs. Eligibility includes ages 18–100, measured height 120–210 cm, weight 30–180 kg and BMI 14–55. Disease responses must have valid yes/no codes. Diabetes is self-reported; obesity strata use measured BMI at least 30; CVD combines the selected heart-disease and stroke items. National disease-distribution estimates are not inferred: the benchmark is deliberately disease- and sex-balanced and unweighted. The sampling frame of LASI principally concerns older adults and spouses, rather than all Indian adult age groups.

The benchmark adapter computes energy from observed age, sex, height, weight and a coarse activity-category mapping. It invokes the production nutrient-bound function directly, assigns disease-routing values of 0.85 for positive status and 0.10 for negative status, and sets sodium/potassium baselines to 2800/2600 mg. Other unavailable nutrient baselines are rule-derived. These values are assumptions, not measured dietary intake or Stage 1 probabilities. This evaluates adapted Stage 2 bounds with Stage 3 and Stage 4; it does not execute the full Stage 2 counterfactual optimization or externally validate Stage 1.

Stage 3 applies three dietary modes to each of the same 100 profiles, yielding 300 solves with two-second limits. The conversational report contains 30 scenarios and 60 scripted turns, plus a matched 12-scenario, 24-turn Qwen subset and a no-change-penalty ablation. Inspection of all saved scenario labels found only diabetes-only cases. The source selects the first ten accepted profiles per mode from the stratum-ordered input, explaining the narrower coverage. Thirty scenarios must not be described as thirty distinct participants; participant identifiers are not retained in the saved dialogue scenario logs.

The prompt-only comparator receives energy, protein and B12 values but not the complete nutrient coefficient matrix or full plate/objective specification. The comparison is therefore not equal-information evidence of general LLM inferiority. The evaluator treats an unsuccessful two-second oracle solve as grounds for a correct refusal, although timeout does not establish infeasibility. Number-proximity checks and truncated answer excerpts limit retrospective verification. No new model inference was performed during this audit.

'''
    t=t.replace('## 3. Results',methods+'## 3. Results',1)
    results='''### 3.5 LASI profile evaluation and audited interpretation

Case-level accepted flags and failure counts reconcile with the saved 300-solve summary. The same 100 unique people were evaluated repeatedly across modes.

| Dietary mode | Accepted / 100 | Reported infeasible | Timeout without valid plan | Median solve time |
|---|---:|---:|---:|---:|
| Vegetarian | 91 | 9 | 0 | 2003.1 ms |
| Combined vegetarian and non-vegetarian | 96 | 1 | 3 | 2005.6 ms |
| Non-vegetarian | 98 | 0 | 2 | 2006.1 ms |

Overall, 285/300 mode-specific solves were accepted. This is not a population-weighted national acceptance rate. The vegetarian low-energy subgroup had 21/27 accepted plans below 1500 kcal, compared with 27/27 in each other mode. Catalog or rule limitations at lower energy therefore warrant investigation. Accepted flags come from the production solver path; saved case rows do not contain full baskets and nutrient certificates for independent numerical replay.

The guarded conversational system was script-scored complete in 60/60 turns. Among its 30 swap requests, 26 produced validated proposals and four were counted successful without a proposal after an unsuccessful oracle search. Thus 60/60 is not a claim that every requested swap was performed or that feasibility was proven for every refusal. The no-penalty ablation also recorded 26 validated proposals. Mean serving change was 9.46 with the penalty versus 16.08 without it, a descriptive reduction of 41.1%; unchanged-dish ratios were 51.0% versus 28.8%. System-specific adaptation of later swap requests and solver time limits prevent interpreting this alone as a controlled causal effect.

On the matched 24-turn subset, the guarded system was scored complete in 24/24 and Qwen in 2/24. Of 12 swap requests, the guarded system recorded ten valid proposals and Qwen none; the two Qwen successes were no-proposal outcomes scored as refusals. The reported zero nutrient violations for Qwen therefore do not demonstrate successful compliant planning. The reported 3/3 unsupported claims relate to three selected nutrient-answer checks, not exhaustive hallucination assessment. These results support feasibility under the tested scripts but require balanced scenario selection, equal information and fuller logging before a strong comparative claim.

The audit also reran the 11 targeted chatbot regression checks: ten passed and one failed because the current parser records an explicit allergy statement ending in a question mark, whereas the earlier test expects no state change. This contract discrepancy must be resolved; the earlier 24-check pass is historical verification, not proof that the current source passes all checks.

'''
    t=t.replace('## 4. Discussion',results+'## 4. Discussion',1)
    t=t.replace('All 24 defined software checks passed: 11 new conversational-repair checks and 13 existing production checks.',
        'The earlier repair report recorded 24 passing software checks: 11 conversational-repair checks and 13 production checks. The current LASI audit reran the 11 chatbot checks and found ten passes and one failing expectation; the full 24-check suite was not rerun.')
    t=t.replace('The current software tests do not replace a fresh comparative study. Two-profile conversational evaluation is too narrow',
        'The new LASI pilot adds real-profile evidence but does not resolve the current comparator-information and scoring limitations. The earlier two-profile conversational evaluation is too narrow')
    t=t.replace('Finally, no completed dietitian evaluation, prospective intervention, longitudinal outcome study, or Indian cohort validation is established by the reviewed artifacts.',
        'Finally, the LASI study is Indian real-profile computational testing, not Indian prediction validation or a clinical intervention. No completed dietitian evaluation, prospective intervention or longitudinal outcome study is established by the reviewed artifacts.')
    t=t.replace('The next study should freeze the current catalog, code, rules, prompts, and test scenarios before execution.',
        'The next study should repair the identified LASI pilot limitations and freeze the catalog, code hashes, rules, full prompts, scenarios and logging before execution. Cover all four disease strata in the conversational suite and distinguish proven infeasibility from timeout or unresolved feasibility.')
    t=t.replace('The central hypothesis—reliable repeated personalization without losing declared constraints—requires a frozen current-version comparative study before a strong effectiveness or novelty claim can be made.',
        'The LASI pilot strengthens Indian profile relevance and provides preliminary evidence for less disruptive revisions. Equal-information comparisons, broader dialogue coverage, reliable refusal scoring and an entirely passing regression suite remain necessary before a strong effectiveness or novelty claim can be made.')
    t+='\n11. Cohort Profile The Longitudinal Ageing Study in India LASI. [Cohort description](https://pmc.ncbi.nlm.nih.gov/articles/PMC9365624/); [official LASI data information](https://iipsindia.ac.in/content/LASI-data).\n'
    P.write_text(t,encoding='utf-8')
    original=ROOT/'BioNutriGPT_Research_Manuscript.md'
    backup=ROOT/'BioNutriGPT_Research_Manuscript.pre_LASI_update.md'
    if not backup.exists():shutil.copy2(original,backup)
    original.write_text(t,encoding='utf-8')
    print('Updated working and root manuscripts; preserved the previous root version.')


if __name__=='__main__':main()
