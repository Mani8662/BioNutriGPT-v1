from pathlib import Path

import json,hashlib

import pandas as pd

R=Path(__file__).resolve().parent;O=R/'Saved Model/stage1_robustness_endpoint_audit_v2';S=R/'Saved Model/stage1_robustness_uncertainty_v1'

def main():

 receipt=json.loads((O/'verification.json').read_text());assert receipt['passed']

 w=pd.read_csv(S/'within_model_deterioration.csv');d=pd.read_csv(S/'between_model_contrasts.csv');seed=json.loads((S/'bootstrap_seed_record.json').read_text())

 sel=d[(d.variant=='calibrated')&d.outcome.isin(['TARGET_DIABETES','TARGET_CVD'])&d.comparator.isin(['catboost','ft_transformer'])].copy()

 sel['excludes']=(sel.diff_deterioration_ci_lower>0)|(sel.diff_deterioration_ci_upper<0)

 summary=[]

 for metric,g in sel.groupby('metric'):

  ex=g[g.excludes];summary.append(f'{metric.upper()}: {len(g)} contrasts; {len(ex)} intervals exclude zero ({sum(ex.diff_deterioration_point<0)} favour BioNutriGPT; {sum(ex.diff_deterioration_point>0)} favour the comparator).')

 statement=' '.join(summary)+' These are exploratory, unadjusted pointwise intervals. Scenario-specific findings do not establish an overall robustness advantage or equivalence.'

 report='# Stage 1 robustness uncertainty: independently verified results\n\nUpdated 2026-11-22. Post-hoc evaluation of frozen predictions for 7,004 temporal participants, six models and eight stress scenarios; raw and calibrated probabilities, three outcomes and two metrics.\n\n'+statement+'\n\n## Independent verification\n\nAll 2,688 interval pairs (5,376 endpoints) were independently reproduced using the exact 10,000 participant resamples. This comprises three intervals in each of 576 within-model rows and two intervals in each of 480 between-model rows. A count-weighted tied-score pair-counting AUROC implementation was used instead of the original expanded-sample trapezoidal calculation. All within-model baseline/stress point estimates were also checked with sklearn. No model was retrained and no prediction was regenerated. Original reports/receipts are retained under stage1_robustness_endpoint_audit_v2/before_correction/.\n\nResample SHA-256: `'+seed['resample_sha256']+'`. Source bindings passed. Full endpoint receipt: Saved Model/stage1_robustness_endpoint_audit_v2/verification.json. The older verifier only checked a bootstrap mean and did not reproduce interval endpoints; it is superseded for that purpose.\n\n## BioNutriGPT calibrated deterioration\n\nPositive deterioration means worse: baseline minus stress for AUROC, stress minus baseline for Brier.\n\n| Outcome | Scenario | Metric | Baseline | Stressed | Deterioration [95% CI] |\n|---|---|---|---:|---:|---|\n'

 for row in w[(w.model=='bionutrigpt')&(w.variant=='calibrated')&w.outcome.isin(['TARGET_DIABETES','TARGET_CVD'])].itertuples():

  report+=f'| {row.outcome} | {row.scenario} | {row.metric} | {row.baseline_point:.6f} | {row.stressed_point:.6f} | {row.deterioration_point:+.6f} [{row.deterioration_ci_lower:+.6f}, {row.deterioration_ci_upper:+.6f}] |\n'

 report+='\n## Direct contrasts whose pointwise intervals exclude zero\n\nDifference = BioNutriGPT deterioration minus comparator deterioration; negative means BioNutriGPT deteriorated less. Absolute stressed performance remains a separate consideration.\n\n| Comparator | Outcome | Scenario | Metric | Difference [95% CI] |\n|---|---|---|---|---|\n'

 for row in sel[sel.excludes].itertuples():report+=f'| {row.comparator} | {row.outcome} | {row.scenario} | {row.metric} | {row.diff_deterioration_point:+.6f} [{row.diff_deterioration_ci_lower:+.6f}, {row.diff_deterioration_ci_upper:+.6f}] |\n'

 report+='\n## Limits\n\nIntervals are conditional on fitted models and the specific perturbation realizations. They do not include training variability, complex survey design or uncertainty across new perturbation seeds. Stress magnitudes are engineering scenarios, not established clinical measurement-error distributions. Obesity and raw-probability results remain exploratory. Smaller deterioration does not automatically mean higher absolute performance. No clinical utility, Indian external validity or journal-tier claim follows.\n'

 (S/'Report.md').write_text(report,encoding='utf-8');(O/'Report.md').write_text(report,encoding='utf-8')

 # Rewrite the uncertainty extension while retaining the original robustness point-estimate section.

 p=R/'Research Paper/Stage1_Robustness_v3_Results.md';text=p.read_text(encoding='utf-8');cut=text.find('Completed post-hoc paired uncertainty');text=text[:cut] if cut>=0 else text+'\n';p.write_text(text+report,encoding='utf-8')

 p=R/'Research Paper/BioNutriGPT_Manuscript_v2.md';lines=p.read_text(encoding='utf-8').splitlines();lines=[('- **Direct paired robustness contrasts:** '+statement+' Full independent reproduction of all 2,688 bootstrap intervals is documented in Saved Model/stage1_robustness_endpoint_audit_v2/verification.json.') if line.startswith('- **Direct Paired Contrasts Against Baselines:**') else line for line in lines];p.write_text('\n'.join(lines)+'\n',encoding='utf-8')

 for name in ['Progress.txt','Current_Model_Explanation.txt','Research Paper/Readiness_and_Evidence.md']:

  p=R/name;text=p.read_text(encoding='utf-8');text+='\nROBUSTNESS INTERVAL AUDIT COMPLETE — 2026-11-22\nIndependently reproduced all 2,688 percentile intervals from 10,000 shared participant resamples and checked point estimates. Original numerical CSVs unchanged. '+statement+' Prior interval-verification claims based on a mean-only spot check are superseded. See Saved Model/stage1_robustness_endpoint_audit_v2/verification.json. Indian external prediction validation and production migration remain unfinished.\n';p.write_text(text,encoding='utf-8')

 # Correct current summaries as well as appending the dated audit history.
 p=R/'Research Paper/Readiness_and_Evidence.md';t=p.read_text(encoding='utf-8');lines=t.splitlines()
 lines=[('  - Inter-model contrasts: '+statement) if line.startswith('  - Inter-model contrasts:') else ('  - Verification: All 2,688 interval pairs independently reproduced; see Saved Model/stage1_robustness_endpoint_audit_v2/verification.json. The prior mean-only check did not verify endpoints.') if line.startswith('  - Verification: verify_stage1_robustness_uncertainty_v1.py') else line for line in lines]
 p.write_text('\n'.join(lines)+'\n',encoding='utf-8')
 p=R/'Research Paper/BioNutriGPT_Manuscript_v2.md';t=p.read_text(encoding='utf-8').replace('and 7-check independent verification receipts','and full independent verification of 2,688 interval pairs');p.write_text(t,encoding='utf-8')

 receipt['original_numerical_csvs_unchanged']=True

 receipt['result_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [S/'within_model_deterioration.csv',S/'between_model_contrasts.csv']}

 (S/'verification.json').write_text(json.dumps(receipt,indent=2));(S/'completed.json').write_text(json.dumps({'status':'COMPLETED_FULL_ENDPOINT_VERIFIED','date':'2026-11-22','receipt':'Saved Model/stage1_robustness_endpoint_audit_v2/verification.json','interval_pairs':2688},indent=2))

 print(statement)

if __name__=='__main__':main()

