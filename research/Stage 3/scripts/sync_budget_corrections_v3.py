from pathlib import Path
import json,re,hashlib
R=Path(__file__).resolve().parent;P=R/'Saved Model/stage3_budget_audit_v3'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=read(P/'analysis_summary.json');v=read(P/'verification.json');assert v['status']=='VERIFIED_PASSED' and v['analysis_summary_sha256']==sha(P/'analysis_summary.json')
table='| Budget | Current valid plans | Target-aware valid plans | Difference (pp), participant-cluster 95% CI |\n| --- | ---: | ---: | --- |\n'
for key,s in a['budget_summary'].items():
 d=s['acceptance_difference_pp'];table+=f"| {key} | {s['variant_a_accepted']}/900 | {s['variant_b_accepted']}/900 | {d['mean_difference']:+.2f} [{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}] |\n"
explain='At five seconds, both objectives returned 897/900 valid plans with no discordant cells in this sample. This is identical observed acceptance, not demonstrated equivalence or noninferiority. The [0, 0] empirical bootstrap interval occurs because every sampled paired difference is zero; it does not establish zero population difference. Each budget contains nine repeated target-source/mode cells for each of 100 participants. Pooled McNemar p-values have been removed; participant-cluster intervals and the 27 within-cell exact tests with their Holm adjustments are retained.'
status='At ten seconds, the target-aware solver recorded status 0 for 501/900 tasks under a configured 1% relative-gap tolerance. Describe this as solver termination within that tolerance, not mathematical proof of an exact integer optimum. The three common failures per budget are solver-reported infeasibilities; additional no-plan timeouts exist at three seconds and for one target-aware task at ten seconds. Their cause must not be generalized as proven extreme-calorie infeasibility without further evidence.'
runtime='At ten seconds, recorded mean runtime was approximately 7.214 seconds for target-aware and 9.091 seconds for current-objective solves. This is an observed timing difference. No experiment established earlier branch-and-bound pruning as its cause. No causal explanation for the runtime difference is claimed.'
scope='The corrected statistical verifier reproduced 42 confidence intervals and 27 valid within-cell exact tests and checked all 3,006 input manifest hashes. Three invalid pooled exact tests were withdrawn. This correction reused the existing reconstructed records and prior arithmetic audit; it did not rerun meal optimization or repeat the full basket reconstruction. Existing verifier fixture checks are retained without treating them as new end-to-end corruption tests. Original v2 evidence remains unchanged.'
report='# Stage 3 budget study: reporting correction v3\n\nUpdated 24 November 2026. Production unchanged; no solver reruns.\n\n'+table+'\n'+explain+'\n\n'+status+'\n\n'+runtime+'\n\n'+scope+'\n\nTarget-error findings remain conditional on jointly successful matched pairs and use the declared engineering scales. They describe target attainment, not health outcomes. Formal matched comparisons across budgets and between Stage 2 target sources should precede a deployment decision; this correction does not claim those additional analyses have been completed.\n\nEvidence: `Saved Model/stage3_budget_audit_v3/analysis_summary.json`, `verification.json`, `correction_manifest.json`. Original reports are archived in `original_documents/`.\n'
(R/'Research Paper/Stage3_Budget_Audit_v3.md').write_text(report,encoding='utf-8')
for name in ['Stage3_Budget_Audit_v2.md','Stage3_Budget_Experiment_v1.md']:
 (R/'Research Paper'/name).write_text('# Historical budget-study report — corrected\n\nUse [reporting correction v3](Stage3_Budget_Audit_v3.md). The complete earlier document is preserved in `Saved Model/stage3_budget_audit_v3/original_documents/`. Original raw experiment records remain unchanged. Invalid pooled p-values, equivalence claims, exact-optimality wording and the untested pruning explanation are superseded.\n',encoding='utf-8')
for name in ['Progress.txt','Current_Model_Explanation.txt','Research Paper/Readiness_and_Evidence.md']:
 p=R/name;s=p.read_text(encoding='utf-8-sig');lines=[]
 for line in s.splitlines():
  # Replace specific incorrect budget claims; preserve unrelated historical studies.
  if ('McNemar' in line or 'mcnemar' in line) and any(k in line for k in ['895/900','891/900','897/900','896/900']):
   lines.append('Budget acceptance reporting corrected: see Stage3_Budget_Audit_v3.md. Pooled exact tests withdrawn; participant-cluster intervals retained.');continue
  if any(k in line for k in ['earlier subtree pruning','earlier branch-and-bound subtree pruning','objective gradient enables']):lines.append(runtime);continue
  if ('501/900' in line or '55.67%' in line) and any(k in line.lower() for k in ['optimal','proved','proves']):lines.append(status);continue
  line=line.replace('30 exact paired binomial','27 within-cell exact paired binomial').replace('30 exact tests','27 within-cell exact tests')
  lines.append(line)
 note='STAGE 3 BUDGET REPORTING CORRECTION v3 — 2026-11-24\n'+explain+'\n'+scope+'\nReport: Research Paper/Stage3_Budget_Audit_v3.md.\n\n'
 p.write_text(note+'\n'.join(lines)+'\n',encoding='utf-8')
p=R/'Research Paper/BioNutriGPT_Manuscript_v2.md';s=p.read_text(encoding='utf-8-sig');start=s.index('### 3.4.2 Controlled solver-budget');end=s.index('### 3.5 Historical',start)
p.write_text(s[:start]+'### 3.4.2 Controlled solver-budget experiment: corrected inference\n\n'+table+'\n'+explain+'\n\n'+status+'\n\n'+runtime+'\n\nTarget-aware outputs had lower target error among jointly successful pairs at each measured budget. This conditional objective-aligned result does not establish clinical benefit. See Stage3_Budget_Audit_v3.md and the v3 statistical verification receipt for corrected reporting.\n\n'+s[end:],encoding='utf-8')
paths=['Progress.txt','Current_Model_Explanation.txt','Research Paper/Readiness_and_Evidence.md','Research Paper/BioNutriGPT_Manuscript_v2.md','Research Paper/Stage3_Budget_Audit_v3.md']
(P/'document_sync.json').write_text(json.dumps({f:sha(R/f) for f in paths},indent=2));print(table)
