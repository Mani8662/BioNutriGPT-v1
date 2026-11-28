"""Non-destructive correction of pooled inference and interpretation."""
from pathlib import Path
import json,shutil,hashlib
R=Path(__file__).resolve().parent;OLD=R/'Saved Model/stage3_budget_audit_v2';OUT=R/'Saved Model/stage3_budget_audit_v3'
OUT.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=json.loads((OLD/'analysis_summary.json').read_text())
for row in a['budget_summary'].values():
 row['paired_discordance'].pop('exact_mcnemar_p',None)
 row['paired_discordance']['inference_note']='Descriptive cell counts only; repeated target sources and modes are clustered within participants. No pooled McNemar test.'
a['audit_version']='stage3_budget_audit_v3'
a['correction']='Remove three invalid pooled exact tests. Retain participant-cluster intervals and 27 within-cell exact tests. Identical observed outcomes do not establish equivalence. Status 0 is termination within configured 1% relative gap, not proof of exact optimum. Runtime mechanism untested.'
(OUT/'analysis_summary.json').write_text(json.dumps(a,indent=2))
for name in ['recomputed_LOCAL.json','input_manifest.json','replay_verification.json']:shutil.copy2(OLD/name,OUT/name)
s=(R/'verify_stage3_budget_statistics_v2.py').read_text().replace('stage3_budget_audit_v2','stage3_budget_audit_v3')
start=s.index('        p_mcnemar = exact_paired_binomial_test(disc_a, disc_b)')
end=s.index('        # Bootstrap verification:',start)
s=s[:start]+'''        assert "exact_mcnemar_p" not in p_disc, "Invalid pooled test retained"

'''+s[end:]
(R/'verify_stage3_budget_statistics_v3.py').write_text(s,encoding='utf-8')
savefiles=['Progress.txt','Current_Model_Explanation.txt','Research Paper/Readiness_and_Evidence.md','Research Paper/BioNutriGPT_Manuscript_v2.md','Research Paper/Stage3_Budget_Audit_v2.md','Research Paper/Stage3_Budget_Experiment_v1.md']
archive=OUT/'original_documents';archive.mkdir(exist_ok=True)
for f in savefiles:
 if not (archive/Path(f).name).exists():shutil.copy2(R/f,archive/Path(f).name)
(OUT/'correction_manifest.json').write_text(json.dumps({'original_analysis_sha256':sha(OLD/'analysis_summary.json'),'corrected_analysis_sha256':sha(OUT/'analysis_summary.json'),'solver_calls':0},indent=2))
print('Prepared corrected v3 analysis and verifier; original v2 remains unchanged.')
