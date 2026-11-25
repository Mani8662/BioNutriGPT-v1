"""Build corrected analysis and extended verification without changing frozen run."""
from pathlib import Path
R=Path(__file__).resolve().parent
s=(R/'analyze_stage2_comparison_study_v1.py').read_text().replace('stage2_comparison_study_v1','stage2_comparison_study_v2')
s=s.replace('import numpy as np','import numpy as np\nfrom scipy.stats import binomtest')
s=s.replace('p_val = float(2.0 * min(np.mean(boot_diffs <= 0), np.mean(boot_diffs >= 0)))','discordant = int(np.count_nonzero(diff))\n    p_val = float(binomtest(int(np.sum(diff > 0)), discordant, .5).pvalue) if discordant else 1.0')
s=s.replace('"ci_lower": float(np.percentile(dist_diff, 2.5)),\n            "ci_upper": float(np.percentile(dist_diff, 97.5)),','"ci_lower": paired_bootstrap_contrast(arm_stats[c_arm]["_raw_dist"], arm_stats[b_arm]["_raw_dist"])["ci_lower_pct"] / 100,\n            "ci_upper": paired_bootstrap_contrast(arm_stats[c_arm]["_raw_dist"], arm_stats[b_arm]["_raw_dist"])["ci_upper_pct"] / 100,')
# The binary p-value is only defined for binary endpoints; distance uses CI only.
s=s.replace('p_val = float(binomtest(int(np.sum(diff > 0)), discordant, .5).pvalue) if discordant else 1.0','p_val = (float(binomtest(int(np.sum(diff > 0)), discordant, .5).pvalue) if discordant else 1.0) if set(np.unique(diff)).issubset({-1.,0.,1.}) else None')
s=s.replace('"two_sided_p": min(1.0, p_val),','"two_sided_p": p_val,')
(R/'analyze_stage2_comparison_study_v2.py').write_text(s,encoding='utf-8')
s=(R/'verify_stage2_comparison_study_v1.py').read_text().replace('stage2_comparison_study_v1','stage2_comparison_study_v2').replace('from verify_stage2_constraint_audit_repaired_v1 import check','from stage2_comparison_checks_v2 import check')
s=s.replace('    rec_path = STUDY_DIR / "verification.json"','    receipt["verification_scope"] = "Fresh model rescoring and numerical/count verification; full statistics verified separately in audit_complete.json"\n    rec_path = STUDY_DIR / "verification.json"')
(R/'verify_stage2_comparison_study_v2.py').write_text(s,encoding='utf-8')
print('Created v2 analysis and model replay verifier.')
