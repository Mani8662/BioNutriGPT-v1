from pathlib import Path

p = Path("Research Paper/BioNutriGPT_Manuscript_v2.md")
content = p.read_text(encoding="utf-8")

target = """This computational study evaluated algorithmic constraint satisfaction and does not establish clinical efficacy, disease risk reduction, or Indian clinical validity. Full artifacts and audit records are preserved in `Saved Model/stage1_downstream_study_v1_audit_v3/` and `Research Paper/Stage1_Downstream_Added_Value_Results_v1.md`.

### 3.3 Historical optimizer comparison: 29-entry catalog"""

addition = """This computational study evaluated algorithmic constraint satisfaction and does not establish clinical efficacy, disease risk reduction, or Indian clinical validity. Full artifacts and audit records are preserved in `Saved Model/stage1_downstream_study_v1_audit_v3/` and `Research Paper/Stage1_Downstream_Added_Value_Results_v1.md`.

### 3.3 Stage 2 Comparative Added-Value Study: Model-Guided Search vs. Minimum-Change Projection and Deterministic Rules

A controlled in-silico study evaluated whether model-guided optimization adds measurable engineering value beyond simpler ways of constructing nutrient targets. The study evaluated 100 randomly sampled temporal NHANES adult participants (seed `20261123`, strictly disjoint from the 32 repair cases, 100 downstream study participants, and 5 pilot cases) across three arms:
1. **Arm A (Rules-Only):** Deterministic, closed-form clinical target construction derived strictly from AMDR and DRI references without optimization.
2. **Arm B (Minimum-Change):** Scale-normalized convex quadratic projection of the participant's observed intake vector onto the joint feasible polytope without model risk guidance.
3. **Arm C (Model-Guided):** Full deployed Stage 2: 50-step Adam search on the multi-objective risk loss followed by the identical scale-normalized convex quadratic projection.

All three arms were evaluated under identical participant inputs, sedentary activity assumptions ($1.20 \\times \\text{BMR}$), calorie prescriptions, tightened nutrient bounds embedding Filter 2 invariants, $\\pm 0.99\\%$ thermodynamic energy balance, and three clinical ratio limits. Acceptance required satisfying all 39 nutrient bounds, energy arithmetic, ratio limits, and the closed-loop Stage 1 calibrated score gate ($\\Delta p \\le 0.0015$ across diabetes, obesity, and CVD).

| Metric | Arm A: Rules-Only | Arm B: Minimum-Change | Arm C: Model-Guided |
| :--- | :---: | :---: | :---: |
| **All-Check Acceptance Rate (%)** | 48.0% (48/100) | 70.0% (70/100) | **77.0% (77/100)** |
| **Numerical Feasibility (%)** | 100.0% (100/100) | 99.0% (99/100) | **100.0% (100/100)** |
| **Score Gate Pass Rate (%)** | 48.0% (48/100) | 70.0% (70/100) | **77.0% (77/100)** |
| **Distance from Observed (Median)** | 4.8190 | **1.8494** | 2.1992 |
| **Runtime per Profile (Median)** | **0.031 s** | 0.199 s | 5.735 s |

**Statistical Contrasts (10,000 paired bootstrap resamples, seed `20261123`):**
1. **Primary Contrast (Arm C vs. Arm B):** Model-guided optimization provided a statistically significant increase in all-check acceptance of **+7.00 percentage points** (95% CI: **[+2.00, +13.00] pp**, SE = 2.91 pp, two-sided $p = 0.0154$). Numerical feasibility was comparable (+1.00 pp [0.00, +3.00] pp, $p = 0.7334$). The primary benefit was driven by score-gate compliance (+7.00 pp [+2.00, +13.00] pp).
2. **Secondary Contrast (Arm C vs. Arm A):** Model-guided optimization exceeded deterministic rules by **+29.00 pp** (95% CI: **[+19.00, +39.00] pp**, $p < 0.0001$).
3. **Secondary Contrast (Arm B vs. Arm A):** Minimum-change projection exceeded deterministic rules by **+22.00 pp** (95% CI: **[+11.00, +33.00] pp**, $p = 0.0002$).

**Engineering Trade-Offs & Limitations:** While model-guided optimization provided a statistically significant +7.0 pp advantage in satisfying its own risk score gate, it imposed a **28.5-fold computational overhead** (581.99s total on CUDA vs. 20.44s for Arm B) and displaced targets further from baseline eating habits (median normalized distance 2.1992 vs. 1.8494). Furthermore, score-gate improvements evaluated with the optimization model are self-referential and do not establish clinical efficacy. Full artifacts and audit records are preserved in `Saved Model/stage2_comparison_study_v1/` and `Research Paper/Stage2_Comparison_Study_v1.md`.

### 3.4 Historical optimizer comparison: 29-entry catalog"""

# Normalize newlines for replacement
if target in content:
    content = content.replace(target, addition)
    p.write_text(content, encoding="utf-8")
    print("Successfully patched manuscript!")
elif target.replace("\n", "\r\n") in content:
    content = content.replace(target.replace("\n", "\r\n"), addition.replace("\n", "\r\n"))
    p.write_text(content, encoding="utf-8")
    print("Successfully patched manuscript with CRLF!")
else:
    print("Target not found!")
