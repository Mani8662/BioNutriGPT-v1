from pathlib import Path

p = Path("Research Paper/BioNutriGPT_Manuscript_v2.md")
content = p.read_text(encoding="utf-8")

target = """These findings do not establish clinical benefit, Indian validity or discrete meal feasibility. See Stage2_Comparison_Study_v2.md and Saved Model/stage2_comparison_study_v2/audit_complete.json.

### 3.4 Historical optimizer comparison: 29-entry catalog"""

addition = """These findings do not establish clinical benefit, Indian validity or discrete meal feasibility. See Stage2_Comparison_Study_v2.md and Saved Model/stage2_comparison_study_v2/audit_complete.json.

### 3.4 Stage 2 to Stage 3 translation study: matched discrete meal synthesis

A matched in-silico study evaluated whether model-guided continuous targets improve downstream discrete meal plan synthesis compared with minimum-change projection or deterministic rules. The study evaluated 100 reserved temporal NHANES adult participants across three dietary modes (`pure_vegetarian`, `veg_nonveg_combo`, `pure_non_vegetarian`) and the three Stage 2 target methods, totaling 900 solves using HiGHS MILP on a 49-dish cooked Indian catalog under a 3.0-second budget.

1. **Contract Analysis & Feasible Set Invariance:** An architectural audit established that Stage 3 enforces component nutrient bounds, energy conservation ($T_{\\text{kcal}}$), and plate variety as hard constraints—all identical across arms. Continuous target vectors $\\mathbf{n}^*$ are excluded from the discrete meal optimizer (slacks bounded to zero). Consequently, mathematical feasible sets and objective functions are identical across arms, yielding identical discrete meal plans for any given profile and mode.
2. **Evaluation A (Diagnostic Meal Feasibility):** Stage 3 feasibility across all 300 solves per arm was 99.3% (298/300) for rules-only, 99.0% (297/300) for minimum-change, and 99.3% (298/300) for model-guided. The paired difference between model-guided and minimum-change was +0.33 percentage points (exact paired binary $p = 1.000$, 1 win, 0 loss), demonstrating that model guidance does not change intrinsic discrete meal feasibility.
3. **Evaluation B (End-to-End Pipeline Acceptance):** When evaluated as an integrated pipeline requiring both Stage 2 score-gate satisfaction and Stage 3 discrete feasibility, overall acceptance was 48.0% (144/300) for rules-only, 69.7% (209/300) for minimum-change, and 77.0% (231/300) for model-guided. The primary paired contrast (Model-Guided vs Minimum-Change) showed a +7.33 pp advantage (95% CI: [+2.00, +13.33] pp, exact $p = 2.74 \\times 10^{-5}$), driven entirely by upstream Stage 2 compliance.

Full artifacts and independent verification records are preserved in `Saved Model/stage2_stage3_translation_study_v1/` and `Research Paper/Stage2_Stage3_Translation_Study_v1.md`.

### 3.5 Historical optimizer comparison: 29-entry catalog"""

# Also renumber 3.4 Historical conversational comparison to 3.6, and 3.5 LASI to 3.7
target_conv = "### 3.4 Historical conversational comparison: 31-entry catalog"
addition_conv = "### 3.6 Historical conversational comparison: 31-entry catalog"

target_lasi = "### 3.5 LASI profile evaluation and audited interpretation"
addition_lasi = "### 3.7 LASI profile evaluation and audited interpretation"

if target in content:
    content = content.replace(target, addition)
    content = content.replace(target_conv, addition_conv)
    content = content.replace(target_lasi, addition_lasi)
    p.write_text(content, encoding="utf-8")
    print("Successfully patched manuscript!")
elif target.replace("\n", "\r\n") in content:
    content = content.replace(target.replace("\n", "\r\n"), addition.replace("\n", "\r\n"))
    content = content.replace(target_conv.replace("\n", "\r\n"), addition_conv.replace("\n", "\r\n"))
    content = content.replace(target_lasi.replace("\n", "\r\n"), addition_lasi.replace("\n", "\r\n"))
    p.write_text(content, encoding="utf-8")
    print("Successfully patched manuscript with CRLF!")
else:
    print("Target not found!")
