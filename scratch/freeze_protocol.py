"""Freeze the experimental protocol and cases for Stage 2 Three-Arm Comparison.

This is executed BEFORE running comparison outcomes on the reserved 100 profiles.
"""
import hashlib
import json
import os
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "Saved Model/stage2_comparison_study_v1"

import stage1_deployment as dep
import stage2_counterfactual_engine as s2
from prepare_stage1_repair_v2 import load

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    OUT.mkdir(parents=True, exist_ok=True)

    sources = [
        ROOT / "Data/stage1_repaired_v2/temporal_clean.csv",
        ROOT / "Data/NHANES_2017_2020_PrePandemic_Raw.csv",
        ROOT / "Saved Model/stage2_comparison_design_v1/selected_participants_LOCAL.json",
        ROOT / "stage2_comparison_arms.py",
        ROOT / "stage2_counterfactual_engine.py",
        ROOT / "stage1_deployment.py",
        ROOT / "nutrient_rule_policy.py",
        dep.BUNDLE / "manifest.json",
        dep.CHECKPOINT,
        dep.PREPROCESSOR,
        dep.BUNDLE / "calibrators.joblib",
        ROOT / "Data/preprocess_for_training.py",
        ROOT / "test_stage2_comparison_arms.py",
        ROOT / "run_stage2_comparison_study_v1.py",
    ]
    binding = {str(p.relative_to(ROOT)): sha(p) for p in sources}

    # Load the 100 reserved participants
    ids_path = ROOT / "Saved Model/stage2_comparison_design_v1/selected_participants_LOCAL.json"
    ids = json.loads(ids_path.read_text(encoding="utf-8"))
    assert len(ids) == 100
    assert len(set(ids)) == 100

    pp = load("freeze_protocol_preprocessing", "Data/preprocess_for_training.py")
    artifacts = s2._load_preprocessing_artifacts(dep.PREPROCESSOR)
    state = pp.PreprocessorState(**{f.name: artifacts[f.name] for f in fields(pp.PreprocessorState)})

    clean = pd.read_csv(ROOT / "Data/stage1_repaired_v2/temporal_clean.csv")
    rawsource = ROOT / "Data/NHANES_2017_2020_PrePandemic_Raw.csv"
    measured = pd.read_csv(rawsource, usecols=["SEQN", "BMXWT", "BMXHT", "RIDAGEYR", "RIAGENDR"]).set_index("SEQN")

    chosen = clean[clean.SEQN.isin(ids)].sort_values("SEQN")
    assert len(chosen) == 100

    inputs = [c for c in clean if c not in set(pp.META_COLS + pp.TARGET_COLS) and c not in pp.MODEL_EXCLUDED_RAW_FEATURES]
    flags = pp.compute_missingness_indicators(clean, inputs)

    cont = chosen[state.base_continuous_cols].astype(float)
    raw_df = pd.concat([
        cont,
        pp.add_clinical_features(cont),
        pp.encode_binary_and_nominal(chosen, state.binary_modes, state.nominal_modes),
        flags.loc[chosen.index]
    ], axis=1)[list(s2.FEATURE_COLS)]

    cases = []
    for i, row in chosen.iterrows():
        x = measured.loc[row.SEQN]
        cases.append({
            "case": len(cases),
            "SEQN": int(row.SEQN),
            "vector": raw_df.loc[i].tolist(),
            "metadata": {
                "weight_kg": float(x.BMXWT),
                "height_cm": float(x.BMXHT),
                "age_years": float(x.RIDAGEYR),
                "is_female": bool(x.RIAGENDR == 2),
                "activity_level": "sedentary",
                "renal_disease": None,
                "pregnant": None,
                "lactating": None,
            }
        })

    cases_path = OUT / "cases_LOCAL.json"
    cases_path.write_text(json.dumps(cases, indent=2), encoding="utf-8")
    cases_sha = sha(cases_path)

    protocol = {
        "version": "stage2_comparison_study_v1",
        "title": "BioNutriGPT Stage 2 Three-Arm Comparative Added-Value Study",
        "research_question": "Does model-guided optimization add measurable engineering value beyond simpler ways of constructing nutrient targets?",
        "sample_size": 100,
        "sample_source": "Reserved disjoint NHANES temporal cohort from Saved Model/stage2_comparison_design_v1",
        "sample_sha256": sha(ids_path),
        "cases_sha256": cases_sha,
        "exclusions_verified": {
            "overlap_with_32_repair_profiles": 0,
            "overlap_with_prior_downstream_study": 0,
            "overlap_with_prior_adapter_pilot": 0,
        },
        "shared_envelope_policy": {
            "activity_assumption": "sedentary (PAL = 1.20) uniformly across all participants and arms",
            "energy_prescription": "Mifflin-St Jeor BMR * 1.20 with -500 kcal deficit if baseline predicted obesity >= 0.50; floor 1200 kcal; bounds [0.95*T_kcal, 1.05*T_kcal]",
            "routing_cutoffs": {"diabetes": 0.50, "obesity": 0.50, "cvd": 0.50},
            "condition_flags": "Baseline calibrated probabilities >= 0.50 trigger branch-specific envelopes (e.g. CVD sodium <= 1500mg, potassium >= 3500mg, fat <= 6%; Diabetes carb <= 45%, fiber floor >= 35g)",
            "immutable_features": "Non-dietary biological, demographic, and lifestyle features strictly preserved (candidate[fixed] == raw[fixed])",
            "linear_constraints": [
                "Thermodynamic energy balance: 4*PROT + 4*CARB + 9*TFAT + 7*ALCO == KCAL +- 1% (+-0.99% projection margin)",
                "Sugar-to-fiber ratio: SUGAR / (FIBER + 1) <= max(base_sugar_to_fiber, 1.0)",
                "Sodium-to-potassium ratio: SODIUM / (POTASSIUM + 1) <= max(base_sodium_to_potassium, 1.0)",
                "Saturated-to-unsaturated fat ratio: SFAT / (MFAT + PFAT + 0.1) <= max(base_sat_to_unsat, 0.4)",
            ],
            "filter_2_invariants": [
                "KCAL >= 1200.0",
                "0.8 * weight_kg <= PROT <= 2.0 * weight_kg",
                "SODI <= 1500.0 (if CVD) else 2300.0",
                "POTA >= 3500.0 (if CVD) else 2600.0",
            ],
            "closed_loop_gate": "Delta p = p_cand - p_0 <= 0.0015 across diabetes, obesity, and CVD rescored through frozen calibrated Stage 1 model",
            "numerical_tolerance": 1e-6,
            "energy_tolerance": 0.01 + 1e-9,
        },
        "arm_definitions": {
            "A_rules_only": {
                "name": "Deterministic Rules-Only Nutrient Targets",
                "method": "Closed-form clinical target construction derived strictly from dietary guidelines, AMDRs, and DRI references",
                "optimizer": None,
                "gradient_search": False,
                "projection_used": False,
                "description": "Sets energy to T_kcal; protein to 1.0 g/kg (bounded by [0.8W, 2.0W]); fat to 25% (or 30% under diabetes); carb by thermodynamic energy balance; fiber to guideline floor; sugar to min(10% energy, ratio cap); saturated fat to 5-7% energy (capped by ratio); MUFA/PUFA by fat breakdown; sodium/potassium to guideline targets capped by ratio; alcohol to 0.0; non-essential carotenoids/caffeine/synthetic folate to 0.0; all other vitamins/minerals set to age/sex DRI RDA/AI reference lower bounds. Uses baseline probabilities solely to define the common routing envelope."
            },
            "B_minimum_change": {
                "name": "Minimum-Change Joint Constraint Projection",
                "method": "Convex quadratic projection of observed nutrient intake vector onto the joint feasible polytope using fixed engineering scales",
                "optimizer": "SLSQP with HiGHS LP feasibility witness warm-start",
                "gradient_search": False,
                "projection_used": True,
                "objective": "min 0.5 * sum_{i=0}^{38} ((n_i - n_obs_i) / s_i)^2 subject to tightened bounds and linear constraints",
                "fallback_behavior": "If SLSQP fails to converge or produces residuals > 1e-6, return verified LP feasibility witness flagged feasible_lp_fallback and minimum_distance_not_established",
                "description": "Finds the closest mathematically feasible target to what the participant currently eats without model-guided loss or risk gradients."
            },
            "C_model_guided": {
                "name": "Full Model-Guided Optimization + Joint Projection",
                "method": "50-step Adam search on multi-objective counterfactual risk loss followed by identical scale-normalized joint projection",
                "optimizer": "PyTorch Adam (lr=0.05, 50 steps) + SLSQP QP projection",
                "gradient_search": True,
                "projection_used": True,
                "objective": "Step 1: 50-step Adam minimizing 2*p_cvd + 2*p_diab + 1*p_obes + 50*ReLU(p-p0) + 0.001*||delta_N||^2 + barrier. Step 2: Scale-normalized convex quadratic projection onto the joint feasible polytope.",
                "fallback_behavior": "Identical LP fallback if SLSQP post-projection fails.",
                "description": "Frozen deployed Stage 2 implementation evaluated under identical common envelope."
            }
        },
        "endpoints": {
            "primary": "All-check acceptance rate (proportion of cases meeting all 39 bounds, energy balance, 3 ratios, Filter 2 invariants, and closed-loop score gate)",
            "secondary": [
                "Numerical feasibility rate (proportion satisfying all numerical constraints)",
                "Closed-loop score gate pass rate (Delta p <= 0.0015 on all 3 outcomes)",
                "Scale-normalized distance from observed intake: sqrt(sum(((n* - n_obs)/s)^2))",
                "Nutrient shifts in major dietary components (KCAL, PROT, CARB, TFAT, SODI, POTA)",
                "Solver status and fallback rate",
                "Execution runtime distribution (median, IQR, 95th percentile)",
                "Per-outcome risk probability changes (Delta p_diabetes, Delta p_obesity, Delta p_cvd)",
            ]
        },
        "statistical_analysis_plan": {
            "comparisons": [
                {"contrast": "Arm C vs Arm B", "role": "Primary Contrast", "purpose": "Isolates the specific value of model-guided Adam search beyond minimum-change projection"},
                {"contrast": "Arm C vs Arm A", "role": "Secondary Contrast", "purpose": "Evaluates full model-guided pipeline against deterministic clinical rules"},
                {"contrast": "Arm B vs Arm A", "role": "Secondary Contrast", "purpose": "Evaluates minimum-change projection against deterministic clinical rules"},
            ],
            "methodology": "Participant-level paired bootstrap (10,000 resamples, fixed random seed 20261123). Reports absolute differences in percentage points and paired 95% percentile confidence intervals [2.5%, 97.5%].",
            "multiplicity_handling": "Hierarchical testing order: Primary contrast (Arm C vs Arm B) tested first at alpha=0.05. Secondary contrasts interpreted with Bonferroni-adjusted alpha=0.025.",
            "missing_data_policy": "All 100 attempted profiles are retained in primary denominators (intention-to-treat equivalent). No failed cases are dropped or replaced with defaults.",
            "interpretation_rules": "Confidence intervals containing zero do not establish clinical or engineering equivalence. Model score improvements evaluated with the optimization model are self-referential and do not prove health benefit."
        },
        "downstream_stage3_scope": "Clearly excluded from this experiment. Stage 3 integer meal planning is not evaluated.",
        "invalidation_conditions": [
            "Source or checkpoint hashes do not match protocol bindings",
            "Non-identical envelopes or participant metadata across arms",
            "Any modification to Stage 1 model weights, calibrators, or routing cutoffs",
            "Silent substitution of missing results with favorable defaults",
        ],
        "hardware": {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
            "torch_threads": torch.get_num_threads(),
            "os": "Windows",
        },
        "hashes": binding,
    }

    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    # Readable markdown protocol explanation
    md_content = f"""# Protocol: Stage 2 Three-Arm Comparative Added-Value Study

**Protocol Version:** `stage2_comparison_study_v1`  
**Date:** 23 November 2026  
**Status:** Frozen prior to execution on reserved cohort  
**Sample:** 100 reserved temporal NHANES adult profiles (Disjoint from 32-case development repair, 100-profile downstream study, and 5-case pilot)  
**Sample SHA-256:** `{sha(ids_path)}`  

## 1. Research Question
Does model-guided optimization add measurable engineering value beyond simpler ways of constructing nutrient targets?

A negative or inconclusive result is completely acceptable. The experiment is not tuned to favor BioNutriGPT.

## 2. Experimental Arms
All three arms evaluate under the exact same participant envelope, calorie prescriptions, nutrient bounds, ratio rules, and closed-loop score gate.

- **Arm A (Rules-only):** Deterministic, closed-form clinical target construction derived strictly from dietary guidelines, AMDRs, and DRI references. Contains zero gradient descent, zero prediction-assisted candidate search, and zero mathematical programming projection. Uses baseline model probabilities solely to define the common routing envelope.
- **Arm B (Minimum-change):** Scale-normalized convex quadratic projection of the participant's observed nutrient intake vector onto the joint feasible polytope. Minimizes deviation from observed intake without model-guided loss or risk gradients.
- **Arm C (Full model-guided):** Frozen deployed Stage 2 implementation: 50-step Adam search on the multi-objective counterfactual risk loss, followed by the identical scale-normalized convex quadratic projection onto the joint feasible polytope.

## 3. Endpoints & Statistical Analysis Plan
- **Primary Endpoint:** All-check acceptance rate (proportion of cases meeting all 39 bounds, energy balance, 3 ratios, Filter 2 invariants, and closed-loop score gate $\\Delta p \\le 0.0015$).
- **Primary Contrast:** Arm C vs. Arm B (isolates the value of model-score guidance).
- **Secondary Contrasts:** Arm C vs. Arm A and Arm B vs. Arm A.
- **Statistical Method:** Participant-level paired bootstrap (10,000 resamples, fixed random seed 20261123). Reports absolute differences in percentage points and paired 95% percentile confidence intervals.

## 4. Integrity and Safeguards
- All 100 attempted profiles remain in the denominator.
- Atomic per-case checkpointing: resume permitted only when inputs, protocol, and code hashes match.
- Solver failure and LP fallback are explicitly distinguished from proven infeasibility.
- Model score improvements evaluated with the optimization model are explicitly disclosed as self-referential.
- Stage 3 meal planning is explicitly outside the scope of this experiment.
"""
    (OUT / "Protocol.md").write_text(md_content, encoding="utf-8")
    print(f"Protocol frozen successfully in {OUT}")
    print(f"Total source hashes bound: {len(binding)}")
    print(f"Cases saved: {len(cases)}, SHA-256: {cases_sha}")

if __name__ == "__main__":
    main()
