#!/usr/bin/env python3
"""Prepare and freeze Stage 2 Component Ablation Study v1.

Samples strictly disjoint pilot (N=3) and full evaluation (N=100) cohorts,
builds the component map, input manifest, variant manifest, and frozen protocol.
Never runs the full evaluation or modifies production code.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import fields
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import stage1_deployment as dep
import stage2_counterfactual_engine as s2
from prepare_stage1_repair_v2 import load

OUT_DIR = ROOT / "Saved Model" / "stage2_ablation_v1"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load clean data and raw demographics
    clean_path = ROOT / "Data/stage1_repaired_v2/temporal_clean.csv"
    raw_path = ROOT / "Data/NHANES_2017_2020_PrePandemic_Raw.csv"
    clean = pd.read_csv(clean_path)
    raw = pd.read_csv(raw_path, usecols=["SEQN", "BMXWT", "BMXHT", "RIDAGEYR", "RIAGENDR"]).set_index("SEQN")
    assert raw.index.is_unique and clean.SEQN.is_unique

    # 2. Eligibility filtering
    pp = load("ablation_cohort_preprocessing", "Data/preprocess_for_training.py")
    inputs = [c for c in clean if c not in set(pp.META_COLS + pp.TARGET_COLS) and c not in pp.MODEL_EXCLUDED_RAW_FEATURES]
    flags = pp.compute_missingness_indicators(clean, inputs)
    joined = clean[["SEQN"]].join(raw, on="SEQN", validate="one_to_one")
    eligible = (
        (flags.sum(axis=1) == 0)
        & joined.BMXWT.between(30, 180)
        & joined.BMXHT.between(120, 210)
        & (joined.RIDAGEYR >= 20)
        & joined.RIAGENDR.isin([1, 2])
    )

    # 3. Known prior evaluations to exclude
    repair_path = ROOT / "Saved Model/stage2_constraint_audit_repaired_v1/cases_LOCAL.json"
    repair_ids = {int(c["SEQN"]) for c in json.loads(repair_path.read_text(encoding="utf-8"))}

    prior_ds_path = ROOT / "Saved Model/stage1_downstream_study_v1/selected_participants_LOCAL.csv"
    prior_ds_ids = set(pd.read_csv(prior_ds_path).SEQN.astype(int))

    pilot_path = ROOT / "Saved Model/stage2_research_adapter_pilot_v1/envelopes_LOCAL.json"
    pilot_ids = {int(c["SEQN"]) for c in json.loads(pilot_path.read_text(encoding="utf-8"))}

    comp_path = ROOT / "Saved Model/stage2_comparison_study_v2/cases_LOCAL.json"
    comp_ids = {int(c["SEQN"]) for c in json.loads(comp_path.read_text(encoding="utf-8"))}

    budget_path = ROOT / "Saved Model/stage3_budget_experiment_v1/cases_LOCAL.json"
    budget_ids = {int(c["SEQN"]) for c in json.loads(budget_path.read_text(encoding="utf-8"))}

    all_prior = repair_ids | prior_ds_ids | pilot_ids | comp_ids | budget_ids
    assert len(all_prior) == 337, f"Expected 337 unique prior IDs, got {len(all_prior)}"

    pool = clean[eligible & ~clean.SEQN.isin(all_prior)].sort_values("SEQN")
    print(f"Total clean temporal records: {len(clean)}")
    print(f"Eligible records: {int(eligible.sum())}")
    print(f"Disjoint eligible pool size: {len(pool)}")
    assert len(pool) >= 103, f"Insufficient pool size: {len(pool)}"

    # 4. Sample 3 pilot cases
    pilot_df = pool.sample(n=3, random_state=20261124).sort_values("SEQN")
    pilot_ids_chosen = set(pilot_df.SEQN.astype(int))

    # 5. Sample 100 evaluation cases from remaining pool
    pool_eval = pool[~pool.SEQN.isin(pilot_ids_chosen)].sort_values("SEQN")
    eval_df = pool_eval.sample(n=100, random_state=20261125).sort_values("SEQN")
    eval_ids_chosen = set(eval_df.SEQN.astype(int))

    assert len(pilot_ids_chosen) == 3
    assert len(eval_ids_chosen) == 100
    assert not (pilot_ids_chosen & all_prior), "Pilot overlaps with prior studies!"
    assert not (eval_ids_chosen & all_prior), "Evaluation cohort overlaps with prior studies!"
    assert not (pilot_ids_chosen & eval_ids_chosen), "Pilot overlaps with evaluation cohort!"

    # 6. Build feature vectors (88 features) and metadata
    artifacts = s2._load_preprocessing_artifacts(dep.PREPROCESSOR)
    state = pp.PreprocessorState(**{f.name: artifacts[f.name] for f in fields(pp.PreprocessorState)})

    def build_cases(df):
        cont = df[state.base_continuous_cols].astype(float)
        raw_df = pd.concat([
            cont,
            pp.add_clinical_features(cont),
            pp.encode_binary_and_nominal(df, state.binary_modes, state.nominal_modes),
            flags.loc[df.index]
        ], axis=1)[list(s2.FEATURE_COLS)]
        cases = []
        for i, row in df.iterrows():
            x = raw.loc[row.SEQN]
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
        return cases

    pilot_cases = build_cases(pilot_df)
    eval_cases = build_cases(eval_df)

    pilot_cases_path = OUT_DIR / "development_participants_LOCAL.json"
    eval_cases_path = OUT_DIR / "evaluation_participants_LOCAL.json"

    pilot_cases_path.write_text(json.dumps(pilot_cases, indent=2), encoding="utf-8")
    eval_cases_path.write_text(json.dumps(eval_cases, indent=2), encoding="utf-8")
    print(f"Saved {len(pilot_cases)} pilot cases to {pilot_cases_path}")
    print(f"Saved {len(eval_cases)} evaluation cases to {eval_cases_path}")

    # 7. Build variant definitions
    variants = {
        "variant_0_reference": {
            "variant_id": "variant_0_reference",
            "name": "Full Deployed Stage 2 Reference",
            "description": "50-step Adam model-guided optimization (model loss + pareto penalty + dietary-change penalty + barrier penalty) + post-Adam clamping + joint convex quadratic projection + downstream filter audits + closed-loop model-score acceptance gate.",
            "include_model_objective": True,
            "include_dietary_change_penalty": True,
            "include_barrier_penalty": True,
            "include_joint_projection": True,
            "include_sequential_filters": True,
            "category": "reference",
        },
        "variant_1_no_model_objective": {
            "variant_id": "variant_1_no_model_objective",
            "name": "Ablate Model-Score Optimization Objective",
            "description": "Sets model risk loss terms (2.0*p_cvd + 2.0*p_diab + 1.0*p_obes) and Pareto penalty (50.0*ReLU(p - p0)) to 0.0 in Adam. Adam minimizes 0.001*||delta_N||^2 + barrier(N, lower, upper) without risk guidance, followed by the identical joint projection.",
            "include_model_objective": False,
            "include_dietary_change_penalty": True,
            "include_barrier_penalty": True,
            "include_joint_projection": True,
            "include_sequential_filters": True,
            "category": "candidate_modifier",
        },
        "variant_2_no_change_penalty": {
            "variant_id": "variant_2_no_change_penalty",
            "name": "Ablate Dietary-Change Penalty in Adam",
            "description": "Sets the 0.001*||delta_N||^2 regularization weight to 0.0 in Adam loss. Optimizes purely for model risk reduction and barrier penalty without penalizing distance from observed intake.",
            "include_model_objective": True,
            "include_dietary_change_penalty": False,
            "include_barrier_penalty": True,
            "include_joint_projection": True,
            "include_sequential_filters": True,
            "category": "candidate_modifier",
        },
        "variant_3_no_barrier_penalty": {
            "variant_id": "variant_3_no_barrier_penalty",
            "name": "Ablate Differentiable Barrier Penalty in Adam",
            "description": "Sets the barrier penalty to 0.0 in Adam loss. Relies purely on post-Adam hard box clamping (torch.clamp) and downstream joint projection to enforce bounds.",
            "include_model_objective": True,
            "include_dietary_change_penalty": True,
            "include_barrier_penalty": False,
            "include_joint_projection": True,
            "include_sequential_filters": True,
            "category": "candidate_modifier",
        },
        "variant_4_no_joint_projection": {
            "variant_id": "variant_4_no_joint_projection",
            "name": "Ablate Joint Constraint Projection",
            "description": "Bypasses the convex quadratic joint projection onto the feasible polytope. Post-Adam clamped candidate (n_adam_clipped) is evaluated directly by downstream filters and the external validator. Directly quantifies joint numerical feasibility of unprojected Adam outputs.",
            "include_model_objective": True,
            "include_dietary_change_penalty": True,
            "include_barrier_penalty": True,
            "include_joint_projection": False,
            "include_sequential_filters": True,
            "category": "candidate_modifier",
        },
        "variant_5_no_sequential_filters": {
            "variant_id": "variant_5_no_sequential_filters",
            "name": "Ablate Sequential Downstream Filters 1-3",
            "description": "Retains full Adam optimization and joint projection, but bypasses sequential execution of Filter 1, Filter 2, and Filter 3 prior to closed-loop rescoring. Verifies the code-path finding that sequential filters are mathematically redundant when preceded by joint projection.",
            "include_model_objective": True,
            "include_dietary_change_penalty": True,
            "include_barrier_penalty": True,
            "include_joint_projection": True,
            "include_sequential_filters": False,
            "category": "candidate_modifier",
        },
    }
    variant_manifest_path = OUT_DIR / "variant_manifest.json"
    variant_manifest_path.write_text(json.dumps(variants, indent=2), encoding="utf-8")
    print(f"Saved variant manifest to {variant_manifest_path}")

    # 8. Build input manifest
    tracked_files = [
        ROOT / "stage2_counterfactual_engine.py",
        ROOT / "stage2_comparison_arms_v2.py",
        ROOT / "stage2_comparison_checks_v2.py",
        ROOT / "stage1_deployment.py",
        ROOT / "nutrient_rule_policy.py",
        ROOT / "train_bionutrigpt_stage1.py",
        clean_path,
        raw_path,
        dep.CHECKPOINT,
        dep.PREPROCESSOR,
        ROOT / "Saved Model/deployment/repaired_stage1_v1/calibrators.joblib",
        ROOT / "Saved Model/deployment/repaired_stage1_v1/manifest.json",
        repair_path,
        prior_ds_path,
        pilot_path,
        comp_path,
        budget_path,
        pilot_cases_path,
        eval_cases_path,
        variant_manifest_path,
    ]
    input_manifest = {
        "status": "frozen_inputs_audited",
        "timestamp": "2026-11-24T15:30:00Z",
        "file_count": len(tracked_files),
        "files": {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in tracked_files},
    }
    input_manifest_path = OUT_DIR / "input_manifest.json"
    input_manifest_path.write_text(json.dumps(input_manifest, indent=2), encoding="utf-8")
    print(f"Saved input manifest ({len(tracked_files)} files) to {input_manifest_path}")

    # 9. Build protocol.json
    protocol = {
        "protocol_version": "stage2_ablation_v1",
        "status": "protocol_frozen_pilot_prepared_full_study_unexecuted",
        "date": "2026-11-24",
        "purpose": "Controlled component ablation of BioNutriGPT Stage 2 counterfactual optimization engine to determine the engineering value of model-score objective, dietary-change penalty, barrier penalty, joint projection, and sequential filters.",
        "sample": {
            "clean_temporal_records": len(clean),
            "eligible_records": int(eligible.sum()),
            "prior_study_exclusions": len(all_prior),
            "disjoint_eligible_pool": len(pool),
            "pilot_sample_size": len(pilot_cases),
            "pilot_seed": 20261124,
            "pilot_seqns": [c["SEQN"] for c in pilot_cases],
            "evaluation_sample_size": len(eval_cases),
            "evaluation_seed": 20261125,
            "evaluation_seqns_sha256": sha(eval_cases_path),
        },
        "variants": list(variants.keys()),
        "fixed_settings": {
            "adam_steps": 50,
            "adam_lr": 0.05,
            "dietary_change_penalty_weight": 0.001,
            "barrier_penalty_weight": 10.0,
            "pareto_penalty_weight": 50.0,
            "model_weights": {"diabetes": 2.0, "obesity": 1.0, "cvd": 2.0},
            "routing_cutoffs": {"diabetes": 0.5, "obesity": 0.5, "cvd": 0.5},
            "score_gate_tolerance": 0.0015,
            "joint_tolerance": 1e-6,
            "energy_margin": 0.0099,
            "reference_scales": "DAILY_REFERENCE_SCALES",
            "model_binding": s2.model_binding(dep.CHECKPOINT, dep.PREPROCESSOR),
        },
        "endpoints": {
            "primary": [
                "numerical_pass_rate",
                "score_gate_pass_rate",
                "overall_accepted_rate",
                "l1_normalized_distance_from_observed",
            ],
            "secondary": [
                "risk_probability_deltas",
                "filter_modification_counts",
                "slsqp_convergence_vs_fallback",
                "runtime_ms",
            ],
        },
        "statistical_specification": {
            "unit_of_inference": "participant",
            "full_sample_n": 100,
            "pilot_sample_n": 3,
            "bootstrap_replicates": 10000,
            "bootstrap_seed": 20261124,
            "multiplicity_family": [
                "variant_1_no_model_objective_vs_ref",
                "variant_2_no_change_penalty_vs_ref",
                "variant_3_no_barrier_penalty_vs_ref",
                "variant_4_no_joint_projection_vs_ref",
                "variant_5_no_sequential_filters_vs_ref",
            ],
            "family_size": 5,
            "multiplicity_adjustment": "Holm-Bonferroni step-down",
            "exact_tests": "two_sided_paired_binomial",
            "disclosure": "Pilot statistics are engineering checks, not population evidence.",
        },
        "reproducibility": {
            "input_manifest_sha256": sha(input_manifest_path),
            "variant_manifest_sha256": sha(variant_manifest_path),
            "pilot_manifest_sha256": sha(pilot_cases_path),
            "eval_manifest_sha256": sha(eval_cases_path),
        },
    }
    protocol_json_path = OUT_DIR / "protocol.json"
    protocol_json_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(f"Saved frozen protocol to {protocol_json_path}")


if __name__ == "__main__":
    main()
