#!/usr/bin/env python3
"""Freeze protocol for Stage 3 Controlled Solver-Budget Experiment.

Evaluates Variant A (current bounds-based) vs Variant B (target-aware)
across 100 participants x 3 Stage 2 arms x 3 dietary modes x 3 budgets (3.0s, 5.0s, 10.0s).
Total solves: 5,400 (2,700 paired tasks).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import stage3_budget_solver as rbs

STUDY_NAME = "stage3_budget_experiment_v1"
STUDY_DIR = ROOT / "Saved Model" / STUDY_NAME
CHECKPOINT_DIR = STUDY_DIR / "checkpoints"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    cases_path = STUDY_DIR / "cases_LOCAL.json"
    assert cases_path.exists(), f"Missing cohort manifest: {cases_path}"
    catalog_path = ROOT / "food_composition_cooked_sourced.json"
    assert catalog_path.exists(), f"Missing food catalog: {catalog_path}"
    catalog_hash = sha(catalog_path)

    bound_sources = [
        "food_composition_cooked_sourced.json",
        "stage3_milp_solver.py",
        "stage3_target_aware_solver.py",
        "stage3_budget_solver.py",
        "test_stage3_budget_solver.py",
        "stage2_stage3_adapter.py",
        "stage2_counterfactual_engine.py",
        "stage2_comparison_arms_v2.py",
        "nutrient_rule_policy.py",
        "stage1_deployment.py",
        "sample_stage3_budget_cohort_v1.py",
        "generate_stage3_budget_stage2_targets.py",
        "run_stage3_budget_experiment_v1.py",
        "Saved Model/stage3_budget_experiment_v1/cases_LOCAL.json",
        "Saved Model/stage3_budget_experiment_v1/stage2_targets/stage2_targets_summary.json",
    ]

    source_hashes = {}
    for rel in bound_sources:
        p = ROOT / rel
        assert p.exists(), f"Missing source: {p}"
        source_hashes[rel] = sha(p)

    protocol = {
        "protocol_version": STUDY_NAME,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "research_question": "Can additional computation recover the target-aware optimizer's valid-plan rate while retaining its lower nutrient-target error?",
        "cohort_size": 100,
        "cohort_exclusions": {
            "description": "Strictly disjoint from 32 repair cases, 100 downstream study cases, 5 pilot cases, and 100 comparison study cases",
            "unique_excluded_seqns": 237,
            "sample_random_state": 20261124,
            "cases_sha256": sha(cases_path),
        },
        "catalog_hash": catalog_hash,
        "arms": ["rules_only", "minimum_change", "model_guided"],
        "dietary_modes": ["pure_vegetarian", "veg_nonveg_combo", "pure_non_vegetarian"],
        "solver_variants": {
            "variant_a": "current (bounds-based: calorie deviation + food selection cost)",
            "variant_b": "target_aware (calorie deviation + food selection cost + normalized L1 target deviations)",
        },
        "budgets_seconds": [3.0, 5.0, 10.0],
        "total_planned_paired_tasks": 2700,
        "total_solves": 5400,
        "target_weight": 1.0,
        "research_max_time_limit_sec": rbs.RESEARCH_MAX_TIME_LIMIT_SEC,
        "normalization_scales": rbs.DAILY_REFERENCE_SCALES,
        "nutrient_units": rbs.NUTRIENT_UNITS,
        "zero_slack_invariant": "Bound slack upper bounds fixed strictly at 0.0 for all active nutrients in both variants",
        "discrete_polytope_invariance": "Proj_{(q,z)}(X_B) == X_A: Hard bounds, category rules, dish limits, integrality are mathematically identical",
        "cold_starts": "Each solve executes independently with fresh HiGHS instance; no warm-starting across budgets",
        "statistical_testing": {
            "binary_discordance_test": "Exact two-sided paired binomial test per cell (arm x mode x budget)",
            "cluster_bootstrap": "Participant-clustered bootstrap (10,000 resamples, seed 20261124) for paired risk differences and target error",
            "target_error_contrast": "Mean Normalized L1 Target Error (NTE_L1) strictly conditioned on jointly successful pairs with equal participant weighting",
            "multiplicity_adjustment": "Holm step-down procedure across the 18 primary comparisons",
        },
        "hardware": {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
            "torch_threads": torch.get_num_threads(),
        },
        "hashes": source_hashes,
    }

    protocol_path = STUDY_DIR / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    md_lines = [
        f"# Protocol: {STUDY_NAME}",
        "",
        "## Research Question",
        "**Can additional computation recover the target-aware optimizer's valid-plan rate while retaining its lower nutrient-target error?**",
        "",
        "## Study Architecture & Factorial Dimensions",
        "- **Cohort**: 100 adult temporal NHANES participants (strictly disjoint from 32-case repair, 100-case downstream, 5-case pilot, and 100-case comparison cohorts; total 237 exclusions).",
        "- **Stage 2 Target Sources (3 arms)**: Rules-Only (RO), Minimum-Change (MC), Model-Guided (MG).",
        "- **Dietary Modes (3 modes)**: Pure Vegetarian, Veg/Non-Veg Combo, Pure Non-Vegetarian.",
        "- **Solver Objectives (2 variants)**:",
        "  - **Variant A (Current Bounds-Based)**: $\\min \\text{Cost}_{\\text{selection}} + \\lambda_{\\text{cal}} |\\text{kcal} - T_{\\text{kcal}}| + M \\cdot \\text{slack}$ (with slack clamped to 0).",
        "  - **Variant B (Target-Aware)**: $\\min \\text{Cost}_{\\text{selection}} + \\lambda_{\\text{cal}} |\\text{kcal} - T_{\\text{kcal}}| + \\lambda_{\\text{target}} \\sum_k \\frac{|y_k - T_k|}{s_k}$ (with slack clamped to 0).",
        "- **Time Budgets (3 levels)**: 3.0 seconds, 5.0 seconds, 10.0 seconds.",
        "- **Total Paired Tasks**: 100 participants $\\times$ 3 arms $\\times$ 3 dietary modes $\\times$ 3 budgets = **2,700 paired tasks**.",
        "- **Total Optimization Solves**: $2,700 \\times 2 = \\mathbf{5,400}$ individual solves.",
        "",
        "## Mathematical Invariants & Implementation Safeguards",
        "1. **Un-clamped Research Wrapper**: `stage3_budget_solver.py` enforces `RESEARCH_MAX_TIME_LIMIT_SEC = 15.0`, passing exactly 3.0s, 5.0s, and 10.0s directly to HiGHS.",
        "2. **Production Baseline Untouched**: `stage3_milp_solver.py` default limits (1.5s default, 3.0s max clamp) remain untouched for clinical callers.",
        "3. **Zero Active Slack**: Component bound slack upper bounds are fixed strictly at 0.0 for all active nutrients.",
        "4. **Identical Feasible Polytope**: Discrete decision variables $(q, z)$, bounds, category constraints, and integrality are identical between Variant A and Variant B.",
        "5. **Equal Participant Weighting**: Statistical contrasts weight each participant equally by averaging within-participant cells first.",
        "6. **Jointly Successful Conditioning**: Target error comparisons and food basket comparisons are strictly conditioned on pairs where both variants found valid plans.",
        "7. **Cryptographic Binding**: All code, manifest, targets, and catalog digests are frozen in `protocol.json` prior to execution.",
    ]
    (STUDY_DIR / "Protocol.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Protocol successfully frozen at {protocol_path}")
    print(f"Bound sources: {len(source_hashes)}")
    print(f"Cohort SHA-256: {sha(cases_path)}")


if __name__ == "__main__":
    main()
