#!/usr/bin/env python3
"""Independent Reconstructive Audit for Stage 3 Controlled Solver-Budget Experiment.

Recalculates every single meal plan decision from raw catalog values without calling
any solver and without trusting saved acceptance or replay flags.

Covers:
1. Manifest of all original evidence files (SHA-256 bound).
2. Protocol, source file, cohort, and food catalog hash integrity checks.
3. Cold-start reconstructive verification for all 2,700 checkpoints (5,400 solves):
   - Discrete serving quantities q in {1, 2, 3}
   - Active dish counts 4 <= k <= 8
   - Food database lookup, non-duplicates, serving weight accounting
   - Category limits (variety min/max, servings min/max)
   - Dietary exclusions (pure vegetarian dish purity)
   - Nutrient totals summed from scratch from raw catalog values
   - Unknown nutrient tracking (unknown active nutrients cause rejection)
   - Zero-slack nutrient constraint envelope bounds (+/- 1e-6 tolerance)
   - Calorie tolerance (+/- 5% + 1e-9)
   - Acceptable Macronutrient Distribution Ranges (AMDR):
     Protein 10-35%, Carbohydrate 45-65%, Total Fat 20-35%
   - Sodium-to-Potassium ratio preservation (<= baseline + 1e-9)
   - Fiber density (>= 14 g / 1000 kcal - 1e-9)
   - Delivered nutrition accounting residual <= 1e-4
   - Attainment metrics recomputation: mean normalized L1 target error (NTE_L1)
     and normalized L2 target distance (NTD_L2) using official FDA scales
4. Verification of effective time limits without silent wrapper clamping.
5. Five deliberate corruption tests on synthetic copies (invalid quantity,
   corrupted totals, corrupted bound, saved flag not trusted, dietary violation).
6. Strict HiGHS solver status classification:
   - Status 0: integer optimality proven under MIP gap <= 1%
   - Status 1: time limit reached with valid plan
   - Status 1: time limit reached without valid plan
   - Status 2: proven infeasibility
7. Matched-mode statistical contrasts with equal participant weighting:
   - Acceptance rates, discordant pairs, exact two-sided paired binomial (McNemar) tests
   - Participant-clustered bootstrap 95% CIs (10,000 resamples, seed 20261124)
   - Attainment contrasts on jointly accepted plans
   - Compositional basket diversity (identical vs distinct food selections)
   - Cell-level breakdown (27 cells) and Holm step-down multiplicity adjustment.
8. Output artifacts:
   - Saved Model/stage3_budget_audit_v2/input_manifest.json
   - Saved Model/stage3_budget_audit_v2/recomputed_LOCAL.json
   - Saved Model/stage3_budget_audit_v2/analysis_summary.json
   - Saved Model/stage3_budget_audit_v2/replay_verification.json
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import scipy.stats as stats

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import stage3_milp_solver as s3
import nutrient_rule_policy as nrp
import repair_translation_evidence_v2 as base
from run_stage3_budget_experiment_v1 import compute_budget_experiment_fingerprint

EXP_DIR = ROOT / "Saved Model" / "stage3_budget_experiment_v1"
AUDIT_DIR = ROOT / "Saved Model" / "stage3_budget_audit_v2"
CHECKPOINTS_DIR = EXP_DIR / "checkpoints"
TARGETS_DIR = EXP_DIR / "stage2_targets"

BUDGETS_SEC = [3.0, 5.0, 10.0]
ARMS = ["rules_only", "minimum_change", "model_guided"]
MODES = ["pure_vegetarian", "veg_nonveg_combo", "pure_non_vegetarian"]
VARIANTS = ["variant_a", "variant_b"]
BOOTSTRAP_SEED = 20261124
BOOTSTRAP_RESAMPLES = 10000


def sha(p: Path | str) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def participant_cluster_bootstrap(
    diffs_per_participant: list[float],
    n_bootstrap: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Compute participant-clustered mean and 95% percentile confidence interval."""
    arr = np.asarray(diffs_per_participant, dtype=float)
    point_est = float(np.mean(arr))
    n = len(arr)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    resampled_means = np.mean(arr[indices], axis=1)
    ci_lower = float(np.percentile(resampled_means, 2.5))
    ci_upper = float(np.percentile(resampled_means, 97.5))
    return {
        "n_participants": n,
        "mean_difference": point_est,
        "ci95": [ci_lower, ci_upper],
    }


def exact_paired_binomial_test(n_discordant_a: int, n_discordant_b: int) -> float:
    """Exact two-sided paired binomial (McNemar) test on discordant pairs."""
    n_disc = n_discordant_a + n_discordant_b
    if n_disc == 0:
        return 1.0
    k_min = min(n_discordant_a, n_discordant_b)
    p_val = float(stats.binomtest(k_min, n_disc, p=0.5, alternative="two-sided").pvalue)
    return p_val


def holm_adjust(p_values: list[float]) -> list[float]:
    """Apply Holm step-down procedure to a list of p-values."""
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    cum_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = p * (m - rank)
        cum_max = max(cum_max, adj)
        adjusted[orig_idx] = min(1.0, cum_max)
    return adjusted


def main():
    print("=" * 80)
    print("Stage 3 Solver-Budget Experiment: Independent Reconstructive Audit v2")
    print("=" * 80)

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Manifest input evidence files
    print("\n[Step 1] Manifesting original evidence files...")
    all_json_files = sorted(EXP_DIR.rglob("*.json"))
    print(f"  Found {len(all_json_files)} input JSON files under {EXP_DIR}.")
    current_manifest = {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in all_json_files}

    manifest_path = AUDIT_DIR / "input_manifest.json"
    if manifest_path.exists():
        existing_manifest = read_json(manifest_path)
        assert existing_manifest == current_manifest, "Input evidence changed since first audit!"
        print("  Existing input manifest verified against current disk bytes.")
    else:
        save_json(manifest_path, current_manifest)
        print(f"  Saved input manifest ({len(current_manifest)} files).")

    # 2. Check protocol and source file hash integrity
    print("\n[Step 2] Auditing protocol and frozen source hashes...")
    protocol_path = EXP_DIR / "protocol.json"
    assert protocol_path.exists(), f"Missing protocol: {protocol_path}"
    protocol = read_json(protocol_path)

    for rel_path, expected_hash in protocol["hashes"].items():
        actual_path = ROOT / rel_path
        assert actual_path.exists(), f"Bound source file missing: {rel_path}"
        actual_hash = sha(actual_path)
        assert actual_hash == expected_hash, f"Source hash mismatch in {rel_path}!"
    print(f"  All {len(protocol['hashes'])} bound source files verified cryptographically.")

    cases_path = EXP_DIR / "cases_LOCAL.json"
    assert cases_path.exists(), f"Missing cohort manifest: {cases_path}"
    assert sha(cases_path) == protocol["cohort_exclusions"]["cases_sha256"], "Cohort hash mismatch!"
    cases = read_json(cases_path)
    assert len(cases) == 100, f"Expected 100 cases, got {len(cases)}"
    print(f"  Cohort manifest verified: 100 participants (SHA-256: {protocol['cohort_exclusions']['cases_sha256'][:16]}...).")

    catalog_path = ROOT / "food_composition_cooked_sourced.json"
    assert catalog_path.exists(), f"Missing catalog: {catalog_path}"
    catalog_hash = sha(catalog_path)
    assert catalog_hash == protocol["catalog_hash"], "Food catalog hash mismatch!"
    foods = s3.load_food_database()
    assert len(foods) == 49, f"Expected 49 food items, got {len(foods)}"
    print(f"  Food catalog verified: 49 items (SHA-256: {catalog_hash[:16]}...).")

    scales = protocol["normalization_scales"]

    # 3. Load Stage 2 Target Envelopes
    print("\n[Step 3] Loading Stage 2 target envelopes across 100 cases x 3 arms...")
    stage2_cache = {}
    for case in cases:
        c_idx = case["case"]
        seqn = case["SEQN"]
        meta = case["metadata"]
        for arm in ARMS:
            tgt_path = TARGETS_DIR / f"case_{c_idx:03d}_{seqn}_{arm}.json"
            assert tgt_path.exists(), f"Missing target file: {tgt_path}"
            tgt = read_json(tgt_path)
            s2_acc = bool(tgt["overall_accepted"])

            envelope = []
            for j, code in enumerate(s3.NUTRIENT_CODES):
                envelope.append({
                    "nutrient": code,
                    "target": float(tgt["target_nutrients"][j]),
                    "min_bound": float(tgt["lower_bounds"][j]),
                    "max_bound": float(tgt["upper_bounds"][j]),
                    "baseline": float(tgt["observed_nutrients"][j]),
                    "rule_provenance": nrp.reviewed_rule_metadata(
                        code, float(tgt["lower_bounds"][j]), float(tgt["upper_bounds"][j])
                    ),
                })

            s3_inp = {
                "patient_metadata": dict(meta),
                "energy_prescription": dict(tgt["energy_prescription"]),
                "nutrient_constraint_envelope": envelope,
            }
            targets_dict = {row["nutrient"]: row["target"] for row in envelope}
            stage2_cache[(c_idx, arm)] = {
                "input": s3_inp,
                "targets": targets_dict,
                "s2_accepted": s2_acc,
            }
    print(f"  Cached {len(stage2_cache)} Stage 2 target inputs.")

    # 4. Audit all 2,700 checkpoints (5,400 solves)
    print("\n[Step 4] Auditing all 2,700 checkpoints from raw bytes (5,400 solves)...")
    cp_files = sorted(CHECKPOINTS_DIR.glob("task_*.json"))
    assert len(cp_files) == 2700, f"Expected 2,700 checkpoints, found {len(cp_files)}"

    recomputed_rows = []
    corruption_tests = []
    max_accounting_residual = 0.0
    max_l1_residual = 0.0
    max_l2_residual = 0.0

    solver_statuses = {
        b: {
            "variant_a": {"optimal_gap_le_1pct": 0, "timeout_valid_plan": 0, "timeout_no_valid_plan": 0, "infeasible": 0},
            "variant_b": {"optimal_gap_le_1pct": 0, "timeout_valid_plan": 0, "timeout_no_valid_plan": 0, "infeasible": 0},
        }
        for b in BUDGETS_SEC
    }

    t0_audit = time.perf_counter()

    for idx, cp_file in enumerate(cp_files):
        cp_data = read_json(cp_file)
        c_idx = cp_data["case_idx"]
        seqn = cp_data["SEQN"]
        arm = cp_data["arm"]
        mode = cp_data["dietary_mode"]
        budget_sec = float(cp_data["budget_seconds"])

        s2_info = stage2_cache[(c_idx, arm)]
        s3_inp = s2_info["input"]
        targets = s2_info["targets"]
        s2_acc = s2_info["s2_accepted"]

        # Fingerprint verification
        expected_fp = compute_budget_experiment_fingerprint(
            case_idx=c_idx,
            seqn=seqn,
            arm=arm,
            dietary_mode=mode,
            budget_sec=budget_sec,
            stage2_input=s3_inp,
            catalog_hash=catalog_hash,
            target_weight=1.0,
            protocol_version="stage3_budget_experiment_v1",
        )
        assert cp_data["fingerprint"] == expected_fp, f"Fingerprint mismatch in {cp_file.name}!"
        assert cp_data["stage2_accepted"] == s2_acc, f"Stage 2 acceptance mismatch in {cp_file.name}!"

        # Audit both Variant A and Variant B
        solves_audited = {}
        for v in VARIANTS:
            z = cp_data[v]

            # Verify time limit preservation
            eff_limit = z.get("effective_time_limit_sec")
            assert eff_limit == budget_sec, f"{cp_file.name} {v}: effective limit {eff_limit} != {budget_sec}"

            # Independent reconstructive basket audit from scratch
            audit_res = base.audit_basket({**z, "dietary_mode": mode}, s3_inp, foods)
            recomputed_passed = audit_res["passed"]

            # Assert complete fidelity with saved flags (without trusting them)
            assert recomputed_passed == z["accepted"] == z["replay_passed"], (
                f"Acceptance disagreement in {cp_file.name} {v}: audit={recomputed_passed}, "
                f"accepted={z['accepted']}, replay={z['replay_passed']}, errors={audit_res['errors']}"
            )

            e2e_accepted = bool(s2_acc and recomputed_passed)
            assert e2e_accepted == z["end_to_end_accepted"], (
                f"End-to-end acceptance disagreement in {cp_file.name} {v}"
            )

            max_accounting_residual = max(max_accounting_residual, audit_res["max_accounting_residual"])
            assert audit_res["max_accounting_residual"] <= 1e-4, (
                f"Accounting residual exceeded in {cp_file.name} {v}: {audit_res['max_accounting_residual']}"
            )

            # Recompute target attainment metrics if plan is accepted
            l1_val = None
            l2_val = None
            target_codes = [k for k in audit_res["active_codes"] if k != "DR1TKCAL"]

            if recomputed_passed:
                per_nut_err = {k: abs(audit_res["totals"][k] - targets[k]) for k in target_codes}
                l1_val = float(np.mean([per_nut_err[k] / scales[k] for k in target_codes]))
                l2_val = float(math.sqrt(sum(((per_nut_err[k] / scales[k]) ** 2) for k in target_codes)))

                saved_attain = z.get("target_attainment", {})
                assert len(target_codes) == saved_attain["target_nutrient_count"]

                l1_diff = abs(l1_val - saved_attain["mean_normalized_l1_target_error"])
                l2_diff = abs(l2_val - saved_attain["normalized_l2_target_distance"])
                max_l1_residual = max(max_l1_residual, l1_diff)
                max_l2_residual = max(max_l2_residual, l2_diff)
                assert l1_diff < 2e-6, f"L1 error mismatch in {cp_file.name} {v}: {l1_diff}"
                assert l2_diff < 2e-6, f"L2 error mismatch in {cp_file.name} {v}: {l2_diff}"

                for k in target_codes:
                    k_diff = abs(per_nut_err[k] - saved_attain["per_nutrient_absolute_error"][k])
                    assert k_diff < 2e-6, f"Per-nutrient error mismatch in {cp_file.name} {v} {k}: {k_diff}"

            # Deliberate corruption checks on synthetic copies (run once on first accepted plan)
            if recomputed_passed and not corruption_tests:
                # 1. Invalid quantity
                bad_q = copy.deepcopy(z)
                bad_q["food_basket"][0]["quantity"] = 4
                assert not base.audit_basket({**bad_q, "dietary_mode": mode}, s3_inp, foods)["passed"]
                corruption_tests.append("invalid_quantity_rejected")

                # 2. Corrupted totals
                bad_tot = copy.deepcopy(z)
                bad_tot["delivered_nutrition"][0]["delivered"] += 10.0
                assert not base.audit_basket({**bad_tot, "dietary_mode": mode}, s3_inp, foods)["passed"]
                corruption_tests.append("corrupted_totals_rejected")

                # 3. Corrupted bound
                bad_inp = copy.deepcopy(s3_inp)
                bad_inp["nutrient_constraint_envelope"][0]["max_bound"] = 0.001
                assert not base.audit_basket({**z, "dietary_mode": mode}, bad_inp, foods)["passed"]
                corruption_tests.append("corrupted_bound_rejected")

                # 4. Saved flag not trusted
                bad_flag = copy.deepcopy(z)
                bad_flag["accepted"] = False
                bad_flag["replay_passed"] = False
                # audit_basket must evaluate to passed regardless of what accepted was set to
                assert base.audit_basket({**bad_flag, "dietary_mode": mode}, s3_inp, foods)["passed"]
                corruption_tests.append("saved_flag_untrusted_verified")

                # 5. Dietary violation (non-veg in vegetarian mode)
                meat_item = next(f for f in foods if f.dietary_type == "non_vegetarian")
                bad_diet = copy.deepcopy(z)
                bad_diet["food_basket"].append({
                    "food_id": meat_item.food_id,
                    "quantity": 1,
                    "name": meat_item.name,
                    "category": meat_item.category,
                    "serving_weight_g": meat_item.serving_weight_g,
                })
                assert not base.audit_basket({**bad_diet, "dietary_mode": "pure_vegetarian"}, s3_inp, foods)["passed"]
                corruption_tests.append("dietary_violation_rejected")

            # Solver status classification
            raw_code = z.get("solver_status_code")
            gap = z.get("solver_gap")
            if raw_code == 0 or (gap is not None and gap <= 0.01 and recomputed_passed):
                solver_statuses[budget_sec][v]["optimal_gap_le_1pct"] += 1
            elif raw_code == 1 and recomputed_passed:
                solver_statuses[budget_sec][v]["timeout_valid_plan"] += 1
            elif raw_code == 1 and not recomputed_passed:
                solver_statuses[budget_sec][v]["timeout_no_valid_plan"] += 1
            elif raw_code == 2:
                solver_statuses[budget_sec][v]["infeasible"] += 1
            else:
                raise ValueError(f"Unclassified solver status: code={raw_code}, accepted={recomputed_passed}")

            canonical_basket = sorted((x["food_id"], x["quantity"]) for x in z["food_basket"])
            solves_audited[v] = {
                "accepted": recomputed_passed,
                "e2e": e2e_accepted,
                "l1": l1_val,
                "l2": l2_val,
                "runtime": float(z["solve_time_seconds"]),
                "basket": canonical_basket,
                "dish_count": len(z["food_basket"]),
                "servings": sum(x["quantity"] for x in z["food_basket"]),
                "mass_g": audit_res["mass_g"],
                "totals": audit_res["totals"] if recomputed_passed else None,
                "solver_status_code": raw_code,
                "solver_gap": gap,
            }

        # Check paired metrics consistency
        pc = cp_data["paired_comparison"]
        joint_acc = bool(solves_audited["variant_a"]["accepted"] and solves_audited["variant_b"]["accepted"])
        assert pc["jointly_accepted"] == joint_acc
        assert pc["discordant_a_only"] == bool(solves_audited["variant_a"]["accepted"] and not solves_audited["variant_b"]["accepted"])
        assert pc["discordant_b_only"] == bool(not solves_audited["variant_a"]["accepted"] and solves_audited["variant_b"]["accepted"])
        assert pc["jointly_rejected"] == bool(not solves_audited["variant_a"]["accepted"] and not solves_audited["variant_b"]["accepted"])

        if joint_acc:
            is_ident = (solves_audited["variant_a"]["basket"] == solves_audited["variant_b"]["basket"])
            assert pc["basket_identical_if_joint"] == is_ident
        else:
            assert pc["basket_identical_if_joint"] is None

        # Record audited result
        for v in VARIANTS:
            recomputed_rows.append({
                "case": c_idx,
                "SEQN": seqn,
                "arm": arm,
                "mode": mode,
                "budget_seconds": budget_sec,
                "variant": v,
                "accepted": solves_audited[v]["accepted"],
                "e2e": solves_audited[v]["e2e"],
                "l1": solves_audited[v]["l1"],
                "l2": solves_audited[v]["l2"],
                "runtime": solves_audited[v]["runtime"],
                "dish_count": solves_audited[v]["dish_count"],
                "servings": solves_audited[v]["servings"],
                "mass_g": solves_audited[v]["mass_g"],
                "basket": solves_audited[v]["basket"],
                "totals": solves_audited[v]["totals"],
                "solver_status_code": solves_audited[v]["solver_status_code"],
                "solver_gap": solves_audited[v]["solver_gap"],
            })

    elapsed_audit = time.perf_counter() - t0_audit
    print(f"  Audited all 2,700 checkpoints (5,400 solves) in {elapsed_audit:.2f}s.")
    print(f"  All 5 corruption tests passed: {corruption_tests}.")
    print(f"  Max accounting residual: {max_accounting_residual:.2e} (<= 1e-4).")
    print(f"  Max target attainment residual: L1={max_l1_residual:.2e}, L2={max_l2_residual:.2e} (< 2e-6).")

    # Save recomputed_LOCAL.json
    save_json(AUDIT_DIR / "recomputed_LOCAL.json", recomputed_rows)
    print(f"  Saved recomputed_LOCAL.json ({len(recomputed_rows)} records).")

    # 5. Statistical Aggregation and Contrasts
    print("\n[Step 5] Recomputing statistical contrasts and budget responses...")
    idx_map = {(r["case"], r["arm"], r["mode"], r["budget_seconds"], r["variant"]): r for r in recomputed_rows}

    budget_analysis = {}
    cell_breakdown = {}
    all_cell_p_values = []
    cell_keys = []

    for b in BUDGETS_SEC:
        b_key = f"{int(b)}s"
        b_rows_a = [r for r in recomputed_rows if r["budget_seconds"] == b and r["variant"] == "variant_a"]
        b_rows_b = [r for r in recomputed_rows if r["budget_seconds"] == b and r["variant"] == "variant_b"]
        n_tasks = len(b_rows_a)
        assert n_tasks == 900

        a_acc = sum(r["accepted"] for r in b_rows_a)
        b_acc = sum(r["accepted"] for r in b_rows_b)
        a_e2e = sum(r["e2e"] for r in b_rows_a)
        b_e2e = sum(r["e2e"] for r in b_rows_b)

        # Discordance counts
        disc_a = 0
        disc_b = 0
        joint_acc = 0
        joint_rej = 0
        ident_baskets = 0
        diff_baskets = 0

        for i in range(100):
            for arm in ARMS:
                for mode in MODES:
                    ra = idx_map[(i, arm, mode, b, "variant_a")]
                    rb = idx_map[(i, arm, mode, b, "variant_b")]
                    if ra["accepted"] and rb["accepted"]:
                        joint_acc += 1
                        if ra["basket"] == rb["basket"]:
                            ident_baskets += 1
                        else:
                            diff_baskets += 1
                    elif ra["accepted"] and not rb["accepted"]:
                        disc_a += 1
                    elif not ra["accepted"] and rb["accepted"]:
                        disc_b += 1
                    else:
                        joint_rej += 1

        mcnemar_p = exact_paired_binomial_test(disc_a, disc_b)

        # Participant-clustered bootstrap for acceptance (9 cells per participant per budget)
        part_diffs_acc = []
        part_diffs_e2e = []
        for i in range(100):
            diff_acc = np.mean([
                float(idx_map[(i, arm, mode, b, "variant_b")]["accepted"]) -
                float(idx_map[(i, arm, mode, b, "variant_a")]["accepted"])
                for arm in ARMS for mode in MODES
            ]) * 100.0  # percentage points
            part_diffs_acc.append(diff_acc)

            diff_e2e = np.mean([
                float(idx_map[(i, arm, mode, b, "variant_b")]["e2e"]) -
                float(idx_map[(i, arm, mode, b, "variant_a")]["e2e"])
                for arm in ARMS for mode in MODES
            ]) * 100.0
            part_diffs_e2e.append(diff_e2e)

        boot_acc = participant_cluster_bootstrap(part_diffs_acc)
        boot_e2e = participant_cluster_bootstrap(part_diffs_e2e)

        # Attainment contrast on jointly accepted plans
        part_l1_a = []
        part_l1_b = []
        part_l1_diff = []
        joint_cells_count = 0

        for i in range(100):
            pairs = [
                (idx_map[(i, arm, mode, b, "variant_a")]["l1"], idx_map[(i, arm, mode, b, "variant_b")]["l1"])
                for arm in ARMS for mode in MODES
                if idx_map[(i, arm, mode, b, "variant_a")]["accepted"] and idx_map[(i, arm, mode, b, "variant_b")]["accepted"]
            ]
            if pairs:
                mean_a = float(np.mean([x for x, y in pairs]))
                mean_b = float(np.mean([y for x, y in pairs]))
                part_l1_a.append(mean_a)
                part_l1_b.append(mean_b)
                part_l1_diff.append(mean_b - mean_a)
                joint_cells_count += len(pairs)

        boot_l1_a = participant_cluster_bootstrap(part_l1_a)
        boot_l1_b = participant_cluster_bootstrap(part_l1_b)
        boot_l1_diff = participant_cluster_bootstrap(part_l1_diff)
        rel_reduc_pct = float((boot_l1_b["mean_difference"] / boot_l1_a["mean_difference"] - 1.0) * 100.0)

        # Runtime summaries
        runtimes_a = [r["runtime"] for r in b_rows_a]
        runtimes_b = [r["runtime"] for r in b_rows_b]

        budget_analysis[b_key] = {
            "budget_seconds": b,
            "total_tasks": n_tasks,
            "variant_a_accepted": a_acc,
            "variant_a_rate_pct": round(a_acc / n_tasks * 100.0, 4),
            "variant_b_accepted": b_acc,
            "variant_b_rate_pct": round(b_acc / n_tasks * 100.0, 4),
            "paired_discordance": {
                "variant_a_only": disc_a,
                "variant_b_only": disc_b,
                "jointly_accepted": joint_acc,
                "jointly_rejected": joint_rej,
                "exact_mcnemar_p": mcnemar_p,
            },
            "acceptance_difference_pp": boot_acc,
            "end_to_end_acceptance_difference_pp": boot_e2e,
            "target_attainment_joint": {
                "joint_cells": joint_cells_count,
                "variant_a_mean_l1": boot_l1_a,
                "variant_b_mean_l1": boot_l1_b,
                "difference_l1": boot_l1_diff,
                "relative_error_reduction_pct": rel_reduc_pct,
            },
            "food_basket_comparison_joint": {
                "identical_baskets": ident_baskets,
                "distinct_baskets": diff_baskets,
                "distinct_percentage": round(diff_baskets / joint_acc * 100.0, 2) if joint_acc else 0.0,
            },
            "solver_diagnostics": {
                "variant_a": solver_statuses[b]["variant_a"],
                "variant_b": solver_statuses[b]["variant_b"],
                "runtime_seconds": {
                    "variant_a": {
                        "median": float(np.median(runtimes_a)),
                        "mean": float(np.mean(runtimes_a)),
                        "std": float(np.std(runtimes_a)),
                    },
                    "variant_b": {
                        "median": float(np.median(runtimes_b)),
                        "mean": float(np.mean(runtimes_b)),
                        "std": float(np.std(runtimes_b)),
                    },
                },
            },
        }

        # Cell-level breakdown for this budget
        cell_breakdown[b_key] = {}
        for arm in ARMS:
            cell_breakdown[b_key][arm] = {}
            for mode in MODES:
                c_a_acc = sum(idx_map[(i, arm, mode, b, "variant_a")]["accepted"] for i in range(100))
                c_b_acc = sum(idx_map[(i, arm, mode, b, "variant_b")]["accepted"] for i in range(100))
                c_disc_a = sum(
                    idx_map[(i, arm, mode, b, "variant_a")]["accepted"] and not idx_map[(i, arm, mode, b, "variant_b")]["accepted"]
                    for i in range(100)
                )
                c_disc_b = sum(
                    not idx_map[(i, arm, mode, b, "variant_a")]["accepted"] and idx_map[(i, arm, mode, b, "variant_b")]["accepted"]
                    for i in range(100)
                )
                c_joint = sum(
                    idx_map[(i, arm, mode, b, "variant_a")]["accepted"] and idx_map[(i, arm, mode, b, "variant_b")]["accepted"]
                    for i in range(100)
                )
                c_p = exact_paired_binomial_test(c_disc_a, c_disc_b)

                c_diffs = [
                    (float(idx_map[(i, arm, mode, b, "variant_b")]["accepted"]) -
                     float(idx_map[(i, arm, mode, b, "variant_a")]["accepted"])) * 100.0
                    for i in range(100)
                ]
                c_boot = participant_cluster_bootstrap(c_diffs)

                cell_breakdown[b_key][arm][mode] = {
                    "n": 100,
                    "variant_a_accepted": c_a_acc,
                    "variant_b_accepted": c_b_acc,
                    "discordant_a_only": c_disc_a,
                    "discordant_b_only": c_disc_b,
                    "jointly_accepted": c_joint,
                    "diff_pp": c_boot["mean_difference"],
                    "ci_95": c_boot["ci95"],
                    "raw_p_value": c_p,
                }
                all_cell_p_values.append(c_p)
                cell_keys.append((b_key, arm, mode))

    # Apply Holm step-down adjustment across all 27 cells
    adjusted_cell_p = holm_adjust(all_cell_p_values)
    for (b_k, a_k, m_k), adj_p in zip(cell_keys, adjusted_cell_p):
        cell_breakdown[b_k][a_k][m_k]["holm_adjusted_p"] = adj_p

    # Construct final analysis object
    analysis_payload = {
        "study_name": "stage3_budget_experiment_v1",
        "audit_version": "stage3_budget_audit_v2",
        "n_participants": 100,
        "n_tasks": 2700,
        "n_solves": 5400,
        "budgets_evaluated": BUDGETS_SEC,
        "budget_summary": budget_analysis,
        "cell_breakdown": cell_breakdown,
        "corruption_tests_passed": corruption_tests,
        "max_accounting_residual": max_accounting_residual,
        "max_l1_reconstruction_residual": max_l1_residual,
        "max_l2_reconstruction_residual": max_l2_residual,
        "invariants_verified": {
            "solver_reruns": 0,
            "zero_slack_enforced": True,
            "unclamped_effective_limits": True,
            "reconstructive_basket_replay": True,
            "saved_acceptance_flags_ignored": True,
            "equal_participant_weighting": True,
        },
        "limitations": [
            "Sample is a temporal NHANES subsample and does not evaluate Indian clinical populations.",
            "Meal catalog contains 49 sourced Indian recipes; does not span full subcontinental regional dietary breadth.",
            "Solvers ran with fixed 1.0 target weight; weight sensitivity was not evaluated across continuous scales.",
            "Multiplicity adjustment is post-hoc over the 27 mode-by-arm-by-budget cells.",
        ],
    }

    save_json(AUDIT_DIR / "analysis_summary.json", analysis_payload)
    print("  Saved analysis_summary.json.")

    # 6. Re-verify input manifest after writing audit outputs
    print("\n[Step 6] Verifying input manifest integrity after audit execution...")
    after_manifest = {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in all_json_files}
    assert after_manifest == current_manifest, "Input files were modified during audit!"
    print("  Input files remain 100% byte-for-byte identical (zero corruption).")

    # 7. Write replay_verification.json receipt
    receipt = {
        "passed": True,
        "audit_version": "stage3_budget_audit_v2",
        "total_checkpoints_audited": len(cp_files),
        "total_solves_reconstructed": len(recomputed_rows),
        "corruption_tests_passed": len(corruption_tests),
        "max_accounting_residual": max_accounting_residual,
        "original_evidence_unchanged": True,
        "solver_calls": 0,
        "source_audit_hash": sha(Path(__file__)),
        "analysis_summary_sha256": sha(AUDIT_DIR / "analysis_summary.json"),
    }
    save_json(AUDIT_DIR / "replay_verification.json", receipt)
    print(f"  Saved replay_verification.json receipt: {receipt}")
    print("\n" + "=" * 80)
    print("AUDIT_STAGE3_BUDGET_V2 COMPLETED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    main()
