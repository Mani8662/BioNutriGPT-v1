#!/usr/bin/env python3
"""BioNutriGPT Experimental Stage 3: Target-Aware Discrete Meal Synthesis.

This module implements an experimental variant of Stage 3 that explicitly minimizes
deviation from Stage 2 continuous nutrient targets while strictly preserving all
existing hard constraints, category variety, plate rules, dietary exclusions,
portion integrality, and the zero-slack invariant on component bounds.

Architectural contract:
1. Feasible polytope X_MILP is identical between Variant A (current bounds-based)
   and Variant B (target-aware objective).
2. Variant B introduces continuous L1 deviation variables (d_k^+, d_k^-) satisfying:
      sum_i F_{k,i} q_i - d_k^+ + d_k^- = target_k
   for all eligible active nutrients k in K_target (excluding DR1TKCAL).
3. The objective minimizes:
      J = gamma * sum_i cost_i * z_i
        + (150 / T_kcal) * (cal_over + cal_under)
        + lambda_target * sum_{k in K_target} (w_k / s_k) * (d_k^+ + d_k^-)
4. Deviation variables cannot purchase violations of hard bounds [L_k, U_k].
5. Energy deviation is penalized once (via cal_over, cal_under) and not duplicated.
6. Report-only and exposure-only fields are excluded from target deviations.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, vstack

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import stage3_milp_solver as s3
import stage2_counterfactual_engine as s2
import nutrient_rule_policy as nrp

# Re-export necessary types and constants from production solver
FoodItem = s3.FoodItem
Stage3Input = s3.Stage3Input
Stage3Output = s3.Stage3Output
NUTRIENT_CODES = s3.NUTRIENT_CODES
NUTRIENT_INDEX = s3.NUTRIENT_INDEX
VALID_CATEGORIES = s3.VALID_CATEGORIES
MAX_SERVINGS_PER_DISH = s3.MAX_SERVINGS_PER_DISH
MIN_ACTIVE_DISHES = s3.MIN_ACTIVE_DISHES
MAX_ACTIVE_DISHES = s3.MAX_ACTIVE_DISHES
DEFAULT_TIME_LIMIT_SEC = 3.0
MAX_TIME_LIMIT_SEC = 5.0

# Explicit standard daily reference normalization scales (FDA Daily Values / RDAs)
DAILY_REFERENCE_SCALES: dict[str, float] = {
    "DR1TKCAL": 2000.0,
    "DR1TPROT": 50.0,
    "DR1TCARB": 275.0,
    "DR1TSUGR": 50.0,
    "DR1TFIBE": 28.0,
    "DR1TTFAT": 78.0,
    "DR1TSFAT": 20.0,
    "DR1TMFAT": 30.0,
    "DR1TPFAT": 20.0,
    "DR1TCHOL": 300.0,
    "DR1TATOC": 15.0,
    "DR1TRET": 900.0,
    "DR1TVARA": 900.0,
    "DR1TACAR": 1000.0,
    "DR1TBCAR": 3000.0,
    "DR1TLYCO": 5000.0,
    "DR1TLZ": 6000.0,
    "DR1TVB1": 1.2,
    "DR1TVB2": 1.3,
    "DR1TNIAC": 16.0,
    "DR1TVB6": 1.7,
    "DR1TFOLA": 400.0,
    "DR1TFA": 400.0,
    "DR1TFF": 400.0,
    "DR1TFDFE": 400.0,
    "DR1TVB12": 2.4,
    "DR1TVC": 90.0,
    "DR1TVK": 120.0,
    "DR1TCALC": 1300.0,
    "DR1TPHOS": 1250.0,
    "DR1TMAGN": 420.0,
    "DR1TIRON": 18.0,
    "DR1TZINC": 11.0,
    "DR1TCOPP": 0.9,
    "DR1TSODI": 2300.0,
    "DR1TPOTA": 4700.0,
    "DR1TSELE": 55.0,
    "DR1TCAFF": 200.0,
    "DR1TALCO": 14.0,
}

# Explicit physical units for all 39 dietary fields
NUTRIENT_UNITS: dict[str, str] = {
    "DR1TKCAL": "kcal",
    "DR1TPROT": "g", "DR1TCARB": "g", "DR1TSUGR": "g", "DR1TFIBE": "g",
    "DR1TTFAT": "g", "DR1TSFAT": "g", "DR1TMFAT": "g", "DR1TPFAT": "g",
    "DR1TCHOL": "mg", "DR1TATOC": "mg", "DR1TRET": "ug", "DR1TVARA": "ug",
    "DR1TACAR": "ug", "DR1TBCAR": "ug", "DR1TLYCO": "ug", "DR1TLZ": "ug",
    "DR1TVB1": "mg", "DR1TVB2": "mg", "DR1TNIAC": "mg", "DR1TVB6": "mg",
    "DR1TFOLA": "ug", "DR1TFA": "ug", "DR1TFF": "ug", "DR1TFDFE": "ug",
    "DR1TVB12": "ug", "DR1TVC": "mg", "DR1TVK": "ug", "DR1TCALC": "mg",
    "DR1TPHOS": "mg", "DR1TMAGN": "mg", "DR1TIRON": "mg", "DR1TZINC": "mg",
    "DR1TCOPP": "mg", "DR1TSODI": "mg", "DR1TPOTA": "mg", "DR1TSELE": "ug",
    "DR1TCAFF": "mg", "DR1TALCO": "g",
}

# Ensure all scales are strictly positive and finite
for code, scale in DAILY_REFERENCE_SCALES.items():
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid reference scale for {code}: {scale}")


def get_eligible_target_nutrients(
    stage: Stage3Input,
    eligible_foods: Sequence[FoodItem],
) -> list[str]:
    """Return active nutrients eligible for continuous target deviation penalization.
    
    Excludes:
    1. DR1TKCAL (energy is handled separately via cal_over, cal_under).
    2. report_only / exposure_reporting nutrients (caffeine, alcohol, carotenoids, etc.).
    3. Any nutrient missing in any eligible food.
    """
    target_nutrients: list[str] = []
    for code in NUTRIENT_CODES:
        if code == "DR1TKCAL":
            continue
        role = stage.rule_metadata.get(code, {}).get("planning_role")
        if role in {"report_only", "exposure_reporting"}:
            continue
        # Ensure complete coverage across eligible foods
        if any(code in f.unknown_nutrients for f in eligible_foods):
            continue
        target_nutrients.append(code)
    return target_nutrients


def compute_target_attainment_metrics(
    delivered_totals: Mapping[str, float],
    stage: Stage3Input,
    target_nutrients: Sequence[str],
) -> dict[str, Any]:
    """Compute normalized L1 error, normalized L2 distance, and per-nutrient errors."""
    per_nutrient_abs_err: dict[str, float] = {}
    per_nutrient_norm_l1: dict[str, float] = {}
    sum_norm_l1 = 0.0
    sum_norm_l2_sq = 0.0

    for code in target_nutrients:
        val = float(delivered_totals.get(code, 0.0))
        tgt = float(stage.targets[code])
        abs_err = abs(val - tgt)
        s = DAILY_REFERENCE_SCALES[code]
        norm_l1 = abs_err / s
        sum_norm_l1 += norm_l1
        sum_norm_l2_sq += (abs_err / s) ** 2
        per_nutrient_abs_err[code] = round(abs_err, 6)
        per_nutrient_norm_l1[code] = round(norm_l1, 6)

    k_count = len(target_nutrients)
    mean_norm_l1 = (sum_norm_l1 / k_count) if k_count > 0 else 0.0
    norm_l2_dist = math.sqrt(sum_norm_l2_sq)

    # Energy error
    t_kcal = float(stage.energy_prescription.get("target_calories", 2000.0))
    deliv_kcal = float(delivered_totals.get("DR1TKCAL", 0.0))
    energy_abs_err = abs(deliv_kcal - t_kcal)
    energy_pct_err = (energy_abs_err / t_kcal * 100.0) if t_kcal > 0 else 100.0

    return {
        "mean_normalized_l1_target_error": round(mean_norm_l1, 6),
        "total_normalized_l1_target_error": round(sum_norm_l1, 6),
        "normalized_l2_target_distance": round(norm_l2_dist, 6),
        "energy_absolute_error_kcal": round(energy_abs_err, 2),
        "energy_percentage_error": round(energy_pct_err, 4),
        "target_nutrient_count": k_count,
        "per_nutrient_absolute_error": per_nutrient_abs_err,
        "per_nutrient_normalized_l1_error": per_nutrient_norm_l1,
    }


def solve_experimental_meal_plan(
    stage2_input: dict,
    food_db: Sequence[FoodItem],
    time_limit_sec: float = DEFAULT_TIME_LIMIT_SEC,
    revision_spec: Mapping[str, Any] | None = None,
    objective_variant: str = "target_aware",
    target_weight: float = 1.0,
) -> Stage3Output:
    """Solve discrete meal plan under current (bounds-based) or target-aware objective.
    
    Parameters
    ----------
    stage2_input : dict
        Validated Stage 2 patient metadata, energy prescription, targets, bounds, and rule metadata.
    food_db : Sequence[FoodItem]
        Available food catalog.
    time_limit_sec : float
        HiGHS solver time limit in seconds.
    revision_spec : Mapping[str, Any] | None
        Optional revisions (dietary mode, exclusions, preferences, locks).
    objective_variant : str
        "current" (Variant A): Calorie deviation + food selection cost.
        "target_aware" (Variant B): Calorie deviation + food selection cost + normalized target deviations.
    target_weight : float
        Scalar multiplier lambda_target on normalized L1 target deviations (used when objective_variant="target_aware").
    
    Returns
    -------
    Stage3Output
        Completed meal plan or explicit failure output.
    """
    if objective_variant not in {"current", "bounds_based", "target_aware"}:
        raise ValueError(f"Unknown objective_variant: {objective_variant!r}. Expected 'current' or 'target_aware'.")
    if not math.isfinite(float(target_weight)) or target_weight < 0:
        raise ValueError(f"target_weight must be finite and non-negative, got {target_weight}")

    stage = s3._parse_stage2_input(stage2_input)
    if not math.isfinite(float(time_limit_sec)) or time_limit_sec <= 0:
        raise ValueError("time_limit_sec must be positive and finite.")
    effective = min(float(time_limit_sec), MAX_TIME_LIMIT_SEC)

    rev = dict(revision_spec or {})
    excluded_ids: set[str] = {str(x) for x in rev.get("excluded_food_ids", [])}
    excluded_ing_codes: set[str] = {str(x) for x in rev.get("excluded_ingredient_codes", [])}
    preferred_ids: set[str] = {str(x) for x in rev.get("preferred_food_ids", [])}
    locked_quantities: dict[str, int] = {str(k): int(v) for k, v in (rev.get("locked_food_quantities") or {}).items()}
    max_dishes_req = rev.get("max_active_dishes")
    max_dishes_cap = None if max_dishes_req is None else max(MIN_ACTIVE_DISHES, min(MAX_ACTIVE_DISHES, int(max_dishes_req)))
    prev_quantities: dict[str, int] = {str(k): int(v) for k, v in (rev.get("previous_quantities") or {}).items()}
    change_penalty = float(rev.get("change_penalty", 0.35))
    raw_mode = str(rev.get("dietary_mode") or "pure_vegetarian").strip().lower()
    dietary_mode = raw_mode if raw_mode in {"pure_vegetarian", "veg_nonveg_combo", "pure_non_vegetarian"} else "pure_vegetarian"

    base_meta = {
        "backend": "scipy.optimize.milp / HiGHS",
        "objective_variant": "current" if objective_variant in {"current", "bounds_based"} else "target_aware",
        "target_weight": float(target_weight) if objective_variant == "target_aware" else 0.0,
        "time_limit_sec": effective,
        "requested_time_limit_sec": float(time_limit_sec),
        "solve_time_ms": 0.0,
        "candidate_validation": [],
        "revision_controls": {
            "dietary_mode": dietary_mode,
            "excluded_food_ids": sorted(excluded_ids),
            "excluded_ingredient_codes": sorted(excluded_ing_codes),
            "preferred_food_ids": sorted(preferred_ids),
            "locked_food_quantities": locked_quantities,
            "max_active_dishes": max_dishes_cap,
            "change_penalty": change_penalty if prev_quantities else 0.0,
        },
    }

    # Conflict check: cannot both exclude and lock the same food
    conflict_lock = excluded_ids.intersection(locked_quantities)
    if conflict_lock:
        return s3._failure_output(
            "conflicting_revision_constraints",
            f"Requested revision both excludes and locks: {sorted(conflict_lock)}.",
            metadata=base_meta,
            details={"conflicting_food_ids": sorted(conflict_lock)},
        )

    required = {
        code for code in NUTRIENT_CODES
        if stage.rule_metadata.get(code, {}).get("planning_role") not in {"report_only", "exposure_reporting"}
    }
    effective_excluded = set(excluded_ids)
    if dietary_mode == "pure_vegetarian":
        for food in food_db:
            if food.dietary_type == "non_vegetarian":
                effective_excluded.add(food.food_id)
    if excluded_ing_codes:
        for food in food_db:
            if excluded_ing_codes.intersection(food.ingredient_codes):
                effective_excluded.add(food.food_id)

    eligible = [
        food for food in food_db
        if food.food_id not in effective_excluded and not required.intersection(food.unknown_nutrients)
    ]
    eligible_ids = {f.food_id for f in eligible}
    missing_locked = sorted(set(locked_quantities) - eligible_ids)
    if missing_locked:
        return s3._failure_output(
            "locked_food_unavailable",
            f"Locked food(s) are excluded or lack complete active data: {missing_locked}.",
            metadata=base_meta,
            details={"missing_locked_food_ids": missing_locked},
        )

    dairy_complete_in_db = [f for f in food_db if f.category == "dairy" and not required.intersection(f.unknown_nutrients)]
    allow_zero_dairy = bool(dairy_complete_in_db) and all(f.food_id in effective_excluded for f in dairy_complete_in_db)

    cat_limits, _ = s3._category_rules_for_mode(dietary_mode, allow_zero_dairy=allow_zero_dairy)
    required_cats = {cat for cat, (low, _) in cat_limits.items() if low > 0}
    missing_categories = sorted(required_cats - {food.category for food in eligible})
    if len(eligible) < MIN_ACTIVE_DISHES or missing_categories:
        return s3._failure_output(
            "insufficient_nutrient_data",
            "The catalog lacks sufficiently documented foods for all required meal categories. No plan was accepted.",
            metadata=base_meta,
            details={
                "missing_categories_with_complete_active_data": missing_categories,
                "eligible_food_count": len(eligible),
                "candidate_food_count": len(food_db),
                "unknown_active_nutrients": sorted({code for food in food_db for code in food.unknown_nutrients if code in required}),
                "unknown_catalog_nutrients": sorted({code for food in food_db for code in food.unknown_nutrients}),
                "excluded_incomplete_food_ids": [f.food_id for f in food_db if f not in eligible],
            },
        )

    s3._validate_catalog_categories(eligible, dietary_mode=dietary_mode, allow_zero_dairy=allow_zero_dairy)

    # Base MILP sparse constraints (identical between variants)
    c, integrality, bounds, constraints, info = s3._make_sparse_constraints(
        eligible, stage, gamma=0.10, dietary_mode=dietary_mode, allow_zero_dairy=allow_zero_dairy
    )
    lower = bounds.lb.copy()
    upper = bounds.ub.copy()
    m = len(eligible)
    q0 = info["indices"]["q"]
    z0 = info["indices"]["z"]

    # Zero-slack invariant on active component bounds (strictly preserved)
    for k, code in enumerate(NUTRIENT_CODES):
        if code in required:
            upper[info["indices"]["e_plus"] + k] = 0.0
            upper[info["indices"]["e_minus"] + k] = 0.0

    # Apply locked food quantities and user preferences
    for i, food in enumerate(eligible):
        if food.food_id in locked_quantities:
            lqty = int(locked_quantities[food.food_id])
            lower[q0 + i] = float(lqty)
            upper[q0 + i] = float(lqty)
            lower[z0 + i] = 1.0 if lqty > 0 else 0.0
            upper[z0 + i] = 1.0 if lqty > 0 else 0.0
        if food.food_id in preferred_ids:
            c[z0 + i] = max(0.0, c[z0 + i] - 0.08)

    # Clinical hard constraints: AMDR, Na:K ratio, fiber density
    F = info["food_matrix"]
    kcal = F[NUTRIENT_INDEX["DR1TKCAL"]]
    extra_rows, extra_lower, extra_upper = [], [], []
    for code, factor, low, high in (
        ("DR1TPROT", 4, 0.10, 0.35),
        ("DR1TCARB", 4, 0.45, 0.65),
        ("DR1TTFAT", 9, 0.20, 0.35),
    ):
        nutrient = F[NUTRIENT_INDEX[code]] * factor
        extra_rows.extend((low * kcal - nutrient, nutrient - high * kcal))
        extra_lower.extend((-np.inf, -np.inf))
        extra_upper.extend((0.0, 0.0))

    ratio = stage.baselines["DR1TSODI"] / (stage.baselines["DR1TPOTA"] + 1.0)
    extra_rows.append(F[NUTRIENT_INDEX["DR1TSODI"]] - ratio * F[NUTRIENT_INDEX["DR1TPOTA"]])
    extra_lower.append(-np.inf)
    extra_upper.append(ratio)

    extra_rows.append(0.014 * kcal - F[NUTRIENT_INDEX["DR1TFIBE"]])
    extra_lower.append(-np.inf)
    extra_upper.append(0.0)

    nbase = len(c)
    extra = np.zeros((len(extra_rows), nbase))
    extra[:, :m] = extra_rows

    if max_dishes_cap is not None:
        cap_row = np.zeros((1, nbase))
        cap_row[0, z0:z0 + m] = 1.0
        extra = np.vstack((extra, cap_row))
        extra_lower.append(float(MIN_ACTIVE_DISHES))
        extra_upper.append(float(max_dishes_cap))

    A = vstack((constraints.A, coo_matrix(extra)), format="csr")
    c_lb = np.r_[constraints.lb, np.asarray(extra_lower, dtype=float)]
    c_ub = np.r_[constraints.ub, np.asarray(extra_upper, dtype=float)]

    # Optional minimal-change penalty relative to previous_quantities
    if prev_quantities and change_penalty > 0:
        n_delta = 2 * m
        c = np.r_[c, np.full(n_delta, change_penalty, dtype=float)]
        integrality = np.r_[integrality, np.zeros(n_delta, dtype=np.int8)]
        lower = np.r_[lower, np.zeros(n_delta, dtype=float)]
        upper = np.r_[upper, np.full(n_delta, np.inf, dtype=float)]
        A_expanded = coo_matrix(np.hstack((A.toarray(), np.zeros((A.shape[0], n_delta)))))
        d_rows = np.zeros((m, nbase + n_delta), dtype=float)
        d_rhs = np.zeros(m, dtype=float)
        for i, food in enumerate(eligible):
            d_rows[i, q0 + i] = 1.0
            d_rows[i, nbase + i] = -1.0       # delta_plus
            d_rows[i, nbase + m + i] = 1.0    # delta_minus
            d_rhs[i] = float(prev_quantities.get(food.food_id, 0))
        A = vstack((A_expanded, coo_matrix(d_rows)), format="csr")
        c_lb = np.r_[c_lb, d_rhs]
        c_ub = np.r_[c_ub, d_rhs]
        nbase += n_delta

    # Target-Aware Variant B: Add L1 goal-programming deviation variables
    target_nutrients: list[str] = []
    if objective_variant == "target_aware" and target_weight > 0:
        target_nutrients = get_eligible_target_nutrients(stage, eligible)
        n_target = len(target_nutrients)
        n_dev = 2 * n_target  # d_plus and d_minus for each target nutrient

        dev_costs = []
        for code in target_nutrients:
            s_k = DAILY_REFERENCE_SCALES[code]
            w_k = 1.0  # Normalized L1 goal programming weight
            dev_costs.append(target_weight * w_k / s_k)

        # Extend c, integrality, lower, upper
        c = np.r_[c, np.asarray(dev_costs, dtype=float), np.asarray(dev_costs, dtype=float)]
        integrality = np.r_[integrality, np.zeros(n_dev, dtype=np.int8)]
        lower = np.r_[lower, np.zeros(n_dev, dtype=float)]
        upper = np.r_[upper, np.full(n_dev, np.inf, dtype=float)]

        # Expand existing constraint matrix columns
        A_padded = coo_matrix(np.hstack((A.toarray(), np.zeros((A.shape[0], n_dev)))))

        # Add target deviation rows: sum_i F_{k,i} q_i - d_j^+ + d_j^- = target_k
        dev_rows = np.zeros((n_target, nbase + n_dev), dtype=float)
        dev_rhs = np.zeros(n_target, dtype=float)

        for j, code in enumerate(target_nutrients):
            k = NUTRIENT_INDEX[code]
            dev_rows[j, :m] = F[k, :]
            dev_rows[j, nbase + j] = -1.0             # - d_plus
            dev_rows[j, nbase + n_target + j] = 1.0   # + d_minus
            dev_rhs[j] = float(stage.targets[code])

        A = vstack((A_padded, coo_matrix(dev_rows)), format="csr")
        c_lb = np.r_[c_lb, dev_rhs]
        c_ub = np.r_[c_ub, dev_rhs]

    linear_constraints = LinearConstraint(A, c_lb, c_ub)
    var_bounds = Bounds(lower, upper)

    started = time.perf_counter()
    try:
        result = milp(
            c=c,
            integrality=integrality,
            bounds=var_bounds,
            constraints=linear_constraints,
            options={"time_limit": effective, "mip_rel_gap": 0.01, "presolve": True, "disp": False},
        )
    except Exception as exc:
        return s3._failure_output(
            "solver_error",
            "The optimizer could not complete the search.",
            metadata=base_meta,
            details={"error": str(exc)},
        )

    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    base_meta.update(
        solve_time_ms=elapsed_ms,
        solver_status_code=int(result.status),
        status_message=str(result.message),
        solver_incumbent_available=result.x is not None,
        variable_count=len(c),
        constraint_count=A.shape[0],
    )

    candidates: list[tuple[str, np.ndarray]] = []
    if result.x is not None:
        values = np.asarray(result.x[:m])
        if np.isfinite(values).all() and np.max(np.abs(values - np.rint(values))) <= 1e-5:
            candidates.append(("solver", np.rint(values).astype(np.int64)))
    if not locked_quantities and max_dishes_cap is None:
        candidates.append(("heuristic", s3._heuristic_incumbent(eligible, stage.energy_prescription["target_calories"])))

    valid: list[tuple[float, str, Stage3Output, dict[str, Any], dict[str, float]]] = []
    for origin, q in candidates:
        candidate = s3._build_output_from_quantities(q, eligible, stage, {})
        validation = s3.validate_candidate(
            candidate,
            food_db,
            stage2_input,
            excluded_ids=sorted(effective_excluded),
            locked_quantities=locked_quantities or None,
            max_active_dishes=max_dishes_cap,
            dietary_mode=dietary_mode,
        )
        base_meta["candidate_validation"].append({"origin": origin, **validation})
        if validation["accepted"]:
            delivered = F @ q.astype(float)
            cal_cost = abs(delivered[NUTRIENT_INDEX["DR1TKCAL"]] - stage.energy_prescription["target_calories"]) * info["calorie_deviation_weight"]
            sel_cost = 0.10 * sum(food.selection_cost for food, quantity in zip(eligible, q) if quantity > 0)
            chg_cost = 0.0
            if prev_quantities and change_penalty > 0:
                chg_cost = change_penalty * sum(abs(int(q[i]) - int(prev_quantities.get(eligible[i].food_id, 0))) for i in range(m))

            target_dev_cost = 0.0
            if objective_variant == "target_aware" and target_weight > 0:
                for code in target_nutrients:
                    k = NUTRIENT_INDEX[code]
                    s_k = DAILY_REFERENCE_SCALES[code]
                    target_dev_cost += target_weight * (abs(delivered[k] - stage.targets[code]) / s_k)

            total_score = cal_cost + sel_cost + chg_cost + target_dev_cost
            breakdown = {
                "calorie_cost": float(cal_cost),
                "selection_cost": float(sel_cost),
                "revision_change_cost": float(chg_cost),
                "target_deviation_cost": float(target_dev_cost),
                "total_score": float(total_score),
            }
            valid.append((total_score, origin, candidate, validation, breakdown))

    if not valid:
        code = "infeasible_active_rules" if result.status == 2 else "timeout_no_valid_plan" if result.status == 1 else "no_valid_candidate"
        return s3._failure_output(
            code,
            "No candidate passed all active nutrient, portion and post-check rules. No plan was accepted.",
            metadata=base_meta,
            details={"solver_message": str(result.message), "candidate_validation": base_meta["candidate_validation"]},
        )

    score, origin, output, validation, cost_breakdown = min(valid, key=lambda row: row[0])
    gap = getattr(result, "mip_gap", None)

    # Compute target attainment metrics on accepted delivered nutrition
    delivered_map = {row["nutrient"]: row["delivered"] for row in output.delivered_nutrition}
    eval_target_nutrients = target_nutrients if target_nutrients else get_eligible_target_nutrients(stage, eligible)
    attainment = compute_target_attainment_metrics(delivered_map, stage, eval_target_nutrients)

    output.solver_metadata = {
        **base_meta,
        "status": "accepted_solver_plan" if origin == "solver" else "accepted_validated_heuristic",
        "optimality_proven": bool(origin == "solver" and result.status == 0),
        "mip_gap": None if gap is None or not math.isfinite(float(gap)) else float(gap),
        "heuristic_fallback_used": origin == "heuristic",
        "heuristic_candidate_selected": origin == "heuristic",
        "active_dish_count": len(output.solved_food_basket),
        "total_servings": sum(row["quantity"] for row in output.solved_food_basket),
        "calorie_error_pct": output.clinical_verification["calories"]["absolute_error_pct"],
        "target_calories": stage.energy_prescription["target_calories"],
        "delivered_calories": output.clinical_verification["calories"]["delivered_kcal"],
        "total_objective_score": float(score),
        "cost_breakdown": cost_breakdown,
        "active_nutrient_slack": 0.0,
        "target_attainment": attainment,
    }
    output.accepted = True
    output.clinical_verification["all_active_checks_passed"] = True
    return output
