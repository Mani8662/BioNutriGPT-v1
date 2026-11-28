"""Stage 2 -> Stage 3 Adapter and Independent Verification Logic.

Provides:
1. build_stage3_case_input: Constructs validated Stage3 input from Stage 2 checkpoints and cases_LOCAL.json.
2. compute_translation_fingerprint: Canonical SHA-256 fingerprint binding participant, arm, mode, targets,
   bounds, baselines, energy prescriptions, catalog hash, and solver settings.
3. replay_stage3_solution: Independent audit verifying food basket arithmetic, dish count, category rules,
   serving limits, active nutrient bounds, AMDR ranges, Na:K ratio, calorie limits, and dietary exclusions.
4. evaluate_translation_endpoints: Evaluates both Diagnostic Target Translation (Evaluation A) and
   End-to-End Pipeline Acceptance (Evaluation B).
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import stage3_milp_solver as s3
import nutrient_rule_policy as nrp

PROTOCOL_VERSION = "stage2_stage3_translation_study_v1"
TIME_LIMIT_SEC = 3.0

def validate_finite(obj: Any, path: str = "") -> None:
    """Recursively validate that all numeric values in obj are finite."""
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError(f"Nonfinite float encountered at {path}: {obj}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            validate_finite(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            validate_finite(v, f"{path}[{i}]")

def compute_translation_fingerprint(
    case_idx: int,
    seqn: int,
    arm: str,
    dietary_mode: str,
    stage2_input: dict[str, Any],
    catalog_hash: str,
    time_limit_sec: float = TIME_LIMIT_SEC,
    protocol_version: str = PROTOCOL_VERSION,
) -> str:
    """Compute deterministic canonical SHA-256 fingerprint for a translation solve task."""
    envelope_rows = []
    for r in stage2_input.get("nutrient_constraint_envelope", []):
        envelope_rows.append({
            "nutrient": str(r["nutrient"]),
            "target": float(r["target"]),
            "min_bound": float(r["min_bound"]),
            "max_bound": float(r["max_bound"]),
            "baseline": float(r["baseline"]),
            "rule_provenance": r.get("rule_provenance", {}),
        })
    envelope_rows.sort(key=lambda x: x["nutrient"])

    payload = {
        "protocol_version": str(protocol_version),
        "case_idx": int(case_idx),
        "SEQN": int(seqn),
        "arm": str(arm),
        "dietary_mode": str(dietary_mode),
        "patient_metadata": stage2_input.get("patient_metadata", {}),
        "energy_prescription": stage2_input.get("energy_prescription", {}),
        "nutrient_constraint_envelope": envelope_rows,
        "bindings": {
            "catalog_hash": str(catalog_hash),
            "nutrient_ruleset_version": stage2_input.get("nutrient_ruleset_version", nrp.RULESET_VERSION),
        },
        "solver_settings": {
            "time_limit_sec": float(time_limit_sec),
            "slack_policy": "zero_active_slack",
            "objective_policy": "calorie_deviation_plus_food_selection",
        },
    }
    validate_finite(payload)
    dumped = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=s3._json_safe)
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()

def replay_stage3_solution(
    basket: Sequence[Mapping[str, Any]],
    foods: Sequence[s3.FoodItem],
    stage2_input: Mapping[str, Any],
    dietary_mode: str,
) -> dict[str, Any]:
    """Independent replay audit of a solved discrete meal plan against catalog and envelope.
    
    Reconstructs nutrient totals from food quantities and catalog values from scratch.
    Verifies serving integrality, portion bounds, category plate rules, active nutrient bounds,
    AMDR macronutrient percentages, Na:K ratio preservation, and calorie bounds.
    """
    lookup = {f.food_id: f for f in foods}
    errors: list[str] = []
    warnings: list[str] = []

    if not basket:
        return {
            "has_plan": False,
            "passed": False,
            "errors": ["empty_food_basket"],
            "totals": dict.fromkeys(s3.NUTRIENT_CODES, 0.0),
            "food_mass_g": 0.0,
            "dish_count": 0,
            "calorie_error_fraction": 1.0,
        }

    # Food validation and totals
    q_map: dict[str, int] = {}
    tot = dict.fromkeys(s3.NUTRIENT_CODES, 0.0)
    mass_g = 0.0
    cat_counts: dict[str, int] = {}
    cat_active: dict[str, int] = {}

    for item in basket:
        fid = str(item.get("food_id", ""))
        q = item.get("quantity")
        if fid not in lookup:
            errors.append(f"unknown_food_id:{fid}")
            continue
        if fid in q_map:
            errors.append(f"duplicate_food_in_basket:{fid}")
            continue
        if type(q) is not int or isinstance(q, bool) or not (1 <= q <= s3.MAX_SERVINGS_PER_DISH):
            errors.append(f"invalid_serving_quantity:{fid}:{q}")
            continue

        q_map[fid] = q
        food = lookup[fid]

        # Mode exclusion check
        if dietary_mode == "pure_vegetarian" and food.dietary_type == "non_vegetarian":
            errors.append(f"non_veg_food_in_pure_veg:{fid}")

        mass_g += q * food.serving_weight_g
        cat_counts[food.category] = cat_counts.get(food.category, 0) + q
        cat_active[food.category] = cat_active.get(food.category, 0) + 1

        for k in s3.NUTRIENT_CODES:
            tot[k] += q * food.nutrients_per_serving[k]

    # Dish count
    dish_count = len(q_map)
    if not (s3.MIN_ACTIVE_DISHES <= dish_count <= s3.MAX_ACTIVE_DISHES):
        errors.append(f"active_dish_count_out_of_range:{dish_count}")

    # Category limits
    cat_limits, serv_limits = s3._category_rules_for_mode(dietary_mode, allow_zero_dairy=False)
    for cat, (lo_a, hi_a) in cat_limits.items():
        a = cat_active.get(cat, 0)
        if not (lo_a <= a <= hi_a):
            errors.append(f"category_variety_rule_violated:{cat}:count={a}_range=[{lo_a},{hi_a}]")

    for cat, (lo_s, hi_s) in serv_limits.items():
        s = cat_counts.get(cat, 0)
        if not (lo_s <= s <= hi_s):
            errors.append(f"category_serving_rule_violated:{cat}:servings={s}_range=[{lo_s},{hi_s}]")

    # Enforce active nutrient bounds
    envelope = stage2_input.get("nutrient_constraint_envelope", [])
    env_map = {r["nutrient"]: r for r in envelope}

    bound_violations: list[str] = []
    for r in envelope:
        k = r["nutrient"]
        role = r.get("rule_provenance", {}).get("planning_role")
        if role in ("report_only", "exposure_reporting"):
            continue

        # Check unknown nutrients
        if any(k in lookup[fid].unknown_nutrients for fid in q_map):
            errors.append(f"active_nutrient_missing_data_in_selected_food:{k}")

        val = tot[k]
        lo = float(r["min_bound"])
        hi = float(r["max_bound"])
        # Standard absolute tolerance of 1e-4
        if not (lo - 1e-4 <= val <= hi + 1e-4):
            bound_violations.append(f"{k}:{val:.3f}_outside_[{lo:.3f},{hi:.3f}]")

    if bound_violations:
        errors.append(f"bound_violations:{bound_violations}")

    # Clinical post-checks
    target_kcal = float(stage2_input.get("energy_prescription", {}).get("target_calories", 2000.0))
    deliv_kcal = tot["DR1TKCAL"]
    cal_err_pct = (abs(deliv_kcal - target_kcal) / target_kcal * 100.0) if target_kcal > 0 else 100.0
    if not (deliv_kcal > 0 and cal_err_pct <= 5.0 + 1e-4):
        errors.append(f"calorie_error_exceeds_5pct:{cal_err_pct:.2f}%")

    # AMDR check (10-35% P, 45-65% C, 20-35% F)
    p_kcal = tot["DR1TPROT"] * 4.0
    c_kcal = tot["DR1TCARB"] * 4.0
    f_kcal = tot["DR1TTFAT"] * 9.0
    denom = deliv_kcal if deliv_kcal > 0 else 1.0

    p_pct = p_kcal / denom * 100.0
    c_pct = c_kcal / denom * 100.0
    f_pct = f_kcal / denom * 100.0

    if not (10.0 - 1e-4 <= p_pct <= 35.0 + 1e-4):
        errors.append(f"amdr_protein_violation:{p_pct:.2f}%")
    if not (45.0 - 1e-4 <= c_pct <= 65.0 + 1e-4):
        errors.append(f"amdr_carb_violation:{c_pct:.2f}%")
    if not (20.0 - 1e-4 <= f_pct <= 35.0 + 1e-4):
        errors.append(f"amdr_fat_violation:{f_pct:.2f}%")

    # Sodium to Potassium ratio check
    na = tot["DR1TSODI"]
    k_val = tot["DR1TPOTA"]
    deliv_ratio = na / (k_val + 1.0)

    base_na = float(env_map["DR1TSODI"]["baseline"])
    base_k = float(env_map["DR1TPOTA"]["baseline"])
    base_ratio = base_na / (base_k + 1.0)

    if not (deliv_ratio <= base_ratio + 1e-4):
        errors.append(f"sodium_potassium_ratio_violation:deliv={deliv_ratio:.4f}_base={base_ratio:.4f}")

    # Fiber adequacy
    fiber = tot["DR1TFIBE"]
    fiber_min = float(env_map["DR1TFIBE"]["min_bound"])
    fiber_per_1000 = fiber / (deliv_kcal / 1000.0) if deliv_kcal > 0 else 0.0
    if not (fiber >= fiber_min - 1e-4 and fiber_per_1000 >= 14.0 - 1e-4):
        errors.append(f"fiber_adequacy_violation:delivered={fiber:.2f}g_per1000={fiber_per_1000:.2f}g")

    return {
        "has_plan": len(errors) == 0,
        "passed": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "totals": tot,
        "food_mass_g": round(mass_g, 2),
        "dish_count": dish_count,
        "calorie_error_pct": round(cal_err_pct, 4),
        "macronutrient_pct": {
            "protein": round(p_pct, 2),
            "carb": round(c_pct, 2),
            "fat": round(f_pct, 2),
        },
        "sodium_potassium_ratio": {
            "delivered": round(deliv_ratio, 4),
            "baseline": round(base_ratio, 4),
        },
        "fiber_g_per_1000_kcal": round(fiber_per_1000, 2),
    }

def compute_soft_target_deviations(
    delivered_totals: Mapping[str, float],
    target_envelope: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute continuous target attainment deviations between delivered meal and continuous targets."""
    # Scale normalization factors matching Stage 2
    scales = {
        "DR1TKCAL": 500.0,
        "DR1TPROT": 50.0, "DR1TCARB": 50.0, "DR1TSUGR": 50.0, "DR1TFIBE": 50.0,
        "DR1TTFAT": 50.0, "DR1TSFAT": 50.0, "DR1TMFAT": 50.0, "DR1TPFAT": 50.0,
        "DR1TSODI": 1000.0, "DR1TPOTA": 1000.0, "DR1TCALC": 1000.0, "DR1TPHOS": 1000.0,
        "DR1TMAGN": 1000.0, "DR1TCHOL": 100.0, "DR1TVC": 100.0,
    }
    weights = {
        "DR1TKCAL": 1.0, "DR1TPROT": 1.0, "DR1TCARB": 1.0, "DR1TSUGR": 1.0, "DR1TFIBE": 1.0,
        "DR1TTFAT": 1.0, "DR1TSFAT": 1.0, "DR1TSODI": 1.0, "DR1TPOTA": 1.0,
    }

    per_nutrient_abs_err: dict[str, float] = {}
    per_nutrient_pct_err: dict[str, float] = {}
    sum_sq_norm = 0.0

    for r in target_envelope:
        k = r["nutrient"]
        val = delivered_totals.get(k, 0.0)
        tgt = float(r["target"])
        abs_err = abs(val - tgt)
        pct_err = (abs_err / tgt * 100.0) if tgt > 0 else (0.0 if abs_err == 0 else 100.0)

        per_nutrient_abs_err[k] = round(abs_err, 4)
        per_nutrient_pct_err[k] = round(pct_err, 4)

        s = scales.get(k, 100.0 if "MCG" in k or "V" in k else 500.0)
        w = weights.get(k, 0.10)
        sum_sq_norm += w * ((val - tgt) / s) ** 2

    norm_dist = math.sqrt(sum_sq_norm)

    # Core active nutrients MAPE
    core_keys = ["DR1TKCAL", "DR1TPROT", "DR1TCARB", "DR1TSUGR", "DR1TFIBE", "DR1TTFAT", "DR1TSFAT", "DR1TSODI", "DR1TPOTA"]
    core_mape = float(np.mean([per_nutrient_pct_err[k] for k in core_keys if k in per_nutrient_pct_err]))

    return {
        "normalized_target_distance": round(norm_dist, 5),
        "core_nutrients_mape": round(core_mape, 3),
        "per_nutrient_absolute_error": per_nutrient_abs_err,
        "per_nutrient_percentage_error": per_nutrient_pct_err,
    }
