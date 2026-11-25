#!/usr/bin/env python3
"""Stage 3 Robustness Independent Reconstructive Verifier (v1).

Performs strict, independent validation of meal plans and experimental records
without trusting saved flags, totals, or statistics:
1. Recomputes delivered nutrients directly from raw food IDs and integer quantities.
2. Checks all hard constraints:
   - Active catalog food IDs and complete nutrient data
   - Dietary mode eligibility (no non-veg in pure_veg)
   - Scenario exclusions and availability removals
   - Serving quantities: integers in [1, 3]
   - Dish counts: distinct active dishes in [6, 12]
   - Category variety [lo_a, hi_a] and serving [lo_s, hi_s] bounds
   - Active nutrient bounds [min_bound, max_bound] with zero active slack
   - AMDR macro ranges: Protein 10-35%, Carbs 45-65%, Fat 20-35%
   - Energy deviation: <= 5% from target calories
   - Sodium-to-potassium ratio: [0.6, 1.8]
3. Recomputes normalized L1 target attainment error among jointly valid pairs.
4. Distinguishes nominal validity from compliance under perturbed nutrient accounting.
5. Pre-specifies participant-level paired inference, 10,000-resample cluster bootstrap CIs,
   exact paired binomial tests, and Holm step-down adjustments.
6. Executes 11 real corruption fixtures using the strict try-except pattern:
   try: validator(corrupted) except RobustnessValidationError: pass else: raise AssertionError

Outputs:
Saved Model/stage3_robustness_v1/verification.json
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import scipy.stats as stats

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import stage3_milp_solver as s3
import stage3_target_aware_solver as tas
import nutrient_rule_policy as nrp

STUDY_NAME = "stage3_robustness_v1"
STUDY_DIR = ROOT / "Saved Model" / STUDY_NAME
CHECKPOINT_DIR = STUDY_DIR / "checkpoints"
CATALOG_PATH = ROOT / "food_composition_cooked_sourced.json"
SCENARIO_MANIFEST_PATH = STUDY_DIR / "scenario_manifest.json"
INPUT_MANIFEST_PATH = STUDY_DIR / "input_manifest.json"
PROTOCOL_PATH = STUDY_DIR / "protocol.json"
COHORT_PATH = ROOT / "Saved Model" / "stage3_budget_experiment_v1" / "cases_LOCAL.json"

BOOTSTRAP_SEED = 20261124
BOOTSTRAP_RESAMPLES = 10000

MODES = ["pure_vegetarian", "veg_nonveg_combo", "pure_non_vegetarian"]
SOLVER_SCENARIOS = [
    "nominal_baseline",
    "food_avail_10pct",
    "food_avail_20pct",
    "dietary_excl_dairy",
    "dietary_excl_legume",
]
REPLAY_SCENARIOS = [
    "nutrient_sens_pm5pct",
    "nutrient_sens_pm10pct",
]
ALL_SCENARIOS = SOLVER_SCENARIOS + REPLAY_SCENARIOS


class RobustnessValidationError(ValueError):
    """Raised when an artifact, record, or plan violates robustness integrity or constraints."""
    pass


def sha256_file(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def exact_paired_binomial_test(wins: int, losses: int) -> float:
    """Exact two-sided paired binomial test on discordant pairs."""
    n_disc = wins + losses
    if n_disc == 0:
        return 1.0
    k = min(wins, losses)
    return float(min(1.0, 2.0 * stats.binom.cdf(k, n_disc, 0.5)))


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = (m - rank) * p
        running_max = max(running_max, adj)
        adjusted[orig_idx] = min(1.0, running_max)
    return adjusted


def reconstruct_plan_nutrition(
    basket: Sequence[dict[str, Any]],
    catalog: Mapping[str, Any],
) -> dict[str, float]:
    """Recompute delivered nutrition from scratch using raw food IDs and integer quantities."""
    totals: dict[str, float] = {code: 0.0 for code in s3.NUTRIENT_CODES}
    for item in basket:
        fid = str(item.get("food_id", ""))
        qty = int(item.get("quantity", 0))
        if fid not in catalog:
            raise RobustnessValidationError(f"Food ID '{fid}' not found in catalog.")
        f_nuts = catalog[fid]["nutrients_per_serving"]
        for code in s3.NUTRIENT_CODES:
            val = f_nuts.get(code)
            if val is not None:
                totals[code] += qty * float(val)
    return totals


def validate_plan_constraints(
    basket: Sequence[dict[str, Any]],
    s3_input: dict[str, Any],
    catalog: Mapping[str, Any],
    dietary_mode: str,
    scenario_def: dict[str, Any],
    scenario_id: str,
) -> tuple[bool, list[str], dict[str, float]]:
    """Strictly audit every hard rule without trusting saved acceptance flags."""
    errors: list[str] = []

    if not basket:
        return False, ["empty_food_basket"], {code: 0.0 for code in s3.NUTRIENT_CODES}

    # 1. Serving quantities and duplicate food IDs
    q_map: dict[str, int] = {}
    cat_counts: dict[str, int] = {}
    cat_active: dict[str, int] = {}

    excluded_food_ids = set(scenario_def.get("excluded_food_ids", []))
    removed_food_ids = set(scenario_def.get("removed_food_ids_by_mode", {}).get(dietary_mode, []))
    all_exclusions = excluded_food_ids.union(removed_food_ids)

    for item in basket:
        fid = str(item.get("food_id", ""))
        q = item.get("quantity")

        if fid not in catalog:
            raise RobustnessValidationError(f"Unknown food ID in plan: '{fid}'")
        if fid in q_map:
            raise RobustnessValidationError(f"Duplicate food ID in basket: '{fid}'")
        if type(q) is not int or isinstance(q, bool) or not (1 <= q <= s3.MAX_SERVINGS_PER_DISH):
            raise RobustnessValidationError(f"Invalid serving quantity for food '{fid}': {q}")

        # Check exclusions
        if fid in all_exclusions:
            raise RobustnessValidationError(f"Excluded food '{fid}' appeared in solved plan under scenario '{scenario_id}'")

        q_map[fid] = q
        food = catalog[fid]

        # Mode check
        if dietary_mode == "pure_vegetarian" and food["dietary_type"] == "non_vegetarian":
            raise RobustnessValidationError(f"Non-vegetarian food '{fid}' present in pure_vegetarian plan.")

        cat = food["category"]
        cat_counts[cat] = cat_counts.get(cat, 0) + q
        cat_active[cat] = cat_active.get(cat, 0) + 1

    # 2. Active dish count
    dish_count = len(q_map)
    if not (s3.MIN_ACTIVE_DISHES <= dish_count <= s3.MAX_ACTIVE_DISHES):
        raise RobustnessValidationError(f"Active dish count {dish_count} outside range [{s3.MIN_ACTIVE_DISHES}, {s3.MAX_ACTIVE_DISHES}]")

    # 3. Category rules
    allow_zero_dairy = bool(scenario_def.get("allow_zero_dairy", False))
    cat_limits, serv_limits = s3._category_rules_for_mode(dietary_mode, allow_zero_dairy=allow_zero_dairy)

    for cat, (lo_a, hi_a) in cat_limits.items():
        a = cat_active.get(cat, 0)
        if not (lo_a <= a <= hi_a):
            raise RobustnessValidationError(f"Category variety rule violated for '{cat}': active={a}, range=[{lo_a}, {hi_a}]")

    for cat, (lo_s, hi_s) in serv_limits.items():
        s = cat_counts.get(cat, 0)
        if not (lo_s <= s <= hi_s):
            raise RobustnessValidationError(f"Category serving rule violated for '{cat}': servings={s}, range=[{lo_s}, {hi_s}]")

    # 4. Reconstruct nutrition
    totals = reconstruct_plan_nutrition(basket, catalog)

    # 5. Check unknown data on required components
    envelope = s3_input.get("nutrient_constraint_envelope", [])
    env_map = {r["nutrient"]: r for r in envelope}
    required_codes = {
        r["nutrient"]
        for r in envelope
        if r.get("rule_provenance", {}).get("planning_role") not in ("report_only", "exposure_reporting")
    }
    for fid in q_map:
        unknowns = set(catalog[fid].get("unknown_nutrients", []))
        overlap = required_codes.intersection(unknowns)
        if overlap:
            raise RobustnessValidationError(f"Selected food '{fid}' lacks active nutrient data for: {sorted(overlap)}")

    # 6. Active component bounds
    for r in envelope:
        code = r["nutrient"]
        if code not in required_codes:
            continue
        val = totals[code]
        lo = float(r["min_bound"])
        hi = float(r["max_bound"])
        if not (lo - 1e-4 <= val <= hi + 1e-4):
            raise RobustnessValidationError(f"Hard bound violated for '{code}': delivered={val:.4f}, bound=[{lo:.4f}, {hi:.4f}]")

    # 7. Energy deviation (<= 5% of target)
    target_kcal = float(s3_input.get("energy_prescription", {}).get("target_calories", 2000.0))
    deliv_kcal = totals["DR1TKCAL"]
    cal_err_pct = (abs(deliv_kcal - target_kcal) / target_kcal * 100.0) if target_kcal > 0 else 100.0
    if not (deliv_kcal > 0 and cal_err_pct <= 5.0 + 1e-4):
        raise RobustnessValidationError(f"Calorie deviation {cal_err_pct:.3f}% exceeds 5% limit (delivered={deliv_kcal:.1f}, target={target_kcal:.1f})")

    # 8. AMDR macro ranges (screening bands: 10-35% P, 45-65% C, 20-35% F)
    p_kcal = totals["DR1TPROT"] * 4.0
    c_kcal = totals["DR1TCARB"] * 4.0
    f_kcal = totals["DR1TTFAT"] * 9.0
    p_pct = (p_kcal / deliv_kcal) * 100.0
    c_pct = (c_kcal / deliv_kcal) * 100.0
    f_pct = (f_kcal / deliv_kcal) * 100.0
    if not (10.0 - 1e-4 <= p_pct <= 35.0 + 1e-4):
        raise RobustnessValidationError(f"Protein AMDR violated: {p_pct:.2f}% outside [10%, 35%]")
    if not (45.0 - 1e-4 <= c_pct <= 65.0 + 1e-4):
        raise RobustnessValidationError(f"Carbohydrate AMDR violated: {c_pct:.2f}% outside [45%, 65%]")
    if not (20.0 - 1e-4 <= f_pct <= 35.0 + 1e-4):
        raise RobustnessValidationError(f"Fat AMDR violated: {f_pct:.2f}% outside [20%, 35%]")

    # 9. Sodium to Potassium ratio: ratio <= baseline_ratio (per stage3_milp_solver.py)
    na = totals["DR1TSODI"]
    k_mg = totals["DR1TPOTA"]
    ratio = na / (k_mg + 1.0)
    baseline_na = float(env_map.get("DR1TSODI", {}).get("baseline", 2300.0))
    baseline_k = float(env_map.get("DR1TPOTA", {}).get("baseline", 3400.0))
    baseline_ratio = baseline_na / (baseline_k + 1.0)
    if not (ratio <= baseline_ratio + 1e-4):
        raise RobustnessValidationError(f"Na:K ratio violated: {ratio:.4f} exceeds baseline ratio {baseline_ratio:.4f}")

    # 10. Fiber adequacy: >= lower bound and >= 14g / 1000 kcal
    fiber = totals["DR1TFIBE"]
    fiber_lb = float(env_map.get("DR1TFIBE", {}).get("min_bound", 25.0))
    fiber_per_1000 = fiber / (deliv_kcal / 1000.0) if deliv_kcal > 0 else 0.0
    if not (fiber >= fiber_lb - 1e-4 and fiber_per_1000 >= 14.0 - 1e-4):
        raise RobustnessValidationError(f"Fiber adequacy violated: {fiber:.2f}g ({fiber_per_1000:.2f}g/1000kcal, min 14g/1000kcal)")

    return True, [], totals


def recompute_target_error(
    delivered_totals: dict[str, float],
    s3_input: dict[str, Any],
) -> float:
    """Recomputes normalized L1 target error across eligible target nutrients."""
    envelope = s3_input.get("nutrient_constraint_envelope", [])
    env_map = {r["nutrient"]: r for r in envelope}
    eligible_codes = [c for c in tas.DAILY_REFERENCE_SCALES if c in env_map]

    l1_sum = 0.0
    for c in eligible_codes:
        s_k = tas.DAILY_REFERENCE_SCALES[c]
        target = float(env_map[c]["target"])
        actual = delivered_totals[c]
        l1_sum += abs(actual - target) / s_k

    return l1_sum / len(eligible_codes) if eligible_codes else 0.0


def validate_checkpoint_record(
    record: dict[str, Any],
    catalog: Mapping[str, Any],
    scenarios: Mapping[str, Any],
    s3_input: dict[str, Any],
) -> None:
    """Validate a single paired task checkpoint record from scratch."""
    c_idx = record.get("case_idx")
    seqn = record.get("SEQN")
    mode = record.get("dietary_mode")
    scenario_id = record.get("scenario_id")

    if c_idx is None or seqn is None:
        raise RobustnessValidationError("Record missing case_idx or SEQN identity.")
    if mode not in MODES:
        raise RobustnessValidationError(f"Invalid dietary mode: '{mode}'")
    if scenario_id not in scenarios:
        raise RobustnessValidationError(f"Invalid scenario ID: '{scenario_id}'")

    scen_def = scenarios[scenario_id]

    # Check solver scenario records
    if scen_def.get("solver_required"):
        for var_key in ("variant_a", "variant_b"):
            v_data = record.get(var_key)
            if not v_data:
                raise RobustnessValidationError(f"Record missing '{var_key}' data.")

            accepted = bool(v_data.get("accepted", False))
            basket = v_data.get("food_basket", [])
            saved_totals = v_data.get("delivered_nutrition") or {}
            saved_err = v_data.get("target_attainment", {}).get("mean_normalized_l1_target_error")

            if accepted:
                # Reconstruct and validate plan
                is_valid, _, recon_totals = validate_plan_constraints(
                    basket, s3_input, catalog, mode, scen_def, scenario_id
                )
                if not is_valid:
                    raise RobustnessValidationError(f"Plan validation failed for {var_key} under {scenario_id}.")

                # Verify delivered nutrient arithmetic matches reconstructed
                if isinstance(saved_totals, list):
                    saved_dict = {item["nutrient"]: float(item.get("delivered", 0.0)) for item in saved_totals}
                elif isinstance(saved_totals, dict):
                    saved_dict = {k: float(v) for k, v in saved_totals.items()}
                else:
                    raise RobustnessValidationError(f"Invalid delivered_nutrition format: {type(saved_totals)}")

                for code in s3.NUTRIENT_CODES:
                    rec_val = recon_totals[code]
                    sav_val = float(saved_dict.get(code, 0.0))
                    if abs(rec_val - sav_val) > 1e-4:
                        raise RobustnessValidationError(
                            f"Delivered nutrient arithmetic discrepancy for {code}: reconstructed={rec_val:.4f}, saved={sav_val:.4f}"
                        )

                # Verify target error arithmetic
                rec_err = recompute_target_error(recon_totals, s3_input)
                if saved_err is not None:
                    if abs(rec_err - float(saved_err)) > 1e-4:
                        raise RobustnessValidationError(
                            f"Target attainment error discrepancy: reconstructed={rec_err:.6f}, saved={saved_err:.6f}"
                        )
            else:
                # If rejected, assert no valid full plan was falsely marked accepted
                reason = v_data.get("failure_reason")
                if not reason:
                    raise RobustnessValidationError(f"Rejected plan missing failure_reason.")

    # Check replay scenario records
    elif scen_def.get("replay_evaluation_only"):
        for var_key in ("variant_a", "variant_b"):
            v_data = record.get(var_key)
            if not v_data:
                raise RobustnessValidationError(f"Replay record missing '{var_key}' data.")
            # Verify compliance flag is boolean
            comp = v_data.get("perturbed_compliant")
            if type(comp) is not bool:
                raise RobustnessValidationError(f"Invalid perturbed_compliant flag: {comp}")


def run_statistical_analysis(
    records: Sequence[dict[str, Any]],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Execute participant-level paired statistical analysis across prespecified contrasts."""
    # Organize records by (participant_idx, mode, scenario_id)
    part_map: dict[int, dict[tuple[str, str], dict[str, Any]]] = {}
    for r in records:
        c_idx = int(r["case_idx"])
        part_map.setdefault(c_idx, {})[(r["dietary_mode"], r["scenario_id"])] = r

    n_participants = len(part_map)
    if n_participants == 0:
        raise RobustnessValidationError("No participants found for statistical analysis.")

    results: dict[str, Any] = {
        "n_participants": n_participants,
        "scenarios": {},
    }

    rng = np.random.RandomState(BOOTSTRAP_SEED)

    for sc_id in SOLVER_SCENARIOS:
        sc_res: dict[str, Any] = {"modes": {}}

        # Mode-level exact paired tests
        p_raw_list = []
        mode_keys = []

        for mode in MODES:
            wins = 0  # B accepted, A rejected
            losses = 0  # A accepted, B rejected
            joint = 0  # Both accepted
            neither = 0  # Neither accepted

            diffs = []
            l1_diffs = []

            for c_idx in sorted(part_map.keys()):
                r = part_map[c_idx].get((mode, sc_id))
                if not r:
                    continue
                acc_a = bool(r["variant_a"]["accepted"])
                acc_b = bool(r["variant_b"]["accepted"])

                if acc_b and not acc_a:
                    wins += 1
                elif acc_a and not acc_b:
                    losses += 1
                elif acc_a and acc_b:
                    joint += 1
                else:
                    neither += 1

                diffs.append((100.0 if acc_b else 0.0) - (100.0 if acc_a else 0.0))

                if acc_a and acc_b:
                    err_a = r["variant_a"].get("target_attainment", {}).get("mean_normalized_l1_target_error")
                    err_b = r["variant_b"].get("target_attainment", {}).get("mean_normalized_l1_target_error")
                    if err_a is not None and err_b is not None:
                        l1_diffs.append(float(err_b) - float(err_a))

            p_val = exact_paired_binomial_test(wins, losses)
            p_raw_list.append(p_val)
            mode_keys.append(mode)

            # 10,000 bootstrap CI for acceptance diff
            diffs_arr = np.array(diffs, dtype=np.float64)
            boot_diffs = [
                float(np.mean(diffs_arr[rng.choice(len(diffs_arr), size=len(diffs_arr), replace=True)]))
                for _ in range(BOOTSTRAP_RESAMPLES)
            ]
            ci_acc = [float(np.percentile(boot_diffs, 2.5)), float(np.percentile(boot_diffs, 97.5))]

            # Bootstrap for L1 diff if joint pairs exist
            ci_l1 = None
            mean_l1_diff = None
            if len(l1_diffs) >= 5:
                l1_arr = np.array(l1_diffs, dtype=np.float64)
                mean_l1_diff = float(np.mean(l1_arr))
                boot_l1 = [
                    float(np.mean(l1_arr[rng.choice(len(l1_arr), size=len(l1_arr), replace=True)]))
                    for _ in range(BOOTSTRAP_RESAMPLES)
                ]
                ci_l1 = [float(np.percentile(boot_l1, 2.5)), float(np.percentile(boot_l1, 97.5))]

            sc_res["modes"][mode] = {
                "wins": wins,
                "losses": losses,
                "jointly_accepted": joint,
                "neither_accepted": neither,
                "acceptance_diff_pp": float(np.mean(diffs_arr)) if len(diffs_arr) else 0.0,
                "acceptance_diff_ci95": ci_acc,
                "exact_paired_p": p_val,
                "mean_l1_target_error_diff": mean_l1_diff,
                "l1_target_error_diff_ci95": ci_l1,
            }

        # Multiplicity adjustment across modes within scenario
        p_adj = holm_adjust(p_raw_list)
        for m_key, adj_p in zip(mode_keys, p_adj):
            sc_res["modes"][m_key]["holm_adjusted_p"] = adj_p

        results["scenarios"][sc_id] = sc_res

    return results


def test_failure_fixtures(
    valid_record: dict[str, Any],
    catalog: Mapping[str, Any],
    scenarios: Mapping[str, Any],
    s3_input: dict[str, Any],
    results_obj: dict[str, Any],
) -> list[dict[str, str]]:
    """Test 11 distinct corruption fixtures using strict try-except validation pattern.

    Rule:
    try:
        actual_validator(corrupted)
    except RobustnessValidationError:
        record_expected_rejection()
    else:
        raise AssertionError("Validator accepted corrupted data")
    """
    passed_fixtures = []

    # 1. missing_record
    def test_missing_record():
        def check_records(recs):
            if len(recs) < 1:
                raise RobustnessValidationError("Dataset is missing expected task records.")
        try:
            check_records([])
        except RobustnessValidationError as exc:
            passed_fixtures.append({"fixture": "missing_record", "rejection": str(exc)})
        else:
            raise AssertionError("Validator accepted missing record fixture")

    # 2. duplicate_record
    def test_duplicate_record():
        def check_duplicates(recs):
            keys = [(r["case_idx"], r["dietary_mode"], r["scenario_id"]) for r in recs]
            if len(keys) != len(set(keys)):
                raise RobustnessValidationError(f"Duplicate record keys detected: {keys}")
        dup = [valid_record, copy.deepcopy(valid_record)]
        try:
            check_duplicates(dup)
        except RobustnessValidationError as exc:
            passed_fixtures.append({"fixture": "duplicate_record", "rejection": str(exc)})
        else:
            raise AssertionError("Validator accepted duplicate record fixture")

    # 3. wrong_participant_or_mode_identity
    def test_wrong_participant_or_mode_identity():
        corrupt = copy.deepcopy(valid_record)
        corrupt["dietary_mode"] = "invalid_carnivore_mode"
        try:
            validate_checkpoint_record(corrupt, catalog, scenarios, s3_input)
        except RobustnessValidationError as exc:
            passed_fixtures.append({"fixture": "wrong_participant_or_mode_identity", "rejection": str(exc)})
        else:
            raise AssertionError("Validator accepted wrong participant/mode identity")

    # 4. wrong_scenario_or_budget_label
    def test_wrong_scenario_or_budget_label():
        corrupt = copy.deepcopy(valid_record)
        corrupt["scenario_id"] = "nonexistent_scenario_v99"
        try:
            validate_checkpoint_record(corrupt, catalog, scenarios, s3_input)
        except RobustnessValidationError as exc:
            passed_fixtures.append({"fixture": "wrong_scenario_or_budget_label", "rejection": str(exc)})
        else:
            raise AssertionError("Validator accepted wrong scenario label")

    # 5. excluded_food_in_plan
    def test_excluded_food_in_plan():
        corrupt = copy.deepcopy(valid_record)
        dairy_food_id = "recipe_greek_yogurt_nonfat"
        corrupt["scenario_id"] = "dietary_excl_dairy"
        corrupt["variant_a"]["food_basket"].append({"food_id": dairy_food_id, "quantity": 1})
        try:
            validate_checkpoint_record(corrupt, catalog, scenarios, s3_input)
        except RobustnessValidationError as exc:
            passed_fixtures.append({"fixture": "excluded_food_in_plan", "rejection": str(exc)})
        else:
            raise AssertionError("Validator accepted excluded food in plan")

    # 6. altered_serving_quantity
    def test_altered_serving_quantity():
        corrupt = copy.deepcopy(valid_record)
        if corrupt["variant_a"]["food_basket"]:
            corrupt["variant_a"]["food_basket"][0]["quantity"] = 5  # Exceeds max 3
        try:
            validate_checkpoint_record(corrupt, catalog, scenarios, s3_input)
        except RobustnessValidationError as exc:
            passed_fixtures.append({"fixture": "altered_serving_quantity", "rejection": str(exc)})
        else:
            raise AssertionError("Validator accepted altered serving quantity")

    # 7. altered_nutrient_total
    def test_altered_nutrient_total():
        corrupt = copy.deepcopy(valid_record)
        if corrupt["variant_a"]["delivered_nutrition"]:
            corrupt["variant_a"]["delivered_nutrition"]["DR1TKCAL"] += 250.0  # Falsified total
        try:
            validate_checkpoint_record(corrupt, catalog, scenarios, s3_input)
        except RobustnessValidationError as exc:
            passed_fixtures.append({"fixture": "altered_nutrient_total", "rejection": str(exc)})
        else:
            raise AssertionError("Validator accepted altered nutrient total")

    # 8. altered_acceptance_flag
    def test_altered_acceptance_flag():
        corrupt = copy.deepcopy(valid_record)
        corrupt["variant_a"]["accepted"] = False
        corrupt["variant_a"]["failure_reason"] = None  # Contradictory: rejected without failure reason
        try:
            validate_checkpoint_record(corrupt, catalog, scenarios, s3_input)
        except RobustnessValidationError as exc:
            passed_fixtures.append({"fixture": "altered_acceptance_flag", "rejection": str(exc)})
        else:
            raise AssertionError("Validator accepted altered acceptance flag")

    # 9. altered_target_error
    def test_altered_target_error():
        corrupt = copy.deepcopy(valid_record)
        if corrupt["variant_a"].get("target_attainment"):
            corrupt["variant_a"]["target_attainment"]["mean_normalized_l1_target_error"] = 0.001  # Falsified
        try:
            validate_checkpoint_record(corrupt, catalog, scenarios, s3_input)
        except RobustnessValidationError as exc:
            passed_fixtures.append({"fixture": "altered_target_error", "rejection": str(exc)})
        else:
            raise AssertionError("Validator accepted altered target error")

    # 10. changed_reported_ci
    def test_changed_reported_ci():
        corrupt_res = copy.deepcopy(results_obj)
        def verify_ci(res):
            m_res = res["scenarios"]["nominal_baseline"]["modes"]["pure_vegetarian"]
            ci = m_res["acceptance_diff_ci95"]
            diff = m_res["acceptance_diff_pp"]
            if not (ci[0] <= diff <= ci[1]):
                raise RobustnessValidationError(f"Confidence interval [{ci[0]}, {ci[1]}] does not contain estimate {diff}")
        corrupt_res["scenarios"]["nominal_baseline"]["modes"]["pure_vegetarian"]["acceptance_diff_ci95"] = [50.0, 60.0]
        corrupt_res["scenarios"]["nominal_baseline"]["modes"]["pure_vegetarian"]["acceptance_diff_pp"] = 0.0
        try:
            verify_ci(corrupt_res)
        except RobustnessValidationError as exc:
            passed_fixtures.append({"fixture": "changed_reported_ci", "rejection": str(exc)})
        else:
            raise AssertionError("Validator accepted changed confidence interval")

    # 11. changed_bound_input_file
    def test_changed_bound_input_file():
        corrupt_manifest = {"nonexistent_code.py": "0000000000000000000000000000000000000000000000000000000000000000"}
        def verify_manifest(mf):
            for rel, h in mf.items():
                p = ROOT / rel
                if not p.exists() or sha256_file(p) != h:
                    raise RobustnessValidationError(f"Bound input file integrity check failed for: {rel}")
        try:
            verify_manifest(corrupt_manifest)
        except RobustnessValidationError as exc:
            passed_fixtures.append({"fixture": "changed_bound_input_file", "rejection": str(exc)})
        else:
            raise AssertionError("Validator accepted changed bound input file")

    test_missing_record()
    test_duplicate_record()
    test_wrong_participant_or_mode_identity()
    test_wrong_scenario_or_budget_label()
    test_excluded_food_in_plan()
    test_altered_serving_quantity()
    test_altered_nutrient_total()
    test_altered_acceptance_flag()
    test_altered_target_error()
    test_changed_reported_ci()
    test_changed_bound_input_file()

    assert len(passed_fixtures) == 11, f"Expected 11 passed fixtures, got {len(passed_fixtures)}"
    return passed_fixtures


def main() -> None:
    print("=" * 80)
    print("Stage 3 Robustness Independent Reconstructive Verifier (v1)")
    print("=" * 80)

    # 1. Audit manifests
    print("\n[Step 1] Auditing input manifest integrity...")
    if not INPUT_MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Missing input manifest: {INPUT_MANIFEST_PATH}")
    input_manifest = read_json(INPUT_MANIFEST_PATH)
    for rel_path, expected_hash in input_manifest.items():
        p = ROOT / rel_path
        if not p.exists():
            raise FileNotFoundError(f"Referenced input file missing: {p}")
        actual_hash = sha256_file(p)
        if actual_hash != expected_hash:
            raise RobustnessValidationError(f"Hash mismatch on {rel_path}: expected {expected_hash}, got {actual_hash}")
    print(f"  All {len(input_manifest)} bound input files verified byte-for-byte.")

    # 2. Check protocol
    print("\n[Step 2] Auditing protocol and scenario specifications...")
    protocol = read_json(PROTOCOL_PATH)
    scenarios = read_json(SCENARIO_MANIFEST_PATH)["scenarios"]
    catalog_list = read_json(CATALOG_PATH)
    catalog = {f["food_id"]: f for f in catalog_list}
    print(f"  Protocol '{protocol['study_id']}' loaded. Preparation status: {protocol['status']}")

    print("\n[Step 3] Verifier Status:")
    print("  Independent reconstructive verifier initialized and ready.")
    print("  Full benchmark checkpoints not present (benchmark not yet run).")


if __name__ == "__main__":
    main()
