#!/usr/bin/env python3
"""Stage 3 Robustness Scenario Generator and Manifest Builder.

Generates and freezes deterministic scenario specifications for the controlled
engineering robustness evaluation of the Stage 3 target-aware meal optimizer.

Defines 7 scenarios:
1. nominal_baseline: Unmodified catalog and standard baseline constraints.
2. food_avail_10pct: Modest ~10% active food removal (nested subset of 20%).
3. food_avail_20pct: Stronger ~20% active food removal (strict superset of 10%).
4. dietary_excl_dairy: Complete dairy exclusion (5 foods, lactose intolerance / vegan preference).
5. dietary_excl_legume: Complete dal/pulse exclusion (9 foods, tests infeasibility detection in PV & substitution in NonVeg).
6. nutrient_sens_pm5pct: +/- 5% uniform nutrient sensitivity perturbation on nominal meal plans.
7. nutrient_sens_pm10pct: +/- 10% uniform nutrient sensitivity perturbation on nominal meal plans.

Outputs:
Saved Model/stage3_robustness_v1/scenario_manifest.json
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import stage3_milp_solver as s3

OUTPUT_DIR = ROOT / "Saved Model" / "stage3_robustness_v1"
CATALOG_PATH = ROOT / "food_composition_cooked_sourced.json"

SEEDS = {
    "food_avail_pv": 202611251,
    "food_avail_all": 202611252,
    "nutrient_sens_5pct": 2026112505,
    "nutrient_sens_10pct": 2026112510,
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate_scenarios() -> dict[str, Any]:
    assert CATALOG_PATH.exists(), f"Catalog missing: {CATALOG_PATH}"
    foods = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    assert len(foods) == 49, f"Expected 49 foods, got {len(foods)}"

    id_to_food = {f["food_id"]: f for f in foods}
    id_to_cat = {f["food_id"]: f["category"] for f in foods}
    id_to_diet = {f["food_id"]: f["dietary_type"] for f in foods}

    pv_foods = sorted([f["food_id"] for f in foods if f["dietary_type"] == "pure_vegetarian"])
    all_foods = sorted([f["food_id"] for f in foods])

    assert len(pv_foods) == 43
    assert len(all_foods) == 49

    # 1. Food Availability: Deterministic Seeded Removals
    # Pure vegetarian: 10% = 4 foods, 20% = 9 foods
    rng_pv = random.Random(SEEDS["food_avail_pv"])
    pv_shuffled = pv_foods.copy()
    rng_pv.shuffle(pv_shuffled)
    rem_pv_10 = sorted(pv_shuffled[:4])
    rem_pv_20 = sorted(pv_shuffled[:9])
    assert set(rem_pv_10).issubset(set(rem_pv_20)), "Nesting invariant violated for PV availability"

    # All foods (combo & non-veg): 10% = 5 foods, 20% = 10 foods
    rng_all = random.Random(SEEDS["food_avail_all"])
    all_shuffled = all_foods.copy()
    rng_all.shuffle(all_shuffled)
    rem_all_10 = sorted(all_shuffled[:5])
    rem_all_20 = sorted(all_shuffled[:10])
    assert set(rem_all_10).issubset(set(rem_all_20)), "Nesting invariant violated for All availability"

    # 2. Dietary Exclusions
    dairy_ids = sorted([f["food_id"] for f in foods if f["category"] == "dairy"])
    legume_ids = sorted([f["food_id"] for f in foods if f["category"] == "dal_pulse"])
    assert len(dairy_ids) == 5, f"Expected 5 dairy foods, got {len(dairy_ids)}"
    assert len(legume_ids) == 9, f"Expected 9 legume foods, got {len(legume_ids)}"

    # 3. Nutrient Sensitivities: Perturbed Catalogs with Macronutrient-Energy Consistency
    def build_perturbed_catalog(delta_max: float, seed: int) -> dict[str, dict[str, float]]:
        rng = random.Random(seed)
        perturbed: dict[str, dict[str, float]] = {}
        for f in foods:
            fid = f["food_id"]
            nom_nutrients = f["nutrients_per_serving"]
            unknowns = set(f.get("unknown_nutrients", []))
            p_nut = {}
            for code in s3.NUTRIENT_CODES:
                if code in nom_nutrients and code not in unknowns:
                    factor = 1.0 + rng.uniform(-delta_max, delta_max)
                    p_nut[code] = max(0.0, float(nom_nutrients[code]) * factor)
                else:
                    p_nut[code] = None

            # Enforce exact macronutrient-energy consistency: 4P + 4C + 9F = E
            prot = p_nut.get("DR1TPROT") or 0.0
            carb = p_nut.get("DR1TCARB") or 0.0
            fat = p_nut.get("DR1TTFAT") or 0.0
            p_nut["DR1TKCAL"] = 4.0 * prot + 4.0 * carb + 9.0 * fat
            perturbed[fid] = p_nut
        return perturbed

    pert_5pct = build_perturbed_catalog(0.05, SEEDS["nutrient_sens_5pct"])
    pert_10pct = build_perturbed_catalog(0.10, SEEDS["nutrient_sens_10pct"])

    scenarios = {
        "nominal_baseline": {
            "scenario_id": "nominal_baseline",
            "scenario_type": "nominal",
            "description": "Unmodified food catalog and standard baseline constraints.",
            "solver_required": True,
            "replay_evaluation_only": False,
            "budget_seconds": 5.0,
            "removed_food_ids_by_mode": {
                "pure_vegetarian": [],
                "veg_nonveg_combo": [],
                "pure_non_vegetarian": [],
            },
            "excluded_food_ids": [],
            "allow_zero_dairy": False,
            "perturbation_range_pct": 0.0,
            "rationale": "Establishes reference unperturbed performance under 5.0s budget.",
        },
        "food_avail_10pct": {
            "scenario_id": "food_avail_10pct",
            "scenario_type": "food_availability",
            "description": "Modest food availability restriction (~10% active food removal, nested subset of 20%).",
            "solver_required": True,
            "replay_evaluation_only": False,
            "budget_seconds": 5.0,
            "seed": {
                "pure_vegetarian": SEEDS["food_avail_pv"],
                "veg_nonveg_combo": SEEDS["food_avail_all"],
                "pure_non_vegetarian": SEEDS["food_avail_all"],
            },
            "removed_food_ids_by_mode": {
                "pure_vegetarian": rem_pv_10,
                "veg_nonveg_combo": rem_all_10,
                "pure_non_vegetarian": rem_all_10,
            },
            "removed_counts_by_mode": {
                "pure_vegetarian": len(rem_pv_10),
                "veg_nonveg_combo": len(rem_all_10),
                "pure_non_vegetarian": len(rem_all_10),
            },
            "eligible_denominators_by_mode": {
                "pure_vegetarian": len(pv_foods),
                "veg_nonveg_combo": len(all_foods),
                "pure_non_vegetarian": len(all_foods),
            },
            "excluded_food_ids": [],
            "allow_zero_dairy": False,
            "perturbation_range_pct": 0.0,
            "rationale": "Simulates modest catalog supply disruption while preserving category feasibility.",
        },
        "food_avail_20pct": {
            "scenario_id": "food_avail_20pct",
            "scenario_type": "food_availability",
            "description": "Stronger food availability restriction (~20% active food removal, strict superset of 10%).",
            "solver_required": True,
            "replay_evaluation_only": False,
            "budget_seconds": 5.0,
            "seed": {
                "pure_vegetarian": SEEDS["food_avail_pv"],
                "veg_nonveg_combo": SEEDS["food_avail_all"],
                "pure_non_vegetarian": SEEDS["food_avail_all"],
            },
            "removed_food_ids_by_mode": {
                "pure_vegetarian": rem_pv_20,
                "veg_nonveg_combo": rem_all_20,
                "pure_non_vegetarian": rem_all_20,
            },
            "removed_counts_by_mode": {
                "pure_vegetarian": len(rem_pv_20),
                "veg_nonveg_combo": len(rem_all_20),
                "pure_non_vegetarian": len(rem_all_20),
            },
            "eligible_denominators_by_mode": {
                "pure_vegetarian": len(pv_foods),
                "veg_nonveg_combo": len(all_foods),
                "pure_non_vegetarian": len(all_foods),
            },
            "is_strict_superset_of_10pct": True,
            "excluded_food_ids": [],
            "allow_zero_dairy": False,
            "perturbation_range_pct": 0.0,
            "rationale": "Simulates acute supply shortage; strictly nests 10% removal for dose-response comparison.",
        },
        "dietary_excl_dairy": {
            "scenario_id": "dietary_excl_dairy",
            "scenario_type": "dietary_exclusion",
            "description": "Complete dairy exclusion (5 items, lactose intolerance / vegan preference).",
            "solver_required": True,
            "replay_evaluation_only": False,
            "budget_seconds": 5.0,
            "excluded_food_ids": dairy_ids,
            "removed_food_ids_by_mode": {
                "pure_vegetarian": [],
                "veg_nonveg_combo": [],
                "pure_non_vegetarian": [],
            },
            "allow_zero_dairy": True,
            "allergy_classification": "dietary_preference_or_intolerance_unverified_cross_contact",
            "safety_note": "Preference/intolerance exclusion only; absence of dairy annotation does not certify allergen-free.",
            "perturbation_range_pct": 0.0,
            "rationale": "Stresses calcium, B12, riboflavin, and protein optimization from non-dairy sources.",
        },
        "dietary_excl_legume": {
            "scenario_id": "dietary_excl_legume",
            "scenario_type": "dietary_exclusion",
            "description": "Complete legume/dal-pulse exclusion (9 items).",
            "solver_required": True,
            "replay_evaluation_only": False,
            "budget_seconds": 5.0,
            "excluded_food_ids": legume_ids,
            "removed_food_ids_by_mode": {
                "pure_vegetarian": [],
                "veg_nonveg_combo": [],
                "pure_non_vegetarian": [],
            },
            "allow_zero_dairy": False,
            "expected_feasibility": {
                "pure_vegetarian": "mathematically_infeasible_missing_category",
                "veg_nonveg_combo": "mathematically_infeasible_missing_category",
                "pure_non_vegetarian": "feasible_substitutes_animal_protein",
            },
            "allergy_classification": "dietary_preference_unverified_cross_contact",
            "perturbation_range_pct": 0.0,
            "rationale": "Evaluates clean failure detection under hard category rules in PV/combo and substitution in NonVeg.",
        },
        "nutrient_sens_pm5pct": {
            "scenario_id": "nutrient_sens_pm5pct",
            "scenario_type": "nutrient_sensitivity",
            "description": "Nutrient sensitivity replay: re-evaluates nominal meal plans under +/-5% uniform random perturbation.",
            "solver_required": False,
            "replay_evaluation_only": True,
            "base_scenario": "nominal_baseline",
            "seed": SEEDS["nutrient_sens_5pct"],
            "perturbation_range_pct": [-5.0, 5.0],
            "macronutrient_energy_consistency": "DR1TKCAL = 4*PROT + 4*CARB + 9*FAT",
            "perturbed_nutrients": pert_5pct,
            "rationale": "Quantifies uncertainty propagation: distinguishes nominal validity from perturbed compliance.",
        },
        "nutrient_sens_pm10pct": {
            "scenario_id": "nutrient_sens_pm10pct",
            "scenario_type": "nutrient_sensitivity",
            "description": "Nutrient sensitivity replay: re-evaluates nominal meal plans under +/-10% uniform random perturbation.",
            "solver_required": False,
            "replay_evaluation_only": True,
            "base_scenario": "nominal_baseline",
            "seed": SEEDS["nutrient_sens_10pct"],
            "perturbation_range_pct": [-10.0, 10.0],
            "macronutrient_energy_consistency": "DR1TKCAL = 4*PROT + 4*CARB + 9*FAT",
            "perturbed_nutrients": pert_10pct,
            "rationale": "Stresses nutritional robustness under higher lab/sourcing variance.",
        },
    }

    manifest = {
        "manifest_version": "stage3_robustness_v1",
        "food_catalog_sha256": sha256_file(CATALOG_PATH),
        "total_catalog_foods": len(foods),
        "seeds": SEEDS,
        "scenarios": scenarios,
        "scenario_counts": {
            "total_scenarios": len(scenarios),
            "solver_scenarios": sum(1 for s in scenarios.values() if s["solver_required"]),
            "replay_scenarios": sum(1 for s in scenarios.values() if s["replay_evaluation_only"]),
        },
    }
    return manifest


def main() -> None:
    print("=" * 70)
    print("Stage 3 Robustness Scenario Generator")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = generate_scenarios()
    out_path = OUTPUT_DIR / "scenario_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Generated scenario manifest: {out_path} ({out_path.stat().st_size} bytes)")
    print(f"Total scenarios: {manifest['scenario_counts']['total_scenarios']}")
    print(f"  Solver scenarios (require optimization): {manifest['scenario_counts']['solver_scenarios']}")
    print(f"  Replay scenarios (nutrient sensitivity): {manifest['scenario_counts']['replay_scenarios']}")


if __name__ == "__main__":
    main()
