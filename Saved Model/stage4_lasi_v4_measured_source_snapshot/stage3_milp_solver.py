#!/usr/bin/env python3
"""BioNutriGPT Stage 3: Discrete Indian Meal Synthesis (MILP / HiGHS).

The engine translates the continuous nutrient envelope emitted by BioNutriGPT
Stage 2 into a discrete basket of Indian cooked dishes. It uses SciPy's
``scipy.optimize.milp`` (HiGHS), integer serving counts, binary food-selection
variables, and two-sided L1 goal-programming slacks so conflicting nutrient
targets do not make the model artificially infeasible.

Food-data provenance
--------------------
When an external IFCT/RecipeDB-derived CSV, Parquet, or JSON file is supplied,
that file is used. The embedded catalog is a transparent *synthetic fallback*
with plausible illustrative per-serving estimates; it is NOT a reproduction
of the copyrighted/official ICMR-NIN IFCT 2017 or RecipeDB tables and must not
be treated as a clinical food-composition reference.

Python 3.10+. Dependencies: numpy, scipy, pandas (CSV/Parquet loading).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from scipy.sparse import vstack


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_DIR / "stage2_counterfactual_prescription.json"
DEFAULT_OUTPUT = PROJECT_DIR / "stage3_indian_meal_plan.json"
DEFAULT_TIME_LIMIT_SEC = 1.50
MAX_TIME_LIMIT_SEC = 3.0
MAX_SERVINGS_PER_DISH = 3
MIN_ACTIVE_DISHES = 4
MAX_ACTIVE_DISHES = 8

# Exactly the 39 Stage 2 nutrient fields (the seven derived ratio features are
# deliberately excluded: they are recalculated upstream and are not foods).
NUTRIENT_CODES: tuple[str, ...] = (
    "DR1TKCAL", "DR1TPROT", "DR1TCARB", "DR1TSUGR", "DR1TFIBE",
    "DR1TTFAT", "DR1TSFAT", "DR1TMFAT", "DR1TPFAT", "DR1TCHOL",
    "DR1TATOC", "DR1TRET", "DR1TVARA", "DR1TACAR", "DR1TBCAR",
    "DR1TLYCO", "DR1TLZ", "DR1TVB1", "DR1TVB2", "DR1TNIAC",
    "DR1TVB6", "DR1TFOLA", "DR1TFA", "DR1TFF", "DR1TFDFE",
    "DR1TVB12", "DR1TVC", "DR1TVK", "DR1TCALC", "DR1TPHOS",
    "DR1TMAGN", "DR1TIRON", "DR1TZINC", "DR1TCOPP", "DR1TSODI",
    "DR1TPOTA", "DR1TSELE", "DR1TCAFF", "DR1TALCO",
)
NUTRIENT_INDEX = {name: i for i, name in enumerate(NUTRIENT_CODES)}

VALID_CATEGORIES = {
    "dal_pulse", "cereal_grain", "vegetable_sabzi", "dairy", "fruit", "oil_fat", "non_veg_protein"
}
CATEGORY_ALIASES = {
    "dal": "dal_pulse", "pulse": "dal_pulse", "pulses": "dal_pulse",
    "grain": "cereal_grain", "cereal": "cereal_grain", "bread": "cereal_grain",
    "vegetable": "vegetable_sabzi", "sabzi": "vegetable_sabzi", "veg": "vegetable_sabzi",
    "curd": "dairy", "milk": "dairy", "fruit": "fruit", "oil": "oil_fat",
    "non_veg": "non_veg_protein", "nonveg": "non_veg_protein", "meat": "non_veg_protein",
    "fish": "non_veg_protein", "egg": "non_veg_protein", "poultry": "non_veg_protein",
}


@dataclass(frozen=True)
class FoodItem:
    """One cooked dish and its nutrient content per household serving."""

    food_id: str
    name: str
    category: str
    serving_unit: str
    serving_weight_g: float
    nutrients_per_serving: Mapping[str, float]
    selection_cost: float = 1.0
    data_source: str = "external"
    unknown_nutrients: tuple[str, ...] = ()
    catalog_version: str = "legacy"
    allergen_info: Mapping[str, Any] = field(default_factory=dict)
    ingredient_codes: tuple[str, ...] = ()
    dietary_type: str = "pure_vegetarian"

    def __post_init__(self) -> None:
        category = CATEGORY_ALIASES.get(self.category.strip().lower(), self.category.strip().lower())
        object.__setattr__(self, "category", category)
        if not self.food_id.strip() or not self.name.strip():
            raise ValueError("FoodItem food_id and name must be non-empty.")
        if category not in VALID_CATEGORIES:
            raise ValueError(f"Unsupported category {category!r} for {self.food_id}.")
        if not math.isfinite(float(self.serving_weight_g)) or self.serving_weight_g <= 0:
            raise ValueError(f"{self.food_id}: serving_weight_g must be positive and finite.")
        if not math.isfinite(float(self.selection_cost)) or self.selection_cost < 0:
            raise ValueError(f"{self.food_id}: selection_cost must be finite and >= 0.")
        missing = sorted(set(NUTRIENT_CODES) - set(self.nutrients_per_serving))
        extra = sorted(set(self.nutrients_per_serving) - set(NUTRIENT_CODES))
        if missing:
            raise ValueError(f"{self.food_id}: missing nutrient fields: {missing}")
        if extra:
            raise ValueError(f"{self.food_id}: unexpected nutrient fields: {extra}")
        clean: dict[str, float] = {}
        for nutrient in NUTRIENT_CODES:
            value = float(self.nutrients_per_serving[nutrient])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{self.food_id}: {nutrient} must be finite and >= 0; got {value}.")
            clean[nutrient] = value
        object.__setattr__(self, "nutrients_per_serving", clean)


@dataclass(frozen=True)
class Stage3Input:
    """Validated Stage 2 bounds and targets required by the MILP."""

    patient_metadata: Mapping[str, Any]
    energy_prescription: Mapping[str, float]
    targets: Mapping[str, float]
    lower_bounds: Mapping[str, float]
    upper_bounds: Mapping[str, float]
    baselines: Mapping[str, float]
    rule_metadata: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


@dataclass
class Stage3Output:
    """Serializable result returned by :func:`solve_meal_plan`."""

    solved_food_basket: list[dict[str, Any]]
    delivered_nutrition: list[dict[str, Any]]
    clinical_verification: dict[str, Any]
    solver_metadata: dict[str, Any]
    accepted: bool = False
    failure: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "failure": self.failure,
            "solved_food_basket": self.solved_food_basket,
            "delivered_nutrition": self.delivered_nutrition,
            "clinical_verification": self.clinical_verification,
            "solver_metadata": self.solver_metadata,
        }


# ---------------------------------------------------------------------------
# Embedded illustrative fallback catalog
# ---------------------------------------------------------------------------
# The 29 micro/secondary nutrient values below are estimated per serving.
# All values are non-negative and use the units of the Stage 1 DR1T* columns.
_MICRO_DEFAULTS: dict[str, float] = {
    "DR1TATOC": 0.5, "DR1TRET": 0.0, "DR1TVARA": 20.0,
    "DR1TACAR": 20.0, "DR1TBCAR": 100.0, "DR1TLYCO": 20.0,
    "DR1TLZ": 50.0, "DR1TVB1": 0.08, "DR1TVB2": 0.08,
    "DR1TNIAC": 0.8, "DR1TVB6": 0.10, "DR1TFOLA": 25.0,
    "DR1TFA": 0.0, "DR1TFF": 20.0, "DR1TFDFE": 25.0,
    "DR1TVB12": 0.05, "DR1TVC": 2.0, "DR1TVK": 8.0,
    "DR1TCALC": 35.0, "DR1TPHOS": 80.0, "DR1TMAGN": 25.0,
    "DR1TIRON": 1.0, "DR1TZINC": 0.5, "DR1TCOPP": 0.12,
    "DR1TSODI": 100.0, "DR1TPOTA": 180.0, "DR1TSELE": 3.0,
    "DR1TCAFF": 0.0, "DR1TALCO": 0.0,
}


def _fallback_food(
    food_id: str, name: str, category: str, unit: str, grams: float,
    kcal: float, protein: float, carbs: float, sugar: float, fiber: float,
    fat: float, sat_fat: float, mono_fat: float, poly_fat: float,
    cholesterol: float = 0.0, *, micros: Optional[Mapping[str, float]] = None,
    sodium: Optional[float] = None, potassium: Optional[float] = None,
    cost: float = 1.0,
) -> FoodItem:
    nutrients = {
        "DR1TKCAL": kcal, "DR1TPROT": protein, "DR1TCARB": carbs,
        "DR1TSUGR": sugar, "DR1TFIBE": fiber, "DR1TTFAT": fat,
        "DR1TSFAT": sat_fat, "DR1TMFAT": mono_fat, "DR1TPFAT": poly_fat,
        "DR1TCHOL": cholesterol,
        **_MICRO_DEFAULTS,
    }
    if micros:
        unknown = set(micros) - set(_MICRO_DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown fallback micro nutrients: {sorted(unknown)}")
        nutrients.update({k: float(v) for k, v in micros.items()})
    if sodium is not None:
        nutrients["DR1TSODI"] = sodium
    if potassium is not None:
        nutrients["DR1TPOTA"] = potassium
    nutrients["DR1TVB12"] = 0.0  # Known subtotal only; unknown flag below is authoritative.
    return FoodItem(
        food_id, name, category, unit, grams, nutrients,
        selection_cost=cost, data_source="synthetic_fallback_estimate",
        unknown_nutrients=("DR1TVB12",),
    )


def _build_fallback_catalog() -> list[FoodItem]:
    """Return 32 illustrative cooked Indian dishes (not official IFCT values)."""
    f = _fallback_food
    foods = [
        # Dals and pulses
        f("dal_moong_tadka", "Moong Dal Tadka", "dal_pulse", "1 Katori (150 g)", 150, 190, 11.5, 27, 2.0, 6.5, 4.5, 0.8, 1.4, 1.8, sodium=420, potassium=430, micros={"DR1TFOLA":90,"DR1TFF":75,"DR1TFDFE":110,"DR1TIRON":2.1,"DR1TMAGN":58,"DR1TPHOS":165,"DR1TZINC":1.3,"DR1TVC":4}),
        f("dal_toor", "Toor Dal", "dal_pulse", "1 Katori (150 g)", 150, 205, 11.0, 30, 2.2, 6.0, 5.0, 0.9, 1.5, 2.0, sodium=390, potassium=410, micros={"DR1TFOLA":85,"DR1TIRON":2.0,"DR1TMAGN":55,"DR1TPHOS":170}),
        f("dal_masoor", "Masoor Dal", "dal_pulse", "1 Katori (150 g)", 150, 185, 12.0, 26, 1.8, 7.0, 3.5, 0.5, 1.0, 1.6, sodium=360, potassium=450, micros={"DR1TFOLA":105,"DR1TIRON":3.0,"DR1TMAGN":62,"DR1TPHOS":180}),
        f("chana_masala", "Chana Masala", "dal_pulse", "1 Katori (150 g)", 150, 240, 12.0, 34, 5.0, 8.0, 7.0, 1.0, 3.0, 2.2, sodium=260, potassium=480, micros={"DR1TFOLA":95,"DR1TIRON":2.6,"DR1TMAGN":65,"DR1TPHOS":175,"DR1TZINC":1.5}),
        f("rajma_curry", "Rajma Curry", "dal_pulse", "1 Katori (150 g)", 150, 220, 12.5, 32, 2.5, 8.0, 5.0, 0.8, 1.5, 2.1, sodium=430, potassium=500, micros={"DR1TFOLA":100,"DR1TIRON":2.4,"DR1TMAGN":70,"DR1TPHOS":190}),
        f("sprouted_moong_usal", "Sprouted Moong Usal", "dal_pulse", "1 Katori (150 g)", 150, 165, 10.0, 25, 3.0, 7.0, 2.5, 0.4, 0.7, 1.0, sodium=120, potassium=520, micros={"DR1TVC":18,"DR1TFOLA":80,"DR1TIRON":2.2}),
        # Cereals and breads
        f("phulka", "Whole-Wheat Phulka", "cereal_grain", "1 Phulka (35 g)", 35, 105, 3.5, 20, 0.5, 3.0, 1.5, 0.2, 0.4, 0.7, sodium=35, potassium=90, micros={"DR1TVB1":0.12,"DR1TNIAC":1.0,"DR1TMAGN":20,"DR1TIRON":1.0}),
        f("brown_rice", "Cooked Brown Rice", "cereal_grain", "1/2 Cup (100 g)", 100, 125, 2.7, 26, 0.2, 1.8, 1.0, 0.2, 0.3, 0.3, sodium=5, potassium=85, micros={"DR1TVB1":0.10,"DR1TMAGN":44,"DR1TPHOS":100,"DR1TSELE":5}),
        f("steamed_rice", "Steamed White Rice", "cereal_grain", "1/2 Cup (100 g)", 100, 130, 2.4, 28, 0.1, 0.4, 0.3, 0.1, 0.1, 0.1, sodium=2, potassium=30, micros={"DR1TVB1":0.03,"DR1TNIAC":0.4}),
        f("jowar_roti", "Jowar Roti", "cereal_grain", "1 Roti (45 g)", 45, 125, 3.5, 25, 0.5, 2.5, 1.8, 0.3, 0.4, 0.8, sodium=8, potassium=110, micros={"DR1TMAGN":35,"DR1TIRON":1.4,"DR1TPHOS":95}),
        f("bajra_roti", "Bajra Roti", "cereal_grain", "1 Roti (45 g)", 45, 135, 3.8, 25, 0.5, 2.8, 2.2, 0.4, 0.6, 0.9, sodium=8, potassium=120, micros={"DR1TMAGN":40,"DR1TIRON":1.6,"DR1TZINC":0.8}),
        f("vegetable_oats_upma", "Vegetable Oats Upma", "cereal_grain", "1 Cup (180 g)", 180, 210, 6.0, 34, 3.0, 5.0, 5.5, 1.0, 2.0, 1.8, sodium=280, potassium=260, micros={"DR1TVC":12,"DR1TMAGN":45,"DR1TFOLA":35}),
        f("idli", "Steamed Idli", "cereal_grain", "2 Idli (120 g)", 120, 150, 4.5, 30, 1.0, 2.0, 1.0, 0.2, 0.3, 0.4, sodium=230, potassium=75, micros={"DR1TVB1":0.08,"DR1TFOLA":25}),
        f("dosa_plain", "Plain Dosa", "cereal_grain", "1 Dosa (100 g)", 100, 185, 4.0, 29, 1.0, 1.5, 5.5, 1.0, 2.0, 2.0, sodium=260, potassium=100),
        f("vegetable_pulao", "Vegetable Pulao", "cereal_grain", "1 Cup (180 g)", 180, 245, 5.0, 42, 3.0, 3.5, 6.0, 1.2, 2.5, 1.8, sodium=180, potassium=260, micros={"DR1TVC":15,"DR1TVARA":100,"DR1TVK":25}),
        # Vegetable dishes
        f("palak_sabzi", "Palak Sabzi", "vegetable_sabzi", "1 Katori (120 g)", 120, 95, 4.0, 10, 2.0, 4.0, 4.0, 0.6, 1.0, 1.8, sodium=180, potassium=560, micros={"DR1TVARA":520,"DR1TBCAR":2800,"DR1TVK":430,"DR1TVC":22,"DR1TFOLA":95,"DR1TIRON":2.5,"DR1TCALC":115}),
        f("bhindi_masala", "Bhindi Masala", "vegetable_sabzi", "1 Katori (120 g)", 120, 115, 3.0, 12, 3.0, 4.0, 6.0, 0.8, 3.0, 1.8, sodium=120, potassium=320, micros={"DR1TVC":18,"DR1TVARA":45,"DR1TVK":35,"DR1TFOLA":35}),
        f("mixed_veg_sabzi", "Mixed Vegetable Sabzi", "vegetable_sabzi", "1 Katori (150 g)", 150, 110, 3.5, 15, 4.0, 5.0, 4.0, 0.7, 1.5, 1.4, sodium=190, potassium=410, micros={"DR1TVARA":260,"DR1TBCAR":1500,"DR1TVC":28,"DR1TVK":45}),
        f("lauki_tomato", "Lauki Tomato Sabzi", "vegetable_sabzi", "1 Katori (150 g)", 150, 75, 2.0, 10, 3.0, 3.0, 3.0, 0.4, 1.2, 1.0, sodium=160, potassium=300, micros={"DR1TVC":10,"DR1TVARA":55}),
        f("aloo_gobi", "Aloo Gobi", "vegetable_sabzi", "1 Katori (150 g)", 150, 135, 3.0, 19, 3.0, 4.0, 5.0, 0.8, 2.0, 1.6, sodium=130, potassium=420, micros={"DR1TVC":30,"DR1TVARA":40,"DR1TVK":25}),
        f("baingan_bharta", "Baingan Bharta", "vegetable_sabzi", "1 Katori (150 g)", 150, 105, 2.5, 12, 4.0, 5.0, 5.0, 0.7, 2.5, 1.4, sodium=190, potassium=350, micros={"DR1TVC":8,"DR1TFOLA":25}),
        f("cabbage_peas", "Cabbage Peas Sabzi", "vegetable_sabzi", "1 Katori (150 g)", 150, 105, 4.0, 15, 4.0, 5.0, 3.0, 0.5, 1.0, 1.0, sodium=170, potassium=360, micros={"DR1TVC":35,"DR1TVK":55,"DR1TFOLA":45}),
        f("cucumber_salad", "Cucumber Carrot Salad", "vegetable_sabzi", "1 Bowl (150 g)", 150, 55, 1.5, 11, 5.0, 3.0, 0.5, 0.1, 0.1, 0.2, sodium=35, potassium=280, micros={"DR1TVC":12,"DR1TVARA":240,"DR1TBCAR":1500}),
        # Dairy
        f("skim_curd", "Skim-Milk Curd", "dairy", "1 Katori (150 g)", 150, 90, 7.5, 10, 9.0, 0.0, 0.5, 0.2, 0.1, 0.0, cholesterol=3, sodium=95, potassium=230, micros={"DR1TVB2":0.20,"DR1TVB12":0.55,"DR1TCALC":180,"DR1TPHOS":140}),
        f("lowfat_curd", "Low-Fat Curd", "dairy", "1 Katori (150 g)", 150, 115, 7.0, 9.0, 8.0, 0.0, 4.5, 2.8, 1.2, 0.2, cholesterol=12, sodium=100, potassium=220, micros={"DR1TVB2":0.18,"DR1TVB12":0.50,"DR1TCALC":160,"DR1TPHOS":135}),
        f("lowfat_paneer", "Low-Fat Paneer Bhurji", "dairy", "1 Katori (120 g)", 120, 190, 24.0, 7.0, 3.0, 1.0, 9.0, 4.5, 3.0, 0.7, cholesterol=35, sodium=180, potassium=240, micros={"DR1TVB2":0.22,"DR1TVB12":0.65,"DR1TCALC":260,"DR1TPHOS":220}),
        # Fruit
        f("guava", "Fresh Guava", "fruit", "1 Medium (100 g)", 100, 68, 2.6, 14, 9.0, 5.4, 1.0, 0.2, 0.1, 0.3, sodium=2, potassium=417, micros={"DR1TVC":228,"DR1TVARA":31,"DR1TFOLA":49}),
        f("apple", "Apple with Skin", "fruit", "1 Medium (150 g)", 150, 78, 0.4, 21, 16.0, 3.6, 0.3, 0.1, 0.0, 0.1, sodium=2, potassium=160, micros={"DR1TVC":7,"DR1TVK":4}),
        f("orange", "Fresh Orange", "fruit", "1 Medium (130 g)", 130, 62, 1.2, 15, 12.0, 3.1, 0.2, 0.0, 0.0, 0.1, sodium=0, potassium=237, micros={"DR1TVC":70,"DR1TFOLA":40}),
        f("banana", "Small Banana", "fruit", "1 Small (100 g)", 100, 89, 1.1, 23, 12.0, 2.6, 0.3, 0.1, 0.0, 0.1, sodium=1, potassium=358, micros={"DR1TVB6":0.37,"DR1TVC":9}),
        # Oils / fats (optional; selection penalty helps avoid unnecessary use)
        f("mustard_oil", "Mustard Oil", "oil_fat", "1 tsp (5 g)", 5, 45, 0.0, 0.0, 0.0, 0.0, 5.0, 0.3, 2.8, 1.4, micros={"DR1TATOC":0.4}, cost=1.5),
        f("groundnut_oil", "Groundnut Oil", "oil_fat", "1 tsp (5 g)", 5, 45, 0.0, 0.0, 0.0, 0.0, 5.0, 1.0, 2.5, 1.4, micros={"DR1TATOC":0.7}, cost=1.5),
    ]
    # Ensure IDs are unique and the fallback covers all required categories.
    ids = [item.food_id for item in foods]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate food IDs in embedded fallback catalog.")
    return foods


# ---------------------------------------------------------------------------
# Input and food database validation
# ---------------------------------------------------------------------------
def _unwrap_stage2_input(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Accept a Stage 2 dossier directly or within the unified pipeline wrapper."""
    candidate: Mapping[str, Any] = payload
    nested = payload.get("stage2_counterfactual_prescription")
    if isinstance(nested, Mapping):
        candidate = {**nested}
        if "patient_metadata" not in candidate and isinstance(payload.get("patient_metadata"), Mapping):
            candidate["patient_metadata"] = payload["patient_metadata"]
    return candidate


def _parse_stage2_input(stage2_input: Mapping[str, Any]) -> Stage3Input:
    if not isinstance(stage2_input, Mapping):
        raise TypeError("stage2_input must be a mapping/dict.")
    dossier = _unwrap_stage2_input(stage2_input)
    metadata = dossier.get("patient_metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("patient_metadata must be a JSON object.")
    energy = dossier.get("energy_prescription")
    if not isinstance(energy, Mapping):
        raise ValueError("Stage 2 dossier is missing energy_prescription.")
    for key in ("BMR", "TDEE", "target_calories"):
        if key not in energy:
            raise ValueError(f"energy_prescription is missing {key!r}.")
        val = float(energy[key])
        if not math.isfinite(val) or val <= 0:
            raise ValueError(f"energy_prescription[{key!r}] must be positive and finite.")

    records = dossier.get("nutrient_constraint_envelope")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError("nutrient_constraint_envelope must be a list of 39 records.")
    if len(records) != len(NUTRIENT_CODES):
        raise ValueError(f"Expected exactly 39 nutrient records; received {len(records)}.")

    targets: dict[str, float] = {}
    lower: dict[str, float] = {}
    upper: dict[str, float] = {}
    baselines: dict[str, float] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("Each nutrient_constraint_envelope entry must be an object.")
        name = str(record.get("nutrient", ""))
        if name not in NUTRIENT_INDEX:
            raise ValueError(f"Unknown Stage 2 nutrient code: {name!r}.")
        if name in targets:
            raise ValueError(f"Duplicate Stage 2 nutrient record: {name}.")
        for key in ("target", "min_bound", "max_bound", "baseline"):
            if key not in record:
                raise ValueError(f"Nutrient {name} is missing {key!r}.")
        target, low, high, baseline = (float(record[k]) for k in ("target", "min_bound", "max_bound", "baseline"))
        if not all(math.isfinite(v) for v in (target, low, high, baseline)):
            raise ValueError(f"Nutrient {name} contains non-finite values.")
        if low < 0 or high < low or target < 0 or baseline < 0:
            raise ValueError(f"Nutrient {name} has invalid non-negative bounds/values.")
        targets[name], lower[name], upper[name], baselines[name] = target, low, high, baseline
    missing = sorted(set(NUTRIENT_CODES) - set(targets))
    if missing:
        raise ValueError(f"Stage 2 envelope is missing nutrients: {missing}")

    # The energy-prescription target is the authoritative primary calorie goal.
    target_kcal = float(energy["target_calories"])
    if "DR1TKCAL" in targets and targets["DR1TKCAL"] > 0:
        # Preserve the upstream target record but use the explicit energy goal
        # consistently for the primary absolute-deviation objective/report.
        targets["DR1TKCAL"] = target_kcal
    return Stage3Input(
        patient_metadata=dict(metadata),
        energy_prescription={k: float(energy[k]) for k in ("BMR", "TDEE", "target_calories")},
        targets=targets, lower_bounds=lower, upper_bounds=upper, baselines=baselines,
        rule_metadata={str(row['nutrient']): dict(row.get('rule_provenance', {})) for row in records},
    )


def _food_from_mapping(row: Mapping[str, Any], source: str) -> FoodItem:
    import ast

    nutrient_raw = row.get("nutrients_per_serving")
    if isinstance(nutrient_raw, str):
        try:
            nutrient_raw = json.loads(nutrient_raw)
        except json.JSONDecodeError:
            try:
                nutrient_raw = ast.literal_eval(nutrient_raw)
            except (ValueError, SyntaxError) as exc:
                raise ValueError(f"Invalid nutrients_per_serving JSON for {row.get('food_id')!r}.") from exc
    if not isinstance(nutrient_raw, Mapping):
        # Also accept a flat tabular representation with one column per nutrient.
        nutrient_raw = {code: row[code] for code in NUTRIENT_CODES if code in row and row[code] is not None}
    selection_cost_raw = row.get("selection_cost", 1.0)
    try:
        selection_cost_value = float(selection_cost_raw)
        if not math.isfinite(selection_cost_value):
            selection_cost_value = 1.0
    except (TypeError, ValueError):
        selection_cost_value = 1.0
    unknown = tuple(code for code in NUTRIENT_CODES
                    if nutrient_raw.get(code) is None
                    or (isinstance(nutrient_raw.get(code), float) and math.isnan(nutrient_raw[code])))
    ing_codes = tuple(sorted({str(ing.get("food_code", "")).strip() for ing in row.get("ingredients", []) if isinstance(ing, Mapping) and ing.get("food_code")}))
    allergen_info = dict(row.get("allergen_info", {})) if isinstance(row.get("allergen_info"), Mapping) else {}
    return FoodItem(
        food_id=str(row["food_id"]),
        name=str(row["name"]),
        category=str(row["category"]),
        serving_unit=str(row["serving_unit"]),
        serving_weight_g=float(row["serving_weight_g"]),
        # Unknown contributions have a zero *known subtotal* for optimization.
        # They remain explicit and block adequacy verification downstream.
        nutrients_per_serving={code: 0.0 if code in unknown else float(nutrient_raw[code]) for code in NUTRIENT_CODES},
        selection_cost=selection_cost_value,
        data_source=source,
        unknown_nutrients=unknown,
        catalog_version=str(row.get("catalog_version", "legacy")),
        allergen_info=allergen_info,
        ingredient_codes=ing_codes,
        dietary_type=str(row.get("dietary_type", "non_vegetarian" if str(row.get("category", "")).strip().lower() == "non_veg_protein" else "pure_vegetarian")),
    )


def load_food_database(path: Optional[str] = None) -> list[FoodItem]:
    """Load an external IFCT/RecipeDB-style food table or use embedded fallback.

    Accepted input formats: CSV, Parquet/PQ, and JSON (array of food objects).
    Each record must provide all 39 values in ``nutrients_per_serving`` or as
    flat columns named by the nutrient codes.
    """
    candidate: Optional[Path] = Path(path).expanduser() if path else None
    if candidate is None:
        for possible in (
            PROJECT_DIR / "food_composition_cooked_sourced.json",
            PROJECT_DIR / "food_composition_ifct2017.csv",
            PROJECT_DIR / "recipe_db_foods.parquet",
            PROJECT_DIR / "food_composition_ifct2017.parquet",
            PROJECT_DIR / "food_composition_ifct2017.json",
        ):
            if possible.is_file():
                candidate = possible
                break
    if candidate is None:
        return _build_fallback_catalog()
    if not candidate.is_file():
        raise FileNotFoundError(f"Food database file not found: {candidate}")

    suffix = candidate.suffix.lower()
    if suffix == ".json":
        with candidate.open("r", encoding="utf-8") as handle:
            rows = json.load(handle)
    elif suffix == ".csv":
        import pandas as pd
        rows = pd.read_csv(candidate).to_dict(orient="records")
    elif suffix in {".parquet", ".pq"}:
        import pandas as pd
        rows = pd.read_parquet(candidate).to_dict(orient="records")
    else:
        raise ValueError("Food database must be .csv, .parquet/.pq, or .json.")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Food database {candidate} must contain a non-empty array/table.")
    foods = [_food_from_mapping(row, source=str(candidate)) for row in rows]
    ids = [food.food_id for food in foods]
    if len(ids) != len(set(ids)):
        raise ValueError("Food database contains duplicate food_id values.")
    _validate_catalog_categories(foods)
    return foods


def _validate_catalog_categories(
    food_db: Sequence[FoodItem],
    dietary_mode: str = "pure_vegetarian",
    allow_zero_dairy: bool = False,
) -> None:
    ids = [food.food_id for food in food_db]
    if len(ids) < 5:
        raise ValueError("Food database needs at least five foods to satisfy category rules.")
    if len(ids) != len(set(ids)):
        raise ValueError("Food database food_id values must be unique.")
    categories = {food.category for food in food_db}
    cat_limits, _ = _category_rules_for_mode(dietary_mode, allow_zero_dairy=allow_zero_dairy)
    required = {cat for cat, (low, _) in cat_limits.items() if low > 0}
    if not required.issubset(categories):
        raise ValueError(f"Food database is missing required categories: {sorted(required - categories)}")


def _category_rules_for_mode(
    dietary_mode: str,
    allow_zero_dairy: bool = False,
) -> tuple[dict[str, tuple[int, float]], dict[str, tuple[float, float]]]:
    """Return (category_variety_limits, category_serving_limits) for the chosen dietary mode."""
    dairy_cat_min = 0 if allow_zero_dairy else 1
    dairy_serv_min = 0.0 if allow_zero_dairy else 1.0
    if dietary_mode == "pure_non_vegetarian":
        return (
            {
                "non_veg_protein": (1, 3),
                "cereal_grain": (1, 3),
                "vegetable_sabzi": (2, np.inf),
                "dal_pulse": (0, 2),
                "dairy": (0, np.inf),
            },
            {
                "non_veg_protein": (2.0, 4.0),
                "cereal_grain": (3.0, 5.0),
                "vegetable_sabzi": (2.0, 4.0),
                "dal_pulse": (0.0, 3.0),
                "dairy": (0.0, 2.0),
            },
        )
    if dietary_mode == "veg_nonveg_combo":
        return (
            {
                "non_veg_protein": (1, 2),
                "dal_pulse": (1, 2),
                "cereal_grain": (1, 3),
                "vegetable_sabzi": (2, np.inf),
                "dairy": (dairy_cat_min, np.inf),
            },
            {
                "non_veg_protein": (1.0, 3.0),
                "dal_pulse": (1.0, 3.0),
                "cereal_grain": (3.0, 5.0),
                "vegetable_sabzi": (2.0, 4.0),
                "dairy": (dairy_serv_min, 2.0),
            },
        )
    return (
        {
            "dal_pulse": (1, 2),
            "cereal_grain": (1, 3),
            "vegetable_sabzi": (2, np.inf),
            "dairy": (dairy_cat_min, np.inf),
        },
        {
            "cereal_grain": (3.0, 5.0),
            "dal_pulse": (2.0, 3.0),
            "vegetable_sabzi": (2.0, 4.0),
            "dairy": (dairy_serv_min, 2.0),
        },
    )


# ---------------------------------------------------------------------------
# MILP construction and optimization
# ---------------------------------------------------------------------------
def _nutrient_priority(name: str) -> float:
    priorities = {
        "DR1TKCAL": 100.0,
        "DR1TPROT": 28.0,
        "DR1TCARB": 8.0,
        "DR1TSUGR": 12.0,
        "DR1TFIBE": 25.0,
        "DR1TTFAT": 8.0,
        "DR1TSFAT": 22.0,
        "DR1TSODI": 30.0,
        "DR1TPOTA": 25.0,
        "DR1TCALC": 5.0,
        "DR1TIRON": 5.0,
        "DR1TMAGN": 5.0,
        "DR1TZINC": 4.0,
        "DR1TVC": 4.0,
        "DR1TVARA": 4.0,
    }
    return priorities.get(name, 1.0)


def _make_sparse_constraints(
    foods: Sequence[FoodItem], stage: Stage3Input, *, gamma: float,
    dietary_mode: str = "pure_vegetarian",
    allow_zero_dairy: bool = False,
) -> tuple[np.ndarray, np.ndarray, Bounds, LinearConstraint, dict[str, Any]]:
    m, n = len(foods), len(NUTRIENT_CODES)
    # Variable layout: [q_0..q_M-1, z_0..z_M-1, e+_0..e+_38,
    #                   e-_0..e-_38, calorie_abs_over, calorie_abs_under].
    q0 = 0
    z0 = m
    ep0 = 2 * m
    em0 = ep0 + n
    cal_over = em0 + n
    cal_under = cal_over + 1
    nvars = cal_under + 1

    F = np.asarray(
        [[float(food.nutrients_per_serving[code]) for food in foods] for code in NUTRIENT_CODES],
        dtype=np.float64,
    )
    if not np.isfinite(F).all() or np.any(F < 0):
        raise ValueError("Food nutrient matrix contains invalid values.")

    c = np.zeros(nvars, dtype=np.float64)
    lower_var = np.zeros(nvars, dtype=np.float64)
    upper_var = np.full(nvars, np.inf, dtype=np.float64)
    integrality = np.zeros(nvars, dtype=np.int32)
    # Serving counts q_i in {0,1,2,3}; binary selection z_i in {0,1}.
    upper_var[q0:q0 + m] = MAX_SERVINGS_PER_DISH
    upper_var[z0:z0 + m] = 1.0
    integrality[q0:q0 + m] = 1
    integrality[z0:z0 + m] = 1
    c[z0:z0 + m] = gamma * np.asarray([food.selection_cost for food in foods])

    # L1 goal-programming penalties are normalized by a nutrient-specific scale
    # to avoid mg/mcg units overwhelming kcal/g variables.
    slack_weights: dict[str, float] = {}
    for k, code in enumerate(NUTRIENT_CODES):
        width = stage.upper_bounds[code] - stage.lower_bounds[code]
        scale = max(width, abs(stage.targets[code]) * 0.10, 1.0)
        weight = _nutrient_priority(code) / scale
        if stage.rule_metadata.get(code, {}).get('planning_role') in {'report_only', 'exposure_reporting'}:
            weight = 0.0
        c[ep0 + k] = weight
        c[em0 + k] = weight
        slack_weights[code] = weight

    target_kcal = stage.energy_prescription["target_calories"]
    # Strong primary objective: absolute calorie deviation as a fraction of goal.
    calorie_deviation_weight = 150.0 / max(target_kcal, 1.0)
    c[cal_over] = calorie_deviation_weight
    c[cal_under] = calorie_deviation_weight

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    lb: list[float] = []
    ub: list[float] = []
    row = 0

    def add_row(entries: Mapping[int, float], low: float, high: float) -> None:
        nonlocal row
        for col, value in entries.items():
            if value != 0.0:
                rows.append(row); cols.append(int(col)); data.append(float(value))
        lb.append(float(low)); ub.append(float(high)); row += 1

    # Nutrient envelope: L_k <= F_k q + e^-_k - e^+_k <= U_k.
    for k, code in enumerate(NUTRIENT_CODES):
        if stage.rule_metadata.get(code, {}).get('planning_role') in {'report_only', 'exposure_reporting'}:
            continue
        entries = {q0 + i: F[k, i] for i in range(m) if F[k, i] != 0}
        entries[ep0 + k] = -1.0
        entries[em0 + k] = 1.0
        add_row(entries, stage.lower_bounds[code], stage.upper_bounds[code])

    # Binary linking in both directions: q <= 3z and z <= q.
    for i in range(m):
        add_row({q0 + i: 1.0, z0 + i: -float(MAX_SERVINGS_PER_DISH)}, -np.inf, 0.0)
        add_row({z0 + i: 1.0, q0 + i: -1.0}, -np.inf, 0.0)

    # Overall active dish diversity.
    add_row({z0 + i: 1.0 for i in range(m)}, MIN_ACTIVE_DISHES, MAX_ACTIVE_DISHES)

    # Category variety & plate composition constraints by dietary_mode.
    category_limits, serving_limits = _category_rules_for_mode(dietary_mode, allow_zero_dairy=allow_zero_dairy)
    for category, (low, high) in category_limits.items():
        indices = [i for i, food in enumerate(foods) if food.category == category]
        if not indices:
            if low > 0:
                raise ValueError(f"No foods available for required category {category!r}.")
            continue
        add_row({z0 + i: 1.0 for i in indices}, low, high)

    for category, (low_q, high_q) in serving_limits.items():
        indices = [i for i, food in enumerate(foods) if food.category == category]
        if indices:
            add_row({q0 + i: 1.0 for i in indices}, low_q, high_q)

    # Absolute target calorie deviation: F_kcal q - over + under = target.
    kcal_idx = NUTRIENT_INDEX["DR1TKCAL"]
    calorie_entries = {q0 + i: F[kcal_idx, i] for i in range(m) if F[kcal_idx, i] != 0}
    calorie_entries[cal_over] = -1.0
    calorie_entries[cal_under] = 1.0
    add_row(calorie_entries, target_kcal, target_kcal)

    A = coo_matrix((np.asarray(data), (np.asarray(rows), np.asarray(cols))), shape=(row, nvars)).tocsr()
    constraints = LinearConstraint(A, np.asarray(lb), np.asarray(ub))
    return c, integrality, Bounds(lower_var, upper_var), constraints, {
        "food_matrix": F,
        "indices": {"q": q0, "z": z0, "e_plus": ep0, "e_minus": em0,
                   "calorie_over": cal_over, "calorie_under": cal_under},
        "slack_weights": slack_weights,
        "calorie_deviation_weight": calorie_deviation_weight,
        "variable_count": nvars,
        "constraint_count": row,
    }


def _heuristic_incumbent(foods: Sequence[FoodItem], target_kcal: float) -> np.ndarray:
    """Fast deterministic fallback if HiGHS stops before finding an incumbent."""
    m = len(foods)
    q = np.zeros(m, dtype=np.int32)
    # Balanced starter plate: 2 cereals, 2 dals, 2 veg dishes, 1 dairy
    required_categories = ["cereal_grain", "cereal_grain", "dal_pulse", "dal_pulse", "vegetable_sabzi", "vegetable_sabzi", "dairy"]
    selected: set[int] = set()
    for category in required_categories:
        candidates = [i for i, f in enumerate(foods) if f.category == category and i not in selected]
        if not candidates:
            continue
        # Pick a dish with a moderate calorie density and low selection cost.
        best = min(candidates, key=lambda i: (foods[i].selection_cost, abs(foods[i].nutrients_per_serving["DR1TKCAL"] - target_kcal / 7.0)))
        q[best] = 1
        selected.add(best)

    def kcal_total() -> float:
        return float(sum(q[i] * foods[i].nutrients_per_serving["DR1TKCAL"] for i in range(m)))

    category_maxima = {"dal_pulse": 2, "cereal_grain": 3, "vegetable_sabzi": 3, "dairy": 2}
    category_serving_maxima = {"cereal_grain": 5, "dal_pulse": 3, "vegetable_sabzi": 4, "dairy": 2}
    while q.sum() < MAX_ACTIVE_DISHES * MAX_SERVINGS_PER_DISH:
        current = kcal_total()
        if current >= target_kcal * 0.995:
            break
        # First ensure cereal servings reach at least 3
        cereal_servings = sum(q[i] for i in selected if foods[i].category == "cereal_grain")
        if cereal_servings < 3:
            cereal_candidates = [i for i in selected if foods[i].category == "cereal_grain" and q[i] < MAX_SERVINGS_PER_DISH]
            if cereal_candidates:
                best_c = min(cereal_candidates, key=lambda i: foods[i].selection_cost)
                q[best_c] += 1
                continue

        candidates = [
            i for i in selected
            if q[i] < MAX_SERVINGS_PER_DISH and
            sum(q[j] for j in selected if foods[j].category == foods[i].category) < category_serving_maxima.get(foods[i].category, 10)
        ]
        if len(selected) < MAX_ACTIVE_DISHES:
            active_by_category = {
                category: sum(1 for i in selected if foods[i].category == category)
                for category in VALID_CATEGORIES
            }
            for i, food in enumerate(foods):
                if i in selected:
                    continue
                category_limit = category_maxima.get(food.category, MAX_ACTIVE_DISHES)
                if active_by_category.get(food.category, 0) < category_limit:
                    candidates.append(i)
        if not candidates:
            break
        best = min(
            candidates,
            key=lambda i: (
                foods[i].selection_cost,
                abs((current + foods[i].nutrients_per_serving["DR1TKCAL"]) - target_kcal),
            ),
        )
        q[best] += 1
        selected.add(best)
    return q


def _build_output_from_quantities(
    q: np.ndarray,
    foods: Sequence[FoodItem],
    stage: Stage3Input,
    solver_meta: Mapping[str, Any],
) -> Stage3Output:
    selected: list[dict[str, Any]] = []
    delivered = {code: 0.0 for code in NUTRIENT_CODES}
    for i, quantity in enumerate(q.tolist()):
        quantity = int(quantity)
        if quantity <= 0:
            continue
        food = foods[i]
        for code in NUTRIENT_CODES:
            delivered[code] += quantity * float(food.nutrients_per_serving[code])
        selected.append({
            "food_id": food.food_id,
            "name": food.name,
            "category": food.category,
            "quantity": quantity,
            "serving_unit": food.serving_unit,
            "total_weight_g": round(quantity * food.serving_weight_g, 2),
            "serving_weight_g": float(food.serving_weight_g),
            "data_source": food.data_source,
            "unknown_nutrients": list(food.unknown_nutrients),
            "catalog_version": food.catalog_version,
        })

    nutrition_records: list[dict[str, Any]] = []
    for code in NUTRIENT_CODES:
        value = float(delivered[code])
        target = float(stage.targets[code])
        percentage_error = (abs(value - target) / abs(target) * 100.0) if target != 0 else (0.0 if value == 0 else None)
        low, high = stage.lower_bounds[code], stage.upper_bounds[code]
        nutrition_records.append({
            "nutrient": code,
            "delivered": round(value, 6),
            "target": target,
            "min_bound": low,
            "max_bound": high,
            "baseline": stage.baselines[code],
            "absolute_error": round(abs(value - target), 6),
            "percentage_error": None if percentage_error is None else round(percentage_error, 4),
            "within_bounds": bool(low - 1e-6 <= value <= high + 1e-6
                                  and not any(qi > 0 and code in food.unknown_nutrients for food, qi in zip(foods, q))),
            "rule_provenance": dict(stage.rule_metadata.get(code, {})),
            "constraint_applied": stage.rule_metadata.get(code, {}).get('planning_role') not in {'report_only', 'exposure_reporting'},
            "unknown_food_count": sum(1 for food, qi in zip(foods, q) if qi > 0 and code in food.unknown_nutrients),
            "value_status": "known_subtotal" if any(qi > 0 and code in food.unknown_nutrients for food, qi in zip(foods, q)) else "complete",
        })

    output = Stage3Output(
        solved_food_basket=selected,
        delivered_nutrition=nutrition_records,
        clinical_verification={},
        solver_metadata=dict(solver_meta),
    )
    output.clinical_verification = verify_basket_safety(output, {
        "patient_metadata": dict(stage.patient_metadata),
        "energy_prescription": dict(stage.energy_prescription),
        "nutrient_constraint_envelope": [
            {"nutrient": code, "baseline": stage.baselines[code], "target": stage.targets[code],
             "min_bound": stage.lower_bounds[code], "max_bound": stage.upper_bounds[code],
             "rule_provenance": dict(stage.rule_metadata.get(code, {}))}
            for code in NUTRIENT_CODES
        ],
    })
    return output


def compute_plan_diff(
    previous_basket: Sequence[Mapping[str, Any]],
    new_basket: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compute a deterministic item-level diff between two meal baskets."""
    prev_map = {str(item["food_id"]): item for item in previous_basket}
    new_map = {str(item["food_id"]): item for item in new_basket}
    added, removed, changed, unchanged = [], [], [], []
    serving_delta = 0
    for fid in sorted(set(prev_map) | set(new_map)):
        if fid not in prev_map:
            qty = int(new_map[fid]["quantity"])
            added.append({"food_id": fid, "name": new_map[fid].get("name", fid), "quantity": qty})
            serving_delta += qty
        elif fid not in new_map:
            qty = int(prev_map[fid]["quantity"])
            removed.append({"food_id": fid, "name": prev_map[fid].get("name", fid), "quantity": qty})
            serving_delta += qty
        else:
            pq, nq = int(prev_map[fid]["quantity"]), int(new_map[fid]["quantity"])
            if pq != nq:
                changed.append({"food_id": fid, "name": new_map[fid].get("name", fid),
                                "previous_quantity": pq, "new_quantity": nq})
                serving_delta += abs(nq - pq)
            else:
                unchanged.append({"food_id": fid, "name": new_map[fid].get("name", fid), "quantity": nq})
    return {
        "added": added,
        "removed": removed,
        "quantity_changed": changed,
        "unchanged": unchanged,
        "total_serving_delta": serving_delta,
    }


def validate_candidate(output: Stage3Output, food_db: Sequence[FoodItem], stage2_input: dict,
                       excluded_ids: Sequence[str] = (),
                       locked_quantities: Mapping[str, int] | None = None,
                       max_active_dishes: int | None = None,
                       dietary_mode: str = "pure_vegetarian") -> dict[str, Any]:
    """Recalculate from IDs/quantities, then validate structure, locks, and active rules."""
    stage = _parse_stage2_input(stage2_input)
    lookup = {food.food_id: i for i, food in enumerate(food_db)}
    q = np.zeros(len(food_db), dtype=np.int64)
    errors: list[str] = []
    basket = output.solved_food_basket
    for row in basket:
        fid, quantity = row.get("food_id"), row.get("quantity")
        if fid not in lookup:
            errors.append(f"Unknown food ID: {fid}")
            continue
        if fid in excluded_ids:
            errors.append(f"Excluded food selected: {fid}")
        if type(quantity) is not int or not 1 <= quantity <= MAX_SERVINGS_PER_DISH:
            errors.append(f"Invalid serving count for {fid}")
            continue
        if q[lookup[fid]]:
            errors.append(f"Duplicate food: {fid}")
        q[lookup[fid]] += quantity
    dish_cap = MAX_ACTIVE_DISHES if max_active_dishes is None else min(MAX_ACTIVE_DISHES, int(max_active_dishes))
    if not MIN_ACTIVE_DISHES <= len(basket) <= dish_cap:
        errors.append(f"Active food count outside {MIN_ACTIVE_DISHES}–{dish_cap}")
    if locked_quantities:
        for lfid, lqty in locked_quantities.items():
            if lfid not in lookup:
                errors.append(f"Locked food missing from catalog: {lfid}")
            elif int(q[lookup[lfid]]) != int(lqty):
                errors.append(f"Locked food changed: {lfid} (expected {lqty}, got {int(q[lookup[lfid]])})")
    counts, active = {}, {}
    for food, quantity in zip(food_db, q):
        if quantity:
            if dietary_mode == "pure_vegetarian" and food.dietary_type == "non_vegetarian":
                errors.append(f"Non-vegetarian food selected in pure_vegetarian mode: {food.food_id}")
            counts[food.category] = counts.get(food.category, 0) + int(quantity)
            active[food.category] = active.get(food.category, 0) + 1
    cat_limits, serv_limits = _category_rules_for_mode(
        dietary_mode,
        allow_zero_dairy=not any(f.category == "dairy" and f.food_id not in excluded_ids for f in food_db) and bool(excluded_ids),
    )
    for category, (low, high) in serv_limits.items():
        if not low <= counts.get(category, 0) <= high:
            errors.append(f"Serving rule failed: {category}")
    for category, (low, high) in cat_limits.items():
        if not low <= active.get(category, 0) <= high:
            errors.append(f"Food variety rule failed: {category}")
    recomputed = _build_output_from_quantities(q, food_db, stage, {})
    actual = {row["nutrient"]: row for row in recomputed.delivered_nutrition}
    supplied = {row.get("nutrient"): row for row in output.delivered_nutrition}
    for code in NUTRIENT_CODES:
        value = supplied.get(code, {}).get("delivered")
        if value is None or not math.isfinite(float(value)) or abs(float(value) - actual[code]["delivered"]) > 1e-4:
            errors.append(f"Nutrient accounting mismatch: {code}")
    required = [code for code in NUTRIENT_CODES
                if stage.rule_metadata.get(code, {}).get("planning_role") not in {"report_only", "exposure_reporting"}]
    unknown = [code for code in required if actual[code]["unknown_food_count"]]
    failed = [code for code in required if not actual[code]["unknown_food_count"] and not actual[code]["within_bounds"]]
    post = recomputed.clinical_verification
    failed_checks = [key for key in ("calories", "macronutrient_split", "sodium_to_potassium_ratio", "fiber_adequacy")
                     if not post[key]["passed"]]
    return {
        "accepted": not errors and not unknown and not failed and not failed_checks,
        "structural_errors": errors, "unknown_active_nutrients": unknown,
        "failed_active_nutrients": failed, "failed_post_checks": failed_checks,
        "active_nutrient_count": len(required),
        "complete_39_field_data": not any(row["unknown_food_count"] for row in actual.values()),
        "clinical_validation_established": False,
        "scope": "Acceptance means active arithmetic/portion checks passed, not clinical certification.",
    }


def _failure_output(code: str, message: str, *, metadata: dict, details: dict) -> Stage3Output:
    """A failed search never exposes a rejected basket as a meal plan."""
    return Stage3Output([], [], {
        "all_active_checks_passed": False, "all_clinical_checks_passed": False,
        "clinical_validation_established": False, "failure_details": details,
    }, {**metadata, "status": code, "optimality_proven": False,
        "heuristic_fallback_used": False}, accepted=False,
        failure={"code": code, "message": message, "details": details})


def solve_meal_plan(
    stage2_input: dict,
    food_db: list[FoodItem],
    time_limit_sec: float = DEFAULT_TIME_LIMIT_SEC,
    revision_spec: Mapping[str, Any] | None = None,
) -> Stage3Output:
    """Return a fully checked active-rule plan or an explicit empty failure.

    Unknown contributions cannot satisfy an active nutrient rule. Only foods
    covering all active fields enter the search. Report-only and beverage-exposure
    fields remain visible. Optional ``revision_spec`` enables controlled revisions:
    excluded foods/ingredients, preferred foods, locked food quantities, maximum
    active dishes, and minimal-change penalties relative to a previous plan.
    """
    stage = _parse_stage2_input(stage2_input)
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

    base_meta = {"backend": "scipy.optimize.milp / HiGHS",
                 "time_limit_sec": effective, "requested_time_limit_sec": float(time_limit_sec),
                 "solve_time_ms": 0.0, "candidate_validation": [],
                 "revision_controls": {
                     "dietary_mode": dietary_mode,
                     "excluded_food_ids": sorted(excluded_ids),
                     "excluded_ingredient_codes": sorted(excluded_ing_codes),
                     "preferred_food_ids": sorted(preferred_ids),
                     "locked_food_quantities": locked_quantities,
                     "max_active_dishes": max_dishes_cap,
                     "change_penalty": change_penalty if prev_quantities else 0.0,
                 }}
    # Conflict check: cannot both exclude and lock the same food
    conflict_lock = excluded_ids.intersection(locked_quantities)
    if conflict_lock:
        return _failure_output("conflicting_revision_constraints",
            f"Requested revision both excludes and locks: {sorted(conflict_lock)}.",
            metadata=base_meta, details={"conflicting_food_ids": sorted(conflict_lock)})

    required = {code for code in NUTRIENT_CODES
                if stage.rule_metadata.get(code, {}).get("planning_role") not in {"report_only", "exposure_reporting"}}
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
        return _failure_output("locked_food_unavailable",
            f"Locked food(s) are excluded or lack complete active data: {missing_locked}.",
            metadata=base_meta, details={"missing_locked_food_ids": missing_locked})

    # Allow zero dairy ONLY when dairy foods exist with complete data in the catalog but were explicitly excluded by user/allergy
    dairy_complete_in_db = [f for f in food_db if f.category == "dairy" and not required.intersection(f.unknown_nutrients)]
    allow_zero_dairy = bool(dairy_complete_in_db) and all(f.food_id in effective_excluded for f in dairy_complete_in_db)

    cat_limits, _ = _category_rules_for_mode(dietary_mode, allow_zero_dairy=allow_zero_dairy)
    required_cats = {cat for cat, (low, _) in cat_limits.items() if low > 0}
    missing_categories = sorted(required_cats - {food.category for food in eligible})
    if len(eligible) < MIN_ACTIVE_DISHES or missing_categories:
        return _failure_output("insufficient_nutrient_data",
            "The catalog lacks sufficiently documented foods for all required meal categories. No plan was accepted.",
            metadata=base_meta, details={
                "missing_categories_with_complete_active_data": missing_categories,
                "eligible_food_count": len(eligible), "candidate_food_count": len(food_db),
                "unknown_active_nutrients": sorted({code for food in food_db for code in food.unknown_nutrients if code in required}),
                "unknown_catalog_nutrients": sorted({code for food in food_db for code in food.unknown_nutrients}),
                "excluded_incomplete_food_ids": [f.food_id for f in food_db if f not in eligible],
            })
    _validate_catalog_categories(eligible, dietary_mode=dietary_mode, allow_zero_dairy=allow_zero_dairy)
    c, integrality, bounds, constraints, info = _make_sparse_constraints(
        eligible, stage, gamma=0.10, dietary_mode=dietary_mode, allow_zero_dairy=allow_zero_dairy
    )
    # Active intervals are acceptance requirements. Do not spend the search on
    # candidates with knowingly nonzero slack, then call them successful.
    lower = bounds.lb.copy()
    upper = bounds.ub.copy()
    m = len(eligible)
    q0 = info["indices"]["q"]
    z0 = info["indices"]["z"]
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

    F = info["food_matrix"]
    kcal = F[NUTRIENT_INDEX["DR1TKCAL"]]
    extra_rows, extra_lower, extra_upper = [], [], []
    for code, factor, low, high in (("DR1TPROT", 4, .10, .35),
                                    ("DR1TCARB", 4, .45, .65),
                                    ("DR1TTFAT", 9, .20, .35)):
        nutrient = F[NUTRIENT_INDEX[code]] * factor
        extra_rows.extend((low * kcal - nutrient, nutrient - high * kcal))
        extra_lower.extend((-np.inf, -np.inf))
        extra_upper.extend((0.0, 0.0))
    ratio = stage.baselines["DR1TSODI"] / (stage.baselines["DR1TPOTA"] + 1.0)
    extra_rows.append(F[NUTRIENT_INDEX["DR1TSODI"]] - ratio * F[NUTRIENT_INDEX["DR1TPOTA"]])
    extra_lower.append(-np.inf)
    extra_upper.append(ratio)
    extra_rows.append(.014 * kcal - F[NUTRIENT_INDEX["DR1TFIBE"]])
    extra_lower.append(-np.inf)
    extra_upper.append(0.0)

    nbase = len(c)
    extra = np.zeros((len(extra_rows), nbase))
    extra[:, :m] = extra_rows

    # Optional max_active_dishes cap
    if max_dishes_cap is not None:
        cap_row = np.zeros((1, nbase))
        cap_row[0, z0:z0 + m] = 1.0
        extra = np.vstack((extra, cap_row))
        extra_lower.append(float(MIN_ACTIVE_DISHES))
        extra_upper.append(float(max_dishes_cap))

    A = vstack((constraints.A, coo_matrix(extra)), format="csr")
    c_lb = np.r_[constraints.lb, np.asarray(extra_lower, dtype=float)]
    c_ub = np.r_[constraints.ub, np.asarray(extra_upper, dtype=float)]

    # Optional minimal-change L1 penalty relative to previous_quantities:
    # q_i - delta_plus_i + delta_minus_i = prev_q_i, minimizing change_penalty * sum(delta_plus + delta_minus)
    if prev_quantities and change_penalty > 0:
        n_delta = 2 * m
        c = np.r_[c, np.full(n_delta, change_penalty, dtype=float)]
        integrality = np.r_[integrality, np.zeros(n_delta, dtype=np.int8)]
        lower = np.r_[lower, np.zeros(n_delta, dtype=float)]
        upper = np.r_[upper, np.full(n_delta, np.inf, dtype=float)]
        pad_existing = coo_matrix((A.shape[0], n_delta))
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

    constraints = LinearConstraint(A, c_lb, c_ub)
    started = time.perf_counter()
    try:
        result = milp(c=c, integrality=integrality, bounds=Bounds(lower, upper),
                      constraints=constraints, options={"time_limit": effective,
                      "mip_rel_gap": .01, "presolve": True, "disp": False})
    except Exception as exc:
        return _failure_output("solver_error", "The optimizer could not complete the search.",
                               metadata=base_meta, details={"error": str(exc)})
    base_meta.update(solve_time_ms=round((time.perf_counter() - started) * 1000, 3),
                     solver_status_code=int(result.status), status_message=str(result.message),
                     solver_incumbent_available=result.x is not None,
                     variable_count=len(c), constraint_count=A.shape[0])
    candidates = []
    if result.x is not None:
        values = np.asarray(result.x[:m])
        if np.isfinite(values).all() and np.max(np.abs(values - np.rint(values))) <= 1e-5:
            candidates.append(("solver", np.rint(values).astype(np.int64)))
    if not locked_quantities and max_dishes_cap is None:
        candidates.append(("heuristic", _heuristic_incumbent(eligible, stage.energy_prescription["target_calories"])))
    valid = []
    for origin, q in candidates:
        candidate = _build_output_from_quantities(q, eligible, stage, {})
        validation = validate_candidate(
            candidate, food_db, stage2_input,
            excluded_ids=sorted(effective_excluded),
            locked_quantities=locked_quantities or None,
            max_active_dishes=max_dishes_cap,
            dietary_mode=dietary_mode,
        )
        base_meta["candidate_validation"].append({"origin": origin, **validation})
        if validation["accepted"]:
            delivered = F @ q.astype(float)
            score = abs(delivered[NUTRIENT_INDEX["DR1TKCAL"]] - stage.energy_prescription["target_calories"]) * info["calorie_deviation_weight"]
            score += .10 * sum(food.selection_cost for food, quantity in zip(eligible, q) if quantity > 0)
            if prev_quantities and change_penalty > 0:
                score += change_penalty * sum(abs(int(q[i]) - int(prev_quantities.get(eligible[i].food_id, 0))) for i in range(m))
            valid.append((score, origin, candidate, validation))
    if not valid:
        code = "infeasible_active_rules" if result.status == 2 else "timeout_no_valid_plan" if result.status == 1 else "no_valid_candidate"
        return _failure_output(code,
            "No candidate passed all active nutrient, portion and post-check rules. No plan was accepted.",
            metadata=base_meta, details={"solver_message": str(result.message),
                                        "candidate_validation": base_meta["candidate_validation"]})
    score, origin, output, validation = min(valid, key=lambda row: row[0])
    gap = getattr(result, "mip_gap", None)
    output.solver_metadata = {**base_meta, "status": "accepted_solver_plan" if origin == "solver" else "accepted_validated_heuristic",
        "optimality_proven": bool(origin == "solver" and result.status == 0),
        "mip_gap": None if gap is None or not math.isfinite(float(gap)) else float(gap),
        "heuristic_fallback_used": origin == "heuristic", "heuristic_candidate_selected": origin == "heuristic",
        "active_dish_count": len(output.solved_food_basket),
        "total_servings": sum(row["quantity"] for row in output.solved_food_basket),
        "calorie_error_pct": output.clinical_verification['calories']['absolute_error_pct'],
        "target_calories": stage.energy_prescription['target_calories'],
        "delivered_calories": output.clinical_verification['calories']['delivered_kcal'],
        "total_slack_penalty": float(score), "active_nutrient_slack": 0.0}
    output.accepted = True
    output.clinical_verification["all_active_checks_passed"] = True
    output.clinical_verification["acceptance_validation"] = validation
    return output


# ---------------------------------------------------------------------------
# Clinical verification and JSON safety
# ---------------------------------------------------------------------------
def verify_basket_safety(output: Stage3Output, stage2_input: dict) -> dict:
    """Verify energy, macro distribution, Na:K, fiber, and the Stage 2 envelope."""
    stage = _parse_stage2_input(stage2_input)
    delivered = {row["nutrient"]: float(row["delivered"]) for row in output.delivered_nutrition}
    missing = sorted(set(NUTRIENT_CODES) - set(delivered))
    if missing:
        raise ValueError(f"Output is missing delivered nutrients: {missing}")

    kcal = delivered["DR1TKCAL"]
    target_kcal = stage.energy_prescription["target_calories"]
    calorie_error_pct = abs(kcal - target_kcal) / target_kcal * 100.0 if target_kcal > 0 else math.inf
    calories_pass = bool(kcal > 0 and calorie_error_pct <= 5.0 + 1e-9)

    p_kcal = delivered["DR1TPROT"] * 4.0
    c_kcal = delivered["DR1TCARB"] * 4.0
    f_kcal = delivered["DR1TTFAT"] * 9.0
    denom = kcal if kcal > 0 else 1.0
    macro_pct = {
        "protein_pct_kcal": p_kcal / denom * 100.0,
        "carb_pct_kcal": c_kcal / denom * 100.0,
        "fat_pct_kcal": f_kcal / denom * 100.0,
    }
    # Adult AMDR-style screening bands; this is a portfolio diagnostic, not a
    # substitute for the individualized Stage 2 nutrient envelope.
    macro_ranges = {
        "protein_pct_kcal": (10.0, 35.0),
        "carb_pct_kcal": (45.0, 65.0),
        "fat_pct_kcal": (20.0, 35.0),
    }
    macro_pass_each = {
        key: bool(low - 1e-9 <= macro_pct[key] <= high + 1e-9)
        for key, (low, high) in macro_ranges.items()
    }
    macro_pass = bool(all(macro_pass_each.values()))

    sodium = delivered["DR1TSODI"]
    potassium = delivered["DR1TPOTA"]
    ratio = sodium / (potassium + 1.0)
    baseline_ratio = stage.baselines["DR1TSODI"] / (stage.baselines["DR1TPOTA"] + 1.0)
    sodium_potassium_pass = bool(ratio <= baseline_ratio + 1e-9)

    fiber = delivered["DR1TFIBE"]
    fiber_lower_bound = stage.lower_bounds["DR1TFIBE"]
    fiber_per_1000_kcal = fiber / (kcal / 1000.0) if kcal > 0 else 0.0
    fiber_pass = bool(fiber >= fiber_lower_bound - 1e-9 and fiber_per_1000_kcal >= 14.0 - 1e-9)

    envelope_checks = {
        row["nutrient"]: bool(
            not row.get("unknown_food_count", 0) and
            stage.lower_bounds[row["nutrient"]] - 1e-6
            <= delivered[row["nutrient"]]
            <= stage.upper_bounds[row["nutrient"]] + 1e-6
        )
        for row in output.delivered_nutrition
    }
    envelope_pass_count = sum(envelope_checks.values())
    envelope_pass = envelope_pass_count == len(NUTRIENT_CODES)

    return {
        "calories": {
            "passed": calories_pass,
            "delivered_kcal": round(kcal, 3),
            "target_kcal": round(target_kcal, 3),
            "absolute_error_pct": round(calorie_error_pct, 4),
            "acceptance_threshold_pct": 5.0,
        },
        "macronutrient_split": {
            "passed": macro_pass,
            "percent_of_kcal": {k: round(v, 3) for k, v in macro_pct.items()},
            "screening_ranges_pct": {k: list(v) for k, v in macro_ranges.items()},
            "individual_pass_flags": macro_pass_each,
        },
        "sodium_to_potassium_ratio": {
            "passed": sodium_potassium_pass,
            "delivered_ratio": round(ratio, 5),
            "baseline_ratio": round(baseline_ratio, 5),
            "sodium_mg": round(sodium, 3),
            "potassium_mg": round(potassium, 3),
        },
        "fiber_adequacy": {
            "passed": fiber_pass,
            "delivered_fiber_g": round(fiber, 3),
            "stage2_minimum_g": round(fiber_lower_bound, 3),
            "fiber_g_per_1000_kcal": round(fiber_per_1000_kcal, 3),
            "minimum_g_per_1000_kcal": 14.0,
        },
        "nutrient_envelope": {
            "passed": envelope_pass,
            "nutrients_within_bounds": int(envelope_pass_count),
            "nutrient_count": len(NUTRIENT_CODES),
            "failed_nutrients": [code for code, passed in envelope_checks.items() if not passed],
            "unknown_nutrients": [row["nutrient"] for row in output.delivered_nutrition if row.get("unknown_food_count", 0)],
            "report_only_nutrients": [code for code in NUTRIENT_CODES if stage.rule_metadata.get(code, {}).get('planning_role') == 'report_only'],
            "verification_scope": "Arithmetic checks only. Report-only fields have no adequacy target; numerical search caps are not ULs. Unknowns still prevent complete nutrient verification.",
        },
        "all_clinical_checks_passed": bool(
            calories_pass and macro_pass and sodium_potassium_pass and fiber_pass and envelope_pass
        ),
        "clinical_note": (
            "Verification is arithmetic against the supplied Stage 2 dossier and catalog values; "
            "it is not a medical diagnosis or proof that a food composition estimate is clinically exact."
        ),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        converted = float(value)
        return converted if math.isfinite(converted) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def print_meal_plan(output: Stage3Output) -> None:
    if not output.accepted:
        print('Stage 3: no accepted meal plan')
        print(json.dumps(output.failure, indent=2, ensure_ascii=False))
        return
    """Print a compact terminal summary of the daily basket and checks."""
    print("\n" + "=" * 92)
    print("BioNutriGPT Stage 3 | Discrete Indian Meal Synthesis")
    print("=" * 92)
    print(f"{'Dish':34} {'Qty':>4} {'Household serving':24} {'Total g':>9}")
    print("-" * 92)
    for row in output.solved_food_basket:
        print(f"{row['name'][:34]:34} {row['quantity']:>4} {row['serving_unit'][:24]:24} {row['total_weight_g']:>9.1f}")
    print("-" * 92)
    delivered = {r["nutrient"]: r["delivered"] for r in output.delivered_nutrition}
    for code in ("DR1TKCAL", "DR1TPROT", "DR1TCARB", "DR1TFIBE", "DR1TSFAT", "DR1TSODI", "DR1TPOTA"):
        print(f"{code:18}: {delivered[code]:>10.2f}")
    checks = output.clinical_verification
    print("\nClinical verification:")
    for key in ("calories", "macronutrient_split", "sodium_to_potassium_ratio", "fiber_adequacy", "nutrient_envelope"):
        item = checks[key]
        print(f"  [{'PASS' if item['passed'] else 'FAIL'}] {key}")
    print(f"\nMILP status: {output.solver_metadata['status']}")
    print(f"Solver time: {output.solver_metadata['solve_time_ms']:.3f} ms")
    print(f"Total slack/objective penalty: {output.solver_metadata['total_slack_penalty']:.6f}")
    print(f"Stage 3 clinical checks: {'PASS' if checks['all_clinical_checks_passed'] else 'NOT ALL PASSED'}")
    print("=" * 92)


def _build_demo_stage2_input() -> dict[str, Any]:
    """Create the requested illustrative 2,064-kcal / 90-g-protein demo dossier."""
    targets = {
        "DR1TKCAL": 2064.0, "DR1TPROT": 90.0, "DR1TCARB": 245.0,
        "DR1TSUGR": 35.0, "DR1TFIBE": 35.0, "DR1TTFAT": 68.0,
        "DR1TSFAT": 12.0, "DR1TMFAT": 25.0, "DR1TPFAT": 20.0,
        "DR1TCHOL": 120.0, "DR1TATOC": 10.0, "DR1TRET": 250.0,
        "DR1TVARA": 800.0, "DR1TACAR": 300.0, "DR1TBCAR": 2500.0,
        "DR1TLYCO": 4000.0, "DR1TLZ": 1200.0, "DR1TVB1": 1.2,
        "DR1TVB2": 1.3, "DR1TNIAC": 16.0, "DR1TVB6": 1.5,
        "DR1TFOLA": 400.0, "DR1TFA": 100.0, "DR1TFF": 300.0,
        "DR1TFDFE": 400.0, "DR1TVB12": 2.4, "DR1TVC": 90.0,
        "DR1TVK": 120.0, "DR1TCALC": 1000.0, "DR1TPHOS": 700.0,
        "DR1TMAGN": 400.0, "DR1TIRON": 8.0, "DR1TZINC": 11.0,
        "DR1TCOPP": 0.9, "DR1TSODI": 1900.0, "DR1TPOTA": 3500.0,
        "DR1TSELE": 55.0, "DR1TCAFF": 0.0, "DR1TALCO": 0.0,
    }
    envelope = []
    for code in NUTRIENT_CODES:
        target = targets[code]
        if code == "DR1TKCAL":
            low, high = target * 0.95, target * 1.05
        elif code == "DR1TSODI":
            low, high = 0.0, 1900.0
        elif code == "DR1TPOTA":
            low, high = 3500.0, 4700.0
        elif code == "DR1TFIBE":
            low, high = 35.0, 60.0
        elif code == "DR1TPROT":
            low, high = 90.0, 150.0
        elif code == "DR1TSFAT":
            low, high = 0.0, 15.0
        else:
            span = max(target * 0.25, 1.0)
            low, high = max(0.0, target - span), target + span
        envelope.append({
            "nutrient": code,
            "baseline": target,
            "target": target,
            "min_bound": low,
            "max_bound": high,
            "delta": 0.0,
        })
    return {
        "patient_metadata": {
            "age_years": 48.0, "is_female": False,
            "weight_kg": 75.0, "height_cm": 170.0,
            "activity_level": "moderate",
        },
        "energy_prescription": {"BMR": 1600.0, "TDEE": 2064.0, "target_calories": 2064.0},
        "nutrient_constraint_envelope": envelope,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="BioNutriGPT Stage 3 discrete Indian meal synthesis using SciPy/HiGHS."
    )
    parser.add_argument("input", nargs="?", help="Stage 2 JSON dossier; omit to run the built-in demo.")
    parser.add_argument("--food-db", help="Optional IFCT/RecipeDB CSV, Parquet, or JSON food database.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help=f"Output JSON path (default: {DEFAULT_OUTPUT}).")
    parser.add_argument("--time-limit", type=float, default=DEFAULT_TIME_LIMIT_SEC,
                        help="HiGHS time limit in seconds; hard-capped at 0.05 seconds.")
    parser.add_argument("--quiet", action="store_true", help="Suppress terminal meal-plan report.")
    args = parser.parse_args(argv)

    if args.input:
        input_path = Path(args.input).expanduser()
        if not input_path.is_file():
            raise FileNotFoundError(f"Stage 2 JSON dossier not found: {input_path}")
        with input_path.open("r", encoding="utf-8") as handle:
            stage2_input = json.load(handle)
    else:
        stage2_input = _build_demo_stage2_input()

    food_db = load_food_database(args.food_db)
    output = solve_meal_plan(stage2_input, food_db, time_limit_sec=args.time_limit)
    report = {
        "stage": "BioNutriGPT Stage 3: Discrete Meal Synthesis",
        "patient_metadata": dict(_parse_stage2_input(stage2_input).patient_metadata),
        "energy_prescription": dict(_parse_stage2_input(stage2_input).energy_prescription),
        "food_database": {
            "food_count": len(food_db),
            "source": "external" if any(f.data_source != "synthetic_fallback_estimate" for f in food_db) else "synthetic_fallback_estimate",
            "provenance_note": (
                "External database supplied by user." if any(f.data_source != "synthetic_fallback_estimate" for f in food_db)
                else "Illustrative fallback estimates; not official IFCT 2017 or RecipeDB values."
            ),
        },
        **output.to_dict(),
        "stage3_clinically_verified": bool(output.clinical_verification["all_clinical_checks_passed"]),
    }
    output_path = Path(args.output).expanduser()
    if not output_path.is_absolute():
        output_path = (Path.cwd() / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(report), handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    temp_path.replace(output_path)

    if not args.quiet:
        print_meal_plan(output)
        print(f"\nJSON report saved to: {output_path}")
        if report["food_database"]["source"] == "synthetic_fallback_estimate":
            print("NOTE: Embedded food nutrient values are illustrative estimates, not official IFCT/RecipeDB data.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
