#!/usr/bin/env python3
"""BioNutriGPT Stage 3: Discrete Indian Meal Synthesis (MILP / HiGHS).

The engine translates the continuous nutrient envelope emitted by BioNutriGPT
Stage 2 into a discrete basket of Indian cooked dishes. It uses SciPy's
``scipy.optimize.milp`` (HiGHS), integer serving counts, binary food-selection
variables, and strict Stage 2 nutrient bounds. It does not relax nutrient
requirements to make an infeasible basket appear acceptable.

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_DIR / "stage2_counterfactual_prescription.json"
DEFAULT_OUTPUT = PROJECT_DIR / "stage3_indian_meal_plan.json"
DEFAULT_TIME_LIMIT_SEC = 2.0
MAX_TIME_LIMIT_SEC = 2.0
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
    "dal_pulse", "cereal_grain", "vegetable_sabzi", "dairy", "fruit", "oil_fat"
}
CATEGORY_ALIASES = {
    "dal": "dal_pulse", "pulse": "dal_pulse", "pulses": "dal_pulse",
    "grain": "cereal_grain", "cereal": "cereal_grain", "bread": "cereal_grain",
    "vegetable": "vegetable_sabzi", "sabzi": "vegetable_sabzi", "veg": "vegetable_sabzi",
    "curd": "dairy", "milk": "dairy", "fruit": "fruit", "oil": "oil_fat",
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


@dataclass
class Stage3Output:
    """Serializable result returned by :func:`solve_meal_plan`."""

    solved_food_basket: list[dict[str, Any]]
    delivered_nutrition: list[dict[str, Any]]
    clinical_verification: dict[str, Any]
    solver_metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
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
    return FoodItem(
        food_id, name, category, unit, grams, nutrients,
        selection_cost=cost, data_source="synthetic_fallback_estimate",
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
    return FoodItem(
        food_id=str(row["food_id"]),
        name=str(row["name"]),
        category=str(row["category"]),
        serving_unit=str(row["serving_unit"]),
        serving_weight_g=float(row["serving_weight_g"]),
        nutrients_per_serving={code: float(nutrient_raw[code]) for code in NUTRIENT_CODES},
        selection_cost=selection_cost_value,
        data_source=source,
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


def _validate_catalog_categories(food_db: Sequence[FoodItem]) -> None:
    ids = [food.food_id for food in food_db]
    if len(ids) < 5:
        raise ValueError("Food database needs at least five foods to satisfy category rules.")
    if len(ids) != len(set(ids)):
        raise ValueError("Food database food_id values must be unique.")
    categories = {food.category for food in food_db}
    required = {"dal_pulse", "cereal_grain", "vegetable_sabzi", "dairy"}
    if not required.issubset(categories):
        raise ValueError(f"Food database is missing required categories: {sorted(required - categories)}")


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

    # Category variety constraints.
    category_limits = {
        "dal_pulse": (1, 2),
        "cereal_grain": (1, 3),
        "vegetable_sabzi": (2, np.inf),
        "dairy": (1, np.inf),
    }
    for category, (low, high) in category_limits.items():
        indices = [i for i, food in enumerate(foods) if food.category == category]
        if not indices:
            raise ValueError(f"No foods available for required category {category!r}.")
        add_row({z0 + i: 1.0 for i in indices}, low, high)

    # Realistic Indian plate composition constraints on total servings (sum of q_i):
    # Guarantees sufficient staple grains across all main meals (breakfast, lunch, dinner),
    # ensures protein across lunch & dinner, and prevents vegetable/dairy over-concentration.
    serving_limits = {
        "cereal_grain": (3.0, 5.0),    # At least 3 servings (e.g. breakfast + lunch + dinner staple)
        "dal_pulse": (2.0, 3.0),       # 2 to 3 servings (lunch & dinner protein)
        "vegetable_sabzi": (2.0, 4.0), # 2 to 4 servings (balanced vegetables)
        # Do not cap dairy at two servings: this catalog reports at most 0.5 mcg
        # B12 per serving, while Stage 2 can require a minimum of 2.4 mcg/day.
        "dairy": (1.0, 8.0),
    }
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
            "within_bounds": bool(low - 1e-6 <= value <= high + 1e-6),
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
             "min_bound": stage.lower_bounds[code], "max_bound": stage.upper_bounds[code]}
            for code in NUTRIENT_CODES
        ],
    })
    return output


def solve_meal_plan(
    stage2_input: dict,
    food_db: list[FoodItem],
    time_limit_sec: float = DEFAULT_TIME_LIMIT_SEC,
) -> Stage3Output:
    """Solve the discrete serving optimization with a bounded HiGHS time limit.

    The supplied time limit is capped at MAX_TIME_LIMIT_SEC. If HiGHS returns no incumbent before that deadline, a
    deterministic calorie-oriented recovery basket is returned and explicitly
    marked as a heuristic fallback (never as an optimal MILP solution).
    """
    stage = _parse_stage2_input(stage2_input)
    _validate_catalog_categories(food_db)
    if not math.isfinite(float(time_limit_sec)) or time_limit_sec <= 0:
        raise ValueError("time_limit_sec must be positive and finite.")
    effective_time_limit = min(float(time_limit_sec), MAX_TIME_LIMIT_SEC)
    if effective_time_limit <= 0:
        raise ValueError("Effective solver time limit must be positive.")

    c, integrality, bounds, constraints, model_info = _make_sparse_constraints(food_db, stage, gamma=0.10)
    # Stage 2 bounds are requirements, not soft preferences. Keep slack
    # variables in the model layout for reporting compatibility, but fix them
    # to zero so an out-of-range basket cannot be selected as a plan.
    indices = model_info["indices"]
    nutrient_count = len(NUTRIENT_CODES)
    strict_upper = bounds.ub.copy()
    strict_upper[indices["e_plus"]:indices["e_plus"] + nutrient_count] = 0.0
    strict_upper[indices["e_minus"]:indices["e_minus"] + nutrient_count] = 0.0
    bounds = Bounds(bounds.lb, strict_upper)
    started = time.perf_counter()
    try:
        result = milp(
            c=c,
            integrality=integrality,
            bounds=bounds,
            constraints=constraints,
            options={
                "time_limit": effective_time_limit,
                "mip_rel_gap": 0.01,
                "presolve": True,
                "disp": False,
            },
        )
    except Exception as exc:
        raise RuntimeError(f"SciPy/HiGHS MILP invocation failed: {exc}") from exc
    solve_ms = (time.perf_counter() - started) * 1000.0

    solver_incumbent_available = result.x is not None
    q_solver: Optional[np.ndarray] = None
    if solver_incumbent_available:
        x = np.asarray(result.x, dtype=np.float64)
        q_start = model_info["indices"]["q"]
        q_solver = np.clip(
            np.rint(x[q_start:q_start + len(food_db)]).astype(np.int32),
            0, MAX_SERVINGS_PER_DISH,
        )

    # The recovery candidate is for preview only. It must never replace a
    # solver incumbent satisfying the strict nutrient envelope.
    q_heuristic = _heuristic_incumbent(food_db, stage.energy_prescription["target_calories"])
    F = model_info["food_matrix"]
    target_kcal = stage.energy_prescription["target_calories"]

    heuristic_selected = q_solver is None
    q = q_heuristic if heuristic_selected else q_solver
    assert q is not None

    # Recompute true nutrient slack and objective terms from the selected integer basket.
    delivered_vector = F @ q.astype(np.float64)
    nutrient_slack_penalty = 0.0
    slack_breakdown: dict[str, float] = {}
    for k, code in enumerate(NUTRIENT_CODES):
        low, high = stage.lower_bounds[code], stage.upper_bounds[code]
        violation = max(low - delivered_vector[k], 0.0) + max(delivered_vector[k] - high, 0.0)
        penalty = violation * model_info["slack_weights"][code]
        nutrient_slack_penalty += penalty
        if violation > 1e-9:
            slack_breakdown[code] = round(float(penalty), 6)

    delivered_kcal = float(delivered_vector[NUTRIENT_INDEX["DR1TKCAL"]])
    calorie_abs_error = abs(delivered_kcal - target_kcal)
    calorie_deviation_penalty = calorie_abs_error * model_info["calorie_deviation_weight"]
    diversity_penalty = 0.10 * sum(food.selection_cost for food, qi in zip(food_db, q) if qi > 0)
    total_penalty = nutrient_slack_penalty + calorie_deviation_penalty + diversity_penalty

    status_code = int(getattr(result, "status", 4))
    if heuristic_selected:
        solver_status = (
            "heuristic_fallback_no_incumbent" if not solver_incumbent_available
            else "heuristic_candidate_selected_after_time_limit"
        )
    elif status_code == 0:
        solver_status = "optimal"
    elif status_code == 1:
        solver_status = "time_limit_with_incumbent"
    else:
        solver_status = f"solver_status_{status_code}_with_incumbent"
    status_message = str(getattr(result, "message", ""))
    gap_value = getattr(result, "mip_gap", None)
    mip_gap = None if gap_value is None or not math.isfinite(float(gap_value)) else float(gap_value)

    solver_meta = {
        "backend": "scipy.optimize.milp / HiGHS",
        "status": solver_status,
        "status_message": status_message,
        "solve_time_ms": round(solve_ms, 3),
        "time_limit_sec": effective_time_limit,
        "requested_time_limit_sec": float(time_limit_sec),
        "time_limit_capped_at_max": bool(float(time_limit_sec) > MAX_TIME_LIMIT_SEC),
        "mip_gap": mip_gap,
        "optimality_proven": bool(not heuristic_selected and getattr(result, "status", None) == 0),
        "solver_incumbent_available": bool(solver_incumbent_available),
        "heuristic_candidate_selected": bool(heuristic_selected),
        "heuristic_fallback_used": bool(heuristic_selected),
        "variable_count": model_info["variable_count"],
        "constraint_count": model_info["constraint_count"],
        "active_dish_count": int(np.count_nonzero(q)),
        "total_servings": int(q.sum()),
        "delivered_calories": round(delivered_kcal, 3),
        "target_calories": round(target_kcal, 3),
        "calorie_error_pct": round(calorie_abs_error / target_kcal * 100.0, 4),
        "total_slack_penalty": round(float(total_penalty), 6),
        "nutrient_slack_penalty": round(float(nutrient_slack_penalty), 6),
        "calorie_deviation_penalty": round(float(calorie_deviation_penalty), 6),
        "diversity_penalty": round(float(diversity_penalty), 6),
        "nutrient_slack_penalty_by_code": slack_breakdown,
    }
    return _build_output_from_quantities(q, food_db, stage, solver_meta)


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
                        help=f"HiGHS time limit in seconds; hard-capped at {MAX_TIME_LIMIT_SEC:g} seconds.")
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
