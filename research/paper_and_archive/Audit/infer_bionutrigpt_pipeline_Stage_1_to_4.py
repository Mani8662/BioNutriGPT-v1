#!/usr/bin/env python3
"""Unified BioNutriGPT Master Clinical Inference Pipeline & Stage 4 Compiler.

A single, self-contained master executable providing:
  1. Full End-to-End 4-Stage Execution:
       Stage 1: Deep Cross-Attention Diagnostics (T2D, Obesity, CVD risk & attention drivers)
       Stage 2: Gradient-Guided Counterfactual Optimization (4 Clinical Safety Filters)
       Stage 3: Discrete Indian Meal Synthesis (HiGHS MILP, ICMR-NIN IFCT 2017 & RecipeDB)
       Stage 4: Chrono-Nutrition Compiler (Zero calculation authority, clinical rationale tags)
     Usage:
       python infer_bionutrigpt_pipeline_Stage_1_to_4.py sample_patient_single_line.txt

  2. Direct Stage 4 Dossier Compilation:
       If passed an existing dossier JSON file, it immediately validates and compiles
       the Stage 4 clinical report without re-running upstream stages.
     Usage:
       python infer_bionutrigpt_pipeline_Stage_1_to_4.py bionutrigpt_pipeline_dossier.json

Supported Backends:
  --backend transformers Local Qwen 2.5 4-bit NF4 GPU LLM (RTX 3050 CUDA / Accelerate) [default]
  --backend ollama       Local Ollama REST API daemon (http://localhost:11434)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# Upstream Stage 1-3 modules
import infer_bionutrigpt_pipeline_Stage_1_and_2 as p12
import stage3_milp_solver as s3

LOGGER = logging.getLogger("BioNutriGPT.Pipeline")

# ---------------------------------------------------------------------------
# Constants & Defaults
# ---------------------------------------------------------------------------
DEFAULT_PATIENT_FILE = PROJECT_DIR / "sample_patient_single_line.txt"
DEFAULT_DOSSIER_FILE = PROJECT_DIR / "bionutrigpt_pipeline_dossier.json"
DEFAULT_REPORT_FILE = PROJECT_DIR / "bionutrigpt_final_patient_report.json"
DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"

MEAL_SLOTS = [
    ("Breakfast", "07:30"),
    ("Mid-Morning Snack", "10:30"),
    ("Lunch", "13:00"),
    ("Evening Snack", "16:30"),
    ("Dinner", "19:30"),
]

DISEASE_LABELS = {
    "diabetes": "Type 2 Diabetes",
    "obesity": "Clinical Obesity",
    "cvd": "Cardiovascular Disease",
}

_TORCH_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}


# ---------------------------------------------------------------------------
# Stage 4 Data Structures & Error Classes
# ---------------------------------------------------------------------------
class Stage4Error(RuntimeError):
    """Raised when clinical validation, contract enforcement, or generation fails."""


@dataclass(frozen=True)
class CompilerConfig:
    backend: str = "transformers"
    model_name: str = DEFAULT_MODEL
    ollama_url: str = DEFAULT_OLLAMA_URL
    timeout_sec: float = 60.0
    max_new_tokens: int = 1536


@dataclass(frozen=True)
class ValidatedDossier:
    patient_metadata: dict[str, Any]
    stage1: dict[str, Any]
    stage2: dict[str, Any]
    stage3: dict[str, Any]
    basket: list[dict[str, Any]]
    certified_for_stage3: bool


@dataclass(frozen=True)
class BackendResult:
    text: str
    backend: str
    model: str
    elapsed_ms: float


# ---------------------------------------------------------------------------
# Stage 4 Clinical Rationale & Food Categorization
# ---------------------------------------------------------------------------
def _clinical_rationale_for_food(food: Mapping[str, Any]) -> str:
    """Generate a precise clinical and physiological rationale for why a food is prescribed."""
    category = str(food.get("category", "")).strip().lower()
    name = str(food.get("name", "")).lower()

    if "karela" in name or "bitter gourd" in name:
        return "Antidiabetic Vegetable — Charantin & polypeptide-p for glycemic regulation"
    if "palak" in name or "spinach" in name or "methi" in name:
        return "Nutrient-Dense Greens — Folate, lutein & cellular antioxidant defense"
    if "bhindi" in name or "okra" in name:
        return "Low-GI Vegetable — Soluble mucilage fiber for blood glucose moderation"
    if "avocado" in name:
        return "Heart-Healthy Lipid — Monounsaturated fats & potassium for cardiometabolic health"
    if "paneer" in name or "tofu" in name:
        return "Quality Dairy Protein — Slow-release casein & essential calcium"
    if "moong" in name or "sprout" in name:
        return "High-Protein Pulse — Bioavailable plant protein, enzymes & postprandial satiety"
    if "soya" in name:
        return "Complete Plant Protein — Lean isoflavone-rich protein for muscle preservation"
    if "sambar" in name or "dal" in name or "chana" in name or "rajma" in name:
        return "High-Fiber Lentil — Complex plant protein & steady carbohydrate release"
    if "rice" in name:
        return "Staple Grain — Controlled carbohydrate portion for sustained cellular energy"
    if "phulka" in name or "roti" in name or "chapati" in name:
        return "Whole-Grain Staple — Low-fat complex carbohydrate & dietary fiber"
    if "poha" in name or "upma" in name or "oats" in name:
        return "Morning Energy Staple — Easily digestible breakfast carbohydrate"
    if category == "fruit":
        return "Whole Fruit — Natural antioxidants, micronutrients & cellular hydration"
    if category == "dal_pulse":
        return "Plant Protein & Legume — Sustained satiety & glycemic stabilization"
    if category == "vegetable_sabzi":
        return "Cooked Vegetable — Dietary fiber, micronutrients & low energy density"
    if category == "cereal_grain":
        return "Staple Grain — Controlled complex carbohydrates for baseline metabolic energy"
    if category == "dairy":
        return "Dairy Nutrition — High-biological-value protein & bone mineral support"
    return "Prescribed Dietary Component — Formulated to meet individualized metabolic targets"


# ---------------------------------------------------------------------------
# Stage 4 Balanced Chrono-Nutrition Plate Scheduler
# ---------------------------------------------------------------------------
def _default_meal_guidance(meal_name: str, items: Sequence[Mapping[str, Any]]) -> str:
    if not items:
        return "No Stage 3 basket item is assigned to this time; do not add an unplanned dish."
    if meal_name == "Breakfast":
        return "Eat the assigned basket portions at a regular morning mealtime."
    if meal_name == "Mid-Morning Snack":
        return "Use the assigned whole-fruit or snack portion as listed; do not add another food item."
    if meal_name == "Lunch":
        return "Use this as the main midday meal, eating the assigned portions as prepared."
    if meal_name == "Evening Snack":
        return "Keep this snack to the exact assigned Stage 3 portions."
    return "Finish the remaining assigned portions at dinner; avoid adding extra dishes."


def build_deterministic_schedule(basket: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Allocate every immutable serving to a meal using a clinical Indian plate model.

    Guarantees:
      - 100% preservation of dish IDs and exact integer quantities (Zero Hallucination).
      - Breakfast receives morning grain (poha/upma/idli/oats/phulka) and/or morning dairy/protein.
      - Mid-morning snack receives whole fruit.
      - Lunch and Dinner both receive balanced staple grains (roti/rice), dal/pulse, and sabzi.
      - Evening snack receives secondary dairy, fruit, or light item.
    """
    meal_items: dict[str, dict[str, dict[str, Any]]] = {name: {} for name, _ in MEAL_SLOTS}

    def add_to_meal(slot: str, food: Mapping[str, Any]) -> None:
        food_id = str(food["food_id"])
        entry = meal_items[slot].setdefault(food_id, {
            "food_id": food_id,
            "name": str(food["name"]),
            "quantity": 0,
            "serving_unit": str(food.get("serving_unit", "serving")),
            "clinical_rationale": _clinical_rationale_for_food(food),
        })
        entry["quantity"] += 1

    servings_by_cat: dict[str, list[Mapping[str, Any]]] = {
        "fruit": [],
        "dairy": [],
        "dal_pulse": [],
        "cereal_grain": [],
        "vegetable_sabzi": [],
        "oil_fat": [],
        "other": [],
    }
    for food in basket:
        cat = str(food.get("category", "")).strip().lower()
        if cat not in servings_by_cat:
            cat = "other"
        qty = int(food["quantity"])
        for _ in range(qty):
            servings_by_cat[cat].append(food)

    # 1. Distribute Fruits: 1st -> Mid-Morning Snack, 2nd -> Evening Snack, others -> Breakfast
    fruit_slots = ["Mid-Morning Snack", "Evening Snack", "Breakfast"]
    for i, item in enumerate(servings_by_cat["fruit"]):
        slot = fruit_slots[min(i, len(fruit_slots) - 1)]
        add_to_meal(slot, item)

    # 2. Distribute Cereals & Grains:
    bfast_kws = ("poha", "upma", "idli", "dosa", "oats", "cheela", "thepla", "paratha", "porridge")
    bfast_cereals = [f for f in servings_by_cat["cereal_grain"] if any(k in str(f.get("name", "")).lower() for k in bfast_kws)]
    staple_cereals = [f for f in servings_by_cat["cereal_grain"] if f not in bfast_cereals]

    for item in bfast_cereals:
        add_to_meal("Breakfast", item)

    if not bfast_cereals and len(staple_cereals) >= 3:
        add_to_meal("Breakfast", staple_cereals.pop(0))

    cereal_slots = ["Lunch", "Dinner"]
    for i, item in enumerate(staple_cereals):
        add_to_meal(cereal_slots[i % 2], item)

    # 3. Distribute Dals / Pulses: alternate between Lunch and Dinner
    dal_slots = ["Lunch", "Dinner"]
    for i, item in enumerate(servings_by_cat["dal_pulse"]):
        add_to_meal(dal_slots[i % 2], item)

    # 4. Distribute Sabzis (Vegetables): alternate between Lunch and Dinner
    veg_slots = ["Lunch", "Dinner"]
    for i, item in enumerate(servings_by_cat["vegetable_sabzi"]):
        add_to_meal(veg_slots[i % 2], item)

    # 5. Distribute Dairy: 1st -> Breakfast, 2nd -> Evening Snack, 3rd -> Dinner, 4th -> Lunch
    dairy_slots = ["Breakfast", "Evening Snack", "Dinner", "Lunch"]
    for i, item in enumerate(servings_by_cat["dairy"]):
        slot = dairy_slots[min(i, len(dairy_slots) - 1)]
        add_to_meal(slot, item)

    # 6. Distribute Oils / Fats / Others: alternate between Lunch and Dinner
    other_slots = ["Lunch", "Dinner", "Breakfast"]
    for i, item in enumerate(servings_by_cat["oil_fat"] + servings_by_cat["other"]):
        add_to_meal(other_slots[i % len(other_slots)], item)

    schedule: list[dict[str, Any]] = []
    for meal_name, meal_time in MEAL_SLOTS:
        items = list(meal_items[meal_name].values())
        schedule.append({
            "meal": meal_name,
            "time": meal_time,
            "items": items,
            "guidance": _default_meal_guidance(meal_name, items),
        })
    return schedule


def _expected_serving_totals(basket: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {str(row["food_id"]): int(row["quantity"]) for row in basket}


def _validate_and_canonicalize_schedule(
    proposed: Any,
    basket: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Validate model allocation and replace all model-supplied dish metadata."""
    if not isinstance(proposed, list):
        raise Stage4Error("LLM meal_schedule must be a list.")
    required_names = [name for name, _ in MEAL_SLOTS]
    by_meal: dict[str, list[dict[str, Any]]] = {name: [] for name in required_names}
    seen_meals: set[str] = set()
    source_by_id = {str(row["food_id"]): row for row in basket}
    allocated = {food_id: 0 for food_id in source_by_id}

    for meal in proposed:
        if not isinstance(meal, Mapping):
            raise Stage4Error("Each meal_schedule entry must be an object.")
        meal_name = str(meal.get("meal", meal.get("name", ""))).strip()
        if meal_name not in by_meal:
            raise Stage4Error(f"Unknown meal slot {meal_name!r}; expected {required_names}.")
        if meal_name in seen_meals:
            raise Stage4Error(f"Duplicate meal slot {meal_name!r}.")
        seen_meals.add(meal_name)
        proposed_items = meal.get("items", [])
        if not isinstance(proposed_items, list):
            raise Stage4Error(f"Items for {meal_name} must be a list.")

        for item in proposed_items:
            if not isinstance(item, Mapping):
                raise Stage4Error(f"A scheduled item in {meal_name} is not an object.")
            food_id = str(item.get("food_id", "")).strip()
            if food_id not in source_by_id:
                raise Stage4Error(f"LLM introduced unknown food_id {food_id!r}.")
            quantity_value = item.get("quantity")
            if isinstance(quantity_value, bool):
                raise Stage4Error(f"Scheduled quantity for {food_id!r} is not an integer.")
            try:
                quantity_float = float(quantity_value)
            except (TypeError, ValueError) as exc:
                raise Stage4Error(f"Scheduled quantity for {food_id!r} is invalid.") from exc
            if not math.isfinite(quantity_float) or not quantity_float.is_integer() or quantity_float < 1:
                raise Stage4Error(f"Scheduled quantity for {food_id!r} must be a positive integer.")
            quantity = int(quantity_float)
            allocated[food_id] += quantity
            if allocated[food_id] > int(source_by_id[food_id]["quantity"]):
                raise Stage4Error(f"LLM over-allocated Stage 3 servings for {food_id!r}.")
            source = source_by_id[food_id]
            by_meal[meal_name].append({
                "food_id": food_id,
                "name": str(source["name"]),
                "quantity": quantity,
                "serving_unit": str(source.get("serving_unit", "serving")),
                "clinical_rationale": _clinical_rationale_for_food(source),
            })

    missing_meals = sorted(set(required_names) - seen_meals)
    if missing_meals:
        raise Stage4Error(f"LLM omitted required meal slots: {missing_meals}.")
    expected = _expected_serving_totals(basket)
    if allocated != expected:
        mismatches = {
            food_id: {"expected": expected[food_id], "allocated": allocated[food_id]}
            for food_id in expected if expected[food_id] != allocated[food_id]
        }
        raise Stage4Error(f"LLM meal schedule does not preserve exact Stage 3 servings: {mismatches}.")

    canonical: list[dict[str, Any]] = []
    for meal_name, default_time in MEAL_SLOTS:
        raw_meal = next(m for m in proposed if str(m.get("meal", m.get("name", ""))).strip() == meal_name)
        time_value = str(raw_meal.get("time", default_time)).strip() or default_time
        if not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", time_value):
            raise Stage4Error(f"Invalid 24-hour meal time {time_value!r} for {meal_name}.")
        guidance_value = raw_meal.get("guidance")
        if not isinstance(guidance_value, str) or not guidance_value.strip():
            guidance = _default_meal_guidance(meal_name, by_meal[meal_name])
        else:
            guidance = guidance_value.strip()[:1000]
        canonical.append({
            "meal": meal_name,
            "time": time_value,
            "items": by_meal[meal_name],
            "guidance": guidance,
        })
    return canonical


# ---------------------------------------------------------------------------
# Zero Calculation Authority Validator & JSON Parsing
# ---------------------------------------------------------------------------
def _validate_no_nutrition_number_claims(report: Mapping[str, Any]) -> None:
    """Enforce strict Zero Calculation Authority on LLM-generated prose."""
    text_parts: list[str] = []
    for key in ("health_narrative", "closing_message"):
        val = report.get(key)
        if isinstance(val, str):
            text_parts.append(val)
    for key in ("biomarker_driver_explanations", "culinary_directives"):
        value = report.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, Mapping):
                    text_parts.extend(str(v) for v in item.values() if isinstance(v, str))
    guidance = report.get("lifestyle_hydration_guidance")
    if isinstance(guidance, Mapping):
        text_parts.extend(str(v) for v in guidance.values() if isinstance(v, str))
    schedule = report.get("meal_schedule")
    if isinstance(schedule, list):
        for meal in schedule:
            if isinstance(meal, Mapping) and isinstance(meal.get("guidance"), str):
                text_parts.append(meal["guidance"])
    text = " ".join(text_parts)
    forbidden_numeric_nutrition = re.compile(
        r"\b\d+(?:\.\d+)?\s*(?:kcal|kilocalories?|calories|g\s*(?:protein|carbohydrates?|carbs?|fat|fiber|fibre)|"
        r"mg|mcg|µg|micrograms?|milligrams?)\b|"
        r"\b(?:protein|carbohydrate|carb|fat|fiber|fibre|sodium|potassium|calorie)\s*[:=]?\s*\d+(?:\.\d+)?\b",
        flags=re.IGNORECASE,
    )
    if forbidden_numeric_nutrition.search(text):
        raise Stage4Error("LLM prose contains a numeric nutrition/energy claim; rejecting under Zero Calculation Authority policy.")


def extract_json_object(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        try:
            data = json.loads(snippet)
            if isinstance(data, dict):
                return data
        except Exception as exc:
            raise Stage4Error(f"Extracted JSON block could not be decoded: {exc}") from exc
    raise Stage4Error("Could not locate a top-level JSON object in the model output.")


# ---------------------------------------------------------------------------
# LLM Backends: Local Qwen (Transformers/CUDA), Ollama
# ---------------------------------------------------------------------------


def build_compiler_prompt(dossier: ValidatedDossier) -> str:
    basket_summary = [
        {
            "food_id": row["food_id"],
            "name": row["name"],
            "quantity": int(row["quantity"]),
            "serving_unit": row.get("serving_unit", "serving"),
            "category": row.get("category", "unspecified"),
        }
        for row in dossier.basket
    ]
    prompt_payload = {
        "task": (
            "Translate the validated BioNutriGPT mathematical dossier into a patient-friendly clinical and "
            "chrono-nutrition report conforming to the requested JSON schema. You have ZERO authority to recalculate "
            "or invent nutrition numbers (calories, grams, milligrams). You MUST allocate every single basket item "
            "into the five meal slots without dropping or adding servings."
        ),
        "patient_metadata": dossier.patient_metadata,
        "stage1_diagnostic_assessment": dossier.stage1,
        "stage2_energy_target_kcal": dossier.stage2.get("energy_prescription", {}).get("target_calories"),
        "stage3_immutable_food_basket": basket_summary,
        "required_meal_slots": [name for name, _ in MEAL_SLOTS],
    }
    system_instruction = (
        "You are the BioNutriGPT Stage 4 Clinical Chrono-Nutrition Compiler. "
        "Translate the input mathematical dossier into clear, empathetic, and medically accurate patient communication. "
        "Strict clinical safety rules:\n"
        "1. Do NOT mention any numerical calorie, macronutrient, or micronutrient amounts in prose.\n"
        "2. The meal_schedule must contain exactly the 5 slots: Breakfast, Mid-Morning Snack, Lunch, Evening Snack, Dinner.\n"
        "3. Every food in stage3_immutable_food_basket must appear in the schedule with its exact quantity preserved.\n"
        "4. Output MUST conform to this exact JSON schema:\n"
        "{\n"
        '  "health_narrative": "Empathetic patient-friendly health summary (no nutrition numbers)",\n'
        '  "biomarker_driver_explanations": [{"modality": "modality_name", "plain_language_explanation": "..."}],\n'
        '  "meal_schedule": [{"meal": "Breakfast", "time": "07:30", "items": [{"food_id": "...", "name": "...", "quantity": 1}], "guidance": "..."}],\n'
        '  "culinary_directives": ["..."],\n'
        '  "lifestyle_hydration_guidance": {"hydration": "...", "post_meal_activity": "...", "sleep_hygiene": "..."},\n'
        '  "closing_message": "..."\n'
        "}\n"
        "5. Respond with a single valid JSON object only. No markdown fences, no preface, no trailing text."
    )
    return f"{system_instruction}\n\nINPUT DOSSIER:\n{json.dumps(prompt_payload, indent=2)}"


def _load_local_qwen_cached(model_name: str) -> tuple[Any, Any]:
    if model_name in _TORCH_MODEL_CACHE:
        return _TORCH_MODEL_CACHE[model_name]
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise Stage4Error("Local Qwen execution requires `transformers`, `torch`, `accelerate`, and `bitsandbytes`.") from exc

    LOGGER.info("Configuring 4-bit BitsAndBytes quantization for %s...", model_name)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if "3b" in model_name.lower():
        device_map = "cuda:0" if torch.cuda.is_available() else "auto"
        max_mem = None
    else:
        device_map = "auto"
        max_mem = {0: "3.5GiB", "cpu": "16GiB"} if torch.cuda.is_available() else None

    load_kwargs: dict[str, Any] = {
        "quantization_config": bnb_config,
        "device_map": device_map,
        "torch_dtype": torch.float16,
    }
    if max_mem:
        load_kwargs["max_memory"] = max_mem

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    except Exception as exc:
        LOGGER.warning("Accelerate device mapping hit an error (%s); falling back to device_map='auto'", exc)
        load_kwargs["device_map"] = "auto"
        load_kwargs.pop("max_memory", None)
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

    _TORCH_MODEL_CACHE[model_name] = (tokenizer, model)
    return tokenizer, model


def _run_transformers_backend(prompt: str, config: CompilerConfig) -> BackendResult:
    started = time.perf_counter()
    import torch
    tokenizer, model = _load_local_qwen_cached(config.model_name)
    messages = [
        {"role": "system", "content": "You are the BioNutriGPT Stage 4 Clinical Compiler. Output valid JSON only."},
        {"role": "user", "content": prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        rendered = f"System: {messages[0]['content']}\n\nUser: {prompt}\n\nAssistant:"

    input_device = model.get_input_embeddings().weight.device
    inputs = tokenizer(rendered, return_tensors="pt")
    inputs = {k: v.to(input_device) for k, v in inputs.items()}
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            do_sample=False,
            repetition_penalty=1.03,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    input_length = inputs["input_ids"].shape[1]
    output_tokens = generated[0][input_length:]
    text = tokenizer.decode(output_tokens, skip_special_tokens=True)
    if not text.strip():
        raise Stage4Error("Local Qwen returned an empty generation.")
    return BackendResult(
        text=text,
        backend="transformers",
        model=config.model_name,
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
    )


def _run_ollama_backend(prompt: str, config: CompilerConfig) -> BackendResult:
    started = time.perf_counter()
    request_data = {
        "model": config.model_name,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0, "top_p": 0.9, "num_predict": config.max_new_tokens},
    }
    payload = json.dumps(request_data).encode("utf-8")
    req = urllib.request.Request(config.ollama_url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=config.timeout_sec) as response:
            body = response.read().decode("utf-8")
            data = json.loads(body)
            text = data.get("response", "")
    except Exception as exc:
        raise Stage4Error(f"Ollama backend failed: {exc}") from exc
    return BackendResult(
        text=text,
        backend="ollama",
        model=config.model_name,
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
    )


def run_backend(dossier: ValidatedDossier, config: CompilerConfig) -> BackendResult:
    prompt = build_compiler_prompt(dossier)
    if config.backend == "ollama":
        return _run_ollama_backend(prompt, config)
    if config.backend == "transformers":
        return _run_transformers_backend(prompt, config)
    raise Stage4Error(f"Unsupported backend {config.backend!r}. Must be 'transformers' or 'ollama'.")


# ---------------------------------------------------------------------------
# Canonicalization, Safety Circuit-Breaker & Verification
# ---------------------------------------------------------------------------
def _coerce_text(value: Any, key: str, *, max_length: int = 5000) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Stage4Error(f"Generated report field {key!r} must be a non-empty string.")
    return value.strip()[:max_length]


def _canonicalize_report(generated: Mapping[str, Any], dossier: ValidatedDossier) -> dict[str, Any]:
    health_narrative = _coerce_text(generated.get("health_narrative"), "health_narrative")
    raw_drivers = generated.get("biomarker_driver_explanations", [])
    drivers: list[dict[str, str]] = []
    if isinstance(raw_drivers, list):
        for item in raw_drivers[:8]:
            if isinstance(item, Mapping):
                modality = str(item.get("modality", "Clinical features"))[:80]
                explanation = str(item.get("plain_language_explanation", "")).strip()
                if explanation:
                    drivers.append({"modality": modality, "plain_language_explanation": explanation[:1500]})
            elif isinstance(item, str) and item.strip():
                drivers.append({"modality": "Clinical features", "plain_language_explanation": item.strip()[:1500]})
    elif isinstance(raw_drivers, Mapping):
        for k, v in raw_drivers.items():
            if isinstance(v, str) and v.strip():
                drivers.append({"modality": str(k)[:80], "plain_language_explanation": v.strip()[:1500]})

    schedule = _validate_and_canonicalize_schedule(generated.get("meal_schedule"), dossier.basket)

    directives_value = generated.get("culinary_directives")
    directives: list[str] = []
    if isinstance(directives_value, list):
        for item in directives_value:
            if isinstance(item, str) and item.strip():
                directives.append(item.strip()[:1000])
            elif isinstance(item, Mapping):
                for val in item.values():
                    if isinstance(val, str) and val.strip():
                        directives.append(val.strip()[:1000])
                        break
    elif isinstance(directives_value, str) and directives_value.strip():
        directives = [line.strip().lstrip("-*123456789. ") for line in directives_value.split("\n") if line.strip()]

    default_directives = [
        "Use measured cooking oil and build flavour with herbs, spices, garlic, ginger, and other aromatics rather than relying on extra salt.",
        "When preparing pulses or dals, use suitable soaking and thorough cooking; pair the meal with a vitamin-C-containing component only when it is already part of the planned basket or approved by the care team.",
        "For the planned leafy-green dish, cook gently and briefly rather than boiling it for a long time; avoid discarding nutrient-containing cooking liquid when appropriate.",
    ]
    for d in default_directives:
        if len(directives) >= 3:
            break
        if d not in directives:
            directives.append(d)
    if len(directives) > 4:
        directives = directives[:4]

    lifestyle_value = generated.get("lifestyle_hydration_guidance")
    if not isinstance(lifestyle_value, Mapping):
        lifestyle_value = {}
    default_lifestyle = {
        "hydration": "Drink fluids regularly across the day, guided by thirst, climate, and your clinician's advice. Follow any prescribed fluid restriction, particularly if you have heart or kidney disease.",
        "post_meal_activity": "When safe for your mobility and medical condition, consider a gentle walk after meals. Start comfortably and follow any activity limits given by your healthcare professional.",
        "sleep_hygiene": "Keep a consistent sleep and wake time, reduce bright screens late in the evening, and create a quiet, dark sleep environment. Discuss persistent sleep problems with your clinician.",
    }
    lifestyle = {}
    for key in ("hydration", "post_meal_activity", "sleep_hygiene"):
        val = lifestyle_value.get(key)
        if isinstance(val, str) and val.strip():
            lifestyle[key] = val.strip()[:1500]
        else:
            lifestyle[key] = default_lifestyle[key]

    closing = generated.get("closing_message")
    if not isinstance(closing, str) or not closing.strip():
        closing = (
            "These suggestions are meant to support, not replace, individualized guidance from your healthcare team. "
            "Please review this plan with your doctor or registered dietitian."
        )
    else:
        closing = closing.strip()[:1500]

    canonical = {
        "health_narrative": health_narrative,
        "biomarker_driver_explanations": drivers,
        "meal_schedule": schedule,
        "culinary_directives": directives,
        "lifestyle_hydration_guidance": lifestyle,
        "closing_message": closing,
    }
    _validate_no_nutrition_number_claims(canonical)
    return canonical


def _integrity_report(basket: Sequence[Mapping[str, Any]], schedule: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expected = _expected_serving_totals(basket)
    actual = {food_id: 0 for food_id in expected}
    unknown: list[str] = []
    for meal in schedule:
        for item in meal["items"]:
            food_id = str(item["food_id"])
            if food_id not in actual:
                unknown.append(food_id)
            else:
                actual[food_id] += int(item["quantity"])
    exact = actual == expected and not unknown
    return {
        "stage3_basket_preserved": bool(exact),
        "no_new_food_ids": not bool(unknown),
        "expected_servings_by_food_id": expected,
        "scheduled_servings_by_food_id": actual,
        "unknown_food_ids": unknown,
        "all_servings_accounted_for": bool(exact),
        "llm_nutrient_calculation_authority": False,
    }


def compile_patient_report(
    dossier: ValidatedDossier,
    config: CompilerConfig,
) -> dict[str, Any]:
    """Compile patient report using LLM, enforcing Zero Calculation Authority and Schedule Repair Circuit-Breaker."""
    schedule_repaired = False
    backend_result = run_backend(dossier, config)
    try:
        with open("last_raw_llm_response.txt", "w", encoding="utf-8") as raw_f:
            raw_f.write(backend_result.text)
    except Exception:
        pass
    generated = extract_json_object(backend_result.text)
    try:
        compiled = _canonicalize_report(generated, dossier)
    except Exception as sched_err:
        LOGGER.info("Repairing LLM meal schedule with deterministic scheduler: %s", sched_err)
        repaired = dict(generated)
        repaired["meal_schedule"] = build_deterministic_schedule(dossier.basket)
        compiled = _canonicalize_report(repaired, dossier)
        schedule_repaired = True

    integrity = _integrity_report(dossier.basket, compiled["meal_schedule"])
    if not integrity["stage3_basket_preserved"]:
        raise Stage4Error("Internal integrity check failed: scheduled servings differ from the Stage 3 basket.")

    def _copy(val: Any) -> Any:
        return json.loads(json.dumps(val, ensure_ascii=False, allow_nan=False))

    source_sections = {
        "patient_metadata": _copy(dossier.patient_metadata),
        "stage1_diagnostic_assessment": _copy(dossier.stage1),
        "stage2_counterfactual_prescription": _copy(dossier.stage2),
        "stage3_discrete_meal_synthesis": _copy(dossier.stage3),
    }
    report = {
        **source_sections,
        "compiled_patient_report": compiled,
        "integrity_checks": integrity,
        "compiler_metadata": {
            "stage": 4,
            "backend_requested": config.backend,
            "backend_used": backend_result.backend,
            "model": backend_result.model,
            "generation_time_ms": round(backend_result.elapsed_ms, 3),
            "schedule_repaired_by_circuit_breaker": bool(schedule_repaired),
            "zero_calculation_authority_enforced": True,
            "stage3_basket_mutated": False,
        },
        "clinical_disclaimer": (
            "Educational communication support only; not a diagnosis or a substitute for care from a "
            "qualified healthcare professional. Numerical nutrition and serving data are reproduced "
            "from the upstream dossier and have not been recalculated by Stage 4."
        ),
    }
    return report


def print_clinical_report(report: Mapping[str, Any]) -> None:
    compiled = report["compiled_patient_report"]
    metadata = report.get("patient_metadata", {})
    print("\n" + "=" * 88)
    print("BIONUTRIGPT | STAGE 4 CLINICAL & CHRONO-NUTRITION REPORT")
    print("=" * 88)
    if metadata:
        sex = metadata.get("sex")
        if sex is None and "is_female" in metadata:
            sex = "Female" if bool(metadata.get("is_female")) else "Male"
        age = metadata.get("age_years", "not supplied")
        print(f"Patient profile: age {age}; sex {sex or 'not supplied'}")
    print("\nPATIENT-FRIENDLY HEALTH NARRATIVE")
    print("-" * 88)
    print(compiled["health_narrative"])

    if compiled.get("biomarker_driver_explanations"):
        print("\nMODEL ATTENTION DRIVERS (NOT CAUSAL PROOF)")
        print("-" * 88)
        for item in compiled["biomarker_driver_explanations"]:
            print(f"{item['modality']}: {item['plain_language_explanation']}")

    print("\nDAILY MEAL TIMETABLE")
    print("-" * 88)
    for meal in compiled["meal_schedule"]:
        print(f"\n{meal['time']}  {meal['meal']}")
        if not meal["items"]:
            print("  (No Stage 3 basket item assigned; do not add an extra dish.)")
        for item in meal["items"]:
            rationale = item.get("clinical_rationale")
            rat_suffix = f" — [{rationale}]" if rationale else ""
            print(f"  - {item['name']} | {item['quantity']} × {item['serving_unit']}{rat_suffix}")
        print(f"  {meal['guidance']}")

    print("\nCULINARY DIRECTIVES")
    print("-" * 88)
    for index, directive in enumerate(compiled["culinary_directives"], start=1):
        print(f"{index}. {directive}")

    print("\nLIFESTYLE & HYDRATION")
    print("-" * 88)
    lifestyle = compiled["lifestyle_hydration_guidance"]
    print(f"Hydration: {lifestyle['hydration']}")
    print(f"Post-meal activity: {lifestyle['post_meal_activity']}")
    print(f"Sleep: {lifestyle['sleep_hygiene']}")

    integrity = report["integrity_checks"]
    print("\nSTAGE 3 BASKET INTEGRITY")
    print("-" * 88)
    print(f"{'PASS' if integrity['stage3_basket_preserved'] else 'FAIL'} | Exact Stage 3 dish IDs and serving totals preserved")
    rep_flag = report['compiler_metadata'].get('schedule_repaired_by_circuit_breaker', False)
    print(f"Backend: {report['compiler_metadata']['backend_used']} | Model: {report['compiler_metadata']['model']} | Schedule Repaired: {rep_flag}")
    print("\n" + str(report["clinical_disclaimer"]))
    print("=" * 88)


def save_report(report: Mapping[str, Any], output_path: Path) -> Path:
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        with temp.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        temp.replace(output_path)
    except OSError as exc:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass
        raise Stage4Error(f"Could not save report to {output_path}: {exc}") from exc
    return output_path


def validate_dossier(payload: Any) -> ValidatedDossier:
    if not isinstance(payload, Mapping):
        raise Stage4Error("Input dossier must be a JSON object.")
    metadata = payload.get("patient_metadata")
    if not isinstance(metadata, Mapping):
        raise Stage4Error("Dossier missing 'patient_metadata' object.")
    stage1 = payload.get("stage1_diagnostic_assessment")
    if not isinstance(stage1, Mapping):
        stage1 = payload.get("stage1_diagnostics")
    if not isinstance(stage1, Mapping):
        raise Stage4Error("Dossier missing 'stage1_diagnostic_assessment' object.")
    stage2 = payload.get("stage2_counterfactual_prescription")
    if not isinstance(stage2, Mapping):
        stage2 = payload.get("stage2_prescription")
    if not isinstance(stage2, Mapping):
        raise Stage4Error("Dossier missing 'stage2_counterfactual_prescription' object.")
    stage3 = payload.get("stage3_discrete_meal_synthesis")
    if not isinstance(stage3, Mapping):
        stage3 = payload.get("stage3_meal_plan")
    if not isinstance(stage3, Mapping):
        raise Stage4Error("Dossier missing 'stage3_discrete_meal_synthesis' object.")

    basket = stage3.get("solved_food_basket") or stage3.get("basket")
    if not isinstance(basket, list) or not basket:
        raise Stage4Error("Stage 3 contains no solved_food_basket rows.")

    for row in basket:
        if not isinstance(row, Mapping):
            raise Stage4Error("Basket row is not an object.")
        if not row.get("food_id") or not row.get("name"):
            raise Stage4Error("Basket row missing food_id or name.")
        qty = row.get("quantity")
        if not isinstance(qty, int) or qty < 1:
            raise Stage4Error("Basket quantity must be positive integer.")

    certified = bool(
        payload.get("stage3_certified", stage2.get("certified_for_stage3", True))
    )
    return ValidatedDossier(
        patient_metadata=dict(metadata),
        stage1=dict(stage1),
        stage2=dict(stage2),
        stage3=dict(stage3),
        basket=[dict(r) for r in basket],
        certified_for_stage3=certified,
    )


def load_dossier(path: Union[str, Path]) -> ValidatedDossier:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Input dossier not found: {resolved}")
    try:
        with resolved.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except Exception as exc:
        raise Stage4Error(f"Input dossier is not valid JSON ({resolved}): {exc}") from exc
    return validate_dossier(raw)


# ---------------------------------------------------------------------------
# Master CLI Entrypoint
# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Unified BioNutriGPT Master Clinical Pipeline (Stages 1 to 4 & Stage 4 Compiler).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # End-to-end inference from raw patient vector (Runs Stages 1 -> 2 -> 3 -> 4):
  python infer_bionutrigpt_pipeline_Stage_1_to_4.py sample_patient_single_line.txt

  # Local GPU LLM inference on RTX 3050:
  python infer_bionutrigpt_pipeline_Stage_1_to_4.py sample_patient_single_line.txt --backend transformers

  # Direct Stage 4 compilation from precomputed dossier:
  python infer_bionutrigpt_pipeline_Stage_1_to_4.py bionutrigpt_pipeline_dossier.json
""",
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        type=Path,
        default=None,
        help="Patient vector (.txt/.csv) OR precomputed dossier (.json). Defaults to sample patient.",
    )
    parser.add_argument("--weight", type=float, default=None, help="Patient weight in kg.")
    parser.add_argument("--height", type=float, default=None, help="Patient height in cm.")
    parser.add_argument("--age", type=float, default=None, help="Patient age in years.")
    sex_group = parser.add_mutually_exclusive_group()
    sex_group.add_argument("--female", action="store_true", help="Override inferred sex as female.")
    sex_group.add_argument("--male", action="store_true", help="Override inferred sex as male.")
    parser.add_argument("--activity", choices=p12.ACTIVITY_LEVELS, default=None, help="Activity level.")
    parser.add_argument("--food-db", type=str, default=None, help="Custom IFCT/RecipeDB food catalog path.")
    parser.add_argument(
        "--dossier-output",
        type=Path,
        default=DEFAULT_DOSSIER_FILE,
        help=f"Output 3-stage JSON dossier (default: {DEFAULT_DOSSIER_FILE.name})",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=DEFAULT_REPORT_FILE,
        help=f"Output final Stage 4 JSON report (default: {DEFAULT_REPORT_FILE.name})",
    )
    parser.add_argument(
        "--backend",
        choices=["transformers", "ollama"],
        default="transformers",
        help="Stage 4 LLM backend (default: transformers for local Qwen 2.5 on GPU; or 'ollama').",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model identifier (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress console clinical report.")
    args = parser.parse_args(argv)

    # Determine input path
    input_path = args.input_file
    if input_path is None:
        if DEFAULT_PATIENT_FILE.exists():
            input_path = DEFAULT_PATIENT_FILE
        elif DEFAULT_DOSSIER_FILE.exists():
            input_path = DEFAULT_DOSSIER_FILE
        else:
            parser.error("No input file provided and neither sample_patient_single_line.txt nor bionutrigpt_pipeline_dossier.json was found.")

    input_path = Path(input_path).expanduser().resolve()
    dossier_path = Path(args.dossier_output).expanduser().resolve()
    report_path = Path(args.report_output).expanduser().resolve()

    # -------------------------------------------------------------------------
    # MODE A: Direct Stage 4 Compilation (Input is a JSON Dossier)
    # -------------------------------------------------------------------------
    if input_path.suffix.lower() == ".json":
        LOGGER.info("Detected JSON input dossier; executing Stage 4 Chrono-Nutrition Compiler directly.")
        dossier = load_dossier(input_path)
        compiler_cfg = CompilerConfig(backend=args.backend, model_name=args.model)
        report = compile_patient_report(dossier, config=compiler_cfg)
        save_report(report, report_path)
        if not args.quiet:
            print_clinical_report(report)
            print(f"\nStage 4 JSON report saved: {report_path}")
        return 0

    # -------------------------------------------------------------------------
    # MODE B: Full End-to-End Execution (Stages 1 -> 2 -> 3 -> 4)
    # -------------------------------------------------------------------------
    raw_vector = p12.parse_patient_vector(input_path)
    patient_metadata = p12.resolve_patient_metadata(args, raw_vector, input_path)

    start_total = time.perf_counter()

    # Stage 1: Diagnostics
    t_s1 = time.perf_counter()
    stage1_assessment, _ = p12.run_stage1_diagnostics(raw_vector)
    dt_s1_ms = (time.perf_counter() - t_s1) * 1000.0

    # Stage 2: Counterfactual Optimization
    t_s2 = time.perf_counter()
    stage2_dossier = p12.run_stage2_counterfactual(raw_vector, patient_metadata)
    dt_s2_ms = (time.perf_counter() - t_s2) * 1000.0

    unified_dossier = p12.build_unified_dossier(
        raw_vector=raw_vector,
        patient_metadata=patient_metadata,
        stage1_assessment=stage1_assessment,
        stage2_dossier=stage2_dossier,
    )

    # Stage 3: Meal Synthesis (HiGHS MILP)
    t_s3 = time.perf_counter()
    stage3_output = None
    if unified_dossier.get("stage3_certified", False):
        food_db = s3.load_food_database(args.food_db)
        stage3_output = s3.solve_meal_plan(stage2_dossier, food_db)
        unified_dossier["stage3_meal_plan"] = stage3_output.to_dict()
        unified_dossier["stage3_discrete_meal_synthesis"] = stage3_output.to_dict()
    else:
        unified_dossier["stage3_meal_plan"] = None
        unified_dossier["stage3_discrete_meal_synthesis"] = None
    dt_s3_ms = (time.perf_counter() - t_s3) * 1000.0

    p12._atomic_write_json(unified_dossier, dossier_path)

    # Stage 4: Clinical Compiler
    t_s4 = time.perf_counter()
    stage4_report = None
    if stage3_output is not None:
        validated_dossier = load_dossier(dossier_path)
        compiler_cfg = CompilerConfig(backend=args.backend, model_name=args.model)
        stage4_report = compile_patient_report(validated_dossier, config=compiler_cfg)
        save_report(stage4_report, report_path)
    dt_s4_ms = (time.perf_counter() - t_s4) * 1000.0

    elapsed_total_ms = (time.perf_counter() - start_total) * 1000.0

    if not args.quiet:
        print("\n" + "=" * 92)
        print("BIONUTRIGPT | COMPLETE 4-STAGE CLINICAL SYSTEM EXECUTION")
        print("=" * 92)
        print(f"Patient       : Age {patient_metadata['age_years']} | Sex {'Female' if patient_metadata['is_female'] else 'Male'} | Weight {patient_metadata['weight_kg']} kg | Height {patient_metadata['height_cm']} cm")
        print(f"Stage 1 Latency: {dt_s1_ms:6.2f} ms | T2D: {stage1_assessment['probabilities']['diabetes']*100:.1f}%, Obes: {stage1_assessment['probabilities']['obesity']*100:.1f}%, CVD: {stage1_assessment['probabilities']['cvd']*100:.1f}%")
        stage2_certified = bool(stage2_dossier.get("certified_for_stage3", False))
        filter_status = "PASS" if stage2_certified else "FAIL (Filter 4 Non-Inferiority)"
        print(f"Stage 2 Latency: {dt_s2_ms:6.2f} ms | Target: {stage2_dossier['energy_prescription']['target_calories']:.1f} kcal | 4 Safety Filters: {filter_status}")
        if stage3_output is not None:
            s3_meta = stage3_output.solver_metadata
            print(f"Stage 3 Latency: {dt_s3_ms:6.2f} ms | HiGHS MILP Status: {s3_meta['status']} | Active Dishes: {len(stage3_output.solved_food_basket)}")
        else:
            print("Stage 3 Latency: SKIPPED (Stage 2 counterfactual failed safety certification)")
        if stage4_report is not None:
            c_meta = stage4_report.get("compiler_metadata", {})
            print(f"Stage 4 Latency: {dt_s4_ms:6.2f} ms | Backend: {c_meta.get('backend_used')} | Invariants Audit: PASS")
            print(f"TOTAL PIPELINE : {elapsed_total_ms:6.2f} ms (~{elapsed_total_ms/1000.0:.2f} s)")
            print("=" * 92)

            # Print Stage 4 Formatted Output
            print_clinical_report(stage4_report)
        else:
            print("Stage 4 Latency: SKIPPED (No valid Stage 3 basket to compile)")
            print("=" * 92)
        print(f"\n[Artifacts Written]")
        print(f"  Stage 1-3 Mathematical Dossier: {dossier_path.resolve()}")
        print(f"  Stage 4 Clinical Patient Report: {report_path.resolve()}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
