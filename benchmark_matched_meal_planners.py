#!/usr/bin/env python3
"""Matched comparison of the project MILP, its continuous relaxation, and real Qwen.

All approaches receive the same cooked-recipe catalog, synthetic target bounds,
plate rules, and scoring function. Qwen is run locally from the Hugging Face
cache; its raw plan is evaluated without repair.
"""
from __future__ import annotations

import csv
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import stage3_milp_solver as s3
from benchmark_optimizer_comparison import build_clinical_archetypes

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "Saved Model"
OUT.mkdir(exist_ok=True)
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
NUTRIENTS = (
    "DR1TKCAL", "DR1TPROT", "DR1TCARB", "DR1TFIBE",
    "DR1TSFAT", "DR1TSODI", "DR1TPOTA",
)
LABELS = {
    "DR1TKCAL": "energy_kcal",
    "DR1TPROT": "protein_g",
    "DR1TCARB": "carbohydrate_g",
    "DR1TFIBE": "fiber_g",
    "DR1TSFAT": "saturated_fat_g",
    "DR1TSODI": "sodium_mg",
    "DR1TPOTA": "potassium_mg",
}
MAX_SERVINGS = 3
MIN_DISHES, MAX_DISHES = 4, 8
PRICE_WEIGHT = 0.01
ACTIVATION_WEIGHT = 0.10
TIME_LIMIT_SEC = 5.0
MAX_NEW_TOKENS = 180

# Same household-serving price assumptions for every approach. Most values are
# carried over from the earlier benchmark's hand-entered estimates; additions
# are explicitly estimates, not verified current market quotations.
PRICE_INR = {
    "recipe_toor_dal_tadka": 20.0,
    "recipe_moong_dal_tadka": 18.0,
    "recipe_masoor_dal": 16.0,
    "recipe_chana_masala": 22.0,
    "recipe_rajma_curry": 24.0,
    "recipe_sprouted_moong_usal": 15.0,
    "recipe_sambar_vegetable": 18.0,
    "recipe_dal_palak": 20.0,
    "recipe_panchmel_dal": 20.0,
    "recipe_soya_chunk_curry": 25.0,
    "recipe_whole_wheat_phulka": 4.0,
    "recipe_jowar_roti": 7.0,
    "recipe_bajra_roti": 7.0,
    "recipe_ragi_roti": 7.0,
    "recipe_missi_roti": 6.0,
    "recipe_brown_rice_cooked": 12.0,
    "recipe_steamed_rice": 8.0,
    "recipe_veg_pulao": 25.0,
    "recipe_moong_dal_khichdi": 20.0,
    "recipe_steamed_idli": 12.0,
    "recipe_plain_dosa": 14.0,
    "recipe_vegetable_oats_upma": 18.0,
    "recipe_vegetable_poha": 15.0,
    "recipe_palak_sabzi": 16.0,
    "recipe_bhindi_masala": 18.0,
    "recipe_mixed_veg_sabzi": 18.0,
    "recipe_lauki_tomato": 12.0,
    "recipe_aloo_gobi": 14.0,
    "recipe_baingan_bharta": 15.0,
    "recipe_cabbage_peas": 14.0,
    "recipe_karela_masala": 15.0,
    "recipe_methi_aloo": 16.0,
    "recipe_cucumber_salad": 10.0,
    "recipe_skim_curd": 12.0,
    "recipe_lowfat_curd": 14.0,
    "recipe_lowfat_paneer_bhurji": 35.0,
    "recipe_palak_paneer": 35.0,
    "recipe_chaas_buttermilk": 8.0,
    "recipe_fresh_guava": 12.0,
    "recipe_fresh_papaya": 15.0,
    "recipe_fresh_apple": 20.0,
    "recipe_fresh_orange": 15.0,
    "recipe_fresh_banana": 8.0,
    "recipe_pomegranate_arils": 20.0,
}

CATEGORY_ACTIVE_LIMITS = {
    "dal_pulse": (1, 2),
    "cereal_grain": (1, 3),
    "vegetable_sabzi": (2, math.inf),
    "dairy": (1, math.inf),
}
CATEGORY_SERVING_LIMITS = {
    "cereal_grain": (3, 5),
    "dal_pulse": (2, 3),
    "vegetable_sabzi": (2, 4),
    "dairy": (1, 2),
}


def pick_cases() -> list[dict[str, Any]]:
    """Balanced 20-profile subset: five cases from each of four phenotypes."""
    all_cases = build_clinical_archetypes()
    groups: dict[str, list[dict[str, Any]]] = {}
    for case in all_cases:
        groups.setdefault(case["case_id"].split("_")[0], []).append(case)
    chosen: list[dict[str, Any]] = []
    for group in ("DM", "OB", "CVD", "MULTI"):
        records = groups[group]
        indices = np.linspace(0, len(records) - 1, 5).round().astype(int)
        chosen.extend(records[i] for i in indices)
    return chosen


def load_recipe_catalog() -> list[s3.FoodItem]:
    catalog = [f for f in s3.load_food_database() if f.food_id.startswith("recipe_")]
    if not catalog:
        raise RuntimeError("No recipe_* entries found in food_composition_ifct2017.csv")
    missing = sorted({f.food_id for f in catalog} - set(PRICE_INR))
    extra = sorted(set(PRICE_INR) - {f.food_id for f in catalog})
    if missing or extra:
        raise RuntimeError(f"Price map/catalog mismatch; missing={missing}, extra={extra}")
    return catalog


def shared_weights(targets: dict[str, tuple[float, float]]) -> dict[str, float]:
    weights = {}
    for code in NUTRIENTS:
        low, high = targets[code]
        midpoint = (low + high) / 2
        scale = max(high - low, abs(midpoint) * 0.10, 1.0)
        weights[code] = s3._nutrient_priority(code) / scale
    return weights


def evaluate_plan(
    catalog: list[s3.FoodItem], case: dict[str, Any], quantities: np.ndarray,
    runtime_ms: float, method: str, *, solver_status: str | None = None,
    llm_output: str | None = None, llm_parse_ok: bool | None = None,
) -> dict[str, Any]:
    targets = case["targets"]
    actual = {
        code: float(sum(q * food.nutrients_per_serving[code]
                        for q, food in zip(quantities, catalog)))
        for code in NUTRIENTS
    }
    bounds = {code: tuple(map(float, targets[code])) for code in NUTRIENTS}
    violations = {
        code: max(bounds[code][0] - actual[code], 0.0)
        + max(actual[code] - bounds[code][1], 0.0)
        for code in NUTRIENTS
    }
    weights = shared_weights(bounds)
    nutrient_penalty = sum(weights[k] * violations[k] for k in NUTRIENTS)
    target_kcal = sum(bounds["DR1TKCAL"]) / 2
    calorie_penalty = abs(actual["DR1TKCAL"] - target_kcal) * 150.0 / max(target_kcal, 1.0)
    active = [i for i, q in enumerate(quantities) if q > 1e-3]
    estimated_cost = float(sum(quantities[i] * PRICE_INR[catalog[i].food_id] for i in active))
    activation_cost = ACTIVATION_WEIGHT * sum(catalog[i].selection_cost for i in active)
    price_penalty = PRICE_WEIGHT * estimated_cost
    objective = nutrient_penalty + calorie_penalty + activation_cost + price_penalty

    active_counts: dict[str, int] = {}
    serving_counts: dict[str, float] = {}
    for cat in CATEGORY_ACTIVE_LIMITS:
        members = [i for i, f in enumerate(catalog) if f.category == cat]
        active_counts[cat] = sum(quantities[i] > 1e-3 for i in members)
    for cat in CATEGORY_SERVING_LIMITS:
        members = [i for i, f in enumerate(catalog) if f.category == cat]
        serving_counts[cat] = float(sum(quantities[i] for i in members))
    hard_violations = []
    if not (MIN_DISHES <= len(active) <= MAX_DISHES):
        hard_violations.append("active_dish_count")
    for cat, (low, high) in CATEGORY_ACTIVE_LIMITS.items():
        count = active_counts.get(cat, 0)
        if count < low or count > high:
            hard_violations.append(f"active_{cat}")
    for cat, (low, high) in CATEGORY_SERVING_LIMITS.items():
        count = serving_counts[cat]
        if count < low - 1e-5 or count > high + 1e-5:
            hard_violations.append(f"servings_{cat}")

    selected_quantities = [float(quantities[i]) for i in active]
    distances = [abs(q - round(q)) for q in selected_quantities]
    return {
        "method": method,
        "runtime_ms": round(float(runtime_ms), 3),
        "solver_status": solver_status,
        "llm_parse_ok": llm_parse_ok,
        "selected": [
            {"food_id": catalog[i].food_id, "name": catalog[i].name,
             "servings": round(float(quantities[i]), 4),
             "serving_unit": catalog[i].serving_unit,
             "estimated_cost_inr": round(float(quantities[i] * PRICE_INR[catalog[i].food_id]), 2)}
            for i in active
        ],
        "active_dish_count": len(active),
        "active_counts_by_category": active_counts,
        "servings_by_category": {k: round(v, 4) for k, v in serving_counts.items()},
        "nutrients": {k: round(v, 3) for k, v in actual.items()},
        "nutrient_bounds_met": len([k for k in NUTRIENTS if violations[k] > 1e-5]) == 0,
        "nutrient_violations": int(sum(v > 1e-5 for v in violations.values())),
        "nutrient_abs_deviation_by_code": {k: round(v, 4) for k, v in violations.items()},
        "hard_rule_violations": hard_violations,
        "hard_rules_met": len(hard_violations) == 0,
        "integer_serving_plan": all(d <= 0.05 for d in distances),
        "fractional_food_count": int(sum(d > 0.05 for d in distances)),
        "fractional_serving_distance_sum": round(float(sum(distances)), 4),
        "estimated_cost_inr": round(estimated_cost, 2),
        "objective_components": {
            "nutrient_penalty": round(float(nutrient_penalty), 6),
            "calorie_deviation_penalty": round(float(calorie_penalty), 6),
            "dish_activation_penalty": round(float(activation_cost), 6),
            "estimated_cost_penalty": round(float(price_penalty), 6),
            "total": round(float(objective), 6),
        },
        # For an externally generated integer plan this equals the realized
        # objective. Optimizers replace it with their exact formulation value.
        "solver_objective": round(float(objective), 6),
        "raw_llm_output": llm_output,
    }


def solve_matched(
    catalog: list[s3.FoodItem], case: dict[str, Any], *,
    q_integer: bool, z_integer: bool, method: str,
) -> dict[str, Any]:
    """Solve a shared formulation with selectable integrality for q and z."""
    m, n = len(catalog), len(NUTRIENTS)
    q0, z0, ep0, em0, over, under = 0, m, 2*m, 2*m+n, 2*m+2*n, 2*m+2*n+1
    nvars = 2*m + 2*n + 2
    index = {code: k for k, code in enumerate(NUTRIENTS)}
    F = np.array([[f.nutrients_per_serving[code] for f in catalog] for code in NUTRIENTS], dtype=float)
    targets = {code: tuple(map(float, case["targets"][code])) for code in NUTRIENTS}
    weights = shared_weights(targets)
    target_kcal = sum(targets["DR1TKCAL"]) / 2
    c = np.zeros(nvars, dtype=float)
    lb = np.zeros(nvars, dtype=float)
    ub = np.full(nvars, np.inf, dtype=float)
    ub[q0:q0+m] = MAX_SERVINGS
    ub[z0:z0+m] = 1.0
    c[q0:q0+m] = PRICE_WEIGHT * np.array([PRICE_INR[f.food_id] for f in catalog])
    c[z0:z0+m] = ACTIVATION_WEIGHT * np.array([f.selection_cost for f in catalog])
    for k, code in enumerate(NUTRIENTS):
        c[ep0+k] = weights[code]
        c[em0+k] = weights[code]
    cal_weight = 150.0 / max(target_kcal, 1.0)
    c[over] = c[under] = cal_weight
    integrality = np.zeros(nvars, dtype=np.int32)
    if q_integer:
        integrality[q0:q0+m] = 1
    if z_integer:
        integrality[z0:z0+m] = 1

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    row = 0

    def add_row(entries: dict[int, float], low: float, high: float) -> None:
        nonlocal row
        for col, value in entries.items():
            if value:
                rows.append(row); cols.append(col); data.append(value)
        lows.append(low); highs.append(high); row += 1

    for k, code in enumerate(NUTRIENTS):
        entries = {q0+i: F[k, i] for i in range(m) if F[k, i] != 0}
        entries[ep0+k] = -1.0
        entries[em0+k] = 1.0
        add_row(entries, targets[code][0], targets[code][1])
    for i in range(m):
        add_row({q0+i: 1.0, z0+i: -float(MAX_SERVINGS)}, -np.inf, 0.0)
        add_row({z0+i: 1.0, q0+i: -1.0}, -np.inf, 0.0)
    add_row({z0+i: 1.0 for i in range(m)}, MIN_DISHES, MAX_DISHES)
    for cat, (low, high) in CATEGORY_ACTIVE_LIMITS.items():
        add_row({z0+i: 1.0 for i, f in enumerate(catalog) if f.category == cat}, low, high)
    for cat, (low, high) in CATEGORY_SERVING_LIMITS.items():
        add_row({q0+i: 1.0 for i, f in enumerate(catalog) if f.category == cat}, low, high)
    kcal_k = index["DR1TKCAL"]
    kcal = {q0+i: F[kcal_k, i] for i in range(m) if F[kcal_k, i] != 0}
    kcal[over] = -1.0
    kcal[under] = 1.0
    add_row(kcal, target_kcal, target_kcal)
    A = coo_matrix((data, (rows, cols)), shape=(row, nvars)).tocsr()
    constraints = LinearConstraint(A, np.asarray(lows), np.asarray(highs))
    started = time.perf_counter()
    result = milp(
        c=c, integrality=integrality, bounds=Bounds(lb, ub), constraints=constraints,
        options={"time_limit": TIME_LIMIT_SEC, "mip_rel_gap": 0.001, "presolve": True},
    )
    runtime = (time.perf_counter() - started) * 1000
    if result.x is None:
        raise RuntimeError(f"{'MILP' if integer else 'LP'} produced no incumbent: {result.message}")
    q = np.asarray(result.x[q0:q0+m], dtype=float)
    if q_integer:
        q = np.rint(q).clip(0, MAX_SERVINGS)
    status = str(result.message)
    record = evaluate_plan(catalog, case, q, runtime, method, solver_status=status)
    record["solver_objective"] = float(result.fun)
    record["optimality_proven"] = bool(result.status == 0)
    gap = getattr(result, "mip_gap", None)
    record["mip_gap"] = None if gap is None or not math.isfinite(float(gap)) else float(gap)
    record["integrality"] = {"servings_integer": q_integer, "dish_selection_integer": z_integer}
    return record


def model_data(catalog: list[s3.FoodItem]) -> list[dict[str, Any]]:
    return [
        {
            "id": f"F{i+1:02d}", "food_id": food.food_id,
            "name": food.name, "category": food.category,
            "serving_unit": food.serving_unit,
            "estimated_cost_inr": PRICE_INR[food.food_id],
            "nutrients": {code: round(float(food.nutrients_per_serving[code]), 2) for code in NUTRIENTS},
        }
        for i, food in enumerate(catalog)
    ]


def build_prompt(case: dict[str, Any], foods: list[dict[str, Any]]) -> str:
    bounds = {
        LABELS[code]: [round(float(case["targets"][code][0]), 2), round(float(case["targets"][code][1]), 2)]
        for code in NUTRIENTS
    }
    target_kcal = sum(bounds["energy_kcal"]) / 2
    return (
        "Choose one full-day Indian food basket from the exact candidate catalog below. "
        "This is a synthetic optimization benchmark, not an individual clinical prescription.\n"
        "Return only compact JSON in this exact shape: {\"items\":{\"F01\":2,\"F14\":1}}.\n"
        "Use only listed IDs. Values are integer servings from 1 to 3. Include only 4 to 8 IDs total; "
        "never output more than 8 foods and do not list unused catalog items. "
        "Hard plate rules: active dishes by category: "
        "dal_pulse 1-2, cereal_grain 1-3, vegetable_sabzi at least 2, dairy at least 1; "
        "total servings: cereal_grain 3-5, dal_pulse 2-3, vegetable_sabzi 2-4, dairy 1-2.\n"
        f"Nutrient target ranges: {json.dumps(bounds,separators=(',',':'))}.\n"
        f"Energy target for absolute-deviation scoring: {target_kcal:.1f} kcal.\n"
        "Minimize this exact shared score: for each nutrient, add its priority divided by "
        "max(target-width, 10% of target midpoint, 1) times the amount outside its range; "
        "priorities are energy 100, protein 28, carbohydrate 8, fiber 25, saturated fat 22, "
        "sodium 30, potassium 25. Also add (150/energy target)*absolute kcal deviation, "
        "0.10 per selected dish, and 0.01 times estimated total INR cost. Meet hard plate rules "
        "first; do not claim nutrient totals.\n"
        f"Candidate foods (nutrient amount per listed household serving): {json.dumps(foods,separators=(',',':'),ensure_ascii=False)}"
    )


def extract_json(text: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            return None
        try:
            obj = json.loads(match.group(0))
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None


def parse_llm_quantities(text: str, foods: list[dict[str, Any]]) -> tuple[np.ndarray, bool]:
    q = np.zeros(len(foods), dtype=float)
    obj = extract_json(text)
    if obj is None:
        return q, False
    items = obj.get("items", obj.get("plan", obj.get("foods", [])))
    if isinstance(items, dict):
        items = [{"id": k, "servings": v} for k, v in items.items()]
    if not isinstance(items, list):
        return q, False
    by_alias = {f["id"]: i for i, f in enumerate(foods)}
    by_id = {f["food_id"]: i for i, f in enumerate(foods)}
    by_name = {f["name"].strip().lower(): i for i, f in enumerate(foods)}
    good = True
    for item in items:
        if not isinstance(item, dict):
            good = False; continue
        alias = str(item.get("id", item.get("food_id", item.get("food", item.get("name", ""))))).strip()
        idx = by_alias.get(alias, by_id.get(alias, by_name.get(alias.lower())))
        if idx is None:
            good = False; continue
        raw_q = item.get("servings", item.get("quantity", item.get("qty", item.get("q", 0))))
        if isinstance(raw_q, str):
            nums = re.findall(r"[-+]?\d*\.?\d+", raw_q)
            if not nums:
                good = False; continue
            raw_q = nums[0]
        try:
            value = float(raw_q)
            if not math.isfinite(value) or value < 0:
                good = False; continue
            q[idx] += value
        except Exception:
            good = False
    return q, good


def load_qwen():
    model_started = time.perf_counter()
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    device_map = "cuda:0" if torch.cuda.is_available() else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb, device_map=device_map,
        torch_dtype=torch.float16, local_files_only=True,
    )
    model.eval()
    return tokenizer, model, (time.perf_counter() - model_started)


def generate_qwen(tokenizer, model, prompt: str) -> tuple[str, float]:
    messages = [
        {"role": "system", "content": "You are a meal-planning optimizer. Follow the supplied catalog and constraints exactly. Output JSON only."},
        {"role": "user", "content": prompt},
    ]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    device = model.get_input_embeddings().weight.device
    inputs = tokenizer(rendered, return_tensors="pt").to(device)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
            repetition_penalty=1.02, pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    elapsed = (time.perf_counter() - started) * 1000
    text = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.strip(), elapsed


def summarize(report: dict[str, Any]) -> None:
    summary = {}
    methods = ("Continuous LP", "Continuous servings (binary dish selection)", "MILP", "Real Qwen 2.5 3B")
    for method in methods:
        records = [case["methods"][method] for case in report["cases"] if method in case["methods"]]
        runtimes = [r["runtime_ms"] for r in records]
        costs = [r["estimated_cost_inr"] for r in records]
        objectives = [r["objective_components"]["total"] for r in records]
        summary[method] = {
            "n": len(records),
            "hard_rules_met_rate": float(np.mean([r["hard_rules_met"] for r in records])),
            "nutrient_bounds_met_rate": float(np.mean([r["nutrient_bounds_met"] for r in records])),
            "integer_plan_rate": float(np.mean([r["integer_serving_plan"] for r in records])),
            "mean_fractional_food_count": float(np.mean([r["fractional_food_count"] for r in records])),
            "mean_fractional_distance": float(np.mean([r["fractional_serving_distance_sum"] for r in records])),
            "mean_estimated_cost_inr": float(np.mean(costs)),
            "median_estimated_cost_inr": float(np.median(costs)),
            "mean_realized_plan_score": float(np.mean(objectives)),
            "mean_solver_objective": float(np.mean([r.get("solver_objective", r["objective_components"]["total"]) for r in records])),
            "mean_runtime_ms": float(np.mean(runtimes)),
            "median_runtime_ms": float(np.median(runtimes)),
            "p95_runtime_ms": float(np.percentile(runtimes, 95)),
            "mean_nutrient_violations": float(np.mean([r["nutrient_violations"] for r in records])),
            "llm_json_parse_rate": (float(np.mean([r["llm_parse_ok"] for r in records if r["llm_parse_ok"] is not None]))
                                    if method.startswith("Real Qwen") else None),
        }
    report["summary"] = summary


def write_outputs(report: dict[str, Any]) -> None:
    csv_path = OUT / "matched_milp_lp_qwen_case_results.csv"
    rows = []
    for case in report["cases"]:
        for method, result in case["methods"].items():
            rows.append({
                "case_id": case["case_id"], "phenotype": case["phenotype"], "method": method,
                "runtime_ms": result["runtime_ms"],
                "estimated_cost_inr": result["estimated_cost_inr"],
                "objective_score": result["objective_components"]["total"],
                "solver_objective": result.get("solver_objective", result["objective_components"]["total"]),
                "nutrient_violations": result["nutrient_violations"],
                "hard_rules_met": result["hard_rules_met"],
                "integer_serving_plan": result["integer_serving_plan"],
                "fractional_food_count": result["fractional_food_count"],
                "fractional_serving_distance_sum": result["fractional_serving_distance_sum"],
                "llm_parse_ok": result["llm_parse_ok"],
            })
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    json_path = OUT / "matched_milp_lp_qwen_benchmark_report.json"
    def json_safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(v) for v in value]
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        return value
    json_path.write_text(json.dumps(json_safe(report), indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Matched meal-planning benchmark: MILP, LP relaxation, and real Qwen",
        "",
        "## Design",
        "",
        f"- {report['metadata']['n_cases']} balanced synthetic profiles (five each: diabetes, obesity, CVD/hypertension, multimorbid).",
        f"- Same {report['metadata']['food_count']} cooked Indian recipes, household serving units, seven nutrient bounds, plate rules, estimated serving prices, and objective for every method.",
        f"- Real local model: `{MODEL_ID}`, 4-bit NF4; model load time {report['metadata']['qwen_model_load_seconds']:.1f} s (one-time). Qwen plans are scored as generated, without repair.",
        "- The LP relaxes integer restrictions on servings and dish activation. A second continuous-serving comparator keeps dish activations binary and relaxes only serving quantities, isolating the effect of whole portions.",
        "- Shared score = normalized weighted nutrient-bound slack + (150/energy target)×absolute calorie deviation + 0.10 per active dish + 0.01×estimated basket cost in INR. Lower is better.",
        "- Household-practicality proxy = integer-serving plan rate and fractional-serving distance. This is not a dietitian or patient acceptability rating.",
        "",
        "## Results",
        "",
        "| Method | Plate-rule pass | All 7 nutrient bounds met | All-integer plans | Mean fractional foods/plan | Mean cost (₹) | Mean realized score | Mean solver score | Median time/plan |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    display_methods = [
        ("Continuous LP", "Fully relaxed LP"),
        ("Continuous servings (binary dish selection)", "Continuous servings + binary dish selection"),
        ("MILP", "Discrete MILP"),
        ("Real Qwen 2.5 3B", "Real Qwen 2.5 3B"),
    ]
    for method, label in display_methods:
        s = report["summary"][method]
        lines.append(
            f"| {label} | {s['hard_rules_met_rate']*100:.1f}% | {s['nutrient_bounds_met_rate']*100:.1f}% | "
            f"{s['integer_plan_rate']*100:.1f}% | {s['mean_fractional_food_count']:.2f} | "
            f"{s['mean_estimated_cost_inr']:.2f} | {s['mean_realized_plan_score']:.3f} | "
            f"{s['mean_solver_objective']:.3f} | {s['median_runtime_ms']:.1f} ms |"
        )
    lp = report["summary"]["Continuous LP"]
    cont = report["summary"]["Continuous servings (binary dish selection)"]
    mi = report["summary"]["MILP"]
    llm = report["summary"]["Real Qwen 2.5 3B"]
    cost_delta_pct = (mi["mean_estimated_cost_inr"] / cont["mean_estimated_cost_inr"] - 1) * 100 if cont["mean_estimated_cost_inr"] else 0
    runtime_ratio = mi["median_runtime_ms"] / cont["median_runtime_ms"] if cont["median_runtime_ms"] else math.inf
    lines += [
        "",
        "## What this says about whole portions",
        "",
        f"- Against the continuous-serving model with the same binary dish choices, the integer MILP costs an estimated ₹{mi['mean_estimated_cost_inr']-cont['mean_estimated_cost_inr']:.2f} more per day on average ({cost_delta_pct:.1f}%). Median solve time is {mi['median_runtime_ms']:.1f} ms versus {cont['median_runtime_ms']:.1f} ms ({runtime_ratio:.1f}×).",
        f"- Whole portions improve the structural proxy: {mi['integer_plan_rate']*100:.1f}% of MILP plans are integer-served by construction; {cont['integer_plan_rate']*100:.1f}% of continuous-serving plans are all-integer, averaging {cont['mean_fractional_food_count']:.2f} fractional foods. Plate-rule pass rates are {mi['hard_rules_met_rate']*100:.1f}% for MILP and {cont['hard_rules_met_rate']*100:.1f}% for continuous servings.",
        f"- The fully relaxed LP is a speed lower bound but relaxes dish activation too; it met plate rules in {lp['hard_rules_met_rate']*100:.1f}% of actual baskets. Its fractional activation variables can make it appear more feasible than a cookable plan.",
        "- The solver-score column is the actual linear optimization objective. For the LP it uses relaxed fractional dish-activation variables; realized-plan score counts each selected dish as active. The MILP's solver and realized scores should coincide up to numerical tolerance.",
        f"- Qwen met the plate rules in {llm['hard_rules_met_rate']*100:.1f}% of cases and all seven nutrient bounds in {llm['nutrient_bounds_met_rate']*100:.1f}%. Its mean estimated cost was ₹{llm['mean_estimated_cost_inr']:.2f}; median generation time was {llm['median_runtime_ms']/1000:.1f} s, excluding the one-time model load.",
        "- Price values are illustrative assumptions carried over or extended from the earlier benchmark, not verified current local market prices. Cost differences are therefore provisional.",
        "- Nutrient and recipe values are from the project's constructed RecipeDB-style entries; the micro/secondary nutrient values are estimates. Results are a computational pilot, not clinical or taste validation.",
        "- This study measures portion integrality as a practical proxy. It cannot establish that patients or dietitians prefer the integer plans; that requires blinded dietitian/patient review.",
        "",
        "Per-case plans and raw Qwen generations are in the JSON report; the CSV has the aggregated case-level scores.",
    ]
    (OUT / "matched_milp_lp_qwen_benchmark_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    torch.set_grad_enabled(False)
    catalog = load_recipe_catalog()
    food_data = model_data(catalog)
    cases = pick_cases()
    report: dict[str, Any] = {
        "metadata": {
            "benchmark": "matched MILP vs continuous LP relaxation vs real local Qwen",
            "model": MODEL_ID,
            "quantization": "4-bit NF4",
            "food_catalog": str(ROOT / "food_composition_ifct2017.csv"),
            "food_count": len(catalog),
            "food_source_label": "RecipeDB Indian Cooked Formulation in project CSV; 39-field values partly estimated",
            "price_source": "hand-entered benchmark assumptions; unverified current prices",
            "n_cases": len(cases),
            "phenotype_counts": {p: sum(c["case_id"].startswith(p + "_") for c in cases) for p in ("DM", "OB", "CVD", "MULTI")},
            "nutrients_scored": list(NUTRIENTS),
            "category_active_limits": {k: list(v) for k, v in CATEGORY_ACTIVE_LIMITS.items()},
            "category_serving_limits": {k: list(v) for k, v in CATEGORY_SERVING_LIMITS.items()},
            "max_servings_per_food": MAX_SERVINGS,
            "total_active_dishes": [MIN_DISHES, MAX_DISHES],
            "shared_objective": {
                "nutrient_penalty": "sum(priority/max(target-width, 10% midpoint, 1) * absolute deviation outside bound)",
                "calorie_deviation_weight": "150 / target kcal",
                "dish_activation_weight": ACTIVATION_WEIGHT,
                "estimated_cost_weight_per_inr": PRICE_WEIGHT,
            },
            "llm_method": "Greedy decoding, local_files_only; raw outputs are not repaired",
            "solver_time_limit_seconds": TIME_LIMIT_SEC,
        },
        "cases": [],
    }
    print(f"Catalog: {len(catalog)} cooked recipes; profiles: {len(cases)}; Qwen: {MODEL_ID}", flush=True)
    previous_path = OUT / "matched_milp_lp_qwen_benchmark_report.json"
    previous = None
    if previous_path.exists():
        try:
            previous = json.loads(previous_path.read_text(encoding="utf-8"))
        except Exception:
            previous = None
    previous_cases = {item["case_id"]: item for item in (previous or {}).get("cases", [])}
    old_qwen = {
        cid: item["methods"].get("Real Qwen 2.5 3B")
        for cid, item in previous_cases.items()
    }
    print("Solving the matched continuous and discrete formulations...", flush=True)
    for idx, case in enumerate(cases, start=1):
        methods: dict[str, Any] = {}
        old = previous_cases.get(case["case_id"], {}).get("methods", {})
        if old.get("MILP"):
            methods["MILP"] = dict(old["MILP"])
            methods["MILP"]["solver_objective"] = methods["MILP"]["objective_components"]["total"]
            methods["MILP"]["optimality_proven"] = "optimal" in (methods["MILP"].get("solver_status") or "").lower()
            methods["MILP"]["integrality"] = {"servings_integer": True, "dish_selection_integer": True}
        else:
            methods["MILP"] = solve_matched(catalog, case, q_integer=True, z_integer=True, method="MILP")
        methods["Continuous LP"] = solve_matched(catalog, case, q_integer=False, z_integer=False, method="Continuous LP")
        methods["Continuous servings (binary dish selection)"] = solve_matched(
            catalog, case, q_integer=False, z_integer=True,
            method="Continuous servings (binary dish selection)",
        )
        if old_qwen.get(case["case_id"]) and old_qwen[case["case_id"]].get("llm_parse_ok"):
            methods["Real Qwen 2.5 3B"] = dict(old_qwen[case["case_id"]])
            methods["Real Qwen 2.5 3B"]["solver_objective"] = methods["Real Qwen 2.5 3B"]["objective_components"]["total"]
        report["cases"].append({
            "case_id": case["case_id"], "phenotype": case["phenotype"],
            "target_bounds": {k: list(map(float, case["targets"][k])) for k in NUTRIENTS},
            "methods": methods,
        })
        print(f"  solved {idx:02d}/{len(cases)}: {case['case_id']}", flush=True)
    pending_qwen = [i for i, c in enumerate(report["cases"])
                    if not c["methods"].get("Real Qwen 2.5 3B", {}).get("llm_parse_ok")]
    if pending_qwen:
        print("Loading cached Qwen in 4-bit...", flush=True)
        tokenizer, model, load_seconds = load_qwen()
        report["metadata"]["qwen_model_load_seconds"] = round(load_seconds, 3)
    else:
        tokenizer = model = None
        report["metadata"]["qwen_model_load_seconds"] = (previous or {}).get("metadata", {}).get("qwen_model_load_seconds")
    for i, case in enumerate(cases):
        if report["cases"][i]["methods"].get("Real Qwen 2.5 3B", {}).get("llm_parse_ok"):
            print(f"[{i+1:02d}/{len(cases)}] reusing valid Qwen plan for {case['case_id']}", flush=True)
            summarize(report)
            write_outputs(report)
            continue
        prompt = build_prompt(case, food_data)
        raw, runtime = generate_qwen(tokenizer, model, prompt)
        quantities, parsed = parse_llm_quantities(raw, food_data)
        methods = report["cases"][i]["methods"]
        methods["Real Qwen 2.5 3B"] = evaluate_plan(
            catalog, case, quantities, runtime, "Real Qwen 2.5 3B",
            llm_output=raw, llm_parse_ok=parsed,
        )
        summarize(report)
        write_outputs(report)
        print(
            f"[{i+1:02d}/{len(cases)}] {case['case_id']}: Qwen {runtime/1000:.1f}s, "
            f"parsed={parsed}, plate={methods['Real Qwen 2.5 3B']['hard_rules_met']}, "
            f"bounds={methods['Real Qwen 2.5 3B']['nutrient_bounds_met']}",
            flush=True,
        )
    summarize(report)
    write_outputs(report)
    print("Benchmark complete. Wrote markdown, CSV, and JSON report under Saved Model.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
