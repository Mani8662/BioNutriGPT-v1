"""Stage 4 Guarded Conversational Assistant & Controlled Revision Service.

Architecture invariants:
1. Only Stage 3 (HiGHS MILP + independent validate_candidate) may propose or accept
   a revised meal basket. The conversational layer never calculates nutrient totals,
   never relaxes protected Stage 2 bounds, and never edits observed Stage 1 intake.
2. Every factual answer is produced by explicit backend tools:
     - tool_get_plan
     - tool_get_nutrient_totals
     - tool_propose_revision
     - tool_explain_constraint_conflict
3. Persistent versioned plan state is stored via plan_store (v1, v2, ..., rollback),
   strictly separated from the immutable observed patient profile.
4. Unknown allergen records are never interpreted as allergen-free.
"""

from __future__ import annotations

import copy
from difflib import SequenceMatcher
import hashlib
import json
import re
from typing import Any, Mapping
import uuid

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, vstack

import pipeline_service as ps
import plan_store


# ---------------------------------------------------------------------------
# Expanded nutrient alias table covering all 39 dietary fields
# ---------------------------------------------------------------------------
NUTRIENT_ALIASES: dict[str, tuple[str, str, str]] = {
    "calories": ("DR1TKCAL", "calories", "kcal"),
    "energy": ("DR1TKCAL", "energy", "kcal"),
    "kcal": ("DR1TKCAL", "calories", "kcal"),
    "protein": ("DR1TPROT", "protein", "g"),
    "carbohydrate": ("DR1TCARB", "carbohydrates", "g"),
    "carbohydrates": ("DR1TCARB", "carbohydrates", "g"),
    "carbs": ("DR1TCARB", "carbohydrates", "g"),
    "sugar": ("DR1TSUGR", "total sugars", "g"),
    "sugars": ("DR1TSUGR", "total sugars", "g"),
    "fiber": ("DR1TFIBE", "dietary fiber", "g"),
    "fibre": ("DR1TFIBE", "dietary fiber", "g"),
    "saturated fat": ("DR1TSFAT", "saturated fat", "g"),
    "mufa": ("DR1TMFAT", "monounsaturated fat (MUFA)", "g"),
    "monounsaturated": ("DR1TMFAT", "monounsaturated fat (MUFA)", "g"),
    "pufa": ("DR1TPFAT", "polyunsaturated fat (PUFA)", "g"),
    "polyunsaturated": ("DR1TPFAT", "polyunsaturated fat (PUFA)", "g"),
    "total fat": ("DR1TTFAT", "total fat", "g"),
    "fat": ("DR1TTFAT", "total fat", "g"),
    "cholesterol": ("DR1TCHOL", "dietary cholesterol", "mg"),
    "vitamin e": ("DR1TATOC", "vitamin E (alpha-tocopherol)", "mg"),
    "alpha-tocopherol": ("DR1TATOC", "vitamin E (alpha-tocopherol)", "mg"),
    "retinol": ("DR1TRET", "retinol", "µg"),
    "vitamin a": ("DR1TVARA", "vitamin A (RAE)", "µg"),
    "beta-carotene": ("DR1TBCAR", "beta-carotene", "µg"),
    "alpha-carotene": ("DR1TACAR", "alpha-carotene", "µg"),
    "lycopene": ("DR1TLYCO", "lycopene", "µg"),
    "lutein": ("DR1TLZ", "lutein + zeaxanthin", "µg"),
    "thiamin": ("DR1TVB1", "thiamin (vitamin B1)", "mg"),
    "vitamin b1": ("DR1TVB1", "thiamin (vitamin B1)", "mg"),
    "riboflavin": ("DR1TVB2", "riboflavin (vitamin B2)", "mg"),
    "vitamin b2": ("DR1TVB2", "riboflavin (vitamin B2)", "mg"),
    "niacin": ("DR1TNIAC", "niacin (vitamin B3)", "mg"),
    "vitamin b3": ("DR1TNIAC", "niacin (vitamin B3)", "mg"),
    "vitamin b6": ("DR1TVB6", "vitamin B6", "mg"),
    "b6": ("DR1TVB6", "vitamin B6", "mg"),
    "folic acid": ("DR1TFA", "synthetic folic acid", "µg"),
    "food folate": ("DR1TFF", "food folate", "µg"),
    "folate equivalents": ("DR1TFDFE", "dietary folate equivalents (DFE)", "µg"),
    "dfe": ("DR1TFDFE", "dietary folate equivalents (DFE)", "µg"),
    "folate": ("DR1TFDFE", "dietary folate equivalents (DFE)", "µg"),
    "vitamin b12": ("DR1TVB12", "vitamin B12", "µg"),
    "b12": ("DR1TVB12", "vitamin B12", "µg"),
    "cobalamin": ("DR1TVB12", "vitamin B12", "µg"),
    "vitamin c": ("DR1TVC", "vitamin C", "mg"),
    "vitamin k": ("DR1TVK", "vitamin K", "µg"),
    "calcium": ("DR1TCALC", "calcium", "mg"),
    "phosphorus": ("DR1TPHOS", "phosphorus", "mg"),
    "magnesium": ("DR1TMAGN", "magnesium", "mg"),
    "iron": ("DR1TIRON", "iron", "mg"),
    "zinc": ("DR1TZINC", "zinc", "mg"),
    "copper": ("DR1TCOPP", "copper", "mg"),
    "sodium": ("DR1TSODI", "sodium", "mg"),
    "salt": ("DR1TSODI", "sodium", "mg"),
    "potassium": ("DR1TPOTA", "potassium", "mg"),
    "selenium": ("DR1TSELE", "selenium", "µg"),
    "caffeine": ("DR1TCAFF", "caffeine", "mg"),
    "alcohol": ("DR1TALCO", "alcohol", "g"),
}

ALLERGEN_KEYWORDS: dict[str, list[str]] = {
    "dairy": ["dairy", "milk", "curd", "yogurt", "paneer", "lactose", "ghee", "butter"],
    "gluten": ["gluten", "wheat", "roti", "phulka"],
    "peanuts": ["peanut", "peanuts", "groundnut", "groundnuts"],
    "tree_nuts": ["almond", "almonds", "cashew", "cashews", "walnut", "nut", "nuts", "tree nut"],
    "mustard": ["mustard"],
    "coconut": ["coconut"],
    "soy": ["soy", "soya"],
    "sesame": ["sesame", "til"],
    "egg": ["egg", "eggs"],
    "shellfish": ["shellfish", "shrimp", "prawn"],
}


# ---------------------------------------------------------------------------
# Context & parsing helpers
# ---------------------------------------------------------------------------
def _context(dossier: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], list[Any]]:
    stage2 = dossier.get("stage2_counterfactual_prescription") or dossier.get("stage2_prescription")
    stage3 = dossier.get("stage3_discrete_meal_synthesis") or dossier.get("stage3_meal_plan")
    if not isinstance(stage2, dict) or not isinstance(stage3, dict):
        raise ValueError("Chat requires a completed Stage 2 and Stage 3 dossier.")
    if not stage3.get("accepted"):
        raise ValueError("There is no accepted meal plan. Resolve the Stage 3 failure before requesting plan changes.")
    basket = stage3.get("solved_food_basket")
    if not isinstance(basket, list) or not basket:
        raise ValueError("Chat requires a nonempty Stage 3 food basket.")
    catalog = ps.s3.load_food_database()
    by_id = {food.food_id: food for food in catalog}
    if any(
        item.get("food_id") not in by_id
        or item.get("catalog_version", "legacy") != by_id[item["food_id"]].catalog_version
        for item in basket
    ):
        raise ValueError("This meal plan uses an older food catalog. Generate a new meal plan before using chat.")
    return stage2, stage3, catalog


def _resolve_active_dossier(
    dossier: dict[str, Any],
    session_id: str | None = None,
    *,
    store_dir: Any = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """If session_id exists in plan_store, prefer its authoritative current version."""
    if not session_id:
        return dossier, None
    try:
        record = plan_store.get_session(session_id, store_dir=store_dir)
    except KeyError:
        record = plan_store.create_session(dossier, session_id=session_id, store_dir=store_dir)
    return plan_store.build_dossier_from_session(record), record


def _normalize_text(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.casefold()).split())


def _stem_token(tok: str) -> str:
    if len(tok) > 4 and tok.endswith("es"):
        return tok[:-2]
    if len(tok) > 3 and tok.endswith("s"):
        return tok[:-1]
    return tok


def _match_food(message: str, items: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Match a single food item by exact ID, name substring, or distinctive token overlap."""
    normalized = _normalize_text(message)
    msg_lower = message.casefold()
    direct = []
    for item in items:
        fid = str(item.get("food_id", "")).casefold()
        name_norm = _normalize_text(str(item.get("name", "")))
        short_name = _normalize_text(re.sub(r"\(.*?\)", " ", str(item.get("name", ""))))
        if (fid and fid in msg_lower) or (name_norm and name_norm in normalized) or (short_name and len(short_name) >= 4 and short_name in normalized):
            direct.append(item)
    if len(direct) == 1:
        return direct[0]
    if len(direct) > 1:
        # Prefer longest exact name match
        direct.sort(key=lambda x: len(str(x.get("name", ""))), reverse=True)
        if len(str(direct[0].get("name", ""))) > len(str(direct[1].get("name", ""))):
            return direct[0]
        return None

    # Strip boilerplate command/clause words and try fuzzy match
    clause = re.split(r"\b(?:but|and|while|except|keeping|keep)\b", normalized, maxsplit=1)[0]
    phrase = re.sub(
        r"\b(please|can|you|i|my|the|from|plan|meal|remove|replace|swap|exclude|without|"
        r"prefer|include|add|want|like|how|much|is|are|what|of|weight|portion|serving|"
        r"dont|don|t|no|dislike|hate|drop|take|out|a|an|for|me|with|in)\b",
        " ",
        clause,
    )
    phrase = " ".join(phrase.split())
    if not phrase or len(phrase) < 3:
        return None

    # Check stemmed token overlap against both dish name and food_id tokens
    stop_words = {
        "cooked", "fresh", "plain", "raw", "recipe", "skim", "low", "fat", "whole", "wheat",
        "steamed", "with", "and", "or", "for", "of", "the", "added", "vitamin", "fluid", "free",
        "sabzi", "sabji", "subji", "fry", "curry", "includes", "foods", "usda", "food", "program",
    }
    phrase_tokens = {_stem_token(t) for t in phrase.split() if len(t) >= 3 and t not in stop_words}
    token_hits = []
    for item in items:
        raw_name_toks = _normalize_text(str(item.get("name", ""))).split()
        raw_id_toks = _normalize_text(str(item.get("food_id", "")).replace("recipe_", "").replace("_", " ")).split()
        item_tokens = {_stem_token(t) for t in (raw_name_toks + raw_id_toks) if len(t) >= 3 and t not in stop_words}
        overlap = phrase_tokens.intersection(item_tokens)
        if overlap:
            token_hits.append((len(overlap), len(overlap) / max(len(phrase_tokens), 1), item))
    if len(token_hits) == 1:
        return token_hits[0][2]
    if len(token_hits) > 1:
        token_hits.sort(key=lambda x: (x[0], x[1]), reverse=True)
        if token_hits[0][0] > token_hits[1][0] or token_hits[0][1] > token_hits[1][1] + 0.15:
            return token_hits[0][2]

    scores = sorted(
        ((SequenceMatcher(None, phrase, _normalize_text(str(item.get("name", "")))).ratio(), item) for item in items),
        key=lambda pair: pair[0],
        reverse=True,
    )
    if scores and scores[0][0] >= 0.72 and (len(scores) == 1 or scores[0][0] - scores[1][0] >= 0.08):
        return scores[0][1]
    return None


def _nutrient(message: str) -> tuple[str, str, str] | None:
    for word, info in sorted(NUTRIENT_ALIASES.items(), key=lambda pair: -len(pair[0])):
        if re.search(r"\b" + re.escape(word) + r"\b", message, re.IGNORECASE):
            return info
    return None


def _recalculate(basket: list[dict[str, Any]], catalog: list[Any]) -> dict[str, float]:
    lookup = {food.food_id: food for food in catalog}
    totals = {code: 0.0 for code in ps.s3.NUTRIENT_CODES}
    for row in basket:
        food = lookup.get(str(row.get("food_id")))
        if food is None:
            raise ValueError("The plan contains a food missing from the current catalog.")
        quantity = row.get("quantity")
        if type(quantity) is not int or quantity < 0:
            raise ValueError("The plan contains an invalid food quantity.")
        for code in totals:
            totals[code] += quantity * float(food.nutrients_per_serving[code])
    return totals


def _fingerprint(stage2: dict[str, Any], stage3: dict[str, Any]) -> str:
    content = {"bounds": stage2, "basket": stage3["solved_food_basket"]}
    return hashlib.sha256(json.dumps(content, sort_keys=True, default=str).encode()).hexdigest()


def _extract_locked_foods(
    message: str,
    basket: list[dict[str, Any]],
) -> tuple[dict[str, int], list[str]]:
    """Extract foods or meal slots the user wants to keep unchanged (e.g., 'keep my lunch')."""
    locked: dict[str, int] = {}
    notes: list[str] = []
    lower = message.casefold()
    keep_match = re.search(r"\b(?:keep|lock|preserve|don'?t\s+change|do\s+not\s+change|retain)\b(.+)$", lower)
    if not keep_match:
        return locked, notes

    keep_clause = keep_match.group(1)
    schedule = ps.p14.build_deterministic_schedule(basket)
    basket_qty = {str(item["food_id"]): int(item["quantity"]) for item in basket}

    slot_keywords = {
        "breakfast": ["breakfast", "morning"],
        "lunch": ["lunch", "midday", "afternoon"],
        "dinner": ["dinner", "night", "supper"],
        "snack": ["snack", "snacks", "evening"],
    }
    for slot_key, kws in slot_keywords.items():
        if any(re.search(r"\b" + kw + r"\b", keep_clause) for kw in kws):
            for slot in schedule:
                if slot_key in str(slot.get("meal", "")).casefold():
                    for item in slot.get("items", []):
                        fid = str(item["food_id"])
                        if fid in basket_qty:
                            locked[fid] = basket_qty[fid]
            if locked:
                notes.append(f"locked {slot_key} items")

    # Also check if a specific food name is mentioned in the keep clause
    specific = _match_food(keep_clause, basket)
    if specific is not None:
        fid = str(specific["food_id"])
        locked[fid] = int(specific["quantity"])
        notes.append(f"locked {specific['name']}")

    return locked, notes


def _detect_allergen(message: str) -> str | None:
    lower = message.casefold()
    for canonical, words in ALLERGEN_KEYWORDS.items():
        for w in words:
            if re.search(r"\b" + re.escape(w) + r"\b", lower):
                return canonical
    m = re.search(r"(?:allergic to|allergy to|intoleran\w+ to)\s+([a-zA-Z ]{2,25})", lower)
    if m:
        return m.group(1).strip().split()[0]
    return None


# ---------------------------------------------------------------------------
# Permitted Backend Tools (Explicit Tool Layer)
# ---------------------------------------------------------------------------
def tool_get_plan(dossier: dict[str, Any], session_record: dict[str, Any] | None = None) -> dict[str, Any]:
    """Tool 1: Retrieve current accepted basket, 5-slot schedule, version, and allergen status."""
    _, stage3, catalog = _context(dossier)
    basket = stage3["solved_food_basket"]
    lookup = {f.food_id: f for f in catalog}
    schedule = ps.p14.build_deterministic_schedule(basket)
    schedule = ps._enrich_schedule_items(schedule, basket)
    total_weight_g = round(sum(int(item["quantity"]) * lookup[item["food_id"]].serving_weight_g for item in basket), 1)
    known_allergens = sorted({
        a
        for item in basket
        for a in (lookup[item["food_id"]].allergen_info or {}).get("known_allergens", [])
    })
    version = int(session_record["current_version"]) if session_record else int(stage3.get("plan_version", 1))
    flagged = (
        _saved_preferences(stage3, session_record)["reported_allergies"]
    )
    return {
        "plan_version": version,
        "basket": basket,
        "meal_schedule": schedule,
        "active_dish_count": len(basket),
        "total_weight_g": total_weight_g,
        "known_allergens_in_basket": known_allergens,
        "unknown_allergens_possible": True,
        "flagged_unsuitable_for_allergies": flagged,
        "clinical_validation_established": False,
    }


def tool_get_nutrient_totals(
    dossier: dict[str, Any],
    nutrient_code: str | None = None,
) -> dict[str, Any]:
    """Tool 2: Independently recalculate nutrient totals from basket x catalog with provenance."""
    stage2, stage3, catalog = _context(dossier)
    basket = stage3["solved_food_basket"]
    totals = _recalculate(basket, catalog)
    lookup = {f.food_id: f for f in catalog}
    env_by_code = {row["nutrient"]: row for row in stage2.get("nutrient_constraint_envelope", [])}
    deliv_by_code = {row["nutrient"]: row for row in stage3.get("delivered_nutrition", [])}

    codes = [nutrient_code] if nutrient_code else list(ps.s3.NUTRIENT_CODES)
    records: dict[str, Any] = {}
    for code in codes:
        if code not in totals:
            continue
        unknown_foods = [
            item["name"] for item in basket if code in lookup[item["food_id"]].unknown_nutrients
        ]
        env = env_by_code.get(code, {})
        deliv = deliv_by_code.get(code, {})
        rule = env.get("rule_provenance") or deliv.get("rule_provenance") or {}
        records[code] = {
            "nutrient": code,
            "delivered": round(totals[code], 4),
            "min_bound": env.get("min_bound", deliv.get("min_bound")),
            "max_bound": env.get("max_bound", deliv.get("max_bound")),
            "unit": rule.get("unit", ""),
            "planning_role": rule.get("planning_role", "unknown"),
            "hard_or_soft": rule.get("hard_or_soft", "hard"),
            "upper_kind": rule.get("upper_kind", ""),
            "source_url": rule.get("source_url", ""),
            "unknown_foods": unknown_foods,
            "complete": len(unknown_foods) == 0,
        }
    return {"totals": records, "catalog_version": catalog[0].catalog_version if catalog else "unknown"}


def tool_explain_constraint_conflict(
    failure: Mapping[str, Any] | None,
    stage2: dict[str, Any],
    catalog: list[Any],
    revision_spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Tool 4: Diagnose exact conflicting constraints/nutrients and test permissible alternatives."""
    fail = dict(failure or {})
    code = str(fail.get("code", "unknown"))
    explanation: dict[str, Any] = {
        "failure_code": code,
        "conflicting_constraints": [],
        "permissible_alternatives": [],
        "summary": str(fail.get("message", "No checked replacement plan was found.")),
    }

    if code == "conflicting_revision_constraints":
        ids = fail.get("details", {}).get("conflicting_food_ids", [])
        explanation["conflicting_constraints"] = [f"Simultaneous exclude + lock on {ids}"]
        explanation["summary"] = (
            f"Your request both removes and locks {', '.join(ids)}. "
            "Specify which dish to remove without locking that same dish."
        )
        return explanation

    if code == "locked_food_unavailable":
        ids = fail.get("details", {}).get("missing_locked_food_ids", [])
        explanation["conflicting_constraints"] = [f"Locked food unavailable: {ids}"]
        explanation["summary"] = (
            f"Locked dish(es) {', '.join(ids)} are not eligible under current exclusions/data requirements."
        )
        return explanation

    if code == "insufficient_nutrient_data":
        cats = fail.get("details", {}).get("missing_categories_with_complete_active_data", [])
        explanation["conflicting_constraints"] = [f"Missing required category: {c}" for c in cats]
        explanation["summary"] = (
            f"Removing that food leaves no complete-data options in required category: {', '.join(cats)}."
            if cats
            else "The remaining catalog has fewer eligible complete-data dishes than the minimum required."
        )
        return explanation

    if code != "infeasible_active_rules":
        return explanation

    # Run a lightweight relaxed MILP (allowing penalized slack on active nutrients) to identify
    # which exact active nutrient bounds cannot be simultaneously satisfied under revision_spec.
    try:
        stage = ps.s3._parse_stage2_input(stage2)
        required = {
            c for c in ps.s3.NUTRIENT_CODES
            if stage.rule_metadata.get(c, {}).get("planning_role") not in {"report_only", "exposure_reporting"}
        }
        excluded = set(revision_spec.get("excluded_food_ids", []))
        eligible = [f for f in catalog if f.food_id not in excluded and not required.intersection(f.unknown_nutrients)]
        if len(eligible) >= ps.s3.MIN_ACTIVE_DISHES:
            c_vec, integrality, bounds, constraints, info = ps.s3._make_sparse_constraints(eligible, stage, gamma=0.05)
            lb = bounds.lb.copy()
            ub = bounds.ub.copy()
            q0, z0 = info["indices"]["q"], info["indices"]["z"]
            locked = revision_spec.get("locked_food_quantities") or {}
            for i, food in enumerate(eligible):
                if food.food_id in locked:
                    lqty = int(locked[food.food_id])
                    lb[q0 + i] = ub[q0 + i] = float(lqty)
                    lb[z0 + i] = ub[z0 + i] = 1.0 if lqty > 0 else 0.0
            A = constraints.A
            c_lb, c_ub = constraints.lb, constraints.ub
            max_dishes = revision_spec.get("max_active_dishes")
            if max_dishes is not None:
                cap_row = np.zeros((1, len(c_vec)))
                cap_row[0, z0:z0 + len(eligible)] = 1.0
                A = vstack((A, coo_matrix(cap_row)), format="csr")
                c_lb = np.r_[c_lb, float(ps.s3.MIN_ACTIVE_DISHES)]
                c_ub = np.r_[c_ub, float(max_dishes)]
            res = milp(
                c=c_vec, integrality=integrality, bounds=Bounds(lb, ub),
                constraints=LinearConstraint(A, c_lb, c_ub),
                options={"time_limit": 1.0, "mip_rel_gap": 0.05, "presolve": True, "disp": False},
            )
            if res.x is not None:
                q_diag = np.rint(res.x[:len(eligible)]).astype(np.int64)
                diag_out = ps.s3._build_output_from_quantities(q_diag, eligible, stage, {})
                conflicts = []
                for row in diag_out.delivered_nutrition:
                    if not row["constraint_applied"]:
                        continue
                    d, lo, hi = float(row["delivered"]), float(row["min_bound"]), float(row["max_bound"])
                    unit = row.get("rule_provenance", {}).get("unit", "")
                    if d < lo - 1e-4:
                        conflicts.append(
                            f"{row['nutrient']} shortfall (closest achievable {d:.1f} {unit} vs required >= {lo:.1f} {unit})"
                        )
                    elif d > hi + 1e-4:
                        conflicts.append(
                            f"{row['nutrient']} excess (closest achievable {d:.1f} {unit} vs limit <= {hi:.1f} {unit})"
                        )
                # Also check post-check rules
                post = diag_out.clinical_verification
                for key in ("calories", "macronutrient_split", "sodium_to_potassium_ratio", "fiber_adequacy"):
                    if not post[key]["passed"]:
                        conflicts.append(f"post-check {key} failed on closest relaxed basket")
                explanation["conflicting_constraints"] = conflicts[:5]
    except Exception:
        pass

    # Test permissible alternatives:
    # 1) If locks were requested, check if dropping the locks makes the exclusion feasible
    if revision_spec.get("locked_food_quantities"):
        unlocked_spec = dict(revision_spec)
        unlocked_spec["locked_food_quantities"] = {}
        alt = ps.s3.solve_meal_plan(stage2, catalog, time_limit_sec=1.2, revision_spec=unlocked_spec)
        if alt.accepted:
            explanation["permissible_alternatives"].append(
                "Allowing other meal slots to adjust (removing the meal-slot lock) makes this exclusion feasible."
            )

    # 2) If max_active_dishes was requested, check max_active_dishes + 1
    if revision_spec.get("max_active_dishes") is not None:
        relaxed_cap = int(revision_spec["max_active_dishes"]) + 1
        if relaxed_cap <= ps.s3.MAX_ACTIVE_DISHES:
            cap_spec = dict(revision_spec)
            cap_spec["max_active_dishes"] = relaxed_cap
            alt = ps.s3.solve_meal_plan(stage2, catalog, time_limit_sec=1.2, revision_spec=cap_spec)
            if alt.accepted:
                explanation["permissible_alternatives"].append(
                    f"Using up to {relaxed_cap} active dishes instead of {revision_spec['max_active_dishes']} is feasible."
                )

    parts = ["No replacement basket satisfied all active nutrient and portion constraints simultaneously."]
    if explanation["conflicting_constraints"]:
        parts.append("Primary bottleneck(s): " + "; ".join(explanation["conflicting_constraints"]) + ".")
    if explanation["permissible_alternatives"]:
        parts.append("Permissible alternative(s): " + " ".join(explanation["permissible_alternatives"]))
    else:
        parts.append(
            "Protected Stage 2 nutrient constraints cannot be relaxed through chat; "
            "you can try removing a different dish or keeping the current plan."
        )
    explanation["summary"] = " ".join(parts)
    return explanation


def _saved_preferences(stage3: dict[str, Any], session_record: dict[str, Any] | None = None) -> dict[str, Any]:
    prefs = copy.deepcopy(stage3.get("preferences_applied", {}))
    prefs.update(copy.deepcopy(stage3.get("revision", {})))
    if session_record:
        prefs.update(copy.deepcopy(session_record.get("preferences", {})))
    prefs["reported_allergies"] = sorted(set(
        prefs.get("reported_allergies", []) + stage3.get("flagged_unsuitable_for_allergies", [])
    ))
    return prefs


def _allergy_exclusions(catalog: list[Any], allergies: list[str]) -> list[str]:
    return [f.food_id for f in catalog if set(allergies).intersection(
        (f.allergen_info or {}).get("known_allergens", [])
    )]


def tool_propose_revision(
    stage2: dict[str, Any],
    stage3: dict[str, Any],
    catalog: list[Any],
    revision_spec: dict[str, Any],
    *,
    reason: str,
    session_id: str | None = None,
    store_dir: Any = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    """Tool 3: Run Stage 3 controlled revision, validate independently, compute diff, store pending proposal.

    Returns (proposal_dict, diff_dict, conflict_explanation_dict).
    """
    prev_basket = stage3["solved_food_basket"]
    prev_quantities = {str(item["food_id"]): int(item["quantity"]) for item in prev_basket}

    record = plan_store.get_session(session_id, store_dir=store_dir) if session_id else None
    prefs = _saved_preferences(stage3, record)
    existing_excl = set(prefs.get("excluded_food_ids", []))
    existing_excl.update(_allergy_exclusions(catalog, prefs["reported_allergies"]))
    new_excl = set(revision_spec.get("excluded_food_ids", []))
    all_excl = sorted(existing_excl | new_excl)
    default_mode = "veg_nonveg_combo" if any(f.category == "non_veg_protein" for f in catalog) else "pure_vegetarian"
    dietary_mode = str(
        revision_spec.get("dietary_mode")
        or stage3.get("revision", {}).get("dietary_mode")
        or stage3.get("preferences_applied", {}).get("dietary_mode")
        or default_mode
    )

    merged_spec: dict[str, Any] = {
        "dietary_mode": dietary_mode,
        "excluded_food_ids": all_excl,
        "excluded_ingredient_codes": sorted(set(prefs.get("excluded_ingredient_codes", [])) | set(revision_spec.get("excluded_ingredient_codes", []))),
        "preferred_food_ids": sorted(set(prefs.get("preferred_food_ids", [])) | set(revision_spec.get("preferred_food_ids", []))),
        "locked_food_quantities": {**prefs.get("locked_food_quantities", {}), **(revision_spec.get("locked_food_quantities") or {})},
        "max_active_dishes": revision_spec.get("max_active_dishes") if revision_spec.get("max_active_dishes") is not None else prefs.get("max_active_dishes"),
        "previous_quantities": prev_quantities,
        "change_penalty": float(revision_spec.get("change_penalty", 0.35)),
    }

    output = ps.s3.solve_meal_plan(stage2, catalog, time_limit_sec=2.0, revision_spec=merged_spec)
    if not output.accepted:
        conflict = tool_explain_constraint_conflict(output.failure, stage2, catalog, merged_spec)
        return None, None, conflict

    validation = ps.s3.validate_candidate(
        output,
        catalog,
        stage2,
        excluded_ids=all_excl,
        locked_quantities=merged_spec["locked_food_quantities"] or None,
        max_active_dishes=merged_spec["max_active_dishes"],
        dietary_mode=dietary_mode,
    )
    if not validation["accepted"]:
        return None, None, {
            "failure_code": "validation_rejected",
            "conflicting_constraints": validation.get("structural_errors", []) + validation.get("failed_active_nutrients", []),
            "permissible_alternatives": [],
            "summary": "The independent plan validator rejected the solver candidate. Your current plan has not changed.",
        }

    diff = ps.s3.compute_plan_diff(prev_basket, output.solved_food_basket)
    proposal_id = f"prop_{uuid.uuid4().hex[:10]}"
    proposal = copy.deepcopy(stage3)
    proposal.update(output.to_dict())
    proposal["revision"] = {
        "proposal_id": proposal_id,
        "status": "proposal",
        "reason": reason,
        "dietary_mode": dietary_mode,
        "excluded_food_id": sorted(new_excl)[0] if new_excl else "",
        "excluded_food_ids": all_excl,
        "locked_food_quantities": merged_spec["locked_food_quantities"],
        "max_active_dishes": merged_spec["max_active_dishes"],
        "preferred_food_ids": merged_spec["preferred_food_ids"],
        "excluded_ingredient_codes": merged_spec["excluded_ingredient_codes"],
        "base_fingerprint": _fingerprint(stage2, stage3),
        "diff": diff,
    }

    if session_id:
        plan_store.store_pending_proposal(
            session_id, proposal_id, proposal, diff, reason, store_dir=store_dir
        )

    return proposal, diff, None


# ---------------------------------------------------------------------------
# Optional guarded local Qwen prose rephrasing (zero calculation authority)
# ---------------------------------------------------------------------------
def _optional_llm_rephrase(deterministic_answer: str, question: str) -> str:
    """Return verified prose until semantic preservation can be enforced.

    Matching number sets cannot validate food identity, negation, or medical
    claims. Do not expose unconstrained LLM rewrites through this compatibility
    option; the separate Stage 4 report compiler is unaffected.
    """
    return deterministic_answer


# ---------------------------------------------------------------------------
# Public API: accept, rollback, answer
# ---------------------------------------------------------------------------
def accept(
    dossier: dict[str, Any],
    proposal: dict[str, Any],
    *,
    session_id: str | None = None,
    store_dir: Any = None,
) -> dict[str, Any]:
    """Recalculate and verify an untrusted proposal before committing a new plan version."""
    active_dossier, session_record = _resolve_active_dossier(dossier, session_id, store_dir=store_dir)
    stage2, stage3, catalog = _context(active_dossier)
    revision = proposal.get("revision")
    if not isinstance(revision, dict) or revision.get("status") != "proposal":
        raise ValueError("A checked proposal is required.")
    if session_record is not None:
        pending = session_record.get("pending_proposals", {}).get(revision.get("proposal_id"))
        if not pending or pending.get("base_version") != session_record["current_version"]:
            raise ValueError("This proposal is no longer pending. Request a new revision.")
        if pending.get("proposal") != proposal:
            raise ValueError("The proposal differs from the server-verified proposal.")
    if revision.get("base_fingerprint") != _fingerprint(stage2, stage3):
        raise ValueError("This proposal belongs to an older plan. Request a new revision.")

    prefs = _saved_preferences(stage3, session_record)
    existing_excl = set(prefs.get("excluded_food_ids", []))
    existing_excl.update(_allergy_exclusions(catalog, prefs["reported_allergies"]))
    excluded_id = str(revision.get("excluded_food_id", "") or "")
    if excluded_id:
        if excluded_id not in {row["food_id"] for row in stage3["solved_food_basket"]}:
            raise ValueError("The excluded food is not in the current plan.")
        expected_excl = sorted(existing_excl | {excluded_id})
        if revision.get("excluded_food_ids") != expected_excl:
            raise ValueError("The proposal has inconsistent food exclusions.")
    else:
        expected_excl = sorted(set(revision.get("excluded_food_ids", [])) | existing_excl)

    locked_quantities = {str(k): int(v) for k, v in (revision.get("locked_food_quantities") or {}).items()}
    max_active_dishes = revision.get("max_active_dishes")

    requested = proposal.get("solved_food_basket")
    if not isinstance(requested, list) or not requested:
        raise ValueError("The proposal has no food basket.")

    by_id = {food.food_id: i for i, food in enumerate(catalog)}
    q = np.zeros(len(catalog), dtype=np.int64)
    for row in requested:
        fid, quantity = row.get("food_id"), row.get("quantity")
        if fid not in by_id or type(quantity) is not int or not 1 <= quantity <= 3 or q[by_id[fid]]:
            raise ValueError("The proposal contains an invalid or duplicate food serving.")
        q[by_id[fid]] = quantity

    stage = ps.s3._parse_stage2_input(stage2)
    output = ps.s3._build_output_from_quantities(
        q, catalog, stage, {"status": "accepted_checked_revision", "optimality_proven": False}
    )
    default_mode = "veg_nonveg_combo" if any(f.category == "non_veg_protein" for f in catalog) else "pure_vegetarian"
    dietary_mode = str(
        revision.get("dietary_mode")
        or stage3.get("revision", {}).get("dietary_mode")
        or stage3.get("preferences_applied", {}).get("dietary_mode")
        or default_mode
    )
    validation = ps.s3.validate_candidate(
        output,
        catalog,
        stage2,
        excluded_ids=expected_excl,
        locked_quantities=locked_quantities or None,
        max_active_dishes=max_active_dishes,
        dietary_mode=dietary_mode,
    )
    if not validation["accepted"]:
        raise ValueError("The proposed basket does not pass the current checks.")

    diff = ps.s3.compute_plan_diff(stage3["solved_food_basket"], output.solved_food_basket)
    output.accepted = True
    output.clinical_verification["all_active_checks_passed"] = True
    output.clinical_verification["acceptance_validation"] = validation

    updated = copy.deepcopy(active_dossier)
    stage3_new = output.to_dict()
    # Numeric acceptance does not establish allergen safety for incomplete records.
    stage3_new["flagged_unsuitable_for_allergies"] = list(prefs["reported_allergies"])
    if stage3.get("preferences_applied"):
        stage3_new["preferences_applied"] = copy.deepcopy(stage3["preferences_applied"])
        stage3_new["preferences_applied"]["excluded_food_ids"] = expected_excl
    stage3_new["revision"] = {
        "dietary_mode": dietary_mode,
        "excluded_food_id": excluded_id,
        "excluded_food_ids": expected_excl,
        "locked_food_quantities": locked_quantities,
        "max_active_dishes": max_active_dishes,
        "preferred_food_ids": list(revision.get("preferred_food_ids", [])),
        "excluded_ingredient_codes": list(revision.get("excluded_ingredient_codes", [])),
        "reason": revision.get("reason", "Accepted user revision"),
        "diff": diff,
        "status": "accepted",
    }

    plan_version = int(stage3.get("plan_version", 1)) + 1
    if session_id and session_record is not None:
        updated_record = plan_store.commit_accepted_version(
            session_id,
            stage3_new,
            diff,
            str(revision.get("reason", "Accepted user revision")),
            updated_preferences={
                "excluded_food_ids": expected_excl,
                "locked_food_quantities": locked_quantities,
                "max_active_dishes": max_active_dishes,
                "preferred_food_ids": list(revision.get("preferred_food_ids", [])),
                "excluded_ingredient_codes": list(revision.get("excluded_ingredient_codes", [])),
            },
            store_dir=store_dir,
        )
        plan_version = int(updated_record["current_version"])
        stage3_new["plan_version"] = plan_version
        updated = plan_store.build_dossier_from_session(updated_record)
    else:
        stage3_new["plan_version"] = plan_version
        updated["stage3_discrete_meal_synthesis"] = stage3_new
        updated["stage3_meal_plan"] = stage3_new

    schedule = ps.p14.build_deterministic_schedule(output.solved_food_basket)
    schedule = ps._enrich_schedule_items(schedule, output.solved_food_basket)
    return {
        "session_id": session_id,
        "plan_version": plan_version,
        "revision_diff": diff,
        "dossier": updated,
        "meal_plan": {
            "accepted": True,
            "failure": None,
            "clinical_verification": output.clinical_verification,
            "solved_food_basket": output.solved_food_basket,
            "meal_schedule": schedule,
            "delivered_nutrition": output.delivered_nutrition,
            "solver_metadata": output.solver_metadata,
            "latency_ms": 0.0,
            "dossier": updated,
        },
    }


def rollback(
    session_id: str,
    target_version: int,
    *,
    store_dir: Any = None,
) -> dict[str, Any]:
    """Roll back a persistent plan session to a previously accepted version."""
    record, target = plan_store.rollback_version(session_id, target_version, store_dir=store_dir)
    updated_dossier = plan_store.build_dossier_from_session(record)
    stage3 = updated_dossier["stage3_discrete_meal_synthesis"]
    schedule = ps.p14.build_deterministic_schedule(stage3["solved_food_basket"])
    schedule = ps._enrich_schedule_items(schedule, stage3["solved_food_basket"])
    return {
        "session_id": session_id,
        "plan_version": int(record["current_version"]),
        "restored_reason": target.get("reason", ""),
        "dossier": updated_dossier,
        "meal_plan": {
            "accepted": True,
            "failure": None,
            "clinical_verification": stage3.get("clinical_verification", {}),
            "solved_food_basket": stage3["solved_food_basket"],
            "meal_schedule": schedule,
            "delivered_nutrition": stage3.get("delivered_nutrition", []),
            "solver_metadata": stage3.get("solver_metadata", {}),
            "latency_ms": 0.0,
            "dossier": updated_dossier,
        },
    }


def _format_diff_summary(diff: Mapping[str, Any]) -> str:
    parts = []
    if diff.get("removed"):
        parts.append("Removed: " + ", ".join(f"{x['quantity']} × {x['name']}" for x in diff["removed"]))
    if diff.get("added"):
        parts.append("Added: " + ", ".join(f"{x['quantity']} × {x['name']}" for x in diff["added"]))
    if diff.get("quantity_changed"):
        parts.append(
            "Adjusted: "
            + ", ".join(f"{x['name']} ({x['previous_quantity']}→{x['new_quantity']})" for x in diff["quantity_changed"])
        )
    if diff.get("unchanged"):
        parts.append(f"Unchanged: {len(diff['unchanged'])} dish(es)")
    return "; ".join(parts) + f" (total serving change: {diff.get('total_serving_delta', 0)})."


def answer(
    message: str,
    dossier: dict[str, Any],
    *,
    session_id: str | None = None,
    use_llm_explanation: bool = False,
    store_dir: Any = None,
) -> dict[str, Any]:
    """Answer grounded questions or produce a checked controlled-revision proposal."""
    active_dossier, session_record = _resolve_active_dossier(dossier, session_id, store_dir=store_dir)
    stage2, stage3, catalog = _context(active_dossier)
    basket = stage3["solved_food_basket"]
    question = message.strip()
    normalized = question.casefold()
    plan_version = int(session_record["current_version"]) if session_record else int(stage3.get("plan_version", 1))
    flagged_allergies = _saved_preferences(stage3, session_record)["reported_allergies"]

    def _finish(
        intent: str,
        text: str,
        sources: list[str],
        tools: list[str],
        *,
        proposed: dict[str, Any] | None = None,
        diff: dict[str, Any] | None = None,
        flagged_unsuitable: bool = False,
    ) -> dict[str, Any]:
        final_text = _optional_llm_rephrase(text, question) if use_llm_explanation else text
        resp: dict[str, Any] = {
            "intent": intent,
            "answer": final_text,
            "sources": sources,
            "tool_calls_used": tools,
            "session_id": session_id,
            "plan_version": plan_version,
            "plan_flagged_unsuitable": bool(flagged_unsuitable or flagged_allergies),
        }
        if proposed is not None:
            resp["proposed_meal_plan"] = proposed
        if diff is not None:
            resp["revision_diff"] = diff
        return resp

    # 1. Allergy reporting — immediately flag existing plan unsuitable; never certify allergen-free
    if re.search(r"\b(allerg\w*|intoleran\w*|anaphylax\w*)\b", normalized):
        allergy_decl_re = re.compile(
            r"\b(?:i\s+am|i'm)\s+(?:severely\s+)?(?:allergic|intolerant)\s+to\b"
            r"|\bi\s+have\s+(?!(?:no|not|never)\b)(?:(?:a|an)\s+)?(?:\w+\s+){0,3}(?:allergy|allergies|intolerance)\b"
            r"|\bmy\s+(?:\w+\s+){0,2}(?:allergy|allergies|intolerance)\s+(?:is|are)\b"
        )
        # Only declarative (non-question) sentences count as allergy declarations so
        # pure questions like "I am allergic to dairy?" do not mutate state, while
        # multi-sentence statements like "I am allergic to almonds. Is my plan safe?"
        # still flag the allergy immediately.
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", question.strip()) if s.strip()]
        declaration = any(
            not sent.endswith("?") and bool(allergy_decl_re.search(sent.lower()))
            for sent in sentences
        )
        if not declaration:
            return _finish(
                "allergy",
                "Allergen records are incomplete, so I cannot guarantee that this plan is allergen-free. "
                "I have not changed your recorded allergies. To report one, say 'I am allergic to [allergen]'. "
                "A clinician or registered dietitian should review a plan when you have an allergy.",
                ["food_catalog:allergen_info"], ["tool_get_plan"],
            )
        allergen = _detect_allergen(question) or "unspecified_allergen"
        if not session_id:
            session_record = plan_store.create_session(active_dossier, store_dir=store_dir)
            session_id = session_record["session_id"]
            plan_version = int(session_record["current_version"])
        plan_store.flag_allergy(session_id, allergen, store_dir=store_dir)
        plan_info = tool_get_plan(active_dossier, session_record)
        lookup = {f.food_id: f for f in catalog}
        matched_items = [
            item["name"]
            for item in basket
            if allergen in (lookup[item["food_id"]].allergen_info or {}).get("known_allergens", [])
        ]
        msg_parts = [f"I have recorded a reported {allergen} allergy/intolerance and flagged your current plan (v{plan_info['plan_version']}) as unsuitable pending review."]
        if matched_items:
            msg_parts.append(f"Dishes in the current basket with documented {allergen} content: {', '.join(matched_items)}.")
        msg_parts.append(
            "Important safety limit: dishes without a listed allergen in this catalog have incomplete allergen records "
            "(unknown allergen status is NOT allergen-free). Please consult a clinician or registered dietitian before eating or substituting."
        )
        return _finish(
            "allergy",
            " ".join(msg_parts),
            ["food_catalog:allergen_info", "plan_store:flag_allergy"],
            ["tool_get_plan"],
            flagged_unsuitable=True,
        )

    # 2. Out-of-scope medical conditions & protected constraint relaxation attempts
    # Guard against food names like 'Kidney bean curry' (Rajma) triggering kidney-disease refusal.
    if re.search(r"\b(medicine|medication|prescrib\w*|diagnos\w*|pregnan\w*|lactat\w*|renal|dialysis|child|pediatric)\b|\bkidney\b(?!\s*beans?\b)", normalized):
        return _finish(
            "unsupported",
            "This assistant cannot evaluate medications, diagnose conditions, or prescribe diets for pregnancy, lactation, renal disease, or pediatric populations. Please consult a clinician or dietitian.",
            ["stage2.rule_provenance:applicability"],
            [],
        )

    if re.search(r"\b(ignore|relax|disable|bypass|override|skip)\b.*\b(limit|bound|rule|constraint|sodium|b12|protein|calorie|fat|potassium)\b", normalized):
        return _finish(
            "unsupported",
            "Protected Stage 2 nutrient constraints (relaxation_policy: protected_no_chat_relaxation) cannot be weakened or bypassed through conversation. Your current plan has not changed.",
            ["stage2.rule_provenance:relaxation_policy"],
            [],
        )

    # 3. Version history & rollback
    if re.search(r"\b(rollback|roll back|revert|undo)\b", normalized):
        m_ver = re.search(r"\b(?:v|version\s*)(\d+)\b", normalized)
        if not session_id or session_record is None:
            return _finish(
                "rollback",
                "Rollback requires a persistent session_id. Create a session via /api/sessions or pass session_id in your chat request.",
                [],
                [],
            )
        target_v = int(m_ver.group(1)) if m_ver else max(1, int(session_record["current_version"]) - 1)
        if target_v == int(session_record["current_version"]):
            return _finish(
                "rollback",
                f"Your session is already on version {target_v}.",
                ["plan_store:versions"],
                ["tool_get_plan"],
            )
        rb = rollback(session_id, target_v, store_dir=store_dir)
        plan_version = int(rb["plan_version"])
        restored_foods = ", ".join(f"{x['quantity']} × {x['name']}" for x in rb["meal_plan"]["solved_food_basket"])
        return _finish(
            "rollback",
            f"Rolled back session {session_id} to version {plan_version} ({rb['restored_reason']}). Active basket: {restored_foods}.",
            ["plan_store:rollback_version"],
            ["tool_get_plan"],
        )

    if re.search(r"\b(history|versions?|what changed|audit)\b", normalized):
        if session_record is not None:
            lines = [
                f"v{v['version']}: {v['reason']}"
                + (f" ({_format_diff_summary(v['diff_from_previous'])})" if v.get("diff_from_previous") else "")
                for v in session_record["versions"]
            ]
            return _finish(
                "history",
                f"Session {session_id} is currently on v{session_record['current_version']}. Version history: " + " | ".join(lines),
                ["plan_store:versions"],
                ["tool_get_plan"],
            )
        rev = stage3.get("revision") or {}
        if rev:
            return _finish(
                "history",
                f"Current plan is v{plan_version}. Last revision: {rev.get('reason', 'exclusion update')} (excluded: {rev.get('excluded_food_ids', [])}).",
                ["stage3.revision"],
                ["tool_get_plan"],
            )
        return _finish(
            "history",
            f"Current plan is v{plan_version} (initial accepted Stage 3 meal plan, no accepted revisions yet).",
            ["stage3.solved_food_basket"],
            ["tool_get_plan"],
        )

    # 4. Stage 1 screening explanation (never promises future risk reduction from meal swaps)
    if re.search(r"\b(risk|probability|chance|screening)\b", normalized):
        conditions = {"diabetes": "diabetes", "obesity": "obesity", "cvd": "cvd", "heart": "cvd", "cardiovascular": "cvd"}
        selected = [code for word, code in conditions.items() if re.search(r"\b" + word + r"\b", normalized)]
        s1_info = active_dossier.get("stage1_diagnostic_assessment", {})
        probabilities = s1_info.get("probabilities", {})
        meta = s1_info.get("model_metadata", {})
        model_ver = meta.get("model_version", "BioNutriGPT-Stage1")
        if selected and selected[0] in probabilities:
            code = selected[0]
            value = float(probabilities[code]) * 100.0
            return _finish(
                "risk",
                f"The Stage 1 model ({model_ver}) estimates {value:.1f}% for prevalent {code} from your recorded observed inputs. "
                "This is a cross-sectional screening estimate in the NHANES study setting — not a medical diagnosis, not your future probability of developing the condition, "
                "and meal plan substitutions do not automatically alter your recorded intake or guarantee reduced disease risk.",
                [f"stage1.probabilities:{code}", "stage1.model_metadata"],
                [],
            )
        return _finish(
            "risk",
            "Ask about diabetes, obesity, or CVD screening to view the recorded Stage 1 screening estimate and its scope limitations.",
            ["stage1.probabilities"],
            [],
        )

    # 5. Controlled Revisions: dish cap ("fewer dishes" / "max N dishes")
    dish_cap_match = re.search(r"\b(?:max(?:imum)?|at most|no more than|limit to)\s+(\d+)\s+dish(?:es)?\b", normalized)
    fewer_dishes = bool(re.search(r"\b(fewer|less|reduce|simplify)\s+(?:the\s+)?(?:number\s+of\s+)?dish(?:es)?\b", normalized))
    if dish_cap_match or fewer_dishes:
        target_cap = int(dish_cap_match.group(1)) if dish_cap_match else max(ps.s3.MIN_ACTIVE_DISHES, len(basket) - 1)
        locked, lock_notes = _extract_locked_foods(question, basket)
        proposal, diff, conflict = tool_propose_revision(
            stage2, stage3, catalog,
            {"max_active_dishes": target_cap, "locked_food_quantities": locked},
            reason=f"Limit plan to at most {target_cap} dishes" + (f" ({', '.join(lock_notes)})" if lock_notes else ""),
            session_id=session_id,
            store_dir=store_dir,
        )
        if proposal is None:
            return _finish(
                "revision",
                f"I tried limiting the plan to at most {target_cap} dishes, but no checked basket passed all rules. "
                f"{(conflict or {}).get('summary', '')} Reason: {(conflict or {}).get('failure_code', 'unknown')}. Your current plan has not changed.",
                ["stage2.nutrient_constraint_envelope", "stage3.failure"],
                ["tool_propose_revision", "tool_explain_constraint_conflict"],
            )
        return _finish(
            "revision",
            f"I found a checked proposal using {len(proposal['solved_food_basket'])} dishes (<= {target_cap}). "
            f"{_format_diff_summary(diff or {})} Review the full revised plan before accepting it.",
            ["stage2.nutrient_constraint_envelope", "stage3.clinical_verification"],
            ["tool_propose_revision"],
            proposed=proposal,
            diff=diff,
        )

    # 6. Controlled Revisions: remove/replace/swap/exclude (with optional meal-slot locking!)
    if re.search(r"\b(remove|replace|swap|exclude|without|dislike|drop)\b", normalized):
        remove_clause = re.split(r"\b(?:but|while|except|keeping|keep|lock|preserve|don'?t\s+change|do\s+not\s+change)\b", question, maxsplit=1, flags=re.IGNORECASE)[0]
        food = _match_food(remove_clause, basket) or _match_food(question, basket)
        if food is None:
            names = ", ".join(str(item["name"]) for item in basket)
            return _finish(
                "unsupported",
                f"I couldn't identify one food to remove from your current basket. Its foods are: {names}. Please use the name shown in the plan.",
                ["stage3.solved_food_basket"],
                ["tool_get_plan"],
            )
        locked, lock_notes = _extract_locked_foods(question, basket)
        fid = str(food["food_id"])
        reason = f"Exclude {food['name']}" + (f" while keeping {', '.join(lock_notes)}" if lock_notes else "")
        proposal, diff, conflict = tool_propose_revision(
            stage2, stage3, catalog,
            {"excluded_food_ids": [fid], "locked_food_quantities": locked},
            reason=reason,
            session_id=session_id,
            store_dir=store_dir,
        )
        if proposal is None:
            return _finish(
                "revision",
                f"I recognized {food['name']}" + (f" ({', '.join(lock_notes)})" if lock_notes else "") + ". "
                f"{(conflict or {}).get('summary', 'No checked replacement plan was found.')} "
                f"Reason: {(conflict or {}).get('failure_code', 'unknown')}. Your current plan has not changed.",
                ["stage2.nutrient_constraint_envelope", "stage3.failure"],
                ["tool_propose_revision", "tool_explain_constraint_conflict"],
            )
        lock_msg = f" while preserving {', '.join(lock_notes)}" if lock_notes else ""
        return _finish(
            "revision",
            f"I found a checked proposal without {food['name']}{lock_msg}. "
            f"{_format_diff_summary(diff or {})} Review the whole revised plan before accepting it.",
            ["stage2.nutrient_constraint_envelope", "stage3.clinical_verification"],
            ["tool_propose_revision"],
            proposed=proposal,
            diff=diff,
        )

    # 7. Controlled Revisions: prefer/include a catalog food
    if re.search(r"\b(prefer|include|add)\b", normalized):
        catalog_dicts = [{"food_id": f.food_id, "name": f.name} for f in catalog]
        wanted = _match_food(question, catalog_dicts)
        if wanted is not None:
            locked, lock_notes = _extract_locked_foods(question, basket)
            # If user explicitly says "include" or "add" a food not yet in basket, lock at least 1 serving
            req_locks = dict(locked)
            if re.search(r"\b(include|add)\b", normalized) and wanted["food_id"] not in {x["food_id"] for x in basket}:
                req_locks[str(wanted["food_id"])] = 1
            proposal, diff, conflict = tool_propose_revision(
                stage2, stage3, catalog,
                {"preferred_food_ids": [str(wanted["food_id"])], "locked_food_quantities": req_locks},
                reason=f"Prefer/include {wanted['name']}",
                session_id=session_id,
                store_dir=store_dir,
            )
            if proposal is None:
                return _finish(
                    "revision",
                    f"I checked including {wanted['name']}, but no checked basket passed all rules. "
                    f"{(conflict or {}).get('summary', '')} Reason: {(conflict or {}).get('failure_code', 'unknown')}. Your current plan has not changed.",
                    ["stage2.nutrient_constraint_envelope", "stage3.failure"],
                    ["tool_propose_revision", "tool_explain_constraint_conflict"],
                )
            return _finish(
                "revision",
                f"I generated a checked proposal favoring {wanted['name']}. {_format_diff_summary(diff or {})} Review before accepting.",
                ["stage2.nutrient_constraint_envelope", "stage3.clinical_verification"],
                ["tool_propose_revision"],
                proposed=proposal,
                diff=diff,
            )

    # 8. Nutrient questions (checked BEFORE generic 'how much' portion match so 'how much vitamin B12' routes here!)
    nutrient = _nutrient(question)
    if nutrient is not None:
        code, label, unit = nutrient
        res = tool_get_nutrient_totals(active_dossier, code)
        rec = res["totals"][code]
        if rec["unknown_foods"]:
            answer_text = (
                f"The known subtotal is {rec['delivered']:.1f} {unit} of {label}. "
                f"Values are unknown for: {', '.join(rec['unknown_foods'])}. I cannot verify the full total or adequacy."
            )
        else:
            answer_text = f"The selected foods provide {rec['delivered']:.1f} {unit} of {label} according to the current food catalog ({res['catalog_version']})."

        role = rec["planning_role"]
        if role == "report_only":
            answer_text += " This field is reported without an independent adequacy target."
        elif role == "exposure_reporting":
            answer_text += " This field is tracked separately as beverage/exposure reporting; drinks outside the basket are not included."
        elif rec["upper_kind"] == "numerical_search_cap_not_UL" and rec["min_bound"] is not None:
            answer_text += f" The recorded planning reference starts at {float(rec['min_bound']):.1f} {unit}. Its numerical search cap is not an intake recommendation or tolerable upper limit."
        elif rec["min_bound"] is not None and rec["max_bound"] is not None:
            answer_text += f" The recorded planning interval is {float(rec['min_bound']):.1f}–{float(rec['max_bound']):.1f} {unit}."

        return _finish(
            "nutrient",
            answer_text,
            ["stage3.solved_food_basket", f"food_catalog:nutrients_per_serving:{code}", "stage2.nutrient_constraint_envelope"],
            ["tool_get_nutrient_totals"],
        )

    # 9. Portion / total weight questions
    if re.search(r"\b(weight|weigh|grams?|how heavy)\b", normalized) and re.search(r"\b(total|all|whole|entire|day|daily|meals|plan)\b", normalized):
        plan_info = tool_get_plan(active_dossier, session_record)
        grams = plan_info["total_weight_g"]
        return _finish(
            "portion",
            f"All selected daily portions weigh about {grams:g} g in total ({grams / 1000:.2f} kg), using the catalog's serving weights. This includes listed foods only, not water or external drinks.",
            ["stage3.solved_food_basket", "food_catalog:serving_weight_g"],
            ["tool_get_plan"],
        )

    if re.search(r"\b(portion|serving|grams?|weight|how much)\b", normalized):
        item = _match_food(question, basket)
        if item is not None:
            catalog_item = next(food for food in catalog if food.food_id == item["food_id"])
            quantity = int(item["quantity"])
            return _finish(
                "portion",
                f"Your plan (v{plan_version}) has {quantity} × {catalog_item.serving_unit} of {catalog_item.name}. "
                f"The catalog records {catalog_item.serving_weight_g:g} g per serving ({quantity * catalog_item.serving_weight_g:g} g total; source: {catalog_item.data_source}).",
                [f"food_catalog:{catalog_item.food_id}", "stage3.solved_food_basket"],
                ["tool_get_plan"],
            )

    # 10. Current plan & meal schedule explanation
    if re.search(r"\b(plan|meal|food|eat|dinner|breakfast|lunch|schedule|basket)\b", normalized):
        plan_info = tool_get_plan(active_dossier, session_record)
        foods = ", ".join(f"{item['quantity']} × {item['name']}" for item in plan_info["basket"])
        flag_note = (
            f" Warning: flagged unsuitable due to reported allergies ({', '.join(plan_info['flagged_unsuitable_for_allergies'])})."
            if plan_info["flagged_unsuitable_for_allergies"]
            else ""
        )
        return _finish(
            "plan",
            f"Your current daily basket (v{plan_info['plan_version']}, {plan_info['active_dish_count']} dishes, ~{plan_info['total_weight_g']:g} g) contains: {foods}.{flag_note} "
            "These are catalog-based estimates; clinical validation has not been established.",
            ["stage3.solved_food_basket", "stage3.meal_schedule"],
            ["tool_get_plan"],
        )

    food = _match_food(question, basket)
    if food is not None:
        return _finish(
            "portion",
            f"{food['name']} is in your current plan (v{plan_version}): {food['quantity']} × {food['serving_unit']}. Ask me to remove or replace it if you want me to check an alternative.",
            ["stage3.solved_food_basket"],
            ["tool_get_plan"],
        )

    return _finish(
        "unsupported",
        "I can explain your current foods, portions, nutrient totals, Stage 1 screening limitations, or version history, "
        "and I can check controlled revisions (remove/replace a food, keep a meal slot like lunch, use fewer dishes, or roll back to an earlier version).",
        [],
        [],
    )


# ---------------------------------------------------------------------------
# Free-Text Conversational Meal-Plan Preference Intake (LLM + Deterministic Verification)
# ---------------------------------------------------------------------------
def _extract_intake_with_qwen(message: str, state: dict[str, Any], catalog_names: list[str]) -> dict[str, Any] | None:
    """Use local Qwen 2.5 3B Instruct to extract structured preference JSON from typed free text."""
    try:
        import infer_bionutrigpt_pipeline_Stage_1_to_4 as p14
        import torch

        if not torch.cuda.is_available():
            return None
        prompt = (
            "Extract the user's dietary preferences from their typed message into STRICT JSON ONLY.\n"
            "Do NOT explain, do NOT add markdown fences.\n"
            "Schema:\n"
            "{\n"
            '  "dietary_mode": "pure_vegetarian" | "veg_nonveg_combo" | "pure_non_vegetarian" | null,\n'
            '  "allergies": [string],\n'
            '  "no_allergies_stated": boolean,\n'
            '  "disliked_foods": [string],\n'
            '  "no_dislikes_stated": boolean\n'
            "}\n"
            "Rules:\n"
            "- 'pure_vegetarian' if user wants pure veg / vegetarian only / no meat / veg.\n"
            "- 'veg_nonveg_combo' if user wants both veg and non-veg / combo / mix / flexible.\n"
            "- 'pure_non_vegetarian' if user wants purely non-veg / pure non-vegetarian / non-veg only / mostly meat & fish.\n"
            "- Put any food allergies or intolerances into 'allergies' (e.g. 'dairy', 'almonds', 'peanuts', 'egg', 'fish', 'gluten').\n"
            "- Put any dishes or ingredients the user dislikes or wants removed/excluded into 'disliked_foods'.\n"
            f"Current known state: {json.dumps(state)}\n"
            f"Sample catalog dishes: {', '.join(catalog_names[:30])}\n"
            f"User typed: {message}\n"
            "JSON:"
        )
        cfg = p14.CompilerConfig(backend="transformers", max_new_tokens=180)
        res = p14._run_transformers_backend(prompt, cfg)
        m = re.search(r"\{.*\}", res.text, re.DOTALL)
        if not m:
            return None
        parsed = json.loads(m.group(0))
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _extract_intake_deterministic(message: str, state: dict[str, Any], catalog: list[Any]) -> dict[str, Any]:
    """Deterministic NLU extraction & verification of free-text dietary mode, allergies, and dislikes."""
    lower = message.casefold().strip()
    out: dict[str, Any] = {
        "dietary_mode": None,
        "allergies": [],
        "no_allergies_stated": False,
        "disliked_foods": [],
        "no_dislikes_stated": False,
    }

    # 1. Dietary mode detection (ordered from most specific to general)
    if re.search(r"\b(combo|both|mix|mixed|veg\s*(?:and|&|\+|/)\s*non[\s-]*veg|vegetarian\s*(?:and|&|\+|/)\s*non[\s-]*vegetarian)\b", lower):
        out["dietary_mode"] = "veg_nonveg_combo"
    elif re.search(r"\b(pure(?:ly)?\s+non[\s-]*veg(?:etarian)?|only\s+non[\s-]*veg(?:etarian)?|strictly\s+non[\s-]*veg(?:etarian)?|non[\s-]*veg(?:etarian)?\s+only|non[\s-]*vegetarian|non[\s-]*veg)\b", lower):
        out["dietary_mode"] = "pure_non_vegetarian"
    elif re.search(r"\b(pure(?:ly)?\s+veg(?:etarian)?|only\s+veg(?:etarian)?|strictly\s+veg(?:etarian)?|vegetarian|veg)\b", lower):
        out["dietary_mode"] = "pure_vegetarian"

    # 2. Explicit 'no allergies' / 'no dislikes' / bare 'none'
    if re.search(r"\b(no\s+allerg\w*|not\s+allergic|no\s+intoleran\w*|zero\s+allerg\w*)\b", lower):
        out["no_allergies_stated"] = True
    if re.search(r"\b(no\s+dislik\w*|nothing\s+to\s+remove|no\s+remov\w*|no\s+exclud\w*|eat\s+everything|like\s+everything|all\s+foods\s+(?:are\s+)?(?:ok|fine))\b", lower):
        out["no_dislikes_stated"] = True

    bare_none = bool(re.fullmatch(r"\s*(?:none|no|nope|nothing|nil|na|n/a|all\s+good|go\s+ahead|generate(?:\s+plan)?)\s*[.!]?\s*", lower))
    if bare_none:
        if state.get("dietary_mode"):
            if not state.get("allergies_answered"):
                out["no_allergies_stated"] = True
            if not state.get("dislikes_answered"):
                out["no_dislikes_stated"] = True

    # 3. Allergies extraction (also catch lactose intolerance with common typos like 'intolorent')
    if re.search(r"\blactose\b", lower) and not out["no_allergies_stated"]:
        if "dairy" not in out["allergies"]:
            out["allergies"].append("dairy")

    for allergy_clause_match in re.finditer(
        r"(?:allerg(?:ic|y|ies)\s*(?:to|:)?|intoleran\w*\s*(?:to|:)?|intoloren\w*\s*(?:to|:)?)\s*([^.;\n]+)",
        lower,
    ):
        if out["no_allergies_stated"]:
            break
        clause = re.split(
            r",|\b(?:remove|exclude|dislike|don'?t\s+like|no\s+dislik|without|prefer|pure|non[\s-]*veg|vegetarian)\b",
            allergy_clause_match.group(1),
            maxsplit=1,
        )[0]
        for canonical, kws in ALLERGEN_KEYWORDS.items():
            if any(re.search(r"\b" + re.escape(w) + r"\b", clause) for w in kws):
                if canonical not in out["allergies"]:
                    out["allergies"].append(canonical)
        if not out["allergies"]:
            for part in re.split(r",|\band\b", clause):
                cleaned = re.sub(r"\b(i|am|have|a|an|the|to|food|foods|my)\b", " ", part).strip()
                if 2 <= len(cleaned) <= 25 and cleaned not in {"no", "none", "nothing"}:
                    out["allergies"].append(cleaned)

    # 4. Disliked / excluded dishes extraction (STOP before allergy/diet-mode clauses!)
    non_dislike_pat = re.compile(
        r"\b(allerg\w*|intoleran\w*|intoloren\w*|lactose|non[\s-]*veg\w*|pure(?:ly)?\s+veg\w*|vegetarian|combo)\b",
        re.IGNORECASE,
    )
    dislike_match = re.search(
        r"(?:remove|exclude|without|don'?t\s+(?:include|like|want|give)|do\s+not\s+(?:include|like|want)|hate|dislike|avoid|skip)\s+([^.;\n]+)",
        lower,
    )
    if dislike_match and not out["no_dislikes_stated"]:
        raw_clause = dislike_match.group(1)
        # Split by comma / 'and' / 'or' and STOP / skip any chunk mentioning allergies or diet style
        for chunk in re.split(r",|\band\b|\bor\b", raw_clause):
            if non_dislike_pat.search(chunk):
                continue
            clean = re.sub(r"\b(from|my|the|meal|plan|please|dish|dishes|food|foods|in|it)\b", " ", chunk).strip()
            if len(clean) >= 3 and clean not in {"none", "nothing", "allergies", "allergy", "dislikes"}:
                out["disliked_foods"].append(clean)

    # Also catch if user directly named any catalog dish when asked about dislikes (e.g., 'cabbage, baingan')
    if state.get("dietary_mode") and not out["disliked_foods"] and not out["no_dislikes_stated"] and not out["allergies"]:
        for food in catalog:
            fid_toks = [t for t in food.food_id.replace("recipe_", "").split("_") if len(t) >= 4 and t not in {"cooked", "fresh", "plain", "whole", "wheat", "steamed", "boiled", "roasted", "light", "baked", "grilled", "lowfat", "nonfat", "greek", "sweet", "seeds"}]
            if any(re.search(r"\b" + re.escape(t) + r"\b", lower) for t in fid_toks):
                if food.food_id not in out["disliked_foods"]:
                    out["disliked_foods"].append(food.food_id)

    return out


def converse_intake_preferences(
    message: str,
    dossier: dict[str, Any],
    current_state: Mapping[str, Any] | None = None,
    *,
    use_llm: bool = True,
) -> dict[str, Any]:
    """Conversational free-text intake: LLM understands diet style, allergies, and dislikes, then triggers Stage 3."""
    catalog = ps.s3.load_food_database()
    state: dict[str, Any] = {
        "dietary_mode": (current_state or {}).get("dietary_mode"),
        "allergies_answered": bool((current_state or {}).get("allergies_answered", False)),
        "allergies": list((current_state or {}).get("allergies", [])),
        "dislikes_answered": bool((current_state or {}).get("dislikes_answered", False)),
        "disliked_terms": list((current_state or {}).get("disliked_terms", [])),
        "excluded_food_ids": list((current_state or {}).get("excluded_food_ids", [])),
    }

    llm_parsed = None
    llm_used = False
    if use_llm:
        llm_parsed = _extract_intake_with_qwen(message, state, [f.name for f in catalog])
        llm_used = llm_parsed is not None

    det_parsed = _extract_intake_deterministic(message, state, catalog)

    # Merge LLM extraction + deterministic verification safely
    valid_modes = {"pure_vegetarian", "veg_nonveg_combo", "pure_non_vegetarian"}
    # Deterministic takes precedence when user explicitly writes unambiguous mode keywords
    chosen_mode = det_parsed["dietary_mode"]
    if not chosen_mode and llm_parsed and llm_parsed.get("dietary_mode") in valid_modes:
        chosen_mode = str(llm_parsed["dietary_mode"])
    if chosen_mode:
        state["dietary_mode"] = chosen_mode

    # Allergies
    merged_allergies = list(state["allergies"])
    for src in (det_parsed["allergies"], (llm_parsed or {}).get("allergies") or []):
        for a in src:
            clean = str(a).strip().lower()
            if clean and clean not in {"none", "no", "null", "n/a"} and clean not in merged_allergies:
                merged_allergies.append(clean)
    state["allergies"] = merged_allergies
    if merged_allergies or det_parsed["no_allergies_stated"] or bool((llm_parsed or {}).get("no_allergies_stated")):
        state["allergies_answered"] = True

    # Dislikes / dishes to remove (filter out any diet-mode or allergy phrases from LLM or regex)
    non_dislike_guard = re.compile(
        r"\b(allerg\w*|intoleran\w*|intoloren\w*|lactose|non[\s-]*veg\w*|vegetarian|combo|pure\s+veg)\b",
        re.IGNORECASE,
    )
    merged_dislikes = list(state["disliked_terms"])
    merged_excl_ids = set(state["excluded_food_ids"])
    raw_dislikes = list(det_parsed["disliked_foods"]) + list((llm_parsed or {}).get("disliked_foods") or [])
    for d in raw_dislikes:
        clean = str(d).strip().lower()
        if not clean or clean in {"none", "no", "null", "n/a", "nothing"}:
            continue
        if non_dislike_guard.search(clean):
            continue
        if clean.startswith("recipe_"):
            merged_excl_ids.add(clean)
        else:
            if clean not in merged_dislikes:
                merged_dislikes.append(clean)
            matched = _match_food(clean, [{"food_id": f.food_id, "name": f.name} for f in catalog])
            if matched:
                merged_excl_ids.add(str(matched["food_id"]))
    state["disliked_terms"] = merged_dislikes
    state["excluded_food_ids"] = sorted(merged_excl_ids)
    if merged_dislikes or merged_excl_ids or det_parsed["no_dislikes_stated"] or bool((llm_parsed or {}).get("no_dislikes_stated")):
        state["dislikes_answered"] = True

    # Determine if we still need to ask follow-up questions
    mode_labels = {
        "pure_vegetarian": "Pure Vegetarian",
        "veg_nonveg_combo": "Vegetarian & Non-Vegetarian Combo",
        "pure_non_vegetarian": "Purely Non-Vegetarian",
    }
    missing_qs: list[str] = []
    if not state["dietary_mode"]:
        missing_qs.append("whether you prefer **pure vegetarian**, a **vegetarian and non-vegetarian combo**, or **purely non-vegetarian**")
    if not state["allergies_answered"]:
        missing_qs.append("whether you have any **food allergies or intolerances** (or type 'no allergies')")
    if not state["dislikes_answered"]:
        missing_qs.append("any **dishes or ingredients you'd like removed** (or type 'none')")

    if missing_qs:
        ack_parts = []
        if state["dietary_mode"]:
            ack_parts.append(f"Diet style: **{mode_labels[state['dietary_mode']]}**")
        if state["allergies"]:
            ack_parts.append(f"Allergies: **{', '.join(state['allergies'])}**")
        elif state["allergies_answered"]:
            ack_parts.append("Allergies: **None**")
        if state["disliked_terms"] or state["excluded_food_ids"]:
            shown = state["disliked_terms"] or state["excluded_food_ids"]
            ack_parts.append(f"Exclude: **{', '.join(shown)}**")
        elif state["dislikes_answered"]:
            ack_parts.append("Dislikes: **None**")

        prefix = f"Got it ({' · '.join(ack_parts)}). " if ack_parts else ""
        ask_text = "Before I generate your meal plan, could you also tell me " + " and ".join(missing_qs) + "?"
        return {
            "assistant_reply": prefix + ask_text,
            "state": state,
            "ready_to_generate": False,
            "llm_used": llm_used,
            "meal_plan": None,
        }

    # All preferences gathered -> run Stage 3 optimization immediately!
    plan = ps.run_optimization(
        copy.deepcopy(dossier),
        dietary_preference=state["dietary_mode"],
        allergies=state["allergies"],
        excluded_food_ids=state["excluded_food_ids"],
        disliked_terms=state["disliked_terms"],
    )
    summary_bits = [f"Diet: {mode_labels[state['dietary_mode']]}"]
    summary_bits.append(f"Allergies: {', '.join(state['allergies']) if state['allergies'] else 'None'}")
    excl_shown = state["disliked_terms"] or state["excluded_food_ids"]
    summary_bits.append(f"Excluded: {', '.join(excl_shown) if excl_shown else 'None'}")

    if plan.get("accepted"):
        allergy_warn = (
            " Note: because an allergy was reported and catalog allergen records are incomplete, "
            "this plan is flagged for clinician/dietitian review before use."
            if state["allergies"]
            else ""
        )
        reply = (
            f"Understood ({' · '.join(summary_bits)}). "
            f"I have generated a checked {len(plan['solved_food_basket'])}-dish meal plan below! "
            "You can use the meal-plan chatbot right underneath to remove or swap any dish anytime."
            + allergy_warn
        )
    else:
        fail_msg = (plan.get("failure") or {}).get("message", "No valid meal plan passed all active constraints.")
        reply = (
            f"Understood ({' · '.join(summary_bits)}), but Stage 3 could not satisfy all active nutrient constraints with those exclusions. "
            f"{fail_msg} You can type adjusted preferences here to try again."
        )

    return {
        "assistant_reply": reply,
        "state": state,
        "ready_to_generate": True,
        "llm_used": llm_used,
        "meal_plan": plan,
    }
