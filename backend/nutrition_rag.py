"""Small local retrieval layer over the bundled food composition dataset.

This retrieves nutrient facts for foods already present in the repository. It is
not a general nutrition guideline library and should not be presented as one.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
FOOD_DB = ROOT / "food_composition_ifct2017.json"
TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)
STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "for", "to", "is", "are", "does",
    "do", "what", "how", "much", "many", "show", "tell", "give", "me", "about", "contains",
    "contain", "per", "serving", "unit", "basis", "food", "foods", "nutrition", "nutrient",
    "nutrients", "protein", "carbohydrate", "carbs", "sugar", "sugars", "fiber", "fibre", "fat",
    "saturated", "cholesterol", "calcium", "iron", "sodium", "potassium", "vitamin", "calories",
    "calorie", "energy", "g", "mg", "kcal", "gram", "grams", "milligram", "milligrams",
}
NUTRIENT_LABELS = {
    "DR1TKCAL": "energy kcal", "DR1TPROT": "protein g", "DR1TCARB": "carbohydrate g",
    "DR1TSUGR": "sugars g", "DR1TFIBE": "fiber g", "DR1TTFAT": "fat g",
    "DR1TSFAT": "saturated fat g", "DR1TCHOL": "cholesterol mg",
    "DR1TCALC": "calcium mg", "DR1TIRON": "iron mg",
    "DR1TSODI": "sodium mg", "DR1TPOTA": "potassium mg", "DR1TVITC": "vitamin C mg",
}


def _foods() -> list[dict[str, Any]]:
    with FOOD_DB.open(encoding="utf-8") as stream:
        rows = json.load(stream)
    return [row for row in rows if isinstance(row, dict) and row.get("name") and isinstance(row.get("nutrients_per_serving"), dict)]


def retrieve_food_facts(query: str, limit: int = 4) -> list[dict[str, Any]]:
    terms = {term for term in TOKEN_RE.findall(query.lower()) if term not in STOPWORDS}
    if not terms:
        return []
    hits = []
    for food in _foods():
        name = str(food["name"])
        searchable = set(TOKEN_RE.findall((name + " " + str(food.get("category", ""))).lower()))
        # Strongly prefer exact food-name matches; avoid broad, unrelated hits.
        overlap = len(terms & searchable)
        if overlap == 0:
            continue
        exact = name.lower() in query.lower()
        score = overlap + (3 if exact else 0)
        nutrients = food["nutrients_per_serving"]
        facts = {label: nutrients[key] for key, label in NUTRIENT_LABELS.items() if key in nutrients}
        hits.append({
            "food_id": food.get("food_id"), "name": name,
            "serving": food.get("serving_unit", "per listed serving"),
            "serving_weight_g": food.get("serving_weight_g"),
            "nutrients": facts, "source": food.get("data_source", "Food composition data"),
            "score": score,
        })
    hits.sort(key=lambda item: (-item["score"], item["name"]))
    return [{k: v for k, v in item.items() if k != "score"} for item in hits[:limit]]
