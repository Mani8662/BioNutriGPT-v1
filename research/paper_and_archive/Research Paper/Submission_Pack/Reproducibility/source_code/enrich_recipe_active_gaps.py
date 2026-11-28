"""Ingredient-level calculation of the eight missing active nutrients for INDB recipes.

Maps recipe ingredients with documented gram weights against three pinned,
checksum-verified food composition sources:
  1. USDA FoodData Central SR Legacy (April 2018 CSV archive) — primary source
     for all 8 NHANES-compatible nutrient definitions.
  2. ICMR-NIN IFCT 2017 (@ifct2017/compositions CSV) — laboratory alpha-tocopherol
     (tocpha) and plant-food provitamin-A carotenoids (bcar/12 + acar/24 + crypxb/24)
     when USDA lacks the Indian ingredient or field. Never used for animal retinol.
  3. GOV.UK CoFID 2021 (McCance & Widdowson Integrated Dataset) — exact INDB-linked
     UK food codes (e.g., UK-13-829 Garam masala, 12-495 Paneer, 13-243 Fenugreek
     leaves, 13-821 Curry leaves, 13-803 Asafoetida) for documented fields only.

Strict provenance policy:
  - No nutrient is guessed or silently zero-filled when absent in all three sources.
  - If even one ingredient in a recipe lacks a documented value for a nutrient,
    that recipe's value remains null (unknown), while full per-ingredient subtotals
    and missing ingredient identities are logged in nutrient_provenance.
  - Calculated sums represent raw/input ingredient totals scaled to standardized
    gram portions without cooking retention factors (explicitly disclosed).
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent
USDA_DIR = ROOT / "Data/food_sources/USDA_SR_Legacy"
IFCT_PATH = ROOT / "Data/food_sources/IFCT2017/ifct2017_compositions.csv"
COFID_PATH = ROOT / "Data/food_sources/UK_CoFID/CoFID_2021.xlsx"

EXPECTED_SHA256 = {
    "usda_sr_legacy_zip": "b80817294b8850530aaedf2e515c02593b1824f763a0ff356e5c2081643e6fd0",
    "ifct2017_compositions_csv": "678c9285b01290ad5cc318a560fa5634618ebd413dd9fe3164d5539125c4aeef",
    "uk_cofid_2021_xlsx": "436e9445ef2adb2a75f3d7edd51302de3adad25385f9795fc94ba58bd030e97d",
}

TARGET_8_USDA_IDS = {
    "DR1TVB12": 1178,  # Vitamin B-12 (ug)
    "DR1TATOC": 1109,  # Vitamin E, alpha-tocopherol (mg)
    "DR1TRET": 1105,   # Retinol (ug)
    "DR1TVARA": 1106,  # Vitamin A, RAE (ug)
    "DR1TFA": 1186,    # Folic acid (ug)
    "DR1TFDFE": 1190,  # Folate, DFE (ug)
    "DR1TCAFF": 1057,  # Caffeine (mg)
    "DR1TALCO": 1018,  # Alcohol, ethyl (g)
}

# Explicit, reviewed ingredient mappings to USDA SR Legacy FDC IDs.
# Unenriched dry/raw counterparts are used; US folic-acid-enriched grains are excluded.
USDA_INGREDIENT_MAP: dict[str, tuple[int, str]] = {
    "A014": (169758, "exact_unenriched_parboiled_rice"),
    "A019": (168893, "exact_whole_grain_wheat_flour"),
    "B001": (173756, "same_species_chickpea_raw_proxy_for_split_dal"),
    "B003": (174259, "exact_species_mungo_beans_raw"),
    "B010": (174256, "exact_species_mung_beans_raw"),
    "B015": (172420, "exact_species_lentils_raw"),
    "B020": (173744, "exact_species_red_kidney_beans_raw"),
    "B021": (172436, "exact_species_pigeon_peas_raw"),
    "B509": (173756, "exact_indb_us_173756_chickpeas_raw"),
    "C015": (169975, "exact_cabbage_raw"),
    "C033": (168462, "exact_spinach_raw"),
    "D031": (169228, "exact_eggplant_raw"),
    "D036": (169986, "exact_cauliflower_raw"),
    "D046": (170483, "exact_drumstick_pods_raw"),
    "D050": (169961, "exact_green_snap_beans_raw"),
    "D056": (169260, "exact_okra_raw"),
    "D061": (170419, "exact_green_peas_raw"),
    "D075": (170457, "exact_tomatoes_red_ripe_raw"),
    "E064": (167763, "exact_tamarinds_raw"),
    "E520": (167746, "exact_lemons_raw_without_peel"),
    "F002": (170393, "exact_carrots_raw"),
    "F006": (170026, "exact_potatoes_flesh_and_skin_raw"),
    "G008": (170497, "exact_green_hot_chili_peppers_raw"),
    "G009": (169997, "exact_coriander_cilantro_leaves_raw"),
    "G012": (169230, "exact_garlic_raw"),
    "G014": (169231, "exact_ginger_root_raw"),
    "G017": (170000, "exact_onions_raw"),
    "G021": (170919, "species_cardamom_spice"),
    "G022": (170106, "exact_red_hot_chili_peppers_raw"),
    "G023": (171321, "exact_cloves_ground"),
    "G024": (170922, "exact_coriander_seed"),
    "G025": (170923, "exact_cumin_seed"),
    "G026": (171324, "exact_fenugreek_seed"),
    "G033": (172231, "exact_turmeric_ground"),
    "G512": (170917, "exact_bay_leaf_dried"),
    "G516": (170932, "pure_red_cayenne_chilli_powder"),
    "G518": (171320, "exact_cinnamon_ground"),
    "G528": (173468, "exact_table_salt"),
    "H007": (170169, "exact_coconut_meat_raw"),
    "H012": (172430, "exact_peanuts_raw"),
    "H013": (170929, "exact_mustard_seed_ground"),
    "K505": (174158, "generic_bottled_water_zero_nutrients"),
    "T501": (173430, "exact_butter_without_salt"),
    "T506": (173412, "exact_anhydrous_butter_oil_ghee"),
    "T508": (171017, "exact_sunflower_oil_linoleic"),
}

# Explicit, reviewed UK CoFID 2021 food code mappings (used where USDA lacks
# the ingredient or specific field; missing 'N'/'Tr'/blank cells stay null).
COFID_INGREDIENT_MAP: dict[str, tuple[str, str]] = {
    "G523": ("13-829", "exact_indb_uk_13_829_garam_masala"),
    "G512": ("13-806", "exact_indb_uk_13_806_bay_leaf_dried"),
    "G024": ("13-875", "exact_cofid_coriander_seeds"),
    "G021": ("13-809", "exact_cofid_cardamom_ground"),
    "G026": ("13-828", "exact_cofid_fenugreek_seeds"),
    "L003": ("12-495", "exact_cofid_cheese_paneer"),
    "C020": ("13-243", "exact_cofid_fenugreek_leaves_raw"),
    "G010": ("13-821", "exact_cofid_curry_leaves_fresh"),
    "G019": ("13-803", "exact_cofid_asafoetida"),
    "B003": ("13-060", "exact_cofid_black_gram_dhuli_urad_dal_raw"),
    "B021": ("13-104", "exact_cofid_pigeon_peas_dal_raw"),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_cofid_num(val: Any) -> float | None:
    if pd.isna(val):
        return None
    text = str(val).strip()
    if text in ("", "N", "Tr", "nan", "NaN"):
        return None
    try:
        f = float(text)
        return f if math.isfinite(f) and f >= 0 else None
    except (ValueError, TypeError):
        return None


def _parse_pos_float(val: Any, scale: float = 1.0) -> float | None:
    if pd.isna(val):
        return None
    try:
        f = float(val) * scale
        return f if math.isfinite(f) and f >= 0 else None
    except (ValueError, TypeError):
        return None


class IngredientGapResolver:
    def __init__(self) -> None:
        # Verify pinned archive/file checksums before reading
        usda_zip = USDA_DIR / "source.zip"
        for label, path in (
            ("usda_sr_legacy_zip", usda_zip),
            ("ifct2017_compositions_csv", IFCT_PATH),
            ("uk_cofid_2021_xlsx", COFID_PATH),
        ):
            actual = _sha256(path)
            if actual != EXPECTED_SHA256[label]:
                raise ValueError(f"Checksum mismatch for {label}: {actual} != {EXPECTED_SHA256[label]}")

        # 1. Load USDA SR Legacy CSVs
        usdir = next(USDA_DIR.rglob("food.csv")).parent
        self.us_foods: dict[int, str] = {}
        with (usdir / "food.csv").open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                self.us_foods[int(row["fdc_id"])] = row["description"]

        target_fdc = {fdc for fdc, _ in USDA_INGREDIENT_MAP.values()}
        target_nids = set(TARGET_8_USDA_IDS.values())
        self.us_vals: dict[tuple[int, int], dict[str, Any]] = {}
        with (usdir / "food_nutrient.csv").open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                fdc = int(row["fdc_id"])
                nid = int(row["nutrient_id"])
                if fdc in target_fdc and nid in target_nids and row["amount"] != "":
                    self.us_vals[(fdc, nid)] = {
                        "amount": float(row["amount"]),
                        "source_food_nutrient_id": row["id"],
                        "derivation_id": row["derivation_id"],
                        "data_points": row["data_points"],
                    }

        # 2. Load IFCT 2017 raw laboratory CSV (masses in g/100g)
        df_ifct = pd.read_csv(IFCT_PATH, low_memory=False)
        self.ifct: dict[str, dict[str, Any]] = {}
        for _, row in df_ifct.iterrows():
            code = str(row["Food Code; code"]).strip()
            self.ifct[code] = {
                "name": str(row["Food Name; name"]).strip(),
                "tocpha_mg": _parse_pos_float(row.get("α-Tocopherol; tocpha"), 1e3),
                "cartb_ug": _parse_pos_float(row.get("β-Carotene; cartb"), 1e6),
                "carta_ug": _parse_pos_float(row.get("α-Carotene; carta"), 1e6),
                "crypxb_ug": _parse_pos_float(row.get("β-Cryptoxanthin; crypxb"), 1e6),
            }

        # 3. Load UK CoFID 2021 sheets
        vit = pd.read_excel(COFID_PATH, sheet_name="1.5 Vitamins", header=0)
        vfrac = pd.read_excel(COFID_PATH, sheet_name="1.6 Vitamin Fractions", header=0)
        prox = pd.read_excel(COFID_PATH, sheet_name="1.3 Proximates", header=0)
        self.cofid_vit = {str(r["Food Code"]).strip(): r for _, r in vit.iterrows()}
        self.cofid_frac = {str(r["Food Code"]).strip(): r for _, r in vfrac.iterrows()}
        self.cofid_prox = {str(r["Food Code"]).strip(): r for _, r in prox.iterrows()}

    def resolve(self, food_code: str, nutrient_code: str) -> tuple[float | None, dict[str, Any]]:
        """Resolve per-100g ingredient value and provenance for one of the 8 active gaps."""
        nid = TARGET_8_USDA_IDS[nutrient_code]

        # Tier 1: USDA SR Legacy direct match
        if food_code in USDA_INGREDIENT_MAP:
            fdc, match_note = USDA_INGREDIENT_MAP[food_code]
            rec = self.us_vals.get((fdc, nid))
            if rec is not None:
                return rec["amount"], {
                    "source": "USDA_SR_Legacy",
                    "fdc_id": fdc,
                    "source_description": self.us_foods[fdc],
                    "match_basis": match_note,
                    "nutrient_id": nid,
                    "source_food_nutrient_id": rec["source_food_nutrient_id"],
                    "derivation_id": rec["derivation_id"],
                    "data_points": rec["data_points"],
                    "value_per_100g": rec["amount"],
                }

        # Tier 2: ICMR-NIN IFCT 2017 laboratory alpha-tocopherol & plant-food HPLC RAE
        if food_code in self.ifct:
            ir = self.ifct[food_code]
            if nutrient_code == "DR1TATOC" and ir["tocpha_mg"] is not None:
                return ir["tocpha_mg"], {
                    "source": "ICMR_NIN_IFCT_2017",
                    "ifct_code": food_code,
                    "source_description": ir["name"],
                    "source_column": "α-Tocopherol; tocpha",
                    "value_per_100g": ir["tocpha_mg"],
                }
            # Plant foods only (categories A-H): preformed retinol = 0; RAE = bcar/12 + acar/24 + crypxb/24
            if nutrient_code == "DR1TVARA" and food_code[:1] in "ABCDEFGH":
                if ir["cartb_ug"] is not None and ir["carta_ug"] is not None and ir["crypxb_ug"] is not None:
                    rae = ir["cartb_ug"] / 12.0 + ir["carta_ug"] / 24.0 + ir["crypxb_ug"] / 24.0
                    return rae, {
                        "source": "ICMR_NIN_IFCT_2017_plant_RAE",
                        "ifct_code": food_code,
                        "source_description": ir["name"],
                        "formula": "cartb/12 + carta/24 + crypxb/24 (plant food, preformed retinol=0)",
                        "value_per_100g": round(rae, 6),
                    }

        # Tier 3: GOV.UK CoFID 2021 exact code match (only documented numeric fields)
        if food_code in COFID_INGREDIENT_MAP:
            cc, match_note = COFID_INGREDIENT_MAP[food_code]
            vr = self.cofid_vit.get(cc)
            fr = self.cofid_frac.get(cc)
            pr = self.cofid_prox.get(cc)
            desc = str(vr["Food Name"]).strip() if vr is not None else cc
            if nutrient_code == "DR1TVB12" and vr is not None:
                v = _parse_cofid_num(vr.get("Vitamin B12 (µg)"))
                if v is not None:
                    return v, {"source": "UK_CoFID_2021", "cofid_code": cc, "source_description": desc,
                               "match_basis": match_note, "source_column": "Vitamin B12 (µg)", "value_per_100g": v}
            if nutrient_code == "DR1TRET" and vr is not None:
                v = _parse_cofid_num(vr.get("Retinol (µg)"))
                if v is not None:
                    return v, {"source": "UK_CoFID_2021", "cofid_code": cc, "source_description": desc,
                               "match_basis": match_note, "source_column": "Retinol (µg)", "value_per_100g": v}
            if nutrient_code == "DR1TATOC" and fr is not None:
                v = _parse_cofid_num(fr.get("Alpha-tocopherol (mg)"))
                if v is not None:
                    return v, {"source": "UK_CoFID_2021", "cofid_code": cc, "source_description": desc,
                               "match_basis": match_note, "source_column": "Alpha-tocopherol (mg)", "value_per_100g": v}
            # Special documented INDB internal spice formulation: BFP001 Garam masala (112 g batch)
            if food_code == "G523" and nutrient_code == "DR1TATOC":
                bfp001_components = (
                    ("G021", 20.0), ("G023", 10.0), ("G024", 40.0),
                    ("G025", 20.0), ("G031", 20.0), ("G518", 2.0),
                )
                tot_g = sum(g for _, g in bfp001_components)
                tot_mg = 0.0
                parts = []
                for scode, grams in bfp001_components:
                    if scode == "G031":
                        val = self.ifct["G031"]["tocpha_mg"]
                        src_ref = "IFCT:G031"
                    else:
                        val, pinfo = self.resolve(scode, "DR1TATOC")
                        src_ref = pinfo.get("source", "unknown")
                    if val is None:
                        tot_mg = None
                        break
                    tot_mg += val * grams / 100.0
                    parts.append({"food_code": scode, "grams": grams, "tocpha_mg_per_100g": val, "source": src_ref})
                if tot_mg is not None:
                    per100 = round(tot_mg * 100.0 / tot_g, 6)
                    return per100, {
                        "source": "INDB_BFP001_spice_components",
                        "indb_recipe_code": "BFP001",
                        "source_description": "Garam masala calculated from INDB recipes.xlsx BFP001 (112 g batch)",
                        "value_per_100g": per100,
                        "components": parts,
                    }
            if nutrient_code == "DR1TVARA" and vr is not None and fr is not None:
                ret = _parse_cofid_num(vr.get("Retinol (µg)"))
                bcar = _parse_cofid_num(fr.get("Beta-carotene (µg)"))
                acar = _parse_cofid_num(fr.get("Alpha-carotene (µg)"))
                if ret is not None and bcar is not None:
                    rae = ret + bcar / 12.0 + ((acar or 0.0) / 24.0)
                    return rae, {"source": "UK_CoFID_2021_RAE", "cofid_code": cc, "source_description": desc,
                                 "match_basis": match_note, "formula": "Retinol + Beta-carotene/12 + Alpha-carotene/24",
                                 "value_per_100g": round(rae, 6)}
            if nutrient_code == "DR1TALCO" and pr is not None:
                v = _parse_cofid_num(pr.get("Alcohol (g)"))
                if v is not None:
                    return v, {"source": "UK_CoFID_2021", "cofid_code": cc, "source_description": desc,
                               "match_basis": match_note, "source_column": "Alcohol (g)", "value_per_100g": v}
            if nutrient_code in ("DR1TFA", "DR1TFDFE") and vr is not None:
                fol = _parse_cofid_num(vr.get("Folate (µg)"))
                if fol == 0.0:
                    return 0.0, {"source": "UK_CoFID_2021_zero_total_folate", "cofid_code": cc,
                                 "source_description": desc, "match_basis": match_note,
                                 "source_column": "Folate (µg) == 0 -> folic_acid=0, DFE=0", "value_per_100g": 0.0}

        return None, {"status": "unknown", "reason": "No documented compatible value in USDA SR Legacy, IFCT 2017, or UK CoFID 2021"}


def enrich_indb_recipes(catalog: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate the 8 missing active nutrients for INDB recipes from ingredient quantities."""
    resolver = IngredientGapResolver()
    report: dict[str, Any] = {
        "sources_sha256": EXPECTED_SHA256,
        "target_active_nutrients": list(TARGET_8_USDA_IDS.keys()),
        "method_note": (
            "Summed per-ingredient documented values scaled to grams_per_standardized_portion / 100 "
            "without cooking retention adjustments. Any recipe with >=1 unmapped/missing ingredient "
            "for a nutrient keeps null (unknown) for that nutrient."
        ),
        "filled_counts_by_nutrient": {k: 0 for k in TARGET_8_USDA_IDS},
        "recipes_complete_all_8": [],
        "recipe_details": [],
    }

    for item in catalog:
        if "recipe_code" not in item.get("recipe_source", {}):
            continue
        rec_summary = {
            "food_id": item["food_id"],
            "name": item["name"],
            "category": item["category"],
            "filled_active_gaps": [],
            "remaining_unknown_active_gaps": {},
        }
        for ncode in TARGET_8_USDA_IDS:
            # Never overwrite an already-populated INDB field
            if item["nutrients_per_serving"].get(ncode) is not None:
                continue
            known_subtotal = 0.0
            mapped_entries = []
            missing_entries = []
            for ing in item["ingredients"]:
                code = str(ing["food_code"])
                grams = float(ing["grams_per_standardized_portion"])
                val100, prov = resolver.resolve(code, ncode)
                if val100 is None:
                    missing_entries.append({
                        "food_code": code,
                        "name": ing["name"],
                        "grams_per_standardized_portion": round(grams, 4),
                        "reason": prov["reason"],
                    })
                else:
                    contrib = val100 * grams / 100.0
                    known_subtotal += contrib
                    mapped_entries.append({
                        "food_code": code,
                        "name": ing["name"],
                        "grams_per_standardized_portion": round(grams, 4),
                        "contribution_per_serving": round(contrib, 6),
                        **prov,
                    })

            if not missing_entries:
                item["nutrients_per_serving"][ncode] = round(known_subtotal, 6)
                item["nutrient_provenance"][ncode] = {
                    "status": "ingredient_calculated_complete",
                    "method": "sum(ingredient_per_100g * grams_per_standardized_portion / 100); no cooking retention factor applied",
                    "ingredient_count": len(mapped_entries),
                    "ingredients": mapped_entries,
                }
                report["filled_counts_by_nutrient"][ncode] += 1
                rec_summary["filled_active_gaps"].append(ncode)
            else:
                item["nutrients_per_serving"][ncode] = None
                item["nutrient_provenance"][ncode] = {
                    "status": "unknown_partial_ingredients",
                    "known_subtotal_per_serving": round(known_subtotal, 6),
                    "mapped_ingredient_count": len(mapped_entries),
                    "missing_ingredient_count": len(missing_entries),
                    "missing_ingredients": missing_entries,
                }
                rec_summary["remaining_unknown_active_gaps"][ncode] = [
                    f"{m['food_code']} ({m['grams_per_standardized_portion']} g)" for m in missing_entries
                ]

        item["unknown_nutrients"] = [k for k, v in item["nutrients_per_serving"].items() if v is None]
        if not rec_summary["remaining_unknown_active_gaps"]:
            report["recipes_complete_all_8"].append(item["food_id"])
        report["recipe_details"].append(rec_summary)

    # Attach explicit, conservative allergen metadata to ALL catalog items.
    # Rule: unknown allergy information is NEVER interpreted as allergen-free.
    ALLERGEN_BY_CODE = {
        "A019": ["gluten", "wheat"],
        "H012": ["peanuts"],
        "H007": ["coconut"],
        "H013": ["mustard"],
        "L003": ["dairy", "milk"],
        "T501": ["dairy", "milk"],
        "T506": ["dairy", "milk"],
    }
    ALLERGEN_BY_FOOD_ID = {
        "recipe_skim_curd": ["dairy", "milk"],
        "recipe_lowfat_curd": ["dairy", "milk"],
        "recipe_skim_milk": ["dairy", "milk"],
        "recipe_lowfat_milk_1pct": ["dairy", "milk"],
        "recipe_greek_yogurt_nonfat": ["dairy", "milk"],
        "recipe_almonds_raw": ["tree_nuts", "almonds"],
        "recipe_boiled_egg": ["egg"],
        "recipe_baked_salmon": ["fish"],
        "recipe_light_tuna": ["fish"],
        "recipe_sardines_bone": ["fish"],
        "recipe_pink_salmon_canned": ["fish"],
        "recipe_firm_tofu_calcium": ["soy"],
    }
    for item in catalog:
        detected: set[str] = set(ALLERGEN_BY_FOOD_ID.get(item["food_id"], []))
        for ing in item.get("ingredients", []):
            code = str(ing.get("food_code", ""))
            detected.update(ALLERGEN_BY_CODE.get(code, []))
        item["allergen_info"] = {
            "known_allergens": sorted(detected),
            "allergen_verification_status": "ingredient_derived_unverified_cross_contact",
            "unknown_allergens_possible": True,
            "safety_note": (
                "Known major allergens detected from recorded source ingredients only. "
                "Absence from this list does NOT certify the dish allergen-free; trace cross-contact "
                "and unrecorded spice/sub-ingredient allergens remain unknown."
            ),
        }

    return report
