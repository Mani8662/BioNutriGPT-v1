"""Build source-traceable household foods and an explicit 44-dish coverage registry.

No nutrient is inferred from a dish name. INDB's published calculations are used
as published, scaled to gram portions; missing or semantically different fields
remain null. USDA is used for single-food entries only.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from build_indian_food_database import NUTRIENT_CODES

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'food_sources/INDB'
COMMIT = 'fbdb62bec519c6e2468582a799fb47e194b42558'
URL = f'https://github.com/lindsayjaacks/Indian-Nutrient-Databank-INDB-/tree/{COMMIT}'
RECIPE_MATCHES = {
    'recipe_moong_dal_tadka': 'ASC151', 'recipe_masoor_dal': 'ASC157',
    'recipe_chana_masala': 'ASC162', 'recipe_rajma_curry': 'ASC165',
    'recipe_sambar_vegetable': 'ASC167', 'recipe_dal_palak': 'BFP172',
    'recipe_panchmel_dal': 'OSR144', 'recipe_whole_wheat_phulka': 'ASC096',
    'recipe_steamed_rice': 'ASC113', 'recipe_veg_pulao': 'ASC115',
    'recipe_moong_dal_khichdi': 'BFP144', 'recipe_steamed_idli': 'ASC144',
    'recipe_plain_dosa': 'BFP148', 'recipe_vegetable_poha': 'BFP045',
    'recipe_bhindi_masala': 'BFP269', 'recipe_aloo_gobi': 'ASC171',
    'recipe_baingan_bharta': 'ASC177', 'recipe_cabbage_peas': 'ASC173',
    'recipe_methi_aloo': 'ASC175',
    'recipe_palak_paneer': 'ASC215',
}
SINGLE_MATCHES = {
    'recipe_brown_rice_cooked': 169704, 'recipe_skim_curd': 170887,
    'recipe_lowfat_curd': 170886, 'recipe_fresh_guava': 173044,
    'recipe_fresh_papaya': 169926, 'recipe_fresh_apple': 171688,
    'recipe_fresh_orange': 169097, 'recipe_fresh_banana': 173944,
    'recipe_pomegranate_arils': 169134,
    # Documented USDA SR Legacy lacto-vegetarian staples & nutrient-gap backups:
    'recipe_skim_milk': 171269,
    'recipe_almonds_raw': 170567,
    'recipe_lowfat_milk_1pct': 170872,
    'recipe_greek_yogurt_nonfat': 170894,
    'recipe_carrots_cooked': 170394,
    'recipe_sweet_potato_cooked': 168484,
    'recipe_spinach_cooked': 168463,
    'recipe_broccoli_cooked': 169967,
    'recipe_oats_cooked': 173905,
    'recipe_sunflower_seeds': 170563,
    'recipe_pumpkin_seeds': 170557,
    'recipe_chickpeas_boiled': 173757,
    'recipe_firm_tofu_calcium': 172448,
    'recipe_kale_cooked': 169238,
    # Documented USDA SR Legacy non-vegetarian protein dishes:
    'recipe_boiled_egg': 173424,
    'recipe_grilled_chicken_breast': 171477,
    'recipe_baked_salmon': 175168,
    'recipe_light_tuna': 173709,
    'recipe_sardines_bone': 175139,
    'recipe_pink_salmon_canned': 173690,
}
NON_VEG_FOOD_IDS = {
    'recipe_boiled_egg',
    'recipe_grilled_chicken_breast',
    'recipe_baked_salmon',
    'recipe_light_tuna',
    'recipe_sardines_bone',
    'recipe_pink_salmon_canned',
}
EXTRA_SINGLE_ITEMS = [
    {'food_id': 'recipe_skim_milk', 'name': 'Skim Milk (1 glass)', 'category': 'dairy', 'serving_weight_g': 250.0, 'selection_cost': 1.0},
    {'food_id': 'recipe_almonds_raw', 'name': 'Raw Almonds (1 handful)', 'category': 'fruit', 'serving_weight_g': 28.0, 'selection_cost': 1.0},
    {'food_id': 'recipe_lowfat_milk_1pct', 'name': 'Low-Fat Milk 1% (1 glass)', 'category': 'dairy', 'serving_weight_g': 250.0, 'selection_cost': 1.05},
    {'food_id': 'recipe_greek_yogurt_nonfat', 'name': 'Plain Nonfat Greek Yogurt (1 bowl)', 'category': 'dairy', 'serving_weight_g': 200.0, 'selection_cost': 1.05},
    {'food_id': 'recipe_carrots_cooked', 'name': 'Boiled Carrots (1 bowl)', 'category': 'vegetable_sabzi', 'serving_weight_g': 120.0, 'selection_cost': 1.05},
    {'food_id': 'recipe_sweet_potato_cooked', 'name': 'Boiled Sweet Potato (1 medium)', 'category': 'vegetable_sabzi', 'serving_weight_g': 150.0, 'selection_cost': 1.05},
    {'food_id': 'recipe_spinach_cooked', 'name': 'Cooked Spinach (1 bowl)', 'category': 'vegetable_sabzi', 'serving_weight_g': 120.0, 'selection_cost': 1.05},
    {'food_id': 'recipe_broccoli_cooked', 'name': 'Steamed Broccoli (1 bowl)', 'category': 'vegetable_sabzi', 'serving_weight_g': 120.0, 'selection_cost': 1.05},
    {'food_id': 'recipe_oats_cooked', 'name': 'Cooked Oatmeal (1 bowl)', 'category': 'cereal_grain', 'serving_weight_g': 200.0, 'selection_cost': 1.05},
    {'food_id': 'recipe_sunflower_seeds', 'name': 'Roasted Sunflower Seeds (1 handful)', 'category': 'fruit', 'serving_weight_g': 28.0, 'selection_cost': 1.05},
    {'food_id': 'recipe_pumpkin_seeds', 'name': 'Roasted Pumpkin Seeds (1 handful)', 'category': 'fruit', 'serving_weight_g': 28.0, 'selection_cost': 1.05},
    {'food_id': 'recipe_chickpeas_boiled', 'name': 'Boiled Chickpeas / Chana (1 bowl)', 'category': 'dal_pulse', 'serving_weight_g': 150.0, 'selection_cost': 1.05},
    {'food_id': 'recipe_firm_tofu_calcium', 'name': 'Calcium-Set Firm Tofu (150 g)', 'category': 'dal_pulse', 'serving_weight_g': 150.0, 'selection_cost': 1.05},
    {'food_id': 'recipe_kale_cooked', 'name': 'Boiled Kale (1 bowl)', 'category': 'vegetable_sabzi', 'serving_weight_g': 130.0, 'selection_cost': 1.05},
    {'food_id': 'recipe_boiled_egg', 'name': 'Hard-Boiled Eggs (2 eggs)', 'category': 'non_veg_protein', 'serving_weight_g': 100.0, 'selection_cost': 1.0},
    {'food_id': 'recipe_grilled_chicken_breast', 'name': 'Roasted Chicken Breast (120 g)', 'category': 'non_veg_protein', 'serving_weight_g': 120.0, 'selection_cost': 1.0},
    {'food_id': 'recipe_baked_salmon', 'name': 'Baked Atlantic Salmon (120 g)', 'category': 'non_veg_protein', 'serving_weight_g': 120.0, 'selection_cost': 1.05},
    {'food_id': 'recipe_light_tuna', 'name': 'Light Tuna in Water (100 g)', 'category': 'non_veg_protein', 'serving_weight_g': 100.0, 'selection_cost': 1.05},
    {'food_id': 'recipe_sardines_bone', 'name': 'Atlantic Sardines with Bone (100 g)', 'category': 'non_veg_protein', 'serving_weight_g': 100.0, 'selection_cost': 1.05},
    {'food_id': 'recipe_pink_salmon_canned', 'name': 'Pink Salmon Canned with Bone (100 g)', 'category': 'non_veg_protein', 'serving_weight_g': 100.0, 'selection_cost': 1.05},
]
# Source column, multiplier. Fatty acids are in mg in INDB, g in NHANES.
# Free sugar != total sugar; vitamin A != an explicitly documented RAE value;
# Total folate and vitamin K1 match the CDC definitions; separate food-folate/
# DFE fields and generic vitamin A/E definitions cannot be substituted.
INDB_FIELDS = {
    'DR1TKCAL': ('energy_kcal', 1), 'DR1TPROT': ('protein_g', 1),
    'DR1TCARB': ('carb_g', 1), 'DR1TTFAT': ('fat_g', 1),
    'DR1TFIBE': ('fibre_g', 1), 'DR1TSFAT': ('sfa_mg', .001),
    'DR1TMFAT': ('mufa_mg', .001), 'DR1TPFAT': ('pufa_mg', .001),
    'DR1TCHOL': ('cholesterol_mg', 1), 'DR1TCALC': ('calcium_mg', 1),
    'DR1TPHOS': ('phosphorus_mg', 1), 'DR1TMAGN': ('magnesium_mg', 1),
    'DR1TSODI': ('sodium_mg', 1), 'DR1TPOTA': ('potassium_mg', 1),
    'DR1TIRON': ('iron_mg', 1), 'DR1TCOPP': ('copper_mg', 1),
    'DR1TSELE': ('selenium_ug', 1), 'DR1TZINC': ('zinc_mg', 1),
    'DR1TVB1': ('vitb1_mg', 1), 'DR1TVB2': ('vitb2_mg', 1),
    'DR1TNIAC': ('vitb3_mg', 1), 'DR1TVB6': ('vitb6_mg', 1),
    'DR1TVC': ('vitc_mg', 1),
    'DR1TFOLA': ('folate_ug', 1),
    'DR1TVK': ('vitk1_ug', 1),
}
USDA_FIELDS = dict(zip(NUTRIENT_CODES, (
    1008, 1003, 1005, 2000, 1079, 1004, 1258, 1292, 1293, 1253,
    1109, 1105, 1106, 1108, 1107, 1122, 1123, 1165, 1166, 1167,
    1175, 1177, 1186, 1187, 1190, 1178, 1162, 1185, 1087, 1091,
    1090, 1089, 1095, 1098, 1093, 1092, 1103, 1057, 1018,
)))


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) and value >= 0 else None
    except (ValueError, TypeError):
        return None


def gram_amount(row):
    """Subset of the authors' INDB.do conversion rules; unknown units stay null."""
    amount = number(row['amount'])
    unit, category = str(row['unit']).strip(), str(row['food_code'])[0]
    if amount is None:
        return None
    factor = None
    if unit == 'g': factor = 1
    elif unit == 'C': factor = 240
    elif unit == 'pinch': return .125
    elif unit == 'sprig' and category in 'GUCA': factor = 2
    elif unit == 'small' and category == 'G': factor = 2
    elif unit == 'ml' and category in 'GHIVXLTKE': factor = 1
    elif unit == 'tsp':
        factor = {'G': 2, 'H': 2, 'B': 2, 'U': 2, 'D': 2, 'A': 2.5,
                  'M': 5, 'V': 2, 'T': 4, 'X': 6, 'W': 6, 'I': 4,
                  'L': 5, 'C': .5, 'K': 5, 'E': 5}.get(category)
    elif unit == 'tbsp':
        factor = {'D': 7, 'G': 7, 'H': 7, 'B': 7, 'U': 7, 'F': 40,
                  'A': 7, 'T': 13.5, 'X': 20, 'W': 20, 'I': 12,
                  'Z': 15, 'L': 15, 'C': 5, 'K': 15, 'E': 15, 'V': 8}.get(category)
    return None if factor is None else amount * factor


def main():
    legacy = json.loads((ROOT / 'food_composition_ifct2017.json').read_text(encoding='utf-8'))
    requested = [item for item in legacy if item['food_id'].startswith('recipe_')]
    assert len(requested) == 44
    requested = requested + EXTRA_SINGLE_ITEMS
    recipes = pd.read_excel(SOURCE / 'recipes.xlsx')
    names = pd.read_excel(SOURCE / 'recipes_names.xlsx').set_index('recipe_code')
    servings = pd.read_excel(SOURCE / 'recipes_servingsize.xlsx').set_index('recipe_code')
    nutrients = pd.read_excel(SOURCE / 'INDB.xlsx').set_index('food_code')
    assert names.index.is_unique and servings.index.is_unique and nutrients.index.is_unique
    usdir = next((ROOT / 'food_sources/USDA_SR_Legacy').rglob('food.csv')).parent
    def csv_rows(name):
        with (usdir / name).open(encoding='utf-8-sig', newline='') as handle:
            yield from csv.DictReader(handle)
    usfoods = {int(row['fdc_id']): row for row in csv_rows('food.csv')}
    usvalues = {(int(row['fdc_id']), int(row['nutrient_id'])): row
                for row in csv_rows('food_nutrient.csv') if int(row['fdc_id']) in SINGLE_MATCHES.values()}
    catalog, registry = [], []
    for old in requested:
        fid = old['food_id']
        record = {'food_id': fid, 'original_name': old['name'], 'status': 'unmatched',
                  'reason': 'No sufficiently comparable source recipe with documented quantities selected'}
        if fid not in RECIPE_MATCHES and fid not in SINGLE_MATCHES:
            registry.append(record)
            continue
        weight = float(old['serving_weight_g'])
        item = {key: old[key] for key in ('food_id', 'category', 'serving_weight_g', 'selection_cost')}
        item.update({'catalog_version': 'sourced-cooked-v5-veg-nonveg',
                     'dietary_type': 'non_vegetarian' if fid in NON_VEG_FOOD_IDS else 'pure_vegetarian',
                     'serving_unit': f'1 standardized {weight:g} g portion',
                     'nutrients_per_serving': {code: None for code in NUTRIENT_CODES},
                     'nutrient_provenance': {}, 'ingredients': []})
        if fid in RECIPE_MATCHES:
            code = RECIPE_MATCHES[fid]
            source_name = str(names.loc[code, 'recipe_name']).strip()
            source_row = nutrients.loc[code]
            ingredient_rows = recipes[recipes.recipe_code == code]
            for excel_row, row in ingredient_rows.iterrows():
                item['ingredients'].append({
                    'source_excel_row': int(excel_row) + 2, 'food_code': str(row['food_code']),
                    'name': str(row['food_name']).strip(), 'source_amount': number(row['amount']),
                    'source_unit': str(row['unit']), 'converted_grams': gram_amount(row),
                    'gram_conversion_source': 'INDB.do ingredient/category rules; not a measured density',
                })
            if ingredient_rows.empty or any(i['converted_grams'] is None for i in item['ingredients']):
                record.update(status='blocked_units', reason='Ingredient unit conversion needs review')
                registry.append(record)
                continue
            total_g = sum(i['converted_grams'] for i in item['ingredients'])
            portion_count = number(servings.loc[code, 'no_of_servings'])
            pieces_per_portion = number(servings.loc[code, 'size_of_servings'])
            if not total_g or not portion_count:
                record.update(status='blocked_yield', reason='Missing source batch mass or serving count')
                registry.append(record)
                continue
            # This is the authors' input-mass calculation basis, not a measured
            # evaporated cooked yield. Preserve that distinction explicitly.
            item['name'] = source_name
            item['data_source'] = f'INDB published recipe {code}; standard variant of {old["name"]}'
            item['recipe_source'] = {'recipe_code': code, 'repository': URL,
                                     'original_project_name': old['name'], 'match_type': 'documented_standard_variant',
                                     'source_batch_input_mass_g': total_g,
                                     'source_no_of_servings': portion_count,
                                     'source_size_of_servings': number(servings.loc[code, 'size_of_servings']),
                                     'source_servings_unit': str(servings.loc[code, 'servings_unit']),
                                     'measured_cooked_yield_g': None,
                                     'yield_basis': 'summed converted ingredient mass, including listed water; no evaporation correction',
                                     'nutrient_basis': 'published INDB per 100 g, without nutrient-retention adjustment'}
            for ncode, (column, factor) in INDB_FIELDS.items():
                value = number(source_row[column])
                if value is not None:
                    item['nutrients_per_serving'][ncode] = value * factor * weight / 100
                    item['nutrient_provenance'][ncode] = {'recipe_code': code, 'source_column': column,
                                                        'multiplier_per_100g_to_serving': factor * weight / 100}
                    if ncode in {'DR1TFOLA', 'DR1TVK'}:
                        item['nutrient_provenance'][ncode]['definition_reference'] = 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DR1TOT_J.htm'
            item['additional_source_nutrients'] = {'free_sugar_g_per_serving': float(source_row['freesugar_g']) * weight / 100}
            # Reconcile the published portion basis against ingredient-mass / count.
            expected_mass = total_g / portion_count / (pieces_per_portion or 1)
            reported_mass = float(source_row['unit_serving_energy_kcal']) / float(source_row['energy_kcal']) * 100
            item['recipe_source']['source_mass_reconciliation_error_g'] = reported_mass - expected_mass
            if abs(reported_mass - expected_mass) > max(1, .02 * expected_mass):
                record.update(status='blocked_mass_reconciliation', reason=f'Source mass mismatch: {reported_mass:g} vs {expected_mass:g} g')
                registry.append(record)
                continue
            for ingredient in item['ingredients']:
                ingredient['grams_per_standardized_portion'] = ingredient['converted_grams'] * weight / total_g
        else:
            fdc = SINGLE_MATCHES[fid]
            item['name'] = usfoods[fdc]['description']
            item['data_source'] = f'USDA SR Legacy {fdc}; single ready-to-eat food proxy'
            item['recipe_source'] = {'fdc_id': fdc, 'source_url': f'https://fdc.nal.usda.gov/food-details/{fdc}/nutrients',
                                     'match_type': 'borrowed_generic_single_food', 'original_project_name': old['name']}
            item['ingredients'] = [{'name': item['name'], 'grams_per_serving': weight,
                                     'source_food_id': fdc, 'preparation_basis': 'as described in source'}]
            for ncode, nutrient_id in USDA_FIELDS.items():
                row = usvalues.get((fdc, nutrient_id))
                value = number(row['amount']) if row else None
                if value is not None:
                    item['nutrients_per_serving'][ncode] = value * weight / 100
                    item['nutrient_provenance'][ncode] = {'fdc_id': fdc, 'nutrient_id': nutrient_id,
                                                        'source_row_id': row['id'], 'derivation_id': row['derivation_id']}
        item['unknown_nutrients'] = [key for key, value in item['nutrients_per_serving'].items() if value is None]
        record.update(status='included', name=item['name'], unknown_nutrients=item['unknown_nutrients'])
        registry.append(record)
        catalog.append(item)

    # Enrich the 8 missing active nutrient fields for INDB recipes from ingredient quantities,
    # preserving null wherever any ingredient lacks a documented compatible value.
    from enrich_recipe_active_gaps import enrich_indb_recipes
    gap_report = enrich_indb_recipes(catalog)
    by_id = {item['food_id']: item for item in catalog}
    for rec in registry:
        if rec['status'] == 'included' and rec['food_id'] in by_id:
            rec['unknown_nutrients'] = by_id[rec['food_id']]['unknown_nutrients']

    manifest = {'repository': URL, 'sha256': {file.name: hashlib.sha256(file.read_bytes()).hexdigest()
                 for file in SOURCE.iterdir() if file.suffix in {'.xlsx', '.do'}},
                'requested_dishes': len(requested), 'included': len(catalog),
                'excluded': len(requested) - len(catalog), 'dishes': registry}
    assert len({item['food_id'] for item in catalog}) == len(catalog)
    assert all(set(item['nutrients_per_serving']) == set(NUTRIENT_CODES) for item in catalog)
    (ROOT / 'food_composition_cooked_sourced.json').write_text(json.dumps(catalog, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    (SOURCE / 'cooked_catalog_coverage.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    (SOURCE / 'ingredient_8_gap_mapping_report.json').write_text(json.dumps(gap_report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(f'Requested {len(requested)}; included {len(catalog)}; excluded {len(requested)-len(catalog)}')
    print('Ingredient 8-gap fills (out of 20 INDB recipes):', gap_report['filled_counts_by_nutrient'])
    print('INDB recipes complete across all 8 active gaps:', gap_report['recipes_complete_all_8'])
    for record in registry:
        if record['status'] != 'included': print(record['food_id'], record['status'], record['reason'])


if __name__ == '__main__':
    main()
