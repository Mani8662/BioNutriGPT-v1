"""Reproducible B12-only enrichment; unmatched values remain null, never zero.

Cross-country generic-food matches are borrowed estimates, not Indian assays.
Composite recipes without an ingredient specification deliberately stay unknown.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'Data/food_sources/USDA_SR_Legacy'
URL = 'https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_sr_legacy_food_csv_2018-04.zip'
SHA256 = 'b80817294b8850530aaedf2e515c02593b1824f763a0ff356e5c2081643e6fd0'

# Explicit, reviewed food/preparation matches. Generic US records are used as
# proxies for local varieties and are labeled accordingly in every record.
MATCHES = {
    'ifct_l002': 172217,  # cow milk, whole, unfortified
    'ifct_m001': 171287,  # whole chicken egg, raw
    'ifct_m004': 173424,  # whole chicken egg, hard-boiled
    'ifct_n005': 171060,  # chicken liver, raw
    'ifct_o021': 172531,  # sheep/lamb liver, raw
    'ifct_o032': 169451,  # beef liver, raw
    'ifct_o042': 172534,  # calf/veal liver, raw
    'ifct_o053': 167862,  # pork liver, raw
    'ifct_c033': 168462,  # spinach, raw
    'ifct_e028': 173044, 'ifct_e029': 173044,
    'ifct_e049': 169926, 'ifct_e047': 169097,
    'ifct_e001': 171688, 'ifct_e002': 171688,
    'ifct_e003': 171688, 'ifct_e004': 171688,
    'ifct_e009': 173944, 'ifct_e010': 173944,
    'ifct_e011': 173944, 'ifct_e012': 173944,
    'recipe_fresh_guava': 173044, 'recipe_fresh_papaya': 169926,
    'recipe_fresh_apple': 171688, 'recipe_fresh_orange': 169097,
    'recipe_fresh_banana': 173944, 'recipe_pomegranate_arils': 169134,
    'recipe_brown_rice_cooked': 169704, 'recipe_steamed_rice': 168878,
    # Plain yogurt proxy only; no substitution of cottage cheese for paneer.
    'recipe_skim_curd': 170887, 'recipe_lowfat_curd': 170886,
}


def enrich(items: list[dict]) -> dict:
    archive = SOURCE / 'source.zip'
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
        raise ValueError('USDA archive checksum differs from pinned source.')
    directory = next(SOURCE.rglob('food.csv')).parent
    def rows(name):
        with (directory / name).open(encoding='utf-8-sig', newline='') as handle:
            yield from csv.DictReader(handle)
    foods = {int(row['fdc_id']): row for row in rows('food.csv')}
    nutrients = {int(row['id']): row for row in rows('nutrient.csv')}
    b12 = [key for key, row in nutrients.items() if row['name'] == 'Vitamin B-12']
    if len(b12) != 1 or nutrients[b12[0]]['unit_name'].upper() not in {'UG', 'MCG'}:
        raise ValueError('B12 identity or unit is unexpected.')
    values = {int(row['fdc_id']): row for row in rows('food_nutrient.csv')
              if int(row['nutrient_id']) == b12[0]}
    report = {'source_url': URL, 'source_sha256': SHA256, 'nutrient_id': b12[0],
              'matched': [], 'unknown': [], 'scope': 'B12 only; other nutrient estimates are unchanged'}
    for item in items:
        fid = item['food_id']
        fdc_id = MATCHES.get(fid)
        record = values.get(fdc_id)
        item['nutrients_per_serving']['DR1TVB12'] = None
        provenance = {'status': 'unknown', 'reason': 'No reviewed food match or documented recipe ingredient quantities'}
        if record is not None and record['amount'] != '':
            per100 = float(record['amount'])
            value = per100 * float(item['serving_weight_g']) / 100.0
            item['nutrients_per_serving']['DR1TVB12'] = round(value, 8)
            provenance = {'status': 'borrowed_generic_food', 'fdc_id': fdc_id,
                          'source_description': foods[fdc_id]['description'],
                          'source_url': f'https://fdc.nal.usda.gov/food-details/{fdc_id}/nutrients',
                          'source_archive_sha256': SHA256, 'nutrient_id': b12[0],
                          'source_food_nutrient_id': record['id'],
                          'source_derivation_id': record['derivation_id'],
                          'source_data_points': record['data_points'],
                          'ug_per_100g': per100, 'serving_weight_g': item['serving_weight_g'],
                          'method': 'USDA proxy per 100 g multiplied by catalog serving weight / 100; not an Indian assay'}
            report['matched'].append({'food_id': fid, 'name': item['name'], **provenance,
                                      'ug_per_serving': item['nutrients_per_serving']['DR1TVB12']})
        else:
            report['unknown'].append({'food_id': fid, 'name': item['name']})
        item.setdefault('nutrient_provenance', {})['DR1TVB12'] = provenance
    return report


def main():
    target = ROOT / 'food_composition_ifct2017.json'
    backup = SOURCE / 'before_b12_repair'
    backup.mkdir(exist_ok=True)
    for name in ('food_composition_ifct2017.json', 'food_composition_ifct2017.csv'):
        if not (backup / name).exists():
            shutil.copy2(ROOT / name, backup / name)
    items = json.loads(target.read_text(encoding='utf-8'))
    report = enrich(items)
    target.write_text(json.dumps(items, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    flat = []
    for item in items:
        row = {key: value for key, value in item.items() if key not in ('nutrients_per_serving', 'nutrient_provenance')}
        row.update(item['nutrients_per_serving'])
        row['b12_provenance_json'] = json.dumps(item['nutrient_provenance']['DR1TVB12'], ensure_ascii=False)
        flat.append(row)
    with (ROOT / 'food_composition_ifct2017.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    (SOURCE / 'b12_mapping_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f"B12 matched: {len(report['matched'])}; unknown: {len(report['unknown'])}")
    for row in report['matched']:
        if row['ug_per_serving']:
            print(row['food_id'], row['ug_per_serving'], 'ug per catalog serving')


if __name__ == '__main__':
    main()
