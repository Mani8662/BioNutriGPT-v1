"""Report exact missing-food coverage and a proposed measurement scope.

No clinical bounds or Stage 1 features are removed by this audit. Availability
does not establish that the existing bound is clinically applicable.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'Data/food_sources/INDB'
CDC = 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DR1TOT_J.htm'
INDB = 'https://github.com/lindsayjaacks/Indian-Nutrient-Databank-INDB-/tree/fbdb62bec519c6e2468582a799fb47e194b42558'

# Scientific/data classifications are proposed scope descriptions, not an
# instruction to silently weaken the current meal-plan acceptance checks.
GAPS = {
    'DR1TSUGR': ('Total sugars', 'g', 'priority_gap', 'INDB supplies free sugars, not total sugars.', 'Obtain total sugars from each ingredient or a compatible prepared-food analysis; do not equate free with total sugar.'),
    'DR1TATOC': ('Vitamin E, alpha-tocopherol', 'mg', 'priority_gap', 'Generic vitamin E does not document the alpha-tocopherol definition across source foods.', 'Resolve source-specific vitamin E conventions or calculate alpha-tocopherol from compatible ingredient records.'),
    'DR1TRET': ('Retinol', 'ug', 'priority_gap', 'INDB generic vitamin A is not separate retinol.', 'Obtain preformed retinol separately; relevant to interpretation of vitamin A upper bounds.'),
    'DR1TVARA': ('Vitamin A, RAE', 'ug RAE', 'priority_gap', 'Generic vitamin A units are not documented RAE.', 'Obtain retinol and provitamin A components with appropriate definitions; do not copy total carotenoids into RAE.'),
    'DR1TACAR': ('Alpha-carotene', 'ug', 'supporting_research', 'INDB reports total carotenoids, not this individual species.', 'Use matching ingredient alpha-carotene values; otherwise leave as an unverified Stage 1 input component.'),
    'DR1TBCAR': ('Beta-carotene', 'ug', 'supporting_research', 'Total carotenoids cannot identify beta-carotene.', 'Use separate beta-carotene measurements with preparation basis.'),
    'DR1TLYCO': ('Lycopene', 'ug', 'supporting_research', 'Total carotenoids cannot identify lycopene.', 'Use separate lycopene measurements; do not invent an adequacy minimum.'),
    'DR1TLZ': ('Lutein plus zeaxanthin', 'ug', 'supporting_research', 'Total carotenoids cannot identify this sum.', 'Use separate lutein and zeaxanthin or their documented sum.'),
    'DR1TFA': ('Folic acid', 'ug', 'supporting_research', 'Source folate is not separately quantified added/synthetic folic acid.', 'Determine fortification and folic-acid content; absence of a fortification record is not a measured zero.'),
    'DR1TFF': ('Food folate', 'ug', 'priority_gap', 'Total folate does not separate natural food folate from folic acid.', 'Obtain the two components before deriving food folate or DFE.'),
    'DR1TFDFE': ('Dietary folate equivalents', 'ug DFE', 'priority_gap', 'DFE cannot be computed from total folate alone.', 'Use compatible food folate and folic acid with the correct conversion basis; review DFE-based target applicability.'),
    'DR1TVB12': ('Vitamin B12', 'ug', 'priority_gap', 'INDB recipe table has no B12 field.', 'Map all recipe ingredients to traceable B12 records, including paneer and spices, and document yield/retention before summing.'),
    'DR1TCAFF': ('Caffeine', 'mg', 'exposure_reporting', 'INDB recipe table has no caffeine field.', 'Use beverage/exposure records and documented ingredients. The meal basket alone is not total daily exposure.'),
    'DR1TALCO': ('Alcohol', 'g', 'exposure_reporting', 'INDB recipe table has no alcohol field.', 'Collect beverage/exposure information and account for cooking losses where relevant; do not infer patient intake from omitted drinks.'),
}
RECOVERED = {
    'DR1TFOLA': ('Total folate', 'ug', 'folate_ug', 'Total folate matches total folate; no mapping to food folate or DFE.'),
    'DR1TVK': ('Vitamin K1 (phylloquinone)', 'ug', 'vitk1_ug', 'NHANES defines this field as phylloquinone, matching K1.'),
}


def main():
    path = ROOT / 'food_composition_cooked_sourced.json'
    foods = json.loads(path.read_text(encoding='utf-8'))
    codes = list(foods[0]['nutrients_per_serving'])
    fields = []
    for code in codes:
        missing = [{'food_id': food['food_id'], 'name': food['name'],
                    'recipe_code': food['recipe_source'].get('recipe_code')}
                   for food in foods if food['nutrients_per_serving'][code] is None]
        record = {'nutrient': code, 'known_food_count': len(foods) - len(missing),
                  'missing_food_count': len(missing), 'missing_foods': missing,
                  'fully_covered': not missing}
        if code in GAPS:
            name, unit, role, reason, action = GAPS[code]
            record.update(name=name, unit=unit, proposed_role=role, gap_reason=reason, next_action=action)
        elif code in RECOVERED:
            name, unit, column, rationale = RECOVERED[code]
            record.update(name=name, unit=unit, proposed_role='covered_measurement',
                          source_column=column, mapping_rationale=rationale,
                          definition_reference=CDC,
                          recovered_foods=[{'food_id': food['food_id'], 'name': food['name'],
                                            'recipe_code': food['recipe_source']['recipe_code']}
                                           for food in foods if 'recipe_code' in food['recipe_source']])
        else:
            record['proposed_role'] = 'covered_measurement'
        fields.append(record)
    covered = [row['nutrient'] for row in fields if row['fully_covered']]
    incomplete = [row['nutrient'] for row in fields if not row['fully_covered']]
    report = {'catalog': path.name, 'catalog_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'food_count': len(foods), 'field_count': len(fields),
              'original_incomplete_field_count': 16, 'recovered_fields': list(RECOVERED),
              'fully_covered_count': len(covered), 'incomplete_count': len(incomplete),
              'sources': [CDC, INDB], 'fields': fields,
              'clinical_validation_established': False,
              'scope_note': f'{len(covered)} fields have full catalog coverage across all {len(foods)} foods, not independently validated clinical targets. Stage 1 feature schema and current acceptance requirements remain in place.'}
    assert len(fields) == 39 and len(covered) + len(incomplete) == 39
    (OUT / 'nutrient_coverage_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    scope = {'status': 'proposed measurement scope; not a clinical prescribing protocol',
             'stage1_features_removed': [], 'catalog_fields_covered': covered,
             'priority_data_gaps': [c for c, v in GAPS.items() if v[2] == 'priority_gap' and c in incomplete],
             'supporting_research_fields': [c for c, v in GAPS.items() if v[2] == 'supporting_research' and c in incomplete],
             'separate_exposure_fields': [c for c, v in GAPS.items() if v[2] == 'exposure_reporting' and c in incomplete],
             'acceptance_rule': 'Report covered-field compliance separately. Unknown fields cannot be certified adequate or silently ignored.',
             'rule_review_required': ['Total folate vs DFE reference values', 'Retinol upper bound vs RAE',
                                      'Total sugar vs free/added sugar limits', 'Individual carotenoid bounds',
                                      'Age, sex, condition, supplements and exposure applicability']}
    (OUT / 'proposed_verification_scope.json').write_text(json.dumps(scope, indent=2), encoding='utf-8')
    lines = ['# Missing nutrient audit', '',
             f'{len(foods)} candidates: 20 INDB recipe variants and nine USDA single-food proxies.',
             f'Full coverage across all {len(foods)} candidates is now {len(covered)} of 39 fields ({len(incomplete)} fields remain incomplete for 1–20 INDB recipes after ingredient-level enrichment from USDA SR Legacy, IFCT 2017, and UK CoFID 2021).',
             'This is source-table coverage, not laboratory or clinical validation. Exact missing food IDs and names are in the JSON audit.', '',
             '| Field | Meaning | Missing foods | Reason / action |', '| --- | --- | ---: | --- |']
    for row in fields:
        if not row['fully_covered']:
            lines.append(f"| {row['nutrient']} | {row['name']} | {row['missing_food_count']}/{len(foods)} | {row['gap_reason']} {row['next_action']} |")
    lines += ['', '## Verification scope', '',
              f'The current catalog separates {len(covered)} fully covered measurements from {len(incomplete)} incomplete fields.',
              'No feature is removed from Stage 1. No clinical constraint is silently dropped. The existing Stage 2',
              'bounds need separate applicability review; a populated field is not proof that',
              'its current threshold is the appropriate clinical reference.', '',
              'Total folate is not DFE, generic vitamin A is not RAE, and free sugar is not total',
              'sugar. Caffeine/alcohol intake includes drinks outside the selected food basket.', '',
              f'Definitions: {CDC}', f'INDB source: {INDB}', '',
              'Reproduce: python build_sourced_cooked_catalog.py; python audit_nutrient_coverage.py',
              'Feasibility: python audit_sourced_cooked_catalog.py', '']
    (OUT / 'nutrient_coverage_audit.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'{len(covered)}/39 fields fully covered; {len(incomplete)} incomplete')


if __name__ == '__main__':
    main()
