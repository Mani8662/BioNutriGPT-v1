# B12 source repair

Source: USDA FoodData Central SR Legacy April 2018 CSV archive, downloaded from
https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_sr_legacy_food_csv_2018-04.zip

Archive SHA256: `b80817294b8850530aaedf2e515c02593b1824f763a0ff356e5c2081643e6fd0`.

`repair_b12_data.py` pins this checksum, identifies Vitamin B-12 by nutrient name
and verifies its microgram unit. Each reviewed match records the FDC food ID,
description, nutrient-row ID, derivation ID, data-point count, source value per
100 g, and catalog serving weight. `b12_mapping_report.json` lists all matches
and unresolved entries. These are borrowed generic US food estimates, not new
Indian laboratory measurements. No claim is made that every USDA record is a
direct assay. Original derivation metadata remains in the archive.

31 of the project's 586 records have explicit food/preparation matches. 555
remain unknown, represented as null in JSON and blank in CSV. Composite cooked
dishes without documented ingredient weights and yields are not inferred from
their names. Plain curd uses a disclosed plain-yogurt proxy; paneer is not
substituted with cottage cheese. Generic fruit matches do not establish local
cultivar measurements. Existing serving weights are used as recorded; this
repair does not validate their household realism or cooking basis.

The untouched original CSV and JSON are in `before_b12_repair/` for recovery.
All food IDs and non-B12 nutrient values were checked for preservation. The
builder invokes the same enrichment so placeholders are not regenerated.

Stage 3 computes known subtotals for optimization and tracks unknown nutrient
contributions separately. Any selected unknown B12 contribution prevents a
complete B12 verification result. This preserves the distinction between an
estimated zero and missing data. Mathematical feasibility is not clinical
validation. Other nutrient placeholders in the legacy recipe builder remain
outside this B12 repair and must be audited before research adequacy claims.

Reproduce:

```
python repair_b12_data.py
python audit_stage3_feasibility.py
```

The second command checks all nutrient bounds with zero slack under the model's
existing food and plate rules. Its post-repair report records numerical results
separately from clinical adequacy. It does not enforce the separate post-solve
sodium-to-potassium check or establish that every selected raw food is a
practical cooked dish.
