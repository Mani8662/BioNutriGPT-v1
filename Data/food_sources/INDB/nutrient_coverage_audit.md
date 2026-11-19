# Missing nutrient audit

49 candidates: 20 INDB recipe variants and nine USDA single-food proxies.
Full coverage across all 49 candidates is now 26 of 39 fields (13 fields remain incomplete for 1–20 INDB recipes after ingredient-level enrichment from USDA SR Legacy, IFCT 2017, and UK CoFID 2021).
This is source-table coverage, not laboratory or clinical validation. Exact missing food IDs and names are in the JSON audit.

| Field | Meaning | Missing foods | Reason / action |
| --- | --- | ---: | --- |
| DR1TSUGR | Total sugars | 20/49 | INDB supplies free sugars, not total sugars. Obtain total sugars from each ingredient or a compatible prepared-food analysis; do not equate free with total sugar. |
| DR1TATOC | Vitamin E, alpha-tocopherol | 2/49 | Generic vitamin E does not document the alpha-tocopherol definition across source foods. Resolve source-specific vitamin E conventions or calculate alpha-tocopherol from compatible ingredient records. |
| DR1TRET | Retinol | 1/49 | INDB generic vitamin A is not separate retinol. Obtain preformed retinol separately; relevant to interpretation of vitamin A upper bounds. |
| DR1TACAR | Alpha-carotene | 20/49 | INDB reports total carotenoids, not this individual species. Use matching ingredient alpha-carotene values; otherwise leave as an unverified Stage 1 input component. |
| DR1TBCAR | Beta-carotene | 20/49 | Total carotenoids cannot identify beta-carotene. Use separate beta-carotene measurements with preparation basis. |
| DR1TLYCO | Lycopene | 20/49 | Total carotenoids cannot identify lycopene. Use separate lycopene measurements; do not invent an adequacy minimum. |
| DR1TLZ | Lutein plus zeaxanthin | 20/49 | Total carotenoids cannot identify this sum. Use separate lutein and zeaxanthin or their documented sum. |
| DR1TFA | Folic acid | 5/49 | Source folate is not separately quantified added/synthetic folic acid. Determine fortification and folic-acid content; absence of a fortification record is not a measured zero. |
| DR1TFF | Food folate | 20/49 | Total folate does not separate natural food folate from folic acid. Obtain the two components before deriving food folate or DFE. |
| DR1TFDFE | Dietary folate equivalents | 5/49 | DFE cannot be computed from total folate alone. Use compatible food folate and folic acid with the correct conversion basis; review DFE-based target applicability. |
| DR1TVB12 | Vitamin B12 | 2/49 | INDB recipe table has no B12 field. Map all recipe ingredients to traceable B12 records, including paneer and spices, and document yield/retention before summing. |
| DR1TCAFF | Caffeine | 19/49 | INDB recipe table has no caffeine field. Use beverage/exposure records and documented ingredients. The meal basket alone is not total daily exposure. |
| DR1TALCO | Alcohol | 8/49 | INDB recipe table has no alcohol field. Collect beverage/exposure information and account for cooking losses where relevant; do not infer patient intake from omitted drinks. |

## Verification scope

The current catalog separates 26 fully covered measurements from 13 incomplete fields.
No feature is removed from Stage 1. No clinical constraint is silently dropped. The existing Stage 2
bounds need separate applicability review; a populated field is not proof that
its current threshold is the appropriate clinical reference.

Total folate is not DFE, generic vitamin A is not RAE, and free sugar is not total
sugar. Caffeine/alcohol intake includes drinks outside the selected food basket.

Definitions: https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/DR1TOT_J.htm
INDB source: https://github.com/lindsayjaacks/Indian-Nutrient-Databank-INDB-/tree/fbdb62bec519c6e2468582a799fb47e194b42558

Reproduce: python build_sourced_cooked_catalog.py; python audit_nutrient_coverage.py
Feasibility: python audit_sourced_cooked_catalog.py
