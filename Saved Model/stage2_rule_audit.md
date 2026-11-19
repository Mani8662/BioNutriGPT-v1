# Stage 2 nutrient-rule review

Ruleset: adult-food-rule-review-v3. All 39 fields reviewed. Stage 1 inputs are unchanged.
These are US reference interpretations and labeled project assumptions, not an Indian clinical protocol.
References describe generally healthy people. Pregnancy/lactation, renal disease, medicines,
supplements, smoking and individual clinical treatment require separate assessment.

## Mandatory versus advisory

Mandatory controls concern input validity, units, portion/plate structure and honest missing-data reporting.
The production solver requires active-interval compliance before acceptance. An RDA/AI is a planning reference,
not a proven one-day clinical safety boundary. The diagnostic may enforce active covered intervals
strictly, but that mathematical check does not establish clinical validity.

## Corrections

- Total sugar is reported separately; its former 25/50 g limit cannot be justified by WHO free-sugar guidance.
- Folate adequacy is assessed in DFE. Total folate and food folate no longer duplicate the DFE minimum.
- The folic-acid ceiling is kept on synthetic folic acid, not total food folate or DFE.
- Vitamin A RAE retains its adequacy reference; the preformed-vitamin-A ceiling is screened separately.
- Four individual carotenoids are reporting/research fields without invented clinical upper limits.
- Supplemental-only or unestablished upper limits are removed from ordinary food intervals.
- Selected age/sex B6, C, K, calcium, phosphorus, magnesium, iron, zinc and copper values are corrected.
- Risk-score-based macronutrient, sodium, potassium, cholesterol and exposure policies remain explicitly
  labeled project assumptions needing review. No universal safety claim is made for them.

A finite numerical search cap (1,000,000 in the field unit) is used where the vector optimizer
requires a finite domain. It is never a UL or recommended dose. Report-only fields have zero
nutrient penalty and no nutrient constraint in Stage 3; unknown data still prevent full verification.

| Field | Historical interval | Reviewed interval | Role | Evidence / applicability |
| --- | --- | --- | --- | --- |
| DR1TKCAL (kcal) | 1863.9–2060.1 | 1863.9–2060.1 | project_assumption | Project energy prescription with 5% tolerance; not an RDA or clinical safety proof. [Source](https://ods.od.nih.gov/HealthInformation/nutrientrecommendations.aspx) |
| DR1TPROT (g) | 59.2–148 | 59.2–148 | project_assumption | Project 0.8–2 g/kg interval plus sex-specific floor; upper endpoint is not a UL. Renal status is unavailable. [Source](https://ods.od.nih.gov/HealthInformation/nutrientrecommendations.aspx) |
| DR1TCARB (g) | 130–294.3 | 130–294.3 | project_assumption | 130 g adult reference plus project score-dependent upper fraction; no universally optimal diabetes carbohydrate percentage. [Source](https://ods.od.nih.gov/HealthInformation/nutrientrecommendations.aspx) |
| DR1TSUGR (g) | 0–50 | 0–294.3 | report_only | Report total sugar separately; WHO free-sugar limits cannot be applied directly to total sugar. [Source](https://www.who.int/publications/i/item/9789241549028) |
| DR1TFIBE (g) | 28–60 | 28–60 | project_assumption | Project fiber goal and computational upper bound; upper endpoint is not an established UL. [Source](https://ods.od.nih.gov/HealthInformation/nutrientrecommendations.aspx) |
| DR1TTFAT (g) | 43.6–76.3 | 43.6–76.3 | project_assumption | 20–35% energy adult reference interval; applicability to this patient and current guidance requires review. [Source](https://ods.od.nih.gov/HealthInformation/nutrientrecommendations.aspx) |
| DR1TSFAT (g) | 0–21.8 | 0–21.8 | project_assumption | Project risk-stratified upper fraction; lower value is not an individualized medical prescription. [Source](https://www.fao.org/4/i1953e/i1953e00.pdf) |
| DR1TMFAT (g) | 21.8–43.6 | 0–76.3 | reference_interval | Nonnegative component bounded by total-fat ceiling; not a separate MUFA RDA or UL. [Source](https://www.fao.org/4/i1953e/i1953e00.pdf) |
| DR1TPFAT (g) | 10.9–21.8 | 10.9–21.8 | project_assumption | Project total-PUFA interval; not a substitute for separate essential LA/ALA requirements. [Source](https://www.fao.org/4/i1953e/i1953e00.pdf) |
| DR1TCHOL (mg) | 0–300 | 0–300 | project_assumption | Legacy project cholesterol limits; 150/300 mg are not universally established ULs. [Source](https://ods.od.nih.gov/HealthInformation/nutrientrecommendations.aspx) |
| DR1TATOC (mg) | 15–1000 | 15–1e+06 | reference_interval | RDA reference; supplemental UL is not a food ceiling. [Source](https://ods.od.nih.gov/factsheets/VitaminE-HealthProfessional/) |
| DR1TRET (ug) | 0–3000 | 0–3000 | reference_interval | Preformed vitamin A UL screening; include supplements and ester forms for a full assessment. [Source](https://ods.od.nih.gov/factsheets/VitaminA-HealthProfessional/) |
| DR1TVARA (ug) | 900–3000 | 900–1e+06 | reference_interval | RAE adequacy reference; preformed-vitamin-A UL must not cap total RAE. [Source](https://ods.od.nih.gov/factsheets/VitaminA-HealthProfessional/) |
| DR1TACAR (ug) | 0–10000 | 0–1e+06 | report_only | Research/reporting field; no independent adequacy minimum or supported clinical ceiling is configured. [Source](https://ods.od.nih.gov/factsheets/VitaminA-HealthProfessional/) |
| DR1TBCAR (ug) | 0–20000 | 0–1e+06 | report_only | Research/reporting field; no independent adequacy minimum or supported clinical ceiling is configured. [Source](https://ods.od.nih.gov/factsheets/VitaminA-HealthProfessional/) |
| DR1TLYCO (ug) | 0–30000 | 0–1e+06 | report_only | Research/reporting field; no independent adequacy minimum or supported clinical ceiling is configured. [Source](https://ods.od.nih.gov/factsheets/VitaminA-HealthProfessional/) |
| DR1TLZ (ug) | 0–20000 | 0–1e+06 | report_only | Research/reporting field; no independent adequacy minimum or supported clinical ceiling is configured. [Source](https://ods.od.nih.gov/factsheets/VitaminA-HealthProfessional/) |
| DR1TVB1 (mg) | 1.2–50 | 1.2–1e+06 | reference_interval | RDA reference; no established UL. [Source](https://ods.od.nih.gov/factsheets/Thiamin-HealthProfessional/) |
| DR1TVB2 (mg) | 1.3–50 | 1.3–1e+06 | reference_interval | RDA reference; no established UL. [Source](https://ods.od.nih.gov/factsheets/Riboflavin-HealthProfessional/) |
| DR1TNIAC (mg) | 16–35 | 16–1e+06 | reference_interval | Food niacin is a proxy for NE; tryptophan contribution is unavailable. Synthetic/supplemental UL is not a whole-food ceiling. [Source](https://ods.od.nih.gov/factsheets/Niacin-HealthProfessional/) |
| DR1TVB6 (mg) | 1.3–100 | 1.3–100 | reference_interval | Age/sex US RDA and UL reference; this is not a universally agreed international ceiling. [Source](https://ods.od.nih.gov/factsheets/VitaminB6-HealthProfessional/) |
| DR1TFOLA (ug) | 400–1000 | 0–1e+06 | report_only | Total folate reporting only; 400 ug DFE is not a total-folate requirement. [Source](https://ods.od.nih.gov/factsheets/Folate-HealthProfessional/) |
| DR1TFA (ug) | 0–1000 | 0–1000 | reference_interval | Synthetic folic acid ceiling requires counting fortified foods and supplements; food basket alone is incomplete exposure. [Source](https://ods.od.nih.gov/factsheets/Folate-HealthProfessional/) |
| DR1TFF (ug) | 400–1000 | 0–1e+06 | report_only | Food folate reporting only; no independent duplicate adequacy target or natural-food UL. [Source](https://ods.od.nih.gov/factsheets/Folate-HealthProfessional/) |
| DR1TFDFE (ug) | 400–1000 | 400–1e+06 | reference_interval | DFE adequacy reference; the folic-acid UL is expressed in ug folic acid, not DFE. [Source](https://ods.od.nih.gov/factsheets/Folate-HealthProfessional/) |
| DR1TVB12 (ug) | 2.4–100 | 2.4–1e+06 | reference_interval | RDA reference; no established UL; missing data remain a priority gap. [Source](https://ods.od.nih.gov/factsheets/VitaminB12-HealthProfessional/) |
| DR1TVC (mg) | 90–2000 | 90–2000 | reference_interval | Age/sex reference; smoking-related additional needs are not implemented. [Source](https://ods.od.nih.gov/factsheets/VitaminC-HealthProfessional/) |
| DR1TVK (ug) | 120–1000 | 120–1e+06 | reference_interval | K1 used as a proxy for the vitamin K AI, not an RDA; no established UL; vitamin K antagonist therapy needs review. [Source](https://ods.od.nih.gov/factsheets/VitaminK-HealthProfessional/) |
| DR1TCALC (mg) | 1000–2500 | 1000–2500 | reference_interval | Age/sex RDA and UL reference; include other foods and supplements. [Source](https://ods.od.nih.gov/factsheets/Calcium-HealthProfessional/) |
| DR1TPHOS (mg) | 700–4000 | 700–4000 | reference_interval | Age-dependent RDA and UL reference; renal disease requires separate assessment. [Source](https://ods.od.nih.gov/factsheets/Phosphorus-HealthProfessional/) |
| DR1TMAGN (mg) | 400–700 | 420–1e+06 | reference_interval | Age/sex RDA reference; supplemental magnesium UL is not a food magnesium ceiling. [Source](https://ods.od.nih.gov/factsheets/Magnesium-HealthProfessional/) |
| DR1TIRON (mg) | 8–45 | 8–45 | reference_interval | Age/sex RDA and UL reference; menstrual status and vegetarian bioavailability need review. [Source](https://ods.od.nih.gov/factsheets/Iron-HealthProfessional/) |
| DR1TZINC (mg) | 11–40 | 11–40 | reference_interval | Age/sex reference; bioavailability and supplements may alter interpretation. [Source](https://ods.od.nih.gov/factsheets/Zinc-HealthProfessional/) |
| DR1TCOPP (mg) | 0.9–10 | 0.9–10 | reference_interval | Age-dependent RDA and UL reference. [Source](https://ods.od.nih.gov/factsheets/Copper-HealthProfessional/) |
| DR1TSODI (mg) | 500–2300 | 500–2300 | project_assumption | Legacy project sodium interval; 500 mg floor and risk-score choice of ceiling require review. [Source](https://ods.od.nih.gov/HealthInformation/nutrientrecommendations.aspx) |
| DR1TPOTA (mg) | 2600–4700 | 2600–4700 | project_assumption | Legacy project interval; AI differs by sex. Screening-score-based potassium escalation requires clinical review. [Source](https://ods.od.nih.gov/factsheets/Potassium-HealthProfessional/) |
| DR1TSELE (ug) | 55–400 | 55–400 | reference_interval | US RDA and UL reference; other authorities can differ. [Source](https://ods.od.nih.gov/factsheets/Selenium-HealthProfessional/) |
| DR1TCAFF (mg) | 0–400 | 0–400 | project_assumption | 400 mg FDA general adult exposure reference; not universal safety, and drinks outside the plan are omitted. [Source](https://www.fda.gov/consumers/consumer-updates/spilling-beans-how-much-caffeine-too-much) |
| DR1TALCO (g) | 0–14 | 0–14 | project_assumption | Legacy project exposure restriction, not a safe drinking allowance or recommendation to drink. [Source](https://ods.od.nih.gov/HealthInformation/nutrientrecommendations.aspx) |

## Reproduction

Run `python audit_stage2_rules.py`, then `python audit_sourced_cooked_catalog.py`.
The saved envelope is an audit fixture: only rules were recomputed. Historical model scores and
targets were not recalculated, and its prior Stage 2 certification is explicitly revoked.
