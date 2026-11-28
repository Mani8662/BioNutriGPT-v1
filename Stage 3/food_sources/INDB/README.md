# Sourced cooked-food catalog repair

Source repository: https://github.com/lindsayjaacks/Indian-Nutrient-Databank-INDB-/tree/fbdb62bec519c6e2468582a799fb47e194b42558

Publication: https://pmc.ncbi.nlm.nih.gov/articles/PMC11277795/

## What changed

`build_sourced_cooked_catalog.py` reviews all 44 legacy recipe IDs. The production
catalog `food_composition_cooked_sourced.json` includes 29 records: 20 documented
INDB standard recipe variants and 9 USDA single-food records (fruit, cooked brown
rice, and plain yogurt proxies). The other 15 are recorded as unmatched in
`cooked_catalog_coverage.json` and excluded, rather than being assigned invented
ingredients. Original IDs are retained for included records, but the displayed
names identify their actual source variants. Do not interpret a retained ID as
proof that its old formulation, nutrient values, or serving name is unchanged.

Stage 3 now prefers this catalog; raw IFCT ingredients cannot enter production
plans through the default mixed catalog. Explicit-path historical audits remain
available. The 586-row file and old benchmark outputs are preserved as legacy
artifacts. Those benchmark results do not describe this new catalog.

## Ingredients, mass, and nutrients

Each INDB record carries its recipe code, source ingredient Excel row, ingredient
food code/name, quantity/unit, and converted gram weight. Conversion rules are
copied from the authors' INDB.do for the units used in included recipes. Source
modeled mass per piece reconciles with published per-piece nutrient values within
1 g or 2%. For chapati, dosa and idli, this requires both the number of portions
and the number of pieces per portion; omitting either creates a 2–4-fold error.

Gram portions use the prior project's serving weights, but labels are changed to
explicit standardized gram portions. These are not validated katori volumes or
piece sizes. Nutrients per portion equal the source's published nutrient value
per 100 g multiplied by grams per portion / 100. INDB fatty acids in mg are
converted to g. Ingredient grams per standardized portion are scaled from the
source ingredient-mass total, including specified water.

The source's mass basis is summed input quantities. Actual evaporated cooked
yields have not been measured for these dishes. `measured_cooked_yield_g` is null.
No retention correction or invented yield factor is applied. The original NIN
raw nutrient workbook needed to reproduce all of the authors' ingredient-level
nutrient arithmetic is not in this repository; this build uses the authors'
published calculated recipe table and independently checks its mass basis. It
does not claim an independent laboratory or full ingredient-nutrient replication.

INDB free sugar is kept as an additional source field and is not copied into the
NHANES total-sugar field. Vitamin A, folate, vitamin K, and other fields whose
definitions do not exactly match the pipeline remain unknown. B12 is absent from
the INDB recipe table, so composite recipe B12 stays unknown. For USDA single
foods, source records and nutrient-row IDs are retained for each mapped nutrient.
No disease-score feedback uses these proposed meals as observed patient intake.

`cooked_catalog_coverage.json` pins source file SHA256 hashes and lists all 44
statuses. The builder fails on duplicate/missing source assumptions rather than
silently filling a nutrient gap. Downloads are the authors' original workbooks;
they were read without modification.

## Feasibility result for the saved example

Reproduce with:

```
python build_sourced_cooked_catalog.py
python audit_sourced_cooked_catalog.py
```

25 of 39 nutrient fields are populated across all 29 candidates; 14 fields lack
complete coverage. Under adult-food-rule-review-v3, the diagnostic makes the 24
active, fully covered intervals hard. Total folate is covered but reporting only.
The historical model was infeasible, including its continuous relaxation. With
the other covered bounds and plate rules retained, achievable MUFA was
7.297–17.593 g, below the historical 21.8 g minimum. The cause was an unsupported
independent MUFA 10%-energy lower bound. FAO Food and Nutrition Paper 91, printed
pp. 15 and 58, describes MUFA by difference within the overall fat pattern:
https://www.fao.org/4/i1953e/i1953e00.pdf

Stage 2 now uses `mufa-component-envelope-v2`: no independent adequacy minimum;
MUFA is nonnegative and capped at the existing total-fat ceiling. These are
component bookkeeping bounds, not a recommended intake or MUFA UL. The audit
applies this change explicitly to a copy of the saved envelope in memory. It
does not overwrite the saved historical model scores. New pipeline runs generate
the corrected rule and provenance. The resulting strict covered-field integer model
is feasible. Its candidate contains 13.506 g MUFA, and passes the calorie,
macronutrient, sodium-to-potassium, and fiber post-checks. A 30-second diagnostic
search does not establish success at the production 0.5-second budget.

The later full rule audit is recorded in `Saved Model/stage2_rule_audit.json` and
`.md`. It corrects definitions, selected age/sex references and scope-specific upper
limits, and labels remaining project assumptions. `reviewed_stage2_envelope.json`
is an audit-only fixture with prior certification explicitly revoked; historical
model scores and targets were not regenerated. See the rule audit for all sources.

The production solver now returns an explicit insufficient-data failure with
no accepted basket or schedule. Missing nutrients remain explicit. No complete
39-field adequacy or clinical-safety claim is supported. See
`cooked_catalog_feasibility.json` for exact outputs and solver statuses.

## Remaining work

Find suitable source formulations for the 15 excluded dishes, obtain compatible
missing nutrient data, validate cooking yields and household servings, and review
the provenance/applicability of Stage 2's nutrient bounds before changing them.
Adding foods such as oil as separately portioned ingredients would require an
explicit model design decision; this repair has not introduced them as dishes.


NUTRIENT COVERAGE AUDIT UPDATE
Two previously missing mappings were recovered using compatible definitions: INDB total folate to DR1TFOLA and vitamin K1 to DR1TVK. The active catalog version is sourced-cooked-v2-folate-k1. There are now 25 fully covered fields and 14 incomplete fields; each remaining gap affects the 20 INDB recipe records. Exact missing food IDs, reasons, and next actions are in Data/food_sources/INDB/nutrient_coverage_audit.json. The proposed verification scope retains Stage 1 features and separates covered measurements, seven priority gaps, five supporting research fields, and two exposure fields. It does not silently drop clinical constraints. The strict 25-field candidate passes its numeric intervals, while full 39-field adequacy remains unverified. RDA/DFE/retinol/sugar rule applicability still requires review.


STAGE 3 PRODUCTION ACCEPTANCE UPDATE
The canonical solver now accepts only candidates passing independent recalculation,
active nutrient completeness/interval checks, plate/portion rules and post-checks.
An invalid cheaper heuristic cannot replace a valid solver candidate. Failure returns
an empty basket and explicit data/infeasibility/timeout/validation status. Stage 4 and
chat acceptance revalidate the plan and cannot compile or accept a rejected candidate.
The real RTX 3050 sample passed Stage 2 checks and stopped at Stage 3 for missing active
data; Stage 4 was not invoked. Nine synthetic/real-data regression checks passed, including
chat revision success and failure. See Saved Model/stage3_production_repair.md and the
measured stage3_production_live_sample.json. Missing data remain unresolved; no rule
was silently waived and no real-food full-adequacy result is claimed.
