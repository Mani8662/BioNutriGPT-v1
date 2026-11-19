# Matched meal-planning benchmark: MILP, LP relaxation, and real Qwen

## Design

- 20 balanced synthetic profiles (five each: diabetes, obesity, CVD/hypertension, multimorbid).
- Same 29 sourced entries (20 INDB recipe variants + 9 USDA single-food proxies; 15 unmatched legacy entries excluded), standardized gram portions, seven covered nutrient bounds, plate rules, estimated serving prices, and objective for every method.
- Real local model: `Qwen/Qwen2.5-3B-Instruct`, 4-bit NF4; model load time 52.9 s (one-time). Qwen plans are scored as generated, without repair.
- The LP relaxes integer restrictions on servings and dish activation. A second continuous-serving comparator keeps dish activations binary and relaxes only serving quantities, isolating the effect of whole portions.
- Shared score = normalized weighted nutrient-bound slack + (150/energy target)×absolute calorie deviation + 0.10 per active dish + 0.01×estimated basket cost in INR. Lower is better.
- Household-practicality proxy = integer-serving plan rate and fractional-serving distance. This is not a dietitian or patient acceptability rating.

## Results

| Method | Plate-rule pass | All 7 nutrient bounds met | All-integer plans | Mean fractional foods/plan | Mean cost (₹) | Mean realized score | Mean solver score | Median time/plan |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fully relaxed LP | 45.0% | 35.0% | 0.0% | 5.05 | 192.08 | 5.271 | 5.015 | 6.8 ms |
| Continuous servings + binary dish selection | 100.0% | 30.0% | 0.0% | 3.80 | 194.12 | 5.369 | 5.369 | 546.6 ms |
| Discrete MILP | 100.0% | 15.0% | 100.0% | 0.00 | 203.65 | 7.427 | 7.427 | 1921.5 ms |
| Real Qwen 2.5 3B | 0.0% | 0.0% | 100.0% | 0.00 | 74.50 | 715.932 | 715.932 | 13924.6 ms |

## What this says about whole portions

- Against the continuous-serving model with the same binary dish choices, the integer MILP costs an estimated ₹9.53 more per day on average (4.9%). Median solve time is 1921.5 ms versus 546.6 ms (3.5×).
- Whole portions improve the structural proxy: 100.0% of MILP plans are integer-served by construction; 0.0% of continuous-serving plans are all-integer, averaging 3.80 fractional foods. Plate-rule pass rates are 100.0% for MILP and 100.0% for continuous servings.
- The fully relaxed LP is a speed lower bound but relaxes dish activation too; it met plate rules in 45.0% of actual baskets. Its fractional activation variables can make it appear more feasible than a cookable plan.
- The solver-score column is the actual linear optimization objective. For the LP it uses relaxed fractional dish-activation variables; realized-plan score counts each selected dish as active. The MILP's solver and realized scores should coincide up to numerical tolerance.
- Qwen met the plate rules in 0.0% of cases and all seven nutrient bounds in 0.0%. Its mean estimated cost was ₹74.50; median generation time was 13.9 s, excluding the one-time model load.
- Price values are illustrative assumptions carried over or extended from the earlier benchmark, not verified current local market prices. Cost differences are therefore provisional.
- Nutrient and recipe values are from `food_composition_cooked_sourced.json` (20 INDB recipe variants + 9 USDA single-food proxies; 23 fully covered nutrients, 16 unsupported nutrients marked unknown). This 7-nutrient matched benchmark covers only a subset of the 23 covered fields; see `Data/food_sources/INDB/cooked_catalog_feasibility.json` for full 39-field missingness and the MUFA (`DR1TMFAT`) feasibility conflict. Results are a computational benchmark, not clinical or taste validation.
- This study measures portion integrality as a practical proxy. It cannot establish that patients or dietitians prefer the integer plans; that requires blinded dietitian/patient review.

Per-case plans and raw Qwen generations are in the JSON report; the CSV has the aggregated case-level scores.
