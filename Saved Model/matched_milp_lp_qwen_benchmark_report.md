# Matched meal-planning benchmark: MILP, LP relaxation, and real Qwen

## Design

- 20 balanced synthetic profiles (five each: diabetes, obesity, CVD/hypertension, multimorbid).
- Same 44 cooked Indian recipes, household serving units, seven nutrient bounds, plate rules, estimated serving prices, and objective for every method.
- Real local model: `Qwen/Qwen2.5-3B-Instruct`, 4-bit NF4; model load time 30.1 s (one-time). Qwen plans are scored as generated, without repair.
- The LP relaxes integer restrictions on servings and dish activation. A second continuous-serving comparator keeps dish activations binary and relaxes only serving quantities, isolating the effect of whole portions.
- Shared score = normalized weighted nutrient-bound slack + (150/energy target)×absolute calorie deviation + 0.10 per active dish + 0.01×estimated basket cost in INR. Lower is better.
- Household-practicality proxy = integer-serving plan rate and fractional-serving distance. This is not a dietitian or patient acceptability rating.

## Results

| Method | Plate-rule pass | All 7 nutrient bounds met | All-integer plans | Mean fractional foods/plan | Mean cost (₹) | Mean realized score | Mean solver score | Median time/plan |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fully relaxed LP | 80.0% | 100.0% | 0.0% | 3.95 | 135.48 | 2.105 | 1.882 | 2.5 ms |
| Continuous servings + binary dish selection | 100.0% | 100.0% | 5.0% | 1.70 | 140.79 | 1.933 | 1.933 | 221.0 ms |
| Discrete MILP | 100.0% | 95.0% | 100.0% | 0.00 | 145.35 | 2.017 | 2.017 | 1882.1 ms |
| Real Qwen 2.5 3B | 0.0% | 0.0% | 100.0% | 0.00 | 55.25 | 691.911 | 691.911 | 6747.8 ms |

## What this says about whole portions

- Against the continuous-serving model with the same binary dish choices, the integer MILP costs an estimated ₹4.56 more per day on average (3.2%). Median solve time is 1882.1 ms versus 221.0 ms (8.5×).
- Whole portions improve the structural proxy: 100.0% of MILP plans are integer-served by construction; 5.0% of continuous-serving plans are all-integer, averaging 1.70 fractional foods. Plate-rule pass rates are 100.0% for MILP and 100.0% for continuous servings.
- The fully relaxed LP is a speed lower bound but relaxes dish activation too; it met plate rules in 80.0% of actual baskets. Its fractional activation variables can make it appear more feasible than a cookable plan.
- The solver-score column is the actual linear optimization objective. For the LP it uses relaxed fractional dish-activation variables; realized-plan score counts each selected dish as active. The MILP's solver and realized scores should coincide up to numerical tolerance.
- Qwen met the plate rules in 0.0% of cases and all seven nutrient bounds in 0.0%. Its mean estimated cost was ₹55.25; median generation time was 6.7 s, excluding the one-time model load.
- Price values are illustrative assumptions carried over or extended from the earlier benchmark, not verified current local market prices. Cost differences are therefore provisional.
- Nutrient and recipe values are from the project's constructed RecipeDB-style entries; the micro/secondary nutrient values are estimates. Results are a computational pilot, not clinical or taste validation.
- This study measures portion integrality as a practical proxy. It cannot establish that patients or dietitians prefer the integer plans; that requires blinded dietitian/patient review.

Per-case plans and raw Qwen generations are in the JSON report; the CSV has the aggregated case-level scores.
