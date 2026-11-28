"""Read-only feasibility audit of the saved Stage 3 patient dossier."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, linprog, milp

import stage3_milp_solver as s3


def main() -> None:
    dossier = json.loads(Path("bionutrigpt_pipeline_dossier.json").read_text(encoding="utf-8"))
    stage2 = dossier["stage2_counterfactual_prescription"]
    # This historical audit targets the repaired 586-row ingredient catalog.
    # The production cooked-only catalog has its own audit script.
    foods = s3.load_food_database(str(Path('food_composition_ifct2017.json')))
    print("B12 known foods", sum('DR1TVB12' not in food.unknown_nutrients for food in foods),
          "B12 unknown foods", sum('DR1TVB12' in food.unknown_nutrients for food in foods), flush=True)
    print("Unknown nutrients use known subtotals; arithmetic feasibility does not establish adequacy.", flush=True)
    stage = s3._parse_stage2_input(stage2)
    c, integrality, bounds, constraints, info = s3._make_sparse_constraints(foods, stage, gamma=0.10)
    idx = info["indices"]
    strict_ub = bounds.ub.copy()
    strict_ub[idx["e_plus"]:idx["calorie_over"]] = 0.0
    strict_bounds = Bounds(bounds.lb, strict_ub)
    A = constraints.A.tocsr()
    lo, hi = constraints.lb, constraints.ub
    upper = np.isfinite(hi)
    lower = np.isfinite(lo)
    lp = linprog(c, A_ub=np.vstack((A[upper].toarray(), -A[lower].toarray())),
                 b_ub=np.concatenate((hi[upper], -lo[lower])),
                 bounds=list(zip(strict_bounds.lb, strict_bounds.ub)), method="highs")
    print("foods", len(foods), "strict_lp_status", lp.status, lp.message, flush=True)
    if lp.status == 2:
        # Find which individual nutrient bounds are necessary for the LP conflict.
        for k, code in enumerate(s3.NUTRIENT_CODES):
            ub = strict_ub.copy()
            ub[idx["e_plus"] + k] = np.inf
            ub[idx["e_minus"] + k] = np.inf
            relaxed = milp(c=np.zeros_like(c), integrality=np.zeros_like(integrality),
                           bounds=Bounds(bounds.lb, ub), constraints=constraints,
                           options={"time_limit": 2.0})
            if relaxed.x is not None:
                print("LP becomes feasible when relaxed:", code, flush=True)
                F = info["food_matrix"]
                objective = np.zeros_like(c)
                objective[:len(foods)] = -F[k]
                optimum = milp(c=objective, integrality=np.zeros_like(integrality),
                               bounds=Bounds(bounds.lb, ub), constraints=constraints,
                               options={"time_limit": 5.0})
                if optimum.x is not None:
                    print("maximum", code, -optimum.fun, "required_minimum", stage.lower_bounds[code], flush=True)
                    selected = sorted(((float(optimum.x[i]), food.name, float(F[k, i]))
                                       for i, food in enumerate(foods) if optimum.x[i] > 1e-5), reverse=True)
                    print("LP selected foods", selected[:15], flush=True)
                integer_result = milp(c=c, integrality=integrality,
                                      bounds=Bounds(bounds.lb, ub), constraints=constraints,
                                      options={"time_limit": 30.0, "mip_rel_gap": 0.01})
                print("MILP with", code, "relaxed:", integer_result.status,
                      integer_result.message, "incumbent", integer_result.x is not None,
                      flush=True)
                if integer_result.x is not None:
                    q = np.rint(integer_result.x[:len(foods)]).astype(int)
                    delivered = info["food_matrix"] @ q
                    print("integer delivered B12", delivered[s3.NUTRIENT_INDEX["DR1TVB12"]],
                          "sodium", delivered[s3.NUTRIENT_INDEX["DR1TSODI"]],
                          "potassium", delivered[s3.NUTRIENT_INDEX["DR1TPOTA"]], flush=True)
                    print("integer basket", [(foods[i].name, int(q[i])) for i in range(len(foods)) if q[i]], flush=True)
        return
    strict = milp(c=c, integrality=integrality, bounds=strict_bounds,
                  constraints=constraints, options={"time_limit": 30.0, "mip_rel_gap": 0.01})
    print("strict_milp_status", strict.status, strict.message,
          "incumbent", strict.x is not None, "gap", getattr(strict, "mip_gap", None), flush=True)
    report = {"strict_lp_status": int(lp.status), "strict_milp_status": int(strict.status),
              "integer_incumbent": strict.x is not None, "clinical_adequacy_proven": False,
              "note": "B12 unknown contributions excluded from known subtotals; source and recipe gaps remain."}
    if strict.x is not None:
        q = np.rint(strict.x[:len(foods)]).astype(int)
        report['basket'] = [{"food_id": food.food_id, "quantity": int(qi),
                             "unknown_nutrients": list(food.unknown_nutrients)}
                            for food, qi in zip(foods, q) if qi]
        report['b12_known_subtotal_ug'] = float(info['food_matrix'][s3.NUTRIENT_INDEX['DR1TVB12']] @ q)
        print("B12 known subtotal", report['b12_known_subtotal_ug'], flush=True)
    Path('Data/food_sources/USDA_SR_Legacy/post_repair_feasibility.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == "__main__":
    main()
