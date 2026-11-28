import json
import math
import sys
import time
from pathlib import Path
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, vstack

LESS_DATA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LESS_DATA))

import stage3_milp_solver as s3
import stage2_stage3_adapter as adapter
import stage2_counterfactual_engine as s2
import stage1_deployment as deployment
import nutrient_rule_policy as nrp

# Load case 0
cases = json.loads((LESS_DATA / "Saved Model" / "stage2_comparison_study_v2" / "cases_LOCAL.json").read_text(encoding="utf-8"))
case_0 = cases[0]
case_idx = case_0["case"]
seqn = case_0["SEQN"]
meta = case_0["metadata"]

cp_dir = LESS_DATA / "Saved Model" / "stage2_comparison_study_v2" / "checkpoints"
artifacts = s2._load_preprocessing_artifacts(deployment.PREPROCESSOR)

w = meta["weight_kg"]
h = meta["height_cm"]
age = meta["age_years"]
is_fem = meta["is_female"]
bmr = 10.0 * w + 6.25 * h - 5.0 * age + (-161.0 if is_fem else 5.0)
tdee = bmr * s2.PAL_FACTORS[meta["activity_level"]]

ro_cp = json.loads((cp_dir / f"case_{case_idx:03d}_{seqn}_rules_only.json").read_text(encoding="utf-8"))
p_d0 = ro_cp["baseline_probabilities"]["diabetes"]
p_o0 = ro_cp["baseline_probabilities"]["obesity"]
p_c0 = ro_cp["baseline_probabilities"]["cvd"]
t_kcal = max(tdee - 500.0, 1200.0) if p_o0 >= s2.CUTOFF_OBESITY else max(tdee, 1200.0)

lower, upper = s2._build_nutrient_bounds(
    artifacts=artifacts, t_kcal=t_kcal, weight_kg=w, is_female=is_fem, age_years=age,
    p_diab_0=p_d0, p_obes_0=p_o0, p_cvd_0=p_c0
)
n_obs = ro_cp["observed_nutrients"]

mg_cp = json.loads((cp_dir / f"case_{case_idx:03d}_{seqn}_model_guided.json").read_text(encoding="utf-8"))
tgt_vec = mg_cp["target_nutrients"]

envelope = []
for i, code in enumerate(s3.NUTRIENT_CODES):
    envelope.append({
        "nutrient": code,
        "target": float(tgt_vec[i]),
        "min_bound": float(lower[i]),
        "max_bound": float(upper[i]),
        "baseline": float(n_obs[i]),
        "rule_provenance": nrp.reviewed_rule_metadata(code, float(lower[i]), float(upper[i]))
    })

s3_input = {
    "patient_metadata": dict(meta),
    "energy_prescription": {"BMR": float(bmr), "TDEE": float(tdee), "target_calories": float(t_kcal)},
    "nutrient_constraint_envelope": envelope,
}

foods = s3.load_food_database()

# Test Variant A (current solver)
t0 = time.perf_counter()
out_a = s3.solve_meal_plan(s3_input, foods, time_limit_sec=3.0, revision_spec={"dietary_mode": "pure_vegetarian"})
t_a = time.perf_counter() - t0
print(f"Variant A solved in {t_a:.3f}s: accepted={out_a.accepted}")
if out_a.accepted:
    dev_a = adapter.compute_soft_target_deviations(
        {row["nutrient"]: row["delivered"] for row in out_a.delivered_nutrition},
        s3_input["nutrient_constraint_envelope"]
    )
    print(f"Variant A normalized target distance: {dev_a['normalized_target_distance']}")
    print(f"Variant A core MAPE: {dev_a['core_nutrients_mape']}%")
    print("Variant A basket:", [(x["food_id"], x["quantity"]) for x in out_a.solved_food_basket])

# Now test Variant B prototype
stage = s3._parse_stage2_input(s3_input)
dietary_mode = "pure_vegetarian"
required = {code for code in s3.NUTRIENT_CODES
            if stage.rule_metadata.get(code, {}).get("planning_role") not in {"report_only", "exposure_reporting"}}
effective_excluded = set()
for food in foods:
    if food.dietary_type == "non_vegetarian":
        effective_excluded.add(food.food_id)

eligible = [f for f in foods if f.food_id not in effective_excluded and not required.intersection(f.unknown_nutrients)]
m = len(eligible)

c, integrality, bounds, constraints, info = s3._make_sparse_constraints(
    eligible, stage, gamma=0.10, dietary_mode=dietary_mode, allow_zero_dairy=False
)
lower_var = bounds.lb.copy()
upper_var = bounds.ub.copy()
q0 = info["indices"]["q"]
z0 = info["indices"]["z"]
for k, code in enumerate(s3.NUTRIENT_CODES):
    if code in required:
        upper_var[info["indices"]["e_plus"] + k] = 0.0
        upper_var[info["indices"]["e_minus"] + k] = 0.0

F = info["food_matrix"]
kcal = F[s3.NUTRIENT_INDEX["DR1TKCAL"]]
extra_rows, extra_lower, extra_upper = [], [], []
for code, factor, low, high in (("DR1TPROT", 4, .10, .35),
                                ("DR1TCARB", 4, .45, .65),
                                ("DR1TTFAT", 9, .20, .35)):
    nutrient = F[s3.NUTRIENT_INDEX[code]] * factor
    extra_rows.extend((low * kcal - nutrient, nutrient - high * kcal))
    extra_lower.extend((-np.inf, -np.inf))
    extra_upper.extend((0.0, 0.0))
ratio = stage.baselines["DR1TSODI"] / (stage.baselines["DR1TPOTA"] + 1.0)
extra_rows.append(F[s3.NUTRIENT_INDEX["DR1TSODI"]] - ratio * F[s3.NUTRIENT_INDEX["DR1TPOTA"]])
extra_lower.append(-np.inf)
extra_upper.append(ratio)
extra_rows.append(.014 * kcal - F[s3.NUTRIENT_INDEX["DR1TFIBE"]])
extra_lower.append(-np.inf)
extra_upper.append(0.0)

nbase = len(c)
extra = np.zeros((len(extra_rows), nbase))
extra[:, :m] = extra_rows

A = vstack((constraints.A, coo_matrix(extra)), format="csr")
c_lb = np.r_[constraints.lb, np.asarray(extra_lower, dtype=float)]
c_ub = np.r_[constraints.ub, np.asarray(extra_upper, dtype=float)]

# Add Target Awareness: L1 deviation variables
target_nutrients = [code for code in s3.NUTRIENT_CODES if code in required and code != "DR1TKCAL"]
n_target = len(target_nutrients)
n_dev = 2 * n_target  # d_plus and d_minus for each target nutrient

target_weight = 2.0
dev_costs = []
for code in target_nutrients:
    scale = s2.DAILY_REFERENCE_SCALES[code]
    w_k = 1.0
    dev_costs.append(target_weight * w_k / scale)

c_expanded = np.r_[c, np.asarray(dev_costs, dtype=float), np.asarray(dev_costs, dtype=float)]
integrality_expanded = np.r_[integrality, np.zeros(n_dev, dtype=np.int8)]
lower_expanded = np.r_[lower_var, np.zeros(n_dev, dtype=float)]
upper_expanded = np.r_[upper_var, np.full(n_dev, np.inf, dtype=float)]

A_existing_padded = coo_matrix(np.hstack((A.toarray(), np.zeros((A.shape[0], n_dev)))))
dev_rows = np.zeros((n_target, nbase + n_dev), dtype=float)
dev_rhs = np.zeros(n_target, dtype=float)

for j, code in enumerate(target_nutrients):
    k = s3.NUTRIENT_INDEX[code]
    dev_rows[j, :m] = F[k, :]
    dev_rows[j, nbase + j] = -1.0             # - d_plus
    dev_rows[j, nbase + n_target + j] = 1.0   # + d_minus
    dev_rhs[j] = float(stage.targets[code])

A_b = vstack((A_existing_padded, coo_matrix(dev_rows)), format="csr")
c_lb_b = np.r_[c_lb, dev_rhs]
c_ub_b = np.r_[c_ub, dev_rhs]

constraints_b = LinearConstraint(A_b, c_lb_b, c_ub_b)
bounds_b = Bounds(lower_expanded, upper_expanded)

t0 = time.perf_counter()
res_b = milp(c=c_expanded, integrality=integrality_expanded, bounds=bounds_b,
             constraints=constraints_b, options={"time_limit": 3.0, "mip_rel_gap": .01, "presolve": True, "disp": False})
t_b = time.perf_counter() - t0
print(f"Variant B solved in {t_b:.3f}s: status={res_b.status}")
if res_b.x is not None:
    q_b = np.rint(res_b.x[:m]).astype(np.int64)
    out_b = s3._build_output_from_quantities(q_b, eligible, stage, {})
    val_b = s3.validate_candidate(out_b, foods, s3_input, excluded_ids=sorted(effective_excluded), dietary_mode=dietary_mode)
    print("Variant B candidate accepted:", val_b["accepted"])
    dev_b = adapter.compute_soft_target_deviations(
        {row["nutrient"]: row["delivered"] for row in out_b.delivered_nutrition},
        s3_input["nutrient_constraint_envelope"]
    )
    print(f"Variant B normalized target distance: {dev_b['normalized_target_distance']}")
    print(f"Variant B core MAPE: {dev_b['core_nutrients_mape']}%")
    print("Variant B basket:", [(x["food_id"], x["quantity"]) for x in out_b.solved_food_basket])
