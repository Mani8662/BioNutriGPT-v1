from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import json
import time
import stage3_milp_solver as s3
import stage2_counterfactual_engine as s2
import stage1_deployment as deployment
import nutrient_rule_policy as nrp

ROOT = Path(".")
cases = json.load(open("Saved Model/stage2_comparison_study_v2/cases_LOCAL.json", encoding="utf-8"))
artifacts = s2._load_preprocessing_artifacts(deployment.PREPROCESSOR)
foods = s3.load_food_database()

c0 = cases[0]
meta = c0["metadata"]
raw = c0["vector"]
seqn = c0["SEQN"]
cp_dir = Path("Saved Model/stage2_comparison_study_v2/checkpoints")
ro_cp = json.load(open(cp_dir / f"case_000_{seqn}_rules_only.json"))
mc_cp = json.load(open(cp_dir / f"case_000_{seqn}_minimum_change.json"))
mg_cp = json.load(open(cp_dir / f"case_000_{seqn}_model_guided.json"))

p_d0 = ro_cp["baseline_probabilities"]["diabetes"]
p_o0 = ro_cp["baseline_probabilities"]["obesity"]
p_c0 = ro_cp["baseline_probabilities"]["cvd"]
w = meta["weight_kg"]
h = meta["height_cm"]
age = meta["age_years"]
is_fem = meta["is_female"]
bmr = 10.0 * w + 6.25 * h - 5.0 * age + (-161.0 if is_fem else 5.0)
tdee = bmr * s2.PAL_FACTORS[meta["activity_level"]]
t_kcal = max(tdee - 500.0, 1200.0) if p_o0 >= s2.CUTOFF_OBESITY else max(tdee, 1200.0)

lower, upper = s2._build_nutrient_bounds(
    artifacts=artifacts, t_kcal=t_kcal, weight_kg=w, is_female=is_fem, age_years=age,
    p_diab_0=p_d0, p_obes_0=p_o0, p_cvd_0=p_c0
)
n_obs = ro_cp["observed_nutrients"]

def build_s3_input(target_vec):
    envelope = []
    for i, code in enumerate(s3.NUTRIENT_CODES):
        envelope.append({
            "nutrient": code,
            "target": float(target_vec[i]),
            "min_bound": float(lower[i]),
            "max_bound": float(upper[i]),
            "baseline": float(n_obs[i]),
            "rule_provenance": nrp.reviewed_rule_metadata(code, float(lower[i]), float(upper[i]))
        })
    return {
        "patient_metadata": dict(meta),
        "energy_prescription": {"BMR": float(bmr), "TDEE": float(tdee), "target_calories": float(t_kcal)},
        "nutrient_constraint_envelope": envelope
    }

for mode in ["pure_vegetarian", "veg_nonveg_combo", "pure_non_vegetarian"]:
    for arm_name, cp in [("rules_only", ro_cp), ("minimum_change", mc_cp), ("model_guided", mg_cp)]:
        s3_in = build_s3_input(cp["target_nutrients"])
        t0 = time.perf_counter()
        out = s3.solve_meal_plan(s3_in, foods, time_limit_sec=3.0, revision_spec={"dietary_mode": mode})
        dt = time.perf_counter() - t0
        basket = [(x["food_id"], x["quantity"]) for x in out.solved_food_basket]
        status = out.solver_metadata.get("status")
        print(f"Mode: {mode:<20} Arm: {arm_name:<16} Accepted: {out.accepted} Time: {dt:.3f}s BasketLen: {len(basket)} Status: {status}")
