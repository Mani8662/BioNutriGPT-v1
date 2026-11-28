import json
import math
import sys
import time
from pathlib import Path
import numpy as np

LESS_DATA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LESS_DATA))

import stage3_milp_solver as s3
import stage2_stage3_adapter as adapter
import stage2_counterfactual_engine as s2
import stage1_deployment as deployment

# Load cases and artifacts
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

s3_input = adapter.build_stage3_case_input(
    case_meta=meta,
    t_kcal=t_kcal,
    observed_nutrients=n_obs,
    target_nutrients=tgt_vec,
    lower_bounds=lower,
    upper_bounds=upper,
)

foods = s3.load_food_database()
print("Successfully built Stage 3 input for case 0 model_guided!")
print("Target calories:", s3_input["energy_prescription"]["target_calories"])
print("Target protein:", s3_input["targets"]["DR1TPROT"])
print("Target fiber:", s3_input["targets"]["DR1TFIBE"])
print("Target sodium:", s3_input["targets"]["DR1TSODI"])
