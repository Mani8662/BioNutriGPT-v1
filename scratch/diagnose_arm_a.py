import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json, numpy as np, torch
import stage2_counterfactual_engine as s2
from verify_stage2_constraint_audit_repaired_v1 import check
from scratch.dev_test_arms import construct_rules_only_target, cases, artifacts, model, device, NAMES, IX

for i, case in enumerate(cases):
    raw, _ = s2.canonicalize_dietary_ratios(case['vector'])
    meta = case['metadata']
    norm_0 = s2._normalize_torch(torch.as_tensor(raw, dtype=torch.float32, device=device).unsqueeze(0), artifacts)
    p_d0, p_o0, p_c0 = s2._predict_probabilities(model, norm_0, no_grad=True)
    p_d0, p_o0, p_c0 = float(p_d0.cpu()), float(p_o0.cpu()), float(p_c0.cpu())
    w, h, age, is_fem = meta['weight_kg'], meta['height_cm'], meta['age_years'], meta['is_female']
    bmr = 10.0 * w + 6.25 * h - 5.0 * age + (-161.0 if is_fem else 5.0)
    tdee = bmr * s2.PAL_FACTORS[meta['activity_level']]
    t_kcal = max(tdee - 500.0, 1200.0) if p_o0 >= s2.CUTOFF_OBESITY else max(tdee, 1200.0)
    lower, upper = s2._build_nutrient_bounds(artifacts=artifacts, t_kcal=t_kcal, weight_kg=w, is_female=is_fem, age_years=age, p_diab_0=p_d0, p_obes_0=p_o0, p_cvd_0=p_c0)
    target = construct_rules_only_target(raw, meta, p_d0, p_o0, p_c0, lower, upper)
    base_dict = {k: raw[IX[k] + len(s2.BIO_COLS)] for k in NAMES}
    num_check = check(target, lower, upper, NAMES, base_dict, meta, p_c0)
    if not num_check['joint']:
        print(f"Case {i:02d} failed: bound_names={num_check['bound_names']}, energy_ok={num_check['energy_ok']}, ratios_ok={num_check['ratios_ok']}, filter2_ok={num_check['filter2_ok']}")
