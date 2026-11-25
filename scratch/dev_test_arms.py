"""Explore Arm A, B, C on the 32 development cases to freeze the protocol and rules."""
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import math
import numpy as np
import torch

import stage2_counterfactual_engine as s2
import nutrient_rule_policy as policy
import stage1_deployment as deployment
from verify_stage2_constraint_audit_repaired_v1 import check

ROOT = Path(__file__).resolve().parent.parent
CASES_PATH = ROOT / "Saved Model/stage2_constraint_audit_repaired_v1/cases_LOCAL.json"
OUT_AUDIT = ROOT / "Saved Model/stage2_joint_constraint_repair_v2"

cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
print(f"Loaded {len(cases)} development cases.")

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
artifacts = s2._load_preprocessing_artifacts(s2.DEFAULT_PREPROCESSOR)
model = s2._load_stage1_model(s2.DEFAULT_CHECKPOINT, artifacts, device)

NAMES = s2.DIET_RAW_COLS
IX = {n: i for i, n in enumerate(NAMES)}

def construct_rules_only_target(raw_vector, metadata, p_diab_0, p_obes_0, p_cvd_0, lower, upper):
    """Deterministic rules-only target construction based strictly on clinical reference guidelines."""
    t_kcal = lower[IX['DR1TKCAL']] / 0.95  # exact target calories T_kcal
    w = metadata["weight_kg"]
    is_fem = metadata["is_female"]
    age = metadata["age_years"]

    target = np.zeros(39, dtype=np.float64)

    # 1. KCAL: prescribed energy requirement
    target[IX['DR1TKCAL']] = t_kcal

    # 2. Protein: standard clinical guideline target: 1.0 g/kg body weight, bounded by envelope
    prot_target = 1.0 * w
    prot_target = max(prot_target, lower[IX['DR1TPROT']])
    prot_target = min(prot_target, upper[IX['DR1TPROT']])
    target[IX['DR1TPROT']] = prot_target

    # 3. Fat: 25% of energy, or 30% if diabetes rule is active to allow lower carb fraction
    fat_pct = 0.30 if p_diab_0 >= s2.CUTOFF_DIABETES else 0.25
    target[IX['DR1TTFAT']] = (fat_pct * t_kcal) / 9.0

    # 4. Alcohol: 0.0 g (standard clinical guideline for medical nutrition therapy)
    target[IX['DR1TALCO']] = 0.0

    # 5. Carbohydrate: determined by thermodynamic energy balance
    carb_kcal = t_kcal - (4.0 * target[IX['DR1TPROT']] + 9.0 * target[IX['DR1TTFAT']] + 7.0 * target[IX['DR1TALCO']])
    target[IX['DR1TCARB']] = carb_kcal / 4.0

    # Check if carb exceeds upper bound under diabetes routing
    if target[IX['DR1TCARB']] > upper[IX['DR1TCARB']]:
        # Under diabetes routing, increase fat fraction up to AMDR limit (35%) to accommodate carb ceiling
        target[IX['DR1TCARB']] = upper[IX['DR1TCARB']]
        rem_kcal = t_kcal - (4.0 * target[IX['DR1TPROT']] + 4.0 * target[IX['DR1TCARB']])
        target[IX['DR1TTFAT']] = rem_kcal / 9.0

    # 6. Saturated fat: clinical target <= 6% (CVD) or <= 7% (general) of energy
    sfat_pct = 0.05 if p_cvd_0 >= s2.CUTOFF_CVD else 0.07
    target[IX['DR1TSFAT']] = (sfat_pct * t_kcal) / 9.0

    # 7. PUFA: 7% of energy; MUFA: remainder of total fat (TTFAT - SFAT - PFAT)
    target[IX['DR1TPFAT']] = (0.07 * t_kcal) / 9.0
    target[IX['DR1TMFAT']] = max(0.0, target[IX['DR1TTFAT']] - target[IX['DR1TSFAT']] - target[IX['DR1TPFAT']])

    # 8. Fiber: guideline target max(14g/1000kcal, floor)
    target[IX['DR1TFIBE']] = lower[IX['DR1TFIBE']]

    # 9. Sugar: WHO recommendation free sugar < 10% energy; total sugar target set to 1.5 * fiber
    target[IX['DR1TSUGR']] = min(1.5 * target[IX['DR1TFIBE']], (0.10 * t_kcal) / 4.0)

    # 10. Cholesterol: 100 mg (CVD) or 150 mg (general)
    target[IX['DR1TCHOL']] = 100.0 if p_cvd_0 >= s2.CUTOFF_CVD else 150.0

    # 11. Sodium: 1500 mg (CVD) or 2000 mg (general)
    target[IX['DR1TSODI']] = 1500.0 if p_cvd_0 >= s2.CUTOFF_CVD else 2000.0

    # 12. Potassium: 3500 mg (CVD) or 3000 mg (general)
    target[IX['DR1TPOTA']] = 3500.0 if p_cvd_0 >= s2.CUTOFF_CVD else 3000.0

    # 13. Micronutrients with positive RDA/AI lower bounds: set to lower bound (RDA/AI reference)
    for code in NAMES:
        idx = IX[code]
        if target[idx] > 0.0:
            continue
        low_val = lower[idx]
        up_val = upper[idx]
        if low_val > 0.0:
            target[idx] = low_val
        else:
            # Nutrients with lower bound 0.0 (carotenoids, caffeine, synthetic folate/retinol)
            # Standard rules set target to 0.0 (or to reference intake / baseline)
            target[idx] = 0.0

    # Documented explicit adjustments:
    # Food folate DR1TFF set to 400 ug, Total folate DR1TFOLA set to 400 ug (to match DR1TFDFE RDA)
    target[IX['DR1TFF']] = 400.0
    target[IX['DR1TFOLA']] = 400.0

    return target

# Let's run a test over the 32 cases
print("Testing candidate Arm A on 32 cases...")
arm_a_passes = 0
arm_a_score_passes = 0
arm_a_num_passes = 0

for i, case in enumerate(cases):
    raw, _ = s2.canonicalize_dietary_ratios(case["vector"])
    meta = case["metadata"]
    # Baseline probabilities
    norm_0 = s2._normalize_torch(torch.as_tensor(raw, dtype=torch.float32, device=device).unsqueeze(0), artifacts)
    p_d0, p_o0, p_c0 = s2._predict_probabilities(model, norm_0, no_grad=True)
    p_d0, p_o0, p_c0 = float(p_d0.cpu()), float(p_o0.cpu()), float(p_c0.cpu())

    weight_kg = meta["weight_kg"]
    height_cm = meta["height_cm"]
    age_years = meta["age_years"]
    is_female = meta["is_female"]
    bmr = 10.0 * weight_kg + 6.25 * height_cm - 5.0 * age_years + (-161.0 if is_female else 5.0)
    tdee = bmr * s2.PAL_FACTORS[meta["activity_level"]]
    t_kcal = max(tdee - 500.0, 1200.0) if p_o0 >= s2.CUTOFF_OBESITY else max(tdee, 1200.0)

    lower, upper = s2._build_nutrient_bounds(
        artifacts=artifacts, t_kcal=t_kcal, weight_kg=weight_kg,
        is_female=is_female, age_years=age_years,
        p_diab_0=p_d0, p_obes_0=p_o0, p_cvd_0=p_c0,
    )

    # Arm A target
    target_a = construct_rules_only_target(raw, meta, p_d0, p_o0, p_c0, lower, upper)
    base_dict = {k: raw[IX[k] + len(s2.BIO_COLS)] for k in NAMES}
    num_check = check(target_a, lower, upper, NAMES, base_dict, meta, p_c0)
    if num_check["joint"]:
        arm_a_num_passes += 1
    else:
        if i == 0:
            print("Case 0 Arm A failure:", num_check)

    # Closed loop score
    raw_t = torch.tensor(raw, dtype=torch.float32, device=device)
    cand_t = s2._construct_candidate_88(raw_t, torch.tensor(target_a, dtype=torch.float32, device=device))
    norm_cand = s2._normalize_torch(cand_t, artifacts)
    p_d, p_o, p_c = s2._predict_probabilities(model, norm_cand, no_grad=True)
    delta_d = float(p_d.cpu()) - p_d0
    delta_o = float(p_o.cpu()) - p_o0
    delta_c = float(p_c.cpu()) - p_c0
    score_pass = (delta_d <= 0.0015) and (delta_o <= 0.0015) and (delta_c <= 0.0015)
    if score_pass:
        arm_a_score_passes += 1
    if num_check["joint"] and score_pass:
        arm_a_passes += 1

print(f"Arm A (Pure deterministic rules): Numerical pass: {arm_a_num_passes}/32, Score pass: {arm_a_score_passes}/32, All-check pass: {arm_a_passes}/32")

# Test Arm B: Minimum-change joint projection of observed intake
print("\nTesting Arm B (Minimum-change projection of observed intake)...")
arm_b_passes = 0
arm_b_score_passes = 0
arm_b_num_passes = 0
arm_b_distances = []

for i, case in enumerate(cases):
    raw, _ = s2.canonicalize_dietary_ratios(case["vector"])
    meta = case["metadata"]
    norm_0 = s2._normalize_torch(torch.as_tensor(raw, dtype=torch.float32, device=device).unsqueeze(0), artifacts)
    p_d0, p_o0, p_c0 = s2._predict_probabilities(model, norm_0, no_grad=True)
    p_d0, p_o0, p_c0 = float(p_d0.cpu()), float(p_o0.cpu()), float(p_c0.cpu())

    weight_kg = meta["weight_kg"]
    height_cm = meta["height_cm"]
    age_years = meta["age_years"]
    is_female = meta["is_female"]
    bmr = 10.0 * weight_kg + 6.25 * height_cm - 5.0 * age_years + (-161.0 if is_female else 5.0)
    tdee = bmr * s2.PAL_FACTORS[meta["activity_level"]]
    t_kcal = max(tdee - 500.0, 1200.0) if p_o0 >= s2.CUTOFF_OBESITY else max(tdee, 1200.0)

    lower, upper = s2._build_nutrient_bounds(
        artifacts=artifacts, t_kcal=t_kcal, weight_kg=weight_kg,
        is_female=is_female, age_years=age_years,
        p_diab_0=p_d0, p_obes_0=p_o0, p_cvd_0=p_c0,
    )

    # Observed nutrient intake vector
    n_obs = raw[len(s2.BIO_COLS) : len(s2.BIO_COLS) + 39].copy()

    # Project observed intake directly onto the joint feasible polytope
    n_proj, audit = s2._project_to_joint_feasible_polytope(
        n_candidate=n_obs, lower=lower, upper=upper, raw_vector=raw,
        metadata=meta, p_cvd_0=p_c0
    )

    # Sequential filters (with repaired safety)
    f1 = s2._apply_filter_1_thermodynamic_balance(n_proj)
    f2 = s2._apply_filter_2_iom_safety(n_proj, weight_kg=weight_kg, p_cvd_0=p_c0)
    base_ratios = raw[len(s2.BIO_COLS) + len(s2.DIET_RAW_COLS) : len(s2.BIO_COLS) + len(s2.DIET_RAW_COLS) + len(s2.DIET_RATIO_COLS)]
    f3 = s2._apply_filter_3_ratio_coherence(n_proj, baseline_ratios_from_vector=base_ratios, lower=lower, upper=upper)

    base_dict = {k: raw[IX[k] + len(s2.BIO_COLS)] for k in NAMES}
    num_check = check(n_proj, lower, upper, NAMES, base_dict, meta, p_c0)
    if num_check["joint"]:
        arm_b_num_passes += 1

    # Closed-loop model score
    raw_t = torch.tensor(raw, dtype=torch.float32, device=device)
    cand_t = s2._construct_candidate_88(raw_t, torch.tensor(n_proj, dtype=torch.float32, device=device))
    norm_cand = s2._normalize_torch(cand_t, artifacts)
    p_d, p_o, p_c = s2._predict_probabilities(model, norm_cand, no_grad=True)
    delta_d = float(p_d.cpu()) - p_d0
    delta_o = float(p_o.cpu()) - p_o0
    delta_c = float(p_c.cpu()) - p_c0
    score_pass = (delta_d <= 0.0015) and (delta_o <= 0.0015) and (delta_c <= 0.0015)
    if score_pass:
        arm_b_score_passes += 1
    if num_check["joint"] and score_pass:
        arm_b_passes += 1

    scales = np.array([s2.DAILY_REFERENCE_SCALES[k] for k in NAMES], dtype=np.float64)
    dist = float(np.sqrt(np.sum(((n_proj - n_obs) / scales) ** 2)))
    arm_b_distances.append(dist)

print(f"Arm B (Minimum-change projection): Numerical pass: {arm_b_num_passes}/32, Score pass: {arm_b_score_passes}/32, All-check pass: {arm_b_passes}/32")
print(f"Arm B Distance from observed: median={np.median(arm_b_distances):.4f}, range=[{np.min(arm_b_distances):.4f}, {np.max(arm_b_distances):.4f}]")

