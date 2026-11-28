"""Test ratio-coherent deterministic rules for Arm A on the 32 dev cases."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json, numpy as np, torch
import stage2_counterfactual_engine as s2
from verify_stage2_constraint_audit_repaired_v1 import check

CASES_PATH = ROOT / "Saved Model/stage2_constraint_audit_repaired_v1/cases_LOCAL.json"
cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
artifacts = s2._load_preprocessing_artifacts(s2.DEFAULT_PREPROCESSOR)
model = s2._load_stage1_model(s2.DEFAULT_CHECKPOINT, artifacts, device)

NAMES = s2.DIET_RAW_COLS
IX = {n: i for i, n in enumerate(NAMES)}

def construct_deterministic_rules_target(raw, meta, p_d0, p_o0, p_c0, lower, upper):
    t_kcal = lower[IX['DR1TKCAL']] / 0.95
    w = meta["weight_kg"]
    age = meta["age_years"]
    is_fem = meta["is_female"]

    target = np.zeros(39, dtype=np.float64)
    target[IX['DR1TKCAL']] = t_kcal

    # Protein: 1.0 g/kg (bounded by [0.8W, 2.0W])
    prot_target = max(1.0 * w, lower[IX['DR1TPROT']])
    prot_target = min(prot_target, upper[IX['DR1TPROT']])
    target[IX['DR1TPROT']] = prot_target

    # Fat percentage
    fat_pct = 0.30 if p_d0 >= s2.CUTOFF_DIABETES else 0.25
    target[IX['DR1TTFAT']] = (fat_pct * t_kcal) / 9.0
    target[IX['DR1TALCO']] = 0.0

    carb_kcal = t_kcal - (4.0 * target[IX['DR1TPROT']] + 9.0 * target[IX['DR1TTFAT']])
    target[IX['DR1TCARB']] = carb_kcal / 4.0

    if target[IX['DR1TCARB']] > upper[IX['DR1TCARB']]:
        target[IX['DR1TCARB']] = upper[IX['DR1TCARB']]
        rem_kcal = t_kcal - (4.0 * target[IX['DR1TPROT']] + 4.0 * target[IX['DR1TCARB']])
        target[IX['DR1TTFAT']] = rem_kcal / 9.0

    # Baseline ratio limits
    base_sf = float(raw[s2.FEATURE_COLS.index('SUGAR_TO_FIBER_RATIO')])
    base_nk = float(raw[s2.FEATURE_COLS.index('SODIUM_POTASSIUM_RATIO')])
    base_su = float(raw[s2.FEATURE_COLS.index('SAT_TO_UNSAT_FAT_RATIO')])
    a_sf = max(base_sf, 1.0)
    a_nk = max(base_nk, 1.0)
    a_su = max(base_su, 0.4)

    # Fiber
    target[IX['DR1TFIBE']] = lower[IX['DR1TFIBE']]

    # Sugar: constrained by WHO guideline (<10% kcal) and participant's sugar/fiber ratio
    max_sugar_ratio = a_sf * (target[IX['DR1TFIBE']] + 1.0) * 0.99
    target[IX['DR1TSUGR']] = min((0.10 * t_kcal) / 4.0, max_sugar_ratio)

    # Fatty acids
    target[IX['DR1TPFAT']] = (0.07 * t_kcal) / 9.0
    # Saturated fat limited by CVD rule (6% or 10%) and sat/unsat fat ratio
    sfat_cap = ((0.05 if p_c0 >= s2.CUTOFF_CVD else 0.07) * t_kcal) / 9.0
    # Let MUFA be total fat - PFAT - SFAT
    # We want SFAT <= a_su * (MFAT + PFAT + 0.1) = a_su * (TTFAT - SFAT + 0.1)
    # => SFAT * (1 + a_su) <= a_su * (TTFAT + 0.1)
    # => SFAT <= (a_su / (1 + a_su)) * (TTFAT + 0.1)
    sfat_ratio_cap = (a_su / (1.0 + a_su)) * (target[IX['DR1TTFAT']] + 0.1) * 0.99
    target[IX['DR1TSFAT']] = min(sfat_cap, sfat_ratio_cap)
    target[IX['DR1TMFAT']] = max(0.0, target[IX['DR1TTFAT']] - target[IX['DR1TSFAT']] - target[IX['DR1TPFAT']])

    target[IX['DR1TCHOL']] = 100.0 if p_c0 >= s2.CUTOFF_CVD else 150.0

    # Potassium
    pota_floor = 3500.0 if p_c0 >= s2.CUTOFF_CVD else 2600.0
    target[IX['DR1TPOTA']] = pota_floor

    # Sodium limited by CVD ceiling and sodium/potassium ratio
    sodi_cap = 1500.0 if p_c0 >= s2.CUTOFF_CVD else 2300.0
    sodi_ratio_cap = a_nk * (target[IX['DR1TPOTA']] + 1.0) * 0.99
    target[IX['DR1TSODI']] = min(sodi_cap, sodi_ratio_cap)

    # Micronutrients with positive lower bound: set to lower bound
    for code in NAMES:
        idx = IX[code]
        if target[idx] > 0.0:
            continue
        low_val = lower[idx]
        if low_val > 0.0:
            target[idx] = low_val
        else:
            target[idx] = 0.0

    target[IX['DR1TFF']] = 400.0
    target[IX['DR1TFOLA']] = 400.0

    return target

# Test on 32 cases
passes = 0
num_passes = 0
score_passes = 0

for i, case in enumerate(cases):
    raw, _ = s2.canonicalize_dietary_ratios(case["vector"])
    meta = case["metadata"]
    norm_0 = s2._normalize_torch(torch.as_tensor(raw, dtype=torch.float32, device=device).unsqueeze(0), artifacts)
    p_d0, p_o0, p_c0 = s2._predict_probabilities(model, norm_0, no_grad=True)
    p_d0, p_o0, p_c0 = float(p_d0.cpu()), float(p_o0.cpu()), float(p_c0.cpu())
    w, h, age, is_fem = meta["weight_kg"], meta["height_cm"], meta["age_years"], meta["is_female"]
    bmr = 10.0 * w + 6.25 * h - 5.0 * age + (-161.0 if is_fem else 5.0)
    tdee = bmr * s2.PAL_FACTORS[meta["activity_level"]]
    t_kcal = max(tdee - 500.0, 1200.0) if p_o0 >= s2.CUTOFF_OBESITY else max(tdee, 1200.0)
    lower, upper = s2._build_nutrient_bounds(artifacts=artifacts, t_kcal=t_kcal, weight_kg=w, is_female=is_fem, age_years=age, p_diab_0=p_d0, p_obes_0=p_o0, p_cvd_0=p_c0)

    target = construct_deterministic_rules_target(raw, meta, p_d0, p_o0, p_c0, lower, upper)
    base_dict = {k: raw[IX[k] + len(s2.BIO_COLS)] for k in NAMES}
    num_check = check(target, lower, upper, NAMES, base_dict, meta, p_c0)
    if num_check["joint"]:
        num_passes += 1
    else:
        print(f"Case {i} failed num: {num_check}")

    raw_t = torch.tensor(raw, dtype=torch.float32, device=device)
    cand_t = s2._construct_candidate_88(raw_t, torch.tensor(target, dtype=torch.float32, device=device))
    norm_cand = s2._normalize_torch(cand_t, artifacts)
    p_d, p_o, p_c = s2._predict_probabilities(model, norm_cand, no_grad=True)
    delta_d = float(p_d.cpu()) - p_d0
    delta_o = float(p_o.cpu()) - p_o0
    delta_c = float(p_c.cpu()) - p_c0
    score_pass = (delta_d <= 0.0015) and (delta_o <= 0.0015) and (delta_c <= 0.0015)
    if score_pass:
        score_passes += 1
    if num_check["joint"] and score_pass:
        passes += 1

print(f"\nArm A (Refined Deterministic Rules): Numerical pass: {num_passes}/32, Score pass: {score_passes}/32, All-check pass: {passes}/32")
