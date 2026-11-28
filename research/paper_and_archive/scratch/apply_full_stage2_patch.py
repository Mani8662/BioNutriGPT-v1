import json
from pathlib import Path
import subprocess

subprocess.run(["git", "checkout", "stage2_counterfactual_engine.py"], check=True)

with open('stage2_counterfactual_engine.py', 'r', encoding='utf-8') as f:
    code = f.read()

# 1. Imports and constants at top
old_top = '''PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = PROJECT_DIR / "Saved Model" / "bionutrigpt_stage1_best.pt"
DEFAULT_PREPROCESSOR = PROJECT_DIR / "Data" / "preprocessing_artifacts.joblib"
DEFAULT_OUTPUT = PROJECT_DIR / "stage2_counterfactual_prescription.json"
DEFAULT_PATIENT = PROJECT_DIR / "sample_patient_single_line.txt"

CUTOFF_DIABETES = 0.617
CUTOFF_OBESITY = 0.445
CUTOFF_CVD = 0.603'''

new_top = '''PROJECT_DIR = Path(__file__).resolve().parent
import stage1_deployment as deployment
DEFAULT_CHECKPOINT = deployment.CHECKPOINT
DEFAULT_PREPROCESSOR = deployment.PREPROCESSOR
DEFAULT_OUTPUT = PROJECT_DIR / "stage2_counterfactual_prescription.json"
DEFAULT_PATIENT = PROJECT_DIR / "sample_patient_single_line.txt"

CUTOFF_DIABETES = deployment.THRESHOLDS['diabetes']
CUTOFF_OBESITY = deployment.THRESHOLDS['obesity']
CUTOFF_CVD = deployment.THRESHOLDS['cvd']'''

assert old_top in code
code = code.replace(old_top, new_top, 1)

# 2. Stage2InfeasibleConstraintsError
old_err = '''class ArtifactContractError(Stage2Error):
    """Raised when a preprocessing/model artifact is missing required data."""'''

new_err = '''class ArtifactContractError(Stage2Error):
    """Raised when a preprocessing/model artifact is missing required data."""


class Stage2InfeasibleConstraintsError(Stage2Error):
    """Raised when the joint nutrient constraint polytope is mathematically empty."""'''

assert old_err in code
code = code.replace(old_err, new_err, 1)

# 3. _validate_patient_vector float64 when REPAIRED
old_vpv = 'return vector.astype(np.float32, copy=True)'
new_vpv = 'return vector.astype(np.float64 if deployment.REPAIRED else np.float32, copy=True)'
assert old_vpv in code
code = code.replace(old_vpv, new_vpv, 1)

# 4. _load_preprocessing_artifacts bundle verify
old_lpa = '''def _load_preprocessing_artifacts(path: Path) -> dict[str, Any]:
    path = Path(path)'''
new_lpa = '''def _load_preprocessing_artifacts(path: Path) -> dict[str, Any]:
    path = Path(path)
    deployment.verify_bundle(preprocessor=path)'''
assert old_lpa in code
code = code.replace(old_lpa, new_lpa, 1)

# 5. _normalize_numpy rounding
old_nn = 'x[0, indices] = transformed[0]'
new_nn = 'x[0, indices] = np.round(transformed[0], 6) if deployment.REPAIRED else transformed[0]'
assert old_nn in code
code = code.replace(old_nn, new_nn, 1)

# 6. _normalize_torch double and straight-through rounding
old_nt1 = 'x = raw_vector_1x88.clone()'
new_nt1 = 'x = raw_vector_1x88.double() if deployment.REPAIRED else raw_vector_1x88.clone()'
assert old_nt1 in code
code = code.replace(old_nt1, new_nt1, 1)

old_nt2 = '''    # Reconstruct without mutating a tensor needed by autograd.
    pieces: list[torch.Tensor] = []
    index_to_selected = {feature_index: position for position, feature_index in enumerate(indices)}
    for feature_index in range(88):'''

new_nt2 = '''    if deployment.REPAIRED:
        # Exact rounded forward values; straight-through gradients for exploratory optimization.
        selected = selected + (torch.round(selected * 1e6) / 1e6 - selected).detach()
    # Reconstruct without mutating a tensor needed by autograd.
    pieces: list[torch.Tensor] = []
    index_to_selected = {feature_index: position for position, feature_index in enumerate(indices)}
    for feature_index in range(len(FEATURE_COLS)):'''
assert old_nt2 in code
code = code.replace(old_nt2, new_nt2, 1)

idx_nt = code.find('def _normalize_torch')
idx_ret = code.find('return result', idx_nt)
code = code[:idx_ret] + 'return result.float() if deployment.REPAIRED else result' + code[idx_ret + len('return result'):]

# 7. _load_stage1_model bundle verify and wrap_model
old_lsm = '''def _load_stage1_model(
    checkpoint_path: Path, artifacts: Mapping[str, Any], device: torch.device
) -> nn.Module:
    """Instantiate the canonical Stage 1 architecture and load its state dict."""
    checkpoint_path = Path(checkpoint_path)'''

new_lsm = '''def _load_stage1_model(
    checkpoint_path: Path, artifacts: Mapping[str, Any], device: torch.device
) -> nn.Module:
    """Instantiate the canonical Stage 1 architecture and load its state dict."""
    checkpoint_path = Path(checkpoint_path)
    deployment.verify_bundle(checkpoint=checkpoint_path)'''
assert old_lsm in code
code = code.replace(old_lsm, new_lsm, 1)

old_lsm2 = '''    model = s1.BioNutriGPTStage1().to(device)
    s1.load_state_dict_checkpoint(model, checkpoint_path, device)
    model.eval()'''

new_lsm2 = '''    model = s1.BioNutriGPTStage1().to(device)
    s1.load_state_dict_checkpoint(model, checkpoint_path, device)
    if deployment.REPAIRED:
        model = deployment.wrap_model(model, device)
    model.eval()'''
assert old_lsm2 in code
code = code.replace(old_lsm2, new_lsm2, 1)

# 8. canonicalize_dietary_ratios input validation & derived features
old_cdr = '''def canonicalize_dietary_ratios(vector):
    """Derive all seven ratios from raw nutrients, retaining an input audit."""
    raw = _validate_patient_vector(vector).copy()
    begin = len(BIO_COLS)
    end = begin + len(DIET_RAW_COLS)
    ratios = _compute_ratios_numpy(raw[begin:end]).astype(np.float32)
    if not np.isfinite(ratios).all():
        raise ValueError('Derived dietary ratios must be finite.')
    supplied = raw[end:end + len(DIET_RATIO_COLS)].copy()
    raw[end:end + len(DIET_RATIO_COLS)] = ratios
    return raw, {
        'policy': 'recompute_from_raw_nutrients_v1',
        'changed_fields': [name for name,a,b in zip(DIET_RATIO_COLS,supplied,ratios) if a != b],
        'supplied': supplied.tolist(), 'derived': ratios.tolist(),
    }'''

new_cdr = '''def canonicalize_dietary_ratios(vector):
    """Derive all seven ratios from raw nutrients, retaining an input audit."""
    raw = _validate_patient_vector(vector).copy()
    original = raw.copy()
    if deployment.REPAIRED:
        flags = [name for name in FEATURE_COLS if name.endswith('_MISS') and raw[FEATURE_COLS.index(name)] != 0]
        if flags:
            raise ValueError(f'Repaired deployment requires complete observed inputs; missingness flags set: {flags}')
        if raw[FEATURE_COLS.index('RIDAGEYR')] < 20:
            raise ValueError('Repaired research deployment supports age 20+ only.')
        binary = ['SEX_FEMALE','SLQ050','PAQ605','PAQ620','PAQ635','PAQ650','PAQ665']
        groups = [[n for n in FEATURE_COLS if n.startswith('SMOKE_')], [n for n in FEATURE_COLS if n.startswith('RACE_')],
                  [n for n in FEATURE_COLS if n.startswith('MARITAL_')]]
        for group in groups:
            values = raw[[FEATURE_COLS.index(n) for n in group]]
            if not np.isin(values, [0,1]).all() or values.sum()!=1:
                raise ValueError(f'Exactly one category must be selected: {group}')
        if any(raw[FEATURE_COLS.index(n)] not in (0,1) for n in binary):
            raise ValueError('Binary features must be encoded as 0 or 1.')
        hdl = raw[FEATURE_COLS.index('LBDHDD')]
        tc = raw[FEATURE_COLS.index('LBXTC')]
        if hdl <= 0 or tc < hdl:
            raise ValueError('Lipid inputs require total cholesterol >= HDL > 0.')
        raw[FEATURE_COLS.index('NON_HDL_CHOL')] = tc - hdl
        raw[FEATURE_COLS.index('CHOL_HDL_RATIO')] = tc / hdl
        raw[FEATURE_COLS.index('NON_HDL_TO_HDL_RATIO')] = (tc - hdl) / hdl
        raw[FEATURE_COLS.index('SLEEP_DEV_FROM_7_5')] = abs(raw[FEATURE_COLS.index('SLEEP_HOURS')] - 7.5)
    begin = len(BIO_COLS)
    end = begin + len(DIET_RAW_COLS)
    ratios = _compute_ratios_numpy(raw[begin:end]).astype(np.float64 if deployment.REPAIRED else np.float32)
    if not np.isfinite(ratios).all():
        raise ValueError('Derived dietary ratios must be finite.')
    supplied = raw[end:end + len(DIET_RATIO_COLS)].copy()
    raw[end:end + len(DIET_RATIO_COLS)] = ratios
    return raw, {
        'policy': 'recompute_from_raw_nutrients_v1',
        'changed_fields': [name for name,a,b in zip(DIET_RATIO_COLS,supplied,ratios) if a != b],
        'supplied': supplied.tolist(), 'derived': ratios.tolist(),
        'other_derived_changes': {name: {'supplied': float(original[FEATURE_COLS.index(name)]), 'derived': float(raw[FEATURE_COLS.index(name)])}
                                  for name in ['NON_HDL_CHOL','CHOL_HDL_RATIO','NON_HDL_TO_HDL_RATIO','SLEEP_DEV_FROM_7_5']
                                  if original[FEATURE_COLS.index(name)] != raw[FEATURE_COLS.index(name)]},
    }'''
assert old_cdr in code
code = code.replace(old_cdr, new_cdr, 1)

# 9. model_binding
old_mb = '''def model_binding(checkpoint_path, preprocessor_path):
    def digest(path):
        with Path(path).open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()
    return {
        'checkpoint_sha256': digest(checkpoint_path),
        'preprocessor_sha256': digest(preprocessor_path),
        'architecture': 'train_bionutrigpt_stage1.BioNutriGPTStage1',
        'feature_order_sha256': hashlib.sha256(json.dumps(FEATURE_COLS).encode()).hexdigest(),
        'probability_transform': 'raw_sigmoid_no_calibration',
        'operating_cutoffs': {'diabetes': CUTOFF_DIABETES, 'obesity': CUTOFF_OBESITY, 'cvd': CUTOFF_CVD},
        'cutoff_provenance': 'legacy_project_constants_not_revalidated_for_repaired_v2',
        'repaired_v2_equivalence_established': False,
    }'''

new_mb = '''def model_binding(checkpoint_path, preprocessor_path):
    manifest = deployment.verify_bundle(checkpoint_path, preprocessor_path)
    def digest(path):
        with Path(path).open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()
    return {
        'checkpoint_sha256': digest(checkpoint_path),
        'preprocessor_sha256': digest(preprocessor_path),
        'architecture': 'train_bionutrigpt_stage1.BioNutriGPTStage1',
        'feature_order_sha256': hashlib.sha256(json.dumps(FEATURE_COLS).encode()).hexdigest(),
        'probability_transform': 'reserved_set_logit_Platt' if deployment.REPAIRED else 'raw_sigmoid_no_calibration',
        'release': deployment.RELEASE,
        'calibrator_sha256': manifest.get('hashes', {}).get('calibrators.joblib'),
        'operating_cutoffs': {'diabetes': CUTOFF_DIABETES, 'obesity': CUTOFF_OBESITY, 'cvd': CUTOFF_CVD},
        'cutoff_provenance': deployment.metadata()['threshold_policy'],
        'research_equivalence_scope': 'See Saved Model/stage1_deployment_verification_v1/verification.json; complete-input parity only.' if deployment.REPAIRED else 'Not established for legacy release',
    }'''
assert old_mb in code
code = code.replace(old_mb, new_mb, 1)

# 10. DAILY_REFERENCE_SCALES, _build_tightened_joint_bounds, _build_linear_constraint_matrices, _project_to_joint_feasible_polytope
target_proj_insert = 'def _apply_filter_1_thermodynamic_balance(n: np.ndarray) -> dict[str, Any]:'
proj_block = '''# Standard adult daily intake scales (US FDA Daily Values / Dietary Reference Intakes).
# Used to scale-normalize deviations during joint quadratic projection.
DAILY_REFERENCE_SCALES: dict[str, float] = {
    "DR1TKCAL": 2000.0, "DR1TPROT": 50.0, "DR1TCARB": 275.0, "DR1TSUGR": 50.0, "DR1TFIBE": 28.0,
    "DR1TTFAT": 78.0, "DR1TSFAT": 20.0, "DR1TMFAT": 30.0, "DR1TPFAT": 20.0, "DR1TCHOL": 300.0,
    "DR1TATOC": 15.0, "DR1TRET": 900.0, "DR1TVARA": 900.0, "DR1TACAR": 1000.0, "DR1TBCAR": 3000.0,
    "DR1TLYCO": 5000.0, "DR1TLZ": 6000.0, "DR1TVB1": 1.2, "DR1TVB2": 1.3, "DR1TNIAC": 16.0,
    "DR1TVB6": 1.7, "DR1TFOLA": 400.0, "DR1TFA": 400.0, "DR1TFF": 400.0, "DR1TFDFE": 400.0,
    "DR1TVB12": 2.4, "DR1TVC": 90.0, "DR1TVK": 120.0, "DR1TCALC": 1300.0, "DR1TPHOS": 1250.0,
    "DR1TMAGN": 420.0, "DR1TIRON": 18.0, "DR1TZINC": 11.0, "DR1TCOPP": 0.9, "DR1TSODI": 2300.0,
    "DR1TPOTA": 4700.0, "DR1TSELE": 55.0, "DR1TCAFF": 200.0, "DR1TALCO": 14.0,
}


def _build_tightened_joint_bounds(
    lower: np.ndarray,
    upper: np.ndarray,
    metadata: Mapping[str, Any],
    p_cvd_0: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Tighten component nutrient bounds with Filter 2 clinical invariants.

    Invariants incorporated into the bounding box:
    - DR1TKCAL >= 1200.0
    - 0.8 * weight_kg <= DR1TPROT <= 2.0 * weight_kg
    - DR1TSODI <= 1500.0 if p_cvd_0 >= CUTOFF_CVD else 2300.0
    - DR1TPOTA >= 3500.0 if p_cvd_0 >= CUTOFF_CVD else 2600.0
    """
    low_tight = np.asarray(lower, dtype=np.float64).copy()
    up_tight = np.asarray(upper, dtype=np.float64).copy()

    weight_kg = float(metadata["weight_kg"])
    low_tight[KCAL] = max(low_tight[KCAL], 1200.0)
    low_tight[PROT] = max(low_tight[PROT], 0.8 * weight_kg)
    up_tight[PROT] = min(up_tight[PROT], 2.0 * weight_kg)

    sodium_ceiling = 1500.0 if p_cvd_0 >= CUTOFF_CVD else 2300.0
    up_tight[SODI] = min(up_tight[SODI], sodium_ceiling)

    potassium_floor = 3500.0 if p_cvd_0 >= CUTOFF_CVD else 2600.0
    low_tight[POTA] = max(low_tight[POTA], potassium_floor)

    return low_tight, up_tight


def _build_linear_constraint_matrices(
    raw_vector: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build A_ub and b_ub for energy balance and clinical ratio envelopes."""
    A: list[np.ndarray] = []
    b: list[float] = []

    def add_row(terms: dict[str, float], rhs: float) -> None:
        row = np.zeros(39, dtype=np.float64)
        for k, v in terms.items():
            row[NUTRIENT_INDEX[k]] = float(v)
        A.append(row)
        b.append(float(rhs))

    # Energy thermodynamic balance: 4*PROT + 4*CARB + 9*TFAT + 7*ALCO ~ KCAL (+- 1%)
    # 4*PROT + 4*CARB + 9*TFAT + 7*ALCO - 1.01*KCAL <= 0
    add_row({'DR1TPROT': 4.0, 'DR1TCARB': 4.0, 'DR1TTFAT': 9.0, 'DR1TALCO': 7.0, 'DR1TKCAL': -1.01}, 0.0)
    # -4*PROT - 4*CARB - 9*TFAT - 7*ALCO + 0.99*KCAL <= 0
    add_row({'DR1TPROT': -4.0, 'DR1TCARB': -4.0, 'DR1TTFAT': -9.0, 'DR1TALCO': -7.0, 'DR1TKCAL': 0.99}, 0.0)

    # Clinical ratios relative to baseline features:
    base_sf = float(raw_vector[FEATURE_COLS.index('SUGAR_TO_FIBER_RATIO')])
    base_nk = float(raw_vector[FEATURE_COLS.index('SODIUM_POTASSIUM_RATIO')])
    base_su = float(raw_vector[FEATURE_COLS.index('SAT_TO_UNSAT_FAT_RATIO')])

    a_sf = max(base_sf, 1.0)
    a_nk = max(base_nk, 1.0)
    a_su = max(base_su, 0.4)

    # SUGAR / (FIBER + 1) <= a_sf  <=>  SUGAR - a_sf * FIBER <= a_sf
    add_row({'DR1TSUGR': 1.0, 'DR1TFIBE': -a_sf}, a_sf)
    # SODIUM / (POTASSIUM + 1) <= a_nk  <=>  SODIUM - a_nk * POTASSIUM <= a_nk
    add_row({'DR1TSODI': 1.0, 'DR1TPOTA': -a_nk}, a_nk)
    # SFAT / (MFAT + PFAT + 0.1) <= a_su  <=>  SFAT - a_su * MFAT - a_su * PFAT <= 0.1 * a_su
    add_row({'DR1TSFAT': 1.0, 'DR1TMFAT': -a_su, 'DR1TPFAT': -a_su}, 0.1 * a_su)

    return np.asarray(A, dtype=np.float64), np.asarray(b, dtype=np.float64)


def _project_to_joint_feasible_polytope(
    n_candidate: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    raw_vector: np.ndarray,
    metadata: Mapping[str, Any],
    p_cvd_0: float,
    *,
    raise_on_infeasible: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Convex quadratic projection of candidate nutrient targets onto the joint feasible polytope.

    Solves:
        min_{n} 1/2 sum_{i=0}^{38} ((n_i - n_cand_i) / s_i)^2
        s.t.  lower_tight <= n <= upper_tight
              A_ub @ n <= b_ub

    where s_i are standard adult reference daily intake scales (DAILY_REFERENCE_SCALES).
    Gradient propagation stops at the projection step (surrogate forward projection).
    """
    import time
    from scipy.optimize import linprog, minimize

    t0 = time.perf_counter()

    n_cand = np.asarray(n_candidate, dtype=np.float64).copy()
    if not np.isfinite(n_cand).all():
        raise Stage2Error("Candidate nutrient target contains non-finite values prior to projection.")

    low_tight, up_tight = _build_tightened_joint_bounds(lower, upper, metadata, p_cvd_0)
    if not np.isfinite(low_tight).all() or not np.isfinite(up_tight).all():
        raise Stage2Error("Nutrient bounds contain non-finite values.")

    if np.any(low_tight > up_tight + 1e-9):
        bad = [DIET_RAW_COLS[i] for i in range(39) if low_tight[i] > up_tight[i] + 1e-9]
        if raise_on_infeasible:
            raise Stage2InfeasibleConstraintsError(
                f"Conflicting nutrient bounds for fields {bad}: lower exceeds upper. Refusing to emit an invalid target."
            )
        return n_cand, {'passed': False, 'feasible': False, 'status': 'infeasible_bounds', 'conflicting_fields': bad}

    A_ub, b_ub = _build_linear_constraint_matrices(raw_vector)

    # Check if n_cand is already strictly feasible
    tol = 1e-6
    in_bounds = np.all(n_cand >= low_tight - tol) and np.all(n_cand <= up_tight + tol)
    in_ineq = np.all(A_ub @ n_cand <= b_ub + tol)
    if in_bounds and in_ineq:
        dt = time.perf_counter() - t0
        return n_cand, {
            'passed': True,
            'feasible': True,
            'status': 'already_feasible',
            'nit': 0,
            'objective_value': 0.0,
            'projection_distance_normalized': 0.0,
            'runtime_ms': dt * 1000.0,
            'max_bound_residual': 0.0,
            'max_linear_residual': 0.0,
        }

    scales = np.array([DAILY_REFERENCE_SCALES[k] for k in DIET_RAW_COLS], dtype=np.float64)
    weights = 1.0 / (scales ** 2)

    # Step 1: LP witness to verify feasibility and provide guaranteed feasible warm start
    c_lp = weights * (np.clip(n_cand, low_tight, up_tight) - n_cand)
    lp = linprog(c=c_lp, A_ub=A_ub, b_ub=b_ub, bounds=list(zip(low_tight, up_tight)), method='highs')
    if not lp.success:
        if raise_on_infeasible:
            raise Stage2InfeasibleConstraintsError(
                f"Joint nutrient constraint set is mathematically infeasible: {lp.message}. Refusing to emit an invalid target."
            )
        return n_cand, {'passed': False, 'feasible': False, 'status': 'infeasible_linear_constraints', 'message': lp.message}

    x0 = lp.x

    # Step 2: SLSQP quadratic projection
    res = minimize(
        lambda x: 0.5 * np.sum(weights * ((x - n_cand) ** 2)),
        x0,
        jac=lambda x: weights * (x - n_cand),
        method='SLSQP',
        bounds=list(zip(low_tight, up_tight)),
        constraints={'type': 'ineq', 'fun': lambda x: b_ub - A_ub @ x, 'jac': lambda x: -A_ub},
        options={'ftol': 1e-12, 'maxiter': 500}
    )

    x_star = np.clip(res.x, low_tight, up_tight)
    # Numerical safety: if SLSQP slightly violates linear constraints, revert to guaranteed LP witness
    if not np.all(A_ub @ x_star <= b_ub + tol):
        x_star = np.clip(lp.x, low_tight, up_tight)

    dt = time.perf_counter() - t0
    dist = float(np.sqrt(np.sum(((x_star - n_cand) / scales) ** 2)))
    obj = float(0.5 * np.sum(weights * ((x_star - n_cand) ** 2)))

    audit = {
        'passed': True,
        'feasible': True,
        'status': 'projected_successfully',
        'nit': int(res.nit),
        'slsqp_success': bool(res.success),
        'objective_value': obj,
        'projection_distance_normalized': dist,
        'runtime_ms': dt * 1000.0,
        'max_bound_residual': float(np.max(np.maximum(0, np.maximum(low_tight - x_star, x_star - up_tight)))),
        'max_linear_residual': float(np.max(np.maximum(0, A_ub @ x_star - b_ub))),
    }
    return x_star, audit

'''
assert target_proj_insert in code
code = code.replace(target_proj_insert, proj_block + target_proj_insert, 1)

# 11. generate_counterfactual_prescription projection call, metadata, audit_filters, certified
target_gen1 = '''    # STEP 5: four sequential verification filters.
    filter1 = _apply_filter_1_thermodynamic_balance(target_nutrients)'''

replace_gen1 = '''    # STEP 4.5: Joint constraint projection.
    # Replace fragile sequential overwrites with exact convex quadratic projection onto the
    # joint feasible polytope defined by component bounds, Filter 2 clinical invariants,
    # thermodynamic energy balance, and clinical ratio envelopes.
    if deployment.REPAIRED:
        target_nutrients, projection_audit = _project_to_joint_feasible_polytope(
            n_candidate=target_nutrients,
            lower=lower,
            upper=upper,
            raw_vector=raw_vector,
            metadata=metadata,
            p_cvd_0=p_cvd_0,
            raise_on_infeasible=True,
        )
    else:
        projection_audit = {
            'status': 'legacy_bypass',
            'passed': True,
            'feasible': True,
        }

    # STEP 5: four sequential verification filters.
    filter1 = _apply_filter_1_thermodynamic_balance(target_nutrients)'''
assert target_gen1 in code
code = code.replace(target_gen1, replace_gen1, 1)

target_gen2 = '''    audit_filters = {
        "filter_1_thermodynamic_balance": {'''
replace_gen2 = '''    audit_filters = {
        "joint_projection": projection_audit,
        "filter_1_thermodynamic_balance": {'''
assert target_gen2 in code
code = code.replace(target_gen2, replace_gen2, 1)

target_gen3 = '''    certified = bool(
        filter1_final and filter2_final and filter3_final and filter4_passed
    )'''
replace_gen3 = '''    certified = bool(
        filter1_final and filter2_final and filter3_final and filter4_passed and projection_audit.get('passed', True)
    )'''
assert target_gen3 in code
code = code.replace(target_gen3, replace_gen3, 1)

target_gen4 = '''        "nutrient_ruleset_version": RULESET_VERSION,
        "clinical_prescribing_validated": False,
        "patient_baseline": {'''

replace_gen4 = '''        "nutrient_ruleset_version": RULESET_VERSION,
        "clinical_prescribing_validated": False,
        "deployment_policy": deployment.metadata(),
        "counterfactual_gradient_scope": "Exploratory surrogate gradients through six-decimal forward rounding; not a causal intervention model." if deployment.REPAIRED else "Legacy exploratory optimization",
        "patient_baseline": {'''
assert target_gen4 in code
code = code.replace(target_gen4, replace_gen4, 1)

target_gen5 = '''        "audit_filters": audit_filters,
        "certified_for_stage3": certified,'''
replace_gen5 = '''        "audit_filters": audit_filters,
        "audit_projection": projection_audit,
        "certified_for_stage3": certified,'''
assert target_gen5 in code
code = code.replace(target_gen5, replace_gen5, 1)

with open('stage2_counterfactual_engine.py', 'w', encoding='utf-8', newline='\r\n') as f:
    f.write(code)

print('Full unified patch applied successfully!')
