import re

with open('stage2_counterfactual_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Error class
target1 = '''class ArtifactContractError(Stage2Error):
    """Raised when a preprocessing/model artifact is missing required data."""'''
replace1 = target1 + '''


class Stage2InfeasibleConstraintsError(Stage2Error):
    """Raised when the joint nutrient constraint polytope is mathematically empty."""'''
assert target1 in content, 'target1 not found'
content = content.replace(target1, replace1, 1)

# 2. Add DAILY_REFERENCE_SCALES and projection helper functions before Filter 1
target2 = 'def _apply_filter_1_thermodynamic_balance(n: np.ndarray) -> dict[str, Any]:'
scales_and_proj = '''# Standard adult daily intake scales (US FDA Daily Values / Dietary Reference Intakes).
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
content = content.replace(target2, scales_and_proj + target2, 1)

# 3. Add projection call into generate_counterfactual_prescription
target3 = '''    # STEP 5: four sequential verification filters.
    filter1 = _apply_filter_1_thermodynamic_balance(target_nutrients)'''
replace3 = '''    # STEP 4.5: Joint constraint projection.
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
assert target3 in content, 'target3 not found'
content = content.replace(target3, replace3, 1)

# 4. Include joint_projection in audit_filters
target4 = '''    audit_filters = {
        "filter_1_thermodynamic_balance": {'''
replace4 = '''    audit_filters = {
        "joint_projection": projection_audit,
        "filter_1_thermodynamic_balance": {'''
assert target4 in content, 'target4 not found'
content = content.replace(target4, replace4, 1)

# 5. certified boolean
target5 = '''    certified = bool(
        filter1_final and filter2_final and filter3_final and filter4_passed
    )'''
replace5 = '''    certified = bool(
        filter1_final and filter2_final and filter3_final and filter4_passed and projection_audit.get('passed', True)
    )'''
assert target5 in content, 'target5 not found'
content = content.replace(target5, replace5, 1)

# 6. Dossier audit_projection field
target6 = '''        "audit_filters": audit_filters,
        "certified_for_stage3": certified,'''
replace6 = '''        "audit_filters": audit_filters,
        "audit_projection": projection_audit,
        "certified_for_stage3": certified,'''
assert target6 in content, 'target6 not found'
content = content.replace(target6, replace6, 1)

with open('stage2_counterfactual_engine.py', 'w', encoding='utf-8', newline='\r\n') as f:
    f.write(content)

print('Patch applied successfully!')
