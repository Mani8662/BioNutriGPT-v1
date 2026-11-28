"""Isolated research rule adapter; no trained model loading or production mutation.

Explicit condition flags replace legacy probability proxies. Rule equations are
copied from the reviewed production envelope and tested for branch parity.
Predicted flags are experimental policy inputs, not diagnoses.
"""
from typing import Any, Mapping, Sequence
import math
import numpy as np
from stage2_counterfactual_engine import (NUTRIENT_INDEX, DIET_RAW_COLS,
    UNSPECIFIED_BOUND_NUTRIENTS, Stage2Error, ArtifactContractError,
    get_default_micronutrient_bounds, _as_float, PAL_FACTORS)
from nutrient_rule_policy import mufa_component_bounds, reviewed_rule_metadata, RULESET_VERSION

def build_flag_bounds(
    *,
    artifacts: Mapping[str, Any],
    t_kcal: float,
    weight_kg: float,
    is_female: bool,
    age_years: float,
    diabetes_rule: bool,
    obesity_rule: bool,
    cvd_rule: bool,
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.full(39, np.nan, dtype=np.float64)
    upper = np.full(39, np.nan, dtype=np.float64)

    def put(name: str, low: float, high: float) -> None:
        idx = NUTRIENT_INDEX[name]
        lower[idx] = float(low)
        upper[idx] = float(high)

    put("DR1TKCAL", t_kcal * 0.95, t_kcal * 1.05)
    prot_min = max(0.8 * weight_kg, 46.0 if is_female else 56.0)
    prot_max = 2.0 * weight_kg
    if prot_min > 2.0 * weight_kg:
        raise Stage2Error(
            f"Conflicting protein rules: minimum {prot_min:g} g exceeds "
            f"the filter maximum {2.0 * weight_kg:g} g. No prescription generated."
        )
    put("DR1TPROT", prot_min, prot_max)

    carb_fraction = 0.45 if diabetes_rule else 0.60
    carb_max = max(135.0, (carb_fraction * t_kcal) / 4.0)
    put("DR1TCARB", 130.0, carb_max)
    # Total sugar is not WHO free/added sugar. Keep a component search domain,
    # then report it without applying the former 25/50 g therapeutic claim.
    put("DR1TSUGR", 0.0, carb_max)
    fiber_min = max(
        35.0 if diabetes_rule else 28.0,
        (14.0 * t_kcal) / 1000.0,
    )
    fiber_max = max(60.0, fiber_min + 10.0)
    put("DR1TFIBE", fiber_min, fiber_max)
    put("DR1TTFAT", (0.20 * t_kcal) / 9.0, (0.35 * t_kcal) / 9.0)
    put(
        "DR1TSFAT", 0.0,
        ((0.06 if cvd_rule else 0.10) * t_kcal) / 9.0,
    )
    # The previous 10–20% energy interval was an unsupported independent
    # adequacy rule. MUFA is a component of fat, not a separate minimum target.
    put("DR1TMFAT", *mufa_component_bounds(float(upper[NUTRIENT_INDEX['DR1TTFAT']])))
    put("DR1TPFAT", (0.05 * t_kcal) / 9.0, (0.10 * t_kcal) / 9.0)
    put("DR1TCHOL", 0.0, 150.0 if cvd_rule else 300.0)
    put("DR1TSODI", 500.0, 1500.0 if cvd_rule else 2300.0)
    put("DR1TPOTA", 3500.0 if cvd_rule else 2600.0, 4700.0)
    alcohol_upper = 0.0 if (cvd_rule or obesity_rule) else 14.0
    put("DR1TALCO", 0.0, alcohol_upper)

    # Built-in values make the engine self-contained. Artifact entries, when
    # present, override matching defaults; partial mappings remain supported.
    unspecified = get_default_micronutrient_bounds(age_years, is_female)
    artifact_bounds = artifacts.get("nutrient_bounds")
    if callable(artifact_bounds):
        try:
            artifact_bounds = artifact_bounds(age_years, is_female)
        except Exception as exc:
            raise ArtifactContractError(
                "artifact['nutrient_bounds'] callable failed for the supplied age/sex."
            ) from exc
    if isinstance(artifact_bounds, Mapping):
        unspecified.update(artifact_bounds)

    for name in UNSPECIFIED_BOUND_NUTRIENTS:
        bound = unspecified[name]
        if not isinstance(bound, Sequence) or isinstance(bound, (str, bytes)) or len(bound) != 2:
            raise ArtifactContractError(
                f"nutrient_bounds[{name!r}] must be a two-value (lower, upper) sequence."
            )
        low = _as_float(bound[0], f"nutrient_bounds[{name}].lower")
        high = _as_float(bound[1], f"nutrient_bounds[{name}].upper")
        if low < 0.0 or high < 0.0:
            raise ArtifactContractError(f"nutrient_bounds[{name}] must be nonnegative.")
        idx = NUTRIENT_INDEX[name]
        lower[idx], upper[idx] = low, high

    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        missing_idx = np.flatnonzero(~(np.isfinite(lower) & np.isfinite(upper))).tolist()
        raise ArtifactContractError(f"Unpopulated nutrient bound indices: {missing_idx}.")
    invalid = [DIET_RAW_COLS[i] for i in range(39) if lower[i] > upper[i]]
    if invalid:
        raise Stage2Error(f"Infeasible nutrient bounds (lower > upper): {invalid}.")
    return lower.astype(np.float32), upper.astype(np.float32)

def build_research_prescription(profile, *, probabilities=None, thresholds=None, arm='rules_only'):
    """Build an envelope; no counterfactual optimization or certification claim."""
    if arm not in ('rules_only','bionutrigpt','catboost','ft_transformer'):
        raise ValueError('Unknown research arm')
    required=('weight_kg','height_cm','age_years','is_female','activity_level','has_diabetes','has_cvd')
    if any(k not in profile for k in required): raise ValueError('Missing required observed metadata')
    for key in ('has_diabetes','has_cvd','renal_disease','pregnant','lactating'):
        if profile.get(key) is not None and type(profile[key]) is not bool: raise ValueError('Clinical flags must be bool or None')
    if type(profile['is_female']) is not bool: raise ValueError('Sex flag must be bool')
    w,h,age=(float(profile[k]) for k in ('weight_kg','height_cm','age_years'))
    if not all(math.isfinite(x) and x>0 for x in (w,h,age)) or age<20: raise ValueError('Outside adult research scope')
    if any(profile.get(k) is True for k in ('renal_disease','pregnant','lactating')): raise ValueError('Excluded clinical condition')
    if profile['activity_level'] not in PAL_FACTORS: raise ValueError('Unknown activity mapping')
    if arm=='rules_only':
        if probabilities is not None or thresholds is not None: raise ValueError('Rules-only arm rejects prediction inputs')
    else:
        if probabilities is None or thresholds is None: raise ValueError('Explicit calibrated probabilities and experimental thresholds required')
        for condition in ('diabetes','cvd'):
            p=float(probabilities[condition]);t=float(thresholds[condition])
            if not math.isfinite(p) or not 0<=p<=1 or not math.isfinite(t) or not 0<t<1: raise ValueError('Invalid probability or threshold')
    flags={};reasons={}
    for condition in ('diabetes','cvd'):
        known=profile['has_'+condition]
        prediction_trigger=arm!='rules_only' and float(probabilities[condition])>=float(thresholds[condition])
        flags[condition]=known is True or prediction_trigger
        reasons[condition]={'observed_history':known,'prediction_trigger':prediction_trigger,'selected_rule':flags[condition],
            'source':'reported_diagnosis' if known is True else 'experimental_prediction' if prediction_trigger else 'no_condition_specific_branch',
            'probability':None if arm=='rules_only' else float(probabilities[condition]),
            'threshold':None if arm=='rules_only' else float(thresholds[condition])}
    bmi=w/(h/100)**2; obese=bmi>=30
    bmr=10*w+6.25*h-5*age+(-161 if profile['is_female'] else 5)
    tdee=bmr*PAL_FACTORS[profile['activity_level']];kcal=max(1200.,tdee-(500. if obese else 0.))
    lower,upper=build_flag_bounds(artifacts={},t_kcal=kcal,weight_kg=w,is_female=profile['is_female'],age_years=age,diabetes_rule=flags['diabetes'],obesity_rule=obese,cvd_rule=flags['cvd'])
    # Baseline dietary values must be supplied for the solver's ratio checks.
    baseline=profile.get('baseline_nutrients')
    envelope=[]
    for i,code in enumerate(DIET_RAW_COLS):
        row={'nutrient':code,'min_bound':float(lower[i]),'max_bound':float(upper[i]),'rule_provenance':reviewed_rule_metadata(code,float(lower[i]),float(upper[i]))}
        if baseline is not None:
            value=float(baseline[code])
            if not math.isfinite(value) or value<0: raise ValueError('Invalid measured baseline nutrient')
            row['baseline']=value
        envelope.append(row)
    return {'research_only':True,'arm':arm,'patient_metadata':{k:profile[k] for k in required},
        'clinical_unknowns':[k for k in ('has_diabetes','has_cvd','renal_disease','pregnant','lactating') if profile.get(k) is None],
        'rule_decisions':reasons,'measured_bmi':bmi,'obesity_rule':obese,
        'obesity_rule_source':'BMI computed from measured height and weight; predicted obesity ignored',
        'energy_prescription':{'BMR':bmr,'TDEE':tdee,'target_calories':kcal,'deficit_kcal':500. if obese else 0.},
        'nutrient_ruleset_version':RULESET_VERSION,'constraint_spec_version':RULESET_VERSION,
        'nutrient_constraint_envelope':envelope,'certified_for_stage3':False,
        'status':'research envelope only; independent feasibility/certificate and explicit target objective required',
        'policy_scope':'Experimental rule branching; not a clinical recommendation or causal risk reduction'}
