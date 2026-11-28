"""Versioned engineering interpretation of the project's nutrient rules."""
from __future__ import annotations

import math

MUFA_POLICY_VERSION = 'mufa-component-envelope-v2'
MUFA_SOURCE = 'https://www.fao.org/4/i1953e/i1953e00.pdf'
RULESET_VERSION = 'adult-food-rule-review-v4-exposure-separated'
# Required only by the finite-vector search implementation. This is not a
# tolerable intake, recommended target, or a safety allowance.
NUMERICAL_SEARCH_CAP = 1_000_000.0

REPORT_ONLY = {'DR1TSUGR', 'DR1TFOLA', 'DR1TFF', 'DR1TACAR', 'DR1TBCAR', 'DR1TLYCO', 'DR1TLZ'}
EXPOSURE_REPORTING = {'DR1TCAFF', 'DR1TALCO'}
ODS = 'https://ods.od.nih.gov/factsheets/{}-HealthProfessional/'
MACRO_SOURCE = 'https://ods.od.nih.gov/HealthInformation/nutrientrecommendations.aspx'
SUGAR_SOURCE = 'https://www.who.int/publications/i/item/9789241549028'


def reviewed_default_bounds(bounds: dict, age: float, female: bool) -> dict:
    """Correct unambiguous food/reference mismatches in the legacy defaults.

    These are US reference values for generally healthy, nonpregnant adults,
    not an Indian clinical protocol. Patient-specific prescribing is not validated.
    """
    b = dict(bounds)
    cap = NUMERICAL_SEARCH_CAP
    b.update({
        'DR1TATOC': (15.0, cap),  # Supplemental UL does not constrain ordinary foods.
        'DR1TRET': (0.0, 2800.0 if age < 19 else 3000.0),
        'DR1TVARA': (700.0 if female else 900.0, cap),
        'DR1TACAR': (0.0, cap), 'DR1TBCAR': (0.0, cap),
        'DR1TLYCO': (0.0, cap), 'DR1TLZ': (0.0, cap),
        'DR1TVB1': ((1.0 if age < 19 else 1.1) if female else 1.2, cap),
        'DR1TVB2': ((1.0 if age < 19 else 1.1) if female else 1.3, cap),
        'DR1TNIAC': (14.0 if female else 16.0, cap),
        'DR1TFOLA': (0.0, cap), 'DR1TFF': (0.0, cap),
        'DR1TFDFE': (400.0, cap),
        'DR1TFA': (0.0, 800.0 if age < 19 else 1000.0),
        'DR1TVB12': (2.4, cap),
        'DR1TVK': (75.0 if age < 19 else (90.0 if female else 120.0), cap),
        'DR1TCALC': (1300.0 if age < 19 else (1200.0 if age >= 71 or (female and age >= 51) else 1000.0),
                      3000.0 if age < 19 else (2000.0 if age >= 51 else 2500.0)),
        'DR1TPHOS': (1250.0 if age < 19 else 700.0, 3000.0 if age >= 71 else 4000.0),
        'DR1TMAGN': ((360.0 if female else 410.0) if age < 19 else
                      ((310.0 if female else 400.0) if age < 31 else (320.0 if female else 420.0)), cap),
        'DR1TIRON': ((15.0 if female else 11.0) if age < 19 else (18.0 if female and age < 51 else 8.0), 45.0),
        'DR1TZINC': (9.0 if female and age < 19 else (8.0 if female else 11.0), 34.0 if age < 19 else 40.0),
        'DR1TCOPP': (0.89 if age < 19 else 0.9, 8.0 if age < 19 else 10.0),
    })
    b['DR1TVB6'] = ((1.2 if female else 1.3) if age < 19 else
                     ((1.5 if female else 1.7) if age >= 51 else 1.3), 80.0 if age < 19 else 100.0)
    b['DR1TVC'] = ((65.0 if female else 75.0) if age < 19 else (75.0 if female else 90.0),
                   1800.0 if age < 19 else 2000.0)
    return b


def reviewed_rule_metadata(code: str, low: float, high: float) -> dict:
    """One explicit review entry for each of the 39 dietary fields."""
    micro = {
        'DR1TATOC': ('VitaminE', 'RDA reference; supplemental UL is not a food ceiling.'),
        'DR1TRET': ('VitaminA', 'Preformed vitamin A UL screening; include supplements and ester forms for a full assessment.'),
        'DR1TVARA': ('VitaminA', 'RAE adequacy reference; preformed-vitamin-A UL must not cap total RAE.'),
        'DR1TVB1': ('Thiamin', 'RDA reference; no established UL.'),
        'DR1TVB2': ('Riboflavin', 'RDA reference; no established UL.'),
        'DR1TNIAC': ('Niacin', 'Food niacin is a proxy for NE; tryptophan contribution is unavailable. Synthetic/supplemental UL is not a whole-food ceiling.'),
        'DR1TVB6': ('VitaminB6', 'Age/sex US RDA and UL reference; this is not a universally agreed international ceiling.'),
        'DR1TFOLA': ('Folate', 'Total folate reporting only; 400 ug DFE is not a total-folate requirement.'),
        'DR1TFA': ('Folate', 'Synthetic folic acid ceiling requires counting fortified foods and supplements; food basket alone is incomplete exposure.'),
        'DR1TFF': ('Folate', 'Food folate reporting only; no independent duplicate adequacy target or natural-food UL.'),
        'DR1TFDFE': ('Folate', 'DFE adequacy reference; the folic-acid UL is expressed in ug folic acid, not DFE.'),
        'DR1TVB12': ('VitaminB12', 'RDA reference; no established UL; missing data remain a priority gap.'),
        'DR1TVC': ('VitaminC', 'Age/sex reference; smoking-related additional needs are not implemented.'),
        'DR1TVK': ('VitaminK', 'K1 used as a proxy for the vitamin K AI, not an RDA; no established UL; vitamin K antagonist therapy needs review.'),
        'DR1TCALC': ('Calcium', 'Age/sex RDA and UL reference; include other foods and supplements.'),
        'DR1TPHOS': ('Phosphorus', 'Age-dependent RDA and UL reference; renal disease requires separate assessment.'),
        'DR1TMAGN': ('Magnesium', 'Age/sex RDA reference; supplemental magnesium UL is not a food magnesium ceiling.'),
        'DR1TIRON': ('Iron', 'Age/sex RDA and UL reference; menstrual status and vegetarian bioavailability need review.'),
        'DR1TZINC': ('Zinc', 'Age/sex reference; bioavailability and supplements may alter interpretation.'),
        'DR1TCOPP': ('Copper', 'Age-dependent RDA and UL reference.'),
        'DR1TSELE': ('Selenium', 'US RDA and UL reference; other authorities can differ.'),
        'DR1TPOTA': ('Potassium', 'Legacy project interval; AI differs by sex. Screening-score-based potassium escalation requires clinical review.'),
    }
    macros = {
        'DR1TKCAL': 'Project energy prescription with 5% tolerance; not an RDA or clinical safety proof.',
        'DR1TPROT': 'Project 0.8–2 g/kg interval plus sex-specific floor; upper endpoint is not a UL. Renal status is unavailable.',
        'DR1TCARB': '130 g adult reference plus project score-dependent upper fraction; no universally optimal diabetes carbohydrate percentage.',
        'DR1TSUGR': 'Report total sugar separately; WHO free-sugar limits cannot be applied directly to total sugar.',
        'DR1TFIBE': 'Project fiber goal and computational upper bound; upper endpoint is not an established UL.',
        'DR1TTFAT': '20–35% energy adult reference interval; applicability to this patient and current guidance requires review.',
        'DR1TSFAT': 'Project risk-stratified upper fraction; lower value is not an individualized medical prescription.',
        'DR1TMFAT': 'Nonnegative component bounded by total-fat ceiling; not a separate MUFA RDA or UL.',
        'DR1TPFAT': 'Project total-PUFA interval; not a substitute for separate essential LA/ALA requirements.',
        'DR1TCHOL': 'Legacy project cholesterol limits; 150/300 mg are not universally established ULs.',
        'DR1TSODI': 'Legacy project sodium interval; 500 mg floor and risk-score choice of ceiling require review.',
        'DR1TCAFF': '400 mg FDA general adult exposure reference; not universal safety, and drinks outside the plan are omitted.',
        'DR1TALCO': 'Legacy project exposure restriction, not a safe drinking allowance or recommendation to drink.',
    }
    if code in {'DR1TACAR', 'DR1TBCAR', 'DR1TLYCO', 'DR1TLZ'}:
        source, note = ODS.format('VitaminA'), 'Research/reporting field; no independent adequacy minimum or supported clinical ceiling is configured.'
    elif code in micro:
        name, note = micro[code]
        source = ODS.format(name)
    elif code in macros:
        note = macros[code]
        source = SUGAR_SOURCE if code == 'DR1TSUGR' else (MUFA_SOURCE if code in {'DR1TMFAT','DR1TPFAT','DR1TSFAT'} else MACRO_SOURCE)
        if code == 'DR1TCAFF': source = 'https://www.fda.gov/consumers/consumer-updates/spilling-beans-how-much-caffeine-too-much'
    else:
        raise ValueError(f'Unreviewed nutrient {code}')
    role = (
        'report_only' if code in REPORT_ONLY
        else 'exposure_reporting' if code in EXPOSURE_REPORTING
        else ('project_assumption' if code in macros and code != 'DR1TMFAT' or code == 'DR1TPOTA' else 'reference_interval')
    )
    gram_fields = {'DR1TPROT','DR1TCARB','DR1TSUGR','DR1TFIBE','DR1TTFAT','DR1TSFAT','DR1TMFAT','DR1TPFAT','DR1TALCO'}
    ug_fields = {'DR1TRET','DR1TVARA','DR1TACAR','DR1TBCAR','DR1TLYCO','DR1TLZ','DR1TFOLA','DR1TFA','DR1TFF','DR1TFDFE','DR1TVB12','DR1TVK','DR1TSELE'}
    lower_kind = ('not_an_adequacy_minimum' if role in {'report_only', 'exposure_reporting'} or low == 0 else
                  'project_choice_requires_review' if role == 'project_assumption' else
                  'AI_reference_proxy' if code == 'DR1TVK' else
                  'RDA_reference_proxy' if code == 'DR1TNIAC' else 'RDA_reference')
    hard_or_soft = (
        'report_only' if role == 'report_only'
        else 'exposure_reporting' if role == 'exposure_reporting'
        else 'hard'
    )
    relaxation_policy = (
        'not_enforced_reporting' if role == 'report_only'
        else 'tracked_separately_as_beverage_exposure' if role == 'exposure_reporting'
        else 'protected_no_chat_relaxation'
    )
    return {'nutrient': code, 'ruleset_version': RULESET_VERSION, 'source_url': source,
            'source_support': 'reference_context_not_endorsement_of_project_threshold' if role == 'project_assumption' else 'see_definition_and_scope_note',
            'unit': 'kcal' if code == 'DR1TKCAL' else ('g' if code in gram_fields else 'ug' if code in ug_fields else 'mg'),
            'reference_population': 'Generally healthy, nonpregnant/nonlactating age 18+; age-specific US reference, not an Indian clinical protocol.',
            'applicability': 'Generally healthy, nonpregnant/nonlactating adults (18+); excludes renal, gestational, and pediatric prescribing.',
            'planning_role': role,
            'hard_or_soft': hard_or_soft,
            'relaxation_policy': relaxation_policy,
            'stage3_enforcement': 'none' if role in {'report_only', 'exposure_reporting'} else 'required_for_acceptance',
            'numeric_lower': float(low), 'numeric_upper': float(high),
            'lower_kind': lower_kind,
            'upper_kind': 'numerical_search_cap_not_UL' if high == NUMERICAL_SEARCH_CAP else 'reference_or_project_bound_see_note',
            'clinical_prescribing_validated': False, 'review_note': note}


def mufa_component_bounds(total_fat_upper_g: float) -> tuple[float, float]:
    """No standalone MUFA minimum; retain the total-fat component ceiling.

    FAO Food and Nutrition Paper 91, pp. 15 and 58 describes MUFA by
    difference from total fat and other fatty acids, not a fixed 10–20% energy
    range. Zero is a nonnegativity bound, not a recommended intake. The upper
    bound is bookkeeping, not a MUFA tolerable upper intake level.
    """
    upper = float(total_fat_upper_g)
    if not math.isfinite(upper) or upper <= 0:
        raise ValueError('Total-fat upper bound must be positive and finite.')
    return 0.0, upper


def mufa_rule_metadata() -> dict:
    return {'policy_version': MUFA_POLICY_VERSION,
            'source_url': MUFA_SOURCE, 'source_pages': '15, 58 (printed pages)',
            'interpretation': 'No independent MUFA adequacy minimum; nonnegative component bounded above by the existing total-fat ceiling.',
            'clinical_claim': 'Neither endpoint is a MUFA RDA or UL.'}
