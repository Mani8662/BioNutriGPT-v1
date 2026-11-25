"""Create a feature-level compatibility report from inspected local LASI metadata."""
from pathlib import Path
from collections import Counter
import json

R=Path(__file__).resolve().parent
O=R/'Saved Model/indian_validation_compatibility_v1'
data=json.loads((O/'lasi_metadata.json').read_text(encoding='utf-8'))
variables={v['name']:v for v in data['metadata']['3_LASI_W1_Individual_v4.sav']['variables']}
rows=[]
for group,features in data['model_feature_groups'].items():
    for feature in features:
        sources=[]
        if group=='BIO_COLS':
            status='unavailable';reason='No measured HDL or total cholesterol in local participant files; diagnosis ht010 is not a lipid assay.'
        elif group=='DIET_COLS':
            status='unavailable';reason='No individual 24-hour nutrient totals; food spending/frequency cannot supply nutrient grams or derived ratios.'
        elif feature=='RIDAGEYR':
            sources=['dm005'];status='direct_candidate';reason='Age in years; harmonize age eligibility and NHANES age top-coding before use.'
        elif feature=='SEX_FEMALE':
            sources=['dm003'];status='direct_candidate';reason='LASI 1=male, 2=female; derive binary only for valid responses.'
        elif feature.startswith('RACE_'):
            status='incompatible';reason='US race/ethnicity categories are not interchangeable with Indian caste, religion or nationality; no automatic other/multiracial assignment.'
        elif feature.endswith('_MISS'):
            status='pipeline_derived';reason='Must be produced by a frozen, justified missing-input policy; not a measured LASI field or evidence of compatibility.'
        elif feature=='INDFMPIR':
            status='incompatible';reason='US family income-to-poverty ratio is not equivalent to Indian household income or expenditure.'
        elif feature=='DMDEDUC2':
            sources=['dm008'];status='harmonization_required';reason='Education systems/codes differ; review no-schooling skip paths and prespecify shared broad categories.'
        elif feature.startswith('MARITAL_'):
            sources=['dm021'];status='harmonization_required';reason='Map explicit categories after checking live-in, deserted and missing responses against NHANES coding.'
        elif feature.startswith('SLEEP_'):
            sources=['tu002'];status='non_equivalent_proxy';reason='Yesterday sleep is not automatically equivalent to usual weekday/workday sleep; units and recall window need questionnaire review.'
        elif feature=='SLQ050':
            sources=['ht219'];status='non_equivalent_proxy';reason='Trouble falling asleep frequency differs from telling a health professional about trouble sleeping.'
        elif feature.startswith('SMOKE_'):
            sources=['hb001','hb003','hb003_a'];status='non_equivalent_proxy';reason='Any smoking/smokeless tobacco history is not NHANES cigarette-history and current frequency definition; do not use es010 vignette-person variables.'
        elif feature=='PAD680':
            status='unavailable';reason='No directly equivalent respondent total daily sedentary minutes identified; difficulty sitting is not sedentary time.'
        else:
            sources=['hb211','hb212','hb213','hb214'];status='non_equivalent_proxy';reason='General activity frequency/duration does not identify NHANES work/recreation/transport domains; coarse frequency cannot supply exact weekly minutes.'
        assert all(s in variables for s in sources)
        rows.append({'feature':feature,'group':group,'status':status,'source_variables':sources,
                     'source_labels':{s:variables[s]['label'] for s in sources},'reason':reason})
assert len(rows)==len({r['feature'] for r in rows})==87
counts=dict(Counter(r['status'] for r in rows))
result={'date':'2026-11-22','scope':'Local LASI archive and current repaired research schema; no external predictions or training',
        'archive_sha256':data['archive_sha256'],'feature_status_counts':counts,'features':rows,
        'decision':'Current 87-input primary-outcome external validation not supported by the available LASI files',
        'carrs':'No CARRS participant data or approved dictionary located in repository; published-domain overlap is not verified variable compatibility'}
(O/'feature_mapping.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
c=data['availability_counts']
report=f'''# Indian prediction validation compatibility audit

Updated 22 November 2026. The local LASI archive does not support a like-for-like external validation of the current 87-input Stage 1 research model. No predictions, retraining or clinical validation were performed.

## Data inspected

Inspected SPSS metadata for six files, including all 2,623 individual and 91 biomarker variables. Individual records: 73,396; biomarker records: 66,859. Their participant identifiers were nonmissing and unique; the one-to-one join contained {c['merged_one_to_one']:,} records. The community file crashed pyreadstat 1.3.8 on Python 3.14 with an access violation and remains unparsed; this limitation does not supply the missing individual assays.

Archive SHA-256: `{data['archive_sha256']}`. Metadata, labels, aggregate counts and all 87 feature decisions are saved under `Saved Model/indian_validation_compatibility_v1/`. No participant rows or IDs were exported by this audit. The repository search located CARRS design documents but no CARRS participant data or approved data dictionary. The older CARRS audit concerns a proposed redesign and does not establish current compatibility.

## Predictor compatibility

The 5 lipid features and all 46 nutrient/derived diet features are unavailable in these local files. No HbA1c assay field was identified. A diagnosis of high cholesterol is not a measured cholesterol value. Household food spending and food-frequency information cannot be substituted for individual nutrient totals.

Feature classifications: {json.dumps(counts)}. These counts describe conservative mapping decisions, not completed transformations. Only age and sex are direct candidates; even age requires review of top-coding. Missingness flags are generated by preprocessing, not independently available predictors.

## Outcome compatibility

| Outcome | Repaired NHANES research definition | Available LASI evidence | Decision |
|---|---|---|---|
| Diabetes | Reported diagnosis OR valid HbA1c at least 6.5%; a negative requires below-threshold HbA1c and compatible diagnosis response | ht003 reported diagnosis; no HbA1c assay in this local release | Cannot reproduce primary phenotype. A reported-diagnosis-only study would change the endpoint and miss undiagnosed disease. |
| CVD | Heart failure, coronary heart disease, angina, heart attack or stroke | ht006 broad chronic heart disease; ht007 stroke; ht006a heart attack; ht006fs4 heart failure and ht006fs5 coronary disease | Partial overlap; broad ht006 also includes rheumatic, congenital and rhythm disease. Angina and skip/negative ascertainment need questionnaire review. Do not equate ht006 OR ht007 with the current endpoint. |
| Obesity | Measured BMI at least 30; secondary exploratory endpoint | bm067 height in cm and bm071 weight in kg | Label can be derived after measurement-quality review. This does not repair missing model inputs. BMI at least 25 is a different endpoint. |

HbA1c, if subsequently obtained, would be used to construct the reference outcome, not fed back into the model as a predictor. Height/weight/BMI must not be added as predictors of a BMI-defined obesity endpoint.

## Aggregate availability

These are unweighted availability counts, not a final eligible cohort or nationally representative prevalence. Audit filters are age 20–100 and valid male/female coding; anthropometry additionally uses height 120–210 cm, weight 30–180 kg and BMI 14–55 as engineering plausibility checks. Final exclusions must consider measurement-compliance variables and the LASI questionnaire.

| Check | Count |
|---|---:|
| Joined records meeting audit age/sex criteria | {c['age20_100_valid_sex']:,} |
| Age 45 or older among those records | {c['age45_plus_within_audit_adults']:,} |
| Plausible measured anthropometry | {c['valid_anthropometry']:,} |
| BMI at least 30 among plausible measurements | {c['bmi30_positive']:,} |
| BMI below 30 among plausible measurements | {c['bmi30_negative']:,} |
| Reported diabetes yes / no / unknown | {c['diabetes_selfreport_yes']:,} / {c['diabetes_selfreport_no']:,} / {c['diabetes_selfreport_unknown']:,} |
| Broad chronic heart disease or stroke yes / both no | {c['broad_heart_or_stroke_yes']:,} / {c['broad_heart_or_stroke_no']:,} |

LASI India individual weights and ssuid are present; the correct strata, cluster structure, module eligibility and nonresponse weights still require the survey documentation. The existing balanced 100-profile planning sample is unsuitable for population calibration assessment or representative prediction-performance claims.

## Decision and next steps

1. Retain LASI as downstream Stage 2–4 planning evidence. Do not label those results Stage 1 Indian external validation.
2. Request the linkable LASI HbA1c/DBS release with assay documentation and sampling weights, or the approved CARRS population-cohort dictionary and participant files. HbA1c alone fixes only part of the label gap, not missing diet/lipids. The CARRS treatment trial restricted to diabetes patients cannot establish diabetes case/noncase discrimination.
3. For CARRS, request baseline age/sex, HDL/total cholesterol with units, individual nutrient totals and recall method/subsample flags, smoking/activity/sleep definitions, reported diagnosis/HbA1c, individual CVD components, measured anthropometry, visit identifiers and design variables. Evaluate the Indian sites separately from non-Indian sites. Match the present prevalent endpoint; incident disease prediction is a different study.
4. If adequate full-model inputs remain unavailable, design a separate shared-variable model in NHANES. Age and sex are the only immediately clear candidates; education/marital categories and lifestyle constructs need questionnaire-level harmonization before inclusion. This is not yet a scientifically sufficient replacement model. A diagnosis-only endpoint must be explicitly named and aligned on both datasets.
5. Freeze features, labels, exclusions and missingness rules before accessing model performance. Train/tune/calibrate only on NHANES development partitions; test once on the eligible Indian cohort. Any Indian recalibration needs a separate adaptation subset and untouched evaluation subset. Compare frozen baseline models using identical records and report calibration, discrimination, confidence intervals and population scope.

Do not fill absent nutrient/lipid blocks with fabricated values, convert Indian social categories into US race flags, use hypothetical-person vignette answers as respondent smoking, or change thresholds to improve reported performance. A defined missing-modality transfer experiment may be explored separately, but it would not be a like-for-like full-input validation and would still require compatible outcomes.

## Feature by feature mapping

| Feature | Status | LASI candidates | Reason |
|---|---|---|---|
'''
for row in rows:
    report+=f"| {row['feature']} | {row['status']} | {', '.join(row['source_variables']) or 'None'} | {row['reason']} |\n"
p=R/'Research Paper/Indian_Validation_Compatibility_v1.md'
p.write_text(report,encoding='utf-8')
print(json.dumps({'feature_status_counts':counts,'availability_counts':c},indent=2))
