import { FeatureFieldInfo } from '../types/api'

export const FEATURE_CATEGORIES = [
  'Demographics & Vitals',
  'Serum Biomarkers',
  'Dietary Macronutrients',
  'Vitamins & Micronutrients',
  'Minerals & Bioactives',
  'Dietary Ratios & AMDR',
  'Lifestyle & Physical Activity',
  'Data Quality & Missingness'
] as const

export const CANONICAL_SAMPLE_FEATURES: number[] = [
  56.0, 178.0, 122.0, 3.178571, 2.178571, 1990.0, 90.0, 250.0, 42.0, 30.0,
  70.0, 18.0, 28.0, 17.0, 220.0, 10.5, 350.0, 900.0, 500.0, 3200.0,
  6000.0, 1800.0, 1.4, 1.6, 18.0, 1.8, 500.0, 120.0, 380.0, 584.0,
  4.2, 100.0, 120.0, 950.0, 1400.0, 380.0, 14.0, 11.0, 1.2, 1900.0,
  3500.0, 90.0, 80.0, 0.0, 18.09, 50.25, 31.65, 15.07, 1.35, 0.54,
  0.39, 7.5, 0.0, 300.0, 0.0, 120.0, 60.0, 60.0, 60.0, 1440.0,
  0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0,
  42.0, 3.2, 5.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
]

export const CANONICAL_SAMPLE_METADATA = {
  weight_kg: 74.0,
  height_cm: 176.0,
  age_years: 42.0,
  is_female: false,
  activity_level: 'sedentary' as const
}

export const FEATURE_DEFINITIONS: FeatureFieldInfo[] = [
  // 1. Serum Biomarkers (5)
  { name: 'LBDHDD', label: 'HDL Cholesterol', category: 'Serum Biomarkers', unit: 'mg/dL', default_val: 56.0, min_val: 10, max_val: 150, step: 1, description: 'High-density lipoprotein ("good") cholesterol.' },
  { name: 'LBXTC', label: 'Total Cholesterol', category: 'Serum Biomarkers', unit: 'mg/dL', default_val: 178.0, min_val: 50, max_val: 500, step: 1, description: 'Total serum cholesterol concentration.' },
  { name: 'NON_HDL_CHOL', label: 'Non-HDL Cholesterol', category: 'Serum Biomarkers', unit: 'mg/dL', default_val: 122.0, min_val: 30, max_val: 450, step: 1, description: 'Total cholesterol minus HDL; primary atherogenic marker.' },
  { name: 'CHOL_HDL_RATIO', label: 'Total / HDL Ratio', category: 'Serum Biomarkers', unit: 'ratio', default_val: 3.18, min_val: 1.0, max_val: 15.0, step: 0.05, description: 'Ratio of Total Cholesterol to HDL.' },
  { name: 'NON_HDL_TO_HDL_RATIO', label: 'Non-HDL / HDL Ratio', category: 'Serum Biomarkers', unit: 'ratio', default_val: 2.18, min_val: 0.5, max_val: 12.0, step: 0.05, description: 'Atherogenic index of plasma lipid profile.' },

  // 2. Dietary Macronutrients (10)
  { name: 'DR1TKCAL', label: 'Total Caloric Intake', category: 'Dietary Macronutrients', unit: 'kcal/day', default_val: 1990.0, min_val: 500, max_val: 6000, step: 10, description: 'Total daily energy intake.' },
  { name: 'DR1TPROT', label: 'Total Protein', category: 'Dietary Macronutrients', unit: 'g/day', default_val: 90.0, min_val: 10, max_val: 300, step: 1, description: 'Daily total dietary protein consumed.' },
  { name: 'DR1TCARB', label: 'Total Carbohydrates', category: 'Dietary Macronutrients', unit: 'g/day', default_val: 250.0, min_val: 20, max_val: 700, step: 1, description: 'Total carbohydrates consumed.' },
  { name: 'DR1TSUGR', label: 'Total Sugars', category: 'Dietary Macronutrients', unit: 'g/day', default_val: 42.0, min_val: 0, max_val: 350, step: 1, description: 'Intrinsic plus added dietary sugars.' },
  { name: 'DR1TFIBE', label: 'Dietary Fiber', category: 'Dietary Macronutrients', unit: 'g/day', default_val: 30.0, min_val: 0, max_val: 100, step: 1, description: 'Total dietary fiber consumed.' },
  { name: 'DR1TTFAT', label: 'Total Fat', category: 'Dietary Macronutrients', unit: 'g/day', default_val: 70.0, min_val: 5, max_val: 250, step: 1, description: 'Total lipid intake across all sources.' },
  { name: 'DR1TSFAT', label: 'Saturated Fatty Acids', category: 'Dietary Macronutrients', unit: 'g/day', default_val: 18.0, min_val: 1, max_val: 100, step: 1, description: 'Saturated fat intake.' },
  { name: 'DR1TMFAT', label: 'Monounsaturated Fatty Acids', category: 'Dietary Macronutrients', unit: 'g/day', default_val: 28.0, min_val: 1, max_val: 120, step: 1, description: 'MUFA intake (e.g., oleic acid).' },
  { name: 'DR1TPFAT', label: 'Polyunsaturated Fatty Acids', category: 'Dietary Macronutrients', unit: 'g/day', default_val: 17.0, min_val: 1, max_val: 80, step: 1, description: 'PUFA intake (omega-3 and omega-6).' },
  { name: 'DR1TCHOL', label: 'Dietary Cholesterol', category: 'Dietary Macronutrients', unit: 'mg/day', default_val: 220.0, min_val: 0, max_val: 1000, step: 5, description: 'Total ingested cholesterol.' },

  // 3. Vitamins & Micronutrients (18)
  { name: 'DR1TATOC', label: 'Vitamin E (α-Tocopherol)', category: 'Vitamins & Micronutrients', unit: 'mg', default_val: 10.5, min_val: 0, max_val: 50, step: 0.5, description: 'Lipophilic antioxidant vitamin.' },
  { name: 'DR1TRET', label: 'Retinol', category: 'Vitamins & Micronutrients', unit: 'mcg', default_val: 350.0, min_val: 0, max_val: 3000, step: 10, description: 'Preformed Vitamin A.' },
  { name: 'DR1TVARA', label: 'Vitamin A (RAE)', category: 'Vitamins & Micronutrients', unit: 'mcg', default_val: 900.0, min_val: 0, max_val: 5000, step: 10, description: 'Retinol activity equivalents.' },
  { name: 'DR1TACAR', label: 'Alpha-Carotene', category: 'Vitamins & Micronutrients', unit: 'mcg', default_val: 500.0, min_val: 0, max_val: 5000, step: 10, description: 'Provitamin A carotenoid.' },
  { name: 'DR1TBCAR', label: 'Beta-Carotene', category: 'Vitamins & Micronutrients', unit: 'mcg', default_val: 3200.0, min_val: 0, max_val: 25000, step: 50, description: 'Antioxidant carotenoid.' },
  { name: 'DR1TLYCO', label: 'Lycopene', category: 'Vitamins & Micronutrients', unit: 'mcg', default_val: 6000.0, min_val: 0, max_val: 30000, step: 50, description: 'Lycopene from tomatoes and fruits.' },
  { name: 'DR1TLZ', label: 'Lutein + Zeaxanthin', category: 'Vitamins & Micronutrients', unit: 'mcg', default_val: 1800.0, min_val: 0, max_val: 15000, step: 50, description: 'Macular carotenoids.' },
  { name: 'DR1TVB1', label: 'Thiamin (Vitamin B1)', category: 'Vitamins & Micronutrients', unit: 'mg', default_val: 1.4, min_val: 0.1, max_val: 10, step: 0.1, description: 'Carbohydrate metabolism cofactor.' },
  { name: 'DR1TVB2', label: 'Riboflavin (Vitamin B2)', category: 'Vitamins & Micronutrients', unit: 'mg', default_val: 1.6, min_val: 0.1, max_val: 10, step: 0.1, description: 'FAD/FMN coenzyme precursor.' },
  { name: 'DR1TNIAC', label: 'Niacin (Vitamin B3)', category: 'Vitamins & Micronutrients', unit: 'mg', default_val: 18.0, min_val: 1, max_val: 100, step: 0.5, description: 'NAD/NADP coenzyme precursor.' },
  { name: 'DR1TVB6', label: 'Vitamin B6 (Pyridoxine)', category: 'Vitamins & Micronutrients', unit: 'mg', default_val: 1.8, min_val: 0.1, max_val: 15, step: 0.1, description: 'Amino acid metabolism cofactor.' },
  { name: 'DR1TFOLA', label: 'Total Folate', category: 'Vitamins & Micronutrients', unit: 'mcg', default_val: 500.0, min_val: 20, max_val: 2000, step: 10, description: 'One-carbon metabolism / methylation.' },
  { name: 'DR1TFA', label: 'Folic Acid', category: 'Vitamins & Micronutrients', unit: 'mcg', default_val: 120.0, min_val: 0, max_val: 1000, step: 5, description: 'Synthetic supplemental folate.' },
  { name: 'DR1TFF', label: 'Food Folate', category: 'Vitamins & Micronutrients', unit: 'mcg', default_val: 380.0, min_val: 10, max_val: 1500, step: 10, description: 'Naturally occurring food folates.' },
  { name: 'DR1TFDFE', label: 'Folate Equiv (DFE)', category: 'Vitamins & Micronutrients', unit: 'mcg', default_val: 584.0, min_val: 20, max_val: 3000, step: 10, description: 'Dietary Folate Equivalents.' },
  { name: 'DR1TVB12', label: 'Vitamin B12', category: 'Vitamins & Micronutrients', unit: 'mcg', default_val: 4.2, min_val: 0, max_val: 50, step: 0.1, description: 'Erythrocyte and neurological cofactor.' },
  { name: 'DR1TVC', label: 'Vitamin C (Ascorbic Acid)', category: 'Vitamins & Micronutrients', unit: 'mg', default_val: 100.0, min_val: 0, max_val: 1000, step: 5, description: 'Antioxidant and collagen cofactor.' },
  { name: 'DR1TVK', label: 'Vitamin K', category: 'Vitamins & Micronutrients', unit: 'mcg', default_val: 120.0, min_val: 5, max_val: 1000, step: 5, description: 'Coagulation and bone mineral factor.' },

  // 4. Minerals & Bioactives (11)
  { name: 'DR1TCALC', label: 'Calcium', category: 'Minerals & Bioactives', unit: 'mg', default_val: 950.0, min_val: 100, max_val: 3000, step: 10, description: 'Bone mineral and vascular signaling.' },
  { name: 'DR1TPHOS', label: 'Phosphorus', category: 'Minerals & Bioactives', unit: 'mg', default_val: 1400.0, min_val: 100, max_val: 4000, step: 10, description: 'Cellular ATP and bone hydroxyapatite.' },
  { name: 'DR1TMAGN', label: 'Magnesium', category: 'Minerals & Bioactives', unit: 'mg', default_val: 380.0, min_val: 50, max_val: 1200, step: 5, description: 'Insulin signaling & enzyme cofactor.' },
  { name: 'DR1TIRON', label: 'Iron', category: 'Minerals & Bioactives', unit: 'mg', default_val: 14.0, min_val: 2, max_val: 60, step: 0.5, description: 'Hemoglobin & cellular oxygen transport.' },
  { name: 'DR1TZINC', label: 'Zinc', category: 'Minerals & Bioactives', unit: 'mg', default_val: 11.0, min_val: 1, max_val: 50, step: 0.5, description: 'Immune & superoxide dismutase cofactor.' },
  { name: 'DR1TCOPP', label: 'Copper', category: 'Minerals & Bioactives', unit: 'mg', default_val: 1.2, min_val: 0.1, max_val: 10, step: 0.1, description: 'Cytochrome c oxidase cofactor.' },
  { name: 'DR1TSODI', label: 'Sodium', category: 'Minerals & Bioactives', unit: 'mg', default_val: 1900.0, min_val: 200, max_val: 8000, step: 20, description: 'Extracellular electrolyte balance.' },
  { name: 'DR1TPOTA', label: 'Potassium', category: 'Minerals & Bioactives', unit: 'mg', default_val: 3500.0, min_val: 500, max_val: 8000, step: 20, description: 'Intracellular electrolyte and vascular health.' },
  { name: 'DR1TSELE', label: 'Selenium', category: 'Minerals & Bioactives', unit: 'mcg', default_val: 90.0, min_val: 5, max_val: 300, step: 1, description: 'Glutathione peroxidase cofactor.' },
  { name: 'DR1TCAFF', label: 'Caffeine', category: 'Minerals & Bioactives', unit: 'mg', default_val: 80.0, min_val: 0, max_val: 800, step: 5, description: 'Daily methylxanthine stimulant.' },
  { name: 'DR1TALCO', label: 'Alcohol', category: 'Minerals & Bioactives', unit: 'g', default_val: 0.0, min_val: 0, max_val: 150, step: 1, description: 'Daily ethanol intake.' },

  // 5. Dietary Quality & AMDR Ratios (7)
  { name: 'PROT_PCT_KCAL', label: 'Protein % of Calories', category: 'Dietary Ratios & AMDR', unit: '%', default_val: 18.09, min_val: 5, max_val: 45, step: 0.1, description: 'Percentage of daily energy from protein.' },
  { name: 'CARB_PCT_KCAL', label: 'Carbohydrate % of Calories', category: 'Dietary Ratios & AMDR', unit: '%', default_val: 50.25, min_val: 10, max_val: 85, step: 0.1, description: 'Percentage of daily energy from carbs.' },
  { name: 'FAT_PCT_KCAL', label: 'Fat % of Calories', category: 'Dietary Ratios & AMDR', unit: '%', default_val: 31.65, min_val: 5, max_val: 60, step: 0.1, description: 'Percentage of daily energy from fat.' },
  { name: 'FIBER_PER_1000_KCAL', label: 'Fiber per 1000 kcal', category: 'Dietary Ratios & AMDR', unit: 'g/1000kcal', default_val: 15.07, min_val: 1, max_val: 40, step: 0.1, description: 'Fiber intake density.' },
  { name: 'SUGAR_TO_FIBER_RATIO', label: 'Sugar-to-Fiber Ratio', category: 'Dietary Ratios & AMDR', unit: 'ratio', default_val: 1.35, min_val: 0.1, max_val: 10, step: 0.05, description: 'Ratio of total sugar to fiber.' },
  { name: 'SODIUM_POTASSIUM_RATIO', label: 'Sodium-to-Potassium Ratio', category: 'Dietary Ratios & AMDR', unit: 'ratio', default_val: 0.54, min_val: 0.1, max_val: 5, step: 0.05, description: 'Cardiovascular electrolyte balance.' },
  { name: 'SAT_TO_UNSAT_FAT_RATIO', label: 'Sat-to-Unsat Fat Ratio', category: 'Dietary Ratios & AMDR', unit: 'ratio', default_val: 0.39, min_val: 0.05, max_val: 3, step: 0.05, description: 'Atherogenic lipid index.' },

  // 6. Lifestyle & Physical Activity (19)
  { name: 'SLEEP_HOURS', label: 'Sleep Duration', category: 'Lifestyle & Physical Activity', unit: 'hours/night', default_val: 7.5, min_val: 3, max_val: 14, step: 0.5, description: 'Nocturnal sleep duration.' },
  { name: 'SLEEP_DEV_FROM_7_5', label: 'Sleep Deviation from 7.5h', category: 'Lifestyle & Physical Activity', unit: 'hours', default_val: 0.0, min_val: -5, max_val: 7, step: 0.5, description: 'Deviation from restorative sleep norm.' },
  { name: 'PAD680', label: 'Sedentary Time', category: 'Lifestyle & Physical Activity', unit: 'min/day', default_val: 300.0, min_val: 0, max_val: 1000, step: 10, description: 'Daily sitting/reclining time.' },
  { name: 'PA_VIG_WORK_MIN_WK', label: 'Vigorous Work Activity', category: 'Lifestyle & Physical Activity', unit: 'min/week', default_val: 0.0, min_val: 0, max_val: 2000, step: 10, description: 'Weekly vigorous occupational work.' },
  { name: 'PA_MOD_WORK_MIN_WK', label: 'Moderate Work Activity', category: 'Lifestyle & Physical Activity', unit: 'min/week', default_val: 120.0, min_val: 0, max_val: 2000, step: 10, description: 'Weekly moderate occupational work.' },
  { name: 'PA_WALK_BIKE_MIN_WK', label: 'Active Transport (Walking/Biking)', category: 'Lifestyle & Physical Activity', unit: 'min/week', default_val: 60.0, min_val: 0, max_val: 1500, step: 10, description: 'Weekly active commuting.' },
  { name: 'PA_VIG_REC_MIN_WK', label: 'Vigorous Exercise', category: 'Lifestyle & Physical Activity', unit: 'min/week', default_val: 60.0, min_val: 0, max_val: 1500, step: 10, description: 'Weekly high-intensity recreational exercise.' },
  { name: 'PA_MOD_REC_MIN_WK', label: 'Moderate Exercise', category: 'Lifestyle & Physical Activity', unit: 'min/week', default_val: 60.0, min_val: 0, max_val: 1500, step: 10, description: 'Weekly moderate recreational exercise.' },
  { name: 'PA_TOTAL_MET_MIN_WK', label: 'Total Physical Activity', category: 'Lifestyle & Physical Activity', unit: 'MET-min/wk', default_val: 1440.0, min_val: 0, max_val: 15000, step: 50, description: 'Total MET minutes of physical activity.' },
  { name: 'SLQ050', label: 'Diagnosed Sleep Disorder', category: 'Lifestyle & Physical Activity', unit: '0 or 1', default_val: 0.0, min_val: 0, max_val: 1, step: 1, description: '1 if diagnosed with sleep apnea or insomnia.' },
  { name: 'PAQ605', label: 'Vigorous Work (Binary)', category: 'Lifestyle & Physical Activity', unit: '0 or 1', default_val: 0.0, min_val: 0, max_val: 1, step: 1, description: 'Performs vigorous work tasks.' },
  { name: 'PAQ620', label: 'Moderate Work (Binary)', category: 'Lifestyle & Physical Activity', unit: '0 or 1', default_val: 1.0, min_val: 0, max_val: 1, step: 1, description: 'Performs moderate work tasks.' },
  { name: 'PAQ635', label: 'Active Transport (Binary)', category: 'Lifestyle & Physical Activity', unit: '0 or 1', default_val: 1.0, min_val: 0, max_val: 1, step: 1, description: 'Uses walking or bicycling for travel.' },
  { name: 'PAQ650', label: 'Vigorous Leisure (Binary)', category: 'Lifestyle & Physical Activity', unit: '0 or 1', default_val: 1.0, min_val: 0, max_val: 1, step: 1, description: 'Engages in vigorous recreational sports.' },
  { name: 'PAQ665', label: 'Moderate Leisure (Binary)', category: 'Lifestyle & Physical Activity', unit: '0 or 1', default_val: 1.0, min_val: 0, max_val: 1, step: 1, description: 'Engages in moderate recreational sports.' },
  { name: 'SMOKE_NEVER', label: 'Smoking: Never Smoker', category: 'Lifestyle & Physical Activity', unit: '0 or 1', default_val: 1.0, min_val: 0, max_val: 1, step: 1, description: 'Has smoked < 100 cigarettes in lifetime.' },
  { name: 'SMOKE_FORMER', label: 'Smoking: Former Smoker', category: 'Lifestyle & Physical Activity', unit: '0 or 1', default_val: 0.0, min_val: 0, max_val: 1, step: 1, description: 'Quit smoking.' },
  { name: 'SMOKE_SOMEDAYS', label: 'Smoking: Intermittent', category: 'Lifestyle & Physical Activity', unit: '0 or 1', default_val: 0.0, min_val: 0, max_val: 1, step: 1, description: 'Currently smokes some days.' },
  { name: 'SMOKE_EVERYDAY', label: 'Smoking: Daily Smoker', category: 'Lifestyle & Physical Activity', unit: '0 or 1', default_val: 0.0, min_val: 0, max_val: 1, step: 1, description: 'Currently smokes every day.' },

  // 7. Demographics & Imputation Flags (17)
  { name: 'RIDAGEYR', label: 'Chronological Age', category: 'Demographics & Vitals', unit: 'years', default_val: 42.0, min_val: 18, max_val: 100, step: 1, description: 'Patient age in completed years.' },
  { name: 'INDFMPIR', label: 'Income-to-Poverty Ratio', category: 'Demographics & Vitals', unit: 'ratio', default_val: 3.2, min_val: 0, max_val: 5, step: 0.1, description: 'Family income divided by poverty threshold.' },
  { name: 'DMDEDUC2', label: 'Education Level Code', category: 'Demographics & Vitals', unit: '1 to 5', default_val: 5.0, min_val: 1, max_val: 5, step: 1, description: '1=<9th grade, 5=College graduate or above.' },
  { name: 'SEX_FEMALE', label: 'Biological Sex: Female', category: 'Demographics & Vitals', unit: '0 or 1', default_val: 0.0, min_val: 0, max_val: 1, step: 1, description: '1 = Female, 0 = Male.' },
  { name: 'RACE_MEX_AMER', label: 'Ethnicity: Mexican American', category: 'Demographics & Vitals', unit: '0 or 1', default_val: 0.0, min_val: 0, max_val: 1, step: 1, description: 'Census race/ethnicity code.' },
  { name: 'RACE_OTHER_HISP', label: 'Ethnicity: Other Hispanic', category: 'Demographics & Vitals', unit: '0 or 1', default_val: 0.0, min_val: 0, max_val: 1, step: 1, description: 'Census race/ethnicity code.' },
  { name: 'RACE_NH_WHITE', label: 'Ethnicity: Non-Hispanic White', category: 'Demographics & Vitals', unit: '0 or 1', default_val: 1.0, min_val: 0, max_val: 1, step: 1, description: 'Census race/ethnicity code.' },
  { name: 'RACE_NH_BLACK', label: 'Ethnicity: Non-Hispanic Black', category: 'Demographics & Vitals', unit: '0 or 1', default_val: 0.0, min_val: 0, max_val: 1, step: 1, description: 'Census race/ethnicity code.' },
  { name: 'RACE_OTHER_MULTI', label: 'Ethnicity: Asian / Multi-Racial', category: 'Demographics & Vitals', unit: '0 or 1', default_val: 0.0, min_val: 0, max_val: 1, step: 1, description: 'Census race/ethnicity code.' },
  { name: 'MARITAL_MARRIED', label: 'Marital: Married', category: 'Demographics & Vitals', unit: '0 or 1', default_val: 1.0, min_val: 0, max_val: 1, step: 1, description: 'Currently married or living with partner.' },
  { name: 'MARITAL_PREV_MARRIED', label: 'Marital: Divorced / Widowed', category: 'Demographics & Vitals', unit: '0 or 1', default_val: 0.0, min_val: 0, max_val: 1, step: 1, description: 'Previously married.' },
  { name: 'MARITAL_NEVER_MARRIED', label: 'Marital: Never Married', category: 'Demographics & Vitals', unit: '0 or 1', default_val: 0.0, min_val: 0, max_val: 1, step: 1, description: 'Single / never married.' },
  { name: 'INDFMPIR_MISS', label: 'Income Missing Indicator', category: 'Data Quality & Missingness', unit: '0 or 1', default_val: 0.0, min_val: 0, max_val: 1, step: 1, description: '1 if income was unrecorded.' },
  { name: 'SLEEP_HOURS_MISS', label: 'Sleep Missing Indicator', category: 'Data Quality & Missingness', unit: '0 or 1', default_val: 0.0, min_val: 0, max_val: 1, step: 1, description: '1 if sleep duration was unrecorded.' },
  { name: 'PAD680_MISS', label: 'Sitting Time Missing Indicator', category: 'Data Quality & Missingness', unit: '0 or 1', default_val: 0.0, min_val: 0, max_val: 1, step: 1, description: '1 if sedentary time was unrecorded.' },
  { name: 'PA_TOTAL_MET_MIN_WK_MISS', label: 'Activity Missing Indicator', category: 'Data Quality & Missingness', unit: '0 or 1', default_val: 0.0, min_val: 0, max_val: 1, step: 1, description: '1 if physical activity was unrecorded.' },
  { name: 'OTHER_FEATURE_MISS', label: 'Composite Missing Indicator', category: 'Data Quality & Missingness', unit: '0 or 1', default_val: 0.0, min_val: 0, max_val: 1, step: 1, description: '1 if any secondary features were imputed.' },
]
