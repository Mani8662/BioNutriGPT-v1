"""BioNutriGPT Backend Pipeline Service.

Provides a thread-safe, cached service layer interfacing between FastAPI
and the canonical 4-stage clinical pipeline located in Models/Less Data/.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

LOGGER = logging.getLogger("BioNutriGPT.Service")

# Dynamically locate canonical Models/Less Data directory regardless of where backend is placed
CURRENT_DIR = Path(__file__).resolve().parent
candidates = [
    CURRENT_DIR.parent,                                 # e.g., Models/Less Data/backend -> Models/Less Data
    CURRENT_DIR.parent / "Models" / "Less Data",        # e.g., repo_root/backend -> repo_root/Models/Less Data
    CURRENT_DIR.parent.parent / "Models" / "Less Data", # e.g., nested subfolder
    CURRENT_DIR.parent / "Less Data",
]
LESS_DATA_DIR = next(
    (c for c in candidates if (c / "infer_bionutrigpt_pipeline_Stage_1_and_2.py").is_file()),
    CURRENT_DIR.parent
)
if str(LESS_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(LESS_DATA_DIR))

# Import canonical upstream modules
import infer_bionutrigpt_pipeline_Stage_1_and_2 as p12
import infer_bionutrigpt_pipeline_Stage_1_to_4 as p14
import nutrient_rule_policy as nrp
import stage3_milp_solver as s3
import train_bionutrigpt_stage1 as s1
import stage1_deployment as deployment

# Locks for thread safety
_MODEL_LOCK = threading.Lock()
_PIPELINE_LOADED = False
_FOOD_CATALOG_CACHE: list[dict[str, Any]] | None = None

# Sample patient default feature vector & metadata
SAMPLE_PATIENT_VALUES: list[float] = [
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

SAMPLE_METADATA: dict[str, Any] = {
    "weight_kg": 74.0,
    "height_cm": 176.0,
    "age_years": 42.0,
    "is_female": False,
    "activity_level": "sedentary"
}

# ---------------------------------------------------------------------------
# Feature dictionary with clinical labels, categories, units, and descriptions
# ---------------------------------------------------------------------------
FEATURE_DEFINITIONS: list[dict[str, Any]] = [
    # 1. Biomarkers (5)
    {"name": "LBDHDD", "label": "HDL-Cholesterol", "category": "Serum Biomarkers", "unit": "mg/dL", "default_val": 56.0, "min_val": 10.0, "max_val": 150.0, "step": 1.0, "description": "High-density lipoprotein ('good') cholesterol."},
    {"name": "LBXTC", "label": "Total Cholesterol", "category": "Serum Biomarkers", "unit": "mg/dL", "default_val": 178.0, "min_val": 50.0, "max_val": 500.0, "step": 1.0, "description": "Total serum cholesterol concentration."},
    {"name": "NON_HDL_CHOL", "label": "Non-HDL Cholesterol", "category": "Serum Biomarkers", "unit": "mg/dL", "default_val": 122.0, "min_val": 30.0, "max_val": 450.0, "step": 1.0, "description": "Total cholesterol minus HDL; primary atherogenic marker."},
    {"name": "CHOL_HDL_RATIO", "label": "Total / HDL Ratio", "category": "Serum Biomarkers", "unit": "ratio", "default_val": 3.18, "min_val": 1.0, "max_val": 15.0, "step": 0.05, "description": "Ratio of Total Cholesterol to HDL."},
    {"name": "NON_HDL_TO_HDL_RATIO", "label": "Non-HDL / HDL Ratio", "category": "Serum Biomarkers", "unit": "ratio", "default_val": 2.18, "min_val": 0.5, "max_val": 12.0, "step": 0.05, "description": "Atherogenic index of plasma lipid profile."},

    # 2. Dietary Macronutrients (10)
    {"name": "DR1TKCAL", "label": "Total Energy Intake", "category": "Dietary Macronutrients", "unit": "kcal", "default_val": 1990.0, "min_val": 500.0, "max_val": 6000.0, "step": 10.0, "description": "Total reported daily caloric intake."},
    {"name": "DR1TPROT", "label": "Total Dietary Protein", "category": "Dietary Macronutrients", "unit": "g", "default_val": 90.0, "min_val": 10.0, "max_val": 300.0, "step": 1.0, "description": "Daily protein consumed across all foods."},
    {"name": "DR1TCARB", "label": "Total Carbohydrates", "category": "Dietary Macronutrients", "unit": "g", "default_val": 250.0, "min_val": 20.0, "max_val": 700.0, "step": 1.0, "description": "Daily carbohydrates consumed."},
    {"name": "DR1TSUGR", "label": "Total Sugars", "category": "Dietary Macronutrients", "unit": "g", "default_val": 42.0, "min_val": 0.0, "max_val": 350.0, "step": 1.0, "description": "Total dietary sugars (intrinsic + added)."},
    {"name": "DR1TFIBE", "label": "Dietary Fiber", "category": "Dietary Macronutrients", "unit": "g", "default_val": 30.0, "min_val": 0.0, "max_val": 100.0, "step": 1.0, "description": "Total dietary fiber consumed."},
    {"name": "DR1TTFAT", "label": "Total Fat", "category": "Dietary Macronutrients", "unit": "g", "default_val": 70.0, "min_val": 5.0, "max_val": 250.0, "step": 1.0, "description": "Total lipid intake across all types."},
    {"name": "DR1TSFAT", "label": "Saturated Fatty Acids", "category": "Dietary Macronutrients", "unit": "g", "default_val": 18.0, "min_val": 1.0, "max_val": 100.0, "step": 1.0, "description": "Saturated fat intake."},
    {"name": "DR1TMFAT", "label": "Monounsaturated Fatty Acids", "category": "Dietary Macronutrients", "unit": "g", "default_val": 28.0, "min_val": 1.0, "max_val": 120.0, "step": 1.0, "description": "MUFA intake (e.g. oleic acid)."},
    {"name": "DR1TPFAT", "label": "Polyunsaturated Fatty Acids", "category": "Dietary Macronutrients", "unit": "g", "default_val": 17.0, "min_val": 1.0, "max_val": 80.0, "step": 1.0, "description": "PUFA intake (omega-3 and omega-6)."},
    {"name": "DR1TCHOL", "label": "Dietary Cholesterol", "category": "Dietary Macronutrients", "unit": "mg", "default_val": 220.0, "min_val": 0.0, "max_val": 1000.0, "step": 5.0, "description": "Total ingested cholesterol."},

    # 3. Dietary Micronutrients & Vitamins (19)
    {"name": "DR1TATOC", "label": "Vitamin E (Alpha-Tocopherol)", "category": "Vitamins & Micronutrients", "unit": "mg", "default_val": 10.5, "min_val": 0.0, "max_val": 50.0, "step": 0.5, "description": "Lipophilic antioxidant vitamin."},
    {"name": "DR1TRET", "label": "Retinol", "category": "Vitamins & Micronutrients", "unit": "mcg", "default_val": 350.0, "min_val": 0.0, "max_val": 3000.0, "step": 10.0, "description": "Preformed Vitamin A."},
    {"name": "DR1TVARA", "label": "Vitamin A (RAE)", "category": "Vitamins & Micronutrients", "unit": "mcg", "default_val": 900.0, "min_val": 0.0, "max_val": 5000.0, "step": 10.0, "description": "Retinol activity equivalents."},
    {"name": "DR1TACAR", "label": "Alpha-Carotene", "category": "Vitamins & Micronutrients", "unit": "mcg", "default_val": 500.0, "min_val": 0.0, "max_val": 5000.0, "step": 10.0, "description": "Provitamin A carotenoid."},
    {"name": "DR1TBCAR", "label": "Beta-Carotene", "category": "Vitamins & Micronutrients", "unit": "mcg", "default_val": 3200.0, "min_val": 0.0, "max_val": 25000.0, "step": 50.0, "description": "Antioxidant carotenoid."},
    {"name": "DR1TLYCO", "label": "Lycopene", "category": "Vitamins & Micronutrients", "unit": "mcg", "default_val": 6000.0, "min_val": 0.0, "max_val": 30000.0, "step": 50.0, "description": "Carotenoid from tomatoes/fruits."},
    {"name": "DR1TLZ", "label": "Lutein + Zeaxanthin", "category": "Vitamins & Micronutrients", "unit": "mcg", "default_val": 1800.0, "min_val": 0.0, "max_val": 15000.0, "step": 50.0, "description": "Macular carotenoids."},
    {"name": "DR1TVB1", "label": "Thiamin (Vitamin B1)", "category": "Vitamins & Micronutrients", "unit": "mg", "default_val": 1.4, "min_val": 0.1, "max_val": 10.0, "step": 0.1, "description": "Carbohydrate metabolism cofactor."},
    {"name": "DR1TVB2", "label": "Riboflavin (Vitamin B2)", "category": "Vitamins & Micronutrients", "unit": "mg", "default_val": 1.6, "min_val": 0.1, "max_val": 10.0, "step": 0.1, "description": "FAD/FMN coenzyme precursor."},
    {"name": "DR1TNIAC", "label": "Niacin (Vitamin B3)", "category": "Vitamins & Micronutrients", "unit": "mg", "default_val": 18.0, "min_val": 1.0, "max_val": 100.0, "step": 0.5, "description": "NAD/NADP coenzyme precursor."},
    {"name": "DR1TVB6", "label": "Vitamin B6", "category": "Vitamins & Micronutrients", "unit": "mg", "default_val": 1.8, "min_val": 0.1, "max_val": 15.0, "step": 0.1, "description": "Pyridoxine amino acid metabolism."},
    {"name": "DR1TFOLA", "label": "Total Folate", "category": "Vitamins & Micronutrients", "unit": "mcg", "default_val": 500.0, "min_val": 20.0, "max_val": 2000.0, "step": 10.0, "description": "One-carbon metabolism / methylation."},
    {"name": "DR1TFA", "label": "Folic Acid", "category": "Vitamins & Micronutrients", "unit": "mcg", "default_val": 120.0, "min_val": 0.0, "max_val": 1000.0, "step": 5.0, "description": "Synthetic supplemental folate."},
    {"name": "DR1TFF", "label": "Food Folate", "category": "Vitamins & Micronutrients", "unit": "mcg", "default_val": 380.0, "min_val": 10.0, "max_val": 1500.0, "step": 10.0, "description": "Naturally occurring food folates."},
    {"name": "DR1TFDFE", "label": "Dietary Folate Equiv (DFE)", "category": "Vitamins & Micronutrients", "unit": "mcg", "default_val": 584.0, "min_val": 20.0, "max_val": 3000.0, "step": 10.0, "description": "Bioavailable folate measure."},
    {"name": "DR1TVB12", "label": "Vitamin B12 (Cobalamin)", "category": "Vitamins & Micronutrients", "unit": "mcg", "default_val": 4.2, "min_val": 0.0, "max_val": 50.0, "step": 0.1, "description": "Nerve & erythrocyte function."},
    {"name": "DR1TVC", "label": "Vitamin C (Ascorbic Acid)", "category": "Vitamins & Micronutrients", "unit": "mg", "default_val": 100.0, "min_val": 0.0, "max_val": 1000.0, "step": 5.0, "description": "Antioxidant / collagen synthesis."},
    {"name": "DR1TVK", "label": "Vitamin K", "category": "Vitamins & Micronutrients", "unit": "mcg", "default_val": 120.0, "min_val": 5.0, "max_val": 1000.0, "step": 5.0, "description": "Coagulation and bone mineral protein."},

    # 4. Dietary Minerals & Bioactives (10)
    {"name": "DR1TCALC", "label": "Calcium", "category": "Minerals & Bioactives", "unit": "mg", "default_val": 950.0, "min_val": 100.0, "max_val": 3000.0, "step": 10.0, "description": "Bone mineral and vascular tone."},
    {"name": "DR1TPHOS", "label": "Phosphorus", "category": "Minerals & Bioactives", "unit": "mg", "default_val": 1400.0, "min_val": 100.0, "max_val": 4000.0, "step": 10.0, "description": "Cellular ATP and bone hydroxyapatite."},
    {"name": "DR1TMAGN", "label": "Magnesium", "category": "Minerals & Bioactives", "unit": "mg", "default_val": 380.0, "min_val": 50.0, "max_val": 1200.0, "step": 5.0, "description": "Enzyme cofactor, insulin signaling."},
    {"name": "DR1TIRON", "label": "Iron", "category": "Minerals & Bioactives", "unit": "mg", "default_val": 14.0, "min_val": 2.0, "max_val": 60.0, "step": 0.5, "description": "Hemoglobin & oxygen transport."},
    {"name": "DR1TZINC", "label": "Zinc", "category": "Minerals & Bioactives", "unit": "mg", "default_val": 11.0, "min_val": 1.0, "max_val": 50.0, "step": 0.5, "description": "Immune and antioxidant superoxide dismutase."},
    {"name": "DR1TCOPP", "label": "Copper", "category": "Minerals & Bioactives", "unit": "mg", "default_val": 1.2, "min_val": 0.1, "max_val": 10.0, "step": 0.1, "description": "Cytochrome c oxidase cofactor."},
    {"name": "DR1TSODI", "label": "Sodium", "category": "Minerals & Bioactives", "unit": "mg", "default_val": 1900.0, "min_val": 200.0, "max_val": 8000.0, "step": 20.0, "description": "Extracellular electrolyte balance."},
    {"name": "DR1TPOTA", "label": "Potassium", "category": "Minerals & Bioactives", "unit": "mg", "default_val": 3500.0, "min_val": 500.0, "max_val": 8000.0, "step": 20.0, "description": "Intracellular electrolyte and vascular health."},
    {"name": "DR1TSELE", "label": "Selenium", "category": "Minerals & Bioactives", "unit": "mcg", "default_val": 90.0, "min_val": 5.0, "max_val": 300.0, "step": 1.0, "description": "Glutathione peroxidase cofactor."},
    {"name": "DR1TCAFF", "label": "Caffeine", "category": "Minerals & Bioactives", "unit": "mg", "default_val": 80.0, "min_val": 0.0, "max_val": 800.0, "step": 5.0, "description": "Methylxanthine stimulant."},
    {"name": "DR1TALCO", "label": "Alcohol", "category": "Minerals & Bioactives", "unit": "g", "default_val": 0.0, "min_val": 0.0, "max_val": 150.0, "step": 1.0, "description": "Daily ethanol intake."},

    # 5. Dietary Quality & AMDR Ratios (7)
    {"name": "PROT_PCT_KCAL", "label": "Protein % of Energy", "category": "Dietary Ratios", "unit": "%", "default_val": 18.09, "min_val": 5.0, "max_val": 45.0, "step": 0.1, "description": "Percentage of calories from protein."},
    {"name": "CARB_PCT_KCAL", "label": "Carbohydrate % of Energy", "category": "Dietary Ratios", "unit": "%", "default_val": 50.25, "min_val": 10.0, "max_val": 85.0, "step": 0.1, "description": "Percentage of calories from carbohydrate."},
    {"name": "FAT_PCT_KCAL", "label": "Fat % of Energy", "category": "Dietary Ratios", "unit": "%", "default_val": 31.65, "min_val": 5.0, "max_val": 60.0, "step": 0.1, "description": "Percentage of calories from fat."},
    {"name": "FIBER_PER_1000_KCAL", "label": "Fiber per 1000 kcal", "category": "Dietary Ratios", "unit": "g/1000kcal", "default_val": 15.07, "min_val": 1.0, "max_val": 40.0, "step": 0.1, "description": "Dietary fiber density."},
    {"name": "SUGAR_TO_FIBER_RATIO", "label": "Sugar-to-Fiber Ratio", "category": "Dietary Ratios", "unit": "ratio", "default_val": 1.35, "min_val": 0.1, "max_val": 10.0, "step": 0.05, "description": "Ratio of total sugar to dietary fiber."},
    {"name": "SODIUM_POTASSIUM_RATIO", "label": "Sodium-to-Potassium Ratio", "category": "Dietary Ratios", "unit": "ratio", "default_val": 0.54, "min_val": 0.1, "max_val": 5.0, "step": 0.05, "description": "Ratio of sodium to potassium."},
    {"name": "SAT_TO_UNSAT_FAT_RATIO", "label": "Sat-to-Unsat Fat Ratio", "category": "Dietary Ratios", "unit": "ratio", "default_val": 0.39, "min_val": 0.05, "max_val": 3.0, "step": 0.05, "description": "Atherogenic lipid index."},

    # 6. Sleep & Physical Activity (19)
    {"name": "SLEEP_HOURS", "label": "Sleep Duration", "category": "Lifestyle & Sleep", "unit": "hours/night", "default_val": 7.5, "min_val": 3.0, "max_val": 14.0, "step": 0.5, "description": "Average nocturnal sleep duration."},
    {"name": "SLEEP_DEV_FROM_7_5", "label": "Sleep Deviation from 7.5h", "category": "Lifestyle & Sleep", "unit": "hours", "default_val": 0.0, "min_val": -5.0, "max_val": 7.0, "step": 0.5, "description": "Absolute or directional deviation from restorative norm."},
    {"name": "PAD680", "label": "Sedentary Activity Time", "category": "Lifestyle & Physical Activity", "unit": "min/day", "default_val": 300.0, "min_val": 0.0, "max_val": 1000.0, "step": 10.0, "description": "Daily minutes spent sitting or reclining."},
    {"name": "PA_VIG_WORK_MIN_WK", "label": "Vigorous Work Activity", "category": "Lifestyle & Physical Activity", "unit": "min/week", "default_val": 0.0, "min_val": 0.0, "max_val": 2000.0, "step": 10.0, "description": "Weekly vigorous occupational activity."},
    {"name": "PA_MOD_WORK_MIN_WK", "label": "Moderate Work Activity", "category": "Lifestyle & Physical Activity", "unit": "min/week", "default_val": 120.0, "min_val": 0.0, "max_val": 2000.0, "step": 10.0, "description": "Weekly moderate occupational activity."},
    {"name": "PA_WALK_BIKE_MIN_WK", "label": "Walking / Bicycling", "category": "Lifestyle & Physical Activity", "unit": "min/week", "default_val": 60.0, "min_val": 0.0, "max_val": 1500.0, "step": 10.0, "description": "Weekly active transport."},
    {"name": "PA_VIG_REC_MIN_WK", "label": "Vigorous Leisure Exercise", "category": "Lifestyle & Physical Activity", "unit": "min/week", "default_val": 60.0, "min_val": 0.0, "max_val": 1500.0, "step": 10.0, "description": "Weekly high-intensity recreation."},
    {"name": "PA_MOD_REC_MIN_WK", "label": "Moderate Leisure Exercise", "category": "Lifestyle & Physical Activity", "unit": "min/week", "default_val": 60.0, "min_val": 0.0, "max_val": 1500.0, "step": 10.0, "description": "Weekly moderate recreation."},
    {"name": "PA_TOTAL_MET_MIN_WK", "label": "Total Physical Activity", "category": "Lifestyle & Physical Activity", "unit": "MET-min/wk", "default_val": 1440.0, "min_val": 0.0, "max_val": 15000.0, "step": 50.0, "description": "Metabolic Equivalent minutes per week."},
    {"name": "SLQ050", "label": "Told Had Sleep Disorder", "category": "Lifestyle & Sleep", "unit": "0 or 1", "default_val": 0.0, "min_val": 0.0, "max_val": 1.0, "step": 1.0, "description": "Self-reported sleep disorder diagnosis."},
    {"name": "PAQ605", "label": "Vigorous Work (Binary)", "category": "Lifestyle & Physical Activity", "unit": "0 or 1", "default_val": 0.0, "min_val": 0.0, "max_val": 1.0, "step": 1.0, "description": "Performs vigorous occupational activity."},
    {"name": "PAQ620", "label": "Moderate Work (Binary)", "category": "Lifestyle & Physical Activity", "unit": "0 or 1", "default_val": 1.0, "min_val": 0.0, "max_val": 1.0, "step": 1.0, "description": "Performs moderate occupational activity."},
    {"name": "PAQ635", "label": "Walk/Bicycle Transport (Binary)", "category": "Lifestyle & Physical Activity", "unit": "0 or 1", "default_val": 1.0, "min_val": 0.0, "max_val": 1.0, "step": 1.0, "description": "Uses active transport."},
    {"name": "PAQ650", "label": "Vigorous Leisure (Binary)", "category": "Lifestyle & Physical Activity", "unit": "0 or 1", "default_val": 1.0, "min_val": 0.0, "max_val": 1.0, "step": 1.0, "description": "Performs vigorous leisure exercise."},
    {"name": "PAQ665", "label": "Moderate Leisure (Binary)", "category": "Lifestyle & Physical Activity", "unit": "0 or 1", "default_val": 1.0, "min_val": 0.0, "max_val": 1.0, "step": 1.0, "description": "Performs moderate leisure exercise."},
    {"name": "SMOKE_NEVER", "label": "Smoking: Never", "category": "Lifestyle & Smoking", "unit": "0 or 1", "default_val": 1.0, "min_val": 0.0, "max_val": 1.0, "step": 1.0, "description": "Lifetime non-smoker."},
    {"name": "SMOKE_FORMER", "label": "Smoking: Former", "category": "Lifestyle & Smoking", "unit": "0 or 1", "default_val": 0.0, "min_val": 0.0, "max_val": 1.0, "step": 1.0, "description": "Quit smoking."},
    {"name": "SMOKE_SOMEDAYS", "label": "Smoking: Some Days", "category": "Lifestyle & Smoking", "unit": "0 or 1", "default_val": 0.0, "min_val": 0.0, "max_val": 1.0, "step": 1.0, "description": "Intermittent smoker."},
    {"name": "SMOKE_EVERYDAY", "label": "Smoking: Every Day", "category": "Lifestyle & Smoking", "unit": "0 or 1", "default_val": 0.0, "min_val": 0.0, "max_val": 1.0, "step": 1.0, "description": "Daily active smoker."},

    # 7. Demographics & Imputation Flags (17)
    {"name": "RIDAGEYR", "label": "Patient Age", "category": "Demographics", "unit": "years", "default_val": 42.0, "min_val": 18.0, "max_val": 100.0, "step": 1.0, "description": "Chronological age in years."},
    {"name": "INDFMPIR", "label": "Income-Poverty Ratio", "category": "Demographics", "unit": "ratio", "default_val": 3.2, "min_val": 0.0, "max_val": 5.0, "step": 0.1, "description": "Ratio of family income to poverty threshold."},
    {"name": "DMDEDUC2", "label": "Education Level Code", "category": "Demographics", "unit": "1 to 5", "default_val": 5.0, "min_val": 1.0, "max_val": 5.0, "step": 1.0, "description": "Education attainment category."},
    {"name": "SEX_FEMALE", "label": "Sex: Female (1=F, 0=M)", "category": "Demographics", "unit": "0 or 1", "default_val": 0.0, "min_val": 0.0, "max_val": 1.0, "step": 1.0, "description": "Biological sex indicator."},
    {"name": "RACE_MEX_AMER", "label": "Race: Mexican American", "category": "Demographics", "unit": "0 or 1", "default_val": 0.0, "min_val": 0.0, "max_val": 1.0, "step": 1.0, "description": "Census race group."},
    {"name": "RACE_OTHER_HISP", "label": "Race: Other Hispanic", "category": "Demographics", "unit": "0 or 1", "default_val": 0.0, "min_val": 0.0, "max_val": 1.0, "step": 1.0, "description": "Census race group."},
    {"name": "RACE_NH_WHITE", "label": "Race: Non-Hispanic White", "category": "Demographics", "unit": "0 or 1", "default_val": 1.0, "min_val": 0.0, "max_val": 1.0, "step": 1.0, "description": "Census race group."},
    {"name": "RACE_NH_BLACK", "label": "Race: Non-Hispanic Black", "category": "Demographics", "unit": "0 or 1", "default_val": 0.0, "min_val": 0.0, "max_val": 1.0, "step": 1.0, "description": "Census race group."},
    {"name": "RACE_OTHER_MULTI", "label": "Race: Asian / Multi-Racial", "category": "Demographics", "unit": "0 or 1", "default_val": 0.0, "min_val": 0.0, "max_val": 1.0, "step": 1.0, "description": "Census race group."},
    {"name": "MARITAL_MARRIED", "label": "Marital: Married", "category": "Demographics", "unit": "0 or 1", "default_val": 1.0, "min_val": 0.0, "max_val": 1.0, "step": 1.0, "description": "Marital status."},
    {"name": "MARITAL_PREV_MARRIED", "label": "Marital: Previously Married", "category": "Demographics", "unit": "0 or 1", "default_val": 0.0, "min_val": 0.0, "max_val": 1.0, "step": 1.0, "description": "Divorced / widowed."},
    {"name": "MARITAL_NEVER_MARRIED", "label": "Marital: Never Married", "category": "Demographics", "unit": "0 or 1", "default_val": 0.0, "min_val": 0.0, "max_val": 1.0, "step": 1.0, "description": "Single."},
    {"name": "INDFMPIR_MISS", "label": "Income Ratio Missing Flag", "category": "Data Quality Flags", "unit": "0 or 1", "default_val": 0.0, "min_val": 0.0, "max_val": 1.0, "step": 1.0, "description": "Informative missingness flag for income."},
    {"name": "SLEEP_HOURS_MISS", "label": "Sleep Hours Missing Flag", "category": "Data Quality Flags", "unit": "0 or 1", "default_val": 0.0, "min_val": 0.0, "max_val": 1.0, "step": 1.0, "description": "Informative missingness flag for sleep."},
    {"name": "PAD680_MISS", "label": "Sedentary Time Missing Flag", "category": "Data Quality Flags", "unit": "0 or 1", "default_val": 0.0, "min_val": 0.0, "max_val": 1.0, "step": 1.0, "description": "Informative missingness flag for sitting time."},
    {"name": "PA_TOTAL_MET_MIN_WK_MISS", "label": "PA Total Missing Flag", "category": "Data Quality Flags", "unit": "0 or 1", "default_val": 0.0, "min_val": 0.0, "max_val": 1.0, "step": 1.0, "description": "Informative missingness flag for physical activity."},
    {"name": "OTHER_FEATURE_MISS", "label": "General Missingness Flag", "category": "Data Quality Flags", "unit": "0 or 1", "default_val": 0.0, "min_val": 0.0, "max_val": 1.0, "step": 1.0, "description": "Composite flag for auxiliary unrecorded features."}
]


# ---------------------------------------------------------------------------
# Text parsing helper
# ---------------------------------------------------------------------------
def parse_raw_text(text: str) -> tuple[list[float], dict[str, Any]]:
    """Parse raw space- or comma-separated numbers and extract metadata comments."""
    text = text.strip()
    if not text:
        raise ValueError("Input text is empty.")

    metadata: dict[str, Any] = {}
    lines = text.splitlines()
    data_tokens: list[str] = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            for key in ("renal_disease", "pregnant", "lactating"):
                match = re.search(r"\b"+key+r"\s*[:=]\s*([^\s,;]*)", line, re.I)
                if match:
                    token = match[1].lower()
                    allowed = {"true": True, "yes": True, "1": True, "false": False, "no": False, "0": False, "unknown": None, "null": None}
                    if token not in allowed:
                        raise ValueError(f"{key} must be yes, no or unknown.")
                    value = allowed[token]
                    if key in metadata and metadata[key] != value:
                        raise ValueError(f"Conflicting declarations for {key}.")
                    metadata[key] = value
            # Check for clinical metadata: weight_kg=74.0 height_cm=176.0 activity_level=sedentary
            w_match = re.search(r'weight(?:_kg)?\s*[:=]\s*([0-9.]+)', line, re.I)
            h_match = re.search(r'height(?:_cm)?\s*[:=]\s*([0-9.]+)', line, re.I)
            a_match = re.search(r'age(?:_years)?\s*[:=]\s*([0-9.]+)', line, re.I)
            f_match = re.search(r'(?:is_)?female\s*[:=]\s*([0-9a-zA-Z]+)', line, re.I)
            act_match = re.search(r'activity(?:_level)?\s*[:=]\s*([a-zA-Z]+)', line, re.I)

            if w_match:
                metadata["weight_kg"] = float(w_match.group(1))
            if h_match:
                metadata["height_cm"] = float(h_match.group(1))
            if a_match:
                metadata["age_years"] = float(a_match.group(1))
            if f_match:
                val = f_match.group(1).lower()
                metadata["is_female"] = val in ("1", "true", "yes", "female")
            if act_match and act_match.group(1).lower() in ("sedentary", "light", "moderate", "active"):
                metadata["activity_level"] = act_match.group(1).lower()
            continue

        # Data line
        cleaned = line.replace("[", " ").replace("]", " ").replace(",", " ")
        data_tokens.extend(cleaned.split())

    if not data_tokens:
        raise ValueError("No numeric values found in input text.")

    try:
        values = [float(t) for t in data_tokens]
    except ValueError as exc:
        raise ValueError(f"Failed to parse numeric value: {exc}") from exc

    if len(values) == 88:
        raise ValueError(
            "Ambiguous 88-value legacy vector rejected to prevent silent feature misalignment. "
            "Provide exactly 87 features matching schema nhanes-87-v2 (HbA1c excluded)."
        )
    if len(values) != 87:
        raise ValueError(f"Expected exactly 87 numeric values, but found {len(values)}.")

    return values, metadata


FEATURE_NAMES_ORDERED: list[str] = [f["name"] for f in FEATURE_DEFINITIONS]
FEATURE_DEFAULTS_BY_NAME: dict[str, float] = {f["name"]: float(f["default_val"]) for f in FEATURE_DEFINITIONS}
FEATURE_UNITS_BY_NAME: dict[str, str] = {f["name"]: str(f["unit"]) for f in FEATURE_DEFINITIONS}


def resolve_feature_vector(
    features: list[float] | None = None,
    named_features: Mapping[str, float] | None = None,
    schema_version: str = "nhanes-87-v2",
) -> tuple[list[float], dict[str, Any]]:
    """Resolve explicit 87-feature vector or named feature map with missing-input audit."""
    if features is None and not named_features:
        raise ValueError("Provide either 'features' (87 values) or 'named_features'.")
    if features is not None:
        if len(features) == 88:
            raise ValueError(
                "Ambiguous 88-value legacy vector rejected to prevent silent feature misalignment. "
                "Provide exactly 87 features matching schema nhanes-87-v2 (HbA1c excluded) or use named_features."
            )
        if len(features) != 87:
            raise ValueError(f"Expected exactly 87 numeric features; got {len(features)}.")
        values = [float(v) for v in features]
        miss_flags = [
            FEATURE_NAMES_ORDERED[i]
            for i, name in enumerate(FEATURE_NAMES_ORDERED)
            if name.endswith("_MISS") and values[i] >= 0.5
        ]
        return values, {
            "schema_version": schema_version,
            "input_mode": "positional_87",
            "observed_feature_count": 87 - len(miss_flags),
            "imputed_default_features": [],
            "missingness_flags_set": miss_flags,
            "units_explicitly_verified": False,
        }

    # Named feature mode: reject unknown names and record which fields fell back to defaults
    provided = dict(named_features or {})
    unknown_keys = sorted(set(provided) - set(FEATURE_NAMES_ORDERED))
    if unknown_keys:
        raise ValueError(f"Unrecognized feature name(s) for schema {schema_version}: {unknown_keys}")
    values = []
    imputed: list[str] = []
    for name in FEATURE_NAMES_ORDERED:
        if name in provided and provided[name] is not None:
            values.append(float(provided[name]))
        else:
            values.append(FEATURE_DEFAULTS_BY_NAME[name])
            if not name.endswith("_MISS"):
                imputed.append(name)
    miss_flags = [
        FEATURE_NAMES_ORDERED[i]
        for i, name in enumerate(FEATURE_NAMES_ORDERED)
        if name.endswith("_MISS") and values[i] >= 0.5
    ]
    return values, {
        "schema_version": schema_version,
        "input_mode": "named_features",
        "observed_feature_count": len(provided),
        "imputed_default_features": imputed,
        "missingness_flags_set": miss_flags,
        "units_explicitly_verified": True,
    }


# ---------------------------------------------------------------------------
# Pipeline Execution Functions
# ---------------------------------------------------------------------------
def run_screening(
    features: list[float] | None = None,
    metadata_overrides: dict[str, Any] | None = None,
    *,
    named_features: Mapping[str, float] | None = None,
    schema_version: str = "nhanes-87-v2",
) -> dict[str, Any]:
    """Execute Stage 1 (Diagnostics) and Stage 2 (Counterfactual Optimization)."""
    with _MODEL_LOCK:
        if schema_version != 'nhanes-87-v2':
            raise ValueError('Unsupported input schema; expected nhanes-87-v2.')
        resolved_values, input_audit = resolve_feature_vector(
            features=features, named_features=named_features, schema_version=schema_version
        )
        if deployment.REPAIRED and input_audit.get('imputed_default_features'):
            raise ValueError('Repaired deployment requires complete inputs; sample defaults cannot replace missing measurements.')
        raw_vector = np.asarray(resolved_values, dtype=np.float64 if deployment.REPAIRED else np.float32)
        raw_vector, derived_audit = p12.s2.canonicalize_dietary_ratios(raw_vector)
        input_audit['derived_feature_audit'] = derived_audit

        # Build dummy CLI args namespace for resolve_patient_metadata
        overrides = metadata_overrides or {}
        args = argparse.Namespace(
            weight=overrides.get("weight_kg"),
            height=overrides.get("height_cm"),
            age=overrides.get("age_years"),
            female=overrides.get("is_female") is True,
            male=overrides.get("is_female") is False,
            activity=overrides.get("activity_level"),
        )
        patient_metadata = p12.resolve_patient_metadata(args, raw_vector)
        for key in ("renal_disease", "pregnant", "lactating"):
            patient_metadata[key] = overrides.get(key)

        # Stage 1: Diagnostics
        t0 = time.perf_counter()
        stage1_assessment, _ = p12.run_stage1_diagnostics(raw_vector)
        dt_s1_ms = (time.perf_counter() - t0) * 1000.0
        stage1_assessment["model_metadata"] = {
            **deployment.metadata(),
            "schema_version": schema_version,
            "input_audit": input_audit,
        }

        # Stage 2: Counterfactual Engine
        t1 = time.perf_counter()
        stage2_dossier = p12.run_stage2_counterfactual(raw_vector, patient_metadata)
        dt_s2_ms = (time.perf_counter() - t1) * 1000.0

        # Refresh explicit Stage 2 constraint specification metadata (v4 exposure-separated)
        for row in stage2_dossier.get("nutrient_constraint_envelope", []):
            code = row.get("nutrient")
            if code:
                row["rule_provenance"] = nrp.reviewed_rule_metadata(
                    code,
                    float(row.get("min_bound", 0.0)),
                    float(row.get("max_bound", 0.0)),
                )
        stage2_dossier["constraint_spec_version"] = nrp.RULESET_VERSION
        stage2_dossier["counterfactual_scope_note"] = (
            "Stage 2 counterfactual model-score changes are exploratory mathematical calculations, "
            "not guaranteed clinical risk reductions. Observed patient diet and proposed meal plans "
            "remain separate records."
        )

        unified_dossier = p12.build_unified_dossier(
            raw_vector=raw_vector,
            patient_metadata=patient_metadata,
            stage1_assessment=stage1_assessment,
            stage2_dossier=stage2_dossier,
        )

        return {
            "patient_metadata": patient_metadata,
            "probabilities": stage1_assessment["probabilities"],
            "classifications": stage1_assessment["classifications"],
            "top_feature_drivers": stage1_assessment["top_feature_drivers"],
            "stage2_prescription": {
                row['nutrient']: row['target']
                for row in stage2_dossier.get('nutrient_constraint_envelope', [])
            },
            "energy_prescription": stage2_dossier.get("energy_prescription", {}),
            "certified_for_stage3": bool(stage2_dossier.get("certified_for_stage3", False)),
            "latency_ms": {
                "stage1_ms": round(dt_s1_ms, 2),
                "stage2_ms": round(dt_s2_ms, 2),
                "total_s1_s2_ms": round(dt_s1_ms + dt_s2_ms, 2)
            },
            "dossier": unified_dossier
        }


def _enrich_schedule_items(
    schedule: list[dict[str, Any]], basket: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Ensure every item in every meal slot has a valid category."""
    cat_by_id = {str(b.get("food_id")): str(b.get("category", "other")) for b in basket}
    for slot in schedule:
        for item in slot.get("items", []):
            fid = str(item.get("food_id"))
            if not item.get("category"):
                item["category"] = cat_by_id.get(fid, "other")
    return schedule


def run_optimization(
    dossier: dict[str, Any],
    revision_spec: Mapping[str, Any] | None = None,
    *,
    dietary_preference: str = "veg_nonveg_combo",
    allergies: list[str] | None = None,
    excluded_food_ids: list[str] | None = None,
    disliked_terms: list[str] | None = None,
    max_active_dishes: int | None = None,
) -> dict[str, Any]:
    """Execute Stage 3 (HiGHS MILP Indian Meal Synthesis) with optional user preferences."""
    with _MODEL_LOCK:
        stage2_dossier = dossier.get("stage2_counterfactual_prescription") or dossier.get("stage2_prescription")
        if not stage2_dossier:
            raise ValueError("Dossier missing 'stage2_counterfactual_prescription'.")

        t0 = time.perf_counter()
        food_db = s3.load_food_database()

        mode_map = {
            "vegetarian": "pure_vegetarian",
            "pure_vegetarian": "pure_vegetarian",
            "veg_nonveg_combo": "veg_nonveg_combo",
            "combo": "veg_nonveg_combo",
            "non_vegetarian": "pure_non_vegetarian",
            "pure_non_vegetarian": "pure_non_vegetarian",
        }
        dietary_mode = mode_map.get(str(dietary_preference or "veg_nonveg_combo").strip().lower(), "veg_nonveg_combo")
        if dietary_mode != "pure_vegetarian" and not any(f.category == "non_veg_protein" for f in food_db):
            dietary_mode = "pure_vegetarian"

        # Merge explicit revision_spec with structured preference fields
        merged_spec: dict[str, Any] = dict(revision_spec or {})
        merged_spec.setdefault("dietary_mode", dietary_mode)
        excl_ids: set[str] = {str(x) for x in merged_spec.get("excluded_food_ids", [])}
        excl_ids.update(str(x) for x in (excluded_food_ids or []))

        # Match free-text disliked terms against dish names and food_ids (including 'lady finger' -> 'lady's fingers')
        clean_dislikes = [t.strip().lower() for t in (disliked_terms or []) if t and t.strip()]
        matched_dislikes: list[str] = []
        for term in clean_dislikes:
            norm_term = re.sub(r"[^a-z0-9 ]", " ", term).strip()
            toks = [t[:-1] if len(t) > 3 and t.endswith("s") else t for t in norm_term.split() if len(t) >= 3]
            for food in food_db:
                hay = re.sub(r"[^a-z0-9 ]", " ", f"{food.food_id} {food.name}".lower())
                if norm_term in hay or (toks and all(tok in hay for tok in toks)):
                    excl_ids.add(food.food_id)
                    matched_dislikes.append(food.food_id)

        # Normalize reported allergies and exclude dishes with documented matching allergens
        allergen_aliases = {
            "dairy": {"dairy", "milk", "curd", "yogurt", "lactose", "paneer", "ghee"},
            "tree_nuts": {"tree_nuts", "tree nut", "tree nuts", "almond", "almonds", "nut", "nuts"},
            "gluten": {"gluten", "wheat", "roti", "phulka"},
            "peanuts": {"peanut", "peanuts", "groundnut"},
            "mustard": {"mustard"},
            "coconut": {"coconut"},
            "egg": {"egg", "eggs"},
            "fish": {"fish", "salmon", "tuna", "seafood"},
        }
        normalized_allergies: list[str] = []
        allergy_excluded_ids: list[str] = []
        for raw_a in (allergies or []):
            a_low = raw_a.strip().lower()
            if not a_low:
                continue
            canonical = a_low
            for key, syns in allergen_aliases.items():
                if a_low in syns or any(s in a_low for s in syns):
                    canonical = key
                    break
            if canonical not in normalized_allergies:
                normalized_allergies.append(canonical)
            for food in food_db:
                known = [str(k).lower() for k in (food.allergen_info or {}).get("known_allergens", [])]
                if canonical in known or a_low in known or a_low in food.name.lower():
                    excl_ids.add(food.food_id)
                    allergy_excluded_ids.append(food.food_id)

        if excl_ids:
            merged_spec["excluded_food_ids"] = sorted(excl_ids)
        if max_active_dishes is not None and merged_spec.get("max_active_dishes") is None:
            merged_spec["max_active_dishes"] = int(max_active_dishes)

        stage3_output = s3.solve_meal_plan(
            stage2_dossier,
            food_db,
            time_limit_sec=2.0,
            revision_spec=merged_spec,
        )
        dt_s3_ms = (time.perf_counter() - t0) * 1000.0

        stage3_dict = stage3_output.to_dict()
        stage3_dict["dietary_mode"] = dietary_mode
        catalog_notes = {
            "pure_vegetarian": "Pure vegetarian plan generated from sourced lacto-vegetarian Indian & USDA entries.",
            "veg_nonveg_combo": "Vegetarian + non-vegetarian combo combining sourced Indian vegetarian dishes with non-veg protein options.",
            "pure_non_vegetarian": (
                "Non-vegetarian-focused plan prioritizing sourced non-veg protein mains alongside required "
                "whole grains and vegetables (essential for dietary fiber and vitamin C)."
            ),
        }
        stage3_dict["preferences_applied"] = {
            "dietary_preference": dietary_mode,
            "dietary_mode": dietary_mode,
            "reported_allergies": normalized_allergies,
            "excluded_food_ids": sorted(excl_ids),
            "allergy_excluded_food_ids": sorted(set(allergy_excluded_ids)),
            "disliked_terms": clean_dislikes,
            "matched_dislike_food_ids": sorted(set(matched_dislikes)),
            "max_active_dishes": merged_spec.get("max_active_dishes"),
            "catalog_note": catalog_notes[dietary_mode],
        }
        stage3_dict["revision"] = {
            "dietary_mode": dietary_mode,
            "excluded_food_ids": sorted(excl_ids),
            "max_active_dishes": merged_spec.get("max_active_dishes"),
            "status": "initial_with_preferences",
        }
        if normalized_allergies:
            stage3_dict["flagged_unsuitable_for_allergies"] = list(normalized_allergies)

        # If solve failed under user preferences, attach detailed conflict diagnosis
        failure_payload = stage3_output.failure
        if not stage3_output.accepted and failure_payload is not None:
            try:
                import chat_service
                conflict = chat_service.tool_explain_constraint_conflict(
                    failure_payload, stage2_dossier, food_db, merged_spec
                )
                failure_payload = dict(failure_payload)
                failure_payload["message"] = conflict.get("summary") or failure_payload.get("message")
                failure_payload["conflict_diagnosis"] = conflict
                stage3_dict["failure"] = failure_payload
            except Exception:
                pass

        dossier["stage3_discrete_meal_synthesis"] = stage3_dict
        dossier["stage3_meal_plan"] = stage3_dict

        # Build 5-slot deterministic meal schedule & enrich categories
        schedule = p14.build_deterministic_schedule(stage3_output.solved_food_basket) if stage3_output.accepted else []
        schedule = _enrich_schedule_items(schedule, stage3_output.solved_food_basket)

        return {
            "accepted": stage3_output.accepted,
            "failure": failure_payload,
            "clinical_verification": stage3_output.clinical_verification,
            "solved_food_basket": stage3_output.solved_food_basket,
            "meal_schedule": schedule,
            "delivered_nutrition": stage3_dict.get("delivered_nutrition", []),
            "solver_metadata": stage3_output.solver_metadata,
            "latency_ms": round(dt_s3_ms, 2),
            "dossier": dossier
        }


def run_compilation(dossier: dict[str, Any], backend: str = "transformers") -> dict[str, Any]:
    """Execute Stage 4 (Clinical Chrono-Nutrition Compiler via Local GPU LLM)."""
    validated_dossier = p14.validate_dossier(dossier)
    compiler_cfg = p14.CompilerConfig(backend=backend)
    report = p14.compile_patient_report(validated_dossier, config=compiler_cfg)
    if "compiled_patient_report" in report and "meal_schedule" in report["compiled_patient_report"]:
        report["compiled_patient_report"]["meal_schedule"] = _enrich_schedule_items(
            report["compiled_patient_report"]["meal_schedule"], validated_dossier.basket
        )
    return report


def run_full_pipeline(
    features: list[float] | None = None,
    metadata_overrides: dict[str, Any] | None = None,
    backend: str = "transformers",
    *,
    named_features: Mapping[str, float] | None = None,
    schema_version: str = "nhanes-87-v2",
) -> dict[str, Any]:
    """Execute complete 4-stage end-to-end pipeline."""
    t_start = time.perf_counter()

    # Stages 1 + 2
    screen_res = run_screening(
        features,
        metadata_overrides,
        named_features=named_features,
        schema_version=schema_version,
    )
    unified_dossier = screen_res["dossier"]

    # Stage 3
    opt_res = None
    if screen_res["certified_for_stage3"]:
        opt_res = run_optimization(unified_dossier)
        unified_dossier = opt_res["dossier"]

    # Stage 4
    report_res = None
    if opt_res is not None and opt_res.get('accepted'):
        report_res = run_compilation(unified_dossier, backend=backend)

    t_total_ms = (time.perf_counter() - t_start) * 1000.0

    s1_ms = screen_res["latency_ms"]["stage1_ms"]
    s2_ms = screen_res["latency_ms"]["stage2_ms"]
    s3_ms = opt_res["latency_ms"] if opt_res else 0.0
    s4_ms = report_res["compiler_metadata"]["generation_time_ms"] if report_res else 0.0

    return {
        "pipeline_status": 'blocked_stage2' if opt_res is None else ('blocked_stage3' if not opt_res.get('accepted') else 'complete'),
        "pipeline_failure": ({"code": "stage2_checks_failed", "message": "Stage 2 programmed checks did not pass. No meal plan was accepted."}
                             if opt_res is None else opt_res.get('failure')),
        "screening": screen_res,
        "meal_plan": opt_res,
        "clinical_report": report_res,
        "pipeline_latency": {
            "stage1_diagnostics_ms": s1_ms,
            "stage2_counterfactual_ms": s2_ms,
            "stage3_milp_solver_ms": s3_ms,
            "stage4_compiler_ms": s4_ms,
            "total_pipeline_ms": round(t_total_ms, 2)
        },
        "mode_a_subsecond": bool((s1_ms + s2_ms + s3_ms + 25.0) < 1000.0)
    }


def get_food_catalog() -> list[dict[str, Any]]:
    """Return list of dishes from the sourced cooked database."""
    global _FOOD_CATALOG_CACHE
    if _FOOD_CATALOG_CACHE is not None:
        return _FOOD_CATALOG_CACHE

    db = s3.load_food_database()
    items: list[dict[str, Any]] = []
    for food in db:
        items.append({
            "food_id": food.food_id,
            "name": food.name,
            "category": food.category,
            "serving_unit": food.serving_unit,
            "serving_weight_g": food.serving_weight_g,
            "data_source": food.data_source,
            "catalog_version": food.catalog_version,
            "unknown_nutrients": list(food.unknown_nutrients),
            "allergen_info": dict(food.allergen_info),
        })
    items.sort(key=lambda x: (x["category"], x["name"]))
    _FOOD_CATALOG_CACHE = items
    return items


def get_hardware_status() -> dict[str, Any]:
    """Check GPU availability, VRAM utilization, and device name."""
    cuda_avail = torch.cuda.is_available()
    device_name = None
    vram_total = None
    vram_free = None

    if cuda_avail:
        device_name = torch.cuda.get_device_name(0)
        total_b = torch.cuda.get_device_properties(0).total_memory
        reserved_b = torch.cuda.memory_reserved(0)
        vram_total = round(total_b / (1024**3), 2)
        vram_free = round((total_b - reserved_b) / (1024**3), 2)

    return {
        "status": "ready",
        "cuda_available": cuda_avail,
        "cuda_device": device_name,
        "vram_total_gb": vram_total,
        "vram_free_gb": vram_free,
        "pipeline_loaded": True
    }


# ---------------------------------------------------------------------------
# Document & Photo Clinical Feature Extraction
# ---------------------------------------------------------------------------

CLINICAL_FEATURE_ALIASES: dict[str, list[str]] = {
    "LBDHDD": [
        "hdl", "hdl cholesterol", "high density lipoprotein", "hdl-c", "hdl-chol", "s. hdl", "serum hdl", "good cholesterol"
    ],
    "LBXTC": [
        "total cholesterol", "s. cholesterol", "serum cholesterol", "cholesterol total", "cholesterol, total", "total chol", "s. chol", "cholesterol"
    ],
    "NON_HDL_CHOL": [
        "non-hdl cholesterol", "non hdl", "non-hdl-c", "non hdl cholesterol", "non-hdl chol"
    ],
    "CHOL_HDL_RATIO": [
        "total / hdl ratio", "chol / hdl ratio", "total/hdl", "tc/hdl", "tc/hdl ratio", "chol/hdl"
    ],
    "NON_HDL_TO_HDL_RATIO": [
        "non-hdl / hdl ratio", "non-hdl/hdl", "non hdl to hdl"
    ],
    "DR1TSODI": [
        "sodium", "serum sodium", "na+", "s. sodium"
    ],
    "DR1TPOTA": [
        "potassium", "serum potassium", "k+", "s. potassium"
    ],
    "DR1TCALC": [
        "calcium", "total calcium", "serum calcium", "ca++", "s. calcium"
    ],
    "DR1TMAGN": [
        "magnesium", "serum magnesium", "mg++", "s. magnesium"
    ],
    "DR1TPHOS": [
        "phosphorus", "serum phosphorus", "phosphate"
    ],
    "DR1TIRON": [
        "iron", "serum iron", "total iron", "fe"
    ],
    "DR1TZINC": [
        "zinc", "serum zinc"
    ],
    "DR1TKCAL": [
        "calories", "energy", "total calories", "total energy", "caloric intake", "kcal"
    ],
    "DR1TPROT": [
        "protein", "total protein", "dietary protein"
    ],
    "DR1TCARB": [
        "carbohydrates", "carbs", "total carbohydrates", "dietary carbohydrate"
    ],
    "DR1TTFAT": [
        "total fat", "dietary fat", "fat"
    ],
    "DR1TSFAT": [
        "saturated fat", "saturated fatty acids", "sat fat"
    ],
    "DR1TMFAT": [
        "monounsaturated fat", "mufa", "mono fat"
    ],
    "DR1TPFAT": [
        "polyunsaturated fat", "pufa", "poly fat"
    ],
    "DR1TFIBE": [
        "dietary fiber", "fiber", "total fiber"
    ],
    "DR1TSUGR": [
        "total sugars", "sugar", "sugars", "dietary sugar"
    ],
    "DR1TCHOL": [
        "dietary cholesterol"
    ],
    "DR1TVC": [
        "vitamin c", "ascorbic acid"
    ],
    "DR1TVB12": [
        "vitamin b12", "cobalamin", "b12"
    ],
    "DR1TVARA": [
        "vitamin a"
    ],
    "DR1TATOC": [
        "vitamin e", "alpha-tocopherol"
    ],
    "SLEEP_HOURS": [
        "sleep", "sleep duration", "sleep hours", "sleep per night", "hours of sleep", "nocturnal sleep"
    ],
}


def extract_metadata_from_text(document_text: str) -> dict[str, Any]:
    """Extract clinical metadata from patient headers (Age, Sex, Weight, Height, Activity)."""
    metadata: dict[str, Any] = {
        "age_years": None,
        "weight_kg": None,
        "height_cm": None,
        "is_female": None,
        "activity_level": None,
    }
    # Age
    age_match = re.search(r"\b(?:age|patient\s+age)\s*[:=]?\s*(\d{1,3})\b", document_text, re.I)
    if not age_match:
        age_match = re.search(r"\b(\d{1,3})\s*(?:years?|yrs?|y)\s*(?:old)?\b", document_text, re.I)
    if age_match:
        try:
            val = float(age_match.group(1))
            if 1 <= val <= 120:
                metadata["age_years"] = val
        except ValueError:
            pass

    # Sex / Gender
    sex_match = re.search(r"\b(?:sex|gender)\s*[:=]?\s*(female|male|f|m)\b", document_text, re.I)
    if not sex_match:
        sex_match = re.search(r"\b(?:\d{1,3}\s*(?:y|yrs?|years?)\s*[/,-]\s*)(female|male|f|m)\b", document_text, re.I)
    if sex_match:
        s = sex_match.group(1).lower()
        metadata["is_female"] = s in ("female", "f")

    # Weight
    w_match = re.search(r"\b(?:weight|wt|body\s+weight)\s*[:=]?\s*(\d{2,3}(?:\.\d+)?)\s*(?:kg|kgs)?\b", document_text, re.I)
    if w_match:
        try:
            w = float(w_match.group(1))
            if 20 <= w <= 300:
                metadata["weight_kg"] = w
        except ValueError:
            pass

    # Height
    h_match = re.search(r"\b(?:height|ht|body\s+height)\s*[:=]?\s*(\d{2,3}(?:\.\d+)?)\s*(?:cm)?\b", document_text, re.I)
    if h_match:
        try:
            h = float(h_match.group(1))
            if 60 <= h <= 250:
                metadata["height_cm"] = h
        except ValueError:
            pass

    # Activity level
    act_match = re.search(r"\b(?:activity\s+level|activity)\s*[:=]?\s*(sedentary|light|moderate|active)\b", document_text, re.I)
    if act_match:
        metadata["activity_level"] = act_match.group(1).lower()

    return metadata


def extract_document_pipeline_features(document_text: str) -> dict[str, Any]:
    """Extract clinical entities from report text and assemble into exact 87-feature project format."""
    features: dict[str, float] = {}
    lines = document_text.splitlines()
    extracted_items: list[dict[str, Any]] = []

    # 1. Header Metadata
    metadata = extract_metadata_from_text(document_text)
    if metadata["age_years"] is not None:
        features["RIDAGEYR"] = metadata["age_years"]
        extracted_items.append({"name": "RIDAGEYR", "label": "Age", "value": metadata["age_years"], "unit": "years", "is_derived": False})
    if metadata["is_female"] is not None:
        features["SEX_FEMALE"] = 1.0 if metadata["is_female"] else 0.0
        extracted_items.append({"name": "SEX_FEMALE", "label": "Sex (Female=1, Male=0)", "value": features["SEX_FEMALE"], "unit": "0 or 1", "is_derived": False})

    unit_aliases = {
        "mg/dL": r"mg\s*/\s*dL",
        "g": r"\bg\b",
        "mg": r"\bmg\b",
        "mcg": r"(?:mcg|µg|ug)",
        "kcal": r"\bkcal\b",
        "hours/night": r"(?:hours?|hrs?)(?:\s*/\s*night)?",
        "years": r"\byears?\b",
        "ratio": r"\bratio\b",
        "%": r"%",
        "min/day": r"min(?:utes)?\s*/\s*day",
        "min/week": r"min(?:utes)?\s*/\s*w(?:ee)?k",
        "MET-min/wk": r"MET",
    }

    # 2. Extract features by alias or label
    for definition in FEATURE_DEFINITIONS:
        code = definition["name"]
        if code in features:
            continue

        aliases = [code, definition["label"]]
        if code in CLINICAL_FEATURE_ALIASES:
            aliases.extend(CLINICAL_FEATURE_ALIASES[code])

        matches = []
        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            matched = False
            for alias in aliases:
                pattern = r"\b" + re.escape(alias) + r"\b"
                m = re.search(pattern, line_str, re.I)
                if m:
                    matched = True
                    tail = line_str[m.end():]
                    numbers = re.findall(r"(?<![\w.])[-+]?(?:\d+\.?\d*|\.\d+)(?![\w.])", tail)
                    if numbers:
                        matches.append((line_str, numbers[0]))
                    break
            if matched and matches:
                break

        if not matches:
            continue

        evidence, raw_value = matches[0]
        expected_unit = definition["unit"]
        unit_pattern = unit_aliases.get(expected_unit)
        has_unit = bool(unit_pattern and re.search(unit_pattern, evidence, re.I))
        is_binary = expected_unit == "0 or 1" and raw_value in {"0", "1"}

        if not has_unit and not is_binary:
            if code not in {"LBXTC", "LBDHDD", "NON_HDL_CHOL", "DR1TKCAL", "DR1TPROT", "DR1TCARB", "DR1TTFAT"}:
                continue

        try:
            val = float(raw_value)
            if math.isfinite(val) and definition["min_val"] <= val <= definition["max_val"]:
                features[code] = val
                extracted_items.append({"name": code, "label": definition["label"], "value": val, "unit": definition["unit"], "is_derived": False})
        except ValueError:
            continue

    # 3. Clinical lipid derivations
    if "LBXTC" in features and "LBDHDD" in features:
        tc = features["LBXTC"]
        hdl = features["LBDHDD"]
        if "NON_HDL_CHOL" not in features:
            val = round(tc - hdl, 2)
            features["NON_HDL_CHOL"] = val
            extracted_items.append({"name": "NON_HDL_CHOL", "label": "Non-HDL Cholesterol (Derived)", "value": val, "unit": "mg/dL", "is_derived": True})
        if "CHOL_HDL_RATIO" not in features and hdl > 0:
            val = round(tc / hdl, 2)
            features["CHOL_HDL_RATIO"] = val
            extracted_items.append({"name": "CHOL_HDL_RATIO", "label": "Total / HDL Ratio (Derived)", "value": val, "unit": "ratio", "is_derived": True})
        if "NON_HDL_TO_HDL_RATIO" not in features and hdl > 0:
            val = round((tc - hdl) / hdl, 2)
            features["NON_HDL_TO_HDL_RATIO"] = val
            extracted_items.append({"name": "NON_HDL_TO_HDL_RATIO", "label": "Non-HDL / HDL Ratio (Derived)", "value": val, "unit": "ratio", "is_derived": True})

    # 4. Dietary macronutrient ratio derivations
    if "DR1TKCAL" in features and features["DR1TKCAL"] > 0:
        kcal = features["DR1TKCAL"]
        if "DR1TPROT" in features and "PROT_PCT_KCAL" not in features:
            val = round((features["DR1TPROT"] * 4.0 / kcal) * 100.0, 2)
            features["PROT_PCT_KCAL"] = val
            extracted_items.append({"name": "PROT_PCT_KCAL", "label": "Protein % of Energy (Derived)", "value": val, "unit": "%", "is_derived": True})
        if "DR1TCARB" in features and "CARB_PCT_KCAL" not in features:
            val = round((features["DR1TCARB"] * 4.0 / kcal) * 100.0, 2)
            features["CARB_PCT_KCAL"] = val
            extracted_items.append({"name": "CARB_PCT_KCAL", "label": "Carb % of Energy (Derived)", "value": val, "unit": "%", "is_derived": True})
        if "DR1TTFAT" in features and "FAT_PCT_KCAL" not in features:
            val = round((features["DR1TTFAT"] * 9.0 / kcal) * 100.0, 2)
            features["FAT_PCT_KCAL"] = val
            extracted_items.append({"name": "FAT_PCT_KCAL", "label": "Fat % of Energy (Derived)", "value": val, "unit": "%", "is_derived": True})
        if "DR1TFIBE" in features and "FIBER_PER_1000_KCAL" not in features:
            val = round(features["DR1TFIBE"] / (kcal / 1000.0), 2)
            features["FIBER_PER_1000_KCAL"] = val
            extracted_items.append({"name": "FIBER_PER_1000_KCAL", "label": "Fiber per 1000 kcal (Derived)", "value": val, "unit": "g/1000kcal", "is_derived": True})

    if "DR1TSUGR" in features and "DR1TFIBE" in features and "SUGAR_TO_FIBER_RATIO" not in features:
        val = round(features["DR1TSUGR"] / (features["DR1TFIBE"] + 1.0), 2)
        features["SUGAR_TO_FIBER_RATIO"] = val
        extracted_items.append({"name": "SUGAR_TO_FIBER_RATIO", "label": "Sugar-to-Fiber Ratio (Derived)", "value": val, "unit": "ratio", "is_derived": True})

    if "DR1TSODI" in features and "DR1TPOTA" in features and "SODIUM_POTASSIUM_RATIO" not in features:
        val = round(features["DR1TSODI"] / (features["DR1TPOTA"] + 1.0), 2)
        features["SODIUM_POTASSIUM_RATIO"] = val
        extracted_items.append({"name": "SODIUM_POTASSIUM_RATIO", "label": "Sodium-to-Potassium Ratio (Derived)", "value": val, "unit": "ratio", "is_derived": True})

    if "SLEEP_HOURS" in features and "SLEEP_DEV_FROM_7_5" not in features:
        val = round(features["SLEEP_HOURS"] - 7.5, 2)
        features["SLEEP_DEV_FROM_7_5"] = val
        extracted_items.append({"name": "SLEEP_DEV_FROM_7_5", "label": "Sleep Deviation from 7.5h (Derived)", "value": val, "unit": "hours", "is_derived": True})

    # 5. Build full 87-feature array in exact FEATURE_DEFINITIONS order
    defaults_by_name = {f["name"]: float(f["default_val"]) for f in FEATURE_DEFINITIONS}
    ordered_87: list[float] = []
    for field in FEATURE_DEFINITIONS:
        name = field["name"]
        if name in features:
            ordered_87.append(features[name])
        else:
            ordered_87.append(defaults_by_name.get(name, 0.0))

    # 6. Format raw vector text with clinical metadata
    raw_parts = [str(round(v, 4) if isinstance(v, float) else v) for v in ordered_87]
    meta_parts = []
    if metadata["weight_kg"] is not None:
        meta_parts.append(f"weight_kg={metadata['weight_kg']}")
    else:
        meta_parts.append("weight_kg=74.0")
    if metadata["height_cm"] is not None:
        meta_parts.append(f"height_cm={metadata['height_cm']}")
    else:
        meta_parts.append("height_cm=176.0")
    if metadata["age_years"] is not None:
        meta_parts.append(f"age_years={metadata['age_years']}")
    elif "RIDAGEYR" in features:
        meta_parts.append(f"age_years={features['RIDAGEYR']}")
    if metadata["is_female"] is not None:
        meta_parts.append(f"is_female={'true' if metadata['is_female'] else 'false'}")
    elif "SEX_FEMALE" in features:
        meta_parts.append(f"is_female={'true' if features['SEX_FEMALE'] == 1.0 else 'false'}")
    if metadata["activity_level"] is not None:
        meta_parts.append(f"activity_level={metadata['activity_level']}")
    else:
        meta_parts.append("activity_level=sedentary")

    raw_vector_text = " ".join(raw_parts) + "\n# CLINICAL_METADATA: " + " ".join(meta_parts)

    summary_labels = [item["label"].replace(" (Derived)", "") for item in extracted_items]
    summary_msg = f"Retrieved {len(extracted_items)} inputs from document ({', '.join(summary_labels[:6])}{', ...' if len(summary_labels) > 6 else ''}). Formatted into 87-feature project order."

    # Fill metadata defaults if not found
    final_metadata = {
        "weight_kg": metadata["weight_kg"] if metadata["weight_kg"] is not None else 74.0,
        "height_cm": metadata["height_cm"] if metadata["height_cm"] is not None else 176.0,
        "age_years": metadata["age_years"] if metadata["age_years"] is not None else (features.get("RIDAGEYR", 42.0)),
        "is_female": metadata["is_female"] if metadata["is_female"] is not None else (bool(features.get("SEX_FEMALE", 0))),
        "activity_level": metadata["activity_level"] or "sedentary",
        "renal_disease": False,
        "pregnant": False,
        "lactating": False,
    }

    return {
        "features": ordered_87,
        "metadata": final_metadata,
        "feature_count": len(ordered_87),
        "is_valid": True,
        "summary": summary_msg,
        "raw_vector_text": raw_vector_text,
        "extracted_items": extracted_items,
    }
