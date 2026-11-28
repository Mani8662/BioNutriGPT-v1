"""
Comprehensive Stress & Fuzz Testing Suite for BioNutriGPT Stage 1 & Stage 2
Tests thousands of possibilities across both real NHANES cohorts and extreme synthetic edge cases.
"""

import sys
import os
import time
import json
import math
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import train_bionutrigpt_stage1 as s1
import stage2_counterfactual_engine as s2
import infer_bionutrigpt_pipeline_Stage_1_and_2 as pipeline

CHECKPOINT_PATH = PROJECT_DIR / "Saved Model" / "bionutrigpt_stage1_best.pt"
PREPROCESSOR_PATH = PROJECT_DIR / "Data" / "preprocessing_artifacts.joblib"
TEST_DATA_PATH = PROJECT_DIR / "Data" / "BioNutriGPT_test_2017_2020_ml_ready.csv"

def run_stress_test(num_real=500, num_synthetic=1500):
    print("=" * 80)
    print("BIONUTRIGPT STAGE 1 & STAGE 2 MASSIVE STRESS TEST")
    print(f"Testing {num_real} Real NHANES Patients + {num_synthetic} Extreme Synthetic Cases")
    print("=" * 80)

    artifacts = s2._load_preprocessing_artifacts(PREPROCESSOR_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = s2._load_stage1_model(CHECKPOINT_PATH, artifacts, device)

    # Load real test dataset
    print(f"\n[1] Loading real test data from {TEST_DATA_PATH.name}...")
    df_test = pd.read_csv(TEST_DATA_PATH)
    print(f"Loaded {len(df_test)} real prospective patients.")

    # Results counters
    total_tested = 0
    stage1_successes = 0
    stage1_crashes = []
    
    stage2_successes = 0
    stage2_certified = 0
    stage2_crashes = []
    
    filter_failures = {
        "filter_1": 0,
        "filter_2": 0,
        "filter_3": 0,
        "filter_4": 0,
    }
    
    xai_successes = 0
    xai_crashes = []
    
    report_successes = 0
    report_crashes = []

    # -------------------------------------------------------------
    # TRACK 1: REAL NHANES PROSPECTIVE PATIENTS
    # -------------------------------------------------------------
    print(f"\n[2] Executing Track 1: {num_real} Real Cohort Patients...")
    feature_cols = list(s1.FEATURE_COLS)
    
    # Sample diverse real patients
    indices = np.linspace(0, len(df_test) - 1, num_real, dtype=int)
    
    start_time = time.time()
    for idx_i, idx in enumerate(indices):
        total_tested += 1
        row = df_test.iloc[idx]
        raw_vec = row[feature_cols].to_numpy(dtype=np.float32)
        
        # Patient metadata from row
        age = float(raw_vec[71]) # RIDAGEYR
        is_female = bool(raw_vec[74] >= 0.5) # SEX_FEMALE
        weight = 75.0 # fallback
        height = 170.0 # fallback
        activity = "sedentary"
        
        meta = {
            "age_years": age,
            "is_female": is_female,
            "weight_kg": weight,
            "height_cm": height,
            "activity_level": activity,
        }
        
        # Test Stage 1
        try:
            s1_assessment, _ = pipeline.run_stage1_diagnostics(
                raw_vector=raw_vec,
                checkpoint_path=CHECKPOINT_PATH,
                preprocessor_path=PREPROCESSOR_PATH,
            )
            stage1_successes += 1
        except Exception as e:
            stage1_crashes.append({"case": f"Real_{idx}", "error": str(e), "tb": traceback.format_exc()})
            continue

        # Test Stage 2
        try:
            s2_dossier = s2.generate_counterfactual_prescription(
                raw_patient_vector_88=raw_vec,
                patient_metadata=meta,
            )
            stage2_successes += 1
            if s2_dossier.get("certified_for_stage3", False):
                stage2_certified += 1
            else:
                filters = s2_dossier.get("audit_filters", {})
                if not filters.get("filter_1_thermodynamic_balance", {}).get("passed_after_all_later_filters", True):
                    filter_failures["filter_1"] += 1
                if not filters.get("filter_2_iom_safety_clamps", {}).get("passed_after_all_later_filters", True):
                    filter_failures["filter_2"] += 1
                if not filters.get("filter_3_clinical_ratio_coherence", {}).get("passed_after_all_later_filters", True):
                    filter_failures["filter_3"] += 1
                if not filters.get("filter_4_closed_loop_rescoring", {}).get("passed", True):
                    filter_failures["filter_4"] += 1
        except Exception as e:
            stage2_crashes.append({"case": f"Real_{idx}", "error": str(e), "tb": traceback.format_exc()})
            continue

        # Test XAI
        try:
            xai = pipeline.compute_clinical_etiology_and_risks(
                raw_vector=raw_vec,
                patient_metadata=meta,
                stage1_assessment=s1_assessment,
                stage2_dossier=s2_dossier,
            )
            xai_successes += 1
        except Exception as e:
            xai_crashes.append({"case": f"Real_{idx}", "error": str(e), "tb": traceback.format_exc()})
            continue

        # Test Clinical Report Rendering
        try:
            rep = pipeline.render_clinical_report(
                raw_vector=raw_vec,
                patient_metadata=meta,
                stage1_assessment=s1_assessment,
                stage2_dossier=s2_dossier,
                stage3_certified=s2_dossier.get("certified_for_stage3", False),
            )
            report_successes += 1
        except Exception as e:
            report_crashes.append({"case": f"Real_{idx}", "error": str(e), "tb": traceback.format_exc()})
            continue

        if (idx_i + 1) % 50 == 0:
            print(f"  Processed {idx_i + 1}/{num_real} real patients (Elapsed: {time.time() - start_time:.1f}s)...")

    print(f"Track 1 Real Patients Completed: {stage1_successes}/{num_real} Stage 1, {stage2_successes}/{num_real} Stage 2.")

    # -------------------------------------------------------------
    # TRACK 2: EXTREME SYNTHETIC FUZZ & BOUNDARY CASES
    # -------------------------------------------------------------
    print(f"\n[3] Executing Track 2: {num_synthetic} Extreme Synthetic Boundary Cases...")
    np.random.seed(42)
    
    # We will test various archetype generators
    base_patient = df_test.iloc[0][feature_cols].to_numpy(dtype=np.float32)

    for i in range(num_synthetic):
        total_tested += 1
        syn_vec = base_patient.copy()
        
        # Select archetype
        mode = i % 10
        if mode == 0:
            # Extreme High Caloric & Nutrient Intake
            syn_vec[6] = np.random.uniform(4000.0, 7500.0) # KCAL
            syn_vec[7] = np.random.uniform(150.0, 300.0)   # PROT
            syn_vec[8] = np.random.uniform(500.0, 900.0)   # CARB
            syn_vec[11] = np.random.uniform(150.0, 300.0)  # TFAT
            weight = np.random.uniform(110.0, 180.0)
            height = np.random.uniform(180.0, 210.0)
            age = np.random.uniform(20.0, 45.0)
            is_female = False
            activity = "active"
        elif mode == 1:
            # Low Caloric / Severe Anorexia / Malnutrition
            syn_vec[6] = np.random.uniform(500.0, 1100.0)  # KCAL
            syn_vec[7] = np.random.uniform(10.0, 40.0)     # PROT
            syn_vec[8] = np.random.uniform(30.0, 100.0)    # CARB
            syn_vec[11] = np.random.uniform(10.0, 30.0)    # TFAT
            weight = np.random.uniform(24.0, 45.0)
            height = np.random.uniform(145.0, 175.0)
            age = np.random.uniform(18.0, 35.0)
            is_female = True
            activity = "sedentary"
        elif mode == 2:
            # Morbid Obesity Multi-Morbid
            syn_vec[0] = np.random.uniform(9.0, 14.0)      # HbA1c
            syn_vec[2] = np.random.uniform(260.0, 450.0)   # Total Chol
            syn_vec[3] = np.random.uniform(200.0, 380.0)   # Non-HDL
            syn_vec[6] = np.random.uniform(3000.0, 5000.0) # KCAL
            weight = np.random.uniform(130.0, 220.0)
            height = np.random.uniform(155.0, 175.0)
            age = np.random.uniform(45.0, 75.0)
            is_female = bool(np.random.rand() > 0.5)
            activity = "sedentary"
        elif mode == 3:
            # Geriatric Sarcopenic (Age 75-85)
            syn_vec[71] = np.random.uniform(75.0, 85.0)    # Age
            syn_vec[7] = np.random.uniform(25.0, 50.0)     # Low Prot
            syn_vec[54] = np.random.uniform(600.0, 960.0)  # High Sitting
            weight = np.random.uniform(45.0, 70.0)
            height = np.random.uniform(150.0, 170.0)
            age = float(syn_vec[71])
            is_female = bool(np.random.rand() > 0.5)
            activity = "light"
        elif mode == 4:
            # Extreme Sodium / Low Potassium (Hypertension / Renal Trap)
            syn_vec[40] = np.random.uniform(5000.0, 10000.0) # Sodium
            syn_vec[41] = np.random.uniform(500.0, 1200.0)   # Potassium
            weight = np.random.uniform(65.0, 95.0)
            height = np.random.uniform(160.0, 185.0)
            age = np.random.uniform(35.0, 65.0)
            is_female = False
            activity = "moderate"
        elif mode == 5:
            # High Sugar Zero Fiber (Glycemic Shock)
            syn_vec[9] = np.random.uniform(120.0, 250.0)   # Sugar
            syn_vec[10] = np.random.uniform(0.0, 5.0)      # Fiber
            weight = np.random.uniform(70.0, 105.0)
            height = np.random.uniform(165.0, 180.0)
            age = np.random.uniform(25.0, 55.0)
            is_female = True
            activity = "sedentary"
        elif mode == 6:
            # High Alcohol & Toxicological Strain
            syn_vec[44] = np.random.uniform(50.0, 150.0)   # Alcohol
            syn_vec[43] = np.random.uniform(400.0, 1200.0) # Caffeine
            weight = np.random.uniform(70.0, 95.0)
            height = np.random.uniform(170.0, 185.0)
            age = np.random.uniform(30.0, 60.0)
            is_female = False
            activity = "moderate"
        elif mode == 7:
            # Severe Sleep Deprivation & Shift Worker
            syn_vec[52] = np.random.uniform(2.0, 4.5)      # Sleep hours
            syn_vec[53] = syn_vec[52] - 7.5                # Sleep deviation
            syn_vec[61] = 1.0                              # Sleep trouble flag
            weight = np.random.uniform(60.0, 90.0)
            height = np.random.uniform(160.0, 180.0)
            age = np.random.uniform(25.0, 50.0)
            is_female = bool(np.random.rand() > 0.5)
            activity = "sedentary"
        elif mode == 8:
            # Ultra-Athletic Hyper-Active (High MET)
            syn_vec[60] = np.random.uniform(4000.0, 12000.0) # Total MET
            syn_vec[55] = np.random.uniform(300.0, 600.0)    # Vigorous work/rec
            weight = np.random.uniform(70.0, 90.0)
            height = np.random.uniform(175.0, 195.0)
            age = np.random.uniform(20.0, 35.0)
            is_female = bool(np.random.rand() > 0.5)
            activity = "active"
        else:
            # Random uniform perturbation of all continuous features
            perturbation = np.random.uniform(0.5, 1.8, size=88).astype(np.float32)
            syn_vec = np.clip(base_patient * perturbation, 0.0, 1e5)
            weight = float(np.random.uniform(45.0, 120.0))
            height = float(np.random.uniform(150.0, 195.0))
            age = float(np.random.uniform(18.0, 80.0))
            is_female = bool(np.random.rand() > 0.5)
            activity = np.random.choice(["sedentary", "light", "moderate", "active"])

        # Recompute ratio features in syn_vec
        kcal = syn_vec[6]
        if kcal > 0:
            syn_vec[45] = (syn_vec[7] * 4.0) / kcal * 100.0
            syn_vec[46] = (syn_vec[8] * 4.0) / kcal * 100.0
            syn_vec[47] = (syn_vec[11] * 9.0) / kcal * 100.0
            syn_vec[48] = syn_vec[10] / (kcal / 1000.0)
        syn_vec[49] = syn_vec[9] / (syn_vec[10] + 1.0)
        syn_vec[50] = syn_vec[40] / (syn_vec[41] + 1.0)
        syn_vec[51] = syn_vec[12] / (syn_vec[13] + syn_vec[14] + 0.1)

        syn_vec[71] = age
        syn_vec[74] = 1.0 if is_female else 0.0

        meta = {
            "age_years": age,
            "is_female": is_female,
            "weight_kg": weight,
            "height_cm": height,
            "activity_level": activity,
        }

        # Test Stage 1
        try:
            s1_assessment, _ = pipeline.run_stage1_diagnostics(
                raw_vector=syn_vec,
                checkpoint_path=CHECKPOINT_PATH,
                preprocessor_path=PREPROCESSOR_PATH,
            )
            stage1_successes += 1
        except Exception as e:
            stage1_crashes.append({"case": f"Syn_{i}_mode{mode}", "error": str(e), "tb": traceback.format_exc()})
            continue

        # Test Stage 2
        try:
            s2_dossier = s2.generate_counterfactual_prescription(
                raw_patient_vector_88=syn_vec,
                patient_metadata=meta,
            )
            stage2_successes += 1
            if s2_dossier.get("certified_for_stage3", False):
                stage2_certified += 1
            else:
                filters = s2_dossier.get("audit_filters", {})
                if not filters.get("filter_1_thermodynamic_balance", {}).get("passed_after_all_later_filters", True):
                    filter_failures["filter_1"] += 1
                if not filters.get("filter_2_iom_safety_clamps", {}).get("passed_after_all_later_filters", True):
                    filter_failures["filter_2"] += 1
                if not filters.get("filter_3_clinical_ratio_coherence", {}).get("passed_after_all_later_filters", True):
                    filter_failures["filter_3"] += 1
                if not filters.get("filter_4_closed_loop_rescoring", {}).get("passed", True):
                    filter_failures["filter_4"] += 1
        except Exception as e:
            stage2_crashes.append({"case": f"Syn_{i}_mode{mode}", "error": str(e), "tb": traceback.format_exc()})
            continue

        # Test XAI
        try:
            xai = pipeline.compute_clinical_etiology_and_risks(
                raw_vector=syn_vec,
                patient_metadata=meta,
                stage1_assessment=s1_assessment,
                stage2_dossier=s2_dossier,
            )
            xai_successes += 1
        except Exception as e:
            xai_crashes.append({"case": f"Syn_{i}_mode{mode}", "error": str(e), "tb": traceback.format_exc()})
            continue

        # Test Clinical Report Rendering
        try:
            rep = pipeline.render_clinical_report(
                raw_vector=syn_vec,
                patient_metadata=meta,
                stage1_assessment=s1_assessment,
                stage2_dossier=s2_dossier,
                stage3_certified=s2_dossier.get("certified_for_stage3", False),
            )
            report_successes += 1
        except Exception as e:
            report_crashes.append({"case": f"Syn_{i}_mode{mode}", "error": str(e), "tb": traceback.format_exc()})
            continue

        if (i + 1) % 200 == 0:
            print(f"  Processed {i + 1}/{num_synthetic} synthetic stress cases (Elapsed: {time.time() - start_time:.1f}s)...")

    # -------------------------------------------------------------
    # STRESS TEST AUDIT REPORT
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STRESS TEST AUDIT SUMMARY REPORT")
    print("=" * 80)
    print(f"Total Test Cases Evaluated : {total_tested}")
    print(f"Stage 1 Diagnostic Passes  : {stage1_successes}/{total_tested} ({stage1_successes/total_tested*100:.2f}%)")
    print(f"Stage 1 Unhandled Crashes  : {len(stage1_crashes)}")
    print(f"Stage 2 Prescriptions Run  : {stage2_successes}/{total_tested} ({stage2_successes/total_tested*100:.2f}%)")
    print(f"Stage 2 Unhandled Crashes  : {len(stage2_crashes)}")
    print(f"Stage 2 Certified Cases    : {stage2_certified}/{stage2_successes} ({stage2_certified/max(1, stage2_successes)*100:.2f}%)")
    print(f"XAI Engine Computations    : {xai_successes}/{total_tested} ({xai_successes/total_tested*100:.2f}%)")
    print(f"Report Text Generations    : {report_successes}/{total_tested} ({report_successes/total_tested*100:.2f}%)")
    
    print("\nStage 2 Audit Filter Rejection Breakdown (Non-Certified Cases):")
    for fname, count in filter_failures.items():
        print(f"  - {fname}: {count} cases")

    if stage1_crashes:
        print("\n[!] Top Stage 1 Crash Causes:")
        for crash in stage1_crashes[:5]:
            print(f"  Case: {crash['case']} -> {crash['error']}")

    if stage2_crashes:
        print("\n[!] Top Stage 2 Crash Causes:")
        for crash in stage2_crashes[:10]:
            print(f"  Case: {crash['case']} -> {crash['error']}")

    # Save detailed crash audit to JSON
    report_data = {
        "total_tested": total_tested,
        "stage1_success_rate": stage1_successes / total_tested,
        "stage2_success_rate": stage2_successes / total_tested,
        "stage2_certified_rate": stage2_certified / max(1, stage2_successes),
        "filter_failures": filter_failures,
        "stage1_crashes": stage1_crashes,
        "stage2_crashes": stage2_crashes,
        "xai_crashes": xai_crashes,
        "report_crashes": report_crashes,
    }
    with open(PROJECT_DIR / "stress_test_audit_results.json", "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"\nDetailed audit results saved to: stress_test_audit_results.json")

if __name__ == "__main__":
    run_stress_test(num_real=200, num_synthetic=1000)
