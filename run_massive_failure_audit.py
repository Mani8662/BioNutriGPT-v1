"""
Massive Empirical Failure-Mode Audit for BioNutriGPT Stage 1 & Stage 2
Tests thousands of possibilities:
- Track A: 5,000 Stage 1 Comprehensive Stress Tests across all continuous and categorical spectra
- Track B: 100 Multi-Morbid & Edge-Case Archetypes through full Stage 2 Counterfactual Optimization
- Track C: End-to-End Pipeline Dossier and Clinical Report Generation
"""

import sys
import time
import json
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
SAMPLE_PATIENT = PROJECT_DIR / "sample_patient_single_line.txt"

def run_audit():
    print("=" * 85)
    print("BIONUTRIGPT STAGE 1 & 2 MASSIVE FAILURE-MODE AUDIT")
    print("=" * 85)

    base_vector = pipeline.parse_patient_vector(SAMPLE_PATIENT)
    artifacts = s2._load_preprocessing_artifacts(PREPROCESSOR_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = s2._load_stage1_model(CHECKPOINT_PATH, artifacts, device)

    # -----------------------------------------------------------------------
    # TRACK A: 5,000 STAGE 1 STRESS TEST CASES
    # -----------------------------------------------------------------------
    print("\n[TRACK A] Testing 5,000 Stage 1 Diagnostic Scenarios...")
    t0 = time.time()
    s1_passes = 0
    s1_failures = []
    
    np.random.seed(1337)
    
    # 5,000 diverse synthetic cases
    for i in range(5000):
        test_vec = base_vector.copy()
        
        # Perturb across massive ranges
        test_vec[0] = np.random.uniform(4.0, 16.0)     # HbA1c
        test_vec[1] = np.random.uniform(15.0, 120.0)   # HDL
        test_vec[2] = np.random.uniform(100.0, 500.0)  # Total Chol
        test_vec[3] = test_vec[2] - test_vec[1]        # Non-HDL
        test_vec[4] = test_vec[2] / (test_vec[1] + 1e-4) # Chol/HDL
        test_vec[5] = test_vec[3] / (test_vec[1] + 1e-4) # Non-HDL/HDL
        
        # Nutrition
        test_vec[6] = np.random.uniform(500.0, 8000.0) # Calories
        test_vec[7] = np.random.uniform(10.0, 350.0)   # Protein
        test_vec[8] = np.random.uniform(20.0, 900.0)   # Carbs
        test_vec[9] = np.random.uniform(0.0, 300.0)    # Sugar
        test_vec[10] = np.random.uniform(0.0, 80.0)    # Fiber
        test_vec[11] = np.random.uniform(5.0, 300.0)   # Total Fat
        test_vec[12] = np.random.uniform(1.0, 100.0)   # Sat Fat
        test_vec[13] = np.random.uniform(1.0, 120.0)   # Mono Fat
        test_vec[14] = np.random.uniform(1.0, 80.0)    # Poly Fat
        test_vec[40] = np.random.uniform(200.0, 12000.0) # Sodium
        test_vec[41] = np.random.uniform(300.0, 8000.0)  # Potassium
        test_vec[44] = np.random.uniform(0.0, 150.0)   # Alcohol
        
        # Ratios
        test_vec[45] = (test_vec[7] * 4.0) / test_vec[6] * 100.0
        test_vec[46] = (test_vec[8] * 4.0) / test_vec[6] * 100.0
        test_vec[47] = (test_vec[11] * 9.0) / test_vec[6] * 100.0
        test_vec[48] = test_vec[10] / (test_vec[6] / 1000.0)
        test_vec[49] = test_vec[9] / (test_vec[10] + 1.0)
        test_vec[50] = test_vec[40] / (test_vec[41] + 1.0)
        test_vec[51] = test_vec[12] / (test_vec[13] + test_vec[14] + 0.1)
        
        # Lifestyle & Demo
        test_vec[52] = np.random.uniform(2.0, 12.0)    # Sleep
        test_vec[53] = test_vec[52] - 7.5
        test_vec[54] = np.random.uniform(60.0, 1000.0) # Sitting
        test_vec[60] = np.random.uniform(0.0, 15000.0) # MET
        test_vec[71] = np.random.uniform(18.0, 85.0)   # Age
        test_vec[74] = 1.0 if np.random.rand() > 0.5 else 0.0 # Sex
        
        try:
            diag, _ = pipeline.run_stage1_diagnostics(test_vec)
            # Verify outputs
            probs = diag["probabilities"]
            for d in ("diabetes", "obesity", "cvd"):
                p = probs[d]
                if not (0.0 <= p <= 1.0) or not np.isfinite(p):
                    raise ValueError(f"Out of range probability for {d}: {p}")
            s1_passes += 1
        except Exception as e:
            s1_failures.append({"case": i, "error": str(e), "tb": traceback.format_exc()})
            
    print(f"Track A Result: {s1_passes}/5000 Stage 1 passes ({s1_passes/50.0:.2f}%) in {time.time()-t0:.2f}s!")
    if s1_failures:
        print(f"  Stage 1 Failures encountered: {len(s1_failures)}")
        for f in s1_failures[:3]:
            print(f"    Case {f['case']}: {f['error']}")

    # -----------------------------------------------------------------------
    # TRACK B: 100 MULTI-MORBID STAGE 2 COUNTERFACTUAL SCENARIOS
    # -----------------------------------------------------------------------
    print("\n[TRACK B] Testing 100 Diverse Clinical Boundary Archetypes through Stage 2...")
    t1 = time.time()
    s2_passes = 0
    s2_certified = 0
    s2_crashes = []
    filter_failures = {"filter_1": 0, "filter_2": 0, "filter_3": 0, "filter_4": 0}

    # 10 Archetypes x 10 variations = 100 rigorous stress tests
    archetypes = [
        {"desc": "Normal Baseline Healthy", "age": 35.0, "weight": 68.0, "height": 172.0, "act": "moderate", "female": False, "hba1c": 5.1, "non_hdl": 110.0, "kcal": 2100.0},
        {"desc": "Elderly Multi-Morbid Diabetic", "age": 78.0, "weight": 85.0, "height": 165.0, "act": "sedentary", "female": True, "hba1c": 10.5, "non_hdl": 210.0, "kcal": 2400.0},
        {"desc": "Severe Morbid Obesity Class III", "age": 42.0, "weight": 185.0, "height": 170.0, "act": "sedentary", "female": False, "hba1c": 7.8, "non_hdl": 195.0, "kcal": 3800.0},
        {"desc": "Low Weight Sarcopenic Anorexia", "age": 24.0, "weight": 38.0, "height": 162.0, "act": "sedentary", "female": True, "hba1c": 4.9, "non_hdl": 130.0, "kcal": 1100.0},
        {"desc": "Athletic Hyper-Caloric Bulking", "age": 26.0, "weight": 110.0, "height": 192.0, "act": "active", "female": False, "hba1c": 5.4, "non_hdl": 140.0, "kcal": 4600.0},
        {"desc": "High Sodium Renal / Hypertension", "age": 62.0, "weight": 88.0, "height": 175.0, "act": "light", "female": False, "hba1c": 6.8, "non_hdl": 220.0, "kcal": 2600.0},
        {"desc": "High Alcohol & Circadian Disruption", "age": 45.0, "weight": 82.0, "height": 178.0, "act": "moderate", "female": False, "hba1c": 5.9, "non_hdl": 180.0, "kcal": 2800.0},
        {"desc": "High Sugar Low Fiber Ketogenic Deficit", "age": 39.0, "weight": 76.0, "height": 168.0, "act": "light", "female": True, "hba1c": 6.2, "non_hdl": 160.0, "kcal": 1900.0},
        {"desc": "Sedentary Desk Worker Metabolic Syndrome", "age": 50.0, "weight": 98.0, "height": 180.0, "act": "sedentary", "female": False, "hba1c": 7.1, "non_hdl": 175.0, "kcal": 2900.0},
        {"desc": "Young Lean Athlete Vegetarian", "age": 21.0, "weight": 62.0, "height": 175.0, "act": "active", "female": True, "hba1c": 4.8, "non_hdl": 95.0, "kcal": 2300.0},
    ]

    case_count = 0
    for arch in archetypes:
        for var in range(10):
            case_count += 1
            syn_vec = base_vector.copy()
            
            # Apply archetype parameters with random jitter
            syn_vec[0] = arch["hba1c"] + np.random.uniform(-0.3, 0.3)
            syn_vec[3] = arch["non_hdl"] + np.random.uniform(-10.0, 10.0)
            syn_vec[6] = arch["kcal"] + np.random.uniform(-100.0, 100.0)
            syn_vec[71] = arch["age"]
            syn_vec[74] = 1.0 if arch["female"] else 0.0
            
            if "Sodium" in arch["desc"]:
                syn_vec[40] = 6500.0
                syn_vec[41] = 1200.0
            if "Alcohol" in arch["desc"]:
                syn_vec[44] = 80.0
                syn_vec[52] = 4.0
            if "Sugar" in arch["desc"]:
                syn_vec[9] = 160.0
                syn_vec[10] = 3.0
            if "Sedentary" in arch["desc"]:
                syn_vec[54] = 800.0
                syn_vec[60] = 100.0
                
            # Recompute ratios
            syn_vec[49] = syn_vec[9] / (syn_vec[10] + 1.0)
            syn_vec[50] = syn_vec[40] / (syn_vec[41] + 1.0)
            syn_vec[51] = syn_vec[12] / (syn_vec[13] + syn_vec[14] + 0.1)

            meta = {
                "age_years": arch["age"],
                "is_female": arch["female"],
                "weight_kg": arch["weight"] + np.random.uniform(-2.0, 2.0),
                "height_cm": arch["height"],
                "activity_level": arch["act"],
            }

            try:
                # Stage 1
                s1_diag, _ = pipeline.run_stage1_diagnostics(syn_vec)
                
                # Stage 2
                s2_dossier = pipeline.run_stage2_counterfactual(syn_vec, meta)
                s2_passes += 1
                
                if s2_dossier.get("certified_for_stage3", False):
                    s2_certified += 1
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
                        
                # XAI and Clinical Report
                xai = pipeline.compute_clinical_etiology_and_risks(syn_vec, meta, s1_diag, s2_dossier)
                rep = pipeline.render_clinical_report(syn_vec, meta, s1_diag, s2_dossier, s2_dossier.get("certified_for_stage3", False))
                dossier = pipeline.build_unified_dossier(syn_vec, meta, s1_diag, s2_dossier)
                
            except Exception as e:
                s2_crashes.append({"case": f"{arch['desc']}_var{var}", "error": str(e), "tb": traceback.format_exc()})

    print(f"Track B Result: {s2_passes}/100 Stage 2 runs completed successfully in {time.time()-t1:.2f}s!")
    print(f"  Stage 2 Zero-Crash Reliability : {s2_passes}/100 ({s2_passes}% crash-free)")
    print(f"  Stage 3 Clinical Certification : {s2_certified}/100 ({s2_certified}% certified)")
    print(f"  Non-Certified Filter Drop Breakdown:")
    for f_k, f_v in filter_failures.items():
        print(f"    - {f_k}: {f_v} cases")
        
    if s2_crashes:
        print(f"  [!] Stage 2 Crashes Encountered: {len(s2_crashes)}")
        for c in s2_crashes[:5]:
            print(f"    Case: {c['case']} -> {c['error']}")

    # Save summary report
    summary = {
        "track_a_stage1_cases": 5000,
        "track_a_passes": s1_passes,
        "track_b_stage2_cases": 100,
        "track_b_passes": s2_passes,
        "track_b_certified": s2_certified,
        "track_b_filter_failures": filter_failures,
        "s1_failures": s1_failures,
        "s2_crashes": s2_crashes,
    }
    with open(PROJECT_DIR / "massive_failure_audit_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\nMassive audit complete! Summary saved to massive_failure_audit_summary.json.")

if __name__ == "__main__":
    run_audit()
