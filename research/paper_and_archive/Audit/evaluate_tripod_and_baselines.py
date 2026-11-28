"""
BioNutriGPT - TRIPOD+AI & NHANES Survey-Weighted Benchmarking Suite
Evaluates BioNutriGPT Stage 1 against standard clinical baselines:
  1. L2-Regularized Logistic Regression
  2. Random Forest Classifier
  3. HistGradientBoostingClassifier (Gradient Boosted Decision Trees)
Incorporates:
  - CDC Survey Weights (WTMECPRP) for population-representative metrics
  - 1,000-sample Bootstrap 95% Confidence Intervals
  - Logistic Calibration Slope & Intercept + Expected Calibration Error (ECE)
  - Decision Curve Analysis (DCA) across decision thresholds [0.05, 0.50]
  - Subgroup stratification by Biological Sex and Age Brackets
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
import torch

script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir))
import train_bionutrigpt_stage1 as s1

def calculate_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)
    
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = bin_ids == b
        if np.sum(mask) > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            ece += (np.sum(mask) / n) * np.abs(bin_acc - bin_conf)
    return float(ece)

def calculate_calibration_slope_intercept(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    eps = 1e-6
    y_prob_clipped = np.clip(y_prob, eps, 1.0 - eps)
    logit_prob = np.log(y_prob_clipped / (1.0 - y_prob_clipped)).reshape(-1, 1)
    
    lr = LogisticRegression(penalty=None, solver='lbfgs')
    try:
        lr.fit(logit_prob, y_true)
        slope = float(lr.coef_[0][0])
        intercept = float(lr.intercept_[0])
    except Exception:
        slope, intercept = 1.0, 0.0
    return slope, intercept

def calculate_dca(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray) -> Dict[str, List[float]]:
    n = len(y_true)
    prevalence = np.mean(y_true)
    net_benefits_model = []
    net_benefits_all = []
    net_benefits_none = []
    
    for pt in thresholds:
        # Treat None
        net_benefits_none.append(0.0)
        
        # Treat All
        nb_all = prevalence - (1.0 - prevalence) * (pt / (1.0 - pt))
        net_benefits_all.append(float(nb_all))
        
        # Model
        tp = np.sum((y_prob >= pt) & (y_true == 1))
        fp = np.sum((y_prob >= pt) & (y_true == 0))
        nb_model = (tp / n) - (fp / n) * (pt / (1.0 - pt))
        net_benefits_model.append(float(nb_model))
        
    return {
        "thresholds": [round(float(t), 3) for t in thresholds],
        "net_benefit_model": [round(v, 4) for v in net_benefits_model],
        "net_benefit_all": [round(v, 4) for v in net_benefits_all],
        "net_benefit_none": [round(v, 4) for v in net_benefits_none]
    }

def bootstrap_metric_ci(y_true: np.ndarray, y_prob: np.ndarray, weights: np.ndarray = None, n_bootstraps: int = 1000, seed: int = 42) -> Dict[str, Dict[str, float]]:
    rng = np.random.RandomState(seed)
    n = len(y_true)
    
    aurocs, praucs, briers = [], [], []
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        y_t_b = y_true[idx]
        y_p_b = y_prob[idx]
        w_b = weights[idx] if weights is not None else None
        
        if len(np.unique(y_t_b)) < 2:
            continue
            
        try:
            aurocs.append(roc_auc_score(y_t_b, y_p_b, sample_weight=w_b))
        except Exception:
            pass
        try:
            praucs.append(average_precision_score(y_t_b, y_p_b, sample_weight=w_b))
        except Exception:
            pass
        try:
            briers.append(brier_score_loss(y_t_b, y_p_b, sample_weight=w_b))
        except Exception:
            pass
            
    def ci_dict(arr):
        if len(arr) == 0:
            return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
        return {
            "mean": round(float(np.mean(arr)), 4),
            "ci_lower": round(float(np.percentile(arr, 2.5)), 4),
            "ci_upper": round(float(np.percentile(arr, 97.5)), 4)
        }
        
    return {
        "auroc": ci_dict(aurocs),
        "prauc": ci_dict(praucs),
        "brier": ci_dict(briers)
    }

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_dir, "Data", "BioNutriGPT_train_2007_2016_ml_ready.csv")
    test_path = os.path.join(base_dir, "Data", "BioNutriGPT_test_2017_2020_ml_ready.csv")
    raw_path = os.path.join(base_dir, "Data", "BioNutriGPT_merged_raw_2005_2020.csv")
    ckpt_path = os.path.join(base_dir, "Saved Model", "bionutrigpt_stage1_best.pt")
    
    print("Loading datasets...")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df_raw = pd.read_csv(raw_path, low_memory=False)
    
    # Merge test set with survey weights and demographics
    cols_to_merge = ["SEQN", "WTMECPRP", "SDMVPSU", "SDMVSTRA", "RIAGENDR", "RIDAGEYR"]
    df_raw_sub = df_raw[[c for c in cols_to_merge if c in df_raw.columns]].drop_duplicates(subset=["SEQN"])
    df_test_merged = df_test.merge(df_raw_sub, on="SEQN", how="left")
    
    # Ensure survey weights are valid
    weights = df_test_merged["WTMECPRP"].values
    if np.isnan(weights).any():
        print("Imputing missing survey weights with mean...")
        weights = np.nan_to_num(weights, nan=np.nanmean(weights))
        
    X_train = df_train[s1.FEATURE_COLS].values
    X_test = df_test[s1.FEATURE_COLS].values
    
    print(f"Features: {len(s1.FEATURE_COLS)} | Train: {X_train.shape} | Test: {X_test.shape}")
    
    # 1. Load BioNutriGPT Stage 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating BioNutriGPT Stage 1 on {device}...")
    model = s1.BioNutriGPTStage1().to(device)
    s1.load_state_dict_checkpoint(model, Path(ckpt_path), device)
    model.eval()
    
    with torch.no_grad():
        x_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        logits_dict, _, _ = model(x_tensor)
        bionutri_probs = np.zeros((len(X_test), len(s1.TARGET_COLS)))
        for t_idx, target_name in enumerate(s1.TARGET_COLS):
            cond_key = target_name.replace("TARGET_", "").lower()
            bionutri_probs[:, t_idx] = torch.sigmoid(logits_dict[cond_key]).cpu().numpy()
        
    # Split train cohort into training (85%) and tuning/validation (15%) for Platt calibration
    df_tr_split, df_val_split = train_test_split(df_train, test_size=0.15, random_state=42)
    print(f"Validation split for Platt probability calibration: N={len(df_val_split)}")
    
    with torch.no_grad():
        x_val_tensor = torch.tensor(df_val_split[s1.FEATURE_COLS].values, dtype=torch.float32).to(device)
        val_logits_dict, _, _ = model(x_val_tensor)
        val_probs = np.zeros((len(df_val_split), len(s1.TARGET_COLS)))
        for t_idx, target_name in enumerate(s1.TARGET_COLS):
            cond_key = target_name.replace("TARGET_", "").lower()
            val_probs[:, t_idx] = torch.sigmoid(val_logits_dict[cond_key]).cpu().numpy()
            
    bionutri_calibrated_probs = np.zeros_like(bionutri_probs)
    for t_idx, target_name in enumerate(s1.TARGET_COLS):
        y_val = df_val_split[target_name].values
        scaler = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        scaler.fit(val_probs[:, t_idx].reshape(-1, 1), y_val)
        bionutri_calibrated_probs[:, t_idx] = scaler.predict_proba(bionutri_probs[:, t_idx].reshape(-1, 1))[:, 1]
        
    predictions = {
        "BioNutriGPT_Stage1": bionutri_probs,
        "BioNutriGPT_Stage1_Calibrated": bionutri_calibrated_probs
    }
    
    # 2. Train and Evaluate Classical Baselines
    print("\nTraining and evaluating classical ML baselines...")
    models_to_benchmark = {
        "Logistic_Regression": lambda: LogisticRegression(C=0.1, max_iter=1000, class_weight="balanced", random_state=42),
        "Random_Forest": lambda: RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_split=10, random_state=42, n_jobs=-1),
        "Gradient_Boosted_Trees": lambda: HistGradientBoostingClassifier(max_iter=150, max_depth=8, learning_rate=0.08, random_state=42)
    }
    
    for name, ctor in models_to_benchmark.items():
        print(f"  Fitting {name} across all 3 conditions...")
        preds = np.zeros_like(bionutri_probs)
        for t_idx, target in enumerate(s1.TARGET_COLS):
            y_tr = df_train[target].values
            clf = ctor()
            clf.fit(X_train, y_tr)
            preds[:, t_idx] = clf.predict_proba(X_test)[:, 1]
        predictions[name] = preds
        
    # 3. Comprehensive Evaluation
    report = {
        "metadata": {
            "train_cohort": "NHANES 2007-2016 (N=23,694)",
            "test_cohort": "NHANES 2017-March 2020 Pre-Pandemic Temporal Holdout (N=7,046)",
            "feature_count": len(s1.FEATURE_COLS),
            "hba1c_included_in_features": False,
            "survey_weight_variable": "WTMECPRP (MEC 2-Year / Pre-Pandemic Exam Weight)"
        },
        "target_summaries": {},
        "model_comparisons": {},
        "decision_curve_analysis": {},
        "subgroup_analyses": {}
    }
    
    dca_thresholds = np.linspace(0.05, 0.50, 19)
    
    for t_idx, target in enumerate(s1.TARGET_COLS):
        y_true = df_test[target].values
        unweighted_prev = float(np.mean(y_true))
        weighted_prev = float(np.average(y_true, weights=weights))
        
        report["target_summaries"][target] = {
            "unweighted_prevalence": round(unweighted_prev, 4),
            "survey_weighted_prevalence": round(weighted_prev, 4),
            "positive_cases": int(np.sum(y_true)),
            "total_samples": len(y_true)
        }
        
        report["model_comparisons"][target] = {}
        
        for m_name, probs_matrix in predictions.items():
            y_prob = probs_matrix[:, t_idx]
            
            # Unweighted
            raw_auroc = roc_auc_score(y_true, y_prob)
            raw_prauc = average_precision_score(y_true, y_prob)
            raw_brier = brier_score_loss(y_true, y_prob)
            
            # Weighted
            wt_auroc = roc_auc_score(y_true, y_prob, sample_weight=weights)
            wt_prauc = average_precision_score(y_true, y_prob, sample_weight=weights)
            wt_brier = brier_score_loss(y_true, y_prob, sample_weight=weights)
            
            # Bootstrap CIs (1,000 runs)
            boot_ci_raw = bootstrap_metric_ci(y_true, y_prob, weights=None, n_bootstraps=1000)
            boot_ci_wt = bootstrap_metric_ci(y_true, y_prob, weights=weights, n_bootstraps=1000)
            
            # Calibration
            slope, intercept = calculate_calibration_slope_intercept(y_true, y_prob)
            ece = calculate_ece(y_true, y_prob, n_bins=10)
            
            report["model_comparisons"][target][m_name] = {
                "unweighted": {
                    "auroc": round(float(raw_auroc), 4),
                    "auroc_95ci": [boot_ci_raw["auroc"]["ci_lower"], boot_ci_raw["auroc"]["ci_upper"]],
                    "prauc": round(float(raw_prauc), 4),
                    "prauc_95ci": [boot_ci_raw["prauc"]["ci_lower"], boot_ci_raw["prauc"]["ci_upper"]],
                    "brier": round(float(raw_brier), 4),
                    "brier_95ci": [boot_ci_raw["brier"]["ci_lower"], boot_ci_raw["brier"]["ci_upper"]]
                },
                "survey_weighted": {
                    "auroc": round(float(wt_auroc), 4),
                    "auroc_95ci": [boot_ci_wt["auroc"]["ci_lower"], boot_ci_wt["auroc"]["ci_upper"]],
                    "prauc": round(float(wt_prauc), 4),
                    "prauc_95ci": [boot_ci_wt["prauc"]["ci_lower"], boot_ci_wt["prauc"]["ci_upper"]],
                    "brier": round(float(wt_brier), 4),
                    "brier_95ci": [boot_ci_wt["brier"]["ci_lower"], boot_ci_wt["brier"]["ci_upper"]]
                },
                "calibration": {
                    "slope": round(float(slope), 4),
                    "intercept": round(float(intercept), 4),
                    "ece_10bin": round(float(ece), 4)
                }
            }
            
        # DCA for BioNutriGPT Stage 1
        dca_res = calculate_dca(y_true, predictions["BioNutriGPT_Stage1"][:, t_idx], dca_thresholds)
        report["decision_curve_analysis"][target] = dca_res
        
    # 4. Demographic Subgroup Analyses (BioNutriGPT Stage 1)
    print("\nEvaluating demographic subgroup discrimination...")
    subgroups = {}
    
    # Biological Sex (1 = Male, 2 = Female in NHANES RIAGENDR)
    # Check if RIAGENDR exists in df_test_merged
    if "RIAGENDR" in df_test_merged.columns:
        subgroups["sex_male"] = df_test_merged["RIAGENDR"] == 1
        subgroups["sex_female"] = df_test_merged["RIAGENDR"] == 2
        
    # Age brackets (use unscaled raw chronological age from NHANES raw merge)
    age_col = "RIDAGEYR_y" if "RIDAGEYR_y" in df_test_merged.columns else "RIDAGEYR"
    ages = df_test_merged[age_col].values
    subgroups["age_18_44"] = (ages >= 18) & (ages < 45)
    subgroups["age_45_64"] = (ages >= 45) & (ages < 65)
    subgroups["age_65_plus"] = ages >= 65
    
    subgroup_report = {}
    bionutri_probs = predictions["BioNutriGPT_Stage1"]
    
    for grp_name, mask in subgroups.items():
        grp_n = int(np.sum(mask))
        subgroup_report[grp_name] = {"n_samples": grp_n, "targets": {}}
        for t_idx, target in enumerate(s1.TARGET_COLS):
            y_sub = df_test[target].values[mask]
            p_sub = bionutri_probs[mask, t_idx]
            w_sub = weights[mask]
            
            if len(np.unique(y_sub)) >= 2:
                subgroup_report[grp_name]["targets"][target] = {
                    "cases": int(np.sum(y_sub)),
                    "unweighted_auroc": round(float(roc_auc_score(y_sub, p_sub)), 4),
                    "survey_weighted_auroc": round(float(roc_auc_score(y_sub, p_sub, sample_weight=w_sub)), 4),
                    "brier": round(float(brier_score_loss(y_sub, p_sub)), 4)
                }
            else:
                subgroup_report[grp_name]["targets"][target] = {"cases": int(np.sum(y_sub)), "status": "insufficient_events"}
                
    report["subgroup_analyses"] = subgroup_report
    
    # Save Report
    out_path = os.path.join(base_dir, "Saved Model", "tripod_baseline_evaluation_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nTRIPOD+AI benchmark report saved to: {out_path}")
    
    # Mirror to Audit
    audit_out_path = os.path.join(base_dir, "Audit", "Saved Model", "tripod_baseline_evaluation_report.json")
    os.makedirs(os.path.dirname(audit_out_path), exist_ok=True)
    with open(audit_out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Mirrored to Audit: {audit_out_path}")
    
    # Print Executive Summary Table
    print("\n" + "="*95)
    print("TRIPOD+AI BENCHMARK SUMMARY (TEMPORAL HOLDOUT NHANES 2017-2020, N=7,046)")
    print("="*95)
    print(f"{'Condition':<12} | {'Model':<30} | {'AUROC (Raw)':<12} | {'AUROC (Survey Wt)':<18} | {'Brier':<8} | {'Slope':<6} | {'ECE':<6}")
    print("-" * 95)
    for target in s1.TARGET_COLS:
        for m_name in predictions.keys():
            m_res = report["model_comparisons"][target][m_name]
            raw_auc = f"{m_res['unweighted']['auroc']:.4f}"
            wt_auc = f"{m_res['survey_weighted']['auroc']:.4f} [{m_res['survey_weighted']['auroc_95ci'][0]:.3f}-{m_res['survey_weighted']['auroc_95ci'][1]:.3f}]"
            brier = f"{m_res['survey_weighted']['brier']:.4f}"
            slope = f"{m_res['calibration']['slope']:.2f}"
            ece = f"{m_res['calibration']['ece_10bin']:.4f}"
            print(f"{target:<12} | {m_name:<30} | {raw_auc:<12} | {wt_auc:<18} | {brier:<8} | {slope:<6} | {ece:<6}")
        print("-" * 95)

if __name__ == "__main__":
    main()
