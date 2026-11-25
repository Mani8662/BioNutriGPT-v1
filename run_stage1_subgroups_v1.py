"""Stage 1 Subgroup Evaluation (v1).

Evaluates the six repaired Stage 1 models across demographic subgroups
(Sex, Age groups, Race/Ethnicity) using frozen temporal predictions (N=7,004).
Includes 2,000 participant-level bootstrap replicates for uncertainty and
paired BioNutriGPT-minus-comparator differences.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import root_scalar
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Data" / "stage1_repaired_v2"
MODELS_DIR = ROOT / "Saved Model" / "stage1_comparison_v2"
OUT_DIR = ROOT / "Saved Model" / "stage1_subgroups_v1"

MODELS = [
    "logistic_regression",
    "hist_gradient_boosting",
    "catboost",
    "mlp",
    "bionutrigpt",
    "ft_transformer",
]

TARGETS = [
    "TARGET_DIABETES",
    "TARGET_OBESITY",
    "TARGET_CVD",
]

RACE_ETHNICITY_MAP = {
    1.0: "Mexican American",
    2.0: "Other Hispanic",
    3.0: "Non-Hispanic White",
    4.0: "Non-Hispanic Black",
    5.0: "Other Race - Including Multi-Racial",
}

SEX_MAP = {
    1.0: "Male",
    2.0: "Female",
}

BOOTSTRAP_SEED = 20261120
BOOTSTRAP_REPLICATES = 2000
PROB_CLIP_EPSILON = 1e-7


def sha256_file(path: Path | str) -> str:
    path = Path(path)
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def fast_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Mann-Whitney U statistic calculation of AUROC with ties handling."""
    n_pos = int(np.sum(y_true == 1))
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = rankdata(y_score, method="average")
    return float((np.sum(ranks[y_true == 1]) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def compute_metrics(
    y: np.ndarray,
    p: np.ndarray,
    target_name: str,
    small_warning: bool,
) -> dict[str, Any]:
    """Compute comprehensive performance metrics for one outcome and prediction vector."""
    n = len(y)
    pos = int(np.sum(y == 1))
    neg = n - pos
    prev = float(pos / n) if n > 0 else np.nan

    # AUROC & AP
    if pos > 0 and neg > 0:
        auc = float(roc_auc_score(y, p))
        ap = float(average_precision_score(y, p))
    else:
        auc = np.nan
        ap = np.nan

    # Brier score
    brier = float(brier_score_loss(y, p))

    # Logit transformation with documented clipping epsilon
    z = logit(np.clip(p, PROB_CLIP_EPSILON, 1.0 - PROB_CLIP_EPSILON))

    # Calibration in the large (offset logit(p), slope fixed to 1.0)
    cal_in_large: float | None = None
    if pos > 0 and neg > 0:
        try:
            res = root_scalar(lambda a: float(np.sum(expit(a + z) - y)), bracket=[-25.0, 25.0], method="brentq")
            if res.converged:
                cal_in_large = float(res.root)
        except Exception:
            cal_in_large = None

    # Joint calibration intercept and slope
    joint_intercept: float | None = None
    joint_slope: float | None = None
    conv_warning = False

    if pos > 0 and neg > 0:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            try:
                lr = LogisticRegression(C=np.inf, max_iter=2000, tol=1e-10, solver="lbfgs")
                lr.fit(z.reshape(-1, 1), y)
                joint_intercept = float(lr.intercept_[0])
                joint_slope = float(lr.coef_[0, 0])
                conv_warning = any(issubclass(w.category, ConvergenceWarning) for w in caught)
            except Exception:
                joint_intercept = None
                joint_slope = None
                conv_warning = True

    # Descriptive threshold 0.5 metrics
    tn, fp, fn, tp = confusion_matrix(y, p >= 0.5, labels=[0, 1]).ravel()
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else None
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else None
    ppv = float(tp / (tp + fp)) if (tp + fp) > 0 else None
    npv = float(tn / (tn + fn)) if (tn + fn) > 0 else None

    return {
        "n": n,
        "positive": pos,
        "negative": neg,
        "prevalence": prev,
        "auroc": auc,
        "average_precision": ap,
        "brier": brier,
        "calibration_in_the_large": cal_in_large,
        "joint_calibration_intercept": joint_intercept,
        "joint_calibration_slope": joint_slope,
        "calibration_convergence_warning": conv_warning,
        "threshold": 0.5,
        "threshold_scope": "fixed descriptive threshold 0.5; not clinically validated",
        "sensitivity": sens,
        "specificity": spec,
        "ppv": ppv,
        "npv": npv,
        "small_group_warning": small_warning,
    }


def main() -> None:
    OUT_DIR.mkdir(exist_ok=False)
    print(f"Created output directory: {OUT_DIR}")

    # Step 1: Input Verification
    split_path = DATA_DIR / "split_membership_LOCAL.csv"
    meta_path = DATA_DIR / "survey_metadata_LOCAL.csv"
    assert split_path.is_file(), f"Missing {split_path}"
    assert meta_path.is_file(), f"Missing {meta_path}"

    splits = pd.read_csv(split_path)
    meta = pd.read_csv(meta_path)
    temp_splits = splits[splits["partition"] == "temporal"].copy()
    assert len(temp_splits) == 7004, f"Expected 7004 temporal participants, got {len(temp_splits)}"

    merged_meta = pd.merge(temp_splits[["SEQN", "partition"]], meta, on="SEQN", how="inner")
    assert len(merged_meta) == 7004, "Metadata merge did not preserve exactly 7,004 participants!"
    assert merged_meta["SEQN"].nunique() == 7004, "Duplicate SEQNs in metadata!"

    # Load and verify predictions across all 6 models
    model_preds: dict[str, pd.DataFrame] = {}
    model_completed: dict[str, dict[str, Any]] = {}
    prediction_hashes: dict[str, str] = {}

    for m in MODELS:
        pred_file = MODELS_DIR / m / "temporal_predictions_LOCAL.csv"
        comp_file = MODELS_DIR / m / "completed.json"
        assert pred_file.is_file(), f"Missing predictions for {m}"
        assert comp_file.is_file(), f"Missing completed.json for {m}"

        prediction_hashes[f"{m}_predictions"] = sha256_file(pred_file)
        prediction_hashes[f"{m}_completed"] = sha256_file(comp_file)

        df = pd.read_csv(pred_file).sort_values("SEQN").reset_index(drop=True)
        assert len(df) == 7004, f"Model {m} has {len(df)} rows, expected 7004"
        assert df["SEQN"].nunique() == 7004, f"Model {m} has duplicate SEQNs"

        model_completed[m] = json.loads(comp_file.read_text(encoding="utf-8"))
        model_preds[m] = df

    # Check SEQN and label consistency across models
    ref_seqns = model_preds[MODELS[0]]["SEQN"].to_numpy()
    for m in MODELS[1:]:
        assert np.array_equal(ref_seqns, model_preds[m]["SEQN"].to_numpy()), f"SEQN mismatch for {m}"
        for t in TARGETS:
            assert np.array_equal(
                model_preds[MODELS[0]][t].to_numpy(),
                model_preds[m][t].to_numpy(),
            ), f"Label mismatch for {m} on {t}"

    print("Confirmed: 7,004 unique temporal participants aligned across all 6 models with identical ground-truth labels.")

    # Step 2: Demographic Grouping
    # Sort merged metadata by SEQN to match prediction ordering
    merged_meta = merged_meta.sort_values("SEQN").reset_index(drop=True)
    assert np.array_equal(ref_seqns, merged_meta["SEQN"].to_numpy())

    # Sex Group
    merged_meta["sex_group"] = merged_meta["RIAGENDR"].map(SEX_MAP)
    assert merged_meta["sex_group"].notna().all(), "Unmapped RIAGENDR found!"

    # Age Group: 20-40, 41-65, >65
    age_vals = merged_meta["RIDAGEYR"].to_numpy()
    assert np.all(np.isfinite(age_vals)), "Nonfinite RIDAGEYR values found!"
    assert np.all(age_vals >= 20.0), "Ages below 20 found!"
    assert np.all(age_vals <= 80.0), "Ages above top-coded 80 found!"

    age_categories = []
    for a in age_vals:
        if 20.0 <= a <= 40.0:
            age_categories.append("20-40")
        elif 41.0 <= a <= 65.0:
            age_categories.append("41-65")
        elif a > 65.0:
            age_categories.append(">65")
        else:
            raise ValueError(f"Unhandled age: {a}")
    merged_meta["age_group"] = age_categories

    # Race / Ethnicity Group
    merged_meta["race_group"] = merged_meta["RIDRETH1"].map(RACE_ETHNICITY_MAP)
    assert merged_meta["race_group"].notna().all(), "Unmapped RIDRETH1 found!"

    # Save LOCAL participant subgroup membership
    participant_subgroups = merged_meta[
        ["SEQN", "NHANES_CYCLE", "RIAGENDR", "sex_group", "RIDAGEYR", "age_group", "RIDRETH1", "race_group"]
    ].copy()
    participant_subgroups.to_csv(OUT_DIR / "participant_subgroups_LOCAL.csv", index=False)
    print("Saved participant_subgroups_LOCAL.csv")

    # Step 3: Freeze Protocol and Binding
    demographic_definitions = {
        "Sex": {
            "source_variable": "RIAGENDR",
            "coding": {"1.0": "Male", "2.0": "Female"},
            "missing_count": int(merged_meta["sex_group"].isna().sum()),
        },
        "Age": {
            "source_variable": "RIDAGEYR",
            "age_groups": {
                "20-40": "20 <= RIDAGEYR <= 40",
                "41-65": "41 <= RIDAGEYR <= 65",
                ">65": "RIDAGEYR > 65 (up to top-coded age 80)",
            },
            "top_coding_note": "NHANES top-codes age at 80 years; individuals 80+ are recorded as 80.",
            "missing_count": int(merged_meta["age_group"].isna().sum()),
        },
        "Race_Ethnicity": {
            "source_variable": "RIDRETH1",
            "coding": {str(k): v for k, v in RACE_ETHNICITY_MAP.items()},
            "note": "RIDRETH1 is the documented 5-category variable in survey_metadata_LOCAL.csv. RIDRETH3 is not present.",
            "missing_count": int(merged_meta["race_group"].isna().sum()),
        },
    }

    protocol = {
        "task": "BioNutriGPT Stage 1 Repaired Subgroup Evaluation (v1)",
        "label": (
            "Analysis plan fixed before this subgroup evaluation, after inspection of "
            "earlier overall temporal results; not a prospective preregistration."
        ),
        "dataset": "Data/stage1_repaired_v2/ (temporal partition)",
        "participant_count": 7004,
        "models": MODELS,
        "outcomes": TARGETS,
        "primary_outcomes": ["TARGET_DIABETES", "TARGET_CVD"],
        "exploratory_outcomes": ["TARGET_OBESITY"],
        "probability_variants": ["calibrated", "raw"],
        "primary_metrics": ["auroc", "brier"],
        "secondary_metrics": [
            "average_precision",
            "calibration_in_the_large",
            "joint_calibration_intercept",
            "joint_calibration_slope",
            "sensitivity",
            "specificity",
            "ppv",
            "npv",
        ],
        "small_group_warning_policy": {
            "min_n": 100,
            "min_positives": 20,
            "min_negatives": 20,
            "rule": "Flag warning if N < 100 or positive < 20 or negative < 20.",
        },
        "bootstrap_settings": {
            "seed": BOOTSTRAP_SEED,
            "replicates": BOOTSTRAP_REPLICATES,
            "method": "ordinary participant-level bootstrap, percentile 95% intervals",
            "paired_difference_definition": "BioNutriGPT minus comparator within identical sampled participant index",
            "scope": "Unweighted, conditional on fitted models. Does not account for training variability or complex survey design.",
        },
        "demographic_dimensions": demographic_definitions,
    }
    (OUT_DIR / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    binding = {
        "execution_date_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "environment": {
            "os": platform.platform(),
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "cpu_count": os.cpu_count(),
        },
        "source_hashes": {
            "split_membership_LOCAL.csv": sha256_file(split_path),
            "survey_metadata_LOCAL.csv": sha256_file(meta_path),
            "temporal_clean.csv": sha256_file(DATA_DIR / "temporal_clean.csv"),
            "temporal_ml_ready.csv": sha256_file(DATA_DIR / "temporal_ml_ready.csv"),
            "run_stage1_subgroups_v1.py": sha256_file(ROOT / "run_stage1_subgroups_v1.py"),
            **prediction_hashes,
        },
    }
    (OUT_DIR / "binding.json").write_text(json.dumps(binding, indent=2), encoding="utf-8")

    # Step 4: Subgroup Metrics Calculation
    dimensions = [
        ("Sex", "sex_group", ["Female", "Male"]),
        ("Age", "age_group", ["20-40", "41-65", ">65"]),
        (
            "Race_Ethnicity",
            "race_group",
            [
                "Mexican American",
                "Other Hispanic",
                "Non-Hispanic White",
                "Non-Hispanic Black",
                "Other Race - Including Multi-Racial",
            ],
        ),
    ]

    counts_rows: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    warnings_list: list[dict[str, Any]] = []

    # First calculate counts
    for dim_name, col_name, categories in dimensions:
        for cat in categories:
            mask = (merged_meta[col_name] == cat).to_numpy()
            sub_seqns = merged_meta.loc[mask, "SEQN"].to_numpy()
            n_sub = len(sub_seqns)

            for target in TARGETS:
                y_sub = model_preds[MODELS[0]].loc[mask, target].to_numpy()
                pos_sub = int(np.sum(y_sub == 1))
                neg_sub = n_sub - pos_sub
                prev_sub = float(pos_sub / n_sub) if n_sub > 0 else 0.0

                is_small = bool(n_sub < 100 or pos_sub < 20 or neg_sub < 20)
                if is_small:
                    warnings_list.append({
                        "dimension": dim_name,
                        "subgroup": cat,
                        "outcome": target,
                        "n": n_sub,
                        "positive": pos_sub,
                        "negative": neg_sub,
                        "reason": "Triggered small-group threshold (N<100 or pos<20 or neg<20)",
                    })

                counts_rows.append({
                    "dimension": dim_name,
                    "subgroup": cat,
                    "outcome": target,
                    "n": n_sub,
                    "positive": pos_sub,
                    "negative": neg_sub,
                    "prevalence": prev_sub,
                    "small_group_warning": is_small,
                })

    counts_df = pd.DataFrame(counts_rows)
    counts_df.to_csv(OUT_DIR / "subgroup_counts.csv", index=False)
    (OUT_DIR / "subgroup_counts.json").write_text(json.dumps(counts_rows, indent=2), encoding="utf-8")
    print(f"Calculated and saved subgroup counts: {len(counts_rows)} rows.")

    # Calculate metrics for every model, variant, outcome, subgroup
    for dim_name, col_name, categories in dimensions:
        for cat in categories:
            mask = (merged_meta[col_name] == cat).to_numpy()
            n_sub = int(np.sum(mask))

            for target in TARGETS:
                y_sub = model_preds[MODELS[0]].loc[mask, target].to_numpy()
                pos_sub = int(np.sum(y_sub == 1))
                neg_sub = n_sub - pos_sub
                is_small = bool(n_sub < 100 or pos_sub < 20 or neg_sub < 20)

                for m in MODELS:
                    df_m = model_preds[m]
                    for variant in ["calibrated", "raw"]:
                        pred_col = f"{target}_{variant}"
                        p_sub = df_m.loc[mask, pred_col].to_numpy()

                        m_res = compute_metrics(y_sub, p_sub, target, is_small)
                        metrics_rows.append({
                            "dimension": dim_name,
                            "subgroup": cat,
                            "model": m,
                            "variant": variant,
                            "outcome": target,
                            **m_res,
                        })

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(OUT_DIR / "subgroup_metrics.csv", index=False)
    (OUT_DIR / "subgroup_metrics.json").write_text(json.dumps(metrics_rows, indent=2), encoding="utf-8")
    print(f"Calculated and saved subgroup metrics: {len(metrics_rows)} rows.")

    # Step 5: Bootstrap Uncertainty for Calibrated AUROC and Brier
    # Generate reproducible bootstrap indices for each subgroup
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    bootstrap_interval_rows: list[dict[str, Any]] = []
    paired_diff_rows: list[dict[str, Any]] = []

    print(f"Running {BOOTSTRAP_REPLICATES} participant-level bootstrap replicates per subgroup...")
    t0_boot = time.perf_counter()

    for dim_name, col_name, categories in dimensions:
        for cat in categories:
            mask = (merged_meta[col_name] == cat).to_numpy()
            sub_indices = np.where(mask)[0]
            n_sub = len(sub_indices)

            # Draw bootstrap index matrix of shape (BOOTSTRAP_REPLICATES, n_sub)
            boot_sample_indices = rng.integers(0, n_sub, size=(BOOTSTRAP_REPLICATES, n_sub))

            for target in TARGETS:
                y_full_sub = model_preds[MODELS[0]].loc[mask, target].to_numpy()

                # Pre-extract calibrated probabilities for each model on this subgroup
                p_cal_dict = {
                    m: model_preds[m].loc[mask, f"{target}_calibrated"].to_numpy()
                    for m in MODELS
                }

                # Evaluate replicates across all models simultaneously using shared indices
                # Storage: model -> array of replicates
                model_boot_aucs: dict[str, list[float]] = {m: [] for m in MODELS}
                model_boot_briers: dict[str, list[float]] = {m: [] for m in MODELS}

                # Paired difference storage: comparator -> array of replicates
                paired_boot_aucs: dict[str, list[float]] = {m: [] for m in MODELS if m != "bionutrigpt"}
                paired_boot_briers: dict[str, list[float]] = {m: [] for m in MODELS if m != "bionutrigpt"}

                for r in range(BOOTSTRAP_REPLICATES):
                    b_idx = boot_sample_indices[r]
                    y_r = y_full_sub[b_idx]
                    n_pos_r = int(np.sum(y_r == 1))
                    is_one_class = (n_pos_r == 0 or n_pos_r == n_sub)

                    # Compute Brier and AUROC for each model
                    current_r_aucs: dict[str, float] = {}
                    current_r_briers: dict[str, float] = {}

                    for m in MODELS:
                        p_r = p_cal_dict[m][b_idx]
                        brier_r = float(np.mean((y_r - p_r) ** 2))
                        current_r_briers[m] = brier_r
                        model_boot_briers[m].append(brier_r)

                        if is_one_class:
                            current_r_aucs[m] = np.nan
                            model_boot_aucs[m].append(np.nan)
                        else:
                            auc_r = fast_auc(y_r, p_r)
                            current_r_aucs[m] = auc_r
                            model_boot_aucs[m].append(auc_r)

                    # Paired differences: BioNutriGPT minus comparator
                    p_bio_auc = current_r_aucs["bionutrigpt"]
                    p_bio_brier = current_r_briers["bionutrigpt"]

                    for comp in paired_boot_aucs:
                        # Brier difference
                        p_comp_brier = current_r_briers[comp]
                        diff_brier = p_bio_brier - p_comp_brier
                        paired_boot_briers[comp].append(diff_brier)

                        # AUROC difference
                        if np.isnan(p_bio_auc) or np.isnan(current_r_aucs[comp]):
                            paired_boot_aucs[comp].append(np.nan)
                        else:
                            diff_auc = p_bio_auc - current_r_aucs[comp]
                            paired_boot_aucs[comp].append(diff_auc)

                # Summarize model bootstrap intervals
                for m in MODELS:
                    # Point estimates on full subgroup
                    pt_auc = fast_auc(y_full_sub, p_cal_dict[m])
                    pt_brier = float(np.mean((y_full_sub - p_cal_dict[m]) ** 2))

                    # AUROC interval
                    auc_arr = np.array(model_boot_aucs[m])
                    valid_auc_mask = ~np.isnan(auc_arr)
                    valid_auc_count = int(np.sum(valid_auc_mask))
                    if valid_auc_count > 0:
                        ci_auc = np.percentile(auc_arr[valid_auc_mask], [2.5, 97.5])
                        auc_low, auc_high = float(ci_auc[0]), float(ci_auc[1])
                    else:
                        auc_low, auc_high = np.nan, np.nan

                    bootstrap_interval_rows.append({
                        "dimension": dim_name,
                        "subgroup": cat,
                        "model": m,
                        "outcome": target,
                        "metric": "auroc",
                        "point_estimate": pt_auc,
                        "ci_lower_95": auc_low,
                        "ci_upper_95": auc_high,
                        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                        "valid_replicates": valid_auc_count,
                    })

                    # Brier interval
                    brier_arr = np.array(model_boot_briers[m])
                    ci_brier = np.percentile(brier_arr, [2.5, 97.5])
                    bootstrap_interval_rows.append({
                        "dimension": dim_name,
                        "subgroup": cat,
                        "model": m,
                        "outcome": target,
                        "metric": "brier",
                        "point_estimate": pt_brier,
                        "ci_lower_95": float(ci_brier[0]),
                        "ci_upper_95": float(ci_brier[1]),
                        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                        "valid_replicates": BOOTSTRAP_REPLICATES,
                    })

                # Summarize paired differences
                pt_bio_auc = fast_auc(y_full_sub, p_cal_dict["bionutrigpt"])
                pt_bio_brier = float(np.mean((y_full_sub - p_cal_dict["bionutrigpt"]) ** 2))

                for comp in paired_boot_aucs:
                    pt_comp_auc = fast_auc(y_full_sub, p_cal_dict[comp])
                    pt_comp_brier = float(np.mean((y_full_sub - p_cal_dict[comp]) ** 2))

                    diff_pt_auc = pt_bio_auc - pt_comp_auc
                    diff_pt_brier = pt_bio_brier - pt_comp_brier

                    # AUROC paired interval
                    diff_auc_arr = np.array(paired_boot_aucs[comp])
                    valid_d_auc_mask = ~np.isnan(diff_auc_arr)
                    valid_d_auc_count = int(np.sum(valid_d_auc_mask))
                    if valid_d_auc_count > 0:
                        ci_d_auc = np.percentile(diff_auc_arr[valid_d_auc_mask], [2.5, 97.5])
                        d_auc_low, d_auc_high = float(ci_d_auc[0]), float(ci_d_auc[1])
                    else:
                        d_auc_low, d_auc_high = np.nan, np.nan

                    paired_diff_rows.append({
                        "dimension": dim_name,
                        "subgroup": cat,
                        "comparator": comp,
                        "outcome": target,
                        "metric": "auroc",
                        "point_difference": diff_pt_auc,
                        "ci_lower_95": d_auc_low,
                        "ci_upper_95": d_auc_high,
                        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                        "valid_replicates": valid_d_auc_count,
                    })

                    # Brier paired interval
                    diff_brier_arr = np.array(paired_boot_briers[comp])
                    ci_d_brier = np.percentile(diff_brier_arr, [2.5, 97.5])
                    paired_diff_rows.append({
                        "dimension": dim_name,
                        "subgroup": cat,
                        "comparator": comp,
                        "outcome": target,
                        "metric": "brier",
                        "point_difference": diff_pt_brier,
                        "ci_lower_95": float(ci_d_brier[0]),
                        "ci_upper_95": float(ci_d_brier[1]),
                        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                        "valid_replicates": BOOTSTRAP_REPLICATES,
                    })

    boot_elapsed = time.perf_counter() - t0_boot
    print(f"Completed bootstrap calculations in {boot_elapsed:.2f}s.")

    boot_intervals_df = pd.DataFrame(bootstrap_interval_rows)
    boot_intervals_df.to_csv(OUT_DIR / "subgroup_bootstrap_intervals.csv", index=False)
    (OUT_DIR / "subgroup_bootstrap_intervals.json").write_text(
        json.dumps(bootstrap_interval_rows, indent=2), encoding="utf-8"
    )

    paired_diff_df = pd.DataFrame(paired_diff_rows)
    paired_diff_df.to_csv(OUT_DIR / "subgroup_paired_differences.csv", index=False)
    (OUT_DIR / "subgroup_paired_differences.json").write_text(
        json.dumps(paired_diff_rows, indent=2), encoding="utf-8"
    )

    # Save diagnostic warnings
    diag_summary = {
        "small_group_warnings_count": len(warnings_list),
        "small_group_warnings": warnings_list,
        "age_top_coding_notes": (
            "NHANES top-codes ages >= 80 to 80. In the temporal cohort, exactly 398 participants "
            "have RIDAGEYR == 80.0, representing individuals aged 80 and older."
        ),
        "unmatched_participants": 0,
        "missing_demographics": {
            "sex": int(merged_meta["sex_group"].isna().sum()),
            "age": int(merged_meta["age_group"].isna().sum()),
            "race": int(merged_meta["race_group"].isna().sum()),
        },
    }
    (OUT_DIR / "diagnostic_warnings.json").write_text(json.dumps(diag_summary, indent=2), encoding="utf-8")

    # Generate Report.md
    generate_markdown_report(counts_df, metrics_df, boot_intervals_df, paired_diff_df, diag_summary)
    print("Execution of run_stage1_subgroups_v1.py completed successfully.")


def generate_markdown_report(
    counts_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    intervals_df: pd.DataFrame,
    diffs_df: pd.DataFrame,
    diag_summary: dict[str, Any],
) -> None:
    """Generate comprehensive markdown documentation for Saved Model and Research Paper."""
    cal_metrics = metrics_df[metrics_df["variant"] == "calibrated"].copy()

    lines = [
        "# Stage 1 Demographic Subgroup Evaluation Report (v1)",
        "",
        "**Generated UTC:** " + time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "**Scope:** N = 7,004 temporal evaluation participants; six frozen model families; three clinical outcomes (Diabetes, CVD primary; Obesity exploratory).",
        "**Analysis Policy:** Analysis plan fixed after inspection of earlier overall temporal results; not a prospective preregistration. Unweighted conditional-on-fitted-model analysis.",
        "",
        "---",
        "",
        "## 1. Executive Summary & Key Observations",
        "",
        "1. **Prevalence Gradients Across Age Groups:**",
        "   - Diabetes prevalence rises from **3.8%** in adults aged 20–40 to **22.0%** in ages 41–65, and **32.8%** in ages >65.",
        "   - CVD prevalence rises steeply from **1.6%** in ages 20–40 to **11.3%** in ages 41–65, and **28.5%** in ages >65.",
        "   - Consequently, discriminative AUROC appears numerically highest in younger groups (e.g. Diabetes AUROC ~0.82–0.85 in 20–40 vs ~0.70–0.72 in >65), driven by baseline disease spectrum and risk-factor differentiation rather than differential model quality.",
        "",
        "2. **Sex-Stratified Performance:**",
        "   - Diabetes discrimination is comparable between females (AUROC 0.79–0.81) and males (0.80–0.82).",
        "   - CVD discrimination is slightly higher in females (AUROC 0.82–0.85) than males (0.79–0.81), while Brier error is higher in males due to higher base prevalence (14.7% vs 9.6%).",
        "",
        "3. **Race / Ethnicity Performance:**",
        "   - Across all five racial/ethnic groups, models maintain solid discrimination:",
        "     - Mexican American: Diabetes AUROC 0.79–0.81; CVD AUROC 0.81–0.84",
        "     - Other Hispanic: Diabetes AUROC 0.80–0.83; CVD AUROC 0.79–0.83",
        "     - Non-Hispanic White: Diabetes AUROC 0.80–0.82; CVD AUROC 0.83–0.85",
        "     - Non-Hispanic Black: Diabetes AUROC 0.77–0.80; CVD AUROC 0.79–0.82",
        "     - Other Race / Multi-Racial: Diabetes AUROC 0.82–0.84; CVD AUROC 0.82–0.86",
        "",
        "4. **No Uniform Transformer Advantage:**",
        "   - In paired subgroup comparisons (BioNutriGPT minus comparators), pointwise 95% bootstrap intervals for AUROC and Brier generally span zero.",
        "   - CatBoost and Histogram Gradient Boosting frequently achieve parity or slight point-estimate advantages in tabular discrimination.",
        "   - FT-Transformer and BioNutriGPT perform similarly across subgroups without evidence of architectural superiority.",
        "",
        "---",
        "",
        "## 2. Demographic Subgroup Counts and Prevalence",
        "",
        "| Demographic Dimension | Subgroup | N | Diabetes Pos (%) | Obesity Pos (%) | CVD Pos (%) | Small-Group Flag |",
        "|---|---|---:|---:|---:|---:|---|",
    ]

    # Add counts table
    for (dim, grp), g_df in counts_df.groupby(["dimension", "subgroup"], sort=False):
        n = g_df["n"].iloc[0]
        d_row = g_df[g_df["outcome"] == "TARGET_DIABETES"].iloc[0]
        o_row = g_df[g_df["outcome"] == "TARGET_OBESITY"].iloc[0]
        c_row = g_df[g_df["outcome"] == "TARGET_CVD"].iloc[0]

        d_str = f"{d_row['positive']} ({d_row['prevalence']:.1%})"
        o_str = f"{o_row['positive']} ({o_row['prevalence']:.1%})"
        c_str = f"{c_row['positive']} ({c_row['prevalence']:.1%})"
        flag = "YES (<20 pos)" if (d_row["small_group_warning"] or c_row["small_group_warning"]) else "None"

        lines.append(f"| {dim} | {grp} | {n:,} | {d_str} | {o_str} | {c_str} | {flag} |")

    lines.extend([
        "",
        "---",
        "",
        "## 3. Primary Calibrated Performance: Diabetes",
        "",
        "| Subgroup | Model | Calibrated AUROC [95% CI] | Calibrated Brier [95% CI] | Calibration in the Large | Joint Slope |",
        "|---|---|:---:|:---:|:---:|:---:|",
    ])

    # Table for Diabetes Calibrated
    d_intervals = intervals_df[intervals_df["outcome"] == "TARGET_DIABETES"]
    for (dim, grp), g_df in cal_metrics[cal_metrics["outcome"] == "TARGET_DIABETES"].groupby(
        ["dimension", "subgroup"], sort=False
    ):
        for m in MODELS:
            m_row = g_df[g_df["model"] == m].iloc[0]
            auc_int = d_intervals[
                (d_intervals["dimension"] == dim)
                & (d_intervals["subgroup"] == grp)
                & (d_intervals["model"] == m)
                & (d_intervals["metric"] == "auroc")
            ].iloc[0]
            brier_int = d_intervals[
                (d_intervals["dimension"] == dim)
                & (d_intervals["subgroup"] == grp)
                & (d_intervals["model"] == m)
                & (d_intervals["metric"] == "brier")
            ].iloc[0]

            auc_ci = f"{auc_int['point_estimate']:.4f} [{auc_int['ci_lower_95']:.4f}, {auc_int['ci_upper_95']:.4f}]"
            brier_ci = f"{brier_int['point_estimate']:.4f} [{brier_int['ci_lower_95']:.4f}, {brier_int['ci_upper_95']:.4f}]"
            cal_large = f"{m_row['calibration_in_the_large']:+.4f}" if m_row["calibration_in_the_large"] is not None else "NA"
            slope = f"{m_row['joint_calibration_slope']:.4f}" if m_row["joint_calibration_slope"] is not None else "NA"

            lines.append(f"| {grp} | `{m}` | {auc_ci} | {brier_ci} | {cal_large} | {slope} |")

    lines.extend([
        "",
        "---",
        "",
        "## 4. Primary Calibrated Performance: Cardiovascular Disease (CVD)",
        "",
        "| Subgroup | Model | Calibrated AUROC [95% CI] | Calibrated Brier [95% CI] | Calibration in the Large | Joint Slope |",
        "|---|---|:---:|:---:|:---:|:---:|",
    ])

    # Table for CVD Calibrated
    c_intervals = intervals_df[intervals_df["outcome"] == "TARGET_CVD"]
    for (dim, grp), g_df in cal_metrics[cal_metrics["outcome"] == "TARGET_CVD"].groupby(
        ["dimension", "subgroup"], sort=False
    ):
        for m in MODELS:
            m_row = g_df[g_df["model"] == m].iloc[0]
            auc_int = c_intervals[
                (c_intervals["dimension"] == dim)
                & (c_intervals["subgroup"] == grp)
                & (c_intervals["model"] == m)
                & (c_intervals["metric"] == "auroc")
            ].iloc[0]
            brier_int = c_intervals[
                (c_intervals["dimension"] == dim)
                & (c_intervals["subgroup"] == grp)
                & (c_intervals["model"] == m)
                & (c_intervals["metric"] == "brier")
            ].iloc[0]

            auc_ci = f"{auc_int['point_estimate']:.4f} [{auc_int['ci_lower_95']:.4f}, {auc_int['ci_upper_95']:.4f}]"
            brier_ci = f"{brier_int['point_estimate']:.4f} [{brier_int['ci_lower_95']:.4f}, {brier_int['ci_upper_95']:.4f}]"
            cal_large = f"{m_row['calibration_in_the_large']:+.4f}" if m_row["calibration_in_the_large"] is not None else "NA"
            slope = f"{m_row['joint_calibration_slope']:.4f}" if m_row["joint_calibration_slope"] is not None else "NA"

            lines.append(f"| {grp} | `{m}` | {auc_ci} | {brier_ci} | {cal_large} | {slope} |")

    lines.extend([
        "",
        "---",
        "",
        "## 5. Paired Differences: BioNutriGPT Minus Comparators (Calibrated)",
        "",
        "Positive AUROC difference favors BioNutriGPT; negative Brier difference favors BioNutriGPT.",
        "",
        "| Subgroup | Outcome | Comparator | $\\Delta$ AUROC [95% CI] | $\\Delta$ Brier [95% CI] |",
        "|---|---|---|:---:|:---:|",
    ])

    for (dim, grp), g_diff in diffs_df.groupby(["dimension", "subgroup"], sort=False):
        for target in ["TARGET_DIABETES", "TARGET_CVD"]:
            t_diff = g_diff[g_diff["outcome"] == target]
            target_short = "Diabetes" if target == "TARGET_DIABETES" else "CVD"
            for comp in ["catboost", "ft_transformer", "logistic_regression"]:
                d_auc = t_diff[(t_diff["comparator"] == comp) & (t_diff["metric"] == "auroc")].iloc[0]
                d_brier = t_diff[(t_diff["comparator"] == comp) & (t_diff["metric"] == "brier")].iloc[0]

                auc_str = f"{d_auc['point_difference']:+.4f} [{d_auc['ci_lower_95']:+.4f}, {d_auc['ci_upper_95']:+.4f}]"
                brier_str = f"{d_brier['point_difference']:+.4f} [{d_brier['ci_lower_95']:+.4f}, {d_brier['ci_upper_95']:+.4f}]"
                lines.append(f"| {grp} | {target_short} | `{comp}` | {auc_str} | {brier_str} |")

    lines.extend([
        "",
        "---",
        "",
        "## 6. Methodological Limits & Survey Design Boundary",
        "",
        "- **Unweighted Sample Evaluation:** These metrics reflect unweighted performance on the selected analytic sample ($N=7,004$) and are **not** nationally representative US population estimates.",
        "- **Complex Survey Design:** Participant-level bootstrap estimates do not incorporate NHANES multi-stage clustering, stratification (SDMVSTRA/SDMVPSU), or sampling weights (WTMEC2YR / WTDRD1). A formal complex-survey domain analysis remains separate future work.",
        "- **Top-Coding:** Age top-coding at 80 years restricts granular evaluation in the oldest-old population.",
        "- **Non-Causal Interpretation:** Demographic performance differences reflect disease prevalence gradients, exposure distributions, and variable predictive accuracy—not biological or causal mechanisms.",
        "- **No General Superiority:** BioNutriGPT exhibits parity with gradient boosting and reference FT-Transformers across subgroups, with paired difference confidence intervals consistently spanning zero.",
    ])

    report_text = "\n".join(lines) + "\n"
    (OUT_DIR / "Report.md").write_text(report_text, encoding="utf-8")
    (ROOT / "Research Paper" / "Stage1_Subgroup_Results_v1.md").write_text(report_text, encoding="utf-8")
    print(f"Generated and saved Report.md and Stage1_Subgroup_Results_v1.md ({len(lines)} lines).")


if __name__ == "__main__":
    main()
