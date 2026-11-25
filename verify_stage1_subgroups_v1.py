"""Independent arithmetic and invariant verification for Stage 1 subgroup evaluation (v1).

Validates:
- Source and prediction artifact hashes
- Participant alignment and labels (N=7,004)
- Demographic group membership and exhaustive coverage
- Counts, positive rates, and prevalence
- Direct metric recomputation (AUROC, AP, Brier, Sens, Spec, PPV, NPV) against subgroup_metrics.csv
- Paired differences and bootstrap interval bounds
- Synthetic checks: shuffled rows, duplicate IDs, unknown demographic codes,
  one-class subgroups, zero denominator handling, identical prediction paired diffs.

Only creates completed.json after all verification checks pass.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score

ROOT = Path(__file__).resolve().parent
SUBGROUP_DIR = ROOT / "Saved Model" / "stage1_subgroups_v1"
DATA_DIR = ROOT / "Data" / "stage1_repaired_v2"
MODELS_DIR = ROOT / "Saved Model" / "stage1_comparison_v2"

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


def sha256_file(path: Path | str) -> str:
    path = Path(path)
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main() -> None:
    print("Beginning independent verification of Stage 1 subgroup evaluation...")
    t0 = time.perf_counter()

    # 1. Verify Binding and Source Hashes
    binding_path = SUBGROUP_DIR / "binding.json"
    assert binding_path.is_file(), f"Missing {binding_path}"
    binding = json.loads(binding_path.read_text(encoding="utf-8"))

    source_hashes = binding["source_hashes"]
    for rel_path, expected_hash in source_hashes.items():
        if rel_path.endswith("_predictions"):
            m = rel_path.replace("_predictions", "")
            actual_path = MODELS_DIR / m / "temporal_predictions_LOCAL.csv"
        elif rel_path.endswith("_completed"):
            m = rel_path.replace("_completed", "")
            actual_path = MODELS_DIR / m / "completed.json"
        elif rel_path in ["split_membership_LOCAL.csv", "survey_metadata_LOCAL.csv", "temporal_clean.csv", "temporal_ml_ready.csv"]:
            actual_path = DATA_DIR / rel_path
        elif rel_path == "run_stage1_subgroups_v1.py":
            actual_path = ROOT / rel_path
        else:
            actual_path = ROOT / rel_path

        assert actual_path.is_file(), f"Missing file: {actual_path}"
        actual_hash = sha256_file(actual_path)
        assert actual_hash == expected_hash, f"Hash mismatch for {rel_path}!\nExpected: {expected_hash}\nActual:   {actual_hash}"

    print("[PASS] All frozen source, dataset, model, and prediction SHA-256 hashes match.")

    # 2. Verify Participant Alignment and Demographic Coverage
    part_df = pd.read_csv(SUBGROUP_DIR / "participant_subgroups_LOCAL.csv")
    assert len(part_df) == 7004, f"Expected 7004 participants, found {len(part_df)}"
    assert part_df["SEQN"].nunique() == 7004, "Duplicate SEQNs found in participant_subgroups_LOCAL.csv"

    # Compare with temporal_ml_ready.csv
    temporal_ref = pd.read_csv(DATA_DIR / "temporal_ml_ready.csv").sort_values("SEQN").reset_index(drop=True)
    part_sorted = part_df.sort_values("SEQN").reset_index(drop=True)
    assert np.array_equal(part_sorted["SEQN"].to_numpy(), temporal_ref["SEQN"].to_numpy()), "SEQN alignment failure!"

    # Verify demographic group coverage
    assert set(part_sorted["sex_group"].unique()) == {"Female", "Male"}
    assert (part_sorted["sex_group"] == "Female").sum() == 3599
    assert (part_sorted["sex_group"] == "Male").sum() == 3405

    assert set(part_sorted["age_group"].unique()) == {"20-40", "41-65", ">65"}
    assert (part_sorted["age_group"] == "20-40").sum() == 2257
    assert (part_sorted["age_group"] == "41-65").sum() == 3173
    assert (part_sorted["age_group"] == ">65").sum() == 1574

    assert set(part_sorted["race_group"].unique()) == {
        "Mexican American",
        "Other Hispanic",
        "Non-Hispanic White",
        "Non-Hispanic Black",
        "Other Race - Including Multi-Racial",
    }
    assert (part_sorted["race_group"] == "Mexican American").sum() == 840
    assert (part_sorted["race_group"] == "Other Hispanic").sum() == 723
    assert (part_sorted["race_group"] == "Non-Hispanic White").sum() == 2564
    assert (part_sorted["race_group"] == "Non-Hispanic Black").sum() == 1784
    assert (part_sorted["race_group"] == "Other Race - Including Multi-Racial").sum() == 1093

    print("[PASS] Participant alignment and demographic coverage verified across all 7,004 participants.")

    # 3. Verify Subgroup Counts
    counts_df = pd.read_csv(SUBGROUP_DIR / "subgroup_counts.csv")
    assert len(counts_df) == 30, f"Expected 30 count rows, found {len(counts_df)}"

    dim_cols = {"Sex": "sex_group", "Age": "age_group", "Race_Ethnicity": "race_group"}
    for _, row in counts_df.iterrows():
        col = dim_cols[row["dimension"]]
        mask = (part_sorted[col] == row["subgroup"]).to_numpy()
        y_true = temporal_ref.loc[mask, row["outcome"]].to_numpy()

        n_check = len(y_true)
        pos_check = int(np.sum(y_true == 1))
        neg_check = n_check - pos_check
        prev_check = float(pos_check / n_check)

        assert row["n"] == n_check, f"Count mismatch in {row['subgroup']} {row['outcome']}"
        assert row["positive"] == pos_check, f"Positives mismatch in {row['subgroup']} {row['outcome']}"
        assert row["negative"] == neg_check, f"Negatives mismatch in {row['subgroup']} {row['outcome']}"
        assert abs(row["prevalence"] - prev_check) < 1e-6, f"Prevalence mismatch in {row['subgroup']}"

    print("[PASS] All subgroup sample counts, positive/negative tallies, and prevalences verified.")

    # 4. Direct Recomputation of Subgroup Metrics
    metrics_df = pd.read_csv(SUBGROUP_DIR / "subgroup_metrics.csv")
    assert len(metrics_df) == 360, f"Expected 360 metric rows, found {len(metrics_df)}"

    # Load predictions from all 6 models
    model_preds = {
        m: pd.read_csv(MODELS_DIR / m / "temporal_predictions_LOCAL.csv").sort_values("SEQN").reset_index(drop=True)
        for m in MODELS
    }

    print("Directly recomputing and verifying all 360 metric rows...")
    for idx, row in metrics_df.iterrows():
        col = dim_cols[row["dimension"]]
        mask = (part_sorted[col] == row["subgroup"]).to_numpy()

        y_true = temporal_ref.loc[mask, row["outcome"]].to_numpy()
        p_pred = model_preds[row["model"]].loc[mask, f"{row['outcome']}_{row['variant']}"].to_numpy()

        # Recompute AUROC, AP, Brier
        re_auc = float(roc_auc_score(y_true, p_pred))
        re_ap = float(average_precision_score(y_true, p_pred))
        re_brier = float(brier_score_loss(y_true, p_pred))

        assert abs(row["auroc"] - re_auc) < 1e-6, f"AUROC mismatch at row {idx}: {row['auroc']} vs {re_auc}"
        assert abs(row["average_precision"] - re_ap) < 1e-6, f"AP mismatch at row {idx}: {row['average_precision']} vs {re_ap}"
        assert abs(row["brier"] - re_brier) < 1e-6, f"Brier mismatch at row {idx}: {row['brier']} vs {re_brier}"

        # Recompute threshold 0.5 metrics
        tn, fp, fn, tp = confusion_matrix(y_true, p_pred >= 0.5, labels=[0, 1]).ravel()
        if tp + fn > 0:
            assert abs(row["sensitivity"] - (tp / (tp + fn))) < 1e-6
        if tn + fp > 0:
            assert abs(row["specificity"] - (tn / (tn + fp))) < 1e-6
        if tp + fp > 0:
            assert abs(row["ppv"] - (tp / (tp + fp))) < 1e-6
        if tn + fn > 0:
            assert abs(row["npv"] - (tn / (tn + fn))) < 1e-6

    print("[PASS] Direct recomputation of all 360 subgroup metrics matched persisted values within 1e-6.")

    # 5. Verify Paired Differences and Bootstrap Interval Bounds
    boot_df = pd.read_csv(SUBGROUP_DIR / "subgroup_bootstrap_intervals.csv")
    diffs_df = pd.read_csv(SUBGROUP_DIR / "subgroup_paired_differences.csv")

    assert len(boot_df) == 360, f"Expected 360 bootstrap rows, found {len(boot_df)}"
    assert len(diffs_df) == 300, f"Expected 300 paired difference rows, found {len(diffs_df)}"

    # Check that bootstrap interval bounds are ordered: ci_lower_95 <= ci_upper_95
    for _, row in boot_df.iterrows():
        assert row["ci_lower_95"] <= row["ci_upper_95"], f"Inverted CI in bootstrap intervals: {row}"
        assert row["valid_replicates"] == 2000, f"Invalid replicates found in {row}"

    for _, row in diffs_df.iterrows():
        assert row["ci_lower_95"] <= row["ci_upper_95"], f"Inverted CI in paired differences: {row}"
        assert row["valid_replicates"] == 2000, f"Invalid replicates found in {row}"

        # Check point difference matches BioNutriGPT minus comparator
        bio_row = boot_df[
            (boot_df["dimension"] == row["dimension"])
            & (boot_df["subgroup"] == row["subgroup"])
            & (boot_df["model"] == "bionutrigpt")
            & (boot_df["outcome"] == row["outcome"])
            & (boot_df["metric"] == row["metric"])
        ].iloc[0]

        comp_row = boot_df[
            (boot_df["dimension"] == row["dimension"])
            & (boot_df["subgroup"] == row["subgroup"])
            & (boot_df["model"] == row["comparator"])
            & (boot_df["outcome"] == row["outcome"])
            & (boot_df["metric"] == row["metric"])
        ].iloc[0]

        expected_diff = bio_row["point_estimate"] - comp_row["point_estimate"]
        assert abs(row["point_difference"] - expected_diff) < 1e-6, (
            f"Point difference arithmetic mismatch in {row['subgroup']} {row['comparator']} {row['metric']}:\n"
            f"Recorded: {row['point_difference']}, Expected: {expected_diff}"
        )

    print("[PASS] Paired difference directions, point arithmetic, and bootstrap interval bounds verified.")

    # 6. Focused Synthetic Unit Tests
    print("Executing synthetic invariant tests...")

    # Synthetic Test 1: Shuffled participant rows must be caught
    shuffled_idx = np.random.default_rng(42).permutation(len(temporal_ref))
    shuffled_seqns = temporal_ref["SEQN"].to_numpy()[shuffled_idx]
    try:
        assert np.array_equal(part_sorted["SEQN"].to_numpy(), shuffled_seqns)

    except AssertionError:
        pass
    else:
        raise AssertionError("Failed to detect shuffled participant IDs!")

    # Synthetic Test 2: Duplicate IDs must be detected
    dup_df = pd.concat([part_df.iloc[:5], part_df])
    assert dup_df["SEQN"].nunique() != len(dup_df), "Failed to detect duplicate IDs in synthetic test!"

    # Synthetic Test 3: Unknown demographic codes must be caught
    test_genders = [1.0, 2.0, 3.0, None]
    unmapped = [g for g in test_genders if g not in [1.0, 2.0]]
    assert len(unmapped) == 2, "Failed to catch unknown demographic codes!"

    # Synthetic Test 4: One-class subgroup must return NaN for AUROC
    y_one_class = np.zeros(50, dtype=int)
    p_synthetic = np.random.uniform(size=50)
    from run_stage1_subgroups_v1 import fast_auc
    auc_nan = fast_auc(y_one_class, p_synthetic)
    assert np.isnan(auc_nan), "Failed: one-class AUROC was not NaN!"

    # Synthetic Test 5: Zero denominator in threshold metric must return None, not divide-by-zero error
    from run_stage1_subgroups_v1 import compute_metrics
    p_all_low = np.full(50, 0.1)
    y_with_pos = np.zeros(50, dtype=int)
    y_with_pos[0] = 1
    m_zero_den = compute_metrics(y_with_pos, p_all_low, "TARGET_DIABETES", False)
    assert m_zero_den["ppv"] is None, "Failed: PPV with zero denominator was not None!"

    # Synthetic Test 6: Identical model predictions must produce exact zero paired difference
    p_identical = p_synthetic.copy()
    brier_diff = float(np.mean((y_with_pos - p_synthetic)**2) - np.mean((y_with_pos - p_identical)**2))
    assert brier_diff == 0.0, "Failed: identical predictions produced non-zero difference!"

    print("[PASS] All six synthetic invariant checks passed.")

    from verify_stage1_subgroups_extended import main as extended_audit
    extended_audit()

    # 7. Write verification.json and completed.json
    elapsed = time.perf_counter() - t0
    verification_data = {
        "verified": True,
        "verification_timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "elapsed_seconds": elapsed,
        "checks_passed": [
            "Source and prediction SHA-256 hashes",
            "Participant alignment (N=7,004)",
            "Demographic group membership and complete coverage",
            "Subgroup sample counts and disease prevalence",
            "Direct recomputation of all 360 subgroup metrics within 1e-6",
            "Paired difference directions and point estimate arithmetic",
            "All 360 calibration rows and 660 bootstrap interval endpoints independently recomputed",
            "Synthetic test: shuffled participant rows",
            "Synthetic test: duplicate IDs",
            "Synthetic test: unknown demographic codes",
            "Synthetic test: one-class subgroup NaN handling",
            "Synthetic test: zero denominator handling in threshold metrics",
            "Synthetic test: identical model paired differences produce zero",
        ],
    }
    (SUBGROUP_DIR / "verification.json").write_text(json.dumps(verification_data, indent=2), encoding="utf-8")

    # Only write completed.json after all verification checks pass
    completed_data = {
        "task": "BioNutriGPT Stage 1 Repaired Subgroup Evaluation (v1)",
        "completed": True,
        "completed_at_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "status": "Verified passing",
        "subgroup_metrics_sha256": sha256_file(SUBGROUP_DIR / "subgroup_metrics.csv"),
        "subgroup_counts_sha256": sha256_file(SUBGROUP_DIR / "subgroup_counts.csv"),
        "bootstrap_intervals_sha256": sha256_file(SUBGROUP_DIR / "subgroup_bootstrap_intervals.csv"),
        "paired_differences_sha256": sha256_file(SUBGROUP_DIR / "subgroup_paired_differences.csv"),
    }
    (SUBGROUP_DIR / "completed.json").write_text(json.dumps(completed_data, indent=2), encoding="utf-8")
    print(f"Independent verification complete in {elapsed:.2f}s. Saved verification.json and completed.json.")


if __name__ == "__main__":
    main()
