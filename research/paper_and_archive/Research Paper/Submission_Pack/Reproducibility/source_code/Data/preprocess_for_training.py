"""
BioNutriGPT ML Preprocessing Pipeline (Zero-Leakage Temporal Train/Test Split).

Preprocesses the two already-cleaned NHANES datasets into training-ready CSVs:
  - Input Train (2007-2016) : BioNutriGPT_train_2007_2016_clean.csv  (23,694 x 69)
  - Input Test  (2017-2020) : BioNutriGPT_test_2017_2020_clean.csv   ( 7,046 x 69)

  - Output Train ML-Ready   : BioNutriGPT_train_2007_2016_ml_ready.csv
  - Output Test  ML-Ready   : BioNutriGPT_test_2017_2020_ml_ready.csv
  - Fitted Artifacts        : preprocessing_artifacts.joblib
  - JSON Verification Log   : preprocessing_report.json

Key ML Safeguards (Separate Processing, Zero Leakage):
  1. Train and Test files are NEVER merged.
  2. Every statistical parameter (missing medians, categorical modes, log1p list,
     [0.1%, 99.9%] winsorization caps, and StandardScaler means/stds) is fitted
     STRICTLY on the Train file (2007-2016) ONLY.
  3. Those frozen parameters are then applied separately to transform Train and
     Test into the exact same feature space with 0 NaNs and 0 infinities.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


META_COLS = ["SEQN", "NHANES_CYCLE"]
TARGET_COLS = ["TARGET_DIABETES", "TARGET_OBESITY", "TARGET_CVD"]
# HbA1c (LBXGH) contributes to the ground-truth diabetes reference label definition.
# It is excluded from the predictor feature set to eliminate target leakage.
MODEL_EXCLUDED_RAW_FEATURES = {"LBXGH"}

BINARY_YES_NO_COLS = ["SLQ050", "PAQ605", "PAQ620", "PAQ635", "PAQ650", "PAQ665"]
SEX_COL = "RIAGENDR"
ORDINAL_COLS = ["DMDEDUC2"]

NOMINAL_CATEGORY_SPECS: dict[str, list[tuple[int, str]]] = {
    "RIDRETH1": [
        (1, "RACE_MEX_AMER"),
        (2, "RACE_OTHER_HISP"),
        (3, "RACE_NH_WHITE"),
        (4, "RACE_NH_BLACK"),
        (5, "RACE_OTHER_MULTI"),
    ],
    "MARITAL_STATUS": [
        (1, "MARITAL_MARRIED"),
        (2, "MARITAL_PREV_MARRIED"),
        (3, "MARITAL_NEVER_MARRIED"),
    ],
    "SMOKING_STATUS": [
        (0, "SMOKE_NEVER"),
        (1, "SMOKE_FORMER"),
        (2, "SMOKE_SOMEDAYS"),
        (3, "SMOKE_EVERYDAY"),
    ],
}

MAJOR_MISS_COLS = ["INDFMPIR", "SLEEP_HOURS", "PAD680", "PA_TOTAL_MET_MIN_WK"]
EXCLUDE_CONTINUOUS = set(
    META_COLS + TARGET_COLS + BINARY_YES_NO_COLS + [SEX_COL] + ORDINAL_COLS
    + list(NOMINAL_CATEGORY_SPECS) + list(MODEL_EXCLUDED_RAW_FEATURES)
)

ENGINEERED_COLS = [
    "CHOL_HDL_RATIO",
    "NON_HDL_TO_HDL_RATIO",
    "PROT_PCT_KCAL",
    "CARB_PCT_KCAL",
    "FAT_PCT_KCAL",
    "FIBER_PER_1000_KCAL",
    "SUGAR_TO_FIBER_RATIO",
    "SODIUM_POTASSIUM_RATIO",
    "SAT_TO_UNSAT_FAT_RATIO",
    "SLEEP_DEV_FROM_7_5",
]

LOWER_QUANTILE = 0.001
UPPER_QUANTILE = 0.999
SKEW_THRESHOLD = 1.0


@dataclass
class PreprocessorState:
    """All parameters learned strictly from the training split."""

    base_continuous_cols: list[str]
    all_continuous_cols: list[str]
    continuous_medians: dict[str, float]
    binary_modes: dict[str, float]
    nominal_modes: dict[str, float]
    log1p_cols: list[str]
    clip_bounds: dict[str, tuple[float, float]]
    scaler: StandardScaler
    feature_cols: list[str]


def compute_missingness_indicators(df: pd.DataFrame, feature_input_cols: list[str]) -> pd.DataFrame:
    """Create explicit 0/1 missingness indicators before any imputation."""
    flags: dict[str, np.ndarray] = {}
    covered = set(MAJOR_MISS_COLS)
    for col in MAJOR_MISS_COLS:
        flags[f"{col}_MISS"] = df[col].isna().astype(np.int8).to_numpy()
    other_cols = [c for c in feature_input_cols if c not in covered]
    other_miss = df[other_cols].isna().any(axis=1).astype(np.int8).to_numpy()
    flags["OTHER_FEATURE_MISS"] = other_miss
    return pd.DataFrame(flags, index=df.index)


def add_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute row-wise clinical lipid, nutritional, and sleep-deviation features."""
    out = pd.DataFrame(index=df.index)
    out["CHOL_HDL_RATIO"] = df["LBXTC"] / df["LBDHDD"]
    out["NON_HDL_TO_HDL_RATIO"] = df["NON_HDL_CHOL"] / df["LBDHDD"]
    out["PROT_PCT_KCAL"] = (df["DR1TPROT"] * 4.0) / df["DR1TKCAL"] * 100.0
    out["CARB_PCT_KCAL"] = (df["DR1TCARB"] * 4.0) / df["DR1TKCAL"] * 100.0
    out["FAT_PCT_KCAL"] = (df["DR1TTFAT"] * 9.0) / df["DR1TKCAL"] * 100.0
    out["FIBER_PER_1000_KCAL"] = df["DR1TFIBE"] / (df["DR1TKCAL"] / 1000.0)
    out["SUGAR_TO_FIBER_RATIO"] = df["DR1TSUGR"] / (df["DR1TFIBE"] + 1.0)
    out["SODIUM_POTASSIUM_RATIO"] = df["DR1TSODI"] / (df["DR1TPOTA"] + 1.0)
    out["SAT_TO_UNSAT_FAT_RATIO"] = df["DR1TSFAT"] / (df["DR1TMFAT"] + df["DR1TPFAT"] + 0.1)
    out["SLEEP_DEV_FROM_7_5"] = (df["SLEEP_HOURS"] - 7.5).abs()
    return out


def encode_binary_and_nominal(
    df: pd.DataFrame,
    binary_modes: dict[str, float],
    nominal_modes: dict[str, float],
) -> pd.DataFrame:
    """Impute binary & nominal categories with train modes and encode cleanly as 0/1."""
    encoded: dict[str, np.ndarray] = {}

    sex_vals = pd.to_numeric(df[SEX_COL], errors="coerce").fillna(binary_modes[SEX_COL])
    encoded["SEX_FEMALE"] = (sex_vals == 2.0).astype(np.int8).to_numpy()

    for col in BINARY_YES_NO_COLS:
        vals = pd.to_numeric(df[col], errors="coerce").fillna(binary_modes[col])
        encoded[col] = (vals == 1.0).astype(np.int8).to_numpy()

    for col, specs in NOMINAL_CATEGORY_SPECS.items():
        vals = pd.to_numeric(df[col], errors="coerce").fillna(nominal_modes[col]).round().astype(int)
        for code, new_name in specs:
            encoded[new_name] = (vals == code).astype(np.int8).to_numpy()

    return pd.DataFrame(encoded, index=df.index)


def fit_preprocessor(train_df: pd.DataFrame) -> PreprocessorState:
    """Learn every imputation, transformation, clipping, and scaling rule from Train only."""
    base_cont_cols = [c for c in train_df.columns if c not in EXCLUDE_CONTINUOUS] + ORDINAL_COLS

    # 1. Imputation medians/modes from Train
    continuous_medians = {c: float(pd.to_numeric(train_df[c], errors="coerce").median()) for c in base_cont_cols}
    binary_modes = {
        c: float(pd.to_numeric(train_df[c], errors="coerce").mode().iloc[0])
        for c in [SEX_COL] + BINARY_YES_NO_COLS
    }
    nominal_modes = {
        c: float(pd.to_numeric(train_df[c], errors="coerce").mode().iloc[0])
        for c in NOMINAL_CATEGORY_SPECS
    }

    # Apply Train imputation on a temporary copy to fit downstream transforms
    cont_df = pd.DataFrame(index=train_df.index)
    for c in base_cont_cols:
        cont_df[c] = pd.to_numeric(train_df[c], errors="coerce").fillna(continuous_medians[c]).astype(np.float64)

    # Row-wise engineered features (requires imputed SLEEP_HOURS + clean labs/nutrition)
    eng_df = add_clinical_features(cont_df)
    all_cont_df = pd.concat([cont_df, eng_df], axis=1)
    all_cont_cols = list(all_cont_df.columns)

    # 2. Identify non-negative right-skewed columns for log1p (strictly on Train)
    log1p_cols: list[str] = []
    for c in all_cont_cols:
        col_vals = all_cont_df[c]
        if col_vals.min() >= 0.0 and float(col_vals.skew()) > SKEW_THRESHOLD:
            log1p_cols.append(c)
            all_cont_df[c] = np.log1p(col_vals)

    # 3. Gentle [0.1%, 99.9%] tail winsorization bounds (strictly on Train)
    clip_bounds: dict[str, tuple[float, float]] = {}
    for c in all_cont_cols:
        lo = float(all_cont_df[c].quantile(LOWER_QUANTILE))
        hi = float(all_cont_df[c].quantile(UPPER_QUANTILE))
        clip_bounds[c] = (lo, hi)
        all_cont_df[c] = all_cont_df[c].clip(lower=lo, upper=hi)

    # 4. StandardScaler fitted strictly on Train
    scaler = StandardScaler()
    scaler.fit(all_cont_df[all_cont_cols].to_numpy(dtype=np.float64))

    feature_cols = (
        all_cont_cols
        + ["SEX_FEMALE"]
        + BINARY_YES_NO_COLS
        + [name for specs in NOMINAL_CATEGORY_SPECS.values() for _, name in specs]
        + [f"{c}_MISS" for c in MAJOR_MISS_COLS]
        + ["OTHER_FEATURE_MISS"]
    )

    return PreprocessorState(
        base_continuous_cols=base_cont_cols,
        all_continuous_cols=all_cont_cols,
        continuous_medians=continuous_medians,
        binary_modes=binary_modes,
        nominal_modes=nominal_modes,
        log1p_cols=log1p_cols,
        clip_bounds=clip_bounds,
        scaler=scaler,
        feature_cols=feature_cols,
    )


def transform_dataset(df: pd.DataFrame, state: PreprocessorState) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Transform a single dataset (Train or Test) separately using frozen Train state."""
    input_feature_cols = [
        c for c in df.columns
        if c not in set(META_COLS + TARGET_COLS) and c not in MODEL_EXCLUDED_RAW_FEATURES
    ]

    # Step 1: Capture missingness stats before imputation
    raw_missing_per_col = {
        c: int(df[c].isna().sum()) for c in input_feature_cols if int(df[c].isna().sum()) > 0
    }
    miss_flags_df = compute_missingness_indicators(df, input_feature_cols)

    # Step 2: Impute base continuous columns with frozen Train medians
    cont_df = pd.DataFrame(index=df.index)
    for c in state.base_continuous_cols:
        cont_df[c] = pd.to_numeric(df[c], errors="coerce").fillna(state.continuous_medians[c]).astype(np.float64)

    # Step 3: Derive clinical lipid / nutrient / sleep features
    eng_df = add_clinical_features(cont_df)
    all_cont_df = pd.concat([cont_df, eng_df], axis=1)[state.all_continuous_cols]

    # Step 4: Apply log1p to train-identified right-skewed columns
    for c in state.log1p_cols:
        # Guard against negative values if unseen test sample is slightly <0 after future changes
        all_cont_df[c] = np.log1p(np.maximum(all_cont_df[c], 0.0))

    # Step 5: Apply frozen [0.1%, 99.9%] winsorization caps
    clipped_values_count = 0
    for c, (lo, hi) in state.clip_bounds.items():
        vals = all_cont_df[c]
        clipped_values_count += int(((vals < lo) | (vals > hi)).sum())
        all_cont_df[c] = vals.clip(lower=lo, upper=hi)

    # Step 6: Scale continuous columns with frozen Train StandardScaler
    scaled_arr = state.scaler.transform(all_cont_df.to_numpy(dtype=np.float64))
    scaled_df = pd.DataFrame(np.round(scaled_arr, 6), columns=state.all_continuous_cols, index=df.index)

    # Step 7: Encode binary & nominal categoricals using frozen Train modes
    cat_df = encode_binary_and_nominal(df, state.binary_modes, state.nominal_modes)

    # Step 8: Assemble final DataFrame (Metadata + Targets + Ordered ML Features)
    meta_targets_df = df[META_COLS + TARGET_COLS].copy()
    for t in TARGET_COLS:
        meta_targets_df[t] = meta_targets_df[t].astype(np.int8)

    out_df = pd.concat([meta_targets_df, scaled_df, cat_df, miss_flags_df], axis=1)
    expected_cols = META_COLS + TARGET_COLS + state.feature_cols
    out_df = out_df[expected_cols]

    feature_matrix = out_df[state.feature_cols].to_numpy(dtype=np.float64)
    stats = {
        "rows": int(len(out_df)),
        "cols": int(out_df.shape[1]),
        "raw_missing_cells_imputed": int(sum(raw_missing_per_col.values())),
        "raw_missing_by_col": raw_missing_per_col,
        "missingness_flag_positives": {c: int(miss_flags_df[c].sum()) for c in miss_flags_df.columns},
        "winsorized_cell_count": int(clipped_values_count),
        "final_feature_nans": int(np.isnan(feature_matrix).sum()),
        "final_feature_infs": int(np.isinf(feature_matrix).sum()),
        "max_abs_scaled_value": round(float(np.abs(scaled_arr).max()), 4),
    }
    return out_df, stats


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Preprocess BioNutriGPT clean Train & Test CSVs separately.")
    parser.add_argument("--train-in", type=Path, default=base_dir / "BioNutriGPT_train_2007_2016_clean.csv")
    parser.add_argument("--test-in", type=Path, default=base_dir / "BioNutriGPT_test_2017_2020_clean.csv")
    parser.add_argument("--train-out", type=Path, default=base_dir / "BioNutriGPT_train_2007_2016_ml_ready.csv")
    parser.add_argument("--test-out", type=Path, default=base_dir / "BioNutriGPT_test_2017_2020_ml_ready.csv")
    parser.add_argument("--artifacts", type=Path, default=base_dir / "preprocessing_artifacts.joblib")
    parser.add_argument("--report", type=Path, default=base_dir / "preprocessing_report.json")
    args = parser.parse_args()

    print(f"[1/4] Loading Train clean CSV: {args.train_in}")
    train_raw = pd.read_csv(args.train_in, low_memory=False)
    print(f"[2/4] Loading Test clean CSV : {args.test_in}")
    test_raw = pd.read_csv(args.test_in, low_memory=False)

    assert set(train_raw["SEQN"]).isdisjoint(set(test_raw["SEQN"])), "SEQN overlap detected between Train and Test!"

    print("[3/4] Fitting preprocessing parameters STRICTLY on Train (2007-2016)...")
    state = fit_preprocessor(train_raw)

    print("      Transforming Train separately ->", args.train_out.name)
    train_ml, train_stats = transform_dataset(train_raw, state)

    print("      Transforming Test separately  ->", args.test_out.name)
    test_ml, test_stats = transform_dataset(test_raw, state)

    assert list(train_ml.columns) == list(test_ml.columns), "Train/Test column mismatch!"
    assert train_stats["final_feature_nans"] == 0 and train_stats["final_feature_infs"] == 0
    assert test_stats["final_feature_nans"] == 0 and test_stats["final_feature_infs"] == 0

    print("[4/4] Saving ML-ready CSVs, joblib artifacts, and JSON audit report...")
    train_ml.to_csv(args.train_out, index=False)
    test_ml.to_csv(args.test_out, index=False)

    joblib.dump(
        {
            "base_continuous_cols": state.base_continuous_cols,
            "all_continuous_cols": state.all_continuous_cols,
            "continuous_medians": state.continuous_medians,
            "binary_modes": state.binary_modes,
            "nominal_modes": state.nominal_modes,
            "log1p_cols": state.log1p_cols,
            "clip_bounds": state.clip_bounds,
            "scaler": state.scaler,
            "meta_cols": META_COLS,
            "target_cols": TARGET_COLS,
            "feature_cols": state.feature_cols,
            "continuous_cols": state.all_continuous_cols,
            "binary_and_onehot_cols": [c for c in state.feature_cols if c not in state.all_continuous_cols],
        },
        args.artifacts,
    )

    report = {
        "train_input": str(args.train_in),
        "test_input": str(args.test_in),
        "train_output": str(args.train_out),
        "test_output": str(args.test_out),
        "zero_seqn_overlap_verified": True,
        "identical_columns_verified": True,
        "total_output_columns": int(train_ml.shape[1]),
        "metadata_cols": META_COLS,
        "target_cols": TARGET_COLS,
        "model_feature_count": len(state.feature_cols),
        "continuous_scaled_count": len(state.all_continuous_cols),
        "engineered_clinical_cols": ENGINEERED_COLS,
        "log1p_transformed_cols": state.log1p_cols,
        "binary_and_onehot_count": len(state.feature_cols) - len(state.all_continuous_cols),
        "train_fitted_medians_for_missing": {
            c: round(state.continuous_medians[c], 4)
            for c in train_stats["raw_missing_by_col"]
            if c in state.continuous_medians
        },
        "train_fitted_modes_for_missing": {
            c: state.binary_modes.get(c, state.nominal_modes.get(c))
            for c in train_stats["raw_missing_by_col"]
            if c in state.binary_modes or c in state.nominal_modes
        },
        "train_stats": train_stats,
        "test_stats": test_stats,
    }
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[Done] Train ML-Ready: {train_ml.shape} | Test ML-Ready: {test_ml.shape}")
    print(f"       Features: {len(state.feature_cols)} ({len(state.all_continuous_cols)} scaled continuous + {len(state.feature_cols) - len(state.all_continuous_cols)} binary/OHE)")
    print(f"       NaNs -> Train: {train_stats['final_feature_nans']}, Test: {test_stats['final_feature_nans']}")


if __name__ == "__main__":
    main()
