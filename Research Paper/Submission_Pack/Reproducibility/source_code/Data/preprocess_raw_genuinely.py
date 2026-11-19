"""
Genuine Raw Dataset Cleaning Pipeline for BioNutriGPT (Option A + Non-Fasting Cohort).

Cleans `BioNutriGPT_merged_raw_2005_2020.csv` purely in real clinical units
(NO ML preprocessing: no imputation, no scaling, no winsorization, no OHE) and
separates into two temporal cohorts:
  - Train Cohort (`2007-2016`) -> `BioNutriGPT_train_2007_2016_clean.csv`
  - Future Test Cohort (`2017-2020`) -> `BioNutriGPT_test_2017_2020_clean.csv`

Design decisions:
  1. Uses the 3 universal non-fasting blood labs (`LBXGH` HbA1c, `LBDHDD` HDL,
     `LBXTC` Total Cholesterol) + derived `NON_HDL_CHOL = LBXTC - LBDHDD`.
     This avoids dropping ~57% of adults who visited in non-fasting sessions and
     keeps 100% real measured laboratory results (0% imputed biomarkers).
  2. Restores all full 2007-2020 WHO GPAQ work/transport/recreation activity
     domains (`PAQ605-PAQ675`, `PAD680`, `PA_TOTAL_MET_MIN_WK`) and `SLQ050`
     sleep disorder status.
  3. Drops `DMDHHSIZ` (omitted by CDC in 2017-2020 P_DEMO) so both train and
     future test share identical real-world measured columns.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


NHANES_ZERO_EPS = 1e-12
TRAIN_CYCLES = (
    "2007-2008",
    "2009-2010",
    "2011-2012",
    "2013-2014",
    "2015-2016",
)
TEST_CYCLES = ("2017-2020",)
VALID_CYCLES = TRAIN_CYCLES + TEST_CYCLES

TARGET_COLS = ["TARGET_DIABETES", "TARGET_OBESITY", "TARGET_CVD"]
NON_FASTING_LAB_COLS = ["LBXGH", "LBDHDD", "LBXTC"]
BIOMARKER_COLS = ["LBXGH", "LBDHDD", "LBXTC", "NON_HDL_CHOL"]

NUTRITION_COLS = [
    "DR1TKCAL", "DR1TPROT", "DR1TCARB", "DR1TSUGR", "DR1TFIBE",
    "DR1TTFAT", "DR1TSFAT", "DR1TMFAT", "DR1TPFAT", "DR1TCHOL",
    "DR1TATOC", "DR1TRET", "DR1TVARA", "DR1TACAR", "DR1TBCAR",
    "DR1TLYCO", "DR1TLZ", "DR1TVB1", "DR1TVB2", "DR1TNIAC",
    "DR1TVB6", "DR1TFOLA", "DR1TFA", "DR1TFF", "DR1TFDFE",
    "DR1TVB12", "DR1TVC", "DR1TVK", "DR1TCALC", "DR1TPHOS",
    "DR1TMAGN", "DR1TIRON", "DR1TZINC", "DR1TCOPP", "DR1TSODI",
    "DR1TPOTA", "DR1TSELE", "DR1TCAFF", "DR1TALCO",
]

DEMO_COLS = ["RIDAGEYR", "INDFMPIR", "RIAGENDR", "DMDEDUC2", "RIDRETH1", "MARITAL_STATUS"]

LIFESTYLE_COLS = [
    "SLQ050",
    "PAQ605",
    "PAQ620",
    "PAQ635",
    "PAQ650",
    "PAQ665",
    "SMOKING_STATUS",
    "SLEEP_HOURS",
    "PAD680",
    "PA_VIG_WORK_MIN_WK",
    "PA_MOD_WORK_MIN_WK",
    "PA_WALK_BIKE_MIN_WK",
    "PA_VIG_REC_MIN_WK",
    "PA_MOD_REC_MIN_WK",
    "PA_TOTAL_MET_MIN_WK",
]

GPAQ_DOMAINS = [
    ("PAQ605", "PAQ610", "PAD615", "PA_VIG_WORK_MIN_WK"),
    ("PAQ620", "PAQ625", "PAD630", "PA_MOD_WORK_MIN_WK"),
    ("PAQ635", "PAQ640", "PAD645", "PA_WALK_BIKE_MIN_WK"),
    ("PAQ650", "PAQ655", "PAD660", "PA_VIG_REC_MIN_WK"),
    ("PAQ665", "PAQ670", "PAD675", "PA_MOD_REC_MIN_WK"),
]

# NHANES Refused / Don't Know sentinel codes
SENTINEL_CODES: dict[str, set[float]] = {
    **{c: {7.0, 9.0} for c in ["DMDEDUC2", "SLQ050", "SMQ020", "SMQ040", "PAQ605", "PAQ620", "PAQ635", "PAQ650", "PAQ665"]},
    **{c: {77.0, 99.0} for c in ["DMDMARTL", "DMDMARTZ", "SLD010H", "SLD012", "PAQ610", "PAQ625", "PAQ640", "PAQ655", "PAQ670"]},
    **{c: {7777.0, 9999.0} for c in ["PAD615", "PAD630", "PAD645", "PAD660", "PAD675", "PAD680"]},
}

# Physiological plausibility bounds for row-level clinical integrity filtering
CLINICAL_ROW_BOUNDS: dict[str, tuple[float, float]] = {
    "DR1TKCAL": (500.0, 8000.0),
    "LBXGH": (3.0, 20.0),
    "LBDHDD": (10.0, 200.0),
    "LBXTC": (50.0, 600.0),
}

FORBIDDEN_LEAKAGE_PREFIXES = ("BMX", "WHQ", "WHD", "BMI", "MCQ", "DIQ", "CDQ")


@dataclass
class AuditLog:
    """Tracks every row filtering decision and sanitization count."""

    raw_rows: int = 0
    raw_cols: int = 0
    steps: list[dict[str, object]] = field(default_factory=list)
    sentinels_replaced: dict[str, int] = field(default_factory=dict)
    float_zeros_fixed: int = 0

    def record(self, stage: str, before: int, after: int, reason: str) -> None:
        self.steps.append(
            {
                "stage": stage,
                "rows_before": int(before),
                "rows_after": int(after),
                "rows_dropped": int(before - after),
                "reason": reason,
            }
        )


def fix_nhanes_float_zeros(df: pd.DataFrame, cols: Iterable[str]) -> tuple[pd.DataFrame, int]:
    """Convert SAS XPT near-zero float artifacts (~5.3976e-79) cleanly to 0.0."""
    target_cols = [c for c in cols if c in df.columns]
    numeric_df = df[target_cols].apply(pd.to_numeric, errors="coerce")
    near_zero_mask = numeric_df.abs() < NHANES_ZERO_EPS
    artifact_count = int((near_zero_mask & (numeric_df != 0.0)).sum().sum())
    df = df.copy()
    df[target_cols] = numeric_df.mask(near_zero_mask, 0.0)
    return df, artifact_count


def sanitize_sentinels(df: pd.DataFrame, audit: AuditLog) -> pd.DataFrame:
    """Replace NHANES Refused / Don't Know sentinel codes with NaN."""
    df = df.copy()
    for col, codes in SENTINEL_CODES.items():
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        bad_mask = series.isin(codes)
        bad_count = int(bad_mask.sum())
        if bad_count > 0:
            audit.sentinels_replaced[col] = bad_count
        df[col] = series.mask(bad_mask, np.nan)
    return df


def harmonize_marital_status(df: pd.DataFrame) -> pd.Series:
    """
    Harmonize NHANES DMDMARTL (2007-2016) & DMDMARTZ (2017-2020) to 3 classes:
      1 = Married or Living with partner (DMDMARTL in {1, 6} or DMDMARTZ == 1)
      2 = Widowed / Divorced / Separated (DMDMARTL in {2, 3, 4} or DMDMARTZ == 2)
      3 = Never married                  (DMDMARTL == 5 or DMDMARTZ == 3)
    """
    out = pd.Series(np.nan, index=df.index, dtype=float)
    if "DMDMARTZ" in df.columns:
        mz = pd.to_numeric(df["DMDMARTZ"], errors="coerce")
        out = out.combine_first(mz.where(mz.isin([1.0, 2.0, 3.0]), np.nan))
    if "DMDMARTL" in df.columns:
        ml = pd.to_numeric(df["DMDMARTL"], errors="coerce")
        mapped = ml.map({1.0: 1.0, 6.0: 1.0, 2.0: 2.0, 3.0: 2.0, 4.0: 2.0, 5.0: 3.0})
        out = out.combine_first(mapped)
    return out


def harmonize_smoking_status(smq020: pd.Series, smq040: pd.Series) -> pd.Series:
    """
    Resolve deterministic smoking skip logic into a 4-level ordinal feature:
      0 = Never smoked (<100 lifetime cigarettes: SMQ020 == 2)
      1 = Former smoker (SMQ020 == 1 and SMQ040 == 3)
      2 = Current some days (SMQ020 == 1 and SMQ040 == 2)
      3 = Current every day (SMQ020 == 1 and SMQ040 == 1)
    """
    out = pd.Series(np.nan, index=smq020.index, dtype=float)
    out.loc[smq020 == 2.0] = 0.0
    out.loc[(smq020 == 1.0) & (smq040 == 3.0)] = 1.0
    out.loc[(smq020 == 1.0) & (smq040 == 2.0)] = 2.0
    out.loc[(smq020 == 1.0) & (smq040 == 1.0)] = 3.0
    return out


def compute_gpaq_weekly_minutes(
    df: pd.DataFrame, gate_col: str, days_col: str, mins_col: str
) -> pd.Series:
    """Compute weekly domain activity minutes from NHANES WHO GPAQ skip logic."""
    gate = df[gate_col]
    days = df[days_col]
    mins = df[mins_col]

    out = pd.Series(np.nan, index=df.index, dtype=float)
    out.loc[gate == 2.0] = 0.0
    valid_yes = (
        (gate == 1.0)
        & days.between(1.0, 7.0, inclusive="both")
        & (mins > 0.0)
        & (mins <= 960.0)
    )
    out.loc[valid_yes] = days.loc[valid_yes] * mins.loc[valid_yes]
    return out


def clean_dataset(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, AuditLog]:
    """Clean 2005-2020 raw NHANES data purely in real clinical units."""
    audit = AuditLog(raw_rows=len(raw_df), raw_cols=raw_df.shape[1])
    work = raw_df.copy()

    # 1. Deduplicate participant IDs
    before = len(work)
    work = work.drop_duplicates(subset=["SEQN"]).copy()
    audit.record("deduplicate_seqn", before, len(work), "Remove duplicate SEQN rows")

    # 2. Restrict to 2007-2020 cycles (drop 2005-2006 lacking WHO GPAQ PAQ605-PAD680)
    before = len(work)
    work["RIDAGEYR"] = pd.to_numeric(work["RIDAGEYR"], errors="coerce")
    work = work[(work["RIDAGEYR"] >= 18.0) & work["NHANES_CYCLE"].isin(VALID_CYCLES)].copy()
    audit.record(
        "filter_cycles_2007_2020",
        before,
        len(work),
        "Drop 2005-2006 cycle (lacks WHO GPAQ PAQ605-PAD680; 100% blank exercise)",
    )

    # 3. Require complete binary targets (0/1) on all 3 outcomes
    before = len(work)
    work = work.dropna(subset=TARGET_COLS).copy()
    for t in TARGET_COLS:
        work[t] = pd.to_numeric(work[t], errors="coerce")
        work = work[work[t].isin([0.0, 1.0])]
    audit.record(
        "require_complete_targets",
        before,
        len(work),
        "Require observed binary {0, 1} labels for Diabetes, Obesity, and CVD",
    )

    # 4. Require all 3 universal non-fasting blood labs measured (0% fabricated labs)
    before = len(work)
    for b in NON_FASTING_LAB_COLS:
        work[b] = pd.to_numeric(work[b], errors="coerce")
    work = work.dropna(subset=NON_FASTING_LAB_COLS).copy()
    audit.record(
        "require_measured_non_fasting_labs",
        before,
        len(work),
        "Require real measured HbA1c (LBXGH), HDL (LBDHDD), and Total Cholesterol (LBXTC)",
    )

    # 5. Require reliable 24h recall (DR1DRSTZ == 1) and all 39 complete nutrition fields
    before = len(work)
    work["DR1DRSTZ"] = pd.to_numeric(work["DR1DRSTZ"], errors="coerce")
    for nut in NUTRITION_COLS:
        work[nut] = pd.to_numeric(work[nut], errors="coerce")
    work = work[work["DR1DRSTZ"] == 1.0].dropna(subset=NUTRITION_COLS).copy()
    audit.record(
        "require_reliable_nutrition",
        before,
        len(work),
        "Require DR1DRSTZ == 1 reliable dietary recall and all 39 DR1T nutrients",
    )

    # 6. Fix SAS floating-point zero underflow & sanitize NHANES sentinel codes
    numeric_candidates = [
        c
        for c in (
            NON_FASTING_LAB_COLS
            + NUTRITION_COLS
            + ["RIDAGEYR", "INDFMPIR", "RIAGENDR", "RIDRETH1", "DMDEDUC2"]
            + ["DMDMARTL", "DMDMARTZ", "SLQ050", "SLD010H", "SLD012", "SMQ020", "SMQ040", "PAD680"]
            + [col for domain in GPAQ_DOMAINS for col in domain[:3]]
        )
        if c in work.columns
    ]
    work, n_fixed = fix_nhanes_float_zeros(work, numeric_candidates)
    audit.float_zeros_fixed = n_fixed
    work = sanitize_sentinels(work, audit)

    # Derive Non-HDL Cholesterol (Total Cholesterol - HDL)
    work["NON_HDL_CHOL"] = work["LBXTC"] - work["LBDHDD"]

    # 7. Filter physiologically implausible dietary recalls & lab assay extremes
    before = len(work)
    valid_mask = pd.Series(True, index=work.index)
    for col, (lo, hi) in CLINICAL_ROW_BOUNDS.items():
        valid_mask &= work[col].between(lo, hi, inclusive="both")
    valid_mask &= work["NON_HDL_CHOL"] > 0.0
    for nut in NUTRITION_COLS:
        valid_mask &= work[nut] >= 0.0
    work = work[valid_mask].copy()
    audit.record(
        "filter_clinical_and_dietary_outliers",
        before,
        len(work),
        "Drop implausible recalls (DR1TKCAL <500 or >8000 kcal) & lab extremes",
    )

    # 8. Harmonize cross-wave demographic & full GPAQ lifestyle features
    work["MARITAL_STATUS"] = harmonize_marital_status(work)

    sleep = pd.to_numeric(work["SLD010H"], errors="coerce").combine_first(
        pd.to_numeric(work["SLD012"], errors="coerce")
    )
    work["SLEEP_HOURS"] = sleep.where(sleep.between(1.0, 16.0, inclusive="both"), np.nan)
    work["SMOKING_STATUS"] = harmonize_smoking_status(work["SMQ020"], work["SMQ040"])
    work["PAD680"] = work["PAD680"].where(work["PAD680"].between(0.0, 1440.0, inclusive="both"), np.nan)

    for gate_col, days_col, mins_col, out_col in GPAQ_DOMAINS:
        work[out_col] = compute_gpaq_weekly_minutes(work, gate_col, days_col, mins_col)

    work["PA_TOTAL_MET_MIN_WK"] = (
        4.0 * (work["PA_MOD_WORK_MIN_WK"] + work["PA_WALK_BIKE_MIN_WK"] + work["PA_MOD_REC_MIN_WK"])
        + 8.0 * (work["PA_VIG_WORK_MIN_WK"] + work["PA_VIG_REC_MIN_WK"])
    )

    clean_cols = (
        ["SEQN", "NHANES_CYCLE"]
        + TARGET_COLS
        + BIOMARKER_COLS
        + DEMO_COLS
        + NUTRITION_COLS
        + LIFESTYLE_COLS
    )
    leaked = [c for c in clean_cols if c.startswith(FORBIDDEN_LEAKAGE_PREFIXES)]
    if leaked:
        raise ValueError(f"Forbidden leakage column detected: {leaked}")

    clean_df = work[clean_cols].copy()
    clean_df["SEQN"] = clean_df["SEQN"].astype(np.int64)
    for t in TARGET_COLS:
        clean_df[t] = clean_df[t].astype(np.int64)

    clean_df = clean_df.sort_values("SEQN").reset_index(drop=True)
    return clean_df, audit


def summarize_split(df: pd.DataFrame) -> dict[str, object]:
    """Summarize one temporal split of the cleaned cohort."""
    return {
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "cycle_counts": {str(k): int(v) for k, v in df["NHANES_CYCLE"].value_counts().items()},
        "remaining_nulls": {k: int(v) for k, v in df.isna().sum().items() if v > 0},
        "target_prevalence": {
            t: {"positives": int(df[t].sum()), "prevalence_pct": round(float(df[t].mean() * 100.0), 2)}
            for t in TARGET_COLS
        },
    }


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Clean 2005-2020 NHANES data (Option A + Non-Fasting Cohort).")
    parser.add_argument("--input", type=str, default=str(base_dir / "BioNutriGPT_merged_raw_2005_2020.csv"))
    parser.add_argument("--output-dir", type=str, default=str(base_dir))
    args = parser.parse_args()

    raw_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading raw CSV: {raw_path}")
    raw_df = pd.read_csv(raw_path, low_memory=False)
    print(f"  Raw shape: {raw_df.shape}")

    clean_df, audit = clean_dataset(raw_df)

    # Option A Temporal Separation: 2007-2016 Train | 2017-2020 Future Test
    train_df = clean_df[clean_df["NHANES_CYCLE"].isin(TRAIN_CYCLES)].reset_index(drop=True)
    test_df = clean_df[clean_df["NHANES_CYCLE"].isin(TEST_CYCLES)].reset_index(drop=True)
    assert not (set(train_df["SEQN"]) & set(test_df["SEQN"])), "SEQN overlap!"

    train_path = out_dir / "BioNutriGPT_train_2007_2016_clean.csv"
    test_path = out_dir / "BioNutriGPT_test_2017_2020_clean.csv"
    report_path = out_dir / "cleaning_report.json"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    features = [c for c in clean_df.columns if c not in (["SEQN", "NHANES_CYCLE"] + TARGET_COLS)]
    report = {
        "source_dataset": str(raw_path),
        "raw_shape": [audit.raw_rows, audit.raw_cols],
        "total_clean_shape": [int(clean_df.shape[0]), int(clean_df.shape[1])],
        "clean_feature_count": len(features),
        "clean_column_names": list(clean_df.columns),
        "float_zero_artifacts_fixed": audit.float_zeros_fixed,
        "nhanes_sentinels_converted_to_nan": audit.sentinels_replaced,
        "filtering_steps": audit.steps,
        "train_2007_2016": summarize_split(train_df),
        "test_2017_2020": summarize_split(test_df),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n--- Cleaning Audit ---")
    for step in audit.steps:
        print(f"  [{step['stage']}] {step['rows_before']} -> {step['rows_after']} (-{step['rows_dropped']}): {step['reason']}")
    print(f"  Float zero artifacts fixed: {audit.float_zeros_fixed}")
    print(f"  Sentinels converted to NaN: {audit.sentinels_replaced}")
    print(f"\nSaved Train (2007-2016) Clean : {train_path} {train_df.shape}")
    print(f"Saved Test  (2017-2020) Clean : {test_path} {test_df.shape}")
    print(f"Saved Cleaning Report         : {report_path}")


if __name__ == "__main__":
    main()
