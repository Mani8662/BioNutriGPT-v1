"""
Download and Merge CDC NHANES 2017-March 2020 Pre-Pandemic Cohort (`P_*` XPTs).

Why this cycle matters (and critical anti-leakage note):
  CDC's `2017-March 2020 Pre-Pandemic` release (`SDDSRVYR = 66`) combines
  `2017-2018` with `2019-March 2020` under re-randomized SEQN IDs (109263-124822).
  Because `2017-2018` participants appear inside this release with new SEQNs,
  merging `2017-2020` with `2007-2016` (and dropping the standalone `2017-2018`
  rows) adds ~3,800+ genuine 2019-March 2020 adults with ZERO duplicate-patient
  leakage across train/val/test.
"""

from __future__ import annotations

import argparse
import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


CDC_PRE_PANDEMIC_BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/"

PRE_PANDEMIC_FILES = [
    ("P_DEMO.xpt", "Demographics"),
    ("P_GHB.xpt", "HbA1c"),
    ("P_GLU.xpt", "Fasting Glucose"),
    ("P_INS.xpt", "Fasting Insulin"),
    ("P_HDL.xpt", "HDL Cholesterol"),
    ("P_TCHOL.xpt", "Total Cholesterol"),
    ("P_TRIGLY.xpt", "Triglycerides & LDL"),
    ("P_BMX.xpt", "Body Measurements (for TARGET_OBESITY only)"),
    ("P_MCQ.xpt", "Medical Conditions (for TARGET_CVD only)"),
    ("P_DIQ.xpt", "Diabetes Questionnaire (for TARGET_DIABETES only)"),
    ("P_DR1TOT.xpt", "Day-1 24h Dietary Recall Total Nutrients"),
    ("P_PAQ.xpt", "Physical Activity (WHO GPAQ)"),
    ("P_SLQ.xpt", "Sleep Disorders"),
    ("P_SMQ.xpt", "Smoking"),
    ("P_ALQ.xpt", "Alcohol Use"),
]


def download_xpt(url: str, timeout: int = 60) -> pd.DataFrame:
    """Fetch a single CDC NHANES SAS XPORT (.xpt) file into a DataFrame."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return pd.read_sas(io.BytesIO(resp.read()), format="xport")


def compute_clinical_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Apply exact BioNutriGPT ground-truth target definitions."""
    out = df.copy()

    # 1. TARGET_DIABETES: HbA1c >= 6.5 or Fasting Glucose >= 126 or DIQ010 == 1
    hba1c = pd.to_numeric(out.get("LBXGH"), errors="coerce")
    glucose = pd.to_numeric(out.get("LBXGLU"), errors="coerce")
    diq = pd.to_numeric(out.get("DIQ010"), errors="coerce")
    diab_cond = (hba1c >= 6.5) | (glucose >= 126.0) | (diq == 1.0)
    diab_eval = hba1c.notna() | glucose.notna() | diq.isin([1.0, 2.0])
    out["TARGET_DIABETES"] = np.where(diab_eval, diab_cond.astype(float), np.nan)

    # 2. TARGET_OBESITY: BMXBMI >= 30.0
    bmi = pd.to_numeric(out.get("BMXBMI"), errors="coerce")
    out["TARGET_OBESITY"] = np.where(bmi.notna(), (bmi >= 30.0).astype(float), np.nan)

    # 3. TARGET_CVD: Heart failure / CHD / Angina / Heart attack / Stroke
    cvd_cols = [c for c in ["MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F"] if c in out.columns]
    any_cvd = np.zeros(len(out), dtype=bool)
    any_valid = np.zeros(len(out), dtype=bool)
    for col in cvd_cols:
        series = pd.to_numeric(out[col], errors="coerce")
        any_cvd |= series == 1.0
        any_valid |= series.isin([1.0, 2.0])
    out["TARGET_CVD"] = np.where(any_valid, any_cvd.astype(float), np.nan)

    return out


def download_and_merge_2017_2020(out_csv: Path, adults_only: bool = True) -> pd.DataFrame:
    """Download all 15 Pre-Pandemic XPT modules, merge by SEQN, and derive targets."""
    tables: list[tuple[str, pd.DataFrame]] = []
    for filename, desc in PRE_PANDEMIC_FILES:
        url = CDC_PRE_PANDEMIC_BASE + filename
        print(f"Downloading {desc:46s} ({filename})...")
        part = download_xpt(url)
        print(f"  -> {part.shape[0]:,} rows, {part.shape[1]} cols")
        tables.append((filename, part))

    merged = tables[0][1].copy()
    for _, part in tables[1:]:
        new_cols = [c for c in part.columns if c == "SEQN" or c not in merged.columns]
        merged = pd.merge(merged, part[new_cols], on="SEQN", how="left")

    merged["NHANES_CYCLE"] = "2017-2020"
    if adults_only and "RIDAGEYR" in merged.columns:
        before = len(merged)
        merged = merged[pd.to_numeric(merged["RIDAGEYR"], errors="coerce") >= 18].copy()
        print(f"\nFiltered to adults (RIDAGEYR >= 18): {before:,} -> {len(merged):,} rows")

    merged = compute_clinical_targets(merged)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    print(f"\nSaved merged 2017-2020 raw cohort -> {out_csv} {merged.shape}")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Download CDC NHANES 2017-March 2020 Pre-Pandemic data.")
    parser.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).resolve().parent / "NHANES_2017_2020_PrePandemic_Raw.csv"),
        help="Output CSV path.",
    )
    parser.add_argument("--include-minors", action="store_true", help="Keep participants under 18 as well.")
    args = parser.parse_args()

    download_and_merge_2017_2020(Path(args.out).resolve(), adults_only=not args.include_minors)


if __name__ == "__main__":
    main()
