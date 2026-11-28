"""Extract a reproducible, stratified real Indian adult cohort from LASI Wave 1.

Reads 4_LASI_W1_Biomarker.sav and 3_LASI_W1_Individual_v4.sav directly from the
downloaded IIPS LASI Wave 1 archive (Data/65dc53fced0231708938236.zip), records
its SHA256 digest, verifies value labels from SPSS metadata, computes full-cohort
summary statistics, and draws a deterministic stratified sample of N=100 real
Indian adults (25 Diabetes, 25 Obesity BMI>=30, 25 CVD, 25 Multimorbid) for
Stage 2/3/4 Indian meal-planning and conversational revision evaluation.

Scope note:
LASI Wave 1 supplies real Indian demographics, measured anthropometry, physical
activity, and prevalent Diabetes/Obesity/CVD status. It does NOT contain serum
lipid assays or 24-hour dietary recalls, so this cohort evaluates Stages 2–4 on
real Indian clinical/anthropometric profiles and does NOT claim external
validation of the 87-feature Stage 1 NHANES Transformer.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import pandas as pd
import pyreadstat

ROOT = Path(__file__).resolve().parent
LASI_ZIP = ROOT / "Data" / "65dc53fced0231708938236.zip"
OUT_COHORT = ROOT / "Data" / "lasi_w1_indian_cohort_100.json"
OUT_SUMMARY = ROOT / "Saved Model" / "lasi_cohort_extraction_summary.json"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _map_activity_level(vig: float | None, mod: float | None) -> str:
    """Map LASI Wave 1 hb211 (vigorous) & hb213 (moderate) frequency codes (1=every day .. 5=hardly ever/never)."""
    v = int(vig) if pd.notna(vig) else 5
    m = int(mod) if pd.notna(mod) else 5
    if v in (1, 2):  # every day or >1/week vigorous
        return "active"
    if v == 3 or m in (1, 2):  # weekly vigorous or regular moderate
        return "moderate"
    if v == 4 or m in (3, 4):  # occasional activity
        return "light"
    return "sedentary"


def main() -> None:
    if not LASI_ZIP.is_file():
        raise FileNotFoundError(f"LASI archive not found at {LASI_ZIP}")
    archive_sha256 = _sha256_file(LASI_ZIP)
    print(f"LASI archive SHA256: {archive_sha256}", flush=True)

    bm_cols = ["prim_key", "state", "residence", "bm067", "bm071", "bm076", "bm017", "bm018"]
    ind_cols = ["prim_key", "dm003", "dm005", "ht002", "ht003", "ht006", "ht007", "ht010", "hb211", "hb213"]

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        with zipfile.ZipFile(LASI_ZIP) as zf:
            bm_path = tmp_dir / "4_LASI_W1_Biomarker.sav"
            bm_path.write_bytes(zf.read("4_LASI_W1_Biomarker.sav"))
            ind_path = tmp_dir / "3_LASI_W1_Individual_v4.sav"
            with zf.open("3_LASI_W1_Individual_v4.sav") as src, ind_path.open("wb") as dst:
                while True:
                    chunk = src.read(4 * 1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)

        df_bm, meta_bm = pyreadstat.read_sav(str(bm_path), usecols=bm_cols)
        df_ind, meta_ind = pyreadstat.read_sav(str(ind_path), usecols=ind_cols)

    print("Value labels verified from SPSS metadata:", flush=True)
    for k in ("dm003", "ht003", "ht006", "ht007", "hb211", "hb213"):
        print(f"  {k}: {meta_ind.variable_value_labels.get(k)}", flush=True)
    state_labels = meta_bm.variable_value_labels.get("state", {})

    # Merge biomarker + individual by prim_key
    df = pd.merge(df_bm, df_ind, on="prim_key", how="inner")
    n_merged = len(df)

    # Filter to valid adult anthropometric & clinical records
    for col in ("bm067", "bm071", "dm005", "dm003", "ht003", "ht006", "ht007"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    valid = df[
        df["dm005"].between(18, 100)
        & df["bm067"].between(120, 210)
        & df["bm071"].between(30, 180)
        & df["dm003"].isin([1, 2])
        & df["ht003"].isin([1, 2])
        & df["ht006"].isin([1, 2])
        & df["ht007"].isin([1, 2])
    ].copy()

    valid["bmi"] = valid["bm071"] / ((valid["bm067"] / 100.0) ** 2)
    valid = valid[valid["bmi"].between(14.0, 55.0)].copy()

    # Decode sex directly from value labels (confirming 2=Female, 1=Male)
    sex_map = meta_ind.variable_value_labels.get("dm003", {})
    female_code = next((int(k) for k, v in sex_map.items() if "female" in str(v).lower()), 2)
    yes_code = 1  # verified from ht003/ht006/ht007 labels (1=Yes, 2=No)

    valid["is_female"] = valid["dm003"].astype(int) == female_code
    valid["has_diabetes"] = valid["ht003"].astype(int) == yes_code
    valid["has_obesity_who30"] = valid["bmi"] >= 30.0
    valid["has_obesity_indian25"] = valid["bmi"] >= 25.0
    valid["has_cvd"] = (valid["ht006"].astype(int) == yes_code) | (valid["ht007"].astype(int) == yes_code)
    valid["has_hypertension"] = pd.to_numeric(valid["ht002"], errors="coerce").fillna(2).astype(int) == yes_code
    valid["activity_level"] = [
        _map_activity_level(v, m) for v, m in zip(valid["hb211"], valid["hb213"])
    ]
    valid["state_name"] = [
        str(state_labels.get(s, s)).strip() if pd.notna(s) else "Unknown"
        for s in valid["state"]
    ]

    # Assign mutually exclusive clinical evaluation strata
    cond_count = (
        valid["has_diabetes"].astype(int)
        + valid["has_obesity_who30"].astype(int)
        + valid["has_cvd"].astype(int)
    )
    valid["stratum"] = "healthy_reference"
    valid.loc[(cond_count == 1) & valid["has_diabetes"], "stratum"] = "diabetes_only"
    valid.loc[(cond_count == 1) & valid["has_obesity_who30"], "stratum"] = "obesity_only"
    valid.loc[(cond_count == 1) & valid["has_cvd"], "stratum"] = "cvd_only"
    valid.loc[cond_count >= 2, "stratum"] = "multimorbid"

    stratum_counts = valid["stratum"].value_counts().to_dict()
    print(f"\nMerged rows: {n_merged} | Valid Indian adults: {len(valid)}", flush=True)
    print("Stratum counts:", stratum_counts, flush=True)

    # Deterministic stratified draw of N=100 real Indian adults (25 per clinical stratum, balanced Male/Female)
    rng = np.random.default_rng(42)
    sampled_rows = []
    for st in ("diabetes_only", "obesity_only", "cvd_only", "multimorbid"):
        sub = valid[valid["stratum"] == st].sort_values("prim_key")
        sub_m = sub[~sub["is_female"]]
        sub_f = sub[sub["is_female"]]
        n_m = min(13, len(sub_m))
        n_f = min(25 - n_m, len(sub_f))
        if n_m + n_f < 25:
            n_m = min(25 - n_f, len(sub_m))
        idx_m = rng.choice(len(sub_m), size=n_m, replace=False)
        idx_f = rng.choice(len(sub_f), size=n_f, replace=False)
        chosen = pd.concat([sub_m.iloc[idx_m], sub_f.iloc[idx_f]], ignore_index=True)
        for _, r in chosen.iterrows():
            sampled_rows.append({
                "lasi_prim_key": str(r["prim_key"]),
                "state": str(r["state_name"]),
                "stratum": str(r["stratum"]),
                "age_years": float(r["dm005"]),
                "is_female": bool(r["is_female"]),
                "height_cm": round(float(r["bm067"]), 1),
                "weight_kg": round(float(r["bm071"]), 1),
                "bmi": round(float(r["bmi"]), 2),
                "waist_cm": round(float(r["bm076"]), 1) if pd.notna(r["bm076"]) and 40 <= float(r["bm076"]) <= 180 else None,
                "sbp_mmhg": round(float(r["bm017"]), 1) if pd.notna(r["bm017"]) and 60 <= float(r["bm017"]) <= 260 else None,
                "dbp_mmhg": round(float(r["bm018"]), 1) if pd.notna(r["bm018"]) and 30 <= float(r["bm018"]) <= 160 else None,
                "activity_level": str(r["activity_level"]),
                "has_diabetes": bool(r["has_diabetes"]),
                "has_obesity_who30": bool(r["has_obesity_who30"]),
                "has_obesity_indian25": bool(r["has_obesity_indian25"]),
                "has_cvd": bool(r["has_cvd"]),
                "has_hypertension": bool(r["has_hypertension"]),
            })

    summary = {
        "source_dataset": "IIPS Longitudinal Ageing Study in India (LASI) Wave 1 (2017-2019)",
        "archive_filename": LASI_ZIP.name,
        "archive_sha256": archive_sha256,
        "scope_note": (
            "LASI Wave 1 provides real Indian adult anthropometry, demographics, physical activity, "
            "and prevalent Diabetes/Obesity/CVD status for Stage 2-4 meal planning and conversational "
            "revision benchmarking. It lacks serum lipids and 24-hour dietary nutrient recalls, so "
            "it is NOT used to claim external validation of the 87-feature Stage 1 NHANES Transformer."
        ),
        "merged_total_n": int(n_merged),
        "valid_indian_adults_n": int(len(valid)),
        "full_cohort_stats": {
            "states_covered": int(valid["state_name"].nunique()),
            "age_mean_sd": [round(float(valid["dm005"].mean()), 2), round(float(valid["dm005"].std()), 2)],
            "female_pct": round(float(valid["is_female"].mean() * 100.0), 2),
            "bmi_mean_sd": [round(float(valid["bmi"].mean()), 2), round(float(valid["bmi"].std()), 2)],
            "diabetes_prevalence_pct": round(float(valid["has_diabetes"].mean() * 100.0), 2),
            "obesity_who30_prevalence_pct": round(float(valid["has_obesity_who30"].mean() * 100.0), 2),
            "obesity_indian25_prevalence_pct": round(float(valid["has_obesity_indian25"].mean() * 100.0), 2),
            "cvd_prevalence_pct": round(float(valid["has_cvd"].mean() * 100.0), 2),
            "hypertension_prevalence_pct": round(float(valid["has_hypertension"].mean() * 100.0), 2),
            "stratum_counts": {k: int(v) for k, v in stratum_counts.items()},
        },
        "stratified_sample_n": len(sampled_rows),
        "stratified_sample_seed": 42,
        "stratified_sample_states": len({r["state"] for r in sampled_rows}),
        "stratified_sample_female_n": sum(1 for r in sampled_rows if r["is_female"]),
        "stratified_sample_male_n": sum(1 for r in sampled_rows if not r["is_female"]),
    }

    OUT_COHORT.write_text(json.dumps({"metadata": summary, "patients": sampled_rows}, indent=2), encoding="utf-8")
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved {len(sampled_rows)} real Indian adult profiles across {summary['stratified_sample_states']} Indian states/UTs to {OUT_COHORT}", flush=True)


if __name__ == "__main__":
    main()
