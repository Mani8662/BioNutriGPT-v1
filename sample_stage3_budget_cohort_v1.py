#!/usr/bin/env python3
"""Sample and freeze a fresh, strictly disjoint cohort of 100 temporal NHANES participants.

Disjoint from:
- 32-case development repair cohort (Saved Model/stage2_constraint_audit_repaired_v1)
- 100-case downstream translation study (Saved Model/stage1_downstream_study_v1)
- 5-case adapter pilot (Saved Model/stage2_research_adapter_pilot_v1)
- 100-case comparison study (Saved Model/stage2_comparison_study_v2)
Total excluded: 237 unique SEQNs.

Output:
Saved Model/stage3_budget_experiment_v1/cases_LOCAL.json
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import fields
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import stage1_deployment as dep
import stage2_counterfactual_engine as s2
from prepare_stage1_repair_v2 import load

OUT_DIR = ROOT / "Saved Model" / "stage3_budget_experiment_v1"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_cases_path = OUT_DIR / "cases_LOCAL.json"
    if out_cases_path.exists():
        print(f"Cohort manifest already exists at {out_cases_path}. Validating...")
        existing = json.loads(out_cases_path.read_text(encoding="utf-8"))
        assert len(existing) == 100
        print(f"Existing cohort verified: 100 cases, SHA-256: {sha(out_cases_path)}")
        return

    # 1. Load data
    clean_path = ROOT / "Data/stage1_repaired_v2/temporal_clean.csv"
    raw_path = ROOT / "Data/NHANES_2017_2020_PrePandemic_Raw.csv"
    clean = pd.read_csv(clean_path)
    raw = pd.read_csv(raw_path, usecols=["SEQN", "BMXWT", "BMXHT", "RIDAGEYR", "RIAGENDR"]).set_index("SEQN")
    assert raw.index.is_unique and clean.SEQN.is_unique

    # 2. Missingness flags and clinical eligibility
    pp = load("budget_cohort_preprocessing", "Data/preprocess_for_training.py")
    inputs = [c for c in clean if c not in set(pp.META_COLS + pp.TARGET_COLS) and c not in pp.MODEL_EXCLUDED_RAW_FEATURES]
    flags = pp.compute_missingness_indicators(clean, inputs)
    joined = clean[["SEQN"]].join(raw, on="SEQN", validate="one_to_one")
    eligible = (
        (flags.sum(axis=1) == 0)
        & joined.BMXWT.between(30, 180)
        & joined.BMXHT.between(120, 210)
        & (joined.RIDAGEYR >= 20)
        & joined.RIAGENDR.isin([1, 2])
    )

    # 3. Known prior evaluations to exclude
    repair_path = ROOT / "Saved Model/stage2_constraint_audit_repaired_v1/cases_LOCAL.json"
    repair_ids = {int(c["SEQN"]) for c in json.loads(repair_path.read_text(encoding="utf-8"))}

    prior_ds_path = ROOT / "Saved Model/stage1_downstream_study_v1/selected_participants_LOCAL.csv"
    prior_ds_ids = set(pd.read_csv(prior_ds_path).SEQN.astype(int))

    pilot_path = ROOT / "Saved Model/stage2_research_adapter_pilot_v1/envelopes_LOCAL.json"
    pilot_ids = {int(c["SEQN"]) for c in json.loads(pilot_path.read_text(encoding="utf-8"))}

    comp_path = ROOT / "Saved Model/stage2_comparison_study_v2/cases_LOCAL.json"
    comp_ids = {int(c["SEQN"]) for c in json.loads(comp_path.read_text(encoding="utf-8"))}

    excluded_ids = repair_ids | prior_ds_ids | pilot_ids | comp_ids
    assert len(excluded_ids) == 237, f"Expected 237 excluded SEQNs, found {len(excluded_ids)}"

    pool = clean[eligible & ~clean.SEQN.isin(excluded_ids)].sort_values("SEQN")
    print(f"Total clean temporal records: {len(clean)}")
    print(f"Eligible records: {int(eligible.sum())}")
    print(f"Pool size after 237 exclusions: {len(pool)}")
    assert len(pool) >= 100, f"Insufficient pool size: {len(pool)}"

    # 4. Sample 100 cases with fixed random seed
    chosen = pool.sample(n=100, random_state=20261124).sort_values("SEQN")
    chosen_ids = set(chosen.SEQN.astype(int).tolist())
    assert len(chosen_ids) == 100
    assert not (chosen_ids & excluded_ids), "Sampled cohort overlaps with excluded SEQNs!"

    # 5. Build feature vectors (87 features) and metadata
    artifacts = s2._load_preprocessing_artifacts(dep.PREPROCESSOR)
    state = pp.PreprocessorState(**{f.name: artifacts[f.name] for f in fields(pp.PreprocessorState)})

    cont = chosen[state.base_continuous_cols].astype(float)
    raw_df = pd.concat([
        cont,
        pp.add_clinical_features(cont),
        pp.encode_binary_and_nominal(chosen, state.binary_modes, state.nominal_modes),
        flags.loc[chosen.index]
    ], axis=1)[list(s2.FEATURE_COLS)]

    cases = []
    for i, row in chosen.iterrows():
        x = raw.loc[row.SEQN]
        cases.append({
            "case": len(cases),
            "SEQN": int(row.SEQN),
            "vector": raw_df.loc[i].tolist(),
            "metadata": {
                "weight_kg": float(x.BMXWT),
                "height_cm": float(x.BMXHT),
                "age_years": float(x.RIDAGEYR),
                "is_female": bool(x.RIAGENDR == 2),
                "activity_level": "sedentary",
                "renal_disease": None,
                "pregnant": None,
                "lactating": None,
            }
        })

    assert len(cases) == 100
    out_cases_path.write_text(json.dumps(cases, indent=2), encoding="utf-8")
    print(f"Saved {len(cases)} cases to {out_cases_path}")
    print(f"Cohort SHA-256: {sha(out_cases_path)}")
    print("Verification:")
    print(f"  Overlap with 32 repair cases: {len(chosen_ids & repair_ids)}")
    print(f"  Overlap with 100 downstream study: {len(chosen_ids & prior_ds_ids)}")
    print(f"  Overlap with 5 pilot cases: {len(chosen_ids & pilot_ids)}")
    print(f"  Overlap with 100 comparison study: {len(chosen_ids & comp_ids)}")


if __name__ == "__main__":
    main()
