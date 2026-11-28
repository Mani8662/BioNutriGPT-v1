"""Statistical Analysis for Stage 2 -> Stage 3 Translation Study.

Calculates:
1. Full denominators, errors, failure categories across 900 solves.
2. Two separate evaluations:
   A. Diagnostic Target-to-Meal Translation (all 100 participants per arm)
   B. End-to-End Pipeline Acceptance (Stage 2 Accepted AND Stage 3 Accepted)
3. Participant-level paired contrasts:
   - Primary: Model-Guided (C) vs Minimum-Change (B)
   - Secondary: Model-Guided (C) vs Rules-Only (A)
   - Secondary: Minimum-Change (B) vs Rules-Only (A)
4. Exact paired binomial discordance tests on discordant pairs.
5. 10,000 paired participant bootstrap resamples with seed 20261123,
   resampling participants with all arms/modes together.
6. Paired mean difference bootstrap 95% CIs for continuous target distances.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STUDY_NAME = "stage2_stage3_translation_study_v1"
STUDY_DIR = ROOT / "Saved Model" / STUDY_NAME
CHECKPOINT_DIR = STUDY_DIR / "checkpoints"
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 20261123


def exact_paired_binary_test(arm1_arr: np.ndarray, arm2_arr: np.ndarray) -> dict[str, Any]:
    """Exact two-sided binomial discordance test on paired binary arrays."""
    # discordant pairs:
    # wins: arm1 == 1 and arm2 == 0
    # losses: arm1 == 0 and arm2 == 1
    wins = int(np.sum((arm1_arr == 1) & (arm2_arr == 0)))
    losses = int(np.sum((arm1_arr == 0) & (arm2_arr == 1)))
    n_discordant = wins + losses

    if n_discordant == 0:
        p_val = 1.0
    else:
        # Two-sided binomial test under H0: p = 0.5
        res = binomtest(wins, n_discordant, p=0.5, alternative="two-sided")
        p_val = float(res.pvalue)

    return {
        "wins": wins,
        "losses": losses,
        "n_discordant": n_discordant,
        "two_sided_p": p_val,
    }


def analyze_study():
    print("=" * 70)
    print(f"Running Statistical Analysis: {STUDY_NAME}")
    print("=" * 70)

    # Load all 900 checkpoints
    cp_files = sorted(CHECKPOINT_DIR.glob("*.json"))
    assert len(cp_files) == 900, f"Expected exactly 900 checkpoints, found {len(cp_files)}"

    records = [json.loads(f.read_text(encoding="utf-8")) for f in cp_files]
    df = pd.DataFrame(records)

    arms = ["rules_only", "minimum_change", "model_guided"]
    modes = ["pure_vegetarian", "veg_nonveg_combo", "pure_non_vegetarian"]
    case_indices = sorted(df["case_idx"].unique())
    n_participants = len(case_indices)
    assert n_participants == 100, f"Expected 100 participants, found {n_participants}"

    analysis_out: dict[str, Any] = {
        "study": STUDY_NAME,
        "sample_size": n_participants,
        "total_solves": len(df),
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
        "arms": {},
        "evaluations": {},
        "paired_contrasts": [],
    }

    # 1. Summary by Arm (across all 3 modes = 300 solves per arm)
    for arm in arms:
        arm_df = df[df["arm"] == arm]
        s2_acc = int(arm_df.drop_duplicates("case_idx")["stage2_accepted"].sum())
        s3_acc = int(arm_df["stage3_accepted"].sum())
        e2e_acc = int(arm_df["end_to_end_accepted"].sum())

        dists = arm_df["normalized_target_distance"].dropna().values
        mapes = arm_df["core_nutrients_mape"].dropna().values
        times = arm_df["solve_time_seconds"].values

        analysis_out["arms"][arm] = {
            "total_attempts": len(arm_df),
            "stage2_accepted_participants": s2_acc,
            "stage2_accepted_pct": round(s2_acc / 100 * 100.0, 1),
            "stage3_accepted_solves": s3_acc,
            "stage3_accepted_pct": round(s3_acc / len(arm_df) * 100.0, 1),
            "end_to_end_accepted_solves": e2e_acc,
            "end_to_end_accepted_pct": round(e2e_acc / len(arm_df) * 100.0, 1),
            "normalized_distance": {
                "mean": round(float(np.mean(dists)), 4) if len(dists) else None,
                "median": round(float(np.median(dists)), 4) if len(dists) else None,
                "std": round(float(np.std(dists)), 4) if len(dists) else None,
                "iqr": round(float(np.percentile(dists, 75) - np.percentile(dists, 25)), 4) if len(dists) else None,
            },
            "core_nutrients_mape": {
                "mean": round(float(np.mean(mapes)), 2) if len(mapes) else None,
                "median": round(float(np.median(mapes)), 2) if len(mapes) else None,
            },
            "solve_time_seconds": {
                "mean": round(float(np.mean(times)), 4),
                "median": round(float(np.median(times)), 4),
            },
        }

    # 2. Detailed Breakdown by Mode and Arm
    cell_stats = {}
    for mode in modes:
        cell_stats[mode] = {}
        for arm in arms:
            cell = df[(df["dietary_mode"] == mode) & (df["arm"] == arm)]
            cell_s2 = int(cell["stage2_accepted"].sum())
            cell_s3 = int(cell["stage3_accepted"].sum())
            cell_e2e = int(cell["end_to_end_accepted"].sum())
            dists = cell["normalized_target_distance"].dropna().values

            cell_stats[mode][arm] = {
                "n": len(cell),
                "stage2_accepted": cell_s2,
                "stage3_accepted": cell_s3,
                "stage3_accepted_pct": round(cell_s3 / len(cell) * 100.0, 1),
                "end_to_end_accepted": cell_e2e,
                "end_to_end_accepted_pct": round(cell_e2e / len(cell) * 100.0, 1),
                "median_target_distance": round(float(np.median(dists)), 4) if len(dists) else None,
            }
    analysis_out["breakdown_by_mode"] = cell_stats

    # 3. Paired Participant Bootstrap (Resample 100 participants with all arms/modes attached)
    rng = np.random.RandomState(BOOTSTRAP_SEED)

    # Pivot wide to participant level
    # Index: case_idx (0..99)
    # Columns: (arm, mode, endpoint)
    pivoted = df.set_index(["case_idx", "arm", "dietary_mode"]).unstack(["arm", "dietary_mode"])

    contrast_specs = [
        ("Primary: Model-Guided (C) vs Minimum-Change (B)", "primary", "model_guided", "minimum_change"),
        ("Secondary: Model-Guided (C) vs Rules-Only (A)", "secondary", "model_guided", "rules_only"),
        ("Secondary: Minimum-Change (B) vs Rules-Only (A)", "secondary", "model_guided", "rules_only"),
        ("Secondary: Minimum-Change (B) vs Rules-Only (A)", "secondary", "minimum_change", "rules_only"),
    ]
    # Remove duplicate secondary
    contrast_specs = [
        ("Primary: Model-Guided (C) vs Minimum-Change (B)", "primary", "model_guided", "minimum_change"),
        ("Secondary: Model-Guided (C) vs Rules-Only (A)", "secondary", "model_guided", "rules_only"),
        ("Secondary: Minimum-Change (B) vs Rules-Only (A)", "secondary", "minimum_change", "rules_only"),
    ]

    contrasts_list = []
    csv_rows = []

    for label, role, arm1, arm2 in contrast_specs:
        contrast_res: dict[str, Any] = {
            "contrast_label": label,
            "role": role,
            "arm_1": arm1,
            "arm_2": arm2,
            "evaluation_a_stage3_feasibility": {},
            "evaluation_b_end_to_end_acceptance": {},
            "soft_target_distance_difference": {},
        }

        # --- Evaluation A: Overall Stage 3 Feasibility (pooled across 3 modes = 300 solves per arm) ---
        arm1_s3 = df[df["arm"] == arm1].sort_values(["case_idx", "dietary_mode"])["stage3_accepted"].values.astype(int)
        arm2_s3 = df[df["arm"] == arm2].sort_values(["case_idx", "dietary_mode"])["stage3_accepted"].values.astype(int)
        bin_s3 = exact_paired_binary_test(arm1_s3, arm2_s3)
        diff_s3_pp = float((np.mean(arm1_s3) - np.mean(arm2_s3)) * 100.0)

        contrast_res["evaluation_a_stage3_feasibility"] = {
            "point_diff_pp": round(diff_s3_pp, 2),
            "exact_p": bin_s3["two_sided_p"],
            "wins": bin_s3["wins"],
            "losses": bin_s3["losses"],
        }

        # --- Evaluation B: Overall End-to-End Acceptance (pooled across 3 modes) ---
        arm1_e2e = df[df["arm"] == arm1].sort_values(["case_idx", "dietary_mode"])["end_to_end_accepted"].values.astype(int)
        arm2_e2e = df[df["arm"] == arm2].sort_values(["case_idx", "dietary_mode"])["end_to_end_accepted"].values.astype(int)
        bin_e2e = exact_paired_binary_test(arm1_e2e, arm2_e2e)
        diff_e2e_pp = float((np.mean(arm1_e2e) - np.mean(arm2_e2e)) * 100.0)

        # Participant bootstrap for End-to-End acceptance & Target distance
        boot_e2e_diffs = []
        boot_dist_diffs = []

        # Prepare per-participant paired vectors
        # For each participant: average e2e rate across 3 modes
        p_arm1_e2e = df[df["arm"] == arm1].groupby("case_idx")["end_to_end_accepted"].mean().values
        p_arm2_e2e = df[df["arm"] == arm2].groupby("case_idx")["end_to_end_accepted"].mean().values

        # For distance: average distance across 3 modes where available
        p_arm1_dist = df[df["arm"] == arm1].groupby("case_idx")["normalized_target_distance"].mean().values
        p_arm2_dist = df[df["arm"] == arm2].groupby("case_idx")["normalized_target_distance"].mean().values
        p_dist_diff = p_arm1_dist - p_arm2_dist

        for _ in range(BOOTSTRAP_RESAMPLES):
            boot_idx = rng.randint(0, n_participants, size=n_participants)
            boot_e2e_diffs.append(np.mean(p_arm1_e2e[boot_idx] - p_arm2_e2e[boot_idx]) * 100.0)
            boot_dist_diffs.append(np.mean(p_dist_diff[boot_idx]))

        e2e_ci_lo = float(np.percentile(boot_e2e_diffs, 2.5))
        e2e_ci_hi = float(np.percentile(boot_e2e_diffs, 97.5))
        e2e_se = float(np.std(boot_e2e_diffs))

        dist_diff_pt = float(np.mean(p_dist_diff))
        dist_ci_lo = float(np.percentile(boot_dist_diffs, 2.5))
        dist_ci_hi = float(np.percentile(boot_dist_diffs, 97.5))
        dist_se = float(np.std(boot_dist_diffs))

        contrast_res["evaluation_b_end_to_end_acceptance"] = {
            "point_diff_pp": round(diff_e2e_pp, 2),
            "ci_lower_pp": round(e2e_ci_lo, 2),
            "ci_upper_pp": round(e2e_ci_hi, 2),
            "std_error_pp": round(e2e_se, 2),
            "exact_p": bin_e2e["two_sided_p"],
            "wins": bin_e2e["wins"],
            "losses": bin_e2e["losses"],
        }

        contrast_res["soft_target_distance_difference"] = {
            "point_diff": round(dist_diff_pt, 4),
            "ci_lower": round(dist_ci_lo, 4),
            "ci_upper": round(dist_ci_hi, 4),
            "std_error": round(dist_se, 4),
        }

        # Cell-specific contrasts for each dietary mode
        per_mode_e2e = {}
        for mode in modes:
            m_a1 = df[(df["dietary_mode"] == mode) & (df["arm"] == arm1)].sort_values("case_idx")["end_to_end_accepted"].values.astype(int)
            m_a2 = df[(df["dietary_mode"] == mode) & (df["arm"] == arm2)].sort_values("case_idx")["end_to_end_accepted"].values.astype(int)
            m_bin = exact_paired_binary_test(m_a1, m_a2)
            m_diff = float((np.mean(m_a1) - np.mean(m_a2)) * 100.0)

            # Mode bootstrap
            m_boot_diffs = []
            for _ in range(BOOTSTRAP_RESAMPLES):
                b_idx = rng.randint(0, n_participants, size=n_participants)
                m_boot_diffs.append(np.mean(m_a1[b_idx] - m_a2[b_idx]) * 100.0)

            per_mode_e2e[mode] = {
                "point_diff_pp": round(m_diff, 2),
                "ci_lower_pp": round(float(np.percentile(m_boot_diffs, 2.5)), 2),
                "ci_upper_pp": round(float(np.percentile(m_boot_diffs, 97.5)), 2),
                "exact_p": m_bin["two_sided_p"],
                "wins": m_bin["wins"],
                "losses": m_bin["losses"],
            }
        contrast_res["per_mode_end_to_end"] = per_mode_e2e

        contrasts_list.append(contrast_res)

        # CSV row
        csv_rows.append({
            "contrast": label,
            "role": role,
            "arm1": arm1,
            "arm2": arm2,
            "s3_feasibility_diff_pp": round(diff_s3_pp, 2),
            "s3_feasibility_exact_p": bin_s3["two_sided_p"],
            "e2e_acceptance_diff_pp": round(diff_e2e_pp, 2),
            "e2e_ci_lower_pp": round(e2e_ci_lo, 2),
            "e2e_ci_upper_pp": round(e2e_ci_hi, 2),
            "e2e_exact_p": bin_e2e["two_sided_p"],
            "dist_diff": round(dist_diff_pt, 4),
            "dist_ci_lower": round(dist_ci_lo, 4),
            "dist_ci_upper": round(dist_ci_hi, 4),
        })

    analysis_out["paired_contrasts"] = contrasts_list

    # Save analysis_summary.json
    summary_path = STUDY_DIR / "analysis_summary.json"
    summary_path.write_text(json.dumps(analysis_out, indent=2), encoding="utf-8")
    print(f"Analysis summary written to {summary_path}")

    # Save paired_contrasts.csv
    csv_path = STUDY_DIR / "paired_contrasts.csv"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"Paired contrasts written to {csv_path}")

    return analysis_out


if __name__ == "__main__":
    analyze_study()
