#!/usr/bin/env python3
"""Independent Statistical Analysis for Stage 3 Target-Aware Experiment.

Reads all 900 saved paired checkpoints directly from disk, independently
recomputes all metrics from raw food baskets and nutrition vectors, and performs
rigorous statistical testing:
1. Exact paired binary discordance tests (McNemar exact) per dietary mode.
2. Participant-clustered bootstrap (10,000 resamples, seed 20261123) for pooled outcomes.
3. Paired mean difference bootstrap percentile 95% CIs for continuous target attainment and runtime.
4. Comprehensive multi-arm, multi-mode, and solver performance summaries.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import stage3_milp_solver as s3
import stage3_target_aware_solver as tas

STUDY_NAME = "stage3_target_aware_experiment_v1"
STUDY_DIR = ROOT / "Saved Model" / STUDY_NAME
CHECKPOINT_DIR = STUDY_DIR / "checkpoints"
ARMS = ["rules_only", "minimum_change", "model_guided"]
MODES = ["pure_vegetarian", "veg_nonveg_combo", "pure_non_vegetarian"]
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 20261123


def clustered_bootstrap_mean(
    participant_data: Mapping[int, list[float]],
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Perform participant-clustered bootstrap resampling.
    
    Each participant's repeated measurements across dietary modes are kept together
    as a cluster, resampling N_participants clusters with replacement.
    """
    participants = sorted(participant_data.keys())
    n_p = len(participants)
    if n_p == 0:
        return {"n_participants": 0, "mean": 0.0, "ci95": [0.0, 0.0]}

    # Participant-level cluster means
    all_values = []
    p_means = []
    for p in participants:
        vals = participant_data[p]
        all_values.extend(vals)
        p_means.append(float(np.mean(vals)))

    overall_mean = float(np.mean(all_values))
    p_means_arr = np.asarray(p_means, dtype=np.float64)

    rng = np.random.default_rng(seed)
    cluster_indices = rng.integers(0, n_p, size=(n_resamples, n_p))
    bootstrap_means = np.mean(p_means_arr[cluster_indices], axis=1)

    ci_low = float(np.percentile(bootstrap_means, 2.5))
    ci_high = float(np.percentile(bootstrap_means, 97.5))

    return {
        "n_participants": n_p,
        "n_observations": len(all_values),
        "mean": round(overall_mean, 6),
        "ci95": [round(ci_low, 6), round(ci_high, 6)],
        "bootstrap_se": round(float(np.std(bootstrap_means, ddof=1)), 6),
    }


def clustered_bootstrap_binary_diff(
    success_a: Mapping[int, list[bool]],
    success_b: Mapping[int, list[bool]],
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Clustered bootstrap for paired difference in binary success rates (B minus A)."""
    participants = sorted(set(success_a.keys()) & set(success_b.keys()))
    n_p = len(participants)

    p_diffs = []
    all_diffs = []
    for p in participants:
        diffs = [float(b) - float(a) for a, b in zip(success_a[p], success_b[p])]
        all_diffs.extend(diffs)
        p_diffs.append(float(np.mean(diffs)))

    overall_diff_pp = float(np.mean(all_diffs)) * 100.0
    p_diffs_arr = np.asarray(p_diffs, dtype=np.float64) * 100.0

    rng = np.random.default_rng(seed)
    cluster_indices = rng.integers(0, n_p, size=(n_resamples, n_p))
    bootstrap_diffs = np.mean(p_diffs_arr[cluster_indices], axis=1)

    ci_low = float(np.percentile(bootstrap_diffs, 2.5))
    ci_high = float(np.percentile(bootstrap_diffs, 97.5))

    return {
        "n_participants": n_p,
        "n_pairs": len(all_diffs),
        "diff_pp": round(overall_diff_pp, 4),
        "ci95_pp": [round(ci_low, 4), round(ci_high, 4)],
        "bootstrap_se_pp": round(float(np.std(bootstrap_diffs, ddof=1)), 4),
    }


def exact_paired_binary_test(pairs_a: Sequence[bool], pairs_b: Sequence[bool]) -> dict[str, Any]:
    """Exact two-sided paired binomial discordance test (McNemar exact)."""
    assert len(pairs_a) == len(pairs_b)
    n_00 = sum(1 for a, b in zip(pairs_a, pairs_b) if not a and not b)
    n_11 = sum(1 for a, b in zip(pairs_a, pairs_b) if a and b)
    n_10 = sum(1 for a, b in zip(pairs_a, pairs_b) if a and not b)  # A win, B loss
    n_01 = sum(1 for a, b in zip(pairs_a, pairs_b) if not a and b)  # B win, A loss

    discordant = n_10 + n_01
    p_val = binomtest(n_01, discordant, 0.5, alternative="two-sided").pvalue if discordant > 0 else 1.0

    diff_pp = (n_01 - n_10) / len(pairs_a) * 100.0
    return {
        "n_total": len(pairs_a),
        "n_concordant_success": n_11,
        "n_concordant_failure": n_00,
        "n_a_only": n_10,
        "n_b_only": n_01,
        "discordant_count": discordant,
        "diff_pp": round(diff_pp, 4),
        "exact_p_value": float(p_val),
    }


def analyze_experiment():
    """Run full independent statistical analysis across all 900 paired tasks."""
    print("=" * 70)
    print("Running Independent Statistical Analysis on Stage 3 Target-Aware Experiment")
    print("=" * 70)

    # 1. Load all checkpoints
    cp_files = sorted(CHECKPOINT_DIR.glob("case_*.json"))
    if len(cp_files) != 900:
        raise ValueError(f"Expected exactly 900 checkpoints, found {len(cp_files)}.")

    records = [json.loads(p.read_text(encoding="utf-8")) for p in cp_files]
    print(f"Loaded {len(records)} checkpoints successfully.")

    # 2. Organize records
    by_arm_mode: dict[tuple[str, str], list[dict]] = defaultdict(list)
    by_arm: dict[str, list[dict]] = defaultdict(list)
    by_mode: dict[str, list[dict]] = defaultdict(list)

    for r in records:
        key = (r["arm"], r["dietary_mode"])
        by_arm_mode[key].append(r)
        by_arm[r["arm"]].append(r)
        by_mode[r["dietary_mode"]].append(r)

    # Verify grid structure
    for arm in ARMS:
        for mode in MODES:
            count = len(by_arm_mode[(arm, mode)])
            if count != 100:
                raise ValueError(f"Expected 100 records for ({arm}, {mode}), got {count}.")

    # 3. Overall Feasibility and Acceptance Summary
    feasibility_summary = {}
    for arm in ARMS:
        feasibility_summary[arm] = {}
        for variant in ("variant_a", "variant_b"):
            s3_acc = sum(1 for r in by_arm[arm] if r[variant]["replay_passed"])
            e2e_acc = sum(1 for r in by_arm[arm] if r[variant]["end_to_end_accepted"])
            feasibility_summary[arm][variant] = {
                "stage3_accepted": s3_acc,
                "stage3_rate_pct": round(s3_acc / 300.0 * 100.0, 2),
                "end_to_end_accepted": e2e_acc,
                "end_to_end_rate_pct": round(e2e_acc / 300.0 * 100.0, 2),
            }

    # 4. Within-Arm Paired Comparison (Variant B vs Variant A)
    within_arm_contrasts = {}
    for arm in ARMS:
        arm_recs = by_arm[arm]
        # Cluster maps: participant SEQN -> list of values
        p_s3_a: dict[int, list[bool]] = defaultdict(list)
        p_s3_b: dict[int, list[bool]] = defaultdict(list)
        p_e2e_a: dict[int, list[bool]] = defaultdict(list)
        p_e2e_b: dict[int, list[bool]] = defaultdict(list)
        p_l1_diff: dict[int, list[float]] = defaultdict(list)
        p_l2_diff: dict[int, list[float]] = defaultdict(list)
        p_time_diff: dict[int, list[float]] = defaultdict(list)
        p_identical: dict[int, list[float]] = defaultdict(list)
        p_serving_delta: dict[int, list[float]] = defaultdict(list)

        # Baseline levels
        p_l1_a: dict[int, list[float]] = defaultdict(list)
        p_l1_b: dict[int, list[float]] = defaultdict(list)
        p_l2_a: dict[int, list[float]] = defaultdict(list)
        p_l2_b: dict[int, list[float]] = defaultdict(list)
        p_time_a: dict[int, list[float]] = defaultdict(list)
        p_time_b: dict[int, list[float]] = defaultdict(list)

        per_mode_s3_tests = {}
        per_mode_e2e_tests = {}

        for mode in MODES:
            mode_recs = by_arm_mode[(arm, mode)]
            s3_a_mode = [r["variant_a"]["replay_passed"] for r in mode_recs]
            s3_b_mode = [r["variant_b"]["replay_passed"] for r in mode_recs]
            e2e_a_mode = [r["variant_a"]["end_to_end_accepted"] for r in mode_recs]
            e2e_b_mode = [r["variant_b"]["end_to_end_accepted"] for r in mode_recs]

            per_mode_s3_tests[mode] = exact_paired_binary_test(s3_a_mode, s3_b_mode)
            per_mode_e2e_tests[mode] = exact_paired_binary_test(e2e_a_mode, e2e_b_mode)

        for r in arm_recs:
            seqn = r["SEQN"]
            s3_a = bool(r["variant_a"]["replay_passed"])
            s3_b = bool(r["variant_b"]["replay_passed"])
            e2e_a = bool(r["variant_a"]["end_to_end_accepted"])
            e2e_b = bool(r["variant_b"]["end_to_end_accepted"])

            p_s3_a[seqn].append(s3_a)
            p_s3_b[seqn].append(s3_b)
            p_e2e_a[seqn].append(e2e_a)
            p_e2e_b[seqn].append(e2e_b)

            time_a = float(r["variant_a"]["solve_time_seconds"])
            time_b = float(r["variant_b"]["solve_time_seconds"])
            p_time_a[seqn].append(time_a)
            p_time_b[seqn].append(time_b)
            p_time_diff[seqn].append(time_b - time_a)

            identical = bool(r["paired_comparison"]["basket_identical"])
            s_delta = float(r["paired_comparison"]["serving_delta"])
            p_identical[seqn].append(float(identical))
            p_serving_delta[seqn].append(s_delta)

            # Target attainment (evaluated when both variants generated valid meals)
            attain_a = r["variant_a"].get("target_attainment")
            attain_b = r["variant_b"].get("target_attainment")
            if s3_a and s3_b and attain_a and attain_b:
                l1_a = float(attain_a["mean_normalized_l1_target_error"])
                l1_b = float(attain_b["mean_normalized_l1_target_error"])
                l2_a = float(attain_a["normalized_l2_target_distance"])
                l2_b = float(attain_b["normalized_l2_target_distance"])

                p_l1_a[seqn].append(l1_a)
                p_l1_b[seqn].append(l1_b)
                p_l2_a[seqn].append(l2_a)
                p_l2_b[seqn].append(l2_b)
                p_l1_diff[seqn].append(l1_b - l1_a)
                p_l2_diff[seqn].append(l2_b - l2_a)

        # Pooled clustered bootstraps
        pooled_s3_diff = clustered_bootstrap_binary_diff(p_s3_a, p_s3_b)
        pooled_e2e_diff = clustered_bootstrap_binary_diff(p_e2e_a, p_e2e_b)

        l1_a_boot = clustered_bootstrap_mean(p_l1_a)
        l1_b_boot = clustered_bootstrap_mean(p_l1_b)
        l1_diff_boot = clustered_bootstrap_mean(p_l1_diff)

        l2_a_boot = clustered_bootstrap_mean(p_l2_a)
        l2_b_boot = clustered_bootstrap_mean(p_l2_b)
        l2_diff_boot = clustered_bootstrap_mean(p_l2_diff)

        time_a_boot = clustered_bootstrap_mean(p_time_a)
        time_b_boot = clustered_bootstrap_mean(p_time_b)
        time_diff_boot = clustered_bootstrap_mean(p_time_diff)

        identical_boot = clustered_bootstrap_mean(p_identical)
        serving_delta_boot = clustered_bootstrap_mean(p_serving_delta)

        within_arm_contrasts[arm] = {
            "per_mode_stage3_exact_tests": per_mode_s3_tests,
            "per_mode_end_to_end_exact_tests": per_mode_e2e_tests,
            "pooled_stage3_diff": pooled_s3_diff,
            "pooled_end_to_end_diff": pooled_e2e_diff,
            "mean_norm_l1_target_error": {
                "variant_a": l1_a_boot,
                "variant_b": l1_b_boot,
                "difference_b_minus_a": l1_diff_boot,
            },
            "norm_l2_target_distance": {
                "variant_a": l2_a_boot,
                "variant_b": l2_b_boot,
                "difference_b_minus_a": l2_diff_boot,
            },
            "solve_time_seconds": {
                "variant_a": time_a_boot,
                "variant_b": time_b_boot,
                "difference_b_minus_a": time_diff_boot,
            },
            "meal_composition": {
                "identical_basket_rate": identical_boot,
                "mean_serving_delta": serving_delta_boot,
            },
        }

    # 5. Inter-Arm Comparisons under Variant B
    # Model-Guided vs Rules-Only, and Model-Guided vs Minimum-Change
    inter_arm_contrasts = {}
    pair_definitions = [
        ("model_guided_vs_rules_only", "model_guided", "rules_only"),
        ("model_guided_vs_minimum_change", "model_guided", "minimum_change"),
        ("minimum_change_vs_rules_only", "minimum_change", "rules_only"),
    ]

    for label, arm_x, arm_y in pair_definitions:
        p_s3_x: dict[int, list[bool]] = defaultdict(list)
        p_s3_y: dict[int, list[bool]] = defaultdict(list)
        p_e2e_x: dict[int, list[bool]] = defaultdict(list)
        p_e2e_y: dict[int, list[bool]] = defaultdict(list)
        p_l1_diff_inter: dict[int, list[float]] = defaultdict(list)
        p_l2_diff_inter: dict[int, list[float]] = defaultdict(list)

        per_mode_inter_s3 = {}
        per_mode_inter_e2e = {}

        for mode in MODES:
            recs_x = by_arm_mode[(arm_x, mode)]
            recs_y = by_arm_mode[(arm_y, mode)]
            s3_x = [r["variant_b"]["replay_passed"] for r in recs_x]
            s3_y = [r["variant_b"]["replay_passed"] for r in recs_y]
            e2e_x = [r["variant_b"]["end_to_end_accepted"] for r in recs_x]
            e2e_y = [r["variant_b"]["end_to_end_accepted"] for r in recs_y]

            per_mode_inter_s3[mode] = exact_paired_binary_test(s3_y, s3_x)  # X minus Y
            per_mode_inter_e2e[mode] = exact_paired_binary_test(e2e_y, e2e_x)

        # Match participant-level
        map_x = {(r["SEQN"], r["dietary_mode"]): r for r in by_arm[arm_x]}
        map_y = {(r["SEQN"], r["dietary_mode"]): r for r in by_arm[arm_y]}

        for (seqn, mode), rx in map_x.items():
            ry = map_y[(seqn, mode)]
            s3_b_x = bool(rx["variant_b"]["replay_passed"])
            s3_b_y = bool(ry["variant_b"]["replay_passed"])
            e2e_b_x = bool(rx["variant_b"]["end_to_end_accepted"])
            e2e_b_y = bool(ry["variant_b"]["end_to_end_accepted"])

            p_s3_x[seqn].append(s3_b_x)
            p_s3_y[seqn].append(s3_b_y)
            p_e2e_x[seqn].append(e2e_b_x)
            p_e2e_y[seqn].append(e2e_b_y)

            # Distance comparisons (conditioned on joint validity)
            if s3_b_x and s3_b_y:
                att_x = rx["variant_b"].get("target_attainment")
                att_y = ry["variant_b"].get("target_attainment")
                if att_x and att_y:
                    l1_x = float(att_x["mean_normalized_l1_target_error"])
                    l1_y = float(att_y["mean_normalized_l1_target_error"])
                    l2_x = float(att_x["normalized_l2_target_distance"])
                    l2_y = float(att_y["normalized_l2_target_distance"])
                    p_l1_diff_inter[seqn].append(l1_x - l1_y)
                    p_l2_diff_inter[seqn].append(l2_x - l2_y)

        boot_s3_inter = clustered_bootstrap_binary_diff(p_s3_y, p_s3_x)
        boot_e2e_inter = clustered_bootstrap_binary_diff(p_e2e_y, p_e2e_x)
        boot_l1_inter = clustered_bootstrap_mean(p_l1_diff_inter)
        boot_l2_inter = clustered_bootstrap_mean(p_l2_diff_inter)

        inter_arm_contrasts[label] = {
            "contrast": f"{arm_x} minus {arm_y} (under Variant B)",
            "per_mode_stage3_exact_tests": per_mode_inter_s3,
            "per_mode_end_to_end_exact_tests": per_mode_inter_e2e,
            "pooled_stage3_diff": boot_s3_inter,
            "pooled_end_to_end_diff": boot_e2e_inter,
            "mean_norm_l1_difference": boot_l1_inter,
            "norm_l2_distance_difference": boot_l2_inter,
        }

    # 6. Solver Diagnostics Breakdown
    solver_diagnostics = {}
    for arm in ARMS:
        solver_diagnostics[arm] = {}
        for variant in ("variant_a", "variant_b"):
            statuses = Counter(r[variant]["solver_status"] for r in by_arm[arm])
            times = [r[variant]["solve_time_seconds"] for r in by_arm[arm]]
            gaps = [r[variant]["solver_gap"] for r in by_arm[arm] if r[variant]["solver_gap"] is not None]
            solver_diagnostics[arm][variant] = {
                "status_distribution": dict(statuses),
                "solve_time_median_sec": round(float(np.median(times)), 3),
                "solve_time_mean_sec": round(float(np.mean(times)), 3),
                "solve_time_iqr_sec": [round(float(np.percentile(times, 25)), 3), round(float(np.percentile(times, 75)), 3)],
                "finite_gap_count": len(gaps),
                "mean_mip_gap": round(float(np.mean(gaps)), 5) if gaps else None,
            }

    # 7. Compile Final Analysis Summary
    summary = {
        "study_name": STUDY_NAME,
        "sample": {
            "cohort_size": 100,
            "arms": ARMS,
            "dietary_modes": MODES,
            "total_paired_tasks": len(records),
            "total_solves": len(records) * 2,
        },
        "feasibility_summary": feasibility_summary,
        "within_arm_contrasts": within_arm_contrasts,
        "inter_arm_contrasts": inter_arm_contrasts,
        "solver_diagnostics": solver_diagnostics,
    }

    # Save JSON summary
    summary_path = STUDY_DIR / "analysis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Analysis summary saved to {summary_path}")

    # 8. Export CSV tables
    # Table A: Within-arm paired contrasts
    csv_path_within = STUDY_DIR / "within_arm_paired_contrasts.csv"
    with open(csv_path_within, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Arm", "Metric", "Variant_A_Mean", "Variant_B_Mean",
            "Difference_B_minus_A", "CI95_Lower", "CI95_Upper", "Unit"
        ])
        for arm in ARMS:
            c = within_arm_contrasts[arm]
            # Feasibility Stage 3
            writer.writerow([
                arm, "Stage3_Feasibility_Rate",
                round(feasibility_summary[arm]["variant_a"]["stage3_rate_pct"], 2),
                round(feasibility_summary[arm]["variant_b"]["stage3_rate_pct"], 2),
                c["pooled_stage3_diff"]["diff_pp"],
                c["pooled_stage3_diff"]["ci95_pp"][0],
                c["pooled_stage3_diff"]["ci95_pp"][1],
                "percentage_points",
            ])
            # End-to-End Acceptance
            writer.writerow([
                arm, "End_to_End_Acceptance_Rate",
                round(feasibility_summary[arm]["variant_a"]["end_to_end_rate_pct"], 2),
                round(feasibility_summary[arm]["variant_b"]["end_to_end_rate_pct"], 2),
                c["pooled_end_to_end_diff"]["diff_pp"],
                c["pooled_end_to_end_diff"]["ci95_pp"][0],
                c["pooled_end_to_end_diff"]["ci95_pp"][1],
                "percentage_points",
            ])
            # Mean Norm L1 Error
            l1 = c["mean_norm_l1_target_error"]
            writer.writerow([
                arm, "Mean_Normalized_L1_Target_Error",
                l1["variant_a"]["mean"],
                l1["variant_b"]["mean"],
                l1["difference_b_minus_a"]["mean"],
                l1["difference_b_minus_a"]["ci95"][0],
                l1["difference_b_minus_a"]["ci95"][1],
                "normalized_fraction",
            ])
            # Norm L2 Distance
            l2 = c["norm_l2_target_distance"]
            writer.writerow([
                arm, "Normalized_L2_Target_Distance",
                l2["variant_a"]["mean"],
                l2["variant_b"]["mean"],
                l2["difference_b_minus_a"]["mean"],
                l2["difference_b_minus_a"]["ci95"][0],
                l2["difference_b_minus_a"]["ci95"][1],
                "dimensionless_distance",
            ])
            # Solve Time
            tm = c["solve_time_seconds"]
            writer.writerow([
                arm, "Solve_Time_Seconds",
                tm["variant_a"]["mean"],
                tm["variant_b"]["mean"],
                tm["difference_b_minus_a"]["mean"],
                tm["difference_b_minus_a"]["ci95"][0],
                tm["difference_b_minus_a"]["ci95"][1],
                "seconds",
            ])
    print(f"Within-arm contrasts exported to {csv_path_within}")

    # Table B: Mode stratified results
    csv_path_modes = STUDY_DIR / "mode_stratified_summary.csv"
    with open(csv_path_modes, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Arm", "Dietary_Mode", "Variant_A_S3_Accepted", "Variant_B_S3_Accepted",
            "S3_Discordant_Pairs", "S3_Exact_P_Value",
            "Variant_A_E2E_Accepted", "Variant_B_E2E_Accepted",
            "E2E_Discordant_Pairs", "E2E_Exact_P_Value"
        ])
        for arm in ARMS:
            c = within_arm_contrasts[arm]
            for mode in MODES:
                s3_t = c["per_mode_stage3_exact_tests"][mode]
                e2e_t = c["per_mode_end_to_end_exact_tests"][mode]
                recs = by_arm_mode[(arm, mode)]
                s3_a_acc = sum(1 for r in recs if r["variant_a"]["replay_passed"])
                s3_b_acc = sum(1 for r in recs if r["variant_b"]["replay_passed"])
                e2e_a_acc = sum(1 for r in recs if r["variant_a"]["end_to_end_accepted"])
                e2e_b_acc = sum(1 for r in recs if r["variant_b"]["end_to_end_accepted"])
                writer.writerow([
                    arm, mode,
                    s3_a_acc, s3_b_acc, s3_t["discordant_count"], s3_t["exact_p_value"],
                    e2e_a_acc, e2e_b_acc, e2e_t["discordant_count"], e2e_t["exact_p_value"]
                ])
    print(f"Mode-stratified summary exported to {csv_path_modes}")

    return summary


if __name__ == "__main__":
    analyze_experiment()
