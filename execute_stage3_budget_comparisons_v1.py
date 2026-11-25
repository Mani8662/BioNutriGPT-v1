#!/usr/bin/env python3
"""Execute Stage 3 Solver-Budget Matched Comparisons (v1).

Performs matched retrospective comparisons across:
1. Budgets: 5s vs 3s, 10s vs 5s, 10s vs 3s across 3 arms x 2 objectives (with common-success sensitivity).
2. Target sources: MG vs MC, MG vs RO, MC vs RO across 2 objectives x 3 budgets.

Weights each participant equally for both point estimates and 10,000-resample cluster bootstraps.
Evaluates exact mode-level hypothesis tests with Holm step-down family adjustments.
Outputs:
Saved Model/stage3_budget_comparisons_v1/comparison_results.json
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy.stats as stats

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "Saved Model" / "stage3_budget_comparisons_v1"
AUDIT_V3_DIR = ROOT / "Saved Model" / "stage3_budget_audit_v3"
EXP_DIR = ROOT / "Saved Model" / "stage3_budget_experiment_v1"
TARGETS_DIR = EXP_DIR / "stage2_targets"

BOOTSTRAP_SEED = 20261124
BOOTSTRAP_RESAMPLES = 10000

ARMS = ["rules_only", "minimum_change", "model_guided"]
MODES = ["pure_vegetarian", "veg_nonveg_combo", "pure_non_vegetarian"]
VARIANTS = ["variant_a", "variant_b"]
BUDGETS_SEC = [3.0, 5.0, 10.0]


def sha(p: Path | str) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8-sig"))


def save_json(p: Path, data: Any) -> None:
    p.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def exact_paired_binomial_test(wins: int, losses: int) -> float:
    n_disc = wins + losses
    if n_disc == 0:
        return 1.0
    k_min = min(wins, losses)
    return float(stats.binomtest(k_min, n_disc, p=0.5, alternative="two-sided").pvalue)


def holm_adjust(p_values: list[float]) -> list[float]:
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    cum_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = p * (m - rank)
        cum_max = max(cum_max, adj)
        adjusted[orig_idx] = min(1.0, cum_max)
    return adjusted


def participant_cluster_bootstrap(
    diffs_per_participant: list[float],
    n_bootstrap: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    arr = np.asarray(diffs_per_participant, dtype=float)
    pt = float(np.mean(arr))
    n = len(arr)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    resampled_means = np.mean(arr[indices], axis=1)
    ci_low = float(np.percentile(resampled_means, 2.5))
    ci_high = float(np.percentile(resampled_means, 97.5))
    return {
        "n_participants": n,
        "mean_difference": pt,
        "ci95": [ci_low, ci_high],
    }


def compute_matched_contrast(
    key_a_fn: Any,
    key_b_fn: Any,
    idx_map: dict[tuple, dict],
    endpoint_name: str,
    is_binary: bool,
) -> dict[str, Any]:
    """Compute matched participant-level difference, point estimate, and bootstrap CI."""
    part_diffs = []
    part_a_vals = []
    part_b_vals = []
    matched_cell_count = 0

    for i in range(100):
        d_modes = []
        a_modes = []
        b_modes = []
        for m in MODES:
            k_a = key_a_fn(i, m)
            k_b = key_b_fn(i, m)
            ra = idx_map.get(k_a)
            rb = idx_map.get(k_b)
            if ra is None or rb is None:
                continue

            if is_binary:
                val_a = 100.0 if ra[endpoint_name] else 0.0
                val_b = 100.0 if rb[endpoint_name] else 0.0
                d_modes.append(val_a - val_b)
                a_modes.append(val_a)
                b_modes.append(val_b)
                matched_cell_count += 1
            else:
                # Continuous endpoints require BOTH plans to be accepted
                if ra["accepted"] and rb["accepted"]:
                    val_a = float(ra[endpoint_name])
                    val_b = float(rb[endpoint_name])
                    d_modes.append(val_a - val_b)
                    a_modes.append(val_a)
                    b_modes.append(val_b)
                    matched_cell_count += 1

        if d_modes:
            part_diffs.append(float(np.mean(d_modes)))
            part_a_vals.append(float(np.mean(a_modes)))
            part_b_vals.append(float(np.mean(b_modes)))

    if not part_diffs:
        return {
            "n_participants": 0,
            "matched_cells": 0,
            "mean_difference": None,
            "ci95": None,
            "condition_a_mean": None,
            "condition_b_mean": None,
        }

    boot_diff = participant_cluster_bootstrap(part_diffs)
    boot_a = participant_cluster_bootstrap(part_a_vals)
    boot_b = participant_cluster_bootstrap(part_b_vals)

    return {
        "n_participants": len(part_diffs),
        "matched_cells": matched_cell_count,
        "mean_difference": boot_diff["mean_difference"],
        "ci95": boot_diff["ci95"],
        "condition_a_mean": {
            "mean": boot_a["mean_difference"],
            "ci95": boot_a["ci95"],
        },
        "condition_b_mean": {
            "mean": boot_b["mean_difference"],
            "ci95": boot_b["ci95"],
        },
    }


def compute_mode_binary_breakdown(
    key_a_fn: Any,
    key_b_fn: Any,
    idx_map: dict[tuple, dict],
    endpoint_name: str,
) -> dict[str, dict[str, Any]]:
    """Compute per-mode counts, wins, losses, exact binomial test p-value, and pointwise CI."""
    res = {}
    for m in MODES:
        n_total = 100
        a_acc = 0
        b_acc = 0
        wins = 0
        losses = 0
        joint = 0
        diffs = []
        for i in range(100):
            ra = idx_map[key_a_fn(i, m)]
            rb = idx_map[key_b_fn(i, m)]
            v_a = bool(ra[endpoint_name])
            v_b = bool(rb[endpoint_name])
            if v_a:
                a_acc += 1
            if v_b:
                b_acc += 1

            if v_a and not v_b:
                wins += 1
            elif not v_a and v_b:
                losses += 1
            elif v_a and v_b:
                joint += 1

            diffs.append((100.0 if v_a else 0.0) - (100.0 if v_b else 0.0))

        pval = exact_paired_binomial_test(wins, losses)
        boot = participant_cluster_bootstrap(diffs)

        res[m] = {
            "n": n_total,
            "condition_a_accepted": a_acc,
            "condition_b_accepted": b_acc,
            "wins": wins,
            "losses": losses,
            "jointly_accepted": joint,
            "diff_pp": boot["mean_difference"],
            "ci95": boot["ci95"],
            "raw_p_value": pval,
        }
    return res


def main():
    print("=" * 80)
    print("Stage 3 Solver-Budget Matched Comparisons Execution (v1)")
    print("=" * 80)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Audit input manifest
    manifest_path = OUT_DIR / "input_manifest.json"
    assert manifest_path.exists(), f"Missing manifest: {manifest_path}"
    manifest = read_json(manifest_path)
    for rel_path, expected_hash in manifest.items():
        actual_path = ROOT / rel_path
        assert actual_path.exists(), f"Input file missing: {rel_path}"
        actual_hash = sha(actual_path)
        assert actual_hash == expected_hash, f"Hash mismatch for {rel_path}!"
    print(f"Verified all {len(manifest)} input files before execution.")

    # 2. Load recomputed rows and Stage 2 targets
    rows_path = AUDIT_V3_DIR / "recomputed_LOCAL.json"
    rows = read_json(rows_path)
    assert len(rows) == 5400, f"Expected 5,400 rows, got {len(rows)}"

    target_calories_map = {}
    for case_file in TARGETS_DIR.glob("case_*.json"):
        d = read_json(case_file)
        target_calories_map[(d["case"], d["arm"])] = float(d["energy_prescription"]["target_calories"])

    # Augment rows with energy error metrics
    for r in rows:
        t_cal = target_calories_map[(r["case"], r["arm"])]
        r["target_calories"] = t_cal
        if r["accepted"] and r.get("totals"):
            e_del = float(r["totals"]["DR1TKCAL"])
            r["energy_error_kcal"] = abs(e_del - t_cal)
            r["energy_error_pct"] = abs(e_del - t_cal) / t_cal * 100.0
        else:
            r["energy_error_kcal"] = None
            r["energy_error_pct"] = None

    idx_map = {(r["case"], r["arm"], r["mode"], r["budget_seconds"], r["variant"]): r for r in rows}
    assert len(idx_map) == 5400

    # 3. Budget Comparisons
    print("\nComputing Budget Comparisons (5s vs 3s, 10s vs 5s, 10s vs 3s)...")
    budget_comparisons = {}
    budget_exact_p_list = []
    budget_test_refs = []

    budget_pairs = [
        ("5s_vs_3s", 5.0, 3.0, "primary"),
        ("10s_vs_5s", 10.0, 5.0, "primary"),
        ("10s_vs_3s", 10.0, 3.0, "secondary"),
    ]

    for v in VARIANTS:
        budget_comparisons[v] = {}
        for arm in ARMS:
            budget_comparisons[v][arm] = {}
            for pair_name, b_hi, b_lo, status_type in budget_pairs:
                key_a = lambda i, m, b=b_hi: (i, arm, m, b, v)
                key_b = lambda i, m, b=b_lo: (i, arm, m, b, v)

                # Primary & Secondary contrasts
                acc_contrast = compute_matched_contrast(key_a, key_b, idx_map, "accepted", is_binary=True)
                e2e_contrast = compute_matched_contrast(key_a, key_b, idx_map, "e2e", is_binary=True)
                time_contrast = compute_matched_contrast(key_a, key_b, idx_map, "runtime", is_binary=False)
                l1_contrast = compute_matched_contrast(key_a, key_b, idx_map, "l1", is_binary=False)
                e_kcal_contrast = compute_matched_contrast(key_a, key_b, idx_map, "energy_error_kcal", is_binary=False)
                e_pct_contrast = compute_matched_contrast(key_a, key_b, idx_map, "energy_error_pct", is_binary=False)
                mass_contrast = compute_matched_contrast(key_a, key_b, idx_map, "mass_g", is_binary=False)
                serv_contrast = compute_matched_contrast(key_a, key_b, idx_map, "servings", is_binary=False)
                dish_contrast = compute_matched_contrast(key_a, key_b, idx_map, "dish_count", is_binary=False)

                # Mode breakdown for primary endpoint
                mode_acc = compute_mode_binary_breakdown(key_a, key_b, idx_map, "accepted")
                for m, m_data in mode_acc.items():
                    budget_exact_p_list.append(m_data["raw_p_value"])
                    budget_test_refs.append(m_data)

                budget_comparisons[v][arm][pair_name] = {
                    "contrast": pair_name,
                    "status": status_type,
                    "condition_a_budget": b_hi,
                    "condition_b_budget": b_lo,
                    "accepted": acc_contrast,
                    "e2e": e2e_contrast,
                    "runtime_seconds": time_contrast,
                    "l1_target_error": l1_contrast,
                    "energy_error_kcal": e_kcal_contrast,
                    "energy_error_pct": e_pct_contrast,
                    "mass_g": mass_contrast,
                    "servings": serv_contrast,
                    "dish_count": dish_contrast,
                    "dietary_modes": mode_acc,
                }

    # Adjust Family A (Budget comparisons) via Holm step-down
    assert len(budget_exact_p_list) == 54, f"Expected 54 budget exact tests, got {len(budget_exact_p_list)}"
    budget_holm_p = holm_adjust(budget_exact_p_list)
    for m_ref, h_p in zip(budget_test_refs, budget_holm_p):
        m_ref["holm_adjusted_p"] = h_p

    # Common-success sensitivity analysis across budgets
    print("\nComputing Common-Success Sensitivity Analysis across all three budgets...")
    budget_sensitivity = {}
    for v in VARIANTS:
        budget_sensitivity[v] = {}
        for arm in ARMS:
            common_participants = []
            common_cell_count = 0
            diff_5_3_part = []
            diff_10_5_part = []
            diff_10_3_part = []

            l1_3s_part = []
            l1_5s_part = []
            l1_10s_part = []

            time_3s_part = []
            time_5s_part = []
            time_10s_part = []

            for i in range(100):
                modes_3s = []
                modes_5s = []
                modes_10s = []
                t_3s = []
                t_5s = []
                t_10s = []
                for m in MODES:
                    r3 = idx_map[(i, arm, m, 3.0, v)]
                    r5 = idx_map[(i, arm, m, 5.0, v)]
                    r10 = idx_map[(i, arm, m, 10.0, v)]
                    if r3["accepted"] and r5["accepted"] and r10["accepted"]:
                        modes_3s.append(r3["l1"])
                        modes_5s.append(r5["l1"])
                        modes_10s.append(r10["l1"])
                        t_3s.append(r3["runtime"])
                        t_5s.append(r5["runtime"])
                        t_10s.append(r10["runtime"])
                        common_cell_count += 1

                if modes_3s:
                    m3 = float(np.mean(modes_3s))
                    m5 = float(np.mean(modes_5s))
                    m10 = float(np.mean(modes_10s))
                    l1_3s_part.append(m3)
                    l1_5s_part.append(m5)
                    l1_10s_part.append(m10)
                    diff_5_3_part.append(m5 - m3)
                    diff_10_5_part.append(m10 - m5)
                    diff_10_3_part.append(m10 - m3)

                    time_3s_part.append(float(np.mean(t_3s)))
                    time_5s_part.append(float(np.mean(t_5s)))
                    time_10s_part.append(float(np.mean(t_10s)))

            budget_sensitivity[v][arm] = {
                "n_participants_all_three_successful": len(l1_3s_part),
                "total_common_cells": common_cell_count,
                "l1_means": {
                    "3s": participant_cluster_bootstrap(l1_3s_part),
                    "5s": participant_cluster_bootstrap(l1_5s_part),
                    "10s": participant_cluster_bootstrap(l1_10s_part),
                },
                "l1_differences": {
                    "5s_minus_3s": participant_cluster_bootstrap(diff_5_3_part),
                    "10s_minus_5s": participant_cluster_bootstrap(diff_10_5_part),
                    "10s_minus_3s": participant_cluster_bootstrap(diff_10_3_part),
                },
                "runtime_means": {
                    "3s": participant_cluster_bootstrap(time_3s_part),
                    "5s": participant_cluster_bootstrap(time_5s_part),
                    "10s": participant_cluster_bootstrap(time_10s_part),
                },
            }

    # 4. Target-Source Comparisons
    print("\nComputing Target-Source Comparisons (MG vs MC, MG vs RO, MC vs RO)...")
    target_source_comparisons = {}
    target_exact_p_list = []
    target_test_refs = []
    target_e2e_p_list = []
    target_e2e_refs = []

    target_contrasts = [
        ("model_guided_vs_minimum_change", "model_guided", "minimum_change"),
        ("model_guided_vs_rules_only", "model_guided", "rules_only"),
        ("minimum_change_vs_rules_only", "minimum_change", "rules_only"),
    ]

    for v in VARIANTS:
        target_source_comparisons[v] = {}
        for b_sec in BUDGETS_SEC:
            b_key = f"{int(b_sec)}s"
            target_source_comparisons[v][b_key] = {}
            for contrast_name, arm_a, arm_b in target_contrasts:
                key_a = lambda i, m, a=arm_a: (i, a, m, b_sec, v)
                key_b = lambda i, m, b=arm_b: (i, b, m, b_sec, v)

                # Primary endpoints
                acc_contrast = compute_matched_contrast(key_a, key_b, idx_map, "accepted", is_binary=True)
                e2e_contrast = compute_matched_contrast(key_a, key_b, idx_map, "e2e", is_binary=True)
                time_contrast = compute_matched_contrast(key_a, key_b, idx_map, "runtime", is_binary=False)

                # Secondary endpoints
                own_target_l1 = compute_matched_contrast(key_a, key_b, idx_map, "l1", is_binary=False)
                e_kcal_contrast = compute_matched_contrast(key_a, key_b, idx_map, "energy_error_kcal", is_binary=False)
                e_pct_contrast = compute_matched_contrast(key_a, key_b, idx_map, "energy_error_pct", is_binary=False)
                mass_contrast = compute_matched_contrast(key_a, key_b, idx_map, "mass_g", is_binary=False)
                serv_contrast = compute_matched_contrast(key_a, key_b, idx_map, "servings", is_binary=False)
                dish_contrast = compute_matched_contrast(key_a, key_b, idx_map, "dish_count", is_binary=False)

                # Mode breakdown for valid-plan rate
                mode_acc = compute_mode_binary_breakdown(key_a, key_b, idx_map, "accepted")
                for m, m_data in mode_acc.items():
                    target_exact_p_list.append(m_data["raw_p_value"])
                    target_test_refs.append(m_data)

                # Mode breakdown for end-to-end acceptance
                mode_e2e = compute_mode_binary_breakdown(key_a, key_b, idx_map, "e2e")
                for m, m_data in mode_e2e.items():
                    target_e2e_p_list.append(m_data["raw_p_value"])
                    target_e2e_refs.append(m_data)

                # Upstream Stage 2 Gate separation
                s2_acc_a = sum(
                    1 for i in range(100)
                    if any(idx_map[(i, arm_a, m, b_sec, v)]["e2e"] or not idx_map[(i, arm_a, m, b_sec, v)]["accepted"]
                           for m in MODES)
                )
                # Count unique Stage 2 passes for arm A and arm B
                s2_pass_a = sum(1 for i in range(100) if read_json(TARGETS_DIR / f"case_{i:03d}_{idx_map[(i, arm_a, 'pure_vegetarian', b_sec, v)]['SEQN']}_{arm_a}.json")["overall_accepted"])
                s2_pass_b = sum(1 for i in range(100) if read_json(TARGETS_DIR / f"case_{i:03d}_{idx_map[(i, arm_b, 'pure_vegetarian', b_sec, v)]['SEQN']}_{arm_b}.json")["overall_accepted"])

                target_source_comparisons[v][b_key][contrast_name] = {
                    "contrast": contrast_name,
                    "condition_a_arm": arm_a,
                    "condition_b_arm": arm_b,
                    "budget_seconds": b_sec,
                    "accepted": acc_contrast,
                    "e2e": e2e_contrast,
                    "runtime_seconds": time_contrast,
                    "own_target_l1_error": own_target_l1,
                    "energy_error_kcal": e_kcal_contrast,
                    "energy_error_pct": e_pct_contrast,
                    "mass_g": mass_contrast,
                    "servings": serv_contrast,
                    "dish_count": dish_contrast,
                    "stage2_gate_decomposition": {
                        "stage2_passed_a": s2_pass_a,
                        "stage2_passed_b": s2_pass_b,
                        "stage2_gate_diff_pp": (s2_pass_a - s2_pass_b) * 1.0,
                    },
                    "dietary_modes_accepted": mode_acc,
                    "dietary_modes_e2e": mode_e2e,
                }

    # Adjust Family B (Target-source valid-plan tests) via Holm step-down
    assert len(target_exact_p_list) == 54, f"Expected 54 target valid-plan exact tests, got {len(target_exact_p_list)}"
    target_holm_p = holm_adjust(target_exact_p_list)
    for m_ref, h_p in zip(target_test_refs, target_holm_p):
        m_ref["holm_adjusted_p"] = h_p

    # Adjust Family C (Target-source end-to-end tests) via Holm step-down
    assert len(target_e2e_p_list) == 54, f"Expected 54 target e2e exact tests, got {len(target_e2e_p_list)}"
    target_e2e_holm_p = holm_adjust(target_e2e_p_list)
    for m_ref, h_p in zip(target_e2e_refs, target_e2e_holm_p):
        m_ref["holm_adjusted_p"] = h_p

    # 5. Assemble final comparison payload
    results_payload = {
        "study_name": "stage3_budget_matched_comparisons_v1",
        "created_at": "2026-11-24T09:55:00+0530",
        "provenance_disclosure": {
            "type": "retrospective_exploratory_analysis",
            "description": "Planned post-hoc after inspecting earlier budget audit results. The cohort is reused for additional endpoints. Exploratory matched comparisons, not a prospectively registered confirmatory trial.",
            "cohort_size": 100,
            "total_records_analyzed": len(rows),
        },
        "budget_comparisons": budget_comparisons,
        "budget_sensitivity_common_success": budget_sensitivity,
        "target_source_comparisons": target_source_comparisons,
        "multiplicity_family_sizes": {
            "family_a_budget": len(budget_exact_p_list),
            "family_b_target_source_accepted": len(target_exact_p_list),
            "family_c_target_source_e2e": len(target_e2e_p_list),
        },
    }

    out_file = OUT_DIR / "comparison_results.json"
    save_json(out_file, results_payload)
    print(f"\nSaved comparison_results.json to {out_file}.")

    # 6. Re-verify input manifest hashes
    for rel_path, expected_hash in manifest.items():
        actual_hash = sha(ROOT / rel_path)
        assert actual_hash == expected_hash, f"Input modified during execution: {rel_path}"
    print("Verified input manifest after execution (100% byte-for-byte identical).")

    print("\n" + "=" * 80)
    print("STAGE 3 BUDGET MATCHED COMPARISONS EXECUTED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    main()
