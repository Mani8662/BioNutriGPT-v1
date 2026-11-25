"""Statistical Analysis of Stage 2 Three-Arm Comparative Added-Value Study.

Recomputes paired participant-level statistics, bootstrap confidence intervals,
nutrient shifts, and summary tables from saved case checkpoints.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
STUDY_DIR = ROOT / "Saved Model/stage2_comparison_study_v1"
CKPT_DIR = STUDY_DIR / "checkpoints"


def load_all_checkpoints(n_cases: int = 100) -> dict[str, list[dict[str, Any]]]:
    arms = ["rules_only", "minimum_change", "model_guided"]
    cases_json = json.loads((STUDY_DIR / "cases_LOCAL.json").read_text(encoding="utf-8"))
    data_by_arm: dict[str, list[dict[str, Any]]] = {a: [] for a in arms}

    for case in cases_json:
        idx = case["case"]
        seqn = case["SEQN"]
        for arm in arms:
            p = CKPT_DIR / f"case_{idx:03d}_{seqn}_{arm}.json"
            if not p.exists():
                raise FileNotFoundError(f"Missing checkpoint: {p}")
            d = json.loads(p.read_text(encoding="utf-8"))
            data_by_arm[arm].append(d)

    return data_by_arm


def paired_bootstrap_contrast(
    arr_c: np.ndarray,
    arr_b: np.ndarray,
    n_resamples: int = 10000,
    seed: int = 20261123,
) -> dict[str, float]:
    """Paired participant bootstrap difference (arr_c - arr_b)."""
    assert len(arr_c) == len(arr_b)
    n = len(arr_c)
    diff = arr_c - arr_b
    point_diff = float(np.mean(diff))

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_resamples, n))
    boot_diffs = np.mean(diff[indices], axis=1)

    ci_low = float(np.percentile(boot_diffs, 2.5))
    ci_high = float(np.percentile(boot_diffs, 97.5))
    se = float(np.std(boot_diffs, ddof=1))
    p_val = float(2.0 * min(np.mean(boot_diffs <= 0), np.mean(boot_diffs >= 0)))

    return {
        "point_diff": point_diff,
        "point_diff_pct": point_diff * 100.0,
        "ci_lower_pct": ci_low * 100.0,
        "ci_upper_pct": ci_high * 100.0,
        "std_error_pct": se * 100.0,
        "two_sided_p": min(1.0, p_val),
    }


def analyze_study():
    print("Loading checkpoints...")
    data = load_all_checkpoints(100)
    arms = ["rules_only", "minimum_change", "model_guided"]
    n = 100

    arm_stats = {}
    for arm in arms:
        records = data[arm]
        acc = np.array([1.0 if r["overall_accepted"] else 0.0 for r in records])
        num = np.array([1.0 if r["numerical_pass"] else 0.0 for r in records])
        score = np.array([1.0 if r["score_gate_pass"] else 0.0 for r in records])
        dists = np.array([r["distance_from_observed_normalized"] for r in records if r["distance_from_observed_normalized"] is not None])
        times = np.array([r["runtime_seconds"] for r in records])

        fallbacks = sum(1 for r in records if r.get("solver_audit", {}).get("projection_audit", {}).get("fallback_used"))

        # Risk score changes
        dd = np.array([r["score_deltas"]["diabetes"] for r in records if r.get("score_deltas")])
        do = np.array([r["score_deltas"]["obesity"] for r in records if r.get("score_deltas")])
        dc = np.array([r["score_deltas"]["cvd"] for r in records if r.get("score_deltas")])

        arm_stats[arm] = {
            "attempted": n,
            "completed": sum(1 for r in records if r["success"]),
            "errors": sum(1 for r in records if not r["success"]),
            "overall_accepted": int(np.sum(acc)),
            "overall_accepted_pct": float(np.mean(acc) * 100.0),
            "numerical_pass": int(np.sum(num)),
            "numerical_pass_pct": float(np.mean(num) * 100.0),
            "score_gate_pass": int(np.sum(score)),
            "score_gate_pass_pct": float(np.mean(score) * 100.0),
            "lp_fallbacks": fallbacks,
            "distance_from_observed": {
                "median": float(np.median(dists)),
                "iqr": float(np.percentile(dists, 75) - np.percentile(dists, 25)),
                "mean": float(np.mean(dists)),
                "std": float(np.std(dists)),
                "min": float(np.min(dists)),
                "max": float(np.max(dists)),
            },
            "runtime_seconds": {
                "total": float(np.sum(times)),
                "median": float(np.median(times)),
                "p95": float(np.percentile(times, 95)),
            },
            "score_deltas": {
                "diabetes_mean": float(np.mean(dd)),
                "obesity_mean": float(np.mean(do)),
                "cvd_mean": float(np.mean(dc)),
            },
            "_raw_acc": acc,
            "_raw_num": num,
            "_raw_score": score,
            "_raw_dist": dists,
        }

    # Paired Contrasts
    contrasts = [
        ("model_guided", "minimum_change", "Primary: Model-Guided (C) vs Minimum-Change (B)", "primary"),
        ("model_guided", "rules_only", "Secondary: Model-Guided (C) vs Rules-Only (A)", "secondary"),
        ("minimum_change", "rules_only", "Secondary: Minimum-Change (B) vs Rules-Only (A)", "secondary"),
    ]

    contrast_results = []
    for c_arm, b_arm, label, role in contrasts:
        # 1. Overall Acceptance
        res_acc = paired_bootstrap_contrast(arm_stats[c_arm]["_raw_acc"], arm_stats[b_arm]["_raw_acc"])
        # 2. Numerical Feasibility
        res_num = paired_bootstrap_contrast(arm_stats[c_arm]["_raw_num"], arm_stats[b_arm]["_raw_num"])
        # 3. Score Gate Pass
        res_score = paired_bootstrap_contrast(arm_stats[c_arm]["_raw_score"], arm_stats[b_arm]["_raw_score"])
        # 4. Distance difference (C - B)
        dist_diff = arm_stats[c_arm]["_raw_dist"] - arm_stats[b_arm]["_raw_dist"]
        res_dist = {
            "point_diff": float(np.mean(dist_diff)),
            "ci_lower": float(np.percentile(dist_diff, 2.5)),
            "ci_upper": float(np.percentile(dist_diff, 97.5)),
        }

        contrast_results.append({
            "contrast_label": label,
            "role": role,
            "arm_1": c_arm,
            "arm_2": b_arm,
            "acceptance": res_acc,
            "numerical_feasibility": res_num,
            "score_gate": res_score,
            "distance_difference": res_dist,
        })

    # Major nutrient shifts across arms
    nutrients = ["KCAL", "PROT", "CARB", "TFAT", "SODI", "POTA"]
    nutrient_shifts = {}
    for nut in nutrients:
        nutrient_shifts[nut] = {}
        for arm in arms:
            deltas = [r["major_nutrient_changes"][nut]["delta"] for r in data[arm] if r.get("major_nutrient_changes")]
            obs = [r["major_nutrient_changes"][nut]["observed"] for r in data[arm] if r.get("major_nutrient_changes")]
            target = [r["major_nutrient_changes"][nut]["target"] for r in data[arm] if r.get("major_nutrient_changes")]
            unit = data[arm][0]["major_nutrient_changes"][nut]["unit"]
            nutrient_shifts[nut][arm] = {
                "unit": unit,
                "observed_mean": float(np.mean(obs)),
                "target_mean": float(np.mean(target)),
                "delta_mean": float(np.mean(deltas)),
                "delta_median": float(np.median(deltas)),
                "delta_iqr": float(np.percentile(deltas, 75) - np.percentile(deltas, 25)),
            }

    # Strip raw arrays for json
    cleaned_arm_stats = {}
    for arm, s in arm_stats.items():
        c = dict(s)
        c.pop("_raw_acc")
        c.pop("_raw_num")
        c.pop("_raw_score")
        c.pop("_raw_dist")
        cleaned_arm_stats[arm] = c

    analysis_summary = {
        "study": "stage2_comparison_study_v1",
        "sample_size": n,
        "bootstrap_resamples": 10000,
        "seed": 20261123,
        "arm_statistics": cleaned_arm_stats,
        "paired_contrasts": contrast_results,
        "major_nutrient_shifts": nutrient_shifts,
    }

    out_path = STUDY_DIR / "analysis_summary.json"
    out_path.write_text(json.dumps(analysis_summary, indent=2), encoding="utf-8")
    print(f"Saved analysis to {out_path}")

    # Write CSV for easy paper reporting
    csv_path = STUDY_DIR / "paired_contrasts.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Contrast", "Role", "Endpoint", "Point Diff (pp)", "95% CI Lower", "95% CI Upper", "SE (pp)", "p-value"])
        for c in contrast_results:
            acc = c["acceptance"]
            writer.writerow([c["contrast_label"], c["role"], "All-Check Acceptance", f"{acc['point_diff_pct']:+.2f}", f"{acc['ci_lower_pct']:+.2f}", f"{acc['ci_upper_pct']:+.2f}", f"{acc['std_error_pct']:.2f}", f"{acc['two_sided_p']:.4f}"])
            num = c["numerical_feasibility"]
            writer.writerow([c["contrast_label"], c["role"], "Numerical Feasibility", f"{num['point_diff_pct']:+.2f}", f"{num['ci_lower_pct']:+.2f}", f"{num['ci_upper_pct']:+.2f}", f"{num['std_error_pct']:.2f}", f"{num['two_sided_p']:.4f}"])
            scr = c["score_gate"]
            writer.writerow([c["contrast_label"], c["role"], "Score-Gate Pass", f"{scr['point_diff_pct']:+.2f}", f"{scr['ci_lower_pct']:+.2f}", f"{scr['ci_upper_pct']:+.2f}", f"{scr['std_error_pct']:.2f}", f"{scr['two_sided_p']:.4f}"])

    print(f"Saved contrasts CSV to {csv_path}")


if __name__ == "__main__":
    analyze_study()
