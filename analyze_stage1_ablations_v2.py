#!/usr/bin/env python3
"""Stage 1 Architecture Ablation Statistical Analysis (v2).

Computes:
1. Seed-level metrics, means, and standard deviations for all 5 variants across 5 seeds.
2. Paired seed contrasts: FULL minus each ablation.
3. Two-axis paired bootstrap (20,000 replicates across 7,004 participants and 5 seeds):
   - Pointwise 95% percentile intervals
   - Bonferroni-adjusted family-wise percentile intervals for 16 primary contrasts
4. Generates ASCII forest plot, summary table, and comprehensive results report.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Data/stage1_repaired_v2"
PROTOCOL_PATH = ROOT / "stage1_ablation_protocol_v2.json"
RESULTS_DIR = ROOT / "Saved Model/stage1_ablation_v2"
REPORT_MD = ROOT / "Research Paper/Stage1_Ablation_v2_Results.md"

VARIANTS = [
    "full",
    "no_moe",
    "mean_pooling",
    "no_modality_embeddings",
    "no_head_shortcuts",
]
ABLATIONS = [
    "no_moe",
    "mean_pooling",
    "no_modality_embeddings",
    "no_head_shortcuts",
]
SEEDS = [42, 123, 456, 789, 2026]
OUTCOMES = ["TARGET_DIABETES", "TARGET_CVD", "TARGET_OBESITY"]
PRIMARY_OUTCOMES = ["TARGET_DIABETES", "TARGET_CVD"]


def load_all_predictions():
    truth_df = pd.read_csv(DATA / "temporal_ml_ready.csv").sort_values("SEQN")
    y_true = {o: truth_df[o].to_numpy(dtype=float) for o in OUTCOMES}
    seqns = truth_df["SEQN"].to_numpy(dtype=np.int64)

    preds_raw: dict[str, dict[int, dict[str, np.ndarray]]] = {v: {} for v in VARIANTS}
    preds_cal: dict[str, dict[int, dict[str, np.ndarray]]] = {v: {} for v in VARIANTS}
    completed_records: dict[str, dict[int, dict[str, Any]]] = {v: {} for v in VARIANTS}

    for v in VARIANTS:
        for s in SEEDS:
            fpath = RESULTS_DIR / v / f"seed_{s}" / "temporal_predictions_LOCAL.csv"
            cpath = RESULTS_DIR / v / f"seed_{s}" / "completed.json"
            if not fpath.exists() or not cpath.exists():
                raise FileNotFoundError(f"Missing run artifacts for variant '{v}' seed {s}: {fpath}")

            rec = json.loads(cpath.read_text(encoding="utf-8"))
            completed_records[v][s] = rec

            df = pd.read_csv(fpath).sort_values("SEQN")
            np.testing.assert_array_equal(df["SEQN"].to_numpy(), seqns)

            preds_raw[v][s] = {}
            preds_cal[v][s] = {}
            for o in OUTCOMES:
                p_r = df[f"{o}_raw"].to_numpy()
                p_c = df[f"{o}_calibrated"].to_numpy()
                assert np.isfinite(p_r).all() and ((p_r >= 0) & (p_r <= 1)).all()
                assert np.isfinite(p_c).all() and ((p_c >= 0) & (p_c <= 1)).all()
                preds_raw[v][s][o] = np.clip(p_r, 1e-7, 1.0 - 1e-7)
                preds_cal[v][s][o] = np.clip(p_c, 1e-7, 1.0 - 1e-7)

    return y_true, seqns, preds_raw, preds_cal, completed_records


def fast_auc(pos_bincount: np.ndarray, neg_bincount: np.ndarray) -> float:
    denom = pos_bincount.sum() * neg_bincount.sum()
    if denom == 0:
        return 0.5
    return float(np.dot(pos_bincount, np.cumsum(neg_bincount) - 0.5 * neg_bincount) / denom)


def run_bootstrap_analysis(
    y_true: dict[str, np.ndarray],
    preds_raw: dict[str, dict[int, dict[str, np.ndarray]]],
    preds_cal: dict[str, dict[int, dict[str, np.ndarray]]],
    n_replicates: int = 20000,
    seed: int = 20261119,
) -> dict[str, Any]:
    N = len(y_true["TARGET_DIABETES"])
    K = len(SEEDS)

    # Pre-rank raw probabilities for fast O(N) AUC computation in bootstrap loops
    ranks_raw: dict[str, dict[int, dict[str, tuple[np.ndarray, int]]]] = {v: {} for v in VARIANTS}
    for v in VARIANTS:
        for s in SEEDS:
            ranks_raw[v][s] = {}
            for o in OUTCOMES:
                p = preds_raw[v][s][o]
                _, rk = np.unique(p, return_inverse=True)
                ranks_raw[v][s][o] = (rk, int(rk.max() + 1))

    # Precompute squared errors for Brier calculation
    sq_err_cal: dict[str, dict[int, dict[str, np.ndarray]]] = {v: {} for v in VARIANTS}
    for v in VARIANTS:
        for s in SEEDS:
            sq_err_cal[v][s] = {}
            for o in OUTCOMES:
                y = y_true[o]
                p = preds_cal[v][s][o]
                sq_err_cal[v][s][o] = (p - y) ** 2

    # Primary contrasts: (ablation, outcome, metric_type)
    # metric_type: 'raw_auroc' or 'cal_brier'
    # 4 ablations x 2 primary outcomes x 2 metrics = 16 primary contrasts
    contrasts = []
    for ab in ABLATIONS:
        for o in PRIMARY_OUTCOMES:
            contrasts.append((ab, o, "raw_auroc"))
            contrasts.append((ab, o, "cal_brier"))

    # Also compute exploratory obesity contrasts
    for ab in ABLATIONS:
        contrasts.append((ab, "TARGET_OBESITY", "raw_auroc"))
        contrasts.append((ab, "TARGET_OBESITY", "cal_brier"))

    contrast_diffs: dict[tuple[str, str, str], list[float]] = {c: [] for c in contrasts}

    print(f"Running two-axis paired bootstrap ({n_replicates:,} replicates across {N:,} participants & {K} seeds)...", flush=True)
    t0 = time.perf_counter()

    for rep in range(n_replicates):
        rng = np.random.default_rng(np.random.SeedSequence([seed, rep]))
        # 1. Sample participants with replacement
        part_idx = rng.integers(N, size=N)
        weights = np.bincount(part_idx, minlength=N)
        # 2. Sample seed indices with replacement
        sampled_seed_indices = rng.integers(K, size=K)
        sampled_seeds = [SEEDS[idx] for idx in sampled_seed_indices]

        # Compute metric for FULL averaged across sampled seeds
        full_metrics: dict[tuple[str, str], float] = {}
        for o in OUTCOMES:
            y = y_true[o]
            w_y = weights * y
            w_not_y = weights * (1.0 - y)

            # AUROC
            auc_vals = []
            for s in sampled_seeds:
                rk, n_rk = ranks_raw["full"][s][o]
                pos_bins = np.bincount(rk, weights=w_y, minlength=n_rk)
                neg_bins = np.bincount(rk, weights=w_not_y, minlength=n_rk)
                auc_vals.append(fast_auc(pos_bins, neg_bins))
            full_metrics[(o, "raw_auroc")] = float(np.mean(auc_vals))

            # Brier
            brier_vals = [
                float(np.dot(weights, sq_err_cal["full"][s][o]) / N)
                for s in sampled_seeds
            ]
            full_metrics[(o, "cal_brier")] = float(np.mean(brier_vals))

        # Compute metrics for each ablation and calculate contrast (FULL - ablation)
        for ab in ABLATIONS:
            for o in OUTCOMES:
                y = y_true[o]
                w_y = weights * y
                w_not_y = weights * (1.0 - y)

                # AUROC
                auc_vals = []
                for s in sampled_seeds:
                    rk, n_rk = ranks_raw[ab][s][o]
                    pos_bins = np.bincount(rk, weights=w_y, minlength=n_rk)
                    neg_bins = np.bincount(rk, weights=w_not_y, minlength=n_rk)
                    auc_vals.append(fast_auc(pos_bins, neg_bins))
                ab_auc = float(np.mean(auc_vals))
                diff_auc = full_metrics[(o, "raw_auroc")] - ab_auc
                contrast_diffs[(ab, o, "raw_auroc")].append(diff_auc)

                # Brier
                brier_vals = [
                    float(np.dot(weights, sq_err_cal[ab][s][o]) / N)
                    for s in sampled_seeds
                ]
                ab_brier = float(np.mean(brier_vals))
                diff_brier = full_metrics[(o, "cal_brier")] - ab_brier
                contrast_diffs[(ab, o, "cal_brier")].append(diff_brier)

    elapsed = time.perf_counter() - t0
    print(f"Bootstrap completed in {elapsed:.1f}s.", flush=True)

    # Calculate intervals: Pointwise 95% [2.5%, 97.5%]
    # Bonferroni adjusted for 16 primary comparisons: alpha = 0.05 / 16 = 0.003125
    # Two-sided quantiles: [alpha/2, 1 - alpha/2] = [0.0015625, 0.9984375]
    alpha_bonf = 0.05 / 16.0
    q_low_bonf = alpha_bonf / 2.0
    q_high_bonf = 1.0 - q_low_bonf

    summary_contrasts = {}
    for c, diff_arr in contrast_diffs.items():
        arr = np.array(diff_arr)
        ab, o, m = c
        is_primary = (ab in ABLATIONS) and (o in PRIMARY_OUTCOMES)
        summary_contrasts[f"{ab}::{o}::{m}"] = {
            "ablation": ab,
            "outcome": o,
            "metric": m,
            "is_primary": is_primary,
            "mean_difference": float(arr.mean()),
            "median_difference": float(np.median(arr)),
            "std_difference": float(arr.std()),
            "ci95_pointwise": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))],
            "ci95_bonferroni_16": [
                float(np.percentile(arr, q_low_bonf * 100.0)),
                float(np.percentile(arr, q_high_bonf * 100.0)),
            ] if is_primary else None,
            "favors_full": bool(
                (m == "raw_auroc" and float(np.percentile(arr, 2.5)) > 0) or
                (m == "cal_brier" and float(np.percentile(arr, 97.5)) < 0)
            ),
            "excludes_zero_pointwise": bool(
                np.percentile(arr, 2.5) > 0 or np.percentile(arr, 97.5) < 0
            ),
            "excludes_zero_bonferroni": bool(
                is_primary and (
                    np.percentile(arr, q_low_bonf * 100.0) > 0 or
                    np.percentile(arr, q_high_bonf * 100.0) < 0
                )
            ),
        }

    return summary_contrasts


def generate_ascii_forest_plot(summary_contrasts: dict[str, Any]) -> str:
    """Generate an ASCII forest plot representing point estimates and 95% CIs."""
    lines = [
        "=========================================================================================",
        "STAGE 1 ABLATION FOREST PLOT: FULL MINUS ABLATION CONTRASTS",
        "Pointwise 95% CI: [====o====] | Bonferroni 95% CI: (....[====o====]....) | Zero: |",
        "For AUROC: Positive favors FULL (+) | For Brier: Negative favors FULL (-)",
        "=========================================================================================",
    ]

    for key, c in summary_contrasts.items():
        if not c["is_primary"]:
            continue
        ab = c["ablation"].replace("_", " ").upper()
        o = c["outcome"].replace("TARGET_", "")
        m = "AUROC" if c["metric"] == "raw_auroc" else "Brier"
        est = c["mean_difference"]
        lo, hi = c["ci95_pointwise"]
        blo, bhi = c["ci95_bonferroni_16"]

        # Scale mapping into a 40-character bar spanning -0.02 to +0.02 (or appropriate range)
        scale = 0.025 if m == "AUROC" else 0.005
        width = 41
        zero_pos = width // 2

        def pos(val):
            p = int(round(zero_pos + (val / scale) * (width // 2)))
            return max(0, min(width - 1, p))

        bar = [" "] * width
        bar[zero_pos] = "|"
        p_lo, p_hi, p_est = pos(lo), pos(hi), pos(est)
        p_blo, p_bhi = pos(blo), pos(bhi)

        for i in range(min(p_blo, p_bhi), max(p_blo, p_bhi) + 1):
            if bar[i] == " ":
                bar[i] = "."
        for i in range(min(p_lo, p_hi), max(p_lo, p_hi) + 1):
            bar[i] = "="
        bar[p_est] = "o"

        bar_str = "".join(bar)
        star = " *" if c["excludes_zero_bonferroni"] else (" †" if c["excludes_zero_pointwise"] else "  ")
        lines.append(
            f"{ab:<22} {o:<8} {m:<6} {est:>+7.4f} [{lo:>+7.4f}, {hi:>+7.4f}] {bar_str}{star}"
        )

    lines.append("=========================================================================================")
    lines.append("Key: † = Pointwise 95% CI excludes zero | * = Bonferroni 16-contrast CI excludes zero")
    return "\n".join(lines)


def run_full_analysis():
    print("Loading Stage 1 ablation predictions across 5 variants and 5 seeds...", flush=True)
    y_true, seqns, preds_raw, preds_cal, completed_records = load_all_predictions()

    # Compute summary tables per variant and seed
    variant_stats = {}
    for v in VARIANTS:
        variant_stats[v] = {
            "parameter_count": completed_records[v][42]["parameter_count"],
            "seeds": {},
            "means": {},
            "stds": {},
        }
        for s in SEEDS:
            rec = completed_records[v][s]
            raw_by_out = {m["outcome"]: m for m in rec["metrics"] if m["prediction_variant"] == "raw"}
            cal_by_out = {m["outcome"]: m for m in rec["metrics"] if m["prediction_variant"] == "calibrated"}
            variant_stats[v]["seeds"][s] = {
                "best_epoch": rec["best_epoch"],
                "training_seconds": rec["training_seconds"],
                "diabetes_raw_auroc": raw_by_out["TARGET_DIABETES"]["auroc"],
                "diabetes_cal_brier": cal_by_out["TARGET_DIABETES"]["brier"],
                "cvd_raw_auroc": raw_by_out["TARGET_CVD"]["auroc"],
                "cvd_cal_brier": cal_by_out["TARGET_CVD"]["brier"],
                "obesity_raw_auroc": raw_by_out["TARGET_OBESITY"]["auroc"],
                "obesity_cal_brier": cal_by_out["TARGET_OBESITY"]["brier"],
            }

        # Calculate cross-seed means and standard deviations
        keys = [
            "diabetes_raw_auroc", "diabetes_cal_brier",
            "cvd_raw_auroc", "cvd_cal_brier",
            "obesity_raw_auroc", "obesity_cal_brier",
            "training_seconds",
        ]
        for k in keys:
            vals = [variant_stats[v]["seeds"][s][k] for s in SEEDS]
            variant_stats[v]["means"][k] = float(np.mean(vals))
            variant_stats[v]["stds"][k] = float(np.std(vals, ddof=1))

    # Run paired 20,000 bootstrap
    bootstrap_results = run_bootstrap_analysis(y_true, preds_raw, preds_cal, n_replicates=20000)
    forest_plot_text = generate_ascii_forest_plot(bootstrap_results)

    # Save summary JSON
    out_json = RESULTS_DIR / "stage1_ablation_summary.json"
    full_summary = {
        "benchmark_version": "stage1-ablation-v2",
        "seeds": SEEDS,
        "variants": VARIANTS,
        "variant_stats": variant_stats,
        "bootstrap_contrasts": bootstrap_results,
    }
    out_json.write_text(json.dumps(full_summary, indent=2), encoding="utf-8")
    print(f"Summary JSON saved: {out_json}")

    # Generate Markdown Report
    md_lines = [
        "# BioNutriGPT Stage 1 Architecture Ablation Study (v2) — Results Report",
        "",
        "## 1. Study Purpose and Methodological Scope",
        "This controlled five-seed architecture-ablation study evaluates whether BioNutriGPT's additional architectural components provide measurable empirical benefits over simpler formulations on the repaired Stage 1 foundation (16,894 training, 3,621 validation, 3,621 calibration, 7,004 temporal evaluation participants).",
        "",
        "The four targeted components are:",
        "1. **Mixture of Experts (MoE)**: 4-expert sparse MoE (top-2 routing) vs. single dense feed-forward network (512 -> 256 -> 256; balancing loss = 0).",
        "2. **Modality Attention Pooling**: learned attention query vectors per modality vs. ordinary unweighted group mean pooling.",
        "3. **Modality Embeddings**: learned modality vectors added during tokenization vs. feature-specific tokenization without modality vectors.",
        "4. **Disease Head Shortcuts**: multi-modality context inputs into disease heads vs. heads receiving only the 256-d patient state representation.",
        "",
        "All models use identical 87 features, identical AdamW hyperparameters (lr=0.0003, weight_decay=0.001, batch_size=64, patience=7), identical row shuffle sequences per seed, identical modular initializations where modules are shared, unpenalized logistic recalibration fitted strictly on calibration participants, and evaluation on the frozen temporal cohort (N=7,004).",
        "",
        "## 2. Performance Summary Across Five Fixed Seeds (Mean ± SD)",
        "",
        "| Architecture Variant | Trainable Parameters | Diabetes Raw AUROC | CVD Raw AUROC | Diabetes Cal. Brier | CVD Cal. Brier | Obesity Raw AUROC | Mean Train Time |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for v in VARIANTS:
        st = variant_stats[v]
        p_count = st["parameter_count"]
        d_auc = f"{st['means']['diabetes_raw_auroc']:.4f} ± {st['stds']['diabetes_raw_auroc']:.4f}"
        c_auc = f"{st['means']['cvd_raw_auroc']:.4f} ± {st['stds']['cvd_raw_auroc']:.4f}"
        d_br = f"{st['means']['diabetes_cal_brier']:.6f} ± {st['stds']['diabetes_cal_brier']:.6f}"
        c_br = f"{st['means']['cvd_cal_brier']:.6f} ± {st['stds']['cvd_cal_brier']:.6f}"
        o_auc = f"{st['means']['obesity_raw_auroc']:.4f} ± {st['stds']['obesity_raw_auroc']:.4f}"
        t_sec = f"{st['means']['training_seconds']:.1f}s"
        md_lines.append(
            f"| **{v.upper()}** | {p_count:,} | {d_auc} | {c_auc} | {d_br} | {c_br} | {o_auc} | {t_sec} |"
        )

    md_lines.extend([
        "",
        "## 3. Two-Axis Paired Bootstrap Analysis (20,000 Replicates)",
        "Contrasts represent **FULL minus Ablation**. Positive AUROC differences favor FULL; negative Brier differences favor FULL. Intervals are constructed using two-axis resampling across 7,004 participants and 5 training seeds.",
        "",
        "| Ablation | Outcome | Metric | Mean Diff | Pointwise 95% CI | Bonferroni 16-Contrast 95% CI | Inference |",
        "|---|---|---|---:|---:|---:|---|",
    ])

    for key, c in bootstrap_results.items():
        if not c["is_primary"]:
            continue
        ab = c["ablation"].upper()
        o = c["outcome"].replace("TARGET_", "")
        m = "Raw AUROC" if c["metric"] == "raw_auroc" else "Calibrated Brier"
        est = c["mean_difference"]
        lo, hi = c["ci95_pointwise"]
        blo, bhi = c["ci95_bonferroni_16"]
        if c["excludes_zero_bonferroni"]:
            inf = "**Statistically robust difference**"
        elif c["excludes_zero_pointwise"]:
            inf = "Nominal difference (not robust after correction)"
        else:
            inf = "Indistinguishable (CI spans zero)"
        md_lines.append(
            f"| {ab} | {o} | {m} | {est:>+7.4f} | [{lo:>+7.4f}, {hi:>+7.4f}] | [{blo:>+7.4f}, {bhi:>+7.4f}] | {inf} |"
        )

    md_lines.extend([
        "",
        "## 4. Primary Contrasts Forest Plot",
        "```",
        forest_plot_text,
        "```",
        "",
        "## 5. Architectural Findings & Scientific Synthesis",
        "1. **Mixture of Experts (MoE)**: Removing MoE reduces trainable parameter count by 46% (from 1,295,623 to 700,675). The empirical difference between FULL and NO_MOE reveals whether sparse top-2 routing and extra capacity justify their computational cost.",
        "2. **Modality Attention Pooling**: Replacing learned query vectors with group mean pooling tests whether learned attention weighting over heterogeneous tokens provides a measurable advantage over simple linear aggregation.",
        "3. **Modality Embeddings**: Testing whether explicit modality tags are required for the tabular Transformer to recognize feature groupings.",
        "4. **Head Shortcuts**: Testing whether direct multi-modality residual shortcuts into specific disease heads outperform passing only the unified patient representation.",
        "",
        "## 6. Limitations and Clinical Scope",
        "- All evaluations are conducted on NHANES cross-sectional data; they do not establish causal disease prevention or Indian clinical effectiveness.",
        "- Five random seeds capture limited training stochasticity; confidence intervals are conditional on the fixed training recipe and modest parameter budget.",
        "- An interval containing zero indicates that the current empirical study did not detect an advantage; it does not prove exact architectural equivalence.",
    ])

    REPORT_MD.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Results report written: {REPORT_MD}")
    print("\n" + forest_plot_text)


if __name__ == "__main__":
    run_full_analysis()
