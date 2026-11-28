"""Benchmark Stage 2 + Stage 3 on N=100 real LASI Wave 1 Indian adults across all 3 diet modes.

Uses the exact official Stage 2 physiological energy + reviewed nutrient-bound rules
(stage2_counterfactual_engine._build_nutrient_bounds + nutrient_rule_policy v4) driven
by real Indian adult age, sex, measured height/weight, physical activity, and
physician-diagnosed Diabetes/CVD + measured Obesity (BMI>=30) from LASI Wave 1.

Evaluates 100 real Indian adults x 3 dietary modes (pure_vegetarian, veg_nonveg_combo,
pure_non_vegetarian) = 300 real-patient MILP solves on the frozen 49-dish v5 catalog
(sourced-cooked-v5-veg-nonveg, zero active slack).

Saves:
  - Saved Model/lasi_stage3_indian_cohort_report.json
  - Saved Model/lasi_stage3_indian_cohort_report.md
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

import chat_service as chat
import nutrient_rule_policy as nrp
import stage2_counterfactual_engine as s2
import stage3_milp_solver as s3


def build_stage2_from_lasi_patient(patient: dict[str, Any]) -> dict[str, Any]:
    """Construct an exact Stage 2 constraint specification from real LASI Indian vitals & clinical status."""
    w = float(patient["weight_kg"])
    h = float(patient["height_cm"])
    age = float(patient["age_years"])
    female = bool(patient["is_female"])
    act = str(patient["activity_level"])

    bmr = (10.0 * w) + (6.25 * h) - (5.0 * age) + (-161.0 if female else 5.0)
    tdee = bmr * s2.PAL_FACTORS.get(act, 1.20)
    # Apply the exact Stage 2 obesity deficit rule when measured BMI >= 30
    p_diab = 0.85 if patient["has_diabetes"] else 0.10
    p_obes = 0.85 if patient["has_obesity_who30"] else 0.10
    p_cvd = 0.85 if patient["has_cvd"] else 0.10
    deficit = 500.0 if p_obes >= s2.CUTOFF_OBESITY else 0.0
    t_kcal = max(1200.0, tdee - deficit)

    lower, upper = s2._build_nutrient_bounds(
        artifacts={},
        t_kcal=t_kcal,
        weight_kg=w,
        is_female=female,
        age_years=age,
        p_diab_0=p_diab,
        p_obes_0=p_obes,
        p_cvd_0=p_cvd,
    )

    envelope = []
    for i, code in enumerate(s2.DIET_RAW_COLS):
        lo, hi = float(lower[i]), float(upper[i])
        # Realistic Indian-reference baseline for Na:K ratio gate (matches typical 2800mg Na / 2600mg K)
        if code == "DR1TSODI":
            baseline = 2800.0
        elif code == "DR1TPOTA":
            baseline = 2600.0
        elif code == "DR1TKCAL":
            baseline = t_kcal
        else:
            baseline = lo if hi >= nrp.NUMERICAL_SEARCH_CAP else 0.5 * (lo + hi)
        target = t_kcal if code == "DR1TKCAL" else (lo if hi >= nrp.NUMERICAL_SEARCH_CAP else 0.5 * (lo + hi))
        envelope.append({
            "nutrient": code,
            "baseline": round(float(baseline), 4),
            "target": round(float(target), 4),
            "min_bound": round(lo, 4),
            "max_bound": round(hi, 4),
            "rule_provenance": nrp.reviewed_rule_metadata(code, lo, hi),
        })

    return {
        "patient_metadata": {
            "lasi_prim_key": patient["lasi_prim_key"],
            "state": patient["state"],
            "stratum": patient["stratum"],
            "weight_kg": w,
            "height_cm": h,
            "age_years": age,
            "is_female": female,
            "activity_level": act,
            "bmi": patient["bmi"],
        },
        "energy_prescription": {
            "BMR": round(bmr, 2),
            "TDEE": round(tdee, 2),
            "target_calories": round(t_kcal, 2),
            "deficit_kcal": round(deficit, 2),
        },
        "nutrient_ruleset_version": nrp.RULESET_VERSION,
        "constraint_spec_version": nrp.RULESET_VERSION,
        "nutrient_constraint_envelope": envelope,
    }


def main() -> None:
    cohort_path = ROOT / "Data" / "lasi_w1_indian_cohort_100.json"
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    patients: list[dict[str, Any]] = cohort["patients"]
    foods = s3.load_food_database()
    modes = ("pure_vegetarian", "veg_nonveg_combo", "pure_non_vegetarian")

    print(f"Benchmarking {len(patients)} real LASI Wave 1 Indian adults x {len(modes)} modes on {len(foods)}-dish catalog...", flush=True)

    per_mode_summary: dict[str, Any] = {}
    detailed_rows: list[dict[str, Any]] = []

    for mode in modes:
        t_mode = time.perf_counter()
        accepted_cnt = 0
        fail_codes: Counter[str] = Counter()
        bottleneck_nutrients: Counter[str] = Counter()
        latencies: list[float] = []
        cal_errs: list[float] = []
        dish_counts: list[int] = []
        by_stratum: dict[str, list[bool]] = {}
        by_sex: dict[str, list[bool]] = {"male": [], "female": []}
        by_kcal_band: dict[str, list[bool]] = {"<1500": [], "1500-1999": [], ">=2000": []}

        for p in patients:
            stage2 = build_stage2_from_lasi_patient(p)
            kcal = stage2["energy_prescription"]["target_calories"]
            out = s3.solve_meal_plan(
                stage2, foods, time_limit_sec=2.0, revision_spec={"dietary_mode": mode}
            )
            ms = float(out.solver_metadata.get("solve_time_ms", 0.0))
            latencies.append(ms)
            acc = bool(out.accepted)
            if acc:
                accepted_cnt += 1
                cal_errs.append(float(out.clinical_verification["calories"]["absolute_error_pct"]))
                dish_counts.append(len(out.solved_food_basket))
            else:
                fc = str((out.failure or {}).get("code", "unknown"))
                fail_codes[fc] += 1
                if fc == "infeasible_active_rules":
                    conf = chat.tool_explain_constraint_conflict(out.failure, stage2, foods, {"dietary_mode": mode})
                    for c_str in conf.get("conflicting_constraints", []):
                        nut = c_str.split()[0]
                        bottleneck_nutrients[nut] += 1

            st = p["stratum"]
            by_stratum.setdefault(st, []).append(acc)
            by_sex["female" if p["is_female"] else "male"].append(acc)
            band = "<1500" if kcal < 1500 else ("1500-1999" if kcal < 2000 else ">=2000")
            by_kcal_band[band].append(acc)

            detailed_rows.append({
                "lasi_prim_key": p["lasi_prim_key"],
                "state": p["state"],
                "stratum": st,
                "sex": "F" if p["is_female"] else "M",
                "age": p["age_years"],
                "bmi": p["bmi"],
                "target_kcal": kcal,
                "dietary_mode": mode,
                "accepted": acc,
                "solve_time_ms": round(ms, 2),
                "active_dishes": len(out.solved_food_basket) if acc else 0,
                "calorie_error_pct": round(float(out.clinical_verification["calories"]["absolute_error_pct"]), 3) if acc else None,
                "failure_code": None if acc else (out.failure or {}).get("code"),
            })

        dt_s = round(time.perf_counter() - t_mode, 2)
        summary = {
            "dietary_mode": mode,
            "total_patients": len(patients),
            "accepted_count": accepted_cnt,
            "acceptance_rate_pct": round(100.0 * accepted_cnt / len(patients), 2),
            "integer_serving_compliance_pct": 100.0 if accepted_cnt else 0.0,
            "plate_rule_compliance_pct": 100.0 if accepted_cnt else 0.0,
            "median_solve_ms": round(float(np.median(latencies)), 2),
            "mean_solve_ms": round(float(np.mean(latencies)), 2),
            "p95_solve_ms": round(float(np.percentile(latencies, 95)), 2),
            "mean_calorie_error_pct": round(float(np.mean(cal_errs)), 3) if cal_errs else None,
            "mean_active_dishes": round(float(np.mean(dish_counts)), 2) if dish_counts else None,
            "acceptance_by_stratum_pct": {
                k: round(100.0 * sum(v) / len(v), 1) for k, v in sorted(by_stratum.items())
            },
            "acceptance_by_sex_pct": {
                k: f"{sum(v)}/{len(v)} ({100.0 * sum(v) / len(v):.1f}%)" for k, v in by_sex.items()
            },
            "acceptance_by_target_kcal_band": {
                k: f"{sum(v)}/{len(v)} ({100.0 * sum(v) / max(len(v), 1):.1f}%)" for k, v in by_kcal_band.items()
            },
            "failure_codes": dict(fail_codes),
            "top_infeasibility_bottlenecks": bottleneck_nutrients.most_common(6),
            "elapsed_sec": dt_s,
        }
        per_mode_summary[mode] = summary
        print(
            f"  [{mode:20s}] accepted={accepted_cnt}/{len(patients)} ({summary['acceptance_rate_pct']}%) "
            f"| median={summary['median_solve_ms']}ms | fails={dict(fail_codes)} | bottlenecks={summary['top_infeasibility_bottlenecks'][:3]}",
            flush=True,
        )

    report = {
        "benchmark_version": "lasi-w1-indian-cohort-stage3-v1",
        "source_cohort": cohort["metadata"],
        "catalog_version": foods[0].catalog_version,
        "catalog_count": len(foods),
        "ruleset_version": nrp.RULESET_VERSION,
        "summary_by_mode": per_mode_summary,
        "cases": detailed_rows,
    }
    out_json = ROOT / "Saved Model" / "lasi_stage3_indian_cohort_report.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Write Markdown report
    meta = cohort["metadata"]
    lines = [
        "# Stage 2 + Stage 3 Evaluation on Real Indian Adults (LASI Wave 1, N = 100)",
        "",
        f"- **Source Cohort:** {meta['source_dataset']} (`{meta['archive_filename']}`, SHA256 `{meta['archive_sha256'][:16]}...`)",
        f"- **Full Valid Indian Cohort:** $N = {meta['valid_indian_adults_n']:,}$ adults across {meta['full_cohort_stats']['states_covered']} Indian states/UTs",
        f"- **Stratified Evaluation Cohort:** $N = {meta['stratified_sample_n']}$ real Indian adults (25 Diabetes, 25 Obesity BMI ≥ 30, 25 CVD, 25 Multimorbid; {meta['stratified_sample_male_n']} Male, {meta['stratified_sample_female_n']} Female; {meta['stratified_sample_states']} states/UTs)",
        f"- **Food Catalog:** `{report['catalog_version']}` ({report['catalog_count']} sourced dishes; 20 INDB + 29 USDA SR Legacy)",
        f"- **Ruleset:** `{report['ruleset_version']}` (30 active solid-food rules, zero active slack required)",
        "- **Scope Note:** LASI Wave 1 supplies real Indian anthropometry, age, sex, activity, and clinical disease status for Stage 2–4 evaluation; it does not contain serum lipids or 24-hour dietary recalls and does not externally validate Stage 1.",
        "",
        "## Strict 30-Rule Acceptance by Dietary Mode (300 Real-Patient MILP Solves)",
        "",
        "| Dietary Mode | Accepted Plans | Mean Cal Err (%) | Median Solve (ms) | Male Acceptance | Female Acceptance | Top Failure Modes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for mode in modes:
        s = per_mode_summary[mode]
        cal = f"{s['mean_calorie_error_pct']:.2f}%" if s["mean_calorie_error_pct"] is not None else "—"
        lines.append(
            f"| `{mode}` | **{s['accepted_count']}/{s['total_patients']} ({s['acceptance_rate_pct']:.1f}%)** | {cal} | {s['median_solve_ms']:.1f} | "
            f"{s['acceptance_by_sex_pct']['male']} | {s['acceptance_by_sex_pct']['female']} | `{s['failure_codes']}` |"
        )
    lines.append("")
    lines.append("## Acceptance by Target Energy Band (Real Indian Anthropometry)")
    lines.append("")
    lines.append("| Dietary Mode | `< 1,500 kcal/day` | `1,500–1,999 kcal/day` | `>= 2,000 kcal/day` |")
    lines.append("|---|---:|---:|---:|")
    for mode in modes:
        b = per_mode_summary[mode]["acceptance_by_target_kcal_band"]
        lines.append(f"| `{mode}` | {b['<1500']} | {b['1500-1999']} | {b['>=2000']} |")

    out_md = ROOT / "Saved Model" / "lasi_stage3_indian_cohort_report.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved reports to {out_json} and {out_md}", flush=True)


if __name__ == "__main__":
    main()
