"""Checkpointed regression evaluation of Stage 2 joint constraint repair across the 32 frozen profiles."""
import copy
from collections import Counter
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linprog

import nutrient_rule_policy as policy
from prepare_stage1_repair_v2 import load
import stage1_deployment as dep
import stage2_counterfactual_engine as s2

ROOT = Path(__file__).resolve().parent
FROZEN_DIR = ROOT / "Saved Model/stage2_constraint_audit_repaired_v1"
OUT = ROOT / "Saved Model/stage2_joint_constraint_repair_v1"
NAMES = s2.DIET_RAW_COLS
IX = {n: i for i, n in enumerate(NAMES)}


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def independently_check(n, low, high, raw, meta, cvd):
    n = np.asarray(n, dtype=float)
    v = dict(zip(NAMES, n))
    base = dict(zip(s2.FEATURE_COLS, raw))
    violations = []
    for k, x, l, h in zip(NAMES, n, low, high):
        if x < l - 1e-6 or x > h + 1e-6:
            violations.append({
                "nutrient": k,
                "value": float(x),
                "lower": float(l),
                "upper": float(h),
                "distance": float(max(l - x, x - h, 0)),
                "active": k not in policy.REPORT_ONLY | policy.EXPOSURE_REPORTING,
            })
    energy = 4 * v["DR1TPROT"] + 4 * v["DR1TCARB"] + 9 * v["DR1TTFAT"] + 7 * v["DR1TALCO"]
    energy_error = abs(energy - v["DR1TKCAL"]) / v["DR1TKCAL"] if v["DR1TKCAL"] > 0 else 1e99
    ratio_values = [
        v["DR1TSUGR"] / (v["DR1TFIBE"] + 1),
        v["DR1TSODI"] / (v["DR1TPOTA"] + 1),
        v["DR1TSFAT"] / (v["DR1TMFAT"] + v["DR1TPFAT"] + 0.1),
    ]
    ratio_limits = [
        max(base["SUGAR_TO_FIBER_RATIO"], 1.0),
        max(base["SODIUM_POTASSIUM_RATIO"], 1.0),
        max(base["SAT_TO_UNSAT_FAT_RATIO"], 0.4),
    ]
    ratios_ok = all(a <= b + 1e-6 for a, b in zip(ratio_values, ratio_limits))
    cvd_rule = cvd >= 0.5
    filter2 = (
        v["DR1TKCAL"] >= 1200
        and v["DR1TSODI"] <= (1500 if cvd_rule else 2300)
        and v["DR1TPOTA"] >= (3500 if cvd_rule else 2600)
        and 0.8 <= v["DR1TPROT"] / meta["weight_kg"] <= 2.0
    )
    return {
        "bound_violations": violations,
        "active_bounds_pass": not any(x["active"] for x in violations),
        "all_search_bounds_pass": not violations,
        "energy_error_fraction": float(energy_error),
        "energy_consistent": bool(energy_error <= 0.01 + 1e-9),
        "ratio_values": list(map(float, ratio_values)),
        "ratio_limits": list(map(float, ratio_limits)),
        "ratios_pass": bool(ratios_ok),
        "filter2_invariants_pass": bool(filter2),
        "joint_numeric_pass": bool(not violations and energy_error <= 0.01 + 1e-9 and ratios_ok and filter2),
    }


def linear_feasibility_oracle(low, high, raw, meta, cvd):
    """Independent linear feasibility witness under tightened bounds, energy balance, and ratio rules."""
    A = []
    b = []
    base = dict(zip(s2.FEATURE_COLS, raw))

    def constraint(terms, rhs):
        row = np.zeros(39)
        for k, v in terms.items():
            row[IX[k]] = v
        A.append(row)
        b.append(rhs)

    constraint({"DR1TPROT": 4, "DR1TCARB": 4, "DR1TTFAT": 9, "DR1TALCO": 7, "DR1TKCAL": -1.01}, 0)
    constraint({"DR1TPROT": -4, "DR1TCARB": -4, "DR1TTFAT": -9, "DR1TALCO": -7, "DR1TKCAL": 0.99}, 0)
    a = max(base["SUGAR_TO_FIBER_RATIO"], 1.0)
    constraint({"DR1TSUGR": 1, "DR1TFIBE": -a}, a)
    a = max(base["SODIUM_POTASSIUM_RATIO"], 1.0)
    constraint({"DR1TSODI": 1, "DR1TPOTA": -a}, a)
    a = max(base["SAT_TO_UNSAT_FAT_RATIO"], 0.4)
    constraint({"DR1TSFAT": 1, "DR1TMFAT": -a, "DR1TPFAT": -a}, 0.1 * a)

    lower = np.asarray(low, dtype=float).copy()
    upper = np.asarray(high, dtype=float).copy()
    lower[IX["DR1TKCAL"]] = max(lower[IX["DR1TKCAL"]], 1200)
    lower[IX["DR1TPROT"]] = max(lower[IX["DR1TPROT"]], 0.8 * meta["weight_kg"])
    upper[IX["DR1TPROT"]] = min(upper[IX["DR1TPROT"]], 2.0 * meta["weight_kg"])
    upper[IX["DR1TSODI"]] = min(upper[IX["DR1TSODI"]], 1500 if cvd >= 0.5 else 2300)
    lower[IX["DR1TPOTA"]] = max(lower[IX["DR1TPOTA"]], 3500 if cvd >= 0.5 else 2600)

    if (lower > upper).any():
        return {"status": "infeasible_bounds", "feasible": False}

    solved = linprog(np.zeros(39), A_ub=A, b_ub=b, bounds=list(zip(lower, upper)), method="highs")
    result = {"status": int(solved.status), "feasible": bool(solved.success), "message": solved.message}
    if solved.success:
        replay = independently_check(solved.x, low, high, raw, meta, cvd)
        result["witness"] = solved.x.tolist()
        result["independent_witness_check"] = replay
        result["max_linear_residual"] = float(max(0, np.max(np.asarray(A) @ solved.x - np.asarray(b))))
        assert result["max_linear_residual"] <= 1e-6
        assert np.all(solved.x >= lower - 1e-6) and np.all(solved.x <= upper + 1e-6)
    return result


def run_repaired_case(case: dict) -> dict:
    snapshots = {}
    low = None
    high = None
    original = {
        name: getattr(s2, name)
        for name in [
            "_run_counterfactual_optimization",
            "_project_to_joint_feasible_polytope",
            "_apply_filter_1_thermodynamic_balance",
            "_apply_filter_2_iom_safety",
            "_apply_filter_3_ratio_coherence",
        ]
    }

    def optimize_wrapper(**kwargs):
        nonlocal low, high
        low = kwargs["lower"].copy()
        high = kwargs["upper"].copy()
        res = original["_run_counterfactual_optimization"](**kwargs)
        snapshots["optimizer_clipped"] = res[0].copy()
        return res

    def proj_wrapper(*args, **kwargs):
        res = original["_project_to_joint_feasible_polytope"](*args, **kwargs)
        snapshots["after_projection"] = res[0].copy()
        return res

    def filter_wrapper(name, label):
        def wrapped(n, *args, **kwargs):
            res = original[name](n, *args, **kwargs)
            snapshots[label] = n.copy()
            return res
        return wrapped

    s2._run_counterfactual_optimization = optimize_wrapper
    s2._project_to_joint_feasible_polytope = proj_wrapper
    for i in [1, 2, 3]:
        name = list(original)[i + 1]
        setattr(s2, name, filter_wrapper(name, f"after_filter_{i}"))

    start = time.perf_counter()
    try:
        raw, _ = s2.canonicalize_dietary_ratios(case["vector"])
        os.environ["BIONUTRIGPT_STAGE2_JSON"] = str(OUT / f"dossier_{case['case']:03d}_LOCAL.json")
        dossier = s2.generate_counterfactual_prescription(raw, case["metadata"])
        p = dossier["patient_baseline"]["probabilities"]
        cvd = p["cvd"]

        checks = {label: independently_check(v, low, high, raw, case["metadata"], cvd) for label, v in snapshots.items()}
        final = checks["after_filter_3"]
        deltas = dossier["audit_filters"]["filter_4_closed_loop_rescoring"]
        closed = all(deltas["delta_" + k] <= 0.0015 for k in ["diabetes", "obesity", "cvd"])
        independent = final["joint_numeric_pass"] and closed

        proj_audit = dossier.get("audit_projection", {})
        return {
            "case": case["case"],
            "SEQN": case.get("SEQN"),
            "certified": dossier["certified_for_stage3"],
            "independent_joint_pass": independent,
            "false_accept": bool(dossier["certified_for_stage3"] and not independent),
            "snapshots": {k: v.tolist() for k, v in snapshots.items()},
            "checks": checks,
            "closed_loop_pass": closed,
            "score_deltas": {k: deltas["delta_" + k] for k in ["diabetes", "obesity", "cvd"]},
            "target_probabilities": deltas["target_probabilities"],
            "baseline_probabilities": p,
            "projection_audit": proj_audit,
            "linear_feasibility": linear_feasibility_oracle(low, high, raw, case["metadata"], cvd),
            "seconds": time.perf_counter() - start,
            "error": None,
        }
    except Exception as exc:
        return {
            "case": case["case"],
            "SEQN": case.get("SEQN"),
            "error": repr(exc),
            "seconds": time.perf_counter() - start,
            "certified": False,
        }
    finally:
        for name, fn in original.items():
            setattr(s2, name, fn)


def main():
    assert dep.REPAIRED, "Repaired evaluation requires release=repaired_v1"
    dep.verify_bundle()
    torch.set_num_threads(os.cpu_count() or 1)

    # Load frozen cases from Saved Model/stage2_constraint_audit_repaired_v1/cases_LOCAL.json
    cases_frozen_path = FROZEN_DIR / "cases_LOCAL.json"
    assert cases_frozen_path.is_file(), f"Missing frozen audit cases: {cases_frozen_path}"
    cases = json.loads(cases_frozen_path.read_text(encoding="utf-8"))
    assert len(cases) == 32

    # Verification of frozen case hashes
    frozen_cases_sha = hashlib.sha256(cases_frozen_path.read_bytes()).hexdigest()
    assert frozen_cases_sha == "7b9019620ca755651973a38dbb91d5b1ae5674637b23773eb46924860be19036", f"Unexpected frozen cases hash: {frozen_cases_sha}"

    paths = [
        cases_frozen_path,
        ROOT / "stage2_counterfactual_engine.py",
        ROOT / "stage1_deployment.py",
        ROOT / "nutrient_rule_policy.py",
        dep.BUNDLE / "manifest.json",
        dep.CHECKPOINT,
        dep.PREPROCESSOR,
        dep.BUNDLE / "calibrators.joblib",
        ROOT / "Data/preprocess_for_training.py",
        Path(__file__),
    ]
    binding = {str(p.relative_to(ROOT)): dep.sha(p) for p in paths}

    protocol = {
        "version": "stage2_joint_constraint_repair_v1",
        "n": 32,
        "sample_source": "Frozen 32 cases from Saved Model/stage2_constraint_audit_repaired_v1/cases_LOCAL.json",
        "sample_sha256": frozen_cases_sha,
        "comparator_audit_version": "stage2_constraint_audit_repaired_v1",
        "method": "50-step Adam counterfactual search followed by scale-normalized convex quadratic projection onto the joint feasible polytope",
        "projection_solver": "SLSQP with HiGHS LP feasibility witness warm start",
        "scale_normalization": "DAILY_REFERENCE_SCALES (US FDA Reference Daily Intakes / Daily Values for 2000 kcal diet)",
        "tightened_bounds_scope": "Nutrient component box tightened with Filter 2 invariants (KCAL >= 1200, 0.8W <= PROT <= 2.0W, SODI <= 1500/2300, POTA >= 3500/2600)",
        "joint_linear_inequalities": "Thermodynamic energy balance (+- 1%) and 3 clinical ratio envelopes (sugar/fiber, sodium/potassium, sat/unsat fat)",
        "closed_loop_gate": "Delta p <= 0.0015 for diabetes, obesity, cvd rescored through calibrated Stage 1 model",
        "bound_tolerance": 1e-6,
        "energy_tolerance": 0.01,
        "release": "repaired_v1",
        "thresholds": dep.THRESHOLDS,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "torch_threads": torch.get_num_threads(),
        "scope": "Regression repair of Stage 2 joint mathematical constraints across 32 frozen profiles; diagnostic engineering evaluation, not clinical trial or causal claim",
        "hashes": binding,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        existing_protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
        assert existing_protocol["hashes"] == protocol["hashes"], "Source hashes changed during run; check protocol"
    else:
        write_json(OUT / "protocol.json", protocol)
        write_json(OUT / "cases_LOCAL.json", cases)

    results = []
    previous_json = os.environ.get("BIONUTRIGPT_STAGE2_JSON")
    try:
        for case in cases:
            case_path = OUT / f"case_{case['case']:03d}_LOCAL.json"
            if case_path.exists():
                res = json.loads(case_path.read_text(encoding="utf-8"))
            else:
                res = run_repaired_case(case)
                write_json(case_path, res)
            results.append(res)
            status_str = f"Case {res['case']:02d}/32: certified={res['certified']} score_pass={res.get('closed_loop_pass')} time={res.get('seconds', 0.0):.2f}s"
            print(status_str, flush=True)
    finally:
        if previous_json is None:
            os.environ.pop("BIONUTRIGPT_STAGE2_JSON", None)
        else:
            os.environ["BIONUTRIGPT_STAGE2_JSON"] = previous_json

    # Verify source hashes remained unchanged during execution
    for name, h in binding.items():
        assert dep.sha(ROOT / name) == h, f"Source changed during regression run: {name}"

    valid = [r for r in results if not r.get("error")]
    summary = {
        "completed_cases": len(results),
        "returned_dossiers": len(valid),
        "errors": len(results) - len(valid),
        "certified": sum(r["certified"] for r in results),
        "false_accepts": sum(r["false_accept"] for r in valid),
        "stage_bound_failures": {
            k: sum(not r["checks"][k]["all_search_bounds_pass"] for r in valid)
            for k in ["optimizer_clipped", "after_projection", "after_filter_1", "after_filter_2", "after_filter_3"]
        },
        "final_energy_failures": sum(not r["checks"]["after_filter_3"]["energy_consistent"] for r in valid),
        "final_ratio_failures": sum(not r["checks"]["after_filter_3"]["ratios_pass"] for r in valid),
        "closed_loop_failures": sum(not r["closed_loop_pass"] for r in valid),
        "linear_feasible": sum(r["linear_feasibility"]["feasible"] for r in valid),
        "linear_feasible_but_rejected": sum(r["linear_feasibility"]["feasible"] and not r["certified"] for r in valid),
        "final_violated_nutrients": dict(Counter(v["nutrient"] for r in valid for v in r["checks"]["after_filter_3"]["bound_violations"])),
        "runtime_seconds": sum(r["seconds"] for r in results),
        "mean_case_seconds": float(np.mean([r["seconds"] for r in results])),
        "source_hashes_unchanged": True,
    }
    write_json(OUT / "summary.json", summary)
    print("\nRegression Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
