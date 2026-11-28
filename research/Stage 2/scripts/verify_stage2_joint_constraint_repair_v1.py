"""Corrected historical-result verification, with fresh frozen-model rescoring.

Numerical archive checks do not call the engine's constraint checker. Model
replay uses deployed preprocessing/candidate construction and separately loaded
research model weights/calibrators. Historical source hashes are checked against
the archived pre-correction engine. New receipts do not overwrite the old run.
"""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
BASELINE_DIR = ROOT / "Saved Model/stage2_constraint_audit_repaired_v1"
REPAIRED_DIR = ROOT / "Saved Model/stage2_joint_constraint_repair_v1"

DIET_NAMES = (
    "DR1TKCAL", "DR1TPROT", "DR1TCARB", "DR1TSUGR", "DR1TFIBE",
    "DR1TTFAT", "DR1TSFAT", "DR1TMFAT", "DR1TPFAT", "DR1TCHOL",
    "DR1TATOC", "DR1TRET", "DR1TVARA", "DR1TACAR", "DR1TBCAR",
    "DR1TLYCO", "DR1TLZ", "DR1TVB1", "DR1TVB2", "DR1TNIAC",
    "DR1TVB6", "DR1TFOLA", "DR1TFA", "DR1TFF", "DR1TFDFE",
    "DR1TVB12", "DR1TVC", "DR1TVK", "DR1TCALC", "DR1TPHOS",
    "DR1TMAGN", "DR1TIRON", "DR1TZINC", "DR1TCOPP", "DR1TSODI",
    "DR1TPOTA", "DR1TSELE", "DR1TCAFF", "DR1TALCO",
)


def independent_check(values, lower, upper, keys, baseline, metadata, cvd):
    n = dict(zip(keys, values))
    lo = np.array(lower, dtype=float)
    hi = np.array(upper, dtype=float)
    v = np.array(values, dtype=float)

    bound_fail = (v < lo - 1e-6) | (v > hi + 1e-6)
    energy = 4.0 * n["DR1TPROT"] + 4.0 * n["DR1TCARB"] + 9.0 * n["DR1TTFAT"] + 7.0 * n["DR1TALCO"]
    energy_ok = n["DR1TKCAL"] > 0 and abs(energy - n["DR1TKCAL"]) / n["DR1TKCAL"] <= 0.01 + 1e-9

    ratios = [
        n["DR1TSUGR"] / (n["DR1TFIBE"] + 1.0),
        n["DR1TSODI"] / (n["DR1TPOTA"] + 1.0),
        n["DR1TSFAT"] / (n["DR1TMFAT"] + n["DR1TPFAT"] + 0.1),
    ]
    limits = [
        max(baseline["DR1TSUGR"] / (baseline["DR1TFIBE"] + 1.0), 1.0),
        max(baseline["DR1TSODI"] / (baseline["DR1TPOTA"] + 1.0), 1.0),
        max(baseline["DR1TSFAT"] / (baseline["DR1TMFAT"] + baseline["DR1TPFAT"] + 0.1), 0.4),
    ]
    ratio_ok = all(a <= b + 1e-6 for a, b in zip(ratios, limits))

    weight_kg = metadata["weight_kg"]
    f2 = (
        n["DR1TKCAL"] >= 1200.0
        and n["DR1TSODI"] <= (1500.0 if cvd >= 0.5 else 2300.0)
        and n["DR1TPOTA"] >= (3500.0 if cvd >= 0.5 else 2600.0)
        and 0.8 <= n["DR1TPROT"] / weight_kg <= 2.0
    )
    joint = bool(not bound_fail.any() and energy_ok and ratio_ok and f2)
    return {
        "bound_names": [k for k, b in zip(keys, bound_fail) if b],
        "energy_ok": bool(energy_ok),
        "ratios_ok": bool(ratio_ok),
        "filter2_ok": bool(f2),
        "joint": joint,
    }


def main():
    protocol = json.loads((REPAIRED_DIR / "protocol.json").read_text(encoding="utf-8"))
    summary = json.loads((REPAIRED_DIR / "summary.json").read_text(encoding="utf-8"))
    baseline_summary = json.loads((BASELINE_DIR / "summary.json").read_text(encoding="utf-8"))
    cases = json.loads((REPAIRED_DIR / "cases_LOCAL.json").read_text(encoding="utf-8"))
    assert len(cases) == 32

    assert cases == json.loads((BASELINE_DIR / "cases_LOCAL.json").read_text())
    correction = ROOT / "Saved Model/stage2_joint_repair_correction_v2"
    correction.mkdir(exist_ok=True)
    # Check original source bindings against the archived original engine
    for p, h in protocol["hashes"].items():
        source = correction / "before_correction/stage2_counterfactual_engine.py" if p == "stage2_counterfactual_engine.py" else ROOT / p
        computed_h = hashlib.sha256(source.read_bytes()).hexdigest()
        assert computed_h == h, f"Hash mismatch for {p}: expected {h}, got {computed_h}"

    counts = Counter()
    stage_bound_counts = Counter()
    violations = Counter()
    errors = []
    case_results = []
    newly_accepted_cases = []
    status_changed_cases = []
    original_accepts = []
    energy_failures = ratio_failures = score_failures = 0

    for case in cases:
        i = case["case"]
        r = json.loads((REPAIRED_DIR / f"case_{i:03d}_LOCAL.json").read_text(encoding="utf-8"))
        baseline_case = json.loads((BASELINE_DIR / f"case_{i:03d}_LOCAL.json").read_text(encoding="utf-8"))
        if r.get("error"):
            errors.append({"case": i, "error": r["error"]})
            continue

        dossier = json.loads((REPAIRED_DIR / f"dossier_{i:03d}_LOCAL.json").read_text(encoding="utf-8"))
        envelope = dossier["nutrient_constraint_envelope"]
        keys = [x["nutrient"] for x in envelope]
        assert len(keys) == len(set(keys)) == 39

        lower = [x["min_bound"] for x in envelope]
        upper = [x["max_bound"] for x in envelope]
        baseline_nutrients = {x["nutrient"]: x["baseline"] for x in envelope}
        p = dossier["patient_baseline"]["probabilities"]
        cvd = p["cvd"]

        # Check all snapshots independently
        for stage, n_vec in r["snapshots"].items():
            obs = independent_check(n_vec, lower, upper, keys, baseline_nutrients, case["metadata"], cvd)
            if obs["bound_names"]:
                stage_bound_counts[stage] += 1

        final_snapshot = r["snapshots"]["after_filter_3"]
        final_obs = independent_check(final_snapshot, lower, upper, keys, baseline_nutrients, case["metadata"], cvd)
        violations.update(final_obs["bound_names"])

        assert np.isfinite(final_snapshot).all()
        energy_failures += not final_obs["energy_ok"]
        ratio_failures += not final_obs["ratios_ok"]
        # Saved nutrient agreement; feature immutability is tested during model replay
        raw_vec = case["vector"]
        target_vals = [x["target"] for x in envelope]
        np.testing.assert_allclose(final_snapshot, target_vals, atol=1e-12)

        # Closed loop rescoring verification
        scores = dossier["audit_filters"]["filter_4_closed_loop_rescoring"]
        deltas = [scores["target_probabilities"][k] - p[k] for k in ["diabetes", "obesity", "cvd"]]
        np.testing.assert_allclose(deltas, [scores["delta_" + k] for k in ["diabetes", "obesity", "cvd"]], atol=1e-12)

        closed = all(d <= 0.0015 for d in deltas)
        assert closed == r["closed_loop_pass"]
        score_failures += not closed

        independently_accepted = closed and final_obs["joint"]
        assert independently_accepted == r["independent_joint_pass"]
        assert r["certified"] == dossier["certified_for_stage3"]

        counts["certified"] += r["certified"]
        counts["independent_joint_pass"] += independently_accepted
        counts["false_accepts"] += r["certified"] and not independently_accepted
        counts["false_rejects"] += not r["certified"] and independently_accepted

        # Check comparison with baseline
        baseline_certified = baseline_case["certified"]
        if baseline_certified: original_accepts.append(i)
        if not baseline_certified and r["certified"]:
            newly_accepted_cases.append(i)
        if baseline_certified != r["certified"]:
            status_changed_cases.append({"case": i, "baseline_certified": baseline_certified, "repaired_certified": r["certified"]})

        # Oracle linear feasibility
        oracle = r["linear_feasibility"]
        counts["linear_feasible"] += oracle["feasible"]
        counts["linear_feasible_but_rejected"] += oracle["feasible"] and not r["certified"]
        if oracle["feasible"]:
            witness = np.array(oracle["witness"])
            lo = np.array(lower)
            hi = np.array(upper)
            assert np.isfinite(witness).all()
            assert np.all(witness >= lo - 1e-6) and np.all(witness <= hi + 1e-6)
            w = dict(zip(keys, witness))
            e = 4*w['DR1TPROT']+4*w['DR1TCARB']+9*w['DR1TTFAT']+7*w['DR1TALCO']
            assert .99*w['DR1TKCAL']-1e-6 <= e <= 1.01*w['DR1TKCAL']+1e-6
            for num, den in [('DR1TSUGR','DR1TFIBE'),('DR1TSODI','DR1TPOTA')]:
                limit = max(baseline_nutrients[num]/(baseline_nutrients[den]+1),1.)
                assert w[num] <= limit*(w[den]+1)+1e-6
            limit = max(baseline_nutrients['DR1TSFAT']/(baseline_nutrients['DR1TMFAT']+baseline_nutrients['DR1TPFAT']+.1),.4)
            assert w['DR1TSFAT'] <= limit*(w['DR1TMFAT']+w['DR1TPFAT']+.1)+1e-6
            assert w['DR1TKCAL'] >= 1200-1e-6
            assert .8*case['metadata']['weight_kg']-1e-6 <= w['DR1TPROT'] <= 2*case['metadata']['weight_kg']+1e-6
            assert w['DR1TSODI'] <= (1500 if cvd>=.5 else 2300)+1e-6
            assert w['DR1TPOTA'] >= (3500 if cvd>=.5 else 2600)-1e-6

        case_results.append({
            "case": i,
            "certified": r["certified"],
            "baseline_certified": baseline_certified,
            "bounds_pass": not bool(final_obs["bound_names"]),
            "energy_pass": final_obs["energy_ok"],
            "ratios_pass": final_obs["ratios_ok"],
            "filter2_pass": final_obs["filter2_ok"],
            "score_gate_pass": closed,
            "score_deltas": r["score_deltas"],
            "projection_nit": r["projection_audit"].get("nit"),
            "projection_distance": r["projection_audit"].get("projection_distance_normalized"),
        })

    # Summary integrity checks
    for k in ["certified", "false_accepts", "linear_feasible", "linear_feasible_but_rejected"]:
        assert counts[k] == summary[k], f"Summary mismatch on {k}: {counts[k]} vs {summary[k]}"
    assert len(errors) == summary["errors"]
    assert dict(violations) == summary["final_violated_nutrients"]

    for stage, count in summary["stage_bound_failures"].items():
        assert stage_bound_counts[stage] == count
    assert energy_failures == summary["final_energy_failures"]
    assert ratio_failures == summary["final_ratio_failures"]
    assert score_failures == summary["closed_loop_failures"]
    assert len(original_accepts) == baseline_summary["certified"]
    originally_accepted = [c["case"] for c in case_results if c["baseline_certified"] and c["certified"]]
    lost_accepts = [c["case"] for c in case_results if c["baseline_certified"] and not c["certified"]]
    assert counts["certified"] - len(original_accepts) == len(newly_accepted_cases) - len(lost_accepts)
    # Genuine frozen-model inference, independent of saved probability differences.
    from review_stage2_joint_repair_v1 import main as replay_model
    replay_model()
    replay = json.loads((correction / "model_replay/independent_review.json").read_text())
    assert replay["passed"] and replay["model_score_gate_replayed_all_32"]
    assert replay["preserved"] == originally_accepted

    verification_record = {
        "verification_passed": True,
        "audit_version": "stage2_joint_constraint_repair_v1",
        "comparator_baseline_version": "stage2_constraint_audit_repaired_v1",
        "cases_evaluated": 32,
        "errors": errors,
        "counts": dict(counts),
        "stage_bound_failures": dict(stage_bound_counts),
        "final_violated_nutrients": dict(violations),
        "paired_comparison": {
            "baseline_certified": baseline_summary["certified"],
            "repaired_certified": summary["certified"],
            "net_certified_gain": summary["certified"] - baseline_summary["certified"],
            "newly_accepted_cases": newly_accepted_cases,
            "originally_accepted_preserved": originally_accepted,
            "baseline_calorie_bound_failures": baseline_summary["stage_bound_failures"]["after_filter_3"],
            "repaired_calorie_bound_failures": summary["stage_bound_failures"]["after_filter_3"],
            "baseline_score_gate_failures": baseline_summary["closed_loop_failures"],
            "repaired_score_gate_failures": summary["closed_loop_failures"],
            "unresolved_score_gate_cases": [c["case"] for c in case_results if not c["score_gate_pass"]],
        },
        "source_bindings_passed": True,
        "source_scope": "Historical run checked against archived original engine; current solver correction tested separately",
        "model_score_replay": replay,
        "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Independent replay and numerical verification of 32 frozen profiles; no model training, no clinical trials",
    }
    (correction / "verification.json").write_text(json.dumps(verification_record, indent=2), encoding="utf-8")
    (correction / "verification_cases_LOCAL.json").write_text(json.dumps(case_results, indent=2), encoding="utf-8")
    print("\nIndependent Verification Passed:")
    print(json.dumps({k:v for k,v in verification_record.items() if k != "model_score_replay"}, indent=2))


if __name__ == "__main__":
    main()
