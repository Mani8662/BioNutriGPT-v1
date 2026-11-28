#!/usr/bin/env python3
"""Independent Verifier for Stage 3 Solver-Budget Matched Comparisons (v2).

Independently reproduces from recomputed_LOCAL.json:
- Comparison membership and matching
- Counts and denominators
- Participant-weighted point estimates and 10,000-resample bootstrap CIs
- Exact per-mode hypothesis tests and Holm step-down adjustments
- Upstream Stage 2 gate decomposition
- Common-success sensitivity results

Also executes 6 synthetic failure fixtures to verify active anomaly detection.
Writes receipt to Saved Model/stage3_budget_comparisons_v1/verification.json.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy.stats as stats

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "Saved Model" / "stage3_budget_comparisons_v2"
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


def verify_bootstrap(data: list[float], saved: dict[str, Any], count_holder: list[int]) -> None:
    arr = np.asarray(data, dtype=float)
    pt = float(np.mean(arr))
    n = len(arr)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, n, size=(BOOTSTRAP_RESAMPLES, n))
    means = np.mean(arr[idx], axis=1)
    ci_low = float(np.percentile(means, 2.5))
    ci_high = float(np.percentile(means, 97.5))

    assert saved["n_participants"] == n, f"Bootstrap n mismatch: {saved['n_participants']} != {n}"
    assert np.isclose(saved["mean_difference"], pt, atol=1e-10), f"Mean diff mismatch: {saved['mean_difference']} != {pt}"
    assert np.allclose(saved["ci95"], [ci_low, ci_high], atol=1e-10), f"CI mismatch: {saved['ci95']} != {[ci_low, ci_high]}"
    count_holder[0] += 1


def independently_compute_matched_diffs(
    key_a_fn: Any,
    key_b_fn: Any,
    idx_map: dict[tuple, dict],
    endpoint_name: str,
    is_binary: bool,
) -> list[float]:
    part_diffs = []
    for i in range(100):
        d_modes = []
        for m in MODES:
            ra = idx_map.get(key_a_fn(i, m))
            rb = idx_map.get(key_b_fn(i, m))
            if ra is None or rb is None:
                continue
            if is_binary:
                v_a = 100.0 if ra[endpoint_name] else 0.0
                v_b = 100.0 if rb[endpoint_name] else 0.0
                d_modes.append(v_a - v_b)
            else:
                if ra["accepted"] and rb["accepted"]:
                    d_modes.append(float(ra[endpoint_name]) - float(rb[endpoint_name]))
        if d_modes:
            part_diffs.append(float(np.mean(d_modes)))
    return part_diffs


def validate_membership(rows, idx_map):
    expected = {(i,a,m,b,v) for i in range(100) for a in ARMS for m in MODES for b in BUDGETS_SEC for v in VARIANTS}
    assert len(rows) == len(expected), "Missing or extra record"
    keys = [(r["case"],r["arm"],r["mode"],r["budget_seconds"],r["variant"]) for r in rows]
    assert len(set(keys)) == len(keys) and set(keys) == expected, "Invalid or duplicate record keys"
    assert set(idx_map) == expected, "Invalid index membership"
    for key, row in idx_map.items():
        assert key == (row["case"],row["arm"],row["mode"],row["budget_seconds"],row["variant"]), "Index/record identity mismatch"
    ids = {}
    for r in rows:
        ids.setdefault(r["case"], set()).add(r["SEQN"])
    assert all(len(v)==1 for v in ids.values()), "Participant identity mismatch"
    assert len({next(iter(v)) for v in ids.values()}) == 100, "Duplicate participant"


def test_failure_fixtures(rows, results, idx_map):
    passed = []
    for name in ["missing_record", "swapped_mode", "invalid_budget", "changed_acceptance", "changed_target_error", "changed_ci"]:
        rr, result = copy.deepcopy(rows), copy.deepcopy(results)
        if name == "missing_record": rr.pop()
        if name == "invalid_budget": rr[0]["budget_seconds"] = 7.5
        ii = {(r["case"],r["arm"],r["mode"],r["budget_seconds"],r["variant"]):r for r in rr}
        k = (0,"rules_only","pure_vegetarian",3.0,"variant_a")
        if name == "swapped_mode":
            other=(0,"rules_only","veg_nonveg_combo",3.0,"variant_a")
            ii[k],ii[other]=ii[other],ii[k]
        if name == "changed_acceptance": ii[k]["accepted"] = not ii[k]["accepted"]
        if name == "changed_target_error":
            k2=next(k for k,r in ii.items() if k[1]=="rules_only" and k[3]==5.0 and k[4]=="variant_b" and r["accepted"] and ii[(k[0],k[1],k[2],3.0,k[4])]["accepted"])
            ii[k2]["l1"] += 10.0
        if name == "changed_ci": result["budget_comparisons"]["variant_b"]["rules_only"]["5s_vs_3s"]["l1_target_error"]["ci95"][0] -= 0.1
        try:
            validate_statistics(rr,result,ii)
        except AssertionError as exc:
            passed.append({"fixture":name,"rejection":str(exc) or "statistical assertion"})
        else:
            raise RuntimeError("Verifier incorrectly accepted corruption: " + name)
    # Positive control: the same validator must accept clean inputs.
    validate_statistics(rows,results,idx_map)
    return passed


def validate_statistics(rows, results, idx_map):
    validate_membership(rows, idx_map)
    intervals_verified = [0]
    exact_tests_verified = 0

    # 3. Verify Budget Comparisons
    print("\n[Step 2] Auditing Budget Comparisons (5s vs 3s, 10s vs 5s, 10s vs 3s)...")
    budget_raw_p = []
    budget_refs = []

    budget_pairs = [
        ("5s_vs_3s", 5.0, 3.0),
        ("10s_vs_5s", 10.0, 5.0),
        ("10s_vs_3s", 10.0, 3.0),
    ]

    for v in VARIANTS:
        for arm in ARMS:
            for pair_name, b_hi, b_lo in budget_pairs:
                saved_contrast = results["budget_comparisons"][v][arm][pair_name]

                key_a = lambda i, m, b=b_hi, a=arm: (i, a, m, b, v)
                key_b = lambda i, m, b=b_lo, a=arm: (i, a, m, b, v)

                # Verify binary endpoints
                for ep in ["accepted", "e2e"]:
                    diffs = independently_compute_matched_diffs(key_a, key_b, idx_map, ep, is_binary=True)
                    verify_bootstrap(diffs, saved_contrast[ep], intervals_verified)

                # Verify continuous endpoints
                for ep in ["l1_target_error", "runtime_seconds", "energy_error_kcal", "energy_error_pct", "mass_g", "servings", "dish_count"]:
                    row_key = "l1" if ep == "l1_target_error" else ("runtime" if ep == "runtime_seconds" else ep)
                    diffs = independently_compute_matched_diffs(key_a, key_b, idx_map, row_key, is_binary=False)
                    if diffs:
                        verify_bootstrap(diffs, saved_contrast[ep], intervals_verified)

                # Verify per-mode breakdown
                for m in MODES:
                    m_saved = saved_contrast["dietary_modes"][m]
                    wins = sum(idx_map[key_a(i, m)]["accepted"] and not idx_map[key_b(i, m)]["accepted"] for i in range(100))
                    losses = sum(not idx_map[key_a(i, m)]["accepted"] and idx_map[key_b(i, m)]["accepted"] for i in range(100))
                    joint = sum(idx_map[key_a(i, m)]["accepted"] and idx_map[key_b(i, m)]["accepted"] for i in range(100))
                    a_acc = sum(idx_map[key_a(i, m)]["accepted"] for i in range(100))
                    b_acc = sum(idx_map[key_b(i, m)]["accepted"] for i in range(100))

                    assert m_saved["condition_a_accepted"] == a_acc
                    assert m_saved["condition_b_accepted"] == b_acc
                    assert m_saved["wins"] == wins
                    assert m_saved["losses"] == losses
                    assert m_saved["jointly_accepted"] == joint

                    p_val = exact_paired_binomial_test(wins, losses)
                    assert abs(m_saved["raw_p_value"] - p_val) < 1e-12
                    exact_tests_verified += 1

                    m_diffs = [(100.0 if idx_map[key_a(i, m)]["accepted"] else 0.0) - (100.0 if idx_map[key_b(i, m)]["accepted"] else 0.0) for i in range(100)]
                    verify_bootstrap(m_diffs, {"n_participants": 100, "mean_difference": m_saved["diff_pp"], "ci95": m_saved["ci95"]}, intervals_verified)

                    budget_raw_p.append(p_val)
                    budget_refs.append(m_saved)

    # Verify Family A Holm step-down adjustment
    assert len(budget_raw_p) == 54
    exp_budget_holm = holm_adjust(budget_raw_p)
    for m_saved, h_exp in zip(budget_refs, exp_budget_holm):
        assert abs(m_saved["holm_adjusted_p"] - h_exp) < 1e-12
    print(f"  Verified 18 budget comparison series and Family A Holm adjustment ({len(budget_raw_p)} tests).")

    # 4. Verify Common-Success Sensitivity Analysis
    print("\n[Step 3] Auditing Common-Success Sensitivity Analysis...")
    for v in VARIANTS:
        for arm in ARMS:
            sens_saved = results["budget_sensitivity_common_success"][v][arm]
            l1_3s_part, l1_5s_part, l1_10s_part = [], [], []
            d_5_3, d_10_5, d_10_3 = [], [], []
            common_cells = 0

            for i in range(100):
                m3, m5, m10 = [], [], []
                for m in MODES:
                    r3 = idx_map[(i, arm, m, 3.0, v)]
                    r5 = idx_map[(i, arm, m, 5.0, v)]
                    r10 = idx_map[(i, arm, m, 10.0, v)]
                    if r3["accepted"] and r5["accepted"] and r10["accepted"]:
                        m3.append(r3["l1"])
                        m5.append(r5["l1"])
                        m10.append(r10["l1"])
                        common_cells += 1
                if m3:
                    avg3, avg5, avg10 = float(np.mean(m3)), float(np.mean(m5)), float(np.mean(m10))
                    l1_3s_part.append(avg3)
                    l1_5s_part.append(avg5)
                    l1_10s_part.append(avg10)
                    d_5_3.append(avg5 - avg3)
                    d_10_5.append(avg10 - avg5)
                    d_10_3.append(avg10 - avg3)

            assert sens_saved["n_participants_all_three_successful"] == len(l1_3s_part)
            assert sens_saved["total_common_cells"] == common_cells

            verify_bootstrap(d_5_3, sens_saved["l1_differences"]["5s_minus_3s"], intervals_verified)
            verify_bootstrap(d_10_5, sens_saved["l1_differences"]["10s_minus_5s"], intervals_verified)
            verify_bootstrap(d_10_3, sens_saved["l1_differences"]["10s_minus_3s"], intervals_verified)

    print(f"  Verified all common-success sensitivity analyses across budgets.")

    # 5. Verify Target-Source Comparisons
    print("\n[Step 4] Auditing Target-Source Comparisons (MG vs MC, MG vs RO, MC vs RO)...")
    target_raw_p = []
    target_test_refs = []
    target_e2e_raw_p = []
    target_e2e_refs = []

    target_contrasts = [
        ("model_guided_vs_minimum_change", "model_guided", "minimum_change"),
        ("model_guided_vs_rules_only", "model_guided", "rules_only"),
        ("minimum_change_vs_rules_only", "minimum_change", "rules_only"),
    ]

    for v in VARIANTS:
        for b_sec in BUDGETS_SEC:
            b_key = f"{int(b_sec)}s"
            for contrast_name, arm_a, arm_b in target_contrasts:
                saved_contrast = results["target_source_comparisons"][v][b_key][contrast_name]

                key_a = lambda i, m, a=arm_a: (i, a, m, b_sec, v)
                key_b = lambda i, m, b=arm_b: (i, b, m, b_sec, v)

                # Verify primary binary endpoints
                for ep in ["accepted", "e2e"]:
                    diffs = independently_compute_matched_diffs(key_a, key_b, idx_map, ep, is_binary=True)
                    verify_bootstrap(diffs, saved_contrast[ep], intervals_verified)

                # Verify continuous endpoints
                for ep in ["own_target_l1_error", "runtime_seconds", "energy_error_kcal", "energy_error_pct", "mass_g", "servings", "dish_count"]:
                    row_key = "l1" if ep == "own_target_l1_error" else ("runtime" if ep == "runtime_seconds" else ep)
                    diffs = independently_compute_matched_diffs(key_a, key_b, idx_map, row_key, is_binary=False)
                    if diffs:
                        verify_bootstrap(diffs, saved_contrast[ep], intervals_verified)

                # Verify Stage 2 gate decomposition
                s2_a = sum(1 for i in range(100) if read_json(TARGETS_DIR / f"case_{i:03d}_{idx_map[(i, arm_a, 'pure_vegetarian', b_sec, v)]['SEQN']}_{arm_a}.json")["overall_accepted"])
                s2_b = sum(1 for i in range(100) if read_json(TARGETS_DIR / f"case_{i:03d}_{idx_map[(i, arm_b, 'pure_vegetarian', b_sec, v)]['SEQN']}_{arm_b}.json")["overall_accepted"])
                decomp = saved_contrast["stage2_gate_decomposition"]
                assert decomp["stage2_passed_a"] == s2_a
                assert decomp["stage2_passed_b"] == s2_b
                assert decomp["stage2_gate_diff_pp"] == (s2_a - s2_b) * 1.0

                # Verify per-mode breakdown for accepted
                for m in MODES:
                    m_saved = saved_contrast["dietary_modes_accepted"][m]
                    wins = sum(idx_map[key_a(i, m)]["accepted"] and not idx_map[key_b(i, m)]["accepted"] for i in range(100))
                    losses = sum(not idx_map[key_a(i, m)]["accepted"] and idx_map[key_b(i, m)]["accepted"] for i in range(100))
                    joint = sum(idx_map[key_a(i, m)]["accepted"] and idx_map[key_b(i, m)]["accepted"] for i in range(100))
                    a_acc = sum(idx_map[key_a(i, m)]["accepted"] for i in range(100))
                    b_acc = sum(idx_map[key_b(i, m)]["accepted"] for i in range(100))

                    assert m_saved["condition_a_accepted"] == a_acc
                    assert m_saved["condition_b_accepted"] == b_acc
                    assert m_saved["wins"] == wins
                    assert m_saved["losses"] == losses
                    assert m_saved["jointly_accepted"] == joint

                    p_val = exact_paired_binomial_test(wins, losses)
                    assert abs(m_saved["raw_p_value"] - p_val) < 1e-12
                    exact_tests_verified += 1

                    m_diffs = [(100.0 if idx_map[key_a(i, m)]["accepted"] else 0.0) - (100.0 if idx_map[key_b(i, m)]["accepted"] else 0.0) for i in range(100)]
                    verify_bootstrap(m_diffs, {"n_participants": 100, "mean_difference": m_saved["diff_pp"], "ci95": m_saved["ci95"]}, intervals_verified)

                    target_raw_p.append(p_val)
                    target_test_refs.append(m_saved)

                # Verify per-mode breakdown for e2e
                for m in MODES:
                    m_e2e_saved = saved_contrast["dietary_modes_e2e"][m]
                    wins = sum(idx_map[key_a(i, m)]["e2e"] and not idx_map[key_b(i, m)]["e2e"] for i in range(100))
                    losses = sum(not idx_map[key_a(i, m)]["e2e"] and idx_map[key_b(i, m)]["e2e"] for i in range(100))
                    joint = sum(idx_map[key_a(i, m)]["e2e"] and idx_map[key_b(i, m)]["e2e"] for i in range(100))
                    a_acc = sum(idx_map[key_a(i, m)]["e2e"] for i in range(100))
                    b_acc = sum(idx_map[key_b(i, m)]["e2e"] for i in range(100))

                    assert m_e2e_saved["condition_a_accepted"] == a_acc
                    assert m_e2e_saved["condition_b_accepted"] == b_acc
                    assert m_e2e_saved["wins"] == wins
                    assert m_e2e_saved["losses"] == losses
                    assert m_e2e_saved["jointly_accepted"] == joint

                    p_val = exact_paired_binomial_test(wins, losses)
                    assert abs(m_e2e_saved["raw_p_value"] - p_val) < 1e-12
                    exact_tests_verified += 1

                    m_diffs = [(100.0 if idx_map[key_a(i, m)]["e2e"] else 0.0) - (100.0 if idx_map[key_b(i, m)]["e2e"] else 0.0) for i in range(100)]
                    verify_bootstrap(m_diffs, {"n_participants": 100, "mean_difference": m_e2e_saved["diff_pp"], "ci95": m_e2e_saved["ci95"]}, intervals_verified)

                    target_e2e_raw_p.append(p_val)
                    target_e2e_refs.append(m_e2e_saved)

    # Verify Family B & Family C Holm adjustments
    assert len(target_raw_p) == 54
    assert len(target_e2e_raw_p) == 54
    exp_target_holm = holm_adjust(target_raw_p)
    exp_e2e_holm = holm_adjust(target_e2e_raw_p)

    for m_saved, h_exp in zip(target_test_refs, exp_target_holm):
        assert abs(m_saved["holm_adjusted_p"] - h_exp) < 1e-12
    for m_saved, h_exp in zip(target_e2e_refs, exp_e2e_holm):
        assert abs(m_saved["holm_adjusted_p"] - h_exp) < 1e-12

    print(f"  Verified all 18 target-source comparisons, gate decompositions, and Holm adjustments.")

    return intervals_verified, exact_tests_verified, budget_raw_p, target_raw_p, target_e2e_raw_p


def main():
    print("=" * 80)
    print("Independent Verifier: Stage 3 Solver-Budget Matched Comparisons (v2)")
    print("=" * 80)

    manifest_path = OUT_DIR / "input_manifest.json"
    results_path = OUT_DIR / "comparison_results.json"
    spec_path = OUT_DIR / "specification.json"

    assert manifest_path.exists(), f"Missing manifest: {manifest_path}"
    assert results_path.exists(), f"Missing results: {results_path}"
    assert spec_path.exists(), f"Missing specification: {spec_path}"

    manifest = read_json(manifest_path)
    results = read_json(results_path)
    spec = read_json(spec_path)

    # 1. Audit input manifest hashes against raw disk bytes
    print("\n[Step 1] Auditing input manifest SHA-256 hashes...")
    for rel_path, expected_hash in manifest.items():
        actual_hash = sha(ROOT / rel_path)
        assert actual_hash == expected_hash, f"Hash mismatch for {rel_path}: {actual_hash} != {expected_hash}"
    print(f"  All {len(manifest)} input files verified byte-for-byte.")

    # 2. Load recomputed rows and augment energy errors independently
    rows = read_json(AUDIT_V3_DIR / "recomputed_LOCAL.json")
    assert len(rows) == 5400

    target_calories_map = {}
    for case_file in TARGETS_DIR.glob("case_*.json"):
        d = read_json(case_file)
        target_calories_map[(d["case"], d["arm"])] = float(d["energy_prescription"]["target_calories"])

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

    intervals_verified, exact_tests_verified, budget_raw_p, target_raw_p, target_e2e_raw_p = validate_statistics(rows, results, idx_map)

    # 6. Test synthetic failure fixtures
    print("\n[Step 5] Testing synthetic failure fixtures against verifier...")
    fixtures_passed = test_failure_fixtures(rows, results, idx_map)
    for fx in fixtures_passed:
        print(f"  [PASS] {fx}")
    assert len(fixtures_passed) == 6

    # 7. Write verification receipt
    receipt = {
        "status": "VERIFIED_PASSED",
        "study_name": "stage3_budget_matched_comparisons_v1",
        "total_solves_verified": len(rows),
        "total_manifest_files_verified": len(manifest),
        "total_bootstrap_intervals_verified": intervals_verified[0],
        "total_exact_tests_verified": exact_tests_verified,
        "multiplicity_families_verified": {
            "family_a_budget": len(budget_raw_p),
            "family_b_target_source_accepted": len(target_raw_p),
            "family_c_target_source_e2e": len(target_e2e_raw_p),
        },
        "failure_fixtures_tested": len(fixtures_passed),
        "failure_fixture_details": fixtures_passed,
        "scope": "Statistical replay of existing reconstructed rows; not a fresh catalog or solver audit.",
        "original_evidence_unchanged": True,
        "solver_calls": 0,
        "results_json_sha256": sha(results_path),
        "specification_sha256": sha(spec_path),
        "verifier_script_sha256": sha(Path(__file__)),
    }

    save_json(OUT_DIR / "verification.json", receipt)
    print(f"\nSaved verification.json receipt: {receipt}")
    print("\n" + "=" * 80)
    print("ALL MATCHED COMPARISON VERIFICATION CHECKS PASSED WITH ZERO DISCREPANCIES.")
    print("=" * 80)


if __name__ == "__main__":
    main()
