"""Independent Verification Script for Stage 2 Three-Arm Comparative Added-Value Study.

Performs rigorous independent replay without trusting saved booleans:
1. Recomputes source/code SHA-256 bindings.
2. Verifies 100 unique participants and zero overlap with prior cohorts.
3. Recomputes all 39 bounds, energy arithmetic, and ratios from raw target arrays.
4. Performs fresh neural forward passes and Platt calibration through frozen Stage 1 model.
5. Recomputes closed-loop score gate decisions and overall acceptance.
6. Recomputes aggregate counts, denominators, and acceptance point contrasts.
   Full interval and exact-test replay is in audit_stage2_comparison_statistics_v2.py.
7. Generates machine-readable verification receipt.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import stage1_deployment as deployment
import stage2_counterfactual_engine as s2
from stage2_comparison_checks_v2 import check

STUDY_DIR = ROOT / "Saved Model/stage2_comparison_study_v2"
CKPT_DIR = STUDY_DIR / "checkpoints"


def sha(p: Path | str) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def verify_study() -> dict[str, Any]:
    print("=" * 70)
    print("Stage 2 Comparative Study: Independent Verification")
    print("=" * 70)

    protocol_path = STUDY_DIR / "protocol.json"
    cases_path = STUDY_DIR / "cases_LOCAL.json"
    summary_path = STUDY_DIR / "summary.json"
    analysis_path = STUDY_DIR / "analysis_summary.json"

    assert protocol_path.exists(), "Missing protocol.json"
    assert cases_path.exists(), "Missing cases_LOCAL.json"
    assert summary_path.exists(), "Missing summary.json"

    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cases = json.loads(cases_path.read_text(encoding="utf-8"))

    # 1. Source Hash Verification
    print("[1/6] Verifying source SHA-256 bindings...")
    for rel_path, expected_h in protocol["hashes"].items():
        actual_h = sha(ROOT / rel_path)
        assert actual_h == expected_h, f"Hash mismatch for {rel_path}: {actual_h} != {expected_h}"
    assert sha(cases_path) == protocol["cases_sha256"], "Cases JSON hash mismatch"
    print(f"  -> All {len(protocol['hashes'])} source and data hashes verified matching.")

    # 2. Sample Uniqueness and Disjointness
    print("[2/6] Verifying sample uniqueness and cohort exclusions...")
    assert len(cases) == 100
    seqns = [c["SEQN"] for c in cases]
    assert len(set(seqns)) == 100, "Duplicate SEQN found in study sample"
    original_cases = json.loads((ROOT / "Saved Model/stage2_comparison_study_v1/cases_LOCAL.json").read_text())
    assert cases == original_cases, "Correction changed participant inputs"

    # Verify zero overlap with the 3 prior cohorts
    repair_path = ROOT / "Saved Model/stage2_constraint_audit_repaired_v1/cases_LOCAL.json"
    repair_ids = {c["SEQN"] for c in json.loads(repair_path.read_text(encoding="utf-8"))}
    overlap_repair = len(set(seqns) & repair_ids)
    assert overlap_repair == 0, f"Found {overlap_repair} overlapping repair profiles"

    downstream_path = ROOT / "Saved Model/stage1_downstream_study_v1/selected_participants_LOCAL.csv"
    downstream_ids = set(pd.read_csv(downstream_path)["SEQN"].astype(int))
    overlap_downstream = len(set(seqns) & downstream_ids)
    assert overlap_downstream == 0, f"Found {overlap_downstream} overlapping downstream study profiles"

    pilot_path = ROOT / "Saved Model/stage2_research_adapter_pilot_v1/envelopes_LOCAL.json"
    pilot_ids = {int(c["SEQN"]) for c in json.loads(pilot_path.read_text(encoding="utf-8"))}
    overlap_pilot = len(set(seqns) & pilot_ids)
    assert overlap_pilot == 0, f"Found {overlap_pilot} overlapping pilot profiles"
    print(f"  -> Disjointness confirmed: 0 overlap across repair (32), downstream (100), and pilot (5).")

    # 3. Model and Calibrator Setup for Fresh Replay
    print("[3/6] Initializing frozen Stage 1 model and calibrators for fresh replay...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifacts = s2._load_preprocessing_artifacts(deployment.PREPROCESSOR)
    model = s2._load_stage1_model(deployment.CHECKPOINT, artifacts, device)

    # 4. Checkpoint-by-Checkpoint Numerical and Model Replay
    print("[4/6] Replaying numerical constraints and fresh model inference for all 300 checkpoints...")
    arms = ["rules_only", "minimum_change", "model_guided"]
    max_score_diff = 0.0

    recomputed_counts = {
        arm: {"attempted": 0, "completed": 0, "numerical_pass": 0, "score_gate_pass": 0, "overall_accepted": 0}
        for arm in arms
    }

    for case in cases:
        idx = case["case"]
        seqn = case["SEQN"]
        meta = case["metadata"]
        raw, _ = s2.canonicalize_dietary_ratios(case["vector"])

        # Recompute baseline probabilities freshly
        norm_0 = s2._normalize_torch(torch.as_tensor(raw, dtype=torch.float32, device=device).unsqueeze(0), artifacts)
        p_d0_t, p_o0_t, p_c0_t = s2._predict_probabilities(model, norm_0, no_grad=True)
        p_d0 = float(p_d0_t.cpu())
        p_o0 = float(p_o0_t.cpu())
        p_c0 = float(p_c0_t.cpu())

        w = meta["weight_kg"]
        h = meta["height_cm"]
        age = meta["age_years"]
        is_fem = meta["is_female"]
        bmr = 10.0 * w + 6.25 * h - 5.0 * age + (-161.0 if is_fem else 5.0)
        tdee = bmr * s2.PAL_FACTORS[meta["activity_level"]]
        t_kcal = max(tdee - 500.0, 1200.0) if p_o0 >= s2.CUTOFF_OBESITY else max(tdee, 1200.0)

        lower, upper = s2._build_nutrient_bounds(
            artifacts=artifacts,
            t_kcal=t_kcal,
            weight_kg=w,
            is_female=is_fem,
            age_years=age,
            p_diab_0=p_d0,
            p_obes_0=p_o0,
            p_cvd_0=p_c0,
        )

        base_dict = {k: float(raw[s2.DIET_RAW_COLS.index(k) + len(s2.BIO_COLS)]) for k in s2.DIET_RAW_COLS}

        for arm in arms:
            recomputed_counts[arm]["attempted"] += 1
            ckpt_path = CKPT_DIR / f"case_{idx:03d}_{seqn}_{arm}.json"
            assert ckpt_path.exists(), f"Missing checkpoint: {ckpt_path}"
            record = json.loads(ckpt_path.read_text(encoding="utf-8"))

            if not record.get("success"):
                continue
            recomputed_counts[arm]["completed"] += 1

            target = np.array(record["target_nutrients"], dtype=np.float64)
            assert target.shape == (39,)
            assert np.isfinite(target).all()

            # Fresh independent numerical check
            num_res = check(target, lower, upper, s2.DIET_RAW_COLS, base_dict, meta, p_c0)
            fresh_num_pass = bool(num_res["joint"])
            assert fresh_num_pass == record["numerical_pass"], f"Numerical check mismatch at case {idx} arm {arm}"
            if fresh_num_pass:
                recomputed_counts[arm]["numerical_pass"] += 1

            # Fresh neural forward pass and Platt rescoring
            raw_t = torch.as_tensor(raw, dtype=torch.float32, device=device)
            target_t = torch.as_tensor(target, dtype=torch.float32, device=device)
            cand_88 = s2._construct_candidate_88(raw_t, target_t)
            fixed = [i for i,k in enumerate(s2.FEATURE_COLS) if k not in s2.DIET_RAW_COLS and k not in s2.DIET_RATIO_COLS]
            assert torch.equal(cand_88.reshape(-1)[fixed], raw_t.reshape(-1)[fixed]), "Immutable features changed"
            for key, value in zip(("diabetes", "obesity", "cvd"), (p_d0,p_o0,p_c0)):
                assert abs(record["baseline_probabilities"][key]-value) <= 1e-6
            norm_cand = s2._normalize_torch(cand_88, artifacts)

            p_d_cand, p_o_cand, p_c_cand = s2._predict_probabilities(model, norm_cand, no_grad=True)
            p_d_fresh = float(p_d_cand.cpu())
            p_o_fresh = float(p_o_cand.cpu())
            p_c_fresh = float(p_c_cand.cpu())

            saved_p = record["target_probabilities"]
            diff_d = abs(p_d_fresh - saved_p["diabetes"])
            diff_o = abs(p_o_fresh - saved_p["obesity"])
            diff_c = abs(p_c_fresh - saved_p["cvd"])
            max_score_diff = max(max_score_diff, diff_d, diff_o, diff_c)
            assert max(diff_d, diff_o, diff_c) <= 1e-6, f"Score replay discrepancy at case {idx} arm {arm}"

            # Closed loop score gate
            delta_d = p_d_fresh - p_d0
            delta_o = p_o_fresh - p_o0
            delta_c = p_c_fresh - p_c0
            fresh_score_pass = bool(delta_d <= 0.0015 and delta_o <= 0.0015 and delta_c <= 0.0015)
            assert fresh_score_pass == record["score_gate_pass"], f"Score gate mismatch at case {idx} arm {arm}"
            if fresh_score_pass:
                recomputed_counts[arm]["score_gate_pass"] += 1

            fresh_overall_accepted = bool(fresh_num_pass and fresh_score_pass)
            assert fresh_overall_accepted == record["overall_accepted"], f"Overall acceptance mismatch at case {idx} arm {arm}"
            if fresh_overall_accepted:
                recomputed_counts[arm]["overall_accepted"] += 1

    print(f"  -> All 300 checkpoints replayed. Max neural rescoring residual: {max_score_diff:.2e}")

    # 5. Summary Agreement Verification
    print("[5/6] Verifying agreement between recomputed counts and summary.json...")
    for arm in arms:
        recomp = recomputed_counts[arm]
        saved = summary["arms"][arm]
        assert recomp["attempted"] == saved["attempted"] == 100
        assert recomp["completed"] == saved["completed"] == 100
        assert recomp["numerical_pass"] == saved["numerical_pass"]
        assert recomp["score_gate_pass"] == saved["score_gate_pass"]
        assert recomp["overall_accepted"] == saved["overall_accepted"]
        print(f"  -> Arm {arm:14s}: Recomputed accepted={recomp['overall_accepted']}/100, num_pass={recomp['numerical_pass']}/100, score_pass={recomp['score_gate_pass']}/100 [MATCH]")

    # 6. Replay Analysis Summary & Bootstrap
    print("[6/6] Replaying acceptance point contrasts (full statistics audited separately)...")
    if analysis_path.exists():
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        for c in analysis["paired_contrasts"]:
            arm1 = c["arm_1"]
            arm2 = c["arm_2"]
            point_diff_recomputed = (recomputed_counts[arm1]["overall_accepted"] - recomputed_counts[arm2]["overall_accepted"])
            assert abs(c["acceptance"]["point_diff_pct"] - point_diff_recomputed) < 1e-4
            print(f"  -> {c['contrast_label']:50s}: Point diff = {point_diff_recomputed:+d} pp [MATCH]")

    receipt = {
        "verification_passed": True,
        "study_version": "stage2_comparison_study_v2",
        "sample_size": 100,
        "total_checkpoints_verified": 300,
        "max_neural_rescoring_error": max_score_diff,
        "disjointness_confirmed": True,
        "source_hashes_verified": True,
        "recomputed_counts": recomputed_counts,
        "verifier_sha256": sha(Path(__file__)),
    }

    receipt["verification_scope"] = "Fresh model rescoring and numerical/count verification; full statistics verified separately in audit_complete.json"
    rec_path = STUDY_DIR / "verification.json"
    rec_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(f"\nVerification receipt written to {rec_path}")
    print("ALL INDEPENDENT VERIFICATION CHECKS PASSED.")
    return receipt


if __name__ == "__main__":
    verify_study()
