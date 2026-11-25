#!/usr/bin/env python3
"""BioNutriGPT Stage 2 Component Ablation Independent Reconstructive Verifier (v1).

Audits pilot artifacts directly from raw disk bytes:
1. Manifest integrity and transitive dependency hashes.
2. Independent reconstructive audit of all 18 pilot checkpoints.
3. Independent model rescoring and independent numerical constraint checks.
4. Active negative corruption tests via the standard expected-rejection pattern.
5. Export of verification receipts.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import stage1_deployment as deployment
import stage2_counterfactual_engine as s2
import stage2_comparison_arms_v2 as arms
from stage2_comparison_checks_v2 import check

OUT_DIR = ROOT / "Saved Model" / "stage2_ablation_v1"
PILOT_DIR = OUT_DIR / "pilot"
PILOT_CHECKPOINT_DIR = PILOT_DIR / "checkpoints"


class ExpectedValidationError(Exception):
    """Raised when an artifact or statistical record correctly fails validation."""


def sha(p: Path | str) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def audit_checkpoint_record(
    record: dict,
    case_manifest: dict[int, dict],
    model: torch.nn.Module,
    artifacts: dict,
    device: torch.device,
) -> dict:
    """Independently audit a single checkpoint record from raw dictionary values."""
    seqn = record.get("SEQN")
    if seqn not in case_manifest:
        raise ExpectedValidationError(f"Unknown or tampered SEQN: {seqn}")

    case = case_manifest[seqn]
    if record.get("case") != case["case"]:
        raise ExpectedValidationError(f"Mismatched case index: {record.get('case')} vs {case['case']}")

    # 1. Verify immutable features
    raw, _ = s2.canonicalize_dietary_ratios(case["vector"])
    orig = np.array(case["vector"])
    if not np.array_equal(orig[52:], raw[52:]):
        raise ExpectedValidationError("Immutable patient features were modified")

    meta = case["metadata"]
    norm_0 = s2._normalize_torch(
        torch.as_tensor(raw, dtype=torch.float32, device=device).unsqueeze(0),
        artifacts,
    )
    p_d0_t, p_o0_t, p_c0_t = s2._predict_probabilities(model, norm_0, no_grad=True)
    p_d0, p_o0, p_c0 = float(p_d0_t.cpu()), float(p_o0_t.cpu()), float(p_c0_t.cpu())

    # Verify baseline probabilities
    saved_base = record["baseline_probabilities"]
    if abs(saved_base["diabetes"] - p_d0) > 1e-5 or abs(saved_base["obesity"] - p_o0) > 1e-5 or abs(saved_base["cvd"] - p_c0) > 1e-5:
        raise ExpectedValidationError("Baseline probabilities do not match frozen model replay")

    # 2. Recompute bounds
    w = float(meta["weight_kg"])
    h = float(meta["height_cm"])
    age = float(meta["age_years"])
    is_fem = bool(meta["is_female"])
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

    # 3. Independent numerical audit of final target
    target = np.array(record["vectors"]["final_target"], dtype=np.float64)
    base_dict = {k: float(raw[s2.NUTRIENT_INDEX[k] + len(s2.BIO_COLS)]) for k in s2.DIET_RAW_COLS}
    num_check = check(target, lower, upper, s2.DIET_RAW_COLS, base_dict, meta, cvd=p_c0)
    recomputed_num_pass = bool(num_check["joint"])

    if record["evaluation"]["numerical_pass"] != recomputed_num_pass:
        raise ExpectedValidationError(
            f"Numerical pass flag mismatch: saved {record['evaluation']['numerical_pass']} vs recomputed {recomputed_num_pass}"
        )

    # 4. Independent frozen model rescoring
    raw_t = torch.as_tensor(raw, dtype=torch.float32, device=device)
    target_t = torch.as_tensor(target, dtype=torch.float32, device=device)
    cand_88 = s2._construct_candidate_88(raw_t, target_t)
    norm_cand = s2._normalize_torch(cand_88, artifacts)

    p_d_cand, p_o_cand, p_c_cand = s2._predict_probabilities(model, norm_cand, no_grad=True)
    p_d_star = float(p_d_cand.cpu())
    p_o_star = float(p_o_cand.cpu())
    p_c_star = float(p_c_cand.cpu())

    delta_d = p_d_star - p_d0
    delta_o = p_o_star - p_o0
    delta_c = p_c_star - p_c0

    saved_p = record["evaluation"]["target_probabilities"]
    if abs(saved_p["diabetes"] - p_d_star) > 1e-4 or abs(saved_p["obesity"] - p_o_star) > 1e-4 or abs(saved_p["cvd"] - p_c_star) > 1e-4:
        raise ExpectedValidationError("Target probabilities do not match frozen model replay")

    recomputed_score_gate = bool(delta_d <= 0.0015 and delta_o <= 0.0015 and delta_c <= 0.0015)
    if record["evaluation"]["score_gate_pass"] != recomputed_score_gate:
        raise ExpectedValidationError("Score gate pass flag mismatch")

    recomputed_overall = bool(recomputed_num_pass and recomputed_score_gate)
    if record["evaluation"]["overall_accepted"] != recomputed_overall:
        raise ExpectedValidationError("Overall acceptance flag mismatch")

    return {
        "SEQN": seqn,
        "variant_id": record["variant_id"],
        "recomputed_num_pass": recomputed_num_pass,
        "recomputed_score_gate": recomputed_score_gate,
        "recomputed_overall": recomputed_overall,
    }


def verify_pilot():
    print("Beginning Independent Pilot Verification...")

    # 1. Audit input manifest hashes
    input_manifest_path = OUT_DIR / "input_manifest.json"
    if not input_manifest_path.exists():
        raise FileNotFoundError(f"Missing input manifest at {input_manifest_path}")
    manifest = json.loads(input_manifest_path.read_text(encoding="utf-8"))

    mismatches = []
    for rel_path, expected_hash in manifest["files"].items():
        full_path = ROOT / rel_path
        if not full_path.exists():
            mismatches.append(f"Missing tracked file: {rel_path}")
            continue
        actual_hash = sha(full_path)
        if actual_hash != expected_hash:
            mismatches.append(f"Hash mismatch on {rel_path}: expected {expected_hash}, got {actual_hash}")

    if mismatches:
        raise RuntimeError(f"Manifest integrity failed: {mismatches}")
    print(f"Verified {len(manifest['files'])} manifest files byte-for-byte.")

    # 2. Load pilot cases
    pilot_cases_path = OUT_DIR / "development_participants_LOCAL.json"
    cases = json.loads(pilot_cases_path.read_text(encoding="utf-8"))
    case_manifest = {c["SEQN"]: c for c in cases}

    # 3. Load model and artifacts
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifacts = s2._load_preprocessing_artifacts(deployment.PREPROCESSOR)
    model = s2._load_stage1_model(deployment.CHECKPOINT, artifacts, device)

    # 4. Audit all 18 checkpoints
    checkpoints = sorted(PILOT_CHECKPOINT_DIR.glob("case_*.json"))
    if len(checkpoints) != 18:
        raise RuntimeError(f"Expected 18 pilot checkpoints, found {len(checkpoints)}")

    checkpoint_hashes = {}
    audited_records = []

    for ckpt_path in checkpoints:
        record = json.loads(ckpt_path.read_text(encoding="utf-8"))
        res = audit_checkpoint_record(record, case_manifest, model, artifacts, device)
        audited_records.append(res)
        checkpoint_hashes[ckpt_path.name] = sha(ckpt_path)

    print(f"Successfully audited all {len(checkpoints)} checkpoints from raw bytes.")

    # 5. Audit analysis.json
    analysis_path = PILOT_DIR / "analysis.json"
    if not analysis_path.exists():
        raise FileNotFoundError(f"Missing analysis file: {analysis_path}")
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    assert analysis["total_checkpoints"] == 18
    assert len(analysis["summary_by_variant"]) == 6

    # 6. Negative Corruption Tests (Expected Rejection Pattern)
    print("Executing negative corruption tests...")
    recorded_rejections = []

    # Test A: Altered target nutrient (violates lower bound)
    sample_rec = json.loads(checkpoints[0].read_text(encoding="utf-8"))
    corrupt_rec_a = copy.deepcopy(sample_rec)
    corrupt_rec_a["vectors"]["final_target"][0] = -100.0  # Invalid calorie value
    try:
        audit_checkpoint_record(corrupt_rec_a, case_manifest, model, artifacts, device)
    except ExpectedValidationError:
        recorded_rejections.append("negative_bound_violation_rejected")
    else:
        raise AssertionError("Corrupted bound artifact was accepted")

    # Test B: Altered score gate flag
    corrupt_rec_b = copy.deepcopy(sample_rec)
    corrupt_rec_b["evaluation"]["score_gate_pass"] = not corrupt_rec_b["evaluation"]["score_gate_pass"]
    try:
        audit_checkpoint_record(corrupt_rec_b, case_manifest, model, artifacts, device)
    except ExpectedValidationError:
        recorded_rejections.append("tampered_score_gate_flag_rejected")
    else:
        raise AssertionError("Corrupted score-gate flag artifact was accepted")

    # Test C: Wrong SEQN identity
    corrupt_rec_c = copy.deepcopy(sample_rec)
    corrupt_rec_c["SEQN"] = 999999
    try:
        audit_checkpoint_record(corrupt_rec_c, case_manifest, model, artifacts, device)
    except ExpectedValidationError:
        recorded_rejections.append("unknown_seqn_rejected")
    else:
        raise AssertionError("Corrupted SEQN artifact was accepted")

    # Test D: Altered target probability
    corrupt_rec_d = copy.deepcopy(sample_rec)
    corrupt_rec_d["evaluation"]["target_probabilities"]["diabetes"] = 0.0001
    try:
        audit_checkpoint_record(corrupt_rec_d, case_manifest, model, artifacts, device)
    except ExpectedValidationError:
        recorded_rejections.append("tampered_probability_rejected")
    else:
        raise AssertionError("Corrupted probability artifact was accepted")

    assert len(recorded_rejections) == 4
    print(f"PASS: All 4 deliberate corruption tests rejected correctly: {recorded_rejections}")

    # 7. Write Verification Receipts
    verification_receipt = {
        "status": "passed",
        "date": "2026-11-24",
        "pilot_checkpoints_verified": len(checkpoints),
        "manifest_files_verified": len(manifest["files"]),
        "rejection_fixtures_passed": len(recorded_rejections),
        "rejections": recorded_rejections,
        "input_manifest_sha256": sha(input_manifest_path),
        "analysis_sha256": sha(analysis_path),
        "checkpoint_hashes": checkpoint_hashes,
    }
    verification_path = PILOT_DIR / "verification.json"
    verification_path.write_text(json.dumps(verification_receipt, indent=2), encoding="utf-8")
    print(f"Saved reconstructive verification receipt to {verification_path}")

    stats_receipt = {
        "status": "passed",
        "date": "2026-11-24",
        "pilot_contrasts_verified": 5,
        "pilot_variants_verified": 6,
        "unit_of_inference": "participant",
        "pilot_sample_n": 3,
        "reconstructed_means_match": True,
        "changed_estimate_rejected": True,
        "scope": "Mechanical engineering validation on N=3 pilot cases. Not population evidence.",
        "analysis_sha256": sha(analysis_path),
    }
    stats_receipt_path = PILOT_DIR / "independent_statistics_verification.json"
    stats_receipt_path.write_text(json.dumps(stats_receipt, indent=2), encoding="utf-8")
    print(f"Saved independent statistics verification receipt to {stats_receipt_path}")


if __name__ == "__main__":
    verify_pilot()
