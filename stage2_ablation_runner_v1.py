#!/usr/bin/env python3
"""BioNutriGPT Stage 2 Component Ablation Runner (v1).

Executes the small engineering pilot (N=3 participants x 6 variants = 18 checkpoints)
and exports atomic checkpoints with complete intermediate audit traces.
Prevents unprompted execution of the full 100-participant evaluation study.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import stage1_deployment as deployment
import stage2_counterfactual_engine as s2
import stage2_ablation_engine_v1 as ablation

OUT_DIR = ROOT / "Saved Model" / "stage2_ablation_v1"
PILOT_DIR = OUT_DIR / "pilot"
PILOT_CHECKPOINT_DIR = PILOT_DIR / "checkpoints"


def sha(p: Path | str) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    temp_path.replace(path)


def run_pilot():
    PILOT_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    pilot_cases_path = OUT_DIR / "development_participants_LOCAL.json"
    if not pilot_cases_path.exists():
        raise FileNotFoundError(f"Pilot cases not found at {pilot_cases_path}")
    cases = json.loads(pilot_cases_path.read_text(encoding="utf-8"))

    variant_manifest_path = OUT_DIR / "variant_manifest.json"
    if not variant_manifest_path.exists():
        raise FileNotFoundError(f"Variant manifest not found at {variant_manifest_path}")
    variants = json.loads(variant_manifest_path.read_text(encoding="utf-8"))

    input_manifest_path = OUT_DIR / "input_manifest.json"
    manifest_sha = sha(input_manifest_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Stage 2 Ablation Pilot on device: {device}")
    artifacts = s2._load_preprocessing_artifacts(deployment.PREPROCESSOR)
    model = s2._load_stage1_model(deployment.CHECKPOINT, artifacts, device)

    # 3 cases x 6 variants = 18 tasks
    completed = 0
    t0_all = time.perf_counter()

    for case in cases:
        case_idx = case["case"]
        seqn = case["SEQN"]

        for v_id, v_config in variants.items():
            ckpt_name = f"case_{case_idx:03d}_{seqn}_{v_id}.json"
            ckpt_path = PILOT_CHECKPOINT_DIR / ckpt_name

            if ckpt_path.exists():
                print(f"Skipping existing checkpoint: {ckpt_name}")
                completed += 1
                continue

            print(f"Running Case {case_idx} (SEQN {seqn}) - {v_id}...")
            rec = ablation.run_ablation_variant(
                case=case,
                variant_config=v_config,
                model=model,
                artifacts=artifacts,
                device=device,
            )
            rec["manifest_sha256"] = manifest_sha
            rec["device"] = str(device)
            write_json_atomic(ckpt_path, rec)
            completed += 1

    dt_total = time.perf_counter() - t0_all
    print(f"\nPilot run complete. Total checkpoints: {completed}/18 in {dt_total:.2f}s")

    # Generate analysis summary
    analysis = generate_pilot_analysis()
    analysis_path = PILOT_DIR / "analysis.json"
    write_json_atomic(analysis_path, analysis)
    print(f"Pilot analysis exported to {analysis_path}")


def generate_pilot_analysis() -> dict[str, Any]:
    checkpoints = sorted(PILOT_CHECKPOINT_DIR.glob("case_*.json"))
    if len(checkpoints) != 18:
        raise RuntimeError(f"Expected 18 pilot checkpoints, found {len(checkpoints)}")

    data_by_variant: dict[str, list[dict]] = {}
    for p in checkpoints:
        d = json.loads(p.read_text(encoding="utf-8"))
        v_id = d["variant_id"]
        data_by_variant.setdefault(v_id, []).append(d)

    summary_by_variant: dict[str, Any] = {}
    for v_id, records in data_by_variant.items():
        n = len(records)
        num_pass = sum(1 for r in records if r["evaluation"]["numerical_pass"])
        gate_pass = sum(1 for r in records if r["evaluation"]["score_gate_pass"])
        overall = sum(1 for r in records if r["evaluation"]["overall_accepted"])
        l1_dists = [r["evaluation"]["distance_from_observed_l1_norm"] for r in records]
        l2_dists = [r["evaluation"]["distance_from_observed_l2_norm"] for r in records]
        times = [r["timing_ms"]["total_ms"] for r in records]

        summary_by_variant[v_id] = {
            "n": n,
            "numerical_pass_rate": num_pass / n,
            "numerical_pass_count": f"{num_pass}/{n}",
            "score_gate_pass_rate": gate_pass / n,
            "score_gate_pass_count": f"{gate_pass}/{n}",
            "overall_accepted_rate": overall / n,
            "overall_accepted_count": f"{overall}/{n}",
            "mean_l1_norm_distance": float(np.mean(l1_dists)),
            "mean_l2_norm_distance": float(np.mean(l2_dists)),
            "mean_runtime_ms": float(np.mean(times)),
            "cases": [
                {
                    "case": r["case"],
                    "SEQN": r["SEQN"],
                    "numerical_pass": r["evaluation"]["numerical_pass"],
                    "score_gate_pass": r["evaluation"]["score_gate_pass"],
                    "overall_accepted": r["evaluation"]["overall_accepted"],
                    "l1_dist": r["evaluation"]["distance_from_observed_l1_norm"],
                    "runtime_ms": r["timing_ms"]["total_ms"],
                    "max_bound_residual": r["residuals"]["max_bound_residual"],
                    "max_linear_residual": r["residuals"]["max_linear_residual"],
                    "score_deltas": r["evaluation"]["score_deltas"],
                }
                for r in records
            ],
        }

    # Contrasts against Variant 0
    ref_records = {r["SEQN"]: r for r in data_by_variant["variant_0_reference"]}
    contrasts: dict[str, Any] = {}
    for v_id, records in data_by_variant.items():
        if v_id == "variant_0_reference":
            continue
        diff_overall = []
        diff_l1 = []
        diff_time = []
        for r in records:
            ref = ref_records[r["SEQN"]]
            diff_overall.append(int(r["evaluation"]["overall_accepted"]) - int(ref["evaluation"]["overall_accepted"]))
            diff_l1.append(r["evaluation"]["distance_from_observed_l1_norm"] - ref["evaluation"]["distance_from_observed_l1_norm"])
            diff_time.append(r["timing_ms"]["total_ms"] - ref["timing_ms"]["total_ms"])

        contrasts[f"{v_id}_minus_ref"] = {
            "n": len(records),
            "paired_overall_acceptance_diff_pp": float(np.mean(diff_overall) * 100.0),
            "paired_mean_l1_distance_diff": float(np.mean(diff_l1)),
            "paired_mean_runtime_diff_ms": float(np.mean(diff_time)),
            "individual_l1_diffs": diff_l1,
        }

    return {
        "status": "pilot_completed",
        "date": "2026-11-24",
        "pilot_sample_size": 3,
        "variants_count": 6,
        "total_checkpoints": len(checkpoints),
        "disclaimer": "Pilot statistics are mechanical sanity checks on N=3 development profiles, not population evidence.",
        "summary_by_variant": summary_by_variant,
        "paired_contrasts": contrasts,
    }


def main():
    parser = argparse.ArgumentParser(description="Stage 2 Ablation Runner v1")
    parser.add_argument("--run-pilot", action="store_true", help="Execute the 3-participant engineering pilot")
    parser.add_argument("--run-full", action="store_true", help="Placeholder for full 100-case evaluation (prohibited)")
    args = parser.parse_args()

    if args.run_full:
        raise RuntimeError(
            "Full 100-case evaluation is prohibited in this task. Only the small engineering pilot is authorized."
        )

    if args.run_pilot:
        run_pilot()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
