#!/usr/bin/env python3
"""Unified BioNutriGPT End-to-End Clinical Inference Pipeline: Stages 1, 2, and 3.

Executes the complete three-stage BioNutriGPT architecture in a single command:
  1. Stage 1: Deep Cross-Attention Diagnostics (T2D, Obesity, CVD risks & attention attribution)
  2. Stage 2: Gradient-Guided Counterfactual Optimization & 4 Thermodynamic Safety Filters
  3. Stage 3: Discrete Indian Meal Synthesis (HiGHS MILP, ICMR-NIN IFCT 2017 & RecipeDB schema)

Usage:
  python infer_bionutrigpt_pipeline_Stage_1_2_and_3.py [patient_file] [options]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import infer_bionutrigpt_pipeline_Stage_1_and_2 as p12
import stage3_milp_solver as s3

DEFAULT_PATIENT = PROJECT_DIR / "sample_patient_single_line.txt"
DEFAULT_OUTPUT = PROJECT_DIR / "bionutrigpt_pipeline_dossier.json"


def render_full_pipeline_report(
    p12_report: str,
    stage3_output: s3.Stage3Output,
) -> str:
    """Combine Stage 1 & 2 diagnostic report with Stage 3 Indian meal synthesis."""
    if not stage3_output.accepted:
        return p12_report + '\n\nStage 3: no accepted meal plan\n' + json.dumps(stage3_output.failure, indent=2)
    lines = [p12_report, ""]
    lines.append("=" * 92)
    lines.append("STAGE 3 | DISCRETE INDIAN MEAL SYNTHESIS (HiGHS MILP)")
    lines.append("=" * 92)
    lines.append(f"{'Dish Name':34} {'Servings':>8} {'Household Unit':24} {'Total Weight':>14}")
    lines.append("-" * 92)
    for item in stage3_output.solved_food_basket:
        lines.append(
            f"{item['name'][:34]:34} {item['quantity']:>8} {item['serving_unit'][:24]:24} {item['total_weight_g']:>12.1f} g"
        )
    lines.append("-" * 92)
    
    delivered = {r["nutrient"]: r["delivered"] for r in stage3_output.delivered_nutrition}
    target = {r["nutrient"]: r["target"] for r in stage3_output.delivered_nutrition}
    lines.append("Delivered Macronutrients & Clinical Key Indicators:")
    lines.append(
        f"  Calories (DR1TKCAL)  : {delivered['DR1TKCAL']:>8.1f} kcal (Target: {target['DR1TKCAL']:.1f} kcal, Error: {stage3_output.solver_metadata['calorie_error_pct']:.2f}%)"
    )
    lines.append(
        f"  Protein  (DR1TPROT)  : {delivered['DR1TPROT']:>8.1f} g    (Target: {target['DR1TPROT']:.1f} g)"
    )
    lines.append(
        f"  Carbs    (DR1TCARB)  : {delivered['DR1TCARB']:>8.1f} g    (Target: {target['DR1TCARB']:.1f} g)"
    )
    lines.append(
        f"  Fiber    (DR1TFIBE)  : {delivered['DR1TFIBE']:>8.1f} g    (Target: {target['DR1TFIBE']:.1f} g)"
    )
    lines.append(
        f"  Sat Fat  (DR1TSFAT)  : {delivered['DR1TSFAT']:>8.1f} g    (Target: {target['DR1TSFAT']:.1f} g)"
    )
    lines.append(
        f"  Sodium   (DR1TSODI)  : {delivered['DR1TSODI']:>8.1f} mg   (Target: {target['DR1TSODI']:.1f} mg)"
    )
    lines.append(
        f"  Potassium(DR1TPOTA)  : {delivered['DR1TPOTA']:>8.1f} mg   (Target: {target['DR1TPOTA']:.1f} mg)"
    )
    
    cv = stage3_output.clinical_verification
    lines.append("\nStage 3 Clinical Safety Verification:")
    for check_name in ("calories", "macronutrient_split", "sodium_to_potassium_ratio", "fiber_adequacy", "nutrient_envelope"):
        res = cv[check_name]
        status = "[PASS]" if res["passed"] else "[WARN]"
        lines.append(f"  {status:7} {check_name.replace('_', ' ').title()}")

    meta = stage3_output.solver_metadata
    lines.append(f"\nSolver Performance: {meta['backend']} in {meta['solve_time_ms']:.2f} ms | Status: {meta['status']}")
    lines.append("=" * 92)
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run complete BioNutriGPT Stage 1, Stage 2, and Stage 3 clinical inference pipeline."
    )
    parser.add_argument(
        "patient_file",
        nargs="?",
        type=Path,
        default=DEFAULT_PATIENT,
        help=f"Patient vector (.txt/.csv/.npy/.json); default: {DEFAULT_PATIENT}",
    )
    parser.add_argument("--weight", type=float, default=None, help="Patient weight in kg.")
    parser.add_argument("--height", type=float, default=None, help="Patient height in cm.")
    parser.add_argument("--age", type=float, default=None, help="Patient age in years.")
    sex_group = parser.add_mutually_exclusive_group()
    sex_group.add_argument("--female", action="store_true", help="Override inferred sex as female.")
    sex_group.add_argument("--male", action="store_true", help="Override inferred sex as male.")
    parser.add_argument(
        "--activity",
        choices=p12.ACTIVITY_LEVELS,
        default=None,
        help="Activity level (default: sedentary).",
    )
    parser.add_argument("--food-db", type=str, default=None, help="Optional custom IFCT/RecipeDB food catalog path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output unified JSON dossier; default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress formatted clinical console report.")
    args = parser.parse_args(argv)

    patient_path = Path(args.patient_file).expanduser()
    output_path = Path(args.output).expanduser()

    raw_vector = p12.parse_patient_vector(patient_path)
    patient_metadata = p12.resolve_patient_metadata(args, raw_vector, patient_path)

    start_total = time.perf_counter()

    # Stage 1: Diagnostic Risk Inference
    stage1_assessment, _ = p12.run_stage1_diagnostics(raw_vector)

    # Stage 2: Gradient-Guided Counterfactual Optimization
    stage2_dossier = p12.run_stage2_counterfactual(raw_vector, patient_metadata)

    # Build Stage 1+2 unified dossier
    unified_dossier = p12.build_unified_dossier(
        raw_vector=raw_vector,
        patient_metadata=patient_metadata,
        stage1_assessment=stage1_assessment,
        stage2_dossier=stage2_dossier,
    )

    stage3_output = None
    if unified_dossier.get("stage3_certified", False):
        food_db = s3.load_food_database(args.food_db)
        stage3_output = s3.solve_meal_plan(stage2_dossier, food_db)
        unified_dossier["stage3_meal_plan"] = stage3_output.to_dict()
    else:
        unified_dossier["stage3_meal_plan"] = None

    elapsed_total_ms = (time.perf_counter() - start_total) * 1000.0
    unified_dossier["total_pipeline_time_ms"] = round(elapsed_total_ms, 2)

    p12._atomic_write_json(unified_dossier, output_path)

    if not args.quiet:
        p12_report = p12.render_clinical_report(
            raw_vector=raw_vector,
            patient_metadata=patient_metadata,
            stage1_assessment=stage1_assessment,
            stage2_dossier=stage2_dossier,
            stage3_certified=unified_dossier["stage3_certified"],
        )
        if stage3_output is not None:
            full_report = render_full_pipeline_report(p12_report, stage3_output)
            print(full_report)
        else:
            print(p12_report)
            print("\n[!] Stage 3 discrete meal plan was skipped because Stage 2 safety filters were not certified.")
        print(f"\nUnified 3-Stage JSON dossier saved to: {output_path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
