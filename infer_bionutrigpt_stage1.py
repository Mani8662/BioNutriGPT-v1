#!/usr/bin/env python3
"""
BioNutriGPT Stage 1 — Command-Line Inference Engine
===================================================
Run interactive patient risk prediction from the command prompt.

Capabilities:
  1. Evaluate an individual patient via interactive CLI prompts.
  2. Evaluate a patient from a JSON profile file (`--input-json patient.json`).
  3. Evaluate a batch of patients from a CSV (`--input-csv patients.csv`).
  4. Evaluate a random sample patient from the 2017-2020 test set (`--demo`).

Outputs:
  - Predicted disease probabilities: P(Diabetes), P(Obesity), P(CVD)
  - Risk categorization (Low, Moderate, High)
  - Top driving biomarkers, lifestyle factors, and nutrients via Transformer attention weights
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir))
import train_bionutrigpt_stage1 as s1


DEFAULT_CHECKPOINT = script_dir / "Saved Model" / "bionutrigpt_stage1_best.pt"
DEFAULT_TEST_CSV = script_dir / "Data" / "BioNutriGPT_test_2017_2020_ml_ready.csv"
DEFAULT_ARTIFACT = script_dir / "Data" / "preprocessing_artifacts.joblib"

# Pre-computed optimal clinical decision thresholds from the 2017-2020 prospective evaluation
CLINICAL_THRESHOLDS = {
    "diabetes": 0.617,
    "obesity": 0.445,
    "cvd": 0.603,
}


def load_model(checkpoint_path: Path, device: torch.device) -> s1.BioNutriGPTStage1:
    """Loads the trained BioNutriGPT Stage 1 weights into evaluation mode."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Trained model checkpoint not found: {checkpoint_path}")
    model = s1.BioNutriGPTStage1().to(device)
    s1.load_state_dict_checkpoint(model, checkpoint_path, device)
    model.eval()
    return model


def parse_single_line_txt(file_path: Path) -> np.ndarray:
    """Parses an 88-element feature vector from a single line in a text file.
    
    Supports:
      - Space-separated numbers: '5.2 56.0 178.0 ...'
      - Comma-separated numbers: '5.2, 56.0, 178.0, ...'
      - Python list style string: '[5.2, 56.0, 178.0, ...]'
    """
    if not file_path.is_file():
        raise FileNotFoundError(f"Input text file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    # Strip bracket enclosures and comments if present
    content = content.replace("[", " ").replace("]", " ")
    # Replace commas with spaces
    content = content.replace(",", " ")

    # Extract all floating point numbers
    tokens = [t.strip() for t in content.split() if t.strip() and not t.startswith("#")]
    values = []
    for token in tokens:
        try:
            values.append(float(token))
        except ValueError:
            continue

    if len(values) == 87:
        pass
    elif len(values) == 88:
        # Detected legacy 88-element vector with HbA1c (LBXGH) at position 0.
        # Drop it to match the repaired 87-feature zero-target-leakage model.
        print("[INFO] Detected legacy 88-feature vector. Dropping index 0 (LBXGH / HbA1c) to match repaired 87-feature zero-leakage model.")
        values = values[1:]
    else:
        raise ValueError(f"Expected 87 numbers (or legacy 88 with HbA1c) in '{file_path}', but found {len(values)} values.")

    return np.array(values, dtype=np.float32)


def preprocess_raw_vector(raw_vector: np.ndarray, artifact_path: Path) -> np.ndarray:
    """Preprocesses a raw clinical 87-feature vector if values are unnormalized."""
    if len(raw_vector) == 88:
        raw_vector = raw_vector[1:]
    # Check if vector looks already normalized (e.g., standard scores usually between -5 and +5)
    # If raw values contain Total Cholesterol (~180) or Calories (~2000), it's unnormalized.
    is_raw = bool(raw_vector[1] > 20.0 or raw_vector[5] > 100.0)
    if not is_raw:
        return raw_vector.copy()

    if not artifact_path.is_file():
        raise FileNotFoundError(f"Preprocessing artifact not found at {artifact_path}")

    import joblib
    art = joblib.load(artifact_path)
    df = pd.DataFrame([raw_vector], columns=s1.FEATURE_COLS)

    for c in art["log1p_cols"]:
        df[c] = np.log1p(np.maximum(df[c], 0))

    for c, (lb, ub) in art["clip_bounds"].items():
        df[c] = df[c].clip(lb, ub)

    c_cols = art["continuous_cols"]
    df[c_cols] = art["scaler"].transform(df[c_cols].values)
    return df.values[0].astype(np.float32)


def predict_patient(
    model: s1.BioNutriGPTStage1,
    feature_vector: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    """Runs a single 87-dimensional patient vector through the network."""
    expected_dim = len(s1.FEATURE_COLS)
    if feature_vector.shape != (expected_dim,):
        if feature_vector.shape == (88,):
            feature_vector = feature_vector[1:]
        else:
            raise ValueError(f"Expected feature vector of shape ({expected_dim},), got {feature_vector.shape}")

    x = torch.from_numpy(feature_vector.astype(np.float32)).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, _, attn = model(x)
        probs = {disease: float(torch.sigmoid(logits[disease]).item()) for disease in ["diabetes", "obesity", "cvd"]}

    # Extract top driving features per modality using learned attention weights
    top_drivers = {}
    for mod, col_list in [("bio", s1.BIO_COLS), ("diet", s1.DIET_COLS), ("life", s1.LIFE_COLS), ("demo", s1.DEMO_COLS)]:
        weights = attn[mod].squeeze(0).squeeze(-1).cpu().numpy()
        top_indices = np.argsort(weights)[::-1][:3]
        top_drivers[mod] = [{"feature": col_list[idx], "attention_pct": round(float(weights[idx] * 100), 2)} for idx in top_indices]

    results = {
        "probabilities": {k: round(v, 4) for k, v in probs.items()},
        "clinical_classifications": {
            disease: {
                "probability": round(probs[disease], 4),
                "threshold": CLINICAL_THRESHOLDS[disease],
                "high_risk": bool(probs[disease] >= CLINICAL_THRESHOLDS[disease]),
                "risk_tier": "High Risk" if probs[disease] >= CLINICAL_THRESHOLDS[disease] else ("Moderate Risk" if probs[disease] >= 0.35 else "Low Risk"),
            }
            for disease in ["diabetes", "obesity", "cvd"]
        },
        "top_modality_drivers": top_drivers,
    }
    return results


def print_patient_report(report: dict[str, Any], patient_id: str = "Anonymous Patient") -> None:
    """Prints a clean, formatted clinical report to the terminal."""
    print("\n" + "=" * 80)
    print(f"  BIONUTRIGPT STAGE 1: CLINICAL RISK ASSESSMENT REPORT")
    print(f"  Patient: {patient_id}")
    print("=" * 80)

    print("\n1. PREDICTED DISEASE RISK PROFILE:")
    print("-" * 80)
    print(f"{'Condition':<28} | {'Probability':<12} | {'Clinical Cutoff':<16} | {'Risk Tier'}")
    print("-" * 80)
    for disease, data in report["clinical_classifications"].items():
        name = "Type 2 Diabetes" if disease == "diabetes" else ("Clinical Obesity" if disease == "obesity" else "Cardiovascular Disease (CVD)")
        prob_str = f"{data['probability']*100:.1f}%"
        cut_str = f">= {data['threshold']*100:.1f}%"
        tier = data["risk_tier"]
        print(f"{name:<28} | {prob_str:<12} | {cut_str:<16} | {tier}")
    print("-" * 80)

    print("\n2. TOP EXPLANATORY DRIVERS (Transformer Attention):")
    print("-" * 80)
    for mod, drivers in report["top_modality_drivers"].items():
        title = "Blood Biomarkers" if mod == "bio" else ("Diet & Nutrition" if mod == "diet" else ("Lifestyle & Sleep" if mod == "life" else "Demographics"))
        print(f"  [{title}]")
        for d in drivers:
            print(f"    - {d['feature']:<25}: {d['attention_pct']:.2f}% attention")
    print("=" * 80 + "\n")


def run_interactive_cli(model: s1.BioNutriGPTStage1, device: torch.device) -> None:
    """Provides a step-by-step interactive intake in the terminal."""
    print("\nInteractive Patient Intake Mode")
    print("--------------------------------")
    print("This mode loads a median patient baseline and allows you to adjust key clinical parameters.\n")

    # Load baseline from test dataset median
    test_df = pd.read_csv(DEFAULT_TEST_CSV)
    features, _ = s1.validate_and_extract_arrays(test_df, "test", s1.EXPECTED_TEST_ROWS)
    patient_vec = np.median(features, axis=0).copy()

    def ask_float(prompt: str, default: float) -> float:
        val = input(f"{prompt} [Default: {default}]: ").strip()
        if not val:
            return default
        try:
            return float(val)
        except ValueError:
            print("Invalid input, using default.")
            return default

    # Key modifiable inputs
    chol_ratio_z = ask_float("Total / HDL Cholesterol ratio score (0.0 = average ~3.8, +2.0 = high risk ~6.0)", 0.0)
    patient_vec[s1.BIO_COLS.index("CHOL_HDL_RATIO")] = chol_ratio_z

    non_hdl_z = ask_float("Non-HDL Cholesterol score (0.0 = average ~140, +2.0 = high ~210)", 0.0)
    patient_vec[s1.BIO_COLS.index("NON_HDL_CHOL")] = non_hdl_z

    sugar_ratio_z = ask_float("Dietary Sugar-to-Fiber ratio score (-1.0 = low sugar/high fiber, +2.0 = high sugar)", 0.0)
    patient_vec[s1.FEATURE_COLS.index("SUGAR_TO_FIBER_RATIO")] = sugar_ratio_z

    sitting_z = ask_float("Daily Sitting Time score (-1.0 = active, +2.0 = highly sedentary)", 0.0)
    patient_vec[s1.FEATURE_COLS.index("PAD680")] = sitting_z

    smoking = input("Is the patient currently a smoker? (yes/no) [Default: no]: ").strip().lower()
    if smoking.startswith("y"):
        patient_vec[s1.FEATURE_COLS.index("SMOKE_NEVER")] = 0.0
        patient_vec[s1.FEATURE_COLS.index("SMOKE_EVERYDAY")] = 1.0

    report = predict_patient(model, patient_vec, device)
    print_patient_report(report, patient_id="Interactive CLI Patient")


def main() -> None:
    parser = argparse.ArgumentParser(description="BioNutriGPT Stage 1 Command-Line Inference")
    parser.add_argument("positional_input", nargs="?", type=Path, default=None, help="Optional direct path to input file (e.g. patient.txt or patient.json)")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="Path to best checkpoint .pt file")
    parser.add_argument("--demo", action="store_true", help="Run inference on a sample patient from the 2017-2020 test set")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive terminal prompt to test custom values")
    parser.add_argument("--input-txt", type=Path, default=None, help="Path to TXT file containing patient features on a single line (space/comma separated)")
    parser.add_argument("--input-json", type=Path, default=None, help="Path to JSON file containing patient feature array or dict")
    parser.add_argument("--input-csv", type=Path, default=None, help="Path to CSV containing preprocessed patients to score in batch")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional destination to save predictions as JSON")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)

    # Route positional input argument if provided without flags (e.g. python infer_bionutrigpt_stage1.py patient.txt)
    if args.positional_input:
        suffix = args.positional_input.suffix.lower()
        if suffix in [".txt", ".dat", ".vec"]:
            args.input_txt = args.positional_input
        elif suffix in [".json"]:
            args.input_json = args.positional_input
        elif suffix in [".csv"]:
            args.input_csv = args.positional_input
        else:
            # Default to text parser if extension is generic or omitted
            args.input_txt = args.positional_input

    if args.demo:
        test_df = pd.read_csv(DEFAULT_TEST_CSV)
        features, targets = s1.validate_and_extract_arrays(test_df, "test", s1.EXPECTED_TEST_ROWS)
        idx = np.random.randint(0, len(test_df))
        patient_seqn = str(test_df.iloc[idx]["SEQN"])
        report = predict_patient(model, features[idx], device)
        print(f"\n[DEMO MODE] Evaluated random unseen patient SEQN: {patient_seqn}")
        print(f"Ground Truth Targets: Diabetes={int(targets[idx, 0])}, Obesity={int(targets[idx, 1])}, CVD={int(targets[idx, 2])}")
        print_patient_report(report, patient_id=f"NHANES SEQN {patient_seqn}")

    elif args.input_txt:
        print(f"Reading patient vector from TXT: {args.input_txt}")
        raw_vec = parse_single_line_txt(args.input_txt)
        proc_vec = preprocess_raw_vector(raw_vec, DEFAULT_ARTIFACT)
        report = predict_patient(model, proc_vec, device)
        print_patient_report(report, patient_id=args.input_txt.name)
        if args.output_json:
            with open(args.output_json, "w", encoding="utf-8") as out_f:
                json.dump(report, out_f, indent=2)
            print(f"Saved prediction report to: {args.output_json}")

    elif args.input_json:
        with open(args.input_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            feature_vec = np.array(data, dtype=np.float32)
        elif isinstance(data, dict) and "features" in data:
            feature_vec = np.array(data["features"], dtype=np.float32)
        elif isinstance(data, dict):
            feature_vec = np.array([float(data[col]) for col in s1.FEATURE_COLS], dtype=np.float32)
        else:
            raise ValueError("Unsupported JSON format.")

        report = predict_patient(model, feature_vec, device)
        print_patient_report(report, patient_id=args.input_json.name)
        if args.output_json:
            with open(args.output_json, "w", encoding="utf-8") as out_f:
                json.dump(report, out_f, indent=2)
            print(f"Saved prediction report to: {args.output_json}")

    elif args.input_csv:
        csv_df = pd.read_csv(args.input_csv)
        features, _ = s1.validate_and_extract_arrays(csv_df, "batch_csv", expected_rows=len(csv_df))
        print(f"Scoring {len(csv_df):,} patients from {args.input_csv}...")
        batch_results = []
        for i in range(len(features)):
            res = predict_patient(model, features[i], device)
            row_dict = {"row_index": i, **res["probabilities"]}
            batch_results.append(row_dict)
        out_df = pd.DataFrame(batch_results)
        dest_csv = args.output_json or script_dir / "batch_predictions.csv"
        out_df.to_csv(dest_csv, index=False)
        print(f"Saved batch predictions ({out_df.shape}) to: {dest_csv}")

    elif args.interactive:
        run_interactive_cli(model, device)

    else:
        # Default behavior: run demo and show instructions
        print("\nNo mode flag provided. Running a quick test demonstration...")
        test_df = pd.read_csv(DEFAULT_TEST_CSV)
        features, targets = s1.validate_and_extract_arrays(test_df, "test", s1.EXPECTED_TEST_ROWS)
        report = predict_patient(model, features[0], device)
        print_patient_report(report, patient_id="Sample Test Patient (Index 0)")
        print("\nCLI Usage Options:")
        print("  python infer_bionutrigpt_stage1.py --interactive           (Interactive terminal intake)")
        print("  python infer_bionutrigpt_stage1.py --demo                  (Random patient from 2017-2020 test set)")
        print("  python infer_bionutrigpt_stage1.py --input-json patient.json")
        print("  python infer_bionutrigpt_stage1.py --input-csv patients.csv")


if __name__ == "__main__":
    main()
