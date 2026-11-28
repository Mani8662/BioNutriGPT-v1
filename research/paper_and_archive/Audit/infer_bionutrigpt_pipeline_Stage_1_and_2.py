#!/usr/bin/env python3
"""Unified BioNutriGPT Stage 1 + Stage 2 clinical inference pipeline.

This script reuses the canonical adjacent modules:
    train_bionutrigpt_stage1.py
    stage2_counterfactual_engine.py

It performs Stage 1 risk inference and modality-attention reporting, delegates
counterfactual optimization and its four verification filters to Stage 2, and
writes a single combined JSON dossier.

Run from any working directory:
    python infer_bionutrigpt_pipeline.py [patient_file] [options]

Python 3.10+. Requires the dependencies/assets used by both existing stages.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Project paths and canonical local-module imports.
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import train_bionutrigpt_stage1 as s1  # noqa: E402
import stage2_counterfactual_engine as s2  # noqa: E402

DEFAULT_CHECKPOINT = PROJECT_DIR / "Saved Model" / "bionutrigpt_stage1_best.pt"
DEFAULT_PREPROCESSOR = PROJECT_DIR / "Data" / "preprocessing_artifacts.joblib"
DEFAULT_PATIENT = PROJECT_DIR / "sample_patient_single_line.txt"
DEFAULT_OUTPUT = PROJECT_DIR / "bionutrigpt_pipeline_dossier.json"

CLINICAL_THRESHOLDS: dict[str, float] = {
    "diabetes": 0.617,
    "obesity": 0.445,
    "cvd": 0.603,
}

ACTIVITY_LEVELS = ("sedentary", "light", "moderate", "active")
DISEASE_LABELS = {
    "diabetes": "Type 2 Diabetes",
    "obesity": "Clinical Obesity",
    "cvd": "Cardiovascular Disease",
}

# PyTorch may emit this benign warning for the specified norm-first Transformer.
# Suppress only this known configuration warning; all other warnings remain visible.
warnings.filterwarnings(
    "ignore",
    message=r"enable_nested_tensor is True, but self\.use_nested_tensor is False.*",
    category=UserWarning,
)


# ---------------------------------------------------------------------------
# Input parsing and metadata resolution.
# ---------------------------------------------------------------------------
def parse_patient_vector(file_path: Path) -> np.ndarray:
    """Read one raw Stage 1 patient vector and return finite float32[88].

    Supported formats:
      * .txt: one comma-separated or whitespace-separated line, optionally
        wrapped in square brackets
      * .csv: one row with 88 columns or 88 rows with one column
      * .npy: NumPy array containing exactly 88 values
      * .json: JSON array containing exactly 88 values
    """
    path = Path(file_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Patient input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".txt":
        content = path.read_text(encoding="utf-8-sig").strip()
        nonempty_lines = [line.strip() for line in content.splitlines() if line.strip()]
        data_lines = [line for line in nonempty_lines if not line.startswith("#")]
        if len(data_lines) != 1:
            raise ValueError(
                f"TXT patient input must contain exactly one numeric data line; "
                f"found {len(data_lines)}."
            )
        line = data_lines[0].strip()
        if line.startswith("[") and line.endswith("]"):
            line = line[1:-1].strip()
        # Supporting commas and spaces together is harmless and makes parsing
        # robust to conventional pretty-printed one-line arrays.
        tokens = line.replace(",", " ").split()
        if not tokens:
            raise ValueError("TXT patient input is empty.")
        try:
            values = np.asarray([float(token) for token in tokens], dtype=np.float32)
        except ValueError as exc:
            raise ValueError("TXT input contains a token that is not a numeric value.") from exc

    elif suffix == ".csv":
        # First interpret the file as a headerless numeric matrix, which is the
        # expected form for one-row or one-column vectors.
        frame = pd.read_csv(path, header=None)
        if frame.shape == (1, s1.NUM_FEATURES):
            values = pd.to_numeric(frame.iloc[0], errors="raise").to_numpy(dtype=np.float32)
        elif frame.shape == (s1.NUM_FEATURES, 1):
            values = pd.to_numeric(frame.iloc[:, 0], errors="raise").to_numpy(dtype=np.float32)
        else:
            # Also tolerate a conventional header row if it leaves one numeric
            # record with the required columns (or the transposed one-column form).
            frame = pd.read_csv(path)
            if frame.shape == (1, s1.NUM_FEATURES):
                values = pd.to_numeric(frame.iloc[0], errors="raise").to_numpy(dtype=np.float32)
            elif frame.shape == (s1.NUM_FEATURES, 1):
                values = pd.to_numeric(frame.iloc[:, 0], errors="raise").to_numpy(dtype=np.float32)
            else:
                raise ValueError(
                    f"CSV patient vector must contain exactly one row with {s1.NUM_FEATURES} columns, "
                    f"or one column with {s1.NUM_FEATURES} rows."
                )

    elif suffix == ".npy":
        values = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32).reshape(-1)

    elif suffix == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON patient vector: {exc}") from exc
        if not isinstance(payload, list):
            raise ValueError(f"JSON patient vector must be an array of {s1.NUM_FEATURES} numeric values.")
        values = np.asarray(payload, dtype=np.float32).reshape(-1)

    else:
        raise ValueError("Unsupported patient file type. Use .txt, .csv, .npy, or .json.")

    if values.shape != (s1.NUM_FEATURES,):
        raise ValueError(
            f"Patient vector must have shape ({s1.NUM_FEATURES},); received {values.shape} "
            f"({values.size} values)."
        )
    if not np.isfinite(values).all():
        bad_indices = np.flatnonzero(~np.isfinite(values)).tolist()
        raise ValueError(f"Patient vector contains non-finite values at indices {bad_indices}.")
    return values.astype(np.float32, copy=True)


def resolve_patient_metadata(
    args: argparse.Namespace,
    raw_vector: np.ndarray,
    patient_file: Path | None = None,
) -> dict[str, Any]:
    """Resolve Stage 2 metadata from CLI overrides, embedded file comments, and raw Stage 1 features."""
    if args.age is not None:
        age_years = float(args.age)
    else:
        age_years = float(raw_vector[s1.FEATURE_COLS.index("RIDAGEYR")])

    if args.female:
        is_female = True
    elif args.male:
        is_female = False
    else:
        is_female = bool(raw_vector[s1.FEATURE_COLS.index("SEX_FEMALE")] >= 0.5)

    # Extract any optional metadata declared in text comment lines (e.g. # METADATA: weight_kg=74.0 height_cm=176.0)
    file_metadata: dict[str, Any] = {}
    if patient_file is not None and Path(patient_file).is_file():
        p = Path(patient_file)
        if p.suffix.lower() == ".txt":
            import re
            for comment_line in p.read_text(encoding="utf-8-sig").splitlines():
                comment_line = comment_line.strip()
                if comment_line.startswith("#"):
                    w_match = re.search(r'weight(?:_kg)?\s*[:=]\s*([0-9.]+)', comment_line, re.I)
                    h_match = re.search(r'height(?:_cm)?\s*[:=]\s*([0-9.]+)', comment_line, re.I)
                    act_match = re.search(r'activity(?:_level)?\s*[:=]\s*([a-zA-Z]+)', comment_line, re.I)
                    if w_match:
                        file_metadata["weight_kg"] = float(w_match.group(1))
                    if h_match:
                        file_metadata["height_cm"] = float(h_match.group(1))
                    if act_match and act_match.group(1).lower() in ACTIVITY_LEVELS:
                        file_metadata["activity_level"] = act_match.group(1).lower()

    default_weight = file_metadata.get("weight_kg", 75.0)
    default_height = file_metadata.get("height_cm", 170.0)
    default_activity = file_metadata.get("activity_level", "sedentary")

    metadata: dict[str, Any] = {
        "age_years": age_years,
        "is_female": is_female,
        "weight_kg": default_weight if args.weight is None else float(args.weight),
        "height_cm": default_height if args.height is None else float(args.height),
        "activity_level": default_activity if args.activity is None else str(args.activity),
    }

    for name in ("age_years", "weight_kg", "height_cm"):
        value = metadata[name]
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite; received {value!r}.")
    if metadata["age_years"] < 0:
        raise ValueError("Age must be greater than or equal to zero.")
    if metadata["weight_kg"] <= 0 or metadata["height_cm"] <= 0:
        raise ValueError("Weight and height must be greater than zero.")
    if metadata["activity_level"] not in ACTIVITY_LEVELS:
        raise ValueError(
            f"activity_level must be one of {list(ACTIVITY_LEVELS)}; "
            f"received {metadata['activity_level']!r}."
        )
    return metadata


# ---------------------------------------------------------------------------
# Stage 1: risk probabilities, threshold classifications, and attention drivers.
# ---------------------------------------------------------------------------
def run_stage1_diagnostics(
    raw_vector: np.ndarray,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    preprocessor_path: Path = DEFAULT_PREPROCESSOR,
) -> tuple[dict[str, Any], torch.device]:
    """Run Stage 1 using the exact preprocessing/model loaders from Stage 2."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifacts = s2._load_preprocessing_artifacts(preprocessor_path)
    model = s2._load_stage1_model(checkpoint_path, artifacts, device)

    normalized_np = s2._normalize_numpy(raw_vector, artifacts)
    normalized_t = torch.as_tensor(normalized_np, dtype=torch.float32, device=device)

    with torch.no_grad():
        model_output = model(normalized_t)
        if not isinstance(model_output, tuple) or len(model_output) != 3:
            raise RuntimeError(
                "Stage 1 forward() must return (logits_dict, loss_moe_balance, attention_weights)."
            )
        logits_dict, _loss_moe_balance, attention_weights = model_output
        if not isinstance(logits_dict, Mapping):
            raise RuntimeError("Stage 1 logits output must be a mapping.")
        if not isinstance(attention_weights, Mapping):
            raise RuntimeError("Stage 1 attention_weights output must be a mapping.")

        probabilities: dict[str, float] = {}
        classifications: dict[str, dict[str, Any]] = {}
        for disease in ("diabetes", "obesity", "cvd"):
            if disease not in logits_dict:
                raise RuntimeError(f"Stage 1 logits are missing the {disease!r} key.")
            probability = float(torch.sigmoid(logits_dict[disease]).reshape(-1)[0].item())
            if not math.isfinite(probability):
                raise RuntimeError(f"Stage 1 produced a non-finite {disease} probability.")
            threshold = CLINICAL_THRESHOLDS[disease]
            positive = probability >= threshold
            probabilities[disease] = probability
            classifications[disease] = {
                "status": "HIGH RISK / POSITIVE" if positive else "NORMAL / LOW RISK",
                "threshold": threshold,
                "positive": bool(positive),
            }

        modality_columns = (
            ("bio", s1.BIO_COLS),
            ("diet", s1.DIET_COLS),
            ("life", s1.LIFE_COLS),
            ("demo", s1.DEMO_COLS),
        )
        top_drivers: dict[str, list[dict[str, Any]]] = {}
        for modality, columns in modality_columns:
            if modality not in attention_weights:
                raise RuntimeError(f"Stage 1 attention output is missing modality {modality!r}.")
            weights_tensor = attention_weights[modality]
            if not isinstance(weights_tensor, torch.Tensor):
                raise RuntimeError(f"Attention weights for {modality!r} are not a tensor.")
            weights = weights_tensor.squeeze(0).squeeze(-1).detach().cpu().numpy()
            if weights.ndim != 1 or len(weights) != len(columns):
                raise RuntimeError(
                    f"Unexpected {modality} attention shape {tuple(weights_tensor.shape)}; "
                    f"expected one weight for each of {len(columns)} features."
                )
            if not np.isfinite(weights).all():
                raise RuntimeError(f"Non-finite attention weights detected for {modality!r}.")
            top_indices = np.argsort(weights)[::-1][: min(3, len(columns))]
            top_drivers[modality] = [
                {
                    "feature": str(columns[int(index)]),
                    "weight_pct": round(float(weights[int(index)] * 100.0), 2),
                }
                for index in top_indices
            ]

    assessment = {
        "probabilities": probabilities,
        "classifications": classifications,
        "top_feature_drivers": top_drivers,
    }
    return assessment, device


# ---------------------------------------------------------------------------
# Stage 2: delegate optimization and all clinical verification to existing S2.
# ---------------------------------------------------------------------------
def run_stage2_counterfactual(
    raw_vector: np.ndarray,
    patient_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Call the canonical Stage 2 public API without duplicating its logic."""
    return s2.generate_counterfactual_prescription(
        raw_patient_vector_88=raw_vector,
        patient_metadata=dict(patient_metadata),
    )


# ---------------------------------------------------------------------------
# Dossier construction and human-readable clinical report.
# ---------------------------------------------------------------------------
def _atomic_write_json(payload: Mapping[str, Any], output_path: Path) -> None:
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    temporary_path.replace(path)


def _format_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    rendered = [[str(cell) for cell in row] for row in rows]
    widths = [len(str(header)) for header in headers]
    for row in rendered:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def format_row(row: Sequence[str]) -> str:
        return "| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) + " |"

    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    output = [separator, format_row([str(h) for h in headers]), separator]
    output.extend(format_row(row) for row in rendered)
    output.append(separator)
    return "\n".join(output)


def _get_nutrient_record(stage2_dossier: Mapping[str, Any], nutrient_name: str) -> Mapping[str, Any]:
    envelope = stage2_dossier.get("nutrient_constraint_envelope", [])
    for record in envelope:
        if isinstance(record, Mapping) and record.get("nutrient") == nutrient_name:
            return record
    raise KeyError(f"Stage 2 nutrient envelope is missing {nutrient_name!r}.")


# ---------------------------------------------------------------------------
# Explainable AI (XAI): Clinical Etiology & Risk Deconstruction
# ---------------------------------------------------------------------------
def compute_clinical_etiology_and_risks(
    raw_vector: np.ndarray,
    patient_metadata: Mapping[str, Any],
    stage1_assessment: Mapping[str, Any],
    stage2_dossier: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute explicit Nutritional and Weight-Related risk stratifications and mechanistic explanations."""
    def get_f(name: str) -> float:
        return float(raw_vector[s1.FEATURE_COLS.index(name)])

    hdl = get_f("LBDHDD")
    tc = get_f("LBXTC")
    non_hdl = get_f("NON_HDL_CHOL")
    chol_hdl = get_f("CHOL_HDL_RATIO")
    non_hdl_hdl = get_f("NON_HDL_TO_HDL_RATIO")

    kcal = get_f("DR1TKCAL")
    prot = get_f("DR1TPROT")
    carb = get_f("DR1TCARB")
    sugar = get_f("DR1TSUGR")
    fiber = get_f("DR1TFIBE")
    tfat = get_f("DR1TTFAT")
    sfat = get_f("DR1TSFAT")
    mfat = get_f("DR1TMFAT")
    pfat = get_f("DR1TPFAT")
    sodi = get_f("DR1TSODI")
    pota = get_f("DR1TPOTA")

    sugar_fiber_ratio = get_f("SUGAR_TO_FIBER_RATIO")
    na_k_ratio = get_f("SODIUM_POTASSIUM_RATIO")
    sat_unsat_ratio = get_f("SAT_TO_UNSAT_FAT_RATIO")

    sleep_hours = get_f("SLEEP_HOURS")
    sleep_dev = get_f("SLEEP_DEV_FROM_7_5")
    pad680_sitting = get_f("PAD680")

    age = float(patient_metadata["age_years"])
    is_female = bool(patient_metadata["is_female"])
    weight = float(patient_metadata["weight_kg"])
    height = float(patient_metadata["height_cm"])
    bmi = weight / ((height / 100.0) ** 2)

    energy = stage2_dossier.get("energy_prescription", {})
    tdee = float(energy.get("TDEE", 2000.0))
    bmr = float(energy.get("BMR", 1600.0))
    energy_delta = kcal - tdee

    sfat_pct = (sfat * 9.0) / (kcal + 1e-4) * 100.0
    sugar_pct = (sugar * 4.0) / (kcal + 1e-4) * 100.0
    dri_fiber = 28.0 if is_female else 38.0

    # Additional Nutrients for Comprehensive Spectrum
    chol = float(raw_vector[15])
    vb12 = float(raw_vector[31])
    vc = float(raw_vector[32])
    calc = float(raw_vector[34])
    magn = float(raw_vector[36])
    iron = float(raw_vector[37])
    zinc = float(raw_vector[38])
    caff = float(raw_vector[43])
    alco = float(raw_vector[44])

    # Additional Lifestyle Features
    walk_bike = float(raw_vector[57])
    pa_total_met = float(raw_vector[60])
    slq050 = float(raw_vector[61])

    prot_per_kg = prot / max(30.0, weight)
    dri_fiber = 28.0 if is_female else 38.0
    dri_iron = 18.0 if (is_female and age < 51) else 8.0
    dri_calc = 1200.0 if (is_female and age > 50 or age > 70) else 1000.0
    dri_magn = 320.0 if is_female else 420.0

    # 1. Comprehensive Nutritional Risk Stratification (All Core Clinical Domains)
    nutritional_risks = [
        {
            "category": "Glycemic Dynamics",
            "metric": "Sugar-to-Fiber Ratio",
            "observed": f"{sugar_fiber_ratio:.2f}",
            "cutoff": "< 1.00",
            "status": "HIGH RISK" if sugar_fiber_ratio > 1.5 else ("BORDERLINE" if sugar_fiber_ratio > 1.0 else "OPTIMAL"),
            "mitigation_action": "Pair starchy carbohydrates with 10–15g of viscous soluble fiber (chia seeds, oats, psyllium, or whole lentils) and replace refined sweets with whole fruits to buffer postprandial glucose absorption.",
            "progression_early": "Post-meal lethargy, reactive hypoglycemia (energy crashes 2 hours after eating), and frequent carbohydrate cravings.",
            "progression_advanced": "Exhaustion of pancreatic beta-cell insulin secretion, progressive fasting hyperglycemia, and microvascular injury (diabetic retinopathy and nephropathy).",
        },
        {
            "category": "Vascular Electrolytes",
            "metric": "Sodium-to-Potassium Ratio",
            "observed": f"{na_k_ratio:.2f}",
            "cutoff": "< 1.00",
            "status": "HIGH RISK" if na_k_ratio > 1.0 else ("BORDERLINE" if na_k_ratio > 0.67 else "OPTIMAL"),
            "mitigation_action": "Limit added salt and processed snacks to keep sodium below 2,000 mg/day; boost potassium-dense foods (spinach, bottle gourd, lentils, coconut water) to exceed 3,500 mg/day.",
            "progression_early": "Subtle fluid retention, transient blood pressure elevations, and increased arterial wall tension.",
            "progression_advanced": "Progressive arterial wall remodeling and stiffening, left ventricular concentric hypertrophy, and substantially elevated risk of hemorrhagic stroke.",
        },
        {
            "category": "Lipid Quality",
            "metric": "Saturated Fat % Calories",
            "observed": f"{sfat_pct:.1f}%",
            "cutoff": "< 10.0%",
            "status": "HIGH RISK" if sfat_pct > 10.0 else "OPTIMAL",
            "mitigation_action": "Moderate cooking fats (ghee, butter, palm oil) to <10% of total calories; substitute with cold-pressed monounsaturated fats (mustard oil, olive oil, sesame oil, nuts).",
            "progression_early": "Downregulation of hepatic LDL receptors, leading to higher circulating concentrations of ApoB atherogenic particles.",
            "progression_advanced": "Subendothelial cholesterol deposition, formation of fibrous atheromatous plaques in coronary and carotid arteries, and heightened myocardial infarction risk.",
        },
        {
            "category": "Microbiome Barrier",
            "metric": "Dietary Fiber Adequacy",
            "observed": f"{fiber:.1f} g/day",
            "cutoff": f"≥ {dri_fiber:.0f} g/day",
            "status": "DEFICIENT" if fiber < dri_fiber * 0.7 else "ADEQUATE",
            "mitigation_action": "Increase intake of unpolished grains (millets, brown rice), leafy greens, and whole legumes to reach your daily target (≥28g females, ≥38g males).",
            "progression_early": "Sluggish gastrointestinal transit, reduced short-chain fatty acid (butyrate) synthesis, and blunted postprandial satiety.",
            "progression_advanced": "Degradation of the protective gut mucosal layer, subclinical endotoxemia (leaky gut), systemic low-grade inflammation, and increased colorectal disease risk.",
        },
        {
            "category": "Muscle Mass Defense",
            "metric": "Protein Intake per kg",
            "observed": f"{prot_per_kg:.2f} g/kg",
            "cutoff": "≥ 0.80 g/kg",
            "status": "DEFICIENT" if prot_per_kg < 0.80 else "ADEQUATE",
            "mitigation_action": "Ensure 20–30g of high-biological-value protein per main meal (paneer, tofu, dal with rice, eggs, fish, or sprouted moong) to maintain at least 0.8–1.0 g/kg body weight.",
            "progression_early": "Delayed muscle recovery after exertion, mild afternoon fatigue, and gradual decline in basal metabolic calorie burn.",
            "progression_advanced": "Accelerated skeletal muscle wasting (sarcopenia), physical frailty, reduced functional mobility, and decreased whole-body glucose disposal capacity.",
        },
        {
            "category": "Bone & Mineral Health",
            "metric": "Calcium Intake",
            "observed": f"{calc:.0f} mg/day",
            "cutoff": f"≥ {dri_calc:.0f} mg/day",
            "status": "DEFICIENT" if calc < dri_calc * 0.7 else "ADEQUATE",
            "mitigation_action": "Consume calcium-rich foods (curd, ragi, sesame seeds, fortified milk) and magnesium sources (almonds, pumpkin seeds, green leafy vegetables) daily.",
            "progression_early": "Occasional muscle cramps, neuromuscular irritability, and compensatory bone remodeling to sustain serum calcium levels.",
            "progression_advanced": "Progressive loss of skeletal bone mineral density (osteopenia transitioning to osteoporosis), increased fracture susceptibility, and vascular wall calcification.",
        },
        {
            "category": "Oxygen Transport",
            "metric": "Iron Intake",
            "observed": f"{iron:.1f} mg/day",
            "cutoff": f"≥ {dri_iron:.0f} mg/day",
            "status": "DEFICIENT" if iron < dri_iron * 0.7 else "ADEQUATE",
            "mitigation_action": "Combine iron-rich plant foods (spinach, lentils, dates) with natural ascorbic acid (fresh lemon juice, amla) to boost non-heme iron absorption threefold.",
            "progression_early": "Mild exertional shortness of breath, diminished exercise endurance, and cognitive brain fog.",
            "progression_advanced": "Microcytic hypochromic anemia, chronic tissue hypoxia, impaired cardiac output, and compromised immune cell surveillance.",
        },
        {
            "category": "Antioxidant Defense",
            "metric": "Vitamin C Intake",
            "observed": f"{vc:.1f} mg/day",
            "cutoff": "≥ 75.0 mg/day",
            "status": "DEFICIENT" if vc < 50.0 else "ADEQUATE",
            "mitigation_action": "Add half a fresh lemon over cooked curries, enjoy citrus fruits, or consume raw amla/guava daily to meet the 75–100 mg/day antioxidant threshold.",
            "progression_early": "Suboptimal collagen synthesis, reduced microvascular capillary resilience, and delayed minor wound healing.",
            "progression_advanced": "Heightened oxidative modification of circulating LDL particles, accelerating endothelial dysfunction and vascular plaque initiation.",
        },
        {
            "category": "Toxicological Strain",
            "metric": "Daily Alcohol Intake",
            "observed": f"{alco:.1f} g/day",
            "cutoff": "≤ 14.0 g/day",
            "status": "EXCESSIVE" if alco > 28.0 else ("ELEVATED" if alco > 14.0 else "SAFE / LOW"),
            "mitigation_action": "Keep alcohol consumption within safe limits (≤1 standard drink/day for women, ≤2 for men) with at least 3–4 consecutive alcohol-free days weekly.",
            "progression_early": "Transient elevation of hepatic triglycerides, suppression of nocturnal REM sleep, and temporary morning blood pressure surges.",
            "progression_advanced": "Progressive hepatic steatosis (metabolic liver disease), persistent sympathetic tone elevation, and cardiometabolic dyslipidemia.",
        },
    ]

    # 2. Comprehensive Weight-Related & Lifestyle Risk Stratification
    sarcopenic_obesity = bool(bmi >= 25.0 and (prot_per_kg < 0.85 or pad680_sitting >= 480))
    weight_risks = [
        {
            "category": "Energy Balance",
            "metric": "Caloric Balance vs TDEE",
            "observed": f"{energy_delta:+.0f} kcal/day",
            "cutoff": "±150 kcal/day",
            "status": "SURPLUS (> +250 kcal)" if energy_delta > 250 else ("DEFICIT (< -300 kcal)" if energy_delta < -300 else "BALANCED"),
            "mitigation_action": "Align daily calorie intake within ±150 kcal of your Mifflin-St Jeor TDEE; for gradual fat loss, introduce a structured 300–400 kcal deficit via mindful portion control.",
            "progression_early": "Positive caloric storage expanding subcutaneous adipose reserves, mild abdominal bloating, and increased insulin secretion.",
            "progression_advanced": "Ectopic visceral lipid accumulation directly surrounding the liver, heart, and pancreas, driving Metabolic Dysfunction-Associated Steatotic Liver Disease (MASLD).",
        },
        {
            "category": "Body Mass Index",
            "metric": "Body Mass Index (BMI)",
            "observed": f"{bmi:.1f} kg/m²",
            "cutoff": "18.5 - 24.9 kg/m²",
            "status": "OBESE (≥ 30.0)" if bmi >= 30.0 else ("OVERWEIGHT (25.0 - 29.9)" if bmi >= 25.0 else ("UNDERWEIGHT" if bmi < 18.5 else "NORMAL")),
            "mitigation_action": "Adopt a consistent calorie-controlled whole-food diet combined with daily brisk walking to gradually bring BMI toward the 18.5–24.9 kg/m² healthy range.",
            "progression_early": "Increased mechanical load on knee and hip joints, mild exertional dyspnea, and subclinical elevation of systemic inflammatory markers (hs-CRP).",
            "progression_advanced": "Chronic systemic adipokine toxicity (elevated TNF-α, IL-6), accelerated joint cartilage degradation (osteoarthritis), and marked peripheral insulin receptor downregulation.",
        },
        {
            "category": "Sarcopenic Adiposity",
            "metric": "Sarcopenic Obesity Index",
            "observed": "High Risk" if sarcopenic_obesity else "Low Risk",
            "cutoff": "Low Risk",
            "status": "HIGH RISK (Fat+Muscle Loss)" if sarcopenic_obesity else "OPTIMAL",
            "mitigation_action": "Incorporate progressive resistance training (bodyweight squats, resistance bands, or light weights) 2–3 times weekly alongside adequate protein intake (1.0–1.2 g/kg).",
            "progression_early": "Normal or high scale weight masking a loss of functional muscle mass, accompanied by early fatigue during daily chores.",
            "progression_advanced": "Severe metabolic decline where lower muscle mass burns fewer resting calories, locking the body into rapid fat regain and frailty.",
        },
        {
            "category": "Sedentary Behavior",
            "metric": "Daily Sitting Time (PAD680)",
            "observed": f"{pad680_sitting:.0f} min/day ({pad680_sitting/60.0:.1f}h)",
            "cutoff": "< 480 min/day (< 8h)",
            "status": "HIGH SEDENTARY RISK" if pad680_sitting >= 480 else "ACCEPTABLE",
            "mitigation_action": "Stand or walk for 2–3 minutes every 45–60 minutes of sitting; consider a height-adjustable standing desk or walking meetings.",
            "progression_early": "Suppression of muscle lipoprotein lipase (LPL) activity, leading to slower post-meal lipid and glucose clearance from circulation.",
            "progression_advanced": "Lower-extremity venous stasis, skeletal muscle insulin resistance, and significantly elevated all-cause cardiovascular mortality.",
        },
        {
            "category": "Physical Inactivity",
            "metric": "Weekly MET Activity",
            "observed": f"{pa_total_met:.0f} MET-min/wk",
            "cutoff": "≥ 600 MET-min/wk",
            "status": "INSUFFICIENT" if pa_total_met < 600 else "ACTIVE",
            "mitigation_action": "Accumulate at least 150 minutes of moderate aerobic exercise (brisk walking at 5 km/h) or 75 minutes of vigorous exercise weekly to meet the 600 MET-min guideline.",
            "progression_early": "Gradual decrease in maximal oxygen uptake (VO2 max) and diminished cardiac stroke volume efficiency.",
            "progression_advanced": "Loss of skeletal muscle mitochondrial density, blunted non-insulin-mediated GLUT4 glucose uptake, and arterial stiffening.",
        },
        {
            "category": "Active Transport",
            "metric": "Walking/Biking Travel",
            "observed": f"{walk_bike:.0f} min/wk",
            "cutoff": "> 0 min/wk",
            "status": "SEDENTARY COMMUTE" if walk_bike < 30 else "ACTIVE COMMUTE",
            "mitigation_action": "Integrate purposeful active travel—such as taking the stairs, parking further away, or walking for short neighborhood errands (aim for 20–30 min/day).",
            "progression_early": "Low non-exercise activity thermogenesis (NEAT), reducing total daily baseline calorie burn.",
            "progression_advanced": "Decreased peripheral circulation, reduced lymphatic drainage, and heightened vulnerability to weight creep over time.",
        },
        {
            "category": "Circadian Metabolic",
            "metric": "Sleep Duration / Deviation",
            "observed": f"{sleep_hours:.1f} hrs (dev: {sleep_dev:+.1f}h)",
            "cutoff": "7.0 - 8.5 hrs/night",
            "status": "SLEEP DEPRIVATION" if sleep_hours < 6.0 else ("EXCESS SLEEP" if sleep_hours > 9.0 else "OPTIMAL"),
            "mitigation_action": "Maintain a regular sleep schedule aiming for 7.0–8.5 hours nightly; avoid screen exposure and heavy meals within 2 hours of bedtime.",
            "progression_early": "Acute hormonal disruption: elevated morning cortisol, increased hunger hormone ghrelin (+28%), and suppressed satiety hormone leptin (-18%).",
            "progression_advanced": "Chronic circadian insulin resistance, habitual evening hyperphagia (cravings for hyper-palatable foods), and blunted metabolic flexibility.",
        },
        {
            "category": "Nocturnal Sleep Quality",
            "metric": "Sleep Disturbance Flag (SLQ050)",
            "observed": "Reported Trouble" if slq050 > 0.5 else "None",
            "cutoff": "None",
            "status": "ELEVATED RISK" if slq050 > 0.5 else "OPTIMAL",
            "mitigation_action": "Optimize sleep hygiene (cool, dark room; minimize alcohol before sleep); if snoring or waking gasping occurs, seek clinical evaluation for obstructive sleep apnea.",
            "progression_early": "Frequent nocturnal micro-arousals triggering nighttime sympathetic nervous system surges and unrefreshing sleep.",
            "progression_advanced": "Sustained nocturnal hypoxemia, loss of physiological blood pressure dipping at night, morning hypertension, and long-term cardiac strain.",
        },
    ]

    # 3. Mechanistic Causal Explanations for Diseases
    probs = stage1_assessment.get("probabilities", {})
    cvd_prob = probs.get("cvd", 0.0) * 100.0
    diab_prob = probs.get("diabetes", 0.0) * 100.0
    obes_prob = probs.get("obesity", 0.0) * 100.0

    cvd_status = "Elevated" if probs.get("cvd", 0.0) >= CLINICAL_THRESHOLDS["cvd"] else "Normal / Controlled"
    diab_status = "Elevated" if probs.get("diabetes", 0.0) >= CLINICAL_THRESHOLDS["diabetes"] else "Normal / Controlled"
    obes_status = "Elevated" if probs.get("obesity", 0.0) >= CLINICAL_THRESHOLDS["obesity"] else "Normal / Controlled"

    # Nuanced clinical narratives conditioned on patient biomarker and dietary states
    cvd_pathology = (
        f"amplified by an adverse dietary Na/K ratio of {na_k_ratio:.2f} (target <1.00) and {sfat_pct:.1f}% saturated fat intake, increasing vascular endothelial shear stress and atheroma vulnerability."
        if (na_k_ratio > 1.0 or sfat_pct > 10.0)
        else f"stabilized by a favorable dietary Na/K ratio of {na_k_ratio:.2f} and controlled saturated fat intake ({sfat_pct:.1f}% of kcal), preserving vascular endothelial nitric oxide signaling."
    )
    cvd_narrative = (
        f"Cardiovascular Disease ({cvd_prob:.1f}% - {cvd_status}): "
        f"Circulating atherogenic sterols show Non-HDL Cholesterol = {non_hdl:.1f} mg/dL (Total/HDL = {chol_hdl:.2f}). "
        f"Nutritional profile is {cvd_pathology}"
    )

    fiber_deficit = max(0.0, dri_fiber - fiber)
    diab_pathology = (
        f"exacerbated by a high sugar-to-fiber ratio of {sugar_fiber_ratio:.2f} (target <1.00) and a {fiber_deficit:.1f}g fiber deficit, precipitating postprandial glycemic excursions that overburden beta-cells."
        if (sugar_fiber_ratio > 1.2 or fiber_deficit > 10.0)
        else f"supported by balanced glycemic intake (sugar-to-fiber ratio {sugar_fiber_ratio:.2f}) and adequate dietary fiber ({fiber:.1f}g/day), attenuating rapid glucose absorption."
    )
    diab_narrative = (
        f"Type 2 Diabetes Screening ({diab_prob:.1f}% - {diab_status}): "
        f"HbA1c-independent metabolic screening based on atherogenic lipids (Total/HDL: {chol_hdl:.2f}, Non-HDL: {non_hdl:.1f} mg/dL). "
        f"Nutritional & metabolic dynamics are {diab_pathology}"
    )

    weight_pathology = (
        f"driven by a positive daily caloric surplus (+{energy_delta:.0f} kcal/day over Mifflin-St Jeor TDEE {tdee:.0f} kcal) and {pad680_sitting/60.0:.1f} hrs/day of sedentary sitting, actively promoting visceral adipose accumulation."
        if energy_delta > 250
        else f"supported by a balanced daily caloric intake ({energy_delta:+.0f} kcal relative to Mifflin-St Jeor TDEE {tdee:.0f} kcal) with {pad680_sitting/60.0:.1f} hrs/day of sitting time, maintaining energetic homeostasis."
    )
    obesity_narrative = (
        f"Weight-Related / Obesity ({obes_prob:.1f}% - {obes_status}): "
        f"Anthropometric state presents BMI = {bmi:.1f} kg/m². "
        f"Energetic and lifestyle balance is {weight_pathology}"
    )

    etiology = {
        "cvd_etiology": cvd_narrative,
        "diabetes_etiology": diab_narrative,
        "obesity_etiology": obesity_narrative,
    }

    return {
        "nutritional_risks": nutritional_risks,
        "weight_risks": weight_risks,
        "etiology_explanations": etiology,
    }


def build_unified_dossier(
    raw_vector: np.ndarray,
    patient_metadata: Mapping[str, Any],
    stage1_assessment: Mapping[str, Any],
    stage2_dossier: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble the exact requested unified JSON contract."""
    stage2_enriched = dict(stage2_dossier)
    full_envelope = stage2_dossier.get("nutrient_constraint_envelope", [])
    if not isinstance(full_envelope, list):
        raise TypeError("Stage 2 nutrient_constraint_envelope must be a list.")
    top_modified = [
        dict(record)
        for record in full_envelope
        if isinstance(record, Mapping)
        and abs(float(record.get("delta", 0.0))) > 0.05
    ]
    top_modified.sort(key=lambda record: abs(float(record["delta"])), reverse=True)
    stage2_enriched["top_modified_nutrients"] = top_modified

    audit_filters = stage2_dossier.get("audit_filters", {})
    filter_names = (
        "filter_1_thermodynamic_balance",
        "filter_2_iom_safety_clamps",
        "filter_3_clinical_ratio_coherence",
        "filter_4_closed_loop_rescoring",
    )
    all_filters_passed = all(
        bool(audit_filters.get(name, {}).get("passed_after_all_later_filters", audit_filters.get(name, {}).get("passed", False)))
        for name in filter_names
    )
    stage3_certified = bool(stage2_dossier.get("certified_for_stage3", False) and all_filters_passed)

    etiology_and_risks = compute_clinical_etiology_and_risks(
        raw_vector=raw_vector,
        patient_metadata=patient_metadata,
        stage1_assessment=stage1_assessment,
        stage2_dossier=stage2_dossier,
    )

    return {
        "patient_metadata": dict(patient_metadata),
        "stage1_diagnostic_assessment": dict(stage1_assessment),
        "nutritional_risk_stratification": etiology_and_risks["nutritional_risks"],
        "weight_related_risk_stratification": etiology_and_risks["weight_risks"],
        "clinical_etiology_explanation": etiology_and_risks["etiology_explanations"],
        "stage2_counterfactual_prescription": stage2_enriched,
        "stage3_certified": stage3_certified,
    }


def render_clinical_report(
    raw_vector: np.ndarray,
    patient_metadata: Mapping[str, Any],
    stage1_assessment: Mapping[str, Any],
    stage2_dossier: Mapping[str, Any],
    stage3_certified: bool,
) -> str:
    """Render a readable fixed-width clinical report for terminal output."""
    lines: list[str] = []
    lines.append("=" * 92)
    lines.append("BIONUTRIGPT | UNIFIED CLINICAL INFERENCE REPORT")
    lines.append("=" * 92)

    sex = "Female" if patient_metadata["is_female"] else "Male"
    lines.extend(
        [
            "\nPATIENT PROFILE",
            "-" * 92,
            f"Age: {patient_metadata['age_years']:.1f} years    Sex: {sex}    "
            f"Weight: {patient_metadata['weight_kg']:.1f} kg    "
            f"Height: {patient_metadata['height_cm']:.1f} cm    "
            f"Activity: {patient_metadata['activity_level']}",
            "\nSTAGE 1 | DIAGNOSTIC RISK STRATIFICATION",
        ]
    )

    classifications = stage1_assessment["classifications"]
    probs = stage1_assessment["probabilities"]
    diagnostic_rows = []
    for disease in ("diabetes", "obesity", "cvd"):
        cls = classifications[disease]
        status = "[HIGH RISK]" if cls["positive"] else "[NORMAL]"
        diagnostic_rows.append(
            (
                DISEASE_LABELS[disease],
                f"{probs[disease] * 100:.2f}%",
                f"{CLINICAL_THRESHOLDS[disease] * 100:.2f}%",
                status,
            )
        )
    lines.append(_format_table(("Disease", "Probability", "Operating Cutoff", "Risk Tier"), diagnostic_rows))

    lines.append("\nSTAGE 1 | TOP ATTENTION DRIVERS BY MODALITY")
    modality_labels = {
        "bio": "Blood biomarkers",
        "diet": "Dietary features",
        "life": "Lifestyle / sleep",
        "demo": "Demographics",
    }
    top_drivers = stage1_assessment["top_feature_drivers"]
    driver_rows: list[tuple[str, str, str]] = []
    for modality in ("bio", "diet", "life", "demo"):
        for rank, item in enumerate(top_drivers[modality], start=1):
            driver_rows.append(
                (modality_labels[modality], f"{rank}. {item['feature']}", f"{item['weight_pct']:.2f}%")
            )
    lines.append(_format_table(("Modality", "Feature", "Attention weight"), driver_rows))

    # Deconstruct Nutritional and Weight-Related Risks
    etiology_and_risks = compute_clinical_etiology_and_risks(
        raw_vector=raw_vector,
        patient_metadata=patient_metadata,
        stage1_assessment=stage1_assessment,
        stage2_dossier=stage2_dossier,
    )

    lines.append("\nNUTRITIONAL RISK STRATIFICATION (DIETARY ETIOLOGY)")
    nutri_table_rows = [
        (item["category"], item["metric"], item["observed"], item["cutoff"], item["status"])
        for item in etiology_and_risks["nutritional_risks"]
    ]
    lines.append(_format_table(("Category", "Metric", "Observed", "Target", "Status"), nutri_table_rows))

    lines.append("\nWEIGHT-RELATED RISK STRATIFICATION (METABOLIC & ANTHROPOMETRIC ETIOLOGY)")
    weight_table_rows = [
        (item["category"], item["metric"], item["observed"], item["cutoff"], item["status"])
        for item in etiology_and_risks["weight_risks"]
    ]
    lines.append(_format_table(("Category", "Metric", "Observed", "Target", "Status"), weight_table_rows))

    lines.append("\nCLINICAL ETIOLOGY & MECHANISTIC ROOT-CAUSE EXPLANATION")
    lines.append("-" * 92)
    etiology_dict = etiology_and_risks["etiology_explanations"]
    lines.append(f"• {etiology_dict['cvd_etiology']}")
    lines.append(f"• {etiology_dict['diabetes_etiology']}")
    lines.append(f"• {etiology_dict['obesity_etiology']}")

    lines.append("\nPERSONALIZED PREVENTATIVE ACTION PLAN (FOR PATIENT & CLINICIAN)")
    lines.append("-" * 92)
    lines.append("[1] TARGETED NUTRITIONAL GUIDANCE:")
    for item in etiology_and_risks["nutritional_risks"]:
        status = item['status']
        lines.append(f"\n  • {item['metric']} (Your Value: {item['observed']} | Healthy Target: {item['cutoff']} | Status: {status})")
        if "OPTIMAL" in status or "SAFE" in status or "ADEQUATE" in status:
            lines.append(f"    - What you should do now    : Healthy level! Keep up your current eating habits.")
        elif "BORDERLINE" in status:
            lines.append(f"    - What you should do now    : {item['mitigation_action']}")
            lines.append(f"    - How you might feel now    : {item['progression_early']}")
        else:
            lines.append(f"    - What you should do now    : {item['mitigation_action']}")
            lines.append(f"    - How you might feel now    : {item['progression_early']}")
            lines.append(f"    - What happens over time    : {item['progression_advanced']}")

    lines.append("\n[2] TARGETED WEIGHT & LIFESTYLE GUIDANCE:")
    for item in etiology_and_risks["weight_risks"]:
        status = item['status']
        lines.append(f"\n  • {item['metric']} (Your Value: {item['observed']} | Healthy Target: {item['cutoff']} | Status: {status})")
        if "OPTIMAL" in status or "BALANCED" in status or "ACCEPTABLE" in status or "ACTIVE" in status or "NORMAL" in status or "SAFE" in status:
            lines.append(f"    - What you should do now    : Healthy level! Continue your current lifestyle routine.")
        elif "BORDERLINE" in status or "MODERATE" in status:
            lines.append(f"    - What you should do now    : {item['mitigation_action']}")
            lines.append(f"    - How you might feel now    : {item['progression_early']}")
        else:
            lines.append(f"    - What you should do now    : {item['mitigation_action']}")
            lines.append(f"    - How you might feel now    : {item['progression_early']}")
            lines.append(f"    - What happens over time    : {item['progression_advanced']}")

    energy = stage2_dossier["energy_prescription"]
    lines.extend(
        [
            "\nSTAGE 2 | ENERGY PRESCRIPTION",
            _format_table(
                ("Metric", "Value"),
                [
                    ("Basal Metabolic Rate (BMR)", f"{energy['BMR']:.1f} kcal/day"),
                    ("Total Daily Energy Expenditure (TDEE)", f"{energy['TDEE']:.1f} kcal/day"),
                    ("Target calorie prescription", f"{energy['target_calories']:.1f} kcal/day"),
                ],
            ),
            "\nSTAGE 2 | KEY DIETARY INTERVENTIONS",
        ]
    )

    intervention_specs = (
        ("Calories", "DR1TKCAL"),
        ("Protein (g)", "DR1TPROT"),
        ("Carbohydrates (g)", "DR1TCARB"),
        ("Total sugars (g)", "DR1TSUGR"),
        ("Dietary fiber (g)", "DR1TFIBE"),
        ("Saturated fat (g)", "DR1TSFAT"),
        ("Sodium (mg)", "DR1TSODI"),
        ("Potassium (mg)", "DR1TPOTA"),
    )
    intervention_rows = []
    for label, nutrient in intervention_specs:
        record = _get_nutrient_record(stage2_dossier, nutrient)
        intervention_rows.append(
            (
                label,
                f"{float(record['baseline']):.2f}",
                f"{float(record['target']):.2f}",
                f"{float(record['delta']):+.2f}",
            )
        )
    lines.append(
        _format_table(
            ("Nutrient", "Baseline", "Prescribed target", "Delta"),
            intervention_rows,
        )
    )

    risk_reductions = stage2_dossier["projected_risk_reductions"]
    risk_rows = []
    for disease in ("diabetes", "obesity", "cvd"):
        record = risk_reductions[disease]
        initial = float(record["initial"])
        target = float(record["target"])
        # Positive value means the post-intervention modeled probability fell.
        absolute_drop = initial - target
        risk_rows.append(
            (
                DISEASE_LABELS[disease],
                f"{initial * 100:.2f}%",
                f"{target * 100:.2f}%",
                f"{absolute_drop * 100:+.2f} pp",
            )
        )
    lines.extend(
        [
            "\nSTAGE 2 | CLOSED-LOOP RISK REDUCTIONS",
            _format_table(
                ("Disease", "Baseline risk", "Post-intervention risk", "Absolute drop"),
                risk_rows,
            ),
            "\nSTAGE 3 | CERTIFICATION",
        ]
    )

    if stage3_certified:
        lines.append("[✔] STAGE 3 CERTIFIED: ALL 4 SAFETY & THERMODYNAMIC FILTERS PASSED")
    else:
        lines.append("[X] STAGE 3 NOT CERTIFIED: ONE OR MORE VERIFICATION FILTERS FAILED")
        audit = stage2_dossier.get("audit_filters", {})
        for index in range(1, 5):
            name = next((key for key in audit if key.startswith(f"filter_{index}_")), None)
            if name is not None:
                state = audit[name]
                passed = state.get("passed_after_all_later_filters", state.get("passed", False))
                lines.append(f"  Filter {index}: {'PASS' if passed else 'FAIL'}")

    lines.append("=" * 92)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI and end-to-end orchestration.
# ---------------------------------------------------------------------------
def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run unified BioNutriGPT Stage 1 diagnostic inference and Stage 2 prescription."
    )
    parser.add_argument(
        "patient_file",
        nargs="?",
        type=Path,
        default=DEFAULT_PATIENT,
        help=f"Patient vector (.txt/.csv/.npy/.json); default: {DEFAULT_PATIENT}",
    )
    parser.add_argument("--weight", type=float, default=None, help="Patient weight in kg (default: 75).")
    parser.add_argument("--height", type=float, default=None, help="Patient height in cm (default: 170).")
    parser.add_argument("--age", type=float, default=None, help="Patient age in years (default: raw vector).")
    sex_group = parser.add_mutually_exclusive_group()
    sex_group.add_argument("--female", action="store_true", help="Override inferred sex as female.")
    sex_group.add_argument("--male", action="store_true", help="Override inferred sex as male.")
    parser.add_argument(
        "--activity",
        choices=ACTIVITY_LEVELS,
        default=None,
        help="Activity level (default: sedentary).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Unified JSON dossier output path; default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress the formatted clinical report.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    patient_path = Path(args.patient_file).expanduser()
    output_path = Path(args.output).expanduser()

    raw_vector = parse_patient_vector(patient_path)
    patient_metadata = resolve_patient_metadata(args, raw_vector, patient_path)

    # Validate critical assets before inference so missing project setup is
    # reported as a concise, actionable error rather than a deep stack trace.
    missing_assets = [
        path for path in (DEFAULT_CHECKPOINT, DEFAULT_PREPROCESSOR)
        if not path.is_file()
    ]
    if missing_assets:
        formatted = "\n".join(f"  - {path}" for path in missing_assets)
        raise FileNotFoundError(
            "Required BioNutriGPT inference assets are missing:\n"
            f"{formatted}\n"
            "Place the Stage 1 checkpoint and preprocessing_artifacts.joblib at the "
            "configured project paths, then rerun this command."
        )

    # Stage 1 diagnostics. Stage 2 uses the same canonical modules and assets.
    stage1_assessment, _device = run_stage1_diagnostics(raw_vector)
    stage2_dossier = run_stage2_counterfactual(raw_vector, patient_metadata)

    unified_dossier = build_unified_dossier(
        raw_vector=raw_vector,
        patient_metadata=patient_metadata,
        stage1_assessment=stage1_assessment,
        stage2_dossier=stage2_dossier,
    )
    _atomic_write_json(unified_dossier, output_path)

    if not args.quiet:
        report = render_clinical_report(
            raw_vector=raw_vector,
            patient_metadata=patient_metadata,
            stage1_assessment=stage1_assessment,
            stage2_dossier=stage2_dossier,
            stage3_certified=unified_dossier["stage3_certified"],
        )
        print(report)
        print(f"\nUnified JSON dossier saved to: {output_path.resolve()}")

    return 0


def _entrypoint() -> int:
    try:
        return main()
    except KeyboardInterrupt:
        print("\nInference cancelled by user.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(_entrypoint())
