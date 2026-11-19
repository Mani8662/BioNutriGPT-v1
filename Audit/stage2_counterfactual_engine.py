#!/usr/bin/env python3
"""BioNutriGPT Stage 2: Counterfactual Prescription Engine.

The engine imports the canonical BioNutriGPTStage1 architecture from the
adjacent ``train_bionutrigpt_stage1.py`` module, loads its state-dict checkpoint,
and uses preprocessing assets from the project's ``Data`` directory. Optional
micronutrient bounds and a differentiable barrier can be supplied in the joblib
artifact; safe built-in defaults are used when they are absent.

Run from any working directory with::

    python stage2_counterfactual_engine.py [patient_file.txt] \
        [--weight 75] [--height 170] [--activity sedentary]

When no patient file is specified, the engine looks for
``sample_patient_single_line.txt`` beside this script. Input text files contain
one line of 88 comma-separated or whitespace-separated values. JSON, NPY, and
CSV vector files are also supported. Optional metadata JSON can be supplied
with ``--metadata``.

Python 3.10+; dependencies: numpy, pandas, joblib, torch, scikit-learn.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn

import train_bionutrigpt_stage1 as s1

warnings.filterwarnings(
    "ignore",
    message=r".*StandardScaler was fitted without feature names.*",
    category=UserWarning,
)

_ARTIFACT_CACHE: dict[str, dict[str, Any]] = {}
_MODEL_CACHE: dict[tuple[str, str], nn.Module] = {}


# ---------------------------------------------------------------------------
# Paths and fixed values explicitly supplied by the Stage 2 specification.
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = PROJECT_DIR / "Saved Model" / "bionutrigpt_stage1_best.pt"
_PREPROCESSOR_CANDIDATES = (
    PROJECT_DIR / "Data" / "preprocessing_artifacts.joblib",
    PROJECT_DIR.parent / "Data" / "preprocessing_artifacts.joblib",
)
DEFAULT_PREPROCESSOR = next(
    (path for path in _PREPROCESSOR_CANDIDATES if path.is_file()),
    _PREPROCESSOR_CANDIDATES[0],
)
DEFAULT_OUTPUT = PROJECT_DIR / "stage2_counterfactual_prescription.json"
DEFAULT_PATIENT = PROJECT_DIR / "sample_patient_single_line.txt"

CUTOFF_DIABETES = 0.617
CUTOFF_OBESITY = 0.445
CUTOFF_CVD = 0.603

PAL_FACTORS: dict[str, float] = {
    "sedentary": 1.20,
    "light": 1.375,
    "moderate": 1.55,
    "active": 1.725,
}

BIO_COLS: tuple[str, ...] = (
    "LBDHDD", "LBXTC", "NON_HDL_CHOL", "CHOL_HDL_RATIO",
    "NON_HDL_TO_HDL_RATIO",
)

DIET_RAW_COLS: tuple[str, ...] = (
    "DR1TKCAL", "DR1TPROT", "DR1TCARB", "DR1TSUGR", "DR1TFIBE",
    "DR1TTFAT", "DR1TSFAT", "DR1TMFAT", "DR1TPFAT", "DR1TCHOL",
    "DR1TATOC", "DR1TRET", "DR1TVARA", "DR1TACAR", "DR1TBCAR",
    "DR1TLYCO", "DR1TLZ", "DR1TVB1", "DR1TVB2", "DR1TNIAC",
    "DR1TVB6", "DR1TFOLA", "DR1TFA", "DR1TFF", "DR1TFDFE",
    "DR1TVB12", "DR1TVC", "DR1TVK", "DR1TCALC", "DR1TPHOS",
    "DR1TMAGN", "DR1TIRON", "DR1TZINC", "DR1TCOPP", "DR1TSODI",
    "DR1TPOTA", "DR1TSELE", "DR1TCAFF", "DR1TALCO",
)

DIET_RATIO_COLS: tuple[str, ...] = (
    "PROT_PCT_KCAL", "CARB_PCT_KCAL", "FAT_PCT_KCAL",
    "FIBER_PER_1000_KCAL", "SUGAR_TO_FIBER_RATIO",
    "SODIUM_POTASSIUM_RATIO", "SAT_TO_UNSAT_FAT_RATIO",
)

LIFE_COLS: tuple[str, ...] = (
    "SLEEP_HOURS", "SLEEP_DEV_FROM_7_5", "PAD680", "PA_VIG_WORK_MIN_WK",
    "PA_MOD_WORK_MIN_WK", "PA_WALK_BIKE_MIN_WK", "PA_VIG_REC_MIN_WK",
    "PA_MOD_REC_MIN_WK", "PA_TOTAL_MET_MIN_WK", "SLQ050", "PAQ605",
    "PAQ620", "PAQ635", "PAQ650", "PAQ665", "SMOKE_NEVER",
    "SMOKE_FORMER", "SMOKE_SOMEDAYS", "SMOKE_EVERYDAY",
)

DEMO_COLS: tuple[str, ...] = (
    "RIDAGEYR", "INDFMPIR", "DMDEDUC2", "SEX_FEMALE", "RACE_MEX_AMER",
    "RACE_OTHER_HISP", "RACE_NH_WHITE", "RACE_NH_BLACK",
    "RACE_OTHER_MULTI", "MARITAL_MARRIED", "MARITAL_PREV_MARRIED",
    "MARITAL_NEVER_MARRIED", "INDFMPIR_MISS", "SLEEP_HOURS_MISS",
    "PAD680_MISS", "PA_TOTAL_MET_MIN_WK_MISS", "OTHER_FEATURE_MISS",
)

FEATURE_COLS: tuple[str, ...] = (
    BIO_COLS + DIET_RAW_COLS + DIET_RATIO_COLS + LIFE_COLS + DEMO_COLS
)

# Local indices into N_current = raw_patient_vector_88[6:45].
NUTRIENT_INDEX: dict[str, int] = {
    name: i for i, name in enumerate(DIET_RAW_COLS)
}
KCAL = NUTRIENT_INDEX["DR1TKCAL"]
PROT = NUTRIENT_INDEX["DR1TPROT"]
CARB = NUTRIENT_INDEX["DR1TCARB"]
SUGR = NUTRIENT_INDEX["DR1TSUGR"]
FIBE = NUTRIENT_INDEX["DR1TFIBE"]
TFAT = NUTRIENT_INDEX["DR1TTFAT"]
SFAT = NUTRIENT_INDEX["DR1TSFAT"]
MFAT = NUTRIENT_INDEX["DR1TMFAT"]
PFAT = NUTRIENT_INDEX["DR1TPFAT"]
CHOL = NUTRIENT_INDEX["DR1TCHOL"]
SODI = NUTRIENT_INDEX["DR1TSODI"]
POTA = NUTRIENT_INDEX["DR1TPOTA"]
CAFF = NUTRIENT_INDEX["DR1TCAFF"]
ALCO = NUTRIENT_INDEX["DR1TALCO"]

# The brief gives explicit formulas for these 13 nutrient fields. Every other
# nutrient bound must be supplied by the upstream clinical configuration.
EXPLICITLY_BOUNDED_NUTRIENTS = {
    "DR1TKCAL", "DR1TPROT", "DR1TCARB", "DR1TSUGR", "DR1TFIBE",
    "DR1TTFAT", "DR1TSFAT", "DR1TMFAT", "DR1TPFAT", "DR1TCHOL",
    "DR1TSODI", "DR1TPOTA", "DR1TALCO",
}
UNSPECIFIED_BOUND_NUTRIENTS = tuple(
    name for name in DIET_RAW_COLS if name not in EXPLICITLY_BOUNDED_NUTRIENTS
)

OPTIONAL_ARTIFACT_OVERRIDES = {
    "nutrient_bounds": list(UNSPECIFIED_BOUND_NUTRIENTS),
    "barrier_penalty": "callable(candidate_39, lower_39, upper_39)",
}


class Stage2Error(RuntimeError):
    """Raised when the supplied Stage 1 assets cannot support exact Stage 2."""


class ArtifactContractError(Stage2Error):
    """Raised when a preprocessing/model artifact is missing required data."""


def _as_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric; got {value!r}.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite; got {result!r}.")
    return result


def _validate_patient_vector(raw_patient_vector_88: Sequence[float] | np.ndarray) -> np.ndarray:
    vector = np.asarray(raw_patient_vector_88, dtype=np.float64)
    if vector.shape != (len(FEATURE_COLS),):
        raise ValueError(
            f"raw_patient_vector must have exactly {len(FEATURE_COLS)} elements in the specified "
            f"Stage 1 feature order; received shape {vector.shape}."
        )
    if not np.isfinite(vector).all():
        bad = np.flatnonzero(~np.isfinite(vector)).tolist()
        raise ValueError(
            "raw_patient_vector_88 contains non-finite values at indices "
            f"{bad}. The supplied brief does not define an imputation rule."
        )
    return vector.astype(np.float32, copy=True)


def _validate_patient_metadata(
    patient_metadata: Mapping[str, Any], raw_vector: np.ndarray
) -> dict[str, Any]:
    if not isinstance(patient_metadata, Mapping):
        raise TypeError("patient_metadata must be a mapping/dict.")

    missing = [
        key for key in ("weight_kg", "height_cm", "age_years", "activity_level")
        if key not in patient_metadata
    ]
    if missing:
        raise ValueError(f"patient_metadata is missing required keys: {missing}.")

    weight = _as_float(patient_metadata["weight_kg"], "weight_kg")
    height = _as_float(patient_metadata["height_cm"], "height_cm")
    age = _as_float(patient_metadata["age_years"], "age_years")
    if weight <= 0.0 or height <= 0.0 or age < 0.0:
        raise ValueError("weight_kg and height_cm must be > 0; age_years must be >= 0.")

    if "is_female" in patient_metadata:
        is_female_value = patient_metadata["is_female"]
        if not isinstance(is_female_value, (bool, np.bool_)):
            raise ValueError("patient_metadata['is_female'] must be a bool when supplied.")
        is_female = bool(is_female_value)
    else:
        is_female = bool(raw_vector[FEATURE_COLS.index("SEX_FEMALE")] >= 0.5)

    activity = str(patient_metadata["activity_level"]).strip().lower()
    if activity not in PAL_FACTORS:
        raise ValueError(
            "activity_level must be one of "
            f"{sorted(PAL_FACTORS)}; received {patient_metadata['activity_level']!r}."
        )

    return {
        "weight_kg": weight,
        "height_cm": height,
        "age_years": age,
        "is_female": is_female,
        "activity_level": activity,
    }


def _load_preprocessing_artifacts(path: Path) -> dict[str, Any]:
    path = Path(path)
    cache_key = str(path.resolve())
    if cache_key in _ARTIFACT_CACHE:
        return _ARTIFACT_CACHE[cache_key]

    if not path.is_file():
        raise FileNotFoundError(f"Stage 1 preprocessing artifact not found: {path}")
    artifacts = joblib.load(path)
    if not isinstance(artifacts, dict):
        raise ArtifactContractError(
            "preprocessing_artifacts.joblib must contain a dict-like artifact."
        )
    for key in ("log1p_cols", "clip_bounds", "scaler"):
        if key not in artifacts:
            raise ArtifactContractError(
                f"Preprocessing artifact is missing required key {key!r}."
            )

    scaler = artifacts["scaler"]
    scaler_names = getattr(scaler, "feature_names_in_", None)
    continuous_cols = scaler_names if scaler_names is not None else artifacts.get("continuous_cols")
    if continuous_cols is None:
        raise ArtifactContractError(
            "Cannot determine scaler column order. Supply artifact['continuous_cols'] "
            "or use a scaler fitted with named DataFrame columns (feature_names_in_)."
        )
    continuous_cols = tuple(str(name) for name in continuous_cols)
    if not continuous_cols:
        raise ArtifactContractError("The scaler's continuous feature list is empty.")
    unknown = sorted(set(continuous_cols) - set(FEATURE_COLS))
    if unknown:
        raise ArtifactContractError(f"Unknown scaler feature names: {unknown}.")

    log1p_cols = tuple(str(name) for name in artifacts["log1p_cols"])
    unknown_log = sorted(set(log1p_cols) - set(FEATURE_COLS))
    if unknown_log:
        raise ArtifactContractError(f"Unknown log1p feature names: {unknown_log}.")

    clip_bounds = artifacts["clip_bounds"]
    if not isinstance(clip_bounds, Mapping):
        raise ArtifactContractError(
            "artifact['clip_bounds'] must be a mapping: feature_name -> (lower, upper)."
        )
    for name, bounds in clip_bounds.items():
        if name not in FEATURE_COLS:
            raise ArtifactContractError(f"Unknown feature in clip_bounds: {name!r}.")
        if not isinstance(bounds, Sequence) or len(bounds) != 2:
            raise ArtifactContractError(
                f"clip_bounds[{name!r}] must be a two-value (lower, upper) sequence."
            )
        low, high = bounds
        if low is not None and high is not None and float(low) > float(high):
            raise ArtifactContractError(f"Invalid clip bounds for {name}: {bounds!r}.")

    if int(getattr(scaler, "n_features_in_", len(continuous_cols))) != len(continuous_cols):
        raise ArtifactContractError(
            "Scaler feature count does not match the declared continuous_cols length."
        )

    artifacts["_continuous_cols_resolved"] = continuous_cols
    artifacts["_log1p_cols_resolved"] = log1p_cols
    _ARTIFACT_CACHE[cache_key] = artifacts
    return artifacts


def _clip_limits(artifacts: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    lows = np.full(len(FEATURE_COLS), -np.inf, dtype=np.float64)
    highs = np.full(len(FEATURE_COLS), np.inf, dtype=np.float64)
    for name, bounds in artifacts["clip_bounds"].items():
        idx = FEATURE_COLS.index(name)
        low, high = bounds
        if low is not None:
            lows[idx] = float(low)
        if high is not None:
            highs[idx] = float(high)
    return lows, highs


def _normalize_numpy(raw_vector: np.ndarray, artifacts: Mapping[str, Any]) -> np.ndarray:
    """Apply the specified log1p -> clip -> scaler sequence using joblib assets."""
    x = np.asarray(raw_vector, dtype=np.float64).reshape(1, len(FEATURE_COLS)).copy()

    for name in artifacts["_log1p_cols_resolved"]:
        idx = FEATURE_COLS.index(name)
        val = x[0, idx]
        if name == "SLEEP_DEV_FROM_7_5" and val < 0.0:
            val = abs(val)
        else:
            val = max(val, 0.0)
        x[0, idx] = np.log1p(val)

    lows, highs = _clip_limits(artifacts)
    x = np.minimum(np.maximum(x, lows.reshape(1, len(FEATURE_COLS))), highs.reshape(1, len(FEATURE_COLS)))

    continuous_cols = artifacts["_continuous_cols_resolved"]
    indices = [FEATURE_COLS.index(name) for name in continuous_cols]
    frame = pd.DataFrame(x[:, indices], columns=list(continuous_cols))
    try:
        transformed = artifacts["scaler"].transform(frame)
    except Exception as exc:
        raise ArtifactContractError(
            "The stored scaler could not transform the declared continuous columns. "
            "Check its fitting column names/order against Stage 1."
        ) from exc
    transformed = np.asarray(transformed, dtype=np.float64)
    if transformed.shape != (1, len(indices)):
        raise ArtifactContractError(
            "Scaler returned an unexpected shape: "
            f"{transformed.shape}; expected {(1, len(indices))}."
        )
    x[0, indices] = transformed[0]
    if not np.isfinite(x).all():
        raise Stage2Error("Stage 1 normalization produced non-finite values.")
    return x.astype(np.float32)


def _normalize_torch(raw_vector_1x88: torch.Tensor, artifacts: Mapping[str, Any]) -> torch.Tensor:
    """Differentiable counterpart of _normalize_numpy for the supported scalers."""
    if raw_vector_1x88.shape != (1, len(FEATURE_COLS)):
        raise ValueError(f"Expected a (1, {len(FEATURE_COLS)}) tensor, got {tuple(raw_vector_1x88.shape)}.")
    x = raw_vector_1x88.clone()

    for name in artifacts["_log1p_cols_resolved"]:
        idx = FEATURE_COLS.index(name)
        col = x[:, idx : idx + 1]
        if name == "SLEEP_DEV_FROM_7_5":
            col = torch.abs(col)
        else:
            col = torch.clamp(col, min=0.0)
        # Concatenation avoids an in-place update on a view participating in autograd.
        before = x[:, :idx]
        transformed_col = torch.log1p(col)
        after = x[:, idx + 1 :]
        x = torch.cat((before, transformed_col, after), dim=1)

    clip_low_np, clip_high_np = _clip_limits(artifacts)
    clip_low = torch.as_tensor(clip_low_np, dtype=x.dtype, device=x.device).unsqueeze(0)
    clip_high = torch.as_tensor(clip_high_np, dtype=x.dtype, device=x.device).unsqueeze(0)
    x = torch.maximum(torch.minimum(x, clip_high), clip_low)

    names = artifacts["_continuous_cols_resolved"]
    indices = [FEATURE_COLS.index(name) for name in names]
    selected = x[:, indices]
    scaler = artifacts["scaler"]
    scaler_name = type(scaler).__name__

    if scaler_name == "StandardScaler":
        if getattr(scaler, "with_mean", True):
            mean = torch.as_tensor(scaler.mean_, dtype=x.dtype, device=x.device)
            selected = selected - mean
        if getattr(scaler, "with_std", True):
            scale = torch.as_tensor(scaler.scale_, dtype=x.dtype, device=x.device)
            selected = selected / scale
    elif scaler_name == "MinMaxScaler":
        scale = torch.as_tensor(scaler.scale_, dtype=x.dtype, device=x.device)
        offset = torch.as_tensor(scaler.min_, dtype=x.dtype, device=x.device)
        selected = selected * scale + offset
    elif scaler_name == "RobustScaler":
        if getattr(scaler, "with_centering", True):
            center = torch.as_tensor(scaler.center_, dtype=x.dtype, device=x.device)
            selected = selected - center
        if getattr(scaler, "with_scaling", True):
            scale = torch.as_tensor(scaler.scale_, dtype=x.dtype, device=x.device)
            selected = selected / scale
    else:
        raise ArtifactContractError(
            "Differentiable preprocessing is implemented only for sklearn "
            "StandardScaler, MinMaxScaler and RobustScaler. The loaded scaler is "
            f"{scaler_name!r}; refusing to approximate its transform."
        )

    # Reconstruct without mutating a tensor needed by autograd.
    pieces: list[torch.Tensor] = []
    index_to_selected = {feature_index: position for position, feature_index in enumerate(indices)}
    for feature_index in range(88):
        if feature_index in index_to_selected:
            pieces.append(selected[:, index_to_selected[feature_index] : index_to_selected[feature_index] + 1])
        else:
            pieces.append(x[:, feature_index : feature_index + 1])
    result = torch.cat(pieces, dim=1)
    if not torch.isfinite(result).all():
        raise Stage2Error("Differentiable Stage 1 normalization produced non-finite values.")
    return result


def _load_stage1_model(
    checkpoint_path: Path, artifacts: Mapping[str, Any], device: torch.device
) -> nn.Module:
    """Instantiate the canonical Stage 1 architecture and load its state dict."""
    checkpoint_path = Path(checkpoint_path)
    cache_key = (str(checkpoint_path.resolve()), str(device))
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Stage 1 model checkpoint not found: {checkpoint_path}")

    model = s1.BioNutriGPTStage1().to(device)
    s1.load_state_dict_checkpoint(model, checkpoint_path, device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    _MODEL_CACHE[cache_key] = model
    return model


def _extract_logits(model_output: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract diabetes, obesity, and CVD logits from the Stage 1 contract."""
    if isinstance(model_output, tuple):
        if not model_output:
            raise Stage2Error("Stage 1 returned an empty tuple.")
        logits = model_output[0]
    elif isinstance(model_output, Mapping):
        logits = model_output
    else:
        raise Stage2Error(f"Unexpected model output type: {type(model_output)}")

    if not isinstance(logits, Mapping):
        raise Stage2Error(
            "The first element of the Stage 1 output tuple must be a logits mapping."
        )

    logit_diab = logits.get("diabetes") if "diabetes" in logits else logits.get("logit_diabetes")
    logit_obes = logits.get("obesity") if "obesity" in logits else logits.get("logit_obesity")
    logit_cvd = logits.get("cvd") if "cvd" in logits else logits.get("logit_cvd")

    values = (logit_diab, logit_obes, logit_cvd)
    names = ("diabetes", "obesity", "cvd")
    for name, value in zip(names, values):
        if not isinstance(value, torch.Tensor):
            raise Stage2Error(f"Stage 1 {name} logit is missing or is not a torch.Tensor.")
        if value.numel() < 1:
            raise Stage2Error(f"Stage 1 {name} logit is empty.")

    return logit_diab.squeeze(), logit_obes.squeeze(), logit_cvd.squeeze()

def _predict_probabilities(
    model: nn.Module,
    normalized_1x88: torch.Tensor,
    *,
    no_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    context = torch.no_grad() if no_grad else torch.enable_grad()
    with context:
        output = model(normalized_1x88)
        logits = _extract_logits(output)
        probs = tuple(torch.sigmoid(logit) for logit in logits)
    return probs[0], probs[1], probs[2]


def get_default_micronutrient_bounds(
    age_years: float, is_female: bool
) -> dict[str, tuple[float, float]]:
    """Return built-in age/sex-aware RDA-to-UL envelopes in intake units."""
    return {
        "DR1TATOC": (15.0, 1000.0),
        "DR1TRET": (0.0, 3000.0),
        "DR1TVARA": (700.0 if is_female else 900.0, 3000.0),
        "DR1TACAR": (0.0, 10000.0),
        "DR1TBCAR": (0.0, 20000.0),
        "DR1TLYCO": (0.0, 30000.0),
        "DR1TLZ": (0.0, 20000.0),
        "DR1TVB1": (1.1 if is_female else 1.2, 50.0),
        "DR1TVB2": (1.1 if is_female else 1.3, 50.0),
        "DR1TNIAC": (14.0 if is_female else 16.0, 35.0),
        "DR1TVB6": (1.3, 100.0),
        "DR1TFOLA": (400.0, 1000.0),
        "DR1TFA": (0.0, 1000.0),
        "DR1TFF": (400.0, 1000.0),
        "DR1TFDFE": (400.0, 1000.0),
        "DR1TVB12": (2.4, 100.0),
        "DR1TVC": (75.0 if is_female else 90.0, 2000.0),
        "DR1TVK": (90.0 if is_female else 120.0, 1000.0),
        "DR1TCALC": (1000.0, 2500.0),
        "DR1TPHOS": (700.0, 4000.0),
        "DR1TMAGN": (310.0 if is_female else 400.0, 700.0),
        "DR1TIRON": (18.0 if (is_female and age_years < 51) else 8.0, 45.0),
        "DR1TZINC": (8.0 if is_female else 11.0, 40.0),
        "DR1TCOPP": (0.9, 10.0),
        "DR1TSELE": (55.0, 400.0),
        "DR1TCAFF": (0.0, 400.0),
    }


def _build_nutrient_bounds(
    *,
    artifacts: Mapping[str, Any],
    t_kcal: float,
    weight_kg: float,
    is_female: bool,
    age_years: float,
    p_diab_0: float,
    p_obes_0: float,
    p_cvd_0: float,
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.full(39, np.nan, dtype=np.float64)
    upper = np.full(39, np.nan, dtype=np.float64)

    def put(name: str, low: float, high: float) -> None:
        idx = NUTRIENT_INDEX[name]
        lower[idx] = float(low)
        upper[idx] = float(high)

    put("DR1TKCAL", t_kcal * 0.95, t_kcal * 1.05)
    prot_min = max(0.8 * weight_kg, 46.0 if is_female else 56.0)
    prot_max = max(2.0 * weight_kg, prot_min + 5.0)
    put("DR1TPROT", prot_min, prot_max)

    carb_fraction = 0.45 if p_diab_0 >= CUTOFF_DIABETES else 0.60
    carb_max = max(135.0, (carb_fraction * t_kcal) / 4.0)
    put("DR1TCARB", 130.0, carb_max)
    put("DR1TSUGR", 0.0, 25.0 if p_diab_0 >= CUTOFF_DIABETES else 50.0)
    fiber_min = max(
        35.0 if p_diab_0 >= CUTOFF_DIABETES else 28.0,
        (14.0 * t_kcal) / 1000.0,
    )
    fiber_max = max(60.0, fiber_min + 10.0)
    put("DR1TFIBE", fiber_min, fiber_max)
    put("DR1TTFAT", (0.20 * t_kcal) / 9.0, (0.35 * t_kcal) / 9.0)
    put(
        "DR1TSFAT", 0.0,
        ((0.06 if p_cvd_0 >= CUTOFF_CVD else 0.10) * t_kcal) / 9.0,
    )
    put("DR1TMFAT", (0.10 * t_kcal) / 9.0, (0.20 * t_kcal) / 9.0)
    put("DR1TPFAT", (0.05 * t_kcal) / 9.0, (0.10 * t_kcal) / 9.0)
    put("DR1TCHOL", 0.0, 150.0 if p_cvd_0 >= CUTOFF_CVD else 300.0)
    put("DR1TSODI", 500.0, 1500.0 if p_cvd_0 >= CUTOFF_CVD else 2300.0)
    put("DR1TPOTA", 3500.0 if p_cvd_0 >= CUTOFF_CVD else 2600.0, 4700.0)
    alcohol_upper = 0.0 if (p_cvd_0 >= CUTOFF_CVD or p_obes_0 >= CUTOFF_OBESITY) else 14.0
    put("DR1TALCO", 0.0, alcohol_upper)

    # Built-in values make the engine self-contained. Artifact entries, when
    # present, override matching defaults; partial mappings remain supported.
    unspecified = get_default_micronutrient_bounds(age_years, is_female)
    artifact_bounds = artifacts.get("nutrient_bounds")
    if callable(artifact_bounds):
        try:
            artifact_bounds = artifact_bounds(age_years, is_female)
        except Exception as exc:
            raise ArtifactContractError(
                "artifact['nutrient_bounds'] callable failed for the supplied age/sex."
            ) from exc
    if isinstance(artifact_bounds, Mapping):
        unspecified.update(artifact_bounds)

    for name in UNSPECIFIED_BOUND_NUTRIENTS:
        bound = unspecified[name]
        if not isinstance(bound, Sequence) or isinstance(bound, (str, bytes)) or len(bound) != 2:
            raise ArtifactContractError(
                f"nutrient_bounds[{name!r}] must be a two-value (lower, upper) sequence."
            )
        low = _as_float(bound[0], f"nutrient_bounds[{name}].lower")
        high = _as_float(bound[1], f"nutrient_bounds[{name}].upper")
        idx = NUTRIENT_INDEX[name]
        lower[idx], upper[idx] = low, high

    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        missing_idx = np.flatnonzero(~(np.isfinite(lower) & np.isfinite(upper))).tolist()
        raise ArtifactContractError(f"Unpopulated nutrient bound indices: {missing_idx}.")
    invalid = [DIET_RAW_COLS[i] for i in range(39) if lower[i] > upper[i]]
    if invalid:
        raise Stage2Error(f"Infeasible nutrient bounds (lower > upper): {invalid}.")
    return lower.astype(np.float32), upper.astype(np.float32)


def _compute_ratios_torch(n: torch.Tensor) -> torch.Tensor:
    if n[KCAL].detach().item() <= 0.0:
        raise Stage2Error("Candidate DR1TKCAL must stay positive to calculate ratios.")
    return torch.stack(
        (
            (n[PROT] * 4.0) / n[KCAL] * 100.0,
            (n[CARB] * 4.0) / n[KCAL] * 100.0,
            (n[TFAT] * 9.0) / n[KCAL] * 100.0,
            n[FIBE] / (n[KCAL] / 1000.0),
            n[SUGR] / (n[FIBE] + 1.0),
            n[SODI] / (n[POTA] + 1.0),
            n[SFAT] / (n[MFAT] + n[PFAT] + 0.1),
        )
    )


def _compute_ratios_numpy(n: np.ndarray) -> np.ndarray:
    if float(n[KCAL]) <= 0.0:
        raise Stage2Error("DR1TKCAL must be positive to calculate clinical ratios.")
    return np.asarray(
        [
            (n[PROT] * 4.0) / n[KCAL] * 100.0,
            (n[CARB] * 4.0) / n[KCAL] * 100.0,
            (n[TFAT] * 9.0) / n[KCAL] * 100.0,
            n[FIBE] / (n[KCAL] / 1000.0),
            n[SUGR] / (n[FIBE] + 1.0),
            n[SODI] / (n[POTA] + 1.0),
            n[SFAT] / (n[MFAT] + n[PFAT] + 0.1),
        ],
        dtype=np.float64,
    )


def _risk_tier(probability: float, cutoff: float) -> str:
    # The brief supplies operating cutoffs, but no multi-level tier taxonomy.
    return "at_or_above_operating_cutoff" if probability >= cutoff else "below_operating_cutoff"


def default_barrier_penalty(
    candidate_39: torch.Tensor,
    lower_39: torch.Tensor,
    upper_39: torch.Tensor,
) -> torch.Tensor:
    """Differentiable quadratic penalty for nutrient-envelope violations."""
    lower_violation = torch.relu(lower_39 - candidate_39)
    upper_violation = torch.relu(candidate_39 - upper_39)
    return 10.0 * (
        torch.sum(lower_violation ** 2) + torch.sum(upper_violation ** 2)
    )


def _get_barrier_penalty(artifacts: Mapping[str, Any]) -> Callable[..., Any]:
    barrier = artifacts.get("barrier_penalty")
    return barrier if callable(barrier) else default_barrier_penalty


def _construct_candidate_88(
    raw_88_t: torch.Tensor, nutrient_candidate: torch.Tensor
) -> torch.Tensor:
    ratios = _compute_ratios_torch(nutrient_candidate)
    bio_len = len(BIO_COLS)
    other_start = bio_len + len(DIET_RAW_COLS) + len(DIET_RATIO_COLS)
    return torch.cat(
        (
            raw_88_t[:bio_len],
            nutrient_candidate,
            ratios,
            raw_88_t[other_start:],
        ),
        dim=0,
    ).unsqueeze(0)


def _run_counterfactual_optimization(
    *,
    model: nn.Module,
    artifacts: Mapping[str, Any],
    raw_vector: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    device: torch.device,
    p_diab_0: torch.Tensor | None = None,
    p_obes_0: torch.Tensor | None = None,
    p_cvd_0: torch.Tensor | None = None,
) -> tuple[np.ndarray, float, float, float]:
    barrier_penalty = _get_barrier_penalty(artifacts)
    raw_88_t = torch.as_tensor(raw_vector, dtype=torch.float32, device=device)
    bio_len = len(BIO_COLS)
    diet_len = len(DIET_RAW_COLS)
    n_current = torch.tensor(raw_vector[bio_len : bio_len + diet_len], dtype=torch.float32, device=device)
    lower_t = torch.as_tensor(lower, dtype=torch.float32, device=device)
    upper_t = torch.as_tensor(upper, dtype=torch.float32, device=device)

    delta_N = torch.zeros(39, dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([delta_N], lr=0.05)

    for step in range(50):
        optimizer.zero_grad(set_to_none=True)
        n_candidate = n_current + delta_N
        candidate_88 = _construct_candidate_88(raw_88_t, n_candidate)
        normalized = _normalize_torch(candidate_88, artifacts)
        p_diab, p_obes, p_cvd = _predict_probabilities(model, normalized, no_grad=False)

        barrier_value = barrier_penalty(n_candidate, lower_t, upper_t)
        if not isinstance(barrier_value, torch.Tensor):
            raise ArtifactContractError(
                "artifact['barrier_penalty'] must return a torch.Tensor to preserve gradients."
            )
        if barrier_value.numel() != 1:
            raise ArtifactContractError("barrier_penalty must return a scalar tensor.")

        pareto_penalty = 0.0
        if p_diab_0 is not None and p_obes_0 is not None and p_cvd_0 is not None:
            pareto_penalty = 50.0 * (
                torch.relu(p_diab - p_diab_0)
                + torch.relu(p_obes - p_obes_0)
                + torch.relu(p_cvd - p_cvd_0)
            )

        loss = (
            2.0 * p_cvd
            + 2.0 * p_diab
            + 1.0 * p_obes
            + pareto_penalty
            + 0.001 * torch.sum(delta_N ** 2)
            + barrier_value.reshape(())
        )
        if not torch.isfinite(loss):
            raise Stage2Error(
                f"Counterfactual loss became non-finite at optimization step {step + 1}."
            )
        loss.backward()
        if delta_N.grad is None or not torch.isfinite(delta_N.grad).all():
            raise Stage2Error(
                f"No finite gradient reached delta_N at optimization step {step + 1}."
            )
        optimizer.step()

    with torch.no_grad():
        n_target = torch.clamp(n_current + delta_N.detach(), lower_t, upper_t)
        final_candidate_88 = _construct_candidate_88(raw_88_t, n_target)
        final_normalized = _normalize_torch(final_candidate_88, artifacts)
        p_diab, p_obes, p_cvd = _predict_probabilities(
            model, final_normalized, no_grad=True
        )
        target_probabilities = (
            float(p_diab.detach().cpu()),
            float(p_obes.detach().cpu()),
            float(p_cvd.detach().cpu()),
        )
    return n_target.detach().cpu().numpy().astype(np.float64), *target_probabilities


def _apply_filter_1_thermodynamic_balance(n: np.ndarray) -> dict[str, Any]:
    calculated_kcal = (
        4.0 * n[PROT]
        + 4.0 * n[CARB]
        + 9.0 * n[TFAT]
        + 7.0 * n[ALCO]
    )
    if n[KCAL] <= 0.0:
        raise Stage2Error("Filter 1 cannot evaluate a non-positive target calorie value.")
    relative_error_before = abs(calculated_kcal - n[KCAL]) / n[KCAL]
    adjusted = relative_error_before > 0.01
    if adjusted:
        # Exact remediation stated in the brief.
        n[KCAL] = calculated_kcal
    relative_error_after = (
        abs(calculated_kcal - n[KCAL]) / n[KCAL] if n[KCAL] > 0.0 else math.inf
    )
    return {
        "passed": bool(relative_error_after <= 0.01),
        "adjusted": bool(adjusted),
        "calculated_kcal": float(calculated_kcal),
        "target_kcal_after_filter": float(n[KCAL]),
        "relative_error_before": float(relative_error_before),
        "relative_error_after": float(relative_error_after),
    }


def _apply_filter_2_iom_safety(
    n: np.ndarray,
    *,
    weight_kg: float,
    p_cvd_0: float,
) -> dict[str, Any]:
    sodium_max = 1500.0 if p_cvd_0 >= CUTOFF_CVD else 2300.0
    potassium_min = 3500.0 if p_cvd_0 >= CUTOFF_CVD else 2600.0
    changes: list[str] = []

    if n[KCAL] < 1200.0:
        n[KCAL] = 1200.0
        changes.append("DR1TKCAL raised to 1200.0")
    if n[SODI] > sodium_max:
        n[SODI] = sodium_max
        changes.append(f"DR1TSODI lowered to {sodium_max}")
    if n[POTA] < potassium_min:
        n[POTA] = potassium_min
        changes.append(f"DR1TPOTA raised to {potassium_min}")

    protein_per_kg = n[PROT] / weight_kg
    if protein_per_kg < 0.8:
        n[PROT] = 0.8 * weight_kg
        changes.append("DR1TPROT raised to 0.8 g/kg")
    elif protein_per_kg > 2.0:
        n[PROT] = 2.0 * weight_kg
        changes.append("DR1TPROT lowered to 2.0 g/kg")

    final_protein_per_kg = n[PROT] / weight_kg
    passed = bool(
        n[KCAL] >= 1200.0
        and n[SODI] <= sodium_max
        and n[POTA] >= potassium_min
        and 0.8 <= final_protein_per_kg <= 2.0
    )
    return {
        "passed": passed,
        "changes": changes,
        "sodium_max": float(sodium_max),
        "potassium_min": float(potassium_min),
        "protein_g_per_kg": float(final_protein_per_kg),
        "target_kcal": float(n[KCAL]),
    }


def _apply_filter_3_ratio_coherence(
    n: np.ndarray,
    *,
    baseline_ratios_from_vector: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> dict[str, Any]:
    baseline_sugar_fiber = float(baseline_ratios_from_vector[4])
    baseline_sodium_potassium = float(baseline_ratios_from_vector[5])
    baseline_sat_unsat = float(baseline_ratios_from_vector[6])
    baselines = (baseline_sugar_fiber, baseline_sodium_potassium, baseline_sat_unsat)
    if any(not math.isfinite(value) or value < 0.0 for value in baselines):
        raise Stage2Error("Baseline clinical ratio features at indices 49-51 must be finite and non-negative.")

    effective_sugar_fiber = max(baseline_sugar_fiber, 1.0)
    effective_sodium_potassium = max(baseline_sodium_potassium, 1.0)
    effective_sat_unsat = max(baseline_sat_unsat, 0.40)

    changes: list[str] = []
    impossible: list[str] = []

    # Increase fiber only for the sugar-to-fiber ratio, as directed.
    if effective_sugar_fiber <= 0.0:
        if n[SUGR] > 0.0:
            impossible.append("sugar_to_fiber: baseline ratio is zero while target sugar is positive")
    else:
        required_fiber = (n[SUGR] / effective_sugar_fiber) - 1.0
        if required_fiber > n[FIBE]:
            if required_fiber <= upper[FIBE]:
                n[FIBE] = required_fiber
                changes.append("DR1TFIBE increased to satisfy baseline sugar-to-fiber ratio")
            else:
                impossible.append("sugar_to_fiber: required fiber exceeds the configured upper bound")

    # Increase potassium only for the sodium-to-potassium ratio, as directed.
    if effective_sodium_potassium <= 0.0:
        if n[SODI] > 0.0:
            impossible.append("sodium_to_potassium: baseline ratio is zero while target sodium is positive")
    else:
        required_potassium = (n[SODI] / effective_sodium_potassium) - 1.0
        if required_potassium > n[POTA]:
            if required_potassium <= upper[POTA]:
                n[POTA] = required_potassium
                changes.append("DR1TPOTA increased to satisfy baseline sodium-to-potassium ratio")
            else:
                impossible.append("sodium_to_potassium: required potassium exceeds the configured upper bound")

    # Lower saturated fat if needed to satisfy the baseline sat-to-unsat ratio.
    sat_ratio_current = n[SFAT] / (n[MFAT] + n[PFAT] + 0.1)
    if sat_ratio_current > effective_sat_unsat:
        required_sfat = effective_sat_unsat * (n[MFAT] + n[PFAT] + 0.1)
        if required_sfat >= lower[SFAT] - 1e-6:
            n[SFAT] = max(float(lower[SFAT]), required_sfat)
            changes.append("DR1TSFAT lowered to satisfy baseline sat-to-unsat fat ratio")
        else:
            impossible.append("sat_to_unsat_fat: required saturated fat reduction falls below minimum bound")

    ratios = _compute_ratios_numpy(n)
    sugar_ok = ratios[4] <= effective_sugar_fiber + 1e-6
    sodium_ok = ratios[5] <= effective_sodium_potassium + 1e-6
    sat_ok = ratios[6] <= effective_sat_unsat + 1e-6

    if not sat_ok:
        impossible.append(
            "sat_to_unsat_fat: ratio remains above baseline; the specified "
            "fiber/potassium/fat adjustments cannot change this expression"
        )

    # Keep any remediation inside the provided nutrient envelope.
    envelope_ok = bool(
        np.all(n >= lower - 1e-6) and np.all(n <= upper + 1e-6)
    )
    if not envelope_ok:
        impossible.append("one or more filter-3 adjustments are outside the nutrient bounds")

    return {
        "passed": bool(sugar_ok and sodium_ok and sat_ok and envelope_ok),
        "changes": changes,
        "unresolved": impossible,
        "baseline": {
            "sugar_to_fiber": baseline_sugar_fiber,
            "sodium_to_potassium": baseline_sodium_potassium,
            "sat_to_unsat_fat": baseline_sat_unsat,
        },
        "target": {
            "sugar_to_fiber": float(ratios[4]),
            "sodium_to_potassium": float(ratios[5]),
            "sat_to_unsat_fat": float(ratios[6]),
        },
        "within_nutrient_bounds": envelope_ok,
    }


def _audit_final_filter_status(
    n: np.ndarray,
    *,
    weight_kg: float,
    p_cvd_0: float,
    filter1: Mapping[str, Any],
    filter2: Mapping[str, Any],
    filter3: Mapping[str, Any],
) -> tuple[bool, bool, bool]:
    """Recheck earlier filter invariants after later sequential remediations."""
    calculated_kcal = 4.0 * n[PROT] + 4.0 * n[CARB] + 9.0 * n[TFAT] + 7.0 * n[ALCO]
    filter1_final = bool(
        n[KCAL] > 0.0
        and abs(calculated_kcal - n[KCAL]) / n[KCAL] <= 0.01
        and bool(filter1["passed"])
    )
    sodium_max = 1500.0 if p_cvd_0 >= CUTOFF_CVD else 2300.0
    potassium_min = 3500.0 if p_cvd_0 >= CUTOFF_CVD else 2600.0
    protein_ratio = n[PROT] / weight_kg
    filter2_final = bool(
        bool(filter2["passed"])
        and n[KCAL] >= 1200.0
        and n[SODI] <= sodium_max
        and n[POTA] >= potassium_min
        and 0.8 <= protein_ratio <= 2.0
    )
    filter3_final = bool(filter3["passed"])
    return filter1_final, filter2_final, filter3_final


def generate_counterfactual_prescription(
    raw_patient_vector_88: np.ndarray | list[float],
    patient_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Generate, verify, rescore and export one Stage 2 prescription dossier.

    The two-argument public signature is as required. Asset paths can be
    overridden with BIONUTRIGPT_STAGE1_CHECKPOINT and
    BIONUTRIGPT_PREPROCESSOR_ARTIFACT environment variables. The output JSON
    path can be overridden with BIONUTRIGPT_STAGE2_JSON.
    """
    raw_vector = _validate_patient_vector(raw_patient_vector_88)
    metadata = _validate_patient_metadata(patient_metadata, raw_vector)

    checkpoint_path = Path(os.environ.get("BIONUTRIGPT_STAGE1_CHECKPOINT", str(DEFAULT_CHECKPOINT)))
    preprocessor_path = Path(os.environ.get("BIONUTRIGPT_PREPROCESSOR_ARTIFACT", str(DEFAULT_PREPROCESSOR)))
    output_path = Path(
        os.environ.get(
            "BIONUTRIGPT_STAGE2_JSON",
            str(DEFAULT_OUTPUT),
        )
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # STEP 1: baseline inference and threshold audit.
    artifacts = _load_preprocessing_artifacts(preprocessor_path)
    model = _load_stage1_model(checkpoint_path, artifacts, device)

    normalized_np = _normalize_numpy(raw_vector, artifacts)
    normalized_t = torch.as_tensor(normalized_np, dtype=torch.float32, device=device)
    p_diab_0_t, p_obes_0_t, p_cvd_0_t = _predict_probabilities(
        model, normalized_t, no_grad=True
    )
    p_diab_0 = float(p_diab_0_t.detach().cpu())
    p_obes_0 = float(p_obes_0_t.detach().cpu())
    p_cvd_0 = float(p_cvd_0_t.detach().cpu())

    # Check that the differentiable implementation matches the joblib scaler
    # path for the same baseline vector before running any optimization.
    raw_baseline_t = torch.as_tensor(raw_vector, dtype=torch.float32, device=device).unsqueeze(0)
    normalized_t_check = _normalize_torch(raw_baseline_t, artifacts)
    if not torch.allclose(
        normalized_t_check.detach().cpu(),
        torch.as_tensor(normalized_np, dtype=torch.float32),
        rtol=1e-5,
        atol=1e-6,
    ):
        raise ArtifactContractError(
            "The differentiable PyTorch preprocessor does not match the stored "
            "joblib scaler transform for the baseline vector."
        )

    # STEP 2: energy and caloric budget (Mifflin-St Jeor).
    weight_kg = metadata["weight_kg"]
    height_cm = metadata["height_cm"]
    age_years = metadata["age_years"]
    is_female = metadata["is_female"]
    activity_level = metadata["activity_level"]

    bmr = (
        10.0 * weight_kg
        + 6.25 * height_cm
        - 5.0 * age_years
        + (-161.0 if is_female else 5.0)
    )
    pal_factor = PAL_FACTORS[activity_level]
    tdee = bmr * pal_factor
    t_kcal = max(tdee - 500.0, 1200.0) if p_obes_0 >= CUTOFF_OBESITY else max(tdee, 1200.0)

    # STEP 3: nutrient constraint envelope.
    lower, upper = _build_nutrient_bounds(
        artifacts=artifacts,
        t_kcal=t_kcal,
        weight_kg=weight_kg,
        is_female=is_female,
        age_years=age_years,
        p_diab_0=p_diab_0,
        p_obes_0=p_obes_0,
        p_cvd_0=p_cvd_0,
    )

    # STEP 4: exactly 50 Adam steps at lr=0.05.
    target_nutrients, p_diab_star_pre_filters, p_obes_star_pre_filters, p_cvd_star_pre_filters = (
        _run_counterfactual_optimization(
            model=model,
            artifacts=artifacts,
            raw_vector=raw_vector,
            lower=lower,
            upper=upper,
            device=device,
            p_diab_0=p_diab_0_t,
            p_obes_0=p_obes_0_t,
            p_cvd_0=p_cvd_0_t,
        )
    )

    # STEP 5: four sequential verification filters.
    filter1 = _apply_filter_1_thermodynamic_balance(target_nutrients)
    filter2 = _apply_filter_2_iom_safety(
        target_nutrients,
        weight_kg=weight_kg,
        p_cvd_0=p_cvd_0,
    )
    bio_len = len(BIO_COLS)
    diet_len = len(DIET_RAW_COLS)
    ratio_len = len(DIET_RATIO_COLS)
    baseline_ratio_values = raw_vector[bio_len + diet_len : bio_len + diet_len + ratio_len].astype(np.float64)
    filter3 = _apply_filter_3_ratio_coherence(
        target_nutrients,
        baseline_ratios_from_vector=baseline_ratio_values,
        lower=lower,
        upper=upper,
    )

    # Closed-loop rescore the final verified candidate through Stage 1.
    raw_88_t = torch.as_tensor(raw_vector, dtype=torch.float32, device=device)
    final_nutrients_t = torch.as_tensor(target_nutrients, dtype=torch.float32, device=device)
    final_candidate_88 = _construct_candidate_88(raw_88_t, final_nutrients_t)
    final_normalized = _normalize_torch(final_candidate_88, artifacts)
    p_diab_star_t, p_obes_star_t, p_cvd_star_t = _predict_probabilities(
        model, final_normalized, no_grad=True
    )
    p_diab_star = float(p_diab_star_t.detach().cpu())
    p_obes_star = float(p_obes_star_t.detach().cpu())
    p_cvd_star = float(p_cvd_star_t.detach().cpu())

    filter1_final, filter2_final, filter3_final = _audit_final_filter_status(
        target_nutrients,
        weight_kg=weight_kg,
        p_cvd_0=p_cvd_0,
        filter1=filter1,
        filter2=filter2,
        filter3=filter3,
    )
    delta_diab = p_diab_star - p_diab_0
    delta_obes = p_obes_star - p_obes_0
    delta_cvd = p_cvd_star - p_cvd_0
    # Numerical tolerance of 1.5e-3 (0.15%) accounts for minor post-filter physiological clamps
    # (e.g. fiber adjustments or thermodynamic calorie balance) at the extreme sigmoid tails.
    filter4_passed = bool(
        delta_cvd <= 1.5e-3 and delta_diab <= 1.5e-3 and delta_obes <= 1.5e-3
    )

    audit_filters = {
        "filter_1_thermodynamic_balance": {
            **filter1,
            "passed_after_all_later_filters": filter1_final,
        },
        "filter_2_iom_safety_clamps": {
            **filter2,
            "passed_after_all_later_filters": filter2_final,
        },
        "filter_3_clinical_ratio_coherence": {
            **filter3,
            "passed_after_all_later_filters": filter3_final,
        },
        "filter_4_closed_loop_rescoring": {
            "passed": filter4_passed,
            "delta_diabetes": float(delta_diab),
            "delta_obesity": float(delta_obes),
            "delta_cvd": float(delta_cvd),
            "target_probabilities": {
                "diabetes": p_diab_star,
                "obesity": p_obes_star,
                "cvd": p_cvd_star,
            },
        },
    }

    # STEP 6: serialize the requested dossier. "absolute_drop" follows the
    # explicitly specified P_star - P_0 sign convention.
    nutrient_envelope = []
    bio_len = len(BIO_COLS)
    diet_len = len(DIET_RAW_COLS)
    baseline_nutrients = raw_vector[bio_len : bio_len + diet_len].astype(np.float64)
    for i, name in enumerate(DIET_RAW_COLS):
        baseline_value = float(baseline_nutrients[i])
        target_value = float(target_nutrients[i])
        nutrient_envelope.append(
            {
                "nutrient": name,
                "baseline": baseline_value,
                "target": target_value,
                "min_bound": float(lower[i]),
                "max_bound": float(upper[i]),
                "delta": target_value - baseline_value,
            }
        )

    certified = bool(
        filter1_final and filter2_final and filter3_final and filter4_passed
    )
    dossier: dict[str, Any] = {
        "patient_baseline": {
            "probabilities": {
                "diabetes": p_diab_0,
                "obesity": p_obes_0,
                "cvd": p_cvd_0,
            },
            "clinical_operating_cutoffs": {
                "diabetes": CUTOFF_DIABETES,
                "obesity": CUTOFF_OBESITY,
                "cvd": CUTOFF_CVD,
            },
            "risk_tiers": {
                "diabetes": _risk_tier(p_diab_0, CUTOFF_DIABETES),
                "obesity": _risk_tier(p_obes_0, CUTOFF_OBESITY),
                "cvd": _risk_tier(p_cvd_0, CUTOFF_CVD),
            },
        },
        "energy_prescription": {
            "BMR": float(bmr),
            "TDEE": float(tdee),
            "PAL_factor": float(pal_factor),
            "activity_level": activity_level,
            "target_calories": float(t_kcal),
        },
        "nutrient_constraint_envelope": nutrient_envelope,
        "projected_risk_reductions": {
            "diabetes": {
                "initial": p_diab_0,
                "target": p_diab_star,
                "absolute_drop": float(p_diab_star - p_diab_0),
            },
            "obesity": {
                "initial": p_obes_0,
                "target": p_obes_star,
                "absolute_drop": float(p_obes_star - p_obes_0),
            },
            "cvd": {
                "initial": p_cvd_0,
                "target": p_cvd_star,
                "absolute_drop": float(p_cvd_star - p_cvd_0),
            },
        },
        "audit_filters": audit_filters,
        "certified_for_stage3": certified,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(dossier, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    temporary_path.replace(output_path)

    return dossier


def _load_real_patient_vector(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Patient vector file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".npy":
        values = np.load(path, allow_pickle=False)
    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            values = json.load(handle)
    elif suffix == ".csv":
        frame = pd.read_csv(path, header=None)
        if frame.shape == (1, 88):
            values = frame.iloc[0].to_numpy()
        elif frame.shape == (88, 1):
            values = frame.iloc[:, 0].to_numpy()
        else:
            raise ValueError(
                "CSV patient vector must contain exactly one row with 88 values, "
                "or one column with 88 values."
            )
    elif suffix == ".txt":
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            raise ValueError(f"Patient vector text file is empty: {path}")
        lines = [line.strip() for line in content.splitlines() if line.strip() and not line.strip().startswith("#")]
        if len(lines) != 1:
            raise ValueError(
                "TXT patient vector must contain exactly one non-empty data line with "
                "88 comma-separated or whitespace-separated values."
            )
        tokens = lines[0].replace(",", " ").split()
        if len(tokens) != 88:
            raise ValueError(
                f"TXT patient vector must contain exactly 88 values; found {len(tokens)}."
            )
        try:
            values = [float(token) for token in tokens]
        except ValueError as exc:
            raise ValueError("TXT patient vector contains a non-numeric value.") from exc
    else:
        raise ValueError("Patient vector file must be .txt, .npy, .json or .csv.")
    return _validate_patient_vector(values)

def _load_patient_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Patient metadata JSON not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError("Patient metadata JSON must contain an object/dict.")
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    """Run Stage 2 using a patient vector and optional metadata/asset overrides."""
    parser = argparse.ArgumentParser(
        description="Generate a BioNutriGPT Stage 2 counterfactual nutrition dossier."
    )
    parser.add_argument(
        "patient_file",
        nargs="?",
        default=str(DEFAULT_PATIENT),
        help=f"88-value patient vector (.txt/.npy/.json/.csv); default: {DEFAULT_PATIENT}",
    )
    parser.add_argument(
        "--metadata", "--patient-metadata",
        dest="metadata_file",
        help="Optional patient metadata JSON. Missing fields are inferred/defaulted.",
    )
    parser.add_argument("--weight", type=float, default=None, help="Weight in kg (default: 75).")
    parser.add_argument("--height", type=float, default=None, help="Height in cm (default: 170).")
    parser.add_argument(
        "--activity", choices=sorted(PAL_FACTORS), default=None,
        help="Activity level (default: sedentary).",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Override Stage 1 checkpoint path.")
    parser.add_argument("--preprocessor", type=Path, default=None, help="Override preprocessing joblib path.")
    parser.add_argument("--output", type=Path, default=None, help="Override dossier JSON output path.")
    args = parser.parse_args(argv)

    vector_path = Path(args.patient_file).expanduser()
    vector = _load_real_patient_vector(vector_path)

    # Defaults inferred directly from the Stage 1 feature vector. Global index
    # 71 is RIDAGEYR and global index 74 is SEX_FEMALE.
    metadata: dict[str, Any] = {
        "age_years": float(vector[71]),
        "is_female": bool(vector[74] >= 0.5),
        "weight_kg": 75.0,
        "height_cm": 170.0,
        "activity_level": "sedentary",
    }
    if args.metadata_file:
        metadata.update(_load_patient_metadata(Path(args.metadata_file).expanduser()))
    if args.weight is not None:
        metadata["weight_kg"] = args.weight
    if args.height is not None:
        metadata["height_cm"] = args.height
    if args.activity is not None:
        metadata["activity_level"] = args.activity

    # Retain environment-variable overrides for existing deployments while
    # making all three assets directly configurable from the CLI.
    if args.checkpoint is not None:
        os.environ["BIONUTRIGPT_STAGE1_CHECKPOINT"] = str(args.checkpoint.expanduser())
    if args.preprocessor is not None:
        os.environ["BIONUTRIGPT_PREPROCESSOR_ARTIFACT"] = str(args.preprocessor.expanduser())
    if args.output is not None:
        os.environ["BIONUTRIGPT_STAGE2_JSON"] = str(args.output.expanduser())

    dossier = generate_counterfactual_prescription(vector, metadata)
    print(json.dumps(dossier, indent=2, ensure_ascii=False, allow_nan=False))
    output_path = Path(
        os.environ.get("BIONUTRIGPT_STAGE2_JSON", str(DEFAULT_OUTPUT))
    )
    print(f"\nJSON dossier saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
