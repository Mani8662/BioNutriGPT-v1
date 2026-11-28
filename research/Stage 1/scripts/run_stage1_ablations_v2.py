#!/usr/bin/env python3
"""BioNutriGPT Stage 1 Architecture Ablation Runner (Version 2).

Controlled 5-seed architecture ablations on the frozen repaired splits:
  - FULL (existing BioNutriGPT architecture)
  - NO_MOE (single dense FFN 512->256->256 replacing 4-expert MoE; aux loss=0)
  - MEAN_POOLING (simple group mean of modality tokens replacing attention pooling)
  - NO_MODALITY_EMBEDDINGS (feature tokenization without learned modality vectors)
  - NO_HEAD_SHORTCUTS (heads receive only 256-d patient state; shortcuts removed)

Seeds: [42, 123, 456, 789, 2026].
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import random
import threading
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import psutil
from scipy.special import expit, logit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
import torch
from torch import nn
import torch.nn.functional as F

from train_bionutrigpt_stage1 import (
    BIO_COLS,
    DEMO_COLS,
    DIET_COLS,
    FEATURE_COLS,
    LIFE_COLS,
    MODALITY_SIZES,
    NUM_FEATURES,
    TARGET_COLS,
    TARGET_INDEX,
    FeatureTokenizer,
    ModalityAttentionPooler,
    SparseMoELayer,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Data/stage1_repaired_v2"
PROTOCOL_PATH = ROOT / "stage1_ablation_protocol_v2.json"
RESULTS_DIR = ROOT / "Saved Model/stage1_ablation_v2"
TARGETS = TARGET_COLS

VARIANTS = [
    "full",
    "no_moe",
    "mean_pooling",
    "no_modality_embeddings",
    "no_head_shortcuts",
]


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(p)


def save_joblib(path: str | Path, value: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    joblib.dump(value, tmp)
    tmp.replace(p)


def save_torch(path: str | Path, value: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    torch.save(value, tmp)
    tmp.replace(p)


def sanitize_probabilities(p: np.ndarray | torch.Tensor) -> np.ndarray:
    arr = np.asarray(p, dtype=float)
    if not np.isfinite(arr).all() or np.any(arr < 0) or np.any(arr > 1):
        raise ValueError("Nonfinite or out-of-range probabilities.")
    return np.clip(arr, 1e-7, 1.0 - 1e-7)


def load_dataset_partitions():
    report = json.loads((DATA / "repair_report.json").read_text(encoding="utf-8"))
    if report["version"] != "stage1-foundation-v2":
        raise ValueError("Unexpected data version.")
    for name, expected_hash in report["output_hashes"].items():
        actual_hash = sha256_file(DATA / name)
        if actual_hash != expected_hash:
            raise ValueError(f"Frozen data hash mismatch for {name}: {actual_hash} != {expected_hash}")

    features = report["feature_order"]
    if len(features) != NUM_FEATURES or features != FEATURE_COLS:
        raise ValueError("Feature ordering mismatch with frozen repair report.")

    parts = {}
    for partition_name in ["training", "validation", "calibration", "temporal"]:
        df = pd.read_csv(DATA / f"{partition_name}_ml_ready.csv")
        expected_cols = ["SEQN", "NHANES_CYCLE"] + TARGETS + features
        if list(df.columns) != expected_cols:
            raise ValueError(f"{partition_name}: schema mismatch.")
        if df["SEQN"].isna().any() or len(df["SEQN"].unique()) != len(df):
            raise ValueError(f"{partition_name}: invalid participant IDs.")
        parts[partition_name] = {
            "x": df[features].to_numpy(dtype=np.float32),
            "y": df[TARGETS].to_numpy(dtype=np.float32),
            "ids": df["SEQN"].to_numpy(dtype=np.int64),
        }

    manifest = pd.read_csv(DATA / "split_membership_LOCAL.csv")
    for k, p in parts.items():
        manifest_seqns = set(manifest.loc[manifest["partition"] == k, "SEQN"])
        if set(p["ids"]) != manifest_seqns:
            raise ValueError(f"Partition {k} mismatch with split manifest.")

    return parts, features


# ---------------------------------------------------------------------------
# Architecture Ablation Modules
# ---------------------------------------------------------------------------


class FeatureTokenizerNoModality(nn.Module):
    """Tokenization without learned modality embedding vectors."""

    def __init__(self, num_features: int = NUM_FEATURES, d_model: int = 128) -> None:
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model
        self.weight = nn.Parameter(torch.empty(num_features, d_model))
        self.bias = nn.Parameter(torch.zeros(num_features, d_model))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.num_features:
            raise ValueError(f"Expected (B, {self.num_features}); got {tuple(x.shape)}.")
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class ModalityMeanPooler(nn.Module):
    """Ordinary unweighted mean pooling across tokens within each physiological modality."""

    def __init__(self, d_model: int = 128) -> None:
        super().__init__()
        self.d_model = d_model
        self.slices = {
            "bio": (0, 5),
            "diet": (5, 51),
            "life": (51, 70),
            "demo": (70, 87),
        }

    def forward(
        self, hidden: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if hidden.ndim != 3 or hidden.shape[1:] != (NUM_FEATURES, self.d_model):
            raise ValueError(f"Unexpected hidden shape: {tuple(hidden.shape)}.")
        pooled = {
            mod_name: hidden[:, start:end, :].mean(dim=1)
            for mod_name, (start, end) in self.slices.items()
        }
        return pooled, {}


class DensePatientFFN(nn.Module):
    """Single dense feed-forward network (512 -> 256 -> 256) replacing MoE."""

    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if u.ndim != 2 or u.shape[1] != 512:
            raise ValueError(f"DensePatientFFN expected (B, 512); got {tuple(u.shape)}.")
        # Returns patient_state (B, 256) and zero auxiliary balancing loss
        return self.net(u), u.new_zeros(())


# Model Variant 1: FULL
class BioNutriGPT_Full(nn.Module):
    """Full BioNutriGPT architecture."""

    def __init__(self, d_model: int = 128) -> None:
        super().__init__()
        self.variant = "full"
        self.feature_tokenizer = FeatureTokenizer(
            num_features=NUM_FEATURES, d_model=d_model, modality_sizes=MODALITY_SIZES
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=256, dropout=0.1, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.modality_pooler = ModalityAttentionPooler(d_model=d_model)
        self.moe = SparseMoELayer(input_dim=512, hidden_dim=256, num_experts=4, dropout=0.1)
        self.diabetes_head = nn.Sequential(
            nn.Linear(384, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1)
        )
        self.cvd_head = nn.Sequential(
            nn.Linear(384, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1)
        )
        self.obesity_head = nn.Sequential(
            nn.Linear(512, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.feature_tokenizer(x)
        hidden = self.transformer(tokens)
        mod_vecs, _ = self.modality_pooler(hidden)
        u = torch.cat([mod_vecs["bio"], mod_vecs["diet"], mod_vecs["life"], mod_vecs["demo"]], dim=-1)
        pat_state, aux = self.moe(u)
        clinical_bio = torch.cat([pat_state, mod_vecs["bio"]], dim=-1)
        obesity_context = torch.cat([pat_state, mod_vecs["diet"], mod_vecs["life"]], dim=-1)
        logits = torch.stack([
            self.diabetes_head(clinical_bio).squeeze(-1),
            self.obesity_head(obesity_context).squeeze(-1),
            self.cvd_head(clinical_bio).squeeze(-1),
        ], dim=1)
        return logits, aux


# Model Variant 2: NO_MOE
class BioNutriGPT_NoMoE(nn.Module):
    """BioNutriGPT with 4-expert MoE replaced by a single dense FFN."""

    def __init__(self, d_model: int = 128) -> None:
        super().__init__()
        self.variant = "no_moe"
        self.feature_tokenizer = FeatureTokenizer(
            num_features=NUM_FEATURES, d_model=d_model, modality_sizes=MODALITY_SIZES
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=256, dropout=0.1, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.modality_pooler = ModalityAttentionPooler(d_model=d_model)
        self.dense_ffn = DensePatientFFN(input_dim=512, hidden_dim=256, dropout=0.1)
        self.diabetes_head = nn.Sequential(
            nn.Linear(384, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1)
        )
        self.cvd_head = nn.Sequential(
            nn.Linear(384, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1)
        )
        self.obesity_head = nn.Sequential(
            nn.Linear(512, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.feature_tokenizer(x)
        hidden = self.transformer(tokens)
        mod_vecs, _ = self.modality_pooler(hidden)
        u = torch.cat([mod_vecs["bio"], mod_vecs["diet"], mod_vecs["life"], mod_vecs["demo"]], dim=-1)
        pat_state, aux = self.dense_ffn(u)
        clinical_bio = torch.cat([pat_state, mod_vecs["bio"]], dim=-1)
        obesity_context = torch.cat([pat_state, mod_vecs["diet"], mod_vecs["life"]], dim=-1)
        logits = torch.stack([
            self.diabetes_head(clinical_bio).squeeze(-1),
            self.obesity_head(obesity_context).squeeze(-1),
            self.cvd_head(clinical_bio).squeeze(-1),
        ], dim=1)
        return logits, aux


# Model Variant 3: MEAN_POOLING
class BioNutriGPT_MeanPooling(nn.Module):
    """BioNutriGPT with ordinary mean pooling replacing learned attention poolers."""

    def __init__(self, d_model: int = 128) -> None:
        super().__init__()
        self.variant = "mean_pooling"
        self.feature_tokenizer = FeatureTokenizer(
            num_features=NUM_FEATURES, d_model=d_model, modality_sizes=MODALITY_SIZES
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=256, dropout=0.1, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.modality_pooler = ModalityMeanPooler(d_model=d_model)
        self.moe = SparseMoELayer(input_dim=512, hidden_dim=256, num_experts=4, dropout=0.1)
        self.diabetes_head = nn.Sequential(
            nn.Linear(384, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1)
        )
        self.cvd_head = nn.Sequential(
            nn.Linear(384, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1)
        )
        self.obesity_head = nn.Sequential(
            nn.Linear(512, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.feature_tokenizer(x)
        hidden = self.transformer(tokens)
        mod_vecs, _ = self.modality_pooler(hidden)
        u = torch.cat([mod_vecs["bio"], mod_vecs["diet"], mod_vecs["life"], mod_vecs["demo"]], dim=-1)
        pat_state, aux = self.moe(u)
        clinical_bio = torch.cat([pat_state, mod_vecs["bio"]], dim=-1)
        obesity_context = torch.cat([pat_state, mod_vecs["diet"], mod_vecs["life"]], dim=-1)
        logits = torch.stack([
            self.diabetes_head(clinical_bio).squeeze(-1),
            self.obesity_head(obesity_context).squeeze(-1),
            self.cvd_head(clinical_bio).squeeze(-1),
        ], dim=1)
        return logits, aux


# Model Variant 4: NO_MODALITY_EMBEDDINGS
class BioNutriGPT_NoModalityEmbeddings(nn.Module):
    """BioNutriGPT without learned modality embedding vectors."""

    def __init__(self, d_model: int = 128) -> None:
        super().__init__()
        self.variant = "no_modality_embeddings"
        self.feature_tokenizer = FeatureTokenizerNoModality(
            num_features=NUM_FEATURES, d_model=d_model
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=256, dropout=0.1, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.modality_pooler = ModalityAttentionPooler(d_model=d_model)
        self.moe = SparseMoELayer(input_dim=512, hidden_dim=256, num_experts=4, dropout=0.1)
        self.diabetes_head = nn.Sequential(
            nn.Linear(384, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1)
        )
        self.cvd_head = nn.Sequential(
            nn.Linear(384, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1)
        )
        self.obesity_head = nn.Sequential(
            nn.Linear(512, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.feature_tokenizer(x)
        hidden = self.transformer(tokens)
        mod_vecs, _ = self.modality_pooler(hidden)
        u = torch.cat([mod_vecs["bio"], mod_vecs["diet"], mod_vecs["life"], mod_vecs["demo"]], dim=-1)
        pat_state, aux = self.moe(u)
        clinical_bio = torch.cat([pat_state, mod_vecs["bio"]], dim=-1)
        obesity_context = torch.cat([pat_state, mod_vecs["diet"], mod_vecs["life"]], dim=-1)
        logits = torch.stack([
            self.diabetes_head(clinical_bio).squeeze(-1),
            self.obesity_head(obesity_context).squeeze(-1),
            self.cvd_head(clinical_bio).squeeze(-1),
        ], dim=1)
        return logits, aux


# Model Variant 5: NO_HEAD_SHORTCUTS
class BioNutriGPT_NoHeadShortcuts(nn.Module):
    """BioNutriGPT with heads receiving only the 256-d patient state representation."""

    def __init__(self, d_model: int = 128) -> None:
        super().__init__()
        self.variant = "no_head_shortcuts"
        self.feature_tokenizer = FeatureTokenizer(
            num_features=NUM_FEATURES, d_model=d_model, modality_sizes=MODALITY_SIZES
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=256, dropout=0.1, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.modality_pooler = ModalityAttentionPooler(d_model=d_model)
        self.moe = SparseMoELayer(input_dim=512, hidden_dim=256, num_experts=4, dropout=0.1)
        # All heads receive ONLY patient_state in R^256 via Linear(256, 64)
        self.diabetes_head = nn.Sequential(
            nn.Linear(256, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1)
        )
        self.cvd_head = nn.Sequential(
            nn.Linear(256, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1)
        )
        self.obesity_head = nn.Sequential(
            nn.Linear(256, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.feature_tokenizer(x)
        hidden = self.transformer(tokens)
        mod_vecs, _ = self.modality_pooler(hidden)
        u = torch.cat([mod_vecs["bio"], mod_vecs["diet"], mod_vecs["life"], mod_vecs["demo"]], dim=-1)
        pat_state, aux = self.moe(u)
        logits = torch.stack([
            self.diabetes_head(pat_state).squeeze(-1),
            self.obesity_head(pat_state).squeeze(-1),
            self.cvd_head(pat_state).squeeze(-1),
        ], dim=1)
        return logits, aux


def build_ablation_model(variant_name: str, seed: int) -> nn.Module:
    """Build an ablation model and initialize shared modules with modular seed mapping."""
    if variant_name == "full":
        model = BioNutriGPT_Full()
    elif variant_name == "no_moe":
        model = BioNutriGPT_NoMoE()
    elif variant_name == "mean_pooling":
        model = BioNutriGPT_MeanPooling()
    elif variant_name == "no_modality_embeddings":
        model = BioNutriGPT_NoModalityEmbeddings()
    elif variant_name == "no_head_shortcuts":
        model = BioNutriGPT_NoHeadShortcuts()
    else:
        raise ValueError(f"Unknown variant: {variant_name}")

    # Submodule-specific deterministic seeding to give shared modules identical initial parameters
    # Tokenizer weight and bias
    torch.manual_seed(seed * 1000 + 1)
    nn.init.xavier_uniform_(model.feature_tokenizer.weight)
    nn.init.zeros_(model.feature_tokenizer.bias)

    # Modality embeddings (if present)
    if hasattr(model.feature_tokenizer, "modality_embedding"):
        torch.manual_seed(seed * 1000 + 2)
        nn.init.normal_(model.feature_tokenizer.modality_embedding, mean=0.0, std=0.02)

    # Transformer layers
    torch.manual_seed(seed * 1000 + 3)
    for p in model.transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.zeros_(p)

    # Modality pooler attention vectors (if present)
    if hasattr(model.modality_pooler, "attention_vectors"):
        torch.manual_seed(seed * 1000 + 4)
        for vector in model.modality_pooler.attention_vectors.values():
            nn.init.normal_(vector, mean=0.0, std=1.0 / math.sqrt(128))

    # Patient aggregation: MoE or Dense FFN
    torch.manual_seed(seed * 1000 + 5)
    if hasattr(model, "moe"):
        for expert in model.moe.experts:
            for p in expert.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
                else:
                    nn.init.zeros_(p)
        nn.init.xavier_uniform_(model.moe.gating_network.weight)
        nn.init.zeros_(model.moe.gating_network.bias)
    elif hasattr(model, "dense_ffn"):
        for p in model.dense_ffn.net.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)

    # Prediction heads
    torch.manual_seed(seed * 1000 + 6)
    for head in [model.diabetes_head, model.cvd_head, model.obesity_head]:
        for p in head.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)

    return model


# ---------------------------------------------------------------------------
# Training, Calibration & Evaluation Engine
# ---------------------------------------------------------------------------


def selection_score(y: np.ndarray, p: np.ndarray, indices: tuple[int, int] = (0, 2)) -> float:
    p = sanitize_probabilities(p)
    return float(np.mean([roc_auc_score(y[:, i], p[:, i]) for i in indices]))


def predict_model(model: nn.Module, x: np.ndarray, device: str, batch_size: int = 128) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            batch_x = torch.as_tensor(x[start : start + batch_size], device=device)
            logits, _ = model(batch_x)
            outputs.append(torch.sigmoid(logits).cpu().numpy())
    return sanitize_probabilities(np.concatenate(outputs, axis=0))


def fit_calibrators(y: np.ndarray, p: np.ndarray) -> list[LogisticRegression]:
    """Fit unpenalized logistic calibrator per outcome using calibration set only."""
    p = sanitize_probabilities(p)
    calibrators = []
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        for i in range(3):
            c = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=2000, tol=1e-10)
            c.fit(logit(p[:, i]).reshape(-1, 1), y[:, i])
            calibrators.append(c)
    return calibrators


def apply_calibrators(calibrators: list[LogisticRegression], p: np.ndarray) -> np.ndarray:
    p = sanitize_probabilities(p)
    return np.column_stack([
        c.predict_proba(logit(p[:, i]).reshape(-1, 1))[:, 1]
        for i, c in enumerate(calibrators)
    ])


def compute_evaluation_metrics(y: np.ndarray, p: np.ndarray) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    p = sanitize_probabilities(p)
    metric_rows = []
    curves = []
    for i, target in enumerate(TARGETS):
        z = logit(p[:, i]).reshape(-1, 1)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            diagnostic = LogisticRegression(C=np.inf, max_iter=2000, tol=1e-10).fit(z, y[:, i])

        tn, fp, fn, tp = confusion_matrix(y[:, i], p[:, i] >= 0.5, labels=[0, 1]).ravel()
        div = lambda a, b: float(a / b) if b else None

        metric_rows.append({
            "outcome": target,
            "n": len(y),
            "positive": int(y[:, i].sum()),
            "auroc": float(roc_auc_score(y[:, i], p[:, i])),
            "average_precision": float(average_precision_score(y[:, i], p[:, i])),
            "brier": float(brier_score_loss(y[:, i], p[:, i])),
            "log_loss": float(log_loss(y[:, i], p[:, i])),
            "calibration_intercept": float(diagnostic.intercept_[0]),
            "calibration_slope": float(diagnostic.coef_[0, 0]),
            "calibration_diagnostic_convergence_warning": any(
                issubclass(w.category, ConvergenceWarning) for w in caught
            ),
            "threshold": 0.5,
            "threshold_scope": "fixed descriptive threshold; not clinically validated",
            "sensitivity": div(tp, tp + fn),
            "specificity": div(tn, tn + fp),
            "ppv": div(tp, tp + fp),
            "npv": div(tn, tn + fn),
        })

        bins = np.minimum((p[:, i] * 10).astype(int), 9)
        for b in range(10):
            mask = bins == b
            curves.append({
                "outcome": target,
                "bin": b,
                "n": int(mask.sum()),
                "mean_probability": float(p[mask, i].mean()) if mask.any() else None,
                "observed_fraction": float(y[mask, i].mean()) if mask.any() else None,
            })
    return metric_rows, curves


@contextmanager
def memory_monitor():
    process = psutil.Process()
    stop = threading.Event()
    stats = {"peak_process_rss_bytes": process.memory_info().rss}

    def poll():
        while not stop.wait(0.1):
            stats["peak_process_rss_bytes"] = max(
                stats["peak_process_rss_bytes"], process.memory_info().rss
            )

    thread = threading.Thread(target=poll, daemon=True)
    thread.start()
    try:
        yield stats
    finally:
        stop.set()
        thread.join()


def measure_latency(model: nn.Module, x: np.ndarray, device: str) -> list[dict[str, Any]]:
    rows = []
    for batch_size in [1, 128]:
        sample = x[:batch_size]
        for _ in range(3):
            predict_model(model, sample, device, batch_size=batch_size)
        timings = []
        for _ in range(20):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            predict_model(model, sample, device, batch_size=batch_size)
            if device == "cuda":
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - t0) * 1000.0)
        rows.append({
            "batch_size": len(sample),
            "median_ms": float(np.median(timings)),
            "p95_ms": float(np.percentile(timings, 95)),
            "inference_device": device,
            "repeats": 20,
        })
    return rows


def train_variant_seed(
    variant: str,
    seed: int,
    data: dict[str, Any],
    protocol: dict[str, Any],
    folder: Path,
    device: str,
) -> dict[str, Any]:
    """Train one ablation model seed, evaluate selection, calibrate, and report metrics."""
    folder.mkdir(parents=True, exist_ok=True)
    completed_path = folder / "completed.json"
    if completed_path.exists():
        record = json.loads(completed_path.read_text(encoding="utf-8"))
        all_ok = True
        for fname, expected_h in record.get("artifact_hashes", {}).items():
            if not (folder / fname).exists() or sha256_file(folder / fname) != expected_h:
                all_ok = False
                break
        if all_ok:
            print(f"[{variant} seed {seed}] already complete and verified.", flush=True)
            return record

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()

    train_settings = protocol["training"]
    var_meta = protocol["variants"][variant]
    aux_coef = float(var_meta["moe_aux_coefficient"])

    start_time = time.perf_counter()
    epoch_logs = []

    with memory_monitor() as memory_stats:
        model = build_ablation_model(variant, seed).to(device)
        param_count = sum(p.numel() for p in model.parameters())
        if param_count != var_meta["parameter_count"]:
            raise ValueError(
                f"Parameter count mismatch for {variant}: got {param_count}, expected {var_meta['parameter_count']}."
            )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_settings["lr"],
            weight_decay=train_settings["weight_decay"],
        )

        train_x = torch.as_tensor(data["training"]["x"], device=device)
        train_y = torch.as_tensor(data["training"]["y"], device=device)

        best_score = -np.inf
        best_epoch = 0
        stale_epochs = 0
        batch_size = train_settings["batch_size"]

        for epoch in range(train_settings["max_epochs"]):
            model.train()
            # Deterministic, seeded row shuffle per epoch and seed
            perm_generator = torch.Generator(device=device).manual_seed(seed * 10000 + epoch)
            order = torch.randperm(len(train_x), generator=perm_generator, device=device)

            total_epoch_loss = 0.0
            for start_idx in range(0, len(train_x), batch_size):
                idx = order[start_idx : start_idx + batch_size]
                optimizer.zero_grad(set_to_none=True)
                logits, aux_loss = model(train_x[idx])
                bce = F.binary_cross_entropy_with_logits(logits, train_y[idx])
                total_loss = bce + aux_coef * aux_loss
                if not torch.isfinite(total_loss):
                    raise ValueError(f"Non-finite loss at epoch {epoch+1}.")
                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), train_settings["gradient_clip"])
                optimizer.step()
                total_epoch_loss += float(total_loss.detach()) * len(idx)

            val_preds = predict_model(model, data["validation"]["x"], device)
            val_score = selection_score(data["validation"]["y"], val_preds)

            epoch_record = {
                "epoch": epoch + 1,
                "training_loss": total_epoch_loss / len(train_x),
                "validation_selection_auroc": val_score,
            }
            epoch_logs.append(epoch_record)
            with (folder / "epochs.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(epoch_record) + "\n")

            if val_score > best_score:
                best_score = val_score
                best_epoch = epoch + 1
                stale_epochs = 0
                save_torch(folder / "model.pt", {k: v.detach().cpu() for k, v in model.state_dict().items()})
            else:
                stale_epochs += 1

            if stale_epochs >= train_settings["patience"]:
                break

        # Load best checkpoint
        model.load_state_dict(torch.load(folder / "model.pt", map_location=device, weights_only=True))
        del train_x, train_y, optimizer
        if device == "cuda":
            torch.cuda.empty_cache()

        # Calibration step (fitted strictly on reserved calibration set)
        cal_preds_raw = predict_model(model, data["calibration"]["x"], device)
        calibrators = fit_calibrators(data["calibration"]["y"], cal_preds_raw)
        save_joblib(folder / "calibrators.joblib", calibrators)

        cal_params = []
        for i, target in enumerate(TARGETS):
            cal_params.append({
                "outcome": target,
                "intercept": float(calibrators[i].intercept_[0]),
                "logit_slope": float(calibrators[i].coef_[0, 0]),
            })

        # Temporal evaluation
        temp_preds_raw = predict_model(model, data["temporal"]["x"], device)
        temp_preds_cal = apply_calibrators(calibrators, temp_preds_raw)

        # Export predictions with SEQN IDs
        val_preds_best = predict_model(model, data["validation"]["x"], device)
        val_preds_cal = apply_calibrators(calibrators, val_preds_best)
        cal_preds_cal = apply_calibrators(calibrators, cal_preds_raw)

        pred_files = {}
        for part_name, r_p, c_p in [
            ("validation", val_preds_best, val_preds_cal),
            ("calibration", cal_preds_raw, cal_preds_cal),
            ("temporal", temp_preds_raw, temp_preds_cal),
        ]:
            df = pd.DataFrame({"SEQN": data[part_name]["ids"]})
            for i, target in enumerate(TARGETS):
                df[target] = data[part_name]["y"][:, i].astype(int)
                df[f"{target}_raw"] = r_p[:, i]
                df[f"{target}_calibrated"] = c_p[:, i]
            fname = f"{part_name}_predictions_LOCAL.csv"
            df.to_csv(folder / fname, index=False)
            pred_files[fname] = sha256_file(folder / fname)

        # Compute temporal metrics
        raw_metrics, raw_curves = compute_evaluation_metrics(data["temporal"]["y"], temp_preds_raw)
        cal_metrics, cal_curves = compute_evaluation_metrics(data["temporal"]["y"], temp_preds_cal)

        all_metrics = [
            {"variant_name": variant, "seed": seed, "prediction_variant": "raw", **m}
            for m in raw_metrics
        ] + [
            {"variant_name": variant, "seed": seed, "prediction_variant": "calibrated", **m}
            for m in cal_metrics
        ]
        pd.DataFrame(all_metrics).to_csv(folder / "temporal_metrics.csv", index=False)

        all_curves = [
            {"variant_name": variant, "seed": seed, "prediction_variant": "raw", **c}
            for c in raw_curves
        ] + [
            {"variant_name": variant, "seed": seed, "prediction_variant": "calibrated", **c}
            for c in cal_curves
        ]
        pd.DataFrame(all_curves).to_csv(folder / "calibration_curve.csv", index=False)

        latency_rows = measure_latency(model, data["temporal"]["x"], device)

    elapsed_sec = time.perf_counter() - start_time
    cuda_peak = int(torch.cuda.max_memory_allocated()) if device == "cuda" else None

    completed_record = {
        "variant": variant,
        "seed": seed,
        "parameter_count": param_count,
        "best_epoch": best_epoch,
        "best_validation_selection_auroc": best_score,
        "total_epochs_trained": len(epoch_logs),
        "training_seconds": elapsed_sec,
        "cuda_peak_allocated_bytes": cuda_peak,
        "cuda_allocator_scope": "PyTorch allocator only",
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "calibration": cal_params,
        "latency": latency_rows,
        "metrics": all_metrics,
        "artifact_hashes": {
            "model.pt": sha256_file(folder / "model.pt"),
            "calibrators.joblib": sha256_file(folder / "calibrators.joblib"),
            "temporal_metrics.csv": sha256_file(folder / "temporal_metrics.csv"),
            "calibration_curve.csv": sha256_file(folder / "calibration_curve.csv"),
            **pred_files,
        },
    }
    write_json(completed_path, completed_record)
    return completed_record


def run_variant(variant: str, target_seed: int | None = None) -> list[dict[str, Any]]:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Allowed: {VARIANTS}")

    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    data, features = load_dataset_partitions()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("RTX 3050 CUDA GPU required for Stage 1 ablation neural training.")

    seeds = [target_seed] if target_seed is not None else protocol["seeds"]
    results = []
    print(f"\n=======================================================", flush=True)
    print(f"Running Stage 1 Ablation Variant: {variant.upper()}", flush=True)
    print(f"Seeds: {seeds} on {device} ({torch.cuda.get_device_name(0)})", flush=True)
    print(f"=======================================================\n", flush=True)

    for s in seeds:
        seed_folder = RESULTS_DIR / variant / f"seed_{s}"
        print(f"--- Variant: {variant.upper()} | Seed: {s} ---", flush=True)
        rec = train_variant_seed(variant, s, data, protocol, seed_folder, device)
        # Print key primary metric summary for user transparency
        t_raw = {m["outcome"]: m for m in rec["metrics"] if m["prediction_variant"] == "raw"}
        t_cal = {m["outcome"]: m for m in rec["metrics"] if m["prediction_variant"] == "calibrated"}
        print(
            f"  Done in {rec['training_seconds']:.1f}s (epoch {rec['best_epoch']}/{rec['total_epochs_trained']}). "
            f"Diabetes raw AUROC: {t_raw['TARGET_DIABETES']['auroc']:.4f}, "
            f"CVD raw AUROC: {t_raw['TARGET_CVD']['auroc']:.4f}, "
            f"Diabetes cal Brier: {t_cal['TARGET_DIABETES']['brier']:.6f}, "
            f"CVD cal Brier: {t_cal['TARGET_CVD']['brier']:.6f}",
            flush=True,
        )
        results.append(rec)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 1 Architecture Ablations v2")
    parser.add_argument("--variant", choices=VARIANTS, help="Variant to train across seeds")
    parser.add_argument("--seed", type=int, help="Optional specific seed to run")
    parser.add_argument("--preflight", action="store_true", help="Perform preflight checks only")
    args = parser.parse_args()

    if args.preflight or args.variant is None:
        print("Checking ablation protocol, architecture modules, and dataset partitions...", flush=True)
        protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        data, features = load_dataset_partitions()
        print(f"Dataset partitions loaded cleanly: {len(data['training']['x'])} train, "
              f"{len(data['validation']['x'])} val, {len(data['calibration']['x'])} cal, "
              f"{len(data['temporal']['x'])} temp.")
        for v in VARIANTS:
            m = build_ablation_model(v, 42)
            n_params = sum(p.numel() for p in m.parameters())
            expected = protocol["variants"][v]["parameter_count"]
            assert n_params == expected, f"{v} parameter count mismatch: {n_params} != {expected}"
            print(f"  Variant {v:<25}: {n_params:>10,} params (matches protocol)")
        print("Preflight checks PASSED cleanly.", flush=True)
        return

    run_variant(args.variant, target_seed=args.seed)


if __name__ == "__main__":
    main()
