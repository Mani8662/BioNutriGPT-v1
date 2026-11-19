#!/usr/bin/env python3
"""Train and prospectively evaluate BioNutriGPT Stage 1.

Expected project layout (relative to this script's project root)::

    Models/
      Less Data/Data/
        BioNutriGPT_train_2007_2016_ml_ready.csv
        BioNutriGPT_test_2017_2020_ml_ready.csv
        preprocessing_artifacts.joblib   # provenance only; not reapplied

Run from any working directory with::

    python train_bionutrigpt_stage1.py

The CSVs are expected to be fully preprocessed. This script deliberately does
not fit or apply another scaler/imputer, and it never feeds patient IDs,
survey-cycle metadata, BMI, weight, or height to the neural network.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# 1. Imports & global seed setting
# ---------------------------------------------------------------------------

SEED = 42
BATCH_SIZE = 128
EPOCHS = 40
EARLY_STOPPING_PATIENCE = 7
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
VALIDATION_SIZE = 3_554

TARGET_COLS = ["TARGET_DIABETES", "TARGET_OBESITY", "TARGET_CVD"]
METADATA_COLS = ["SEQN", "NHANES_CYCLE"]

BIO_COLS = [
    "LBDHDD",
    "LBXTC",
    "NON_HDL_CHOL",
    "CHOL_HDL_RATIO",
    "NON_HDL_TO_HDL_RATIO",
]

DIET_COLS = [
    "DR1TKCAL",
    "DR1TPROT",
    "DR1TCARB",
    "DR1TSUGR",
    "DR1TFIBE",
    "DR1TTFAT",
    "DR1TSFAT",
    "DR1TMFAT",
    "DR1TPFAT",
    "DR1TCHOL",
    "DR1TATOC",
    "DR1TRET",
    "DR1TVARA",
    "DR1TACAR",
    "DR1TBCAR",
    "DR1TLYCO",
    "DR1TLZ",
    "DR1TVB1",
    "DR1TVB2",
    "DR1TNIAC",
    "DR1TVB6",
    "DR1TFOLA",
    "DR1TFA",
    "DR1TFF",
    "DR1TFDFE",
    "DR1TVB12",
    "DR1TVC",
    "DR1TVK",
    "DR1TCALC",
    "DR1TPHOS",
    "DR1TMAGN",
    "DR1TIRON",
    "DR1TZINC",
    "DR1TCOPP",
    "DR1TSODI",
    "DR1TPOTA",
    "DR1TSELE",
    "DR1TCAFF",
    "DR1TALCO",
    "PROT_PCT_KCAL",
    "CARB_PCT_KCAL",
    "FAT_PCT_KCAL",
    "FIBER_PER_1000_KCAL",
    "SUGAR_TO_FIBER_RATIO",
    "SODIUM_POTASSIUM_RATIO",
    "SAT_TO_UNSAT_FAT_RATIO",
]

LIFE_COLS = [
    "SLEEP_HOURS",
    "SLEEP_DEV_FROM_7_5",
    "PAD680",
    "PA_VIG_WORK_MIN_WK",
    "PA_MOD_WORK_MIN_WK",
    "PA_WALK_BIKE_MIN_WK",
    "PA_VIG_REC_MIN_WK",
    "PA_MOD_REC_MIN_WK",
    "PA_TOTAL_MET_MIN_WK",
    "SLQ050",
    "PAQ605",
    "PAQ620",
    "PAQ635",
    "PAQ650",
    "PAQ665",
    "SMOKE_NEVER",
    "SMOKE_FORMER",
    "SMOKE_SOMEDAYS",
    "SMOKE_EVERYDAY",
]

DEMO_COLS = [
    "RIDAGEYR",
    "INDFMPIR",
    "DMDEDUC2",
    "SEX_FEMALE",
    "RACE_MEX_AMER",
    "RACE_OTHER_HISP",
    "RACE_NH_WHITE",
    "RACE_NH_BLACK",
    "RACE_OTHER_MULTI",
    "MARITAL_MARRIED",
    "MARITAL_PREV_MARRIED",
    "MARITAL_NEVER_MARRIED",
    "INDFMPIR_MISS",
    "SLEEP_HOURS_MISS",
    "PAD680_MISS",
    "PA_TOTAL_MET_MIN_WK_MISS",
    "OTHER_FEATURE_MISS",
]

FEATURE_COLS = BIO_COLS + DIET_COLS + LIFE_COLS + DEMO_COLS
NUM_FEATURES = 87
NUM_TARGETS = 3
MODALITY_SIZES = {
    "bio": len(BIO_COLS),
    "diet": len(DIET_COLS),
    "life": len(LIFE_COLS),
    "demo": len(DEMO_COLS),
}

EXPECTED_TRAIN_ROWS = 23_694
EXPECTED_TEST_ROWS = 7_046
EXPECTED_COLUMN_COUNT = 92
EXPECTED_TRAIN_CLASS_COUNTS = {
    "TARGET_DIABETES": (3_022, 20_672),
    "TARGET_CVD": (2_205, 21_489),
    "TARGET_OBESITY": (8_932, 14_762),
}

# The target tensor is ordered exactly as TARGET_COLS: diabetes, obesity, CVD.
TARGET_INDEX = {
    "diabetes": TARGET_COLS.index("TARGET_DIABETES"),
    "obesity": TARGET_COLS.index("TARGET_OBESITY"),
    "cvd": TARGET_COLS.index("TARGET_CVD"),
}

# Assert the specified partition is internally consistent before any data load.
assert len(BIO_COLS) == 5
assert len(DIET_COLS) == 46
assert len(LIFE_COLS) == 19
assert len(DEMO_COLS) == 17
assert len(FEATURE_COLS) == NUM_FEATURES
assert len(set(FEATURE_COLS)) == NUM_FEATURES
assert len(FEATURE_COLS) == 87
assert "LBXGH" not in FEATURE_COLS
assert len(METADATA_COLS) + len(TARGET_COLS) + len(FEATURE_COLS) == EXPECTED_COLUMN_COUNT

FORBIDDEN_MODEL_INPUTS = {
    "BMXBMI",
    "BMXWT",
    "BMXHT",
    "SEQN",
    "NHANES_CYCLE",
    *TARGET_COLS,
}
assert not (set(FEATURE_COLS) & FORBIDDEN_MODEL_INPUTS), (
    "Leakage/metadata/target columns must not be model inputs."
)


def set_global_seed(seed: int = SEED) -> None:
    """Set reproducible seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# 2. Dataset class
# ---------------------------------------------------------------------------


class NHANESDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """NHANES feature/label dataset.

    Each item is (features, targets), with shapes (87,) and (3,). Targets are
    ordered as diabetes, obesity, and CVD. Metadata is validated separately
    and is intentionally not returned to the model.
    """

    def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
        if features.ndim != 2 or features.shape[1] != NUM_FEATURES:
            raise ValueError(
                f"Expected features with shape (N, {NUM_FEATURES}); got {features.shape}."
            )
        if targets.ndim != 2 or targets.shape[1] != NUM_TARGETS:
            raise ValueError(
                f"Expected targets with shape (N, {NUM_TARGETS}); got {targets.shape}."
            )
        if features.shape[0] != targets.shape[0]:
            raise ValueError("Feature and target row counts do not match.")

        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.targets[index]


def validate_and_extract_arrays(
    dataframe: pd.DataFrame,
    dataset_name: str,
    expected_rows: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate the exact schema and return leakage-safe X and ordered y."""
    if dataframe.empty:
        raise ValueError(f"{dataset_name} is empty.")

    if expected_rows is not None and len(dataframe) != expected_rows:
        raise ValueError(
            f"{dataset_name} should contain {expected_rows:,} rows; "
            f"found {len(dataframe):,}. Check that the intended cohort file was loaded."
        )

    expected_columns = set(METADATA_COLS + TARGET_COLS + FEATURE_COLS)
    actual_columns = set(dataframe.columns)
    missing = sorted(expected_columns - actual_columns)
    unexpected = sorted(actual_columns - expected_columns)
    if missing or unexpected or len(dataframe.columns) != EXPECTED_COLUMN_COUNT:
        details = []
        if missing:
            details.append(f"missing columns: {missing}")
        if unexpected:
            details.append(f"unexpected columns: {unexpected}")
        details.append(
            f"expected exactly {EXPECTED_COLUMN_COUNT} columns, found {len(dataframe.columns)}"
        )
        raise ValueError(f"Invalid schema for {dataset_name}: " + "; ".join(details))

    # Explicitly ensure these fields cannot enter the model feature matrix.
    if set(FEATURE_COLS) & FORBIDDEN_MODEL_INPUTS:
        raise AssertionError("A forbidden metadata, target, or anthropometric field is an input.")
    for forbidden_anthropometric in ("BMXBMI", "BMXWT", "BMXHT"):
        if forbidden_anthropometric in FEATURE_COLS:
            raise AssertionError(f"Anthropometric leakage: {forbidden_anthropometric}")

    try:
        features = dataframe.loc[:, FEATURE_COLS].to_numpy(dtype=np.float32, copy=True)
        targets = dataframe.loc[:, TARGET_COLS].to_numpy(dtype=np.float32, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Non-numeric input or target column in {dataset_name}: {exc}") from exc

    if not np.isfinite(features).all():
        bad = np.argwhere(~np.isfinite(features))[0]
        raise ValueError(
            f"{dataset_name} contains NaN/Inf in feature row {int(bad[0])}, "
            f"column {FEATURE_COLS[int(bad[1])]}. The input CSV must already be cleaned."
        )
    if not np.isfinite(targets).all():
        raise ValueError(f"{dataset_name} contains NaN/Inf target values.")

    unique_targets = set(np.unique(targets).tolist())
    if not unique_targets.issubset({0.0, 1.0}):
        raise ValueError(
            f"{dataset_name} targets must be binary 0/1 values; found {sorted(unique_targets)}."
        )

    # Metadata is required for dataset auditing, but never copied into X.
    if dataframe.loc[:, METADATA_COLS].isna().any().any():
        raise ValueError(f"{dataset_name} contains missing metadata values.")

    return features, targets


def validate_historical_training_distribution(dataframe: pd.DataFrame) -> None:
    """Fail fast if the fixed positive weights do not match this cohort."""
    for target_name, (expected_positive, expected_negative) in EXPECTED_TRAIN_CLASS_COUNTS.items():
        positive = int((dataframe[target_name] == 1).sum())
        negative = int((dataframe[target_name] == 0).sum())
        if (positive, negative) != (expected_positive, expected_negative):
            raise ValueError(
                f"Unexpected historical distribution for {target_name}: "
                f"expected {expected_positive:,} positives/{expected_negative:,} negatives, "
                f"found {positive:,}/{negative:,}. The specified pos_weight may no longer apply."
            )


# ---------------------------------------------------------------------------
# 3. Architecture modules
# ---------------------------------------------------------------------------


class FeatureTokenizer(nn.Module):
    """Feature-wise linear tokenizer with a learnable modality embedding."""

    def __init__(
        self,
        num_features: int = NUM_FEATURES,
        d_model: int = 128,
        modality_sizes: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        if modality_sizes is None:
            modality_sizes = MODALITY_SIZES
        if sum(modality_sizes.values()) != num_features:
            raise ValueError("Modality sizes must sum to the number of input features.")

        self.num_features = num_features
        self.d_model = d_model
        self.weight = nn.Parameter(torch.empty(num_features, d_model))
        self.bias = nn.Parameter(torch.zeros(num_features, d_model))
        self.modality_embedding = nn.Parameter(torch.empty(len(modality_sizes), d_model))

        modality_ids: list[int] = []
        for modality_id, modality_size in enumerate(modality_sizes.values()):
            modality_ids.extend([modality_id] * modality_size)
        self.register_buffer(
            "feature_to_modality",
            torch.tensor(modality_ids, dtype=torch.long),
            persistent=True,
        )

        nn.init.xavier_uniform_(self.weight)
        nn.init.normal_(self.modality_embedding, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.num_features:
            raise ValueError(
                f"FeatureTokenizer expected (B, {self.num_features}); got {tuple(x.shape)}."
            )
        tokens = x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        modality_vectors = self.modality_embedding[self.feature_to_modality]
        return tokens + modality_vectors.unsqueeze(0)


class ModalityAttentionPooler(nn.Module):
    """Learn one attention pooling vector for each physiological modality."""

    def __init__(self, d_model: int = 128) -> None:
        super().__init__()
        self.d_model = d_model
        self.modality_order = list(MODALITY_SIZES.keys())
        self.modality_sizes = dict(MODALITY_SIZES)
        self.attention_vectors = nn.ParameterDict(
            {
                name: nn.Parameter(torch.empty(d_model))
                for name in self.modality_order
            }
        )
        for vector in self.attention_vectors.values():
            nn.init.normal_(vector, mean=0.0, std=1.0 / math.sqrt(d_model))

    def forward(
        self, hidden: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        if hidden.ndim != 3 or hidden.shape[1:] != (NUM_FEATURES, self.d_model):
            raise ValueError(
                "ModalityAttentionPooler received an unexpected hidden tensor shape: "
                f"{tuple(hidden.shape)}."
            )

        pooled: dict[str, torch.Tensor] = {}
        attention_weights: dict[str, torch.Tensor] = {}
        start = 0
        scale = math.sqrt(self.d_model)
        for modality_name in self.modality_order:
            modality_size = self.modality_sizes[modality_name]
            end = start + modality_size
            modality_hidden = hidden[:, start:end, :]
            scores = torch.einsum(
                "bkd,d->bk", modality_hidden, self.attention_vectors[modality_name]
            ) / scale
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)
            pooled[modality_name] = torch.sum(weights * modality_hidden, dim=1)
            attention_weights[modality_name] = weights
            start = end

        return pooled, attention_weights


class SparseMoELayer(nn.Module):
    """Four-expert sparse mixture with top-2 routing and load-balancing loss."""

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_experts: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_experts != 4:
            raise ValueError("BioNutriGPT Stage 1 specifies exactly four experts.")
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(num_experts)
            ]
        )
        self.gating_network = nn.Linear(input_dim, num_experts)

    def forward(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if u.ndim != 2 or u.shape[1] != 512:
            raise ValueError(f"SparseMoELayer expected (B, 512); got {tuple(u.shape)}.")

        routing_logits = self.gating_network(u)  # (B, 4)
        top2_logits, top2_indices = torch.topk(routing_logits, k=2, dim=-1)
        top2_weights = torch.softmax(top2_logits, dim=-1)  # (B, 2)

        # Evaluate all experts, then gather only the two routed outputs per row.
        all_expert_outputs = torch.stack([expert(u) for expert in self.experts], dim=1)
        gather_indices = top2_indices.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
        selected_expert_outputs = torch.gather(all_expert_outputs, dim=1, index=gather_indices)
        patient_state = torch.sum(
            selected_expert_outputs * top2_weights.unsqueeze(-1), dim=1
        )

        importance = torch.softmax(routing_logits, dim=-1).mean(dim=0)
        top2_mask = F.one_hot(top2_indices, num_classes=self.num_experts).sum(dim=1)
        load = top2_mask.to(dtype=u.dtype).mean(dim=0)
        loss_moe_balance = 4.0 * torch.sum(importance * load)

        return patient_state, loss_moe_balance


class BioNutriGPTStage1(nn.Module):
    """Multimodal transformer, attention pooler, sparse MoE, and clinical heads."""

    def __init__(self) -> None:
        super().__init__()
        d_model = 128
        self.feature_tokenizer = FeatureTokenizer(
            num_features=NUM_FEATURES,
            d_model=d_model,
            modality_sizes=MODALITY_SIZES,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.modality_pooler = ModalityAttentionPooler(d_model=d_model)
        self.moe = SparseMoELayer(
            input_dim=512,
            hidden_dim=256,
            num_experts=4,
            dropout=0.1,
        )

        self.diabetes_head = nn.Sequential(
            nn.Linear(384, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )
        self.cvd_head = nn.Sequential(
            nn.Linear(384, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )
        self.obesity_head = nn.Sequential(
            nn.Linear(512, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
        tokens = self.feature_tokenizer(x)  # (B, 87, 128)
        hidden = self.transformer(tokens)  # (B, 87, 128)
        modality_vectors, attention_weights = self.modality_pooler(hidden)

        u = torch.cat(
            [
                modality_vectors["bio"],
                modality_vectors["diet"],
                modality_vectors["life"],
                modality_vectors["demo"],
            ],
            dim=-1,
        )  # (B, 512)
        patient_state, loss_moe_balance = self.moe(u)  # (B, 256)

        clinical_bio = torch.cat([patient_state, modality_vectors["bio"]], dim=-1)
        obesity_context = torch.cat(
            [
                patient_state,
                modality_vectors["diet"],
                modality_vectors["life"],
            ],
            dim=-1,
        )

        logits = {
            "diabetes": self.diabetes_head(clinical_bio).squeeze(-1),
            "obesity": self.obesity_head(obesity_context).squeeze(-1),
            "cvd": self.cvd_head(clinical_bio).squeeze(-1),
        }
        return logits, loss_moe_balance, attention_weights


# ---------------------------------------------------------------------------
# 4. Training engine
# ---------------------------------------------------------------------------


def build_criteria(device: torch.device) -> dict[str, nn.BCEWithLogitsLoss]:
    """Construct the specified hardcoded positive-weighted BCE objectives."""
    return {
        "diabetes": nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([6.84], dtype=torch.float32, device=device)
        ),
        "cvd": nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([9.74], dtype=torch.float32, device=device)
        ),
        "obesity": nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([1.65], dtype=torch.float32, device=device)
        ),
    }


def compute_multitask_loss(
    logits: dict[str, torch.Tensor],
    targets: torch.Tensor,
    loss_moe_balance: torch.Tensor,
    criteria: dict[str, nn.BCEWithLogitsLoss],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply the exact three-task BCE sum plus 0.01 times MoE balance loss."""
    task_losses = {
        "diabetes": criteria["diabetes"](
            logits["diabetes"], targets[:, TARGET_INDEX["diabetes"]]
        ),
        "cvd": criteria["cvd"](logits["cvd"], targets[:, TARGET_INDEX["cvd"]]),
        "obesity": criteria["obesity"](
            logits["obesity"], targets[:, TARGET_INDEX["obesity"]]
        ),
    }
    total_loss = (
        task_losses["diabetes"]
        + task_losses["cvd"]
        + task_losses["obesity"]
        + 0.01 * loss_moe_balance
    )
    return total_loss, task_losses


def train_one_epoch(
    model: BioNutriGPTStage1,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criteria: dict[str, nn.BCEWithLogitsLoss],
    device: torch.device,
) -> float:
    """Run one training epoch and return the sample-weighted mean objective."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for features, targets in dataloader:
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits, loss_moe_balance, _ = model(features)
        loss, _ = compute_multitask_loss(logits, targets, loss_moe_balance, criteria)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss encountered.")
        loss.backward()
        optimizer.step()

        batch_size = int(features.shape[0])
        total_loss += float(loss.detach().item()) * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError("Training DataLoader produced no batches.")
    return total_loss / total_samples


@torch.no_grad()
def evaluate(
    model: BioNutriGPTStage1,
    dataloader: DataLoader,
    criteria: dict[str, nn.BCEWithLogitsLoss],
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate validation loss and mean AUROC without updating model weights."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_targets: list[np.ndarray] = []
    all_probabilities: dict[str, list[np.ndarray]] = {
        "diabetes": [],
        "obesity": [],
        "cvd": [],
    }

    for features, targets in dataloader:
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits, loss_moe_balance, _ = model(features)
        loss, _ = compute_multitask_loss(logits, targets, loss_moe_balance, criteria)

        batch_size = int(features.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        all_targets.append(targets.cpu().numpy())
        for task_name in all_probabilities:
            all_probabilities[task_name].append(
                torch.sigmoid(logits[task_name]).cpu().numpy()
            )

    if total_samples == 0:
        raise RuntimeError("Validation DataLoader produced no batches.")

    y_true = np.concatenate(all_targets, axis=0)
    auc_by_task: dict[str, float] = {}
    for task_name, target_index in TARGET_INDEX.items():
        labels = y_true[:, target_index]
        if np.unique(labels).size != 2:
            raise ValueError(
                f"Validation fold contains only one class for {task_name}; "
                "AUROC and early stopping cannot be computed."
            )
        probabilities = np.concatenate(all_probabilities[task_name], axis=0)
        auc_by_task[task_name] = float(roc_auc_score(labels, probabilities))

    mean_auroc = float(np.mean(list(auc_by_task.values())))
    return {
        "loss": total_loss / total_samples,
        "auroc": auc_by_task,
        "mean_auroc": mean_auroc,
    }


def atomic_save_state_dict(model: nn.Module, checkpoint_path: Path) -> None:
    """Write a state-dict checkpoint atomically to avoid partial checkpoint files."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    torch.save(model.state_dict(), temporary_path)
    os.replace(temporary_path, checkpoint_path)


def fit(
    model: BioNutriGPTStage1,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    criteria: dict[str, nn.BCEWithLogitsLoss],
    device: torch.device,
    checkpoint_path: Path,
) -> list[dict[str, float]]:
    """Train with AdamW, cosine annealing, AUROC early stopping, and best save."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=40,
        eta_min=1e-6,
    )

    best_validation_mean_auroc = -math.inf
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criteria=criteria,
            device=device,
        )
        validation_result = evaluate(
            model=model,
            dataloader=validation_loader,
            criteria=criteria,
            device=device,
        )

        current_lr = float(optimizer.param_groups[0]["lr"])
        epoch_record = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "validation_loss": float(validation_result["loss"]),
            "validation_mean_auroc": float(validation_result["mean_auroc"]),
            "validation_diabetes_auroc": float(validation_result["auroc"]["diabetes"]),
            "validation_obesity_auroc": float(validation_result["auroc"]["obesity"]),
            "validation_cvd_auroc": float(validation_result["auroc"]["cvd"]),
            "learning_rate": current_lr,
        }
        history.append(epoch_record)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.5f} | "
            f"val_loss={validation_result['loss']:.5f} | "
            f"val_AUROC (diab/obes/cvd)="
            f"{validation_result['auroc']['diabetes']:.4f}/"
            f"{validation_result['auroc']['obesity']:.4f}/"
            f"{validation_result['auroc']['cvd']:.4f} | "
            f"mean={validation_result['mean_auroc']:.4f} | lr={current_lr:.2e}"
        )

        if validation_result["mean_auroc"] > best_validation_mean_auroc:
            best_validation_mean_auroc = validation_result["mean_auroc"]
            epochs_without_improvement = 0
            atomic_save_state_dict(model, checkpoint_path)
            print(f"  Saved new best checkpoint: {checkpoint_path}")
        else:
            epochs_without_improvement += 1

        scheduler.step()

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(
                "Early stopping: validation mean AUROC did not improve for "
                f"{EARLY_STOPPING_PATIENCE} consecutive epochs."
            )
            break

    if not checkpoint_path.is_file():
        raise RuntimeError("Training ended without producing a best-model checkpoint.")

    print(f"Best validation mean AUROC: {best_validation_mean_auroc:.4f}")
    return history


# ---------------------------------------------------------------------------
# 5. Temporal external evaluation and reporting
# ---------------------------------------------------------------------------


def load_state_dict_checkpoint(model: nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    """Load a weights-only state dictionary, with compatibility for older PyTorch."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Best checkpoint not found: {checkpoint_path}")
    try:
        state_dict = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:  # PyTorch versions predating the weights_only argument.
        state_dict = torch.load(checkpoint_path, map_location=device)
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint does not contain a model state dictionary.")
    model.load_state_dict(state_dict, strict=True)


@torch.no_grad()
def run_prospective_test(
    model: BioNutriGPTStage1,
    test_csv_path: Path,
    checkpoint_path: Path,
    metrics_json_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Reload the best model and evaluate the held-out 2017–2020 cohort."""
    print("\nLoading best checkpoint for prospective temporal evaluation...")
    load_state_dict_checkpoint(model, checkpoint_path, device)
    model.eval()

    test_dataframe = pd.read_csv(test_csv_path)
    test_features, test_targets = validate_and_extract_arrays(
        test_dataframe,
        dataset_name="2017–2020 prospective test dataset",
        expected_rows=EXPECTED_TEST_ROWS,
    )
    test_dataset = NHANESDataset(test_features, test_targets)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    collected_targets: list[np.ndarray] = []
    collected_probabilities: dict[str, list[np.ndarray]] = {
        "diabetes": [],
        "obesity": [],
        "cvd": [],
    }

    for features, targets in test_loader:
        features = features.to(device, non_blocking=True)
        logits, _, _ = model(features)
        collected_targets.append(targets.numpy())
        for task_name in collected_probabilities:
            collected_probabilities[task_name].append(
                torch.sigmoid(logits[task_name]).cpu().numpy()
            )

    y_true = np.concatenate(collected_targets, axis=0)
    task_metrics: dict[str, dict[str, float]] = {}

    for task_name, target_index in TARGET_INDEX.items():
        labels = y_true[:, target_index].astype(np.int64, copy=False)
        probabilities = np.concatenate(collected_probabilities[task_name], axis=0)
        if np.unique(labels).size != 2:
            raise ValueError(
                f"Prospective test cohort contains only one class for {task_name}; "
                "AUROC is undefined."
            )
        task_metrics[task_name] = {
            "auroc": float(roc_auc_score(labels, probabilities)),
            "pr_auc_average_precision": float(
                average_precision_score(labels, probabilities)
            ),
            "brier_score": float(brier_score_loss(labels, probabilities)),
        }

    mean_metrics = {
        metric_name: float(
            np.mean([task_metrics[task][metric_name] for task in task_metrics])
        )
        for metric_name in ("auroc", "pr_auc_average_precision", "brier_score")
    }
    report: dict[str, Any] = {
        "evaluation": "temporal_external_validation",
        "cohort": "NHANES 2017-2020",
        "n_samples": int(len(test_dataset)),
        "target_order": TARGET_COLS,
        "metrics": {
            "diabetes": task_metrics["diabetes"],
            "obesity": task_metrics["obesity"],
            "cvd": task_metrics["cvd"],
            "mean": mean_metrics,
        },
        "checkpoint": str(checkpoint_path),
    }

    display_table = pd.DataFrame.from_dict(report["metrics"], orient="index")
    display_table.index.name = "Outcome"
    display_table = display_table.rename(
        columns={
            "auroc": "AUROC",
            "pr_auc_average_precision": "PR-AUC (AP)",
            "brier_score": "Brier Score",
        }
    )
    print("\nProspective temporal test results (2017–2020):")
    print(display_table.to_string(float_format=lambda value: f"{value:.4f}"))

    metrics_json_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_json_path.open("w", encoding="utf-8") as json_file:
        json.dump(report, json_file, indent=2, ensure_ascii=False)
        json_file.write("\n")
    print(f"\nSaved prospective evaluation report: {metrics_json_path}")

    return report


# ---------------------------------------------------------------------------
# 6. End-to-end execution
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    # Fallbacks: script_dir / "Data" or script_dir.parent / "Data"
    if (script_dir / "Data").is_dir():
        default_data_dir = script_dir / "Data"
    else:
        default_data_dir = script_dir.parent / "Data"

    default_output_dir = script_dir / "saved model"

    parser = argparse.ArgumentParser(
        description="Train BioNutriGPT Stage 1 and run 2017–2020 temporal evaluation."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir,
        help="Path to folder containing the ML-ready CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Directory to save model checkpoint and evaluation JSON (defaults to Less Data/saved model).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_csv_path = data_dir / "BioNutriGPT_train_2007_2016_ml_ready.csv"
    test_csv_path = data_dir / "BioNutriGPT_test_2017_2020_ml_ready.csv"
    preprocessing_artifact_path = data_dir / "preprocessing_artifacts.joblib"
    checkpoint_path = output_dir / "bionutrigpt_stage1_best.pt"
    metrics_json_path = output_dir / "stage1_test_evaluation_report.json"

    set_global_seed(SEED)

    print("BioNutriGPT Stage 1")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Training data: {train_csv_path}")
    print(f"Prospective test data: {test_csv_path}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    for required_path in (train_csv_path, test_csv_path):
        if not required_path.is_file():
            raise FileNotFoundError(
                f"Required dataset not found: {required_path}\n"
                "Place the CSV at the specified project-relative path or pass --project-root."
            )

    if preprocessing_artifact_path.is_file():
        print(
            f"Preprocessing artifact found: {preprocessing_artifact_path} "
            "(not reapplied; CSVs are specified as already preprocessed)."
        )
    else:
        print(
            "Note: preprocessing_artifacts.joblib was not found. It is not required "
            "for this training run because the input CSVs are already preprocessed."
        )

    train_dataframe = pd.read_csv(train_csv_path)
    train_features, train_targets = validate_and_extract_arrays(
        train_dataframe,
        dataset_name="2007–2016 training dataset",
        expected_rows=EXPECTED_TRAIN_ROWS,
    )
    validate_historical_training_distribution(train_dataframe)

    # Use an integer validation size to obtain exactly 20,140/3,554 rows while
    # retaining sklearn's stratified train_test_split and random_state=42.
    all_indices = np.arange(len(train_dataframe))
    train_indices, validation_indices = train_test_split(
        all_indices,
        test_size=VALIDATION_SIZE,
        random_state=SEED,
        stratify=train_targets[:, TARGET_INDEX["diabetes"]],
    )
    if len(train_indices) != 20_140 or len(validation_indices) != 3_554:
        raise AssertionError(
            "Unexpected split sizes: "
            f"train={len(train_indices)}, validation={len(validation_indices)}."
        )

    train_dataset = NHANESDataset(
        train_features[train_indices], train_targets[train_indices]
    )
    validation_dataset = NHANESDataset(
        train_features[validation_indices], train_targets[validation_indices]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_generator = torch.Generator()
    train_generator.manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        generator=train_generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    print(
        f"Rows: train={len(train_dataset):,}, "
        f"validation={len(validation_dataset):,}, "
        f"prospective test={EXPECTED_TEST_ROWS:,}"
    )
    print(
        "Input dimensions: "
        f"{len(BIO_COLS)} bio + {len(DIET_COLS)} diet + "
        f"{len(LIFE_COLS)} lifestyle + {len(DEMO_COLS)} demographic "
        f"= {len(FEATURE_COLS)} features"
    )
    print("Excluded from model input: SEQN, NHANES_CYCLE, BMXBMI, BMXWT, BMXHT")

    model = BioNutriGPTStage1().to(device)
    criteria = build_criteria(device)
    output_dir.mkdir(parents=True, exist_ok=True)

    fit(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        criteria=criteria,
        device=device,
        checkpoint_path=checkpoint_path,
    )

    run_prospective_test(
        model=model,
        test_csv_path=test_csv_path,
        checkpoint_path=checkpoint_path,
        metrics_json_path=metrics_json_path,
        device=device,
    )

    print("\nBioNutriGPT Stage 1 pipeline completed successfully.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        raise
