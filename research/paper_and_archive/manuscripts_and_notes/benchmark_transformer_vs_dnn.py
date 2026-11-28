#!/usr/bin/env python3
"""
BioNutriGPT Stage 1 vs. Deep Neural Network (DNN & ResNet) Benchmark Suite.

Directly compares BioNutriGPT Stage 1 (Per-Feature Tabular Transformer + MoE + Skip Highways)
against standard deep neural network baselines on tabular data:
  1. Standard Deep Neural Network (Multi-Layer Perceptron / MLP with BatchNorm & Dropout)
  2. Tabular Deep ResNet (Residual MLP with Skip Connections, Gorishniy et al., NeurIPS 2021)
  3. BioNutriGPT Stage 1 (Our Per-Feature Tabular Transformer + Switch MoE)

Trained on: NHANES Historical Cohort (2007-2016, N = 23,694; 20,140 train, 3,554 val)
Evaluated on: NHANES Prospective Holdout (2017-March 2020 Pre-Pandemic, N = 7,046)
Evaluation: Official CDC Pre-Pandemic Examination Weights (WTMECPRP) + 1,000 Bootstrap 95% CIs.
"""

from __future__ import annotations

import os
import sys
import json
import time
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.linear_model import LogisticRegression

# Ensure working directory imports
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import train_bionutrigpt_stage1 as s1

SEED = 42

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ---------------------------------------------------------------------------
# Deep Neural Network Baselines
# ---------------------------------------------------------------------------

class StandardDeepMLP(nn.Module):
    """Standard 4-layer Deep Neural Network (MLP) for Tabular Clinical Data."""
    def __init__(self, in_features: int = 87, hidden_dims: list[int] = [256, 128, 64], dropout: float = 0.2):
        super().__init__()
        layers = []
        prev_dim = in_features
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        self.backbone = nn.Sequential(*layers)
        self.head_diabetes = nn.Linear(prev_dim, 1)
        self.head_obesity = nn.Linear(prev_dim, 1)
        self.head_cvd = nn.Linear(prev_dim, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.backbone(x)
        return {
            "diabetes": self.head_diabetes(feat).squeeze(-1),
            "obesity": self.head_obesity(feat).squeeze(-1),
            "cvd": self.head_cvd(feat).squeeze(-1),
        }

class ResNetBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.norm1 = nn.BatchNorm1d(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.BatchNorm1d(dim)
        self.linear2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.linear1(self.act(self.norm1(x)))
        out = self.dropout(out)
        out = self.linear2(self.act(self.norm2(out)))
        out = self.dropout(out)
        return residual + out

class TabularResNet(nn.Module):
    """Deep Tabular ResNet with residual skip connections (Gorishniy et al., NeurIPS 2021 style)."""
    def __init__(self, in_features: int = 87, hidden_dim: int = 128, num_blocks: int = 3, dropout: float = 0.2):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        self.blocks = nn.ModuleList([ResNetBlock(hidden_dim, dropout) for _ in range(num_blocks)])
        self.head_norm = nn.BatchNorm1d(hidden_dim)
        self.head_diabetes = nn.Linear(hidden_dim, 1)
        self.head_obesity = nn.Linear(hidden_dim, 1)
        self.head_cvd = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.input_layer(x)
        for block in self.blocks:
            feat = block(feat)
        feat = self.head_norm(feat)
        return {
            "diabetes": self.head_diabetes(feat).squeeze(-1),
            "obesity": self.head_obesity(feat).squeeze(-1),
            "cvd": self.head_cvd(feat).squeeze(-1),
        }

# ---------------------------------------------------------------------------
# Training Loop for DNN Baselines
# ---------------------------------------------------------------------------

def train_dnn_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    pos_weights: dict[str, torch.Tensor],
    device: torch.device,
    epochs: int = 40,
    lr: float = 3e-4,
    patience: int = 7,
    save_path: Path = None,
) -> nn.Module:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    bce_loss_fns = {
        k: nn.BCEWithLogitsLoss(pos_weight=pos_weights[k].to(device))
        for k in pos_weights
    }

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y_dict in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            logits_dict = model(x)
            loss = sum(bce_loss_fns[k](logits_dict[k], y_dict[k].to(device)) for k in logits_dict)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y_dict in val_loader:
                x = x.to(device)
                logits_dict = model(x)
                loss = sum(bce_loss_fns[k](logits_dict[k], y_dict[k].to(device)) for k in logits_dict)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            if save_path:
                torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if save_path and save_path.exists():
        model.load_state_dict(torch.load(save_path, map_location=device))
    return model

# ---------------------------------------------------------------------------
# Metric Evaluation Utilities
# ---------------------------------------------------------------------------

def calculate_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = bin_ids == b
        if np.sum(mask) > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            ece += (np.sum(mask) / n) * np.abs(bin_acc - bin_conf)
    return float(ece)

def calculate_calibration_slope(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    eps = 1e-6
    y_prob_clipped = np.clip(y_prob, eps, 1.0 - eps)
    logit_prob = np.log(y_prob_clipped / (1.0 - y_prob_clipped)).reshape(-1, 1)
    lr = LogisticRegression(penalty=None, solver='lbfgs')
    try:
        lr.fit(logit_prob, y_true)
        return float(lr.coef_[0][0])
    except Exception:
        return 1.0

def bootstrap_eval(y_true: np.ndarray, y_prob: np.ndarray, weights: np.ndarray = None, n_bootstraps: int = 1000, seed: int = 42):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    aurocs, praucs, briers = [], [], []
    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, n)
        y_t_b = y_true[idx]
        y_p_b = y_prob[idx]
        w_b = weights[idx] if weights is not None else None
        if len(np.unique(y_t_b)) < 2:
            continue
        try:
            aurocs.append(roc_auc_score(y_t_b, y_p_b, sample_weight=w_b))
        except Exception:
            pass
        try:
            praucs.append(average_precision_score(y_t_b, y_p_b, sample_weight=w_b))
        except Exception:
            pass
        try:
            briers.append(brier_score_loss(y_t_b, y_p_b, sample_weight=w_b))
        except Exception:
            pass
    return {
        "auroc_mean": float(np.mean(aurocs)),
        "auroc_ci": [float(np.percentile(aurocs, 2.5)), float(np.percentile(aurocs, 97.5))],
        "prauc_mean": float(np.mean(praucs)),
        "prauc_ci": [float(np.percentile(praucs, 2.5)), float(np.percentile(praucs, 97.5))],
        "brier_mean": float(np.mean(briers)),
        "brier_ci": [float(np.percentile(briers, 2.5)), float(np.percentile(briers, 97.5))],
    }

# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

class TabularDataset(Dataset):
    def __init__(self, x: np.ndarray, y_dict: dict[str, np.ndarray]):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y_dict = {k: torch.tensor(v, dtype=torch.float32) for k, v in y_dict.items()}

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], {k: v[idx] for k, v in self.y_dict.items()}

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"================================================================================")
    print(f"BIONUTRIGPT STAGE 1 VS. DEEP NEURAL NETWORK (DNN & RESNET) BENCHMARK")
    print(f"Device: {device} | SEED: {SEED}")
    print(f"================================================================================")

    data_dir = BASE_DIR / "Data"
    saved_model_dir = BASE_DIR / "Saved Model"
    train_path = data_dir / "BioNutriGPT_train_2007_2016_ml_ready.csv"
    test_path = data_dir / "BioNutriGPT_test_2017_2020_ml_ready.csv"
    raw_path = data_dir / "BioNutriGPT_merged_raw_2005_2020.csv"
    stage1_ckpt = saved_model_dir / "bionutrigpt_stage1_best.pt"

    print("Loading datasets...")
    df_train_full = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df_raw = pd.read_csv(raw_path, low_memory=False)

    # Merge survey weights into test
    cols_to_merge = ["SEQN", "WTMECPRP", "SDMVPSU", "SDMVSTRA", "RIAGENDR", "RIDAGEYR"]
    df_raw_sub = df_raw[[c for c in cols_to_merge if c in df_raw.columns]].drop_duplicates(subset=["SEQN"])
    df_test_merged = df_test.merge(df_raw_sub, on="SEQN", how="left")
    weights = df_test_merged["WTMECPRP"].values
    if np.isnan(weights).any():
        weights = np.nan_to_num(weights, nan=np.nanmean(weights))

    # Split train_full into train (20,140) and val (3,554) exactly like train_bionutrigpt_stage1.py
    from sklearn.model_selection import train_test_split
    df_train, df_val = train_test_split(df_train_full, test_size=3554, random_state=SEED, shuffle=True)

    X_train = df_train[s1.FEATURE_COLS].values
    X_val = df_val[s1.FEATURE_COLS].values
    X_test = df_test[s1.FEATURE_COLS].values

    targets = ["diabetes", "obesity", "cvd"]
    target_col_map = {"diabetes": "TARGET_DIABETES", "obesity": "TARGET_OBESITY", "cvd": "TARGET_CVD"}

    y_train = {k: df_train[target_col_map[k]].values for k in targets}
    y_val = {k: df_val[target_col_map[k]].values for k in targets}
    y_test = {k: df_test[target_col_map[k]].values for k in targets}

    # Compute pos_weights matching train_bionutrigpt_stage1
    pos_weights = {}
    for k in targets:
        n_pos = np.sum(y_train[k] == 1)
        n_neg = len(y_train[k]) - n_pos
        pos_weights[k] = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
        print(f"Target '{k}': Pos={n_pos}, Neg={n_neg}, pos_weight={pos_weights[k].item():.4f}")

    train_dataset = TabularDataset(X_train, y_train)
    val_dataset = TabularDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    # ---------------------------------------------------------------------------
    # Model 1: Train Standard Deep Neural Network (MLP)
    # ---------------------------------------------------------------------------
    print("\n[1/3] Training Standard Deep Neural Network (4-Layer MLP with BatchNorm & Dropout)...")
    mlp_path = saved_model_dir / "dnn_mlp_baseline.pt"
    mlp_model = StandardDeepMLP(in_features=len(s1.FEATURE_COLS))
    mlp_model = train_dnn_model(
        mlp_model, train_loader, val_loader, pos_weights, device,
        epochs=40, lr=1e-3, patience=7, save_path=mlp_path
    )
    print("Standard Deep MLP training complete.")

    # ---------------------------------------------------------------------------
    # Model 2: Train Tabular Deep ResNet (Residual MLP)
    # ---------------------------------------------------------------------------
    print("\n[2/3] Training Tabular Deep ResNet (Residual MLP with Skip Connections)...")
    resnet_path = saved_model_dir / "dnn_resnet_baseline.pt"
    resnet_model = TabularResNet(in_features=len(s1.FEATURE_COLS))
    resnet_model = train_dnn_model(
        resnet_model, train_loader, val_loader, pos_weights, device,
        epochs=40, lr=1e-3, patience=7, save_path=resnet_path
    )
    print("Tabular Deep ResNet training complete.")

    # ---------------------------------------------------------------------------
    # Model 3: Load BioNutriGPT Stage 1
    # ---------------------------------------------------------------------------
    print("\n[3/3] Loading BioNutriGPT Stage 1 (Per-Feature Transformer + MoE)...")
    bionutri_model = s1.BioNutriGPTStage1().to(device)
    s1.load_state_dict_checkpoint(bionutri_model, stage1_ckpt, device)
    bionutri_model.eval()

    # ---------------------------------------------------------------------------
    # Generate Predictions on Prospective Test Set (N = 7,046)
    # ---------------------------------------------------------------------------
    print("\nGenerating predictions on prospective test cohort (N = 7,046)...")
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        # Standard MLP
        mlp_model.eval()
        mlp_logits = mlp_model(X_test_tensor)
        mlp_probs = {k: torch.sigmoid(mlp_logits[k]).cpu().numpy() for k in targets}

        # ResNet
        resnet_model.eval()
        resnet_logits = resnet_model(X_test_tensor)
        resnet_probs = {k: torch.sigmoid(resnet_logits[k]).cpu().numpy() for k in targets}

        # BioNutriGPT
        bio_logits, _, _ = bionutri_model(X_test_tensor)
        bio_probs = {k: torch.sigmoid(bio_logits[k]).cpu().numpy() for k in targets}

    all_models = {
        "Standard_Deep_MLP": mlp_probs,
        "Tabular_Deep_ResNet": resnet_probs,
        "BioNutriGPT_Stage1": bio_probs
    }

    # ---------------------------------------------------------------------------
    # Compute Comprehensive Evaluation Metrics
    # ---------------------------------------------------------------------------
    print("\nComputing survey-weighted metrics with 1,000 bootstrap iterations...")
    benchmark_results: dict[str, Any] = {}

    for t in targets:
        y_true = y_test[t]
        benchmark_results[t] = {}
        print(f"\n=======================================================")
        print(f"TARGET: {t.upper()} (Prevalence: {np.mean(y_true)*100:.2f}%)")
        print(f"=======================================================")

        for m_name, probs_dict in all_models.items():
            prob = probs_dict[t]
            # Unweighted
            raw_eval = bootstrap_eval(y_true, prob, weights=None, n_bootstraps=1000, seed=SEED)
            # Survey-weighted
            wt_eval = bootstrap_eval(y_true, prob, weights=weights, n_bootstraps=1000, seed=SEED)
            # Calibration
            slope = calculate_calibration_slope(y_true, prob)
            ece = calculate_ece(y_true, prob)

            benchmark_results[t][m_name] = {
                "unweighted": raw_eval,
                "survey_weighted": wt_eval,
                "calibration": {
                    "slope": round(slope, 4),
                    "ece": round(ece, 4)
                }
            }
            print(f"[{m_name}]")
            print(f"  Survey-Weighted AUROC: {wt_eval['auroc_mean']:.4f} [95% CI: {wt_eval['auroc_ci'][0]:.4f} - {wt_eval['auroc_ci'][1]:.4f}]")
            print(f"  Survey-Weighted PR-AUC: {wt_eval['prauc_mean']:.4f} [95% CI: {wt_eval['prauc_ci'][0]:.4f} - {wt_eval['prauc_ci'][1]:.4f}]")
            print(f"  Brier Score: {wt_eval['brier_mean']:.4f} | Calibration Slope: {slope:.2f} | ECE: {ece:.4f}")

    # Save benchmark report
    out_json = saved_model_dir / "transformer_vs_deep_neural_network_benchmark.json"
    with open(out_json, "w") as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"\nBenchmark report successfully saved to: {out_json}")

    # Print Formatted Markdown Comparison Table
    print("\n" + "="*80)
    print("FINAL PUBLICATION-GRADE COMPARISON TABLE")
    print("="*80)
    print("| Target Disease | Model Architecture | Wt. AUROC [95% CI] | Wt. PR-AUC | Brier Score | Calib. Slope | ECE |")
    print("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |")
    for t in targets:
        for m_name in all_models.keys():
            res = benchmark_results[t][m_name]
            wt = res["survey_weighted"]
            cal = res["calibration"]
            ci_str = f"[{wt['auroc_ci'][0]:.3f}–{wt['auroc_ci'][1]:.3f}]"
            print(f"| {t.capitalize():<14} | {m_name:<20} | {wt['auroc_mean']:.4f} {ci_str:<13} | {wt['prauc_mean']:.4f} | {wt['brier_mean']:.4f} | {cal['slope']:<11.2f} | {cal['ece']:.4f} |")

if __name__ == "__main__":
    main()
