# Stage 1 Architecture Ablation Protocol and Plan (Version 2)

**Status:** ALL 5 VARIANTS COMPLETED (25 training runs across 5 fixed seeds) & STATISTICAL ANALYSIS VERIFIED.  
**Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU (4,096 MiB GDDR6 VRAM, CUDA available).  
**Output Directory:** `Saved Model/stage1_ablation_v2/` (completely isolated from previous comparison artifacts and production checkpoints).  
**Results Report:** See [Stage1_Ablation_v2_Results.md](file:///c:/Users/MANI%20KUMAR/CURRENT/GitHub/BioNutriGPT/Models/Less%20Data/Research%20Paper/Stage1_Ablation_v2_Results.md) and [stage1_ablation_summary.json](file:///c:/Users/MANI%20KUMAR/CURRENT/GitHub/BioNutriGPT/Models/Less%20Data/Saved%20Model/stage1_ablation_v2/stage1_ablation_summary.json).

---

## 1. Scientific Objective
This controlled study tests whether BioNutriGPT's additional architectural mechanisms provide measurable empirical predictive benefits over simpler, more parsimonious neural alternatives on the repaired Stage 1 foundation:
1. **Mixture of Experts (MoE)**: 4-expert sparse routing vs. a single dense feed-forward network.
2. **Modality Attention Pooling**: learned attention query vectors per physiological modality vs. simple unweighted group mean pooling.
3. **Modality Embeddings**: learned modality tokens added during feature tokenization vs. feature-specific tokenization without modality vectors.
4. **Disease Head Shortcuts**: multi-modality context inputs into disease heads vs. heads receiving only the 256-d patient state representation.

> [!NOTE]
> These are **component-removal experiments**, not fully parameter-matched isolations of every theoretical sub-mechanism. When a module (such as MoE) is replaced with a standard FFN, both routing dynamics and total parameter capacity change. If a component does not demonstrate a statistically robust advantage, a simpler architecture is preferred.

---

## 2. Architecture Variants and Parameter Counts

| Variant | Key Architectural Change | Trainable Parameters | Parameter Change vs. FULL |
|---|---|---:|---:|
| **FULL** | Complete existing BioNutriGPT architecture | **1,295,623** | Baseline (0) |
| **NO_MOE** | 4-expert MoE replaced with 1 dense FFN (`512 -> 256 -> 256`, GELU, Dropout 0.1); balancing loss = 0 | **700,675** | **-594,948 (-45.9%)** |
| **MEAN_POOLING** | Learned attention poolers replaced by ordinary group mean of modality tokens (`bio`, `diet`, `life`, `demo`) | **1,295,111** | **-512 (-0.04%)** |
| **NO_MODALITY_EMBEDDINGS** | Modality vectors ($4 \times 128$) removed from tokenization; tokens $= x \cdot W + b$ | **1,295,111** | **-512 (-0.04%)** |
| **NO_HEAD_SHORTCUTS** | All disease heads receive only $u \in \mathbb{R}^{256}$ via `Linear(256, 64) -> GELU -> Dropout -> Linear(64, 1)` | **1,262,855** | **-32,768 (-2.5%)** |

---

## 3. Fixed Experimental Controls Across All Variants

To ensure strict scientific control, every component outside the specific ablation is held strictly constant:
- **Cohort & Partitions:**
  - Training: 16,894 participants
  - Validation: 3,621 participants (used strictly for selection & early stopping)
  - Calibration: 3,621 participants (reserved strictly for post-selection recalibration)
  - Temporal evaluation: 7,004 participants (from NHANES 2017–March 2020)
- **Features:** Exact same 87 predictors, identical order, identical training-only preprocessor.
- **Seeds:** Exact same 5 seeds across all variants: `[42, 123, 456, 789, 2026]`.
- **Training Hyperparameters:**
  - Optimizer: AdamW (`lr = 0.0003`, `weight_decay = 0.001`, verified from `stage1_comparison_v2`)
  - Batch size: 64
  - Max epochs: 40 (with early stopping patience 7 on validation mean diabetes/CVD AUROC)
  - Loss: Unweighted binary cross-entropy averaged equally over three targets (diabetes, obesity, cvd)
  - Gradient clipping: 1.0 (float32)
  - MoE balancing loss coefficient: 0.01 for variants retaining MoE; exactly 0.0 for NO_MOE.
- **Deterministic Shuffle & Initializations:**
  - Per-epoch row shuffle generator is seeded by `seed * 10000 + epoch`, ensuring identical batch compositions across all variants.
  - Submodules shared across variants (e.g., tokenizers, Transformer layers, heads) receive modular deterministic seeding so architecture changes do not perturb the initial weights of untouched components.
- **Calibration:**
  - One unpenalized logistic recalibrator per outcome fitted strictly on calibration logits after checkpoint selection.
  - No recalibration search or temporal threshold tuning is permitted.

---

## 4. Primary Endpoints and Uncertainty Analysis

- **Primary Metrics:**
  1. Diabetes raw AUROC
  2. CVD raw AUROC
  3. Diabetes calibrated Brier score
  4. CVD calibrated Brier score
- **Secondary Metrics:**
  - Obesity AUROC, Average Precision, Brier, and Log Loss
  - Raw Brier score and log-loss for diabetes and CVD
  - Calibration diagnostics (intercept and slope)
  - Memory and latency profiles (batch sizes 1 and 128, CUDA-synchronized)
- **Statistical Uncertainty Protocol:**
  - Two-axis paired bootstrap across 7,004 participants and 5 seed pairs ($20,000$ deterministic replicates).
  - Contrasts: $\Delta = \text{Metric}(\text{FULL}) - \text{Metric}(\text{Ablation})$.
  - Family-wise Bonferroni percentile intervals adjusted across all 16 primary contrasts (4 ablations $\times$ 2 outcomes $\times$ 2 metrics; $\alpha = 0.05 / 16 = 0.003125$).
  - An interval containing zero indicates that the current empirical study did not establish a component advantage.

---

## 5. Software Preflight Verification

All 12 pre-training software checks passed cleanly in `verify_stage1_ablations_v2.py`:
1. Output shapes $(B, 3)$ and finite losses verified.
2. Parameter counts match specifications exactly.
3. NO_MOE has no routing network and returns zero auxiliary loss.
4. MEAN_POOLING matches independent group mean calculations.
5. NO_MODALITY_EMBEDDINGS retains feature-specific tokenization without modality vectors.
6. NO_HEAD_SHORTCUTS receives only 256-d patient representations.
7. Shared modules receive identical initial weights for a given seed.
8. Seeded row shuffles match across variants.
9. Disjoint partitions verified (zero ID leakage).
10. Checkpoint save/reload reproduces predictions bitwise.
11. Data hash tamper checks verified.
12. Bootstrap pairing and tie-handling verified.
