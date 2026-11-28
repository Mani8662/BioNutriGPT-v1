# BioNutriGPT Stage 1 Architecture Ablation Study (v2) — Results Report

## 1. Study Purpose and Methodological Scope
This controlled five-seed architecture-ablation study evaluates whether BioNutriGPT's additional architectural components provide measurable empirical benefits over simpler formulations on the repaired Stage 1 foundation (16,894 training, 3,621 validation, 3,621 calibration, 7,004 temporal evaluation participants).

The four targeted components are:
1. **Mixture of Experts (MoE)**: 4-expert sparse MoE (top-2 routing) vs. single dense feed-forward network (512 -> 256 -> 256; balancing loss = 0).
2. **Modality Attention Pooling**: learned attention query vectors per modality vs. ordinary unweighted group mean pooling.
3. **Modality Embeddings**: learned modality vectors added during tokenization vs. feature-specific tokenization without modality vectors.
4. **Disease Head Shortcuts**: multi-modality context inputs into disease heads vs. heads receiving only the 256-d patient state representation.

All models use identical 87 features, identical AdamW hyperparameters (lr=0.0003, weight_decay=0.001, batch_size=64, patience=7), identical row shuffle sequences per seed, identical modular initializations where modules are shared, unpenalized logistic recalibration fitted strictly on calibration participants, and evaluation on the frozen temporal cohort (N=7,004).

## 2. Performance Summary Across Five Fixed Seeds (Mean ± SD)

| Architecture Variant | Trainable Parameters | Diabetes Raw AUROC | CVD Raw AUROC | Diabetes Cal. Brier | CVD Cal. Brier | Obesity Raw AUROC | Mean Train Time |
|---|---:|---:|---:|---:|---:|---:|---:|
| **FULL** | 1,295,623 | 0.8090 ± 0.0011 | 0.8316 ± 0.0013 | 0.121532 ± 0.000186 | 0.086997 ± 0.000266 | 0.7501 ± 0.0027 | 308.9s |
| **NO_MOE** | 700,675 | 0.8086 ± 0.0014 | 0.8315 ± 0.0005 | 0.121432 ± 0.000454 | 0.087058 ± 0.000112 | 0.7521 ± 0.0030 | 260.7s |
| **MEAN_POOLING** | 1,295,111 | 0.8075 ± 0.0010 | 0.8305 ± 0.0004 | 0.121759 ± 0.000198 | 0.087260 ± 0.000103 | 0.7509 ± 0.0014 | 155.1s |
| **NO_MODALITY_EMBEDDINGS** | 1,295,111 | 0.8073 ± 0.0024 | 0.8303 ± 0.0021 | 0.121884 ± 0.000504 | 0.087245 ± 0.000433 | 0.7483 ± 0.0026 | 186.1s |
| **NO_HEAD_SHORTCUTS** | 1,262,855 | 0.8074 ± 0.0017 | 0.8313 ± 0.0026 | 0.121923 ± 0.000482 | 0.087149 ± 0.000489 | 0.7485 ± 0.0015 | 138.7s |

## 3. Two-Axis Paired Bootstrap Analysis (20,000 Replicates)
Contrasts represent **FULL minus Ablation**. Positive AUROC differences favor FULL; negative Brier differences favor FULL. Intervals are constructed using two-axis resampling across 7,004 participants and 5 training seeds.

| Ablation | Outcome | Metric | Mean Diff | Pointwise 95% CI | Bonferroni 16-Contrast 95% CI | Inference |
|---|---|---|---:|---:|---:|---|
| NO_MOE | DIABETES | Raw AUROC | +0.0004 | [-0.0032, +0.0039] | [-0.0052, +0.0058] | Indistinguishable (CI spans zero) |
| NO_MOE | DIABETES | Calibrated Brier | +0.0001 | [-0.0008, +0.0010] | [-0.0013, +0.0015] | Indistinguishable (CI spans zero) |
| NO_MOE | CVD | Raw AUROC | +0.0002 | [-0.0030, +0.0032] | [-0.0048, +0.0047] | Indistinguishable (CI spans zero) |
| NO_MOE | CVD | Calibrated Brier | -0.0001 | [-0.0006, +0.0005] | [-0.0009, +0.0009] | Indistinguishable (CI spans zero) |
| MEAN_POOLING | DIABETES | Raw AUROC | +0.0015 | [-0.0007, +0.0036] | [-0.0019, +0.0053] | Indistinguishable (CI spans zero) |
| MEAN_POOLING | DIABETES | Calibrated Brier | -0.0002 | [-0.0007, +0.0003] | [-0.0010, +0.0006] | Indistinguishable (CI spans zero) |
| MEAN_POOLING | CVD | Raw AUROC | +0.0011 | [-0.0009, +0.0033] | [-0.0022, +0.0046] | Indistinguishable (CI spans zero) |
| MEAN_POOLING | CVD | Calibrated Brier | -0.0003 | [-0.0007, +0.0002] | [-0.0010, +0.0004] | Indistinguishable (CI spans zero) |
| NO_MODALITY_EMBEDDINGS | DIABETES | Raw AUROC | +0.0016 | [-0.0011, +0.0045] | [-0.0026, +0.0062] | Indistinguishable (CI spans zero) |
| NO_MODALITY_EMBEDDINGS | DIABETES | Calibrated Brier | -0.0004 | [-0.0011, +0.0003] | [-0.0015, +0.0007] | Indistinguishable (CI spans zero) |
| NO_MODALITY_EMBEDDINGS | CVD | Raw AUROC | +0.0013 | [-0.0014, +0.0042] | [-0.0029, +0.0061] | Indistinguishable (CI spans zero) |
| NO_MODALITY_EMBEDDINGS | CVD | Calibrated Brier | -0.0002 | [-0.0008, +0.0003] | [-0.0012, +0.0006] | Indistinguishable (CI spans zero) |
| NO_HEAD_SHORTCUTS | DIABETES | Raw AUROC | +0.0015 | [-0.0012, +0.0043] | [-0.0027, +0.0059] | Indistinguishable (CI spans zero) |
| NO_HEAD_SHORTCUTS | DIABETES | Calibrated Brier | -0.0004 | [-0.0011, +0.0003] | [-0.0015, +0.0006] | Indistinguishable (CI spans zero) |
| NO_HEAD_SHORTCUTS | CVD | Raw AUROC | +0.0003 | [-0.0032, +0.0038] | [-0.0052, +0.0055] | Indistinguishable (CI spans zero) |
| NO_HEAD_SHORTCUTS | CVD | Calibrated Brier | -0.0002 | [-0.0007, +0.0005] | [-0.0011, +0.0009] | Indistinguishable (CI spans zero) |

## 4. Primary Contrasts Forest Plot
```
=========================================================================================
STAGE 1 ABLATION FOREST PLOT: FULL MINUS ABLATION CONTRASTS
Pointwise 95% CI: [====o====] | Bonferroni 95% CI: (....[====o====]....) | Zero: |
For AUROC: Positive favors FULL (+) | For Brier: Negative favors FULL (-)
=========================================================================================
NO MOE                 DIABETES AUROC  +0.0004 [-0.0032, +0.0039]                 .===o===..                 
NO MOE                 DIABETES Brier  +0.0001 [-0.0008, +0.0010]                ..===o====..                
NO MOE                 CVD      AUROC  +0.0002 [-0.0030, +0.0032]                 ..==o===.                  
NO MOE                 CVD      Brier  -0.0001 [-0.0006, +0.0005]                 .===o==..                  
MEAN POOLING           DIABETES AUROC  +0.0015 [-0.0007, +0.0036]                   .==o==.                  
MEAN POOLING           DIABETES Brier  -0.0002 [-0.0007, +0.0003]                 .==o==.                    
MEAN POOLING           CVD      AUROC  +0.0011 [-0.0009, +0.0033]                   .==o==.                  
MEAN POOLING           CVD      Brier  -0.0003 [-0.0007, +0.0002]                 .==o==.                    
NO MODALITY EMBEDDINGS DIABETES AUROC  +0.0016 [-0.0011, +0.0045]                   .==o===.                 
NO MODALITY EMBEDDINGS DIABETES Brier  -0.0004 [-0.0011, +0.0003]               ..===o==..                   
NO MODALITY EMBEDDINGS CVD      AUROC  +0.0013 [-0.0014, +0.0042]                   .==o==..                 
NO MODALITY EMBEDDINGS CVD      Brier  -0.0002 [-0.0008, +0.0003]                ..==o==.                    
NO HEAD SHORTCUTS      DIABETES AUROC  +0.0015 [-0.0012, +0.0043]                   .==o==..                 
NO HEAD SHORTCUTS      DIABETES Brier  -0.0004 [-0.0011, +0.0003]               ..==o===..                   
NO HEAD SHORTCUTS      CVD      AUROC  +0.0003 [-0.0032, +0.0038]                 .===o===.                  
NO HEAD SHORTCUTS      CVD      Brier  -0.0002 [-0.0007, +0.0005]                 .==o===..                  
=========================================================================================
Key: † = Pointwise 95% CI excludes zero | * = Bonferroni 16-contrast CI excludes zero
```

## 5. Architectural Findings & Scientific Synthesis
1. **Mixture of Experts (MoE)**: Removing MoE reduces trainable parameter count by 46% (from 1,295,623 to 700,675). The empirical difference between FULL and NO_MOE reveals whether sparse top-2 routing and extra capacity justify their computational cost.
2. **Modality Attention Pooling**: Replacing learned query vectors with group mean pooling tests whether learned attention weighting over heterogeneous tokens provides a measurable advantage over simple linear aggregation.
3. **Modality Embeddings**: Testing whether explicit modality tags are required for the tabular Transformer to recognize feature groupings.
4. **Head Shortcuts**: Testing whether direct multi-modality residual shortcuts into specific disease heads outperform passing only the unified patient representation.

## 6. Limitations and Clinical Scope
- All evaluations are conducted on NHANES cross-sectional data; they do not establish causal disease prevention or Indian clinical effectiveness.
- Five random seeds capture limited training stochasticity; confidence intervals are conditional on the fixed training recipe and modest parameter budget.
- An interval containing zero indicates that the current empirical study did not detect an advantage; it does not prove exact architectural equivalence.