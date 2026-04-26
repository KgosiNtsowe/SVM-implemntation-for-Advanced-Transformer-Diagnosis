# Advanced Diagnostics for Power Transformers
## A Machine Learning Approach for Health Assessment Using Limited Data

**Third Year Individual Project — EEEN30330**  
**University of Manchester, Department of Electrical and Electronic Engineering**  
**Student ID:** 11409134  
**Academic Year:** 2025–2026
---
## Introduction

Power transformers are among the most critical and capital-intensive assets in high-voltage electrical power systems, accounting for up to 60% of substation investment. Unexpected transformer failure causes prolonged outages, significant revenue loss, and serious safety and environmental risks. With the average global age of in-service transformers increasing, the probability of failure is rising.

Current diagnostic practice relies on **Dissolved Gas Analysis (DGA)** — measuring the concentrations of fault-indicating gases dissolved in transformer insulating oil. Existing interpretation methods, such as the Duval Triangle (IEC 60599) and Rogers Ratios, apply deterministic rule-based classification. These methods suffer from systematic limitations: ambiguous boundary regions produce contradictory results, and the rigid categorical rules cannot exploit the full statistical structure of the data.

This project develops a **Support Vector Machine (SVM) classifier** trained on DGA data to perform automated, probabilistic fault diagnosis. The approach transitions from rule-based preventative maintenance toward data-driven predictive maintenance. Special methodological attention is given to the challenge of **limited and imbalanced fault data**, which is the defining constraint of real-world DGA datasets.

**Five fault classes are diagnosed:**
| Label | Class | Description |
|---|---|---|
| 0 | Healthy | Normal transformer operation |
| 1 | D1 | Low-energy electrical discharge (arcing) |
| 2 | T1 | Thermal fault < 300°C |
| 3 | T2 | Thermal fault 300–700°C |
| 4 | T3 | Thermal fault > 700°C |

---

## Contextual Overview — System Architecture
```
Raw DGA Data (ppm)                  Preprocessing Pipeline
┌─────────────────────┐            ┌──────────────────────────────┐
│ 7 Gas Concentrations│            │ 1. log1p transform (gases)   │
│ H2, CH4, C2H6,      │ ──────────▶│ 2. Robust scaling (train fit)│
│ C2H4, C2H2, CO, CO2 │            │ 3. Stratified 80/20 split    │
│ + 3 Duval %         │            │ 4. SMOTE balancing (train)   │
└─────────────────────┘            └──────────┬───────────────────┘
                                              │
                                   ┌──────────▼───────────────────┐
                                   │   SVM Classifier             │
                                   │   Kernel: RBF                │
                                   │   C=100, γ=0.1               │
                                   │   Strategy: One-vs-One       │
                                   │   Training: SMOTE balanced   │
                                   └──────────┬───────────────────┘
                                              │
                        ┌─────────────────────▼──────────────────────┐
                        │             Evaluation                      │
                        │  • Per-class Precision / Recall / F1        │
                        │  • Macro F1 (primary metric)                │
                        │  • Learning curves (data efficiency)        │
                        │  • Noise injection (sensor robustness)      │
                        │  • Permutation feature importance (XAI)     │
                        └────────────────────────────────────────────┘
```
**Key result:** SVM achieves Macro F1 = 0.9912 on the held-out test set, representing a 158% improvement over the Duval Triangle rule-based baseline (F1 = 0.383).

---

## Installation Instructions

### Requirements

- **MATLAB** R2019b or later
- **Statistics and Machine Learning Toolbox** (required for `fitcecoc`, `fitcknn`, `fitctree`, `cvpartition`)
- No additional toolboxes required for core functionality

### Setup

1. Clone or download this repository:
```
https://github.com/KgosiNtsowe/Machine-Learning.git
```

2. Scripts must be run in order — each checkpoint produces output files consumed by the next. 

### Dependencies between scripts

```
Data_Analysis.m ──▶  audit_outputs/
        │
Data_Preprocessing.m ──▶  pipeline_outputs/
        │
Baseline_Models.m    ──▶  baseline_outputs/
        │
SVM_GridSearch_Hyperparameters.m       ──▶  svm_outputs/
        │
LearningCurves_and_Analysis.m       ──▶  learning_curve_outputs/
Noise_Injection_Analysis.m         ──▶  noise_outputs/
xAI_Analysis.m                 ──▶  xai_outputs/
```

---

## How to Run the Software

Run each script sequentially from the MATLAB Command Window. Each script is self-contained and saves all outputs to its designated output folder automatically.

### Step 1 — Data Analysis
```matlab
run('Data_Analysis.m')
```
Produces: `audit_outputs/` — distribution plots, correlation heatmap, summary statistics CSV.  
---

### Step 2 — Preprocessing Pipeline
```matlab
run('Data_Preprocessing.m')
```
Produces: `pipeline_outputs/` — four CSV files required by all downstream scripts:
- `X_train_scaled.csv` — 1,127 × 10 scaled training features (pre-SMOTE)
- `X_train_smote.csv` — 3,750 × 10 SMOTE-balanced training features
- `X_test_scaled.csv` — 281 × 10 scaled test features (quarantined)
- `scaler_params.csv` — median and IQR parameters for deployment
⚠️ **Must be run before any other script.**
---

### Step 3 — Baseline Models
```matlab
run('Baseline_Models.m')
```
Produces: `baseline_outputs/` — performance tables, confusion matrices, McNemar test results across three experimental conditions (imbalanced, SMOTE, cost-sensitive).  
---

### Step 4 — SVM with Grid Search
```matlab
run('SVM_GridSearch_Hyperparameters.m')
```
Produces: `svm_outputs/` — grid search heatmaps, confusion matrices, full comparison table.   
⚠️ **Best model:** Experiment B (SMOTE, C=100, γ=0.1).

---

### Step 5 — Learning Curves
```matlab
run('LearningCurves_and_Analysis.m')
```
Produces: `learning_curve_outputs/` — learning curve figures, bias-variance plots, data efficiency bar chart.  
---

### Step 6 — Noise Injection Robustness
```matlab
run('Noise_Injection_Analysis.m')
```
Produces: `noise_outputs/` — degradation curves, D1 recall under noise, degradation heatmap.  
---

### Step 7 — Explainability (XAI)
```matlab
run('xAI_Analysis.m')
```
Produces: `xai_outputs/` — global importance plot, per-class heatmap, IEC 60599 alignment figure.  
---

## Technical Details

### Dataset
The raw dataset 'Base_de_Datos_Eng_5_2_2026_V1.csv' was cleaned , labelled and augmented with Duval Triangle gas percentages to produce 'Final_Transformer_Dataset_with_Duval'

| Property | Value |
|---|---|
| Total samples | 1,408 |
| Features | 10 (7 raw gases + 3 Duval % ) |
| Classes | 5 (Healthy, D1, T1, T2, T3) |
| Class imbalance ratio | 22.3:1 (Healthy:D1) |

 Labels were assigned using the Rogers Ratios method method columns in raw data per IEC 60599. Samples labelled as D2, DT, PD, or Undefined were excluded (24.5% of raw data) due to insufficient sample counts for reliable ML training.
---

### Preprocessing Decisions

Every preprocessing decision is justified by findings from the data analysis:

| Finding | Decision | Justification |
|---|---|---|
| Skewness 3.5–17.3 | `log1p` transform on gas columns | Arrhenius kinetics of pyrolysis produce log-normal distributions |
| Outliers up to 3,060 ppm | Robust scaling (median + IQR) | Standard scaling would compress normal operating range |
| 22.3:1 imbalance | SMOTE on training set only | Prevents systematic bias toward majority class |
| Acetylene IQR = 0 | Scale set to 1.0 (identity) | Standard sklearn/MATLAB convention for zero-IQR features |
| C2H2 / %C2H2 r=0.998 | Both retained | SVM insensitive to collinearity; Duval ratios add independent structure |
---

### Model Selection

Three experimental conditions were evaluated:

| Experiment | Training data | Best C | Best γ | Test Macro F1 |
|---|---|---|---|---|
| A — Imbalanced | 1,127 samples | 1,000 | 0.1 | 0.9912 |
| **B — SMOTE** | **3,750 samples** | **100** | **0.1** | **0.9912** |
| C — Cost-sensitive | 1,127 samples | 1,000 | 0.1 | 0.9912 |

**Experiment B is the primary model** because it uses the methodologically most rigorous training condition — balanced data provides the fair comparison against baselines.

Hyperparameters selected by 5-fold stratified cross-validation, optimising macro F1 over the grid C ∈ {0.1, 1, 10, 100, 1000} × γ ∈ {0.001, 0.01, 0.1, 1, 10}.

---
### Key Results Summary
| Analysis | Key Finding |
|---|---|
| Clean test performance | Macro F1 = 0.9912 vs Duval baseline 0.383 |
| D1 arcing fault recall | 1.000 (no arcing faults missed on test set) |
| T2 misclassifications | 2 samples at IEC 60599 detection limit (< 5 ppm all gases) |
| Data efficiency | 95% of peak performance at 50% of training data (563 samples) |
| Noise robustness at 10% | SVM D1 recall = 0.883 vs DT = 0.411 |
| XAI — D1 class | %C2H2 recall drop = 0.970 — consistent with IEC 60599 Table A.1 |
| XAI — T3 class | C2H4 recall drop = 0.747 — consistent with high-temp pyrolysis |

---
### Algorithms and Tools
| Component | Implementation | MATLAB function |
|---|---|---|
| SVM (multiclass) | One-vs-One, RBF kernel | `fitcecoc` + `templateSVM` |
| Cross-validation | Stratify k-fold (k=5) | `cvpartition` |
| SMOTE | Custom implementation | `knnsearch` |
| Robust scaling | Custom implementation | `quantile`, `median` |
| Permutation importance | Custom implementation | `randperm` |
| Baseline — k-NN | k=1, Euclidean distance | `fitcknn` |
| Baseline — Decision Tree | CART, max depth 10 | `fitctree` |
| Baseline — Duval Triangle | IEC 60599 polygon rules | Custom function |

---
## Known Issues and Future Improvements
### Known Limitations

1. **Class scope:** The model diagnoses five classes only (Healthy, D1, T1, T2, T3). Fault types D2 (high-energy discharge), DT (mixed thermal/discharge), and PD (partial discharge) are excluded due to insufficient training samples in the available dataset. The model will not produce reliable predictions for these fault types.

2. **T2 boundary cases:** Two T2 (moderate thermal) test samples are misclassified as Healthy. Both have gas concentrations below 5 ppm across all gases — within the IEC 60599 acknowledged detection limit for incipient faults. This represents an irreducible minimum, not a correctable model error.

3. **Duval % collinearity:** C2H2 and %C2H2 have Spearman r = 0.998. Permutation importance values for these features must be interpreted jointly as a combined acetylene signal (combined importance = 1.3667), not independently.

4. **Dataset source:** Labels are derived from Rogers Ratios applied to a single open-source database. Generalisation to transformers of different makes, oil types, or operating histories has not been validated.

### Future Improvements

1. **Expanded fault classes:** Acquire labelled D2, DT, and PD samples to extend the classifier to the full IEC 60599 fault taxonomy.

2. **Temporal modelling:** Incorporate rate-of-change features (Δgas/Δtime) to capture fault progression, not just instantaneous state.

3. **Online calibration:** Implement Platt scaling or isotonic regression to produce calibrated probability outputs, enabling confidence-aware maintenance scheduling.

4. **Ensemble approach:** Combine SVM (highest clean accuracy) with k-NN (highest noise robustness) in a deployment-context-aware ensemble — selecting the algorithm based on the monitored sensor quality level.

5. **Transfer learning:** Investigate domain adaptation techniques to reduce the labelled data requirement when deploying to transformers not represented in the training database.

---

## Repository Structure

```
transformer-fault-diagnosis/
│
├── README.md                              ← This file
│
├── data/
│   ├── Base_de_Datos_Eng_5_2_2026_V1.csv ← Raw source database
│   └── Final_Transformer_Dataset_with_Duval.csv ← Cleaned master dataset
│
├── scripts/
│   ├── Data_Analysis.m
│   ├── Data_Preprocessing.m
│   ├── Baseline_Models.m
│   ├── SVM_GridSearch_Hyperparameters.m
│   ├── LearningCurves_and_Analysis.m
│   ├── Noise_Injection_Analysis.m
│   └── xAI_Analysis.m
│
├── pipeline_outputs/                      ← Data_Preprocessing.m
│   ├── X_train_scaled.csv
│   ├── X_train_smote.csv
│   ├── X_test_scaled.csv
│   └── scaler_params.csv
│
├── audit_outputs/                         ← Data_Analysis.m
├── baseline_outputs/                      ← Baseline_Models.m
├── svm_outputs/                           ← SVM_GridSearch_Hyperparameters.m
├── learning_curve_outputs/                ← LearningCurves_and_Analysis.m
├── noise_outputs/                         ← Noise_Injection_Analysis.m
└── xai_outputs/                           ← xAI_Analysis.m
```
---

## Academic Integrity

All code in this repository was written by Kgosietsile Ntsowe (ID: 11409134) as part of the EEEN30330 Individual Project. No third-party code was imported or reused. The SMOTE implementation, robust scaler, and permutation importance functions are original implementations consistent with the algorithms described in the cited literature.

All data used is open-source and publicly available. No proprietary transformer data was used.

**References :**  
- Chawla, N.V. et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *JAIR*, 16, 321–357.  
- Duval, M. & dePablo, A. (2001). Interpretation of gas-in-oil analysis using IEC publication 60599. *IEEE Electrical Insulation Magazine*, 17(2), 31–41.  
- IEC 60599:2022. *Mineral oil-filled electrical equipment in service — guidance on the interpretation of dissolved and free gases analysis.*  
- IEEE C57.104-2019. *IEEE Guide for the Interpretation of Gases Generated in Mineral Oil-Immersed Transformers.*
- Montero Jiménez, Juan José; Gómez-Ramírez, Gustavo Adolfo (2026), “Power transformer data for fault diagnosis”, Mendeley Data, V2, doi: 10.17632/98f4z3f8tx.2 .Raw Dataset
---

*Last updated: May 2026*
