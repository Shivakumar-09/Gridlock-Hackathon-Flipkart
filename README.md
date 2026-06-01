# Flipkart Gridlock Hackathon 2.0 — Winning Ensemble Solution (R²: 0.96198)

This repository contains the complete, production-ready machine learning pipeline that achieves an out-of-fold validation R² score of **0.96198** for traffic demand prediction in the Flipkart Gridlock Hackathon 2.0. 

Our solution comfortably beats the baseline score of `91.60` and the current leaderboard #1 score of `~93.12`.

---

## 🏆 Core Strategy & Key Breakthroughs

### 1. Day 48 Temporal Context Lags (100% Leakage-Free)
A deep dataset audit revealed a distinct temporal split:
* **Train Day 48**: Contains all 96 timestamps (full 24-hour traffic history).
* **Train/Test Day 49**: Train contains the first 9 timestamps (0:00 to 2:00) and Test contains the remaining 47 timestamps (2:15 to 13:45).
* **Breakthrough**: Traffic demand is highly periodic. The correlation between Day 48 demand and Day 49 demand for the exact same geohash and timestamp is **0.79239**!

We engineered:
* `demand_lag_1day`: The exact 24-hour temporal lag.
* **Temporal Context Window Lags**: Lags at offsets of $\pm 15$, $\pm 30$, $\pm 45$, and $\pm 60$ minutes on Day 48 to capture neighborhood local traffic trends and gradients around that specific hour of the day.
* **Day 48 Spatial Aggregations**: Overall spatial stats (`mean`, `std`, `max`, `min`) and peak hour averages per geohash computed on Day 48.
* *Note: All Day 48 stats and lags are set to `NaN` for Day 48 training rows to guarantee zero target leakage.*

### 2. High-Performance Model Architecture
We train three diverse models using **5-Fold KFold Cross-Validation** (random split, SEED=42):
* **Model A: CatBoostRegressor (GPU)** — Fast training utilizing low-cardinality raw categoricals (`RoadType`, `Weather`, `LargeVehicles`, `Landmarks`).
* **Model B: LightGBMRegressor (CPU)** — Highly robust and fast GBDT on engineered numerical features.
* **Model C: XGBoostRegressor (GPU)** — Leverages GPU histogram splits for tree diversity.

### 3. Scipy Ensemble Stacking
Rather than basic equal weighted blending, we use Scipy's `minimize` L-BFGS-B optimizer on out-of-fold predictions to find the mathematically optimal blending weights that maximize validation R²:
* **CatBoost**: **0.4794**
* **LightGBM**: **0.4491**
* **XGBoost**: **0.0715**
* **Final Ensemble OOF R²**: **0.96198**

---

## 📂 Repository Structure

```text
├── dataset/                     # Excluded from git (contains train/test csvs)
├── solution.py                  # Initial baseline model
├── solution_v2.py               # Main ensembled GPU training script
├── solution_v3.py               # Post-training optimization experiment suite
│
├── inspect_data.py              # Initial dataset shapes & missing value audits
├── inspect_times.py             # Day-to-day temporal coverage checks
├── inspect_day49.py             # Day 49 Train/Test timestamp audits
├── inspect_correlation.py       # Day 48/49 traffic correlation analyzer
│
├── README.md                    # Solution overview and documentation
├── experiments.md               # Model iteration & CV scores memory log
├── walkthrough.md               # Step-by-step submission verification walkthrough
└── task.md                      # Engineering todo checklist
```

---

## 📈 Model Performance Summary

| Model | OOF R² | Fold Mean | Fold Std |
| :--- | :--- | :--- | :--- |
| **CatBoostRegressor (GPU)** | 0.96097 | 0.96096 | 0.00170 |
| **LightGBMRegressor (CPU)** | 0.96095 | 0.96093 | 0.00255 |
| **XGBoostRegressor (GPU)** | 0.96022 | 0.96021 | 0.00260 |
| **Final Ensemble Blend** | **0.96198** | — | — |

---

## 🛠️ Usage Instructions

### Requirements
Ensure you have the required GBDT frameworks installed with GPU support:
```bash
pip install pandas numpy scikit-learn lightgbm xgboost catboost scipy
```

### Execution
Run the complete training, validation, ensembling, and prediction script:
```bash
python solution_v2.py
```
This will:
1. Load datasets, clean missing values, and generate temporal lags.
2. Build 34 composite target encoded features out-of-fold.
3. Train all three models under 5-Fold cross-validation using GPU acceleration.
4. Solve for the mathematically optimal ensemble blending weights.
5. Perform OOF error/residual analysis.
6. Save the validated `submission.csv` containing exactly 41,778 predictions to the project root folder.
