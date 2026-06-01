# 🚦 Flipkart Gridlock Hackathon 2.0

## High-Performance Traffic Demand Prediction using Ensemble Learning

### Final Validation R² Score: **0.96198**

This repository contains our complete end-to-end machine learning solution built for **Flipkart Gridlock Hackathon 2.0**, focused on large-scale **traffic demand forecasting** using spatial, temporal, and contextual traffic signals.

The final ensemble achieved an **Out-of-Fold Validation R² Score of 0.96198**, making it a highly competitive leaderboard solution optimized for real-world traffic prediction.

---

# 🏆 Project Highlights

* ✅ Final Validation R²: **0.96198**
* ✅ Built using **CatBoost + LightGBM + XGBoost Ensemble**
* ✅ Optimized with **GPU-accelerated training**
* ✅ Leakage-safe temporal feature engineering
* ✅ Advanced ensemble blending with **Scipy optimization**
* ✅ End-to-end automated pipeline for training, validation, and submission generation

---

# Problem Statement

The challenge focuses on predicting **traffic demand** across different locations and timestamps using historical traffic patterns and road metadata.

Each prediction is influenced by:

* 📍 Spatial location (`geohash`)
* ⏰ Time of day (`timestamp`)
* 🛣️ Road characteristics (`RoadType`, `NumberofLanes`)
* 🚚 Vehicle composition (`LargeVehicles`)
* 🌦️ Environmental conditions (`Weather`, `Temperature`)
* 🏙️ Urban context (`Landmarks`)

---

# Core Approach & Key Insights

## 1. Temporal Traffic Pattern Discovery

A detailed exploratory analysis revealed a strong daily repetition pattern in traffic movement.

### Key finding:

Traffic demand from **Day 48** showed a high correlation with traffic demand on **Day 49** for the same geohash and timestamp.

### Observed correlation:

```text
0.79239
```

This became the foundation of our modeling strategy.

---

## 2. Leakage-Free Temporal Feature Engineering

To capture temporal dependencies without leaking target information, we engineered:

### Temporal Lag Features

* `demand_lag_1day`
* `lag_15min`
* `lag_30min`
* `lag_45min`
* `lag_60min`

These capture short-term traffic behavior around the same time window.

---

### Spatial Aggregations

Per-geohash aggregated statistics:

* mean demand
* standard deviation
* min demand
* max demand
* peak-hour average

These features help model recurring traffic behavior per location.

---

## 3. Multi-Model Ensemble Architecture

To maximize predictive performance, three complementary gradient boosting models were trained under **5-Fold Cross Validation**.

---

### 🚀 CatBoostRegressor (GPU)

Primary model for handling structured categorical traffic features efficiently.

Used for:

* `RoadType`
* `Weather`
* `LargeVehicles`
* `Landmarks`

---

### ⚡ LightGBMRegressor

Fast and highly effective on engineered numerical traffic features.

Strong performance on:

* lag-based temporal signals
* aggregation statistics
* interaction features

---

### 🌳 XGBoostRegressor (GPU)

Used to increase model diversity and strengthen ensemble generalization.

---

# Ensemble Optimization

Instead of equal averaging, we used **Scipy’s optimization (`L-BFGS-B`)** to find the optimal blending weights by maximizing Out-of-Fold R².

## Final Blend Weights

| Model    |     Weight |
| -------- | ---------: |
| CatBoost | **0.4794** |
| LightGBM | **0.4491** |
| XGBoost  | **0.0715** |

---

# 📈 Model Performance

| Model              | OOF R² Score |
| ------------------ | -----------: |
| CatBoostRegressor  |  **0.96097** |
| LightGBMRegressor  |  **0.96095** |
| XGBoostRegressor   |  **0.96022** |
| **Final Ensemble** |  **0.96198** |

---

# 📂 Project Structure

```text
Gridlock-Hackathon-Flipkart/
│
├── dataset/                     # Local dataset folder (not pushed to Git)
│
├── solution.py                  # Initial baseline model
├── solution_v2.py              # Final ensemble training pipeline
├── solution_v3.py              # Post-training optimization experiments
│
├── inspect_data.py             # Dataset audit & profiling
├── inspect_times.py            # Temporal pattern analysis
├── inspect_day49.py            # Day-wise timestamp inspection
├── inspect_correlation.py      # Traffic correlation analysis
│
├── experiments.md              # Experiment history & score tracking
├── walkthrough.md              # Submission workflow & project notes
├── task.md                     # Project task checklist
│
└── README.md
```

---

# Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* CatBoost
* LightGBM
* XGBoost
* SciPy

---

# Installation

Install required dependencies:

```bash
pip install pandas numpy scikit-learn lightgbm xgboost catboost scipy
```

---

# Run the Pipeline

Execute:

```bash
python solution_v2.py
```

This pipeline will:

* load the training and test datasets
* perform feature engineering
* train all models using cross-validation
* optimize ensemble blending weights
* evaluate validation performance
* generate `submission.csv`

---

# Output

The pipeline generates:

```text
submission.csv
```

Submission format:

```csv
Index,demand
```

Contains:

```text
41,778 predictions
```

ready for upload to HackerEarth.

---

# Final Result

## Best Validation Score

```text
R² = 0.96198
```

This project demonstrates how **temporal feature engineering**, **domain-specific traffic insights**, and **ensemble learning** can be combined to build a highly accurate traffic demand forecasting system.

---

# Author

### Shiva Kumar

Built for **Flipkart Gridlock Hackathon 2.0** with a focus on:

* Machine Learning
* Traffic Forecasting
* Ensemble Modeling
* Real-world Predictive Systems
