# 🚦 Flipkart Gridlock Hackathon 2.0

## Traffic Demand Prediction using Ensemble Machine Learning

### Public Leaderboard Score: **91.19290**

### Cross-Validation (OOF) R² Score: **0.96198**

---

## 📌 Overview

This repository contains my complete end-to-end machine learning solution developed for **Flipkart Gridlock Hackathon 2.0**.

The objective of the competition was to predict traffic demand using spatial, temporal, environmental, and road network information. The solution combines advanced feature engineering, temporal traffic pattern analysis, and ensemble learning to build a robust demand forecasting pipeline.

The final solution achieved:

* **Public Leaderboard Score:** 91.19290
* **Out-of-Fold Validation R²:** 0.96198
* **Models:** CatBoost, LightGBM, XGBoost
* **Ensemble Optimization:** Scipy L-BFGS-B

---

# 🏆 Key Highlights

✅ Public Leaderboard Score: **91.19290**

✅ Validation R² Score: **0.96198**

✅ CatBoost + LightGBM + XGBoost Ensemble

✅ Temporal Feature Engineering

✅ Spatial Aggregation Features

✅ GPU Accelerated Training

✅ Reproducible Submission Pipeline

✅ HackerEarth Submission Ready

---

# 📊 Problem Statement

The challenge focuses on forecasting traffic demand across multiple geographic locations and timestamps.

The prediction depends on:

* 📍 Geographical Location (`geohash`)
* ⏰ Time Information (`timestamp`)
* 🛣️ Road Infrastructure (`RoadType`, `NumberofLanes`)
* 🚚 Vehicle Distribution (`LargeVehicles`)
* 🌦️ Environmental Factors (`Weather`, `Temperature`)
* 🏙️ Nearby Landmarks (`Landmarks`)

Accurate demand forecasting can help improve traffic management, transportation planning, and urban mobility systems.

---

# 🔍 Exploratory Data Analysis

A detailed analysis of the dataset revealed strong temporal patterns.

### Major Observation

Traffic demand exhibits daily periodic behavior.

Demand values from Day 48 showed strong predictive power for corresponding locations and times on Day 49.

### Observed Correlation

0.79239

This insight formed the foundation of the feature engineering strategy.

---

# ⚙️ Feature Engineering

## Temporal Features

The following lag-based features were engineered:

* demand_lag_1day
* lag_15min
* lag_30min
* lag_45min
* lag_60min

These features capture local traffic trends surrounding a particular timestamp.

---

## Spatial Aggregations

Per-geohash statistics:

* Mean Demand
* Standard Deviation
* Maximum Demand
* Minimum Demand
* Peak Hour Average

These features help model recurring traffic behavior at specific locations.

---

# 🤖 Model Architecture

Three complementary gradient boosting models were trained using 5-Fold Cross Validation.

## 🚀 CatBoostRegressor

Optimized for categorical traffic features:

* RoadType
* Weather
* LargeVehicles
* Landmarks

Validation R²:

0.96097

---

## ⚡ LightGBMRegressor

Optimized for engineered numerical features:

* Temporal lags
* Aggregations
* Interaction features

Validation R²:

0.96095

---

## 🌳 XGBoostRegressor

Used to increase ensemble diversity and improve generalization.

Validation R²:

0.96022

---

# 🎯 Ensemble Optimization

Instead of simple averaging, Scipy's L-BFGS-B optimizer was used to determine optimal blending weights.

## Final Ensemble Weights

| Model    | Weight |
| -------- | ------ |
| CatBoost | 0.4794 |
| LightGBM | 0.4491 |
| XGBoost  | 0.0715 |

---

# 📈 Model Performance

| Model              | OOF R²      |
| ------------------ | ----------- |
| CatBoost           | 0.96097     |
| LightGBM           | 0.96095     |
| XGBoost            | 0.96022     |
| **Final Ensemble** | **0.96198** |

---

# 📂 Repository Structure

```text
Gridlock-Hackathon-Flipkart/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── solution.py
├── solution_v2.py
│
├── inspect_data.py
├── inspect_times.py
├── inspect_day49.py
├── inspect_correlation.py
│
├── experiments.md
├── walkthrough.md
│
└── submission.csv
```

---

# 🛠️ Installation

Install required packages:

```bash
pip install -r requirements.txt
```

---

# ▶️ Reproducing the Results

Run:

```bash
python solution_v2.py
```

The pipeline performs:

1. Data Loading
2. Feature Engineering
3. Cross Validation
4. Model Training
5. Ensemble Optimization
6. Submission Generation

---

# 📤 Output

The final output file:

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

# 📋 Competition Results

| Metric                   | Score    |
| ------------------------ | -------- |
| Public Leaderboard Score | 91.19290 |
| OOF Validation R²        | 0.96198  |

### Note

The public leaderboard score is evaluated on a hidden test set and therefore differs from the validation score obtained during model training.

---

# 👨‍💻 Author

**Shiva Kumar**

Built for Flipkart Gridlock Hackathon 2.0

Areas of Focus:

* Machine Learning
* Traffic Demand Forecasting
* Ensemble Learning
* Feature Engineering
* Real-World Predictive Analytics

---

## ⭐ Acknowledgements

Special thanks to Flipkart and HackerEarth for organizing Gridlock Hackathon 2.0 and providing an opportunity to work on a real-world traffic forecasting challenge.
