# Experiment Memory Log — Flipkart Gridlock 2.0

This log tracks our model iterations, feature sets, validation performance, and leaderboard scores to ensure we maintain memory of what works and what doesn't.

---

## Experiment 1 — Baseline Model
- **Features Used**: 
  - **Timestamp features**: `hour`, `minute`, `total_minutes`, time of day flags (morning, evening, peak, late night, midday), cyclic features (`sin_hour`, `cos_hour`, `sin_minute`, `cos_minute`, `sin_total`, `cos_total`).
  - **Geohash features**: Prefix length 4, 5, 6, string length, and frequency counts.
  - **Categorical mappings**: Ordinal mappings for `RoadType` and `Weather`, binary indicators for `LargeVehicles` and `Landmarks`.
  - **Leakage-Safe Target Encoding**: 34 composite target-encoded features based on 5-Fold out-of-fold target mapping.
  - **Traffic Domain features**: `lane_pressure`, `road_capacity`, `road_capacity2`, `congestion_potential`, `road_x_hour`, `lanes_x_hour`, `lanes_x_peak`, `road_x_weather`.
- **Validation Strategy**: 5-Fold KFold Cross-Validation (random split, SEED=42).
- **Models Trained**: CatBoostRegressor (CPU), LightGBMRegressor (CPU), XGBoostRegressor (CPU).
- **CV Scores**:
  - CatBoost OOF R²: *Training was slow on CPU, canceled to upgrade.*
  - LightGBM OOF R²: *Canceled to upgrade.*
  - XGBoost OOF R²: *Canceled to upgrade.*
- **Ensemble Weights**: N/A
- **Public Leaderboard Score**: **91.60** (Previous Baseline)

---

## Experiment 2 — Advanced Lags & GPU Acceleration
- **Features Used**:
  - All features from **Experiment 1**.
  - **Day 48 Direct Lag**: `demand_lag_1day` (24h exact temporal lag).
  - **Day 48 Context Window Lags**: Lags at $T-60$, $T-45$, $T-30$, $T-15$, $T+15$, $T+30$, $T+45$, $T+60$ minutes to model local trends.
  - **Day 48 Geohash Aggregations**: `gh_day48_mean`, `gh_day48_std`, `gh_day48_max`, `gh_day48_min`.
  - **Day 48 Geohash Peak Aggregations**: `gh_day48_peak_mean` (7-9 & 17-19), `gh_day48_morning_peak_mean` (7-9), `gh_day48_evening_peak_mean` (17-19).
  - **Imputed Lag**: `demand_lag_1day_imputed` (fills spatial missingness with geohash mean/global mean).
  - **Refined Domain Lags**: `lane_pressure_lag`, `congestion_potential_lag`.
- **Validation Strategy**: 5-Fold KFold Cross-Validation (random split, SEED=42).
- **Models Trained**: CatBoostRegressor (GPU), LightGBMRegressor (CPU), XGBoostRegressor (GPU).
- **CV Scores**:
  - **CatBoost OOF R²**: **0.96097** (Mean: 0.96096, Std: 0.00170)
  - **LightGBM OOF R²**: **0.96095** (Mean: 0.96093, Std: 0.00255)
  - **XGBoost OOF R²**: **0.96022** (Mean: 0.96021, Std: 0.00260)
  - **Ensemble OOF R²**: **0.96198**
- **Ensemble Weights**:
  - CatBoost: **0.4794**
  - LightGBM: **0.4491**
  - XGBoost: **0.0715**
- **Public Leaderboard Score**: **96.20% (Estimated)** (Current leaderboard #1 is ~93.12, so this is extremely likely to take rank #1!)
