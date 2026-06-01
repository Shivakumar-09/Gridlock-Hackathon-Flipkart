# =============================================================================
# FLIPKART GRIDLOCK HACKATHON 2.0 — ADVANCED SOLUTION
# Objective: Maximize R² on traffic demand prediction
# Target: Beat leaderboard #1 (~93.12)
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import os, gc, time
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

SEED = 42
np.random.seed(SEED)

DATA_DIR = r"c:\hackathon\flipkart\dataset"
OUT_DIR  = r"c:\hackathon\flipkart"

# =============================================================================
# PHASE 1 — DEEP DATASET AUDIT
# =============================================================================
print("=" * 70)
print("PHASE 1 — DEEP DATASET AUDIT")
print("=" * 70)

train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sub   = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

print(f"Train shape : {train.shape}")
print(f"Test  shape : {test.shape}")
print(f"Sub   shape : {sub.shape}")

print(f"\nDuplicates: train={train.duplicated().sum()}, test={test.duplicated().sum()}")
print(f"\nTarget stats:\n{train['demand'].describe()}")
print(f"Skewness={train['demand'].skew():.3f}  Kurtosis={train['demand'].kurtosis():.3f}")

train_hashes = set(train["geohash"].unique())
test_hashes  = set(test["geohash"].unique())
overlap = train_hashes & test_hashes
print(f"\nGeohash overlap: {len(overlap)}/{len(test_hashes)} ({len(overlap)/len(test_hashes)*100:.1f}%)")

# =============================================================================
# PHASE 2 — FEATURE ENGINEERING
# =============================================================================
print("\n" + "=" * 70)
print("PHASE 2 — FEATURE ENGINEERING")
print("=" * 70)

def parse_ts(ts):
    parts = str(ts).split(":")
    h = int(parts[0])
    m = int(parts[1]) if len(parts) > 1 else 0
    return h, m

def base_features(df):
    df = df.copy()

    # Timestamp
    df[["hour", "minute"]] = pd.DataFrame(
        df["timestamp"].apply(parse_ts).tolist(), index=df.index
    )
    df["total_minutes"] = df["hour"] * 60 + df["minute"]

    # Time flags
    df["peak_hour"]    = ((df["hour"].between(7, 9)) | (df["hour"].between(17, 19))).astype(int)
    df["morning_peak"] = df["hour"].between(7, 9).astype(int)
    df["evening_peak"] = df["hour"].between(17, 19).astype(int)
    df["late_night"]   = ((df["hour"] >= 23) | (df["hour"] <= 4)).astype(int)
    df["midday"]       = df["hour"].between(11, 14).astype(int)

    # Cyclic
    df["sin_hour"]   = np.sin(2 * np.pi * df["hour"]   / 24)
    df["cos_hour"]   = np.cos(2 * np.pi * df["hour"]   / 24)
    df["sin_minute"] = np.sin(2 * np.pi * df["minute"] / 60)
    df["cos_minute"] = np.cos(2 * np.pi * df["minute"] / 60)
    df["sin_total"]  = np.sin(2 * np.pi * df["total_minutes"] / 1440)
    df["cos_total"]  = np.cos(2 * np.pi * df["total_minutes"] / 1440)

    # Geohash prefixes
    df["gh_prefix4"] = df["geohash"].str[:4]
    df["gh_prefix5"] = df["geohash"].str[:5]
    df["gh_prefix6"] = df["geohash"].str[:6]
    df["geohash_len"] = df["geohash"].str.len()

    # Categoricals: fill & encode
    df["RoadType"]      = df["RoadType"].fillna("Unknown")
    df["Weather"]       = df["Weather"].fillna("Unknown")
    df["LargeVehicles"] = df["LargeVehicles"].fillna("Unknown")
    df["Landmarks"]     = df["Landmarks"].fillna("Unknown")

    road_map    = {"Highway": 4, "Street": 3, "Residential": 2, "Unknown": 1}
    weather_map = {"Sunny": 4, "Foggy": 3, "Rainy": 2, "Snowy": 1, "Unknown": 0}
    df["road_type_ord"]  = df["RoadType"].map(road_map).fillna(1).astype(int)
    df["weather_ord"]    = df["Weather"].map(weather_map).fillna(0).astype(int)
    df["large_veh_bin"]  = (df["LargeVehicles"] == "Allowed").astype(int)
    df["landmarks_bin"]  = (df["Landmarks"] == "Yes").astype(int)

    # Numeric: fill
    df["NumberofLanes"]  = df["NumberofLanes"].fillna(df["NumberofLanes"].median())
    df["Temperature"]    = df["Temperature"].fillna(df["Temperature"].median())
    df["temp_extreme"]   = ((df["Temperature"] > 35) | (df["Temperature"] < 0)).astype(int)
    df["temp_band"]      = pd.cut(
        df["Temperature"],
        bins=[-100, 0, 10, 20, 30, 100],
        labels=[0, 1, 2, 3, 4]
    ).astype(float).fillna(2)

    # Domain features
    df["road_capacity"]  = df["NumberofLanes"] * df["large_veh_bin"]
    df["road_capacity2"] = df["NumberofLanes"] * df["road_type_ord"]
    df["congestion_potential"] = (
        df["road_type_ord"] * 0.4 +
        df["NumberofLanes"] * 0.3 +
        df["landmarks_bin"] * 0.2 +
        df["peak_hour"]     * 0.1
    )
    df["road_x_hour"]    = df["road_type_ord"] * df["hour"]
    df["lanes_x_hour"]   = df["NumberofLanes"] * df["hour"]
    df["lanes_x_peak"]   = df["NumberofLanes"] * df["peak_hour"]
    df["road_x_weather"] = df["road_type_ord"] * df["weather_ord"]

    return df

print("Extracting base features...")
train = base_features(train)
test  = base_features(test)

# Geohash frequency
gh_freq  = train["geohash"].value_counts().to_dict()
gh4_freq = train["gh_prefix4"].value_counts().to_dict()
gh5_freq = train["gh_prefix5"].value_counts().to_dict()

for df in [train, test]:
    df["geohash_freq"]    = df["geohash"].map(gh_freq).fillna(0)
    df["gh_prefix4_freq"] = df["gh_prefix4"].map(gh4_freq).fillna(0)
    df["gh_prefix5_freq"] = df["gh_prefix5"].map(gh5_freq).fillna(0)

# =============================================================================
# ADVANCED TEMPORAL LAG & HISTORICAL FEATURES (DAY 48 AS SOURCE)
# =============================================================================
print("Creating Day 48 lag and historical features...")

# Pivot/Lookup from Day 48 train data
df_48 = train[train['day'] == 48].copy()

# Day 48 geohash-specific stats
gh_stats_48 = df_48.groupby('geohash')['demand'].agg(
    gh_day48_mean='mean',
    gh_day48_std='std',
    gh_day48_max='max',
    gh_day48_min='min'
).reset_index()

gh_peak_48 = df_48[df_48['hour'].isin([7,8,9,17,18,19])].groupby('geohash')['demand'].mean().rename('gh_day48_peak_mean').reset_index()
gh_mpeak_48 = df_48[df_48['hour'].isin([7,8,9])].groupby('geohash')['demand'].mean().rename('gh_day48_morning_peak_mean').reset_index()
gh_epeak_48 = df_48[df_48['hour'].isin([17,18,19])].groupby('geohash')['demand'].mean().rename('gh_day48_evening_peak_mean').reset_index()

# Merge overall statistics directly
train = pd.merge(train, gh_stats_48, on='geohash', how='left')
train = pd.merge(train, gh_peak_48, on='geohash', how='left')
train = pd.merge(train, gh_mpeak_48, on='geohash', how='left')
train = pd.merge(train, gh_epeak_48, on='geohash', how='left')

test = pd.merge(test, gh_stats_48, on='geohash', how='left')
test = pd.merge(test, gh_peak_48, on='geohash', how='left')
test = pd.merge(test, gh_mpeak_48, on='geohash', how='left')
test = pd.merge(test, gh_epeak_48, on='geohash', how='left')

# Set all stats features to NaN on day 48 to prevent leakage
stats_cols = ['gh_day48_mean', 'gh_day48_std', 'gh_day48_max', 'gh_day48_min', 
              'gh_day48_peak_mean', 'gh_day48_morning_peak_mean', 'gh_day48_evening_peak_mean']
train.loc[train['day'] == 48, stats_cols] = np.nan

# Merge direct lag demand and neighbor offset demand
df_48_ref = df_48[['geohash', 'total_minutes', 'demand']].copy()

# Add exact 24h lag
exact_lag = df_48_ref.rename(columns={'demand': 'demand_lag_1day'})
train = pd.merge(train, exact_lag, on=['geohash', 'total_minutes'], how='left')
test  = pd.merge(test, exact_lag, on=['geohash', 'total_minutes'], how='left')

# Add neighbor lags to build temporal context
offsets = [-60, -45, -30, -15, 15, 30, 45, 60]
for offset in offsets:
    df_48_ref_offset = df_48_ref.copy()
    df_48_ref_offset['total_minutes'] = df_48_ref_offset['total_minutes'] - offset
    col_name = f"demand_lag_1day_offset_{offset}" if offset > 0 else f"demand_lag_1day_offset_m{abs(offset)}"
    df_48_ref_offset = df_48_ref_offset.rename(columns={'demand': col_name})
    
    train = pd.merge(train, df_48_ref_offset, on=['geohash', 'total_minutes'], how='left')
    test  = pd.merge(test, df_48_ref_offset, on=['geohash', 'total_minutes'], how='left')

# Set all lag features to NaN on day 48 to prevent leakage
lag_cols = [c for c in train.columns if 'demand_lag_1day' in c]
train.loc[train['day'] == 48, lag_cols] = np.nan

# Imputed versions of the main 24h lag to handle spatial sparseness
train['demand_lag_1day_imputed'] = train['demand_lag_1day'].fillna(train['gh_day48_mean']).fillna(0.09394)
test['demand_lag_1day_imputed'] = test['demand_lag_1day'].fillna(test['gh_day48_mean']).fillna(0.09394)

print("Lag features done.")

# =============================================================================
# LEAKAGE-SAFE TARGET ENCODING (KFOLD OOF)
# =============================================================================
print("Building leakage-safe target encodings...")

TARGET   = "demand"
N_TE_FOLD = 5
kf_te    = KFold(n_splits=N_TE_FOLD, shuffle=True, random_state=SEED)

def kfold_te(train_df, test_df, group_cols, target=TARGET, n_folds=N_TE_FOLD, agg="mean"):
    if isinstance(group_cols, str):
        group_cols = [group_cols]

    key_name = "__".join(group_cols)
    col_name = f"te__{key_name}__{agg}"

    tr_key = train_df[group_cols].astype(str).agg("|".join, axis=1)
    te_key = test_df[group_cols].astype(str).agg("|".join, axis=1)

    tr_enc  = np.zeros(len(train_df))
    global_mean = train_df[target].mean()

    for _, (tr_idx, val_idx) in enumerate(
            KFold(n_splits=n_folds, shuffle=True, random_state=SEED).split(train_df)):

        tmp = train_df.iloc[tr_idx].copy()
        tmp["__key"] = tr_key.iloc[tr_idx].values
        fold_map = tmp.groupby("__key")[target].agg(agg)

        val_keys = tr_key.iloc[val_idx]
        tr_enc[val_idx] = val_keys.map(fold_map).fillna(global_mean).values

    tmp_all = train_df.copy()
    tmp_all["__key"] = tr_key.values
    global_map = tmp_all.groupby("__key")[target].agg(agg)

    te_enc = te_key.map(global_map).fillna(global_mean).values
    return tr_enc, te_enc, col_name

ENCODING_SPECS = [
    ("geohash",                               "mean"),
    ("geohash",                               "std"),
    ("geohash",                               "max"),
    ("geohash",                               "median"),
    ("gh_prefix4",                            "mean"),
    ("gh_prefix5",                            "mean"),
    ("gh_prefix6",                            "mean"),
    ("RoadType",                              "mean"),
    ("Weather",                               "mean"),
    ("NumberofLanes",                         "mean"),
    (["geohash", "hour"],                     "mean"),
    (["geohash", "hour"],                     "std"),
    (["geohash", "day"],                      "mean"),
    (["geohash", "Weather"],                  "mean"),
    (["geohash", "peak_hour"],                "mean"),
    (["geohash", "morning_peak"],             "mean"),
    (["geohash", "evening_peak"],             "mean"),
    (["geohash", "landmarks_bin"],            "mean"),
    (["geohash", "large_veh_bin"],            "mean"),
    (["geohash", "NumberofLanes"],            "mean"),
    (["gh_prefix4", "hour"],                  "mean"),
    (["gh_prefix4", "day"],                   "mean"),
    (["gh_prefix4", "Weather"],               "mean"),
    (["gh_prefix5", "hour"],                  "mean"),
    (["gh_prefix5", "day"],                   "mean"),
    (["gh_prefix6", "hour"],                  "mean"),
    (["RoadType", "hour"],                    "mean"),
    (["RoadType", "Weather"],                 "mean"),
    (["RoadType", "peak_hour"],               "mean"),
    (["Weather", "hour"],                     "mean"),
    (["Weather", "large_veh_bin"],            "mean"),
    (["road_type_ord", "hour"],               "mean"),
    (["geohash", "Weather", "hour"],          "mean"),
    (["gh_prefix4", "hour", "Weather"],       "mean"),
]

te_cols = []
for spec_group, spec_agg in ENCODING_SPECS:
    tr_enc, te_enc, cname = kfold_te(train, test, spec_group, agg=spec_agg)
    train[cname] = tr_enc
    test[cname]  = te_enc
    te_cols.append(cname)

print(f"{len(te_cols)} TE features created.")

# =============================================================================
# DOMAIN FEATURES (REFINED WITH LAGS)
# =============================================================================
gh_mean_col     = "te__geohash__mean"
gh_std_col      = "te__geohash__std"
gh_hour_col     = "te__geohash__hour__mean"
gh_weather_col  = "te__geohash__Weather__mean"

for df in [train, test]:
    df["lane_pressure"] = df[gh_mean_col] / (df["NumberofLanes"] + 1e-6)
    df["landmark_congestion"] = df["landmarks_bin"] * df[gh_hour_col] * (1 + df["peak_hour"])
    df["weather_sensitivity"] = (df[gh_weather_col] / (df[gh_mean_col] + 1e-9)).clip(0, 5)
    df["congestion_score"] = (
        df[gh_hour_col]          * 0.40 +
        df["lane_pressure"]      * 0.30 +
        df["landmark_congestion"]* 0.20 +
        df["congestion_potential"]* 0.10
    )
    df["gh_cv"]   = (df[gh_std_col] / (df[gh_mean_col] + 1e-9)).clip(0, 10)
    df["gh_rank"] = df[gh_mean_col].rank(pct=True)

    # Contextual capacity pressure using our Day 48 lags
    df["lane_pressure_lag"] = df["demand_lag_1day_imputed"] / (df["NumberofLanes"] + 1e-6)
    df["congestion_potential_lag"] = (
        df["road_type_ord"] * 0.3 +
        df["NumberofLanes"] * 0.2 +
        df["demand_lag_1day_imputed"] * 0.4 +
        df["peak_hour"] * 0.1
    )

print("Domain features from TE & Lags done.")

# =============================================================================
# FEATURE LISTS
# =============================================================================
BASE_NUM = [
    "hour", "minute", "total_minutes",
    "peak_hour", "morning_peak", "evening_peak", "late_night", "midday",
    "sin_hour", "cos_hour", "sin_minute", "cos_minute", "sin_total", "cos_total",
    "day",
    "NumberofLanes", "large_veh_bin", "landmarks_bin",
    "road_type_ord", "weather_ord",
    "Temperature", "temp_extreme", "temp_band",
    "road_capacity", "road_capacity2", "congestion_potential",
    "geohash_len", "geohash_freq", "gh_prefix4_freq", "gh_prefix5_freq",
    "lane_pressure", "landmark_congestion", "weather_sensitivity",
    "congestion_score", "gh_cv", "gh_rank",
    "road_x_hour", "lanes_x_hour", "lanes_x_peak", "road_x_weather",
    "lane_pressure_lag", "congestion_potential_lag"
]

LAG_NUM = [
    "demand_lag_1day", "demand_lag_1day_imputed",
    "gh_day48_mean", "gh_day48_std", "gh_day48_max", "gh_day48_min",
    "gh_day48_peak_mean", "gh_day48_morning_peak_mean", "gh_day48_evening_peak_mean"
] + [f"demand_lag_1day_offset_{o}" if o > 0 else f"demand_lag_1day_offset_m{abs(o)}" for o in offsets]

CAT_COLS = ["RoadType", "Weather", "LargeVehicles", "Landmarks"]

ALL_NUM = BASE_NUM + LAG_NUM + te_cols
ALL_CB  = ALL_NUM + CAT_COLS

print(f"\nNumeric features : {len(ALL_NUM)}")
print(f"CatBoost features: {len(ALL_CB)}")

# Prepare arrays
X_train_num = train[ALL_NUM].copy().fillna(-999)
X_test_num  = test[ALL_NUM].copy().fillna(-999)
X_train_cb  = train[ALL_CB].copy().fillna(-999)
X_test_cb   = test[ALL_CB].copy().fillna(-999)
y_train     = train[TARGET].copy()

cat_idx_cb = [ALL_CB.index(c) for c in CAT_COLS]

# =============================================================================
# PHASE 3 & 4 — MODEL TRAINING + VALIDATION (5-Fold KFold)
# =============================================================================
print("\n" + "=" * 70)
print("PHASE 3 & 4 — MODEL TRAINING + VALIDATION (5-Fold KFold on GPU)")
print("=" * 70)

N_FOLDS = 5
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL A — CatBoost (GPU Accelerated)
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- MODEL A: CatBoost ---")
cb_oof  = np.zeros(len(train))
cb_test = np.zeros(len(test))
cb_scores = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_cb, y_train)):
    t0 = time.time()
    print(f"  Fold {fold+1}/{N_FOLDS}...", end=" ", flush=True)
    X_tr, X_val = X_train_cb.iloc[tr_idx], X_train_cb.iloc[val_idx]
    y_tr, y_val = y_train.iloc[tr_idx],    y_train.iloc[val_idx]

    model = CatBoostRegressor(
        iterations=2500,
        learning_rate=0.035,
        depth=6,
        l2_leaf_reg=3.0,
        min_data_in_leaf=15,
        random_strength=0.5,
        border_count=254,
        cat_features=cat_idx_cb,
        eval_metric="RMSE",
        loss_function="RMSE",
        task_type="GPU",            # Leverage RTX 2050 GPU!
        random_seed=SEED + fold,
        verbose=0,
        early_stopping_rounds=150,
        use_best_model=True,
    )
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)

    val_pred = model.predict(X_val)
    score    = r2_score(y_val, val_pred)
    cb_scores.append(score)
    cb_oof[val_idx]  = val_pred
    cb_test         += model.predict(X_test_cb) / N_FOLDS
    print(f"R²={score:.5f}  ({time.time()-t0:.1f}s)")

cb_oof_r2 = r2_score(y_train, cb_oof)
print(f"\nCatBoost OOF R²: {cb_oof_r2:.5f}  Mean={np.mean(cb_scores):.5f}  Std={np.std(cb_scores):.5f}")

# ─────────────────────────────────────────────────────────────────────────────
# MODEL B — LightGBM (CPU)
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- MODEL B: LightGBM ---")
lgb_oof  = np.zeros(len(train))
lgb_test = np.zeros(len(test))
lgb_scores = []
lgb_models = []

LGB_PARAMS = {
    "objective":         "regression",
    "metric":            "rmse",
    "num_leaves":        127,
    "max_depth":         -1,
    "learning_rate":     0.025,
    "feature_fraction":  0.70,
    "bagging_fraction":  0.80,
    "bagging_freq":      5,
    "min_child_samples": 20,
    "lambda_l1":         0.10,
    "lambda_l2":         0.30,
    "n_jobs":            -1,
    "seed":              SEED,
    "verbose":           -1,
}

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_num, y_train)):
    t0 = time.time()
    print(f"  Fold {fold+1}/{N_FOLDS}...", end=" ", flush=True)
    X_tr, X_val = X_train_num.iloc[tr_idx], X_train_num.iloc[val_idx]
    y_tr, y_val = y_train.iloc[tr_idx],     y_train.iloc[val_idx]

    dtr = lgb.Dataset(X_tr, label=y_tr)
    dvl = lgb.Dataset(X_val, label=y_val, reference=dtr)

    model = lgb.train(
        LGB_PARAMS,
        dtr,
        num_boost_round=5000,
        valid_sets=[dvl],
        callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(-1)],
    )
    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    score    = r2_score(y_val, val_pred)
    lgb_scores.append(score)
    lgb_oof[val_idx]  = val_pred
    lgb_test          += model.predict(X_test_num, num_iteration=model.best_iteration) / N_FOLDS
    lgb_models.append(model)
    print(f"R²={score:.5f}  ({time.time()-t0:.1f}s)")

lgb_oof_r2 = r2_score(y_train, lgb_oof)
print(f"\nLightGBM OOF R²: {lgb_oof_r2:.5f}  Mean={np.mean(lgb_scores):.5f}  Std={np.std(lgb_scores):.5f}")

# ─────────────────────────────────────────────────────────────────────────────
# MODEL C — XGBoost (GPU Accelerated)
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- MODEL C: XGBoost ---")
xgb_oof  = np.zeros(len(train))
xgb_test = np.zeros(len(test))
xgb_scores = []

XGB_PARAMS = {
    "objective":         "reg:squarederror",
    "eval_metric":       "rmse",
    "max_depth":         6,
    "learning_rate":     0.035,
    "subsample":         0.80,
    "colsample_bytree":  0.70,
    "colsample_bylevel": 0.70,
    "min_child_weight":  15,
    "reg_alpha":         0.10,
    "reg_lambda":        0.30,
    "tree_method":       "hist",
    "device":            "cuda",     # Leverage RTX 2050 GPU!
    "random_state":      SEED,
    "n_jobs":            -1,
    "verbosity":         0,
}

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_num, y_train)):
    t0 = time.time()
    print(f"  Fold {fold+1}/{N_FOLDS}...", end=" ", flush=True)
    X_tr, X_val = X_train_num.iloc[tr_idx], X_train_num.iloc[val_idx]
    y_tr, y_val = y_train.iloc[tr_idx],     y_train.iloc[val_idx]

    dtr = xgb.DMatrix(X_tr, label=y_tr)
    dvl = xgb.DMatrix(X_val, label=y_val)

    model = xgb.train(
        XGB_PARAMS,
        dtr,
        num_boost_round=5000,
        evals=[(dvl, "val")],
        early_stopping_rounds=200,
        verbose_eval=False,
    )
    val_pred = model.predict(dvl)
    score    = r2_score(y_val, val_pred)
    xgb_scores.append(score)
    xgb_oof[val_idx]  = val_pred
    xgb_test          += model.predict(xgb.DMatrix(X_test_num)) / N_FOLDS
    print(f"R²={score:.5f}  ({time.time()-t0:.1f}s)")

xgb_oof_r2 = r2_score(y_train, xgb_oof)
print(f"\nXGBoost OOF R²: {xgb_oof_r2:.5f}  Mean={np.mean(xgb_scores):.5f}  Std={np.std(xgb_scores):.5f}")

# =============================================================================
# PHASE 5 — ENSEMBLE OPTIMIZATION
# =============================================================================
print("\n" + "=" * 70)
print("PHASE 5 — ENSEMBLE OPTIMIZATION")
print("=" * 70)

oof_mat  = np.column_stack([cb_oof, lgb_oof, xgb_oof])
test_mat = np.column_stack([cb_test, lgb_test, xgb_test])

# Grid search
step = 0.05
grid_results = []
for w1 in np.arange(0, 1 + 1e-9, step):
    for w2 in np.arange(0, 1 - w1 + 1e-9, step):
        w3 = 1 - w1 - w2
        if w3 < -1e-9:
            continue
        w3 = max(w3, 0)
        blend = w1 * cb_oof + w2 * lgb_oof + w3 * xgb_oof
        grid_results.append((r2_score(y_train, blend), w1, w2, w3))

grid_results.sort(reverse=True)
bscore, bw1, bw2, bw3 = grid_results[0]
print(f"Grid best: CB={bw1:.2f} LGB={bw2:.2f} XGB={bw3:.2f} R²={bscore:.5f}")

# Fine-tune with scipy
def neg_r2(w):
    wn = np.array(w); wn = wn / wn.sum()
    return -r2_score(y_train, oof_mat @ wn)

res = minimize(neg_r2, x0=[bw1, bw2, bw3], method="L-BFGS-B",
               bounds=[(0,1)]*3, options={"maxiter":2000,"ftol":1e-14})
opt_w = np.array(res.x); opt_w = opt_w / opt_w.sum()
opt_r2 = r2_score(y_train, oof_mat @ opt_w)
print(f"Opt  best: CB={opt_w[0]:.4f} LGB={opt_w[1]:.4f} XGB={opt_w[2]:.4f} R²={opt_r2:.5f}")

final_pred = np.clip(test_mat @ opt_w, 0.0, 1.0)

# =============================================================================
# PHASE 6 — ERROR ANALYSIS
# =============================================================================
print("\n" + "=" * 70)
print("PHASE 6 — ERROR ANALYSIS")
print("=" * 70)

blend_oof = np.clip(oof_mat @ opt_w, 0.0, 1.0)
train["oof_pred"] = blend_oof
train["residual"] = y_train - blend_oof
train["abs_err"]  = train["residual"].abs()

print("\nWorst geohashes (top 10 by MAE):")
worst_gh = (train.groupby("geohash")
            .agg(mae=("abs_err","mean"), n=("abs_err","count"))
            .sort_values("mae", ascending=False).head(10))
print(worst_gh)

print("\nHour-wise MAE:")
print(train.groupby("hour")["abs_err"].mean().round(5))

print(f"\nRows with |error|>0.10: {(train['abs_err']>0.10).sum()}")
print(f"Underprediction (>0.05): {(train['residual']>0.05).sum()}")
print(f"Overprediction (<-0.05): {(train['residual']<-0.05).sum()}")

# =============================================================================
# PHASE 7 — FINAL SUBMISSION
# =============================================================================
print("\n" + "=" * 70)
print("PHASE 7 — FINAL SUBMISSION")
print("=" * 70)

assert len(final_pred) == len(test), f"Length mismatch! {len(final_pred)} vs {len(test)}"
assert len(final_pred) == 41778, f"Row count wrong: {len(final_pred)}"

submission = pd.DataFrame({"Index": test["Index"].astype(int), "demand": final_pred})
assert submission["demand"].isnull().sum() == 0, "NaN in predictions!"

sub_path = os.path.join(OUT_DIR, "submission.csv")
submission.to_csv(sub_path, index=False)
print(f"\nSaved: {sub_path}")
print(f"Rows  : {len(submission)}")
print(f"Range : [{final_pred.min():.6f}, {final_pred.max():.6f}]")
print(f"Mean  : {final_pred.mean():.6f}")
print(submission.head())

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"""
Model             OOF R²       Fold Mean     Fold Std
─────────────────────────────────────────────────────
CatBoost        {cb_oof_r2:.5f}      {np.mean(cb_scores):.5f}       {np.std(cb_scores):.5f}
LightGBM        {lgb_oof_r2:.5f}      {np.mean(lgb_scores):.5f}       {np.std(lgb_scores):.5f}
XGBoost         {xgb_oof_r2:.5f}      {np.mean(xgb_scores):.5f}       {np.std(xgb_scores):.5f}

Ensemble (opt)  {opt_r2:.5f}
Weights: CB={opt_w[0]:.4f}  LGB={opt_w[1]:.4f}  XGB={opt_w[2]:.4f}
""")

# Feature importance
fi = pd.DataFrame({
    "feature":    ALL_NUM,
    "importance": lgb_models[-1].feature_importance(importance_type="gain"),
}).sort_values("importance", ascending=False)
print("Top 25 LightGBM features by gain:")
print(fi.head(25).to_string(index=False))
