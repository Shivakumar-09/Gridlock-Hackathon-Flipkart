# =============================================================================
# FLIPKART GRIDLOCK HACKATHON 2.0 — POST-TRAINING OPTIMIZATION & STACKING
# Objective: Maximize R² on traffic demand prediction by running 4 advanced experiments
# Benchmark: 0.96198
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
# PHASE 1 — LOAD DATASET
# =============================================================================
print("=" * 70)
print("PHASE 1 — LOAD DATASET")
print("=" * 70)

train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sub   = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

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
    df[["hour", "minute"]] = pd.DataFrame(df["timestamp"].apply(parse_ts).tolist(), index=df.index)
    df["total_minutes"] = df["hour"] * 60 + df["minute"]
    df["peak_hour"]    = ((df["hour"].between(7, 9)) | (df["hour"].between(17, 19))).astype(int)
    df["morning_peak"] = df["hour"].between(7, 9).astype(int)
    df["evening_peak"] = df["hour"].between(17, 19).astype(int)
    df["late_night"]   = ((df["hour"] >= 23) | (df["hour"] <= 4)).astype(int)
    df["midday"]       = df["hour"].between(11, 14).astype(int)

    df["sin_hour"]   = np.sin(2 * np.pi * df["hour"]   / 24)
    df["cos_hour"]   = np.cos(2 * np.pi * df["hour"]   / 24)
    df["sin_minute"] = np.sin(2 * np.pi * df["minute"] / 60)
    df["cos_minute"] = np.cos(2 * np.pi * df["minute"] / 60)
    df["sin_total"]  = np.sin(2 * np.pi * df["total_minutes"] / 1440)
    df["cos_total"]  = np.cos(2 * np.pi * df["total_minutes"] / 1440)

    df["gh_prefix4"] = df["geohash"].str[:4]
    df["gh_prefix5"] = df["geohash"].str[:5]
    df["gh_prefix6"] = df["geohash"].str[:6]
    df["geohash_len"] = df["geohash"].str.len()

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

    df["NumberofLanes"]  = df["NumberofLanes"].fillna(df["NumberofLanes"].median())
    df["Temperature"]    = df["Temperature"].fillna(df["Temperature"].median())
    df["temp_extreme"]   = ((df["Temperature"] > 35) | (df["Temperature"] < 0)).astype(int)
    df["temp_band"]      = pd.cut(df["Temperature"], bins=[-100, 0, 10, 20, 30, 100], labels=[0, 1, 2, 3, 4]).astype(float).fillna(2)

    df["road_capacity"]  = df["NumberofLanes"] * df["large_veh_bin"]
    df["road_capacity2"] = df["NumberofLanes"] * df["road_type_ord"]
    df["congestion_potential"] = (df["road_type_ord"] * 0.4 + df["NumberofLanes"] * 0.3 + df["landmarks_bin"] * 0.2 + df["peak_hour"] * 0.1)
    df["road_x_hour"]    = df["road_type_ord"] * df["hour"]
    df["lanes_x_hour"]   = df["NumberofLanes"] * df["hour"]
    df["lanes_x_peak"]   = df["NumberofLanes"] * df["peak_hour"]
    df["road_x_weather"] = df["road_type_ord"] * df["weather_ord"]
    return df

print("Computing base features...")
train = base_features(train)
test  = base_features(test)

gh_freq  = train["geohash"].value_counts().to_dict()
gh4_freq = train["gh_prefix4"].value_counts().to_dict()
gh5_freq = train["gh_prefix5"].value_counts().to_dict()

for df in [train, test]:
    df["geohash_freq"]    = df["geohash"].map(gh_freq).fillna(0)
    df["gh_prefix4_freq"] = df["gh_prefix4"].map(gh4_freq).fillna(0)
    df["gh_prefix5_freq"] = df["gh_prefix5"].map(gh5_freq).fillna(0)

# Day 48 Lags & Stats
print("Merging Day 48 temporal context lags...")
df_48 = train[train['day'] == 48].copy()

gh_stats_48 = df_48.groupby('geohash')['demand'].agg(
    gh_day48_mean='mean', gh_day48_std='std', gh_day48_max='max', gh_day48_min='min'
).reset_index()

gh_peak_48 = df_48[df_48['hour'].isin([7,8,9,17,18,19])].groupby('geohash')['demand'].mean().rename('gh_day48_peak_mean').reset_index()
gh_mpeak_48 = df_48[df_48['hour'].isin([7,8,9])].groupby('geohash')['demand'].mean().rename('gh_day48_morning_peak_mean').reset_index()
gh_epeak_48 = df_48[df_48['hour'].isin([17,18,19])].groupby('geohash')['demand'].mean().rename('gh_day48_evening_peak_mean').reset_index()

train = pd.merge(train, gh_stats_48, on='geohash', how='left')
train = pd.merge(train, gh_peak_48, on='geohash', how='left')
train = pd.merge(train, gh_mpeak_48, on='geohash', how='left')
train = pd.merge(train, gh_epeak_48, on='geohash', how='left')

test = pd.merge(test, gh_stats_48, on='geohash', how='left')
test = pd.merge(test, gh_peak_48, on='geohash', how='left')
test = pd.merge(test, gh_mpeak_48, on='geohash', how='left')
test = pd.merge(test, gh_epeak_48, on='geohash', how='left')

stats_cols = ['gh_day48_mean', 'gh_day48_std', 'gh_day48_max', 'gh_day48_min', 
              'gh_day48_peak_mean', 'gh_day48_morning_peak_mean', 'gh_day48_evening_peak_mean']
train.loc[train['day'] == 48, stats_cols] = np.nan

df_48_ref = df_48[['geohash', 'total_minutes', 'demand']].copy()
exact_lag = df_48_ref.rename(columns={'demand': 'demand_lag_1day'})
train = pd.merge(train, exact_lag, on=['geohash', 'total_minutes'], how='left')
test  = pd.merge(test, exact_lag, on=['geohash', 'total_minutes'], how='left')

offsets = [-60, -45, -30, -15, 15, 30, 45, 60]
for offset in offsets:
    df_48_ref_offset = df_48_ref.copy()
    df_48_ref_offset['total_minutes'] = df_48_ref_offset['total_minutes'] - offset
    col_name = f"demand_lag_1day_offset_{offset}" if offset > 0 else f"demand_lag_1day_offset_m{abs(offset)}"
    df_48_ref_offset = df_48_ref_offset.rename(columns={'demand': col_name})
    train = pd.merge(train, df_48_ref_offset, on=['geohash', 'total_minutes'], how='left')
    test  = pd.merge(test, df_48_ref_offset, on=['geohash', 'total_minutes'], how='left')

lag_cols = [c for c in train.columns if 'demand_lag_1day' in c]
train.loc[train['day'] == 48, lag_cols] = np.nan

train['demand_lag_1day_imputed'] = train['demand_lag_1day'].fillna(train['gh_day48_mean']).fillna(0.09394)
test['demand_lag_1day_imputed'] = test['demand_lag_1day'].fillna(test['gh_day48_mean']).fillna(0.09394)

# =============================================================================
# EXPERIMENT B — ADDING STRONGER TRIPLE-INTERACT TARGET ENCODINGS
# =============================================================================
print("Computing advanced out-of-fold target encodings...")
TARGET   = "demand"
N_TE_FOLD = 5

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

# Baseline target encodings
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

# Triple interact encodings (Experiment B)
EXPERIMENT_B_SPECS = [
    (["geohash", "day", "hour"],              "mean"),
    (["geohash", "RoadType"],                 "mean")
]

te_cols = []
for spec_group, spec_agg in ENCODING_SPECS:
    tr_enc, te_enc, cname = kfold_te(train, test, spec_group, agg=spec_agg)
    train[cname] = tr_enc
    test[cname]  = te_enc
    te_cols.append(cname)

exp_b_cols = []
for spec_group, spec_agg in EXPERIMENT_B_SPECS:
    tr_enc, te_enc, cname = kfold_te(train, test, spec_group, agg=spec_agg)
    train[cname] = tr_enc
    test[cname]  = te_enc
    exp_b_cols.append(cname)

# Refined domain features
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

    df["lane_pressure_lag"] = df["demand_lag_1day_imputed"] / (df["NumberofLanes"] + 1e-6)
    df["congestion_potential_lag"] = (
        df["road_type_ord"] * 0.3 +
        df["NumberofLanes"] * 0.2 +
        df["demand_lag_1day_imputed"] * 0.4 +
        df["peak_hour"] * 0.1
    )

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

# Define feature sets
ALL_NUM_BASE = BASE_NUM + LAG_NUM + te_cols
ALL_NUM_EXPB = BASE_NUM + LAG_NUM + te_cols + exp_b_cols

ALL_CB_BASE  = ALL_NUM_BASE + CAT_COLS
ALL_CB_EXPB  = ALL_NUM_EXPB + CAT_COLS

y_train = train[TARGET].copy()

# =============================================================================
# MODEL TRAINING FUNCTION (GPU ACCELERATED)
# =============================================================================
N_FOLDS = 5
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

def train_pipeline(num_features, cb_features, feature_name_log="Base Features"):
    print(f"\n--- Running 5-Fold CV Training ({feature_name_log}) ---")
    
    # Prepare arrays
    X_train_num = train[num_features].copy().fillna(-999)
    X_test_num  = test[num_features].copy().fillna(-999)
    X_train_cb  = train[cb_features].copy().fillna(-999)
    X_test_cb   = test[cb_features].copy().fillna(-999)
    
    cat_idx_cb = [cb_features.index(c) for c in CAT_COLS]
    
    # 1. CatBoost
    print("  Training CatBoost...")
    cb_oof  = np.zeros(len(train))
    cb_test = np.zeros(len(test))
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_cb, y_train)):
        model = CatBoostRegressor(
            iterations=2500, learning_rate=0.035, depth=6, l2_leaf_reg=3.0,
            min_data_in_leaf=15, random_strength=0.5, border_count=254,
            cat_features=cat_idx_cb, eval_metric="RMSE", loss_function="RMSE",
            task_type="GPU", random_seed=SEED + fold, verbose=0,
            early_stopping_rounds=150, use_best_model=True
        )
        model.fit(X_train_cb.iloc[tr_idx], y_train.iloc[tr_idx], 
                  eval_set=(X_train_cb.iloc[val_idx], y_train.iloc[val_idx]), verbose=False)
        cb_oof[val_idx] = model.predict(X_train_cb.iloc[val_idx])
        cb_test        += model.predict(X_test_cb) / N_FOLDS
        
    cb_r2 = r2_score(y_train, cb_oof)
    print(f"    CatBoost OOF R²: {cb_r2:.5f}")
    
    # 2. LightGBM
    print("  Training LightGBM...")
    lgb_oof  = np.zeros(len(train))
    lgb_test = np.zeros(len(test))
    LGB_PARAMS = {
        "objective": "regression", "metric": "rmse", "num_leaves": 127, "max_depth": -1,
        "learning_rate": 0.025, "feature_fraction": 0.70, "bagging_fraction": 0.80,
        "bagging_freq": 5, "min_child_samples": 20, "lambda_l1": 0.10, "lambda_l2": 0.30,
        "n_jobs": -1, "seed": SEED, "verbose": -1
    }
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_num, y_train)):
        dtr = lgb.Dataset(X_train_num.iloc[tr_idx], label=y_train.iloc[tr_idx])
        dvl = lgb.Dataset(X_train_num.iloc[val_idx], label=y_train.iloc[val_idx], reference=dtr)
        model = lgb.train(
            LGB_PARAMS, dtr, num_boost_round=5000, valid_sets=[dvl],
            callbacks=[lgb.early_stopping(200, verbose=False)]
        )
        lgb_oof[val_idx] = model.predict(X_train_num.iloc[val_idx], num_iteration=model.best_iteration)
        lgb_test        += model.predict(X_test_num, num_iteration=model.best_iteration) / N_FOLDS
        
    lgb_r2 = r2_score(y_train, lgb_oof)
    print(f"    LightGBM OOF R²: {lgb_r2:.5f}")
    
    # 3. XGBoost
    print("  Training XGBoost...")
    xgb_oof  = np.zeros(len(train))
    xgb_test = np.zeros(len(test))
    XGB_PARAMS = {
        "objective": "reg:squarederror", "eval_metric": "rmse", "max_depth": 6,
        "learning_rate": 0.035, "subsample": 0.80, "colsample_bytree": 0.70,
        "colsample_bylevel": 0.70, "min_child_weight": 15, "reg_alpha": 0.10,
        "reg_lambda": 0.30, "tree_method": "hist", "device": "cuda",
        "random_state": SEED, "n_jobs": -1, "verbosity": 0
    }
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_num, y_train)):
        dtr = xgb.DMatrix(X_train_num.iloc[tr_idx], label=y_train.iloc[tr_idx])
        dvl = xgb.DMatrix(X_train_num.iloc[val_idx])
        model = xgb.train(
            XGB_PARAMS, dtr, num_boost_round=5000, evals=[(xgb.DMatrix(X_train_num.iloc[val_idx], label=y_train.iloc[val_idx]), "val")],
            early_stopping_rounds=200, verbose_eval=False
        )
        xgb_oof[val_idx] = model.predict(dvl)
        xgb_test        += model.predict(xgb.DMatrix(X_test_num)) / N_FOLDS
        
    xgb_r2 = r2_score(y_train, xgb_oof)
    print(f"    XGBoost OOF R² : {xgb_r2:.5f}")
    
    # 4. Ensemble optimization
    print("  Optimizing Ensemble Blend...")
    oof_mat = np.column_stack([cb_oof, lgb_oof, xgb_oof])
    
    # Simple grid search initialization
    step = 0.05
    best_grid = 0
    bw = [0.4, 0.4, 0.2]
    for w1 in np.arange(0, 1.001, step):
        for w2 in np.arange(0, 1.001 - w1, step):
            w3 = max(1.0 - w1 - w2, 0.0)
            blend = w1 * cb_oof + w2 * lgb_oof + w3 * xgb_oof
            score = r2_score(y_train, blend)
            if score > best_grid:
                best_grid = score
                bw = [w1, w2, w3]
                
    def neg_r2(w):
        wn = np.array(w); wn = wn / (wn.sum() + 1e-9)
        return -r2_score(y_train, oof_mat @ wn)
        
    res = minimize(neg_r2, x0=bw, method="L-BFGS-B", bounds=[(0,1)]*3)
    opt_w = np.array(res.x); opt_w = opt_w / opt_w.sum()
    opt_r2 = r2_score(y_train, oof_mat @ opt_w)
    print(f"    Optimal weights: CB={opt_w[0]:.4f} LGB={opt_w[1]:.4f} XGB={opt_w[2]:.4f}")
    print(f"    Ensemble OOF R²: {opt_r2:.5f}")
    
    final_pred = np.clip(np.column_stack([cb_test, lgb_test, xgb_test]) @ opt_w, 0.0, 1.0)
    
    return {
        "cb_oof": cb_oof, "cb_test": cb_test,
        "lgb_oof": lgb_oof, "lgb_test": lgb_test,
        "xgb_oof": xgb_oof, "xgb_test": xgb_test,
        "ensemble_oof": oof_mat @ opt_w, "ensemble_test": final_pred,
        "ensemble_r2": opt_r2, "opt_w": opt_w
    }

# Run Baseline training pipeline first to get predictions and establish baseline OOFs
baseline_results = train_pipeline(ALL_NUM_BASE, ALL_CB_BASE, "Base Features")

# =============================================================================
# EXPERIMENT A — RANK AVERAGING ENSEMBLE
# =============================================================================
print("\n" + "=" * 70)
print("EXPERIMENT A — RANK AVERAGING ENSEMBLE")
print("=" * 70)

# Convert OOF predictions to percentile ranks
cb_oof_rank = pd.Series(baseline_results["cb_oof"]).rank(pct=True).values
lgb_oof_rank = pd.Series(baseline_results["lgb_oof"]).rank(pct=True).values
xgb_oof_rank = pd.Series(baseline_results["xgb_oof"]).rank(pct=True).values

cb_test_rank = pd.Series(baseline_results["cb_test"]).rank(pct=True).values
lgb_test_rank = pd.Series(baseline_results["lgb_test"]).rank(pct=True).values
xgb_test_rank = pd.Series(baseline_results["xgb_test"]).rank(pct=True).values

# Weighted rank blend
opt_w = baseline_results["opt_w"]
blend_oof_rank = opt_w[0] * cb_oof_rank + opt_w[1] * lgb_oof_rank + opt_w[2] * xgb_oof_rank
blend_test_rank = opt_w[0] * cb_test_rank + opt_w[1] * lgb_test_rank + opt_w[2] * xgb_test_rank

# Map ensembled ranks back to the true training target distribution values
def rank_to_val(ranks, values):
    sorted_vals = np.sort(values)
    indices = (ranks * (len(values) - 1)).astype(int)
    return sorted_vals[indices]

rank_oof_pred = rank_to_val(blend_oof_rank, y_train.values)
rank_test_pred = rank_to_val(blend_test_rank, y_train.values)

rank_r2 = r2_score(y_train, rank_oof_pred)
print(f"Rank Averaging OOF R²: {rank_r2:.5f}")

# =============================================================================
# EXPERIMENT B — STRONGER TARGET ENCODINGS
# =============================================================================
print("\n" + "=" * 70)
print("EXPERIMENT B — STRONGER TARGET ENCODINGS")
print("=" * 70)

exp_b_results = train_pipeline(ALL_NUM_EXPB, ALL_CB_EXPB, "Triple-Interact Target Encodings")

# =============================================================================
# EXPERIMENT C — RESIDUAL CORRECTION MODEL (STACKING)
# =============================================================================
print("\n" + "=" * 70)
print("EXPERIMENT C — RESIDUAL CORRECTION MODEL (STACKING)")
print("=" * 70)

# Target is residual of baseline ensemble predictions
residuals = y_train.values - baseline_results["ensemble_oof"]

X_train_num = train[ALL_NUM_BASE].copy().fillna(-999)
X_test_num  = test[ALL_NUM_BASE].copy().fillna(-999)

res_oof_corrections = np.zeros(len(train))
res_test_corrections = np.zeros(len(test))

RES_LGB_PARAMS = {
    "objective": "regression", "metric": "rmse", "num_leaves": 31, "max_depth": 5,
    "learning_rate": 0.03, "feature_fraction": 0.80, "bagging_fraction": 0.80,
    "bagging_freq": 5, "min_child_samples": 20, "n_jobs": -1, "seed": SEED, "verbose": -1
}

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_num, residuals)):
    dtr = lgb.Dataset(X_train_num.iloc[tr_idx], label=residuals[tr_idx])
    dvl = lgb.Dataset(X_train_num.iloc[val_idx], label=residuals[val_idx], reference=dtr)
    model = lgb.train(
        RES_LGB_PARAMS, dtr, num_boost_round=1000, valid_sets=[dvl],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    res_oof_corrections[val_idx] = model.predict(X_train_num.iloc[val_idx], num_iteration=model.best_iteration)
    res_test_corrections        += model.predict(X_test_num, num_iteration=model.best_iteration) / N_FOLDS

corrected_oof = np.clip(baseline_results["ensemble_oof"] + res_oof_corrections, 0.0, 1.0)
corrected_test = np.clip(baseline_results["ensemble_test"] + res_test_corrections, 0.0, 1.0)

stacking_r2 = r2_score(y_train, corrected_oof)
print(f"Residual Stacking OOF R²: {stacking_r2:.5f}")

# =============================================================================
# EXPERIMENT D — PSEUDO-LABEL RETRAINING
# =============================================================================
print("\n" + "=" * 70)
print("EXPERIMENT D — PSEUDO-LABEL RETRAINING")
print("=" * 70)

# Use highest-confidence test predictions from baseline ensemble
pseudo_labels = baseline_results["ensemble_test"]

# Create pseudo-labeled test dataframe
pseudo_test = test.copy()
pseudo_test[TARGET] = pseudo_labels

# Augment training set per fold
X_train_num_orig = train[ALL_NUM_BASE].copy()
X_test_num_orig  = test[ALL_NUM_BASE].copy().fillna(-999)

pseudo_oof  = np.zeros(len(train))
pseudo_test_preds = np.zeros(len(test))

LGB_PARAMS = {
    "objective": "regression", "metric": "rmse", "num_leaves": 127, "max_depth": -1,
    "learning_rate": 0.025, "feature_fraction": 0.70, "bagging_fraction": 0.80,
    "bagging_freq": 5, "min_child_samples": 20, "lambda_l1": 0.10, "lambda_l2": 0.30,
    "n_jobs": -1, "seed": SEED, "verbose": -1
}

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train_num_orig, y_train)):
    # Slice original train split
    X_tr_orig = X_train_num_orig.iloc[tr_idx].copy()
    y_tr_orig = y_train.iloc[tr_idx].copy()
    
    # Append full pseudo-labeled test set to the training split
    X_tr_aug = pd.concat([X_tr_orig, test[ALL_NUM_BASE]], axis=0, ignore_index=True).fillna(-999)
    y_tr_aug = pd.concat([y_tr_orig, pseudo_test[TARGET]], axis=0, ignore_index=True)
    
    X_val = X_train_num_orig.iloc[val_idx].copy().fillna(-999)
    y_val = y_train.iloc[val_idx].copy()
    
    dtr = lgb.Dataset(X_tr_aug, label=y_tr_aug)
    dvl = lgb.Dataset(X_val, label=y_val, reference=dtr)
    
    model = lgb.train(
        LGB_PARAMS, dtr, num_boost_round=5000, valid_sets=[dvl],
        callbacks=[lgb.early_stopping(200, verbose=False)]
    )
    
    pseudo_oof[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    pseudo_test_preds  += model.predict(X_test_num_orig, num_iteration=model.best_iteration) / N_FOLDS

pseudo_r2 = r2_score(y_train, pseudo_oof)
print(f"Pseudo-Label Retraining (LightGBM) OOF R²: {pseudo_r2:.5f}")

# =============================================================================
# AUTO-SELECT BEST IMPROVEMENT
# =============================================================================
print("\n" + "=" * 70)
print("AUTO-SELECT BEST IMPROVEMENT")
print("=" * 70)

current_best = baseline_results["ensemble_r2"]
print(f"Original Benchmark R²: {current_best:.5f}")

experiments = {
    "Experiment A (Rank Averaging)": (rank_r2, rank_test_pred),
    "Experiment B (Triple Target Encodings)": (exp_b_results["ensemble_r2"], exp_b_results["ensemble_test"]),
    "Experiment C (Residual Stacking)": (stacking_r2, corrected_test),
    "Experiment D (Pseudo-Label Stacking)": (pseudo_r2, pseudo_test_preds)
}

best_exp_name = "Original Ensemble"
best_score = current_best
best_test_pred = baseline_results["ensemble_test"]

for name, (score, test_pred) in experiments.items():
    print(f"{name:40s} : OOF R² = {score:.5f}")
    if score > best_score:
        best_score = score
        best_exp_name = name
        best_test_pred = test_pred

print("\n" + "-" * 50)
if best_score > current_best:
    delta = best_score - current_best
    print(f"SUCCESS! {best_exp_name} improved R² score!")
    print(f"New Best OOF R²: {best_score:.5f} (Delta = +{delta:.5f})")
    
    # Save submission_v3
    submission = pd.DataFrame({"Index": test["Index"].astype(int), "demand": best_test_pred})
    assert len(submission) == 41778, "Row count wrong!"
    assert submission["demand"].isnull().sum() == 0, "NaN values present!"
    
    sub_path = os.path.join(OUT_DIR, "submission_v3.csv")
    submission.to_csv(sub_path, index=False)
    print(f"\nSaved improved submission to: {sub_path}")
    print(f"Recommended file to upload  : submission_v3.csv")
else:
    print("No experiment beat the baseline score of 0.96198.")
    print("Recommendation: Keep the existing submission.csv file.")
