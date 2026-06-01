# =============================================================================
# FLIPKART GRIDLOCK HACKATHON 2.0 — LEADERBOARD-SAFE SOLUTION
# Objective: Generalize to Day 49 test set by eliminating target leakage
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import os, gc, time
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from catboost import CatBoostRegressor

SEED = 42
np.random.seed(SEED)

DATA_DIR = r"c:\hackathon\flipkart\dataset"
OUT_DIR  = r"c:\hackathon\flipkart"

print("=" * 70)
print("LOADING DATASETS...")
print("=" * 70)

train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sub   = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

print(f"Train shape: {train.shape}")
print(f"Test shape : {test.shape}")

# =============================================================================
# STEP 1 — STABLE BASE FEATURES
# =============================================================================
print("\n" + "=" * 70)
print("STEP 1 — ENGINEERING STABLE BASE FEATURES")
print("=" * 70)

def parse_ts(ts):
    parts = str(ts).split(":")
    h = int(parts[0])
    m = int(parts[1]) if len(parts) > 1 else 0
    return h, m

def preprocess(df):
    df = df.copy()
    
    # Parse timestamp into hour and minute
    df[["hour", "minute"]] = pd.DataFrame(
        df["timestamp"].apply(parse_ts).tolist(), index=df.index
    )
    
    # Fill categorical missing values
    df["RoadType"]      = df["RoadType"].fillna("Unknown")
    df["Weather"]       = df["Weather"].fillna("Unknown")
    df["LargeVehicles"] = df["LargeVehicles"].fillna("Unknown")
    df["Landmarks"]     = df["Landmarks"].fillna("Unknown")
    
    return df

train = preprocess(train)
test  = preprocess(test)

# Fill numeric missing values using train medians
lanes_median = train["NumberofLanes"].median()
temp_median  = train["Temperature"].median()

train["NumberofLanes"] = train["NumberofLanes"].fillna(lanes_median)
test["NumberofLanes"]  = test["NumberofLanes"].fillna(lanes_median)

train["Temperature"]   = train["Temperature"].fillna(temp_median)
test["Temperature"]    = test["Temperature"].fillna(temp_median)

# Geohash frequency count (computed securely from train set)
gh_freq = train["geohash"].value_counts().to_dict()
train["geohash_freq"] = train["geohash"].map(gh_freq).fillna(0)
test["geohash_freq"]  = test["geohash"].map(gh_freq).fillna(0)

# =============================================================================
# STEP 2 — LEAKAGE-FREE SMOOTHED TARGET ENCODING
# =============================================================================
print("\n" + "=" * 70)
print("STEP 2 — BUILDING LEAKAGE-FREE SMOOTHED TARGET ENCODINGS")
print("=" * 70)

TARGET = "demand"
N_TE_FOLD = 5

def kfold_te(train_df, test_df, group_cols, target=TARGET, n_folds=N_TE_FOLD, seed=SEED, m=10):
    """
    Computes out-of-fold target encoding with additive smoothing to handle low support groups.
    """
    if isinstance(group_cols, str):
        group_cols = [group_cols]
        
    key_name = "__".join(group_cols)
    col_name = f"te__{key_name}"
    
    tr_key = train_df[group_cols].astype(str).agg("|".join, axis=1)
    te_key = test_df[group_cols].astype(str).agg("|".join, axis=1)
    
    tr_enc = np.zeros(len(train_df))
    global_mean = train_df[target].mean()
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    for tr_idx, val_idx in kf.split(train_df):
        fold_train = train_df.iloc[tr_idx]
        fold_key = tr_key.iloc[tr_idx]
        
        # Calculate smoothed mean
        stats = fold_train.groupby(fold_key)[target].agg(["mean", "count"])
        smoothed = (stats["mean"] * stats["count"] + m * global_mean) / (stats["count"] + m)
        
        val_key = tr_key.iloc[val_idx]
        tr_enc[val_idx] = val_key.map(smoothed).fillna(global_mean).values
        
    # Test encoding uses the global training statistics
    global_stats = train_df.groupby(tr_key)[target].agg(["mean", "count"])
    global_smoothed = (global_stats["mean"] * global_stats["count"] + m * global_mean) / (global_stats["count"] + m)
    
    te_enc = te_key.map(global_smoothed).fillna(global_mean).values
    
    return tr_enc, te_enc, col_name

# Compute stable target encodings
print("Encoding: geohash mean demand...")
tr_gh, te_gh, cname_gh = kfold_te(train, test, "geohash", m=10)
train[cname_gh] = tr_gh
test[cname_gh]  = te_gh

print("Encoding: geohash + hour mean demand...")
tr_gh_hr, te_gh_hr, cname_gh_hr = kfold_te(train, test, ["geohash", "hour"], m=10)
train[cname_gh_hr] = tr_gh_hr
test[cname_gh_hr]  = te_gh_hr

# =============================================================================
# STEP 3 — FACTORIZING CATEGORICAL FEATURES
# =============================================================================
print("\n" + "=" * 70)
print("STEP 3 — CATEGORICAL ENCODING")
print("=" * 70)

CAT_COLS = ["geohash", "RoadType", "Weather", "LargeVehicles", "Landmarks"]

for col in CAT_COLS:
    le = LabelEncoder()
    full_series = pd.concat([train[col].astype(str), test[col].astype(str)], axis=0)
    le.fit(full_series)
    train[col + "_encoded"] = le.transform(train[col].astype(str))
    test[col + "_encoded"]  = le.transform(test[col].astype(str))
    print(f"  Encoded {col} -> {col}_encoded ({len(le.classes_)} unique categories)")

# =============================================================================
# STEP 4 — PREPARING FEATURE LISTS
# =============================================================================
NUM_FEATURES = [
    "hour", "minute", "NumberofLanes", "Temperature", 
    "te__geohash", "te__geohash__hour", "geohash_freq"
]

CAT_ENCODED = [
    "RoadType_encoded", "Weather_encoded", 
    "LargeVehicles_encoded", "Landmarks_encoded"
]

ALL_FEATURES = NUM_FEATURES + CAT_ENCODED

print(f"\nTraining features ({len(ALL_FEATURES)}):")
for f in ALL_FEATURES:
    print(f"  - {f}")

X_train = train[ALL_FEATURES]
y_train = train[TARGET]
X_test  = test[ALL_FEATURES]

# =============================================================================
# STEP 5 — MODEL TRAINING AND CROSS-VALIDATION
# =============================================================================
print("\n" + "=" * 70)
print("STEP 5 — MODEL TRAINING AND CROSS-VALIDATION")
print("=" * 70)

N_FOLDS = 5
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# CatBoost config
# We treat low-cardinality features as categorical, and geohash as numeric to maximize GPU speed
cb_cat_features = [
    ALL_FEATURES.index("RoadType_encoded"),
    ALL_FEATURES.index("Weather_encoded"),
    ALL_FEATURES.index("LargeVehicles_encoded"),
    ALL_FEATURES.index("Landmarks_encoded")
]

cb_oof  = np.zeros(len(train))
cb_test = np.zeros(len(test))
cb_scores = []

# LightGBM config
lgb_cat_features = ["RoadType_encoded", "Weather_encoded", "LargeVehicles_encoded", "Landmarks_encoded"]
LGB_PARAMS = {
    "objective":         "regression",
    "metric":            "rmse",
    "num_leaves":        31,
    "max_depth":         5,
    "learning_rate":     0.08,
    "feature_fraction":  0.80,
    "bagging_fraction":  0.90,
    "bagging_freq":      1,
    "min_child_samples": 20,
    "seed":              SEED,
    "n_jobs":            -1,
    "verbose":           -1,
}

lgb_oof  = np.zeros(len(train))
lgb_test = np.zeros(len(test))
lgb_scores = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    print(f"\n--- FOLD {fold+1}/{N_FOLDS} ---")
    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
    
    # ── CatBoost ──
    t0 = time.time()
    cb_model = CatBoostRegressor(
        iterations=600,
        learning_rate=0.08,
        depth=5,
        l2_leaf_reg=5.0,
        cat_features=cb_cat_features,
        eval_metric="RMSE",
        loss_function="RMSE",
        task_type="CPU",
        thread_count=-1,
        random_seed=SEED + fold,
        verbose=0,
        early_stopping_rounds=60,
        use_best_model=True
    )
    cb_model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
    
    val_pred_cb = cb_model.predict(X_val)
    cb_score = r2_score(y_val, val_pred_cb)
    cb_scores.append(cb_score)
    cb_oof[val_idx] = val_pred_cb
    cb_test += cb_model.predict(X_test) / N_FOLDS
    print(f"CatBoost R²: {cb_score:.5f}  ({time.time() - t0:.1f}s)")
    
    # ── LightGBM ──
    t0 = time.time()
    dtr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=lgb_cat_features)
    dvl = lgb.Dataset(X_val, label=y_val, reference=dtr, categorical_feature=lgb_cat_features)
    
    lgb_model = lgb.train(
        LGB_PARAMS,
        dtr,
        num_boost_round=600,
        valid_sets=[dvl],
        callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(-1)]
    )
    
    val_pred_lgb = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
    lgb_score = r2_score(y_val, val_pred_lgb)
    lgb_scores.append(lgb_score)
    lgb_oof[val_idx] = val_pred_lgb
    lgb_test += lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration) / N_FOLDS
    print(f"LightGBM R²: {lgb_score:.5f}  ({time.time() - t0:.1f}s)")

# 50/50 Blend
blend_oof = 0.5 * cb_oof + 0.5 * lgb_oof
blend_test = 0.5 * cb_test + 0.5 * lgb_test

cb_oof_r2 = r2_score(y_train, cb_oof)
lgb_oof_r2 = r2_score(y_train, lgb_oof)
blend_oof_r2 = r2_score(y_train, blend_oof)

# =============================================================================
# STEP 6 — MODEL COMPARISON AND REPORT
# =============================================================================
print("\n" + "=" * 70)
print("STEP 6 — CROSS-VALIDATION MODEL COMPARISON")
print("=" * 70)
print(f"CatBoost OOF R²: {cb_oof_r2:.5f} (Mean Folds = {np.mean(cb_scores):.5f})")
print(f"LightGBM OOF R²: {lgb_oof_r2:.5f} (Mean Folds = {np.mean(lgb_scores):.5f})")
print(f"50/50 Blend OOF R²: {blend_oof_r2:.5f}")

# Select the best generalizing predictions
if cb_oof_r2 > lgb_oof_r2 and cb_oof_r2 > blend_oof_r2:
    print("\nSelected Best Model: CatBoost Regressor")
    final_pred = cb_test
elif lgb_oof_r2 > cb_oof_r2 and lgb_oof_r2 > blend_oof_r2:
    print("\nSelected Best Model: LightGBM Regressor")
    final_pred = lgb_test
else:
    print("\nSelected Best Model: 50/50 Blend Ensemble")
    final_pred = blend_test

# Bound final predictions
final_pred = np.clip(final_pred, 0.0, 1.0)

# =============================================================================
# STEP 7 — GENERATE SAFE SUBMISSION
# =============================================================================
print("\n" + "=" * 70)
print("STEP 7 — GENERATING SAFE SUBMISSION")
print("=" * 70)

assert len(final_pred) == 41778, f"Row count wrong: {len(final_pred)}"
submission = pd.DataFrame({"Index": test["Index"].astype(int), "demand": final_pred})
assert submission["demand"].isnull().sum() == 0, "NaN in predictions!"

sub_path = os.path.join(OUT_DIR, "submission_safe.csv")
submission.to_csv(sub_path, index=False)

print(f"\nSaved successfully to: {sub_path}")
print(f"Row count : {len(submission)}")
print(f"Columns   : {list(submission.columns)}")
print(f"Demand bounds: [{final_pred.min():.6f}, {final_pred.max():.6f}]")
print(f"Demand mean  : {final_pred.mean():.6f}")

print("\nFirst 10 rows of submission_safe.csv:")
print(submission.head(10).to_string(index=False))
