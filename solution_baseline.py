# =============================================================================
# FLIPKART GRIDLOCK HACKATHON 2.0 — STARTER BASELINE MODEL
# Target: Traffic Demand Prediction
# Validation Strategy: Leakage-Safe 5-Fold Cross-Validation
# =============================================================================

import os
import gc
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

# Paths configuration
DATA_DIR = r"c:\hackathon\flipkart\archive\dataset"
OUT_DIR  = r"c:\hackathon\flipkart"

def parse_ts(ts):
    """Parses timestamp HH:MM into hour and minute components."""
    parts = str(ts).split(":")
    h = int(parts[0])
    m = int(parts[1]) if len(parts) > 1 else 0
    return h, m

def preprocess(df):
    """Preprocesses spatial, temporal, and categorical fields."""
    df = df.copy()
    
    # Parse timestamp into hour and minute
    df[["hour", "minute"]] = pd.DataFrame(
        df["timestamp"].apply(parse_ts).tolist(), index=df.index
    )
    df["total_minutes"] = df["hour"] * 60 + df["minute"]
    
    # Cyclic time features
    df["sin_hour"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour"] / 24)
    
    # Fill categorical missing values
    df["RoadType"]      = df["RoadType"].fillna("Unknown")
    df["Weather"]       = df["Weather"].fillna("Unknown")
    df["LargeVehicles"] = df["LargeVehicles"].fillna("Unknown")
    df["Landmarks"]     = df["Landmarks"].fillna("Unknown")
    
    return df

def kfold_target_encoding(train_df, test_df, group_cols, target="demand", n_folds=5, seed=SEED, m=10):
    """
    Computes out-of-fold target encoding with additive smoothing to prevent leakage.
    m is the smoothing parameter (weight given to global mean).
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

def main():
    print("=" * 70)
    print("PHASE 1 — LOADING DATASETS")
    print("=" * 70)
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    
    print(f"Train shape: {train.shape}")
    print(f"Test shape : {test.shape}")
    
    print("\n" + "=" * 70)
    print("PHASE 2 — ENGINEERING BASE FEATURES")
    print("=" * 70)
    train = preprocess(train)
    test  = preprocess(test)
    
    # Fill numeric missing values using train medians to avoid test set leakage
    lanes_median = train["NumberofLanes"].median()
    temp_median  = train["Temperature"].median()
    
    train["NumberofLanes"] = train["NumberofLanes"].fillna(lanes_median)
    test["NumberofLanes"]  = test["NumberofLanes"].fillna(lanes_median)
    
    train["Temperature"]   = train["Temperature"].fillna(temp_median)
    test["Temperature"]    = test["Temperature"].fillna(temp_median)
    
    # Frequency counts
    gh_freq = train["geohash"].value_counts().to_dict()
    train["geohash_freq"] = train["geohash"].map(gh_freq).fillna(0)
    test["geohash_freq"]  = test["geohash"].map(gh_freq).fillna(0)
    
    # Target Encodings
    print("Computing smoothed target encodings...")
    tr_gh, te_gh, cname_gh = kfold_target_encoding(train, test, "geohash", m=10)
    train[cname_gh] = tr_gh
    test[cname_gh]  = te_gh
    
    tr_gh_hr, te_gh_hr, cname_gh_hr = kfold_target_encoding(train, test, ["geohash", "hour"], m=10)
    train[cname_gh_hr] = tr_gh_hr
    test[cname_gh_hr]  = te_gh_hr
    
    # Encode categorical features
    cat_cols = ["RoadType", "Weather", "LargeVehicles", "Landmarks"]
    for col in cat_cols:
        le = LabelEncoder()
        full_series = pd.concat([train[col].astype(str), test[col].astype(str)], axis=0)
        le.fit(full_series)
        train[col + "_encoded"] = le.transform(train[col].astype(str))
        test[col + "_encoded"]  = le.transform(test[col].astype(str))
        
    num_features = [
        "hour", "minute", "NumberofLanes", "Temperature", 
        "te__geohash", "te__geohash__hour", "geohash_freq",
        "sin_hour", "cos_hour"
    ]
    cat_encoded = [c + "_encoded" for c in cat_cols]
    features = num_features + cat_encoded
    
    X_train = train[features]
    y_train = train["demand"]
    X_test  = test[features]
    
    print(f"Features selected for model ({len(features)}): {features}")
    
    print("\n" + "=" * 70)
    print("PHASE 3 — TRAINING WITH 5-FOLD CROSS-VALIDATION")
    print("=" * 70)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(train))
    test_preds = np.zeros(len(test))
    scores = []
    
    lgb_params = {
        "objective":         "regression",
        "metric":            "rmse",
        "num_leaves":        63,
        "max_depth":         7,
        "learning_rate":     0.05,
        "feature_fraction":  0.80,
        "bagging_fraction":  0.90,
        "bagging_freq":      1,
        "min_child_samples": 20,
        "seed":              SEED,
        "n_jobs":            -1,
        "verbose":           -1,
    }
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        t0 = time.time()
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
        
        dtr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_encoded)
        dvl = lgb.Dataset(X_val, label=y_val, reference=dtr, categorical_feature=cat_encoded)
        
        model = lgb.train(
            lgb_params,
            dtr,
            num_boost_round=1000,
            valid_sets=[dvl],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        
        val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        score = r2_score(y_val, val_pred)
        scores.append(score)
        
        oof_preds[val_idx] = val_pred
        test_preds += model.predict(X_test, num_iteration=model.best_iteration) / 5
        print(f"Fold {fold+1} R2: {score:.5f} | Time: {time.time() - t0:.1f}s")
        
    mean_r2 = np.mean(scores)
    std_r2 = np.std(scores)
    overall_r2 = r2_score(y_train, oof_preds)
    print(f"\nMean CV R2: {mean_r2:.5f} +/- {std_r2:.5f}")
    print(f"Overall OOF R2: {overall_r2:.5f}")
    
    print("\n" + "=" * 70)
    print("PHASE 4 — GENERATING SUBMISSION")
    print("=" * 70)
    
    test_preds = np.clip(test_preds, 0.0, 1.0)
    submission = pd.DataFrame({"Index": test["Index"].astype(int), "demand": test_preds})
    
    sub_path = os.path.join(OUT_DIR, "submission.csv")
    submission.to_csv(sub_path, index=False)
    print(f"Saved submission to: {sub_path}")
    print(f"Total rows: {len(submission)}")
    print(f"Range of predictions: [{test_preds.min():.5f}, {test_preds.max():.5f}]")
    print(f"Mean prediction: {test_preds.mean():.5f}")

if __name__ == '__main__':
    main()
