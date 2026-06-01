import pandas as pd
import numpy as np
import os

DATA_DIR = r"c:\hackathon\flipkart\dataset"
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

print(f"Train columns: {train.columns.tolist()}")
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Train day range: {train['day'].min()} to {train['day'].max()} (Total: {train['day'].nunique()} days)")
print(f"Test day range: {test['day'].min()} to {test['day'].max()} (Total: {test['day'].nunique()} days)")
print(f"Train days unique: {sorted(train['day'].unique())}")
print(f"Test days unique: {sorted(test['day'].unique())}")
print(f"Timestamp samples: {train['timestamp'].head(10).tolist()}")
print(f"Geohash sample: {train['geohash'].head(5).tolist()}")
print(f"Target demand summary:\n{train['demand'].describe()}")
print(f"Missing values in Train:\n{train.isnull().sum()}")
print(f"Missing values in Test:\n{test.isnull().sum()}")
