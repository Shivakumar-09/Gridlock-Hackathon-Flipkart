import pandas as pd
import numpy as np
import os

DATA_DIR = r"c:\hackathon\flipkart\dataset"
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))

df_48 = train[train['day'] == 48].copy()
df_49 = train[train['day'] == 49].copy()

merged = pd.merge(df_49, df_48, on=['geohash', 'timestamp'], suffixes=('_49', '_48'))

print(f"Number of overlapping (geohash, timestamp) pairs between Day 48 and Day 49: {len(merged)}")
correlation = merged['demand_49'].corr(merged['demand_48'])
print(f"Correlation between Day 48 demand and Day 49 demand for the same geohash at the same time: {correlation:.5f}")

from sklearn.metrics import r2_score
r2 = r2_score(merged['demand_49'], merged['demand_48'])
print(f"R2 score of using Day 48 demand as direct prediction for Day 49 demand: {r2:.5f}")
