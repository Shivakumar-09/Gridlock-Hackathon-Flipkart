import pandas as pd
import numpy as np
import os

DATA_DIR = r"c:\hackathon\flipkart\dataset"
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

print(f"Train day 48 rows: {len(train[train['day'] == 48])}")
print(f"Train day 49 rows: {len(train[train['day'] == 49])}")
print(f"Test day 49 rows: {len(test[test['day'] == 49])}")

print("\nTrain day 48 unique timestamps count:", train[train['day'] == 48]['timestamp'].nunique())
print("Train day 49 unique timestamps count:", train[train['day'] == 49]['timestamp'].nunique())
print("Test day 49 unique timestamps count:", test[test['day'] == 49]['timestamp'].nunique())

train_48_ts = set(train[train['day'] == 48]['timestamp'].unique())
train_49_ts = set(train[train['day'] == 49]['timestamp'].unique())
test_49_ts = set(test[test['day'] == 49]['timestamp'].unique())

print(f"Train 48 timestamps overlap with Train 49: {len(train_48_ts & train_49_ts)} / {len(train_49_ts)}")
print(f"Test 49 timestamps overlap with Train 49: {len(test_49_ts & train_49_ts)} / {len(test_49_ts)}")
print(f"Test 49 timestamps overlap with Train 48: {len(test_49_ts & train_48_ts)} / {len(test_49_ts)}")

print("\nTrain unique geohashes:", train['geohash'].nunique())
print("Test unique geohashes:", test['geohash'].nunique())
overlap = set(train['geohash'].unique()) & set(test['geohash'].unique())
print("Geohash overlap:", len(overlap))

# Let's see some geohash-timestamp combinations
print("\nTrain day 49 geohash-timestamp unique combinations:", len(train[train['day'] == 49].groupby(['geohash', 'timestamp'])))
print("Test day 49 geohash-timestamp unique combinations:", len(test[test['day'] == 49].groupby(['geohash', 'timestamp'])))

# Are there overlapping geohash-timestamp in train and test on day 49?
train_49_pairs = set(train[train['day'] == 49].apply(lambda r: f"{r['geohash']}_{r['timestamp']}", axis=1))
test_49_pairs = set(test[test['day'] == 49].apply(lambda r: f"{r['geohash']}_{r['timestamp']}", axis=1))
print("Overlapping (geohash, timestamp) pairs on day 49 between train and test:", len(train_49_pairs & test_49_pairs))
