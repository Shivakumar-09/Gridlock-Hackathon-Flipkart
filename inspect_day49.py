import pandas as pd
import numpy as np
import os

DATA_DIR = r"c:\hackathon\flipkart\dataset"
train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

def parse_ts(ts):
    parts = str(ts).split(":")
    h = int(parts[0])
    m = int(parts[1]) if len(parts) > 1 else 0
    return h * 60 + m

train['minutes'] = train['timestamp'].apply(parse_ts)
test['minutes'] = test['timestamp'].apply(parse_ts)

train_49 = train[train['day'] == 49]
test_49 = test[test['day'] == 49]

tr_49_min = sorted(train_49['minutes'].unique())
te_49_min = sorted(test_49['minutes'].unique())

print("Train day 49 minutes:", tr_49_min)
print("Test day 49 minutes:", te_49_min)

def to_time(m):
    return f"{m//60}:{m%60:02d}"

print("Train day 49 times:", [to_time(m) for m in tr_49_min])
print("Test day 49 times:", [to_time(m) for m in te_49_min])

print("\nTrain day 49 unique geohashes:", train_49['geohash'].nunique())
print("Test day 49 unique geohashes:", test_49['geohash'].nunique())
overlap_49 = set(train_49['geohash'].unique()) & set(test_49['geohash'].unique())
print("Overlap geohashes on day 49:", len(overlap_49))

gh_ts_48 = train[train['day'] == 48].groupby('geohash')['timestamp'].nunique()
print("\nDay 48 timestamps per geohash summary:")
print(gh_ts_48.describe())
