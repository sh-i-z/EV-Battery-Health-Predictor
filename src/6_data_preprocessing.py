
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# Ye libraries kyun?
# - pandas: Dataset load aur manipulate karne ke liye.
# - GroupShuffleSplit: Battery-wise training aur testing data divide karne ke liye.


# Load cleaned dataset
data = pd.read_csv("data/ev_battery_health_cleaned.csv")

# Check dataset shape
print("Dataset Shape:", data.shape)

# Check unique batteries
print("Unique Batteries:", data["battery_id"].nunique()) #nunique() batayega ki dataset mein kitni distinct batteries hain.


# ====== Feature Preparation ======

# Convert date columns
date_features = [
    "manufacture_date",
    "observation_date",
    "last_service_date"
]

for column in date_features:
    data[column] = pd.to_datetime(data[column], errors="coerce")


# Create days since last service
data["days_since_last_service"] = (
    data["observation_date"] - data["last_service_date"]
).dt.days


# Separate battery IDs for group splitting
groups = data["battery_id"]


# SOH prediction
X_soh = data.drop(columns=[
    "battery_id",
    "observation_id",
    "capacity_fade_pct",
    "state_of_health_pct",
    "remaining_useful_life_cycles",
    "failure_risk_label",
    "degradation_category",
    "manufacture_date",
    "observation_date",
    "last_service_date"
])

y_soh = data["state_of_health_pct"]


# RUL prediction
X_rul = X_soh.copy()
X_rul["capacity_fade_pct"] = data["capacity_fade_pct"]

y_rul = data["remaining_useful_life_cycles"]


print("SOH Features:", X_soh.shape)
print("RUL Features:", X_rul.shape)

#===================Data Split===========-=====
# Step 6.4 — Train-Test Split

# Create group-based train-test splitter
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

# Generate train-test indices
train_idx, test_idx = next(
    splitter.split(X_soh, y_soh, groups=groups)
)


# Split SOH data
X_soh_train = X_soh.iloc[train_idx]
X_soh_test = X_soh.iloc[test_idx]

y_soh_train = y_soh.iloc[train_idx]
y_soh_test = y_soh.iloc[test_idx]

print("SOH Training Shape:", X_soh_train.shape)
print("SOH Testing Shape:", X_soh_test.shape)


# Step 6.5 — Verify battery separation

# Get unique battery IDs from both sets
train_batteries = set(groups.iloc[train_idx])
test_batteries = set(groups.iloc[test_idx])

# groups mein humne saare battery_id store kiye the.
# train_idx mein training rows ke indexes hain.
# .iloc[train_idx] sirf training rows ke battery IDs nikalta hai.
# set() duplicate IDs remove kar deta hai.

# Find common batteries
common_batteries = train_batteries.intersection(test_batteries)

print("Training Batteries:", len(train_batteries))
print("Testing Batteries:", len(test_batteries))
print("Common Batteries:", len(common_batteries))


# Step 6.6 — Split RUL data

# Split RUL data
X_rul_train = X_rul.iloc[train_idx]
X_rul_test = X_rul.iloc[test_idx]

y_rul_train = y_rul.iloc[train_idx]
y_rul_test = y_rul.iloc[test_idx]

print("RUL Training Shape:", X_rul_train.shape)
print("RUL Testing Shape:", X_rul_test.shape)


# ====== Feature Preprocessing ======

# Step 6.7 — Separate numerical and categorical features

# Identify numerical features
numerical_features = X_soh_train.select_dtypes(
    include=["number"]
).columns.tolist()

# Identify categorical features
categorical_features = X_soh_train.select_dtypes(
    include=["object", "str"]
).columns.tolist()

print("Numerical Features:", numerical_features)
print("\nCategorical Features:", categorical_features)

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Numerical preprocessing pipeline
numerical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical preprocessing pipeline
categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine numerical and categorical preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)

# Step 6.9 — Apply Preprocessing to SOH and RUL

# Fit preprocessing on SOH training data
X_soh_train_processed = preprocessor.fit_transform(X_soh_train)

# Apply same preprocessing to SOH testing data
X_soh_test_processed = preprocessor.transform(X_soh_test)

print("SOH Training Processed Shape:", X_soh_train_processed.shape)
print("SOH Testing Processed Shape:", X_soh_test_processed.shape)

## Step 6.9.2 — RUL preprocessing

# Identify RUL numerical features
rul_numerical_features = X_rul_train.select_dtypes(
    include=["number"]
).columns.tolist()

# Identify RUL categorical features
rul_categorical_features = X_rul_train.select_dtypes(
    include=["object", "str"]
).columns.tolist()

# Create RUL preprocessor
rul_preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, rul_numerical_features),
        ("cat", categorical_pipeline, rul_categorical_features)
    ]
)

# Fit and transform RUL training data
X_rul_train_processed = rul_preprocessor.fit_transform(X_rul_train)

# Transform RUL testing data
X_rul_test_processed = rul_preprocessor.transform(X_rul_test)

print("RUL Training Processed Shape:", X_rul_train_processed.shape)
print("RUL Testing Processed Shape:", X_rul_test_processed.shape)
# fit() → Training data se median, mode aur scaling parameters learn karta hai.
# transform() → Learned parameters apply karta hai.
# fit_transform() → Dono operations ek saath karta hai.

# Step 6.12 — Check missing values after preprocessing

# Step 6.12 — Check missing values after preprocessing

import numpy as np

print("\n===== MISSING VALUES CHECK =====")
# np.isnan() checks for missing numerical values (NaN).
print("SOH Training:", np.isnan(X_soh_train_processed).sum())
print("SOH Testing:", np.isnan(X_soh_test_processed).sum())

print("RUL Training:", np.isnan(X_rul_train_processed).sum())
print("RUL Testing:", np.isnan(X_rul_test_processed).sum())