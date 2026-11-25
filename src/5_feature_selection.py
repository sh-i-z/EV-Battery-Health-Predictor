import pandas as pd

# Load cleaned dataset
data = pd.read_csv("data/ev_battery_health_cleaned.csv")

# Display all column names
print("\n===== DATASET COLUMNS =====")
# convert colms to list
print(data.columns.tolist())

# Display dataset shape
print("\n===== DATASET SHAPE =====")

print(data.shape)

# STEP 5.2: INSPECT TARGET COLUMNS

target_columns = [
    "state_of_health_pct",
    "remaining_useful_life_cycles"
]

print("\n===== TARGET COLUMN INFORMATION =====")

print(data[target_columns].describe())

# STEP 5.3: INSPECT TARGET-RELATED COLUMNS

print("\n===== TARGET RELATED COLUMNS =====")

print(
    data[
        [
            # step ka purpose specifically target leakage identify karna
            "capacity_fade_pct",
            "state_of_health_pct",
            "remaining_useful_life_cycles",
            "failure_risk_label",
            "degradation_category"
        ]
    ].head(10)
)

# STEP 5.4: CHECK TARGET-RELATED LABEL DISTRIBUTION

print("\n===== FAILURE RISK LABEL DISTRIBUTION =====")

print(data["failure_risk_label"].value_counts())  # .values_count() : Isse humein labels ki distribution samajh aayegi.

print("\n===== DEGRADATION CATEGORY DISTRIBUTION =====")

print(data["degradation_category"].value_counts())

# STEP 5.5: CHECK CAPACITY FADE AND SOH RELATIONSHIP

print("\n===== CAPACITY FADE AND SOH RELATIONSHIP =====")

print(
    data["capacity_fade_pct"].corr(
        data["state_of_health_pct"]
    )
)

# STEP 5.6: DEFINE COLUMNS TO EXCLUDE

columns_to_exclude = [
    "battery_id",
    "observation_id",
    "capacity_fade_pct",
    "state_of_health_pct",
    "remaining_useful_life_cycles",
    "failure_risk_label",
    "degradation_category"
]

print("\n===== COLUMNS TO EXCLUDE =====")

print(columns_to_exclude)

# STEP 5.7: DEFINE FEATURES AND TARGETS

# Columns not used as model inputs
columns_to_exclude = [
    "battery_id",
    "observation_id",
    "capacity_fade_pct",
    "state_of_health_pct",
    "remaining_useful_life_cycles",
    "failure_risk_label",
    "degradation_category"
]

# SOH target
y_soh = data["state_of_health_pct"]

# RUL target
y_rul = data["remaining_useful_life_cycles"]

# Common input features
X = data.drop(columns=columns_to_exclude)

print("\n===== FEATURE DATASET =====")
print(X.shape)

print("\n===== SOH TARGET =====")
print(y_soh.shape)

print("\n===== RUL TARGET =====")
print(y_rul.shape)

# STEP 5.8: DISPLAY SELECTED FEATURES

print("\n===== SELECTED INPUT FEATURES =====")

print(X.columns.tolist())

# STEP 5.9: IDENTIFY NUMERICAL AND CATEGORICAL FEATURES

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "str"]
).columns.tolist()

print("\n===== NUMERICAL FEATURES =====")
print(numerical_features)

print("\n===== CATEGORICAL FEATURES =====")
print(categorical_features)

print("\nTotal numerical features:", len(numerical_features))
print("Total categorical features:", len(categorical_features))

# STEP 5.10: IDENTIFY DATE FEATURES

date_features = [
    "manufacture_date",
    "observation_date",
    "last_service_date"
]

categorical_features = [
    column for column in categorical_features
    if column not in date_features
]

print("\n===== DATE FEATURES =====")
print(date_features)

print("\n===== ACTUAL CATEGORICAL FEATURES =====")
print(categorical_features)

# STEP 5.11: CONVERT DATE FEATURES TO DATETIME

for column in date_features:

    data[column] = pd.to_datetime(
        data[column],
        errors="coerce"
    )

print("\n===== DATE FEATURES DATA TYPES =====")

print(data[date_features].dtypes)

# STEP 5.12: CREATE DAYS SINCE LAST SERVICE

data["days_since_last_service"] = (
    data["observation_date"] - data["last_service_date"]
).dt.days

print("\n===== DAYS SINCE LAST SERVICE =====")

print(data["days_since_last_service"].describe())

print("\nMissing values:")

print(data["days_since_last_service"].isnull().sum())

# STEP 5.13: CHECK NEGATIVE SERVICE DAYS

negative_service_days = data[
    data["days_since_last_service"] < 0
]

print("\n===== NEGATIVE SERVICE DAYS =====")

print(len(negative_service_days))

# STEP 5.14: REMOVE RAW DATE COLUMNS

X = X.drop(
    columns=[
        "manufacture_date",
        "observation_date",
        "last_service_date"
    ]
)

# Add the newly created feature
X["days_since_last_service"] = data["days_since_last_service"]

print("\n===== UPDATED FEATURE DATASET =====")
print(X.shape)

print("\n===== UPDATED FEATURES =====")
print(X.columns.tolist())

# STEP 5.15: CREATE SEPARATE FEATURE SETS

# SOH prediction features
X_soh = X.copy()

# RUL prediction features
X_rul = X.copy()

# Targets
y_soh = data["state_of_health_pct"]
y_rul = data["remaining_useful_life_cycles"]

print("\n===== SOH FEATURES =====")
print(X_soh.shape)

print("\n===== RUL FEATURES =====")
print(X_rul.shape)

# Check relationship between capacity fade and RUL
correlation = data["capacity_fade_pct"].corr(
    data["remaining_useful_life_cycles"]
)

print("Capacity Fade vs RUL Correlation:", correlation)

# Add capacity fade only to RUL features
X_rul["capacity_fade_pct"] = data["capacity_fade_pct"]

print("SOH FEATURES:", X_soh.shape)
print("RUL FEATURES:", X_rul.shape)

print("===== SOH MISSING VALUES =====")
print(X_soh.isnull().sum().sum())

print("\n===== RUL MISSING VALUES =====")
print(X_rul.isnull().sum().sum())

print("===== SOH MISSING VALUES =====")
print(X_soh.isnull().sum()[X_soh.isnull().sum() > 0])

print("\n===== RUL MISSING VALUES =====")
print(X_rul.isnull().sum()[X_rul.isnull().sum() > 0])

print("===== FINAL FEATURE SUMMARY =====")

print("SOH input features:", len(X_soh.columns))
print("RUL input features:", len(X_rul.columns))

print("SOH target:", y_soh.name)
print("RUL target:", y_rul.name)

print("\nSOH dataset shape:", X_soh.shape)
print("RUL dataset shape:", X_rul.shape)