
import pandas as pd
import joblib

# Load cleaned dataset
data = pd.read_csv("data/ev_battery_health_cleaned.csv")

# Select one sample battery
sample = data.iloc[[0]].copy()

# Convert date columns
date_features = [
    "manufacture_date",
    "observation_date",
    "last_service_date"
]

for column in date_features:
    sample[column] = pd.to_datetime(
        sample[column], errors="coerce"
    )

# Create days since last service
sample["days_since_last_service"] = (
    sample["observation_date"] -
    sample["last_service_date"]
).dt.days


# Prepare SOH input
X_soh = sample.drop(columns=[
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

# Prepare RUL input
X_rul = X_soh.copy()
X_rul["capacity_fade_pct"] = sample["capacity_fade_pct"]


# Load preprocessing pipelines
soh_preprocessor = joblib.load(
    "data/soh_preprocessor.pkl"
)

rul_preprocessor = joblib.load(
    "data/rul_preprocessor.pkl"
)


# Transform sample input
X_soh_processed = soh_preprocessor.transform(X_soh)
X_rul_processed = rul_preprocessor.transform(X_rul)


# Load trained Random Forest models
soh_model = joblib.load(
    "data/soh_random_forest.pkl"
)

rul_model = joblib.load(
    "data/rul_random_forest.pkl"
)


# Generate predictions
soh_prediction = soh_model.predict(
    X_soh_processed
)

rul_prediction = rul_model.predict(
    X_rul_processed
)


# Display predictions
print("\n===== EV BATTERY HEALTH PREDICTION =====")

print("Predicted SOH:",
      round(soh_prediction[0], 2), "%")

print("Predicted RUL:",
      round(rul_prediction[0], 2), "cycles")


# Display actual values for comparison
print("\n===== ACTUAL VALUES =====")

print("Actual SOH:",
      sample["state_of_health_pct"].iloc[0], "%")

print("Actual RUL:",
      sample["remaining_useful_life_cycles"].iloc[0],
      "cycles")
