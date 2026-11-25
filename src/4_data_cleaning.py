
# ==================================================
# EV BATTERY HEALTH PREDICTOR
# STEP 4: DATA CLEANING
# ==================================================

import pandas as pd

# STEP 1: Load original dataset
data = pd.read_csv("data/ev_battery_health.csv")

# STEP 2: Create a separate copy
# Original dataset safe rahega
cleaned_data = data.copy()

# STEP 3: Check dataset shape
print("\n===== ORIGINAL DATASET SHAPE =====")
print(data.shape)

# # STEP 4: Check duplicate records
# print("\n===== DUPLICATE RECORDS =====")
# print("Total duplicates:", cleaned_data.duplicated().sum())

# # STEP 5: Check missing values
# print("\n===== MISSING VALUES =====")
# print(cleaned_data.isnull().sum())

# # STEP 6: Calculate missing value percentage
# print("\n===== MISSING VALUE PERCENTAGE =====")

# missing_percentage = (
#     cleaned_data.isnull().sum() / len(cleaned_data)
# ) * 100

# print(
#     missing_percentage[missing_percentage > 0]
#     .sort_values(ascending=False)
# )

# ==================================================
# ==================================================

# STEP 7: INVESTIGATE MISSING SERVICE DATES

print("\n===== SERVICE DATE ANALYSIS =====")

# Identify rows where service date is missing
missing_service_date = cleaned_data["last_service_date"].isnull()

# Count missing and available service dates
print("\nMissing service dates:")
print(missing_service_date.sum())

print("\nAvailable service dates:")
print((~missing_service_date).sum())


# Check service count for missing service dates
print("\n===== SERVICE COUNT WHEN DATE IS MISSING =====")

print(
    cleaned_data.loc[
        missing_service_date, "service_count"
    ].value_counts(dropna=False).sort_index()
)

# STEP 8: SERVICE COUNT WHEN DATE IS AVAILABLE

available_service_date = cleaned_data["last_service_date"].notnull()

print("\n===== SERVICE COUNT WHEN DATE IS AVAILABLE =====")

print(
    cleaned_data.loc[
        available_service_date, "service_count"
    ].value_counts(dropna=False).sort_index()
)


# STEP 9: HANDLE MISSING NUMERICAL VALUES

# Identify numerical columns
numerical_columns = cleaned_data.select_dtypes(
    include=["int64", "float64"]
).columns

# Fill missing values using median
for column in numerical_columns:

    # Target columns ko modify nahi karna
    if column not in [
        "state_of_health_pct",
        "remaining_useful_life_cycles"
    ]:

        median_value = cleaned_data[column].median()

        cleaned_data[column] = cleaned_data[column].fillna(
            median_value
        )

# Check missing numerical values after filling
print("\n===== NUMERICAL MISSING VALUES AFTER CLEANING =====")

print(cleaned_data[numerical_columns].isnull().sum())


# STEP 10: CHECK MISSING CATEGORICAL VALUES

categorical_columns = cleaned_data.select_dtypes(
    include=["object", "str"]
).columns

print("\n===== CATEGORICAL MISSING VALUES =====")

print(cleaned_data[categorical_columns].isnull().sum())

# STEP 11: HANDLE MISSING CATEGORICAL VALUES

columns_to_fill = [
    "chemistry",
    "manufacturer",
    "vehicle_model",
    "climate_zone",
    "bms_firmware_version"
]

for column in columns_to_fill:

    mode_value = cleaned_data[column].mode()[0]

    cleaned_data[column] = cleaned_data[column].fillna(mode_value)

# Check missing values after filling
print("\n===== CATEGORICAL VALUES AFTER CLEANING =====")

print(cleaned_data[columns_to_fill].isnull().sum())