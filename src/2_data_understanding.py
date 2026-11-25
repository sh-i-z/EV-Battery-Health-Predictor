
# ==================================================
# EV BATTERY HEALTH PREDICTOR
# STEP 2: DATA UNDERSTANDING
# ==================================================

import pandas as pd

# STEP 1: Load the dataset
data = pd.read_csv("data/ev_battery_health.csv")


# STEP 2: Display first 5 records
print("\n===== FIRST 5 RECORDS =====")
print(data.head())


# STEP 3: Display last 5 records
print("\n===== LAST 5 RECORDS =====")
print(data.tail())


# STEP 4: Dataset dimensions
print("\n===== DATASET SHAPE =====")
print("Rows:", data.shape[0])
print("Columns:", data.shape[1])


# STEP 5: Column names
print("\n===== COLUMN NAMES =====")
print(data.columns.tolist())


# STEP 6: Dataset information
print("\n===== DATASET INFORMATION =====")
data.info()


# STEP 7: Data types
print("\n===== DATA TYPES =====")
print(data.dtypes)


# STEP 8: Missing values
print("\n===== MISSING VALUES =====")
print(data.isnull().sum())


# STEP 9: Missing value percentage
print("\n===== MISSING VALUE PERCENTAGE =====")
missing_percentage = (data.isnull().sum() / len(data)) * 100
print(missing_percentage[missing_percentage > 0].sort_values(ascending=False))


# STEP 10: Duplicate records
print("\n===== DUPLICATE RECORDS =====")
print("Total duplicates:", data.duplicated().sum())


# STEP 11: Unique values in categorical columns
print("\n===== UNIQUE CATEGORICAL VALUES =====")

categorical_columns = data.select_dtypes(include=["object"]).columns

for column in categorical_columns:
    print(f"\n{column}:")
    print("Unique values:", data[column].nunique())


# ==================================================
# END OF STEP 2
# ==================================================
