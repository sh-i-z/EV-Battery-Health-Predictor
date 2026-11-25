
# ==================================================
# EV BATTERY HEALTH PREDICTOR
# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
# ==================================================

import pandas as pd
import matplotlib.pyplot as plt

# STEP 1: Load dataset
data = pd.read_csv("data/ev_battery_health.csv")


# STEP 2: Statistical Analysis
print("\n===== STATISTICAL SUMMARY =====")
print(data.describe())
data["chemistry"].describe()

# STEP 3: SOH DISTRIBUTION HISTOGRAM

plt.figure(figsize=(8, 5))

plt.hist(
    data["state_of_health_pct"],
    bins=30,
    edgecolor="pink"
)

plt.title("Distribution of Battery State of Health")
plt.xlabel("State of Health (%)")
plt.ylabel("Number of Observations")

# plt.tight_layout()
# plt.show()

# STEP 4: BATTERY AGE VS SOH

plt.figure(figsize=(8, 5))

plt.scatter(
    data["battery_age_days"],
    data["state_of_health_pct"],
    alpha=0.3
)

plt.title("Battery Age vs State of Health")
plt.xlabel("Battery Age (Days)")
plt.ylabel("State of Health (%)")

plt.tight_layout()
# plt.show()


# STEP 5: CYCLE COUNT VS SOH

plt.figure(figsize=(8, 5))

plt.scatter(
    data["cycle_count"],
    data["state_of_health_pct"],
    alpha=0.3
)

plt.title("Cycle Count vs State of Health")
plt.xlabel("Number of Charge-Discharge Cycles")
plt.ylabel("State of Health (%)")

plt.tight_layout()
plt.show()


# STEP 6: BATTERY CHEMISTRY DISTRIBUTION

# Count observations for each battery chemistry
chemistry_counts = data["chemistry"].value_counts()

# Create bar chart
plt.figure(figsize=(8, 5))

plt.bar(
    chemistry_counts.index,
    chemistry_counts.values,
    edgecolor="black"
)

plt.title("Distribution of Battery Chemistries")
plt.xlabel("Battery Chemistry")
plt.ylabel("Number of Observations")

plt.tight_layout()
plt.show()

# STEP 7: CORRELATION HEATMAP

import seaborn  as sns 

#Select numerical coloumns
numerical_data = data.select_dtypes(include=["number"])

# Calculate correlation
correlation_matrix = numerical_data.corr()

# Create heatmap
plt.figure(figsize=(14, 10))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
# corr() — Numerical columns ke beech correlation calculate karta hai.
# annot=True — Heatmap ke cells mein correlation values show karta hai.
# cmap="coolwarm" — Colors select karta hai.
# fmt=".2f" — Values ko 2 decimal places tak show karta hai.

plt.title("Correlation Heatmap of EV Battery Dataset")

plt.tight_layout()
plt.show()
