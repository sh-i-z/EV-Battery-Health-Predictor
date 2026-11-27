
import numpy as np
import joblib

# Load trained Random Forest models
soh_model = joblib.load("data/soh_random_forest.pkl")
rul_model = joblib.load("data/rul_random_forest.pkl")

# Load preprocessed data
data = np.load("data/processed_data.npz")

# Get feature importance values
soh_importance = soh_model.feature_importances_
rul_importance = rul_model.feature_importances_

# Display feature importance array sizes
print("SOH Features:", len(soh_importance))
print("RUL Features:", len(rul_importance))


# Get processed feature names
soh_feature_names = data["soh_feature_names"]
rul_feature_names = data["rul_feature_names"]


# Create feature importance tables
soh_results = sorted(
    zip(soh_feature_names, soh_importance),
    key=lambda x: x[1],
    reverse=True
)

rul_results = sorted(
    zip(rul_feature_names, rul_importance),
    key=lambda x: x[1],
    reverse=True
)


# Display Top 10 SOH features
print("\n========== TOP 10 SOH FEATURES ==========")

for feature, importance in soh_results[:10]:
    print(f"{feature}: {importance:.4f}")


# Display Top 10 RUL features
print("\n========== TOP 10 RUL FEATURES ==========")

for feature, importance in rul_results[:10]:
    print(f"{feature}: {importance:.4f}")


import matplotlib.pyplot as plt

# Extract Top 10 SOH features
soh_top10 = soh_results[:10]

soh_names = [item[0] for item in soh_top10]
soh_values = [item[1] for item in soh_top10]

# Plot SOH feature importance
plt.figure(figsize=(10, 6))

plt.barh(soh_names[::-1], soh_values[::-1])

plt.title("Top 10 SOH Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")

plt.tight_layout()
plt.savefig("data/soh_feature_importance.png")
plt.show()


# Extract Top 10 RUL features
rul_top10 = rul_results[:10]

rul_names = [item[0] for item in rul_top10]
rul_values = [item[1] for item in rul_top10]

# Plot RUL feature importance
plt.figure(figsize=(10, 6))

plt.barh(rul_names[::-1], rul_values[::-1])

plt.title("Top 10 RUL Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")

plt.tight_layout()
plt.savefig("data/rul_feature_importance.png")
plt.show()

print("Feature importance graphs saved successfully!")
    


