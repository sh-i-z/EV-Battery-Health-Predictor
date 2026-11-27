
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load preprocessed testing data
data = np.load("data/processed_data.npz")

# Load SOH testing data
X_soh_test = data["X_soh_test"]
y_soh_test = data["y_soh_test"]

# Load RUL testing data
X_rul_test = data["X_rul_test"]
y_rul_test = data["y_rul_test"]

# Load trained Random Forest models
soh_model = joblib.load("data/soh_random_forest.pkl")
rul_model = joblib.load("data/rul_random_forest.pkl")

# Generate predictions
soh_pred = soh_model.predict(X_soh_test)
rul_pred = rul_model.predict(X_rul_test)


# SOH Actual vs Predicted Graph
plt.figure(figsize=(10, 5))

plt.plot(y_soh_test[:100], label="Actual SOH")
plt.plot(soh_pred[:100], label="Predicted SOH")

plt.title("SOH: Actual vs Predicted")
plt.xlabel("Test Samples")
plt.ylabel("State of Health (%)")
plt.legend()
plt.grid(True)

plt.savefig("data/soh_actual_vs_predicted.png")
plt.show()


# RUL Actual vs Predicted Graph
plt.figure(figsize=(10, 5))

plt.plot(y_rul_test[:100], label="Actual RUL")
plt.plot(rul_pred[:100], label="Predicted RUL")

plt.title("RUL: Actual vs Predicted")
plt.xlabel("Test Samples")
plt.ylabel("Remaining Useful Life (Cycles)")
plt.legend()
plt.grid(True)

plt.savefig("data/rul_actual_vs_predicted.png")
plt.show()

print("Visualization completed successfully!")
