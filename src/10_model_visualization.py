
import numpy as np
import joblib
import matplotlib.pyplot as plt


# ===== STEP 1: LOAD TESTING DATA =====

data = np.load("data/processed_data.npz")

# SOH testing data
X_soh_test = data["X_soh_test"]
y_soh_test = data["y_soh_test"]

# RUL testing data
X_rul_test = data["X_rul_test"]
y_rul_test = data["y_rul_test"]


# ===== STEP 2: LOAD TRAINED MODELS =====

soh_model = joblib.load("data/soh_random_forest.pkl")
rul_model = joblib.load("data/rul_random_forest.pkl")


# ===== STEP 3: GENERATE PREDICTIONS =====

soh_pred = soh_model.predict(X_soh_test)
rul_pred = rul_model.predict(X_rul_test)


# ===== STEP 4: ACTUAL VS PREDICTED GRAPHS =====

# SOH Graph
plt.figure(figsize=(10, 5))

plt.plot(y_soh_test[:100], label="Actual SOH")
plt.plot(soh_pred[:100], label="Predicted SOH")

plt.title("SOH: Actual vs Predicted")
plt.xlabel("Test Samples")
plt.ylabel("State of Health (%)")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("data/soh_actual_vs_predicted.png")
plt.show()


# RUL Graph
plt.figure(figsize=(10, 5))

plt.plot(y_rul_test[:100], label="Actual RUL")
plt.plot(rul_pred[:100], label="Predicted RUL")

plt.title("RUL: Actual vs Predicted")
plt.xlabel("Test Samples")
plt.ylabel("Remaining Useful Life (Cycles)")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("data/rul_actual_vs_predicted.png")
plt.show()


# ===== STEP 5: CALCULATE RESIDUALS =====

# Residual = Actual - Predicted

soh_residuals = y_soh_test - soh_pred
rul_residuals = y_rul_test - rul_pred


# ===== STEP 6: SOH RESIDUAL ANALYSIS =====

plt.figure(figsize=(10, 5))

plt.scatter(soh_pred, soh_residuals, alpha=0.5)

# Zero error reference line
plt.axhline(y=0, color="red", linestyle="--")

plt.title("SOH Residual Analysis")
plt.xlabel("Predicted SOH")
plt.ylabel("Residuals")

plt.grid(True)
plt.tight_layout()

plt.savefig("data/soh_residuals.png")
plt.show()


# ===== STEP 7: RUL RESIDUAL ANALYSIS =====

plt.figure(figsize=(10, 5))

plt.scatter(rul_pred, rul_residuals, alpha=0.5)

# Zero error reference line
plt.axhline(y=0, color="red", linestyle="--")

plt.title("RUL Residual Analysis")
plt.xlabel("Predicted RUL")
plt.ylabel("Residuals")

plt.grid(True)
plt.tight_layout()

plt.savefig("data/rul_residuals.png")
plt.show()


print("\nAll visualization and residual analysis graphs saved successfully!")
