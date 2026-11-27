
import numpy as np
import joblib

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Load preprocessed testing data
data = np.load("data/processed_data.npz")


# Function to evaluate a model
def evaluate_model(model, X_test, y_test, model_name):

    # Generate predictions
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Display results
    print(f"\n{model_name}")
    print("-" * 40)
    print("MAE:", mae)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("R2 Score:", r2)


# Load SOH testing data
X_soh_test = data["X_soh_test"]
y_soh_test = data["y_soh_test"]

# Load RUL testing data
X_rul_test = data["X_rul_test"]
y_rul_test = data["y_rul_test"]


# Load trained models
soh_lr = joblib.load("data/soh_linear_regression.pkl")
soh_rf = joblib.load("data/soh_random_forest.pkl")

rul_lr = joblib.load("data/rul_linear_regression.pkl")
rul_rf = joblib.load("data/rul_random_forest.pkl")


# Evaluate SOH models
print("\n========== SOH MODEL EVALUATION ==========")

evaluate_model(
    soh_lr,
    X_soh_test,
    y_soh_test,
    "SOH Linear Regression"
)

evaluate_model(
    soh_rf,
    X_soh_test,
    y_soh_test,
    "SOH Random Forest"
)


# Evaluate RUL models
print("\n========== RUL MODEL EVALUATION ==========")

evaluate_model(
    rul_lr,
    X_rul_test,
    y_rul_test,
    "RUL Linear Regression"
)

evaluate_model(
    rul_rf,
    X_rul_test,
    y_rul_test,
    "RUL Random Forest"
)





#===================================

import numpy as np
import joblib
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Load preprocessed testing data
data = np.load("data/processed_data.npz")

# Load testing data
X_soh_test = data["X_soh_test"]
y_soh_test = data["y_soh_test"]

X_rul_test = data["X_rul_test"]
y_rul_test = data["y_rul_test"]


# Load trained models
soh_lr = joblib.load("data/soh_linear_regression.pkl")
soh_rf = joblib.load("data/soh_random_forest.pkl")

rul_lr = joblib.load("data/rul_linear_regression.pkl")
rul_rf = joblib.load("data/rul_random_forest.pkl")


# Store evaluation results
results = []


# Function to evaluate models
def evaluate_model(model, X_test, y_test, model_name, target):

    # Generate predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Display results
    print(f"\n{model_name}")
    print("-" * 40)
    print("MAE:", mae)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("R2 Score:", r2)

    # Store results
    results.append({
        "Target": target,
        "Model": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2
    })


# Evaluate SOH models
print("\n========== SOH MODEL EVALUATION ==========")

evaluate_model(
    soh_lr, X_soh_test, y_soh_test,
    "Linear Regression", "SOH"
)

evaluate_model(
    soh_rf, X_soh_test, y_soh_test,
    "Random Forest", "SOH"
)


# Evaluate RUL models
print("\n========== RUL MODEL EVALUATION ==========")

evaluate_model(
    rul_lr, X_rul_test, y_rul_test,
    "Linear Regression", "RUL"
)

evaluate_model(
    rul_rf, X_rul_test, y_rul_test,
    "Random Forest", "RUL"
)


# Convert results into DataFrame
results_df = pd.DataFrame(results)

# Save results as CSV
results_df.to_csv(
    "data/model_evaluation_results.csv",
    index=False
)

print("\nEvaluation results saved successfully!")



