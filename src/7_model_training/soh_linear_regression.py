import numpy as np
from sklearn.linear_model import LinearRegression

# Load preprocessed data
data = np.load("data/processed_data.npz")

# SOH training data
X_train = data["X_soh_train"]
y_train = data["y_soh_train"]

# SOH testing data
X_test = data["X_soh_test"]
y_test = data["y_soh_test"]

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

print("SOH Linear Regression Model Trained Successfully!")

# Generate predictions on testing data
y_pred = model.predict(X_test)

print("\nFirst 10 Actual SOH Values:")
print(y_test[:10])

print("\nFirst 10 Predicted SOH Values:")
print(y_pred[:10])

import joblib

# Save trained model
joblib.dump(
    model,
    "data/soh_linear_regression.pkl"
)

print("SOH Linear Regression Model Saved Successfully!")