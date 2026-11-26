
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor

# Load preprocessed data
data = np.load("data/processed_data.npz")

# RUL training data
X_train = data["X_rul_train"]
y_train = data["y_rul_train"]

# RUL testing data
X_test = data["X_rul_test"]
y_test = data["y_rul_test"]

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# Create Random Forest Regressor model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# Train the model
model.fit(X_train, y_train)

print("\nRUL Random Forest Model Trained Successfully!")

# Generate predictions on testing data
y_pred = model.predict(X_test)

print("\nFirst 10 Actual RUL Values:")
print(y_test[:10])

print("\nFirst 10 Predicted RUL Values:")
print(y_pred[:10])

# Save trained model
joblib.dump(
    model,
    "data/rul_random_forest.pkl"
)

print("\nRUL Random Forest Model Saved Successfully!")
