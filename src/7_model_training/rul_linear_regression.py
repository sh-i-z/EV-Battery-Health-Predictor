
import numpy as np
import joblib

from sklearn.linear_model import LinearRegression

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

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

print("\nRUL Linear Regression Model Trained Successfully!")

# Generate predictions on testing data
y_pred = model.predict(X_test)

print("\nFirst 10 Actual RUL Values:")
print(y_test[:10])

print("\nFirst 10 Predicted RUL Values:")
print(y_pred[:10])

# Save trained model
joblib.dump(
    model,
    "data/rul_linear_regression.pkl"
)

print("\nRUL Linear Regression Model Saved Successfully!")
