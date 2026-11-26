import numpy as np
from sklearn.linear_model import LinearRegression

# Load preprocessed data
data = np.load("data/processed_data.npz")  #numpy: Preprocessed .npz data load karne ke liye.

# SOH training data
X_train = data["X_soh_train"]
y_train = data["y_soh_train"]

# SOH testing data
X_test = data["X_soh_test"]
y_test = data["y_soh_test"]

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# Create Linear Regression model
model = LinearRegression()