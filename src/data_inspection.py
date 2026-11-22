import pandas as pd

data = pd.read_csv("data/ev_battery_health.csv")

print("===== FIRST 5 ROWS =====")
print(data.head())
print(data.shape)
print(data.columns)

print("\n===== DATASET INFORMATION =====")
# data.info()

print("\n===== END =====")