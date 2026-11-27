# 🔋 EV Battery Health Predictor

**Predicting State of Health (SOH) and Remaining Useful Life (RUL) of Electric Vehicle Batteries using Machine Learning.**

A machine learning project built on a physics-informed synthetic dataset containing **50,000 observations across 2,600 EV batteries**, covering four cell chemistries: LFP, NMC, NCA, and LMO.

The project focuses on battery degradation analysis, feature importance, regression model comparison, and battery health prediction.

---

## 📌 Overview

Electric vehicle adoption is accelerating worldwide, making battery health monitoring increasingly important.

The battery pack is one of the most expensive components of an electric vehicle. Its degradation depends on several factors, including battery chemistry, climate, charging habits, driving patterns, and maintenance history.

This creates two important applications:

- **Predictive Maintenance:** Estimating battery degradation and identifying batteries that may require attention.
- **Battery Second-Life Economy:** Understanding when batteries approach the end of their useful automotive life and may be considered for alternative applications.

This project uses Machine Learning regression models to predict two important battery health indicators:

- **State of Health (SOH):** Current battery health expressed as a percentage.
- **Remaining Useful Life (RUL):** Estimated number of remaining charge-discharge cycles until the battery reaches 80% SOH.

The project compares Linear Regression and Random Forest Regression models for both prediction tasks.

## 🎯 Project Objectives

- Understand EV battery degradation data.
- Perform exploratory data analysis and data visualization.
- Clean and preprocess the dataset.
- Select relevant features for SOH and RUL prediction.
- Train multiple machine learning regression models.
- Evaluate and compare model performance.
- Identify important features using Random Forest feature importance.
- Visualize actual versus predicted values.
- Analyze prediction errors using residual plots.
- Generate SOH and RUL predictions using trained models.

## 📊 Dataset Information

The project uses a fully synthetic, physics-informed EV battery health dataset.

| Dataset Information | Details |
|---|---|
| Main dataset | `ev_battery_health.csv` |
| Total observations | 50,000 |
| Total columns | 35 |
| Unique battery packs | 2,600 |
| Cell chemistries | LFP, NMC, NCA, LMO |
| Data type | Synthetic longitudinal data |
| Random seed | 42 |

### Dataset Files

| File | Description |
|---|---|
| `ev_battery_health.csv` | Original dataset containing 50,000 battery observations |
| `ev_battery_health_cleaned.csv` | Cleaned dataset used for further analysis |
| `data_dictionary.csv` | Column names, data types, units, descriptions, and example values |
| `processed_data.npz` | Preprocessed training and testing arrays |

Each battery has multiple longitudinal observations identified by `battery_id` and `observation_id`.

## 🗂️ Column Descriptions

The dataset contains 35 columns covering battery characteristics, usage patterns, environmental conditions, electrical measurements, maintenance information, and prediction targets.

| Column | Type | Unit | Description |
|---|---|---|---|
| battery_id | String | - | Unique battery pack identifier |
| observation_id | String | - | Unique observation identifier |
| chemistry | Categorical | - | LFP / NMC / NCA / LMO |
| manufacturer | Categorical | - | Synthetic pack manufacturer |
| vehicle_model | Categorical | - | EV model or platform |
| pack_capacity_kwh | Float | kWh | Nominal rated capacity |
| manufacture_date | Date | - | Battery manufacturing date |
| observation_date | Date | - | Date of observation |
| battery_age_days | Integer | Days | Battery age at observation |
| cycle_count | Integer | Cycles | Cumulative equivalent full cycles |
| climate_zone | Categorical | - | Operating climate |
| avg_daily_distance_km | Float | km | Average daily driving distance |
| avg_charging_power_kw | Float | kW | Average charging power |
| fast_charge_ratio | Float | 0–1 | Share of DC fast-charge sessions |
| avg_depth_of_discharge_pct | Float | % | Average depth of discharge |
| avg_state_of_charge_pct | Float | % | Average state of charge |
| charge_cycles_per_month | Float | /month | Charging frequency |
| total_energy_throughput_kwh | Float | kWh | Cumulative energy throughput |
| avg_ambient_temp_c | Float | °C | Average ambient temperature |
| humidity_pct | Float | % | Average humidity |
| altitude_m | Float | m | Operating altitude |
| internal_resistance_mohm | Float | mΩ | Internal resistance |
| voltage_avg_v | Float | V | Average pack voltage |
| current_avg_a | Float | A | Average current draw |
| charge_time_avg_min | Float | min | Average charging duration |
| peak_temp_during_fast_charge_c | Float | °C | Peak fast-charge temperature |
| capacity_fade_pct | Float | % | Cumulative capacity fade |
| service_count | Integer | Count | Maintenance events |
| last_service_date | Date | - | Most recent service date |
| bms_firmware_version | Categorical | - | Battery Management System firmware |
| recall_flag | Binary | 0/1 | Manufacturer recall indicator |
| state_of_health_pct | Float | % | Current capacity relative to nominal capacity |
| remaining_useful_life_cycles | Integer | Cycles | Cycles remaining until 80% SOH |
| failure_risk_label | Categorical | - | Low / Medium / High risk |
| degradation_category | Categorical | - | Normal / Accelerated / Critical |

**Note:** This implementation focuses on regression tasks using SOH and RUL as target variables. Failure risk and degradation category are included in the dataset but classification models have not yet been implemented.

Full column information is available in `data_dictionary.csv`.

## 🎯 Suggested Dataset Use Cases

The dataset supports several possible machine learning applications:

- **Regression:** Predict SOH and RUL.
- **Classification:** Predict failure risk and degradation category.
- **Anomaly Detection:** Identify unusual sensor readings in resistance, temperature, and charging time.
- **Time-Series Forecasting:** Study battery degradation over cycle count and battery age.
- **Clustering:** Group batteries based on usage and degradation patterns.
- **Feature Importance / SHAP:** Investigate which features influence model predictions.

## ⚙️ Dataset Methodology

The dataset is fully synthetic and generated using Python with Pandas and NumPy, using random seed 42.

It uses a physics-informed degradation model:

- Four battery chemistries have different degradation parameters, temperature sensitivities, depth-of-discharge sensitivities, and fast-charging penalties.
- Each battery has persistent random-effect characteristics, such as manufacturing quality and driver usage intensity.
- SOH follows a nonlinear degradation curve combining cycle-based and calendar-based aging.
- Electrical measurements, including internal resistance, are derived from SOH and chemistry to preserve correlations.
- Missing values are injected into feature columns.
- Outliers are injected into selected sensor columns to simulate telemetry noise.
- Failure risk classes are intentionally imbalanced.

The degradation parameters are loosely informed by published battery-aging research. However, the dataset does not contain actual vehicle or laboratory measurements.

## 🔄 Project Workflow

The project follows a complete machine learning workflow:

1. Data Understanding
2. Exploratory Data Analysis (EDA)
3. Data Cleaning
4. Feature Selection
5. Data Preprocessing
6. Model Training
7. Model Evaluation
8. Feature Importance Analysis
9. Actual vs Predicted Visualization
10. Residual Analysis
11. Final Prediction Testing

## 🧹 Data Preprocessing

The following preprocessing techniques were implemented:

- Numerical missing values handled using