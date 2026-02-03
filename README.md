# Energy Usage Forecasting — Individual Household Electric Power Consumption

**Author:** Faysel Nessro  
**Date:** December 2025  

---
## Table of Contents

1. [Overview](#overview)
2. [Data Understanding](#1-data-understanding)
3. [Data Preparation](#2-data-preparation)
4. [Feature Engineering](#3-feature-engineering)
5. [Modeling](#4-modeling)
6. [Hyperparameter Tuning](#5-hyperparameter-tuning)
7. [Evaluation & Error Analysis](#6-evaluation--error-analysis)
8. [Deployment](#7-deployment)
9. [Results](#8-results)
10. [Skills Demonstrated](#9-skills-demonstrated)
11. [Limitations & Future Work](#10-limitations--future-work)
12. [Setup & Usage Instructions](#12-setup--usage-instructions)
13. [How It Works](#13-how-it-works)
14. [Contributing](#14-contributing)
15. [License](#15-license)


## Overview

This project predicts **hourly household energy consumption** using advanced time-series and machine-learning models.  
It combines **feature engineering**, **classical forecasting models**, and **gradient boosting models** to achieve high accuracy.  
The final model (XGBoost) achieves **RMSE ≈ 0.33**, outperforming Prophet and SARIMA.

**Business Impact:**  
- Optimizing energy pricing  
- Grid load balancing  
- Scheduling renewable energy supply  
- Detecting anomalies in household consumption  

**Problem Type:** Time Series Forecasting (Univariate/Multivariate)  
**Target Variable:** `Global_active_power`  
**Forecast Horizon:** Next 24 hours  

**Success Metrics:**  
| Metric | Target |
|--------|--------|
| RMSE   | ≤ 0.35 |
| MAPE   | ≤ 15%  |
| Runtime & Model Stability | — |

---


## 1. Data Understanding

**Dataset:** UCI Individual Household Electric Power Consumption  

- **Observations:** 2,075,259 (Dec 2006 – Nov 2010)  
- **Features (9):** Global_active_power, Global_reactive_power, Voltage, Global_intensity, Sub_metering_1–3  

**Exploratory Steps:**  
- Load raw data  
- Parse datetime (Date + Time)  
- Handle missing values (`?` → NaN)  
- Resample: 1-min → 1-hour, 1-day  
- Plot seasonality: daily, weekly, yearly patterns  

**Key Observations:**  
- Daily peaks: 7–9am, 6–9pm  
- Weekends differ from weekdays  
- Seasonal effects (higher winter consumption)  
- Missing blocks handled via linear interpolation  

---

## 2. Data Preparation

**Preprocessing Steps:**  
- Fill missing values via interpolation  
- Convert numeric columns  
- Add time features: Hour, Day of Week, Month, Is_Weekend  
- Lag features: 1h, 24h, 7d  
- Rolling windows: 3h, 6h, 12h, 24h  

**Train / Validation / Test Split:**

| Set        | Period       |
| ---------- | ------------ |
| Train      | 2007–2009    |
| Validation | Jan–Apr 2010 |
| Test       | May–Aug 2010 |

---

## 3. Feature Engineering

**Created Features:**  
- Time-based: hour, day, weekday, weekofyear, month, year  
- Lag features: 1, 24, 168 hours  
- Rolling/Window features: mean, std, sum for windows `[3,6,12,24,48,72,96,168]`  
- Other Consumption = `Global_active_power*1000/60 - sub_meterings`  

**Example Rolling Feature Calculation:**  

```python
df['Global_active_power_roll24_mean'] = df['Global_active_power'].rolling(24).mean()
```

## 4. Modeling

**Models Implemented:**

| Model    | Type       | Expected RMSE |
| -------- | ---------- | ------------- |
| Naive    | Baseline   | 0.65          |
| SARIMA   | Classical  | 0.45          |
| Prophet  | Structural | 0.42          |
| XGBoost  | ML         | 0.33          |

**Final Model Choice:** XGBoost with lag + rolling + calendar features.

---

## 5. Hyperparameter Tuning

**Parameters Tuned:**

* learning_rate
* max_depth
* n_estimators
* subsample
* colsample_bytree

**Result:** RMSE improved from 0.36 → 0.33

---

## 6. Evaluation & Error Analysis

***Visualizations:**

* Forecast vs Actual
* Residual plots
* Feature importance (LightGBM/XGBoost)

**Error Analysis:**

* Larger errors during holiday periods

* Outliers handled by rolling mean smoothing

**Example Forecast Plot:**


---
## 7. Deployment

**Streamlit Dashboard:**

* Pages: EDA | Model Comparison | Forecast 24h | Upload CSV | Feature Importance

* Widgets: Model selector, forecast horizon, start date

**Run the app:**

```
streamlit run app.py
```
---

## 8. Results

| Model    | RMSE |
| -------- | ---- |
| Naive    | 0.65 |
| SARIMA   | 0.45 |
| Prophet  | 0.42 |
| XGBoost  | 0.33 |

**Interpretation:** XGBoost achieved the best performance for 24-hour forecasts. Ensemble or hybrid approaches may further improve results.

---
## 9. Skills Demonstrated

* Time series forecasting at scale
* Feature engineering (lags, rolling windows, calendar features)
* Classical models (ARIMA/SARIMA, Prophet)
* Gradient boosting models (LightGBM, XGBoost)
* Hyperparameter tuning & evaluation
* Streamlit deployment for interactive dashboards
---

## 10. Limitations & Future Work

**Limitations:**

* Exogenous variables (holidays, weather) not used

* Extreme spikes not perfectly captured

**Future Work:**

* NeuralProphet or LSTM/GRU for deep learning forecasting

* Global models for multiple households

* Anomaly detection module

  ---

## 11. Repository Structure

```
energy-usage-forecasting/
├──app/
│   └── app.py
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   └── processed/
├──enviroment
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Modeling.ipynb
│   ├── 03_Tuning.ipynb
│   └── 04_Evaluation.ipynb
├── src/
│   ├── preprocess.py
│   ├── features.py
│   ├── models.py
│   └── utils.py
├── results/
│   ├── comparison.png
│   ├── forecast_example.png
│   └── feature_importance.png
└── LICENSE
```
---
## 12. Setup & Usage Instructions
Follow these steps to run the Energy Usage Forecasting app locally:

**Step 1: Clone the Repository**
```
git clone https://github.com/FayselN/energy-usage-forecasting.git
cd energy-usage-forecasting
```
**Step 2: Create a Python Virtual Environment**

```python -m venv env
```
**Step 3: Activate the Virtual Environment**

Windows (PowerShell):
```
.\env\Scripts\Activate
```
**Step 4: Install Dependencies**
```
pip install -r requirements.txt
```
**Step 5: Verify Required Files**

Processed data: data/processed/df_hourly.csv

Trained model: src/xgb_model1.json

**Step 6: Run the Streamlit App**
```
streamlit run app/app.py
```
**Step 7: Using the App**

* Set the forecast horizon in the sidebar (default: 168 hours / 7 days).

* Click “Predict next 7 days (168h)” → forecasts are displayed as charts.

* Download the forecast CSV using the “Download forecast CSV” button.

**Step 8: Stop the App**

* Press Ctrl + C in the terminal.

Note: Only processed data and model files are required. Large raw data (data/raw/household_power_consumption.txt) is not included in the repository.


---
## 13. How It Works
The app forecasts energy usage using a recursive XGBoost model with hourly data:

**1. Data Preprocessing**

* Raw energy data is cleaned, missing values handled, and aggregated to hourly usage.

* Time-based features are created (hour, day, weekday, month, weekend flag).

* Lag and rolling statistics features (lag1, lag24, lag168, rolling mean/std) are added.

**2. Modeling**

* XGBoost is trained on historical energy usage using engineered features.

* Recursive forecasting predicts future values hour-by-hour, using previous predictions as input for the next hour.

**3. Forecast Visualization**

* Streamlit displays a line chart of forecasted energy usage.

* Users can download the forecast for further analysis.

**4. Portfolio Ready**

* Includes notebooks showing EDA, modeling, hyperparameter tuning, and evaluation.

* Interactive dashboard demonstrates ML workflow and forecasting performance.


---
## 14. Contributing
Contributions are welcome and appreciated.

If you would like to contribute to this project, please follow these steps:

**1. Fork the repository**

* Click the Fork button on GitHub.

**2. Clone your fork**
```
git clone https://github.com/your-username/energy-usage-forecasting.git
cd energy-usage-forecasting
```
**3.Make your changes**

* Improve models or feature engineering

* Enhance visualizations or Streamlit UI

* Optimize preprocessing or forecasting logic

* Add tests or documentation

**4. Commit your changes**
```
git commit -m "Add meaningful commit message"
```
**5. Push to your fork**
```
git push origin feature/your-feature-name
```
**6.Open a Pull Request**

   * Describe clearly what you changed and why.

Guidelines

* Keep code clean and readable.

* Follow the existing project structure.

* Ensure the Streamlit app runs successfully before submitting.

* Avoid committing large raw datasets or sensitive files.

---
## 15. License
This project is licensed under the MIT License

---

Faysel Nessro

Machine Learning • Data Science • software engineer

Feel free to reach out or open issues/questions.

