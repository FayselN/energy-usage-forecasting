
import os

# Project Root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data Paths
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "household_power_consumption.txt")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed", "df_hourly.csv")

# Model Paths
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
MODEL_JSON_PATH = os.path.join(SRC_DIR, "xgb_model1.json")
MODEL_JOBLIB_PATH = os.path.join(SRC_DIR, "xgboost_model1.pkl")
FEATURES_JSON_PATH = os.path.join(SRC_DIR, "features_list.json")

# Forecasting Defaults
DEFAULT_FORECAST_HORIZON = 168
