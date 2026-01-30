
import xgboost as xgb
import os
import sys

# Need to set up path since we are running as script
sys.path.append(os.getcwd())
from src.config import MODEL_JSON_PATH

print(f"Loading from {MODEL_JSON_PATH}")
if not os.path.exists(MODEL_JSON_PATH):
    print("File not found!")
    sys.exit(1)

try:
    model = xgb.Booster()
    model.load_model(MODEL_JSON_PATH)
    print("Loaded successfully as Booster")
except Exception as e:
    print(f"Failed to load as Booster: {e}")

try:
    from xgboost import XGBRegressor
    model2 = XGBRegressor()
    model2.load_model(MODEL_JSON_PATH)
    print("Loaded successfully as XGBRegressor")
    
except Exception as e:
    print(f"Failed to load as XGBRegressor: {e}")
