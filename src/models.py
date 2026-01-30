import os
import json
import joblib
from xgboost import XGBRegressor
from src.config import MODEL_JSON_PATH, MODEL_JOBLIB_PATH, FEATURES_JSON_PATH

def train_xgb(X_train, y_train, path_json=MODEL_JSON_PATH, path_joblib=MODEL_JOBLIB_PATH, **params):
    params_default = dict(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42,
        verbosity=0
    )
    params_default.update(params)
    model = XGBRegressor(**params_default)
    model.fit(X_train, y_train)
    os.makedirs(os.path.dirname(path_json) or '.', exist_ok=True)
    model.save_model(path_json)
    joblib.dump(model, path_joblib)
    return model

import xgboost as xgb

class XGBWrapper:
    def __init__(self, booster):
        self.booster = booster
        
    def predict(self, X):
        # Ensure X is DMatrix compatible (numpy array or similar)
        dmatrix = xgb.DMatrix(X)
        return self.booster.predict(dmatrix)

def load_model_xgb(path_json=MODEL_JSON_PATH):
    if not os.path.exists(path_json):
        raise FileNotFoundError(f"XGBoost model not found at {path_json}")
    # Load as native Booster to avoid sklearn compatibility issues
    booster = xgb.Booster()
    booster.load_model(path_json)
    return XGBWrapper(booster)

def save_features_list(feature_list, path=FEATURES_JSON_PATH):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(feature_list, f)

def load_features_list(path=FEATURES_JSON_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Features list not found at {path}")
    with open(path, 'r') as f:
        return json.load(f)

