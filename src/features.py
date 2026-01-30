# src/features.py
"""
Feature engineering functions:
- add_time_features
- add_lags
- add_rollings
- build_features (history -> features for training)
Also exports FEATURES list used by model (saved by models.py during training).
"""

import pandas as pd
import numpy as np

ROLL_WINDOWS = [3,6,12,24,48,72,96,168]
LAGS = [1,24,168]

def add_time_features(df):
    df = df.copy()
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')
    df = df.sort_index()
    df['hour'] = df.index.hour
    df['day'] = df.index.day
    df['weekday'] = df.index.weekday
    df['weekofyear'] = df.index.isocalendar().week.astype(int)
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['is_weekend'] = (df.index.weekday >= 5).astype(int)
    return df

def add_lags(df, target='Global_active_power', lags=LAGS):
    df = df.copy()
    for lag in lags:
        df[f'lag{lag}'] = df[target].shift(lag)
    return df

def add_rollings(df, target='Global_active_power', windows=ROLL_WINDOWS):
    df = df.copy()
    for w in windows:
        df[f'roll{w}_mean'] = df[target].rolling(window=w).mean()
        df[f'roll{w}_std']  = df[target].rolling(window=w).std()
        df[f'roll{w}_sum']  = df[target].rolling(window=w).sum()
    return df

def build_features(df_hourly, drop_na=True):
    """From hourly dataframe (index=datetime), create full feature matrix for training."""
    df = df_hourly.copy()
    df = add_time_features(df)
    df = add_lags(df, target='Global_active_power')
    df = add_rollings(df, target='Global_active_power')
    if drop_na:
        df = df.dropna()
    return df

# helper to construct the minimal set you'll need for recursive forecasting.
# Choose a sensible subset of features to use in the app model if you trained with many.
def default_feature_list(df):
    # pick time features + common lags + roll24 mean/std + regressors
    cols = [
        'hour','day','weekday','month','is_weekend',
        'lag1','lag24','lag168',
        'roll24_mean','roll24_std',
        'Sub_metering_1','Sub_metering_2','Sub_metering_3',
        'Voltage','Global_intensity','Other_Consumption'
    ]
    # adapt names to actual column names in df (roll24_mean / roll24_std)
    # ensure roll24 names exist in df, else adapt to roll24_mean naming scheme
    return [c for c in cols if c in df.columns]

