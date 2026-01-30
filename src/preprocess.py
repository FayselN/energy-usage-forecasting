# src/preprocess.py
"""
Preprocessing: read raw .txt dataset, clean, convert to hourly dataframe,
compute Other_Consumption (kWh) and save processed CSV to data/processed/df_hourly.csv
"""

import os
import pandas as pd
import numpy as np
from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH

def read_raw(txt_path=RAW_DATA_PATH):
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Raw file not found at {txt_path}. Place the raw file there.")
    df = pd.read_csv(
        txt_path,
        sep=';',
        parse_dates={'datetime': [0, 1]},
        infer_datetime_format=True,
        low_memory=False,
        na_values=['nan', '?']
    )
    return df

def preprocess_to_hourly(df):
    # ensure numeric
    cols_num = ['Global_active_power','Global_reactive_power','Voltage',
                'Global_intensity','Sub_metering_1','Sub_metering_2','Sub_metering_3']
    for c in cols_num:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    df = df.dropna(subset=['Global_active_power']).copy()
    df.set_index('datetime', inplace=True)
    df = df.sort_index()

    agg = {
        'Global_active_power': lambda x: x.sum() / 60.0,   # kWh per hour
        'Global_reactive_power': lambda x: x.sum() / 60.0,
        'Voltage': 'mean',
        'Global_intensity': 'mean',
        'Sub_metering_1': 'sum',  # Wh per hour -> converted below
        'Sub_metering_2': 'sum',
        'Sub_metering_3': 'sum'
    }

    df_hourly = df.resample('H').agg(agg)

    # convert sub-metering Wh -> kWh
    for c in ['Sub_metering_1','Sub_metering_2','Sub_metering_3']:
        df_hourly[c] = df_hourly[c] / 1000.0

    # compute Other_Consumption (kWh): total - sum(submeters)
    df_hourly['Other_Consumption'] = df_hourly['Global_active_power'] - (
        df_hourly['Sub_metering_1'] + df_hourly['Sub_metering_2'] + df_hourly['Sub_metering_3']
    )
    df_hourly['Other_Consumption'] = df_hourly['Other_Consumption'].clip(lower=0)

    # drop rows with NaN
    df_hourly = df_hourly.dropna()
    return df_hourly

def process_and_save(raw_path=RAW_DATA_PATH, out_path=PROCESSED_DATA_PATH, force=False):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path) and not force:
        print(f"Processed file already exists at {out_path}. Use force=True to reprocess.")
        return pd.read_csv(out_path, parse_dates=['datetime'], index_col='datetime')
    df_raw = read_raw(raw_path)
    df_hourly = preprocess_to_hourly(df_raw)
    df_hourly.to_csv(out_path)
    print(f"Saved processed hourly data to {out_path} (shape: {df_hourly.shape})")
    return df_hourly

if __name__ == "__main__":
    process_and_save()

