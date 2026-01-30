# app/app.py
"""
Streamlit app for energy usage forecasting (recursive 168-hour forecast).
- Shows processed df_hourly
- Runs recursive forecast using src/xg_model1.json
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta

# project imports
from src import preprocess, features, models, utils
from src.config import PROCESSED_DATA_PATH, MODEL_JSON_PATH, FEATURES_JSON_PATH, DEFAULT_FORECAST_HORIZON

st.set_page_config(page_title="Energy Usage Forecasting", layout="wide")

def load_or_prepare_data():
    if os.path.exists(PROCESSED_DATA_PATH):
        df = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=['datetime'], index_col='datetime')
        # st.success("Loaded processed data.")
    else:
        with st.spinner("Preprocessing raw data (this may take a while)..."):
            df = preprocess.process_and_save()
        st.success(f"Processed raw data and saved to {PROCESSED_DATA_PATH}")
    return df

def recursive_forecast(df_history, model, features_list, horizon=DEFAULT_FORECAST_HORIZON):
    # Adapted recursive forecasting that uses features functions
    df_all = df_history.copy().sort_index()
    df_all = features.add_time_features(df_all)
    # Ensure some basic columns exist
    for col in ['Sub_metering_1','Sub_metering_2','Sub_metering_3','Voltage','Global_intensity']:
        if col not in df_all.columns:
            df_all[col] = 0.0

    # Precompute lags/rolls for history
    df_all = features.add_lags(df_all, target='Global_active_power')
    df_all = features.add_rollings(df_all, target='Global_active_power', windows=[24, 168])

    future_index = pd.date_range(start=df_all.index[-1] + pd.Timedelta(hours=1), periods=horizon, freq='H')
    preds = []

    for t in future_index:
        last = df_all.iloc[-1:].copy()
        last.index = [t]

        # update time features
        last['hour'] = t.hour; last['day'] = t.day; last['weekday'] = t.weekday()
        last['month'] = t.month; last['is_weekend'] = int(t.weekday()>=5)

        # regressors: try to use last week's same hour if available
        for col in ['Sub_metering_1','Sub_metering_2','Sub_metering_3','Voltage','Global_intensity']:
            if col in df_all.columns and len(df_all) >= 168:
                last[col] = df_all[col].iloc[-168 + (len(preds) % 168)]
            elif col in df_all.columns:
                last[col] = df_all[col].iloc[-1]
            else:
                last[col] = 0.0

        # update lags
        last['lag168'] = df_all['Global_active_power'].iloc[-168] if len(df_all) >= 168 else df_all['Global_active_power'].iloc[-1]
        last['lag24'] = df_all['Global_active_power'].iloc[-24] if len(df_all) >= 24 else df_all['Global_active_power'].iloc[-1]
        last['lag1'] = df_all['Global_active_power'].iloc[-1]

        # update rolling 24 mean/std
        window_vals = list(df_all['Global_active_power'].iloc[-23:]) + [last['lag1'].values[0]]
        last['roll24_mean'] = np.mean(window_vals)
        last['roll24_std'] = np.std(window_vals)

        # ensure all model features exist
        for f in features_list:
            if f not in last.columns:
                last[f] = 0.0

        X_row = last[features_list].iloc[0].values.reshape(1, -1)
        X_row = np.nan_to_num(X_row, nan=0.0, posinf=0.0, neginf=0.0)

        yhat = model.predict(X_row)[0]
        preds.append(yhat)

        new_row = last.copy()
        new_row['Global_active_power'] = yhat
        new_row['Other_Consumption'] = max(0.0, new_row['Global_active_power'] - (
            new_row.get('Sub_metering_1',0.0) + new_row.get('Sub_metering_2',0.0) + new_row.get('Sub_metering_3',0.0)
        ))
        df_all = pd.concat([df_all, new_row])

    forecast_series = pd.Series(preds, index=future_index, name='Global_active_power_forecast')
    return forecast_series, df_all

def main():
    st.title("⚡ Energy Usage Forecasting")
    st.markdown("### Production-Ready Forecasting Dashboard")

    with st.sidebar:
        st.header("Controls")
        horizon_hours = st.slider("Forecast Horizon (Hours)", min_value=24, max_value=720, value=DEFAULT_FORECAST_HORIZON, step=24)
        run_forecast = st.button("Predict Future Consumption", type="primary")
        
        st.divider()
        st.write(f"**Model Path:** `{os.path.basename(MODEL_JSON_PATH)}`")

    # Load Data
    try:
        df = load_or_prepare_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    # Metrics Section
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Date Range Start", df.index.min().strftime('%Y-%m-%d'))
    with col3:
        st.metric("Date Range End", df.index.max().strftime('%Y-%m-%d'))
    
    st.subheader("Historical Data Visualization")
    # Plotly Chart for History (Last 30 days default to avoid lag)
    df_plot = df.tail(24 * 30).reset_index()
    fig_hist = px.line(df_plot, x='datetime', y='Global_active_power', title='Global Active Power (Last 30 Days)')
    st.plotly_chart(fig_hist, width="stretch")

    # Load Model
    # Load Model
    model = None
    try:
        model = models.load_model_xgb(MODEL_JSON_PATH)
    except Exception as e:
        st.error(f"Could not load model assets. Ensure `{MODEL_JSON_PATH}` exists. Error: {e}")

    # Load Features
    features_list = []
    try:
        features_list = models.load_features_list(FEATURES_JSON_PATH)
        st.sidebar.success("Model & Features Loaded Successfully")
    except Exception:
        # Fallback: derive a sensible feature list using the features module
        # st.warning("Features list not found. Using default feature set derived from data.")
        try:
             # Create a dummy dataframe with all potential columns to see what we get
             df_temp = features.build_features(df.head(200)) # Use a subset for speed
             features_list = df_temp.drop(columns=['Global_active_power','datetime'] , errors='ignore').columns.tolist()
             st.sidebar.warning("Using default features")
        except Exception as e:
             st.error(f"Failed to generate default features: {e}")

    if run_forecast and model and features_list:
        with st.spinner(f"Generating recursive forecast for {horizon_hours} hours..."):
            forecast_series, df_all_forecast = recursive_forecast(df, model, features_list, horizon=horizon_hours)

        st.success("Forecast generated successfully!")
        
        # Combine history (last 7 days) and forecast for visualization
        history_snippet = df['Global_active_power'].iloc[-168:]
        
        hist_trace = go.Scatter(x=history_snippet.index, y=history_snippet, mode='lines', name='Historical (Last 7 Days)', line=dict(color='blue'))
        forecast_trace = go.Scatter(x=forecast_series.index, y=forecast_series, mode='lines', name='Forecast', line=dict(color='red', dash='dash'))
        
        fig_forecast = go.Figure(data=[hist_trace, forecast_trace])
        fig_forecast.update_layout(title=f"Energy Consumption Forecast (Next {horizon_hours} Hours)", xaxis_title="Time", yaxis_title="Global Active Power (kW)")
        st.plotly_chart(fig_forecast, width="stretch")

        # Data Preview & Download
        col_preview, col_download = st.columns([2, 1])
        with col_preview:
            st.subheader("Forecast Data")
            preview = forecast_series.reset_index().rename(columns={'index':'datetime','Global_active_power_forecast':'forecast'})
            st.dataframe(preview, height=200)
        
        with col_download:
            st.subheader("Actions")
            csv = preview.to_csv(index=False)
            st.download_button("Download CSV", data=csv, file_name=f"forecast_{horizon_hours}h.csv", mime="text/csv")

if __name__ == "__main__":
    main()
