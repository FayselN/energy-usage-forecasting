# src/utils.py
"""
Helper utilities for plotting and IO used by the app.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_history_and_forecast(history_series, forecast_series, savepath=None, title="History + Forecast"):
    plt.figure(figsize=(14,5))
    plt.plot(history_series.index, history_series.values, label='History', linewidth=2)
    plt.plot(forecast_series.index, forecast_series.values, label='Forecast', linewidth=2, linestyle='--')
    plt.title(title)
    plt.xlabel('Datetime')
    plt.ylabel(history_series.name or 'value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        plt.savefig(savepath)
    plt.close()

def df_preview_markdown(df, n=10):
    """Return a small markdown-friendly preview of the dataframe."""
    return df.head(n).to_markdown()

