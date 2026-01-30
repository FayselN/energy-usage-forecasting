
import pytest
import pandas as pd
import numpy as np
from src import preprocess

def test_preprocess_to_hourly_basic():
    # Create sample DataFrame
    dates = pd.date_range(start='2022-01-01', periods=120, freq='T')
    data = {
        'Global_active_power': np.random.rand(120) * 2,
        'Global_reactive_power': np.random.rand(120) * 0.2,
        'Voltage': np.random.rand(120) * 230,
        'Global_intensity': np.random.rand(120) * 10,
        'Sub_metering_1': np.random.rand(120) * 5,
        'Sub_metering_2': np.random.rand(120) * 5,
        'Sub_metering_3': np.random.rand(120) * 5,
    }
    df = pd.DataFrame(data)
    df['datetime'] = dates
    
    # Run preprocessing
    df_hourly = preprocess.preprocess_to_hourly(df)
    
    # Assertions
    assert len(df_hourly) == 2  # 120 minutes = 2 hours
    assert 'Other_Consumption' in df_hourly.columns
    assert df_hourly.index.freq == 'H'
    assert not df_hourly.isnull().values.any()

def test_other_consumption_calculation():
    # Setup consistent values to check math
    # 2 hours of data
    dates = pd.date_range(start='2022-01-01', periods=120, freq='T')
    df = pd.DataFrame(index=range(120))
    df['datetime'] = dates
    # Global active power in kW, others in Wh (raw data format)
    # 1000 Wh = 1 kWh
    # Let's say GAP is 60 kW sum per MINUTE? 
    # Wait, the logic in preprocess is: GAP sum / 60.0 -> kWh
    # Submetering is sum -> Wh -> /1000 -> kWh
    
    df['Global_active_power'] = 1.0 # 1.0 per minute. Sum = 60. GAP_kWh = 60/60 = 1.0 kWh?
    df['Global_reactive_power'] = 0.1
    df['Voltage'] = 240
    df['Global_intensity'] = 5
    df['Sub_metering_1'] = 0
    df['Sub_metering_2'] = 0
    df['Sub_metering_3'] = 0 
    
    
    df_hourly = preprocess.preprocess_to_hourly(df)
    
    # If GAP is 1.0 every minute, sum is 60. /60 = 1.0 kWh.
    # Submetering is 0.
    # Other = GAP - Subs = 1.0 - 0 = 1.0
    
    assert np.isclose(df_hourly['Global_active_power'].iloc[0], 1.0)
    assert np.isclose(df_hourly['Other_Consumption'].iloc[0], 1.0)
