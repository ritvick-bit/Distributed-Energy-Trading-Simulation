import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import xgboost as xgb
import os
import json

# Paths for models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Global models (load on import or first use)
_TEMP_MODEL = None
_WIND_MODEL = None

def load_models():
    global _TEMP_MODEL, _WIND_MODEL
    if _TEMP_MODEL is None:
        _TEMP_MODEL = xgb.XGBRegressor()
        model_path = os.path.join(MODELS_DIR, "temp_model.json")
        if os.path.exists(model_path):
            _TEMP_MODEL.load_model(model_path)
            
    if _WIND_MODEL is None:
        _WIND_MODEL = xgb.XGBRegressor()
        model_path = os.path.join(MODELS_DIR, "wind_model.json")
        if os.path.exists(model_path):
            _WIND_MODEL.load_model(model_path)
    
    return _TEMP_MODEL, _WIND_MODEL

def simulate_day_from_early_morning_lags(prev_day_data, date, randomness_factor=0.1):
    """
    Simulates hourly weather data for a single day based on previous day's patterns
    using trained XGBoost models.
    """
    load_models()
    
    hours = np.arange(24)
    day_of_week = date.weekday()
    day_of_month = date.day
    
    # Initialize arrays
    temperature = np.zeros(24)
    wind_speed = np.zeros(24)
    
    # Seed from previous day or use defaults
    if prev_day_data and 'temperature' in prev_day_data:
        curr_temp = prev_day_data['temperature'][-1]
        curr_wind = prev_day_data['wind_speed'][-1]
    else:
        # Realistic defaults for Sugar Land July
        curr_temp = 80.0
        curr_wind = 5.0
        
    for h in range(24):
        # Features: ['hour', 'day_of_week', 'day_of_month', 'temp_lag1', 'wspd_lag1']
        features = pd.DataFrame([[h, day_of_week, day_of_month, curr_temp, curr_wind]], 
                                columns=['hour', 'day_of_week', 'day_of_month', 'temp_lag1', 'wspd_lag1'])
        
        # Predict
        if _TEMP_MODEL:
            pred_temp = _TEMP_MODEL.predict(features)[0]
        else:
            # Fallback to sinusoidal if model missing
            pred_temp = 85 + 10 * np.sin(2 * np.pi * (h - 10) / 24)
            
        if _WIND_MODEL:
            pred_wind = _WIND_MODEL.predict(features)[0]
        else:
            pred_wind = 5 + np.random.normal(0, 1)
            
        # Add randomness/noise
        curr_temp = pred_temp + np.random.normal(0, 1.5) * randomness_factor
        curr_wind = max(0, pred_wind + np.random.normal(0, 1.0) * randomness_factor)
        
        temperature[h] = curr_temp
        wind_speed[h] = curr_wind

    # Solar Radiation: Gaussian bell curve centered at noon (Stil approximate as usually not in model)
    peak_solar = np.random.normal(900, 100)
    solar_curve = np.exp(-0.5 * ((hours - 13) / 2.5) ** 2)
    cloud_cover = np.random.uniform(0.8, 1.0, 24)
    for i in range(1, 24):
        cloud_cover[i] = cloud_cover[i-1] * 0.7 + np.random.uniform(0.8, 1.0) * 0.3
    
    solar_radiation = peak_solar * solar_curve * cloud_cover
    solar_radiation = np.clip(solar_radiation, 0, None)
    
    # Rainfall: Occasional spikes
    rain = np.zeros(24)
    if np.random.random() < 0.2:
        rain_start = np.random.randint(12, 20)
        rain_duration = np.random.randint(1, 4)
        rain[rain_start:min(rain_start+rain_duration, 24)] = 1
        temperature[rain_start:min(rain_start+rain_duration, 24)] -= 5
        solar_radiation[rain_start:min(rain_start+rain_duration, 24)] *= 0.2

    df = pd.DataFrame({
        'timestamp': [date + timedelta(hours=int(h)) for h in hours],
        'temperature': temperature,
        'solar_radiation': solar_radiation,
        'wind_speed': wind_speed,
        'rain': rain
    })
    
    return df

def generate_monte_carlo_weather(start_date, num_days, num_scenarios=10):
    """
    Generates multiple scenarios of weather data.
    """
    scenarios = []
    
    for i in range(num_scenarios):
        scenario_data = []
        current_date = start_date
        
        # Initial 'previous day'
        prev_data = None
        
        for day in range(num_days):
            daily_df = simulate_day_from_early_morning_lags(prev_data, current_date, randomness_factor=0.2 + (i*0.01))
            daily_df['scenario_id'] = i
            scenario_data.append(daily_df)
            
            prev_data = daily_df.to_dict(orient='list')
            current_date += timedelta(days=1)
            
        full_scenario = pd.concat(scenario_data, ignore_index=True)
        scenarios.append(full_scenario)
        
    return pd.concat(scenarios, ignore_index=True)
