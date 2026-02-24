import os
import pandas as pd
import numpy as np
from datetime import datetime
from meteostat import hourly, stations, config
import xgboost as xgb
import json

# Setup directories
base_dir = r"c:\Users\Ritvi\OneDrive\Desktop\Antigravity Projects\ADERS"
scripts_dir = os.path.join(base_dir, "scripts")
models_dir = os.path.join(base_dir, "models")
os.makedirs(models_dir, exist_ok=True)

# Allow large requests
config.block_large_requests = False

def fetch_data():
    print("Searching for Sugar Land station...")
    # Sugar Land Regional Airport station
    station_id = '72244' # WMO ID for Sugar Land KSGR
    
    # Alternatively find by name/location
    # stations = Stations()
    # stations = stations.nearby(29.62, -95.65)
    # station = stations.fetch(1)
    # station_id = station.index[0]
    
    # Fetch July data for the last 5 years
    start = datetime(2019, 7, 1)
    end = datetime(2023, 7, 31)
    
    print(f"Fetching data for station {station_id} from {start} to {end}...")
    data = hourly(station_id, start, end)
    df = data.fetch()
    
    # Filter for July only (in case multiple years are fetched)
    df = df[df.index.month == 7]
    
    # Unit conversion: Celsius to Fahrenheit, km/h to mph
    df['temp'] = (df['temp'] * 9/5) + 32
    df['wspd'] = df['wspd'] / 1.60934
    
    print("July Data summary after conversion:")
    print(df[['temp', 'wspd']].describe())
    
    print(f"Fetched and converted {len(df)} records.")
    return df

def preprocess_data(df):
    # Select only necessary columns to avoid dropping rows due to unrelated NaNs
    df = df[['temp', 'wspd']]
    
    print(f"Preprocessing {len(df)} rows...")
    # Interpolate missing values
    df = df.interpolate(method='linear')
    print(f"After interpolation: {df.isnull().sum().sum()} total NaNs")
    
    # Create features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    
    # Create lag features (previous hour temperature and wind)
    df['temp_lag1'] = df['temp'].shift(1)
    df['wspd_lag1'] = df['wspd'].shift(1)
    
    print(f"Before dropna: {len(df)} rows")
    # Drop first row due to lag
    df = df.dropna()
    print(f"After dropna: {len(df)} rows")
    
    return df

def train_models(df):
    features = ['hour', 'day_of_week', 'day_of_month', 'temp_lag1', 'wspd_lag1']
    
    print("Training Temperature model...")
    reg_temp = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    reg_temp.fit(df[features], df['temp'])
    reg_temp.save_model(os.path.join(models_dir, "temp_model.json"))
    
    print("Training Wind Speed model...")
    reg_wind = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
    reg_wind.fit(df[features], df['wspd'])
    reg_wind.save_model(os.path.join(models_dir, "wind_model.json"))
    
    # Save training metadata
    metadata = {
        "features": features,
        "station": "Sugar Land (KSGR)",
        "training_period": "July 2019-2023",
        "last_trained": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(os.path.join(models_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
    
    print("Models saved successfully.")
    
    # Sample prediction check
    test_feat = df[features].iloc[:5]
    print("\nSample predictions (Internal check):")
    print("Actual Temp:", df['temp'].iloc[:5].values)
    print("Pred Temp:  ", reg_temp.predict(test_feat))

if __name__ == "__main__":
    df = fetch_data()
    if not df.empty:
        df = preprocess_data(df)
        train_models(df)
    else:
        print("No data found. Check station ID or connectivity.")
