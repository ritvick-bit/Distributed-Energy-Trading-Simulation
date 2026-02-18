import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def simulate_day_from_early_morning_lags(prev_day_data, date, randomness_factor=0.1):
    """
    Simulates hourly weather data for a single day based on previous day's patterns
    and some randomness.
    
    Args:
        prev_day_data (dict): Dictionary containing 'temp', 'solar', 'wind' lists/arrays for previous day.
        date (datetime): The date to simulate.
        randomness_factor (float): Magnitude of random noise.
        
    Returns:
        pd.DataFrame: Hourly dataframe for the simulated day.
    """
    hours = np.arange(24)
    
    # Simple persistence model with noise
    # In a real scenario, this would use the XGBoost model mentioned in the report
    # converting the logic from the decision tree image would be complex without the model file.
    # We approximate with sinusoidal patterns + noise + persistence.
    
    # Temperature: Daily cycle
    # Base temp trend (e.g. slowly increasing in July)
    day_of_year = date.timetuple().tm_yday
    base_temp = 80 + 5 * np.sin(2 * np.pi * (day_of_year - 100) / 365) # Seasonal
    
    # Hourly variation (Cardioidal/Sinusoidal)
    # Min at 4AM, Max at 3PM (15:00)
    temp_variation = -10 * np.cos(2 * np.pi * (hours - 4) / 24) 
    
    # Add persistence from previous day (if available) or random noise
    noise = np.random.normal(0, 2, 24) * randomness_factor
    temperature = base_temp + temp_variation + noise
    
    # Clipping to realistic Texas Summer values
    temperature = np.clip(temperature, 70, 105)
    
    # Solar Radiation: Gaussian bell curve centered at noon
    # Peak intensity varies
    peak_solar = np.random.normal(900, 100) # W/m^2
    solar_curve = np.exp(-0.5 * ((hours - 13) / 2.5) ** 2) # Peak at 1pm
    cloud_cover = np.random.uniform(0.8, 1.0, 24) # Random cloud factor
    # Make clouds correlated (if it's cloudy at 10, likely cloudy at 11)
    for i in range(1, 24):
        cloud_cover[i] = cloud_cover[i-1] * 0.7 + np.random.uniform(0.8, 1.0) * 0.3
    
    solar_radiation = peak_solar * solar_curve * cloud_cover
    solar_radiation = np.clip(solar_radiation, 0, None)
    
    # Wind Speed: Weibull distribution approximation or just random walk
    wind_speed = np.abs(np.random.normal(5, 2, 24))
    
    # Rainfall: Occasional spikes
    rain = np.zeros(24)
    if np.random.random() < 0.2: # 20% chance of rain in a day
        rain_start = np.random.randint(12, 20)
        rain_duration = np.random.randint(1, 4)
        rain[rain_start:min(rain_start+rain_duration, 24)] = 1
        # Drop temp and solar during rain
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
