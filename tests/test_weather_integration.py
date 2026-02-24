import sys
import os
from datetime import datetime
import pandas as pd

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from weather_utils import simulate_day_from_early_morning_lags, generate_monte_carlo_weather

def test_single_day_simulation():
    print("Testing single day simulation...")
    date = datetime(2026, 7, 1)
    df = simulate_day_from_early_morning_lags(None, date)
    
    print(f"Generated {len(df)} hours of weather.")
    print("Sample data:")
    print(df.head())
    
    assert len(df) == 24
    assert 'temperature' in df.columns
    assert 'wind_speed' in df.columns
    assert 'solar_radiation' in df.columns
    
    # Check if values are within realistic ranges for July
    assert df['temperature'].min() > 60
    assert df['temperature'].max() < 115
    print("Single day simulation test passed!")

def test_monte_carlo_weather():
    print("\nTesting Monte Carlo weather generation...")
    start_date = datetime(2026, 7, 1)
    num_days = 2
    num_scenarios = 3
    
    df = generate_monte_carlo_weather(start_date, num_days, num_scenarios)
    
    print(f"Generated {len(df)} records across {num_scenarios} scenarios.")
    print(f"Scenario IDs: {df['scenario_id'].unique()}")
    
    assert len(df) == 24 * num_days * num_scenarios
    assert len(df['scenario_id'].unique()) == num_scenarios
    print("Monte Carlo weather generation test passed!")

if __name__ == "__main__":
    try:
        test_single_day_simulation()
        test_monte_carlo_weather()
        print("\nAll weather tests passed successfully!")
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
