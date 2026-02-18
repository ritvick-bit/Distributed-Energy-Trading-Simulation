import pandas as pd
import numpy as np
from datetime import datetime
from simulation import run_simulation
from weather_utils import generate_monte_carlo_weather

def test_simulation_run():
    print("Testing Weather Generation...")
    start_date = datetime(2024, 7, 14)
    weather = generate_monte_carlo_weather(start_date, num_days=2, num_scenarios=1)
    print(f"Weather generated: {len(weather)} rows")
    
    print("Testing Simulation Logic...")
    scenario = weather[weather['scenario_id'] == 0]
    results = run_simulation(scenario, num_prosumers=5, num_consumers=5, feeder_limit=1000, ramp_limit=500)
    
    print(f"Simulation results: {len(results)} rows")
    print("Columns:", results.columns.tolist())
    
    if 'failed' in results.columns and 'net_grid_load' in results.columns:
        print("SUCCESS: Simulation ran and produced expected columns.")
    else:
        print("FAILURE: Missing columns in results.")

if __name__ == "__main__":
    test_simulation_run()
