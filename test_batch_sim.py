import pandas as pd
import numpy as np
from datetime import datetime
from simulation import run_batch_simulations
from weather_utils import generate_monte_carlo_weather

def test_batch_run():
    print("Generating Weather...")
    start_date = datetime(2024, 7, 14)
    # Generate small weather set
    weather = generate_monte_carlo_weather(start_date, num_days=2, num_scenarios=2)
    
    print("Running Batch Simulation (5 samples)...")
    batch_results = run_batch_simulations(5, weather)
    
    print(f"Batch results: {len(batch_results)} rows")
    print("Columns:", batch_results.columns.tolist())
    
    required_cols = ['pv_pct', 'ev_pct', 'num_homes', 'failed', 'total_failures']
    missing = [c for c in required_cols if c not in batch_results.columns]
    
    if not missing:
        print("SUCCESS: Batch simulation produced expected columns.")
    else:
        print(f"FAILURE: Missing columns: {missing}")

if __name__ == "__main__":
    test_batch_run()
