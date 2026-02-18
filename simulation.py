import numpy as np
import pandas as pd

class Household:
    def __init__(self, id, has_solar, has_ev, battery_capacity=10):
        self.id = id
        self.has_solar = has_solar
        self.has_ev = has_ev
        self.battery_capacity = battery_capacity # kWh
        self.battery_soc = 0.5 * battery_capacity # Start at 50%
        self.tokens = 50 # Initial tokens
        
        # Base load profile (randomized slightly per house)
        # Peak in evening (6-9pm)
        self.base_load = np.random.uniform(0.5, 2.0) 
        
    def step(self, hour, solar_irradiance, temp, price):
        """
        Executes one hour step.
        Returns (net_demand, grid_export, grid_import)
        """
        # 1. Calculate Consumption
        # Higher temp -> Air Conditioning load
        ac_load = max(0, (temp - 75) * 0.1) if temp > 75 else 0
        
        # Human activity profile (simple hour-based multiplier)
        if 7 <= hour <= 9: activity = 1.2 # Morning
        elif 17 <= hour <= 21: activity = 1.5 # Evening
        elif 22 <= hour or hour <= 5: activity = 0.5 # Night
        else: activity = 0.8 # Day
        
        consumption = (self.base_load * activity) + ac_load
        
        # EV Charging (if present and night time)
        ev_load = 0
        if self.has_ev and (20 <= hour or hour <= 6):
            # Simple logic: Charge if low price or just scheduled
            ev_load = 7.0 # 7kW charger
            
        total_load = consumption + ev_load
        
        # 2. Calculate Generation
        generation = 0
        if self.has_solar:
            # Simple conversion: Irradiance (W/m2) / 1000 * Efficiency * Panel Area
            # Assuming 5kW system
            efficiency = 0.18
            panel_area = 30 # m2
            generation = (solar_irradiance / 1000) * efficiency * panel_area
            generation = max(0, generation)
            
        # 3. Battery Logic (Greedy)
        net_energy = generation - total_load
        
        grid_export = 0
        grid_import = 0
        
        if net_energy > 0:
            # Surplus: Charge battery first
            charge_space = self.battery_capacity - self.battery_soc
            to_battery = min(net_energy, charge_space)
            self.battery_soc += to_battery
            grid_export = net_energy - to_battery
        else:
            # Deficit: Discharge battery first
            deficit = abs(net_energy)
            from_battery = min(deficit, self.battery_soc)
            self.battery_soc -= from_battery
            grid_import = deficit - from_battery
            
        # Update Tokens (Simple mechanism)
        # Sell export, Buy import
        cost = grid_import * price
        revenue = grid_export * price
        self.tokens = self.tokens - cost + revenue
        
        return {
            "id": self.id,
            "load": total_load,
            "generation": generation,
            "net": grid_export - grid_import, # Positive = Export
            "battery_soc": self.battery_soc,
            "tokens": self.tokens
        }

class Grid:
    def __init__(self, feeder_limit=1000, ramp_rate_limit=500):
        self.feeder_limit = feeder_limit
        self.ramp_rate_limit = ramp_rate_limit
        self.prev_net_load = 0
        self.failures = []
        
    def check_constraints(self, hour, current_net_load):
        """
        Checks if grid constraints are violated.
        current_net_load: Total community load (Positive = Consumption from Grid, Negative = Backfeed)
        """
        issues = []
        
        # 1. Feeder Limit (Capacity)
        if abs(current_net_load) > self.feeder_limit:
            issues.append("Feeder Overload")
            
        # 2. Ramp Rate (Change in load per hour)
        ramp = abs(current_net_load - self.prev_net_load)
        if ramp > self.ramp_rate_limit:
            issues.append("Ramp Rate Violation")
            
        self.prev_net_load = current_net_load
        
        if issues:
            self.failures.append({
                "hour": hour,
                "issues": issues,
                "load": current_net_load
            })
        return len(issues) > 0, issues

class Market:
    def __init__(self, base_price=0.15):
        self.base_price = base_price
        
    def calculate_price(self, total_demand, total_supply):
        # Dynamic pricing based on net demand
        # Simplified linear model adapted from snippet
        net_demand = total_demand - total_supply
        
        # If High Demand (Net > 0), Price increases
        # If High Supply (Net < 0), Price decreases
        price_adjustment = net_demand * 0.0001 # Sensitivity factor
        price = self.base_price + price_adjustment
        return max(0.05, price) # Floor price

def run_simulation(weather_df, num_prosumers, num_consumers, feeder_limit, ramp_limit):
    # Initialize Agents
    households = []
    # Prosumers (Solar + Battery + Maybe EV)
    for i in range(num_prosumers):
        households.append(Household(i, has_solar=True, has_ev=np.random.choice([True, False])))
        
    # Consumers (No Solar)
    for i in range(num_consumers):
        households.append(Household(num_prosumers + i, has_solar=False, has_ev=np.random.choice([True, False])))
        
    grid = Grid(feeder_limit, ramp_limit)
    market = Market(base_price=0.12)
    
    results = []
    
    # Run Loop
    for idx, row in weather_df.iterrows():
        hour = row['timestamp'].hour
        temp = row['temperature']
        solar = row['solar_radiation']
        
        # 1. Market Clearing Price (Predictive or Previous Step - simplified here to per-step)
        # We need to aggregate demand/supply to set price? 
        # Or let price be defined by previous step's imbalance?
        # Let's use a base price for now, or update it after collecting bids.
        # Ideally: Households submit bids -> Market Clears -> Households act.
        # Simplified: Households act based on 'expected' price (base), then we calculate system balance cost.
        
        current_price = market.base_price # Simplification
        
        step_metrics = []
        total_export = 0
        total_import = 0
        
        for hh in households:
            met = hh.step(hour, solar, temp, current_price)
            step_metrics.append(met)
            if met['net'] > 0:
                total_export += met['net']
            else:
                total_import += abs(met['net'])
                
        # Update Market Price for NEXT step or record effective price
        clearing_price = market.calculate_price(total_import, total_export)
        
        # Grid Check
        net_grid_load = total_import - total_export
        failed, issues = grid.check_constraints(idx, net_grid_load)
        
        results.append({
            "timestamp": row['timestamp'],
            "hour": hour,
            "total_import": total_import,
            "total_export": total_export,
            "net_grid_load": net_grid_load,
            "price": clearing_price,
            "failed": failed,
            "issues": issues,
            "temp": temp
        })
        
    return pd.DataFrame(results)

def run_batch_simulations(n_samples, weather_scenarios):
    """
    Runs multiple simulations with randomized parameters to generate a dataset for sensitivity analysis.
    """
    batch_results = []
    
    # Pre-compute or fetch a set of weather scenarios to pick from
    # We assume weather_scenarios provided is a DataFrame with multiple scenario_id
    available_scenarios = weather_scenarios['scenario_id'].unique()
    
    for i in range(n_samples):
        # 1. Randomize Inputs
        num_homes = np.random.randint(50, 200) # Community size
        pv_pct = np.random.uniform(0, 1.0) # % of homes with Solar
        ev_pct = np.random.uniform(0, 1.0) # % of homes with EV
        
        num_prosumers = int(num_homes * pv_pct)
        num_consumers = num_homes - num_prosumers
        
        # Grid Constraints (Randomized)
        feeder_limit = np.random.randint(500, 3000)
        ramp_limit = np.random.randint(100, 1000)
        
        # Pick a random weather scenario
        scen_id = np.random.choice(available_scenarios)
        scenario_df = weather_scenarios[weather_scenarios['scenario_id'] == scen_id].reset_index(drop=True)
        
        # 2. Run Simulation
        # We need to capture failures
        sim_df = run_simulation(scenario_df, num_prosumers, num_consumers, feeder_limit, ramp_limit)
        
        # 3. Aggregate Metrics
        has_failed = sim_df['failed'].any()
        total_failures = sim_df['failed'].sum()
        max_import = sim_df['total_import'].max()
        max_export = sim_df['total_export'].max()
        max_ramp = sim_df['net_grid_load'].diff().abs().max()
        
        batch_results.append({
            "num_homes": num_homes,
            "pv_pct": pv_pct,
            "ev_pct": ev_pct,
            "feeder_cap": feeder_limit,
            "ramp_cap": ramp_limit,
            "failed": has_failed,
            "total_failures": total_failures,
            "max_import_kw": max_import,
            "max_export_kw": max_export,
            "max_ramp_kw": max_ramp,
            "scenario_id": scen_id
        })
        
    return pd.DataFrame(batch_results)
