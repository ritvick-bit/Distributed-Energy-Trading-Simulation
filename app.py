import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

from weather_utils import generate_monte_carlo_weather, simulate_day_from_early_morning_lags
from simulation import run_simulation, Household, Grid, Market

st.set_page_config(page_title="P2P Energy Trading Simulation", layout="wide")

st.title("⚡ P2P Energy Trading Simulation")

# --- Sidebar Controls ---
st.sidebar.header("Simulation Controls")

st.sidebar.subheader("Community")
num_prosumers = st.sidebar.slider("Number of Prosumers (Solar+Batt)", 0, 50, 10)
num_consumers = st.sidebar.slider("Number of Consumers", 0, 50, 10)

st.sidebar.subheader("Grid Constraints")
feeder_limit = st.sidebar.slider("Feeder Limit (kW)", 100, 2000, 800)
ramp_limit = st.sidebar.slider("Ramp Rate Limit (kW/h)", 50, 1000, 400)

st.sidebar.subheader("Market")
base_price = st.sidebar.slider("Base Market Price ($/kWh)", 0.05, 0.50, 0.12)
energy_per_token = st.sidebar.slider("Energy per Token (kWh/token)", 0.5, 5.0, 1.0)

st.sidebar.subheader("Weather Params")
num_scenarios = st.sidebar.slider("Monte Carlo Scenarios", 1, 20, 5)

# --- Main Simulation ---

# 1. Weather Generation (Monte Carlo)
st.subheader("1. Weather Monte Carlo Simulation")
start_date = datetime(2024, 7, 14)
with st.spinner("Generating Weather Scenarios..."):
    # Generate scenarios
    # We generate a few paths to show variance
    weather_scenarios = generate_monte_carlo_weather(start_date, num_days=14, num_scenarios=num_scenarios)

# Visualization 1: Monte Carlo Paths (Matching requested image style)
# We want to show Temp, Solar, Wind for each scenario
colors = px.colors.qualitative.Plotly

from plotly.subplots import make_subplots
fig_mc = make_subplots(rows=4, cols=1, shared_xaxes=True, 
                       subplot_titles=("Temperature (°F)", "Humidity (Mock)", "Solar Radiation (W/m²)", "Wind Speed (m/s)"))


for i in range(num_scenarios):
    scenario_data = weather_scenarios[weather_scenarios['scenario_id'] == i]
    color = colors[i % len(colors)]
    
    # Temp
    fig_mc.add_trace(go.Scatter(x=scenario_data['timestamp'], y=scenario_data['temperature'],
                                mode='lines', name=f'Temp - Path {i}',
                                line=dict(color=color, width=1), legendgroup=f'Path {i}'),
                     row=1, col=1)
    
    # Humidity (Mock for visual as we calculated Rain/Temp but not explicit humidity in simple model - derived)
    # Inverse to temp roughly
    humidity = 100 - (scenario_data['temperature'] - 60) * 1.5 + np.random.normal(0, 5, len(scenario_data))
    fig_mc.add_trace(go.Scatter(x=scenario_data['timestamp'], y=humidity,
                                mode='lines', name=f'Humidity - Path {i}',
                                line=dict(color=color, width=1), legendgroup=f'Path {i}', showlegend=False),
                     row=2, col=1)

    # Solar
    fig_mc.add_trace(go.Scatter(x=scenario_data['timestamp'], y=scenario_data['solar_radiation'],
                                mode='lines', name=f'Solar - Path {i}',
                                line=dict(color=color, width=1), legendgroup=f'Path {i}', showlegend=False),
                     row=3, col=1)
    
    # Wind
    fig_mc.add_trace(go.Scatter(x=scenario_data['timestamp'], y=scenario_data['wind_speed'],
                                mode='lines', name=f'Wind - Path {i}',
                                line=dict(color=color, width=1), legendgroup=f'Path {i}', showlegend=False),
                     row=4, col=1)

fig_mc.update_layout(height=800, title_text="Monte Carlo Weather Scenarios")
st.plotly_chart(fig_mc, use_container_width=True)

# Visualization 2: Heatmap (Temp vs Time of Day)
st.subheader("2. Temperature vs Time of Day Heatmap")
# Scatter plot with alpha
# Using matplotlib for the specific 'scatter density' look or Plotly density heatmap
fig_heat, ax = plt.subplots(figsize=(10, 6))
# Flatten all scenarios
all_temps = weather_scenarios['temperature']
all_hours = weather_scenarios['timestamp'].dt.hour

ax.scatter(all_hours, all_temps, alpha=0.1, color='orange', s=50)
ax.set_title("Temperature vs. Time of Day")
ax.set_xlabel("Hour of Day (0-23)")
ax.set_ylabel("Temperature (°F)")
ax.grid(True, linestyle='--', alpha=0.7)
ax.set_xticks(range(0, 24, 2))

st.pyplot(fig_heat)

# --- Tabs for Modes ---
tab1, tab2 = st.tabs(["Single Simulation", "Sensitivity Analysis"])

with tab1:
    st.subheader("3. Grid Performance Simulation")
    run_btn = st.button("Run Simulation (Current Settings)")

    if run_btn:
        with st.spinner("Simulating Grid Operations..."):
            # Select one scenario (e.g. median solar) or just the first one for the demo run
            # Or running all and averaging?
            # Let's run the FIRST scenario to show detailed time-series behavior
            selected_scenario = weather_scenarios[weather_scenarios['scenario_id'] == 0].reset_index(drop=True)
            
            sim_results = run_simulation(selected_scenario, num_prosumers, num_consumers, feeder_limit, ramp_limit)
            
            # Metrics
            total_failures = sim_results['failed'].sum()
            total_steps = len(sim_results)
            failure_rate = (total_failures / total_steps) * 100
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Failures (Hours)", int(total_failures))
            col2.metric("Failure Rate", f"{failure_rate:.2f}%")
            col3.metric("Avg Market Price", f"${sim_results['price'].mean():.2f}")
            
            # Plot Net Load & Constraints
            fig_sim = go.Figure()
            
            # Grid Load
            fig_sim.add_trace(go.Scatter(x=sim_results['timestamp'], y=sim_results['net_grid_load'],
                                         mode='lines', name='Net Grid Load (kW)',
                                         line=dict(color='blue')))
            
            # Limits
            fig_sim.add_hline(y=feeder_limit, line_dash="dash", line_color="red", annotation_text="Feeder Limit (+)")
            fig_sim.add_hline(y=-feeder_limit, line_dash="dash", line_color="red", annotation_text="Feeder Limit (-)")
            
            # Highlight Failures
            failures = sim_results[sim_results['failed'] == True]
            if not failures.empty:
                fig_sim.add_trace(go.Scatter(x=failures['timestamp'], y=failures['net_grid_load'],
                                             mode='markers', name='Failure Event',
                                             marker=dict(color='red', size=8, symbol='x')))
                
            fig_sim.update_layout(title="Grid Net Load vs Constraints", xaxis_title="Time", yaxis_title="Power (kW)")
            st.plotly_chart(fig_sim, use_container_width=True)
            
            # Price Chart
            fig_price = go.Figure()
            fig_price.add_trace(go.Scatter(x=sim_results['timestamp'], y=sim_results['price'],
                                           mode='lines', name='Market Price',
                                           line=dict(color='green')))
            fig_price.update_layout(title="Dynamic Market Price", xaxis_title="Time", yaxis_title="Price ($/kWh)")
            st.plotly_chart(fig_price, use_container_width=True)

with tab2:
    st.subheader("Sensitivity Analysis (Batch Run)")
    st.write("Run multiple simulations with randomized inputs to understand drivers of grid failure.")
    
    batch_btn = st.button("Run Batch Analysis (100 Scenarios)")
    
    if batch_btn:
        import seaborn as sns
        from simulation import run_batch_simulations
        
        with st.spinner("Running 100 Simulations... (This may take a moment)"):
            batch_results = run_batch_simulations(100, weather_scenarios)
            st.success(f"Simulation Complete. Processed {len(batch_results)} scenarios.")
            
            # Show Raw Data
            st.dataframe(batch_results.head())
            
            # 1. Correlation Matrix
            st.write("### 1. Feature Correlation Matrix")
            # Select numeric columns relevant for correlation
            corr_cols = ['pv_pct', 'ev_pct', 'num_homes', 'feeder_cap', 'ramp_cap', 'max_import_kw', 'max_ramp_kw', 'total_failures']
            corr_matrix = batch_results[corr_cols].corr()
            
            fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
            st.pyplot(fig_corr)
            
            # 2. Failure Probability Heatmap (PV vs EV)
            st.write("### 2. Failure Probability Heatmap (PV% vs EV%)")
            # Binning
            batch_results['pv_bin'] = pd.cut(batch_results['pv_pct'], bins=5)
            batch_results['ev_bin'] = pd.cut(batch_results['ev_pct'], bins=5)
            
            # Create a boolean/binary failure column for probability
            batch_results['failed_bool'] = batch_results['failed'].astype(int)
            
            failure_heatmap = batch_results.pivot_table(index='pv_bin', columns='ev_bin', values='failed_bool', aggfunc='mean')
            
            fig_fail, ax_fail = plt.subplots(figsize=(10, 8))
            sns.heatmap(failure_heatmap, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_fail)
            ax_fail.set_title("Probability of Failure")
            st.pyplot(fig_fail)
            
            # 3. Scatter Plots
            st.write("### 3. Failure Distribution Scatters")
            col_scat1, col_scat2 = st.columns(2)
            
            with col_scat1:
                fig_s1 = px.scatter(batch_results, x='pv_pct', y='num_homes', color='failed', 
                                    title="PV% vs Homes (Color=Failed)",
                                    labels={'pv_pct': 'PV Percentage', 'num_homes': 'Number of Homes'})
                st.plotly_chart(fig_s1, use_container_width=True)
                
            with col_scat2:
                fig_s2 = px.scatter(batch_results, x='num_homes', y='max_ramp_kw', color='failed',
                                    title="Max Ramp Stress vs Homes",
                                    labels={'max_ramp_kw': 'Max Ramp (kW)', 'num_homes': 'Number of Homes'})
                st.plotly_chart(fig_s2, use_container_width=True)

