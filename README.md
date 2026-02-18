# Distributed Energy Trading Simulation

This project simulates a decentralized peer-to-peer (P2P) energy trading network, modeling weather-driven energy production and household energy consumption.

## Features

### 1. Grid Performance Simulation
- **Weather Generation**: Uses a Monte Carlo approach to generate synthetic weather scenarios (Temp, Solar, Wind) tailored to Sugar Land, TX climate patterns (July).
- **Simulation Engine**: Models 2 weeks of hourly energy flow.
- **Visualizations**: 
    - Monte Carlo Weather Paths.
    - Grid Performance (Net Load vs Constraints).
    - Market Price dynamics.

### 2. Sensitivity Analysis
- **Batch Simulation**: Run 100+ scenarios with randomized parameters.
- **Correlation Matrix**: Identify factors driving grid failures.
- **Failure Heatmap**: Visualize failure probability matrices.

## Getting Started

### Prerequisites

Ensure you have Python installed. Install dependencies:

```bash
pip install -r requirements.txt
```

### Running the Application

To start the interactive dashboard:

```bash
# If running from the root of the repository
python -m streamlit run app.py
```

## Project Structure

- `app.py`: Main Streamlit application.
- `simulation.py`: Core simulation logic (Households, Grid, Market).
- `weather_utils.py`: Weather generation and Monte Carlo logic.
- `requirements.txt`: Python dependencies.
