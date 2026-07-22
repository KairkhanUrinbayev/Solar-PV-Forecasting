# Solar PV Forecasting and Inverter Optimisation

Desktop application for analysing photovoltaic AC power data, comparing forecasting models, and optimising inverter setpoints.

The current user interface is in Russian.

## Features

- Loads `.xlsx` datasets and validates required columns.
- Cleans, sorts, and visualises PV power time series.
- Seasonal naive forecasting, ARIMA forecasting, and LSTM forecasting.
- MILP optimisation of inverter setpoints with power and ramp-rate constraints.
- PyQt5 desktop interface with in-app logs and charts.

## Input data

Excel files must contain these columns in the first row:

| Column | Meaning | Example |
| --- | --- | --- |
| `datetime` | Timestamp | `2025-01-01 12:15:00` |
| `power_ac` | AC PV power in kW | `5280.7` |

The application is designed for 15-minute measurements. It removes negative values and values above the configured 99th percentile.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

ARIMA requires `pmdarima`; LSTM requires TensorFlow. MILP optimisation additionally requires the external GLPK solver. Install GLPK and make `glpsol` available in your system `PATH`. If needed, set an absolute executable path in `config.py` or the `GLPK_EXECUTABLE` environment variable.

## Project structure

```text
Solar-PV-Forecasting/
├── config.py            # Central project settings
├── data_loader.py       # Excel loading and cleaning
├── forecast.py          # Naive, ARIMA, and LSTM models
├── optimization.py      # Pyomo / GLPK MILP model
├── plots.py             # Matplotlib drawing functions
├── gui.py               # PyQt interface and workflow
├── main.py              # Application entry point
└── requirements.txt
```

## Notes

The LSTM models are intentionally trained when their buttons are clicked, matching the original project behaviour. For a production deployment, persist trained models and move training to a background worker so that the interface remains responsive.
