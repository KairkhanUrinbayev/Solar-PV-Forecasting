"""Central configuration for the Solar PV Forecasting application."""

# Input data preprocessing
TIMESTAMP_COLUMN = "timestamp"
POWER_COLUMN = "power_ac"
TIMEZONE_OFFSET_HOURS = 0  # Keep source times unchanged unless explicitly configured.
MAX_POWER_QUANTILE = 0.99

# Common labels used by public PV / SCADA datasets. Matching ignores case,
# spaces, underscores, and punctuation.
TIMESTAMP_COLUMN_ALIASES = (
    "datetime", "timestamp", "date time", "date_time", "date", "time",
    "time stamp", "utc time", "local time",
)
POWER_COLUMN_ALIASES = (
    "power_ac", "ac power", "active power", "active power kw", "pv power",
    "pv output", "output power", "power", "generation", "acpower", "ac power kw",
)

# Forecasting
NAIVE_SHIFT = 672  # 7 days for 15-minute measurements
ARIMA_LOOKBACK_DAYS = 14
ARIMA_MIN_SAMPLES = 100
ARIMA_TEST_POINTS = 48
LSTM_LOOKBACK_DAYS = 7
LSTM_WINDOW_SIZE = 48
LSTM_UNITS = 64
LSTM_EPOCHS = 5
LSTM_BATCH_SIZE = 32
FORECAST_HORIZON = 96  # 24 hours for 15-minute measurements
SAMPLES_PER_HOUR = 4

# MILP optimisation
INVERTER_MAX_POWER_KW = 8500.0
RAMP_LIMIT_KW = 100.0
GLPK_EXECUTABLE = None  # Set an absolute path here only if GLPK is not in PATH.
