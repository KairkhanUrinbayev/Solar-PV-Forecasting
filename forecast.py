"""Forecasting models for photovoltaic AC power."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

from config import (
    ARIMA_LOOKBACK_DAYS,
    ARIMA_MIN_SAMPLES,
    ARIMA_TEST_POINTS,
    FORECAST_HORIZON,
    LSTM_BATCH_SIZE,
    LSTM_EPOCHS,
    LSTM_LOOKBACK_DAYS,
    LSTM_UNITS,
    LSTM_WINDOW_SIZE,
    NAIVE_SHIFT,
    POWER_COLUMN,
    SAMPLES_PER_HOUR,
    TIMESTAMP_COLUMN,
)


def calculate_mae(actual: np.ndarray | pd.Series, predicted: np.ndarray | pd.Series) -> tuple[float, float]:
    """Return MAE in kW and as a percentage of the largest observed value."""
    actual_values = np.asarray(actual).reshape(-1)
    predicted_values = np.asarray(predicted).reshape(-1)
    mae = float(mean_absolute_error(actual_values, predicted_values))
    maximum = float(max(np.max(actual_values), np.max(predicted_values)))
    return mae, (mae / maximum * 100 if maximum > 0 else 0.0)


def naive_forecast(data: pd.DataFrame, shift: int = NAIVE_SHIFT) -> tuple[pd.DataFrame, float, float]:
    """Create a seasonal naive forecast using power from ``shift`` observations ago."""
    if len(data) <= shift:
        raise ValueError(f"Для наивного прогноза нужно больше {shift} наблюдений.")
    result = data.copy()
    result["naive"] = result[POWER_COLUMN].shift(shift)
    result = result.dropna(subset=["naive"])
    mae, mae_percent = calculate_mae(result[POWER_COLUMN], result["naive"])
    return result, mae, mae_percent


def arima_forecast(data: pd.DataFrame) -> tuple[pd.Series, pd.Series, float, float]:
    """Fit auto-ARIMA on recent data and forecast a held-out final segment."""
    try:
        import pmdarima as pm
    except ImportError as error:
        raise ImportError("Для ARIMA установите pmdarima: pip install pmdarima") from error

    recent_start = data[TIMESTAMP_COLUMN].max() - pd.Timedelta(days=ARIMA_LOOKBACK_DAYS)
    recent = data.loc[data[TIMESTAMP_COLUMN] >= recent_start].sort_values(TIMESTAMP_COLUMN)
    series = recent.set_index(TIMESTAMP_COLUMN)[POWER_COLUMN].dropna()
    if len(series) < ARIMA_MIN_SAMPLES:
        raise ValueError(f"Недостаточно данных для ARIMA: нужно минимум {ARIMA_MIN_SAMPLES} точек.")

    test_size = min(ARIMA_TEST_POINTS, max(1, len(series) // 4))
    train, actual = series.iloc[:-test_size], series.iloc[-test_size:]
    model = pm.auto_arima(train, seasonal=False, suppress_warnings=True, stepwise=True)
    predicted = pd.Series(np.maximum(model.predict(n_periods=test_size), 0), index=actual.index)
    mae, mae_percent = calculate_mae(actual, predicted)
    return actual, predicted, mae, mae_percent


def prepare_lstm_data(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, MinMaxScaler, pd.Series, pd.DataFrame]:
    """Build two-feature (power and hour) sequences for LSTM training."""
    start = data[TIMESTAMP_COLUMN].max() - pd.Timedelta(days=LSTM_LOOKBACK_DAYS)
    recent = data.loc[data[TIMESTAMP_COLUMN] >= start].sort_values(TIMESTAMP_COLUMN).copy()
    if len(recent) <= LSTM_WINDOW_SIZE:
        raise ValueError(f"Для LSTM нужно больше {LSTM_WINDOW_SIZE} наблюдений.")

    recent["hour_norm"] = (recent[TIMESTAMP_COLUMN].dt.hour * SAMPLES_PER_HOUR + recent[TIMESTAMP_COLUMN].dt.minute // 15) / 95.0
    scaler = MinMaxScaler()
    scaled_power = scaler.fit_transform(recent[[POWER_COLUMN]])
    hour_values = recent["hour_norm"].to_numpy().reshape(-1, 1)

    features, targets = [], []
    for index in range(LSTM_WINDOW_SIZE, len(recent)):
        features.append(np.concatenate((scaled_power[index - LSTM_WINDOW_SIZE:index], hour_values[index - LSTM_WINDOW_SIZE:index]), axis=1))
        targets.append(scaled_power[index])
    return np.asarray(features), np.asarray(targets), scaler, recent[TIMESTAMP_COLUMN].iloc[LSTM_WINDOW_SIZE:], recent


def build_lstm_model(input_shape: tuple[int, int]):
    """Create and compile the LSTM regression model."""
    try:
        from tensorflow.keras import Input, Sequential
        from tensorflow.keras.layers import Dense, LSTM
    except ImportError as error:
        raise ImportError("Для LSTM установите TensorFlow: pip install tensorflow") from error
    model = Sequential([Input(shape=input_shape), LSTM(LSTM_UNITS), Dense(1)])
    model.compile(optimizer="adam", loss="mae")
    return model


def train_lstm_model(model, features: np.ndarray, targets: np.ndarray) -> None:
    """Train an LSTM silently using project configuration."""
    model.fit(features, targets, epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH_SIZE, verbose=0)


def lstm_forecast(data: pd.DataFrame) -> tuple[pd.Series, np.ndarray, np.ndarray, float, float]:
    """Train LSTM on recent observations and return fitted-value evaluation."""
    features, targets, scaler, timestamps, _ = prepare_lstm_data(data)
    model = build_lstm_model(features.shape[1:])
    train_lstm_model(model, features, targets)
    predicted = scaler.inverse_transform(model.predict(features, verbose=0))
    actual = scaler.inverse_transform(targets)
    mae, mae_percent = calculate_mae(actual, predicted)
    return timestamps, actual.ravel(), predicted.ravel(), mae, mae_percent


def lstm_future_forecast(data: pd.DataFrame, horizon: int = FORECAST_HORIZON) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Train an LSTM and recursively predict the next ``horizon`` 15-minute values."""
    features, targets, scaler, _, recent = prepare_lstm_data(data)
    model = build_lstm_model(features.shape[1:])
    train_lstm_model(model, features, targets)

    scaled_power = scaler.transform(recent[[POWER_COLUMN]])
    hours = recent["hour_norm"].to_numpy().tolist()
    generated = []
    for _ in range(horizon):
        power_window = scaled_power[-LSTM_WINDOW_SIZE:]
        hour_window = np.asarray(hours[-LSTM_WINDOW_SIZE:]).reshape(-1, 1)
        feature = np.concatenate((power_window, hour_window), axis=1)[None, :, :]
        next_scaled = float(model.predict(feature, verbose=0)[0, 0])
        generated.append(next_scaled)
        scaled_power = np.vstack((scaled_power, [[next_scaled]]))
        hours.append((hours[-1] + 1 / 96.0) % 1.0)

    start = recent[TIMESTAMP_COLUMN].iloc[-1] + pd.Timedelta(minutes=15)
    timestamps = pd.date_range(start=start, periods=horizon, freq="15min")
    values = np.maximum(scaler.inverse_transform(np.asarray(generated).reshape(-1, 1)).ravel(), 0)
    return timestamps, values
