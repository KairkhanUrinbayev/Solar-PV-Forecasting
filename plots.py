"""Pure Matplotlib plotting functions for the application."""


def _style_axis(axis, title: str) -> None:
    axis.set_title(title)
    axis.set_xlabel("Время")
    axis.set_ylabel("Мощность (кВт)")
    axis.legend()
    axis.grid(True, alpha=0.3)


def plot_raw(data, axis) -> None:
    """Draw the original power time series."""
    axis.clear()
    axis.plot(data["timestamp"], data["power_ac"], label="Мощность", color="orange", linewidth=0.7)
    _style_axis(axis, "Временной ряд мощности")


def plot_naive(data, axis, mae: float, mae_percent: float) -> None:
    """Draw actual values and the seasonal naive forecast."""
    axis.clear()
    axis.plot(data["timestamp"], data["power_ac"], label="Фактическая", linewidth=0.7)
    axis.plot(data["timestamp"], data["naive"], label="Наивный прогноз", linestyle="--", linewidth=0.7)
    _style_axis(axis, f"Наивный прогноз\nMAE = {mae:.2f} кВт ({mae_percent:.1f}%)")


def plot_arima(actual, predicted, axis, mae: float, mae_percent: float) -> None:
    """Draw ARIMA prediction against held-out observations."""
    axis.clear()
    axis.plot(actual.index, actual.values, label="Фактическая", linewidth=0.7)
    axis.plot(predicted.index, predicted.values, label="ARIMA прогноз", linestyle="--", linewidth=0.7)
    _style_axis(axis, f"ARIMA прогноз\nMAE = {mae:.2f} кВт ({mae_percent:.1f}%)")


def plot_lstm(timestamps, actual, predicted, axis, mae: float, mae_percent: float) -> None:
    """Draw LSTM fitted-value evaluation."""
    axis.clear()
    axis.plot(timestamps, actual, label="Фактическая", linewidth=0.7)
    axis.plot(timestamps, predicted, label="LSTM прогноз", linestyle="--", linewidth=0.7)
    _style_axis(axis, f"LSTM прогноз\nMAE = {mae:.2f} кВт ({mae_percent:.1f}%)")


def plot_milp(timestamps, forecast, setpoints, axis, mae: float, mae_percent: float) -> None:
    """Draw LSTM future forecast and MILP inverter setpoints."""
    axis.clear()
    axis.plot(timestamps, forecast, label="LSTM прогноз", linewidth=0.7)
    axis.plot(timestamps, setpoints, label="MILP уставка", linestyle="--", linewidth=0.7)
    _style_axis(axis, f"MILP-оптимизация уставок\nMAE = {mae:.2f} кВт ({mae_percent:.1f}%)")
