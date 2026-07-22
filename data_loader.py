"""Flexible loading and validation of photovoltaic power data."""

from pathlib import Path

import pandas as pd

from config import (
    MAX_POWER_QUANTILE,
    POWER_COLUMN,
    POWER_COLUMN_ALIASES,
    TIMESTAMP_COLUMN,
    TIMEZONE_OFFSET_HOURS,
    TIMESTAMP_COLUMN_ALIASES,
)


def _normalise_name(name: object) -> str:
    """Return a comparable version of a column name."""
    return "".join(character for character in str(name).lower() if character.isalnum())


def _read_tabular_file(path: str | Path) -> pd.DataFrame:
    """Read an Excel workbook or CSV file based on its extension."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Файл не найден: {file_path}")
    if file_path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(file_path)
    if file_path.suffix.lower() == ".csv":
        return pd.read_csv(file_path, sep=None, engine="python")
    raise ValueError("Поддерживаются только файлы Excel (.xlsx, .xls) и CSV (.csv).")


def available_columns(path: str | Path) -> list[str]:
    """Return the source column names so the GUI can offer manual mapping."""
    return [str(column) for column in _read_tabular_file(path).columns]


def suggest_columns(columns: list[str]) -> tuple[str | None, str | None]:
    """Suggest timestamp and AC-power columns using common public-data labels."""
    normalised = {_normalise_name(column): column for column in columns}
    timestamp = next((normalised.get(_normalise_name(alias)) for alias in TIMESTAMP_COLUMN_ALIASES if _normalise_name(alias) in normalised), None)
    power = next((normalised.get(_normalise_name(alias)) for alias in POWER_COLUMN_ALIASES if _normalise_name(alias) in normalised), None)
    return timestamp, power


def load_data(
    path: str | Path,
    timestamp_column: str | None = None,
    power_column: str | None = None,
    timezone_offset_hours: int = TIMEZONE_OFFSET_HOURS,
) -> pd.DataFrame:
    """Load, map, clean, and time-sort a PV dataset from Excel or CSV.

    Source column names can be supplied explicitly. If omitted, known labels
    such as ``Timestamp`` and ``AC Power`` are detected automatically.
    """
    data = _read_tabular_file(path)
    suggested_time, suggested_power = suggest_columns([str(column) for column in data.columns])
    timestamp_column = timestamp_column or suggested_time
    power_column = power_column or suggested_power
    if timestamp_column not in data.columns or power_column not in data.columns:
        raise ValueError("Не удалось определить столбцы времени и мощности. Выберите их вручную.")
    if timestamp_column == power_column:
        raise ValueError("Столбцы времени и мощности должны быть разными.")

    data = data.rename(columns={timestamp_column: TIMESTAMP_COLUMN, power_column: POWER_COLUMN}).copy()
    data[TIMESTAMP_COLUMN] = pd.to_datetime(data[TIMESTAMP_COLUMN], errors="coerce")
    if timezone_offset_hours:
        data[TIMESTAMP_COLUMN] += pd.Timedelta(hours=timezone_offset_hours)
    data[POWER_COLUMN] = pd.to_numeric(data[POWER_COLUMN], errors="coerce")
    data = data.dropna(subset=[TIMESTAMP_COLUMN, POWER_COLUMN])

    upper_limit = data[POWER_COLUMN].quantile(MAX_POWER_QUANTILE)
    data = data.loc[(data[POWER_COLUMN] >= 0) & (data[POWER_COLUMN] <= upper_limit)]
    data = data.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)

    if data.empty:
        raise ValueError("После очистки не осталось пригодных наблюдений.")
    return data


# Backward-compatible name for code that already imports load_excel.
load_excel = load_data
