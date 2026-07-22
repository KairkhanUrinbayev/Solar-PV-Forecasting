"""PyQt user interface for Solar PV Forecasting."""

from PyQt5.QtWidgets import QFileDialog, QInputDialog, QLabel, QMainWindow, QPushButton, QTextEdit, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from data_loader import available_columns, load_data, suggest_columns
from forecast import arima_forecast, calculate_mae, lstm_forecast, lstm_future_forecast, naive_forecast
from optimization import optimise_setpoints
from plots import plot_arima, plot_lstm, plot_milp, plot_naive, plot_raw


class ForecastApp(QMainWindow):
    """Main application window and orchestration layer."""

    def __init__(self) -> None:
        super().__init__()
        self.data = None
        self.setWindowTitle("Солнечный прогноз и оптимизация")
        self.setGeometry(100, 100, 1000, 850)

        self.instruction_button = QPushButton("Инструкция по Excel")
        self.load_button = QPushButton("Загрузить Excel")
        self.raw_button = QPushButton("Временной ряд мощности")
        self.naive_button = QPushButton("Наивный прогноз")
        self.arima_button = QPushButton("ARIMA прогноз")
        self.lstm_button = QPushButton("LSTM прогноз")
        self.milp_button = QPushButton("MILP-оптимизация")
        self.status = QTextEdit(readOnly=True)
        self.canvas = FigureCanvas(Figure(figsize=(10, 5)))

        layout = QVBoxLayout()
        for button in (self.instruction_button, self.load_button, self.raw_button, self.naive_button, self.arima_button, self.lstm_button, self.milp_button):
            layout.addWidget(button)
        layout.addWidget(QLabel("Лог:"))
        layout.addWidget(self.status)
        layout.addWidget(self.canvas)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.instruction_button.clicked.connect(self.show_instructions)
        self.load_button.clicked.connect(self.load_file)
        self.raw_button.clicked.connect(self.show_raw)
        self.naive_button.clicked.connect(self.show_naive)
        self.arima_button.clicked.connect(self.show_arima)
        self.lstm_button.clicked.connect(self.show_lstm)
        self.milp_button.clicked.connect(self.show_milp)

    def log(self, message: str) -> None:
        """Append a message to the GUI log and standard output."""
        self.status.append(message)
        print(message)

    def show_instructions(self) -> None:
        self.log("Поддерживаются Excel и CSV. Программа автоматически ищет столбцы времени и мощности; если не найдёт, предложит выбрать их вручную. Рекомендуемый интервал: 15 минут.")

    def load_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите данные солнечной станции",
            "",
            "Data files (*.xlsx *.xls *.csv)",
        )
        if not path:
            return
        try:
            columns = available_columns(path)
            timestamp_column, power_column = suggest_columns(columns)
            if timestamp_column is None:
                timestamp_column, accepted = QInputDialog.getItem(
                    self, "Выбор времени", "Какой столбец содержит дату и время?", columns, 0, False
                )
                if not accepted:
                    return
            if power_column is None:
                power_column, accepted = QInputDialog.getItem(
                    self, "Выбор мощности", "Какой столбец содержит AC-мощность в кВт?", columns, 0, False
                )
                if not accepted:
                    return
            self.data = load_data(path, timestamp_column, power_column)
            self.log(f"Файл загружен и очищен: {len(self.data)} наблюдений.")
        except Exception as error:
            self.log(f"Ошибка загрузки: {error}")

    def _require_data(self):
        if self.data is None:
            raise ValueError("Сначала загрузите Excel-файл.")
        return self.data

    def _axis(self):
        self.canvas.figure.clear()
        return self.canvas.figure.add_subplot(111)

    def _draw(self) -> None:
        self.canvas.figure.tight_layout()
        self.canvas.draw()

    def show_raw(self) -> None:
        try:
            plot_raw(self._require_data(), self._axis())
            self._draw()
        except Exception as error:
            self.log(f"Ошибка: {error}")

    def show_naive(self) -> None:
        try:
            result, mae, percent = naive_forecast(self._require_data())
            plot_naive(result, self._axis(), mae, percent)
            self._draw()
            self.log(f"MAE наивного прогноза: {mae:.2f} кВт ({percent:.1f}%).")
        except Exception as error:
            self.log(f"Ошибка наивного прогноза: {error}")

    def show_arima(self) -> None:
        try:
            self.log("Обучение ARIMA…")
            actual, predicted, mae, percent = arima_forecast(self._require_data())
            plot_arima(actual, predicted, self._axis(), mae, percent)
            self._draw()
            self.log(f"MAE ARIMA: {mae:.2f} кВт ({percent:.1f}%).")
        except Exception as error:
            self.log(f"Ошибка ARIMA: {error}")

    def show_lstm(self) -> None:
        try:
            self.log("Обучение LSTM…")
            timestamps, actual, predicted, mae, percent = lstm_forecast(self._require_data())
            plot_lstm(timestamps, actual, predicted, self._axis(), mae, percent)
            self._draw()
            self.log(f"MAE LSTM: {mae:.2f} кВт ({percent:.1f}%).")
        except Exception as error:
            self.log(f"Ошибка LSTM: {error}")

    def show_milp(self) -> None:
        try:
            self.log("LSTM-прогноз и MILP-оптимизация…")
            timestamps, forecast = lstm_future_forecast(self._require_data())
            setpoints = optimise_setpoints(forecast.tolist())
            mae, percent = calculate_mae(forecast, setpoints)
            plot_milp(timestamps, forecast, setpoints, self._axis(), mae, percent)
            self._draw()
            self.log(f"MILP завершена. MAE: {mae:.2f} кВт ({percent:.1f}%).")
        except Exception as error:
            self.log(f"Ошибка MILP: {error}")
