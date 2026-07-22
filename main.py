"""Application entry point."""

import sys

from PyQt5.QtWidgets import QApplication

from gui import ForecastApp


def main() -> int:
    """Create and run the Qt application."""
    app = QApplication(sys.argv)
    window = ForecastApp()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())