"""
BIL Security ML - Main Application Entry Point

Launch the intrusion detection GUI application.
"""

import sys
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from src.gui import MainWindow


def main():
    """Launch the application."""
    # High DPI support
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    
    app = QApplication(sys.argv)
    app.setApplicationName("BIL Security ML")
    app.setOrganizationName("COMP4560")
    app.setApplicationVersion("1.0.0")
    
    # Set application style
    app.setStyle("Fusion")
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
