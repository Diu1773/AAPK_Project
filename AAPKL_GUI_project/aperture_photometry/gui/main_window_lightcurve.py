"""
Main window for the light curve GUI.
"""

from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QTabWidget

from .light_curve import VariableStarTab, EclipseTab, AsteroidTab


class MainWindowLightCurve(QMainWindow):
    """Main application window for light curve workflows."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AAPKI Light Curve GUI")
        self.setMinimumSize(1100, 800)
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        layout = QVBoxLayout(central)

        tabs = QTabWidget()
        tabs.addTab(VariableStarTab(), "Variable Star")
        tabs.addTab(EclipseTab(), "Eclipsing")
        tabs.addTab(AsteroidTab(), "Asteroid")

        layout.addWidget(tabs)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready")
