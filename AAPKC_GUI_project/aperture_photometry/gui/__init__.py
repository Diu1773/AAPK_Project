"""GUI components for aperture photometry application."""

from .main_window_workflow import MainWindowWorkflow

# Backward-compatible alias.
MainWindow = MainWindowWorkflow

__all__ = ["MainWindowWorkflow", "MainWindow"]
