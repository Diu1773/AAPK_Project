"""Compatibility entry point for the workflow GUI."""

from .main_window_workflow import MainWindowWorkflow


class MainWindowLightCurve(MainWindowWorkflow):
    """Legacy alias that now opens the workflow main window."""

    pass
