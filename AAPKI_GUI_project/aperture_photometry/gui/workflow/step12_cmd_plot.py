"""
Step 12: CMD Plot
Loads calibrated CMD tables and opens the CMD viewer.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QVBoxLayout, QLabel, QGroupBox, QTextEdit, QMessageBox

from .step_window_base import StepWindowBase
from .step11_zeropoint_calibration import CmdViewerWindow
from ...utils.step_paths import step9_dir, step11_dir, step12_dir


class CmdPlotWindow(StepWindowBase):
    """Step 12: CMD Plot"""

    def __init__(self, params, file_manager, project_state, main_window):
        self.file_manager = file_manager
        self.viewer = None
        super().__init__(
            step_index=11,
            step_name="CMD Plot",
            params=params,
            project_state=project_state,
            main_window=main_window
        )
        self.setup_step_ui()
        self.restore_state()

    def setup_step_ui(self):
        info = QLabel("Open CMD viewer using calibrated ZP products.")
        info.setStyleSheet("QLabel { background-color: #E3F2FD; padding: 10px; border-radius: 5px; }")
        self.content_layout.addWidget(info)

        self.viewer_group = QGroupBox("CMD Viewer")
        self.viewer_layout = QVBoxLayout(self.viewer_group)
        self.viewer_placeholder = QLabel("CMD viewer will appear here after ZP calibration.")
        self.viewer_placeholder.setAlignment(Qt.AlignCenter)
        self.viewer_placeholder.setStyleSheet("QLabel { color: #666666; padding: 12px; }")
        self.viewer_layout.addWidget(self.viewer_placeholder)
        self.content_layout.addWidget(self.viewer_group)

        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("QTextEdit { font-family: monospace; font-size: 9pt; }")
        log_layout.addWidget(self.log_text)
        self.content_layout.addWidget(log_group)

    def log(self, message: str):
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")

    def open_cmd_viewer(self):
        input_dir = step11_dir(self.params.P.result_dir)
        if not input_dir.exists():
            input_dir = self.params.P.result_dir
        output_dir = step12_dir(self.params.P.result_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        wide_path = input_dir / "median_by_ID_filter_wide_cmd.csv"
        if not wide_path.exists():
            wide_path = input_dir / "median_by_ID_filter_wide.csv"
        if not wide_path.exists():
            self._reset_viewer("CMD wide CSV not found")
            QMessageBox.warning(self, "Missing Data", "CMD wide CSV not found")
            return

        idx_candidates = [
            step9_dir(self.params.P.result_dir) / "photometry_index.csv",
            step9_dir(self.params.P.result_dir) / "phot_index.csv",
            self.params.P.result_dir / "photometry_index.csv",
            self.params.P.result_dir / "phot_index.csv",
            self.params.P.result_dir / "phot" / "photometry_index.csv",
            self.params.P.result_dir / "phot" / "phot_index.csv",
        ]
        idx_path = next((p for p in idx_candidates if p.exists()), None)
        if idx_path and wide_path.stat().st_mtime < idx_path.stat().st_mtime:
            self.log(f"WARNING: {wide_path.name} is older than {idx_path.name}; rerun ZP calibration.")

        df = pd.read_csv(wide_path)
        self._reset_viewer()
        viewer = CmdViewerWindow(df, output_dir, self, embedded=True)
        self.viewer_layout.addWidget(viewer)
        self.viewer = viewer
        self.viewer_placeholder.setVisible(False)

    def _reset_viewer(self, placeholder_text: str | None = None):
        if self.viewer is not None:
            self.viewer.setParent(None)
            self.viewer.deleteLater()
            self.viewer = None
        if placeholder_text:
            self.viewer_placeholder.setText(placeholder_text)
        self.viewer_placeholder.setVisible(self.viewer is None)

    def validate_step(self) -> bool:
        input_dir = step11_dir(self.params.P.result_dir)
        if not input_dir.exists():
            input_dir = self.params.P.result_dir
        return (input_dir / "median_by_ID_filter_wide_cmd.csv").exists() or (input_dir / "median_by_ID_filter_wide.csv").exists()

    def save_state(self):
        self.project_state.store_step_data("cmd_plot", {})

    def restore_state(self):
        if self.validate_step():
            try:
                self.open_cmd_viewer()
            except Exception:
                pass
        return
