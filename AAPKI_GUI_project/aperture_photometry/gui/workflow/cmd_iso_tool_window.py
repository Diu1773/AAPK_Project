"""
CMD + Isochrone Tool
Open CMD viewer and Isochrone Model (Step 13) from a selected result folder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QLineEdit, QPushButton, QLabel, QMessageBox, QFileDialog
)

from .step11_zeropoint_calibration import CmdViewerWindow
from .step13_isochrone_model import IsochroneModelWindow


class _ParamsNamespaceProxy:
    def __init__(self, base, result_dir: Path):
        object.__setattr__(self, "_base", base)
        object.__setattr__(self, "result_dir", result_dir)

    def __getattr__(self, name):
        return getattr(self._base, name)

    def __setattr__(self, name, value):
        if name == "result_dir":
            object.__setattr__(self, "result_dir", value)
            return
        setattr(self._base, name, value)


class ParamsProxy:
    def __init__(self, base_params, result_dir: Path):
        self._base = base_params
        self.param_file = getattr(base_params, "param_file", None)
        self.param_hash = getattr(base_params, "param_hash", None)
        self.P = _ParamsNamespaceProxy(base_params.P, result_dir)

    def save_toml(self):
        if hasattr(self._base, "save_toml"):
            return self._base.save_toml()
        return False


class CmdIsoToolWindow(QMainWindow):
    """Tool window with CMD plot and Isochrone Model tabs."""

    def __init__(
        self,
        params,
        file_manager,
        project_state,
        main_window,
        initial_result_dir: Optional[Path] = None,
        parent=None
    ):
        super().__init__(parent)
        self.base_params = params
        self.file_manager = file_manager
        self.project_state = project_state
        self.main_window = main_window

        self.result_dir: Optional[Path] = None
        self.cmd_viewer = None
        self.iso_window = None
        self.params_proxy = None

        self.setWindowTitle("CMD + Isochrone Tool")
        self.setMinimumSize(1100, 800)

        self._build_ui()

        if initial_result_dir:
            self.load_result_dir(Path(initial_result_dir))

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Result folder selection
        row = QHBoxLayout()
        row.addWidget(QLabel("Result folder:"))
        self.result_path_edit = QLineEdit()
        self.result_path_edit.setReadOnly(True)
        row.addWidget(self.result_path_edit)
        btn_browse = QPushButton("Browse")
        btn_browse.clicked.connect(self.browse_result_dir)
        row.addWidget(btn_browse)
        btn_reload = QPushButton("Reload")
        btn_reload.clicked.connect(self.reload_result_dir)
        row.addWidget(btn_reload)
        layout.addLayout(row)

        self.info_label = QLabel("Select a result folder to load CMD data.")
        self.info_label.setStyleSheet("QLabel { color: #666666; }")
        layout.addWidget(self.info_label)

        # Tabs
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, stretch=1)

        # CMD tab
        self.cmd_tab = QWidget()
        self.cmd_layout = QVBoxLayout(self.cmd_tab)
        self.cmd_placeholder = QLabel("CMD viewer will appear here after loading a result folder.")
        self.cmd_placeholder.setAlignment(Qt.AlignCenter)
        self.cmd_placeholder.setStyleSheet("QLabel { color: #666666; padding: 12px; }")
        self.cmd_layout.addWidget(self.cmd_placeholder)
        self.tabs.addTab(self.cmd_tab, "CMD Plot")

        # Isochrone tab
        self.iso_tab = QWidget()
        self.iso_layout = QVBoxLayout(self.iso_tab)
        self.iso_placeholder = QLabel("Isochrone tools will appear here after loading a result folder.")
        self.iso_placeholder.setAlignment(Qt.AlignCenter)
        self.iso_placeholder.setStyleSheet("QLabel { color: #666666; padding: 12px; }")
        self.iso_layout.addWidget(self.iso_placeholder)
        self.tabs.addTab(self.iso_tab, "Isochrone (Step 13)")

    def browse_result_dir(self):
        start_dir = str(getattr(self.base_params.P, "result_dir", Path.cwd()))
        selected = QFileDialog.getExistingDirectory(self, "Select Result Folder", start_dir)
        if selected:
            self.load_result_dir(Path(selected))

    def reload_result_dir(self):
        if self.result_dir is None:
            self.browse_result_dir()
            return
        self.load_result_dir(self.result_dir)

    def load_result_dir(self, result_dir: Path):
        self.result_dir = result_dir
        self.result_path_edit.setText(str(result_dir))

        wide_path = self._find_cmd_wide_csv(result_dir)
        if wide_path is None:
            self._reset_cmd_viewer("CMD wide CSV not found in selected folder.")
            self._reset_iso_tab("CMD wide CSV not found in selected folder.")
            self.info_label.setText("CMD wide CSV not found.")
            QMessageBox.warning(
                self, "Missing Data",
                "CMD wide CSV not found.\n"
                "Expected: median_by_ID_filter_wide_cmd.csv or median_by_ID_filter_wide.csv"
            )
            return

        try:
            df = pd.read_csv(wide_path)
        except Exception as e:
            self._reset_cmd_viewer("Failed to load CMD CSV.")
            self._reset_iso_tab("Failed to load CMD CSV.")
            QMessageBox.critical(
                self, "Load Error",
                f"Failed to load CMD CSV:\n{wide_path}\n\n{e}"
            )
            return

        self.info_label.setText(f"Loaded: {wide_path}")
        self._load_cmd_viewer(df, result_dir)
        self._load_iso_window(result_dir)

    def _load_cmd_viewer(self, df: pd.DataFrame, result_dir: Path):
        self._reset_cmd_viewer()
        viewer = CmdViewerWindow(df, result_dir, parent=self.cmd_tab, embedded=True)
        self.cmd_layout.addWidget(viewer)
        self.cmd_viewer = viewer
        self.cmd_placeholder.setVisible(False)

    def _load_iso_window(self, result_dir: Path):
        self._reset_iso_tab()
        self.params_proxy = ParamsProxy(self.base_params, result_dir)
        iso_window = IsochroneModelWindow(
            self.params_proxy,
            self.file_manager,
            self.project_state,
            self.main_window
        )
        iso_window.setWindowFlags(Qt.Widget)
        for btn in (iso_window.btn_previous, iso_window.btn_next, iso_window.btn_complete):
            btn.setEnabled(False)
            btn.setVisible(False)
        self.iso_layout.addWidget(iso_window)
        self.iso_window = iso_window
        self.iso_placeholder.setVisible(False)

    def _reset_cmd_viewer(self, placeholder_text: Optional[str] = None):
        if self.cmd_viewer is not None:
            self.cmd_viewer.setParent(None)
            self.cmd_viewer.deleteLater()
            self.cmd_viewer = None
        if placeholder_text:
            self.cmd_placeholder.setText(placeholder_text)
        self.cmd_placeholder.setVisible(self.cmd_viewer is None)

    def _reset_iso_tab(self, placeholder_text: Optional[str] = None):
        if self.iso_window is not None:
            self.iso_window.setParent(None)
            self.iso_window.deleteLater()
            self.iso_window = None
        if placeholder_text:
            self.iso_placeholder.setText(placeholder_text)
        self.iso_placeholder.setVisible(self.iso_window is None)

    @staticmethod
    def _find_cmd_wide_csv(result_dir: Path) -> Optional[Path]:
        filenames = [
            "median_by_ID_filter_wide_cmd.csv",
            "median_by_ID_filter_wide.csv",
        ]
        candidates = [result_dir / name for name in filenames]
        existing = [p for p in candidates if p.exists()]
        if existing:
            return max(existing, key=lambda p: p.stat().st_mtime)

        found = []
        for name in filenames:
            found.extend(result_dir.rglob(name))
        if not found:
            return None
        return max(found, key=lambda p: p.stat().st_mtime)
