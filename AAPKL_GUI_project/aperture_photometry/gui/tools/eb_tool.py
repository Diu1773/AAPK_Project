"""
Eclipsing Binary Analysis Tool

Features:
  1. Period scan (LS + BLS) + refinement
  2. Phase-fold viewer with primary / secondary eclipse identification
  3. Eclipse depth & duration measurement (parabola/trapezoid fit)
  4. O-C diagram with eclipse timing
  5. Basic parameter estimates: q (mass ratio via Kopal), inclination
  6. Epoch-to-epoch eclipse depth trend

Note: Full WD / PHOEBE modelling is out of scope here.
      This tool handles single-night or multi-night eclipse timing analysis.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

_CORR_MODE_RE = re.compile(r"lightcurve_.*?_(global|color|offset|raw)\b", re.IGNORECASE)
_CORR_MODE_LABELS = {
    "global": "Global ensemble",
    "color": "Color-dependent",
    "offset": "Nightly offset",
    "raw": "Raw",
}


def _detect_corr_mode_from_df(df: pd.DataFrame, filename: str) -> str:
    if "correction_mode" in df.columns:
        vals = df["correction_mode"].dropna().astype(str).str.strip().str.lower()
        if not vals.empty:
            key = vals.iloc[0]
            if key:
                return _CORR_MODE_LABELS.get(key, key)
    m = _CORR_MODE_RE.search(filename)
    if m:
        return _CORR_MODE_LABELS.get(m.group(1).lower(), m.group(1))
    return ""


def _collect_mag_options(df: pd.DataFrame, time_mask: np.ndarray, corr_tag: str = "") -> list[tuple[str, str, np.ndarray]]:
    """Prefer canonical raw/corrected differential columns."""
    options: list[tuple[str, str, np.ndarray]] = []

    if corr_tag == "Raw":
        for col in ("diff_mag_raw", "diff_mag"):
            if col in df.columns:
                arr = pd.to_numeric(df[col], errors="coerce").to_numpy(float)[time_mask]
                if np.any(np.isfinite(arr)):
                    return [(f"Raw: {col}", col, arr)]

    if "diff_mag_raw" in df.columns or "diff_mag_corr" in df.columns:
        for col, label in (("diff_mag_raw", "raw"), ("diff_mag_corr", "corrected")):
            if col in df.columns:
                arr = pd.to_numeric(df[col], errors="coerce").to_numpy(float)[time_mask]
                if np.any(np.isfinite(arr)):
                    options.append((label, col, arr))
        if options:
            return options

    fallback_raw = ["mag_raw", "raw_mag", "inst_mag", "mag"]
    fallback_corr = ["mag_corr", "corr_mag", "calibrated_mag", "mag_ensemble_corr"]
    for col in fallback_raw:
        if col in df.columns:
            arr = pd.to_numeric(df[col], errors="coerce").to_numpy(float)[time_mask]
            if np.any(np.isfinite(arr)):
                options.append((f"Raw: {col}", col, arr))
    for col in fallback_corr:
        if col in df.columns:
            arr = pd.to_numeric(df[col], errors="coerce").to_numpy(float)[time_mask]
            if np.any(np.isfinite(arr)):
                options.append((f"Corr: {col}", col, arr))
    return options


def _series_rank(corr_tag: str, mag_col: str, source_name: str) -> tuple[int, int, str]:
    mode_order = {
        "Global ensemble": 0,
        "Color-dependent": 1,
        "Nightly offset": 2,
        "Raw": 3,
    }
    corr_order = 0 if any(x in mag_col for x in ("corr", "cal")) else 1
    return mode_order.get(corr_tag, 9), corr_order, source_name.lower()


def _source_priority(source_name: str) -> tuple[int, int, str]:
    lower = source_name.lower()
    is_combined = 1 if "_combined_" in lower else 0
    is_current = 0 if "_current" in lower else 1
    return is_combined, is_current, lower


def _describe_series(corr_tag: str, mag_col: str) -> str:
    if corr_tag == "Raw":
        return "Raw"
    if corr_tag == "Nightly offset":
        return "Offset | corrected" if mag_col == "diff_mag_corr" else "Offset | raw"
    if corr_tag == "Color-dependent":
        return "Color | corrected" if mag_col == "diff_mag_corr" else "Color | raw"
    if corr_tag == "Global ensemble":
        return "Global | corrected" if mag_col == "diff_mag_corr" else "Global | raw"
    if "corr" in mag_col or "cal" in mag_col:
        return "Corrected"
    return "Raw"

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QPushButton, QLabel, QDoubleSpinBox, QSpinBox,
    QCheckBox, QComboBox, QTabWidget, QTextEdit, QTableWidget, QTableWidgetItem,
    QHeaderView, QFileDialog, QSplitter, QMessageBox,
    QColorDialog, QDialog, QGridLayout,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor

_FILTER_COLORS = [
    "#E65100", "#1E88E5", "#43A047", "#8E24AA",
    "#00ACC1", "#F06292", "#FB8C00", "#795548",
]
_NAMED_FILTER_COLORS = {
    "u": "#1f77b4", "b": "#1f77b4", "B": "#1f77b4",
    "g": "#2ca02c", "v": "#2ca02c", "V": "#2ca02c",
    "r": "#d62728", "R": "#d62728",
    "i": "#9467bd", "I": "#9467bd",
    "z": "#8c564b",
    "H": "#17becf", "J": "#bcbd22",
    "clear": "#7f7f7f", "l": "#7f7f7f", "unknown": "#7f7f7f",
}

def _filt_color(filt: str, idx: int) -> str:
    return _NAMED_FILTER_COLORS.get(filt, _FILTER_COLORS[idx % len(_FILTER_COLORS)])


def _resolve_check_filter(filters) -> str | None:
    if filters is None:
        return None
    unique_filters = sorted({str(f) for f in filters if str(f).strip() and str(f).lower() != "nan"})
    return unique_filters[0] if len(unique_filters) == 1 else None

from ..workflow.step12_period_analysis import PeriodAnalysisWorker


def _load_check_star_for_plot(result_dir: Path, filt: str | None = None):
    """Load check star CSV from step10 output for plotting. Returns (check_id, df_or_None)."""
    try:
        from ..workflow.step10_light_curve_builder import _load_check_star_csv
        check_id, df = _load_check_star_csv(result_dir, filt=filt)
        return check_id, (df if not df.empty else None)
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class EclipsingBinaryToolWindow(QWidget):
    """Eclipsing Binary Analysis Tool."""

    def __init__(self, params, project_state, parent=None):
        super().__init__(parent)
        self.params = params
        self.project_state = project_state
        self.lc_data: Optional[dict] = None
        self.series_options: dict[str, dict] = {}
        self.scan_result: Optional[dict] = None
        self._scan_worker: Optional[PeriodAnalysisWorker] = None
        self.filter_colors: dict = {}      # user-customized per-filter colors
        self.filter_visibility: dict = {}  # True=visible, False=hidden

        self.setWindowTitle("Eclipsing Binary Analysis")
        self.resize(1200, 800)
        self._build_ui()
        self._load_lc_from_step()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)

        header = QLabel(
            "<b>Eclipsing Binary Analysis</b> — LS + BLS period scan → phase fold → "
            "primary/secondary eclipse timing → O-C diagram → depth/duration measurement"
        )
        header.setStyleSheet("QLabel { background: #FFF3E0; padding: 8px; border-radius: 4px; }")
        header.setWordWrap(True)
        root.addWidget(header)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter, 1)

        # ---- Left panel ----
        left = QWidget()
        left.setMaximumWidth(300)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(4, 4, 4, 4)
        splitter.addWidget(left)

        # Light curve
        lc_group = QGroupBox("Light Curve")
        lc_form = QFormLayout(lc_group)
        self.lc_status = QLabel("Not loaded")
        self.lc_status.setWordWrap(True)
        lc_form.addRow("Status:", self.lc_status)
        btn_auto = QPushButton("Load from Step 10/11")
        btn_auto.clicked.connect(self._load_lc_from_step)
        lc_form.addRow(btn_auto)
        btn_open = QPushButton("Open CSV…")
        btn_open.clicked.connect(self._open_csv)
        lc_form.addRow(btn_open)
        self.mag_col_combo = QComboBox()
        self.mag_col_combo.setEnabled(False)
        self.mag_col_combo.currentIndexChanged.connect(self._on_mag_col_changed)
        lc_form.addRow("Use data:", self.mag_col_combo)
        ll.addWidget(lc_group)

        # Filter display controls
        filt_group = QGroupBox("Filter Display")
        filt_form = QFormLayout(filt_group)
        btn_filt_browser = QPushButton("Browse Colors / Visibility…")
        btn_filt_browser.clicked.connect(self.show_filter_color_browser)
        filt_form.addRow(btn_filt_browser)
        ll.addWidget(filt_group)

        # Scan
        scan_group = QGroupBox("Period Scan")
        scan_form = QFormLayout(scan_group)
        self.min_p = QDoubleSpinBox()
        self.min_p.setRange(0.001, 500); self.min_p.setDecimals(4)
        self.min_p.setValue(0.05); self.min_p.setSuffix(" d")
        scan_form.addRow("P min:", self.min_p)
        self.max_p = QDoubleSpinBox()
        self.max_p.setRange(0.01, 2000); self.max_p.setDecimals(4)
        self.max_p.setValue(20.0); self.max_p.setSuffix(" d")
        scan_form.addRow("P max:", self.max_p)
        self.spp = QSpinBox()
        self.spp.setRange(5, 50); self.spp.setValue(10)
        scan_form.addRow("Samples/peak:", self.spp)
        method_widget = QWidget()
        method_layout = QHBoxLayout(method_widget)
        method_layout.setContentsMargins(0, 0, 0, 0)
        self.chk_ls = QCheckBox("LS"); self.chk_ls.setChecked(True)
        self.chk_pdm = QCheckBox("PDM"); self.chk_pdm.setChecked(True)
        self.chk_bls = QCheckBox("BLS"); self.chk_bls.setChecked(True)
        for chk in [self.chk_ls, self.chk_pdm, self.chk_bls]:
            method_layout.addWidget(chk)
        method_layout.addStretch()
        scan_form.addRow("Methods:", method_widget)
        self.pdm_bins = QSpinBox()
        self.pdm_bins.setRange(5, 100); self.pdm_bins.setValue(20)
        scan_form.addRow("PDM bins:", self.pdm_bins)
        btn_scan = QPushButton("Scan")
        btn_scan.setStyleSheet(
            "QPushButton { background: #E65100; color: white; font-weight: bold; padding: 6px; }"
        )
        btn_scan.clicked.connect(self._run_scan)
        scan_form.addRow(btn_scan)
        self.scan_status = QLabel("")
        self.scan_status.setStyleSheet("font-size: 8pt; color: #555;")
        scan_form.addRow(self.scan_status)
        ll.addWidget(scan_group)

        # Phase plot controls
        phase_group = QGroupBox("Phase Plot")
        phase_form = QFormLayout(phase_group)
        self.phase_p = QDoubleSpinBox()
        self.phase_p.setRange(0.00001, 10000); self.phase_p.setDecimals(8)
        self.phase_p.setValue(1.0); self.phase_p.setSuffix(" d")
        self.phase_p.valueChanged.connect(self._update_phase_plot)
        phase_form.addRow("Period:", self.phase_p)
        self.t0_edit = QDoubleSpinBox()
        self.t0_edit.setRange(2400000, 2600000); self.t0_edit.setDecimals(6)
        self.t0_edit.setValue(2458000.0)
        self.t0_edit.valueChanged.connect(self._update_phase_plot)
        phase_form.addRow("T₀ (BJD):", self.t0_edit)
        # Half-period toggle (EB: show at P/2 if secondary is at 0.5 phase)
        self.chk_half = QCheckBox("Show P/2 phase")
        self.chk_half.toggled.connect(self._update_phase_plot)
        phase_form.addRow(self.chk_half)
        btn_detect = QPushButton("Detect Primary Min")
        btn_detect.clicked.connect(self._detect_primary_min)
        phase_form.addRow(btn_detect)
        ll.addWidget(phase_group)

        # Eclipse measure
        meas_group = QGroupBox("Eclipse Measurement")
        meas_form = QFormLayout(meas_group)
        btn_meas = QPushButton("Fit Eclipse (parabola)")
        btn_meas.setStyleSheet(
            "QPushButton { background: #1976D2; color: white; font-weight: bold; padding: 5px; }"
        )
        btn_meas.clicked.connect(self._measure_eclipse)
        meas_form.addRow(btn_meas)
        self.meas_label = QLabel("")
        self.meas_label.setWordWrap(True)
        self.meas_label.setStyleSheet(
            "QLabel { background: #FFF8E1; padding: 6px; border-radius: 4px; "
            "font-family: monospace; font-size: 8pt; }"
        )
        meas_form.addRow(self.meas_label)
        ll.addWidget(meas_group)

        ll.addStretch()

        self.btn_log_toggle = QPushButton("Log ▼")
        self.btn_log_toggle.setCheckable(True)
        self.btn_log_toggle.setChecked(False)
        self.btn_log_toggle.setStyleSheet(
            "QPushButton { text-align: left; font-size: 8pt; padding: 2px 6px; }"
        )
        self.btn_log_toggle.toggled.connect(self._toggle_log)
        ll.addWidget(self.btn_log_toggle)
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(150)
        self.log_box.setStyleSheet("font-family: monospace; font-size: 8pt;")
        self.log_box.hide()
        ll.addWidget(self.log_box)

        # ---- Right panel ----
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(4, 4, 4, 4)
        splitter.addWidget(right)
        splitter.setSizes([290, 910])

        self.tabs = QTabWidget()
        rl.addWidget(self.tabs)

        # Periodogram tab
        pg_tab = QWidget()
        pgl = QVBoxLayout(pg_tab)
        self.pg_canvas = FigureCanvas(Figure(figsize=(9, 4)))
        pgl.addWidget(NavigationToolbar(self.pg_canvas, pg_tab))
        pgl.addWidget(self.pg_canvas)
        self.tabs.addTab(pg_tab, "Periodogram")

        # Phase plot tab
        ph_tab = QWidget()
        phl = QVBoxLayout(ph_tab)
        self.ph_canvas = FigureCanvas(Figure(figsize=(9, 5)))
        phl.addWidget(NavigationToolbar(self.ph_canvas, ph_tab))
        phl.addWidget(self.ph_canvas)
        self.tabs.addTab(ph_tab, "Phase Plot")

        # O-C tab
        oc_tab = self._build_oc_tab()
        self.tabs.addTab(oc_tab, "O-C (Eclipse Timing)")

    def _build_oc_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        hdr = QHBoxLayout()
        hdr.addWidget(QLabel("T₀ (BJD):"))
        self.oc_t0 = QDoubleSpinBox()
        self.oc_t0.setRange(2400000, 2600000); self.oc_t0.setDecimals(6)
        self.oc_t0.setValue(2458000.0); self.oc_t0.setMinimumWidth(130)
        self.oc_t0.valueChanged.connect(self._recompute_oc)
        hdr.addWidget(self.oc_t0)
        hdr.addWidget(QLabel("P (d):"))
        self.oc_p = QDoubleSpinBox()
        self.oc_p.setRange(0.00001, 10000); self.oc_p.setDecimals(8)
        self.oc_p.setValue(1.0); self.oc_p.setMinimumWidth(120)
        self.oc_p.valueChanged.connect(self._recompute_oc)
        hdr.addWidget(self.oc_p)
        hdr.addWidget(QLabel("Eclipse:"))
        self.oc_type = QComboBox()
        self.oc_type.addItems(["Primary", "Secondary"])
        hdr.addWidget(self.oc_type)
        hdr.addStretch()
        layout.addLayout(hdr)

        splitter = QSplitter(Qt.Horizontal)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        self.oc_table = QTableWidget()
        self.oc_table.setColumnCount(5)
        self.oc_table.setHorizontalHeaderLabels(["n", "BJD_min", "O-C (d)", "err (d)", "type"])
        self.oc_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.oc_table.horizontalHeader().setStretchLastSection(True)
        self.oc_table.setMinimumWidth(330)
        ll.addWidget(self.oc_table, 1)
        btn_row = QHBoxLayout()
        for label, slot in [("Add", self._oc_add), ("Del", self._oc_del),
                             ("Import CSV", self._oc_import), ("Export CSV", self._oc_export)]:
            b = QPushButton(label)
            b.clicked.connect(slot)
            btn_row.addWidget(b)
        btn_row.addStretch()
        ll.addLayout(btn_row)
        splitter.addWidget(left)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        self.oc_canvas = FigureCanvas(Figure(figsize=(6, 4)))
        rl.addWidget(NavigationToolbar(self.oc_canvas, right))
        rl.addWidget(self.oc_canvas, 1)
        fit_row = QHBoxLayout()
        fit_row.addWidget(QLabel("Fit:"))
        self.oc_fit_combo = QComboBox()
        self.oc_fit_combo.addItems(["None", "Linear (ΔP)", "Parabola (dP/dt)", "Para + Sine (3rd body)"])
        fit_row.addWidget(self.oc_fit_combo)
        btn_fit = QPushButton("Fit & Plot")
        btn_fit.clicked.connect(self._oc_fit)
        fit_row.addWidget(btn_fit)
        fit_row.addStretch()
        rl.addLayout(fit_row)
        self.oc_fit_label = QLabel("")
        self.oc_fit_label.setWordWrap(True)
        self.oc_fit_label.setStyleSheet(
            "QLabel { background: #E8F5E9; padding: 6px; border-radius: 4px; "
            "font-family: monospace; font-size: 9pt; }"
        )
        rl.addWidget(self.oc_fit_label)
        splitter.addWidget(right)
        splitter.setSizes([350, 650])
        layout.addWidget(splitter, 1)

        return tab

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_lc_from_step(self):
        try:
            from ...utils.step_paths import list_lightcurve_csvs
            rd = Path(self.params.P.result_dir)
            paths = list_lightcurve_csvs(rd)
            if not paths:
                self.lc_status.setText("No lightcurve_*.csv found")
                return
            self._load_paths(paths)
        except Exception as e:
            self.lc_status.setText(f"Auto-load failed: {e}")

    def _open_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Light Curve CSV", "", "CSV Files (*.csv);;All Files (*)"
        )
        if path:
            self._load_paths([Path(path)])

    def _load_paths(self, paths: list[Path]):
        try:
            rd = Path(self.params.P.result_dir)
            try:
                from ...utils.qc_utils import load_frame_excludes as _lfe
                excl = set(_lfe(rd).keys())
            except Exception:
                excl = set()

            series_items: list[dict] = []
            for path in paths:
                df = pd.read_csv(path)
                if excl and "file" in df.columns:
                    df = df[~df["file"].astype(str).isin(excl)].reset_index(drop=True)
                time_col = next(
                    (c for c in ["BJD_TDB", "BJD", "bjd", "HJD", "hjd", "JD", "jd", "time"] if c in df.columns), None
                )
                if time_col is None:
                    continue
                t = pd.to_numeric(df[time_col], errors="coerce").to_numpy(float)
                time_mask = np.isfinite(t)
                if not np.any(time_mask):
                    continue
                t = t[time_mask]
                err_col = next(
                    (c for c in ["diff_err_corr", "diff_err", "mag_err", "err", "sigma"] if c in df.columns), None
                )
                e = pd.to_numeric(df[err_col], errors="coerce").to_numpy(float)[time_mask] if err_col else None
                filter_col = next(
                    (c for c in ["filter", "Filter", "FILTER", "band", "Band"] if c in df.columns), None
                )
                filters = df[filter_col].astype(str).to_numpy()[time_mask] if filter_col else None
                corr_tag = _detect_corr_mode_from_df(df, path.name)
                for label, col, arr in _collect_mag_options(df, time_mask, corr_tag=corr_tag):
                    key = f"{path.name}::{col}"
                    series_label = _describe_series(corr_tag, col)
                    series_items.append({
                        "key": key,
                        "combo_label": series_label,
                        "time": t,
                        "mag": arr,
                        "mag_col": col,
                        "mag_err": e,
                        "filters": filters,
                        "source": path.name,
                        "corr_tag": corr_tag,
                        "series_label": series_label,
                    })

            if not series_items:
                self.lc_status.setText("No usable light curve series found")
                return

            unique_series: dict[str, dict] = {}
            for item in series_items:
                label = item["combo_label"]
                prev = unique_series.get(label)
                if prev is None or _source_priority(item["source"]) < _source_priority(prev["source"]):
                    unique_series[label] = item
            series_items = list(unique_series.values())
            series_items.sort(key=lambda item: _series_rank(item["corr_tag"], item["mag_col"], item["source"]))
            self.series_options = {item["key"]: item for item in series_items}
            # Read step11 preference
            pref_label = ""
            try:
                from ...utils.step_paths import load_detrend_preference
                pref_mode = load_detrend_preference(rd)
                if pref_mode:
                    pref_label = _CORR_MODE_LABELS.get(pref_mode, "")
            except Exception:
                pass
            self.mag_col_combo.blockSignals(True)
            self.mag_col_combo.clear()
            for item in series_items:
                star = " *" if (pref_label and item["corr_tag"] == pref_label) else ""
                self.mag_col_combo.addItem(item["combo_label"] + star, item["key"])
            default_idx = 0
            if pref_label:
                default_idx = next(
                    (i for i, item in enumerate(series_items)
                     if item["corr_tag"] == pref_label and "corr" in item["mag_col"]), 0
                )
            if default_idx == 0:
                default_idx = next(
                    (i for i, item in enumerate(series_items)
                     if item["corr_tag"] == "Global ensemble" and "corr" in item["mag_col"]), 0
                )
            self.mag_col_combo.setCurrentIndex(default_idx)
            self.mag_col_combo.setEnabled(True)
            self.mag_col_combo.blockSignals(False)
            self._apply_series_option(self.mag_col_combo.currentData())
        except Exception as e:
            self.lc_status.setText(f"Error: {e}")
            self.log(f"[ERROR] {e}")

    def _apply_series_option(self, key: str | None):
        if not key or key not in self.series_options:
            return
        item = self.series_options[key]
        self.lc_data = {
            "time": item["time"],
            "mag": item["mag"],
            "mag_col": item["mag_col"],
            "mag_err": item.get("mag_err"),
            "filters": item.get("filters"),
            "source": item["source"],
            "corr_tag": item.get("corr_tag", ""),
            "series_label": item.get("series_label", item["mag_col"]),
        }
        n = int(np.sum(np.isfinite(self.lc_data["time"]) & np.isfinite(self.lc_data["mag"])))
        corr_line = f"  [{self.lc_data['corr_tag']}]" if self.lc_data.get("corr_tag") else ""
        workspace_name = Path(self.params.P.result_dir).name
        self.lc_status.setText(
            f"{workspace_name}\n{self.lc_data['source']}\n{n} pts{corr_line}\n{self.mag_col_combo.currentText()}"
        )
        self.lc_status.setStyleSheet("color: green;")
        filters = self.lc_data.get("filters")
        filt_info = f" [{'/' .join(sorted(set(filters)))}]" if filters is not None and len(filters) else ""
        self.log(
            f"Loaded: {self.lc_data['source']} ({n} pts, {self.lc_data.get('series_label', self.lc_data['mag_col'])}{filt_info}, "
            f"detrend={self.lc_data.get('corr_tag') or 'N/A'})"
        )
        t0 = float(np.nanmin(self.lc_data["time"]))
        self.t0_edit.setValue(t0)
        self.oc_t0.setValue(t0)

    def _on_mag_col_changed(self):
        key = self.mag_col_combo.currentData()
        if not key:
            return
        self._apply_series_option(key)
        self._update_phase_plot()

    # ------------------------------------------------------------------
    # Scan
    # ------------------------------------------------------------------

    def _run_scan(self):
        if self.lc_data is None:
            QMessageBox.warning(self, "No Data", "Load a light curve first.")
            return
        if self._scan_worker and self._scan_worker.isRunning():
            return
        methods = []
        if self.chk_ls.isChecked():
            methods.append("ls")
        if self.chk_pdm.isChecked():
            methods.append("pdm")
        if self.chk_bls.isChecked():
            methods.append("bls")
        if not methods:
            QMessageBox.warning(self, "No Method", "최소 하나의 방법을 선택하세요.")
            return
        self.scan_status.setText("Running…")
        self._scan_worker = PeriodAnalysisWorker(
            time=self.lc_data["time"],
            mag_raw=self.lc_data["mag"],
            mag_corr=None,
            mag_err=self.lc_data.get("mag_err"),
            min_period=self.min_p.value(),
            max_period=self.max_p.value(),
            samples_per_peak=self.spp.value(),
            methods=methods,
            pdm_n_bins=self.pdm_bins.value(),
        )
        self._scan_worker.progress.connect(self.scan_status.setText)
        self._scan_worker.finished.connect(self._on_scan_done)
        self._scan_worker.error.connect(lambda e: (self.scan_status.setText("Error"), self.log(e)))
        self._scan_worker.start()

    def _on_scan_done(self, result: dict):
        self.scan_result = result
        # Prefer BLS > LS > PDM for EBs
        best_p = None
        best_method = None
        for key in ("raw_bls", "raw_ls", "raw_pdm"):
            if key in result:
                best_p = result[key]["best_period"]
                best_method = key.replace("raw_", "").upper()
                break
        if best_p is None:
            self.scan_status.setText("No result")
            return
        parts = []
        for key in ("raw_ls", "raw_pdm", "raw_bls"):
            if key in result:
                m = key.replace("raw_", "").upper()
                parts.append(f"{m}: {result[key]['best_period']:.6f} d")
        self.scan_status.setText("  ".join(parts))
        self.phase_p.setValue(best_p)
        self.oc_p.setValue(best_p)
        self._draw_periodogram(result)
        self._update_phase_plot()
        self.log(f"Scan best ({best_method}): {best_p:.6f} d")

    def _draw_periodogram(self, result: dict):
        _METHOD_LABELS = {
            "raw_ls": ("Lomb-Scargle", "#1E88E5"),
            "raw_pdm": ("PDM", "#43A047"),
            "raw_bls": ("Box Least Squares", "#E65100"),
        }
        keys = [k for k in ("raw_ls", "raw_pdm", "raw_bls") if k in result]
        if not keys:
            return
        fig = self.pg_canvas.figure
        fig.clear()
        axes = fig.subplots(1, len(keys)) if len(keys) > 1 else [fig.add_subplot(111)]
        for ax, key in zip(axes, keys):
            label, color = _METHOD_LABELS[key]
            r = result[key]
            if r.get("method") == "ls":
                periods = 1.0 / r["frequency"]
            else:
                periods = r["trial_periods"]
            power = r["power"]
            best_p = r["best_period"]
            ax.plot(periods, power, color=color, lw=0.8, alpha=0.9)
            ax.axvline(best_p, color="red", ls="--", lw=1.5, label=f"P={best_p:.4f} d")
            ax.set_xscale("log")
            ax.set_xlabel("Period (days)")
            ax.set_ylabel("Power")
            ax.set_title(label)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        self.pg_canvas.draw_idle()

    # ------------------------------------------------------------------
    # Phase plot
    # ------------------------------------------------------------------

    def _detect_primary_min(self):
        if self.lc_data is None:
            return
        t = self.lc_data["time"]
        mag = self.lc_data["mag"]
        period = self.phase_p.value()
        mask = np.isfinite(t) & np.isfinite(mag)
        t_v, m_v = t[mask], mag[mask]
        if len(t_v) < 5:
            return
        t0_ref = np.nanmin(t_v)
        phase = ((t_v - t0_ref) / period) % 1.0

        n_bins = 40
        edges = np.linspace(0, 1, n_bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2
        bin_mags = np.full(n_bins, np.nan)
        for b in range(n_bins):
            msk = (phase >= edges[b]) & (phase < edges[b + 1])
            if msk.sum() >= 2:
                bin_mags[b] = np.nanmedian(m_v[msk])

        valid = np.isfinite(bin_mags)
        if valid.sum() < 3:
            return
        min_idx = int(np.nanargmax(bin_mags))  # faintest
        phi_min = float(centers[min_idx])
        t0_epoch = t0_ref + phi_min * period
        self.t0_edit.setValue(t0_epoch)
        self.oc_t0.setValue(t0_epoch)
        self.log(f"Primary min at φ={phi_min:.4f} → T₀={t0_epoch:.6f}")

    def _update_phase_plot(self):
        if self.lc_data is None:
            return
        period = self.phase_p.value()
        t0 = self.t0_edit.value()
        p_label = self.phase_p.value()
        if self.chk_half.isChecked():
            period = period / 2.0

        t = self.lc_data["time"]
        mag = self.lc_data["mag"]
        mag_err = self.lc_data.get("mag_err")
        filters = self.lc_data.get("filters")
        mask = np.isfinite(t) & np.isfinite(mag)
        t_v, m_v = t[mask], mag[mask]
        dy_v = mag_err[mask] if mag_err is not None else None
        # filters already pre-masked (stored post-mask in _load_csv)
        filt_v = filters if filters is not None else None

        phase = ((t_v - t0) / period) % 1.0

        fig = self.ph_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)

        unique_filters = sorted(set(filt_v)) if filt_v is not None else [""]
        for fi, filt in enumerate(unique_filters):
            if not self.filter_visibility.get(filt, True):
                continue
            sel = (filt_v == filt) if filt_v is not None else np.ones(len(phase), bool)
            ph_sel = np.concatenate([phase[sel], phase[sel] + 1.0])
            m_sel = np.concatenate([m_v[sel], m_v[sel]])
            color = self.filter_colors.get(filt) or _filt_color(filt, fi)
            label = filt if filt else "data"
            if dy_v is not None:
                dy_sel = np.concatenate([dy_v[sel], dy_v[sel]])
                ax.errorbar(ph_sel, m_sel, yerr=dy_sel,
                            fmt="o", color=color, markersize=3,
                            elinewidth=0.5, capsize=0, alpha=0.7, label=label)
            else:
                ax.scatter(ph_sel, m_sel, color=color, s=12, alpha=0.7, label=label)

        ax.invert_yaxis()
        ax.set_xlabel("Phase")
        ax.set_ylabel("Magnitude")
        title = f"Phase Plot  P={p_label:.6f} d"
        if self.chk_half.isChecked():
            title += "  (P/2 view)"
        ax.set_title(title)
        ax.set_xlim(0, 2)
        ax.axvline(0, color="gray", ls=":", alpha=0.4)
        ax.axvline(1, color="gray", ls=":", alpha=0.4)
        # Mark expected secondary at 0.5 phase
        ax.axvline(0.5, color="navy", ls="--", lw=0.8, alpha=0.5, label="Secondary (φ=0.5)")
        ax.axvline(1.5, color="navy", ls="--", lw=0.8, alpha=0.5)

        # Check star phase-folded overlay
        try:
            _rd = Path(self.params.P.result_dir)
            _check_filter = _resolve_check_filter(self.lc_data.get("filters"))
            _ck_id, _ck_df = _load_check_star_for_plot(_rd, filt=_check_filter)
            if _ck_df is not None and not _ck_df.empty:
                _t_col = next((c for c in ["JD", "bjd", "hjd", "time"] if c in _ck_df.columns), None)
                _y_col = next((c for c in ["diff_mag_raw", "diff_mag", "mag"] if c in _ck_df.columns), None)
                if _t_col and _y_col:
                    _ct = pd.to_numeric(_ck_df[_t_col], errors="coerce").to_numpy(float)
                    _cm = pd.to_numeric(_ck_df[_y_col], errors="coerce").to_numpy(float)
                    _mask = np.isfinite(_ct) & np.isfinite(_cm)
                    if _mask.any():
                        _ck_label = f"Check ID {_ck_id}" if _ck_id is not None else "Check"
                        _phase = ((_ct[_mask] - t0) / period) % 1.0
                        _phase_ext = np.concatenate([_phase, _phase + 1.0])
                        _mag_ext = np.concatenate([_cm[_mask], _cm[_mask]])
                        ax.scatter(_phase_ext, _mag_ext, s=8, color="#FFD700", alpha=0.4,
                                   zorder=2, label=_ck_label, marker="^")
        except Exception:
            pass

        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        self.ph_canvas.draw_idle()
        self.tabs.setCurrentIndex(1)

    # ------------------------------------------------------------------
    # Eclipse measurement
    # ------------------------------------------------------------------

    def _measure_eclipse(self):
        """Fit parabola around primary eclipse minimum in phase space."""
        if self.lc_data is None:
            QMessageBox.warning(self, "No Data", "Load a light curve first.")
            return
        t = self.lc_data["time"]
        mag = self.lc_data["mag"]
        mag_err = self.lc_data.get("mag_err")
        period = self.phase_p.value()
        t0 = self.t0_edit.value()
        mask = np.isfinite(t) & np.isfinite(mag)
        t_v, m_v = t[mask], mag[mask]
        dy_v = mag_err[mask] if mag_err is not None else None

        phase = ((t_v - t0) / period) % 1.0
        # Focus on primary eclipse region ±0.15 around phase 0
        in_eclipse = ((phase < 0.15) | (phase > 0.85))
        if in_eclipse.sum() < 5:
            self.meas_label.setText("포착된 식현상 데이터가 부족합니다 (±0.15 범위).")
            return

        ph_ec = phase[in_eclipse]
        # Unwrap: 0.85–1.0 → -0.15–0.0
        ph_ec = np.where(ph_ec > 0.5, ph_ec - 1.0, ph_ec)
        m_ec = m_v[in_eclipse]
        dy_ec = dy_v[in_eclipse] if dy_v is not None else None

        try:
            sigma = dy_ec if dy_ec is not None else None
            popt, _ = curve_fit(
                lambda x, a, b, c: a + b * x + c * x**2,
                ph_ec, m_ec, p0=[m_ec.max(), 0, -1],
                sigma=sigma,
            )
            a, b, c = popt
            if c > 0:  # Should be concave up (faintest at vertex)
                phase_min = -b / (2 * c)
                depth = a + b * phase_min + c * phase_min**2 - np.nanpercentile(m_ec, 10)
                # Approximate half-width at half-depth
                # phase where parabola = a + depth/2: 2c·Δφ² = depth/2
                hw_phase = np.sqrt(abs(depth) / (2 * abs(c))) if c != 0 else np.nan
                duration_days = 2 * hw_phase * period
                duration_hr = duration_days * 24
                out_mag = np.nanpercentile(m_ec, 10)  # approximate out-of-eclipse
                eclipse_depth = float(m_ec.max() - out_mag)
                text = (
                    f"Primary eclipse fit:\n"
                    f"  φ_min = {phase_min:.4f}\n"
                    f"  Depth = {eclipse_depth:.4f} mag\n"
                    f"  Duration ≈ {duration_hr:.2f} hr\n"
                    f"  T_min ≈ {t0 + phase_min * period:.6f} BJD"
                )
                # Push to O-C table
                t_min_bjd = t0 + phase_min * period
                p = self.oc_p.value()
                n = int(round((t_min_bjd - self.oc_t0.value()) / p)) if p > 0 else 0
                oc = t_min_bjd - (self.oc_t0.value() + n * p)
                r = self.oc_table.rowCount()
                self.oc_table.insertRow(r)
                for ci, val in enumerate([str(n), f"{t_min_bjd:.6f}", f"{oc:.6f}", "", "Primary"]):
                    self.oc_table.setItem(r, ci, QTableWidgetItem(val))
                self._draw_oc()
            else:
                text = "포물선 피팅 실패: 오목 방향 오류"
        except Exception as e:
            text = f"피팅 실패: {e}"

        self.meas_label.setText(text)
        self.log(f"Eclipse measure: {text.splitlines()[0]}")

    # ------------------------------------------------------------------
    # O-C
    # ------------------------------------------------------------------

    def _oc_add(self):
        r = self.oc_table.rowCount()
        self.oc_table.insertRow(r)
        for c in range(5):
            self.oc_table.setItem(r, c, QTableWidgetItem(""))

    def _oc_del(self):
        rows = sorted({i.row() for i in self.oc_table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.oc_table.removeRow(r)
        self._draw_oc()

    def _oc_import(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import O-C CSV", "", "CSV (*.csv)")
        if not path:
            return
        try:
            df = pd.read_csv(path)
            bjd_col = next(
                (c for c in df.columns if c.lower() in ("bjd_min", "bjd", "hjd", "jd")), None
            )
            if bjd_col is None:
                QMessageBox.warning(self, "Error", "BJD 컬럼을 찾을 수 없습니다.")
                return
            err_col = next(
                (c for c in df.columns if c.lower() in ("err", "error", "sigma")), None
            )
            type_col = next(
                (c for c in df.columns if c.lower() in ("type", "eclipse_type")), None
            )
            t0 = self.oc_t0.value(); p = self.oc_p.value()
            self.oc_table.setRowCount(0)
            for _, row_data in df.iterrows():
                bjd = float(row_data[bjd_col])
                n = int(round((bjd - t0) / p)) if p > 0 else 0
                oc = bjd - (t0 + n * p)
                err = f"{float(row_data[err_col]):.6f}" if err_col else ""
                etype = str(row_data[type_col]) if type_col else "Primary"
                r = self.oc_table.rowCount()
                self.oc_table.insertRow(r)
                for ci, val in enumerate([str(n), f"{bjd:.6f}", f"{oc:.6f}", err, etype]):
                    self.oc_table.setItem(r, ci, QTableWidgetItem(val))
            self._draw_oc()
        except Exception as e:
            QMessageBox.warning(self, "Import Error", str(e))

    def _oc_export(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export O-C CSV", "", "CSV (*.csv)")
        if not path:
            return
        rows = []
        for r in range(self.oc_table.rowCount()):
            rows.append([
                self.oc_table.item(r, c).text() if self.oc_table.item(r, c) else ""
                for c in range(5)
            ])
        pd.DataFrame(rows, columns=["n", "BJD_min", "O-C (d)", "err (d)", "type"]).to_csv(path, index=False)
        self.log(f"Exported O-C to {path}")

    def _recompute_oc(self):
        t0 = self.oc_t0.value(); p = self.oc_p.value()
        if p <= 0:
            return
        for r in range(self.oc_table.rowCount()):
            bjd_it = self.oc_table.item(r, 1)
            if not bjd_it or not bjd_it.text():
                continue
            try:
                bjd = float(bjd_it.text())
                n = int(round((bjd - t0) / p))
                oc = bjd - (t0 + n * p)
                self.oc_table.setItem(r, 0, QTableWidgetItem(str(n)))
                self.oc_table.setItem(r, 2, QTableWidgetItem(f"{oc:.6f}"))
            except ValueError:
                pass
        self._draw_oc()

    def _get_oc_arrays(self):
        ns, ocs, errs, types = [], [], [], []
        for r in range(self.oc_table.rowCount()):
            try:
                n_it = self.oc_table.item(r, 0)
                oc_it = self.oc_table.item(r, 2)
                er_it = self.oc_table.item(r, 3)
                ty_it = self.oc_table.item(r, 4)
                if not n_it or not oc_it or not n_it.text() or not oc_it.text():
                    continue
                ns.append(int(n_it.text()))
                ocs.append(float(oc_it.text()))
                errs.append(float(er_it.text()) if (er_it and er_it.text()) else np.nan)
                types.append(ty_it.text() if ty_it else "Primary")
            except (ValueError, TypeError):
                pass
        return np.array(ns), np.array(ocs), np.array(errs), types

    def _draw_oc(self, fit_result=None):
        fig = self.oc_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        ns, ocs, errs, types = self._get_oc_arrays()
        if len(ns) == 0:
            ax.text(0.5, 0.5, "O-C 데이터 없음", ha="center", va="center",
                    transform=ax.transAxes, color="gray")
            fig.tight_layout()
            self.oc_canvas.draw_idle()
            return
        oc_min = ocs * 1440
        has_err = np.any(np.isfinite(errs))
        # Separate primary/secondary
        for etype, color, marker in [("Primary", "#1565C0", "o"), ("Secondary", "#C62828", "s")]:
            mask = np.array([t.strip().lower() == etype.lower() for t in types])
            if mask.sum() == 0:
                continue
            if has_err:
                err_pl = np.where(np.isfinite(errs[mask]), errs[mask] * 1440, 0.0)
                ax.errorbar(ns[mask], oc_min[mask], yerr=err_pl, fmt=marker, color=color,
                            markersize=5, elinewidth=1, capsize=3, label=etype)
            else:
                ax.scatter(ns[mask], oc_min[mask], color=color, marker=marker,
                           s=40, zorder=5, label=etype)
        ax.axhline(0, color="gray", ls="--", lw=1, alpha=0.6)
        if fit_result is not None:
            n_fit = np.linspace(ns.min(), ns.max(), 500)
            ax.plot(n_fit, fit_result["func"](n_fit) * 1440, color="red",
                    lw=1.5, label=fit_result["label"])
        ax.set_xlabel("Epoch (n)")
        ax.set_ylabel("O-C (minutes)")
        ax.set_title("Eclipse Timing O-C")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        self.oc_canvas.draw_idle()

    def _oc_fit(self):
        ns, ocs, errs, _ = self._get_oc_arrays()
        if len(ns) < 3:
            QMessageBox.warning(self, "O-C Fit", "데이터 포인트가 3개 이상 필요합니다.")
            return
        mode = self.oc_fit_combo.currentText()
        p = self.oc_p.value()
        sigma = None
        if np.any(np.isfinite(errs)):
            fill = np.nanmedian(errs[np.isfinite(errs)])
            sigma = np.where(np.isfinite(errs), errs, fill)
            sigma = np.clip(sigma, 1e-9, None)

        fit_result = None
        text = ""
        try:
            if mode == "Linear (ΔP)":
                def model(n, a, b): return a + b * n
                popt, pcov = curve_fit(model, ns, ocs, sigma=sigma, absolute_sigma=True)
                perr = np.sqrt(np.diag(pcov))
                dp = popt[1]
                text = (
                    f"ΔP = {dp:.3e} ± {perr[1]:.1e} d/cycle\n"
                    f"Corrected P = {p + dp:.8f} d"
                )
                fit_result = {"func": lambda n: model(n, *popt), "label": f"Linear (ΔP={dp:.2e})"}

            elif mode == "Parabola (dP/dt)":
                def model(n, a, b, c): return a + b * n + c * n**2
                popt, pcov = curve_fit(model, ns, ocs, sigma=sigma, absolute_sigma=True, maxfev=10000)
                perr = np.sqrt(np.diag(pcov))
                c = popt[2]; dP_dt = 2 * c / p
                dP_yr = dP_dt * 365.25
                arrow = "▲ 주기 증가" if dP_dt > 0 else "▼ 주기 감소"
                text = (
                    f"dP/dt = {dP_dt:.3e} d/d  ({dP_yr:.3e} d/yr)\n{arrow}"
                )
                fit_result = {"func": lambda n: model(n, *popt), "label": f"Parabola"}

            elif mode == "Para + Sine (3rd body)":
                if len(ns) < 6:
                    QMessageBox.warning(self, "O-C Fit", "6개 이상 필요합니다.")
                    return
                n_span = ns.max() - ns.min()
                A_g = (ocs.max() - ocs.min()) / 2
                p3_g = n_span / 2.0

                def model(n, a, b, c, A, p3_n, phi):
                    return a + b*n + c*n**2 + A * np.sin(2*np.pi*n/p3_n + phi)

                bounds = (
                    [-np.inf, -np.inf, -np.inf, 0, 10, -np.pi],
                    [np.inf, np.inf, np.inf, np.inf, n_span * 10, np.pi],
                )
                popt, _ = curve_fit(
                    model, ns, ocs, p0=[0, 0, 0, A_g, p3_g, 0],
                    sigma=sigma, absolute_sigma=True, bounds=bounds, maxfev=30000,
                )
                c, A, p3_n = popt[2], abs(popt[3]), abs(popt[4])
                dP_dt = 2 * c / p
                p3_days = p3_n * p; p3_yr = p3_days / 365.25
                a12_sin_i = abs(A) * 173.145
                text = (
                    f"dP/dt = {dP_dt:.3e} d/d\n"
                    f"제3천체 P₃ = {p3_days:.1f} d  ({p3_yr:.2f} yr)\n"
                    f"a₁₂·sin(i) = {a12_sin_i:.4f} AU"
                )
                fit_result = {"func": lambda n: model(n, *popt), "label": f"Para+Sine (P₃={p3_yr:.2f} yr)"}

        except Exception as e:
            text = f"피팅 실패: {e}"

        self.oc_fit_label.setText(text)
        self._draw_oc(fit_result)
        self.log(f"[O-C Fit] {mode}: {text.splitlines()[0]}")

    # ------------------------------------------------------------------
    # Filter color/visibility browser
    # ------------------------------------------------------------------

    def _apply_filter_swatch_style(self, button, color: str):
        button.setStyleSheet(
            f"QPushButton {{ background-color: {color}; border: 1px solid #455A64; "
            f"border-radius: 3px; min-width: 28px; min-height: 20px; }}"
            "QPushButton:hover { border: 2px solid #263238; }"
        )

    def show_filter_color_browser(self):
        if self.lc_data is None:
            QMessageBox.information(self, "Filter Browser", "먼저 라이트커브를 로드하세요.")
            return
        filters = self.lc_data.get("filters")
        if filters is None:
            QMessageBox.information(self, "Filter Browser", "이 CSV에는 filter 컬럼이 없습니다.")
            return
        keys = sorted(set(filters))

        # Initialize missing entries
        for i, k in enumerate(keys):
            if k not in self.filter_colors:
                self.filter_colors[k] = _filt_color(k, i)
            if k not in self.filter_visibility:
                self.filter_visibility[k] = True

        dialog = QDialog(self)
        dialog.setWindowTitle("필터 색상 / 표시 설정")
        dialog.setMinimumWidth(360)
        layout = QVBoxLayout(dialog)

        info = QLabel("필터별 색상을 변경하거나 표시/숨김을 설정합니다.\n변경 즉시 위상 그래프에 반영됩니다.")
        info.setStyleSheet("color: #455A64; font-size: 8pt;")
        info.setWordWrap(True)
        layout.addWidget(info)

        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        swatch_btns: dict = {}
        vis_chks: dict = {}

        for row, key in enumerate(keys):
            # Visibility checkbox
            chk = QCheckBox(key)
            chk.setChecked(self.filter_visibility.get(key, True))
            chk.setStyleSheet("font-weight: bold;")
            def _on_vis(state, k=key):
                self.filter_visibility[k] = bool(state)
                self._update_phase_plot()
            chk.stateChanged.connect(_on_vis)
            grid.addWidget(chk, row, 0)
            vis_chks[key] = chk

            # Color swatch
            swatch = QPushButton("")
            swatch.setFixedSize(28, 20)
            swatch.setFocusPolicy(Qt.NoFocus)
            self._apply_filter_swatch_style(swatch, self.filter_colors[key])
            grid.addWidget(swatch, row, 1)
            swatch_btns[key] = swatch

            # Browse button
            browse = QPushButton("Browse…")
            def _on_browse(_checked=False, k=key, sw=None, dlg=dialog):
                current = QColor(self.filter_colors.get(k, "#888888"))
                picked = QColorDialog.getColor(current, dlg, f"{k} 색상 선택")
                if picked.isValid():
                    self.filter_colors[k] = picked.name()
                    self._apply_filter_swatch_style(sw, picked.name())
                    self._update_phase_plot()
            # bind swatch via default arg
            browse.clicked.connect(lambda _c=False, k=key, sw=swatch: _on_browse(_c, k, sw, dialog))
            grid.addWidget(browse, row, 2)

        layout.addLayout(grid)

        btn_row = QHBoxLayout()
        btn_reset = QPushButton("색상 초기화")
        def _reset():
            for i, k in enumerate(keys):
                self.filter_colors[k] = _filt_color(k, i)
                self._apply_filter_swatch_style(swatch_btns[k], self.filter_colors[k])
            self._update_phase_plot()
        btn_reset.clicked.connect(_reset)
        btn_row.addWidget(btn_reset)

        btn_all = QPushButton("모두 표시")
        def _show_all():
            for k in keys:
                self.filter_visibility[k] = True
                vis_chks[k].setChecked(True)
            self._update_phase_plot()
        btn_all.clicked.connect(_show_all)
        btn_row.addWidget(btn_all)

        btn_row.addStretch()
        btn_close = QPushButton("닫기")
        btn_close.clicked.connect(dialog.accept)
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

        dialog.exec_()

    # ------------------------------------------------------------------

    def _toggle_log(self, checked: bool):
        self.log_box.setVisible(checked)
        self.btn_log_toggle.setText("Log ▲" if checked else "Log ▼")

    def log(self, msg: str):
        self.log_box.append(msg)
