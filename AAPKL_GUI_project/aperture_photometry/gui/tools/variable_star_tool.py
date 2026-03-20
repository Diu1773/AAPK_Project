"""
Variable Star Analysis Tool

Detailed period analysis + refinement + O-C diagram for variable stars
(Mira, RR Lyr, Cepheid, Delta Sct, etc.).

Pipeline:
  1. Load light curve (from Step 10/11 output)
  2. Quick LS/PDM scan  →  coarse best period
  3. Fine-grid refinement + bootstrap σ_P
  4. Phase-fold viewer with harmonic selector
  5. O-C diagram with linear / parabola / parabola+sine fits
  6. Basic Fourier decomposition (R21, φ21)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

_CORR_MODE_RE = re.compile(r"lightcurve_.*?_(global|color|offset|raw)\b", re.IGNORECASE)
_TARGET_ID_RE = re.compile(r"lightcurve_(?:combined_)?ID(\d+)_", re.IGNORECASE)
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


def _detect_target_id_from_df(df: pd.DataFrame, filename: str) -> int | None:
    m = _TARGET_ID_RE.search(filename)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    for col in ("target_id", "star_id", "ID"):
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").dropna().astype(int)
        uniq = sorted(set(vals.tolist()))
        if len(uniq) == 1:
            return int(uniq[0])
    return None


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

from astropy.timeseries import LombScargle

from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QPushButton, QLabel, QDoubleSpinBox, QSpinBox,
    QCheckBox, QTabWidget, QTextEdit, QTableWidget, QTableWidgetItem,
    QHeaderView, QFileDialog, QSplitter, QMessageBox, QComboBox, QLineEdit,
    QColorDialog, QDialog, QGridLayout,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor

_FILTER_COLORS = [
    "#1E88E5", "#E53935", "#43A047", "#FB8C00",
    "#8E24AA", "#00ACC1", "#F06292", "#795548",
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


def _resolve_check_filter(filters, selected_filter: str | None = None) -> str | None:
    if selected_filter and selected_filter != "__all__":
        return selected_filter
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


def _pick_check_overlay_cols(df: pd.DataFrame, preferred_mag_col: str | None = None) -> tuple[str | None, str | None]:
    time_col = next((c for c in ["BJD_TDB", "BJD", "bjd", "HJD", "hjd", "JD", "jd", "time"] if c in df.columns), None)
    mag_candidates: list[str] = []
    preferred = str(preferred_mag_col or "").strip()
    if preferred:
        if preferred == "diff_mag_corr":
            mag_candidates.extend(["diff_mag_corr", "diff_mag_raw", "diff_mag", "mag"])
        elif preferred == "diff_mag_raw":
            mag_candidates.extend(["diff_mag_raw", "diff_mag_corr", "diff_mag", "mag"])
        else:
            mag_candidates.append(preferred)
    mag_candidates.extend(["diff_mag_corr", "diff_mag_raw", "diff_mag", "mag_ensemble_corr", "mag"])
    seen: set[str] = set()
    mag_col = None
    for col in mag_candidates:
        if col in seen:
            continue
        seen.add(col)
        if col in df.columns:
            mag_col = col
            break
    return time_col, mag_col


# ---------------------------------------------------------------------------
# Worker: fine-grid refinement + bootstrap (LS-based)
# ---------------------------------------------------------------------------

class RefineBootstrapWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, time, mag, mag_err, center_period,
                 n_bootstrap: int = 300, zoom_factor: int = 100,
                 method: str = "ls", pdm_n_bins: int = 10):
        super().__init__()
        self.time = np.asarray(time, dtype=float)
        self.mag = np.asarray(mag, dtype=float)
        self.mag_err = np.asarray(mag_err, dtype=float) if mag_err is not None else None
        self.center_period = float(center_period)
        self.n_bootstrap = int(n_bootstrap)
        self.zoom_factor = int(zoom_factor)
        self.method = method  # "ls" or "pdm"
        self.pdm_n_bins = pdm_n_bins
        self._stop = False

    def stop(self):
        self._stop = True

    # -- PDM helpers ----------------------------------------------------------

    def _pdm_theta_array(self, t, y, trial_periods):
        """Compute PDM theta for an array of trial periods (vectorized).

        Uses bincount to avoid Python bin loops — ~10-50x faster than naive.
        """
        var_total = np.var(y)
        if var_total == 0:
            return np.ones(len(trial_periods))

        n_bins = self.pdm_n_bins
        t_min = t.min()
        dt = t - t_min
        y2 = y * y
        n_periods = len(trial_periods)
        theta = np.ones(n_periods)

        for i in range(n_periods):
            phase = (dt / trial_periods[i]) % 1.0
            bi = np.clip((phase * n_bins).astype(np.int32), 0, n_bins - 1)
            counts = np.bincount(bi, minlength=n_bins)
            sums = np.bincount(bi, weights=y, minlength=n_bins)
            sum_sq = np.bincount(bi, weights=y2, minlength=n_bins)
            # within-bin SS = sum(x²) - sum(x)²/n, dof = n-1
            good = counts >= 2
            if not good.any():
                continue
            c_g = counts[good]
            ss = sum_sq[good] - sums[good] ** 2 / c_g  # sum of squares
            dof = c_g - 1
            theta[i] = ss.sum() / dof.sum() / var_total
        return theta

    # -- Main run -------------------------------------------------------------

    def run(self):
        try:
            t, y, dy = self._filter_valid()
            if len(t) < 10:
                self.error.emit("Not enough valid data points (< 10)")
                return

            baseline = t.max() - t.min()
            f_center = 1.0 / self.center_period
            df_coarse = 1.0 / (10.0 * baseline)
            df_fine = df_coarse / self.zoom_factor
            half_range = 10.0 * df_coarse
            f_fine = np.arange(f_center - half_range, f_center + half_range + df_fine, df_fine)
            f_fine = f_fine[f_fine > 0]
            p_fine = 1.0 / f_fine[::-1]  # period grid (ascending)

            self.progress.emit(f"Fine grid search ({self.method.upper()})…")

            if self.method == "pdm":
                theta_fine = self._pdm_theta_array(t, y, p_fine)
                power_fine = 1.0 - theta_fine  # higher = better
                refined_period = self._parabola_peak_period(p_fine, power_fine)
            else:
                ls = (LombScargle(t, y, dy) if (dy is not None and np.any(dy > 0))
                      else LombScargle(t, y))
                power_freq = ls.power(f_fine)
                power_fine = power_freq[::-1]  # match p_fine order
                refined_period = self._parabola_peak(f_fine, power_freq)

            boot_periods = []
            n_data = len(t)
            for i in range(self.n_bootstrap):
                if self._stop:
                    break
                if i % 20 == 0:
                    self.progress.emit(f"Bootstrap {i}/{self.n_bootstrap}…")
                idx = np.random.choice(n_data, n_data, replace=True)
                tb, yb = t[idx], y[idx]

                if self.method == "pdm":
                    theta_b = self._pdm_theta_array(tb, yb, p_fine)
                    pwr_b = 1.0 - theta_b
                    boot_periods.append(self._parabola_peak_period(p_fine, pwr_b))
                else:
                    dyb = dy[idx] if dy is not None else None
                    ls_b = (LombScargle(tb, yb, dyb) if (dyb is not None and np.any(dyb > 0))
                            else LombScargle(tb, yb))
                    pwr_b = ls_b.power(f_fine)
                    boot_periods.append(self._parabola_peak(f_fine, pwr_b))

            boot_periods = np.array(boot_periods)
            med = np.median(boot_periods)
            mad = np.median(np.abs(boot_periods - med))
            keep = np.abs(boot_periods - med) < 5 * 1.4826 * mad
            sigma_p = (float(np.std(boot_periods[keep])) if keep.sum() >= 5
                       else float(np.std(boot_periods)))

            self.finished.emit({
                "refined_period": float(refined_period),
                "sigma_p": sigma_p,
                "boot_periods": boot_periods,
                "fine_periods": p_fine,
                "fine_power": power_fine,
                "method": self.method,
            })
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")

    def _parabola_peak(self, freq, power):
        """Parabola interpolation in frequency space → return period."""
        idx = int(np.argmax(power))
        if 0 < idx < len(power) - 1:
            f3 = freq[idx - 1: idx + 2]
            p3 = power[idx - 1: idx + 2]
            try:
                coeffs = np.polyfit(f3, p3, 2)
                if coeffs[0] < 0:
                    fv = -coeffs[1] / (2.0 * coeffs[0])
                    if f3[0] < fv < f3[2]:
                        return 1.0 / fv
            except Exception:
                pass
        return 1.0 / freq[idx]

    def _parabola_peak_period(self, periods, power):
        """Parabola interpolation in period space → return period."""
        idx = int(np.argmax(power))
        if 0 < idx < len(power) - 1:
            p3 = periods[idx - 1: idx + 2]
            pw3 = power[idx - 1: idx + 2]
            try:
                coeffs = np.polyfit(p3, pw3, 2)
                if coeffs[0] < 0:
                    pv = -coeffs[1] / (2.0 * coeffs[0])
                    if p3[0] < pv < p3[2]:
                        return float(pv)
            except Exception:
                pass
        return float(periods[idx])

    def _filter_valid(self):
        mask = np.isfinite(self.time) & np.isfinite(self.mag)
        if self.mag_err is not None:
            mask &= np.isfinite(self.mag_err) & (self.mag_err > 0)
        dy = self.mag_err[mask] if self.mag_err is not None else None
        return self.time[mask], self.mag[mask], dy


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class VariableStarToolWindow(QWidget):
    """Variable Star Analysis Tool — period refinement, O-C, Fourier."""

    def __init__(self, params, project_state, parent=None):
        super().__init__(parent)
        self.params = params
        self.project_state = project_state
        self.lc_data: Optional[dict] = None
        self.series_options: dict[str, dict] = {}
        self.scan_result: Optional[dict] = None
        self.refined_period: Optional[float] = None
        self.sigma_period: Optional[float] = None
        self._scan_worker: Optional[PeriodAnalysisWorker] = None
        self._refine_worker: Optional[RefineBootstrapWorker] = None
        self.filter_colors: dict = {}      # user-customized per-filter colors
        self.filter_visibility: dict = {}  # True=visible, False=hidden
        self.workspace_dir = Path(self.params.P.result_dir)

        self.setWindowTitle("Variable Star Analysis")
        self.resize(1200, 800)
        self._build_ui()
        self._load_lc_from_workspace()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QVBoxLayout(self)

        header = QLabel(
            "<b>Variable Star Analysis</b> — Lomb-Scargle scan → fine-grid refinement → "
            "bootstrap σ_P → phase plot → O-C diagram → Fourier decomposition"
        )
        header.setStyleSheet("QLabel { background: #E8EAF6; padding: 8px; border-radius: 4px; }")
        header.setWordWrap(True)
        root.addWidget(header)

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter, 1)

        # ---- Left panel (controls) ----
        left = QWidget()
        left.setMaximumWidth(310)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(4, 4, 4, 4)
        splitter.addWidget(left)

        # Light curve
        lc_group = QGroupBox("Light Curve")
        lc_form = QFormLayout(lc_group)
        self.lc_status = QLabel("Not loaded")
        self.lc_status.setWordWrap(True)
        lc_form.addRow("Status:", self.lc_status)
        ws_row = QWidget()
        ws_layout = QHBoxLayout(ws_row)
        ws_layout.setContentsMargins(0, 0, 0, 0)
        self.workspace_edit = QLineEdit(str(self.workspace_dir))
        btn_workspace = QPushButton("Browse…")
        btn_workspace.clicked.connect(self._browse_workspace)
        btn_reload = QPushButton("Load")
        btn_reload.clicked.connect(self._load_lc_from_workspace)
        ws_layout.addWidget(self.workspace_edit, 1)
        ws_layout.addWidget(btn_workspace)
        ws_layout.addWidget(btn_reload)
        lc_form.addRow("Workspace:", ws_row)
        self.mag_col_combo = QComboBox()
        self.mag_col_combo.setEnabled(False)
        self.mag_col_combo.currentIndexChanged.connect(self._on_mag_col_changed)
        lc_form.addRow("Use data:", self.mag_col_combo)
        self.analysis_filter_combo = QComboBox()
        self.analysis_filter_combo.setEnabled(False)
        self.analysis_filter_combo.currentIndexChanged.connect(self._on_analysis_filter_changed)
        lc_form.addRow("Filter:", self.analysis_filter_combo)
        left_layout.addWidget(lc_group)

        # Filter display controls
        filt_group = QGroupBox("Filter Display")
        filt_form = QFormLayout(filt_group)
        btn_filt_browser = QPushButton("Browse Colors / Visibility…")
        btn_filt_browser.clicked.connect(self.show_filter_color_browser)
        filt_form.addRow(btn_filt_browser)
        left_layout.addWidget(filt_group)

        # Period scan
        scan_group = QGroupBox("Period Scan")
        scan_form = QFormLayout(scan_group)
        self.min_p = QDoubleSpinBox()
        self.min_p.setRange(0.001, 500); self.min_p.setDecimals(4)
        self.min_p.setValue(0.05); self.min_p.setSuffix(" d")
        scan_form.addRow("P min:", self.min_p)
        self.max_p = QDoubleSpinBox()
        self.max_p.setRange(0.01, 2000); self.max_p.setDecimals(4)
        self.max_p.setValue(100.0); self.max_p.setSuffix(" d")
        scan_form.addRow("P max:", self.max_p)
        self.spp = QSpinBox()
        self.spp.setRange(5, 50); self.spp.setValue(10)
        scan_form.addRow("Samples/peak:", self.spp)
        method_row = QHBoxLayout()
        self.chk_ls = QCheckBox("LS"); self.chk_ls.setChecked(True)
        self.chk_pdm = QCheckBox("PDM"); self.chk_pdm.setChecked(True)
        self.chk_bls = QCheckBox("BLS"); self.chk_bls.setChecked(False)
        method_row.addWidget(self.chk_ls)
        method_row.addWidget(self.chk_pdm)
        method_row.addWidget(self.chk_bls)
        scan_form.addRow("Methods:", method_row)
        self.pdm_bins = QSpinBox()
        self.pdm_bins.setRange(5, 50); self.pdm_bins.setValue(10)
        scan_form.addRow("PDM bins:", self.pdm_bins)
        btn_scan = QPushButton("Scan")
        btn_scan.setStyleSheet(
            "QPushButton { background: #1976D2; color: white; font-weight: bold; padding: 6px; }"
        )
        btn_scan.clicked.connect(self._run_scan)
        scan_form.addRow(btn_scan)
        self.scan_status = QLabel("")
        self.scan_status.setStyleSheet("font-size: 8pt; color: #555;")
        scan_form.addRow(self.scan_status)
        left_layout.addWidget(scan_group)

        # Refine
        refine_group = QGroupBox("Refine & Bootstrap")
        refine_form = QFormLayout(refine_group)
        rp_row = QHBoxLayout()
        self.center_p = QDoubleSpinBox()
        self.center_p.setRange(0.00001, 10000); self.center_p.setDecimals(8)
        self.center_p.setValue(1.0); self.center_p.setSuffix(" d")
        rp_row.addWidget(self.center_p)
        btn_from_scan = QPushButton("← Best")
        btn_from_scan.setMaximumWidth(55)
        btn_from_scan.clicked.connect(self._set_center_from_scan)
        rp_row.addWidget(btn_from_scan)
        refine_form.addRow("Center P:", rp_row)
        self.n_boot = QSpinBox()
        self.n_boot.setRange(50, 2000); self.n_boot.setValue(300); self.n_boot.setSingleStep(50)
        refine_form.addRow("N bootstrap:", self.n_boot)
        self.refine_method_combo = QComboBox()
        self.refine_method_combo.addItem("Lomb-Scargle", "ls")
        self.refine_method_combo.addItem("PDM", "pdm")
        refine_form.addRow("Method:", self.refine_method_combo)
        self.btn_refine = QPushButton("Refine & Bootstrap")
        self.btn_refine.setStyleSheet(
            "QPushButton { background: #7B1FA2; color: white; font-weight: bold; padding: 6px; }"
            "QPushButton:disabled { background: #BDBDBD; }"
        )
        self.btn_refine.setEnabled(False)
        self.btn_refine.clicked.connect(self._run_refine)
        refine_form.addRow(self.btn_refine)
        self.refine_status = QLabel("")
        self.refine_status.setStyleSheet("font-size: 8pt; color: #555;")
        refine_form.addRow(self.refine_status)
        left_layout.addWidget(refine_group)

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
        btn_detect_t0 = QPushButton("Detect T₀ (min)")
        btn_detect_t0.clicked.connect(self._detect_t0)
        phase_form.addRow(btn_detect_t0)
        left_layout.addWidget(phase_group)

        left_layout.addStretch()

        # Log (collapsible)
        self.btn_log_toggle = QPushButton("Log ▼")
        self.btn_log_toggle.setCheckable(True)
        self.btn_log_toggle.setChecked(False)
        self.btn_log_toggle.setStyleSheet(
            "QPushButton { text-align: left; font-size: 8pt; padding: 2px 6px; }"
        )
        self.btn_log_toggle.toggled.connect(self._toggle_log)
        left_layout.addWidget(self.btn_log_toggle)
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumHeight(150)
        self.log_box.setStyleSheet("font-family: monospace; font-size: 8pt;")
        self.log_box.hide()
        left_layout.addWidget(self.log_box)

        # ---- Right panel (tabs) ----
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 4, 4)
        splitter.addWidget(right)
        splitter.setSizes([300, 900])

        self.tabs = QTabWidget()
        right_layout.addWidget(self.tabs)

        # Periodogram tab
        pg_tab = QWidget()
        pg_layout = QVBoxLayout(pg_tab)
        self.pg_canvas = FigureCanvas(Figure(figsize=(8, 4)))
        pg_layout.addWidget(NavigationToolbar(self.pg_canvas, pg_tab))
        pg_layout.addWidget(self.pg_canvas)
        self.tabs.addTab(pg_tab, "Periodogram")

        # Refine tab
        ref_tab = QWidget()
        ref_layout = QVBoxLayout(ref_tab)
        self.refine_label = QLabel("Refine & Bootstrap를 실행하세요.")
        self.refine_label.setStyleSheet(
            "QLabel { background: #F3E5F5; padding: 8px; border-radius: 4px; font-weight: bold; }"
        )
        ref_layout.addWidget(self.refine_label)
        ref_splitter = QSplitter(Qt.Horizontal)
        ref_left = QWidget()
        ref_left_l = QVBoxLayout(ref_left); ref_left_l.setContentsMargins(0, 0, 0, 0)
        self.ref_canvas = FigureCanvas(Figure(figsize=(5, 4)))
        ref_left_l.addWidget(NavigationToolbar(self.ref_canvas, ref_left))
        ref_left_l.addWidget(self.ref_canvas)
        ref_splitter.addWidget(ref_left)
        ref_right = QWidget()
        ref_right_l = QVBoxLayout(ref_right); ref_right_l.setContentsMargins(0, 0, 0, 0)
        self.boot_canvas = FigureCanvas(Figure(figsize=(5, 4)))
        ref_right_l.addWidget(NavigationToolbar(self.boot_canvas, ref_right))
        ref_right_l.addWidget(self.boot_canvas)
        ref_splitter.addWidget(ref_right)
        ref_layout.addWidget(ref_splitter, 1)
        self.tabs.addTab(ref_tab, "Refine")

        # Phase plot tab
        ph_tab = QWidget()
        ph_layout = QVBoxLayout(ph_tab)
        self.ph_canvas = FigureCanvas(Figure(figsize=(8, 5)))
        ph_layout.addWidget(NavigationToolbar(self.ph_canvas, ph_tab))
        ph_layout.addWidget(self.ph_canvas)
        self.tabs.addTab(ph_tab, "Phase Plot")

        # O-C tab
        oc_tab = self._build_oc_tab()
        self.tabs.addTab(oc_tab, "O-C")

        # Fourier tab
        fo_tab = self._build_fourier_tab()
        self.tabs.addTab(fo_tab, "Fourier")

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
        btn_from_refine = QPushButton("← Refined")
        btn_from_refine.clicked.connect(self._oc_from_refine)
        hdr.addWidget(btn_from_refine)
        hdr.addStretch()
        layout.addLayout(hdr)

        splitter = QSplitter(Qt.Horizontal)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        self.oc_table = QTableWidget()
        self.oc_table.setColumnCount(4)
        self.oc_table.setHorizontalHeaderLabels(["n", "BJD_obs", "O-C (d)", "err (d)"])
        self.oc_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.oc_table.horizontalHeader().setStretchLastSection(True)
        self.oc_table.setMinimumWidth(300)
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
        splitter.setSizes([330, 670])
        layout.addWidget(splitter, 1)

        return tab

    def _build_fourier_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Harmonics:"))
        self.n_harm = QSpinBox()
        self.n_harm.setRange(1, 8); self.n_harm.setValue(4)
        ctrl.addWidget(self.n_harm)
        ctrl.addWidget(QLabel("Filter:"))
        self.fourier_filter_combo = QComboBox()
        self.fourier_filter_combo.setMinimumWidth(80)
        self.fourier_filter_combo.addItem("All")
        ctrl.addWidget(self.fourier_filter_combo)
        btn_fourier = QPushButton("Decompose")
        btn_fourier.setStyleSheet(
            "QPushButton { background: #00796B; color: white; font-weight: bold; padding: 6px 12px; }"
        )
        btn_fourier.clicked.connect(self._run_fourier)
        ctrl.addWidget(btn_fourier)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        # Korean help text
        help_text = (
            "<b>[푸리에 분해 파라미터 물리적 의미]</b><br>"
            "광도곡선을 사인/코사인 급수로 분해: "
            "m(t) = A₀ + Σ Aₖ·cos(2πkt/P + φₖ)<br>"
            "<b>A₀</b>: 평균 밝기 (등급)<br>"
            "<b>Aₖ</b>: k번째 고조파 진폭 — 클수록 광도곡선이 해당 주기성분을 많이 포함<br>"
            "<b>φₖ</b>: k번째 고조파 위상 (rad)<br>"
            "<b>R₂₁ = A₂/A₁</b>: 광도곡선 <u>비대칭성 지수</u>. "
            "크면(>0.3) 급상승-완만한 하강(RRab·세페이드), 작으면(~0.1) 사인형(RRc·W UMa). "
            "별 종류 분류에 직접 사용됨.<br>"
            "<b>φ₂₁ = φ₂ − 2φ₁</b>: 2차 고조파의 <u>위상 비틀림</u>. "
            "RRab ≈ 2–5 rad, 세페이드는 주기에 따라 선형 증가. "
            "R₂₁–φ₂₁ 공간에서 같은 종류의 별들이 군집을 이룸.<br>"
            "<i>참고: Simon &amp; Lee (1981), Kovács &amp; Buchler (1988)</i>"
        )
        help_label = QLabel(help_text)
        help_label.setWordWrap(True)
        help_label.setStyleSheet(
            "QLabel { background: #FFF8E1; padding: 8px; border-radius: 4px; "
            "font-size: 8pt; border: 1px solid #FFE082; }"
        )
        layout.addWidget(help_label)

        self.fourier_label = QLabel("")
        self.fourier_label.setStyleSheet(
            "QLabel { background: #E0F2F1; padding: 8px; border-radius: 4px; "
            "font-family: monospace; font-size: 9pt; }"
        )
        self.fourier_label.setWordWrap(True)
        layout.addWidget(self.fourier_label)

        self.fourier_canvas = FigureCanvas(Figure(figsize=(8, 4)))
        layout.addWidget(NavigationToolbar(self.fourier_canvas, tab))
        layout.addWidget(self.fourier_canvas, 1)

        return tab

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _current_workspace_dir(self) -> Path:
        text = self.workspace_edit.text().strip() if hasattr(self, "workspace_edit") else ""
        path = Path(text) if text else Path(self.workspace_dir)
        self.workspace_dir = path
        return path

    def _browse_workspace(self):
        start_dir = self._current_workspace_dir()
        start = str(start_dir.parent if start_dir.exists() else Path(self.params.P.result_dir).parent)
        path = QFileDialog.getExistingDirectory(self, "Workspace 선택", start)
        if path:
            self.workspace_edit.setText(path)
            self._load_lc_from_workspace()

    def _load_lc_from_workspace(self):
        """Load all available workspace outputs and expose them in the Use data combo."""
        try:
            from ...utils.step_paths import list_lightcurve_csvs
            rd = self._current_workspace_dir()
            paths = list_lightcurve_csvs(rd)
            if not paths:
                self._clear_loaded_workspace_state(f"No lightcurve_*.csv found in\n{rd}")
                return
            self._load_paths(paths)
        except Exception as e:
            self._clear_loaded_workspace_state(f"Workspace load failed: {e}")

    def _clear_loaded_workspace_state(self, status: str):
        self.lc_data = None
        self.series_options = {}
        self.scan_result = None
        self.refined_period = None
        self.sigma_period = None
        self.mag_col_combo.blockSignals(True)
        self.mag_col_combo.clear()
        self.mag_col_combo.setEnabled(False)
        self.mag_col_combo.blockSignals(False)
        self.analysis_filter_combo.blockSignals(True)
        self.analysis_filter_combo.clear()
        self.analysis_filter_combo.addItem("All", "__all__")
        self.analysis_filter_combo.setEnabled(False)
        self.analysis_filter_combo.blockSignals(False)
        self.lc_status.setText(status)
        self.lc_status.setStyleSheet("color: #C62828;")
        for canvas_name in ("pg_canvas", "ref_canvas", "boot_canvas", "ph_canvas", "oc_canvas", "fourier_canvas"):
            canvas = getattr(self, canvas_name, None)
            if canvas is None:
                continue
            fig = getattr(canvas, "figure", None)
            if fig is None:
                continue
            fig.clear()
            canvas.draw_idle()

    def _load_paths(self, paths: list[Path]):
        try:
            rd = self._current_workspace_dir()
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
                    (c for c in ["BJD_TDB", "BJD", "bjd", "HJD", "hjd", "JD", "jd", "time"] if c in df.columns),
                    None
                )
                if time_col is None:
                    continue

                t = pd.to_numeric(df[time_col], errors="coerce").to_numpy(float)
                time_mask = np.isfinite(t)
                if not np.any(time_mask):
                    continue
                t = t[time_mask]

                err_col = next(
                    (c for c in ["diff_err_corr", "diff_err", "mag_err", "err", "sigma"] if c in df.columns),
                    None
                )
                e = pd.to_numeric(df[err_col], errors="coerce").to_numpy(float)[time_mask] if err_col else None

                filter_col = next(
                    (c for c in ["filter", "Filter", "FILTER", "band", "Band"] if c in df.columns), None
                )
                filters = df[filter_col].astype(str).to_numpy()[time_mask] if filter_col else None
                corr_tag = _detect_corr_mode_from_df(df, path.name)
                target_id = _detect_target_id_from_df(df, path.name)

                for label, col, arr in _collect_mag_options(df, time_mask, corr_tag=corr_tag):
                    key = f"{path.name}::{col}"
                    series_items.append({
                        "key": key,
                        "time": t,
                        "mag": arr,
                        "mag_col": col,
                        "mag_err": e,
                        "filters": filters,
                        "source": path.name,
                        "corr_tag": corr_tag,
                        "series_label": _describe_series(corr_tag, col),
                        "target_id": target_id,
                    })

            if not series_items:
                self._clear_loaded_workspace_state("No usable light curve series found")
                return

            multi_target = len({item["target_id"] for item in series_items if item.get("target_id") is not None}) > 1
            for item in series_items:
                if multi_target:
                    tid = item.get("target_id")
                    item["combo_label"] = f"ID{tid} | {item['series_label']}" if tid is not None else f"{item['source']} | {item['series_label']}"
                else:
                    item["combo_label"] = item["series_label"]

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
            self._clear_loaded_workspace_state(f"Error: {e}")
            self.log(f"[ERROR] {e}")

    def _apply_series_option(self, key: str | None):
        if not key or key not in self.series_options:
            return
        item = self.series_options[key]
        selected_filter = self._refresh_analysis_filter_combo(item.get("filters"))
        t = item["time"]
        mag = item["mag"]
        mag_err = item.get("mag_err")
        filters = item.get("filters")
        if selected_filter and selected_filter != "__all__" and filters is not None:
            mask = (filters == selected_filter)
            t = t[mask]
            mag = mag[mask]
            mag_err = mag_err[mask] if mag_err is not None else None
            filters = filters[mask]
        self.lc_data = {
            "time": t,
            "mag": mag,
            "mag_col": item["mag_col"],
            "mag_err": mag_err,
            "filters": filters,
            "source": item["source"],
            "corr_tag": item.get("corr_tag", ""),
            "analysis_filter": selected_filter,
            "series_label": item.get("series_label", item["mag_col"]),
        }
        n = int(np.sum(np.isfinite(self.lc_data["time"]) & np.isfinite(self.lc_data["mag"])))
        corr_line = f"  [{self.lc_data['corr_tag']}]" if self.lc_data.get("corr_tag") else ""
        workspace_dir = self._current_workspace_dir()
        workspace_name = workspace_dir.name
        workspace_type = ""
        try:
            from ...utils.run_workspace import load_run_manifest
            run_meta = load_run_manifest(workspace_dir)
            run_type = str(run_meta.get("run_type") or "").strip().lower()
            if run_type:
                workspace_type = f" [{run_type}]"
        except Exception:
            pass
        self.lc_status.setText(
            f"{workspace_name}{workspace_type}\n{self.lc_data['source']}\n{n} pts{corr_line}\n{self.mag_col_combo.currentText()}"
        )
        self.lc_status.setStyleSheet("color: green;")
        filt_label = self.lc_data.get("analysis_filter", "__all__")
        filt_info = f", filter={filt_label}" if filt_label and filt_label != "__all__" else ""
        self.log(
            f"Loaded: {self.lc_data['source']}  ({n} pts, {self.lc_data.get('series_label', self.lc_data['mag_col'])}"
            f"{filt_info}, detrend={self.lc_data.get('corr_tag') or 'N/A'})"
        )
        t0_guess = float(np.nanmin(self.lc_data["time"]))
        self.t0_edit.setValue(t0_guess)
        self.oc_t0.setValue(t0_guess)
        self._update_fourier_filter_combo()

    def _refresh_analysis_filter_combo(self, filters) -> str:
        current = self.analysis_filter_combo.currentData()
        filter_values = filters.tolist() if filters is not None else []
        unique_filters = sorted({str(f) for f in filter_values if str(f).strip() and str(f).lower() != "nan"})
        self.analysis_filter_combo.blockSignals(True)
        self.analysis_filter_combo.clear()
        if unique_filters:
            for f in unique_filters:
                self.analysis_filter_combo.addItem(f, f)
            self.analysis_filter_combo.addItem("All", "__all__")
            target = current if current in unique_filters or current == "__all__" else unique_filters[0]
        else:
            self.analysis_filter_combo.addItem("All", "__all__")
            target = "__all__"
        idx = max(self.analysis_filter_combo.findData(target), 0)
        self.analysis_filter_combo.setCurrentIndex(idx)
        self.analysis_filter_combo.setEnabled(self.analysis_filter_combo.count() > 0)
        self.analysis_filter_combo.blockSignals(False)
        return self.analysis_filter_combo.currentData()

    def _update_fourier_filter_combo(self):
        self.fourier_filter_combo.blockSignals(True)
        self.fourier_filter_combo.clear()
        self.fourier_filter_combo.addItem("All")
        if self.lc_data is not None:
            filters = self.lc_data.get("filters")
            if filters is not None:
                for f in sorted(set(filters)):
                    self.fourier_filter_combo.addItem(f)
        self.fourier_filter_combo.blockSignals(False)

    def _on_mag_col_changed(self):
        key = self.mag_col_combo.currentData()
        if not key:
            return
        self._apply_series_option(key)
        self._update_phase_plot()

    def _on_analysis_filter_changed(self):
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
        if self.chk_ls.isChecked(): methods.append("ls")
        if self.chk_pdm.isChecked(): methods.append("pdm")
        if self.chk_bls.isChecked(): methods.append("bls")
        if not methods:
            QMessageBox.warning(self, "No Method", "메서드를 하나 이상 선택하세요.")
            return

        self.scan_status.setText("Running…")
        mag = self.lc_data["mag"]
        self._scan_worker = PeriodAnalysisWorker(
            time=self.lc_data["time"],
            mag_raw=mag,
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

    def _on_scan_done(self, results: dict):
        # results keyed as "raw_ls", "raw_pdm", "raw_bls"
        self.scan_result = results
        best_period, best_power, fap = self._best_from_results(results)
        fap_str = f"{fap:.2e}" if np.isfinite(fap) else "—"
        methods_run = [k.split("_", 1)[1].upper() for k in results]
        self.scan_status.setText(
            f"Best P = {best_period:.6f} d  [{'/'.join(methods_run)}]"
        )
        self.center_p.setValue(best_period)
        self.phase_p.setValue(best_period)
        self.oc_p.setValue(best_period)
        self.btn_refine.setEnabled(True)
        self._draw_periodogram(results)
        self._update_phase_plot()
        self.log(f"Scan done: P={best_period:.6f} d, power={best_power:.4f}, FAP={fap_str}")

    def _best_from_results(self, results: dict):
        """Pick best period across all methods (prefer ls > pdm > bls)."""
        for key in ("raw_ls", "raw_pdm", "raw_bls"):
            d = results.get(key)
            if d and "error" not in d and np.isfinite(d.get("best_period", np.nan)):
                return float(d["best_period"]), float(d["best_power"]), float(d.get("fap", np.nan))
        return np.nan, np.nan, np.nan

    def _draw_periodogram(self, results: dict):
        method_labels = {"ls": "Lomb-Scargle", "pdm": "PDM (1-θ)", "bls": "BLS"}
        method_colors = {"ls": "#1E88E5", "pdm": "#E53935", "bls": "#FF9800"}
        y_labels = {"ls": "LS Power", "pdm": "1 - θ", "bls": "BLS Power"}

        n = len(results)
        fig = self.pg_canvas.figure
        fig.clear()
        if n == 0:
            self.pg_canvas.draw_idle()
            return
        axes = fig.subplots(1, n, squeeze=False)[0]

        for ax, (key, data) in zip(axes, results.items()):
            method = key.split("_", 1)[1]
            if "error" in data:
                ax.text(0.5, 0.5, data["error"], ha="center", va="center",
                        transform=ax.transAxes, fontsize=9)
                ax.set_title(method_labels.get(method, method))
                continue
            if "frequency" in data:
                periods = 1.0 / data["frequency"]
            else:
                periods = data["trial_periods"]
            power = data["power"]
            best = data["best_period"]
            color = method_colors.get(method, "#666")
            ax.plot(periods, power, color=color, lw=0.8, alpha=0.9)
            ax.axvline(best, color="red", ls="--", lw=1.5, label=f"P={best:.6f} d")
            ax.scatter([best], [data["best_power"]], color="red", s=50, zorder=5)
            for p in data.get("top_periods", [])[1:4]:
                ax.axvline(p, color="orange", ls=":", lw=0.8, alpha=0.6)
            ax.set_xscale("log")
            ax.set_xlabel("Period (days)")
            ax.set_ylabel(y_labels.get(method, "Power"))
            ax.set_title(f"{method_labels.get(method, method)}\nP={best:.6f} d")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        self.pg_canvas.draw_idle()

    # ------------------------------------------------------------------
    # Refine
    # ------------------------------------------------------------------

    def _set_center_from_scan(self):
        if self.scan_result:
            best, _, _ = self._best_from_results(self.scan_result)
            if np.isfinite(best):
                self.center_p.setValue(best)

    def _run_refine(self):
        if self.lc_data is None:
            return
        if self._refine_worker and self._refine_worker.isRunning():
            return
        self.btn_refine.setEnabled(False)
        self.refine_status.setText("Running…")
        self.tabs.setCurrentIndex(1)  # Refine tab
        refine_method = self.refine_method_combo.currentData() or "ls"
        self._refine_worker = RefineBootstrapWorker(
            self.lc_data["time"], self.lc_data["mag"], self.lc_data.get("mag_err"),
            center_period=self.center_p.value(),
            n_bootstrap=self.n_boot.value(),
            method=refine_method,
        )
        self._refine_worker.progress.connect(self.refine_status.setText)
        self._refine_worker.finished.connect(self._on_refine_done)
        self._refine_worker.error.connect(self._on_refine_error)
        self._refine_worker.start()

    def _on_refine_done(self, result: dict):
        self.btn_refine.setEnabled(True)
        p = result["refined_period"]
        sig = result["sigma_p"]
        method_tag = result.get("method", "ls").upper()
        self.refined_period = p
        self.sigma_period = sig
        self.refine_status.setText(f"[{method_tag}] P = {p:.8f} ± {sig:.2e} d")
        self.refine_label.setText(
            f"P = {p:.8f} d  ±  {sig:.2e} d   (1σ bootstrap, N={self.n_boot.value()})"
        )
        self.phase_p.setValue(p)
        self.oc_p.setValue(p)
        self._draw_refine_plots(result)
        self._update_phase_plot()
        self.log(f"Refined: P={p:.8f} ± {sig:.2e} d")

    def _on_refine_error(self, msg: str):
        self.btn_refine.setEnabled(True)
        self.refine_status.setText("Error")
        QMessageBox.warning(self, "Refine Error", msg)

    def _draw_refine_plots(self, result: dict):
        fp = result["fine_periods"]
        pw = result["fine_power"]
        p_best = result["refined_period"]
        sig = result["sigma_p"]

        fig1 = self.ref_canvas.figure
        fig1.clear()
        ax1 = fig1.add_subplot(111)
        ax1.plot(fp, pw, color="#7B1FA2", lw=0.8)
        ax1.axvline(p_best, color="red", ls="--", lw=1.5, label=f"P={p_best:.8f} d")
        ax1.set_xlabel("Period (days)")
        ax1.set_ylabel("LS Power")
        ax1.set_title("Fine Period Grid")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        fig1.tight_layout()
        self.ref_canvas.draw_idle()

        bp = result["boot_periods"]
        fig2 = self.boot_canvas.figure
        fig2.clear()
        ax2 = fig2.add_subplot(111)
        ax2.hist(bp, bins=min(40, len(bp) // 3 + 1), color="#7B1FA2", alpha=0.7, edgecolor="white")
        ax2.axvline(p_best, color="red", ls="--", lw=2, label=f"P={p_best:.8f}")
        ax2.axvline(p_best - sig, color="orange", ls=":", lw=1.5)
        ax2.axvline(p_best + sig, color="orange", ls=":", lw=1.5, label=f"±1σ={sig:.2e}")
        ax2.set_xlabel("Bootstrap Period (d)")
        ax2.set_ylabel("Count")
        ax2.set_title(f"Bootstrap Distribution (N={len(bp)})")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        self.boot_canvas.draw_idle()

    # ------------------------------------------------------------------
    # Phase plot
    # ------------------------------------------------------------------

    def _detect_t0(self):
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
        n_bins = 30
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
        min_idx = int(np.nanargmax(bin_mags))  # faintest = eclipse minimum
        phi_min = float(centers[min_idx])
        # Sub-bin parabola
        hw = 4
        idxs = np.array([(min_idx + d) % n_bins for d in range(-hw, hw + 1)])
        phi_unwrap = np.unwrap(centers[idxs] * 2 * np.pi) / (2 * np.pi)
        mag_fit = bin_mags[idxs]
        vf = np.isfinite(mag_fit)
        if vf.sum() >= 3:
            try:
                coeffs = np.polyfit(phi_unwrap[vf], mag_fit[vf], 2)
                if coeffs[0] > 0:
                    pv = -coeffs[1] / (2 * coeffs[0]) % 1.0
                    if abs(pv - phi_min) < 0.15:
                        phi_min = float(pv)
            except Exception:
                pass
        t0_epoch = t0_ref + phi_min * period
        self.t0_edit.setValue(t0_epoch)
        self.oc_t0.setValue(t0_epoch)
        self.log(f"T₀ detected: {t0_epoch:.6f} BJD  (φ_min={phi_min:.4f})")

    def _update_phase_plot(self):
        if self.lc_data is None:
            return
        period = self.phase_p.value()
        t0 = self.t0_edit.value()
        t = self.lc_data["time"]
        mag = self.lc_data["mag"]
        mag_err = self.lc_data.get("mag_err")
        filters = self.lc_data.get("filters")
        mask = np.isfinite(t) & np.isfinite(mag)
        t_v, m_v = t[mask], mag[mask]
        dy_v = mag_err[mask] if mag_err is not None else None
        filt_v = filters[mask] if filters is not None else None

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
        ax.set_title(f"Phase Plot  P={period:.6f} d   T₀={t0:.4f}  [{self.lc_data['source']}]")
        ax.set_xlim(0, 2)
        ax.axvline(0, color="gray", ls=":", alpha=0.4)
        ax.axvline(1, color="gray", ls=":", alpha=0.4)
        ax.grid(True, alpha=0.3)
        if len(unique_filters) > 1 or (unique_filters and unique_filters[0] != ""):
            ax.legend(fontsize=8, title="Filter")

        # Check star phase-folded overlay
        try:
            _rd = self._current_workspace_dir()
            _check_filter = _resolve_check_filter(
                self.lc_data.get("filters"),
                self.lc_data.get("analysis_filter"),
            )
            _ck_id, _ck_df = _load_check_star_for_plot(_rd, filt=_check_filter)
            if _ck_df is not None and not _ck_df.empty:
                _t_col, _y_col = _pick_check_overlay_cols(_ck_df, self.lc_data.get("mag_col"))
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
                        ax.legend(fontsize=8)
        except Exception:
            pass

        fig.tight_layout()
        self.ph_canvas.draw_idle()

    # ------------------------------------------------------------------
    # O-C
    # ------------------------------------------------------------------

    def _oc_from_refine(self):
        if self.refined_period is not None:
            self.oc_p.setValue(self.refined_period)
        if self.t0_edit.value():
            self.oc_t0.setValue(self.t0_edit.value())

    def _oc_add(self):
        r = self.oc_table.rowCount()
        self.oc_table.insertRow(r)
        for c in range(4):
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
                (c for c in df.columns if c.lower() in ("bjd_obs", "bjd", "hjd", "jd")), None
            )
            if bjd_col is None:
                QMessageBox.warning(self, "Error", "BJD 컬럼을 찾을 수 없습니다.")
                return
            err_col = next(
                (c for c in df.columns if c.lower() in ("err", "error", "sigma")), None
            )
            t0 = self.oc_t0.value()
            p = self.oc_p.value()
            self.oc_table.setRowCount(0)
            for _, row_data in df.iterrows():
                bjd = float(row_data[bjd_col])
                n = int(round((bjd - t0) / p)) if p > 0 else 0
                oc = bjd - (t0 + n * p)
                err = f"{float(row_data[err_col]):.6f}" if err_col else ""
                r = self.oc_table.rowCount()
                self.oc_table.insertRow(r)
                self.oc_table.setItem(r, 0, QTableWidgetItem(str(n)))
                self.oc_table.setItem(r, 1, QTableWidgetItem(f"{bjd:.6f}"))
                self.oc_table.setItem(r, 2, QTableWidgetItem(f"{oc:.6f}"))
                self.oc_table.setItem(r, 3, QTableWidgetItem(err))
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
                for c in range(4)
            ])
        pd.DataFrame(rows, columns=["n", "BJD_obs", "O-C (d)", "err (d)"]).to_csv(path, index=False)
        self.log(f"Exported O-C to {path}")

    def _recompute_oc(self):
        t0 = self.oc_t0.value()
        p = self.oc_p.value()
        if p <= 0:
            return
        for r in range(self.oc_table.rowCount()):
            bjd_item = self.oc_table.item(r, 1)
            if not bjd_item or not bjd_item.text():
                continue
            try:
                bjd = float(bjd_item.text())
                n = int(round((bjd - t0) / p))
                oc = bjd - (t0 + n * p)
                self.oc_table.setItem(r, 0, QTableWidgetItem(str(n)))
                self.oc_table.setItem(r, 2, QTableWidgetItem(f"{oc:.6f}"))
            except ValueError:
                pass
        self._draw_oc()

    def _get_oc_arrays(self):
        ns, ocs, errs = [], [], []
        for r in range(self.oc_table.rowCount()):
            try:
                n_it = self.oc_table.item(r, 0)
                oc_it = self.oc_table.item(r, 2)
                er_it = self.oc_table.item(r, 3)
                if not n_it or not oc_it or not n_it.text() or not oc_it.text():
                    continue
                ns.append(int(n_it.text()))
                ocs.append(float(oc_it.text()))
                errs.append(float(er_it.text()) if (er_it and er_it.text()) else np.nan)
            except (ValueError, TypeError):
                pass
        return np.array(ns), np.array(ocs), np.array(errs)

    def _draw_oc(self, fit_result=None):
        fig = self.oc_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        ns, ocs, errs = self._get_oc_arrays()
        if len(ns) == 0:
            ax.text(0.5, 0.5, "O-C 데이터 없음", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="gray")
            fig.tight_layout()
            self.oc_canvas.draw_idle()
            return
        oc_min = ocs * 1440
        has_err = np.any(np.isfinite(errs))
        if has_err:
            err_pl = np.where(np.isfinite(errs), errs * 1440, 0.0)
            ax.errorbar(ns, oc_min, yerr=err_pl, fmt="o", color="#1565C0",
                        markersize=5, elinewidth=1, capsize=3, label="O-C")
        else:
            ax.scatter(ns, oc_min, color="#1565C0", s=40, zorder=5, label="O-C")
        ax.axhline(0, color="gray", ls="--", lw=1, alpha=0.6)
        if fit_result is not None:
            n_fit = np.linspace(ns.min(), ns.max(), 500)
            ax.plot(n_fit, fit_result["func"](n_fit) * 1440, color="red",
                    lw=1.5, label=fit_result["label"])
        ax.set_xlabel("Epoch (n)")
        ax.set_ylabel("O-C (minutes)")
        ax.set_title("O-C Diagram")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        self.oc_canvas.draw_idle()

    def _oc_fit(self):
        ns, ocs, errs = self._get_oc_arrays()
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
                    f"O-C = {popt[0]*1440:.3f} + {dp*1440:.4f}·n  (min)\n"
                    f"ΔP = {dp:.3e} ± {perr[1]:.1e} d/cycle\n"
                    f"Corrected P = {p + dp:.8f} d"
                )
                fit_result = {"func": lambda n: model(n, *popt), "label": f"Linear (ΔP={dp:.2e})"}

            elif mode == "Parabola (dP/dt)":
                def model(n, a, b, c): return a + b * n + c * n**2
                popt, pcov = curve_fit(model, ns, ocs, sigma=sigma, absolute_sigma=True, maxfev=10000)
                perr = np.sqrt(np.diag(pcov))
                c = popt[2]
                dP_dt = 2 * c / p
                dP_yr = dP_dt * 365.25
                arrow = "▲ 주기 증가" if dP_dt > 0 else "▼ 주기 감소"
                text = (
                    f"c = {c:.3e} ± {perr[2]:.1e} d/cycle²\n"
                    f"dP/dt = {dP_dt:.3e} d/d  ({dP_yr:.3e} d/yr)\n"
                    f"{arrow}"
                )
                fit_result = {"func": lambda n: model(n, *popt), "label": f"Parabola (dP/dt={dP_dt:.2e})"}

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
                p3_days = p3_n * p
                p3_yr = p3_days / 365.25
                a12_sin_i = abs(A) * 173.145
                text = (
                    f"dP/dt = {dP_dt:.3e} d/d\n"
                    f"제3천체 P₃ = {p3_days:.1f} d  ({p3_yr:.2f} yr)\n"
                    f"진폭 A = {A*1440:.2f} min\n"
                    f"a₁₂·sin(i) = {a12_sin_i:.4f} AU"
                )
                fit_result = {"func": lambda n: model(n, *popt), "label": f"Para+Sine (P₃={p3_yr:.2f} yr)"}

        except Exception as e:
            text = f"피팅 실패: {e}"

        self.oc_fit_label.setText(text)
        self._draw_oc(fit_result)
        self.log(f"[O-C Fit] {mode}: {text.splitlines()[0]}")

    # ------------------------------------------------------------------
    # Fourier decomposition
    # ------------------------------------------------------------------

    def _run_fourier(self):
        if self.lc_data is None:
            QMessageBox.warning(self, "No Data", "Load a light curve first.")
            return
        period = self.phase_p.value()
        if period <= 0:
            return

        t = self.lc_data["time"]
        mag = self.lc_data["mag"]
        filters = self.lc_data.get("filters")
        t0 = self.t0_edit.value()

        # Filter selection
        sel_filter = self.fourier_filter_combo.currentText()
        if sel_filter != "All" and filters is not None:
            fmask = (filters == sel_filter)
        else:
            fmask = np.ones(len(t), bool)

        mask = np.isfinite(t) & np.isfinite(mag) & fmask
        t_v, m_v = t[mask], mag[mask]
        filt_v = filters[mask] if filters is not None else None

        if len(t_v) < 20:
            QMessageBox.warning(self, "Fourier", "데이터가 너무 적습니다 (< 20).")
            return

        nh = self.n_harm.value()
        phase = ((t_v - t0) / period) % 1.0
        omega = 2.0 * np.pi / period

        # Design matrix: 1, cos(kωt), sin(kωt)
        A = np.column_stack(
            [np.ones(len(t_v))] +
            [f(k * omega * t_v) for k in range(1, nh + 1) for f in (np.cos, np.sin)]
        )
        try:
            coeff, _, _, _ = np.linalg.lstsq(A, m_v, rcond=None)
        except Exception as e:
            QMessageBox.warning(self, "Fourier", f"LSQ failed: {e}")
            return

        a0 = coeff[0]
        a_k = coeff[1::2]
        b_k = coeff[2::2]
        amp_k = np.sqrt(a_k**2 + b_k**2)
        phi_k = np.arctan2(b_k, a_k)

        # Simon-Lee parameters
        def _wrap(x): return (x + np.pi) % (2 * np.pi) - np.pi
        R21 = amp_k[1] / amp_k[0] if len(amp_k) > 1 and amp_k[0] > 0 else np.nan
        phi21 = _wrap(phi_k[1] - 2 * phi_k[0]) if len(phi_k) > 1 else np.nan
        R31 = amp_k[2] / amp_k[0] if len(amp_k) > 2 and amp_k[0] > 0 else np.nan
        phi31 = _wrap(phi_k[2] - 3 * phi_k[0]) if len(phi_k) > 2 else np.nan

        # Dense model for amplitude metrics
        ph_model = np.linspace(0, 1, 1000)
        t_model = t0 + ph_model * period
        A_model = np.column_stack(
            [np.ones(1000)] +
            [f(k * omega * t_model) for k in range(1, nh + 1) for f in (np.cos, np.sin)]
        )
        mag_model = A_model @ coeff

        amp_ptp = float(np.max(mag_model) - np.min(mag_model))
        obs_range = float(np.percentile(m_v, 95) - np.percentile(m_v, 5))
        residuals = m_v - (A @ coeff)
        rms_res = float(np.std(residuals))

        # Classification hint
        if np.isfinite(R21):
            if R21 > 0.4:
                hint = "→ RRab / Cepheid 특징  (비대칭 광도곡선, 급상승·완만한 하강)"
            elif R21 > 0.25:
                hint = "→ δ Sct / 혼합형"
            elif R21 > 0.1:
                hint = "→ RRc / W UMa 특징  (대칭적·사인형 광도곡선)"
            else:
                hint = "→ 거의 순수 사인형  (소진폭 맥동)"
        else:
            hint = ""

        filt_label = sel_filter if sel_filter != "All" else "All filters"
        text = f"[{filt_label}   N={nh}고조파   n={len(t_v)} pts]\n"
        text += "\n=== 등급 변화 ===\n"
        text += f"모델 진폭 (peak-to-peak) : {amp_ptp:.4f} mag\n"
        text += f"관측 범위 (5–95 pctile)  : {obs_range:.4f} mag\n"
        text += f"잔차 RMS                 : {rms_res:.4f} mag\n"
        text += "\n=== 고조파 계수 ===\n"
        text += f"A₀ = {a0:.4f} mag\n"
        for k in range(nh):
            text += f"A{k+1} = {amp_k[k]:.4f}   φ{k+1} = {phi_k[k]:.4f} rad\n"
        text += "\n=== Simon-Lee 파라미터 ===\n"
        text += f"R₂₁ = A₂/A₁     = {R21:.4f}\n"
        text += f"φ₂₁ = φ₂−2φ₁   = {phi21:.4f} rad\n"
        if np.isfinite(R31):
            text += f"R₃₁ = A₃/A₁     = {R31:.4f}\n"
            text += f"φ₃₁ = φ₃−3φ₁   = {phi31:.4f} rad\n"
        text += f"\n{hint}"
        self.fourier_label.setText(text)

        # Plot
        fig = self.fourier_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)

        if sel_filter == "All" and filt_v is not None:
            for fi, filt in enumerate(sorted(set(filt_v))):
                sel = filt_v == filt
                color = self.filter_colors.get(filt) or _filt_color(filt, fi)
                ax.scatter(phase[sel], m_v[sel], color=color, s=8, alpha=0.6, label=filt)
        else:
            color = (self.filter_colors.get(sel_filter) or _filt_color(sel_filter, 0)
                     if sel_filter != "All" else "#1E88E5")
            ax.scatter(phase, m_v, color=color, s=8, alpha=0.6, label=filt_label)

        ax.plot(ph_model, mag_model, color="red", lw=1.5, label=f"Fourier (N={nh})")
        # Mark max/min of model
        ax.axhline(np.max(mag_model), color="gray", ls=":", lw=0.8, alpha=0.7)
        ax.axhline(np.min(mag_model), color="gray", ls=":", lw=0.8, alpha=0.7)
        ax.invert_yaxis()
        ax.set_xlabel("Phase")
        ax.set_ylabel("Magnitude")
        r21_str = f"{R21:.3f}" if np.isfinite(R21) else "—"
        ax.set_title(f"Fourier [{filt_label}]   Δm={amp_ptp:.3f}   R₂₁={r21_str}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        self.fourier_canvas.draw_idle()

        self.log(f"Fourier [{filt_label}] N={nh}: Δm={amp_ptp:.3f}, R21={R21:.4f}, φ21={phi21:.4f}")

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
    # Helpers
    # ------------------------------------------------------------------

    def _toggle_log(self, checked: bool):
        self.log_box.setVisible(checked)
        self.btn_log_toggle.setText("Log ▲" if checked else "Log ▼")

    def log(self, msg: str):
        self.log_box.append(msg)
