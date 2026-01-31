"""
Step 13: Period Analysis (Lomb-Scargle Periodogram)

- Lomb-Scargle periodogram for period detection
- Phase folded light curve plots
- Support for both raw and corrected magnitudes
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from astropy.timeseries import LombScargle

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QGroupBox,
    QLineEdit,
    QCheckBox,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QFileDialog,
    QTextEdit,
    QWidget,
    QComboBox,
    QFormLayout,
    QDoubleSpinBox,
    QSpinBox,
    QTabWidget,
    QSplitter,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from .step_window_base import StepWindowBase
from ...utils.step_paths import step11_dir, step12_dir, step13_period_dir


def _safe_float(value, default: float = np.nan) -> float:
    try:
        return float(value)
    except Exception:
        return default


class PeriodAnalysisWorker(QThread):
    """Worker thread for period analysis (Lomb-Scargle, PDM, BLS)."""
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(
        self,
        time: np.ndarray,
        mag_raw: np.ndarray,
        mag_corr: np.ndarray,
        mag_err: Optional[np.ndarray],
        min_period: float,
        max_period: float,
        samples_per_peak: int = 10,
        methods: Optional[List[str]] = None,
        pdm_n_bins: int = 10,
    ):
        super().__init__()
        self.time = time
        self.mag_raw = mag_raw
        self.mag_corr = mag_corr
        self.mag_err = mag_err
        self.min_period = min_period
        self.max_period = max_period
        self.samples_per_peak = samples_per_peak
        self.methods = methods or ["ls"]
        self.pdm_n_bins = pdm_n_bins

    def run(self):
        try:
            results = {}

            for method in self.methods:
                # Raw magnitude
                self.progress.emit(f"Computing {method.upper()} for raw magnitudes...")
                raw_result = self._compute(
                    self.time, self.mag_raw, self.mag_err, "raw", method
                )
                results[f"raw_{method}"] = raw_result

                # Corrected magnitude
                if self.mag_corr is not None and np.any(np.isfinite(self.mag_corr)):
                    self.progress.emit(f"Computing {method.upper()} for corrected magnitudes...")
                    corr_result = self._compute(
                        self.time, self.mag_corr, self.mag_err, "corr", method
                    )
                    results[f"corr_{method}"] = corr_result

            self.finished.emit(results)

        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")

    def _filter_valid(
        self, time: np.ndarray, mag: np.ndarray, mag_err: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        mask = np.isfinite(time) & np.isfinite(mag)
        if mag_err is not None:
            mask &= np.isfinite(mag_err)
        t = time[mask]
        y = mag[mask]
        dy = mag_err[mask] if mag_err is not None else None
        return t, y, dy

    def _compute(
        self,
        time: np.ndarray,
        mag: np.ndarray,
        mag_err: Optional[np.ndarray],
        label: str,
        method: str,
    ) -> dict:
        if method == "ls":
            return self._compute_ls(time, mag, mag_err, label)
        elif method == "pdm":
            return self._compute_pdm(time, mag, mag_err, label)
        elif method == "bls":
            return self._compute_bls(time, mag, mag_err, label)
        else:
            return {"label": label, "method": method, "error": f"Unknown method: {method}"}

    def _compute_ls(
        self,
        time: np.ndarray,
        mag: np.ndarray,
        mag_err: Optional[np.ndarray],
        label: str,
    ) -> dict:
        """Compute Lomb-Scargle periodogram."""
        t, y, dy = self._filter_valid(time, mag, mag_err)

        if len(t) < 10:
            return {
                "label": label, "method": "ls",
                "error": "Not enough data points (< 10)",
                "best_period": np.nan, "best_power": np.nan,
            }

        if dy is not None and np.any(dy > 0):
            ls = LombScargle(t, y, dy)
        else:
            ls = LombScargle(t, y)

        min_freq = 1.0 / self.max_period
        max_freq = 1.0 / self.min_period

        frequency, power = ls.autopower(
            minimum_frequency=min_freq,
            maximum_frequency=max_freq,
            samples_per_peak=self.samples_per_peak,
        )

        best_idx = np.argmax(power)
        best_freq = frequency[best_idx]
        best_power = power[best_idx]
        best_period = 1.0 / best_freq if best_freq > 0 else np.nan

        peak_indices, _ = find_peaks(power, height=0.1 * best_power)
        if len(peak_indices) == 0:
            peak_indices = [best_idx]

        sorted_peaks = sorted(peak_indices, key=lambda i: power[i], reverse=True)[:5]
        top_periods = [1.0 / frequency[i] for i in sorted_peaks]
        top_powers = [power[i] for i in sorted_peaks]

        try:
            fap = ls.false_alarm_probability(best_power)
        except Exception:
            fap = np.nan

        return {
            "label": label, "method": "ls",
            "frequency": frequency, "power": power,
            "best_period": best_period, "best_power": best_power,
            "best_freq": best_freq, "fap": fap,
            "top_periods": top_periods, "top_powers": top_powers,
            "n_points": len(t), "time": t, "mag": y, "mag_err": dy,
        }

    def _compute_pdm(
        self,
        time: np.ndarray,
        mag: np.ndarray,
        mag_err: Optional[np.ndarray],
        label: str,
    ) -> dict:
        """Phase Dispersion Minimization (Stellingwerf 1978).

        Less susceptible to daily aliasing than Lomb-Scargle because
        it measures phase-binned scatter rather than fitting sinusoids.
        """
        t, y, dy = self._filter_valid(time, mag, mag_err)

        if len(t) < 10:
            return {
                "label": label, "method": "pdm",
                "error": "Not enough data points (< 10)",
                "best_period": np.nan, "best_power": np.nan,
            }

        total_var = np.var(y)
        if total_var == 0:
            return {
                "label": label, "method": "pdm",
                "error": "Zero variance in data",
                "best_period": np.nan, "best_power": np.nan,
            }

        # Build trial period grid (same density as LS)
        baseline = t.max() - t.min()
        n_trials = max(1000, int(self.samples_per_peak * baseline / self.min_period))
        n_trials = min(n_trials, 50000)
        trial_periods = np.linspace(self.min_period, self.max_period, n_trials)

        n_bins = self.pdm_n_bins
        theta = np.empty(n_trials, dtype=float)

        # Weights (1/sigma^2) or uniform
        if dy is not None and np.any(dy > 0):
            w = 1.0 / (dy ** 2)
        else:
            w = np.ones(len(t), dtype=float)

        t0 = t.min()
        for ip, p in enumerate(trial_periods):
            phase = ((t - t0) / p) % 1.0
            bin_idx = np.clip((phase * n_bins).astype(int), 0, n_bins - 1)

            s2_bins = 0.0
            n_bins_used = 0
            for b in range(n_bins):
                mask_b = bin_idx == b
                nb = np.sum(mask_b)
                if nb < 2:
                    continue
                yb = y[mask_b]
                wb = w[mask_b]
                wmean = np.average(yb, weights=wb)
                s2_bins += np.sum(wb * (yb - wmean) ** 2) / np.sum(wb)
                n_bins_used += 1

            if n_bins_used > 0:
                theta[ip] = (s2_bins / n_bins_used) / total_var
            else:
                theta[ip] = 1.0

        # PDM: best period = minimum theta
        # Convert to "power" = 1 - theta for consistent peak-finding
        power = 1.0 - theta

        best_idx = np.argmax(power)
        best_period = trial_periods[best_idx]
        best_power = power[best_idx]
        best_theta = theta[best_idx]

        # Find top peaks
        peak_indices, _ = find_peaks(power, height=0.1 * best_power)
        if len(peak_indices) == 0:
            peak_indices = [best_idx]
        sorted_peaks = sorted(peak_indices, key=lambda i: power[i], reverse=True)[:5]
        top_periods = [trial_periods[i] for i in sorted_peaks]
        top_powers = [power[i] for i in sorted_peaks]

        return {
            "label": label, "method": "pdm",
            "trial_periods": trial_periods,
            "theta": theta, "power": power,
            "best_period": best_period, "best_power": best_power,
            "best_theta": best_theta, "fap": np.nan,
            "top_periods": top_periods, "top_powers": top_powers,
            "n_points": len(t), "time": t, "mag": y, "mag_err": dy,
        }

    def _compute_bls(
        self,
        time: np.ndarray,
        mag: np.ndarray,
        mag_err: Optional[np.ndarray],
        label: str,
    ) -> dict:
        """Box Least Squares (Kovacs et al. 2002).

        Optimal for eclipsing binaries and transiting exoplanets.
        """
        from astropy.timeseries import BoxLeastSquares

        t, y, dy = self._filter_valid(time, mag, mag_err)

        if len(t) < 10:
            return {
                "label": label, "method": "bls",
                "error": "Not enough data points (< 10)",
                "best_period": np.nan, "best_power": np.nan,
            }

        if dy is not None and np.any(dy > 0):
            bls = BoxLeastSquares(t, y, dy=dy)
        else:
            bls = BoxLeastSquares(t, y)

        baseline = t.max() - t.min()
        # Duration must be strictly less than min_period
        max_dur = min(self.min_period * 0.5, self.max_period * 0.25, baseline * 0.25)
        min_dur = max(self.min_period * 0.01, max_dur * 0.05)
        if min_dur >= max_dur:
            min_dur = max_dur * 0.1
        durations = np.linspace(min_dur, max_dur, 10)

        try:
            result = bls.autopower(
                durations,
                minimum_period=self.min_period,
                maximum_period=self.max_period,
            )
        except Exception as e:
            return {
                "label": label, "method": "bls",
                "error": f"BLS failed: {e}",
                "best_period": np.nan, "best_power": np.nan,
            }

        power = result.power
        periods = result.period

        best_idx = np.argmax(power)
        best_period = float(periods[best_idx])
        best_power = float(power[best_idx])

        peak_indices, _ = find_peaks(power, height=0.1 * best_power)
        if len(peak_indices) == 0:
            peak_indices = [best_idx]
        sorted_peaks = sorted(peak_indices, key=lambda i: power[i], reverse=True)[:5]
        top_periods = [float(periods[i]) for i in sorted_peaks]
        top_powers = [float(power[i]) for i in sorted_peaks]

        return {
            "label": label, "method": "bls",
            "trial_periods": np.array(periods, dtype=float),
            "power": np.array(power, dtype=float),
            "best_period": best_period, "best_power": best_power,
            "fap": np.nan,
            "top_periods": top_periods, "top_powers": top_powers,
            "n_points": len(t), "time": t, "mag": y, "mag_err": dy,
        }


class PeriodAnalysisWindow(StepWindowBase):
    """Step 13: Period Analysis with Lomb-Scargle Periodogram"""

    def __init__(self, params, file_manager, project_state, main_window):
        self.file_manager = file_manager
        self.worker = None
        self.results = {}
        self.lc_data = None
        self.current_filter = None

        super().__init__(
            step_index=12,
            step_name="Period Analysis",
            params=params,
            project_state=project_state,
            main_window=main_window,
        )

        self.setup_step_ui()
        self.restore_state()

    def setup_step_ui(self):
        info = QLabel(
            "Period analysis: Lomb-Scargle, PDM, BLS.\n"
            "Multiple methods reduce aliasing; consensus = true period."
        )
        info.setStyleSheet("QLabel { background-color: #E3F2FD; padding: 10px; border-radius: 5px; }")
        info.setWordWrap(True)
        self.content_layout.addWidget(info)

        # Data selection
        data_group = QGroupBox("Data Selection")
        data_layout = QFormLayout(data_group)

        # Filter selection
        self.filter_combo = QComboBox()
        self.filter_combo.currentIndexChanged.connect(self._on_filter_changed)
        data_layout.addRow("Filter:", self.filter_combo)

        # Target ID
        self.target_id_spin = QSpinBox()
        self.target_id_spin.setRange(1, 99999)
        self.target_id_spin.setValue(1)
        data_layout.addRow("Target ID:", self.target_id_spin)

        # Load data button
        load_row = QHBoxLayout()
        self.btn_load = QPushButton("Load Light Curve")
        self.btn_load.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; font-weight: bold; padding: 8px 15px; }"
        )
        self.btn_load.clicked.connect(self._load_light_curve)
        load_row.addWidget(self.btn_load)

        self.data_status = QLabel("No data loaded")
        load_row.addWidget(self.data_status)
        load_row.addStretch()
        data_layout.addRow("", load_row)

        self.content_layout.addWidget(data_group)

        # Period search parameters
        param_group = QGroupBox("Period Search Parameters")
        param_layout = QFormLayout(param_group)

        self.min_period_spin = QDoubleSpinBox()
        self.min_period_spin.setRange(0.001, 100.0)
        self.min_period_spin.setDecimals(4)
        self.min_period_spin.setValue(0.01)
        self.min_period_spin.setSuffix(" days")
        param_layout.addRow("Min Period:", self.min_period_spin)

        self.max_period_spin = QDoubleSpinBox()
        self.max_period_spin.setRange(0.01, 1000.0)
        self.max_period_spin.setDecimals(4)
        self.max_period_spin.setValue(10.0)
        self.max_period_spin.setSuffix(" days")
        param_layout.addRow("Max Period:", self.max_period_spin)

        self.samples_spin = QSpinBox()
        self.samples_spin.setRange(5, 100)
        self.samples_spin.setValue(10)
        param_layout.addRow("Samples per peak:", self.samples_spin)

        # Method selection
        method_row = QHBoxLayout()
        self.chk_ls = QCheckBox("Lomb-Scargle")
        self.chk_ls.setChecked(True)
        method_row.addWidget(self.chk_ls)
        self.chk_pdm = QCheckBox("PDM")
        self.chk_pdm.setChecked(True)
        method_row.addWidget(self.chk_pdm)
        self.chk_bls = QCheckBox("BLS")
        self.chk_bls.setChecked(False)
        method_row.addWidget(self.chk_bls)
        method_row.addStretch()
        param_layout.addRow("Methods:", method_row)

        self.pdm_bins_spin = QSpinBox()
        self.pdm_bins_spin.setRange(5, 50)
        self.pdm_bins_spin.setValue(10)
        param_layout.addRow("PDM bins:", self.pdm_bins_spin)

        self.content_layout.addWidget(param_group)

        # Run button
        run_layout = QHBoxLayout()
        self.btn_run = QPushButton("Compute Periodogram")
        self.btn_run.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 10px 20px; }"
        )
        self.btn_run.clicked.connect(self._run_analysis)
        self.btn_run.setEnabled(False)
        run_layout.addWidget(self.btn_run)

        self.progress_label = QLabel("")
        run_layout.addWidget(self.progress_label)
        run_layout.addStretch()
        self.content_layout.addLayout(run_layout)

        # Results tabs
        self.tabs = QTabWidget()

        # Periodogram tab
        periodogram_tab = QWidget()
        periodogram_layout = QVBoxLayout(periodogram_tab)

        self.periodogram_canvas = FigureCanvas(Figure(figsize=(10, 5)))
        periodogram_layout.addWidget(self.periodogram_canvas)

        self.tabs.addTab(periodogram_tab, "Periodogram")

        # Phase plot tab
        phase_tab = QWidget()
        phase_layout = QVBoxLayout(phase_tab)

        # Period selection for phase plot
        phase_control = QHBoxLayout()
        phase_control.addWidget(QLabel("Period for phase plot:"))

        self.phase_period_combo = QComboBox()
        self.phase_period_combo.currentIndexChanged.connect(self._update_phase_plot)
        phase_control.addWidget(self.phase_period_combo)

        self.phase_period_edit = QDoubleSpinBox()
        self.phase_period_edit.setRange(0.0001, 10000.0)
        self.phase_period_edit.setDecimals(6)
        self.phase_period_edit.setSuffix(" days")
        self.phase_period_edit.valueChanged.connect(self._update_phase_plot_custom)
        phase_control.addWidget(self.phase_period_edit)

        phase_control.addStretch()
        phase_layout.addLayout(phase_control)

        self.phase_canvas = FigureCanvas(Figure(figsize=(10, 6)))
        phase_layout.addWidget(self.phase_canvas)

        self.tabs.addTab(phase_tab, "Phase Plot")

        # Results table tab
        results_tab = QWidget()
        results_layout = QVBoxLayout(results_tab)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(6)
        self.results_table.setHorizontalHeaderLabels([
            "Data", "Best Period (days)", "Power", "FAP", "N Points", "Top 3 Periods"
        ])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.results_table.horizontalHeader().setStretchLastSection(True)
        results_layout.addWidget(self.results_table)

        self.tabs.addTab(results_tab, "Results")

        self.content_layout.addWidget(self.tabs)

        # Log
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(100)
        self.log_text.setStyleSheet("QTextEdit { font-family: monospace; font-size: 9pt; }")
        self.content_layout.addWidget(self.log_text)

        # Initialize
        self._scan_available_data()

    def _scan_available_data(self):
        """Scan for available light curve data.

        Step 11 outputs: lightcurve_ID{id}_raw.csv, lightcurve_combined_ID{id}_raw.csv
        Step 12 outputs: lightcurve_ID{id}_{mode}.csv (mode=offset/color/global)
        Filters are stored as a column inside the CSV, not in the filename.
        """
        self.filter_combo.clear()

        result_dir = Path(self.params.P.result_dir)
        step11_out = step11_dir(result_dir)
        step12_out = step12_dir(result_dir)

        filters_found = set()
        lc_file = self._find_best_lc_file(step11_out, step12_out, None)

        if lc_file is not None:
            try:
                df = pd.read_csv(lc_file, nrows=5000)
                if "filter" in df.columns:
                    for flt in df["filter"].dropna().astype(str).str.strip().str.lower().unique():
                        if flt and flt != "nan":
                            filters_found.add(flt)
            except Exception:
                pass

        # Fallback: scan all lightcurve CSVs for filter column
        if not filters_found:
            for d in [step12_out, step11_out]:
                if not d.exists():
                    continue
                for f in d.glob("lightcurve_*.csv"):
                    try:
                        df_head = pd.read_csv(f, nrows=200)
                        if "filter" in df_head.columns:
                            for flt in df_head["filter"].dropna().astype(str).str.strip().str.lower().unique():
                                if flt and flt != "nan":
                                    filters_found.add(flt)
                    except Exception:
                        continue
                if filters_found:
                    break

        if filters_found:
            self.filter_combo.addItems(sorted(filters_found))
        else:
            self.filter_combo.addItem("(no data)")

    def _find_best_lc_file(
        self,
        step11_out: Path,
        step12_out: Path,
        target_id: Optional[int],
    ) -> Optional[Path]:
        """Find the best available light curve CSV, preferring Step 12 (detrended)."""
        candidates: List[Path] = []

        if target_id is not None:
            # Step 12 outputs (detrended) - prefer global > color > offset
            for mode in ["global", "color", "offset"]:
                candidates.append(step12_out / f"lightcurve_ID{target_id}_{mode}.csv")
            # Step 11 outputs (raw)
            candidates.append(step11_out / f"lightcurve_combined_ID{target_id}_raw.csv")
            candidates.append(step11_out / f"lightcurve_ID{target_id}_raw.csv")

        # Glob fallbacks (any target ID)
        for d in [step12_out, step11_out]:
            if d.exists():
                for f in sorted(d.glob("lightcurve_*.csv"), reverse=True):
                    if f not in candidates:
                        candidates.append(f)

        for cand in candidates:
            if cand.exists():
                return cand
        return None

    def _on_filter_changed(self, index: int):
        self.current_filter = self.filter_combo.currentText()

    def _load_light_curve(self):
        """Load light curve data for selected target and filter."""
        result_dir = Path(self.params.P.result_dir)
        target_id = self.target_id_spin.value()
        flt = self.filter_combo.currentText()

        if not flt or flt == "(no data)":
            QMessageBox.warning(self, "No Data", "No light curve data available.")
            return

        step11_out = step11_dir(result_dir)
        step12_out = step12_dir(result_dir)

        lc_file = self._find_best_lc_file(step11_out, step12_out, target_id)

        if lc_file is None:
            QMessageBox.warning(
                self, "Not Found",
                f"Could not find light curve data for ID {target_id}.\n"
                "Run Step 11 (Light Curve Builder) first."
            )
            return

        self.log(f"Loading: {lc_file}")

        try:
            df = pd.read_csv(lc_file)
            self.log(f"Loaded {len(df)} rows, columns: {list(df.columns)}")

            # Find time column
            time_col = None
            for col in ["JD", "jd", "HJD", "hjd", "BJD", "bjd", "time", "rel_time_hr"]:
                if col in df.columns:
                    time_col = col
                    break

            if time_col is None:
                QMessageBox.warning(self, "Error", "No time column (JD/HJD/BJD) found.")
                return

            # Filter by filter column
            if "filter" in df.columns:
                df_flt = df[df["filter"].astype(str).str.strip().str.lower() == flt.lower()].copy()
                if df_flt.empty:
                    self.log(f"[WARN] Filter '{flt}' not found in data, using all rows")
                    df_flt = df
            else:
                df_flt = df

            # Filter by target ID if ID column exists
            df_target = df_flt
            if "ID" in df_target.columns:
                df_id = df_target[df_target["ID"] == target_id].copy()
                if df_id.empty:
                    self.log(f"[WARN] ID {target_id} not found, using all data")
                else:
                    df_target = df_id

            # Find magnitude columns
            mag_raw_col = None
            mag_corr_col = None
            mag_err_col = None

            for col in ["diff_mag_raw", "mag_raw", "raw_mag", "inst_mag", "mag"]:
                if col in df_target.columns:
                    mag_raw_col = col
                    break

            for col in ["diff_mag_corr", "diff_mag", "mag_corr", "corr_mag", "calibrated_mag"]:
                if col in df_target.columns:
                    mag_corr_col = col
                    break

            # If no raw, use corr as raw
            if mag_raw_col is None and mag_corr_col is not None:
                mag_raw_col = mag_corr_col
                mag_corr_col = None

            for col in ["diff_err", "diff_err_corr", "mag_err", "err", "sigma", "diff_mag_err", "comp_err"]:
                if col in df_target.columns:
                    mag_err_col = col
                    break

            if mag_raw_col is None:
                QMessageBox.warning(
                    self, "Error",
                    f"No magnitude column found.\n"
                    f"Available columns: {list(df_target.columns)}"
                )
                return

            self.lc_data = {
                "time": df_target[time_col].to_numpy(float),
                "mag_raw": df_target[mag_raw_col].to_numpy(float),
                "mag_corr": df_target[mag_corr_col].to_numpy(float) if mag_corr_col else None,
                "mag_err": df_target[mag_err_col].to_numpy(float) if mag_err_col else None,
                "filter": flt,
                "target_id": target_id,
                "source_file": str(lc_file),
            }

            n_valid = np.sum(np.isfinite(self.lc_data["time"]) & np.isfinite(self.lc_data["mag_raw"]))
            self.data_status.setText(f"Loaded: {n_valid} points ({lc_file.name})")
            self.data_status.setStyleSheet("color: green;")
            self.btn_run.setEnabled(True)

            self.log(f"Time: {time_col}, Raw: {mag_raw_col}, Corr: {mag_corr_col}, Err: {mag_err_col}")
            self.log(f"Filter: {flt}, Target ID: {target_id}, Valid points: {n_valid}")

        except Exception as e:
            QMessageBox.warning(self, "Load Error", str(e))
            self.log(f"[ERROR] {e}")

    def _run_analysis(self):
        """Run period analysis with selected methods."""
        if self.lc_data is None:
            QMessageBox.warning(self, "No Data", "Load light curve data first.")
            return

        if self.worker is not None and self.worker.isRunning():
            return

        min_period = self.min_period_spin.value()
        max_period = self.max_period_spin.value()
        samples = self.samples_spin.value()

        if min_period >= max_period:
            QMessageBox.warning(self, "Invalid Range", "Min period must be less than max period.")
            return

        methods = []
        if self.chk_ls.isChecked():
            methods.append("ls")
        if self.chk_pdm.isChecked():
            methods.append("pdm")
        if self.chk_bls.isChecked():
            methods.append("bls")
        if not methods:
            QMessageBox.warning(self, "No Method", "Select at least one method.")
            return

        self.btn_run.setEnabled(False)
        self.progress_label.setText("Computing...")

        self.worker = PeriodAnalysisWorker(
            time=self.lc_data["time"],
            mag_raw=self.lc_data["mag_raw"],
            mag_corr=self.lc_data["mag_corr"],
            mag_err=self.lc_data["mag_err"],
            min_period=min_period,
            max_period=max_period,
            samples_per_peak=samples,
            methods=methods,
            pdm_n_bins=self.pdm_bins_spin.value(),
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_progress(self, msg: str):
        self.progress_label.setText(msg)
        self.log(msg)

    def _on_error(self, msg: str):
        self.btn_run.setEnabled(True)
        self.progress_label.setText("Error")
        QMessageBox.warning(self, "Error", msg)
        self.log(f"[ERROR] {msg}")

    def _on_finished(self, results: dict):
        self.btn_run.setEnabled(True)
        self.progress_label.setText("Done")
        self.results = results

        self._update_periodogram_plot()
        self._update_results_table()
        self._populate_phase_periods()
        self._update_phase_plot()
        self._save_results()

        self.log("Analysis complete")

    def _update_periodogram_plot(self):
        """Update periodogram plot.

        Results are keyed as '{data}_{method}', e.g. 'raw_ls', 'corr_pdm'.
        Layout: rows = methods, cols = data types (raw / corr).
        """
        fig = self.periodogram_canvas.figure
        fig.clear()

        if not self.results:
            self.periodogram_canvas.draw_idle()
            return

        # Determine grid layout
        method_labels = {"ls": "Lomb-Scargle", "pdm": "PDM (1-\u0398)", "bls": "BLS"}
        data_labels = {"raw": "Raw", "corr": "Corrected"}
        method_colors = {"ls": "#1E88E5", "pdm": "#E53935", "bls": "#FF9800"}
        y_labels = {"ls": "LS Power", "pdm": "1 - \u0398", "bls": "BLS Power"}

        methods_present = []
        data_types_present = []
        for key in self.results:
            parts = key.split("_", 1)
            if len(parts) == 2:
                dt, mt = parts
                if mt not in methods_present:
                    methods_present.append(mt)
                if dt not in data_types_present:
                    data_types_present.append(dt)

        n_rows = len(methods_present) or 1
        n_cols = len(data_types_present) or 1
        axes = fig.subplots(n_rows, n_cols, squeeze=False)

        for ri, method in enumerate(methods_present):
            for ci, dtype in enumerate(data_types_present):
                ax = axes[ri][ci]
                key = f"{dtype}_{method}"
                data = self.results.get(key)
                if data is None or "error" in (data or {}):
                    err_msg = data.get("error", "No data") if data else "No data"
                    ax.text(0.5, 0.5, err_msg, ha="center", va="center",
                            transform=ax.transAxes, fontsize=9)
                    ax.set_title(f"{data_labels.get(dtype, dtype)} / {method_labels.get(method, method)}")
                    continue

                power = data["power"]
                best_period = data["best_period"]
                best_power = data["best_power"]

                # Determine x-axis (periods)
                if "frequency" in data:
                    periods = 1.0 / data["frequency"]
                elif "trial_periods" in data:
                    periods = data["trial_periods"]
                else:
                    ax.text(0.5, 0.5, "No period axis", ha="center", va="center",
                            transform=ax.transAxes)
                    continue

                color = method_colors.get(method, "#666")
                ax.plot(periods, power, color=color, lw=0.8, alpha=0.8)
                ax.axvline(best_period, color="red", ls="--", lw=1.5, alpha=0.8,
                           label=f"P={best_period:.6f}d")
                ax.scatter([best_period], [best_power], color="red", s=50, zorder=5)

                ax.set_xlabel("Period (days)")
                ax.set_ylabel(y_labels.get(method, "Power"))
                ax.set_title(
                    f"{data_labels.get(dtype, dtype)} / {method_labels.get(method, method)}\n"
                    f"P = {best_period:.6f} d"
                )
                ax.set_xscale("log")
                ax.legend(loc="upper right", fontsize=7)
                ax.grid(True, alpha=0.3)

        fig.tight_layout()
        self.periodogram_canvas.draw_idle()

    def _update_results_table(self):
        """Update results summary table."""
        self.results_table.setColumnCount(7)
        self.results_table.setHorizontalHeaderLabels([
            "Method", "Data", "Best Period (days)", "Power", "FAP", "N Points", "Top 3 Periods"
        ])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.setRowCount(0)

        if not self.results:
            return

        method_labels = {"ls": "Lomb-Scargle", "pdm": "PDM", "bls": "BLS"}
        data_labels = {"raw": "Raw", "corr": "Corrected"}

        for key, data in self.results.items():
            row = self.results_table.rowCount()
            self.results_table.insertRow(row)

            parts = key.split("_", 1)
            dtype = parts[0] if len(parts) == 2 else key
            method = parts[1] if len(parts) == 2 else ""

            self.results_table.setItem(row, 0, QTableWidgetItem(method_labels.get(method, method)))
            self.results_table.setItem(row, 1, QTableWidgetItem(data_labels.get(dtype, dtype)))

            if "error" in data:
                self.results_table.setItem(row, 2, QTableWidgetItem(data["error"]))
                continue

            self.results_table.setItem(row, 2, QTableWidgetItem(f"{data['best_period']:.6f}"))
            self.results_table.setItem(row, 3, QTableWidgetItem(f"{data['best_power']:.4f}"))

            fap = data.get("fap", np.nan)
            fap_str = f"{fap:.2e}" if np.isfinite(fap) else "-"
            self.results_table.setItem(row, 4, QTableWidgetItem(fap_str))

            self.results_table.setItem(row, 5, QTableWidgetItem(str(data.get("n_points", 0))))

            top_periods = data.get("top_periods", [])[:3]
            top_str = ", ".join(f"{p:.4f}" for p in top_periods)
            self.results_table.setItem(row, 6, QTableWidgetItem(top_str))

    def _populate_phase_periods(self):
        """Populate phase period combo box with results from all methods."""
        self.phase_period_combo.blockSignals(True)
        self.phase_period_combo.clear()

        method_labels = {"ls": "LS", "pdm": "PDM", "bls": "BLS"}
        data_labels = {"raw": "Raw", "corr": "Corr"}

        periods = []
        seen = set()
        for key, data in self.results.items():
            if "error" in data:
                continue
            parts = key.split("_", 1)
            dtype = parts[0] if len(parts) == 2 else key
            method = parts[1] if len(parts) == 2 else ""

            ml = method_labels.get(method, method.upper())
            dl = data_labels.get(dtype, dtype)
            best_p = data.get("best_period", np.nan)
            if np.isfinite(best_p):
                tag = f"{ml}/{dl}"
                periods.append((f"{tag}: {best_p:.6f} d", best_p))
                # Add x2 and /2 harmonics (de-duplicate)
                p2 = round(best_p * 2, 8)
                ph = round(best_p / 2, 8)
                if p2 not in seen:
                    periods.append((f"{tag} x2: {p2:.6f} d", p2))
                    seen.add(p2)
                if ph not in seen:
                    periods.append((f"{tag} /2: {ph:.6f} d", ph))
                    seen.add(ph)

        for label, p in periods:
            self.phase_period_combo.addItem(label, p)

        if periods:
            self.phase_period_edit.setValue(periods[0][1])

        self.phase_period_combo.blockSignals(False)

    def _update_phase_plot(self, index: int = 0):
        """Update phase folded plot."""
        if self.phase_period_combo.count() == 0:
            return

        period = self.phase_period_combo.currentData()
        if period is None or not np.isfinite(period) or period <= 0:
            return

        self.phase_period_edit.blockSignals(True)
        self.phase_period_edit.setValue(period)
        self.phase_period_edit.blockSignals(False)

        self._draw_phase_plot(period)

    def _update_phase_plot_custom(self):
        """Update phase plot with custom period."""
        period = self.phase_period_edit.value()
        if period <= 0:
            return
        self._draw_phase_plot(period)

    def _draw_phase_plot(self, period: float):
        """Draw phase folded light curve for raw and corrected data."""
        fig = self.phase_canvas.figure
        fig.clear()

        if not self.results:
            self.phase_canvas.draw_idle()
            return

        ax = fig.add_subplot(111)

        colors = {"raw": "#1E88E5", "corr": "#43A047"}
        markers = {"raw": "o", "corr": "s"}
        labels_map = {"raw": "Raw", "corr": "Corrected"}

        # Collect one set of time/mag per data type (raw/corr), from any method
        plotted_dtypes = set()
        for key, data in self.results.items():
            if "error" in data:
                continue
            parts = key.split("_", 1)
            dtype = parts[0] if len(parts) == 2 else key
            if dtype in plotted_dtypes:
                continue
            plotted_dtypes.add(dtype)

            t = data["time"]
            mag = data["mag"]
            mag_err = data.get("mag_err")

            # Compute phase
            t0 = np.nanmin(t)
            phase = ((t - t0) / period) % 1.0

            # Plot two cycles for clarity
            phase_ext = np.concatenate([phase, phase + 1.0])
            mag_ext = np.concatenate([mag, mag])

            color = colors.get(dtype, "#666")
            marker = markers.get(dtype, "o")
            label = labels_map.get(dtype, dtype)

            if mag_err is not None and np.any(np.isfinite(mag_err)):
                err_ext = np.concatenate([mag_err, mag_err])
                ax.errorbar(
                    phase_ext, mag_ext, yerr=err_ext,
                    fmt=marker, color=color, markersize=4,
                    elinewidth=0.5, capsize=0, alpha=0.7,
                    label=label
                )
            else:
                ax.scatter(
                    phase_ext, mag_ext, c=color, marker=marker,
                    s=20, alpha=0.7, label=label
                )

        ax.invert_yaxis()  # Magnitudes are inverted
        ax.set_xlabel("Phase")
        ax.set_ylabel("Magnitude")
        ax.set_title(f"Phase Folded Light Curve (P = {period:.6f} days)")
        ax.set_xlim(0, 2)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        ax.axvline(0, color="gray", ls=":", alpha=0.5)
        ax.axvline(1, color="gray", ls=":", alpha=0.5)

        fig.tight_layout()
        self.phase_canvas.draw_idle()

    def _save_results(self):
        """Save analysis results."""
        if not self.results or self.lc_data is None:
            return

        result_dir = Path(self.params.P.result_dir)
        out_dir = step13_period_dir(result_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        flt = self.lc_data.get("filter", "unknown")
        target_id = self.lc_data.get("target_id", 0)

        # Save summary JSON
        summary = {
            "filter": flt,
            "target_id": target_id,
            "source_file": self.lc_data.get("source_file", ""),
            "min_period": self.min_period_spin.value(),
            "max_period": self.max_period_spin.value(),
            "results": {},
        }

        for key, data in self.results.items():
            if "error" in data:
                summary["results"][key] = {"error": data["error"]}
            else:
                entry = {
                    "method": data.get("method", ""),
                    "best_period": float(data["best_period"]),
                    "best_power": float(data["best_power"]),
                    "fap": float(data.get("fap", np.nan)) if np.isfinite(data.get("fap", np.nan)) else None,
                    "n_points": int(data.get("n_points", 0)),
                    "top_periods": [float(p) for p in data.get("top_periods", [])],
                }
                if "best_theta" in data:
                    entry["best_theta"] = float(data["best_theta"])
                summary["results"][key] = entry

        summary_path = out_dir / f"period_analysis_{flt}_ID{target_id}.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        self.log(f"Saved: {summary_path}")

        # Save periodogram data as CSV
        for key, data in self.results.items():
            if "error" in data:
                continue

            if "frequency" in data:
                df = pd.DataFrame({
                    "frequency": data["frequency"],
                    "period": 1.0 / data["frequency"],
                    "power": data["power"],
                })
            elif "trial_periods" in data:
                df = pd.DataFrame({
                    "period": data["trial_periods"],
                    "power": data["power"],
                })
                if "theta" in data:
                    df["theta"] = data["theta"]
            else:
                continue

            csv_path = out_dir / f"periodogram_{flt}_{key}_ID{target_id}.csv"
            df.to_csv(csv_path, index=False)

    def log(self, msg: str):
        if self.log_text is not None:
            self.log_text.append(msg)

    def validate_step(self) -> bool:
        result_dir = Path(self.params.P.result_dir)
        out_dir = step13_period_dir(result_dir)
        return out_dir.exists() and any(out_dir.glob("period_analysis_*.json"))

    def save_state(self):
        state = {
            "min_period": self.min_period_spin.value(),
            "max_period": self.max_period_spin.value(),
            "samples_per_peak": self.samples_spin.value(),
        }
        self.project_state.store_step_data("period_analysis", state)

    def restore_state(self):
        state = self.project_state.get_step_data("period_analysis")
        if not state:
            return
        if "min_period" in state:
            self.min_period_spin.setValue(float(state["min_period"]))
        if "max_period" in state:
            self.max_period_spin.setValue(float(state["max_period"]))
        if "samples_per_peak" in state:
            self.samples_spin.setValue(int(state["samples_per_peak"]))
