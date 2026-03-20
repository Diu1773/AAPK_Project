"""
Step 12: Period Analysis (Lomb-Scargle / PDM / BLS)

Quick period scan.  Detailed analysis (refine, bootstrap, T₀, O-C,
variable-star / transit / EB fitting) lives in the Tools menu.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QGroupBox,
    QCheckBox,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QTextEdit,
    QWidget,
    QComboBox,
    QFormLayout,
    QDoubleSpinBox,
    QSpinBox,
    QTabWidget,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from .step_window_base import StepWindowBase
from ...analysis.light_curve.period_analysis_service import run_period_analysis
from ...analysis.light_curve.period_io_service import (
    detect_corr_mode_from_df,
    load_period_lightcurve_csv,
    save_period_analysis_outputs,
)
from ...utils.io_utils import read_csv_int64_source_id, coerce_int64_source_id
from ...utils.step_paths import (
    step9_selection_dir,
    step10_dir,
    step12_period_dir,
    find_best_lightcurve_csv,
    load_detrend_preference,
)


def _load_check_star_for_plot(result_dir: Path, filt: str | None = None):
    """Load check star CSV from step10 output for plotting. Returns (check_id, df_or_None)."""
    try:
        from .step10_light_curve_builder import _load_check_star_csv
        check_id, df = _load_check_star_csv(result_dir, filt=filt)
        return check_id, (df if not df.empty else None)
    except Exception:
        return None, None


def _load_step9_source_to_id_map(result_dir: Path, flt: str | None = None) -> dict[int, int]:
    step9_out = step9_selection_dir(result_dir)
    if not step9_out.exists():
        return {}

    key = str(flt or "").strip().lower()
    candidates: list[tuple[Path, str]] = []
    if key:
        candidates.extend(
            [
                (step9_out / f"master_catalog_{key}.tsv", "\t"),
                (step9_out / f"id_mapping_{key}.csv", ","),
            ]
        )
    candidates.extend((p, "\t") for p in sorted(step9_out.glob("master_catalog_*.tsv")))
    candidates.extend((p, ",") for p in sorted(step9_out.glob("id_mapping_*.csv")))

    mapping: dict[int, int] = {}
    for path, sep in candidates:
        if not path.exists():
            continue
        try:
            df = read_csv_int64_source_id(path, sep=sep)
        except Exception:
            continue
        if not {"source_id", "ID"} <= set(df.columns):
            continue
        sid_vals = coerce_int64_source_id(df["source_id"])
        id_vals = pd.to_numeric(df["ID"], errors="coerce").astype("Int64")
        for sid_val, id_val in zip(sid_vals, id_vals):
            if pd.isna(sid_val) or pd.isna(id_val):
                continue
            sid_int = int(sid_val)
            if sid_int not in mapping:
                mapping[sid_int] = int(id_val)
    return mapping


def _load_step9_target_id(result_dir: Path) -> int | None:
    step9_out = step9_selection_dir(result_dir)
    if not step9_out.exists():
        return None

    target_ids: set[int] = set()
    for sp in sorted(step9_out.glob("selection_*.json")):
        try:
            data = json.loads(sp.read_text(encoding="utf-8"))
        except Exception:
            continue
        tid = data.get("target_id")
        if tid is None:
            flt = sp.stem.replace("selection_", "")
            sid_map = _load_step9_source_to_id_map(result_dir, flt)
            target_sid = data.get("target_source_id")
            if target_sid is not None and int(target_sid) in sid_map:
                tid = int(sid_map[int(target_sid)])
        if tid is not None:
            target_ids.add(int(tid))

    if len(target_ids) == 1:
        return next(iter(target_ids))
    return None


def _detect_corr_mode(filename: str) -> tuple[str, str]:
    return detect_corr_mode_from_df(pd.DataFrame(), filename)


def _compute_1day_aliases(period: float) -> list[float]:
    """Compute 1-day sampling aliases of a period.

    Formula: 1/f_alias = 1/f_true ± n/f_samp  (f_samp = 1 day⁻¹, n=1,2)
    Returns list of alias periods (positive, finite only).
    """
    f = 1.0 / period
    aliases = []
    for n in (1, 2):
        for sign in (+1, -1):
            f_alias = f + sign * n
            if f_alias > 0:
                aliases.append(1.0 / f_alias)
    return aliases


def _is_1day_alias(p1: float, p2: float, tol: float = 0.005) -> bool:
    """Check if two periods are related by 1-day sampling alias."""
    for alias_p in _compute_1day_aliases(p1):
        if abs(alias_p - p2) / p2 < tol:
            return True
    return False


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
            results = run_period_analysis(
                time=self.time,
                mag_raw=self.mag_raw,
                mag_corr=self.mag_corr,
                mag_err=self.mag_err,
                min_period=self.min_period,
                max_period=self.max_period,
                samples_per_peak=self.samples_per_peak,
                methods=self.methods,
                pdm_n_bins=self.pdm_n_bins,
                progress_cb=self.progress.emit,
            )
            self.finished.emit(results)
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")


class PeriodAnalysisWindow(StepWindowBase):
    """Step 12: Period Analysis — quick scan with Lomb-Scargle / PDM / BLS."""

    def __init__(self, params, file_manager, project_state, main_window):
        self.file_manager = file_manager
        self.worker = None
        self.results = {}
        self.lc_data = None
        self.current_filter = None
        self._ui_ready = False

        super().__init__(
            step_index=11,
            step_name="Period Analysis",
            params=params,
            project_state=project_state,
            main_window=main_window,
        )

        self.setup_step_ui()
        self.restore_state()
        self._auto_load_target_id()
        self._ui_ready = True
        self._load_light_curve(silent=True)

    def _auto_load_target_id(self):
        """Step 10 → Step 9 merged selection 순서로 target ID 자동 로드."""
        rd = Path(self.params.P.result_dir)
        sel_path = step10_dir(rd) / "comp_selection.json"
        if sel_path.exists():
            try:
                data = json.loads(sel_path.read_text(encoding="utf-8"))
                target_id = data.get("target_id")
                if target_id is not None:
                    self.target_id_spin.setValue(int(target_id))
                    self.target_hint.setText("(Step 10)")
                    return
            except Exception:
                pass
        tid = _load_step9_target_id(rd)
        if tid is not None:
            self.target_id_spin.setValue(int(tid))
            self.target_hint.setText("(Step 9)")
            return
        self.target_hint.setText("")

    def setup_step_ui(self):
        info = QLabel(
            "Period analysis: Lomb-Scargle, PDM, BLS.\n"
            "Multiple methods reduce aliasing; consensus = true period.\n"
            "For detailed analysis (refine, bootstrap, O-C, transit fit) → Tools menu."
        )
        info.setStyleSheet("QLabel { background-color: #E3F2FD; padding: 10px; border-radius: 5px; }")
        info.setWordWrap(True)
        self.content_layout.addWidget(info)

        # Data selection
        data_group = QGroupBox("Data Selection")
        data_layout = QFormLayout(data_group)

        target_row = QHBoxLayout()
        self.target_id_spin = QSpinBox()
        self.target_id_spin.setRange(1, 99999)
        self.target_id_spin.setValue(1)
        self.target_id_spin.valueChanged.connect(self._on_target_id_changed)
        target_row.addWidget(self.target_id_spin)
        self.target_hint = QLabel("")
        self.target_hint.setStyleSheet("QLabel { color: #388E3C; font-size: 8pt; }")
        target_row.addWidget(self.target_hint)
        target_row.addStretch()
        data_layout.addRow("Target ID:", target_row)

        self.source_label = QLabel("—")
        self.source_label.setStyleSheet("QLabel { font-family: monospace; font-size: 9pt; }")
        data_layout.addRow("Data source:", self.source_label)

        self.filter_combo = QComboBox()
        self.filter_combo.currentIndexChanged.connect(self._on_filter_changed)
        data_layout.addRow("Filter:", self.filter_combo)

        self.data_status = QLabel("Loading light curve data...")
        self.data_status.setWordWrap(True)
        data_layout.addRow("Status:", self.data_status)

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

        alias_row = QHBoxLayout()
        self.chk_alias = QCheckBox("Show 1-day aliases of best period")
        self.chk_alias.setChecked(True)
        self.chk_alias.toggled.connect(self._update_periodogram_plot)
        alias_row.addWidget(self.chk_alias)
        alias_row.addStretch()
        periodogram_layout.addLayout(alias_row)

        self.periodogram_canvas = FigureCanvas(Figure(figsize=(10, 5)))
        periodogram_layout.addWidget(self.periodogram_canvas)

        self.tabs.addTab(periodogram_tab, "Periodogram")

        # Phase plot tab
        phase_tab = QWidget()
        phase_layout = QVBoxLayout(phase_tab)

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
        self.results_table.setColumnCount(7)
        self.results_table.setHorizontalHeaderLabels([
            "Method", "Data", "Best Period (days)", "Power", "FAP", "Alias?", "Top 3 Periods"
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

        self._scan_available_data()

    # ------------------------------------------------------------------
    # Data scanning / loading
    # ------------------------------------------------------------------

    def _scan_available_data(self):
        """Auto-select best lightcurve and populate filter combo."""
        result_dir = Path(self.params.P.result_dir)
        target_id = self.target_id_spin.value()

        lc_path = find_best_lightcurve_csv(result_dir, target_id)
        self._auto_lc_path = lc_path

        if lc_path and lc_path.exists():
            mode_key, mode_label = _detect_corr_mode(lc_path.name)
            pref = load_detrend_preference(result_dir, target_id)
            if pref:
                mode_label = _CORR_MODE_LABELS.get(pref, mode_label)
            self.source_label.setText(mode_label)

            # Scan filters
            filters_found: set[str] = set()
            try:
                df_head = pd.read_csv(lc_path, nrows=500)
                if "filter" in df_head.columns:
                    for flt in df_head["filter"].dropna().astype(str).str.strip().str.lower().unique():
                        if flt and flt != "nan":
                            filters_found.add(flt)
            except Exception:
                pass
        else:
            self.source_label.setText("No data")
            filters_found = set()

        self.filter_combo.blockSignals(True)
        self.filter_combo.clear()
        if filters_found:
            self.filter_combo.addItems(sorted(filters_found))
        else:
            self.filter_combo.addItem("(no data)")
        self.filter_combo.blockSignals(False)
        self.current_filter = self.filter_combo.currentText()

    def _on_filter_changed(self, index: int):
        self.current_filter = self.filter_combo.currentText()
        if self._ui_ready:
            self._load_light_curve(silent=True)

    def _on_target_id_changed(self, value: int):
        if self._ui_ready:
            # Re-scan sources for this target ID
            self._scan_available_data()
            self._load_light_curve(silent=True)

    def _set_data_status(self, text: str, ok: bool = False):
        color = "#1B5E20" if ok else "#B71C1C"
        self.data_status.setText(text)
        self.data_status.setStyleSheet(
            f"color: {color}; font-family: monospace; font-size: 8pt;"
        )

    def _clear_analysis_results(self):
        self.results = {}
        self.results_table.setRowCount(0)
        self.phase_period_combo.blockSignals(True)
        self.phase_period_combo.clear()
        self.phase_period_combo.blockSignals(False)
        self.periodogram_canvas.figure.clear()
        self.periodogram_canvas.draw_idle()
        self.phase_canvas.figure.clear()
        self.phase_canvas.draw_idle()
        self.progress_label.setText("")

    def _load_light_curve(self, silent: bool = False):
        target_id = self.target_id_spin.value()
        flt = self.filter_combo.currentText()

        if not flt or flt == "(no data)":
            self.lc_data = None
            self.btn_run.setEnabled(False)
            self._clear_analysis_results()
            self._set_data_status("No light curve data available.", ok=False)
            if not silent:
                QMessageBox.warning(self, "No Data", "No light curve data available.")
            return

        lc_file = getattr(self, "_auto_lc_path", None)
        if not lc_file or not lc_file.exists():
            self.lc_data = None
            self.btn_run.setEnabled(False)
            self._clear_analysis_results()
            self._set_data_status("No data source found.", ok=False)
            return

        self.log(f"Loading: {lc_file}")

        try:
            self.lc_data = load_period_lightcurve_csv(lc_file, flt, target_id)
            self.current_filter = flt
            self._clear_analysis_results()
            n_valid = np.sum(np.isfinite(self.lc_data["time"]) & np.isfinite(self.lc_data["mag_raw"]))
            corr_info = f"corr={self.lc_data.get('col_corr')}" if self.lc_data.get("col_corr") else "corr=없음"
            err_info = f"err={self.lc_data.get('col_err')}" if self.lc_data.get("col_err") else "err=없음"
            corr_mode_label = self.lc_data.get("corr_mode_label", "Unknown")
            self._set_data_status(
                f"{n_valid}점  [{lc_file.name}]\n"
                f"Detrend: {corr_mode_label}  |  raw={self.lc_data.get('col_raw')}  {corr_info}\n"
                f"{err_info}  time={self.lc_data.get('col_time')}",
                ok=True,
            )
            self.btn_run.setEnabled(True)

            self.log(f"Loaded {self.lc_data.get('n_rows', 0)} rows, columns: {self.lc_data.get('columns', [])}")
            self.log(
                f"Time: {self.lc_data.get('col_time')}, Raw: {self.lc_data.get('col_raw')}, "
                f"Corr: {self.lc_data.get('col_corr')}, Err: {self.lc_data.get('col_err')}"
            )
            self.log(f"Filter: {flt}, Target ID: {target_id}, Valid points: {n_valid}, Detrend: {corr_mode_label}")

        except Exception as e:
            self.lc_data = None
            self.btn_run.setEnabled(False)
            self._clear_analysis_results()
            self._set_data_status(f"Load error: {e}", ok=False)
            if not silent:
                QMessageBox.warning(self, "Load Error", str(e))
            self.log(f"[ERROR] {e}")

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def _run_analysis(self):
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
        self._log_alias_warnings()
        self.log("Analysis complete")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def _update_periodogram_plot(self):
        fig = self.periodogram_canvas.figure
        fig.clear()

        if not self.results:
            self.periodogram_canvas.draw_idle()
            return

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

                # 1/day alias lines of the best period
                if hasattr(self, "chk_alias") and self.chk_alias.isChecked():
                    p_min = self.min_period_spin.value()
                    p_max = self.max_period_spin.value()
                    for k, ap in enumerate(_compute_1day_aliases(best_period)):
                        if p_min <= ap <= p_max:
                            ax.axvline(ap, color="orange", ls="--", lw=1.2, alpha=0.7,
                                       label=f"1d-alias {ap:.4f}d" if k == 0 else f"{ap:.4f}d")

                ax.legend(loc="upper right", fontsize=7)
                ax.grid(True, alpha=0.3)

        fig.tight_layout()
        self.periodogram_canvas.draw_idle()

    def _update_results_table(self):
        self.results_table.setRowCount(0)

        if not self.results:
            return

        method_labels = {"ls": "Lomb-Scargle", "pdm": "PDM", "bls": "BLS"}
        data_labels = {"raw": "Raw", "corr": "Corrected"}

        # Collect all best periods per data type for cross-method alias detection
        best_periods_by_dtype: dict[str, dict[str, float]] = {}
        for key, data in self.results.items():
            if "error" in data:
                continue
            parts = key.split("_", 1)
            if len(parts) == 2:
                dt, mt = parts
                best_periods_by_dtype.setdefault(dt, {})[mt] = data["best_period"]

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

            # Alias detection: check if this method's best period is a 1-day alias
            # of another method's best period (same data type)
            alias_tag = ""
            bp = data["best_period"]
            others = best_periods_by_dtype.get(dtype, {})
            for other_method, other_p in others.items():
                if other_method == method:
                    continue
                if _is_1day_alias(bp, other_p):
                    oml = method_labels.get(other_method, other_method.upper())
                    alias_tag = f"1d-alias of {oml} ({other_p:.4f}d)"
                    break
            item_alias = QTableWidgetItem(alias_tag)
            if alias_tag:
                item_alias.setForeground(Qt.red)
            self.results_table.setItem(row, 5, item_alias)

            top_periods = data.get("top_periods", [])[:3]
            top_str = ", ".join(f"{p:.4f}" for p in top_periods)
            self.results_table.setItem(row, 6, QTableWidgetItem(top_str))

    def _populate_phase_periods(self):
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
        period = self.phase_period_edit.value()
        if period <= 0:
            return
        self._draw_phase_plot(period)

    def _draw_phase_plot(self, period: float):
        fig = self.phase_canvas.figure
        fig.clear()

        if not self.results:
            self.phase_canvas.draw_idle()
            return

        ax = fig.add_subplot(111)

        colors = {"raw": "#1E88E5", "corr": "#43A047"}
        markers = {"raw": "o", "corr": "s"}
        col_raw = self.lc_data.get("col_raw", "") if self.lc_data else ""
        col_corr = self.lc_data.get("col_corr", "") if self.lc_data else ""
        labels_map = {
            "raw": f"Raw ({col_raw})" if col_raw else "Raw",
            "corr": f"Corrected ({col_corr})" if col_corr else "Corrected",
        }

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

            t0 = np.nanmin(t)
            phase = ((t - t0) / period) % 1.0
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

        ax.invert_yaxis()
        ax.set_xlabel("Phase")
        ax.set_ylabel("Magnitude")
        src_name = Path(self.lc_data.get("source_file", "")).name if self.lc_data else ""
        ax.set_title(f"Phase Folded Light Curve  P = {period:.6f} d\n{src_name}", fontsize=9)
        ax.set_xlim(0, 2)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.axvline(0, color="gray", ls=":", alpha=0.5)
        ax.axvline(1, color="gray", ls=":", alpha=0.5)

        # Check star phase-folded overlay
        try:
            result_dir = Path(self.params.P.result_dir)
            _flt = self.lc_data.get("filter", "") if self.lc_data else ""
            _ck_id, _ck_df = _load_check_star_for_plot(result_dir, _flt)
            if _ck_df is not None and not _ck_df.empty:
                _t_col = next((c for c in ["BJD_TDB", "BJD", "bjd", "JD", "hjd", "time"] if c in _ck_df.columns), None)
                _y_col = next((c for c in ["diff_mag_raw", "diff_mag", "mag"] if c in _ck_df.columns), None)
                if _t_col and _y_col:
                    if _flt and "filter" in _ck_df.columns:
                        _ck_df = _ck_df[_ck_df["filter"].astype(str) == _flt]
                    _ct = pd.to_numeric(_ck_df[_t_col], errors="coerce").to_numpy(float)
                    _cm = pd.to_numeric(_ck_df[_y_col], errors="coerce").to_numpy(float)
                    _mask = np.isfinite(_ct) & np.isfinite(_cm)
                    if _mask.any():
                        _ck_label = f"Check ID {_ck_id}" if _ck_id is not None else "Check"
                        _t0 = np.nanmin(_ct[_mask])
                        _phase = ((_ct[_mask] - _t0) / period) % 1.0
                        _phase_ext = np.concatenate([_phase, _phase + 1.0])
                        _mag_ext = np.concatenate([_cm[_mask], _cm[_mask]])
                        ax.scatter(_phase_ext, _mag_ext, s=8, color="#FFD700", alpha=0.4,
                                   zorder=2, label=_ck_label, marker="^")
                        ax.legend(loc="upper right", fontsize=8)
        except Exception:
            pass

        fig.tight_layout()
        self.phase_canvas.draw_idle()

    # ------------------------------------------------------------------
    # Save / validate
    # ------------------------------------------------------------------

    def _log_alias_warnings(self):
        """Log warnings when different methods disagree due to 1-day aliases."""
        if not self.results:
            return
        # Group best periods by data type
        by_dtype: dict[str, dict[str, float]] = {}
        for key, data in self.results.items():
            if "error" in data:
                continue
            parts = key.split("_", 1)
            if len(parts) == 2:
                dt, mt = parts
                by_dtype.setdefault(dt, {})[mt] = data["best_period"]

        method_labels = {"ls": "Lomb-Scargle", "pdm": "PDM", "bls": "BLS"}
        for dtype, methods in by_dtype.items():
            if len(methods) < 2:
                continue
            keys = list(methods.keys())
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    m1, m2 = keys[i], keys[j]
                    p1, p2 = methods[m1], methods[m2]
                    if abs(p1 - p2) / max(p1, p2) < 0.005:
                        continue  # same period, no alias issue
                    if _is_1day_alias(p1, p2):
                        ml1 = method_labels.get(m1, m1.upper())
                        ml2 = method_labels.get(m2, m2.upper())
                        self.log(
                            f"[ALIAS WARNING] {dtype}: {ml1}={p1:.6f}d ↔ "
                            f"{ml2}={p2:.6f}d are 1-day aliases! "
                            f"PDM is generally more reliable for ground-based data."
                        )

    def _save_results(self):
        if not self.results or self.lc_data is None:
            return

        result_dir = Path(self.params.P.result_dir)
        out_dir = step12_period_dir(result_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        flt = self.lc_data.get("filter", "unknown")
        target_id = self.lc_data.get("target_id", 0)

        summary_path = save_period_analysis_outputs(
            result_dir=result_dir,
            lc_data=self.lc_data,
            results=self.results,
            min_period=self.min_period_spin.value(),
            max_period=self.max_period_spin.value(),
        )
        self.log(f"Saved: {summary_path}")

    def log(self, msg: str):
        if self.log_text is not None:
            self.log_text.append(msg)

    def validate_step(self) -> bool:
        result_dir = Path(self.params.P.result_dir)
        out_dir = step12_period_dir(result_dir)
        return out_dir.exists() and any(out_dir.glob("period_analysis_*.json"))

    def save_state(self):
        state = {
            "min_period": self.min_period_spin.value(),
            "max_period": self.max_period_spin.value(),
            "samples_per_peak": self.samples_spin.value(),
            "pdm_bins": self.pdm_bins_spin.value(),
            "use_ls": self.chk_ls.isChecked(),
            "use_pdm": self.chk_pdm.isChecked(),
            "use_bls": self.chk_bls.isChecked(),
            "show_alias": self.chk_alias.isChecked(),
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
        if "pdm_bins" in state:
            self.pdm_bins_spin.setValue(int(state["pdm_bins"]))
        if "use_ls" in state:
            self.chk_ls.setChecked(bool(state["use_ls"]))
        if "use_pdm" in state:
            self.chk_pdm.setChecked(bool(state["use_pdm"]))
        if "use_bls" in state:
            self.chk_bls.setChecked(bool(state["use_bls"]))
        if "show_alias" in state:
            self.chk_alias.setChecked(bool(state["show_alias"]))
