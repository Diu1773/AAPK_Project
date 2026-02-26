from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from astropy.wcs.utils import proj_plane_pixel_scales

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backend_bases import MouseButton
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from matplotlib.patches import Ellipse

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QMessageBox,
    QTabWidget,
    QGroupBox,
    QFormLayout,
    QComboBox,
    QDoubleSpinBox,
    QSpinBox,
    QCheckBox,
    QSlider,
    QPlainTextEdit,
)

from .io import ResolvedInputs, load_ref_fits_with_wcs, load_star_table, resolve_inputs
from . import analysis as an


class ClusterStructureWindow(QWidget):
    """Standalone tool for cluster structure analysis."""

    def __init__(self, params, result_dir: Optional[Path] = None, parent=None):
        super().__init__(parent)
        self.params = params
        self.result_dir = Path(result_dir or getattr(params.P, "result_dir", Path.cwd()))

        self.inputs: Optional[ResolvedInputs] = None
        self.star_meta: dict = {}

        self.df_stars = pd.DataFrame()
        self.ref_data = None
        self.ref_header = None
        self.ref_wcs = None
        self.ref_shape = (0, 0)
        self.pixel_scale_arcsec = np.nan
        self.ref_bounds = None  # (x0, x1, y0, y1)

        self.center_summary: Optional[an.CenterSummary] = None
        self.center_manual: Optional[tuple[float, float]] = None
        self.selected_center: Optional[tuple[float, float, str]] = None
        self.footprint_local_xy = None
        self.footprint_pix_xy = None

        self.df_all = pd.DataFrame()
        self.df_bright = pd.DataFrame()
        self.area_df = pd.DataFrame()
        self.profile_all = pd.DataFrame()
        self.profile_bright = pd.DataFrame()
        self.fit_all: dict = {}
        self.fit_bright: dict = {}
        self.fit_profile_all = pd.DataFrame()
        self.fit_profile_bright = pd.DataFrame()
        self.fit_filter_min_f_area = 0.60
        self.fit_filter_info: dict = {}
        self.completeness_info: dict = {"available": False}
        self.struct_metrics: dict = {}
        self.shape_info: dict = {}
        self.pmem_curve_df = pd.DataFrame()

        tname = str(getattr(getattr(params, "P", None), "target_name", "") or "").strip()
        if tname:
            self.setWindowTitle(f"{tname} - Analyze Cluster Structure")
        else:
            self.setWindowTitle("Analyze Cluster Structure")
        self.setMinimumSize(1280, 860)

        self._build_ui()
        self._load_inputs()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs, 1)

        self.tab_overview = QWidget()
        self.tab_annuli = QWidget()
        self.tab_profile = QWidget()
        self.tabs.addTab(self.tab_overview, "1) Centering")
        self.tabs.addTab(self.tab_annuli, "2) Annuli & Footprint")
        self.tabs.addTab(self.tab_profile, "3) Profile & Fit")

        self._build_tab_overview()
        self._build_tab_annuli()
        self._build_tab_profile()

        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(120)
        log_layout.addWidget(self.log_text)
        root.addWidget(log_group)

    def _build_tab_overview(self):
        layout = QVBoxLayout(self.tab_overview)

        path_group = QGroupBox("Resolved Inputs")
        path_form = QFormLayout(path_group)
        self.lbl_star_table = QLabel("-")
        self.lbl_ref_fits = QLabel("-")
        self.lbl_footprint_mode = QLabel("ref")
        for lbl in (self.lbl_star_table, self.lbl_ref_fits):
            lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        path_form.addRow("star_table_path:", self.lbl_star_table)
        path_form.addRow("ref_fits_path:", self.lbl_ref_fits)
        path_form.addRow("footprint_mode:", self.lbl_footprint_mode)
        layout.addWidget(path_group)

        ctrl_group = QGroupBox("Centering Controls")
        ctrl = QFormLayout(ctrl_group)

        pmem_row = QWidget()
        pmem_row_l = QHBoxLayout(pmem_row)
        pmem_row_l.setContentsMargins(0, 0, 0, 0)
        pmem_row_l.setSpacing(8)
        self.pmem_slider = QSlider(Qt.Horizontal)
        self.pmem_slider.setRange(0, 100)
        self.pmem_slider.setValue(50)
        self.pmem_slider.valueChanged.connect(self._on_pmem_slider_changed)
        self.lbl_pmem_value = QLabel("0.50")
        pmem_row_l.addWidget(self.pmem_slider, 1)
        pmem_row_l.addWidget(self.lbl_pmem_value)
        ctrl.addRow("pmem_min:", pmem_row)

        ruwe_row = QWidget()
        ruwe_row_l = QHBoxLayout(ruwe_row)
        ruwe_row_l.setContentsMargins(0, 0, 0, 0)
        ruwe_row_l.setSpacing(8)
        self.chk_ruwe = QCheckBox("Use RUWE cut")
        self.spin_ruwe = QDoubleSpinBox()
        self.spin_ruwe.setRange(0.5, 10.0)
        self.spin_ruwe.setSingleStep(0.1)
        self.spin_ruwe.setValue(2.0)
        self.spin_ruwe.setEnabled(False)
        self.chk_ruwe.toggled.connect(self.spin_ruwe.setEnabled)
        ruwe_row_l.addWidget(self.chk_ruwe)
        ruwe_row_l.addWidget(self.spin_ruwe)
        ctrl.addRow("RUWE:", ruwe_row)

        self.spin_iter_radius = QDoubleSpinBox()
        self.spin_iter_radius.setRange(10.0, 20000.0)
        self.spin_iter_radius.setSingleStep(10.0)
        self.spin_iter_radius.setValue(1200.0)
        ctrl.addRow("Iterative radius (arcsec):", self.spin_iter_radius)

        self.combo_center_choice = QComboBox()
        self.combo_center_choice.addItem("AUTO", "auto")
        self.combo_center_choice.addItem("Iterative Mean", "iter")
        self.combo_center_choice.addItem("KDE Peak", "kde")
        self.combo_center_choice.addItem("Manual (picked)", "manual")
        ctrl.addRow("Center mode:", self.combo_center_choice)

        manual_row = QWidget()
        manual_row_l = QHBoxLayout(manual_row)
        manual_row_l.setContentsMargins(0, 0, 0, 0)
        manual_row_l.setSpacing(8)
        self.chk_manual_pick = QCheckBox("Enable manual pick")
        self.chk_manual_pick.setToolTip("Use double-left-click on image to set manual center candidate.")
        self.btn_manual_clear = QPushButton("Clear Manual")
        self.btn_manual_clear.clicked.connect(self._clear_manual_center)
        manual_row_l.addWidget(self.chk_manual_pick)
        manual_row_l.addWidget(self.btn_manual_clear)
        ctrl.addRow("Manual center:", manual_row)

        row_btn = QWidget()
        row_btn_l = QHBoxLayout(row_btn)
        row_btn_l.setContentsMargins(0, 0, 0, 0)
        row_btn_l.setSpacing(8)
        self.btn_center_recompute = QPushButton("Run Centering")
        self.btn_center_recompute.clicked.connect(self._recompute_center_only)
        row_btn_l.addWidget(self.btn_center_recompute)
        ctrl.addRow("Actions:", row_btn)

        layout.addWidget(ctrl_group)

        self.fig_center = Figure(figsize=(10, 5))
        self.canvas_center = FigureCanvas(self.fig_center)
        self.canvas_center.mpl_connect("button_press_event", self._on_center_plot_click)
        self.toolbar_center = NavigationToolbar(self.canvas_center, self.tab_overview)
        layout.addWidget(self.toolbar_center)
        layout.addWidget(self.canvas_center, 1)

        self.text_center = QPlainTextEdit()
        self.text_center.setReadOnly(True)
        self.text_center.setMaximumHeight(110)
        layout.addWidget(self.text_center)

    def _build_tab_annuli(self):
        layout = QVBoxLayout(self.tab_annuli)

        ctrl_group = QGroupBox("Annuli & Area Correction")
        ctrl = QFormLayout(ctrl_group)

        self.combo_bin_mode = QComboBox()
        self.combo_bin_mode.addItem("Fixed-width", "fixed")
        self.combo_bin_mode.addItem("Equal-count (weighted)", "equal")
        ctrl.addRow("Binning mode:", self.combo_bin_mode)

        self.spin_fixed_width = QDoubleSpinBox()
        self.spin_fixed_width.setRange(1.0, 5000.0)
        self.spin_fixed_width.setSingleStep(5.0)
        self.spin_fixed_width.setValue(30.0)
        ctrl.addRow("delta_r (arcsec):", self.spin_fixed_width)

        self.spin_equal_bins = QSpinBox()
        self.spin_equal_bins.setRange(3, 200)
        self.spin_equal_bins.setValue(18)
        ctrl.addRow("Equal-count bins:", self.spin_equal_bins)

        self.spin_rmin = QDoubleSpinBox()
        self.spin_rmin.setRange(0.0, 10000.0)
        self.spin_rmin.setSingleStep(5.0)
        self.spin_rmin.setValue(0.0)
        ctrl.addRow("r_min (arcsec):", self.spin_rmin)

        rmax_row = QWidget()
        rmax_row_l = QHBoxLayout(rmax_row)
        rmax_row_l.setContentsMargins(0, 0, 0, 0)
        rmax_row_l.setSpacing(8)
        self.chk_rmax_auto = QCheckBox("Auto")
        self.chk_rmax_auto.setChecked(True)
        self.spin_rmax = QDoubleSpinBox()
        self.spin_rmax.setRange(10.0, 200000.0)
        self.spin_rmax.setSingleStep(10.0)
        self.spin_rmax.setValue(1800.0)
        self.spin_rmax.setEnabled(False)
        self.chk_rmax_auto.toggled.connect(lambda v: self.spin_rmax.setEnabled(not v))
        rmax_row_l.addWidget(self.chk_rmax_auto)
        rmax_row_l.addWidget(self.spin_rmax)
        ctrl.addRow("r_max:", rmax_row)

        self.spin_mc_samples = QSpinBox()
        self.spin_mc_samples.setRange(500, 200000)
        self.spin_mc_samples.setSingleStep(500)
        self.spin_mc_samples.setValue(6000)
        ctrl.addRow("MC samples per bin:", self.spin_mc_samples)

        fp_row = QWidget()
        fp_row_l = QHBoxLayout(fp_row)
        fp_row_l.setContentsMargins(0, 0, 0, 0)
        fp_row_l.setSpacing(8)
        self.combo_footprint = QComboBox()
        self.combo_footprint.addItem("Reference footprint", "ref")
        self.combo_footprint.addItem("Intersection footprint (TODO)", "intersection")
        fp_row_l.addWidget(self.combo_footprint)
        ctrl.addRow("Footprint mode:", fp_row)

        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 2_000_000_000)
        self.spin_seed.setValue(42)
        ctrl.addRow("Random seed:", self.spin_seed)

        self.btn_annuli_recompute = QPushButton("Recompute Annuli")
        self.btn_annuli_recompute.clicked.connect(self._recompute_annuli_only)
        ctrl.addRow("Action:", self.btn_annuli_recompute)

        layout.addWidget(ctrl_group)

        self.fig_annuli = Figure(figsize=(11, 5.5))
        self.canvas_annuli = FigureCanvas(self.fig_annuli)
        layout.addWidget(NavigationToolbar(self.canvas_annuli, self.tab_annuli))
        layout.addWidget(self.canvas_annuli, 1)

        self.text_annuli = QPlainTextEdit()
        self.text_annuli.setReadOnly(True)
        self.text_annuli.setMaximumHeight(110)
        layout.addWidget(self.text_annuli)

    def _build_tab_profile(self):
        layout = QVBoxLayout(self.tab_profile)

        ctrl_group = QGroupBox("Profile & Model Fit")
        ctrl = QFormLayout(ctrl_group)

        self.combo_model = QComboBox()
        self.combo_model.addItem("EFF", "eff")
        self.combo_model.addItem("King", "king")
        self.combo_model.addItem("EFF + King", "both")
        ctrl.addRow("Model:", self.combo_model)

        self.combo_bg = QComboBox()
        self.combo_bg.addItem("Outer-bins median", "outer")
        self.combo_bg.addItem("Joint fit Sigma_bg", "joint")
        ctrl.addRow("Background:", self.combo_bg)

        bright_row = QWidget()
        bright_row_l = QHBoxLayout(bright_row)
        bright_row_l.setContentsMargins(0, 0, 0, 0)
        bright_row_l.setSpacing(8)
        self.chk_bright = QCheckBox("Enable BRIGHT subset")
        self.spin_mag_cut = QDoubleSpinBox()
        self.spin_mag_cut.setRange(-10.0, 40.0)
        self.spin_mag_cut.setDecimals(3)
        self.spin_mag_cut.setSingleStep(0.1)
        self.spin_mag_cut.setValue(17.0)
        bright_row_l.addWidget(self.chk_bright)
        bright_row_l.addWidget(QLabel("mag_cut <="))
        bright_row_l.addWidget(self.spin_mag_cut)
        ctrl.addRow("BRIGHT:", bright_row)

        bs_row = QWidget()
        bs_row_l = QHBoxLayout(bs_row)
        bs_row_l.setContentsMargins(0, 0, 0, 0)
        bs_row_l.setSpacing(8)
        self.chk_bootstrap = QCheckBox("Bootstrap")
        self.spin_bootstrap = QSpinBox()
        self.spin_bootstrap.setRange(20, 2000)
        self.spin_bootstrap.setSingleStep(20)
        self.spin_bootstrap.setValue(200)
        self.spin_bootstrap.setEnabled(False)
        self.chk_bootstrap.toggled.connect(self.spin_bootstrap.setEnabled)
        bs_row_l.addWidget(self.chk_bootstrap)
        bs_row_l.addWidget(self.spin_bootstrap)
        ctrl.addRow("Uncertainty:", bs_row)

        self.spin_fit_farea_min = QDoubleSpinBox()
        self.spin_fit_farea_min.setRange(0.0, 1.0)
        self.spin_fit_farea_min.setSingleStep(0.05)
        self.spin_fit_farea_min.setDecimals(2)
        self.spin_fit_farea_min.setValue(float(self.fit_filter_min_f_area))
        ctrl.addRow("Fit min f_area:", self.spin_fit_farea_min)

        run_row = QWidget()
        run_row_l = QHBoxLayout(run_row)
        run_row_l.setContentsMargins(0, 0, 0, 0)
        run_row_l.setSpacing(8)
        self.btn_fit = QPushButton("Run Profile/Fit")
        self.btn_fit.clicked.connect(self._run_fit_only)
        self.btn_save_results = QPushButton("Save Results")
        self.btn_save_results.clicked.connect(lambda: self._run_pipeline(preview=False, save_outputs=True))
        run_row_l.addWidget(self.btn_fit)
        run_row_l.addWidget(self.btn_save_results)
        ctrl.addRow("Action:", run_row)

        layout.addWidget(ctrl_group)

        self.fig_profile = Figure(figsize=(10.5, 6.5))
        self.canvas_profile = FigureCanvas(self.fig_profile)
        layout.addWidget(NavigationToolbar(self.canvas_profile, self.tab_profile))
        layout.addWidget(self.canvas_profile, 1)

        self.text_profile = QPlainTextEdit()
        self.text_profile.setReadOnly(True)
        self.text_profile.setMaximumHeight(135)
        layout.addWidget(self.text_profile)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self.log_text.appendPlainText(f"[{ts}] {msg}")

    def _on_pmem_slider_changed(self, value: int):
        self.lbl_pmem_value.setText(f"{value / 100.0:.2f}")

    def _pmem_min(self) -> float:
        return float(self.pmem_slider.value()) / 100.0

    def _footprint_mode(self) -> str:
        mode = str(self.combo_footprint.currentData() or "ref")
        if mode == "intersection":
            self._log("Footprint intersection is TODO. Fallback to reference footprint.")
            return "ref"
        return "ref"

    def _load_inputs(self):
        try:
            self.inputs = resolve_inputs(self.params, self.result_dir)
            self.df_stars, self.star_meta = load_star_table(self.result_dir, self.inputs.star_table_path)
            self.ref_data, self.ref_header, self.ref_wcs = load_ref_fits_with_wcs(self.inputs.ref_fits_path)
            self.ref_shape = tuple(np.asarray(self.ref_data).shape[:2])

            try:
                sc = proj_plane_pixel_scales(self.ref_wcs.celestial) * 3600.0
                self.pixel_scale_arcsec = float(np.nanmean(sc))
            except Exception:
                self.pixel_scale_arcsec = float(getattr(self.params.P, "pixel_scale_arcsec", np.nan))

            ra = self.df_stars["ra"].to_numpy(float)
            dec = self.df_stars["dec"].to_numpy(float)
            xpix, ypix = self.ref_wcs.celestial.world_to_pixel_values(ra, dec)
            self.df_stars["x_pix"] = np.asarray(xpix, float)
            self.df_stars["y_pix"] = np.asarray(ypix, float)

            has_mag = bool(self.star_meta.get("mag_col")) and np.isfinite(self.df_stars["mag"].to_numpy(float)).sum() > 5
            self.chk_bright.setEnabled(has_mag)
            if has_mag:
                mags = pd.to_numeric(self.df_stars["mag"], errors="coerce").to_numpy(float)
                if "mag_err" in self.df_stars.columns:
                    mag_err = pd.to_numeric(self.df_stars["mag_err"], errors="coerce").to_numpy(float)
                else:
                    mag_err = np.full(len(self.df_stars), np.nan, dtype=float)
                self.completeness_info = self._estimate_completeness_info(mags, mag_err)
                rec_cut = float(self.completeness_info.get("recommended_mag_cut", np.nan))
                if np.isfinite(rec_cut):
                    self.spin_mag_cut.setValue(rec_cut)
                    self._log(
                        f"Completeness rec: mag_cut<={rec_cut:.3f} "
                        f"(hist_drop={self.completeness_info.get('hist_drop_mag', np.nan):.3f}, "
                        f"err_rise={self.completeness_info.get('err_rise_mag', np.nan):.3f})"
                    )
                self.chk_bright.setChecked(True)
            else:
                self.completeness_info = {"available": False, "reason": "mag_column_missing"}
                self.chk_bright.setChecked(False)

            self.ref_bounds = self._resolve_effective_bounds()
            self.footprint_pix_xy = self._footprint_polygon_pix()

            self.lbl_star_table.setText(str(self.inputs.star_table_path))
            self.lbl_ref_fits.setText(str(self.inputs.ref_fits_path))
            self.lbl_footprint_mode.setText("ref")

            self._log(
                f"Loaded inputs | stars={len(self.df_stars)} | table={self.inputs.star_table_path.name} | "
                f"ref={self.inputs.ref_frame} | pmem_source={self.star_meta.get('pmem_source', '-')}"
            )
            pmem_mode = str(self.star_meta.get("pmem_fallback_mode", "none"))
            pmem_cov = float(self.star_meta.get("pmem_finite_fraction", 0.0)) * 100.0
            if bool(self.star_meta.get("pmem_fallback_used", False)):
                self._log(
                    f"WARNING pmem coverage incomplete: mode={pmem_mode}, finite={pmem_cov:.1f}%"
                )
            if pmem_mode == "all_missing":
                self._log("ERROR no finite pmem values found. Provide membership table/key mapping first.")
                return

            self._run_pipeline(preview=True, save_outputs=False)
        except Exception as e:
            QMessageBox.critical(self, "Load Failed", f"Failed to load cluster-structure inputs:\n{e}")
            self._log(f"ERROR load_inputs: {e}")

    def _resolve_effective_bounds(self) -> tuple[float, float, float, float]:
        ny, nx = self.ref_shape
        x0, x1 = 0.0, float(max(nx - 1, 0))
        y0, y1 = 0.0, float(max(ny - 1, 0))

        if self.inputs is None:
            return x0, x1, y0, y1

        using_cropped = "cropped" in str(self.inputs.ref_fits_path.parent).lower()
        crect = self.inputs.crop_rect if isinstance(self.inputs.crop_rect, dict) else None
        if (not using_cropped) and crect:
            try:
                cx0 = float(crect.get("x0", x0))
                cx1 = float(crect.get("x1", x1))
                cy0 = float(crect.get("y0", y0))
                cy1 = float(crect.get("y1", y1))
                x0 = max(0.0, min(cx0, cx1))
                x1 = min(float(nx - 1), max(cx0, cx1))
                y0 = max(0.0, min(cy0, cy1))
                y1 = min(float(ny - 1), max(cy0, cy1))
            except Exception:
                pass
        return x0, x1, y0, y1

    def _footprint_polygon_pix(self) -> np.ndarray:
        x0, x1, y0, y1 = self.ref_bounds
        return np.array(
            [
                [x0, y0],
                [x1, y0],
                [x1, y1],
                [x0, y1],
                [x0, y0],
            ],
            dtype=float,
        )

    def _footprint_polygon_local(self, center_ra: float, center_dec: float) -> np.ndarray:
        poly = np.asarray(self.footprint_pix_xy, float)
        ra, dec = self.ref_wcs.celestial.pixel_to_world_values(poly[:, 0], poly[:, 1])
        x, y = an.sky_to_local_xy(np.asarray(ra, float), np.asarray(dec, float), center_ra, center_dec)
        return np.column_stack([x, y])

    def _build_work_samples(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self.df_stars is None or self.df_stars.empty:
            return pd.DataFrame(), pd.DataFrame()

        df = self.df_stars.copy()
        x0, x1, y0, y1 = self.ref_bounds
        x = pd.to_numeric(df["x_pix"], errors="coerce").to_numpy(float)
        y = pd.to_numeric(df["y_pix"], errors="coerce").to_numpy(float)
        inb = np.isfinite(x) & np.isfinite(y) & (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
        df = df.loc[inb].copy()

        pmem = pd.to_numeric(df["pmem"], errors="coerce").to_numpy(float)
        m = np.isfinite(pmem) & (pmem >= self._pmem_min())

        if self.chk_ruwe.isChecked():
            rv = pd.to_numeric(df["ruwe"], errors="coerce").to_numpy(float)
            m &= (~np.isfinite(rv)) | (rv <= float(self.spin_ruwe.value()))

        df_all = df.loc[m].copy()
        if df_all.empty:
            return df_all, pd.DataFrame()

        if self.chk_bright.isChecked() and self.chk_bright.isEnabled():
            mag = pd.to_numeric(df_all["mag"], errors="coerce").to_numpy(float)
            mb = np.isfinite(mag) & (mag <= float(self.spin_mag_cut.value()))
            df_bright = df_all.loc[mb].copy()
        else:
            df_bright = pd.DataFrame(columns=df_all.columns)

        return df_all.reset_index(drop=True), df_bright.reset_index(drop=True)

    def _estimate_completeness_info(self, mags: np.ndarray, mag_err: np.ndarray) -> dict:
        m = np.isfinite(mags)
        n_mag = int(np.sum(m))
        if n_mag < 20:
            return {
                "available": False,
                "reason": "too_few_mag_samples",
                "n_mag": n_mag,
                "recommended_mag_cut": float("nan"),
            }

        x = np.asarray(mags[m], float)
        nbin = int(np.clip(np.sqrt(n_mag), 20, 70))
        counts, edges = np.histogram(x, bins=nbin)
        centers = 0.5 * (edges[:-1] + edges[1:])

        i_peak = int(np.nanargmax(counts))
        peak_count = float(counts[i_peak])
        peak_mag = float(centers[i_peak])
        drop_thr = max(3.0, 0.35 * peak_count)
        hist_drop_mag = float("nan")
        for i in range(i_peak + 1, len(counts) - 1):
            if (counts[i] <= drop_thr) and (counts[i + 1] <= drop_thr):
                hist_drop_mag = float(centers[i])
                break

        err_rise_mag = float("nan")
        n_mag_err = 0
        me = np.asarray(mag_err, float)
        mm = m & np.isfinite(me) & (me >= 0)
        n_mag_err = int(np.sum(mm))
        if n_mag_err >= 20:
            xm = np.asarray(mags[mm], float)
            em = np.asarray(me[mm], float)
            q = np.unique(np.quantile(xm, np.linspace(0.0, 1.0, 13)))
            if len(q) >= 4:
                bc = []
                be = []
                for i in range(len(q) - 1):
                    if i == len(q) - 2:
                        sel = (xm >= q[i]) & (xm <= q[i + 1])
                    else:
                        sel = (xm >= q[i]) & (xm < q[i + 1])
                    if int(np.sum(sel)) < 4:
                        continue
                    bc.append(0.5 * (q[i] + q[i + 1]))
                    be.append(float(np.nanmedian(em[sel])))
                if len(bc) >= 4:
                    bc = np.asarray(bc, float)
                    be = np.asarray(be, float)
                    n_base = max(2, len(be) // 4)
                    base = float(np.nanmedian(be[:n_base]))
                    if not np.isfinite(base) or base <= 0:
                        base = float(np.nanpercentile(be[np.isfinite(be)], 30.0))
                    thr = max(0.08, 3.0 * base)
                    for i in range(1, len(be) - 1):
                        if (be[i] >= thr) and (be[i + 1] >= thr):
                            err_rise_mag = float(bc[i])
                            break

        cands = [v for v in (hist_drop_mag, err_rise_mag) if np.isfinite(v)]
        if cands:
            rec = float(min(cands))  # safer(brighter) side for completeness-safe subset.
            rec_method = "safe_min(hist_drop, err_rise)"
        else:
            rec = float(np.nanpercentile(x, 80.0))
            rec_method = "p80_fallback"

        return {
            "available": True,
            "method": rec_method,
            "n_mag": n_mag,
            "n_mag_err": n_mag_err,
            "hist_peak_mag": peak_mag,
            "hist_peak_count": int(peak_count),
            "hist_drop_mag": hist_drop_mag,
            "err_rise_mag": err_rise_mag,
            "recommended_mag_cut": rec,
        }

    def _compute_membership_curve(self):
        self.pmem_curve_df = pd.DataFrame()
        if self.df_stars is None or self.df_stars.empty:
            return

        df = self.df_stars.copy()
        if self.ref_bounds is not None:
            x0, x1, y0, y1 = self.ref_bounds
            x = pd.to_numeric(df["x_pix"], errors="coerce").to_numpy(float)
            y = pd.to_numeric(df["y_pix"], errors="coerce").to_numpy(float)
            inb = np.isfinite(x) & np.isfinite(y) & (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
            df = df.loc[inb].copy()

        if self.chk_ruwe.isChecked():
            rv = pd.to_numeric(df["ruwe"], errors="coerce").to_numpy(float)
            mruwe = (~np.isfinite(rv)) | (rv <= float(self.spin_ruwe.value()))
            df = df.loc[mruwe].copy()

        p = pd.to_numeric(df["pmem"], errors="coerce").to_numpy(float)
        p = p[np.isfinite(p)]
        if len(p) < 5:
            return
        p = np.clip(p, 0.0, 1.0)

        thr = np.linspace(0.0, 1.0, 51)
        n_ge = np.array([int(np.sum(p >= t)) for t in thr], dtype=int)
        sum_ge = np.array([float(np.sum(p[p >= t])) for t in thr], dtype=float)
        n0 = int(max(n_ge[0], 1))
        self.pmem_curve_df = pd.DataFrame(
            {
                "p_threshold": thr,
                "n_ge_p": n_ge,
                "frac_ge_p": np.asarray(n_ge, float) / float(n0),
                "sum_p_ge_p": sum_ge,
            }
        )

    def _profile_for_fit(self, profile_df: pd.DataFrame, label: str) -> pd.DataFrame:
        if profile_df is None or profile_df.empty:
            self.fit_filter_info[label] = {
                "mode": "empty",
                "n_total": 0,
                "n_used": 0,
                "n_dropped": 0,
                "f_area_min_config": float(self.fit_filter_min_f_area),
            }
            return pd.DataFrame()

        df = profile_df.copy()
        s = pd.to_numeric(df["Sigma"], errors="coerce").to_numpy(float)
        fa = pd.to_numeric(df["f_area"], errors="coerce").to_numpy(float)
        m_strict = np.isfinite(s) & np.isfinite(fa) & (fa >= float(self.fit_filter_min_f_area))

        if int(np.sum(m_strict)) >= 6:
            mode = "strict_f_area"
            m_use = m_strict
        else:
            mode = "fallback_f_area_positive"
            m_use = np.isfinite(s) & np.isfinite(fa) & (fa > 0.0)

        out = df.loc[m_use].copy().reset_index(drop=True)
        self.fit_filter_info[label] = {
            "mode": mode,
            "n_total": int(len(df)),
            "n_used": int(len(out)),
            "n_dropped": int(len(df) - len(out)),
            "f_area_min_config": float(self.fit_filter_min_f_area),
            "f_area_min_used": float(np.nanmin(pd.to_numeric(out["f_area"], errors="coerce").to_numpy(float)))
            if not out.empty
            else float("nan"),
        }
        return out

    def _choose_final_center(self) -> tuple[float, float, str]:
        if self.center_summary is None:
            return np.nan, np.nan, "none"
        choice = str(self.combo_center_choice.currentData() or "auto")

        if choice == "manual" and self.center_manual is not None:
            return float(self.center_manual[0]), float(self.center_manual[1]), "manual"
        if choice == "iter":
            return float(self.center_summary.iter_ra), float(self.center_summary.iter_dec), "iter"
        if choice == "kde":
            return float(self.center_summary.kde_ra), float(self.center_summary.kde_dec), "kde"
        if choice == "moment":
            return float(self.center_summary.moment_ra), float(self.center_summary.moment_dec), "moment"
        return float(self.center_summary.auto_ra), float(self.center_summary.auto_dec), f"auto:{self.center_summary.auto_method}"

    def _auto_rmax(self, radii_arcsec: np.ndarray, center_ra: float, center_dec: float) -> float:
        fp = self._footprint_polygon_local(center_ra, center_dec)
        rv_fp = np.hypot(fp[:, 0], fp[:, 1])
        r_fp = float(np.nanmax(rv_fp)) if np.isfinite(rv_fp).any() else np.nan
        r_star = float(np.nanpercentile(radii_arcsec[np.isfinite(radii_arcsec)], 99.0)) if np.isfinite(radii_arcsec).any() else np.nan
        if np.isfinite(r_fp) and np.isfinite(r_star):
            return max(30.0, min(0.98 * r_fp, 1.1 * r_star))
        if np.isfinite(r_fp):
            return max(30.0, 0.98 * r_fp)
        if np.isfinite(r_star):
            return max(30.0, 1.1 * r_star)
        return 600.0

    def _centering_view_bounds(self) -> tuple[float, float, float, float]:
        x0, x1, y0, y1 = [float(v) for v in self.ref_bounds]
        if self.selected_center is None or self.ref_wcs is None:
            return x0, x1, y0, y1
        if self.df_all is None or self.df_all.empty:
            return x0, x1, y0, y1

        try:
            cx, cy = self.ref_wcs.celestial.world_to_pixel_values(self.selected_center[0], self.selected_center[1])
            cx = float(cx)
            cy = float(cy)
        except Exception:
            return x0, x1, y0, y1

        xs = pd.to_numeric(self.df_all["x_pix"], errors="coerce").to_numpy(float)
        ys = pd.to_numeric(self.df_all["y_pix"], errors="coerce").to_numpy(float)
        m = np.isfinite(xs) & np.isfinite(ys)
        if int(np.sum(m)) < 5:
            return x0, x1, y0, y1

        dx = np.abs(xs[m] - cx)
        dy = np.abs(ys[m] - cy)
        half_w = float(np.nanpercentile(dx, 97.0)) * 1.20 + 40.0
        half_h = float(np.nanpercentile(dy, 97.0)) * 1.20 + 40.0

        full_w = max(1.0, x1 - x0)
        full_h = max(1.0, y1 - y0)
        half_w = float(np.clip(half_w, 120.0, 0.5 * full_w))
        half_h = float(np.clip(half_h, 120.0, 0.5 * full_h))

        vx0 = cx - half_w
        vx1 = cx + half_w
        vy0 = cy - half_h
        vy1 = cy + half_h

        if vx0 < x0:
            vx1 += (x0 - vx0)
            vx0 = x0
        if vx1 > x1:
            vx0 -= (vx1 - x1)
            vx1 = x1
        if vy0 < y0:
            vy1 += (y0 - vy0)
            vy0 = y0
        if vy1 > y1:
            vy0 -= (vy1 - y1)
            vy1 = y1

        vx0 = max(x0, vx0)
        vx1 = min(x1, vx1)
        vy0 = max(y0, vy0)
        vy1 = min(y1, vy1)
        if (vx1 - vx0) < 40.0 or (vy1 - vy0) < 40.0:
            return x0, x1, y0, y1
        return vx0, vx1, vy0, vy1

    def _compute_structural_metrics(self):
        self.struct_metrics = {}
        if self.profile_all is None or self.profile_all.empty:
            return

        bg_outer = an.estimate_background_outer(self.profile_all, n_outer=max(2, int(0.2 * len(self.profile_all))))
        fit_choice = an.summarize_fit_choice(self.fit_all)

        bg_fit = float("nan")
        bg_source = "outer"
        preferred = str(fit_choice.get("preferred", "none"))
        for mk in (preferred, "eff", "king"):
            fres = self.fit_all.get(mk, {}) if isinstance(self.fit_all, dict) else {}
            if fres.get("ok", False):
                bg_try = float(fres.get("params", {}).get("Sigma_bg", np.nan))
                if np.isfinite(bg_try):
                    bg_fit = bg_try
                    bg_source = f"fit:{mk}"
                    break

        sigma_bg_used = bg_fit if np.isfinite(bg_fit) else float(bg_outer)
        r_lim = an.estimate_limiting_radius(self.profile_all, sigma_bg_used)
        r_half = an.estimate_half_number_radius(self.profile_all)

        self.struct_metrics = {
            "sigma_bg_outer": float(bg_outer) if np.isfinite(bg_outer) else None,
            "sigma_bg_used": float(sigma_bg_used) if np.isfinite(sigma_bg_used) else None,
            "sigma_bg_source": bg_source,
            "r_lim_arcsec": float(r_lim) if np.isfinite(r_lim) else None,
            "r_half_number_arcsec": float(r_half) if np.isfinite(r_half) else None,
            "fit_choice": fit_choice,
            "fit_filter": self.fit_filter_info,
            "shape_moment": self.shape_info if isinstance(self.shape_info, dict) else {},
        }

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------
    def _run_pipeline(self, preview: bool, save_outputs: bool):
        if self.df_stars is None or self.df_stars.empty:
            QMessageBox.warning(self, "No Data", "No star table loaded.")
            return

        self.fit_filter_min_f_area = float(self.spin_fit_farea_min.value())
        self._compute_membership_curve()

        self.df_all, self.df_bright = self._build_work_samples()
        if len(self.df_all) < 10:
            QMessageBox.warning(self, "Too Few Stars", "Not enough stars after filters (need >=10).")
            return

        ra = self.df_all["ra"].to_numpy(float)
        dec = self.df_all["dec"].to_numpy(float)
        w = np.clip(self.df_all["pmem"].to_numpy(float), 0.0, 1.0)

        grid = 100 if preview else 180
        self.center_summary = an.compute_centers(
            ra,
            dec,
            w,
            iter_radius_arcsec=float(self.spin_iter_radius.value()),
            grid_size=grid,
        )
        c_ra, c_dec, c_method = self._choose_final_center()
        self.selected_center = (c_ra, c_dec, c_method)

        x_as, y_as = an.sky_to_local_xy(ra, dec, c_ra, c_dec)
        r_as = np.hypot(x_as, y_as)
        self.df_all = self.df_all.copy()
        self.df_all["x_as"] = x_as
        self.df_all["y_as"] = y_as
        self.df_all["r_as"] = r_as
        self.shape_info = an.estimate_shape_moments(x_as, y_as, w)
        if self.selected_center is not None and isinstance(self.shape_info, dict):
            self.shape_info["center_ra"] = float(self.selected_center[0])
            self.shape_info["center_dec"] = float(self.selected_center[1])

        if self.df_bright is not None and not self.df_bright.empty:
            bx, by = an.sky_to_local_xy(
                self.df_bright["ra"].to_numpy(float),
                self.df_bright["dec"].to_numpy(float),
                c_ra,
                c_dec,
            )
            self.df_bright = self.df_bright.copy()
            self.df_bright["x_as"] = bx
            self.df_bright["y_as"] = by
            self.df_bright["r_as"] = np.hypot(bx, by)

        if self.chk_rmax_auto.isChecked():
            rmax = self._auto_rmax(r_as, c_ra, c_dec)
            self.spin_rmax.setValue(float(rmax))
        else:
            rmax = float(self.spin_rmax.value())

        rmin = float(self.spin_rmin.value())
        if rmax <= (rmin + 5.0):
            rmax = rmin + 30.0
            self.spin_rmax.setValue(float(rmax))

        mode = str(self.combo_bin_mode.currentData() or "fixed")
        edges = an.build_annulus_edges(
            r_as,
            w,
            mode=mode,
            r_min=rmin,
            r_max=rmax,
            fixed_width=float(self.spin_fixed_width.value()),
            n_equal_bins=int(self.spin_equal_bins.value()),
        )

        mc = int(self.spin_mc_samples.value())
        if preview:
            mc = min(mc, 2500)

        self.footprint_local_xy = self._footprint_polygon_local(c_ra, c_dec)
        self.area_df = an.compute_area_efficiency(edges, self.footprint_local_xy, mc_samples=mc, seed=int(self.spin_seed.value()))

        self.profile_all = an.compute_profile(r_as, w, edges, self.area_df)
        if self.df_bright is not None and not self.df_bright.empty:
            br = self.df_bright["r_as"].to_numpy(float)
            bw = np.clip(self.df_bright["pmem"].to_numpy(float), 0.0, 1.0)
            self.profile_bright = an.compute_profile(br, bw, edges, self.area_df)
        else:
            self.profile_bright = pd.DataFrame()

        self.fit_filter_info = {}
        self.fit_profile_all = self._profile_for_fit(self.profile_all, label="all")
        self.fit_profile_bright = self._profile_for_fit(self.profile_bright, label="bright")

        if preview:
            self.fit_all = an.fit_profile_models(
                self.fit_profile_all,
                model_key="eff",
                bg_mode="outer",
                bootstrap=False,
                bootstrap_n=0,
                seed=int(self.spin_seed.value()),
            )
            self.fit_bright = {}
        else:
            model_key = str(self.combo_model.currentData() or "both")
            bg_mode = str(self.combo_bg.currentData() or "outer")
            do_boot = bool(self.chk_bootstrap.isChecked())
            n_boot = int(self.spin_bootstrap.value()) if do_boot else 0
            self.fit_all = an.fit_profile_models(
                self.fit_profile_all,
                model_key=model_key,
                bg_mode=bg_mode,
                bootstrap=do_boot,
                bootstrap_n=n_boot,
                seed=int(self.spin_seed.value()),
            )
            self.fit_bright = (
                an.fit_profile_models(
                    self.fit_profile_bright,
                    model_key=model_key,
                    bg_mode=bg_mode,
                    bootstrap=False,
                    bootstrap_n=0,
                    seed=int(self.spin_seed.value()) + 17,
                )
                if (self.fit_profile_bright is not None and not self.fit_profile_bright.empty)
                else {}
            )

        self._compute_structural_metrics()

        self._draw_center_tab()
        self._draw_annuli_tab()
        self._draw_profile_tab()

        self._cache_intermediate(preview=preview)

        if save_outputs:
            self._save_outputs()
            QMessageBox.information(
                self,
                "Saved",
                f"Cluster structure outputs saved to:\n{self.inputs.output_dir}",
            )

        if save_outputs:
            mode_text = "Save"
        elif preview:
            mode_text = "Run"
        else:
            mode_text = "Run"
        self._log(
            f"{mode_text} done | stars(all)={len(self.df_all)} | bins={len(self.area_df)} | center={self.selected_center[2]}"
        )

    def _recompute_center_only(self):
        self._run_pipeline(preview=True, save_outputs=False)

    def _recompute_annuli_only(self):
        self._run_pipeline(preview=True, save_outputs=False)

    def _run_fit_only(self):
        self._run_pipeline(preview=False, save_outputs=False)

    def _clear_manual_center(self):
        if self.center_manual is None:
            return
        self.center_manual = None
        self._log("Manual center candidate cleared.")
        self._draw_center_tab()

    def _reset_center_view(self):
        if not self.fig_center.axes or self.ref_bounds is None:
            return
        x0, x1, y0, y1 = self.ref_bounds
        ax = self.fig_center.axes[0]
        ax.set_xlim(float(x0), float(x1))
        ax.set_ylim(float(y0), float(y1))
        self.canvas_center.draw_idle()

    def _zoom_out_center_view(self):
        if not self.fig_center.axes or self.ref_bounds is None:
            return
        ax = self.fig_center.axes[0]
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        cx = 0.5 * (float(x0) + float(x1))
        cy = 0.5 * (float(y0) + float(y1))
        half_w = 0.5 * abs(float(x1) - float(x0)) * 1.35
        half_h = 0.5 * abs(float(y1) - float(y0)) * 1.35

        bx0, bx1, by0, by1 = [float(v) for v in self.ref_bounds]
        nx0 = max(bx0, cx - half_w)
        nx1 = min(bx1, cx + half_w)
        ny0 = max(by0, cy - half_h)
        ny1 = min(by1, cy + half_h)

        if (nx1 - nx0) < 20.0 or (ny1 - ny0) < 20.0:
            nx0, nx1, ny0, ny1 = bx0, bx1, by0, by1

        ax.set_xlim(nx0, nx1)
        ax.set_ylim(ny0, ny1)
        self.canvas_center.draw_idle()

    def _on_center_plot_click(self, event):
        if not self.chk_manual_pick.isChecked():
            return
        if event.inaxes is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        if self.ref_wcs is None:
            return
        if not bool(getattr(event, "dblclick", False)):
            return
        if event.button not in (1, MouseButton.LEFT):
            return
        mode = str(getattr(getattr(self, "toolbar_center", None), "mode", "") or "").strip()
        if mode:
            return
        try:
            ra, dec = self.ref_wcs.celestial.pixel_to_world_values(float(event.xdata), float(event.ydata))
            self.center_manual = (float(ra), float(dec))
            self._log(f"Manual center candidate updated (double-click): RA={ra:.7f}, Dec={dec:.7f}")
            self._draw_center_tab()
        except Exception as e:
            self._log(f"Manual center click failed: {e}")

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def _draw_center_tab(self):
        self.fig_center.clear()
        ax = self.fig_center.add_subplot(111)

        img = self.ref_data
        if img is None:
            self.canvas_center.draw_idle()
            return

        x0, x1, y0, y1 = self._centering_view_bounds()
        vmin = np.nanpercentile(img, 1.0)
        vmax = np.nanpercentile(img, 99.0)
        ax.imshow(img, origin="lower", cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
        # Keep displayed image visually centered in the axes regardless of panel aspect ratio.
        ax.set_aspect("equal", adjustable="box")
        ax.set_anchor("C")

        xs = self.df_all["x_pix"].to_numpy(float)
        ys = self.df_all["y_pix"].to_numpy(float)
        cc = self.df_all["pmem"].to_numpy(float)
        if len(xs) > 7000:
            idx = np.linspace(0, len(xs) - 1, 7000).astype(int)
            xs = xs[idx]
            ys = ys[idx]
            cc = cc[idx]
        sc = ax.scatter(xs, ys, c=cc, s=10, cmap="viridis", alpha=0.65, edgecolors="none")
        self.fig_center.colorbar(sc, ax=ax, pad=0.01, fraction=0.03, label="Pmem")

        if self.center_summary is not None:
            for name, ra, dec, color in [
                ("iter", self.center_summary.iter_ra, self.center_summary.iter_dec, "cyan"),
                ("kde", self.center_summary.kde_ra, self.center_summary.kde_dec, "magenta"),
            ]:
                try:
                    cx, cy = self.ref_wcs.celestial.world_to_pixel_values(ra, dec)
                    ax.plot(cx, cy, marker="*", markersize=14, color=color, markeredgecolor="black", zorder=6)
                    ax.text(cx + 8, cy + 8, name, color=color, fontsize=9, weight="bold")
                except Exception:
                    pass

        if self.center_manual is not None:
            try:
                mx, my = self.ref_wcs.celestial.world_to_pixel_values(self.center_manual[0], self.center_manual[1])
                ax.plot(mx, my, marker="x", markersize=12, color="yellow", markeredgewidth=2.0, zorder=7)
                ax.text(mx + 8, my - 12, "manual", color="yellow", fontsize=9, weight="bold")
            except Exception:
                pass

        if self.selected_center is not None:
            try:
                cx, cy = self.ref_wcs.celestial.world_to_pixel_values(self.selected_center[0], self.selected_center[1])
                ax.plot(cx, cy, marker="+", markersize=18, color="red", markeredgewidth=2.0, zorder=8)
            except Exception:
                pass

        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_title("Center candidates on reference image")
        ax.set_xlabel("X (pix)")
        ax.set_ylabel("Y (pix)")
        self.canvas_center.draw_idle()

        if self.center_summary is not None:
            lines = [
                f"iter   : RA={self.center_summary.iter_ra:.8f}, Dec={self.center_summary.iter_dec:.8f}",
                f"kde    : RA={self.center_summary.kde_ra:.8f}, Dec={self.center_summary.kde_dec:.8f}",
                f"moment : RA={self.center_summary.moment_ra:.8f}, Dec={self.center_summary.moment_dec:.8f}",
                f"auto   : method={self.center_summary.auto_method}, RA={self.center_summary.auto_ra:.8f}, Dec={self.center_summary.auto_dec:.8f}",
                f"iter<->kde sep = {self.center_summary.iter_kde_sep_arcsec:.3f} arcsec",
            ]
            if self.selected_center is not None:
                lines.append(
                    f"chosen: method={self.selected_center[2]}, RA={self.selected_center[0]:.8f}, Dec={self.selected_center[1]:.8f}"
                )
            if self.center_manual is not None:
                lines.append(
                    f"manual: RA={self.center_manual[0]:.8f}, Dec={self.center_manual[1]:.8f}"
                )
            lines.append(
                f"manual_pick={'ON' if self.chk_manual_pick.isChecked() else 'OFF'} (double-click only, applied only when Center mode=Manual)"
            )
            self.text_center.setPlainText("\n".join(lines))

    def _draw_annuli_tab(self):
        self.fig_annuli.clear()
        ax_img = self.fig_annuli.add_subplot(1, 2, 1)
        ax_fa = self.fig_annuli.add_subplot(1, 2, 2)

        img = self.ref_data
        if img is None:
            self.canvas_annuli.draw_idle()
            return

        vmin = np.nanpercentile(img, 1.0)
        vmax = np.nanpercentile(img, 99.0)
        ax_img.imshow(img, origin="lower", cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")

        xs = self.df_all["x_pix"].to_numpy(float)
        ys = self.df_all["y_pix"].to_numpy(float)
        ax_img.scatter(xs, ys, s=7, c="#65c36e", alpha=0.55, edgecolors="none")

        if self.selected_center is not None:
            cx, cy = self.ref_wcs.celestial.world_to_pixel_values(self.selected_center[0], self.selected_center[1])
            ax_img.plot(cx, cy, "r+", markersize=15, markeredgewidth=2.0)

            pix_scale = self.pixel_scale_arcsec if np.isfinite(self.pixel_scale_arcsec) and self.pixel_scale_arcsec > 0 else 1.0
            if self.area_df is not None and not self.area_df.empty:
                rin = pd.to_numeric(self.area_df["r_in"], errors="coerce").to_numpy(float)
                rout = pd.to_numeric(self.area_df["r_out"], errors="coerce").to_numpy(float)
                edges = np.unique(np.concatenate([rin[:1], rout]))
                edges = edges[np.isfinite(edges)]
                edges = np.sort(edges)

                for i, r_edge in enumerate(edges):
                    if r_edge <= 0.0:
                        continue
                    rad_pix = float(r_edge) / pix_scale
                    is_rmin = i == 0
                    c = Circle(
                        (cx, cy),
                        rad_pix,
                        fill=False,
                        edgecolor=("#ff1744" if is_rmin else "#ffa726"),
                        linewidth=(1.9 if is_rmin else 1.0),
                        alpha=(0.95 if is_rmin else 0.8),
                    )
                    ax_img.add_patch(c)
                    if is_rmin:
                        ax_img.text(
                            cx + 8,
                            cy + rad_pix + 8,
                            f"r_min={float(r_edge):.0f}\"",
                            color="#ff1744",
                            fontsize=8,
                            weight="bold",
                        )

            # Shape diagnostics from weighted second moments (if available).
            try:
                if isinstance(self.shape_info, dict) and bool(self.shape_info.get("available", False)):
                    pa = float(self.shape_info.get("pa_deg_from_x_ccw", np.nan))
                    a_sig = float(self.shape_info.get("a_sigma_arcsec", np.nan))
                    b_sig = float(self.shape_info.get("b_sigma_arcsec", np.nan))
                    ecc = float(self.shape_info.get("ellipticity", np.nan))
                    if np.isfinite(pa) and np.isfinite(a_sig) and np.isfinite(b_sig):
                        w_pix = max(2.0, (2.0 * a_sig) / max(pix_scale, 1e-9))
                        h_pix = max(2.0, (2.0 * b_sig) / max(pix_scale, 1e-9))
                        ell = Ellipse(
                            (cx, cy),
                            width=w_pix,
                            height=h_pix,
                            angle=pa,
                            fill=False,
                            edgecolor="#66ff66",
                            linewidth=1.4,
                            alpha=0.95,
                        )
                        ax_img.add_patch(ell)

                        pa_rad = math.radians(pa)
                        rline = max(40.0, min(300.0, 2.5 * a_sig))
                        x1_as = -rline * math.cos(pa_rad)
                        y1_as = -rline * math.sin(pa_rad)
                        x2_as = +rline * math.cos(pa_rad)
                        y2_as = +rline * math.sin(pa_rad)
                        ra2, dec2 = an.local_xy_to_sky(
                            np.array([x1_as, x2_as], dtype=float),
                            np.array([y1_as, y2_as], dtype=float),
                            float(self.selected_center[0]),
                            float(self.selected_center[1]),
                        )
                        px2, py2 = self.ref_wcs.celestial.world_to_pixel_values(ra2, dec2)
                        ax_img.plot(px2, py2, "-", color="#66ff66", linewidth=1.6, alpha=0.95)
                        if np.isfinite(ecc):
                            ax_img.text(
                                cx + 12,
                                cy - 20,
                                f"e={ecc:.3f}, PA={pa:.1f}deg",
                                color="#66ff66",
                                fontsize=8,
                                weight="bold",
                            )
            except Exception:
                pass

        if self.footprint_pix_xy is not None:
            poly = np.asarray(self.footprint_pix_xy, float)
            ax_img.plot(poly[:, 0], poly[:, 1], color="cyan", linewidth=1.2)

        x0, x1, y0, y1 = self.ref_bounds
        ax_img.set_xlim(x0, x1)
        ax_img.set_ylim(y0, y1)
        ax_img.set_title("Annuli + footprint overlay")
        ax_img.set_xlabel("X (pix)")
        ax_img.set_ylabel("Y (pix)")

        if self.area_df is not None and not self.area_df.empty:
            rmid = 0.5 * (self.area_df["r_in"].to_numpy(float) + self.area_df["r_out"].to_numpy(float))
            fa = self.area_df["f_area"].to_numpy(float)
            ax_fa.plot(rmid, fa, "o-", color="#1e88e5", markersize=4)
            ax_fa.set_ylim(-0.05, 1.05)
        ax_fa.set_title("Area fraction f_area vs r")
        ax_fa.set_xlabel("r (arcsec)")
        ax_fa.set_ylabel("f_area")
        ax_fa.grid(alpha=0.25)

        self.fig_annuli.tight_layout()
        self.canvas_annuli.draw_idle()

        if self.area_df is not None and not self.area_df.empty:
            txt = [
                f"bins={len(self.area_df)}",
                f"f_area median={float(np.nanmedian(self.area_df['f_area'].to_numpy(float))):.3f}",
                f"f_area min={float(np.nanmin(self.area_df['f_area'].to_numpy(float))):.3f}",
                f"r range = [{float(self.area_df['r_in'].min()):.1f}, {float(self.area_df['r_out'].max()):.1f}] arcsec",
                f"r_min used = {float(self.area_df['r_in'].min()):.1f} arcsec",
            ]
            if isinstance(self.shape_info, dict) and bool(self.shape_info.get("available", False)):
                txt.append(
                    f"shape: e={float(self.shape_info.get('ellipticity', np.nan)):.3f}, "
                    f"PA={float(self.shape_info.get('pa_deg_from_x_ccw', np.nan)):.2f} deg"
                )
            self.text_annuli.setPlainText("\n".join(txt))

    def _draw_profile_tab(self):
        self.fig_profile.clear()
        ax = self.fig_profile.add_subplot(2, 1, 1)
        axr = self.fig_profile.add_subplot(2, 1, 2, sharex=ax)

        if self.profile_all is not None and not self.profile_all.empty:
            rr = self.profile_all["r_mid"].to_numpy(float)
            yy = self.profile_all["Sigma"].to_numpy(float)
            ee = self.profile_all["Sigma_err"].to_numpy(float)
            ax.errorbar(rr, yy, yerr=ee, fmt="o", ms=4, color="#1976d2", ecolor="#90caf9", label="ALL")

        if self.profile_bright is not None and not self.profile_bright.empty:
            br = self.profile_bright["r_mid"].to_numpy(float)
            by = self.profile_bright["Sigma"].to_numpy(float)
            be = self.profile_bright["Sigma_err"].to_numpy(float)
            ax.errorbar(br, by, yerr=be, fmt="s", ms=4, color="#ef6c00", ecolor="#ffcc80", label="BRIGHT")

        fit_txt = []
        for name, fit_res, color in [
            ("EFF", self.fit_all.get("eff", {}), "#2e7d32"),
            ("King", self.fit_all.get("king", {}), "#8e24aa"),
        ]:
            if not fit_res.get("ok", False):
                continue
            rfit = np.asarray(fit_res.get("r_fit", []), float)
            yfit = np.asarray(fit_res.get("y_fit", []), float)
            res = np.asarray(fit_res.get("residuals", []), float)
            if len(rfit) and len(yfit):
                order = np.argsort(rfit)
                ax.plot(rfit[order], yfit[order], "-", color=color, linewidth=1.8, label=f"{name} fit")
            if len(rfit) and len(res):
                axr.plot(rfit, res, ".-", color=color, alpha=0.85, label=f"{name} residual")
            p = fit_res.get("params", {})
            pe = fit_res.get("errors", {})
            ptxt = ", ".join(
                [
                    f"{k}={float(v):.4g}" + (f"+/-{float(pe[k]):.2g}" if k in pe and np.isfinite(pe[k]) else "")
                    for k, v in p.items()
                    if np.isfinite(v)
                ]
            )
            fit_txt.append(
                f"{name}: {ptxt} | rms={float(fit_res.get('rms', np.nan)):.4g} | AIC={float(fit_res.get('aic', np.nan)):.3f}"
            )

        if self.fit_filter_info:
            finfo = self.fit_filter_info.get("all", {})
            if finfo:
                fit_txt.append(
                    f"Fit bins(all): {finfo.get('n_used', 0)}/{finfo.get('n_total', 0)} "
                    f"(drop={finfo.get('n_dropped', 0)}, mode={finfo.get('mode', '-')}, "
                    f"f_area_min={finfo.get('f_area_min_config', np.nan):.2f})"
                )
        if self.struct_metrics:
            r_lim = self.struct_metrics.get("r_lim_arcsec", None)
            r_half = self.struct_metrics.get("r_half_number_arcsec", None)
            choice = self.struct_metrics.get("fit_choice", {})
            if r_lim is not None:
                fit_txt.append(f"r_lim ~= {float(r_lim):.2f} arcsec")
            if r_half is not None:
                fit_txt.append(f"r_half(number) ~= {float(r_half):.2f} arcsec")
            if isinstance(choice, dict):
                fit_txt.append(f"fit_choice: {choice.get('note', '-')}")
        if self.pmem_curve_df is not None and not self.pmem_curve_df.empty:
            p80 = int(
                pd.to_numeric(
                    self.pmem_curve_df.loc[self.pmem_curve_df["p_threshold"] >= 0.8, "n_ge_p"],
                    errors="coerce",
                ).max()
                if np.any(self.pmem_curve_df["p_threshold"].to_numpy(float) >= 0.8)
                else 0
            )
            fit_txt.append(f"N(>=0.8)={p80} (see diagnostics/n_ge_p_vs_p.png)")

        ax.set_title("Radial Surface Density Sigma(r)")
        ax.set_ylabel("Sigma (weighted / arcsec^2)")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8)

        axr.axhline(0.0, color="gray", linewidth=1.0)
        axr.set_xlabel("r (arcsec)")
        axr.set_ylabel("Residual")
        axr.grid(alpha=0.25)
        axr.legend(loc="best", fontsize=8)

        self.fig_profile.tight_layout()
        self.canvas_profile.draw_idle()

        if fit_txt:
            self.text_profile.setPlainText("\n".join(fit_txt))
        else:
            self.text_profile.setPlainText("Fit not available. Check profile bins/filters.")

    def _draw_overview_tab(self):
        return

    # ------------------------------------------------------------------
    # Cache + Save
    # ------------------------------------------------------------------
    @staticmethod
    def _sanitize_json_value(v):
        if isinstance(v, (np.floating, float)):
            return float(v) if np.isfinite(v) else None
        if isinstance(v, (np.integer, int)):
            return int(v)
        if isinstance(v, (list, tuple)):
            return [ClusterStructureWindow._sanitize_json_value(x) for x in v]
        if isinstance(v, dict):
            return {str(k): ClusterStructureWindow._sanitize_json_value(val) for k, val in v.items()}
        return v

    def _cache_intermediate(self, preview: bool):
        if self.inputs is None:
            return
        try:
            cache = self.inputs.cache_dir

            if self.df_all is not None and not self.df_all.empty and {"x_as", "y_as", "r_as"} <= set(self.df_all.columns):
                cache_cols = [c for c in ["star_id", "x_as", "y_as", "r_as", "pmem", "mag"] if c in self.df_all.columns]
                self.df_all[cache_cols].to_csv(cache / "stars_xy.csv", index=False)

            if self.center_summary is not None and self.center_summary.kde_grid_z is not None:
                np.save(cache / "kde_grid.npy", np.asarray(self.center_summary.kde_grid_z, float))
                if self.center_summary.kde_grid_x is not None:
                    np.save(cache / "kde_grid_x.npy", np.asarray(self.center_summary.kde_grid_x, float))
                if self.center_summary.kde_grid_y is not None:
                    np.save(cache / "kde_grid_y.npy", np.asarray(self.center_summary.kde_grid_y, float))

            if self.area_df is not None and not self.area_df.empty:
                np.save(cache / "area_mc_bins.npy", self.area_df["f_area"].to_numpy(float))

            prev_state = {
                "preview": bool(preview),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "pmem_min": self._pmem_min(),
                "ruwe_enable": bool(self.chk_ruwe.isChecked()),
                "ruwe_max": float(self.spin_ruwe.value()),
                "bright_enable": bool(self.chk_bright.isChecked() and self.chk_bright.isEnabled()),
                "mag_cut": float(self.spin_mag_cut.value()),
                "fit_filter_min_f_area": float(self.fit_filter_min_f_area),
                "seed": int(self.spin_seed.value()),
            }
            (cache / "preview_state.json").write_text(json.dumps(prev_state, indent=2), encoding="utf-8")
        except Exception as e:
            self._log(f"cache write failed: {e}")

    def _save_outputs(self):
        if self.inputs is None:
            return
        out = self.inputs.output_dir
        diag = self.inputs.diagnostics_dir

        inputs_json = {
            "star_table_path": str(self.inputs.star_table_path),
            "ref_fits_path": str(self.inputs.ref_fits_path),
            "ref_meta_path": str(self.inputs.ref_meta_path),
            "footprint_mode": self._footprint_mode(),
            "membership": {
                "pmem_source": self.star_meta.get("pmem_source", None),
                "pmem_col": self.star_meta.get("pmem_col", None),
                "pmem_fallback_mode": self.star_meta.get("pmem_fallback_mode", None),
                "pmem_fallback_used": bool(self.star_meta.get("pmem_fallback_used", False)),
                "pmem_all_unity": bool(self.star_meta.get("pmem_all_unity", False)),
                "pmem_finite_fraction": self.star_meta.get("pmem_finite_fraction", None),
            },
            "completeness": self.completeness_info,
            "settings": {
                "pmem_min": self._pmem_min(),
                "ruwe_enable": bool(self.chk_ruwe.isChecked()),
                "ruwe_max": float(self.spin_ruwe.value()),
                "bright_enable": bool(self.chk_bright.isChecked() and self.chk_bright.isEnabled()),
                "mag_cut": float(self.spin_mag_cut.value()),
                "seed": int(self.spin_seed.value()),
                "fit_filter_min_f_area": float(self.fit_filter_min_f_area),
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        (out / "inputs.json").write_text(json.dumps(self._sanitize_json_value(inputs_json), indent=2), encoding="utf-8")

        if self.center_summary is not None and self.selected_center is not None:
            center_json = {
                "chosen_center": {
                    "ra": self.selected_center[0],
                    "dec": self.selected_center[1],
                    "method": self.selected_center[2],
                },
                "candidates": {
                    "iter": {"ra": self.center_summary.iter_ra, "dec": self.center_summary.iter_dec},
                    "kde": {"ra": self.center_summary.kde_ra, "dec": self.center_summary.kde_dec},
                    "moment": {"ra": self.center_summary.moment_ra, "dec": self.center_summary.moment_dec},
                    "manual": (
                        {"ra": self.center_manual[0], "dec": self.center_manual[1]} if self.center_manual is not None else None
                    ),
                },
                "deltas_arcsec": {
                    "iter_kde": self.center_summary.iter_kde_sep_arcsec,
                    "iter_auto": an.angular_sep_arcsec(
                        self.center_summary.iter_ra,
                        self.center_summary.iter_dec,
                        self.selected_center[0],
                        self.selected_center[1],
                    ),
                    "kde_auto": an.angular_sep_arcsec(
                        self.center_summary.kde_ra,
                        self.center_summary.kde_dec,
                        self.selected_center[0],
                        self.selected_center[1],
                    ),
                },
            }
            (out / "center.json").write_text(json.dumps(self._sanitize_json_value(center_json), indent=2), encoding="utf-8")

        if self.area_df is not None and not self.area_df.empty:
            ann_cfg = {
                "binning_mode": str(self.combo_bin_mode.currentData() or "fixed"),
                "r_min": float(self.spin_rmin.value()),
                "r_max": float(self.spin_rmax.value()),
                "r_edges": self.area_df["r_in"].tolist() + [float(self.area_df["r_out"].iloc[-1])],
                "mc_samples": int(self.spin_mc_samples.value()),
                "seed": int(self.spin_seed.value()),
            }
            (out / "annuli_config.json").write_text(json.dumps(self._sanitize_json_value(ann_cfg), indent=2), encoding="utf-8")
            self.area_df.to_csv(out / "area_efficiency.csv", index=False)

        if self.profile_all is not None and not self.profile_all.empty:
            self.profile_all.to_csv(out / "profile_radial.csv", index=False)
        if self.profile_bright is not None and not self.profile_bright.empty:
            self.profile_bright.to_csv(out / "profile_radial_bright.csv", index=False)
        if self.pmem_curve_df is not None and not self.pmem_curve_df.empty:
            self.pmem_curve_df.to_csv(out / "membership_p_threshold_curve.csv", index=False)
        if isinstance(self.shape_info, dict) and self.shape_info:
            (out / "ellipticity.json").write_text(
                json.dumps(self._sanitize_json_value(self.shape_info), indent=2),
                encoding="utf-8",
            )

        eff = self.fit_all.get("eff", {}) if isinstance(self.fit_all, dict) else {}
        if eff.get("ok", False):
            eff_payload = {k: v for k, v in eff.items() if k not in ("r_fit", "y_fit", "residuals")}
            eff_payload["formula"] = "Sigma(r)=Sigma0*(1+(r/a)^2)^(-gamma/2)+Sigma_bg"
            eff_payload["fit_choice"] = self.struct_metrics.get("fit_choice", {}) if isinstance(self.struct_metrics, dict) else {}
            eff_payload["fit_filter"] = self.fit_filter_info.get("all", {}) if isinstance(self.fit_filter_info, dict) else {}
            (out / "fit_eff.json").write_text(
                json.dumps(self._sanitize_json_value(eff_payload), indent=2),
                encoding="utf-8",
            )
        king = self.fit_all.get("king", {}) if isinstance(self.fit_all, dict) else {}
        if king.get("ok", False):
            king_payload = {k: v for k, v in king.items() if k not in ("r_fit", "y_fit", "residuals")}
            king_payload["formula"] = (
                "Sigma(r)=Sigma0*(1/sqrt(1+(r/rc)^2)-1/sqrt(1+(rt/rc)^2))^2 + Sigma_bg "
                "for r<rt; Sigma_bg for r>=rt"
            )
            king_payload["fit_choice"] = self.struct_metrics.get("fit_choice", {}) if isinstance(self.struct_metrics, dict) else {}
            king_payload["fit_filter"] = self.fit_filter_info.get("all", {}) if isinstance(self.fit_filter_info, dict) else {}
            (out / "fit_king.json").write_text(
                json.dumps(self._sanitize_json_value(king_payload), indent=2),
                encoding="utf-8",
            )
        if isinstance(self.struct_metrics, dict) and self.struct_metrics:
            (out / "structural_metrics.json").write_text(
                json.dumps(self._sanitize_json_value(self.struct_metrics), indent=2),
                encoding="utf-8",
            )

        try:
            self.fig_center.savefig(diag / "overlay_center.png", dpi=170)
            self.fig_annuli.savefig(diag / "overlay_annuli.png", dpi=170)
            self.fig_profile.savefig(diag / "sigma_r.png", dpi=170)
        except Exception as e:
            self._log(f"figure save warning: {e}")

        if self.pmem_curve_df is not None and not self.pmem_curve_df.empty:
            f = Figure(figsize=(6, 4))
            c = FigureCanvas(f)
            ax = f.add_subplot(111)
            px = pd.to_numeric(self.pmem_curve_df["p_threshold"], errors="coerce").to_numpy(float)
            ny = pd.to_numeric(self.pmem_curve_df["n_ge_p"], errors="coerce").to_numpy(float)
            fy = pd.to_numeric(self.pmem_curve_df["frac_ge_p"], errors="coerce").to_numpy(float)
            ax.plot(px, ny, "-", color="#1565c0", linewidth=1.8, label="N(>=P)")
            ax2 = ax.twinx()
            ax2.plot(px, fy, "--", color="#ef6c00", linewidth=1.5, label="Frac(>=P)")
            ax.set_xlabel("P threshold")
            ax.set_ylabel("N(>=P)")
            ax2.set_ylabel("Frac(>=P)")
            ax.set_title("Membership Threshold Curve")
            ax.grid(alpha=0.25)
            f.tight_layout()
            f.savefig(diag / "n_ge_p_vs_p.png", dpi=170)
            del c

        if self.area_df is not None and not self.area_df.empty:
            f = Figure(figsize=(6, 4))
            c = FigureCanvas(f)
            ax = f.add_subplot(111)
            rmid = 0.5 * (self.area_df["r_in"].to_numpy(float) + self.area_df["r_out"].to_numpy(float))
            ax.plot(rmid, self.area_df["f_area"].to_numpy(float), "o-", color="#1e88e5")
            ax.set_xlabel("r (arcsec)")
            ax.set_ylabel("f_area")
            ax.set_ylim(-0.05, 1.05)
            ax.grid(alpha=0.25)
            f.tight_layout()
            f.savefig(diag / "f_area_vs_r.png", dpi=170)
            del c

        self._save_residual_plot(diag / "residuals_eff.png", self.fit_all.get("eff", {}), title="EFF residuals")
        self._save_residual_plot(diag / "residuals_king.png", self.fit_all.get("king", {}), title="King residuals")

    def _save_residual_plot(self, path: Path, fit_res: dict, title: str):
        if not fit_res or not fit_res.get("ok", False):
            return
        rr = np.asarray(fit_res.get("r_fit", []), float)
        res = np.asarray(fit_res.get("residuals", []), float)
        if len(rr) == 0 or len(res) == 0:
            return
        f = Figure(figsize=(6, 4))
        c = FigureCanvas(f)
        ax = f.add_subplot(111)
        ax.axhline(0.0, color="gray", linewidth=1.0)
        ax.plot(rr, res, "o-", markersize=3)
        ax.set_xlabel("r (arcsec)")
        ax.set_ylabel("Residual")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        f.tight_layout()
        f.savefig(path, dpi=170)
        del c
