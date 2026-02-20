"""
Step 13: Isochrone Model
Ported from AAPKI_GUI.ipynb Cell 16 (isochrone fitting).

Extended with automatic isochrone fitting:
- AutoFit mode: Global search + local refinement for initial parameter estimation
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import Slider, Button
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.spatial import cKDTree

from PyQt5.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QGroupBox, QMessageBox,
    QTextEdit, QFormLayout, QDoubleSpinBox, QComboBox,
    QLineEdit, QWidget, QFileDialog, QProgressBar,
    QTabWidget, QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from .step_window_base import StepWindowBase
from ...utils.step_paths import step11_dir, step13_dir
from ...analysis.isochrone_fitter_v2 import IsochroneFitterV2, FitMode, FitResult, FitBounds, GridScanResult

# SDSS extinction coefficients: R = A / E(B-V)
SDSS_R = {"g": 3.303, "r": 2.285, "i": 1.698, "z": 1.263}

# Default PARSEC isochrone column indices for each band
SDSS_ISO_COL = {"g": 29, "r": 30, "i": 31, "z": 32}

# Allowed color pairs and magnitude bands
ALLOWED_COLOR_PAIRS = [("g", "r"), ("g", "i"), ("r", "i")]
ALLOWED_MAG_BANDS = ["g", "r", "i"]


class FitWorker(QThread):
    """Background worker for isochrone fitting"""

    finished = pyqtSignal(object)  # FitResult or Exception
    progress = pyqtSignal(float, str)  # progress (0-1), message

    def __init__(self, fitter: IsochroneFitterV2,
                 obs_color: np.ndarray, obs_mag: np.ndarray,
                 obs_color_err: np.ndarray, obs_mag_err: np.ndarray,
                 mode: FitMode, bounds: FitBounds, snr_min: float,
                 fit_kwargs: Optional[dict] = None):
        super().__init__()
        self.fitter = fitter
        self.obs_color = obs_color
        self.obs_mag = obs_mag
        self.obs_color_err = obs_color_err
        self.obs_mag_err = obs_mag_err
        self.mode = mode
        self.bounds = bounds
        self.snr_min = snr_min
        self.fit_kwargs = fit_kwargs or {}

    def run(self):
        try:
            # Set progress callback
            self.fitter.progress_callback = lambda p, m: self.progress.emit(p, m)

            result = self.fitter.fit(
                self.obs_color, self.obs_mag,
                self.obs_color_err, self.obs_mag_err,
                mode=self.mode,
                bounds=self.bounds,
                snr_min=self.snr_min,
                **self.fit_kwargs
            )
            self.finished.emit(result)

        except Exception as e:
            self.finished.emit(e)


class GridScanWorker(QThread):
    """Background worker for grid scan fitting."""

    finished = pyqtSignal(object)  # GridScanResult or Exception
    progress = pyqtSignal(float, str)

    def __init__(self, fitter: IsochroneFitterV2,
                 obs_color: np.ndarray, obs_mag: np.ndarray,
                 obs_color_err: np.ndarray, obs_mag_err: np.ndarray,
                 bounds: FitBounds, snr_min: float,
                 fit_kwargs: Optional[dict] = None):
        super().__init__()
        self.fitter = fitter
        self.obs_color = obs_color
        self.obs_mag = obs_mag
        self.obs_color_err = obs_color_err
        self.obs_mag_err = obs_mag_err
        self.bounds = bounds
        self.snr_min = snr_min
        self.fit_kwargs = fit_kwargs or {}

    def run(self):
        try:
            self.fitter.progress_callback = lambda p, m: self.progress.emit(p, m)
            result = self.fitter.fit_grid_scan(
                self.obs_color, self.obs_mag,
                self.obs_color_err, self.obs_mag_err,
                bounds=self.bounds,
                snr_min=self.snr_min,
                **self.fit_kwargs,
            )
            self.finished.emit(result)
        except Exception as e:
            self.finished.emit(e)


class IsochroneViewerWindow(QWidget):
    """Interactive isochrone viewer using matplotlib sliders."""

    def __init__(self, df: pd.DataFrame, iso_raw: np.ndarray, params,
                 parent=None, embedded=False,
                 band_mag: str = "g", band_color: tuple = ("g", "r"),
                 iso_col_1: int = 29, iso_col_2: int = 30):
        super().__init__(parent)
        self.df = df
        self.iso_raw = iso_raw
        self.params = params
        self.embedded = bool(embedded)
        self.band_mag = band_mag
        self.band_color = band_color
        self.iso_col_1 = iso_col_1
        self.iso_col_2 = iso_col_2
        self.color_label = f"{band_color[0]}-{band_color[1]}"
        self.extinction_label = f"E({self.color_label})"

        if not self.embedded:
            self.setWindowTitle("Isochrone Viewer")
            self.resize(1200, 900)
            self.setMinimumSize(900, 700)
        else:
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        fig_size = (14, 10) if not self.embedded else (11, 8)
        self.figure = Figure(figsize=fig_size)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        if not self.embedded:
            self.canvas.setMinimumSize(800, 600)
        layout.addWidget(self.canvas, stretch=1)

        self._build_plot()

    def current_slider_values(self):
        """Return current manual slider state as a fit-initial guess dict."""
        if not all(hasattr(self, attr) for attr in ("s_age", "s_mh", "s_vshift", "s_hshift")):
            return None
        return {
            "log_age": float(self.s_age.val),
            "metallicity": float(self.s_mh.val),
            "distance_mod": float(self.s_vshift.val),
            "extinction_gr": float(self.s_hshift.val),
        }

    def _build_plot(self):
        mpl.rcParams['axes.unicode_minus'] = False
        self.figure.clear()

        teff_vmin = 2400.0
        teff_vmax = 40000.0
        ob_norm = Normalize(vmin=teff_vmin, vmax=teff_vmax, clip=True)

        anchors = [
            (2400, "#E53935"),
            (3200, "#FF6A3D"),
            (4500, "#FFB84D"),
            (5800, "#FFE36A"),
            (6500, "#FFF6C7"),
            (8000, "#FFFFFF"),
            (10000, "#FFFFFF"),
            (20000, "#2D5BFF"),
            (40000, "#7A3CFF"),
        ]
        anchors = sorted(anchors, key=lambda x: x[0])
        pos = [(t - teff_vmin) / (teff_vmax - teff_vmin) for t, _ in anchors]
        pos[0] = 0.0
        pos[-1] = 1.0

        ob_cmap = LinearSegmentedColormap.from_list(
            "obafgkm_like",
            list(zip(pos, [c for _, c in anchors])),
            N=256
        )
        ob_cmap.set_bad("#777777")

        gr_x = np.array([-0.40, -0.20, 0.00, 0.30, 0.45, 0.80, 1.40, 1.80], float)
        gr_t = np.array([35000, 20000, 10000, 7500, 6000, 4500, 3200, 2400], float)

        def teff_from_gr(gr):
            gr = np.asarray(gr, float)
            t = np.interp(gr, gr_x, gr_t)
            return np.clip(t, teff_vmin, teff_vmax)

        available_ages = np.unique(self.iso_raw[:, 2])
        available_mhs = np.unique(self.iso_raw[:, 1])

        b1, b2 = self.band_color
        bm = self.band_mag
        std1 = f"mag_std_{b1}"
        std2 = f"mag_std_{b2}"
        inst1 = f"mag_inst_{b1}"
        inst2 = f"mag_inst_{b2}"
        stdm = f"mag_std_{bm}"
        instm = f"mag_inst_{bm}"

        if std1 in self.df.columns and std2 in self.df.columns:
            obs_band1 = self.df[std1].to_numpy(float)
            obs_band2 = self.df[std2].to_numpy(float)
        else:
            obs_band1 = self.df.get(inst1, pd.Series([], dtype=float)).to_numpy(float)
            obs_band2 = self.df.get(inst2, pd.Series([], dtype=float)).to_numpy(float)
        if stdm in self.df.columns:
            obs_mag = self.df[stdm].to_numpy(float)
        elif instm in self.df.columns:
            obs_mag = self.df[instm].to_numpy(float)
        else:
            obs_mag = obs_band1.copy()

        obs_color = obs_band1 - obs_band2
        mask = np.isfinite(obs_mag) & np.isfinite(obs_color)
        obs_mag, obs_color = obs_mag[mask], obs_color[mask]
        obs_pts = np.c_[obs_color, obs_mag]
        obs_teff = teff_from_gr(obs_color)

        gs = self.figure.add_gridspec(2, 2, width_ratios=[2.5, 1], height_ratios=[3, 1], hspace=0.3, wspace=0.2)
        ax_cmd = self.figure.add_subplot(gs[0, 0])
        ax_hist = self.figure.add_subplot(gs[0, 1])
        ax_res = self.figure.add_subplot(gs[1, 0])

        # Leave more space at bottom for sliders (0.22 for sliders + margin)
        self.figure.subplots_adjust(left=0.08, right=0.88, bottom=0.22, top=0.95)

        self.figure.patch.set_facecolor("black")
        for ax in (ax_cmd, ax_hist, ax_res):
            ax.set_facecolor("black")
            for sp in ax.spines.values():
                sp.set_color("white")
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")

        sc_obs = ax_cmd.scatter(obs_color, obs_mag, s=3, alpha=0.85, linewidths=0, c=obs_teff, cmap=ob_cmap, norm=ob_norm, label="Observed")
        sc_iso = ax_cmd.scatter([np.nan], [np.nan], s=12, alpha=0.95, linewidths=0, c=[np.nan], cmap=ob_cmap, norm=ob_norm, label="Isochrone", zorder=6)

        ax_cmd.invert_yaxis()
        ax_cmd.set_xlabel(f"Standard ({self.color_label})")
        ax_cmd.set_ylabel(f"Standard {bm}")
        ax_cmd.grid(True, linestyle=":", alpha=0.35)
        ax_cmd.legend(loc="upper right")

        res_scat = ax_res.scatter([], [], s=3, alpha=0.75, linewidths=0, color="cyan")
        ax_res.axhline(0, color="white", lw=1, ls="--", alpha=0.6)
        ax_res.set_xlabel(f"Standard {bm}")
        ax_res.set_ylabel("Residual (NN dist in CMD)")

        sm = mpl.cm.ScalarMappable(norm=ob_norm, cmap=ob_cmap)
        sm.set_array([])
        cbar = self.figure.colorbar(sm, ax=[ax_cmd, ax_res], fraction=0.03, pad=0.02)
        cbar.set_label("Teff (K) + OBAFGKM-like color", color="white")
        cbar.ax.tick_params(colors="white")
        for sp in cbar.ax.spines.values():
            sp.set_color("white")

        c1 = self.iso_col_1
        c2 = self.iso_col_2
        # mag band column: same as band1 if band_mag == band_color[0]
        cm = c1 if self.band_mag == self.band_color[0] else c2

        def get_iso_points(age, mh, h_shift, v_shift):
            m = (self.iso_raw[:, 2] == age) & (self.iso_raw[:, 1] == mh)
            filtered = self.iso_raw[m]
            if len(filtered) == 0:
                return np.array([]), np.array([])
            mag_model = filtered[:, cm] + v_shift
            color_model = (filtered[:, c1] - filtered[:, c2]) + h_shift
            return color_model, mag_model

        def style_axis_dark(ax):
            ax.set_facecolor("black")
            for sp in ax.spines.values():
                sp.set_color("white")
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")

        age_init = float(getattr(self.params.P, "iso_age_init", 9.7))
        mh_init = float(getattr(self.params.P, "iso_mh_init", -0.1))
        if len(available_ages) > 0:
            age_init = float(available_ages[np.argmin(np.abs(available_ages - age_init))])
        if len(available_mhs) > 0:
            mh_init = float(available_mhs[np.argmin(np.abs(available_mhs - mh_init))])

        ax_color = "#222222"
        s_age = Slider(self.figure.add_axes([0.2, 0.15, 0.6, 0.02], facecolor=ax_color),
                       "log Age", available_ages.min(), available_ages.max(),
                       valinit=age_init, valstep=available_ages)
        s_mh = Slider(self.figure.add_axes([0.2, 0.12, 0.6, 0.02], facecolor=ax_color),
                      "[Fe/H]", available_mhs.min(), available_mhs.max(),
                      valinit=mh_init, valstep=available_mhs)
        s_hshift = Slider(self.figure.add_axes([0.2, 0.09, 0.6, 0.02], facecolor=ax_color),
                          self.extinction_label, -0.1, 0.8,
                          valinit=float(getattr(self.params.P, "iso_eg_r_init", 0.0033)),
                          valstep=0.0001)
        s_vshift = Slider(self.figure.add_axes([0.2, 0.06, 0.6, 0.02], facecolor=ax_color),
                          "Dist. Mod", 5.0, 20.0,
                          valinit=float(getattr(self.params.P, "iso_dm_init", 9.46)),
                          valstep=0.01)

        for s in (s_age, s_mh, s_hshift, s_vshift):
            s.label.set_color("white")
            s.valtext.set_color("white")

        resetax = self.figure.add_axes([0.85, 0.01, 0.1, 0.04], facecolor="#111111")
        button = Button(resetax, "Reset", color="#333333", hovercolor="#444444")
        button.label.set_color("white")

        self.s_age = s_age
        self.s_mh = s_mh
        self.s_hshift = s_hshift
        self.s_vshift = s_vshift
        self.reset_button = button

        def update(_):
            age, mh = s_age.val, s_mh.val
            h_s, v_s = s_hshift.val, s_vshift.val

            # Keep current manual values in sync for subsequent auto-fit runs.
            self.params.P.iso_age_init = float(age)
            self.params.P.iso_mh_init = float(mh)
            self.params.P.iso_eg_r_init = float(h_s)
            self.params.P.iso_dm_init = float(v_s)

            new_gr, new_g = get_iso_points(age, mh, h_s, v_s)

            if len(new_gr) > 0:
                iso_teff = teff_from_gr(new_gr)
                sc_iso.set_offsets(np.c_[new_gr, new_g])
                sc_iso.set_array(iso_teff)
            else:
                sc_iso.set_offsets(np.c_[[np.nan], [np.nan]].T)
                sc_iso.set_array(np.array([np.nan]))

            if len(new_gr) > 0 and len(obs_pts) > 0:
                iso_pts = np.c_[new_gr, new_g]
                tree = cKDTree(iso_pts)
                dist, _ = tree.query(obs_pts)

                res_scat.set_offsets(np.c_[obs_mag, dist])
                ax_res.set_xlim(ax_cmd.get_ylim())
                ax_res.set_ylim(0, np.percentile(dist, 95))

                ax_hist.clear()
                style_axis_dark(ax_hist)
                hi = np.percentile(dist, 98)
                ax_hist.hist(dist, bins=30, range=(0, hi), color="deepskyblue", edgecolor="white", alpha=0.75)
                ax_hist.set_title(f"Mean Res: {np.mean(dist):.4f}", color="white")
            else:
                ax_hist.clear()
                style_axis_dark(ax_hist)
                ax_hist.set_title("No isochrone points", color="white")

            ax_cmd.set_title(f"Age: 10^{age:.2f} | [Fe/H]: {mh:.2f} | DM: {v_s:.2f} | {self.extinction_label}: {h_s:.4f}", color="white")
            self.canvas.draw_idle()

        def reset(_):
            s_age.reset()
            s_mh.reset()
            s_hshift.reset()
            s_vshift.reset()

        s_age.on_changed(update)
        s_mh.on_changed(update)
        s_hshift.on_changed(update)
        s_vshift.on_changed(update)
        button.on_clicked(reset)

        update(None)


class IsochroneModelWindow(StepWindowBase):
    """Step 13: Isochrone Model"""

    def __init__(self, params, file_manager, project_state, main_window):
        self.file_manager = file_manager
        self.viewer = None
        self.iso_path_edit = None
        # Data cache – avoids re-reading files on every slider change
        self._cache_key = None          # (iso_path, csv_path)
        self._cached_df = None          # pd.DataFrame
        self._cached_iso_raw = None     # np.ndarray
        self._cached_iso_file = None    # Path
        self._cc_ax = None              # color-color axes cache
        # ZP perpendicular offset from auto-fit
        self._cc_zp_perp = 0.0
        self._cc_perp_norm = np.array([0.0, 0.0])

        super().__init__(
            step_index=12,
            step_name="Isochrone Model",
            params=params,
            project_state=project_state,
            main_window=main_window
        )

        self.setup_step_ui()
        self.restore_state()

    def setup_step_ui(self):
        info = QLabel("Load isochrone data, explore with sliders, or run automatic fitting.")
        info.setStyleSheet("QLabel { background-color: #E3F2FD; padding: 10px; border-radius: 5px; }")
        self.content_layout.addWidget(info)

        # === File Selection ===
        file_group = QGroupBox("Isochrone File")
        file_layout = QHBoxLayout(file_group)
        self.iso_path_edit = QLineEdit()
        self.iso_path_edit.setPlaceholderText("Select iso_data.dat")
        file_layout.addWidget(self.iso_path_edit)
        btn_browse = QPushButton("Browse")
        btn_browse.clicked.connect(self.browse_iso_file)
        file_layout.addWidget(btn_browse)
        self.content_layout.addWidget(file_group)

        # === Filter Selection ===
        filter_group = QGroupBox("Band Selection")
        filter_layout = QHBoxLayout(filter_group)
        filter_layout.addWidget(QLabel("Color (X):"))
        self.color_combo = QComboBox()
        self.color_combo.addItems([f"{a}-{b}" for a, b in ALLOWED_COLOR_PAIRS])
        self.color_combo.setCurrentIndex(0)  # g-r default
        self.color_combo.currentIndexChanged.connect(self._on_band_changed)
        filter_layout.addWidget(self.color_combo)
        filter_layout.addWidget(QLabel("Mag (Y):"))
        self.mag_combo = QComboBox()
        self.mag_combo.addItems(ALLOWED_MAG_BANDS)
        self.mag_combo.setCurrentIndex(0)  # g default
        self.mag_combo.currentIndexChanged.connect(self._on_band_changed)
        filter_layout.addWidget(self.mag_combo)
        filter_layout.addStretch()
        self.content_layout.addWidget(filter_group)

        # === Tabs ===
        self.tabs = QTabWidget()
        self.content_layout.addWidget(self.tabs, stretch=1)

        # --- Tab 0: Color-Color Diagram ---
        cc_tab = QWidget()
        cc_layout = QVBoxLayout(cc_tab)
        cc_layout.setContentsMargins(6, 6, 6, 6)

        # Plot container
        self.cc_fig = Figure(figsize=(9, 7))
        self.cc_canvas = FigureCanvas(self.cc_fig)
        self.cc_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        cc_layout.addWidget(self.cc_canvas, stretch=1)

        # Controls row
        cc_controls = QHBoxLayout()
        cc_controls.addWidget(QLabel("E(B-V):"))
        self.cc_ebv_spin = QDoubleSpinBox()
        self.cc_ebv_spin.setRange(0.0, 2.0)
        self.cc_ebv_spin.setValue(0.0)
        self.cc_ebv_spin.setDecimals(4)
        self.cc_ebv_spin.setSingleStep(0.01)
        self.cc_ebv_spin.valueChanged.connect(self._update_cc_plot)
        cc_controls.addWidget(self.cc_ebv_spin)

        cc_controls.addWidget(QLabel("Ref log(Age):"))
        self.cc_age_spin = QDoubleSpinBox()
        self.cc_age_spin.setRange(6.0, 10.5)
        self.cc_age_spin.setValue(8.5)
        self.cc_age_spin.setDecimals(2)
        self.cc_age_spin.setSingleStep(0.05)
        self.cc_age_spin.valueChanged.connect(self._update_cc_plot)
        cc_controls.addWidget(self.cc_age_spin)

        cc_controls.addWidget(QLabel("Ref [M/H]:"))
        self.cc_mh_spin = QDoubleSpinBox()
        self.cc_mh_spin.setRange(-2.0, 1.0)
        self.cc_mh_spin.setValue(0.0)
        self.cc_mh_spin.setDecimals(2)
        self.cc_mh_spin.setSingleStep(0.05)
        self.cc_mh_spin.valueChanged.connect(self._update_cc_plot)
        cc_controls.addWidget(self.cc_mh_spin)

        cc_layout.addLayout(cc_controls)

        # Action buttons row
        cc_action = QHBoxLayout()
        btn_cc_autofit = QPushButton("Auto Fit E(B-V)")
        btn_cc_autofit.setStyleSheet(
            "QPushButton { background-color: #00897B; color: white; font-weight: bold; padding: 8px 16px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #00796B; }"
        )
        btn_cc_autofit.clicked.connect(self._autofit_ebv)
        cc_action.addWidget(btn_cc_autofit)

        btn_cc_apply = QPushButton("Apply E to Bounds")
        btn_cc_apply.setStyleSheet(
            "QPushButton { background-color: #1565C0; color: white; font-weight: bold; padding: 8px 16px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #0D47A1; }"
        )
        btn_cc_apply.clicked.connect(self._apply_ebv_to_bounds)
        cc_action.addWidget(btn_cc_apply)

        self.cc_result_label = QLabel("")
        self.cc_result_label.setStyleSheet("QLabel { color: #333; font-weight: bold; }")
        cc_action.addWidget(self.cc_result_label)
        cc_action.addStretch()
        cc_layout.addLayout(cc_action)

        self.cc_tab_index = self.tabs.addTab(cc_tab, "Color-Color")

        # --- Tab 1: Auto Fit ---
        fit_tab = QWidget()
        fit_layout = QVBoxLayout(fit_tab)

        # Bounds configuration
        bounds_group = QGroupBox("Parameter Bounds")
        bounds_form = QFormLayout(bounds_group)

        # log(Age) bounds - M38 is ~200-300 Myr (log age ~8.3-8.5)
        age_row = QHBoxLayout()
        self.age_min = QDoubleSpinBox()
        self.age_min.setRange(6.0, 10.5)
        self.age_min.setValue(8.0)
        self.age_min.setDecimals(1)
        self.age_min.setSingleStep(0.1)
        age_row.addWidget(QLabel("min:"))
        age_row.addWidget(self.age_min)
        self.age_max = QDoubleSpinBox()
        self.age_max.setRange(6.0, 10.5)
        self.age_max.setValue(9.0)
        self.age_max.setDecimals(1)
        self.age_max.setSingleStep(0.1)
        age_row.addWidget(QLabel("max:"))
        age_row.addWidget(self.age_max)
        bounds_form.addRow("log(Age):", age_row)

        # [M/H] bounds - M38 is near solar
        mh_row = QHBoxLayout()
        self.mh_min = QDoubleSpinBox()
        self.mh_min.setRange(-2.0, 1.0)
        self.mh_min.setValue(-0.3)
        self.mh_min.setDecimals(1)
        self.mh_min.setSingleStep(0.1)
        mh_row.addWidget(QLabel("min:"))
        mh_row.addWidget(self.mh_min)
        self.mh_max = QDoubleSpinBox()
        self.mh_max.setRange(-2.0, 1.0)
        self.mh_max.setValue(0.3)
        self.mh_max.setDecimals(1)
        self.mh_max.setSingleStep(0.1)
        mh_row.addWidget(QLabel("max:"))
        mh_row.addWidget(self.mh_max)
        bounds_form.addRow("[M/H]:", mh_row)

        # (m-M) bounds - M38 is ~1000 pc (DM ~10)
        dm_row = QHBoxLayout()
        self.dm_min = QDoubleSpinBox()
        self.dm_min.setRange(0.0, 20.0)
        self.dm_min.setValue(9.0)
        self.dm_min.setDecimals(1)
        self.dm_min.setSingleStep(0.5)
        dm_row.addWidget(QLabel("min:"))
        dm_row.addWidget(self.dm_min)
        self.dm_max = QDoubleSpinBox()
        self.dm_max.setRange(0.0, 20.0)
        self.dm_max.setValue(12.0)
        self.dm_max.setDecimals(1)
        self.dm_max.setSingleStep(0.5)
        dm_row.addWidget(QLabel("max:"))
        dm_row.addWidget(self.dm_max)
        bounds_form.addRow("(m-M)₀:", dm_row)

        # E(g-r) bounds - M38 has moderate reddening ~0.25
        egr_row = QHBoxLayout()
        self.egr_min = QDoubleSpinBox()
        self.egr_min.setRange(0.0, 1.0)
        self.egr_min.setValue(0.0)
        self.egr_min.setDecimals(2)
        self.egr_min.setSingleStep(0.05)
        egr_row.addWidget(QLabel("min:"))
        egr_row.addWidget(self.egr_min)
        self.egr_max = QDoubleSpinBox()
        self.egr_max.setRange(0.0, 1.0)
        self.egr_max.setValue(0.5)
        self.egr_max.setDecimals(2)
        self.egr_max.setSingleStep(0.05)
        egr_row.addWidget(QLabel("max:"))
        egr_row.addWidget(self.egr_max)
        bounds_form.addRow("E(color):", egr_row)

        # SNR minimum - lowered for more stars
        snr_row = QHBoxLayout()
        self.snr_min_spin = QDoubleSpinBox()
        self.snr_min_spin.setRange(1.0, 100.0)
        self.snr_min_spin.setValue(5.0)
        self.snr_min_spin.setDecimals(1)
        snr_row.addWidget(self.snr_min_spin)
        snr_row.addStretch()
        bounds_form.addRow("Min SNR:", snr_row)

        fit_layout.addWidget(bounds_group)

        # Fitting button (single recommended method)
        btn_group = QGroupBox("Run Fitting")
        btn_layout = QHBoxLayout(btn_group)

        self.btn_autofit = QPushButton("Run Auto Fit (Recommended)\nGlobal + Local refinement")
        self.btn_autofit.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                font-weight: bold;
                padding: 14px 22px;
                font-size: 11pt;
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #F57C00; }
            QPushButton:disabled { background-color: #BDBDBD; }
        """)
        self.btn_autofit.clicked.connect(self.run_fitting)
        btn_layout.addWidget(self.btn_autofit)

        self.btn_gridscan = QPushButton("Run Grid Scan\nExhaustive age/[M/H] search")
        self.btn_gridscan.setStyleSheet("""
            QPushButton {
                background-color: #7B1FA2;
                color: white;
                font-weight: bold;
                padding: 14px 22px;
                font-size: 11pt;
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #6A1B9A; }
            QPushButton:disabled { background-color: #BDBDBD; }
        """)
        self.btn_gridscan.clicked.connect(self.run_grid_scan)
        btn_layout.addWidget(self.btn_gridscan)

        btn_layout.addStretch()

        fit_layout.addWidget(btn_group)
        fit_hint = QLabel("Auto fit is an initial guess tool. Final science fit should be validated with CMD viewer sliders.")
        fit_hint.setStyleSheet("QLabel { color: #546E7A; font-style: italic; }")
        fit_layout.addWidget(fit_hint)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p% - %v")
        fit_layout.addWidget(self.progress_bar)

        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet("QLabel { color: #666; font-style: italic; }")
        fit_layout.addWidget(self.progress_label)

        # Results display
        results_group = QGroupBox("Fit Results")
        results_layout = QVBoxLayout(results_group)
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setStyleSheet("""
            QTextEdit {
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 10pt;
                background-color: #1E1E1E;
                color: #D4D4D4;
                border: 1px solid #3C3C3C;
            }
        """)
        self.results_text.setMinimumHeight(200)
        self.results_text.setPlaceholderText("Fit results will appear here...")
        results_layout.addWidget(self.results_text)

        # Action buttons after fitting
        action_row = QHBoxLayout()
        self.btn_apply = QPushButton("Apply to CMD Viewer")
        self.btn_apply.setEnabled(False)
        self.btn_apply.clicked.connect(self.apply_fit_to_viewer)
        action_row.addWidget(self.btn_apply)

        self.btn_export = QPushButton("Export Results")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.export_fit_results)
        action_row.addWidget(self.btn_export)

        self.btn_membership = QPushButton("Compute Membership")
        self.btn_membership.setEnabled(False)
        self.btn_membership.clicked.connect(self.compute_membership)
        action_row.addWidget(self.btn_membership)

        action_row.addStretch()
        results_layout.addLayout(action_row)

        fit_layout.addWidget(results_group)
        fit_layout.addStretch()

        self.auto_fit_tab_index = self.tabs.addTab(fit_tab, "Auto Fit")

        # --- Tab 2: CMD Viewer (default tab) ---
        manual_tab = QWidget()
        manual_layout = QVBoxLayout(manual_tab)
        manual_layout.setContentsMargins(6, 6, 6, 6)

        self.viewer_container = QWidget()
        self.viewer_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.viewer_layout = QVBoxLayout(self.viewer_container)
        self.viewer_layout.setContentsMargins(0, 0, 0, 0)
        self.viewer_layout.setSpacing(0)
        manual_layout.addWidget(self.viewer_container, stretch=1)
        self.viewer_placeholder = None
        self._show_viewer_placeholder(
            "Select an isochrone file to render CMD + sliders.\n"
            "Auto Fit results can be applied directly to this viewer."
        )

        self.cmd_viewer_tab_index = self.tabs.addTab(manual_tab, "CMD Viewer")
        self.tabs.setCurrentIndex(self.cc_tab_index)

        # --- Log Window ---
        log_row = QHBoxLayout()
        btn_log = QPushButton("Open Log")
        btn_log.setStyleSheet("QPushButton { background-color: #607D8B; color: white; font-weight: bold; padding: 8px 15px; }")
        btn_log.clicked.connect(self.show_log_window)
        log_row.addWidget(btn_log)
        log_row.addStretch()
        self.content_layout.addLayout(log_row)

        self.log_window = QWidget(self, Qt.Window)
        self.log_window.setWindowTitle("Isochrone Log")
        self.log_window.resize(700, 350)
        log_layout = QVBoxLayout(self.log_window)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("QTextEdit { font-family: monospace; font-size: 9pt; }")
        log_layout.addWidget(self.log_text)

        # Internal state
        self.fitter: Optional[IsochroneFitterV2] = None
        self.fit_result: Optional[FitResult] = None
        self.fit_worker: Optional[FitWorker] = None
        self.grid_scan_worker: Optional[GridScanWorker] = None
        self.grid_scan_result: Optional[GridScanResult] = None
        self.heatmap_tab_index: Optional[int] = None
        self.cmd_df: Optional[pd.DataFrame] = None

    def log(self, message: str):
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")

    def _get_iso_path(self) -> str:
        iso_path = ""
        if self.iso_path_edit is not None:
            iso_path = self.iso_path_edit.text().strip()
        if not iso_path:
            iso_path = str(getattr(self.params.P, "iso_file_path", ""))
        return iso_path

    def _get_band_config(self):
        """Return current band selection as a dict."""
        color_text = self.color_combo.currentText()  # e.g. "g-r"
        parts = color_text.split("-")
        b1, b2 = parts[0].strip(), parts[1].strip()
        bm = self.mag_combo.currentText().strip()
        return {
            "band_color": (b1, b2),
            "band_mag": bm,
            "iso_col_1": SDSS_ISO_COL.get(b1, 29),
            "iso_col_2": SDSS_ISO_COL.get(b2, 30),
            "iso_col_mag": SDSS_ISO_COL.get(bm, SDSS_ISO_COL.get(b1, 29)),
            "R_band1": SDSS_R.get(b1, 3.303),
            "R_band2": SDSS_R.get(b2, 2.285),
            "R_mag": SDSS_R.get(bm, SDSS_R.get(b1, 3.303)),
        }

    def _on_band_changed(self):
        """Refresh CMD viewer when band selection changes."""
        if self._get_iso_path():
            self.refresh_cmd_viewer(show_error=False)

    def _load_cmd_and_iso_data(self, show_error=True):
        iso_path = self._get_iso_path()
        if not iso_path:
            if show_error:
                QMessageBox.warning(self, "Missing File", "Select an isochrone data file first")
            return None, None, None
        iso_file = Path(iso_path)
        if not iso_file.exists():
            if show_error:
                QMessageBox.warning(self, "Missing File", f"Isochrone file not found: {iso_file}")
            return None, None, None

        input_dir = step11_dir(self.params.P.result_dir)
        if not input_dir.exists():
            input_dir = self.params.P.result_dir
        wide_path = input_dir / "median_by_ID_filter_wide_cmd.csv"
        if not wide_path.exists():
            wide_path = input_dir / "median_by_ID_filter_wide.csv"
        if not wide_path.exists():
            if show_error:
                QMessageBox.warning(self, "Missing Data", "CMD wide CSV not found")
            return None, None, None

        # ---- cache check ----
        cache_key = (str(iso_file), str(wide_path))
        if (
            self._cache_key == cache_key
            and self._cached_df is not None
            and self._cached_iso_raw is not None
        ):
            return self._cached_df, self._cached_iso_raw, self._cached_iso_file

        try:
            df = pd.read_csv(wide_path)
        except Exception as e:
            if show_error:
                QMessageBox.critical(self, "Error", f"Failed to load CMD data: {e}")
            return None, None, None

        try:
            iso_raw = np.genfromtxt(iso_file, comments="#")
            iso_raw = iso_raw[~np.isnan(iso_raw).any(axis=1)]
            if iso_raw.size == 0:
                if show_error:
                    QMessageBox.warning(self, "Data Error", "Isochrone file is empty")
                return None, None, None
        except Exception as e:
            if show_error:
                QMessageBox.critical(self, "Error", f"Failed to parse isochrone file: {e}")
            return None, None, None

        # ---- store in cache ----
        self._cache_key = cache_key
        self._cached_df = df
        self._cached_iso_raw = iso_raw
        self._cached_iso_file = iso_file

        return df, iso_raw, iso_file

    def _invalidate_cache(self):
        """Force reload on next access (call after file change)."""
        self._cache_key = None
        self._cached_df = None
        self._cached_iso_raw = None
        self._cached_iso_file = None
        self._cc_ax = None  # force full redraw of color-color plot

    def _show_viewer_placeholder(self, message: str):
        self._clear_viewer_widget()
        placeholder = QLabel(message)
        placeholder.setAlignment(Qt.AlignCenter)
        placeholder.setStyleSheet(
            "QLabel { color: #607D8B; font-size: 11pt; border: 1px dashed #B0BEC5; padding: 24px; }"
        )
        self.viewer_layout.addWidget(placeholder, stretch=1)
        self.viewer_placeholder = placeholder

    def _clear_viewer_widget(self):
        while self.viewer_layout.count():
            item = self.viewer_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self.viewer = None

    def refresh_cmd_viewer(self, show_error=True) -> bool:
        df, iso_raw, iso_file = self._load_cmd_and_iso_data(show_error=show_error)
        if df is None or iso_raw is None:
            self._show_viewer_placeholder(
                "CMD viewer not ready.\nSelect an isochrone file and ensure Step 12 output exists."
            )
            return False

        self._clear_viewer_widget()
        bc = self._get_band_config()
        viewer = IsochroneViewerWindow(
            df, iso_raw, self.params, self.viewer_container, embedded=True,
            band_mag=bc["band_mag"], band_color=bc["band_color"],
            iso_col_1=bc["iso_col_1"], iso_col_2=bc["iso_col_2"],
        )
        self.viewer_layout.addWidget(viewer, stretch=1)
        self.viewer = viewer
        self.log(f"CMD viewer updated: {iso_file.name}  bands={bc['band_color'][0]}-{bc['band_color'][1]}, mag={bc['band_mag']}")
        return True

    def _get_fit_initial_guess(self):
        """Build fit initial guess from current CMD slider state when available."""
        values = None
        if self.viewer is not None and hasattr(self.viewer, "current_slider_values"):
            try:
                values = self.viewer.current_slider_values()
            except Exception:
                values = None

        if not values:
            values = {
                "log_age": float(getattr(self.params.P, "iso_age_init", 9.7)),
                "metallicity": float(getattr(self.params.P, "iso_mh_init", -0.1)),
                "distance_mod": float(getattr(self.params.P, "iso_dm_init", 9.46)),
                "extinction_gr": float(getattr(self.params.P, "iso_eg_r_init", 0.0033)),
            }

        return np.array(
            [
                values["log_age"],
                values["metallicity"],
                values["distance_mod"],
                values["extinction_gr"],
            ],
            dtype=float,
        )

    # =========================================================================
    # Color-Color Diagram
    # =========================================================================

    def _update_cc_plot(self):
        """Draw/update the color-color diagram (cached – fast on slider changes)."""
        df, iso_raw, iso_file = self._load_cmd_and_iso_data(show_error=False)
        if df is None or iso_raw is None:
            return

        # ---- First draw: full rebuild ----
        need_full = not hasattr(self, "_cc_ax") or self._cc_ax is None

        if need_full:
            self.cc_fig.clear()
            ax = self.cc_fig.add_subplot(111)
            self._cc_ax = ax

            # Observed colors: (g-r) vs (r-i)
            def _get(band):
                s = f"mag_std_{band}"
                i = f"mag_inst_{band}"
                if s in df.columns:
                    return df[s].to_numpy(float)
                if i in df.columns:
                    return df[i].to_numpy(float)
                return None

            obs_g = _get("g")
            obs_r = _get("r")
            obs_i = _get("i")
            if obs_g is None or obs_r is None or obs_i is None:
                ax.text(0.5, 0.5, "Need g, r, i bands for color-color diagram",
                        ha='center', va='center', transform=ax.transAxes, fontsize=12)
                self.cc_canvas.draw()
                return

            obs_gr = obs_g - obs_r
            obs_ri = obs_r - obs_i
            mask = np.isfinite(obs_gr) & np.isfinite(obs_ri)
            obs_gr, obs_ri = obs_gr[mask], obs_ri[mask]

            ax.scatter(obs_ri, obs_gr, s=3, alpha=0.6, color="gray", label="Observed", zorder=1)
            ax.set_xlabel("(r - i)", fontsize=12)
            ax.set_ylabel("(g - r)", fontsize=12)
            ax.grid(True, linestyle=":", alpha=0.4)

            # Placeholder artists for isochrone lines (updated below)
            self._cc_line_unred, = ax.plot([], [], 'c-', lw=1.5, alpha=0.7, label="Unreddened isochrone", zorder=2)
            self._cc_line_red, = ax.plot([], [], 'r-', lw=2.0, alpha=0.9, zorder=3)
            self._cc_arrow = None
            ax.legend(loc="upper left", fontsize=9)
            self.cc_fig.tight_layout()

        ax = self._cc_ax

        # ---- Update isochrone lines (fast path) ----
        ref_age = self.cc_age_spin.value()
        ref_mh = self.cc_mh_spin.value()
        col_g = SDSS_ISO_COL["g"]
        col_r = SDSS_ISO_COL["r"]
        col_i = SDSS_ISO_COL["i"]

        available_ages = np.unique(iso_raw[:, 2])
        available_mhs = np.unique(iso_raw[:, 1])
        nearest_age = float(available_ages[np.argmin(np.abs(available_ages - ref_age))])
        nearest_mh = float(available_mhs[np.argmin(np.abs(available_mhs - ref_mh))])

        iso_mask = (iso_raw[:, 2] == nearest_age) & (iso_raw[:, 1] == nearest_mh)
        iso_sub = iso_raw[iso_mask]

        if len(iso_sub) > 0:
            iso_g = iso_sub[:, col_g]
            iso_r = iso_sub[:, col_r]
            iso_i = iso_sub[:, col_i]
            iso_gr0 = iso_g - iso_r
            iso_ri0 = iso_r - iso_i

            sort_idx = np.argsort(iso_ri0)
            iso_gr0 = iso_gr0[sort_idx]
            iso_ri0 = iso_ri0[sort_idx]

            self._cc_line_unred.set_data(iso_ri0, iso_gr0)

            ebv = self.cc_ebv_spin.value()
            R_g, R_r, R_i = SDSS_R["g"], SDSS_R["r"], SDSS_R["i"]
            dEgr = (R_g - R_r) * ebv
            dEri = (R_r - R_i) * ebv

            self._cc_line_red.set_data(iso_ri0 + dEri, iso_gr0 + dEgr)
            self._cc_line_red.set_label(f"Reddened E(B-V)={ebv:.4f}")

            # Reddening vector arrow
            if self._cc_arrow is not None:
                self._cc_arrow.remove()
                self._cc_arrow = None
            mid = len(iso_ri0) // 3
            if mid < len(iso_ri0) and ebv > 1e-6:
                self._cc_arrow = ax.annotate(
                    "", xy=(iso_ri0[mid] + dEri, iso_gr0[mid] + dEgr),
                    xytext=(iso_ri0[mid], iso_gr0[mid]),
                    arrowprops=dict(arrowstyle="->", color="yellow", lw=2))

            self.cc_result_label.setText(
                f"E(B-V)={ebv:.4f}  |  E(g-r)={dEgr:.4f}  |  E(r-i)={dEri:.4f}"
            )

        ax.set_title(f"Color-Color Diagram  |  ref: log(Age)={nearest_age:.2f}, [M/H]={nearest_mh:.2f}")
        ax.legend(loc="upper left", fontsize=9)
        self.cc_canvas.draw_idle()

    def _autofit_ebv(self):
        """Auto-fit E(B-V) by sliding the isochrone along the reddening vector
        and minimising the median distance to the observed stellar locus."""
        df, iso_raw, _ = self._load_cmd_and_iso_data(show_error=True)
        if df is None or iso_raw is None:
            return

        def _get(band):
            s = f"mag_std_{band}"
            i = f"mag_inst_{band}"
            if s in df.columns:
                return df[s].to_numpy(float)
            if i in df.columns:
                return df[i].to_numpy(float)
            return None

        obs_g, obs_r, obs_i = _get("g"), _get("r"), _get("i")
        if obs_g is None or obs_r is None or obs_i is None:
            QMessageBox.warning(self, "Missing Data", "Need g, r, i bands for color-color fitting")
            return

        obs_gr = obs_g - obs_r
        obs_ri = obs_r - obs_i
        mask = np.isfinite(obs_gr) & np.isfinite(obs_ri)
        obs_gr, obs_ri = obs_gr[mask], obs_ri[mask]

        # Get reference isochrone locus
        ref_age = self.cc_age_spin.value()
        ref_mh = self.cc_mh_spin.value()
        available_ages = np.unique(iso_raw[:, 2])
        available_mhs = np.unique(iso_raw[:, 1])
        nearest_age = float(available_ages[np.argmin(np.abs(available_ages - ref_age))])
        nearest_mh = float(available_mhs[np.argmin(np.abs(available_mhs - ref_mh))])

        iso_mask = (iso_raw[:, 2] == nearest_age) & (iso_raw[:, 1] == nearest_mh)
        iso_sub = iso_raw[iso_mask]
        if len(iso_sub) < 10:
            QMessageBox.warning(self, "No Data", "Not enough isochrone points for this age/[M/H]")
            return

        iso_g = iso_sub[:, SDSS_ISO_COL["g"]]
        iso_r = iso_sub[:, SDSS_ISO_COL["r"]]
        iso_i = iso_sub[:, SDSS_ISO_COL["i"]]
        iso_gr0 = iso_g - iso_r
        iso_ri0 = iso_r - iso_i

        R_g, R_r, R_i = SDSS_R["g"], SDSS_R["r"], SDSS_R["i"]
        dgr_per_ebv = R_g - R_r   # E(g-r) per E(B-V) = 1.018
        dri_per_ebv = R_r - R_i   # E(r-i) per E(B-V) = 0.587

        obs_pts = np.column_stack([obs_ri, obs_gr])

        # --- Slide isochrone along reddening vector ---
        # Metric: Gaussian kernel overlap (obs → iso)
        sigma = 0.05
        ebv_trials = np.linspace(-0.1, 1.5, 800)
        scores = np.empty(len(ebv_trials))
        for j, ebv in enumerate(ebv_trials):
            shifted = np.column_stack([iso_ri0 + dri_per_ebv * ebv,
                                       iso_gr0 + dgr_per_ebv * ebv])
            iso_tree = cKDTree(shifted)
            dist, _ = iso_tree.query(obs_pts)
            scores[j] = np.sum(np.exp(-0.5 * (dist / sigma) ** 2))

        best_idx = int(np.argmax(scores))
        best_ebv = float(ebv_trials[best_idx])
        best_ebv = max(0.0, best_ebv)

        # Reset ZP offset (manual visual adjustment only)
        self._cc_zp_perp = 0.0
        self._cc_perp_norm = np.array([0.0, 0.0])

        self.cc_ebv_spin.setValue(best_ebv)

        # Warn if reddening is small (calibration errors may dominate)
        warning = ""
        if best_ebv < 0.1:
            warning = " ⚠ Low E(B-V): result may be unreliable — use literature value"
        self.log(f"Color-color auto-fit: E(B-V)={best_ebv:.4f}, "
                 f"E(g-r)={dgr_per_ebv * best_ebv:.4f}, "
                 f"E(r-i)={dri_per_ebv * best_ebv:.4f}{warning}")

    def _apply_ebv_to_bounds(self):
        """Apply the determined E(B-V) to the fitting bounds as E(color)."""
        ebv = self.cc_ebv_spin.value()
        bc = self._get_band_config()
        R1 = bc["R_band1"]
        R2 = bc["R_band2"]
        e_color = (R1 - R2) * ebv

        # Set tight bounds: ±20% or ±0.02, whichever is larger
        margin = max(e_color * 0.2, 0.02)
        lo = max(0.0, e_color - margin)
        hi = e_color + margin

        self.egr_min.setValue(round(lo, 4))
        self.egr_max.setValue(round(hi, 4))

        b1, b2 = bc["band_color"]
        self.log(f"Applied E(B-V)={ebv:.4f} → E({b1}-{b2})={e_color:.4f}, bounds=[{lo:.4f}, {hi:.4f}]")
        QMessageBox.information(
            self, "Applied",
            f"E(B-V) = {ebv:.4f}\n"
            f"E({b1}-{b2}) = {e_color:.4f}\n\n"
            f"Extinction bounds set to [{lo:.4f}, {hi:.4f}]"
        )

    # =========================================================================
    # Fitting Helpers
    # =========================================================================

    def _extract_cmd_columns(self):
        """Extract color/mag arrays from self.cmd_df using current band selection.
        Returns (color, mag, color_err, mag_err) or None on failure.
        """
        bc = self._get_band_config()
        b1, b2 = bc["band_color"]
        bm = bc["band_mag"]

        def _get_col(band, prefer_std=True):
            std = f"mag_std_{band}"
            inst = f"mag_inst_{band}"
            err = f"mag_inst_err_{band}"
            if prefer_std and std in self.cmd_df.columns:
                vals = self.cmd_df[std].to_numpy(float)
            elif inst in self.cmd_df.columns:
                vals = self.cmd_df[inst].to_numpy(float)
            else:
                return None, None
            errs = self.cmd_df.get(err, pd.Series(np.full(len(vals), 0.01))).to_numpy(float)
            return vals, errs

        v1, e1 = _get_col(b1)
        v2, e2 = _get_col(b2)
        if v1 is None or v2 is None:
            QMessageBox.critical(self, "Error", f"CMD data missing {b1}/{b2} magnitude columns")
            return None
        vm, em = _get_col(bm)
        if vm is None:
            vm, em = v1.copy(), e1.copy()

        color = v1 - v2
        color_err = np.sqrt(e1**2 + e2**2)
        return color, vm, color_err, em

    def _create_fitter(self, iso_path: str):
        """Create IsochroneFitterV2 with current band config."""
        bc = self._get_band_config()
        col_mh = int(getattr(self.params.P, "iso_col_mh", 1))
        col_age = int(getattr(self.params.P, "iso_col_age", 2))
        fit_fraction = float(getattr(self.params.P, "iso_fit_fraction", 0.85))

        self.fitter = IsochroneFitterV2(
            iso_path,
            col_mh=col_mh,
            col_age=col_age,
            col_g=bc["iso_col_1"],
            col_r=bc["iso_col_2"],
            col_mag=bc["iso_col_mag"],
            fit_fraction=fit_fraction,
            R_band1=bc["R_band1"],
            R_band2=bc["R_band2"],
            R_mag=bc["R_mag"],
        )

    # =========================================================================
    # Fitting Methods
    # =========================================================================

    def run_fitting(self):
        """Run the single recommended auto-fit pipeline."""

        cmd_df, _, iso_file = self._load_cmd_and_iso_data(show_error=True)
        if cmd_df is None or iso_file is None:
            return
        self.cmd_df = cmd_df

        extracted = self._extract_cmd_columns()
        if extracted is None:
            return
        color, mag, color_err, mag_err = extracted

        try:
            self._create_fitter(str(iso_file))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load isochrone: {e}")
            return

        # Get bounds from UI
        bounds = FitBounds(
            log_age=(self.age_min.value(), self.age_max.value()),
            metallicity=(self.mh_min.value(), self.mh_max.value()),
            distance_mod=(self.dm_min.value(), self.dm_max.value()),
            extinction_gr=(self.egr_min.value(), self.egr_max.value())
        )

        snr_min = self.snr_min_spin.value()

        # Disable buttons during fitting
        self._set_fitting_ui_enabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Starting autofit...")

        self.log(f"Starting autofit (hessian mode, multi-start) with {len(mag)} stars...")

        # Run in background thread
        initial_guess = self._get_fit_initial_guess()
        n_starts = int(getattr(self.params.P, "iso_autofit_starts", 6))
        de_maxiter = int(getattr(self.params.P, "iso_autofit_de_maxiter", 120))
        local_maxiter = int(getattr(self.params.P, "iso_autofit_local_maxiter", 200))
        fit_seed = int(getattr(self.params.P, "iso_autofit_seed", 42))
        em_iters = int(getattr(self.params.P, "iso_autofit_em_iters", 3))
        fit_kwargs = {
            "de_maxiter": de_maxiter,
            "local_maxiter": local_maxiter,
            "n_starts": n_starts,
            "seed": fit_seed,
            "initial_guess": initial_guess,
            "em_iters": em_iters,
            "membership_sigma_scale": float(getattr(self.params.P, "iso_autofit_membership_sigma_scale", 1.3)),
            "weight_floor": float(getattr(self.params.P, "iso_autofit_weight_floor", 0.05)),
        }
        self.log(
            "Initial guess | "
            f"logAge={initial_guess[0]:.3f}, [M/H]={initial_guess[1]:.3f}, "
            f"DM={initial_guess[2]:.3f}, E(g-r)={initial_guess[3]:.4f}"
        )
        self.log(
            f"AutoFit settings | n_starts={n_starts}, de_maxiter={de_maxiter}, "
            f"local_maxiter={local_maxiter}, em_iters={em_iters}, seed={fit_seed}"
        )
        self.fit_worker = FitWorker(
            self.fitter, color, mag, color_err, mag_err,
            FitMode.HESSIAN, bounds, snr_min,
            fit_kwargs=fit_kwargs
        )
        self.fit_worker.progress.connect(self._on_fit_progress)
        self.fit_worker.finished.connect(self._on_fit_complete)
        self.fit_worker.start()

    def _set_fitting_ui_enabled(self, enabled: bool):
        """Enable/disable fitting UI elements"""
        self.btn_autofit.setEnabled(enabled)
        self.btn_gridscan.setEnabled(enabled)

    def _on_fit_progress(self, progress: float, message: str):
        """Update progress bar"""
        pct = int(np.clip(progress, 0.0, 1.0) * 100.0)
        self.progress_bar.setValue(pct)
        self.progress_bar.setFormat(f"{pct}%")
        self.progress_label.setText(message)

    def _on_fit_complete(self, result):
        """Handle fitting completion"""
        self.progress_bar.setVisible(False)
        self._set_fitting_ui_enabled(True)

        if isinstance(result, Exception):
            self.log(f"Fitting failed: {result}")
            QMessageBox.critical(self, "Fitting Error", str(result))
            self.progress_label.setText("Fitting failed")
            return

        self.fit_result = result
        self.log(f"Fitting complete in {result.elapsed_sec:.2f} sec")
        if not result.converged:
            self.log("Auto fit did not fully converge; use CMD Viewer sliders for manual refinement.")
        self.progress_label.setText(f"Complete in {result.elapsed_sec:.2f} sec")

        # Display results
        self.results_text.setPlainText(result.summary())

        # Enable action buttons
        self.btn_apply.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.btn_membership.setEnabled(True)

    def apply_fit_to_viewer(self):
        """Apply fit results to CMD viewer parameters and refresh the viewer."""
        if self.fit_result is None:
            return

        # Store in params for viewer to use
        self.params.P.iso_age_init = self.fit_result.log_age
        self.params.P.iso_mh_init = self.fit_result.metallicity
        self.params.P.iso_dm_init = self.fit_result.distance_mod
        self.params.P.iso_eg_r_init = self.fit_result.extinction_gr

        self.save_state()
        self.persist_params()
        self.log("Applied fit results to parameters")
        if self.refresh_cmd_viewer(show_error=True):
            self.tabs.setCurrentIndex(self.cmd_viewer_tab_index)

    # -----------------------------------------------------------------
    # Grid Scan
    # -----------------------------------------------------------------

    def run_grid_scan(self):
        """Run exhaustive grid scan over all (age, mh) pairs within bounds."""
        cmd_df, _, iso_file = self._load_cmd_and_iso_data(show_error=True)
        if cmd_df is None or iso_file is None:
            return
        self.cmd_df = cmd_df

        extracted = self._extract_cmd_columns()
        if extracted is None:
            return
        color, mag, color_err, mag_err = extracted

        try:
            self._create_fitter(str(iso_file))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load isochrone: {e}")
            return

        bounds = FitBounds(
            log_age=(self.age_min.value(), self.age_max.value()),
            metallicity=(self.mh_min.value(), self.mh_max.value()),
            distance_mod=(self.dm_min.value(), self.dm_max.value()),
            extinction_gr=(self.egr_min.value(), self.egr_max.value()),
        )
        snr_min = self.snr_min_spin.value()

        # Count grid cells
        n_ages = int(np.sum(
            (self.fitter.ages >= bounds.log_age[0]) & (self.fitter.ages <= bounds.log_age[1])
        ))
        n_mhs = int(np.sum(
            (self.fitter.metallicities >= bounds.metallicity[0])
            & (self.fitter.metallicities <= bounds.metallicity[1])
        ))

        self._set_fitting_ui_enabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_label.setText(f"Grid scan: {n_ages} x {n_mhs} = {n_ages * n_mhs} cells...")
        self.log(f"Starting grid scan: {n_ages} ages x {n_mhs} [M/H] = {n_ages * n_mhs} cells")

        initial_guess = self._get_fit_initial_guess()
        fit_kwargs = {
            "initial_dm": float(initial_guess[2]),
            "initial_egr": float(initial_guess[3]),
            "local_maxiter": int(getattr(self.params.P, "iso_gridscan_local_maxiter", 50)),
        }

        self.grid_scan_worker = GridScanWorker(
            self.fitter, color, mag, color_err, mag_err,
            bounds, snr_min, fit_kwargs=fit_kwargs,
        )
        self.grid_scan_worker.progress.connect(self._on_fit_progress)
        self.grid_scan_worker.finished.connect(self._on_grid_scan_complete)
        self.grid_scan_worker.start()

    def _on_grid_scan_complete(self, result):
        """Handle grid scan completion."""
        self.progress_bar.setVisible(False)
        self._set_fitting_ui_enabled(True)

        if isinstance(result, Exception):
            self.log(f"Grid scan failed: {result}")
            QMessageBox.critical(self, "Grid Scan Error", str(result))
            self.progress_label.setText("Grid scan failed")
            return

        self.grid_scan_result = result
        self.fit_result = result.best_fit

        self.log(f"Grid scan complete: {result.n_evaluated} cells in {result.elapsed_sec:.2f} sec")
        self.progress_label.setText(f"Grid scan: {result.n_evaluated} cells in {result.elapsed_sec:.2f} sec")

        self.results_text.setPlainText(result.summary())

        self.btn_apply.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.btn_membership.setEnabled(True)

        self._show_grid_heatmap(result)

    def _show_grid_heatmap(self, grid_result: GridScanResult):
        """Create or update the Grid Heatmap tab."""
        # Remove old heatmap tab if exists
        if self.heatmap_tab_index is not None:
            for i in range(self.tabs.count()):
                if self.tabs.tabText(i) == "Grid Heatmap":
                    self.tabs.removeTab(i)
                    break
            self.heatmap_tab_index = None

        heatmap_widget = QWidget()
        heatmap_layout = QVBoxLayout(heatmap_widget)
        heatmap_layout.setContentsMargins(5, 5, 5, 5)

        fig = Figure(figsize=(10, 7))
        canvas = FigureCanvas(fig)
        canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        heatmap_layout.addWidget(canvas, stretch=1)

        ax = fig.add_subplot(111)

        chi2 = grid_result.chi2_grid
        ages = grid_result.grid_ages
        mhs = grid_result.grid_mhs

        finite_vals = chi2[np.isfinite(chi2)]
        if len(finite_vals) == 0:
            ax.text(0.5, 0.5, "No valid grid cells", ha='center', va='center',
                    transform=ax.transAxes, fontsize=14)
            canvas.draw()
            self.heatmap_tab_index = self.tabs.addTab(heatmap_widget, "Grid Heatmap")
            self.tabs.setCurrentIndex(self.heatmap_tab_index)
            return

        vmin = np.nanmin(chi2)
        vmax = np.nanpercentile(finite_vals, 95)

        masked_chi2 = np.ma.masked_invalid(chi2)

        im = ax.pcolormesh(
            mhs, ages, masked_chi2,
            cmap='viridis_r',
            vmin=vmin, vmax=vmax,
            shading='nearest',
        )
        cbar = fig.colorbar(im, ax=ax, label='$\\chi^2$')

        # Mark best-fit point
        best_idx = np.unravel_index(np.nanargmin(chi2), chi2.shape)
        best_age = ages[best_idx[0]]
        best_mh = mhs[best_idx[1]]
        best_chi2 = chi2[best_idx[0], best_idx[1]]
        best_dm = grid_result.dm_grid[best_idx[0], best_idx[1]]
        best_egr = grid_result.egr_grid[best_idx[0], best_idx[1]]

        ax.plot(best_mh, best_age, 'r*', markersize=18, markeredgecolor='white',
                markeredgewidth=1.2, zorder=10,
                label=f'Best: age={best_age:.2f}, [M/H]={best_mh:.2f}')

        ax.set_xlabel('[M/H]', fontsize=12)
        ax.set_ylabel('log(Age)', fontsize=12)
        bc = self._get_band_config()
        elabel = f"E({bc['band_color'][0]}-{bc['band_color'][1]})"
        ax.set_title(
            f'Grid Scan  |  best $\\chi^2$={best_chi2:.1f}'
            f'  DM={best_dm:.2f}  {elabel}={best_egr:.4f}',
            fontsize=11,
        )
        ax.legend(loc='upper right', fontsize=10)

        fig.tight_layout()
        canvas.draw()

        self.heatmap_tab_index = self.tabs.addTab(heatmap_widget, "Grid Heatmap")
        self.tabs.setCurrentIndex(self.heatmap_tab_index)

    def export_fit_results(self):
        """Export fitting results to files"""
        if self.fit_result is None:
            return

        result_dir = step13_dir(self.params.P.result_dir)
        result_dir.mkdir(parents=True, exist_ok=True)

        # Export summary text
        summary_path = result_dir / "isochrone_fit_result.txt"
        with open(summary_path, 'w') as f:
            f.write(self.fit_result.summary())

        # Export as JSON
        import json
        json_path = result_dir / "isochrone_fit_result.json"
        fit_dict = {
            "log_age": self.fit_result.log_age,
            "log_age_err": self.fit_result.log_age_err,
            "metallicity": self.fit_result.metallicity,
            "metallicity_err": self.fit_result.metallicity_err,
            "distance_mod": self.fit_result.distance_mod,
            "distance_mod_err": self.fit_result.distance_mod_err,
            "extinction_gr": self.fit_result.extinction_gr,
            "extinction_gr_err": self.fit_result.extinction_gr_err,
            "age_gyr": self.fit_result.age_gyr,
            "distance_pc": self.fit_result.distance_pc,
            "chi2": self.fit_result.chi2,
            "reduced_chi2": self.fit_result.reduced_chi2,
            "n_stars": self.fit_result.n_stars,
            "fit_mode": self.fit_result.fit_mode,
            "elapsed_sec": self.fit_result.elapsed_sec
        }
        with open(json_path, 'w') as f:
            json.dump(fit_dict, f, indent=2)

        export_paths = [str(summary_path), str(json_path)]

        # Export grid scan data if available
        if self.grid_scan_result is not None:
            gs = self.grid_scan_result
            npz_path = result_dir / "grid_scan_chi2.npz"
            np.savez(
                npz_path,
                chi2_grid=gs.chi2_grid,
                dm_grid=gs.dm_grid,
                egr_grid=gs.egr_grid,
                grid_ages=gs.grid_ages,
                grid_mhs=gs.grid_mhs,
            )
            export_paths.append(str(npz_path))

        self.log(f"Exported results to {result_dir}")
        QMessageBox.information(
            self, "Exported",
            "Results exported to:\n" + "\n".join(export_paths)
        )

    def compute_membership(self):
        """Compute membership probabilities and save to CSV"""
        if self.fit_result is None or self.fitter is None or self.cmd_df is None:
            return

        # Get CMD data using current band selection
        bc = self._get_band_config()
        b1, b2 = bc["band_color"]
        bm = bc["band_mag"]
        std1, inst1 = f"mag_std_{b1}", f"mag_inst_{b1}"
        std2, inst2 = f"mag_std_{b2}", f"mag_inst_{b2}"
        stdm, instm = f"mag_std_{bm}", f"mag_inst_{bm}"

        v1 = self.cmd_df[std1].to_numpy(float) if std1 in self.cmd_df.columns else self.cmd_df[inst1].to_numpy(float)
        v2 = self.cmd_df[std2].to_numpy(float) if std2 in self.cmd_df.columns else self.cmd_df[inst2].to_numpy(float)
        vm = self.cmd_df[stdm].to_numpy(float) if stdm in self.cmd_df.columns else self.cmd_df.get(instm, pd.Series(v1)).to_numpy(float)
        color = v1 - v2

        # Compute membership
        prob = self.fitter.compute_membership(self.fit_result, color, vm)

        # Add to dataframe
        self.cmd_df["membership_prob"] = prob
        self.cmd_df["is_member"] = prob > 0.5

        n_members = (prob > 0.5).sum()
        n_likely = (prob > 0.8).sum()

        # Save
        result_dir = step13_dir(self.params.P.result_dir)
        result_dir.mkdir(parents=True, exist_ok=True)
        output_path = result_dir / "cmd_with_membership.csv"
        self.cmd_df.to_csv(output_path, index=False)

        self.log(f"Computed membership: {n_members} members (P>0.5), {n_likely} likely (P>0.8)")
        QMessageBox.information(
            self, "Membership Computed",
            f"Membership probabilities computed:\n"
            f"- {n_members} members (P > 0.5)\n"
            f"- {n_likely} likely members (P > 0.8)\n\n"
            f"Saved to: {output_path}"
        )

    def show_log_window(self):
        self.log_window.show()
        self.log_window.raise_()
        self.log_window.activateWindow()

    def browse_iso_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Isochrone File", str(Path.cwd()), "Data Files (*.dat *.txt);;All Files (*.*)")
        if path:
            self.iso_path_edit.setText(path)
            self.params.P.iso_file_path = path
            self._invalidate_cache()
            self.save_state()
            self.persist_params()
            self.refresh_cmd_viewer(show_error=True)
            self._update_cc_plot()
            self.update_navigation_buttons()

    def open_viewer(self):
        if self.refresh_cmd_viewer(show_error=True):
            self.tabs.setCurrentIndex(self.cmd_viewer_tab_index)

    def validate_step(self) -> bool:
        iso_path = ""
        if getattr(self, "iso_path_edit", None) is not None:
            iso_path = self.iso_path_edit.text().strip()
        if not iso_path:
            iso_path = str(getattr(self.params.P, "iso_file_path", ""))
        if not iso_path:
            return False
        if not Path(iso_path).exists():
            return False
        input_dir = step11_dir(self.params.P.result_dir)
        if not input_dir.exists():
            input_dir = self.params.P.result_dir
        return (input_dir / "median_by_ID_filter_wide_cmd.csv").exists() or (input_dir / "median_by_ID_filter_wide.csv").exists()

    def save_state(self):
        state_data = {
            "iso_file_path": self.iso_path_edit.text().strip() or str(getattr(self.params.P, "iso_file_path", "")),
            "iso_age_init": getattr(self.params.P, "iso_age_init", 9.7),
            "iso_mh_init": getattr(self.params.P, "iso_mh_init", -0.1),
            "iso_eg_r_init": getattr(self.params.P, "iso_eg_r_init", 0.0033),
            "iso_dm_init": getattr(self.params.P, "iso_dm_init", 9.46),
            "band_color_idx": self.color_combo.currentIndex(),
            "band_mag_idx": self.mag_combo.currentIndex(),
        }
        self.project_state.store_step_data("isochrone_model", state_data)

    def restore_state(self):
        state_data = self.project_state.get_step_data("isochrone_model")
        if state_data:
            for key, val in state_data.items():
                if hasattr(self.params.P, key):
                    setattr(self.params.P, key, val)
            if state_data.get("iso_file_path"):
                if self.iso_path_edit is not None:
                    self.iso_path_edit.setText(state_data["iso_file_path"])
            if "band_color_idx" in state_data:
                idx = int(state_data["band_color_idx"])
                if 0 <= idx < self.color_combo.count():
                    self.color_combo.setCurrentIndex(idx)
            if "band_mag_idx" in state_data:
                idx = int(state_data["band_mag_idx"])
                if 0 <= idx < self.mag_combo.count():
                    self.mag_combo.setCurrentIndex(idx)
        if self.iso_path_edit is not None and not self.iso_path_edit.text().strip():
            iso_path = str(getattr(self.params.P, "iso_file_path", "") or "")
            if iso_path:
                self.iso_path_edit.setText(iso_path)
        if self._get_iso_path():
            self.refresh_cmd_viewer(show_error=False)
            self._update_cc_plot()
        self.update_navigation_buttons()
