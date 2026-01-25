"""
Step 10: Aperture Overlay
Overlay apertures/annuli and labels on a selected frame.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.visualization import ZScaleInterval
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyArrowPatch

from PyQt5.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QGroupBox, QMessageBox,
    QTextEdit, QDialog, QFormLayout, QDialogButtonBox, QDoubleSpinBox,
    QSpinBox, QCheckBox, QComboBox, QWidget, QSlider, QShortcut
)
from PyQt5.QtGui import QKeySequence
from PyQt5.QtCore import Qt

from .step_window_base import StepWindowBase
from ...utils.step_paths import (
    step1_dir,
    step2_cropped_dir,
    step6_dir,
    step9_dir,
    legacy_step5_refbuild_dir,
    legacy_step7_refbuild_dir,
)


class ApertureOverlayWindow(StepWindowBase):
    """Step 10: Aperture Overlay"""

    def __init__(self, params, file_manager, project_state, main_window):
        self.file_manager = file_manager
        self.file_list = []
        self.use_cropped = False
        self.current_filename = None
        self.image_data = None
        self.header = None
        self.current_file_index = 0
        self.master_df = None
        self.ap_df = None
        self.overlay_loaded = False

        # Matplotlib
        self.figure = None
        self.canvas = None
        self.ax = None
        self._imshow_obj = None
        self._normalized_cache = None
        self.xlim_original = None
        self.ylim_original = None
        self.panning = False
        self.pan_start = None
        self.cursor_x = None
        self.cursor_y = None
        self._pending_xlim = None
        self._pending_ylim = None
        self._file_filter_map = {}
        self._file_frame_key_map = {}
        self._frame_key_map = {}
        self._frame_keys_by_filter = {}
        self._filter_order = []
        self._shortcuts = []

        # Stretch plot window (2D Plot)
        self.stretch_plot_dialog = None
        self.stretch_plot_canvas = None
        self.stretch_plot_ax = None
        self.stretch_plot_fig = None
        self.stretch_plot_info_label = None
        self._stretch_vmin = None
        self._stretch_vmax = None
        self._stretch_data_range = None
        self._stretch_dragging = False
        self._stretch_drag_target = None
        self._stretch_marker_min_line = None
        self._stretch_marker_max_line = None

        super().__init__(
            step_index=9,
            step_name="Aperture Overlay",
            params=params,
            project_state=project_state,
            main_window=main_window
        )

        self.setup_step_ui()
        self.restore_state()

    def setup_step_ui(self):
        info = QLabel(
            "Keyboard: [.] next filter | [[/]] prev/next frame\n"
            "Mouse: Wheel to zoom | Right-click drag to pan"
        )
        info.setStyleSheet("QLabel { background-color: #E3F2FD; padding: 10px; border-radius: 5px; }")
        self.content_layout.addWidget(info)

        control_layout = QHBoxLayout()
        btn_params = QPushButton("Overlay Parameters")
        btn_params.setStyleSheet("QPushButton { background-color: #9C27B0; color: white; font-weight: bold; padding: 8px 15px; }")
        btn_params.clicked.connect(self.open_parameters_dialog)
        control_layout.addWidget(btn_params)

        self.btn_load = QPushButton("Load Overlay")
        self.btn_load.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px 15px; }")
        self.btn_load.clicked.connect(self.load_and_display)

        btn_log = QPushButton("Open Log")
        btn_log.setStyleSheet("QPushButton { background-color: #607D8B; color: white; font-weight: bold; padding: 8px 15px; }")
        btn_log.clicked.connect(self.show_log_window)
        control_layout.addWidget(btn_log)

        control_layout.addStretch()
        self.content_layout.addLayout(control_layout)

        select_group = QGroupBox("Frame Selection")
        select_layout = QHBoxLayout(select_group)
        select_layout.addWidget(QLabel("File:"))
        self.file_combo = QComboBox()
        self.file_combo.currentIndexChanged.connect(self.on_file_changed)
        select_layout.addWidget(self.file_combo)
        select_layout.addWidget(self.btn_load)
        self.content_layout.addWidget(select_group)

        viewer_group = QGroupBox("Overlay Viewer")
        viewer_layout = QVBoxLayout(viewer_group)

        control_layout = QHBoxLayout()
        control_layout.addWidget(QLabel("Stretch:"))
        self.scale_combo = QComboBox()
        self.scale_combo.addItems([
            "Auto Stretch (Siril)",
            "Asinh Stretch",
            "Midtone (MTF)",
            "Histogram Eq",
            "Log Stretch",
            "Sqrt Stretch",
            "Linear (1-99%)",
            "ZScale (IRAF)",
        ])
        self.scale_combo.currentIndexChanged.connect(self.on_stretch_changed)
        control_layout.addWidget(self.scale_combo)

        control_layout.addWidget(QLabel("Intensity:"))
        self.stretch_slider = QSlider(Qt.Horizontal)
        self.stretch_slider.setMinimum(1)
        self.stretch_slider.setMaximum(100)
        self.stretch_slider.setValue(25)
        self.stretch_slider.setFixedWidth(120)
        self.stretch_slider.sliderReleased.connect(self.redisplay_image)
        self.stretch_slider.valueChanged.connect(self.update_stretch_label)
        control_layout.addWidget(self.stretch_slider)

        self.stretch_value_label = QLabel("25")
        self.stretch_value_label.setFixedWidth(30)
        control_layout.addWidget(self.stretch_value_label)

        control_layout.addWidget(QLabel("Black:"))
        self.black_slider = QSlider(Qt.Horizontal)
        self.black_slider.setMinimum(0)
        self.black_slider.setMaximum(100)
        self.black_slider.setValue(0)
        self.black_slider.setFixedWidth(80)
        self.black_slider.sliderReleased.connect(self.redisplay_image)
        self.black_slider.valueChanged.connect(self.update_black_label)
        control_layout.addWidget(self.black_slider)

        self.black_value_label = QLabel("0")
        self.black_value_label.setFixedWidth(25)
        control_layout.addWidget(self.black_value_label)

        btn_reset_zoom = QPushButton("Reset Zoom")
        btn_reset_zoom.clicked.connect(self.reset_zoom)
        control_layout.addWidget(btn_reset_zoom)

        btn_reset_stretch = QPushButton("Reset Stretch")
        btn_reset_stretch.clicked.connect(self.reset_stretch)
        control_layout.addWidget(btn_reset_stretch)

        btn_2d_plot = QPushButton("2D Plot")
        btn_2d_plot.setStyleSheet("QPushButton { background-color: #FF9800; color: white; font-weight: bold; }")
        btn_2d_plot.clicked.connect(self.open_stretch_plot)
        control_layout.addWidget(btn_2d_plot)

        control_layout.addStretch()
        viewer_layout.addLayout(control_layout)

        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumSize(600, 500)
        self.ax = self.figure.add_subplot(111)
        self.figure.subplots_adjust(left=0.08, right=0.95, bottom=0.08, top=0.95)
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        self.canvas.setFocus()
        self.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.canvas.mpl_connect("button_press_event", self.on_button_press)
        self.canvas.mpl_connect("button_release_event", self.on_button_release)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.canvas.mpl_connect("key_press_event", self.on_key_press)
        viewer_layout.addWidget(self.canvas)
        self.content_layout.addWidget(viewer_group, stretch=1)

        sc_dot = QShortcut(QKeySequence("."), self)
        sc_dot.setContext(Qt.ApplicationShortcut)
        sc_dot.activated.connect(self.cycle_filter)
        self._shortcuts.append(sc_dot)
        sc_dot_key = QShortcut(QKeySequence(Qt.Key_Period), self)
        sc_dot_key.setContext(Qt.ApplicationShortcut)
        sc_dot_key.activated.connect(self.cycle_filter)
        self._shortcuts.append(sc_dot_key)
        sc_prev = QShortcut(QKeySequence(Qt.Key_BracketLeft), self)
        sc_prev.setContext(Qt.ApplicationShortcut)
        sc_prev.activated.connect(lambda: self.navigate_frame(-1))
        self._shortcuts.append(sc_prev)
        sc_next = QShortcut(QKeySequence(Qt.Key_BracketRight), self)
        sc_next.setContext(Qt.ApplicationShortcut)
        sc_next.activated.connect(lambda: self.navigate_frame(1))
        self._shortcuts.append(sc_next)

        self.log_window = QWidget(self, Qt.Window)
        self.log_window.setWindowTitle("Aperture Overlay Log")
        self.log_window.resize(700, 350)
        log_layout = QVBoxLayout(self.log_window)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("QTextEdit { font-family: monospace; font-size: 9pt; }")
        log_layout.addWidget(self.log_text)

        self.populate_file_list()
        self.load_master_catalog()
        self.load_aperture_by_frame()

    def log(self, message: str):
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")

    def show_log_window(self):
        self.log_window.show()
        self.log_window.raise_()
        self.log_window.activateWindow()

    def populate_file_list(self):
        cropped_dir = step2_cropped_dir(self.params.P.result_dir)
        if cropped_dir.exists() and list(cropped_dir.glob("*.fit*")):
            files = sorted([f.name for f in cropped_dir.glob("*.fit*")])
            self.use_cropped = True
        else:
            if not self.file_manager.filenames:
                try:
                    self.file_manager.scan_files()
                except Exception:
                    pass
            files = self.file_manager.filenames
            self.use_cropped = False
        self.file_list = list(files)
        self.file_combo.clear()
        self.file_combo.addItems(self.file_list)
        self._file_filter_map = {}
        self._file_frame_key_map = {}
        self._frame_key_map = {}
        self._frame_keys_by_filter = {}
        self._filter_order = []
        self._load_filter_map_from_index()
        if self.file_list:
            idx = int(getattr(self.params.P, "inspect_index", 0))
            if 0 <= idx < len(self.file_list):
                self.file_combo.setCurrentIndex(idx)
        if hasattr(self, "log_text"):
            self.log(f"Frames loaded: {len(self.file_list)} | use_cropped={self.use_cropped}")

    def on_file_changed(self, index):
        if index >= 0 and index < len(self.file_list):
            self.current_file_index = index
            self.params.P.inspect_index = index
            self.save_state()
            if self.overlay_loaded:
                self.load_and_display(keep_view=True)

    def load_master_catalog(self):
        master_path = step6_dir(self.params.P.result_dir) / "ref_catalog.tsv"
        if not master_path.exists():
            master_path = legacy_step5_refbuild_dir(self.params.P.result_dir) / "ref_catalog.tsv"
        if not master_path.exists():
            master_path = legacy_step7_refbuild_dir(self.params.P.result_dir) / "master_catalog.tsv"
        if not master_path.exists():
            master_path = self.params.P.result_dir / "master_catalog.tsv"
        if master_path.exists():
            try:
                self.master_df = pd.read_csv(master_path, sep="\t")
            except Exception:
                self.master_df = None
        if hasattr(self, "log_text"):
            n = len(self.master_df) if isinstance(self.master_df, pd.DataFrame) else 0
            self.log(f"Master catalog: {n} rows")

    def load_aperture_by_frame(self):
        ap_path = step9_dir(self.params.P.result_dir) / "aperture_by_frame.csv"
        if not ap_path.exists():
            ap_path = self.params.P.result_dir / "aperture_by_frame.csv"
        if ap_path.exists():
            try:
                self.ap_df = pd.read_csv(ap_path)
            except Exception:
                self.ap_df = None
        if hasattr(self, "log_text"):
            n = len(self.ap_df) if isinstance(self.ap_df, pd.DataFrame) else 0
            self.log(f"Aperture by frame: {n} rows")

    @staticmethod
    def _pick_col(cols, cands):
        for c in cands:
            if c in cols:
                return c
        return None

    @staticmethod
    def _as_bool(x, default=False):
        if isinstance(x, bool):
            return x
        if x is None:
            return default
        if isinstance(x, (int, np.integer)):
            return bool(x)
        s = str(x).strip().lower()
        if s in ("1", "true", "t", "yes", "y", "on"):
            return True
        if s in ("0", "false", "f", "no", "n", "off"):
            return False
        return default

    def _build_lab_frame(self, fname: str) -> pd.DataFrame | None:
        result_dir = self.params.P.result_dir
        step9_out = step9_dir(result_dir)
        cache_dir = self.params.P.cache_dir

        phot_path = step9_out / f"{fname}_photometry.tsv"
        if not phot_path.exists():
            phot_path = result_dir / f"{fname}_photometry.tsv"
        phot = None
        if phot_path.exists():
            try:
                phot = pd.read_csv(phot_path, sep="\t")
            except Exception:
                try:
                    phot = pd.read_csv(phot_path)
                except Exception:
                    phot = None

        if phot is not None and ("ID" in phot.columns):
            cols = phot.columns
            cx = "xcenter" if "xcenter" in cols else None
            cy = "ycenter" if "ycenter" in cols else None
            ix = "x_init" if "x_init" in cols else None
            iy = "y_init" if "y_init" in cols else None

            use_centroid = self._as_bool(getattr(self.params.P, "overlay_use_phot_centroid", True), True)
            if use_centroid and (cx is not None) and (cy is not None):
                x_use = phot[cx].to_numpy(float)
                y_use = phot[cy].to_numpy(float)
                if (ix is not None) and (iy is not None):
                    mnan = ~np.isfinite(x_use) | ~np.isfinite(y_use)
                    x_use[mnan] = phot[ix].to_numpy(float)[mnan]
                    y_use[mnan] = phot[iy].to_numpy(float)[mnan]
            else:
                if (ix is None) or (iy is None):
                    if (cx is None) or (cy is None):
                        raise RuntimeError("photometry file missing x/y columns")
                    x_use = phot[cx].to_numpy(float)
                    y_use = phot[cy].to_numpy(float)
                else:
                    x_use = phot[ix].to_numpy(float)
                    y_use = phot[iy].to_numpy(float)

            lab = pd.DataFrame({
                "ID": phot["ID"].astype(int).to_numpy(),
                "x_frame": x_use,
                "y_frame": y_use,
                "mag": phot["mag"].to_numpy(float) if "mag" in cols else np.nan,
                "mag_err": phot["mag_err"].to_numpy(float) if "mag_err" in cols else np.nan,
            })
            return lab

        fm_path = step9_out / "frame_sourceid_to_ID.tsv"
        if not fm_path.exists():
            fm_path = result_dir / "frame_sourceid_to_ID.tsv"
        if fm_path.exists():
            try:
                fm = pd.read_csv(fm_path, sep="\t")
            except Exception:
                fm = pd.read_csv(fm_path)
            c_file = self._pick_col(fm.columns, ["file", "fname", "frame"])
            c_id = self._pick_col(fm.columns, ["ID", "id"])
            c_x = self._pick_col(fm.columns, ["x", "x_det", "x_pix", "x0"])
            c_y = self._pick_col(fm.columns, ["y", "y_det", "y_pix", "y0"])
            if c_file and c_id and c_x and c_y:
                sub = fm[fm[c_file].astype(str) == str(fname)].copy()
                if len(sub):
                    return pd.DataFrame({
                        "ID": sub[c_id].astype(int).to_numpy(),
                        "x_frame": sub[c_x].astype(float).to_numpy(),
                        "y_frame": sub[c_y].astype(float).to_numpy(),
                        "mag": np.nan,
                        "mag_err": np.nan,
                    })

        idm_path = cache_dir / "idmatch" / f"idmatch_{fname}.csv"
        if idm_path.exists():
            idm = pd.read_csv(idm_path)
            c_sid = self._pick_col(idm.columns, ["source_id", "sourceid", "sid"])
            c_x = self._pick_col(idm.columns, ["x", "x_det", "x_pix", "x0"])
            c_y = self._pick_col(idm.columns, ["y", "y_det", "y_pix", "y0"])
            if c_sid and c_x and c_y:
                sid_series = pd.to_numeric(idm[c_sid], errors="coerce")
                mask = sid_series.notna()
                sid = sid_series[mask].astype(np.int64).to_numpy()
                x = idm.loc[mask, c_x].astype(float).to_numpy()
                y = idm.loc[mask, c_y].astype(float).to_numpy()

                map_path = step6_dir(result_dir) / "sourceid_to_ID.csv"
                if not map_path.exists():
                    map_path = legacy_step5_refbuild_dir(result_dir) / "sourceid_to_ID.csv"
                if not map_path.exists():
                    map_path = legacy_step7_refbuild_dir(result_dir) / "sourceid_to_ID.csv"
                if not map_path.exists():
                    map_path = result_dir / "sourceid_to_ID.csv"
                if map_path.exists():
                    mp = pd.read_csv(map_path)
                    if ("source_id" in mp.columns) and ("ID" in mp.columns):
                        sid2id = dict(zip(mp["source_id"].astype(np.int64), mp["ID"].astype(int)))
                    else:
                        sid2id = {}
                else:
                    sid2id = {}
                if not sid2id and (self.master_df is not None) and ("source_id" in self.master_df.columns):
                    sid2id = dict(zip(self.master_df["source_id"].astype(np.int64), self.master_df["ID"].astype(int)))

                ID = np.array([sid2id.get(s, -1) for s in sid], dtype=int)
                ok = ID >= 0
                return pd.DataFrame({
                    "ID": ID[ok],
                    "x_frame": x[ok],
                    "y_frame": y[ok],
                    "mag": np.nan,
                    "mag_err": np.nan,
                }).drop_duplicates("ID", keep="first").reset_index(drop=True)

        return None

    def load_and_display(self, keep_view: bool = False):
        if not self.file_list:
            QMessageBox.warning(self, "Warning", "No files found to display")
            return
        if self.master_df is None:
            self.load_master_catalog()
        if self.ap_df is None:
            self.load_aperture_by_frame()
        if self.master_df is None or self.ap_df is None:
            QMessageBox.warning(self, "Missing Data", "master_catalog.tsv or aperture_by_frame.csv not found")
            return

        fname = self.file_combo.currentText()
        if not fname:
            return
        self.current_filename = fname

        if keep_view and self.ax is not None:
            self._pending_xlim = self.ax.get_xlim()
            self._pending_ylim = self.ax.get_ylim()
        else:
            self._pending_xlim = None
            self._pending_ylim = None

        if self.use_cropped:
            file_path = step2_cropped_dir(self.params.P.result_dir) / fname
        else:
            file_path = self.params.get_file_path(fname)
        if not file_path.exists():
            QMessageBox.warning(self, "Missing File", f"File not found: {file_path}")
            return

        try:
            hdul = fits.open(file_path)
            self.image_data = hdul[0].data.astype(float).copy()
            self.header = hdul[0].header.copy()
            hdul.close()
        except Exception as e:
            QMessageBox.warning(self, "Load Error", str(e))
            return

        row = self.ap_df[self.ap_df["file"].astype(str) == str(fname)]
        if row.empty:
            QMessageBox.warning(self, "Missing Aperture", f"No aperture_by_frame entry for {fname}")
            return
        r_ap = float(row["r_ap"].values[0])
        r_in = float(row["r_in"].values[0])
        r_out = float(row["r_out"].values[0])

        lab_frame = self._build_lab_frame(fname)
        if lab_frame is None or len(lab_frame) == 0:
            QMessageBox.warning(self, "Missing Positions", "Could not build frame positions from photometry/idmatch")
            return

        lab = lab_frame.merge(self.master_df[["ID", "x_ref", "y_ref"]], on="ID", how="left")

        label_limit = int(getattr(self.params.P, "overlay_max_labels", 2000))
        label_font = float(getattr(self.params.P, "overlay_label_fontsize", 6.0))
        label_offset = float(getattr(self.params.P, "overlay_label_offset_px", 3.0))
        show_id_no_mag = self._as_bool(getattr(self.params.P, "overlay_show_id_when_no_mag", False), False)
        show_ref = self._as_bool(getattr(self.params.P, "overlay_show_ref_pos", True), True)
        show_shift = self._as_bool(getattr(self.params.P, "overlay_show_shift_vectors", False), False)
        shift_max = int(getattr(self.params.P, "overlay_shift_max_vectors", 300))
        shift_min = float(getattr(self.params.P, "overlay_shift_min_px", 1.5))

        lab["_has_mag"] = np.isfinite(lab.get("mag", np.nan))
        lab["_mag_for_sort"] = lab["mag"].fillna(99.0) if "mag" in lab.columns else 99.0
        lab = lab.sort_values(by=["_has_mag", "_mag_for_sort", "ID"], ascending=[False, True, True])
        lab_sel = lab.head(label_limit).copy()

        self.current_file_index = self.file_combo.currentIndex()
        self._normalized_cache = None
        self._imshow_obj = None
        self.xlim_original = None
        self.ylim_original = None
        self.reset_stretch_plot_values()
        self._file_filter_map.setdefault(fname, self._extract_filter_from_header(self.header))
        self._file_frame_key_map.setdefault(
            fname,
            self._extract_frame_key(fname, self._file_filter_map.get(fname, ""))
        )

        self.display_image(full_redraw=True)

        xy_frame = lab[["x_frame", "y_frame"]].to_numpy(float)
        xy_frame = xy_frame[np.isfinite(xy_frame).all(axis=1)]
        for (x, y) in xy_frame:
            self.ax.add_patch(Circle((x, y), r_ap, ec="gold", fc="none", lw=0.9, alpha=0.95))
            self.ax.add_patch(Circle((x, y), r_in, ec="cyan", fc="none", lw=0.6, alpha=0.70))
            self.ax.add_patch(Circle((x, y), r_out, ec="cyan", fc="none", lw=0.6, alpha=0.50))

        if show_ref:
            xy_ref = lab[["x_ref", "y_ref"]].to_numpy(float)
            xy_ref = xy_ref[np.isfinite(xy_ref).all(axis=1)]
            for (x, y) in xy_ref:
                self.ax.add_patch(Circle((x, y), r_ap, ec="orange", fc="none", lw=0.4, alpha=0.35))

        if show_shift and show_ref:
            sub = lab.copy()
            m = np.isfinite(sub["x_ref"]) & np.isfinite(sub["y_ref"]) & np.isfinite(sub["x_frame"]) & np.isfinite(sub["y_frame"])
            sub = sub.loc[m].copy()
            sub["dx"] = sub["x_frame"] - sub["x_ref"]
            sub["dy"] = sub["y_frame"] - sub["y_ref"]
            sub["dr"] = np.hypot(sub["dx"], sub["dy"])
            sub = sub[sub["dr"] >= shift_min].sort_values("dr", ascending=False).head(shift_max)

            for _, r in sub.iterrows():
                x0, y0 = float(r["x_ref"]), float(r["y_ref"])
                x1, y1 = float(r["x_frame"]), float(r["y_frame"])
                arr = FancyArrowPatch((x0, y0), (x1, y1),
                                      arrowstyle='->', mutation_scale=8,
                                      lw=0.6, color='magenta', alpha=0.65)
                self.ax.add_patch(arr)

        for _, r in lab_sel.iterrows():
            x = float(r["x_frame"])
            y = float(r["y_frame"])
            if (not np.isfinite(x)) or (not np.isfinite(y)):
                continue
            ID = int(r["ID"])
            if ("mag" in r) and np.isfinite(r["mag"]):
                txt = f"{ID}  m={float(r['mag']):.2f}"
            else:
                if not show_id_no_mag:
                    continue
                txt = f"{ID}"

            self.ax.text(
                x + label_offset, y + label_offset, txt,
                color="yellow", fontsize=label_font, ha="left", va="bottom",
                bbox=dict(boxstyle="round,pad=0.12", fc=(0, 0, 0, 0.45), ec="none")
            )

        self.canvas.draw_idle()
        self.overlay_loaded = True
        self.canvas.setFocus()

        n_ref_ok = int(np.isfinite(lab["x_ref"]).sum()) if "x_ref" in lab.columns else 0
        n_fr_ok = int(np.isfinite(lab["x_frame"]).sum()) if "x_frame" in lab.columns else 0
        self.log(f"Overlay {fname}: frame_xy_ok={n_fr_ok} ref_xy_ok={n_ref_ok}")
        if show_ref and ("x_ref" in lab.columns):
            m = np.isfinite(lab["x_ref"]) & np.isfinite(lab["y_ref"]) & np.isfinite(lab["x_frame"]) & np.isfinite(lab["y_frame"])
            if m.any():
                dr = np.hypot(lab.loc[m, "x_frame"] - lab.loc[m, "x_ref"], lab.loc[m, "y_frame"] - lab.loc[m, "y_ref"])
                self.log(f"Shift px: min/med/max = {dr.min():.3f}/{np.median(dr):.3f}/{dr.max():.3f}")

    def display_image(self, full_redraw: bool = False):
        if self.image_data is None:
            return

        normalized = self.normalize_image()
        if normalized is None:
            return
        stretched = self.apply_stretch(normalized)

        if self._imshow_obj is not None and not full_redraw:
            self._imshow_obj.set_data(stretched)
            stretch_name = self.scale_combo.currentText()
            self.ax.set_title(f"{self.current_filename} | {stretch_name}")
            self.canvas.draw_idle()
            return

        xlim_current = self._pending_xlim if self._pending_xlim is not None else (
            self.ax.get_xlim() if self.xlim_original else None
        )
        ylim_current = self._pending_ylim if self._pending_ylim is not None else (
            self.ax.get_ylim() if self.ylim_original else None
        )

        self.ax.clear()
        self._imshow_obj = self.ax.imshow(
            stretched,
            cmap="gray",
            origin="lower",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        self.ax.set_xlabel("X (pixels)")
        self.ax.set_ylabel("Y (pixels)")
        stretch_name = self.scale_combo.currentText()
        self.ax.set_title(f"{self.current_filename} | {stretch_name}")

        if self.xlim_original is None:
            self.xlim_original = self.ax.get_xlim()
            self.ylim_original = self.ax.get_ylim()
        elif xlim_current is not None:
            self.ax.set_xlim(xlim_current)
            self.ax.set_ylim(ylim_current)
        elif self._pending_xlim is not None:
            self.ax.set_xlim(self._pending_xlim)
            self.ax.set_ylim(self._pending_ylim)

        self._pending_xlim = None
        self._pending_ylim = None

        self.canvas.draw()

    def on_stretch_changed(self, index):
        self._normalized_cache = None
        self.reset_stretch_plot_values()
        self.display_image()

    def reset_stretch(self):
        self.stretch_slider.setValue(25)
        self.black_slider.setValue(0)
        self.scale_combo.setCurrentIndex(0)
        self._normalized_cache = None
        self.reset_stretch_plot_values()
        self.display_image()

    def update_stretch_label(self, value):
        self.stretch_value_label.setText(str(value))

    def update_black_label(self, value):
        self.black_value_label.setText(str(value))

    def open_stretch_plot(self):
        """Open stretch plot window showing histogram with draggable min/max markers"""
        if self.image_data is None:
            QMessageBox.warning(self, "Warning", "Load an image first")
            return

        if self.stretch_plot_dialog is not None and self.stretch_plot_dialog.isVisible():
            self.stretch_plot_dialog.raise_()
            self.stretch_plot_dialog.activateWindow()
            self._update_stretch_plot()
            return

        self.stretch_plot_dialog = QDialog(self)
        self.stretch_plot_dialog.setWindowTitle("2D Plot - Stretch Control")
        self.stretch_plot_dialog.resize(500, 250)

        layout = QVBoxLayout(self.stretch_plot_dialog)

        self.stretch_plot_info_label = QLabel("Drag min/max markers to adjust stretch")
        self.stretch_plot_info_label.setStyleSheet(
            "QLabel { padding: 5px; background-color: #E3F2FD; border-radius: 3px; }"
        )
        layout.addWidget(self.stretch_plot_info_label)

        self.stretch_plot_fig = Figure(figsize=(6, 2.5))
        self.stretch_plot_canvas = FigureCanvas(self.stretch_plot_fig)
        self.stretch_plot_ax = self.stretch_plot_fig.add_subplot(111)
        self.stretch_plot_fig.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.9)

        self.stretch_plot_canvas.mpl_connect('button_press_event', self._on_stretch_plot_press)
        self.stretch_plot_canvas.mpl_connect('motion_notify_event', self._on_stretch_plot_motion)
        self.stretch_plot_canvas.mpl_connect('button_release_event', self._on_stretch_plot_release)

        layout.addWidget(self.stretch_plot_canvas)

        hint_label = QLabel("Click and drag < > markers to adjust min/max | Changes apply in real-time")
        hint_label.setStyleSheet("QLabel { color: #666; font-size: 10px; }")
        layout.addWidget(hint_label)

        self.stretch_plot_dialog.show()
        self._update_stretch_plot()

    def _update_stretch_plot(self):
        """Update the stretch plot histogram and markers"""
        if self.stretch_plot_ax is None or self.image_data is None:
            return

        ax = self.stretch_plot_ax
        ax.clear()

        data = self.image_data.copy()
        finite_mask = np.isfinite(data)
        if not finite_mask.any():
            return

        flat = data[finite_mask].flatten()
        p_low, p_high = np.percentile(flat, [1, 99])
        display_data = flat[(flat >= p_low) & (flat <= p_high)]
        if len(display_data) == 0:
            display_data = flat

        self._stretch_data_range = (float(p_low), float(p_high))

        if self._stretch_vmin is None or self._stretch_vmax is None:
            stretch_idx = self.scale_combo.currentIndex()
            if stretch_idx == 6:
                vmin = np.percentile(flat, 1)
                vmax = np.percentile(flat, 99)
            elif stretch_idx == 7:
                vmin, vmax = self.calculate_zscale()
            else:
                _, median_val, std_val = sigma_clipped_stats(flat, sigma=3.0, maxiters=5)
                vmin = max(np.min(flat), median_val - 2.8 * std_val)
                vmax = min(np.max(flat), np.percentile(flat, 99.9))

            if vmax <= vmin:
                vmin = np.min(flat)
                vmax = np.max(flat)

            self._stretch_vmin = float(vmin)
            self._stretch_vmax = float(vmax)

        ax.hist(display_data, bins=128, color='#3a6ea5', edgecolor='none', alpha=0.7)
        ax.set_xlim(p_low, p_high)

        vmin = self._stretch_vmin
        vmax = self._stretch_vmax

        vmin_display = max(p_low, min(p_high, vmin))
        vmax_display = max(p_low, min(p_high, vmax))

        self._stretch_marker_min_line = ax.axvline(
            vmin_display, color='#FF5722', linewidth=2, linestyle='-', label=f"Min: {vmin:.1f}"
        )
        self._stretch_marker_max_line = ax.axvline(
            vmax_display, color='#4CAF50', linewidth=2, linestyle='-', label=f"Max: {vmax:.1f}"
        )

        y_max = ax.get_ylim()[1]
        ax.text(vmin_display, y_max * 0.95, '<', color='#FF5722', fontsize=14,
                ha='center', va='top', fontweight='bold')
        ax.text(vmax_display, y_max * 0.95, '>', color='#4CAF50', fontsize=14,
                ha='center', va='top', fontweight='bold')

        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Count')
        ax.set_title('Image Histogram')
        ax.legend(loc='upper right', fontsize=8)

        if self.stretch_plot_info_label:
            stretch_name = self.scale_combo.currentText()
            self.stretch_plot_info_label.setText(
                f"Stretch: {stretch_name} | Min: {vmin:.2f} | Max: {vmax:.2f}"
            )

        self.stretch_plot_canvas.draw_idle()

    def _on_stretch_plot_press(self, event):
        """Handle mouse press on stretch plot"""
        if event.inaxes != self.stretch_plot_ax or event.xdata is None:
            return
        if self._stretch_vmin is None or self._stretch_vmax is None:
            return

        x = event.xdata
        dist_to_min = abs(x - self._stretch_vmin)
        dist_to_max = abs(x - self._stretch_vmax)
        self._stretch_drag_target = "min" if dist_to_min < dist_to_max else "max"
        self._stretch_dragging = True

    def _on_stretch_plot_motion(self, event):
        """Handle mouse motion on stretch plot (dragging)"""
        if not self._stretch_dragging or event.xdata is None:
            return

        x = event.xdata
        if self._stretch_drag_target == "min":
            new_val = min(x, self._stretch_vmax - 1)
            self._stretch_vmin = new_val
        else:
            new_val = max(x, self._stretch_vmin + 1)
            self._stretch_vmax = new_val

        self._update_stretch_plot()
        self._apply_custom_stretch()

    def _on_stretch_plot_release(self, event):
        """Handle mouse release on stretch plot"""
        self._stretch_dragging = False
        self._stretch_drag_target = None

    def _apply_custom_stretch(self):
        """Apply custom vmin/vmax stretch to the image"""
        if self.image_data is None:
            return
        if self._stretch_vmin is None or self._stretch_vmax is None:
            return

        vmin = self._stretch_vmin
        vmax = self._stretch_vmax
        if vmax <= vmin:
            vmax = vmin + 1

        data = self.image_data.copy()
        normalized = (data - vmin) / (vmax - vmin + 1e-10)
        normalized = np.clip(normalized, 0, 1)
        stretched = self.apply_stretch(normalized)

        if self._imshow_obj is not None:
            self._imshow_obj.set_data(stretched)
            self.canvas.draw_idle()

    def reset_stretch_plot_values(self):
        """Reset stretch plot values when changing image or stretch mode"""
        self._stretch_vmin = None
        self._stretch_vmax = None
        if self.stretch_plot_dialog and self.stretch_plot_dialog.isVisible():
            self._update_stretch_plot()

    def apply_stretch(self, data):
        stretch_idx = self.scale_combo.currentIndex()
        intensity = self.stretch_slider.value() / 100.0
        black_point = self.black_slider.value() / 100.0

        data = np.clip((data - black_point) / (1.0 - black_point + 1e-10), 0, 1)

        if stretch_idx == 0:
            return self._stretch_auto_siril(data, intensity)
        if stretch_idx == 1:
            return self._stretch_asinh(data, intensity)
        if stretch_idx == 2:
            return self._stretch_mtf(data, intensity)
        if stretch_idx == 3:
            return self._stretch_histogram_eq(data)
        if stretch_idx == 4:
            return self._stretch_log(data, intensity)
        if stretch_idx == 5:
            return self._stretch_sqrt(data, intensity)
        return data

    def _stretch_auto_siril(self, data, intensity):
        finite = data[np.isfinite(data)]
        if len(finite) == 0:
            return data
        median_val = np.median(finite)
        mad = np.median(np.abs(finite - median_val))
        sigma = mad * 1.4826
        shadows = max(0.0, median_val - 2.8 * sigma)
        stretched = (data - shadows) / (1.0 - shadows + 1e-10)
        stretched = np.clip(stretched, 0, 1)
        midtone = 0.15 + (1.0 - intensity) * 0.35
        return self._mtf_function(stretched, midtone)

    def _stretch_asinh(self, data, intensity):
        beta = 1.0 + intensity * 15.0
        stretched = np.arcsinh(data * beta) / np.arcsinh(beta)
        return np.clip(stretched, 0, 1)

    def _stretch_mtf(self, data, intensity):
        midtone = 0.05 + (1.0 - intensity) * 0.45
        return self._mtf_function(data, midtone)

    def _mtf_function(self, data, midtone):
        m = np.clip(midtone, 0.001, 0.999)
        result = np.zeros_like(data)
        mask = data > 0
        result[mask] = (m - 1) * data[mask] / ((2 * m - 1) * data[mask] - m)
        result[data == 0] = 0
        result[data == 1] = 1
        return np.clip(result, 0, 1)

    def _stretch_histogram_eq(self, data):
        finite = data[np.isfinite(data)]
        if len(finite) == 0:
            return data
        hist, bin_edges = np.histogram(finite.flatten(), bins=65536, range=(0, 1))
        cdf = hist.cumsum()
        cdf = cdf / cdf[-1]
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        stretched = np.interp(data, bin_centers, cdf)
        return np.clip(stretched, 0, 1)

    def _stretch_log(self, data, intensity):
        a = 100 + intensity * 900
        stretched = np.log(1 + a * data) / np.log(1 + a)
        return np.clip(stretched, 0, 1)

    def _stretch_sqrt(self, data, intensity):
        power = 0.2 + (1.0 - intensity) * 0.8
        stretched = np.power(data, power)
        return np.clip(stretched, 0, 1)

    def normalize_image(self):
        if self.image_data is None:
            return None

        stretch_idx = self.scale_combo.currentIndex()
        cache_key = (id(self.image_data), stretch_idx)
        if self._normalized_cache is not None and self._normalized_cache[0] == cache_key:
            return self._normalized_cache[1].copy()

        finite = np.isfinite(self.image_data)
        if not finite.any():
            return np.zeros_like(self.image_data)

        data = self.image_data.copy()
        if stretch_idx == 6:
            vmin = np.percentile(data[finite], 1)
            vmax = np.percentile(data[finite], 99)
        elif stretch_idx == 7:
            vmin, vmax = self.calculate_zscale()
        else:
            _, median_val, std_val = sigma_clipped_stats(data[finite], sigma=3.0, maxiters=5)
            vmin = max(np.min(data[finite]), median_val - 2.8 * std_val)
            vmax = min(np.max(data[finite]), np.percentile(data[finite], 99.9))

        if vmax <= vmin:
            vmin = np.min(data[finite])
            vmax = np.max(data[finite])

        normalized = (data - vmin) / (vmax - vmin + 1e-10)
        normalized = np.clip(normalized, 0, 1)
        self._normalized_cache = (cache_key, normalized)
        return normalized.copy()

    def calculate_zscale(self):
        finite = np.isfinite(self.image_data)
        if not finite.any():
            return 0, 1
        data = self.image_data[finite]
        vmin, vmax = ZScaleInterval().get_limits(data)
        if vmax <= vmin:
            vmin = float(np.min(data))
            vmax = float(np.max(data))
        return float(vmin), float(vmax)

    def redisplay_image(self):
        self.display_image()

    def on_button_press(self, event):
        if event.button == 3:
            self.panning = True
            self.pan_start = (event.xdata, event.ydata)

    def on_button_release(self, event):
        if event.button == 3:
            self.panning = False
            self.pan_start = None

    def on_motion(self, event):
        if self.panning and self.pan_start is not None and event.inaxes == self.ax:
            dx = self.pan_start[0] - event.xdata
            dy = self.pan_start[1] - event.ydata
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            self.ax.set_xlim([xlim[0] + dx, xlim[1] + dx])
            self.ax.set_ylim([ylim[0] + dy, ylim[1] + dy])
            self.canvas.draw()
            return

        if event.inaxes == self.ax and event.xdata is not None and event.ydata is not None:
            self.cursor_x = event.xdata
            self.cursor_y = event.ydata
        else:
            self.cursor_x = None
            self.cursor_y = None

    def on_scroll(self, event):
        if event.inaxes != self.ax:
            return
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        zoom_factor = 1.2 if event.button == "up" else 0.8
        xdata, ydata = event.xdata, event.ydata
        x_range = (xlim[1] - xlim[0]) * zoom_factor
        y_range = (ylim[1] - ylim[0]) * zoom_factor
        new_xlim = [
            xdata - x_range * (xdata - xlim[0]) / (xlim[1] - xlim[0]),
            xdata + x_range * (xlim[1] - xdata) / (xlim[1] - xlim[0]),
        ]
        new_ylim = [
            ydata - y_range * (ydata - ylim[0]) / (ylim[1] - ylim[0]),
            ydata + y_range * (ylim[1] - ydata) / (ylim[1] - ylim[0]),
        ]
        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self.canvas.draw()

    def reset_zoom(self):
        if self.xlim_original is not None:
            self.ax.set_xlim(self.xlim_original)
            self.ax.set_ylim(self.ylim_original)
            self.canvas.draw()

    def keyPressEvent(self, event):
        key = event.text().lower()
        if key == ".":
            self.cycle_filter()
        elif event.key() == Qt.Key_BracketLeft:
            self.navigate_frame(-1)
        elif event.key() == Qt.Key_BracketRight:
            self.navigate_frame(1)
        else:
            super().keyPressEvent(event)

    def on_key_press(self, event):
        if event.key == ".":
            self.cycle_filter()
        elif event.key == "[":
            self.navigate_frame(-1)
        elif event.key == "]":
            self.navigate_frame(1)

    def navigate_frame(self, direction: int):
        """Navigate to previous/next frame within the SAME filter"""
        if not self.file_list:
            return
        self._build_frame_key_map()
        current_filter = self._file_filter_map.get(self.current_filename, "")

        # Build list of file indices for current filter
        filter_indices = []
        for idx, fname in enumerate(self.file_list):
            fkey = self._file_filter_map.get(fname, "")
            if fkey == current_filter:
                filter_indices.append(idx)

        if not filter_indices:
            # Fallback: cycle through all files
            new_index = (self.current_file_index + direction) % len(self.file_list)
        else:
            # Find current position within filter's files
            try:
                pos = filter_indices.index(self.current_file_index)
            except ValueError:
                pos = 0
            # Move within filter only
            pos = (pos + direction) % len(filter_indices)
            new_index = filter_indices[pos]

        self.file_combo.blockSignals(True)
        self.file_combo.setCurrentIndex(new_index)
        self.file_combo.blockSignals(False)
        self.current_file_index = new_index
        self.params.P.inspect_index = new_index
        self.load_and_display(keep_view=True)

    def cycle_filter(self):
        """Cycle to next filter, keeping the same frame position (index within filter)"""
        if not self.file_list or self.header is None:
            return
        self._build_frame_key_map()
        current_filter = self._file_filter_map.get(self.current_filename, "")
        filters = [f for f in self._filter_order if f]
        if len(filters) <= 1:
            self.log("Filter cycle skipped: only one filter found")
            return

        # Find next filter
        try:
            current_filter_idx = filters.index(current_filter)
            next_filter = filters[(current_filter_idx + 1) % len(filters)]
        except ValueError:
            next_filter = filters[0]

        # Build file indices for current and next filter
        current_filter_indices = []
        next_filter_indices = []
        for idx, fname in enumerate(self.file_list):
            fkey = self._file_filter_map.get(fname, "")
            if fkey == current_filter:
                current_filter_indices.append(idx)
            if fkey == next_filter:
                next_filter_indices.append(idx)

        if not next_filter_indices:
            return

        # Find current position within current filter's files
        try:
            pos_in_filter = current_filter_indices.index(self.current_file_index)
        except ValueError:
            pos_in_filter = 0

        # Go to same position in next filter (or last if shorter)
        target_pos = min(pos_in_filter, len(next_filter_indices) - 1)
        new_index = next_filter_indices[target_pos]

        self.file_combo.blockSignals(True)
        self.file_combo.setCurrentIndex(new_index)
        self.file_combo.blockSignals(False)
        self.current_file_index = new_index
        self.params.P.inspect_index = new_index
        self.load_and_display(keep_view=True)
        self.log(f"Filter cycle: {next_filter}")

    @staticmethod
    def _normalize_filter_key(value: str | None) -> str:
        if value is None:
            return ""
        return str(value).strip().upper()

    def _extract_filter_from_header(self, header) -> str:
        if header is None:
            return ""
        for key in ("FILTER", "FILTER1", "FILTER2", "FILTNAM"):
            val = header.get(key)
            if val:
                return self._normalize_filter_key(val)
        return ""

    @staticmethod
    def _infer_filter_from_filename(fname: str) -> str:
        base = Path(fname).name
        for ext in (".fits", ".fit", ".fts", ".fz", ".gz"):
            if base.lower().endswith(ext):
                base = base[: -len(ext)]
        parts = [p for p in base.replace(".", "_").replace("-", "_").split("_") if p]
        for token in reversed(parts):
            cand = token.lower()
            if 1 <= len(cand) <= 3 and cand.isalpha():
                return cand.upper()
        return ""

    def _extract_frame_key(self, fname: str, filter_key: str) -> str:
        name = Path(fname).name
        base = name
        for ext in (".fits", ".fit", ".fts", ".fz", ".gz"):
            if base.lower().endswith(ext):
                base = base[: -len(ext)]
        base_lower = base.lower()
        filt = str(filter_key or "").lower() or self._infer_filter_from_filename(fname).lower()
        if filt:
            for sep in ("-", "_", "."):
                suffix = f"{sep}{filt}"
                if base_lower.endswith(suffix):
                    return base[: -len(suffix)]
        return base

    def _load_filter_map_from_index(self):
        # 1) headers.csv에서 먼저 로드 (Step 1에서 생성됨 - FITS 안 읽어도 됨)
        headers_path = step1_dir(self.params.P.result_dir) / "headers.csv"
        if not headers_path.exists():
            headers_path = self.params.P.result_dir / "headers.csv"
        if headers_path.exists():
            try:
                hdf = pd.read_csv(headers_path)
                if "Filename" in hdf.columns and "FILTER" in hdf.columns:
                    for fname, filt in zip(hdf["Filename"].astype(str), hdf["FILTER"].astype(str)):
                        fkey = self._normalize_filter_key(filt)
                        if fkey:
                            self._file_filter_map[fname] = fkey
                            if fname.lower().startswith("crop_"):
                                self._file_filter_map[fname[5:]] = fkey
                            if fname.lower().startswith("cropped_"):
                                self._file_filter_map[fname[8:]] = fkey
            except Exception:
                pass

        # 2) photometry_index.csv에서 보충 (Step 9 이후)
        idx_path = step9_dir(self.params.P.result_dir) / "photometry_index.csv"
        if not idx_path.exists():
            idx_path = self.params.P.result_dir / "photometry_index.csv"
        if not idx_path.exists():
            return
        try:
            df = pd.read_csv(idx_path)
        except Exception:
            return
        if "file" not in df.columns or "filter" not in df.columns:
            return
        for fname, filt in zip(df["file"].astype(str), df["filter"].astype(str)):
            if fname in self._file_filter_map:
                continue  # 이미 있으면 스킵
            fkey = self._normalize_filter_key(filt)
            if not fkey:
                fkey = self._infer_filter_from_filename(fname)
            self._file_filter_map[fname] = fkey
            if fname.lower().startswith("crop_"):
                self._file_filter_map[fname[5:]] = fkey
            if fname.lower().startswith("cropped_"):
                self._file_filter_map[fname[8:]] = fkey

    def _build_frame_key_map(self):
        if self._frame_key_map:
            return
        for idx, fname in enumerate(self.file_list):
            fkey = self._file_filter_map.get(fname)
            if not fkey:
                try:
                    if self.use_cropped:
                        fpath = step2_cropped_dir(self.params.P.result_dir) / fname
                    else:
                        fpath = self.params.get_file_path(fname)
                    hdr = fits.getheader(fpath)
                    fkey = self._extract_filter_from_header(hdr)
                except Exception:
                    fkey = ""
                if not fkey:
                    fkey = self._infer_filter_from_filename(fname)
                self._file_filter_map[fname] = fkey
            frame_key = self._extract_frame_key(fname, fkey)
            self._file_frame_key_map[fname] = frame_key
            self._frame_key_map[(frame_key, fkey)] = idx
            self._frame_keys_by_filter.setdefault(fkey, []).append(frame_key)
        self._filter_order = [f for f in sorted(self._frame_keys_by_filter.keys()) if f]

    def open_parameters_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Overlay Parameters")
        dialog.resize(460, 420)
        layout = QVBoxLayout(dialog)
        form = QFormLayout()

        self.param_labels = QSpinBox()
        self.param_labels.setRange(0, 20000)
        self.param_labels.setValue(int(getattr(self.params.P, "overlay_max_labels", 2000)))
        form.addRow("Max labels:", self.param_labels)

        self.param_label_fs = QDoubleSpinBox()
        self.param_label_fs.setRange(1.0, 20.0)
        self.param_label_fs.setValue(float(getattr(self.params.P, "overlay_label_fontsize", 6.0)))
        form.addRow("Label font size:", self.param_label_fs)

        self.param_label_offset = QDoubleSpinBox()
        self.param_label_offset.setRange(0.0, 20.0)
        self.param_label_offset.setValue(float(getattr(self.params.P, "overlay_label_offset_px", 3.0)))
        form.addRow("Label offset px:", self.param_label_offset)

        self.param_show_id = QCheckBox("Show ID when mag missing")
        self.param_show_id.setChecked(bool(getattr(self.params.P, "overlay_show_id_when_no_mag", False)))
        form.addRow("", self.param_show_id)

        self.param_use_centroid = QCheckBox("Use photometry centroid")
        self.param_use_centroid.setChecked(bool(getattr(self.params.P, "overlay_use_phot_centroid", True)))
        form.addRow("", self.param_use_centroid)

        self.param_show_ref = QCheckBox("Show ref positions")
        self.param_show_ref.setChecked(bool(getattr(self.params.P, "overlay_show_ref_pos", True)))
        form.addRow("", self.param_show_ref)

        self.param_show_shift = QCheckBox("Show shift vectors")
        self.param_show_shift.setChecked(bool(getattr(self.params.P, "overlay_show_shift_vectors", False)))
        form.addRow("", self.param_show_shift)

        self.param_shift_max = QSpinBox()
        self.param_shift_max.setRange(0, 10000)
        self.param_shift_max.setValue(int(getattr(self.params.P, "overlay_shift_max_vectors", 300)))
        form.addRow("Shift max vectors:", self.param_shift_max)

        self.param_shift_min = QDoubleSpinBox()
        self.param_shift_min.setRange(0.0, 50.0)
        self.param_shift_min.setValue(float(getattr(self.params.P, "overlay_shift_min_px", 1.5)))
        form.addRow("Shift min px:", self.param_shift_min)

        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(lambda: self.save_parameters(dialog))
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec_()

    def save_parameters(self, dialog):
        self.params.P.overlay_max_labels = self.param_labels.value()
        self.params.P.overlay_label_fontsize = self.param_label_fs.value()
        self.params.P.overlay_label_offset_px = self.param_label_offset.value()
        self.params.P.overlay_show_id_when_no_mag = self.param_show_id.isChecked()
        self.params.P.overlay_use_phot_centroid = self.param_use_centroid.isChecked()
        self.params.P.overlay_show_ref_pos = self.param_show_ref.isChecked()
        self.params.P.overlay_show_shift_vectors = self.param_show_shift.isChecked()
        self.params.P.overlay_shift_max_vectors = self.param_shift_max.value()
        self.params.P.overlay_shift_min_px = self.param_shift_min.value()
        self.persist_params()
        self.save_state()
        QMessageBox.information(dialog, "Success", "Parameters saved!")
        dialog.accept()

    def validate_step(self) -> bool:
        ap_path = step9_dir(self.params.P.result_dir) / "aperture_by_frame.csv"
        if not ap_path.exists():
            ap_path = self.params.P.result_dir / "aperture_by_frame.csv"
        master_path = step6_dir(self.params.P.result_dir) / "ref_catalog.tsv"
        if not master_path.exists():
            master_path = legacy_step5_refbuild_dir(self.params.P.result_dir) / "ref_catalog.tsv"
        if not master_path.exists():
            master_path = legacy_step7_refbuild_dir(self.params.P.result_dir) / "master_catalog.tsv"
        if not master_path.exists():
            master_path = self.params.P.result_dir / "master_catalog.tsv"
        return ap_path.exists() and master_path.exists()

    def save_state(self):
        state_data = {
            "inspect_index": getattr(self.params.P, "inspect_index", 0),
            "overlay_max_labels": getattr(self.params.P, "overlay_max_labels", 2000),
            "overlay_label_fontsize": getattr(self.params.P, "overlay_label_fontsize", 6.0),
            "overlay_label_offset_px": getattr(self.params.P, "overlay_label_offset_px", 3.0),
            "overlay_show_id_when_no_mag": getattr(self.params.P, "overlay_show_id_when_no_mag", False),
            "overlay_use_phot_centroid": getattr(self.params.P, "overlay_use_phot_centroid", True),
            "overlay_show_ref_pos": getattr(self.params.P, "overlay_show_ref_pos", True),
            "overlay_show_shift_vectors": getattr(self.params.P, "overlay_show_shift_vectors", False),
            "overlay_shift_max_vectors": getattr(self.params.P, "overlay_shift_max_vectors", 300),
            "overlay_shift_min_px": getattr(self.params.P, "overlay_shift_min_px", 1.5),
        }
        self.project_state.store_step_data("aperture_overlay", state_data)

    def restore_state(self):
        state_data = self.project_state.get_step_data("aperture_overlay")
        if state_data:
            for key, val in state_data.items():
                if hasattr(self.params.P, key):
                    setattr(self.params.P, key, val)
            idx = int(getattr(self.params.P, "inspect_index", 0))
            if self.file_list and 0 <= idx < len(self.file_list):
                self.file_combo.setCurrentIndex(idx)
