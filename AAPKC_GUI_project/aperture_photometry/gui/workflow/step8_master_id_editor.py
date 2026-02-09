"""
Step 7: Master Star IDs Editor
Ported from AAPKI_GUI.ipynb Cell 10 (GUI adaptation).
"""

from __future__ import annotations

import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from astropy.visualization import ZScaleInterval
from astropy.coordinates import SkyCoord
import astropy.units as u
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QGroupBox, QMessageBox,
    QTextEdit, QDialog, QFormLayout, QDialogButtonBox, QProgressBar,
    QCheckBox, QSpinBox, QDoubleSpinBox, QLineEdit, QTableWidget,
    QTableWidgetItem, QHeaderView, QAbstractItemView, QWidget, QComboBox,
    QSlider
)
from PyQt5.QtCore import Qt

from .step_window_base import StepWindowBase
from ...utils.step_paths import step2_cropped_dir, step5_dir, step6_dir, step8_dir, crop_is_active


class MasterIdEditorWindow(StepWindowBase):
    """Step 7: Master Star IDs Editor"""

    def __init__(self, params, file_manager, project_state, main_window):
        self.file_manager = file_manager
        self.file_list = []
        self.use_cropped = False
        self.current_filename = None
        self.image_data = None
        self.header = None
        self.idmatch_df = None
        self.master_ids = set()
        self.selected_source_id = None
        self.last_click_xy = None
        self.gaia_df = None  # Gaia catalog cache
        self.master_gmag_map = {}
        self.internal_id_map = {}  # source_id -> internal_id
        self.source_id_from_internal = {}  # internal_id -> source_id
        self._auto_master_dirty = False

        # Matplotlib components
        self.figure = None
        self.canvas = None
        self.ax = None
        self._imshow_obj = None
        self._normalized_cache = None
        self.xlim_original = None
        self.ylim_original = None
        self.panning = False
        self.pan_start = None
        self.hover_xy = None  # Track mouse hover position for G key

        super().__init__(
            step_index=7,
            step_name="Master ID Editor",
            params=params,
            project_state=project_state,
            main_window=main_window
        )

        self.setup_step_ui()
        self.restore_state()
        self.setFocusPolicy(Qt.StrongFocus)

    def setup_step_ui(self):
        info = QLabel(
            "Edit master_star_ids.csv using idmatch overlays.\n"
            "Shortcuts: A=Add (detected or undetected star), D=Remove, Shift+D=Remove Box, G=Radial Profile (at cursor), [ / ] = Prev/Next frame"
        )
        info.setStyleSheet("QLabel { background-color: #E3F2FD; padding: 10px; border-radius: 5px; }")
        self.content_layout.addWidget(info)

        control_layout = QHBoxLayout()
        btn_params = QPushButton("Editor Parameters")
        btn_params.setStyleSheet("QPushButton { background-color: #9C27B0; color: white; font-weight: bold; padding: 8px 15px; }")
        btn_params.clicked.connect(self.open_parameters_dialog)
        control_layout.addWidget(btn_params)

        btn_log = QPushButton("Log")
        btn_log.setStyleSheet("QPushButton { background-color: #607D8B; color: white; font-weight: bold; padding: 8px 15px; }")
        btn_log.clicked.connect(self.show_log_window)
        control_layout.addWidget(btn_log)

        self.content_layout.addLayout(control_layout)

        # Selected info
        select_layout = QHBoxLayout()
        self.selected_label = QLabel("Selected source_id: (none)")
        select_layout.addWidget(self.selected_label)
        select_layout.addStretch()
        self.content_layout.addLayout(select_layout)

        # Viewer + table
        main_layout = QHBoxLayout()

        viewer_group = QGroupBox("Preview")
        viewer_layout = QVBoxLayout(viewer_group)

        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("File:"))
        self.file_combo = QComboBox()
        self.file_combo.currentIndexChanged.connect(self.on_file_changed)
        file_layout.addWidget(self.file_combo)
        viewer_layout.addLayout(file_layout)

        # Stretch controls (from Step 4)
        stretch_layout = QHBoxLayout()
        stretch_layout.addWidget(QLabel("Stretch:"))
        self.scale_combo = QComboBox()
        self.scale_combo.addItems([
            "Auto Stretch (Siril)",
            "Asinh Stretch",
            "Midtone (MTF)",
            "Histogram Eq",
            "Log Stretch",
            "Sqrt Stretch",
            "Linear (1-99%)",
            "ZScale (IRAF)"
        ])
        self.scale_combo.currentIndexChanged.connect(self.on_stretch_changed)
        stretch_layout.addWidget(self.scale_combo)

        stretch_layout.addWidget(QLabel("Intensity:"))
        self.stretch_slider = QSlider(Qt.Horizontal)
        self.stretch_slider.setMinimum(1)
        self.stretch_slider.setMaximum(100)
        self.stretch_slider.setValue(25)
        self.stretch_slider.setFixedWidth(100)
        self.stretch_slider.sliderReleased.connect(self.redisplay_image)
        self.stretch_slider.valueChanged.connect(self.update_stretch_label)
        stretch_layout.addWidget(self.stretch_slider)

        self.stretch_value_label = QLabel("25")
        self.stretch_value_label.setFixedWidth(25)
        stretch_layout.addWidget(self.stretch_value_label)

        stretch_layout.addWidget(QLabel("Black:"))
        self.black_slider = QSlider(Qt.Horizontal)
        self.black_slider.setMinimum(0)
        self.black_slider.setMaximum(100)
        self.black_slider.setValue(0)
        self.black_slider.setFixedWidth(60)
        self.black_slider.sliderReleased.connect(self.redisplay_image)
        self.black_slider.valueChanged.connect(self.update_black_label)
        stretch_layout.addWidget(self.black_slider)

        self.black_value_label = QLabel("0")
        self.black_value_label.setFixedWidth(20)
        stretch_layout.addWidget(self.black_value_label)

        btn_reset_zoom = QPushButton("Reset Zoom")
        btn_reset_zoom.clicked.connect(self.reset_zoom)
        stretch_layout.addWidget(btn_reset_zoom)

        stretch_layout.addStretch()
        viewer_layout.addLayout(stretch_layout)

        self.figure = Figure(figsize=(6, 5))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.canvas.setFocusPolicy(Qt.ClickFocus)

        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.canvas.mpl_connect('button_press_event', self.on_button_press)
        self.canvas.mpl_connect('button_release_event', self.on_button_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('button_press_event', self.on_click)

        viewer_layout.addWidget(self.canvas)
        main_layout.addWidget(viewer_group, stretch=2)

        table_group = QGroupBox("Master IDs")
        table_layout = QVBoxLayout(table_group)

        self.master_table = QTableWidget()
        self.master_table.setColumnCount(3)
        self.master_table.setHorizontalHeaderLabels(["ID", "source_id", "G mag"])
        self.master_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.master_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.master_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.master_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.master_table.itemSelectionChanged.connect(self.on_table_selection_changed)
        table_layout.addWidget(self.master_table)

        main_layout.addWidget(table_group, stretch=1)

        self.content_layout.addLayout(main_layout)

        self.log_window = QWidget(self, Qt.Window)
        self.log_window.setWindowTitle("Master ID Log")
        self.log_window.resize(700, 350)
        log_layout = QVBoxLayout(self.log_window)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("QTextEdit { font-family: monospace; font-size: 9pt; }")
        log_layout.addWidget(self.log_text)

        self.populate_file_list()
        self.load_master_ids()
        self.load_gaia_catalog()

    def log(self, message: str):
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")

    def populate_file_list(self):
        cropped_dir = step2_cropped_dir(self.params.P.result_dir)
        legacy_cropped = self.params.P.result_dir / "cropped"
        crop_active = crop_is_active(self.params.P.result_dir)
        if crop_active and cropped_dir.exists() and list(cropped_dir.glob("*.fit*")):
            files = sorted([f.name for f in cropped_dir.glob("*.fit*")])
            self.use_cropped = True
        elif crop_active and legacy_cropped.exists() and list(legacy_cropped.glob("*.fit*")):
            files = sorted([f.name for f in legacy_cropped.glob("*.fit*")])
            self.use_cropped = True
            cropped_dir = legacy_cropped
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

    def load_master_ids(self):
        master_path = step8_dir(self.params.P.result_dir) / "master_star_ids.csv"
        if not master_path.exists():
            master_path = step6_dir(self.params.P.result_dir) / "master_star_ids.csv"
        if not master_path.exists():
            master_path = self.params.P.result_dir / "master_star_ids.csv"
        self.internal_id_map = {}
        self.source_id_from_internal = {}
        if master_path.exists():
            try:
                df = pd.read_csv(master_path)
                if "source_id" in df.columns:
                    self.master_ids = set(df["source_id"].dropna().astype("int64").tolist())
                    if "g_mag" in df.columns:
                        gmag_series = pd.to_numeric(df["g_mag"], errors="coerce")
                        sid_series = df["source_id"].astype("int64")
                        self.master_gmag_map = dict(zip(sid_series.tolist(), gmag_series.tolist()))
                    self.log(f"Loaded {len(self.master_ids)} master IDs from {master_path.name}")
                    self.update_master_table()
            except Exception as e:
                self.log(f"Error loading master IDs: {e}")

    def load_gaia_catalog(self):
        """Load Gaia catalog for source info lookup"""
        from astropy.table import Table
        gaia_path = step5_dir(self.params.P.result_dir) / "gaia_fov.ecsv"
        if not gaia_path.exists():
            gaia_path = self.params.P.result_dir / "gaia_fov.ecsv"
        if gaia_path.exists():
            try:
                tab = Table.read(str(gaia_path), format="ascii.ecsv")
                # Lowercase column names
                cols = list(tab.colnames)
                lower = [c.lower() for c in cols]
                if lower != cols:
                    tab.rename_columns(cols, lower)
                self.gaia_df = tab.to_pandas()
                # Convert source_id to int64 for consistent matching
                if "source_id" in self.gaia_df.columns:
                    self.gaia_df["source_id"] = self.gaia_df["source_id"].astype("int64")
                self.log(f"Gaia catalog loaded: {len(self.gaia_df)} sources")
                if self.master_ids:
                    self.update_master_table()
                    self.update_overlay()
            except Exception as e:
                self.log(f"Failed to load Gaia catalog: {e}")
                self.gaia_df = None

    def get_gaia_info(self, source_id: int) -> str:
        """Get Gaia info string for a source_id"""
        if self.gaia_df is None or len(self.gaia_df) == 0:
            return ""
        try:
            row = self.gaia_df[self.gaia_df["source_id"] == source_id]
            if len(row) == 0:
                return ""
            row = row.iloc[0]
            parts = []
            # Magnitudes
            g = row.get("phot_g_mean_mag", np.nan)
            bp = row.get("phot_bp_mean_mag", np.nan)
            rp = row.get("phot_rp_mean_mag", np.nan)
            if np.isfinite(g):
                parts.append(f"G={g:.3f}")
            if np.isfinite(bp):
                parts.append(f"BP={bp:.3f}")
            if np.isfinite(rp):
                parts.append(f"RP={rp:.3f}")
            # Color
            if np.isfinite(bp) and np.isfinite(rp):
                parts.append(f"BP-RP={bp-rp:.3f}")
            # Coordinates
            ra = row.get("ra", np.nan)
            dec = row.get("dec", np.nan)
            if np.isfinite(ra) and np.isfinite(dec):
                parts.append(f"RA={ra:.5f}")
                parts.append(f"Dec={dec:.5f}")
            # Proper motion
            pmra = row.get("pmra", np.nan)
            pmdec = row.get("pmdec", np.nan)
            if np.isfinite(pmra) and np.isfinite(pmdec):
                parts.append(f"PM=({pmra:.2f},{pmdec:.2f})mas/yr")
            return " | ".join(parts)
        except Exception:
            return ""

    def update_master_table(self):
        """Update master table with internal ID (brightness sorted)"""
        # Get G mag for each source_id and sort by brightness
        frame_sids = None
        if self.idmatch_df is not None and (not self.idmatch_df.empty) and "source_id" in self.idmatch_df.columns:
            frame_sids = set(self.idmatch_df["source_id"].astype("int64").tolist())

        id_gmag_list = []
        for sid in self.master_ids:
            if frame_sids is not None and sid not in frame_sids:
                continue
            g_mag = self._get_gmag_for_source(sid)
            id_gmag_list.append((sid, g_mag))

        # Sort by G mag (brightest first), NaN at end
        id_gmag_list.sort(key=lambda x: (np.isnan(x[1]), x[1]))

        # Build internal ID mapping
        self.internal_id_map = {}  # source_id -> internal_id
        self.source_id_from_internal = {}  # internal_id -> source_id

        self.master_table.setRowCount(len(id_gmag_list))
        for i, (sid, g_mag) in enumerate(id_gmag_list):
            internal_id = i + 1
            self.internal_id_map[sid] = internal_id
            self.source_id_from_internal[internal_id] = sid

            self.master_table.setItem(i, 0, QTableWidgetItem(str(internal_id)))
            self.master_table.setItem(i, 1, QTableWidgetItem(str(sid)))
            g_str = f"{g_mag:.3f}" if np.isfinite(g_mag) else "-"
            self.master_table.setItem(i, 2, QTableWidgetItem(g_str))

        n_total = len(id_gmag_list)
        n_gmag = sum(1 for _, g in id_gmag_list if np.isfinite(g))
        if hasattr(self, "log_text"):
            self.log(f"Master IDs: {n_total} | Gmag available: {n_gmag}")

        self.update_navigation_buttons()

    def on_table_selection_changed(self):
        """Handle table selection change"""
        rows = self.master_table.selectionModel().selectedRows()
        if rows:
            row_idx = rows[0].row()
            sid_item = self.master_table.item(row_idx, 1)
            if sid_item:
                try:
                    self.selected_source_id = int(sid_item.text())
                    in_master = "✓ IN MASTER"
                    internal_id = self.internal_id_map.get(self.selected_source_id, "?")
                    self.selected_label.setText(
                        f"Selected: ID {internal_id} | source_id: {self.selected_source_id} ({in_master})"
                    )
                    self.update_overlay()
                except ValueError:
                    pass

    def select_source_in_table(self, source_id: int):
        """Select a source_id in the master table"""
        for row in range(self.master_table.rowCount()):
            sid_item = self.master_table.item(row, 1)
            if sid_item and int(sid_item.text()) == source_id:
                self.master_table.blockSignals(True)
                self.master_table.selectRow(row)
                self.master_table.scrollToItem(sid_item)
                self.master_table.blockSignals(False)
                break

    def on_file_changed(self, index):
        if index < 0 or index >= len(self.file_list):
            return
        self.load_and_display()

    def load_and_display(self):
        filename = self.file_combo.currentText()
        if not filename:
            return
        try:
            if self.use_cropped:
                cropped_dir = step2_cropped_dir(self.params.P.result_dir)
                if not cropped_dir.exists():
                    cropped_dir = self.params.P.result_dir / "cropped"
                file_path = cropped_dir / filename
            else:
                file_path = self.params.P.data_dir / filename
            with fits.open(file_path) as hdul:
                self.image_data = hdul[0].data.astype(float)
                self.header = hdul[0].header
            self.current_filename = filename
            self.xlim_original = None
            self.ylim_original = None
            self._imshow_obj = None
            self._normalized_cache = None
            self.load_idmatch_for_file(filename)
            self.display_image(full_redraw=True)
            self.update_overlay()
            if self._auto_master_dirty:
                self.save_master_ids(log_action="auto_add")
                self._auto_master_dirty = False
            else:
                self.update_master_table()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load: {str(e)}")

    def load_idmatch_for_file(self, filename):
        idmatch_path = self.params.P.cache_dir / "idmatch" / f"idmatch_{filename}.csv"
        if idmatch_path.exists():
            try:
                df = pd.read_csv(idmatch_path)
                if {"x", "y", "source_id"} <= set(df.columns):
                    self.idmatch_df = df
                    self._auto_add_detections_to_master(df)
                    return
            except Exception:
                pass
        self.idmatch_df = pd.DataFrame(columns=["x", "y", "source_id"])

    def _auto_add_detections_to_master(self, df: pd.DataFrame):
        try:
            sids = set(pd.to_numeric(df["source_id"], errors="coerce").dropna().astype("int64").tolist())
        except Exception:
            return
        new_ids = sids - self.master_ids
        if not new_ids:
            return
        self.master_ids |= new_ids
        self._auto_master_dirty = True

    def display_image(self, full_redraw=False):
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

        xlim_current = self.ax.get_xlim() if self.xlim_original else None
        ylim_current = self.ax.get_ylim() if self.ylim_original else None

        self.ax.clear()
        self._imshow_obj = self.ax.imshow(
            stretched, cmap='gray', origin='lower',
            vmin=0, vmax=1, interpolation='nearest'
        )
        stretch_name = self.scale_combo.currentText()
        self.ax.set_title(f"{self.current_filename} | {stretch_name}")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")

        if self.xlim_original is None:
            self.xlim_original = self.ax.get_xlim()
            self.ylim_original = self.ax.get_ylim()
        elif xlim_current is not None:
            self.ax.set_xlim(xlim_current)
            self.ax.set_ylim(ylim_current)

        self.canvas.draw_idle()

    def update_overlay(self):
        if self.idmatch_df is None or self.idmatch_df.empty:
            self.canvas.draw_idle()
            return
        for coll in self.ax.collections[:]:
            coll.remove()

        x = self.idmatch_df["x"].to_numpy(float)
        y = self.idmatch_df["y"].to_numpy(float)
        sids = self.idmatch_df["source_id"].astype("int64").to_numpy()
        in_master = np.array([sid in self.master_ids for sid in sids])

        if len(x):
            self.ax.scatter(x[~in_master], y[~in_master], s=20, facecolors='none',
                            edgecolors='yellow', linewidths=0.8, alpha=0.7)
            self.ax.scatter(x[in_master], y[in_master], s=30, facecolors='none',
                            edgecolors='lime', linewidths=1.2, alpha=0.8)

        if self.selected_source_id is not None:
            sel = self.idmatch_df[self.idmatch_df["source_id"] == self.selected_source_id]
            if len(sel):
                self.ax.scatter(sel["x"], sel["y"], s=60, facecolors='none',
                                edgecolors='red', linewidths=1.5, alpha=0.9)
        self.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        if event.button != 1:
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        self.setFocus()
        self.last_click_xy = (x, y)

        # Try to find nearby detected source
        search_r = float(getattr(self.params.P, "search_radius_px", 7.0))
        found_detected = False

        if self.idmatch_df is not None and not self.idmatch_df.empty:
            dx = self.idmatch_df["x"].to_numpy(float) - x
            dy = self.idmatch_df["y"].to_numpy(float) - y
            dist2 = dx * dx + dy * dy
            if dist2.size > 0:
                i = int(np.argmin(dist2))
                if dist2[i] <= search_r * search_r:
                    found_detected = True
                    self.selected_source_id = int(self.idmatch_df.iloc[i]["source_id"])
                    in_master = "✓ IN MASTER" if self.selected_source_id in self.master_ids else "✗ not in master"
                    internal_id = self.internal_id_map.get(self.selected_source_id, "-")
                    self.selected_label.setText(
                        f"Selected: ID {internal_id} | source_id: {self.selected_source_id} ({in_master})"
                    )

                    # Log with frame name
                    gaia_info = self.get_gaia_info(self.selected_source_id)
                    row = self.idmatch_df.iloc[i]
                    px_x, px_y = row["x"], row["y"]
                    frame = self.current_filename or "?"
                    self.log(f"[{frame}] Selected ID {internal_id}: {self.selected_source_id} | px=({px_x:.1f}, {px_y:.1f}) | {in_master}")
                    if gaia_info:
                        self.log(f"  Gaia: {gaia_info}")

                    # Auto-select in table if in master
                    if self.selected_source_id in self.master_ids:
                        self.select_source_in_table(self.selected_source_id)

                    self.update_overlay()

        if not found_detected:
            # No detected source nearby
            self.selected_source_id = None
            self.selected_label.setText(f"No detection at ({x:.1f}, {y:.1f}) - click on a circled star")
            self.update_overlay()

    def add_selected(self):
        """Add selected source to master (only detected sources with circles)"""
        frame = self.current_filename or "?"

        # Only allow adding detected sources (those with circles)
        if self.selected_source_id is None:
            self.log(f"[{frame}] No detected source selected - click on a circled star first")
            return

        if self.selected_source_id in self.master_ids:
            self.log(f"[{frame}] Already in master: {self.selected_source_id}")
            return

        self.master_ids.add(self.selected_source_id)
        gaia_info = self.get_gaia_info(self.selected_source_id)
        self.log(f"[{frame}] ✓ ADDED to master: {self.selected_source_id}")
        if gaia_info:
            self.log(f"  Gaia: {gaia_info}")
        self.save_master_ids(log_action="added")

    def remove_selected(self):
        frame = self.current_filename or "?"
        if self.selected_source_id is None:
            self.log(f"[{frame}] No source selected to remove")
            return
        if self.selected_source_id not in self.master_ids:
            self.log(f"[{frame}] Source {self.selected_source_id} not in master list")
            return
        internal_id = self.internal_id_map.get(self.selected_source_id, "?")
        gaia_info = self.get_gaia_info(self.selected_source_id)
        self.master_ids.remove(self.selected_source_id)
        self.log(f"[{frame}] ✗ REMOVED from master: ID {internal_id} ({self.selected_source_id})")
        if gaia_info:
            self.log(f"  Was: {gaia_info}")
        self.save_master_ids(log_action="removed")

    def remove_box(self):
        frame = self.current_filename or "?"
        if self.last_click_xy is None or self.idmatch_df is None or self.idmatch_df.empty:
            self.log(f"[{frame}] No position for box removal")
            return
        x0, y0 = self.last_click_xy
        box = int(getattr(self.params.P, "bulk_drop_box_px", 200))
        half = box / 2.0
        df = self.idmatch_df
        in_box = (df["x"].between(x0 - half, x0 + half) &
                  df["y"].between(y0 - half, y0 + half))
        sids = set(df.loc[in_box, "source_id"].astype("int64").tolist())
        # Only remove those that are in master
        to_remove = sids & self.master_ids
        if not to_remove:
            self.log(f"[{frame}] No master sources in box ({box}x{box}px at {x0:.0f},{y0:.0f})")
            return
        self.master_ids -= to_remove
        self.log(f"[{frame}] ✗ BOX REMOVED {len(to_remove)} sources from master ({box}x{box}px)")
        self.save_master_ids(log_action=f"box_removed_{len(to_remove)}")

    def save_master_ids(self, log_action: str = None):
        output_dir = step8_dir(self.params.P.result_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        master_path = output_dir / "master_star_ids.csv"
        backup_path = output_dir / "master_star_ids.orig.csv"
        if master_path.exists() and (not backup_path.exists()):
            try:
                backup_path.write_text(master_path.read_text(encoding="utf-8"), encoding="utf-8")
            except Exception:
                pass

        # Build sorted list with G mag
        id_gmag_list = []
        for sid in self.master_ids:
            g_mag = self._get_gmag_for_source(sid)
            id_gmag_list.append((sid, g_mag))

        # Sort by G mag (brightest first)
        id_gmag_list.sort(key=lambda x: (np.isnan(x[1]), x[1]))

        # Create DataFrame with internal_id
        rows = []
        for i, (sid, g_mag) in enumerate(id_gmag_list):
            rows.append({
                "internal_id": i + 1,
                "source_id": sid,
                "g_mag": g_mag if np.isfinite(g_mag) else None
            })
        df = pd.DataFrame(rows)
        df.to_csv(master_path, index=False)

        # Only log save summary if no specific action (e.g., on load or undo)
        if log_action is None:
            n_sources = len(self.master_ids)
            self.log(f"Saved {n_sources} sources to {master_path.name}")

        self.save_state()
        self.update_master_table()
        self.update_overlay()
        self.update_navigation_buttons()

    def _get_gmag_for_source(self, source_id: int) -> float:
        """Get G magnitude for a source_id from Gaia catalog"""
        if self.gaia_df is None or len(self.gaia_df) == 0:
            return float(self.master_gmag_map.get(int(source_id), np.nan))
        try:
            # Handle type mismatch - convert both to int64
            sid_col = self.gaia_df["source_id"]
            if sid_col.dtype != np.int64:
                sid_col = sid_col.astype(np.int64)
            mask = sid_col == int(source_id)
            row = self.gaia_df[mask]
            if len(row) > 0:
                val = float(row.iloc[0].get("phot_g_mean_mag", np.nan))
                if np.isfinite(val):
                    return val
        except Exception:
            pass
        return float(self.master_gmag_map.get(int(source_id), np.nan))

    def _refine_centroid(self, x: float, y: float) -> tuple | None:
        """
        Refine centroid near (x, y) to verify there's a real star.
        Returns (xc, yc, med) if star found, None otherwise.
        Ported from AAPKI_GUI.ipynb _refine_centroid function.
        """
        if self.image_data is None:
            return None

        img = self.image_data
        H, W = img.shape
        seed_fwhm_px = float(getattr(self.params.P, "fwhm_seed_px", 5.0))

        r = max(int(round(3.5 * max(seed_fwhm_px, 2.0))), 8)
        xi, yi = int(round(x)), int(round(y))
        x0, x1 = max(0, xi - r), min(W, xi + r + 1)
        y0, y1 = max(0, yi - r), min(H, yi + r + 1)

        if (x1 - x0) < 9 or (y1 - y0) < 9:
            return None

        cut = img[y0:y1, x0:x1]
        try:
            _, med, _ = sigma_clipped_stats(cut, sigma=3.0, maxiters=5, mask=~np.isfinite(cut))
        except Exception:
            return None

        Z = cut - med
        Z[~np.isfinite(Z)] = 0.0
        Z[Z < 0] = 0.0
        S = np.nansum(Z)

        if S <= 0:
            return None

        yy, xx = np.mgrid[y0:y1, x0:x1]
        xc = float(np.nansum(xx * Z) / S)
        yc = float(np.nansum(yy * Z) / S)

        return xc, yc, float(med)

    def _get_wcs(self) -> WCS | None:
        """Get WCS from header or .wcs file"""
        # First try header
        if self.header is not None:
            try:
                w = WCS(self.header, relax=True)
                if w.has_celestial:
                    return w
            except Exception:
                pass

        # Try .wcs file
        if self.current_filename:
            if self.use_cropped:
                cropped_dir = step2_cropped_dir(self.params.P.result_dir)
                if not cropped_dir.exists():
                    cropped_dir = self.params.P.result_dir / "cropped"
                fits_path = cropped_dir / self.current_filename
            else:
                fits_path = self.params.P.data_dir / self.current_filename
            wcs_path = fits_path.with_suffix(".wcs")
            if wcs_path.exists():
                try:
                    from astropy.io import fits as afits
                    with afits.open(wcs_path) as whdul:
                        w = WCS(whdul[0].header, relax=True)
                        if w.has_celestial:
                            return w
                except Exception:
                    pass

        return None

    def add_undetected_star(self):
        """
        Add undetected star at click position (A key for non-detected positions).
        1. Verify there's actually a star using centroid refinement
        2. Find matching Gaia source
        3. Add to master
        """
        frame = self.current_filename or "?"

        if self.last_click_xy is None:
            self.log(f"[{frame}] No position clicked")
            return

        x, y = self.last_click_xy

        # Check if this position is already in idmatch
        if self.idmatch_df is not None and not self.idmatch_df.empty:
            dx = self.idmatch_df["x"].to_numpy(float) - x
            dy = self.idmatch_df["y"].to_numpy(float) - y
            dist2 = dx * dx + dy * dy
            search_r = float(getattr(self.params.P, "search_radius_px", 7.0))
            if dist2.size > 0 and np.min(dist2) <= search_r * search_r:
                # Already detected - use normal add
                self.log(f"[{frame}] Position is already detected - using regular add")
                self.add_selected()
                return

        # Verify there's a star at this position
        centroid = self._refine_centroid(x, y)
        if centroid is None:
            self.log(f"[{frame}] No star detected at ({x:.1f}, {y:.1f}) - centroid refinement failed")
            return

        xc, yc, med = centroid
        self.log(f"[{frame}] Star detected at ({xc:.1f}, {yc:.1f}) - searching Gaia catalog...")

        # Get WCS for coordinate conversion
        w = self._get_wcs()
        if w is None:
            self.log(f"[{frame}] No WCS available - cannot match to Gaia")
            return

        # Convert pixel to sky coordinates
        try:
            sky = w.celestial.pixel_to_world(xc, yc)
        except Exception as e:
            self.log(f"[{frame}] WCS conversion failed: {e}")
            return

        # Match to Gaia catalog
        if self.gaia_df is None or len(self.gaia_df) == 0:
            self.log(f"[{frame}] Gaia catalog not loaded")
            return

        gaia = self.gaia_df
        if "ra" not in gaia.columns or "dec" not in gaia.columns or "source_id" not in gaia.columns:
            self.log(f"[{frame}] Gaia catalog missing required columns")
            return

        try:
            gsky = SkyCoord(
                ra=gaia["ra"].to_numpy(float) * u.deg,
                dec=gaia["dec"].to_numpy(float) * u.deg
            )
            idx, sep2d, _ = sky.match_to_catalog_sky(gsky)
            max_sep = float(getattr(self.params.P, "gaia_add_max_sep_arcsec", 2.0))
            sep_arcsec = float(sep2d.arcsec)

            if sep_arcsec > max_sep:
                self.log(f"[{frame}] No Gaia source within {max_sep}\" of ({xc:.1f}, {yc:.1f}) - nearest is {sep_arcsec:.2f}\"")
                return

            sid = int(gaia.iloc[int(idx)]["source_id"])
            if sid in self.master_ids:
                # Already in master - select it and show info
                internal_id = self.internal_id_map.get(sid, "?")
                gaia_info = self.get_gaia_info(sid)
                self.log(f"[{frame}] ★ Already in master: ID {internal_id} | source_id: {sid} (sep={sep_arcsec:.2f}\")")
                if gaia_info:
                    self.log(f"  Gaia: {gaia_info}")
                # Select this source and highlight in table
                self.selected_source_id = sid
                self.selected_label.setText(f"Selected: ID {internal_id} | source_id: {sid} (✓ IN MASTER - not detected in this frame)")
                self.select_source_in_table(sid)
                self.update_overlay()
                return

            self.master_ids.add(sid)
            gaia_info = self.get_gaia_info(sid)
            self.log(f"[{frame}] ✓ ADDED undetected star: {sid} (sep={sep_arcsec:.2f}\")")
            if gaia_info:
                self.log(f"  Gaia: {gaia_info}")
            self.save_master_ids(log_action="added_undetected")
            # Select the newly added source
            self.selected_source_id = sid

        except Exception as e:
            self.log(f"[{frame}] Gaia matching failed: {e}")

    def open_parameters_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Editor Parameters")
        dialog.resize(420, 280)
        layout = QVBoxLayout(dialog)
        form = QFormLayout()

        self.param_search = QDoubleSpinBox()
        self.param_search.setRange(1.0, 50.0)
        self.param_search.setValue(float(getattr(self.params.P, "search_radius_px", 7.0)))
        form.addRow("Search Radius (px):", self.param_search)

        self.param_box = QSpinBox()
        self.param_box.setRange(10, 2000)
        self.param_box.setValue(int(getattr(self.params.P, "bulk_drop_box_px", 200)))
        form.addRow("Remove Box Size (px):", self.param_box)

        self.param_gaia_sep = QDoubleSpinBox()
        self.param_gaia_sep.setRange(0.1, 10.0)
        self.param_gaia_sep.setDecimals(1)
        self.param_gaia_sep.setValue(float(getattr(self.params.P, "gaia_add_max_sep_arcsec", 2.0)))
        form.addRow("Gaia Add Max Sep (\"):", self.param_gaia_sep)

        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(lambda: self.save_parameters(dialog))
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec_()

    def save_parameters(self, dialog):
        self.params.P.search_radius_px = self.param_search.value()
        self.params.P.bulk_drop_box_px = self.param_box.value()
        self.params.P.gaia_add_max_sep_arcsec = self.param_gaia_sep.value()
        self.save_state()
        QMessageBox.information(dialog, "Success", "Parameters saved!")
        dialog.accept()

    def validate_step(self) -> bool:
        master_path = step8_dir(self.params.P.result_dir) / "master_star_ids.csv"
        if not master_path.exists():
            master_path = step6_dir(self.params.P.result_dir) / "master_star_ids.csv"
        if not master_path.exists():
            master_path = self.params.P.result_dir / "master_star_ids.csv"
        return master_path.exists()

    def save_state(self):
        state_data = {
            "search_radius_px": getattr(self.params.P, "search_radius_px", 7.0),
            "bulk_drop_box_px": getattr(self.params.P, "bulk_drop_box_px", 200),
            "gaia_add_max_sep_arcsec": getattr(self.params.P, "gaia_add_max_sep_arcsec", 2.0),
        }
        self.project_state.store_step_data("master_id_editor", state_data)

    def restore_state(self):
        state_data = self.project_state.get_step_data("master_id_editor")
        if state_data:
            for key, val in state_data.items():
                if hasattr(self.params.P, key):
                    setattr(self.params.P, key, val)

    def show_log_window(self):
        self.log_window.show()
        self.log_window.raise_()
        self.log_window.activateWindow()

    def step_frame(self, delta: int):
        if not self.file_list:
            return
        idx = self.file_combo.currentIndex()
        idx = max(0, min(len(self.file_list) - 1, idx + delta))
        if idx != self.file_combo.currentIndex():
            self.file_combo.setCurrentIndex(idx)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_A:
            # If a detected source is selected, add it
            # Otherwise, try to add undetected star at click position
            if self.selected_source_id is not None:
                self.add_selected()
            else:
                self.add_undetected_star()
            return
        if event.key() == Qt.Key_D:
            if event.modifiers() & Qt.ShiftModifier:
                self.remove_box()
            else:
                self.remove_selected()
            return
        if event.key() == Qt.Key_G:
            self.show_radial_profile()
            return
        if event.key() == Qt.Key_BracketLeft or event.key() == Qt.Key_Comma:
            self.step_frame(-1)
            return
        if event.key() == Qt.Key_BracketRight or event.key() == Qt.Key_Period:
            self.step_frame(1)
            return
        super().keyPressEvent(event)

    def show_radial_profile(self):
        """Show radial profile at mouse hover position (G key) - updates if already open"""
        if self.image_data is None:
            self.log("No image loaded for radial profile")
            return

        # Use hover position (preferred) or last click position
        if self.hover_xy is not None:
            x, y = self.hover_xy
        elif self.last_click_xy is not None:
            x, y = self.last_click_xy
        else:
            self.log("Move mouse over image first for radial profile")
            return
        frame = self.current_filename or "?"

        try:
            from astropy.stats import sigma_clipped_stats
            from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

            xc, yc = float(x), float(y)

            # Calculate radial profile
            rmax = 50
            ry0 = max(0, int(yc - rmax))
            ry1 = min(self.image_data.shape[0], int(yc + rmax))
            rx0 = max(0, int(xc - rmax))
            rx1 = min(self.image_data.shape[1], int(xc + rmax))

            region = self.image_data[ry0:ry1, rx0:rx1]
            yy, xx = np.mgrid[ry0:ry1, rx0:rx1]
            rr = np.sqrt((xx - xc)**2 + (yy - yc)**2)

            # Background subtraction
            _, reg_median, _ = sigma_clipped_stats(region, sigma=3.0)
            region_sub = region - reg_median

            # Radial bins
            dr = 0.5
            edges = np.arange(0, rmax, dr)
            centers = 0.5 * (edges[:-1] + edges[1:])
            profile = np.full_like(centers, np.nan)

            for i in range(len(centers)):
                mask = (rr >= edges[i]) & (rr < edges[i+1])
                if np.any(mask):
                    vals = region_sub[mask]
                    vals = vals[np.isfinite(vals)]
                    if vals.size > 0:
                        profile[i] = np.mean(vals)

            # Create dialog if not exists
            if not hasattr(self, 'radial_dialog') or self.radial_dialog is None or not self.radial_dialog.isVisible():
                self.radial_dialog = QDialog(self)
                self.radial_dialog.setWindowTitle("Radial Profile (G key to update)")
                self.radial_dialog.resize(600, 400)

                layout = QVBoxLayout(self.radial_dialog)
                self.prof_fig = Figure(figsize=(6, 4))
                self.prof_canvas = FigureCanvas(self.prof_fig)
                self.prof_ax = self.prof_fig.add_subplot(111)
                layout.addWidget(NavigationToolbar(self.prof_canvas, self.radial_dialog))
                layout.addWidget(self.prof_canvas)

            # Plot (always update)
            self.prof_ax.clear()
            self.prof_ax.plot(centers, profile, 'o-', color='steelblue', markersize=3, linewidth=1)
            self.prof_ax.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
            self.prof_ax.set_xlabel('Radius (pixels)')
            self.prof_ax.set_ylabel('Pixel Value - Background (ADU)')
            self.prof_ax.set_title(f'{frame} | Position ({int(xc)}, {int(yc)})')
            self.prof_ax.grid(True, alpha=0.3)

            # Estimate FWHM
            fwhm_note = ""
            peak = np.nanmax(profile) if np.isfinite(profile).any() else 0
            if peak > 0:
                half = 0.5 * peak
                idx = np.where((profile[:-1] >= half) & (profile[1:] < half))[0]
                if len(idx) > 0:
                    i = idx[0]
                    x1_p, y1_p = centers[i], profile[i]
                    x2_p, y2_p = centers[i+1], profile[i+1]
                    if y1_p != y2_p:
                        r_half = x1_p + (half - y1_p) * (x2_p - x1_p) / (y2_p - y1_p)
                        fwhm_px = 2.0 * r_half
                        pixscale = float(getattr(self.params.P, "pixel_scale_arcsec", 0.4))
                        fwhm_arcsec = fwhm_px * pixscale
                        self.prof_ax.axvline(r_half, color='orange', linestyle='--', linewidth=2)
                        fwhm_note = f'FWHM: {fwhm_arcsec:.2f}" ({fwhm_px:.2f} px)'
                        self.prof_ax.text(0.02, 0.95, fwhm_note,
                                         transform=self.prof_ax.transAxes, ha='left', va='top',
                                         fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
                        self.log(f"[{frame}] Radial profile at ({int(xc)},{int(yc)}): FWHM={fwhm_arcsec:.2f}\" ({fwhm_px:.2f}px)")

            self.prof_fig.tight_layout()
            self.prof_canvas.draw()
            self.radial_dialog.show()
            self.radial_dialog.raise_()

        except Exception as e:
            self.log(f"[{frame}] Radial profile error: {e}")

    # Zoom/pan from Step4
    def reset_zoom(self):
        if self.xlim_original is not None:
            self.ax.set_xlim(self.xlim_original)
            self.ax.set_ylim(self.ylim_original)
            self.canvas.draw_idle()

    # Stretch functions (from Step4)
    def on_stretch_changed(self, index):
        self._normalized_cache = None
        self.display_image()

    def update_stretch_label(self, value):
        self.stretch_value_label.setText(str(value))

    def update_black_label(self, value):
        self.black_value_label.setText(str(value))

    def redisplay_image(self):
        self.display_image()

    def normalize_image(self):
        if self.image_data is None:
            return None

        stretch_idx = self.scale_combo.currentIndex()
        cache_key = (id(self.image_data), stretch_idx)
        if self._normalized_cache is not None:
            if self._normalized_cache[0] == cache_key:
                return self._normalized_cache[1].copy()

        finite = np.isfinite(self.image_data)
        if not finite.any():
            return np.zeros_like(self.image_data)

        data = self.image_data.copy()

        if stretch_idx == 6:  # Linear (1-99%)
            vmin = np.percentile(data[finite], 1)
            vmax = np.percentile(data[finite], 99)
        elif stretch_idx == 7:  # ZScale (IRAF)
            vmin, vmax = self.calculate_zscale()
        else:
            mean_val, median_val, std_val = sigma_clipped_stats(data[finite], sigma=3.0, maxiters=5)
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
        mean_val, median_val, std_val = sigma_clipped_stats(data, sigma=3.0, maxiters=5)

        vmin = float(median_val - 2.8 * std_val)
        vmax_percentile = np.percentile(data, 99.5)
        vmax_sigma = median_val + 6.0 * std_val
        vmax = float(min(vmax_percentile, vmax_sigma))

        if vmax <= vmin:
            vmin = float(np.min(data))
            vmax = float(np.max(data))

        return vmin, vmax

    def apply_stretch(self, data):
        stretch_idx = self.scale_combo.currentIndex()
        intensity = self.stretch_slider.value() / 100.0
        black_point = self.black_slider.value() / 100.0

        data = np.clip((data - black_point) / (1.0 - black_point + 1e-10), 0, 1)

        if stretch_idx == 0:  # Auto Stretch (Siril)
            return self.stretch_auto_siril(data, intensity)
        if stretch_idx == 1:  # Asinh
            return self.stretch_asinh(data, intensity)
        if stretch_idx == 2:  # MTF
            return self.stretch_mtf(data, intensity)
        if stretch_idx == 3:  # Histogram Eq
            return self.stretch_histogram_eq(data)
        if stretch_idx == 4:  # Log
            return self.stretch_log(data, intensity)
        if stretch_idx == 5:  # Sqrt
            return self.stretch_sqrt(data, intensity)
        return data

    def stretch_auto_siril(self, data, intensity):
        finite = data[np.isfinite(data)]
        if len(finite) == 0:
            return data

        median_val = np.median(finite)
        mad = np.median(np.abs(finite - median_val))
        sigma = mad * 1.4826

        shadows = max(0, median_val - 2.8 * sigma)
        stretched = (data - shadows) / (1.0 - shadows + 1e-10)
        stretched = np.clip(stretched, 0, 1)

        midtone = 0.15 + (1.0 - intensity) * 0.35
        return self.mtf_function(stretched, midtone)

    def stretch_asinh(self, data, intensity):
        beta = 1.0 + intensity * 15.0
        stretched = np.arcsinh(data * beta) / np.arcsinh(beta)
        return np.clip(stretched, 0, 1)

    def stretch_mtf(self, data, intensity):
        midtone = 0.05 + (1.0 - intensity) * 0.45
        return self.mtf_function(data, midtone)

    def mtf_function(self, data, midtone):
        m = np.clip(midtone, 0.001, 0.999)
        result = np.zeros_like(data)
        mask = data > 0
        result[mask] = (m - 1) * data[mask] / ((2 * m - 1) * data[mask] - m)
        result[data == 0] = 0
        result[data == 1] = 1
        return np.clip(result, 0, 1)

    def stretch_histogram_eq(self, data):
        finite = data[np.isfinite(data)]
        if len(finite) == 0:
            return data

        hist, bin_edges = np.histogram(finite.flatten(), bins=65536, range=(0, 1))
        cdf = hist.cumsum()
        cdf = cdf / cdf[-1]
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        return np.clip(np.interp(data, bin_centers, cdf), 0, 1)

    def stretch_log(self, data, intensity):
        a = 100 + intensity * 900
        return np.clip(np.log(1 + a * data) / np.log(1 + a), 0, 1)

    def stretch_sqrt(self, data, intensity):
        power = 0.2 + (1.0 - intensity) * 0.8
        return np.clip(np.power(data, power), 0, 1)

    def on_scroll(self, event):
        if event.inaxes != self.ax:
            return
        scale = 1.2 if event.button == 'down' else 1 / 1.2
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        xdata, ydata = event.xdata, event.ydata
        new_width = (xlim[1] - xlim[0]) * scale
        new_height = (ylim[1] - ylim[0]) * scale
        relx = (xlim[1] - xdata) / (xlim[1] - xlim[0])
        rely = (ylim[1] - ydata) / (ylim[1] - ylim[0])
        self.ax.set_xlim([xdata - new_width * (1 - relx), xdata + new_width * relx])
        self.ax.set_ylim([ydata - new_height * (1 - rely), ydata + new_height * rely])
        self.canvas.draw_idle()

    def on_button_press(self, event):
        if event.button == 3:
            self.panning = True
            self.pan_start = (event.xdata, event.ydata)

    def on_button_release(self, event):
        if event.button == 3:
            self.panning = False
            self.pan_start = None

    def on_motion(self, event):
        # Track hover position for G key
        if event.inaxes == self.ax and event.xdata is not None and event.ydata is not None:
            self.hover_xy = (event.xdata, event.ydata)

        if not self.panning or event.inaxes != self.ax:
            return
        if self.pan_start is None or event.xdata is None:
            return
        dx = self.pan_start[0] - event.xdata
        dy = self.pan_start[1] - event.ydata
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        self.ax.set_xlim([xlim[0] + dx, xlim[1] + dx])
        self.ax.set_ylim([ylim[0] + dy, ylim[1] + dy])
        self.canvas.draw_idle()
