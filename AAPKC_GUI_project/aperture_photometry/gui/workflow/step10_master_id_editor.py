"""
Step 10: Master Star IDs Editor
Ported from AAPKI_GUI.ipynb Cell 10 (GUI adaptation).
"""

from __future__ import annotations

import time
import json
import threading
from concurrent.futures import ThreadPoolExecutor
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
from matplotlib.patches import Circle as MplCircle

from PyQt5.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QGroupBox, QMessageBox,
    QTextEdit, QDialog, QFormLayout, QDialogButtonBox, QProgressBar,
    QCheckBox, QSpinBox, QDoubleSpinBox, QLineEdit, QTableWidget,
    QTableWidgetItem, QHeaderView, QAbstractItemView, QWidget, QComboBox,
    QSlider, QColorDialog, QFrame
)
from PyQt5.QtGui import QColor
from PyQt5.QtCore import Qt, QPoint

from .step_window_base import StepWindowBase
from ...utils.step_paths import step2_cropped_dir, step5_dir, step6_dir, step7_dir, step8_dir, step9_dir, step6_psf_dir, crop_is_active
from ...utils.io_utils import (
    parse_int64_scalar,
    parse_int64_series,
    read_csv_int64_source_id,
    read_ecsv_int64_source_id,
)


class MasterIdEditorWindow(StepWindowBase):
    """Step 10: Master Star IDs Editor"""

    def __init__(self, params, file_manager, project_state, main_window):
        self.file_manager = file_manager
        self.file_list = []
        self.use_cropped = False
        self.current_filename = None
        self.image_data = None
        self.header = None
        self._file_filter_map = {}
        self.idmatch_df = None
        self.master_ids = set()
        self.selected_source_id = None
        self.last_click_xy = None
        self.gaia_df = None  # Gaia catalog cache
        self._gaia_gmag_map = {}
        self.membership_by_source = {}
        self.membership_by_gaia = {}
        self.membership_by_id = {}
        self._membership_loaded = False
        self._membership_log_once = False
        self.master_gmag_map = {}
        # Fixed ID map (source_id -> display ID); IDs must stay stable across sessions.
        self.internal_id_map = {}
        self.source_id_from_internal = {}
        self._global_id_map = {}
        self._auto_master_dirty = False

        # Matplotlib components
        self.figure = None
        self.canvas = None
        self.ax = None
        self._imshow_obj = None
        self._normalized_cache = None   # single-slot legacy (unused now)
        self._norm_cache: dict = {}     # (filename, stretch_idx) -> normalized_array (LRU 5)
        self.xlim_original = None
        self.ylim_original = None
        self.panning = False
        self.pan_start = None
        self.hover_xy = None  # Track mouse hover position for G key

        # Persistent scatter artists (avoids remove/recreate each overlay update)
        self._scat_unmatched = None
        self._scat_removed = None
        self._scat_local = None
        self._scat_gaia = None
        self._scat_member = None
        self._scat_selected = None
        self._scat_psf_iter2 = None

        # PSF iter2 source tracking (det_uid < 0)
        self.psf_iter2_ids: set = set()

        # CMD ROI (circle in image pixel coords, saved to cmd_roi.json)
        self._roi_circle: dict | None = None   # {cx, cy, radius}
        self._roi_patch = None                 # matplotlib Circle artist on ax
        self._roi_preview_patch = None         # temporary preview during drag
        self._roi_mode = False                 # draw-mode toggle
        self._roi_drag_start: tuple | None = None

        # Overlay color customization
        self._overlay_colors: dict = {
            "gaia":      "#00FF00",
            "member":    "#FF4DA6",
            "psf_iter2": "#FF5722",
            "local":     "#00BCD4",
            "removed":   "#FFEB3B",
            "unmatched": "#FF9800",
            "selected":  "#FF0000",
        }
        self._color_window: "QWidget | None" = None
        self._color_btns: dict = {}
        self._overlay_visible: dict = {k: True for k in (
            "gaia", "member", "psf_iter2", "local", "removed", "unmatched", "selected"
        )}

        # Frame data caches: filename -> data  (LRU, max _FITS_CACHE_SIZE entries)
        self._fits_cache: dict = {}        # filename -> (image_data, header)
        self._fits_cache_order: list = []  # LRU order
        self._FITS_CACHE_SIZE = max(3, int(getattr(params.P, "step8_fits_cache_size", 8)))
        self._idmatch_cache: dict = {}     # filename -> idmatch_df (all frames, small)
        self._idmatch_arr_cache: dict = {} # filename -> (x, y, sids) numpy arrays
        self._prefetch_lock = threading.Lock()
        self._prefetch_pending: set[str] = set()
        self._prefetch_executor = ThreadPoolExecutor(max_workers=1)
        # Display cache: (filename, stretch_idx, intensity, black_point) -> stretched image (LRU)
        self._display_cache: dict = {}
        self._display_cache_order: list = []

        super().__init__(
            step_index=9,
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
            "Shortcuts: A=Add (detected or undetected star), D=Remove, Shift+D=Remove Box, G=Radial Profile (at cursor), [ / ] = Prev/Next frame\n"
            "Overlay: Gaia(master)=green, Membership(member)=pink (when enabled)"
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

        self.btn_colors = QPushButton("🎨 Colors")
        self.btn_colors.setCheckable(True)
        self.btn_colors.setStyleSheet(
            "QPushButton { background-color: #455A64; color: white; font-weight: bold; padding: 8px 12px; }"
            "QPushButton:checked { background-color: #78909C; }"
        )
        self.btn_colors.toggled.connect(self._toggle_color_panel)
        control_layout.addWidget(self.btn_colors)

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

        # CMD ROI controls
        roi_layout = QHBoxLayout()
        self.btn_set_roi = QPushButton("Set CMD ROI")
        self.btn_set_roi.setCheckable(True)
        self.btn_set_roi.setToolTip("Click to enter ROI draw mode, then click+drag on the image to define a circle.\nThis ROI filters sources shown in the CMD (ZP/photometry are unaffected).")
        self.btn_set_roi.setStyleSheet(
            "QPushButton { background-color: #37474F; color: white; font-weight: bold; padding: 4px 10px; }"
            "QPushButton:checked { background-color: #00BCD4; color: black; }"
        )
        self.btn_set_roi.toggled.connect(self._on_set_roi_toggled)
        roi_layout.addWidget(self.btn_set_roi)

        self.btn_clear_roi = QPushButton("Clear ROI")
        self.btn_clear_roi.setStyleSheet("QPushButton { background-color: #546E7A; color: white; padding: 4px 10px; }")
        self.btn_clear_roi.clicked.connect(self._on_clear_roi)
        roi_layout.addWidget(self.btn_clear_roi)

        self.roi_info_label = QLabel("No ROI set")
        self.roi_info_label.setStyleSheet("QLabel { color: #90A4AE; font-size: 9pt; }")
        roi_layout.addWidget(self.roi_info_label)
        roi_layout.addStretch()
        viewer_layout.addLayout(roi_layout)

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
        self.master_table.setColumnCount(4)
        self.master_table.setHorizontalHeaderLabels(["ID", "source_id", "G mag", "Pmem"])
        self.master_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.master_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.master_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.master_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.master_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.master_table.itemSelectionChanged.connect(self.on_table_selection_changed)
        table_layout.addWidget(self.master_table)

        main_layout.addWidget(table_group, stretch=1)

        self.content_layout.addLayout(main_layout)

        # Floating color-legend window (built now, shown on demand)
        self._color_window = self._build_color_window()

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
        self.load_psf_new_sources()
        self.load_gaia_catalog()
        self._load_roi()

    def log(self, message: str):
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")

    def populate_file_list(self):
        self._file_filter_map = {}
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

        base_count = len(files)
        # Hide unsolved frames: keep only Step7 rows with wcs_ok=True when available.
        stats_path = step7_dir(self.params.P.result_dir) / "step7_frame_stats.csv"
        if stats_path.exists():
            try:
                s7 = pd.read_csv(stats_path)
            except Exception:
                s7 = pd.DataFrame()
            if (not s7.empty) and ("file" in s7.columns):
                if "filter" in s7.columns:
                    for fname, filt in zip(s7["file"].astype(str), s7["filter"].astype(str)):
                        self._file_filter_map[fname] = self._normalize_filter_key(filt)
                if "wcs_ok" in s7.columns:
                    wcs_ok = s7["wcs_ok"].astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "on"})
                else:
                    wcs_ok = pd.Series(True, index=s7.index, dtype=bool)
                keep = set(s7.loc[wcs_ok, "file"].astype(str).tolist())
                files = [f for f in files if f in keep]
                self.log(f"Step10 frame filter (wcs_ok): {len(files)}/{base_count} kept")

        # Also hide frames without Step9 ID-match output.
        idmatch_dir = self.params.P.cache_dir / "idmatch"
        if idmatch_dir.exists():
            before_idm = len(files)
            files = [f for f in files if (idmatch_dir / f"idmatch_{f}.csv").exists()]
            self.log(f"Step10 frame filter (Step9 idmatch csv): {len(files)}/{before_idm} kept")

        self.file_list = list(files)
        self.file_combo.clear()
        self.file_combo.addItems(self.file_list)
        self._load_filter_map_from_index()
        with self._prefetch_lock:
            self._fits_cache.clear()
            self._fits_cache_order.clear()
            self._idmatch_arr_cache.clear()
            self._display_cache.clear()
            self._display_cache_order.clear()
            self._norm_cache.clear()
            self._prefetch_pending.clear()

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

    def _load_filter_map_from_index(self):
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
            fkey = self._normalize_filter_key(filt)
            if not fkey:
                fkey = self._infer_filter_from_filename(fname)
            self._file_filter_map[fname] = fkey
            if fname.lower().startswith("crop_"):
                self._file_filter_map[fname[5:]] = fkey
            if fname.lower().startswith("cropped_"):
                self._file_filter_map[fname[8:]] = fkey

    def _get_filter_for_file(self, fname: str) -> str:
        fkey = self._file_filter_map.get(fname, "")
        if fkey:
            return fkey
        if fname == self.current_filename and self.header is not None:
            fkey = self._extract_filter_from_header(self.header)
        if not fkey:
            fkey = self._infer_filter_from_filename(fname)
        self._file_filter_map[fname] = fkey
        return fkey

    def load_master_ids(self):
        master_path = step8_dir(self.params.P.result_dir) / "master_star_ids.csv"
        if not master_path.exists():
            master_path = step6_dir(self.params.P.result_dir) / "master_star_ids.csv"
        if not master_path.exists():
            master_path = self.params.P.result_dir / "master_star_ids.csv"
        self.internal_id_map = {}
        self.source_id_from_internal = {}
        self._load_global_id_map()
        if master_path.exists():
            try:
                df = pd.read_csv(master_path)
                if "source_id" in df.columns:
                    sid_vals = parse_int64_series(df["source_id"]).dropna().astype("int64")
                    self.master_ids = set(sid_vals.tolist())
                    if "g_mag" in df.columns:
                        gmag_series = pd.to_numeric(df["g_mag"], errors="coerce")
                        sid_series = parse_int64_series(df["source_id"])
                        valid_sid = sid_series.notna()
                        sid_clean = sid_series.loc[valid_sid].astype("int64")
                        gmag_clean = gmag_series.loc[valid_sid]
                        self.master_gmag_map = dict(zip(sid_clean.tolist(), gmag_clean.tolist()))
                    # Repair legacy float-rounded Gaia IDs from old Step8 save files.
                    sid_repair_map = self._repair_legacy_step8_source_ids()
                    # Prefer saved fixed ID mapping when available.
                    if "ID" in df.columns:
                        id_vals = pd.to_numeric(df["ID"], errors="coerce")
                    elif "internal_id" in df.columns:
                        # Legacy compatibility.
                        id_vals = pd.to_numeric(df["internal_id"], errors="coerce")
                    else:
                        id_vals = pd.Series([np.nan] * len(df))
                    sid_series = parse_int64_series(df["source_id"])
                    for sid_v, id_v in zip(sid_series.tolist(), id_vals):
                        if not (pd.notna(sid_v) and np.isfinite(id_v)):
                            continue
                        sid_i = int(sid_repair_map.get(int(sid_v), int(sid_v)))
                        id_i = int(id_v)
                        if sid_i in self.internal_id_map:
                            continue
                        if id_i in self.source_id_from_internal and self.source_id_from_internal[id_i] != sid_i:
                            continue
                        self.internal_id_map[sid_i] = id_i
                        self.source_id_from_internal[id_i] = sid_i
                    for sid_i in sorted(self.master_ids):
                        self._ensure_stable_id(sid_i)
                    # Fallback: preload Gaia G magnitudes from Step8 RefBuild catalog (legacy path).
                    self._load_step6_gmag_map()
                    self.log(f"Loaded {len(self.master_ids)} master IDs from {master_path.name}")
                    self.update_master_table()
            except Exception as e:
                self.log(f"Error loading master IDs: {e}")

    def load_psf_new_sources(self):
        """Step 6 PSF iter2에서 발견된 새 소스(det_uid < 0)를 마스터 목록에 추가.

        PSF 측광 iter2에서 잔차 이미지로부터 새로 검출된 별들은 음수 det_uid를
        가진다. 이 소스들은 RefBuild ref catalog에 없으므로 별도로 master_id를
        부여하여 마스터 목록에 추가한다.
        """
        psf_dir = step6_psf_dir(self.params.P.result_dir)
        if not psf_dir.exists():
            return

        rows = []
        for tsv in sorted(psf_dir.glob("photometry_*.tsv")):
            try:
                df = pd.read_csv(tsv, sep="\t")
                if "det_uid" not in df.columns or "iter_found" not in df.columns:
                    continue
                new = df[(pd.to_numeric(df["det_uid"], errors="coerce") < 0) &
                         (pd.to_numeric(df["iter_found"], errors="coerce") > 1)].copy()
                if not new.empty:
                    fname = tsv.name.replace("photometry_", "").replace(".tsv", "")
                    new["_source_file"] = fname
                    rows.append(new)
            except Exception:
                continue

        if not rows:
            return

        all_new = pd.concat(rows, ignore_index=True)
        # per det_uid 대표값 (첫 번째 등장 프레임 기준)
        unique_new = all_new.groupby("det_uid", as_index=False).first()
        n_new = len(unique_new)
        if n_new == 0:
            return

        # 기존 master_id 최대값 다음부터 부여 (Gaia source_id와 충돌 없이 음수 psf_uid 사용)
        # PSF 새 소스는 source_id = det_uid (음수) 로 내부 식별
        added = 0
        for _, row in unique_new.iterrows():
            det_uid = int(row["det_uid"])
            if det_uid not in self.internal_id_map:
                self._ensure_stable_id(det_uid)
                self.master_ids.add(det_uid)
                added += 1
            self.psf_iter2_ids.add(det_uid)

        if added > 0:
            self.log(f"PSF new sources: {added} iter2 소스 master_id 부여 완료 (det_uid<0)")
            self.update_master_table()

    def _repair_legacy_step8_source_ids(self) -> dict[int, int]:
        """Repair legacy Step8 source_ids affected by float rounding of Gaia IDs.

        Old runs could save Gaia source_id via float path, producing offsets that are
        typically multiples of 128. This remaps positive IDs to Step8 canonical IDs
        when a unique nearby candidate exists.
        """
        sid_map: dict[int, int] = {}
        step6_master = step6_dir(self.params.P.result_dir) / "master_star_ids.csv"
        if not step6_master.exists():
            return sid_map
        try:
            df6 = pd.read_csv(step6_master)
            s6 = parse_int64_series(df6.get("source_id", pd.Series(dtype="Int64"))).dropna().astype("int64")
            canon_pos = {int(v) for v in s6.tolist() if int(v) > 0}
        except Exception:
            return sid_map
        if not canon_pos:
            return sid_map

        repaired_ids: set[int] = set()
        repaired_count = 0
        ambiguous = 0
        unresolved = 0
        for sid in sorted(int(v) for v in self.master_ids):
            if sid <= 0:
                repaired_ids.add(sid)
                continue
            if sid in canon_pos:
                repaired_ids.add(sid)
                continue
            cands: list[tuple[int, int]] = []
            for delta in (-128, 128, -256, 256, -384, 384, -512, 512):
                cand = sid + delta
                if cand in canon_pos:
                    cands.append((abs(delta), cand))
            if not cands:
                unresolved += 1
                repaired_ids.add(sid)
                continue
            cands.sort(key=lambda x: x[0])
            best_abs = cands[0][0]
            best = sorted({cand for ad, cand in cands if ad == best_abs})
            if len(best) != 1:
                ambiguous += 1
                repaired_ids.add(sid)
                continue
            new_sid = int(best[0])
            repaired_ids.add(new_sid)
            sid_map[sid] = new_sid
            repaired_count += 1

        if sid_map:
            # Move any saved g_mag entries to repaired keys.
            for old_sid, new_sid in sid_map.items():
                if old_sid in self.master_gmag_map and new_sid not in self.master_gmag_map:
                    self.master_gmag_map[new_sid] = self.master_gmag_map[old_sid]
            self.master_ids = repaired_ids
            self.log(
                "[Step10] Repaired legacy source_id precision drift: "
                f"mapped={repaired_count}, ambiguous={ambiguous}, unresolved={unresolved}"
            )
        return sid_map

    def _load_step6_gmag_map(self):
        """Fallback G-mag map from Step8 RefBuild catalog (legacy step6_refbuild path)."""
        candidates = [
            step6_dir(self.params.P.result_dir) / "ref_catalog.tsv",
            self.params.P.result_dir / "ref_catalog.tsv",
        ]
        for path in candidates:
            if not path.exists():
                continue
            try:
                df = pd.read_csv(path, sep="\t")
            except Exception:
                continue
            if "source_id" not in df.columns:
                continue
            g_col = None
            for cand in ("phot_g_mean_mag", "gaia_G", "g_mag"):
                if cand in df.columns:
                    g_col = cand
                    break
            if g_col is None:
                continue
            sid = parse_int64_series(df["source_id"])
            g = pd.to_numeric(df[g_col], errors="coerce")
            valid = sid.notna() & g.notna()
            if not valid.any():
                continue
            sid_i = sid.loc[valid].astype("int64")
            g_v = g.loc[valid].astype(float)
            loaded = 0
            for s, gv in zip(sid_i.tolist(), g_v.tolist()):
                if not np.isfinite(gv):
                    continue
                self.master_gmag_map[int(s)] = float(gv)
                loaded += 1
            if loaded > 0:
                self.log(f"Gmag fallback loaded from {path.name}: {loaded} rows")
                break

    def _load_global_id_map(self):
        """Load source_id -> ID map generated by Step 8/9 (legacy path compatible)."""
        self._global_id_map = {}
        candidates = [
            step6_dir(self.params.P.result_dir) / "sourceid_to_ID.csv",
            step7_dir(self.params.P.result_dir) / "sourceid_to_ID.csv",
            self.params.P.result_dir / "sourceid_to_ID.csv",
        ]
        for path in candidates:
            if not path.exists():
                continue
            try:
                df = read_csv_int64_source_id(path)
            except Exception:
                continue
            if not {"source_id", "ID"} <= set(df.columns):
                continue
            sid_vals = parse_int64_series(df["source_id"])
            id_vals = pd.to_numeric(df["ID"], errors="coerce")
            for sid_v, id_v in zip(sid_vals, id_vals):
                if pd.notna(sid_v) and np.isfinite(id_v):
                    self._global_id_map[int(sid_v)] = int(id_v)
            if self._global_id_map:
                break

    def _next_available_id(self) -> int:
        used = {int(v) for v in self.source_id_from_internal.keys() if int(v) > 0}
        used.update(int(v) for v in self._global_id_map.values() if int(v) > 0)
        if not used:
            return 1
        return max(used) + 1

    def _ensure_stable_id(self, source_id: int) -> int:
        sid = int(source_id)
        if sid in self.internal_id_map:
            return int(self.internal_id_map[sid])

        # Prefer global ID mapping from upstream ID match when conflict-free.
        gid = self._global_id_map.get(sid)
        if gid is not None:
            gid = int(gid)
            owner = self.source_id_from_internal.get(gid)
            if owner is None or int(owner) == sid:
                self.internal_id_map[sid] = gid
                self.source_id_from_internal[gid] = sid
                return gid

        # Allocate new stable ID (never per-frame re-number).
        new_id = self._next_available_id()
        while new_id in self.source_id_from_internal:
            new_id += 1
        self.internal_id_map[sid] = int(new_id)
        self.source_id_from_internal[int(new_id)] = sid
        return int(new_id)

    def load_gaia_catalog(self):
        """Load Gaia catalog for source info lookup"""
        candidates = [
            step5_dir(self.params.P.result_dir) / "gaia_derived.csv",
            self.params.P.result_dir / "gaia_derived.csv",
            step5_dir(self.params.P.result_dir) / "gaia_fov.ecsv",
            self.params.P.result_dir / "gaia_fov.ecsv",
        ]
        for gaia_path in candidates:
            if not gaia_path.exists():
                continue
            try:
                if gaia_path.suffix.lower() == ".ecsv":
                    self.gaia_df = read_ecsv_int64_source_id(gaia_path)
                else:
                    self.gaia_df = pd.read_csv(gaia_path)
                if "source_id" in self.gaia_df.columns:
                    sid_series = parse_int64_series(self.gaia_df["source_id"])
                    valid = sid_series.notna()
                    dropped = int((~valid).sum())
                    if dropped > 0:
                        self.gaia_df = self.gaia_df.loc[valid].copy()
                        sid_series = sid_series.loc[valid]
                        self.log(f"Gaia catalog: dropped {dropped} rows with invalid source_id")
                    self.gaia_df["source_id"] = sid_series.astype("int64")
                self._rebuild_gaia_gmag_map()
                self.log(f"Gaia catalog loaded: {len(self.gaia_df)} sources ({gaia_path.name})")
                if self.master_ids:
                    self.update_master_table()
                    self.update_overlay()
                return
            except Exception as e:
                self.log(f"Failed to load Gaia catalog ({gaia_path.name}): {e}")
                self.gaia_df = None
                self._gaia_gmag_map = {}

    def _rebuild_gaia_gmag_map(self):
        self._gaia_gmag_map = {}
        if self.gaia_df is None or len(self.gaia_df) == 0:
            return
        if "source_id" not in self.gaia_df.columns:
            return
        g_col = None
        for cand in ("phot_g_mean_mag", "gaia_G", "g_mag"):
            if cand in self.gaia_df.columns:
                g_col = cand
                break
        if g_col is None:
            return
        sid = parse_int64_series(self.gaia_df["source_id"])
        g = pd.to_numeric(self.gaia_df[g_col], errors="coerce")
        valid = sid.notna() & g.notna()
        if not valid.any():
            return
        sid_i = sid.loc[valid].astype("int64").to_numpy(np.int64, copy=False)
        g_v = g.loc[valid].astype(float).to_numpy(float, copy=False)
        for s, gv in zip(sid_i.tolist(), g_v.tolist()):
            if np.isfinite(gv):
                self._gaia_gmag_map[int(s)] = float(gv)

    @staticmethod
    def _pick_membership_col(cols) -> str | None:
        for c in ("gaia_pmem", "pmem_gaia", "membership_prob_gaia", "membership_prob", "pmem"):
            if c in cols:
                return c
        return None

    @staticmethod
    def _logpdf_gauss(x: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
        d = x.shape[1]
        cov_r = np.asarray(cov, float) + np.eye(d) * 1e-6
        sign, logdet = np.linalg.slogdet(cov_r)
        if sign <= 0:
            return np.full(x.shape[0], -np.inf, dtype=float)
        try:
            inv = np.linalg.inv(cov_r)
        except Exception:
            return np.full(x.shape[0], -np.inf, dtype=float)
        diff = x - mu[None, :]
        q = np.einsum("ni,ij,nj->n", diff, inv, diff)
        return -0.5 * (d * np.log(2.0 * np.pi) + logdet + q)

    def _fit_two_component_gmm(self, x_fit: np.ndarray):
        n, d = x_fit.shape
        if n < max(30, d * 8):
            return None

        center = np.nanmedian(x_fit, axis=0)
        mad = np.nanmedian(np.abs(x_fit - center), axis=0)
        mad = np.where(np.isfinite(mad) & (mad > 1e-6), mad, 1.0)
        z = (x_fit - center[None, :]) / mad[None, :]
        d2 = np.sum(z * z, axis=1)
        q40 = float(np.nanquantile(d2, 0.40))
        m0 = d2 <= q40
        if m0.sum() < max(12, d * 3) or m0.sum() > (n - max(12, d * 3)):
            order = np.argsort(d2)
            m0 = np.zeros(n, dtype=bool)
            m0[order[: max(n // 2, 1)]] = True
        m1 = ~m0

        def _cov(arr: np.ndarray) -> np.ndarray:
            if arr.shape[0] < (d + 1):
                return np.eye(d, dtype=float)
            c = np.cov(arr, rowvar=False)
            if np.ndim(c) == 0:
                c = np.eye(d, dtype=float) * float(c)
            return np.asarray(c, float) + np.eye(d, dtype=float) * 1e-4

        pi = np.array([max(m0.mean(), 1e-3), max(m1.mean(), 1e-3)], dtype=float)
        pi /= pi.sum()
        mu = np.vstack([
            np.nanmean(x_fit[m0], axis=0),
            np.nanmean(x_fit[m1], axis=0),
        ])
        cov = np.stack([_cov(x_fit[m0]), _cov(x_fit[m1])], axis=0)
        last_ll = -np.inf

        for _ in range(80):
            lp0 = np.log(max(pi[0], 1e-9)) + self._logpdf_gauss(x_fit, mu[0], cov[0])
            lp1 = np.log(max(pi[1], 1e-9)) + self._logpdf_gauss(x_fit, mu[1], cov[1])
            m = np.maximum(lp0, lp1)
            e0 = np.exp(lp0 - m)
            e1 = np.exp(lp1 - m)
            den = e0 + e1 + 1e-12
            r0 = e0 / den
            r1 = e1 / den
            nk = np.array([r0.sum(), r1.sum()], dtype=float)
            if np.any(nk < (d + 2)):
                break
            pi = nk / float(n)
            mu[0] = (r0[:, None] * x_fit).sum(axis=0) / nk[0]
            mu[1] = (r1[:, None] * x_fit).sum(axis=0) / nk[1]
            for k, rk in enumerate((r0, r1)):
                diff = x_fit - mu[k][None, :]
                cov[k] = (diff.T * rk).dot(diff) / max(nk[k], 1.0)
                cov[k] += np.eye(d, dtype=float) * 1e-4
            ll = float(np.sum(m + np.log(den)))
            if np.isfinite(last_ll):
                if abs(ll - last_ll) < 1e-4 * max(1.0, abs(last_ll)):
                    break
            last_ll = ll

        det0 = abs(float(np.linalg.det(cov[0])))
        det1 = abs(float(np.linalg.det(cov[1])))
        cluster_idx = 0 if det0 <= det1 else 1
        return {
            "pi": pi,
            "mu": mu,
            "cov": cov,
            "cluster_idx": int(cluster_idx),
        }

    def _compute_membership_from_master(self) -> bool:
        master_candidates = [
            step6_dir(self.params.P.result_dir) / "master_catalog.tsv",
            self.params.P.result_dir / "master_catalog.tsv",
        ]
        master_path = next((p for p in master_candidates if p.exists()), None)
        if master_path is None:
            return False

        try:
            df = pd.read_csv(master_path, sep="\t")
        except Exception:
            return False
        if df.empty:
            return False
        req = ("pmra", "pmdec", "parallax")
        if not all(c in df.columns for c in req):
            return False

        pmra = pd.to_numeric(df["pmra"], errors="coerce").to_numpy(float)
        pmdec = pd.to_numeric(df["pmdec"], errors="coerce").to_numpy(float)
        plx = pd.to_numeric(df["parallax"], errors="coerce").to_numpy(float)
        finite = np.isfinite(pmra) & np.isfinite(pmdec) & np.isfinite(plx)
        if int(finite.sum()) < 30:
            return False

        fit_mask = finite.copy()
        if "ruwe" in df.columns:
            ruwe = pd.to_numeric(df["ruwe"], errors="coerce").to_numpy(float)
            fit_mask &= (~np.isfinite(ruwe)) | (ruwe <= 2.0)
        if "visibility_periods_used" in df.columns:
            vpu = pd.to_numeric(df["visibility_periods_used"], errors="coerce").to_numpy(float)
            fit_mask &= (~np.isfinite(vpu)) | (vpu >= 8.0)
        if int(fit_mask.sum()) < 25:
            fit_mask = finite.copy()

        x_fit = np.column_stack([pmra[fit_mask], pmdec[fit_mask], plx[fit_mask]])
        model = self._fit_two_component_gmm(x_fit)
        if model is None:
            return False

        x_all = np.column_stack([pmra[finite], pmdec[finite], plx[finite]])
        pi = np.asarray(model["pi"], float)
        mu = np.asarray(model["mu"], float)
        cov = np.asarray(model["cov"], float)
        k_cluster = int(model["cluster_idx"])

        lp0 = np.log(max(pi[0], 1e-9)) + self._logpdf_gauss(x_all, mu[0], cov[0])
        lp1 = np.log(max(pi[1], 1e-9)) + self._logpdf_gauss(x_all, mu[1], cov[1])
        m = np.maximum(lp0, lp1)
        e0 = np.exp(lp0 - m)
        e1 = np.exp(lp1 - m)
        den = e0 + e1 + 1e-12
        r0 = e0 / den
        r1 = e1 / den
        p_cluster = r0 if k_cluster == 0 else r1

        pmem = np.full(len(df), np.nan, dtype=float)
        pmem[finite] = np.clip(p_cluster, 0.0, 1.0)

        src = parse_int64_series(df["source_id"]).tolist() if "source_id" in df.columns else [pd.NA] * len(df)
        gaia = parse_int64_series(df["gaia_source_id"]).tolist() if "gaia_source_id" in df.columns else [pd.NA] * len(df)
        ids = pd.to_numeric(df["ID"], errors="coerce").tolist() if "ID" in df.columns else [np.nan] * len(df)

        n_src = 0
        n_gaia = 0
        n_id = 0
        for s, g, i, p in zip(src, gaia, ids, pmem.tolist()):
            if not np.isfinite(p):
                continue
            pp = float(np.clip(p, 0.0, 1.0))
            if pd.notna(s):
                self.membership_by_source[int(s)] = pp
                n_src += 1
            if pd.notna(g):
                self.membership_by_gaia[int(g)] = pp
                n_gaia += 1
            if np.isfinite(i):
                self.membership_by_id[int(i)] = pp
                n_id += 1

        self.log(
            "Membership map computed from master astrometry "
            f"(source={n_src}, gaia={n_gaia}, id={n_id}, fit={int(fit_mask.sum())})"
        )
        return (n_src + n_gaia + n_id) > 0

    def _load_membership_map(self):
        self.membership_by_source = {}
        self.membership_by_gaia = {}
        self.membership_by_id = {}
        self._membership_loaded = True

        result_dir = self.params.P.result_dir
        candidates = [
            step5_dir(result_dir) / "gaia_derived.csv",
            result_dir / "gaia_derived.csv",
            result_dir / "cmd_with_gaia_membership.csv",
            result_dir / "step11" / "cmd_with_gaia_membership.csv",
            result_dir / "cmd_with_membership.csv",
            result_dir / "step12" / "cmd_with_membership.csv",
            result_dir / "median_by_ID_filter_wide_cmd.csv",
            result_dir / "step11" / "median_by_ID_filter_wide_cmd.csv",
            step6_dir(result_dir) / "master_catalog.tsv",
            result_dir / "master_catalog.tsv",
        ]

        for path in candidates:
            if not path.exists():
                continue
            try:
                if path.suffix.lower() == ".tsv":
                    df = pd.read_csv(path, sep="\t")
                else:
                    df = pd.read_csv(path)
            except Exception:
                continue
            if df.empty:
                continue

            p_col = self._pick_membership_col(df.columns)
            if p_col is None:
                continue
            prob = pd.to_numeric(df[p_col], errors="coerce").to_numpy(float)

            n_src = 0
            n_gaia = 0
            n_id = 0
            if "source_id" in df.columns:
                src = parse_int64_series(df["source_id"])
                for sid, p in zip(src.tolist(), prob.tolist()):
                    if pd.isna(sid) or (not np.isfinite(p)):
                        continue
                    self.membership_by_source[int(sid)] = float(np.clip(p, 0.0, 1.0))
                    n_src += 1

            if "gaia_source_id" in df.columns:
                gs = parse_int64_series(df["gaia_source_id"])
                for sid, p in zip(gs.tolist(), prob.tolist()):
                    if pd.isna(sid) or (not np.isfinite(p)):
                        continue
                    self.membership_by_gaia[int(sid)] = float(np.clip(p, 0.0, 1.0))
                    n_gaia += 1

            if "ID" in df.columns:
                ids = pd.to_numeric(df["ID"], errors="coerce")
                for id_v, p in zip(ids.tolist(), prob.tolist()):
                    if (not np.isfinite(id_v)) or (not np.isfinite(p)):
                        continue
                    self.membership_by_id[int(id_v)] = float(np.clip(p, 0.0, 1.0))
                    n_id += 1

            if (n_src + n_gaia + n_id) > 0:
                self.log(
                    f"Membership map loaded: source={n_src}, gaia={n_gaia}, id={n_id} "
                    f"from {path.name} ({p_col})"
                )
                return

        # Fallback: compute membership from RefBuild astrometric columns (Step8; legacy path compatible).
        if self._compute_membership_from_master():
            return

        if not self._membership_log_once:
            self.log("Membership map not found (gaia_pmem column/file missing).")
            self._membership_log_once = True

    def _ensure_membership_map(self, force: bool = False):
        if (not force) and (not bool(getattr(self.params.P, "step8_membership_overlay_enable", False))):
            return
        if not self._membership_loaded:
            self._load_membership_map()

    def _membership_prob_for_sid(self, sid: int) -> float:
        s = int(sid)
        p = self.membership_by_source.get(s, None)
        if p is not None:
            return float(p)
        if s > 0:
            p = self.membership_by_gaia.get(s, None)
            if p is not None:
                return float(p)
        id_v = self.internal_id_map.get(s, self._global_id_map.get(s, None))
        if id_v is not None:
            p = self.membership_by_id.get(int(id_v), None)
            if p is not None:
                return float(p)
        return np.nan

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
        """Update master table using stable ID (not frame-dependent numbering)."""
        self._ensure_membership_map(force=True)
        rows = []
        for sid in sorted(self.master_ids):
            sid_i = int(sid)
            fixed_id = self._ensure_stable_id(sid_i)
            g_mag = self._get_gmag_for_source(sid_i)
            pmem = self._membership_prob_for_sid(sid_i)
            rows.append((int(fixed_id), sid_i, g_mag, pmem))

        rows.sort(key=lambda x: int(x[0]))

        self.master_table.blockSignals(True)
        self.master_table.setUpdatesEnabled(False)
        try:
            self.master_table.setRowCount(len(rows))
            for i, (fixed_id, sid, g_mag, pmem) in enumerate(rows):
                self.master_table.setItem(i, 0, QTableWidgetItem(str(fixed_id)))
                self.master_table.setItem(i, 1, QTableWidgetItem(str(sid)))
                if np.isfinite(g_mag):
                    g_str = f"{g_mag:.3f}"
                elif int(sid) < 0:
                    g_str = "local"
                else:
                    g_str = "-"
                self.master_table.setItem(i, 2, QTableWidgetItem(g_str))
                p_str = f"{pmem:.3f}" if np.isfinite(pmem) else "-"
                self.master_table.setItem(i, 3, QTableWidgetItem(p_str))
        finally:
            self.master_table.setUpdatesEnabled(True)
            self.master_table.blockSignals(False)

        n_total = len(rows)
        n_gmag = sum(1 for _, _, g, _ in rows if np.isfinite(g))
        n_pmem = sum(1 for _, _, _, p in rows if np.isfinite(p))
        if hasattr(self, "log_text"):
            self.log(f"Master IDs: {n_total} | Gmag: {n_gmag} | Pmem: {n_pmem} | Stable IDs")

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
        # Fast switch: avoid full redraw if shape is unchanged.
        self.load_and_display(quick_switch=True)

    def _get_fits_path(self, filename):
        if self.use_cropped:
            cropped_dir = step2_cropped_dir(self.params.P.result_dir)
            if not cropped_dir.exists():
                cropped_dir = self.params.P.result_dir / "cropped"
            return cropped_dir / filename
        return self.params.P.data_dir / filename

    def _load_fits_cached(self, filename):
        """Return (image_data, header) from cache or disk. LRU eviction."""
        with self._prefetch_lock:
            if filename in self._fits_cache:
                # Move to end (most recently used)
                try:
                    self._fits_cache_order.remove(filename)
                except ValueError:
                    pass
                self._fits_cache_order.append(filename)
                return self._fits_cache[filename]
        file_path = self._get_fits_path(filename)
        with fits.open(file_path, memmap=False) as hdul:
            # Use float32 to reduce memory bandwidth on frame switching.
            data_raw = hdul[0].data
            if data_raw is None:
                raise ValueError(f"FITS image data is empty: {file_path.name}")
            data = np.asarray(data_raw, dtype=np.float32)
            header = hdul[0].header.copy()
        with self._prefetch_lock:
            # In case prefetch completed first, return existing cache entry.
            if filename in self._fits_cache:
                try:
                    self._fits_cache_order.remove(filename)
                except ValueError:
                    pass
                self._fits_cache_order.append(filename)
                return self._fits_cache[filename]
            # Evict oldest if over limit
            while len(self._fits_cache_order) >= self._FITS_CACHE_SIZE:
                oldest = self._fits_cache_order.pop(0)
                self._fits_cache.pop(oldest, None)
            self._fits_cache[filename] = (data, header)
            self._fits_cache_order.append(filename)
            return data, header

    def _prefetch_fits_worker(self, filename: str):
        try:
            file_path = self._get_fits_path(filename)
            if not file_path.exists():
                return
            with fits.open(file_path, memmap=False) as hdul:
                data_raw = hdul[0].data
                if data_raw is None:
                    return
                data = np.asarray(data_raw, dtype=np.float32)
                header = hdul[0].header.copy()
            with self._prefetch_lock:
                if filename not in self._fits_cache:
                    while len(self._fits_cache_order) >= self._FITS_CACHE_SIZE:
                        oldest = self._fits_cache_order.pop(0)
                        self._fits_cache.pop(oldest, None)
                    self._fits_cache[filename] = (data, header)
                    self._fits_cache_order.append(filename)
        except Exception:
            pass
        finally:
            with self._prefetch_lock:
                self._prefetch_pending.discard(filename)

    def _schedule_prefetch_neighbors(self):
        if not self.file_list:
            return
        idx = self.file_combo.currentIndex()
        if idx < 0:
            return
        candidates = []
        if idx + 1 < len(self.file_list):
            candidates.append(self.file_list[idx + 1])
        if idx - 1 >= 0:
            candidates.append(self.file_list[idx - 1])
        for fname in candidates:
            with self._prefetch_lock:
                if fname in self._fits_cache or fname in self._prefetch_pending:
                    continue
                self._prefetch_pending.add(fname)
            try:
                self._prefetch_executor.submit(self._prefetch_fits_worker, fname)
            except Exception:
                with self._prefetch_lock:
                    self._prefetch_pending.discard(fname)

    def load_and_display(self, quick_switch=False):
        filename = self.file_combo.currentText()
        if not filename:
            return
        try:
            new_data, new_header = self._load_fits_cached(filename)
            # Detect size change — forces full redraw if image shape differs
            shape_changed = (self.image_data is None or new_data.shape != self.image_data.shape)
            self.image_data = new_data
            self.header = new_header
            self.current_filename = filename
            self._normalized_cache = None  # Always invalidate (new image data)
            if not quick_switch or shape_changed:
                self.xlim_original = None
                self.ylim_original = None
                self._imshow_obj = None
            self.load_idmatch_for_file(filename)
            full_redraw = self._imshow_obj is None
            self.display_image(full_redraw=full_redraw)
            if not full_redraw:
                # Fast path skipped display_image's full rebuild; refresh ROI for new WCS
                self._redraw_roi_patch()
            self.update_overlay()
            if self._auto_master_dirty:
                self.save_master_ids(log_action="auto_add")
                self._auto_master_dirty = False
            self._schedule_prefetch_neighbors()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load: {str(e)}")

    def closeEvent(self, event):
        try:
            self._prefetch_executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        super().closeEvent(event)

    def load_idmatch_for_file(self, filename):
        # Check cache first
        if filename in self._idmatch_cache:
            self.idmatch_df = self._idmatch_cache[filename]
            arr = self._idmatch_arr_cache.get(filename, None)
            if arr is not None:
                self._auto_add_detections_to_master(self.idmatch_df, arr[2])
            else:
                self._auto_add_detections_to_master(self.idmatch_df)
            return
        idmatch_path = self.params.P.cache_dir / "idmatch" / f"idmatch_{filename}.csv"
        if idmatch_path.exists():
            try:
                df = read_csv_int64_source_id(idmatch_path)
                if {"x", "y", "source_id"} <= set(df.columns):
                    clean = df[["x", "y", "source_id"]].copy()
                    clean["x"] = pd.to_numeric(clean["x"], errors="coerce")
                    clean["y"] = pd.to_numeric(clean["y"], errors="coerce")
                    valid = (
                        np.isfinite(clean["x"].to_numpy(float))
                        & np.isfinite(clean["y"].to_numpy(float))
                    )
                    dropped = int((~valid).sum())
                    if dropped > 0:
                        self.log(f"[{filename}] idmatch: dropped {dropped} invalid rows (x/y)")
                    clean = clean.loc[valid].copy()
                    unmatched = int(clean["source_id"].apply(
                        lambda s: pd.isna(s)
                    ).sum())
                    if unmatched > 0:
                        self.log(f"[{filename}] idmatch: unmatched detections kept={unmatched}")
                    clean["source_id"] = parse_int64_series(clean["source_id"]).fillna(0).astype("int64")
                    result = clean.reset_index(drop=True)
                    x_arr = result["x"].to_numpy(float, copy=False)
                    y_arr = result["y"].to_numpy(float, copy=False)
                    sid_arr = result["source_id"].to_numpy(np.int64, copy=False)
                    self._idmatch_cache[filename] = result
                    self._idmatch_arr_cache[filename] = (x_arr, y_arr, sid_arr)
                    self.idmatch_df = result
                    self._auto_add_detections_to_master(self.idmatch_df, sid_arr)
                    return
            except Exception:
                pass
        empty = pd.DataFrame(columns=["x", "y", "source_id"])
        self._idmatch_cache[filename] = empty
        self._idmatch_arr_cache[filename] = (
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=np.int64),
        )
        self.idmatch_df = empty

    def _auto_add_detections_to_master(self, df: pd.DataFrame, sid_arr: np.ndarray | None = None):
        if sid_arr is None:
            try:
                sid_arr = parse_int64_series(df["source_id"]).dropna().astype("int64").to_numpy(np.int64, copy=False)
            except Exception:
                return
        if sid_arr.size == 0:
            return
        sids = set(np.unique(sid_arr[sid_arr != 0]).tolist())
        new_ids = sids - self.master_ids
        if not new_ids:
            return
        self.master_ids |= new_ids
        self._auto_master_dirty = True

    def display_image(self, full_redraw=False):
        if self.image_data is None:
            return

        stretched = self._get_stretched_display_cached()
        if stretched is None:
            return

        filt = ""
        if self.header is not None:
            filt = str(self.header.get("FILTER", self.header.get("filter", ""))).strip()
        stretch_name = self.scale_combo.currentText()
        title = f"{self.current_filename}"
        if filt:
            title += f" [{filt}]"
        title += f" | {stretch_name}"

        if self._imshow_obj is not None and not full_redraw:
            self._imshow_obj.set_data(stretched)
            self.ax.set_title(title)
            self.canvas.draw_idle()
            return

        xlim_current = self.ax.get_xlim() if self.xlim_original else None
        ylim_current = self.ax.get_ylim() if self.ylim_original else None

        self.ax.clear()
        # ax.clear() destroys all artists — reset scatter refs
        self._scat_unmatched = None
        self._scat_removed = None
        self._scat_local = None
        self._scat_gaia = None
        self._scat_member = None
        self._scat_selected = None

        self._imshow_obj = self.ax.imshow(
            stretched, cmap='gray', origin='lower',
            vmin=0, vmax=1, interpolation='nearest'
        )
        # Pre-create persistent scatter artists (updated via set_offsets, no remove/recreate)
        c = self._overlay_colors
        self._scat_unmatched = self.ax.scatter([], [], s=20, facecolors='none',
                                               edgecolors=c["unmatched"], linewidths=0.8, alpha=0.7)
        self._scat_removed   = self.ax.scatter([], [], s=22, facecolors='none',
                                               edgecolors=c["removed"], linewidths=0.9, alpha=0.75)
        self._scat_local     = self.ax.scatter([], [], s=26, facecolors='none',
                                               edgecolors=c["local"], linewidths=1.0, alpha=0.8)
        self._scat_gaia      = self.ax.scatter([], [], s=28, facecolors='none',
                                               edgecolors=c["gaia"], linewidths=1.1, alpha=0.85)
        self._scat_psf_iter2 = self.ax.scatter([], [], s=32, facecolors='none',
                                               edgecolors=c["psf_iter2"], linewidths=1.2, alpha=0.85)
        self._scat_member    = self.ax.scatter([], [], s=30, facecolors='none',
                                               edgecolors=c["member"], linewidths=1.2, alpha=0.9)
        self._scat_selected  = self.ax.scatter([], [], s=60, facecolors='none',
                                               edgecolors=c["selected"], linewidths=1.5, alpha=0.9)

        self.ax.set_title(title)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")

        # Re-add ROI patch (ax.clear() removes it)
        self._roi_patch = None
        self._roi_preview_patch = None
        self._redraw_roi_patch()

        if self.xlim_original is None:
            self.xlim_original = self.ax.get_xlim()
            self.ylim_original = self.ax.get_ylim()
        elif xlim_current is not None:
            self.ax.set_xlim(xlim_current)
            self.ax.set_ylim(ylim_current)

        self.canvas.draw_idle()

    @staticmethod
    def _safe_offsets(x_arr, y_arr):
        """Return Nx2 offsets array (empty-safe) for set_offsets()."""
        if len(x_arr) == 0:
            return np.zeros((0, 2), dtype=float)
        return np.column_stack([x_arr, y_arr])

    def update_overlay(self):
        # Scatter artists may not exist yet (before first full_redraw)
        if self._scat_gaia is None:
            self.canvas.draw_idle()
            return

        empty = np.zeros((0, 2), dtype=float)

        if self.idmatch_df is None or self.idmatch_df.empty:
            for s in (self._scat_unmatched, self._scat_removed,
                      self._scat_local, self._scat_psf_iter2, self._scat_gaia, self._scat_member, self._scat_selected):
                s.set_offsets(empty)
            self.canvas.draw_idle()
            return

        arr = self._idmatch_arr_cache.get(self.current_filename or "", None)
        if arr is None:
            x = pd.to_numeric(self.idmatch_df["x"], errors="coerce").to_numpy(float)
            y = pd.to_numeric(self.idmatch_df["y"], errors="coerce").to_numpy(float)
            sid = np.asarray(pd.to_numeric(self.idmatch_df["source_id"], errors="coerce"), dtype=float)
            valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(sid)
            if not np.any(valid):
                for s in (self._scat_unmatched, self._scat_removed,
                          self._scat_local, self._scat_gaia, self._scat_member, self._scat_selected):
                    s.set_offsets(empty)
                self.canvas.draw_idle()
                return
            x = x[valid]
            y = y[valid]
            sids = sid[valid].astype(np.int64, copy=False)
            self._idmatch_arr_cache[self.current_filename or ""] = (x, y, sids)
        else:
            x, y, sids = arr

        if sids.size == 0:
            for s in (self._scat_unmatched, self._scat_removed,
                      self._scat_local, self._scat_psf_iter2, self._scat_gaia, self._scat_member, self._scat_selected):
                s.set_offsets(empty)
            self.canvas.draw_idle()
            return

        if self.master_ids:
            master_vals = np.fromiter((int(v) for v in self.master_ids), dtype=np.int64, count=len(self.master_ids))
            in_master = np.isin(sids, master_vals)
        else:
            in_master = np.zeros_like(sids, dtype=bool)
        is_unmatched  = sids == 0
        is_matched    = ~is_unmatched
        is_gaia       = sids > 0
        is_local      = sids < 0
        is_removed    = is_matched & ~in_master
        is_gaia_master = is_matched & in_master & is_gaia
        is_local_master = is_matched & in_master & is_local

        is_member = np.zeros_like(is_gaia_master, dtype=bool)
        if bool(getattr(self.params.P, "step8_membership_overlay_enable", False)):
            self._ensure_membership_map()
            thr = float(getattr(self.params.P, "step8_membership_threshold", 0.5))
            idx_gaia = np.where(is_gaia_master)[0]
            if idx_gaia.size > 0:
                sid_gaia = sids[idx_gaia]
                pmem = np.fromiter(
                    (self._membership_prob_for_sid(int(s)) for s in sid_gaia),
                    dtype=float,
                    count=sid_gaia.size,
                )
                is_member[idx_gaia] = np.isfinite(pmem) & (pmem >= thr)
        is_gaia_nonmember = is_gaia_master & (~is_member)

        # Split local sources: PSF iter2 (det_uid<0 from PSF residuals) vs other local
        if self.psf_iter2_ids:
            _psf2_arr = np.fromiter(self.psf_iter2_ids, dtype=np.int64, count=len(self.psf_iter2_ids))
            is_psf_iter2   = is_local_master & np.isin(sids, _psf2_arr)
            is_local_other = is_local_master & ~is_psf_iter2
        else:
            is_psf_iter2   = np.zeros(len(sids), dtype=bool)
            is_local_other = is_local_master

        vis = self._overlay_visible

        def _off(mask, key):
            return self._safe_offsets(x[mask], y[mask]) if vis.get(key, True) else empty

        self._scat_unmatched.set_offsets(_off(is_unmatched,    "unmatched"))
        self._scat_removed.set_offsets(  _off(is_removed,      "removed"))
        self._scat_local.set_offsets(    _off(is_local_other,  "local"))
        self._scat_psf_iter2.set_offsets(_off(is_psf_iter2,    "psf_iter2"))
        self._scat_gaia.set_offsets(     _off(is_gaia_nonmember, "gaia"))
        self._scat_member.set_offsets(   _off(is_member,       "member"))

        if self.selected_source_id is not None:
            sel_mask = sids == self.selected_source_id
            self._scat_selected.set_offsets(
                self._safe_offsets(x[sel_mask], y[sel_mask]) if vis.get("selected", True) else empty
            )
        else:
            self._scat_selected.set_offsets(empty)

        self.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        if event.button != 1:
            return
        if self._roi_mode:
            return  # handled by on_button_press/release
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
                    sid = int(self.idmatch_df.iloc[i]["source_id"])
                    if sid == 0:
                        self.selected_source_id = None
                        frame = self.current_filename or "?"
                        px_x = float(self.idmatch_df.iloc[i]["x"])
                        px_y = float(self.idmatch_df.iloc[i]["y"])
                        self.selected_label.setText(
                            f"Selected: unmatched detection at ({px_x:.1f}, {px_y:.1f})"
                        )
                        self.log(f"[{frame}] Unmatched detection selected | px=({px_x:.1f}, {px_y:.1f})")
                        self.update_overlay()
                        return
                    self.selected_source_id = sid
                    in_master = "✓ IN MASTER" if self.selected_source_id in self.master_ids else "✗ not in master"
                    internal_id = self.internal_id_map.get(
                        self.selected_source_id,
                        self._global_id_map.get(self.selected_source_id, "-")
                    )
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
        self._ensure_stable_id(int(self.selected_source_id))
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
        sid_vals = parse_int64_series(df.loc[in_box, "source_id"])
        sids = set(sid_vals[sid_vals.notna()].astype("int64").tolist())
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
        sid2id_path = output_dir / "sourceid_to_ID.csv"
        backup_path = output_dir / "master_star_ids.orig.csv"
        if master_path.exists() and (not backup_path.exists()):
            try:
                backup_path.write_text(master_path.read_text(encoding="utf-8"), encoding="utf-8")
            except Exception:
                pass

        # Build stable-ID rows and persist mapping.
        rows = []
        for sid in sorted(int(s) for s in self.master_ids):
            fixed_id = self._ensure_stable_id(sid)
            g_mag = self._get_gmag_for_source(sid)
            rows.append({
                "ID": int(fixed_id),
                "source_id": sid,
                "g_mag": g_mag if np.isfinite(g_mag) else None
            })
        df = pd.DataFrame(rows)
        if len(df):
            df = df.sort_values(["ID", "source_id"]).reset_index(drop=True)
        df.to_csv(master_path, index=False)
        if len(df):
            df[["source_id", "ID"]].to_csv(sid2id_path, index=False)

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
        sid = int(source_id)
        val = self._gaia_gmag_map.get(sid, np.nan)
        if np.isfinite(val):
            return float(val)
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
                internal_id = self._ensure_stable_id(sid)
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
            self._ensure_stable_id(sid)
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

        self.param_mem_overlay = QCheckBox("Enable membership color overlay")
        self.param_mem_overlay.setChecked(bool(getattr(self.params.P, "step8_membership_overlay_enable", False)))
        form.addRow("Membership Overlay:", self.param_mem_overlay)

        self.param_mem_thr = QDoubleSpinBox()
        self.param_mem_thr.setRange(0.0, 1.0)
        self.param_mem_thr.setDecimals(2)
        self.param_mem_thr.setSingleStep(0.05)
        self.param_mem_thr.setValue(float(getattr(self.params.P, "step8_membership_threshold", 0.5)))
        form.addRow("Membership P threshold:", self.param_mem_thr)

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
        self.params.P.step8_membership_overlay_enable = self.param_mem_overlay.isChecked()
        self.params.P.step8_membership_threshold = self.param_mem_thr.value()
        self._membership_loaded = False
        self._ensure_membership_map(force=True)
        self.update_overlay()
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

    # ── Color legend floating window ─────────────────────────────────────────

    def _build_color_window(self) -> QWidget:
        """Build the overlay color/visibility window (floating, Qt.Tool)."""
        win = QWidget(None, Qt.Tool | Qt.WindowTitleHint | Qt.WindowCloseButtonHint)
        win.setWindowTitle("Overlay Colors")
        win.setFixedWidth(200)
        win.closeEvent = lambda e: (self.btn_colors.setChecked(False), e.accept())

        layout = QVBoxLayout(win)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        hint = QLabel("Left-click: toggle on/off\nRight-click: change color")
        hint.setStyleSheet("color: #888; font-size: 8pt;")
        layout.addWidget(hint)

        entries = [
            ("gaia",      "Gaia (master)"),
            ("member",    "Membership"),
            ("psf_iter2", "PSF iter2 (new)"),
            ("local",     "Local (other)"),
            ("removed",   "Removed"),
            ("unmatched", "Unmatched"),
            ("selected",  "Selected"),
        ]
        for key, label_text in entries:
            row_w = QWidget()
            row = QHBoxLayout(row_w)
            row.setContentsMargins(0, 2, 0, 2)
            row.setSpacing(8)

            btn = QPushButton()
            btn.setFixedSize(24, 24)
            btn.setCheckable(True)
            btn.setChecked(True)
            btn.setToolTip("Left-click: toggle  |  Right-click: change color")
            btn.setContextMenuPolicy(Qt.CustomContextMenu)
            btn.customContextMenuRequested.connect(lambda _, k=key: self._on_pick_color(k))
            btn.toggled.connect(lambda on, k=key: self._toggle_layer_visible(k, on))
            self._color_btns[key] = btn
            self._refresh_color_btn(key)

            lbl = QLabel(label_text)
            row.addWidget(btn)
            row.addWidget(lbl)
            row.addStretch()
            layout.addWidget(row_w)

        layout.addStretch()
        return win

    def _refresh_color_btn(self, key: str):
        """Update a color button's appearance based on current color + visibility."""
        btn = self._color_btns.get(key)
        if btn is None:
            return
        visible = self._overlay_visible.get(key, True)
        color = self._overlay_colors.get(key, "#ffffff")
        if visible:
            btn.setStyleSheet(
                f"QPushButton {{ background-color: {color}; border: 2px solid #aaa; border-radius: 3px; }}"
                f"QPushButton:hover {{ border: 2px solid #fff; }}"
            )
        else:
            btn.setStyleSheet(
                "QPushButton { background-color: #333; border: 2px solid #555; border-radius: 3px; "
                "color: #666; }"
            )

    def _toggle_layer_visible(self, key: str, on: bool):
        """Toggle a specific overlay layer on/off."""
        self._overlay_visible[key] = on
        self._refresh_color_btn(key)
        self.update_overlay()

    def _on_pick_color(self, key: str):
        cur = QColor(self._overlay_colors.get(key, "#ffffff"))
        col = QColorDialog.getColor(cur, self, f"Pick color — {key}")
        if not col.isValid():
            return
        hex_color = col.name()
        self._overlay_colors[key] = hex_color
        self._refresh_color_btn(key)
        # Update live scatter artist edge color
        scat_map = {
            "gaia":      "_scat_gaia",
            "member":    "_scat_member",
            "psf_iter2": "_scat_psf_iter2",
            "local":     "_scat_local",
            "removed":   "_scat_removed",
            "unmatched": "_scat_unmatched",
            "selected":  "_scat_selected",
        }
        scat = getattr(self, scat_map.get(key, ""), None)
        if scat is not None:
            scat.set_edgecolors(hex_color)
            if self.canvas is not None:
                self.canvas.draw_idle()

    def _toggle_color_panel(self, checked: bool):
        if self._color_window is None:
            return
        if checked:
            # Position to the left of this widget
            top_left = self.mapToGlobal(QPoint(0, 0))
            win_w = self._color_window.sizeHint().width() or 200
            x = max(0, top_left.x() - win_w - 8)
            y = top_left.y()
            self._color_window.move(x, y)
            self._color_window.show()
            self._color_window.raise_()
        else:
            self._color_window.hide()

    @staticmethod
    def _apply_color_btn_style(btn: QPushButton, color: str):
        """Legacy helper kept for compat — use _refresh_color_btn instead."""
        btn.setStyleSheet(
            f"QPushButton {{ background-color: {color}; border: 2px solid #aaa; border-radius: 3px; }}"
            f"QPushButton:hover {{ border: 2px solid #fff; }}"
        )

    # ────────────────────────────────────────────────────────────────────────

    def save_state(self):
        state_data = {
            "search_radius_px": getattr(self.params.P, "search_radius_px", 7.0),
            "bulk_drop_box_px": getattr(self.params.P, "bulk_drop_box_px", 200),
            "gaia_add_max_sep_arcsec": getattr(self.params.P, "gaia_add_max_sep_arcsec", 2.0),
            "step8_membership_overlay_enable": getattr(self.params.P, "step8_membership_overlay_enable", False),
            "step8_membership_threshold": getattr(self.params.P, "step8_membership_threshold", 0.5),
        }
        self.project_state.store_step_data("master_id_editor", state_data)

    def restore_state(self):
        state_data = self.project_state.get_step_data("master_id_editor")
        if state_data:
            for key, val in state_data.items():
                if hasattr(self.params.P, key):
                    setattr(self.params.P, key, val)

    # ------------------------------------------------------------------
    # CMD ROI helpers
    # ------------------------------------------------------------------
    def _roi_path(self) -> Path:
        return step8_dir(self.params.P.result_dir) / "cmd_roi.json"

    def _load_roi(self):
        p = self._roi_path()
        try:
            if p.exists():
                self._roi_circle = json.loads(p.read_text())
            else:
                self._roi_circle = None
        except Exception:
            self._roi_circle = None
        self._update_roi_info_label()

    def _save_roi(self):
        p = self._roi_path()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            if self._roi_circle:
                p.write_text(json.dumps(self._roi_circle))
            elif p.exists():
                p.unlink()
        except Exception:
            pass

    def _pixels_to_sky_roi(self, cx_px: float, cy_px: float, r_px: float) -> dict | None:
        """Convert pixel-space ROI to sky ROI {ra_deg, dec_deg, radius_arcsec} using current frame WCS."""
        if self.header is None:
            return None
        try:
            wcs = WCS(self.header, naxis=2)
            ra, dec = wcs.all_pix2world([[cx_px, cy_px]], 0)[0]
            # pixel scale: sqrt(|det(CD matrix)|) in deg/px → arcsec/px
            ps_deg = float(np.sqrt(abs(np.linalg.det(wcs.pixel_scale_matrix))))
            ps_arcsec = ps_deg * 3600.0
            if ps_arcsec <= 0:
                ps_arcsec = float(getattr(self.params.P, "pixel_scale_arcsec", 1.0))
            return {
                "ra_deg": float(ra),
                "dec_deg": float(dec),
                "radius_arcsec": r_px * ps_arcsec,
            }
        except Exception:
            return None

    def _sky_roi_to_pixels(self) -> tuple[float, float, float] | None:
        """Convert stored sky ROI to pixel coords for the current frame. Returns (cx, cy, r_px) or None."""
        if self._roi_circle is None or self.header is None:
            return None
        try:
            wcs = WCS(self.header, naxis=2)
            ra = self._roi_circle["ra_deg"]
            dec = self._roi_circle["dec_deg"]
            r_arcsec = self._roi_circle["radius_arcsec"]
            cx_px, cy_px = wcs.all_world2pix([[ra, dec]], 0)[0]
            ps_deg = float(np.sqrt(abs(np.linalg.det(wcs.pixel_scale_matrix))))
            ps_arcsec = ps_deg * 3600.0
            if ps_arcsec <= 0:
                ps_arcsec = float(getattr(self.params.P, "pixel_scale_arcsec", 1.0))
            r_px = r_arcsec / ps_arcsec
            return float(cx_px), float(cy_px), float(r_px)
        except Exception:
            return None

    def _update_roi_info_label(self):
        if self._roi_circle:
            ra = self._roi_circle["ra_deg"]
            dec = self._roi_circle["dec_deg"]
            r = self._roi_circle["radius_arcsec"]
            self.roi_info_label.setText(f"ROI: RA={ra:.4f} Dec={dec:.4f}  r={r:.0f}\"")
            self.roi_info_label.setStyleSheet("QLabel { color: #00E5FF; font-size: 9pt; }")
        else:
            self.roi_info_label.setText("No ROI set")
            self.roi_info_label.setStyleSheet("QLabel { color: #90A4AE; font-size: 9pt; }")

    def _redraw_roi_patch(self):
        """Draw (or remove) the ROI circle patch, projected to the current frame's pixel coords."""
        if self._roi_patch is not None:
            try:
                self._roi_patch.remove()
            except Exception:
                pass
            self._roi_patch = None
        if self._roi_circle is not None:
            pix = self._sky_roi_to_pixels()
            if pix is not None:
                cx_px, cy_px, r_px = pix
                self._roi_patch = MplCircle(
                    (cx_px, cy_px), r_px, fill=False,
                    edgecolor='#00E5FF', linestyle='--', linewidth=1.5, alpha=0.85
                )
                self.ax.add_patch(self._roi_patch)
        self.canvas.draw_idle()

    def _on_set_roi_toggled(self, checked: bool):
        self._roi_mode = checked
        if checked:
            self.btn_set_roi.setText("Cancel (click+drag to draw)")
            self._roi_drag_start = None
        else:
            self.btn_set_roi.setText("Set CMD ROI")
            self._roi_drag_start = None
            if self._roi_preview_patch is not None:
                try:
                    self._roi_preview_patch.remove()
                except Exception:
                    pass
                self._roi_preview_patch = None
            self.canvas.draw_idle()

    def _on_clear_roi(self):
        self._roi_circle = None
        self._save_roi()
        self._update_roi_info_label()
        self._redraw_roi_patch()

    # ------------------------------------------------------------------

    def show_log_window(self):
        self.log_window.show()
        self.log_window.raise_()
        self.log_window.activateWindow()

    def step_frame(self, delta: int):
        if not self.file_list:
            return
        idx = self.file_combo.currentIndex()
        if idx < 0:
            return
        current_fname = self.file_combo.itemText(idx)
        current_filter = self._get_filter_for_file(current_fname)
        if not current_filter:
            current_filter = ""

        filter_indices = []
        for i, fname in enumerate(self.file_list):
            if self._get_filter_for_file(fname) == current_filter:
                filter_indices.append(i)

        if len(filter_indices) <= 1:
            new_idx = (idx + delta) % len(self.file_list)
        else:
            try:
                pos = filter_indices.index(idx)
            except ValueError:
                pos = 0
            new_idx = filter_indices[(pos + delta) % len(filter_indices)]

        if new_idx != idx:
            # Disconnect signal temporarily to call load_and_display with quick_switch flag
            self.file_combo.blockSignals(True)
            self.file_combo.setCurrentIndex(new_idx)
            self.file_combo.blockSignals(False)
            self.load_and_display(quick_switch=True)

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
        self._norm_cache.clear()  # Stretch changed — invalidate all normalized caches
        self._display_cache.clear()
        self._display_cache_order.clear()
        self.display_image()

    def update_stretch_label(self, value):
        self.stretch_value_label.setText(str(value))

    def update_black_label(self, value):
        self.black_value_label.setText(str(value))

    def redisplay_image(self):
        self.display_image()

    def _get_stretched_display_cached(self):
        if self.image_data is None:
            return None
        stretch_idx = int(self.scale_combo.currentIndex())
        intensity = int(self.stretch_slider.value())
        black_point = int(self.black_slider.value())
        cache_key = (self.current_filename, stretch_idx, intensity, black_point)
        if cache_key in self._display_cache:
            try:
                self._display_cache_order.remove(cache_key)
            except ValueError:
                pass
            self._display_cache_order.append(cache_key)
            return self._display_cache[cache_key]

        normalized = self.normalize_image()
        if normalized is None:
            return None
        stretched = self.apply_stretch(normalized).astype(np.float32, copy=False)

        while len(self._display_cache_order) >= self._FITS_CACHE_SIZE:
            oldest = self._display_cache_order.pop(0)
            self._display_cache.pop(oldest, None)
        self._display_cache[cache_key] = stretched
        self._display_cache_order.append(cache_key)
        return stretched

    def normalize_image(self):
        if self.image_data is None:
            return None

        stretch_idx = self.scale_combo.currentIndex()
        cache_key = (self.current_filename, stretch_idx)
        if cache_key in self._norm_cache:
            return self._norm_cache[cache_key]

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
        # LRU: evict oldest if over limit
        if len(self._norm_cache) >= self._FITS_CACHE_SIZE:
            oldest = next(iter(self._norm_cache))
            del self._norm_cache[oldest]
        self._norm_cache[cache_key] = normalized.astype(np.float32, copy=False)

        return self._norm_cache[cache_key]

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
        if self._roi_mode and event.button == 1 and event.inaxes == self.ax:
            if event.xdata is not None and event.ydata is not None:
                self._roi_drag_start = (event.xdata, event.ydata)
            return
        if event.button == 3:
            self.panning = True
            self.pan_start = (event.xdata, event.ydata)

    def on_button_release(self, event):
        if self._roi_mode and event.button == 1:
            if self._roi_drag_start is not None and event.inaxes == self.ax:
                x0, y0 = self._roi_drag_start
                x1 = event.xdata if event.xdata is not None else x0
                y1 = event.ydata if event.ydata is not None else y0
                r_px = float(np.hypot(x1 - x0, y1 - y0))
                if r_px >= 1.0:
                    roi = self._pixels_to_sky_roi(x0, y0, r_px)
                    if roi is not None:
                        self._roi_circle = roi
                        self._save_roi()
                        self._update_roi_info_label()
                        if self._roi_preview_patch is not None:
                            try:
                                self._roi_preview_patch.remove()
                            except Exception:
                                pass
                            self._roi_preview_patch = None
                        self._redraw_roi_patch()
                    else:
                        self.log("ROI: no WCS in current frame header, cannot convert to sky coords")
            self._roi_drag_start = None
            # exit draw mode
            self.btn_set_roi.blockSignals(True)
            self.btn_set_roi.setChecked(False)
            self.btn_set_roi.setText("Set CMD ROI")
            self.btn_set_roi.blockSignals(False)
            self._roi_mode = False
            return
        if event.button == 3:
            self.panning = False
            self.pan_start = None

    def on_motion(self, event):
        # Track hover position for G key
        if event.inaxes == self.ax and event.xdata is not None and event.ydata is not None:
            self.hover_xy = (event.xdata, event.ydata)

        # ROI drag preview
        if self._roi_mode and self._roi_drag_start is not None and event.inaxes == self.ax:
            if event.xdata is not None and event.ydata is not None:
                x0, y0 = self._roi_drag_start
                r = float(np.hypot(event.xdata - x0, event.ydata - y0))
                if self._roi_preview_patch is not None:
                    try:
                        self._roi_preview_patch.remove()
                    except Exception:
                        pass
                self._roi_preview_patch = MplCircle(
                    (x0, y0), r, fill=False,
                    edgecolor='#00E5FF', linestyle=':', linewidth=1.5, alpha=0.7
                )
                self.ax.add_patch(self._roi_preview_patch)
                self.canvas.draw_idle()
            return

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
