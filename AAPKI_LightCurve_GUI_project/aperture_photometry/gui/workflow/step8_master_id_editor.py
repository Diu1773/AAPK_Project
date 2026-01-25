"""
Step 8: Target/Comparison Selection (Filter-based)

완전 리팩토링:
- Reference Build 출력물(step6_refbuild/)을 입력으로 사용
- 필터별 타겟/비교성 선택
- Step 8에서 최종 master catalog 저장 (step8_selection/master_catalog_{filter}.tsv)
- 출력: step8_selection/ 폴더
"""

from __future__ import annotations

import time
import json
import re
from pathlib import Path
from typing import Dict, Set, Optional

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from astropy.coordinates import SkyCoord
import astropy.units as u
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.patheffects as pe

from PyQt5.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QGroupBox, QMessageBox,
    QTextEdit, QDialog, QFormLayout, QDialogButtonBox, QCheckBox, QSpinBox,
    QDoubleSpinBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QWidget, QComboBox, QSlider, QTabWidget, QShortcut
)
from PyQt5.QtGui import QKeySequence
from PyQt5.QtCore import Qt

from .step_window_base import StepWindowBase
from ...utils.step_paths import (
    step2_cropped_dir,
    step5_dir,
    step6_dir,
    step7_dir,
    legacy_step5_refbuild_dir,
    legacy_step6_idmatch_dir,
    legacy_step7_wcs_dir,
    legacy_step7_refbuild_dir,
)


_DATE_RE = re.compile(r"(20\d{6})")


def _extract_date_key(filename: str) -> str:
    match = _DATE_RE.search(str(filename))
    return match.group(1) if match else ""


def _resolve_idmatch_path(step7_dir: Path, legacy_dir: Path, filename: str) -> Path:
    date_key = _extract_date_key(filename)
    if date_key:
        dated = step7_dir / date_key / f"idmatch_{filename}.csv"
        if dated.exists():
            return dated
    direct = step7_dir / f"idmatch_{filename}.csv"
    if direct.exists():
        return direct
    matches = list(step7_dir.glob(f"*/idmatch_{filename}.csv"))
    if matches:
        return matches[0]
    legacy_direct = legacy_dir / f"idmatch_{filename}.csv"
    if legacy_direct.exists():
        return legacy_direct
    legacy_matches = list(legacy_dir.glob(f"*/idmatch_{filename}.csv"))
    return legacy_matches[0] if legacy_matches else direct


class MasterIdEditorWindow(StepWindowBase):
    """Step 8: Target/Comparison Selection (Filter-based)"""

    def __init__(self, params, file_manager, project_state, main_window):
        self.file_manager = file_manager
        self.file_list = []
        self.use_cropped = False
        self.current_filename = None
        self.current_filter = None
        self.image_data = None
        self.header = None
        self.ref_header = None
        self.idmatch_df = None

        # 필터별 데이터
        self.filter_frames: Dict[str, list] = {}  # filter -> frames
        self.filter_catalogs: Dict[str, pd.DataFrame] = {}  # filter -> reference catalog
        self.filter_targets: Dict[str, Optional[int]] = {}  # filter -> target source_id
        self.filter_comparisons: Dict[str, Set[int]] = {}  # filter -> comparison source_ids
        self.filter_master_ids: Dict[str, Set[int]] = {}  # filter -> final master source_ids (Step 8)
        self.catalog_ids: Set[int] = set()  # current filter reference catalog source_ids
        self._filter_all_source_ids: Dict[str, Set[int]] = {}
        self.step6_master_df: Optional[pd.DataFrame] = None
        self._step6_gmag_map: Dict[int, float] = {}
        self._step6_radec_map: Dict[int, tuple] = {}
        self._step6_bp_map: Dict[int, float] = {}
        self._step6_rp_map: Dict[int, float] = {}
        self._gaia_gmag_map: Dict[int, float] = {}
        self._gaia_radec_map: Dict[int, tuple] = {}
        self._gaia_bp_map: Dict[int, float] = {}
        self._gaia_rp_map: Dict[int, float] = {}

        self.master_ids: Set[int] = set()
        self.target_source_id: Optional[int] = None
        self.comparison_ids: Set[int] = set()
        self.selected_source_id: Optional[int] = None
        self.last_click_xy = None
        self.gaia_df = None
        self.sid_to_id: Dict[int, int] = {}
        self.id_to_sid: Dict[int, int] = {}
        self._pending_frame_index = None

        # Stable ID registry (per filter) - IDs persist across sessions
        self._id_registry: Dict[str, Dict[int, int]] = {}  # filter -> {source_id: stable_id}
        self._next_id: Dict[str, int] = {}  # filter -> next available ID
        self._retired_ids: Dict[str, Set[int]] = {}  # filter -> set of retired IDs (never reuse)

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
        self.hover_xy = None

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
            step_index=7,
            step_name="Target/Comparison Selection",
            params=params,
            project_state=project_state,
            main_window=main_window
        )

        self.setup_step_ui()
        self.restore_state()
        self.setFocusPolicy(Qt.StrongFocus)
        self._shortcuts = []
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
        sc_prev.activated.connect(lambda: self.step_frame(-1))
        self._shortcuts.append(sc_prev)
        sc_next = QShortcut(QKeySequence(Qt.Key_BracketRight), self)
        sc_next.setContext(Qt.ApplicationShortcut)
        sc_next.activated.connect(lambda: self.step_frame(1))
        self._shortcuts.append(sc_next)

    def setup_step_ui(self):
        info = QLabel(
            "Select target/comparison from ALL detected sources.\n"
            "ID: 고정 번호 (1,2,3...) | Gaia ID: 매칭된 Gaia DR3 source_id (없으면 '-')\n"
            "Shortcuts: T=target, C=comp, A=add, D=remove, Shift+D=box remove, G=profile, .=filter, [/]=frame"
        )
        info.setStyleSheet("QLabel { background-color: #E3F2FD; padding: 10px; border-radius: 5px; }")
        info.setWordWrap(True)
        self.content_layout.addWidget(info)

        # Reference Build 상태 및 필터 선택
        filter_group = QGroupBox("Filter Selection")
        filter_layout = QHBoxLayout(filter_group)

        filter_layout.addWidget(QLabel("Filter:"))
        self.filter_combo = QComboBox()
        self.filter_combo.setMinimumWidth(100)
        self.filter_combo.currentIndexChanged.connect(self.on_filter_changed)
        filter_layout.addWidget(self.filter_combo)

        self.step7_status_label = QLabel("Checking Reference Build...")
        filter_layout.addWidget(self.step7_status_label)

        filter_layout.addStretch()

        btn_log = QPushButton("Log")
        btn_log.setStyleSheet("QPushButton { background-color: #607D8B; color: white; font-weight: bold; padding: 8px 15px; }")
        btn_log.clicked.connect(self.show_log_window)
        filter_layout.addWidget(btn_log)

        self.content_layout.addWidget(filter_group)

        # 타겟/비교성 컨트롤
        select_group = QGroupBox("Target / Comparison")
        select_layout = QHBoxLayout(select_group)

        self.target_label = QLabel("Target: (none)")
        self.target_label.setStyleSheet("QLabel { font-weight: bold; color: #C62828; }")
        select_layout.addWidget(self.target_label)

        self.comparison_label = QLabel("Comparisons: 0")
        self.comparison_label.setStyleSheet("QLabel { font-weight: bold; color: #D32F2F; }")
        select_layout.addWidget(self.comparison_label)

        btn_set_target = QPushButton("Target (T)")
        btn_set_target.clicked.connect(self.set_target_selected)
        select_layout.addWidget(btn_set_target)

        btn_toggle_comp = QPushButton("Comparison (C)")
        btn_toggle_comp.clicked.connect(self.toggle_comparison_selected)
        select_layout.addWidget(btn_toggle_comp)

        self.comp_reco_count = QSpinBox()
        self.comp_reco_count.setRange(1, 20)
        self.comp_reco_count.setValue(5)
        select_layout.addWidget(QLabel("Recommend:"))
        select_layout.addWidget(self.comp_reco_count)

        btn_auto_comp = QPushButton("Auto Select")
        btn_auto_comp.clicked.connect(self.auto_select_comparisons)
        select_layout.addWidget(btn_auto_comp)

        btn_clear_comp = QPushButton("Clear Comps")
        btn_clear_comp.clicked.connect(self.clear_comparisons)
        select_layout.addWidget(btn_clear_comp)

        btn_simbad = QPushButton("SIMBAD Target")
        btn_simbad.clicked.connect(self.select_target_from_simbad)
        select_layout.addWidget(btn_simbad)

        btn_copy_to_all = QPushButton("Copy to All Filters")
        btn_copy_to_all.setToolTip("Copy current target/comparison selection to all other filters")
        btn_copy_to_all.clicked.connect(self.copy_selection_to_all_filters)
        select_layout.addWidget(btn_copy_to_all)

        select_layout.addStretch()

        # 선택된 소스만 표시 체크박스
        self.show_selected_only = QCheckBox("Show selected only")
        self.show_selected_only.setToolTip("Check to show only target and comparison stars in overlay")
        self.show_selected_only.stateChanged.connect(self.update_overlay)
        select_layout.addWidget(self.show_selected_only)

        self.content_layout.addWidget(select_group)

        # 선택 정보
        select_info_layout = QHBoxLayout()
        self.selected_label = QLabel("Selected: (none)")
        select_info_layout.addWidget(self.selected_label)
        select_info_layout.addStretch()
        self.content_layout.addLayout(select_info_layout)

        legend_group = QGroupBox("Overlay Legend")
        legend_layout = QHBoxLayout(legend_group)
        legend_layout.setSpacing(16)

        self.overlay_toggles = {}

        def _legend_item(key: str, color: str, text: str, checked: bool = True) -> QWidget:
            cb = QCheckBox()
            cb.setChecked(checked)
            cb.stateChanged.connect(self.update_overlay)
            swatch = QLabel()
            swatch.setFixedSize(12, 12)
            swatch.setStyleSheet(
                f"QLabel {{ background-color: {color}; border: 1px solid #555; }}"
            )
            label = QLabel(text)
            row = QHBoxLayout()
            row.setSpacing(6)
            row.addWidget(cb)
            row.addWidget(swatch)
            row.addWidget(label)
            wrap = QWidget()
            wrap.setLayout(row)
            self.overlay_toggles[key] = cb
            return wrap

        legend_layout.addWidget(_legend_item("matched_gaia", "#4CAF50", "ID matched + Gaia matched"))
        legend_layout.addWidget(_legend_item("matched_no_gaia", "#00BCD4", "ID matched (no Gaia)"))
        legend_layout.addWidget(_legend_item("ref_only", "#FFD54F", "Ref only (missing in frame)"))
        legend_layout.addWidget(_legend_item("detection_only", "#FF9800", "Detection only (unmatched)"))
        legend_layout.addWidget(_legend_item("comparison", "#D32F2F", "Comparison stars"))
        legend_layout.addWidget(_legend_item("target", "#C62828", "Target star"))
        legend_layout.addStretch()
        self.content_layout.addWidget(legend_group)

        # Viewer + table
        main_layout = QHBoxLayout()

        viewer_group = QGroupBox("Preview")
        viewer_layout = QVBoxLayout(viewer_group)

        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("Frame:"))
        self.file_combo = QComboBox()
        self.file_combo.currentIndexChanged.connect(self.on_file_changed)
        file_layout.addWidget(self.file_combo)
        viewer_layout.addLayout(file_layout)

        # Stretch controls
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
        stretch_layout.addWidget(self.stretch_slider)

        stretch_layout.addWidget(QLabel("Black:"))
        self.black_slider = QSlider(Qt.Horizontal)
        self.black_slider.setMinimum(0)
        self.black_slider.setMaximum(100)
        self.black_slider.setValue(0)
        self.black_slider.setFixedWidth(60)
        self.black_slider.sliderReleased.connect(self.redisplay_image)
        stretch_layout.addWidget(self.black_slider)

        btn_reset_zoom = QPushButton("Reset Zoom")
        btn_reset_zoom.clicked.connect(self.reset_zoom)
        stretch_layout.addWidget(btn_reset_zoom)

        btn_2d_plot = QPushButton("2D Plot")
        btn_2d_plot.setStyleSheet("QPushButton { background-color: #FF9800; color: white; font-weight: bold; }")
        btn_2d_plot.clicked.connect(self.open_stretch_plot)
        stretch_layout.addWidget(btn_2d_plot)

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

        # Detection table (all sources in current frame)
        table_group = QGroupBox("Detected Sources")
        table_layout = QVBoxLayout(table_group)

        self.master_table = QTableWidget()
        self.master_table.setColumnCount(7)
        self.master_table.setHorizontalHeaderLabels(["ID", "x", "y", "G mag", "Gaia ID", "Status", "Role"])
        # Add tooltips to column headers
        header = self.master_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        header.setStretchLastSection(True)
        self.master_table.setHorizontalHeaderItem(0, QTableWidgetItem("ID"))
        self.master_table.horizontalHeaderItem(0).setToolTip("고정 ID (1,2,3...) - 이미지에 표시됨")
        self.master_table.setHorizontalHeaderItem(4, QTableWidgetItem("Gaia ID"))
        self.master_table.horizontalHeaderItem(4).setToolTip("Gaia DR3 source_id (매칭 안되면 '-')")
        self.master_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.master_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.master_table.itemSelectionChanged.connect(self.on_table_selection_changed)
        table_layout.addWidget(self.master_table)

        main_layout.addWidget(table_group, stretch=1)

        self.content_layout.addLayout(main_layout)

        # 로그 윈도우
        self.log_window = QWidget(self, Qt.Window)
        self.log_window.setWindowTitle("Selection Log")
        self.log_window.resize(800, 400)
        log_layout = QVBoxLayout(self.log_window)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("QTextEdit { font-family: monospace; font-size: 9pt; }")
        log_layout.addWidget(self.log_text)

        # 초기화
        self.check_step7_status()
        self.load_gaia_catalog()
        self.load_step6_master_sources()

    def log(self, message: str):
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")

    def _restore_file_context(self):
        """Restore data_dir and file_path_map after restart if needed."""
        if getattr(self.params.P, "file_path_map", None):
            return

        if self.file_manager and getattr(self.file_manager, "path_map", None):
            if self.file_manager.path_map:
                self.params.P.file_path_map = {k: str(v) for k, v in self.file_manager.path_map.items()}
                return

        if not self.project_state:
            return

        state = self.project_state.get_step_data("file_selection")
        if not state:
            return

        data_dir = state.get("data_dir")
        if data_dir:
            self.params.P.data_dir = data_dir

        prefix = state.get("filename_prefix")
        if prefix:
            self.params.P.filename_prefix = prefix

        if self.file_manager:
            try:
                if state.get("multi_night") and state.get("night_dirs"):
                    root_dir = state.get("root_dir") or data_dir
                    night_dirs = [Path(p) for p in state.get("night_dirs", []) if p]
                    if root_dir:
                        self.file_manager.set_multi_night_dirs(Path(root_dir), night_dirs)
                else:
                    self.file_manager.clear_multi_night_dirs()

                if not self.file_manager.path_map:
                    self.file_manager.scan_files()
            except Exception as e:
                self.log(f"File scan warning: {e}")

            if self.file_manager.path_map:
                self.params.P.file_path_map = {k: str(v) for k, v in self.file_manager.path_map.items()}

    def check_step7_status(self):
        """Step 7/ID Match 출력물 확인 및 필터 로드 (Reference Build 없이도 동작)"""
        step7_out = step7_dir(self.params.P.result_dir)
        legacy_idmatch = legacy_step6_idmatch_dir(self.params.P.result_dir)
        step6_out = step6_dir(self.params.P.result_dir)
        legacy_refbuild = legacy_step5_refbuild_dir(self.params.P.result_dir)

        # Step 7 필터 정보 로드 (필수)
        filter_frames_path = step7_out / "step7_filter_frames.json"
        if not filter_frames_path.exists():
            filter_frames_path = legacy_idmatch / "step6_filter_frames.json"
        if not filter_frames_path.exists():
            self.step7_status_label.setText("Step 7 not complete. Run ID Match first.")
            self.step7_status_label.setStyleSheet("color: red;")
            return

        try:
            self.filter_frames.clear()
            self.filter_catalogs.clear()

            with open(filter_frames_path, "r", encoding="utf-8") as f:
                self.filter_frames = json.load(f)

            filters = list(self.filter_frames.keys())
            if not filters:
                self.step7_status_label.setText("No filters found in Step 7")
                self.step7_status_label.setStyleSheet("color: red;")
                return
        except Exception as e:
            self.log(f"Filter frames load error: {e}")
            self.step7_status_label.setText(f"Load error: {e}")
            self.step7_status_label.setStyleSheet("color: red;")
            return

        try:
            # Reference Build catalog 로드 (선택사항 - catalog_ids만 표시)
            step7_available = False
            meta_path = step6_out / "ref_build_meta.json"
            if not meta_path.exists():
                meta_path = legacy_refbuild / "ref_build_meta.json"
            if not meta_path.exists():
                meta_path = legacy_step7_refbuild_dir(self.params.P.result_dir) / "step7_meta.json"
            if meta_path.exists():
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        _ = json.load(f)
                except Exception as e:
                    self.log(f"Reference build meta warning: {e}")

            for flt in filters:
                catalog_path = step6_out / f"ref_catalog_{flt}.tsv"
                if not catalog_path.exists():
                    catalog_path = legacy_refbuild / f"ref_catalog_{flt}.tsv"
                if not catalog_path.exists():
                    catalog_path = legacy_step7_refbuild_dir(self.params.P.result_dir) / f"master_catalog_{flt}.tsv"
                if not catalog_path.exists():
                    # per-date fallback: pick the first matching ref catalog
                    matches = sorted(step6_out.glob(f"ref_catalog_{flt}_*.tsv"))
                    if matches:
                        catalog_path = matches[0]
                if catalog_path.exists():
                    try:
                        self.filter_catalogs[flt] = pd.read_csv(catalog_path, sep="\t")
                        step7_available = True
                        self.log(f"Ref catalog loaded: {catalog_path.name}")
                    except Exception as e:
                        self.log(f"Reference build load warning ({flt}): {e}")

            # 필터 콤보박스 업데이트
            self.filter_combo.blockSignals(True)
            self.filter_combo.clear()
            self.filter_combo.addItems(filters)
            self.filter_combo.blockSignals(False)

            # 기존 선택 로드
            self.load_selections()
            self.load_master_catalogs()

            n_filters = len(filters)
            if step7_available:
                total_sources = sum(len(cat) for cat in self.filter_catalogs.values())
                self.step7_status_label.setText(f"{n_filters} filters, {total_sources} catalog sources")
                self.step7_status_label.setStyleSheet("color: green;")
            else:
                self.step7_status_label.setText(
                    f"{n_filters} filters (Reference Build skipped - all detections available)"
                )
                self.step7_status_label.setStyleSheet("color: #FF9800;")  # Orange

            self._enrich_filter_catalogs_with_gaia()

            # 첫 번째 필터 선택
            if filters:
                self.filter_combo.setCurrentIndex(0)
                self.on_filter_changed(0)

        except Exception as e:
            self.step7_status_label.setText(f"Error: {e}")
            self.step7_status_label.setStyleSheet("color: red;")
            self.log(f"Error loading data: {e}")

    def load_selections(self):
        """기존 선택 데이터 로드"""
        step8_dir = self.params.P.result_dir / "step8_selection"
        if not step8_dir.exists():
            return

        # filter_frames 기준으로 선택 로드 (filter_catalogs 의존성 제거)
        for flt in self.filter_frames.keys():
            selection_path = step8_dir / f"selection_{flt}.json"
            if selection_path.exists():
                try:
                    with open(selection_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    target_sid = data.get("target_source_id")
                    if target_sid is not None:
                        self.filter_targets[flt] = int(target_sid)

                    comp_sids = data.get("comparison_source_ids", [])
                    self.filter_comparisons[flt] = set(int(s) for s in comp_sids if s is not None)

                except Exception as e:
                    self.log(f"Error loading selection for {flt}: {e}")

    def load_gaia_catalog(self):
        """Gaia 카탈로그 로드"""
        from astropy.table import Table
        gaia_path = step5_dir(self.params.P.result_dir) / "gaia_fov.ecsv"
        if not gaia_path.exists():
            gaia_path = legacy_step7_wcs_dir(self.params.P.result_dir) / "gaia_fov.ecsv"
        if not gaia_path.exists():
            gaia_path = self.params.P.result_dir / "gaia_fov.ecsv"
        if gaia_path.exists():
            try:
                tab = Table.read(str(gaia_path), format="ascii.ecsv")
                cols = list(tab.colnames)
                lower = [c.lower() for c in cols]
                if lower != cols:
                    tab.rename_columns(cols, lower)
                self.gaia_df = tab.to_pandas()
                if "source_id" in self.gaia_df.columns:
                    self.gaia_df["source_id"] = self.gaia_df["source_id"].astype("int64")
                self._gaia_gmag_map = {}
                self._gaia_radec_map = {}
                if self.gaia_df is not None and "source_id" in self.gaia_df.columns:
                    sid_series = pd.to_numeric(self.gaia_df["source_id"], errors="coerce")
                    g_col = None
                    for c in ("phot_g_mean_mag", "gaia_g", "gaia_G"):
                        if c in self.gaia_df.columns:
                            g_col = c
                            break
                    if g_col:
                        g_vals = pd.to_numeric(self.gaia_df[g_col], errors="coerce")
                        mask = sid_series.notna() & g_vals.notna()
                        self._gaia_gmag_map = {
                            int(sid): float(g)
                            for sid, g in zip(sid_series[mask], g_vals[mask])
                        }
                    if "ra" in self.gaia_df.columns and "dec" in self.gaia_df.columns:
                        ra_vals = pd.to_numeric(self.gaia_df["ra"], errors="coerce")
                        dec_vals = pd.to_numeric(self.gaia_df["dec"], errors="coerce")
                        mask = sid_series.notna() & ra_vals.notna() & dec_vals.notna()
                        self._gaia_radec_map = {
                            int(sid): (float(ra), float(dec))
                            for sid, ra, dec in zip(sid_series[mask], ra_vals[mask], dec_vals[mask])
                        }
                self.log(f"Gaia catalog: {len(self.gaia_df)} sources")
                self._enrich_filter_catalogs_with_gaia()
            except Exception as e:
                self.log(f"Failed to load Gaia: {e}")
                self.gaia_df = None
                self._gaia_gmag_map = {}
                self._gaia_radec_map = {}

    def load_step6_master_sources(self):
        """Step 7 master source list (ra/dec, Gaia mags)"""
        step6_path = self.params.P.result_dir / "step7_idmatch" / "step7_master_sources.csv"
        if not step6_path.exists():
            step6_path = legacy_step6_idmatch_dir(self.params.P.result_dir) / "step6_master_sources.csv"
        if not step6_path.exists():
            return
        try:
            df = pd.read_csv(step6_path)
            if "source_id" not in df.columns:
                return
            df["source_id"] = pd.to_numeric(df["source_id"], errors="coerce")
            df = df.dropna(subset=["source_id"])
            df["source_id"] = df["source_id"].astype("int64")
            df = df.set_index("source_id", drop=False)
            self.step6_master_df = df

            self._step6_gmag_map = {}
            self._step6_radec_map = {}
            g_col = None
            for col in ("phot_g_mean_mag", "gaia_g", "gaia_G"):
                if col in df.columns:
                    g_col = col
                    break
            if g_col:
                g_vals = pd.to_numeric(df[g_col], errors="coerce")
                mask = g_vals.notna()
                self._step6_gmag_map = {
                    int(sid): float(g)
                    for sid, g in zip(df.loc[mask, "source_id"], g_vals[mask])
                }
            if "ra_deg" in df.columns and "dec_deg" in df.columns:
                ra_vals = pd.to_numeric(df["ra_deg"], errors="coerce")
                dec_vals = pd.to_numeric(df["dec_deg"], errors="coerce")
                mask = ra_vals.notna() & dec_vals.notna()
                self._step6_radec_map = {
                    int(sid): (float(ra), float(dec))
                    for sid, ra, dec in zip(df.loc[mask, "source_id"], ra_vals[mask], dec_vals[mask])
                }

            self._build_gmag_from_gaia_radec()
            self.log(f"Step6 master sources: {len(df)}")
        except Exception as e:
            self.step6_master_df = None
            self._step6_gmag_map = {}
            self._step6_radec_map = {}
            self.log(f"Failed to load Step6 master sources: {e}")

    def _enrich_filter_catalogs_with_gaia(self):
        if not self.filter_catalogs:
            return
        if self.gaia_df is None or self.gaia_df.empty:
            return
        if "ra" not in self.gaia_df.columns or "dec" not in self.gaia_df.columns:
            return

        g_col = None
        for col in ("phot_g_mean_mag", "gaia_g", "gaia_G"):
            if col in self.gaia_df.columns:
                g_col = col
                break
        bp_col = None
        for col in ("phot_bp_mean_mag", "gaia_bp", "gaia_BP"):
            if col in self.gaia_df.columns:
                bp_col = col
                break
        rp_col = None
        for col in ("phot_rp_mean_mag", "gaia_rp", "gaia_RP"):
            if col in self.gaia_df.columns:
                rp_col = col
                break

        gaia_ra = pd.to_numeric(self.gaia_df["ra"], errors="coerce")
        gaia_dec = pd.to_numeric(self.gaia_df["dec"], errors="coerce")
        gaia_g = pd.to_numeric(self.gaia_df[g_col], errors="coerce") if g_col else None
        gaia_bp = pd.to_numeric(self.gaia_df[bp_col], errors="coerce") if bp_col else None
        gaia_rp = pd.to_numeric(self.gaia_df[rp_col], errors="coerce") if rp_col else None

        mask = gaia_ra.notna() & gaia_dec.notna()
        if g_col:
            mask &= gaia_g.notna()
        if not mask.any():
            return

        gaia_sky = SkyCoord(gaia_ra[mask].to_numpy(float) * u.deg,
                            gaia_dec[mask].to_numpy(float) * u.deg,
                            frame="icrs")
        gaia_g_vals = gaia_g[mask].to_numpy(float) if g_col else None
        gaia_bp_vals = gaia_bp[mask].to_numpy(float) if bp_col else None
        gaia_rp_vals = gaia_rp[mask].to_numpy(float) if rp_col else None

        radius = float(getattr(self.params.P, "ref_wcs_match_radius_arcsec", 2.0))
        if not np.isfinite(radius) or radius <= 0:
            radius = float(getattr(self.params.P, "idmatch_tol_arcsec", 2.0))
        if not np.isfinite(radius) or radius <= 0:
            radius = 2.0

        for flt, cat in list(self.filter_catalogs.items()):
            if "ra_deg" not in cat.columns or "dec_deg" not in cat.columns:
                continue
            ra = pd.to_numeric(cat["ra_deg"], errors="coerce")
            dec = pd.to_numeric(cat["dec_deg"], errors="coerce")
            src_mask = ra.notna() & dec.notna()
            if not src_mask.any():
                continue

            src_idx = np.where(src_mask)[0]
            src_sky = SkyCoord(ra[src_mask].to_numpy(float) * u.deg,
                               dec[src_mask].to_numpy(float) * u.deg,
                               frame="icrs")
            idx, sep2d, _ = src_sky.match_to_catalog_sky(gaia_sky)
            ok = sep2d.arcsec <= radius
            if not np.any(ok):
                continue

            cat = cat.copy()
            if g_col:
                out_g = pd.to_numeric(cat["gaia_G"], errors="coerce").to_numpy() if "gaia_G" in cat.columns else np.full(len(cat), np.nan)
                fill_idx = src_idx[ok]
                fill_vals = gaia_g_vals[idx[ok]]
                needs = ~np.isfinite(out_g[fill_idx])
                out_g[fill_idx[needs]] = fill_vals[needs]
                cat["gaia_G"] = out_g
                cat["gaia_g"] = out_g
            if bp_col:
                out_bp = pd.to_numeric(cat["gaia_BP"], errors="coerce").to_numpy() if "gaia_BP" in cat.columns else np.full(len(cat), np.nan)
                fill_idx = src_idx[ok]
                fill_vals = gaia_bp_vals[idx[ok]]
                needs = ~np.isfinite(out_bp[fill_idx])
                out_bp[fill_idx[needs]] = fill_vals[needs]
                cat["gaia_BP"] = out_bp
                cat["gaia_bp"] = out_bp
            if rp_col:
                out_rp = pd.to_numeric(cat["gaia_RP"], errors="coerce").to_numpy() if "gaia_RP" in cat.columns else np.full(len(cat), np.nan)
                fill_idx = src_idx[ok]
                fill_vals = gaia_rp_vals[idx[ok]]
                needs = ~np.isfinite(out_rp[fill_idx])
                out_rp[fill_idx[needs]] = fill_vals[needs]
                cat["gaia_RP"] = out_rp
                cat["gaia_rp"] = out_rp
            if "gaia_bp" in cat.columns and "gaia_rp" in cat.columns:
                try:
                    cat["color_gr"] = pd.to_numeric(cat["gaia_bp"], errors="coerce") - pd.to_numeric(cat["gaia_rp"], errors="coerce")
                except Exception:
                    pass

            self.filter_catalogs[flt] = cat

    def _build_gmag_from_gaia_radec(self):
        if self.gaia_df is None or self.gaia_df.empty:
            return
        if not self._step6_radec_map:
            return
        if "ra" not in self.gaia_df.columns or "dec" not in self.gaia_df.columns:
            return
        g_col = None
        for col in ("phot_g_mean_mag", "gaia_g", "gaia_G"):
            if col in self.gaia_df.columns:
                g_col = col
                break
        if not g_col:
            return

        gaia_ra = pd.to_numeric(self.gaia_df["ra"], errors="coerce")
        gaia_dec = pd.to_numeric(self.gaia_df["dec"], errors="coerce")
        gaia_g = pd.to_numeric(self.gaia_df[g_col], errors="coerce")
        mask = gaia_ra.notna() & gaia_dec.notna() & gaia_g.notna()
        if not mask.any():
            return
        gaia_sky = SkyCoord(gaia_ra[mask].to_numpy(float) * u.deg,
                            gaia_dec[mask].to_numpy(float) * u.deg,
                            frame="icrs")
        gaia_g = gaia_g[mask].to_numpy(float)

        src_sids = []
        src_ra = []
        src_dec = []
        for sid, (ra, dec) in self._step6_radec_map.items():
            if sid in self._step6_gmag_map:
                continue
            if not np.isfinite(ra) or not np.isfinite(dec):
                continue
            src_sids.append(int(sid))
            src_ra.append(float(ra))
            src_dec.append(float(dec))
        if not src_sids:
            return
        src_sky = SkyCoord(np.asarray(src_ra) * u.deg, np.asarray(src_dec) * u.deg, frame="icrs")
        idx, sep2d, _ = src_sky.match_to_catalog_sky(gaia_sky)

        radius = float(getattr(self.params.P, "ref_wcs_match_radius_arcsec", 2.0))
        if not np.isfinite(radius) or radius <= 0:
            radius = float(getattr(self.params.P, "idmatch_tol_arcsec", 2.0))
        if not np.isfinite(radius) or radius <= 0:
            radius = 2.0

        matched = 0
        for i, sid in enumerate(src_sids):
            if sep2d.arcsec[i] <= radius:
                g_val = gaia_g[int(idx[i])]
                if np.isfinite(g_val):
                    self._step6_gmag_map[sid] = float(g_val)
                    matched += 1
        if matched > 0:
            self.log(f"Gaia G matched to ref sources: {matched}")

    def load_master_catalogs(self):
        """Load Step 8 master catalogs if they already exist."""
        self.filter_master_ids.clear()
        step8_dir = self.params.P.result_dir / "step8_selection"
        if not step8_dir.exists():
            return
        for flt in self.filter_frames.keys():
            path = step8_dir / f"master_catalog_{flt}.tsv"
            if not path.exists():
                continue
            try:
                df = pd.read_csv(path, sep="\t")
                if "source_id" not in df.columns:
                    continue
                sids = pd.to_numeric(df["source_id"], errors="coerce").dropna().astype("int64")
                self.filter_master_ids[flt] = set(sids.tolist())
                self.log(f"Master catalog loaded: {path.name} ({len(self.filter_master_ids[flt])} sources)")
            except Exception as e:
                self.log(f"Failed to load master catalog for {flt}: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Stable ID Registry Management
    # ─────────────────────────────────────────────────────────────────────────

    def _load_id_registry(self, flt: str) -> None:
        """Load persistent ID registry for a filter, or migrate from legacy format."""
        if flt in self._id_registry:
            return  # Already loaded

        step8_dir = self.params.P.result_dir / "step8_selection"
        registry_path = step8_dir / f"id_registry_{flt}.json"

        if registry_path.exists():
            # Load existing persistent registry
            try:
                data = json.loads(registry_path.read_text(encoding="utf-8"))
                self._id_registry[flt] = {
                    int(k): int(v) for k, v in data.get("source_id_to_stable_id", {}).items()
                }
                self._next_id[flt] = int(data.get("next_id", 1))
                self._retired_ids[flt] = set(int(x) for x in data.get("retired_ids", []))
                return
            except Exception:
                pass

        # Migration: try to load from existing master_catalog
        legacy_path = step8_dir / f"master_catalog_{flt}.tsv"
        if legacy_path.exists():
            try:
                df = pd.read_csv(legacy_path, sep="\t")
                if {"source_id", "ID"} <= set(df.columns):
                    sid_col = pd.to_numeric(df["source_id"], errors="coerce").dropna().astype("int64")
                    id_col = pd.to_numeric(df["ID"], errors="coerce").dropna().astype("int64")
                    if len(sid_col) == len(id_col):
                        self._id_registry[flt] = dict(zip(sid_col.tolist(), id_col.tolist()))
                        max_id = id_col.max() if not id_col.empty else 0
                        self._next_id[flt] = int(max_id) + 1
                        self._retired_ids[flt] = set()
                        # Save migrated registry
                        self._save_id_registry(flt)
                        self.log(f"ID registry migrated from legacy: {flt} ({len(self._id_registry[flt])} IDs)")
                        return
            except Exception:
                pass

        # Fresh start
        self._id_registry[flt] = {}
        self._next_id[flt] = 1
        self._retired_ids[flt] = set()

    def _save_id_registry(self, flt: str) -> None:
        """Persist the ID registry for a filter."""
        if flt not in self._id_registry:
            return

        step8_dir = self.params.P.result_dir / "step8_selection"
        step8_dir.mkdir(parents=True, exist_ok=True)
        registry_path = step8_dir / f"id_registry_{flt}.json"

        data = {
            "source_id_to_stable_id": {
                str(k): v for k, v in self._id_registry[flt].items()
            },
            "next_id": self._next_id.get(flt, 1),
            "retired_ids": sorted(self._retired_ids.get(flt, set())),
        }
        registry_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _get_or_assign_stable_id(self, flt: str, source_id: int) -> int:
        """Get existing stable ID or assign a new one. Retired IDs are never reused."""
        self._load_id_registry(flt)

        source_id = int(source_id)
        registry = self._id_registry[flt]

        if source_id in registry:
            return registry[source_id]

        # Assign next available ID, skipping retired ones
        new_id = self._next_id.get(flt, 1)
        retired = self._retired_ids.get(flt, set())
        while new_id in retired:
            new_id += 1

        registry[source_id] = new_id
        self._next_id[flt] = new_id + 1
        return new_id

    def _retire_stable_id(self, flt: str, source_id: int) -> None:
        """Mark a source's ID as retired (source removed). The ID will never be reused."""
        self._load_id_registry(flt)

        source_id = int(source_id)
        registry = self._id_registry.get(flt, {})

        if source_id in registry:
            stable_id = registry[source_id]
            if flt not in self._retired_ids:
                self._retired_ids[flt] = set()
            self._retired_ids[flt].add(stable_id)
            del registry[source_id]

    # ─────────────────────────────────────────────────────────────────────────

    def _collect_filter_source_ids(self, flt: str) -> Set[int]:
        if flt in self._filter_all_source_ids:
            return set(self._filter_all_source_ids[flt])

        step6_dir = self.params.P.result_dir / "step7_idmatch"
        legacy_idmatch = legacy_step6_idmatch_dir(self.params.P.result_dir)
        frames = self.filter_frames.get(flt, [])
        sids: Set[int] = set()

        for fname in frames:
            p = _resolve_idmatch_path(step6_dir, legacy_idmatch, fname)
            if not p.exists():
                continue
            try:
                df = pd.read_csv(p, usecols=["source_id"])
                if "source_id" not in df.columns:
                    continue
                vals = pd.to_numeric(df["source_id"], errors="coerce").dropna().astype("int64")
                sids.update(vals.tolist())
            except Exception:
                continue

        if not sids and self.step6_master_df is not None:
            sids = set(self.step6_master_df["source_id"].astype("int64").tolist())

        self._filter_all_source_ids[flt] = set(sids)
        return set(sids)

    def _ensure_master_ids_for_filter(self, flt: str):
        if flt in self.filter_master_ids:
            self.master_ids = self.filter_master_ids[flt]
            return
        sids = self._collect_filter_source_ids(flt)
        self.filter_master_ids[flt] = set(sids)
        self.master_ids = self.filter_master_ids[flt]

    def _get_gmag_for_source(self, source_id: int) -> float:
        sid = int(source_id)
        if sid in self._step6_gmag_map:
            return float(self._step6_gmag_map[sid])
        if sid in self._gaia_gmag_map:
            return float(self._gaia_gmag_map[sid])
        return float("nan")

    def _get_color_for_source(self, source_id: int) -> float:
        """Get BP-RP color index for a source"""
        sid = int(source_id)
        # Check filter catalog first
        if self.current_filter and self.current_filter in self.filter_catalogs:
            cat = self.filter_catalogs[self.current_filter]
            if "source_id" in cat.columns:
                row = cat[cat["source_id"] == sid]
                if not row.empty:
                    # Check color_gr (BP-RP) column
                    for col in ("color_gr", "bp_rp", "BP_RP"):
                        if col in row.columns:
                            val = pd.to_numeric(row.iloc[0][col], errors="coerce")
                            if np.isfinite(val):
                                return float(val)
                    # Compute from BP and RP
                    bp_val = np.nan
                    rp_val = np.nan
                    for bp_col in ("gaia_BP", "gaia_bp", "phot_bp_mean_mag"):
                        if bp_col in row.columns:
                            bp_val = pd.to_numeric(row.iloc[0][bp_col], errors="coerce")
                            if np.isfinite(bp_val):
                                break
                    for rp_col in ("gaia_RP", "gaia_rp", "phot_rp_mean_mag"):
                        if rp_col in row.columns:
                            rp_val = pd.to_numeric(row.iloc[0][rp_col], errors="coerce")
                            if np.isfinite(rp_val):
                                break
                    if np.isfinite(bp_val) and np.isfinite(rp_val):
                        return float(bp_val - rp_val)

        # Check step6 master
        if self.step6_master_df is not None and sid in self.step6_master_df.index:
            src = self.step6_master_df.loc[sid]
            bp_val = np.nan
            rp_val = np.nan
            for bp_col in ("phot_bp_mean_mag", "gaia_bp", "gaia_BP"):
                if bp_col in src.index and pd.notna(src[bp_col]):
                    bp_val = float(src[bp_col])
                    break
            for rp_col in ("phot_rp_mean_mag", "gaia_rp", "gaia_RP"):
                if rp_col in src.index and pd.notna(src[rp_col]):
                    rp_val = float(src[rp_col])
                    break
            if np.isfinite(bp_val) and np.isfinite(rp_val):
                return float(bp_val - rp_val)

        # Check gaia_df
        if self.gaia_df is not None and not self.gaia_df.empty:
            if "source_id" in self.gaia_df.columns:
                row = self.gaia_df[self.gaia_df["source_id"] == sid]
                if not row.empty:
                    bp_val = np.nan
                    rp_val = np.nan
                    for bp_col in ("phot_bp_mean_mag", "gaia_bp", "gaia_BP"):
                        if bp_col in row.columns:
                            bp_val = pd.to_numeric(row.iloc[0][bp_col], errors="coerce")
                            if np.isfinite(bp_val):
                                break
                    for rp_col in ("phot_rp_mean_mag", "gaia_rp", "gaia_RP"):
                        if rp_col in row.columns:
                            rp_val = pd.to_numeric(row.iloc[0][rp_col], errors="coerce")
                            if np.isfinite(rp_val):
                                break
                    if np.isfinite(bp_val) and np.isfinite(rp_val):
                        return float(bp_val - rp_val)

        return float("nan")

    def _get_radec_for_source(self, source_id: int) -> tuple:
        sid = int(source_id)
        if sid in self._step6_radec_map:
            return self._step6_radec_map[sid]
        if sid in self._gaia_radec_map:
            return self._gaia_radec_map[sid]
        if self.idmatch_df is not None and not self.idmatch_df.empty:
            df = self.idmatch_df
            for ra_col, dec_col in (("ra_gaia", "dec_gaia"), ("ra", "dec"), ("RA", "DEC")):
                if ra_col in df.columns and dec_col in df.columns:
                    sub = df[df["source_id"] == sid]
                    if len(sub):
                        ra = pd.to_numeric(sub.iloc[0][ra_col], errors="coerce")
                        dec = pd.to_numeric(sub.iloc[0][dec_col], errors="coerce")
                        if np.isfinite(ra) and np.isfinite(dec):
                            return float(ra), float(dec)
        return (float("nan"), float("nan"))

    def _get_xy_ref_for_source(self, source_id: int, ra_deg: float, dec_deg: float) -> tuple:
        sid = int(source_id)
        if self.idmatch_df is not None and not self.idmatch_df.empty:
            sub = self.idmatch_df[self.idmatch_df["source_id"] == sid]
            if len(sub):
                try:
                    return float(sub.iloc[0]["x"]), float(sub.iloc[0]["y"])
                except Exception:
                    pass

        if np.isfinite(ra_deg) and np.isfinite(dec_deg):
            w = self._get_wcs()
            if w is not None:
                try:
                    sc = SkyCoord(ra_deg * u.deg, dec_deg * u.deg, frame="icrs")
                    x_pix, y_pix = w.celestial.world_to_pixel(sc)
                    return float(x_pix), float(y_pix)
                except Exception:
                    pass

        return (float("nan"), float("nan"))

    def _build_master_row_for_sid(self, source_id: int) -> dict:
        sid = int(source_id)
        row = {"source_id": sid}

        # Gaia ID와 match_status 결정
        if sid > 0:
            # 양수 = Gaia source_id
            row["gaia_id"] = sid
            row["match_status"] = "matched"
        else:
            # 음수 = 로컬 ID (Gaia 매칭 안됨)
            row["gaia_id"] = np.nan
            # 왜 매칭 안됐는지 판단
            g_mag = self._get_gmag_for_source(sid)
            if not np.isfinite(g_mag):
                row["match_status"] = "no_gaia_in_radius"
            elif g_mag > 20.0:
                row["match_status"] = "too_faint_for_gaia"
            else:
                row["match_status"] = "no_gaia_match"

        ra_deg, dec_deg = self._get_radec_for_source(sid)
        if np.isfinite(ra_deg):
            row["ra_deg"] = float(ra_deg)
        if np.isfinite(dec_deg):
            row["dec_deg"] = float(dec_deg)

        x_ref, y_ref = self._get_xy_ref_for_source(sid, ra_deg, dec_deg)
        if np.isfinite(x_ref):
            row["x_ref"] = float(x_ref)
        if np.isfinite(y_ref):
            row["y_ref"] = float(y_ref)
        if (not np.isfinite(ra_deg)) and np.isfinite(x_ref) and np.isfinite(y_ref):
            w = self._get_wcs()
            if w is not None:
                try:
                    ra_new, dec_new = w.all_pix2world([x_ref], [y_ref], 0)
                    ra_new = float(ra_new[0])
                    dec_new = float(dec_new[0])
                    if np.isfinite(ra_new) and np.isfinite(dec_new):
                        row["ra_deg"] = ra_new
                        row["dec_deg"] = dec_new
                except Exception:
                    pass

        if self.step6_master_df is not None and sid in self.step6_master_df.index:
            src = self.step6_master_df.loc[sid]
            for col in ("n_frames", "sep_median", "sep_std"):
                if col in src.index and pd.notna(src[col]):
                    row[col] = src[col]

            if "phot_bp_mean_mag" in src.index and pd.notna(src["phot_bp_mean_mag"]):
                row["gaia_bp"] = float(src["phot_bp_mean_mag"])
                row["gaia_BP"] = float(src["phot_bp_mean_mag"])
            if "phot_rp_mean_mag" in src.index and pd.notna(src["phot_rp_mean_mag"]):
                row["gaia_rp"] = float(src["phot_rp_mean_mag"])
                row["gaia_RP"] = float(src["phot_rp_mean_mag"])

        g_mag = self._get_gmag_for_source(sid)
        if np.isfinite(g_mag):
            row["gaia_g"] = float(g_mag)
            row["gaia_G"] = float(g_mag)

        if "gaia_bp" in row and "gaia_rp" in row:
            try:
                row["color_gr"] = float(row["gaia_bp"]) - float(row["gaia_rp"])
            except Exception:
                pass

        return row

    def save_master_catalog(self, flt: Optional[str] = None, log_action: Optional[str] = None):
        if flt is None:
            flt = self.current_filter
        if not flt:
            return

        master_ids = self.filter_master_ids.get(flt)
        if master_ids is None:
            return

        step8_dir = self.params.P.result_dir / "step8_selection"
        step8_dir.mkdir(parents=True, exist_ok=True)

        base_df = None
        if flt in self.filter_catalogs:
            base_df = self.filter_catalogs[flt].copy()
            if "source_id" in base_df.columns:
                base_df["source_id"] = pd.to_numeric(base_df["source_id"], errors="coerce")
                base_df = base_df.dropna(subset=["source_id"])
                base_df["source_id"] = base_df["source_id"].astype("int64")
                base_df = base_df[base_df["source_id"].isin(master_ids)]
        if base_df is None:
            base_df = pd.DataFrame(columns=["source_id"])

        only_source_col = set(base_df.columns) <= {"source_id"}
        if base_df.empty or only_source_col:
            base_df = pd.DataFrame([self._build_master_row_for_sid(sid) for sid in sorted(master_ids)])
        else:
            present = set()
            if "source_id" in base_df.columns and not base_df.empty:
                present = set(base_df["source_id"].astype("int64").tolist())
            missing = set(master_ids) - present

            if missing:
                extra_rows = [self._build_master_row_for_sid(sid) for sid in sorted(missing)]
                extra_df = pd.DataFrame(extra_rows)
                if base_df.empty:
                    base_df = extra_df
                else:
                    base_df = pd.concat([base_df, extra_df], ignore_index=True, sort=False)

        df_out = base_df.copy()
        if "source_id" not in df_out.columns:
            df_out["source_id"] = list(master_ids)

        # Load/initialize ID registry and assign stable IDs
        # (IDs are preserved across saves - only new sources get new IDs)
        self._load_id_registry(flt)
        df_out["ID"] = df_out["source_id"].apply(
            lambda sid: self._get_or_assign_stable_id(flt, int(sid))
        )
        # Sort by stable ID for consistent display order
        df_out = df_out.sort_values("ID")
        # Save updated ID registry
        self._save_id_registry(flt)

        # Role 컬럼 추가 (Target/Comparison)
        target_sid = self.filter_targets.get(flt)
        comp_sids = self.filter_comparisons.get(flt, set())

        def get_role(sid):
            sid = int(sid)
            if target_sid is not None and sid == target_sid:
                return "T"
            elif sid in comp_sids:
                return "C"
            return ""

        df_out["role"] = df_out["source_id"].apply(get_role)

        # gaia_id 컬럼이 없으면 source_id에서 생성
        if "gaia_id" not in df_out.columns:
            df_out["gaia_id"] = df_out["source_id"].apply(
                lambda x: x if int(x) > 0 else np.nan
            )
        if "match_status" not in df_out.columns:
            df_out["match_status"] = df_out["source_id"].apply(
                lambda x: "matched" if int(x) > 0 else "no_gaia_match"
            )

        # 컬럼 순서 정리 (ID 우선, gaia_id는 참조용)
        output_cols = ["ID", "x_ref", "y_ref", "ra_deg", "dec_deg", "role", "gaia_id", "match_status"]
        for col in [
            "gaia_G", "gaia_BP", "gaia_RP",
            "gaia_g", "gaia_bp", "gaia_rp", "color_gr",
            "n_frames", "x_mean", "y_mean", "x_std", "y_std",
        ]:
            if col in df_out.columns and col not in output_cols:
                output_cols.append(col)
        # source_id는 내부용으로 마지막에
        if "source_id" not in output_cols:
            output_cols.append("source_id")
        for col in df_out.columns:
            if col not in output_cols:
                output_cols.append(col)

        df_out = df_out[[c for c in output_cols if c in df_out.columns]]

        out_path = step8_dir / f"master_catalog_{flt}.tsv"
        df_out.to_csv(out_path, sep="\t", index=False, na_rep="NaN", encoding="utf-8-sig")

        # ID 매핑 파일도 저장 (다른 Step에서 참조용)
        id_map_path = step8_dir / f"id_mapping_{flt}.csv"
        id_map_df = df_out[["ID", "source_id", "gaia_id", "role", "x_ref", "y_ref"]].copy()
        id_map_df.to_csv(id_map_path, index=False, na_rep="NaN")

        if log_action:
            self.log(f"Master catalog saved ({flt}): {len(df_out)} sources [{log_action}]")
        else:
            self.log(f"Master catalog saved ({flt}): {len(df_out)} sources")

    def _get_wcs(self) -> Optional[WCS]:
        if self.header is not None:
            try:
                w = WCS(self.header, relax=True)
                if w.has_celestial:
                    return w
            except Exception:
                pass
        self._load_ref_header()
        if self.ref_header is not None:
            try:
                w = WCS(self.ref_header, relax=True)
                if w.has_celestial:
                    return w
            except Exception:
                pass
        return None

    def _load_ref_header(self) -> None:
        if self.ref_header is not None:
            return
        meta_path = step6_dir(self.params.P.result_dir) / "ref_build_meta.json"
        if not meta_path.exists():
            meta_path = legacy_step5_refbuild_dir(self.params.P.result_dir) / "ref_build_meta.json"
        if not meta_path.exists():
            meta_path = legacy_step7_refbuild_dir(self.params.P.result_dir) / "ref_build_meta.json"
        if not meta_path.exists():
            return
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            ref_frame = meta.get("ref_frame")
        except Exception:
            ref_frame = None
        if not ref_frame:
            return

        data_dir = Path(getattr(self.params.P, "data_dir", ""))
        cropped_dir = step2_cropped_dir(self.params.P.result_dir)
        candidates = []
        try:
            candidates.append(Path(self.params.get_file_path(ref_frame)))
        except Exception:
            pass
        candidates.extend([
            cropped_dir / ref_frame,
            data_dir / ref_frame,
        ])
        for cand in candidates:
            if cand.exists():
                try:
                    self.ref_header = fits.getheader(cand)
                    return
                except Exception:
                    continue


    def on_filter_changed(self, index):
        """필터 변경 처리"""
        if index < 0:
            return

        self.current_filter = self.filter_combo.currentText()
        if not self.current_filter:
            return

        self.log(f"Filter changed to: {self.current_filter}")

        # 해당 필터의 카탈로그/마스터 로드 (있으면)
        self.catalog_ids = set()
        self.master_ids = set()
        self.sid_to_id = {}
        self.id_to_sid = {}

        if self.current_filter in self.filter_catalogs:
            cat = self.filter_catalogs[self.current_filter]
            self.catalog_ids = set(cat["source_id"].astype("int64").tolist())

            # ID 매핑 (reference catalog 있을 때만)
            self.sid_to_id = dict(zip(
                cat["source_id"].astype("int64"),
                cat["ID"].astype("int64")
            ))
            self.id_to_sid = dict(zip(
                cat["ID"].astype("int64"),
                cat["source_id"].astype("int64")
            ))

        # Step 8 master (final) load/init
        self._ensure_master_ids_for_filter(self.current_filter)

        # 타겟/비교성 로드
        self.target_source_id = self.filter_targets.get(self.current_filter)
        self.comparison_ids = self.filter_comparisons.get(self.current_filter, set()).copy()

        # 파일 목록 업데이트
        self.populate_file_list()
        self.update_target_labels()
        self.update_master_table()

    def populate_file_list(self):
        """현재 필터의 파일 목록 로드"""
        if not self.current_filter:
            return

        frames = self.filter_frames.get(self.current_filter, [])

        # cropped 디렉토리 확인
        cropped_dir = step2_cropped_dir(self.params.P.result_dir)
        self.use_cropped = cropped_dir.exists() and list(cropped_dir.glob("*.fit*"))

        self.file_list = list(frames)
        self.file_combo.blockSignals(True)
        self.file_combo.clear()
        self.file_combo.addItems(self.file_list)
        self.file_combo.blockSignals(False)

        if self.file_list:
            if self._pending_frame_index is not None:
                idx = max(0, min(len(self.file_list) - 1, int(self._pending_frame_index)))
                self._pending_frame_index = None
                self.file_combo.setCurrentIndex(idx)
                self.on_file_changed(idx)
            else:
                self.file_combo.setCurrentIndex(0)
                self.on_file_changed(0)

    def on_file_changed(self, index):
        if index < 0 or index >= len(self.file_list):
            return
        self.load_and_display()

    def load_and_display(self):
        filename = self.file_combo.currentText()
        if not filename:
            return
        try:
            self._restore_file_context()
            # 파일 경로 찾기 (여러 위치 시도)
            file_path = None
            cropped_dir = step2_cropped_dir(self.params.P.result_dir)
            legacy_cropped_dir = self.params.P.result_dir / "cropped"
            data_dir = Path(self.params.P.data_dir)
            mapped_path = None
            try:
                mapped_path = self.params.get_file_path(filename)
            except Exception:
                mapped_path = None

            # 후보 경로들
            candidates = []
            if mapped_path:
                candidates.append(Path(mapped_path))
            candidates.extend([
                cropped_dir / filename,
                legacy_cropped_dir / filename,
                data_dir / filename,
            ])
            # 확장자 없으면 추가
            if not filename.endswith(('.fits', '.fit', '.fts', '.fit.fz', '.fits.fz')):
                candidates.extend([
                    cropped_dir / f"{filename}.fits",
                    cropped_dir / f"{filename}.fit",
                    cropped_dir / f"{filename}.fit.fz",
                    cropped_dir / f"{filename}.fits.fz",
                    legacy_cropped_dir / f"{filename}.fits",
                    legacy_cropped_dir / f"{filename}.fit",
                    legacy_cropped_dir / f"{filename}.fit.fz",
                    legacy_cropped_dir / f"{filename}.fits.fz",
                    data_dir / f"{filename}.fits",
                    data_dir / f"{filename}.fit",
                    data_dir / f"{filename}.fit.fz",
                    data_dir / f"{filename}.fits.fz",
                ])

            # 후보에서 존재하는 파일 찾기
            for cand in candidates:
                if cand.exists():
                    file_path = cand
                    break

            # 못 찾으면 glob 시도
            if file_path is None:
                for d in [cropped_dir, data_dir]:
                    if d.exists():
                        matches = list(d.glob(f"{filename}*")) + list(d.glob(f"*{filename}*"))
                        for m in matches:
                            if m.is_file() and m.name.lower().endswith(('.fits', '.fit', '.fts', '.fit.fz', '.fits.fz')):
                                file_path = m
                                break
                    if file_path:
                        break

            if file_path is None or not file_path.exists():
                raise FileNotFoundError(f"Cannot find {filename}")

            with fits.open(file_path) as hdul:
                self.image_data = hdul[0].data.astype(float)
                self.header = hdul[0].header
            self.current_filename = filename
            self.xlim_original = None
            self.ylim_original = None
            self._imshow_obj = None
            self._normalized_cache = None
            self.reset_stretch_plot_values()
            self.load_idmatch_for_file(filename)
            self.display_image(full_redraw=True)
            self.update_overlay()
            self.update_master_table()
        except Exception as e:
            import traceback
            self.log(f"[ERROR] Failed to load {filename}: {e}\n{traceback.format_exc()}")
            QMessageBox.critical(self, "Error", f"Failed to load {filename}:\n{str(e)}")

    def load_idmatch_for_file(self, filename):
        """Step 7 idmatch 파일 로드"""
        step6_dir = self.params.P.result_dir / "step7_idmatch"
        legacy_idmatch = legacy_step6_idmatch_dir(self.params.P.result_dir)
        idmatch_path = _resolve_idmatch_path(step6_dir, legacy_idmatch, filename)

        if idmatch_path.exists():
            try:
                df = pd.read_csv(idmatch_path)
                if {"x", "y", "source_id"} <= set(df.columns):
                    self.idmatch_df = df
                    try:
                        sids = pd.to_numeric(df["source_id"], errors="coerce")
                        n_total = int(len(sids))
                        n_gaia = int((sids > 0).sum())
                        n_local = int((sids < 0).sum())
                        self.log(f"IDMatch loaded: {idmatch_path} (total {n_total}, Gaia {n_gaia}, local {n_local})")
                    except Exception:
                        self.log(f"IDMatch loaded: {idmatch_path}")
                    return
            except Exception:
                pass

        self.log(f"IDMatch missing or invalid: {idmatch_path}")
        self.idmatch_df = pd.DataFrame(columns=["x", "y", "source_id"])

    def display_image(self, full_redraw=False):
        if self.image_data is None:
            return

        normalized = self.normalize_image()
        if normalized is None:
            return

        stretched = self.apply_stretch(normalized)

        if self._imshow_obj is not None and not full_redraw:
            self._imshow_obj.set_data(stretched)
            self.canvas.draw_idle()
            return

        xlim_current = self.ax.get_xlim() if self.xlim_original else None
        ylim_current = self.ax.get_ylim() if self.ylim_original else None

        self.ax.clear()
        self._imshow_obj = self.ax.imshow(
            stretched, cmap='gray', origin='lower',
            vmin=0, vmax=1, interpolation='nearest'
        )
        self.ax.set_title(f"{self.current_filename} | {self.current_filter}")
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
        for txt in list(self.ax.texts):
            try:
                txt.remove()
            except Exception:
                pass

        x = self.idmatch_df["x"].to_numpy(float)
        y = self.idmatch_df["y"].to_numpy(float)
        sid_vals = pd.to_numeric(self.idmatch_df["source_id"], errors="coerce")
        sids = sid_vals.fillna(-1).astype("int64").to_numpy()

        in_master = np.array([sid in self.master_ids for sid in sids])
        is_target = (self.target_source_id is not None) & (sids == self.target_source_id)
        is_comp = np.array([sid in self.comparison_ids for sid in sids])

        ref_ids: Set[int] = set()
        gaia_ids: Set[int] = set()
        if self.current_filter in self.filter_catalogs:
            ref_cat = self.filter_catalogs[self.current_filter]
            if "source_id" in ref_cat.columns:
                ref_sid = pd.to_numeric(ref_cat["source_id"], errors="coerce")
                ref_sid = ref_sid.dropna().astype("int64")
                ref_ids = set(ref_sid.tolist())
            g_col = None
            for col in ("gaia_G", "phot_g_mean_mag", "gaia_g"):
                if col in ref_cat.columns:
                    g_col = col
                    break
            if g_col is not None and "source_id" in ref_cat.columns:
                g_vals = pd.to_numeric(ref_cat[g_col], errors="coerce")
                sid_vals2 = pd.to_numeric(ref_cat["source_id"], errors="coerce")
                mask = g_vals.notna() & sid_vals2.notna()
                gaia_ids = set(sid_vals2[mask].astype("int64").tolist())

        is_matched = sids != -1
        is_gaia_matched = np.array([sid in gaia_ids for sid in sids])

        # "Show selected only" 모드: 타겟과 비교성만 표시
        show_selected_only = self.show_selected_only.isChecked()
        def _toggle_enabled(key: str, default: bool = True) -> bool:
            if hasattr(self, "overlay_toggles"):
                cb = self.overlay_toggles.get(key)
                if cb is not None:
                    return cb.isChecked()
            return default

        show_matched_gaia = _toggle_enabled("matched_gaia", True)
        show_matched_no_gaia = _toggle_enabled("matched_no_gaia", True)
        show_ref_only = _toggle_enabled("ref_only", True)
        show_detection_only = _toggle_enabled("detection_only", True)
        show_comp = _toggle_enabled("comparison", True)
        show_target = _toggle_enabled("target", True)

        if len(x):
            matched_gaia = is_matched & is_gaia_matched & ~is_comp & ~is_target
            matched_no_gaia = is_matched & ~is_gaia_matched & ~is_comp & ~is_target
            detection_only = (~is_matched) & ~is_comp & ~is_target

            if not show_selected_only:
                if show_detection_only:
                    # Detection only (unmatched) - orange
                    self.ax.scatter(x[detection_only], y[detection_only], s=20, facecolors='none',
                                    edgecolors='#FF9800', linewidths=0.8, alpha=0.7)

                if show_matched_no_gaia:
                    # Matched (no Gaia) - cyan
                    self.ax.scatter(x[matched_no_gaia], y[matched_no_gaia], s=26, facecolors='none',
                                    edgecolors='#00BCD4', linewidths=1.0, alpha=0.8)

                if show_matched_gaia:
                    # Matched + Gaia - green
                    self.ax.scatter(x[matched_gaia], y[matched_gaia],
                                    s=28, facecolors='none', edgecolors='#4CAF50', linewidths=1.1, alpha=0.85)

            # 비교성 (빨간 원)
            if show_comp:
                self.ax.scatter(x[is_comp], y[is_comp], s=36, facecolors='none',
                                edgecolors='#D32F2F', linewidths=1.5, alpha=0.9, marker='o')
            # 타겟 (빨간 사각형)
            if show_target:
                self.ax.scatter(x[is_target], y[is_target], s=46, facecolors='none',
                                edgecolors='#C62828', linewidths=1.8, alpha=0.95, marker='s')

            label_mask = np.isfinite(x) & np.isfinite(y) & (sids != -1)
            if show_selected_only:
                label_mask &= ((is_comp & show_comp) | (is_target & show_target))
            else:
                label_mask &= (
                    (matched_gaia & show_matched_gaia)
                    | (matched_no_gaia & show_matched_no_gaia)
                    | (is_comp & show_comp)
                    | (is_target & show_target)
                )
            if np.any(label_mask):
                label_offset = 4.0
                label_style = dict(
                    color="#FFD54F",
                    fontsize=8,
                    fontweight="bold",
                    ha="right",
                    va="bottom",
                    alpha=0.98,
                    clip_on=True,
                    zorder=6,
                    path_effects=[pe.withStroke(linewidth=1.4, foreground="#000000")],
                )
                for xi, yi, sid in zip(x[label_mask], y[label_mask], sids[label_mask]):
                    # Use stable_id (1, 2, 3...) instead of source_id (Gaia ID)
                    display_id = self.sid_to_id.get(int(sid), int(sid))
                    self.ax.text(
                        xi - label_offset,
                        yi + label_offset,
                        str(display_id),
                        **label_style,
                    )

        # Ref-only (missing in current frame) - yellow
        if not show_selected_only and ref_ids and show_ref_only:
            matched_ids = set(int(s) for s in sids[sids != -1])
            missing_ids = sorted(ref_ids - matched_ids)
            if missing_ids and self.current_filter in self.filter_catalogs:
                ref_cat = self.filter_catalogs[self.current_filter]
                if {"ra_deg", "dec_deg"} <= set(ref_cat.columns):
                    ref_sub = ref_cat[ref_cat["source_id"].isin(missing_ids)].copy()
                    ra_vals = pd.to_numeric(ref_sub["ra_deg"], errors="coerce")
                    dec_vals = pd.to_numeric(ref_sub["dec_deg"], errors="coerce")
                    mask = ra_vals.notna() & dec_vals.notna()
                    if mask.any():
                        w = self._get_wcs()
                        if w is not None:
                            x_ref, y_ref = w.all_world2pix(
                                ra_vals[mask].to_numpy(float),
                                dec_vals[mask].to_numpy(float),
                                0,
                            )
                            self.ax.scatter(
                                x_ref, y_ref, s=18, facecolors="none",
                                edgecolors="#FFD54F", linewidths=0.9, alpha=0.7
                            )
                            label_offset = 4.0
                            label_style = dict(
                                color="#FFD54F",
                                fontsize=8,
                                fontweight="bold",
                                ha="right",
                                va="bottom",
                                alpha=0.98,
                                clip_on=True,
                                zorder=6,
                                path_effects=[pe.withStroke(linewidth=1.4, foreground="#000000")],
                            )
                            for xi, yi, sid in zip(x_ref, y_ref, ref_sub.loc[mask, "source_id"].astype("int64")):
                                if np.isfinite(xi) and np.isfinite(yi):
                                    # Use stable_id (1, 2, 3...) instead of source_id
                                    display_id = self.sid_to_id.get(int(sid), int(sid))
                                    self.ax.text(
                                        xi - label_offset,
                                        yi + label_offset,
                                        str(display_id),
                                        **label_style,
                                    )

        # 선택된 소스 (현재 클릭한 것)
        if self.selected_source_id is not None:
            sel = self.idmatch_df[self.idmatch_df["source_id"] == self.selected_source_id]
            if len(sel):
                self.ax.scatter(sel["x"], sel["y"], s=70, facecolors='none',
                                edgecolors='red', linewidths=1.8, alpha=0.9)

        self.canvas.draw_idle()

    def update_master_table(self):
        """검출 테이블 업데이트 - idmatch_df의 모든 검출 표시
        컬럼: ID, x, y, G mag, Gaia ID, Status, Role
        """
        if self.idmatch_df is None or self.idmatch_df.empty or "source_id" not in self.idmatch_df.columns:
            self.master_table.setRowCount(0)
            return

        # Gaia G mag 컬럼 확인
        g_col = None
        for col in ["phot_g_mean_mag", "gaia_G", "gaia_g"]:
            if col in self.idmatch_df.columns:
                g_col = col
                break

        rows = []
        for _, row in self.idmatch_df.iterrows():
            sid_val = pd.to_numeric(row["source_id"], errors="coerce")
            if not np.isfinite(sid_val):
                continue
            sid = int(sid_val)
            x_pos = pd.to_numeric(row["x"], errors="coerce")
            y_pos = pd.to_numeric(row["y"], errors="coerce")
            x_pos = float(x_pos) if np.isfinite(x_pos) else float("nan")
            y_pos = float(y_pos) if np.isfinite(y_pos) else float("nan")
            g_val = row.get(g_col, np.nan) if g_col else np.nan
            g_mag = pd.to_numeric(g_val, errors="coerce")
            g_mag = float(g_mag) if np.isfinite(g_mag) else np.nan
            if not np.isfinite(g_mag):
                g_mag = self._get_gmag_for_source(sid)

            # Gaia ID와 Status 결정
            if sid > 0:
                # Truncate long Gaia IDs: show last 8 digits with "..." prefix
                gaia_str = str(sid)
                gaia_id_str = f"...{gaia_str[-8:]}" if len(gaia_str) > 10 else gaia_str
                match_status = "matched"
            else:
                gaia_id_str = "-"
                if not np.isfinite(g_mag):
                    match_status = "no_gaia_in_radius"
                elif g_mag > 20.0:
                    match_status = "too_faint"
                else:
                    match_status = "no_match"

            # Role 결정
            role = ""
            if self.target_source_id is not None and sid == self.target_source_id:
                role = "T"
            elif sid in self.comparison_ids:
                role = "C"

            rows.append((sid, x_pos, y_pos, g_mag, gaia_id_str, match_status, role))

        # Role 우선 (Target, Comp), 그 다음 G mag 순으로 정렬
        def sort_key(r):
            role_order = 0 if r[6] == "T" else (1 if r[6] == "C" else 2)
            g_val = r[3] if np.isfinite(r[3]) else 999
            return (role_order, g_val)

        rows.sort(key=sort_key)

        # ID 매핑 생성 - 안정적인 ID 사용 (ID registry 기반)
        self.sid_to_id = {}
        self.id_to_sid = {}
        if self.current_filter:
            self._load_id_registry(self.current_filter)

        self.master_table.setRowCount(len(rows))
        self._sid_to_row = {}
        for i, (sid, x_pos, y_pos, g_mag, gaia_id_str, match_status, role) in enumerate(rows):
            # 안정적인 ID 사용 (ID registry 기반)
            if self.current_filter and sid in self.master_ids:
                stable_id = self._get_or_assign_stable_id(self.current_filter, sid)
            else:
                stable_id = i + 1  # Fallback for non-master sources
            self.sid_to_id[sid] = stable_id
            self.id_to_sid[stable_id] = sid

            self.master_table.setItem(i, 0, QTableWidgetItem(str(stable_id)))
            self.master_table.setItem(i, 1, QTableWidgetItem(f"{x_pos:.1f}"))
            self.master_table.setItem(i, 2, QTableWidgetItem(f"{y_pos:.1f}"))
            g_str = f"{g_mag:.2f}" if np.isfinite(g_mag) else "-"
            self.master_table.setItem(i, 3, QTableWidgetItem(g_str))
            self.master_table.setItem(i, 4, QTableWidgetItem(gaia_id_str))
            self.master_table.setItem(i, 5, QTableWidgetItem(match_status))
            self.master_table.setItem(i, 6, QTableWidgetItem(role))
            self._sid_to_row[int(sid)] = i
            self._sid_to_row[int(sid)] = i

    def on_table_selection_changed(self):
        rows = self.master_table.selectionModel().selectedRows()
        if rows:
            row_idx = rows[0].row()
            id_item = self.master_table.item(row_idx, 0)  # 내부 ID (column 0)
            if id_item:
                try:
                    internal_id = int(id_item.text())
                    # 내부 ID에서 source_id 가져오기
                    self.selected_source_id = self.id_to_sid.get(internal_id)
                    if self.selected_source_id is not None:
                        status = self.master_table.item(row_idx, 5).text() if self.master_table.item(row_idx, 5) else ""
                        role = self.master_table.item(row_idx, 6).text() if self.master_table.item(row_idx, 6) else ""
                        role_str = f" [{role}]" if role else ""
                        self.selected_label.setText(f"Selected: ID {internal_id}{role_str} ({status})")
                        self.update_overlay()
                except ValueError:
                    pass
        else:
            self.selected_source_id = None

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

        search_r = float(getattr(self.params.P, "search_radius_px", 7.0))

        if self.idmatch_df is not None and not self.idmatch_df.empty:
            dx = self.idmatch_df["x"].to_numpy(float) - x
            dy = self.idmatch_df["y"].to_numpy(float) - y
            dist2 = dx * dx + dy * dy
            valid = np.isfinite(dist2)
            if dist2.size > 0 and np.any(valid):
                i = int(np.argmin(np.where(valid, dist2, np.inf)))
                if dist2[i] <= search_r * search_r:
                    sid_val = pd.to_numeric(self.idmatch_df.iloc[i]["source_id"], errors="coerce")
                    if not np.isfinite(sid_val):
                        return
                    self.selected_source_id = int(sid_val)
                    # Type: Gaia vs Local, display stable ID
                    src_type = "Gaia" if self.selected_source_id > 0 else "Local"
                    display_id = self.sid_to_id.get(self.selected_source_id, "?")
                    flags = []
                    if self.selected_source_id in self.master_ids:
                        flags.append("M")
                    if self.selected_source_id in self.catalog_ids:
                        flags.append("C")
                    flag_text = f" [{','.join(flags)}]" if flags else ""
                    self.selected_label.setText(
                        f"Selected: ID {display_id} ({src_type}){flag_text} - "
                        "Press T=target, C=comp, A/D=master"
                    )
                    self._select_table_row_by_sid(self.selected_source_id)
                    self.update_overlay()
                    return

        self.selected_source_id = None
        self.selected_label.setText(f"No detection at ({x:.1f}, {y:.1f})")
        self.master_table.clearSelection()
        self.update_overlay()

    def update_target_labels(self):
        if self.target_source_id is None:
            self.target_label.setText("Target: (none)")
        else:
            # Show stable ID (1, 2, 3...) with source type
            display_id = self.sid_to_id.get(self.target_source_id, "?")
            src_type = "Gaia" if self.target_source_id > 0 else "Local"
            self.target_label.setText(f"Target: ID {display_id} ({src_type})")
        # Show comparison count with IDs if few
        n_comp = len(self.comparison_ids)
        if n_comp <= 5 and n_comp > 0:
            comp_ids = [str(self.sid_to_id.get(sid, "?")) for sid in sorted(self.comparison_ids)]
            self.comparison_label.setText(f"Comparisons: {', '.join(comp_ids)}")
        else:
            self.comparison_label.setText(f"Comparisons: {n_comp}")

    def _select_table_row_by_sid(self, source_id: int):
        if source_id is None:
            return
        row = getattr(self, "_sid_to_row", {}).get(int(source_id))
        if row is None:
            return
        self.master_table.blockSignals(True)
        self.master_table.selectRow(row)
        self.master_table.blockSignals(False)

    def _drop_from_selection(self, sids: Set[int]):
        if not sids:
            return
        changed = False
        if self.target_source_id in sids:
            self.target_source_id = None
            self.filter_targets[self.current_filter] = None
            changed = True
        if self.comparison_ids & sids:
            self.comparison_ids -= sids
            self.filter_comparisons[self.current_filter] = self.comparison_ids.copy()
            changed = True
        if changed:
            self.update_target_labels()
            self.save_selection()

    def _add_to_master(self, sids: Set[int], reason: str = "added") -> Set[int]:
        if not self.current_filter:
            return set()
        added = set()
        for sid in sids:
            sid = int(sid)
            if sid not in self.master_ids:
                self.master_ids.add(sid)
                added.add(sid)
        if added:
            self.filter_master_ids[self.current_filter] = self.master_ids
            self.save_master_catalog(log_action=reason)
            self.update_master_table()
            self.update_overlay()
        return added

    def _remove_from_master(self, sids: Set[int], reason: str = "removed") -> Set[int]:
        if not self.current_filter:
            return set()
        to_remove = set(int(sid) for sid in sids if int(sid) in self.master_ids)
        if not to_remove:
            return set()
        self.master_ids -= to_remove
        self.filter_master_ids[self.current_filter] = self.master_ids
        self._drop_from_selection(to_remove)
        # Retire IDs for removed sources (IDs will never be reused)
        for sid in to_remove:
            self._retire_stable_id(self.current_filter, sid)
        self._save_id_registry(self.current_filter)
        self.save_master_catalog(log_action=reason)
        self.update_master_table()
        self.update_overlay()
        return to_remove

    def add_master_selected(self):
        if self.selected_source_id is None:
            QMessageBox.information(self, "Master", "Select a detected source first.")
            return
        sid = int(self.selected_source_id)
        if sid in self.master_ids:
            self.log(f"Master add skipped (already in master): {sid}")
            return
        added = self._add_to_master({sid}, reason="added")
        if added:
            self.log(f"Master add: {sid}")

    def remove_master_selected(self):
        if self.selected_source_id is None:
            QMessageBox.information(self, "Master", "Select a source first.")
            return
        sid = int(self.selected_source_id)
        if sid not in self.master_ids:
            self.log(f"Master remove skipped (not in master): {sid}")
            return
        removed = self._remove_from_master({sid}, reason="removed")
        if removed:
            self.log(f"Master remove: {sid}")

    def remove_master_box(self):
        if self.last_click_xy is None or self.idmatch_df is None or self.idmatch_df.empty:
            self.log("Master box remove skipped (no position or detections).")
            return
        x0, y0 = self.last_click_xy
        box = int(getattr(self.params.P, "bulk_drop_box_px", 200))
        half = box / 2.0
        df = self.idmatch_df
        in_box = df["x"].between(x0 - half, x0 + half) & df["y"].between(y0 - half, y0 + half)
        sids = set(df.loc[in_box, "source_id"].astype("int64").tolist())
        removed = self._remove_from_master(sids, reason="box_removed")
        if removed:
            self.log(f"Master box remove: {len(removed)} sources ({box}x{box}px)")

    def set_target_selected(self):
        if self.selected_source_id is None:
            QMessageBox.information(self, "Target", "Select a source first.")
            return
        sid = int(self.selected_source_id)

        if sid not in self.master_ids:
            self._add_to_master({sid}, reason="auto_add_target")

        if self.target_source_id == sid:
            # 토글 해제
            self.target_source_id = None
            self.filter_targets[self.current_filter] = None
        else:
            self.target_source_id = sid
            self.filter_targets[self.current_filter] = sid
            self.comparison_ids.discard(sid)
            self.filter_comparisons[self.current_filter] = self.comparison_ids.copy()

        self.update_target_labels()
        self.save_selection()
        self.update_master_table()
        self.update_overlay()
        self.log(f"Target set: {self.target_source_id}")

    def toggle_comparison_selected(self):
        if self.selected_source_id is None:
            QMessageBox.information(self, "Comparison", "Select a source first.")
            return
        sid = int(self.selected_source_id)

        if self.target_source_id == sid:
            QMessageBox.information(self, "Comparison", "Target cannot be a comparison star.")
            return

        if sid not in self.master_ids:
            self._add_to_master({sid}, reason="auto_add_comp")

        if sid in self.comparison_ids:
            self.comparison_ids.remove(sid)
            action = "removed"
        else:
            self.comparison_ids.add(sid)
            action = "added"

        self.filter_comparisons[self.current_filter] = self.comparison_ids.copy()
        self.update_target_labels()
        self.save_selection()
        self.update_master_table()
        self.update_overlay()
        self.log(f"Comparison {action}: {sid}")

    def auto_select_comparisons(self):
        """비교성 자동 선택: 밝은 등급 + 비슷한 색지수 우선"""
        if self.target_source_id is None:
            QMessageBox.information(self, "Comparison", "Set a target first.")
            return

        # Get target info
        target_mag = self._get_gmag_for_source(self.target_source_id)
        target_color = self._get_color_for_source(self.target_source_id)

        # Reference catalog이 있는 경우 더 많은 후보 확보
        cat = None
        if self.current_filter and self.current_filter in self.filter_catalogs:
            cat = self.filter_catalogs[self.current_filter].copy()

        # 후보 수집: idmatch_df 또는 catalog에서
        candidates = []
        seen_sids = set()

        # idmatch_df에서 후보 수집
        if self.idmatch_df is not None and not self.idmatch_df.empty:
            for _, row in self.idmatch_df.iterrows():
                sid_val = pd.to_numeric(row.get("source_id"), errors="coerce")
                if not np.isfinite(sid_val):
                    continue
                sid = int(sid_val)
                if sid == self.target_source_id or sid in seen_sids:
                    continue
                seen_sids.add(sid)

                g_mag = self._get_gmag_for_source(sid)
                color_val = self._get_color_for_source(sid)

                if np.isfinite(g_mag):
                    candidates.append({
                        "source_id": sid,
                        "g_mag": g_mag,
                        "color": color_val,
                        "x": row.get("x", np.nan),
                        "y": row.get("y", np.nan),
                    })

        # catalog에서 추가 후보 수집
        if cat is not None and "source_id" in cat.columns:
            g_col = None
            for col in ("gaia_G", "gaia_g", "phot_g_mean_mag"):
                if col in cat.columns:
                    g_col = col
                    break
            c_col = None
            for col in ("color_gr", "bp_rp", "BP_RP"):
                if col in cat.columns:
                    c_col = col
                    break

            for _, row in cat.iterrows():
                sid_val = pd.to_numeric(row.get("source_id"), errors="coerce")
                if not np.isfinite(sid_val):
                    continue
                sid = int(sid_val)
                if sid == self.target_source_id or sid in seen_sids:
                    continue
                seen_sids.add(sid)

                g_mag = row[g_col] if g_col and g_col in row else np.nan
                g_mag = pd.to_numeric(g_mag, errors="coerce")
                if not np.isfinite(g_mag):
                    g_mag = self._get_gmag_for_source(sid)

                color_val = row[c_col] if c_col and c_col in row else np.nan
                color_val = pd.to_numeric(color_val, errors="coerce")
                if not np.isfinite(color_val):
                    color_val = self._get_color_for_source(sid)

                if np.isfinite(g_mag):
                    candidates.append({
                        "source_id": sid,
                        "g_mag": g_mag,
                        "color": color_val,
                        "x": row.get("x_mean", row.get("x", np.nan)),
                        "y": row.get("y_mean", row.get("y", np.nan)),
                    })

        if len(candidates) == 0:
            QMessageBox.information(self, "Comparison", "No valid candidates with magnitude information.")
            return

        cand_df = pd.DataFrame(candidates)

        # 추천 전략: 밝은 등급 + 비슷한 색지수 우선
        # 1. 색지수가 있는 타겟인 경우: 색지수가 비슷한 것 중 밝은 별
        # 2. 색지수가 없는 타겟인 경우: 밝은 별 우선

        color_tol = 0.5  # BP-RP 허용 차이 (mag)
        n_pick = int(self.comp_reco_count.value())

        if np.isfinite(target_color):
            # 색지수 차이 계산
            cand_df["d_color"] = np.abs(cand_df["color"] - target_color)
            # 색지수가 비슷한 것 중 밝은 별 (낮은 g_mag)
            cand_df["has_similar_color"] = cand_df["d_color"] <= color_tol
            # 점수: 색지수가 비슷하면 0, 아니면 100, 그 다음 밝기 순
            cand_df["score"] = (~cand_df["has_similar_color"]).astype(int) * 100 + cand_df["g_mag"]
            cand_df = cand_df.sort_values("score")

            # 결과 로깅
            similar_color_count = cand_df["has_similar_color"].sum()
            self.log(f"Target: G={target_mag:.2f}, BP-RP={target_color:.2f}")
            self.log(f"Found {similar_color_count} candidates with similar color (|d| <= {color_tol})")
        else:
            # 색지수 없는 경우: 밝은 별 우선
            cand_df = cand_df.sort_values("g_mag")
            self.log(f"Target: G={target_mag:.2f}, color unavailable - selecting brightest stars")

        # 상위 N개 선택
        picked = cand_df.head(n_pick)

        # 결과 출력
        self.log("Recommended comparisons:")
        for _, row in picked.iterrows():
            sid = int(row["source_id"])
            g = row["g_mag"]
            c = row["color"]
            if np.isfinite(target_color) and np.isfinite(c):
                d_c = abs(c - target_color)
                self.log(f"  ID={sid}: G={g:.2f}, BP-RP={c:.2f}, dColor={d_c:.2f}")
            elif np.isfinite(c):
                self.log(f"  ID={sid}: G={g:.2f}, BP-RP={c:.2f}")
            else:
                self.log(f"  ID={sid}: G={g:.2f}, BP-RP=-")

        # 선택 적용
        picked_sids = set(int(sid) for sid in picked["source_id"])
        self.comparison_ids.update(picked_sids)
        self._add_to_master(picked_sids, reason="auto_add_comp")

        self.filter_comparisons[self.current_filter] = self.comparison_ids.copy()
        self.update_target_labels()
        self.save_selection()
        self.update_master_table()
        self.update_overlay()

        # 결과 메시지
        msg = f"Added {len(picked)} comparison stars:\n\n"
        for _, row in picked.iterrows():
            sid = int(row["source_id"])
            g = row["g_mag"]
            c = row["color"]
            if np.isfinite(c):
                msg += f"  ID {sid}: G={g:.2f}, BP-RP={c:.2f}\n"
            else:
                msg += f"  ID {sid}: G={g:.2f}\n"

        QMessageBox.information(self, "Comparison", msg)

    def clear_comparisons(self):
        if not self.comparison_ids:
            return

        self.comparison_ids.clear()
        self.filter_comparisons[self.current_filter] = set()
        self.update_target_labels()
        self.save_selection()
        self.update_master_table()
        self.update_overlay()
        self.log("Comparisons cleared")

    def copy_selection_to_all_filters(self):
        """현재 필터의 타겟/비교성 선택을 모든 필터에 복사"""
        if self.target_source_id is None:
            QMessageBox.information(self, "Copy Selection", "Set a target first in current filter.")
            return

        if not self.current_filter:
            return

        current_target = self.target_source_id
        current_comps = self.comparison_ids.copy()

        copied_filters = []

        # filter_frames 기준으로 복사 (filter_catalogs 의존성 제거)
        for flt in self.filter_frames.keys():
            if flt == self.current_filter:
                continue

            # 동일 source_id가 다른 필터에서도 유효하다고 가정 (WCS 기반 매칭)
            self.filter_targets[flt] = current_target
            self.filter_comparisons[flt] = current_comps.copy()

            # 파일 저장
            self._save_selection_for_filter(flt, current_target, current_comps)
            copied_filters.append(f"{flt} ({len(current_comps)} comps)")

        # 결과 메시지
        msg = f"Selection copied from '{self.current_filter}':\n\n"
        if copied_filters:
            msg += f"Copied to: {', '.join(copied_filters)}"

        self.log(f"Copy selection: {len(copied_filters)} filters updated")
        QMessageBox.information(self, "Copy Selection", msg)

    def _save_selection_for_filter(self, flt: str, target_sid: int, comp_sids: set):
        """특정 필터의 선택 저장 (catalog 없이 동작)"""
        step8_dir = self.params.P.result_dir / "step8_selection"
        step8_dir.mkdir(parents=True, exist_ok=True)

        # Load master catalog to map source_id -> ID (final ID)
        master_path = step8_dir / f"master_catalog_{flt}.tsv"
        source_to_id = {}
        if master_path.exists():
            try:
                mc = pd.read_csv(master_path, sep="\t")
                if "source_id" in mc.columns and "ID" in mc.columns:
                    source_to_id = dict(zip(mc["source_id"].astype(int), mc["ID"].astype(int)))
            except Exception:
                pass

        # Map source_ids to final IDs
        target_id = source_to_id.get(int(target_sid)) if target_sid else None
        comp_ids = sorted([source_to_id.get(int(sid)) for sid in comp_sids if source_to_id.get(int(sid)) is not None])

        data = {
            "filter": flt,
            "target_id": target_id,
            "target_source_id": int(target_sid),
            "comparison_ids": comp_ids,
            "comparison_source_ids": sorted([int(sid) for sid in comp_sids]),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        selection_path = step8_dir / f"selection_{flt}.json"
        with open(selection_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Also ensure master_ids for this filter includes the selection
        all_sids = comp_sids.copy()
        if target_sid is not None:
            all_sids.add(int(target_sid))
        if flt not in self.filter_master_ids:
            self.filter_master_ids[flt] = set()
        self.filter_master_ids[flt].update(all_sids)

        # Save master catalog for this filter
        self.save_master_catalog(flt=flt, log_action="copy_selection")

        self.log(f"  Saved selection for {flt}: target={target_sid}, {len(comp_sids)} comps")

    def select_target_from_simbad(self):
        """SIMBAD 좌표로 타겟 선택 (Step 1에서 입력한 대상 좌표 사용)"""
        # Step 1에서 저장한 좌표를 params.P에서 직접 읽기
        ra_deg = getattr(self.params.P, "target_ra_deg", None)
        dec_deg = getattr(self.params.P, "target_dec_deg", None)

        if ra_deg is None or dec_deg is None:
            QMessageBox.information(
                self, "Target",
                "No target coordinates found.\n"
                "Enter target name in Step 1 first."
            )
            return

        try:
            ra_deg = float(ra_deg)
            dec_deg = float(dec_deg)
        except (TypeError, ValueError):
            QMessageBox.information(self, "Target", "Invalid target coordinates.")
            return

        if not (np.isfinite(ra_deg) and np.isfinite(dec_deg)):
            QMessageBox.information(self, "Target", "Target coordinates are not valid.")
            return

        # idmatch_df에서 가장 가까운 검출 찾기 (Gaia catalog 대신)
        if self.idmatch_df is None or self.idmatch_df.empty:
            QMessageBox.information(self, "Target", "No detections loaded. Select a frame first.")
            return

        # RA/Dec 컬럼 확인
        ra_col = None
        dec_col = None
        for rc in ["ra_gaia", "ra", "RA"]:
            if rc in self.idmatch_df.columns:
                ra_col = rc
                break
        for dc in ["dec_gaia", "dec", "DEC"]:
            if dc in self.idmatch_df.columns:
                dec_col = dc
                break

        if ra_col is None or dec_col is None:
            if {"x", "y"} <= set(self.idmatch_df.columns) and self.header is not None:
                try:
                    w = WCS(self.header, relax=True)
                except Exception:
                    w = None
                if w is not None and w.has_celestial:
                    xy = self.idmatch_df[["x", "y"]].to_numpy(float)
                    ra_vals, dec_vals = w.all_pix2world(xy[:, 0], xy[:, 1], 0)
                    self.idmatch_df["ra"] = ra_vals
                    self.idmatch_df["dec"] = dec_vals
                    ra_col = "ra"
                    dec_col = "dec"
            if ra_col is None or dec_col is None:
                QMessageBox.information(self, "Target", "No RA/Dec columns in detection data.")
                return

        try:
            sc_target = SkyCoord(ra_deg * u.deg, dec_deg * u.deg, frame="icrs")

            # 유효한 좌표만 필터
            df = self.idmatch_df.copy()
            valid_mask = df[ra_col].notna() & df[dec_col].notna()
            df_valid = df[valid_mask]

            if df_valid.empty:
                QMessageBox.information(self, "Target", "No sources with valid coordinates.")
                return

            sc_det = SkyCoord(
                df_valid[ra_col].to_numpy(float) * u.deg,
                df_valid[dec_col].to_numpy(float) * u.deg,
                frame="icrs"
            )
            idx, sep2d, _ = sc_target.match_to_catalog_sky(sc_det)
            sep_arcsec = float(np.atleast_1d(sep2d.arcsec)[0])

            max_sep = float(getattr(self.params.P, "gaia_add_max_sep_arcsec", 5.0))
            if sep_arcsec > max_sep:
                QMessageBox.information(
                    self, "Target",
                    f"No detection within {max_sep}\" of target.\n"
                    f"Closest: {sep_arcsec:.1f}\" away."
                )
                return

            sid = int(df_valid.iloc[int(idx)]["source_id"])

            if sid not in self.master_ids:
                self._add_to_master({sid}, reason="auto_add_target")

            self.target_source_id = sid
            self.filter_targets[self.current_filter] = sid
            self.comparison_ids.discard(sid)
            self.filter_comparisons[self.current_filter] = self.comparison_ids.copy()

            self.update_target_labels()
            self.save_selection()
            self.update_master_table()
            self.update_overlay()

            # 대상 이름 가져오기
            target_name = getattr(self.params.P, "target_name", "Unknown")
            self.log(f"Target '{target_name}' matched: source_id={sid} (sep={sep_arcsec:.2f}\")")
            self._select_table_row_by_sid(sid)

            QMessageBox.information(
                self, "Target Set",
                f"Target '{target_name}' matched to source_id {sid}\n"
                f"Separation: {sep_arcsec:.2f} arcsec"
            )

        except Exception as e:
            QMessageBox.warning(self, "Target", f"Failed: {e}")

    def save_selection(self):
        """현재 필터의 선택 저장"""
        if not self.current_filter:
            return

        step8_dir = self.params.P.result_dir / "step8_selection"
        step8_dir.mkdir(parents=True, exist_ok=True)

        # Load master catalog to map source_id -> ID (final ID)
        master_path = step8_dir / f"master_catalog_{self.current_filter}.tsv"
        source_to_id = {}
        if master_path.exists():
            try:
                mc = pd.read_csv(master_path, sep="\t")
                if "source_id" in mc.columns and "ID" in mc.columns:
                    source_to_id = dict(zip(mc["source_id"].astype(int), mc["ID"].astype(int)))
            except Exception:
                pass

        # Map source_ids to final IDs
        target_id = source_to_id.get(int(self.target_source_id)) if self.target_source_id else None
        comp_ids = sorted([source_to_id.get(int(sid)) for sid in self.comparison_ids if source_to_id.get(int(sid)) is not None])

        # 필터별 저장 (catalog 없이도 동작)
        data = {
            "filter": self.current_filter,
            "target_id": target_id,
            "target_source_id": int(self.target_source_id) if self.target_source_id is not None else None,
            "comparison_ids": comp_ids,
            "comparison_source_ids": sorted([int(sid) for sid in self.comparison_ids]),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        selection_path = step8_dir / f"selection_{self.current_filter}.json"
        with open(selection_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        # Save selection summary inside Step 8 directory
        self._save_legacy_selection()

        # Ensure selected IDs are in master_ids
        all_sids = self.comparison_ids.copy()
        if self.target_source_id is not None:
            all_sids.add(int(self.target_source_id))
        if self.current_filter not in self.filter_master_ids:
            self.filter_master_ids[self.current_filter] = set()
        self.filter_master_ids[self.current_filter].update(all_sids)
        self.master_ids = self.filter_master_ids[self.current_filter]

        # Always save/update master catalog when selection changes
        self.save_master_catalog(log_action="selection_update")

        self.save_state()
        self.update_navigation_buttons()

    def _save_legacy_selection(self):
        """Save target_selection.json inside Step 8 directory."""
        try:
            all_targets = {}
            all_comps = {}

            for flt, target_sid in self.filter_targets.items():
                if target_sid is not None:
                    all_targets[flt] = target_sid

            for flt, comp_sids in self.filter_comparisons.items():
                if comp_sids:
                    all_comps[flt] = sorted(comp_sids)

            # 첫 번째 필터 기준으로 legacy 형식 (filter_frames 기준)
            first_filter = list(self.filter_frames.keys())[0] if self.filter_frames else None

            data = {
                "target_source_id": self.filter_targets.get(first_filter) if first_filter else None,
                "comparison_source_ids": sorted(self.filter_comparisons.get(first_filter, set())) if first_filter else [],
                "filter_targets": all_targets,
                "filter_comparisons": {k: sorted(v) for k, v in all_comps.items()}
            }

            step8_dir = self.params.P.result_dir / "step8_selection"
            step8_dir.mkdir(parents=True, exist_ok=True)
            legacy_path = step8_dir / "target_selection.json"
            with open(legacy_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            self.log(f"Legacy save failed: {e}")

    def validate_step(self) -> bool:
        step8_dir = self.params.P.result_dir / "step8_selection"
        if not step8_dir.exists():
            return False

        # 적어도 하나의 필터에 타겟이 설정되어야 함 (filter_frames 기준)
        for flt in self.filter_frames.keys():
            selection_path = step8_dir / f"selection_{flt}.json"
            if selection_path.exists():
                try:
                    with open(selection_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if data.get("target_source_id") is not None:
                        return True
                except Exception:
                    pass

        return False

    def save_state(self):
        state_data = {
            "current_filter": self.current_filter,
            "filter_targets": {k: v for k, v in self.filter_targets.items() if v is not None},
            "filter_comparisons": {k: sorted(v) for k, v in self.filter_comparisons.items() if v},
            "show_selected_only": self.show_selected_only.isChecked()
        }
        self.project_state.store_step_data("target_selection", state_data)

    def restore_state(self):
        state_data = self.project_state.get_step_data("target_selection")
        if state_data:
            saved_filter = state_data.get("current_filter")
            if saved_filter and saved_filter in [self.filter_combo.itemText(i) for i in range(self.filter_combo.count())]:
                idx = self.filter_combo.findText(saved_filter)
                if idx >= 0:
                    self.filter_combo.setCurrentIndex(idx)

            # 체크박스 상태 복원
            if "show_selected_only" in state_data:
                self.show_selected_only.setChecked(bool(state_data["show_selected_only"]))

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

    def cycle_filter(self):
        if self.filter_combo.count() <= 1:
            return
        current_idx = self.filter_combo.currentIndex()
        if current_idx < 0:
            return
        frame_idx = self.file_combo.currentIndex()
        self._pending_frame_index = frame_idx
        next_idx = (current_idx + 1) % self.filter_combo.count()
        self.filter_combo.setCurrentIndex(next_idx)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_A:
            self.add_master_selected()
            return
        if event.key() == Qt.Key_D:
            if event.modifiers() & Qt.ShiftModifier:
                self.remove_master_box()
            else:
                self.remove_master_selected()
            return
        if event.key() == Qt.Key_T:
            self.set_target_selected()
            return
        if event.key() == Qt.Key_C:
            self.toggle_comparison_selected()
            return
        if event.key() == Qt.Key_G:
            self.show_radial_profile()
            return
        if event.key() == Qt.Key_Period:
            self.cycle_filter()
            return
        if event.key() == Qt.Key_BracketLeft or event.key() == Qt.Key_Comma:
            self.step_frame(-1)
            return
        if event.key() == Qt.Key_BracketRight:
            self.step_frame(1)
            return
        super().keyPressEvent(event)

    def show_radial_profile(self):
        if self.image_data is None:
            return

        if self.hover_xy is not None:
            x, y = self.hover_xy
        elif self.last_click_xy is not None:
            x, y = self.last_click_xy
        else:
            return

        # (간략화된 프로파일 표시)
        self.log(f"Radial profile at ({x:.1f}, {y:.1f})")

    # === Zoom/Pan/Stretch 함수들 ===

    def reset_zoom(self):
        if self.xlim_original is not None:
            self.ax.set_xlim(self.xlim_original)
            self.ax.set_ylim(self.ylim_original)
            self.canvas.draw_idle()

    def on_stretch_changed(self, index):
        self._normalized_cache = None
        self.reset_stretch_plot_values()
        self.display_image()

    def redisplay_image(self):
        self.display_image()

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

    def normalize_image(self):
        if self.image_data is None:
            return None

        finite = np.isfinite(self.image_data)
        if not finite.any():
            return np.zeros_like(self.image_data)

        data = self.image_data.copy()
        mean_val, median_val, std_val = sigma_clipped_stats(data[finite], sigma=3.0, maxiters=5)
        vmin = max(np.min(data[finite]), median_val - 2.8 * std_val)
        vmax = min(np.max(data[finite]), np.percentile(data[finite], 99.9))

        if vmax <= vmin:
            vmin = np.min(data[finite])
            vmax = np.max(data[finite])

        normalized = (data - vmin) / (vmax - vmin + 1e-10)
        return np.clip(normalized, 0, 1)

    def apply_stretch(self, data):
        stretch_idx = self.scale_combo.currentIndex()
        intensity = self.stretch_slider.value() / 100.0
        black_point = self.black_slider.value() / 100.0

        data = np.clip((data - black_point) / (1.0 - black_point + 1e-10), 0, 1)

        if stretch_idx == 0:  # Auto
            return self._stretch_auto(data, intensity)
        if stretch_idx == 1:  # Asinh
            beta = 1.0 + intensity * 15.0
            return np.clip(np.arcsinh(data * beta) / np.arcsinh(beta), 0, 1)
        if stretch_idx == 2:  # MTF
            m = 0.05 + (1.0 - intensity) * 0.45
            return self._mtf(data, m)
        if stretch_idx == 3:  # Histogram Eq
            return self._histogram_eq(data)
        if stretch_idx == 4:  # Log
            a = 100 + intensity * 900
            return np.clip(np.log(1 + a * data) / np.log(1 + a), 0, 1)
        if stretch_idx == 5:  # Sqrt
            power = 0.2 + (1.0 - intensity) * 0.8
            return np.clip(np.power(data, power), 0, 1)
        return data

    def _stretch_auto(self, data, intensity):
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
        return self._mtf(stretched, midtone)

    def _mtf(self, data, m):
        m = np.clip(m, 0.001, 0.999)
        result = np.zeros_like(data)
        mask = (data > 0) & (data < 1)
        result[mask] = (m - 1) * data[mask] / ((2 * m - 1) * data[mask] - m)
        result[data >= 1] = 1
        return np.clip(result, 0, 1)

    def _histogram_eq(self, data):
        finite = data[np.isfinite(data)]
        if len(finite) == 0:
            return data
        hist, bin_edges = np.histogram(finite.flatten(), bins=65536, range=(0, 1))
        cdf = hist.cumsum()
        cdf = cdf / cdf[-1]
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        return np.clip(np.interp(data, bin_centers, cdf), 0, 1)

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
