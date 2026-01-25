"""
Step 12: Detrend & Night Merge
"""

from __future__ import annotations

from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
from astropy.time import Time

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QGroupBox,
    QLineEdit,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QFileDialog,
    QTextEdit,
    QWidget,
    QFormLayout,
    QDoubleSpinBox,
    QSpinBox,
    QCheckBox,
    QComboBox,
    QListWidget,
    QListWidgetItem,
    QRadioButton,
    QButtonGroup,
    QTabWidget,
    QSplitter,
    QFrame,
    QSizePolicy,
)
from PyQt5.QtCore import Qt

from .step_window_base import StepWindowBase
from ...analysis.light_curve.global_ensemble import solve_global_ensemble
from ...utils.step_paths import step1_dir, step9_dir, step11_dir, step12_dir, step11_zeropoint_dir


def _safe_float(value, default: float = np.nan) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _fmt_float(value, default: str = "") -> str:
    try:
        if value is None:
            return default
        v = float(value)
        if not np.isfinite(v):
            return default
        return f"{v:.5f}"
    except Exception:
        return default


def _normalize_filter_key(value: str | None) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    if text in ("", "nan", "none"):
        return ""
    return text


def _parse_color_expr(expr: str | None) -> tuple[str, str] | None:
    if not expr:
        return None
    s = str(expr).strip().lower().replace(" ", "")
    if "-" in s:
        parts = [p for p in s.split("-") if p]
    elif "_" in s:
        parts = [p for p in s.split("_") if p]
    else:
        return None
    if len(parts) != 2:
        return None
    return parts[0], parts[1]


def _parse_jd(date_obs: str | None) -> float:
    if not date_obs:
        return np.nan
    try:
        return float(Time(str(date_obs).strip()).jd)
    except Exception:
        return np.nan


def _load_headers_table(result_dir: Path) -> pd.DataFrame:
    headers_path = step1_dir(result_dir) / "headers.csv"
    if not headers_path.exists():
        headers_path = result_dir / "headers.csv"
    if not headers_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(headers_path)
    except Exception:
        return pd.DataFrame()


class DetrendNightMergeWindow(StepWindowBase):
    """Step 12: Nightly detrend + merge"""

    def __init__(self, params, file_manager, project_state, main_window):
        self.file_manager = file_manager
        self.datasets: list[tuple[str, Path]] = []
        self.raw_df = pd.DataFrame()
        self.corrected_df = pd.DataFrame()
        self.params_df = pd.DataFrame()
        self.global_mean_df = pd.DataFrame()
        self.global_diagnostics: dict = {}
        self.global_input_df = pd.DataFrame()

        self.comp_active_ids: list[int] = []
        self.comp_candidate_ids: list[int] = []
        self.color_map_by_filter: dict[str, str] = {}
        self.color_by = "Date"
        self.mode = "offset"  # "offset" | "color" | "global"
        self.sigma_clip = True
        self.clip_sigma = 3.0
        self.clip_iters = 2
        self.x_axis_mode = "time"
        self.phase_cycles = 2.0  # 기본 2주기 표시

        self.delta_c_map: dict[str, float] = {}

        self.time_masks: list[tuple[float, float]] = []
        self.phase_masks: list[tuple[float, float]] = []
        self.phase_period = 0.0
        self.phase_t0 = 0.0

        # Global ensemble (method C) params
        self.global_min_comps = 3
        self.global_sigma = 3.0
        self.global_iters = 3
        self.global_rms_pct = 20.0
        self.global_rms_threshold = 0.0
        self.global_frame_sigma = 3.0
        self.global_gauge = "meanZ0"
        self.global_robust = True
        self.global_interp_missing = False
        self.global_normalize = False

        super().__init__(
            step_index=11,
            step_name="Detrend & Night Merge",
            params=params,
            project_state=project_state,
            main_window=main_window,
        )
        self.setup_step_ui()
        self.restore_state()

    def setup_step_ui(self):
        # Main horizontal splitter
        main_splitter = QSplitter(Qt.Horizontal)
        self.content_layout.addWidget(main_splitter, 1)

        # ===== LEFT PANEL =====
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        # Info banner (compact)
        info = QLabel("여러 밤의 차등측광 데이터를 보정하여 병합합니다.")
        info.setStyleSheet(
            "QLabel { background-color: #E3F2FD; padding: 6px; border-radius: 4px; font-size: 9pt; }"
        )
        left_layout.addWidget(info)

        # Left tabs
        self.left_tabs = QTabWidget()
        self.left_tabs.setStyleSheet("QTabWidget::pane { border: 1px solid #ccc; }")
        left_layout.addWidget(self.left_tabs, 1)

        # ----- Tab 1: Data & Target -----
        data_tab = QWidget()
        data_layout = QVBoxLayout(data_tab)
        data_layout.setSpacing(8)

        # Dataset section (compact)
        dataset_group = QGroupBox("데이터셋")
        dataset_layout = QVBoxLayout(dataset_group)
        dataset_layout.setSpacing(4)

        self.dataset_table = QTableWidget()
        self.dataset_table.setColumnCount(2)
        self.dataset_table.setHorizontalHeaderLabels(["라벨", "경로"])
        self.dataset_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.dataset_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.dataset_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.dataset_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.dataset_table.setMaximumHeight(100)
        dataset_layout.addWidget(self.dataset_table)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)
        btn_add = QPushButton("폴더 추가")
        btn_add.clicked.connect(self.add_dataset)
        btn_row.addWidget(btn_add)
        btn_remove = QPushButton("삭제")
        btn_remove.clicked.connect(self.remove_selected_dataset)
        btn_row.addWidget(btn_remove)
        btn_use_current = QPushButton("현재 result 사용")
        btn_use_current.clicked.connect(self.use_current_dataset)
        btn_row.addWidget(btn_use_current)
        dataset_layout.addLayout(btn_row)
        data_layout.addWidget(dataset_group)

        # Target section (compact)
        target_group = QGroupBox("대상 설정")
        target_layout = QFormLayout(target_group)
        target_layout.setSpacing(4)

        target_row = QHBoxLayout()
        self.target_edit = QLineEdit()
        self.target_edit.setMaximumWidth(80)
        self.target_edit.setPlaceholderText("ID")
        target_row.addWidget(self.target_edit)
        btn_load = QPushButton("Step11에서 로드")
        btn_load.clicked.connect(self.load_from_step11)
        target_row.addWidget(btn_load)
        btn_load_raw = QPushButton("Raw 로드")
        btn_load_raw.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; }")
        btn_load_raw.clicked.connect(self.load_raw_data)
        target_row.addWidget(btn_load_raw)
        target_layout.addRow("Target ID:", target_row)

        self.comp_label = QLabel("Comp: -")
        self.comp_label.setStyleSheet("QLabel { color: #455A64; font-size: 9pt; }")
        self.comp_label.setWordWrap(True)
        target_layout.addRow(self.comp_label)
        data_layout.addWidget(target_group)

        # Date & Filter selection
        selection_row = QHBoxLayout()

        date_group = QGroupBox("날짜")
        date_layout = QVBoxLayout(date_group)
        date_layout.setContentsMargins(4, 4, 4, 4)
        self.date_list = QListWidget()
        self.date_list.setMaximumHeight(120)
        self.date_list.itemChanged.connect(self._on_date_selection_changed)
        date_layout.addWidget(self.date_list)
        selection_row.addWidget(date_group)

        filter_group = QGroupBox("필터 / 색상")
        filter_layout = QFormLayout(filter_group)
        filter_layout.setSpacing(4)
        self.filter_combo = QComboBox()
        self.filter_combo.addItem("All")
        self.filter_combo.currentIndexChanged.connect(self._on_filter_changed)
        filter_layout.addRow("필터:", self.filter_combo)
        self.color_by_combo = QComboBox()
        self.color_by_combo.addItems(["Date", "Filter"])
        self.color_by_combo.currentIndexChanged.connect(self._on_color_by_changed)
        filter_layout.addRow("Color by:", self.color_by_combo)
        selection_row.addWidget(filter_group)

        data_layout.addLayout(selection_row)
        data_layout.addStretch()
        self.left_tabs.addTab(data_tab, "데이터")

        # ----- Tab 2: Correction Mode -----
        mode_tab = QWidget()
        mode_layout = QVBoxLayout(mode_tab)
        mode_layout.setSpacing(8)

        # Analysis feedback panel
        self.analysis_panel = QGroupBox("데이터 분석 결과")
        self.analysis_panel.setStyleSheet(
            "QGroupBox { font-weight: bold; } "
            "QGroupBox::title { color: #1565C0; }"
        )
        analysis_layout = QVBoxLayout(self.analysis_panel)
        analysis_layout.setSpacing(4)

        self.analysis_text = QLabel("데이터를 로드하면 분석 결과가 표시됩니다.")
        self.analysis_text.setWordWrap(True)
        self.analysis_text.setStyleSheet(
            "QLabel { background-color: #FAFAFA; padding: 8px; border-radius: 4px; font-size: 9pt; }"
        )
        analysis_layout.addWidget(self.analysis_text)

        self.recommendation_label = QLabel("")
        self.recommendation_label.setWordWrap(True)
        self.recommendation_label.setStyleSheet(
            "QLabel { background-color: #E8F5E9; padding: 8px; border-radius: 4px; font-size: 9pt; font-weight: bold; }"
        )
        analysis_layout.addWidget(self.recommendation_label)
        mode_layout.addWidget(self.analysis_panel)

        # Mode selection
        mode_group = QGroupBox("보정 모드 선택")
        mode_group_layout = QVBoxLayout(mode_group)
        mode_group_layout.setSpacing(6)

        self.mode_group = QButtonGroup(self)

        # Offset mode
        offset_frame = QFrame()
        offset_frame.setStyleSheet("QFrame { border: 1px solid #E0E0E0; border-radius: 4px; padding: 4px; }")
        offset_layout = QVBoxLayout(offset_frame)
        offset_layout.setSpacing(2)
        self.mode_offset = QRadioButton("Offset Only (ZP₀)")
        self.mode_offset.setStyleSheet("QRadioButton { font-weight: bold; }")
        offset_layout.addWidget(self.mode_offset)
        offset_desc = QLabel(
            "Δm_corr = Δm_raw - ZP₀\n"
            "• 밤별 영점 오프셋만 보정\n"
            "• Target-Comp 색차가 작을 때 (|ΔC| < 0.3)"
        )
        offset_desc.setStyleSheet("QLabel { color: #616161; font-size: 8pt; margin-left: 16px; }")
        offset_layout.addWidget(offset_desc)
        mode_group_layout.addWidget(offset_frame)

        # Color mode
        color_frame = QFrame()
        color_frame.setStyleSheet("QFrame { border: 1px solid #E0E0E0; border-radius: 4px; padding: 4px; }")
        color_layout = QVBoxLayout(color_frame)
        color_layout.setSpacing(2)
        self.mode_color = QRadioButton("Color-dependent (ZP₀ + k''·ΔC·X)")
        self.mode_color.setStyleSheet("QRadioButton { font-weight: bold; }")
        color_layout.addWidget(self.mode_color)
        color_desc = QLabel(
            "Δm_corr = Δm_raw - ZP₀ - k''·ΔC·X\n"
            "• 2차 소광계수(k'')로 색차 보정\n"
            "• Target-Comp 색차가 클 때 (|ΔC| ≥ 0.3)"
        )
        color_desc.setStyleSheet("QLabel { color: #616161; font-size: 8pt; margin-left: 16px; }")
        color_layout.addWidget(color_desc)

        self.chk_global_k2 = QCheckBox("Global k'' (전체 데이터로 k'' 한 번 피팅)")
        self.chk_global_k2.setStyleSheet("QCheckBox { margin-left: 16px; }")
        self.chk_global_k2.setChecked(True)
        color_layout.addWidget(self.chk_global_k2)
        mode_group_layout.addWidget(color_frame)

        # Global ensemble mode (method C)
        global_frame = QFrame()
        global_frame.setStyleSheet("QFrame { border: 1px solid #E0E0E0; border-radius: 4px; padding: 4px; }")
        global_layout = QVBoxLayout(global_frame)
        global_layout.setSpacing(2)
        self.mode_global = QRadioButton("Global Ensemble (Method C)")
        self.mode_global.setStyleSheet("QRadioButton { font-weight: bold; }")
        global_layout.addWidget(self.mode_global)
        global_desc = QLabel(
            "Δm_corr = m_target - Z_t\n"
            "• 프레임별 Z_t를 전역 최소제곱으로 동시 추정\n"
            "• 밤 경계/기준선 자동 정렬"
        )
        global_desc.setStyleSheet("QLabel { color: #616161; font-size: 8pt; margin-left: 16px; }")
        global_layout.addWidget(global_desc)
        mode_group_layout.addWidget(global_frame)

        self.mode_group.addButton(self.mode_offset)
        self.mode_group.addButton(self.mode_color)
        self.mode_group.addButton(self.mode_global)
        self.mode_offset.setChecked(True)
        self.mode_offset.toggled.connect(lambda: self._set_mode("offset"))
        self.mode_color.toggled.connect(lambda: self._set_mode("color"))
        self.mode_global.toggled.connect(lambda: self._set_mode("global"))

        self.color_status_label = QLabel("")
        self.color_status_label.setStyleSheet("QLabel { color: #D32F2F; font-size: 9pt; }")
        self.color_status_label.setWordWrap(True)
        mode_group_layout.addWidget(self.color_status_label)

        mode_layout.addWidget(mode_group)

        # Color index mapping (compact)
        self.color_map_group = QGroupBox("Color Index (ΔC) 설정")
        self.color_map_layout = QFormLayout(self.color_map_group)
        self.color_map_layout.setSpacing(4)
        self.color_map_combos = {}
        mode_layout.addWidget(self.color_map_group)

        mode_layout.addStretch()
        self.left_tabs.addTab(mode_tab, "보정 모드")

        # ----- Tab 3: Phase & Options -----
        options_tab = QWidget()
        options_layout = QVBoxLayout(options_tab)
        options_layout.setSpacing(8)

        # Phase folding
        phase_group = QGroupBox("Phase Folding")
        phase_layout = QFormLayout(phase_group)
        phase_layout.setSpacing(4)

        self.phase_mode_combo = QComboBox()
        self.phase_mode_combo.addItems(["Time (JD)", "Phase"])
        self.phase_mode_combo.currentIndexChanged.connect(self._on_xaxis_changed)
        phase_layout.addRow("X축:", self.phase_mode_combo)

        self.spin_period = QDoubleSpinBox()
        self.spin_period.setDecimals(6)
        self.spin_period.setRange(0.0, 1000.0)
        self.spin_period.valueChanged.connect(self._on_phase_params_changed)
        phase_layout.addRow("주기 (days):", self.spin_period)

        t0_row = QHBoxLayout()
        self.spin_t0 = QDoubleSpinBox()
        self.spin_t0.setDecimals(6)
        self.spin_t0.setRange(0.0, 3000000.0)
        self.spin_t0.setToolTip("기준 시점 (예: 주극소 JD). 0이면 min(JD) 자동 사용")
        self.spin_t0.valueChanged.connect(self._on_phase_params_changed)
        t0_row.addWidget(self.spin_t0)
        btn_t0_auto = QPushButton("Auto")
        btn_t0_auto.setMaximumWidth(50)
        btn_t0_auto.setToolTip("데이터의 min(JD)로 설정")
        btn_t0_auto.clicked.connect(self._auto_set_t0)
        t0_row.addWidget(btn_t0_auto)
        phase_layout.addRow("T₀ (JD):", t0_row)

        t0_note = QLabel("※ T₀ = 기준시점 (주극소/극대). 0이면 min(JD) 사용")
        t0_note.setStyleSheet("QLabel { color: #757575; font-size: 8pt; }")
        phase_layout.addRow("", t0_note)

        self.spin_cycles = QDoubleSpinBox()
        self.spin_cycles.setDecimals(2)
        self.spin_cycles.setRange(1.0, 5.0)
        self.spin_cycles.setSingleStep(0.5)
        self.spin_cycles.setValue(2.0)  # 기본 2주기
        self.spin_cycles.valueChanged.connect(self._on_phase_params_changed)
        phase_layout.addRow("표시 주기:", self.spin_cycles)
        options_layout.addWidget(phase_group)

        # Sigma clipping
        clip_group = QGroupBox("Sigma Clipping (이상치 제거)")
        clip_layout = QVBoxLayout(clip_group)
        clip_layout.setSpacing(4)

        clip_desc = QLabel(
            "피팅 시 잔차가 Nσ 이상인 이상치를 반복 제거합니다.\n"
            "(구름, 장비 오류 등으로 인한 outlier 제거용)"
        )
        clip_desc.setStyleSheet("QLabel { color: #616161; font-size: 8pt; }")
        clip_desc.setWordWrap(True)
        clip_layout.addWidget(clip_desc)

        clip_form = QFormLayout()
        clip_form.setSpacing(4)

        self.chk_clip = QCheckBox()
        self.chk_clip.setChecked(True)
        self.chk_clip.stateChanged.connect(self._on_clip_changed)
        clip_form.addRow("활성화:", self.chk_clip)

        self.spin_clip = QDoubleSpinBox()
        self.spin_clip.setDecimals(1)
        self.spin_clip.setRange(1.0, 10.0)
        self.spin_clip.setValue(self.clip_sigma)
        self.spin_clip.setToolTip("이 값 × σ 이상의 잔차를 가진 점을 제거")
        self.spin_clip.valueChanged.connect(self._on_clip_changed)
        clip_form.addRow("σ 임계값:", self.spin_clip)

        self.spin_iters = QSpinBox()
        self.spin_iters.setRange(1, 5)
        self.spin_iters.setValue(self.clip_iters)
        self.spin_iters.setToolTip("클리핑 후 재피팅 반복 횟수")
        self.spin_iters.valueChanged.connect(self._on_clip_changed)
        clip_form.addRow("반복 횟수:", self.spin_iters)

        clip_layout.addLayout(clip_form)
        options_layout.addWidget(clip_group)

        # Global ensemble parameters
        global_group = QGroupBox("Global Ensemble (Method C)")
        global_layout = QFormLayout(global_group)
        global_layout.setSpacing(4)

        self.spin_global_min_comps = QSpinBox()
        self.spin_global_min_comps.setRange(1, 50)
        self.spin_global_min_comps.setValue(self.global_min_comps)
        global_layout.addRow("최소 비교성 수:", self.spin_global_min_comps)

        self.spin_global_sigma = QDoubleSpinBox()
        self.spin_global_sigma.setRange(1.0, 10.0)
        self.spin_global_sigma.setDecimals(1)
        self.spin_global_sigma.setValue(self.global_sigma)
        global_layout.addRow("σ 클립:", self.spin_global_sigma)

        self.spin_global_iters = QSpinBox()
        self.spin_global_iters.setRange(1, 5)
        self.spin_global_iters.setValue(self.global_iters)
        global_layout.addRow("반복 횟수:", self.spin_global_iters)

        self.spin_global_rms_pct = QDoubleSpinBox()
        self.spin_global_rms_pct.setRange(0.0, 80.0)
        self.spin_global_rms_pct.setDecimals(1)
        self.spin_global_rms_pct.setValue(self.global_rms_pct)
        global_layout.addRow("RMS 상위% 제거:", self.spin_global_rms_pct)

        self.spin_global_rms_thr = QDoubleSpinBox()
        self.spin_global_rms_thr.setRange(0.0, 1.0)
        self.spin_global_rms_thr.setDecimals(4)
        self.spin_global_rms_thr.setValue(self.global_rms_threshold)
        self.spin_global_rms_thr.setToolTip("0이면 비활성")
        global_layout.addRow("RMS 절대컷:", self.spin_global_rms_thr)

        self.spin_global_frame_sigma = QDoubleSpinBox()
        self.spin_global_frame_sigma.setRange(1.0, 10.0)
        self.spin_global_frame_sigma.setDecimals(1)
        self.spin_global_frame_sigma.setValue(self.global_frame_sigma)
        global_layout.addRow("프레임 σ:", self.spin_global_frame_sigma)

        self.combo_global_gauge = QComboBox()
        self.combo_global_gauge.addItems(["meanZ0", "ref"])
        self.combo_global_gauge.setCurrentText(self.global_gauge)
        global_layout.addRow("Gauge:", self.combo_global_gauge)

        self.chk_global_robust = QCheckBox("Robust (MAD)")
        self.chk_global_robust.setChecked(self.global_robust)
        global_layout.addRow(self.chk_global_robust)

        self.chk_global_interp = QCheckBox("Z_t 보간 (부족 프레임)")
        self.chk_global_interp.setChecked(self.global_interp_missing)
        global_layout.addRow(self.chk_global_interp)

        self.chk_global_normalize = QCheckBox("Target 중앙값 0으로 정규화")
        self.chk_global_normalize.setChecked(self.global_normalize)
        global_layout.addRow(self.chk_global_normalize)

        options_layout.addWidget(global_group)

        options_layout.addStretch()
        self.left_tabs.addTab(options_tab, "옵션")

        # ----- Tab 4: Log -----
        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("QTextEdit { font-family: monospace; font-size: 9pt; }")
        log_layout.addWidget(self.log_text)
        self.left_tabs.addTab(log_tab, "로그")

        # Action buttons (at bottom of left panel)
        btn_group = QFrame()
        btn_group.setStyleSheet("QFrame { background-color: #ECEFF1; border-radius: 4px; padding: 4px; }")
        btn_layout = QHBoxLayout(btn_group)
        btn_layout.setSpacing(8)

        btn_apply = QPushButton("Fit && Apply (저장)")
        btn_apply.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 6px 16px; }"
        )
        btn_apply.setToolTip("피팅 수행 후 종합 결과 파일 자동 저장")
        btn_apply.clicked.connect(self.fit_and_apply)
        btn_layout.addWidget(btn_apply)

        btn_revert = QPushButton("Revert")
        btn_revert.setToolTip("보정 결과 초기화")
        btn_revert.clicked.connect(self.revert_raw)
        btn_layout.addWidget(btn_revert)

        btn_layout.addStretch()
        left_layout.addWidget(btn_group)

        main_splitter.addWidget(left_widget)

        # ===== RIGHT PANEL =====
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)

        # Plot canvas
        self.plot_canvas = FigureCanvas(Figure(figsize=(9, 7)))
        self.ax_raw = self.plot_canvas.figure.add_subplot(311)
        self.ax_corr = self.plot_canvas.figure.add_subplot(312)
        self.ax_diag = self.plot_canvas.figure.add_subplot(313)
        right_layout.addWidget(self.plot_canvas, 3)

        # Results table
        result_group = QGroupBox("피팅 결과")
        result_layout = QVBoxLayout(result_group)
        result_layout.setContentsMargins(4, 4, 4, 4)
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(9)
        self.result_table.setHorizontalHeaderLabels(
            ["Date", "Filter", "N", "ZP₀", "±σ(ZP)", "k''", "±σ(k'')", "RMS전", "RMS후"]
        )
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.result_table.setMaximumHeight(150)
        result_layout.addWidget(self.result_table)
        right_layout.addWidget(result_group, 1)

        main_splitter.addWidget(right_widget)
        main_splitter.setSizes([320, 680])

    def log(self, msg: str):
        self.log_text.append(msg)

    def _update_analysis_panel(self):
        """Update analysis panel with data statistics and recommendations."""
        if self.raw_df.empty:
            self.analysis_text.setText("데이터를 로드하면 분석 결과가 표시됩니다.")
            self.recommendation_label.setText("")
            self.recommendation_label.setVisible(False)
            return

        if self.mode == "global":
            n_points = len(self.raw_df)
            n_dates = self.raw_df["date"].nunique() if "date" in self.raw_df.columns else 0
            filters = sorted({_normalize_filter_key(f) for f in self.raw_df.get("filter", []) if str(f).strip()})
            self.analysis_text.setText(
                f"<b>Global Ensemble:</b> {n_points}점, {n_dates}일, 필터: {', '.join(filters) or 'N/A'}"
            )
            self.recommendation_label.setText("")
            self.recommendation_label.setVisible(False)
            return

        lines = []

        # 1. Data summary
        n_points = len(self.raw_df)
        n_dates = self.raw_df["date"].nunique() if "date" in self.raw_df.columns else 0
        filters = sorted({_normalize_filter_key(f) for f in self.raw_df.get("filter", []) if str(f).strip()})
        lines.append(f"<b>데이터:</b> {n_points}점, {n_dates}일, 필터: {', '.join(filters) or 'N/A'}")

        # 2. Airmass range
        airmass = self.raw_df["airmass"].to_numpy(float)
        airmass = airmass[np.isfinite(airmass)]
        if airmass.size > 0:
            x_min, x_max = np.min(airmass), np.max(airmass)
            x_range = x_max - x_min
            lines.append(f"<b>Airmass:</b> {x_min:.3f} ~ {x_max:.3f} (ΔX = {x_range:.3f})")
        else:
            x_range = 0.0
            lines.append("<b>Airmass:</b> 데이터 없음")

        # 3. Color index difference (ΔC)
        delta_c_values = []
        delta_c_info = []
        for fkey, dc in self.delta_c_map.items():
            if np.isfinite(dc):
                delta_c_values.append(abs(dc))
                delta_c_info.append(f"{fkey or 'all'}: {dc:+.3f}")

        if delta_c_values:
            max_dc = max(delta_c_values)
            lines.append(f"<b>|ΔC| (Target-Comp):</b> {', '.join(delta_c_info)}")
        else:
            max_dc = 0.0
            lines.append("<b>|ΔC|:</b> 계산 불가 (color index 설정 필요)")

        # 4. Expected color term effect
        if delta_c_values and airmass.size > 0:
            # Typical k'' ~ 0.02-0.05 mag/mag for broad-band filters
            k2_typical = 0.03
            effect_max = k2_typical * max_dc * x_range
            lines.append(f"<b>예상 색항 효과:</b> ~{effect_max:.4f} mag (k''≈0.03 가정)")

        # 5. Raw scatter
        raw_mag = self.raw_df["diff_mag_raw"].to_numpy(float)
        raw_mag = raw_mag[np.isfinite(raw_mag)]
        if raw_mag.size > 0:
            raw_rms = np.std(raw_mag)
            lines.append(f"<b>Raw RMS:</b> {raw_rms:.4f} mag")

        self.analysis_text.setText("<br>".join(lines))

        # Generate recommendation
        recommendation = self._generate_recommendation(max_dc, x_range, n_dates)
        if recommendation:
            self.recommendation_label.setText(recommendation)
            self.recommendation_label.setVisible(True)
        else:
            self.recommendation_label.setVisible(False)

    def _generate_recommendation(self, max_dc: float, x_range: float, n_dates: int) -> str:
        """Generate mode recommendation based on data characteristics."""
        reasons = []

        # Decision logic based on astronomical data science principles
        use_color_mode = False
        color_mode_warning = False

        # 1. Check airmass range FIRST - critical for k'' fitting stability
        airmass_sufficient = x_range >= 0.3

        # 2. Check color index difference
        if max_dc >= 0.5:
            reasons.append(f"|ΔC| = {max_dc:.2f} ≥ 0.5: 색차가 매우 큼")
            if airmass_sufficient:
                use_color_mode = True
            else:
                color_mode_warning = True
                reasons.append(f"  → 단, ΔX = {x_range:.2f} < 0.3: k'' 피팅 불안정 우려")
        elif max_dc >= 0.3:
            reasons.append(f"|ΔC| = {max_dc:.2f} ≥ 0.3: 색차가 상당함")
            if airmass_sufficient:
                use_color_mode = True
            else:
                color_mode_warning = True
        elif max_dc > 0:
            reasons.append(f"|ΔC| = {max_dc:.2f} < 0.3: 색차가 작음")

        # 3. Check airmass range details
        if x_range >= 0.5:
            reasons.append(f"ΔX = {x_range:.2f} ≥ 0.5: airmass 범위 충분")
        elif x_range >= 0.3:
            reasons.append(f"ΔX = {x_range:.2f}: airmass 범위 적절")
        else:
            reasons.append(f"ΔX = {x_range:.2f} < 0.3: airmass 범위 좁음 (k'' 피팅 불안정)")
            use_color_mode = False  # Override - not enough airmass range

        # 4. Multi-night consideration
        if n_dates > 1:
            reasons.append(f"{n_dates}일 데이터: 밤별 ZP₀ 보정 필수")

        # Build recommendation message
        if use_color_mode:
            mode_text = "⚠️ <b>Color 모드 권장</b>"
            style = "background-color: #FFF3E0; color: #E65100;"
        elif color_mode_warning:
            mode_text = "⚠️ <b>Offset 모드 권장</b> (색차 있으나 ΔX 부족)"
            style = "background-color: #FFF8E1; color: #F57C00;"
        else:
            mode_text = "✓ <b>Offset 모드 적합</b>"
            style = "background-color: #E8F5E9; color: #2E7D32;"

        self.recommendation_label.setStyleSheet(
            f"QLabel {{ {style} padding: 8px; border-radius: 4px; font-size: 9pt; }}"
        )

        return f"{mode_text}<br>{'<br>'.join('• ' + r for r in reasons)}"

    def add_dataset(self):
        path = QFileDialog.getExistingDirectory(self, "result_dir 선택", str(Path.cwd()))
        if not path:
            return
        self._add_dataset(Path(path))

    def use_current_dataset(self):
        self._add_dataset(self.params.P.result_dir)

    def _add_dataset(self, path: Path):
        path = Path(path)
        if not path.exists():
            return
        label = path.name
        for _, p in self.datasets:
            if p == path:
                return
        self.datasets.append((label, path))
        self.refresh_dataset_table()

    def remove_selected_dataset(self):
        rows = self.dataset_table.selectionModel().selectedRows()
        if not rows:
            return
        idx = rows[0].row()
        if 0 <= idx < len(self.datasets):
            self.datasets.pop(idx)
        self.refresh_dataset_table()

    def refresh_dataset_table(self):
        self.dataset_table.setRowCount(0)
        for label, path in self.datasets:
            r = self.dataset_table.rowCount()
            self.dataset_table.insertRow(r)
            self.dataset_table.setItem(r, 0, QTableWidgetItem(label))
            self.dataset_table.setItem(r, 1, QTableWidgetItem(str(path)))

    def load_from_step11(self):
        if not self.datasets:
            self.use_current_dataset()
        if not self.datasets:
            return
        base_dir = step11_dir(self.params.P.result_dir)
        sel_path = base_dir / "comp_selection.json"
        if not sel_path.exists():
            base_dir = step11_dir(self.datasets[0][1])
            sel_path = base_dir / "comp_selection.json"
        if not sel_path.exists():
            raw_paths = list(base_dir.glob("lightcurve_ID*_raw.csv"))
            if not raw_paths and self.datasets:
                raw_paths = list(step11_dir(self.datasets[0][1]).glob("lightcurve_ID*_raw.csv"))
            if raw_paths:
                target_id = self._parse_target_id_from_name(raw_paths[0].name)
                if target_id is not None:
                    self.target_edit.setText(str(target_id))
                    self.log(f"[LOAD] Target ID from raw filename: {target_id}")
            return
        try:
            data = json.loads(sel_path.read_text(encoding="utf-8"))
            target_id = data.get("target_id")
            if target_id is not None:
                self.target_edit.setText(str(target_id))
            self.comp_active_ids = [int(x) for x in data.get("comp_active_ids", []) if str(x).strip()]
            self.comp_candidate_ids = [int(x) for x in data.get("comp_candidate_ids", []) if str(x).strip()]
            self._update_comp_label()
        except Exception:
            return

    def _parse_target_id_from_name(self, name: str) -> int | None:
        m = re.search(r"lightcurve_ID(\d+)_raw\.csv", name)
        if not m:
            return None
        try:
            return int(m.group(1))
        except Exception:
            return None

    def load_raw_data(self):
        if not self.datasets:
            self.use_current_dataset()
        if not self.datasets:
            QMessageBox.information(self, "Detrend", "데이터셋이 없습니다.")
            return
        target_text = self.target_edit.text().strip()
        if not target_text:
            base_dir = step11_dir(self.params.P.result_dir)
            raw_paths = list(base_dir.glob("lightcurve_ID*_raw.csv"))
            if not raw_paths and self.datasets:
                raw_paths = list(step11_dir(self.datasets[0][1]).glob("lightcurve_ID*_raw.csv"))
            if raw_paths:
                target_id = self._parse_target_id_from_name(raw_paths[0].name)
                if target_id is not None:
                    self.target_edit.setText(str(target_id))
                    target_text = str(target_id)
                    self.log(f"[LOAD] Target ID from raw filename: {target_id}")
        if not target_text:
            QMessageBox.information(self, "Detrend", "대상 ID가 필요합니다.")
            return
        target_id = int(target_text)

        raw_frames = []
        for label, result_dir in self.datasets:
            result_dir = Path(result_dir)
            step11_path = step11_dir(result_dir)

            # Try multiple possible filenames
            candidates = [
                step11_path / f"lightcurve_ID{target_id}_raw.csv",
                step11_path / f"lightcurve_combined_ID{target_id}_raw.csv",
            ]
            raw_path = None
            for cand in candidates:
                if cand.exists():
                    raw_path = cand
                    break

            if raw_path is None:
                self.log(f"[LOAD] Missing raw in {step11_path}")
                self.log(f"  Tried: lightcurve_ID{target_id}_raw.csv, lightcurve_combined_ID{target_id}_raw.csv")
                # List available files
                if step11_path.exists():
                    available = list(step11_path.glob("lightcurve_*.csv"))
                    if available:
                        self.log(f"  Available: {[f.name for f in available[:5]]}")
                continue

            try:
                df = pd.read_csv(raw_path)
                self.log(f"[LOAD] Loaded: {raw_path.name} ({len(df)} rows)")
            except Exception as e:
                self.log(f"[LOAD] Failed to read {raw_path.name}: {e}")
                continue
            df = df.copy()
            df["dataset"] = label
            raw_frames.append(df)

        if not raw_frames:
            step11_path = step11_dir(self.params.P.result_dir)
            msg = f"Raw 데이터를 찾지 못했습니다.\n\n경로: {step11_path}\n\nStep 11에서 먼저 'Build Light Curve'를 실행하세요."
            QMessageBox.information(self, "Detrend", msg)
            return

        self.raw_df = pd.concat(raw_frames, ignore_index=True)
        if "diff_mag_raw" not in self.raw_df.columns and "diff_mag" in self.raw_df.columns:
            self.raw_df["diff_mag_raw"] = self.raw_df["diff_mag"]
        if "diff_mag_raw" not in self.raw_df.columns:
            self.raw_df["diff_mag_raw"] = np.nan
        if "diff_err" not in self.raw_df.columns:
            self.raw_df["diff_err"] = np.nan
        self.raw_df["diff_mag_raw"] = pd.to_numeric(self.raw_df["diff_mag_raw"], errors="coerce")
        self.raw_df["airmass"] = pd.to_numeric(self.raw_df.get("airmass", np.nan), errors="coerce")
        self.raw_df["JD"] = pd.to_numeric(self.raw_df.get("JD", np.nan), errors="coerce")
        if "date" not in self.raw_df.columns:
            self.raw_df["date"] = "unknown"
        self._fill_date_from_jd()
        self._fill_airmass_from_headers()

        self._load_comp_selection()
        self._populate_date_list()
        self._refresh_filter_combo(self.raw_df.get("filter", pd.Series([], dtype=str)).astype(str).tolist())
        self._rebuild_color_map_controls(self.raw_df.get("filter", pd.Series([], dtype=str)).astype(str).tolist())
        self._update_plots()
        self.log(f"[LOAD] Raw points: {len(self.raw_df)}")

        self._refresh_delta_c_map()
        self._log_color_index_info()
        self._update_color_mode_enabled()
        self._update_analysis_panel()

    def _load_global_ensemble_df(self) -> pd.DataFrame:
        if not self.datasets:
            self.use_current_dataset()
        if not self.datasets:
            raise RuntimeError("No datasets available")

        target_text = self.target_edit.text().strip()
        if not target_text:
            raise RuntimeError("Target ID is required")
        target_id = int(target_text)

        if not self.comp_active_ids and not self.comp_candidate_ids:
            self._load_comp_selection()
        comp_ids = self.comp_active_ids or self.comp_candidate_ids
        comp_ids = [int(c) for c in comp_ids if str(c).strip() and int(c) != target_id]
        if not comp_ids:
            raise RuntimeError("Comparison IDs not found")

        rows = []
        for label, result_dir in self.datasets:
            result_dir = Path(result_dir)
            idx_path = step9_dir(result_dir) / "photometry_index.csv"
            if not idx_path.exists():
                idx_path = result_dir / "photometry_index.csv"
            if not idx_path.exists():
                self.log(f"[GLOBAL] photometry_index.csv missing: {result_dir}")
                continue

            try:
                idx = pd.read_csv(idx_path)
            except Exception as e:
                self.log(f"[GLOBAL] Failed to read index: {e}")
                continue
            if "file" not in idx.columns:
                self.log("[GLOBAL] photometry_index.csv missing 'file'")
                continue

            headers_df = _load_headers_table(result_dir)
            jd_map = {}
            filt_map = {}
            if not headers_df.empty and "Filename" in headers_df.columns:
                if "JD" in headers_df.columns:
                    jd_map = dict(
                        zip(
                            headers_df["Filename"].astype(str),
                            pd.to_numeric(headers_df["JD"], errors="coerce"),
                        )
                    )
                if "DATE-OBS" in headers_df.columns:
                    jd_map = dict(
                        zip(
                            headers_df["Filename"].astype(str),
                            headers_df["DATE-OBS"].astype(str),
                        )
                    )
                for col in ("FILTER", "filter"):
                    if col in headers_df.columns:
                        filt_map = dict(zip(headers_df["Filename"].astype(str), headers_df[col].astype(str)))
                        break

            for _, row in idx.iterrows():
                fname = str(row.get("file", "")).strip()
                if not fname:
                    continue
                rel_path = str(row.get("path", "") or "").strip()
                if rel_path:
                    tsv_path = step9_dir(result_dir) / rel_path
                else:
                    tsv_path = step9_dir(result_dir) / f"{fname}_photometry.tsv"
                if not tsv_path.exists():
                    tsv_path = result_dir / tsv_path.name
                if not tsv_path.exists():
                    continue

                try:
                    phot = pd.read_csv(tsv_path, sep="\t")
                except Exception:
                    phot = pd.read_csv(tsv_path)

                id_col = None
                for cand in ("ID", "id"):
                    if cand in phot.columns:
                        id_col = cand
                        break
                if not id_col:
                    continue

                phot[id_col] = pd.to_numeric(phot[id_col], errors="coerce").astype("Int64")
                wanted = [target_id] + comp_ids
                phot = phot[phot[id_col].isin(wanted)].copy()
                if phot.empty:
                    continue

                filt_val = ""
                if "FILTER" in phot.columns:
                    filt_val = str(phot["FILTER"].iloc[0]).strip()
                if not filt_val:
                    filt_val = str(row.get("filter", "") or "")
                if not filt_val and fname in filt_map:
                    filt_val = str(filt_map.get(fname, "") or "")
                filt_key = _normalize_filter_key(filt_val)

                date_obs = jd_map.get(fname)
                jd_val = _safe_float(date_obs)
                if not np.isfinite(jd_val):
                    jd_val = _parse_jd(date_obs) if date_obs else np.nan

                for _, r in phot.iterrows():
                    sid = int(r[id_col]) if pd.notna(r[id_col]) else None
                    if sid is None:
                        continue
                    mag = _safe_float(r.get("mag"))
                    err = _safe_float(r.get("mag_err"))
                    if not np.isfinite(mag):
                        continue
                    time_id = f"{label}:{fname}" if label else fname
                    rows.append(
                        dict(
                            time_id=time_id,
                            jd=jd_val,
                            filter=filt_key,
                            star_id=sid,
                            mag_inst=mag,
                            err=err,
                            file=fname,
                            dataset=label,
                        )
                    )

        if not rows:
            raise RuntimeError("No photometry rows found for global ensemble")
        return pd.DataFrame(rows)

    def _fill_date_from_jd(self) -> None:
        if self.raw_df.empty or "JD" not in self.raw_df.columns:
            return
        date_col = self.raw_df.get("date", pd.Series(["unknown"] * len(self.raw_df))).astype(str)
        jd = self.raw_df["JD"].to_numpy(float)
        fill_mask = np.isfinite(jd) & date_col.astype(str).str.strip().str.lower().isin(["", "nan", "none", "unknown"])
        if not np.any(fill_mask):
            self._fill_date_from_filename()
            return
        try:
            times = Time(jd[fill_mask], format="jd").to_datetime()
            date_vals = [t.strftime("%Y-%m-%d") for t in times]
            date_col = date_col.to_numpy(object)
            date_col[fill_mask] = date_vals
            self.raw_df["date"] = date_col
            self._fill_date_from_filename()
        except Exception:
            return

    def _fill_date_from_filename(self) -> None:
        if self.raw_df.empty or "file" not in self.raw_df.columns:
            return
        date_col = self.raw_df.get("date", pd.Series(["unknown"] * len(self.raw_df))).astype(str).to_numpy(object)
        files = self.raw_df["file"].astype(str).tolist()
        for i, fname in enumerate(files):
            if str(date_col[i]).strip().lower() not in ["", "nan", "none", "unknown"]:
                continue
            m = re.search(r"(\d{4}-\d{2}-\d{2})", fname)
            if m:
                date_col[i] = m.group(1)
                continue
            m = re.search(r"(\d{8})", fname)
            if m:
                d = m.group(1)
                date_col[i] = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        self.raw_df["date"] = date_col

    def _fill_airmass_from_headers(self) -> None:
        if self.raw_df.empty or "file" not in self.raw_df.columns:
            return
        missing_mask = self.raw_df["airmass"].isna()
        if missing_mask.sum() == 0:
            return
        dataset_map = {label: Path(path) for label, path in self.datasets} if self.datasets else {}
        filled = 0
        total_missing = int(missing_mask.sum())
        if not dataset_map:
            dataset_map = {"": Path(self.params.P.result_dir)}
            self.raw_df["dataset"] = self.raw_df.get("dataset", "")
        for label, result_dir in dataset_map.items():
            if "dataset" in self.raw_df.columns:
                sel = (self.raw_df["dataset"].astype(str) == str(label)) & self.raw_df["airmass"].isna()
            else:
                sel = self.raw_df["airmass"].isna()
            if not np.any(sel):
                continue
            headers_path = step1_dir(result_dir) / "headers.csv"
            if not headers_path.exists():
                headers_path = Path(result_dir) / "headers.csv"
            if not headers_path.exists():
                continue
            try:
                hdf = pd.read_csv(headers_path)
            except Exception:
                continue
            if "Filename" not in hdf.columns:
                continue
            col_air = None
            for cand in ("AIRMASS", "airmass"):
                if cand in hdf.columns:
                    col_air = cand
                    break
            if col_air is None:
                continue
            amap = dict(zip(hdf["Filename"].astype(str), pd.to_numeric(hdf[col_air], errors="coerce")))
            files = self.raw_df.loc[sel, "file"].astype(str)
            vals = files.map(amap)
            n_fill = int(vals.notna().sum())
            if n_fill == 0:
                continue
            self.raw_df.loc[sel, "airmass"] = vals
            filled += n_fill
        if filled > 0:
            self.log(f"[LOAD] Filled airmass from headers: {filled}/{total_missing}")

    def _update_comp_label(self) -> None:
        if self.comp_active_ids:
            ids_text = ", ".join(str(i) for i in self.comp_active_ids[:5])
            if len(self.comp_active_ids) > 5:
                ids_text += f" +{len(self.comp_active_ids) - 5}"
            self.comp_label.setText(f"Comp ({len(self.comp_active_ids)}): {ids_text}")
        elif self.comp_candidate_ids:
            ids_text = ", ".join(str(i) for i in self.comp_candidate_ids[:5])
            if len(self.comp_candidate_ids) > 5:
                ids_text += f" +{len(self.comp_candidate_ids) - 5}"
            self.comp_label.setText(f"Comp candidates ({len(self.comp_candidate_ids)}): {ids_text}")
        else:
            self.comp_label.setText("Comp: -")

    def _load_comp_selection(self) -> None:
        if not self.datasets:
            return
        base_dir = step11_dir(self.params.P.result_dir)
        sel_path = base_dir / "comp_selection.json"
        if not sel_path.exists() and self.datasets:
            sel_path = step11_dir(self.datasets[0][1]) / "comp_selection.json"
        if not sel_path.exists():
            return
        try:
            data = json.loads(sel_path.read_text(encoding="utf-8"))
            self.comp_active_ids = [int(x) for x in data.get("comp_active_ids", []) if str(x).strip()]
            self.comp_candidate_ids = [int(x) for x in data.get("comp_candidate_ids", []) if str(x).strip()]
            self._update_comp_label()
        except Exception:
            return

    def _compute_delta_c_map(self, df: pd.DataFrame) -> dict[str, float]:
        if df.empty:
            return {}
        if "color_index" not in df.columns or "color_index_ref" not in df.columns:
            return {}
        delta = pd.to_numeric(df["color_index"], errors="coerce") - pd.to_numeric(
            df["color_index_ref"], errors="coerce"
        )
        if "filter" in df.columns:
            filters = df["filter"].astype(str).map(_normalize_filter_key)
        else:
            filters = pd.Series([""] * len(df))
        out: dict[str, float] = {}
        temp = pd.DataFrame({"filter": filters, "delta_c": delta})
        for fkey, sub in temp.groupby("filter"):
            vals = pd.to_numeric(sub["delta_c"], errors="coerce").to_numpy(float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            out[str(fkey)] = float(np.nanmedian(vals))
        if "" not in out and "filter" not in df.columns:
            vals = pd.to_numeric(delta, errors="coerce").to_numpy(float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                out[""] = float(np.nanmedian(vals))
        return out

    def _load_color_median_table(self, result_dir: Path) -> pd.DataFrame:
        candidates = [
            step11_zeropoint_dir(result_dir) / "median_by_ID_filter_wide_cmd.csv",
            step11_zeropoint_dir(result_dir) / "median_by_ID_filter_wide.csv",
            result_dir / "median_by_ID_filter_wide_cmd.csv",
            result_dir / "median_by_ID_filter_wide.csv",
        ]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()

    def _available_color_expressions(self, result_dir: Path, bands_hint: list[str] | None = None) -> list[str]:
        df = self._load_color_median_table(result_dir)
        bands = set()
        if not df.empty:
            for col in df.columns:
                if col.startswith("mag_std_"):
                    bands.add(col.replace("mag_std_", "").strip().lower())
                elif col.startswith("mag_inst_"):
                    bands.add(col.replace("mag_inst_", "").strip().lower())
        if not bands and bands_hint:
            bands = {str(b).strip().lower() for b in bands_hint if str(b).strip()}
        bands = sorted({b for b in bands if b})
        exprs = []
        for a in bands:
            for b in bands:
                if a == b:
                    continue
                exprs.append(f"{a}-{b}")
        return sorted(set(exprs))

    def _compute_delta_c_map_from_median(
        self,
        result_dir: Path,
        target_id: int,
        comp_ids: list[int],
        mapping: dict[str, str],
    ) -> dict[str, float]:
        if not mapping:
            return {}
        df = self._load_color_median_table(result_dir)
        if df.empty or "ID" not in df.columns:
            return {}
        ids = pd.to_numeric(df["ID"], errors="coerce").astype("Int64")
        out: dict[str, float] = {}
        for fkey, expr in mapping.items():
            bands = _parse_color_expr(expr)
            if not bands:
                continue
            col_a = None
            col_b = None
            for prefix in ("mag_std_", "mag_inst_"):
                cand_a = f"{prefix}{bands[0]}"
                cand_b = f"{prefix}{bands[1]}"
                if cand_a in df.columns and cand_b in df.columns:
                    col_a = cand_a
                    col_b = cand_b
                    break
            if col_a is None or col_b is None:
                continue
            color = pd.to_numeric(df[col_a], errors="coerce") - pd.to_numeric(df[col_b], errors="coerce")
            cmap: dict[int, float] = {}
            for id_val, c in zip(ids.to_numpy(), color.to_numpy(float)):
                if pd.isna(id_val):
                    continue
                cmap[int(id_val)] = float(c) if np.isfinite(c) else np.nan
            target_color = cmap.get(target_id)
            if target_color is None or not np.isfinite(target_color):
                continue
            comp_colors = [cmap.get(int(cid), np.nan) for cid in comp_ids]
            comp_mean = float(np.nanmean(comp_colors)) if comp_colors else np.nan
            if not np.isfinite(comp_mean):
                continue
            out[_normalize_filter_key(fkey)] = float(target_color - comp_mean)
        return out

    def _compute_delta_c_map_from_raw(self, df: pd.DataFrame, mapping: dict[str, str]) -> dict[str, float]:
        if df.empty or not mapping:
            return {}
        if "filter" not in df.columns or "mag" not in df.columns or "comp_avg" not in df.columns:
            return {}
        data = df.copy()
        data["filter_key"] = data["filter"].astype(str).map(_normalize_filter_key)
        data["mag"] = pd.to_numeric(data["mag"], errors="coerce")
        data["comp_avg"] = pd.to_numeric(data["comp_avg"], errors="coerce")
        med_target = data.groupby("filter_key")["mag"].median()
        med_comp = data.groupby("filter_key")["comp_avg"].median()
        out: dict[str, float] = {}
        for fkey, expr in mapping.items():
            bands = _parse_color_expr(expr)
            if not bands:
                continue
            b1, b2 = bands
            if b1 not in med_target.index or b2 not in med_target.index:
                continue
            t1 = _safe_float(med_target.get(b1))
            t2 = _safe_float(med_target.get(b2))
            c1 = _safe_float(med_comp.get(b1))
            c2 = _safe_float(med_comp.get(b2))
            if not all(np.isfinite(v) for v in (t1, t2, c1, c2)):
                continue
            out[_normalize_filter_key(fkey)] = float((t1 - t2) - (c1 - c2))
        return out

    def _refresh_delta_c_map(self) -> None:
        result_dir = self.datasets[0][1] if self.datasets else self.params.P.result_dir
        target_text = self.target_edit.text().strip()
        mapping = self.color_map_by_filter
        if mapping:
            delta_map = self._compute_delta_c_map_from_raw(self.raw_df, mapping)
        else:
            delta_map = {}
        if not delta_map:
            if target_text:
                target_id = int(target_text)
                comp_ids = self.comp_active_ids if self.comp_active_ids else self.comp_candidate_ids
                if comp_ids and mapping:
                    delta_map = self._compute_delta_c_map_from_median(
                        Path(result_dir), target_id, comp_ids, mapping
                    )
            if not delta_map:
                delta_map = self._compute_delta_c_map(self.raw_df)
        self.delta_c_map = delta_map

    def _update_color_mode_enabled(self) -> bool:
        has_color = False
        if self.delta_c_map:
            has_color = any(np.isfinite(v) for v in self.delta_c_map.values())
        self.mode_color.setEnabled(has_color)
        if not has_color:
            self.color_status_label.setText("ⓘ Color mode 사용 불가: ΔC 데이터 없음")
        else:
            self.color_status_label.setText("")
        return has_color

    def _rebuild_color_map_controls(self, filters: list[str]) -> None:
        if not hasattr(self, "color_map_layout"):
            return
        result_dir = self.datasets[0][1] if self.datasets else self.params.P.result_dir
        options = self._available_color_expressions(Path(result_dir), filters)

        while self.color_map_layout.rowCount() > 0:
            self.color_map_layout.removeRow(0)

        self.color_map_combos = {}
        if not options:
            note = QLabel("색지수 데이터 없음")
            note.setStyleSheet("QLabel { color: #9E9E9E; font-size: 9pt; }")
            self.color_map_layout.addRow(note)
            return

        mapping = self._normalize_color_map(getattr(self.params.P, "lightcurve_color_index_by_filter", {}))
        if self.color_map_by_filter:
            mapping.update(self.color_map_by_filter)
        keys = sorted({_normalize_filter_key(f) for f in filters if str(f).strip()})
        if not keys:
            keys = sorted(mapping.keys())

        for fkey in keys:
            combo = QComboBox()
            combo.addItem("None")
            for expr in options:
                combo.addItem(expr)
            default_expr = mapping.get(fkey, "")
            if default_expr and combo.findText(default_expr) >= 0:
                combo.setCurrentText(default_expr)
            combo.currentIndexChanged.connect(self._on_color_map_changed)
            self.color_map_layout.addRow(f"{fkey}:", combo)
            self.color_map_combos[fkey] = combo
        self._on_color_map_changed()

    def _normalize_color_map(self, mapping) -> dict[str, str]:
        if not isinstance(mapping, dict):
            return {}
        out: dict[str, str] = {}
        for key, value in mapping.items():
            fkey = _normalize_filter_key(key)
            expr = str(value).strip().lower()
            if fkey and expr:
                out[fkey] = expr
        return out

    def _on_color_map_changed(self) -> None:
        if not hasattr(self, "color_map_combos"):
            return
        mapping: dict[str, str] = {}
        for fkey, combo in self.color_map_combos.items():
            expr = combo.currentText().strip().lower()
            if expr and expr != "none":
                mapping[fkey] = expr
        self.color_map_by_filter = mapping
        self._refresh_delta_c_map()
        self._log_color_index_info()
        self._update_color_mode_enabled()
        self._update_analysis_panel()

    def _delta_c_for_filter(self, fkey: str) -> float:
        if not self.delta_c_map:
            return np.nan
        key = _normalize_filter_key(fkey)
        if key in self.delta_c_map:
            return self.delta_c_map[key]
        if "" in self.delta_c_map:
            return self.delta_c_map[""]
        return np.nan

    def _log_color_index_info(self) -> None:
        mapping = self.color_map_by_filter or getattr(self.params.P, "lightcurve_color_index_by_filter", {}) or {}
        if mapping:
            pairs = ", ".join(f"{k}:{v}" for k, v in mapping.items())
            self.log(f"[COLOR] color_index_by_filter = {pairs}")
        if self.delta_c_map:
            for fkey in sorted(self.delta_c_map):
                self.log(f"[COLOR] ΔC median {fkey or 'all'} = {self.delta_c_map[fkey]:.5f}")

    def _populate_date_list(self):
        self.date_list.blockSignals(True)
        selected_dates = self._selected_dates()
        self.date_list.clear()
        dates = sorted({str(d) for d in self.raw_df.get("date", []) if str(d)})
        if selected_dates:
            dates = [d for d in dates if d in selected_dates]
        for d in dates:
            item = QListWidgetItem(d)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.date_list.addItem(item)
        self.date_list.blockSignals(False)

    def _on_date_selection_changed(self, _item: QListWidgetItem):
        self._update_plots()

    def _selected_dates(self) -> set[str]:
        dates = set()
        for i in range(self.date_list.count()):
            item = self.date_list.item(i)
            if item.checkState() == Qt.Checked:
                dates.add(item.text())
        return dates

    def _refresh_filter_combo(self, filters: list[str]) -> None:
        current = self.filter_combo.currentText() if hasattr(self, "filter_combo") else "All"
        keys = sorted({_normalize_filter_key(f) for f in filters if str(f).strip()})
        self.filter_combo.blockSignals(True)
        self.filter_combo.clear()
        self.filter_combo.addItem("All")
        for key in keys:
            self.filter_combo.addItem(key)
        if current and self.filter_combo.findText(current) >= 0:
            self.filter_combo.setCurrentText(current)
        elif keys:
            self.filter_combo.setCurrentIndex(1)
        self.filter_combo.blockSignals(False)

    def _on_filter_changed(self, _idx: int) -> None:
        self._update_plots()

    def _on_color_by_changed(self, _idx: int) -> None:
        self.color_by = self.color_by_combo.currentText() if hasattr(self, "color_by_combo") else "Date"
        self._update_plots()

    def _filter_linestyle(self, fkey: str) -> str:
        mapping = {
            "g": "-",
            "r": "--",
            "i": ":",
            "b": "-.",
            "v": (0, (3, 1, 1, 1)),
        }
        key = _normalize_filter_key(fkey)
        return mapping.get(key, "-")

    def _filter_color(self, fkey: str) -> str:
        mapping = {
            "g": "#2ca02c",
            "r": "#d62728",
            "i": "#9467bd",
            "b": "#1f77b4",
            "v": "#2ca02c",
            "z": "#8c564b",
        }
        key = _normalize_filter_key(fkey)
        return mapping.get(key, "#1f77b4")

    def _phase_xy(self, df: pd.DataFrame, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.x_axis_mode != "phase" or self.phase_period <= 0 or "JD" not in df.columns:
            x = df["JD"].to_numpy(float) if "JD" in df.columns else np.arange(len(df), dtype=float)
            return x, y
        jd = pd.to_numeric(df["JD"], errors="coerce").to_numpy(float)
        if not np.any(np.isfinite(jd)):
            return jd, y
        t0 = self.phase_t0 if self.phase_t0 > 0 else np.nanmin(jd)
        phase = ((jd - t0) / self.phase_period) % 1.0
        cycles = float(self.phase_cycles) if self.phase_cycles > 1 else 1.0
        full = int(cycles)
        frac = cycles - full
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        if full < 1:
            full = 1
        for k in range(full):
            xs.append(phase + k)
            ys.append(y)
        if frac > 1e-6:
            mask = phase <= (frac + 1e-9)
            xs.append(phase[mask] + full)
            ys.append(y[mask])
        return (np.concatenate(xs), np.concatenate(ys)) if xs else (phase, y)

    def _set_mode(self, mode: str):
        if self.mode == mode:
            return
        self.mode = mode
        is_global = mode == "global"
        if hasattr(self, "color_map_group"):
            self.color_map_group.setEnabled(not is_global)
        if hasattr(self, "chk_global_k2"):
            self.chk_global_k2.setEnabled(not is_global)
        if hasattr(self, "mode_color"):
            self.mode_color.setEnabled(True)
        if not self.raw_df.empty:
            self.corrected_df = pd.DataFrame()
            self.params_df = pd.DataFrame()
            self.global_mean_df = pd.DataFrame()
            self.global_diagnostics = {}
            self._update_results_table()
            self._update_plots()
            self.log(f"[MODE] Switched to {mode}. Run Fit && Apply to update.")

    def _on_clip_changed(self):
        self.sigma_clip = bool(self.chk_clip.isChecked())
        self.clip_sigma = float(self.spin_clip.value())
        self.clip_iters = int(self.spin_iters.value())

    def _on_xaxis_changed(self, idx: int) -> None:
        self.x_axis_mode = "phase" if idx == 1 else "time"
        self._update_plots()

    def _on_phase_params_changed(self):
        self.phase_period = float(self.spin_period.value())
        self.phase_t0 = float(self.spin_t0.value())
        if hasattr(self, "spin_cycles"):
            self.phase_cycles = float(self.spin_cycles.value())
        if self.x_axis_mode == "phase":
            self._update_plots()

    def _auto_set_t0(self):
        """Set T0 to min(JD) from loaded data."""
        if self.raw_df.empty:
            QMessageBox.information(self, "T₀ Auto", "먼저 데이터를 로드하세요.")
            return
        if "JD" not in self.raw_df.columns:
            QMessageBox.information(self, "T₀ Auto", "JD 컬럼이 없습니다.")
            return
        jd = pd.to_numeric(self.raw_df["JD"], errors="coerce").to_numpy(float)
        jd = jd[np.isfinite(jd)]
        if jd.size == 0:
            return
        t0_auto = float(np.min(jd))
        self.spin_t0.setValue(t0_auto)
        self.log(f"[PHASE] T₀ auto-set to min(JD) = {t0_auto:.6f}")

    def _mask_by_ranges(self, df: pd.DataFrame) -> np.ndarray:
        return np.ones(len(df), dtype=bool)

    def fit_and_apply(self):
        if self.mode == "global":
            self._run_global_ensemble()
            return
        if self.raw_df.empty:
            QMessageBox.information(self, "Detrend", "Raw 데이터가 없습니다.")
            return

        dates = self._selected_dates()
        if not dates:
            QMessageBox.information(self, "Detrend", "날짜를 하나 이상 선택하세요.")
            return

        # Color mode validation
        if self.mode == "color":
            has_delta_c = any(np.isfinite(self._delta_c_for_filter(fkey)) for fkey in self.delta_c_map)
            if not has_delta_c:
                msg = (
                    "Color mode를 사용하려면 ΔC (색지수 차이)가 필요합니다.\n\n"
                    "해결방법:\n"
                    "• Color Index 설정에서 필터별 색지수 조합 선택\n"
                    "• Step 11에서 color_index 컬럼이 있는 데이터 생성\n\n"
                    "현재는 Offset 모드로 진행합니다."
                )
                QMessageBox.warning(self, "Color Mode", msg)
                self.color_status_label.setText("⚠ ΔC 없음 - Offset 모드 사용")
                self.mode = "offset"
                self.mode_offset.setChecked(True)
            else:
                self.color_status_label.setText("")

        fit_df = self.raw_df
        use_global_k2 = self.chk_global_k2.isChecked() if hasattr(self, "chk_global_k2") else True

        # Color mode with global k'' fitting
        global_k2_by_filter: dict[str, tuple[float, float]] = {}

        if self.mode == "color" and use_global_k2:
            self.log("[FIT] Global k'' fitting mode enabled")
            all_filters = [""]
            if "filter" in fit_df.columns:
                all_filters = sorted({_normalize_filter_key(f) for f in fit_df["filter"].astype(str)})

            for fkey in all_filters:
                sub = fit_df
                if fkey and "filter" in sub.columns:
                    sub = fit_df[fit_df["filter"].astype(str).map(_normalize_filter_key) == fkey]
                if sub.empty:
                    continue

                delta_c_const = self._delta_c_for_filter(fkey)
                if not np.isfinite(delta_c_const):
                    self.log(f"[FIT] Global k'' for {fkey or 'all'}: ΔC missing, skipped")
                    continue

                date_mask = sub["date"].astype(str).isin([str(d) for d in dates])
                sub_all = sub[date_mask]
                if sub_all.empty:
                    continue

                mask = self._mask_by_ranges(sub_all)
                y = sub_all["diff_mag_raw"].to_numpy(float)
                x_air = sub_all["airmass"].to_numpy(float)
                x = x_air * float(delta_c_const)
                err = sub_all.get("diff_err", pd.Series([np.nan] * len(sub_all))).to_numpy(float)
                w = np.where(np.isfinite(err) & (err > 0), 1.0 / (err * err), 1.0)

                base_mask = mask & np.isfinite(y) & np.isfinite(x)
                if np.sum(base_mask) < 5:
                    continue

                zp, k2, zp_err, k2_err, _ = self._fit_linear(y, x, w, base_mask, fit_slope=True)
                global_k2_by_filter[fkey] = (k2, k2_err)
                self.log(f"[FIT] Global k'' ({fkey or 'all'}): {k2:.5f} ± {k2_err:.5f}")

                # Warn if k'' is unreasonably large (typical: 0.02-0.05)
                if abs(k2) > 0.15:
                    x_range = np.ptp(x_air[base_mask])
                    self.log(f"[WARNING] |k''| = {abs(k2):.3f} >> 0.05 (비정상)")
                    self.log(f"  → Airmass 범위 ΔX = {x_range:.3f} (좁으면 피팅 불안정)")
                    self.log(f"  → Offset 모드 권장 (k'' 피팅 불가)")

        params_rows = []
        for date_val in sorted(dates):
            sub_date = fit_df[fit_df["date"].astype(str) == str(date_val)]
            if sub_date.empty:
                continue
            filters = [""]
            if "filter" in sub_date.columns:
                filters = sorted({_normalize_filter_key(f) for f in sub_date["filter"].astype(str)})

            for fkey in filters:
                sub = sub_date
                if fkey:
                    sub = sub_date[sub_date["filter"].astype(str).map(_normalize_filter_key) == fkey]
                if sub.empty:
                    continue

                mask = self._mask_by_ranges(sub)
                y = sub["diff_mag_raw"].to_numpy(float)
                x_air = sub["airmass"].to_numpy(float)
                err = sub.get("diff_err", pd.Series([np.nan] * len(sub))).to_numpy(float)
                w = np.where(np.isfinite(err) & (err > 0), 1.0 / (err * err), 1.0)

                x = x_air
                delta_c_const = np.nan
                if self.mode == "color":
                    delta_c_const = self._delta_c_for_filter(fkey)
                    if not np.isfinite(delta_c_const):
                        self.log(f"[FIT] {date_val}/{fkey or 'all'}: ΔC missing, skipped")
                        continue
                    x = x_air * float(delta_c_const)

                base_mask = mask & np.isfinite(y)
                if self.mode != "offset":
                    base_mask = base_mask & np.isfinite(x)
                if not np.any(base_mask):
                    n_total = len(sub)
                    n_y = int(np.sum(np.isfinite(y)))
                    n_x = int(np.sum(np.isfinite(x)))
                    n_mask = int(np.sum(mask))
                    self.log(
                        f"[FIT] {date_val}/{fkey or 'all'}: no valid points "
                        f"(N={n_total}, y={n_y}, x={n_x}, mask={n_mask})"
                    )
                    continue

                if self.mode == "offset":
                    zp, slope, zp_err, slope_err, used_mask = self._fit_linear(
                        y, x, w, base_mask, fit_slope=False
                    )
                elif self.mode == "color" and use_global_k2 and fkey in global_k2_by_filter:
                    k2_global, k2_global_err = global_k2_by_filter[fkey]
                    y_adjusted = y - k2_global * x
                    zp, _, zp_err, _, used_mask = self._fit_linear(
                        y_adjusted, x, w, base_mask, fit_slope=False
                    )
                    slope = k2_global
                    slope_err = k2_global_err
                elif self.mode == "color":
                    zp, slope, zp_err, slope_err, used_mask = self._fit_linear(
                        y, x, w, base_mask, fit_slope=True
                    )
                else:
                    zp, slope, zp_err, slope_err, used_mask = self._fit_linear(
                        y, x, w, base_mask, fit_slope=False
                    )

                y_fit = zp + slope * x
                rms_before = np.nanstd(y[base_mask])
                rms_after = np.nanstd((y - y_fit)[used_mask]) if np.any(used_mask) else np.nan

                params_rows.append({
                    "date": date_val,
                    "filter": fkey,
                    "zp_offset": zp,
                    "zp_offset_err": zp_err,
                    "ext_slope": slope,
                    "ext_slope_err": slope_err,
                    "n_used": int(np.sum(used_mask)),
                    "rms_before": rms_before,
                    "rms_after": rms_after,
                    "global_k2": use_global_k2 and self.mode == "color",
                })

        if not params_rows:
            self.log("[FIT] No fit groups. Check airmass/ΔC/Date selection.")
            QMessageBox.information(self, "Detrend", "피팅할 데이터가 없습니다.")
            return

        self.params_df = pd.DataFrame(params_rows)
        self.corrected_df = self._apply_params(self.raw_df, self.params_df)
        self._update_results_table()
        self._update_plots()
        self.log(f"[FIT] Applied corrections for {len(self.params_df)} groups")
        self._log_fit_summary()
        self._save_comprehensive_results()

    def _run_global_ensemble(self) -> None:
        try:
            df_global = self._load_global_ensemble_df()
        except Exception as e:
            QMessageBox.warning(self, "Global Ensemble", str(e))
            return

        target_text = self.target_edit.text().strip()
        if not target_text:
            QMessageBox.warning(self, "Global Ensemble", "Target ID가 필요합니다.")
            return
        target_id = int(target_text)

        comp_ids = self.comp_active_ids or self.comp_candidate_ids
        comp_ids = [int(c) for c in comp_ids if str(c).strip() and int(c) != target_id]
        if not comp_ids:
            QMessageBox.warning(self, "Global Ensemble", "비교성 ID가 필요합니다.")
            return

        self.global_min_comps = int(self.spin_global_min_comps.value())
        self.global_sigma = float(self.spin_global_sigma.value())
        self.global_iters = int(self.spin_global_iters.value())
        self.global_rms_pct = float(self.spin_global_rms_pct.value())
        self.global_rms_threshold = float(self.spin_global_rms_thr.value())
        self.global_frame_sigma = float(self.spin_global_frame_sigma.value())
        self.global_gauge = str(self.combo_global_gauge.currentText())
        self.global_robust = bool(self.chk_global_robust.isChecked())
        self.global_interp_missing = bool(self.chk_global_interp.isChecked())
        self.global_normalize = bool(self.chk_global_normalize.isChecked())

        self.log(
            "[GLOBAL] min_comps={mc} sigma={sg} iters={it} rms_pct={rp} rms_thr={rt} frame_sigma={fs} gauge={g}".format(
                mc=self.global_min_comps,
                sg=self.global_sigma,
                it=self.global_iters,
                rp=self.global_rms_pct,
                rt=self.global_rms_threshold,
                fs=self.global_frame_sigma,
                g=self.global_gauge,
            )
        )

        try:
            result = solve_global_ensemble(
                df_global,
                target_id=target_id,
                comp_ids=comp_ids,
                min_comps=self.global_min_comps,
                sigma=self.global_sigma,
                n_iter=self.global_iters,
                gauge=self.global_gauge,
                per_filter=True,
                robust=self.global_robust,
                rms_clip_pct=self.global_rms_pct,
                rms_clip_threshold=self.global_rms_threshold if self.global_rms_threshold > 0 else None,
                frame_sigma=self.global_frame_sigma,
                interp_missing=self.global_interp_missing,
                normalize_target=self.global_normalize,
                log=self.log,
            )
        except Exception as e:
            QMessageBox.warning(self, "Global Ensemble", f"Fit failed: {e}")
            return

        self.global_input_df = df_global
        self.params_df = result.get("zp_df", pd.DataFrame())
        self.global_mean_df = result.get("mean_df", pd.DataFrame())
        self.corrected_df = result.get("lc_df", pd.DataFrame())
        self.raw_df = self.corrected_df.copy()
        self.global_diagnostics = result.get("diagnostics", {}) or {}

        if "JD" not in self.raw_df.columns and "jd" in self.raw_df.columns:
            self.raw_df["JD"] = self.raw_df["jd"]
        if "JD" not in self.raw_df.columns:
            self.raw_df["JD"] = np.nan
        if "date" not in self.raw_df.columns:
            self.raw_df["date"] = "unknown"
        self._fill_date_from_jd()

        self._populate_date_list()
        self._refresh_filter_combo(self.raw_df.get("filter", pd.Series([], dtype=str)).astype(str).tolist())
        self._update_results_table()
        self._update_plots()
        self._update_analysis_panel()
        self.log("[GLOBAL] Fit complete")
        self._save_comprehensive_results()

    def _log_fit_summary(self):
        """Log fit summary with astronomical interpretation."""
        if self.params_df.empty:
            return

        self.log("\n" + "=" * 50)
        self.log("[SUMMARY] 피팅 결과 분석")
        self.log("=" * 50)

        # Global RMS improvement (all data combined)
        if not self.corrected_df.empty:
            raw_all = pd.to_numeric(self.corrected_df["diff_mag_raw"], errors="coerce").to_numpy(float)
            corr_all = pd.to_numeric(self.corrected_df["diff_mag_corr"], errors="coerce").to_numpy(float)
            raw_all = raw_all[np.isfinite(raw_all)]
            corr_all = corr_all[np.isfinite(corr_all)]

            if raw_all.size > 0 and corr_all.size > 0:
                global_rms_before = np.std(raw_all)
                global_rms_after = np.std(corr_all)
                global_improve = (1 - global_rms_after / global_rms_before) * 100 if global_rms_before > 0 else 0
                self.log(f"  전체 RMS: {global_rms_before:.4f} → {global_rms_after:.4f} mag ({global_improve:+.1f}%)")

                if self.mode == "offset" and abs(global_improve) < 1:
                    self.log("  → Offset 모드: 밤별 RMS는 동일, 밤간 정렬로 개선")

        # Per-night average RMS (for color mode comparison)
        rms_before = self.params_df["rms_before"].mean()
        rms_after = self.params_df["rms_after"].mean()
        if np.isfinite(rms_before) and np.isfinite(rms_after) and rms_before > 0:
            improvement = (1 - rms_after / rms_before) * 100
            self.log(f"  밤별 평균 RMS: {rms_before:.4f} → {rms_after:.4f} mag ({improvement:+.1f}%)")

            if self.mode == "color" and improvement < -5:
                self.log("  → [경고] 밤별 RMS 증가: 과적합 가능성 (데이터 부족)")

        # Mode-specific analysis
        if self.mode == "color":
            slopes = self.params_df["ext_slope"].to_numpy(float)
            slopes = slopes[np.isfinite(slopes)]
            if slopes.size > 0:
                k2_mean = np.mean(slopes)
                k2_std = np.std(slopes) if slopes.size > 1 else 0.0
                self.log(f"  k'' (2차 소광계수): {k2_mean:.4f} ± {k2_std:.4f} mag/mag")

                # Typical k'' values for reference (should be 0.02-0.05)
                if abs(k2_mean) < 0.01:
                    self.log("  → k'' ≈ 0: 색항 효과 미미 (Offset 모드로 충분)")
                elif abs(k2_mean) <= 0.05:
                    self.log("  → k'' 정상 범위 (0.02-0.05 typical for broad-band)")
                elif abs(k2_mean) <= 0.15:
                    self.log("  → k'' 다소 큼: 데이터 품질 확인 권장")
                else:
                    self.log(f"  → [경고] |k''| = {abs(k2_mean):.2f} >> 0.05 (비정상)")
                    self.log("    원인: Airmass 범위(ΔX)가 좁아 k'' 피팅 불안정")
                    self.log("    해결: Offset 모드 사용 권장")

        # ZP scatter analysis
        zps = self.params_df["zp_offset"].to_numpy(float)
        zps = zps[np.isfinite(zps)]
        if zps.size > 1:
            zp_scatter = np.std(zps)
            self.log(f"  밤별 ZP 산포: {zp_scatter:.4f} mag")
            if zp_scatter > 0.3:
                self.log("  → [경고] ZP 산포가 매우 큼")
                if self.mode == "color":
                    self.log("    원인: k'' 과적합으로 ZP₀가 보상하는 중일 수 있음")
                    self.log("    해결: Offset 모드로 다시 시도 권장")
            elif zp_scatter > 0.1:
                self.log("  → 밤별 조건 변화가 큼 (정상 범위)")

        # Night-by-night alignment verification
        self._log_alignment_verification()

        self.log("=" * 50 + "\n")

    def _log_alignment_verification(self):
        """Log per-night raw mean, ZP, and corrected mean to verify alignment."""
        if self.corrected_df.empty or self.params_df.empty:
            return

        self.log("\n  [밤간 정렬 검증]")
        self.log("  " + "-" * 46)
        self.log(f"  {'Date':<12} {'Raw평균':>10} {'ZP₀':>10} {'Corr평균':>10} {'검증':>6}")
        self.log("  " + "-" * 46)

        df = self.corrected_df
        dates = sorted(df["date"].astype(str).unique())

        alignment_ok = True
        corr_means = []

        for date_val in dates:
            date_mask = df["date"].astype(str) == date_val
            sub = df[date_mask]

            raw_vals = pd.to_numeric(sub["diff_mag_raw"], errors="coerce").to_numpy(float)
            corr_vals = pd.to_numeric(sub["diff_mag_corr"], errors="coerce").to_numpy(float)

            raw_mean = np.nanmean(raw_vals) if np.any(np.isfinite(raw_vals)) else np.nan
            corr_mean = np.nanmean(corr_vals) if np.any(np.isfinite(corr_vals)) else np.nan

            # Get ZP for this date (may have multiple filters, take mean)
            zp_rows = self.params_df[self.params_df["date"].astype(str) == date_val]
            zp_mean = zp_rows["zp_offset"].mean() if not zp_rows.empty else np.nan

            # Verify: raw_mean ≈ ZP (for offset mode, ZP = weighted mean of raw)
            if np.isfinite(raw_mean) and np.isfinite(zp_mean):
                diff = abs(raw_mean - zp_mean)
                check = "✓" if diff < 0.05 else "~"
            else:
                check = "-"

            if np.isfinite(corr_mean):
                corr_means.append(corr_mean)

            self.log(f"  {date_val:<12} {raw_mean:>10.4f} {zp_mean:>10.4f} {corr_mean:>10.4f} {check:>6}")

        self.log("  " + "-" * 46)

        # Check if corrected means are aligned
        if len(corr_means) > 1:
            corr_scatter = np.std(corr_means)
            self.log(f"  보정 후 밤간 평균 산포: {corr_scatter:.4f} mag")
            if corr_scatter < 0.02:
                self.log("  → ✓ 밤간 정렬 양호")
            elif corr_scatter < 0.05:
                self.log("  → ~ 밤간 정렬 적절")
            else:
                self.log("  → ⚠ 밤간 정렬 불완전 (필터별 차이 또는 피팅 문제)")
                alignment_ok = False

    def _fit_linear(self, y, x, w, base_mask, fit_slope: bool = True):
        mask = base_mask.copy()
        zp = 0.0
        slope = 0.0
        zp_err = np.nan
        slope_err = np.nan

        for _ in range(self.clip_iters if self.sigma_clip else 1):
            if np.sum(mask) < 2:
                break
            yy = y[mask]
            xx = x[mask]
            ww = w[mask]

            if not fit_slope:
                zp = float(np.average(yy, weights=ww))
                slope = 0.0
                if np.sum(ww) > 0:
                    zp_err = float(np.sqrt(1.0 / np.sum(ww)))
                slope_err = 0.0
            else:
                A = np.vstack([np.ones_like(xx), xx]).T
                Aw = A * np.sqrt(ww[:, None])
                yw = yy * np.sqrt(ww)
                try:
                    coeff, residuals, rank, s = np.linalg.lstsq(Aw, yw, rcond=None)
                    zp = float(coeff[0])
                    slope = float(coeff[1])

                    n = len(yy)
                    if n > 2:
                        resid_fit = yy - (zp + slope * xx)
                        mse = np.sum(ww * resid_fit**2) / (n - 2)
                        try:
                            AtWA = A.T @ np.diag(ww) @ A
                            cov = mse * np.linalg.inv(AtWA)
                            zp_err = float(np.sqrt(cov[0, 0]))
                            slope_err = float(np.sqrt(cov[1, 1]))
                        except Exception:
                            zp_err = np.nan
                            slope_err = np.nan
                except Exception:
                    zp, slope = 0.0, 0.0
                    zp_err, slope_err = np.nan, np.nan

            resid = y - (zp + slope * x)
            sigma = np.nanstd(resid[mask])
            if not self.sigma_clip or not np.isfinite(sigma) or sigma == 0:
                break
            mask = mask & (np.abs(resid) <= self.clip_sigma * sigma)

        return zp, slope, zp_err, slope_err, mask

    def _apply_params(self, df: pd.DataFrame, params_df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["diff_mag_corr"] = out["diff_mag_raw"]

        if "diff_err" not in out.columns:
            out["diff_err"] = np.nan
        out["diff_err_corr"] = out["diff_err"].copy()

        for _, row in params_df.iterrows():
            date_val = str(row["date"])
            zp = _safe_float(row.get("zp_offset"))
            zp_err = _safe_float(row.get("zp_offset_err", 0.0))
            slope = _safe_float(row.get("ext_slope"))
            slope_err = _safe_float(row.get("ext_slope_err", 0.0))

            idx = out["date"].astype(str) == date_val
            fval = _normalize_filter_key(row.get("filter", ""))
            if fval and "filter" in out.columns:
                idx &= out["filter"].astype(str).map(_normalize_filter_key) == fval

            x = out.loc[idx, "airmass"].to_numpy(float)
            if self.mode == "color":
                delta_c_const = self._delta_c_for_filter(fval)
                if not np.isfinite(delta_c_const):
                    continue
                x = x * float(delta_c_const)

            out.loc[idx, "diff_mag_corr"] = out.loc[idx, "diff_mag_raw"] - zp - slope * x

            raw_err = out.loc[idx, "diff_err"].to_numpy(float)
            raw_var = np.where(np.isfinite(raw_err), raw_err**2, 0.0)
            zp_var = zp_err**2 if np.isfinite(zp_err) else 0.0
            slope_var = slope_err**2 if np.isfinite(slope_err) else 0.0
            corr_var = raw_var + zp_var + (x**2) * slope_var
            out.loc[idx, "diff_err_corr"] = np.sqrt(corr_var)

        return out

    def _update_results_table(self):
        self.result_table.setRowCount(0)
        if self.params_df.empty:
            return

        if self.mode == "global":
            self.result_table.setColumnCount(6)
            self.result_table.setHorizontalHeaderLabels(
                ["Frame", "Filter", "N", "Z_t", "±σ(Z)", "χ²_red"]
            )
            for _, row in self.params_df.iterrows():
                r = self.result_table.rowCount()
                self.result_table.insertRow(r)
                self.result_table.setItem(r, 0, QTableWidgetItem(str(row.get("time_id", ""))))
                ftext = str(row.get("filter", "")).strip()
                self.result_table.setItem(r, 1, QTableWidgetItem(ftext if ftext else "all"))
                self.result_table.setItem(r, 2, QTableWidgetItem(str(int(row.get("n_used", 0)))))
                self.result_table.setItem(r, 3, QTableWidgetItem(_fmt_float(row.get("Z"))))
                self.result_table.setItem(r, 4, QTableWidgetItem(_fmt_float(row.get("Z_err"))))
                self.result_table.setItem(r, 5, QTableWidgetItem(_fmt_float(row.get("chi2_red"))))
            return

        self.result_table.setColumnCount(9)
        if self.mode == "color":
            is_global = self.params_df.get("global_k2", pd.Series([False])).iloc[0]
            slope_label = "k'' (global)" if is_global else "k'' (nightly)"
        else:
            slope_label = "-"

        self.result_table.setHorizontalHeaderLabels(
            ["Date", "Filter", "N", "ZP₀", "±σ(ZP)", slope_label, "±σ(k'')", "RMS전", "RMS후"]
        )

        for _, row in self.params_df.iterrows():
            r = self.result_table.rowCount()
            self.result_table.insertRow(r)
            self.result_table.setItem(r, 0, QTableWidgetItem(str(row.get("date", ""))))
            ftext = str(row.get("filter", "")).strip()
            self.result_table.setItem(r, 1, QTableWidgetItem(ftext if ftext else "all"))
            self.result_table.setItem(r, 2, QTableWidgetItem(str(int(row.get("n_used", 0)))))
            self.result_table.setItem(r, 3, QTableWidgetItem(_fmt_float(row.get("zp_offset"))))
            self.result_table.setItem(r, 4, QTableWidgetItem(_fmt_float(row.get("zp_offset_err"))))
            self.result_table.setItem(r, 5, QTableWidgetItem(_fmt_float(row.get("ext_slope"))))
            self.result_table.setItem(r, 6, QTableWidgetItem(_fmt_float(row.get("ext_slope_err"))))
            self.result_table.setItem(r, 7, QTableWidgetItem(_fmt_float(row.get("rms_before"))))
            self.result_table.setItem(r, 8, QTableWidgetItem(_fmt_float(row.get("rms_after"))))

    def _update_plots(self):
        self.ax_raw.clear()
        self.ax_corr.clear()
        self.ax_diag.clear()

        if self.raw_df.empty:
            self.plot_canvas.draw()
            return

        selected_dates = self._selected_dates()
        if selected_dates:
            dates = sorted(selected_dates)
        else:
            dates = sorted({str(d) for d in self.raw_df.get("date", []) if str(d)})
        date_colors = {}
        palette = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        ]
        for i, d in enumerate(dates):
            date_colors[d] = palette[i % len(palette)]

        filter_sel = "All"
        if hasattr(self, "filter_combo"):
            filter_sel = self.filter_combo.currentText() or "All"
        filter_key = "" if filter_sel == "All" else _normalize_filter_key(filter_sel)

        raw = self.raw_df.copy()
        if filter_key and "filter" in raw.columns:
            raw = raw[raw["filter"].astype(str).map(_normalize_filter_key) == filter_key]
        if dates:
            raw = raw[raw["date"].astype(str).isin(dates)]

        x_label = "JD"
        if self.x_axis_mode == "phase" and self.phase_period > 0 and "JD" in raw.columns:
            x_label = "Phase"
            if self.phase_cycles > 1:
                x_label = f"Phase (0-{self.phase_cycles:g})"

        if self.color_by == "Filter" and "filter" in raw.columns:
            for fkey, sub in raw.groupby(raw["filter"].astype(str).map(_normalize_filter_key)):
                y = sub["diff_mag_raw"].to_numpy(float)
                x, y = self._phase_xy(sub, y)
                m = np.isfinite(x) & np.isfinite(y)
                if np.any(m):
                    self.ax_raw.plot(
                        x[m], y[m], marker="o", linestyle="None",
                        color=self._filter_color(fkey), markersize=3, alpha=0.7, label=fkey or "all"
                    )
        else:
            for d in dates:
                sub = raw[raw["date"].astype(str) == str(d)]
                y = sub["diff_mag_raw"].to_numpy(float)
                x, y = self._phase_xy(sub, y)
                m = np.isfinite(x) & np.isfinite(y)
                if np.any(m):
                    self.ax_raw.plot(x[m], y[m], marker="o", linestyle="None", color=date_colors[d], markersize=3, alpha=0.7, label=d)

        self.ax_raw.set_title("Raw", fontsize=10)
        self.ax_raw.set_xlabel(x_label, fontsize=9)
        self.ax_raw.set_ylabel("Δmag", fontsize=9)
        self.ax_raw.invert_yaxis()
        self.ax_raw.grid(True, alpha=0.3)
        self.ax_raw.tick_params(labelsize=8)

        corr = self.corrected_df if not self.corrected_df.empty else raw
        if filter_key and "filter" in corr.columns:
            corr = corr[corr["filter"].astype(str).map(_normalize_filter_key) == filter_key]
        if dates and "date" in corr.columns:
            corr = corr[corr["date"].astype(str).isin(dates)]

        if self.color_by == "Filter" and "filter" in corr.columns:
            for fkey, sub in corr.groupby(corr["filter"].astype(str).map(_normalize_filter_key)):
                y = sub["diff_mag_corr"].to_numpy(float) if "diff_mag_corr" in sub.columns else sub["diff_mag_raw"].to_numpy(float)
                x, y = self._phase_xy(sub, y)
                m = np.isfinite(x) & np.isfinite(y)
                if np.any(m):
                    self.ax_corr.plot(
                        x[m], y[m], marker="o", linestyle="None",
                        color=self._filter_color(fkey), markersize=3, alpha=0.7, label=fkey or "all"
                    )
        else:
            if "date" in corr.columns:
                for d in dates:
                    sub = corr[corr["date"].astype(str) == str(d)]
                    y = sub["diff_mag_corr"].to_numpy(float) if "diff_mag_corr" in sub.columns else sub["diff_mag_raw"].to_numpy(float)
                    x, y = self._phase_xy(sub, y)
                    m = np.isfinite(x) & np.isfinite(y)
                    if np.any(m):
                        self.ax_corr.plot(x[m], y[m], marker="o", linestyle="None", color=date_colors[d], markersize=3, alpha=0.7, label=d)
            else:
                # No date column, plot all data
                y = corr["diff_mag_corr"].to_numpy(float) if "diff_mag_corr" in corr.columns else corr["diff_mag_raw"].to_numpy(float)
                x, y = self._phase_xy(corr, y)
                m = np.isfinite(x) & np.isfinite(y)
                if np.any(m):
                    self.ax_corr.plot(x[m], y[m], marker="o", linestyle="None", color="#1f77b4", markersize=3, alpha=0.7)

        self.ax_corr.set_title("Corrected", fontsize=10)
        self.ax_corr.set_xlabel(x_label, fontsize=9)
        self.ax_corr.set_ylabel("Δmag", fontsize=9)
        self.ax_corr.invert_yaxis()
        self.ax_corr.grid(True, alpha=0.3)
        self.ax_corr.tick_params(labelsize=8)

        if self.mode == "global":
            self._plot_global_diagnostics(dates, filter_key, date_colors)
            for ax in (self.ax_raw, self.ax_corr, self.ax_diag):
                handles, labels = ax.get_legend_handles_labels()
                if handles and len(handles) <= 8:
                    ax.legend(loc="best", fontsize=7, framealpha=0.8)
            self.plot_canvas.figure.tight_layout()
            self.plot_canvas.draw()
            return

        # Diagnostic plot
        diag = raw
        if self.color_by == "Filter" and "filter" in diag.columns:
            for fkey, sub in diag.groupby(diag["filter"].astype(str).map(_normalize_filter_key)):
                x = sub["airmass"].to_numpy(float)
                y = sub["diff_mag_raw"].to_numpy(float)
                m = np.isfinite(x) & np.isfinite(y)
                if np.any(m):
                    self.ax_diag.plot(
                        x[m], y[m], marker="o", linestyle="None",
                        color=self._filter_color(fkey), markersize=3, alpha=0.7, label=fkey or "all"
                    )
        else:
            for d in dates:
                sub = diag[diag["date"].astype(str) == str(d)]
                x = sub["airmass"].to_numpy(float)
                y = sub["diff_mag_raw"].to_numpy(float)
                m = np.isfinite(x) & np.isfinite(y)
                if np.any(m):
                    self.ax_diag.plot(x[m], y[m], marker="o", linestyle="None", color=date_colors[d], markersize=3, alpha=0.7, label=d)

        # Fit lines
        if not self.params_df.empty:
            for _, row in self.params_df.iterrows():
                date_val = str(row.get("date", ""))
                if dates and date_val not in dates:
                    continue
                fkey = _normalize_filter_key(row.get("filter", ""))
                if filter_key and fkey != filter_key:
                    continue
                sub = diag[diag["date"].astype(str) == date_val]
                if fkey and "filter" in sub.columns:
                    sub = sub[sub["filter"].astype(str).map(_normalize_filter_key) == fkey]
                xvals = pd.to_numeric(sub.get("airmass", pd.Series([], dtype=float)), errors="coerce").to_numpy(float)
                xvals = xvals[np.isfinite(xvals)]
                if xvals.size == 0:
                    continue
                xmin = float(np.nanmin(xvals))
                xmax = float(np.nanmax(xvals))
                if not np.isfinite(xmin) or not np.isfinite(xmax):
                    continue
                if xmin == xmax:
                    xmin -= 0.05
                    xmax += 0.05
                xline = np.linspace(xmin, xmax, 50)
                xfit = xline
                if self.mode == "color":
                    delta_c_const = self._delta_c_for_filter(fkey)
                    if not np.isfinite(delta_c_const):
                        continue
                    xfit = xline * float(delta_c_const)
                yline = float(row.get("zp_offset", 0.0)) + float(row.get("ext_slope", 0.0)) * xfit
                linestyle = "-" if filter_key else self._filter_linestyle(fkey)
                line_color = date_colors.get(date_val, "#333333") if self.color_by == "Date" else self._filter_color(fkey)
                self.ax_diag.plot(xline, yline, color=line_color, linestyle=linestyle, linewidth=1.5, alpha=0.9)

        self.ax_diag.set_title("Δmag vs Airmass (diagnostics)", fontsize=10)
        self.ax_diag.set_xlabel("Airmass", fontsize=9)
        self.ax_diag.set_ylabel("Δmag", fontsize=9)
        self.ax_diag.invert_yaxis()
        self.ax_diag.grid(True, alpha=0.3)
        self.ax_diag.tick_params(labelsize=8)

        # Legends (compact)
        for ax in (self.ax_raw, self.ax_corr, self.ax_diag):
            handles, labels = ax.get_legend_handles_labels()
            if handles and len(handles) <= 8:
                ax.legend(loc="best", fontsize=7, framealpha=0.8)

        self.plot_canvas.figure.tight_layout()
        self.plot_canvas.draw()

    def _plot_global_diagnostics(self, dates, filter_key, date_colors) -> None:
        self.ax_diag.clear()
        if self.params_df.empty:
            self.ax_diag.text(0.5, 0.5, "No global ZP available", ha="center", va="center")
            return

        diag_df = self.params_df.copy()
        ref_df = self.corrected_df if not self.corrected_df.empty else self.raw_df
        if not ref_df.empty and "time_id" in ref_df.columns:
            if "JD" in ref_df.columns:
                jd_map = ref_df.groupby("time_id")["JD"].median()
                diag_df["JD"] = diag_df["time_id"].map(jd_map)
            if "date" in ref_df.columns:
                date_map = ref_df.groupby("time_id")["date"].first()
                diag_df["date"] = diag_df["time_id"].map(date_map)

        if filter_key and "filter" in diag_df.columns:
            diag_df = diag_df[diag_df["filter"].astype(str).map(_normalize_filter_key) == filter_key]
        if dates and "date" in diag_df.columns:
            diag_df = diag_df[diag_df["date"].astype(str).isin(dates)]

        x = diag_df.get("JD", pd.Series([], dtype=float))
        y = diag_df.get("Z", pd.Series([], dtype=float))

        if self.color_by == "Filter" and "filter" in diag_df.columns:
            for fkey, sub in diag_df.groupby(diag_df["filter"].astype(str).map(_normalize_filter_key)):
                xs = pd.to_numeric(sub.get("JD", pd.Series([], dtype=float)), errors="coerce").to_numpy(float)
                ys = pd.to_numeric(sub.get("Z", pd.Series([], dtype=float)), errors="coerce").to_numpy(float)
                m = np.isfinite(xs) & np.isfinite(ys)
                if np.any(m):
                    self.ax_diag.plot(
                        xs[m], ys[m], marker="o", linestyle="None",
                        color=self._filter_color(fkey), markersize=3, alpha=0.7, label=fkey or "all"
                    )
        elif "date" in diag_df.columns:
            for d in sorted(diag_df["date"].astype(str).unique().tolist()):
                sub = diag_df[diag_df["date"].astype(str) == str(d)]
                xs = pd.to_numeric(sub.get("JD", pd.Series([], dtype=float)), errors="coerce").to_numpy(float)
                ys = pd.to_numeric(sub.get("Z", pd.Series([], dtype=float)), errors="coerce").to_numpy(float)
                m = np.isfinite(xs) & np.isfinite(ys)
                if np.any(m):
                    self.ax_diag.plot(xs[m], ys[m], marker="o", linestyle="None", color=date_colors.get(d, "#333333"), markersize=3, alpha=0.7, label=d)
        else:
            xs = pd.to_numeric(x, errors="coerce").to_numpy(float)
            ys = pd.to_numeric(y, errors="coerce").to_numpy(float)
            m = np.isfinite(xs) & np.isfinite(ys)
            if np.any(m):
                self.ax_diag.plot(xs[m], ys[m], marker="o", linestyle="None", color="#333333", markersize=3, alpha=0.7)

        self.ax_diag.set_title("Global Z_t vs JD", fontsize=10)
        self.ax_diag.set_xlabel("JD", fontsize=9)
        self.ax_diag.set_ylabel("Z_t", fontsize=9)
        self.ax_diag.grid(True, alpha=0.3)
        self.ax_diag.tick_params(labelsize=8)

    def _save_comprehensive_results(self) -> None:
        """Save comprehensive result files including formula, corrections, residuals, and summary."""
        if self.corrected_df.empty:
            return
        if self.mode == "global":
            self._save_global_results()
            return
        target_text = self.target_edit.text().strip()
        if not target_text:
            return

        target_id = int(target_text)
        out_dir = step12_dir(self.params.P.result_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        mode_tag = self.mode if self.mode in ("offset", "color") else "mode"

        try:
            # ===== 1. Main corrected light curve with residuals =====
            out_df = self.corrected_df.copy()

            # Add residuals (diff_mag_corr - median)
            corr_vals = pd.to_numeric(out_df["diff_mag_corr"], errors="coerce")
            median_corr = np.nanmedian(corr_vals)
            out_df["residual"] = corr_vals - median_corr

            # Add phase if period is set
            if self.phase_period > 0 and "JD" in out_df.columns:
                jd = pd.to_numeric(out_df["JD"], errors="coerce").to_numpy(float)
                t0 = self.phase_t0 if self.phase_t0 > 0 else np.nanmin(jd)
                out_df["phase"] = ((jd - t0) / self.phase_period) % 1.0

            # Add fit value (what was subtracted)
            out_df["fit_value"] = out_df["diff_mag_raw"] - out_df["diff_mag_corr"]

            # Reorder columns for clarity
            priority_cols = [
                "file", "JD", "date", "filter", "airmass",
                "diff_mag_raw", "diff_mag_corr", "fit_value", "residual",
                "diff_err", "diff_err_corr"
            ]
            if "phase" in out_df.columns:
                priority_cols.insert(3, "phase")
            other_cols = [c for c in out_df.columns if c not in priority_cols]
            out_df = out_df[[c for c in priority_cols if c in out_df.columns] + other_cols]

            lc_path = out_dir / f"lightcurve_ID{target_id}_{mode_tag}.csv"
            out_df.to_csv(lc_path, index=False)
            self.log(f"[SAVE] Light curve: {lc_path.name}")

            # ===== 2. Fit parameters by date/filter =====
            if not self.params_df.empty:
                params_out = self.params_df.copy()
                # Add formula info
                if self.mode == "offset":
                    params_out["formula"] = "Δm_corr = Δm_raw - ZP₀"
                else:
                    params_out["formula"] = "Δm_corr = Δm_raw - ZP₀ - k''·ΔC·X"

                params_path = out_dir / f"fit_params_ID{target_id}_{mode_tag}.csv"
                params_out.to_csv(params_path, index=False)
                self.log(f"[SAVE] Fit params: {params_path.name}")

            # ===== 3. Summary report (text file) =====
            summary_path = out_dir / f"summary_ID{target_id}_{mode_tag}.txt"
            self._write_summary_report(summary_path, out_df)
            self.log(f"[SAVE] Summary: {summary_path.name}")

            # ===== 4. Save plot as PNG =====
            plot_path = out_dir / f"plot_ID{target_id}_{mode_tag}.png"
            self.plot_canvas.figure.savefig(plot_path, dpi=150, bbox_inches="tight")
            self.log(f"[SAVE] Plot: {plot_path.name}")

            self.log(f"[SAVE] 모든 결과 저장 완료: {out_dir}")

        except Exception as e:
            self.log(f"[SAVE] Failed: {e}")
            QMessageBox.warning(self, "Save Error", f"저장 실패: {e}")

    def _save_global_results(self) -> None:
        if self.corrected_df.empty:
            return
        target_text = self.target_edit.text().strip()
        if not target_text:
            return
        target_id = int(target_text)
        out_dir = step12_dir(self.params.P.result_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            lc_path = out_dir / f"lightcurve_ID{target_id}_global.csv"
            self.corrected_df.to_csv(lc_path, index=False)
            self.log(f"[SAVE] Global light curve: {lc_path.name}")

            if not self.params_df.empty:
                zp_path = out_dir / f"global_zp_ID{target_id}.csv"
                self.params_df.to_csv(zp_path, index=False)
                self.log(f"[SAVE] Global ZP: {zp_path.name}")

            if not self.global_mean_df.empty:
                mean_path = out_dir / f"global_mean_ID{target_id}.csv"
                self.global_mean_df.to_csv(mean_path, index=False)
                self.log(f"[SAVE] Global mean: {mean_path.name}")

            if self.global_diagnostics:
                diag_path = out_dir / f"global_diagnostics_ID{target_id}.json"
                with open(diag_path, "w", encoding="utf-8") as f:
                    json.dump(self.global_diagnostics, f, indent=2)
                self.log(f"[SAVE] Diagnostics: {diag_path.name}")

            summary_path = out_dir / f"summary_ID{target_id}_global.txt"
            lines = [
                "=" * 60,
                "GLOBAL ENSEMBLE SUMMARY",
                "=" * 60,
                "",
                "Mode: GLOBAL (method C)",
                "Formula: mag_inst = M_i + Z_t",
                "Target curve: m_corr = mag_inst_target - Z_t",
                "",
                f"Target ID: {target_id}",
                f"Frames: {len(self.params_df)}",
                f"Points: {len(self.corrected_df)}",
            ]
            summary_path.write_text("\n".join(lines), encoding="utf-8")
            self.log(f"[SAVE] Summary: {summary_path.name}")
        except Exception as e:
            self.log(f"[SAVE] Global failed: {e}")
            QMessageBox.warning(self, "Save Error", f"저장 실패: {e}")

    def _write_summary_report(self, path: Path, df: pd.DataFrame) -> None:
        """Write a comprehensive text summary report."""
        lines = []
        lines.append("=" * 60)
        lines.append("DETREND & NIGHT MERGE - SUMMARY REPORT")
        lines.append("=" * 60)
        lines.append("")

        # Mode and formula
        lines.append(f"[Mode] {self.mode.upper()}")
        if self.mode == "offset":
            lines.append("  Formula: Δm_corr = Δm_raw - ZP₀")
            lines.append("  (Nightly zero-point offset only)")
        else:
            lines.append("  Formula: Δm_corr = Δm_raw - ZP₀ - k''·ΔC·X")
            lines.append("  (Color-dependent extinction correction)")
            if self.chk_global_k2.isChecked():
                lines.append("  k'' fitting: GLOBAL (single k'' for all nights)")
            else:
                lines.append("  k'' fitting: NIGHTLY (separate k'' per night)")
        lines.append("")

        # Data summary
        lines.append("[Data Summary]")
        lines.append(f"  Total points: {len(df)}")
        if "date" in df.columns:
            dates = sorted(df["date"].astype(str).unique())
            lines.append(f"  Dates: {len(dates)} nights")
            for d in dates:
                n = len(df[df["date"].astype(str) == d])
                lines.append(f"    - {d}: {n} points")
        if "filter" in df.columns:
            filters = sorted({_normalize_filter_key(f) for f in df["filter"].astype(str) if str(f).strip()})
            lines.append(f"  Filters: {', '.join(filters) or 'N/A'}")
        lines.append("")

        # Color index info
        if self.delta_c_map:
            lines.append("[Color Index (ΔC = Target - Comp)]")
            for fkey, dc in sorted(self.delta_c_map.items()):
                lines.append(f"  {fkey or 'all'}: ΔC = {dc:+.4f} mag")
            lines.append("")

        # Fit parameters
        if not self.params_df.empty:
            lines.append("[Fit Parameters by Date/Filter]")
            lines.append("-" * 60)
            if self.mode == "offset":
                lines.append(f"{'Date':<12} {'Filter':<6} {'N':>5} {'ZP₀':>10} {'±σ':>8} {'RMS_before':>10} {'RMS_after':>10}")
            else:
                lines.append(f"{'Date':<12} {'Filter':<6} {'N':>5} {'ZP₀':>10} {'k\"':>10} {'RMS_before':>10} {'RMS_after':>10}")
            lines.append("-" * 60)
            for _, row in self.params_df.iterrows():
                date = str(row.get("date", ""))[:12]
                filt = str(row.get("filter", "") or "all")[:6]
                n = int(row.get("n_used", 0))
                zp = row.get("zp_offset", np.nan)
                zp_err = row.get("zp_offset_err", np.nan)
                slope = row.get("ext_slope", np.nan)
                rms_b = row.get("rms_before", np.nan)
                rms_a = row.get("rms_after", np.nan)
                if self.mode == "offset":
                    lines.append(f"{date:<12} {filt:<6} {n:>5} {zp:>10.5f} {zp_err:>8.5f} {rms_b:>10.5f} {rms_a:>10.5f}")
                else:
                    lines.append(f"{date:<12} {filt:<6} {n:>5} {zp:>10.5f} {slope:>10.5f} {rms_b:>10.5f} {rms_a:>10.5f}")
            lines.append("")

        # Statistics
        lines.append("[Correction Statistics]")
        raw_vals = pd.to_numeric(df.get("diff_mag_raw", pd.Series()), errors="coerce").to_numpy(float)
        corr_vals = pd.to_numeric(df.get("diff_mag_corr", pd.Series()), errors="coerce").to_numpy(float)
        raw_vals = raw_vals[np.isfinite(raw_vals)]
        corr_vals = corr_vals[np.isfinite(corr_vals)]

        if raw_vals.size > 0:
            lines.append(f"  Raw:  mean={np.mean(raw_vals):.5f}, std={np.std(raw_vals):.5f}, "
                        f"min={np.min(raw_vals):.5f}, max={np.max(raw_vals):.5f}")
        if corr_vals.size > 0:
            lines.append(f"  Corr: mean={np.mean(corr_vals):.5f}, std={np.std(corr_vals):.5f}, "
                        f"min={np.min(corr_vals):.5f}, max={np.max(corr_vals):.5f}")

        if raw_vals.size > 0 and corr_vals.size > 0:
            rms_improve = (1 - np.std(corr_vals) / np.std(raw_vals)) * 100
            lines.append(f"  RMS improvement: {rms_improve:+.1f}%")
        lines.append("")

        # Residual statistics
        if "residual" in df.columns:
            resid = pd.to_numeric(df["residual"], errors="coerce").to_numpy(float)
            resid = resid[np.isfinite(resid)]
            if resid.size > 0:
                lines.append("[Residual Statistics (Δm_corr - median)]")
                lines.append(f"  mean={np.mean(resid):.6f}, std={np.std(resid):.6f}")
                lines.append(f"  median={np.median(resid):.6f}, MAD={np.median(np.abs(resid - np.median(resid))):.6f}")
                pct = np.percentile(resid, [5, 25, 50, 75, 95])
                lines.append(f"  percentiles: 5%={pct[0]:.5f}, 25%={pct[1]:.5f}, 50%={pct[2]:.5f}, "
                            f"75%={pct[3]:.5f}, 95%={pct[4]:.5f}")
                lines.append("")

        # Sigma clipping info
        lines.append("[Sigma Clipping]")
        lines.append(f"  Enabled: {self.sigma_clip}")
        if self.sigma_clip:
            lines.append(f"  Threshold: {self.clip_sigma}σ")
            lines.append(f"  Iterations: {self.clip_iters}")
        lines.append("")

        # Phase info
        if self.phase_period > 0:
            lines.append("[Phase Folding]")
            lines.append(f"  Period: {self.phase_period:.6f} days")
            lines.append(f"  T₀: {self.phase_t0:.6f} JD")
            lines.append("")

        lines.append("=" * 60)
        lines.append("Generated by AAPKI Step 12: Detrend & Night Merge")
        lines.append("=" * 60)

        path.write_text("\n".join(lines), encoding="utf-8")

    def revert_raw(self):
        self.corrected_df = pd.DataFrame()
        self.params_df = pd.DataFrame()
        self._update_results_table()
        self._update_plots()

    def validate_step(self) -> bool:
        return True

    def save_state(self):
        state_data = {
            "datasets": [str(p) for _, p in self.datasets],
            "target_id": self.target_edit.text().strip(),
            "mode": self.mode,
            "clip_sigma": self.clip_sigma,
            "clip_iters": self.clip_iters,
            "sigma_clip": self.sigma_clip,
            "x_axis_mode": self.x_axis_mode,
            "phase_period": self.phase_period,
            "phase_t0": self.phase_t0,
            "phase_cycles": self.phase_cycles,
            "color_map_by_filter": self.color_map_by_filter,
            "color_by": self.color_by,
            "use_global_k2": self.chk_global_k2.isChecked() if hasattr(self, "chk_global_k2") else True,
            "global_min_comps": self.global_min_comps,
            "global_sigma": self.global_sigma,
            "global_iters": self.global_iters,
            "global_rms_pct": self.global_rms_pct,
            "global_rms_threshold": self.global_rms_threshold,
            "global_frame_sigma": self.global_frame_sigma,
            "global_gauge": self.global_gauge,
            "global_robust": self.global_robust,
            "global_interp_missing": self.global_interp_missing,
            "global_normalize": self.global_normalize,
        }
        self.project_state.store_step_data("detrend_merge", state_data)

    def restore_state(self):
        state_data = self.project_state.get_step_data("detrend_merge")
        if state_data:
            for path in state_data.get("datasets", []):
                self._add_dataset(Path(path))
            self.target_edit.setText(state_data.get("target_id", ""))
            self.mode = state_data.get("mode", "offset")
            self.clip_sigma = float(state_data.get("clip_sigma", 3.0))
            self.clip_iters = int(state_data.get("clip_iters", 2))
            self.sigma_clip = bool(state_data.get("sigma_clip", True))
            self.x_axis_mode = state_data.get("x_axis_mode", self.x_axis_mode)
            self.phase_period = float(state_data.get("phase_period", 0.0))
            self.phase_t0 = float(state_data.get("phase_t0", 0.0))
            self.phase_cycles = float(state_data.get("phase_cycles", self.phase_cycles))
            self.color_map_by_filter = self._normalize_color_map(state_data.get("color_map_by_filter", {}))
            self.color_by = state_data.get("color_by", self.color_by)
            use_global_k2 = state_data.get("use_global_k2", True)
            if hasattr(self, "chk_global_k2"):
                self.chk_global_k2.setChecked(bool(use_global_k2))
            self.global_min_comps = int(state_data.get("global_min_comps", self.global_min_comps))
            self.global_sigma = float(state_data.get("global_sigma", self.global_sigma))
            self.global_iters = int(state_data.get("global_iters", self.global_iters))
            self.global_rms_pct = float(state_data.get("global_rms_pct", self.global_rms_pct))
            self.global_rms_threshold = float(state_data.get("global_rms_threshold", self.global_rms_threshold))
            self.global_frame_sigma = float(state_data.get("global_frame_sigma", self.global_frame_sigma))
            self.global_gauge = str(state_data.get("global_gauge", self.global_gauge))
            self.global_robust = bool(state_data.get("global_robust", self.global_robust))
            self.global_interp_missing = bool(state_data.get("global_interp_missing", self.global_interp_missing))
            self.global_normalize = bool(state_data.get("global_normalize", self.global_normalize))

        if not self.color_map_by_filter:
            self.color_map_by_filter = self._normalize_color_map(
                getattr(self.params.P, "lightcurve_color_index_by_filter", {})
            )

        # Sync UI
        if self.mode == "color":
            self.mode_color.setChecked(True)
        elif self.mode == "global":
            self.mode_global.setChecked(True)
        else:
            self.mode = "offset"
            self.mode_offset.setChecked(True)

        self.chk_clip.setChecked(self.sigma_clip)
        self.spin_clip.setValue(self.clip_sigma)
        self.spin_iters.setValue(self.clip_iters)
        self.spin_period.setValue(self.phase_period)
        self.spin_t0.setValue(self.phase_t0)
        if hasattr(self, "spin_cycles"):
            self.spin_cycles.setValue(self.phase_cycles)
        if hasattr(self, "phase_mode_combo"):
            self.phase_mode_combo.setCurrentIndex(1 if self.x_axis_mode == "phase" else 0)
        if hasattr(self, "color_by_combo"):
            self.color_by_combo.setCurrentText(self.color_by)
        if hasattr(self, "spin_global_min_comps"):
            self.spin_global_min_comps.setValue(self.global_min_comps)
        if hasattr(self, "spin_global_sigma"):
            self.spin_global_sigma.setValue(self.global_sigma)
        if hasattr(self, "spin_global_iters"):
            self.spin_global_iters.setValue(self.global_iters)
        if hasattr(self, "spin_global_rms_pct"):
            self.spin_global_rms_pct.setValue(self.global_rms_pct)
        if hasattr(self, "spin_global_rms_thr"):
            self.spin_global_rms_thr.setValue(self.global_rms_threshold)
        if hasattr(self, "spin_global_frame_sigma"):
            self.spin_global_frame_sigma.setValue(self.global_frame_sigma)
        if hasattr(self, "combo_global_gauge"):
            self.combo_global_gauge.setCurrentText(self.global_gauge)
        if hasattr(self, "chk_global_robust"):
            self.chk_global_robust.setChecked(self.global_robust)
        if hasattr(self, "chk_global_interp"):
            self.chk_global_interp.setChecked(self.global_interp_missing)
        if hasattr(self, "chk_global_normalize"):
            self.chk_global_normalize.setChecked(self.global_normalize)
