"""Tools-native multi-night merger workflow."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QGroupBox,
    QListWidget,
    QListWidgetItem,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QTextEdit,
    QFileDialog,
    QMessageBox,
    QLineEdit,
    QComboBox,
    QAbstractItemView,
    QCheckBox,
    QDoubleSpinBox,
    QSpinBox,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QFont

from ...analysis.light_curve.period_analysis_service import run_period_analysis
from ...analysis.light_curve.period_io_service import (
    load_period_lightcurve_csv,
    save_period_analysis_outputs,
)
from ...analysis.merge.id_match import (
    extract_row_float,
    reconcile_workspace_catalogs,
)
from ...analysis.merge.workspace_build import materialize_merged_workspace
from ...analysis.merge.workspace_scan import (
    default_merged_output_dir as _default_output_dir,
    folder_tag as _folder_tag,
    load_master_catalogs_by_filter as _load_master_catalogs_by_filter,
    load_selection_payloads as _load_selection_payloads,
    normalize_filter_key as _normalize_filter_key,
    read_step5_index as _read_step5_index,
    scan_merge_input_workspace,
    workspace_scan_signature,
)
from ...core.project_state import ProjectState
from ...utils.io_utils import (
    coerce_int64_source_id,
)
from ...utils.run_workspace import (
    build_merged_workspace_dir,
)
from ...utils.step_paths import (
    step9_selection_dir,
    step10_dir,
    step11_dir,
    step12_period_dir,
    find_best_lightcurve_csv,
)
class _MergedParamsProxy:
    """Read-only-ish params wrapper for merged runtime windows."""

    def __init__(self, base_params, merged_result_dir: Path):
        self._base = base_params
        self.P = copy.deepcopy(base_params.P)
        self.P.result_dir = Path(merged_result_dir)
        self.P.cache_dir = Path(merged_result_dir) / "cache"
        if not hasattr(self.P, "file_path_map"):
            self.P.file_path_map = {}

    def __getattr__(self, name):
        return getattr(self._base, name)

    def save_toml(self):
        return False


class _MergedFileManagerProxy:
    def __init__(self, filenames: list[str], night_assignments: dict[str, int], path_map: dict[str, str] | None = None):
        self.filenames = list(filenames)
        self.night_assignments = dict(night_assignments)
        self.path_map = dict(path_map or {})

    def get_file_path(self, filename: str) -> Path | None:
        p = self.path_map.get(filename)
        return Path(p) if p else None


class _MergerPeriodWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(
        self,
        lc_data: dict,
        min_period: float,
        max_period: float,
        samples_per_peak: int,
        methods: list[str],
        pdm_n_bins: int,
    ):
        super().__init__()
        self.lc_data = lc_data
        self.min_period = min_period
        self.max_period = max_period
        self.samples_per_peak = samples_per_peak
        self.methods = methods
        self.pdm_n_bins = pdm_n_bins

    def run(self):
        try:
            results = run_period_analysis(
                time=self.lc_data["time"],
                mag_raw=self.lc_data["mag_raw"],
                mag_corr=self.lc_data["mag_corr"],
                mag_err=self.lc_data["mag_err"],
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


class _MergerStep10Worker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, window, target_id: int, comp_ids: list[int]):
        super().__init__()
        self.window = window
        self.target_id = int(target_id)
        self.comp_ids = [int(c) for c in comp_ids]

    def run(self):
        original_log = getattr(self.window, "log", None)

        def _worker_log(msg, *args, **kwargs):
            self.progress.emit(str(msg))

        try:
            self.window.log = _worker_log
            summary = self.window._build_light_curve_core(self.target_id, list(self.comp_ids))
            self.finished.emit(summary or {})
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")
        finally:
            if original_log is not None:
                self.window.log = original_log


class _MergerStep11Worker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(
        self,
        window,
        selected_dates: set[str],
        use_global_k2: bool,
        target_id: int | None,
        comp_ids: list[int],
    ):
        super().__init__()
        self.window = window
        self.selected_dates = set(selected_dates)
        self.use_global_k2 = bool(use_global_k2)
        self.target_id = int(target_id) if target_id is not None else None
        self.comp_ids = [int(c) for c in comp_ids]

    def run(self):
        original_log = getattr(self.window, "log", None)

        def _worker_log(msg, *args, **kwargs):
            self.progress.emit(str(msg))

        try:
            self.window.log = _worker_log
            self.window.fit_and_apply(
                update_ui=False,
                save_outputs=False,
                selected_dates=self.selected_dates,
                use_global_k2=self.use_global_k2,
                target_id_override=self.target_id,
                comp_ids_override=self.comp_ids,
                sync_controls=False,
            )
            self.finished.emit(
                {
                    "mode": str(getattr(self.window, "mode", "offset")),
                    "n_params": int(len(getattr(self.window, "params_df", []))),
                    "n_points": int(len(getattr(self.window, "corrected_df", []))),
                }
            )
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")
        finally:
            if original_log is not None:
                self.window.log = original_log


class MultiNightMergerWindow(QMainWindow):
    """Merger workflow for previously processed result folders."""

    STEP_TITLES = [
        "Step 1  폴더 선택",
        "Step 2  ID 매칭",
        "Step 3  선택",
        "Step 4  Light Curve",
        "Step 5  Detrend",
        "Step 6  Period",
    ]

    # For StepWindowBase child reuse.
    step_names = [
        "File Selection",
        "Crop",
        "Sky Preview",
        "Source Detection",
        "Photometry",
        "WCS",
        "Ref Build",
        "ID Match",
        "Selection",
        "Light Curve",
        "Detrend",
        "Period Analysis",
    ]

    def __init__(self, params, project_state, main_window):
        super().__init__()
        self.params = params
        self.project_state = project_state
        self.main_window = main_window
        self.current_step_window = None

        self.folders: list[Path] = [Path(params.P.result_dir)]
        self.folder_tags: dict[str, str] = {}
        self.folder_scan_rows: list[dict] = []
        self._workspace_scan_cache: dict[str, tuple[tuple, dict]] = {}
        self._catalog_cache: dict[str, tuple[tuple, dict[str, pd.DataFrame]]] = {}
        self._selection_payload_cache: dict[str, tuple[tuple, dict[str, dict]]] = {}
        self._id_match_cache: dict[tuple, dict] = {}

        self.match_summary_rows: list[dict] = []
        self.match_records: list[dict] = []
        self.merged_catalogs: dict[str, pd.DataFrame] = {}
        self.local_id_maps: dict[str, dict[str, dict[int, dict[str, int]]]] = {}
        self.base_selection_by_filter: dict[str, dict] = {}

        self.selection_target_by_filter: dict[str, int | None] = {}
        self.selection_comp_by_filter: dict[str, set[int]] = {}
        self.selection_check_by_filter: dict[str, int | None] = {}
        self._selection_row_to_sid: dict[int, int] = {}
        self._selection_filter_ready = False

        self.merged_result_dir: Path | None = None
        self.merged_runtime_params = None
        self.merged_runtime_project_state: ProjectState | None = None
        self.merged_runtime_file_manager = None
        self.step10_runtime_window = None
        self.step11_runtime_window = None
        self.step10_worker: _MergerStep10Worker | None = None
        self.step11_worker: _MergerStep11Worker | None = None
        self.step12_worker: _MergerPeriodWorker | None = None
        self.step12_lc_data: dict | None = None

        self.setWindowTitle("Multi-Night Merger Workflow")
        self.resize(1200, 820)
        self._setup_ui()

    # ───────────────────────── UI ─────────────────────────

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        header = QHBoxLayout()
        btn_back = QPushButton("← 메인으로")
        btn_back.setFixedWidth(110)
        btn_back.clicked.connect(self._go_back)
        header.addWidget(btn_back)

        title = QLabel("Multi-Night Merger Workflow")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        header.addWidget(title, 1)
        header.addSpacing(110)
        root.addLayout(header)

        step_bar = QHBoxLayout()
        step_bar.setSpacing(2)
        self._step_btns: list[QPushButton] = []
        for i, text in enumerate(self.STEP_TITLES):
            btn = QPushButton(text)
            btn.setCheckable(True)
            btn.setMinimumHeight(30)
            btn.clicked.connect(lambda checked, idx=i: self._go_to_step(idx))
            step_bar.addWidget(btn)
            self._step_btns.append(btn)
        root.addLayout(step_bar)

        self._pages: list[QWidget] = [
            self._make_step1(),
            self._make_step2(),
            self._make_step3(),
            self._make_step4(),
            self._make_step5(),
            self._make_step6(),
        ]

        self._page_container = QWidget()
        self._page_layout = QVBoxLayout(self._page_container)
        self._page_layout.setContentsMargins(0, 0, 0, 0)
        for page in self._pages:
            self._page_layout.addWidget(page)
            page.hide()
        root.addWidget(self._page_container, 1)

        nav = QHBoxLayout()
        self.btn_prev = QPushButton("◀ 이전")
        self.btn_prev.clicked.connect(self._prev_step)
        self.btn_next = QPushButton("다음 ▶")
        self.btn_next.clicked.connect(self._next_step)
        self.btn_next.setStyleSheet(
            "QPushButton { background:#1565C0; color:white; font-weight:bold; padding:4px 16px; }"
            "QPushButton:hover { background:#0D47A1; }"
        )
        nav.addWidget(self.btn_prev)
        nav.addStretch()
        nav.addWidget(self.btn_next)
        root.addLayout(nav)

        self._current_step = 0
        self._refresh_output_dir_default(force=True)
        self._refresh_folder_list()
        self._go_to_step(0)

    def _make_info_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet("QLabel { background:#E3F2FD; padding:8px; border-radius:4px; }")
        return label

    def _make_step1(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        layout.addWidget(self._make_info_label(
            "RESULT_* 또는 MERGED_* workspace 폴더들을 선택합니다.\n"
            "이후 MERGED_<target>_<start>_<end> workspace를 새로 생성합니다."
        ))

        grp = QGroupBox("입력 result 폴더")
        grp_layout = QVBoxLayout(grp)
        self.folder_list = QListWidget()
        self.folder_list.setMinimumHeight(180)
        grp_layout.addWidget(self.folder_list)

        btn_row = QHBoxLayout()
        btn_add = QPushButton("폴더 추가")
        btn_add.clicked.connect(self._on_add_folder)
        btn_remove = QPushButton("선택 제거")
        btn_remove.clicked.connect(self._on_remove_folder)
        btn_up = QPushButton("위로")
        btn_up.clicked.connect(lambda: self._move_selected_folder(-1))
        btn_down = QPushButton("아래로")
        btn_down.clicked.connect(lambda: self._move_selected_folder(+1))
        btn_row.addWidget(btn_add)
        btn_row.addWidget(btn_remove)
        btn_row.addWidget(btn_up)
        btn_row.addWidget(btn_down)
        btn_row.addStretch()
        grp_layout.addLayout(btn_row)
        layout.addWidget(grp)

        out_grp = QGroupBox("출력 merged result 폴더")
        out_layout = QHBoxLayout(out_grp)
        self.output_dir_edit = QLineEdit()
        btn_browse_out = QPushButton("찾기...")
        btn_browse_out.clicked.connect(self._browse_output_dir)
        out_layout.addWidget(self.output_dir_edit, 1)
        out_layout.addWidget(btn_browse_out)
        layout.addWidget(out_grp)

        info_grp = QGroupBox("폴더 스캔")
        info_layout = QVBoxLayout(info_grp)
        self.folder_info_table = QTableWidget(0, 10)
        self.folder_info_table.setHorizontalHeaderLabels(["폴더", "Label", "Type", "Start", "End", "Step 5", "Step 9", "Step 10", "필터", "상태"])
        self.folder_info_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.folder_info_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.folder_info_table.setMinimumHeight(180)
        info_layout.addWidget(self.folder_info_table)
        btn_scan = QPushButton("폴더 스캔")
        btn_scan.clicked.connect(self._scan_folders)
        info_layout.addWidget(btn_scan)
        layout.addWidget(info_grp)
        layout.addStretch()
        return page

    def _make_step2(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        layout.addWidget(self._make_info_label(
            "base 폴더의 selection / master catalog를 기준으로 canonical merged ID를 만듭니다.\n"
            "매칭 우선순위는 Gaia source_id → 기존 canonical source_id → 좌표 근접 매칭입니다."
        ))

        top = QHBoxLayout()
        top.addWidget(QLabel("Position match radius (arcsec):"))
        self.match_radius_combo = QComboBox()
        for value in ("1.0", "1.5", "2.0", "3.0", "5.0"):
            self.match_radius_combo.addItem(value)
        self.match_radius_combo.setCurrentText("2.0")
        top.addWidget(self.match_radius_combo)
        top.addStretch()
        btn_match = QPushButton("ID 매칭 실행")
        btn_match.clicked.connect(self._run_id_match)
        top.addWidget(btn_match)
        layout.addLayout(top)

        self.match_status_label = QLabel("매칭 결과: 아직 실행 안 됨")
        self.match_status_label.setStyleSheet("QLabel { background:#FAFAFA; padding:6px; border-radius:4px; }")
        layout.addWidget(self.match_status_label)

        self.match_table = QTableWidget(0, 7)
        self.match_table.setHorizontalHeaderLabels(
            ["폴더", "필터", "Exact SID", "Positional", "New", "총 rows", "상태"]
        )
        self.match_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.match_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.match_table.setMinimumHeight(240)
        layout.addWidget(self.match_table)

        self.match_log = QTextEdit()
        self.match_log.setReadOnly(True)
        self.match_log.setMinimumHeight(160)
        self.match_log.setStyleSheet("font-size:8pt; font-family:monospace;")
        layout.addWidget(self.match_log)
        return page

    def _make_step3(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        layout.addWidget(self._make_info_label(
            "merged canonical catalog에서 target / comparison / check를 다시 선택합니다.\n"
            "기본값은 base 폴더 selection을 가져오고, 저장 시 merged workspace가 materialize 됩니다."
        ))

        top = QHBoxLayout()
        top.addWidget(QLabel("Filter:"))
        self.selection_filter_combo = QComboBox()
        self.selection_filter_combo.currentIndexChanged.connect(self._on_selection_filter_changed)
        top.addWidget(self.selection_filter_combo)
        top.addStretch()

        btn_load_base = QPushButton("Base Selection 불러오기")
        btn_load_base.clicked.connect(self._load_selection_defaults_from_base)
        top.addWidget(btn_load_base)

        btn_build_workspace = QPushButton("Merged Workspace 생성")
        btn_build_workspace.setStyleSheet(
            "QPushButton { background:#4CAF50; color:white; font-weight:bold; padding:4px 12px; }"
            "QPushButton:hover { background:#388E3C; }"
        )
        btn_build_workspace.clicked.connect(self._build_merged_workspace)
        top.addWidget(btn_build_workspace)
        layout.addLayout(top)

        self.selection_status_label = QLabel("선택 상태: 아직 merged catalog 없음")
        self.selection_status_label.setStyleSheet("QLabel { background:#FAFAFA; padding:6px; border-radius:4px; }")
        layout.addWidget(self.selection_status_label)

        role_row = QHBoxLayout()
        btn_t = QPushButton("Target")
        btn_t.clicked.connect(self._set_selection_target)
        btn_c = QPushButton("Comp")
        btn_c.clicked.connect(self._toggle_selection_comp)
        btn_k = QPushButton("Check")
        btn_k.clicked.connect(self._set_selection_check)
        btn_clear = QPushButton("Clear All")
        btn_clear.clicked.connect(self._clear_selection_roles)
        role_row.addWidget(btn_t)
        role_row.addWidget(btn_c)
        role_row.addWidget(btn_k)
        role_row.addWidget(btn_clear)
        role_row.addStretch()
        layout.addLayout(role_row)

        self.selection_table = QTableWidget(0, 7)
        self.selection_table.setHorizontalHeaderLabels(
            ["ID", "source_id", "Gaia", "Gmag", "folders", "role", "note"]
        )
        self.selection_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.selection_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.selection_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.selection_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.selection_table.setMinimumHeight(300)
        layout.addWidget(self.selection_table)

        self.selection_log = QTextEdit()
        self.selection_log.setReadOnly(True)
        self.selection_log.setMinimumHeight(140)
        self.selection_log.setStyleSheet("font-size:8pt; font-family:monospace;")
        layout.addWidget(self.selection_log)
        return page

    def _make_child_step_page(self, title: str, button_text: str, callback):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(self._make_info_label(title))
        status = QLabel("준비 안 됨")
        status.setStyleSheet("QLabel { background:#FAFAFA; padding:6px; border-radius:4px; }")
        layout.addWidget(status)
        btn = QPushButton(button_text)
        btn.clicked.connect(callback)
        btn.setStyleSheet(
            "QPushButton { background:#1565C0; color:white; font-weight:bold; padding:6px 18px; }"
            "QPushButton:hover { background:#0D47A1; }"
        )
        layout.addWidget(btn)
        layout.addStretch()
        return page, status

    def _make_step4(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(self._make_info_label(
            "Merged workspace에서 직접 Step 10 raw light curve를 생성합니다.\n"
            "간단 실행은 여기서 처리하고, 세부 QC/플롯이 필요할 때만 Step 10 창을 엽니다."
        ))
        self.step10_status_label = QLabel("Merged workspace 없음")
        self.step10_status_label.setStyleSheet("QLabel { background:#FAFAFA; padding:6px; border-radius:4px; }")
        layout.addWidget(self.step10_status_label)

        btn_row = QHBoxLayout()
        self.btn_step10_run = QPushButton("빠른 생성 실행")
        self.btn_step10_run.setStyleSheet(
            "QPushButton { background:#2E7D32; color:white; font-weight:bold; padding:6px 18px; }"
            "QPushButton:hover { background:#1B5E20; }"
        )
        self.btn_step10_run.clicked.connect(self._run_step10_inline)
        btn_row.addWidget(self.btn_step10_run)

        btn_open = QPushButton("Step 10 창 열기")
        btn_open.clicked.connect(lambda: self.open_step(9))
        btn_row.addWidget(btn_open)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.step10_progress_label = QLabel("")
        self.step10_progress_label.setStyleSheet("QLabel { color:#1565C0; }")
        layout.addWidget(self.step10_progress_label)

        self.step10_log = QTextEdit()
        self.step10_log.setReadOnly(True)
        self.step10_log.setMinimumHeight(180)
        self.step10_log.setStyleSheet("font-size:8pt; font-family:monospace;")
        layout.addWidget(self.step10_log)
        return page

    def _make_step5(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(self._make_info_label(
            "Merged workspace에서 직접 Step 11 detrend를 실행합니다.\n"
            "빠른 실행은 여기서 처리하고, 세부 옵션 조정이 필요할 때만 Step 11 창을 엽니다."
        ))
        self.step11_status_label = QLabel("Merged workspace 없음")
        self.step11_status_label.setStyleSheet("QLabel { background:#FAFAFA; padding:6px; border-radius:4px; }")
        layout.addWidget(self.step11_status_label)

        opts = QGroupBox("빠른 보정 옵션")
        opts_layout = QHBoxLayout(opts)
        self.step11_mode_combo = QComboBox()
        self.step11_mode_combo.addItem("Offset", "offset")
        self.step11_mode_combo.addItem("Color", "color")
        self.step11_mode_combo.addItem("Global Ensemble", "global")
        self.step11_global_k2_quick = QCheckBox("Global k''")
        opts_layout.addWidget(QLabel("Mode"))
        opts_layout.addWidget(self.step11_mode_combo)
        opts_layout.addWidget(self.step11_global_k2_quick)
        opts_layout.addStretch()
        layout.addWidget(opts)

        btn_row = QHBoxLayout()
        self.btn_step11_run = QPushButton("빠른 보정 실행")
        self.btn_step11_run.setStyleSheet(
            "QPushButton { background:#6A1B9A; color:white; font-weight:bold; padding:6px 18px; }"
            "QPushButton:hover { background:#4A148C; }"
        )
        self.btn_step11_run.clicked.connect(self._run_step11_inline)
        btn_row.addWidget(self.btn_step11_run)

        btn_open = QPushButton("Step 11 창 열기")
        btn_open.clicked.connect(lambda: self.open_step(10))
        btn_row.addWidget(btn_open)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.step11_progress_label = QLabel("")
        self.step11_progress_label.setStyleSheet("QLabel { color:#6A1B9A; }")
        layout.addWidget(self.step11_progress_label)

        self.step11_log = QTextEdit()
        self.step11_log.setReadOnly(True)
        self.step11_log.setMinimumHeight(180)
        self.step11_log.setStyleSheet("font-size:8pt; font-family:monospace;")
        layout.addWidget(self.step11_log)
        return page

    def _make_step6(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(self._make_info_label(
            "Merged workspace에서 직접 Period Analysis를 실행합니다.\n"
            "간단한 분석은 여기서 바로 돌리고, 고급 옵션은 Step 12 창을 열어 계속 작업할 수 있습니다."
        ))

        self.step12_status_label = QLabel("Merged workspace 없음")
        self.step12_status_label.setStyleSheet("QLabel { background:#FAFAFA; padding:6px; border-radius:4px; }")
        layout.addWidget(self.step12_status_label)

        opts = QGroupBox("빠른 분석 옵션")
        opts_layout = QHBoxLayout(opts)
        self.step12_filter_combo = QComboBox()
        self.step12_filter_combo.setMinimumWidth(90)
        self.step12_filter_combo.addItem("(no data)")
        self.step12_min_period_spin = QDoubleSpinBox()
        self.step12_min_period_spin.setDecimals(6)
        self.step12_min_period_spin.setRange(1e-6, 1e6)
        self.step12_min_period_spin.setValue(0.01)
        self.step12_max_period_spin = QDoubleSpinBox()
        self.step12_max_period_spin.setDecimals(6)
        self.step12_max_period_spin.setRange(1e-6, 1e6)
        self.step12_max_period_spin.setValue(10.0)
        self.step12_samples_spin = QSpinBox()
        self.step12_samples_spin.setRange(2, 1000)
        self.step12_samples_spin.setValue(10)
        self.step12_pdm_bins_spin = QSpinBox()
        self.step12_pdm_bins_spin.setRange(4, 100)
        self.step12_pdm_bins_spin.setValue(10)
        self.step12_chk_ls = QCheckBox("LS")
        self.step12_chk_ls.setChecked(True)
        self.step12_chk_pdm = QCheckBox("PDM")
        self.step12_chk_bls = QCheckBox("BLS")
        opts_layout.addWidget(QLabel("Filter"))
        opts_layout.addWidget(self.step12_filter_combo)
        opts_layout.addWidget(QLabel("Min P"))
        opts_layout.addWidget(self.step12_min_period_spin)
        opts_layout.addWidget(QLabel("Max P"))
        opts_layout.addWidget(self.step12_max_period_spin)
        opts_layout.addWidget(QLabel("Samples"))
        opts_layout.addWidget(self.step12_samples_spin)
        opts_layout.addWidget(QLabel("PDM bins"))
        opts_layout.addWidget(self.step12_pdm_bins_spin)
        opts_layout.addWidget(self.step12_chk_ls)
        opts_layout.addWidget(self.step12_chk_pdm)
        opts_layout.addWidget(self.step12_chk_bls)
        opts_layout.addStretch()
        layout.addWidget(opts)

        btn_row = QHBoxLayout()
        self.btn_step12_run = QPushButton("빠른 분석 실행")
        self.btn_step12_run.setStyleSheet(
            "QPushButton { background:#2E7D32; color:white; font-weight:bold; padding:6px 18px; }"
            "QPushButton:hover { background:#1B5E20; }"
        )
        self.btn_step12_run.clicked.connect(self._run_step12_headless)
        btn_row.addWidget(self.btn_step12_run)

        btn_open = QPushButton("Step 12 창 열기")
        btn_open.clicked.connect(lambda: self.open_step(11))
        btn_row.addWidget(btn_open)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.step12_progress_label = QLabel("")
        self.step12_progress_label.setStyleSheet("QLabel { color:#1565C0; }")
        layout.addWidget(self.step12_progress_label)

        self.step12_log = QTextEdit()
        self.step12_log.setReadOnly(True)
        self.step12_log.setMinimumHeight(180)
        self.step12_log.setStyleSheet("font-size:8pt; font-family:monospace;")
        layout.addWidget(self.step12_log)
        return page

    # ───────────────────────── folder scan ─────────────────────────

    def _refresh_folder_list(self):
        self.folder_list.clear()
        for i, p in enumerate(self.folders):
            label = f"[BASE] {p}" if i == 0 else str(p)
            item = QListWidgetItem(label)
            if i == 0:
                item.setForeground(QColor("#1565C0"))
            self.folder_list.addItem(item)

    def _invalidate_merger_state(self):
        self.folder_scan_rows = []
        self.match_summary_rows = []
        self.match_records = []
        self.merged_catalogs = {}
        self.local_id_maps = {}
        self.base_selection_by_filter = {}
        self.selection_target_by_filter = {}
        self.selection_comp_by_filter = {}
        self.selection_check_by_filter = {}
        self._selection_row_to_sid = {}
        self.merged_result_dir = None
        self.merged_runtime_params = None
        self.merged_runtime_project_state = None
        self.merged_runtime_file_manager = None
        self.step10_runtime_window = None
        self.step11_runtime_window = None
        self.step12_lc_data = None
        if hasattr(self, "folder_info_table"):
            self.folder_info_table.setRowCount(0)
        if hasattr(self, "match_table"):
            self.match_table.setRowCount(0)
        if hasattr(self, "selection_table"):
            self.selection_table.setRowCount(0)
        if hasattr(self, "selection_filter_combo"):
            self.selection_filter_combo.blockSignals(True)
            self.selection_filter_combo.clear()
            self.selection_filter_combo.blockSignals(False)
        if hasattr(self, "match_status_label"):
            self.match_status_label.setText("매칭 결과: 아직 실행 안 됨")
        if hasattr(self, "selection_status_label"):
            self.selection_status_label.setText("선택 상태: 아직 merged catalog 없음")
        if hasattr(self, "match_log"):
            self.match_log.clear()
        if hasattr(self, "selection_log"):
            self.selection_log.clear()
        if hasattr(self, "step12_log"):
            self.step12_log.clear()
        if hasattr(self, "step12_progress_label"):
            self.step12_progress_label.setText("")
        if hasattr(self, "step10_log"):
            self.step10_log.clear()
        if hasattr(self, "step11_log"):
            self.step11_log.clear()
        if hasattr(self, "step10_progress_label"):
            self.step10_progress_label.setText("")
        if hasattr(self, "step11_progress_label"):
            self.step11_progress_label.setText("")
        if hasattr(self, "step10_status_label"):
            self._refresh_runtime_status_labels()

    def _step9_signature(self, result_dir: Path) -> tuple:
        s9 = step9_selection_dir(result_dir)
        return (
            str(result_dir.resolve()),
            max((p.stat().st_mtime for p in s9.glob("master_catalog_*.tsv")), default=None),
            max((p.stat().st_mtime for p in s9.glob("selection_*.json")), default=None),
        )

    def _get_cached_selection_payloads(self, result_dir: Path) -> dict[str, dict]:
        cache_key = str(result_dir.resolve())
        signature = self._step9_signature(result_dir)
        cached = self._selection_payload_cache.get(cache_key)
        if cached and cached[0] == signature:
            return cached[1]
        payloads = _load_selection_payloads(result_dir)
        self._selection_payload_cache[cache_key] = (signature, payloads)
        return payloads

    def _get_cached_master_catalogs(self, result_dir: Path) -> dict[str, pd.DataFrame]:
        cache_key = str(result_dir.resolve())
        signature = self._step9_signature(result_dir)
        cached = self._catalog_cache.get(cache_key)
        if cached and cached[0] == signature:
            return cached[1]
        catalogs = _load_master_catalogs_by_filter(result_dir)
        self._catalog_cache[cache_key] = (signature, catalogs)
        return catalogs

    def _id_match_signature(self) -> tuple:
        return (
            tuple(str(folder.resolve()) for folder in self.folders),
            tuple(self._step9_signature(folder) for folder in self.folders),
            float(self.match_radius_combo.currentText()),
        )

    def _refresh_output_dir_default(self, force: bool = False):
        if not self.folders:
            self.output_dir_edit.clear()
            return
        new_default = build_merged_workspace_dir(self.folders)
        current_text = self.output_dir_edit.text().strip()
        if force or not current_text:
            self.output_dir_edit.setText(str(new_default))
            return
        try:
            current_path = Path(current_text)
        except Exception:
            current_path = None
        old_default = _default_output_dir(self.folders[0])
        if current_path is not None and current_path == old_default:
            self.output_dir_edit.setText(str(new_default))

    def _on_add_folder(self):
        start_dir = self.folders[0].parent if self.folders else Path(self.params.P.result_dir).parent
        folder = QFileDialog.getExistingDirectory(self, "result 폴더 선택", str(start_dir))
        if not folder:
            return
        p = Path(folder)
        if any(existing.resolve() == p.resolve() for existing in self.folders):
            return
        self.folders.append(p)
        self._invalidate_merger_state()
        self._refresh_folder_list()
        self._refresh_output_dir_default(force=True)

    def _on_remove_folder(self):
        row = self.folder_list.currentRow()
        if row < 0:
            return
        self.folders.pop(row)
        self._invalidate_merger_state()
        self._refresh_folder_list()
        self._refresh_output_dir_default(force=True)

    def _move_selected_folder(self, delta: int):
        row = self.folder_list.currentRow()
        if row < 0:
            return
        new_row = row + delta
        if new_row < 0 or new_row >= len(self.folders):
            return
        self.folders[row], self.folders[new_row] = self.folders[new_row], self.folders[row]
        self._invalidate_merger_state()
        self._refresh_folder_list()
        self.folder_list.setCurrentRow(new_row)
        self._refresh_output_dir_default(force=True)

    def _browse_output_dir(self):
        path = QFileDialog.getExistingDirectory(
            self,
            "MERGED workspace 폴더 선택",
            str(Path(self.output_dir_edit.text()).parent if self.output_dir_edit.text().strip() else build_merged_workspace_dir(self.folders).parent),
        )
        if path:
            current_name = build_merged_workspace_dir(self.folders).name if self.folders else "MERGED_workspace"
            self.output_dir_edit.setText(str(Path(path) / current_name))

    def _scan_folders(self):
        self.folder_scan_rows = []
        self.folder_info_table.setRowCount(0)
        for folder in self.folders:
            cache_key = str(folder.resolve())
            signature = workspace_scan_signature(folder)
            cached = self._workspace_scan_cache.get(cache_key)
            if cached and cached[0] == signature:
                row_info = dict(cached[1])
            else:
                row_info = scan_merge_input_workspace(folder)
                self._workspace_scan_cache[cache_key] = (signature, dict(row_info))
            self.folder_scan_rows.append(row_info)

            row = self.folder_info_table.rowCount()
            self.folder_info_table.insertRow(row)
            self.folder_info_table.setItem(row, 0, QTableWidgetItem(folder.name))
            self.folder_info_table.setItem(row, 1, QTableWidgetItem(str(row_info["label"])))
            self.folder_info_table.setItem(row, 2, QTableWidgetItem(str(row_info["run_type"])))
            self.folder_info_table.setItem(row, 3, QTableWidgetItem(str(row_info["date_start"])))
            self.folder_info_table.setItem(row, 4, QTableWidgetItem(str(row_info["date_end"])))
            for col_idx, key in enumerate(("has_step5", "has_step9", "has_step10"), start=5):
                ok = bool(row_info[key])
                item = QTableWidgetItem("OK" if ok else "없음")
                item.setForeground(QColor("#2E7D32") if ok else QColor("#C62828"))
                self.folder_info_table.setItem(row, col_idx, item)
            filters = list(row_info.get("filters") or [])
            merge_ready = bool(row_info.get("merge_ready"))
            self.folder_info_table.setItem(row, 8, QTableWidgetItem(", ".join(filters) if filters else "—"))
            status_item = QTableWidgetItem("사용 가능" if merge_ready else "입력 부족")
            status_item.setForeground(QColor("#2E7D32") if merge_ready else QColor("#C62828"))
            self.folder_info_table.setItem(row, 9, status_item)

    def _validate_selected_workspaces(self) -> tuple[bool, str]:
        if len(self.folders) < 2:
            return False, "Merge하려면 최소 2개의 RESULT/MERGED workspace를 선택하세요."
        if not self.folder_scan_rows:
            self._scan_folders()
        invalid_rows = [row for row in self.folder_scan_rows if not row.get("merge_ready")]
        if invalid_rows:
            return False, "Step 5 / Step 9 / Step 10이 모두 있는 workspace만 머저할 수 있습니다."
        labels = []
        seen = set()
        for row in self.folder_scan_rows:
            label = str(row.get("label") or "").strip()
            key = label.lower()
            if label and key not in seen:
                labels.append(label)
                seen.add(key)
        if len(labels) > 1:
            return False, "서로 다른 target/label의 RESULT를 한 번에 머저할 수 없습니다."
        return True, ""

    # ───────────────────────── Step 2: ID match ─────────────────────────

    def _run_id_match(self):
        self.match_log.clear()
        self.match_table.setRowCount(0)
        self.match_summary_rows = []
        self.match_records = []
        self.merged_catalogs = {}
        self.local_id_maps = {}
        self.base_selection_by_filter = {}

        if not self.folder_scan_rows:
            self._scan_folders()
        ok, msg = self._validate_selected_workspaces()
        if not ok:
            QMessageBox.warning(self, "ID Match", msg)
            return

        base_folder = self.folders[0]
        self.base_selection_by_filter = self._get_cached_selection_payloads(base_folder)
        catalogs_by_folder = {str(folder): self._get_cached_master_catalogs(folder) for folder in self.folders}
        self.folder_tags = {str(folder): _folder_tag(i, folder) for i, folder in enumerate(self.folders)}

        all_filters = sorted({
            flt for folder in self.folders
            for flt in catalogs_by_folder.get(str(folder), {}).keys()
        })
        if not all_filters:
            self.match_status_label.setText("매칭 실패: master_catalog 없음")
            self.match_log.append("[ERR] 어떤 폴더에서도 master_catalog_*.tsv 를 찾지 못했습니다.")
            return

        id_sig = self._id_match_signature()
        cached = self._id_match_cache.get(id_sig)
        if cached is None:
            cached = reconcile_workspace_catalogs(
                self.folders,
                catalogs_by_folder,
                self.folder_tags,
                float(self.match_radius_combo.currentText()),
                logger=self.match_log.append,
            )
            self._id_match_cache[id_sig] = cached
        else:
            self.match_log.append("[MATCH] Using cached ID match result")

        self.merged_catalogs = cached["canonical_by_filter"]
        self.local_id_maps = cached["local_id_maps"]
        self.match_summary_rows = cached["match_summary_rows"]
        self.match_records = cached["match_records"]
        self._update_match_table()
        self._load_selection_defaults_from_base()
        self._refresh_selection_filter_combo()
        n_rows = sum(len(df) for df in self.merged_catalogs.values())
        self.match_status_label.setText(
            f"매칭 완료: filters={len(self.merged_catalogs)} canonical rows={n_rows} mapping rows={len(self.match_records)}"
        )

    def _update_match_table(self):
        self.match_table.setRowCount(0)
        for row_data in self.match_summary_rows:
            row = self.match_table.rowCount()
            self.match_table.insertRow(row)
            self.match_table.setItem(row, 0, QTableWidgetItem(str(row_data["folder"])))
            self.match_table.setItem(row, 1, QTableWidgetItem(str(row_data["filter"])))
            self.match_table.setItem(row, 2, QTableWidgetItem(str(row_data["exact"])))
            self.match_table.setItem(row, 3, QTableWidgetItem(str(row_data["pos"])))
            self.match_table.setItem(row, 4, QTableWidgetItem(str(row_data["new"])))
            self.match_table.setItem(row, 5, QTableWidgetItem(str(row_data["total"])))
            status_item = QTableWidgetItem(str(row_data["status"]))
            status_item.setForeground(QColor("#2E7D32") if row_data["status"] == "OK" or row_data["status"] == "base" else QColor("#C62828"))
            self.match_table.setItem(row, 6, status_item)

    # ───────────────────────── Step 3: selection ─────────────────────────

    def _refresh_selection_filter_combo(self):
        self.selection_filter_combo.blockSignals(True)
        self.selection_filter_combo.clear()
        for flt in sorted(self.merged_catalogs.keys()):
            self.selection_filter_combo.addItem(flt)
        self.selection_filter_combo.blockSignals(False)
        self._selection_filter_ready = self.selection_filter_combo.count() > 0
        if self.selection_filter_combo.count():
            self.selection_filter_combo.setCurrentIndex(0)
            self._on_selection_filter_changed(0)
        else:
            self.selection_table.setRowCount(0)

    def _load_selection_defaults_from_base(self):
        self.selection_target_by_filter = {}
        self.selection_comp_by_filter = {}
        self.selection_check_by_filter = {}
        for flt, df in self.merged_catalogs.items():
            available = set(coerce_int64_source_id(df["source_id"]).dropna().astype("int64").tolist()) if "source_id" in df.columns else set()
            payload = self.base_selection_by_filter.get(flt, {})
            target_sid = payload.get("target_source_id")
            if target_sid is not None and int(target_sid) in available:
                self.selection_target_by_filter[flt] = int(target_sid)
            else:
                self.selection_target_by_filter[flt] = None
            comp_sids = set()
            for sid in payload.get("comparison_source_ids", []):
                if sid is not None and int(sid) in available:
                    comp_sids.add(int(sid))
            self.selection_comp_by_filter[flt] = comp_sids
            check_sid = payload.get("check_source_id")
            self.selection_check_by_filter[flt] = int(check_sid) if check_sid is not None and int(check_sid) in available else None
        self.selection_log.append("[SEL] Base selection loaded.")
        if self._selection_filter_ready:
            self._update_selection_table()

    def _current_selection_filter(self) -> str | None:
        if self.selection_filter_combo.count() <= 0:
            return None
        return _normalize_filter_key(self.selection_filter_combo.currentText())

    def _on_selection_filter_changed(self, index: int):
        if index < 0:
            return
        self._update_selection_table()

    def _role_for_sid(self, flt: str, sid: int) -> str:
        sid = int(sid)
        if self.selection_target_by_filter.get(flt) == sid:
            return "T"
        if sid in self.selection_comp_by_filter.get(flt, set()):
            return "C"
        if self.selection_check_by_filter.get(flt) == sid:
            return "K"
        return ""

    def _selected_sid_from_table(self) -> int | None:
        row = self.selection_table.currentRow()
        if row < 0:
            return None
        return self._selection_row_to_sid.get(row)

    def _update_selection_table(self):
        flt = self._current_selection_filter()
        if not flt or flt not in self.merged_catalogs:
            self.selection_table.setRowCount(0)
            return

        df = self.merged_catalogs[flt].copy()
        if "ID" in df.columns:
            df = df.sort_values("ID")
        self.selection_table.setRowCount(0)
        self._selection_row_to_sid = {}

        for _, row in df.iterrows():
            sid_val = coerce_int64_source_id(pd.Series([row.get("source_id")])).iloc[0]
            if pd.isna(sid_val):
                continue
            sid = int(sid_val)
            row_idx = self.selection_table.rowCount()
            self.selection_table.insertRow(row_idx)
            self._selection_row_to_sid[row_idx] = sid

            stable_id = pd.to_numeric(pd.Series([row.get("ID")]), errors="coerce").iloc[0]
            gaia_text = "Gaia" if sid > 0 else "Local"
            gmag = extract_row_float(row, "gaia_G", "gaia_g")
            folder_count = int(pd.to_numeric(pd.Series([row.get("folder_count", 1)]), errors="coerce").iloc[0] or 1)
            note = str(row.get("match_status", ""))

            self.selection_table.setItem(row_idx, 0, QTableWidgetItem(str(int(stable_id)) if np.isfinite(stable_id) else "—"))
            self.selection_table.setItem(row_idx, 1, QTableWidgetItem(str(sid)))
            self.selection_table.setItem(row_idx, 2, QTableWidgetItem(gaia_text))
            self.selection_table.setItem(row_idx, 3, QTableWidgetItem(f"{gmag:.3f}" if np.isfinite(gmag) else "—"))
            self.selection_table.setItem(row_idx, 4, QTableWidgetItem(str(folder_count)))
            self.selection_table.setItem(row_idx, 5, QTableWidgetItem(self._role_for_sid(flt, sid)))
            self.selection_table.setItem(row_idx, 6, QTableWidgetItem(note or ""))

        tgt = self.selection_target_by_filter.get(flt)
        comps = self.selection_comp_by_filter.get(flt, set())
        chk = self.selection_check_by_filter.get(flt)
        self.selection_status_label.setText(
            f"Filter {flt} | Target={tgt if tgt is not None else '—'} | "
            f"Comps={len(comps)} | Check={chk if chk is not None else '—'}"
        )

    def _set_selection_target(self):
        flt = self._current_selection_filter()
        sid = self._selected_sid_from_table()
        if not flt or sid is None:
            return
        self.selection_target_by_filter[flt] = int(sid)
        self.selection_comp_by_filter.setdefault(flt, set()).discard(int(sid))
        if self.selection_check_by_filter.get(flt) == int(sid):
            self.selection_check_by_filter[flt] = None
        self.selection_log.append(f"[SEL] {flt}: target={sid}")
        self._update_selection_table()

    def _toggle_selection_comp(self):
        flt = self._current_selection_filter()
        sid = self._selected_sid_from_table()
        if not flt or sid is None:
            return
        sid = int(sid)
        if self.selection_target_by_filter.get(flt) == sid:
            QMessageBox.information(self, "Selection", "Target cannot also be a comparison star.")
            return
        comps = self.selection_comp_by_filter.setdefault(flt, set())
        if sid in comps:
            comps.remove(sid)
            action = "removed"
        else:
            comps.add(sid)
            if self.selection_check_by_filter.get(flt) == sid:
                self.selection_check_by_filter[flt] = None
            action = "added"
        self.selection_log.append(f"[SEL] {flt}: comp {action} {sid}")
        self._update_selection_table()

    def _set_selection_check(self):
        flt = self._current_selection_filter()
        sid = self._selected_sid_from_table()
        if not flt or sid is None:
            return
        sid = int(sid)
        if self.selection_target_by_filter.get(flt) == sid:
            QMessageBox.information(self, "Selection", "Target cannot also be a check star.")
            return
        prev = self.selection_check_by_filter.get(flt)
        self.selection_check_by_filter[flt] = None if prev == sid else sid
        self.selection_comp_by_filter.setdefault(flt, set()).discard(sid)
        self.selection_log.append(f"[SEL] {flt}: check={self.selection_check_by_filter.get(flt)}")
        self._update_selection_table()

    def _clear_selection_roles(self):
        flt = self._current_selection_filter()
        if not flt:
            return
        self.selection_target_by_filter[flt] = None
        self.selection_comp_by_filter[flt] = set()
        self.selection_check_by_filter[flt] = None
        self.selection_log.append(f"[SEL] {flt}: cleared")
        self._update_selection_table()

    # ───────────────────────── merged workspace build ─────────────────────────

    def _build_merged_workspace(self):
        if not self.merged_catalogs:
            QMessageBox.warning(self, "Merged Workspace", "Step 2 ID match를 먼저 실행하세요.")
            return

        output_dir_text = self.output_dir_edit.text().strip()
        if not output_dir_text:
            QMessageBox.warning(self, "Merged Workspace", "출력 폴더를 지정하세요.")
            return

        out_dir = Path(output_dir_text)
        try:
            build_info = materialize_merged_workspace(
                out_dir=out_dir,
                folders=self.folders,
                folder_tags=self.folder_tags,
                local_id_maps=self.local_id_maps,
                merged_catalogs=self.merged_catalogs,
                selection_target_by_filter=self.selection_target_by_filter,
                selection_comp_by_filter=self.selection_comp_by_filter,
                selection_check_by_filter=self.selection_check_by_filter,
                match_records=self.match_records,
            )
        except RuntimeError as e:
            QMessageBox.warning(self, "Merged Workspace", str(e))
            return

        self.merged_result_dir = out_dir
        self._build_merged_runtime_context(
            build_info["night_assignments"],
            build_info["path_map"],
        )
        self._refresh_runtime_status_labels()
        self.selection_log.append(f"[BUILD] merged workspace ready: {out_dir}")
        QMessageBox.information(self, "Merged Workspace", f"생성 완료:\n{out_dir}")
        self._go_to_step(3)

    def _build_merged_runtime_context(self, night_assignments: dict[str, int], path_map: dict[str, str]):
        if self.merged_result_dir is None:
            return
        self.step10_runtime_window = None
        self.step11_runtime_window = None
        out_dir = Path(self.merged_result_dir)
        self.merged_runtime_params = _MergedParamsProxy(self.params, out_dir)
        self.merged_runtime_params.P.file_path_map = dict(path_map)
        idx = _read_step5_index(out_dir)
        filenames = idx["file"].astype(str).tolist() if not idx.empty and "file" in idx.columns else []
        self.merged_runtime_file_manager = _MergedFileManagerProxy(filenames, night_assignments, path_map)
        self.merged_runtime_project_state = ProjectState(out_dir)
        self.merged_runtime_project_state.state["project_name"] = out_dir.name
        self.merged_runtime_project_state.state["completed_steps"] = sorted(set(range(9)))
        self.merged_runtime_project_state.state["current_step"] = 9
        self.merged_runtime_project_state.save()

    # ───────────────────────── child workflow launch ─────────────────────────

    def _attach_runtime_log(self, window, append_fn, *, forward_original: bool = False):
        if getattr(window, "_merger_log_attached", False):
            return
        original_log = getattr(window, "log", None)

        def _wrapped_log(msg, *args, **kwargs):
            if forward_original and callable(original_log):
                original_log(msg, *args, **kwargs)
            text = str(msg)
            append_fn(text)

        window.log = _wrapped_log
        window._merger_log_attached = True

    def _step10_log_append(self, msg: str):
        if hasattr(self, "step10_log"):
            self.step10_log.append(msg)

    def _step11_log_append(self, msg: str):
        if hasattr(self, "step11_log"):
            self.step11_log.append(msg)

    def _get_or_create_step10_runtime_window(self):
        if self.step10_runtime_window is not None:
            return self.step10_runtime_window
        from ..workflow.step10_light_curve_builder import LightCurveBuilderWindow
        window = LightCurveBuilderWindow(
            self.merged_runtime_params,
            self.merged_runtime_file_manager,
            self.merged_runtime_project_state,
            self,
            runtime_mode=True,
        )
        window._auto_load_ids()
        window.show_log_window = lambda: None
        window.plot_current_comparison = lambda *args, **kwargs: None
        self._attach_runtime_log(window, self._step10_log_append, forward_original=False)
        self.step10_runtime_window = window
        return window

    def _get_or_create_step11_runtime_window(self):
        if self.step11_runtime_window is not None:
            return self.step11_runtime_window
        from ..workflow.step11_detrend_merge import DetrendNightMergeWindow
        window = DetrendNightMergeWindow(
            self.merged_runtime_params,
            self.merged_runtime_file_manager,
            self.merged_runtime_project_state,
            self,
            runtime_mode=True,
        )
        window._auto_load_ids()
        window._update_results_table = lambda *args, **kwargs: None
        self._attach_runtime_log(window, self._step11_log_append, forward_original=False)
        window.load_raw_data(silent=True)
        self.step11_runtime_window = window
        return window

    def _run_step10_inline(self):
        if self.merged_result_dir is None:
            QMessageBox.warning(self, "Light Curve", "Merged workspace를 먼저 생성하세요.")
            return
        if self.step10_worker is not None:
            QMessageBox.information(self, "Light Curve", "이미 Step10 생성이 진행 중입니다.")
            return
        from ..workflow.step10_light_curve_builder import _load_selection_ids, _safe_int_list
        self.btn_step10_run.setEnabled(False)
        self.step10_progress_label.setText("Building...")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            window = self._get_or_create_step10_runtime_window()
            self._step10_log_append(f"[MERGER] Building Step10 in {self.merged_result_dir}")
            target_text = window.target_edit.text().strip()
            comp_ids = _safe_int_list(window.comp_edit.text())
            if not target_text or not comp_ids:
                target_id, loaded_comp_ids = _load_selection_ids(self.merged_result_dir)
                if not target_text and target_id is not None:
                    target_text = str(int(target_id))
                if not comp_ids:
                    comp_ids = list(loaded_comp_ids)
            if not target_text:
                raise RuntimeError("대상 ID가 필요합니다.")
            if not comp_ids:
                raise RuntimeError("비교성 ID가 필요합니다.")
            window.target_edit.setText(str(target_text))
            window.comp_edit.setText(",".join(str(int(c)) for c in comp_ids))
            target_id = int(target_text)
            self.step10_worker = _MergerStep10Worker(window, target_id, comp_ids)
            self.step10_worker.progress.connect(self._on_step10_run_progress)
            self.step10_worker.finished.connect(self._on_step10_run_finished)
            self.step10_worker.error.connect(self._on_step10_run_error)
            self.step10_worker.start()
            self._refresh_runtime_status_labels()
        except Exception as e:
            self.step10_progress_label.setText("Error")
            self._step10_log_append(f"[ERROR] {e}")
            QMessageBox.warning(self, "Light Curve", str(e))
            QApplication.restoreOverrideCursor()
            self.btn_step10_run.setEnabled(True)

    def _run_step11_inline(self):
        if self.merged_result_dir is None:
            QMessageBox.warning(self, "Detrend", "Merged workspace를 먼저 생성하세요.")
            return
        if self.step11_worker is not None:
            QMessageBox.information(self, "Detrend", "이미 Step11 보정이 진행 중입니다.")
            return
        self.btn_step11_run.setEnabled(False)
        self.step11_progress_label.setText("Running...")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            window = self._get_or_create_step11_runtime_window()
            window.restore_state()
            mode = str(self.step11_mode_combo.currentData() or "offset")
            window.mode = mode
            if mode == "global":
                window.mode_global.setChecked(True)
            elif mode == "color":
                window.mode_color.setChecked(True)
            else:
                window.mode_offset.setChecked(True)
            if hasattr(window, "chk_global_k2"):
                window.chk_global_k2.setChecked(bool(self.step11_global_k2_quick.isChecked()))
            window._sync_state_from_controls()
            window._load_comp_selection()
            target_text = window.target_edit.text().strip()
            target_id = int(target_text) if target_text else None
            comp_ids = [int(c) for c in (window.comp_active_ids or window.comp_candidate_ids or [])]
            selected_dates = set(window._selected_dates())
            use_global_k2 = bool(window.chk_global_k2.isChecked()) if hasattr(window, "chk_global_k2") else False
            self._step11_log_append(f"[MERGER] Running Step11 mode={mode}")
            self.step11_worker = _MergerStep11Worker(
                window=window,
                selected_dates=selected_dates,
                use_global_k2=use_global_k2,
                target_id=target_id,
                comp_ids=comp_ids,
            )
            self.step11_worker.progress.connect(self._on_step11_run_progress)
            self.step11_worker.finished.connect(self._on_step11_run_finished)
            self.step11_worker.error.connect(self._on_step11_run_error)
            self.step11_worker.start()
            self._refresh_runtime_status_labels()
        except Exception as e:
            self.step11_progress_label.setText("Error")
            self._step11_log_append(f"[ERROR] {e}")
            QMessageBox.warning(self, "Detrend", str(e))
            QApplication.restoreOverrideCursor()
            self.btn_step11_run.setEnabled(True)

    def _on_step10_run_progress(self, msg: str):
        self._step10_log_append(msg)

    def _on_step10_run_finished(self, summary: dict):
        self.step10_worker = None
        QApplication.restoreOverrideCursor()
        self.btn_step10_run.setEnabled(True)
        self.step10_progress_label.setText("Done")
        try:
            if self.step10_runtime_window is not None:
                self.step10_runtime_window.save_state()
        except Exception as e:
            self._step10_log_append(f"[WARN] save_state failed: {e}")
        self.step11_runtime_window = None
        self.step12_lc_data = None
        self._refresh_runtime_status_labels()
        if summary:
            self._step10_log_append(
                "[MERGER] Step10 done: datasets={ds}, outputs={out}, valid={valid}/{total}".format(
                    ds=summary.get("n_datasets", 0),
                    out=summary.get("n_outputs", 0),
                    valid=summary.get("n_valid", 0),
                    total=summary.get("n_total", 0),
                )
            )

    def _on_step10_run_error(self, msg: str):
        self.step10_worker = None
        QApplication.restoreOverrideCursor()
        self.btn_step10_run.setEnabled(True)
        self.step10_progress_label.setText("Error")
        self._step10_log_append(f"[ERROR] {msg}")
        self._refresh_runtime_status_labels()
        QMessageBox.warning(self, "Light Curve", msg)

    def _on_step11_run_progress(self, msg: str):
        self._step11_log_append(msg)

    def _on_step11_run_finished(self, summary: dict):
        self.step11_worker = None
        QApplication.restoreOverrideCursor()
        self.btn_step11_run.setEnabled(True)
        window = self.step11_runtime_window
        if window is None:
            self.step11_progress_label.setText("Error")
            return
        try:
            if window.mode == "color":
                window.mode_color.setChecked(True)
            elif window.mode == "global":
                window.mode_global.setChecked(True)
            else:
                window.mode_offset.setChecked(True)
            window._populate_date_list()
            window._refresh_filter_combo(window.raw_df.get("filter", pd.Series([], dtype=str)).astype(str).tolist())
            window._refresh_delta_c_map()
            window._update_color_mode_enabled()
            window._update_results_table()
            window._update_plots()
            window._update_analysis_panel()
            if window.mode != "global":
                window._log_fit_summary()
            window._save_comprehensive_results()
            window.save_state()
            self.step12_lc_data = None
            self._refresh_runtime_status_labels()
            self.step11_progress_label.setText("Done")
            if summary:
                self._step11_log_append(
                    "[MERGER] Step11 done: mode={mode}, groups={groups}, points={pts}".format(
                        mode=summary.get("mode", window.mode),
                        groups=summary.get("n_params", 0),
                        pts=summary.get("n_points", 0),
                    )
                )
        except Exception as e:
            self.step11_progress_label.setText("Error")
            self._step11_log_append(f"[ERROR] finalize failed: {e}")
            self._refresh_runtime_status_labels()
            QMessageBox.warning(self, "Detrend", str(e))

    def _on_step11_run_error(self, msg: str):
        self.step11_worker = None
        QApplication.restoreOverrideCursor()
        self.btn_step11_run.setEnabled(True)
        self.step11_progress_label.setText("Error")
        self._step11_log_append(f"[ERROR] {msg}")
        self._refresh_runtime_status_labels()
        QMessageBox.warning(self, "Detrend", msg)

    def _refresh_runtime_status_labels(self):
        merged_dir = self.merged_result_dir
        if merged_dir is None:
            self.step10_status_label.setText("Merged workspace 없음")
            self.step11_status_label.setText("Merged workspace 없음")
            self.step12_status_label.setText("Merged workspace 없음")
            if hasattr(self, "btn_step10_run"):
                self.btn_step10_run.setEnabled(False)
            if hasattr(self, "btn_step11_run"):
                self.btn_step11_run.setEnabled(False)
            self._refresh_step12_filter_options()
            return

        s10 = step10_dir(merged_dir)
        s11 = step11_dir(merged_dir)
        s12 = step12_period_dir(merged_dir)

        s10_ready = bool(list(s10.glob("lightcurve_ID*_raw.csv"))) or bool(list(s10.glob("comp_selection.json")))
        s11_ready = bool(list(s11.glob("lightcurve_ID*_current.csv"))) or bool(list(s11.glob("lightcurve_ID*_global.csv")))
        s12_ready = bool(list(s12.glob("period_analysis_*_ID*.json")))

        self.step10_status_label.setText(
            f"Workspace: {merged_dir}\nStep10 outputs: {'있음' if s10_ready else '없음'}"
        )
        self.step11_status_label.setText(
            f"Workspace: {merged_dir}\nStep11 outputs: {'있음' if s11_ready else '없음'}"
        )
        self.step12_status_label.setText(
            f"Workspace: {merged_dir}\nStep12 outputs: {'있음' if s12_ready else '없음'}"
        )
        step10_busy = self.step10_worker is not None
        step11_busy = self.step11_worker is not None
        step12_busy = self.step12_worker is not None
        if hasattr(self, "btn_step10_run"):
            self.btn_step10_run.setEnabled(not step10_busy and not step11_busy and not step12_busy)
        if hasattr(self, "btn_step11_run"):
            self.btn_step11_run.setEnabled(not step10_busy and not step11_busy and not step12_busy)
        if hasattr(self, "btn_step12_run"):
            self.btn_step12_run.setEnabled(not step10_busy and not step11_busy and not step12_busy)
        self._refresh_step12_filter_options()

    def _merged_step12_target_id(self) -> int | None:
        if self.merged_result_dir is None:
            return None
        payloads = _load_selection_payloads(self.merged_result_dir)
        target_ids = {
            int(payload["target_id"])
            for payload in payloads.values()
            if payload.get("target_id") is not None
        }
        if len(target_ids) == 1:
            return next(iter(target_ids))
        return None

    def _refresh_step12_filter_options(self):
        if not hasattr(self, "step12_filter_combo"):
            return
        combo = self.step12_filter_combo
        combo.blockSignals(True)
        combo.clear()

        merged_dir = self.merged_result_dir
        target_id = self._merged_step12_target_id()
        if merged_dir is None or target_id is None:
            combo.addItem("(no data)")
            combo.blockSignals(False)
            self.btn_step12_run.setEnabled(False)
            return

        lc_path = find_best_lightcurve_csv(merged_dir, target_id)
        filters: list[str] = []
        if lc_path is not None and lc_path.exists():
            try:
                df = pd.read_csv(lc_path, usecols=lambda c: c == "filter")
                if "filter" in df.columns:
                    filters = sorted({
                        str(v).strip()
                        for v in df["filter"].dropna().astype(str).tolist()
                        if str(v).strip()
                    })
            except Exception:
                filters = []

        if not filters:
            combo.addItem("(no data)")
            self.btn_step12_run.setEnabled(False)
        else:
            combo.addItems(filters)
            self.btn_step12_run.setEnabled(True)
        combo.blockSignals(False)

    def _step12_log_append(self, msg: str):
        if hasattr(self, "step12_log"):
            self.step12_log.append(msg)

    def _run_step12_headless(self):
        merged_dir = self.merged_result_dir
        if merged_dir is None:
            QMessageBox.warning(self, "Period Analysis", "Merged workspace를 먼저 생성하세요.")
            return
        if self.step12_worker is not None and self.step12_worker.isRunning():
            return

        target_id = self._merged_step12_target_id()
        if target_id is None:
            QMessageBox.warning(self, "Period Analysis", "Merged target ID를 결정하지 못했습니다.")
            return
        flt = self.step12_filter_combo.currentText().strip()
        if not flt or flt == "(no data)":
            QMessageBox.warning(self, "Period Analysis", "분석할 필터가 없습니다.")
            return

        lc_path = find_best_lightcurve_csv(merged_dir, target_id)
        if lc_path is None or not lc_path.exists():
            QMessageBox.warning(self, "Period Analysis", "Step10/11 light curve를 찾지 못했습니다.")
            return

        min_period = self.step12_min_period_spin.value()
        max_period = self.step12_max_period_spin.value()
        if min_period >= max_period:
            QMessageBox.warning(self, "Period Analysis", "Min period must be less than max period.")
            return

        methods: list[str] = []
        if self.step12_chk_ls.isChecked():
            methods.append("ls")
        if self.step12_chk_pdm.isChecked():
            methods.append("pdm")
        if self.step12_chk_bls.isChecked():
            methods.append("bls")
        if not methods:
            QMessageBox.warning(self, "Period Analysis", "최소 1개 방법을 선택하세요.")
            return

        try:
            self.step12_lc_data = load_period_lightcurve_csv(lc_path, flt, target_id)
        except Exception as e:
            QMessageBox.warning(self, "Period Analysis", str(e))
            self._step12_log_append(f"[ERROR] {e}")
            return

        self.btn_step12_run.setEnabled(False)
        self.step12_progress_label.setText("Computing...")
        self._step12_log_append(f"Loading: {lc_path}")
        self._step12_log_append(
            f"Filter: {flt}, Target ID: {target_id}, "
            f"Detrend: {self.step12_lc_data.get('corr_mode_label', 'Unknown')}"
        )
        self.step12_worker = _MergerPeriodWorker(
            lc_data=self.step12_lc_data,
            min_period=min_period,
            max_period=max_period,
            samples_per_peak=self.step12_samples_spin.value(),
            methods=methods,
            pdm_n_bins=self.step12_pdm_bins_spin.value(),
        )
        self.step12_worker.progress.connect(self._on_step12_run_progress)
        self.step12_worker.finished.connect(self._on_step12_run_finished)
        self.step12_worker.error.connect(self._on_step12_run_error)
        self.step12_worker.start()

    def _on_step12_run_progress(self, msg: str):
        self.step12_progress_label.setText(msg)
        self._step12_log_append(msg)

    def _on_step12_run_error(self, msg: str):
        self.step12_worker = None
        self.btn_step12_run.setEnabled(True)
        self.step12_progress_label.setText("Error")
        self._step12_log_append(f"[ERROR] {msg}")
        QMessageBox.warning(self, "Period Analysis", msg)

    def _on_step12_run_finished(self, results: dict):
        self.step12_worker = None
        self.btn_step12_run.setEnabled(True)
        self.step12_progress_label.setText("Done")
        if self.merged_result_dir is None or self.step12_lc_data is None:
            return
        summary_path = save_period_analysis_outputs(
            result_dir=self.merged_result_dir,
            lc_data=self.step12_lc_data,
            results=results,
            min_period=self.step12_min_period_spin.value(),
            max_period=self.step12_max_period_spin.value(),
        )
        self._step12_log_append(f"Saved: {summary_path}")
        self._step12_log_append("Analysis complete")
        self._refresh_runtime_status_labels()

    def on_step_completed(self, step_index: int):
        self._refresh_runtime_status_labels()

    def open_step(self, step_index: int):
        if step_index <= 8:
            # Return to merger selection page.
            self.show()
            self.raise_()
            self.activateWindow()
            self._go_to_step(2)
            return

        if self.merged_result_dir is None:
            QMessageBox.warning(self, "Merged Workflow", "Merged workspace를 먼저 생성하세요.")
            return
        if self.merged_runtime_params is None or self.merged_runtime_project_state is None:
            QMessageBox.warning(self, "Merged Workflow", "Merged runtime context 초기화 실패")
            return

        if self.current_step_window is not None and self.current_step_window.isVisible():
            self.current_step_window.close()

        if step_index == 9:
            from ..workflow.step10_light_curve_builder import LightCurveBuilderWindow
            self.current_step_window = LightCurveBuilderWindow(
                self.merged_runtime_params,
                self.merged_runtime_file_manager,
                self.merged_runtime_project_state,
                self,
            )
        elif step_index == 10:
            from ..workflow.step11_detrend_merge import DetrendNightMergeWindow
            self.current_step_window = DetrendNightMergeWindow(
                self.merged_runtime_params,
                self.merged_runtime_file_manager,
                self.merged_runtime_project_state,
                self,
            )
        elif step_index == 11:
            from ..workflow.step12_period_analysis import PeriodAnalysisWindow
            self.current_step_window = PeriodAnalysisWindow(
                self.merged_runtime_params,
                self.merged_runtime_file_manager,
                self.merged_runtime_project_state,
                self,
            )
        else:
            return

        self.current_step_window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.current_step_window.show()
        self.current_step_window.raise_()
        self.current_step_window.activateWindow()

    # ───────────────────────── navigation ─────────────────────────

    def _go_to_step(self, idx: int):
        for i, (page, btn) in enumerate(zip(self._pages, self._step_btns)):
            page.setVisible(i == idx)
            btn.setChecked(i == idx)
            btn.setStyleSheet(
                "QPushButton { background:#1565C0; color:white; font-weight:bold; }"
                if i == idx else ""
            )
        self._current_step = idx
        self.btn_prev.setEnabled(idx > 0)
        self.btn_next.setEnabled(idx < len(self._pages) - 1)
        if idx >= 3:
            self._refresh_runtime_status_labels()

    def _prev_step(self):
        if self._current_step > 0:
            self._go_to_step(self._current_step - 1)

    def _next_step(self):
        if self._current_step == 0:
            ok, msg = self._validate_selected_workspaces()
            if not ok:
                QMessageBox.warning(self, "Merger", msg)
                return
        if self._current_step == 0 and not self.folders:
            QMessageBox.warning(self, "Merger", "폴더를 먼저 선택하세요.")
            return
        if self._current_step == 1 and not self.merged_catalogs:
            QMessageBox.warning(self, "Merger", "Step 2 ID 매칭을 먼저 실행하세요.")
            return
        if self._current_step == 2 and self.merged_result_dir is None:
            QMessageBox.warning(self, "Merger", "Merged workspace를 먼저 생성하세요.")
            return
        if self._current_step < len(self._pages) - 1:
            self._go_to_step(self._current_step + 1)

    # ───────────────────────── back ─────────────────────────

    def _go_back(self):
        if self.current_step_window is not None and self.current_step_window.isVisible():
            self.current_step_window.close()
        self.hide()
        self.main_window.show()
        self.main_window.raise_()
        self.main_window.activateWindow()

    def closeEvent(self, event):
        self._go_back()
        event.ignore()


class MultiNightMergerToolWindow(MultiNightMergerWindow):
    """Tool-facing alias kept for stable imports."""

    pass
