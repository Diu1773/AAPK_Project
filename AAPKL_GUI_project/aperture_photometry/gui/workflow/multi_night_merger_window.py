"""
Multi-night merger workflow.

Front steps are merger-specific:
  1. Select existing result folders
  2. Reconcile IDs across folders (Gaia first, positional fallback)
  3. Choose merged target / comp / check set

Back steps reuse the normal workflow on a materialized merged workspace:
  4. Step 10 Light Curve Builder
  5. Step 11 Detrend
  6. Step 12 Period Analysis
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

from PyQt5.QtWidgets import (
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
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont

from ...core.project_state import ProjectState
from ...utils.io_utils import (
    read_csv_int64_source_id,
    coerce_int64_source_id,
    load_file_path_map,
    load_headers_table,
    load_night_assignments,
)
from ...utils.photometry_loader import load_frame_photometry
from ...utils.run_workspace import (
    build_merged_workspace_dir,
    infer_result_workspace_date_range,
    infer_result_workspace_label,
    infer_workspace_date_range,
    infer_workspace_label,
    load_run_manifest,
    write_run_manifest,
)
from ...utils.step_paths import (
    step1_dir,
    step5_photometry_dir,
    step9_selection_dir,
    step10_dir,
    step11_dir,
    step12_period_dir,
)


def _normalize_filter_key(value) -> str:
    return str(value or "").strip().lower() or "unknown"


def _folder_tag(index: int, folder: Path) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in folder.name)
    return f"F{index + 1:02d}_{safe}"


def _default_output_dir(base_folder: Path) -> Path:
    return build_merged_workspace_dir([base_folder])


def _run_meta(result_dir: Path) -> dict:
    meta = load_run_manifest(result_dir)
    if meta:
        return meta
    start_date, end_date = infer_workspace_date_range(result_dir)
    label = infer_workspace_label(result_dir)
    return {
        "run_type": "result",
        "label": label,
        "date_start": start_date,
        "date_end": end_date,
        "result_dir": str(result_dir),
    }


def _read_step5_index(result_dir: Path) -> pd.DataFrame:
    idx_path = step5_photometry_dir(result_dir) / "photometry_index.csv"
    if not idx_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(idx_path)
    except Exception:
        return pd.DataFrame()


def _load_selection_payloads(result_dir: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    s9 = step9_selection_dir(result_dir)
    if not s9.exists():
        return out
    for path in sorted(s9.glob("selection_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        flt = _normalize_filter_key(data.get("filter") or path.stem.replace("selection_", ""))
        out[flt] = data
    return out


def _load_master_catalogs_by_filter(result_dir: Path) -> dict[str, pd.DataFrame]:
    catalogs: dict[str, pd.DataFrame] = {}
    s9 = step9_selection_dir(result_dir)
    if not s9.exists():
        return catalogs
    for path in sorted(s9.glob("master_catalog_*.tsv")):
        flt = _normalize_filter_key(path.stem.replace("master_catalog_", ""))
        try:
            df = read_csv_int64_source_id(path, sep="\t")
        except Exception:
            continue
        if df is None or df.empty or "ID" not in df.columns:
            continue
        df = df.copy()
        if "source_id" in df.columns:
            df["source_id"] = coerce_int64_source_id(df["source_id"]).astype("Int64")
        df["ID"] = pd.to_numeric(df["ID"], errors="coerce").astype("Int64")
        for col in ("ra_deg", "dec_deg", "gaia_G", "gaia_id"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        catalogs[flt] = df
    return catalogs


def _first_valid(series) -> float:
    vals = pd.to_numeric(series, errors="coerce")
    vals = vals[np.isfinite(vals)]
    return float(vals.iloc[0]) if len(vals) else float("nan")


def _extract_row_float(row: pd.Series, *cols: str) -> float:
    for col in cols:
        if col in row.index:
            val = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
            if np.isfinite(val):
                return float(val)
    return float("nan")


def _row_radec(row: pd.Series) -> tuple[float, float]:
    return (
        _extract_row_float(row, "ra_deg", "ra", "RA"),
        _extract_row_float(row, "dec_deg", "dec", "DEC"),
    )


def _append_folder_tag(existing: str, tag: str) -> str:
    parts = [p for p in str(existing or "").split(",") if p]
    if tag not in parts:
        parts.append(tag)
    return ",".join(parts)


def _load_target_radec(result_dir: Path, target_id: int) -> tuple[float, float]:
    catalogs = _load_master_catalogs_by_filter(result_dir)
    for df in catalogs.values():
        row = df[pd.to_numeric(df["ID"], errors="coerce") == int(target_id)]
        if row.empty:
            continue
        ra = _first_valid(row["ra_deg"]) if "ra_deg" in row.columns else float("nan")
        dec = _first_valid(row["dec_deg"]) if "dec_deg" in row.columns else float("nan")
        if np.isfinite(ra) and np.isfinite(dec):
            return ra, dec
    return float("nan"), float("nan")


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
        btn_row.addWidget(btn_add)
        btn_row.addWidget(btn_remove)
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
        self.folder_info_table = QTableWidget(0, 9)
        self.folder_info_table.setHorizontalHeaderLabels(["폴더", "Type", "Start", "End", "Step 5", "Step 9", "Step 10", "필터", "상태"])
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
        page, label = self._make_child_step_page(
            "Merged workspace의 Step 10 Light Curve Builder를 엽니다.",
            "Step 10 열기",
            lambda: self.open_step(9),
        )
        self.step10_status_label = label
        return page

    def _make_step5(self) -> QWidget:
        page, label = self._make_child_step_page(
            "Merged workspace의 Step 11 Detrend window를 엽니다.",
            "Step 11 열기",
            lambda: self.open_step(10),
        )
        self.step11_status_label = label
        return page

    def _make_step6(self) -> QWidget:
        page, label = self._make_child_step_page(
            "Merged workspace의 Step 12 Period Analysis window를 엽니다.",
            "Step 12 열기",
            lambda: self.open_step(11),
        )
        self.step12_status_label = label
        return page

    # ───────────────────────── folder scan ─────────────────────────

    def _refresh_folder_list(self):
        self.folder_list.clear()
        for i, p in enumerate(self.folders):
            label = f"[BASE] {p}" if i == 0 else str(p)
            item = QListWidgetItem(label)
            if i == 0:
                item.setForeground(QColor("#1565C0"))
                item.setFlags(item.flags() & ~Qt.ItemIsSelectable)
            self.folder_list.addItem(item)

    def _refresh_output_dir_default(self, force: bool = False):
        if not self.folders:
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
        folder = QFileDialog.getExistingDirectory(self, "result 폴더 선택", str(self.folders[0].parent))
        if not folder:
            return
        p = Path(folder)
        if any(existing.resolve() == p.resolve() for existing in self.folders):
            return
        self.folders.append(p)
        self._refresh_folder_list()
        self._refresh_output_dir_default(force=True)

    def _on_remove_folder(self):
        row = self.folder_list.currentRow()
        if row <= 0:
            return
        self.folders.pop(row)
        self._refresh_folder_list()
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
            meta = _run_meta(folder)
            idx = _read_step5_index(folder)
            catalogs = _load_master_catalogs_by_filter(folder)
            selection_payloads = _load_selection_payloads(folder)
            s10 = step10_dir(folder)
            has_step10 = bool(list(s10.glob("lightcurve_ID*_raw.csv"))) or bool(list(s10.glob("lightcurve_combined_ID*_raw.csv")))
            has_step5 = not idx.empty
            has_step9 = bool(catalogs) and bool(selection_payloads)
            merge_ready = bool(has_step5 and has_step9 and has_step10)

            filters = sorted(set(catalogs) | set(selection_payloads))
            row_info = {
                "folder": folder,
                "run_type": str(meta.get("run_type", "result")),
                "date_start": meta.get("date_start") or "—",
                "date_end": meta.get("date_end") or "—",
                "has_step5": has_step5,
                "has_step9": has_step9,
                "has_step10": has_step10,
                "filters": filters,
                "merge_ready": merge_ready,
            }
            self.folder_scan_rows.append(row_info)

            row = self.folder_info_table.rowCount()
            self.folder_info_table.insertRow(row)
            self.folder_info_table.setItem(row, 0, QTableWidgetItem(folder.name))
            self.folder_info_table.setItem(row, 1, QTableWidgetItem(str(row_info["run_type"])))
            self.folder_info_table.setItem(row, 2, QTableWidgetItem(str(row_info["date_start"])))
            self.folder_info_table.setItem(row, 3, QTableWidgetItem(str(row_info["date_end"])))
            for col_idx, key in enumerate(("has_step5", "has_step9", "has_step10"), start=4):
                ok = bool(row_info[key])
                item = QTableWidgetItem("OK" if ok else "없음")
                item.setForeground(QColor("#2E7D32") if ok else QColor("#C62828"))
                self.folder_info_table.setItem(row, col_idx, item)
            self.folder_info_table.setItem(row, 7, QTableWidgetItem(", ".join(filters) if filters else "—"))
            status_item = QTableWidgetItem("사용 가능" if merge_ready else "입력 부족")
            status_item.setForeground(QColor("#2E7D32") if merge_ready else QColor("#C62828"))
            self.folder_info_table.setItem(row, 8, status_item)

    # ───────────────────────── Step 2: ID match ─────────────────────────

    def _next_generated_negative_source_id(self, current_catalogs: dict[str, pd.DataFrame]) -> int:
        min_sid = 0
        for df in current_catalogs.values():
            if df is None or df.empty or "source_id" not in df.columns:
                continue
            sid_vals = coerce_int64_source_id(df["source_id"]).dropna().astype("int64")
            if not sid_vals.empty:
                min_sid = min(min_sid, int(sid_vals.min()))
        return min_sid - 1 if min_sid <= 0 else -1

    def _best_positional_match(self, row: pd.Series, canonical_df: pd.DataFrame, tol_arcsec: float) -> tuple[int | None, float]:
        ra, dec = _row_radec(row)
        if not (np.isfinite(ra) and np.isfinite(dec)):
            return None, float("nan")
        if canonical_df is None or canonical_df.empty or "ra_deg" not in canonical_df.columns or "dec_deg" not in canonical_df.columns:
            return None, float("nan")

        cand = canonical_df.copy()
        cand_ra = pd.to_numeric(cand["ra_deg"], errors="coerce")
        cand_dec = pd.to_numeric(cand["dec_deg"], errors="coerce")
        mask = cand_ra.notna() & cand_dec.notna()
        if not mask.any():
            return None, float("nan")

        sc = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
        csc = SkyCoord(cand_ra[mask].to_numpy(float) * u.deg, cand_dec[mask].to_numpy(float) * u.deg, frame="icrs")
        sep = sc.separation(csc).arcsec
        if len(sep) == 0:
            return None, float("nan")
        best_i = int(np.argmin(sep))
        best_sep = float(sep[best_i])
        if not np.isfinite(best_sep) or best_sep > tol_arcsec:
            return None, best_sep
        best_rows = cand.loc[mask].reset_index(drop=True)
        return int(pd.to_numeric(best_rows.loc[best_i, "source_id"], errors="coerce")), best_sep

    def _canonicalize_catalog_row(
        self,
        row: pd.Series,
        merged_id: int,
        merged_source_id: int,
        folder_tag: str,
    ) -> dict:
        data = row.to_dict()
        data["ID"] = int(merged_id)
        data["source_id"] = int(merged_source_id)
        data["gaia_id"] = int(merged_source_id) if int(merged_source_id) > 0 else np.nan
        data["match_status"] = "matched" if int(merged_source_id) > 0 else "no_gaia_match"
        data["folder_count"] = 1
        data["folder_tags"] = folder_tag
        return data

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
        if len(self.folders) < 2:
            QMessageBox.information(self, "ID Match", "Merge하려면 최소 2개의 RESULT/MERGED workspace가 필요합니다.")
            return

        invalid_rows = [row for row in self.folder_scan_rows if not row.get("merge_ready")]
        if invalid_rows:
            names = "\n".join(f"- {row['folder'].name}" for row in invalid_rows[:8])
            QMessageBox.warning(
                self,
                "ID Match",
                "다음 입력은 Step 5 / Step 9 / Step 10 산출물이 부족합니다:\n\n"
                f"{names}"
            )
            return

        base_folder = self.folders[0]
        self.base_selection_by_filter = _load_selection_payloads(base_folder)
        catalogs_by_folder = {str(folder): _load_master_catalogs_by_filter(folder) for folder in self.folders}
        self.folder_tags = {str(folder): _folder_tag(i, folder) for i, folder in enumerate(self.folders)}

        all_filters = sorted({
            flt for folder in self.folders
            for flt in catalogs_by_folder.get(str(folder), {}).keys()
        })
        if not all_filters:
            self.match_status_label.setText("매칭 실패: master_catalog 없음")
            self.match_log.append("[ERR] 어떤 폴더에서도 master_catalog_*.tsv 를 찾지 못했습니다.")
            return

        pos_tol = float(self.match_radius_combo.currentText())
        next_negative_sid = self._next_generated_negative_source_id({})
        canonical_by_filter: dict[str, pd.DataFrame] = {}
        next_id_by_filter: dict[str, int] = {}

        for folder in self.folders:
            folder_key = str(folder)
            folder_tag = self.folder_tags[folder_key]
            self.local_id_maps.setdefault(folder_key, {})
            filter_catalogs = catalogs_by_folder.get(folder_key, {})

            for flt in all_filters:
                df = filter_catalogs.get(flt)
                if df is None or df.empty:
                    continue

                if flt not in canonical_by_filter:
                    canonical_by_filter[flt] = pd.DataFrame()
                    next_id_by_filter[flt] = 1

                canon = canonical_by_filter[flt].copy()
                local_map: dict[int, dict[str, int]] = {}
                n_exact = 0
                n_pos = 0
                n_new = 0

                # Base folder seeds the canonical catalog and preserves its IDs/source_ids when possible.
                if folder == base_folder and canon.empty:
                    seeded_rows = []
                    max_id = 0
                    for _, row in df.iterrows():
                        local_id = pd.to_numeric(pd.Series([row.get("ID")]), errors="coerce").iloc[0]
                        if not np.isfinite(local_id):
                            continue
                        sid_val = coerce_int64_source_id(pd.Series([row.get("source_id")])).iloc[0]
                        if pd.isna(sid_val):
                            sid = next_negative_sid
                            next_negative_sid -= 1
                        else:
                            sid = int(sid_val)
                        merged_id = int(local_id)
                        max_id = max(max_id, merged_id)
                        seeded_rows.append(self._canonicalize_catalog_row(row, merged_id, sid, folder_tag))
                        local_map[int(local_id)] = {
                            "merged_id": merged_id,
                            "merged_source_id": sid,
                        }
                        self.match_records.append({
                            "folder": folder.name,
                            "folder_tag": folder_tag,
                            "filter": flt,
                            "local_id": int(local_id),
                            "local_source_id": None if pd.isna(sid_val) else int(sid_val),
                            "merged_id": merged_id,
                            "merged_source_id": sid,
                            "method": "base",
                            "sep_arcsec": np.nan,
                            "status": "base",
                        })
                    canon = pd.DataFrame(seeded_rows)
                    next_id_by_filter[flt] = max_id + 1 if max_id > 0 else 1
                    canonical_by_filter[flt] = canon
                    self.local_id_maps[folder_key][flt] = local_map
                    self.match_summary_rows.append({
                        "folder": folder.name,
                        "filter": flt,
                        "exact": len(seeded_rows),
                        "pos": 0,
                        "new": 0,
                        "total": len(seeded_rows),
                        "status": "base",
                    })
                    continue

                canon_sid_map = {}
                if not canon.empty and "source_id" in canon.columns:
                    sid_vals = coerce_int64_source_id(canon["source_id"]).astype("Int64")
                    for idx_row, sid_val in enumerate(sid_vals):
                        if pd.notna(sid_val) and int(sid_val) not in canon_sid_map:
                            canon_sid_map[int(sid_val)] = idx_row

                used_canonical_sids: set[int] = set()
                for _, row in df.iterrows():
                    local_id = pd.to_numeric(pd.Series([row.get("ID")]), errors="coerce").iloc[0]
                    if not np.isfinite(local_id):
                        continue
                    local_id = int(local_id)
                    sid_val = coerce_int64_source_id(pd.Series([row.get("source_id")])).iloc[0]
                    sid_int = None if pd.isna(sid_val) else int(sid_val)

                    matched_sid = None
                    match_method = ""
                    sep_arcsec = float("nan")

                    if sid_int is not None and sid_int in canon_sid_map:
                        matched_sid = sid_int
                        match_method = "source_id"
                    else:
                        matched_sid, sep_arcsec = self._best_positional_match(row, canon, pos_tol)
                        if matched_sid is not None and matched_sid not in used_canonical_sids:
                            match_method = "position"
                        else:
                            matched_sid = None

                    if matched_sid is not None and matched_sid in canon_sid_map:
                        canon_idx = canon_sid_map[matched_sid]
                        merged_id = int(pd.to_numeric(pd.Series([canon.iloc[canon_idx]["ID"]]), errors="coerce").iloc[0])
                        local_map[local_id] = {
                            "merged_id": merged_id,
                            "merged_source_id": int(matched_sid),
                        }
                        used_canonical_sids.add(int(matched_sid))
                        if match_method == "source_id":
                            n_exact += 1
                        else:
                            n_pos += 1
                        canon.at[canon_idx, "folder_count"] = int(pd.to_numeric(pd.Series([canon.iloc[canon_idx].get("folder_count", 1)]), errors="coerce").iloc[0] or 1) + 1
                        canon.at[canon_idx, "folder_tags"] = _append_folder_tag(canon.iloc[canon_idx].get("folder_tags", ""), folder_tag)
                        self.match_records.append({
                            "folder": folder.name,
                            "folder_tag": folder_tag,
                            "filter": flt,
                            "local_id": local_id,
                            "local_source_id": sid_int,
                            "merged_id": merged_id,
                            "merged_source_id": int(matched_sid),
                            "method": match_method,
                            "sep_arcsec": sep_arcsec,
                            "status": "matched",
                        })
                        continue

                    # New canonical source.
                    merged_id = next_id_by_filter.get(flt, 1)
                    next_id_by_filter[flt] = merged_id + 1

                    if sid_int is not None and sid_int not in canon_sid_map:
                        merged_source_id = sid_int
                    else:
                        merged_source_id = next_negative_sid
                        next_negative_sid -= 1

                    new_row = self._canonicalize_catalog_row(row, merged_id, merged_source_id, folder_tag)
                    canon = pd.concat([canon, pd.DataFrame([new_row])], ignore_index=True, sort=False)
                    canon_sid_map[int(merged_source_id)] = len(canon) - 1
                    local_map[local_id] = {
                        "merged_id": merged_id,
                        "merged_source_id": int(merged_source_id),
                    }
                    n_new += 1
                    self.match_records.append({
                        "folder": folder.name,
                        "folder_tag": folder_tag,
                        "filter": flt,
                        "local_id": local_id,
                        "local_source_id": sid_int,
                        "merged_id": merged_id,
                        "merged_source_id": int(merged_source_id),
                        "method": "new",
                        "sep_arcsec": np.nan,
                        "status": "new",
                    })

                canon = canon.sort_values("ID").reset_index(drop=True)
                canonical_by_filter[flt] = canon
                self.local_id_maps[folder_key][flt] = local_map
                self.match_summary_rows.append({
                    "folder": folder.name,
                    "filter": flt,
                    "exact": n_exact,
                    "pos": n_pos,
                    "new": n_new,
                    "total": len(local_map),
                    "status": "OK" if local_map else "empty",
                })
                self.match_log.append(
                    f"[MATCH] {folder.name} / {flt}: exact={n_exact} positional={n_pos} new={n_new} total={len(local_map)}"
                )

        self.merged_catalogs = canonical_by_filter
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
            gmag = _extract_row_float(row, "gaia_G", "gaia_g")
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

    def _selection_to_id_map(self, flt: str, source_ids: set[int]) -> dict[int, int]:
        df = self.merged_catalogs.get(flt)
        if df is None or df.empty or "source_id" not in df.columns or "ID" not in df.columns:
            return {}
        sid_vals = coerce_int64_source_id(df["source_id"])
        id_vals = pd.to_numeric(df["ID"], errors="coerce").astype("Int64")
        out = {}
        for sid_val, id_val in zip(sid_vals, id_vals):
            if pd.isna(sid_val) or pd.isna(id_val):
                continue
            sid_int = int(sid_val)
            if sid_int in source_ids:
                out[sid_int] = int(id_val)
        return out

    def _write_selection_outputs(self, out_dir: Path):
        s9 = step9_selection_dir(out_dir)
        s9.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")

        for flt, df in self.merged_catalogs.items():
            if df is None or df.empty:
                continue
            df_out = df.copy()
            target_sid = self.selection_target_by_filter.get(flt)
            comp_sids = self.selection_comp_by_filter.get(flt, set())
            check_sid = self.selection_check_by_filter.get(flt)

            def _role(sid):
                sid = int(sid)
                if target_sid is not None and sid == int(target_sid):
                    return "T"
                if sid in comp_sids:
                    return "C"
                if check_sid is not None and sid == int(check_sid):
                    return "K"
                return ""

            df_out["role"] = [ _role(int(sid)) for sid in coerce_int64_source_id(df_out["source_id"]).fillna(-999999).astype("int64") ]
            if "gaia_id" not in df_out.columns:
                df_out["gaia_id"] = [
                    int(sid) if int(sid) > 0 else np.nan
                    for sid in coerce_int64_source_id(df_out["source_id"]).fillna(-999999).astype("int64")
                ]
            if "match_status" not in df_out.columns:
                df_out["match_status"] = [
                    "matched" if int(sid) > 0 else "no_gaia_match"
                    for sid in coerce_int64_source_id(df_out["source_id"]).fillna(-999999).astype("int64")
                ]

            output_cols = ["ID", "x_ref", "y_ref", "ra_deg", "dec_deg", "role", "gaia_id", "match_status"]
            for col in [
                "gaia_G", "gaia_BP", "gaia_RP",
                "gaia_g", "gaia_bp", "gaia_rp", "color_gr",
                "folder_count", "folder_tags", "source_id",
            ]:
                if col in df_out.columns and col not in output_cols:
                    output_cols.append(col)
            for col in df_out.columns:
                if col not in output_cols:
                    output_cols.append(col)
            df_out = df_out[[c for c in output_cols if c in df_out.columns]]
            df_out = df_out.sort_values("ID")

            cat_path = s9 / f"master_catalog_{flt}.tsv"
            df_out.to_csv(cat_path, sep="\t", index=False, na_rep="NaN", encoding="utf-8-sig")

            id_map_path = s9 / f"id_mapping_{flt}.csv"
            id_map_df = df_out[[c for c in ["ID", "source_id", "gaia_id", "role", "x_ref", "y_ref"] if c in df_out.columns]].copy()
            id_map_df.to_csv(id_map_path, index=False, na_rep="NaN")

            sel_sids = set(int(s) for s in comp_sids if s is not None)
            if target_sid is not None:
                sel_sids.add(int(target_sid))
            if check_sid is not None:
                sel_sids.add(int(check_sid))
            sid_to_id = self._selection_to_id_map(flt, sel_sids)

            data = {
                "filter": flt,
                "target_id": sid_to_id.get(int(target_sid)) if target_sid is not None else None,
                "target_source_id": int(target_sid) if target_sid is not None else None,
                "comparison_ids": sorted(int(sid_to_id[sid]) for sid in comp_sids if sid in sid_to_id),
                "comparison_source_ids": sorted(int(sid) for sid in comp_sids),
                "check_id": sid_to_id.get(int(check_sid)) if check_sid is not None else None,
                "check_source_id": int(check_sid) if check_sid is not None else None,
                "timestamp": stamp,
            }
            (s9 / f"selection_{flt}.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def _build_merged_workspace(self):
        if not self.merged_catalogs:
            QMessageBox.warning(self, "Merged Workspace", "Step 2 ID match를 먼저 실행하세요.")
            return

        output_dir_text = self.output_dir_edit.text().strip()
        if not output_dir_text:
            QMessageBox.warning(self, "Merged Workspace", "출력 폴더를 지정하세요.")
            return

        out_dir = Path(output_dir_text)
        s1 = step1_dir(out_dir)
        s5 = step5_photometry_dir(out_dir)
        s9 = step9_selection_dir(out_dir)
        s1.mkdir(parents=True, exist_ok=True)
        s5.mkdir(parents=True, exist_ok=True)
        s9.mkdir(parents=True, exist_ok=True)
        step10_dir(out_dir).mkdir(parents=True, exist_ok=True)
        step11_dir(out_dir).mkdir(parents=True, exist_ok=True)
        step12_period_dir(out_dir).mkdir(parents=True, exist_ok=True)

        merged_headers_rows: list[dict] = []
        merged_index_rows: list[dict] = []
        merged_night_assignments: dict[str, int] = {}
        merged_path_map: dict[str, str] = {}

        next_merged_night = 1
        for folder in self.folders:
            folder_key = str(folder)
            folder_tag = self.folder_tags.get(folder_key, folder.name)
            source_path_map = load_file_path_map(folder)
            idx = _read_step5_index(folder)
            if idx.empty or "file" not in idx.columns:
                continue

            headers_df = load_headers_table(folder)
            header_lookup = {}
            if not headers_df.empty and "Filename" in headers_df.columns:
                header_lookup = {str(fn): row.to_dict() for fn, row in headers_df.set_index("Filename").iterrows()}

            night_map_raw = load_night_assignments(folder)
            local_night_ids: dict[str, int] = {}
            for _, row in idx.iterrows():
                fname = str(row.get("file", "")).strip()
                if not fname:
                    continue
                night_id = night_map_raw.get(fname)
                if night_id is None:
                    nid_val = pd.to_numeric(pd.Series([row.get("night_id")]), errors="coerce").iloc[0]
                    night_id = int(nid_val) if np.isfinite(nid_val) and int(nid_val) > 0 else 1
                local_night_ids[fname] = int(night_id)

            local_to_merged: dict[int, int] = {}
            for local_night in sorted(set(local_night_ids.values())):
                local_to_merged[local_night] = next_merged_night
                next_merged_night += 1

            for _, row in idx.iterrows():
                fname = str(row.get("file", "")).strip()
                if not fname:
                    continue
                flt = _normalize_filter_key(row.get("filter", row.get("FILTER", "")))
                local_map = self.local_id_maps.get(folder_key, {}).get(flt, {})
                if not local_map:
                    continue

                phot_df = load_frame_photometry(folder, fname, flt)
                if phot_df is None or phot_df.empty or "ID" not in phot_df.columns:
                    continue

                local_ids = pd.to_numeric(phot_df["ID"], errors="coerce").astype("Int64")
                phot_df = phot_df.loc[local_ids.notna()].copy()
                phot_df["ID_local"] = local_ids[local_ids.notna()].astype(int)
                phot_df = phot_df[phot_df["ID_local"].isin(local_map.keys())].copy()
                if phot_df.empty:
                    continue

                phot_df["source_id"] = phot_df["ID_local"].map(lambda lid: int(local_map[int(lid)]["merged_source_id"]))
                phot_df["ID"] = phot_df["ID_local"].map(lambda lid: int(local_map[int(lid)]["merged_id"]))

                merged_fname = f"{folder_tag}__{fname}"
                phot_df["file"] = merged_fname
                phot_df["source_folder"] = folder_tag
                phot_df["original_file"] = fname

                out_phot_path = s5 / f"{merged_fname}_photometry.tsv"
                phot_df.to_csv(out_phot_path, sep="\t", index=False, na_rep="NaN")

                original_path = source_path_map.get(fname)
                if original_path:
                    merged_path_map[merged_fname] = str(original_path)
                merged_night_id = local_to_merged.get(local_night_ids.get(fname, 1), 1)
                merged_night_assignments[merged_fname] = merged_night_id

                row_dict = row.to_dict()
                row_dict["file"] = merged_fname
                row_dict["filter"] = flt
                row_dict["night_id"] = merged_night_id
                row_dict["source_folder"] = folder_tag
                row_dict["original_file"] = fname
                row_dict["path"] = str(out_phot_path)
                merged_index_rows.append(row_dict)

                header_row = header_lookup.get(fname, {}).copy()
                if not header_row:
                    header_row = {"Filename": merged_fname, "FILTER": flt}
                else:
                    header_row["Filename"] = merged_fname
                header_row["SourceFolder"] = folder_tag
                header_row["OriginalFilename"] = fname
                merged_headers_rows.append(header_row)

        if not merged_index_rows:
            QMessageBox.warning(self, "Merged Workspace", "병합 가능한 Step 5 photometry rows를 만들지 못했습니다.")
            return

        pd.DataFrame(merged_index_rows).to_csv(s5 / "photometry_index.csv", index=False)
        if merged_headers_rows:
            pd.DataFrame(merged_headers_rows).drop_duplicates(subset=["Filename"], keep="first").to_csv(s1 / "headers.csv", index=False)
        (s1 / "night_assignments.json").write_text(
            json.dumps({"night_assignments": merged_night_assignments}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (s1 / "file_path_map.json").write_text(
            json.dumps(merged_path_map, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        self._write_selection_outputs(out_dir)

        manifest = {
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "input_folders": [str(p) for p in self.folders],
            "folder_tags": self.folder_tags,
            "filters": sorted(self.merged_catalogs.keys()),
            "merged_result_dir": str(out_dir),
            "records": len(self.match_records),
        }
        (out_dir / "merge_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        pd.DataFrame(self.match_records).to_csv(out_dir / "merge_id_map.csv", index=False)
        write_run_manifest(
            out_dir,
            run_type="merged",
            root_dir=out_dir.parent,
            input_result_dirs=self.folders,
            target_name=infer_result_workspace_label(self.folders),
            storage_mode="full",
        )

        self.merged_result_dir = out_dir
        self._build_merged_runtime_context(merged_night_assignments, merged_path_map)
        self._refresh_runtime_status_labels()
        self.selection_log.append(f"[BUILD] merged workspace ready: {out_dir}")
        QMessageBox.information(self, "Merged Workspace", f"생성 완료:\n{out_dir}")
        self._go_to_step(3)

    def _build_merged_runtime_context(self, night_assignments: dict[str, int], path_map: dict[str, str]):
        if self.merged_result_dir is None:
            return
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

    def _refresh_runtime_status_labels(self):
        merged_dir = self.merged_result_dir
        if merged_dir is None:
            self.step10_status_label.setText("Merged workspace 없음")
            self.step11_status_label.setText("Merged workspace 없음")
            self.step12_status_label.setText("Merged workspace 없음")
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
            from .step10_light_curve_builder import LightCurveBuilderWindow
            self.current_step_window = LightCurveBuilderWindow(
                self.merged_runtime_params,
                self.merged_runtime_file_manager,
                self.merged_runtime_project_state,
                self,
            )
        elif step_index == 10:
            from .step11_detrend_merge import DetrendNightMergeWindow
            self.current_step_window = DetrendNightMergeWindow(
                self.merged_runtime_params,
                self.merged_runtime_file_manager,
                self.merged_runtime_project_state,
                self,
            )
        elif step_index == 11:
            from .step12_period_analysis import PeriodAnalysisWindow
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
        if self._current_step == 0 and len(self.folders) < 2:
            QMessageBox.warning(self, "Merger", "Merge하려면 최소 2개의 RESULT/MERGED workspace를 선택하세요.")
            return
        if self._current_step == 0:
            if not self.folder_scan_rows:
                self._scan_folders()
            invalid_rows = [row for row in self.folder_scan_rows if not row.get("merge_ready")]
            if invalid_rows:
                QMessageBox.warning(self, "Merger", "Step 5 / Step 9 / Step 10이 모두 있는 workspace만 머저할 수 있습니다.")
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
