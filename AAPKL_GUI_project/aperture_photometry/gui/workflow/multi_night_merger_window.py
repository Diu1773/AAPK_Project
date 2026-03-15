"""
Multi-Night Light Curve Merger
Merges same-target observations processed in separate result folders,
using Gaia source_id cross-matching to reconcile local star IDs.
"""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from astropy.time import Time

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QGroupBox, QListWidget, QListWidgetItem,
    QTableWidget, QTableWidgetItem, QHeaderView, QTextEdit,
    QFileDialog, QMessageBox, QSplitter, QSizePolicy,
    QProgressBar, QSpinBox, QDoubleSpinBox, QCheckBox, QFormLayout,
    QComboBox,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor

from ...utils.step_paths import step6_dir, step8_dir, step10_dir, step11_dir
from ...utils.io_utils import read_csv_int64_source_id, coerce_int64_source_id
from ...utils.astro_utils import compute_bjd_tdb_array


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _load_target_radec(result_dir: Path, target_id: int) -> tuple[float, float]:
    """Return (ra_deg, dec_deg) for target_id from master_catalog, or (nan, nan)."""
    s8 = step8_dir(result_dir)
    candidates = list(s8.glob("master_catalog_*.tsv")) if s8.exists() else []
    candidates += [s8 / "master_catalog.tsv"] if s8.exists() else []
    for path in candidates:
        if not path.exists():
            continue
        try:
            df = read_csv_int64_source_id(path, sep="\t")
            if "ID" not in df.columns:
                continue
            row = df[pd.to_numeric(df["ID"], errors="coerce") == int(target_id)]
            if row.empty:
                continue
            ra = float(pd.to_numeric(row["ra_deg"].values[0], errors="coerce")) if "ra_deg" in df.columns else float("nan")
            dec = float(pd.to_numeric(row["dec_deg"].values[0], errors="coerce")) if "dec_deg" in df.columns else float("nan")
            if np.isfinite(ra) and np.isfinite(dec):
                return ra, dec
        except Exception:
            continue
    return float("nan"), float("nan")


def _load_master_catalog(result_dir: Path) -> pd.DataFrame:
    """Load master_catalog from step8 (preferred) or step6."""
    for base_dir in (step8_dir(result_dir), step6_dir(result_dir)):
        if not base_dir.exists():
            continue
        candidates = sorted(base_dir.glob("master_catalog*.tsv"))
        for p in candidates:
            try:
                df = read_csv_int64_source_id(p, sep="\t")
                if {"source_id", "ID"} <= set(df.columns):
                    return df
            except Exception:
                continue
    return pd.DataFrame()


def _build_id_to_source_map(catalog: pd.DataFrame) -> dict[int, int]:
    """ID → Gaia source_id"""
    if not {"source_id", "ID"} <= set(catalog.columns):
        return {}
    sid_vals = coerce_int64_source_id(catalog["source_id"])
    id_vals = pd.to_numeric(catalog["ID"], errors="coerce").astype("Int64")
    out: dict[int, int] = {}
    for id_val, sid_val in zip(id_vals, sid_vals):
        if pd.isna(id_val) or pd.isna(sid_val):
            continue
        out[int(id_val)] = int(sid_val)
    return out


def _build_source_to_id_map(catalog: pd.DataFrame) -> dict[int, int]:
    """Gaia source_id → local ID"""
    if not {"source_id", "ID"} <= set(catalog.columns):
        return {}
    sid_vals = coerce_int64_source_id(catalog["source_id"])
    id_vals = pd.to_numeric(catalog["ID"], errors="coerce").astype("Int64")
    out: dict[int, int] = {}
    for sid_val, id_val in zip(sid_vals, id_vals):
        if pd.isna(sid_val) or pd.isna(id_val):
            continue
        if int(sid_val) not in out:
            out[int(sid_val)] = int(id_val)
    return out


def _load_selection_ids(result_dir: Path) -> tuple[int | None, list[int]]:
    """Load target_id and comp_ids from step8 selection files."""
    s8 = step8_dir(result_dir)
    # per-filter selection_{filter}.json
    for sel_file in sorted(s8.glob("selection_*.json")):
        try:
            data = json.loads(sel_file.read_text(encoding="utf-8"))
            tid = data.get("target_id")
            cids = data.get("comparison_ids", [])
            if tid is not None:
                return int(tid), [int(c) for c in cids if c is not None]
        except Exception:
            continue
    # legacy target_selection.json
    for p in (s8 / "target_selection.json", result_dir / "target_selection.json"):
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                tid = data.get("target_id")
                cids = data.get("comparison_ids", [])
                if tid is not None:
                    return int(tid), [int(c) for c in cids if c is not None]
            except Exception:
                continue
    return None, []


def _load_raw_csv(result_dir: Path, local_target_id: int) -> pd.DataFrame | None:
    """Load cached lightcurve_ID{id}_raw.csv from step10_dir."""
    out_dir = step10_dir(result_dir)
    candidates = list(out_dir.glob(f"lightcurve_ID{local_target_id}_*_raw.csv"))
    if not candidates:
        candidates = list(out_dir.glob(f"lightcurve_ID{local_target_id}_raw.csv"))
    if not candidates:
        return None
    dfs = []
    for p in sorted(candidates):
        try:
            dfs.append(pd.read_csv(p))
        except Exception:
            continue
    return pd.concat(dfs, ignore_index=True) if dfs else None


# ─────────────────────────────────────────────
# Build worker
# ─────────────────────────────────────────────

class MergerBuildWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(pd.DataFrame, str)   # merged_df, error_msg

    def __init__(self, folders: list[Path], id_mapping: list[dict]):
        super().__init__()
        self.folders = folders
        self.id_mapping = id_mapping  # [{folder, target_id, comp_ids, status}]

    def run(self):
        frames = []
        errors = []
        for entry in self.id_mapping:
            folder: Path = entry["folder"]
            tid: int | None = entry.get("target_id")
            status: str = entry.get("status", "")
            if tid is None or "fail" in status.lower():
                errors.append(f"{folder.name}: ID 매핑 실패, 건너뜀")
                self.progress.emit(f"[SKIP] {folder.name}: no valid target_id")
                continue
            self.progress.emit(f"[LOAD] {folder.name} target_id={tid}")
            df = _load_raw_csv(folder, tid)
            if df is None:
                errors.append(f"{folder.name}: lightcurve_ID{tid}_raw.csv 없음 — Step 10 먼저 실행하세요")
                self.progress.emit(f"[ERR] {folder.name}: no raw CSV for ID {tid}")
                continue
            df = df.copy()
            df["_source_folder"] = folder.name
            frames.append(df)
            self.progress.emit(f"[OK]  {folder.name}: {len(df)} rows")

        if not frames:
            self.finished.emit(pd.DataFrame(), "\n".join(errors) if errors else "로드된 데이터 없음")
            return

        merged = pd.concat(frames, ignore_index=True)
        if "JD" in merged.columns:
            merged = merged.sort_values("JD").reset_index(drop=True)
        err_str = "\n".join(errors) if errors else ""
        self.finished.emit(merged, err_str)


# ─────────────────────────────────────────────
# Main window
# ─────────────────────────────────────────────

class MultiNightMergerWindow(QMainWindow):
    """Standalone window for merging multi-night light curves."""

    STEP_TITLES = [
        "Step 1  폴더 선택",
        "Step 2  ID 매칭",
        "Step 3  빌드 & 병합",
        "Step 4  디트렌드 & 플롯",
    ]

    def __init__(self, params, project_state, main_window):
        super().__init__()
        self.params = params
        self.project_state = project_state
        self.main_window = main_window

        self.folders: list[Path] = [Path(params.P.result_dir)]
        self.id_mapping: list[dict] = []   # populated in step 2
        self.merged_df: pd.DataFrame = pd.DataFrame()
        self.corrected_df: pd.DataFrame = pd.DataFrame()
        self._build_worker: MergerBuildWorker | None = None

        self.setWindowTitle("Multi-Night Light Curve Merger")
        self.resize(1100, 750)
        self._setup_ui()

    # ── UI construction ──────────────────────────────

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ── Header bar ──
        header = QHBoxLayout()
        btn_back = QPushButton("← 메인으로")
        btn_back.setFixedWidth(110)
        btn_back.setStyleSheet(
            "QPushButton { background:#607D8B; color:white; font-weight:bold; "
            "padding:4px 10px; border-radius:4px; }"
            "QPushButton:hover { background:#455A64; }"
        )
        btn_back.clicked.connect(self._go_back)
        header.addWidget(btn_back)

        title = QLabel("Multi-Night Light Curve Merger")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        header.addWidget(title, 1)

        # Placeholder to balance layout
        header.addSpacing(110)
        root.addLayout(header)

        # ── Step navigation bar ──
        step_bar = QHBoxLayout()
        step_bar.setSpacing(2)
        self._step_btns: list[QPushButton] = []
        for i, title_text in enumerate(self.STEP_TITLES):
            btn = QPushButton(title_text)
            btn.setCheckable(True)
            btn.setMinimumHeight(30)
            btn.clicked.connect(lambda checked, idx=i: self._go_to_step(idx))
            step_bar.addWidget(btn)
            self._step_btns.append(btn)
        root.addLayout(step_bar)

        # ── Step pages (manual show/hide) ──
        self._pages: list[QWidget] = []
        self._pages.append(self._make_step1())
        self._pages.append(self._make_step2())
        self._pages.append(self._make_step3())
        self._pages.append(self._make_step4())

        self._page_container = QWidget()
        self._page_layout = QVBoxLayout(self._page_container)
        self._page_layout.setContentsMargins(0, 0, 0, 0)
        for page in self._pages:
            self._page_layout.addWidget(page)
            page.hide()
        root.addWidget(self._page_container, 1)

        # ── Bottom nav ──
        nav = QHBoxLayout()
        self.btn_prev = QPushButton("◀ 이전")
        self.btn_prev.clicked.connect(self._prev_step)
        self.btn_next = QPushButton("다음 ▶")
        self.btn_next.setStyleSheet(
            "QPushButton { background:#1565C0; color:white; font-weight:bold; padding:4px 16px; }"
            "QPushButton:hover { background:#0D47A1; }"
        )
        self.btn_next.clicked.connect(self._next_step)
        nav.addWidget(self.btn_prev)
        nav.addStretch()
        nav.addWidget(self.btn_next)
        root.addLayout(nav)

        self._current_step = 0
        self._go_to_step(0)

    # ── Step 1: Folder Selection ──────────────────────

    def _make_step1(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(8)

        info = QLabel(
            "같은 타겟을 여러 날 밤 처리한 result 폴더들을 추가하세요.\n"
            "현재 프로젝트 폴더는 자동으로 포함됩니다."
        )
        info.setStyleSheet("QLabel { background:#E3F2FD; padding:8px; border-radius:4px; }")
        info.setWordWrap(True)
        layout.addWidget(info)

        grp = QGroupBox("Result 폴더 목록")
        grp_layout = QVBoxLayout(grp)

        self.folder_list = QListWidget()
        self.folder_list.setMinimumHeight(200)
        self._refresh_folder_list()
        grp_layout.addWidget(self.folder_list)

        btn_row = QHBoxLayout()
        btn_add = QPushButton("폴더 추가")
        btn_add.setStyleSheet("background:#4CAF50; color:white; font-weight:bold; padding:4px 12px;")
        btn_add.clicked.connect(self._on_add_folder)
        btn_remove = QPushButton("선택 제거")
        btn_remove.clicked.connect(self._on_remove_folder)
        btn_row.addWidget(btn_add)
        btn_row.addWidget(btn_remove)
        btn_row.addStretch()
        grp_layout.addLayout(btn_row)
        layout.addWidget(grp)

        # Folder info table
        info_grp = QGroupBox("폴더별 정보")
        info_layout = QVBoxLayout(info_grp)
        self.folder_info_table = QTableWidget(0, 4)
        self.folder_info_table.setHorizontalHeaderLabels(["폴더", "프레임 수", "필터", "Step 10 캐시"])
        self.folder_info_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.folder_info_table.setMaximumHeight(160)
        self.folder_info_table.setEditTriggers(QTableWidget.NoEditTriggers)
        info_layout.addWidget(self.folder_info_table)

        btn_scan = QPushButton("폴더 스캔")
        btn_scan.clicked.connect(self._scan_folders)
        info_layout.addWidget(btn_scan)
        layout.addWidget(info_grp)
        layout.addStretch()
        return page

    def _refresh_folder_list(self):
        self.folder_list.clear()
        for i, p in enumerate(self.folders):
            label = f"[현재]  {p}" if i == 0 else str(p)
            item = QListWidgetItem(label)
            if i == 0:
                item.setForeground(QColor("#1565C0"))
                item.setFlags(item.flags() & ~Qt.ItemIsSelectable)
            self.folder_list.addItem(item)

    def _on_add_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "result 폴더 선택", str(self.folders[0].parent)
        )
        if not folder:
            return
        p = Path(folder)
        if any(f.resolve() == p.resolve() for f in self.folders):
            return
        self.folders.append(p)
        self._refresh_folder_list()

    def _on_remove_folder(self):
        row = self.folder_list.currentRow()
        if row <= 0:
            return
        self.folders.pop(row)
        self._refresh_folder_list()

    def _scan_folders(self):
        self.folder_info_table.setRowCount(0)
        for folder in self.folders:
            row = self.folder_info_table.rowCount()
            self.folder_info_table.insertRow(row)
            self.folder_info_table.setItem(row, 0, QTableWidgetItem(folder.name))

            idx_path = folder / "step9_photometry" / "photometry_index.csv"
            if not idx_path.exists():
                idx_path = folder / "photometry_index.csv"
            n_frames = ""
            filters = ""
            if idx_path.exists():
                try:
                    idx = pd.read_csv(idx_path)
                    n_frames = str(len(idx))
                    if "filter" in idx.columns:
                        filters = ", ".join(sorted(idx["filter"].dropna().unique()))
                except Exception:
                    pass
            self.folder_info_table.setItem(row, 1, QTableWidgetItem(n_frames))
            self.folder_info_table.setItem(row, 2, QTableWidgetItem(filters))

            cached = list((folder / "step10_lightcurve").glob("lightcurve_ID*_raw.csv")) if (folder / "step10_lightcurve").exists() else []
            cache_str = f"{len(cached)}개" if cached else "없음"
            item = QTableWidgetItem(cache_str)
            item.setForeground(QColor("#2E7D32") if cached else QColor("#C62828"))
            self.folder_info_table.setItem(row, 3, item)

    # ── Step 2: ID Matching ───────────────────────────

    def _make_step2(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(8)

        info = QLabel(
            "base 폴더의 target/comp ID를 Gaia source_id 기준으로 각 폴더의 local ID와 매칭합니다."
        )
        info.setStyleSheet("QLabel { background:#E3F2FD; padding:8px; border-radius:4px; }")
        info.setWordWrap(True)
        layout.addWidget(info)

        self.match_status_label = QLabel("매칭 결과: (아직 실행 안 됨)")
        self.match_status_label.setStyleSheet(
            "QLabel { background:#FAFAFA; padding:6px; border-radius:4px; font-size:9pt; }"
        )
        layout.addWidget(self.match_status_label)

        # Matching table
        grp = QGroupBox("ID 매핑 테이블")
        grp_layout = QVBoxLayout(grp)
        self.match_table = QTableWidget(0, 5)
        self.match_table.setHorizontalHeaderLabels(
            ["폴더", "target_id (local)", "Gaia source_id", "comp_ids (local)", "상태"]
        )
        self.match_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.match_table.setMinimumHeight(200)
        self.match_table.setEditTriggers(QTableWidget.NoEditTriggers)
        grp_layout.addWidget(self.match_table)

        btn_row = QHBoxLayout()
        btn_run = QPushButton("ID 매칭 실행")
        btn_run.setStyleSheet("background:#1565C0; color:white; font-weight:bold; padding:4px 16px;")
        btn_run.clicked.connect(self._run_id_match)
        btn_row.addWidget(btn_run)
        btn_row.addStretch()
        grp_layout.addLayout(btn_row)
        layout.addWidget(grp)

        # Log
        self.match_log = QTextEdit()
        self.match_log.setReadOnly(True)
        self.match_log.setMaximumHeight(120)
        self.match_log.setStyleSheet("font-size:8pt; font-family:monospace;")
        layout.addWidget(self.match_log)
        layout.addStretch()
        return page

    def _run_id_match(self):
        self.match_log.clear()
        self.id_mapping = []

        if len(self.folders) < 1:
            self.match_log.append("폴더를 먼저 추가하세요.")
            return

        base_folder = self.folders[0]
        base_tid, base_cids = _load_selection_ids(base_folder)
        if base_tid is None:
            self.match_log.append(f"[ERR] base 폴더({base_folder.name})에서 target ID를 찾을 수 없습니다.")
            self.match_log.append("      Step 8 (Master ID Editor)를 완료했는지 확인하세요.")
            return

        self.match_log.append(f"[BASE] {base_folder.name}: target_id={base_tid}, comp_ids={base_cids}")

        # Get Gaia source_ids from base catalog
        base_catalog = _load_master_catalog(base_folder)
        if base_catalog.empty:
            self.match_log.append(f"[ERR] base 폴더 master_catalog 없음.")
            return

        id2src = _build_id_to_source_map(base_catalog)
        base_target_src = id2src.get(base_tid)
        base_comp_srcs = [id2src.get(c) for c in base_cids]

        self.match_log.append(f"      Gaia source_id: target={base_target_src}, comp={base_comp_srcs}")

        self.match_table.setRowCount(0)
        n_ok = 0
        for folder in self.folders:
            if folder.resolve() == base_folder.resolve():
                # base folder uses IDs as-is
                tid_local = base_tid
                cids_local = base_cids
                status = "base"
            else:
                catalog = _load_master_catalog(folder)
                if catalog.empty:
                    self.match_log.append(f"[WARN] {folder.name}: master_catalog 없음, 건너뜀")
                    tid_local = None
                    cids_local = []
                    status = "fail: catalog 없음"
                else:
                    src2id = _build_source_to_id_map(catalog)
                    tid_local = src2id.get(base_target_src) if base_target_src is not None else None
                    cids_local = [src2id.get(s) for s in base_comp_srcs if s is not None and s in src2id]
                    if tid_local is None:
                        status = "fail: target Gaia 미발견"
                        self.match_log.append(f"[WARN] {folder.name}: target Gaia ID {base_target_src} 미발견")
                    else:
                        n_matched_comp = len(cids_local)
                        status = f"OK (comp {n_matched_comp}/{len(base_cids)})"
                        self.match_log.append(f"[OK]  {folder.name}: target_id={tid_local}, comp_ids={cids_local}")

            entry = {
                "folder": folder,
                "target_id": tid_local,
                "comp_ids": [c for c in cids_local if c is not None],
                "target_gaia_src": base_target_src,
                "status": status,
            }
            self.id_mapping.append(entry)

            row = self.match_table.rowCount()
            self.match_table.insertRow(row)
            self.match_table.setItem(row, 0, QTableWidgetItem(folder.name))
            self.match_table.setItem(row, 1, QTableWidgetItem(str(tid_local) if tid_local is not None else "—"))
            self.match_table.setItem(row, 2, QTableWidgetItem(str(base_target_src) if base_target_src else "—"))
            cids_str = ", ".join(str(c) for c in cids_local if c is not None)
            self.match_table.setItem(row, 3, QTableWidgetItem(cids_str or "—"))
            status_item = QTableWidgetItem(status)
            status_item.setForeground(
                QColor("#2E7D32") if "OK" in status or status == "base"
                else QColor("#C62828")
            )
            self.match_table.setItem(row, 4, status_item)
            if "OK" in status or status == "base":
                n_ok += 1

        self.match_status_label.setText(
            f"매칭 완료: {n_ok}/{len(self.folders)} 폴더 성공"
        )

    # ── Step 3: Build & Merge ─────────────────────────

    def _make_step3(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(8)

        info = QLabel(
            "각 폴더의 lightcurve_ID*_raw.csv를 로드하여 병합합니다.\n"
            "캐시 파일이 없으면 Step 10을 먼저 실행하세요."
        )
        info.setStyleSheet("QLabel { background:#E3F2FD; padding:8px; border-radius:4px; }")
        info.setWordWrap(True)
        layout.addWidget(info)

        btn_build = QPushButton("병합 실행")
        btn_build.setStyleSheet(
            "QPushButton { background:#4CAF50; color:white; font-weight:bold; padding:6px 20px; }"
            "QPushButton:hover { background:#388E3C; }"
        )
        btn_build.clicked.connect(self._run_build)
        layout.addWidget(btn_build)

        self.build_progress = QProgressBar()
        self.build_progress.setRange(0, 0)
        self.build_progress.setVisible(False)
        layout.addWidget(self.build_progress)

        self.build_log = QTextEdit()
        self.build_log.setReadOnly(True)
        self.build_log.setStyleSheet("font-size:8pt; font-family:monospace;")
        self.build_log.setMinimumHeight(160)
        layout.addWidget(self.build_log)

        # Stats after merge
        self.merge_stats_label = QLabel("")
        self.merge_stats_label.setStyleSheet(
            "QLabel { background:#E8F5E9; padding:8px; border-radius:4px; font-weight:bold; }"
        )
        self.merge_stats_label.setWordWrap(True)
        layout.addWidget(self.merge_stats_label)

        # Save
        save_row = QHBoxLayout()
        self.btn_save_merged = QPushButton("병합 CSV 저장")
        self.btn_save_merged.setEnabled(False)
        self.btn_save_merged.clicked.connect(self._save_merged_csv)
        save_row.addWidget(self.btn_save_merged)
        save_row.addStretch()
        layout.addLayout(save_row)
        layout.addStretch()
        return page

    def _run_build(self):
        if not self.id_mapping:
            QMessageBox.warning(self, "병합", "Step 2 ID 매칭을 먼저 실행하세요.")
            return

        self.build_log.clear()
        self.merge_stats_label.setText("")
        self.build_progress.setVisible(True)
        self.btn_save_merged.setEnabled(False)

        self._build_worker = MergerBuildWorker(self.folders, self.id_mapping)
        self._build_worker.progress.connect(lambda msg: self.build_log.append(msg))
        self._build_worker.finished.connect(self._on_build_finished)
        self._build_worker.start()

    def _on_build_finished(self, merged: pd.DataFrame, err_msg: str):
        self.build_progress.setVisible(False)

        if err_msg:
            self.build_log.append(f"\n[경고]\n{err_msg}")

        if merged.empty:
            self.merged_df = merged
            self.merge_stats_label.setText("병합 실패: 로드된 데이터 없음")
            return

        # Attempt BJD_TDB conversion
        merged = self._add_bjd_column(merged)
        self.merged_df = merged

        n_frames = len(merged)
        n_nights = merged["_source_folder"].nunique() if "_source_folder" in merged.columns else "?"
        jd_min = merged["JD"].min() if "JD" in merged.columns else float("nan")
        jd_max = merged["JD"].max() if "JD" in merged.columns else float("nan")
        bjd_note = "  BJD_TDB 컬럼 추가됨" if "BJD_TDB" in merged.columns else ""

        self.merge_stats_label.setText(
            f"병합 완료: {n_frames} 포인트, {n_nights}개 폴더  |  "
            f"JD {jd_min:.3f} ~ {jd_max:.3f}{bjd_note}"
        )
        self.btn_save_merged.setEnabled(True)
        self.build_log.append(f"\n총 {n_frames}개 포인트 병합 완료.")

    def _add_bjd_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add BJD_TDB column from JD if site coords and target coords are available."""
        if "JD" not in df.columns or "BJD_TDB" in df.columns:
            return df
        try:
            site_lat = float(getattr(self.params.P, "site_lat_deg", float("nan")))
            site_lon = float(getattr(self.params.P, "site_lon_deg", float("nan")))
            site_alt = float(getattr(self.params.P, "site_alt_m", 0.0))
            if not (np.isfinite(site_lat) and np.isfinite(site_lon)):
                return df
            # Use base folder target ID
            base_entry = next((e for e in self.id_mapping if e.get("target_id") is not None), None)
            if base_entry is None:
                return df
            target_id = base_entry["target_id"]
            tgt_ra, tgt_dec = _load_target_radec(self.folders[0], target_id)
            if not (np.isfinite(tgt_ra) and np.isfinite(tgt_dec)):
                # Fallback: params.P.target
                target_cfg = getattr(self.params.P, "target", None)
                if target_cfg is not None:
                    tgt_ra = float(getattr(target_cfg, "ra_deg", float("nan")) or float("nan"))
                    tgt_dec = float(getattr(target_cfg, "dec_deg", float("nan")) or float("nan"))
            if not (np.isfinite(tgt_ra) and np.isfinite(tgt_dec)):
                return df
            jd_arr = pd.to_numeric(df["JD"], errors="coerce").to_numpy(float)
            bjd_arr = compute_bjd_tdb_array(jd_arr, tgt_ra, tgt_dec, site_lat, site_lon, site_alt)
            df = df.copy()
            df["BJD_TDB"] = bjd_arr
            self.build_log.append(f"[BJD] site=({site_lat:.4f}, {site_lon:.4f}), target=({tgt_ra:.6f}, {tgt_dec:.6f})")
        except Exception as e:
            self.build_log.append(f"[BJD 변환 실패] {e}")
        return df

    def _save_merged_csv(self):
        if self.merged_df.empty:
            return
        base = self.folders[0]
        default_path = str(base.parent / f"{base.name}_merged_raw.csv")
        path, _ = QFileDialog.getSaveFileName(self, "병합 CSV 저장", default_path, "CSV (*.csv)")
        if path:
            self.merged_df.to_csv(path, index=False)
            QMessageBox.information(self, "저장", f"저장 완료:\n{path}")

    # ── Step 4: Detrend & Plot ────────────────────────

    def _make_step4(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(6)

        splitter = QSplitter(Qt.Horizontal)

        # Left: controls
        ctrl_widget = QWidget()
        ctrl_widget.setMaximumWidth(260)
        ctrl_layout = QVBoxLayout(ctrl_widget)
        ctrl_layout.setSpacing(6)

        detrend_grp = QGroupBox("야간 오프셋 보정")
        detrend_form = QFormLayout(detrend_grp)
        detrend_form.setSpacing(4)

        self.chk_detrend = QCheckBox("적용")
        self.chk_detrend.setChecked(True)
        detrend_form.addRow("Nightly offset:", self.chk_detrend)

        self.chk_linear = QCheckBox("선형 트렌드 제거 (야간 내 drift 보정)")
        self.chk_linear.setChecked(False)
        detrend_form.addRow("Linear detrend:", self.chk_linear)

        self.chk_clip = QCheckBox("적용")
        self.chk_clip.setChecked(True)
        detrend_form.addRow("Sigma clip:", self.chk_clip)

        self.spin_sigma = QDoubleSpinBox()
        self.spin_sigma.setRange(1.0, 10.0)
        self.spin_sigma.setSingleStep(0.5)
        self.spin_sigma.setValue(3.0)
        detrend_form.addRow("σ:", self.spin_sigma)

        ctrl_layout.addWidget(detrend_grp)

        plot_grp = QGroupBox("플롯 옵션")
        plot_form = QFormLayout(plot_grp)
        plot_form.setSpacing(4)

        self.filter_combo = QComboBox()
        self.filter_combo.addItem("All")
        plot_form.addRow("필터:", self.filter_combo)

        ctrl_layout.addWidget(plot_grp)

        btn_plot = QPushButton("플롯 갱신")
        btn_plot.setStyleSheet(
            "QPushButton { background:#1565C0; color:white; font-weight:bold; padding:4px 12px; }"
            "QPushButton:hover { background:#0D47A1; }"
        )
        btn_plot.clicked.connect(self._update_plot)
        ctrl_layout.addWidget(btn_plot)

        btn_export = QPushButton("보정 CSV 내보내기")
        btn_export.clicked.connect(self._export_corrected)
        ctrl_layout.addWidget(btn_export)

        ctrl_layout.addStretch()
        splitter.addWidget(ctrl_widget)

        # Right: plot
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.fig = Figure(figsize=(8, 5), tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        right_layout.addWidget(self.canvas)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter, 1)
        return page

    def _update_plot(self):
        if self.merged_df.empty:
            QMessageBox.information(self, "플롯", "Step 3 병합을 먼저 실행하세요.")
            return

        df = self.merged_df.copy()
        flt = self.filter_combo.currentText()
        if flt != "All" and "filter" in df.columns:
            df = df[df["filter"] == flt]
        if df.empty:
            return

        # Populate filter combo once
        if self.filter_combo.count() == 1 and "filter" in self.merged_df.columns:
            for f in sorted(self.merged_df["filter"].dropna().unique()):
                if self.filter_combo.findText(f) < 0:
                    self.filter_combo.addItem(f)

        # Nightly offset + optional linear detrend correction
        if self.chk_detrend.isChecked() and "_source_folder" in df.columns and "delta_mag" in df.columns:
            corrected_parts = []
            x_col_for_trend = "BJD_TDB" if "BJD_TDB" in df.columns else ("JD" if "JD" in df.columns else None)
            use_linear = self.chk_linear.isChecked() and x_col_for_trend is not None
            for folder_name, grp in df.groupby("_source_folder"):
                vals = pd.to_numeric(grp["delta_mag"], errors="coerce")
                sigma = self.spin_sigma.value()
                if self.chk_clip.isChecked():
                    med = np.nanmedian(vals)
                    mad = np.nanmedian(np.abs(vals - med))
                    clip_mask = np.abs(vals - med) < sigma * mad * 1.4826
                else:
                    clip_mask = np.ones(len(vals), dtype=bool)
                grp = grp.copy()
                valid = clip_mask & np.isfinite(vals)
                if use_linear and valid.sum() >= 3:
                    t_vals = pd.to_numeric(grp[x_col_for_trend], errors="coerce")
                    t_c = t_vals[valid].to_numpy()
                    y_c = vals[valid].to_numpy()
                    t_mean = t_c.mean()
                    coeffs = np.polyfit(t_c - t_mean, y_c, 1)
                    trend = np.polyval(coeffs, t_vals.to_numpy() - t_mean)
                    # Remove linear trend, keep zero-mean for the clipped subset
                    residuals = vals.to_numpy() - trend
                    offset = np.nanmedian(residuals[valid])
                    grp["delta_mag_corr"] = residuals - offset
                else:
                    offset = np.nanmedian(vals[valid])
                    grp["delta_mag_corr"] = vals - offset
                corrected_parts.append(grp)
            df = pd.concat(corrected_parts, ignore_index=True) if corrected_parts else df
            self.corrected_df = df
            y_col = "delta_mag_corr"
        else:
            self.corrected_df = df
            y_col = "delta_mag" if "delta_mag" in df.columns else (df.columns[-1] if len(df.columns) else None)

        self.fig.clear()
        ax = self.fig.add_subplot(111)

        x_col = next((c for c in ("BJD_TDB", "JD") if c in df.columns), None)
        if x_col is None or y_col is None or y_col not in df.columns:
            ax.text(0.5, 0.5, "플롯할 컬럼 없음", transform=ax.transAxes, ha="center")
            self.canvas.draw()
            return

        colors = ["#1565C0", "#C62828", "#2E7D32", "#F57F17", "#6A1B9A", "#00695C"]
        folders = df["_source_folder"].unique() if "_source_folder" in df.columns else ["all"]
        for i, folder_name in enumerate(folders):
            mask = df["_source_folder"] == folder_name if "_source_folder" in df.columns else pd.Series([True] * len(df))
            sub = df[mask]
            x = pd.to_numeric(sub[x_col], errors="coerce")
            y = pd.to_numeric(sub[y_col], errors="coerce")
            err_col = "delta_mag_err" if "delta_mag_err" in sub.columns else None
            c = colors[i % len(colors)]
            if err_col:
                e = pd.to_numeric(sub[err_col], errors="coerce")
                ax.errorbar(x, y, yerr=e, fmt="o", color=c, markersize=3,
                            elinewidth=0.8, capsize=1.5, label=folder_name, alpha=0.8)
            else:
                ax.scatter(x, y, color=c, s=10, label=folder_name, alpha=0.8)

        ax.invert_yaxis()
        ax.set_xlabel(x_col)
        ax.set_ylabel("Δm (corr)" if y_col == "delta_mag_corr" else "Δm")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)
        self.canvas.draw()

    def _export_corrected(self):
        if self.corrected_df.empty:
            QMessageBox.information(self, "내보내기", "먼저 플롯을 실행하세요.")
            return
        base = self.folders[0]
        default_path = str(base.parent / f"{base.name}_merged_corrected.csv")
        path, _ = QFileDialog.getSaveFileName(self, "보정 CSV 내보내기", default_path, "CSV (*.csv)")
        if path:
            self.corrected_df.to_csv(path, index=False)
            QMessageBox.information(self, "저장", f"저장 완료:\n{path}")

    # ── Step navigation ───────────────────────────────

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
        # Auto-actions on entering steps
        if idx == 3 and not self.merged_df.empty:
            self._populate_filter_combo()

    def _populate_filter_combo(self):
        self.filter_combo.clear()
        self.filter_combo.addItem("All")
        if "filter" in self.merged_df.columns:
            for f in sorted(self.merged_df["filter"].dropna().unique()):
                self.filter_combo.addItem(f)

    def _prev_step(self):
        if self._current_step > 0:
            self._go_to_step(self._current_step - 1)

    def _next_step(self):
        if self._current_step < len(self._pages) - 1:
            self._go_to_step(self._current_step + 1)

    # ── Back to main ─────────────────────────────────

    def _go_back(self):
        self.hide()
        self.main_window.show()
        self.main_window.raise_()
        self.main_window.activateWindow()

    def closeEvent(self, event):
        self._go_back()
        event.ignore()  # Don't destroy, just hide
