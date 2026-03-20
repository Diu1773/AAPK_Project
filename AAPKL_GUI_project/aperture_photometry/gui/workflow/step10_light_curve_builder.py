"""
Step 10: Light Curve Builder
"""

from __future__ import annotations

from pathlib import Path
import json
import re
import time

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from astropy.time import Time
from astropy.io import fits

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (
    QApplication,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QGroupBox,
    QLineEdit,
    QCheckBox,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QTextEdit,
    QWidget,
    QDialog,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QDoubleSpinBox,
    QSpinBox,
    QSlider,
    QColorDialog,
    QTabWidget,
    QFileDialog,
    QListWidget,
    QListWidgetItem,
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QKeySequence, QColor
from PyQt5.QtWidgets import QShortcut, QStyle, QStyleOptionSlider, QSplitter, QProgressBar

from .step_window_base import StepWindowBase
from ...utils.common_helpers import safe_float as _safe_float, normalize_filter_key as _normalize_filter_key, parse_jd as _parse_jd
from ...utils.io_utils import (
    read_csv_int64_source_id,
    coerce_int64_source_id,
    load_night_assignments as _load_night_assignments_util,
    load_headers_table as _load_headers_table_util,
)
from ...utils.photometry_loader import load_frame_photometry


class _QcComputeWorker(QThread):
    """Background worker for run_comp_qc heavy computation."""
    finished = pyqtSignal(list)   # emits rows list
    error = pyqtSignal(str)

    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    def run(self):
        try:
            rows = self._fn()
            self.finished.emit(rows)
        except Exception as e:
            self.error.emit(str(e))


class ClickableSlider(QSlider):
    """QSlider that jumps to clicked position instead of stepping."""

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            opt = QStyleOptionSlider()
            self.initStyleOption(opt)
            groove = self.style().subControlRect(
                QStyle.CC_Slider, opt, QStyle.SC_SliderGroove, self
            )
            handle = self.style().subControlRect(
                QStyle.CC_Slider, opt, QStyle.SC_SliderHandle, self
            )
            if self.orientation() == Qt.Horizontal:
                slider_length = handle.width()
                slider_min = groove.x()
                slider_max = groove.right() - slider_length + 1
                pos = event.pos().x()
            else:
                slider_length = handle.height()
                slider_min = groove.y()
                slider_max = groove.bottom() - slider_length + 1
                pos = event.pos().y()

            if slider_max != slider_min:
                value = self.minimum() + (self.maximum() - self.minimum()) * (pos - slider_min) / (slider_max - slider_min)
                value = int(round(value))
                value = max(self.minimum(), min(self.maximum(), value))
                self.setValue(value)
                # Emit sliderReleased to trigger plot update
                self.sliderReleased.emit()
                event.accept()
                return
        super().mousePressEvent(event)
from ...utils.astro_utils import compute_airmass_from_header, compute_bjd_tdb_array
from ...utils.step_paths import (
    step1_dir,
    step2_cropped_dir,
    step5_photometry_dir,
    step7_refbuild_dir,
    step9_selection_dir,
    step10_dir,
    legacy_step11_extinction_dir,
    legacy_step11_zeropoint_dir,
    tool_extinction_dir,
)
from ...utils.qc_utils import load_frame_excludes, save_frame_excludes as save_frame_excludes_file
from ...analysis.light_curve.lightcurve_output_service import (
    annotate_raw_lightcurve,
    save_combined_raw_outputs,
    save_dataset_raw_outputs,
)


def _safe_int_list(text: str) -> list[int]:
    if not text:
        return []
    items = []
    for part in text.replace(";", ",").split(","):
        s = part.strip()
        if not s:
            continue
        try:
            items.append(int(s))
        except Exception:
            continue
    return items


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


def _build_source_to_id_map(df: pd.DataFrame) -> dict[int, int]:
    if not {"source_id", "ID"} <= set(df.columns):
        return {}
    sid_vals = coerce_int64_source_id(df["source_id"])
    id_vals = pd.to_numeric(df["ID"], errors="coerce").astype("Int64")
    mapping: dict[int, int] = {}
    for sid_val, id_val in zip(sid_vals, id_vals):
        if pd.isna(sid_val) or pd.isna(id_val):
            continue
        sid_int = int(sid_val)
        if sid_int not in mapping:
            mapping[sid_int] = int(id_val)
    return mapping


def _select_rows_by_source_id(df: pd.DataFrame, source_id: int | None) -> pd.DataFrame:
    if source_id is None or "source_id" not in df.columns:
        return pd.DataFrame()
    sid_series = coerce_int64_source_id(df["source_id"])
    return df.loc[sid_series == int(source_id)]


def _fmt_percent(value, default: str = "") -> str:
    try:
        if value is None:
            return default
        v = float(value)
        if not np.isfinite(v):
            return default
        return f"{v * 100:.1f}%"
    except Exception:
        return default


def _date_from_dateobs(date_obs: str | None) -> str:
    if not date_obs:
        return "unknown"
    try:
        t = Time(str(date_obs).strip())
        return t.to_datetime().strftime("%Y-%m-%d")
    except Exception:
        return "unknown"


def _display_date_from_dateobs(date_obs: str | None, tz_offset_hours: float = 0.0) -> str:
    """Convert DATE-OBS to a display date, optionally shifted to site local time."""
    if not date_obs:
        return "unknown"
    try:
        t = Time(str(date_obs).strip())
        tz_days = float(tz_offset_hours) / 24.0
        if np.isfinite(tz_days) and tz_days != 0.0:
            t = Time(t.jd + tz_days, format="jd", scale="utc")
        return t.to_datetime().strftime("%Y-%m-%d")
    except Exception:
        return _date_from_dateobs(date_obs)


def _load_headers_table(result_dir: Path) -> pd.DataFrame:
    return _load_headers_table_util(result_dir)


def _load_headers_map(result_dir: Path) -> dict:
    df = _load_headers_table(result_dir)
    if df.empty:
        return {}
    if "Filename" in df.columns and "DATE-OBS" in df.columns:
        return dict(zip(df["Filename"].astype(str), df["DATE-OBS"].astype(str)))
    return {}


def _load_night_assignments(result_dir: Path) -> dict[str, int]:
    """Load filename -> night_id mapping from step1 night_assignments.json."""
    return _load_night_assignments_util(result_dir)


def _parse_color_index(expr: str | None) -> tuple[str, str] | None:
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


def _normalize_color_index_by_filter(mapping) -> dict[str, str]:
    if not isinstance(mapping, dict):
        return {}
    out: dict[str, str] = {}
    for key, value in mapping.items():
        fkey = _normalize_filter_key(key)
        expr = str(value).strip()
        if fkey and expr:
            out[fkey] = expr
    return out


# Standard color index combinations (blue filter - red filter)
COLOR_INDEX_PAIRS = [
    ("g", "r"),  # SDSS
    ("b", "v"),  # Johnson
    ("v", "r"),  # Johnson
    ("r", "i"),  # SDSS
    ("v", "i"),  # Johnson-Cousins
    ("b", "r"),  # Extended
]


def _auto_detect_color_index(available_filters: set[str]) -> tuple[str, str] | None:
    """Automatically detect the best color index pair from available filters."""
    filters_lower = {_normalize_filter_key(f) for f in available_filters}
    for f1, f2 in COLOR_INDEX_PAIRS:
        if f1 in filters_lower and f2 in filters_lower:
            return (f1, f2)
    return None


def _compute_star_median_mags(
    result_dir: Path,
    star_ids: list[int],
    filters: list[str],
) -> dict[int, dict[str, float]]:
    """Compute median magnitude for each star in each filter.

    Returns: {star_id: {"g": mag_g, "r": mag_r, ...}}
    """
    idx_path = step5_photometry_dir(result_dir) / "photometry_index.csv"
    if not idx_path.exists():
        return {}

    try:
        idx_df = pd.read_csv(idx_path)
    except Exception:
        return {}

    if "file" not in idx_df.columns:
        return {}

    # Normalize filters
    filters_normalized = [_normalize_filter_key(f) for f in filters]

    # Collect magnitudes per star per filter
    star_mags: dict[int, dict[str, list[float]]] = {
        int(sid): {f: [] for f in filters_normalized} for sid in star_ids
    }

    # --- Resolve TSV paths & filter for each frame upfront ---
    frame_info = []
    for _, idx_row in idx_df.iterrows():
        fname = str(idx_row["file"])
        phot_path = step5_photometry_dir(result_dir) / f"{fname}_photometry.tsv"
        if not phot_path.exists():
            continue

        filt = ""
        if "filter" in idx_df.columns:
            filt = _normalize_filter_key(str(idx_row.get("filter", "")))
        elif "FILTER" in idx_df.columns:
            filt = _normalize_filter_key(str(idx_row.get("FILTER", "")))

        if not filt or filt not in filters_normalized:
            continue
        frame_info.append((fname, filt))

    n_workers = min(8, len(frame_info)) if frame_info else 1
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        tsv_dfs = list(
            pool.map(
                lambda fi: load_frame_photometry(result_dir, fi[0], fi[1]),
                frame_info,
            )
        )

    # --- Process loaded DataFrames ---
    for (_, filt), df in zip(frame_info, tsv_dfs):
        if df is None or df.empty:
            continue
        if "ID" not in df.columns:
            continue

        try:
            df_ids = pd.to_numeric(df["ID"], errors="coerce").astype("Int64")
        except Exception:
            continue

        for sid in star_ids:
            sid_int = int(sid)
            mask = df_ids == sid_int
            if not mask.any():
                continue
            row = df[mask]
            if row.empty:
                continue
            mag = _safe_float(row["mag"].iloc[0])
            if np.isfinite(mag):
                star_mags[sid_int][filt].append(mag)

    # Compute medians
    result: dict[int, dict[str, float]] = {}
    for sid in star_ids:
        sid_int = int(sid)
        result[sid_int] = {}
        for filt in filters_normalized:
            mags = star_mags[sid_int].get(filt, [])
            if mags:
                result[sid_int][filt] = float(np.nanmedian(mags))
            else:
                result[sid_int][filt] = np.nan

    return result


def _compute_color_indices(
    star_median_mags: dict[int, dict[str, float]],
    color_pair: tuple[str, str],
) -> dict[int, float]:
    """Compute color index for each star.

    color_pair: (blue_filter, red_filter), e.g., ("g", "r")
    Returns: {star_id: color_index}
    """
    f_blue, f_red = color_pair
    result: dict[int, float] = {}
    for sid, mags in star_median_mags.items():
        m_blue = mags.get(f_blue, np.nan)
        m_red = mags.get(f_red, np.nan)
        if np.isfinite(m_blue) and np.isfinite(m_red):
            result[sid] = m_blue - m_red
        else:
            result[sid] = np.nan
    return result


def _normalize_color_term_by_filter(mapping) -> dict[str, float]:
    if not isinstance(mapping, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in mapping.items():
        fkey = _normalize_filter_key(key)
        try:
            kval = float(value)
        except Exception:
            continue
        if fkey:
            out[fkey] = kval
    return out


def _load_frame_airmass_map(result_dir: Path) -> tuple[dict[str, float], dict[str, str]]:
    # Check tool_extinction_dir first, then legacy_step11_zeropoint_dir
    path = tool_extinction_dir(result_dir) / "frame_airmass.csv"
    if not path.exists():
        path = legacy_step11_zeropoint_dir(result_dir) / "frame_airmass.csv"
    if not path.exists():
        path = result_dir / "frame_airmass.csv"
    if not path.exists():
        return {}, {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}, {}
    if "file" not in df.columns or "airmass" not in df.columns:
        return {}, {}
    file_col = df["file"].astype(str)
    airmass_col = pd.to_numeric(df["airmass"], errors="coerce")
    filt_col = df["filter"].astype(str) if "filter" in df.columns else pd.Series([""] * len(df))
    airmass_map = dict(zip(file_col, airmass_col))
    filter_map = dict(zip(file_col, filt_col))
    return airmass_map, filter_map


def _load_extfit_map(result_dir: Path) -> dict[str, dict[str, float]]:
    """필터별 소광계수 로드 (tool_extinction 우선, 레거시 호환)"""
    # Priority: tool_extinction_dir > legacy_step11_extinction_dir > legacy paths
    candidates = [
        tool_extinction_dir(result_dir) / "extinction_fit_by_filter.csv",
        legacy_step11_extinction_dir(result_dir) / "step11_extinction_fit_by_filter.csv",
        legacy_step11_zeropoint_dir(result_dir) / "step11_extinction" / "step11_extinction_fit_by_filter.csv",
        result_dir / "step11_extinction" / "step11_extinction_fit_by_filter.csv",
        result_dir / "extinction" / "extinction_fit_by_filter.csv",
    ]
    path = None
    for cand in candidates:
        if cand.exists():
            path = cand
            break
    if path is None:
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    if "filter" not in df.columns:
        return {}
    out: dict[str, dict[str, float]] = {}
    for _, row in df.iterrows():
        fkey = _normalize_filter_key(row.get("filter", ""))
        if not fkey:
            continue
        try:
            k1 = float(row.get("k1", row.get("k", np.nan)))  # k1 또는 k
        except Exception:
            k1 = np.nan
        try:
            k2 = float(row.get("k2", np.nan))
        except Exception:
            k2 = np.nan
        try:
            k_color = float(row.get("k_color", np.nan))  # 색 의존 소광 계수 k''
        except Exception:
            k_color = np.nan
        out[fkey] = {"k1": k1, "k2": k2, "k_color": k_color}
    return out


def _load_extfit_map_by_date(result_dir: Path) -> dict[tuple[str, str], dict[str, float]]:
    """(date, filter)별 소광계수 + m0 로드 (밤별 영점 보정용)"""
    # Priority: tool_extinction_dir > legacy_step11_extinction_dir > legacy paths
    candidates = [
        tool_extinction_dir(result_dir) / "extinction_fit_by_filter.csv",
        legacy_step11_extinction_dir(result_dir) / "step11_extinction_fit_by_filter.csv",
        legacy_step11_zeropoint_dir(result_dir) / "step11_extinction" / "step11_extinction_fit_by_filter.csv",
        result_dir / "step11_extinction" / "step11_extinction_fit_by_filter.csv",
        result_dir / "extinction" / "extinction_fit_by_filter.csv",
    ]
    path = None
    for cand in candidates:
        if cand.exists():
            path = cand
            break
    if path is None:
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    if "filter" not in df.columns:
        return {}

    out: dict[tuple[str, str], dict[str, float]] = {}
    for _, row in df.iterrows():
        date_val = str(row.get("date", "unknown")).strip()
        fkey = _normalize_filter_key(row.get("filter", ""))
        if not fkey:
            continue

        try:
            k1 = float(row.get("k1", row.get("k", np.nan)))
        except Exception:
            k1 = np.nan
        try:
            k2 = float(row.get("k2", np.nan))
        except Exception:
            k2 = np.nan
        try:
            k_color = float(row.get("k_color", np.nan))
        except Exception:
            k_color = np.nan
        try:
            m0 = float(row.get("m0", np.nan))  # 밤별 영점 오프셋
        except Exception:
            m0 = np.nan

        out[(date_val, fkey)] = {"k1": k1, "k2": k2, "k_color": k_color, "m0": m0}
    return out


def _extract_date_from_path(path: Path | str | None = None, fname: str = "") -> str:
    """폴더 경로 또는 파일명에서 날짜 추출 (YYYY-MM-DD 또는 YYYYMMDD)

    우선순위:
    1. 폴더 경로에서 날짜 추출 (result_dir 이름)
    2. 파일명에서 날짜 추출 (날짜__파일명 형식)
    """
    import re

    # 1. 폴더 경로에서 날짜 추출
    if path is not None:
        path_str = str(path)
        # 경로의 각 부분에서 날짜 패턴 찾기
        for part in reversed(Path(path_str).parts):
            # YYYY-MM-DD 패턴
            m = re.match(r"(\d{4}-\d{2}-\d{2})", part)
            if m:
                return m.group(1)
            # YYYYMMDD 패턴
            m = re.match(r"(\d{8})", part)
            if m:
                d = m.group(1)
                return f"{d[:4]}-{d[4:6]}-{d[6:8]}"

    # 2. 파일명에서 날짜 추출 (레거시: 날짜__파일명 형식)
    if fname and "__" in fname:
        folder_part = fname.split("__")[0]
        m = re.match(r"(\d{4}-\d{2}-\d{2})", folder_part)
        if m:
            return m.group(1)
        m = re.match(r"(\d{8})", folder_part)
        if m:
            d = m.group(1)
            return f"{d[:4]}-{d[4:6]}-{d[6:8]}"

    return "unknown"


def _extract_date_from_filename(fname: str) -> str:
    """파일명에서 날짜 추출 (레거시 호환)"""
    return _extract_date_from_path(fname=fname)


def _resolve_fits_path(
    data_dir: Path,
    result_dir: Path,
    fname: str,
    file_path_map: dict | None = None,
) -> Path | None:
    if isinstance(file_path_map, dict):
        mapped = file_path_map.get(fname)
        if mapped:
            return Path(mapped)
    candidates = [
        data_dir / fname,
        result_dir / fname,
        step2_cropped_dir(result_dir) / fname,
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def _get_color_index_map(result_dir: Path, color_index_by_filter: dict[str, str]) -> dict[str, dict[int, float]]:
    if not color_index_by_filter:
        return {}
    # Check legacy_step11_zeropoint_dir for median data
    candidates = [
        legacy_step11_zeropoint_dir(result_dir) / "median_by_ID_filter_wide.csv",
        legacy_step11_zeropoint_dir(result_dir) / "median_by_ID_filter_wide_cmd.csv",
        result_dir / "median_by_ID_filter_wide.csv",
        result_dir / "median_by_ID_filter_wide_cmd.csv",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    if "ID" not in df.columns:
        return {}
    ids = pd.to_numeric(df["ID"], errors="coerce").astype("Int64")
    id_vals = ids.to_numpy()
    out: dict[str, dict[int, float]] = {}
    for filt, expr in color_index_by_filter.items():
        bands = _parse_color_index(expr)
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
        for id_val, c in zip(id_vals, color.to_numpy(float)):
            if pd.isna(id_val):
                continue
            cmap[int(id_val)] = float(c) if np.isfinite(c) else np.nan
        out[filt] = cmap
    return out


def _load_selection_ids(result_dir: Path) -> tuple[int | None, list[int]]:
    # per-filter selection_{filter}.json 우선 (가장 최신)
    filter_sel = _load_selection_ids_by_filter(result_dir)
    if filter_sel:
        target_ids = {
            int(sel["target_id"])
            for sel in filter_sel.values()
            if sel.get("target_id") is not None
        }
        comp_sets = {
            tuple(int(x) for x in sel.get("comparison_ids", []) if x is not None)
            for sel in filter_sel.values()
            if sel.get("target_id") is not None
        }
        if len(target_ids) == 1 and len(comp_sets) == 1:
            return next(iter(target_ids)), list(next(iter(comp_sets)))
        if len(target_ids) == 1:
            # Filters may legitimately carry different comp sets (or some may be
            # empty). Seed Step 10 with the union so plotting/QC can start from
            # the full candidate pool without silently preferring one filter.
            union_comp_ids = sorted({
                int(x)
                for sel in filter_sel.values()
                for x in sel.get("comparison_ids", [])
                if x is not None
            })
            return next(iter(target_ids)), union_comp_ids
        return None, []

    # fallback: legacy target_selection.json + source_id mapping
    selection_path = step9_selection_dir(result_dir) / "target_selection.json"
    if not selection_path.exists():
        selection_path = result_dir / "target_selection.json"
    if not selection_path.exists():
        return None, []
    try:
        data = json.loads(selection_path.read_text(encoding="utf-8"))
        target_id = data.get("target_id")
        comp_ids = data.get("comparison_ids", [])
        target_sid = data.get("target_source_id")
        comp_sids = data.get("comparison_source_ids", [])
        filter_key = None
        filter_targets = data.get("filter_targets")
        if isinstance(filter_targets, dict) and filter_targets:
            filter_key = next(iter(filter_targets.keys()))
        if not filter_key:
            filter_key = data.get("filter")

        def _load_step8_final_id_map(flt: str | None) -> dict[int, int]:
            step9_out = step9_selection_dir(result_dir)
            if not step9_out.exists():
                return {}
            candidates = []
            if flt:
                candidates.append(step9_out / f"master_catalog_{_normalize_filter_key(flt)}.tsv")
            candidates.extend(sorted(step9_out.glob("master_catalog_*.tsv")))
            for path in candidates:
                if not path.exists():
                    continue
                try:
                    df = read_csv_int64_source_id(path, sep="\t")
                    id_map = _build_source_to_id_map(df)
                    if id_map:
                        return id_map
                except Exception:
                    continue
            return {}

        final_id_map = _load_step8_final_id_map(filter_key)
        if final_id_map:
            if target_sid is not None:
                target_id = final_id_map.get(int(target_sid))
            if comp_sids:
                comp_ids = [final_id_map.get(int(s)) for s in comp_sids if int(s) in final_id_map]
        if target_id is None:
            src_path = step7_refbuild_dir(result_dir) / "sourceid_to_ID.csv"
            src_id = target_sid
            if src_path.exists() and src_id is not None:
                try:
                    df = read_csv_int64_source_id(src_path)
                    if {"source_id", "ID"} <= set(df.columns):
                        row = _select_rows_by_source_id(df, int(src_id))
                        if not row.empty:
                            id_val = pd.to_numeric(row.iloc[0]["ID"], errors="coerce")
                            if pd.notna(id_val):
                                target_id = int(id_val)
                except Exception:
                    target_id = None
        if (not comp_ids) and comp_sids:
            src_path = step7_refbuild_dir(result_dir) / "sourceid_to_ID.csv"
            src_ids = [int(s) for s in comp_sids if s is not None]
            if src_path.exists() and src_ids:
                try:
                    df = read_csv_int64_source_id(src_path)
                    if {"source_id", "ID"} <= set(df.columns):
                        sid_series = coerce_int64_source_id(df["source_id"])
                        id_series = pd.to_numeric(df["ID"], errors="coerce").astype("Int64")
                        mask = sid_series.notna() & sid_series.isin(src_ids) & id_series.notna()
                        comp_ids = id_series[mask].astype("int64").tolist()
                except Exception:
                    comp_ids = []
        if target_id is not None:
            target_id = int(target_id)
        comp_ids = [int(x) for x in comp_ids if x is not None]
        return target_id, comp_ids
    except Exception:
        return None, []


def _load_selection_ids_by_filter(result_dir: Path) -> dict:
    """필터별 selection 로드 (Step 8에서 저장한 selection_{filter}.json)"""
    step9_out = step9_selection_dir(result_dir)
    filter_selections = {}

    if not step9_out.exists():
        return {}

    id_map_cache: dict[str, dict[int, int]] = {}

    def _load_step8_id_map(flt: str) -> dict[int, int]:
        key = _normalize_filter_key(flt)
        if key in id_map_cache:
            return id_map_cache[key]
        mapping: dict[int, int] = {}
        candidates = [
            (step9_out / f"master_catalog_{key}.tsv", "\t"),
            (step9_out / f"id_mapping_{key}.csv", ","),
        ]
        for path, sep in candidates:
            if not path.exists():
                continue
            try:
                df = read_csv_int64_source_id(path, sep=sep)
            except Exception:
                continue
            id_map = _build_source_to_id_map(df)
            for sid_int, id_val in id_map.items():
                if sid_int not in mapping:
                    mapping[sid_int] = id_val
        id_map_cache[key] = mapping
        return mapping

    def _load_step7_refbuild_id_map() -> dict[int, int]:
        key = "__step7_refbuild__"
        if key in id_map_cache:
            return id_map_cache[key]
        candidates = [step7_refbuild_dir(result_dir) / "sourceid_to_ID.csv"]
        mapping: dict[int, int] = {}
        for path in candidates:
            if not path.exists():
                continue
            try:
                df = read_csv_int64_source_id(path)
            except Exception:
                continue
            id_map = _build_source_to_id_map(df)
            for sid_int, id_val in id_map.items():
                mapping[sid_int] = id_val
            if mapping:
                break
        id_map_cache[key] = mapping
        return mapping

    for sel_path in sorted(step9_out.glob("selection_*.json")):
        flt = sel_path.stem.replace("selection_", "")
        try:
            data = json.loads(sel_path.read_text(encoding="utf-8"))
            target_id = data.get("target_id")
            comp_ids = data.get("comparison_ids", [])
            target_source_id = data.get("target_source_id")
            comp_source_ids = data.get("comparison_source_ids", [])

            target_id_val = int(target_id) if target_id is not None else None
            comp_id_vals = [int(x) for x in comp_ids if x is not None]

            # Recover IDs from source IDs when selection JSON has null/empty final IDs.
            if target_id_val is None and target_source_id is not None:
                sid_map = _load_step8_id_map(flt)
                if int(target_source_id) in sid_map:
                    target_id_val = int(sid_map[int(target_source_id)])
                else:
                    refbuild_map = _load_step7_refbuild_id_map()
                    if int(target_source_id) in refbuild_map:
                        target_id_val = int(refbuild_map[int(target_source_id)])
            if not comp_id_vals and comp_source_ids:
                sid_map = _load_step8_id_map(flt)
                if sid_map:
                    comp_id_vals = sorted({
                        int(sid_map[int(x)])
                        for x in comp_source_ids
                        if x is not None and int(x) in sid_map
                    })
                if not comp_id_vals:
                    refbuild_map = _load_step7_refbuild_id_map()
                    comp_id_vals = sorted({
                        int(refbuild_map[int(x)])
                        for x in comp_source_ids
                        if x is not None and int(x) in refbuild_map
                    })

            # Check star
            check_source_id = data.get("check_source_id")
            check_id = data.get("check_id")
            check_id_val = int(check_id) if check_id is not None else None
            if check_id_val is None and check_source_id is not None:
                sid_map = _load_step8_id_map(flt)
                if int(check_source_id) in sid_map:
                    check_id_val = int(sid_map[int(check_source_id)])
                else:
                    refbuild_map = _load_step7_refbuild_id_map()
                    if int(check_source_id) in refbuild_map:
                        check_id_val = int(refbuild_map[int(check_source_id)])

            filter_selections[flt] = {
                "target_id": target_id_val,
                "comparison_ids": comp_id_vals,
                "target_source_id": int(target_source_id) if target_source_id is not None else None,
                "comparison_source_ids": [int(x) for x in comp_source_ids if x is not None],
                "check_id": check_id_val,
                "check_source_id": int(check_source_id) if check_source_id is not None else None,
            }
        except Exception:
            continue

    return filter_selections


def _load_check_star_meta_by_filter(result_dir: Path) -> dict[str, dict[str, int]]:
    """Load per-filter check star metadata from Step 8 selection JSONs."""
    out: dict[str, dict[str, int]] = {}
    filter_sel = _load_selection_ids_by_filter(result_dir)
    for flt, sel in sorted(filter_sel.items()):
        key = _normalize_filter_key(flt)
        if not key:
            continue
        entry: dict[str, int] = {}
        check_id = sel.get("check_id")
        check_source_id = sel.get("check_source_id")
        if check_id is not None:
            entry["check_id"] = int(check_id)
        if check_source_id is not None:
            entry["check_source_id"] = int(check_source_id)
        if entry:
            out[key] = entry
    return out


def _load_check_star_id(result_dir: Path, filt: str | None = None):
    """Load check star ID, optionally for a specific filter."""
    meta_by_filter = _load_check_star_meta_by_filter(result_dir)
    if filt:
        entry = meta_by_filter.get(_normalize_filter_key(filt), {})
        cid = entry.get("check_id")
        return int(cid) if cid is not None else None
    for flt in sorted(meta_by_filter):
        cid = meta_by_filter[flt].get("check_id")
        if cid is not None:
            return int(cid)
    return None


def _load_check_star_csv(result_dir: Path, filt: str | None = None) -> tuple:
    """Load check star CSV from step10 output.

    If ``filt`` is given, prefer the filter-specific check-star CSV for that band.
    Otherwise prefer the combined check-star CSV that merges all configured filters.
    """
    out_dir = step10_dir(result_dir)
    if not out_dir.exists():
        return None, pd.DataFrame()

    if filt:
        filt_key = _normalize_filter_key(filt)
        check_id = _load_check_star_id(result_dir, filt_key)
        candidates = []
        if check_id is not None and filt_key:
            candidates.append(out_dir / f"lightcurve_check_{filt_key}_ID{check_id}_raw.csv")
            candidates.append(out_dir / f"lightcurve_check_ID{check_id}_raw.csv")
        candidates.append(out_dir / "lightcurve_check_combined_raw.csv")
        for path in candidates:
            if not path.exists():
                continue
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            if "filter" in df.columns and filt_key:
                df = df[df["filter"].astype(str).map(_normalize_filter_key) == filt_key].copy()
            if "check_id" in df.columns and check_id is not None:
                df = df[pd.to_numeric(df["check_id"], errors="coerce") == int(check_id)].copy()
            if not df.empty:
                return check_id, df
        return check_id, pd.DataFrame()

    combined_path = out_dir / "lightcurve_check_combined_raw.csv"
    if combined_path.exists():
        try:
            df = pd.read_csv(combined_path)
            cid = None
            if "check_id" in df.columns:
                ids = sorted({
                    int(x) for x in pd.to_numeric(df["check_id"], errors="coerce").dropna().astype(int).tolist()
                })
                if len(ids) == 1:
                    cid = ids[0]
            return cid, df
        except Exception:
            pass

    check_id = _load_check_star_id(result_dir)
    if check_id is not None:
        p = out_dir / f"lightcurve_check_ID{check_id}_raw.csv"
        if p.exists():
            try:
                return check_id, pd.read_csv(p)
            except Exception:
                pass

    for p in sorted(out_dir.glob("lightcurve_check_ID*_raw.csv")):
        try:
            cid = int(p.stem.replace("lightcurve_check_ID", "").replace("_raw", ""))
            return cid, pd.read_csv(p)
        except Exception:
            continue

    return None, pd.DataFrame()


def _load_target_radec(result_dir: Path, target_id: int) -> tuple[float, float]:
    """Look up target RA/Dec from master_catalog.tsv.

    Returns (ra_deg, dec_deg), or (nan, nan) if not found.
    """
    step9_out = step9_selection_dir(result_dir)
    candidates = list(step9_out.glob("master_catalog_*.tsv")) if step9_out.exists() else []
    candidates += [step9_out / "master_catalog.tsv"] if step9_out.exists() else []
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
            ra = float(pd.to_numeric(row["ra_deg"].values[0], errors="coerce")) if "ra_deg" in df.columns else np.nan
            dec = float(pd.to_numeric(row["dec_deg"].values[0], errors="coerce")) if "dec_deg" in df.columns else np.nan
            if np.isfinite(ra) and np.isfinite(dec):
                return ra, dec
        except Exception:
            continue
    return np.nan, np.nan


# 필터별 일관된 색상 매핑
FILTER_COLORS = {
    "g": "#2ca02c",   # 녹색
    "r": "#d62728",   # 빨간색
    "i": "#9467bd",   # 보라색
    "z": "#8c564b",   # 갈색
    "u": "#1f77b4",   # 파란색
    "b": "#1f77b4",   # 파란색
    "v": "#2ca02c",   # 녹색
    "clear": "#7f7f7f",  # 회색
    "l": "#7f7f7f",   # luminance - 회색
    "unknown": "#7f7f7f",
}


class LightCurveBuilderWindow(StepWindowBase):
    """Step 10: Light curve builder (diff/abs)."""

    def __init__(self, params, file_manager, project_state, main_window, runtime_mode: bool = False):
        self.file_manager = file_manager
        self.runtime_mode = bool(runtime_mode)
        self.datasets = []  # will be set to single entry in setup
        self.dataset_panel_expanded = False
        self.comp_ids_list = []
        self.comp_index = 0
        self.comp_candidate_ids = []

        # 파라미터 설정 (기본값)
        self.opt_diff = True  # 차등 라이트커브 (raw)
        self.x_axis_mode = "time"  # "time" or "phase"
        self.phase_period = 0.0  # 주기 (일)
        self.phase_t0 = 0.0  # 기준 시각 (JD)
        self.phase_cycles = 1.0  # 표시할 phase 사이클 수
        # 슬라이더 범위 설정
        self.period_min = 0.01  # 최소 주기 (일)
        self.period_max = 10.0  # 최대 주기 (일)
        self.t0_min = 0.0  # T0 오프셋 최소
        self.t0_max = 1.0  # T0 오프셋 최대 (주기 대비 비율)

        # 필터별 selection 캐시
        self._filter_selections: dict = {}

        # Diff series 캐시 (파일 재로드 방지)
        self._diff_series_cache: dict[tuple, pd.DataFrame] = {}
        self._diff_series_cache_key: tuple | None = None
        # Check star series 캐시 (result_dir, comp_ids_tuple) → (ids_dict, df)
        self._check_series_cache: dict[tuple, tuple[dict, pd.DataFrame]] = {}

        # FITS 헤더 캐시 (매번 FITS 파일 열기 방지)
        self._header_cache: dict[str, fits.Header | None] = {}

        # 측광 TSV 캐시 (파일별 측광 데이터)
        self._photometry_cache: dict[str, pd.DataFrame] = {}
        self._photometry_cache_dir: Path | None = None

        # QC 캐시
        self.qc_rows: list[dict] = []
        self._qc_table_block = False
        self.qc_sigma = 3.0
        self.qc_rms_max = 0.02
        self.qc_outlier_frac_max = 0.1
        self.qc_min_points = 10
        self.qc_scale_mode = "Robust(MAD)"
        self.qc_scale_mad_value = 5.0
        self.qc_scale_fixed_value = 0.2
        self.qc_date_last = None

        # Frame QC (manual exclude)
        self._frame_exclude_cache: dict[str, dict[str, set[str]]] = {}
        self._frame_exclude_dirty: set[str] = set()
        self._frame_qc_selected: str | None = None
        self._selected_point_xy: tuple[float, float] | None = None  # data coords of selected point
        self._frame_qc_selected_dir: Path | None = None
        self._frame_qc_total = 0
        self._plot_point_map: dict[object, dict[str, object]] = {}
        self._lc_ever_plotted: bool = False
        self._qc_plot_point_map: dict[object, dict[str, object]] = {}
        self._qc_scope_df: pd.DataFrame | None = None
        self._qc_scope_files_all_filters: list[str] | None = None
        self._qc_date_label_map: dict[str, str] = {}
        self._frame_qc_done_cache: dict[str, bool] = {}
        # photometry_index 읽기 캐시 (result_dir → (mtime, df))
        self._qc_idx_cache: dict[str, tuple[float, pd.DataFrame]] = {}
        # headers 캐시
        self._qc_headers_cache: dict[str, tuple[float, dict, pd.DataFrame]] = {}

        # 필터별 플롯 표시/색상 설정
        self.filter_visibility: dict[str, bool] = {}
        self.filter_colors: dict[str, str] = {}
        self._filter_keys: list[str] = []

        super().__init__(
            step_index=9,
            step_name="Light Curve Builder",
            params=params,
            project_state=project_state,
            main_window=main_window,
        )
        self.setFocusPolicy(Qt.StrongFocus)
        if self.runtime_mode:
            self._setup_runtime_ui()
        else:
            self.setup_step_ui()
            self.restore_state()
            self._auto_load_ids()

    def _setup_runtime_ui(self):
        """Build a lightweight runtime-only UI for merger inline execution."""
        note = QLabel("Runtime mode: merger inline Step10 execution")
        note.setStyleSheet("QLabel { color: #607D8B; font-size: 9pt; }")
        self.content_layout.addWidget(note)

        self.target_edit = QLineEdit()
        self.comp_edit = QLineEdit()
        self.id_info_label = QLabel("Target / Comp: (runtime)")
        self.id_info_label.setWordWrap(True)
        self.plot_info_label = QLabel("Comparison: (none)")
        self.lbl_comp_count = QLabel("Active comps: 0")

        rd = Path(self.params.P.result_dir)
        self.datasets = [(rd.name, rd)]

        self.log_window = QWidget(self, Qt.Window)
        self.log_window.setWindowTitle("Light Curve Log")
        log_layout = QVBoxLayout(self.log_window)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)

    def setup_step_ui(self):
        info = QLabel(
            "대상/비교성 선택 결과를 이용해 라이트커브를 생성합니다.\n"
            "RAW 차등측광을 생성하고 비교성 QC를 수행합니다."
        )
        info.setStyleSheet("QLabel { background-color: #E3F2FD; padding: 10px; border-radius: 5px; }")
        self.content_layout.addWidget(info)

        # Hidden QLineEdits (used internally by build logic)
        self.target_edit = QLineEdit()
        self.comp_edit = QLineEdit()

        # Read-only info bar for target / comparison IDs (auto-loaded from Step 9)
        self.id_info_label = QLabel("Target / Comp: (loading...)")
        self.id_info_label.setStyleSheet(
            "QLabel { background-color: #E8F5E9; padding: 8px 12px; border-radius: 5px;"
            " font-weight: bold; color: #2E7D32; }"
        )
        self.id_info_label.setWordWrap(True)
        self.content_layout.addWidget(self.id_info_label)

        # Fix base dataset to current result_dir
        rd = Path(self.params.P.result_dir)
        self.datasets = [(rd.name, rd)]

        # ----- 추가 데이터셋 패널 -----
        ds_group = QGroupBox("추가 데이터셋")
        ds_group.setStyleSheet("QGroupBox { font-size: 8pt; } QGroupBox::title { color: #555; }")
        self.ds_group = ds_group
        ds_vbox = QVBoxLayout(ds_group)
        ds_vbox.setContentsMargins(4, 4, 4, 4)
        ds_vbox.setSpacing(2)

        ds_header_row = QHBoxLayout()
        ds_header_row.setSpacing(6)
        self.btn_ds_toggle = QPushButton("▶ 펼치기")
        self.btn_ds_toggle.setCheckable(True)
        self.btn_ds_toggle.setChecked(False)
        self.btn_ds_toggle.setStyleSheet(
            "QPushButton { border: none; text-align: left; color: #555; font-size: 8pt; padding: 0px; }"
            "QPushButton:checked { color: #1565C0; }"
        )
        self.btn_ds_toggle.toggled.connect(lambda checked: self._set_dataset_panel_expanded(checked, persist=True))
        ds_header_row.addWidget(self.btn_ds_toggle)

        self.ds_summary_label = QLabel()
        self.ds_summary_label.setStyleSheet("QLabel { color: #607D8B; font-size: 8pt; }")
        ds_header_row.addWidget(self.ds_summary_label)
        ds_header_row.addStretch()
        ds_vbox.addLayout(ds_header_row)

        self.ds_container = QWidget()
        ds_container_layout = QVBoxLayout(self.ds_container)
        ds_container_layout.setContentsMargins(0, 0, 0, 0)
        ds_container_layout.setSpacing(4)

        self.ds_list_widget = QListWidget()
        self.ds_list_widget.setMaximumHeight(64)
        self.ds_list_widget.setStyleSheet("QListWidget { font-size: 8pt; }")
        self.ds_list_widget.addItem(f"[현재] {rd.name}")
        self.ds_list_widget.item(0).setFlags(Qt.ItemIsEnabled)
        ds_container_layout.addWidget(self.ds_list_widget)

        ds_btn_row = QHBoxLayout()
        ds_btn_row.setSpacing(4)
        btn_ds_add = QPushButton("폴더 추가")
        btn_ds_add.setMaximumHeight(22)
        btn_ds_add.setStyleSheet("font-size: 8pt;")
        btn_ds_add.clicked.connect(self._on_add_dataset)
        btn_ds_remove = QPushButton("제거")
        btn_ds_remove.setMaximumHeight(22)
        btn_ds_remove.setStyleSheet("font-size: 8pt;")
        btn_ds_remove.clicked.connect(self._on_remove_dataset)
        ds_btn_row.addWidget(btn_ds_add)
        ds_btn_row.addWidget(btn_ds_remove)
        ds_btn_row.addStretch()
        ds_container_layout.addLayout(ds_btn_row)
        ds_vbox.addWidget(self.ds_container)
        self.content_layout.addWidget(ds_group)
        self._update_dataset_summary()
        self._set_dataset_panel_expanded(False, persist=False)

        self.tab_widget = QTabWidget()
        self.light_tab = QWidget()
        self.qc_tab = QWidget()
        self.light_layout = QVBoxLayout(self.light_tab)
        self.qc_layout = QVBoxLayout(self.qc_tab)
        self.tab_widget.addTab(self.qc_tab, "Comparison QC")
        self.tab_widget.addTab(self.light_tab, "Light Curve")
        self.content_layout.addWidget(self.tab_widget, 1)

        plot_group = QGroupBox("Target - Comparison Light Curve")
        plot_group.setStyleSheet(
            "QGroupBox { background-color: #F7F9FB; border: 1px solid #CFD8DC; border-radius: 8px; margin-top: 8px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #37474F; font-weight: bold; }"
        )
        plot_layout = QVBoxLayout(plot_group)
        plot_hint = QLabel("←/→ 키로 비교성을 전환합니다.")
        plot_hint.setStyleSheet("QLabel { color: #455A64; }")
        plot_layout.addWidget(plot_hint)

        self.plot_info_label = QLabel("Comparison: (none)")
        self.plot_info_label.setStyleSheet("QLabel { font-weight: bold; }")
        plot_layout.addWidget(self.plot_info_label)

        # 컨트롤 버튼 row
        btn_row = QHBoxLayout()

        # Parameters 버튼
        btn_params = QPushButton("Parameters")
        btn_params.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; padding: 4px 10px; }")
        btn_params.clicked.connect(self.show_parameters_dialog)
        btn_row.addWidget(btn_params)

        # X축 모드 선택
        btn_row.addWidget(QLabel("X축:"))
        self.x_axis_combo = QComboBox()
        self.x_axis_combo.addItems(["Time (hr)", "Phase"])
        self.x_axis_combo.setCurrentIndex(0)
        self.x_axis_combo.currentIndexChanged.connect(self._on_xaxis_changed)
        btn_row.addWidget(self.x_axis_combo)

        self.btn_filter_colors = QPushButton("Browse Colors")
        self.btn_filter_colors.setStyleSheet(
            "QPushButton { background-color: #ECEFF1; border: 1px solid #B0BEC5; border-radius: 4px; padding: 4px 10px; }"
            "QPushButton:hover { border: 1px solid #78909C; }"
        )
        self.btn_filter_colors.clicked.connect(self.show_filter_color_browser)
        btn_row.addWidget(self.btn_filter_colors)

        btn_row.addStretch()

        # Plot 버튼 (자동 저장 포함)
        self.btn_plot = QPushButton("Plot && Save")
        self.btn_plot.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 4px 16px; font-size: 10pt; }")
        self.btn_plot.clicked.connect(self.plot_and_save)
        btn_row.addWidget(self.btn_plot)

        plot_layout.addLayout(btn_row)

        self.plot_canvas = FigureCanvas(Figure(figsize=(8, 3.5)))
        self.plot_toolbar = NavigationToolbar(self.plot_canvas, self)
        plot_layout.addWidget(self.plot_toolbar)
        self.plot_ax = self.plot_canvas.figure.add_subplot(111)
        self.plot_canvas.setFocusPolicy(Qt.ClickFocus)
        self.plot_canvas.setMinimumHeight(220)
        self.plot_canvas.setStyleSheet("background-color: #FFFFFF; border: 1px solid #ECEFF1;")
        self.plot_canvas.mpl_connect("button_press_event", self._on_plot_click)
        plot_layout.addWidget(self.plot_canvas, 1)

        # Phase Folding 슬라이더
        phase_box = QGroupBox("Phase Folding")
        phase_layout = QVBoxLayout(phase_box)

        # Period 슬라이더 (클릭으로 이동 가능)
        period_row = QHBoxLayout()
        period_row.addWidget(QLabel("Period:"))
        self.period_slider = ClickableSlider(Qt.Horizontal)
        self.period_slider.setRange(0, 1000)  # 0~1000 -> period_min ~ period_max
        self.period_slider.setValue(0)
        self.period_slider.setSingleStep(1)
        self.period_slider.setPageStep(50)
        self.period_slider.valueChanged.connect(self._on_period_slider_preview)  # Preview only (no plot)
        self.period_slider.sliderReleased.connect(self._on_period_slider_released)  # Plot on release
        period_row.addWidget(self.period_slider)
        self.period_label = QLabel("0.000 d")
        self.period_label.setMinimumWidth(80)
        period_row.addWidget(self.period_label)
        phase_layout.addLayout(period_row)

        # T0 슬라이더 (클릭으로 이동 가능)
        t0_row = QHBoxLayout()
        t0_row.addWidget(QLabel("T0 offset:"))
        self.t0_slider = ClickableSlider(Qt.Horizontal)
        self.t0_slider.setRange(0, 1000)  # 0~1000 -> 0 ~ period (주기 내 오프셋)
        self.t0_slider.setValue(0)
        self.t0_slider.setSingleStep(1)
        self.t0_slider.setPageStep(50)
        self.t0_slider.valueChanged.connect(self._on_t0_slider_preview)  # Preview only (no plot)
        self.t0_slider.sliderReleased.connect(self._on_t0_slider_released)  # Plot on release
        t0_row.addWidget(self.t0_slider)
        self.t0_label = QLabel("0.000")
        self.t0_label.setMinimumWidth(80)
        t0_row.addWidget(self.t0_label)
        phase_layout.addLayout(t0_row)

        plot_layout.addWidget(phase_box)

        # Frame QC bar — below phase sliders
        frame_qc_row = QHBoxLayout()
        frame_qc_row.addWidget(QLabel("더블클릭=선택  D=제외  A=복원  R=리셋"))
        self.frame_qc_selected_label = QLabel("Selected: (none)")
        self.frame_qc_selected_label.setStyleSheet("QLabel { font-weight: bold; color: #1565C0; }")
        frame_qc_row.addWidget(self.frame_qc_selected_label, 1)
        self.frame_qc_summary_label = QLabel("Excluded: 0/0")
        self.frame_qc_summary_label.setStyleSheet("QLabel { color: #546E7A; }")
        frame_qc_row.addWidget(self.frame_qc_summary_label)
        self.btn_frame_qc_save = QPushButton("Save")
        self.btn_frame_qc_save.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 3px 10px; }"
            "QPushButton:disabled { background-color: #CFD8DC; color: #90A4AE; }"
        )
        self.btn_frame_qc_save.clicked.connect(self.save_frame_excludes)
        self.btn_frame_qc_save.setEnabled(False)
        frame_qc_row.addWidget(self.btn_frame_qc_save)
        self.btn_frame_qc_clear = QPushButton("Reset")
        self.btn_frame_qc_clear.setStyleSheet(
            "QPushButton { background-color: #F44336; color: white; font-weight: bold; padding: 3px 10px; }"
            "QPushButton:disabled { background-color: #CFD8DC; color: #90A4AE; }"
        )
        self.btn_frame_qc_clear.clicked.connect(self.clear_frame_excludes)
        self.btn_frame_qc_clear.setEnabled(False)
        frame_qc_row.addWidget(self.btn_frame_qc_clear)
        plot_layout.addLayout(frame_qc_row)

        self.light_layout.addWidget(plot_group, 1)

        qc_group = QGroupBox("Comparison QC")
        qc_group.setStyleSheet(
            "QGroupBox { background-color: #F7F9FB; border: 1px solid #CFD8DC; border-radius: 8px; margin-top: 8px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #37474F; font-weight: bold; }"
        )
        qc_layout = QVBoxLayout(qc_group)

        qc_btn_row = QHBoxLayout()
        btn_qc_params = QPushButton("QC Parameters")
        btn_qc_params.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; padding: 4px 10px; }")
        btn_qc_params.clicked.connect(self.show_qc_parameters_dialog)
        qc_btn_row.addWidget(btn_qc_params)

        self.btn_qc_run = QPushButton("Run QC")
        self.btn_qc_run.setStyleSheet(
            "QPushButton { background-color: #FF9800; color: white; font-weight: bold; padding: 4px 12px; }"
            "QPushButton:disabled { background-color: #CFD8DC; color: #90A4AE; }"
        )
        self.btn_qc_run.clicked.connect(self.run_comp_qc)
        qc_btn_row.addWidget(self.btn_qc_run)

        self.btn_qc_auto = QPushButton("Auto Use")
        self.btn_qc_auto.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 4px 12px; }"
            "QPushButton:disabled { background-color: #CFD8DC; color: #90A4AE; }"
        )
        self.btn_qc_auto.clicked.connect(self.auto_use_qc)
        qc_btn_row.addWidget(self.btn_qc_auto)

        qc_btn_row.addStretch()
        self.lbl_comp_count = QLabel("Active comps: 0")
        self.lbl_comp_count.setStyleSheet("QLabel { font-weight: bold; color: #37474F; }")
        qc_btn_row.addWidget(self.lbl_comp_count)
        qc_layout.addLayout(qc_btn_row)

        self.lbl_qc_thresholds = QLabel()
        self.lbl_qc_thresholds.setStyleSheet("QLabel { color: #546E7A; }")
        qc_layout.addWidget(self.lbl_qc_thresholds)
        self._update_qc_threshold_label()

        self.qc_progress_bar = QProgressBar()
        self.qc_progress_bar.setRange(0, 0)  # indeterminate (marquee)
        self.qc_progress_bar.setMaximumHeight(6)
        self.qc_progress_bar.setTextVisible(False)
        self.qc_progress_bar.setStyleSheet(
            "QProgressBar { border: none; background-color: #ECEFF1; border-radius: 3px; }"
            "QProgressBar::chunk { background-color: #FF9800; border-radius: 3px; }"
        )
        self.qc_progress_bar.hide()
        qc_layout.addWidget(self.qc_progress_bar)

        self.qc_status_label = QLabel("")
        self.qc_status_label.setStyleSheet("QLabel { color: #546E7A; font-weight: bold; }")
        self.qc_status_label.hide()
        qc_layout.addWidget(self.qc_status_label)

        self.qc_table = QTableWidget()
        self.qc_table.setColumnCount(7)
        self.qc_table.setHorizontalHeaderLabels(
            ["Use", "ID", "N", "RMS", "σ(night)", "MAD", "Out%"]
        )
        _qc_hdr = self.qc_table.horizontalHeader()
        _qc_hdr.setSectionResizeMode(QHeaderView.ResizeToContents)
        _qc_hdr.setStretchLastSection(False)
        _qc_hdr.setSectionResizeMode(6, QHeaderView.Fixed)
        self.qc_table.setColumnWidth(6, 52)
        self.qc_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.qc_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.qc_table.itemSelectionChanged.connect(self._on_qc_selection_changed)
        self.qc_table.itemChanged.connect(self._on_qc_table_changed)

        # Stats panel (right of table)
        stats_group = QGroupBox("Selected Comp Stats")
        stats_vbox = QVBoxLayout(stats_group)
        stats_vbox.setSpacing(4)
        self.qc_stats_night_cb = QCheckBox("밤별 구분")
        self.qc_stats_night_cb.stateChanged.connect(self._refresh_qc_stats_panel)
        stats_vbox.addWidget(self.qc_stats_night_cb)
        self.qc_stats_table = QTableWidget()
        self.qc_stats_table.setColumnCount(5)
        self.qc_stats_table.setHorizontalHeaderLabels(["Filter", "N", "Median", "σ", "MAD"])
        _sh = self.qc_stats_table.horizontalHeader()
        _sh.setSectionResizeMode(QHeaderView.ResizeToContents)
        _sh.setStretchLastSection(True)
        self.qc_stats_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.qc_stats_table.setSelectionBehavior(QTableWidget.SelectRows)
        stats_vbox.addWidget(self.qc_stats_table)
        stats_group.setMinimumWidth(200)

        # Top pane: table (left) + stats (right) — horizontal splitter
        top_splitter = QSplitter(Qt.Horizontal)
        top_splitter.setChildrenCollapsible(False)
        top_splitter.addWidget(self.qc_table)
        top_splitter.addWidget(stats_group)
        top_splitter.setStretchFactor(0, 3)
        top_splitter.setStretchFactor(1, 2)
        top_splitter.setMinimumHeight(160)

        # Bottom pane: date/filter controls + preview canvas
        check_group = QGroupBox("Comparison Preview")
        check_layout = QVBoxLayout(check_group)
        check_layout.setContentsMargins(4, 4, 4, 4)
        check_layout.setSpacing(4)

        preview_ctrl = QHBoxLayout()
        preview_ctrl.addWidget(QLabel("Date:"))
        self.qc_date_combo = QComboBox()
        self.qc_date_combo.addItem("All")
        self.qc_date_combo.currentIndexChanged.connect(self._on_qc_date_changed)
        preview_ctrl.addWidget(self.qc_date_combo)
        preview_ctrl.addWidget(QLabel("Filter:"))
        self.qc_filter_combo = QComboBox()
        self.qc_filter_combo.addItem("All")
        self.qc_filter_combo.currentIndexChanged.connect(self._on_qc_preview_changed)
        preview_ctrl.addWidget(self.qc_filter_combo)
        preview_ctrl.addStretch()
        check_layout.addLayout(preview_ctrl)

        self.check_plot_canvas = FigureCanvas(Figure(figsize=(6, 4)))
        self.check_plot_ax = self.check_plot_canvas.figure.add_subplot(111)
        self.check_plot_canvas.setMinimumHeight(260)
        self.check_plot_canvas.setStyleSheet("background-color: #FFFFFF; border: 1px solid #ECEFF1;")
        self.check_plot_canvas.mpl_connect("button_press_event", self._on_qc_plot_click)
        self.check_plot_toolbar = NavigationToolbar(self.check_plot_canvas, self)
        check_layout.addWidget(self.check_plot_toolbar)
        check_layout.addWidget(self.check_plot_canvas, 1)

        qc_frame_row = QHBoxLayout()
        qc_frame_row.addWidget(QLabel("더블클릭=선택  D=제외  A=복원  R=리셋"))
        self.qc_frame_selected_label = QLabel("Selected: (none)")
        self.qc_frame_selected_label.setStyleSheet("QLabel { font-weight: bold; color: #1565C0; }")
        qc_frame_row.addWidget(self.qc_frame_selected_label, 1)
        self.qc_frame_summary_label = QLabel("Excluded: 0/0")
        self.qc_frame_summary_label.setStyleSheet("QLabel { color: #546E7A; }")
        qc_frame_row.addWidget(self.qc_frame_summary_label)
        self.btn_qc_frame_save = QPushButton("Save")
        self.btn_qc_frame_save.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 3px 10px; }"
            "QPushButton:disabled { background-color: #CFD8DC; color: #90A4AE; }"
        )
        self.btn_qc_frame_save.clicked.connect(self.save_frame_excludes)
        self.btn_qc_frame_save.setEnabled(False)
        qc_frame_row.addWidget(self.btn_qc_frame_save)
        self.btn_qc_frame_reset = QPushButton("Reset")
        self.btn_qc_frame_reset.setStyleSheet(
            "QPushButton { background-color: #F44336; color: white; font-weight: bold; padding: 3px 10px; }"
            "QPushButton:disabled { background-color: #CFD8DC; color: #90A4AE; }"
        )
        self.btn_qc_frame_reset.clicked.connect(self.clear_frame_excludes)
        self.btn_qc_frame_reset.setEnabled(False)
        qc_frame_row.addWidget(self.btn_qc_frame_reset)
        check_layout.addLayout(qc_frame_row)

        # Vertical splitter: top=table+stats, bottom=preview
        qc_splitter = QSplitter(Qt.Vertical)
        qc_splitter.setChildrenCollapsible(False)
        qc_splitter.addWidget(top_splitter)
        qc_splitter.addWidget(check_group)
        qc_splitter.setStretchFactor(0, 2)
        qc_splitter.setStretchFactor(1, 3)

        qc_layout.addWidget(qc_splitter, 1)
        self.qc_layout.addWidget(qc_group, 1)

        self.shortcut_prev = QShortcut(QKeySequence(Qt.Key_Left), self)
        self.shortcut_prev.activated.connect(lambda: self._step_comp(-1))
        self.shortcut_next = QShortcut(QKeySequence(Qt.Key_Right), self)
        self.shortcut_next.activated.connect(lambda: self._step_comp(1))
        self.shortcut_exclude = QShortcut(QKeySequence(Qt.Key_D), self)
        self.shortcut_exclude.activated.connect(self._exclude_selected_frame)
        self.shortcut_include = QShortcut(QKeySequence(Qt.Key_A), self)
        self.shortcut_include.activated.connect(self._include_selected_frame)
        self.shortcut_reset = QShortcut(QKeySequence(Qt.Key_R), self)
        self.shortcut_reset.activated.connect(self.clear_frame_excludes)

        self.tab_widget.currentChanged.connect(self._on_tab_changed)

        log_row = QHBoxLayout()
        btn_log = QPushButton("Log")
        btn_log.setStyleSheet("QPushButton { background-color: #607D8B; color: white; font-weight: bold; padding: 6px 12px; }")
        btn_log.clicked.connect(self.show_log_window)
        log_row.addWidget(btn_log)
        log_row.addStretch()
        self.content_layout.addLayout(log_row)

        self.log_window = QWidget(self, Qt.Window)
        self.log_window.setWindowTitle("Light Curve Log")
        self.log_window.resize(800, 400)
        log_layout = QVBoxLayout(self.log_window)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("QTextEdit { font-family: monospace; font-size: 9pt; }")
        log_layout.addWidget(self.log_text)

        # build button moved to plot group
        self._update_qc_gate_ui()
        self._init_qc_view()

    def log(self, msg: str):
        self.log_text.append(msg)

    def _on_tab_changed(self, idx: int) -> None:
        self._update_frame_qc_summary()
        self._update_qc_gate_ui()

    def _on_qc_date_changed(self, _idx: int) -> None:
        self.qc_date_last = self.qc_date_combo.currentText()
        self._on_qc_preview_changed()

    def _on_qc_preview_changed(self, *_args) -> None:
        comp_id = self._get_qc_selected_comp_id()
        if comp_id is not None:
            self._plot_comp_preview(comp_id)

    def _compute_airmass(self, header: fits.Header | None) -> float:
        if header is None:
            return np.nan
        lat = float(getattr(self.params.P, "site_lat_deg", 0.0))
        lon = float(getattr(self.params.P, "site_lon_deg", 0.0))
        alt = float(getattr(self.params.P, "site_alt_m", 0.0))
        tz = float(getattr(self.params.P, "site_tz_offset_hours", 0.0))
        formula = getattr(self.params.P, "airmass_formula", None)
        try:
            hdr = header.copy()
            if "AIRMASS" in hdr:
                del hdr["AIRMASS"]
        except Exception:
            hdr = header
        info = compute_airmass_from_header(hdr, lat, lon, alt, tz, formula=formula)
        am = _safe_float(info.get("airmass", np.nan))
        if np.isfinite(am):
            return float(am)
        return _safe_float(header.get("AIRMASS", np.nan))

    def show_log_window(self):
        self.log_window.show()
        self.log_window.raise_()
        self.log_window.activateWindow()

    def show_parameters_dialog(self):
        """파라미터 설정 다이얼로그"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Light Curve Parameters")
        dialog.setMinimumWidth(400)
        layout = QVBoxLayout(dialog)

        layout.addWidget(QLabel("─" * 40))
        layout.addWidget(QLabel("QC 자동 선택 기준:"))

        form_qc = QFormLayout()
        spin_qc_rms = QDoubleSpinBox()
        spin_qc_rms.setDecimals(4)
        spin_qc_rms.setRange(0.0, 1.0)
        spin_qc_rms.setValue(self.qc_rms_max)
        spin_qc_rms.setSuffix(" mag")
        form_qc.addRow("RMS 최대:", spin_qc_rms)

        spin_qc_sigma = QDoubleSpinBox()
        spin_qc_sigma.setDecimals(2)
        spin_qc_sigma.setRange(1.0, 10.0)
        spin_qc_sigma.setValue(self.qc_sigma)
        form_qc.addRow("Outlier sigma(MAD):", spin_qc_sigma)

        spin_qc_frac = QDoubleSpinBox()
        spin_qc_frac.setDecimals(3)
        spin_qc_frac.setRange(0.0, 1.0)
        spin_qc_frac.setSingleStep(0.01)
        spin_qc_frac.setValue(self.qc_outlier_frac_max)
        form_qc.addRow("Outlier frac 최대:", spin_qc_frac)

        spin_qc_n = QSpinBox()
        spin_qc_n.setRange(1, 1000)
        spin_qc_n.setValue(self.qc_min_points)
        form_qc.addRow("최소 포인트:", spin_qc_n)

        layout.addLayout(form_qc)

        layout.addWidget(QLabel("─" * 40))
        layout.addWidget(QLabel("QC Preview Y-Scale:"))

        form_scale = QFormLayout()
        combo_scale = QComboBox()
        combo_scale.addItems(["Auto", "Robust(MAD)", "Fixed"])
        combo_scale.setCurrentText(self.qc_scale_mode)
        form_scale.addRow("Mode:", combo_scale)

        spin_scale_mad = QDoubleSpinBox()
        spin_scale_mad.setDecimals(2)
        spin_scale_mad.setRange(0.5, 50.0)
        spin_scale_mad.setValue(self.qc_scale_mad_value)
        form_scale.addRow("MAD x:", spin_scale_mad)

        spin_scale_fixed = QDoubleSpinBox()
        spin_scale_fixed.setDecimals(3)
        spin_scale_fixed.setRange(0.001, 5.0)
        spin_scale_fixed.setValue(self.qc_scale_fixed_value)
        form_scale.addRow("±mag:", spin_scale_fixed)

        layout.addLayout(form_scale)

        layout.addWidget(QLabel("─" * 40))
        layout.addWidget(QLabel("Phase Folding 슬라이더 범위 설정:"))

        form2 = QFormLayout()

        # 주기 최소값
        spin_period_min = QDoubleSpinBox()
        spin_period_min.setDecimals(4)
        spin_period_min.setRange(0.001, 100.0)
        spin_period_min.setValue(self.period_min)
        spin_period_min.setSuffix(" days")
        form2.addRow("Period 최소:", spin_period_min)

        # 주기 최대값
        spin_period_max = QDoubleSpinBox()
        spin_period_max.setDecimals(4)
        spin_period_max.setRange(0.01, 1000.0)
        spin_period_max.setValue(self.period_max)
        spin_period_max.setSuffix(" days")
        form2.addRow("Period 최대:", spin_period_max)

        # Phase 표시 범위 (사이클 수)
        spin_phase_cycles = QDoubleSpinBox()
        spin_phase_cycles.setDecimals(2)
        spin_phase_cycles.setRange(1.0, 5.0)
        spin_phase_cycles.setSingleStep(0.1)
        spin_phase_cycles.setValue(float(self.phase_cycles))
        spin_phase_cycles.setSuffix(" cycles")
        form2.addRow("Phase 범위:", spin_phase_cycles)

        layout.addLayout(form2)

        # 버튼
        btn_row = QHBoxLayout()
        btn_save = QPushButton("Save")
        btn_save.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        btn_save.clicked.connect(dialog.accept)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(dialog.reject)
        btn_row.addStretch()
        btn_row.addWidget(btn_save)
        btn_row.addWidget(btn_cancel)
        layout.addLayout(btn_row)

        if dialog.exec_() == QDialog.Accepted:
            self.qc_rms_max = float(spin_qc_rms.value())
            self.qc_sigma = float(spin_qc_sigma.value())
            self.qc_outlier_frac_max = float(spin_qc_frac.value())
            self.qc_min_points = int(spin_qc_n.value())
            self.qc_scale_mode = combo_scale.currentText()
            self.qc_scale_mad_value = float(spin_scale_mad.value())
            self.qc_scale_fixed_value = float(spin_scale_fixed.value())
            self.period_min = spin_period_min.value()
            self.period_max = spin_period_max.value()
            self.phase_cycles = float(spin_phase_cycles.value())
            self.log(
                f"[PARAM] qc_rms_max={self.qc_rms_max:.4f}, qc_sigma={self.qc_sigma:.2f}, "
                f"qc_outlier_frac={self.qc_outlier_frac_max:.3f}, qc_min_points={self.qc_min_points}, "
                f"qc_scale_mode={self.qc_scale_mode}, "
                f"qc_scale_mad={self.qc_scale_mad_value:.2f}, qc_scale_fixed={self.qc_scale_fixed_value:.3f}, "
                f"period_range=[{self.period_min:.4f}, {self.period_max:.4f}], "
                f"phase_cycles={self.phase_cycles}"
            )
            self.save_state()
            # 슬라이더 업데이트
            self._update_sliders_from_values()
            self._update_qc_threshold_label()
            self._on_qc_preview_changed()
            if self.x_axis_mode == "phase":
                self.plot_current_comparison()

    def show_qc_parameters_dialog(self):
        """QC 파라미터 설정 다이얼로그 (QC 전용)"""
        dialog = QDialog(self)
        dialog.setWindowTitle("QC Parameters")
        dialog.setMinimumWidth(360)
        layout = QVBoxLayout(dialog)

        layout.addWidget(QLabel("QC 자동 선택 기준:"))
        form_qc = QFormLayout()

        spin_qc_rms = QDoubleSpinBox()
        spin_qc_rms.setDecimals(4)
        spin_qc_rms.setRange(0.0, 1.0)
        spin_qc_rms.setValue(self.qc_rms_max)
        spin_qc_rms.setSuffix(" mag")
        form_qc.addRow("RMS 최대:", spin_qc_rms)

        spin_qc_sigma = QDoubleSpinBox()
        spin_qc_sigma.setDecimals(2)
        spin_qc_sigma.setRange(1.0, 10.0)
        spin_qc_sigma.setValue(self.qc_sigma)
        form_qc.addRow("Outlier sigma(MAD):", spin_qc_sigma)

        spin_qc_frac = QDoubleSpinBox()
        spin_qc_frac.setDecimals(3)
        spin_qc_frac.setRange(0.0, 1.0)
        spin_qc_frac.setSingleStep(0.01)
        spin_qc_frac.setValue(self.qc_outlier_frac_max)
        form_qc.addRow("Outlier frac 최대:", spin_qc_frac)

        spin_qc_n = QSpinBox()
        spin_qc_n.setRange(1, 1000)
        spin_qc_n.setValue(self.qc_min_points)
        form_qc.addRow("최소 포인트:", spin_qc_n)

        layout.addLayout(form_qc)

        layout.addWidget(QLabel("QC Preview Y-Scale:"))
        form_scale = QFormLayout()
        combo_scale = QComboBox()
        combo_scale.addItems(["Auto", "Robust(MAD)", "Fixed"])
        combo_scale.setCurrentText(self.qc_scale_mode)
        form_scale.addRow("Mode:", combo_scale)

        spin_scale_mad = QDoubleSpinBox()
        spin_scale_mad.setDecimals(2)
        spin_scale_mad.setRange(0.5, 50.0)
        spin_scale_mad.setValue(self.qc_scale_mad_value)
        form_scale.addRow("MAD x:", spin_scale_mad)

        spin_scale_fixed = QDoubleSpinBox()
        spin_scale_fixed.setDecimals(3)
        spin_scale_fixed.setRange(0.001, 5.0)
        spin_scale_fixed.setValue(self.qc_scale_fixed_value)
        form_scale.addRow("±mag:", spin_scale_fixed)

        layout.addLayout(form_scale)

        btn_row = QHBoxLayout()
        btn_save = QPushButton("Save")
        btn_save.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        btn_save.clicked.connect(dialog.accept)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(dialog.reject)
        btn_row.addStretch()
        btn_row.addWidget(btn_save)
        btn_row.addWidget(btn_cancel)
        layout.addLayout(btn_row)

        if dialog.exec_() == QDialog.Accepted:
            self.qc_rms_max = float(spin_qc_rms.value())
            self.qc_sigma = float(spin_qc_sigma.value())
            self.qc_outlier_frac_max = float(spin_qc_frac.value())
            self.qc_min_points = int(spin_qc_n.value())
            self.qc_scale_mode = combo_scale.currentText()
            self.qc_scale_mad_value = float(spin_scale_mad.value())
            self.qc_scale_fixed_value = float(spin_scale_fixed.value())
            self.log(
                f"[PARAM][QC] rms_max={self.qc_rms_max:.4f}, sigma={self.qc_sigma:.2f}, "
                f"outlier_frac={self.qc_outlier_frac_max:.3f}, min_points={self.qc_min_points}, "
                f"scale_mode={self.qc_scale_mode}, "
                f"scale_mad={self.qc_scale_mad_value:.2f}, scale_fixed={self.qc_scale_fixed_value:.3f}"
            )
            self.save_state()
            self._update_qc_threshold_label()
            self._on_qc_preview_changed()

    def _on_xaxis_changed(self, idx: int):
        """X축 모드 변경"""
        modes = ["time", "phase"]
        self.x_axis_mode = modes[idx] if idx < len(modes) else "time"
        self.plot_current_comparison()  # 플롯 갱신

    def _on_period_slider_preview(self, value: int):
        """Period 슬라이더 드래그 중 - 라벨만 업데이트 (플롯 없음)"""
        # 0~1000 -> period_min ~ period_max (로그 스케일)
        if value == 0:
            self.phase_period = 0.0
        else:
            # 로그 스케일로 변환
            import math
            log_min = math.log10(self.period_min)
            log_max = math.log10(self.period_max)
            log_val = log_min + (log_max - log_min) * (value / 1000.0)
            self.phase_period = 10 ** log_val
        self.period_label.setText(f"{self.phase_period:.4f} d")

    def _on_period_slider_released(self):
        """Period 슬라이더 릴리즈 - 플롯 업데이트"""
        if self.x_axis_mode == "phase":
            self.plot_current_comparison()

    def _on_t0_slider_preview(self, value: int):
        """T0 슬라이더 드래그 중 - 라벨만 업데이트 (플롯 없음)"""
        # 0~1000 -> 0 ~ phase_period
        if self.phase_period > 0:
            self.phase_t0 = self.phase_period * (value / 1000.0)
        else:
            self.phase_t0 = 0.0
        self.t0_label.setText(f"{self.phase_t0:.4f} d")

    def _on_t0_slider_released(self):
        """T0 슬라이더 릴리즈 - 플롯 업데이트"""
        if self.x_axis_mode == "phase":
            self.plot_current_comparison()

    def _update_sliders_from_values(self):
        """현재 period/t0 값으로 슬라이더 위치 업데이트"""
        import math
        # Period 슬라이더
        if self.phase_period <= 0:
            self.period_slider.setValue(0)
        else:
            log_min = math.log10(self.period_min)
            log_max = math.log10(self.period_max)
            log_val = math.log10(max(self.phase_period, self.period_min))
            slider_val = int(1000 * (log_val - log_min) / (log_max - log_min))
            self.period_slider.setValue(min(max(slider_val, 0), 1000))
        self.period_label.setText(f"{self.phase_period:.4f} d")

        # T0 슬라이더
        if self.phase_period > 0:
            slider_val = int(1000 * (self.phase_t0 / self.phase_period))
            self.t0_slider.setValue(min(max(slider_val, 0), 1000))
        else:
            self.t0_slider.setValue(0)
        self.t0_label.setText(f"{self.phase_t0:.4f} d")

    def _filter_key_for_ui(self, value: str) -> str:
        key = _normalize_filter_key(value)
        if key in ("", "nan", "none", "unknown"):
            return "unknown"
        return key

    def _get_filter_color(self, key: str) -> str:
        return self.filter_colors.get(key, FILTER_COLORS.get(key, "#ff7f0e"))

    def _apply_filter_swatch_style(self, button: QPushButton, color: str) -> None:
        button.setStyleSheet(
            f"QPushButton {{ background-color: {color}; border: 1px solid #455A64; border-radius: 3px; min-width: 24px; min-height: 18px; }}"
            "QPushButton:hover { border: 1px solid #263238; }"
        )

    def _ensure_filter_colors(self, filters: list[str]) -> None:
        keys = sorted({self._filter_key_for_ui(f) for f in filters})
        if keys == self._filter_keys:
            for key in keys:
                if key not in self.filter_colors:
                    self.filter_colors[key] = FILTER_COLORS.get(key, "#ff7f0e")
            return
        self._filter_keys = keys
        for key in keys:
            if key not in self.filter_colors:
                self.filter_colors[key] = FILTER_COLORS.get(key, "#ff7f0e")

    def _available_filter_keys(self) -> list[str]:
        keys = list(self._filter_keys)
        if not keys:
            keys = sorted(self.filter_colors.keys())
        return keys

    def _reset_filter_colors(self, keys: list[str] | None = None) -> None:
        target_keys = keys or self._available_filter_keys()
        for key in target_keys:
            self.filter_colors[key] = FILTER_COLORS.get(key, "#ff7f0e")
        self.save_state()
        self.plot_current_comparison()

    def _choose_filter_color(self, key: str, preview_button: QPushButton | None = None, parent: QWidget | None = None) -> bool:
        current = QColor(self._get_filter_color(key))
        picked = QColorDialog.getColor(current, parent or self, f"Select color for {key}")
        if not picked.isValid():
            return False
        self.filter_colors[key] = picked.name()
        if preview_button is not None:
            self._apply_filter_swatch_style(preview_button, self._get_filter_color(key))
        self.save_state()
        self.plot_current_comparison()
        return True

    def show_filter_color_browser(self):
        keys = self._available_filter_keys()
        if not keys:
            QMessageBox.information(self, "Filter Colors", "먼저 라이트커브를 한 번 표시한 뒤 색상을 변경하세요.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Filter Colors")
        dialog.setMinimumWidth(320)
        layout = QVBoxLayout(dialog)

        info = QLabel("필터별 색상을 선택합니다.")
        info.setStyleSheet("QLabel { color: #455A64; }")
        layout.addWidget(info)

        swatch_buttons: dict[str, QPushButton] = {}
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(6)

        for row, key in enumerate(keys):
            label = QLabel(key)
            label.setStyleSheet("QLabel { color: #37474F; font-weight: bold; }")
            grid.addWidget(label, row, 0)

            swatch = QPushButton("")
            swatch.setFixedSize(28, 20)
            swatch.setFocusPolicy(Qt.NoFocus)
            self._apply_filter_swatch_style(swatch, self._get_filter_color(key))
            grid.addWidget(swatch, row, 1)
            swatch_buttons[key] = swatch

            browse_btn = QPushButton("Browse...")
            browse_btn.clicked.connect(
                lambda _checked=False, k=key, sw=swatch, dlg=dialog: self._choose_filter_color(k, sw, dlg)
            )
            grid.addWidget(browse_btn, row, 2)

        layout.addLayout(grid)

        btn_row = QHBoxLayout()
        btn_reset = QPushButton("Reset Colors")
        btn_reset.clicked.connect(lambda: self._reset_filter_color_browser(keys, swatch_buttons))
        btn_row.addWidget(btn_reset)
        btn_row.addStretch()

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(dialog.accept)
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

        dialog.exec_()

    def _reset_filter_color_browser(self, keys: list[str], swatch_buttons: dict[str, QPushButton]) -> None:
        self._reset_filter_colors(keys)
        for key, swatch in swatch_buttons.items():
            self._apply_filter_swatch_style(swatch, self._get_filter_color(key))

    def _expand_phase_cycles(self, df: pd.DataFrame) -> pd.DataFrame:
        cycles = float(self.phase_cycles)
        if cycles <= 1.0 or "phase" not in df.columns:
            return df
        full = int(np.floor(cycles))
        frac = cycles - full
        frames = []
        full = max(full, 1)
        for k in range(full):
            dup = df.copy()
            dup["phase"] = dup["phase"] + k
            frames.append(dup)
        if frac > 1e-6:
            dup = df.copy()
            dup["phase"] = dup["phase"] + full
            dup = dup[dup["phase"] <= (full + frac + 1e-9)]
            frames.append(dup)
        return pd.concat(frames, ignore_index=True)

    def plot_and_save(self):
        """Plot 버튼 클릭 시 라이트커브 빌드 + 자동저장 + 플롯"""
        self.build_light_curve()

    def _auto_load_ids(self):
        """Step 9 selection에서 target/comp ID를 자동 로드."""
        rd = Path(self.params.P.result_dir)
        target_id, comp_ids = _load_selection_ids(rd)
        if target_id is not None:
            self.target_edit.setText(str(target_id))
        if comp_ids:
            self.comp_edit.setText(",".join(str(i) for i in comp_ids))
            self.comp_candidate_ids = list(comp_ids)
        self._update_comp_ids_from_input()
        self._update_id_info_label()

    def _update_id_info_label(self):
        """Read-only info label 갱신."""
        t = self.target_edit.text().strip()
        c = self.comp_edit.text().strip()
        if t and c:
            self.id_info_label.setText(f"Target: ID {t}  |  Comp: {c}")
            self.id_info_label.setStyleSheet(
                "QLabel { background-color: #E8F5E9; padding: 8px 12px; border-radius: 5px;"
                " font-weight: bold; color: #2E7D32; }"
            )
        elif t:
            self.id_info_label.setText(f"Target: ID {t}  |  Comp: (없음)")
            self.id_info_label.setStyleSheet(
                "QLabel { background-color: #FFF8E1; padding: 8px 12px; border-radius: 5px;"
                " font-weight: bold; color: #F57C00; }"
            )
        else:
            self.id_info_label.setText("Target / Comp: Step 9에서 선택해주세요")
            self.id_info_label.setStyleSheet(
                "QLabel { background-color: #FFEBEE; padding: 8px 12px; border-radius: 5px;"
                " font-weight: bold; color: #C62828; }"
            )

    def clear_diff_series_cache(self, clear_headers: bool = False):
        """캐시 클리어 (데이터셋/선택 변경 시 호출)"""
        self._diff_series_cache.clear()
        self._photometry_cache.clear()
        self._photometry_cache_dir = None
        self._check_series_cache.clear()
        if clear_headers:
            self._header_cache.clear()
            self.log("[CACHE] All caches cleared")
        else:
            self.log("[CACHE] Diff series + photometry cache cleared")

    def load_from_selection(self):
        """Alias for _auto_load_ids (backwards compat)."""
        self._auto_load_ids()
        self.plot_current_comparison()
        self._update_qc_gate_ui()

    def _update_comp_ids_from_input(self):
        self.comp_ids_list = _safe_int_list(self.comp_edit.text())
        if self.comp_ids_list:
            # Keep candidate list in sync with currently active IDs loaded from selection/input.
            merged = sorted({int(x) for x in self.comp_candidate_ids} | set(self.comp_ids_list))
            self.comp_candidate_ids = merged
        elif not self.comp_candidate_ids:
            self.comp_candidate_ids = []
        if self.comp_index >= len(self.comp_ids_list):
            self.comp_index = 0
        self._update_plot_info()
        self._update_comp_count_label()

    def _update_plot_info(self):
        if not self.comp_ids_list:
            self.plot_info_label.setText("Comparison: (none)")
            return
        comp_id = self.comp_ids_list[self.comp_index]
        self.plot_info_label.setText(
            f"Comparison {self.comp_index + 1}/{len(self.comp_ids_list)} | ID {comp_id}"
        )

    def _step_comp(self, delta: int):
        if not self.comp_ids_list:
            self._update_plot_info()
            return
        self.comp_index = (self.comp_index + delta) % len(self.comp_ids_list)
        self._update_plot_info()
        self.plot_current_comparison()

    def _get_header(self, result_dir: Path, fname: str, cache: dict) -> fits.Header | None:
        if fname in cache:
            return cache[fname]
        fpath = _resolve_fits_path(
            Path(self.params.P.data_dir),
            result_dir,
            fname,
            getattr(self.params.P, "file_path_map", None),
        )
        if fpath is None:
            cache[fname] = None
            return None
        try:
            hdr = fits.getheader(fpath)
        except Exception:
            hdr = None
        cache[fname] = hdr
        return hdr

    def _map_comp_source_id(self, sel: dict, comp_id: int) -> int | None:
        comp_ids_all = sel.get("comparison_ids", [])
        comp_source_ids_all = sel.get("comparison_source_ids", [])
        try:
            idx = comp_ids_all.index(comp_id)
        except ValueError:
            return None
        if 0 <= idx < len(comp_source_ids_all):
            return comp_source_ids_all[idx]
        return None

    def _get_photometry_df(self, result_dir: Path, fname: str) -> pd.DataFrame | None:
        """Load photometry TSV with caching."""
        # Clear cache if result_dir changed
        if self._photometry_cache_dir != result_dir:
            self._photometry_cache.clear()
            self._photometry_cache_dir = result_dir

        if fname in self._photometry_cache:
            return self._photometry_cache[fname]

        df = load_frame_photometry(result_dir, fname)

        self._photometry_cache[fname] = df
        return df

    def _preload_photometry_cache(self, result_dir: Path, filenames: list[str]):
        """Bulk-preload photometry TSVs using ThreadPoolExecutor."""
        if self._photometry_cache_dir != result_dir:
            self._photometry_cache.clear()
            self._photometry_cache_dir = result_dir

        # Only load files not yet cached
        to_load = [fn for fn in filenames if fn not in self._photometry_cache]
        if not to_load:
            return

        def _load_one(fname):
            df = load_frame_photometry(result_dir, fname)
            return fname, df

        n_workers = min(8, len(to_load))
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for fname, df in pool.map(_load_one, to_load):
                self._photometry_cache[fname] = df

    def _build_star_mag_series(
        self,
        result_dir: Path,
        star_id: int,
        verbose: bool = True,
        include_excluded: bool = False,
        files_override: list[str] | None = None,
        preload: bool = True,
    ) -> pd.DataFrame:
        idx_path = step5_photometry_dir(result_dir) / "photometry_index.csv"
        if not idx_path.exists():
            if verbose:
                self.log(f"[DEBUG] photometry_index.csv not found in {result_dir}")
            return pd.DataFrame()
        idx = pd.read_csv(idx_path)
        if "file" not in idx.columns:
            if verbose:
                self.log(f"[DEBUG] photometry_index.csv missing 'file' column")
            return pd.DataFrame()
        exclude_map = self._get_frame_exclude_map(result_dir)
        if exclude_map and not include_excluded:
            before = len(idx)
            idx = idx[~idx["file"].astype(str).isin(exclude_map.keys())]
            if verbose and before != len(idx):
                self.log(f"[Frame QC] Excluded frames: {before} → {len(idx)}")
        available_files = idx["file"].astype(str).tolist()
        if files_override is not None:
            allowed = set(available_files)
            files = [str(fname) for fname in files_override if str(fname) in allowed]
        else:
            files = available_files
        if preload:
            self._preload_photometry_cache(result_dir, files)
        headers_map = _load_headers_map(result_dir)
        headers_df = _load_headers_table(result_dir)

        filter_selections = _load_selection_ids_by_filter(result_dir)

        filter_map = {}
        if "filter" in idx.columns:
            filter_map = dict(zip(idx["file"].astype(str), idx["filter"].astype(str)))
        elif "FILTER" in idx.columns:
            filter_map = dict(zip(idx["file"].astype(str), idx["FILTER"].astype(str)))

        header_filter_map = {}
        if not headers_df.empty and "Filename" in headers_df.columns:
            for col in ("FILTER", "filter"):
                if col in headers_df.columns:
                    header_filter_map = dict(zip(headers_df["Filename"].astype(str), headers_df[col].astype(str)))
                    break

        # Use instance-level header cache (self._header_cache)
        times = []
        dates = []
        filters = []
        mags = []
        mag_errs = []

        n_found = 0
        n_fits_read = 0

        for fname in files:
            # 1) DATE-OBS, FILTER: headers.csv에서 먼저 시도
            date_obs = headers_map.get(fname) if headers_map else None
            jd = _parse_jd(date_obs)

            filt_val = filter_map.get(fname, "")
            if not filt_val:
                filt_val = header_filter_map.get(fname, "")

            # 2) 정보가 부족한 경우에만 FITS 헤더 읽기 (lazy load)
            need_fits = (not np.isfinite(jd)) or (not filt_val) or (not date_obs)
            hdr = None
            if need_fits:
                hdr = self._get_header(result_dir, fname, self._header_cache)
                if hdr is not None:
                    n_fits_read += 1
                    if not np.isfinite(jd):
                        jd = _parse_jd(hdr.get("DATE-OBS"))
                    if not date_obs:
                        date_obs = hdr.get("DATE-OBS")
                    if not filt_val:
                        filt_val = hdr.get("FILTER", hdr.get("FILTER1", hdr.get("FILTER2", "")))

            times.append(jd)
            if date_obs:
                dates.append(_date_from_dateobs(date_obs))
            else:
                dates.append(_extract_date_from_path(result_dir, fname))
            filt_key = _normalize_filter_key(filt_val)
            filters.append(filt_key)

            # Use photometry cache
            df = self._get_photometry_df(result_dir, fname)
            if df is None or df.empty:
                mags.append(np.nan)
                mag_errs.append(np.nan)
                continue

            use_source_id = False
            comp_source_id = None
            if filt_key in filter_selections:
                sel = filter_selections[filt_key]
                comp_source_id = self._map_comp_source_id(sel, star_id)
                use_source_id = True

            row = pd.DataFrame()
            if use_source_id and comp_source_id is not None and "source_id" in df.columns:
                row = _select_rows_by_source_id(df, int(comp_source_id))
            if row.empty and "ID" in df.columns:
                row = df[df["ID"] == int(star_id)]

            if row.empty:
                mags.append(np.nan)
                mag_errs.append(np.nan)
            else:
                n_found += 1
                mags.append(_safe_float(row["mag"].values[0]))
                mag_errs.append(_safe_float(row["mag_err"].values[0]) if "mag_err" in row.columns else np.nan)

        tarr = np.array(times, float)
        if np.all(~np.isfinite(tarr)):
            tarr = np.arange(len(files), dtype=float)
        t0 = np.nanmedian(tarr)
        rel_time_hr = (tarr - t0) * 24.0

        if verbose:
            total = len(files)
            self.log(f"[DEBUG] Star series ID={star_id} found in {n_found}/{total} frames")

        return pd.DataFrame({
            "file": files,
            "filter": filters,
            "date": dates,
            "JD": tarr,
            "rel_time_hr": rel_time_hr,
            "mag": np.array(mags, float),
            "mag_err": np.array(mag_errs, float),
        })

    def _build_ensemble_series(
        self,
        result_dir: Path,
        target_id: int,
        comp_ids: list[int],
        verbose: bool = True,
        target_source_id_by_filter: dict[str, int] | None = None,
    ) -> pd.DataFrame:
        idx_path = step5_photometry_dir(result_dir) / "photometry_index.csv"
        if not idx_path.exists():
            if verbose:
                self.log(f"[DEBUG] photometry_index.csv not found in {result_dir}")
            return pd.DataFrame()
        idx = pd.read_csv(idx_path)
        if "file" not in idx.columns:
            if verbose:
                self.log(f"[DEBUG] photometry_index.csv missing 'file' column")
            return pd.DataFrame()
        files = idx["file"].astype(str).tolist()
        self._preload_photometry_cache(result_dir, files)
        headers_map = _load_headers_map(result_dir)
        headers_df = _load_headers_table(result_dir)

        filter_selections = _load_selection_ids_by_filter(result_dir)

        filter_map = {}
        if "filter" in idx.columns:
            filter_map = dict(zip(idx["file"].astype(str), idx["filter"].astype(str)))
        elif "FILTER" in idx.columns:
            filter_map = dict(zip(idx["file"].astype(str), idx["FILTER"].astype(str)))

        header_filter_map = {}
        if not headers_df.empty and "Filename" in headers_df.columns:
            for col in ("FILTER", "filter"):
                if col in headers_df.columns:
                    header_filter_map = dict(zip(headers_df["Filename"].astype(str), headers_df[col].astype(str)))
                    break

        # Night assignment map: basename→night_id, merged from all sources
        night_id_map: dict[str, int] = {}
        raw_na = _load_night_assignments(result_dir)
        night_id_map.update({Path(str(k)).name: int(v) for k, v in raw_na.items() if int(v) > 0})
        fm = getattr(self, "file_manager", None)
        if fm is not None:
            for k, v in getattr(fm, "night_assignments", {}).items():
                if int(v) > 0:
                    night_id_map[Path(str(k)).name] = int(v)
        if "night_id" in idx.columns:
            for fn, nid in zip(idx["file"].astype(str), pd.to_numeric(idx["night_id"], errors="coerce")):
                bn = Path(fn).name
                if not pd.isna(nid) and int(nid) > 0 and bn not in night_id_map:
                    night_id_map[bn] = int(nid)

        # Use instance-level header cache (self._header_cache)
        times = []
        dates = []
        night_ids = []
        filters = []
        airmasses = []
        mags = []
        mag_errs = []
        comp_avgs = []
        comp_errs = []
        diffs = []
        diff_errs = []

        # headers.csv에서 airmass 컬럼 확인
        header_airmass_map = {}
        if not headers_df.empty and "Filename" in headers_df.columns:
            for col in ("AIRMASS", "airmass", "AM"):
                if col in headers_df.columns:
                    for _, row in headers_df.iterrows():
                        fn = str(row["Filename"])
                        am_val = pd.to_numeric(row[col], errors="coerce")
                        if np.isfinite(am_val):
                            header_airmass_map[fn] = float(am_val)
                    break

        n_target_found = 0
        n_comp_found = 0

        for fname in files:
            # 1) DATE-OBS, FILTER, AIRMASS: headers.csv에서 먼저 시도
            date_obs = headers_map.get(fname) if headers_map else None
            jd = _parse_jd(date_obs)

            filt_val = filter_map.get(fname, "")
            if not filt_val:
                filt_val = header_filter_map.get(fname, "")

            am = header_airmass_map.get(fname, np.nan)

            # 2) 정보가 부족한 경우에만 FITS 헤더 읽기 (lazy load)
            need_fits = (not date_obs) or (not filt_val) or (not np.isfinite(am))
            if need_fits:
                hdr = self._get_header(result_dir, fname, self._header_cache)
                if hdr is not None:
                    if not date_obs:
                        date_obs = hdr.get("DATE-OBS")
                        jd = _parse_jd(date_obs)
                    if not filt_val:
                        filt_val = hdr.get("FILTER", hdr.get("FILTER1", hdr.get("FILTER2", "")))
                    if not np.isfinite(am):
                        am = self._compute_airmass(hdr)

            times.append(jd)
            dates.append(_date_from_dateobs(date_obs) if date_obs else _extract_date_from_path(result_dir, fname))
            night_ids.append(night_id_map.get(fname, 0))
            filt_key = _normalize_filter_key(filt_val)
            filters.append(filt_key)
            airmasses.append(am if np.isfinite(am) else np.nan)

            df = self._get_photometry_df(result_dir, fname)
            if df is None or df.empty:
                mags.append(np.nan)
                mag_errs.append(np.nan)
                comp_avgs.append(np.nan)
                comp_errs.append(np.nan)
                diffs.append(np.nan)
                diff_errs.append(np.nan)
                continue

            use_source_id = False
            target_source_id = None
            comp_source_map: dict[int, int] = {}
            if filt_key in filter_selections:
                sel = filter_selections[filt_key]
                override_target_sid = None
                if target_source_id_by_filter is not None:
                    override_target_sid = target_source_id_by_filter.get(filt_key)
                if override_target_sid is not None:
                    target_source_id = int(override_target_sid)
                else:
                    sel_target_id = sel.get("target_id")
                    if sel_target_id is not None and int(sel_target_id) == int(target_id):
                        target_source_id = sel.get("target_source_id")
                    else:
                        target_source_id = self._map_comp_source_id(sel, target_id)
                for cid in comp_ids:
                    sid = self._map_comp_source_id(sel, cid)
                    if sid is not None:
                        comp_source_map[int(cid)] = int(sid)
                use_source_id = True

            # Target
            row_t = pd.DataFrame()
            if use_source_id and target_source_id is not None and "source_id" in df.columns:
                row_t = _select_rows_by_source_id(df, int(target_source_id))
            if row_t.empty and "ID" in df.columns:
                row_t = df[df["ID"] == int(target_id)]

            if not row_t.empty:
                n_target_found += 1
                tmag = _safe_float(row_t["mag"].values[0])
                terr = _safe_float(row_t["mag_err"].values[0]) if "mag_err" in row_t.columns else np.nan
            else:
                tmag = np.nan
                terr = np.nan

            # Comparison ensemble
            cmags = []
            cerrs = []
            for cid in comp_ids:
                row_c = pd.DataFrame()
                if use_source_id and cid in comp_source_map and "source_id" in df.columns:
                    row_c = _select_rows_by_source_id(df, int(comp_source_map[cid]))
                if row_c.empty and "ID" in df.columns:
                    row_c = df[df["ID"] == int(cid)]
                if not row_c.empty and np.isfinite(_safe_float(row_c["mag"].values[0])):
                    cmags.append(_safe_float(row_c["mag"].values[0]))
                    cerrs.append(_safe_float(row_c["mag_err"].values[0]) if "mag_err" in row_c.columns else np.nan)
            if cmags:
                n_comp_found += 1
                cmags_arr = np.array(cmags, dtype=float)
                cerrs_arr = np.array(cerrs, dtype=float)
                valid_w = np.isfinite(cerrs_arr) & (cerrs_arr > 0)
                if np.any(valid_w):
                    w = 1.0 / (cerrs_arr[valid_w] ** 2)
                    comp_mean = float(np.sum(cmags_arr[valid_w] * w) / np.sum(w))
                    comp_err = float(1.0 / np.sqrt(np.sum(w)))
                else:
                    comp_mean = float(np.nanmean(cmags_arr))
                    comp_err = float(np.nanmean(cerrs_arr)) if cerrs else np.nan
            else:
                comp_mean = np.nan
                comp_err = np.nan

            mags.append(tmag)
            mag_errs.append(terr)
            comp_avgs.append(comp_mean)
            comp_errs.append(comp_err)

            if np.isfinite(tmag) and np.isfinite(comp_mean):
                diff = tmag - comp_mean
                diffs.append(diff)
                if np.isfinite(terr) and np.isfinite(comp_err):
                    diff_errs.append(float(np.sqrt(terr * terr + comp_err * comp_err)))
                else:
                    diff_errs.append(terr if np.isfinite(terr) else np.nan)
            else:
                diffs.append(np.nan)
                diff_errs.append(np.nan)

        tarr = np.array(times, float)
        if np.all(~np.isfinite(tarr)):
            tarr = np.arange(len(files), dtype=float)
        t0 = np.nanmedian(tarr)
        rel_time_hr = (tarr - t0) * 24.0

        # BJD_TDB 계산
        bjd_arr = self._compute_bjd_array(tarr, result_dir, target_id)

        if verbose:
            total = len(files)
            self.log(f"[DEBUG] Ensemble series (Target={target_id}) frames={total}")
            self.log(f"[DEBUG] Target found: {n_target_found}/{total}")
            self.log(f"[DEBUG] Comp ensemble available: {n_comp_found}/{total}")
            if np.any(np.isfinite(bjd_arr)):
                delta = np.nanmedian(bjd_arr - tarr) * 86400
                self.log(f"[BJD] BJD_TDB computed, median correction {delta:+.1f}s")
            else:
                self.log("[BJD] BJD_TDB not computed (missing site coords or target RA/Dec)")

        return pd.DataFrame({
            "file": files,
            "filter": filters,
            "date": dates,
            "night_id": night_ids,
            "JD": tarr,
            "BJD_TDB": bjd_arr,
            "rel_time_hr": rel_time_hr,
            "mag": np.array(mags, float),
            "mag_err": np.array(mag_errs, float),
            "comp_avg": np.array(comp_avgs, float),
            "comp_err": np.array(comp_errs, float),
            "diff_mag_raw": np.array(diffs, float),
            "diff_err": np.array(diff_errs, float),
            "airmass": np.array(airmasses, float),
        })

    def _build_check_star_series(
        self,
        result_dir: Path,
        comp_ids: list[int],
        verbose: bool = False,
    ) -> tuple[dict[str, int], pd.DataFrame]:
        # 캐시 확인
        _ck_key = (str(result_dir), tuple(sorted(comp_ids)))
        if _ck_key in self._check_series_cache:
            return self._check_series_cache[_ck_key]

        meta_by_filter = _load_check_star_meta_by_filter(result_dir)
        check_ids_by_filter = {
            flt: int(meta["check_id"])
            for flt, meta in meta_by_filter.items()
            if meta.get("check_id") is not None
        }
        check_source_ids_by_filter = {
            flt: int(meta["check_source_id"])
            for flt, meta in meta_by_filter.items()
            if meta.get("check_source_id") is not None
        }
        if not check_ids_by_filter and not check_source_ids_by_filter:
            return {}, pd.DataFrame()

        fallback_id = next(iter(check_ids_by_filter.values()), -1)
        check_df = self._build_ensemble_series(
            result_dir,
            int(fallback_id),
            comp_ids,
            verbose=verbose,
            target_source_id_by_filter=check_source_ids_by_filter or None,
        )
        if check_df.empty:
            return check_ids_by_filter, check_df

        valid_filters = set(check_ids_by_filter) | set(check_source_ids_by_filter)
        if "filter" in check_df.columns and valid_filters:
            filter_keys = check_df["filter"].astype(str).map(_normalize_filter_key)
            check_df = check_df[filter_keys.isin(valid_filters)].copy()
            if not check_df.empty:
                check_df["check_id"] = filter_keys.loc[check_df.index].map(check_ids_by_filter)
        if "check_id" in check_df.columns:
            check_ids = pd.to_numeric(check_df["check_id"], errors="coerce")
            check_df = check_df[check_ids.notna()].copy()
            if not check_df.empty:
                check_df["check_id"] = pd.to_numeric(check_df["check_id"], errors="coerce").astype(int)
        if "JD" in check_df.columns and not check_df.empty:
            check_df = check_df.sort_values("JD").reset_index(drop=True)
        result = (check_ids_by_filter, check_df)
        self._check_series_cache[_ck_key] = result
        return result

    def _build_diff_series(self, result_dir: Path, target_id: int, comp_id: int, verbose: bool = True) -> pd.DataFrame:
        # 캐시 키 생성
        cache_key = (str(result_dir), int(target_id), int(comp_id))
        if cache_key in self._diff_series_cache:
            if verbose:
                self.log(f"[CACHE] Using cached diff series for target={target_id}, comp={comp_id}")
            return self._diff_series_cache[cache_key].copy()

        idx_path = step5_photometry_dir(result_dir) / "photometry_index.csv"
        if not idx_path.exists():
            if verbose:
                self.log(f"[DEBUG] photometry_index.csv not found in {result_dir}")
            return pd.DataFrame()
        idx = pd.read_csv(idx_path)
        if "file" not in idx.columns:
            if verbose:
                self.log(f"[DEBUG] photometry_index.csv missing 'file' column")
            return pd.DataFrame()
        # Apply frame exclusions to the series computation
        exclude_map = self._get_frame_exclude_map(result_dir)
        if exclude_map:
            excluded_set = set(exclude_map.keys())
            idx = idx[~idx["file"].astype(str).isin(excluded_set)].reset_index(drop=True)
        files = idx["file"].astype(str).tolist()
        self._preload_photometry_cache(result_dir, files)
        headers_map = _load_headers_map(result_dir)
        headers_df = _load_headers_table(result_dir)

        # 필터별 selection 로드 (source_id 사용)
        filter_selections = _load_selection_ids_by_filter(result_dir)
        legacy_target_id, legacy_comp_ids = _load_selection_ids(result_dir)

        if verbose and filter_selections:
            self.log(f"[DEBUG] Filter-specific selections loaded: {list(filter_selections.keys())}")

        filter_map = {}
        if "filter" in idx.columns:
            filter_map = dict(zip(idx["file"].astype(str), idx["filter"].astype(str)))
        elif "FILTER" in idx.columns:
            filter_map = dict(zip(idx["file"].astype(str), idx["FILTER"].astype(str)))

        header_filter_map = {}
        if not headers_df.empty and "Filename" in headers_df.columns:
            for col in ("FILTER", "filter"):
                if col in headers_df.columns:
                    header_filter_map = dict(zip(headers_df["Filename"].astype(str), headers_df[col].astype(str)))
                    break

        # Night assignment map for this series
        diff_night_id_map: dict[str, int] = {}
        raw_na = _load_night_assignments(result_dir)
        diff_night_id_map.update({Path(str(k)).name: int(v) for k, v in raw_na.items() if int(v) > 0})
        fm_d = getattr(self, "file_manager", None)
        if fm_d is not None:
            for k, v in getattr(fm_d, "night_assignments", {}).items():
                if int(v) > 0:
                    diff_night_id_map[Path(str(k)).name] = int(v)
        if "night_id" in idx.columns:
            for fn, nid in zip(idx["file"].astype(str), pd.to_numeric(idx["night_id"], errors="coerce")):
                bn = Path(fn).name
                if not pd.isna(nid) and int(nid) > 0 and bn not in diff_night_id_map:
                    diff_night_id_map[bn] = int(nid)

        times = []
        diffs = []
        filters = []
        airmasses = []
        diff_night_ids = []

        # headers.csv에서 airmass 컬럼 확인 (FITS 안 읽기 위해)
        header_airmass_map = {}
        if not headers_df.empty and "Filename" in headers_df.columns:
            for col in ("AIRMASS", "airmass", "AM"):
                if col in headers_df.columns:
                    am_series = pd.to_numeric(headers_df[col], errors="coerce")
                    fnames_series = headers_df["Filename"].astype(str)
                    header_airmass_map = {
                        fn: float(am)
                        for fn, am in zip(fnames_series, am_series)
                        if np.isfinite(am)
                    }
                    break

        # 디버깅 통계
        n_target_found = 0
        n_comp_found = 0
        n_both_found = 0
        n_phot_missing = 0
        n_fits_read = 0
        missing_target_frames = []
        missing_comp_frames = []

        for fname in files:
            # 1) DATE-OBS: headers.csv에서 먼저 시도
            date_obs = headers_map.get(fname) if headers_map else None
            jd = _parse_jd(date_obs)

            # 2) FILTER: filter_map 또는 header_filter_map에서 시도
            filt_val = filter_map.get(fname, "")
            if not filt_val:
                filt_val = header_filter_map.get(fname, "")

            # 3) AIRMASS: header_airmass_map에서 시도
            am = header_airmass_map.get(fname, np.nan)

            # 4) 정보가 부족한 경우에만 FITS 헤더 읽기 (lazy load)
            need_fits = (not np.isfinite(jd)) or (not filt_val) or (not np.isfinite(am))
            if need_fits:
                hdr = self._get_header(result_dir, fname, self._header_cache)
                if hdr is not None:
                    n_fits_read += 1
                    if not np.isfinite(jd):
                        jd = _parse_jd(hdr.get("DATE-OBS"))
                    if not filt_val:
                        filt_val = hdr.get("FILTER", hdr.get("FILTER1", hdr.get("FILTER2", "")))
                    if not np.isfinite(am):
                        am = self._compute_airmass(hdr)

            times.append(jd)
            filt_key = _normalize_filter_key(filt_val)
            filters.append(filt_key)
            airmasses.append(float(am) if np.isfinite(am) else np.nan)
            diff_night_ids.append(diff_night_id_map.get(fname, 0))

            df = self._get_photometry_df(result_dir, fname)
            if df is None or df.empty:
                diffs.append(np.nan)
                n_phot_missing += 1
                continue

            # 필터별 selection 또는 legacy selection 사용
            use_source_id = False
            target_source_id = None
            comp_source_id = None

            if filt_key in filter_selections:
                sel = filter_selections[filt_key]
                target_source_id = sel.get("target_source_id")
                comp_source_id = self._map_comp_source_id(sel, comp_id)
                use_source_id = True

            # 타겟 매칭 (source_id 우선, 없으면 ID)
            row_t = pd.DataFrame()
            if use_source_id and target_source_id is not None and "source_id" in df.columns:
                row_t = _select_rows_by_source_id(df, int(target_source_id))
            if row_t.empty and "ID" in df.columns:
                row_t = df[df["ID"] == int(target_id)]

            # 비교성 매칭 (source_id 우선, 없으면 ID)
            row_c = pd.DataFrame()
            if use_source_id and comp_source_id is not None and "source_id" in df.columns:
                row_c = _select_rows_by_source_id(df, int(comp_source_id))
            if row_c.empty and "ID" in df.columns:
                row_c = df[df["ID"] == int(comp_id)]

            target_found = not row_t.empty
            comp_found = not row_c.empty

            if target_found:
                n_target_found += 1
            else:
                missing_target_frames.append(fname)
            if comp_found:
                n_comp_found += 1
            else:
                missing_comp_frames.append(fname)

            if target_found and comp_found:
                n_both_found += 1
                tmag = float(row_t["mag"].values[0])
                cmag = float(row_c["mag"].values[0])
                diffs.append(tmag - cmag)
            else:
                diffs.append(np.nan)

        # 디버깅 로그 출력
        if verbose:
            total = len(files)
            self.log(f"[DEBUG] === Diff Series Build (Target={target_id}, Comp={comp_id}) ===")
            self.log(f"[DEBUG] Total frames: {total}, FITS headers read: {n_fits_read}")
            self.log(f"[DEBUG] Target ID={target_id} found in: {n_target_found}/{total} frames ({100*n_target_found/max(total,1):.1f}%)")
            self.log(f"[DEBUG] Comp ID={comp_id} found in: {n_comp_found}/{total} frames ({100*n_comp_found/max(total,1):.1f}%)")
            self.log(f"[DEBUG] Both found (valid points): {n_both_found}/{total} frames")
            if n_phot_missing > 0:
                self.log(f"[WARN] Photometry TSV missing for {n_phot_missing} frames")
            if missing_target_frames and len(missing_target_frames) <= 10:
                self.log(f"[WARN] Target missing in: {', '.join(missing_target_frames[:10])}")
            elif missing_target_frames:
                self.log(f"[WARN] Target missing in {len(missing_target_frames)} frames (first 5: {', '.join(missing_target_frames[:5])})")
            if missing_comp_frames and len(missing_comp_frames) <= 10:
                self.log(f"[WARN] Comp missing in: {', '.join(missing_comp_frames[:10])}")
            elif missing_comp_frames:
                self.log(f"[WARN] Comp missing in {len(missing_comp_frames)} frames (first 5: {', '.join(missing_comp_frames[:5])})")

        tarr = np.array(times, float)
        if np.all(~np.isfinite(tarr)):
            tarr = np.arange(len(files), dtype=float)
        t0 = np.nanmedian(tarr)
        rel_time_hr = (tarr - t0) * 24.0

        bjd_arr = self._compute_bjd_array(tarr, result_dir, target_id)

        result_df = pd.DataFrame({
            "file": files,
            "filter": filters,
            "night_id": diff_night_ids,
            "JD": tarr,
            "BJD_TDB": bjd_arr,
            "rel_time_hr": rel_time_hr,
            "diff_mag": np.array(diffs, float),
            "airmass": np.array(airmasses, float),
        })

        # 캐시에 저장
        self._diff_series_cache[cache_key] = result_df.copy()
        if verbose:
            self.log(f"[CACHE] Stored diff series for target={target_id}, comp={comp_id} ({len(result_df)} rows)")

        return result_df

    def _current_result_dir(self) -> Path | None:
        if not self.datasets:
            return None
        return Path(self.datasets[0][1])

    def _compute_bjd_array(self, tarr: np.ndarray, result_dir: Path, target_id: int) -> np.ndarray:
        """Compute BJD_TDB array; returns NaN array if site coords or target RA/Dec are missing."""
        site_lat = float(getattr(self.params.P, "site_lat_deg", np.nan))
        site_lon = float(getattr(self.params.P, "site_lon_deg", np.nan))
        site_alt = float(getattr(self.params.P, "site_alt_m", 0.0))
        tgt_ra, tgt_dec = _load_target_radec(result_dir, target_id)
        if not np.isfinite(tgt_ra) and hasattr(self.params.P, "target"):
            tgt_ra = float(getattr(self.params.P.target, "ra_deg", np.nan) or np.nan)
            tgt_dec = float(getattr(self.params.P.target, "dec_deg", np.nan) or np.nan)
        if np.isfinite(site_lat) and np.isfinite(site_lon) and np.isfinite(tgt_ra) and np.isfinite(tgt_dec):
            return compute_bjd_tdb_array(tarr, tgt_ra, tgt_dec, site_lat, site_lon, site_alt)
        return np.full(len(tarr), np.nan)

    def _frame_qc_marker_path(self, result_dir: Path) -> Path:
        return step10_dir(result_dir) / "frame_qc_done.json"

    def _load_frame_qc_done(self, result_dir: Path) -> bool:
        key = str(result_dir)
        if key in self._frame_qc_done_cache:
            return self._frame_qc_done_cache[key]
        marker = self._frame_qc_marker_path(result_dir)
        exclude_path = step10_dir(result_dir) / "frame_exclude.csv"
        done = marker.exists() or exclude_path.exists()
        self._frame_qc_done_cache[key] = done
        return done

    def _mark_frame_qc_done(self, result_dir: Path) -> None:
        path = self._frame_qc_marker_path(result_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        exclude_map = self._get_frame_exclude_map(result_dir)
        payload = {
            "done": True,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "excluded": int(len(exclude_map)),
        }
        try:
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            pass
        self._frame_qc_done_cache[str(result_dir)] = True

    def _is_frame_qc_ready(self, result_dir: Path | None) -> bool:
        if result_dir is None:
            return True
        done = self._load_frame_qc_done(result_dir)
        dirty = str(result_dir) in self._frame_exclude_dirty
        return done and not dirty

    def _update_qc_gate_ui(self) -> None:
        if hasattr(self, "btn_plot"):
            self.btn_plot.setEnabled(True)
        # Enable save/reset buttons only when there are unsaved frame exclusion changes
        result_dir = self._current_result_dir()
        dirty = result_dir is not None and str(result_dir) in self._frame_exclude_dirty
        has_excludes = False
        if result_dir is not None:
            has_excludes = bool(self._get_frame_exclude_map(result_dir))
        for attr in ("btn_frame_qc_save", "btn_qc_frame_save"):
            btn = getattr(self, attr, None)
            if btn is not None:
                btn.setEnabled(dirty)
        for attr in ("btn_frame_qc_clear", "btn_qc_frame_reset"):
            btn = getattr(self, attr, None)
            if btn is not None:
                btn.setEnabled(dirty or has_excludes)

    def _get_frame_exclude_map(self, result_dir: Path) -> dict[str, set[str]]:
        key = str(result_dir)
        if key not in self._frame_exclude_cache:
            self._frame_exclude_cache[key] = load_frame_excludes(result_dir)
        return self._frame_exclude_cache[key]

    def _set_selected_frame(self, fname: str | None, result_dir: Path | None) -> None:
        self._frame_qc_selected = fname
        self._frame_qc_selected_dir = result_dir
        if not fname or result_dir is None:
            self.frame_qc_selected_label.setText("Selected: (none)")
            if hasattr(self, "qc_frame_selected_label"):
                self.qc_frame_selected_label.setText("Selected: (none)")
            return
        exclude_map = self._get_frame_exclude_map(result_dir)
        status = "excluded" if fname in exclude_map else "included"
        text = f"Selected: {fname} ({status})"
        self.frame_qc_selected_label.setText(text)
        if hasattr(self, "qc_frame_selected_label"):
            self.qc_frame_selected_label.setText(text)

    def _draw_qc_selection_indicator(self, xy: tuple[float, float]) -> None:
        """Draw red hollow circle on QC preview plot at given data coords."""
        prev = getattr(self, "_qc_selection_indicator_artist", None)
        if prev is not None:
            try:
                prev.remove()
            except Exception:
                pass
        sc = self.check_plot_ax.scatter([xy[0]], [xy[1]], s=220, facecolors="none",
                                        edgecolors="red", linewidths=2.0, zorder=20)
        self._qc_selection_indicator_artist = sc
        self.check_plot_canvas.draw_idle()

    def _draw_selection_indicator(self) -> None:
        """Draw red hollow circle around the selected point on the light curve plot."""
        xy = getattr(self, "_selected_point_xy", None)
        if xy is None:
            return
        ax = self.plot_ax
        # Remove previous indicator if any
        prev = getattr(self, "_selection_indicator_artist", None)
        if prev is not None:
            try:
                prev.remove()
            except Exception:
                pass
        sc = ax.scatter([xy[0]], [xy[1]], s=220, facecolors="none",
                        edgecolors="red", linewidths=2.0, zorder=20)
        self._selection_indicator_artist = sc
        self.plot_canvas.draw_idle()

    def _refresh_qc_preview(self) -> None:
        comp_id = getattr(self, "_qc_last_selected_comp_id", None)
        if comp_id is not None:
            self._plot_comp_preview(int(comp_id))

    def _refresh_lc_plot(self) -> None:
        """Replot light curve only if it has been built at least once."""
        if getattr(self, "_lc_ever_plotted", False):
            self.plot_current_comparison()

    def _update_frame_qc_summary(self) -> None:
        result_dir = self._current_result_dir()
        total = int(self._frame_qc_total or 0)
        if result_dir is None:
            self.frame_qc_summary_label.setText("Excluded: 0/0")
            return
        exclude_map = self._get_frame_exclude_map(result_dir)
        exc = len(exclude_map)
        dirty = str(result_dir) in self._frame_exclude_dirty
        suffix = " (unsaved)" if dirty else ""
        text = f"Excluded: {exc}/{total}{suffix}"
        self.frame_qc_summary_label.setText(text)
        if hasattr(self, "qc_frame_summary_label"):
            self.qc_frame_summary_label.setText(text)

    def _on_plot_click(self, event) -> None:
        self.plot_canvas.setFocus()
        if not event.dblclick:
            return
        if event.inaxes != self.plot_ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        TOLERANCE_PX = 20
        ax = self.plot_ax
        click_disp = ax.transData.transform([[event.xdata, event.ydata]])[0]
        best_dist = float("inf")
        best_fname = None
        best_xy = None
        for artist, meta in self._plot_point_map.items():
            offsets = artist.get_offsets()
            files = meta.get("files", [])
            if len(offsets) == 0 or len(files) == 0:
                continue
            try:
                disp = ax.transData.transform(np.asarray(offsets))
            except Exception:
                continue
            dists = np.linalg.norm(disp - click_disp, axis=1)
            idx = int(np.argmin(dists))
            if dists[idx] < best_dist and idx < len(files):
                best_dist = dists[idx]
                best_fname = str(files[idx])
                arr = np.asarray(offsets)
                best_xy = (float(arr[idx, 0]), float(arr[idx, 1]))
        if best_fname and best_dist <= TOLERANCE_PX:
            result_dir = self._current_result_dir()
            self._selected_point_xy = best_xy
            self._set_selected_frame(best_fname, result_dir)
            self._draw_selection_indicator()

    def _exclude_selected_frame(self) -> None:
        if not self._frame_qc_selected or self._frame_qc_selected_dir is None:
            return
        exclude_map = self._get_frame_exclude_map(self._frame_qc_selected_dir)
        if self._frame_qc_selected in exclude_map:
            return  # already excluded — nothing to do
        exclude_map.setdefault(self._frame_qc_selected, set()).add("manual")
        self._frame_exclude_dirty.add(str(self._frame_qc_selected_dir))
        self._set_selected_frame(self._frame_qc_selected, self._frame_qc_selected_dir)
        self._update_frame_qc_summary()
        self._refresh_qc_stats_panel()
        self._refresh_qc_preview()
        self._refresh_lc_plot()
        self._update_qc_gate_ui()

    def _exclude_selected_frame_all_filters(self) -> None:
        if not self._frame_qc_selected or self._frame_qc_selected_dir is None:
            return
        result_dir = self._frame_qc_selected_dir
        files_all = self._qc_scope_files_all_filters or []
        if not files_all:
            self._exclude_selected_frame()
            return
        key = self._frame_group_key(self._frame_qc_selected)
        if not key:
            self._exclude_selected_frame()
            return
        targets = [f for f in files_all if self._frame_group_key(f) == key]
        if not targets or len(targets) == len(files_all):
            targets = [self._frame_qc_selected]
        exclude_map = self._get_frame_exclude_map(result_dir)
        for fname in targets:
            exclude_map.setdefault(fname, set()).add("manual_all_filters")
        self._frame_exclude_dirty.add(str(result_dir))
        self._set_selected_frame(self._frame_qc_selected, result_dir)
        self._update_frame_qc_summary()
        self._refresh_qc_stats_panel()
        self._refresh_qc_preview()
        self._refresh_lc_plot()
        self._update_qc_gate_ui()

    def _include_selected_frame(self) -> None:
        if not self._frame_qc_selected or self._frame_qc_selected_dir is None:
            return
        exclude_map = self._get_frame_exclude_map(self._frame_qc_selected_dir)
        if self._frame_qc_selected not in exclude_map:
            return  # already included — nothing to do
        del exclude_map[self._frame_qc_selected]
        self._frame_exclude_dirty.add(str(self._frame_qc_selected_dir))
        self._set_selected_frame(self._frame_qc_selected, self._frame_qc_selected_dir)
        self._update_frame_qc_summary()
        self._refresh_qc_stats_panel()
        self._refresh_qc_preview()
        self._refresh_lc_plot()
        self._update_qc_gate_ui()

    def clear_frame_excludes(self) -> None:
        result_dir = self._current_result_dir()
        if result_dir is None:
            return
        rkey = str(result_dir)
        had_excludes = bool(self._get_frame_exclude_map(result_dir))
        self._frame_exclude_cache[rkey] = {}
        # If there were any excludes (in-memory or on disk), mark dirty so Save can persist the empty state
        if had_excludes or rkey in self._frame_exclude_dirty:
            self._frame_exclude_dirty.add(rkey)
        self._frame_qc_selected = None
        self._selected_point_xy = None
        self._update_frame_qc_summary()
        self._refresh_qc_stats_panel()
        self._refresh_qc_preview()
        self._refresh_lc_plot()
        self._update_qc_gate_ui()
        self.log("[Frame QC] Reset all exclusions.")

    def save_frame_excludes(self) -> None:
        result_dir = self._current_result_dir()
        if result_dir is None:
            return
        exclude_map = self._get_frame_exclude_map(result_dir)
        try:
            saved_path = save_frame_excludes_file(result_dir, exclude_map)
            if str(result_dir) in self._frame_exclude_dirty:
                self._frame_exclude_dirty.remove(str(result_dir))
            self._mark_frame_qc_done(result_dir)
            self._update_frame_qc_summary()
            self._update_qc_gate_ui()
            # Invalidate series caches so next plot/build uses updated exclusions
            rdir_str = str(result_dir)
            self._diff_series_cache = {k: v for k, v in self._diff_series_cache.items() if k[0] != rdir_str}
            self._check_series_cache = {k: v for k, v in self._check_series_cache.items() if k[0] != rdir_str}
            self.log(f"[Frame QC] Saved {len(exclude_map)} exclusion(s).")
        except Exception as e:
            self.log(f"[Frame QC] Save failed: {e}")
            QMessageBox.warning(self, "Frame QC", f"제외 프레임 저장 실패:\n{e}")

    def plot_current_comparison(self):
        if not self.datasets:
            return
        target_id_text = self.target_edit.text().strip()
        if not target_id_text:
            target_id, comp_ids = _load_selection_ids(self.datasets[0][1])
            if target_id is None:
                self.log("Target ID missing for plot.")
                return
            self.target_edit.setText(str(target_id))
            if comp_ids:
                self.comp_edit.setText(",".join(str(i) for i in comp_ids))
        self._update_comp_ids_from_input()
        if not self.comp_ids_list:
            self.log("Comparison IDs missing for plot.")
            return
        comp_id = self.comp_ids_list[self.comp_index]
        target_id = int(self.target_edit.text().strip())
        result_dir = Path(self.datasets[0][1])
        df = self._build_diff_series(result_dir, target_id, comp_id)
        if df.empty:
            self.log("No light curve data to plot.")
            return
        y_col = "diff_mag_raw" if "diff_mag_raw" in df.columns else "diff_mag"
        self._ensure_filter_colors(df["filter"].astype(str).tolist())
        self._lc_ever_plotted = True
        self.plot_ax.clear()
        self._plot_point_map = {}
        plotted_y = []

        # X축 선택
        if self.x_axis_mode == "phase":
            # 위상 계산
            if "BJD_TDB" in df.columns and df["BJD_TDB"].notna().any():
                jd_col = "BJD_TDB"
            elif "JD" in df.columns:
                jd_col = "JD"
            elif "jd" in df.columns:
                jd_col = "jd"
            else:
                jd_col = None
            if jd_col is None:
                self.log("[WARN] JD column missing, cannot compute phase")
                x_column = "rel_time_hr"
                x_label = "Time (hr)"
            elif self.phase_period <= 0:
                self.log("[WARN] Period not set. Use Parameters to set period.")
                x_column = "rel_time_hr"
                x_label = "Time (hr)"
            else:
                jd = df[jd_col].to_numpy(float)
                t0 = self.phase_t0 if self.phase_t0 > 0 else np.nanmin(jd)
                phase = ((jd - t0) / self.phase_period) % 1.0
                df = df.copy()
                df["phase"] = phase
                df = self._expand_phase_cycles(df)
                x_column = "phase"
                x_label = "Phase" if self.phase_cycles <= 1 else f"Phase (0-{self.phase_cycles:g})"
        else:
            x_column = "rel_time_hr"
            x_label = "Time (hr)"

        result_dir = self._current_result_dir()
        exclude_map = self._get_frame_exclude_map(result_dir) if result_dir is not None else {}
        excluded_files = set(exclude_map.keys())
        self._frame_qc_total = int(df["file"].nunique()) if "file" in df.columns else 0
        self._update_frame_qc_summary()

        y_label = "dmag (raw)"
        for filt, sub in df.groupby("filter"):
            fkey = self._filter_key_for_ui(filt)
            label = fkey
            c = self._get_filter_color(fkey)

            x = sub[x_column].to_numpy(float)
            y = sub[y_col].to_numpy(float)
            files = sub["file"].astype(str).to_numpy() if "file" in sub.columns else np.array([""] * len(sub))

            m = np.isfinite(x) & np.isfinite(y)
            if not np.any(m):
                continue
            excl = np.array([f in excluded_files for f in files], dtype=bool)
            m_in = m & ~excl
            m_ex = m & excl

            if np.any(m_in):
                sc = self.plot_ax.scatter(
                    x[m_in], y[m_in], s=12, color=c, label=label, alpha=0.7, picker=5
                )
                self._plot_point_map[sc] = {"files": files[m_in].tolist()}
                plotted_y.extend(y[m_in].tolist())
            if np.any(m_ex):
                scx = self.plot_ax.scatter(
                    x[m_ex], y[m_ex], s=16, color="#9E9E9E", marker="x", alpha=0.9, picker=5
                )
                self._plot_point_map[scx] = {"files": files[m_ex].tolist()}
                plotted_y.extend(y[m_ex].tolist())

        # Check star overlay
        try:
            check_ids_by_filter, check_df = self._build_check_star_series(
                result_dir,
                self.comp_ids_list,
                verbose=False,
            )
            if not check_df.empty:
                if self.x_axis_mode == "phase" and "phase" not in check_df.columns:
                    if "JD" in check_df.columns and self.phase_period > 0:
                        jd_c = check_df["JD"].to_numpy(float)
                        t0_c = self.phase_t0 if self.phase_t0 > 0 else np.nanmin(jd_c)
                        phase_c = ((jd_c - t0_c) / self.phase_period) % 1.0
                        check_df = check_df.copy()
                        check_df["phase"] = phase_c
                        check_df = self._expand_phase_cycles(check_df)
                ck_x_col = x_column if x_column in check_df.columns else "rel_time_hr"
                ck_y_col = "diff_mag_raw" if "diff_mag_raw" in check_df.columns else "diff_mag"
                if ck_x_col in check_df.columns and ck_y_col in check_df.columns:
                    cx = check_df[ck_x_col].to_numpy(float)
                    cy = pd.to_numeric(check_df[ck_y_col], errors="coerce").to_numpy(float)
                    m_ck = np.isfinite(cx) & np.isfinite(cy)
                    if np.any(m_ck):
                        unique_check_ids = sorted(set(check_ids_by_filter.values()))
                        check_label = f"Check (ID {unique_check_ids[0]})" if len(unique_check_ids) == 1 else "Check"
                        self.plot_ax.scatter(
                            cx[m_ck], cy[m_ck], s=6, color="#FFD700", alpha=0.5,
                            zorder=2, label=check_label, marker="."
                        )
        except Exception:
            pass

        self.plot_ax.set_xlabel(x_label)
        self.plot_ax.set_ylabel(f"{y_label} (Target - Comparison)")
        if self.x_axis_mode == "phase" and self.phase_cycles > 1:
            self.plot_ax.set_xlim(0, float(self.phase_cycles))
        self.plot_ax.grid(True, alpha=0.3)

        # Auto-scale y-axis: median-centered with robust range
        all_y = np.array([v for v in plotted_y if np.isfinite(v)])
        if len(all_y) >= 2:
            med = float(np.nanmedian(all_y))
            mad = float(np.nanmedian(np.abs(all_y - med)))
            half = max(mad * 5.0, 0.05)
            self.plot_ax.set_ylim(med + half, med - half)  # inverted y
        else:
            self.plot_ax.invert_yaxis()

        handles, labels = self.plot_ax.get_legend_handles_labels()
        if handles:
            self.plot_ax.legend(loc="best", fontsize=8)

        # Restore selection indicator if a point is still selected
        self._selection_indicator_artist = None
        if getattr(self, "_selected_point_xy", None) is not None:
            self._draw_selection_indicator()
        else:
            self.plot_canvas.draw_idle()

    def _compute_comp_qc(
        self,
        result_dir: Path,
        target_id: int,
        comp_ids: list[int],
        files_override: list[str] | None = None,
        verbose: bool = True,
    ) -> list[dict]:
        """Check-star QC: each comp is treated as target against remaining comps as ensemble."""
        rows = []
        active_set = set(self.comp_ids_list)
        if not hasattr(self, "_qc_checkstar_cache"):
            self._qc_checkstar_cache: dict[int, pd.DataFrame] = {}

        for comp_id in comp_ids:
            other_comps = [c for c in comp_ids if c != comp_id]
            if not other_comps:
                # Only 1 comp — fall back to absolute mag scatter
                df = self._build_star_mag_series(result_dir, comp_id, verbose=False,
                                                  files_override=files_override)
                if df.empty:
                    rows.append({"comp_id": int(comp_id), "n": 0, "rms": np.nan, "mad": np.nan,
                                 "sigma_nights": np.nan, "outliers": 0, "outlier_frac": np.nan,
                                 "use": comp_id in active_set})
                    continue
                y = df["mag"].to_numpy(float)
                m = np.isfinite(y)
                yv = y[m]
                med = np.nanmedian(yv)
                yv = yv - med
                rms = float(np.nanstd(yv))
                mad = float(np.nanmedian(np.abs(yv - np.nanmedian(yv))))
                rows.append({"comp_id": int(comp_id), "n": int(m.sum()), "rms": rms, "mad": mad,
                             "sigma_nights": np.nan, "outliers": 0, "outlier_frac": np.nan,
                             "use": comp_id in active_set})
                continue

            # Check star: build diff series with this comp as "target"
            df = self._build_ensemble_series(result_dir, comp_id, other_comps, verbose=False)
            if files_override:
                df = df[df["file"].isin(set(files_override))].copy()
            self._qc_checkstar_cache[comp_id] = df  # cache for preview plot

            if df.empty:
                rows.append({"comp_id": int(comp_id), "n": 0, "rms": np.nan, "mad": np.nan,
                             "sigma_nights": np.nan, "outliers": 0, "outlier_frac": np.nan,
                             "use": comp_id in active_set})
                continue

            y = pd.to_numeric(df["diff_mag_raw"], errors="coerce").to_numpy(float)
            night_ids = df["night_id"].to_numpy(int) if "night_id" in df.columns else np.zeros(len(df), int)
            m = np.isfinite(y)
            n = int(m.sum())
            if n <= 1:
                rows.append({"comp_id": int(comp_id), "n": n, "rms": np.nan, "mad": np.nan,
                             "sigma_nights": np.nan, "outliers": 0, "outlier_frac": np.nan,
                             "use": comp_id in active_set})
                continue

            yv = y[m]
            rms = float(np.nanstd(yv))
            med = float(np.nanmedian(yv))
            mad = float(np.nanmedian(np.abs(yv - med)))

            # Per-night medians → σ(nights) = night-to-night stability
            night_medians = []
            for nid in sorted(set(night_ids[m])):
                if nid <= 0:
                    continue
                nm = m & (night_ids == nid)
                if nm.sum() > 0:
                    night_medians.append(float(np.nanmedian(y[nm])))
            sigma_nights = float(np.std(night_medians, ddof=1)) if len(night_medians) >= 2 else np.nan

            # Outliers (3σ from median)
            outlier_count = int(np.sum(np.abs(yv - med) > self.qc_sigma * mad)) if np.isfinite(mad) and mad > 0 else 0
            outlier_frac = outlier_count / max(n, 1)

            if verbose:
                sn_str = f"{sigma_nights:.4f}" if np.isfinite(sigma_nights) else "nan"
                self.log(f"[QC] Comp {comp_id}: RMS={rms:.4f} σ_nights={sn_str}")

            rows.append({
                "comp_id": int(comp_id),
                "n": n,
                "rms": rms,
                "mad": mad,
                "sigma_nights": sigma_nights,
                "night_medians": night_medians,
                "outliers": outlier_count,
                "outlier_frac": float(outlier_frac),
                "use": comp_id in active_set,
            })

        return rows

    def _save_comp_qc_summary(self, result_dir: Path, rows: list[dict]) -> None:
        if not rows:
            return
        out_dir = step10_dir(result_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        path = out_dir / "comp_qc_summary.csv"
        try:
            df.to_csv(path, index=False)
            self.log(f"[QC] Saved {path.name}")
        except Exception as e:
            self.log(f"[QC] Failed to save summary: {e}")

    def _qc_summary_meta_path(self, result_dir: Path) -> Path:
        return step10_dir(result_dir) / "comp_qc_summary.meta.json"

    def _build_qc_signature(
        self,
        result_dir: Path,
        target_id: int,
        comp_ids: list[int],
        files_override: list[str] | None = None,
    ) -> dict:
        idx_path = step5_photometry_dir(result_dir) / "photometry_index.csv"
        try:
            idx_mtime_ns = int(idx_path.stat().st_mtime_ns)
        except OSError:
            idx_mtime_ns = 0
        current_date = self.qc_date_combo.currentText() if hasattr(self, "qc_date_combo") else "All"
        return {
            "target_id": int(target_id),
            "comp_ids": sorted(int(c) for c in comp_ids),
            "date_selection": current_date or "All",
            "files_override": sorted(str(f) for f in files_override) if files_override is not None else None,
            "excluded_files": sorted(str(k) for k in self._get_frame_exclude_map(result_dir).keys()),
            "qc_sigma": float(self.qc_sigma),
            "photometry_index_mtime_ns": idx_mtime_ns,
        }

    def _load_cached_comp_qc_summary(self, result_dir: Path, signature: dict) -> list[dict] | None:
        path = step10_dir(result_dir) / "comp_qc_summary.csv"
        meta_path = self._qc_summary_meta_path(result_dir)
        if not path.exists() or not meta_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta != signature:
                return None
            df = pd.read_csv(path)
        except Exception:
            return None
        if df.empty or "comp_id" not in df.columns:
            return None
        rows = df.to_dict(orient="records")
        for row in rows:
            try:
                row["comp_id"] = int(row.get("comp_id"))
            except Exception:
                pass
            row["use"] = bool(row.get("use", False))
        return rows

    def _save_comp_qc_meta(self, result_dir: Path, signature: dict | None) -> None:
        if signature is None:
            return
        try:
            self._qc_summary_meta_path(result_dir).write_text(
                json.dumps(signature, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _populate_qc_table(self, rows: list[dict]) -> None:
        self._qc_table_block = True
        self.qc_table.blockSignals(True)
        self.qc_table.setUpdatesEnabled(False)
        self.qc_table.clearContents()
        self.qc_table.setRowCount(len(rows))
        for r, row in enumerate(rows):

            item_use = QTableWidgetItem()
            item_use.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable)
            item_use.setCheckState(Qt.Checked if row.get("use", False) else Qt.Unchecked)
            self.qc_table.setItem(r, 0, item_use)

            self.qc_table.setItem(r, 1, QTableWidgetItem(str(row.get("comp_id", ""))))
            self.qc_table.setItem(r, 2, QTableWidgetItem(str(row.get("n", ""))))
            self.qc_table.setItem(r, 3, QTableWidgetItem(_fmt_float(row.get("rms"))))
            self.qc_table.setItem(r, 4, QTableWidgetItem(_fmt_float(row.get("sigma_nights"))))
            self.qc_table.setItem(r, 5, QTableWidgetItem(_fmt_float(row.get("mad"))))
            self.qc_table.setItem(r, 6, QTableWidgetItem(_fmt_percent(row.get("outlier_frac"))))

        self.qc_table.setUpdatesEnabled(True)
        self.qc_table.blockSignals(False)
        self._qc_table_block = False
        self._update_comp_count_label()

    def _update_comp_count_label(self):
        if not hasattr(self, "lbl_comp_count"):
            return
        n_active = len(self.comp_ids_list)
        self.lbl_comp_count.setText(f"Active comps: {n_active}")

    def _update_qc_threshold_label(self) -> None:
        if not hasattr(self, "lbl_qc_thresholds"):
            return
        self.lbl_qc_thresholds.setText(
            "Auto Use 기준: "
            f"RMS<= {self.qc_rms_max:.4f} mag, "
            f"Outlier<= {self.qc_outlier_frac_max:.3f}, "
            f"Min N>= {self.qc_min_points}, "
            f"Sigma= {self.qc_sigma:.2f}"
        )

    def _get_qc_result_dir(self) -> Path:
        if self.datasets:
            return Path(self.datasets[0][1])
        return Path(self.params.P.result_dir)

    def _load_qc_night_id_map(self, result_dir: Path, idx: pd.DataFrame) -> dict[str, int]:
        """Return basename→night_id map. Merges all sources; result is cached per result_dir."""
        cache_key = str(result_dir)
        if not hasattr(self, "_night_id_map_cache"):
            self._night_id_map_cache: dict[str, dict[str, int]] = {}

        if cache_key not in self._night_id_map_cache:
            night_id_map: dict[str, int] = {}
            # Source 1: file_manager (in-memory, fastest)
            fm = getattr(self, "file_manager", None)
            if fm is not None:
                for k, v in getattr(fm, "night_assignments", {}).items():
                    if int(v) > 0:
                        night_id_map[Path(str(k)).name] = int(v)
            # Source 2: disk JSON (step1 ground truth)
            if not night_id_map:
                raw = _load_night_assignments(result_dir)
                night_id_map.update({Path(str(k)).name: int(v) for k, v in raw.items() if int(v) > 0})
            self._night_id_map_cache[cache_key] = night_id_map

        base_map = dict(self._night_id_map_cache[cache_key])
        # Source 3: CSV column fills gaps only (may be stale/partial)
        if "night_id" in idx.columns:
            for fname, night_id in zip(idx["file"].astype(str), pd.to_numeric(idx["night_id"], errors="coerce")):
                bn = Path(str(fname)).name
                if not pd.isna(night_id) and int(night_id) > 0 and bn not in base_map:
                    base_map[bn] = int(night_id)
        return base_map

    def _build_qc_date_label_map(
        self,
        result_dir: Path,
        idx: pd.DataFrame | None = None,
    ) -> tuple[list[str], dict[str, str]]:
        if idx is None:
            idx_path = step5_photometry_dir(result_dir) / "photometry_index.csv"
            if not idx_path.exists():
                self._qc_date_label_map = {}
                return [], {}
            try:
                idx = pd.read_csv(idx_path)
            except Exception:
                self._qc_date_label_map = {}
                return [], {}
        if "file" not in idx.columns:
            self._qc_date_label_map = {}
            return [], {}

        files = idx["file"].astype(str).tolist()
        if not files:
            self._qc_date_label_map = {}
            return [], {}

        headers_map = _load_headers_map(result_dir)
        tz = float(getattr(self.params.P, "site_tz_offset_hours", 0.0))
        base_dates: dict[str, str] = {}
        for fname in files:
            date_obs = headers_map.get(fname)
            if date_obs:
                base_dates[fname] = _display_date_from_dateobs(date_obs, tz)
            else:
                base_dates[fname] = _extract_date_from_path(result_dir, fname)

        night_id_map = self._load_qc_night_id_map(result_dir, idx)
        unique_nights = sorted({night_id_map.get(fname, 0) for fname in files if night_id_map.get(fname, 0) > 0})

        if len(unique_nights) <= 1:
            self._qc_date_label_map = dict(base_dates)
            return files, dict(base_dates)

        night_dates: dict[int, str] = {}
        for night_id in unique_nights:
            candidates = [
                base_dates.get(fname, "unknown")
                for fname in files
                if night_id_map.get(fname, 0) == night_id and base_dates.get(fname, "unknown") != "unknown"
            ]
            night_dates[night_id] = candidates[0] if candidates else "unknown"

        labels: dict[str, str] = {}
        for fname in files:
            night_id = night_id_map.get(fname, 0)
            if night_id > 0:
                labels[fname] = f"N{night_id}"
            else:
                labels[fname] = base_dates.get(fname, "unknown")

        self._qc_date_label_map = dict(labels)
        return files, labels

    def _refresh_qc_filter_combo(self, filters: list[str]) -> None:
        if not hasattr(self, "qc_filter_combo"):
            return
        current = self.qc_filter_combo.currentText()
        self.qc_filter_combo.blockSignals(True)
        self.qc_filter_combo.clear()
        self.qc_filter_combo.addItem("All")
        keys = sorted({self._filter_key_for_ui(f) for f in filters if f})
        for key in keys:
            self.qc_filter_combo.addItem(key)
        if current:
            idx = self.qc_filter_combo.findText(current)
            if idx >= 0:
                self.qc_filter_combo.setCurrentIndex(idx)
        self.qc_filter_combo.blockSignals(False)

    def _refresh_qc_date_combo(self, dates: list[str]) -> None:
        if not hasattr(self, "qc_date_combo"):
            return
        current = self.qc_date_combo.currentText()
        self.qc_date_combo.blockSignals(True)
        self.qc_date_combo.clear()
        self.qc_date_combo.addItem("All")
        keys = list(dict.fromkeys(str(d) for d in dates if d and str(d) != "unknown"))
        for key in keys:
            self.qc_date_combo.addItem(key)
        # Restore previous selection; if not found, keep "All"
        prefer = getattr(self, "qc_date_last", "")
        if prefer and prefer in keys:
            self.qc_date_combo.setCurrentText(prefer)
        elif current and current in keys:
            self.qc_date_combo.setCurrentText(current)
        # else: leave "All" selected
        self.qc_date_combo.blockSignals(False)

    def _get_qc_selected_comp_id(self) -> int | None:
        rows = self.qc_table.selectionModel().selectedRows()
        if not rows:
            return None
        row_idx = rows[0].row()
        item = self.qc_table.item(row_idx, 1)
        if not item:
            return None
        try:
            return int(item.text())
        except Exception:
            return None

    def _select_qc_comp_id(self, comp_id: int) -> bool:
        for row_idx in range(self.qc_table.rowCount()):
            item = self.qc_table.item(row_idx, 1)
            if not item:
                continue
            try:
                cid = int(item.text())
            except Exception:
                continue
            if cid == int(comp_id):
                self.qc_table.setCurrentCell(row_idx, 1)
                self.qc_table.selectRow(row_idx)
                return True
        return False

    def run_comp_qc(self):
        if getattr(self, "_qc_running", False):
            return
        if not self.datasets:
            return

        # Collect params on main thread before spawning worker
        prev_comp_id = self._get_qc_selected_comp_id()
        if prev_comp_id is None:
            prev_comp_id = getattr(self, "_qc_last_selected_comp_id", None)

        target_id_text = self.target_edit.text().strip()
        if not target_id_text:
            target_id, comp_ids = _load_selection_ids(self.datasets[0][1])
            if target_id is None:
                self.log("[QC] Target ID missing")
                return
            self.target_edit.setText(str(target_id))
            if comp_ids:
                self.comp_edit.setText(",".join(str(i) for i in comp_ids))
                self.comp_candidate_ids = list(comp_ids)
        target_id = int(self.target_edit.text().strip())
        if not self.comp_candidate_ids:
            self.comp_candidate_ids = list(_safe_int_list(self.comp_edit.text()))

        result_dir = self._get_qc_result_dir()
        qc_files_all, qc_date_label_map = self._build_qc_date_label_map(result_dir)
        if qc_date_label_map:
            self._refresh_qc_date_combo([qc_date_label_map.get(fname, "") for fname in qc_files_all])

        date_sel = None
        files_for_metrics = None
        if hasattr(self, "qc_date_combo"):
            current_date = self.qc_date_combo.currentText()
            if current_date and current_date != "All":
                date_sel = current_date
        if date_sel:
            files_for_metrics = [fname for fname in qc_files_all if qc_date_label_map.get(fname) == date_sel]

        qc_signature = self._build_qc_signature(
            result_dir,
            target_id,
            list(self.comp_candidate_ids),
            files_override=files_for_metrics,
        )
        cached_rows = self._load_cached_comp_qc_summary(result_dir, qc_signature)
        if cached_rows is not None:
            self.qc_rows = cached_rows
            self._populate_qc_table(cached_rows)
            restored = False
            if prev_comp_id is not None:
                restored = self._select_qc_comp_id(prev_comp_id)
            if not restored and self.qc_table.rowCount() > 0:
                self.qc_table.setCurrentCell(0, 1)
                self.qc_table.selectRow(0)
            if hasattr(self, "qc_status_label"):
                self.qc_status_label.setText("QC loaded")
                self.qc_status_label.show()
            self.log(f"[QC] Loaded cached QC for {len(cached_rows)} comp(s).")
            if getattr(self, "_pending_auto_use", False):
                self._pending_auto_use = False
                self._apply_auto_use_thresholds()
            return

        self._qc_running = True
        self._qc_prev_comp_id = prev_comp_id
        self._qc_compute_result_dir = result_dir
        self._qc_compute_signature = qc_signature

        # Disable buttons and show progress bar while running
        if hasattr(self, "btn_qc_run"):
            self.btn_qc_run.setEnabled(False)
            self.btn_qc_run.setText("Running...")
        if hasattr(self, "btn_qc_auto"):
            self.btn_qc_auto.setEnabled(False)
        if hasattr(self, "qc_progress_bar"):
            self.qc_progress_bar.show()
        if hasattr(self, "qc_status_label"):
            self.qc_status_label.setText("QC running...")
            self.qc_status_label.show()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.log(f"[QC] Computing QC for {len(self.comp_candidate_ids)} comp(s)...")

        comp_candidates = list(self.comp_candidate_ids)
        files_override = files_for_metrics

        def _compute():
            return self._compute_comp_qc(
                result_dir, target_id, comp_candidates,
                files_override=files_override, verbose=True,
            )

        self._qc_worker = _QcComputeWorker(_compute)
        self._qc_worker.finished.connect(self._on_qc_rows_ready)
        self._qc_worker.error.connect(self._on_qc_error)
        self._qc_worker.start()

    def _on_qc_rows_ready(self, rows: list) -> None:
        self._qc_running = False
        if hasattr(self, "btn_qc_run"):
            self.btn_qc_run.setEnabled(True)
            self.btn_qc_run.setText("Run QC")
        if hasattr(self, "btn_qc_auto"):
            self.btn_qc_auto.setEnabled(True)
        if hasattr(self, "qc_progress_bar"):
            self.qc_progress_bar.hide()
        if hasattr(self, "qc_status_label"):
            self.qc_status_label.setText("QC done")
            self.qc_status_label.show()
        QApplication.restoreOverrideCursor()

        result_dir = getattr(self, "_qc_compute_result_dir", None)
        prev_comp_id = getattr(self, "_qc_prev_comp_id", None)

        self.qc_rows = rows
        self._populate_qc_table(rows)
        restored = False
        if prev_comp_id is not None:
            restored = self._select_qc_comp_id(prev_comp_id)
        if not restored and self.qc_table.rowCount() > 0:
            self.qc_table.setCurrentCell(0, 1)
            self.qc_table.selectRow(0)
        if result_dir is not None:
            self._save_comp_qc_summary(result_dir, rows)
            self._save_comp_qc_meta(result_dir, getattr(self, "_qc_compute_signature", None))
        self.log(f"[QC] Done. {len(rows)} comp(s) evaluated.")
        # If auto_use was pending (user clicked Auto Use before QC was run)
        if getattr(self, "_pending_auto_use", False):
            self._pending_auto_use = False
            self._apply_auto_use_thresholds()

    def _on_qc_error(self, msg: str) -> None:
        self._qc_running = False
        if hasattr(self, "btn_qc_run"):
            self.btn_qc_run.setEnabled(True)
            self.btn_qc_run.setText("Run QC")
        if hasattr(self, "btn_qc_auto"):
            self.btn_qc_auto.setEnabled(True)
        if hasattr(self, "qc_progress_bar"):
            self.qc_progress_bar.hide()
        if hasattr(self, "qc_status_label"):
            self.qc_status_label.setText("QC error")
            self.qc_status_label.show()
        QApplication.restoreOverrideCursor()
        self.log(f"[QC] Error: {msg}")

    def auto_use_qc(self):
        if not self.qc_rows:
            # QC not yet run — run it and apply thresholds when done
            self._pending_auto_use = True
            self.run_comp_qc()
            return
        self._apply_auto_use_thresholds()

    def _apply_auto_use_thresholds(self):
        if not self.qc_rows:
            return
        df = pd.DataFrame(self.qc_rows)
        if df.empty or "rms" not in df.columns:
            return
        df = df.copy()
        rms_max = float(self.qc_rms_max)
        frac_max = float(self.qc_outlier_frac_max)
        min_n = int(self.qc_min_points)
        if rms_max <= 0:
            rms_max = np.inf
        df["use_auto"] = (
            (df["n"].astype(float) >= min_n)
            & (df["rms"].astype(float) <= rms_max)
            & (df["outlier_frac"].astype(float) <= frac_max)
        )
        if df["use_auto"].sum() == 0:
            self.log("[QC] Auto Use: no comps passed thresholds (kept current selection)")
            return
        self.log(
            f"[QC] Auto Use: rms<= {self.qc_rms_max:.4f}, "
            f"outlier_frac<= {self.qc_outlier_frac_max:.3f}, min_n>= {self.qc_min_points}"
        )
        self._qc_table_block = True
        for row_idx in range(self.qc_table.rowCount()):
            item_id = self.qc_table.item(row_idx, 1)
            item_use = self.qc_table.item(row_idx, 0)
            if not item_id or not item_use:
                continue
            try:
                cid = int(item_id.text())
            except Exception:
                continue
            use = bool(df[df["comp_id"] == cid]["use_auto"].iloc[0]) if cid in df["comp_id"].values else False
            item_use.setCheckState(Qt.Checked if use else Qt.Unchecked)
        self._qc_table_block = False
        self.apply_qc_selection()

    def apply_qc_selection(self):
        comp_ids = []
        for row_idx in range(self.qc_table.rowCount()):
            item_id = self.qc_table.item(row_idx, 1)
            item_use = self.qc_table.item(row_idx, 0)
            if not item_id:
                continue
            try:
                cid = int(item_id.text())
            except Exception:
                continue
            if item_use and item_use.checkState() == Qt.Checked:
                comp_ids.append(cid)
        self.comp_ids_list = comp_ids
        self.comp_edit.setText(",".join(str(i) for i in comp_ids))
        self._update_comp_ids_from_input()
        self.plot_current_comparison()
        self.log(f"[QC] Applied comp list: {comp_ids}")

    def _on_qc_table_changed(self, item: QTableWidgetItem):
        if self._qc_table_block:
            return
        if item.column() != 0:
            return
        self.apply_qc_selection()

    def _on_qc_selection_changed(self):
        comp_id = self._get_qc_selected_comp_id()
        if comp_id is None:
            return
        self._qc_last_selected_comp_id = comp_id
        self._plot_comp_preview(comp_id)
        self._refresh_qc_stats_panel()

    def _refresh_qc_stats_panel(self, *_):
        comp_id = self._get_qc_selected_comp_id()
        if not hasattr(self, "qc_stats_table") or comp_id is None:
            return
        by_night = hasattr(self, "qc_stats_night_cb") and self.qc_stats_night_cb.isChecked()

        cache = getattr(self, "_qc_checkstar_cache", {})
        df = cache.get(int(comp_id))
        if df is None or df.empty:
            self.qc_stats_table.setRowCount(0)
            return

        # Apply current frame exclusions to stats
        result_dir = self._current_result_dir()
        if result_dir is not None:
            excl = set(self._get_frame_exclude_map(result_dir).keys())
            if excl and "file" in df.columns:
                df = df[~df["file"].astype(str).isin(excl)]

        y_col = "diff_mag_raw" if "diff_mag_raw" in df.columns else ("diff_mag" if "diff_mag" in df.columns else "mag")
        filt_col = "filter" if "filter" in df.columns else None
        night_col = "night_id" if "night_id" in df.columns else None

        rows = []
        groups = []
        if by_night and night_col and filt_col:
            for (flt, night), sub in df.groupby([filt_col, night_col]):
                groups.append((f"{flt} N{night}", sub))
        elif filt_col:
            for flt, sub in df.groupby(filt_col):
                groups.append((str(flt), sub))
        else:
            groups.append(("all", df))

        for label, sub in groups:
            y = pd.to_numeric(sub[y_col], errors="coerce").dropna().to_numpy(float)
            y = y[np.isfinite(y)]
            if len(y) == 0:
                continue
            med = float(np.nanmedian(y))
            sig = float(np.nanstd(y))
            mad = float(np.nanmedian(np.abs(y - med)))
            rows.append((label, len(y), med, sig, mad))

        self.qc_stats_table.setRowCount(len(rows))
        for i, (label, n, med, sig, mad) in enumerate(rows):
            self.qc_stats_table.setItem(i, 0, QTableWidgetItem(label))
            self.qc_stats_table.setItem(i, 1, QTableWidgetItem(str(n)))
            self.qc_stats_table.setItem(i, 2, QTableWidgetItem(f"{med:.4f}"))
            sig_item = QTableWidgetItem(f"{sig:.4f}")
            if sig > 0.03:
                sig_item.setForeground(QColor("#E53935"))
            elif sig > 0.015:
                sig_item.setForeground(QColor("#FB8C00"))
            self.qc_stats_table.setItem(i, 3, sig_item)
            self.qc_stats_table.setItem(i, 4, QTableWidgetItem(f"{mad:.4f}"))

    def _init_qc_view(self) -> None:
        """Initialize QC tab: populate table with all candidates checked (no metric computation)."""
        if not self.datasets:
            return
        if not self.target_edit.text().strip() or not self.comp_edit.text().strip():
            target_id, comp_ids = _load_selection_ids(self.datasets[0][1])
            if target_id is not None and not self.target_edit.text().strip():
                self.target_edit.setText(str(target_id))
            if comp_ids and not self.comp_edit.text().strip():
                self.comp_edit.setText(",".join(str(i) for i in comp_ids))
                self.comp_candidate_ids = list(comp_ids)
        self._update_comp_ids_from_input()
        # Populate table with all candidates checked — no heavy QC computation on open
        if self.comp_candidate_ids and self.qc_table.rowCount() == 0:
            active_set = set(self.comp_ids_list)
            rows = [{"comp_id": cid, "use": cid in active_set or not active_set}
                    for cid in self.comp_candidate_ids]
            self._populate_qc_table(rows)

    def _on_qc_plot_click(self, event) -> None:
        self.check_plot_canvas.setFocus()
        if not event.dblclick:
            return
        if event.inaxes != self.check_plot_ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        TOLERANCE_PX = 20
        ax = self.check_plot_ax
        click_disp = ax.transData.transform([[event.xdata, event.ydata]])[0]
        best_dist = float("inf")
        best_fname = None
        best_xy = None
        for artist, meta in self._qc_plot_point_map.items():
            offsets = artist.get_offsets()
            files = meta.get("files", [])
            if len(offsets) == 0 or len(files) == 0:
                continue
            try:
                disp = ax.transData.transform(np.asarray(offsets))
            except Exception:
                continue
            dists = np.linalg.norm(disp - click_disp, axis=1)
            idx = int(np.argmin(dists))
            if dists[idx] < best_dist and idx < len(files):
                best_dist = dists[idx]
                best_fname = str(files[idx])
                arr = np.asarray(offsets)
                best_xy = (float(arr[idx, 0]), float(arr[idx, 1]))
        if best_fname and best_dist <= TOLERANCE_PX:
            result_dir = self._current_result_dir()
            self._set_selected_frame(best_fname, result_dir)
            self._draw_qc_selection_indicator(best_xy)

    def _frame_group_key(self, fname: str) -> str:
        stem = Path(str(fname)).stem
        nums = re.findall(r"\d+", stem)
        if not nums:
            return ""
        for num in reversed(nums):
            if len(num) != 8:
                return num
        return ""

    def _collect_qc_preview_files(self, result_dir: Path) -> tuple[list[str], list[str], list[str]]:
        """Return (files_all_filters, files_filtered_by_filter_sel, filters_all) for QC preview."""
        idx_path = step5_photometry_dir(result_dir) / "photometry_index.csv"
        if not idx_path.exists():
            return [], [], []

        # Cache photometry_index by mtime
        cache_key = str(idx_path)
        try:
            mtime = idx_path.stat().st_mtime
        except OSError:
            mtime = 0.0
        cached = self._qc_idx_cache.get(cache_key)
        if cached and cached[0] == mtime:
            idx = cached[1]
        else:
            try:
                idx = pd.read_csv(idx_path)
            except Exception:
                return [], [], []
            self._qc_idx_cache[cache_key] = (mtime, idx)

        if "file" not in idx.columns:
            return [], [], []

        files = idx["file"].astype(str).tolist()
        _, qc_date_label_map = self._build_qc_date_label_map(result_dir, idx)
        filter_map = {}
        if "filter" in idx.columns:
            filter_map = dict(zip(idx["file"].astype(str), idx["filter"].astype(str)))
        elif "FILTER" in idx.columns:
            filter_map = dict(zip(idx["file"].astype(str), idx["FILTER"].astype(str)))

        # Cache headers by result_dir
        h_key = str(result_dir)
        h_cached = self._qc_headers_cache.get(h_key)
        if h_cached:
            headers_map, headers_df = h_cached[1], h_cached[2]
        else:
            headers_map = _load_headers_map(result_dir)
            headers_df = _load_headers_table(result_dir)
            self._qc_headers_cache[h_key] = (mtime, headers_map, headers_df)
        header_filter_map = {}
        if not headers_df.empty and "Filename" in headers_df.columns:
            for col in ("FILTER", "filter"):
                if col in headers_df.columns:
                    header_filter_map = dict(zip(headers_df["Filename"].astype(str), headers_df[col].astype(str)))
                    break

        date_sel = self.qc_date_combo.currentText() if hasattr(self, "qc_date_combo") else "All"
        if date_sel and date_sel != "All":
            files = [fname for fname in files if qc_date_label_map.get(fname) == date_sel]

        files_all = list(files)

        filters_all = list(dict.fromkeys(
            fkey for fname in files_all
            for fkey in [self._filter_key_for_ui(str(filter_map.get(fname, "") or header_filter_map.get(fname, "")))]
            if fkey
        ))

        filter_sel = self.qc_filter_combo.currentText() if hasattr(self, "qc_filter_combo") else "All"
        if filter_sel and filter_sel not in ("All", "", None):
            filtered = [
                fname for fname in files_all
                if self._filter_key_for_ui(str(filter_map.get(fname, "") or header_filter_map.get(fname, ""))) == filter_sel
            ]
            return files_all, filtered, filters_all

        return files_all, files_all, filters_all

    def _plot_comp_preview(self, comp_id: int) -> None:
        if not self.datasets:
            return
        result_dir = self._get_qc_result_dir()
        files_all, files_use, filters_all = self._collect_qc_preview_files(result_dir)
        self._qc_scope_files_all_filters = files_all
        # Use cached check-star diff series if available; otherwise build on demand.
        cache = getattr(self, "_qc_checkstar_cache", {})
        if int(comp_id) in cache:
            df = cache[int(comp_id)].copy()
            _use_diff = True
            # date/filter selection 적용
            if files_use is not None and set(files_use) != set(files_all):
                df = df[df["file"].isin(set(files_use))].copy()
        else:
            qc_comp_ids = [int(c) for c in (self.comp_candidate_ids or self.comp_ids_list) if str(c).strip()]
            other_comps = [cid for cid in qc_comp_ids if int(cid) != int(comp_id)]
            if other_comps:
                df = self._build_ensemble_series(
                    result_dir,
                    int(comp_id),
                    other_comps,
                    verbose=False,
                )
                cache[int(comp_id)] = df.copy()
                if files_use is not None and set(files_use) != set(files_all):
                    df = df[df["file"].isin(set(files_use))].copy()
                _use_diff = not df.empty
            else:
                df = self._build_star_mag_series(
                    result_dir,
                    int(comp_id),
                    verbose=False,
                    include_excluded=True,
                    files_override=files_use,
                    preload=False,
                )
                _use_diff = False
        self.check_plot_ax.clear()
        self._qc_selection_indicator_artist = None  # axes cleared; old artist is detached
        self._qc_plot_point_map = {}
        self._qc_plot_points = {"x": [], "y": [], "file": []}
        if df.empty:
            self.check_plot_canvas.draw_idle()
            return
        if filters_all:
            self._refresh_qc_filter_combo(filters_all)
        self._qc_scope_df = df.copy()

        x = df["JD"].to_numpy(float) if "JD" in df.columns else df.get("rel_time_hr", pd.Series(np.arange(len(df)))).to_numpy(float)
        y = df["diff_mag_raw"].to_numpy(float) if _use_diff and "diff_mag_raw" in df.columns else df["mag"].to_numpy(float)
        night_ids_plot = df["night_id"].to_numpy(int) if "night_id" in df.columns else np.zeros(len(df), int)
        files = df["file"].astype(str).to_numpy()
        filters = df["filter"].astype(str).tolist() if "filter" in df.columns else ["?"] * len(df)
        plotted_y = []
        excluded_files = set(self._get_frame_exclude_map(result_dir).keys())
        if np.isfinite(x).any() and np.isfinite(y).any():
            for fkey in sorted(set(filters)):
                key_ui = self._filter_key_for_ui(fkey)
                idx = [i for i, f in enumerate(filters) if f == fkey]
                if not idx:
                    continue
                xv = x[idx]
                yv = y[idx]
                fv = files[idx]
                m = np.isfinite(xv) & np.isfinite(yv)
                if not np.any(m):
                    continue
                color = self._get_filter_color(key_ui)
                excl = np.array([f in excluded_files for f in fv], dtype=bool)
                m_in = m & ~excl
                m_ex = m & excl
                if np.any(m_in):
                    sc = self.check_plot_ax.scatter(
                        xv[m_in], yv[m_in], s=10, color=color, alpha=0.8, picker=5, label=key_ui
                    )
                    self._qc_plot_point_map[sc] = {"files": fv[m_in].tolist()}
                    self._qc_plot_points["x"].extend(xv[m_in].tolist())
                    self._qc_plot_points["y"].extend(yv[m_in].tolist())
                    self._qc_plot_points["file"].extend(fv[m_in].tolist())
                if np.any(m_ex):
                    scx = self.check_plot_ax.scatter(
                        xv[m_ex], yv[m_ex], s=14, color="#9E9E9E", marker="x", alpha=0.9, picker=5
                    )
                    self._qc_plot_point_map[scx] = {"files": fv[m_ex].tolist()}
                    self._qc_plot_points["x"].extend(xv[m_ex].tolist())
                    self._qc_plot_points["y"].extend(yv[m_ex].tolist())
                    self._qc_plot_points["file"].extend(fv[m_ex].tolist())
                med = np.nanmedian(yv[m])
                if np.isfinite(med):
                    self.check_plot_ax.axhline(med, color=color, linestyle="--", linewidth=1, alpha=0.7)
                plotted_y.extend(yv[m].tolist())
        self.check_plot_ax.set_title(f"Comp ID {comp_id}")
        self.check_plot_ax.set_xlabel("Time (hr)")
        ylabel = "Δmag (check star)" if _use_diff else "Inst. Mag"
        self.check_plot_ax.set_ylabel(ylabel, fontsize=9)
        if _use_diff:
            self.check_plot_ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
            self.check_plot_ax.invert_yaxis()
        self.check_plot_ax.grid(True, alpha=0.3)
        if plotted_y:
            y_arr = np.array(plotted_y, float)
            scale_mode = self.qc_scale_mode
            if scale_mode.startswith("Robust"):
                med = np.nanmedian(y_arr)
                mad = np.nanmedian(np.abs(y_arr - med))
                if np.isfinite(med) and np.isfinite(mad) and mad > 0:
                    k = float(self.qc_scale_mad_value)
                    self.check_plot_ax.set_ylim(med + k * mad, med - k * mad)
            elif scale_mode == "Fixed":
                med = np.nanmedian(y_arr)
                half = float(self.qc_scale_fixed_value)
                self.check_plot_ax.set_ylim(med + half, med - half)
            else:
                self.check_plot_ax.invert_yaxis()
        else:
            self.check_plot_ax.invert_yaxis()
        handles, labels = self.check_plot_ax.get_legend_handles_labels()
        if handles:
            self.check_plot_ax.legend(loc="best", fontsize=8)
        # Redraw selection indicator (single source of truth)
        if self._frame_qc_selected:
            try:
                sel_idx = np.where(files == str(self._frame_qc_selected))[0]
                if len(sel_idx) > 0:
                    i0 = int(sel_idx[0])
                    sc = self.check_plot_ax.scatter(
                        [x[i0]], [y[i0]], s=220, facecolors="none",
                        edgecolors="red", linewidths=2.0, zorder=20
                    )
                    self._qc_selection_indicator_artist = sc
            except Exception:
                pass
        self.check_plot_canvas.draw_idle()

    def build_light_curve(self):
        if not self.datasets:
            QMessageBox.information(self, "Light Curve", "데이터셋이 없습니다.")
            return

        target_id = self.target_edit.text().strip()
        if not target_id:
            target_id, comp_ids = _load_selection_ids(self.datasets[0][1])
            if target_id is None:
                QMessageBox.information(self, "Light Curve", "대상 ID가 필요합니다.")
                return
        else:
            target_id = int(target_id)
            comp_ids = _safe_int_list(self.comp_edit.text())

        self._update_comp_ids_from_input()
        if not comp_ids:
            QMessageBox.information(self, "Light Curve", "비교성 ID가 필요합니다.")
            return

        active_comp_ids = list(comp_ids)

        self.log("=" * 60)
        self.log("[BUILD] Starting Light Curve Build (RAW)")
        self.log(f"[BUILD] Target ID: {target_id}")
        self.log(f"[BUILD] Active Comp IDs: {active_comp_ids}")
        self.log(f"[BUILD] Datasets: {len(self.datasets)}")

        # QC 요약 저장
        if active_comp_ids:
            qc_rows = self._compute_comp_qc(self.datasets[0][1], target_id, active_comp_ids, verbose=False)
            self._save_comp_qc_summary(Path(self.datasets[0][1]), qc_rows)

        combined_raw = []
        single_dataset_mode = len(self.datasets) == 1
        for label, result_dir in self.datasets:
            result_dir = Path(result_dir)
            raw_df = self._build_ensemble_series(result_dir, target_id, active_comp_ids, verbose=True)
            if raw_df.empty:
                self.log(f"[{label}] Raw light curve empty")
                continue
            raw_df = annotate_raw_lightcurve(raw_df, label, logger=self.log)

            # Export check star light curve if defined
            check_ids_by_filter = {}
            check_df = pd.DataFrame()
            try:
                check_ids_by_filter, check_df = self._build_check_star_series(
                    result_dir,
                    active_comp_ids,
                    verbose=False,
                )
                if check_ids_by_filter and check_df.empty:
                    self.log("  Check star configured but no usable check-star light curve was built")
            except Exception as e:
                self.log(f"  Check star export failed: {e}")

            save_dataset_raw_outputs(
                result_dir=result_dir,
                target_id=target_id,
                raw_df=raw_df,
                check_ids_by_filter=check_ids_by_filter,
                check_df=check_df,
                logger=self.log,
            )
            combined_raw.append(raw_df)

        # combined outputs
        save_combined_raw_outputs(
            base_result_dir=Path(self.params.P.result_dir),
            target_id=target_id,
            combined_raw=combined_raw,
            single_dataset_mode=single_dataset_mode,
            comp_candidate_ids=self.comp_candidate_ids,
            active_comp_ids=active_comp_ids,
            logger=self.log,
        )

        self.log("=" * 60)
        self.log("[BUILD] Light Curve Build Complete (RAW)")
        if combined_raw:
            all_data = pd.concat(combined_raw, ignore_index=True)
            valid_y = all_data["diff_mag_raw"].dropna()
            n_total = len(all_data)
            n_valid = len(valid_y)
            if n_valid > 0:
                y_mean = valid_y.mean()
                y_std = valid_y.std()
                y_range = valid_y.max() - valid_y.min()
                self.log(f"[RESULT] RAW: {n_valid}/{n_total} valid points")
                self.log(f"[RESULT] RAW: mean={y_mean:.4f}, std={y_std:.4f}, range={y_range:.4f} mag")
            else:
                self.log(f"[RESULT] RAW: 0/{n_total} valid points - CHECK DETECTION!")
        self.log("=" * 60)

        self.save_state()
        self.plot_current_comparison()
        self.show_log_window()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Left:
            self._step_comp(-1)
            return
        if event.key() == Qt.Key_Right:
            self._step_comp(1)
            return
        if event.key() == Qt.Key_D:
            if event.modifiers() & Qt.ShiftModifier:
                self._exclude_selected_frame_all_filters()
            else:
                self._exclude_selected_frame()
            return
        if event.key() == Qt.Key_A:
            self._include_selected_frame()
            return
        if event.key() == Qt.Key_R:
            self.clear_frame_excludes()
            return
        super().keyPressEvent(event)

    def validate_step(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Multi-dataset helpers
    # ------------------------------------------------------------------

    def _dataset_summary_text(self) -> str:
        count = len(self.datasets)
        if count <= 1:
            return "현재 1개 폴더 사용 중"
        return f"총 {count}개 폴더 사용 중"

    def _update_dataset_summary(self) -> None:
        if hasattr(self, "ds_summary_label"):
            self.ds_summary_label.setText(self._dataset_summary_text())
            dataset_names = ", ".join(label for label, _ in self.datasets)
            self.ds_summary_label.setToolTip(dataset_names)

    def _set_dataset_panel_expanded(self, expanded: bool, persist: bool = False) -> None:
        self.dataset_panel_expanded = bool(expanded)
        if hasattr(self, "btn_ds_toggle"):
            self.btn_ds_toggle.blockSignals(True)
            self.btn_ds_toggle.setChecked(self.dataset_panel_expanded)
            self.btn_ds_toggle.setText("▼ 접기" if self.dataset_panel_expanded else "▶ 펼치기")
            self.btn_ds_toggle.blockSignals(False)
        if hasattr(self, "ds_container"):
            self.ds_container.setVisible(self.dataset_panel_expanded)
        if persist:
            self.save_state()

    def _on_add_dataset(self):
        p = QFileDialog.getExistingDirectory(self, "추가 result 폴더 선택", str(self.params.P.result_dir))
        if not p:
            return
        p = Path(p)
        for _, existing in self.datasets:
            if existing.resolve() == p.resolve():
                return
        self.datasets.append((p.name, p))
        self.ds_list_widget.addItem(p.name)
        self._update_dataset_summary()
        self.save_state()

    def _on_remove_dataset(self):
        row = self.ds_list_widget.currentRow()
        if row <= 0:  # row 0 is locked (current result_dir)
            return
        self.ds_list_widget.takeItem(row)
        if row < len(self.datasets):
            self.datasets.pop(row)
        self._update_dataset_summary()
        self.save_state()

    def save_state(self):
        state_data = {
            "build_diff": self.opt_diff,
            "comp_candidates": ",".join(str(i) for i in self.comp_candidate_ids),
            "qc_rms_max": self.qc_rms_max,
            "qc_sigma": self.qc_sigma,
            "qc_outlier_frac": self.qc_outlier_frac_max,
            "qc_min_points": self.qc_min_points,
            "qc_scale_mode": self.qc_scale_mode,
            "qc_scale_mad": self.qc_scale_mad_value,
            "qc_scale_fixed": self.qc_scale_fixed_value,
            "x_axis_mode": self.x_axis_mode,
            "phase_period": self.phase_period,
            "phase_t0": self.phase_t0,
            "phase_cycles": self.phase_cycles,
            "period_min": self.period_min,
            "period_max": self.period_max,
            "filter_visibility": self.filter_visibility,
            "filter_colors": self.filter_colors,
            "dataset_panel_expanded": self.dataset_panel_expanded,
            "extra_result_dirs": [str(p) for _, p in self.datasets[1:]],
        }
        self.project_state.store_step_data("light_curve", state_data)

    def restore_state(self):
        state_data = self.project_state.get_step_data("light_curve")
        if state_data:
            self.opt_diff = bool(state_data.get("build_diff", True))
            candidates_text = state_data.get("comp_candidates", "")
            self.comp_candidate_ids = _safe_int_list(candidates_text)
            self.qc_rms_max = float(state_data.get("qc_rms_max", self.qc_rms_max))
            self.qc_sigma = float(state_data.get("qc_sigma", self.qc_sigma))
            self.qc_outlier_frac_max = float(state_data.get("qc_outlier_frac", self.qc_outlier_frac_max))
            self.qc_min_points = int(state_data.get("qc_min_points", self.qc_min_points))
            self.qc_scale_mode = state_data.get("qc_scale_mode", self.qc_scale_mode)
            self.qc_scale_mad_value = float(state_data.get("qc_scale_mad", self.qc_scale_mad_value))
            self.qc_scale_fixed_value = float(state_data.get("qc_scale_fixed", self.qc_scale_fixed_value))
            self.x_axis_mode = state_data.get("x_axis_mode", "time")
            if self.x_axis_mode not in ("time", "phase"):
                self.x_axis_mode = "time"
            self.phase_period = float(state_data.get("phase_period", 0.0))
            self.phase_t0 = float(state_data.get("phase_t0", 0.0))
            self.phase_cycles = max(1.0, float(state_data.get("phase_cycles", 1.0)))
            self.period_min = float(state_data.get("period_min", 0.01))
            self.period_max = float(state_data.get("period_max", 10.0))
            self.dataset_panel_expanded = bool(state_data.get("dataset_panel_expanded", False))
            self.filter_visibility = {
                str(k): bool(v) for k, v in (state_data.get("filter_visibility", {}) or {}).items()
            }
            self.filter_colors = {
                str(k): str(v) for k, v in (state_data.get("filter_colors", {}) or {}).items()
            }
            # X축 콤보박스 동기화 (time=0, phase=1)
            if hasattr(self, "x_axis_combo"):
                mode_map = {"time": 0, "phase": 1}
                self.x_axis_combo.setCurrentIndex(mode_map.get(self.x_axis_mode, 0))
            # 슬라이더 동기화
            if hasattr(self, "period_slider"):
                self._update_sliders_from_values()
        self._update_comp_ids_from_input()
        self._update_qc_threshold_label()
        self._update_qc_gate_ui()
        # Restore extra datasets
        if state_data:
            for dir_str in state_data.get("extra_result_dirs", []):
                p = Path(dir_str)
                if not p.exists():
                    continue
                if any(existing.resolve() == p.resolve() for _, existing in self.datasets):
                    continue
                self.datasets.append((p.name, p))
                if hasattr(self, "ds_list_widget"):
                    self.ds_list_widget.addItem(p.name)
        self._update_dataset_summary()
        self._set_dataset_panel_expanded(self.dataset_panel_expanded, persist=False)
