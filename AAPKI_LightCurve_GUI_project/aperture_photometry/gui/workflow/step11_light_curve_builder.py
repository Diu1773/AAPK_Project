"""
Step 11: Light Curve Builder
"""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from astropy.time import Time
from astropy.io import fits

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (
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
    QFileDialog,
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
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence, QColor
from PyQt5.QtWidgets import QShortcut, QStyle, QStyleOptionSlider

from .step_window_base import StepWindowBase


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
from ...utils.astro_utils import compute_airmass_from_header
from ...utils.step_paths import (
    step1_dir,
    step2_cropped_dir,
    step6_dir,
    step8_dir,
    step9_dir,
    step11_dir,
    step11_extinction_dir,
    step11_zeropoint_dir,
    tool_extinction_dir,
    legacy_step5_refbuild_dir,
    legacy_step7_refbuild_dir,
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


def _parse_jd(date_obs: str | None) -> float:
    if not date_obs:
        return np.nan
    try:
        return float(Time(str(date_obs).strip()).jd)
    except Exception:
        return np.nan


def _date_from_dateobs(date_obs: str | None) -> str:
    if not date_obs:
        return "unknown"
    try:
        t = Time(str(date_obs).strip())
        return t.to_datetime().strftime("%Y-%m-%d")
    except Exception:
        return "unknown"


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


def _load_headers_map(result_dir: Path) -> dict:
    df = _load_headers_table(result_dir)
    if df.empty:
        return {}
    if "Filename" in df.columns and "DATE-OBS" in df.columns:
        return dict(zip(df["Filename"].astype(str), df["DATE-OBS"].astype(str)))
    return {}


def _normalize_filter_key(value: str | None) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


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
    idx_path = step9_dir(result_dir) / "photometry_index.csv"
    if not idx_path.exists():
        idx_path = result_dir / "photometry_index.csv"
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

    for _, idx_row in idx_df.iterrows():
        fname = str(idx_row["file"])
        phot_path = step9_dir(result_dir) / f"{fname}_photometry.tsv"
        if not phot_path.exists():
            phot_path = result_dir / f"{fname}_photometry.tsv"
        if not phot_path.exists():
            continue

        try:
            df = pd.read_csv(phot_path, sep="\t")
        except Exception:
            try:
                df = pd.read_csv(phot_path)
            except Exception:
                continue

        # Get filter for this frame
        filt = ""
        if "filter" in idx_df.columns:
            filt = _normalize_filter_key(str(idx_row.get("filter", "")))
        elif "FILTER" in idx_df.columns:
            filt = _normalize_filter_key(str(idx_row.get("FILTER", "")))

        if not filt or filt not in filters_normalized:
            continue

        if "ID" not in df.columns:
            continue

        # Convert ID column to int for matching
        try:
            df_ids = pd.to_numeric(df["ID"], errors="coerce").astype("Int64")
        except Exception:
            continue

        # Extract magnitudes for each star
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
    # Check tool_extinction_dir first, then legacy step11_zeropoint_dir
    path = tool_extinction_dir(result_dir) / "frame_airmass.csv"
    if not path.exists():
        path = step11_zeropoint_dir(result_dir) / "frame_airmass.csv"
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
    # Priority: tool_extinction_dir > step11_extinction_dir > legacy paths
    candidates = [
        tool_extinction_dir(result_dir) / "extinction_fit_by_filter.csv",
        step11_extinction_dir(result_dir) / "step11_extinction_fit_by_filter.csv",
        step11_zeropoint_dir(result_dir) / "step11_extinction" / "step11_extinction_fit_by_filter.csv",
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
    # Priority: tool_extinction_dir > step11_extinction_dir > legacy paths
    candidates = [
        tool_extinction_dir(result_dir) / "extinction_fit_by_filter.csv",
        step11_extinction_dir(result_dir) / "step11_extinction_fit_by_filter.csv",
        step11_zeropoint_dir(result_dir) / "step11_extinction" / "step11_extinction_fit_by_filter.csv",
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
    # Check legacy step11_zeropoint_dir for median data
    candidates = [
        step11_zeropoint_dir(result_dir) / "median_by_ID_filter_wide.csv",
        step11_zeropoint_dir(result_dir) / "median_by_ID_filter_wide_cmd.csv",
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
    selection_path = step8_dir(result_dir) / "target_selection.json"
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
            step8_out = step8_dir(result_dir)
            if not step8_out.exists():
                return {}
            candidates = []
            if flt:
                candidates.append(step8_out / f"master_catalog_{_normalize_filter_key(flt)}.tsv")
            candidates.extend(sorted(step8_out.glob("master_catalog_*.tsv")))
            for path in candidates:
                if not path.exists():
                    continue
                try:
                    df = pd.read_csv(path, sep="\t")
                    if {"source_id", "ID"} <= set(df.columns):
                        sid = pd.to_numeric(df["source_id"], errors="coerce").dropna().astype("int64")
                        fid = pd.to_numeric(df["ID"], errors="coerce").dropna().astype("int64")
                        id_map = dict(zip(sid.tolist(), fid.tolist()))
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
            src_path = step6_dir(result_dir) / "sourceid_to_ID.csv"
            if not src_path.exists():
                src_path = legacy_step5_refbuild_dir(result_dir) / "sourceid_to_ID.csv"
            if not src_path.exists():
                src_path = legacy_step7_refbuild_dir(result_dir) / "sourceid_to_ID.csv"
            if not src_path.exists():
                src_path = result_dir / "sourceid_to_ID.csv"
            src_id = target_sid
            if src_path.exists() and src_id is not None:
                try:
                    df = pd.read_csv(src_path)
                    if {"source_id", "ID"} <= set(df.columns):
                        row = df[df["source_id"].astype("int64") == int(src_id)]
                        if not row.empty:
                            target_id = int(row.iloc[0]["ID"])
                except Exception:
                    target_id = None
        if (not comp_ids) and comp_sids:
            src_path = step6_dir(result_dir) / "sourceid_to_ID.csv"
            if not src_path.exists():
                src_path = legacy_step5_refbuild_dir(result_dir) / "sourceid_to_ID.csv"
            if not src_path.exists():
                src_path = legacy_step7_refbuild_dir(result_dir) / "sourceid_to_ID.csv"
            if not src_path.exists():
                src_path = result_dir / "sourceid_to_ID.csv"
            src_ids = [int(s) for s in comp_sids if s is not None]
            if src_path.exists() and src_ids:
                try:
                    df = pd.read_csv(src_path)
                    if {"source_id", "ID"} <= set(df.columns):
                        df["source_id"] = df["source_id"].astype("int64")
                        df["ID"] = df["ID"].astype("int64")
                        sel = df[df["source_id"].isin(src_ids)]
                        comp_ids = sel["ID"].astype("int64").tolist()
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
    step8_out = step8_dir(result_dir)
    filter_selections = {}

    if not step8_out.exists():
        return {}

    for sel_path in step8_out.glob("selection_*.json"):
        flt = sel_path.stem.replace("selection_", "")
        try:
            data = json.loads(sel_path.read_text(encoding="utf-8"))
            target_id = data.get("target_id")
            comp_ids = data.get("comparison_ids", [])
            target_source_id = data.get("target_source_id")
            comp_source_ids = data.get("comparison_source_ids", [])

            filter_selections[flt] = {
                "target_id": int(target_id) if target_id is not None else None,
                "comparison_ids": [int(x) for x in comp_ids if x is not None],
                "target_source_id": int(target_source_id) if target_source_id is not None else None,
                "comparison_source_ids": [int(x) for x in comp_source_ids if x is not None],
            }
        except Exception:
            continue

    return filter_selections


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
    """Step 11: Light curve builder (diff/abs)."""

    def __init__(self, params, file_manager, project_state, main_window):
        self.file_manager = file_manager
        self.datasets = []
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

        # 필터별 플롯 표시/색상 설정
        self.filter_visibility: dict[str, bool] = {}
        self.filter_colors: dict[str, str] = {}
        self._filter_keys: list[str] = []
        self.filter_control_map: dict[str, QPushButton] = {}

        super().__init__(
            step_index=10,
            step_name="Light Curve Builder",
            params=params,
            project_state=project_state,
            main_window=main_window,
        )
        self.setFocusPolicy(Qt.StrongFocus)
        self.setup_step_ui()
        self.restore_state()

    def setup_step_ui(self):
        info = QLabel(
            "대상/비교성 선택 결과를 이용해 라이트커브를 생성합니다.\n"
            "RAW 차등측광을 생성하고 비교성 QC를 수행합니다."
        )
        info.setStyleSheet("QLabel { background-color: #E3F2FD; padding: 10px; border-radius: 5px; }")
        self.content_layout.addWidget(info)

        dataset_group = QGroupBox("데이터셋 (result_dir)")
        dataset_layout = QVBoxLayout(dataset_group)

        self.dataset_table = QTableWidget()
        self.dataset_table.setColumnCount(2)
        self.dataset_table.setHorizontalHeaderLabels(["라벨", "경로"])
        self.dataset_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.dataset_table.horizontalHeader().setStretchLastSection(True)
        self.dataset_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.dataset_table.setEditTriggers(QTableWidget.NoEditTriggers)
        dataset_layout.addWidget(self.dataset_table)

        btn_row = QHBoxLayout()
        btn_add = QPushButton("결과 폴더 추가")
        btn_add.clicked.connect(self.add_dataset)
        btn_row.addWidget(btn_add)

        btn_remove = QPushButton("선택 삭제")
        btn_remove.clicked.connect(self.remove_selected_dataset)
        btn_row.addWidget(btn_remove)

        btn_use_current = QPushButton("현재 result_dir 사용")
        btn_use_current.clicked.connect(self.use_current_dataset)
        btn_row.addWidget(btn_use_current)
        btn_row.addStretch()
        dataset_layout.addLayout(btn_row)
        self.content_layout.addWidget(dataset_group)

        target_group = QGroupBox("대상 / 비교성")
        target_layout = QHBoxLayout(target_group)
        target_layout.addWidget(QLabel("대상 ID:"))
        self.target_edit = QLineEdit()
        self.target_edit.setMaximumWidth(120)
        target_layout.addWidget(self.target_edit)

        target_layout.addWidget(QLabel("비교성 IDs:"))
        self.comp_edit = QLineEdit()
        self.comp_edit.setPlaceholderText("comma-separated")
        target_layout.addWidget(self.comp_edit)

        btn_load = QPushButton("선택값 불러오기")
        btn_load.clicked.connect(self.load_from_selection)
        target_layout.addWidget(btn_load)
        self.content_layout.addWidget(target_group)

        self.tab_widget = QTabWidget()
        self.light_tab = QWidget()
        self.qc_tab = QWidget()
        self.light_layout = QVBoxLayout(self.light_tab)
        self.qc_layout = QVBoxLayout(self.qc_tab)
        self.tab_widget.addTab(self.light_tab, "Light Curve")
        self.tab_widget.addTab(self.qc_tab, "Comparison QC")
        self.content_layout.addWidget(self.tab_widget)

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

        btn_row.addStretch()

        # Plot 버튼 (자동 저장 포함)
        btn_plot = QPushButton("Plot && Save")
        btn_plot.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 4px 16px; font-size: 10pt; }")
        btn_plot.clicked.connect(self.plot_and_save)
        btn_row.addWidget(btn_plot)

        plot_layout.addLayout(btn_row)

        filter_box = QGroupBox("Filters")
        filter_box.setStyleSheet(
            "QGroupBox { background-color: #EEF2F5; border: 1px solid #CFD8DC; border-radius: 6px; margin-top: 6px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; color: #455A64; font-weight: bold; }"
        )
        filter_layout = QVBoxLayout(filter_box)
        filter_btn_row = QHBoxLayout()
        self.btn_filter_all = QPushButton("All")
        self.btn_filter_all.setStyleSheet("QPushButton { background-color: #ECEFF1; border: 1px solid #B0BEC5; border-radius: 4px; padding: 2px 8px; }")
        self.btn_filter_all.clicked.connect(lambda: self._set_all_filters_visible(True))
        filter_btn_row.addWidget(self.btn_filter_all)
        self.btn_filter_none = QPushButton("None")
        self.btn_filter_none.setStyleSheet("QPushButton { background-color: #ECEFF1; border: 1px solid #B0BEC5; border-radius: 4px; padding: 2px 8px; }")
        self.btn_filter_none.clicked.connect(lambda: self._set_all_filters_visible(False))
        filter_btn_row.addWidget(self.btn_filter_none)
        self.btn_filter_reset = QPushButton("Reset Colors")
        self.btn_filter_reset.setStyleSheet("QPushButton { background-color: #ECEFF1; border: 1px solid #B0BEC5; border-radius: 4px; padding: 2px 8px; }")
        self.btn_filter_reset.clicked.connect(self._reset_filter_colors)
        filter_btn_row.addWidget(self.btn_filter_reset)
        filter_btn_row.addStretch()
        filter_layout.addLayout(filter_btn_row)

        self.filter_controls_layout = QGridLayout()
        self.filter_controls_layout.setSpacing(6)
        filter_layout.addLayout(self.filter_controls_layout)
        plot_layout.addWidget(filter_box)

        self.plot_canvas = FigureCanvas(Figure(figsize=(8, 4.8)))
        self.plot_ax = self.plot_canvas.figure.add_subplot(111)
        self.plot_canvas.setFocusPolicy(Qt.ClickFocus)
        self.plot_canvas.setMinimumHeight(480)
        self.plot_canvas.setStyleSheet("background-color: #FFFFFF; border: 1px solid #ECEFF1;")
        plot_layout.addWidget(self.plot_canvas)

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

        self.light_layout.addWidget(plot_group)

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

        self.btn_qc_auto = QPushButton("Auto Use")
        self.btn_qc_auto.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 4px 12px; }")
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

        self.qc_table = QTableWidget()
        self.qc_table.setColumnCount(6)
        self.qc_table.setHorizontalHeaderLabels(
            ["Use", "ID", "N(valid)", "RMS", "MAD", "Outlier%"]
        )
        self.qc_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.qc_table.horizontalHeader().setStretchLastSection(True)
        self.qc_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.qc_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.qc_table.itemSelectionChanged.connect(self._on_qc_selection_changed)
        self.qc_table.itemChanged.connect(self._on_qc_table_changed)
        qc_layout.addWidget(self.qc_table)

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
        qc_layout.addLayout(preview_ctrl)

        check_group = QGroupBox("Comparison Preview")
        check_layout = QVBoxLayout(check_group)
        self.check_plot_canvas = FigureCanvas(Figure(figsize=(6, 3)))
        self.check_plot_ax = self.check_plot_canvas.figure.add_subplot(111)
        self.check_plot_canvas.setMinimumHeight(280)
        self.check_plot_canvas.setStyleSheet("background-color: #FFFFFF; border: 1px solid #ECEFF1;")
        check_layout.addWidget(self.check_plot_canvas)
        qc_layout.addWidget(check_group)

        self.qc_layout.addWidget(qc_group)

        self.shortcut_prev = QShortcut(QKeySequence(Qt.Key_Left), self)
        self.shortcut_prev.activated.connect(lambda: self._step_comp(-1))
        self.shortcut_next = QShortcut(QKeySequence(Qt.Key_Right), self)
        self.shortcut_next.activated.connect(lambda: self._step_comp(1))

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

    def log(self, msg: str):
        self.log_text.append(msg)

    def _on_tab_changed(self, idx: int) -> None:
        if self.tab_widget.widget(idx) == self.qc_tab:
            self.run_comp_qc()

    def _on_qc_date_changed(self, _idx: int) -> None:
        self.qc_date_last = self.qc_date_combo.currentText()
        self.run_comp_qc()
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

    def _apply_color_button_style(self, button: QPushButton, color: str, visible: bool) -> None:
        if visible:
            bg = color
            border = "#455A64"
        else:
            bg = "#ECEFF1"
            border = "#B0BEC5"
        button.setStyleSheet(
            f"QPushButton {{ background-color: {bg}; border: 1px solid {border}; border-radius: 3px; min-width: 20px; min-height: 14px; }}"
            "QPushButton:hover { border: 1px solid #263238; }"
        )

    def _clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            elif item.layout() is not None:
                self._clear_layout(item.layout())

    def _ensure_filter_controls(self, filters: list[str]) -> None:
        keys = sorted({self._filter_key_for_ui(f) for f in filters})
        if keys == self._filter_keys and self.filter_control_map:
            return
        self._filter_keys = keys
        self._clear_layout(self.filter_controls_layout)
        self.filter_control_map = {}

        for row, key in enumerate(keys):
            if key not in self.filter_visibility:
                self.filter_visibility[key] = True
            if key not in self.filter_colors:
                self.filter_colors[key] = FILTER_COLORS.get(key, "#ff7f0e")

            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(6)
            row_layout.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

            label = QLabel(f"{key}:")
            label.setStyleSheet("QLabel { color: #37474F; }")

            btn = QPushButton("")
            btn.setToolTip("Left click: show/hide\nRight click: change color")
            btn.setCheckable(True)
            btn.setChecked(self.filter_visibility.get(key, True))
            btn.toggled.connect(lambda checked, k=key: self._on_filter_toggle(k, checked))
            btn.setContextMenuPolicy(Qt.CustomContextMenu)
            btn.customContextMenuRequested.connect(lambda _pos, k=key: self._choose_filter_color(k))
            btn.setFixedSize(20, 14)
            self._apply_color_button_style(btn, self._get_filter_color(key), btn.isChecked())

            row_layout.addWidget(label)
            row_layout.addWidget(btn)
            row_layout.addStretch()

            self.filter_controls_layout.addWidget(row_widget, row, 0, alignment=Qt.AlignLeft)
            self.filter_control_map[key] = btn

    def _on_filter_toggle(self, key: str, visible: bool) -> None:
        self.filter_visibility[key] = bool(visible)
        btn = self.filter_control_map.get(key)
        if btn is not None:
            self._apply_color_button_style(btn, self._get_filter_color(key), visible)
        self.plot_current_comparison()

    def _set_all_filters_visible(self, visible: bool) -> None:
        for key, btn in self.filter_control_map.items():
            btn.blockSignals(True)
            btn.setChecked(visible)
            btn.blockSignals(False)
            self.filter_visibility[key] = bool(visible)
            self._apply_color_button_style(btn, self._get_filter_color(key), visible)
        self.plot_current_comparison()

    def _reset_filter_colors(self) -> None:
        for key, btn in self.filter_control_map.items():
            self.filter_colors[key] = FILTER_COLORS.get(key, "#ff7f0e")
            self._apply_color_button_style(btn, self._get_filter_color(key), btn.isChecked())
        self.plot_current_comparison()

    def _choose_filter_color(self, key: str) -> None:
        current = QColor(self._get_filter_color(key))
        picked = QColorDialog.getColor(current, self, f"Select color for {key}")
        if not picked.isValid():
            return
        self.filter_colors[key] = picked.name()
        if key in self.filter_control_map:
            btn = self.filter_control_map[key]
            self._apply_color_button_style(btn, self._get_filter_color(key), btn.isChecked())
        self.plot_current_comparison()

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

    def add_dataset(self):
        path = QFileDialog.getExistingDirectory(self, "result_dir 선택", str(Path.cwd()))
        if not path:
            return
        self._add_dataset(Path(path))

    def use_current_dataset(self):
        self._add_dataset(self.params.P.result_dir)

    def clear_diff_series_cache(self, clear_headers: bool = False):
        """캐시 클리어 (데이터셋/선택 변경 시 호출)"""
        self._diff_series_cache.clear()
        self._photometry_cache.clear()
        self._photometry_cache_dir = None
        if clear_headers:
            self._header_cache.clear()
            self.log("[CACHE] All caches cleared")
        else:
            self.log("[CACHE] Diff series + photometry cache cleared")

    def _add_dataset(self, path: Path):
        path = Path(path)
        if not path.exists():
            return
        label = path.name
        for _, p in self.datasets:
            if p == path:
                return
        self.datasets.append((label, path))
        self.clear_diff_series_cache()  # 새 데이터셋 추가 시 캐시 클리어
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

    def load_from_selection(self):
        if not self.datasets:
            self.use_current_dataset()
        if not self.datasets:
            return
        target_id, comp_ids = _load_selection_ids(self.datasets[0][1])
        if target_id is not None:
            self.target_edit.setText(str(target_id))
        if comp_ids:
            self.comp_edit.setText(",".join(str(i) for i in comp_ids))
            self.comp_candidate_ids = list(comp_ids)
        self._update_comp_ids_from_input()
        self.plot_current_comparison()

    def _update_comp_ids_from_input(self):
        self.comp_ids_list = _safe_int_list(self.comp_edit.text())
        if not self.comp_candidate_ids:
            self.comp_candidate_ids = list(self.comp_ids_list)
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

        phot_path = step9_dir(result_dir) / f"{fname}_photometry.tsv"
        if not phot_path.exists():
            phot_path = result_dir / f"{fname}_photometry.tsv"
        if not phot_path.exists():
            self._photometry_cache[fname] = None
            return None

        try:
            df = pd.read_csv(phot_path, sep="\t")
        except Exception:
            try:
                df = pd.read_csv(phot_path)
            except Exception:
                df = None

        self._photometry_cache[fname] = df
        return df

    def _build_star_mag_series(self, result_dir: Path, star_id: int, verbose: bool = True) -> pd.DataFrame:
        idx_path = step9_dir(result_dir) / "photometry_index.csv"
        if not idx_path.exists():
            idx_path = result_dir / "photometry_index.csv"
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
                row = df[df["source_id"].astype("int64") == int(comp_source_id)]
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

    def _build_ensemble_series(self, result_dir: Path, target_id: int, comp_ids: list[int], verbose: bool = True) -> pd.DataFrame:
        idx_path = step9_dir(result_dir) / "photometry_index.csv"
        if not idx_path.exists():
            idx_path = result_dir / "photometry_index.csv"
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
            filt_key = _normalize_filter_key(filt_val)
            filters.append(filt_key)
            airmasses.append(am if np.isfinite(am) else np.nan)

            phot_path = step9_dir(result_dir) / f"{fname}_photometry.tsv"
            if not phot_path.exists():
                phot_path = result_dir / f"{fname}_photometry.tsv"
            if not phot_path.exists():
                mags.append(np.nan)
                mag_errs.append(np.nan)
                comp_avgs.append(np.nan)
                comp_errs.append(np.nan)
                diffs.append(np.nan)
                diff_errs.append(np.nan)
                continue
            try:
                df = pd.read_csv(phot_path, sep="\t")
            except Exception:
                df = pd.read_csv(phot_path)

            use_source_id = False
            target_source_id = None
            comp_source_map: dict[int, int] = {}
            if filt_key in filter_selections:
                sel = filter_selections[filt_key]
                target_source_id = sel.get("target_source_id")
                for cid in comp_ids:
                    sid = self._map_comp_source_id(sel, cid)
                    if sid is not None:
                        comp_source_map[int(cid)] = int(sid)
                use_source_id = True

            # Target
            row_t = pd.DataFrame()
            if use_source_id and target_source_id is not None and "source_id" in df.columns:
                row_t = df[df["source_id"].astype("int64") == int(target_source_id)]
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
                    row_c = df[df["source_id"].astype("int64") == int(comp_source_map[cid])]
                if row_c.empty and "ID" in df.columns:
                    row_c = df[df["ID"] == int(cid)]
                if not row_c.empty and np.isfinite(_safe_float(row_c["mag"].values[0])):
                    cmags.append(_safe_float(row_c["mag"].values[0]))
                    cerrs.append(_safe_float(row_c["mag_err"].values[0]) if "mag_err" in row_c.columns else np.nan)
            if cmags:
                n_comp_found += 1
                comp_mean = float(np.nanmean(cmags))
                comp_err = float(np.nanmean(cerrs)) if cerrs else np.nan
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

        if verbose:
            total = len(files)
            self.log(f"[DEBUG] Ensemble series (Target={target_id}) frames={total}")
            self.log(f"[DEBUG] Target found: {n_target_found}/{total}")
            self.log(f"[DEBUG] Comp ensemble available: {n_comp_found}/{total}")

        return pd.DataFrame({
            "file": files,
            "filter": filters,
            "date": dates,
            "JD": tarr,
            "rel_time_hr": rel_time_hr,
            "mag": np.array(mags, float),
            "mag_err": np.array(mag_errs, float),
            "comp_avg": np.array(comp_avgs, float),
            "comp_err": np.array(comp_errs, float),
            "diff_mag_raw": np.array(diffs, float),
            "diff_err": np.array(diff_errs, float),
            "airmass": np.array(airmasses, float),
        })

    def _build_diff_series(self, result_dir: Path, target_id: int, comp_id: int, verbose: bool = True) -> pd.DataFrame:
        # 캐시 키 생성
        cache_key = (str(result_dir), int(target_id), int(comp_id))
        if cache_key in self._diff_series_cache:
            if verbose:
                self.log(f"[CACHE] Using cached diff series for target={target_id}, comp={comp_id}")
            return self._diff_series_cache[cache_key].copy()

        idx_path = step9_dir(result_dir) / "photometry_index.csv"
        if not idx_path.exists():
            idx_path = result_dir / "photometry_index.csv"
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

        times = []
        diffs = []
        filters = []
        airmasses = []

        # headers.csv에서 airmass 컬럼 확인 (FITS 안 읽기 위해)
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

            phot_path = step9_dir(result_dir) / f"{fname}_photometry.tsv"
            if not phot_path.exists():
                phot_path = step9_dir(result_dir) / f"{fname}_photometry.tsv"
                if not phot_path.exists():
                    phot_path = result_dir / f"{fname}_photometry.tsv"
            if not phot_path.exists():
                diffs.append(np.nan)
                n_phot_missing += 1
                continue
            try:
                df = pd.read_csv(phot_path, sep="\t")
            except Exception:
                df = pd.read_csv(phot_path)

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
                row_t = df[df["source_id"].astype("int64") == int(target_source_id)]
            if row_t.empty and "ID" in df.columns:
                row_t = df[df["ID"] == int(target_id)]

            # 비교성 매칭 (source_id 우선, 없으면 ID)
            row_c = pd.DataFrame()
            if use_source_id and comp_source_id is not None and "source_id" in df.columns:
                row_c = df[df["source_id"].astype("int64") == int(comp_source_id)]
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

        result_df = pd.DataFrame({
            "file": files,
            "filter": filters,
            "JD": tarr,
            "rel_time_hr": rel_time_hr,
            "diff_mag": np.array(diffs, float),
            "airmass": np.array(airmasses, float),
        })

        # 캐시에 저장
        self._diff_series_cache[cache_key] = result_df.copy()
        if verbose:
            self.log(f"[CACHE] Stored diff series for target={target_id}, comp={comp_id} ({len(result_df)} rows)")

        return result_df

    def plot_current_comparison(self):
        if not self.datasets:
            self.use_current_dataset()
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
        self._ensure_filter_controls(df["filter"].astype(str).tolist())
        self.plot_ax.clear()

        # X축 선택
        if self.x_axis_mode == "phase":
            # 위상 계산
            jd_col = "JD" if "JD" in df.columns else ("jd" if "jd" in df.columns else None)
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

        y_label = "dmag (raw)"
        for filt, sub in df.groupby("filter"):
            fkey = self._filter_key_for_ui(filt)
            if not self.filter_visibility.get(fkey, True):
                continue
            label = fkey
            c = self._get_filter_color(fkey)

            x = sub[x_column].to_numpy(float)
            y = sub[y_col].to_numpy(float)

            m = np.isfinite(x) & np.isfinite(y)
            if not np.any(m):
                continue
            self.plot_ax.plot(x[m], y[m], marker="o", linestyle="None", color=c, label=label,
                             markersize=3, alpha=0.7)  # 크기 줄이고 투명도 추가

        self.plot_ax.set_xlabel(x_label)
        self.plot_ax.set_ylabel(f"{y_label} (Target - Comparison)")
        if self.x_axis_mode == "phase" and self.phase_cycles > 1:
            self.plot_ax.set_xlim(0, float(self.phase_cycles))
        self.plot_ax.grid(True, alpha=0.3)
        self.plot_ax.invert_yaxis()
        handles, labels = self.plot_ax.get_legend_handles_labels()
        if handles:
            self.plot_ax.legend(loc="best", fontsize=8)
        self.plot_canvas.draw()

    def _compute_comp_qc(
        self,
        result_dir: Path,
        target_id: int,
        comp_ids: list[int],
        date_filter: str | None = None,
        verbose: bool = True,
    ) -> list[dict]:
        rows = []
        active_set = set(self.comp_ids_list)
        for comp_id in comp_ids:
            df = self._build_star_mag_series(result_dir, comp_id, verbose=False)
            if df.empty:
                rows.append({
                    "comp_id": int(comp_id),
                    "n": 0,
                    "rms": np.nan,
                    "mad": np.nan,
                    "outliers": 0,
                    "outlier_frac": np.nan,
                    "use": comp_id in active_set,
                })
                continue
            if date_filter:
                df = df[df["date"].astype(str) == str(date_filter)]
            y = df["mag"].to_numpy(float)
            filters = df["filter"].astype(str).tolist()
            # filter-wise median alignment
            y_adj = np.full_like(y, np.nan, dtype=float)
            for fkey in sorted(set(filters)):
                idx = [i for i, f in enumerate(filters) if f == fkey]
                if not idx:
                    continue
                vals = y[idx]
                if not np.any(np.isfinite(vals)):
                    continue
                med = np.nanmedian(vals)
                if not np.isfinite(med):
                    continue
                y_adj[idx] = vals - med
            m = np.isfinite(y_adj)
            n = int(np.sum(m))
            if n <= 1:
                rows.append({
                    "comp_id": int(comp_id),
                    "n": n,
                    "rms": np.nan,
                    "mad": np.nan,
                    "outliers": 0,
                    "outlier_frac": np.nan,
                    "use": comp_id in active_set,
                })
                continue
            yv = y_adj[m] if np.isfinite(y_adj).any() else y[m]
            med = np.nanmedian(yv)
            mad = np.nanmedian(np.abs(yv - med))
            rms = float(np.nanstd(yv))
            outlier_count = 0
            if np.isfinite(mad) and mad > 0:
                outlier_count = int(np.sum(np.abs(yv - med) > self.qc_sigma * mad))
            outlier_frac = outlier_count / max(n, 1)

            rows.append({
                "comp_id": int(comp_id),
                "n": n,
                "rms": rms,
                "mad": float(mad) if np.isfinite(mad) else np.nan,
                "outliers": outlier_count,
                "outlier_frac": float(outlier_frac),
                "use": comp_id in active_set,
            })

        if verbose:
            self.log(f"[QC] Computed metrics for {len(rows)} comps")
        return rows

    def _save_comp_qc_summary(self, result_dir: Path, rows: list[dict]) -> None:
        if not rows:
            return
        out_dir = step11_dir(result_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        path = out_dir / "comp_qc_summary.csv"
        try:
            df.to_csv(path, index=False)
            self.log(f"[QC] Saved {path.name}")
        except Exception as e:
            self.log(f"[QC] Failed to save summary: {e}")

    def _populate_qc_table(self, rows: list[dict]) -> None:
        self._qc_table_block = True
        self.qc_table.setRowCount(0)
        for row in rows:
            r = self.qc_table.rowCount()
            self.qc_table.insertRow(r)

            item_use = QTableWidgetItem()
            item_use.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable)
            item_use.setCheckState(Qt.Checked if row.get("use", False) else Qt.Unchecked)
            self.qc_table.setItem(r, 0, item_use)

            self.qc_table.setItem(r, 1, QTableWidgetItem(str(row.get("comp_id", ""))))
            self.qc_table.setItem(r, 2, QTableWidgetItem(str(row.get("n", ""))))
            self.qc_table.setItem(r, 3, QTableWidgetItem(_fmt_float(row.get("rms"))))
            self.qc_table.setItem(r, 4, QTableWidgetItem(_fmt_float(row.get("mad"))))
            self.qc_table.setItem(r, 5, QTableWidgetItem(_fmt_percent(row.get("outlier_frac"))))

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
        keys = sorted({str(d) for d in dates if d})
        for key in keys:
            self.qc_date_combo.addItem(key)
        # Prefer last selected date > current selection > first date
        prefer = self.qc_date_last
        if prefer and prefer in keys:
            self.qc_date_combo.setCurrentText(prefer)
        elif current and current in keys:
            self.qc_date_combo.setCurrentText(current)
        elif keys:
            self.qc_date_combo.setCurrentText(keys[0])
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
        if not self.datasets:
            self.use_current_dataset()
        if not self.datasets:
            return
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
        if self.comp_candidate_ids:
            date_probe = self._build_star_mag_series(result_dir, int(self.comp_candidate_ids[0]), verbose=False)
            if not date_probe.empty and "date" in date_probe.columns:
                self._refresh_qc_date_combo(date_probe["date"].astype(str).tolist())
        date_sel = None
        if hasattr(self, "qc_date_combo"):
            current_date = self.qc_date_combo.currentText()
            if current_date and current_date != "All":
                date_sel = current_date
        rows = self._compute_comp_qc(result_dir, target_id, self.comp_candidate_ids, date_filter=date_sel, verbose=True)
        self.qc_rows = rows
        self._populate_qc_table(rows)
        restored = False
        if prev_comp_id is not None:
            restored = self._select_qc_comp_id(prev_comp_id)
        if not restored and self.qc_table.rowCount() > 0:
            self.qc_table.setCurrentCell(0, 1)
            self.qc_table.selectRow(0)
        self._save_comp_qc_summary(result_dir, rows)

    def auto_use_qc(self):
        if not self.qc_rows:
            self.run_comp_qc()
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

    def _plot_comp_preview(self, comp_id: int) -> None:
        if not self.datasets:
            self.use_current_dataset()
        if not self.datasets:
            return
        result_dir = self._get_qc_result_dir()
        df = self._build_star_mag_series(result_dir, int(comp_id), verbose=False)
        self.check_plot_ax.clear()
        if df.empty:
            self.check_plot_canvas.draw()
            return
        if "date" in df.columns:
            self._refresh_qc_date_combo(df["date"].astype(str).tolist())
        date_sel = self.qc_date_combo.currentText() if hasattr(self, "qc_date_combo") else "All"
        if date_sel not in ("All", "", None) and "date" in df.columns:
            df = df[df["date"].astype(str) == str(date_sel)]

        x = df["rel_time_hr"].to_numpy(float)
        y = df["mag"].to_numpy(float)
        filters = df["filter"].astype(str).tolist()
        self._refresh_qc_filter_combo(filters)
        filter_sel = self.qc_filter_combo.currentText() if hasattr(self, "qc_filter_combo") else "All"
        plotted_y = []
        if np.isfinite(x).any() and np.isfinite(y).any():
            for fkey in sorted(set(filters)):
                key_ui = self._filter_key_for_ui(fkey)
                if filter_sel not in ("All", "", None) and key_ui != filter_sel:
                    continue
                idx = [i for i, f in enumerate(filters) if f == fkey]
                if not idx:
                    continue
                xv = x[idx]
                yv = y[idx]
                m = np.isfinite(xv) & np.isfinite(yv)
                if not np.any(m):
                    continue
                color = self._get_filter_color(key_ui)
                self.check_plot_ax.plot(xv[m], yv[m], marker="o", linestyle="None",
                                        color=color, markersize=3, alpha=0.8, label=key_ui)
                med = np.nanmedian(yv[m])
                if np.isfinite(med):
                    self.check_plot_ax.axhline(med, color=color, linestyle="--", linewidth=1, alpha=0.7)
                plotted_y.extend(yv[m].tolist())
        self.check_plot_ax.set_title(f"Comp ID {comp_id}")
        self.check_plot_ax.set_xlabel("Time (hr)")
        self.check_plot_ax.set_ylabel("mag")
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
        self.check_plot_canvas.draw()

    def build_light_curve(self):
        if not self.datasets:
            self.use_current_dataset()
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
        if self.comp_candidate_ids:
            qc_rows = self._compute_comp_qc(self.datasets[0][1], target_id, self.comp_candidate_ids, verbose=False)
            self._save_comp_qc_summary(Path(self.datasets[0][1]), qc_rows)

        P = self.params.P
        color_index_by_filter = _normalize_color_index_by_filter(
            getattr(P, "lightcurve_color_index_by_filter", {})
        )

        combined_raw = []
        for label, result_dir in self.datasets:
            result_dir = Path(result_dir)
            raw_df = self._build_ensemble_series(result_dir, target_id, active_comp_ids, verbose=True)
            if raw_df.empty:
                self.log(f"[{label}] Raw light curve empty")
                continue
            raw_df = raw_df.copy()
            raw_df["dataset"] = label
            raw_df["diff_mag"] = raw_df["diff_mag_raw"]

            # Auto-compute color indices from raw_df (already has mag, comp_avg per filter)
            available_filters = {_normalize_filter_key(f) for f in raw_df["filter"].astype(str).unique() if f}
            color_pair = _auto_detect_color_index(available_filters)

            if color_pair and "mag" in raw_df.columns and "comp_avg" in raw_df.columns:
                f_blue, f_red = color_pair
                self.log(f"[{label}] Auto-detected color index: {f_blue}-{f_red}")

                # Compute median mag per filter from raw_df
                target_mags_by_filter = {}
                comp_mags_by_filter = {}
                for fkey, sub in raw_df.groupby(raw_df["filter"].astype(str).map(_normalize_filter_key)):
                    t_vals = sub["mag"].dropna()
                    c_vals = sub["comp_avg"].dropna()
                    if len(t_vals) > 0:
                        target_mags_by_filter[fkey] = float(np.nanmedian(t_vals))
                    if len(c_vals) > 0:
                        comp_mags_by_filter[fkey] = float(np.nanmedian(c_vals))

                # Calculate color index
                t_blue = target_mags_by_filter.get(f_blue, np.nan)
                t_red = target_mags_by_filter.get(f_red, np.nan)
                c_blue = comp_mags_by_filter.get(f_blue, np.nan)
                c_red = comp_mags_by_filter.get(f_red, np.nan)

                target_color = t_blue - t_red if np.isfinite(t_blue) and np.isfinite(t_red) else np.nan
                comp_color = c_blue - c_red if np.isfinite(c_blue) and np.isfinite(c_red) else np.nan

                if np.isfinite(target_color):
                    self.log(f"[{label}] Target color ({f_blue}-{f_red}): {target_color:.3f}")
                if np.isfinite(comp_color):
                    self.log(f"[{label}] Comp color ({f_blue}-{f_red}): {comp_color:.3f}")
                if np.isfinite(target_color) and np.isfinite(comp_color):
                    delta_c = target_color - comp_color
                    self.log(f"[{label}] ΔC (target - comp): {delta_c:.3f}")

                raw_df["color_index"] = target_color
                raw_df["color_index_ref"] = comp_color
                raw_df["color_pair"] = f"{f_blue}-{f_red}"
            else:
                self.log(f"[{label}] No color index available (need 2+ filters: g/r, B/V, etc.)")
                raw_df["color_index"] = np.nan
                raw_df["color_index_ref"] = np.nan

            out_dir = step11_dir(result_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"lightcurve_ID{target_id}_raw.csv"
            raw_df.to_csv(out_path, index=False)
            self.log(f"[{label}] Saved {out_path.name}")
            combined_raw.append(raw_df)

        # combined outputs
        if combined_raw:
            base_dir = step11_dir(self.params.P.result_dir)
            base_dir.mkdir(parents=True, exist_ok=True)
            comb = pd.concat(combined_raw, ignore_index=True)
            comb = comb.sort_values("JD")
            comb_path = base_dir / f"lightcurve_combined_ID{target_id}_raw.csv"
            comb.to_csv(comb_path, index=False)
            self.log(f"[combined] Saved {comb_path.name}")

        # comp selection 저장
        if combined_raw:
            sel_path = step11_dir(self.params.P.result_dir) / "comp_selection.json"
            payload = {
                "target_id": target_id,
                "comp_candidate_ids": [int(x) for x in self.comp_candidate_ids],
                "comp_active_ids": [int(x) for x in active_comp_ids],
            }
            try:
                sel_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                self.log(f"[combined] Saved {sel_path.name}")
            except Exception as e:
                self.log(f"[WARN] Failed to save comp_selection.json: {e}")

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
        super().keyPressEvent(event)

    def validate_step(self) -> bool:
        return True

    def save_state(self):
        state_data = {
            "datasets": [str(p) for _, p in self.datasets],
            "target_id": self.target_edit.text().strip(),
            "comp_ids": self.comp_edit.text().strip(),
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
        }
        self.project_state.store_step_data("light_curve", state_data)

    def restore_state(self):
        state_data = self.project_state.get_step_data("light_curve")
        if state_data:
            for path in state_data.get("datasets", []):
                self._add_dataset(Path(path))
            self.target_edit.setText(state_data.get("target_id", ""))
            self.comp_edit.setText(state_data.get("comp_ids", ""))
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
