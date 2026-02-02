"""
Helpers for frame-level QC filtering (frame_quality.csv).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple, Dict, List

import pandas as pd

from .step_paths import step5_dir, legacy_step7_wcs_dir


def resolve_frame_quality_path(result_dir: Path) -> Optional[Path]:
    """Resolve the best available frame_quality.csv path."""
    candidates = [
        step5_dir(result_dir) / "frame_quality.csv",
        legacy_step7_wcs_dir(result_dir) / "frame_quality.csv",
        Path(result_dir) / "frame_quality.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _passed_mask(series: pd.Series) -> pd.Series:
    """Normalize a 'passed' column to a boolean mask."""
    if series.dtype == bool:
        return series
    values = series.astype(str).str.strip().str.lower()
    return values.isin({"true", "1", "yes", "y", "on"})


def filter_files_by_qc(
    result_dir: Path,
    files: Iterable[str],
    *,
    require_qc: bool = True,
) -> Tuple[List[str], Dict[str, object]]:
    """Filter files using frame_quality.csv (QC pass list).

    Returns (filtered_files, info). If QC is not applied, filtered_files == input list.
    """
    file_list = list(files)
    info: Dict[str, object] = {
        "applied": False,
        "reason": "disabled" if not require_qc else "missing",
        "path": None,
        "total": len(file_list),
        "kept": len(file_list),
        "dropped": 0,
    }
    if not require_qc:
        return file_list, info

    qpath = resolve_frame_quality_path(result_dir)
    if not qpath:
        return file_list, info

    info["path"] = str(qpath)
    try:
        dfq = pd.read_csv(qpath)
    except Exception as exc:
        info["reason"] = f"read_error:{exc}"
        return file_list, info

    if "file" not in dfq.columns:
        info["reason"] = "missing_file_column"
        return file_list, info

    if "passed" in dfq.columns:
        mask = _passed_mask(dfq["passed"])
        good = set(dfq.loc[mask, "file"].astype(str).tolist())
    elif "exclude_reason" in dfq.columns:
        reasons = dfq["exclude_reason"].fillna("").astype(str)
        good = set(dfq.loc[reasons.str.strip() == "", "file"].astype(str).tolist())
    else:
        info["reason"] = "missing_passed_column"
        return file_list, info

    filtered = [f for f in file_list if f in good]
    info["applied"] = True
    info["reason"] = "ok"
    info["kept"] = len(filtered)
    info["dropped"] = len(file_list) - len(filtered)
    return filtered, info
