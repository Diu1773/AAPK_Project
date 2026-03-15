"""
Helpers for frame-level QC filtering (frame_quality.csv).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple, Dict, List

import numpy as np
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


def resolve_frame_wcs_qc_path(result_dir: Path) -> Optional[Path]:
    """Resolve the best available frame_wcs_qc.csv path."""
    candidates = [
        step5_dir(result_dir) / "frame_wcs_qc.csv",
        legacy_step7_wcs_dir(result_dir) / "frame_wcs_qc.csv",
        Path(result_dir) / "frame_wcs_qc.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def filter_files_by_wcs_qc(
    result_dir: Path,
    files: Iterable[str],
    *,
    require_qc: bool = True,
    require_wcs_ok: bool = True,
    min_match_rate: float = 0.20,
    min_match_n: int = 20,
    max_rms_px: float = 3.5,
    min_inlier_rate: float = 0.50,
    max_p99_px: float = 5.0,
) -> Tuple[List[str], Dict[str, object]]:
    """Filter files using frame_wcs_qc.csv produced by Step 5.

    Returns (filtered_files, info). If WCS QC is not applied, filtered_files == input list.
    """
    file_list = list(files)
    info: Dict[str, object] = {
        "applied": False,
        "reason": "disabled" if not require_qc else "missing",
        "path": None,
        "total": len(file_list),
        "kept": len(file_list),
        "dropped": 0,
        "checks": [],
    }
    if not require_qc:
        return file_list, info

    qpath = resolve_frame_wcs_qc_path(result_dir)
    if not qpath:
        return file_list, info

    info["path"] = str(qpath)
    try:
        dfq = pd.read_csv(qpath)
    except Exception as exc:
        info["reason"] = f"read_error:{exc}"
        return file_list, info

    if "file" not in dfq.columns:
        if "fname" in dfq.columns:
            dfq = dfq.rename(columns={"fname": "file"})
        else:
            info["reason"] = "missing_file_column"
            return file_list, info

    mask = pd.Series(True, index=dfq.index, dtype=bool)
    checks_used: List[str] = []

    if require_wcs_ok and "wcs_ok" in dfq.columns:
        mask &= _passed_mask(dfq["wcs_ok"])
        checks_used.append("wcs_ok")

    if min_match_n > 0 and "n_match" in dfq.columns:
        n_match = pd.to_numeric(dfq["n_match"], errors="coerce")
        mask &= n_match.notna() & (n_match >= int(min_match_n))
        checks_used.append(f"n_match>={int(min_match_n)}")

    if np.isfinite(min_match_rate) and min_match_rate > 0:
        rate_cols = [c for c in ("match_rate", "match_rate_cat", "match_rate_eff") if c in dfq.columns]
        if rate_cols:
            rate_eff = pd.concat(
                [pd.to_numeric(dfq[c], errors="coerce") for c in rate_cols],
                axis=1,
            ).max(axis=1, skipna=True)
            mask &= rate_eff.notna() & (rate_eff >= float(min_match_rate))
            checks_used.append(f"match_rate_eff>={float(min_match_rate):.3f}")

    if np.isfinite(max_rms_px) and max_rms_px > 0 and "rms_px" in dfq.columns:
        rms = pd.to_numeric(dfq["rms_px"], errors="coerce")
        mask &= rms.notna() & (rms <= float(max_rms_px))
        checks_used.append(f"rms_px<={float(max_rms_px):.3f}")

    if np.isfinite(min_inlier_rate) and min_inlier_rate > 0 and "inlier_rate" in dfq.columns:
        inlier = pd.to_numeric(dfq["inlier_rate"], errors="coerce")
        mask &= inlier.notna() & (inlier >= float(min_inlier_rate))
        checks_used.append(f"inlier_rate>={float(min_inlier_rate):.3f}")

    p99_col = None
    if "resid_p99_px" in dfq.columns:
        p99_col = "resid_p99_px"
    elif "resid_peak_px" in dfq.columns:
        p99_col = "resid_peak_px"
    if np.isfinite(max_p99_px) and max_p99_px > 0 and p99_col is not None:
        p99 = pd.to_numeric(dfq[p99_col], errors="coerce")
        mask &= p99.notna() & (p99 <= float(max_p99_px))
        checks_used.append(f"{p99_col}<={float(max_p99_px):.3f}")

    if not checks_used:
        if "wcs_qc_pass" in dfq.columns:
            mask = _passed_mask(dfq["wcs_qc_pass"])
            checks_used.append("wcs_qc_pass")
        else:
            info["reason"] = "missing_metric_columns"
            return file_list, info

    good = set(dfq.loc[mask, "file"].astype(str).tolist())
    filtered = [f for f in file_list if f in good]
    info["applied"] = True
    info["reason"] = "ok"
    info["kept"] = len(filtered)
    info["dropped"] = len(file_list) - len(filtered)
    info["checks"] = checks_used
    return filtered, info
