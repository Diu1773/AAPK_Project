"""Reusable save/output helpers for Step 10 raw light curves."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from ...utils.common_helpers import normalize_filter_key
from ...utils.step_paths import step10_dir


def _auto_detect_color_index(available_filters: set[str]) -> tuple[str, str] | None:
    filters_lower = {normalize_filter_key(f) for f in available_filters}
    if {"g", "r"} <= filters_lower:
        return ("g", "r")
    if {"g", "i"} <= filters_lower:
        return ("g", "i")
    if {"r", "i"} <= filters_lower:
        return ("r", "i")
    if {"b", "v"} <= filters_lower:
        return ("b", "v")
    if {"v", "r"} <= filters_lower:
        return ("v", "r")
    return None


def annotate_raw_lightcurve(
    raw_df: pd.DataFrame,
    dataset_label: str,
    logger: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    out = raw_df.copy()
    out["dataset"] = dataset_label
    out["diff_mag"] = out["diff_mag_raw"]

    available_filters = {
        normalize_filter_key(f)
        for f in out["filter"].astype(str).unique()
        if f
    }
    color_pair = _auto_detect_color_index(available_filters)

    if not color_pair or "mag" not in out.columns or "comp_avg" not in out.columns:
        if logger is not None:
            logger(f"[{dataset_label}] No color index available (need 2+ filters: g/r, B/V, etc.)")
        out["color_index"] = np.nan
        out["color_index_ref"] = np.nan
        return out

    f_blue, f_red = color_pair
    if logger is not None:
        logger(f"[{dataset_label}] Auto-detected color index: {f_blue}-{f_red}")

    target_mags_by_filter: dict[str, float] = {}
    comp_mags_by_filter: dict[str, float] = {}
    for fkey, sub in out.groupby(out["filter"].astype(str).map(normalize_filter_key)):
        t_vals = sub["mag"].dropna()
        c_vals = sub["comp_avg"].dropna()
        if len(t_vals) > 0:
            target_mags_by_filter[fkey] = float(np.nanmedian(t_vals))
        if len(c_vals) > 0:
            comp_mags_by_filter[fkey] = float(np.nanmedian(c_vals))

    t_blue = target_mags_by_filter.get(f_blue, np.nan)
    t_red = target_mags_by_filter.get(f_red, np.nan)
    c_blue = comp_mags_by_filter.get(f_blue, np.nan)
    c_red = comp_mags_by_filter.get(f_red, np.nan)

    target_color = t_blue - t_red if np.isfinite(t_blue) and np.isfinite(t_red) else np.nan
    comp_color = c_blue - c_red if np.isfinite(c_blue) and np.isfinite(c_red) else np.nan

    if logger is not None and np.isfinite(target_color):
        logger(f"[{dataset_label}] Target color ({f_blue}-{f_red}): {target_color:.3f}")
    if logger is not None and np.isfinite(comp_color):
        logger(f"[{dataset_label}] Comp color ({f_blue}-{f_red}): {comp_color:.3f}")
    if logger is not None and np.isfinite(target_color) and np.isfinite(comp_color):
        logger(f"[{dataset_label}] ΔC (target - comp): {target_color - comp_color:.3f}")

    out["color_index"] = target_color
    out["color_index_ref"] = comp_color
    out["color_pair"] = f"{f_blue}-{f_red}"
    return out


def save_dataset_raw_outputs(
    result_dir: Path,
    target_id: int,
    raw_df: pd.DataFrame,
    check_ids_by_filter: dict[str, int] | None = None,
    check_df: pd.DataFrame | None = None,
    logger: Callable[[str], None] | None = None,
) -> None:
    out_dir = step10_dir(result_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"lightcurve_ID{target_id}_raw.csv"
    raw_df.to_csv(out_path, index=False)
    if logger is not None:
        logger(f"[{result_dir.name}] Saved {out_path.name}")

    if check_df is None or check_df.empty:
        return

    check_df.to_csv(out_dir / "lightcurve_check_combined_raw.csv", index=False)
    check_ids_by_filter = check_ids_by_filter or {}
    for filt_key, check_id in sorted(check_ids_by_filter.items()):
        part = check_df[
            check_df["filter"].astype(str).map(normalize_filter_key) == filt_key
        ].copy()
        if part.empty:
            continue
        part_path = out_dir / f"lightcurve_check_{filt_key}_ID{int(check_id)}_raw.csv"
        part.to_csv(part_path, index=False)
        if logger is not None:
            logger(f"  Check star saved → {part_path.name}")

    if "check_id" not in check_df.columns:
        return

    for check_id in sorted({
        int(x) for x in pd.to_numeric(check_df["check_id"], errors="coerce").dropna().astype(int).tolist()
    }):
        part = check_df[pd.to_numeric(check_df["check_id"], errors="coerce") == int(check_id)].copy()
        if part.empty:
            continue
        legacy_path = out_dir / f"lightcurve_check_ID{int(check_id)}_raw.csv"
        part.to_csv(legacy_path, index=False)


def save_combined_raw_outputs(
    base_result_dir: Path,
    target_id: int,
    combined_raw: list[pd.DataFrame],
    single_dataset_mode: bool,
    comp_candidate_ids: list[int],
    active_comp_ids: list[int],
    logger: Callable[[str], None] | None = None,
) -> None:
    if not combined_raw:
        return

    base_dir = step10_dir(base_result_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    comb = pd.concat(combined_raw, ignore_index=True).sort_values("JD")
    comb_path = base_dir / f"lightcurve_combined_ID{target_id}_raw.csv"
    if single_dataset_mode:
        if comb_path.exists():
            try:
                comb_path.unlink()
                if logger is not None:
                    logger(f"[combined] Removed stale {comb_path.name} (single dataset)")
            except Exception as e:
                if logger is not None:
                    logger(f"[combined] Failed to remove stale {comb_path.name}: {e}")
        if logger is not None:
            logger("[combined] Skipped combined raw output (single dataset)")
    else:
        comb.to_csv(comb_path, index=False)
        if logger is not None:
            logger(f"[combined] Saved {comb_path.name}")

    sel_path = base_dir / "comp_selection.json"
    payload = {
        "target_id": int(target_id),
        "comp_candidate_ids": [int(x) for x in comp_candidate_ids],
        "comp_active_ids": [int(x) for x in active_comp_ids],
    }
    try:
        sel_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if logger is not None:
            logger(f"[combined] Saved {sel_path.name}")
    except Exception as e:
        if logger is not None:
            logger(f"[WARN] Failed to save comp_selection.json: {e}")
