"""I/O helpers for Step 12 period analysis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ...utils.step_paths import step12_period_dir


_CORR_MODE_LABELS = {
    "global": "Global ensemble",
    "color": "Color-dependent",
    "offset": "Nightly offset",
    "raw": "Raw (no correction)",
}


def detect_corr_mode(filename: str) -> tuple[str, str]:
    import re

    match = re.search(r"lightcurve_.*?_(global|color|offset|raw)\b", filename, re.IGNORECASE)
    if match:
        key = match.group(1).lower()
        return key, _CORR_MODE_LABELS.get(key, key)
    return "unknown", "Unknown"


def detect_corr_mode_from_df(df: pd.DataFrame, filename: str) -> tuple[str, str]:
    if "correction_mode" in df.columns:
        vals = df["correction_mode"].dropna().astype(str).str.strip().str.lower()
        if not vals.empty:
            key = vals.iloc[0]
            if key:
                return key, _CORR_MODE_LABELS.get(key, key)
    return detect_corr_mode(filename)


def load_period_lightcurve_csv(lc_file: Path, flt: str, target_id: int) -> dict:
    df = pd.read_csv(lc_file)

    time_col = None
    for col in ["BJD_TDB", "BJD", "bjd", "JD", "jd", "HJD", "hjd", "time", "rel_time_hr"]:
        if col in df.columns:
            time_col = col
            break
    if time_col is None:
        raise ValueError("No time column (JD/HJD/BJD) found.")

    if "filter" in df.columns:
        df_flt = df[df["filter"].astype(str).str.strip().str.lower() == flt.lower()].copy()
        if df_flt.empty:
            df_flt = df
    else:
        df_flt = df

    df_target = df_flt
    if "ID" in df_target.columns:
        df_id = df_target[df_target["ID"] == target_id].copy()
        if not df_id.empty:
            df_target = df_id

    mag_raw_col = None
    mag_corr_col = None
    mag_err_col = None
    for col in ["diff_mag_raw", "mag_raw", "raw_mag", "inst_mag", "mag"]:
        if col in df_target.columns:
            mag_raw_col = col
            break
    for col in ["diff_mag_corr", "diff_mag", "mag_corr", "corr_mag", "calibrated_mag"]:
        if col in df_target.columns:
            mag_corr_col = col
            break
    if mag_raw_col is None and mag_corr_col is not None:
        mag_raw_col = mag_corr_col
        mag_corr_col = None
    for col in ["diff_err", "diff_err_corr", "mag_err", "err", "sigma", "diff_mag_err", "comp_err"]:
        if col in df_target.columns:
            mag_err_col = col
            break
    if mag_raw_col is None:
        raise ValueError(f"No magnitude column found. Available columns: {list(df_target.columns)}")

    corr_mode_key, corr_mode_label = detect_corr_mode_from_df(df_target, lc_file.name)
    payload = {
        "time": df_target[time_col].to_numpy(float),
        "mag_raw": df_target[mag_raw_col].to_numpy(float),
        "mag_corr": df_target[mag_corr_col].to_numpy(float) if mag_corr_col else None,
        "mag_err": df_target[mag_err_col].to_numpy(float) if mag_err_col else None,
        "filter": flt,
        "target_id": int(target_id),
        "source_file": str(lc_file),
        "col_raw": mag_raw_col,
        "col_corr": mag_corr_col,
        "col_err": mag_err_col,
        "col_time": time_col,
        "corr_mode": corr_mode_key,
        "corr_mode_label": corr_mode_label,
        "n_rows": int(len(df)),
        "columns": list(df.columns),
    }
    return payload


def save_period_analysis_outputs(
    result_dir: Path,
    lc_data: dict,
    results: dict,
    min_period: float,
    max_period: float,
) -> Path:
    out_dir = step12_period_dir(result_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    flt = lc_data.get("filter", "unknown")
    target_id = lc_data.get("target_id", 0)
    summary = {
        "filter": flt,
        "target_id": target_id,
        "source_file": lc_data.get("source_file", ""),
        "corr_mode": lc_data.get("corr_mode", "unknown"),
        "corr_mode_label": lc_data.get("corr_mode_label", "Unknown"),
        "min_period": float(min_period),
        "max_period": float(max_period),
        "results": {},
    }

    for key, data in results.items():
        if "error" in data:
            summary["results"][key] = {"error": data["error"]}
        else:
            entry = {
                "method": data.get("method", ""),
                "best_period": float(data["best_period"]),
                "best_power": float(data["best_power"]),
                "fap": float(data.get("fap", np.nan)) if np.isfinite(data.get("fap", np.nan)) else None,
                "n_points": int(data.get("n_points", 0)),
                "top_periods": [float(p) for p in data.get("top_periods", [])],
            }
            if "best_theta" in data:
                entry["best_theta"] = float(data["best_theta"])
            summary["results"][key] = entry

    summary_path = out_dir / f"period_analysis_{flt}_ID{target_id}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for key, data in results.items():
        if "error" in data:
            continue
        if "frequency" in data:
            df = pd.DataFrame({
                "frequency": data["frequency"],
                "period": 1.0 / data["frequency"],
                "power": data["power"],
            })
        elif "trial_periods" in data:
            df = pd.DataFrame({
                "period": data["trial_periods"],
                "power": data["power"],
            })
            if "theta" in data:
                df["theta"] = data["theta"]
        else:
            continue
        csv_path = out_dir / f"periodogram_{flt}_{key}_ID{target_id}.csv"
        df.to_csv(csv_path, index=False)

    return summary_path
