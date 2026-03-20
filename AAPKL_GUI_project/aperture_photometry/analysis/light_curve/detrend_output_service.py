"""Reusable output helpers for Step 11 detrend results."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ...utils.common_helpers import normalize_filter_key
from ...utils.step_paths import step11_current_meta_path


def annotate_step11_output(df: pd.DataFrame, mode_tag: str, formula: str) -> pd.DataFrame:
    out = df.copy()
    if "JD" in out.columns and "jd" in out.columns:
        out = out.drop(columns=["jd"])
    out = out.drop(columns=["diff_mag"], errors="ignore")
    out["correction_mode"] = mode_tag
    out["correction_formula"] = formula
    return out


def write_step11_current_meta(
    result_dir: Path,
    target_id: int,
    mode_tag: str,
    formula: str,
    lc_path: Path,
    params_path: Path | None = None,
    summary_path: Path | None = None,
    plot_path: Path | None = None,
    extra_files: list[Path] | None = None,
) -> Path:
    meta = {
        "target_id": int(target_id),
        "mode": str(mode_tag),
        "formula": str(formula),
        "saved_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "lightcurve_file": lc_path.name,
        "params_file": params_path.name if params_path is not None else "",
        "summary_file": summary_path.name if summary_path is not None else "",
        "plot_file": plot_path.name if plot_path is not None else "",
        "extra_files": [p.name for p in (extra_files or [])],
    }
    meta_path = step11_current_meta_path(result_dir, target_id)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta_path


def build_global_summary_text(
    target_id: int,
    params_df: pd.DataFrame,
    corrected_df: pd.DataFrame,
    formula: str,
) -> str:
    lines = [
        "=" * 60,
        "GLOBAL ENSEMBLE SUMMARY",
        "=" * 60,
        "",
        "Mode: GLOBAL (method C)",
        "Formula: mag_inst = M_i + Z_t",
        "Target curve: m_corr = mag_inst_target - Z_t",
        f"Differential output: {formula}",
        "",
        f"Target ID: {target_id}",
        f"Frames: {len(params_df)}",
        f"Points: {len(corrected_df)}",
    ]
    return "\n".join(lines)


def build_detrend_summary_report_text(
    mode: str,
    use_global_k2: bool,
    delta_c_map: dict[str, float],
    params_df: pd.DataFrame,
    df: pd.DataFrame,
    sigma_clip: bool,
    clip_sigma: float,
    clip_iters: int,
    phase_period: float,
    phase_t0: float,
) -> str:
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("DETREND & NIGHT MERGE - SUMMARY REPORT")
    lines.append("=" * 60)
    lines.append("")

    lines.append(f"[Mode] {mode.upper()}")
    if mode == "offset":
        lines.append("  Formula: Δm_corr = Δm_raw - ZP₀")
        lines.append("  (Nightly zero-point offset only)")
    else:
        lines.append("  Formula: Δm_corr = Δm_raw - ZP₀ - k''·ΔC·X")
        lines.append("  (Color-dependent extinction correction)")
        if use_global_k2:
            lines.append("  k'' fitting: GLOBAL (single k'' for all nights)")
        else:
            lines.append("  k'' fitting: NIGHTLY (separate k'' per night)")
    lines.append("")

    lines.append("[Data Summary]")
    lines.append(f"  Total points: {len(df)}")
    if "date" in df.columns:
        dates = sorted(df["date"].astype(str).unique())
        lines.append(f"  Dates: {len(dates)} nights")
        for d in dates:
            n = len(df[df["date"].astype(str) == d])
            lines.append(f"    - {d}: {n} points")
    if "filter" in df.columns:
        filters = sorted({normalize_filter_key(f) for f in df["filter"].astype(str) if str(f).strip()})
        lines.append(f"  Filters: {', '.join(filters) or 'N/A'}")
    lines.append("")

    if delta_c_map:
        lines.append("[Color Index (ΔC = Target - Comp)]")
        for fkey, dc in sorted(delta_c_map.items()):
            lines.append(f"  {fkey or 'all'}: ΔC = {dc:+.4f} mag")
        lines.append("")

    if not params_df.empty:
        lines.append("[Fit Parameters by Date/Filter]")
        lines.append("-" * 60)
        if mode == "offset":
            lines.append(f"{'Date':<12} {'Filter':<6} {'N':>5} {'ZP₀':>10} {'±σ':>8} {'RMS_before':>10} {'RMS_after':>10}")
        else:
            lines.append(f"{'Date':<12} {'Filter':<6} {'N':>5} {'ZP₀':>10} {'k\"':>10} {'RMS_before':>10} {'RMS_after':>10}")
        lines.append("-" * 60)
        for _, row in params_df.iterrows():
            date = str(row.get("date", ""))[:12]
            filt = str(row.get("filter", "") or "all")[:6]
            n = int(row.get("n_used", 0))
            zp = row.get("zp_offset", np.nan)
            zp_err = row.get("zp_offset_err", np.nan)
            slope = row.get("ext_slope", np.nan)
            rms_b = row.get("rms_before", np.nan)
            rms_a = row.get("rms_after", np.nan)
            if mode == "offset":
                lines.append(f"{date:<12} {filt:<6} {n:>5} {zp:>10.5f} {zp_err:>8.5f} {rms_b:>10.5f} {rms_a:>10.5f}")
            else:
                lines.append(f"{date:<12} {filt:<6} {n:>5} {zp:>10.5f} {slope:>10.5f} {rms_b:>10.5f} {rms_a:>10.5f}")
        lines.append("")

    lines.append("[Correction Statistics]")
    raw_vals = pd.to_numeric(df.get("diff_mag_raw", pd.Series()), errors="coerce").to_numpy(float)
    corr_vals = pd.to_numeric(df.get("diff_mag_corr", pd.Series()), errors="coerce").to_numpy(float)
    raw_vals = raw_vals[np.isfinite(raw_vals)]
    corr_vals = corr_vals[np.isfinite(corr_vals)]

    if raw_vals.size > 0:
        lines.append(
            f"  Raw:  mean={np.mean(raw_vals):.5f}, std={np.std(raw_vals):.5f}, "
            f"min={np.min(raw_vals):.5f}, max={np.max(raw_vals):.5f}"
        )
    if corr_vals.size > 0:
        lines.append(
            f"  Corr: mean={np.mean(corr_vals):.5f}, std={np.std(corr_vals):.5f}, "
            f"min={np.min(corr_vals):.5f}, max={np.max(corr_vals):.5f}"
        )
    if raw_vals.size > 0 and corr_vals.size > 0:
        rms_improve = (1 - np.std(corr_vals) / np.std(raw_vals)) * 100
        lines.append(f"  RMS improvement: {rms_improve:+.1f}%")
    lines.append("")

    if "residual" in df.columns:
        resid = pd.to_numeric(df["residual"], errors="coerce").to_numpy(float)
        resid = resid[np.isfinite(resid)]
        if resid.size > 0:
            lines.append("[Residual Statistics (Δm_corr - median)]")
            lines.append(f"  mean={np.mean(resid):.6f}, std={np.std(resid):.6f}")
            lines.append(
                f"  median={np.median(resid):.6f}, "
                f"MAD={np.median(np.abs(resid - np.median(resid))):.6f}"
            )
            pct = np.percentile(resid, [5, 25, 50, 75, 95])
            lines.append(
                f"  percentiles: 5%={pct[0]:.5f}, 25%={pct[1]:.5f}, 50%={pct[2]:.5f}, "
                f"75%={pct[3]:.5f}, 95%={pct[4]:.5f}"
            )
            lines.append("")

    lines.append("[Sigma Clipping]")
    lines.append(f"  Enabled: {sigma_clip}")
    if sigma_clip:
        lines.append(f"  Threshold: {clip_sigma}σ")
        lines.append(f"  Iterations: {clip_iters}")
    lines.append("")

    if phase_period > 0:
        lines.append("[Phase Folding]")
        lines.append(f"  Period: {phase_period:.6f} days")
        lines.append(f"  T₀: {phase_t0:.6f} JD")
        lines.append("")

    lines.append("=" * 60)
    lines.append("Generated by AAPKI Step 11: Detrend & Night Merge")
    lines.append("=" * 60)
    return "\n".join(lines)
