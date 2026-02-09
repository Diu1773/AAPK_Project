"""
QA Report Window: Publication-Quality Photometry Validation

This module provides comprehensive QA/QC validation for aperture photometry,
generating publication-ready statistics, plots, and reports.

Key validations:
1. SNR-Error Model Verification (RMS vs predicted error)
2. Centroid Quality Assessment (Δr distribution)
3. Frame Quality Metrics (FWHM, sky_sigma, n_goodmag)
4. Aperture Correction Validation (before/after comparison)
5. Background Estimation Quality
6. Saturation/Edge Flag Summary
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import pandas as pd

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# Suppress matplotlib layout warnings
warnings.filterwarnings("ignore", message=".*tight_layout.*")

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGroupBox, QMessageBox, QTextEdit, QTabWidget, QTableWidget,
    QTableWidgetItem, QHeaderView, QProgressBar, QSplitter, QFileDialog,
    QCheckBox, QSpinBox, QDoubleSpinBox, QFormLayout, QScrollArea, QComboBox,
    QDialog, QDialogButtonBox, QGridLayout, QFrame
)
from PyQt5.QtGui import QFont, QColor

try:  # Python 3.11+
    import tomllib  # type: ignore
except Exception:
    import tomli as tomllib  # type: ignore
try:
    import tomli_w  # type: ignore
except Exception:
    tomli_w = None

from ...utils.step_paths import step9_dir


class QAReportWorker(QThread):
    """Worker thread for QA report generation"""
    progress = pyqtSignal(int, int, str)
    log = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, result_dir: Path, params: Dict = None):
        super().__init__()
        self.result_dir = Path(result_dir)
        self._stop_requested = False

        # QA Parameters with defaults
        self.params = params or {}
        self.min_n_frames = self.params.get("min_n_frames", 3)
        self.min_snr = self.params.get("min_snr", 0.0)
        self.max_chi2_nu = self.params.get("max_chi2_nu", 100.0)
        self.max_delta_r = self.params.get("max_delta_r", 10.0)
        self.exclude_saturated = self.params.get("exclude_saturated", False)
        self.enabled_filters = self.params.get("enabled_filters", None)  # None = all
        self.selected_frames = self.params.get("selected_frames", None)  # None = all frames
        self.skip_frame_flagging = self.params.get("skip_frame_flagging", False)  # Skip re-flagging when frames selected

    def stop(self):
        self._stop_requested = True

    def _log(self, msg: str):
        self.log.emit(msg)

    def _apply_frame_zp(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply frame ZP correction to magnitudes if available."""
        zp_path = self.result_dir / "frame_zeropoint.csv"
        if not zp_path.exists():
            return df
        try:
            zp_df = pd.read_csv(zp_path)
            if not {"file", "filter", "zp_frame"}.issubset(zp_df.columns):
                return df
            zp_df["filter"] = zp_df["filter"].astype(str).str.strip().str.lower()
            out = df.copy()
            out["FILTER"] = out["FILTER"].astype(str).str.strip().str.lower()
            out = out.merge(
                zp_df[["file", "filter", "zp_frame", "zp_scatter", "n_ref"]],
                left_on=["frame", "FILTER"],
                right_on=["file", "filter"],
                how="left"
            )
            out["zp_frame"] = out["zp_frame"].fillna(0.0)
            out["mag_zp"] = out["mag"] + out["zp_frame"]
            out["zp_sigma"] = np.nan
            nref = pd.to_numeric(out.get("n_ref"), errors="coerce")
            zpsc = pd.to_numeric(out.get("zp_scatter"), errors="coerce")
            m = np.isfinite(nref) & (nref > 0) & np.isfinite(zpsc)
            out.loc[m, "zp_sigma"] = zpsc[m] / np.sqrt(nref[m])
            return out
        except Exception:
            return df

    def run(self):
        try:
            results = {}

            # 1. Load all photometry TSVs
            self._log("Loading photometry data...")
            tsvs = sorted(self.result_dir.glob("*_photometry.tsv"))
            if not tsvs:
                raise FileNotFoundError("No *_photometry.tsv files found")

            all_rows = []
            for i, tsv in enumerate(tsvs):
                if self._stop_requested:
                    return
                df = pd.read_csv(tsv, sep="\t")
                df["frame"] = tsv.stem.replace("_photometry", "")
                all_rows.append(df)
                self.progress.emit(i + 1, len(tsvs), f"Loading {tsv.name}")

            big = pd.concat(all_rows, ignore_index=True)
            self._log(f"Loaded {len(tsvs)} frames, {len(big)} measurements")

            # Store original data for frame quality (BEFORE any filtering)
            big_original = big.copy()

            # Apply filter exclusions (preserve original case)
            if self.enabled_filters is not None and len(self.enabled_filters) > 0:
                if "FILTER" in big.columns:
                    big["FILTER_clean"] = big["FILTER"].astype(str).str.strip()
                    before = len(big)
                    big = big[big["FILTER_clean"].isin(self.enabled_filters)].copy()
                    big = big.drop(columns=["FILTER_clean"])
                    excluded = before - len(big)
                    self._log(f"Filter exclusion: {before} → {len(big)} ({excluded} excluded, enabled: {self.enabled_filters})")
                    # Also filter original for frame quality
                    big_original = big_original[big_original["FILTER"].astype(str).str.strip().isin(self.enabled_filters)].copy()

            # Apply frame selection (from Frames tab checkboxes) - BEFORE other filters
            if self.selected_frames is not None and len(self.selected_frames) > 0:
                if "frame" in big.columns:
                    before = len(big)
                    big = big[big["frame"].isin(self.selected_frames)].copy()
                    excluded = before - len(big)
                    self._log(f"Frame selection: {before} → {len(big)} ({excluded} excluded, {len(self.selected_frames)} frames selected)")
                    big_original = big_original[big_original["frame"].isin(self.selected_frames)].copy()

            # 4. Frame Quality - Compute BEFORE SNR/saturation filters to show real frame quality
            self._log("Computing frame quality metrics (on original data)...")
            results["frame_quality"] = self._compute_frame_quality(big_original)

            # Now apply quality filters for error model and centroid analysis
            # Apply SNR cut
            if self.min_snr > 0 and "snr" in big.columns:
                before = len(big)
                big = big[big["snr"] >= self.min_snr].copy()
                self._log(f"SNR >= {self.min_snr}: {before} → {len(big)}")

            # Apply saturation exclusion
            if self.exclude_saturated and "is_saturated" in big.columns:
                before = len(big)
                big = big[big["is_saturated"] != True].copy()
                self._log(f"Exclude saturated: {before} → {len(big)}")

            # 2. Error Model Validation (on filtered data)
            self._log("Computing error model validation (raw)...")
            results["error_model_raw"] = self._compute_error_model(big)
            results["error_model"] = results["error_model_raw"]
            if "overall" in results["error_model_raw"]:
                em = results["error_model_raw"]["overall"]
                self._log(
                    "Error model source=raw | "
                    f"RMS/σ_pred={em.get('rms_over_pred_global_med', float('nan')):.3f}, "
                    f"χ²/ν={em.get('chi2_nu_global_med', float('nan')):.3f}"
                )

            big_zp = self._apply_frame_zp(big)
            if "mag_zp" in big_zp.columns:
                self._log("Computing error model validation (frame ZP)...")
                results["error_model_zp"] = self._compute_error_model(big_zp, mag_col="mag_zp")
                if "overall" in results["error_model_zp"]:
                    em = results["error_model_zp"]["overall"]
                    self._log(
                        "Error model source=zp | "
                        f"RMS/σ_pred={em.get('rms_over_pred_global_med', float('nan')):.3f}, "
                        f"χ²/ν={em.get('chi2_nu_global_med', float('nan')):.3f}"
                    )

            # 3. Centroid Quality (on filtered data)
            self._log("Computing centroid quality...")
            results["centroid"] = self._compute_centroid_qa(big)

            # 5. Background Quality
            self._log("Computing background quality...")
            results["background"] = self._compute_background_qa(big)

            # 6. Saturation Summary
            self._log("Computing saturation summary...")
            results["saturation"] = self._compute_saturation_summary(big)

            # 7. Aperture Correction Validation
            self._log("Computing aperture correction validation...")
            results["apcorr"] = self._compute_apcorr_validation(big)

            # 8. Publication Summary Table
            self._log("Generating publication summary...")
            results["publication"] = self._generate_publication_summary(results)

            # Save results
            self._save_results(results)

            self._log("QA Report complete!")
            self.finished.emit(results)

        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n{traceback.format_exc()}")

    def _compute_error_model(self, df: pd.DataFrame, mag_col: str = "mag") -> Dict:
        """Compute RMS vs predicted error validation"""
        # Filter valid measurements
        valid = df[
            np.isfinite(df[mag_col]) &
            np.isfinite(df["mag_err"]) &
            (df["snr"] > 0) &
            (df["mag_err"] > 0)
        ].copy()

        if "delta_r" in valid.columns and np.isfinite(self.max_delta_r):
            before = len(valid)
            valid = valid[valid["delta_r"] <= self.max_delta_r].copy()
            removed = before - len(valid)
            if removed > 0:
                self._log(f"Centroid filter (Δr <= {self.max_delta_r:.2f}px): removed {removed} rows")

        if "recenter_capped" in valid.columns:
            before = len(valid)
            valid = valid[~valid["recenter_capped"].fillna(False)].copy()
            removed = before - len(valid)
            if removed > 0:
                self._log(f"Centroid capped filter: removed {removed} rows")

        if len(valid) < 10:
            return {"error": "Insufficient valid data"}

        err_col = "mag_err"
        if mag_col == "mag_zp" and "zp_sigma" in valid.columns:
            err_col = "mag_err_eff"
            zp_sig = pd.to_numeric(valid["zp_sigma"], errors="coerce").to_numpy(float)
            mag_err = valid["mag_err"].to_numpy(float)
            valid[err_col] = np.sqrt(mag_err**2 + np.where(np.isfinite(zp_sig), zp_sig**2, 0.0))

        # Group by ID and FILTER
        grp = valid.groupby(["ID", "FILTER"]).agg(
            mag_mean=(mag_col, "mean"),
            mag_med=(mag_col, "median"),
            mag_std=(mag_col, "std"),
            mag_mad=(mag_col, lambda x: 1.4826 * np.median(np.abs(x - np.median(x)))),
            mag_err_med=(err_col, "median"),
            snr_med=("snr", "median"),
            n_obs=(mag_col, "count")
        ).reset_index()

        # Filter by minimum observations
        grp = grp[grp["n_obs"] >= self.min_n_frames].copy()

        if len(grp) == 0:
            return {"error": f"No stars with >= {self.min_n_frames} observations"}

        # Compute chi2_nu early for filtering
        grp["rms_over_pred_raw"] = grp["mag_std"] / grp["mag_err_med"]
        grp["chi2_nu_raw"] = (grp["mag_std"]**2 / grp["mag_err_med"]**2)

        self._log(f"Before chi2 cut: {len(grp)} stars, chi2_nu max={grp['chi2_nu_raw'].max():.1f}, threshold={self.max_chi2_nu}")

        # Apply chi2_nu outlier cut (always apply if threshold is set)
        if self.max_chi2_nu < 999:
            before = len(grp)
            grp = grp[grp["chi2_nu_raw"] <= self.max_chi2_nu].copy()
            removed = before - len(grp)
            self._log(f"χ²/ν <= {self.max_chi2_nu}: {before} → {len(grp)} stars ({removed} outliers removed)")

        # Compute validation metrics
        grp["rms_over_pred"] = grp["mag_std"] / grp["mag_err_med"]
        grp["mad_over_pred"] = grp["mag_mad"] / grp["mag_err_med"]
        grp["chi2_per_obs"] = (grp["mag_std"]**2 / grp["mag_err_med"]**2) * (grp["n_obs"] - 1)
        grp["chi2_nu"] = grp["chi2_per_obs"] / np.maximum(grp["n_obs"] - 1, 1)

        # Per-filter summary
        summary_by_filter = grp.groupby("FILTER").agg(
            n_stars=("ID", "nunique"),
            n_obs_total=("n_obs", "sum"),
            rms_over_pred_med=("rms_over_pred", "median"),
            rms_over_pred_std=("rms_over_pred", "std"),
            mad_over_pred_med=("mad_over_pred", "median"),
            chi2_nu_med=("chi2_nu", "median"),
            chi2_nu_std=("chi2_nu", "std"),
            snr_p10=("snr_med", lambda x: np.nanpercentile(x, 10)),
            snr_p90=("snr_med", lambda x: np.nanpercentile(x, 90)),
        ).reset_index()

        # Estimate systematic floor from bright stars (SNR > 30)
        # The floor is the RMS that cannot be reduced even with perfect photon statistics
        bright_mask = grp["snr_med"] > 30
        sigma_sys_mag = np.nan
        sigma_sys_note = ""
        if bright_mask.sum() >= 5:
            bright_rms = grp.loc[bright_mask, "mag_std"].dropna()
            if len(bright_rms) >= 5:
                # Use median of bright star RMS as systematic floor
                sigma_sys_mag = float(np.nanmedian(bright_rms))
                sigma_sys_note = f"Estimated from {bright_mask.sum()} bright stars (SNR>30)"
                self._log(f"Systematic floor estimate: σ_sys = {sigma_sys_mag:.4f} mag")

        # Overall summary
        overall = {
            "n_stars": int(grp["ID"].nunique()),
            "n_observations": int(grp["n_obs"].sum()),
            "rms_over_pred_global_med": float(np.nanmedian(grp["rms_over_pred"])),
            "rms_over_pred_global_std": float(np.nanstd(grp["rms_over_pred"])),
            "chi2_nu_global_med": float(np.nanmedian(grp["chi2_nu"])),
            "chi2_nu_global_std": float(np.nanstd(grp["chi2_nu"])),
            "sigma_sys_mag": sigma_sys_mag,
            "sigma_sys_note": sigma_sys_note,
        }

        return {
            "by_star": grp,
            "by_filter": summary_by_filter,
            "overall": overall
        }

    def _compute_centroid_qa(self, df: pd.DataFrame) -> Dict:
        """Compute centroid shift statistics"""
        required = {"x_init", "y_init", "xcenter", "ycenter"}
        if not required.issubset(df.columns):
            return {"error": "Missing centroid columns (x_init, y_init, xcenter, ycenter)"}

        valid = df[
            np.isfinite(df["x_init"]) & np.isfinite(df["y_init"]) &
            np.isfinite(df["xcenter"]) & np.isfinite(df["ycenter"])
        ].copy()

        valid["delta_x"] = valid["xcenter"] - valid["x_init"]
        valid["delta_y"] = valid["ycenter"] - valid["y_init"]
        valid["delta_r"] = np.sqrt(valid["delta_x"]**2 + valid["delta_y"]**2)

        dr = valid["delta_r"].dropna()

        summary = {
            "n_measurements": int(len(dr)),
            "delta_r_median": float(np.median(dr)),
            "delta_r_mean": float(np.mean(dr)),
            "delta_r_std": float(np.std(dr)),
            "delta_r_p90": float(np.percentile(dr, 90)),
            "delta_r_p95": float(np.percentile(dr, 95)),
            "delta_r_p99": float(np.percentile(dr, 99)),
            "delta_r_max": float(np.max(dr)),
            "n_outliers_gt1px": int(np.sum(dr > 1.0)),
            "n_outliers_gt2px": int(np.sum(dr > 2.0)),
        }

        data_cols = ["ID", "frame", "delta_x", "delta_y", "delta_r"]
        if "FILTER" in valid.columns:
            data_cols.append("FILTER")

        return {
            "data": valid[data_cols],
            "summary": summary
        }

    def _compute_frame_quality(self, df: pd.DataFrame) -> Dict:
        """Compute per-frame quality metrics"""
        # Per-frame statistics including filter
        agg_dict = {
            "ID": "count",
            "mag": [lambda x: np.isfinite(x).sum(), lambda x: (~np.isfinite(x)).sum()],
            "snr": ["median", "mean"],
            "mag_err": "median",
        }

        # Include FILTER if available
        if "FILTER" in df.columns:
            frame_stats = df.groupby("frame").agg(
                n_targets=("ID", "count"),
                n_goodmag=("mag", lambda x: np.isfinite(x).sum()),
                n_nan=("mag", lambda x: (~np.isfinite(x)).sum()),
                snr_med=("snr", "median"),
                snr_mean=("snr", "mean"),
                mag_err_med=("mag_err", "median"),
                filter=("FILTER", "first"),  # Get filter from first row
            ).reset_index()
        else:
            frame_stats = df.groupby("frame").agg(
                n_targets=("ID", "count"),
                n_goodmag=("mag", lambda x: np.isfinite(x).sum()),
                n_nan=("mag", lambda x: (~np.isfinite(x)).sum()),
                snr_med=("snr", "median"),
                snr_mean=("snr", "mean"),
                mag_err_med=("mag_err", "median"),
            ).reset_index()
            frame_stats["filter"] = ""

        frame_stats["nan_fraction"] = frame_stats["n_nan"] / frame_stats["n_targets"]
        frame_stats["goodmag_fraction"] = frame_stats["n_goodmag"] / frame_stats["n_targets"]

        # Load aperture info if available
        ap_path = self.result_dir / "aperture_by_frame.csv"
        if ap_path.exists():
            ap_df = pd.read_csv(ap_path)
            if "file" in ap_df.columns and "fwhm_used" in ap_df.columns:
                ap_df = ap_df.rename(columns={"file": "frame"})
                # Handle frame name matching
                frame_stats["frame_base"] = frame_stats["frame"].str.replace("_photometry", "")
                ap_df["frame_base"] = ap_df["frame"].astype(str)
                frame_stats = frame_stats.merge(
                    ap_df[["frame_base", "fwhm_used", "r_ap", "r_in", "r_out"]],
                    on="frame_base", how="left"
                )

        # Identify outlier frames (skip if frames already selected by user)
        if self.skip_frame_flagging:
            # Don't re-flag - all selected frames are considered good
            frame_stats["fwhm_flag"] = False
            frame_stats["goodmag_flag"] = False
            frame_stats["snr_flag"] = False
            frame_stats["qa_flag"] = False
        else:
            mode = str(self.params.get("frame_flag_mode", "absolute")).lower()
            if mode == "percentile":
                if "fwhm_used" in frame_stats.columns:
                    fwhm_p90 = np.nanpercentile(frame_stats["fwhm_used"], 90)
                    frame_stats["fwhm_flag"] = frame_stats["fwhm_used"] > fwhm_p90
                goodmag_p10 = np.nanpercentile(frame_stats["goodmag_fraction"], 10)
                frame_stats["goodmag_flag"] = frame_stats["goodmag_fraction"] < goodmag_p10
                snr_p10 = np.nanpercentile(frame_stats["snr_med"], 10)
                frame_stats["snr_flag"] = frame_stats["snr_med"] < snr_p10
            else:
                snr_min = float(self.params.get("frame_snr_min", 10.0))
                goodmag_min = float(self.params.get("frame_goodmag_min", 0.9))
                frame_stats["snr_flag"] = frame_stats["snr_med"] < snr_min
                frame_stats["goodmag_flag"] = frame_stats["goodmag_fraction"] < goodmag_min
                frame_stats["fwhm_flag"] = False
                if "fwhm_used" in frame_stats.columns:
                    fwhm_mode = str(self.params.get("frame_fwhm_mode", "scale")).lower()
                    if fwhm_mode == "absolute":
                        fwhm_abs = float(self.params.get("frame_fwhm_abs", 8.0))
                        frame_stats["fwhm_flag"] = frame_stats["fwhm_used"] > fwhm_abs
                    else:
                        fwhm_scale = float(self.params.get("frame_fwhm_scale", 1.5))
                        fwhm_med = np.nanmedian(frame_stats["fwhm_used"])
                        frame_stats["fwhm_flag"] = frame_stats["fwhm_used"] > (fwhm_med * fwhm_scale)

            # Combined flag
            frame_stats["qa_flag"] = (
                frame_stats.get("fwhm_flag", False) |
                frame_stats["goodmag_flag"] |
                frame_stats["snr_flag"]
            )

        summary = {
            "n_frames": int(len(frame_stats)),
            "n_flagged": int(frame_stats["qa_flag"].sum()),
            "goodmag_fraction_med": float(np.nanmedian(frame_stats["goodmag_fraction"])),
            "snr_med_med": float(np.nanmedian(frame_stats["snr_med"])),
        }

        if "fwhm_used" in frame_stats.columns:
            summary["fwhm_med"] = float(np.nanmedian(frame_stats["fwhm_used"]))
            summary["fwhm_p90"] = float(np.nanpercentile(frame_stats["fwhm_used"], 90))

        return {
            "data": frame_stats,
            "summary": summary
        }

    def _compute_background_qa(self, df: pd.DataFrame) -> Dict:
        """Compute background estimation quality"""
        required = {"bkg_median_adu", "bkg_std_adu", "n_sky"}
        if not required.issubset(df.columns):
            return {"error": "Missing background columns"}

        valid = df[
            np.isfinite(df["bkg_median_adu"]) &
            np.isfinite(df["bkg_std_adu"]) &
            (df["n_sky"] > 0)
        ].copy()

        # Per-frame background statistics
        agg_dict = dict(
            bkg_median_med=("bkg_median_adu", "median"),
            bkg_median_std=("bkg_median_adu", "std"),
            bkg_std_med=("bkg_std_adu", "median"),
            bkg_std_std=("bkg_std_adu", "std"),
            n_sky_med=("n_sky", "median"),
            n_sky_min=("n_sky", "min"),
        )
        if "FILTER" in valid.columns:
            agg_dict["filter"] = ("FILTER", "first")
        frame_bkg = valid.groupby("frame").agg(**agg_dict).reset_index()

        by_filter = None
        if "FILTER" in valid.columns:
            by_filter = valid.groupby("FILTER").agg(
                n_measurements=("bkg_median_adu", "count"),
                n_frames=("frame", "nunique"),
                bkg_median_global=("bkg_median_adu", "median"),
                bkg_std_global=("bkg_std_adu", "median"),
                n_sky_global_med=("n_sky", "median"),
                n_sky_global_min=("n_sky", "min"),
            ).reset_index()

        # Flag low n_sky: frames with n_sky below absolute threshold (50px)
        # Use min() to ensure flag only triggers for genuinely low n_sky values
        # Fixed: was max() which incorrectly flagged most frames when n_sky_p10 >> 50
        n_sky_threshold = 50  # Minimum acceptable sky pixel count
        frame_bkg["n_sky_min"] = pd.to_numeric(frame_bkg["n_sky_min"], errors="coerce")
        frame_bkg["n_sky_flag"] = frame_bkg["n_sky_min"] < n_sky_threshold

        # Flag high bkg_std
        bkg_std_p90 = np.nanpercentile(frame_bkg["bkg_std_med"], 90)
        frame_bkg["bkg_std_flag"] = frame_bkg["bkg_std_med"] > bkg_std_p90

        summary = {
            "n_measurements": int(len(valid)),
            "n_frames": int(len(frame_bkg)),
            "bkg_median_global": float(np.nanmedian(valid["bkg_median_adu"])),
            "bkg_std_global": float(np.nanmedian(valid["bkg_std_adu"])),
            "n_sky_global_med": float(np.nanmedian(valid["n_sky"])),
            "n_sky_global_min": int(valid["n_sky"].min()),
            "n_sky_threshold": int(n_sky_threshold),
            "n_frames_low_nsky": int(frame_bkg["n_sky_flag"].sum()),
            "n_frames_high_bkgstd": int(frame_bkg["bkg_std_flag"].sum()),
        }

        return {
            "by_frame": frame_bkg,
            "summary": summary,
            "by_filter": by_filter
        }

    def _compute_saturation_summary(self, df: pd.DataFrame) -> Dict:
        """Compute saturation statistics"""
        # Check if is_saturated column exists
        if "is_saturated" not in df.columns:
            return {
                "error": "is_saturated column not found in photometry output",
                "recommendation": "Add is_saturated to TSV output in step9_forced_photometry.py"
            }

        n_total = len(df)
        n_sat = int(df["is_saturated"].sum())

        # Per-star saturation
        star_sat = df.groupby("ID").agg(
            n_obs=("mag", "count"),
            n_sat=("is_saturated", "sum")
        ).reset_index()
        star_sat["sat_fraction"] = star_sat["n_sat"] / star_sat["n_obs"]

        n_stars_any_sat = int((star_sat["n_sat"] > 0).sum())
        n_stars_all_sat = int((star_sat["sat_fraction"] == 1.0).sum())

        summary = {
            "n_total": n_total,
            "n_saturated": n_sat,
            "saturation_fraction": float(n_sat / n_total) if n_total > 0 else 0.0,
            "n_stars_any_sat": n_stars_any_sat,
            "n_stars_all_sat": n_stars_all_sat,
        }

        return {
            "by_star": star_sat,
            "summary": summary
        }

    def _compute_apcorr_validation(self, df: pd.DataFrame) -> Dict:
        """Validate aperture correction effectiveness"""
        if "apcorr_applied" not in df.columns:
            return {"error": "apcorr_applied column not found"}

        valid = df[np.isfinite(df["mag"]) & np.isfinite(df["mag_err"])].copy()

        # Separate applied vs not applied
        applied = valid[valid["apcorr_applied"] == True]
        not_applied = valid[valid["apcorr_applied"] == False]

        if len(applied) < 10 or len(not_applied) < 10:
            # All frames have same apcorr status - compare to expected
            if len(applied) > 10:
                # Check scatter per star
                star_rms = applied.groupby(["ID", "FILTER"])["mag"].std().dropna()
                return {
                    "mode": "applied_only",
                    "n_applied": int(len(applied)),
                    "n_not_applied": int(len(not_applied)),
                    "star_rms_med": float(star_rms.median()),
                    "star_rms_p90": float(np.percentile(star_rms, 90)),
                }
            else:
                star_rms = not_applied.groupby(["ID", "FILTER"])["mag"].std().dropna()
                return {
                    "mode": "not_applied_only",
                    "n_applied": int(len(applied)),
                    "n_not_applied": int(len(not_applied)),
                    "star_rms_med": float(star_rms.median()),
                    "star_rms_p90": float(np.percentile(star_rms, 90)),
                }

        # Compare RMS for same stars in both modes
        rms_applied = applied.groupby(["ID", "FILTER"])["mag"].std().reset_index(name="rms_applied")
        rms_not = not_applied.groupby(["ID", "FILTER"])["mag"].std().reset_index(name="rms_not_applied")

        merged = rms_applied.merge(rms_not, on=["ID", "FILTER"], how="inner")
        merged["rms_improvement"] = merged["rms_not_applied"] - merged["rms_applied"]
        merged["rms_ratio"] = merged["rms_applied"] / merged["rms_not_applied"]

        summary = {
            "mode": "comparison",
            "n_applied": int(len(applied)),
            "n_not_applied": int(len(not_applied)),
            "n_stars_compared": int(len(merged)),
            "rms_applied_med": float(merged["rms_applied"].median()),
            "rms_not_applied_med": float(merged["rms_not_applied"].median()),
            "rms_improvement_med": float(merged["rms_improvement"].median()),
            "rms_ratio_med": float(merged["rms_ratio"].median()),
            "n_improved": int((merged["rms_improvement"] > 0).sum()),
            "improvement_fraction": float((merged["rms_improvement"] > 0).mean()),
        }

        return {
            "comparison": merged,
            "summary": summary
        }

    def _select_error_model_block(self, results: Dict) -> Dict | None:
        source = str(self.params.get("error_model_source", "raw")).lower()
        if source == "raw":
            return results.get("error_model_raw")
        if source == "zp":
            block = results.get("error_model_zp")
            return block if block is not None else results.get("error_model_raw")
        return results.get("error_model_raw")

    def _generate_publication_summary(self, results: Dict) -> Dict:
        """Generate publication-ready summary table"""
        rows = []

        # Error model by filter
        em_src = self._select_error_model_block(results)

        if em_src is not None and "by_filter" in em_src:
            for _, row in em_src["by_filter"].iterrows():
                rows.append({
                    "Filter": row["FILTER"],
                    "N_stars": int(row["n_stars"]),
                    "N_obs": int(row["n_obs_total"]),
                    "RMS/σ_pred": f"{row['rms_over_pred_med']:.2f}±{row['rms_over_pred_std']:.2f}",
                    "χ²/ν": f"{row['chi2_nu_med']:.2f}±{row['chi2_nu_std']:.2f}",
                    "SNR_range": f"{row['snr_p10']:.0f}-{row['snr_p90']:.0f}",
                })

        # Centroid quality
        if "centroid" in results and "summary" in results["centroid"]:
            cs = results["centroid"]["summary"]
            rows.append({
                "Metric": "Centroid Shift",
                "Value": f"Δr_95% = {cs['delta_r_p95']:.2f} px",
                "N_outliers": f"{cs['n_outliers_gt1px']} (>1px)",
            })

        # Frame quality
        if "frame_quality" in results and "summary" in results["frame_quality"]:
            fq = results["frame_quality"]["summary"]
            rows.append({
                "Metric": "Frame Quality",
                "Value": f"{fq['n_frames']} frames, {fq['n_flagged']} flagged",
                "SNR_med": f"{fq['snr_med_med']:.1f}",
            })

        return {
            "table": pd.DataFrame(rows) if rows else pd.DataFrame(),
            "latex": self._to_latex_table(rows) if rows else ""
        }

    def _to_latex_table(self, rows: List[Dict]) -> str:
        """Convert summary to LaTeX table format"""
        if not rows:
            return ""

        lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{Photometry QA Summary}",
            r"\label{tab:qa_summary}",
            r"\begin{tabular}{lcccc}",
            r"\hline",
        ]

        # Check if it's error model data
        if "Filter" in rows[0]:
            lines.append(r"Filter & $N_\mathrm{stars}$ & $N_\mathrm{obs}$ & RMS/$\sigma_\mathrm{pred}$ & $\chi^2/\nu$ \\")
            lines.append(r"\hline")
            for row in rows:
                if "Filter" in row:
                    lines.append(f"{row['Filter']} & {row['N_stars']} & {row['N_obs']} & {row['RMS/σ_pred']} & {row['χ²/ν']} \\\\")

        lines.extend([
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
        ])

        return "\n".join(lines)

    def _save_results(self, results: Dict):
        """Save QA results to CSV files"""
        qa_dir = self.result_dir / "qa_report"
        qa_dir.mkdir(exist_ok=True)

        # Error model
        for key, suffix in (("error_model_raw", "raw"), ("error_model_zp", "zp")):
            if key in results:
                em = results[key]
                if "by_star" in em and isinstance(em["by_star"], pd.DataFrame):
                    em["by_star"].to_csv(qa_dir / f"qa_error_model_by_star_{suffix}.csv", index=False, na_rep="NaN")
                if "by_filter" in em and isinstance(em["by_filter"], pd.DataFrame):
                    em["by_filter"].to_csv(qa_dir / f"qa_error_model_by_filter_{suffix}.csv", index=False, na_rep="NaN")
                if "overall" in em:
                    with open(qa_dir / f"qa_error_model_summary_{suffix}.json", "w") as f:
                        json.dump(em["overall"], f, indent=2)
        if "error_model_raw" in results and "overall" in results["error_model_raw"]:
            with open(qa_dir / "qa_error_model_summary.json", "w") as f:
                json.dump(results["error_model_raw"]["overall"], f, indent=2)
            em_raw = results["error_model_raw"]
            if "by_star" in em_raw and isinstance(em_raw["by_star"], pd.DataFrame):
                em_raw["by_star"].to_csv(qa_dir / "qa_error_model_by_star.csv", index=False, na_rep="NaN")
            if "by_filter" in em_raw and isinstance(em_raw["by_filter"], pd.DataFrame):
                em_raw["by_filter"].to_csv(qa_dir / "qa_error_model_by_filter.csv", index=False, na_rep="NaN")

        # Centroid
        if "centroid" in results and "data" in results["centroid"]:
            if isinstance(results["centroid"]["data"], pd.DataFrame):
                results["centroid"]["data"].to_csv(qa_dir / "qa_centroid_shift.csv", index=False, na_rep="NaN")
            if "summary" in results["centroid"]:
                with open(qa_dir / "qa_centroid_summary.json", "w") as f:
                    json.dump(results["centroid"]["summary"], f, indent=2)

        # Frame quality
        if "frame_quality" in results and "data" in results["frame_quality"]:
            if isinstance(results["frame_quality"]["data"], pd.DataFrame):
                results["frame_quality"]["data"].to_csv(qa_dir / "qa_frame_quality.csv", index=False, na_rep="NaN")

        # Background
        if "background" in results and "by_frame" in results["background"]:
            if isinstance(results["background"]["by_frame"], pd.DataFrame):
                results["background"]["by_frame"].to_csv(qa_dir / "qa_background.csv", index=False, na_rep="NaN")

        self._log(f"QA results saved to {qa_dir}")


class QAReportWindow(QMainWindow):
    """
    Publication-Quality QA Report Window

    Provides comprehensive validation of photometry pipeline results
    for inclusion in scientific publications.
    """

    def __init__(self, params, result_dir: Path, parent=None):
        super().__init__(parent)
        self.params = params
        base_dir = Path(result_dir)
        step_dir = step9_dir(base_dir)
        if not (base_dir / "photometry_index.csv").exists() and (step_dir / "photometry_index.csv").exists():
            self.result_dir = step_dir
        else:
            self.result_dir = base_dir
        self.worker = None
        self.results = {}

        # QA Parameters (default: Publication Ready)
        self.qa_params = {
            "min_n_frames": 3,
            "min_snr": 5.0,
            "max_chi2_nu": 5.0,  # Strict default for publication
            "max_delta_r": 2.0,
            "exclude_saturated": True,
            "enabled_filters": None,  # None = all filters
            "error_model_source": "raw",  # raw/zp for publication verdict
            "frame_flag_mode": "absolute",  # absolute/percentile
            "frame_snr_min": 10.0,
            "frame_goodmag_min": 0.9,
            "frame_fwhm_mode": "scale",  # scale/absolute
            "frame_fwhm_scale": 1.5,
            "frame_fwhm_abs": 8.0,
        }

        # Detected filters from data
        self.available_filters = []
        self._scan_available_filters()
        self._load_qa_params()

        self.setWindowTitle("QA Report - Publication Quality Validation")
        self.setMinimumSize(1200, 800)

        self.setup_ui()

    def closeEvent(self, event):
        self._save_qa_params()
        super().closeEvent(event)

    def _scan_available_filters(self):
        """Scan photometry index or TSVs to detect available filters (preserve original case)"""
        self.available_filters = []
        try:
            filters_found = set()
            index_path = self.result_dir / "photometry_index.csv"
            if index_path.exists():
                try:
                    idx = pd.read_csv(index_path)
                    if "filter" in idx.columns:
                        filters_found.update(idx["filter"].astype(str).str.strip().tolist())
                    elif "FILTER" in idx.columns:
                        filters_found.update(idx["FILTER"].astype(str).str.strip().tolist())
                except Exception:
                    pass
            if not filters_found:
                tsvs = sorted(self.result_dir.glob("*_photometry.tsv"))
                for tsv in tsvs[:20]:  # Sample first 20 files
                    try:
                        df = pd.read_csv(tsv, sep="\t", nrows=1)
                        if "FILTER" in df.columns:
                            filt = str(df["FILTER"].iloc[0]).strip()
                            filters_found.add(filt)
                    except Exception:
                        pass
            self.available_filters = sorted(filters_found)
        except Exception as e:
            print(f"Error scanning filters: {e}")

    def _select_error_model_block(self, results: Dict) -> Dict | None:
        source = str(self.qa_params.get("error_model_source", "raw")).lower()
        if source == "raw":
            return results.get("error_model_raw")
        if source == "zp":
            block = results.get("error_model_zp")
            if block is None:
                self.log("Error model source=zp not available; using raw.")
                return results.get("error_model_raw")
            return block
        return results.get("error_model_raw")

    def _qa_param_path(self) -> Path:
        return Path(getattr(self.params, "param_file", "parameters.toml"))

    def _load_qa_params(self):
        path = self._qa_param_path()
        if not path.exists():
            return
        try:
            with path.open("rb") as f:
                data = tomllib.load(f)
            tools = data.get("tools", {}) if isinstance(data, dict) else {}
            cfg = tools.get("qa_report", {}) if isinstance(tools, dict) else {}
            if not cfg:
                cfg = data.get("qa_report", {})
        except Exception:
            return

        if isinstance(cfg, dict):
            enabled_filters_all = cfg.get("enabled_filters_all", True)
            enabled_filters = cfg.get("enabled_filters", [])
            if enabled_filters_all or not enabled_filters:
                self.qa_params["enabled_filters"] = None
            elif isinstance(enabled_filters, list):
                self.qa_params["enabled_filters"] = [str(f) for f in enabled_filters]
            source = str(cfg.get("error_model_source", self.qa_params["error_model_source"])).lower()
            self.qa_params["error_model_source"] = source if source in ("raw", "zp") else "raw"
            for key in [
                "min_n_frames", "min_snr", "max_chi2_nu", "max_delta_r",
                "exclude_saturated", "frame_flag_mode", "frame_snr_min",
                "frame_goodmag_min", "frame_fwhm_mode", "frame_fwhm_scale",
                "frame_fwhm_abs",
            ]:
                if key in cfg:
                    self.qa_params[key] = cfg[key]

        if self.available_filters and self.qa_params["enabled_filters"]:
            allowed = set(self.available_filters)
            self.qa_params["enabled_filters"] = [
                f for f in self.qa_params["enabled_filters"] if f in allowed
            ] or None

    def _save_qa_params(self):
        path = self._qa_param_path()
        if tomli_w is None:
            self.log("tomli_w not available; QA params not saved.")
            return
        data = {}
        if path.exists():
            try:
                with path.open("rb") as f:
                    data = tomllib.load(f)
            except Exception:
                data = {}
        cfg = {
            "min_n_frames": int(self.qa_params.get("min_n_frames", 3)),
            "min_snr": float(self.qa_params.get("min_snr", 5.0)),
            "max_chi2_nu": float(self.qa_params.get("max_chi2_nu", 5.0)),
            "max_delta_r": float(self.qa_params.get("max_delta_r", 2.0)),
            "exclude_saturated": bool(self.qa_params.get("exclude_saturated", True)),
            "error_model_source": str(self.qa_params.get("error_model_source", "raw")).lower(),
            "frame_flag_mode": str(self.qa_params.get("frame_flag_mode", "absolute")),
            "frame_snr_min": float(self.qa_params.get("frame_snr_min", 10.0)),
            "frame_goodmag_min": float(self.qa_params.get("frame_goodmag_min", 0.9)),
            "frame_fwhm_mode": str(self.qa_params.get("frame_fwhm_mode", "scale")),
            "frame_fwhm_scale": float(self.qa_params.get("frame_fwhm_scale", 1.5)),
            "frame_fwhm_abs": float(self.qa_params.get("frame_fwhm_abs", 8.0)),
        }
        enabled = self.qa_params.get("enabled_filters")
        if enabled is None:
            cfg["enabled_filters_all"] = True
            cfg["enabled_filters"] = []
        else:
            cfg["enabled_filters_all"] = False
            cfg["enabled_filters"] = [str(f) for f in enabled]
        tools = data.get("tools")
        if not isinstance(tools, dict):
            tools = {}
        tools["qa_report"] = cfg
        data["tools"] = tools
        with path.open("wb") as f:
            tomli_w.dump(data, f)

    def setup_ui(self):
        """Setup user interface"""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # === Header ===
        header = QLabel("Publication-Quality Photometry Validation")
        header.setFont(QFont("Arial", 16, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("QLabel { color: #1565C0; padding: 10px; }")
        layout.addWidget(header)

        # === Control Panel ===
        control_group = QGroupBox("Report Generation")
        control_layout = QHBoxLayout(control_group)

        # QA Parameters button
        self.btn_params = QPushButton("QA Parameters")
        self.btn_params.setStyleSheet("""
            QPushButton {
                background-color: #9C27B0;
                color: white;
                font-weight: bold;
                padding: 10px 15px;
                font-size: 12px;
            }
            QPushButton:hover { background-color: #7B1FA2; }
        """)
        self.btn_params.clicked.connect(self.open_parameters_dialog)
        control_layout.addWidget(self.btn_params)

        self.btn_generate = QPushButton("Generate QA Report")
        self.btn_generate.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                padding: 10px 20px;
                font-size: 12px;
            }
            QPushButton:hover { background-color: #45a049; }
        """)
        self.btn_generate.clicked.connect(self.generate_report)
        control_layout.addWidget(self.btn_generate)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-weight: bold;
                padding: 10px 15px;
            }
        """)
        self.btn_stop.clicked.connect(self.stop_report)
        self.btn_stop.setEnabled(False)
        control_layout.addWidget(self.btn_stop)

        # Filter status label
        self.filter_status_label = QLabel()
        self._update_filter_status_label()
        self.filter_status_label.setStyleSheet("QLabel { color: #1565C0; font-weight: bold; }")
        control_layout.addWidget(self.filter_status_label)

        control_layout.addStretch()

        self.btn_save_plots = QPushButton("Save All Plots")
        self.btn_save_plots.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                padding: 10px 15px;
            }
        """)
        self.btn_save_plots.clicked.connect(self.save_all_plots)
        self.btn_save_plots.setEnabled(False)
        control_layout.addWidget(self.btn_save_plots)

        self.btn_export_latex = QPushButton("Export LaTeX")
        self.btn_export_latex.setStyleSheet("""
            QPushButton {
                background-color: #9C27B0;
                color: white;
                font-weight: bold;
                padding: 10px 15px;
            }
        """)
        self.btn_export_latex.clicked.connect(self.export_latex)
        self.btn_export_latex.setEnabled(False)
        control_layout.addWidget(self.btn_export_latex)

        layout.addWidget(control_group)

        # === Progress ===
        progress_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        progress_layout.addWidget(self.progress_bar)
        self.progress_label = QLabel("Ready")
        self.progress_label.setMinimumWidth(300)
        progress_layout.addWidget(self.progress_label)
        layout.addLayout(progress_layout)

        # === Tab Widget for Results ===
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #ccc; }
            QTabBar::tab {
                padding: 8px 16px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background-color: #E3F2FD;
                border-bottom: 2px solid #2196F3;
            }
        """)

        # Tab 1: Error Model
        self.tab_error = QWidget()
        self.setup_error_model_tab()
        self.tabs.addTab(self.tab_error, "Error")

        # Tab 2: Centroid QA
        self.tab_centroid = QWidget()
        self.setup_centroid_tab()
        self.tabs.addTab(self.tab_centroid, "Centroid")

        # Tab 3: Frame Quality
        self.tab_frame = QWidget()
        self.setup_frame_quality_tab()
        self.tabs.addTab(self.tab_frame, "Frames")

        # Tab 4: Background
        self.tab_background = QWidget()
        self.setup_background_tab()
        self.tabs.addTab(self.tab_background, "Bkg")

        # Tab 5: Publication Summary
        self.tab_publication = QWidget()
        self.setup_publication_tab()
        self.tabs.addTab(self.tab_publication, "Publish")

        # Tab 6: Log
        self.tab_log = QWidget()
        self.setup_log_tab()
        self.tabs.addTab(self.tab_log, "Log")

        layout.addWidget(self.tabs, stretch=1)

    def setup_error_model_tab(self):
        """Setup error model validation tab"""
        layout = QVBoxLayout(self.tab_error)

        # Description
        desc = QLabel(
            "<b>SNR-Error Model Validation</b><br>"
            "Compares predicted photometric errors (σ_pred = 1.0857/SNR) with observed "
            "frame-to-frame scatter (RMS). Literature expectation: RMS/σ_pred ≈ 1.0 and χ²/ν ≈ 1.0"
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("QLabel { background-color: #E3F2FD; padding: 10px; border-radius: 5px; }")
        layout.addWidget(desc)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Error model view:"))
        self.error_mode_combo = QComboBox()
        self.error_mode_combo.addItems(["Raw (photometry)", "Frame ZP corrected"])
        self.error_mode_combo.currentIndexChanged.connect(self._update_error_model_display)
        mode_row.addWidget(self.error_mode_combo)
        mode_row.addStretch()
        layout.addLayout(mode_row)

        # Splitter for table and plots
        splitter = QSplitter(Qt.Horizontal)

        # Summary table
        table_widget = QWidget()
        table_layout = QVBoxLayout(table_widget)
        table_layout.addWidget(QLabel("<b>Per-Filter Summary:</b>"))
        self.error_table = QTableWidget()
        self.error_table.setColumnCount(6)
        self.error_table.setHorizontalHeaderLabels([
            "Filter", "N_stars", "N_obs", "RMS/σ_pred", "χ²/ν", "SNR Range"
        ])
        self.error_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table_layout.addWidget(self.error_table)
        splitter.addWidget(table_widget)

        # Plots
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        self.error_figure = Figure(figsize=(10, 8))
        self.error_canvas = FigureCanvas(self.error_figure)
        plot_layout.addWidget(NavigationToolbar(self.error_canvas, self))
        plot_layout.addWidget(self.error_canvas)
        splitter.addWidget(plot_widget)

        splitter.setSizes([400, 800])
        layout.addWidget(splitter)

    def setup_centroid_tab(self):
        """Setup centroid QA tab"""
        layout = QVBoxLayout(self.tab_centroid)

        desc = QLabel(
            "<b>Centroid Quality Assessment</b><br>"
            "Analyzes the shift between initial detection position and refined centroid. "
            "Large shifts (>1 FWHM) may indicate matching or detection problems."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("QLabel { background-color: #E8F5E9; padding: 10px; border-radius: 5px; }")
        layout.addWidget(desc)

        splitter = QSplitter(Qt.Horizontal)

        # Summary
        summary_widget = QWidget()
        summary_layout = QVBoxLayout(summary_widget)
        summary_layout.addWidget(QLabel("<b>Summary Statistics:</b>"))
        self.centroid_summary = QTextEdit()
        self.centroid_summary.setReadOnly(True)
        self.centroid_summary.setMaximumHeight(200)
        summary_layout.addWidget(self.centroid_summary)
        summary_layout.addStretch()
        splitter.addWidget(summary_widget)

        # Plot
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        self.centroid_figure = Figure(figsize=(10, 6))
        self.centroid_canvas = FigureCanvas(self.centroid_figure)
        plot_layout.addWidget(NavigationToolbar(self.centroid_canvas, self))
        plot_layout.addWidget(self.centroid_canvas)
        splitter.addWidget(plot_widget)

        splitter.setSizes([300, 900])
        layout.addWidget(splitter)

    def setup_frame_quality_tab(self):
        """Setup frame quality tab"""
        layout = QVBoxLayout(self.tab_frame)

        desc = QLabel(
            "<b>Frame Quality Metrics</b><br>"
            "Per-frame quality assessment. Check/uncheck 'Use' to include/exclude frames from error model."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("QLabel { background-color: #FFF3E0; padding: 10px; border-radius: 5px; }")
        layout.addWidget(desc)

        # Control buttons
        frame_ctrl_layout = QHBoxLayout()

        self.btn_check_all = QPushButton("Check All")
        self.btn_check_all.clicked.connect(lambda: self._set_all_frame_checks(True))
        frame_ctrl_layout.addWidget(self.btn_check_all)

        self.btn_uncheck_flagged = QPushButton("Uncheck Flagged")
        self.btn_uncheck_flagged.clicked.connect(self._uncheck_flagged_frames)
        frame_ctrl_layout.addWidget(self.btn_uncheck_flagged)

        self.btn_apply_frame_selection = QPushButton("Apply & Regenerate")
        self.btn_apply_frame_selection.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                font-weight: bold;
                padding: 5px 10px;
            }
        """)
        self.btn_apply_frame_selection.clicked.connect(self._apply_frame_selection)
        frame_ctrl_layout.addWidget(self.btn_apply_frame_selection)

        self.btn_reset_qa = QPushButton("Reset All")
        self.btn_reset_qa.setStyleSheet("""
            QPushButton {
                background-color: #607D8B;
                color: white;
                font-weight: bold;
                padding: 5px 10px;
            }
        """)
        self.btn_reset_qa.clicked.connect(self._reset_qa_params)
        frame_ctrl_layout.addWidget(self.btn_reset_qa)

        frame_ctrl_layout.addStretch()

        self.frame_selection_label = QLabel("Selected: 0/0 frames")
        self.frame_selection_label.setStyleSheet("QLabel { font-weight: bold; color: #1565C0; }")
        frame_ctrl_layout.addWidget(self.frame_selection_label)

        layout.addLayout(frame_ctrl_layout)

        # Table with checkbox column
        self.frame_table = QTableWidget()
        self.frame_table.setColumnCount(10)
        self.frame_table.setHorizontalHeaderLabels([
            "Use", "Frame", "Filter", "N_tgt", "N_good", "Good%", "SNR", "FWHM", "Flag", "Reason"
        ])
        self.frame_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.frame_table.setColumnWidth(0, 40)  # Use checkbox
        self.frame_table.setColumnWidth(2, 50)  # Filter
        self.frame_table.setColumnWidth(3, 50)  # N_tgt
        self.frame_table.setColumnWidth(4, 55)  # N_good
        self.frame_table.setColumnWidth(5, 55)  # Good%
        self.frame_table.setColumnWidth(6, 50)  # SNR
        self.frame_table.setColumnWidth(7, 130)  # FWHM
        self.frame_table.setColumnWidth(8, 40)  # Flag
        self.frame_table.itemChanged.connect(self._on_frame_checkbox_changed)
        layout.addWidget(self.frame_table)

        # Store frame data for selection
        self.frame_data = None
        self.frame_checkboxes = {}

    def setup_background_tab(self):
        """Setup background QA tab"""
        layout = QVBoxLayout(self.tab_background)

        desc = QLabel(
            "<b>Background Estimation Quality</b><br>"
            "Validates annulus-based sky estimation. Low n_sky or high bkg_std "
            "indicates problematic background measurements."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("QLabel { background-color: #F3E5F5; padding: 10px; border-radius: 5px; }")
        layout.addWidget(desc)

        splitter = QSplitter(Qt.Horizontal)

        # Summary
        summary_widget = QWidget()
        summary_layout = QVBoxLayout(summary_widget)
        self.background_summary = QTextEdit()
        self.background_summary.setReadOnly(True)
        summary_layout.addWidget(self.background_summary)
        splitter.addWidget(summary_widget)

        # Plot
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        self.background_figure = Figure(figsize=(10, 6))
        self.background_canvas = FigureCanvas(self.background_figure)
        plot_layout.addWidget(NavigationToolbar(self.background_canvas, self))
        plot_layout.addWidget(self.background_canvas)
        splitter.addWidget(plot_widget)

        splitter.setSizes([300, 900])
        layout.addWidget(splitter)

    def setup_publication_tab(self):
        """Setup publication summary tab"""
        layout = QVBoxLayout(self.tab_publication)

        desc = QLabel(
            "<b>Publication-Ready Summary</b><br>"
            "Formatted tables and statistics ready for inclusion in scientific publications."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("QLabel { background-color: #ECEFF1; padding: 10px; border-radius: 5px; }")
        layout.addWidget(desc)

        # Verdict panel
        verdict_group = QGroupBox("Pipeline Validation Verdict")
        verdict_layout = QVBoxLayout(verdict_group)
        self.verdict_label = QLabel("Run QA Report to see validation results")
        self.verdict_label.setFont(QFont("Arial", 12))
        self.verdict_label.setAlignment(Qt.AlignCenter)
        verdict_layout.addWidget(self.verdict_label)
        layout.addWidget(verdict_group)

        # LaTeX output
        latex_group = QGroupBox("LaTeX Table")
        latex_layout = QVBoxLayout(latex_group)
        self.latex_text = QTextEdit()
        self.latex_text.setReadOnly(True)
        self.latex_text.setFont(QFont("Courier", 9))
        latex_layout.addWidget(self.latex_text)
        layout.addWidget(latex_group)

        # Summary statistics
        stats_group = QGroupBox("Key Statistics for Paper")
        stats_layout = QVBoxLayout(stats_group)
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        stats_layout.addWidget(self.stats_text)
        layout.addWidget(stats_group)

    def setup_log_tab(self):
        """Setup log tab"""
        layout = QVBoxLayout(self.tab_log)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Courier", 9))
        layout.addWidget(self.log_text)

    def log(self, msg: str):
        """Append to log"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {msg}")

    def _update_filter_status_label(self):
        """Update the filter status label"""
        if self.qa_params["enabled_filters"] is None:
            if self.available_filters:
                text = f"Filters: ALL ({', '.join(self.available_filters)})"
            else:
                text = "Filters: ALL"
        else:
            enabled = self.qa_params["enabled_filters"]
            disabled = [f for f in self.available_filters if f not in enabled]
            if disabled:
                text = f"Filters: {', '.join(enabled)} (excluded: {', '.join(disabled)})"
            else:
                text = f"Filters: {', '.join(enabled)}"
        self.filter_status_label.setText(text)

    def open_parameters_dialog(self):
        """Open QA parameters dialog with filter on/off controls"""
        dialog = QDialog(self)
        dialog.setWindowTitle("QA Report Parameters")
        dialog.setMinimumWidth(500)

        layout = QVBoxLayout(dialog)

        # === Filter Selection ===
        filter_group = QGroupBox("Filter Selection")
        filter_layout = QVBoxLayout(filter_group)

        filter_desc = QLabel("Enable/disable filters for QA analysis:")
        filter_desc.setStyleSheet("color: gray;")
        filter_layout.addWidget(filter_desc)

        # Create checkboxes for each detected filter
        self.filter_checkboxes = {}
        filter_grid = QGridLayout()

        if not self.available_filters:
            filter_layout.addWidget(QLabel("No filters detected. Run Forced Photometry first."))
        else:
            for i, filt in enumerate(self.available_filters):
                cb = QCheckBox(f"{filt} band")  # Keep original case from FITS header
                cb.setStyleSheet("QCheckBox { font-weight: bold; font-size: 12px; }")

                # Check if this filter is currently enabled
                if self.qa_params["enabled_filters"] is None:
                    cb.setChecked(True)
                else:
                    cb.setChecked(filt in self.qa_params["enabled_filters"])

                self.filter_checkboxes[filt] = cb
                filter_grid.addWidget(cb, i // 3, i % 3)

            filter_layout.addLayout(filter_grid)

            # Quick buttons
            quick_layout = QHBoxLayout()
            btn_all = QPushButton("Select All")
            btn_all.clicked.connect(lambda: self._set_all_filters(True))
            quick_layout.addWidget(btn_all)

            btn_none = QPushButton("Select None")
            btn_none.clicked.connect(lambda: self._set_all_filters(False))
            quick_layout.addWidget(btn_none)

            quick_layout.addStretch()
            filter_layout.addLayout(quick_layout)

        layout.addWidget(filter_group)

        # === Quality Cuts ===
        cuts_group = QGroupBox("Quality Cuts")
        cuts_layout = QFormLayout(cuts_group)

        # Min frames per star
        self.spin_min_frames = QSpinBox()
        self.spin_min_frames.setRange(2, 50)
        self.spin_min_frames.setValue(self.qa_params["min_n_frames"])
        self.spin_min_frames.setToolTip("Minimum number of frames for a star to be included in error model validation")
        cuts_layout.addRow("Min frames per star:", self.spin_min_frames)

        # Min SNR
        self.spin_min_snr = QDoubleSpinBox()
        self.spin_min_snr.setRange(0.0, 100.0)
        self.spin_min_snr.setValue(self.qa_params["min_snr"])
        self.spin_min_snr.setToolTip("Exclude measurements with SNR below this value")
        cuts_layout.addRow("Min SNR:", self.spin_min_snr)

        # Max chi2_nu (outlier cut)
        self.spin_max_chi2 = QDoubleSpinBox()
        self.spin_max_chi2.setRange(1.0, 1000.0)
        self.spin_max_chi2.setValue(self.qa_params["max_chi2_nu"])
        self.spin_max_chi2.setToolTip("Exclude stars with reduced chi² above this value (outlier removal)")
        cuts_layout.addRow("Max χ²/ν (outlier cut):", self.spin_max_chi2)

        # Max delta_r
        self.spin_max_dr = QDoubleSpinBox()
        self.spin_max_dr.setRange(0.5, 20.0)
        self.spin_max_dr.setValue(self.qa_params["max_delta_r"])
        self.spin_max_dr.setToolTip("Exclude measurements with centroid shift above this value (pixels)")
        cuts_layout.addRow("Max Δr (px):", self.spin_max_dr)

        # Exclude saturated
        self.check_exclude_sat = QCheckBox("Exclude saturated sources")
        self.check_exclude_sat.setChecked(self.qa_params["exclude_saturated"])
        cuts_layout.addRow("", self.check_exclude_sat)

        layout.addWidget(cuts_group)

        # === Frame Quality Thresholds ===
        frame_group = QGroupBox("Frame Quality Thresholds")
        frame_layout = QFormLayout(frame_group)

        self.combo_frame_mode = QComboBox()
        self.combo_frame_mode.addItems(["Absolute", "Percentile"])
        self.combo_frame_mode.setCurrentIndex(0 if self.qa_params.get("frame_flag_mode") == "absolute" else 1)
        frame_layout.addRow("Flagging mode:", self.combo_frame_mode)

        self.spin_frame_snr = QDoubleSpinBox()
        self.spin_frame_snr.setRange(0.0, 200.0)
        self.spin_frame_snr.setValue(float(self.qa_params.get("frame_snr_min", 10.0)))
        self.spin_frame_snr.setToolTip("Flag frames with median SNR below this value")
        frame_layout.addRow("SNR med min:", self.spin_frame_snr)

        self.spin_frame_goodmag = QDoubleSpinBox()
        self.spin_frame_goodmag.setRange(0.0, 1.0)
        self.spin_frame_goodmag.setSingleStep(0.05)
        self.spin_frame_goodmag.setValue(float(self.qa_params.get("frame_goodmag_min", 0.9)))
        self.spin_frame_goodmag.setToolTip("Flag frames with good mag fraction below this value")
        frame_layout.addRow("Good mag frac min:", self.spin_frame_goodmag)

        self.combo_fwhm_mode = QComboBox()
        self.combo_fwhm_mode.addItems(["Scale (x median)", "Absolute (arcsec)"])
        self.combo_fwhm_mode.setCurrentIndex(0 if self.qa_params.get("frame_fwhm_mode") == "scale" else 1)
        frame_layout.addRow("FWHM mode:", self.combo_fwhm_mode)

        self.spin_fwhm_scale = QDoubleSpinBox()
        self.spin_fwhm_scale.setRange(1.0, 5.0)
        self.spin_fwhm_scale.setSingleStep(0.1)
        self.spin_fwhm_scale.setValue(float(self.qa_params.get("frame_fwhm_scale", 1.5)))
        self.spin_fwhm_scale.setToolTip("Flag frames with FWHM > scale * median FWHM")
        frame_layout.addRow("FWHM scale:", self.spin_fwhm_scale)

        self.spin_fwhm_abs = QDoubleSpinBox()
        self.spin_fwhm_abs.setRange(0.0, 20.0)
        self.spin_fwhm_abs.setSingleStep(0.1)
        self.spin_fwhm_abs.setValue(float(self.qa_params.get("frame_fwhm_abs", 8.0)))
        self.spin_fwhm_abs.setToolTip("Flag frames with FWHM > absolute threshold")
        frame_layout.addRow("FWHM abs:", self.spin_fwhm_abs)

        layout.addWidget(frame_group)

        # === Error Model Source (for publication verdict) ===
        source_group = QGroupBox("Error Model Source")
        source_layout = QFormLayout(source_group)
        self.combo_error_source = QComboBox()
        self.combo_error_source.addItems(["Raw (instrumental)", "ZP (frame corrected)"])
        source_map = {
            "raw": 0,
            "zp": 1,
        }
        current = source_map.get(str(self.qa_params.get("error_model_source", "raw")).lower(), 0)
        self.combo_error_source.setCurrentIndex(current)
        self.combo_error_source.setToolTip("Select which error model to use for Publish verdict and summary")
        source_layout.addRow("Use for Publish:", self.combo_error_source)
        layout.addWidget(source_group)

        # === Preset Buttons ===
        preset_group = QGroupBox("Presets")
        preset_layout = QHBoxLayout(preset_group)

        btn_default = QPushButton("Default (All Data)")
        btn_default.clicked.connect(lambda: self._apply_preset("default"))
        preset_layout.addWidget(btn_default)

        btn_strict = QPushButton("Strict (High Quality)")
        btn_strict.clicked.connect(lambda: self._apply_preset("strict"))
        preset_layout.addWidget(btn_strict)

        btn_publication = QPushButton("Publication Ready")
        btn_publication.clicked.connect(lambda: self._apply_preset("publication"))
        preset_layout.addWidget(btn_publication)

        layout.addWidget(preset_group)

        # === Dialog Buttons ===
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        # Show dialog
        if dialog.exec_() == QDialog.Accepted:
            self._save_parameters_from_dialog()

    def _set_all_filters(self, checked: bool):
        """Set all filter checkboxes to checked or unchecked"""
        for cb in self.filter_checkboxes.values():
            cb.setChecked(checked)

    def _apply_preset(self, preset: str):
        """Apply parameter preset"""
        if preset == "default":
            # No filtering - use all data
            self.spin_min_frames.setValue(2)
            self.spin_min_snr.setValue(0.0)
            self.spin_max_chi2.setValue(999.0)  # No chi2 cut
            self.spin_max_dr.setValue(10.0)
            self.check_exclude_sat.setChecked(False)
            self._set_all_filters(True)

        elif preset == "strict":
            self.spin_min_frames.setValue(5)
            self.spin_min_snr.setValue(10.0)
            self.spin_max_chi2.setValue(5.0)
            self.spin_max_dr.setValue(1.5)
            self.check_exclude_sat.setChecked(True)

        elif preset == "publication":
            # Recommended for paper
            self.spin_min_frames.setValue(3)
            self.spin_min_snr.setValue(5.0)
            self.spin_max_chi2.setValue(5.0)
            self.spin_max_dr.setValue(2.0)
            self.check_exclude_sat.setChecked(True)

    def _save_parameters_from_dialog(self):
        """Save parameters from dialog to self.qa_params"""
        # Get enabled filters
        enabled = [f for f, cb in self.filter_checkboxes.items() if cb.isChecked()]
        if len(enabled) == len(self.available_filters):
            self.qa_params["enabled_filters"] = None  # All enabled
        elif len(enabled) == 0:
            QMessageBox.warning(self, "Warning", "At least one filter must be enabled!")
            self.qa_params["enabled_filters"] = None
        else:
            self.qa_params["enabled_filters"] = enabled

        # Get quality cuts
        self.qa_params["min_n_frames"] = self.spin_min_frames.value()
        self.qa_params["min_snr"] = self.spin_min_snr.value()
        self.qa_params["max_chi2_nu"] = self.spin_max_chi2.value()
        self.qa_params["max_delta_r"] = self.spin_max_dr.value()
        self.qa_params["exclude_saturated"] = self.check_exclude_sat.isChecked()
        source_map = {0: "raw", 1: "zp"}
        self.qa_params["error_model_source"] = source_map.get(self.combo_error_source.currentIndex(), "raw")
        self.qa_params["frame_flag_mode"] = "absolute" if self.combo_frame_mode.currentIndex() == 0 else "percentile"
        self.qa_params["frame_snr_min"] = self.spin_frame_snr.value()
        self.qa_params["frame_goodmag_min"] = self.spin_frame_goodmag.value()
        self.qa_params["frame_fwhm_mode"] = "scale" if self.combo_fwhm_mode.currentIndex() == 0 else "absolute"
        self.qa_params["frame_fwhm_scale"] = self.spin_fwhm_scale.value()
        self.qa_params["frame_fwhm_abs"] = self.spin_fwhm_abs.value()

        # Update status label
        self._update_filter_status_label()
        self._save_qa_params()

        self.log(f"Parameters updated: filters={self.qa_params['enabled_filters']}, "
                 f"min_snr={self.qa_params['min_snr']}, exclude_sat={self.qa_params['exclude_saturated']}")

    def generate_report(self, from_frame_selection=False):
        """Generate QA report"""
        if self.worker and self.worker.isRunning():
            return

        self.log_text.clear()
        self.log("Starting QA report generation...")
        self._save_qa_params()

        # If not from frame selection, reset frame-specific params
        if not from_frame_selection:
            self.qa_params["selected_frames"] = None
            self.qa_params["skip_frame_flagging"] = False

        self.log(f"Parameters: {self.qa_params}")

        self.worker = QAReportWorker(
            self.result_dir,
            params=self.qa_params
        )
        self.worker.progress.connect(self.on_progress)
        self.worker.log.connect(self.log)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)

        self.btn_generate.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setValue(0)

        self.worker.start()

    def stop_report(self):
        """Stop report generation"""
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.log("Stop requested...")

    def on_progress(self, current: int, total: int, msg: str):
        """Update progress"""
        pct = int(100 * current / total) if total > 0 else 0
        self.progress_bar.setValue(pct)
        self.progress_label.setText(f"{current}/{total} | {msg}")

    def on_finished(self, results: Dict):
        """Handle completion"""
        self.btn_generate.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_save_plots.setEnabled(True)
        self.btn_export_latex.setEnabled(True)
        self.progress_label.setText("Complete")
        self.progress_bar.setValue(100)

        self.results = results
        self.update_displays()
        self.log("QA report complete!")

    def on_error(self, error: str):
        """Handle error"""
        self.btn_generate.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_label.setText("Error")
        self.log(f"ERROR: {error}")
        QMessageBox.critical(self, "Error", f"QA Report failed:\n{error}")

    def update_displays(self):
        """Update all display panels with results"""
        # Error model tab
        self._update_error_model_display()

        # Centroid tab
        self._update_centroid_display()

        # Frame quality tab
        self._update_frame_quality_display()

        # Background tab
        self._update_background_display()

        # Publication tab
        self._update_publication_display()

    def _update_error_model_display(self):
        """Update error model tab"""
        if "error_model_raw" not in self.results:
            return

        use_zp = False
        if hasattr(self, "error_mode_combo"):
            self.error_mode_combo.setEnabled("error_model_zp" in self.results)
            if "error_model_zp" not in self.results and self.error_mode_combo.currentIndex() == 1:
                self.error_mode_combo.setCurrentIndex(0)
            use_zp = (self.error_mode_combo.currentIndex() == 1)

        em = self.results.get("error_model_zp") if use_zp else self.results.get("error_model_raw")
        if em is None:
            return
        if "error" in em:
            self.log(f"Error model: {em['error']}")
            return

        # Update table
        if "by_filter" in em and isinstance(em["by_filter"], pd.DataFrame):
            df = em["by_filter"]
            self.error_table.setRowCount(len(df))
            for i, (_, row) in enumerate(df.iterrows()):
                self.error_table.setItem(i, 0, QTableWidgetItem(str(row["FILTER"])))
                self.error_table.setItem(i, 1, QTableWidgetItem(str(int(row["n_stars"]))))
                self.error_table.setItem(i, 2, QTableWidgetItem(str(int(row["n_obs_total"]))))
                self.error_table.setItem(i, 3, QTableWidgetItem(f"{row['rms_over_pred_med']:.2f}±{row['rms_over_pred_std']:.2f}"))
                self.error_table.setItem(i, 4, QTableWidgetItem(f"{row['chi2_nu_med']:.2f}±{row['chi2_nu_std']:.2f}"))
                self.error_table.setItem(i, 5, QTableWidgetItem(f"{row['snr_p10']:.0f}-{row['snr_p90']:.0f}"))

        # Update plots
        if "by_star" in em and isinstance(em["by_star"], pd.DataFrame):
            self._plot_error_model(em["by_star"])

    def _plot_error_model(self, df: pd.DataFrame):
        """Generate error model plots"""
        self.error_figure.clear()

        # 2x2 subplot
        axes = self.error_figure.subplots(2, 2)

        filters = df["FILTER"].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(filters)))

        for filt, color in zip(filters, colors):
            sub = df[df["FILTER"] == filt]

            # Plot 1: RMS vs magnitude
            ax = axes[0, 0]
            ax.scatter(sub["mag_mean"], sub["mag_std"], alpha=0.5, s=10, c=[color], label=filt)
            ax.set_xlabel("Mean Magnitude")
            ax.set_ylabel("Observed RMS")
            ax.set_title("RMS vs Magnitude")
            ax.legend(fontsize=8)

            # Plot 2: RMS vs predicted error
            ax = axes[0, 1]
            ax.scatter(sub["mag_err_med"], sub["mag_std"], alpha=0.5, s=10, c=[color], label=filt)
            ax.set_xlabel("Predicted Error (mag_err)")
            ax.set_ylabel("Observed RMS")
            ax.set_title("RMS vs Predicted Error")

        # Add 1:1 line
        lim = max(df["mag_err_med"].max(), df["mag_std"].max()) * 1.1
        axes[0, 1].plot([0, lim], [0, lim], "r--", lw=1.5, label="1:1")
        axes[0, 1].set_xlim(0, lim)
        axes[0, 1].set_ylim(0, lim)
        axes[0, 1].legend(fontsize=8)

        # Plot 3: RMS/mag_err ratio vs SNR
        ax = axes[1, 0]
        for filt, color in zip(filters, colors):
            sub = df[df["FILTER"] == filt]
            ax.scatter(sub["snr_med"], sub["rms_over_pred"], alpha=0.5, s=10, c=[color], label=filt)
        ax.axhline(1.0, color="r", ls="--", lw=1.5)
        ax.set_xlabel("Median SNR")
        ax.set_ylabel("RMS / mag_err")
        ax.set_title("Error Model Ratio vs SNR")
        ax.set_ylim(0, 3)
        ax.legend(fontsize=8)

        # Plot 4: Reduced chi² histogram
        ax = axes[1, 1]
        for filt, color in zip(filters, colors):
            sub = df[df["FILTER"] == filt]
            chi2 = sub["chi2_nu"].dropna()
            if len(chi2) > 5:
                ax.hist(chi2, bins=30, alpha=0.5, color=color, label=filt, density=True)
        ax.axvline(1.0, color="r", ls="--", lw=1.5)
        ax.set_xlabel("Reduced χ² (per star)")
        ax.set_ylabel("Density")
        ax.set_title("χ²/ν Distribution (expected=1)")
        ax.legend(fontsize=8)

        self.error_figure.tight_layout()
        self.error_canvas.draw()

    def _update_centroid_display(self):
        """Update centroid tab"""
        if "centroid" not in self.results:
            return

        ct = self.results["centroid"]
        if "error" in ct:
            self.centroid_summary.setText(f"Error: {ct['error']}")
            return

        if "summary" in ct:
            s = ct["summary"]
            text = f"""Centroid Shift Statistics:
─────────────────────────
N measurements: {s['n_measurements']:,}

Δr (pixels):
  Median: {s['delta_r_median']:.3f}
  Mean: {s['delta_r_mean']:.3f}
  Std: {s['delta_r_std']:.3f}

Percentiles:
  90th: {s['delta_r_p90']:.3f}
  95th: {s['delta_r_p95']:.3f}
  99th: {s['delta_r_p99']:.3f}
  Max: {s['delta_r_max']:.3f}

Outliers:
  > 1 px: {s['n_outliers_gt1px']:,}
  > 2 px: {s['n_outliers_gt2px']:,}
"""
            if "data" in ct and isinstance(ct["data"], pd.DataFrame) and "FILTER" in ct["data"].columns:
                per_filter_lines = []
                df = ct["data"].copy()
                for filt, sub in df.groupby("FILTER"):
                    dr = sub["delta_r"].dropna()
                    if dr.empty:
                        continue
                    per_filter_lines.append(
                        f"{filt}: N={len(dr):,} | med={np.median(dr):.3f} "
                        f"mean={np.mean(dr):.3f} std={np.std(dr):.3f}"
                    )
                if per_filter_lines:
                    text += "\nPer-filter Δr:\n  " + "\n  ".join(per_filter_lines) + "\n"
            self.centroid_summary.setText(text)

        # Plot
        if "data" in ct and isinstance(ct["data"], pd.DataFrame):
            self._plot_centroid(ct["data"])

    def _plot_centroid(self, df: pd.DataFrame):
        """Generate centroid plots"""
        self.centroid_figure.clear()
        axes = self.centroid_figure.subplots(1, 2)

        dr = df["delta_r"].dropna()
        filters = []
        colors = {}
        if "FILTER" in df.columns:
            filters = sorted(df["FILTER"].dropna().astype(str).unique())
            colors = self._get_filter_colors(filters)

        # Histogram
        ax = axes[0]
        if filters:
            for filt in filters:
                sub = df[df["FILTER"] == filt]["delta_r"].dropna()
                if sub.empty:
                    continue
                ax.hist(sub, bins=50, alpha=0.5, color=colors[filt], label=filt)
        else:
            ax.hist(dr, bins=50, edgecolor="black", alpha=0.7)
        if len(dr):
            ax.axvline(np.median(dr), color="r", ls="--", lw=2, label=f"median={np.median(dr):.2f}")
            ax.axvline(np.percentile(dr, 95), color="orange", ls="--", lw=2, label=f"95%={np.percentile(dr, 95):.2f}")
        ax.set_xlabel("Centroid Shift Δr (pixels)")
        ax.set_ylabel("Count")
        ax.set_title("Centroid Shift Distribution")
        ax.legend()

        # 2D scatter
        ax = axes[1]
        if filters:
            for filt in filters:
                sub = df[df["FILTER"] == filt]
                ax.scatter(sub["delta_x"], sub["delta_y"], alpha=0.3, s=6, color=colors[filt], label=filt)
        else:
            ax.scatter(df["delta_x"], df["delta_y"], alpha=0.3, s=5)
        ax.axhline(0, color="gray", ls="--", lw=0.5)
        ax.axvline(0, color="gray", ls="--", lw=0.5)
        circle = plt.Circle((0, 0), 1.0, fill=False, color="r", ls="--", lw=1.5, label="1 px")
        ax.add_patch(circle)
        ax.set_xlabel("Δx (pixels)")
        ax.set_ylabel("Δy (pixels)")
        ax.set_title("Centroid Shift Vector")
        ax.set_aspect("equal")
        ax.legend()

        self.centroid_figure.tight_layout()
        self.centroid_canvas.draw()

    def _update_frame_quality_display(self):
        """Update frame quality tab"""
        if "frame_quality" not in self.results:
            return

        fq = self.results["frame_quality"]
        if "data" in fq and isinstance(fq["data"], pd.DataFrame):
            df = fq["data"].copy()
            if "filter" in df.columns:
                if self.available_filters:
                    order_filters = [f.strip().lower() for f in self.available_filters]
                else:
                    order_filters = list(pd.unique(df.sort_values("frame")["filter"].astype(str).str.strip().str.lower()))
                order = {f: i for i, f in enumerate(order_filters)}
                df["_filter_rank"] = df["filter"].astype(str).str.strip().str.lower().map(order).fillna(99)
                df = df.sort_values(["_filter_rank", "frame"])
            self.frame_data = df.copy()
            self.frame_checkboxes = {}

            # Block signals while populating
            self.frame_table.blockSignals(True)
            self.frame_table.setRowCount(len(df))

            for i, (_, row) in enumerate(df.iterrows()):
                frame_name = str(row["frame"])
                is_flagged = row.get("qa_flag", False)

                # Column 0: Use checkbox
                checkbox_item = QTableWidgetItem()
                checkbox_item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                checkbox_item.setCheckState(Qt.Unchecked if is_flagged else Qt.Checked)
                self.frame_table.setItem(i, 0, checkbox_item)
                self.frame_checkboxes[i] = {"frame": frame_name, "item": checkbox_item}

                # Column 1: Frame name
                self.frame_table.setItem(i, 1, QTableWidgetItem(frame_name))

                # Column 2: Filter (from data)
                filter_val = str(row.get("filter", "")) if pd.notna(row.get("filter", "")) else ""
                self.frame_table.setItem(i, 2, QTableWidgetItem(filter_val))

                # Column 3: N_targets
                self.frame_table.setItem(i, 3, QTableWidgetItem(str(int(row["n_targets"]))))

                # Column 4: N_goodmag
                self.frame_table.setItem(i, 4, QTableWidgetItem(str(int(row["n_goodmag"]))))

                # Column 5: Good%
                self.frame_table.setItem(i, 5, QTableWidgetItem(f"{row['goodmag_fraction']*100:.1f}%"))

                # Column 6: SNR_med
                self.frame_table.setItem(i, 6, QTableWidgetItem(f"{row['snr_med']:.1f}"))

                # Column 7: FWHM
                fwhm_px = row.get("fwhm_used", np.nan)
                pixscale = float(getattr(self.params.P, "pixel_scale_arcsec", np.nan))
                if np.isfinite(fwhm_px):
                    if np.isfinite(pixscale) and pixscale > 0:
                        fwhm_arcsec = fwhm_px * pixscale
                        fwhm_str = f'{fwhm_arcsec:.2f}" ({fwhm_px:.2f} px)'
                    else:
                        fwhm_str = f"{fwhm_px:.2f} px"
                else:
                    fwhm_str = "N/A"
                self.frame_table.setItem(i, 7, QTableWidgetItem(fwhm_str))

                # Column 8: Flagged
                flagged = "Yes" if is_flagged else "No"
                flag_item = QTableWidgetItem(flagged)
                if is_flagged:
                    flag_item.setBackground(QColor("#FFEB3B"))  # Yellow
                self.frame_table.setItem(i, 8, flag_item)

                # Column 9: Reason
                reasons = []
                if row.get("fwhm_flag", False):
                    reasons.append("FWHM")
                if row.get("goodmag_flag", False):
                    reasons.append("LowGood")
                if row.get("snr_flag", False):
                    reasons.append("LowSNR")
                self.frame_table.setItem(i, 9, QTableWidgetItem(", ".join(reasons)))

                # Color the row if flagged
                if is_flagged:
                    for col in range(1, 10):
                        item = self.frame_table.item(i, col)
                        if item:
                            item.setBackground(QColor("#FFF9C4"))  # Light yellow

            self.frame_table.blockSignals(False)
            self._update_frame_selection_label()

    def _on_frame_checkbox_changed(self, item):
        """Handle frame checkbox state change"""
        if item.column() == 0:
            self._update_frame_selection_label()

    def _update_frame_selection_label(self):
        """Update the frame selection count label"""
        checked = 0
        total = self.frame_table.rowCount()
        for i in range(total):
            item = self.frame_table.item(i, 0)
            if item and item.checkState() == Qt.Checked:
                checked += 1
        self.frame_selection_label.setText(f"Selected: {checked}/{total} frames")

    def _set_all_frame_checks(self, checked: bool):
        """Set all frame checkboxes to checked or unchecked"""
        self.frame_table.blockSignals(True)
        for i in range(self.frame_table.rowCount()):
            item = self.frame_table.item(i, 0)
            if item:
                item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
        self.frame_table.blockSignals(False)
        self._update_frame_selection_label()

    def _uncheck_flagged_frames(self):
        """Uncheck all flagged frames"""
        self.frame_table.blockSignals(True)
        for i in range(self.frame_table.rowCount()):
            flag_item = self.frame_table.item(i, 8)  # Flag column
            if flag_item and flag_item.text() == "Yes":
                checkbox = self.frame_table.item(i, 0)
                if checkbox:
                    checkbox.setCheckState(Qt.Unchecked)
        self.frame_table.blockSignals(False)
        self._update_frame_selection_label()

    def _apply_frame_selection(self):
        """Apply frame selection and regenerate QA report"""
        # Get list of selected frames
        selected_frames = []
        for i in range(self.frame_table.rowCount()):
            checkbox = self.frame_table.item(i, 0)
            frame_item = self.frame_table.item(i, 1)
            if checkbox and frame_item and checkbox.checkState() == Qt.Checked:
                selected_frames.append(frame_item.text())

        if not selected_frames:
            QMessageBox.warning(self, "No Frames", "At least one frame must be selected!")
            return

        # Store selected frames in qa_params
        self.qa_params["selected_frames"] = selected_frames
        self.qa_params["skip_frame_flagging"] = True  # Don't re-flag selected frames
        self.log(f"Applied frame selection: {len(selected_frames)} frames selected (flagging disabled)")

        # Regenerate report (with frame selection flag)
        self.generate_report(from_frame_selection=True)

    def _reset_qa_params(self):
        """Reset all QA parameters to defaults and regenerate"""
        # Reset to publication-ready defaults
        self.qa_params = {
            "min_n_frames": 3,
            "min_snr": 5.0,
            "max_chi2_nu": 5.0,
            "max_delta_r": 2.0,
            "exclude_saturated": True,
            "enabled_filters": None,
            "error_model_source": "raw",
            "frame_flag_mode": "absolute",
            "frame_snr_min": 10.0,
            "frame_goodmag_min": 0.9,
            "frame_fwhm_mode": "scale",
            "frame_fwhm_scale": 1.5,
            "frame_fwhm_abs": 8.0,
            "selected_frames": None,  # Reset frame selection
            "skip_frame_flagging": False,  # Re-enable flagging
        }
        self._update_filter_status_label()
        self._save_qa_params()
        self.log("QA parameters reset to defaults (Publication Ready)")

        # Regenerate report
        self.generate_report()

    def _update_background_display(self):
        """Update background tab"""
        if "background" not in self.results:
            return

        bg = self.results["background"]
        if "error" in bg:
            self.background_summary.setText(f"Error: {bg['error']}")
            return

        if "summary" in bg:
            s = bg["summary"]
            text = f"""Background Estimation Quality:
─────────────────────────────
N measurements: {s['n_measurements']:,}
N frames: {s['n_frames']}

Global Statistics:
  Background median: {s['bkg_median_global']:.1f} ADU
  Background std: {s['bkg_std_global']:.2f} ADU
  n_sky median: {s['n_sky_global_med']:.0f} pixels
  n_sky min: {s['n_sky_global_min']} pixels

Quality Flags:
  Low n_sky frames: {s['n_frames_low_nsky']}
  High bkg_std frames: {s['n_frames_high_bkgstd']}
"""
            if "by_filter" in bg and isinstance(bg["by_filter"], pd.DataFrame):
                per_filter_lines = []
                for _, row in bg["by_filter"].iterrows():
                    per_filter_lines.append(
                        f"{row['FILTER']}: N={int(row['n_measurements']):,} "
                        f"frames={int(row['n_frames'])} "
                        f"bkg_med={row['bkg_median_global']:.1f} "
                        f"bkg_std={row['bkg_std_global']:.2f} "
                        f"n_sky_med={row['n_sky_global_med']:.0f} "
                        f"n_sky_min={row['n_sky_global_min']:.0f}"
                    )
                if per_filter_lines:
                    text += "\nPer-filter:\n  " + "\n  ".join(per_filter_lines) + "\n"
            self.background_summary.setText(text)

        # Plot
        if "by_frame" in bg and isinstance(bg["by_frame"], pd.DataFrame):
            self._plot_background(bg["by_frame"])

    def _plot_background(self, df: pd.DataFrame):
        """Generate background plots"""
        self.background_figure.clear()
        axes = self.background_figure.subplots(1, 2)

        # n_sky distribution
        ax = axes[0]
        if "filter" in df.columns:
            filters = sorted(df["filter"].dropna().astype(str).unique())
            colors = self._get_filter_colors(filters)
            for filt in filters:
                sub = df[df["filter"] == filt]["n_sky_med"].dropna()
                if sub.empty:
                    continue
                ax.hist(sub, bins=30, alpha=0.5, color=colors[filt], label=filt)
        else:
            ax.hist(df["n_sky_med"].dropna(), bins=30, edgecolor="black", alpha=0.7)
        n_sky_threshold = 50
        try:
            if "background" in self.results and "summary" in self.results["background"]:
                n_sky_threshold = int(self.results["background"]["summary"].get("n_sky_threshold", 50))
        except Exception:
            n_sky_threshold = 50
        ax.axvline(n_sky_threshold, color="r", ls="--", lw=1.5, label=f"Warning threshold ({n_sky_threshold})")
        ax.set_xlabel("Median n_sky (pixels)")
        ax.set_ylabel("Count")
        ax.set_title("Sky Pixel Count Distribution")
        ax.legend()

        # bkg_std distribution
        ax = axes[1]
        if "filter" in df.columns:
            filters = sorted(df["filter"].dropna().astype(str).unique())
            colors = self._get_filter_colors(filters)
            for filt in filters:
                sub = df[df["filter"] == filt]["bkg_std_med"].dropna()
                if sub.empty:
                    continue
                ax.hist(sub, bins=30, alpha=0.5, color=colors[filt], label=filt)
        else:
            ax.hist(df["bkg_std_med"].dropna(), bins=30, edgecolor="black", alpha=0.7)
        p90 = np.nanpercentile(df["bkg_std_med"], 90)
        ax.axvline(p90, color="r", ls="--", lw=1.5, label=f"90th percentile ({p90:.2f})")
        ax.set_xlabel("Median Background Std (ADU)")
        ax.set_ylabel("Count")
        ax.set_title("Background Noise Distribution")
        ax.legend()

        self.background_figure.tight_layout()
        self.background_canvas.draw()

    def _get_filter_colors(self, filters: List[str]) -> Dict[str, Any]:
        cmap = plt.cm.tab10
        colors = cmap(np.linspace(0, 1, max(len(filters), 1)))
        return {f: colors[i] for i, f in enumerate(filters)}

    def _update_publication_display(self):
        """Update publication summary tab"""
        # Verdict
        em_block = self._select_error_model_block(self.results)
        if em_block is not None:
            em_block = em_block.get("overall")

        if em_block is not None:
            em = em_block
            rms_ratio = em.get("rms_over_pred_global_med", np.nan)
            chi2_nu = em.get("chi2_nu_global_med", np.nan)

            if np.isfinite(rms_ratio) and np.isfinite(chi2_nu):
                if 0.8 <= rms_ratio <= 1.2 and 0.7 <= chi2_nu <= 1.3:
                    verdict = "✅ PASS: Error model validated for publication"
                    color = "#4CAF50"
                elif 0.6 <= rms_ratio <= 1.5 and 0.5 <= chi2_nu <= 2.0:
                    verdict = "⚠️ MARGINAL: Error model acceptable with caveats"
                    color = "#FF9800"
                else:
                    verdict = "❌ FAIL: Error model requires investigation"
                    color = "#f44336"

                # Build filter info string
                if self.qa_params["enabled_filters"] is None:
                    filter_info = "All filters"
                else:
                    filter_info = ", ".join(self.qa_params["enabled_filters"])
                source_label = str(self.qa_params.get("error_model_source", "raw")).lower()
                if source_label not in ("raw", "zp"):
                    source_label = "raw"

                per_filter_lines = []
                em_full = self._select_error_model_block(self.results)
                if em_full and "by_filter" in em_full:
                    for _, row in em_full["by_filter"].iterrows():
                        per_filter_lines.append(
                            f"{row['FILTER']}: RMS/σ={row['rms_over_pred_med']:.2f}, "
                            f"χ²/ν={row['chi2_nu_med']:.2f} (N={int(row['n_stars'])})"
                        )
                per_filter_text = "\n".join(per_filter_lines) if per_filter_lines else "N/A"

                self.verdict_label.setText(
                    f"{verdict}\n\n"
                    f"RMS/σ_pred = {rms_ratio:.2f} (expected: 1.0±0.2)\n"
                    f"χ²/ν = {chi2_nu:.2f} (expected: 1.0±0.3)\n\n"
                    f"Error model: {source_label}\n"
                    f"Per-filter: {per_filter_text}\n"
                    f"─── Applied Cuts ───\n"
                    f"Filters: {filter_info}\n"
                    f"Min SNR: {self.qa_params['min_snr']}, "
                    f"Exclude saturated: {self.qa_params['exclude_saturated']}"
                )
                self.verdict_label.setStyleSheet(f"QLabel {{ color: {color}; }}")

        # LaTeX table
        if "publication" in self.results and "latex" in self.results["publication"]:
            self.latex_text.setText(self.results["publication"]["latex"])

        # Stats text
        stats_lines = []
        em_block_full = self._select_error_model_block(self.results)
        if em_block_full and "overall" in em_block_full:
            em = em_block_full["overall"]
            stats_lines.append("=== Error Model Validation ===")
            stats_lines.append(f"N_stars: {em.get('n_stars', 'N/A')}")
            stats_lines.append(f"N_observations: {em.get('n_observations', 'N/A')}")
            stats_lines.append(f"RMS/σ_pred (global): {em.get('rms_over_pred_global_med', np.nan):.3f} ± {em.get('rms_over_pred_global_std', np.nan):.3f}")
            stats_lines.append(f"χ²/ν (global): {em.get('chi2_nu_global_med', np.nan):.3f} ± {em.get('chi2_nu_global_std', np.nan):.3f}")
            if "by_filter" in em_block_full:
                stats_lines.append("")
                stats_lines.append("=== Error Model by Filter ===")
                for _, row in em_block_full["by_filter"].iterrows():
                    stats_lines.append(
                        f"{row['FILTER']}: RMS/σ={row['rms_over_pred_med']:.3f}, "
                        f"χ²/ν={row['chi2_nu_med']:.3f}, "
                        f"N_stars={int(row['n_stars'])}, N_obs={int(row['n_obs_total'])}"
                    )

        if "centroid" in self.results and "summary" in self.results["centroid"]:
            ct = self.results["centroid"]["summary"]
            stats_lines.append("\n=== Centroid Quality ===")
            stats_lines.append(f"Δr (95th percentile): {ct.get('delta_r_p95', np.nan):.3f} px")
            stats_lines.append(f"Outliers > 1px: {ct.get('n_outliers_gt1px', 'N/A')}")

        if "frame_quality" in self.results and "summary" in self.results["frame_quality"]:
            fq = self.results["frame_quality"]["summary"]
            stats_lines.append("\n=== Frame Quality ===")
            stats_lines.append(f"N_frames: {fq.get('n_frames', 'N/A')}")
            stats_lines.append(f"Flagged frames: {fq.get('n_flagged', 'N/A')}")

        self.stats_text.setText("\n".join(stats_lines))

    def save_all_plots(self):
        """Save all plots to files"""
        qa_dir = self.result_dir / "qa_report"
        qa_dir.mkdir(exist_ok=True)

        try:
            if "error_model_raw" in self.results and "by_star" in self.results["error_model_raw"]:
                self._plot_error_model(self.results["error_model_raw"]["by_star"])
                self.error_figure.savefig(qa_dir / "qa_error_model_plot_raw.png", dpi=150, bbox_inches="tight")
                self.error_figure.savefig(qa_dir / "qa_error_model_plot.png", dpi=150, bbox_inches="tight")
            if "error_model_zp" in self.results and "by_star" in self.results["error_model_zp"]:
                self._plot_error_model(self.results["error_model_zp"]["by_star"])
                self.error_figure.savefig(qa_dir / "qa_error_model_plot_zp.png", dpi=150, bbox_inches="tight")
            self.centroid_figure.savefig(qa_dir / "qa_centroid_plot.png", dpi=150, bbox_inches="tight")
            self.background_figure.savefig(qa_dir / "qa_background_plot.png", dpi=150, bbox_inches="tight")

            self.log(f"Plots saved to {qa_dir}")
            QMessageBox.information(self, "Saved", f"Plots saved to:\n{qa_dir}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to save plots:\n{str(e)}")

    def export_latex(self):
        """Export LaTeX table to file"""
        if "publication" not in self.results or "latex" not in self.results["publication"]:
            QMessageBox.warning(self, "No Data", "Generate QA report first")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export LaTeX",
            str(self.result_dir / "qa_report" / "qa_table.tex"),
            "LaTeX Files (*.tex)"
        )

        if file_path:
            Path(file_path).write_text(self.results["publication"]["latex"], encoding="utf-8")
            self.log(f"LaTeX exported to {file_path}")
            QMessageBox.information(self, "Exported", f"LaTeX table saved to:\n{file_path}")
