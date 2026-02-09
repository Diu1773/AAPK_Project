"""
Extinction (Airmass Fit) Tool Window
Fits per-filter extinction coefficients using instrumental magnitudes.
"""

from __future__ import annotations

import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QTextEdit,
    QGroupBox, QMessageBox, QComboBox, QCheckBox, QSplitter, QTableWidget,
    QTableWidgetItem, QHeaderView, QDoubleSpinBox, QProgressBar, QDialog,
    QDialogButtonBox, QFormLayout, QSpinBox, QTabWidget
)

from ...utils.astro_utils import compute_airmass_from_header
from ...utils.constants import get_parallel_workers
from ...utils.photometry_utils import (
    phot_one_star as _phot_one_star,
    get_numeric_array,
    MAG_ERR_COEFF,
    MAD_TO_SIGMA,
)
from ...utils.step_paths import (
    step2_cropped_dir,
    step4_dir,
    step5_dir,
    step6_dir,
    step7_dir,
    step9_dir,
    step10_dir,
    legacy_step11_extinction_dir,
    tool_extinction_dir,
    crop_is_active,
    legacy_step5_refbuild_dir,
    legacy_step6_idmatch_dir,
    legacy_step7_wcs_dir,
    legacy_step7_refbuild_dir,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

K2_SIGNIFICANCE = 0.005        # quadratic extinction term significance threshold
KCOLOR_SIGNIFICANCE = 0.01    # color-extinction term significance threshold

# Gaia DR3 -> SDSS polynomial transforms (Evans+ 2018, A&A 616, A4)
GAIA_TO_SDSS = {
    "g": {"coeffs": [0.2199, -0.6365, -0.1548, 0.0064], "color_range": (0.3, 3.0)},
    "r": {"coeffs": [-0.09837, 0.08592, 0.1907, -0.1701, 0.02263], "color_range": (0.0, 3.0)},
    "i": {"coeffs": [-0.293, 0.6404, -0.09609, -0.002104], "color_range": (0.5, 2.0)},
}


class ExtinctionFitWorker(QThread):
    progress = pyqtSignal(int, int, str)
    log = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, params, data_dir: Path, result_dir: Path, *,
                 mode: str = "ensemble", task: str = "all", phot_df: pd.DataFrame | None = None,
                 ap_scale: float = 1.0,
                 ann_in_scale: float = 4.0, ann_out_scale: float = 2.0):
        super().__init__()
        self.params = params
        self.data_dir = Path(data_dir)
        self.result_dir = Path(result_dir)
        self.mode = mode  # "ensemble" or "gaia"
        self.task = task  # "photometry", "fit", or "all"
        self.phot_df = phot_df
        self.ap_scale = ap_scale
        self.ann_in_scale = ann_in_scale
        self.ann_out_scale = ann_out_scale
        self._stop_requested = False
        self.max_workers = get_parallel_workers(params)

    def stop(self):
        self._stop_requested = True

    def _log(self, msg: str):
        self.log.emit(msg)

    def _filter_variable_stars(
        self,
        sub: pd.DataFrame,
        scatter_col: str,
        method: str,
        sigma: float,
        min_frames: int,
        label: str = "",
    ) -> pd.DataFrame:
        """Remove variable stars based on per-star scatter in *scatter_col*.

        Returns the filtered DataFrame (or the original if nothing removed).
        """
        if sigma <= 0:
            return sub
        star_counts = sub.groupby("source_id")["file"].nunique()
        good_ids = star_counts[star_counts >= min_frames].index
        sub_var = sub[sub["source_id"].isin(good_ids)]
        if sub_var.empty:
            return sub
        if method == "std":
            star_scatter = sub_var.groupby("source_id")[scatter_col].std(ddof=1)
        else:
            star_scatter = sub_var.groupby("source_id")[scatter_col].apply(
                lambda s: float(np.nanmedian(np.abs(s - np.nanmedian(s)))))
        vals = star_scatter.to_numpy(float)
        med_s = float(np.nanmedian(vals)) if vals.size else np.nan
        mad_s = float(np.nanmedian(np.abs(vals - med_s))) if vals.size else np.nan
        sig_s = MAD_TO_SIGMA * mad_s if mad_s > 0 else float(np.nanstd(vals))
        if not (np.isfinite(sig_s) and sig_s > 0):
            return sub
        thresh = med_s + sigma * sig_s
        bad_ids = set(star_scatter[star_scatter > thresh].index)
        if bad_ids:
            before = sub["source_id"].nunique()
            sub = sub[~sub["source_id"].isin(bad_ids)].copy()
            after = sub["source_id"].nunique()
            self._log(f"{label} var-star filter ({method}, {sigma:.1f}σ): "
                      f"{before - after} stars removed (thr={thresh:.4f})")
        return sub

    @staticmethod
    def _as_float(value, default: float) -> float:
        try:
            if value is None:
                return float(default)
            v = float(value)
            return v if np.isfinite(v) else float(default)
        except Exception:
            return float(default)

    @staticmethod
    def _as_int(value, default: int) -> int:
        try:
            if value is None:
                return int(default)
            return int(value)
        except Exception:
            return int(default)

    @staticmethod
    def _robust_median_and_err(arr):
        x = np.asarray(arr, float)
        x = x[np.isfinite(x)]
        if len(x) == 0:
            return (np.nan, np.nan, 0)
        med = float(np.median(x))
        mad = float(np.median(np.abs(x - med)))
        err = float(MAD_TO_SIGMA * mad / np.sqrt(max(len(x), 1)))
        return (med, err, int(len(x)))

    def _poly_eval(self, x, coeffs):
        x = np.asarray(x, float)
        y = np.zeros_like(x, dtype=float)
        p = np.ones_like(x, dtype=float)
        for a in coeffs:
            y += a * p
            p *= x
        return y

    def _robust_linfit(self, x, y, clip_sigma=3.0, iters=5, min_n=10):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        m0 = np.isfinite(x) & np.isfinite(y)
        x = x[m0]
        y = y[m0]
        if len(x) < min_n:
            return (np.nan, np.nan, np.nan, 0, np.nan)
        k, zp = np.polyfit(x, y, 1)
        base_n = len(x)
        for _ in range(int(iters)):
            yhat = zp + k * x
            r = y - yhat
            med = np.nanmedian(r)
            mad = np.nanmedian(np.abs(r - med)) + 1e-12
            sig = MAD_TO_SIGMA * mad
            keep = np.abs(r - med) <= float(clip_sigma) * sig
            if keep.sum() < min_n:
                break
            if keep.sum() == len(x):
                break
            x, y = x[keep], y[keep]
            k, zp = np.polyfit(x, y, 1)
        yhat = zp + k * x
        scatter = float(np.nanstd(y - yhat)) if len(x) else np.nan
        outlier_frac = float(1.0 - (len(x) / max(base_n, 1)))
        return (float(k), float(zp), scatter, int(len(x)), outlier_frac)

    def _robust_weighted_linfit(self, x, y, w=None, clip_sigma=3.0, iters=5, min_n=10):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        if w is not None:
            w = np.asarray(w, float)
            m0 = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
        else:
            m0 = np.isfinite(x) & np.isfinite(y)
        x = x[m0]
        y = y[m0]
        w = w[m0] if w is not None else None
        if len(x) < min_n:
            return (np.nan, np.nan, np.nan, 0, np.nan)

        def _fit(xx, yy, ww):
            if ww is None:
                k, zp = np.polyfit(xx, yy, 1)
            else:
                k, zp = np.polyfit(xx, yy, 1, w=ww)
            return k, zp

        k, zp = _fit(x, y, w)
        base_n = len(x)
        for _ in range(int(iters)):
            yhat = zp + k * x
            r = y - yhat
            med = np.nanmedian(r)
            mad = np.nanmedian(np.abs(r - med)) + 1e-12
            sig = MAD_TO_SIGMA * mad
            keep = np.abs(r - med) <= float(clip_sigma) * sig
            if keep.sum() < min_n:
                break
            if keep.sum() == len(x):
                break
            x = x[keep]
            y = y[keep]
            if w is not None:
                w = w[keep]
            k, zp = _fit(x, y, w)

        yhat = zp + k * x
        scatter = float(np.nanstd(y - yhat)) if len(x) else np.nan
        outlier_frac = float(1.0 - (len(x) / max(base_n, 1)))
        return (float(k), float(zp), scatter, int(len(x)), outlier_frac)

    def _fit_ensemble_once(
        self,
        sub: pd.DataFrame,
        clip_sigma: float,
        iters: int,
        min_n: int,
    ):
        """Fit m_ij = s_i + z_j + k1*X_j for one date+filter group."""
        if sub.empty:
            return None

        sub = sub.reset_index(drop=True)
        x = get_numeric_array(sub, "airmass")
        y = get_numeric_array(sub, "mag_inst")
        star_ids = pd.to_numeric(sub["source_id"], errors="coerce").astype("int64").to_numpy()
        frame_ids = sub["file"].astype(str).to_numpy()

        base_mask = np.isfinite(x) & np.isfinite(y)
        if base_mask.sum() < min_n:
            return None

        unique_stars = np.unique(star_ids[base_mask])
        unique_frames = np.unique(frame_ids[base_mask])
        n_star = len(unique_stars)
        n_frame = len(unique_frames)
        if n_star == 0 or n_frame == 0:
            return None

        star_index = {sid: i for i, sid in enumerate(unique_stars)}
        frame_index = {fid: i for i, fid in enumerate(unique_frames)}
        ref_frame = unique_frames[0]
        ref_idx = frame_index[ref_frame]

        star_idx = np.array([star_index[s] for s in star_ids], dtype=int)
        frame_idx = np.array([frame_index[f] for f in frame_ids], dtype=int)

        # SNR-based weights (if available)
        has_snr = "snr" in sub.columns
        if has_snr:
            snr_arr = get_numeric_array(sub, "snr", default=1.0)
            sig_mag = MAG_ERR_COEFF / np.clip(snr_arr, 1e-6, None)
            w_all = 1.0 / np.clip(sig_mag ** 2, 1e-12, None)
        else:
            w_all = np.ones(len(sub), dtype=float)

        def build_matrix(mask):
            idx = np.where(mask)[0]
            n_obs = len(idx)
            n_params = n_star + (n_frame - 1) + 1
            A = np.zeros((n_obs, n_params), dtype=float)
            rows = np.arange(n_obs)
            A[rows, star_idx[idx]] = 1.0
            if n_frame > 1:
                fidx = frame_idx[idx]
                m = fidx != ref_idx
                if m.any():
                    fcol = fidx[m].copy()
                    fcol = np.where(fcol > ref_idx, fcol - 1, fcol)
                    A[rows[m], n_star + fcol] = 1.0
            A[:, -1] = x[idx]
            return A, y[idx], idx

        mask_fit = base_mask.copy()
        base_n = int(mask_fit.sum())
        if base_n < min_n:
            return None

        for _ in range(int(iters)):
            A, yv, idx = build_matrix(mask_fit)
            if len(yv) < min_n:
                break
            W = np.sqrt(w_all[idx])
            coef, _, _, _ = np.linalg.lstsq(W[:, None] * A, W * yv, rcond=None)
            r = yv - (A @ coef)
            med = np.nanmedian(r)
            mad = np.nanmedian(np.abs(r - med)) + 1e-12
            sig = MAD_TO_SIGMA * mad
            keep = np.abs(r - med) <= float(clip_sigma) * sig
            if keep.sum() < min_n:
                break
            if keep.sum() == len(yv):
                break
            new_mask = mask_fit.copy()
            new_mask[idx] = keep
            mask_fit = new_mask

        A, yv, idx = build_matrix(mask_fit)
        if len(yv) < min_n:
            return None
        W = np.sqrt(w_all[idx])
        coef, _, _, _ = np.linalg.lstsq(W[:, None] * A, W * yv, rcond=None)

        s_vals = coef[:n_star]
        z_vals = np.zeros(n_frame, dtype=float)
        if n_frame > 1:
            frame_cols = coef[n_star:-1]
            for fi in range(n_frame):
                if fi == ref_idx:
                    z_vals[fi] = 0.0
                else:
                    col = fi if fi < ref_idx else fi - 1
                    z_vals[fi] = frame_cols[col]
        k1 = float(coef[-1])

        # residuals for all points in sub
        yhat = s_vals[star_idx] + z_vals[frame_idx] + k1 * x
        resid = y - yhat
        delta_m = y - (s_vals[star_idx] + z_vals[frame_idx])
        scatter = float(np.nanstd(resid[base_mask])) if base_n else np.nan
        outlier_frac = float(1.0 - (mask_fit.sum() / max(base_n, 1)))

        return {
            "k1": k1,
            "s_vals": s_vals,
            "z_vals": z_vals,
            "star_index": star_index,
            "frame_index": frame_index,
            "ref_frame": ref_frame,
            "resid": resid,
            "delta_m": delta_m,
            "scatter": scatter,
            "n_used": int(mask_fit.sum()),
            "outlier_frac": outlier_frac,
            "n_star": n_star,
            "n_frame": n_frame,
        }

    def _robust_quadfit(self, x, y, clip_sigma=3.0, iters=5, min_n=15):
        """2차 다항식 피팅: y = k2*x^2 + k1*x + zp"""
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        m0 = np.isfinite(x) & np.isfinite(y)
        x = x[m0]
        y = y[m0]
        if len(x) < min_n:
            return (np.nan, np.nan, np.nan, np.nan, 0, np.nan, False)
        try:
            coeffs = np.polyfit(x, y, 2)  # [k2, k1, zp]
            k2, k1, zp = coeffs
        except Exception:
            return (np.nan, np.nan, np.nan, np.nan, 0, np.nan, False)
        base_n = len(x)
        for _ in range(int(iters)):
            yhat = zp + k1 * x + k2 * x * x
            r = y - yhat
            med = np.nanmedian(r)
            mad = np.nanmedian(np.abs(r - med)) + 1e-12
            sig = MAD_TO_SIGMA * mad
            keep = np.abs(r - med) <= float(clip_sigma) * sig
            if keep.sum() < min_n:
                break
            if keep.sum() == len(x):
                break
            x, y = x[keep], y[keep]
            try:
                coeffs = np.polyfit(x, y, 2)
                k2, k1, zp = coeffs
            except Exception:
                break
        yhat = zp + k1 * x + k2 * x * x
        scatter = float(np.nanstd(y - yhat)) if len(x) else np.nan
        outlier_frac = float(1.0 - (len(x) / max(base_n, 1)))
        # k2 significance check
        k2_significant = abs(k2) > K2_SIGNIFICANCE if np.isfinite(k2) else False
        return (float(k1), float(k2), float(zp), scatter, int(len(x)), outlier_frac, k2_significant)

    def _robust_color_extinction_fit(self, X, C, y, clip_sigma=3.0, iters=5, min_n=15):
        """
        색 의존 소광 피팅: delta = k' * X + k'' * C * X + zp

        Parameters:
        - X: airmass 배열
        - C: 색지수 배열 (예: g-r, BP-RP)
        - y: delta (m_ref - m_inst) 배열

        Returns:
        - k_prime: 평균 소광 계수
        - k_double_prime: 색 의존 소광 계수
        - zp: 제로포인트
        - scatter, n_used, outlier_frac, significant
        """
        X = np.asarray(X, float)
        C = np.asarray(C, float)
        y = np.asarray(y, float)

        m0 = np.isfinite(X) & np.isfinite(C) & np.isfinite(y)
        X = X[m0]
        C = C[m0]
        y = y[m0]

        if len(X) < min_n:
            return (np.nan, np.nan, np.nan, np.nan, 0, np.nan, False)

        # 다중 선형 회귀: y = a0 + a1*X + a2*(C*X)
        # Design matrix: [1, X, C*X]
        try:
            A = np.column_stack([np.ones_like(X), X, C * X])
            coeffs, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)
            zp, k_prime, k_double_prime = coeffs
        except Exception:
            return (np.nan, np.nan, np.nan, np.nan, 0, np.nan, False)

        base_n = len(X)
        for _ in range(int(iters)):
            yhat = zp + k_prime * X + k_double_prime * C * X
            r = y - yhat
            med = np.nanmedian(r)
            mad = np.nanmedian(np.abs(r - med)) + 1e-12
            sig = MAD_TO_SIGMA * mad
            keep = np.abs(r - med) <= float(clip_sigma) * sig
            if keep.sum() < min_n:
                break
            if keep.sum() == len(X):
                break
            X, C, y = X[keep], C[keep], y[keep]
            try:
                A = np.column_stack([np.ones_like(X), X, C * X])
                coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
                zp, k_prime, k_double_prime = coeffs
            except Exception:
                break

        yhat = zp + k_prime * X + k_double_prime * C * X
        scatter = float(np.nanstd(y - yhat)) if len(X) else np.nan
        outlier_frac = float(1.0 - (len(X) / max(base_n, 1)))

        # k''가 유의미한지 체크
        k_double_significant = abs(k_double_prime) > KCOLOR_SIGNIFICANCE if np.isfinite(k_double_prime) else False

        return (float(k_prime), float(k_double_prime), float(zp), scatter, int(len(X)), outlier_frac, k_double_significant)

    # ------------------------------------------------------------------
    # Ensemble mode: allstar photometry + median-subtraction fit
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_idmatch(result_dir: Path, fname: str) -> Path | None:
        """Locate idmatch_{fname}.csv from step 7."""
        import re
        s7 = step7_dir(result_dir)
        # date-keyed subdirectory
        m = re.search(r"(20\d{6})", str(fname))
        if m:
            dated = s7 / m.group(1) / f"idmatch_{fname}.csv"
            if dated.exists():
                return dated
        direct = s7 / f"idmatch_{fname}.csv"
        if direct.exists():
            return direct
        # glob fallback
        hits = list(s7.glob(f"*/idmatch_{fname}.csv"))
        if hits:
            return hits[0]
        # legacy step6_idmatch
        leg = legacy_step6_idmatch_dir(result_dir)
        lp = leg / f"idmatch_{fname}.csv"
        if lp.exists():
            return lp
        hits2 = list(leg.glob(f"*/idmatch_{fname}.csv"))
        return hits2[0] if hits2 else None

    @staticmethod
    def _load_fwhm_meta(cache_dir: Path, fname: str, default: float = 6.0) -> float:
        """Load median FWHM from detection metadata."""
        import json as _json
        meta = cache_dir / f"detect_{fname}.json"
        if meta.exists():
            try:
                d = _json.loads(meta.read_text(encoding="utf-8"))
                for k in ("fwhm_med_rad_px", "fwhm_med_px", "fwhm_px", "fwhm_med"):
                    v = d.get(k)
                    if v is not None:
                        v = float(v)
                        if np.isfinite(v) and v > 0:
                            return v
            except Exception:
                pass
        return default

    def _run_allstar_photometry(self) -> pd.DataFrame:
        """Built-in allstar aperture photometry using step 7 ID-match data.

        Returns DataFrame with columns:
            file, source_id, ID, mag_inst, snr, airmass, date, filter
        """
        P = self.params.P
        result_dir = self.result_dir

        gain = self._as_float(getattr(P, "gain_e_per_adu", 1.0), 1.0)
        sat_adu = self._as_float(getattr(P, "saturation_adu", 60000.0), 60000.0)
        fwhm_default = self._as_float(getattr(P, "fwhm_pix_guess", 6.0), 6.0)
        ann_sigma = self._as_float(getattr(P, "annulus_sigma_clip", 3.0), 3.0)

        lat = self._as_float(getattr(P, "site_lat_deg", 0.0), 0.0)
        lon = self._as_float(getattr(P, "site_lon_deg", 0.0), 0.0)
        alt = self._as_float(getattr(P, "site_alt_m", 0.0), 0.0)
        tz = self._as_float(getattr(P, "site_tz_offset_hours", 0.0), 0.0)

        use_cropped = crop_is_active(result_dir)
        cache_dir = result_dir / "cache"
        if not cache_dir.exists():
            cache_dir = result_dir

        # Discover frames: prefer photometry_index.csv, fall back to step7 idmatch files
        file_list = []
        filter_map: dict[str, str] = {}

        idx_path = step9_dir(result_dir) / "photometry_index.csv"
        if not idx_path.exists():
            idx_path = result_dir / "photometry_index.csv"
        if idx_path.exists():
            idx = pd.read_csv(idx_path)
            if "file" in idx.columns:
                file_list = idx["file"].astype(str).tolist()
            if "filter" in idx.columns:
                for _, r in idx.iterrows():
                    filter_map[str(r["file"])] = str(r["filter"]).strip().lower()

        if not file_list:
            # Fall back: scan idmatch files from step7
            s7 = step7_dir(result_dir)
            if s7.exists():
                for p in sorted(s7.rglob("idmatch_*.csv")):
                    fname = p.stem.replace("idmatch_", "")
                    if fname not in file_list:
                        file_list.append(fname)

        if not file_list:
            raise FileNotFoundError("No frames found for allstar photometry")

        self._log(f"Allstar photometry: {len(file_list)} frames")

        # Load frame quality if available (step 4/5 QC)
        qc_exclude: set[str] = set()
        for qp in [step5_dir(result_dir) / "frame_quality.csv",
                    legacy_step7_wcs_dir(result_dir) / "frame_quality.csv",
                    result_dir / "frame_quality.csv"]:
            if qp.exists():
                try:
                    dfq = pd.read_csv(qp)
                    if "passed" in dfq.columns and "file" in dfq.columns:
                        qc_exclude = set(dfq.loc[dfq["passed"] == False, "file"].astype(str))
                        self._log(f"Frame QC: {len(qc_exclude)} frames excluded by step4/5 QC")
                except Exception:
                    pass
                break

        cached_df = None
        cached_files: set[str] = set()
        if isinstance(self.phot_df, pd.DataFrame) and not self.phot_df.empty and "file" in self.phot_df.columns:
            cached_df = self.phot_df.copy()
            cached_df["file"] = cached_df["file"].astype(str)
            valid_files = set(file_list) - set(qc_exclude)
            cached_df = cached_df[cached_df["file"].isin(valid_files)]
            cached_files = set(cached_df["file"].unique())
            self._log(f"Cached photometry: {len(cached_files)} frames loaded")

        rows: list[dict] = []
        files_to_process = [f for f in file_list if f not in qc_exclude and f not in cached_files]
        total = len(files_to_process)
        if total == 0:
            if cached_df is not None:
                self._log("All frames covered by cache; skipping photometry")
                return cached_df
            raise RuntimeError("All frames excluded by QC")

        max_workers = min(self.max_workers, total) if total > 0 else 1
        if max_workers <= 1 or total <= 1:
            for fi, fname in enumerate(files_to_process, 1):
                if self._stop_requested:
                    break
                frame_rows, n_good = self._process_one_frame(
                    fname, filter_map, use_cropped, cache_dir,
                    lat, lon, alt, tz,
                    fwhm_default, ann_sigma,
                    gain, sat_adu,
                )
                if frame_rows:
                    rows.extend(frame_rows)
                self.progress.emit(fi, total, fname)
                if fi % 20 == 0 or fi == total:
                    self._log(f"[{fi}/{total}] {fname}: {n_good} stars measured")
        else:
            self._log(f"Parallel photometry: {max_workers} workers")
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                for fname in files_to_process:
                    futures[executor.submit(
                        self._process_one_frame,
                        fname, filter_map, use_cropped, cache_dir,
                        lat, lon, alt, tz,
                        fwhm_default, ann_sigma,
                        gain, sat_adu,
                    )] = fname

                done = 0
                for fut in as_completed(futures):
                    if self._stop_requested:
                        break
                    fname = futures[fut]
                    try:
                        frame_rows, n_good = fut.result()
                    except Exception as exc:
                        self._log(f"  [WARN] Frame {fname} failed: {exc}")
                        frame_rows, n_good = [], 0
                    if frame_rows:
                        rows.extend(frame_rows)
                    done += 1
                    self.progress.emit(done, total, fname)
                    if done % 20 == 0 or done == total:
                        self._log(f"[{done}/{total}] {fname}: {n_good} stars measured")

        if not rows:
            raise RuntimeError("Allstar photometry produced no valid measurements")

        df_new = pd.DataFrame(rows)
        if cached_df is not None:
            df = pd.concat([cached_df, df_new], ignore_index=True)
        else:
            df = df_new
        self._log(f"Allstar photometry complete: {len(df)} measurements, "
                  f"{df['source_id'].nunique()} unique stars, {df['file'].nunique()} frames")
        return df

    def _process_one_frame(
        self,
        fname: str,
        filter_map: dict[str, str],
        use_cropped: bool,
        cache_dir: Path,
        lat: float,
        lon: float,
        alt: float,
        tz: float,
        fwhm_default: float,
        ann_sigma: float,
        gain: float,
        sat_adu: float,
    ) -> tuple[list[dict], int]:
        if self._stop_requested:
            return ([], 0)

        result_dir = self.result_dir

        # Resolve FITS path
        if use_cropped:
            fpath = step2_cropped_dir(result_dir) / fname
        else:
            fpath = self.params.get_file_path(fname) if hasattr(self.params, "get_file_path") else None
        if fpath is None or not fpath.exists():
            fpath = step2_cropped_dir(result_dir) / fname
        if not fpath.exists():
            return ([], 0)

        # Load idmatch
        idm_path = self._resolve_idmatch(result_dir, fname)
        if idm_path is None or not idm_path.exists():
            return ([], 0)

        try:
            df_id = pd.read_csv(idm_path)
        except Exception:
            return ([], 0)
        if "source_id" not in df_id.columns or "x" not in df_id.columns:
            return ([], 0)

        # Keep only matched stars (source_id is not NaN)
        df_id = df_id.dropna(subset=["source_id", "x", "y"])
        if df_id.empty:
            return ([], 0)

        # Load image
        try:
            img = fits.getdata(fpath).astype(float)
            hdr = fits.getheader(fpath)
        except Exception:
            return ([], 0)

        # Airmass & date
        try:
            am_info = compute_airmass_from_header(hdr, lat, lon, alt, tz)
            airmass = float(am_info.get("airmass", np.nan))
        except Exception:
            airmass = np.nan
        obs_date = self._extract_date_from_file(fname, hdr)
        filt = filter_map.get(fname, str(hdr.get("FILTER", "")).strip().lower())

        # FWHM & aperture params
        fwhm = float(self._load_fwhm_meta(cache_dir, fname, fwhm_default))
        r_ap = self.ap_scale * fwhm
        r_in = max(self.ann_in_scale * fwhm, r_ap)
        r_out = r_in + self.ann_out_scale * fwhm

        h, w = img.shape
        n_good = 0
        rows: list[dict] = []
        for _, star in df_id.iterrows():
            xc = float(star["x"])
            yc = float(star["y"])
            sid = int(star["source_id"])

            # Skip stars too close to edge
            if xc - r_out < 0 or xc + r_out >= w or yc - r_out < 0 or yc + r_out >= h:
                continue

            rdnoise = float(getattr(self.params.P, "rdnoise_e", 7.5) or 7.5)
            result = _phot_one_star(
                img, xc, yc, r_ap, r_in, r_out,
                sigma_clip_val=ann_sigma, gain=gain,
                rn_param_e=rdnoise, sat_adu=sat_adu,
            )
            flux_e, _sigma_e, snr = result[0], result[1], result[2]
            is_sat = result[8]
            if is_sat or flux_e <= 0 or not np.isfinite(flux_e):
                continue
            mag_inst = float(-2.5 * np.log10(flux_e))
            if not np.isfinite(mag_inst):
                continue

            fid = int(star.get("ID", star.get("det_idx", 0))) if "ID" in star.index else 0
            rows.append({
                "file": fname,
                "source_id": sid,
                "ID": fid,
                "mag_inst": mag_inst,
                "snr": snr,
                "airmass": airmass,
                "date": obs_date,
                "filter": filt,
            })
            n_good += 1

        return (rows, n_good)

    def _run_ensemble_fit(self, phot_df: pd.DataFrame):
        """Ensemble extinction fit with frame zeropoint separation.

        Model per date+filter:
            m_ij = s_i + z_j + k1 * X_j + eps
        where s_i is star constant, z_j is frame zeropoint, and k1 is extinction.
        """
        P = self.params.P
        result_dir = self.result_dir
        clip_sigma = self._as_float(getattr(P, "zp_clip_sigma", 3.0), 3.0)
        fit_iters = self._as_int(getattr(P, "zp_fit_iters", 5), 5)
        min_match = self._as_int(getattr(P, "min_master_gaia_matches", 10), 10)
        snr_cut = self._as_float(getattr(P, "extinction_snr_min", 10.0), 10.0)

        df = phot_df.copy()

        # SNR filter
        if "snr" in df.columns:
            n_before = len(df)
            df = df[df["snr"] >= snr_cut]
            self._log(f"SNR filter (>={snr_cut}): {n_before} → {len(df)}")

        # Remove stars with too few frames (< 3)
        star_counts = df.groupby("source_id")["file"].nunique()
        good_stars = set(star_counts[star_counts >= 3].index)
        df = df[df["source_id"].isin(good_stars)]
        self._log(f"Stars with >=3 frames: {len(good_stars)}")

        if df.empty:
            raise RuntimeError("No data after quality filters")

        # Also reject frames with no valid airmass
        n_before_am = len(df)
        nan_am_frames = df.loc[~np.isfinite(df["airmass"]), "file"].nunique()
        df = df[np.isfinite(df["airmass"])].copy()
        if nan_am_frames > 0:
            self._log(f"Airmass NaN: dropped {n_before_am - len(df)} points from {nan_am_frames} frames")
        if df.empty:
            raise RuntimeError("No data after airmass filtering (all frames had NaN airmass)")
        self._log(f"After SNR/airmass filters: {len(df)} points, {df['file'].nunique()} frames, "
                  f"{df['source_id'].nunique()} stars")

        # Step 3: fit per date+filter
        fit_rows = []
        point_rows = []

        dates_seen = sorted(df["date"].unique())
        filters_seen = sorted(df["filter"].unique())
        self._log(f"Dates: {dates_seen}")
        self._log(f"Filters: {filters_seen}")

        var_method_raw = getattr(P, "extinction_varstar_method", "mad")
        var_method = str(var_method_raw if var_method_raw else "mad").strip().lower()
        var_sigma = self._as_float(getattr(P, "extinction_varstar_sigma", 3.0), 3.0)
        var_min_frames = self._as_int(getattr(P, "extinction_varstar_min_frames", 5), 5)

        qc_method_raw = getattr(P, "extinction_frame_qc_method", "mad")
        qc_method = str(qc_method_raw if qc_method_raw else "mad").strip().lower()
        qc_sigma = self._as_float(getattr(P, "extinction_frame_qc_sigma", 3.0), 3.0)

        groups = list(df.groupby(["date", "filter"]))
        n_groups = len(groups)
        for i_grp, ((date_val, filt), sub) in enumerate(groups):
            self.progress.emit(i_grp, n_groups, f"Ensemble {date_val}/{filt}")
            sub = sub.copy()
            n_pts = len(sub)

            use_dx = bool(getattr(P, "extinction_delta_x_enable", True))
            dx_min = self._as_float(getattr(P, "extinction_delta_x_min", 0.3), 0.3)
            if use_dx:
                x = get_numeric_array(sub, "airmass")
                dx = float(np.nanmax(x) - np.nanmin(x)) if np.isfinite(x).any() else np.nan
                if dx < dx_min:
                    self._log(f"[{date_val}][{filt}] skipped: ΔX={dx:.3f} < {dx_min:.3f}")
                    continue

            if n_pts < min_match:
                self._log(f"[{date_val}][{filt}] skipped: only {n_pts} points (min={min_match})")
                continue

            fit = self._fit_ensemble_once(sub, clip_sigma, fit_iters, min_match)
            if fit is None or not np.isfinite(fit.get("k1", np.nan)):
                self._log(f"[{date_val}][{filt}] FAILED: fit did not converge")
                continue

            sub["resid"] = fit["resid"]
            sub["delta_m"] = fit["delta_m"]

            # Variable star filter (residual-based)
            if var_sigma > 0:
                n_before = sub["source_id"].nunique()
                sub = self._filter_variable_stars(
                    sub, "resid", var_method, var_sigma, var_min_frames,
                    label=f"[{date_val}][{filt}]")
                if sub["source_id"].nunique() < n_before:
                    fit = self._fit_ensemble_once(sub, clip_sigma, fit_iters, min_match)
                    if fit is None:
                        continue
                    sub["resid"] = fit["resid"]
                    sub["delta_m"] = fit["delta_m"]

            # Frame QC (residual median)
            if qc_sigma > 0:
                frame_stats = sub.groupby("file").agg(
                    resid_med=("resid", "median"),
                    resid_mad=("resid", lambda s: float(np.nanmedian(np.abs(s - np.nanmedian(s))))),
                    n_stars=("resid", "count"),
                )
                if qc_method == "std":
                    global_med = float(np.nanmean(frame_stats["resid_med"]))
                    global_sig = float(np.nanstd(frame_stats["resid_med"]))
                else:
                    global_med = float(np.nanmedian(frame_stats["resid_med"]))
                    global_mad = float(np.nanmedian(np.abs(frame_stats["resid_med"] - global_med)))
                    global_sig = MAD_TO_SIGMA * global_mad if global_mad > 0 else float(np.nanstd(frame_stats["resid_med"]))
                if global_sig > 0:
                    bad_mask = np.abs(frame_stats["resid_med"] - global_med) > qc_sigma * global_sig
                    bad_frames = set(frame_stats.index[bad_mask])
                    if bad_frames:
                        self._log(f"[{date_val}][{filt}] Frame QC ({qc_method}, {qc_sigma:.1f}σ): "
                                  f"{len(bad_frames)} frames removed")
                        sub = sub[~sub["file"].isin(bad_frames)].copy()
                        fit = self._fit_ensemble_once(sub, clip_sigma, fit_iters, min_match)
                        if fit is None:
                            continue
                        sub["resid"] = fit["resid"]
                        sub["delta_m"] = fit["delta_m"]

            if sub.empty:
                continue

            k1 = float(fit["k1"])
            scatter = float(fit.get("scatter", np.nan))
            n_used = int(fit.get("n_used", len(sub)))
            out_frac = float(fit.get("outlier_frac", np.nan))
            n_star = int(sub["source_id"].nunique())
            n_frame = int(sub["file"].nunique())
            m0 = float(np.nanmean(fit.get("z_vals"))) if fit.get("z_vals") is not None else np.nan

            fit_rows.append({
                "date": date_val,
                "filter": filt,
                "k1": k1,
                "k2": 0.0,
                "zp": m0,
                "m0": m0,
                "scatter": scatter,
                "n_total": n_pts,
                "n_used": n_used,
                "outlier_fraction": out_frac,
                "fit_order": 1,
                "n_stars": n_star,
                "n_frames": n_frame,
                "method": "ensemble",
            })

            for _, r in sub.iterrows():
                point_rows.append({
                    "date": date_val, "filter": filt,
                    "airmass": float(r["airmass"]),
                    "delta_m": float(r["delta_m"]),
                    "resid": float(r["resid"]),
                })

            self._log(f"[{date_val}][{filt}] k1={k1:.5f}, scatter={scatter:.4f}, "
                      f"n={n_used}/{n_pts} ({n_star} stars, {n_frame} frames)")

        if not fit_rows:
            raise RuntimeError("No valid ensemble extinction fits produced")

        fit_df = pd.DataFrame(fit_rows)
        points_df = pd.DataFrame(point_rows)

        # Save results
        out_dir = tool_extinction_dir(self.result_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fit_path = out_dir / "ensemble_extinction_by_filter.csv"
        pts_path = out_dir / "ensemble_phot_points.csv"
        fit_df.to_csv(fit_path, index=False)
        points_df.to_csv(pts_path, index=False)
        self._log(f"Saved {fit_path.name} ({len(fit_df)} rows)")
        self._log(f"Saved {pts_path.name} ({len(points_df)} rows)")

        # Backward-compatible filename for downstream loaders
        legacy_fit = out_dir / "extinction_fit_by_filter.csv"
        try:
            fit_df.to_csv(legacy_fit, index=False)
            self._log(f"Saved {legacy_fit.name} ({len(fit_df)} rows)")
        except Exception:
            self._log("Warning: failed to write legacy extinction_fit_by_filter.csv")

        # Also save the raw allstar photometry
        raw_path = out_dir / "ensemble_allstar_phot.csv"
        phot_df.to_csv(raw_path, index=False)
        self._log(f"Saved {raw_path.name} ({len(phot_df)} rows)")

        self.finished.emit({
            "fit": fit_df, "points": points_df,
            "mode": "ensemble",
        })

    def _run_per_star_fit(self, phot_df: pd.DataFrame):
        """Per-star Bouguer line fitting (k_i), then aggregate to k1."""
        P = self.params.P
        clip_sigma = self._as_float(getattr(P, "zp_clip_sigma", 3.0), 3.0)
        fit_iters = self._as_int(getattr(P, "zp_fit_iters", 5), 5)
        min_points = self._as_int(getattr(P, "min_master_gaia_matches", 10), 10)
        snr_cut = self._as_float(getattr(P, "extinction_snr_min", 10.0), 10.0)

        star_min_frames = self._as_int(getattr(P, "extinction_star_min_frames", 8), 8)
        star_rms_max = self._as_float(getattr(P, "extinction_star_rms_max", 0.10), 0.10)
        star_snr_med_min = self._as_float(getattr(P, "extinction_star_snr_med_min", 10.0), 10.0)
        min_good_stars = self._as_int(getattr(P, "extinction_min_good_stars", 3), 3)
        use_weights = bool(getattr(P, "extinction_star_use_weights", True))

        use_dx = bool(getattr(P, "extinction_delta_x_enable", True))
        dx_min_global = self._as_float(getattr(P, "extinction_delta_x_min", 0.3), 0.3)

        var_method_raw = getattr(P, "extinction_varstar_method", "mad")
        var_method = str(var_method_raw if var_method_raw else "mad").strip().lower()
        var_sigma = self._as_float(getattr(P, "extinction_varstar_sigma", 3.0), 3.0)
        var_min_frames = self._as_int(getattr(P, "extinction_varstar_min_frames", 5), 5)

        df = phot_df.copy()
        if "snr" in df.columns:
            n_before = len(df)
            df = df[df["snr"] >= snr_cut]
            self._log(f"SNR filter (>={snr_cut}): {n_before} → {len(df)}")

        nan_am = ~np.isfinite(df["airmass"])
        nan_mag = ~np.isfinite(df["mag_inst"])
        n_drop = int((nan_am | nan_mag).sum())
        if n_drop > 0:
            self._log(f"Dropped {n_drop} points with NaN airmass/mag_inst "
                      f"({int(nan_am.sum())} airmass, {int(nan_mag.sum())} mag)")
        df = df[~nan_am & ~nan_mag].copy()
        if df.empty:
            raise RuntimeError("No data after SNR/airmass filters")

        fit_rows = []
        point_rows = []

        groups = list(df.groupby(["date", "filter"]))
        n_groups = len(groups)
        for i_grp, ((date_val, filt), sub) in enumerate(groups):
            self.progress.emit(i_grp, n_groups, f"Per-star {date_val}/{filt}")
            sub = sub.copy()
            n_pts = len(sub)

            if use_dx:
                x_all = get_numeric_array(sub, "airmass")
                if np.isfinite(x_all).any():
                    dx_all = float(np.nanmax(x_all) - np.nanmin(x_all))
                    if dx_all < dx_min_global:
                        self._log(f"[{date_val}][{filt}] skipped: ΔX={dx_all:.3f} < {dx_min_global:.3f}")
                        continue

            if n_pts < min_points:
                self._log(f"[{date_val}][{filt}] skipped: only {n_pts} points (min={min_points})")
                continue

            # Variable star pre-filter (mag_inst scatter)
            if var_sigma > 0:
                sub = self._filter_variable_stars(
                    sub, "mag_inst", var_method, var_sigma, var_min_frames,
                    label=f"[{date_val}][{filt}]")

            k_list = []
            m0_map: dict[int, float] = {}
            star_rms: dict[int, float] = {}
            stats = {"total": 0, "frames": 0, "snr": 0, "fit": 0, "rms": 0}

            for sid, ssub in sub.groupby("source_id"):
                stats["total"] += 1
                n_i = len(ssub)
                if n_i < star_min_frames:
                    continue
                stats["frames"] += 1

                if "snr" in ssub.columns:
                    med_snr = float(np.nanmedian(get_numeric_array(ssub, "snr")))
                    if np.isfinite(star_snr_med_min) and star_snr_med_min > 0 and med_snr < star_snr_med_min:
                        continue
                stats["snr"] += 1

                x = get_numeric_array(ssub, "airmass")
                y = get_numeric_array(ssub, "mag_inst")
                if not np.isfinite(x).any() or not np.isfinite(y).any():
                    continue

                dx_i = float(np.nanmax(x) - np.nanmin(x))

                w = None
                if use_weights and "snr" in ssub.columns:
                    snr = get_numeric_array(ssub, "snr")
                    sig_mag = MAG_ERR_COEFF / np.clip(snr, 1e-6, None)
                    w = 1.0 / np.clip(sig_mag ** 2, 1e-12, None)

                k_i, m0_i, rms_i, n_used, _ = self._robust_weighted_linfit(
                    x, y, w=w, clip_sigma=clip_sigma, iters=fit_iters, min_n=star_min_frames
                )
                if not np.isfinite(k_i):
                    continue
                stats["fit"] += 1
                if np.isfinite(star_rms_max) and star_rms_max > 0 and np.isfinite(rms_i) and rms_i > star_rms_max:
                    continue
                stats["rms"] += 1

                k_list.append(float(k_i))
                m0_map[int(sid)] = float(m0_i)
                star_rms[int(sid)] = float(rms_i) if np.isfinite(rms_i) else np.nan

            if len(k_list) < min_good_stars:
                dx_note = f"ΔX≥{dx_min_global:.2f}" if use_dx else "ΔX off"
                self._log(
                    f"[{date_val}][{filt}] skipped: good stars={len(k_list)} < {min_good_stars} "
                    f"(min_frames={star_min_frames}, {dx_note}, RMS≤{star_rms_max:.2f}, "
                    f"medianSNR≥{star_snr_med_min:.1f}) | "
                    f"stats: total={stats['total']}, frames_ok={stats['frames']}, "
                    f"snr_ok={stats['snr']}, fit_ok={stats['fit']}, rms_ok={stats['rms']}"
                )
                continue

            k_arr = np.asarray(k_list, float)
            k1 = float(np.nanmedian(k_arr))
            k_mad = float(np.nanmedian(np.abs(k_arr - k1)))
            k1_err = float(MAD_TO_SIGMA * k_mad / np.sqrt(max(len(k_arr), 1)))

            # Build points for plotting using global k1
            good_ids = set(m0_map.keys())
            sub_good = sub[sub["source_id"].isin(good_ids)].copy()
            sub_good["m0_i"] = sub_good["source_id"].map(m0_map)
            sub_good["delta_m"] = sub_good["mag_inst"] - sub_good["m0_i"]
            sub_good["resid"] = sub_good["mag_inst"] - (sub_good["m0_i"] + k1 * sub_good["airmass"])

            for _, r in sub_good.iterrows():
                point_rows.append({
                    "date": date_val, "filter": filt,
                    "airmass": float(r["airmass"]),
                    "delta_m": float(r["delta_m"]),
                    "resid": float(r["resid"]),
                })

            fit_rows.append({
                "date": date_val,
                "filter": filt,
                "k1": k1,
                "k1_err": k1_err,
                "k2": 0.0,
                "zp": np.nan,
                "m0": np.nan,
                "scatter": float(np.nanmedian(list(star_rms.values()))) if star_rms else np.nan,
                "n_total": n_pts,
                "n_used": int(len(sub_good)),
                "outlier_fraction": float(1.0 - (len(sub_good) / max(n_pts, 1))),
                "fit_order": 1,
                "n_stars": int(len(k_list)),
                "n_frames": int(sub_good["file"].nunique()),
                "method": "per_star",
            })

            self._log(f"[{date_val}][{filt}] k1={k1:.5f} ±{k1_err:.5f}, "
                      f"stars={len(k_list)}, frames={int(sub_good['file'].nunique())}")

        if not fit_rows:
            raise RuntimeError("No valid per-star extinction fits produced")

        fit_df = pd.DataFrame(fit_rows)
        points_df = pd.DataFrame(point_rows)

        # Save results
        out_dir = tool_extinction_dir(self.result_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fit_path = out_dir / "ensemble_extinction_by_filter.csv"
        pts_path = out_dir / "ensemble_phot_points.csv"
        fit_df.to_csv(fit_path, index=False)
        points_df.to_csv(pts_path, index=False)
        self._log(f"Saved {fit_path.name} ({len(fit_df)} rows)")
        self._log(f"Saved {pts_path.name} ({len(points_df)} rows)")

        # Backward-compatible filename
        legacy_fit = out_dir / "extinction_fit_by_filter.csv"
        try:
            fit_df.to_csv(legacy_fit, index=False)
            self._log(f"Saved {legacy_fit.name} ({len(fit_df)} rows)")
        except Exception:
            self._log("Warning: failed to write legacy extinction_fit_by_filter.csv")

        self.finished.emit({
            "fit": fit_df, "points": points_df,
            "mode": "per_star",
        })

    def _run_median_fit(self, phot_df: pd.DataFrame):
        """Median-subtract Δm vs X fit (legacy)."""
        P = self.params.P
        clip_sigma = self._as_float(getattr(P, "zp_clip_sigma", 3.0), 3.0)
        fit_iters = self._as_int(getattr(P, "zp_fit_iters", 5), 5)
        min_points = self._as_int(getattr(P, "min_master_gaia_matches", 10), 10)
        snr_cut = self._as_float(getattr(P, "extinction_snr_min", 10.0), 10.0)
        use_weights = bool(getattr(P, "extinction_star_use_weights", True))

        use_dx = bool(getattr(P, "extinction_delta_x_enable", True))
        dx_min_global = self._as_float(getattr(P, "extinction_delta_x_min", 0.3), 0.3)

        var_method_raw = getattr(P, "extinction_varstar_method", "mad")
        var_method = str(var_method_raw if var_method_raw else "mad").strip().lower()
        var_sigma = self._as_float(getattr(P, "extinction_varstar_sigma", 3.0), 3.0)
        var_min_frames = self._as_int(getattr(P, "extinction_varstar_min_frames", 5), 5)

        qc_method_raw = getattr(P, "extinction_frame_qc_method", "mad")
        qc_method = str(qc_method_raw if qc_method_raw else "mad").strip().lower()
        qc_sigma = self._as_float(getattr(P, "extinction_frame_qc_sigma", 3.0), 3.0)

        df = phot_df.copy()
        if "snr" in df.columns:
            n_before = len(df)
            df = df[df["snr"] >= snr_cut]
            self._log(f"SNR filter (>={snr_cut}): {n_before} → {len(df)}")

        nan_am = ~np.isfinite(df["airmass"])
        nan_mag = ~np.isfinite(df["mag_inst"])
        n_drop = int((nan_am | nan_mag).sum())
        if n_drop > 0:
            self._log(f"Dropped {n_drop} points with NaN airmass/mag_inst "
                      f"({int(nan_am.sum())} airmass, {int(nan_mag.sum())} mag)")
        df = df[~nan_am & ~nan_mag].copy()
        if df.empty:
            raise RuntimeError("No data after SNR/airmass filters")

        fit_rows = []
        point_rows = []

        groups = list(df.groupby(["date", "filter"]))
        n_groups = len(groups)
        for i_grp, ((date_val, filt), sub) in enumerate(groups):
            self.progress.emit(i_grp, n_groups, f"Median {date_val}/{filt}")
            sub = sub.copy()
            n_pts = len(sub)

            if use_dx:
                x_all = get_numeric_array(sub, "airmass")
                if np.isfinite(x_all).any():
                    dx_all = float(np.nanmax(x_all) - np.nanmin(x_all))
                    if dx_all < dx_min_global:
                        self._log(f"[{date_val}][{filt}] skipped: ΔX={dx_all:.3f} < {dx_min_global:.3f}")
                        continue

            if n_pts < min_points:
                self._log(f"[{date_val}][{filt}] skipped: only {n_pts} points (min={min_points})")
                continue

            # per-star median within this date+filter
            med_map = sub.groupby("source_id")["mag_inst"].median()
            sub["delta_m"] = sub["mag_inst"] - sub["source_id"].map(med_map)

            # Variable star prefilter (delta_m scatter)
            if var_sigma > 0:
                sub = self._filter_variable_stars(
                    sub, "delta_m", var_method, var_sigma, var_min_frames,
                    label=f"[{date_val}][{filt}]")

            # Frame QC (delta_m median)
            if qc_sigma > 0:
                frame_stats = sub.groupby("file").agg(
                    delta_med=("delta_m", "median"),
                    delta_mad=("delta_m", lambda s: float(np.nanmedian(np.abs(s - np.nanmedian(s))))),
                    n_stars=("delta_m", "count"),
                )
                if qc_method == "std":
                    global_med = float(np.nanmean(frame_stats["delta_med"]))
                    global_sig = float(np.nanstd(frame_stats["delta_med"]))
                else:
                    global_med = float(np.nanmedian(frame_stats["delta_med"]))
                    global_mad = float(np.nanmedian(np.abs(frame_stats["delta_med"] - global_med)))
                    global_sig = MAD_TO_SIGMA * global_mad if global_mad > 0 else float(np.nanstd(frame_stats["delta_med"]))
                if global_sig > 0:
                    bad_mask = np.abs(frame_stats["delta_med"] - global_med) > qc_sigma * global_sig
                    bad_frames = set(frame_stats.index[bad_mask])
                    if bad_frames:
                        self._log(f"[{date_val}][{filt}] Frame QC ({qc_method}, {qc_sigma:.1f}σ): "
                                  f"{len(bad_frames)} frames removed")
                        sub = sub[~sub["file"].isin(bad_frames)]

            if sub.empty:
                continue

            x = get_numeric_array(sub, "airmass")
            y = get_numeric_array(sub, "delta_m")
            w = None
            if use_weights and "snr" in sub.columns:
                snr = get_numeric_array(sub, "snr")
                sig_mag = MAG_ERR_COEFF / np.clip(snr, 1e-6, None)
                w = 1.0 / np.clip(sig_mag ** 2, 1e-12, None)

            k1, zp, scatter, n_used, out_frac = self._robust_weighted_linfit(
                x, y, w=w, clip_sigma=clip_sigma, iters=fit_iters, min_n=min_points
            )
            if not np.isfinite(k1):
                self._log(f"[{date_val}][{filt}] FAILED: fit did not converge")
                continue

            resid = y - (zp + k1 * x)
            for xi, yi, ri in zip(x, y, resid):
                point_rows.append({
                    "date": date_val, "filter": filt,
                    "airmass": float(xi),
                    "delta_m": float(yi),
                    "resid": float(ri),
                })

            fit_rows.append({
                "date": date_val,
                "filter": filt,
                "k1": float(k1),
                "k2": 0.0,
                "zp": float(zp),
                "m0": np.nan,
                "scatter": float(scatter),
                "n_total": n_pts,
                "n_used": int(n_used),
                "outlier_fraction": float(out_frac),
                "fit_order": 1,
                "n_stars": int(sub["source_id"].nunique()),
                "n_frames": int(sub["file"].nunique()),
                "method": "median",
            })

            self._log(f"[{date_val}][{filt}] k1={k1:.5f}, scatter={scatter:.4f}, "
                      f"n={n_used}/{n_pts} ({sub['source_id'].nunique()} stars, {sub['file'].nunique()} frames)")

        if not fit_rows:
            raise RuntimeError("No valid median-subtract extinction fits produced")

        fit_df = pd.DataFrame(fit_rows)
        points_df = pd.DataFrame(point_rows)

        out_dir = tool_extinction_dir(self.result_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fit_path = out_dir / "ensemble_extinction_by_filter.csv"
        pts_path = out_dir / "ensemble_phot_points.csv"
        fit_df.to_csv(fit_path, index=False)
        points_df.to_csv(pts_path, index=False)
        self._log(f"Saved {fit_path.name} ({len(fit_df)} rows)")
        self._log(f"Saved {pts_path.name} ({len(points_df)} rows)")

        legacy_fit = out_dir / "extinction_fit_by_filter.csv"
        try:
            fit_df.to_csv(legacy_fit, index=False)
            self._log(f"Saved {legacy_fit.name} ({len(fit_df)} rows)")
        except Exception:
            self._log("Warning: failed to write legacy extinction_fit_by_filter.csv")

        self.finished.emit({
            "fit": fit_df, "points": points_df,
            "mode": "median",
        })

    def _save_photometry(self, phot_df: pd.DataFrame):
        out_dir = tool_extinction_dir(self.result_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        raw_path = out_dir / "ensemble_allstar_phot.csv"
        phot_df.to_csv(raw_path, index=False)
        self._log(f"Saved {raw_path.name} ({len(phot_df)} rows)")

    def _load_photometry(self) -> pd.DataFrame:
        out_dir = tool_extinction_dir(self.result_dir)
        raw_path = out_dir / "ensemble_allstar_phot.csv"
        if not raw_path.exists():
            raise FileNotFoundError("ensemble_allstar_phot.csv not found. Run photometry first.")
        return pd.read_csv(raw_path)

    def _extract_date_from_file(self, fname: str, hdr=None) -> str:
        """파일명/폴더명 또는 헤더에서 날짜 추출 (YYYY-MM-DD or YYYYMMDD)"""
        import re
        # 1. 파일명에 날짜 폴더가 포함된 경우 (예: "2024-01-15__pp_image.fits")
        if "__" in fname:
            folder_part = fname.split("__")[0]
            # YYYY-MM-DD 패턴
            m = re.match(r"(\d{4}-\d{2}-\d{2})", folder_part)
            if m:
                return m.group(1)
            # YYYYMMDD 패턴
            m = re.match(r"(\d{8})", folder_part)
            if m:
                d = m.group(1)
                return f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        # 2. DATE-OBS 헤더에서 추출
        if hdr is not None:
            date_obs = str(hdr.get("DATE-OBS", ""))
            if date_obs:
                m = re.match(r"(\d{4}-\d{2}-\d{2})", date_obs)
                if m:
                    return m.group(1)
        return "unknown"

    def _build_frame_airmass(self, idx: pd.DataFrame) -> pd.DataFrame:
        P = self.params.P
        lat = self._as_float(getattr(P, "site_lat_deg", 0.0), 0.0)
        lon = self._as_float(getattr(P, "site_lon_deg", 0.0), 0.0)
        alt = self._as_float(getattr(P, "site_alt_m", 0.0), 0.0)
        tz = self._as_float(getattr(P, "site_tz_offset_hours", 0.0), 0.0)

        rows = []
        for _, r in idx.iterrows():
            fname = str(r.get("file", "")).strip()
            if fname == "":
                continue
            fpath = self.params.get_file_path(fname)
            if not fpath.exists():
                cand = step2_cropped_dir(self.result_dir) / fname
                if cand.exists():
                    fpath = cand
            if not fpath.exists():
                continue
            try:
                hdr = fits.getheader(fpath)
                info = compute_airmass_from_header(hdr, lat, lon, alt, tz)
                filt = str(r.get("filter", hdr.get("FILTER", ""))).strip().lower()
                obs_date = self._extract_date_from_file(fname, hdr)
                rows.append({
                    "file": fname,
                    "filter": filt,
                    "date": obs_date,
                    **info,
                })
            except Exception:
                continue
        df = pd.DataFrame(rows)
        if df.empty:
            df = pd.DataFrame(columns=["file", "filter", "date", "airmass", "airmass_source", "alt_deg", "zenith_deg", "datetime_utc", "datetime_local", "ra_deg", "dec_deg"])
        if len(df):
            out_path = step10_dir(self.result_dir) / "frame_airmass.csv"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_path, index=False)
            self._log(f"Saved {out_path.name} | rows={len(df)}")
        return df

    def run(self):
        """Entry point: dispatch to ensemble or gaia-absolute mode."""
        try:
            if self.mode == "ensemble":
                if self.task == "photometry":
                    phot_df = self._run_allstar_photometry()
                    self._save_photometry(phot_df)
                    self.finished.emit({
                        "phot": phot_df,
                        "mode": "ensemble",
                        "task": "photometry",
                    })
                elif self.task == "fit":
                    phot_df = self.phot_df if isinstance(self.phot_df, pd.DataFrame) else None
                    if phot_df is None or phot_df.empty:
                        phot_df = self._load_photometry()
                    self._run_ensemble_fit(phot_df)
                else:
                    phot_df = self._run_allstar_photometry()
                    self._run_ensemble_fit(phot_df)
            elif self.mode == "per_star":
                if self.task == "photometry":
                    phot_df = self._run_allstar_photometry()
                    self._save_photometry(phot_df)
                    self.finished.emit({
                        "phot": phot_df,
                        "mode": "per_star",
                        "task": "photometry",
                    })
                elif self.task == "fit":
                    phot_df = self.phot_df if isinstance(self.phot_df, pd.DataFrame) else None
                    if phot_df is None or phot_df.empty:
                        phot_df = self._load_photometry()
                    self._run_per_star_fit(phot_df)
                else:
                    phot_df = self._run_allstar_photometry()
                    self._run_per_star_fit(phot_df)
            elif self.mode == "median":
                if self.task == "photometry":
                    phot_df = self._run_allstar_photometry()
                    self._save_photometry(phot_df)
                    self.finished.emit({
                        "phot": phot_df,
                        "mode": "median",
                        "task": "photometry",
                    })
                elif self.task == "fit":
                    phot_df = self.phot_df if isinstance(self.phot_df, pd.DataFrame) else None
                    if phot_df is None or phot_df.empty:
                        phot_df = self._load_photometry()
                    self._run_median_fit(phot_df)
                else:
                    phot_df = self._run_allstar_photometry()
                    self._run_median_fit(phot_df)
            else:
                self._run_gaia_absolute()
        except Exception as e:
            self.error.emit(str(e))

    def _run_gaia_absolute(self):
        """Original Gaia-absolute extinction fit (requires step 9 data)."""
        try:
            P = self.params.P
            result_dir = self.result_dir

            idx_candidates = [
                step9_dir(result_dir) / "photometry_index.csv",
                result_dir / "photometry_index.csv",
                result_dir / "phot_index.csv",
                result_dir / "phot" / "photometry_index.csv",
                result_dir / "phot" / "phot_index.csv",
            ]
            idx_path = next((p for p in idx_candidates if p.exists()), None)
            if idx_path is None:
                raise FileNotFoundError("photometry index csv not found")
            idx = pd.read_csv(idx_path)
            if "path" not in idx.columns:
                for cand in ("phot_tsv", "tsv", "out", "output"):
                    if cand in idx.columns:
                        idx = idx.rename(columns={cand: "path"})
                        break
            if "file" not in idx.columns:
                for cand in ("fname", "frame", "image", "fits", "name"):
                    if cand in idx.columns:
                        idx = idx.rename(columns={cand: "file"})
                        break
            if "filter" in idx.columns:
                idx["filter"] = idx["filter"].astype(str).str.strip().str.lower()
            elif "FILTER" in idx.columns:
                idx["filter"] = idx["FILTER"].astype(str).str.strip().str.lower()
            else:
                idx["filter"] = ""

            rows = []
            total = len(idx)
            for i, r in idx.iterrows():
                if self._stop_requested:
                    self.finished.emit({"stopped": True})
                    return
                p = r.get("path", "")
                p = str(p) if p is not None else ""
                if p.strip() == "":
                    continue
                tsv = step9_dir(self.result_dir) / p
                if not tsv.exists():
                    tsv = self.result_dir / p
                if not tsv.exists():
                    continue
                dfp = pd.read_csv(tsv, sep="\t")
                mag_col = None
                for cand in ("mag_inst", "mag", "mag_ap", "mag_apcorr"):
                    if cand in dfp.columns:
                        mag_col = cand
                        break
                if mag_col is None:
                    continue
                err_col = None
                for cand in ("mag_err", "emag", "emag_inst", "magerr"):
                    if cand in dfp.columns:
                        err_col = cand
                        break
                if err_col is None:
                    dfp["mag_err"] = np.nan
                    err_col = "mag_err"
                snr_col = "snr" if "snr" in dfp.columns else None
                tmp = dfp[["ID", "FILTER", mag_col, err_col] + ([snr_col] if snr_col else [])].copy()
                tmp = tmp.rename(columns={mag_col: "mag_inst", err_col: "mag_err"})
                if snr_col is None:
                    tmp["snr"] = np.nan
                else:
                    tmp = tmp.rename(columns={snr_col: "snr"})
                tmp["file"] = str(r.get("file", ""))
                rows.append(tmp)
                self.progress.emit(i + 1, total, str(r.get("file", "")))

            if not rows:
                raise RuntimeError("No photometry data found")
            all_df = pd.concat(rows, ignore_index=True)
            all_df["FILTER"] = all_df["FILTER"].astype(str).str.strip().str.lower()

            grp = (
                all_df.groupby(["ID", "FILTER"])
                .agg(
                    mag_inst_med=("mag_inst", lambda s: self._robust_median_and_err(s)[0]),
                    n_frames=("mag_inst", lambda s: self._robust_median_and_err(s)[2]),
                    snr_med=("snr", lambda s: float(np.nanmedian(np.asarray(s, float))) if np.isfinite(np.asarray(s, float)).any() else np.nan),
                )
                .reset_index()
            )

            wide_mag = grp.pivot_table(index="ID", columns="FILTER", values="mag_inst_med", aggfunc="median")
            wide_snr = grp.pivot_table(index="ID", columns="FILTER", values="snr_med", aggfunc="median")
            wide_mag.columns = [f"mag_inst_{c}" for c in wide_mag.columns]
            wide_snr.columns = [f"snr_{c}" for c in wide_snr.columns]
            wide = pd.concat([wide_mag, wide_snr], axis=1).reset_index()

            master_path = step6_dir(result_dir) / "ref_catalog.tsv"
            if not master_path.exists():
                master_path = legacy_step5_refbuild_dir(result_dir) / "ref_catalog.tsv"
            if not master_path.exists():
                master_path = legacy_step7_refbuild_dir(result_dir) / "master_catalog.tsv"
            if not master_path.exists():
                master_path = result_dir / "master_catalog.tsv"
            if not master_path.exists():
                raise FileNotFoundError("master_catalog.tsv missing")
            master = pd.read_csv(master_path, sep="\t")
            if "ID" not in master.columns:
                raise RuntimeError("master_catalog.tsv missing ID column")

            merge_cols = ["ID"]
            if "source_id" in master.columns:
                merge_cols.append("source_id")
            for col in ("gaia_G", "gaia_BP", "gaia_RP", "gmag", "bpmag", "rpmag", "phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag"):
                if col in master.columns and col not in merge_cols:
                    merge_cols.append(col)
            df = wide.merge(master[merge_cols], on="ID", how="left")

            g_col = None
            bp_col = None
            rp_col = None
            for cand in ("gaia_G", "gmag", "phot_g_mean_mag"):
                if cand in df.columns:
                    g_col = cand
                    break
            for cand in ("gaia_BP", "bpmag", "phot_bp_mean_mag"):
                if cand in df.columns:
                    bp_col = cand
                    break
            for cand in ("gaia_RP", "rpmag", "phot_rp_mean_mag"):
                if cand in df.columns:
                    rp_col = cand
                    break

            if g_col is None or bp_col is None or rp_col is None:
                if "source_id" not in df.columns:
                    raise RuntimeError("Gaia mags not available and source_id missing")
                gaia_path = step5_dir(result_dir) / "gaia_fov.ecsv"
                if not gaia_path.exists():
                    gaia_path = legacy_step7_wcs_dir(result_dir) / "gaia_fov.ecsv"
                if not gaia_path.exists():
                    gaia_path = result_dir / "gaia_fov.ecsv"
                if not gaia_path.exists():
                    raise RuntimeError("gaia_fov.ecsv not found")
                t_gaia = Table.read(gaia_path, format="ascii.ecsv")
                gaia_df = t_gaia.to_pandas()
                gaia_df["source_id"] = gaia_df["source_id"].astype("int64")
                gaia_cols = ["source_id", "phot_g_mean_mag"]
                if "phot_bp_mean_mag" in gaia_df.columns:
                    gaia_cols.append("phot_bp_mean_mag")
                if "phot_rp_mean_mag" in gaia_df.columns:
                    gaia_cols.append("phot_rp_mean_mag")
                df = df.merge(gaia_df[gaia_cols], on="source_id", how="left")
                g_col = "phot_g_mean_mag"
                bp_col = "phot_bp_mean_mag"
                rp_col = "phot_rp_mean_mag"

            df["gaia_G"] = pd.to_numeric(df[g_col], errors="coerce").astype(float)
            df["gaia_BP"] = pd.to_numeric(df[bp_col], errors="coerce").astype(float)
            df["gaia_RP"] = pd.to_numeric(df[rp_col], errors="coerce").astype(float)
            dfm = df[np.isfinite(df["gaia_G"]) & np.isfinite(df["gaia_BP"]) & np.isfinite(df["gaia_RP"])].copy()
            dfm["gaia_BP_RP"] = dfm["gaia_BP"] - dfm["gaia_RP"]

            out_cal = dfm.copy()
            xcol = out_cal["gaia_BP_RP"].to_numpy(float)
            G = out_cal["gaia_G"].to_numpy(float)

            G_minus = {}
            for band, info in GAIA_TO_SDSS.items():
                lo, hi = info["color_range"]
                m = np.isfinite(xcol) & (xcol >= lo) & (xcol <= hi)
                arr = np.full_like(G, np.nan)
                arr[m] = self._poly_eval(xcol[m], info["coeffs"])
                G_minus[band] = arr

            out_cal["sdss_g_ref"] = G - G_minus["g"]
            out_cal["sdss_r_ref"] = G - G_minus["r"]
            out_cal["sdss_i_ref"] = G - G_minus["i"]

            g_inst = out_cal.get("mag_inst_g", pd.Series(np.full(len(out_cal), np.nan))).to_numpy(float)
            r_inst = out_cal.get("mag_inst_r", pd.Series(np.full(len(out_cal), np.nan))).to_numpy(float)
            i_inst = out_cal.get("mag_inst_i", pd.Series(np.full(len(out_cal), np.nan))).to_numpy(float)

            color_gr = g_inst - r_inst
            color_ri = r_inst - i_inst

            clip_sigma = self._as_float(getattr(P, "zp_clip_sigma", 3.0), 3.0)
            fit_iters = self._as_int(getattr(P, "zp_fit_iters", 5), 5)
            min_match = self._as_int(getattr(P, "min_master_gaia_matches", 10), 10)

            delta_g = out_cal["sdss_g_ref"].to_numpy(float) - g_inst
            mg = np.isfinite(delta_g) & np.isfinite(color_gr) & np.isfinite(g_inst)
            ct_g = self._robust_linfit(color_gr[mg], delta_g[mg], clip_sigma=clip_sigma, iters=fit_iters, min_n=min_match)[0]
            if not np.isfinite(ct_g):
                ct_g = 0.0

            delta_r = out_cal["sdss_r_ref"].to_numpy(float) - r_inst
            mr = np.isfinite(delta_r) & np.isfinite(color_gr) & np.isfinite(r_inst)
            ct_r = self._robust_linfit(color_gr[mr], delta_r[mr], clip_sigma=clip_sigma, iters=fit_iters, min_n=min_match)[0]
            if not np.isfinite(ct_r):
                ct_r = 0.0

            delta_i = out_cal["sdss_i_ref"].to_numpy(float) - i_inst
            mi = np.isfinite(delta_i) & np.isfinite(color_ri) & np.isfinite(i_inst)
            ct_i = self._robust_linfit(color_ri[mi], delta_i[mi], clip_sigma=clip_sigma, iters=fit_iters, min_n=min_match)[0]
            if not np.isfinite(ct_i):
                ct_i = 0.0

            out_cal["color_gr"] = color_gr
            out_cal["color_ri"] = color_ri

            frame_airmass = self._build_frame_airmass(idx)
            if frame_airmass is None or frame_airmass.empty:
                raise RuntimeError("frame_airmass.csv missing and airmass computation failed")

            cal_cols = ["ID", "sdss_g_ref", "sdss_r_ref", "sdss_i_ref", "color_gr", "color_ri", "gaia_BP_RP"]
            obs = all_df.merge(out_cal[cal_cols], on="ID", how="left")
            # date 컬럼도 함께 merge
            merge_cols = ["file", "filter", "airmass"]
            if "date" in frame_airmass.columns:
                merge_cols.append("date")
            obs = obs.merge(frame_airmass[merge_cols], left_on=["file", "FILTER"], right_on=["file", "filter"], how="left")
            if "date" not in obs.columns:
                obs["date"] = "unknown"

            obs["ref_mag"] = np.nan
            obs.loc[obs["FILTER"] == "g", "ref_mag"] = obs.loc[obs["FILTER"] == "g", "sdss_g_ref"]
            obs.loc[obs["FILTER"] == "r", "ref_mag"] = obs.loc[obs["FILTER"] == "r", "sdss_r_ref"]
            obs.loc[obs["FILTER"] == "i", "ref_mag"] = obs.loc[obs["FILTER"] == "i", "sdss_i_ref"]

            obs["color_term"] = np.nan
            obs.loc[obs["FILTER"] == "g", "color_term"] = ct_g * obs.loc[obs["FILTER"] == "g", "color_gr"]
            obs.loc[obs["FILTER"] == "r", "color_term"] = ct_r * obs.loc[obs["FILTER"] == "r", "color_gr"]
            obs.loc[obs["FILTER"] == "i", "color_term"] = ct_i * obs.loc[obs["FILTER"] == "i", "color_ri"]

            obs["delta"] = obs["ref_mag"] - (obs["mag_inst"] + obs["color_term"])
            obs["cal_ok"] = False
            bp = obs["gaia_BP_RP"].to_numpy(float)
            obs.loc[(obs["FILTER"] == "g") & np.isfinite(bp) & (bp >= 0.3) & (bp <= 3.0), "cal_ok"] = True
            obs.loc[(obs["FILTER"] == "r") & np.isfinite(bp) & (bp >= 0.0) & (bp <= 3.0), "cal_ok"] = True
            obs.loc[(obs["FILTER"] == "i") & np.isfinite(bp) & (bp >= 0.5) & (bp <= 2.0), "cal_ok"] = True

            snr_cut = self._as_float(getattr(P, "cmd_snr_calib_min", 20.0), 20.0)
            obs["snr_ok"] = True
            if "snr" in obs.columns:
                svals = obs["snr"].to_numpy(float)
                obs["snr_ok"] = np.isfinite(svals) & (svals >= snr_cut)

            filters_seen = sorted(set(all_df["FILTER"].dropna().astype(str)))
            out_dir = result_dir / "extinction"
            out_dir.mkdir(parents=True, exist_ok=True)
            step11_out = legacy_step11_extinction_dir(result_dir)
            step11_out.mkdir(parents=True, exist_ok=True)
            stats_rows = []
            for filt in filters_seen:
                fmask = obs["FILTER"] == filt
                n_total = int(fmask.sum())
                n_ref = int(np.isfinite(obs.loc[fmask, "ref_mag"]).sum())
                n_air = int(np.isfinite(obs.loc[fmask, "airmass"]).sum())
                n_color = int(np.isfinite(obs.loc[fmask, "color_term"]).sum())
                n_snr = int(obs.loc[fmask, "snr_ok"].sum()) if "snr_ok" in obs.columns else n_total
                n_cal = int(obs.loc[fmask, "cal_ok"].sum())
                n_delta = int(np.isfinite(obs.loc[fmask, "delta"]).sum())
                self._log(
                    f"Filter[{filt}] total={n_total}, ref={n_ref}, color={n_color}, "
                    f"airmass={n_air}, snr_ok={n_snr}, cal_ok={n_cal}, delta_ok={n_delta}"
                )
                stats_rows.append({
                    "filter": filt,
                    "n_total": n_total,
                    "n_ref": n_ref,
                    "n_color": n_color,
                    "n_airmass": n_air,
                    "n_snr_ok": n_snr,
                    "n_cal_ok": n_cal,
                    "n_delta_ok": n_delta,
                })

            if stats_rows:
                stats_df = pd.DataFrame(stats_rows)
                stats_path = out_dir / "extinction_fit_filter_stats.csv"
                stats_df.to_csv(stats_path, index=False)
                self._log(f"Saved {stats_path.name} | rows={len(stats_df)}")

            obs = obs[np.isfinite(obs["delta"]) & np.isfinite(obs["airmass"]) & obs["cal_ok"] & obs["snr_ok"]].copy()

            # 날짜별 분류 통계
            dates_seen = sorted(set(obs["date"].dropna().astype(str)))
            self._log(f"Dates found: {dates_seen}")
            if len(obs):
                for (date_val, filt), sub in obs.groupby(["date", "FILTER"]):
                    self._log(f"Fit candidates [{date_val}][{filt}]: {len(sub)} points")

            fit_rows = []
            point_rows = []
            use_quadratic = bool(getattr(P, "extinction_use_quadratic", True))
            use_color_extinction = bool(getattr(P, "extinction_use_color_dependent", True))
            min_quad = int(getattr(P, "extinction_min_points_quadratic", 20))
            min_color = int(getattr(P, "extinction_min_points_color", 30))

            # 날짜별 + 필터별 피팅
            for (date_val, filt), sub in obs.groupby(["date", "FILTER"]):
                x = sub["airmass"].to_numpy(float)
                y = sub["delta"].to_numpy(float)

                # 색지수 선택: g/r 필터면 g-r, i 필터면 r-i
                if filt in ("g", "r"):
                    color_col = "color_gr"
                elif filt == "i":
                    color_col = "color_ri"
                else:
                    color_col = "color_gr"  # 기본값

                color = sub[color_col].to_numpy(float) if color_col in sub.columns else np.full(len(x), np.nan)

                # 1차 피팅 (기본)
                k1_lin, zp_lin, scatter_lin, n_ref_lin, out_frac_lin = self._robust_linfit(
                    x, y, clip_sigma=clip_sigma, iters=fit_iters, min_n=min_match
                )

                # 색 의존 소광 피팅: delta = k' * X + k'' * C * X
                k_prime, k_color, zp_col, scatter_col, n_ref_col, out_frac_col, k_col_sig = (
                    np.nan, np.nan, np.nan, np.nan, 0, np.nan, False
                )
                n_valid_color = np.sum(np.isfinite(x) & np.isfinite(color) & np.isfinite(y))
                if use_color_extinction and n_valid_color >= min_color:
                    k_prime, k_color, zp_col, scatter_col, n_ref_col, out_frac_col, k_col_sig = self._robust_color_extinction_fit(
                        x, color, y, clip_sigma=clip_sigma, iters=fit_iters, min_n=min_color
                    )
                    if k_col_sig and np.isfinite(k_color):
                        self._log(f"[{date_val}][{filt}] color extinction fit: k'={k_prime:.4f}, k''={k_color:.4f} (n={n_ref_col})")

                # 2차 피팅 시도 (에어매스 비선형)
                k1_q, k2_q, zp_q, scatter_q, n_ref_q, out_frac_q, k2_sig = (
                    np.nan, np.nan, np.nan, np.nan, 0, np.nan, False
                )
                if use_quadratic and len(x) >= min_quad:
                    k1_q, k2_q, zp_q, scatter_q, n_ref_q, out_frac_q, k2_sig = self._robust_quadfit(
                        x, y, clip_sigma=clip_sigma, iters=fit_iters, min_n=min_quad
                    )

                # 최선의 피팅 선택
                # 우선순위: 색 의존 피팅 > 2차 피팅 > 1차 피팅
                k1, k2, k_color_final, zp = k1_lin, 0.0, 0.0, zp_lin
                scatter, n_ref, out_frac = scatter_lin, n_ref_lin, out_frac_lin
                fit_order = 1

                # 색 의존 피팅이 유의미하면 사용
                if k_col_sig and np.isfinite(k_color) and scatter_col < scatter_lin * 0.90:
                    k1, k2, k_color_final, zp = k_prime, 0.0, k_color, zp_col
                    scatter, n_ref, out_frac = scatter_col, n_ref_col, out_frac_col
                    fit_order = 3  # color-dependent
                    yhat = zp + k1 * x + k_color_final * color * x
                    self._log(f"[{date_val}][{filt}] using COLOR-DEPENDENT fit: k'={k1:.4f}, k''={k_color_final:.4f}")
                # 2차 피팅이 유의미하면 사용
                elif k2_sig and np.isfinite(k2_q) and scatter_q < scatter_lin * 0.95:
                    k1, k2, zp = k1_q, k2_q, zp_q
                    scatter, n_ref, out_frac = scatter_q, n_ref_q, out_frac_q
                    fit_order = 2
                    yhat = zp + k1 * x + k2 * x * x
                    self._log(f"[{date_val}][{filt}] using quadratic fit (k2={k2:.4f})")
                else:
                    yhat = zp + k1 * x

                if not np.isfinite(k1):
                    self._log(f"[{date_val}][{filt}] FAILED: insufficient points (n={len(x)})")
                    continue

                resid = y - yhat
                fit_rows.append({
                    "date": date_val,
                    "filter": filt,
                    "k1": k1,
                    "k2": k2,
                    "k_color": k_color_final,
                    "zp": zp,
                    "scatter": scatter,
                    "n_ref": n_ref,
                    "outlier_fraction": out_frac,
                    "fit_order": fit_order,
                    "method": "gaia",
                })
                for xi, yi, ri, ci in zip(x, y, resid, color):
                    point_rows.append({"date": date_val, "filter": filt, "airmass": xi, "delta": yi, "resid": ri, "color": ci})

            fit_df = pd.DataFrame(fit_rows)
            if fit_df.empty:
                raise RuntimeError("No valid extinction fits produced")

            # Step 11 전용 폴더에 저장
            fit_path = step11_out / "step11_extinction_fit_by_filter.csv"
            fit_df.to_csv(fit_path, index=False)
            self._log(f"Saved {fit_path.name} | rows={len(fit_df)}")

            # 기존 경로에도 저장 (호환성)
            fit_path_old = out_dir / "extinction_fit_by_filter.csv"
            fit_df.to_csv(fit_path_old, index=False)

            for _, row in fit_df.iterrows():
                k2_str = f", k2={row['k2']:.4f}" if row.get('k2', 0) != 0 else ""
                k_color_str = f", k''={row['k_color']:.4f}" if row.get('k_color', 0) != 0 else ""
                fit_type = {1: "linear", 2: "quadratic", 3: "color-dep"}.get(row.get('fit_order', 1), "linear")
                date_str = row.get('date', 'unknown')
                self._log(
                    f"[{date_str}][{row['filter']}] k'={row['k1']:.4f}{k2_str}{k_color_str}, zp={row['zp']:.4f}, "
                    f"scatter={row['scatter']:.4f}, n={int(row['n_ref'])}, type={fit_type}"
                )

            points_df = pd.DataFrame(point_rows)
            points_path = step11_out / "step11_extinction_fit_points.csv"
            points_df.to_csv(points_path, index=False)
            self._log(f"Saved {points_path.name} | rows={len(points_df)}")

            self.finished.emit({"fit": fit_df, "points": points_df, "mode": "gaia"})
        except Exception as e:
            self.error.emit(str(e))


class ExtinctionFitWindow(QWidget):
    """Extinction (Airmass Fit) tool window."""

    def __init__(self, params, data_dir: Path, result_dir: Path, parent=None):
        super().__init__(parent)
        self.params = params
        self.data_dir = Path(data_dir)
        self.result_dir = Path(result_dir)
        self.worker = None

        self.setWindowTitle("Extinction (Airmass Fit)")
        self.resize(1000, 750)
        self._current_mode = "ensemble"  # track which mode was last run

        layout = QVBoxLayout(self)

        info = QLabel(
            "Atmospheric extinction fitting tool.\n"
            "Ensemble: solve s_i + z_j + k1*X (step 7+)\n"
            "Per-star: fit k_i per star, then aggregate k1\n"
            "Median-subtract: Δm vs X fit (legacy)"
        )
        info.setStyleSheet("QLabel { background-color: #E3F2FD; padding: 8px; border-radius: 5px; }")
        layout.addWidget(info)

        # Mode & aperture settings
        settings_group = QGroupBox("Settings")
        settings_layout = QHBoxLayout(settings_group)

        settings_layout.addWidget(QLabel("Method:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "Ensemble (s_i + z_j + k1*X)",
            "Per-star Bouguer (k_i)",
            "Median-subtract (weighted)",
        ])
        self.method_combo.setCurrentIndex(0)
        settings_layout.addWidget(self.method_combo)

        settings_layout.addStretch()
        btn_params = QPushButton("Parameters")
        btn_params.setStyleSheet("QPushButton { background-color: #9C27B0; color: white; font-weight: bold; padding: 6px 12px; }")
        btn_params.clicked.connect(self.open_parameters_dialog)
        settings_layout.addWidget(btn_params)
        layout.addWidget(settings_group)

        # Tabs: Photometry | Fit
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # --- Photometry tab ---
        phot_tab = QWidget()
        phot_layout = QVBoxLayout(phot_tab)
        phot_info = QLabel("Allstar photometry (step 7+) for extinction fitting.")
        phot_info.setStyleSheet("QLabel { background-color: #E3F2FD; padding: 8px; border-radius: 5px; }")
        phot_layout.addWidget(phot_info)

        phot_controls = QHBoxLayout()
        self.chk_use_cache = QCheckBox("Use cached photometry")
        self.chk_use_cache.setChecked(True)
        phot_controls.addWidget(self.chk_use_cache)

        self.btn_run_phot = QPushButton("Run Photometry")
        self.btn_run_phot.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 6px 12px; }")
        self.btn_run_phot.clicked.connect(self.run_photometry)
        phot_controls.addWidget(self.btn_run_phot)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; padding: 6px 12px; }")
        self.btn_stop.clicked.connect(self.stop_fit)
        self.btn_stop.setEnabled(False)
        phot_controls.addWidget(self.btn_stop)

        phot_controls.addStretch()
        phot_layout.addLayout(phot_controls)

        progress_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        self.progress_label = QLabel("Ready")
        self.progress_label.setMinimumWidth(320)
        progress_layout.addWidget(self.progress_label)
        phot_layout.addLayout(progress_layout)

        self.tabs.addTab(phot_tab, "Photometry")

        # --- Fit tab ---
        fit_tab = QWidget()
        fit_layout = QVBoxLayout(fit_tab)
        fit_info = QLabel("Extinction fit from allstar photometry (date + filter).")
        fit_info.setStyleSheet("QLabel { background-color: #E3F2FD; padding: 8px; border-radius: 5px; }")
        fit_layout.addWidget(fit_info)

        fit_controls = QHBoxLayout()
        self.btn_run_fit = QPushButton("Run Fit")
        self.btn_run_fit.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 6px 12px; }")
        self.btn_run_fit.clicked.connect(self.run_fit)
        fit_controls.addWidget(self.btn_run_fit)

        self.btn_save = QPushButton("Save Plots")
        self.btn_save.setStyleSheet("QPushButton { background-color: #607D8B; color: white; font-weight: bold; padding: 6px 12px; }")
        self.btn_save.clicked.connect(self.save_plots)
        self.btn_save.setEnabled(False)
        fit_controls.addWidget(self.btn_save)

        fit_controls.addStretch()
        fit_layout.addLayout(fit_controls)

        # 날짜/필터 선택 UI
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Date:"))
        self.date_combo = QComboBox()
        self.date_combo.addItem("All Dates")
        self.date_combo.currentIndexChanged.connect(self._on_filter_changed)
        filter_layout.addWidget(self.date_combo)

        filter_layout.addWidget(QLabel("Filter:"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItem("All Filters")
        self.filter_combo.currentIndexChanged.connect(self._on_filter_changed)
        filter_layout.addWidget(self.filter_combo)

        self.chk_show_fit = QCheckBox("Show Fit Lines")
        self.chk_show_fit.setChecked(True)
        self.chk_show_fit.stateChanged.connect(self._on_filter_changed)
        filter_layout.addWidget(self.chk_show_fit)

        filter_layout.addStretch()
        fit_layout.addLayout(filter_layout)

        plot_group = QGroupBox("Fit Diagnostics")
        plot_layout = QVBoxLayout(plot_group)
        self.figure = Figure(figsize=(7, 5))
        self.canvas = FigureCanvas(self.figure)
        plot_layout.addWidget(NavigationToolbar(self.canvas, self))
        plot_layout.addWidget(self.canvas)
        fit_layout.addWidget(plot_group)

        self.tabs.addTab(fit_tab, "Extinction Fit")

        # Shared log (bottom)
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("QTextEdit { font-family: monospace; font-size: 9pt; }")
        log_layout.addWidget(self.log_text)
        layout.addWidget(log_group)

        self.points_df = None
        self.fit_df = None
        self.phot_df = None

    def log(self, message: str):
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")

    def on_progress(self, current: int, total: int, fname: str):
        pct = int(100 * current / max(total, 1))
        self.progress_bar.setValue(pct)
        self.progress_label.setText(f"{current}/{total} | {fname}")

    def run_photometry(self):
        if self.worker and self.worker.isRunning():
            return
        self.log_text.clear()
        self.btn_run_phot.setEnabled(False)
        self.btn_run_fit.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_save.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Starting photometry...")

        cached_df = None
        if self.chk_use_cache.isChecked():
            out_dir = tool_extinction_dir(self.result_dir)
            raw_path = out_dir / "ensemble_allstar_phot.csv"
            if raw_path.exists():
                try:
                    cached_df = pd.read_csv(raw_path)
                    self.phot_df = cached_df
                    self.log(f"Loaded cached photometry ({raw_path.name}, {len(cached_df)} rows)")
                except Exception:
                    self.log("Cached photometry load failed; recomputing.")

        ap_scale = ExtinctionFitWorker._as_float(getattr(self.params.P, "phot_aperture_scale", 1.0), 1.0)
        ann_in_scale = ExtinctionFitWorker._as_float(getattr(self.params.P, "fitsky_annulus_scale", 4.0), 4.0)
        ann_out_scale = ExtinctionFitWorker._as_float(getattr(self.params.P, "fitsky_dannulus_scale", 2.0), 2.0)

        if self.method_combo.currentIndex() == 0:
            mode = "ensemble"
        elif self.method_combo.currentIndex() == 1:
            mode = "per_star"
        else:
            mode = "median"
        self.worker = ExtinctionFitWorker(
            self.params, self.data_dir, self.result_dir,
            mode=mode, task="photometry",
            phot_df=cached_df,
            ap_scale=ap_scale,
            ann_in_scale=ann_in_scale,
            ann_out_scale=ann_out_scale,
        )
        self.worker.progress.connect(self.on_progress)
        self.worker.log.connect(self.log)
        self.worker.finished.connect(self.on_photometry_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_photometry_finished(self, results: dict):
        self.btn_run_phot.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_label.setText("Photometry complete")
        phot_df = results.get("phot")
        if isinstance(phot_df, pd.DataFrame):
            self.phot_df = phot_df
        self.btn_run_fit.setEnabled(True)
        self.log("Photometry complete")

    def open_parameters_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Extinction Parameters")
        dialog.resize(420, 360)
        layout = QVBoxLayout(dialog)
        form = QFormLayout()

        # Variable-star filter group
        var_group = QGroupBox("Variable-star filter")
        var_form = QFormLayout(var_group)

        method_combo = QComboBox()
        method_combo.addItems(["MAD", "STD"])
        current_method = str(getattr(self.params.P, "extinction_varstar_method", "mad") or "mad").strip().lower()
        method_combo.setCurrentIndex(1 if current_method == "std" else 0)
        var_form.addRow("Metric:", method_combo)

        var_sigma = QDoubleSpinBox()
        var_sigma.setRange(0.5, 10.0)
        var_sigma.setSingleStep(0.1)
        var_sigma.setValue(float(getattr(self.params.P, "extinction_varstar_sigma", 3.0) or 3.0))
        var_form.addRow("Sigma:", var_sigma)

        var_min_frames = QSpinBox()
        var_min_frames.setRange(3, 100)
        var_min_frames.setValue(int(getattr(self.params.P, "extinction_varstar_min_frames", 5) or 5))
        var_form.addRow("Min frames:", var_min_frames)
        layout.addWidget(var_group)

        # Frame QC group
        qc_group = QGroupBox("Frame QC")
        qc_form = QFormLayout(qc_group)

        qc_method_combo = QComboBox()
        qc_method_combo.addItems(["MAD", "STD"])
        qc_method = str(getattr(self.params.P, "extinction_frame_qc_method", "mad") or "mad").strip().lower()
        qc_method_combo.setCurrentIndex(1 if qc_method == "std" else 0)
        qc_form.addRow("Metric:", qc_method_combo)

        qc_sigma = QDoubleSpinBox()
        qc_sigma.setRange(0.5, 10.0)
        qc_sigma.setSingleStep(0.1)
        qc_sigma.setValue(float(getattr(self.params.P, "extinction_frame_qc_sigma", 3.0) or 3.0))
        qc_form.addRow("Sigma:", qc_sigma)
        layout.addWidget(qc_group)

        # Method parameters group
        method_group = QGroupBox("Method parameters")
        method_form = QFormLayout(method_group)

        snr_cut = QDoubleSpinBox()
        snr_cut.setRange(0.0, 100.0)
        snr_cut.setSingleStep(0.5)
        snr_cut.setValue(float(getattr(self.params.P, "extinction_snr_min", 10.0) or 10.0))
        method_form.addRow("SNR min:", snr_cut)

        clip_sigma = QDoubleSpinBox()
        clip_sigma.setRange(0.5, 10.0)
        clip_sigma.setSingleStep(0.1)
        clip_sigma.setValue(float(getattr(self.params.P, "zp_clip_sigma", 3.0) or 3.0))
        method_form.addRow("Fit clip sigma:", clip_sigma)

        fit_iters = QSpinBox()
        fit_iters.setRange(1, 20)
        fit_iters.setValue(int(getattr(self.params.P, "zp_fit_iters", 5) or 5))
        method_form.addRow("Fit iterations:", fit_iters)

        min_match = QSpinBox()
        min_match.setRange(3, 200)
        min_match.setValue(int(getattr(self.params.P, "min_master_gaia_matches", 10) or 10))
        method_form.addRow("Min points:", min_match)

        delta_x_enable = QCheckBox("Enable ΔX minimum")
        delta_x_enable.setChecked(bool(getattr(self.params.P, "extinction_delta_x_enable", True)))
        method_form.addRow("ΔX filter:", delta_x_enable)

        delta_x_min = QDoubleSpinBox()
        delta_x_min.setRange(0.0, 2.0)
        delta_x_min.setSingleStep(0.05)
        delta_x_min.setValue(float(getattr(self.params.P, "extinction_delta_x_min", 0.3) or 0.3))
        method_form.addRow("ΔX min:", delta_x_min)

        star_min_frames = QSpinBox()
        star_min_frames.setRange(3, 200)
        star_min_frames.setValue(int(getattr(self.params.P, "extinction_star_min_frames", 8) or 8))
        method_form.addRow("Per-star min frames:", star_min_frames)

        star_rms_max = QDoubleSpinBox()
        star_rms_max.setRange(0.0, 1.0)
        star_rms_max.setSingleStep(0.01)
        star_rms_max.setValue(float(getattr(self.params.P, "extinction_star_rms_max", 0.10) or 0.10))
        method_form.addRow("Per-star RMS max:", star_rms_max)

        star_snr_min = QDoubleSpinBox()
        star_snr_min.setRange(0.0, 200.0)
        star_snr_min.setSingleStep(1.0)
        star_snr_min.setValue(float(getattr(self.params.P, "extinction_star_snr_med_min", 10.0) or 10.0))
        method_form.addRow("Per-star median SNR min:", star_snr_min)

        min_good_stars = QSpinBox()
        min_good_stars.setRange(1, 200)
        min_good_stars.setValue(int(getattr(self.params.P, "extinction_min_good_stars", 3) or 3))
        method_form.addRow("Min good stars:", min_good_stars)

        use_weights = QCheckBox("Use SNR weights (fit)")
        use_weights.setChecked(bool(getattr(self.params.P, "extinction_star_use_weights", True)))
        method_form.addRow("Weights:", use_weights)

        layout.addWidget(method_group)

        scale_group = QGroupBox("Aperture / Annulus")
        scale_form = QFormLayout(scale_group)

        ap_scale = QDoubleSpinBox()
        ap_scale.setRange(0.5, 5.0)
        ap_scale.setSingleStep(0.1)
        ap_scale.setValue(float(getattr(self.params.P, "phot_aperture_scale", 1.0) or 1.0))
        ap_scale.setToolTip("r_ap (aperture radius) = scale × FWHM (typical ~1.0)")
        scale_form.addRow("Aperture r_ap (×FWHM):", ap_scale)

        ann_in_scale = QDoubleSpinBox()
        ann_in_scale.setRange(1.0, 10.0)
        ann_in_scale.setSingleStep(0.5)
        ann_in_scale.setValue(float(getattr(self.params.P, "fitsky_annulus_scale", 4.0) or 4.0))
        ann_in_scale.setToolTip("r_in (annulus inner radius) = scale × FWHM (typical ~4.0)")
        scale_form.addRow("Annulus r_in (×FWHM):", ann_in_scale)

        ann_out_scale = QDoubleSpinBox()
        ann_out_scale.setRange(0.5, 10.0)
        ann_out_scale.setSingleStep(0.5)
        ann_out_scale.setValue(float(getattr(self.params.P, "fitsky_dannulus_scale", 2.0) or 2.0))
        ann_out_scale.setToolTip("d_annulus (annulus thickness) = scale × FWHM (typical 2–3)")
        scale_form.addRow("Annulus d_annulus (×FWHM):", ann_out_scale)

        ann_sigma = QDoubleSpinBox()
        ann_sigma.setRange(0.5, 10.0)
        ann_sigma.setSingleStep(0.1)
        ann_sigma.setValue(float(getattr(self.params.P, "annulus_sigma_clip", 3.0) or 3.0))
        scale_form.addRow("Annulus sigma clip:", ann_sigma)

        layout.addWidget(scale_group)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec_() == QDialog.Accepted:
            self.params.P.extinction_varstar_method = "std" if method_combo.currentText() == "STD" else "mad"
            self.params.P.extinction_varstar_sigma = float(var_sigma.value())
            self.params.P.extinction_varstar_min_frames = int(var_min_frames.value())
            self.params.P.extinction_frame_qc_method = "std" if qc_method_combo.currentText() == "STD" else "mad"
            self.params.P.extinction_frame_qc_sigma = float(qc_sigma.value())

            self.params.P.extinction_snr_min = float(snr_cut.value())
            self.params.P.zp_clip_sigma = float(clip_sigma.value())
            self.params.P.zp_fit_iters = int(fit_iters.value())
            self.params.P.min_master_gaia_matches = int(min_match.value())
            self.params.P.extinction_delta_x_enable = bool(delta_x_enable.isChecked())
            self.params.P.extinction_delta_x_min = float(delta_x_min.value())

            self.params.P.extinction_star_min_frames = int(star_min_frames.value())
            self.params.P.extinction_star_rms_max = float(star_rms_max.value())
            self.params.P.extinction_star_snr_med_min = float(star_snr_min.value())
            self.params.P.extinction_min_good_stars = int(min_good_stars.value())
            self.params.P.extinction_star_use_weights = bool(use_weights.isChecked())

            self.params.P.phot_aperture_scale = float(ap_scale.value())
            self.params.P.fitsky_annulus_scale = float(ann_in_scale.value())
            self.params.P.fitsky_dannulus_scale = float(ann_out_scale.value())
            self.params.P.annulus_sigma_clip = float(ann_sigma.value())


            saved = False
            try:
                saved = bool(self.params.save_toml())
            except Exception:
                saved = False

            if saved:
                QMessageBox.information(self, "Saved", "Extinction parameters saved.")
            else:
                QMessageBox.information(self, "Saved", "Parameters updated in memory.")

    def run_fit(self):
        if self.worker and self.worker.isRunning():
            return
        self.log_text.clear()
        self.figure.clear()
        self.canvas.draw()
        self.btn_run_fit.setEnabled(False)
        self.btn_run_phot.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_save.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Starting fit...")

        if self.method_combo.currentIndex() == 0:
            mode = "ensemble"
        elif self.method_combo.currentIndex() == 1:
            mode = "per_star"
        else:
            mode = "median"
        self._current_mode = mode

        if self.phot_df is None:
            out_dir = tool_extinction_dir(self.result_dir)
            raw_path = out_dir / "ensemble_allstar_phot.csv"
            if not raw_path.exists():
                QMessageBox.warning(self, "Missing photometry", "Run photometry first (no ensemble_allstar_phot.csv).")
                self.btn_run_fit.setEnabled(True)
                self.btn_run_phot.setEnabled(True)
                self.btn_stop.setEnabled(False)
                return

        ap_scale = ExtinctionFitWorker._as_float(getattr(self.params.P, "phot_aperture_scale", 1.0), 1.0)
        ann_in_scale = ExtinctionFitWorker._as_float(getattr(self.params.P, "fitsky_annulus_scale", 4.0), 4.0)
        ann_out_scale = ExtinctionFitWorker._as_float(getattr(self.params.P, "fitsky_dannulus_scale", 2.0), 2.0)

        self.worker = ExtinctionFitWorker(
            self.params, self.data_dir, self.result_dir,
            mode=mode, task="fit", phot_df=self.phot_df,
            ap_scale=ap_scale,
            ann_in_scale=ann_in_scale,
            ann_out_scale=ann_out_scale,
        )
        self.worker.progress.connect(self.on_progress)
        self.worker.log.connect(self.log)
        self.worker.finished.connect(self.on_fit_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def stop_fit(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.log("Stop requested")

    def on_fit_finished(self, results):
        self.btn_run_fit.setEnabled(True)
        self.btn_run_phot.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.fit_df = results.get("fit")
        self.points_df = results.get("points")
        if isinstance(self.fit_df, pd.DataFrame) and isinstance(self.points_df, pd.DataFrame):
            # 콤보박스 채우기
            self._populate_combos()
            self._plot_results()
            self.btn_save.setEnabled(True)
        self.log("Extinction fit complete")
        self.progress_label.setText("Done")

    def _populate_combos(self):
        """날짜/필터 콤보박스 채우기"""
        # 날짜 콤보
        self.date_combo.blockSignals(True)
        self.date_combo.clear()
        self.date_combo.addItem("All Dates")
        if self.fit_df is not None and "date" in self.fit_df.columns:
            dates = sorted(set(self.fit_df["date"].dropna().astype(str)))
            for d in dates:
                self.date_combo.addItem(d)
        self.date_combo.blockSignals(False)

        # 필터 콤보
        self.filter_combo.blockSignals(True)
        self.filter_combo.clear()
        self.filter_combo.addItem("All Filters")
        if self.fit_df is not None and "filter" in self.fit_df.columns:
            filters = sorted(set(self.fit_df["filter"].dropna().astype(str)))
            for f in filters:
                self.filter_combo.addItem(f)
        self.filter_combo.blockSignals(False)

    def _on_filter_changed(self):
        """필터/날짜 변경 시 플롯 갱신"""
        self._plot_results()

    def on_error(self, message: str):
        self.btn_run_phot.setEnabled(True)
        self.btn_run_fit.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_label.setText("Error")
        QMessageBox.critical(self, "Error", f"Extinction fit failed:\n{message}")

    def _plot_results(self):
        self.figure.clear()
        ax1, ax2 = self.figure.subplots(2, 1)

        if self.fit_df is None or self.fit_df.empty:
            self.canvas.draw()
            return

        mode = self._current_mode
        is_ensemble = mode == "ensemble"
        # Ensemble/per-star/median use "delta_m"; Gaia uses "delta"
        y_col = "delta_m" if mode in ("ensemble", "per_star", "median") else "delta"

        # 선택된 날짜/필터 가져오기
        sel_date = self.date_combo.currentText()
        sel_filter = self.filter_combo.currentText()
        show_fit = self.chk_show_fit.isChecked()

        # 데이터 필터링
        fit_df = self.fit_df.copy()
        points_df = self.points_df.copy() if self.points_df is not None else pd.DataFrame()

        if sel_date != "All Dates" and "date" in fit_df.columns:
            fit_df = fit_df[fit_df["date"] == sel_date]
            if not points_df.empty and "date" in points_df.columns:
                points_df = points_df[points_df["date"] == sel_date]

        if sel_filter != "All Filters" and "filter" in fit_df.columns:
            fit_df = fit_df[fit_df["filter"] == sel_filter]
            if not points_df.empty and "filter" in points_df.columns:
                points_df = points_df[points_df["filter"] == sel_filter]

        x_min = None
        x_max = None
        if not points_df.empty and "airmass" in points_df.columns:
            x_min = float(points_df["airmass"].min())
            x_max = float(points_df["airmass"].max())

        # 색상: 날짜별로 구분
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        markers = ["o", "s", "^", "v", "D", "<", ">", "p", "h", "*"]

        for i, (_, row) in enumerate(fit_df.iterrows()):
            filt = str(row["filter"])
            date_val = str(row.get("date", "unknown"))
            k1 = float(row.get("k1", row.get("k", 0)))
            k2 = float(row.get("k2", 0)) if "k2" in row else 0.0
            zp = float(row["zp"])
            fit_order = int(row.get("fit_order", 1)) if "fit_order" in row else (2 if k2 != 0 else 1)
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]

            # 해당 날짜+필터의 포인트 데이터
            if not points_df.empty and y_col in points_df.columns:
                mask = points_df["filter"] == filt
                if "date" in points_df.columns:
                    mask &= points_df["date"] == date_val
                sub = points_df[mask]
            else:
                sub = pd.DataFrame()

            label_prefix = f"{date_val}/{filt}" if sel_date == "All Dates" else filt
            if len(sub) and "airmass" in sub.columns:
                ax1.scatter(sub["airmass"], sub[y_col], s=8, alpha=0.4,
                            color=color, marker=marker, label=f"{label_prefix} (n={len(sub)})")
                x = np.linspace(sub["airmass"].min(), sub["airmass"].max(), 100)
            elif x_min is not None and x_max is not None:
                x = np.linspace(x_min, x_max, 100)
            else:
                x = np.linspace(1.0, 2.5, 100)

            if show_fit and np.isfinite(k1) and np.isfinite(zp):
                if fit_order == 2 and k2 != 0:
                    y_fit = zp + k1 * x + k2 * x * x
                    label = f"{label_prefix}: k1={k1:.4f}, k2={k2:.4f}"
                else:
                    y_fit = zp + k1 * x
                    label = f"{label_prefix}: k1={k1:.4f}"
                ax1.plot(x, y_fit, lw=2, color=color, linestyle="-", label=label)

        ax1.set_xlabel("Airmass (X)", fontsize=10)
        if mode == "ensemble":
            ax1.set_ylabel("Δm (m - s_i - z_j, mag)", fontsize=10)
        elif mode == "per_star":
            ax1.set_ylabel("Δm (m - m0_i, mag)", fontsize=10)
        elif mode == "median":
            ax1.set_ylabel("Δm (median-subtracted, mag)", fontsize=10)
        else:
            ax1.set_ylabel("Δ = m_ref - m_inst (mag)", fontsize=10)
        if mode == "ensemble":
            title = "Ensemble Extinction Fit"
        elif mode == "per_star":
            title = "Per-star Bouguer Fit"
        elif mode == "median":
            title = "Median-subtract Fit"
        else:
            title = "Gaia Absolute Extinction Fit"
        if sel_date != "All Dates":
            title += f" [{sel_date}]"
        if sel_filter != "All Filters":
            title += f" [{sel_filter}]"
        ax1.set_title(title, fontsize=11, fontweight="bold")
        if len(fit_df) > 0:
            ax1.legend(loc="best", fontsize=7, ncol=min(3, max(len(fit_df), 1)))
        ax1.grid(True, alpha=0.3)

        # 잔차 히스토그램
        if not points_df.empty and "resid" in points_df.columns:
            if sel_date == "All Dates":
                grp_cols = ["date", "filter"] if "date" in points_df.columns else ["filter"]
            else:
                grp_cols = ["filter"]
            for i, (key, sub) in enumerate(points_df.groupby(grp_cols)):
                color = colors[i % len(colors)]
                resid = sub["resid"].dropna()
                if len(resid) > 0:
                    if isinstance(key, tuple):
                        lbl = "/".join(str(k) for k in key)
                    else:
                        lbl = str(key)
                    ax2.hist(resid, bins=30, alpha=0.5, color=color, label=f"{lbl} (σ={resid.std():.3f})")
        ax2.set_xlabel("Residual (mag)", fontsize=10)
        ax2.set_ylabel("Count", fontsize=10)
        ax2.set_title("Residual Histogram", fontsize=11, fontweight="bold")
        ax2.legend(loc="best", fontsize=7)
        ax2.grid(True, alpha=0.3)

        self.figure.tight_layout()
        self.canvas.draw()

    def save_plots(self):
        try:
            # tool_extinction 폴더에 저장
            out_dir = tool_extinction_dir(self.result_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            if self._current_mode == "ensemble":
                prefix = "ensemble"
            elif self._current_mode == "per_star":
                prefix = "per_star"
            elif self._current_mode == "median":
                prefix = "median"
            else:
                prefix = "gaia"
            out = out_dir / f"{prefix}_extinction_fit_plot.png"
            self.figure.savefig(out, dpi=150, bbox_inches="tight")
            self.log(f"Saved {out}")

            # 기존 경로에도 저장 (호환성)
            step11_out = legacy_step11_extinction_dir(self.result_dir)
            step11_out.mkdir(parents=True, exist_ok=True)
            out2 = step11_out / "step11_extinction_fit_plot.png"
            self.figure.savefig(out2, dpi=150, bbox_inches="tight")
            self.log(f"Saved {out2}")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to save plot:\n{e}")
