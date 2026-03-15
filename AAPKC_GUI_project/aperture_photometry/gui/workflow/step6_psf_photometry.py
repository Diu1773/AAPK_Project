"""
Step 6 – PSF Photometry  (photutils EPSFBuilder + PSFPhotometry, skippable)

Reads:
    step4_detection/detect_{fname}.csv   (det_uid, x, y)
    step5_aperture/photometry_{fname}.tsv (flux_net_adu as initial flux_0 estimate, optional)
    <FITS image>

Writes to step6_psf/:
    photometry_{fname}.tsv   – det_uid, x_fit, y_fit, mag_psf, mag_psf_err,
                                chi2, iter_found, flags_psf
    epsf_model_{filter}_{frame_stem}.fits – oversampled ePSF model (per-frame)
    residual_iter{N}_{fname}.fits – sky-subtracted residual image (per iteration)
    starsub_iter{N}_{fname}.fits  – raw image with fitted stars removed (per iteration)
    photometry_index.csv     – per-frame summary

Step can be SKIPPED: clicking "Skip PSF" marks step as complete and
passes control to Step 7 (WCS). Then Step 8 (RefBuild) will use Step 5
aperture results when PSF outputs are unavailable.
"""
from __future__ import annotations

import json
import traceback
import time
import copy
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from threading import Lock

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.nddata import NDData
from astropy.stats import sigma_clipped_stats
from scipy.spatial import cKDTree

from PyQt5.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QGroupBox, QMessageBox,
    QTextEdit, QDialog, QFormLayout, QDialogButtonBox, QProgressBar,
    QCheckBox, QSpinBox, QDoubleSpinBox, QWidget, QTabWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QSplitter, QComboBox, QListWidget,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle, Patch

from .step_window_base import StepWindowBase
from ...utils.step_paths import (
    step2_cropped_dir, step4_dir, step5_aperture_dir, step6_psf_dir,
    crop_is_active,
)
from ...utils import normalize_filter_name
from ...utils.constants import get_parallel_workers


# ── Scalar helpers ────────────────────────────────────────────────────────────

def _to_float(val, default):
    try:
        if val is None:
            return float(default)
        out = float(val)
        return out if np.isfinite(out) else float(default)
    except Exception:
        return float(default)


def _to_int(val, default):
    try:
        if val is None:
            return int(default)
        return int(float(val))
    except Exception:
        return int(default)


def _safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def _odd_int(value: float, min_value: int = 3, max_value: int | None = None) -> int:
    """Convert to odd integer within optional bounds."""
    try:
        v = int(round(float(value)))
    except Exception:
        v = int(min_value)
    v = max(int(min_value), v)
    if max_value is not None:
        v = min(int(max_value), v)
    if v % 2 == 0:
        v += 1
    if max_value is not None and v > int(max_value):
        v = int(max_value) - 1
        if v < int(min_value):
            v = int(min_value)
        if v % 2 == 0:
            v = max(int(min_value), v - 1)
    return int(v)


# Korean → ASCII translation for matplotlib (matplotlib default font lacks Korean glyphs)
_KO_TO_ASCII = {
    "신규검출 (step4 미검출)": "New (not in step4)",
    "재검출 (step4 기검출)": "Re-detected (step4)",
    "경계소스": "Edge",
}

_PSF_MODE_PRESETS = {
    "normal": {
        "psf_n_stars_max": 50,
        "psf_isolation_fwhm_mult": 3.0,
        "psf_fit_shape_fwhm_mult": 1.5,
        "psf_max_iter": 2,
        "psf_redetect_sigma": 4.0,
        "psf_duplicate_radius_fwhm_mult": 0.8,
        "psf_new_sources_cap_per_iter": 70,
        "psf_new_sources_cap_frac": 0.02,
        "psf_fit_init_max_sources": 0,
        "psf_substar_neighbor_r_fwhm_mult": 8.0,
        "psf_substar_max_sources": 1500,
        "psf_conv_new_frac": 0.02,
        "psf_flux_conv_threshold": 0.01,
        "psf_use_grouper": True,
        "psf_redetect_sharp_lo": 0.15,
        "psf_redetect_sharp_hi": 0.95,
        "psf_redetect_round_abs_max": 0.8,
    },
    "crowded": {
        "psf_n_stars_max": 30,
        "psf_isolation_fwhm_mult": 2.0,
        "psf_fit_shape_fwhm_mult": 1.2,
        "psf_max_iter": 3,
        "psf_redetect_sigma": 4.5,
        "psf_duplicate_radius_fwhm_mult": 0.4,
        "psf_new_sources_cap_per_iter": 50,
        "psf_new_sources_cap_frac": 0.015,
        "psf_fit_init_max_sources": 3000,
        "psf_substar_neighbor_r_fwhm_mult": 5.0,
        "psf_substar_max_sources": 1000,
        "psf_conv_new_frac": 0.02,
        "psf_flux_conv_threshold": 0.01,
        "psf_use_grouper": True,
        "psf_redetect_sharp_lo": 0.2,
        "psf_redetect_sharp_hi": 0.9,
        "psf_redetect_round_abs_max": 0.6,
    },
    "faint": {
        "psf_n_stars_max": 40,
        "psf_isolation_fwhm_mult": 2.5,
        "psf_fit_shape_fwhm_mult": 2.0,
        "psf_max_iter": 3,
        "psf_redetect_sigma": 3.0,
        "psf_duplicate_radius_fwhm_mult": 1.0,
        "psf_new_sources_cap_per_iter": 100,
        "psf_new_sources_cap_frac": 0.05,
        "psf_fit_init_max_sources": 0,
        "psf_substar_neighbor_r_fwhm_mult": 8.0,
        "psf_substar_max_sources": 1500,
        "psf_conv_new_frac": 0.03,
        "psf_flux_conv_threshold": 0.01,
        "psf_use_grouper": False,
        "psf_redetect_sharp_lo": 0.1,
        "psf_redetect_sharp_hi": 0.95,
        "psf_redetect_round_abs_max": 0.9,
    },
}


def _clone_psf_model(model):
    """Return a per-frame copy of PSF model to avoid thread-shared mutation."""
    try:
        return model.copy()
    except Exception:
        try:
            return copy.deepcopy(model)
        except Exception:
            return model


# ── FITS helpers ───────────────────────────────────────────────────────────────

def _get_filter_lower(fits_path: Path) -> str:
    try:
        h = fits.getheader(fits_path)
        f = h.get("FILTER", None)
        if f is None:
            return "unknown"
        return normalize_filter_name(f)
    except Exception:
        return "unknown"


def _get_exptime(fits_path: Path, default=1.0) -> float:
    try:
        h = fits.getheader(fits_path)
        for k in ("EXPTIME", "EXPOSURE", "ITIME", "ELAPTIME"):
            if k in h:
                v = float(h[k])
                if np.isfinite(v) and v > 0:
                    return v
    except Exception:
        pass
    return float(default)


# ── Detect helpers ────────────────────────────────────────────────────────────

def _load_detect_positions(fname: str, cache_dir: Path, result_dir: Path):
    candidates = [
        cache_dir / f"detect_{fname}.csv",
        step4_dir(result_dir) / f"detect_{fname}.csv",
    ]
    for p in candidates:
        if p.exists() and p.stat().st_size > 0:
            try:
                df = pd.read_csv(p)
                x_col = "x" if "x" in df.columns else ("xcenter" if "xcenter" in df.columns else None)
                y_col = "y" if "y" in df.columns else ("ycenter" if "ycenter" in df.columns else None)
                if x_col is None or y_col is None:
                    continue
                out = pd.DataFrame({"x": pd.to_numeric(df[x_col], errors="coerce"),
                                    "y": pd.to_numeric(df[y_col], errors="coerce")})
                if "det_uid" in df.columns:
                    det_uid_raw = pd.to_numeric(df["det_uid"], errors="coerce")
                    if det_uid_raw.notna().any():
                        missing = det_uid_raw.isna()
                        if missing.any():
                            fallback = np.arange(len(det_uid_raw), dtype=float)
                            det_uid_raw.loc[missing] = fallback[missing.to_numpy()]
                        out["det_uid"] = det_uid_raw.to_numpy(dtype=np.int64, copy=False)
                    else:
                        out["det_uid"] = np.arange(len(df), dtype=np.int64)
                else:
                    out["det_uid"] = np.arange(len(df), dtype=np.int64)
                for flux_col in ("flux", "peak", "amplitude"):
                    if flux_col in df.columns:
                        out["flux_init"] = pd.to_numeric(df[flux_col], errors="coerce")
                        break
                # Pass through morphology quality metrics for EPSF star selection
                for _src, _dst in (
                    ("sharpness", "sharpness"),
                    ("roundness",  "roundness"),
                    ("roundness1", "roundness"),   # DAOStarFinder; prefer plain "roundness"
                    ("elongation", "elong"),
                    ("elong",      "elong"),
                ):
                    if _src in df.columns and _dst not in out.columns:
                        out[_dst] = pd.to_numeric(df[_src], errors="coerce")
                out = out.dropna(subset=["x", "y"])
                return out
            except Exception:
                continue
    return None


def _load_fwhm_from_meta(fname: str, cache_dir: Path, result_dir: Path,
                          params_fwhm_guess=6.0) -> float:
    candidates = [
        cache_dir / f"detect_{fname}.json",
        step4_dir(result_dir) / f"detect_{fname}.json",
    ]
    for p in sorted([c for c in candidates if c.exists()],
                    key=lambda q: q.stat().st_mtime_ns, reverse=True):
        try:
            meta = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        # Prefer explicit FWHM(px); keep *_rad* as fallback for legacy metadata.
        for k in ("fwhm_med_px", "fwhm_px", "fwhm_med", "fwhm_med_rad_px"):
            v = meta.get(k, None)
            if v is not None:
                try:
                    v = float(v)
                    if np.isfinite(v) and v > 0:
                        return v
                except Exception:
                    continue
    return float(params_fwhm_guess)


# ── PSF Worker ────────────────────────────────────────────────────────────────

class Step6PSFWorker(QThread):
    """Per-frame EPSFBuilder + PSFPhotometry worker.

    Algorithm per frame:
    1. Load detected positions from detect_{fname}.csv
    2. Select bright isolated stars for EPSF building
    3. Build oversampled EPSF with EPSFBuilder
    4. Run iterative PSFPhotometry: fit → residual → detect new → re-fit
    5. Save residual FITS and epsf_model FITS
    6. Emit per-frame result
    """
    progress = pyqtSignal(int, int, str)
    worker_status = pyqtSignal(int, str, str, int)  # worker_id, frame, stage, progress(0-100)
    frame_done = pyqtSignal(str, dict)
    epsf_ready = pyqtSignal(str, str, object)      # display_key, frame_name, epsf_array (numpy)
    residual_ready = pyqtSignal(str, object, object)  # fname, residual_meta(dict), new_xy (or None)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str, str)
    log = pyqtSignal(str)

    FLAG_SAT = 1
    FLAG_EDGE = 2
    FLAG_FIT_FAIL = 4

    def __init__(self, file_list, params, data_dir, result_dir, cache_dir, use_cropped=False):
        super().__init__()
        self.file_list = list(file_list)
        self.params = params
        self.data_dir = Path(data_dir)
        self.result_dir = Path(result_dir)
        self.cache_dir = Path(cache_dir)
        self.use_cropped = use_cropped
        w_override = _to_int(getattr(self.params.P, "psf_parallel_workers", 0), 0)
        self.max_workers = max(1, w_override) if w_override > 0 else get_parallel_workers(params)
        self._workers_override = (w_override > 0)
        self._executor = None
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True
        self._log("Stop requested — finishing running frames, cancelling queued frames.")

    def _log(self, msg):
        self.log.emit(msg)

    def _resolve_fits_path(self, fname: str) -> Path | None:
        if self.use_cropped and crop_is_active(self.result_dir):
            cdir = step2_cropped_dir(self.result_dir)
            cpath = cdir / fname
            if cpath.exists():
                return cpath
            legacy = self.result_dir / "cropped" / fname
            if legacy.exists():
                return legacy
        fpath = self.data_dir / fname
        return fpath if fpath.exists() else None

    def run(self):  # noqa: C901
        try:
            try:
                from photutils.psf import EPSFBuilder, extract_stars, PSFPhotometry
                from photutils.detection import DAOStarFinder
                from photutils.background import LocalBackground, MMMBackground, Background2D, MedianBackground
                try:
                    from photutils.psf import SourceGrouper
                    _has_grouper = True
                except ImportError:
                    _has_grouper = False
                from astropy.table import Table, vstack as astropy_vstack
                import photutils as _pu
                self._log(f"photutils version: {_pu.__version__}")
            except ImportError as e:
                self.error.emit("IMPORT", f"photutils required: {e}")
                self.finished.emit({})
                return

            P = self.params.P
            output_dir = step6_psf_dir(self.result_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            GAIN = _to_float(getattr(P, "gain_e_per_adu", 1.0), 1.0)
            ZP = _to_float(getattr(P, "zp_initial", 25.0), 25.0)
            rn_e = _to_float(getattr(P, "rdnoise_e", 7.5), 7.5)
            sat_adu = _to_float(getattr(P, "saturation_adu", 60000.0), 60000.0)
            min_snr = _to_float(getattr(P, "min_snr_for_mag", 3.0), 3.0)
            fwhm_guess = _to_float(getattr(P, "fwhm_pix_guess", 6.0), 6.0)

            oversampling = _to_int(getattr(P, "psf_epsf_oversampling", 2), 2)
            epsf_size_fwhm_mult = _to_float(getattr(P, "psf_epsf_size_fwhm_mult", 4.0), 4.0)
            n_stars_max = _to_int(getattr(P, "psf_n_stars_max", 50), 50)
            isolation_mult = _to_float(getattr(P, "psf_isolation_fwhm_mult", 3.0), 3.0)
            flux_pct_lo = _to_float(getattr(P, "psf_flux_percentile_lo", 75.0), 75.0)
            flux_pct_hi = _to_float(getattr(P, "psf_flux_percentile_hi", 95.0), 95.0)
            fit_shape_fwhm_mult = _to_float(getattr(P, "psf_fit_shape_fwhm_mult", 1.5), 1.5)
            max_iter = _to_int(getattr(P, "psf_max_iter", 2), 2)
            redetect_sigma = _to_float(getattr(P, "psf_redetect_sigma", 3.5), 3.5)
            # EPSF star selection quality cuts (tighter than re-detection cuts)
            epsf_sharp_lo       = _to_float(getattr(P, "psf_epsf_sharp_lo",      0.3), 0.3)
            epsf_sharp_hi       = _to_float(getattr(P, "psf_epsf_sharp_hi",      0.8), 0.8)
            epsf_round_abs_max  = _to_float(getattr(P, "psf_epsf_round_abs_max", 0.5), 0.5)
            epsf_elong_max      = _to_float(getattr(P, "psf_epsf_elong_max",     1.3), 1.3)
            # IterativePSFPhotometry iteration mode: "new" (fast) or "all" (accurate, slow)
            fit_mode_cfg = str(getattr(P, "psf_fit_mode", "new")).strip().lower()
            if fit_mode_cfg not in ("new", "all"):
                fit_mode_cfg = "new"
            redetect_sharp_lo = _to_float(getattr(P, "psf_redetect_sharp_lo", 0.15), 0.15)
            redetect_sharp_hi = _to_float(getattr(P, "psf_redetect_sharp_hi", 0.95), 0.95)
            redetect_round_abs_max = _to_float(getattr(P, "psf_redetect_round_abs_max", 0.8), 0.8)
            duplicate_radius_px_cfg = _to_float(getattr(P, "psf_duplicate_radius_px", np.nan), np.nan)
            duplicate_radius_mult = _to_float(getattr(P, "psf_duplicate_radius_fwhm_mult", 0.8), 0.8)
            new_sources_cap_per_iter = _to_int(getattr(P, "psf_new_sources_cap_per_iter", 70), 70)
            new_sources_cap_frac = _to_float(getattr(P, "psf_new_sources_cap_frac", 0.02), 0.02)
            conv_new_frac = _to_float(getattr(P, "psf_conv_new_frac", 0.02), 0.02)
            flux_conv_threshold = _to_float(getattr(P, "psf_flux_conv_threshold", 0.01), 0.01)
            fit_init_max_sources = _to_int(getattr(P, "psf_fit_init_max_sources", 0), 0)
            use_error_image = bool(getattr(P, "psf_use_error_image", False))
            use_grouper = bool(getattr(P, "psf_use_grouper", True))
            model_mode = str(getattr(P, "psf_model_mode", "per_frame")).strip().lower()
            max_workers = max(1, int(self.max_workers))

            if model_mode != "per_frame":
                self._log(f"PSF mode '{model_mode}' is disabled; forcing per_frame")
                model_mode = "per_frame"
            use_shared_filter_epsf = False

            redetect_sigma = max(1.0, redetect_sigma)
            new_sources_cap_per_iter = max(0, new_sources_cap_per_iter)
            new_sources_cap_frac = min(max(0.0, new_sources_cap_frac), 1.0)
            conv_new_frac = min(max(0.0, conv_new_frac), 1.0)
            duplicate_radius_mult = max(0.0, duplicate_radius_mult)
            if np.isfinite(duplicate_radius_px_cfg):
                duplicate_radius_px_cfg = max(0.0, float(duplicate_radius_px_cfg))
            dedup_enabled = bool(
                (np.isfinite(duplicate_radius_px_cfg) and duplicate_radius_px_cfg > 0.0)
                or (duplicate_radius_mult > 0.0)
            )
            # Legacy sentinel values (-999/999) effectively disable morphology cuts and
            # can explode residual re-detections in crowded fields.
            if redetect_sharp_lo <= -900.0 and redetect_sharp_hi >= 900.0:
                redetect_sharp_lo, redetect_sharp_hi = 0.15, 0.95
            if redetect_round_abs_max >= 9.0:
                redetect_round_abs_max = 0.8
            # If user/state has extremely loose residual cuts (e.g. sharp=[0,1], round=2),
            # tighten them to suppress ring/halo false detections.
            if redetect_sharp_lo <= 0.01 and redetect_sharp_hi >= 0.99 and redetect_round_abs_max >= 1.5:
                redetect_sharp_lo, redetect_sharp_hi, redetect_round_abs_max = 0.15, 0.95, 0.8

            self._log(
                "PSF settings | "
                f"model_mode={model_mode} | fit_mode={fit_mode_cfg} | "
                f"max_iter={max_iter} | redetect_sigma={redetect_sigma:.2f} | "
                f"cap_iter={new_sources_cap_per_iter} | cap_frac={new_sources_cap_frac:.3f} | "
                f"use_error_image={'on' if use_error_image else 'off'} | "
                f"use_grouper={'on' if use_grouper else 'off'}"
            )
            self._log(
                f"PSF redetect cuts | sharp=[{redetect_sharp_lo:.2f},{redetect_sharp_hi:.2f}] "
                f"| |round|<={redetect_round_abs_max:.2f}"
            )
            if np.isfinite(duplicate_radius_px_cfg):
                self._log(f"PSF dedup radius: {duplicate_radius_px_cfg:.2f}px (absolute)")
            else:
                self._log(f"PSF dedup radius: {duplicate_radius_mult:.2f}xFWHM (legacy)")
            if not use_grouper:
                self._log("PSF fit mode: iterative 'new' (grouper off; photutils 2.3 requires grouper for mode='all')")
            self._log(
                f"PSF scales | epsf_cutout={epsf_size_fwhm_mult:.2f}xFWHM | "
                f"fit_window={fit_shape_fwhm_mult:.2f}xFWHM | "
                "subtract_window≈2xEPSF"
            )

            frames = list(self.file_list)
            total = len(frames)
            index_rows = []
            counters = {"processed": 0, "no_detect": 0, "no_fits": 0, "stopped": 0}
            completed = [0]
            epsf_cache: dict[str, object] = {}  # filter → epsf model
            epsf_cache_lock = Lock()
            if self._workers_override:
                self._log(f"PSF parallel workers={max_workers} (Step6 override)")
            else:
                self._log(f"PSF parallel workers={max_workers}")

            run_t0 = time.time()
            last_hb = 0.0
            last_stall_log = 0.0
            last_done_count = -1

            def _fmt_eta(sec: float) -> str:
                s = int(max(0, round(float(sec))))
                h, rem = divmod(s, 3600)
                m, ss = divmod(rem, 60)
                if h > 0:
                    return f"{h:d}:{m:02d}:{ss:02d}"
                return f"{m:02d}:{ss:02d}"

            def process_single_frame(fname: str):
                if self._stop_requested:
                    self.worker_status.emit(0, fname, "Stopped", 100)
                    return {"file": fname, "status": "stopped"}

                wid = int(threading.get_ident() % 10000)
                # Stage update: frame picked by worker
                self.progress.emit(completed[0], total, f"RUN | {fname}")
                self.worker_status.emit(wid, fname, "Load", 5)
                img_path = self._resolve_fits_path(fname)
                if img_path is None:
                    self.worker_status.emit(wid, fname, "No FITS", 100)
                    return {"file": fname, "status": "no_fits", "reason": "no FITS"}

                det_df = _load_detect_positions(fname, self.cache_dir, self.result_dir)
                if det_df is None or len(det_df) == 0:
                    self.worker_status.emit(wid, fname, "No detect", 100)
                    return {"file": fname, "status": "no_detect", "reason": "no detect csv"}

                try:
                    img = fits.getdata(img_path).astype(np.float32, copy=False)
                except Exception as e:
                    self.worker_status.emit(wid, fname, "FITS error", 100)
                    return {"file": fname, "status": "no_fits", "reason": f"FITS read: {e}"}

                this_filter = _get_filter_lower(img_path)
                exptime = _get_exptime(img_path, default=1.0)
                fwhm_med = _load_fwhm_from_meta(fname, self.cache_dir, self.result_dir, fwhm_guess)
                fwhm_safe = max(float(fwhm_med), 1.0)
                # Size controls are driven primarily by FWHM multipliers for per-frame adaptation.
                epsf_size_frame = _odd_int(
                    float(epsf_size_fwhm_mult) * fwhm_safe,
                    min_value=25,
                    max_value=101,
                )
                fit_shape_frame = _odd_int(
                    float(fit_shape_fwhm_mult) * fwhm_safe,
                    min_value=9,
                    max_value=31,
                )
                if fit_shape_frame >= epsf_size_frame:
                    fit_shape_frame = _odd_int(max(3, epsf_size_frame - 4), min_value=3, max_value=31)
                render_shape_frame = _odd_int(
                    max(float(epsf_size_frame) * 2.0, float(fit_shape_frame)),
                    min_value=11,
                    max_value=201,
                )

                epsf_cache_key = this_filter if use_shared_filter_epsf else f"{this_filter}:{fname}"
                h, w = img.shape

                try:
                    self.worker_status.emit(wid, fname, "Background", 20)
                    # 2D background with source masking: better than scalar median for
                    # crowded cluster fields where a scalar median absorbs stellar flux.
                    # box_size ~10% of image; filter_size smooths the background map.
                    _box = max(32, min(128, h // 16, w // 16))
                    try:
                        from astropy.stats import SigmaClip as _SigmaClip
                        # SigmaClip within each box rejects bright star pixels —
                        # equivalent to source masking, no external mask needed.
                        _bkg2d = Background2D(img, (_box, _box), filter_size=(5, 5),
                                              sigma_clip=_SigmaClip(sigma=3.0),
                                              bkg_estimator=MedianBackground())
                        bkg_map = np.asarray(_bkg2d.background, dtype=np.float32)
                        bkg_rms_scalar = float(_bkg2d.background_rms_median)
                        bkg_med = float(_bkg2d.background_median)
                        bkg_std = float(bkg_rms_scalar)
                        img_sub = (img - bkg_map).astype(np.float32, copy=False)
                        del bkg_map
                        self._log(f"  [BKG] Background2D | box={_box}px | "
                                  f"bkg_med={bkg_med:.2f} rms={bkg_std:.3f}")
                    except Exception as _bkg_e:
                        # Fallback to scalar sigma-clipped stats
                        self._log(f"  [BKG] Background2D failed ({_bkg_e}); using scalar median")
                        _, bkg_med, bkg_std = sigma_clipped_stats(img, sigma=3.0, maxiters=3)
                        bkg_rms_scalar = float(bkg_std)
                        img_sub = (img - float(bkg_med)).astype(np.float32, copy=False)

                    if self._stop_requested:
                        self.worker_status.emit(wid, fname, "Stopped", 100)
                        return {"file": fname, "status": "stopped", "reason": "stop requested"}

                    epsf_emit_arr = None
                    with epsf_cache_lock:
                        epsf_model = epsf_cache.get(epsf_cache_key)
                    if epsf_model is None:
                        self.worker_status.emit(wid, fname, "EPSF build", 40)
                        xy_all = det_df[["x", "y"]].to_numpy(float)
                        finite_xy = np.isfinite(xy_all[:, 0]) & np.isfinite(xy_all[:, 1])
                        xy_all = xy_all[finite_xy]
                        if len(xy_all) < 5:
                            raise RuntimeError("Too few detected sources for EPSF building")

                        if "flux_init" in det_df.columns:
                            fluxes = det_df["flux_init"].to_numpy(float)[finite_xy]
                        else:
                            xi = xy_all[:, 0].astype(int).clip(0, w - 1)
                            yi = xy_all[:, 1].astype(int).clip(0, h - 1)
                            fluxes = img_sub[yi, xi]

                        valid_flux = np.isfinite(fluxes)
                        lo = np.nanpercentile(fluxes[valid_flux], flux_pct_lo) if np.any(valid_flux) else -np.inf
                        hi = np.nanpercentile(fluxes[valid_flux], flux_pct_hi) if np.any(valid_flux) else np.inf
                        in_range = valid_flux & (fluxes >= lo) & (fluxes <= hi)

                        peak_vals = img[
                            xy_all[:, 1].astype(int).clip(0, h - 1),
                            xy_all[:, 0].astype(int).clip(0, w - 1),
                        ]
                        not_sat = peak_vals < sat_adu
                        in_range = in_range & not_sat

                        epsf_half = epsf_size_frame // 2 + 5
                        not_edge = (
                            (xy_all[:, 0] >= epsf_half) & (xy_all[:, 0] <= w - 1 - epsf_half) &
                            (xy_all[:, 1] >= epsf_half) & (xy_all[:, 1] <= h - 1 - epsf_half)
                        )
                        in_range = in_range & not_edge

                        # Morphology quality filter — only well-behaved stars in EPSF
                        # Falls back to pre-morph selection if filter is too aggressive.
                        _in_range_pre_morph = in_range.copy()
                        _morph_applied = False
                        if "sharpness" in det_df.columns:
                            _sharp = det_df["sharpness"].to_numpy(float)[finite_xy]
                            in_range = in_range & np.isfinite(_sharp) & (_sharp >= epsf_sharp_lo) & (_sharp <= epsf_sharp_hi)
                            _morph_applied = True
                        if "roundness" in det_df.columns:
                            _round = det_df["roundness"].to_numpy(float)[finite_xy]
                            in_range = in_range & np.isfinite(_round) & (np.abs(_round) <= epsf_round_abs_max)
                            _morph_applied = True
                        if "elong" in det_df.columns:
                            _elong = det_df["elong"].to_numpy(float)[finite_xy]
                            in_range = in_range & np.isfinite(_elong) & (_elong <= epsf_elong_max)
                            _morph_applied = True
                        if _morph_applied:
                            n_pre = int(np.sum(_in_range_pre_morph))
                            n_post = int(np.sum(in_range))
                            if n_post < 5:
                                self._log(
                                    f"[EPSF] morphology filter left {n_post} stars → fallback to pre-morph ({n_pre})"
                                )
                                in_range = _in_range_pre_morph
                            else:
                                self._log(
                                    f"[EPSF] morphology filter: {n_pre} → {n_post} "
                                    f"(sharp=[{epsf_sharp_lo:.2f},{epsf_sharp_hi:.2f}] "
                                    f"|round|≤{epsf_round_abs_max:.2f} elong≤{epsf_elong_max:.2f})"
                                )

                        xy_cand = xy_all[in_range]
                        fluxes_cand = fluxes[in_range]
                        if len(xy_cand) < 3:
                            raise RuntimeError("Too few candidates after flux/sat/edge filter")

                        if len(xy_cand) >= 2:
                            tree = cKDTree(xy_cand)
                            nn_dists, _ = tree.query(xy_cand, k=min(2, len(xy_cand)), workers=1)
                            nn_d = nn_dists[:, 1] if nn_dists.ndim > 1 else nn_dists
                            isolated = nn_d > isolation_mult * fwhm_med
                            n_iso = int(np.count_nonzero(isolated))
                            if np.any(isolated):
                                xy_iso = xy_cand[isolated]
                                fl_iso = fluxes_cand[isolated]
                                self._log(
                                    f"[EPSF] isolate pass | frame={fname} | cand={len(xy_cand)} | "
                                    f"isolated={n_iso} (thr={isolation_mult:.2f}xFWHM)"
                                )
                            else:
                                xy_iso = xy_cand
                                fl_iso = fluxes_cand
                                # P4-9: isolation fallback → WARN level (EPSF quality degraded)
                                self._log(
                                    f"[WARN][EPSF] isolated=0 for {fname} | "
                                    f"falling back to {len(xy_cand)} non-isolated candidates. "
                                    f"EPSF may be contaminated by neighbours. "
                                    f"Consider increasing isolation_fwhm_mult (current={isolation_mult:.1f}) "
                                    f"or using a less crowded frame."
                                )
                                self.log.emit(
                                    f"⚠ EPSF isolation fallback [{fname}]: "
                                    f"no isolated stars (thr={isolation_mult:.1f}×FWHM). "
                                    f"PSF model may be degraded — check log."
                                )
                        else:
                            xy_iso = xy_cand
                            fl_iso = fluxes_cand

                        order = np.argsort(fl_iso)[::-1]
                        xy_iso = xy_iso[order][:n_stars_max]

                        from astropy.table import Table as AstropyTable
                        star_table = AstropyTable({"x": xy_iso[:, 0], "y": xy_iso[:, 1]})
                        nddata = NDData(data=img_sub)
                        stars_extracted = extract_stars(nddata, star_table, size=epsf_size_frame)
                        if len(stars_extracted) < 3:
                            raise RuntimeError(f"Only {len(stars_extracted)} stars extracted; need ≥3")

                        builder = EPSFBuilder(
                            oversampling=oversampling,
                            maxiters=10,
                            progress_bar=False,
                            smoothing_kernel="quadratic",
                        )
                        if self._stop_requested:
                            self.worker_status.emit(wid, fname, "Stopped", 100)
                            return {"file": fname, "status": "stopped", "reason": "stop requested"}
                        epsf, _ = builder(stars_extracted)

                        # ── P4-10: EPSF quality check ─────────────────────────────────
                        try:
                            _ed = np.asarray(epsf.data, dtype=float)
                            _ed_pos = np.where(_ed > 0, _ed, 0.0)
                            _peak = float(np.nanmax(_ed_pos)) if _ed_pos.size else 0.0
                            if _peak > 0:
                                _norm = _ed_pos / _peak
                                # Double-peak check: count pixels > 50% of peak
                                # For a clean PSF, these should form one connected blob
                                _high = (_norm > 0.5).astype(float)
                                from scipy.ndimage import label as _label
                                _labeled, _n_blobs = _label(_high)
                                if _n_blobs > 1:
                                    self._log(
                                        f"[WARN][EPSF] {fname}: possible double-peak PSF "
                                        f"({_n_blobs} blobs above 50% peak). "
                                        f"Check focus/tracking."
                                    )
                                # Asymmetry check: compare quadrant sums
                                _cy, _cx = np.array(_ed.shape) // 2
                                _q1 = float(_ed[:_cy, :_cx].sum())
                                _q2 = float(_ed[:_cy, _cx:].sum())
                                _q3 = float(_ed[_cy:, :_cx].sum())
                                _q4 = float(_ed[_cy:, _cx:].sum())
                                _qtot = _q1 + _q2 + _q3 + _q4
                                if _qtot > 0:
                                    _qmax = max(_q1, _q2, _q3, _q4) / _qtot
                                    if _qmax > 0.45:  # >45% in one quadrant = asymmetric
                                        self._log(
                                            f"[WARN][EPSF] {fname}: asymmetric PSF "
                                            f"(max quadrant fraction={_qmax:.2f}). "
                                            f"Possible tracking drift or coma."
                                        )
                        except Exception:
                            pass
                        # ─────────────────────────────────────────────────────────────

                        # ── IRAF SUBSTAR-style iterative PSF rebuild ──────────────────
                        # For each rebuild pass:
                        #   1. Render model of ALL detected sources (rough flux)
                        #   2. Render model of PSF-selection stars only
                        #   3. cleaned = img_sub - all_model + psf_star_model
                        #      → each PSF star cutout is free of neighbours
                        #   4. Re-extract cutouts from cleaned image → rebuild EPSF
                        n_substar_iters = _to_int(getattr(P, "psf_substar_iters", 1), 1)
                        substar_neighbor_r_mult = _to_float(getattr(P, "psf_substar_neighbor_r_fwhm_mult", 8.0), 8.0)
                        substar_max_sources = _to_int(getattr(P, "psf_substar_max_sources", 1500), 1500)
                        if n_substar_iters > 0 and len(xy_all) > 1:
                            try:
                                from photutils.datasets import make_model_image as _mmi
                                from astropy.nddata import NDData as _NDData

                                # Use the detection-stage flux already computed above
                                # (from det_df["flux_init"] or peak-pixel fallback).
                                # This avoids re-running aperture photometry and is
                                # consistent with the positions in xy_all.
                                _all_flux = np.where(
                                    np.isfinite(fluxes) & (fluxes > 0),
                                    fluxes,
                                    0.0,
                                )
                                # Speed optimization:
                                # substar neighbor-cleaning needs sources affecting PSF stars,
                                # not necessarily every detection in the frame.
                                _neighbor_r = max(
                                    float(substar_neighbor_r_mult) * float(fwhm_safe),
                                    float(epsf_size_frame),
                                )
                                _src_tree = cKDTree(np.asarray(xy_all, dtype=float))
                                _neighbor_set = set()
                                for _px, _py in np.asarray(xy_iso, dtype=float):
                                    _hits = _src_tree.query_ball_point([float(_px), float(_py)], r=float(_neighbor_r))
                                    _neighbor_set.update(int(h) for h in _hits)
                                if _neighbor_set:
                                    _idx_nei = np.array(sorted(_neighbor_set), dtype=int)
                                else:
                                    _idx_nei = np.arange(len(xy_all), dtype=int)

                                if substar_max_sources > 0 and len(_idx_nei) > substar_max_sources:
                                    _fsel = np.asarray(_all_flux[_idx_nei], dtype=float)
                                    _fsel = np.where(np.isfinite(_fsel), _fsel, -np.inf)
                                    _ord = np.argsort(_fsel)[::-1][:int(substar_max_sources)]
                                    _idx_nei = _idx_nei[_ord]

                                _xy_sub = np.asarray(xy_all[_idx_nei], dtype=float)
                                _all_flux_sub = np.asarray(_all_flux[_idx_nei], dtype=float)
                                _psf_nn_tree = cKDTree(_xy_sub) if len(_xy_sub) else None
                                self._log(
                                    f"[EPSF] substar sources | frame={fname} | "
                                    f"all={len(xy_all)} near_psf={len(_xy_sub)} "
                                    f"(r={_neighbor_r:.1f}px, cap={substar_max_sources})"
                                )

                                _render_sz = int(epsf_size_frame)
                                _rough_epsf = epsf

                                for _si in range(n_substar_iters):
                                    # Full source model (all detected)
                                    _all_tbl = AstropyTable({
                                        "x_0": np.asarray(_xy_sub[:, 0], dtype=float),
                                        "y_0": np.asarray(_xy_sub[:, 1], dtype=float),
                                        "flux": _all_flux_sub,
                                    })
                                    _full_model = np.asarray(
                                        _mmi(
                                            img_sub.shape,
                                            _clone_psf_model(_rough_epsf),
                                            _all_tbl,
                                            model_shape=(_render_sz, _render_sz),
                                            x_name="x_0",
                                            y_name="y_0",
                                        ),
                                        dtype=float,
                                    )

                                    # PSF-star-only model (add back after subtraction)
                                    _psf_flux = np.zeros(len(xy_iso), dtype=float)
                                    if _psf_nn_tree is not None and len(xy_iso):
                                        _d_psf, _i_psf = _psf_nn_tree.query(
                                            np.asarray(xy_iso, dtype=float), k=1, workers=1
                                        )
                                        _psf_flux = _all_flux_sub[np.asarray(_i_psf, dtype=int)]
                                    _psf_tbl = AstropyTable({
                                        "x_0": np.asarray(xy_iso[:, 0], dtype=float),
                                        "y_0": np.asarray(xy_iso[:, 1], dtype=float),
                                        "flux": _psf_flux,
                                    })
                                    _psf_only_model = np.asarray(
                                        _mmi(
                                            img_sub.shape,
                                            _clone_psf_model(_rough_epsf),
                                            _psf_tbl,
                                            model_shape=(_render_sz, _render_sz),
                                            x_name="x_0",
                                            y_name="y_0",
                                        ),
                                        dtype=float,
                                    )

                                    # Neighbour-cleaned image:
                                    # img_sub - all_model + psf_only_model
                                    # ≡ img_sub - neighbour_model
                                    _cleaned_img = (
                                        img_sub.astype(float)
                                        - _full_model
                                        + _psf_only_model
                                    )
                                    _nddata_clean = _NDData(data=_cleaned_img)
                                    _stars_clean = extract_stars(
                                        _nddata_clean, star_table, size=epsf_size_frame
                                    )
                                    if len(_stars_clean) < 3:
                                        self._log(
                                            f"[EPSF] substar {_si+1}/{n_substar_iters}"
                                            f" | too few clean stars ({len(_stars_clean)})"
                                            " → stop"
                                        )
                                        break
                                    _rough_epsf, _ = builder(_stars_clean)
                                    self._log(
                                        f"[EPSF] substar {_si+1}/{n_substar_iters}"
                                        f" | n_psf={len(_stars_clean)}"
                                        f" | neighbours from {len(_xy_sub)} sources"
                                    )

                                epsf = _rough_epsf

                            except Exception as _se:
                                self._log(
                                    f"[EPSF] substar rebuild error: {_se}"
                                    " | using initial EPSF"
                                )

                        epsf_emit_arr = epsf.data.copy()

                        if use_shared_filter_epsf:
                            epsf_path = output_dir / f"epsf_model_{this_filter}.fits"
                        else:
                            epsf_path = output_dir / f"epsf_model_{this_filter}_{Path(fname).stem}.fits"
                        hdr = fits.Header()
                        hdr["FILTER"] = this_filter
                        hdr["OVERSAMPL"] = oversampling
                        hdr["NSTARS"] = len(stars_extracted)
                        hdr["EPSFSIZE"] = int(epsf_size_frame)
                        fits.writeto(str(epsf_path), epsf.data.astype(np.float32), hdr, overwrite=True)
                        self._log(
                            f"[EPSF] filter={this_filter} | "
                            f"n_stars={len(stars_extracted)} | oversampling={oversampling} | "
                            f"epsf_size={epsf_size_frame} | fit_shape={fit_shape_frame}"
                        )
                        with epsf_cache_lock:
                            if epsf_cache_key not in epsf_cache:
                                epsf_cache[epsf_cache_key] = epsf
                            epsf_model = epsf_cache[epsf_cache_key]

                    from astropy.table import Table as AstropyTable
                    xy_det = det_df[["x", "y"]].to_numpy(float)
                    finite_xy = np.isfinite(xy_det[:, 0]) & np.isfinite(xy_det[:, 1])
                    xy_det = xy_det[finite_xy]
                    det_uids = det_df["det_uid"].to_numpy(int)[finite_xy]

                    # Remove edge detections that cannot support fit window.
                    edge_init = fit_shape_frame // 2 + 2
                    valid_init = (
                        (xy_det[:, 0] >= edge_init) & (xy_det[:, 0] < (w - edge_init)) &
                        (xy_det[:, 1] >= edge_init) & (xy_det[:, 1] < (h - edge_init))
                    )
                    n_init_drop = int(np.count_nonzero(~valid_init))
                    xy_det = xy_det[valid_init]
                    det_uids = det_uids[valid_init]
                    if len(xy_det) == 0:
                        return {
                            "file": fname,
                            "status": "no_valid_init",
                            "reason": f"all detections near edge for fit_shape={fit_shape_frame}",
                        }

                    # Exclude saturated sources from PSF fitting.
                    # EPSF cannot model saturated profiles; including them degrades
                    # the fit for nearby unsaturated sources as well.
                    xi_init = xy_det[:, 0].astype(int).clip(0, w - 1)
                    yi_init = xy_det[:, 1].astype(int).clip(0, h - 1)
                    not_sat_init = img[yi_init, xi_init] < sat_adu
                    n_sat_drop = int(np.count_nonzero(~not_sat_init))
                    if n_sat_drop > 0:
                        self._log(
                            f"  [init_params] excluded {n_sat_drop} saturated sources "
                            f"(peak ≥ {sat_adu:.0f} ADU) from PSF fitting"
                        )
                    xy_det = xy_det[not_sat_init]
                    det_uids = det_uids[not_sat_init]
                    if len(xy_det) == 0:
                        return {
                            "file": fname,
                            "status": "no_valid_init",
                            "reason": f"all detections saturated (sat_adu={sat_adu:.0f})",
                        }

                    ap_tsv = step5_aperture_dir(self.result_dir) / f"photometry_{fname}.tsv"
                    flux_init_map = {}
                    if ap_tsv.exists():
                        try:
                            df_ap = pd.read_csv(ap_tsv, sep="\t")
                            _ap_cols = set(df_ap.columns)
                            if "det_uid" in _ap_cols:
                                _uid = pd.to_numeric(df_ap["det_uid"], errors="coerce")
                                # Use ADU flux (matches PSF fitting image units).
                                # flux_net_adu is the sky-subtracted aperture flux in ADU.
                                # flux_e is in electrons = flux_net_adu × GAIN (10× smaller
                                # for gain=0.1); using electrons as flux_0 shifts the LM
                                # optimizer 10× from the true minimum and causes flux
                                # redistribution errors in crowded group fits.
                                if "flux_net_adu" in _ap_cols:
                                    _flx = pd.to_numeric(df_ap["flux_net_adu"], errors="coerce")
                                elif "flux_e" in _ap_cols:
                                    _flx = pd.to_numeric(df_ap["flux_e"], errors="coerce") / max(GAIN, 1e-6)
                                else:
                                    _flx = None
                                if _flx is not None:
                                    _ok = _uid.notna() & _flx.notna() & (_flx > 0)
                                    if _ok.any():
                                        for _u, _v in zip(
                                            _uid.loc[_ok].to_numpy(dtype=np.int64, copy=False),
                                            _flx.loc[_ok].to_numpy(dtype=float, copy=False),
                                        ):
                                            flux_init_map[int(_u)] = float(_v)
                        except Exception:
                            pass

                    default_flux = max(1.0, float(bkg_std) * 10.0)
                    init_flux_list = []
                    for _uid, (x0, y0) in zip(det_uids, xy_det):
                        v = flux_init_map.get(int(_uid), np.nan)
                        if np.isfinite(v) and float(v) > 0:
                            init_flux_list.append(float(v))
                            continue
                        xi0 = int(np.clip(round(float(x0)), 0, w - 1))
                        yi0 = int(np.clip(round(float(y0)), 0, h - 1))
                        pv = _safe_float(img_sub[yi0, xi0], np.nan)
                        if not np.isfinite(pv):
                            pv = default_flux
                        init_flux_list.append(max(default_flux, float(pv)))
                    init_flux = np.asarray(init_flux_list, dtype=float)
                    if fit_init_max_sources > 0 and len(xy_det) > fit_init_max_sources:
                        _ord_fit = np.argsort(np.where(np.isfinite(init_flux), init_flux, -np.inf))[::-1][:fit_init_max_sources]
                        xy_det = xy_det[_ord_fit]
                        det_uids = det_uids[_ord_fit]
                        init_flux = init_flux[_ord_fit]
                        self._log(
                            f"  [INIT] capped initial fit sources: kept={len(xy_det)} "
                            f"(psf_fit_init_max_sources={fit_init_max_sources})"
                        )
                    init_params = AstropyTable({"x_0": xy_det[:, 0], "y_0": xy_det[:, 1], "flux_0": init_flux})

                    # ── IterativePSFPhotometry  (Stetson 1987 / DAOPHOT style) ──────────
                    # localbkg_estimator=None: background already removed by Background2D.
                    # SourceGrouper(2.5×FWHM): Stetson's critical separation — sources
                    #   within this radius are fitted SIMULTANEOUSLY, correctly accounting
                    #   for mutual flux contamination (crowded cluster requirement).
                    # mode='all': every iteration refits ALL sources on the original data
                    #   (not just the residual), allowing later-found faint stars to improve
                    #   the fit of already-found bright neighbors.
                    # Note: photutils 2.3.0 introduced a 'flat model' that eliminates
                    #   the compound-model recursion crash seen in 2.2.0 for large groups.
                    #   If a RecursionError occurs, we fall back to no grouper.
                    # ─────────────────────────────────────────────────────────────────────

                    # Error image for photon-noise-correct flux_err
                    if use_error_image:
                        try:
                            from photutils.utils import calc_total_error
                            error_img = calc_total_error(img_sub, bkg_rms_scalar, GAIN)
                        except Exception:
                            error_img = None
                    else:
                        error_img = None

                    # Re-detection finder (used internally by IterativePSFPhotometry)
                    # Per-filter sigma overrides default redetect_sigma when specified.
                    _sigma_key = f"psf_redetect_sigma_{this_filter}"
                    _sigma_override = _to_float(getattr(P, _sigma_key, float("nan")), float("nan"))
                    if np.isfinite(_sigma_override) and _sigma_override > 0:
                        redetect_sigma_eff = _sigma_override
                    else:
                        redetect_sigma_eff = float(redetect_sigma)

                    redetect_finder = DAOStarFinder(
                        fwhm=fwhm_safe,
                        threshold=redetect_sigma_eff * bkg_std,
                        peakmax=sat_adu,
                        sharplo=redetect_sharp_lo,
                        sharphi=redetect_sharp_hi,
                        roundlo=-redetect_round_abs_max,
                        roundhi=redetect_round_abs_max,
                    )
                    if redetect_sigma_eff != redetect_sigma:
                        self._log(
                            f"  [REDETECT] filter={this_filter} sigma override: {redetect_sigma:.2f} -> {redetect_sigma_eff:.2f}"
                        )

                    ap_rad = max(int(round(fwhm_safe * 2.0)), fit_shape_frame // 2 + 1)

                    def _build_iterative_phot(with_grouper: bool, n_seed: int):
                        from photutils.psf import IterativePSFPhotometry
                        import inspect as _ins
                        psf_m = _clone_psf_model(epsf_model)
                        kw: dict = dict(
                            psf_model=psf_m,
                            fit_shape=fit_shape_frame,
                            finder=redetect_finder,
                            aperture_radius=ap_rad,
                            localbkg_estimator=None,
                        )
                        sig = _ins.signature(IterativePSFPhotometry).parameters
                        if "maxiters" in sig:
                            kw["maxiters"] = max_iter
                        if "mode" in sig:
                            # mode='new' (default): iter1 fits all, iter2+ only new sources — fast
                            # mode='all': every iteration refits ALL sources — accurate but O(n×iter)
                            #   → can be slow for large fields; a performance warning is logged
                            if fit_mode_cfg == "all" and n_seed > 800:
                                self._log(
                                    f"  [PSF] fit_mode='all' | {n_seed} sources "
                                    "— expect significantly slower fitting"
                                )
                            kw["mode"] = fit_mode_cfg
                        if with_grouper and _has_grouper and "grouper" in sig:
                            kw["grouper"] = SourceGrouper(
                                min_separation=2.5 * fwhm_safe
                            )
                        return IterativePSFPhotometry(**kw)

                    def _results_to_init_params(results_tbl, photometry_obj=None):
                        if results_tbl is None or len(results_tbl) == 0:
                            return None
                        if photometry_obj is not None and hasattr(photometry_obj, "results_to_init_params"):
                            try:
                                tbl = photometry_obj.results_to_init_params()
                                if tbl is not None and len(tbl) > 0:
                                    return tbl
                            except Exception as _ri:
                                self._log(f"  [PSF] results_to_init_params fallback: {_ri}")
                        try:
                            cols = list(results_tbl.colnames)
                            x_col = "x_fit" if "x_fit" in cols else ("x_0" if "x_0" in cols else None)
                            y_col = "y_fit" if "y_fit" in cols else ("y_0" if "y_0" in cols else None)
                            f_col = next((c for c in ("flux_fit", "flux", "flux_0") if c in cols), None)
                            if x_col is None or y_col is None or f_col is None:
                                return None
                            x_arr = np.asarray(results_tbl[x_col], dtype=float)
                            y_arr = np.asarray(results_tbl[y_col], dtype=float)
                            f_arr = np.asarray(results_tbl[f_col], dtype=float)
                            keep = (
                                np.isfinite(x_arr) &
                                np.isfinite(y_arr) &
                                np.isfinite(f_arr) &
                                (f_arr > 0)
                            )
                            if not np.any(keep):
                                return None
                            return AstropyTable({
                                "x_0": x_arr[keep],
                                "y_0": y_arr[keep],
                                "flux_0": f_arr[keep],
                            })
                        except Exception:
                            return None

                    def _run_iterative_fit(seed_params, stage_label: str):
                        fit_reason = None
                        fit_photometry = None
                        fit_results = None
                        for _attempt, _use_grouper in enumerate(attempt_plan):
                            if self._stop_requested:
                                return None, None, "stopped"
                            try:
                                fit_photometry = _build_iterative_phot(
                                    with_grouper=_use_grouper,
                                    n_seed=len(seed_params),
                                )
                                if _attempt == 1:
                                    self._log(
                                        f"  [PSF] {stage_label} retry without SourceGrouper (fallback)"
                                    )
                                call_kw = {"init_params": seed_params}
                                if error_img is not None:
                                    call_kw["error"] = error_img
                                fit_results = fit_photometry(img_sub, **call_kw)
                                fit_reason = None
                                break
                            except RecursionError as _re:
                                self._log(
                                    f"  [PSF] {stage_label} RecursionError with grouper "
                                    f"(photutils<2.3 compound-model bug): {_re}. Retrying without grouper."
                                )
                                fit_reason = str(_re)
                                if _attempt + 1 < len(attempt_plan):
                                    continue
                            except Exception as _fe:
                                fit_reason = str(_fe)
                                self._log(
                                    f"  [PSF] {stage_label} fit failed (attempt {_attempt+1}): {fit_reason}"
                                )
                                if _attempt + 1 < len(attempt_plan):
                                    continue
                                break
                        return fit_photometry, fit_results, fit_reason

                    def _render_model_from_results(results_tbl, photometry_obj=None):
                        if results_tbl is None or len(results_tbl) == 0:
                            return None
                        try:
                            cols = list(results_tbl.colnames)
                            x_col = "x_fit" if "x_fit" in cols else ("x_0" if "x_0" in cols else None)
                            y_col = "y_fit" if "y_fit" in cols else ("y_0" if "y_0" in cols else None)
                            f_col = next((c for c in ("flux_fit", "flux", "flux_0") if c in cols), None)
                            if x_col is None or y_col is None or f_col is None:
                                return None
                            x_arr = np.asarray(results_tbl[x_col], dtype=float)
                            y_arr = np.asarray(results_tbl[y_col], dtype=float)
                            f_arr = np.asarray(results_tbl[f_col], dtype=float)
                            keep = (
                                np.isfinite(x_arr) &
                                np.isfinite(y_arr) &
                                np.isfinite(f_arr) &
                                (f_arr > 0)
                            )
                            if not np.any(keep):
                                return None
                            from photutils.datasets import make_model_image as _make_model_image
                            pt = AstropyTable()
                            pt["x_0"] = np.asarray(x_arr[keep], dtype=float)
                            pt["y_0"] = np.asarray(y_arr[keep], dtype=float)
                            pt["flux"] = np.asarray(f_arr[keep], dtype=float)
                            out = _make_model_image(
                                img_sub.shape,
                                _clone_psf_model(epsf_model),
                                pt,
                                model_shape=(int(render_shape_frame), int(render_shape_frame)),
                                x_name="x_0",
                                y_name="y_0",
                            )
                            return np.asarray(out, dtype=np.float32)
                        except Exception as _re:
                            self._log(f"  [DIAG] wide model render failed: {_re}")
                            if photometry_obj is not None:
                                try:
                                    out = photometry_obj.make_model_image(
                                        img_sub.shape,
                                        psf_shape=(int(render_shape_frame), int(render_shape_frame)),
                                    )
                                    return np.asarray(out, dtype=np.float32)
                                except Exception as _pe:
                                    self._log(f"  [DIAG] make_model_image fallback failed: {_pe}")
                        return None

                    fit_fail_reason = None
                    phot_result = None
                    photometry = None
                    model_img = None

                    self.progress.emit(completed[0], total, f"FIT | {fname}")
                    self.worker_status.emit(wid, fname, "PSF fit", 70)
                    attempt_plan = [False]
                    if use_grouper and _has_grouper:
                        attempt_plan = [True, False]
                    photometry, phot_result, fit_fail_reason = _run_iterative_fit(init_params, "pass1")
                    if fit_fail_reason == "stopped":
                        return {"file": fname, "status": "stopped"}
                    refine_pass_max_sources = 2500
                    if phot_result is not None and len(phot_result) > 0:
                        refine_init = _results_to_init_params(phot_result, photometry_obj=photometry)
                        if refine_init is not None and len(refine_init) > 0:
                            if len(refine_init) <= refine_pass_max_sources:
                                photometry_refine, phot_result_refine, refine_reason = _run_iterative_fit(
                                    refine_init,
                                    "pass2",
                                )
                                if refine_reason == "stopped":
                                    return {"file": fname, "status": "stopped"}
                                if phot_result_refine is not None and len(phot_result_refine) > 0:
                                    self._log(
                                        f"  [PSF] refine pass accepted | seed={len(refine_init)} "
                                        f"fit={len(phot_result_refine)}"
                                    )
                                    photometry = photometry_refine
                                    phot_result = phot_result_refine
                                    fit_fail_reason = None
                                elif refine_reason:
                                    self._log(
                                        f"  [PSF] refine pass failed; keeping pass1 solution | {refine_reason}"
                                    )
                            else:
                                self._log(
                                    f"  [PSF] refine pass skipped | seed={len(refine_init)} "
                                    f"> {refine_pass_max_sources}"
                                )

                    raw_iter_counts: dict[int, int] = {}
                    n_new_raw_total = 0
                    n_new_kept_total = 0
                    raw_new_xy = np.zeros((0, 2), dtype=float)

                    if phot_result is not None and len(phot_result) > 0 and "iter_detected" in phot_result.colnames:
                        try:
                            _x0 = np.asarray(phot_result["x_fit"], dtype=float)
                            _y0 = np.asarray(phot_result["y_fit"], dtype=float)
                            _it_raw0 = np.asarray(phot_result["iter_detected"], dtype=float)
                            _it0 = np.where(np.isfinite(_it_raw0), _it_raw0, 1.0).astype(int)
                            _uniq_raw, _cnt_raw = np.unique(_it0[_it0 > 1], return_counts=True) if np.any(_it0 > 1) else ([], [])
                            raw_iter_counts = {int(i): int(c) for i, c in zip(_uniq_raw, _cnt_raw)}
                            n_new_raw_total = int(np.sum(_it0 > 1))
                            n_new_kept_total = n_new_raw_total
                            _m_raw_xy = np.isfinite(_x0) & np.isfinite(_y0) & (_it0 > 1)
                            if np.any(_m_raw_xy):
                                raw_new_xy = np.column_stack([_x0[_m_raw_xy], _y0[_m_raw_xy]])
                                _n_show = min(6, len(raw_new_xy))
                                _pts = ", ".join(
                                    [f"({raw_new_xy[i,0]:.2f},{raw_new_xy[i,1]:.2f})" for i in range(_n_show)]
                                )
                                self._log(
                                    f"  [RAWXY] iter>1 raw first={_n_show}/{len(raw_new_xy)} | {_pts}"
                                )
                                try:
                                    _tree_seed = cKDTree(np.asarray(xy_det, dtype=float))
                                    _d_seed, _ = _tree_seed.query(raw_new_xy, k=1, workers=1)
                                    _seed_tol = 1.0  # px
                                    _n_near = int(np.sum(np.asarray(_d_seed, dtype=float) <= _seed_tol))
                                    _n_far = int(len(_d_seed) - _n_near)
                                    self._log(
                                        f"  [RAWXY] vs Step4 seed | near<=1.00px={_n_near} | far={_n_far}"
                                    )
                                except Exception:
                                    pass
                        except Exception:
                            pass

                    # De-duplicate residual re-detections against iter1 fitted sources first.
                    # Keep this radius in PIXELS (not xFWHM) to avoid deleting legitimate
                    # nearby sources in crowded fields.
                    if (
                        phot_result is not None
                        and len(phot_result) > 0
                        and "iter_detected" in phot_result.colnames
                        and dedup_enabled
                    ):
                        try:
                            _x = np.asarray(phot_result["x_fit"], dtype=float)
                            _y = np.asarray(phot_result["y_fit"], dtype=float)
                            _it_raw = np.asarray(phot_result["iter_detected"], dtype=float)
                            _it = np.where(np.isfinite(_it_raw), _it_raw, 1.0).astype(int)
                            _finite = np.isfinite(_x) & np.isfinite(_y)
                            _idx_base = np.where(_finite & (_it <= 1))[0]
                            _idx_new = np.where(_finite & (_it > 1))[0]
                            if len(_idx_base) and len(_idx_new):
                                _xy_base = np.column_stack([_x[_idx_base], _y[_idx_base]])
                                _xy_new = np.column_stack([_x[_idx_new], _y[_idx_new]])
                                _tree = cKDTree(_xy_base)
                                _dnn, _ = _tree.query(_xy_new, k=1, workers=1)
                                if len(_dnn):
                                    try:
                                        _dnn_arr = np.asarray(_dnn, dtype=float)
                                        self._log(
                                            "  [DEDUP] d_nn(iter2->iter1) px | "
                                            f"p50={np.nanpercentile(_dnn_arr, 50):.2f} "
                                            f"p90={np.nanpercentile(_dnn_arr, 90):.2f} "
                                            f"p99={np.nanpercentile(_dnn_arr, 99):.2f}"
                                        )
                                    except Exception:
                                        pass
                                if np.isfinite(duplicate_radius_px_cfg):
                                    _dup_r_px = float(duplicate_radius_px_cfg)
                                else:
                                    _dup_r_px = float(max(0.0, duplicate_radius_mult * fwhm_safe))
                                _keep_new = np.asarray(_dnn, dtype=float) > _dup_r_px
                                if np.any(~_keep_new):
                                    _drop_n = int(np.sum(~_keep_new))
                                    _keep_mask = np.ones(len(phot_result), dtype=bool)
                                    _keep_mask[_idx_new[~_keep_new]] = False
                                    phot_result = phot_result[_keep_mask]
                                    _it2_raw = np.asarray(phot_result["iter_detected"], dtype=float)
                                    _it2 = np.where(np.isfinite(_it2_raw), _it2_raw, 1.0).astype(int)
                                    n_new_kept_total = int(np.sum(_it2 > 1))
                                    self._log(
                                        f"  [DEDUP] dropped near-duplicate iter>1 sources: {_drop_n} "
                                        f"(r<{_dup_r_px:.2f}px)"
                                    )
                        except Exception as _de:
                            self._log(f"  [DEDUP] skipped: {_de}")

                    # Apply residual new-source cap (absolute + fraction of initial seed list)
                    # after dedup, so cap budget is used on unique candidates.
                    if phot_result is not None and len(phot_result) > 0 and "iter_detected" in phot_result.colnames:
                        try:
                            _cols_cap = list(phot_result.colnames)
                            _it_raw = np.asarray(phot_result["iter_detected"], dtype=float)
                            _it = np.where(np.isfinite(_it_raw), _it_raw, 1.0).astype(int)
                            _new_now = int(np.sum(_it > 1))

                            _cap_abs = int(new_sources_cap_per_iter) if int(new_sources_cap_per_iter) > 0 else None
                            _cap_frac_n = None
                            if float(new_sources_cap_frac) > 0:
                                _cap_frac_n = int(np.floor(float(new_sources_cap_frac) * max(1, len(init_params))))

                            if _cap_abs is not None and _cap_frac_n is not None:
                                _cap_new = min(_cap_abs, _cap_frac_n)
                            elif _cap_abs is not None:
                                _cap_new = _cap_abs
                            else:
                                _cap_new = _cap_frac_n

                            if _cap_new is not None:
                                _cap_new = max(0, int(_cap_new))
                                if _new_now > _cap_new:
                                    _ff_col_cap = next((c for c in ("flux_fit", "flux") if c in _cols_cap), None)
                                    _flux_all = (
                                        np.asarray(phot_result[_ff_col_cap], dtype=float)
                                        if _ff_col_cap is not None else
                                        np.full(len(phot_result), np.nan, dtype=float)
                                    )
                                    _idx_new = np.where(_it > 1)[0]
                                    if len(_idx_new):
                                        _m = _flux_all[_idx_new]
                                        _m = np.where(np.isfinite(_m), _m, -np.inf)
                                        _order = np.argsort(_m)[::-1]
                                        _keep_new_idx = _idx_new[_order[:_cap_new]]
                                        _keep_mask = (_it <= 1)
                                        _keep_mask[_keep_new_idx] = True
                                        phot_result = phot_result[_keep_mask]
                                        _it_kept_raw = np.asarray(phot_result["iter_detected"], dtype=float)
                                        _it_kept = np.where(np.isfinite(_it_kept_raw), _it_kept_raw, 1.0).astype(int)
                                        _new_after_cap = int(np.sum(_it_kept > 1))
                                        self._log(
                                            f"  [CAP] new sources capped | raw={n_new_raw_total} dedup={_new_now} kept={_new_after_cap} "
                                            f"(cap={_cap_new}, abs={new_sources_cap_per_iter}, frac={new_sources_cap_frac:.3f})"
                                        )
                                        n_new_kept_total = _new_after_cap
                                else:
                                    n_new_kept_total = _new_now
                            else:
                                n_new_kept_total = _new_now
                        except Exception as _ce:
                            self._log(f"  [CAP] cap logic skipped: {_ce}")

                    # ── Diagnostics ──────────────────────────────────────────────────────
                    if phot_result is not None and len(phot_result) > 0:
                        try:
                            _cols = list(phot_result.colnames)
                            _ff_col = next((c for c in ("flux_fit", "flux") if c in _cols), None)
                            if _ff_col:
                                _ff = np.asarray(phot_result[_ff_col], dtype=float)
                                _ff_pos = _ff[np.isfinite(_ff) & (_ff > 0)]
                                self._log(
                                    f"  [DIAG] {fname} | n_fit={len(_ff)} | "
                                    f"flux_fit: n>0={len(_ff_pos)} "
                                    f"med={np.nanmedian(_ff):.2f} max={np.nanmax(_ff):.2f} | "
                                    f"img_sub peak={float(np.nanmax(img_sub)):.2f} bkg_std={float(bkg_std):.3f}"
                                )
                            if "group_size" in _cols:
                                _gs = np.asarray(phot_result["group_size"], dtype=int)
                                self._log(
                                    f"  [DIAG] group_size: max={_gs.max()} "
                                    f"med={np.median(_gs):.0f} "
                                    f"n_groups={len(np.unique(phot_result['group_id']))}"
                                )
                            if "iter_detected" in _cols:
                                _idet_raw = np.asarray(phot_result["iter_detected"], dtype=float)
                                _idet = np.where(np.isfinite(_idet_raw), _idet_raw, 1.0).astype(int)
                                self._log(
                                    f"  [DIAG] iters used={_idet.max()} | "
                                    f"new sources (iter>1)={int(np.sum(_idet > 1))}"
                                )
                        except Exception as _de:
                            self._log(f"  [DIAG] diag error: {_de}")

                    # ── Model image & residual ────────────────────────────────────────────
                    residual = img_sub.copy()
                    if phot_result is not None and len(phot_result) > 0:
                        model_img = _render_model_from_results(phot_result, photometry_obj=photometry)
                        if model_img is not None:
                            residual = img_sub - model_img
                            self._log(
                                f"  [DIAG] model_img sum={float(np.nansum(model_img)):.2f} "
                                f"peak={float(np.nanmax(model_img)):.2f} | "
                                f"img_sub peak={float(np.nanmax(img_sub)):.2f} | "
                                f"subtract_shape={render_shape_frame}"
                            )

                    _, _, res_std = sigma_clipped_stats(residual, sigma=3.0, maxiters=3)

                    # n_new_total: kept sources first detected in iteration > 1
                    n_new_total = int(n_new_kept_total)
                    if n_new_total <= 0 and phot_result is not None and "iter_detected" in phot_result.colnames:
                        _iter_raw = np.asarray(phot_result["iter_detected"], dtype=float)
                        _iter_safe = np.where(np.isfinite(_iter_raw), _iter_raw, 1.0).astype(int)
                        n_new_total = int(np.sum(_iter_safe > 1))

                    # ── Save starsub / residual for cutout viewer ─────────────────────────
                    fit_xy = np.zeros((0, 2), dtype=float)
                    fit_flux = np.zeros((0,), dtype=float)
                    fit_iter = np.zeros((0,), dtype=int)
                    n_fit = 0
                    if phot_result is not None and len(phot_result) > 0:
                        try:
                            x_it = np.asarray(phot_result["x_fit"], dtype=float)
                            y_it = np.asarray(phot_result["y_fit"], dtype=float)
                            if "flux_fit" in phot_result.colnames:
                                f_it = np.asarray(phot_result["flux_fit"], dtype=float)
                            elif "flux" in phot_result.colnames:
                                f_it = np.asarray(phot_result["flux"], dtype=float)
                            else:
                                f_it = np.full(len(x_it), np.nan, dtype=float)
                            if "iter_detected" in phot_result.colnames:
                                it_raw = np.asarray(phot_result["iter_detected"], dtype=float)
                                it_safe = np.where(np.isfinite(it_raw), it_raw, 1.0).astype(int)
                            else:
                                it_safe = np.ones(len(x_it), dtype=int)
                            valid_it = np.isfinite(x_it) & np.isfinite(y_it) & np.isfinite(f_it)
                            if np.any(valid_it):
                                fit_xy = np.column_stack([x_it[valid_it], y_it[valid_it]])
                                fit_flux = f_it[valid_it]
                                fit_iter = it_safe[valid_it]
                            n_fit = int(len(fit_xy))
                        except Exception:
                            pass

                    starsub_raw = img - model_img if model_img is not None else img_sub + float(bkg_med)
                    init_xy_ui = np.column_stack(
                        [np.asarray(init_params["x_0"], dtype=float), np.asarray(init_params["y_0"], dtype=float)]
                    ) if len(init_params) > 0 else np.zeros((0, 2), dtype=float)
                    iter_max_used = int(np.max(fit_iter)) if len(fit_iter) else 1
                    iter_max_used = max(1, min(iter_max_used, max(1, int(max_iter))))

                    iter_records = []
                    if len(fit_iter):
                        try:
                            _uniq, _cnt = np.unique(fit_iter, return_counts=True)
                            _iter_counts = ", ".join([f"i{int(i)}={int(c)}" for i, c in zip(_uniq, _cnt)])
                            self._log(f"  [DIAG] iter source counts: {_iter_counts}")
                        except Exception:
                            pass

                    def _render_model_subset(xy_sub: np.ndarray, flux_sub: np.ndarray) -> np.ndarray:
                        if len(xy_sub) == 0 or len(flux_sub) == 0:
                            return np.zeros_like(img_sub, dtype=np.float32)
                        try:
                            from photutils.datasets import make_model_image as _make_model_image
                            xy_sub = np.asarray(xy_sub, dtype=float)
                            flux_sub = np.asarray(flux_sub, dtype=float)
                            valid = (
                                np.isfinite(xy_sub[:, 0]) &
                                np.isfinite(xy_sub[:, 1]) &
                                np.isfinite(flux_sub) &
                                (flux_sub > 0)
                            )
                            if not np.any(valid):
                                return np.zeros_like(img_sub, dtype=np.float32)
                            xy_sub = xy_sub[valid]
                            flux_sub = flux_sub[valid]

                            # Use a wider rendering footprint than fit window.
                            # fit_shape covers only the PSF core; residual subtraction
                            # needs the wings too.  Use 2× epsf_size so the rendered
                            # stamp captures flux out to ~4×FWHM from each source.
                            pt = AstropyTable()
                            pt["x_0"] = np.asarray(xy_sub[:, 0], dtype=float)
                            pt["y_0"] = np.asarray(xy_sub[:, 1], dtype=float)
                            pt["flux"] = np.asarray(flux_sub, dtype=float)
                            mod = _clone_psf_model(epsf_model)
                            out = _make_model_image(
                                img_sub.shape,
                                mod,
                                pt,
                                model_shape=(int(render_shape_frame), int(render_shape_frame)),
                                x_name="x_0",
                                y_name="y_0",
                            )
                            return np.asarray(out, dtype=np.float32)
                        except Exception as _re:
                            self._log(f"  [DIAG] iter model render failed: {_re}")
                            # Never return full final model here; this helper is for
                            # subset-by-iteration rendering used by diagnostics.
                            return np.zeros_like(img_sub, dtype=np.float32)

                    for it_no in range(1, iter_max_used + 1):
                        m_le = fit_iter <= it_no if len(fit_iter) else np.zeros((0,), dtype=bool)
                        m_eq = fit_iter == it_no if len(fit_iter) else np.zeros((0,), dtype=bool)
                        fit_xy_i = fit_xy[m_le] if len(fit_xy) else np.zeros((0, 2), dtype=float)
                        fit_flux_i = fit_flux[m_le] if len(fit_flux) else np.zeros((0,), dtype=float)
                        if it_no <= 1:
                            applied_xy_i = init_xy_ui
                            det_xy_i = np.zeros((0, 2), dtype=float)
                        else:
                            applied_xy_i = fit_xy[fit_iter < it_no] if len(fit_iter) else np.zeros((0, 2), dtype=float)
                            det_xy_i = fit_xy[m_eq] if len(fit_xy) else np.zeros((0, 2), dtype=float)

                        if len(applied_xy_i) and len(det_xy_i):
                            box_xy_i = np.vstack([applied_xy_i, det_xy_i])
                        elif len(applied_xy_i):
                            box_xy_i = applied_xy_i
                        elif len(det_xy_i):
                            box_xy_i = det_xy_i
                        else:
                            box_xy_i = fit_xy_i

                        # Both residual_i and starsub_i are derived from the EXACT
                        # photometry.make_model_image() result where available.
                        #
                        # Last iter  → use exact model_img directly.
                        # Earlier iter → exact_full_model minus later-iter contributions
                        #   (add back later-iter _render_model_subset so only iter1..N remain).
                        # Fallback (model_img is None) → _render_model_subset only.
                        model_i = _render_model_subset(fit_xy_i, fit_flux_i)  # fallback only
                        if model_img is not None:
                            if it_no == iter_max_used:
                                # Exact: photometry.make_model_image() covers all fitted sources
                                residual_i = np.asarray(residual, dtype=np.float32)
                                starsub_i = np.asarray(starsub_raw, dtype=np.float32)
                                res_std_i = float(res_std)
                            else:
                                later_mask = (
                                    (fit_iter > it_no) if len(fit_iter) > 0
                                    else np.zeros(0, dtype=bool)
                                )
                                if np.any(later_mask):
                                    later_contrib = _render_model_subset(
                                        fit_xy[later_mask], fit_flux[later_mask]
                                    )
                                    residual_i = np.asarray(
                                        residual + later_contrib, dtype=np.float32
                                    )
                                    starsub_i = np.asarray(
                                        starsub_raw + later_contrib, dtype=np.float32
                                    )
                                else:
                                    residual_i = np.asarray(residual, dtype=np.float32)
                                    starsub_i = np.asarray(starsub_raw, dtype=np.float32)
                                _, _, res_std_i = sigma_clipped_stats(
                                    residual_i, sigma=3.0, maxiters=3
                                )
                        else:
                            residual_i = (img_sub - model_i).astype(np.float32, copy=False)
                            starsub_i = (img - model_i).astype(np.float32, copy=False)
                            _, _, res_std_i = sigma_clipped_stats(
                                residual_i, sigma=3.0, maxiters=3
                            )

                        hdr_it = fits.Header()
                        hdr_it["FILTER"] = this_filter
                        hdr_it["BKGMED"] = float(bkg_med)
                        hdr_it["ITER"] = int(it_no)
                        residual_iter_path = output_dir / f"residual_iter{it_no}_{fname}"
                        starsub_iter_path = output_dir / f"starsub_iter{it_no}_{fname}"
                        fitxy_iter_path = output_dir / f"fitxy_iter{it_no}_{fname}.npy"
                        modelxy_iter_path = output_dir / f"modelxy_iter{it_no}_{fname}.npy"
                        appliedxy_iter_path = output_dir / f"appliedxy_iter{it_no}_{fname}.npy"
                        detxy_iter_path = output_dir / f"detxy_iter{it_no}_{fname}.npy"
                        boxxy_iter_path = output_dir / f"boxxy_iter{it_no}_{fname}.npy"
                        fits.writeto(str(residual_iter_path), residual_i, hdr_it, overwrite=True)
                        fits.writeto(str(starsub_iter_path), starsub_i, hdr_it, overwrite=True)
                        np.save(str(fitxy_iter_path), np.asarray(fit_xy_i, dtype=np.float32))
                        # modelxy: sources first detected at THIS iter only (not cumulative).
                        # iter1 → initial seeds (1023), iter2 → new residual detections (70).
                        fit_xy_this = fit_xy[m_eq] if len(fit_xy) else np.zeros((0, 2), dtype=float)
                        fit_flux_this = fit_flux[m_eq] if len(fit_flux) else np.zeros((0,), dtype=float)
                        if it_no == 1:
                            # iter1: "new" mask is empty; use all iter1 sources
                            fit_xy_this = fit_xy[fit_iter == 1] if len(fit_iter) else np.zeros((0, 2), dtype=float)
                            fit_flux_this = fit_flux[fit_iter == 1] if len(fit_iter) else np.zeros((0,), dtype=float)
                        _m_model = (
                            np.isfinite(fit_xy_this[:, 0]) &
                            np.isfinite(fit_xy_this[:, 1]) &
                            np.isfinite(fit_flux_this) &
                            (fit_flux_this > 0)
                        ) if len(fit_xy_this) else np.zeros((0,), dtype=bool)
                        model_xy_i = fit_xy_this[_m_model] if len(fit_xy_this) else np.zeros((0, 2), dtype=float)
                        np.save(str(modelxy_iter_path), np.asarray(model_xy_i, dtype=np.float32))
                        np.save(str(appliedxy_iter_path), np.asarray(applied_xy_i, dtype=np.float32))
                        np.save(str(detxy_iter_path), np.asarray(det_xy_i, dtype=np.float32))
                        np.save(str(boxxy_iter_path), np.asarray(box_xy_i, dtype=np.float32))

                        n_new_i_kept = int(np.sum(m_eq)) if it_no > 1 else 0
                        n_new_i_raw = int(raw_iter_counts.get(int(it_no), n_new_i_kept)) if it_no > 1 else 0
                        iter_records.append({
                            "iter": int(it_no),
                            "fit_shape_px": _to_int(fit_shape_frame, 9),
                            "epsf_size_px": _to_int(epsf_size_frame, 25),
                            "n_fit": int(len(fit_xy_i)),
                            "residual_std": float(res_std_i),
                            "n_new_raw": int(n_new_i_raw),
                            "n_new_kept": int(n_new_i_kept),
                            "n_applied_prev": int(len(applied_xy_i)),
                            "residual_path": residual_iter_path.name,
                            "starsub_path": starsub_iter_path.name,
                            "fitxy_path": fitxy_iter_path.name,
                            "modelxy_path": modelxy_iter_path.name,
                            "detxy_path": detxy_iter_path.name,
                            "appliedxy_path": appliedxy_iter_path.name,
                            "boxxy_path": boxxy_iter_path.name,
                        })
                    self._log(
                        f"  fit done | n_fit={n_fit} | n_new={n_new_total} | "
                        f"residual_std={res_std:.4f}"
                    )

                    if (phot_result is None) or (len(phot_result) == 0):
                        reason = fit_fail_reason or "no fitted sources"
                        if n_init_drop > 0:
                            reason = f"{reason} | dropped_edge_init={n_init_drop}"
                        self.worker_status.emit(wid, fname, "Fit failed", 100)
                        return {
                            "file": fname,
                            "status": "fit_failed",
                            "reason": reason,
                        }

                    phot_rows = []
                    if phot_result is not None:
                        x_fit = np.array(phot_result["x_fit"])
                        y_fit = np.array(phot_result["y_fit"])
                        flux_fit = np.array(phot_result["flux_fit"])
                        flux_err = (np.array(phot_result["flux_err"]) if "flux_err" in phot_result.colnames else np.full(len(x_fit), np.nan))
                        chi2 = (np.array(phot_result["qfit"]) if "qfit" in phot_result.colnames else np.full(len(x_fit), np.nan))
                        flags_col = np.zeros(len(x_fit), dtype=int)

                        valid_fit_xy = np.isfinite(x_fit) & np.isfinite(y_fit)
                        for k in np.where(valid_fit_xy)[0]:
                            xi = int(round(float(x_fit[k])))
                            yi = int(round(float(y_fit[k])))
                            if 0 <= xi < w and 0 <= yi < h and img[yi, xi] >= sat_adu:
                                flags_col[k] |= self.FLAG_SAT
                            edge_m = fit_shape_frame // 2 + 1
                            if xi < edge_m or xi >= w - edge_m or yi < edge_m or yi >= h - edge_m:
                                flags_col[k] |= self.FLAG_EDGE

                        if len(xy_det) and len(x_fit):
                            src_xy = np.column_stack([x_fit, y_fit])
                            tree_ref = cKDTree(xy_det)
                            matched_det_uids = np.full(len(x_fit), -1, dtype=int)
                            nn_dists = np.full(len(x_fit), np.inf, dtype=float)
                            valid_src = np.isfinite(src_xy[:, 0]) & np.isfinite(src_xy[:, 1])
                            if np.any(valid_src):
                                q_dists, q_idx = tree_ref.query(src_xy[valid_src], k=1, workers=1)
                                matched_det_uids[valid_src] = det_uids[q_idx]
                                nn_dists[valid_src] = q_dists
                            match_tol = 2.0 * fwhm_med
                        else:
                            matched_det_uids = np.arange(len(x_fit), dtype=int)
                            nn_dists = np.zeros(len(x_fit))
                            match_tol = np.inf

                        _psf_only_uid = -1  # counts down for sources with no step4 match
                        _used_det_uids = set()
                        _uid_collision = 0
                        for k in range(len(x_fit)):
                            xk = float(x_fit[k]) if np.isfinite(x_fit[k]) else np.nan
                            yk = float(y_fit[k]) if np.isfinite(y_fit[k]) else np.nan
                            if not (np.isfinite(xk) and np.isfinite(yk)):
                                continue
                            fe = float(flux_fit[k]) * GAIN  # ADU → electrons (same as step5)
                            se = float(flux_err[k]) * GAIN if np.isfinite(flux_err[k]) else np.nan
                            snr = fe / se if (np.isfinite(se) and se > 0) else np.nan
                            if np.isfinite(snr) and snr >= min_snr and fe > 0:
                                mag_psf = ZP - 2.5 * np.log10(max(fe, 1e-30) / exptime)
                                mag_psf_err = (2.5 / np.log(10) * se / fe if (np.isfinite(se) and fe > 0) else np.nan)
                            else:
                                mag_psf = np.nan
                                mag_psf_err = np.nan
                            # Assign unique negative UIDs for PSF-only new detections
                            # (no matching step4 source within match_tol).
                            # Negative UIDs are excluded by downstream steps that join
                            # on step4 det_uid; they are preserved for traceability.
                            if nn_dists[k] <= match_tol:
                                cand_uid = int(matched_det_uids[k])
                                # Keep det_uid unique per frame: when two PSF components
                                # map to the same Step4 seed, keep first as seed UID and
                                # force others to PSF-only negative UIDs.
                                if cand_uid not in _used_det_uids:
                                    det_uid = cand_uid
                                    _used_det_uids.add(cand_uid)
                                else:
                                    _uid_collision += 1
                                    det_uid = _psf_only_uid
                                    _psf_only_uid -= 1
                            else:
                                det_uid = _psf_only_uid
                                _psf_only_uid -= 1
                                cand_uid = -1
                            if "iter_detected" in phot_result.colnames:
                                iter_val = _safe_float(phot_result["iter_detected"][k], np.nan)
                                iter_found = int(iter_val) if np.isfinite(iter_val) and iter_val > 0 else 1
                            else:
                                iter_found = 1
                            phot_rows.append({
                                "det_uid": det_uid,
                                "seed_uid": int(cand_uid) if np.isfinite(cand_uid) else -1,
                                "x_fit": round(xk, 4),
                                "y_fit": round(yk, 4),
                                "FILTER": this_filter,
                                "flux_psf_e": round(fe, 4) if np.isfinite(fe) else np.nan,
                                "flux_psf_err_e": round(float(se), 4) if np.isfinite(se) else np.nan,
                                "mag_psf": round(mag_psf, 6) if np.isfinite(mag_psf) else np.nan,
                                "mag_psf_err": round(mag_psf_err, 6) if np.isfinite(mag_psf_err) else np.nan,
                                "snr_psf": round(float(snr), 3) if np.isfinite(snr) else np.nan,
                                "qfit": round(float(chi2[k]), 4) if np.isfinite(chi2[k]) else np.nan,
                                "iter_found": iter_found,
                                "flags_psf": int(flags_col[k]),
                                "exptime": round(exptime, 4),
                            })
                        if _uid_collision > 0:
                            self._log(
                                f"  [UID] det_uid collision resolved: {_uid_collision} "
                                f"(assigned PSF-only negative det_uid)"
                            )

                    df_out = pd.DataFrame(phot_rows)

                    # ── Flux unit sanity check (P1-2) ────────────────────────
                    # PSF fitting runs on img_sub (ADU); flux_fit is in ADU.
                    # flux_psf_e = flux_fit * GAIN.  If the ratio PSF/aperture
                    # deviates far from 1.0 across bright sources, GAIN or the
                    # aperture data may be in the wrong unit.
                    if flux_init_map and len(df_out) > 5:
                        try:
                            _psf_e = pd.to_numeric(df_out["flux_psf_e"], errors="coerce")
                            _det_uid_col = pd.to_numeric(df_out["det_uid"], errors="coerce")
                            _ap_e_vals = np.array([
                                flux_init_map.get(int(u), np.nan) * GAIN
                                for u in _det_uid_col
                            ], dtype=float)
                            _ratio = _psf_e.to_numpy(float) / _ap_e_vals
                            _ratio_ok = _ratio[np.isfinite(_ratio) & (_ratio > 0)]
                            if len(_ratio_ok) >= 5:
                                med_ratio = float(np.median(_ratio_ok))
                                if not (0.3 < med_ratio < 3.0):
                                    self._log(
                                        f"  [WARN] flux unit mismatch? "
                                        f"median(psf_e/ap_e)={med_ratio:.3f} "
                                        f"(expected ~1.0). Check GAIN setting."
                                    )
                                else:
                                    self._log(
                                        f"  [UNIT] flux sanity OK: "
                                        f"median(psf_e/ap_e)={med_ratio:.3f} n={len(_ratio_ok)}"
                                    )
                        except Exception:
                            pass
                    # ─────────────────────────────────────────────────────────

                    out_tsv = output_dir / f"photometry_{fname}.tsv"
                    df_out.to_csv(out_tsv, sep="\t", index=False, encoding="utf-8-sig")
                    # Save step4 seed positions so the UI can tag iter2+ detections
                    # as "신규검출 (step4 미검출)" vs "재검출 (step4 기검출)".
                    seed_xy_path = output_dir / f"seed_xy_{fname}.npy"
                    np.save(str(seed_xy_path), init_xy_ui.astype(np.float32))

                    residual_meta = {
                        "file": fname,
                        "filter": this_filter,
                        "bkg_med": float(bkg_med),
                        "n_new_raw": int(n_new_raw_total),
                        "rawxy_iter2_path": f"rawxy_iter2_{fname}.npy",
                        "seedxy_path": seed_xy_path.name,
                        "iters": iter_records,
                    }
                    self.worker_status.emit(wid, fname, "Save", 95)
                    # Keep final products and metadata for UI reload/QA.
                    res_path = output_dir / f"residual_{fname}"
                    starsub_path = output_dir / f"starsub_{fname}"
                    hdr_res = fits.Header()
                    hdr_res["FILTER"] = this_filter
                    hdr_res["BKGMED"] = float(bkg_med)
                    fits.writeto(str(res_path), residual.astype(np.float32), hdr_res, overwrite=True)
                    fits.writeto(str(starsub_path), (residual + float(bkg_med)).astype(np.float32), hdr_res, overwrite=True)
                    meta_path = output_dir / f"residual_meta_{fname}.json"
                    meta_path.write_text(json.dumps(residual_meta, ensure_ascii=False, indent=2), encoding="utf-8")
                    rawxy_iter2_path = output_dir / f"rawxy_iter2_{fname}.npy"
                    np.save(str(rawxy_iter2_path), np.asarray(raw_new_xy, dtype=np.float32))

                    merged_new_xy = None
                    if phot_result is not None and len(phot_result) > 0:
                        try:
                            if "iter_detected" in phot_result.colnames:
                                x_all = np.asarray(phot_result["x_fit"], dtype=float)
                                y_all = np.asarray(phot_result["y_fit"], dtype=float)
                                it_raw = np.asarray(phot_result["iter_detected"], dtype=float)
                                it_all = np.where(np.isfinite(it_raw), it_raw, 1.0).astype(int)
                                m_new = np.isfinite(x_all) & np.isfinite(y_all) & (it_all > 1)
                                if np.any(m_new):
                                    merged_new_xy = np.column_stack([x_all[m_new], y_all[m_new]])
                        except Exception:
                            merged_new_xy = None
                    n_rows = len(phot_rows)
                    n_good = int(df_out["mag_psf"].notna().sum()) if not df_out.empty else 0
                    idx_row = {
                        "file": fname,
                        "filter": this_filter,
                        "n": n_rows,
                        "n_goodmag": n_good,
                        "n_fail": n_rows - n_good,
                        "n_new_iter": int(n_new_total),
                    }
                    self.worker_status.emit(wid, fname, "Done", 100)
                    return {
                        "file": fname,
                        "status": "processed",
                        "idx_row": idx_row,
                        "epsf_key": f"{fname} | {this_filter}" if epsf_emit_arr is not None else None,
                        "epsf_frame": fname if epsf_emit_arr is not None else None,
                        "epsf_arr": epsf_emit_arr,
                        "residual_meta": residual_meta,
                        "new_xy": merged_new_xy,
                    }
                except Exception as frame_e:
                    self.worker_status.emit(wid, fname, "Error", 100)
                    return {"file": fname, "status": "error", "reason": f"{frame_e}\n{traceback.format_exc()}"}

            ex = ThreadPoolExecutor(max_workers=max_workers)
            self._executor = ex
            future_map: dict = {}
            next_idx = 0

            def _submit_next():
                nonlocal next_idx
                if next_idx >= total:
                    return False
                fname_n = frames[next_idx]
                future_map[ex.submit(process_single_frame, fname_n)] = fname_n
                next_idx += 1
                return True

            for _ in range(min(max_workers, total)):
                _submit_next()

            try:
                while future_map:
                    # Stop mode: cancel queued (not-started) futures and do not submit new ones.
                    if self._stop_requested:
                        n_cancel = 0
                        for fut, fname_c in list(future_map.items()):
                            if fut.cancel():
                                del future_map[fut]
                                completed[0] += 1
                                counters["stopped"] += 1
                                n_cancel += 1
                                self.progress.emit(completed[0], total, fname_c)
                                self._log(f"[{completed[0]}/{total}] STOP {fname_c} | cancelled")
                        if n_cancel > 0:
                            self._log(f"Stop requested | cancelled pending={n_cancel}")
                        if not future_map:
                            break

                    done, _ = wait(tuple(future_map.keys()), timeout=0.2, return_when=FIRST_COMPLETED)
                    now = time.time()
                    n_done = int(completed[0])
                    n_running = int(len(future_map))
                    n_queued = int(max(0, total - next_idx))
                    progress_changed = (n_done != last_done_count)
                    if progress_changed:
                        last_done_count = n_done

                    if (now - last_hb) >= 8.0:
                        eta_txt = "--:--"
                        if n_done > 0:
                            elapsed = max(0.0, now - run_t0)
                            eta_txt = _fmt_eta((elapsed / n_done) * max(0, total - n_done))

                        if progress_changed:
                            self._log(
                                f"[PROG] done={n_done}/{total} | running={n_running} | queued={n_queued} | ETA~{eta_txt}"
                            )
                            self.progress.emit(n_done, total, f"RUN={n_running} QUEUE={n_queued} ETA~{eta_txt}")
                            last_hb = now
                            last_stall_log = now
                        else:
                            # Long fit phases can run for minutes; avoid spamming identical lines.
                            if (now - last_stall_log) >= 30.0:
                                active_names = list(future_map.values())
                                active_txt = ", ".join(active_names[:3]) if active_names else "-"
                                self._log(
                                    f"[PROG] waiting | done={n_done}/{total} | running={n_running} | "
                                    f"queued={n_queued} | active={active_txt} | ETA~{eta_txt}"
                                )
                                self.progress.emit(n_done, total, f"RUN={n_running} QUEUE={n_queued} ETA~{eta_txt}")
                                last_stall_log = now
                                last_hb = now
                    if not done:
                        continue

                    for fut in done:
                        fname = future_map.pop(fut, None)
                        if fname is None:
                            continue

                        if fut.cancelled():
                            completed[0] += 1
                            counters["stopped"] += 1
                            self.progress.emit(completed[0], total, fname)
                            self._log(f"[{completed[0]}/{total}] STOP {fname} | cancelled")
                            continue

                        try:
                            out = fut.result()
                        except Exception as e:
                            out = {"file": fname, "status": "error", "reason": str(e)}

                        completed[0] += 1
                        self.progress.emit(completed[0], total, out.get("file", fname))
                        status = out.get("status", "error")

                        if status == "processed":
                            idx_row = out.get("idx_row", {})
                            if idx_row:
                                index_rows.append(idx_row)
                                self.frame_done.emit(out["file"], idx_row)
                            if out.get("epsf_key") and out.get("epsf_arr") is not None:
                                self.epsf_ready.emit(out["epsf_key"], out.get("epsf_frame", out["file"]), out["epsf_arr"])
                            self.residual_ready.emit(out["file"], out.get("residual_meta"), out.get("new_xy"))
                            counters["processed"] += 1
                            self._log(
                                f"[{completed[0]}/{total}] OK {out['file']} | "
                                f"f={idx_row.get('filter', '?')} n={idx_row.get('n', 0)} "
                                f"good={idx_row.get('n_goodmag', 0)} new_iter={idx_row.get('n_new_iter', 0)}"
                            )
                        elif status == "no_detect":
                            counters["no_detect"] += 1
                            self._log(f"[{completed[0]}/{total}] SKIP {out['file']} | reason={out.get('reason', status)}")
                        elif status == "no_fits":
                            counters["no_fits"] += 1
                            self._log(f"[{completed[0]}/{total}] SKIP {out['file']} | reason={out.get('reason', status)}")
                        elif status == "stopped":
                            counters["stopped"] += 1
                            self._log(f"[{completed[0]}/{total}] STOP {out['file']} | reason={out.get('reason', status)}")
                        elif status == "fit_failed":
                            self._log(f"[{completed[0]}/{total}] FAIL {out['file']} | reason={out.get('reason', status)}")
                        elif status == "no_valid_init":
                            self._log(f"[{completed[0]}/{total}] SKIP {out['file']} | reason={out.get('reason', status)}")
                        else:
                            self._log(f"[{completed[0]}/{total}] ERROR {out['file']} | {out.get('reason', 'unknown')}")

                        # Keep pipeline fed only while not stopping.
                        if not self._stop_requested:
                            _submit_next()
            finally:
                remaining_unscheduled = max(0, total - next_idx)
                if self._stop_requested and remaining_unscheduled > 0:
                    counters["stopped"] += remaining_unscheduled
                    self._log(f"Stop requested | not_submitted={remaining_unscheduled}")
                try:
                    ex.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    pass
                self._executor = None

            if index_rows:
                pd.DataFrame(index_rows).to_csv(output_dir / "photometry_index.csv", index=False)

            self._log(
                f"Done | processed={counters['processed']} | "
                f"no_detect={counters['no_detect']} | no_fits={counters['no_fits']} | "
                f"stopped={counters['stopped']}"
            )
            self.finished.emit({"frames": total, **counters})

        except Exception as e:
            self.error.emit("PSF_WORKER", f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
            self.finished.emit({})


# ── PSF Photometry Window ─────────────────────────────────────────────────────

class PSFPhotometryWindow(StepWindowBase):
    """Step 6 – PSF Photometry (skippable).

    If skipped, Step 8 RefBuild falls back to Step 5 aperture results.
    """

    def __init__(self, params, file_manager, project_state, main_window):
        self.file_manager = file_manager
        self.worker = None
        self.file_list = []
        self.use_cropped = False
        self.log_window = None
        self._skip_psf = False
        # In-memory cache (lost on window close → reloaded from disk in restore_state)
        self._last_epsf: dict[str, np.ndarray] = {}          # display_key → epsf array
        self._residual_meta: dict[str, dict] = {}            # fname  → residual metadata + iter records
        self._last_new_xy: dict[str, np.ndarray | None] = {} # fname  → new-detect XY or None
        self._cutout_idx: int = 0  # current star index in cutout viewer
        self._run_started_ts: float | None = None
        self._log_worker_bars: dict[int, tuple] = {}   # worker_id → (QLabel, QProgressBar)
        self._log_worker_frame: dict[int, str] = {}    # worker_id → current frame name

        super().__init__(
            step_index=5,
            step_name="PSF Photometry",
            params=params,
            project_state=project_state,
            main_window=main_window,
        )
        self.setup_step_ui()
        self.restore_state()

    def setup_step_ui(self):
        info = QLabel(
            "Optional PSF photometry using photutils EPSFBuilder.\n"
            "Click Skip PSF to continue to Step 7 (WCS); Step 8 (RefBuild) will use aperture results."
        )
        info.setWordWrap(True)
        info.setStyleSheet("QLabel { background-color: #E8F5E9; padding: 8px; margin-bottom: 6px; }")
        self.content_layout.addWidget(info)

        # ── Controls ──────────────────────────────────────────────────────────
        ctrl = QHBoxLayout()
        self.btn_params = QPushButton("Parameters")
        self.btn_params.clicked.connect(self.open_parameters_dialog)
        ctrl.addWidget(self.btn_params)

        self.btn_skip = QPushButton("Skip PSF →")
        self.btn_skip.setStyleSheet(
            "QPushButton { background-color: #FF7043; color: white; font-weight: bold; padding: 8px 20px; }"
        )
        self.btn_skip.setToolTip("Skip PSF photometry; Step 8 RefBuild will use aperture results from Step 5.")
        self.btn_skip.clicked.connect(self.skip_psf)
        ctrl.addWidget(self.btn_skip)

        ctrl.addStretch()

        self.btn_run = QPushButton("Run PSF")
        self.btn_run.setStyleSheet(
            "QPushButton { background-color: #388E3C; color: white; font-weight: bold; padding: 8px 24px; }"
        )
        self.btn_run.clicked.connect(self.run_psf)
        ctrl.addWidget(self.btn_run)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_psf)
        ctrl.addWidget(self.btn_stop)

        self.btn_log = QPushButton("Log")
        self.btn_log.clicked.connect(self.show_log_window)
        ctrl.addWidget(self.btn_log)

        self.content_layout.addLayout(ctrl)

        # ── Progress ──────────────────────────────────────────────────────────
        prog = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        prog.addWidget(self.progress_bar, 1)
        self.progress_label = QLabel("Ready")
        self.progress_label.setMinimumWidth(420)
        prog.addWidget(self.progress_label)
        self.content_layout.addLayout(prog)


        # ── Skip status label ─────────────────────────────────────────────────
        self.skip_label = QLabel("")
        self.skip_label.setStyleSheet("QLabel { color: #FF7043; font-weight: bold; padding: 4px; }")
        self.content_layout.addWidget(self.skip_label)

        # ── Diagnostic tabs ───────────────────────────────────────────────────
        self.main_tabs = QTabWidget()
        self.content_layout.addWidget(self.main_tabs)

        # Tab 0: EPSF Model
        epsf_tab = QWidget()
        epsf_layout = QVBoxLayout(epsf_tab)
        epsf_top = QHBoxLayout()
        epsf_top.addWidget(QLabel("Model:"))
        self.epsf_filter_combo = QComboBox()
        self.epsf_filter_combo.currentTextChanged.connect(self._on_epsf_filter_changed)
        epsf_top.addWidget(self.epsf_filter_combo)
        epsf_top.addStretch()
        epsf_layout.addLayout(epsf_top)

        self.epsf_fig = Figure(figsize=(8, 4))
        self.epsf_canvas = FigureCanvas(self.epsf_fig)
        self.epsf_toolbar = NavigationToolbar(self.epsf_canvas, self)
        epsf_layout.addWidget(self.epsf_toolbar)
        epsf_layout.addWidget(self.epsf_canvas)
        self.main_tabs.addTab(epsf_tab, "EPSF Model")

        # Tab 1: Cutout viewer – Raw | Star-subtracted per star, per iter
        res_tab = QWidget()
        res_layout = QVBoxLayout(res_tab)

        # Row 1: frame / iter / mode selectors
        res_top = QHBoxLayout()
        res_top.addWidget(QLabel("Frame:"))
        self.res_file_combo = QComboBox()
        self.res_file_combo.currentTextChanged.connect(self._on_residual_frame_changed)
        res_top.addWidget(self.res_file_combo)
        res_top.addWidget(QLabel("Iter:"))
        self.res_iter_combo = QComboBox()
        self.res_iter_combo.currentTextChanged.connect(self._on_residual_iter_changed)
        res_top.addWidget(self.res_iter_combo)
        res_top.addStretch()
        res_layout.addLayout(res_top)

        # Row 2: star navigation (◀ / label / ▶)
        res_nav = QHBoxLayout()
        self.res_prev_btn = QPushButton("◀")
        self.res_prev_btn.setFixedWidth(36)
        self.res_prev_btn.clicked.connect(self._on_cutout_prev)
        res_nav.addWidget(self.res_prev_btn)
        self.res_star_label = QLabel("—")
        self.res_star_label.setMinimumWidth(70)
        self.res_star_label.setAlignment(Qt.AlignCenter)
        res_nav.addWidget(self.res_star_label)
        self.res_next_btn = QPushButton("▶")
        self.res_next_btn.setFixedWidth(36)
        self.res_next_btn.clicked.connect(self._on_cutout_next)
        res_nav.addWidget(self.res_next_btn)
        res_nav.addStretch()
        self.res_info_label = QLabel("")
        res_nav.addWidget(self.res_info_label)
        res_layout.addLayout(res_nav)

        self.res_fig = Figure(figsize=(8, 4))
        self.res_canvas = FigureCanvas(self.res_fig)
        self.res_toolbar = NavigationToolbar(self.res_canvas, self)
        res_layout.addWidget(self.res_toolbar)
        res_layout.addWidget(self.res_canvas)
        self.main_tabs.addTab(res_tab, "Residuals")

        # Tab 2: Photometry Table
        phot_tab = QWidget()
        phot_layout = QVBoxLayout(phot_tab)
        self.frame_table = QTableWidget()
        self.frame_table.setColumnCount(6)
        self.frame_table.setHorizontalHeaderLabels(
            ["Frame", "Filter", "N_psf", "N_goodmag", "N_fail", "N_new_iter"]
        )
        self.frame_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for c in range(1, 6):
            self.frame_table.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeToContents)
        self.frame_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        phot_layout.addWidget(self.frame_table)
        self.main_tabs.addTab(phot_tab, "Photometry")

        # Tab 3: QC Report (PSF statistics + Ap vs PSF comparison)
        qc_tab = QWidget()
        qc_outer = QVBoxLayout(qc_tab)

        qc_top_bar = QHBoxLayout()
        qc_refresh_btn = QPushButton("Refresh QC")
        qc_refresh_btn.clicked.connect(self._refresh_qc)
        qc_top_bar.addWidget(qc_refresh_btn)
        qc_top_bar.addStretch()
        qc_outer.addLayout(qc_top_bar)

        qc_splitter = QSplitter(Qt.Vertical)
        qc_outer.addWidget(qc_splitter, 1)

        # Top: PSF statistics summary text
        self.qc_text = QTextEdit()
        self.qc_text.setReadOnly(True)
        self.qc_text.setStyleSheet("QTextEdit { font-family: monospace; font-size: 9pt; }")
        self.qc_text.setMaximumHeight(300)
        qc_splitter.addWidget(self.qc_text)

        # Bottom: Ap vs PSF matplotlib plot
        cmp_widget = QWidget()
        cmp_layout = QVBoxLayout(cmp_widget)
        cmp_top = QHBoxLayout()
        cmp_refresh_btn = QPushButton("Refresh Plot")
        cmp_refresh_btn.clicked.connect(self._plot_mag_comparison)
        cmp_top.addWidget(cmp_refresh_btn)
        cmp_top.addWidget(QLabel("Filter:"))
        self.cmp_filter_combo = QComboBox()
        self.cmp_filter_combo.addItem("all")
        self.cmp_filter_combo.currentTextChanged.connect(self._plot_mag_comparison)
        cmp_top.addWidget(self.cmp_filter_combo)
        cmp_top.addWidget(QLabel("Frame:"))
        self.cmp_frame_combo = QComboBox()
        self.cmp_frame_combo.addItem("all")
        self.cmp_frame_combo.currentTextChanged.connect(self._plot_mag_comparison)
        cmp_top.addWidget(self.cmp_frame_combo)
        self.cmp_flags0_only = QCheckBox("flags=0 only")
        self.cmp_flags0_only.setChecked(False)
        self.cmp_flags0_only.toggled.connect(self._plot_mag_comparison)
        cmp_top.addWidget(self.cmp_flags0_only)
        cmp_top.addWidget(QLabel("SNR ≥"))
        self.cmp_snr_min = QDoubleSpinBox()
        self.cmp_snr_min.setRange(0.0, 200.0)
        self.cmp_snr_min.setSingleStep(1.0)
        self.cmp_snr_min.setValue(0.0)
        self.cmp_snr_min.setDecimals(1)
        self.cmp_snr_min.setToolTip("0 = off")
        self.cmp_snr_min.valueChanged.connect(self._plot_mag_comparison)
        cmp_top.addWidget(self.cmp_snr_min)
        cmp_top.addWidget(QLabel("qfit ≤"))
        self.cmp_qfit_max = QDoubleSpinBox()
        self.cmp_qfit_max.setRange(0.0, 10.0)
        self.cmp_qfit_max.setSingleStep(0.05)
        self.cmp_qfit_max.setValue(0.0)
        self.cmp_qfit_max.setDecimals(3)
        self.cmp_qfit_max.setToolTip("0 = off")
        self.cmp_qfit_max.valueChanged.connect(self._plot_mag_comparison)
        cmp_top.addWidget(self.cmp_qfit_max)
        cmp_top.addWidget(QLabel("|Δmag| ≤"))
        self.cmp_dmag_clip = QDoubleSpinBox()
        self.cmp_dmag_clip.setRange(0.0, 5.0)
        self.cmp_dmag_clip.setSingleStep(0.05)
        self.cmp_dmag_clip.setValue(0.0)
        self.cmp_dmag_clip.setDecimals(3)
        self.cmp_dmag_clip.setToolTip("0 = off")
        self.cmp_dmag_clip.valueChanged.connect(self._plot_mag_comparison)
        cmp_top.addWidget(self.cmp_dmag_clip)
        self.cmp_stats_label = QLabel("")
        self.cmp_stats_label.setWordWrap(True)
        cmp_top.addWidget(self.cmp_stats_label, 1)
        cmp_layout.addLayout(cmp_top)
        self.cmp_fig = Figure(figsize=(10, 4))
        self.cmp_canvas = FigureCanvas(self.cmp_fig)
        self.cmp_toolbar = NavigationToolbar(self.cmp_canvas, self)
        cmp_layout.addWidget(self.cmp_toolbar)
        cmp_layout.addWidget(self.cmp_canvas)
        qc_splitter.addWidget(cmp_widget)

        self.main_tabs.addTab(qc_tab, "QC")

        self.main_tabs.setCurrentIndex(0)

        # ── Log window ────────────────────────────────────────────────────────
        self.log_window = QWidget(self, Qt.Window)
        self.log_window.setWindowTitle("Step6 PSF Photometry Log")
        self.log_window.resize(900, 420)
        log_layout = QVBoxLayout(self.log_window)

        log_splitter = QSplitter(Qt.Horizontal)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("QTextEdit { font-family: monospace; font-size: 9pt; }")
        log_splitter.addWidget(self.log_text)

        _log_worker_group = QGroupBox("Workers")
        _log_worker_group_layout = QVBoxLayout(_log_worker_group)
        self._log_worker_layout = QVBoxLayout()
        _log_worker_group_layout.addLayout(self._log_worker_layout)
        _log_worker_group_layout.addStretch()
        log_splitter.addWidget(_log_worker_group)

        log_splitter.setStretchFactor(0, 2)
        log_splitter.setStretchFactor(1, 1)
        log_layout.addWidget(log_splitter)

        # Keyboard shortcuts: ← → navigate cutout stars
        from PyQt5.QtWidgets import QShortcut
        from PyQt5.QtGui import QKeySequence
        QShortcut(QKeySequence(Qt.Key_Left),  self).activated.connect(self._on_cutout_prev)
        QShortcut(QKeySequence(Qt.Key_Right), self).activated.connect(self._on_cutout_next)

        self.populate_file_list()
        self.update_frame_table()
        self._update_skip_label()

    # ── File list ─────────────────────────────────────────────────────────────

    def populate_file_list(self):
        crop_active = crop_is_active(self.params.P.result_dir)
        cropped_dir = step2_cropped_dir(self.params.P.result_dir)
        legacy_cropped = self.params.P.result_dir / "cropped"
        if crop_active and cropped_dir.exists() and list(cropped_dir.glob("*.fit*")):
            files = sorted([f.name for f in cropped_dir.glob("*.fit*")])
            self.use_cropped = True
        elif crop_active and legacy_cropped.exists() and list(legacy_cropped.glob("*.fit*")):
            files = sorted([f.name for f in legacy_cropped.glob("*.fit*")])
            self.use_cropped = True
        else:
            if not self.file_manager.filenames:
                try:
                    self.file_manager.scan_files()
                except Exception:
                    pass
            files = self.file_manager.filenames
            self.use_cropped = False
        files = list(files)

        # Hard gate: downstream should skip frames where apcorr was not applied.
        apcorr_sum = step5_aperture_dir(self.params.P.result_dir) / "apcorr_summary.csv"
        if not apcorr_sum.exists():
            legacy_sum = self.params.P.result_dir / "apcorr_summary.csv"
            if legacy_sum.exists():
                apcorr_sum = legacy_sum
        if apcorr_sum.exists():
            try:
                df_apc = pd.read_csv(apcorr_sum)
                if (not df_apc.empty) and {"file", "apply"} <= set(df_apc.columns):
                    ok_vals = df_apc["apply"].astype(str).str.strip().str.lower().isin(
                        {"true", "1", "yes", "y", "on"}
                    )
                    ok_set = set(df_apc.loc[ok_vals, "file"].astype(str).map(lambda s: Path(str(s)).name))
                    before_n = len(files)
                    files = [f for f in files if Path(str(f)).name in ok_set]
                    self.log(f"[APCORR] apply=True frame filter: {len(files)}/{before_n} kept")
            except Exception as e:
                self.log(f"[APCORR] frame filter skipped ({e})")

        self.file_list = list(files)

    # ── Actions ───────────────────────────────────────────────────────────────

    def skip_psf(self):
        self._skip_psf = True
        self.save_state()
        self._update_skip_label()
        self.update_navigation_buttons()
        self.log("PSF skipped — Step 8 RefBuild will use Step 5 aperture results.")

    def run_psf(self):
        if not self.file_list:
            QMessageBox.warning(self, "Warning", "No files to process")
            return
        if self.worker and self.worker.isRunning():
            return
        if not (step5_aperture_dir(self.params.P.result_dir) / "photometry_index.csv").exists():
            QMessageBox.warning(
                self, "Prerequisite",
                "Step 5 Aperture Photometry must be completed first."
            )
            return

        self._skip_psf = False
        self.log_text.clear()
        self.frame_table.setRowCount(0)
        self._last_epsf.clear()
        self._residual_meta.clear()
        self._last_new_xy.clear()
        self.epsf_filter_combo.clear()
        self.res_file_combo.clear()
        self.res_iter_combo.clear()
        # Keep frame selector populated during run so UI is not visually blank
        # before first residual metadata arrives.
        self.res_file_combo.addItems(self.file_list)
        if self.res_file_combo.count() > 0:
            self.res_file_combo.setCurrentIndex(0)

        # Remove stale residual/model artifacts for selected frames to avoid mixing old runs.
        out_dir = step6_psf_dir(self.params.P.result_dir)
        for fname in self.file_list:
            patterns = [
                f"residual_meta_{fname}.json",
                f"residual_{fname}",
                f"starsub_{fname}",
                f"residual_iter*_{fname}",
                f"starsub_iter*_{fname}",
                f"fitxy_iter*_{fname}.npy",
                f"detxy_iter*_{fname}.npy",
                f"appliedxy_iter*_{fname}.npy",
                f"boxxy_iter*_{fname}.npy",
            ]
            for pat in patterns:
                for p in out_dir.glob(pat):
                    try:
                        p.unlink()
                    except Exception:
                        pass

        # Clear log window worker bars from previous run
        self._log_worker_bars.clear()
        self._log_worker_frame.clear()
        while self._log_worker_layout.count():
            item = self._log_worker_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.log(f"Start PSF photometry | {len(self.file_list)} frames")
        self._run_started_ts = time.time()

        self.worker = Step6PSFWorker(
            self.file_list, self.params,
            self.params.P.data_dir, self.params.P.result_dir,
            self.params.P.cache_dir, self.use_cropped,
        )
        self.worker.progress.connect(self.on_progress)
        self.worker.worker_status.connect(self.on_worker_status)
        self.worker.frame_done.connect(self.on_frame_done)
        self.worker.epsf_ready.connect(self.on_epsf_ready)
        self.worker.residual_ready.connect(self.on_residual_ready)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.log.connect(self.log)

        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_skip.setEnabled(False)
        self.progress_bar.setMaximum(len(self.file_list))
        self.progress_bar.setValue(0)
        self.progress_label.setText(f"0/{len(self.file_list)} | ETA --:-- | Starting...")
        self.worker.start()
        self.show_log_window()

    def stop_psf(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.btn_stop.setEnabled(False)
            self.progress_label.setText("Stopping... (running frames will finish current fit)")

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def on_worker_status(self, worker_id: int, frame: str, stage: str, pct: int):
        # Update log window worker bars
        w_key = int(worker_id)
        self._log_worker_frame[w_key] = frame
        if w_key not in self._log_worker_bars:
            row_w = QWidget()
            row_h = QHBoxLayout(row_w)
            row_h.setContentsMargins(2, 1, 2, 1)
            row_h.setSpacing(4)
            wid_lbl = QLabel(f"W{w_key + 1}")
            wid_lbl.setFixedWidth(22)
            name_lbl = QLabel()
            stage_lbl = QLabel()
            stage_lbl.setFixedWidth(72)
            stage_lbl.setAlignment(Qt.AlignCenter)
            stage_lbl.setStyleSheet("QLabel { color: #888; font-size: 8pt; }")
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setFixedHeight(14)
            bar.setFixedWidth(80)
            bar.setTextVisible(True)
            row_h.addWidget(wid_lbl)
            row_h.addWidget(name_lbl, 1)   # stretch
            row_h.addWidget(stage_lbl)
            row_h.addWidget(bar)
            self._log_worker_layout.addWidget(row_w)
            self._log_worker_bars[w_key] = (name_lbl, stage_lbl, bar)
        name_lbl, stage_lbl, bar = self._log_worker_bars[w_key]
        name_lbl.setText(frame)
        stage_lbl.setText(f"[{stage}]")
        bar.setValue(int(max(0, min(100, int(pct)))))

    def on_progress(self, current, total, filename):
        self.progress_bar.setValue(current)
        eta_txt = "pending"
        if self._run_started_ts is not None and current > 0 and total > 0:
            elapsed = max(0.0, time.time() - float(self._run_started_ts))
            per_frame = elapsed / float(current)
            eta_sec = max(0.0, per_frame * float(max(0, total - current)))
            eta_txt = self._fmt_duration(eta_sec)
        self.progress_label.setText(f"{current}/{total} | ETA {eta_txt} | {filename}")

    def on_frame_done(self, filename, result):
        r = self.frame_table.rowCount()
        self.frame_table.insertRow(r)
        self.frame_table.setItem(r, 0, QTableWidgetItem(filename))
        self.frame_table.setItem(r, 1, QTableWidgetItem(str(result.get("filter", ""))))
        self.frame_table.setItem(r, 2, QTableWidgetItem(str(result.get("n", 0))))
        self.frame_table.setItem(r, 3, QTableWidgetItem(str(result.get("n_goodmag", 0))))
        self.frame_table.setItem(r, 4, QTableWidgetItem(str(result.get("n_fail", 0))))
        self.frame_table.setItem(r, 5, QTableWidgetItem(str(result.get("n_new_iter", 0))))
        self.frame_table.scrollToBottom()
        # Mark log window worker bar as done
        for w_key, fname in self._log_worker_frame.items():
            if fname == filename:
                if w_key in self._log_worker_bars:
                    name_lbl, stage_lbl, bar = self._log_worker_bars[w_key]
                    stage_lbl.setText("[Done]")
                    bar.setValue(100)
                break

    def on_epsf_ready(self, display_key: str, _frame_name: str, epsf_arr: np.ndarray):
        self._last_epsf[display_key] = epsf_arr
        current = self.epsf_filter_combo.currentText()
        self.epsf_filter_combo.blockSignals(True)
        self.epsf_filter_combo.clear()
        self.epsf_filter_combo.addItems(sorted(self._last_epsf.keys()))
        if current in self._last_epsf:
            self.epsf_filter_combo.setCurrentText(current)
        else:
            try:
                self.epsf_filter_combo.setCurrentText(display_key)
            except Exception:
                self.epsf_filter_combo.setCurrentIndex(0)
        self.epsf_filter_combo.blockSignals(False)
        self._plot_epsf(display_key)

    def on_residual_ready(self, fname: str, residual_meta: dict, new_xy):
        if isinstance(residual_meta, dict):
            self._residual_meta[fname] = residual_meta
        self._last_new_xy[fname] = new_xy  # ndarray or None
        current = self.res_file_combo.currentText()
        self.res_file_combo.blockSignals(True)
        self.res_file_combo.clear()
        self.res_file_combo.addItems(sorted(self._residual_meta.keys()))
        if current in self._residual_meta:
            self.res_file_combo.setCurrentText(current)
        else:
            self.res_file_combo.setCurrentIndex(self.res_file_combo.count() - 1)
        self.res_file_combo.blockSignals(False)
        self._cutout_idx = 0
        self._refresh_residual_iter_combo(fname)
        self._plot_cutout(fname)

    def on_error(self, src, err):
        self.log(f"ERROR {src}: {err}")

    def on_finished(self, summary):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_skip.setEnabled(True)
        elapsed_txt = ""
        if self._run_started_ts is not None:
            elapsed_txt = f" | elapsed {self._fmt_duration(time.time() - float(self._run_started_ts))}"
        self.progress_label.setText(f"Done{elapsed_txt}")
        self._run_started_ts = None
        self.log(f"PSF done: {summary}")
        self._cleanup_worker()
        self._update_skip_label()
        self.update_frame_table()  # refresh Photometry tab from disk
        self._refresh_qc()           # refresh QC tab (stats + Ap vs PSF plot)
        self.save_state()
        self.update_navigation_buttons()

    # ── EPSF plot ─────────────────────────────────────────────────────────────

    def _on_epsf_filter_changed(self, display_key: str):
        self._plot_epsf(display_key)

    @staticmethod
    def _fmt_duration(seconds: float) -> str:
        s = int(max(0, round(float(seconds))))
        h, rem = divmod(s, 3600)
        m, sec = divmod(rem, 60)
        if h > 0:
            return f"{h:d}:{m:02d}:{sec:02d}"
        return f"{m:02d}:{sec:02d}"

    def _plot_epsf(self, display_key: str):
        if display_key not in self._last_epsf:
            return
        epsf_arr = self._last_epsf[display_key]

        # Clear ENTIRE figure to prevent colorbar accumulation
        self.epsf_fig.clf()
        ax2d = self.epsf_fig.add_subplot(121)
        ax_rad = self.epsf_fig.add_subplot(122)

        vmax = np.nanpercentile(epsf_arr, 99)
        im = ax2d.imshow(epsf_arr, origin="lower", cmap="viridis",
                         norm=mcolors.PowerNorm(gamma=0.5, vmin=0, vmax=max(vmax, 1e-10)))
        self.epsf_fig.colorbar(im, ax=ax2d, fraction=0.046, pad=0.04)
        ax2d.set_title(f"ePSF — {display_key}", fontsize=9)
        ax2d.set_xlabel("px (oversampled)", fontsize=8)
        ax2d.set_ylabel("px (oversampled)", fontsize=8)

        # Radial profile
        cy, cx = np.array(epsf_arr.shape) / 2.0
        yy, xx = np.mgrid[0:epsf_arr.shape[0], 0:epsf_arr.shape[1]]
        rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        r_flat = rr.ravel()
        v_flat = epsf_arr.ravel()
        order = np.argsort(r_flat)
        ax_rad.plot(r_flat[order], v_flat[order], ".", markersize=1, alpha=0.3, color="#1565C0")
        ax_rad.set_xlabel("Radius (oversampled px)", fontsize=8)
        ax_rad.set_ylabel("ePSF value", fontsize=8)
        ax_rad.set_title("Radial profile", fontsize=9)
        ax_rad.grid(True, alpha=0.3)

        self.epsf_fig.tight_layout()
        self.epsf_canvas.draw_idle()

    # ── Residual plot ─────────────────────────────────────────────────────────

    def _on_residual_frame_changed(self, fname: str):
        self._cutout_idx = 0
        self._refresh_residual_iter_combo(fname)
        self._plot_cutout(fname)

    def _on_residual_iter_changed(self, _iter_label: str):
        self._cutout_idx = 0
        fname = self.res_file_combo.currentText().strip()
        if fname:
            self._plot_cutout(fname)

    def _on_cutout_prev(self):
        if self._cutout_idx > 0:
            self._cutout_idx -= 1
            fname = self.res_file_combo.currentText().strip()
            if fname:
                self._plot_cutout(fname)

    def _on_cutout_next(self):
        self._cutout_idx += 1
        fname = self.res_file_combo.currentText().strip()
        if fname:
            self._plot_cutout(fname)

    def _get_iter_records(self, fname: str) -> list[dict]:
        meta = self._residual_meta.get(fname, {})
        recs = meta.get("iters", []) if isinstance(meta, dict) else []
        if not isinstance(recs, list):
            return []
        out = [r for r in recs if isinstance(r, dict)]
        out.sort(key=lambda d: int(d.get("iter", 0)))
        return out

    def _get_iter_record_by_no(self, fname: str, iter_no: int) -> dict | None:
        for rec in self._get_iter_records(fname):
            if int(rec.get("iter", -1)) == int(iter_no):
                return rec
        return None

    def _refresh_residual_iter_combo(self, fname: str):
        recs = self._get_iter_records(fname)
        self.res_iter_combo.blockSignals(True)
        self.res_iter_combo.clear()
        for rec in recs:
            i = int(rec.get("iter", 0))
            self.res_iter_combo.addItem(f"{i}", i)
        if self.res_iter_combo.count() > 0:
            self.res_iter_combo.setCurrentIndex(self.res_iter_combo.count() - 1)
        self.res_iter_combo.blockSignals(False)

    def _load_starsub_for_iter(self, fname: str, rec: dict) -> np.ndarray | None:
        out_dir = step6_psf_dir(self.params.P.result_dir)
        meta = self._residual_meta.get(fname, {})
        bkg_med = float(meta.get("bkg_med", 0.0)) if isinstance(meta, dict) else 0.0

        starsub_name = str(rec.get("starsub_path", "")).strip()
        if starsub_name:
            p = out_dir / starsub_name
            if p.exists():
                try:
                    return fits.getdata(str(p)).astype(float)
                except Exception:
                    pass

        residual_name = str(rec.get("residual_path", "")).strip()
        if residual_name:
            p = out_dir / residual_name
            if p.exists():
                try:
                    res = fits.getdata(str(p)).astype(float)
                    return res + bkg_med
                except Exception:
                    pass

        return None

    def _load_xy_npy_for_iter(self, rec: dict, key: str, max_points: int = 500) -> np.ndarray:
        out_dir = step6_psf_dir(self.params.P.result_dir)
        arr_name = str(rec.get(key, "")).strip()
        if not arr_name:
            return np.zeros((0, 2), dtype=float)
        p = out_dir / arr_name
        if not p.exists():
            return np.zeros((0, 2), dtype=float)
        try:
            arr = np.load(str(p), allow_pickle=False)
            arr = np.asarray(arr, dtype=float)
            if arr.ndim != 2 or arr.shape[1] < 2:
                return np.zeros((0, 2), dtype=float)
            arr = arr[:, :2]
            finite = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1])
            arr = arr[finite]
            if int(max_points) > 0:
                return arr[:max(0, int(max_points))]
            return arr
        except Exception:
            return np.zeros((0, 2), dtype=float)

    def _load_boxxy_for_iter(self, rec: dict, max_boxes: int = 500) -> np.ndarray:
        # Preferred: delta boxes (applied-from-previous + detected-this-iter).
        arr = self._load_xy_npy_for_iter(rec, "boxxy_path", max_points=max_boxes)
        if len(arr):
            return arr
        # Fallback 1: compose from separate arrays if present.
        arr_applied = self._load_xy_npy_for_iter(rec, "appliedxy_path", max_points=0)
        arr_detected = self._load_xy_npy_for_iter(rec, "detxy_path", max_points=0)
        if len(arr_applied) and len(arr_detected):
            arr = np.vstack([arr_applied, arr_detected])
        elif len(arr_applied):
            arr = arr_applied
        elif len(arr_detected):
            arr = arr_detected
        else:
            # Fallback 2 (legacy): all fitted stars in iter.
            arr = self._load_xy_npy_for_iter(rec, "fitxy_path", max_points=0)
        if int(max_boxes) > 0:
            return arr[:max(0, int(max_boxes))]
        return arr

    def _load_modelxy_for_iter(self, rec: dict, max_points: int = 500) -> np.ndarray:
        arr = self._load_xy_npy_for_iter(rec, "modelxy_path", max_points=max_points)
        if len(arr):
            return arr
        # Backward compatibility with runs before modelxy_path existed:
        # iter>=2 should map to "new in this iter" rather than cumulative fit list.
        iter_no = int(rec.get("iter", 1))
        if iter_no > 1:
            arr = self._load_xy_npy_for_iter(rec, "detxy_path", max_points=max_points)
            if len(arr):
                return arr
        return self._load_xy_npy_for_iter(rec, "fitxy_path", max_points=max_points)

    def _resolve_fits_path_window(self, fname: str) -> Path | None:
        if self.use_cropped and crop_is_active(self.params.P.result_dir):
            cdir = step2_cropped_dir(self.params.P.result_dir)
            cpath = cdir / fname
            if cpath.exists():
                return cpath
            legacy = self.params.P.result_dir / "cropped" / fname
            if legacy.exists():
                return legacy
        fpath = Path(self.params.P.data_dir) / fname
        return fpath if fpath.exists() else None

    # ── Cutout viewer ─────────────────────────────────────────────────────────

    def _plot_cutout(self, fname: str):  # noqa: C901
        """Show Raw | Star-subtracted cutouts for the selected star."""
        if fname not in self._residual_meta:
            self.res_fig.clf()
            ax = self.res_fig.add_subplot(111)
            ax.text(
                0.5, 0.5,
                "No residual result yet for this frame.\n(Still processing or frame skipped/failed)",
                transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="gray",
            )
            ax.set_title(fname, fontsize=9)
            self.res_star_label.setText("0/0")
            self.res_info_label.setText("waiting for residual_meta...")
            self.res_canvas.draw_idle()
            return
        recs = self._get_iter_records(fname)
        if not recs:
            return

        iter_no = self.res_iter_combo.currentData()
        selected = recs[-1]
        if iter_no is not None:
            for rec in recs:
                if int(rec.get("iter", -1)) == int(iter_no):
                    selected = rec
                    break

        # Fixed semantics requested by user:
        # - iter1: detected/fitted stars in iter1
        # - iter>=2: stars detected from residual(iter-1)
        iter_val = int(selected.get("iter", 0))
        det_xy = self._load_xy_npy_for_iter(selected, "detxy_path", max_points=0)
        model_xy = self._load_modelxy_for_iter(selected, max_points=0)
        if iter_val <= 1:
            xy_list = model_xy
            mode_label = "iter1 fitted stars"
        else:
            xy_list = det_xy if len(det_xy) > 0 else model_xy
            mode_label = f"iter{iter_val} detected-from-residual"

        res_std = float(selected.get("residual_std", np.nan))
        n_new_raw = int(selected.get("n_new_raw", 0))
        n_new_kept = int(selected.get("n_new_kept", 0))

        # Cutout half-size: driven by epsf_size_px so the PSF footprint is visible
        epsf_sz = int(selected.get("epsf_size_px", 25))
        half = max(epsf_sz // 2 + 4, 10)

        # Load raw FITS
        raw_img = None
        try:
            p = self._resolve_fits_path_window(fname)
            if p is not None:
                raw_img = fits.getdata(str(p)).astype(float)
        except Exception:
            pass

        # Load starsub (= raw - PSF model, background intact)
        starsub_img = self._load_starsub_for_iter(fname, selected)
        prev_sub_img = None
        if iter_val >= 2:
            prev_rec = self._get_iter_record_by_no(fname, iter_val - 1)
            if prev_rec is not None:
                prev_sub_img = self._load_starsub_for_iter(fname, prev_rec)

        full_cut_sz = 2 * half + 1

        def _cut_at(img, x_val: float, y_val: float):
            if img is None:
                return None
            nr, nc = img.shape
            cx = int(round(float(x_val)))
            cy = int(round(float(y_val)))
            x0 = max(0, cx - half)
            x1 = min(nc, cx + half + 1)
            y0 = max(0, cy - half)
            y1 = min(nr, cy + half + 1)
            cut = img[y0:y1, x0:x1]
            if cut.size == 0:
                return None
            if cut.shape == (full_cut_sz, full_cut_sz):
                return cut
            # Edge source: pad with NaN so the star stays centred in the panel.
            # NaN renders as the colormap bad-colour (neutral), making the
            # padding region visually distinct from real background.
            padded = np.full((full_cut_sz, full_cut_sz), np.nan, dtype=np.float64)
            dst_y = max(0, half - cy)
            dst_x = max(0, half - cx)
            padded[dst_y:dst_y + (y1 - y0), dst_x:dst_x + (x1 - x0)] = cut
            return padded

        # ── Filter xy_list to photometry-successful sources only ─────────────
        # Load the step6 photometry TSV and keep only positions where
        # mag_psf is finite and FLAG_SAT is not set.  Saturated / edge /
        # fit-fail sources have NaN mag_psf or FLAG_SAT=1 — showing their
        # cutouts is misleading (over-subtraction rings, clipped PSF, etc.).
        if len(xy_list) > 0:
            try:
                _psf_tsv = step6_psf_dir(self.params.P.result_dir) / f"photometry_{fname}.tsv"
                if _psf_tsv.exists():
                    _df_phot = pd.read_csv(_psf_tsv, sep="\t")
                    _good = (
                        pd.to_numeric(_df_phot.get("mag_psf", pd.Series(dtype=float)), errors="coerce").notna() &
                        ((pd.to_numeric(_df_phot.get("flags_psf", pd.Series(0, index=_df_phot.index)),
                                        errors="coerce").fillna(0).astype(int) & self.FLAG_SAT) == 0)
                    )
                    _good_xy = _df_phot.loc[_good, ["x_fit", "y_fit"]].to_numpy(dtype=float)
                    if len(_good_xy) > 0:
                        _tree_good = cKDTree(_good_xy)
                        _d, _ = _tree_good.query(xy_list, k=1, workers=1)
                        _match_r = 1.5  # px
                        _keep = np.asarray(_d, dtype=float) <= _match_r
                        if np.any(_keep):
                            xy_list = xy_list[_keep]
            except Exception:
                pass  # on any error fall through and show all sources

        # Move edge sources to the end so idx=0 always shows a well-centred star.
        if raw_img is not None and len(xy_list) > 1:
            _nr, _nc = raw_img.shape
            _not_edge = (
                (xy_list[:, 0] >= half) & (xy_list[:, 0] < _nc - half) &
                (xy_list[:, 1] >= half) & (xy_list[:, 1] < _nr - half)
            )
            if np.any(_not_edge) and not np.all(_not_edge):
                _order = np.concatenate([np.where(_not_edge)[0], np.where(~_not_edge)[0]])
                xy_list = xy_list[_order]

        n = int(len(xy_list))
        idx = max(0, min(self._cutout_idx, n - 1)) if n > 0 else 0
        self._cutout_idx = idx
        self.res_star_label.setText(f"{idx + 1}/{n}" if n > 0 else "0/0")

        self.res_fig.clf()
        if n == 0:
            self.res_info_label.setText(
                f"iter {iter_val} | stars={n} | "
                f"new(raw/used)={n_new_raw}/{n_new_kept} | res_std={res_std:.3f}"
            )
            ax = self.res_fig.add_subplot(111)
            ax.text(
                0.5, 0.5,
                f"No sources for {mode_label}.",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=10, color="gray",
            )
            ax.set_title(f"{fname}  iter {iter_val}", fontsize=9)
            self.res_canvas.draw_idle()
            return

        x_c, y_c = float(xy_list[idx, 0]), float(xy_list[idx, 1])

        # ── Classify iter2+ detection: "신규검출" vs "재검출" ──────────────────
        # Compare the star's fitted position against step4 seed positions.
        # Within 2px → the seed existed in step4 (iter1 fit it but subtraction was poor).
        # Beyond 2px → genuinely new source not in step4.
        seed_tag = ""
        if iter_val >= 2:
            out_dir_ui = step6_psf_dir(self.params.P.result_dir)
            meta_ui = self._residual_meta.get(fname, {})
            seedxy_name = meta_ui.get("seedxy_path", "") if isinstance(meta_ui, dict) else ""
            if seedxy_name:
                seed_path_ui = out_dir_ui / seedxy_name
                if seed_path_ui.exists():
                    try:
                        seed_xy_ui = np.load(str(seed_path_ui)).astype(float)
                        if len(seed_xy_ui) > 0:
                            d_seed = np.hypot(seed_xy_ui[:, 0] - x_c, seed_xy_ui[:, 1] - y_c)
                            if np.min(d_seed) <= 2.0:
                                seed_tag = "재검출 (step4 기검출)"
                            else:
                                seed_tag = "신규검출 (step4 미검출)"
                    except Exception:
                        pass

        # Edge tag: shown when the star is within `half` pixels of the image boundary.
        edge_tag = ""
        if raw_img is not None:
            _nr, _nc = raw_img.shape
            if not (half <= x_c < _nc - half and half <= y_c < _nr - half):
                edge_tag = "경계소스"

        tags = "  " + "  ".join(f"[{t}]" for t in [seed_tag, edge_tag] if t) if (seed_tag or edge_tag) else ""
        self.res_info_label.setText(
            f"iter {iter_val} | stars={n} | "
            f"new(raw/used)={n_new_raw}/{n_new_kept} | res_std={res_std:.3f} | "
            f"xy=({x_c:.2f},{y_c:.2f}){tags}"
        )

        def _cut(img):
            return _cut_at(img, x_c, y_c)

        cut_raw = _cut(raw_img)
        cut_sub = _cut(starsub_img)

        # Requested flow view:
        # iter1: Raw vs After iter1
        # iterN>=2: After iterN-1 | Detected on residual(iterN-1) | After iterN
        panels: list[dict] = []
        if iter_val <= 1:
            panels = [
                {"img": cut_raw, "title": "Raw", "mark_detect": False},
                {"img": cut_sub, "title": "After iter 1", "mark_detect": False},
            ]
        else:
            cut_prev_sub = _cut(prev_sub_img)
            detect_panel_title = f"Detected on residual iter {iter_val - 1}"
            if seed_tag:
                detect_panel_title += f"\n[{_KO_TO_ASCII.get(seed_tag, seed_tag)}]"
            panels = [
                {"img": cut_prev_sub, "title": f"After iter {iter_val - 1}", "mark_detect": False},
                {
                    "img": cut_prev_sub,
                    "title": detect_panel_title,
                    "mark_detect": True,
                },
                {"img": cut_sub, "title": f"After iter {iter_val}", "mark_detect": False},
            ]

        # Use one grayscale stretch for all panels.
        gray_vmin = gray_vmax = None
        for p in panels:
            img_i = p.get("img", None)
            if img_i is not None and img_i.size > 0:
                gray_vmin, gray_vmax = np.nanpercentile(img_i, [1, 99])
                break

        for i, p in enumerate(panels):
            cut = p.get("img", None)
            title = str(p.get("title", ""))
            mark_detect = bool(p.get("mark_detect", False))
            ax = self.res_fig.add_subplot(1, len(panels), i + 1)
            if cut is not None and cut.size > 0:
                im = ax.imshow(
                    cut,
                    origin="lower",
                    cmap="gray",
                    vmin=gray_vmin,
                    vmax=gray_vmax,
                    interpolation="nearest",
                )
                self.res_fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                # Crosshair at star center
                cy_c = (cut.shape[0] - 1) / 2.0
                cx_c = (cut.shape[1] - 1) / 2.0
                ax.axhline(cy_c, color="#FF4444", lw=0.8, alpha=0.55, ls="--")
                ax.axvline(cx_c, color="#FF4444", lw=0.8, alpha=0.55, ls="--")
                if mark_detect:
                    ax.plot(
                        [cx_c], [cy_c],
                        marker="o", markersize=8,
                        markerfacecolor="none",
                        markeredgecolor="#FF5555",
                        markeredgewidth=1.2,
                    )
            else:
                ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                        ha="center", va="center", fontsize=10, color="gray")
            ax.set_title(title, fontsize=8)
            ax.set_xlabel("Δx (px)", fontsize=7)
            ax.set_ylabel("Δy (px)", fontsize=7)

        suptitle_tags = "  " + "  ".join(f"[{_KO_TO_ASCII.get(t, t)}]" for t in [seed_tag, edge_tag] if t) if (seed_tag or edge_tag) else ""
        self.res_fig.suptitle(
            f"{fname}  |  {mode_label} #{idx + 1}/{n}{suptitle_tags}",
            fontsize=8, y=1.01,
        )
        self.res_fig.tight_layout()
        self.res_canvas.draw_idle()

    # ── Frame table refresh ───────────────────────────────────────────────────

    def update_frame_table(self):
        idx_path = step6_psf_dir(self.params.P.result_dir) / "photometry_index.csv"
        if not idx_path.exists() or not hasattr(self, "frame_table"):
            return
        try:
            idx = pd.read_csv(idx_path)
        except Exception:
            return
        self.frame_table.setRowCount(len(idx))
        for r, row in enumerate(idx.itertuples(index=False)):
            self.frame_table.setItem(r, 0, QTableWidgetItem(str(getattr(row, "file", ""))))
            self.frame_table.setItem(r, 1, QTableWidgetItem(str(getattr(row, "filter", ""))))
            self.frame_table.setItem(r, 2, QTableWidgetItem(str(int(_safe_float(getattr(row, "n", 0), 0)))))
            self.frame_table.setItem(r, 3, QTableWidgetItem(str(int(_safe_float(getattr(row, "n_goodmag", 0), 0)))))
            self.frame_table.setItem(r, 4, QTableWidgetItem(str(int(_safe_float(getattr(row, "n_fail", 0), 0)))))
            self.frame_table.setItem(r, 5, QTableWidgetItem(str(int(_safe_float(getattr(row, "n_new_iter", 0), 0)))))

    # ── QC Report ─────────────────────────────────────────────────────────────

    def _refresh_qc(self):
        """Compute PSF QC statistics and update the QC tab text + Ap vs PSF plot."""
        if not hasattr(self, "qc_text"):
            return
        psf_dir = step6_psf_dir(self.params.P.result_dir)
        idx_path = psf_dir / "photometry_index.csv"
        if not idx_path.exists():
            self.qc_text.setPlainText("photometry_index.csv not found.\nRun Step 6 first.")
            self._cmp_merged_df = None
            self._plot_mag_comparison()
            return
        try:
            idx = pd.read_csv(idx_path)
            tsv_files = sorted(psf_dir.glob("photometry_*.tsv"))
            if not tsv_files:
                self.qc_text.setPlainText("No photometry TSV files found.")
                return
            all_df = pd.concat([pd.read_csv(f, sep="\t") for f in tsv_files], ignore_index=True)
            meta_rows = []
            for mf in sorted(psf_dir.glob("residual_meta_*.json")):
                try:
                    m = json.loads(mf.read_text(encoding="utf-8"))
                    for it in m.get("iters", []):
                        meta_rows.append({
                            "filter": m.get("filter", "?"),
                            "iter": it.get("iter"),
                            "residual_std": it.get("residual_std", np.nan),
                        })
                except Exception:
                    pass
            meta_df = pd.DataFrame(meta_rows) if meta_rows else pd.DataFrame()

            good = all_df[all_df["flags_psf"] == 0].copy() if "flags_psf" in all_df.columns else all_df.copy()
            filters = sorted(all_df["FILTER"].dropna().unique().tolist()) if "FILTER" in all_df.columns else []
            n_total = len(all_df)
            n_clean = len(good)

            W = 60
            lines = []
            lines.append("─" * W)
            lines.append("  PSF Photometry QC Report")
            lines.append("─" * W)
            filt_counts = "  ".join(
                f"{f}:{(idx['filter'] == f).sum()}" for f in filters if "filter" in idx.columns
            )
            lines.append(f"  총 프레임    : {len(idx)}  ({filt_counts})")
            lines.append(f"  총 검출 소스 : {n_total:,}")
            lines.append(f"  flags=0      : {n_clean:,} / {n_total:,} = {100 * n_clean / max(n_total, 1):.1f}%")
            lines.append("")

            # 1. Filter stats
            lines.append("  1. 필터별 검출 통계")
            lines.append(f"  {'필터':^4}  {'프레임':^6}  {'평균검출':^8}  {'성공률':^7}  {'실패율':^6}  {'iter2추가':^9}")
            lines.append("  " + "─" * 52)
            for filt in filters:
                si = idx[idx["filter"] == filt] if "filter" in idx.columns else pd.DataFrame()
                if si.empty:
                    continue
                avg_n = si["n"].mean() if "n" in si.columns else 0
                avg_g = si["n_goodmag"].mean() if "n_goodmag" in si.columns else avg_n
                ok_pct = 100 * avg_g / max(avg_n, 1)
                avg_new = si["n_new_iter"].mean() if "n_new_iter" in si.columns else 0
                lines.append(
                    f"  {filt:^4}  {len(si):^6}  {avg_n:^8.0f}  {ok_pct:^6.1f}%  "
                    f"{100 - ok_pct:^5.1f}%  avg {avg_new:.1f}"
                )
            lines.append("")

            # 2. Mag range & error
            lines.append("  2. 등급 범위 (mag_psf, flags=0)")
            lines.append(f"  {'필터':^4}  {'범위':^15}  {'평균':^6}  {'중앙값':^6}  {'σ':^5}  {'err중앙값':^9}")
            lines.append("  " + "─" * 58)
            for filt in filters:
                sub = good[good["FILTER"] == filt] if "FILTER" in good.columns else pd.DataFrame()
                mag = sub["mag_psf"].dropna() if "mag_psf" in sub.columns else pd.Series()
                err = sub["mag_psf_err"].dropna() if "mag_psf_err" in sub.columns else pd.Series()
                if mag.empty:
                    continue
                lines.append(
                    f"  {filt:^4}  {mag.min():.2f} ~ {mag.max():.2f}  "
                    f"{mag.mean():^6.2f}  {mag.median():^6.2f}  {mag.std():^5.2f}  "
                    f"{err.median():^9.4f}" if not err.empty else
                    f"  {filt:^4}  {mag.min():.2f} ~ {mag.max():.2f}  "
                    f"{mag.mean():^6.2f}  {mag.median():^6.2f}  {mag.std():^5.2f}  {'N/A':^9}"
                )
            lines.append("")

            # 3. SNR
            if "snr_psf" in good.columns:
                lines.append("  3. SNR 분포 (flags=0)")
                lines.append(f"  {'필터':^4}  {'10%':^8}  {'median':^8}  {'90%':^8}")
                lines.append("  " + "─" * 36)
                for filt in filters:
                    sub = good[good["FILTER"] == filt]["snr_psf"].dropna() if "FILTER" in good.columns else pd.Series()
                    if sub.empty:
                        continue
                    lines.append(
                        f"  {filt:^4}  {np.percentile(sub, 10):^8.1f}  "
                        f"{sub.median():^8.1f}  {np.percentile(sub, 90):^8.1f}"
                    )
                lines.append("")

            # 4. qfit
            if "qfit" in good.columns:
                lines.append("  4. PSF 적합 품질 (qfit, flags=0)")
                lines.append(f"  {'필터':^4}  {'중앙값':^8}  {'>5 비율':^8}")
                lines.append("  " + "─" * 28)
                for filt in filters:
                    sub = good[good["FILTER"] == filt]["qfit"].dropna() if "FILTER" in good.columns else pd.Series()
                    if sub.empty:
                        continue
                    bad_pct = 100 * (sub > 5).sum() / max(len(sub), 1)
                    warn = " ⚠" if bad_pct > 5 else ""
                    lines.append(f"  {filt:^4}  {sub.median():^8.3f}  {bad_pct:^7.1f}%{warn}")
                lines.append("")

            # 5. Residual STD
            if not meta_df.empty and "residual_std" in meta_df.columns:
                lines.append("  5. Residual STD (ADU, per frame mean)")
                lines.append(f"  {'필터':^4}  {'iter1':^10}  {'iter2':^10}")
                lines.append("  " + "─" * 30)
                for filt in filters:
                    i1 = meta_df[(meta_df["filter"] == filt) & (meta_df["iter"] == 1)]["residual_std"]
                    i2 = meta_df[(meta_df["filter"] == filt) & (meta_df["iter"] == 2)]["residual_std"]
                    if i1.empty:
                        continue
                    i2_mean = f"{i2.mean():.2f}" if not i2.empty else "N/A"
                    lines.append(f"  {filt:^4}  {i1.mean():^10.2f}  {i2_mean:^10}")
                lines.append("")

            self.qc_text.setPlainText("\n".join(lines))
        except Exception as e:
            self.qc_text.setPlainText(f"QC 생성 오류: {e}")

        self._cmp_merged_df = None
        self._plot_mag_comparison()

    # ── Aperture vs PSF magnitude comparison ──────────────────────────────────

    def _plot_mag_comparison(self):  # noqa: C901
        """Scatter: mag_ap (Step5) vs mag_psf (Step6), merged on det_uid."""
        if not hasattr(self, "cmp_fig"):
            return

        _FILT_COLORS = {
            "u": "#9467bd", "g": "#2ca02c", "r": "#d62728",
            "i": "#ff7f0e", "z": "#8c564b", "b": "#1f77b4",
            "v": "#bcbd22", "ha": "#e377c2",
        }

        ap_dir = step5_aperture_dir(self.params.P.result_dir)
        psf_dir = step6_psf_dir(self.params.P.result_dir)

        # Load and merge TSVs — cached; only re-read from disk when _cmp_merged_df is None
        # (step5 already outputs only apcorr=True frames, so no extra filter needed here)
        if not hasattr(self, "_cmp_merged_df") or self._cmp_merged_df is None:
            merged_rows = []
            split_excluded_total = 0
            for psf_tsv in sorted(psf_dir.glob("photometry_*.tsv")):
                fname_key = psf_tsv.name[len("photometry_"):]
                ap_tsv = ap_dir / f"photometry_{fname_key}"
                if not ap_tsv.exists():
                    continue
                try:
                    df_ap = pd.read_csv(ap_tsv, sep="\t")
                    df_psf = pd.read_csv(psf_tsv, sep="\t")
                except Exception:
                    continue
                if "det_uid" not in df_ap.columns or "det_uid" not in df_psf.columns:
                    continue
                ap_cols = [c for c in ("det_uid", "mag_ap", "mag_ap_err", "r_ap_px") if c in df_ap.columns]
                try:
                    # Crowd-safe compare path:
                    # if seed_uid + flux_psf_e exist, aggregate all split PSF components
                    # back to their original Step4 seed before AP-vs-PSF merge.
                    if {"seed_uid", "flux_psf_e", "exptime"} <= set(df_psf.columns):
                        zp = _to_float(getattr(self.params.P, "zp_initial", 25.0), 25.0)
                        p = df_psf.copy()
                        p["seed_uid"] = pd.to_numeric(p["seed_uid"], errors="coerce")
                        p["flux_psf_e"] = pd.to_numeric(p["flux_psf_e"], errors="coerce")
                        p["exptime"] = pd.to_numeric(p["exptime"], errors="coerce")
                        p = p[
                            np.isfinite(p["seed_uid"]) &
                            (p["seed_uid"] >= 0) &
                            np.isfinite(p["flux_psf_e"]) &
                            (p["flux_psf_e"] > 0) &
                            np.isfinite(p["exptime"]) &
                            (p["exptime"] > 0)
                        ].copy()
                        if len(p) == 0:
                            continue
                        agg_map = {
                            "flux_psf_e": "sum",
                            "exptime": "median",
                        }
                        for c in ("FILTER", "qfit", "iter_found", "snr_psf", "flags_psf"):
                            if c in p.columns:
                                agg_map[c] = "median" if c in {"qfit", "iter_found"} else "first"
                        g = p.groupby("seed_uid", as_index=False).agg(agg_map)
                        comp = p.groupby("seed_uid", as_index=False).size().rename(columns={"size": "n_comp"})
                        g = g.merge(comp, on="seed_uid", how="left")
                        # Strict compare mode: for AP-vs-PSF calibration consistency,
                        # exclude decomposed seeds (one AP seed split into multiple PSF components).
                        if bool(getattr(self.params.P, "step6_compare_exclude_split", True)):
                            n_before_g = int(len(g))
                            g = g[g["n_comp"] == 1].copy()
                            split_excluded_total += max(0, n_before_g - int(len(g)))
                            if len(g) == 0:
                                continue
                        g["det_uid"] = g["seed_uid"].astype(int)
                        g["mag_psf"] = zp - 2.5 * np.log10(
                            np.maximum(g["flux_psf_e"].to_numpy(float), 1e-30)
                            / np.maximum(g["exptime"].to_numpy(float), 1e-30)
                        )
                        psf_cols = [c for c in ("det_uid", "mag_psf", "FILTER", "qfit",
                                                "iter_found", "snr_psf", "flags_psf") if c in g.columns]
                        m = df_ap[ap_cols].merge(g[psf_cols], on="det_uid", how="inner")
                    else:
                        psf_cols = [c for c in ("det_uid", "mag_psf", "mag_psf_err", "FILTER", "qfit",
                                                "iter_found", "snr_psf", "flags_psf") if c in df_psf.columns]
                        m = df_ap[ap_cols].merge(df_psf[psf_cols], on="det_uid", how="inner")
                    # Strip .tsv so FRAME matches apcorr_summary.csv "file" column (FITS filename)
                    m["FRAME"] = fname_key[:-4] if fname_key.endswith(".tsv") else fname_key
                    merged_rows.append(m)
                except Exception:
                    continue
            self._cmp_merged_df = pd.concat(merged_rows, ignore_index=True) if merged_rows else pd.DataFrame()
            self._cmp_split_excluded_total = int(split_excluded_total)

        self.cmp_fig.clf()

        def _empty(msg):
            ax = self.cmp_fig.add_subplot(111)
            ax.text(0.5, 0.5, msg, transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, color="gray")
            self.cmp_canvas.draw_idle()
            self.cmp_stats_label.setText(msg)

        if self._cmp_merged_df.empty:
            _empty("No matched data.\nRun Step 5 and Step 6 first.")
            return

        df = self._cmp_merged_df.copy()

        # Refresh filter/frame selectors from available merged data.
        if hasattr(self, "cmp_filter_combo"):
            try:
                _fvals = sorted(df["FILTER"].dropna().astype(str).unique().tolist()) if "FILTER" in df.columns else []
                _cur = self.cmp_filter_combo.currentText().strip() or "all"
                self.cmp_filter_combo.blockSignals(True)
                self.cmp_filter_combo.clear()
                self.cmp_filter_combo.addItem("all")
                for _v in _fvals:
                    self.cmp_filter_combo.addItem(_v)
                self.cmp_filter_combo.setCurrentText(_cur if _cur in (["all"] + _fvals) else "all")
                self.cmp_filter_combo.blockSignals(False)
            except Exception:
                pass
        if hasattr(self, "cmp_frame_combo"):
            try:
                _frames = sorted(df["FRAME"].dropna().astype(str).unique().tolist()) if "FRAME" in df.columns else []
                _cur = self.cmp_frame_combo.currentText().strip() or "all"
                self.cmp_frame_combo.blockSignals(True)
                self.cmp_frame_combo.clear()
                self.cmp_frame_combo.addItem("all")
                for _v in _frames:
                    self.cmp_frame_combo.addItem(_v)
                self.cmp_frame_combo.setCurrentText(_cur if _cur in (["all"] + _frames) else "all")
                self.cmp_frame_combo.blockSignals(False)
            except Exception:
                pass

        df["mag_ap"] = pd.to_numeric(df.get("mag_ap"), errors="coerce")
        df["mag_psf"] = pd.to_numeric(df.get("mag_psf"), errors="coerce")
        df = df[np.isfinite(df["mag_ap"]) & np.isfinite(df["mag_psf"])].copy()

        if len(df) == 0:
            _empty("All magnitudes are NaN.\nCheck step5/step6 outputs.")
            return

        df["delta"] = df["mag_ap"] - df["mag_psf"]
        n_before = int(len(df))

        # Pre-convert numeric filter columns once
        if "flags_psf" in df.columns:
            df["flags_psf"] = pd.to_numeric(df["flags_psf"], errors="coerce")
        if "snr_psf" in df.columns:
            df["snr_psf"] = pd.to_numeric(df["snr_psf"], errors="coerce")
        if "qfit" in df.columns:
            df["qfit"] = pd.to_numeric(df["qfit"], errors="coerce")

        # Selector filters (filter/frame)
        if hasattr(self, "cmp_filter_combo") and "FILTER" in df.columns:
            _fsel = str(self.cmp_filter_combo.currentText()).strip()
            if _fsel and _fsel.lower() != "all":
                df = df[df["FILTER"].astype(str).str.lower() == _fsel.lower()].copy()
        if hasattr(self, "cmp_frame_combo") and "FRAME" in df.columns:
            _rsel = str(self.cmp_frame_combo.currentText()).strip()
            if _rsel and _rsel.lower() != "all":
                df = df[df["FRAME"].astype(str) == _rsel].copy()

        # User filters
        if getattr(self, "cmp_flags0_only", None) is not None and self.cmp_flags0_only.isChecked() and "flags_psf" in df.columns:
            df = df[np.isfinite(df["flags_psf"]) & (df["flags_psf"] == 0)].copy()
        if getattr(self, "cmp_snr_min", None) is not None:
            _snr_min = float(self.cmp_snr_min.value())
            if _snr_min > 0 and "snr_psf" in df.columns:
                df = df[np.isfinite(df["snr_psf"]) & (df["snr_psf"] >= _snr_min)].copy()
        if getattr(self, "cmp_qfit_max", None) is not None:
            _qmax = float(self.cmp_qfit_max.value())
            if _qmax > 0 and "qfit" in df.columns:
                df = df[np.isfinite(df["qfit"]) & (df["qfit"] <= _qmax)].copy()
        if getattr(self, "cmp_dmag_clip", None) is not None:
            _dclip = float(self.cmp_dmag_clip.value())
            if _dclip > 0:
                df = df[np.isfinite(df["delta"]) & (np.abs(df["delta"]) <= _dclip)].copy()

        if len(df) == 0:
            _empty("No data after filters.\nRelax SNR/qfit/|Δmag| settings.")
            return

        filt_col = "FILTER" if "FILTER" in df.columns else None

        ax1 = self.cmp_fig.add_subplot(121)
        ax2 = self.cmp_fig.add_subplot(122)

        stats_parts = []
        groups = df.groupby(filt_col) if filt_col else [("all", df)]
        for filt, sub in groups:
            color = _FILT_COLORS.get(str(filt).lower(), "#999999")
            ax1.scatter(sub["mag_ap"], sub["mag_psf"],
                        s=4, alpha=0.35, color=color, label=str(filt), rasterized=True)
            ax2.scatter(sub["mag_ap"], sub["delta"],
                        s=4, alpha=0.35, color=color, label=str(filt), rasterized=True)
            n = len(sub)
            med = float(np.nanmedian(sub["delta"]))
            std = float(np.nanstd(sub["delta"]))
            stats_parts.append(f"{filt}: N={n}  Δmed={med:+.3f}  σ={std:.3f}")

        # 1:1 reference line (ax1)
        all_mag = np.concatenate([df["mag_ap"].values, df["mag_psf"].values])
        lo, hi = np.nanmin(all_mag) - 0.2, np.nanmax(all_mag) + 0.2
        ax1.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5, zorder=0)
        ax1.set_xlim(lo, hi)
        ax1.set_ylim(lo, hi)
        ax1.set_xlabel("mag_ap", fontsize=9)
        ax1.set_ylabel("mag_psf", fontsize=9)
        ax1.set_title("Aperture vs PSF magnitude", fontsize=9)
        ax1.legend(fontsize=7, markerscale=2, loc="upper left")

        # Zero ± 0.05 reference lines (ax2) + robust median guide
        dmed_all = float(np.nanmedian(df["delta"]))
        ax2.axhline(
            dmed_all,
            color="#D62728",
            lw=2.0,
            ls="-",
            alpha=0.95,
            zorder=1,
            label=f"Δmag median {dmed_all:+.3f}",
        )
        ax2.axhline(0.0,  color="k",    lw=0.8, ls="--", alpha=0.6, zorder=0)
        ax2.axhline(+0.05, color="gray", lw=0.5, ls=":",  alpha=0.5, zorder=0)
        ax2.axhline(-0.05, color="gray", lw=0.5, ls=":",  alpha=0.5, zorder=0)
        ax2.set_xlabel("mag_ap", fontsize=9)
        ax2.set_ylabel("Δmag  (Ap − PSF)", fontsize=9)
        ax2.set_title("Δmag vs mag_ap", fontsize=9)
        ax2.legend(fontsize=7, markerscale=2, loc="upper left")

        self.cmp_fig.tight_layout()
        self.cmp_canvas.draw_idle()
        self.cmp_stats_label.setText(
            f"N={len(df)}/{n_before}  |  split_excluded={int(getattr(self, '_cmp_split_excluded_total', 0))}  |  "
            + "  |  ".join(stats_parts)
        )

    # ── Load existing results from disk (called on restore_state) ─────────────

    def _load_from_disk(self):
        """Reload EPSF models and residual images from disk into memory caches."""
        out_dir = step6_psf_dir(self.params.P.result_dir)
        if not out_dir.exists():
            return
        self._residual_meta.clear()

        def _epsf_display_key_from_path(epsf_path: Path) -> str:
            stem = epsf_path.stem  # epsf_model_{filter}_{frame_stem} or legacy epsf_model_{filter}
            body = stem.replace("epsf_model_", "", 1)
            if "_" not in body:
                return ""
            filt, frame_stem = body.split("_", 1)
            return f"{frame_stem} | {filt}"

        # Load EPSF FITS files
        for epsf_path in out_dir.glob("epsf_model_*.fits"):
            try:
                display_key = _epsf_display_key_from_path(epsf_path)
                if not display_key:
                    continue
                arr = fits.getdata(str(epsf_path)).astype(float)
                self._last_epsf[display_key] = arr
            except Exception:
                pass

        if self._last_epsf:
            self.epsf_filter_combo.blockSignals(True)
            self.epsf_filter_combo.clear()
            self.epsf_filter_combo.addItems(sorted(self._last_epsf.keys()))
            self.epsf_filter_combo.setCurrentIndex(0)
            self.epsf_filter_combo.blockSignals(False)
            first_filter = self.epsf_filter_combo.currentText()
            if first_filter:
                self._plot_epsf(first_filter)

        # Load residual metadata (preferred, supports iteration-wise view).
        for meta_path in sorted(out_dir.glob("residual_meta_*.json")):
            try:
                name = meta_path.name
                if not name.startswith("residual_meta_") or not name.endswith(".json"):
                    continue
                fname = name[len("residual_meta_"):-len(".json")]
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if isinstance(meta, dict):
                    self._residual_meta[fname] = meta
                    self._last_new_xy.setdefault(fname, None)
            except Exception:
                pass

        if self._residual_meta:
            self.res_file_combo.blockSignals(True)
            self.res_file_combo.clear()
            self.res_file_combo.addItems(sorted(self._residual_meta.keys()))
            self.res_file_combo.setCurrentIndex(0)
            self.res_file_combo.blockSignals(False)
            first_fname = self.res_file_combo.currentText()
            if first_fname:
                self._refresh_residual_iter_combo(first_fname)
                self._plot_cutout(first_fname)

        self._refresh_qc()  # refresh QC tab (stats + Ap vs PSF plot)

    # ── Parameters dialog ─────────────────────────────────────────────────────

    def open_parameters_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Step6 PSF Parameters")
        dialog.resize(520, 660)
        layout = QVBoxLayout(dialog)
        form = QFormLayout()

        # ── Field mode preset ──────────────────────────────────────────────
        mode_combo = QComboBox()
        mode_combo.addItem("Normal (일반)", "normal")
        mode_combo.addItem("Crowded (구상성단/혼잡장)", "crowded")
        mode_combo.addItem("Faint (희미한 필드)", "faint")
        mode_combo.addItem("Custom (수동)", "custom")
        _saved_mode = str(getattr(self.params.P, "psf_mode", "normal"))
        _mi = mode_combo.findData(_saved_mode)
        mode_combo.setCurrentIndex(_mi if _mi >= 0 else 0)
        form.addRow("Field mode:", mode_combo)

        self.p_model_mode = QComboBox()
        self.p_model_mode.addItems(["per_frame"])
        self.p_model_mode.setCurrentText("per_frame")
        form.addRow("Model mode:", self.p_model_mode)

        self.p_workers = QSpinBox()
        self.p_workers.setRange(0, 64)
        self.p_workers.setValue(_to_int(getattr(self.params.P, "psf_parallel_workers", 0), 0))
        self.p_workers.setToolTip("0 = auto/global parallel workers")
        form.addRow("PSF workers (0=auto):", self.p_workers)

        self.p_oversampling = QSpinBox()
        self.p_oversampling.setRange(1, 8)
        self.p_oversampling.setValue(_to_int(getattr(self.params.P, "psf_epsf_oversampling", 2), 2))
        form.addRow("EPSF oversampling:", self.p_oversampling)

        self.p_epsf_mult = QDoubleSpinBox()
        self.p_epsf_mult.setRange(1.0, 10.0)
        self.p_epsf_mult.setSingleStep(0.1)
        self.p_epsf_mult.setValue(_to_float(getattr(self.params.P, "psf_epsf_size_fwhm_mult", 4.0), 4.0))
        form.addRow("EPSF cutout (×FWHM):", self.p_epsf_mult)

        self.p_n_stars = QSpinBox()
        self.p_n_stars.setRange(5, 500)
        self.p_n_stars.setValue(_to_int(getattr(self.params.P, "psf_n_stars_max", 50), 50))
        form.addRow("Max stars for EPSF:", self.p_n_stars)

        self.p_isolation = QDoubleSpinBox()
        self.p_isolation.setRange(1.0, 10.0)
        self.p_isolation.setSingleStep(0.5)
        self.p_isolation.setValue(_to_float(getattr(self.params.P, "psf_isolation_fwhm_mult", 3.0), 3.0))
        form.addRow("Isolation (×FWHM):", self.p_isolation)

        self.p_fit_mult = QDoubleSpinBox()
        self.p_fit_mult.setRange(0.5, 5.0)
        self.p_fit_mult.setSingleStep(0.1)
        self.p_fit_mult.setValue(_to_float(getattr(self.params.P, "psf_fit_shape_fwhm_mult", 1.5), 1.5))
        form.addRow("Fit window (×FWHM):", self.p_fit_mult)

        self.p_max_iter = QSpinBox()
        self.p_max_iter.setRange(1, 10)
        self.p_max_iter.setValue(_to_int(getattr(self.params.P, "psf_max_iter", 2), 2))
        form.addRow("Max iterations:", self.p_max_iter)

        self.p_redetect = QDoubleSpinBox()
        self.p_redetect.setRange(1.0, 10.0)
        self.p_redetect.setSingleStep(0.5)
        self.p_redetect.setValue(_to_float(getattr(self.params.P, "psf_redetect_sigma", 4.0), 4.0))
        form.addRow("Re-detect σ (base):", self.p_redetect)

        def _make_filter_sigma_spin(attr, label):
            sp = QDoubleSpinBox()
            sp.setRange(0.0, 10.0)
            sp.setSingleStep(0.5)
            sp.setDecimals(1)
            sp.setSpecialValueText("base")
            _v = _to_float(getattr(self.params.P, attr, float("nan")), float("nan"))
            sp.setValue(0.0 if not np.isfinite(_v) else float(_v))
            sp.setToolTip("0 = use base sigma")
            form.addRow(label, sp)
            return sp

        self.p_redetect_g = _make_filter_sigma_spin("psf_redetect_sigma_g", "  Re-detect σ g-band:")
        self.p_redetect_r = _make_filter_sigma_spin("psf_redetect_sigma_r", "  Re-detect σ r-band:")
        self.p_redetect_i = _make_filter_sigma_spin("psf_redetect_sigma_i", "  Re-detect σ i-band:")

        self.p_dup_mult = QDoubleSpinBox()
        self.p_dup_mult.setRange(0.0, 5.0)
        self.p_dup_mult.setSingleStep(0.1)
        self.p_dup_mult.setValue(_to_float(getattr(self.params.P, "psf_duplicate_radius_fwhm_mult", 0.8), 0.8))
        form.addRow("Duplicate radius (×FWHM):", self.p_dup_mult)

        self.p_dup_px = QDoubleSpinBox()
        self.p_dup_px.setRange(0.0, 50.0)
        self.p_dup_px.setSingleStep(0.1)
        self.p_dup_px.setDecimals(2)
        _dup_px = _to_float(getattr(self.params.P, "psf_duplicate_radius_px", np.nan), np.nan)
        self.p_dup_px.setValue(0.0 if not np.isfinite(_dup_px) else float(_dup_px))
        self.p_dup_px.setToolTip("0이면 비활성(×FWHM 값 사용), >0이면 절대 px 반경 사용")
        form.addRow("Duplicate radius (px override):", self.p_dup_px)

        self.p_cap_per_iter = QSpinBox()
        self.p_cap_per_iter.setRange(0, 50000)
        self.p_cap_per_iter.setSingleStep(50)
        self.p_cap_per_iter.setValue(_to_int(getattr(self.params.P, "psf_new_sources_cap_per_iter", 70), 70))
        form.addRow("Max new/iter (abs):", self.p_cap_per_iter)

        self.p_cap_frac = QDoubleSpinBox()
        self.p_cap_frac.setRange(0.0, 1.0)
        self.p_cap_frac.setSingleStep(0.01)
        self.p_cap_frac.setValue(_to_float(getattr(self.params.P, "psf_new_sources_cap_frac", 0.02), 0.02))
        form.addRow("Max new/iter (frac):", self.p_cap_frac)

        self.p_fit_init_max = QSpinBox()
        self.p_fit_init_max.setRange(0, 200000)
        self.p_fit_init_max.setSingleStep(100)
        self.p_fit_init_max.setValue(_to_int(getattr(self.params.P, "psf_fit_init_max_sources", 0), 0))
        self.p_fit_init_max.setToolTip("0이면 초기 피팅 소스 무제한")
        form.addRow("Initial fit source cap (0=off):", self.p_fit_init_max)

        self.p_substar_nei_mult = QDoubleSpinBox()
        self.p_substar_nei_mult.setRange(2.0, 30.0)
        self.p_substar_nei_mult.setSingleStep(0.5)
        self.p_substar_nei_mult.setValue(_to_float(getattr(self.params.P, "psf_substar_neighbor_r_fwhm_mult", 8.0), 8.0))
        form.addRow("Substar neighbor radius (×FWHM):", self.p_substar_nei_mult)

        self.p_substar_max_src = QSpinBox()
        self.p_substar_max_src.setRange(0, 200000)
        self.p_substar_max_src.setSingleStep(100)
        self.p_substar_max_src.setValue(_to_int(getattr(self.params.P, "psf_substar_max_sources", 1500), 1500))
        self.p_substar_max_src.setToolTip("0이면 substar 이웃 소스 캡 무제한")
        form.addRow("Substar max neighbor sources:", self.p_substar_max_src)

        self.p_conv_new = QDoubleSpinBox()
        self.p_conv_new.setRange(0.0, 1.0)
        self.p_conv_new.setSingleStep(0.005)
        self.p_conv_new.setValue(_to_float(getattr(self.params.P, "psf_conv_new_frac", 0.02), 0.02))
        form.addRow("Converge new frac <", self.p_conv_new)

        self.p_conv_flux = QDoubleSpinBox()
        self.p_conv_flux.setRange(0.0, 1.0)
        self.p_conv_flux.setSingleStep(0.001)
        self.p_conv_flux.setValue(_to_float(getattr(self.params.P, "psf_flux_conv_threshold", 0.01), 0.01))
        form.addRow("Converge std improve <", self.p_conv_flux)

        self.p_use_grouper = QCheckBox("Use SourceGrouper (crowded field simultaneous fit)")
        self.p_use_grouper.setChecked(bool(getattr(self.params.P, "psf_use_grouper", True)))
        form.addRow("", self.p_use_grouper)

        self.p_use_error_img = QCheckBox("Use error image (slower, higher RAM)")
        self.p_use_error_img.setChecked(bool(getattr(self.params.P, "psf_use_error_image", False)))
        form.addRow("", self.p_use_error_img)

        self.p_sharp_lo = QDoubleSpinBox()
        self.p_sharp_lo.setRange(0.0, 1.0)
        self.p_sharp_lo.setSingleStep(0.05)
        self.p_sharp_lo.setDecimals(2)
        self.p_sharp_lo.setValue(_to_float(getattr(self.params.P, "psf_redetect_sharp_lo", 0.15), 0.15))
        form.addRow("Re-detect sharpness min:", self.p_sharp_lo)

        self.p_sharp_hi = QDoubleSpinBox()
        self.p_sharp_hi.setRange(0.0, 1.0)
        self.p_sharp_hi.setSingleStep(0.05)
        self.p_sharp_hi.setDecimals(2)
        self.p_sharp_hi.setValue(_to_float(getattr(self.params.P, "psf_redetect_sharp_hi", 0.95), 0.95))
        form.addRow("Re-detect sharpness max:", self.p_sharp_hi)

        self.p_round_max = QDoubleSpinBox()
        self.p_round_max.setRange(0.0, 2.0)
        self.p_round_max.setSingleStep(0.05)
        self.p_round_max.setDecimals(2)
        self.p_round_max.setValue(_to_float(getattr(self.params.P, "psf_redetect_round_abs_max", 0.8), 0.8))
        form.addRow("Re-detect |roundness| max:", self.p_round_max)

        self.p_save_residuals = QCheckBox("Save residual FITS (required for iter viewer)")
        self.p_save_residuals.setChecked(True)
        self.p_save_residuals.setEnabled(False)
        form.addRow("", self.p_save_residuals)

        layout.addLayout(form)

        # ── mode logic ────────────────────────────────────────────────────
        _manual_widgets = [
            self.p_n_stars, self.p_isolation, self.p_fit_mult, self.p_max_iter,
            self.p_redetect, self.p_dup_mult, self.p_dup_px,
            self.p_cap_per_iter, self.p_cap_frac, self.p_fit_init_max,
            self.p_substar_nei_mult, self.p_substar_max_src,
            self.p_conv_new, self.p_conv_flux, self.p_use_grouper,
            self.p_sharp_lo, self.p_sharp_hi, self.p_round_max,
        ]

        def _apply_mode_to_widgets(mode_key):
            p = _PSF_MODE_PRESETS.get(mode_key, _PSF_MODE_PRESETS["normal"])
            self.p_n_stars.setValue(p["psf_n_stars_max"])
            self.p_isolation.setValue(p["psf_isolation_fwhm_mult"])
            self.p_fit_mult.setValue(p["psf_fit_shape_fwhm_mult"])
            self.p_max_iter.setValue(p["psf_max_iter"])
            self.p_redetect.setValue(p["psf_redetect_sigma"])
            self.p_dup_mult.setValue(p["psf_duplicate_radius_fwhm_mult"])
            self.p_dup_px.setValue(0.0)
            self.p_cap_per_iter.setValue(p["psf_new_sources_cap_per_iter"])
            self.p_cap_frac.setValue(p["psf_new_sources_cap_frac"])
            self.p_fit_init_max.setValue(p["psf_fit_init_max_sources"])
            self.p_substar_nei_mult.setValue(p["psf_substar_neighbor_r_fwhm_mult"])
            self.p_substar_max_src.setValue(p["psf_substar_max_sources"])
            self.p_conv_new.setValue(p["psf_conv_new_frac"])
            self.p_conv_flux.setValue(p["psf_flux_conv_threshold"])
            self.p_use_grouper.setChecked(p["psf_use_grouper"])
            self.p_sharp_lo.setValue(p["psf_redetect_sharp_lo"])
            self.p_sharp_hi.setValue(p["psf_redetect_sharp_hi"])
            self.p_round_max.setValue(p["psf_redetect_round_abs_max"])

        def _refresh_mode_ui():
            is_custom = (mode_combo.currentData() == "custom")
            if not is_custom:
                _apply_mode_to_widgets(mode_combo.currentData())
            for w in _manual_widgets:
                w.setEnabled(is_custom)

        mode_combo.currentIndexChanged.connect(lambda *_: _refresh_mode_ui())
        _refresh_mode_ui()

        btns = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        btns.accepted.connect(lambda: self._save_params(dialog, mode_combo.currentData()))
        btns.rejected.connect(dialog.reject)
        layout.addWidget(btns)
        dialog.exec_()


    def _save_params(self, dialog, mode_key="normal"):
        self.params.P.psf_mode = mode_key
        self.params.P.psf_model_mode = "per_frame"
        self.params.P.psf_parallel_workers = self.p_workers.value()
        self.params.P.psf_epsf_oversampling = self.p_oversampling.value()
        self.params.P.psf_epsf_size_fwhm_mult = self.p_epsf_mult.value()
        self.params.P.psf_n_stars_max = self.p_n_stars.value()
        self.params.P.psf_isolation_fwhm_mult = self.p_isolation.value()
        self.params.P.psf_fit_shape_fwhm_mult = self.p_fit_mult.value()
        self.params.P.psf_max_iter = self.p_max_iter.value()
        self.params.P.psf_redetect_sigma = self.p_redetect.value()
        def _spin_to_sigma(sp):
            v = sp.value()
            return float("nan") if v <= 0.0 else v
        self.params.P.psf_redetect_sigma_g = _spin_to_sigma(self.p_redetect_g)
        self.params.P.psf_redetect_sigma_r = _spin_to_sigma(self.p_redetect_r)
        self.params.P.psf_redetect_sigma_i = _spin_to_sigma(self.p_redetect_i)
        self.params.P.psf_duplicate_radius_fwhm_mult = self.p_dup_mult.value()
        self.params.P.psf_duplicate_radius_px = self.p_dup_px.value() if self.p_dup_px.value() > 0 else np.nan
        self.params.P.psf_new_sources_cap_per_iter = self.p_cap_per_iter.value()
        self.params.P.psf_new_sources_cap_frac = self.p_cap_frac.value()
        self.params.P.psf_fit_init_max_sources = self.p_fit_init_max.value()
        self.params.P.psf_substar_neighbor_r_fwhm_mult = self.p_substar_nei_mult.value()
        self.params.P.psf_substar_max_sources = self.p_substar_max_src.value()
        self.params.P.psf_conv_new_frac = self.p_conv_new.value()
        self.params.P.psf_flux_conv_threshold = self.p_conv_flux.value()
        self.params.P.psf_use_grouper = self.p_use_grouper.isChecked()
        self.params.P.psf_use_error_image = self.p_use_error_img.isChecked()
        self.params.P.psf_redetect_sharp_lo = self.p_sharp_lo.value()
        self.params.P.psf_redetect_sharp_hi = self.p_sharp_hi.value()
        self.params.P.psf_redetect_round_abs_max = self.p_round_max.value()
        self.params.P.psf_save_residuals = self.p_save_residuals.isChecked()
        self.save_state()
        self.persist_params()
        QMessageBox.information(dialog, "Saved", "Parameters saved.")
        dialog.accept()

    # ── Thread cleanup ────────────────────────────────────────────────────────

    def _cleanup_worker(self, timeout_ms=5000):
        if not self.worker:
            return
        if self.worker.isRunning():
            self.worker.stop()
            self.worker.quit()
            self.worker.wait(timeout_ms)
        try:
            self.worker.deleteLater()
        except Exception:
            pass
        self.worker = None

    # ── Log ───────────────────────────────────────────────────────────────────

    def log(self, message: str):
        ts = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{ts}] {message}")

    def show_log_window(self):
        self.log_window.show()
        self.log_window.raise_()
        self.log_window.activateWindow()

    # ── Skip label ────────────────────────────────────────────────────────────

    def _update_skip_label(self):
        if not hasattr(self, "skip_label"):
            return
        if self._skip_psf:
            self.skip_label.setText(
                "PSF SKIPPED — Step 8 RefBuild will use Step 5 aperture results."
            )
        else:
            psf_idx = step6_psf_dir(self.params.P.result_dir) / "photometry_index.csv"
            if psf_idx.exists():
                self.skip_label.setText("PSF photometry results available.")
            else:
                self.skip_label.setText("")

    # ── Validation / State ────────────────────────────────────────────────────

    def validate_step(self) -> bool:
        """Step 6 is always valid: either PSF was run or it was skipped."""
        if self._skip_psf:
            return True
        psf_idx = step6_psf_dir(self.params.P.result_dir) / "photometry_index.csv"
        return psf_idx.exists()

    def save_state(self):
        self.project_state.store_step_data("psf_photometry", {
            "skip_psf": self._skip_psf,
            "psf_model_mode": getattr(self.params.P, "psf_model_mode", "per_frame"),
            "psf_parallel_workers": getattr(self.params.P, "psf_parallel_workers", 0),
            "psf_epsf_oversampling": getattr(self.params.P, "psf_epsf_oversampling", 2),
            "psf_epsf_size_px": getattr(self.params.P, "psf_epsf_size_px", 25),
            "psf_epsf_size_fwhm_mult": getattr(self.params.P, "psf_epsf_size_fwhm_mult", 4.0),
            "psf_n_stars_max": getattr(self.params.P, "psf_n_stars_max", 50),
            "psf_isolation_fwhm_mult": getattr(self.params.P, "psf_isolation_fwhm_mult", 3.0),
            "psf_fit_shape_px": getattr(self.params.P, "psf_fit_shape_px", 5),
            "psf_fit_shape_fwhm_mult": getattr(self.params.P, "psf_fit_shape_fwhm_mult", 1.5),
            "psf_use_grouper": getattr(self.params.P, "psf_use_grouper", True),
            "psf_max_iter": getattr(self.params.P, "psf_max_iter", 2),
            "psf_redetect_sigma": getattr(self.params.P, "psf_redetect_sigma", 4.0),
            "psf_redetect_sigma_g": getattr(self.params.P, "psf_redetect_sigma_g", float("nan")),
            "psf_redetect_sigma_r": getattr(self.params.P, "psf_redetect_sigma_r", float("nan")),
            "psf_redetect_sigma_i": getattr(self.params.P, "psf_redetect_sigma_i", float("nan")),
            "psf_duplicate_radius_fwhm_mult": getattr(self.params.P, "psf_duplicate_radius_fwhm_mult", 0.8),
            "psf_duplicate_radius_px": getattr(self.params.P, "psf_duplicate_radius_px", np.nan),
            "psf_new_sources_cap_per_iter": getattr(self.params.P, "psf_new_sources_cap_per_iter", 70),
            "psf_new_sources_cap_frac": getattr(self.params.P, "psf_new_sources_cap_frac", 0.02),
            "psf_fit_init_max_sources": getattr(self.params.P, "psf_fit_init_max_sources", 0),
            "psf_substar_neighbor_r_fwhm_mult": getattr(self.params.P, "psf_substar_neighbor_r_fwhm_mult", 8.0),
            "psf_substar_max_sources": getattr(self.params.P, "psf_substar_max_sources", 1500),
            "psf_conv_new_frac": getattr(self.params.P, "psf_conv_new_frac", 0.02),
            "psf_flux_conv_threshold": getattr(self.params.P, "psf_flux_conv_threshold", 0.01),
            "psf_use_error_image": getattr(self.params.P, "psf_use_error_image", False),
            "psf_redetect_sharp_lo": getattr(self.params.P, "psf_redetect_sharp_lo", 0.15),
            "psf_redetect_sharp_hi": getattr(self.params.P, "psf_redetect_sharp_hi", 0.95),
            "psf_redetect_round_abs_max": getattr(self.params.P, "psf_redetect_round_abs_max", 0.8),
            "psf_save_residuals": getattr(self.params.P, "psf_save_residuals", True),
        })

    def restore_state(self):
        state = self.project_state.get_step_data("psf_photometry")
        if state:
            self._skip_psf = bool(state.get("skip_psf", False))
            for k, v in state.items():
                if k != "skip_psf" and hasattr(self.params.P, k):
                    setattr(self.params.P, k, v)
        if str(getattr(self.params.P, "psf_model_mode", "per_frame")).strip().lower() != "per_frame":
            self.params.P.psf_model_mode = "per_frame"
        # Clamp fit_shape_fwhm_mult to a sensible minimum (< 1.0 is unusable).
        _fmult = _to_float(getattr(self.params.P, "psf_fit_shape_fwhm_mult", 1.5), 1.5)
        if _fmult < 1.0:
            self.params.P.psf_fit_shape_fwhm_mult = 1.5
        # Migrate legacy defaults to tuned defaults unless user explicitly changed them.
        _rsig = _to_float(getattr(self.params.P, "psf_redetect_sigma", 4.0), 4.0)
        if abs(_rsig - 6.0) < 1e-6 or abs(_rsig - 7.5) < 1e-6:
            # 6.0 and 7.5 were old defaults; migrate to current default.
            self.params.P.psf_redetect_sigma = 4.0
        _cap_abs = _to_int(getattr(self.params.P, "psf_new_sources_cap_per_iter", 70), 70)
        if _cap_abs == 100:
            self.params.P.psf_new_sources_cap_per_iter = 70
        _cap_frac = _to_float(getattr(self.params.P, "psf_new_sources_cap_frac", 0.02), 0.02)
        if abs(_cap_frac - 0.04) < 1e-6:
            self.params.P.psf_new_sources_cap_frac = 0.02
        _slo = _to_float(getattr(self.params.P, "psf_redetect_sharp_lo", 0.15), 0.15)
        _shi = _to_float(getattr(self.params.P, "psf_redetect_sharp_hi", 0.95), 0.95)
        _rnd = _to_float(getattr(self.params.P, "psf_redetect_round_abs_max", 0.8), 0.8)
        if _slo <= -900.0 and _shi >= 900.0:
            self.params.P.psf_redetect_sharp_lo = 0.15
            self.params.P.psf_redetect_sharp_hi = 0.95
        if _rnd >= 9.0:
            self.params.P.psf_redetect_round_abs_max = 0.8
        # Migrate overly-loose legacy UI values.
        if _slo <= 0.01 and _shi >= 0.99 and _rnd >= 1.5:
            self.params.P.psf_redetect_sharp_lo = 0.15
            self.params.P.psf_redetect_sharp_hi = 0.95
            self.params.P.psf_redetect_round_abs_max = 0.8
        self._update_skip_label()
        self.update_frame_table()
        # Reload EPSF + residual images from disk
        self._load_from_disk()

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.stop_psf()
        self._cleanup_worker(timeout_ms=10000)
        super().closeEvent(event)
