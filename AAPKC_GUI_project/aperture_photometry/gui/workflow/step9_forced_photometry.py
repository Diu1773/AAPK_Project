"""
Step 9-Aperture: Forced Aperture Photometry (per-frame ID-matched XY)
Ported from AAPKI_GUI.ipynb Cell 12 (GUI adaptation).

Features:
- Parallel processing with ThreadPoolExecutor
- Real-time per-frame result updates
"""

from __future__ import annotations

import json
import re
import math
import time
import zlib
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.stats import SigmaClip, sigma_clipped_stats

from PyQt5.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QGroupBox, QMessageBox,
    QTextEdit, QDialog, QFormLayout, QDialogButtonBox, QProgressBar,
    QCheckBox, QSpinBox, QDoubleSpinBox, QLineEdit, QWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, QTabWidget,
    QSplitter, QListWidget
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from .step_window_base import StepWindowBase
from .aperture_photometry_worker import ApertureWorker
from ...utils.step_paths import step2_cropped_dir, step5_dir, step6_dir, step7_dir, step9_dir, crop_is_active
from ...utils.constants import get_parallel_workers
from ...utils import normalize_filter_name
from ...utils.io_utils import read_csv_int64_source_id, parse_int64_series


def _as_bool(x, default=False):
    if isinstance(x, bool):
        return x
    if x is None:
        return default
    if isinstance(x, (int, np.integer)):
        return bool(x)
    s = str(x).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    return default


def _safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def _is_up_to_date(out_path, deps):
    try:
        t_out = Path(out_path).stat().st_mtime
        for d in deps:
            if Path(d).stat().st_mtime > t_out:
                return False
        return True
    except Exception:
        return False


def _norm_path_key(path_value) -> str:
    if path_value is None:
        return ""
    try:
        s = str(path_value).strip().replace("\\", "/")
    except Exception:
        s = str(path_value).replace("\\", "/")
    if len(s) >= 3 and s[1] == ":" and s[2] == "/" and s[0].isalpha():
        s = f"/mnt/{s[0].lower()}/{s[3:]}"
    while "//" in s:
        s = s.replace("//", "/")
    if len(s) > 1 and s.endswith("/"):
        s = s[:-1]
    return s.lower()


def _fast_file_crc32(path: Path, sample_bytes: int = 8192) -> int | None:
    """Compute a cheap content fingerprint from head/mid/tail samples."""
    try:
        st = path.stat()
        size = int(st.st_size)
        if size <= 0:
            return None
        n = int(max(1024, sample_bytes))
        crc = 0
        with path.open("rb") as fh:
            head = fh.read(n)
            crc = zlib.crc32(head, crc)
            if size > 2 * n:
                mid_pos = max(0, size // 2 - n // 2)
                fh.seek(mid_pos)
                mid = fh.read(n)
                crc = zlib.crc32(mid, crc)
            if size > n:
                tail_pos = max(0, size - n)
                fh.seek(tail_pos)
                tail = fh.read(n)
                crc = zlib.crc32(tail, crc)
        return int(crc & 0xFFFFFFFF)
    except Exception:
        return None


def _build_source_signature(path: Path, *, use_cropped: bool, include_crc: bool = False) -> dict:
    sig = {
        "cache_schema": 3,
        "source_path": _norm_path_key(path),
        "source_use_cropped": bool(use_cropped),
        "source_size": None,
        "source_mtime_ns": None,
        "source_crc32": None,
    }
    try:
        st = path.stat()
        sig["source_size"] = int(st.st_size)
        sig["source_mtime_ns"] = int(st.st_mtime_ns)
    except Exception:
        pass
    if include_crc:
        sig["source_crc32"] = _fast_file_crc32(path)
    return sig


def _source_signature_compatible(saved_sig: dict, current_sig: dict) -> bool:
    if not isinstance(saved_sig, dict) or not isinstance(current_sig, dict):
        return False
    if bool(saved_sig.get("source_use_cropped", None)) != bool(current_sig.get("source_use_cropped", None)):
        return False
    if _norm_path_key(saved_sig.get("source_path")) != _norm_path_key(current_sig.get("source_path")):
        return False
    try:
        saved_size = int(saved_sig.get("source_size"))
        curr_size = int(current_sig.get("source_size"))
    except Exception:
        return False
    if saved_size <= 0 or curr_size <= 0:
        return False
    if saved_size != curr_size:
        return False

    # Legacy signatures (schema<3) were mtime/size-centric; keep them soft-compatible
    # to avoid one-time cache miss storms and upgrade on first cache hit.
    try:
        saved_schema = int(saved_sig.get("cache_schema", 0) or 0)
    except Exception:
        saved_schema = 0
    if saved_schema < 3:
        return True

    try:
        saved_mtime_ns = int(saved_sig.get("source_mtime_ns"))
        curr_mtime_ns = int(current_sig.get("source_mtime_ns"))
    except Exception:
        saved_mtime_ns = -1
        curr_mtime_ns = -2
    if saved_mtime_ns > 0 and curr_mtime_ns > 0 and saved_mtime_ns == curr_mtime_ns:
        return True

    saved_crc = current_crc = None
    try:
        saved_crc = int(saved_sig.get("source_crc32"))
    except Exception:
        pass
    try:
        current_crc = int(current_sig.get("source_crc32"))
    except Exception:
        pass
    if saved_crc is None or current_crc is None:
        return False
    if saved_crc < 0 or current_crc < 0:
        return False
    return saved_crc == current_crc


def _has_valid_crc32(value) -> bool:
    try:
        return int(value) >= 0
    except Exception:
        return False


def _should_log_cache_detail(done_idx: int, total_frames: int, *, head_n: int, tail_n: int, every_n: int) -> bool:
    if total_frames <= 0:
        return True
    idx = max(int(done_idx), 1)
    head_n = max(int(head_n), 0)
    tail_n = max(int(tail_n), 0)
    every_n = int(every_n)
    if idx <= head_n:
        return True
    if tail_n > 0 and idx > max(total_frames - tail_n, head_n):
        return True
    return bool(every_n > 0 and idx % every_n == 0)


def _get_filter_lower(fits_path: Path):
    try:
        h = fits.getheader(fits_path)
        f = h.get("FILTER", None)
        if f is None:
            return "unknown"
        return normalize_filter_name(f)
    except Exception:
        return "unknown"


def _get_exptime_fallback(fits_path: Path, default=1.0):
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


def _circle_mask(shape, cx, cy, r):
    h, w = shape
    y = np.arange(h)[:, None]
    x = np.arange(w)[None, :]
    return (x - cx) ** 2 + (y - cy) ** 2 <= (r * r)


def _refine_local_centroid(img, x, y, fwhm_used, cbox_scale):
    h, w = img.shape
    r = int(max(cbox_scale * max(float(fwhm_used), 2.0), 6.0))
    xi, yi = int(round(x)), int(round(y))
    x0, x1 = max(0, xi - r), min(w, xi + r + 1)
    y0, y1 = max(0, yi - r), min(h, yi + r + 1)
    if (x1 - x0) < 9 or (y1 - y0) < 9:
        return (x, y)

    cut = img[y0:y1, x0:x1]
    _, med, _ = sigma_clipped_stats(cut, sigma=3.0)
    z = cut - med
    z[~np.isfinite(z)] = 0.0
    z[z < 0] = 0.0
    s = np.nansum(z)
    if s <= 0:
        return (x, y)

    yy, xx = np.mgrid[y0:y1, x0:x1]
    xc = float(np.nansum(xx * z) / s)
    yc = float(np.nansum(yy * z) / s)
    return (xc, yc)


def _phot_one_target(
    img_cut, xc, yc, r_ap, r_in, r_out,
    sigma_clip_val=3.0, maxiters=5,
    gain=1.0, rn_param_e=7.5, sky_frame_e=np.nan,
    sky_sigma_mode="local", sky_sigma_includes_rn=True, min_n_sky_for_local=50,
    sat_adu=60000.0, datamax_adu=None
):
    ap_mask = _circle_mask(img_cut.shape, xc, yc, r_ap)
    ann_out_mask = _circle_mask(img_cut.shape, xc, yc, r_out)
    ann_in_mask = _circle_mask(img_cut.shape, xc, yc, r_in)
    ann_mask = ann_out_mask & (~ann_in_mask)

    ann_vals = img_cut[ann_mask]
    ann_vals = ann_vals[np.isfinite(ann_vals)]
    if ann_vals.size:
        sc = SigmaClip(sigma=sigma_clip_val, maxiters=maxiters)
        vv = sc(ann_vals)
        vv = vv.compressed() if np.ma.isMaskedArray(vv) else np.asarray(vv)
        vv = vv[np.isfinite(vv)]
        bkg_med = float(np.nanmedian(vv)) if vv.size else float(np.nanmedian(ann_vals))
        bkg_std = float(np.nanstd(vv, ddof=1)) if vv.size > 1 else float(np.nanstd(ann_vals, ddof=1)) if ann_vals.size > 1 else 0.0
        n_sky = int(len(vv)) if vv.size else int(len(ann_vals))
    else:
        bkg_med = 0.0
        bkg_std = 0.0
        n_sky = 0

    ap_sum = float(np.nansum(img_cut[ap_mask]))
    ap_area = float(np.count_nonzero(ap_mask))
    flux_net_adu = ap_sum - bkg_med * ap_area
    flux_e = flux_net_adu * gain

    # Background variance per pixel (in electrons^2)
    # The annulus std already includes sky Poisson and read noise.
    # Use a selectable source to avoid double-counting when a frame-level sigma is available.
    sigma_local_e2 = (max(bkg_std, 0.0) * gain) ** 2 if np.isfinite(bkg_std) else np.nan
    sigma_frame_e2 = float(sky_frame_e) ** 2 if np.isfinite(sky_frame_e) else np.nan

    sky_sigma_mode = str(sky_sigma_mode or "local").strip().lower()
    use_local = np.isfinite(sigma_local_e2) and bkg_std > 0 and (n_sky >= min_n_sky_for_local)

    if sky_sigma_mode == "frame":
        sigma_pix_e2 = sigma_frame_e2 if np.isfinite(sigma_frame_e2) else sigma_local_e2
    elif sky_sigma_mode == "max":
        sigma_pix_e2 = np.nanmax([sigma_local_e2, sigma_frame_e2])
    else:
        sigma_pix_e2 = sigma_local_e2 if use_local else sigma_frame_e2

    if not np.isfinite(sigma_pix_e2):
        sigma_pix_e2 = 0.0

    if sky_sigma_includes_rn:
        sigma_pix_e2 = max(sigma_pix_e2 - rn_param_e ** 2, 0.0)

    # Full error propagation formula (CCD equation):
    # σ² = N_star + N_ap*σ_pix² + (N_ap²/n_sky)*σ_pix² + N_ap*RN²
    #
    # Note: Previous version had var_bkg_poisson = ap_area * bkg_e (Poisson from sky)
    # This was DOUBLE-COUNTING because sigma_pix_e2 already includes sky Poisson noise.
    # Removed to fix RMS/pred < 1 and χ²/ν < 1 (error overestimation).
    #
    # Terms:
    # 1. var_source: Source photon Poisson noise
    # 2. var_bkg_in_ap: Background variance contribution to aperture pixels
    # 3. var_bkg_est: Uncertainty in background level estimation (finite annulus)
    # 4. var_readnoise: Read noise per pixel in aperture
    var_source = max(flux_e, 0.0)  # Source Poisson (only if positive)
    var_bkg_in_ap = ap_area * sigma_pix_e2  # Background noise in aperture (Poisson + systematics)
    var_bkg_est = (ap_area ** 2 / max(n_sky, 1)) * sigma_pix_e2  # Background level uncertainty
    var_readnoise = ap_area * rn_param_e ** 2 if sky_sigma_includes_rn else 0.0  # Read noise

    var_e = var_source + var_bkg_in_ap + var_bkg_est + var_readnoise
    sigma_e = float(np.sqrt(max(var_e, 0.0)))
    snr = float(flux_e / sigma_e) if sigma_e > 0 else np.nan
    peak_adu = float(np.nanmax(img_cut[ap_mask])) if ap_area > 0 else np.nan
    is_sat = bool(np.isfinite(peak_adu) and (peak_adu >= float(sat_adu)))
    is_nonlinear = False
    if datamax_adu is not None and np.isfinite(datamax_adu) and float(datamax_adu) > 0:
        is_nonlinear = bool(np.isfinite(peak_adu) and (peak_adu >= float(datamax_adu)))

    return (flux_e, sigma_e, snr, ap_sum, bkg_med, bkg_std, ap_area, n_sky, is_sat, is_nonlinear)


class ForcedPhotometryWorker(QThread):
    """Worker thread for forced photometry with parallel processing."""
    progress = pyqtSignal(int, int, str)
    frame_done = pyqtSignal(str, dict)  # filename, result_dict
    finished = pyqtSignal(dict)
    error = pyqtSignal(str, str)
    log = pyqtSignal(str)

    def __init__(self, file_list, params, data_dir, result_dir, cache_dir, use_cropped=False):
        super().__init__()
        self.file_list = list(file_list)
        self.params = params
        self.data_dir = Path(data_dir)
        self.result_dir = Path(result_dir)
        self.cache_dir = Path(cache_dir)
        self.use_cropped = use_cropped
        self.max_workers = get_parallel_workers(params)  # Central config
        self._stop_requested = False
        self._write_lock = Lock()

    def stop(self):
        self._stop_requested = True

    def _log(self, msg: str):
        self.log.emit(msg)

    def run(self):
        try:
            P = self.params.P
            result_dir = self.result_dir
            output_dir = step9_dir(result_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            cache_dir = self.cache_dir

            master_path = step6_dir(result_dir) / "master_catalog.tsv"
            if not master_path.exists():
                master_path = result_dir / "master_catalog.tsv"
            if not master_path.exists():
                raise RuntimeError("master_catalog.tsv not found")
            master = pd.read_csv(master_path, sep="\t")

            ap_mode = str(getattr(P, "aperture_mode", getattr(P, "ap_mode", "apcorr"))).strip().lower()
            resume_global = bool(getattr(P, "resume_mode", True))
            step9_use_cache = bool(getattr(P, "step9_use_cache", False))
            resume = bool(resume_global and step9_use_cache)
            force_rephot = bool(getattr(P, "force_rephot", False))

            GAIN = float(getattr(P, "gain_e_per_adu", 0.1))
            ZP = float(getattr(P, "zp_initial", 25.0))
            ann_sigma = float(getattr(P, "annulus_sigma_clip", 3.0))
            ann_maxiter = int(getattr(P, "fitsky_max_iter", 5))
            neigh_scale = float(getattr(P, "annulus_neighbor_mask_scale", 1.3))
            cbox_scale = float(getattr(P, "center_cbox_scale", 1.5))
            min_snr_for_mag = float(getattr(P, "min_snr_for_mag", 3.0))
            sat_adu = float(getattr(P, "saturation_adu", 60000.0))
            datamax_adu = float(getattr(P, "datamax_adu", np.nan))
            rn_param_e = float(getattr(P, "rdnoise_e", 1.39))
            bkg_use_segm_mask = bool(getattr(P, "bkg_use_segm_mask", True))
            recenter_aperture = bool(getattr(P, "recenter_aperture", True))
            max_recenter_shift = float(getattr(P, "max_recenter_shift", 2.0))  # Max allowed centroid shift (px)
            centroid_outlier_px = float(getattr(P, "centroid_outlier_px", 1.0))
            sky_sigma_mode = str(getattr(P, "sky_sigma_mode", "local")).strip().lower()
            sky_sigma_includes_rn = bool(getattr(P, "sky_sigma_includes_rn", True))
            min_n_sky_for_local = int(getattr(P, "sky_sigma_min_n_sky", 50))
            cache_log_every_n = max(int(getattr(P, "cache_log_every_n", 20)), 0)
            cache_log_head_n = max(int(getattr(P, "cache_log_head_n", 3)), 0)
            cache_log_tail_n = max(int(getattr(P, "cache_log_tail_n", 2)), 0)

            self._log(
                "Start forced photometry | "
                f"frames={len(self.file_list)} | resume_global={resume_global} | "
                f"step9_use_cache={step9_use_cache} | cache_active={resume} | force_rephot={force_rephot} | "
                f"ap_mode={ap_mode} | ZP={ZP} (ADU/sec) | gain={GAIN} e-/ADU | "
                f"min_snr={min_snr_for_mag} | use_cropped={self.use_cropped}"
            )

            ap_path = output_dir / "aperture_by_frame.csv"
            if not ap_path.exists():
                legacy_ap = result_dir / "aperture_by_frame.csv"
                if legacy_ap.exists():
                    ap_path = legacy_ap
            if not ap_path.exists():
                raise RuntimeError(
                    "aperture_by_frame.csv not found. "
                    f"Looked for: '{output_dir / 'aperture_by_frame.csv'}', '{result_dir / 'aperture_by_frame.csv'}'. "
                    "Run Step9 Apcorr tab first."
                )

            df_ap = pd.read_csv(ap_path)
            required_ap_cols = {"file", "r_ap", "r_in", "r_out"}
            missing_ap_cols = sorted(list(required_ap_cols - set(df_ap.columns)))
            if missing_ap_cols:
                raise RuntimeError(
                    f"aperture_by_frame.csv is invalid: missing columns {missing_ap_cols} "
                    f"(file='{ap_path}')."
                )
            if df_ap.empty:
                raise RuntimeError(
                    f"aperture_by_frame.csv is empty (file='{ap_path}'). "
                    "Run Step9 Apcorr tab again and check Step5/Step7 inputs."
                )

            # Normalize file key to basename for robust matching.
            df_ap = df_ap.copy()
            df_ap["file"] = df_ap["file"].astype(str).map(lambda s: Path(str(s)).name)

            apcorr_path = output_dir / "apcorr_summary.csv"
            if not apcorr_path.exists():
                legacy_apcorr = result_dir / "apcorr_summary.csv"
                if legacy_apcorr.exists():
                    apcorr_path = legacy_apcorr
            if ap_mode in ("apcorr", "auto") and (not apcorr_path.exists()):
                raise RuntimeError(
                    "apcorr_summary.csv not found. "
                    f"Looked for: '{output_dir / 'apcorr_summary.csv'}', '{result_dir / 'apcorr_summary.csv'}'. "
                    "Run Step9 Apcorr tab first."
                )
            apcorr_df = pd.read_csv(apcorr_path) if apcorr_path.exists() else None
            if apcorr_df is not None and (not apcorr_df.empty) and ("file" in apcorr_df.columns):
                apcorr_df = apcorr_df.copy()
                apcorr_df["file"] = apcorr_df["file"].astype(str).map(lambda s: Path(str(s)).name)

            # frame sourceid->ID mapping
            frame_map_path = step7_dir(result_dir) / "frame_sourceid_to_ID.tsv"
            if not frame_map_path.exists():
                frame_map_path = result_dir / "frame_sourceid_to_ID.tsv"
            df_frame_map = pd.read_csv(frame_map_path, sep="\t") if frame_map_path.exists() else None

            sourceid_to_ID_path = step6_dir(result_dir) / "sourceid_to_ID.csv"
            if not sourceid_to_ID_path.exists():
                sourceid_to_ID_path = result_dir / "sourceid_to_ID.csv"
            sid2id = None
            if sourceid_to_ID_path.exists():
                try:
                    df_sid2id = read_csv_int64_source_id(sourceid_to_ID_path)
                    if {"source_id", "ID"} <= set(df_sid2id.columns):
                        sid_series = parse_int64_series(df_sid2id["source_id"])
                        id_series = pd.to_numeric(df_sid2id["ID"], errors="coerce")
                        valid = sid_series.notna() & id_series.notna()
                        sid2id = dict(zip(sid_series.loc[valid].astype("int64"), id_series.loc[valid].astype(int)))
                except Exception:
                    sid2id = None

            # sky sigma
            _sky_df = None
            _sky_src = None
            sky_csv = output_dir / "frame_sky_sigma.csv"
            if not sky_csv.exists():
                legacy_sky = result_dir / "frame_sky_sigma.csv"
                if legacy_sky.exists():
                    sky_csv = legacy_sky
            if sky_csv.exists():
                _sky_df = pd.read_csv(sky_csv)
                _sky_src = "frame_sky_sigma.csv"
            else:
                fq_path = step5_dir(result_dir) / "frame_quality.csv"
                if not fq_path.exists():
                    fq_path = result_dir / "frame_quality.csv"
                if fq_path.exists():
                    _sky_df = pd.read_csv(fq_path)
                _sky_src = "frame_quality.csv"

            def _sky_sigma_for(fname):
                try:
                    if _sky_df is None or _sky_df.empty or "file" not in _sky_df.columns:
                        return np.nan
                    row = _sky_df[_sky_df["file"] == fname]
                    if row.empty:
                        return np.nan
                    for col in ("sky_sigma_med_e", "sky_sigma_e"):
                        if col in row.columns:
                            v = float(row[col].values[0])
                            return v if np.isfinite(v) else np.nan
                    for col in ("sky_sigma_med_adu", "sky_sigma_adu"):
                        if col in row.columns:
                            v = float(row[col].values[0])
                            return (v * GAIN) if np.isfinite(v) else np.nan
                except Exception:
                    return np.nan
                return np.nan

            def _pick_col(cols, cands):
                for c in cands:
                    if c in cols:
                        return c
                return None

            def _load_frame_targets(fname):
                if df_frame_map is not None and (not df_frame_map.empty):
                    cols = df_frame_map.columns
                    c_file = _pick_col(cols, ["file", "fname", "frame"])
                    c_id = _pick_col(cols, ["ID", "id"])
                    c_sid = _pick_col(cols, ["source_id", "sourceid", "sid"])
                    c_x = _pick_col(cols, ["x", "x_det", "x_pix", "x0"])
                    c_y = _pick_col(cols, ["y", "y_det", "y_pix", "y0"])
                    c_sep = _pick_col(cols, ["sep_arcsec", "sep", "dist_arcsec"])
                    if c_file and c_id and c_x and c_y:
                        sub = df_frame_map[df_frame_map[c_file].astype(str) == str(fname)].copy()
                        if len(sub):
                            out = pd.DataFrame({
                                "ID": pd.to_numeric(sub[c_id], errors="coerce").astype("Int64"),
                                "x": pd.to_numeric(sub[c_x], errors="coerce"),
                                "y": pd.to_numeric(sub[c_y], errors="coerce"),
                            })
                            if c_sid:
                                out["source_id"] = parse_int64_series(sub[c_sid])
                            if c_sep:
                                out["sep_arcsec"] = pd.to_numeric(sub[c_sep], errors="coerce")
                            out = out.dropna(subset=["ID", "x", "y"])
                            return out

                # fallback: idmatch CSV + master map
                idmatch_dir = cache_dir / "idmatch"
                p = idmatch_dir / f"idmatch_{fname}.csv"
                if p.exists():
                    try:
                        df = read_csv_int64_source_id(p)
                        if {"source_id", "x", "y"} <= set(df.columns):
                            if sid2id is None:
                                return pd.DataFrame(columns=["ID", "x", "y", "source_id"])
                            df["ID"] = df["source_id"].map(sid2id)
                            df = df.dropna(subset=["ID", "x", "y"])
                            return df[["ID", "x", "y", "source_id"]]
                    except Exception:
                        pass
                return pd.DataFrame(columns=["ID", "x", "y", "source_id"])

            def _det_xy_for(fname):
                det_csv = cache_dir / f"detect_{fname}.csv"
                if det_csv.exists() and det_csv.stat().st_size > 0:
                    try:
                        tmp = pd.read_csv(det_csv)
                        if {"x", "y"} <= set(tmp.columns) and len(tmp):
                            xy = tmp[["x", "y"]].to_numpy(float)
                            xy = xy[np.isfinite(xy).all(axis=1)]
                            return xy
                    except Exception:
                        pass
                return np.zeros((0, 2), float)

            def _cached_counts(fname, out_tsv):
                n = 0
                n_goodmag = 0
                if out_tsv.exists() and out_tsv.stat().st_size > 0:
                    try:
                        df = pd.read_csv(out_tsv, sep="\t")
                        n = int(len(df))
                        if "mag" in df.columns:
                            n_goodmag = int(pd.to_numeric(df["mag"], errors="coerce").notna().sum())
                    except Exception:
                        pass
                try:
                    targets = int(len(_load_frame_targets(fname)))
                except Exception:
                    targets = 0
                n_fail = max(int(targets) - int(n), 0)
                return n, n_goodmag, n_fail, targets

            fail_csv = output_dir / "phot_forced_fail.tsv"
            debug_json = output_dir / "phot_forced_debug.json"
            _fail_rows_all = []
            debug_frames = []
            index_rows = []

            frames = list(self.file_list)
            total = len(frames)
            counters = {"cached": 0, "processed": 0, "no_targets": 0, "no_aperture": 0, "stopped": 0}
            completed_count = [0]  # Use list for mutable in closure
            cache_logs_suppressed = [0]
            cache_sig_upgrades = [0]

            # Per-frame processing function (for parallel execution)
            def process_single_frame(fname):
                if self._stop_requested:
                    dbg_row = dict(file=fname, cached=False, reason="stop_requested")
                    return fname, None, dbg_row, [], "stopped"

                if self.use_cropped:
                    cropped_dir = step2_cropped_dir(result_dir)
                    if not cropped_dir.exists():
                        cropped_dir = result_dir / "cropped"
                    fpath = cropped_dir / fname
                else:
                    fpath = self.data_dir / fname

                out_tsv = output_dir / f"{fname}_photometry.tsv"
                sig_json = output_dir / f"{fname}_photometry.sig.json"
                source_sig = None
                deps = [master_path, ap_path]
                idmatch_path = cache_dir / "idmatch" / f"idmatch_{fname}.csv"
                if idmatch_path.exists():
                    deps.append(idmatch_path)

                this_filter = _get_filter_lower(fpath)

                # Check cache
                cache_ok = False
                saved_sig = None
                saved_schema = 0
                sig_upgraded = False
                need_sig_upgrade = False
                if resume and (not force_rephot) and out_tsv.exists():
                    cache_ok = _is_up_to_date(out_tsv, deps)
                    if cache_ok:
                        if sig_json.exists():
                            try:
                                saved_sig = json.loads(sig_json.read_text(encoding="utf-8"))
                            except Exception:
                                # Keep cache soft (deps-based) for broken signature files.
                                saved_sig = None
                        if isinstance(saved_sig, dict):
                            try:
                                saved_schema = int(saved_sig.get("cache_schema", 0) or 0)
                            except Exception:
                                saved_schema = 0
                            source_sig = _build_source_signature(
                                fpath,
                                use_cropped=bool(self.use_cropped),
                                include_crc=False
                            )
                            cache_ok = _source_signature_compatible(saved_sig, source_sig)
                            # Schema>=3 can fall back to CRC compare only when mtime changed.
                            if (not cache_ok) and saved_schema >= 3 and _has_valid_crc32(saved_sig.get("source_crc32")):
                                source_sig = _build_source_signature(
                                    fpath,
                                    use_cropped=bool(self.use_cropped),
                                    include_crc=True
                                )
                                cache_ok = _source_signature_compatible(saved_sig, source_sig)
                            need_sig_upgrade = cache_ok and (
                                saved_schema < 3 or (not _has_valid_crc32(saved_sig.get("source_crc32")))
                            )
                        else:
                            # Missing/invalid signature: allow deps-based cache and backfill signature.
                            need_sig_upgrade = True
                if cache_ok:
                    if need_sig_upgrade:
                        if source_sig is None or (not _has_valid_crc32(source_sig.get("source_crc32"))):
                            source_sig = _build_source_signature(
                                fpath,
                                use_cropped=bool(self.use_cropped),
                                include_crc=True
                            )
                        try:
                            with self._write_lock:
                                sig_json.write_text(json.dumps(source_sig, ensure_ascii=False, indent=2), encoding="utf-8")
                            sig_upgraded = True
                        except Exception:
                            sig_upgraded = False
                    n, n_goodmag, n_fail, targets = _cached_counts(fname, out_tsv)
                    idx_row = dict(
                        file=fname, filter=this_filter,
                        n="cached", n_goodmag=n_goodmag, n_fail=n_fail,
                        targets=targets, path=str(out_tsv.name)
                    )
                    dbg_row = dict(
                        file=fname, cached=True,
                        n=n, n_goodmag=n_goodmag, n_fail=n_fail, targets=targets,
                        cache_sig_upgraded=sig_upgraded, cache_saved_schema=saved_schema
                    )
                    return fname, idx_row, dbg_row, [], "cached"

                tgt = _load_frame_targets(fname)
                n_tgt = int(len(tgt))
                if n_tgt == 0:
                    idx_row = dict(file=fname, filter=this_filter, n=0, path=str(out_tsv.name))
                    dbg_row = dict(file=fname, cached=False, targets=0, reason="no_targets")
                    return fname, idx_row, dbg_row, [], "no_targets"

                row = df_ap[df_ap["file"].astype(str) == str(fname)]
                if row.empty:
                    dbg_row = dict(file=fname, cached=False, targets=n_tgt, reason="no_aperture_by_frame")
                    return fname, None, dbg_row, [], "no_aperture"

                r_ap_val = float(row["r_ap"].values[0])
                r_in_val = float(row["r_in"].values[0])
                r_out_val = float(row["r_out"].values[0])
                fwhm_used = float(row["fwhm_used"].values[0])

                exptime = _get_exptime_fallback(fpath, default=1.0)
                sky_frame_e = _sky_sigma_for(fname)

                img = fits.getdata(fpath).astype(np.float32)
                h, w = img.shape

                # apcorr
                apply_flag, c_apcorr, rel_sc = (False, np.nan, np.nan)
                if apcorr_df is not None and ap_mode in ("apcorr", "auto"):
                    row_apc = apcorr_df[apcorr_df["file"].astype(str) == str(fname)]
                    if not row_apc.empty:
                        c_apcorr = _safe_float(row_apc["apcorr"].values[0], np.nan) if "apcorr" in row_apc.columns else np.nan
                        rel_sc = _safe_float(row_apc["rel_scatter"].values[0], np.nan) if "rel_scatter" in row_apc.columns else np.nan
                        apply_flag = _as_bool(row_apc["apply"].values[0], False) if "apply" in row_apc.columns else False

                rows = []
                frame_fail_rows = []
                n_goodmag = 0
                n_fail = 0

                id2sid = {}
                if "source_id" in tgt.columns:
                    id2sid = dict(zip(tgt["ID"].astype(int), tgt["source_id"].astype("Int64")))

                for _, tr in tgt.iterrows():
                    if self._stop_requested:
                        dbg_row = dict(
                            file=fname, cached=False,
                            targets=n_tgt, out_rows=len(rows),
                            n_goodmag=n_goodmag, n_fail=n_fail,
                            apcorr_applied=bool(apply_flag),
                            reason="stop_requested"
                        )
                        return fname, None, dbg_row, frame_fail_rows, "stopped"
                    ID = int(tr["ID"])
                    x0 = float(tr["x"])
                    y0 = float(tr["y"])
                    if not (np.isfinite(x0) and np.isfinite(y0)):
                        n_fail += 1
                        frame_fail_rows.append(dict(file=fname, ID=ID, reason="bad_xy"))
                        continue

                    xc, yc = (x0, y0)
                    recenter_capped = False
                    delta_r = 0.0
                    if recenter_aperture:
                        xc_new, yc_new = _refine_local_centroid(img, x0, y0, fwhm_used, cbox_scale)
                        delta_r = np.sqrt((xc_new - x0)**2 + (yc_new - y0)**2)
                        if delta_r > max_recenter_shift:
                            recenter_capped = True
                        else:
                            xc, yc = xc_new, yc_new

                    pad = int(max(r_out_val + 5, 10))
                    xi, yi = int(round(xc)), int(round(yc))
                    x0c, x1c = max(0, xi - pad), min(w, xi + pad + 1)
                    y0c, y1c = max(0, yi - pad), min(h, yi + pad + 1)
                    cut = img[y0c:y1c, x0c:x1c]
                    xc_cut = xc - x0c
                    yc_cut = yc - y0c

                    (flux_e, sigma_e, snr, ap_sum_adu, bkg_med_adu, bkg_std_adu,
                     ap_area, n_sky, is_sat, is_nonlinear) = _phot_one_target(
                        cut, xc_cut, yc_cut, r_ap_val, r_in_val, r_out_val,
                        sigma_clip_val=ann_sigma, maxiters=ann_maxiter,
                        gain=GAIN, rn_param_e=rn_param_e, sky_frame_e=sky_frame_e,
                        sky_sigma_mode=sky_sigma_mode, sky_sigma_includes_rn=sky_sigma_includes_rn,
                        min_n_sky_for_local=min_n_sky_for_local,
                        sat_adu=sat_adu, datamax_adu=datamax_adu
                    )

                    flux_corr_e = flux_e
                    snr_corr = snr
                    if apply_flag and np.isfinite(c_apcorr) and c_apcorr > 0:
                        flux_corr_e = flux_e * c_apcorr
                        sigma_corr_e = sigma_e * c_apcorr
                        snr_corr = float(flux_corr_e / sigma_corr_e) if sigma_corr_e > 0 else snr

                    safe_gain = max(GAIN, 1e-12)
                    flux_net_adu = flux_e / safe_gain
                    flux_corr_adu = flux_corr_e / safe_gain
                    sigma_adu = sigma_e / safe_gain
                    rate_adu = flux_corr_adu / max(exptime, 1e-9)
                    sigma_rate_adu = sigma_adu / max(exptime, 1e-9)
                    bad_signal = bool(is_sat or is_nonlinear)
                    if (not bad_signal) and snr >= min_snr_for_mag and flux_corr_adu > 0:
                        mag = float(-2.5 * np.log10(flux_corr_adu / max(exptime, 1e-9)) + ZP)
                        mag_err = float(1.0857 / max(snr_corr, 1e-9))
                        n_goodmag += 1
                    else:
                        mag = np.nan
                        mag_err = np.nan

                    centroid_outlier = bool(delta_r > centroid_outlier_px) if np.isfinite(delta_r) else False
                    rows.append(dict(
                        ID=ID, source_id=int(id2sid.get(ID, -1)),
                        x_init=x0, y_init=y0, xcenter=xc, ycenter=yc, FILTER=this_filter,
                        delta_r=delta_r, recenter_capped=recenter_capped, centroid_outlier=centroid_outlier,
                        r_ap_px=r_ap_val, r_in_px=r_in_val, r_out_px=r_out_val,
                        ap_sum_adu=ap_sum_adu, bkg_median_adu=bkg_med_adu, bkg_std_adu=bkg_std_adu,
                        n_sky=n_sky, rdnoise_frame_e=rn_param_e, sky_frame_sigma_e=sky_frame_e,
                        flux_net_adu=flux_net_adu, flux_corr_adu=flux_corr_adu, snr=snr,
                        rate_adu_s=rate_adu, rate_err_adu_s=sigma_rate_adu,
                        mag=mag, mag_err=mag_err,
                        apcorr_applied=bool(apply_flag), apcorr=c_apcorr,
                        is_saturated=is_sat, is_nonlinear=is_nonlinear
                    ))

                df_out = pd.DataFrame(rows)
                if source_sig is None or (not _has_valid_crc32(source_sig.get("source_crc32"))):
                    source_sig = _build_source_signature(
                        fpath,
                        use_cropped=bool(self.use_cropped),
                        include_crc=True
                    )
                with self._write_lock:
                    df_out.to_csv(out_tsv, sep="\t", index=False, na_rep="NaN")
                    try:
                        sig_json.write_text(json.dumps(source_sig, ensure_ascii=False, indent=2), encoding="utf-8")
                    except Exception:
                        pass

                idx_row = dict(
                    file=fname, filter=this_filter,
                    n=len(df_out), n_goodmag=n_goodmag, n_fail=n_fail,
                    targets=n_tgt, path=str(out_tsv.name)
                )
                dbg_row = dict(
                    file=fname, cached=False,
                    targets=n_tgt, out_rows=len(df_out),
                    n_goodmag=n_goodmag, n_fail=n_fail,
                    apcorr_applied=bool(apply_flag),
                    apcorr=float(c_apcorr) if np.isfinite(c_apcorr) else None,
                    rel_scatter=float(rel_sc) if np.isfinite(rel_sc) else None,
                    sky_sigma_source=_sky_src, sky_frame_e=float(sky_frame_e) if np.isfinite(sky_frame_e) else None
                )
                return fname, idx_row, dbg_row, frame_fail_rows, "processed"

            # Parallel execution
            self._log(f"Starting parallel photometry with {self.max_workers} workers...")

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_fname = {executor.submit(process_single_frame, f): f for f in frames}

                for future in as_completed(future_to_fname):
                    try:
                        fname, idx_row, dbg_row, fail_rows, status = future.result()

                        # Update counters
                        if status == "cached":
                            counters["cached"] += 1
                        elif status == "processed":
                            counters["processed"] += 1
                        elif status == "no_targets":
                            counters["no_targets"] += 1
                        elif status == "no_aperture":
                            counters["no_aperture"] += 1
                        elif status == "stopped":
                            counters["stopped"] += 1

                        # Collect results
                        if idx_row:
                            index_rows.append(idx_row)
                        if dbg_row:
                            debug_frames.append(dbg_row)
                            if bool(dbg_row.get("cache_sig_upgraded", False)):
                                cache_sig_upgrades[0] += 1
                        if fail_rows:
                            _fail_rows_all.extend(fail_rows)

                        # Emit per-frame result for GUI update
                        completed_count[0] += 1
                        self.progress.emit(completed_count[0], total, fname)

                        # Emit frame_done for real-time table update
                        if idx_row:
                            self.frame_done.emit(fname, {
                                "file": fname,
                                "filter": idx_row.get("filter", ""),
                                "n": idx_row.get("n", 0),
                                "n_goodmag": idx_row.get("n_goodmag", 0),
                                "n_fail": idx_row.get("n_fail", 0),
                                "targets": idx_row.get("targets", 0),
                                "status": status,
                            })

                        # Detailed per-frame log
                        filt = str((idx_row or {}).get("filter", "unknown"))
                        if status == "processed" and idx_row:
                            n_rows = int(_safe_float(idx_row.get("n", 0), 0.0))
                            n_good = int(_safe_float(idx_row.get("n_goodmag", 0), 0.0))
                            n_fail = int(_safe_float(idx_row.get("n_fail", 0), 0.0))
                            n_tgt = int(_safe_float(idx_row.get("targets", 0), 0.0))
                            ap_applied = bool((dbg_row or {}).get("apcorr_applied", False))
                            apcorr_val = _safe_float((dbg_row or {}).get("apcorr", np.nan), np.nan)
                            rel_sc = _safe_float((dbg_row or {}).get("rel_scatter", np.nan), np.nan)
                            ap_label = "on" if ap_applied else "off"
                            if np.isfinite(apcorr_val):
                                ap_label += f" c={apcorr_val:.4f}"
                            if np.isfinite(rel_sc):
                                ap_label += f" rs={rel_sc:.4f}"
                            self._log(
                                f"[Frame {completed_count[0]}/{total}] OK {fname} | "
                                f"f={filt} targets={n_tgt} rows={n_rows} good={n_good} fail={n_fail} "
                                f"apcorr={ap_label}"
                            )
                        elif status == "cached" and idx_row:
                            n_good = int(_safe_float(idx_row.get("n_goodmag", 0), 0.0))
                            n_tgt = int(_safe_float(idx_row.get("targets", 0), 0.0))
                            if _should_log_cache_detail(
                                completed_count[0],
                                total,
                                head_n=cache_log_head_n,
                                tail_n=cache_log_tail_n,
                                every_n=cache_log_every_n,
                            ):
                                self._log(
                                    f"[Frame {completed_count[0]}/{total}] CACHE {fname} | "
                                    f"f={filt} targets={n_tgt} good={n_good}"
                                )
                            else:
                                cache_logs_suppressed[0] += 1
                        elif status == "no_targets":
                            self._log(
                                f"[Frame {completed_count[0]}/{total}] SKIP {fname} | "
                                "reason=no_targets (Step7 mapping empty)"
                            )
                        elif status == "no_aperture":
                            self._log(
                                f"[Frame {completed_count[0]}/{total}] SKIP {fname} | "
                                "reason=no_aperture_by_frame (run Apcorr first)"
                            )
                        elif status == "stopped":
                            self._log(
                                f"[Frame {completed_count[0]}/{total}] STOP {fname} | "
                                "stop requested"
                            )

                    except Exception as e:
                        self._log(f"Error processing {future_to_fname[future]}: {e}")

            if _fail_rows_all:
                pd.DataFrame(_fail_rows_all).to_csv(fail_csv, sep="\t", index=False, encoding="utf-8-sig")

            idx_path = output_dir / "photometry_index.csv"
            pd.DataFrame(index_rows).to_csv(idx_path, index=False)

            debug_payload = {
                "cell": 12,
                "frames": len(frames),
                "master_ids": int(len(master)),
                "ap_mode": ap_mode,
                "params": {
                    "GAIN": GAIN, "ZP": ZP,
                    "ann_sigma": ann_sigma, "ann_maxiter": ann_maxiter,
                    "neigh_scale": neigh_scale, "CBOX_SCALE": cbox_scale,
                    "min_snr_for_mag": min_snr_for_mag,
                    "sat_adu": sat_adu, "rdnoise_e": rn_param_e,
                    "bkg_use_segm_mask": bool(bkg_use_segm_mask),
                    "recenter_aperture": bool(recenter_aperture),
                    "max_recenter_shift": max_recenter_shift,
                    "RESUME_GLOBAL": bool(resume_global),
                    "STEP9_USE_CACHE": bool(step9_use_cache),
                    "RESUME_ACTIVE": bool(resume),
                    "FORCE_REPHOT": bool(force_rephot),
                    "cache_log_every_n": cache_log_every_n,
                    "cache_log_head_n": cache_log_head_n,
                    "cache_log_tail_n": cache_log_tail_n,
                },
                "inputs": {
                    "master_catalog": str(master_path.name),
                    "frame_sourceid_to_ID": str(frame_map_path.name) if frame_map_path.exists() else None,
                    "idmatch_dir": "idmatch",
                    "apcorr_summary": str(apcorr_path.name) if apcorr_path.exists() else None,
                },
                "per_frame": debug_frames,
            }
            debug_json.write_text(json.dumps(debug_payload, indent=2, ensure_ascii=False), encoding="utf-8")

            if cache_logs_suppressed[0] > 0:
                self._log(
                    "CACHE detail logs sampled | "
                    f"first={cache_log_head_n} last={cache_log_tail_n} every={cache_log_every_n} | "
                    f"suppressed={cache_logs_suppressed[0]}"
                )
            if cache_sig_upgrades[0] > 0:
                self._log(f"CACHE signatures upgraded={cache_sig_upgrades[0]}")

            self._log(
                "Done | "
                f"processed={counters['processed']} | cached={counters['cached']} | "
                f"no_targets={counters['no_targets']} | no_aperture={counters['no_aperture']} | "
                f"stopped={counters['stopped']} | "
                f"fail_rows={len(_fail_rows_all)}"
            )
            self._log(f"Saved: {idx_path.name}, {debug_json.name}")

            self.finished.emit({
                "frames": len(frames),
                "processed": counters["processed"],
                "cached": counters["cached"],
                "no_targets": counters["no_targets"],
                "no_aperture": counters["no_aperture"],
                "stopped": counters["stopped"],
                "fail_rows": len(_fail_rows_all),
            })
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            self.error.emit("WORKER", error_msg)
            self.finished.emit({})


class ForcedPhotometryWindow(StepWindowBase):
    """Step 9-Aperture: Forced Aperture Photometry"""

    def __init__(self, params, file_manager, project_state, main_window):
        self.file_manager = file_manager
        self.worker = None
        self.apcorr_worker = None
        self._pending_phot_after_apcorr = False
        self.file_list = []
        self.use_cropped = False
        self.log_window = None
        self._apcorr_summary_df = pd.DataFrame()
        self._growth_curve_df = pd.DataFrame()
        self._apcorr_filter_by_file = {}
        self._filter_cache = {}

        super().__init__(
            step_index=8,
            step_name="Aperture Photometry",
            params=params,
            project_state=project_state,
            main_window=main_window
        )

        self.setup_step_ui()
        self.restore_state()

    def setup_step_ui(self):
        info = QLabel("Forced aperture photometry per frame based on ID-matched positions.")
        info.setStyleSheet("QLabel { background-color: #E3F2FD; padding: 10px; border-radius: 5px; }")
        self.content_layout.addWidget(info)

        progress_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        self.progress_label = QLabel("Ready")
        self.progress_label.setMinimumWidth(350)
        progress_layout.addWidget(self.progress_label)
        self.content_layout.addLayout(progress_layout)

        self.main_tabs = QTabWidget()
        self.content_layout.addWidget(self.main_tabs)

        # Photometry tab
        summary_tab = QWidget()
        summary_layout = QVBoxLayout(summary_tab)
        phot_control = QHBoxLayout()
        self.btn_phot_params = QPushButton("Photometry Parameters")
        self.btn_phot_params.setStyleSheet("QPushButton { background-color: #9C27B0; color: white; font-weight: bold; padding: 8px 15px; }")
        self.btn_phot_params.clicked.connect(self.open_parameters_dialog)
        phot_control.addWidget(self.btn_phot_params)
        phot_control.addStretch()
        self.btn_run_phot = QPushButton("Run Photometry")
        self.btn_run_phot.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px 20px; }")
        self.btn_run_phot.clicked.connect(self.run_photometry)
        phot_control.addWidget(self.btn_run_phot)
        self.btn_stop_phot = QPushButton("Stop")
        self.btn_stop_phot.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; padding: 8px 15px; }")
        self.btn_stop_phot.clicked.connect(self.stop_photometry)
        self.btn_stop_phot.setEnabled(False)
        phot_control.addWidget(self.btn_stop_phot)
        self.btn_log_phot = QPushButton("Log")
        self.btn_log_phot.setStyleSheet("QPushButton { background-color: #607D8B; color: white; font-weight: bold; padding: 8px 15px; }")
        self.btn_log_phot.clicked.connect(self.show_log_window)
        phot_control.addWidget(self.btn_log_phot)
        summary_layout.addLayout(phot_control)

        table_group = QGroupBox("Per-Frame Photometry Summary")
        table_layout = QVBoxLayout(table_group)
        self.frame_table = QTableWidget()
        self.frame_table.setColumnCount(6)
        self.frame_table.setHorizontalHeaderLabels(
            ["Frame", "Filter", "N_rows", "N_goodmag", "N_fail", "Targets"]
        )
        self.frame_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.frame_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.frame_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.frame_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.frame_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.frame_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self.frame_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        table_layout.addWidget(self.frame_table)
        summary_layout.addWidget(table_group)

        # Apcorr tab — growth curve based optimal aperture finder
        apcorr_tab = QWidget()
        apcorr_layout = QVBoxLayout(apcorr_tab)
        apcorr_control_run = QHBoxLayout()
        self.btn_apcorr_params = QPushButton("Apcorr Parameters")
        self.btn_apcorr_params.setStyleSheet("QPushButton { background-color: #795548; color: white; font-weight: bold; padding: 8px 15px; }")
        self.btn_apcorr_params.clicked.connect(self.open_parameters_dialog)
        apcorr_control_run.addWidget(self.btn_apcorr_params)
        apcorr_control_run.addStretch()
        self.btn_run_apcorr = QPushButton("Run Apcorr")
        self.btn_run_apcorr.setStyleSheet("QPushButton { background-color: #2E7D32; color: white; font-weight: bold; padding: 8px 20px; }")
        self.btn_run_apcorr.clicked.connect(self.run_apcorr)
        apcorr_control_run.addWidget(self.btn_run_apcorr)
        self.btn_stop_apcorr = QPushButton("Stop")
        self.btn_stop_apcorr.setStyleSheet("QPushButton { background-color: #c62828; color: white; font-weight: bold; padding: 8px 15px; }")
        self.btn_stop_apcorr.clicked.connect(self.stop_apcorr)
        self.btn_stop_apcorr.setEnabled(False)
        apcorr_control_run.addWidget(self.btn_stop_apcorr)
        self.btn_log_apcorr = QPushButton("Log")
        self.btn_log_apcorr.setStyleSheet("QPushButton { background-color: #607D8B; color: white; font-weight: bold; padding: 8px 15px; }")
        self.btn_log_apcorr.clicked.connect(self.show_log_window)
        apcorr_control_run.addWidget(self.btn_log_apcorr)
        apcorr_layout.addLayout(apcorr_control_run)

        apcorr_note = QLabel(
            "Growth curve: finds optimal aperture (min error) and computes aperture correction.\n"
            "n_stars = valid stars at each radius, summary n_stars = valid count at optimal radius."
        )
        apcorr_note.setStyleSheet("QLabel { background-color: #FFF8E1; padding: 8px; border-radius: 4px; }")
        apcorr_layout.addWidget(apcorr_note)

        apcorr_control = QHBoxLayout()
        self.apcorr_status = QLabel("growth curve: not loaded")
        apcorr_control.addWidget(self.apcorr_status)
        apcorr_control.addStretch()
        self.btn_refresh_apcorr = QPushButton("Refresh")
        self.btn_refresh_apcorr.setStyleSheet("QPushButton { background-color: #607D8B; color: white; font-weight: bold; padding: 6px 12px; }")
        self.btn_refresh_apcorr.clicked.connect(self.update_apcorr_tables)
        apcorr_control.addWidget(self.btn_refresh_apcorr)
        apcorr_layout.addLayout(apcorr_control)

        # Splitter: left=dual-subplot plot, right=frame list + per-frame table
        self.apcorr_splitter = QSplitter(Qt.Horizontal)

        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        self.apcorr_fig = Figure(figsize=(6.4, 5.6))
        self.apcorr_canvas = FigureCanvas(self.apcorr_fig)
        self.apcorr_ax_mag = self.apcorr_fig.add_subplot(211)   # growth curve (mag)
        self.apcorr_ax_err = self.apcorr_fig.add_subplot(212)   # error U-shape
        self.apcorr_fig.subplots_adjust(hspace=0.35)
        plot_layout.addWidget(self.apcorr_canvas)
        self.apcorr_splitter.addWidget(plot_panel)

        side_panel = QWidget()
        side_layout = QVBoxLayout(side_panel)
        side_layout.addWidget(QLabel("Frames"))
        self.apcorr_file_list = QListWidget()
        self.apcorr_file_list.currentTextChanged.connect(self._on_apcorr_file_changed)
        side_layout.addWidget(self.apcorr_file_list, stretch=1)

        side_layout.addWidget(QLabel("Growth Curve (selected frame)"))
        self.apcorr_file_cand_table = QTableWidget()
        self.apcorr_file_cand_table.setColumnCount(7)
        self.apcorr_file_cand_table.setHorizontalHeaderLabels([
            "scale", "r (px)", "med_mag", "med_mag_err",
            "med_SNR", "n_stars(valid)", "selected"
        ])
        for col in range(7):
            mode = QHeaderView.Stretch if col in (0, 1) else QHeaderView.ResizeToContents
            self.apcorr_file_cand_table.horizontalHeader().setSectionResizeMode(col, mode)
        self.apcorr_file_cand_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        side_layout.addWidget(self.apcorr_file_cand_table, stretch=2)
        self.apcorr_splitter.addWidget(side_panel)

        self.apcorr_splitter.setStretchFactor(0, 3)
        self.apcorr_splitter.setStretchFactor(1, 2)
        apcorr_layout.addWidget(self.apcorr_splitter, stretch=1)

        # Summary table: one row per frame
        apcorr_sel_group = QGroupBox("Growth Curve Summary")
        apcorr_sel_layout = QVBoxLayout(apcorr_sel_group)
        self.apcorr_table = QTableWidget()
        self.apcorr_table.setColumnCount(9)
        self.apcorr_table.setHorizontalHeaderLabels([
            "Frame", "FWHM(px)", "Opt r(px)", "Opt scale",
            "Apcorr", "SNR_opt", "mag_err_opt", "n_stars(opt)", "apply"
        ])
        self.apcorr_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for col in range(1, 9):
            self.apcorr_table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeToContents)
        self.apcorr_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        apcorr_sel_layout.addWidget(self.apcorr_table)
        apcorr_layout.addWidget(apcorr_sel_group)
        self.main_tabs.addTab(apcorr_tab, "Apcorr QC")
        self.main_tabs.addTab(summary_tab, "Photometry")

        # Overlay tab (moved from old standalone overlay step)
        overlay_tab = QWidget()
        overlay_layout = QVBoxLayout(overlay_tab)
        try:
            from .aperture_overlay_panel import ApertureOverlayWindow

            self.overlay_panel = ApertureOverlayWindow(
                self.params, self.file_manager, self.project_state, self.main_window
            )
            # Embed old step window as a tab panel: hide step header/nav only.
            for attr in ("title_label", "btn_previous", "btn_complete", "btn_next"):
                w = getattr(self.overlay_panel, attr, None)
                if w is not None:
                    w.hide()
            overlay_layout.addWidget(self.overlay_panel)
        except Exception as e:
            self.overlay_panel = None
            warn = QLabel(
                "Overlay panel failed to load.\n"
                f"{type(e).__name__}: {e}"
            )
            warn.setStyleSheet("QLabel { color: #b71c1c; padding: 10px; }")
            overlay_layout.addWidget(warn)

        self.main_tabs.addTab(overlay_tab, "Overlay")
        self.main_tabs.setCurrentIndex(0)

        self.log_window = QWidget(self, Qt.Window)
        self.log_window.setWindowTitle("Photometry Log")
        self.log_window.resize(800, 400)
        log_layout = QVBoxLayout(self.log_window)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("QTextEdit { font-family: monospace; font-size: 9pt; }")
        log_layout.addWidget(self.log_text)

        self.populate_file_list()
        self.update_frame_table()
        self.update_apcorr_tables()

    def log(self, message: str):
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")

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
            cropped_dir = legacy_cropped
        else:
            if not self.file_manager.filenames:
                try:
                    self.file_manager.scan_files()
                except Exception:
                    pass
            files = self.file_manager.filenames
            self.use_cropped = False
        self.file_list = list(files)

    def _step9_artifact_path(self, filename: str) -> Path:
        p = step9_dir(self.params.P.result_dir) / filename
        if p.exists():
            return p
        legacy = self.params.P.result_dir / filename
        if legacy.exists():
            return legacy
        return p

    def _apcorr_prereq_ok(self):
        ap_path = self._step9_artifact_path("aperture_by_frame.csv")
        if not ap_path.exists():
            return False, f"Missing aperture file: {ap_path}"

        try:
            df_ap = pd.read_csv(ap_path)
        except Exception as e:
            return False, f"Failed to read aperture file: {ap_path} ({e})"

        required_cols = {"file", "r_ap", "r_in", "r_out"}
        missing_cols = sorted(list(required_cols - set(df_ap.columns)))
        if missing_cols:
            return False, f"Invalid aperture file (missing {missing_cols}): {ap_path}"
        if df_ap.empty:
            return False, f"Aperture file is empty: {ap_path}"

        if self.file_list:
            want = set(str(f) for f in self.file_list)
            have = set(df_ap["file"].astype(str).map(lambda s: Path(str(s)).name))
            if not (want & have):
                return False, (
                    f"Aperture file has no matching rows for current frames "
                    f"(need {len(want)} frames, overlap=0): {ap_path}"
                )

        ap_mode = str(getattr(self.params.P, "aperture_mode", getattr(self.params.P, "ap_mode", "apcorr"))).strip().lower()
        if ap_mode in ("apcorr", "auto"):
            apcorr_path = self._step9_artifact_path("apcorr_summary.csv")
            if not apcorr_path.exists():
                return False, f"Missing apcorr summary: {apcorr_path}"
            try:
                df_apc = pd.read_csv(apcorr_path)
            except Exception as e:
                return False, f"Failed to read apcorr summary: {apcorr_path} ({e})"
            if df_apc.empty:
                return False, f"Apcorr summary is empty: {apcorr_path}"
            if "file" in df_apc.columns and self.file_list:
                want = set(str(f) for f in self.file_list)
                have = set(df_apc["file"].astype(str).map(lambda s: Path(str(s)).name))
                if not (want & have):
                    return False, (
                        f"Apcorr summary has no matching rows for current frames "
                        f"(need {len(want)} frames, overlap=0): {apcorr_path}"
                    )

        return True, ""

    def open_parameters_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Photometry Parameters")
        dialog.resize(480, 520)
        layout = QVBoxLayout(dialog)
        form = QFormLayout()

        form.addRow(QLabel("── Basic photometry ────────────────────"))

        self.param_zp = QDoubleSpinBox()
        self.param_zp.setRange(10.0, 40.0)
        self.param_zp.setValue(float(getattr(self.params.P, "zp_initial", 25.0)))
        form.addRow("Zero Point:", self.param_zp)

        self.param_snr = QDoubleSpinBox()
        self.param_snr.setRange(0.5, 20.0)
        self.param_snr.setValue(float(getattr(self.params.P, "min_snr_for_mag", 3.0)))
        form.addRow("Min SNR:", self.param_snr)

        self.param_recenter = QCheckBox("Enable")
        self.param_recenter.setChecked(bool(getattr(self.params.P, "recenter_aperture", True)))
        form.addRow("Recenter Aperture:", self.param_recenter)

        self.param_ap_mode = QLineEdit()
        self.param_ap_mode.setText(str(getattr(self.params.P, "aperture_mode", "apcorr")))
        form.addRow("Aperture Mode:", self.param_ap_mode)

        form.addRow(QLabel("── Execution / cache ───────────────────"))

        self.param_force = QCheckBox("Force re-phot")
        self.param_force.setChecked(bool(getattr(self.params.P, "force_rephot", False)))
        form.addRow("Force:", self.param_force)
        self.param_step9_cache = QCheckBox("Enable Step9 cache")
        self.param_step9_cache.setChecked(bool(getattr(self.params.P, "step9_use_cache", False)))
        form.addRow("Step9 Cache:", self.param_step9_cache)

        layout.addLayout(form)

        # Aperture/annulus scales
        scale_group = QGroupBox("Aperture/Annulus Scales")
        scale_form = QFormLayout(scale_group)

        self.param_ap_scale = QDoubleSpinBox()
        self.param_ap_scale.setRange(0.5, 5.0)
        self.param_ap_scale.setSingleStep(0.1)
        self.param_ap_scale.setValue(float(getattr(self.params.P, "phot_aperture_scale", 1.0)))
        scale_form.addRow("Aperture scale (×FWHM):", self.param_ap_scale)

        self.param_ann_in = QDoubleSpinBox()
        self.param_ann_in.setRange(1.0, 10.0)
        self.param_ann_in.setSingleStep(0.5)
        self.param_ann_in.setValue(float(getattr(self.params.P, "fitsky_annulus_scale", 4.0)))
        scale_form.addRow("Annulus inner scale (×FWHM):", self.param_ann_in)

        self.param_ann_out = QDoubleSpinBox()
        self.param_ann_out.setRange(0.5, 10.0)
        self.param_ann_out.setSingleStep(0.5)
        self.param_ann_out.setValue(float(getattr(self.params.P, "fitsky_dannulus_scale", 2.0)))
        scale_form.addRow("Annulus width scale (×FWHM):", self.param_ann_out)

        self.param_ann_gap = QDoubleSpinBox()
        self.param_ann_gap.setRange(0.0, 50.0)
        self.param_ann_gap.setSingleStep(0.5)
        self.param_ann_gap.setValue(float(getattr(self.params.P, "annulus_min_gap_px", 6.0)))
        scale_form.addRow("Min annulus gap (px):", self.param_ann_gap)

        self.param_ann_minw = QDoubleSpinBox()
        self.param_ann_minw.setRange(0.0, 100.0)
        self.param_ann_minw.setSingleStep(0.5)
        self.param_ann_minw.setValue(float(getattr(self.params.P, "annulus_min_width_px", 12.0)))
        scale_form.addRow("Min annulus width (px):", self.param_ann_minw)

        self.param_sigma_clip = QDoubleSpinBox()
        self.param_sigma_clip.setRange(0.5, 10.0)
        self.param_sigma_clip.setSingleStep(0.1)
        self.param_sigma_clip.setValue(float(getattr(self.params.P, "annulus_sigma_clip", 3.0)))
        scale_form.addRow("Annulus sigma clip:", self.param_sigma_clip)

        layout.addWidget(scale_group)

        apcorr_group = QGroupBox("Apcorr Gate")
        apcorr_form = QFormLayout(apcorr_group)
        self.param_apcorr_apply = QCheckBox("Enable")
        self.param_apcorr_apply.setChecked(bool(getattr(self.params.P, "apcorr_apply", True)))
        apcorr_form.addRow("Apply Apcorr:", self.param_apcorr_apply)
        self.param_apcorr_min_n = QSpinBox()
        self.param_apcorr_min_n.setRange(3, 500)
        self.param_apcorr_min_n.setValue(int(getattr(self.params.P, "apcorr_use_min_n", 20)))
        apcorr_form.addRow("Apcorr Min N:", self.param_apcorr_min_n)
        self.param_apcorr_scatter = QDoubleSpinBox()
        self.param_apcorr_scatter.setRange(0.01, 1.0)
        self.param_apcorr_scatter.setSingleStep(0.01)
        self.param_apcorr_scatter.setValue(float(getattr(self.params.P, "apcorr_scatter_max", 0.05)))
        apcorr_form.addRow("Apcorr Scatter Max:", self.param_apcorr_scatter)
        layout.addWidget(apcorr_group)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(lambda: self.save_parameters(dialog))
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec_()

    def save_parameters(self, dialog):
        self.params.P.zp_initial = self.param_zp.value()
        self.params.P.min_snr_for_mag = self.param_snr.value()
        self.params.P.recenter_aperture = self.param_recenter.isChecked()
        self.params.P.aperture_mode = self.param_ap_mode.text().strip()
        self.params.P.force_rephot = self.param_force.isChecked()
        self.params.P.step9_use_cache = self.param_step9_cache.isChecked()
        self.params.P.phot_aperture_scale = self.param_ap_scale.value()
        self.params.P.fitsky_annulus_scale = self.param_ann_in.value()
        self.params.P.fitsky_dannulus_scale = self.param_ann_out.value()
        self.params.P.annulus_min_gap_px = self.param_ann_gap.value()
        self.params.P.annulus_min_width_px = self.param_ann_minw.value()
        self.params.P.annulus_sigma_clip = self.param_sigma_clip.value()
        self.params.P.apcorr_apply = self.param_apcorr_apply.isChecked()
        self.params.P.apcorr_use_min_n = self.param_apcorr_min_n.value()
        self.params.P.apcorr_scatter_max = self.param_apcorr_scatter.value()
        self.save_state()
        saved = self.persist_params()
        msg = "Parameters saved to TOML." if saved else "Parameters saved (TOML save failed)."
        QMessageBox.information(dialog, "Success", msg)
        dialog.accept()

    def run_photometry(self):
        if not self.file_list:
            QMessageBox.warning(self, "Warning", "No files to process")
            return
        if self.apcorr_worker and self.apcorr_worker.isRunning():
            QMessageBox.warning(self, "Busy", "Apcorr is running. Stop/finish Apcorr first.")
            return
        if self.worker and self.worker.isRunning():
            return

        ok, reason = self._apcorr_prereq_ok()
        if not ok:
            self.log(f"[AUTO] Photometry prerequisite missing: {reason}")
            self.log("[AUTO] Running Apcorr first to generate required files.")
            self._pending_phot_after_apcorr = True
            self.run_apcorr()
            return

        self._pending_phot_after_apcorr = False
        self.log_text.clear()
        self.log(
            "Params | "
            f"files={len(self.file_list)} | use_cropped={self.use_cropped} | "
            f"step9_use_cache={getattr(self.params.P, 'step9_use_cache', False)} | "
            f"force_rephot={getattr(self.params.P, 'force_rephot', False)} | "
            f"ap_mode={getattr(self.params.P, 'aperture_mode', 'apcorr')} | "
            f"min_snr={getattr(self.params.P, 'min_snr_for_mag', 3.0)}"
        )
        if hasattr(self, "frame_table"):
            self.frame_table.setRowCount(0)

        self.worker = ForcedPhotometryWorker(
            self.file_list,
            self.params,
            self.params.P.data_dir,
            self.params.P.result_dir,
            self.params.P.cache_dir,
            self.use_cropped
        )
        self.worker.progress.connect(self.on_progress)
        self.worker.frame_done.connect(self.on_frame_done)  # Real-time table update
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.log.connect(self.log)

        self.btn_run_phot.setEnabled(False)
        self.btn_stop_phot.setEnabled(True)
        if hasattr(self, "btn_run_apcorr"):
            self.btn_run_apcorr.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(len(self.file_list))
        self.progress_label.setText(f"0/{len(self.file_list)} | Starting...")
        self.worker.start()
        self.show_log_window()

    def stop_photometry(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()

    def run_apcorr(self):
        if not self.file_list:
            QMessageBox.warning(self, "Warning", "No files to process")
            return
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "Busy", "Photometry is running. Stop/finish photometry first.")
            return
        if self.apcorr_worker and self.apcorr_worker.isRunning():
            return

        if hasattr(self, "apcorr_table"):
            self.apcorr_table.setRowCount(0)
        if hasattr(self, "apcorr_status"):
            self.apcorr_status.setText("growth curve: running...")
        self.log("Start Apcorr build (aperture_by_frame + apcorr_summary)")

        self.apcorr_worker = ApertureWorker(
            self.file_list,
            self.params,
            self.params.P.data_dir,
            self.params.P.result_dir,
            self.params.P.cache_dir,
            self.use_cropped
        )
        self.apcorr_worker.progress.connect(self.on_apcorr_progress)
        self.apcorr_worker.finished.connect(self.on_apcorr_finished)
        self.apcorr_worker.error.connect(self.on_apcorr_error)

        self.btn_run_apcorr.setEnabled(False)
        self.btn_stop_apcorr.setEnabled(True)
        if hasattr(self, "btn_run_phot"):
            self.btn_run_phot.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(len(self.file_list))
        self.progress_label.setText(f"0/{len(self.file_list)} | Apcorr...")
        self.apcorr_worker.start()
        self.show_log_window()

    def stop_apcorr(self):
        if self.apcorr_worker and self.apcorr_worker.isRunning():
            self.apcorr_worker.stop()

    def on_apcorr_progress(self, current, total, filename):
        self.progress_bar.setValue(current)
        self.progress_label.setText(f"{current}/{total} | Apcorr {filename}")

    def on_apcorr_finished(self, summary):
        if hasattr(self, "btn_run_apcorr"):
            self.btn_run_apcorr.setEnabled(True)
        if hasattr(self, "btn_stop_apcorr"):
            self.btn_stop_apcorr.setEnabled(False)
        if hasattr(self, "btn_run_phot"):
            self.btn_run_phot.setEnabled(True)
        self.progress_label.setText("Done")
        self.log(f"Apcorr done: {summary}")
        self._cleanup_apcorr_worker(timeout_ms=5000)
        self.update_apcorr_tables()
        self.save_state()
        self.update_navigation_buttons()

        if self._pending_phot_after_apcorr:
            ok, reason = self._apcorr_prereq_ok()
            if ok:
                self.log("Apcorr prerequisite ready. Starting photometry automatically.")
                self._pending_phot_after_apcorr = False
                self.run_photometry()
            else:
                self._pending_phot_after_apcorr = False
                self.log(f"[AUTO] Apcorr finished but prerequisite check failed: {reason}")
                QMessageBox.warning(
                    self,
                    "Apcorr Incomplete",
                    "Apcorr run finished, but required files for photometry are still invalid.\n"
                    f"{reason}"
                )

    def on_apcorr_error(self, filename, error):
        self.log(f"APCORR ERROR {filename}: {error}")

    def on_progress(self, current, total, filename):
        self.progress_bar.setValue(current)
        self.progress_label.setText(f"{current}/{total} | {filename}")

    def on_frame_done(self, filename, result):
        """Real-time update of frame table when a frame completes."""
        if not hasattr(self, "frame_table"):
            return

        r = self.frame_table.rowCount()
        self.frame_table.insertRow(r)
        self.frame_table.setItem(r, 0, QTableWidgetItem(str(result.get("file", filename))))
        self.frame_table.setItem(r, 1, QTableWidgetItem(str(result.get("filter", ""))))
        n_val = result.get("n", 0)
        self.frame_table.setItem(r, 2, QTableWidgetItem(str(n_val) if n_val != "cached" else "cached"))
        self.frame_table.setItem(r, 3, QTableWidgetItem(str(result.get("n_goodmag", 0))))
        self.frame_table.setItem(r, 4, QTableWidgetItem(str(result.get("n_fail", 0))))
        self.frame_table.setItem(r, 5, QTableWidgetItem(str(result.get("targets", 0))))

        # Scroll to the new row
        self.frame_table.scrollToBottom()

    def on_finished(self, summary):
        self.btn_run_phot.setEnabled(True)
        self.btn_stop_phot.setEnabled(False)
        if hasattr(self, "btn_run_apcorr"):
            self.btn_run_apcorr.setEnabled(True)
        self.progress_label.setText("Done")
        self.log(f"Photometry done: {summary}")

        # Deterministic worker cleanup to avoid QThread destruction race.
        self._cleanup_worker(timeout_ms=5000)

        # Don't call update_frame_table() here - already updated via frame_done
        idx_path = step9_dir(self.params.P.result_dir) / "photometry_index.csv"
        if not idx_path.exists():
            idx_path = self.params.P.result_dir / "photometry_index.csv"
        if idx_path.exists():
            try:
                idx = pd.read_csv(idx_path)
                if not idx.empty and "filter" in idx.columns:
                    by_f = idx.groupby("filter").agg(
                        frames=("file", "count"),
                        n_rows=("n", "sum"),
                        n_good=("n_goodmag", "sum"),
                        n_fail=("n_fail", "sum"),
                        targets=("targets", "sum"),
                    )
                    for filt, row in by_f.iterrows():
                        self.log(
                            f"Filter[{filt}] frames={int(row['frames'])} | "
                            f"rows={int(row['n_rows'])} | good={int(row['n_good'])} | "
                            f"fail={int(row['n_fail'])} | targets={int(row['targets'])}"
                        )
            except Exception:
                pass
        self.save_state()
        self.update_apcorr_tables()
        self.update_navigation_buttons()

    def on_error(self, filename, error):
        self.log(f"ERROR {filename}: {error}")

    def _cleanup_worker(self, timeout_ms=5000):
        """Safely cleanup worker thread after completion or stop request."""
        if not self.worker:
            return True

        worker = self.worker
        if worker.isRunning():
            try:
                worker.stop()
            except Exception:
                pass
            worker.quit()
            if not worker.wait(int(timeout_ms)):
                self.log("Worker is still running; close is deferred.")
                return False

        try:
            worker.deleteLater()
        except Exception:
            pass
        self.worker = None
        return True

    def _cleanup_apcorr_worker(self, timeout_ms=5000):
        if not self.apcorr_worker:
            return True
        worker = self.apcorr_worker
        if worker.isRunning():
            try:
                worker.stop()
            except Exception:
                pass
            worker.quit()
            if not worker.wait(int(timeout_ms)):
                self.log("Apcorr worker is still running; close is deferred.")
                return False
        try:
            worker.deleteLater()
        except Exception:
            pass
        self.apcorr_worker = None
        return True

    def show_log_window(self):
        self.log_window.show()
        self.log_window.raise_()
        self.log_window.activateWindow()

    def update_frame_table(self):
        idx_path = step9_dir(self.params.P.result_dir) / "photometry_index.csv"
        if not idx_path.exists():
            idx_path = self.params.P.result_dir / "photometry_index.csv"
        if not idx_path.exists() or not hasattr(self, "frame_table"):
            return
        try:
            idx = pd.read_csv(idx_path)
        except Exception:
            return
        if idx.empty:
            self.frame_table.setRowCount(0)
            return
        cols = {c.lower(): c for c in idx.columns}
        idx = idx.copy()
        if "filter" in cols:
            idx["filter"] = idx[cols["filter"]]
        elif "FILTER" in idx.columns:
            idx["filter"] = idx["FILTER"]
        if "file" in idx.columns:
            def _frame_num(val):
                m = re.search(r"(\d+)", str(val))
                return int(m.group(1)) if m else 0
            idx["_frame_num"] = idx["file"].map(_frame_num)
            order_base = idx.sort_values(["_frame_num", "file"])
            order_filters = list(pd.unique(order_base["filter"].astype(str).str.strip().str.lower()))
            order = {f: i for i, f in enumerate(order_filters)}
            idx["_filter_rank"] = idx["filter"].astype(str).str.strip().str.lower().map(order).fillna(99)
            idx = idx.sort_values(["_filter_rank", "_frame_num", "file"])

        def _fmt_count(val):
            if isinstance(val, str):
                s = val.strip().lower()
                if s == "cached":
                    return "cached"
            if pd.isna(val):
                return "0"
            try:
                return str(int(val))
            except Exception:
                try:
                    return str(int(float(val)))
                except Exception:
                    return str(val)

        self.frame_table.setRowCount(0)
        for _, row in idx.iterrows():
            r = self.frame_table.rowCount()
            self.frame_table.insertRow(r)
            self.frame_table.setItem(r, 0, QTableWidgetItem(str(row.get("file", ""))))
            self.frame_table.setItem(r, 1, QTableWidgetItem(str(row.get("filter", ""))))
            self.frame_table.setItem(r, 2, QTableWidgetItem(_fmt_count(row.get("n", 0))))
            self.frame_table.setItem(r, 3, QTableWidgetItem(_fmt_count(row.get("n_goodmag", 0))))
            self.frame_table.setItem(r, 4, QTableWidgetItem(_fmt_count(row.get("n_fail", 0))))
            self.frame_table.setItem(r, 5, QTableWidgetItem(_fmt_count(row.get("targets", 0))))

    def _refresh_apcorr_filter_map(self):
        mapping = {}
        idx_path = step9_dir(self.params.P.result_dir) / "photometry_index.csv"
        if not idx_path.exists():
            idx_path = self.params.P.result_dir / "photometry_index.csv"
        if idx_path.exists():
            try:
                idx = pd.read_csv(idx_path)
                if {"file", "filter"} <= set(idx.columns):
                    for _, row in idx.iterrows():
                        mapping[str(row["file"])] = str(row["filter"]).strip().lower()
            except Exception:
                pass
        self._apcorr_filter_by_file = mapping

    def _filter_color(self, filt: str) -> str:
        key = str(filt or "unknown").strip().lower()
        palette = {
            "u": "#8E24AA",
            "g": "#43A047",
            "r": "#E53935",
            "i": "#3949AB",
            "z": "#6D4C41",
            "y": "#00897B",
            "ha": "#FB8C00",
            "oiii": "#00ACC1",
            "sii": "#6D4C41",
            "unknown": "#546E7A",
        }
        return palette.get(key, "#546E7A")

    def _filter_from_filename(self, fname: str) -> str:
        stem = Path(str(fname)).stem.lower()
        m = re.search(r"(?:^|[_-])([ugrizy])(?:[_-]|$)", stem)
        if m:
            return normalize_filter_name(m.group(1))
        tokens = [t for t in re.split(r"[^a-z0-9]+", stem) if t]
        for tok in reversed(tokens):
            norm = normalize_filter_name(tok)
            if norm and norm != "unknown":
                return norm
        return "unknown"

    def _resolve_filter_for_file(self, fname: str) -> str:
        key = str(fname)
        if key in self._apcorr_filter_by_file:
            v = str(self._apcorr_filter_by_file.get(key, "")).strip().lower()
            if v:
                return v
        if key in self._filter_cache:
            return self._filter_cache[key]

        filt = self._filter_from_filename(key)
        if filt == "unknown":
            try:
                data_dir = Path(self.params.P.data_dir)
            except Exception:
                data_dir = Path(".")
            probe_paths = []
            if self.use_cropped:
                crop_dir = step2_cropped_dir(self.params.P.result_dir)
                probe_paths.append(crop_dir / key)
                probe_paths.append((self.params.P.result_dir / "cropped") / key)
            probe_paths.append(data_dir / key)
            for probe in probe_paths:
                if probe.exists():
                    filt = _get_filter_lower(probe)
                    if filt and filt != "unknown":
                        break
        filt = str(filt or "unknown").strip().lower()
        self._filter_cache[key] = filt
        return filt

    def update_apcorr_tables(self):
        if not hasattr(self, "apcorr_table"):
            return

        base_dir = step9_dir(self.params.P.result_dir)
        sum_path = base_dir / "apcorr_summary.csv"
        gc_path = base_dir / "growth_curve.csv"
        if not sum_path.exists():
            sum_path = self.params.P.result_dir / "apcorr_summary.csv"
        if not gc_path.exists():
            gc_path = self.params.P.result_dir / "growth_curve.csv"

        # Load summary
        if sum_path.exists():
            try:
                df_sum = pd.read_csv(sum_path)
            except Exception:
                df_sum = pd.DataFrame()
        else:
            df_sum = pd.DataFrame()

        # Load growth curve data
        if gc_path.exists():
            try:
                df_gc = pd.read_csv(gc_path)
            except Exception:
                df_gc = pd.DataFrame()
        else:
            df_gc = pd.DataFrame()

        self._apcorr_summary_df = df_sum.copy()
        self._growth_curve_df = df_gc.copy()
        self._refresh_apcorr_filter_map()

        # Fill summary table
        self.apcorr_table.setRowCount(0)
        if not df_sum.empty:
            for _, row in df_sum.iterrows():
                r = self.apcorr_table.rowCount()
                self.apcorr_table.insertRow(r)
                vals = [
                    str(row.get("file", "")),
                    f"{row.get('fwhm_used', np.nan):.2f}" if np.isfinite(float(row.get("fwhm_used", np.nan))) else "",
                    f"{row.get('r_optimal', np.nan):.2f}" if np.isfinite(float(row.get("r_optimal", np.nan))) else "",
                    f"{row.get('optimal_scale', np.nan):.2f}" if np.isfinite(float(row.get("optimal_scale", np.nan))) else "",
                    f"{row.get('apcorr', np.nan):.4f}" if np.isfinite(float(row.get("apcorr", np.nan))) else "",
                    f"{row.get('snr_optimal', np.nan):.1f}" if np.isfinite(float(row.get("snr_optimal", np.nan))) else "",
                    f"{row.get('mag_err_optimal', np.nan):.4f}" if np.isfinite(float(row.get("mag_err_optimal", np.nan))) else "",
                    str(int(row.get("n_used", 0))) if np.isfinite(float(row.get("n_used", 0))) else "0",
                    str(row.get("apply", False)),
                ]
                for c, text in enumerate(vals):
                    self.apcorr_table.setItem(r, c, QTableWidgetItem(text))

        # Status bar
        if hasattr(self, "apcorr_status"):
            if not df_sum.empty:
                status = f"growth curve: {len(df_sum)} frames"
                if "apply" in df_sum.columns:
                    try:
                        n_apply = int(df_sum["apply"].astype(str).str.lower().isin(["1", "true", "yes", "y"]).sum())
                        status += f", apply={n_apply}"
                        if n_apply == 0:
                            status += " (all false: tune scatter_max/min_n)"
                    except Exception:
                        pass
                self.apcorr_status.setText(status)
            else:
                self.apcorr_status.setText("growth curve: not found")

        # Populate frame list
        if hasattr(self, "apcorr_file_list"):
            files = []
            if not df_gc.empty and "file" in df_gc.columns:
                files = sorted([str(x) for x in pd.unique(df_gc["file"].astype(str)) if str(x).strip()])
            elif not df_sum.empty and "file" in df_sum.columns:
                files = sorted([str(x) for x in pd.unique(df_sum["file"].astype(str)) if str(x).strip()])
            self.apcorr_file_list.blockSignals(True)
            self.apcorr_file_list.clear()
            if files:
                self.apcorr_file_list.addItems(files)
            self.apcorr_file_list.blockSignals(False)
            if files:
                self.apcorr_file_list.setCurrentRow(0)
                self._on_apcorr_file_changed(files[0])
            else:
                if hasattr(self, "apcorr_file_cand_table"):
                    self.apcorr_file_cand_table.setRowCount(0)
                self._clear_apcorr_plot("No growth curve data")

    def _clear_apcorr_plot(self, message: str):
        if not hasattr(self, "apcorr_ax_mag"):
            return
        for ax in (self.apcorr_ax_mag, self.apcorr_ax_err):
            ax.clear()
            ax.text(0.5, 0.5, message, ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="#888")
            ax.set_axis_off()
        self.apcorr_canvas.draw_idle()

    def _on_apcorr_file_changed(self, filename: str):
        if not filename or not hasattr(self, "apcorr_file_cand_table"):
            return

        df_gc = getattr(self, "_growth_curve_df", None)
        if df_gc is None or df_gc.empty or "file" not in df_gc.columns:
            self.apcorr_file_cand_table.setRowCount(0)
            self._clear_apcorr_plot("No growth curve data")
            return

        sub = df_gc[df_gc["file"].astype(str) == str(filename)].copy()
        if sub.empty:
            self.apcorr_file_cand_table.setRowCount(0)
            self._clear_apcorr_plot("No growth curve data for this frame")
            return

        sub = sub.sort_values("r_px", kind="stable")

        # Fill per-frame table
        self.apcorr_file_cand_table.setRowCount(0)
        for _, row in sub.iterrows():
            r = self.apcorr_file_cand_table.rowCount()
            self.apcorr_file_cand_table.insertRow(r)
            vals = [
                f"{row.get('scale', np.nan):.2f}" if np.isfinite(float(row.get("scale", np.nan))) else "",
                f"{row.get('r_px', np.nan):.2f}" if np.isfinite(float(row.get("r_px", np.nan))) else "",
                f"{row.get('median_mag', np.nan):.4f}" if np.isfinite(float(row.get("median_mag", np.nan))) else "",
                f"{row.get('median_mag_err', np.nan):.4f}" if np.isfinite(float(row.get("median_mag_err", np.nan))) else "",
                f"{row.get('median_snr', np.nan):.1f}" if np.isfinite(float(row.get("median_snr", np.nan))) else "",
                str(int(row.get("n_stars", 0))) if np.isfinite(float(row.get("n_stars", 0))) else "0",
                str(row.get("selected", False)),
            ]
            for c, text in enumerate(vals):
                self.apcorr_file_cand_table.setItem(r, c, QTableWidgetItem(text))

        # --- Plot ---
        if not hasattr(self, "apcorr_ax_mag"):
            return

        ax_mag = self.apcorr_ax_mag
        ax_err = self.apcorr_ax_err
        ax_mag.clear()
        ax_err.clear()

        r_px = pd.to_numeric(sub["r_px"], errors="coerce").to_numpy(float)
        med_mag = pd.to_numeric(sub.get("median_mag", np.nan), errors="coerce").to_numpy(float)
        med_err = pd.to_numeric(sub.get("median_mag_err", np.nan), errors="coerce").to_numpy(float)
        sel_mask = sub.get("selected", False)
        if hasattr(sel_mask, "map"):
            sel_mask = sel_mask.map(lambda v: _as_bool(v, False)).to_numpy(bool)
        else:
            sel_mask = np.zeros(len(sub), dtype=bool)

        finite_mag = np.isfinite(r_px) & np.isfinite(med_mag)
        finite_err = np.isfinite(r_px) & np.isfinite(med_err)

        if not np.any(finite_mag) and not np.any(finite_err):
            self._clear_apcorr_plot("No finite growth curve points")
            return

        filt_sel = self._resolve_filter_for_file(filename)
        color_sel = self._filter_color(filt_sel)

        # Get summary info for this frame
        df_sum = getattr(self, "_apcorr_summary_df", pd.DataFrame())
        apply_on = False
        r_optimal = np.nan
        apcorr_val = np.nan
        fwhm_used = np.nan
        if not df_sum.empty and "file" in df_sum.columns:
            frame_sum = df_sum[df_sum["file"].astype(str) == str(filename)]
            if not frame_sum.empty:
                row_s = frame_sum.iloc[0]
                apply_on = _as_bool(row_s.get("apply", False), False)
                r_optimal = float(row_s.get("r_optimal", np.nan))
                apcorr_val = float(row_s.get("apcorr", np.nan))
                fwhm_used = float(row_s.get("fwhm_used", np.nan))

        # --- Top subplot: Growth Curve (Magnitude vs Aperture Radius) ---
        if np.any(finite_mag):
            ax_mag.plot(r_px[finite_mag], med_mag[finite_mag], "-o",
                        color=color_sel, linewidth=1.5, markersize=5,
                        markeredgecolor="white", markeredgewidth=0.5,
                        label=f"{filt_sel}")
            if np.any(finite_err & finite_mag):
                both = finite_mag & finite_err
                ax_mag.errorbar(r_px[both], med_mag[both],
                                yerr=med_err[both],
                                fmt="none", ecolor=color_sel, alpha=0.3,
                                capsize=2, linewidth=0.7)

        # Mark optimal aperture
        if np.isfinite(r_optimal):
            ax_mag.axvline(r_optimal, color="#E53935", linewidth=1.5,
                           linestyle="--", alpha=0.8, label=f"optimal r={r_optimal:.1f} px")
            # Shade optimal region
            ax_mag.axvspan(r_optimal * 0.9, r_optimal * 1.1,
                           color="#E53935", alpha=0.08)

        if np.isfinite(fwhm_used) and fwhm_used > 0:
            ax_mag.axvline(
                fwhm_used, color="#6D4C41", linewidth=1.2, linestyle=":",
                alpha=0.8, label=f"FWHM={fwhm_used:.2f} px"
            )

        # Brighter stars are lower magnitude values.
        ax_mag.invert_yaxis()
        ax_mag.set_ylabel("Median Magnitude (inst)", fontsize=9)
        title_str = f"{filename} | filter={filt_sel}"
        if np.isfinite(apcorr_val):
            title_str += f" | apcorr={apcorr_val:.4f}"
        title_str += f" | apply={'ON' if apply_on else 'OFF'}"
        ax_mag.set_title(title_str, fontsize=9)
        ax_mag.grid(True, alpha=0.25)
        ax_mag.legend(fontsize=8, frameon=False, loc="upper right")

        # --- Bottom subplot: Error curve (U-shape) ---
        if np.any(finite_err):
            ax_err.plot(r_px[finite_err], med_err[finite_err], "-s",
                        color="#1565C0", linewidth=1.8, markersize=5,
                        markeredgecolor="white", markeredgewidth=0.5,
                        label="median mag_err")

        # Mark optimal (minimum error)
        if np.isfinite(r_optimal) and np.any(finite_err):
            opt_err_val = med_err[sel_mask & finite_err]
            if len(opt_err_val) > 0:
                ax_err.plot(r_optimal, opt_err_val[0], "o",
                            color="#E53935", markersize=10, markeredgecolor="white",
                            markeredgewidth=1.5, zorder=5,
                            label=f"min err={opt_err_val[0]:.4f}")
            ax_err.axvline(r_optimal, color="#E53935", linewidth=1.5,
                           linestyle="--", alpha=0.8)
            ax_err.axvspan(r_optimal * 0.9, r_optimal * 1.1,
                           color="#E53935", alpha=0.08)

        if np.isfinite(fwhm_used) and fwhm_used > 0:
            ax_err.axvline(
                fwhm_used, color="#6D4C41", linewidth=1.2, linestyle=":",
                alpha=0.8, label=f"FWHM={fwhm_used:.2f} px"
            )

        if np.isfinite(fwhm_used) and fwhm_used > 0:
            sec_mag = ax_mag.secondary_xaxis(
                "top",
                functions=(lambda x: x / fwhm_used, lambda s: s * fwhm_used),
            )
            sec_mag.set_xlabel("Radius (×FWHM)", fontsize=8)
            sec_mag.tick_params(labelsize=8)

            sec_err = ax_err.secondary_xaxis(
                "top",
                functions=(lambda x: x / fwhm_used, lambda s: s * fwhm_used),
            )
            sec_err.set_xlabel("Radius (×FWHM)", fontsize=8)
            sec_err.tick_params(labelsize=8)

        x_label = "Aperture radius (px)"
        if np.isfinite(fwhm_used) and fwhm_used > 0:
            x_label += f" | FWHM={fwhm_used:.2f}px"
        ax_err.set_xlabel(x_label, fontsize=9)
        ax_err.set_ylabel("Median mag error", fontsize=9)
        ax_err.set_title("Error vs Aperture (U-shape: Poisson left, sky noise right)", fontsize=9)
        ax_err.grid(True, alpha=0.25)
        ax_err.legend(fontsize=8, frameon=False, loc="upper right")

        self.apcorr_fig.tight_layout()
        self.apcorr_canvas.draw_idle()

    def closeEvent(self, event):
        """Ensure worker thread is stopped before closing window"""
        if self.worker and self.worker.isRunning():
            self.stop_photometry()
        if self.apcorr_worker and self.apcorr_worker.isRunning():
            self.stop_apcorr()
        if getattr(self, "overlay_panel", None) is not None:
            try:
                if hasattr(self.overlay_panel, "log_window") and self.overlay_panel.log_window is not None:
                    self.overlay_panel.log_window.close()
            except Exception:
                pass
        if not self._cleanup_worker(timeout_ms=10000):
            QMessageBox.warning(
                self,
                "Background Task Running",
                "Photometry worker is still stopping. Please wait a few seconds and close again.",
            )
            event.ignore()
            return
        if not self._cleanup_apcorr_worker(timeout_ms=10000):
            QMessageBox.warning(
                self,
                "Background Task Running",
                "Apcorr worker is still stopping. Please wait a few seconds and close again.",
            )
            event.ignore()
            return
        super().closeEvent(event)

    def validate_step(self) -> bool:
        idx_path = step9_dir(self.params.P.result_dir) / "photometry_index.csv"
        if not idx_path.exists():
            idx_path = self.params.P.result_dir / "photometry_index.csv"
        return idx_path.exists()

    def save_state(self):
        state_data = {
            "force_rephot": getattr(self.params.P, "force_rephot", False),
            "step9_use_cache": getattr(self.params.P, "step9_use_cache", False),
            "aperture_mode": getattr(self.params.P, "aperture_mode", "apcorr"),
            "min_snr_for_mag": getattr(self.params.P, "min_snr_for_mag", 3.0),
            "phot_aperture_scale": getattr(self.params.P, "phot_aperture_scale", 1.0),
            "fitsky_annulus_scale": getattr(self.params.P, "fitsky_annulus_scale", 4.0),
            "fitsky_dannulus_scale": getattr(self.params.P, "fitsky_dannulus_scale", 2.0),
            "annulus_min_gap_px": getattr(self.params.P, "annulus_min_gap_px", 6.0),
            "annulus_min_width_px": getattr(self.params.P, "annulus_min_width_px", 12.0),
            "annulus_sigma_clip": getattr(self.params.P, "annulus_sigma_clip", 3.0),
            "apcorr_apply": getattr(self.params.P, "apcorr_apply", True),
            "apcorr_use_min_n": getattr(self.params.P, "apcorr_use_min_n", 20),
            "apcorr_scatter_max": getattr(self.params.P, "apcorr_scatter_max", 0.05),
        }
        self.project_state.store_step_data("forced_photometry", state_data)

    def restore_state(self):
        state_data = self.project_state.get_step_data("forced_photometry")
        if state_data:
            for key, val in state_data.items():
                if hasattr(self.params.P, key):
                    setattr(self.params.P, key, val)
