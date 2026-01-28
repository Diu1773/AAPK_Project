"""
Step 7: Star ID Matching (WCS-based)

- Uses a fixed ref catalog (RA/Dec) from Step 6
- Converts detections to sky coordinates via per-frame WCS
- Sky-matches within a configurable arcsec radius
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import proj_plane_pixel_scales
import astropy.units as u

from PyQt5.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QGroupBox, QMessageBox,
    QTextEdit, QFormLayout, QProgressBar,
    QDoubleSpinBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QWidget, QDialog, QDialogButtonBox, QTabWidget, QComboBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from .step_window_base import StepWindowBase
from ...utils.step_paths import (
    step2_cropped_dir,
    crop_is_active,
    step4_dir,
    step5_dir,
    step6_dir,
    step7_dir,
    legacy_step5_refbuild_dir,
    legacy_step7_refbuild_dir,
    legacy_step6_idmatch_dir,
)


_DATE_RE = re.compile(r"(20\d{6})")


def _astap_wcs_candidates(fits_path: Path) -> List[Path]:
    return [
        fits_path.with_suffix(".wcs"),
        Path(str(fits_path) + ".wcs"),
        fits_path.parent / (fits_path.stem + ".wcs"),
        fits_path.parent / (fits_path.name + ".wcs"),
    ]


def _parse_astap_wcs_file(wcs_path: Path) -> dict:
    d: Dict[str, object] = {}
    try:
        lines = wcs_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return d
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        if "/" in s:
            s = s.split("/", 1)[0].strip()
        if "=" not in s:
            continue
        key, val = [t.strip() for t in s.split("=", 1)]
        if not key:
            continue
        if val.startswith("'") and val.endswith("'"):
            d[key] = val.strip("'")
            continue
        try:
            if "." in val or "E" in val.upper():
                d[key] = float(val)
            else:
                d[key] = int(val)
        except Exception:
            d[key] = val
    return d


def _parse_date_key(value: str, params) -> Optional[str]:
    mode = str(getattr(params.P, "night_parse_mode", "regex") or "regex").strip().lower()
    if mode == "split":
        delim = str(getattr(params.P, "night_parse_split_delim", "_"))
        parts = value.split(delim) if delim else [value]
        idx = int(getattr(params.P, "night_parse_split_index", -1))
        if idx < 0:
            idx = len(parts) + idx
        if idx < 0 or idx >= len(parts):
            return None
        return parts[idx]
    if mode == "last_digits":
        n_digits = max(1, int(getattr(params.P, "night_parse_last_digits", 8)))
        m = re.search(rf"(\\d{{{n_digits}}})$", value)
        return m.group(1) if m else None
    try:
        pattern = str(getattr(params.P, "night_parse_regex", r".*_(\d{8})"))
        m = re.search(pattern, value)
    except re.error:
        return None
    if not m:
        return None
    if m.groupdict().get("date"):
        return m.group("date")
    if m.groups():
        return m.group(1)
    return m.group(0)


def _extract_date_key(filename: str, params=None) -> str:
    if params is None or not hasattr(params, "P"):
        match = _DATE_RE.search(str(filename))
        return match.group(1) if match else "unknown_date"
    date_key = None
    try:
        data_dir = Path(getattr(params.P, "data_dir", "."))
        file_path = Path(params.get_file_path(filename))
        if file_path.parent != data_dir:
            date_key = _parse_date_key(file_path.parent.name, params)
        if not date_key:
            date_key = _parse_date_key(file_path.name, params)
    except Exception:
        date_key = None
    if not date_key:
        date_key = _parse_date_key(str(filename), params)
    return date_key or "unknown_date"


def _safe_float(val, default=np.nan):
    try:
        if val is None:
            return default
        return float(val)
    except Exception:
        return default


def _apply_transform(xy: np.ndarray, mat: np.ndarray) -> np.ndarray:
    return xy @ mat[:, :2].T + mat[:, 2]


def _estimate_similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    if len(src) == 0:
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean

    cov = (src_c.T @ dst_c) / len(src)
    u, s, vt = np.linalg.svd(cov)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[1, :] *= -1
        r = vt.T @ u.T

    var_src = np.mean(np.sum(src_c ** 2, axis=1))
    scale = np.sum(s) / var_src if var_src > 0 else 1.0
    t = dst_mean - scale * (r @ src_mean)

    mat = np.zeros((2, 3), dtype=float)
    mat[:, :2] = scale * r
    mat[:, 2] = t
    return mat


def _estimate_affine(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    if len(src) == 0:
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    a = np.column_stack([src, np.ones(len(src))])
    m, _, _, _ = np.linalg.lstsq(a, dst, rcond=None)
    return m.T


def _estimate_transform(
    ref_xy: np.ndarray,
    det_xy: np.ndarray,
    mode: str,
    init_radius: float,
    min_pairs: int,
    shift_tol_px: float | None = None,
) -> Tuple[np.ndarray, int]:
    if len(ref_xy) == 0 or len(det_xy) == 0:
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), 0

    ref_med = np.nanmedian(ref_xy, axis=0)
    det_med = np.nanmedian(det_xy, axis=0)
    shift = det_med - ref_med

    ref_shifted = ref_xy + shift
    tree = cKDTree(det_xy)
    d, idx = tree.query(ref_shifted, k=1, distance_upper_bound=init_radius)
    valid = np.isfinite(d)
    if not np.any(valid):
        mat = np.array([[1.0, 0.0, shift[0]], [0.0, 1.0, shift[1]]])
        return mat, 0

    pairs = []
    for i, (dist, j) in enumerate(zip(d, idx)):
        if not np.isfinite(dist):
            continue
        if j >= len(det_xy):
            continue
        pairs.append((i, int(j), float(dist)))

    pairs.sort(key=lambda r: r[2])
    used_det = set()
    ref_idx = []
    det_idx = []
    for i, j, _ in pairs:
        if j in used_det:
            continue
        used_det.add(j)
        ref_idx.append(i)
        det_idx.append(j)

    if len(ref_idx) < max(2, min_pairs):
        mat = np.array([[1.0, 0.0, shift[0]], [0.0, 1.0, shift[1]]])
        return mat, len(ref_idx)

    ref_idx = np.array(ref_idx, dtype=int)
    det_idx = np.array(det_idx, dtype=int)

    if shift_tol_px is not None and shift_tol_px > 0:
        dx = det_xy[det_idx, 0] - ref_xy[ref_idx, 0]
        dy = det_xy[det_idx, 1] - ref_xy[ref_idx, 1]
        med_dx = np.nanmedian(dx)
        med_dy = np.nanmedian(dy)
        tol = float(shift_tol_px)
        good = (np.abs(dx - med_dx) <= tol) & (np.abs(dy - med_dy) <= tol)
        if np.sum(good) >= max(2, min_pairs):
            ref_idx = ref_idx[good]
            det_idx = det_idx[good]

    src = ref_xy[ref_idx]
    dst = det_xy[det_idx]

    mode = (mode or "similarity").lower()
    if mode == "affine":
        mat = _estimate_affine(src, dst)
    elif mode == "shift":
        mat = np.array([[1.0, 0.0, shift[0]], [0.0, 1.0, shift[1]]])
    else:
        mat = _estimate_similarity(src, dst)

    return mat, len(ref_idx)


class IdMatchWorker(QThread):
    progress = pyqtSignal(int, int, str)
    log = pyqtSignal(str)
    file_done = pyqtSignal(str, dict)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str, str)

    def __init__(
        self,
        file_list,
        params,
        data_dir,
        result_dir,
        cache_dir,
        ref_frame: str,
        ref_filter: str,
        match_r_fwhm: float,
        init_r_fwhm: float,
        ratio_max: float,
        min_pairs: int,
        transform_mode: str,
        mutual_nearest: bool,
        ref_per_date: bool,
        two_pass_enable: bool = True,
        tight_radius_arcsec: float = 1.0,
        loose_radius_arcsec: float = 3.0,
    ):
        super().__init__()
        self.file_list = list(file_list)
        self.params = params
        self.data_dir = Path(data_dir)
        self.result_dir = Path(result_dir)
        self.cache_dir = Path(cache_dir)
        self.ref_frame = ref_frame
        self.ref_filter = ref_filter
        self.match_r_fwhm = float(match_r_fwhm)
        self.init_r_fwhm = float(init_r_fwhm)
        self.ratio_max = float(ratio_max)
        self.min_pairs = int(min_pairs)
        self.transform_mode = transform_mode
        self.mutual_nearest = bool(mutual_nearest)
        self.ref_per_date = bool(ref_per_date)
        self.two_pass_enable = bool(two_pass_enable)
        self.tight_radius_arcsec = float(tight_radius_arcsec)
        self.loose_radius_arcsec = float(loose_radius_arcsec)
        self._stop_requested = False
        self._log_file = None
        self._ref_cache: Dict[str, Tuple[np.ndarray, SkyCoord]] = {}

    def stop(self):
        self._stop_requested = True

    def _log(self, msg: str):
        self.log.emit(msg)
        if self._log_file is not None:
            try:
                self._log_file.write(msg + "\n")
                self._log_file.flush()
            except Exception:
                pass

    def _load_meta(self, fname: str) -> dict:
        meta_path = self.cache_dir / f"detect_{fname}.json"
        if not meta_path.exists():
            fallback = step4_dir(self.result_dir) / f"detect_{fname}.json"
            if fallback.exists():
                meta_path = fallback
        if meta_path.exists():
            try:
                return json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _resolve_detect_csv(self, fname: str) -> Optional[Path]:
        cand = self.cache_dir / f"detect_{fname}.csv"
        if cand.exists():
            return cand
        fallback = step4_dir(self.result_dir) / f"detect_{fname}.csv"
        if fallback.exists():
            return fallback
        return None

    def _load_detect_xy(self, fname: str) -> np.ndarray:
        p = self._resolve_detect_csv(fname)
        if p is not None and p.exists() and p.stat().st_size > 0:
            try:
                df = pd.read_csv(p)
                if {"x", "y"} <= set(df.columns) and len(df):
                    xy = df[["x", "y"]].to_numpy(float)
                    xy = xy[np.isfinite(xy).all(axis=1)]
                    return xy
            except Exception:
                pass
        return np.zeros((0, 2), float)

    def _resolve_fits_path(self, fname: str) -> Optional[Path]:
        step5_out = step5_dir(self.result_dir)
        cand = step5_out / fname
        if cand.exists():
            return cand
        if crop_is_active(self.result_dir):
            cand = step2_cropped_dir(self.result_dir) / fname
            if cand.exists():
                return cand
        try:
            return Path(self.params.get_file_path(fname))
        except Exception:
            return None

    def _load_wcs_for_frame(self, fname: str) -> Optional[WCS]:
        fits_path = self._resolve_fits_path(fname)
        candidates = []
        if fits_path is not None and fits_path.exists():
            candidates.append(fits_path)
        try:
            orig = Path(self.params.get_file_path(fname))
            if orig.exists() and orig not in candidates:
                candidates.append(orig)
        except Exception:
            pass
        for path in candidates:
            try:
                hdr = fits.getheader(path)
                w = WCS(hdr, relax=True)
                if w.has_celestial:
                    return w
                for wcs_path in _astap_wcs_candidates(path):
                    if not wcs_path.exists():
                        continue
                    wcs_dict = _parse_astap_wcs_file(wcs_path)
                    if not wcs_dict:
                        continue
                    hdr2 = fits.Header()
                    for k, v in wcs_dict.items():
                        try:
                            hdr2[k] = v
                        except Exception:
                            pass
                    w2 = WCS(hdr2, relax=True)
                    if w2.has_celestial:
                        return w2
            except Exception:
                continue
        return None

    @staticmethod
    def _pixscale_arcsec(w: WCS) -> float:
        try:
            sc = proj_plane_pixel_scales(w.celestial) * 3600.0
            return float(np.nanmean(sc))
        except Exception:
            return float("nan")

    def _load_ref_catalog(self, date_key: Optional[str] = None) -> pd.DataFrame:
        step6_out = step6_dir(self.result_dir)
        candidates: List[Path] = []
        if date_key:
            candidates.append(step6_out / f"ref_catalog_{self.ref_filter}_{date_key}.tsv")
        candidates.append(step6_out / f"ref_catalog_{self.ref_filter}.tsv")
        candidates.append(legacy_step5_refbuild_dir(self.result_dir) / f"ref_catalog_{self.ref_filter}.tsv")
        candidates.append(legacy_step7_refbuild_dir(self.result_dir) / f"master_catalog_{self.ref_filter}.tsv")
        ref_path = None
        for cand in candidates:
            if cand.exists():
                ref_path = cand
                break
        if ref_path is None:
            raise FileNotFoundError(
                f"Reference catalog not found for filter '{self.ref_filter}'"
            )
        df = pd.read_csv(ref_path, sep="\t")
        if not {"source_id", "ra_deg", "dec_deg"} <= set(df.columns):
            raise RuntimeError(f"Invalid ref catalog (missing ra/dec): {ref_path}")
        df = df.copy()
        df["source_id"] = pd.to_numeric(df["source_id"], errors="coerce")
        df["ra_deg"] = pd.to_numeric(df["ra_deg"], errors="coerce")
        df["dec_deg"] = pd.to_numeric(df["dec_deg"], errors="coerce")
        df = df.dropna(subset=["source_id", "ra_deg", "dec_deg"]).copy()
        df["source_id"] = df["source_id"].astype("int64")
        return df

    def _get_ref_for_date(self, date_key: str, master_df: pd.DataFrame) -> Tuple[np.ndarray, SkyCoord]:
        if not self.ref_per_date:
            cached = self._ref_cache.get("global")
            if cached is not None:
                return cached
            ref_ids = master_df["source_id"].astype("int64").to_numpy()
            ref_sky = SkyCoord(master_df["ra_deg"].to_numpy(float) * u.deg,
                               master_df["dec_deg"].to_numpy(float) * u.deg,
                               frame="icrs")
            self._ref_cache["global"] = (ref_ids, ref_sky)
            return ref_ids, ref_sky

        cache_key = date_key or "unknown_date"
        if cache_key in self._ref_cache:
            return self._ref_cache[cache_key]
        try:
            df = self._load_ref_catalog(cache_key)
            if df.empty:
                raise RuntimeError("empty ref catalog")
        except Exception as e:
            self._log(f"[IDMATCH] date={cache_key}: per-date ref not found, fallback to global ({e})")
            df = master_df
        ref_ids = df["source_id"].astype("int64").to_numpy()
        ref_sky = SkyCoord(df["ra_deg"].to_numpy(float) * u.deg,
                           df["dec_deg"].to_numpy(float) * u.deg,
                           frame="icrs")
        self._ref_cache[cache_key] = (ref_ids, ref_sky)
        return ref_ids, ref_sky

    def run(self):
        try:
            self._run_impl()
        except Exception as e:
            import traceback
            self._log(f"[ERROR] {e}\n{traceback.format_exc()}")
            self.error.emit("WORKER", str(e))
            self.finished.emit({})
        finally:
            if self._log_file is not None:
                try:
                    self._log_file.close()
                except Exception:
                    pass
                self._log_file = None

    def _run_impl(self):
        if not self.file_list:
            raise RuntimeError("No frames to process")

        import time as _time
        out_dir = step7_dir(self.result_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / f"step7_idmatch_{_time.strftime('%Y%m%d_%H%M%S')}.log"
        try:
            self._log_file = open(log_path, "w", encoding="utf-8")
            self._log(f"[IDMATCH] Log file: {log_path}")
        except Exception as e:
            self._log(f"[WARN] Could not open log file: {e}")

        match_radius_arcsec = _safe_float(getattr(self.params.P, "idmatch_tol_arcsec", None), np.nan)
        self._log(
            "[IDMATCH] params: wcs_match_radius_arcsec={r}, match_r_fwhm={mf:.2f}".format(
                r=(f"{match_radius_arcsec:.2f}" if np.isfinite(match_radius_arcsec) else "auto"),
                mf=self.match_r_fwhm,
            )
        )
        param_path = getattr(self.params, "param_file", None)
        if param_path:
            self._log(f"[IDMATCH] param_file={param_path}")

        master_df = self._load_ref_catalog(None)
        global_ref_ids = master_df["source_id"].astype("int64").to_numpy()
        global_ref_sky = SkyCoord(master_df["ra_deg"].to_numpy(float) * u.deg,
                                  master_df["dec_deg"].to_numpy(float) * u.deg,
                                  frame="icrs")
        id_to_index = {int(sid): i for i, sid in enumerate(global_ref_ids)}
        self._ref_cache["global"] = (global_ref_ids, global_ref_sky)
        self._log(f"[IDMATCH] ref_filter={self.ref_filter} n_ref_global={len(global_ref_ids)}")
        if self.ref_per_date:
            self._log("[IDMATCH] per-date reference enabled")

        filter_frames: Dict[str, List[str]] = {}
        frame_stats = []

        n_ref = len(global_ref_ids)
        match_count = np.zeros(n_ref, dtype=int)
        sum_x = np.zeros(n_ref, dtype=float)
        sum_y = np.zeros(n_ref, dtype=float)
        sum_x2 = np.zeros(n_ref, dtype=float)
        sum_y2 = np.zeros(n_ref, dtype=float)

        total = len(self.file_list)
        for k, fname in enumerate(self.file_list, 1):
            if self._stop_requested:
                break

            date_key = _extract_date_key(fname, self.params)
            meta = self._load_meta(fname)
            filt = str(meta.get("filter", "") or "").strip().lower() or "unknown"
            filter_frames.setdefault(filt, []).append(fname)

            det_xy = self._load_detect_xy(fname)
            n_det = len(det_xy)
            fwhm_px = _safe_float(meta.get("fwhm_px"), np.nan)
            if not np.isfinite(fwhm_px) or fwhm_px <= 0:
                fwhm_px = float(getattr(self.params.P, "fwhm_pix_guess", 6.0))

            wcs = self._load_wcs_for_frame(fname)
            wcs_ok = wcs is not None
            pix_scale = self._pixscale_arcsec(wcs) if wcs_ok else np.nan

            ref_ids, ref_sky = self._get_ref_for_date(date_key, master_df)

            if not wcs_ok or n_det == 0:
                if n_det > 0:
                    df_out = pd.DataFrame({
                        "det_idx": np.arange(n_det, dtype=int),
                        "x": det_xy[:, 0],
                        "y": det_xy[:, 1],
                        "ra_deg": np.nan,
                        "dec_deg": np.nan,
                        "source_id": np.nan,
                        "sep_arcsec": np.nan,
                        "file": fname,
                        "filter": filt,
                    })
                else:
                    df_out = pd.DataFrame(columns=[
                        "det_idx", "x", "y", "ra_deg", "dec_deg",
                        "source_id", "sep_arcsec", "file", "filter",
                    ])

                out_sub = out_dir / date_key if date_key else out_dir
                out_sub.mkdir(parents=True, exist_ok=True)
                df_out.to_csv(out_sub / f"idmatch_{fname}.csv", index=False)

                stat = {
                    "file": fname,
                    "filter": filt,
                    "date_key": date_key,
                    "n_det": int(n_det),
                    "n_match": 0,
                    "n_pairs": 0,
                    "n_unmatched": int(n_det),  # All detections unmatched
                    "match_rate": 0.0,
                    "match_radius_arcsec": float(match_radius_arcsec) if np.isfinite(match_radius_arcsec) else np.nan,
                    "sep_med_arcsec": np.nan,
                    "sep_p90_arcsec": np.nan,
                    "dup_rate": np.nan,
                    "dup_count": 0,
                    "dx_med_px": np.nan,
                    "dy_med_px": np.nan,
                    "dx_rms_px": np.nan,
                    "dy_rms_px": np.nan,
                    "wcs_ok": bool(wcs_ok),
                    "pix_scale_arcsec": float(pix_scale) if np.isfinite(pix_scale) else np.nan,
                }
                self._log(
                    f"[IDMATCH] {fname}: wcs_ok={wcs_ok} det={n_det} match=0 "
                    f"rate=0.000 match_r={stat['match_radius_arcsec']:.2f}"
                )
                frame_stats.append(stat)
                self.file_done.emit(fname, stat)
                self.progress.emit(k, total, fname)
                continue

            try:
                ra, dec = wcs.all_pix2world(det_xy[:, 0], det_xy[:, 1], 0)
                det_sky = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
            except Exception as e:
                self._log(f"[IDMATCH] {fname}: WCS conversion failed: {e}")
                ra = np.full(n_det, np.nan)
                dec = np.full(n_det, np.nan)
                det_sky = None

            if det_sky is None:
                match_r = float(match_radius_arcsec) if np.isfinite(match_radius_arcsec) else np.nan
                df_out = pd.DataFrame({
                    "det_idx": np.arange(n_det, dtype=int),
                    "x": det_xy[:, 0],
                    "y": det_xy[:, 1],
                    "ra_deg": ra,
                    "dec_deg": dec,
                    "source_id": np.nan,
                    "sep_arcsec": np.nan,
                    "file": fname,
                    "filter": filt,
                })
                n_pairs = 0
                dx_med = np.nan
                dy_med = np.nan
                dx_rms = np.nan
                dy_rms = np.nan
            else:
                # Determine match radii
                if np.isfinite(match_radius_arcsec) and match_radius_arcsec > 0:
                    match_r = float(match_radius_arcsec)
                else:
                    match_r = float(self.match_r_fwhm * fwhm_px * pix_scale) if np.isfinite(pix_scale) and pix_scale > 0 else float(self.match_r_fwhm * fwhm_px)
                match_r = max(0.5, match_r)

                idx, sep2d, _ = det_sky.match_to_catalog_sky(ref_sky)
                sep_arcsec = sep2d.arcsec

                # Two-pass matching if enabled
                if self.two_pass_enable:
                    tight_r = self.tight_radius_arcsec
                    loose_r = self.loose_radius_arcsec

                    # Pass 1: Tight radius (high confidence)
                    ok_tight = sep_arcsec <= tight_r
                    best_det = {}
                    best_sep = {}
                    best_confidence = {}
                    for det_i, ref_i in enumerate(idx):
                        if not ok_tight[det_i]:
                            continue
                        ref_i = int(ref_i)
                        sep_val = float(sep_arcsec[det_i])
                        if ref_i not in best_sep or sep_val < best_sep[ref_i]:
                            best_sep[ref_i] = sep_val
                            best_det[ref_i] = det_i
                            best_confidence[ref_i] = "high"

                    # Track matched detections and refs
                    matched_det_set = set(best_det.values())
                    matched_ref_set = set(best_det.keys())

                    # Pass 2: Loose radius for unmatched detections
                    ok_loose = (sep_arcsec <= loose_r) & (sep_arcsec > tight_r)
                    for det_i, ref_i in enumerate(idx):
                        if det_i in matched_det_set:
                            continue  # Already matched in pass 1
                        if not ok_loose[det_i]:
                            continue
                        ref_i = int(ref_i)
                        if ref_i in matched_ref_set:
                            continue  # Ref already claimed in pass 1
                        sep_val = float(sep_arcsec[det_i])
                        if ref_i not in best_sep or sep_val < best_sep[ref_i]:
                            best_sep[ref_i] = sep_val
                            best_det[ref_i] = det_i
                            best_confidence[ref_i] = "low"

                    # Use loose radius for reporting
                    match_r = loose_r
                    ok = sep_arcsec <= loose_r
                else:
                    # Single-pass matching (original behavior)
                    ok = sep_arcsec <= match_r
                    best_det = {}
                    best_sep = {}
                    best_confidence = {}
                    for det_i, ref_i in enumerate(idx):
                        if not ok[det_i]:
                            continue
                        ref_i = int(ref_i)
                        sep_val = float(sep_arcsec[det_i])
                        if ref_i not in best_sep or sep_val < best_sep[ref_i]:
                            best_sep[ref_i] = sep_val
                            best_det[ref_i] = det_i
                            best_confidence[ref_i] = "high"

                n_pairs = len(best_det)
                dx_med = np.nan
                dy_med = np.nan
                dx_rms = np.nan
                dy_rms = np.nan
                if n_pairs > 0 and wcs_ok:
                    try:
                        ref_idx = np.fromiter(best_det.keys(), dtype=int, count=n_pairs)
                        det_idx = np.fromiter(best_det.values(), dtype=int, count=n_pairs)
                        ref_coords = ref_sky[ref_idx]
                        x_ref, y_ref = wcs.all_world2pix(
                            ref_coords.ra.deg,
                            ref_coords.dec.deg,
                            0,
                        )
                        dx = det_xy[det_idx, 0] - np.asarray(x_ref)
                        dy = det_xy[det_idx, 1] - np.asarray(y_ref)
                        dx_med = float(np.nanmedian(dx)) if len(dx) else np.nan
                        dy_med = float(np.nanmedian(dy)) if len(dy) else np.nan
                        dx_rms = float(np.sqrt(np.nanmean(dx ** 2))) if len(dx) else np.nan
                        dy_rms = float(np.sqrt(np.nanmean(dy ** 2))) if len(dy) else np.nan
                    except Exception:
                        dx_med = np.nan
                        dy_med = np.nan
                        dx_rms = np.nan
                        dy_rms = np.nan

                source_id = np.full(n_det, np.nan)
                sep_out = np.full(n_det, np.nan)
                confidence_out = np.array([""] * n_det, dtype=object)
                for ref_i, det_i in best_det.items():
                    sid = int(ref_ids[ref_i])
                    source_id[det_i] = sid
                    sep_out[det_i] = best_sep[ref_i]
                    confidence_out[det_i] = best_confidence.get(ref_i, "")
                    global_i = id_to_index.get(sid)
                    if global_i is None:
                        continue
                    match_count[global_i] += 1
                    sum_x[global_i] += det_xy[det_i, 0]
                    sum_y[global_i] += det_xy[det_i, 1]
                    sum_x2[global_i] += det_xy[det_i, 0] ** 2
                    sum_y2[global_i] += det_xy[det_i, 1] ** 2

                df_out = pd.DataFrame({
                    "det_idx": np.arange(n_det, dtype=int),
                    "x": det_xy[:, 0],
                    "y": det_xy[:, 1],
                    "ra_deg": ra,
                    "dec_deg": dec,
                    "source_id": source_id,
                    "sep_arcsec": sep_out,
                    "match_confidence": confidence_out,
                    "file": fname,
                    "filter": filt,
                })

            out_sub = out_dir / date_key if date_key else out_dir
            out_sub.mkdir(parents=True, exist_ok=True)
            df_out.to_csv(out_sub / f"idmatch_{fname}.csv", index=False)

            n_match = int(np.isfinite(df_out["source_id"]).sum())
            match_rate = float(n_match / n_det) if n_det > 0 else 0.0
            sep_vals = df_out["sep_arcsec"].to_numpy(float)
            sep_vals = sep_vals[np.isfinite(sep_vals)]
            sep_med = float(np.nanmedian(sep_vals)) if len(sep_vals) else np.nan
            sep_p90 = float(np.nanpercentile(sep_vals, 90)) if len(sep_vals) else np.nan

            dup_rate = np.nan
            dup_count = 0
            if det_sky is not None:
                try:
                    counts = pd.Series(idx[ok]).value_counts()
                    dup = counts[counts > 1].sum()
                    dup_count = int(dup)
                    dup_rate = float(dup / max(n_match, 1))
                except Exception:
                    dup_rate = np.nan
                    dup_count = 0

            # Count unmatched detections
            n_unmatched = n_det - n_match

            stat = {
                "file": fname,
                "filter": filt,
                "date_key": date_key,
                "n_det": int(n_det),
                "n_match": int(n_match),
                "n_pairs": int(n_pairs),
                "n_unmatched": int(n_unmatched),
                "match_rate": float(match_rate),
                "match_radius_arcsec": float(match_r),
                "sep_med_arcsec": sep_med,
                "sep_p90_arcsec": sep_p90,
                "dup_rate": dup_rate,
                "dup_count": int(dup_count),
                "dx_med_px": dx_med,
                "dy_med_px": dy_med,
                "dx_rms_px": dx_rms,
                "dy_rms_px": dy_rms,
                "wcs_ok": bool(wcs_ok),
                "pix_scale_arcsec": float(pix_scale) if np.isfinite(pix_scale) else np.nan,
            }
            self._log(
                f"[IDMATCH] {fname}: wcs_ok={wcs_ok} det={n_det} match={n_match} "
                f"rate={match_rate:.3f} sep_med={sep_med:.2f} match_r={match_r:.2f}"
            )
            frame_stats.append(stat)
            self.file_done.emit(fname, stat)
            self.progress.emit(k, total, fname)

        # Write summary files
        try:
            stats_df = pd.DataFrame(frame_stats)
            stats_df.to_csv(out_dir / "step7_frame_stats.csv", index=False)
        except Exception:
            pass

        try:
            (out_dir / "step7_filter_frames.json").write_text(
                json.dumps(filter_frames, indent=2), encoding="utf-8"
            )
        except Exception:
            pass

        try:
            master_rows = []
            for i, sid in enumerate(global_ref_ids):
                n_frames = match_count[i]
                if n_frames <= 0:
                    continue
                if n_frames > 0:
                    mean_x = sum_x[i] / n_frames
                    mean_y = sum_y[i] / n_frames
                    var_x = max(0.0, (sum_x2[i] / n_frames) - mean_x ** 2)
                    var_y = max(0.0, (sum_y2[i] / n_frames) - mean_y ** 2)
                else:
                    mean_x = np.nan
                    mean_y = np.nan
                    var_x = np.nan
                    var_y = np.nan
                master_rows.append({
                    "source_id": int(sid),
                    "n_frames": int(n_frames),
                    "ra_deg": float(global_ref_sky[i].ra.deg),
                    "dec_deg": float(global_ref_sky[i].dec.deg),
                    "x_mean": float(mean_x),
                    "y_mean": float(mean_y),
                    "x_std": float(np.sqrt(var_x)) if np.isfinite(var_x) else np.nan,
                    "y_std": float(np.sqrt(var_y)) if np.isfinite(var_y) else np.nan,
                })
            pd.DataFrame(master_rows).to_csv(out_dir / "step7_master_sources.csv", index=False)
        except Exception:
            pass

        # Compute summary statistics
        miss_ref_count = int(np.sum(match_count == 0))
        matched_ref_count = int(np.sum(match_count > 0))
        total_det_count = sum(s.get("n_det", 0) for s in frame_stats)
        total_match_count = sum(s.get("n_match", 0) for s in frame_stats)
        total_unmatched_count = sum(s.get("n_unmatched", 0) for s in frame_stats)
        total_dup_count = sum(s.get("dup_count", 0) for s in frame_stats)

        # Log summary statistics
        self._log("=" * 60)
        self._log(f"[IDMATCH][STATS] n_ref={n_ref} matched_refs={matched_ref_count} miss_ref_count={miss_ref_count}")
        self._log(f"[IDMATCH][STATS] total_det={total_det_count} total_match={total_match_count} total_unmatched={total_unmatched_count}")
        self._log(f"[IDMATCH][STATS] total_dup_count={total_dup_count}")

        summary = {
            "total": len(frame_stats),
            "ok": int(np.sum([1 for s in frame_stats if s.get("n_match", 0) > 0])),
            "n_ref": n_ref,
            "matched_ref_count": matched_ref_count,
            "miss_ref_count": miss_ref_count,
            "total_det_count": total_det_count,
            "total_match_count": total_match_count,
            "total_unmatched_count": total_unmatched_count,
            "total_dup_count": total_dup_count,
        }

        # Write summary metadata
        try:
            meta = {
                "n_frames": len(frame_stats),
                "n_frames_with_matches": int(np.sum([1 for s in frame_stats if s.get("n_match", 0) > 0])),
                "n_ref": n_ref,
                "matched_ref_count": matched_ref_count,
                "miss_ref_count": miss_ref_count,
                "total_det_count": total_det_count,
                "total_match_count": total_match_count,
                "total_unmatched_count": total_unmatched_count,
                "total_dup_count": total_dup_count,
                "ref_filter": self.ref_filter,
                "ref_frame": self.ref_frame,
                "ref_per_date": self.ref_per_date,
                "two_pass_enable": self.two_pass_enable,
                "tight_radius_arcsec": self.tight_radius_arcsec,
                "loose_radius_arcsec": self.loose_radius_arcsec,
                "match_r_fwhm": self.match_r_fwhm,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            (out_dir / "step7_idmatch_meta.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )
        except Exception:
            pass

        self.finished.emit(summary)

class StarIdMatchingWindow(StepWindowBase):
    """Step 7: Star ID Matching (WCS-based)"""

    def __init__(self, params, file_manager, project_state, main_window):
        self.file_manager = file_manager
        self.worker = None
        self.results = {}
        self.log_window = None

        super().__init__(
            step_index=6,
            step_name="Star ID Matching",
            params=params,
            project_state=project_state,
            main_window=main_window,
        )

        self.setup_step_ui()
        self.restore_state()

    def setup_step_ui(self):
        info = QLabel(
            "WCS-based matching using the fixed ref catalog (RA/Dec).\n"
            "Each frame is matched in sky coordinates within the arcsec radius."
        )
        info.setStyleSheet("QLabel { background-color: #E3F2FD; padding: 10px; border-radius: 5px; }")
        info.setWordWrap(True)
        self.content_layout.addWidget(info)

        control_layout = QHBoxLayout()

        btn_params = QPushButton("Matching Parameters")
        btn_params.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; font-weight: bold; padding: 8px 15px; }"
        )
        btn_params.clicked.connect(self.open_parameters_dialog)
        control_layout.addWidget(btn_params)

        self.btn_run = QPushButton("Run ID Matching")
        self.btn_run.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px 20px; }"
        )
        self.btn_run.clicked.connect(self.run_idmatch)
        control_layout.addWidget(self.btn_run)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setStyleSheet(
            "QPushButton { background-color: #f44336; color: white; font-weight: bold; padding: 8px 15px; }"
        )
        self.btn_stop.clicked.connect(self.stop_idmatch)
        self.btn_stop.setEnabled(False)
        control_layout.addWidget(self.btn_stop)

        control_layout.addStretch()
        btn_log = QPushButton("Show Log")
        btn_log.setStyleSheet(
            "QPushButton { background-color: #607D8B; color: white; font-weight: bold; padding: 8px 15px; }"
        )
        btn_log.clicked.connect(self.show_log_window)
        control_layout.addWidget(btn_log)

        self.content_layout.addLayout(control_layout)

        progress_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        self.progress_label = QLabel("Ready")
        self.progress_label.setMinimumWidth(300)
        progress_layout.addWidget(self.progress_label)
        self.content_layout.addLayout(progress_layout)

        self.tabs = QTabWidget()

        frames_tab = QWidget()
        frames_layout = QVBoxLayout(frames_tab)
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels([
            "File", "Filter", "N_det", "N_match", "Rate"
        ])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.results_table.setMinimumHeight(150)
        frames_layout.addWidget(self.results_table)
        self.tabs.addTab(frames_tab, "Frames")

        summary_tab = QWidget()
        summary_layout = QVBoxLayout(summary_tab)
        self.stats_table = QTableWidget()
        self.stats_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.stats_table.setMinimumHeight(150)
        summary_layout.addWidget(self.stats_table)
        self.tabs.addTab(summary_tab, "Stats")

        plot_tab = QWidget()
        plot_layout = QVBoxLayout(plot_tab)
        plot_controls = QHBoxLayout()
        plot_controls.addWidget(QLabel("Date:"))
        self.plot_date_combo = QComboBox()
        self.plot_date_combo.addItem("All")
        self.plot_date_combo.currentIndexChanged.connect(self._on_plot_date_changed)
        plot_controls.addWidget(self.plot_date_combo)
        plot_controls.addStretch()
        plot_layout.addLayout(plot_controls)
        self.plot_canvas = FigureCanvas(Figure(figsize=(8, 4)))
        self.plot_canvas.setMinimumHeight(260)
        plot_layout.addWidget(self.plot_canvas)
        self.tabs.addTab(plot_tab, "Plot")

        self.content_layout.addWidget(self.tabs)

        self.log_window = QWidget(self, Qt.Window)
        self.log_window.setWindowTitle("ID Match Log")
        self.log_window.resize(900, 500)
        log_layout = QVBoxLayout(self.log_window)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("QTextEdit { font-family: monospace; font-size: 9pt; }")
        log_layout.addWidget(self.log_text)

    def open_parameters_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("ID Match Parameters")
        dialog.resize(460, 240)

        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        layout.addLayout(form)

        arcsec_spin = QDoubleSpinBox()
        arcsec_spin.setRange(0.0, 30.0)
        arcsec_spin.setDecimals(2)
        arcsec_spin.setValue(float(getattr(self.params.P, "idmatch_tol_arcsec", 2.0) or 0.0))
        form.addRow("Sky match radius (arcsec, 0=auto):", arcsec_spin)

        match_r_spin = QDoubleSpinBox()
        match_r_spin.setRange(0.1, 5.0)
        match_r_spin.setDecimals(2)
        match_r_spin.setValue(float(getattr(self.params.P, "idmatch_match_r_fwhm", 0.8)))
        form.addRow("Fallback radius (FWHM x):", match_r_spin)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec_() == QDialog.Accepted:
            self.params.P.idmatch_tol_arcsec = arcsec_spin.value()
            self.params.P.idmatch_match_r_fwhm = match_r_spin.value()
            self.persist_params()
            QMessageBox.information(dialog, "Success", "Parameters saved!")

    def run_idmatch(self):
        if self.worker and self.worker.isRunning():
            return

        step6_out = step6_dir(self.params.P.result_dir)
        meta_path = step6_out / "ref_build_meta.json"
        if not meta_path.exists():
            meta_path = legacy_step5_refbuild_dir(self.params.P.result_dir) / "ref_build_meta.json"
        if not meta_path.exists():
            meta_path = legacy_step7_refbuild_dir(self.params.P.result_dir) / "ref_build_meta.json"
        if not meta_path.exists():
            QMessageBox.warning(self, "Missing Reference", "Run Reference Build first.")
            return
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        ref_frame = meta.get("ref_frame")
        ref_filter = meta.get("ref_filter")
        ref_per_date = bool(meta.get("ref_per_date", False))
        if not ref_frame or not ref_filter:
            QMessageBox.warning(self, "Missing Reference", "Reference info not found in ref_build_meta.json")
            return
        if bool(getattr(self.params.P, "ref_per_date", False)) and not ref_per_date:
            self.log("[WARN] Per-date ref is ON in parameters, but ref_build_meta.json is global. Re-run Step 6.")

        files = list(self.file_manager.filenames) if self.file_manager else []
        if not files and self.file_manager:
            try:
                files = list(self.file_manager.scan_files())
            except Exception as e:
                QMessageBox.warning(self, "Warning", f"No frames found: {e}")
                return
        if not files:
            QMessageBox.warning(self, "Warning", "No frames found")
            return

        cache_dir = Path(self.params.P.cache_dir)
        step4_out = step4_dir(self.params.P.result_dir)
        available = [
            f for f in files
            if (cache_dir / f"detect_{f}.csv").exists()
            or (step4_out / f"detect_{f}.csv").exists()
        ]
        missing = [f for f in files if f not in available]
        if missing:
            self.log(f"[WARN] Missing detection cache for {len(missing)} frames; skipping.")
        files = available
        if not files:
            QMessageBox.warning(self, "Warning", "No frames with detection cache. Run Source Detection first.")
            return

        # Get two-pass matching parameters from config
        idmatch_cfg = getattr(self.params.P, "idmatch", None)
        two_pass_enable = bool(getattr(idmatch_cfg, "two_pass_enable", True)) if idmatch_cfg else True
        tight_radius_arcsec = float(getattr(idmatch_cfg, "tight_radius_arcsec", 1.0)) if idmatch_cfg else 1.0
        loose_radius_arcsec = float(getattr(idmatch_cfg, "loose_radius_arcsec", 3.0)) if idmatch_cfg else 3.0

        self.worker = IdMatchWorker(
            file_list=files,
            params=self.params,
            data_dir=self.params.P.data_dir,
            result_dir=self.params.P.result_dir,
            cache_dir=self.params.P.cache_dir,
            ref_frame=str(ref_frame),
            ref_filter=str(ref_filter),
            match_r_fwhm=float(getattr(self.params.P, "idmatch_match_r_fwhm", 0.8)),
            init_r_fwhm=float(getattr(self.params.P, "idmatch_init_r_fwhm", 5.0)),
            ratio_max=float(getattr(self.params.P, "idmatch_ratio_max", 0.7)),
            min_pairs=int(getattr(self.params.P, "idmatch_min_pairs", 15)),
            transform_mode=str(getattr(self.params.P, "idmatch_transform_mode", "similarity")),
            mutual_nearest=bool(getattr(self.params.P, "idmatch_mutual_nearest", True)),
            ref_per_date=ref_per_date,
            two_pass_enable=two_pass_enable,
            tight_radius_arcsec=tight_radius_arcsec,
            loose_radius_arcsec=loose_radius_arcsec,
        )
        self.worker.progress.connect(self.on_progress)
        self.worker.log.connect(self.log)
        self.worker.file_done.connect(self.on_file_done)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)

        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(len(files))
        self.progress_label.setText(f"0/{len(files)} | Starting...")
        self.worker.start()
        self.show_log_window()

    def stop_idmatch(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()

    def on_progress(self, current, total, filename):
        self.progress_bar.setValue(current)
        self.progress_label.setText(f"{current}/{total} | {filename}")

    def on_file_done(self, filename, stat):
        self.results[filename] = stat
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)
        self.results_table.setItem(row, 0, QTableWidgetItem(filename))
        self.results_table.setItem(row, 1, QTableWidgetItem(str(stat.get("filter", ""))))
        self.results_table.setItem(row, 2, QTableWidgetItem(str(stat.get("n_det", 0))))
        self.results_table.setItem(row, 3, QTableWidgetItem(str(stat.get("n_match", 0))))
        rate = stat.get("match_rate", 0.0)
        self.results_table.setItem(row, 4, QTableWidgetItem(f"{rate:.3f}"))

    def on_error(self, filename, error):
        self.log(f"ERROR {filename}: {error}")

    def on_finished(self, summary):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_label.setText("Done")
        self._update_stats_summary()
        self._update_plot_tab()
        self.save_state(summary)
        self.update_navigation_buttons()

    def _update_stats_summary(self):
        if not self.results:
            self.stats_table.setRowCount(0)
            self.stats_table.setColumnCount(0)
            return
        try:
            df = pd.DataFrame(self.results.values())
        except Exception:
            self.stats_table.setRowCount(0)
            self.stats_table.setColumnCount(0)
            return

        if "filter" not in df.columns:
            df["filter"] = ""
        group = df.groupby("filter", dropna=False)
        rows = []
        for flt, g in group:
            match_rate = pd.to_numeric(g.get("match_rate"), errors="coerce")
            sep_med = pd.to_numeric(g.get("sep_med_arcsec"), errors="coerce")
            dup_rate = pd.to_numeric(g.get("dup_rate"), errors="coerce")
            wcs_ok = g.get("wcs_ok")
            wcs_ok_n = int(np.sum(wcs_ok == True)) if wcs_ok is not None else 0
            rows.append({
                "filter": str(flt),
                "n_frames": int(len(g)),
                "wcs_ok": wcs_ok_n,
                "match_rate_med": float(match_rate.median()) if match_rate is not None else np.nan,
                "match_rate_p90": float(match_rate.quantile(0.9)) if match_rate is not None else np.nan,
                "sep_med_med": float(sep_med.median()) if sep_med is not None else np.nan,
                "dup_rate_med": float(dup_rate.median()) if dup_rate is not None else np.nan,
            })

        cols = ["filter", "n_frames", "wcs_ok", "match_rate_med", "match_rate_p90", "sep_med_med", "dup_rate_med"]
        self.stats_table.setColumnCount(len(cols))
        self.stats_table.setRowCount(len(rows))
        self.stats_table.setHorizontalHeaderLabels(cols)
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.stats_table.horizontalHeader().setStretchLastSection(True)

        for r, row in enumerate(rows):
            for c, col in enumerate(cols):
                val = row.get(col, "")
                if isinstance(val, float):
                    text = f"{val:.3f}" if np.isfinite(val) else ""
                else:
                    text = str(val)
                self.stats_table.setItem(r, c, QTableWidgetItem(text))

    def _update_plot_tab(self) -> None:
        if not hasattr(self, "plot_canvas") or self.plot_canvas is None:
            return
        fig = self.plot_canvas.figure
        fig.clear()
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        stats_path = step7_dir(self.params.P.result_dir) / "step7_frame_stats.csv"
        if not stats_path.exists():
            ax1.text(0.5, 0.5, "No ID match stats available", ha="center", va="center")
            ax2.axis("off")
            self.plot_canvas.draw_idle()
            return
        try:
            df = pd.read_csv(stats_path)
        except Exception:
            ax1.text(0.5, 0.5, "Failed to read ID match stats", ha="center", va="center")
            ax2.axis("off")
            self.plot_canvas.draw_idle()
            return
        if df.empty:
            ax1.text(0.5, 0.5, "No ID match stats available", ha="center", va="center")
            ax2.axis("off")
            self.plot_canvas.draw_idle()
            return

        if "date_key" in df.columns and hasattr(self, "plot_date_combo"):
            dates = sorted(df["date_key"].fillna("unknown_date").astype(str).unique().tolist())
            items = ["All"] + dates
            prev = self.plot_date_combo.currentText() if self.plot_date_combo.count() else "All"
            self.plot_date_combo.blockSignals(True)
            self.plot_date_combo.clear()
            self.plot_date_combo.addItems(items)
            if prev in items:
                self.plot_date_combo.setCurrentText(prev)
            else:
                self.plot_date_combo.setCurrentIndex(0)
            self.plot_date_combo.blockSignals(False)

        selected_date = None
        if hasattr(self, "plot_date_combo"):
            text = self.plot_date_combo.currentText()
            if text and text != "All":
                selected_date = text

        if "match_rate" in df.columns:
            df["match_rate"] = pd.to_numeric(df["match_rate"], errors="coerce")
        else:
            df["match_rate"] = np.nan
        if "sep_med_arcsec" in df.columns:
            df["sep_med_arcsec"] = pd.to_numeric(df["sep_med_arcsec"], errors="coerce")
        else:
            df["sep_med_arcsec"] = np.nan
        if "dx_med_px" in df.columns:
            df["dx_med_px"] = pd.to_numeric(df["dx_med_px"], errors="coerce")
        else:
            df["dx_med_px"] = np.nan
        if "dy_med_px" in df.columns:
            df["dy_med_px"] = pd.to_numeric(df["dy_med_px"], errors="coerce")
        else:
            df["dy_med_px"] = np.nan
        if "wcs_ok" in df.columns:
            df["wcs_ok"] = df["wcs_ok"].astype(bool)

        df_plot = df
        if selected_date and "date_key" in df.columns:
            df_plot = df[df["date_key"].astype(str) == str(selected_date)]
            if df_plot.empty:
                df_plot = df

        score = df_plot["match_rate"] / (df_plot["sep_med_arcsec"].abs() + 1e-3)
        best_idx = score.idxmax() if score.notna().any() else None

        # Plot 1: match rate vs sep
        ax1.scatter(
            df_plot["sep_med_arcsec"],
            df_plot["match_rate"],
            s=28,
            alpha=0.75,
            color="#1E88E5",
            edgecolors="none",
        )
        ax1.set_title("Match Rate vs Sep (arcsec)")
        ax1.set_xlabel("Sep med (arcsec)")
        ax1.set_ylabel("Match rate")
        ax1.grid(True, alpha=0.2)

        # Plot 2: dx/dy median
        ax2.axhline(0, color="#90A4AE", lw=1)
        ax2.axvline(0, color="#90A4AE", lw=1)
        ax2.scatter(
            df_plot["dx_med_px"],
            df_plot["dy_med_px"],
            s=28,
            alpha=0.75,
            color="#43A047",
            edgecolors="none",
        )
        ax2.set_title("Median Δx/Δy (px)")
        ax2.set_xlabel("dx_med (px)")
        ax2.set_ylabel("dy_med (px)")
        ax2.grid(True, alpha=0.2)

        if best_idx is not None and best_idx in df_plot.index:
            r = df_plot.loc[best_idx]
            ax1.scatter(r["sep_med_arcsec"], r["match_rate"],
                        s=140, marker="*", color="#FF5252", edgecolors="#212121", linewidths=0.8, zorder=5)
            ax2.scatter(r["dx_med_px"], r["dy_med_px"],
                        s=140, marker="*", color="#FF5252", edgecolors="#212121", linewidths=0.8, zorder=5)
            label = str(r.get("file", "best"))
            ax1.text(r["sep_med_arcsec"], r["match_rate"], label,
                     fontsize=8, ha="left", va="bottom", color="#D32F2F")

        fig.tight_layout()
        self.plot_canvas.draw_idle()

    def _on_plot_date_changed(self, index: int) -> None:
        if index < 0:
            return
        self._update_plot_tab()

    def validate_step(self) -> bool:
        out_dir = step7_dir(self.params.P.result_dir)
        return (out_dir / "step7_frame_stats.csv").exists()

    def save_state(self, summary: Optional[dict] = None):
        if summary is None:
            summary = self.results if isinstance(self.results, dict) else {}
        summary = summary or {}

        state_data = {
            "idmatch_complete": bool(summary),
            "n_files": summary.get("total", 0) if summary else 0,
            "match_radius_arcsec": float(getattr(self.params.P, "idmatch_tol_arcsec", 0.0) or 0.0),
            "match_r_fwhm": float(getattr(self.params.P, "idmatch_match_r_fwhm", 0.8)),
            "init_r_fwhm": float(getattr(self.params.P, "idmatch_init_r_fwhm", 5.0)),
            "ratio_max": float(getattr(self.params.P, "idmatch_ratio_max", 0.7)),
            "min_pairs": int(getattr(self.params.P, "idmatch_min_pairs", 15)),
            "transform_mode": str(getattr(self.params.P, "idmatch_transform_mode", "similarity")),
            "mutual_nearest": bool(getattr(self.params.P, "idmatch_mutual_nearest", True)),
        }
        self.project_state.store_step_data("star_id_match", state_data)

        self.persist_params()

    def restore_state(self):
        state = self.project_state.get_step_data("star_id_match")
        if not state:
            return
        if "match_r_fwhm" in state:
            self.params.P.idmatch_match_r_fwhm = float(state["match_r_fwhm"])
        if "match_radius_arcsec" in state:
            self.params.P.idmatch_tol_arcsec = float(state["match_radius_arcsec"])
        if "init_r_fwhm" in state:
            self.params.P.idmatch_init_r_fwhm = float(state["init_r_fwhm"])
        if "ratio_max" in state:
            self.params.P.idmatch_ratio_max = float(state["ratio_max"])
        if "min_pairs" in state:
            self.params.P.idmatch_min_pairs = int(state["min_pairs"])
        if "transform_mode" in state:
            mode = str(state["transform_mode"]).lower()
            if mode in ("shift", "affine", "similarity"):
                self.params.P.idmatch_transform_mode = mode
        if "mutual_nearest" in state:
            self.params.P.idmatch_mutual_nearest = bool(state["mutual_nearest"])
        self._update_plot_tab()

    def log(self, msg: str):
        if self.log_text is not None:
            self.log_text.append(msg)

    def show_log_window(self):
        if self.log_window is not None:
            self.log_window.show()
            self.log_window.raise_()
