"""
Step 9: Forced Photometry (per-frame ID-matched XY)
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
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from .step_window_base import StepWindowBase
from .aperture_photometry_worker import ApertureWorker
from ...utils.step_paths import (
    step2_cropped_dir,
    step5_dir,
    step6_dir,
    step7_dir,
    step9_dir,
    crop_is_active,
    legacy_step5_refbuild_dir,
    legacy_step6_idmatch_dir,
    legacy_step7_wcs_dir,
    legacy_step7_refbuild_dir,
)
from ...utils.constants import get_parallel_workers
from ...utils.header_cache import HeaderCache


_DATE_RE = re.compile(r"(20\d{6})")


def _extract_date_key(filename: str) -> str:
    match = _DATE_RE.search(str(filename))
    return match.group(1) if match else ""


def _resolve_idmatch_path(result_dir: Path, cache_dir: Path, fname: str) -> Path:
    step7_out = step7_dir(result_dir)
    date_key = _extract_date_key(fname)
    if date_key:
        dated = step7_out / date_key / f"idmatch_{fname}.csv"
        if dated.exists():
            return dated
    direct = step7_out / f"idmatch_{fname}.csv"
    if direct.exists():
        return direct
    matches = list(step7_out.glob(f"*/idmatch_{fname}.csv"))
    if matches:
        return matches[0]
    legacy_dir = legacy_step6_idmatch_dir(result_dir)
    legacy_path = legacy_dir / f"idmatch_{fname}.csv"
    if legacy_path.exists():
        return legacy_path
    legacy_matches = list(legacy_dir.glob(f"*/idmatch_{fname}.csv"))
    if legacy_matches:
        return legacy_matches[0]
    cache_path = Path(cache_dir) / "idmatch" / f"idmatch_{fname}.csv"
    if cache_path.exists():
        return cache_path
    return direct


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


def _get_filter_lower(fits_path: Path, header_cache: HeaderCache = None, filename: str = None):
    """Get filter name from FITS file, using HeaderCache if available."""
    # Try HeaderCache first (uses headers.csv)
    if header_cache is not None and filename:
        filt = header_cache.get_filter(filename, fits_path)
        if filt != "unknown":
            return filt

    # Fallback to direct FITS read
    try:
        h = fits.getheader(fits_path)
        f = h.get("FILTER", None)
        if f is None:
            return "unknown"
        return str(f).strip().lower()
    except Exception:
        return "unknown"


def _normalize_filter_key(value: str | None) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _get_exptime_fallback(fits_path: Path, default=1.0, header_cache: HeaderCache = None, filename: str = None):
    """Get exposure time from FITS file, using HeaderCache if available."""
    # Try HeaderCache first (uses headers.csv)
    if header_cache is not None and filename:
        exptime = header_cache.get_exptime(filename, fits_path, default=default)
        if exptime != default:
            return exptime

    # Fallback to direct FITS read
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
    var_source = max(flux_e, 0.0)
    var_bkg_in_ap = ap_area * sigma_pix_e2
    var_bkg_est = (ap_area ** 2 / max(n_sky, 1)) * sigma_pix_e2
    var_readnoise = ap_area * rn_param_e ** 2 if sky_sigma_includes_rn else 0.0

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
        self.max_workers = get_parallel_workers(params)
        self._stop_requested = False
        self._write_lock = Lock()
        # HeaderCache for efficient metadata access (uses headers.csv from Step 1)
        self._header_cache = HeaderCache(result_dir, data_dir)

    def stop(self):
        self._stop_requested = True

    def _log(self, msg):
        self.log.emit(msg)

    def run(self):
        try:
            self._run_impl()
        except Exception as e:
            self._log(f"[ERROR] {e}\\n{traceback.format_exc()}")
            self.error.emit("WORKER", str(e))
            self.finished.emit({})

    def _run_impl(self):
        P = self.params.P
        try:
            result_dir = self.result_dir
            output_dir = step9_dir(result_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            cache_dir = self.cache_dir

            # Step 8 selection 로드 (final ID map 사용)
            step8_dir = result_dir / "step8_selection"
            target_sids = {}  # filter -> target source_id
            comp_sids = {}    # filter -> set of comparison source_ids
            selected_sids_by_filter: dict[str, set[int]] = {}
            all_source_ids = set()
            final_id_maps: dict[str, dict[int, int]] = {}

            def _load_step8_final_maps(step8_path: Path) -> dict[str, dict[int, int]]:
                maps: dict[str, dict[int, int]] = {}
                if not step8_path.exists():
                    return maps
                for mp in step8_path.glob("master_catalog_*.tsv"):
                    flt = _normalize_filter_key(mp.stem.replace("master_catalog_", ""))
                    if not flt:
                        continue
                    try:
                        df = pd.read_csv(mp, sep="\t")
                        if {"source_id", "ID"} <= set(df.columns):
                            sid = pd.to_numeric(df["source_id"], errors="coerce").dropna().astype("int64")
                            fid = pd.to_numeric(df["ID"], errors="coerce").dropna().astype("int64")
                            maps[flt] = dict(zip(sid.tolist(), fid.tolist()))
                    except Exception:
                        continue
                return maps

            if step8_dir.exists():
                final_id_maps = _load_step8_final_maps(step8_dir)
                for sel_file in step8_dir.glob("selection_*.json"):
                    try:
                        with open(sel_file, "r", encoding="utf-8") as f:
                            sel = json.load(f)
                        flt = _normalize_filter_key(sel.get("filter", sel_file.stem.replace("selection_", "")))
                        if not flt:
                            continue
                        tid = sel.get("target_source_id")
                        cids = sel.get("comparison_source_ids", [])
                        if tid is not None:
                            target_sids[flt] = int(tid)
                            all_source_ids.add(int(tid))
                        comp_sids[flt] = set(int(c) for c in cids if c is not None)
                        all_source_ids.update(comp_sids[flt])
                        selected = set(comp_sids[flt])
                        if tid is not None:
                            selected.add(int(tid))
                        if selected:
                            selected_sids_by_filter[flt] = selected
                    except Exception as e:
                        self._log(f"[WARN] Failed to load {sel_file.name}: {e}")

            if not all_source_ids:
                raise RuntimeError("No targets/comparisons found. Complete Step 8 first.")

            self._log(f"Loaded {len(target_sids)} target(s), {sum(len(v) for v in comp_sids.values())} comparison(s) from Step 8")
            self._log(f"Target source_ids: {target_sids}")
            self._log(f"All source_ids to photometer: {sorted(all_source_ids)}")

            # final ID map per filter
            sid2id_by_filter: dict[str, dict[int, int]] = {}
            for flt, sids in selected_sids_by_filter.items():
                fmap = final_id_maps.get(flt, {})
                if fmap:
                    sid2id_map = {sid: fmap.get(sid) for sid in sids if sid in fmap}
                    missing = sorted(set(sids) - set(sid2id_map))
                    if missing:
                        self._log(f"[WARN] Missing final IDs in Step8 master ({flt}): {missing[:5]}...")
                else:
                    sid2id_map = {}
                if not sid2id_map:
                    # fallback: sequential per-filter IDs
                    sorted_sids = sorted(sids)
                    sid2id_map = {sid: idx + 1 for idx, sid in enumerate(sorted_sids)}
                    if not fmap:
                        self._log(f"[WARN] Step8 master missing for '{flt}'. Using fallback IDs.")
                sid2id_by_filter[flt] = sid2id_map

            # master catalog compatibility file (for caching/deps)
            master_rows = []
            for flt, sid2id_map in sid2id_by_filter.items():
                for sid, fid in sorted(sid2id_map.items(), key=lambda x: x[1]):
                    master_rows.append({"filter": flt, "source_id": int(sid), "ID": int(fid)})
            master_path = output_dir / "pseudo_master.csv"
            if master_rows:
                try:
                    pd.DataFrame(master_rows).to_csv(master_path, index=False)
                except Exception as e:
                    self._log(f"[WARN] Failed to write pseudo master: {e}")

            ap_mode = str(getattr(P, "aperture_mode", getattr(P, "ap_mode", "apcorr"))).strip().lower()
            resume = bool(getattr(P, "resume_mode", True))
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
            max_recenter_shift = float(getattr(P, "max_recenter_shift", 2.0))
            centroid_outlier_px = float(getattr(P, "centroid_outlier_px", 1.0))
            sky_sigma_mode = str(getattr(P, "sky_sigma_mode", "local")).strip().lower()
            sky_sigma_includes_rn = bool(getattr(P, "sky_sigma_includes_rn", True))
            min_n_sky_for_local = int(getattr(P, "sky_sigma_min_n_sky", 50))

            self._log(
                "Start forced photometry | "
                f"frames={len(self.file_list)} | resume={resume} | force_rephot={force_rephot} | "
                f"ap_mode={ap_mode} | ZP={ZP} (ADU/sec) | gain={GAIN} e-/ADU | "
                f"min_snr={min_snr_for_mag} | use_cropped={self.use_cropped}"
            )

            ap_path = output_dir / "aperture_by_frame.csv"
            if not ap_path.exists():
                legacy_ap = result_dir / "aperture_by_frame.csv"
                if legacy_ap.exists():
                    ap_path = legacy_ap
            if not ap_path.exists():
                self._log(f"[INFO] aperture_by_frame.csv not found. Computing apertures for {len(self.file_list)} frames...")
                self._log("[INFO] This may take a while. Please wait...")
                self.progress.emit(0, len(self.file_list), "Computing apertures...")
                ap_worker = ApertureWorker(
                    self.file_list,
                    self.params,
                    self.data_dir,
                    self.result_dir,
                    self.cache_dir,
                    self.use_cropped,
                    output_dir=output_dir,  # Save directly to step9 output dir
                )
                ap_worker.run()
                self._log("[INFO] Aperture computation complete.")
            if not ap_path.exists():
                raise RuntimeError("aperture_by_frame.csv not found (auto-build failed)")

            df_ap = pd.read_csv(ap_path)
            apcorr_path = output_dir / "apcorr_summary.csv"
            if not apcorr_path.exists():
                legacy_apcorr = result_dir / "apcorr_summary.csv"
                if legacy_apcorr.exists():
                    apcorr_path = legacy_apcorr
            apcorr_df = pd.read_csv(apcorr_path) if apcorr_path.exists() else None

            # source_id -> final ID 매핑 (Step 8 master 기반)
            df_frame_map = None  # Reference Build 없이 동작 - idmatch_df에서 직접 좌표 읽음

            # Reference Build 파일이 있으면 추가로 로드 (선택사항)
            frame_map_path = step6_dir(result_dir) / "frame_sourceid_to_ID.tsv"
            if not frame_map_path.exists():
                frame_map_path = legacy_step5_refbuild_dir(result_dir) / "frame_sourceid_to_ID.tsv"
            if not frame_map_path.exists():
                frame_map_path = legacy_step7_refbuild_dir(result_dir) / "frame_sourceid_to_ID.tsv"
            if not frame_map_path.exists():
                frame_map_path = result_dir / "frame_sourceid_to_ID.tsv"
            if frame_map_path.exists():
                try:
                    df_frame_map = pd.read_csv(frame_map_path, sep="\t")
                except Exception:
                    df_frame_map = None

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
                    fq_path = legacy_step7_wcs_dir(result_dir) / "frame_quality.csv"
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

            def _load_frame_targets(fname, filt_key: str):
                sid2id_map = sid2id_by_filter.get(filt_key, {})
                if df_frame_map is not None and (not df_frame_map.empty):
                    cols = df_frame_map.columns
                    c_file = _pick_col(cols, ["file", "fname", "frame"])
                    c_id = _pick_col(cols, ["ID", "id"])
                    c_sid = _pick_col(cols, ["source_id", "sourceid", "sid"])
                    c_x = _pick_col(cols, ["x", "x_det", "x_pix", "x0"])
                    c_y = _pick_col(cols, ["y", "y_det", "y_pix", "y0"])
                    c_sep = _pick_col(cols, ["sep_arcsec", "sep", "dist_arcsec"])
                    if c_file and c_x and c_y:
                        sub = df_frame_map[df_frame_map[c_file].astype(str) == str(fname)].copy()
                        if len(sub):
                            if not c_sid:
                                return pd.DataFrame(columns=["ID", "x", "y", "source_id"])
                            out = pd.DataFrame({
                                "x": pd.to_numeric(sub[c_x], errors="coerce"),
                                "y": pd.to_numeric(sub[c_y], errors="coerce"),
                            })
                            out["source_id"] = pd.to_numeric(sub[c_sid], errors="coerce").astype("Int64")
                            if sid2id_map:
                                out["ID"] = out["source_id"].map(sid2id_map).astype("Int64")
                            if "ID" not in out.columns and c_id and not sid2id_map:
                                out["ID"] = pd.to_numeric(sub[c_id], errors="coerce").astype("Int64")
                            if c_sep:
                                out["sep_arcsec"] = pd.to_numeric(sub[c_sep], errors="coerce")
                            out = out.dropna(subset=["ID", "x", "y"])
                            return out[["ID", "x", "y", "source_id"]]

                # fallback: idmatch CSV + master map
                p = _resolve_idmatch_path(result_dir, cache_dir, fname)
                if p.exists():
                    try:
                        df = pd.read_csv(p)
                        if {"source_id", "x", "y"} <= set(df.columns):
                            df["source_id"] = pd.to_numeric(df["source_id"], errors="coerce").astype("Int64")
                            if not sid2id_map:
                                return pd.DataFrame(columns=["ID", "x", "y", "source_id"])
                            # 매칭 전 source_id 개수
                            n_before = len(df)
                            df["ID"] = df["source_id"].map(sid2id_map)
                            df = df.dropna(subset=["ID", "x", "y"])
                            # 첫 프레임만 로그
                            if not hasattr(self, '_logged_first_frame'):
                                self._log(f"[DEBUG] First frame {fname}: {n_before} detections, {len(df)} matched to selection")
                                if len(df) > 0:
                                    self._log(f"[DEBUG] Matched source_ids: {df['source_id'].tolist()[:10]}...")
                                self._logged_first_frame = True
                            return df[["ID", "x", "y", "source_id"]]
                    except Exception as e:
                        self._log(f"[WARN] Failed to load idmatch for {fname}: {e}")
                else:
                    if not hasattr(self, '_logged_missing_idmatch'):
                        self._log(f"[WARN] idmatch not found: {p}")
                        self._logged_missing_idmatch = True
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

            def _cached_counts(fname, out_tsv, filt_key):
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
                    targets = int(len(_load_frame_targets(fname, filt_key)))
                except Exception:
                    targets = 0
                n_fail = max(int(targets) - int(n), 0)
                return n, n_goodmag, n_fail, targets

            def _neighbor_mask(img, xc, yc, fwhm_used, xys, r_exclude=0.0):
                if xys.size == 0:
                    return None
                r = float(fwhm_used) * neigh_scale
                if not np.isfinite(r) or r <= 0:
                    return None
                dx = xys[:, 0] - xc
                dy = xys[:, 1] - yc
                rr = np.hypot(dx, dy)
                mask = (rr < r) & (rr > float(r_exclude))
                if not np.any(mask):
                    return None
                return xys[mask]

            def _mask_neighbors(cut, x0, y0, fwhm_used, xys):
                if xys is None or len(xys) == 0:
                    return None
                yy, xx = np.mgrid[:cut.shape[0], :cut.shape[1]]
                mask = np.zeros(cut.shape, dtype=bool)
                r2 = float(fwhm_used) ** 2
                for x, y in xys:
                    if not np.isfinite(x) or not np.isfinite(y):
                        continue
                    xc = x - x0
                    yc = y - y0
                    if xc < 0 or yc < 0 or xc >= cut.shape[1] or yc >= cut.shape[0]:
                        continue
                    rr2 = (xx - xc) ** 2 + (yy - yc) ** 2
                    mask |= rr2 < r2
                return mask

            def _cutout(img, xc, yc, r_out_val):
                h, w = img.shape
                pad = int(max(r_out_val + 5, 10))
                xi, yi = int(round(xc)), int(round(yc))
                x0, x1 = max(0, xi - pad), min(w, xi + pad + 1)
                y0, y1 = max(0, yi - pad), min(h, yi + pad + 1)
                cut = img[y0:y1, x0:x1]
                xc_cut = xc - x0
                yc_cut = yc - y0
                return cut, xc_cut, yc_cut, x0, y0

            def _clip_image(img, mask):
                if mask is None:
                    return img
                z = img.copy()
                z[mask] = np.nan
                return z

            def _draw_apertures(img, xc, yc, r_ap_val, r_in_val, r_out_val, mask=None):
                if mask is None:
                    return img
                z = img.copy()
                yy, xx = np.mgrid[:img.shape[0], :img.shape[1]]
                r2 = (xx - xc) ** 2 + (yy - yc) ** 2
                z[(r2 <= r_ap_val ** 2) & mask] = np.nan
                z[(r2 >= r_in_val ** 2) & (r2 <= r_out_val ** 2) & mask] = np.nan
                return z

            fail_csv = output_dir / "phot_forced_fail.tsv"
            debug_json = output_dir / "phot_forced_debug.json"
            _fail_rows_all = []
            debug_frames = []
            index_rows = []

            frames = list(self.file_list)
            total = len(frames)
            counters = {"cached": 0, "processed": 0, "no_targets": 0, "no_aperture": 0, "no_file": 0, "error": 0}
            completed_count = [0]

            def process_single_frame(fname):
                try:
                    if self._stop_requested:
                        return fname, None, None, None, "stopped"

                    if self.use_cropped:
                        cropped_dir = step2_cropped_dir(result_dir)
                        if not cropped_dir.exists():
                            cropped_dir = result_dir / "cropped"
                        fpath = cropped_dir / fname
                    else:
                        fpath = self.data_dir / fname
                        if not fpath.exists():
                            try:
                                fpath = Path(self.params.get_file_path(fname))
                            except Exception:
                                pass

                    out_tsv = output_dir / f"{fname}_photometry.tsv"
                    if not fpath.exists():
                        idx_row = dict(
                            file=fname, filter="unknown", n=0, n_goodmag=0, n_fail=0,
                            targets=0, path=str(out_tsv.name)
                        )
                        dbg_row = dict(file=fname, cached=False, targets=0, reason="file_not_found")
                        return fname, idx_row, dbg_row, [], "no_file"
                    deps = [fpath, master_path, ap_path]
                    idmatch_path = _resolve_idmatch_path(result_dir, cache_dir, fname)
                    if idmatch_path.exists():
                        deps.append(idmatch_path)

                    this_filter = _get_filter_lower(fpath, self._header_cache, fname)

                    if resume and (not force_rephot) and out_tsv.exists() and _is_up_to_date(out_tsv, deps):
                        n, n_goodmag, n_fail, targets = _cached_counts(fname, out_tsv, this_filter)
                        idx_row = dict(
                            file=fname, filter=this_filter,
                            n="cached", n_goodmag=n_goodmag, n_fail=n_fail,
                            targets=targets, path=str(out_tsv.name)
                        )
                        dbg_row = dict(
                            file=fname, cached=True,
                            n=n, n_goodmag=n_goodmag, n_fail=n_fail, targets=targets
                        )
                        return fname, idx_row, dbg_row, [], "cached"

                    tgt = _load_frame_targets(fname, this_filter)
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

                    exptime = _get_exptime_fallback(fpath, default=1.0, header_cache=self._header_cache, filename=fname)
                    sky_frame_e = _sky_sigma_for(fname)

                    img = fits.getdata(fpath).astype(np.float32)
                    h, w = img.shape

                    apply_flag, c_apcorr, rel_sc = (False, np.nan, np.nan)
                    if apcorr_df is not None and ap_mode in ("apcorr", "auto"):
                        row_apc = apcorr_df[apcorr_df["file"].astype(str) == str(fname)]
                        if not row_apc.empty:
                            c_apcorr = float(row_apc["apcorr"].values[0]) if "apcorr" in row_apc.columns else np.nan
                            rel_sc = float(row_apc["rel_scatter"].values[0]) if "rel_scatter" in row_apc.columns else np.nan
                            apply_flag = bool(row_apc["apply"].values[0]) if "apply" in row_apc.columns else False

                    rows = []
                    frame_fail_rows = []
                    n_goodmag = 0
                    n_fail = 0

                    id2sid = {}
                    if "source_id" in tgt.columns:
                        id2sid = dict(zip(tgt["ID"].astype(int), tgt["source_id"].astype("Int64")))

                    if bkg_use_segm_mask:
                        det_xy = _det_xy_for(fname)
                    else:
                        det_xy = np.zeros((0, 2), float)
                except Exception as e:
                    idx_row = dict(
                        file=fname, filter="unknown", n=0, n_goodmag=0, n_fail=0,
                        targets=0, path=str(out_tsv.name)
                    )
                    dbg_row = dict(
                        file=fname, cached=False, targets=0, reason="exception", error=str(e)
                    )
                    return fname, idx_row, dbg_row, [], "error"

                try:
                    for _, tr in tgt.iterrows():
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
                            delta_r = math.hypot(xc_new - x0, yc_new - y0)
                            if delta_r > max_recenter_shift:
                                recenter_capped = True
                            else:
                                xc, yc = xc_new, yc_new

                        if xc < 0 or xc >= w or yc < 0 or yc >= h:
                            n_fail += 1
                            frame_fail_rows.append(dict(file=fname, ID=ID, reason="xy_outside"))
                            continue

                        cut, xc_cut, yc_cut, x0_cut, y0_cut = _cutout(img, xc, yc, r_out_val)

                        r_exclude = max(float(r_ap_val), float(fwhm_used))
                        neigh = _neighbor_mask(img, xc, yc, fwhm_used, det_xy, r_exclude=r_exclude)
                        neigh_mask = _mask_neighbors(cut, x0_cut, y0_cut, fwhm_used, neigh) if bkg_use_segm_mask else None
                        if neigh_mask is not None:
                            cut = _clip_image(cut, neigh_mask)

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
                    with self._write_lock:
                        df_out.to_csv(out_tsv, sep="\t", index=False, na_rep="NaN")

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
                        sky_sigma_source=_sky_src, sky_frame_e=float(sky_frame_e) if np.isfinite(sky_frame_e) else None
                    )
                    return fname, idx_row, dbg_row, frame_fail_rows, "processed"
                except Exception as e:
                    idx_row = dict(
                        file=fname, filter=this_filter,
                        n=0, n_goodmag=0, n_fail=0,
                        targets=n_tgt, path=str(out_tsv.name)
                    )
                    dbg_row = dict(
                        file=fname, cached=False, targets=n_tgt, reason="exception", error=str(e)
                    )
                    return fname, idx_row, dbg_row, [], "error"

            self._log(f"Starting parallel photometry with {self.max_workers} workers...")

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_fname = {executor.submit(process_single_frame, f): f for f in frames}

                for future in as_completed(future_to_fname):
                    if self._stop_requested:
                        break

                    try:
                        fname, idx_row, dbg_row, fail_rows, status = future.result()

                        if status == "cached":
                            counters["cached"] += 1
                        elif status == "processed":
                            counters["processed"] += 1
                        elif status == "no_targets":
                            counters["no_targets"] += 1
                        elif status == "no_aperture":
                            counters["no_aperture"] += 1
                        elif status == "no_file":
                            counters["no_file"] += 1
                        elif status == "error":
                            counters["error"] += 1
                            if dbg_row and dbg_row.get("error"):
                                self._log(f"[ERROR] {fname}: {dbg_row.get('error')}")

                        if idx_row:
                            index_rows.append(idx_row)
                        if dbg_row:
                            debug_frames.append(dbg_row)
                        if fail_rows:
                            _fail_rows_all.extend(fail_rows)

                        completed_count[0] += 1
                        self.progress.emit(completed_count[0], total, fname)

                        if idx_row:
                            self.frame_done.emit(fname, {
                                "file": fname,
                                "filter": idx_row.get("filter", ""),
                                "n": idx_row.get("n", 0),
                                "n_goodmag": idx_row.get("n_goodmag", 0),
                                "n_fail": idx_row.get("n_fail", 0),
                                "targets": idx_row.get("targets", 0),
                            })

                    except Exception:
                        completed_count[0] += 1
                        self.progress.emit(completed_count[0], total, "error")
                        continue

            if _fail_rows_all:
                pd.DataFrame(_fail_rows_all).to_csv(fail_csv, sep="\t", index=False, encoding="utf-8-sig")

            idx_path = output_dir / "photometry_index.csv"
            index_cols = ["file", "filter", "n", "n_goodmag", "n_fail", "targets", "path"]
            pd.DataFrame(index_rows, columns=index_cols).to_csv(idx_path, index=False)

            try:
                with open(debug_json, "w", encoding="utf-8") as f:
                    json.dump(debug_frames, f, indent=2)
            except Exception:
                pass

            summary = dict(
                n_frames=len(frames),
                n_cached=counters["cached"],
                n_processed=counters["processed"],
                n_no_targets=counters["no_targets"],
                n_no_aperture=counters["no_aperture"],
                n_no_file=counters["no_file"],
                n_error=counters["error"],
                index_path=str(idx_path)
            )
            self.finished.emit(summary)

        except Exception as e:
            self._log(f"[ERROR] {e}\\n{traceback.format_exc()}")
            self.error.emit("WORKER", str(e))
            self.finished.emit({})


class ForcedPhotometryWindow(StepWindowBase):
    """Step 9: Forced Photometry"""

    def __init__(self, params, file_manager, project_state, main_window):
        self.file_manager = file_manager
        self.worker = None
        self.file_list = []
        self.use_cropped = False
        self.log_window = None

        super().__init__(
            step_index=8,
            step_name="Forced Photometry",
            params=params,
            project_state=project_state,
            main_window=main_window
        )

        self.setup_step_ui()
        self.restore_state()

    def setup_step_ui(self):
        info = QLabel("Forced photometry per frame based on ID-matched positions.")
        info.setStyleSheet("QLabel { background-color: #E3F2FD; padding: 10px; border-radius: 5px; }")
        self.content_layout.addWidget(info)

        control_layout = QHBoxLayout()
        btn_params = QPushButton("Photometry Parameters")
        btn_params.setStyleSheet("QPushButton { background-color: #9C27B0; color: white; font-weight: bold; padding: 8px 15px; }")
        btn_params.clicked.connect(self.open_parameters_dialog)
        control_layout.addWidget(btn_params)

        control_layout.addStretch()

        self.btn_run = QPushButton("Run Photometry")
        self.btn_run.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px 20px; }")
        self.btn_run.clicked.connect(self.run_photometry)
        control_layout.addWidget(self.btn_run)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; padding: 8px 15px; }")
        self.btn_stop.clicked.connect(self.stop_photometry)
        self.btn_stop.setEnabled(False)
        control_layout.addWidget(self.btn_stop)

        btn_log = QPushButton("Log")
        btn_log.setStyleSheet("QPushButton { background-color: #607D8B; color: white; font-weight: bold; padding: 8px 15px; }")
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
        self.progress_label.setMinimumWidth(350)
        progress_layout.addWidget(self.progress_label)
        self.content_layout.addLayout(progress_layout)

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
        self.content_layout.addWidget(table_group)

        self.log_window = QWidget(self, Qt.Window)
        self.log_window.setWindowTitle("Photometry Log")
        self.log_window.resize(800, 400)
        log_layout = QVBoxLayout(self.log_window)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("QTextEdit { font-family: monospace; font-size: 9pt; }")
        log_layout.addWidget(self.log_text)

        self._restore_file_context()
        self.populate_file_list()
        self.update_frame_table()

    def log(self, message: str):
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")

    def _restore_file_context(self):
        """Restore data_dir and file_path_map after restart if needed."""
        if getattr(self.params.P, "file_path_map", None):
            return

        if self.file_manager and getattr(self.file_manager, "path_map", None):
            if self.file_manager.path_map:
                self.params.P.file_path_map = {k: str(v) for k, v in self.file_manager.path_map.items()}
                return

        if not self.project_state:
            return

        state = self.project_state.get_step_data("file_selection")
        if not state:
            return

        data_dir = state.get("data_dir")
        if data_dir:
            self.params.P.data_dir = data_dir

        prefix = state.get("filename_prefix")
        if prefix:
            self.params.P.filename_prefix = prefix

        if self.file_manager:
            try:
                if state.get("multi_night") and state.get("night_dirs"):
                    root_dir = state.get("root_dir") or data_dir
                    night_dirs = [Path(p) for p in state.get("night_dirs", []) if p]
                    if root_dir:
                        self.file_manager.set_multi_night_dirs(Path(root_dir), night_dirs)
                else:
                    self.file_manager.clear_multi_night_dirs()

                if not self.file_manager.path_map:
                    self.file_manager.scan_files()
            except Exception as e:
                self.log(f"File scan warning: {e}")

            if self.file_manager.path_map:
                self.params.P.file_path_map = {k: str(v) for k, v in self.file_manager.path_map.items()}

    def populate_file_list(self):
        self._restore_file_context()
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

    def open_parameters_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Photometry Parameters")
        dialog.resize(480, 520)
        layout = QVBoxLayout(dialog)
        form = QFormLayout()

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

        self.param_neighbor_mask = QCheckBox("Enable")
        self.param_neighbor_mask.setChecked(bool(getattr(self.params.P, "bkg_use_segm_mask", False)))
        self.param_neighbor_mask.setToolTip("Exclude nearby detections from background annulus")
        form.addRow("Neighbor Mask:", self.param_neighbor_mask)

        self.param_ap_mode = QLineEdit()
        self.param_ap_mode.setText(str(getattr(self.params.P, "aperture_mode", "apcorr")))
        form.addRow("Aperture Mode:", self.param_ap_mode)

        self.param_force = QCheckBox("Force re-phot")
        self.param_force.setChecked(bool(getattr(self.params.P, "force_rephot", False)))
        form.addRow("Force:", self.param_force)

        layout.addLayout(form)

        scale_group = QGroupBox("Aperture/Annulus Scales")
        scale_form = QFormLayout(scale_group)

        self.param_ap_scale = QDoubleSpinBox()
        self.param_ap_scale.setRange(0.5, 5.0)
        self.param_ap_scale.setSingleStep(0.1)
        self.param_ap_scale.setValue(float(getattr(self.params.P, "phot_aperture_scale", 1.0)))
        scale_form.addRow("Aperture scale (xFWHM):", self.param_ap_scale)

        self.param_ann_in = QDoubleSpinBox()
        self.param_ann_in.setRange(1.0, 10.0)
        self.param_ann_in.setSingleStep(0.5)
        self.param_ann_in.setValue(float(getattr(self.params.P, "fitsky_annulus_scale", 4.0)))
        scale_form.addRow("Annulus inner scale (xFWHM):", self.param_ann_in)

        self.param_ann_out = QDoubleSpinBox()
        self.param_ann_out.setRange(0.5, 10.0)
        self.param_ann_out.setSingleStep(0.5)
        self.param_ann_out.setValue(float(getattr(self.params.P, "fitsky_dannulus_scale", 2.0)))
        scale_form.addRow("Annulus width scale (xFWHM):", self.param_ann_out)

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
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(lambda: self.save_parameters(dialog))
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec_()

    def save_parameters(self, dialog):
        self.params.P.zp_initial = self.param_zp.value()
        self.params.P.min_snr_for_mag = self.param_snr.value()
        self.params.P.recenter_aperture = self.param_recenter.isChecked()
        self.params.P.bkg_use_segm_mask = self.param_neighbor_mask.isChecked()
        self.params.P.aperture_mode = self.param_ap_mode.text().strip()
        self.params.P.force_rephot = self.param_force.isChecked()
        self.params.P.phot_aperture_scale = self.param_ap_scale.value()
        self.params.P.fitsky_annulus_scale = self.param_ann_in.value()
        self.params.P.fitsky_dannulus_scale = self.param_ann_out.value()
        self.params.P.annulus_min_gap_px = self.param_ann_gap.value()
        self.params.P.annulus_min_width_px = self.param_ann_minw.value()
        self.params.P.annulus_sigma_clip = self.param_sigma_clip.value()
        self.save_state()
        QMessageBox.information(dialog, "Success", "Parameters saved!")
        dialog.accept()

    def run_photometry(self):
        self._restore_file_context()
        self.populate_file_list()
        if not self.file_list:
            QMessageBox.warning(self, "Warning", "No files to process")
            return
        if self.worker and self.worker.isRunning():
            return

        self.log_text.clear()
        self.log(
            "Params | "
            f"files={len(self.file_list)} | use_cropped={self.use_cropped} | "
            f"force_rephot={getattr(self.params.P, 'force_rephot', False)} | "
            f"ap_mode={getattr(self.params.P, 'aperture_mode', 'apcorr')} | "
            f"min_snr={getattr(self.params.P, 'min_snr_for_mag', 3.0)} | "
            f"neighbor_mask={bool(getattr(self.params.P, 'bkg_use_segm_mask', False))}"
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
        self.worker.frame_done.connect(self.on_frame_done)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.log.connect(self.log)

        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(len(self.file_list))
        self.progress_label.setText(f"0/{len(self.file_list)} | Starting...")
        self.worker.start()
        self.show_log_window()

    def stop_photometry(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()

    def on_progress(self, current, total, filename):
        self.progress_bar.setValue(current)
        self.progress_label.setText(f"{current}/{total} | {filename}")

    def on_frame_done(self, filename, result):
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

        self.frame_table.scrollToBottom()

    def on_finished(self, summary):
        try:
            self.btn_run.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self.progress_label.setText("Done")
            self.log(f"Photometry done: {summary}")

            if self.worker:
                self.worker.quit()
                from PyQt5.QtCore import QTimer
                QTimer.singleShot(500, self._cleanup_worker)

            idx_path = step9_dir(self.params.P.result_dir) / "photometry_index.csv"
            if not idx_path.exists():
                idx_path = self.params.P.result_dir / "photometry_index.csv"
            if idx_path.exists():
                if idx_path.stat().st_size == 0:
                    self.log("[WARN] photometry_index.csv is empty")
                    self.save_state()
                    self.update_navigation_buttons()
                    return
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
                except Exception as e:
                    self.log(f"[WARN] Failed to read index: {e}")
            self.save_state()
            self.update_navigation_buttons()
        except Exception as e:
            import traceback
            self.log(f"[ERROR] on_finished crashed: {e}\n{traceback.format_exc()}")

    def on_error(self, filename, error):
        self.log(f"ERROR {filename}: {error}")

    def _cleanup_worker(self):
        try:
            if self.worker:
                if self.worker.isRunning():
                    self.worker.wait(1000)
                try:
                    self.worker.deleteLater()
                except Exception:
                    pass
                self.worker = None
        except Exception:
            self.worker = None

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
                m = re.search(r"(\\d+)", str(val))
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

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.stop_photometry()
            self.worker.wait(5000)
        event.accept()

    def validate_step(self) -> bool:
        idx_path = step9_dir(self.params.P.result_dir) / "photometry_index.csv"
        if not idx_path.exists():
            idx_path = self.params.P.result_dir / "photometry_index.csv"
        return idx_path.exists()

    def save_state(self):
        state_data = {
            "force_rephot": getattr(self.params.P, "force_rephot", False),
            "aperture_mode": getattr(self.params.P, "aperture_mode", "apcorr"),
            "min_snr_for_mag": getattr(self.params.P, "min_snr_for_mag", 3.0),
            "bkg_use_segm_mask": getattr(self.params.P, "bkg_use_segm_mask", False),
            "phot_aperture_scale": getattr(self.params.P, "phot_aperture_scale", 1.0),
            "fitsky_annulus_scale": getattr(self.params.P, "fitsky_annulus_scale", 4.0),
            "fitsky_dannulus_scale": getattr(self.params.P, "fitsky_dannulus_scale", 2.0),
            "annulus_min_gap_px": getattr(self.params.P, "annulus_min_gap_px", 6.0),
            "annulus_min_width_px": getattr(self.params.P, "annulus_min_width_px", 12.0),
            "annulus_sigma_clip": getattr(self.params.P, "annulus_sigma_clip", 3.0),
        }
        self.project_state.store_step_data("forced_photometry", state_data)

    def restore_state(self):
        state_data = self.project_state.get_step_data("forced_photometry")
        if state_data:
            for key, val in state_data.items():
                if hasattr(self.params.P, key):
                    setattr(self.params.P, key, val)
