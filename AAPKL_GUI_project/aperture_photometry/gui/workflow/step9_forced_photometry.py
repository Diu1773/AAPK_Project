"""
Step 9: Aperture Photometry (per-frame ID-matched XY positions)
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
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QTabWidget, QSplitter, QListWidget,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from .step_window_base import StepWindowBase
from .aperture_photometry_worker import ApertureWorker
from .aperture_overlay_tab import ApertureOverlayWindow
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
from ...utils.common_helpers import safe_float as _safe_float, normalize_filter_key as _normalize_filter_key
from ...utils.qc_utils import filter_files_by_qc, filter_files_by_wcs_qc
from ...utils.io_utils import read_csv_int64_source_id, coerce_int64_source_id
from ...utils.photometry_utils import (
    circle_mask as _circle_mask,
    refine_local_centroid as _refine_local_centroid,
    phot_one_star as _phot_one_target,
)


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


class ForcedPhotometryWorker(QThread):
    """Worker thread for aperture photometry with parallel processing."""
    progress = pyqtSignal(int, int, str)
    frame_done = pyqtSignal(str, dict)  # filename, result_dict
    finished = pyqtSignal(dict)
    error = pyqtSignal(str, str)
    log = pyqtSignal(str)

    def __init__(self, file_list, params, data_dir, result_dir, cache_dir, use_cropped=False, night_assignments=None):
        super().__init__()
        self.file_list = list(file_list)
        self.night_assignments: dict = dict(night_assignments) if night_assignments else {}
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
            check_sids = {}   # filter -> check source_id
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
                        df = read_csv_int64_source_id(mp, sep="\t")
                        if {"source_id", "ID"} <= set(df.columns):
                            sid = coerce_int64_source_id(df["source_id"]).dropna().astype("int64")
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
                        check_sid = sel.get("check_source_id")
                        if tid is not None:
                            target_sids[flt] = int(tid)
                            all_source_ids.add(int(tid))
                        comp_sids[flt] = set(int(c) for c in cids if c is not None)
                        all_source_ids.update(comp_sids[flt])
                        if check_sid is not None:
                            check_sids[flt] = int(check_sid)
                            all_source_ids.add(int(check_sid))
                        selected = set(comp_sids[flt])
                        if tid is not None:
                            selected.add(int(tid))
                        if check_sid is not None:
                            selected.add(int(check_sid))
                        if selected:
                            selected_sids_by_filter[flt] = selected
                    except Exception as e:
                        self._log(f"[WARN] Failed to load {sel_file.name}: {e}")

            if not all_source_ids:
                raise RuntimeError("No targets/comparisons/check stars found. Complete Step 8 first.")

            self._log(
                f"Loaded {len(target_sids)} target(s), "
                f"{sum(len(v) for v in comp_sids.values())} comparison(s), "
                f"{len(check_sids)} check star(s) from Step 8"
            )
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
            apcorr_path = output_dir / "apcorr_summary.csv"
            if not apcorr_path.exists():
                legacy_apcorr = result_dir / "apcorr_summary.csv"
                if legacy_apcorr.exists():
                    apcorr_path = legacy_apcorr

            apcorr_cand_path = output_dir / "apcorr_candidates.csv"
            if not apcorr_cand_path.exists():
                legacy_apcorr_cand = result_dir / "apcorr_candidates.csv"
                if legacy_apcorr_cand.exists():
                    apcorr_cand_path = legacy_apcorr_cand

            if not ap_path.exists():
                self._log("[WARN] aperture_by_frame.csv not found. Run Apcorr first.")
                self.error.emit("MISSING_APCORR", "aperture_by_frame.csv not found. Run Apcorr first.")
                self.finished.emit({})
                return

            # Re-resolve to prefer newly generated Step9 outputs.
            ap_path = output_dir / "aperture_by_frame.csv" if (output_dir / "aperture_by_frame.csv").exists() else ap_path
            apcorr_path = output_dir / "apcorr_summary.csv" if (output_dir / "apcorr_summary.csv").exists() else apcorr_path
            df_ap = pd.read_csv(ap_path)
            apcorr_df = pd.read_csv(apcorr_path) if apcorr_path.exists() else None

            # ── apcorr 요약 로그 ──────────────────────────────────────────────
            if apcorr_df is not None and not apcorr_df.empty:
                n_total = len(apcorr_df)
                n_apply = int(apcorr_df["apply"].sum()) if "apply" in apcorr_df.columns else 0
                n_reject = n_total - n_apply
                self._log(f"[APCORR] 로드: {apcorr_path.name}  총 {n_total}프레임  apply=True:{n_apply}  False:{n_reject}")
                if "apply_reason" in apcorr_df.columns:
                    reason_counts = apcorr_df[apcorr_df["apply"] == False]["apply_reason"].value_counts()
                    for reason, cnt in reason_counts.items():
                        self._log(f"[APCORR]   거부 사유 '{reason}': {cnt}프레임")
                if "apcorr" in apcorr_df.columns:
                    vals = pd.to_numeric(apcorr_df["apcorr"], errors="coerce").dropna()
                    if len(vals):
                        self._log(f"[APCORR]   apcorr 분포: median={vals.median():.4f}×  "
                                  f"min={vals.min():.4f}  max={vals.max():.4f}  "
                                  f"(단위: flux ratio, ref/opt)")
                        if vals.max() > 3.0:
                            self._log(f"[APCORR]   ⚠ apcorr > 3× 프레임 있음 — isolation/background 문제 의심")
                if "n_used" in apcorr_df.columns:
                    nu = pd.to_numeric(apcorr_df["n_used"], errors="coerce").dropna()
                    if len(nu):
                        self._log(f"[APCORR]   n_used: median={nu.median():.0f}  min={nu.min():.0f}  max={nu.max():.0f}  "
                                  f"(기준: apcorr_use_min_n={getattr(P, 'apcorr_use_min_n', 20)})")
                if "mag_err_optimal" in apcorr_df.columns:
                    me = pd.to_numeric(apcorr_df["mag_err_optimal"], errors="coerce").dropna()
                    if len(me):
                        self._log(f"[APCORR]   err_optimal: median={me.median():.4f}  "
                                  f"(기준: apcorr_scatter_max={getattr(P, 'apcorr_scatter_max', 0.05)})")
            elif apcorr_path.exists():
                self._log("[APCORR] apcorr_summary.csv 비어있음")
            else:
                self._log(f"[APCORR] apcorr_summary.csv 없음 ({apcorr_path})")
            # ─────────────────────────────────────────────────────────────────

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

            # Pre-index sky_df by filename to avoid repeated filtering
            _sky_index = {}
            if _sky_df is not None and not _sky_df.empty and "file" in _sky_df.columns:
                for _, _row in _sky_df.iterrows():
                    _sky_index[str(_row["file"])] = _row

            def _sky_sigma_for(fname):
                try:
                    row = _sky_index.get(fname)
                    if row is None:
                        return np.nan
                    for col in ("sky_sigma_med_e", "sky_sigma_e"):
                        if col in row.index:
                            v = float(row[col])
                            return v if np.isfinite(v) else np.nan
                    for col in ("sky_sigma_med_adu", "sky_sigma_adu"):
                        if col in row.index:
                            v = float(row[col])
                            return (v * GAIN) if np.isfinite(v) else np.nan
                except Exception:
                    return np.nan
                return np.nan

            def _pick_col(cols, cands):
                for c in cands:
                    if c in cols:
                        return c
                return None

            # Pre-index df_frame_map by filename (avoid repeated .astype(str) per frame)
            _frame_map_index = {}
            _fm_c_file = _fm_c_id = _fm_c_sid = _fm_c_x = _fm_c_y = _fm_c_sep = None
            if df_frame_map is not None and not df_frame_map.empty:
                _fm_cols = df_frame_map.columns
                _fm_c_file = _pick_col(_fm_cols, ["file", "fname", "frame"])
                _fm_c_id = _pick_col(_fm_cols, ["ID", "id"])
                _fm_c_sid = _pick_col(_fm_cols, ["source_id", "sourceid", "sid"])
                _fm_c_x = _pick_col(_fm_cols, ["x", "x_det", "x_pix", "x0"])
                _fm_c_y = _pick_col(_fm_cols, ["y", "y_det", "y_pix", "y0"])
                _fm_c_sep = _pick_col(_fm_cols, ["sep_arcsec", "sep", "dist_arcsec"])
                if _fm_c_file and _fm_c_x and _fm_c_y:
                    for fn, grp in df_frame_map.groupby(df_frame_map[_fm_c_file].astype(str)):
                        _frame_map_index[str(fn)] = grp

            def _load_frame_targets(fname, filt_key: str):
                sid2id_map = sid2id_by_filter.get(filt_key, {})
                if _frame_map_index:
                    sub = _frame_map_index.get(fname)
                    if sub is not None and len(sub):
                        if not _fm_c_sid:
                            return pd.DataFrame(columns=["ID", "x", "y", "source_id"])
                        out = pd.DataFrame({
                            "x": pd.to_numeric(sub[_fm_c_x], errors="coerce"),
                            "y": pd.to_numeric(sub[_fm_c_y], errors="coerce"),
                        })
                        out["source_id"] = coerce_int64_source_id(sub[_fm_c_sid])
                        if sid2id_map:
                            out["ID"] = out["source_id"].map(sid2id_map).astype("Int64")
                        if "ID" not in out.columns and _fm_c_id and not sid2id_map:
                            out["ID"] = pd.to_numeric(sub[_fm_c_id], errors="coerce").astype("Int64")
                        if _fm_c_sep:
                            out["sep_arcsec"] = pd.to_numeric(sub[_fm_c_sep], errors="coerce")
                        out = out.dropna(subset=["ID", "x", "y"])
                        return out[["ID", "x", "y", "source_id"]]

                # fallback: idmatch CSV + master map
                p = _resolve_idmatch_path(result_dir, cache_dir, fname)
                if p.exists():
                    try:
                        df = read_csv_int64_source_id(p)
                        if {"source_id", "x", "y"} <= set(df.columns):
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
                            apply_reason = str(row_apc["apply_reason"].values[0]) if "apply_reason" in row_apc.columns else ""
                            n_used_apc = int(row_apc["n_used"].values[0]) if "n_used" in row_apc.columns else -1
                            err_opt_apc = float(row_apc["mag_err_optimal"].values[0]) if "mag_err_optimal" in row_apc.columns else np.nan
                            if apply_flag:
                                self._log(
                                    f"[APCORR] {fname} apply=OK  "
                                    f"apcorr={c_apcorr:.4f}× (flux ratio)  "
                                    f"n={n_used_apc}  err_opt={err_opt_apc:.4f}"
                                )
                            else:
                                self._log(
                                    f"[APCORR] {fname} apply=X  사유={apply_reason}  "
                                    f"apcorr={c_apcorr:.4f}×  n={n_used_apc}  err_opt={err_opt_apc:.4f}"
                                )
                        else:
                            self._log(f"[APCORR] {fname} apcorr_summary에 항목 없음 → 미보정")

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
                            # c_apcorr = flux(large_ref) / flux(optimal) — 플럭스 비율
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
                            idx_row.setdefault("night_id", self.night_assignments.get(fname, 0))
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
            index_cols = ["file", "filter", "night_id", "n", "n_goodmag", "n_fail", "targets", "path"]
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
    """Step 9: Aperture Photometry"""

    def __init__(self, params, file_manager, project_state, main_window):
        self.file_manager = file_manager
        self.worker = None
        self.apcorr_worker = None
        self.file_list = []
        self.use_cropped = False
        self.log_window = None
        self._apcorr_summary_df = pd.DataFrame()
        self._growth_curve_df = pd.DataFrame()

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
        # Tab widget: Apcorr QC | Photometry | Aperture Overlay
        self.step_tabs = QTabWidget()
        self.content_layout.addWidget(self.step_tabs, stretch=1)

        # --- Tab 1: Photometry (added after Apcorr QC below) ---
        phot_tab = QWidget()
        phot_layout = QVBoxLayout(phot_tab)

        info = QLabel("Forced photometry per frame based on ID-matched positions.")
        info.setStyleSheet("QLabel { background-color: #E3F2FD; padding: 10px; border-radius: 5px; }")
        phot_layout.addWidget(info)

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

        phot_layout.addLayout(control_layout)

        progress_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        self.progress_label = QLabel("Ready")
        self.progress_label.setMinimumWidth(350)
        progress_layout.addWidget(self.progress_label)
        phot_layout.addLayout(progress_layout)

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
        phot_layout.addWidget(table_group)

        # phot_tab is inserted via insertTab(1, ...) after Apcorr QC is built

        # --- Tab 2: Aperture Overlay (embedded) ---
        self._overlay_obj = ApertureOverlayWindow(
            self.params,
            self.file_manager,
            self.project_state,
            self.main_window,
            step_index_override=self.step_index,
            embedded=True,
        )
        self._overlay_obj.title_label.hide()
        overlay_content = self._overlay_obj.centralWidget()
        self.step_tabs.addTab(overlay_content, "Aperture Overlay")

        # --- Tab 3: Apcorr QC ---
        apcorr_tab = QWidget()
        apcorr_layout = QVBoxLayout(apcorr_tab)

        apcorr_note = QLabel(
            "Growth curve analysis: run Apcorr first to find the optimal aperture per frame, "
            "then run Photometry."
        )
        apcorr_note.setStyleSheet("QLabel { background-color: #FFF8E1; padding: 8px; border-radius: 4px; }")
        apcorr_note.setWordWrap(True)
        apcorr_layout.addWidget(apcorr_note)

        apcorr_ctrl = QHBoxLayout()
        self.btn_apcorr_params = QPushButton("Parameters")
        self.btn_apcorr_params.setStyleSheet("QPushButton { background-color: #795548; color: white; font-weight: bold; padding: 8px 15px; }")
        self.btn_apcorr_params.clicked.connect(self.open_parameters_dialog)
        apcorr_ctrl.addWidget(self.btn_apcorr_params)
        apcorr_ctrl.addStretch()
        self.btn_run_apcorr = QPushButton("Run Apcorr")
        self.btn_run_apcorr.setStyleSheet("QPushButton { background-color: #2E7D32; color: white; font-weight: bold; padding: 8px 20px; }")
        self.btn_run_apcorr.clicked.connect(self.run_apcorr)
        apcorr_ctrl.addWidget(self.btn_run_apcorr)
        self.btn_stop_apcorr = QPushButton("Stop")
        self.btn_stop_apcorr.setStyleSheet("QPushButton { background-color: #c62828; color: white; font-weight: bold; padding: 8px 15px; }")
        self.btn_stop_apcorr.clicked.connect(self.stop_apcorr)
        self.btn_stop_apcorr.setEnabled(False)
        apcorr_ctrl.addWidget(self.btn_stop_apcorr)
        self.btn_apcorr_refresh = QPushButton("Refresh")
        self.btn_apcorr_refresh.setStyleSheet("QPushButton { background-color: #607D8B; color: white; font-weight: bold; padding: 8px 12px; }")
        self.btn_apcorr_refresh.clicked.connect(self.refresh_apcorr_qc)
        apcorr_ctrl.addWidget(self.btn_apcorr_refresh)
        btn_apcorr_log = QPushButton("Log")
        btn_apcorr_log.setStyleSheet("QPushButton { background-color: #455A64; color: white; font-weight: bold; padding: 8px 12px; }")
        btn_apcorr_log.clicked.connect(self.show_log_window)
        apcorr_ctrl.addWidget(btn_apcorr_log)
        apcorr_layout.addLayout(apcorr_ctrl)

        apcorr_prog_row = QHBoxLayout()
        self.apcorr_progress_bar = QProgressBar()
        self.apcorr_progress_bar.setMinimum(0)
        self.apcorr_progress_bar.setValue(0)
        apcorr_prog_row.addWidget(self.apcorr_progress_bar)
        self.apcorr_status_label = QLabel("Ready")
        self.apcorr_status_label.setMinimumWidth(300)
        apcorr_prog_row.addWidget(self.apcorr_status_label)
        apcorr_layout.addLayout(apcorr_prog_row)

        self.apcorr_splitter = QSplitter(Qt.Horizontal)

        # Left: plot
        plot_panel = QWidget()
        plot_layout = QVBoxLayout(plot_panel)
        self.apcorr_fig = Figure(figsize=(6.4, 5.6))
        self.apcorr_canvas = FigureCanvas(self.apcorr_fig)
        self.apcorr_ax_mag = self.apcorr_fig.add_subplot(211)
        self.apcorr_ax_err = self.apcorr_fig.add_subplot(212)
        self.apcorr_fig.subplots_adjust(hspace=0.35)
        plot_layout.addWidget(self.apcorr_canvas)
        self.apcorr_splitter.addWidget(plot_panel)

        # Right: frame picker + candidate table
        side_panel = QWidget()
        side_layout = QVBoxLayout(side_panel)
        side_layout.addWidget(QLabel("Frames"))
        self.apcorr_file_list = QListWidget()
        self.apcorr_file_list.currentTextChanged.connect(self._on_apcorr_file_changed)
        side_layout.addWidget(self.apcorr_file_list, stretch=1)

        side_layout.addWidget(QLabel("Growth Curve (selected frame)"))
        self.apcorr_file_cand_table = QTableWidget()
        self.apcorr_file_cand_table.setColumnCount(7)
        self.apcorr_file_cand_table.setHorizontalHeaderLabels(
            ["scale", "r (px)", "med_mag", "med_mag_err", "med_SNR", "n_stars", "selected"]
        )
        for col in range(7):
            mode = QHeaderView.Stretch if col in (0, 1) else QHeaderView.ResizeToContents
            self.apcorr_file_cand_table.horizontalHeader().setSectionResizeMode(col, mode)
        side_layout.addWidget(self.apcorr_file_cand_table, stretch=2)
        self.apcorr_splitter.addWidget(side_panel)

        self.apcorr_splitter.setStretchFactor(0, 3)
        self.apcorr_splitter.setStretchFactor(1, 2)
        apcorr_layout.addWidget(self.apcorr_splitter, stretch=1)

        apcorr_sum_group = QGroupBox("Apcorr Summary")
        apcorr_sum_layout = QVBoxLayout(apcorr_sum_group)
        self.apcorr_sum_table = QTableWidget()
        self.apcorr_sum_table.setColumnCount(8)
        self.apcorr_sum_table.setHorizontalHeaderLabels(
            ["Frame", "FWHM(px)", "Opt r(px)", "Opt scale", "Apcorr", "SNR_opt", "mag_err_opt", "apply"]
        )
        self.apcorr_sum_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for col in range(1, 8):
            self.apcorr_sum_table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeToContents)
        self.apcorr_sum_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        apcorr_sum_layout.addWidget(self.apcorr_sum_table)
        apcorr_layout.addWidget(apcorr_sum_group)

        # Final tab order: Apcorr QC (0), Photometry (1), Aperture Overlay (2)
        self.step_tabs.insertTab(0, apcorr_tab, "Apcorr QC")
        self.step_tabs.insertTab(1, phot_tab, "Photometry")
        self.step_tabs.setCurrentIndex(0)

        # --- Log window (shared, floating) ---
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
        self.refresh_apcorr_qc()

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
        use_qc = bool(getattr(self.params.P, "phot_use_qc_pass_only", False))
        files, qc_info = filter_files_by_qc(Path(self.params.P.result_dir), files, require_qc=use_qc)
        if use_qc:
            if qc_info.get("applied"):
                self.log(f"[QC] Frame QC filter: {qc_info['kept']}/{qc_info['total']} kept.")
            elif qc_info.get("path") is None:
                self.log("[QC] frame_quality.csv not found; using all frames.")
            else:
                self.log(f"[QC] frame_quality.csv ignored ({qc_info['reason']}); using all frames.")
        files, wcs_info = filter_files_by_wcs_qc(Path(self.params.P.result_dir), files)
        if wcs_info.get("applied"):
            self.log(f"[WCS QC] {wcs_info['kept']}/{wcs_info['total']} frames passed WCS QC.")
        elif wcs_info.get("path") is None:
            self.log("[WCS QC] frame_wcs_qc.csv not found; using all frames.")
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
        self.param_ann_in.setToolTip("r_in (annulus inner radius) = scale × FWHM (typical ~4.0)")
        scale_form.addRow("Annulus r_in (×FWHM):", self.param_ann_in)

        self.param_ann_out = QDoubleSpinBox()
        self.param_ann_out.setRange(0.5, 10.0)
        self.param_ann_out.setSingleStep(0.5)
        self.param_ann_out.setValue(float(getattr(self.params.P, "fitsky_dannulus_scale", 2.0)))
        self.param_ann_out.setToolTip("d_annulus (annulus thickness) = scale × FWHM (typical 2–3)")
        scale_form.addRow("Annulus d_annulus (×FWHM):", self.param_ann_out)

        self.param_sigma_clip = QDoubleSpinBox()
        self.param_sigma_clip.setRange(0.5, 10.0)
        self.param_sigma_clip.setSingleStep(0.1)
        self.param_sigma_clip.setValue(float(getattr(self.params.P, "annulus_sigma_clip", 3.0)))
        scale_form.addRow("Annulus sigma clip:", self.param_sigma_clip)

        layout.addWidget(scale_group)

        apcorr_group = QGroupBox("Aperture Correction (Apcorr)")
        apcorr_form = QFormLayout(apcorr_group)

        self.param_apcorr_scatter = QDoubleSpinBox()
        self.param_apcorr_scatter.setRange(0.01, 0.50)
        self.param_apcorr_scatter.setSingleStep(0.01)
        self.param_apcorr_scatter.setDecimals(3)
        self.param_apcorr_scatter.setValue(float(getattr(self.params.P, "apcorr_scatter_max", 0.05)))
        self.param_apcorr_scatter.setToolTip("Max allowed rel_scatter (1.4826×MAD/apcorr). Raise to 0.10–0.20 for sparse fields.")
        apcorr_form.addRow("Max rel_scatter:", self.param_apcorr_scatter)

        self.param_apcorr_scale = QDoubleSpinBox()
        self.param_apcorr_scale.setRange(1.5, 8.0)
        self.param_apcorr_scale.setSingleStep(0.5)
        self.param_apcorr_scale.setValue(float(getattr(self.params.P, "apcorr_large_ref_scale", 5.0)))
        self.param_apcorr_scale.setToolTip("Large reference aperture size (×FWHM). Smaller = less PSF-wing noise but less encircled energy.")
        apcorr_form.addRow("Large ref scale (×FWHM):", self.param_apcorr_scale)

        self.param_apcorr_min_n = QSpinBox()
        self.param_apcorr_min_n.setRange(3, 100)
        self.param_apcorr_min_n.setValue(int(getattr(self.params.P, "apcorr_use_min_n", 20)))
        self.param_apcorr_min_n.setToolTip("Minimum number of reference stars required to compute apcorr.")
        apcorr_form.addRow("Min stars:", self.param_apcorr_min_n)

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
        self.params.P.bkg_use_segm_mask = self.param_neighbor_mask.isChecked()
        self.params.P.aperture_mode = self.param_ap_mode.text().strip()
        self.params.P.force_rephot = self.param_force.isChecked()
        self.params.P.phot_aperture_scale = self.param_ap_scale.value()
        self.params.P.fitsky_annulus_scale = self.param_ann_in.value()
        self.params.P.fitsky_dannulus_scale = self.param_ann_out.value()
        self.params.P.annulus_sigma_clip = self.param_sigma_clip.value()
        self.params.P.apcorr_scatter_max = self.param_apcorr_scatter.value()
        self.params.P.apcorr_large_ref_scale = self.param_apcorr_scale.value()
        self.params.P.apcorr_use_min_n = self.param_apcorr_min_n.value()
        self.save_state()
        QMessageBox.information(dialog, "Success", "Parameters saved!")
        dialog.accept()

    # ------------------------------------------------------------------
    # Apcorr run / stop
    # ------------------------------------------------------------------

    def run_apcorr(self):
        self._restore_file_context()
        self.populate_file_list()
        if not self.file_list:
            QMessageBox.warning(self, "Apcorr", "No files found.")
            return
        if self.apcorr_worker and self.apcorr_worker.isRunning():
            return
        output_dir = step9_dir(Path(self.params.P.result_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        cache_dir = output_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        data_dir = Path(self.params.P.data_dir)
        self.apcorr_worker = ApertureWorker(
            self.file_list,
            self.params,
            data_dir,
            Path(self.params.P.result_dir),
            cache_dir,
            self.use_cropped,
            output_dir=output_dir,
        )
        self.apcorr_worker.progress.connect(self._on_apcorr_progress)
        self.apcorr_worker.finished.connect(self._on_apcorr_finished)
        self.apcorr_worker.error.connect(self._on_apcorr_error)
        self.apcorr_worker.log.connect(self.log)
        self.btn_run_apcorr.setEnabled(False)
        self.btn_stop_apcorr.setEnabled(True)
        self.btn_run.setEnabled(False)
        self.apcorr_progress_bar.setValue(0)
        self.apcorr_progress_bar.setMaximum(len(self.file_list))
        self.apcorr_status_label.setText("Running Apcorr...")
        self._apcorr_start_time = time.monotonic()
        self.apcorr_worker.start()

    def stop_apcorr(self):
        if self.apcorr_worker and self.apcorr_worker.isRunning():
            self.apcorr_worker.stop()
            self.apcorr_status_label.setText("Stopping...")

    def _on_apcorr_progress(self, current: int, total: int, msg: str):
        self.apcorr_progress_bar.setMaximum(max(total, 1))
        self.apcorr_progress_bar.setValue(current)
        eta_str = ""
        if current > 0 and total > 0 and hasattr(self, "_apcorr_start_time"):
            elapsed = time.monotonic() - self._apcorr_start_time
            remaining = elapsed / current * (total - current)
            if remaining < 60:
                eta_str = f" | ETA {int(remaining)}s"
            else:
                eta_str = f" | ETA {int(remaining // 60)}m{int(remaining % 60):02d}s"
        self.apcorr_status_label.setText(f"{msg}{eta_str}")

    def _on_apcorr_finished(self, result: dict):
        self.btn_run_apcorr.setEnabled(True)
        self.btn_stop_apcorr.setEnabled(False)
        self.btn_run.setEnabled(True)
        n = result.get("frames", result.get("total", 0))
        self.apcorr_status_label.setText(f"Apcorr complete: {n} frames")
        self.refresh_apcorr_qc()

    def _on_apcorr_error(self, kind: str, msg: str):
        self.btn_run_apcorr.setEnabled(True)
        self.btn_stop_apcorr.setEnabled(False)
        self.btn_run.setEnabled(True)
        self.apcorr_status_label.setText(f"Apcorr error: {msg[:80]}")
        self.log(f"[APCORR ERROR] {kind}: {msg}")

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

        night_assignments = getattr(self.file_manager, "night_assignments", {}) if self.file_manager else {}
        self.worker = ForcedPhotometryWorker(
            self.file_list,
            self.params,
            self.params.P.data_dir,
            self.params.P.result_dir,
            self.params.P.cache_dir,
            self.use_cropped,
            night_assignments=night_assignments,
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
        self._phot_start_time = time.monotonic()
        self.worker.start()
        self.show_log_window()

    def stop_photometry(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()

    def on_progress(self, current, total, filename):
        self.progress_bar.setValue(current)
        eta_str = ""
        if current > 0 and total > 0 and hasattr(self, "_phot_start_time"):
            elapsed = time.monotonic() - self._phot_start_time
            remaining = elapsed / current * (total - current)
            if remaining < 60:
                eta_str = f" | ETA {int(remaining)}s"
            else:
                eta_str = f" | ETA {int(remaining // 60)}m{int(remaining % 60):02d}s"
        self.progress_label.setText(f"{current}/{total}{eta_str} | {filename}")

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
            self.refresh_apcorr_qc()
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

    def _resolve_apcorr_paths(self) -> tuple[Path, Path]:
        result_dir = Path(self.params.P.result_dir)
        out_dir = step9_dir(result_dir)
        summary = out_dir / "apcorr_summary.csv"
        gc = out_dir / "growth_curve.csv"
        if not summary.exists():
            summary = result_dir / "apcorr_summary.csv"
        if not gc.exists():
            gc = result_dir / "growth_curve.csv"
        return summary, gc

    @staticmethod
    def _to_bool_value(v) -> bool:
        if isinstance(v, bool):
            return v
        if v is None:
            return False
        s = str(v).strip().lower()
        return s in ("1", "true", "yes", "y", "on")

    def refresh_apcorr_qc(self):
        if not hasattr(self, "apcorr_file_list"):
            return

        summary_path, gc_path = self._resolve_apcorr_paths()
        try:
            df_sum = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
        except Exception:
            df_sum = pd.DataFrame()
        try:
            df_gc = pd.read_csv(gc_path) if gc_path.exists() else pd.DataFrame()
        except Exception:
            df_gc = pd.DataFrame()

        self._apcorr_summary_df = df_sum
        self._growth_curve_df = df_gc

        files = []
        if not df_gc.empty and "file" in df_gc.columns:
            files = sorted([str(x) for x in pd.unique(df_gc["file"].astype(str)) if str(x).strip()])
        elif not df_sum.empty and "file" in df_sum.columns:
            files = sorted([str(x) for x in pd.unique(df_sum["file"].astype(str)) if str(x).strip()])

        self.apcorr_file_list.blockSignals(True)
        self.apcorr_file_list.clear()
        self.apcorr_file_list.addItems(files)
        self.apcorr_file_list.blockSignals(False)

        if not df_sum.empty and "apply" in df_sum.columns:
            try:
                n_apply = int(df_sum["apply"].map(self._to_bool_value).sum())
                self.apcorr_status_label.setText(
                    f"Apcorr QC: {len(df_sum)} frames | apply={n_apply}/{len(df_sum)}"
                )
            except Exception:
                self.apcorr_status_label.setText(f"Apcorr QC: {len(df_sum)} frames")
        elif summary_path.exists():
            self.apcorr_status_label.setText("Apcorr QC: summary loaded")
        else:
            self.apcorr_status_label.setText("Apcorr QC: apcorr_summary.csv not found")

        if files:
            self.apcorr_file_list.setCurrentRow(0)
            self._on_apcorr_file_changed(files[0])
        else:
            if hasattr(self, "apcorr_file_cand_table"):
                self.apcorr_file_cand_table.setRowCount(0)
            for ax in (self.apcorr_ax_mag, self.apcorr_ax_err):
                ax.clear()
                ax.text(0.5, 0.5, "No apcorr data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="#888")
                ax.set_axis_off()
            self.apcorr_canvas.draw_idle()

        self._refresh_apcorr_sum_table(df_sum)

    def _refresh_apcorr_sum_table(self, df_sum: pd.DataFrame):
        if not hasattr(self, "apcorr_sum_table"):
            return
        self.apcorr_sum_table.setRowCount(0)
        if df_sum.empty:
            return
        for _, row in df_sum.iterrows():
            r = self.apcorr_sum_table.rowCount()
            self.apcorr_sum_table.insertRow(r)
            vals = [
                str(row.get("file", "")),
                f"{float(row.get('fwhm_used', row.get('fwhm_med', np.nan))):.2f}"
                    if np.isfinite(_safe_float(row.get("fwhm_used", row.get("fwhm_med")), np.nan)) else "",
                f"{float(row.get('r_optimal', np.nan)):.2f}"
                    if np.isfinite(_safe_float(row.get("r_optimal"), np.nan)) else "",
                f"{float(row.get('optimal_scale', np.nan)):.2f}"
                    if np.isfinite(_safe_float(row.get("optimal_scale"), np.nan)) else "",
                f"{float(row.get('apcorr', np.nan)):.4f}"
                    if np.isfinite(_safe_float(row.get("apcorr"), np.nan)) else "",
                f"{float(row.get('snr_optimal', np.nan)):.1f}"
                    if np.isfinite(_safe_float(row.get("snr_optimal"), np.nan)) else "",
                f"{float(row.get('mag_err_optimal', np.nan)):.4f}"
                    if np.isfinite(_safe_float(row.get("mag_err_optimal"), np.nan)) else "",
                "✓" if self._to_bool_value(row.get("apply", False)) else "✗",
            ]
            for c, text in enumerate(vals):
                self.apcorr_sum_table.setItem(r, c, QTableWidgetItem(text))

    def _on_apcorr_file_changed(self, filename: str):
        if not filename or not hasattr(self, "apcorr_file_cand_table"):
            return
        df_gc = getattr(self, "_growth_curve_df", None)
        if df_gc is None or df_gc.empty or "file" not in df_gc.columns:
            self.apcorr_file_cand_table.setRowCount(0)
            for ax in (self.apcorr_ax_mag, self.apcorr_ax_err):
                ax.clear()
                ax.text(0.5, 0.5, "No growth curve data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="#888")
                ax.set_axis_off()
            self.apcorr_canvas.draw_idle()
            return

        sub = df_gc[df_gc["file"].astype(str) == str(filename)].copy()
        if sub.empty:
            self.apcorr_file_cand_table.setRowCount(0)
            for ax in (self.apcorr_ax_mag, self.apcorr_ax_err):
                ax.clear()
                ax.text(0.5, 0.5, "No data for this frame", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="#888")
                ax.set_axis_off()
            self.apcorr_canvas.draw_idle()
            return

        sub = sub.sort_values("r_px", kind="stable")
        self.apcorr_file_cand_table.setRowCount(0)
        for _, row in sub.iterrows():
            r = self.apcorr_file_cand_table.rowCount()
            self.apcorr_file_cand_table.insertRow(r)
            vals = [
                f"{_safe_float(row.get('scale'), np.nan):.2f}",
                f"{_safe_float(row.get('r_px'), np.nan):.2f}",
                f"{_safe_float(row.get('median_mag'), np.nan):.4f}",
                f"{_safe_float(row.get('median_mag_err'), np.nan):.4f}",
                f"{_safe_float(row.get('median_snr'), np.nan):.1f}",
                str(int(_safe_float(row.get("n_stars", 0), 0))),
                str(row.get("selected", False)),
            ]
            for c, text in enumerate(vals):
                self.apcorr_file_cand_table.setItem(r, c, QTableWidgetItem(text))

        # 2-panel growth curve plot
        ax_mag = self.apcorr_ax_mag
        ax_err = self.apcorr_ax_err
        ax_mag.clear()
        ax_err.clear()

        r_px = pd.to_numeric(sub["r_px"], errors="coerce").to_numpy(float)
        med_mag = pd.to_numeric(sub.get("median_mag", np.nan), errors="coerce").to_numpy(float)
        med_err = pd.to_numeric(sub.get("median_mag_err", np.nan), errors="coerce").to_numpy(float)

        finite_mag = np.isfinite(r_px) & np.isfinite(med_mag)
        finite_err = np.isfinite(r_px) & np.isfinite(med_err)

        if not np.any(finite_mag) and not np.any(finite_err):
            for ax in (ax_mag, ax_err):
                ax.text(0.5, 0.5, "No finite growth curve points", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="#888")
                ax.set_axis_off()
            self.apcorr_canvas.draw_idle()
            return

        # Get summary info
        df_sum = getattr(self, "_apcorr_summary_df", pd.DataFrame())
        r_optimal = np.nan
        fwhm_used = np.nan
        apcorr_val = np.nan
        apply_on = False
        if not df_sum.empty and "file" in df_sum.columns:
            row_s_df = df_sum[df_sum["file"].astype(str) == str(filename)]
            if not row_s_df.empty:
                row_s = row_s_df.iloc[0]
                r_optimal = float(_safe_float(row_s.get("r_optimal"), np.nan))
                fwhm_used = float(_safe_float(row_s.get("fwhm_used"), np.nan))
                apcorr_val = float(_safe_float(row_s.get("apcorr"), np.nan))
                apply_on = self._to_bool_value(row_s.get("apply", False))

        color = "#1565C0"
        if np.any(finite_mag):
            ax_mag.plot(r_px[finite_mag], med_mag[finite_mag], "-o", color=color,
                        linewidth=1.5, markersize=5, markeredgecolor="white", markeredgewidth=0.5)
        if np.isfinite(r_optimal):
            ax_mag.axvline(r_optimal, color="#E53935", linewidth=1.5, linestyle="--",
                           alpha=0.8, label=f"opt r={r_optimal:.1f}px")
        if np.isfinite(fwhm_used) and fwhm_used > 0:
            ax_mag.axvline(fwhm_used, color="#6D4C41", linewidth=1.2, linestyle=":",
                           alpha=0.8, label=f"FWHM={fwhm_used:.2f}px")
        ax_mag.invert_yaxis()
        ax_mag.set_ylabel("Inst Magnitude", fontsize=9)
        title = str(filename)
        if np.isfinite(apcorr_val):
            title += f" | apcorr={apcorr_val:.4f}"
        title += f" | apply={'ON' if apply_on else 'OFF'}"
        ax_mag.set_title(title, fontsize=9)
        ax_mag.grid(True, alpha=0.25)
        ax_mag.legend(fontsize=8, frameon=False)

        if np.any(finite_err):
            ax_err.plot(r_px[finite_err], med_err[finite_err], "-s", color="#E53935",
                        linewidth=1.8, markersize=5, markeredgecolor="white", markeredgewidth=0.5)
        if np.isfinite(r_optimal):
            ax_err.axvline(r_optimal, color="#E53935", linewidth=1.5, linestyle="--", alpha=0.8)
        if np.isfinite(fwhm_used) and fwhm_used > 0:
            ax_err.axvline(fwhm_used, color="#6D4C41", linewidth=1.2, linestyle=":", alpha=0.8)
        ax_err.set_xlabel("Aperture radius (px)", fontsize=9)
        ax_err.set_ylabel("Median mag_err", fontsize=9)
        ax_err.set_title("Error vs Aperture (U-shape)", fontsize=9)
        ax_err.grid(True, alpha=0.25)

        self.apcorr_fig.tight_layout()
        self.apcorr_canvas.draw_idle()

    def closeEvent(self, event):
        if self.apcorr_worker and self.apcorr_worker.isRunning():
            self.stop_apcorr()
            self.apcorr_worker.wait(3000)
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
            "annulus_sigma_clip": getattr(self.params.P, "annulus_sigma_clip", 3.0),
        }
        self.project_state.store_step_data("forced_photometry", state_data)

    def restore_state(self):
        state_data = self.project_state.get_step_data("forced_photometry")
        if state_data:
            for key, val in state_data.items():
                if hasattr(self.params.P, key):
                    setattr(self.params.P, key, val)
