"""
Step 6: Star ID Matching (WCS-based, Gaia source_id)
Ported from AAPKI_GUI.ipynb Cell 8.
"""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

try:
    from astroquery.gaia import Gaia
    _HAS_GAIA = True
except Exception:
    _HAS_GAIA = False

from PyQt5.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QGroupBox, QMessageBox,
    QTextEdit, QDialog, QFormLayout, QDialogButtonBox, QProgressBar,
    QCheckBox, QSpinBox, QDoubleSpinBox, QLineEdit, QTableWidget,
    QTableWidgetItem, QHeaderView, QAbstractItemView, QWidget
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from .step_window_base import StepWindowBase
from ...utils.step_paths import step2_cropped_dir, step5_dir, step6_dir, crop_is_active


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
        if x is None:
            return default
        s = str(x).strip()
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


def _is_up_to_date(out_path: Path, in_paths):
    try:
        out_m = out_path.stat().st_mtime
    except Exception:
        return False
    for p in in_paths:
        try:
            if Path(p).stat().st_mtime > out_m:
                return False
        except Exception:
            return False
    return True


def _table_cols_lower(tab: Table) -> Table:
    cols = list(tab.colnames)
    lower = [c.lower() for c in cols]
    if lower != cols:
        if len(set(lower)) != len(lower):
            raise RuntimeError(f"[IDMatch] Gaia colnames lower() 충돌: {cols}")
        tab.rename_columns(cols, lower)
    return tab


class IdMatchWorker(QThread):
    """Worker thread for ID matching"""
    progress = pyqtSignal(int, int, str)
    file_done = pyqtSignal(str, dict)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str, str)

    def __init__(self, file_list, params, data_dir, result_dir, cache_dir, use_cropped=False):
        super().__init__()
        self.file_list = list(file_list)
        self.params = params
        self.data_dir = Path(data_dir)
        self.result_dir = Path(result_dir)
        self.cache_dir = Path(cache_dir)
        self.use_cropped = use_cropped
        self._local_ra = []
        self._local_dec = []
        self._local_ids = []
        self._next_local_id = -1
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def _load_detect_xy(self, fname):
        p = self.cache_dir / f"detect_{fname}.csv"
        if p.exists() and p.stat().st_size > 0:
            try:
                df = pd.read_csv(p)
                if {"x", "y"} <= set(df.columns) and len(df):
                    xy = df[["x", "y"]].to_numpy(float)
                    xy = xy[np.isfinite(xy).all(axis=1)]
                    return xy
            except Exception:
                pass
        return np.zeros((0, 2), float)

    def _resolve_fits_path(self, fname: str, require_wcs: bool = False) -> Path | None:
        candidates = []
        if self.use_cropped:
            cropped_dir = step2_cropped_dir(self.result_dir)
            if not cropped_dir.exists():
                cropped_dir = self.result_dir / "cropped"
            candidates.append(cropped_dir / fname)
        candidates.append(self.data_dir / fname)
        candidates.append(self.result_dir / fname)
        fallback = None
        for p in candidates:
            if p.exists():
                if not require_wcs:
                    return p
                if fallback is None:
                    fallback = p
                try:
                    hdr = fits.getheader(p)
                    w = WCS(hdr, relax=True)
                    if getattr(w, "has_celestial", False):
                        return p
                    # Fallback: .wcs 파일이 있으면 WCS 있는 것으로 간주
                    wcs_file = p.with_suffix(".wcs")
                    if not wcs_file.exists():
                        wcs_file = Path(str(p) + ".wcs")
                    if wcs_file.exists():
                        return p
                except Exception:
                    pass
        return fallback

    def _load_wcs_from_file(self, fits_path: Path, hdr: fits.Header) -> WCS | None:
        """FITS 헤더 또는 .wcs 파일에서 WCS 로드"""
        # 먼저 헤더에서 시도
        try:
            w = WCS(hdr, relax=True)
            if w.has_celestial:
                return w
        except Exception:
            pass

        # .wcs 파일에서 로드 시도
        wcs_candidates = [
            fits_path.with_suffix(".wcs"),
            Path(str(fits_path) + ".wcs"),
            fits_path.parent / (fits_path.stem + ".wcs"),
        ]
        for wcs_file in wcs_candidates:
            if not wcs_file.exists():
                continue
            try:
                lines = wcs_file.read_text(encoding="utf-8", errors="ignore").splitlines()
                wcs_dict = {}
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
                        wcs_dict[key] = val.strip("'")
                        continue
                    try:
                        if "." in val or "E" in val.upper():
                            wcs_dict[key] = float(val)
                        else:
                            wcs_dict[key] = int(val)
                    except Exception:
                        wcs_dict[key] = val

                # 헤더에 주입
                for k, v in wcs_dict.items():
                    try:
                        hdr[k] = v
                    except Exception:
                        pass

                w = WCS(hdr, relax=True)
                if w.has_celestial:
                    return w
            except Exception:
                pass
        return None

    def _match_unmatched_to_local(self, det_sky: SkyCoord, matched_det_idx: np.ndarray, tol_arcsec: float) -> pd.DataFrame:
        if len(det_sky) == 0:
            return pd.DataFrame()

        all_idx = np.arange(len(det_sky))
        unmatched_mask = np.ones(len(det_sky), dtype=bool)
        unmatched_mask[matched_det_idx] = False
        if not np.any(unmatched_mask):
            return pd.DataFrame()

        um_idx = all_idx[unmatched_mask]
        um_sky = det_sky[unmatched_mask]

        if self._local_ids:
            local_sky = SkyCoord(
                ra=np.array(self._local_ra) * u.deg,
                dec=np.array(self._local_dec) * u.deg
            )
            idx, sep2d, _ = um_sky.match_to_catalog_sky(local_sky)
            sep_arcsec = sep2d.arcsec
        else:
            idx = np.full(len(um_idx), -1, dtype=int)
            sep_arcsec = np.full(len(um_idx), np.nan, dtype=float)

        used_local_ids = set()
        rows = []
        for k, det_i in enumerate(um_idx):
            assign_existing = False
            local_id = None
            if self._local_ids and idx[k] >= 0 and np.isfinite(sep_arcsec[k]) and sep_arcsec[k] <= tol_arcsec:
                candidate_id = self._local_ids[int(idx[k])]
                if candidate_id not in used_local_ids:
                    local_id = candidate_id
                    assign_existing = True

            if not assign_existing:
                local_id = self._next_local_id
                self._next_local_id -= 1
                self._local_ids.append(local_id)
                self._local_ra.append(float(um_sky[k].ra.deg))
                self._local_dec.append(float(um_sky[k].dec.deg))

            used_local_ids.add(local_id)
            rows.append({
                "det_idx": int(det_i),
                "source_id": int(local_id),
                "sep_arcsec": float(sep_arcsec[k]) if assign_existing else np.nan,
            })

        return pd.DataFrame(rows)

    def _query_gaia(self, center: SkyCoord, radius_deg: float, mag_max: float):
        if not _HAS_GAIA:
            raise RuntimeError("astroquery.gaia not available")
        adql = f"""
    SELECT
      source_id, ra, dec,
      phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
      pmra, pmdec, pmra_error, pmdec_error,
      parallax, parallax_error
    FROM gaiadr3.gaia_source
    WHERE 1=CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {center.ra.deg:.8f}, {center.dec.deg:.8f}, {radius_deg:.8f})
    )
        """.strip()
        Gaia.ROW_LIMIT = -1
        job = Gaia.launch_job_async(adql, dump_to_file=False)
        tab = job.get_results()
        if "phot_g_mean_mag" in tab.colnames and np.isfinite(mag_max):
            tab = tab[np.isfinite(tab["phot_g_mean_mag"]) & (tab["phot_g_mean_mag"] <= mag_max)]
        return tab

    def run(self):
        try:
            P = self.params.P
            result_dir = self.result_dir
            output_dir = step6_dir(result_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            cache_dir = self.cache_dir

            # frames
            frames = list(self.file_list)
            if not frames:
                raise RuntimeError("[IDMatch] 대상 프레임이 없습니다.")

            pix_scale_arcsec = _safe_float(getattr(P, "pixel_scale_arcsec", np.nan))
            if not np.isfinite(pix_scale_arcsec) or pix_scale_arcsec <= 0:
                raise RuntimeError("[IDMatch] pixel_scale_arcsec 미확정")

            g_match = _safe_float(getattr(P, "idmatch_gaia_g_limit", 18.0), 18.0)
            tol_arcsec = _safe_float(getattr(P, "idmatch_tol_arcsec", np.nan))
            if not np.isfinite(tol_arcsec) or tol_arcsec <= 0:
                tol_px = _safe_float(getattr(P, "idmatch_tol_px", 2.0), 2.0)
                tol_arcsec = tol_px * pix_scale_arcsec

            resume = _as_bool(getattr(P, "resume_mode", True), True)
            force = _as_bool(getattr(P, "force_idmatch", False), False)

            # Gaia cache ensure
            gaia_dir = step5_dir(result_dir)
            gaia_cache = gaia_dir / "gaia_fov.ecsv"
            if not gaia_cache.exists():
                legacy_gaia = result_dir / "gaia_fov.ecsv"
                if legacy_gaia.exists():
                    gaia_cache = legacy_gaia
            if not gaia_cache.exists():
                raise FileNotFoundError("[IDMatch] gaia_fov.ecsv가 없습니다. WCS Solve 먼저 실행하세요.")

            t_gaia = Table.read(str(gaia_cache), format="ascii.ecsv")
            t_gaia = _table_cols_lower(t_gaia)

            req = ["source_id", "ra", "dec", "phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag", "pmra", "pmdec"]
            missing = [c for c in req if c not in t_gaia.colnames]

            if missing:
                # Requery Gaia if columns missing
                ra0 = float(getattr(P, "target_ra_deg", 0.0) or 0.0)
                dec0 = float(getattr(P, "target_dec_deg", 0.0) or 0.0)
                if np.isfinite(ra0) and np.isfinite(dec0) and _HAS_GAIA:
                    rad_scale = float(getattr(P, "platesolve_gaia_radius_scale", 1.35))
                    max_r = 0.0
                    for fn in frames[:min(len(frames), 20)]:
                        fpath = result_dir / fn
                        if not fpath.exists():
                            fpath = self.data_dir / fn
                        try:
                            ny, nx = fits.getdata(fpath, memmap=False).shape
                        except Exception:
                            continue
                        fov_w_deg = (nx * pix_scale_arcsec) / 3600.0
                        fov_h_deg = (ny * pix_scale_arcsec) / 3600.0
                        r = 0.5 * float(np.hypot(fov_w_deg, fov_h_deg))
                        max_r = max(max_r, r)
                    radius_deg = max_r * rad_scale
                    center = SkyCoord(ra=ra0 * u.deg, dec=dec0 * u.deg)
                    t_gaia = self._query_gaia(center, radius_deg, g_match)
                    t_gaia = _table_cols_lower(t_gaia)
                    t_gaia.write(str(gaia_cache), format="ascii.ecsv", overwrite=True)
                else:
                    raise RuntimeError(f"[IDMatch] gaia_fov.ecsv 누락컬럼: {missing}")

            t_use = t_gaia
            if "phot_g_mean_mag" in t_use.colnames and np.isfinite(g_match):
                mask = np.isfinite(t_use["phot_g_mean_mag"]) & (t_use["phot_g_mean_mag"] <= g_match)
                t_use = t_use[mask]

            gaia_sky = SkyCoord(
                ra=np.asarray(t_use["ra"], float) * u.deg,
                dec=np.asarray(t_use["dec"], float) * u.deg
            )

            out_dir = cache_dir / "idmatch"
            out_dir.mkdir(parents=True, exist_ok=True)

            rows_summary = []
            all_master_ids = set()
            empty_cols = [
                "file", "source_id", "x", "y", "ra_deg", "dec_deg", "sep_arcsec",
                "gaia_G", "gaia_BP", "gaia_RP", "gmag", "bpmag", "rpmag",
                "pmra_masyr", "pmdec_masyr", "pmra_err", "pmdec_err"
            ]

            total = len(frames)
            for k, fname in enumerate(frames, 1):
                if self._stop_requested:
                    break

                path = self._resolve_fits_path(fname, require_wcs=True)
                if path is None:
                    continue

                out_csv = out_dir / f"idmatch_{fname}.csv"
                out_json = out_dir / f"idmatch_{fname}.json"

                use_cache = resume and (not force) and out_csv.exists() and out_json.exists() and _is_up_to_date(out_csv, [path, gaia_cache])
                meta = None
                if use_cache:
                    try:
                        meta = json.loads(out_json.read_text(encoding="utf-8"))
                    except Exception:
                        meta = dict(file=fname, ok=True, reason="cache", n_det=None, n_matched=None, tol_arcsec=float(tol_arcsec))
                    if str(meta.get("reason", "")) == "no_wcs":
                        wcs_path = self._resolve_fits_path(fname, require_wcs=True)
                        if wcs_path is not None:
                            try:
                                hdr_chk = fits.getheader(wcs_path)
                                w_chk = WCS(hdr_chk, relax=True)
                                if getattr(w_chk, "has_celestial", False):
                                    use_cache = False
                            except Exception:
                                pass
                if use_cache:
                    rows_summary.append(meta)
                    try:
                        df_cached = pd.read_csv(out_csv, usecols=["source_id"])
                        if len(df_cached):
                            all_master_ids.update(df_cached["source_id"].astype("int64").tolist())
                    except Exception:
                        pass
                    self.progress.emit(k, total, fname)
                    continue

                try:
                    hdr = fits.getheader(path)
                    w = self._load_wcs_from_file(path, hdr)
                except Exception:
                    w = None

                if w is None:
                    meta = dict(file=fname, ok=False, reason="no_wcs", n_det=0, n_matched=0, tol_arcsec=float(tol_arcsec))
                    out_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
                    pd.DataFrame([], columns=empty_cols).to_csv(out_csv, index=False)
                    rows_summary.append(meta)
                    self.progress.emit(k, total, fname)
                    continue

                xy = self._load_detect_xy(fname)
                n_det = int(len(xy))
                if n_det == 0:
                    meta = dict(file=fname, ok=False, reason="no_detect", n_det=0, n_matched=0, tol_arcsec=float(tol_arcsec))
                    out_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
                    pd.DataFrame([], columns=empty_cols).to_csv(out_csv, index=False)
                    rows_summary.append(meta)
                    self.progress.emit(k, total, fname)
                    continue

                ra, dec = w.all_pix2world(xy[:, 0], xy[:, 1], 0)
                sky_det = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
                if len(gaia_sky) == 0:
                    idx = np.zeros(len(sky_det), dtype=int)
                    sep_arc = np.full(len(sky_det), np.inf, dtype=float)
                    ok = np.zeros(len(sky_det), dtype=bool)
                else:
                    idx, sep2d, _ = sky_det.match_to_catalog_sky(gaia_sky)
                    sep_arc = sep2d.arcsec
                    ok = np.isfinite(sep_arc) & (sep_arc <= tol_arcsec)

                matched_det_idx = np.where(ok)[0]
                if np.any(ok):
                    df_match = pd.DataFrame({
                        "det_idx": matched_det_idx,
                        "gaia_idx": idx[ok],
                        "sep_arcsec": sep_arc[ok],
                    }).sort_values("sep_arcsec").drop_duplicates("gaia_idx", keep="first")
                    matched_det_idx = df_match["det_idx"].to_numpy(int)
                else:
                    df_match = pd.DataFrame(columns=["det_idx", "gaia_idx", "sep_arcsec"])

                local_df = self._match_unmatched_to_local(sky_det, matched_det_idx, tol_arcsec)

                source_id = np.full(n_det, np.nan)
                ra_out = np.full(n_det, np.nan)
                dec_out = np.full(n_det, np.nan)
                sep_out = np.full(n_det, np.nan)

                gaia_G = np.full(n_det, np.nan)
                gaia_BP = np.full(n_det, np.nan)
                gaia_RP = np.full(n_det, np.nan)
                pmra = np.full(n_det, np.nan)
                pmdec = np.full(n_det, np.nan)
                pmra_e = np.full(n_det, np.nan)
                pmde_e = np.full(n_det, np.nan)

                if not df_match.empty:
                    di = df_match["det_idx"].to_numpy(int)
                    gi = df_match["gaia_idx"].to_numpy(int)
                    sep_out[di] = df_match["sep_arcsec"].to_numpy(float)
                    source_id[di] = np.asarray(t_use["source_id"])[gi].astype(np.int64, copy=False)
                    ra_out[di] = np.asarray(t_use["ra"])[gi].astype(float)
                    dec_out[di] = np.asarray(t_use["dec"])[gi].astype(float)
                    if "phot_g_mean_mag" in t_use.colnames:
                        gaia_G[di] = np.asarray(t_use["phot_g_mean_mag"])[gi]
                    if "phot_bp_mean_mag" in t_use.colnames:
                        gaia_BP[di] = np.asarray(t_use["phot_bp_mean_mag"])[gi]
                    if "phot_rp_mean_mag" in t_use.colnames:
                        gaia_RP[di] = np.asarray(t_use["phot_rp_mean_mag"])[gi]
                    if "pmra" in t_use.colnames:
                        pmra[di] = np.asarray(t_use["pmra"])[gi]
                    if "pmdec" in t_use.colnames:
                        pmdec[di] = np.asarray(t_use["pmdec"])[gi]
                    if "pmra_error" in t_use.colnames:
                        pmra_e[di] = np.asarray(t_use["pmra_error"])[gi]
                    if "pmdec_error" in t_use.colnames:
                        pmde_e[di] = np.asarray(t_use["pmdec_error"])[gi]

                if not local_df.empty:
                    li = local_df["det_idx"].to_numpy(int)
                    source_id[li] = local_df["source_id"].to_numpy(int)
                    sep_out[li] = local_df["sep_arcsec"].to_numpy(float)
                    ra_out[li] = np.asarray(ra)[li]
                    dec_out[li] = np.asarray(dec)[li]

                df = pd.DataFrame({
                    "file": fname,
                    "source_id": source_id,
                    "x": xy[:, 0],
                    "y": xy[:, 1],
                    "ra_deg": ra_out,
                    "dec_deg": dec_out,
                    "sep_arcsec": sep_out,
                    "gaia_G": gaia_G,
                    "gaia_BP": gaia_BP,
                    "gaia_RP": gaia_RP,
                    "gmag": gaia_G,
                    "bpmag": gaia_BP,
                    "rpmag": gaia_RP,
                    "pmra_masyr": pmra,
                    "pmdec_masyr": pmdec,
                    "pmra_err": pmra_e,
                    "pmdec_err": pmde_e,
                })

                df = df.dropna(subset=["source_id", "x", "y"]).copy()
                df["source_id"] = pd.to_numeric(df["source_id"], errors="coerce").astype("int64")
                df.to_csv(out_csv, index=False)

                n_gaia = int(len(df_match))
                n_local = int(len(local_df))
                n_matched = int(len(df))
                sep_vals = pd.to_numeric(df.get("sep_arcsec", np.nan), errors="coerce").to_numpy(float)
                sep_med = float(np.nanmedian(sep_vals)) if n_matched and np.isfinite(sep_vals).any() else np.nan
                sep_max = float(np.nanmax(sep_vals)) if n_matched and np.isfinite(sep_vals).any() else np.nan
                meta = dict(
                    file=fname, ok=True, reason="ok",
                    n_det=n_det, n_matched=n_matched, n_gaia=n_gaia, n_local=n_local,
                    tol_arcsec=float(tol_arcsec),
                    sep_med_arcsec=sep_med,
                    sep_max_arcsec=sep_max,
                )
                out_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
                rows_summary.append(meta)

                all_master_ids.update(df["source_id"].astype("int64").tolist())

                self.file_done.emit(fname, meta)
                self.progress.emit(k, total, fname)

            summary_csv = output_dir / "idmatch_summary.csv"
            pd.DataFrame(rows_summary).to_csv(summary_csv, index=False)

            if len(all_master_ids) == 0:
                for p in sorted(out_dir.glob("idmatch_*.csv")):
                    try:
                        df = pd.read_csv(p, usecols=["source_id"])
                        if len(df):
                            all_master_ids.update(df["source_id"].astype("int64").tolist())
                    except Exception:
                        pass

            master_csv = output_dir / "master_star_ids.csv"
            pd.DataFrame({"source_id": sorted(all_master_ids)}).to_csv(master_csv, index=False)

            reason_counts = {}
            for meta in rows_summary:
                reason = str(meta.get("reason", ""))
                reason_counts[reason] = reason_counts.get(reason, 0) + 1

            self.finished.emit({
                "total": len(rows_summary),
                "matched": len(all_master_ids),
                "reason_counts": reason_counts,
                "tol_arcsec": float(tol_arcsec),
            })
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            self.error.emit("WORKER", error_msg)
            self.finished.emit({})


class StarIdMatchingWindow(StepWindowBase):
    """Step 6: Star ID Matching"""

    def __init__(self, params, file_manager, project_state, main_window):
        self.file_manager = file_manager
        self.worker = None
        self.results = {}
        self.log_window = None
        self.file_list = []
        self.use_cropped = False

        super().__init__(
            step_index=5,
            step_name="Catalog Matching",
            params=params,
            project_state=project_state,
            main_window=main_window
        )

        self.setup_step_ui()
        self.restore_state()

    def setup_step_ui(self):
        info = QLabel("Match detected sources to Gaia source_id using WCS.")
        info.setStyleSheet("QLabel { background-color: #E3F2FD; padding: 10px; border-radius: 5px; }")
        self.content_layout.addWidget(info)

        control_layout = QHBoxLayout()
        btn_params = QPushButton("ID Match Parameters")
        btn_params.setStyleSheet("QPushButton { background-color: #9C27B0; color: white; font-weight: bold; padding: 8px 15px; }")
        btn_params.clicked.connect(self.open_parameters_dialog)
        control_layout.addWidget(btn_params)

        control_layout.addStretch()

        self.btn_run = QPushButton("Run ID Match")
        self.btn_run.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px 20px; }")
        self.btn_run.clicked.connect(self.run_match)
        control_layout.addWidget(self.btn_run)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; padding: 8px 15px; }")
        self.btn_stop.clicked.connect(self.stop_match)
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

        results_group = QGroupBox("ID Match Summary")
        results_layout = QVBoxLayout(results_group)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(6)
        self.results_table.setHorizontalHeaderLabels([
            "File", "OK", "N_det", "N_matched", "Reason", "Sep_med (arcsec)"
        ])
        self.results_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        results_layout.addWidget(self.results_table)
        self.content_layout.addWidget(results_group)

        self.setup_log_window()
        self.populate_file_list()

    def setup_log_window(self):
        if self.log_window is not None:
            return
        self.log_window = QWidget(self, Qt.Window)
        self.log_window.setWindowTitle("ID Match Log")
        self.log_window.resize(800, 400)
        layout = QVBoxLayout(self.log_window)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("QTextEdit { font-family: monospace; font-size: 9pt; }")
        layout.addWidget(self.log_text)

    def show_log_window(self):
        if self.log_window is None:
            self.setup_log_window()
        self.log_window.show()
        self.log_window.raise_()
        self.log_window.activateWindow()

    def log(self, message: str):
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")

    def populate_file_list(self):
        cropped_dir = step2_cropped_dir(self.params.P.result_dir)
        legacy_cropped = self.params.P.result_dir / "cropped"
        crop_active = crop_is_active(self.params.P.result_dir)
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
        dialog.setWindowTitle("ID Match Parameters")
        dialog.resize(460, 320)

        layout = QVBoxLayout(dialog)
        form = QFormLayout()

        self.param_g_limit = QDoubleSpinBox()
        self.param_g_limit.setRange(10.0, 22.0)
        self.param_g_limit.setSingleStep(0.5)
        self.param_g_limit.setValue(float(getattr(self.params.P, "idmatch_gaia_g_limit", 18.0)))
        form.addRow("Gaia G Limit:", self.param_g_limit)

        self.param_tol_arcsec = QDoubleSpinBox()
        self.param_tol_arcsec.setRange(0.0, 10.0)
        self.param_tol_arcsec.setSingleStep(0.1)
        tol_arcsec = _safe_float(getattr(self.params.P, "idmatch_tol_arcsec", np.nan))
        self.param_tol_arcsec.setValue(tol_arcsec if np.isfinite(tol_arcsec) else 0.0)
        form.addRow("Match Tol (arcsec):", self.param_tol_arcsec)

        self.param_tol_px = QDoubleSpinBox()
        self.param_tol_px.setRange(0.5, 10.0)
        self.param_tol_px.setSingleStep(0.5)
        self.param_tol_px.setValue(float(getattr(self.params.P, "idmatch_tol_px", 2.0)))
        form.addRow("Match Tol (px):", self.param_tol_px)

        self.param_force = QCheckBox("Force re-match")
        self.param_force.setChecked(bool(getattr(self.params.P, "force_idmatch", False)))
        form.addRow("Force:", self.param_force)

        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(lambda: self.save_parameters(dialog))
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec_()

    def save_parameters(self, dialog):
        self.params.P.idmatch_gaia_g_limit = self.param_g_limit.value()
        tol_arcsec = float(self.param_tol_arcsec.value())
        self.params.P.idmatch_tol_arcsec = tol_arcsec if tol_arcsec > 0 else None
        self.params.P.idmatch_tol_px = self.param_tol_px.value()
        self.params.P.force_idmatch = self.param_force.isChecked()
        self.save_state()
        QMessageBox.information(dialog, "Success", "Parameters saved!")
        dialog.accept()

    def run_match(self):
        if not self.file_list:
            QMessageBox.warning(self, "Warning", "No files to process")
            return
        if self.worker and self.worker.isRunning():
            return

        self.results = {}
        self.results_table.setRowCount(0)
        self.log_text.clear()

        self.worker = IdMatchWorker(
            self.file_list,
            self.params,
            self.params.P.data_dir,
            self.params.P.result_dir,
            self.params.P.cache_dir,
            self.use_cropped
        )
        self.worker.progress.connect(self.on_progress)
        self.worker.file_done.connect(self.on_file_done)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)

        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(len(self.file_list))
        self.progress_label.setText(f"0/{len(self.file_list)} | Starting...")
        self.worker.start()
        self.show_log_window()

    def stop_match(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()

    def on_progress(self, current, total, filename):
        self.progress_bar.setValue(current)
        self.progress_label.setText(f"{current}/{total} | {filename}")

    def on_file_done(self, filename, result):
        self.results[filename] = result
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)
        self.results_table.setItem(row, 0, QTableWidgetItem(filename))
        self.results_table.setItem(row, 1, QTableWidgetItem(str(result.get("ok", False))))
        self.results_table.setItem(row, 2, QTableWidgetItem(str(result.get("n_det", ""))))
        self.results_table.setItem(row, 3, QTableWidgetItem(str(result.get("n_matched", ""))))
        self.results_table.setItem(row, 4, QTableWidgetItem(str(result.get("reason", ""))))
        sep = result.get("sep_med_arcsec")
        sep_str = f"{sep:.3f}" if isinstance(sep, float) and np.isfinite(sep) else "-"
        self.results_table.setItem(row, 5, QTableWidgetItem(sep_str))
        self.log(f"{filename}: ok={result.get('ok', False)} matched={result.get('n_matched')}")

    def on_error(self, filename, error):
        self.log(f"ERROR {filename}: {error}")

    def on_finished(self, summary):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_label.setText("Done")
        if summary:
            n_master = summary.get("matched", 0)
            tol_arcsec = summary.get("tol_arcsec", None)
            tol_str = f"{tol_arcsec:.3f}\"" if isinstance(tol_arcsec, float) and np.isfinite(tol_arcsec) else "-"
            self.log(f"ID match done: N_master={n_master} | tol={tol_str}")
            rc = summary.get("reason_counts", {})
            if rc:
                parts = [f"{k}:{v}" for k, v in sorted(rc.items())]
                self.log(f"Reasons: {', '.join(parts)}")
            if n_master == 0:
                self.log("Hint: check WCS solved + detection cache + match tolerance / Gaia limit.")
        self.save_state()
        self.update_navigation_buttons()

    def validate_step(self) -> bool:
        master_path = step6_dir(self.params.P.result_dir) / "master_star_ids.csv"
        if not master_path.exists():
            master_path = self.params.P.result_dir / "master_star_ids.csv"
        return master_path.exists()

    def save_state(self):
        state_data = {
            "idmatch_complete": (step6_dir(self.params.P.result_dir) / "master_star_ids.csv").exists()
            or (self.params.P.result_dir / "master_star_ids.csv").exists(),
            "idmatch_gaia_g_limit": getattr(self.params.P, "idmatch_gaia_g_limit", 18.0),
            "idmatch_tol_arcsec": getattr(self.params.P, "idmatch_tol_arcsec", np.nan),
            "idmatch_tol_px": getattr(self.params.P, "idmatch_tol_px", 2.0),
            "force_idmatch": getattr(self.params.P, "force_idmatch", False),
        }
        self.project_state.store_step_data("star_id_match", state_data)

    def restore_state(self):
        state_data = self.project_state.get_step_data("star_id_match")
        if state_data:
            for key, val in state_data.items():
                if hasattr(self.params.P, key):
                    setattr(self.params.P, key, val)
