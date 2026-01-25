"""
Step 5: WCS Plate Solving (ASTAP)
Minimal GUI wrapper for ASTAP-based WCS solving with cache/summary output.
"""

from __future__ import annotations

import json
import time
import subprocess
import threading
import traceback
import warnings
import os
import shlex
import shutil
from pathlib import Path, PureWindowsPath
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table

try:
    from astroquery.gaia import Gaia
    _HAS_GAIA = True
except Exception:
    _HAS_GAIA = False

from scipy.spatial import cKDTree as KDTree

from PyQt5.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QGroupBox, QMessageBox,
    QTextEdit, QComboBox, QDialog, QFormLayout, QLineEdit, QDialogButtonBox,
    QProgressBar, QCheckBox, QSpinBox, QDoubleSpinBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QAbstractItemView, QWidget, QTabWidget,
    QListWidget, QListWidgetItem
)

# Astrometry.net support
try:
    from astroquery.astrometry_net import AstrometryNet
    _HAS_ASTROMETRY_NET = True
except Exception:
    _HAS_ASTROMETRY_NET = False
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from .step_window_base import StepWindowBase
from ...utils.step_paths import step2_cropped_dir, step5_dir, crop_is_active
from ...utils.constants import get_parallel_workers


class WcsWorker(QThread):
    """Worker thread for ASTAP WCS solving"""
    progress = pyqtSignal(int, int, str)
    file_done = pyqtSignal(str, dict)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str, str)

    def __init__(self, file_list, params, data_dir, result_dir, cache_dir,
                 use_cropped=False, target_coord=None):
        super().__init__()
        self.file_list = file_list
        self.params = params
        self.data_dir = Path(data_dir)
        self.result_dir = Path(result_dir)
        self.cache_dir = Path(cache_dir)
        self.use_cropped = use_cropped
        self.target_coord = target_coord
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def _resolve_exe(self, exe: str) -> str:
        p = Path(exe)
        if p.exists():
            return str(p)
        return exe

    def _astap_wcs_candidates(self, fits_path: Path):
        return [
            fits_path.with_suffix(".wcs"),
            Path(str(fits_path) + ".wcs"),
            fits_path.parent / (fits_path.stem + ".wcs"),
            fits_path.parent / (fits_path.name + ".wcs"),
        ]

    def _parse_astap_wcs_file(self, wcs_path: Path) -> dict:
        d = {}
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

    def _inject_wcs_into_header(self, hdr: fits.Header, wcs_dict: dict):
        for k, v in wcs_dict.items():
            try:
                hdr[k] = v
            except Exception:
                pass

    def _try_ingest_wcs(self, fits_path: Path, hdr: fits.Header) -> bool:
        for wp in self._astap_wcs_candidates(fits_path):
            if not wp.exists():
                continue
            wcsd = self._parse_astap_wcs_file(wp)
            if not wcsd:
                continue
            self._inject_wcs_into_header(hdr, wcsd)
            try:
                w = WCS(hdr, relax=True)
                if w.has_celestial:
                    return True
            except Exception:
                pass
        # Fallback: ASTAP이 직접 FITS에 WCS를 쓴 경우 (헤더 재로드)
        try:
            with fits.open(fits_path, memmap=False) as hdul_check:
                hdr_check = hdul_check[0].header
                w_check = WCS(hdr_check, relax=True)
                if w_check.has_celestial:
                    # WCS 키워드 복사
                    for key in hdr_check.keys():
                        if key.startswith(('CRVAL', 'CRPIX', 'CDELT', 'CD1_', 'CD2_',
                                          'CTYPE', 'CUNIT', 'CROTA', 'PC1_', 'PC2_')):
                            hdr[key] = hdr_check[key]
                    return True
        except Exception:
            pass
        return False

    def _pixscale_from_wcs(self, w: WCS) -> float:
        try:
            sc = proj_plane_pixel_scales(w.celestial) * 3600.0
            return float(np.mean(sc))
        except Exception:
            return float("nan")

    def _load_fwhm_for_frame(self, fname: str):
        meta_json = self.cache_dir / f"detect_{fname}.json"
        if meta_json.exists():
            try:
                meta = json.loads(meta_json.read_text(encoding="utf-8"))
                fpx = float(meta.get("fwhm_med_rad_px", meta.get("fwhm_med_px", np.nan)))
                farc = float(meta.get("fwhm_med_rad_arcsec", meta.get("fwhm_med_arc", np.nan)))
                return fpx, farc
            except Exception:
                pass
        return float(getattr(self.params.P, "fwhm_seed_px", 6.0)), np.nan

    def _load_gaia_cache_if_ok(self, path: Path):
        if not path.exists():
            return None
        try:
            tab = Table.read(path, format="ascii.ecsv")
            cols = [c.lower() for c in tab.colnames]
            for need in ("source_id", "ra", "dec"):
                if need not in cols:
                    return None
            tab.rename_columns(tab.colnames, cols)
            return tab.to_pandas()
        except Exception:
            return None

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
        return tab.to_pandas()

    def _load_or_query_gaia(self, center: SkyCoord, radius_deg: float):
        output_dir = step5_dir(self.result_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        cache_path = output_dir / "gaia_fov.ecsv"
        meta_path = output_dir / "gaia_fov_meta.json"
        legacy_cache = self.result_dir / "gaia_fov.ecsv"
        legacy_meta = self.result_dir / "gaia_fov_meta.json"
        read_cache = cache_path if cache_path.exists() else legacy_cache
        read_meta = meta_path if meta_path.exists() else legacy_meta
        retry = int(getattr(self.params.P, "gaia_retry", 2))
        backoff_s = float(getattr(self.params.P, "gaia_backoff_s", 6.0))
        mag_max = float(getattr(self.params.P, "gaia_mag_max", 18.0))
        allow_no_cache = bool(getattr(self.params.P, "gaia_allow_no_cache", True))

        # 캐시 유효성 체크 - 좌표가 맞는지 확인
        cache_valid = False
        df_cache = self._load_gaia_cache_if_ok(read_cache)
        if df_cache is not None and read_meta.exists():
            try:
                meta = json.loads(read_meta.read_text(encoding="utf-8"))
                cached_ra = float(meta.get("center_ra_deg", 0))
                cached_dec = float(meta.get("center_dec_deg", 0))
                cached_radius = float(meta.get("radius_deg", 0))
                # 중심 좌표가 0.1도 이내이고 반경이 비슷하면 캐시 사용
                dist_deg = np.hypot(center.ra.deg - cached_ra, center.dec.deg - cached_dec)
                if dist_deg < 0.1 and cached_radius >= radius_deg * 0.9:
                    cache_valid = True
            except Exception:
                pass
        elif df_cache is not None:
            # 메타 파일이 없으면 일단 캐시 사용 (이전 버전 호환)
            cache_valid = True

        if cache_valid and df_cache is not None:
            return df_cache, "cache"
        if not _HAS_GAIA:
            if allow_no_cache:
                return pd.DataFrame(), "no_gaia_module"
            raise RuntimeError("astroquery.gaia not available and no cache")

        last_err = None
        for att in range(1, max(1, retry) + 1):
            try:
                df = self._query_gaia(center, radius_deg, mag_max)
                df.columns = [c.lower() for c in df.columns]
                try:
                    Table.from_pandas(df).write(cache_path, format="ascii.ecsv", overwrite=True)
                    # 메타데이터 저장
                    meta_path.write_text(json.dumps({
                        "center_ra_deg": float(center.ra.deg),
                        "center_dec_deg": float(center.dec.deg),
                        "radius_deg": float(radius_deg),
                        "mag_max": float(mag_max),
                        "n_stars": len(df),
                    }, indent=2), encoding="utf-8")
                except Exception:
                    pass
                return df, "query"
            except Exception as e:
                last_err = e
                if att < retry:
                    time.sleep(backoff_s)

        df_cache = self._load_gaia_cache_if_ok(read_cache)
        if df_cache is not None:
            return df_cache, "cache(after_fail)"
        if allow_no_cache:
            return pd.DataFrame(), f"fail_no_cache:{type(last_err).__name__}"
        raise RuntimeError(f"Gaia query failed: {last_err}")

    def _refine_crpix_by_match(self, w: WCS, hdr: fits.Header, det_xy: np.ndarray,
                               gaia_df: pd.DataFrame, fwhm_px: float, max_match: int):
        if w is None or (not w.has_celestial):
            return False, "no_wcs", np.nan, np.nan, 0
        if det_xy.size == 0:
            return False, "no_det", np.nan, np.nan, 0
        if gaia_df is None or len(gaia_df) == 0:
            return False, "gaia_unavailable", np.nan, np.nan, 0

        try:
            ra = gaia_df["ra"].to_numpy(float)
            dec = gaia_df["dec"].to_numpy(float)
        except Exception:
            return False, "gaia_cols_missing", np.nan, np.nan, 0

        nx = int(hdr.get("NAXIS1", 0))
        ny = int(hdr.get("NAXIS2", 0))
        if nx <= 0 or ny <= 0:
            return False, "bad_shape", np.nan, np.nan, 0

        try:
            xg, yg = w.celestial.world_to_pixel(SkyCoord(ra * u.deg, dec * u.deg))
            xg = np.asarray(xg, float)
            yg = np.asarray(yg, float)
            okb = np.isfinite(xg) & np.isfinite(yg) & (xg >= 0) & (xg < nx) & (yg >= 0) & (yg < ny)
            if okb.sum() == 0:
                return False, "gaia_outside", np.nan, np.nan, 0
            gaia_xy = np.vstack([xg[okb], yg[okb]]).T
        except Exception as e:
            return False, f"world2pix_fail:{e}", np.nan, np.nan, 0

        r_match = max(3.0, float(fwhm_px) * float(getattr(self.params.P, "wcs_refine_match_r_fwhm", 1.6)))
        tree = KDTree(gaia_xy)
        d, j = tree.query(det_xy, k=1)
        m = np.isfinite(d) & (d <= r_match)
        min_match = int(getattr(self.params.P, "wcs_refine_min_match", 50))
        if m.sum() < min_match:
            return False, f"match_too_small:{m.sum()}", np.nan, np.nan, int(m.sum())

        det_m = det_xy[m]
        gaia_m = gaia_xy[j[m]]

        if det_m.shape[0] > int(max_match):
            order = np.argsort(d[m])[:int(max_match)]
            det_m, gaia_m = det_m[order], gaia_m[order]

        dx = det_m[:, 0] - gaia_m[:, 0]
        dy = det_m[:, 1] - gaia_m[:, 1]
        dx_med = float(np.median(dx))
        dy_med = float(np.median(dy))

        if "CRPIX1" in hdr and "CRPIX2" in hdr:
            hdr["CRPIX1"] = float(hdr["CRPIX1"]) + dx_med
            hdr["CRPIX2"] = float(hdr["CRPIX2"]) + dy_med
        else:
            return False, "no_crpix", np.nan, np.nan, det_m.shape[0]

        w2 = WCS(hdr, relax=True)
        pix_fit = self._pixscale_from_wcs(w2)
        if not np.isfinite(pix_fit):
            pix_fit = float(getattr(self.params.P, "pixel_scale_arcsec", np.nan))

        resid_arc = np.hypot(dx - dx_med, dy - dy_med) * float(pix_fit)
        resid_med = float(np.median(resid_arc)) if resid_arc.size else np.nan
        resid_max = float(np.max(resid_arc)) if resid_arc.size else np.nan
        return True, f"m1={det_m.shape[0]}", resid_med, resid_max, int(det_m.shape[0])

    def _run_astap(self, fits_path: Path, fov_deg: float, radius_deg: float, timeout_s: float):
        exe = self._resolve_exe(str(getattr(self.params.P, "astap_exe", "astap_cli.exe")))
        db = str(getattr(self.params.P, "astap_database", "") or "").strip()
        z = int(getattr(self.params.P, "astap_downsample_z", 2))
        s = int(getattr(self.params.P, "astap_max_stars_s", 500))
        cmd = [
            exe,
            "-f", str(fits_path),
            "-fov", f"{fov_deg:.6f}",
            "-r", f"{radius_deg:.3f}",
            "-z", str(z),
            "-s", str(s),
        ]
        if db:
            cmd += ["-D", db]
        try:
            start = time.time()
            cp = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
            dt = time.time() - start
            ok = (cp.returncode == 0)
            return ok, cp.returncode, dt, cp.stdout, cp.stderr, cmd
        except subprocess.TimeoutExpired:
            return False, -999, timeout_s, "", "timeout", cmd
        except Exception as e:
            return False, -998, 0.0, "", str(e), cmd

    def run(self):
        try:
            warnings.filterwarnings("ignore", message="Keyword name*HIERARCH*")
            warnings.filterwarnings("ignore", message="The WCS transformation has more axes*")

            results = []
            files = list(self.file_list)
            total = len(files)
            pix_arc = float(getattr(self.params.P, "pixel_scale_arcsec", np.nan))
            if not np.isfinite(pix_arc) or pix_arc <= 0:
                raise RuntimeError("pixel_scale_arcsec is not set; run instrument setup first.")

            output_dir = step5_dir(self.result_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            cropped_dir = step2_cropped_dir(self.result_dir)
            if not cropped_dir.exists():
                cropped_dir = self.result_dir / "cropped"

            # Optional QC filtering
            require_qc = bool(getattr(self.params.P, "wcs_require_qc_pass", True))
            if require_qc:
                qpath = output_dir / "frame_quality.csv"
                if not qpath.exists():
                    legacy_qpath = self.result_dir / "frame_quality.csv"
                    if legacy_qpath.exists():
                        qpath = legacy_qpath
                if qpath.exists():
                    try:
                        dfq = pd.read_csv(qpath)
                        good = set(dfq.loc[dfq["passed"] == True, "file"].astype(str).tolist())
                        files = [f for f in files if f in good]
                    except Exception:
                        pass

            if not files:
                raise RuntimeError("No files to process (QC filter removed all files).")

            astap_timeout = float(getattr(self.params.P, "astap_timeout_s", 120.0))
            astap_radius = float(getattr(self.params.P, "astap_search_radius_deg", 8.0))
            astap_db = str(getattr(self.params.P, "astap_database", "D50") or "").strip()
            astap_fov_fudge = float(getattr(self.params.P, "astap_fov_fudge", 1.0))
            meta_dir = self.cache_dir / "wcs_solve"
            meta_dir.mkdir(parents=True, exist_ok=True)
            log_path = self.cache_dir / "wcs_solve.log"

            def L(msg):
                try:
                    with open(log_path, "a", encoding="utf-8") as fh:
                        fh.write(msg + "\n")
                except Exception:
                    pass

            L(f"[WCS] astap_timeout_s={astap_timeout} astap_radius_deg={astap_radius} astap_db={astap_db or 'default'} astap_fov_fudge={astap_fov_fudge}")

            # Determine Gaia center - PRIORITY: FITS header > project_state
            # FITS header OBJCTRA/OBJCTDEC is more reliable as it comes from the actual observation
            header_coord = None
            try:
                sample = files[0]
                if self.use_cropped:
                    fits_path = cropped_dir / sample
                else:
                    fits_path = self.data_dir / sample
                with fits.open(fits_path, memmap=False) as hdul:
                    hdr = hdul[0].header
                    ra0 = hdr.get("OBJCTRA", None)
                    dec0 = hdr.get("OBJCTDEC", None)
                    if ra0 is not None and dec0 is not None:
                        header_coord = SkyCoord(str(ra0), str(dec0), unit=(u.hourangle, u.deg))
            except Exception:
                header_coord = None

            # Decide which coordinate to use
            center_coord = None
            if header_coord is not None and self.target_coord is not None:
                # Both available - check if they match
                sep_deg = float(header_coord.separation(self.target_coord).deg)
                if sep_deg > 5.0:
                    L(f"[WCS] WARNING: FITS header coords differ by {sep_deg:.2f}deg from project_state, using header")
                    center_coord = header_coord
                else:
                    center_coord = self.target_coord
            elif header_coord is not None:
                center_coord = header_coord
            elif self.target_coord is not None:
                center_coord = self.target_coord

            if center_coord is None:
                raise RuntimeError("Target coordinate not set (SIMBAD/OBJCTRA/OBJCTDEC missing).")

            # Gaia query/cache
            gaia_fudge = float(getattr(self.params.P, "gaia_radius_fudge", 1.35))
            sample = files[0]
            if self.use_cropped:
                sample_path = cropped_dir / sample
            else:
                sample_path = self.data_dir / sample
            with fits.open(sample_path, memmap=False, ignore_missing_simple=True) as hdul:
                data0 = hdul[0].data
                if data0 is None:
                    raise RuntimeError("First frame data is None")
                ny0, nx0 = data0.shape
            fov_w = (nx0 * pix_arc) / 3600.0
            fov_h = (ny0 * pix_arc) / 3600.0
            diag_deg = float(np.hypot(fov_w, fov_h))
            gaia_r = float(0.5 * diag_deg * gaia_fudge)
            gaia_df, gaia_src = self._load_or_query_gaia(center_coord, gaia_r)
            L(f"[Gaia] center=({center_coord.ra.deg:.6f},{center_coord.dec.deg:.6f}) r={gaia_r:.4f}deg source={gaia_src} N={len(gaia_df)}")

            def solve_one(filename):
                if self._stop_requested:
                    return filename, None

                if self.use_cropped:
                    fits_path = cropped_dir / filename
                else:
                    fits_path = self.data_dir / filename

                status = "fail"
                pix_fit = np.nan
                wcs_ok = False
                refine_note = ""
                resid_med = np.nan
                resid_max = np.nan
                match_n = 0

                with fits.open(fits_path, memmap=False, ignore_missing_simple=True) as hdul:
                    hdr = hdul[0].header
                    data = hdul[0].data
                    if data is None:
                        return filename, {"ok": False, "status": "data_none"}
                    ny, nx = data.shape

                fov_w_deg = (nx * pix_arc) / 3600.0 * astap_fov_fudge
                fov_h_deg = (ny * pix_arc) / 3600.0 * astap_fov_fudge
                fov_deg = float(max(fov_w_deg, fov_h_deg))

                ok_astap, rc, dt, out_s, err_s, cmd = self._run_astap(
                    fits_path, fov_deg=fov_deg, radius_deg=astap_radius, timeout_s=astap_timeout
                )
                if not ok_astap:
                    L(f"{filename}: ASTAP fail rc={rc} dt={dt:.1f}s err={str(err_s)[:120]}")
                    return filename, {
                        "ok": False,
                        "status": f"astap_fail rc={rc}",
                        "pix_fit": pix_fit,
                        "elapsed": dt,
                        "refine": refine_note,
                    }

                # WCS를 FITS 파일에 저장 (writeto 사용 - Windows 호환성)
                with fits.open(fits_path, memmap=False, ignore_missing_simple=True) as hdul:
                    hdr = hdul[0].header
                    data = hdul[0].data
                    try:
                        w0 = WCS(hdr, relax=True)
                        wcs_ok = w0.has_celestial
                    except Exception:
                        wcs_ok = False

                    if not wcs_ok:
                        wcs_ok = self._try_ingest_wcs(fits_path, hdr)

                    if wcs_ok:
                        w = WCS(hdr, relax=True)
                        pix_fit = self._pixscale_from_wcs(w)

                        refine_enable = bool(getattr(self.params.P, "wcs_refine_enable", True))
                        if refine_enable and gaia_df is not None:
                            det_csv = self.cache_dir / f"detect_{filename}.csv"
                            if det_csv.exists():
                                try:
                                    det_xy = pd.read_csv(det_csv)[["x", "y"]].to_numpy(float)
                                except Exception:
                                    det_xy = np.zeros((0, 2), float)
                            else:
                                det_xy = np.zeros((0, 2), float)

                            fwhm_px, _ = self._load_fwhm_for_frame(filename)
                            ok_ref, note, rmed, rmax, nmatch = self._refine_crpix_by_match(
                                w, hdr, det_xy, gaia_df,
                                fwhm_px=float(fwhm_px),
                                max_match=int(getattr(self.params.P, "wcs_refine_max_match", 600))
                            )
                            refine_note = note
                            if ok_ref:
                                w2 = WCS(hdr, relax=True)
                                pix_fit = self._pixscale_from_wcs(w2)
                                resid_med = rmed
                                resid_max = rmax
                                match_n = nmatch

                        hdr["WCS_OK"] = (True, "WCS solve success")
                        hdr["WCSPIXI"] = (float(pix_arc), "pixscale input (arcsec/pix)")
                        if np.isfinite(pix_fit):
                            hdr["WCSPIXF"] = (float(pix_fit), "pixscale fit (arcsec/pix)")
                        if refine_note:
                            hdr["WCSREFN"] = (str(refine_note)[:68], "refine note")
                        if np.isfinite(resid_med):
                            hdr["WCSRMD"] = (float(resid_med), "ref resid med (arcsec)")
                        if np.isfinite(resid_max):
                            hdr["WCSRMAX"] = (float(resid_max), "ref resid max (arcsec)")
                        status = "ok"
                    else:
                        hdr["WCS_OK"] = (False, "WCS solve failed")
                        status = "wcs_missing"

                # writeto로 확실하게 저장 (Windows 호환)
                fits.writeto(fits_path, data, hdr, overwrite=True)

                meta = {
                    "fname": filename,
                    "ok": bool(wcs_ok),
                    "status": status,
                    "pix_fit": float(pix_fit) if np.isfinite(pix_fit) else None,
                    "elapsed": float(dt),
                    "refine": refine_note,
                    "resid_med": float(resid_med) if np.isfinite(resid_med) else None,
                    "resid_max": float(resid_max) if np.isfinite(resid_max) else None,
                    "match_n": int(match_n),
                    "gaia_source": str(gaia_src),
                }
                L(
                    f"{filename}: {status} pix_fit={pix_fit:.4f} dt={dt:.1f}s "
                    f"refine={refine_note or '-'} resid_med={resid_med if np.isfinite(resid_med) else '-'} "
                    f"match_n={match_n}"
                )
                (meta_dir / f"wcs_{filename}.json").write_text(
                    json.dumps(meta, indent=2), encoding="utf-8"
                )
                return filename, meta

            completed = 0
            max_workers = get_parallel_workers(self.params)  # Central config
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(solve_one, f): f for f in files}
                for fut in as_completed(futures):
                    if self._stop_requested:
                        break
                    fname = futures[fut]
                    completed += 1
                    try:
                        fn, res = fut.result()
                        if res is not None:
                            results.append(res)
                            self.file_done.emit(fn, res)
                        else:
                            self.error.emit(fname, "stopped")
                    except Exception as e:
                        self.error.emit(fname, str(e))
                    self.progress.emit(completed, len(files), fname)

            # Save summary CSV
            try:
                df = pd.DataFrame(results)
                df.to_csv(output_dir / "wcs_solve_summary.csv", index=False)
            except Exception:
                pass

            summary = {
                "total": len(results),
                "ok": sum(1 for r in results if r.get("ok")),
            }
            self.finished.emit(summary)
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            self.error.emit("WORKER", error_msg)
            self.finished.emit({})


class AstrometryNetWorker(QThread):
    """Worker thread for local astrometry.net (solve-field via WSL) WCS solving

    Features:
    - Parallel processing with ThreadPoolExecutor (8 workers)
    - Gaia catalog query and caching (gaia_fov.ecsv)
    - WCS refinement with Gaia matching
    - Residual calculation
    """
    progress = pyqtSignal(int, int, str)
    file_done = pyqtSignal(str, dict)
    refine_done = pyqtSignal(str, dict)
    log_message = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str, str)

    def __init__(self, file_list, params, data_dir, result_dir, cache_dir,
                 use_cropped=False, target_coord=None):
        super().__init__()
        self.file_list = file_list
        self.params = params
        self.data_dir = Path(data_dir)
        self.result_dir = Path(result_dir)
        self.cache_dir = Path(cache_dir)
        self.use_cropped = use_cropped
        self.target_coord = target_coord
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def _pixscale_from_wcs(self, w: WCS):
        try:
            sc = proj_plane_pixel_scales(w.celestial) * 3600.0
            return float(np.mean(sc))
        except Exception:
            return float("nan")

    def _load_gaia_cache_if_ok(self, path: Path):
        if not path.exists():
            return None
        try:
            tab = Table.read(path, format="ascii.ecsv")
            cols = [c.lower() for c in tab.colnames]
            for need in ("source_id", "ra", "dec"):
                if need not in cols:
                    return None
            tab.rename_columns(tab.colnames, cols)
            return tab.to_pandas()
        except Exception:
            return None

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
        return tab.to_pandas()

    def _load_or_query_gaia(self, center: SkyCoord, radius_deg: float):
        step5_out = step5_dir(self.result_dir)
        step5_out.mkdir(parents=True, exist_ok=True)
        cache_path = step5_out / "gaia_fov.ecsv"
        meta_path = step5_out / "gaia_fov_meta.json"
        legacy_cache = self.result_dir / "gaia_fov.ecsv"
        legacy_meta = self.result_dir / "gaia_fov_meta.json"
        retry = int(getattr(self.params.P, "gaia_retry", 2))
        backoff_s = float(getattr(self.params.P, "gaia_backoff_s", 6.0))
        mag_max = float(getattr(self.params.P, "gaia_mag_max", 18.0))
        allow_no_cache = bool(getattr(self.params.P, "gaia_allow_no_cache", True))

        cache_valid = False
        df_cache = None
        meta_probe = None
        for cpath, mpath in ((cache_path, meta_path), (legacy_cache, legacy_meta)):
            df_cache = self._load_gaia_cache_if_ok(cpath)
            if df_cache is not None:
                meta_probe = mpath
                break

        if df_cache is not None and meta_probe is not None and meta_probe.exists():
            try:
                meta = json.loads(meta_probe.read_text(encoding="utf-8"))
                cached_ra = float(meta.get("center_ra_deg", 0))
                cached_dec = float(meta.get("center_dec_deg", 0))
                cached_radius = float(meta.get("radius_deg", 0))
                dist_deg = np.hypot(center.ra.deg - cached_ra, center.dec.deg - cached_dec)
                if dist_deg < 0.1 and cached_radius >= radius_deg * 0.9:
                    cache_valid = True
            except Exception:
                pass
        elif df_cache is not None:
            cache_valid = True

        if cache_valid and df_cache is not None:
            return df_cache, "cache"
        if not _HAS_GAIA:
            if allow_no_cache:
                return pd.DataFrame(), "no_gaia_module"
            raise RuntimeError("astroquery.gaia not available and no cache")

        last_err = None
        for att in range(1, max(1, retry) + 1):
            try:
                df = self._query_gaia(center, radius_deg, mag_max)
                df.columns = [c.lower() for c in df.columns]
                try:
                    Table.from_pandas(df).write(cache_path, format="ascii.ecsv", overwrite=True)
                    meta_path.write_text(json.dumps({
                        "center_ra_deg": float(center.ra.deg),
                        "center_dec_deg": float(center.dec.deg),
                        "radius_deg": float(radius_deg),
                        "mag_max": float(mag_max),
                        "n_stars": len(df),
                    }, indent=2), encoding="utf-8")
                except Exception:
                    pass
                return df, "query"
            except Exception as e:
                last_err = e
                if att < retry:
                    time.sleep(backoff_s)

        df_cache = self._load_gaia_cache_if_ok(cache_path)
        if df_cache is not None:
            return df_cache, "cache(after_fail)"
        if allow_no_cache:
            return pd.DataFrame(), f"fail_no_cache:{type(last_err).__name__}"
        raise RuntimeError(f"Gaia query failed: {last_err}")

    def _load_fwhm_for_frame(self, fname: str):
        meta_json = self.cache_dir / f"detect_{fname}.json"
        if meta_json.exists():
            try:
                meta = json.loads(meta_json.read_text(encoding="utf-8"))
                fpx = float(meta.get("fwhm_med_rad_px", meta.get("fwhm_px", meta.get("fwhm_med_px", np.nan))))
                farc = float(meta.get("fwhm_med_rad_arcsec", meta.get("fwhm_arcsec", meta.get("fwhm_med_arc", np.nan))))
                return fpx, farc
            except Exception:
                pass
        return float(getattr(self.params.P, "fwhm_seed_px", 6.0)), np.nan

    def _load_detect_xy(self, fname: str):
        csv_path = self.cache_dir / f"detect_{fname}.csv"
        if not csv_path.exists():
            return np.empty((0, 2)), None
        try:
            df = pd.read_csv(csv_path)
            xy = df[["x", "y"]].values
            return xy, df
        except Exception:
            return np.empty((0, 2)), None

    def _refine_crpix_by_match(self, w: WCS, hdr: fits.Header, det_xy: np.ndarray,
                               gaia_df: pd.DataFrame, fwhm_px: float, max_match: int):
        if w is None or (not w.has_celestial):
            return False, "no_wcs", np.nan, np.nan, 0
        if det_xy.size == 0:
            return False, "no_det", np.nan, np.nan, 0
        if gaia_df is None or len(gaia_df) == 0:
            return False, "gaia_unavailable", np.nan, np.nan, 0

        try:
            ra = gaia_df["ra"].to_numpy(float)
            dec = gaia_df["dec"].to_numpy(float)
        except Exception:
            return False, "gaia_cols_missing", np.nan, np.nan, 0

        nx = int(hdr.get("NAXIS1", 0))
        ny = int(hdr.get("NAXIS2", 0))
        if nx <= 0 or ny <= 0:
            return False, "bad_shape", np.nan, np.nan, 0

        try:
            xg, yg = w.celestial.world_to_pixel(SkyCoord(ra * u.deg, dec * u.deg))
            xg = np.asarray(xg, float)
            yg = np.asarray(yg, float)
            okb = np.isfinite(xg) & np.isfinite(yg) & (xg >= 0) & (xg < nx) & (yg >= 0) & (yg < ny)
            if okb.sum() == 0:
                return False, "gaia_outside", np.nan, np.nan, 0
            gaia_xy = np.vstack([xg[okb], yg[okb]]).T
        except Exception as e:
            return False, f"world2pix_fail:{e}", np.nan, np.nan, 0

        r_match = max(3.0, float(fwhm_px) * float(getattr(self.params.P, "wcs_refine_match_r_fwhm", 1.6)))
        tree = KDTree(gaia_xy)
        d, j = tree.query(det_xy, k=1)
        m = np.isfinite(d) & (d <= r_match)
        min_match = int(getattr(self.params.P, "wcs_refine_min_match", 50))
        if m.sum() < min_match:
            return False, f"match_too_small:{m.sum()}", np.nan, np.nan, int(m.sum())

        det_m = det_xy[m]
        gaia_m = gaia_xy[j[m]]

        if det_m.shape[0] > int(max_match):
            order = np.argsort(d[m])[:int(max_match)]
            det_m, gaia_m = det_m[order], gaia_m[order]

        dx = det_m[:, 0] - gaia_m[:, 0]
        dy = det_m[:, 1] - gaia_m[:, 1]
        dx_med = float(np.median(dx))
        dy_med = float(np.median(dy))

        if "CRPIX1" in hdr and "CRPIX2" in hdr:
            hdr["CRPIX1"] = float(hdr["CRPIX1"]) + dx_med
            hdr["CRPIX2"] = float(hdr["CRPIX2"]) + dy_med
        else:
            return False, "no_crpix", np.nan, np.nan, det_m.shape[0]

        w2 = WCS(hdr, relax=True)
        pix_fit = self._pixscale_from_wcs(w2)
        if not np.isfinite(pix_fit):
            pix_fit = float(getattr(self.params.P, "pixel_scale_arcsec", np.nan))

        resid_arc = np.hypot(dx - dx_med, dy - dy_med) * float(pix_fit)
        resid_med = float(np.median(resid_arc)) if resid_arc.size else np.nan
        resid_max = float(np.max(resid_arc)) if resid_arc.size else np.nan
        return True, f"m1={det_m.shape[0]}", resid_med, resid_max, int(det_m.shape[0])

    def _win_to_wsl_path(self, path: Path) -> str:
        try:
            wp = PureWindowsPath(str(path))
            if wp.drive:
                drive = wp.drive.rstrip(":").lower()
                parts = "/".join(wp.parts[1:])
                return f"/mnt/{drive}/{parts}"
        except Exception:
            pass
        return str(path).replace("\\", "/")

    def _run_solve_field(
        self,
        fits_path: Path,
        center_coord: SkyCoord | None,
        scale_low: float,
        scale_high: float,
        radius_deg: float,
        downsample: int,
        timeout_s: float,
        outdir: Path,
        use_wsl: bool,
        stage_in_outdir: bool = True,
        use_cache: bool = False,
        max_objs: int | None = None,
        cpulimit_s: float | None = None,
    ):
        outdir.mkdir(parents=True, exist_ok=True)
        stem = fits_path.stem
        new_path = outdir / f"{stem}.new"
        solved_path = outdir / f"{stem}.solved"
        if use_cache and new_path.exists() and solved_path.exists():
            return True, 0.0, "cache_hit", "", [], new_path
        for p in outdir.glob(f"{stem}.*"):
            try:
                p.unlink()
            except Exception:
                pass
        staged_path = fits_path
        if stage_in_outdir:
            try:
                staged_path = outdir / fits_path.name
                if staged_path != fits_path:
                    shutil.copy2(fits_path, staged_path)
            except Exception:
                staged_path = fits_path

        cmd_str = str(getattr(self.params.P, "astnet_local_command", "solve-field"))
        cmd_base = shlex.split(cmd_str) if cmd_str.strip() else ["solve-field"]
        if use_wsl and cmd_base and cmd_base[0].lower() != "wsl":
            cmd = ["wsl"] + cmd_base
        else:
            cmd = cmd_base

        outdir_arg = self._win_to_wsl_path(outdir) if use_wsl else str(outdir)
        fits_arg = self._win_to_wsl_path(staged_path) if use_wsl else str(staged_path)

        cmd += [
            "--dir", outdir_arg,
            "--scale-units", "arcsecperpix",
            "--scale-low", f"{scale_low:.5f}",
            "--scale-high", f"{scale_high:.5f}",
            "--downsample", str(int(downsample)),
            "--no-verify",
            "--no-plots",
            "--overwrite",
            fits_arg,
        ]
        if max_objs is not None and int(max_objs) > 0:
            cmd += ["--objs", str(int(max_objs))]
        if cpulimit_s is not None and float(cpulimit_s) > 0:
            cmd += ["--cpulimit", f"{float(cpulimit_s):.1f}"]

        if center_coord is not None:
            cmd += [
                "--ra", f"{center_coord.ra.deg:.6f}",
                "--dec", f"{center_coord.dec.deg:.6f}",
                "--radius", f"{radius_deg:.3f}",
            ]

        try:
            start = time.time()
            cp = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
            dt = time.time() - start
            ok = (cp.returncode == 0 and new_path.exists() and solved_path.exists())
            if staged_path != fits_path:
                try:
                    staged_path.unlink()
                except Exception:
                    pass
            return ok, dt, cp.stdout, cp.stderr, cmd, new_path
        except subprocess.TimeoutExpired:
            return False, timeout_s, "", "timeout", cmd, None
        except Exception as e:
            return False, 0.0, "", str(e), cmd, None

    def _safe_header_update(self, fits_path: Path, new_hdr: fits.Header) -> None:
        for i in range(5):
            try:
                with fits.open(fits_path, mode="update", memmap=False) as hdul:
                    hdr = hdul[0].header
                    for key in new_hdr.keys():
                        if key.startswith(("CRVAL", "CRPIX", "CTYPE", "CUNIT", "CDELT",
                                           "CD1_", "CD2_", "PC1_", "PC2_", "CROTA",
                                           "PV", "LONPOLE", "LATPOLE", "RADESYS", "EQUINOX", "WCSAXES")):
                            try:
                                hdr[key] = new_hdr[key]
                            except Exception:
                                pass
                    hdr["WCS_OK"] = (True, "WCS solve by astrometry.net (local)")
                    hdr["WCSSRC"] = ("ASTNET_WSL", "WCS source")
                return
            except Exception:
                time.sleep(0.4 * (i + 1))
        raise RuntimeError("failed to update FITS header (locked)")

    def _get_file_path(self, filename: str) -> Path:
        """Get full path for a filename, considering cropped option"""
        if self.use_cropped:
            cropped_dir = step2_cropped_dir(self.result_dir)
            if not cropped_dir.exists():
                cropped_dir = self.result_dir / "cropped"
            return cropped_dir / filename
        return self.data_dir / filename

    def run(self):
        results = []
        total = len(self.file_list)

        # --- Load parameters ---
        pix_arc = float(getattr(self.params.P, "pixel_scale_arcsec", np.nan))
        if not np.isfinite(pix_arc) or pix_arc <= 0:
            self.error.emit("WORKER", "pixel_scale_arcsec is not set; run instrument setup first.")
            self.finished.emit({})
            return

        use_wsl = bool(getattr(self.params.P, "astnet_local_use_wsl", True))
        timeout_s = float(getattr(self.params.P, "astnet_local_timeout_s", 300.0))

        # Speed optimization for high-resolution images
        downsample = int(getattr(self.params.P, "astnet_local_downsample", 2))
        max_objs = int(getattr(self.params.P, "astnet_local_max_objs", 200))

        scale_low = float(getattr(self.params.P, "astnet_local_scale_low", 0.0))
        scale_high = float(getattr(self.params.P, "astnet_local_scale_high", 0.0))
        radius_deg = float(getattr(self.params.P, "astnet_local_radius_deg", 15.0))
        keep_outputs = bool(getattr(self.params.P, "astnet_local_keep_outputs", True))
        use_cache = bool(getattr(self.params.P, "astnet_local_use_cache", True))
        cpulimit_s = float(getattr(self.params.P, "astnet_local_cpulimit_s", 15.0))
        max_workers = get_parallel_workers(self.params)  # Central config

        if scale_low <= 0 or scale_high <= 0:
            scale_low = float(pix_arc) * 0.85
            scale_high = float(pix_arc) * 1.15

        outdir = self.cache_dir / "wcs_solve" / "astnet_local"

        # --- Parallel processing ---
        self.log_message.emit(f"Starting parallel plate solving with {max_workers} workers...")
        self.log_message.emit(f"  Scale: {scale_low:.4f} - {scale_high:.4f} arcsec/px")
        self.log_message.emit(f"  Downsample: {downsample}, Max objs: {max_objs}")

        def process_single_file(filename):
            if self._stop_requested:
                return filename, {"ok": False, "status": "stopped"}

            fits_path = self._get_file_path(filename)

            if not fits_path.exists():
                return filename, {"ok": False, "status": "file_not_found"}

            # Get center coordinates from header
            center_coord = None
            try:
                with fits.open(fits_path, memmap=False) as hdul:
                    hdr0 = hdul[0].header
                    ra0 = hdr0.get("OBJCTRA", None)
                    dec0 = hdr0.get("OBJCTDEC", None)
                    if ra0 is not None and dec0 is not None:
                        center_coord = SkyCoord(str(ra0), str(dec0), unit=(u.hourangle, u.deg))
            except Exception:
                center_coord = None

            if center_coord is None and self.target_coord is not None:
                center_coord = self.target_coord

            # Run solve-field
            ok, dt, out_s, err_s, cmd, new_path = self._run_solve_field(
                fits_path, center_coord, scale_low, scale_high, radius_deg,
                downsample, timeout_s, outdir, use_wsl, True, use_cache, max_objs, cpulimit_s
            )

            result = {
                "ok": False,
                "status": "fail",
                "ra": 0.0, "dec": 0.0, "pixscale": 0.0,
                "elapsed_s": float(dt),
            }

            if ok and new_path is not None and new_path.exists():
                try:
                    with fits.open(new_path, memmap=False) as hdul_new:
                        new_hdr = hdul_new[0].header
                        w = WCS(new_hdr, relax=True)
                        if w.has_celestial:
                            self._safe_header_update(fits_path, new_hdr)

                            pix_fit = float(np.mean(proj_plane_pixel_scales(w.celestial) * 3600.0))
                            cx = float(new_hdr.get("NAXIS1", 0)) / 2.0
                            cy = float(new_hdr.get("NAXIS2", 0)) / 2.0
                            ra_dec = w.pixel_to_world(cx, cy)

                            result = {
                                "ok": True,
                                "status": "solved",
                                "ra": float(ra_dec.ra.deg),
                                "dec": float(ra_dec.dec.deg),
                                "pixscale": pix_fit,
                                "elapsed_s": float(dt),
                                "wcs_header": dict(new_hdr),
                                "fits_path": str(fits_path),
                            }
                except Exception as e:
                    result = {"ok": False, "status": f"error: {e}", "elapsed_s": float(dt)}

            # Clean up temp files
            if not keep_outputs:
                for p in outdir.glob(f"{fits_path.stem}.*"):
                    try:
                        p.unlink()
                    except Exception:
                        pass

            return filename, result

        # --- Execute with ThreadPoolExecutor ---
        file_results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(process_single_file, f): f for f in self.file_list}

            completed_count = 0
            for future in as_completed(future_to_file):
                if self._stop_requested:
                    break

                fname = future_to_file[future]
                try:
                    filename, res = future.result()
                    res["filename"] = filename
                    results.append(res)
                    file_results[filename] = res

                    self.file_done.emit(filename, res)

                    if res.get("ok"):
                        ra_val = res.get('ra', 0)
                        dec_val = res.get('dec', 0)
                        self.log_message.emit(f"[OK] {filename} (RA={ra_val:.4f}, Dec={dec_val:.4f})")
                    else:
                        self.log_message.emit(f"[FAIL] {filename}: {res.get('status')}")

                except Exception as e:
                    self.error.emit(fname, str(e))

                completed_count += 1
                self.progress.emit(completed_count, total, f"Solved {completed_count}/{total}")

        # --- Gaia query and WCS Refine ---
        if self._stop_requested:
            self.finished.emit({"total": len(results), "ok": sum(1 for r in results if r.get("ok"))})
            return

        solved_frames = [r for r in results if r.get("ok")]
        center_coord = None
        if solved_frames:
            first_ok = solved_frames[0]
            ra_center = float(first_ok.get("ra", 0))
            dec_center = float(first_ok.get("dec", 0))
            if np.isfinite(ra_center) and np.isfinite(dec_center):
                center_coord = SkyCoord(ra=ra_center * u.deg, dec=dec_center * u.deg)

        gaia_df = None
        if center_coord is not None:
            self.log_message.emit("[Gaia] Querying Gaia catalog for ID matching...")
            self.progress.emit(total, total, "Querying Gaia catalog...")

            sample_fname = self.file_list[0]
            sample_path = self._get_file_path(sample_fname)
            try:
                with fits.open(sample_path, memmap=False) as hdul:
                    ny, nx = hdul[0].data.shape
                fov_w = (nx * pix_arc) / 3600.0
                fov_h = (ny * pix_arc) / 3600.0
                diag_deg = float(np.hypot(fov_w, fov_h))
                gaia_fudge = float(getattr(self.params.P, "gaia_radius_fudge", 1.35))
                gaia_r = float(0.5 * diag_deg * gaia_fudge)

                gaia_df, gaia_src = self._load_or_query_gaia(center_coord, gaia_r)
                self.log_message.emit(f"[Gaia] center=({center_coord.ra.deg:.6f},{center_coord.dec.deg:.6f}) r={gaia_r:.4f}deg source={gaia_src} N={len(gaia_df)}")
            except Exception as e:
                self.log_message.emit(f"[Gaia] Query error: {e}")
                gaia_df = pd.DataFrame()

        # WCS Refine
        if gaia_df is not None and len(gaia_df) > 0:
            self.log_message.emit("[Refine] Starting WCS refinement with Gaia matching...")
            refine_max_match = int(getattr(self.params.P, "wcs_refine_max_match", 300))

            for i, res in enumerate(solved_frames):
                if self._stop_requested:
                    break

                filename = res.get("filename", "")
                fits_path_str = res.get("fits_path", "")
                if not fits_path_str:
                    continue

                fits_path = Path(fits_path_str)
                if not fits_path.exists():
                    continue

                try:
                    det_xy, _ = self._load_detect_xy(filename)
                    fwhm_px, _ = self._load_fwhm_for_frame(filename)

                    with fits.open(fits_path, mode="update", memmap=False) as hdul:
                        hdr = hdul[0].header
                        w = WCS(hdr, relax=True)

                        ok_refine, refine_note, resid_med, resid_max, match_n = self._refine_crpix_by_match(
                            w, hdr, det_xy, gaia_df, fwhm_px, refine_max_match
                        )

                        res["refine"] = refine_note
                        res["resid_med"] = resid_med
                        res["resid_max"] = resid_max
                        res["match_n"] = match_n

                        if ok_refine:
                            hdr["WCSSRC"] = ("ASTNET_REFINED", "WCS source (refined with Gaia)")
                            self.log_message.emit(f"  [Refine] {filename}: {refine_note}, resid_med={resid_med:.2f}\"")
                        else:
                            self.log_message.emit(f"  [Refine] {filename}: skip - {refine_note}")

                    self.refine_done.emit(filename, {
                        "refine": res.get("refine", "-"),
                        "resid_med": res.get("resid_med", np.nan),
                        "resid_max": res.get("resid_max", np.nan),
                        "match_n": res.get("match_n", 0),
                    })

                except Exception as e:
                    self.log_message.emit(f"  [Refine] {filename}: error - {e}")
                    res["refine"] = f"error:{e}"
                    res["resid_med"] = np.nan
                    res["resid_max"] = np.nan
                    res["match_n"] = 0
                    self.refine_done.emit(filename, {
                        "refine": res["refine"],
                        "resid_med": res["resid_med"],
                        "resid_max": res["resid_max"],
                        "match_n": res["match_n"],
                    })

        summary = {
            "total": len(results),
            "ok": sum(1 for r in results if r.get("ok")),
        }
        self.finished.emit(summary)


class WcsPropagateWorker(QThread):
    """Worker thread to propagate WCS from solved frames to all frames with offset correction"""
    progress = pyqtSignal(int, int, str)
    file_done = pyqtSignal(str, dict)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str, str)

    def __init__(self, ref_frame, ref_wcs_dict, all_files, params, data_dir, result_dir, cache_dir, use_cropped=False):
        super().__init__()
        self.ref_frame = ref_frame
        self.ref_wcs_dict = ref_wcs_dict
        self.all_files = all_files
        self.params = params
        self.data_dir = Path(data_dir)
        self.result_dir = Path(result_dir)
        self.cache_dir = Path(cache_dir)
        self.use_cropped = use_cropped
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def _load_detections(self, filename):
        csv_path = self.cache_dir / f"detect_{filename}.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                return df[["x", "y"]].to_numpy(float)
            except Exception:
                pass
        return np.zeros((0, 2), float)

    def _compute_offset(self, ref_xy, tgt_xy, max_shift_px=50):
        if len(ref_xy) < 5 or len(tgt_xy) < 5:
            return 0.0, 0.0, 0

        tree = KDTree(ref_xy)
        d, j = tree.query(tgt_xy, k=1)
        good = (d < max_shift_px) & np.isfinite(d)
        if good.sum() < 5:
            return 0.0, 0.0, 0

        ref_matched = ref_xy[j[good]]
        tgt_matched = tgt_xy[good]
        dx = tgt_matched[:, 0] - ref_matched[:, 0]
        dy = tgt_matched[:, 1] - ref_matched[:, 1]
        dx_med = float(np.median(dx))
        dy_med = float(np.median(dy))
        return dx_med, dy_med, int(good.sum())

    def run(self):
        results = []
        total = len(self.all_files)
        cropped_dir = step2_cropped_dir(self.result_dir)
        if not cropped_dir.exists():
            cropped_dir = self.result_dir / "cropped"

        try:
            ref_hdr = fits.Header()
            for key, val in self.ref_wcs_dict.items():
                if key and key not in ('', 'SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'EXTEND'):
                    try:
                        ref_hdr[key] = val
                    except Exception:
                        pass

            ref_wcs = WCS(ref_hdr, relax=True)
            if not ref_wcs.has_celestial:
                self.error.emit("WORKER", "Reference WCS is not valid")
                self.finished.emit({"total": 0, "ok": 0})
                return

            ref_xy = self._load_detections(self.ref_frame)

            for idx, filename in enumerate(self.all_files):
                if self._stop_requested:
                    break

                self.progress.emit(idx, total, filename)

                if filename == self.ref_frame:
                    result = {"fname": filename, "ok": True, "status": "reference", "dx": 0, "dy": 0, "n_match": 0}
                    results.append(result)
                    self.file_done.emit(filename, result)
                    continue

                try:
                    if self.use_cropped:
                        fits_path = cropped_dir / filename
                    else:
                        fits_path = self.data_dir / filename

                    if not fits_path.exists():
                        result = {"fname": filename, "ok": False, "status": "file_not_found", "dx": 0, "dy": 0, "n_match": 0}
                        results.append(result)
                        self.file_done.emit(filename, result)
                        continue

                    tgt_xy = self._load_detections(filename)
                    dx, dy, n_match = self._compute_offset(ref_xy, tgt_xy)

                    with fits.open(fits_path, mode='update', memmap=False) as hdul:
                        hdr = hdul[0].header
                        for key, val in self.ref_wcs_dict.items():
                            if key and key not in ('', 'SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2', 'EXTEND'):
                                try:
                                    hdr[key] = val
                                except Exception:
                                    pass

                        if "CRPIX1" in hdr and "CRPIX2" in hdr:
                            hdr["CRPIX1"] = float(hdr["CRPIX1"]) - dx
                            hdr["CRPIX2"] = float(hdr["CRPIX2"]) - dy

                        hdr["WCS_OK"] = (True, "WCS propagated with offset")
                        hdr["WCSSRC"] = ("PROPAGATED", "WCS source")
                        hdr["WCSREFR"] = (self.ref_frame[:40], "WCS reference frame")
                        hdr["WCSDX"] = (dx, "X offset from ref (px)")
                        hdr["WCSDY"] = (dy, "Y offset from ref (px)")
                        hdr["WCSNMAT"] = (n_match, "N matched sources")
                        hdul.flush()

                    result = {"fname": filename, "ok": True, "status": "propagated", "dx": dx, "dy": dy, "n_match": n_match}
                    results.append(result)
                    self.file_done.emit(filename, result)

                except Exception as e:
                    self.error.emit(filename, str(e))
                    result = {"fname": filename, "ok": False, "status": "error", "dx": 0, "dy": 0, "n_match": 0}
                    results.append(result)
                    self.file_done.emit(filename, result)

            try:
                df = pd.DataFrame(results)
                output_dir = step5_dir(self.result_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                df.to_csv(output_dir / "wcs_propagate_summary.csv", index=False)
            except Exception:
                pass

        except Exception as e:
            self.error.emit("WORKER", str(e))

        summary = {
            "total": len(results),
            "ok": sum(1 for r in results if r.get("ok")),
        }
        self.finished.emit(summary)


class WcsPlateSolvingWindow(StepWindowBase):
    """Step 5: WCS Plate Solving"""

    def __init__(self, params, file_manager, project_state, main_window):
        self.file_manager = file_manager
        self.worker = None
        self.results = {}
        self.stop_requested = False
        self.log_window = None

        self.file_list = []
        self.use_cropped = False

        super().__init__(
            step_index=4,
            step_name="WCS Plate Solving",
            params=params,
            project_state=project_state,
            main_window=main_window
        )

        self.setup_step_ui()
        self.restore_state()

    def setup_step_ui(self):
        # Create tab widget
        self.tab_widget = QTabWidget()
        self.content_layout.addWidget(self.tab_widget)

        # ASTAP Tab
        self.astap_tab = QWidget()
        self.setup_astap_tab()
        self.tab_widget.addTab(self.astap_tab, "ASTAP (Local)")

        # Local Plate Solving Tab (solve-field via WSL)
        self.astrometrynet_tab = QWidget()
        self.setup_astrometrynet_tab()
        self.tab_widget.addTab(self.astrometrynet_tab, "Local Solve (WSL)")

        self.setup_log_window()
        self.populate_file_list()

    def setup_astap_tab(self):
        """Setup ASTAP tab UI"""
        layout = QVBoxLayout(self.astap_tab)

        info = QLabel(
            "Solve WCS using ASTAP (local). Best for star-rich fields like open clusters."
        )
        info.setStyleSheet("QLabel { background-color: #E3F2FD; padding: 10px; border-radius: 5px; }")
        layout.addWidget(info)

        control_layout = QHBoxLayout()
        btn_params = QPushButton("ASTAP Parameters")
        btn_params.setStyleSheet("QPushButton { background-color: #9C27B0; color: white; font-weight: bold; padding: 8px 15px; }")
        btn_params.clicked.connect(self.open_parameters_dialog)
        control_layout.addWidget(btn_params)

        control_layout.addStretch()

        self.btn_run = QPushButton("Run ASTAP")
        self.btn_run.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px 20px; }")
        self.btn_run.clicked.connect(self.run_wcs)
        control_layout.addWidget(self.btn_run)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; padding: 8px 15px; }")
        self.btn_stop.clicked.connect(self.stop_wcs)
        self.btn_stop.setEnabled(False)
        control_layout.addWidget(self.btn_stop)

        btn_log = QPushButton("Log")
        btn_log.setStyleSheet("QPushButton { background-color: #607D8B; color: white; font-weight: bold; padding: 8px 15px; }")
        btn_log.clicked.connect(self.show_log_window)
        control_layout.addWidget(btn_log)

        layout.addLayout(control_layout)

        progress_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)

        self.progress_label = QLabel("Ready")
        self.progress_label.setMinimumWidth(350)
        progress_layout.addWidget(self.progress_label)
        layout.addLayout(progress_layout)

        results_group = QGroupBox("WCS Results")
        results_layout = QVBoxLayout(results_group)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(6)
        self.results_table.setHorizontalHeaderLabels([
            "File", "Status", "PixScale Fit", "Refine", "Resid Med", "Elapsed (s)"
        ])
        self.results_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        results_layout.addWidget(self.results_table)

        layout.addWidget(results_group)

    def setup_astrometrynet_tab(self):
        """Setup Local Plate Solving tab UI (solve-field via WSL)"""
        layout = QVBoxLayout(self.astrometrynet_tab)

        # Info banner
        info_text = "Local plate solving using solve-field (via WSL). Parallel processing with Gaia catalog matching."
        info_style = "QLabel { background-color: #E3F2FD; padding: 10px; border-radius: 5px; }"
        info = QLabel(info_text)
        info.setStyleSheet(info_style)
        layout.addWidget(info)

        # Settings group
        settings_group = QGroupBox("Solve Settings")
        settings_layout = QHBoxLayout(settings_group)

        settings_layout.addWidget(QLabel("Downsample:"))
        self.spin_downsample = QSpinBox()
        self.spin_downsample.setRange(1, 8)
        self.spin_downsample.setValue(int(getattr(self.params.P, "astnet_local_downsample", 2)))
        self.spin_downsample.setToolTip("Image downsampling factor for faster solving")
        settings_layout.addWidget(self.spin_downsample)

        settings_layout.addWidget(QLabel("Timeout (s):"))
        self.spin_timeout = QSpinBox()
        self.spin_timeout.setRange(30, 600)
        self.spin_timeout.setValue(int(getattr(self.params.P, "astnet_local_timeout_s", 300)))
        self.spin_timeout.setToolTip("Timeout per frame in seconds")
        settings_layout.addWidget(self.spin_timeout)

        settings_layout.addStretch()
        layout.addWidget(settings_group)

        # Hidden API key field for backward compatibility
        self.astrometrynet_api_key = QLineEdit()
        self.astrometrynet_api_key.setVisible(False)

        # Reference frame selection
        ref_group = QGroupBox("Frame List")
        ref_layout = QVBoxLayout(ref_group)

        ref_info = QLabel("Frames listed below will be solved automatically (no selection required).")
        ref_info.setWordWrap(True)
        ref_layout.addWidget(ref_info)

        self.ref_frame_list = QListWidget()
        self.ref_frame_list.setSelectionMode(QListWidget.NoSelection)
        self.ref_frame_list.setMaximumHeight(150)
        ref_layout.addWidget(self.ref_frame_list)

        layout.addWidget(ref_group)

        # Control buttons
        control_layout = QHBoxLayout()

        btn_astnet_params = QPushButton("Astrometry.net Parameters")
        btn_astnet_params.setStyleSheet(
            "QPushButton { background-color: #9C27B0; color: white; font-weight: bold; padding: 8px 15px; }"
        )
        btn_astnet_params.clicked.connect(self.open_astrometrynet_parameters_dialog)
        control_layout.addWidget(btn_astnet_params)

        control_layout.addStretch()

        self.btn_solve_astrometrynet = QPushButton("Solve All Frames")
        self.btn_solve_astrometrynet.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px 20px; }")
        self.btn_solve_astrometrynet.clicked.connect(self.run_astrometrynet_solve)
        self.btn_solve_astrometrynet.setEnabled(True)  # Always enabled for local solving
        control_layout.addWidget(self.btn_solve_astrometrynet)

        self.btn_propagate_wcs = QPushButton("Propagate WCS to All")
        self.btn_propagate_wcs.setStyleSheet("QPushButton { background-color: #FF9800; color: white; font-weight: bold; padding: 8px 20px; }")
        self.btn_propagate_wcs.clicked.connect(self.propagate_wcs_to_all)
        self.btn_propagate_wcs.setEnabled(False)
        control_layout.addWidget(self.btn_propagate_wcs)

        btn_log2 = QPushButton("Log")
        btn_log2.setStyleSheet("QPushButton { background-color: #607D8B; color: white; font-weight: bold; padding: 8px 15px; }")
        btn_log2.clicked.connect(self.show_log_window)
        control_layout.addWidget(btn_log2)

        layout.addLayout(control_layout)

        # Progress bar
        progress_layout = QHBoxLayout()
        self.astrometrynet_progress = QProgressBar()
        self.astrometrynet_progress.setMinimum(0)
        self.astrometrynet_progress.setMaximum(100)
        self.astrometrynet_progress.setValue(0)
        progress_layout.addWidget(self.astrometrynet_progress)

        self.astrometrynet_status = QLabel("Ready")
        self.astrometrynet_status.setMinimumWidth(350)
        progress_layout.addWidget(self.astrometrynet_status)
        layout.addLayout(progress_layout)

        # Results table
        results_group = QGroupBox("Plate Solving Results")
        results_layout = QVBoxLayout(results_group)

        self.astrometrynet_results_table = QTableWidget()
        self.astrometrynet_results_table.setColumnCount(7)
        self.astrometrynet_results_table.setHorizontalHeaderLabels([
            "File", "Status", "Refine", "Resid Med", "Resid Max", "RA", "Dec"
        ])
        self.astrometrynet_results_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.astrometrynet_results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        results_layout.addWidget(self.astrometrynet_results_table)

        layout.addWidget(results_group)

    def auto_select_ref_frame(self):
        best_frame = None
        best_count = 0

        for fname in self.file_list:
            detect_csv = self.params.P.cache_dir / f"detect_{fname}.csv"
            if detect_csv.exists():
                try:
                    df = pd.read_csv(detect_csv)
                    if len(df) > best_count:
                        best_count = len(df)
                        best_frame = fname
                except Exception:
                    pass

        if best_frame:
            for i in range(self.ref_frame_list.count()):
                item = self.ref_frame_list.item(i)
                if item.text() == best_frame:
                    item.setSelected(True)
                    self.log(f"Auto-selected: {best_frame} ({best_count} stars)")
                    break
        else:
            QMessageBox.warning(self, "Warning", "No detection data found. Run Step 4 first.")

    def run_astrometrynet_solve(self):
        """Run local solve-field (via WSL) with parallel processing and Gaia matching"""
        if not self.file_list:
            QMessageBox.warning(self, "Warning", "No frames found to solve")
            return
        file_list = list(self.file_list)

        # Save GUI settings to params
        self.params.P.astnet_local_downsample = self.spin_downsample.value()
        self.params.P.astnet_local_timeout_s = self.spin_timeout.value()
        self.persist_params()

        # Get target coordinate if available
        target_coord = None
        try:
            ra_str = getattr(self.params.P, "target_ra", "")
            dec_str = getattr(self.params.P, "target_dec", "")
            if ra_str and dec_str:
                target_coord = SkyCoord(str(ra_str), str(dec_str), unit=(u.hourangle, u.deg))
        except Exception:
            pass

        # Start local solving in thread
        self.astrometrynet_worker = AstrometryNetWorker(
            file_list=file_list,
            params=self.params,
            data_dir=self.params.P.data_dir,
            result_dir=self.params.P.result_dir,
            cache_dir=self.params.P.cache_dir,
            use_cropped=self.use_cropped,
            target_coord=target_coord,
        )
        self.astrometrynet_worker.progress.connect(self.on_astrometrynet_progress)
        self.astrometrynet_worker.file_done.connect(self.on_astrometrynet_file_done)
        self.astrometrynet_worker.refine_done.connect(self.on_astrometrynet_refine_done)
        self.astrometrynet_worker.log_message.connect(self.log)
        self.astrometrynet_worker.finished.connect(self.on_astrometrynet_finished)
        self.astrometrynet_worker.error.connect(self.on_astrometrynet_error)

        self.btn_solve_astrometrynet.setEnabled(False)
        self.astrometrynet_progress.setValue(0)
        self.astrometrynet_status.setText("Starting local plate solving...")
        self.log("=" * 50)
        self.log("Starting local plate solving (solve-field via WSL)...")
        self.log(f"Workers: {get_parallel_workers(self.params)}, Downsample: {self.spin_downsample.value()}")
        self.log(f"Frames: {len(file_list)}")
        self.astrometrynet_worker.start()
        self.show_log_window()

    def on_astrometrynet_progress(self, current, total, status):
        pct = int(100 * current / max(1, total))
        self.astrometrynet_progress.setValue(pct)
        self.astrometrynet_status.setText(status)

    def on_astrometrynet_file_done(self, filename, result):
        row = self.astrometrynet_results_table.rowCount()
        self.astrometrynet_results_table.insertRow(row)
        self.astrometrynet_results_table.setItem(row, 0, QTableWidgetItem(filename))
        self.astrometrynet_results_table.setItem(row, 1, QTableWidgetItem(result.get("status", "")))
        self.astrometrynet_results_table.setItem(row, 2, QTableWidgetItem("-"))
        self.astrometrynet_results_table.setItem(row, 3, QTableWidgetItem("-"))
        self.astrometrynet_results_table.setItem(row, 4, QTableWidgetItem("-"))
        self.astrometrynet_results_table.setItem(row, 5, QTableWidgetItem(f"{result.get('ra', 0):.6f}"))
        self.astrometrynet_results_table.setItem(row, 6, QTableWidgetItem(f"{result.get('dec', 0):.6f}"))

        if result.get("ok"):
            self.results[filename] = result
            self.log(f"Solved: {filename} (RA={result.get('ra', 0):.4f}, Dec={result.get('dec', 0):.4f})")

    def on_astrometrynet_error(self, filename, error):
        self.log(f"ERROR {filename}: {error}")

    def on_astrometrynet_refine_done(self, filename, result):
        """Update table with refinement results (residuals)"""
        # Find row in table and update refine columns
        for row in range(self.astrometrynet_results_table.rowCount()):
            item = self.astrometrynet_results_table.item(row, 0)
            if item and item.text() == filename:
                refine_note = result.get("refine", "-")
                resid_med = result.get("resid_med", float("nan"))
                resid_max = result.get("resid_max", float("nan"))
                match_n = result.get("match_n", 0)

                self.astrometrynet_results_table.setItem(row, 2, QTableWidgetItem(refine_note))
                if np.isfinite(resid_med):
                    self.astrometrynet_results_table.setItem(row, 3, QTableWidgetItem(f"{resid_med:.2f}\""))
                if np.isfinite(resid_max):
                    self.astrometrynet_results_table.setItem(row, 4, QTableWidgetItem(f"{resid_max:.2f}\""))
                break

    def on_astrometrynet_finished(self, summary):
        self.btn_solve_astrometrynet.setEnabled(True)
        n_ok = summary.get("ok", 0)
        self.astrometrynet_progress.setValue(100)
        self.astrometrynet_status.setText(f"Done: {n_ok}/{summary.get('total', 0)} solved")
        if n_ok > 0:
            self.btn_propagate_wcs.setEnabled(True)
            self.log(f"Local solve: {n_ok} frames solved successfully")
        self.save_state()
        self.update_navigation_buttons()

    def propagate_wcs_to_all(self):
        solved_frame = None
        solved_wcs = None
        for fname, res in self.results.items():
            if res.get("ok") and res.get("wcs_header"):
                solved_frame = fname
                solved_wcs = res.get("wcs_header")
                break

        if not solved_frame or not solved_wcs:
            QMessageBox.warning(self, "Error", "No solved WCS available")
            return

        self.log(f"Propagating WCS from {solved_frame} to all frames...")
        self.astrometrynet_status.setText("Propagating WCS...")

        # Clear previous propagation results
        rows_to_remove = []
        for row in range(self.astrometrynet_results_table.rowCount()):
            status_item = self.astrometrynet_results_table.item(row, 1)
            if status_item and status_item.text() not in ("solved",):
                rows_to_remove.append(row)
        for row in reversed(rows_to_remove):
            self.astrometrynet_results_table.removeRow(row)

        self.propagate_worker = WcsPropagateWorker(
            solved_frame,
            solved_wcs,
            self.file_list,
            self.params,
            self.params.P.data_dir,
            self.params.P.result_dir,
            self.params.P.cache_dir,
            self.use_cropped
        )
        self.propagate_worker.progress.connect(self.on_propagate_progress)
        self.propagate_worker.file_done.connect(self.on_propagate_file_done)
        self.propagate_worker.finished.connect(self.on_propagate_finished)
        self.propagate_worker.error.connect(lambda f, e: self.log(f"Propagate error {f}: {e}"))

        self.btn_propagate_wcs.setEnabled(False)
        self.astrometrynet_progress.setValue(0)
        self.propagate_worker.start()

    def on_propagate_progress(self, current, total, fname):
        pct = int(100 * current / max(1, total))
        self.astrometrynet_progress.setValue(pct)
        self.astrometrynet_status.setText(f"Propagating: {fname} ({current}/{total})")

    def on_propagate_file_done(self, filename, result):
        row = self.astrometrynet_results_table.rowCount()
        self.astrometrynet_results_table.insertRow(row)
        self.astrometrynet_results_table.setItem(row, 0, QTableWidgetItem(filename))
        self.astrometrynet_results_table.setItem(row, 1, QTableWidgetItem(result.get("status", "")))
        dx = result.get("dx", 0)
        dy = result.get("dy", 0)
        n_match = result.get("n_match", 0)
        self.astrometrynet_results_table.setItem(row, 2, QTableWidgetItem(f"{dx:+.2f}" if dx != 0 else "0.00"))
        self.astrometrynet_results_table.setItem(row, 3, QTableWidgetItem(f"{dy:+.2f}" if dy != 0 else "0.00"))
        self.astrometrynet_results_table.setItem(row, 4, QTableWidgetItem(str(n_match) if n_match > 0 else "-"))
        self.astrometrynet_results_table.setItem(row, 5, QTableWidgetItem("-"))
        self.astrometrynet_results_table.setItem(row, 6, QTableWidgetItem("-"))
        if result.get("status") == "reference":
            for col in range(7):
                item = self.astrometrynet_results_table.item(row, col)
                if item:
                    item.setBackground(Qt.green)
        elif not result.get("ok"):
            for col in range(7):
                item = self.astrometrynet_results_table.item(row, col)
                if item:
                    item.setBackground(Qt.red)
        if result.get("ok"):
            self.results[filename] = result

    def on_propagate_finished(self, summary):
        self.btn_propagate_wcs.setEnabled(True)
        n_ok = summary.get("ok", 0)
        total = summary.get("total", 0)
        self.astrometrynet_progress.setValue(100)
        self.astrometrynet_status.setText(f"Done: {n_ok}/{total} frames updated")
        self.log(f"WCS propagation complete: {n_ok}/{total} frames updated")
        self.save_state()
        self.update_navigation_buttons()

    def _get_credentials_path(self):
        step_dir = step5_dir(self.params.P.result_dir)
        step_dir.mkdir(parents=True, exist_ok=True)
        step_path = step_dir / ".credentials.json"
        legacy_path = self.params.P.result_dir / ".credentials.json"
        if legacy_path.exists() and not step_path.exists():
            return legacy_path
        return step_path

    def _load_api_key(self):
        cred_path = self._get_credentials_path()
        if cred_path.exists():
            try:
                creds = json.loads(cred_path.read_text(encoding="utf-8"))
                return creds.get("astrometry_net_api_key", "")
            except Exception:
                pass
        return ""

    def _save_api_key(self, api_key: str):
        cred_path = self._get_credentials_path()
        try:
            creds = {}
            if cred_path.exists():
                try:
                    creds = json.loads(cred_path.read_text(encoding="utf-8"))
                except Exception:
                    pass
            creds["astrometry_net_api_key"] = api_key
            cred_path.write_text(json.dumps(creds, indent=2), encoding="utf-8")
        except Exception as e:
            self.log(f"Failed to save API key: {e}")

    def _toggle_api_key_visibility(self):
        if self.astrometrynet_api_key.echoMode() == QLineEdit.Password:
            self.astrometrynet_api_key.setEchoMode(QLineEdit.Normal)
            self.btn_show_key.setText("Hide")
        else:
            self.astrometrynet_api_key.setEchoMode(QLineEdit.Password)
            self.btn_show_key.setText("Show")

    def setup_log_window(self):
        if self.log_window is not None:
            return
        self.log_window = QWidget(self, Qt.Window)
        self.log_window.setWindowTitle("WCS Log")
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

        self.file_list = list(files)

        # Also populate ref_frame_list for Astrometry.net tab
        self.ref_frame_list.clear()
        for fname in self.file_list:
            item = QListWidgetItem(fname)
            # Add star count info if available
            detect_csv = self.params.P.cache_dir / f"detect_{fname}.csv"
            if detect_csv.exists():
                try:
                    df = pd.read_csv(detect_csv)
                    item.setToolTip(f"{len(df)} sources detected")
                except Exception:
                    pass
            self.ref_frame_list.addItem(item)

    def open_parameters_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("WCS Parameters")
        dialog.resize(500, 420)

        layout = QVBoxLayout(dialog)
        form = QFormLayout()

        self.param_astap_exe = QLineEdit(str(getattr(self.params.P, "astap_exe", "astap_cli.exe")))
        form.addRow("ASTAP CLI Path:", self.param_astap_exe)

        self.param_timeout = QDoubleSpinBox()
        self.param_timeout.setRange(10, 1000)
        self.param_timeout.setValue(float(getattr(self.params.P, "astap_timeout_s", 120.0)))
        form.addRow("Timeout (s):", self.param_timeout)

        self.param_radius = QDoubleSpinBox()
        self.param_radius.setRange(0.5, 30.0)
        self.param_radius.setValue(float(getattr(self.params.P, "astap_search_radius_deg", 8.0)))
        form.addRow("Search Radius (deg):", self.param_radius)

        self.param_astap_db = QComboBox()
        self.param_astap_db.addItems(["D50", "D80"])
        current_db = str(getattr(self.params.P, "astap_database", "D50"))
        idx = self.param_astap_db.findText(current_db)
        if idx >= 0:
            self.param_astap_db.setCurrentIndex(idx)
        form.addRow("ASTAP Star DB:", self.param_astap_db)

        self.param_annotate_variables = QCheckBox("Enable")
        self.param_annotate_variables.setChecked(bool(getattr(self.params.P, "astap_annotate_variables", False)))
        self.param_annotate_variables.setToolTip("ASTAP 변광성 데이터베이스로 변광성 주석 표시 (별도 설치 필요)")
        form.addRow("Annotate Variable Stars:", self.param_annotate_variables)

        self.param_fov_fudge = QDoubleSpinBox()
        self.param_fov_fudge.setRange(0.5, 2.0)
        self.param_fov_fudge.setSingleStep(0.05)
        self.param_fov_fudge.setValue(float(getattr(self.params.P, "astap_fov_fudge", 1.0)))
        form.addRow("FOV Fudge:", self.param_fov_fudge)

        self.param_downsample = QSpinBox()
        self.param_downsample.setRange(1, 8)
        self.param_downsample.setValue(int(getattr(self.params.P, "astap_downsample_z", 2)))
        form.addRow("Downsample Z:", self.param_downsample)

        self.param_max_stars = QSpinBox()
        self.param_max_stars.setRange(50, 5000)
        self.param_max_stars.setValue(int(getattr(self.params.P, "astap_max_stars_s", 500)))
        form.addRow("Max Stars (S):", self.param_max_stars)

        self.param_require_qc = QCheckBox("Enable")
        self.param_require_qc.setChecked(bool(getattr(self.params.P, "wcs_require_qc_pass", True)))
        form.addRow("QC Pass Only:", self.param_require_qc)

        self.param_refine_enable = QCheckBox("Enable")
        self.param_refine_enable.setChecked(bool(getattr(self.params.P, "wcs_refine_enable", True)))
        form.addRow("Refine CRPIX:", self.param_refine_enable)

        self.param_refine_max_match = QSpinBox()
        self.param_refine_max_match.setRange(50, 5000)
        self.param_refine_max_match.setValue(int(getattr(self.params.P, "wcs_refine_max_match", 600)))
        form.addRow("Refine Max Match:", self.param_refine_max_match)

        self.param_refine_match_r = QDoubleSpinBox()
        self.param_refine_match_r.setRange(0.5, 5.0)
        self.param_refine_match_r.setSingleStep(0.1)
        self.param_refine_match_r.setValue(float(getattr(self.params.P, "wcs_refine_match_r_fwhm", 1.6)))
        form.addRow("Refine Match R (×FWHM):", self.param_refine_match_r)

        self.param_refine_min_match = QSpinBox()
        self.param_refine_min_match.setRange(5, 500)
        self.param_refine_min_match.setValue(int(getattr(self.params.P, "wcs_refine_min_match", 50)))
        form.addRow("Refine Min Match:", self.param_refine_min_match)

        self.param_gaia_fudge = QDoubleSpinBox()
        self.param_gaia_fudge.setRange(0.5, 3.0)
        self.param_gaia_fudge.setSingleStep(0.05)
        self.param_gaia_fudge.setValue(float(getattr(self.params.P, "gaia_radius_fudge", 1.35)))
        form.addRow("Gaia Radius Fudge:", self.param_gaia_fudge)

        self.param_gaia_mag_max = QDoubleSpinBox()
        self.param_gaia_mag_max.setRange(10.0, 22.0)
        self.param_gaia_mag_max.setSingleStep(0.5)
        self.param_gaia_mag_max.setValue(float(getattr(self.params.P, "gaia_mag_max", 18.0)))
        form.addRow("Gaia Mag Max:", self.param_gaia_mag_max)

        self.param_gaia_retry = QSpinBox()
        self.param_gaia_retry.setRange(0, 10)
        self.param_gaia_retry.setValue(int(getattr(self.params.P, "gaia_retry", 2)))
        form.addRow("Gaia Retry:", self.param_gaia_retry)

        self.param_gaia_backoff = QDoubleSpinBox()
        self.param_gaia_backoff.setRange(0.0, 30.0)
        self.param_gaia_backoff.setSingleStep(1.0)
        self.param_gaia_backoff.setValue(float(getattr(self.params.P, "gaia_backoff_s", 6.0)))
        form.addRow("Gaia Backoff (s):", self.param_gaia_backoff)

        self.param_gaia_allow_no_cache = QCheckBox("Allow")
        self.param_gaia_allow_no_cache.setChecked(bool(getattr(self.params.P, "gaia_allow_no_cache", True)))
        form.addRow("Gaia Allow No Cache:", self.param_gaia_allow_no_cache)

        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(lambda: self.save_parameters(dialog))
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec_()

    def open_astrometrynet_parameters_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Astrometry.net Parameters")
        dialog.resize(520, 520)

        layout = QVBoxLayout(dialog)
        form = QFormLayout()

        self.param_astnet_enable = QCheckBox("Enable")
        self.param_astnet_enable.setChecked(bool(getattr(self.params.P, "astnet_local_enable", False)))
        form.addRow("Enable Local Solve:", self.param_astnet_enable)

        self.param_astnet_use_wsl = QCheckBox("Use WSL")
        self.param_astnet_use_wsl.setChecked(bool(getattr(self.params.P, "astnet_local_use_wsl", True)))
        form.addRow("Use WSL:", self.param_astnet_use_wsl)

        self.param_astnet_command = QLineEdit(str(getattr(self.params.P, "astnet_local_command", "solve-field")))
        form.addRow("solve-field Command:", self.param_astnet_command)

        self.param_astnet_timeout = QDoubleSpinBox()
        self.param_astnet_timeout.setRange(30, 3600)
        self.param_astnet_timeout.setValue(float(getattr(self.params.P, "astnet_local_timeout_s", 300.0)))
        form.addRow("Timeout (s):", self.param_astnet_timeout)

        self.param_astnet_downsample = QSpinBox()
        self.param_astnet_downsample.setRange(1, 8)
        self.param_astnet_downsample.setValue(int(getattr(self.params.P, "astnet_local_downsample", 2)))
        form.addRow("Downsample:", self.param_astnet_downsample)

        self.param_astnet_scale_low = QDoubleSpinBox()
        self.param_astnet_scale_low.setRange(0.0, 10.0)
        self.param_astnet_scale_low.setDecimals(5)
        self.param_astnet_scale_low.setValue(float(getattr(self.params.P, "astnet_local_scale_low", 0.0)))
        form.addRow("Scale Low (arcsec/pix):", self.param_astnet_scale_low)

        self.param_astnet_scale_high = QDoubleSpinBox()
        self.param_astnet_scale_high.setRange(0.0, 10.0)
        self.param_astnet_scale_high.setDecimals(5)
        self.param_astnet_scale_high.setValue(float(getattr(self.params.P, "astnet_local_scale_high", 0.0)))
        form.addRow("Scale High (arcsec/pix):", self.param_astnet_scale_high)

        self.param_astnet_radius = QDoubleSpinBox()
        self.param_astnet_radius.setRange(0.1, 30.0)
        self.param_astnet_radius.setValue(float(getattr(self.params.P, "astnet_local_radius_deg", 8.0)))
        form.addRow("Radius (deg):", self.param_astnet_radius)

        self.param_astnet_keep_outputs = QCheckBox("Keep")
        self.param_astnet_keep_outputs.setChecked(bool(getattr(self.params.P, "astnet_local_keep_outputs", True)))
        form.addRow("Keep Outputs:", self.param_astnet_keep_outputs)

        self.param_astnet_use_cache = QCheckBox("Use Cache")
        self.param_astnet_use_cache.setChecked(bool(getattr(self.params.P, "astnet_local_use_cache", True)))
        form.addRow("Use Cached Outputs:", self.param_astnet_use_cache)

        self.param_astnet_max_objs = QSpinBox()
        self.param_astnet_max_objs.setRange(100, 20000)
        self.param_astnet_max_objs.setValue(int(getattr(self.params.P, "astnet_local_max_objs", 2000)))
        form.addRow("Max Objects:", self.param_astnet_max_objs)

        self.param_astnet_cpulimit = QDoubleSpinBox()
        self.param_astnet_cpulimit.setRange(1, 300)
        self.param_astnet_cpulimit.setValue(float(getattr(self.params.P, "astnet_local_cpulimit_s", 30.0)))
        form.addRow("CPU Limit (s):", self.param_astnet_cpulimit)

        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(lambda: self.save_astrometrynet_parameters(dialog))
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec_()

    def save_parameters(self, dialog):
        self.params.P.astap_exe = self.param_astap_exe.text().strip()
        self.params.P.astap_timeout_s = self.param_timeout.value()
        self.params.P.astap_search_radius_deg = self.param_radius.value()
        self.params.P.astap_database = self.param_astap_db.currentText()
        self.params.P.astap_annotate_variables = self.param_annotate_variables.isChecked()
        self.params.P.astap_fov_fudge = self.param_fov_fudge.value()
        self.params.P.astap_downsample_z = self.param_downsample.value()
        self.params.P.astap_max_stars_s = self.param_max_stars.value()
        self.params.P.wcs_require_qc_pass = self.param_require_qc.isChecked()
        self.params.P.wcs_refine_enable = self.param_refine_enable.isChecked()
        self.params.P.wcs_refine_max_match = self.param_refine_max_match.value()
        self.params.P.wcs_refine_match_r_fwhm = self.param_refine_match_r.value()
        self.params.P.wcs_refine_min_match = self.param_refine_min_match.value()
        self.params.P.gaia_radius_fudge = self.param_gaia_fudge.value()
        self.params.P.gaia_mag_max = self.param_gaia_mag_max.value()
        self.params.P.gaia_retry = self.param_gaia_retry.value()
        self.params.P.gaia_backoff_s = self.param_gaia_backoff.value()
        self.params.P.gaia_allow_no_cache = self.param_gaia_allow_no_cache.isChecked()
        self.persist_params()
        self.save_state()
        QMessageBox.information(dialog, "Success", "Parameters saved!")
        dialog.accept()

    def save_astrometrynet_parameters(self, dialog):
        self.params.P.astnet_local_enable = self.param_astnet_enable.isChecked()
        self.params.P.astnet_local_use_wsl = self.param_astnet_use_wsl.isChecked()
        self.params.P.astnet_local_command = self.param_astnet_command.text().strip()
        self.params.P.astnet_local_timeout_s = self.param_astnet_timeout.value()
        self.params.P.astnet_local_downsample = self.param_astnet_downsample.value()
        self.params.P.astnet_local_scale_low = self.param_astnet_scale_low.value()
        self.params.P.astnet_local_scale_high = self.param_astnet_scale_high.value()
        self.params.P.astnet_local_radius_deg = self.param_astnet_radius.value()
        self.params.P.astnet_local_keep_outputs = self.param_astnet_keep_outputs.isChecked()
        self.params.P.astnet_local_use_cache = self.param_astnet_use_cache.isChecked()
        self.params.P.astnet_local_max_objs = self.param_astnet_max_objs.value()
        self.params.P.astnet_local_cpulimit_s = self.param_astnet_cpulimit.value()
        self.persist_params()
        self.save_state()
        QMessageBox.information(dialog, "Success", "Astrometry.net parameters saved!")
        dialog.accept()

    def run_wcs(self):
        if not self.file_list:
            QMessageBox.warning(self, "Warning", "No files to process")
            return
        if self.worker and self.worker.isRunning():
            return

        self.results = {}
        self.results_table.setRowCount(0)
        self.log_text.clear()
        self.stop_requested = False

        # Get target from params.P (loaded from TOML)
        target_coord = None
        ra = getattr(self.params.P, "target_ra_deg", None)
        dec = getattr(self.params.P, "target_dec_deg", None)
        if ra is not None and dec is not None:
            try:
                target_coord = SkyCoord(float(ra) * u.deg, float(dec) * u.deg)
            except Exception:
                target_coord = None

        self.worker = WcsWorker(
            self.file_list,
            self.params,
            self.params.P.data_dir,
            self.params.P.result_dir,
            self.params.P.cache_dir,
            self.use_cropped,
            target_coord=target_coord
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

    def stop_wcs(self):
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
        self.results_table.setItem(row, 1, QTableWidgetItem(str(result.get("status", ""))))
        pix_fit = result.get("pix_fit")
        pix_str = f"{pix_fit:.4f}" if isinstance(pix_fit, float) and np.isfinite(pix_fit) else "-"
        self.results_table.setItem(row, 2, QTableWidgetItem(pix_str))
        refine = result.get("refine", "")
        self.results_table.setItem(row, 3, QTableWidgetItem(str(refine)))
        resid_med = result.get("resid_med")
        resid_str = f"{resid_med:.3f}" if isinstance(resid_med, float) and np.isfinite(resid_med) else "-"
        self.results_table.setItem(row, 4, QTableWidgetItem(resid_str))
        elapsed = result.get("elapsed", 0.0)
        self.results_table.setItem(row, 5, QTableWidgetItem(f"{elapsed:.1f}"))
        pix_fit_log = result.get("pix_fit")
        pix_log = f"{pix_fit_log:.4f}" if isinstance(pix_fit_log, float) and np.isfinite(pix_fit_log) else "-"
        resid_log = result.get("resid_med")
        resid_str = f"{resid_log:.3f}" if isinstance(resid_log, float) and np.isfinite(resid_log) else "-"
        self.log(f"{filename}: {result.get('status', '')} pix={pix_log} refine={refine or '-'} resid_med={resid_str}")

    def on_error(self, filename, error):
        self.log(f"ERROR {filename}: {error}")

    def on_finished(self, summary):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_label.setText("Done")
        if summary:
            self.log(f"WCS done: {summary.get('ok', 0)}/{summary.get('total', 0)} OK")
        self.save_state()
        self.update_navigation_buttons()

    def validate_step(self) -> bool:
        return len(self.results) > 0

    def save_state(self):
        state_data = {
            "wcs_complete": len(self.results) > 0,
            "n_files": len(self.results),
            "use_cropped": self.use_cropped,
            "astap_exe": getattr(self.params.P, "astap_exe", "astap_cli.exe"),
            "astap_timeout_s": getattr(self.params.P, "astap_timeout_s", 120.0),
            "astap_search_radius_deg": getattr(self.params.P, "astap_search_radius_deg", 8.0),
            "astap_database": getattr(self.params.P, "astap_database", "D50"),
            "astap_annotate_variables": getattr(self.params.P, "astap_annotate_variables", False),
            "astap_fov_fudge": getattr(self.params.P, "astap_fov_fudge", 1.0),
            "astap_downsample_z": getattr(self.params.P, "astap_downsample_z", 2),
            "astap_max_stars_s": getattr(self.params.P, "astap_max_stars_s", 500),
            "wcs_require_qc_pass": getattr(self.params.P, "wcs_require_qc_pass", True),
            "wcs_refine_enable": getattr(self.params.P, "wcs_refine_enable", True),
            "wcs_refine_max_match": getattr(self.params.P, "wcs_refine_max_match", 600),
            "wcs_refine_match_r_fwhm": getattr(self.params.P, "wcs_refine_match_r_fwhm", 1.6),
            "wcs_refine_min_match": getattr(self.params.P, "wcs_refine_min_match", 50),
            "gaia_radius_fudge": getattr(self.params.P, "gaia_radius_fudge", 1.35),
            "gaia_mag_max": getattr(self.params.P, "gaia_mag_max", 18.0),
            "gaia_retry": getattr(self.params.P, "gaia_retry", 2),
            "gaia_backoff_s": getattr(self.params.P, "gaia_backoff_s", 6.0),
            "gaia_allow_no_cache": getattr(self.params.P, "gaia_allow_no_cache", True),
        }
        self.project_state.store_step_data("wcs_plate_solve", state_data)

    def persist_params(self):
        """Save current parameters to TOML file."""
        try:
            if hasattr(self.params, 'save_toml'):
                self.params.save_toml()
        except Exception as e:
            print(f"[WcsPlateSolvingWindow] Failed to persist params: {e}")

    def restore_state(self):
        state_data = self.project_state.get_step_data("wcs_plate_solve")
        if state_data:
            for key, val in state_data.items():
                if hasattr(self.params.P, key):
                    setattr(self.params.P, key, val)
