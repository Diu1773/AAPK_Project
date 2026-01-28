"""
Step 5: WCS Plate Solving (ASTAP)
Minimal GUI wrapper for ASTAP-based WCS solving with cache/summary output.
"""

from __future__ import annotations

import json
import time
import subprocess
import threading
import warnings
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

from PyQt5.QtCore import Qt, QThread, pyqtSignal

from .step_window_base import StepWindowBase
from ...utils.step_paths import (
    step2_cropped_dir,
    crop_is_active,
    step4_dir,
    step5_dir,
    step6_dir,
    legacy_step5_refbuild_dir,
    legacy_step7_wcs_dir,
    legacy_step7_refbuild_dir,
)
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

    def _copy_wcs_keywords(self, src_hdr: fits.Header, dst_hdr: fits.Header):
        prefixes = (
            "CRVAL", "CRPIX", "CTYPE", "CUNIT", "CDELT",
            "CD1_", "CD2_", "PC1_", "PC2_", "CROTA",
            "PV", "LONPOLE", "LATPOLE", "RADESYS", "EQUINOX", "WCSAXES",
        )
        for key in src_hdr.keys():
            if key.startswith(prefixes):
                try:
                    dst_hdr[key] = src_hdr[key]
                except Exception:
                    pass

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
            if staged_path != fits_path:
                try:
                    staged_path.unlink()
                except Exception:
                    pass
            return False, timeout_s, "", "timeout", cmd, None
        except Exception as e:
            if staged_path != fits_path:
                try:
                    staged_path.unlink()
                except Exception:
                    pass
            return False, 0.0, "", str(e), cmd, None

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

    def _wcs_rotation_deg(self, w: WCS) -> float:
        """Extract rotation angle from WCS CD matrix (degrees, E of N)."""
        try:
            if not w.has_celestial:
                return float("nan")
            # Get CD matrix or compute from PC+CDELT
            if hasattr(w.wcs, 'cd') and w.wcs.cd is not None:
                cd = w.wcs.cd
            elif hasattr(w.wcs, 'pc') and w.wcs.pc is not None:
                pc = w.wcs.pc
                cdelt = w.wcs.cdelt
                cd = pc * cdelt[:, np.newaxis]
            else:
                return float("nan")
            # Rotation from CD matrix: theta = atan2(-CD1_2, CD2_2)
            rot_rad = np.arctan2(-cd[0, 1], cd[1, 1])
            return float(np.degrees(rot_rad))
        except Exception:
            return float("nan")

    def _wcs_center_coords(self, w: WCS, nx: int, ny: int) -> tuple:
        """Get center RA/Dec from WCS."""
        try:
            if not w.has_celestial:
                return (float("nan"), float("nan"))
            cx, cy = nx / 2.0, ny / 2.0
            sky = w.pixel_to_world(cx, cy)
            return (float(sky.ra.deg), float(sky.dec.deg))
        except Exception:
            return (float("nan"), float("nan"))

    def _wcs_sip_order(self, hdr) -> int:
        """Get SIP distortion polynomial order (0 if none)."""
        try:
            # Check for SIP keywords
            a_order = hdr.get("A_ORDER", 0)
            b_order = hdr.get("B_ORDER", 0)
            return max(int(a_order), int(b_order))
        except Exception:
            return 0

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
        step5_out = step5_dir(self.result_dir)
        step5_out.mkdir(parents=True, exist_ok=True)
        cache_path = step5_out / "gaia_fov.ecsv"
        meta_path = step5_out / "gaia_fov_meta.json"
        legacy_candidates = [
            (
                legacy_step7_wcs_dir(self.result_dir) / "gaia_fov.ecsv",
                legacy_step7_wcs_dir(self.result_dir) / "gaia_fov_meta.json",
            ),
            (
                self.result_dir / "gaia_fov.ecsv",
                self.result_dir / "gaia_fov_meta.json",
            ),
        ]
        retry = int(getattr(self.params.P, "gaia_retry", 2))
        backoff_s = float(getattr(self.params.P, "gaia_backoff_s", 6.0))
        mag_max = float(getattr(self.params.P, "gaia_mag_max", 18.0))
        allow_no_cache = bool(getattr(self.params.P, "gaia_allow_no_cache", True))

        # 캐시 유효성 체크 - 좌표가 맞는지 확인
        cache_valid = False
        df_cache = None
        meta_probe = None
        for cpath, mpath in [(cache_path, meta_path), *legacy_candidates]:
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

        df_cache = self._load_gaia_cache_if_ok(cache_path)
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

            # Optional QC filtering
            require_qc = bool(getattr(self.params.P, "wcs_require_qc_pass", True))
            if require_qc:
                step5_out = step5_dir(self.result_dir)
                step5_out.mkdir(parents=True, exist_ok=True)
                qpath = step5_out / "frame_quality.csv"
                if not qpath.exists():
                    qpath = legacy_step7_wcs_dir(self.result_dir) / "frame_quality.csv"
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
            astnet_local_enable = bool(getattr(self.params.P, "astnet_local_enable", False))
            astnet_use_wsl = bool(getattr(self.params.P, "astnet_local_use_wsl", True))
            astnet_timeout_s = float(getattr(self.params.P, "astnet_local_timeout_s", 300.0))
            astnet_downsample = int(getattr(self.params.P, "astnet_local_downsample", 2))
            astnet_scale_low = float(getattr(self.params.P, "astnet_local_scale_low", 0.0))
            astnet_scale_high = float(getattr(self.params.P, "astnet_local_scale_high", 0.0))
            astnet_radius_deg = float(getattr(self.params.P, "astnet_local_radius_deg", 8.0))
            astnet_keep_outputs = bool(getattr(self.params.P, "astnet_local_keep_outputs", True))
            astnet_use_cache = bool(getattr(self.params.P, "astnet_local_use_cache", True))
            astnet_max_objs = int(getattr(self.params.P, "astnet_local_max_objs", 2000))
            astnet_cpulimit_s = float(getattr(self.params.P, "astnet_local_cpulimit_s", 30.0))
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
            L(f"[WCS] astnet_local_enable={astnet_local_enable} use_wsl={astnet_use_wsl} timeout_s={astnet_timeout_s} downsample={astnet_downsample}")

            # Determine Gaia center - PRIORITY: FITS header > project_state
            # FITS header OBJCTRA/OBJCTDEC is more reliable as it comes from the actual observation
            header_coord = None
            try:
                sample = files[0]
                if self.use_cropped:
                    fits_path = step2_cropped_dir(self.result_dir) / sample
                else:
                    fits_path = self.params.get_file_path(sample)
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
                sample_path = step2_cropped_dir(self.result_dir) / sample
            else:
                sample_path = self.params.get_file_path(sample)
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
                    fits_path = step2_cropped_dir(self.result_dir) / filename
                else:
                    fits_path = self.params.get_file_path(filename)

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

                ok_astap, rc, dt_astap, out_s, err_s, cmd = self._run_astap(
                    fits_path, fov_deg=fov_deg, radius_deg=astap_radius, timeout_s=astap_timeout
                )
                cmd_str = " ".join(str(c) for c in cmd)
                if not ok_astap:
                    L(f"{filename}: ASTAP fail rc={rc} dt={dt_astap:.1f}s err={str(err_s)[:120]}")
                    L(f"{filename}: ASTAP cmd={cmd_str}")
                    if not astnet_local_enable:
                        return filename, {
                            "ok": False,
                            "status": f"astap_fail rc={rc}",
                            "pix_fit": pix_fit,
                            "elapsed": float(dt_astap),
                            "refine": refine_note,
                        }

                astnet_ok = False
                astnet_dt = np.nan
                astnet_stdout = ""
                astnet_stderr = ""
                astnet_cmd = []
                astnet_new_path = None
                solver = "astap"
                used_elapsed = float(dt_astap)

                # WCS를 FITS 파일에 저장 (writeto 사용 - Windows 호환성)
                with fits.open(fits_path, memmap=False, ignore_missing_simple=True) as hdul:
                    hdr = hdul[0].header
                    data = hdul[0].data
                    wcs_ok = False
                    if ok_astap:
                        try:
                            w0 = WCS(hdr, relax=True)
                            wcs_ok = w0.has_celestial
                        except Exception:
                            wcs_ok = False

                        if not wcs_ok:
                            wcs_ok = self._try_ingest_wcs(fits_path, hdr)

                    if (not ok_astap or not wcs_ok) and astnet_local_enable:
                        scale_low = astnet_scale_low
                        scale_high = astnet_scale_high
                        if scale_low <= 0 or scale_high <= 0:
                            scale_low = float(pix_arc) * 0.85
                            scale_high = float(pix_arc) * 1.15

                        outdir = meta_dir / "astnet_local"
                        astnet_ok, astnet_dt, astnet_stdout, astnet_stderr, astnet_cmd, astnet_new_path = (
                            self._run_solve_field(
                                fits_path,
                                center_coord=center_coord,
                                scale_low=scale_low,
                                scale_high=scale_high,
                                radius_deg=astnet_radius_deg,
                                downsample=astnet_downsample,
                                timeout_s=astnet_timeout_s,
                                outdir=outdir,
                                use_wsl=astnet_use_wsl,
                                use_cache=astnet_use_cache,
                                max_objs=astnet_max_objs,
                                cpulimit_s=astnet_cpulimit_s,
                            )
                        )
                        cmd_wsl = " ".join(str(c) for c in astnet_cmd)
                        L(f"[ASTNET_WSL] {filename} ok={astnet_ok} dt={astnet_dt:.1f}s")
                        L(f"[ASTNET_WSL] cmd={cmd_wsl}")
                        if astnet_stdout:
                            L(f"[ASTNET_WSL] stdout={str(astnet_stdout)[:200]}")
                        if astnet_stderr:
                            L(f"[ASTNET_WSL] stderr={str(astnet_stderr)[:200]}")

                        if astnet_ok and astnet_new_path is not None and astnet_new_path.exists():
                            try:
                                with fits.open(astnet_new_path, memmap=False) as hdul_new:
                                    new_hdr = hdul_new[0].header
                                    self._copy_wcs_keywords(new_hdr, hdr)
                                w_new = WCS(hdr, relax=True)
                                wcs_ok = w_new.has_celestial
                            except Exception:
                                wcs_ok = False

                            if wcs_ok:
                                hdr["WCSSRC"] = ("ASTNET_WSL", "WCS source")
                                solver = "astnet_wsl"
                                used_elapsed = float(astnet_dt)

                        if not astnet_keep_outputs:
                            for p in outdir.glob(f"{fits_path.stem}.*"):
                                try:
                                    p.unlink()
                                except Exception:
                                    pass

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

                        # Use final WCS (refined if available)
                        w_final = WCS(hdr, relax=True)

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

                        # Additional WCS quality stats
                        wcs_rot_deg = self._wcs_rotation_deg(w_final)
                        nx = int(hdr.get("NAXIS1", 0))
                        ny = int(hdr.get("NAXIS2", 0))
                        center_ra, center_dec = self._wcs_center_coords(w_final, nx, ny)
                        sip_order = self._wcs_sip_order(hdr)

                        if np.isfinite(wcs_rot_deg):
                            hdr["WCSROT"] = (float(wcs_rot_deg), "WCS rotation (deg, E of N)")
                        if np.isfinite(center_ra):
                            hdr["WCSCRA"] = (float(center_ra), "WCS center RA (deg)")
                        if np.isfinite(center_dec):
                            hdr["WCSCDEC"] = (float(center_dec), "WCS center Dec (deg)")
                        if sip_order > 0:
                            hdr["WCSSIP"] = (int(sip_order), "SIP distortion order")

                        status = "ok_astnet_wsl" if solver == "astnet_wsl" else "ok"
                    else:
                        hdr["WCS_OK"] = (False, "WCS solve failed")
                        status = f"astap_fail rc={rc}" if not ok_astap else "wcs_missing"
                        # Set defaults for failed WCS
                        wcs_rot_deg = np.nan
                        center_ra = np.nan
                        center_dec = np.nan
                        sip_order = 0

                # writeto로 확실하게 저장 (Windows 호환)
                fits.writeto(fits_path, data, hdr, overwrite=True)

                meta = {
                    "fname": filename,
                    "ok": bool(wcs_ok),
                    "status": status,
                    "pix_fit": float(pix_fit) if np.isfinite(pix_fit) else None,
                    "elapsed": float(used_elapsed),
                    "refine": refine_note,
                    "resid_med": float(resid_med) if np.isfinite(resid_med) else None,
                    "resid_max": float(resid_max) if np.isfinite(resid_max) else None,
                    "match_n": int(match_n),
                    "gaia_source": str(gaia_src),
                    "solver": solver,
                    # New WCS quality stats
                    "wcs_rot_deg": float(wcs_rot_deg) if np.isfinite(wcs_rot_deg) else None,
                    "center_ra_deg": float(center_ra) if np.isfinite(center_ra) else None,
                    "center_dec_deg": float(center_dec) if np.isfinite(center_dec) else None,
                    "sip_order": int(sip_order) if sip_order > 0 else None,
                    # Solver details
                    "astap_ok": bool(ok_astap),
                    "astap_rc": int(rc),
                    "astap_elapsed": float(dt_astap),
                    "astnet_wsl_ok": bool(astnet_ok),
                    "astnet_wsl_elapsed": float(astnet_dt) if np.isfinite(astnet_dt) else None,
                    "astnet_wsl_cmd": " ".join(str(c) for c in astnet_cmd) if astnet_cmd else "",
                    "astnet_wsl_stdout": str(astnet_stdout)[:2000],
                    "astnet_wsl_stderr": str(astnet_stderr)[:2000],
                }
                L(
                    f"{filename}: {status} pix_fit={pix_fit:.4f} dt={used_elapsed:.1f}s "
                    f"refine={refine_note or '-'} resid_med={resid_med if np.isfinite(resid_med) else '-'} "
                    f"match_n={match_n}"
                )
                (meta_dir / f"wcs_{filename}.json").write_text(
                    json.dumps(meta, indent=2), encoding="utf-8"
                )
                return filename, meta

            completed = 0
            max_workers = int(getattr(self.params.P, "wcs_max_workers", 1))
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
                step5_out = step5_dir(self.result_dir)
                step5_out.mkdir(parents=True, exist_ok=True)
                df.to_csv(step5_out / "wcs_solve_summary.csv", index=False)
            except Exception:
                pass

            summary = {
                "total": len(results),
                "ok": sum(1 for r in results if r.get("ok")),
            }
            self.finished.emit(summary)
        except Exception as e:
            self.error.emit("WORKER", str(e))
            self.finished.emit({})


class AstrometryNetWorker(QThread):
    """Worker thread for local astrometry.net (solve-field) WCS solving"""
    progress = pyqtSignal(int, int, str)
    file_done = pyqtSignal(str, dict)
    refine_done = pyqtSignal(str, dict)  # 파일명, refine 결과
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
        legacy_candidates = [
            (
                legacy_step7_wcs_dir(self.result_dir) / "gaia_fov.ecsv",
                legacy_step7_wcs_dir(self.result_dir) / "gaia_fov_meta.json",
            ),
            (
                self.result_dir / "gaia_fov.ecsv",
                self.result_dir / "gaia_fov_meta.json",
            ),
        ]
        retry = int(getattr(self.params.P, "gaia_retry", 2))
        backoff_s = float(getattr(self.params.P, "gaia_backoff_s", 6.0))
        mag_max = float(getattr(self.params.P, "gaia_mag_max", 18.0))
        allow_no_cache = bool(getattr(self.params.P, "gaia_allow_no_cache", True))

        cache_valid = False
        df_cache = None
        meta_probe = None
        for cpath, mpath in [(cache_path, meta_path), *legacy_candidates]:
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
        if not meta_json.exists():
            fallback = step4_dir(self.result_dir) / f"detect_{fname}.json"
            if fallback.exists():
                meta_json = fallback
        if meta_json.exists():
            try:
                meta = json.loads(meta_json.read_text(encoding="utf-8"))
                fpx = float(
                    meta.get(
                        "fwhm_med_rad_px",
                        meta.get("fwhm_med_px", meta.get("fwhm_px", np.nan)),
                    )
                )
                farc = float(
                    meta.get(
                        "fwhm_med_rad_arcsec",
                        meta.get("fwhm_med_arc", meta.get("fwhm_arcsec", np.nan)),
                    )
                )
                return fpx, farc
            except Exception:
                pass
        return float(getattr(self.params.P, "fwhm_seed_px", 6.0)), np.nan

    def _load_detect_xy(self, fname: str):
        csv_path = self.cache_dir / f"detect_{fname}.csv"
        if not csv_path.exists():
            fallback = step4_dir(self.result_dir) / f"detect_{fname}.csv"
            if fallback.exists():
                csv_path = fallback
            else:
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

    def run(self):
        results = []
        total = len(self.file_list)

        # --- 파라미터 로드 ---
        pix_arc = float(getattr(self.params.P, "pixel_scale_arcsec", np.nan))
        if not np.isfinite(pix_arc) or pix_arc <= 0:
            self.error.emit("WORKER", "pixel_scale_arcsec is not set; run instrument setup first.")
            self.finished.emit({})
            return

        use_wsl = bool(getattr(self.params.P, "astnet_local_use_wsl", True))
        timeout_s = float(getattr(self.params.P, "astnet_local_timeout_s", 300.0))

        downsample = int(getattr(self.params.P, "astnet_local_downsample", 2))
        max_objs = int(getattr(self.params.P, "astnet_local_max_objs", 2000))
        scale_low = float(getattr(self.params.P, "astnet_local_scale_low", 0.0))
        scale_high = float(getattr(self.params.P, "astnet_local_scale_high", 0.0))
        radius_deg = float(getattr(self.params.P, "astnet_local_radius_deg", 8.0))
        keep_outputs = bool(getattr(self.params.P, "astnet_local_keep_outputs", True))
        use_cache = bool(getattr(self.params.P, "astnet_local_use_cache", True))
        cpulimit_s = float(getattr(self.params.P, "astnet_local_cpulimit_s", 30.0))
        max_workers = get_parallel_workers(self.params)

        if scale_low <= 0 or scale_high <= 0:
            scale_low = float(pix_arc) * 0.85
            scale_high = float(pix_arc) * 1.15

        outdir = self.cache_dir / "wcs_solve" / "astnet_local"

        self.log_message.emit(f"Starting parallel plate solving with {max_workers} workers...")
        self.log_message.emit(f"  Scale: {scale_low:.4f} - {scale_high:.4f} arcsec/px")
        self.log_message.emit(f"  Downsample: {downsample}, Max objs: {max_objs}")

        # 내부 함수: 단일 파일 처리 로직 (스레드에서 실행됨)
        def process_single_file(filename):
            if self._stop_requested:
                return filename, {"ok": False, "status": "stopped"}

            if self.use_cropped:
                fits_path = step2_cropped_dir(self.result_dir) / filename
            else:
                fits_path = self.params.get_file_path(filename)

            if not fits_path.exists():
                return filename, {"ok": False, "status": "file_not_found"}

            # 헤더에서 중심 좌표 읽기
            center_coord = None
            try:
                # memmap=False로 읽어서 파일 핸들 즉시 반환 유도
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

            # solve-field 실행
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
                            # 헤더 업데이트 (Thread Safe하게 처리하기 위해 Lock 사용 권장되나, 
                            # 파일이 서로 다르므로 여기선 직접 호출)
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

            # 임시 파일 정리
            if not keep_outputs:
                for p in outdir.glob(f"{fits_path.stem}.*"):
                    try: p.unlink()
                    except Exception: pass

            return filename, result

        file_results = {}  # filename -> result 매핑
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 모든 작업을 큐에 등록
            future_to_file = {executor.submit(process_single_file, f): f for f in self.file_list}

            completed_count = 0
            for future in as_completed(future_to_file):
                if self._stop_requested:
                    break

                fname = future_to_file[future]
                try:
                    filename, res = future.result()
                    res["filename"] = filename  # 결과에 파일명 저장
                    results.append(res)
                    file_results[filename] = res

                    # UI 업데이트 시그널
                    self.file_done.emit(filename, res)

                    # 로그 출력 (성공/실패 여부에 따라)
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

        # --- Gaia 쿼리 및 WCS Refine ---
        if self._stop_requested:
            self.finished.emit({"total": len(results), "ok": sum(1 for r in results if r.get("ok"))})
            return

        # 성공한 프레임에서 중심 좌표 얻기
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

            # FOV 계산
            sample_fname = self.file_list[0]
            if self.use_cropped:
                sample_path = step2_cropped_dir(self.result_dir) / sample_fname
            else:
                sample_path = self.params.get_file_path(sample_fname)
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

        # WCS Refine 수행
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
                    # Detection 로드
                    det_xy, _ = self._load_detect_xy(filename)
                    fwhm_px, _ = self._load_fwhm_for_frame(filename)

                    with fits.open(fits_path, mode="update", memmap=False) as hdul:
                        hdr = hdul[0].header
                        w = WCS(hdr, relax=True)

                        ok_refine, refine_note, resid_med, resid_max, match_n = self._refine_crpix_by_match(
                            w, hdr, det_xy, gaia_df, fwhm_px, refine_max_match
                        )

                        # 결과 업데이트
                        res["refine"] = refine_note
                        res["resid_med"] = resid_med
                        res["resid_max"] = resid_max
                        res["match_n"] = match_n

                        if ok_refine:
                            hdr["WCSSRC"] = ("ASTNET_REFINED", "WCS source (refined with Gaia)")
                            self.log_message.emit(f"  [Refine] {filename}: {refine_note}, resid_med={resid_med:.2f}\"")
                        else:
                            self.log_message.emit(f"  [Refine] {filename}: skip - {refine_note}")

                    # Refine 결과 시그널 emit
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

                self.progress.emit(total + i + 1, total + len(solved_frames), f"Refining {i+1}/{len(solved_frames)}")
        else:
            self.log_message.emit("[Refine] Skipped - no Gaia data available")

        # --- 마무리 ---
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

        # Astrometry.net Tab
        self.astrometrynet_tab = QWidget()
        self.setup_astrometrynet_tab()
        self.tab_widget.addTab(self.astrometrynet_tab, "Astrometry.net (Local)")

        self.setup_log_window()
        self.populate_file_list()

    def setup_astap_tab(self):
        """Setup ASTAP tab UI"""
        layout = QVBoxLayout(self.astap_tab)

        info = QLabel(
            "Solve WCS for all frames using ASTAP (local)."
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
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        results_layout.addWidget(self.results_table)

        layout.addWidget(results_group)

    def setup_astrometrynet_tab(self):
        """Setup Astrometry.net tab UI"""
        layout = QVBoxLayout(self.astrometrynet_tab)

        info_text = "Solve WCS for all frames using local astrometry.net (solve-field)."
        info_style = "QLabel { background-color: #E8F5E9; padding: 10px; border-radius: 5px; }"
        info = QLabel(info_text)
        info.setStyleSheet(info_style)
        layout.addWidget(info)

        ref_group = QGroupBox("Frame List")
        ref_layout = QVBoxLayout(ref_group)

        ref_info = QLabel("Frames listed below will be solved automatically.")
        ref_info.setWordWrap(True)
        ref_layout.addWidget(ref_info)

        self.ref_frame_list = QListWidget()
        self.ref_frame_list.setSelectionMode(QListWidget.NoSelection)
        self.ref_frame_list.setMaximumHeight(150)
        ref_layout.addWidget(self.ref_frame_list)

        layout.addWidget(ref_group)

        control_layout = QHBoxLayout()

        btn_astnet_params = QPushButton("Astrometry.net Parameters")
        btn_astnet_params.setStyleSheet(
            "QPushButton { background-color: #9C27B0; color: white; font-weight: bold; padding: 8px 15px; }"
        )
        btn_astnet_params.clicked.connect(self.open_astrometrynet_parameters_dialog)
        control_layout.addWidget(btn_astnet_params)

        control_layout.addStretch()

        self.btn_solve_astrometrynet = QPushButton("Solve All Frames")
        self.btn_solve_astrometrynet.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; padding: 8px 20px; }")
        self.btn_solve_astrometrynet.clicked.connect(self.run_astrometrynet_solve)
        control_layout.addWidget(self.btn_solve_astrometrynet)

        self.btn_stop_astrometrynet = QPushButton("Stop")
        self.btn_stop_astrometrynet.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; padding: 8px 15px; }")
        self.btn_stop_astrometrynet.clicked.connect(self.stop_astrometrynet_solve)
        self.btn_stop_astrometrynet.setEnabled(False)
        control_layout.addWidget(self.btn_stop_astrometrynet)

        btn_log2 = QPushButton("Log")
        btn_log2.setStyleSheet("QPushButton { background-color: #607D8B; color: white; font-weight: bold; padding: 8px 15px; }")
        btn_log2.clicked.connect(self.show_log_window)
        control_layout.addWidget(btn_log2)

        layout.addLayout(control_layout)

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

        results_group = QGroupBox("Astrometry.net Results")
        results_layout = QVBoxLayout(results_group)

        self.astrometrynet_results_table = QTableWidget()
        self.astrometrynet_results_table.setColumnCount(8)
        self.astrometrynet_results_table.setHorizontalHeaderLabels([
            "File", "Status", "RA", "Dec", "PixScale", "Refine", "Resid(\")", "Elapsed (s)"
        ])
        self.astrometrynet_results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.astrometrynet_results_table.horizontalHeader().setStretchLastSection(True)
        self.astrometrynet_results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.astrometrynet_results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        results_layout.addWidget(self.astrometrynet_results_table)

        layout.addWidget(results_group)

    def auto_select_ref_frame(self):
        best_frame = None
        best_count = 0

        for fname in self.file_list:
            detect_csv = self.params.P.cache_dir / f"detect_{fname}.csv"
            if not detect_csv.exists():
                alt = step4_dir(self.params.P.result_dir) / f"detect_{fname}.csv"
                if alt.exists():
                    detect_csv = alt
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

    def select_all_ref_frames(self):
        for i in range(self.ref_frame_list.count()):
            self.ref_frame_list.item(i).setSelected(True)

    def run_astrometrynet_solve(self):
        if not self.file_list:
            QMessageBox.warning(self, "Warning", "No frames found to solve")
            return
        file_list = list(self.file_list)

        target_coord = None
        ra = getattr(self.params.P, "target_ra_deg", None)
        dec = getattr(self.params.P, "target_dec_deg", None)
        if ra is not None and dec is not None:
            try:
                target_coord = SkyCoord(float(ra) * u.deg, float(dec) * u.deg)
            except Exception:
                target_coord = None

        self.astrometrynet_results_table.setRowCount(0)
        self.results = {}

        # Start solving in thread
        self.astrometrynet_worker = AstrometryNetWorker(
            file_list,
            self.params,
            self.params.P.data_dir,
            self.params.P.result_dir,
            self.params.P.cache_dir,
            self.use_cropped,
            target_coord=target_coord,
        )
        self.astrometrynet_worker.progress.connect(self.on_astrometrynet_progress)
        self.astrometrynet_worker.file_done.connect(self.on_astrometrynet_file_done)
        self.astrometrynet_worker.refine_done.connect(self.on_astrometrynet_refine_done)
        self.astrometrynet_worker.log_message.connect(self.log)
        self.astrometrynet_worker.finished.connect(self.on_astrometrynet_finished)
        self.astrometrynet_worker.error.connect(self.on_astrometrynet_error)

        self.btn_solve_astrometrynet.setEnabled(False)
        self.btn_stop_astrometrynet.setEnabled(True)
        self.astrometrynet_progress.setValue(0)
        self.astrometrynet_status.setText("Starting local astrometry.net...")
        self.log("=" * 50)
        self.log("Starting local astrometry.net (solve-field) plate solving...")
        self.log(f"Frames: {len(file_list)}")
        self.astrometrynet_worker.start()
        self.show_log_window()

    def stop_astrometrynet_solve(self):
        if self.astrometrynet_worker and self.astrometrynet_worker.isRunning():
            self.astrometrynet_worker.stop()

    def on_astrometrynet_progress(self, current, total, status):
        pct = int(100 * current / max(1, total))
        self.astrometrynet_progress.setValue(pct)
        self.astrometrynet_status.setText(status)

    def on_astrometrynet_file_done(self, filename, result):
        row = self.astrometrynet_results_table.rowCount()
        self.astrometrynet_results_table.insertRow(row)
        self.astrometrynet_results_table.setItem(row, 0, QTableWidgetItem(filename))
        self.astrometrynet_results_table.setItem(row, 1, QTableWidgetItem(result.get("status", "")))
        ra = float(result.get("ra", 0.0))
        dec = float(result.get("dec", 0.0))
        pixscale = float(result.get("pixscale", 0.0))
        refine = result.get("refine", "-")
        resid_med = result.get("resid_med", np.nan)
        elapsed = float(result.get("elapsed_s", 0.0))
        self.astrometrynet_results_table.setItem(row, 2, QTableWidgetItem(f"{ra:.6f}" if np.isfinite(ra) else "-"))
        self.astrometrynet_results_table.setItem(row, 3, QTableWidgetItem(f"{dec:.6f}" if np.isfinite(dec) else "-"))
        self.astrometrynet_results_table.setItem(row, 4, QTableWidgetItem(f"{pixscale:.4f}" if np.isfinite(pixscale) and pixscale > 0 else "-"))
        self.astrometrynet_results_table.setItem(row, 5, QTableWidgetItem(str(refine) if refine else "-"))
        self.astrometrynet_results_table.setItem(row, 6, QTableWidgetItem(f"{resid_med:.2f}" if np.isfinite(resid_med) else "-"))
        self.astrometrynet_results_table.setItem(row, 7, QTableWidgetItem(f"{elapsed:.1f}" if np.isfinite(elapsed) and elapsed > 0 else "-"))

        if result.get("ok"):
            self.results[filename] = result
            self.log(f"Astrometry.net solved: {filename} (RA={result.get('ra', 0):.4f}, Dec={result.get('dec', 0):.4f})")

    def on_astrometrynet_error(self, filename, error):
        self.log(f"Astrometry.net ERROR {filename}: {error}")

    def on_astrometrynet_refine_done(self, filename, result):
        """Refine 결과로 테이블 업데이트"""
        # 파일명으로 해당 행 찾기
        for row in range(self.astrometrynet_results_table.rowCount()):
            item = self.astrometrynet_results_table.item(row, 0)
            if item and item.text() == filename:
                refine = result.get("refine", "-")
                resid_med = result.get("resid_med", np.nan)
                self.astrometrynet_results_table.setItem(row, 5, QTableWidgetItem(str(refine) if refine else "-"))
                self.astrometrynet_results_table.setItem(row, 6, QTableWidgetItem(f"{resid_med:.2f}" if np.isfinite(resid_med) else "-"))
                # results에도 업데이트
                if filename in self.results:
                    self.results[filename].update(result)
                break

    def on_astrometrynet_finished(self, summary):
        self.btn_solve_astrometrynet.setEnabled(True)
        self.btn_stop_astrometrynet.setEnabled(False)
        n_ok = summary.get("ok", 0)
        self.astrometrynet_progress.setValue(100)
        self.astrometrynet_status.setText(f"Done: {n_ok}/{summary.get('total', 0)} solved")
        if n_ok > 0:
            self.log(f"Astrometry.net: {n_ok} frames solved successfully")
        self.save_state()
        self.update_navigation_buttons()

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
            cropped_dir = legacy_cropped
        else:
            if self.file_manager and not self.file_manager.filenames:
                try:
                    self.file_manager.scan_files()
                except Exception:
                    pass
            files = list(self.file_manager.filenames) if self.file_manager else []
            self.use_cropped = False

        self.file_list = list(files)

        # Also populate ref_frame_list for Astrometry.net tab
        self.ref_frame_list.clear()
        for fname in self.file_list:
            item = QListWidgetItem(fname)
            # Add star count info if available
            detect_csv = self.params.P.cache_dir / f"detect_{fname}.csv"
            if not detect_csv.exists():
                alt = step4_dir(self.params.P.result_dir) / f"detect_{fname}.csv"
                if alt.exists():
                    detect_csv = alt
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
        dialog.resize(560, 720)

        layout = QVBoxLayout(dialog)
        wcs_form = QFormLayout()

        self.param_astap_exe = QLineEdit(str(getattr(self.params.P, "astap_exe", "astap_cli.exe")))
        wcs_form.addRow("ASTAP CLI Path:", self.param_astap_exe)

        self.param_timeout = QDoubleSpinBox()
        self.param_timeout.setRange(10, 1000)
        self.param_timeout.setValue(float(getattr(self.params.P, "astap_timeout_s", 120.0)))
        wcs_form.addRow("Timeout (s):", self.param_timeout)

        self.param_radius = QDoubleSpinBox()
        self.param_radius.setRange(0.5, 30.0)
        self.param_radius.setValue(float(getattr(self.params.P, "astap_search_radius_deg", 8.0)))
        wcs_form.addRow("Search Radius (deg):", self.param_radius)

        self.param_astap_db = QComboBox()
        self.param_astap_db.addItems(["D50", "D80"])
        current_db = str(getattr(self.params.P, "astap_database", "D50"))
        idx = self.param_astap_db.findText(current_db)
        if idx >= 0:
            self.param_astap_db.setCurrentIndex(idx)
        wcs_form.addRow("ASTAP Star DB:", self.param_astap_db)

        self.param_annotate_variables = QCheckBox("Enable")
        self.param_annotate_variables.setChecked(bool(getattr(self.params.P, "astap_annotate_variables", False)))
        self.param_annotate_variables.setToolTip("ASTAP 변광성 데이터베이스로 변광성 주석 표시 (별도 설치 필요)")
        wcs_form.addRow("Annotate Variable Stars:", self.param_annotate_variables)

        self.param_fov_fudge = QDoubleSpinBox()
        self.param_fov_fudge.setRange(0.5, 2.0)
        self.param_fov_fudge.setSingleStep(0.05)
        self.param_fov_fudge.setValue(float(getattr(self.params.P, "astap_fov_fudge", 1.0)))
        wcs_form.addRow("FOV Fudge:", self.param_fov_fudge)

        self.param_downsample = QSpinBox()
        self.param_downsample.setRange(1, 8)
        self.param_downsample.setValue(int(getattr(self.params.P, "astap_downsample_z", 2)))
        wcs_form.addRow("Downsample Z:", self.param_downsample)

        self.param_max_stars = QSpinBox()
        self.param_max_stars.setRange(50, 5000)
        self.param_max_stars.setValue(int(getattr(self.params.P, "astap_max_stars_s", 500)))
        wcs_form.addRow("Max Stars (S):", self.param_max_stars)

        self.param_max_workers = QSpinBox()
        self.param_max_workers.setRange(1, 16)
        self.param_max_workers.setValue(int(getattr(self.params.P, "wcs_max_workers", 1)))
        wcs_form.addRow("Max Workers:", self.param_max_workers)

        self.param_require_qc = QCheckBox("Enable")
        self.param_require_qc.setChecked(bool(getattr(self.params.P, "wcs_require_qc_pass", True)))
        wcs_form.addRow("QC Pass Only:", self.param_require_qc)

        self.param_refine_enable = QCheckBox("Enable")
        self.param_refine_enable.setChecked(bool(getattr(self.params.P, "wcs_refine_enable", True)))
        wcs_form.addRow("Refine CRPIX:", self.param_refine_enable)

        self.param_refine_max_match = QSpinBox()
        self.param_refine_max_match.setRange(50, 5000)
        self.param_refine_max_match.setValue(int(getattr(self.params.P, "wcs_refine_max_match", 600)))
        wcs_form.addRow("Refine Max Match:", self.param_refine_max_match)

        self.param_refine_match_r = QDoubleSpinBox()
        self.param_refine_match_r.setRange(0.5, 5.0)
        self.param_refine_match_r.setSingleStep(0.1)
        self.param_refine_match_r.setValue(float(getattr(self.params.P, "wcs_refine_match_r_fwhm", 1.6)))
        wcs_form.addRow("Refine Match R (×FWHM):", self.param_refine_match_r)

        self.param_refine_min_match = QSpinBox()
        self.param_refine_min_match.setRange(5, 500)
        self.param_refine_min_match.setValue(int(getattr(self.params.P, "wcs_refine_min_match", 50)))
        wcs_form.addRow("Refine Min Match:", self.param_refine_min_match)

        self.param_gaia_fudge = QDoubleSpinBox()
        self.param_gaia_fudge.setRange(0.5, 3.0)
        self.param_gaia_fudge.setSingleStep(0.05)
        self.param_gaia_fudge.setValue(float(getattr(self.params.P, "gaia_radius_fudge", 1.35)))
        wcs_form.addRow("Gaia Radius Fudge:", self.param_gaia_fudge)

        self.param_gaia_mag_max = QDoubleSpinBox()
        self.param_gaia_mag_max.setRange(10.0, 25.0)
        self.param_gaia_mag_max.setSingleStep(0.5)
        self.param_gaia_mag_max.setValue(float(getattr(self.params.P, "gaia_mag_max", 18.0)))
        wcs_form.addRow("Gaia Mag Max:", self.param_gaia_mag_max)

        self.param_gaia_retry = QSpinBox()
        self.param_gaia_retry.setRange(0, 10)
        self.param_gaia_retry.setValue(int(getattr(self.params.P, "gaia_retry", 2)))
        wcs_form.addRow("Gaia Retry:", self.param_gaia_retry)

        self.param_gaia_backoff = QDoubleSpinBox()
        self.param_gaia_backoff.setRange(0.0, 30.0)
        self.param_gaia_backoff.setSingleStep(1.0)
        self.param_gaia_backoff.setValue(float(getattr(self.params.P, "gaia_backoff_s", 6.0)))
        wcs_form.addRow("Gaia Backoff (s):", self.param_gaia_backoff)

        self.param_gaia_allow_no_cache = QCheckBox("Allow")
        self.param_gaia_allow_no_cache.setChecked(bool(getattr(self.params.P, "gaia_allow_no_cache", True)))
        wcs_form.addRow("Gaia Allow No Cache:", self.param_gaia_allow_no_cache)

        layout.addLayout(wcs_form)

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
        self.params.P.wcs_max_workers = self.param_max_workers.value()
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
            "wcs_max_workers": getattr(self.params.P, "wcs_max_workers", 1),
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

    def restore_state(self):
        state_data = self.project_state.get_step_data("wcs_plate_solve")
        if state_data:
            for key, val in state_data.items():
                if hasattr(self.params.P, key):
                    setattr(self.params.P, key, val)
