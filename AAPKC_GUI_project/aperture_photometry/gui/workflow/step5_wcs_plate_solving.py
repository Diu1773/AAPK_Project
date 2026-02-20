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
import os
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
    crop_rect_path,
    step4_dir,
    step5_dir,
    legacy_step5_refbuild_dir,
    legacy_step7_wcs_dir,
    legacy_step7_refbuild_dir,
)
from ...utils.constants import get_parallel_workers
from ...utils.io_utils import parse_int64_series


def _tail_text(value: str | None, limit: int = 800, max_lines: int = 8) -> str:
    if value is None:
        return ""
    s = str(value).replace("\r\n", "\n").replace("\r", "\n").strip()
    if not s:
        return ""
    lines = [ln.strip() for ln in s.split("\n") if ln.strip()]
    if max_lines > 0 and len(lines) > max_lines:
        lines = lines[-max_lines:]
    one_line = " | ".join(lines)
    if len(one_line) > limit:
        one_line = "..." + one_line[-limit:]
    return one_line


def _exc_brief(exc: Exception, limit: int = 260) -> str:
    return _tail_text(f"{type(exc).__name__}: {exc}", limit=limit, max_lines=4)


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


def _build_source_signature(path: Path, *, use_cropped: bool) -> dict:
    sig = {
        "source_path": _norm_path_key(path),
        "source_use_cropped": bool(use_cropped),
        "source_size": None,
        "source_mtime_ns": None,
    }
    try:
        st = path.stat()
        sig["source_size"] = int(st.st_size)
        sig["source_mtime_ns"] = int(st.st_mtime_ns)
    except Exception:
        pass
    return sig


def _source_signature_matches(saved_sig: dict, current_sig: dict) -> bool:
    if not isinstance(saved_sig, dict) or not isinstance(current_sig, dict):
        return False
    if bool(saved_sig.get("source_use_cropped", None)) != bool(current_sig.get("source_use_cropped", None)):
        return False
    if _norm_path_key(saved_sig.get("source_path", "")) != _norm_path_key(current_sig.get("source_path", "")):
        return False
    try:
        saved_mtime_ns = int(saved_sig.get("source_mtime_ns"))
        curr_mtime_ns = int(current_sig.get("source_mtime_ns"))
    except Exception:
        return False
    return saved_mtime_ns == curr_mtime_ns


def _source_signature_detection_compatible(saved_sig: dict, current_sig: dict) -> bool:
    """Compatibility check for Step4 detection caches consumed by Step5.

    Step5 can update FITS headers in-place after solving, which changes mtime
    without invalidating detection XY. In that case, accept cache when path,
    crop mode, and file size still match.
    """
    if _source_signature_matches(saved_sig, current_sig):
        return True
    if not isinstance(saved_sig, dict) or not isinstance(current_sig, dict):
        return False
    if bool(saved_sig.get("source_use_cropped", None)) != bool(current_sig.get("source_use_cropped", None)):
        return False
    if _norm_path_key(saved_sig.get("source_path", "")) != _norm_path_key(current_sig.get("source_path", "")):
        return False
    try:
        saved_size = int(saved_sig.get("source_size"))
        curr_size = int(current_sig.get("source_size"))
    except Exception:
        return False
    if saved_size <= 0 or curr_size <= 0:
        return False
    return saved_size == curr_size


def _coerce_source_id_int64(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize Gaia source_id as signed int64 without float precision loss."""
    if df is None or df.empty or "source_id" not in df.columns:
        return df
    out = df.copy()
    sid = parse_int64_series(out["source_id"])
    valid = sid.notna()
    out = out.loc[valid].copy()
    out["source_id"] = sid.loc[valid].astype("int64")
    return out


def _canonicalize_windows_path_case(path_value) -> Path:
    """Best-effort case normalization for Windows paths.

    When data lives on case-sensitive external drives, incoming paths may have
    case variants (e.g. m13 vs M13). WSL access then fails even though Windows
    can open the file. Resolve each component by case-insensitive lookup.
    """
    try:
        wp = PureWindowsPath(str(path_value))
    except Exception:
        return Path(path_value)
    if not wp.drive:
        ptxt = str(path_value).replace("\\", "/")
        if not ptxt.startswith("/mnt/"):
            return Path(path_value)
        cur = Path("/")
        for part in Path(ptxt).parts[1:]:
            if not part:
                continue
            cand = cur / part
            if cand.exists():
                cur = cand
                continue
            try:
                low = part.lower()
                match = None
                for ent in os.scandir(cur):
                    name = ent.name
                    if name.lower() == low:
                        match = name
                        break
                if match is not None:
                    cur = cur / match
                    continue
            except Exception:
                pass
            cur = cand
        return cur

    root = Path(f"{wp.drive}\\")
    cur = root
    parts = list(wp.parts[1:])
    if not parts:
        return cur

    for part in parts:
        if not part:
            continue
        cand = cur / part
        if cand.exists():
            cur = cand
            continue
        try:
            low = part.lower()
            match = None
            for ent in os.scandir(cur):
                name = ent.name
                if name.lower() == low:
                    match = name
                    break
            if match is not None:
                cur = cur / match
                continue
        except Exception:
            pass
        cur = cand
    return cur


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
            canon = _canonicalize_windows_path_case(path)
            wp = PureWindowsPath(str(canon))
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
        wcs_path = outdir / f"{stem}.wcs"
        sig_path = outdir / f"{stem}.input.json"
        source_sig = _build_source_signature(fits_path, use_cropped=bool(self.use_cropped))
        if use_cache and new_path.exists() and (solved_path.exists() or wcs_path.exists()):
            cache_ok = False
            try:
                if sig_path.exists():
                    saved_sig = json.loads(sig_path.read_text(encoding="utf-8"))
                    cache_ok = _source_signature_detection_compatible(saved_sig, source_sig)
            except Exception:
                cache_ok = False
            if cache_ok:
                return True, 0.0, "cache_hit", "", [], new_path
        for p in outdir.glob(f"{stem}.*"):
            try:
                p.unlink()
            except Exception:
                pass
        staged_path = fits_path
        if stage_in_outdir and (not use_wsl):
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
        if not staged_path.exists():
            return False, 0.0, "", f"input_missing:{staged_path}", cmd, None
        if use_wsl and cmd and str(cmd[0]).lower() == "wsl":
            try:
                chk = subprocess.run(
                    ["wsl", "test", "-f", fits_arg],
                    capture_output=True,
                    text=True,
                    timeout=5.0,
                )
                if chk.returncode != 0:
                    err_msg = (
                        f"wsl_path_unavailable:{fits_arg} | "
                        "WSL cannot read this file path. Use ASTAP solver or move data to a local "
                        "drive/folder that WSL can access (e.g. C:) and rerun."
                    )
                    return False, 0.0, "", err_msg, cmd, None
            except FileNotFoundError:
                return (
                    False,
                    0.0,
                    "",
                    "wsl_not_found: WSL command not available. Disable WSL solve and use ASTAP.",
                    cmd,
                    None,
                )
            except Exception:
                pass

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
            # Some solve-field builds can return non-zero even when solution artifacts are created.
            # Prefer artifact existence over process return code.
            wcs_path = outdir / f"{stem}.wcs"
            ok = bool(new_path.exists() and (solved_path.exists() or wcs_path.exists()))
            if ok:
                try:
                    sig_path.write_text(json.dumps(source_sig, indent=2), encoding="utf-8")
                except Exception:
                    pass
            if staged_path != fits_path:
                try:
                    staged_path.unlink()
                except Exception:
                    pass
            return ok, dt, cp.stdout, cp.stderr, cmd, new_path
        except subprocess.TimeoutExpired as e:
            if staged_path != fits_path:
                try:
                    staged_path.unlink()
                except Exception:
                    pass
            out_s = e.stdout or ""
            err_s = e.stderr or ""
            err_msg = "timeout"
            err_tail = _tail_text(err_s, limit=1000, max_lines=10)
            if err_tail:
                err_msg = f"timeout | {err_tail}"
            return False, timeout_s, out_s, err_msg, cmd, None
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

    def _resolve_source_fits_path(self, fname: str) -> Path | None:
        if self.use_cropped:
            cand = step2_cropped_dir(self.result_dir) / fname
            if cand.exists():
                return cand
            legacy = self.result_dir / "cropped" / fname
            if legacy.exists():
                return legacy
        try:
            orig = Path(self.params.get_file_path(fname))
            if orig.exists():
                return orig
        except Exception:
            pass
        return None

    def _compatible_detect_signature(self, fname: str) -> dict | None:
        src = self._resolve_source_fits_path(fname)
        if src is None or not src.exists():
            return None
        return _build_source_signature(src, use_cropped=bool(self.use_cropped))

    def _detect_meta_matches(self, payload: dict, sig_now: dict, meta_path: Path) -> bool:
        if not isinstance(payload, dict) or not isinstance(sig_now, dict):
            return False
        try:
            schema = int(payload.get("cache_schema", 0) or 0)
        except Exception:
            schema = 0
        if schema < 2:
            return False
        if not _source_signature_detection_compatible(payload, sig_now):
            return False
        return True

    def _legacy_detect_cache_allowed(self, marker_path: Path) -> bool:
        try:
            marker_mtime = int(marker_path.stat().st_mtime_ns)
        except Exception:
            return False
        if self.use_cropped:
            rect_path = crop_rect_path(self.result_dir)
            if rect_path.exists():
                try:
                    rect_mtime = int(rect_path.stat().st_mtime_ns)
                    if marker_mtime < rect_mtime:
                        return False
                except Exception:
                    return False
        return True

    def _load_fwhm_for_frame(self, fname: str):
        sig_now = self._compatible_detect_signature(fname)
        if sig_now is None:
            return float(getattr(self.params.P, "fwhm_seed_px", 6.0)), np.nan
        candidates = [
            self.cache_dir / f"detect_{fname}.json",
            step4_dir(self.result_dir) / f"detect_{fname}.json",
        ]
        candidates = [p for p in candidates if p.exists()]
        candidates.sort(key=lambda p: p.stat().st_mtime_ns if p.exists() else 0, reverse=True)
        for meta_json in candidates:
            try:
                meta = json.loads(meta_json.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not self._detect_meta_matches(meta, sig_now, meta_json):
                continue
            try:
                fpx = float(meta.get("fwhm_med_rad_px", meta.get("fwhm_med_px", np.nan)))
                farc = float(meta.get("fwhm_med_rad_arcsec", meta.get("fwhm_med_arc", np.nan)))
                return fpx, farc
            except Exception:
                continue
        # Backward compatibility: legacy detect cache (schema<2) can be used
        # when it is newer than the current crop selection marker.
        for meta_json in candidates:
            try:
                meta = json.loads(meta_json.read_text(encoding="utf-8"))
            except Exception:
                continue
            try:
                schema = int(meta.get("cache_schema", 0) or 0)
            except Exception:
                schema = 0
            if schema >= 2:
                continue
            if not self._legacy_detect_cache_allowed(meta_json):
                continue
            try:
                fpx = float(meta.get("fwhm_med_rad_px", meta.get("fwhm_med_px", np.nan)))
                farc = float(meta.get("fwhm_med_rad_arcsec", meta.get("fwhm_med_arc", np.nan)))
                return fpx, farc
            except Exception:
                continue
        return float(getattr(self.params.P, "fwhm_seed_px", 6.0)), np.nan

    def _load_detect_xy(self, fname: str) -> np.ndarray:
        sig_now = self._compatible_detect_signature(fname)
        if sig_now is None:
            return np.zeros((0, 2), float)
        candidates = [
            self.cache_dir / f"detect_{fname}.csv",
            step4_dir(self.result_dir) / f"detect_{fname}.csv",
        ]
        for path in candidates:
            if not path.exists():
                continue
            meta_path = path.with_suffix(".json")
            if not meta_path.exists():
                continue
            try:
                payload = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not self._detect_meta_matches(payload, sig_now, meta_path):
                continue
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            if not {"x", "y"} <= set(df.columns):
                continue
            xy = df[["x", "y"]].to_numpy(float)
            xy = xy[np.isfinite(xy).all(axis=1)]
            return xy
        # Backward compatibility for legacy detect cache.
        for csv_path in candidates:
            if not csv_path.exists():
                continue
            meta_path = csv_path.with_suffix(".json")
            schema = 0
            if meta_path.exists():
                try:
                    payload = json.loads(meta_path.read_text(encoding="utf-8"))
                    schema = int(payload.get("cache_schema", 0) or 0)
                except Exception:
                    continue
            if schema >= 2:
                continue
            marker_path = meta_path if meta_path.exists() else csv_path
            if not self._legacy_detect_cache_allowed(marker_path):
                continue
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue
            if not {"x", "y"} <= set(df.columns):
                continue
            xy = df[["x", "y"]].to_numpy(float)
            xy = xy[np.isfinite(xy).all(axis=1)]
            return xy
        return np.zeros((0, 2), float)

    def _empty_wcs_qc_metrics(self, n_detect: int = 0) -> dict:
        return {
            "n_detect": int(max(0, n_detect)),
            "n_catalog_in_fov": 0,
            "n_match": 0,
            "n_inlier": 0,
            "match_rate": np.nan,
            "match_rate_cat": np.nan,
            "match_rate_eff": np.nan,
            "match_radius_arcsec": np.nan,
            "match_radius_px": np.nan,
            "dx_med_px": np.nan,
            "dy_med_px": np.nan,
            "resid_med_px": np.nan,
            "resid_mad_px": np.nan,
            "resid_peak_px": np.nan,
            "resid_p99_px": np.nan,
            "rms_px": np.nan,
            "inlier_rate": np.nan,
            "resid_vs_radius_slope": np.nan,
            "edge_resid_ratio": np.nan,
            "center_offset_arcsec": np.nan,
            "pix_scale_input_arcsec": np.nan,
            "pix_scale_fit_arcsec": np.nan,
            "scale_delta_pct": np.nan,
        }

    def _compute_wcs_qc_metrics(
        self,
        *,
        w: WCS | None,
        det_xy: np.ndarray,
        nx: int,
        ny: int,
        gaia_ra_deg: np.ndarray,
        gaia_dec_deg: np.ndarray,
        pix_input_arcsec: float,
        pix_fit_arcsec: float,
        center_coord: SkyCoord | None,
    ) -> dict:
        out = self._empty_wcs_qc_metrics(n_detect=len(det_xy))
        if np.isfinite(pix_input_arcsec):
            out["pix_scale_input_arcsec"] = float(pix_input_arcsec)
        if np.isfinite(pix_fit_arcsec):
            out["pix_scale_fit_arcsec"] = float(pix_fit_arcsec)
        if np.isfinite(pix_input_arcsec) and pix_input_arcsec > 0 and np.isfinite(pix_fit_arcsec):
            out["scale_delta_pct"] = float((pix_fit_arcsec - pix_input_arcsec) / pix_input_arcsec * 100.0)

        if w is not None and w.has_celestial and center_coord is not None:
            try:
                c_ra, c_dec = self._wcs_center_coords(w, nx, ny)
                if np.isfinite(c_ra) and np.isfinite(c_dec):
                    c_sky = SkyCoord(c_ra * u.deg, c_dec * u.deg, frame="icrs")
                    out["center_offset_arcsec"] = float(c_sky.separation(center_coord).arcsec)
            except Exception:
                pass

        if w is None or (not w.has_celestial):
            return out
        if len(det_xy) == 0:
            return out
        if gaia_ra_deg.size == 0 or gaia_dec_deg.size == 0:
            return out

        try:
            xg, yg = w.celestial.all_world2pix(gaia_ra_deg, gaia_dec_deg, 0)
            xg = np.asarray(xg, float)
            yg = np.asarray(yg, float)
        except Exception:
            return out

        ok_g = (
            np.isfinite(xg)
            & np.isfinite(yg)
            & (xg >= 0.0)
            & (xg < float(nx))
            & (yg >= 0.0)
            & (yg < float(ny))
        )
        if not np.any(ok_g):
            return out

        gaia_xy = np.column_stack((xg[ok_g], yg[ok_g]))
        out["n_catalog_in_fov"] = int(len(gaia_xy))

        pix_use = pix_fit_arcsec if np.isfinite(pix_fit_arcsec) and pix_fit_arcsec > 0 else pix_input_arcsec
        match_r_arcsec = float(getattr(self.params.P, "wcs_qc_match_radius_arcsec", 2.0))
        if not np.isfinite(match_r_arcsec) or match_r_arcsec <= 0:
            match_r_arcsec = 2.0
        if np.isfinite(pix_use) and pix_use > 0:
            match_r_px = float(match_r_arcsec / pix_use)
        else:
            match_r_px = float(getattr(self.params.P, "wcs_qc_match_radius_px", 2.5))
        match_r_px = float(np.clip(match_r_px, 1.0, 25.0))
        out["match_radius_arcsec"] = float(match_r_arcsec)
        out["match_radius_px"] = float(match_r_px)

        tree = KDTree(gaia_xy)
        d, j = tree.query(det_xy, k=1)
        d = np.asarray(d, float)
        j = np.asarray(j, int)
        ok = np.isfinite(d) & (d <= match_r_px) & (j >= 0) & (j < len(gaia_xy))
        if not np.any(ok):
            return out

        det_candidates = np.where(ok)[0]
        order = np.argsort(d[det_candidates])
        used_gaia = set()
        keep_det = []
        keep_gaia = []
        for ord_idx in order:
            det_i = int(det_candidates[ord_idx])
            gaia_i = int(j[det_i])
            if gaia_i in used_gaia:
                continue
            used_gaia.add(gaia_i)
            keep_det.append(det_i)
            keep_gaia.append(gaia_i)
        if not keep_det:
            return out

        det_keep = np.asarray(keep_det, dtype=int)
        gaia_keep = np.asarray(keep_gaia, dtype=int)
        dx = det_xy[det_keep, 0] - gaia_xy[gaia_keep, 0]
        dy = det_xy[det_keep, 1] - gaia_xy[gaia_keep, 1]
        r = np.hypot(dx, dy)
        finite_r = np.isfinite(r)
        if not np.any(finite_r):
            return out
        if not np.all(finite_r):
            dx = dx[finite_r]
            dy = dy[finite_r]
            r = r[finite_r]
            gaia_keep = gaia_keep[finite_r]

        n_match = int(len(r))
        out["n_match"] = n_match
        out["match_rate"] = float(n_match / max(int(len(det_xy)), 1))
        out["match_rate_cat"] = float(n_match / max(int(len(gaia_xy)), 1))
        out["match_rate_eff"] = float(max(out["match_rate"], out["match_rate_cat"]))
        if n_match == 0:
            return out

        dx_med = float(np.nanmedian(dx)) if len(dx) else np.nan
        dy_med = float(np.nanmedian(dy)) if len(dy) else np.nan
        out["dx_med_px"] = dx_med
        out["dy_med_px"] = dy_med

        resid_med = float(np.nanmedian(r))
        resid_mad = float(1.4826 * np.nanmedian(np.abs(r - resid_med)))
        resid_p99 = float(np.nanpercentile(r, 99))
        out["resid_med_px"] = resid_med
        out["resid_mad_px"] = resid_mad
        out["resid_p99_px"] = resid_p99
        out["resid_peak_px"] = resid_p99

        clip_sigma = float(getattr(self.params.P, "wcs_qc_clip_sigma", 3.0))
        if not np.isfinite(clip_sigma) or clip_sigma <= 0:
            clip_sigma = 3.0
        if np.isfinite(resid_mad) and resid_mad > 0:
            inlier = np.abs(r - resid_med) <= clip_sigma * resid_mad
        else:
            resid_std = float(np.nanstd(r))
            if np.isfinite(resid_std) and resid_std > 0:
                inlier = np.abs(r - float(np.nanmean(r))) <= clip_sigma * resid_std
            else:
                inlier = np.ones(len(r), dtype=bool)
        n_inlier = int(np.sum(inlier))
        r_in = r[inlier] if n_inlier > 0 else r
        out["n_inlier"] = n_inlier
        out["inlier_rate"] = float(n_inlier / max(n_match, 1))
        out["rms_px"] = float(np.sqrt(np.nanmean(r_in ** 2))) if len(r_in) else np.nan

        if len(det_keep) >= 8:
            cx = float(nx) / 2.0
            cy = float(ny) / 2.0
            rr = np.hypot(gaia_xy[gaia_keep, 0] - cx, gaia_xy[gaia_keep, 1] - cy)
            max_rr = max(float(np.hypot(max(cx, 1.0), max(cy, 1.0))), 1.0)
            rho = rr / max_rr
            if np.isfinite(np.nanstd(rho)) and float(np.nanstd(rho)) > 1e-6:
                try:
                    out["resid_vs_radius_slope"] = float(np.polyfit(rho, r, 1)[0])
                except Exception:
                    out["resid_vs_radius_slope"] = np.nan
            core = r[rho <= 0.4]
            edge = r[rho >= 0.8]
            if len(core) >= 3 and len(edge) >= 3:
                core_med = float(np.nanmedian(core))
                if np.isfinite(core_med) and core_med > 1e-9:
                    out["edge_resid_ratio"] = float(np.nanmedian(edge) / core_med)

        return out

    def _evaluate_wcs_qc_pass(self, metrics: dict, *, wcs_ok: bool) -> tuple[bool, list[str]]:
        def _num(key: str) -> float:
            try:
                return float(metrics.get(key, np.nan))
            except Exception:
                return np.nan

        reasons: list[str] = []

        require_wcs_ok = bool(getattr(self.params.P, "wcs_qc_require_wcs_ok", True))
        if require_wcs_ok and not wcs_ok:
            reasons.append("wcs_fail")

        n_detect = int(metrics.get("n_detect", 0) or 0)
        n_match = int(metrics.get("n_match", 0) or 0)
        if n_detect <= 0:
            reasons.append("no_detect_data")

        min_match_n = int(getattr(self.params.P, "wcs_qc_min_match_n", 20))
        if min_match_n > 0 and n_match < min_match_n:
            reasons.append("low_match_n")

        min_match_rate = float(getattr(self.params.P, "wcs_qc_min_match_rate", 0.20))
        mrate_det = _num("match_rate")
        mrate_cat = _num("match_rate_cat")
        mrate_eff = _num("match_rate_eff")
        if not np.isfinite(mrate_eff):
            if np.isfinite(mrate_det) or np.isfinite(mrate_cat):
                mrate_eff = float(np.nanmax([mrate_det, mrate_cat]))
        if np.isfinite(min_match_rate) and min_match_rate > 0:
            if (not np.isfinite(mrate_eff)) or (mrate_eff < min_match_rate):
                reasons.append("low_match_rate")

        max_rms_px = float(getattr(self.params.P, "wcs_qc_max_rms_px", 2.5))
        if n_match > 0 and np.isfinite(max_rms_px) and max_rms_px > 0:
            rms_px = _num("rms_px")
            if (not np.isfinite(rms_px)) or (rms_px > max_rms_px):
                reasons.append("high_rms")

        max_p99_px = float(getattr(self.params.P, "wcs_qc_max_p99_px", 5.0))
        if n_match > 0 and np.isfinite(max_p99_px) and max_p99_px > 0:
            p99_px = _num("resid_p99_px")
            if (not np.isfinite(p99_px)) or (p99_px > max_p99_px):
                reasons.append("high_p99")

        min_inlier_rate = float(getattr(self.params.P, "wcs_qc_min_inlier_rate", 0.50))
        if n_match > 0 and np.isfinite(min_inlier_rate) and min_inlier_rate > 0:
            inlier_rate = _num("inlier_rate")
            if (not np.isfinite(inlier_rate)) or (inlier_rate < min_inlier_rate):
                reasons.append("low_inlier")

        max_edge_ratio = float(getattr(self.params.P, "wcs_qc_max_edge_ratio", 0.0))
        edge_ratio = _num("edge_resid_ratio")
        if np.isfinite(max_edge_ratio) and max_edge_ratio > 0:
            if np.isfinite(edge_ratio) and edge_ratio > max_edge_ratio:
                reasons.append("edge_resid")

        max_center_off = float(getattr(self.params.P, "wcs_qc_max_center_offset_arcsec", 0.0))
        center_off = _num("center_offset_arcsec")
        if np.isfinite(max_center_off) and max_center_off > 0:
            if (not np.isfinite(center_off)) or (center_off > max_center_off):
                reasons.append("center_offset")

        return len(reasons) == 0, reasons

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
            return _coerce_source_id_int64(tab.to_pandas())
        except Exception:
            return None

    def _query_gaia(self, center: SkyCoord, radius_deg: float, mag_max: float):
        if not _HAS_GAIA:
            raise RuntimeError("astroquery.gaia not available")
        if self._stop_requested:
            raise RuntimeError("stopped")
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
        if self._stop_requested:
            raise RuntimeError("stopped")
        try:
            job = Gaia.launch_job_async(adql, dump_to_file=False)
            tab = job.get_results()
        except Exception as e:
            raise RuntimeError(f"Gaia TAP async query failed: {_exc_brief(e)}") from e
        if "phot_g_mean_mag" in tab.colnames and np.isfinite(mag_max):
            tab = tab[np.isfinite(tab["phot_g_mean_mag"]) & (tab["phot_g_mean_mag"] <= mag_max)]
        return _coerce_source_id_int64(tab.to_pandas())

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

        def _cache_mag_max(df_in: pd.DataFrame, meta_in: dict | None) -> float:
            try:
                if isinstance(meta_in, dict) and ("mag_max" in meta_in):
                    v = float(meta_in.get("mag_max"))
                    if np.isfinite(v):
                        return v
            except Exception:
                pass
            try:
                if "phot_g_mean_mag" in df_in.columns:
                    g = pd.to_numeric(df_in["phot_g_mean_mag"], errors="coerce")
                    if g.notna().any():
                        return float(g.max())
            except Exception:
                pass
            return np.nan

        def _filter_cache_by_mag(df_in: pd.DataFrame) -> pd.DataFrame:
            if not np.isfinite(mag_max):
                return df_in
            if "phot_g_mean_mag" not in df_in.columns:
                return df_in
            g = pd.to_numeric(df_in["phot_g_mean_mag"], errors="coerce")
            keep = g.notna() & (g <= float(mag_max))
            return df_in.loc[keep].copy()

        def _cache_covers_field(df_in: pd.DataFrame, ctr: SkyCoord, rad_deg: float) -> bool:
            # Guard against stale/misaligned Gaia caches that cover only part of the requested field.
            try:
                ra = pd.to_numeric(df_in.get("ra"), errors="coerce")
                dec = pd.to_numeric(df_in.get("dec"), errors="coerce")
            except Exception:
                return False
            m = ra.notna() & dec.notna()
            n = int(m.sum())
            if n <= 0:
                return False
            if n < 50:
                return True
            ra_v = ra[m].to_numpy(float)
            dec_v = dec[m].to_numpy(float)
            cos_dec = float(np.cos(np.deg2rad(float(ctr.dec.deg))))
            if not np.isfinite(cos_dec) or cos_dec <= 0:
                cos_dec = 1.0
            dx = (ra_v - float(ctr.ra.deg)) * cos_dec
            dy = dec_v - float(ctr.dec.deg)
            if dx.size == 0 or dy.size == 0:
                return False
            side_frac = 0.60 if n >= 200 else 0.45
            need = float(rad_deg) * side_frac
            min_x = float(np.nanmin(dx))
            max_x = float(np.nanmax(dx))
            min_y = float(np.nanmin(dy))
            max_y = float(np.nanmax(dy))
            return bool(
                np.isfinite(min_x)
                and np.isfinite(max_x)
                and np.isfinite(min_y)
                and np.isfinite(max_y)
                and (min_x <= -need)
                and (max_x >= need)
                and (min_y <= -need)
                and (max_y >= need)
            )

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
                cached_mag_max = _cache_mag_max(df_cache, meta)
                # 중심 좌표가 0.1도 이내이고 반경이 비슷하면 캐시 사용
                dist_deg = np.hypot(center.ra.deg - cached_ra, center.dec.deg - cached_dec)
                same_field = bool(dist_deg < 0.03 and cached_radius >= radius_deg * 0.9)
                mag_ok = (not np.isfinite(mag_max)) or (not np.isfinite(cached_mag_max)) or (cached_mag_max + 1e-6 >= mag_max)
                coverage_ok = _cache_covers_field(df_cache, center, radius_deg)
                if same_field and mag_ok and coverage_ok:
                    cache_valid = True
                elif same_field and mag_ok and (not coverage_ok):
                    self._log("[Gaia] cache rejected: field coverage mismatch; re-querying Gaia.")
            except Exception:
                pass
        elif df_cache is not None:
            # 메타 파일이 없으면 일단 캐시 사용 (이전 버전 호환)
            cached_mag_max = _cache_mag_max(df_cache, None)
            mag_ok = (not np.isfinite(mag_max)) or (not np.isfinite(cached_mag_max)) or (cached_mag_max + 1e-6 >= mag_max)
            coverage_ok = _cache_covers_field(df_cache, center, radius_deg)
            if mag_ok and coverage_ok:
                cache_valid = True
            elif mag_ok and (not coverage_ok):
                self._log("[Gaia] legacy cache rejected: field coverage mismatch; re-querying Gaia.")

        if cache_valid and df_cache is not None:
            return _filter_cache_by_mag(df_cache), "cache"
        if not _HAS_GAIA:
            if allow_no_cache:
                return pd.DataFrame(), "no_gaia_module"
            raise RuntimeError("astroquery.gaia not available and no cache")

        last_err = None
        for att in range(1, max(1, retry) + 1):
            if self._stop_requested:
                raise RuntimeError("stopped")
            try:
                df = self._query_gaia(center, radius_deg, mag_max)
                df.columns = [c.lower() for c in df.columns]
                try:
                    df_out = df.copy()
                    if "source_id" in df_out.columns:
                        df_out = _coerce_source_id_int64(df_out)
                    Table.from_pandas(df_out).write(cache_path, format="ascii.ecsv", overwrite=True)
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
                if self._stop_requested:
                    raise RuntimeError("stopped")
                last_err = e
                if att < retry:
                    slept = 0.0
                    while slept < backoff_s:
                        if self._stop_requested:
                            raise RuntimeError("stopped")
                        dt = min(0.25, backoff_s - slept)
                        time.sleep(dt)
                        slept += dt

        df_cache = self._load_gaia_cache_if_ok(cache_path)
        if df_cache is not None:
            cached_mag_max = _cache_mag_max(df_cache, None)
            mag_ok = (not np.isfinite(mag_max)) or (not np.isfinite(cached_mag_max)) or (cached_mag_max + 1e-6 >= mag_max)
            if mag_ok:
                return _filter_cache_by_mag(df_cache), "cache(after_fail)"
            if allow_no_cache:
                if last_err is None:
                    return pd.DataFrame(), f"cache_too_shallow:{cached_mag_max:.2f}<{mag_max:.2f}"
                return pd.DataFrame(), (
                    f"cache_too_shallow:{cached_mag_max:.2f}<{mag_max:.2f};"
                    f"query_failed:{_exc_brief(last_err, limit=140)}"
                )
            if last_err is None:
                raise RuntimeError(
                    f"Gaia cache too shallow (cache_mag_max={cached_mag_max:.2f}, required={mag_max:.2f})"
                )
            raise RuntimeError(
                f"Gaia query failed and cache too shallow "
                f"(cache_mag_max={cached_mag_max:.2f}, required={mag_max:.2f}): {last_err}"
            )
        if allow_no_cache:
            if last_err is None:
                return pd.DataFrame(), "fail_no_cache:unknown"
            return pd.DataFrame(), f"fail_no_cache:{_exc_brief(last_err, limit=180)}"
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
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stdout_s = ""
            stderr_s = ""
            while True:
                if self._stop_requested:
                    try:
                        proc.terminate()
                        proc.wait(timeout=2.0)
                    except Exception:
                        try:
                            proc.kill()
                        except Exception:
                            pass
                    try:
                        out_s, err_s = proc.communicate(timeout=0.5)
                        stdout_s = out_s or ""
                        stderr_s = err_s or ""
                    except Exception:
                        pass
                    err_msg = "stopped"
                    err_tail = _tail_text(stderr_s, limit=1000, max_lines=10)
                    if err_tail:
                        err_msg = f"stopped | {err_tail}"
                    return False, -997, time.time() - start, stdout_s, err_msg, cmd

                elapsed = time.time() - start
                if elapsed >= timeout_s:
                    try:
                        proc.terminate()
                        proc.wait(timeout=2.0)
                    except Exception:
                        try:
                            proc.kill()
                        except Exception:
                            pass
                    try:
                        out_s, err_s = proc.communicate(timeout=0.5)
                        stdout_s = out_s or ""
                        stderr_s = err_s or ""
                    except Exception:
                        pass
                    err_msg = "timeout"
                    err_tail = _tail_text(stderr_s, limit=1000, max_lines=10)
                    if err_tail:
                        err_msg = f"timeout | {err_tail}"
                    return False, -999, timeout_s, stdout_s, err_msg, cmd

                try:
                    out_s, err_s = proc.communicate(timeout=0.2)
                    stdout_s = out_s or ""
                    stderr_s = err_s or ""
                    break
                except subprocess.TimeoutExpired:
                    continue

            dt = time.time() - start
            rc = int(proc.returncode if proc.returncode is not None else -998)
            ok = (rc == 0)
            return ok, rc, dt, stdout_s, stderr_s, cmd
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
                ts = time.strftime("%Y-%m-%d %H:%M:%S")
                line = f"{ts} {msg}"
                try:
                    with open(log_path, "a", encoding="utf-8") as fh:
                        fh.write(line + "\n")
                except Exception:
                    pass

            def log_cmd_failure(tag, fname, reason, cmd=None, stdout=None, stderr=None):
                L(f"{fname}: {tag} fail reason={reason}")
                if cmd:
                    L(f"{fname}: {tag} cmd={' '.join(str(c) for c in cmd)}")
                out_tail = _tail_text(stdout, limit=1600, max_lines=12)
                err_tail = _tail_text(stderr, limit=1600, max_lines=12)
                if out_tail:
                    L(f"{fname}: {tag} stdout_tail={out_tail}")
                if err_tail:
                    L(f"{fname}: {tag} stderr_tail={err_tail}")

            L("=" * 60)
            L(f"[WCS] start files={len(files)} use_cropped={self.use_cropped} cache_dir={self.cache_dir}")
            L(f"[WCS] astap_timeout_s={astap_timeout} astap_radius_deg={astap_radius} astap_db={astap_db or 'default'} astap_fov_fudge={astap_fov_fudge}")
            L(
                f"[WCS] astnet_local_enable={astnet_local_enable} use_wsl={astnet_use_wsl} "
                f"timeout_s={astnet_timeout_s} downsample={astnet_downsample} "
                f"scale=[{astnet_scale_low:.5f},{astnet_scale_high:.5f}] radius_deg={astnet_radius_deg}"
            )

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
            if self._stop_requested:
                self.finished.emit({"stopped": True, "total": 0, "ok": 0, "wcs_qc_pass": 0})
                return
            gaia_fudge = float(getattr(self.params.P, "gaia_radius_fudge", 1.35))
            gaia_mag_max = float(getattr(self.params.P, "gaia_mag_max", 18.0))
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
            L(
                f"[WCS] sample={sample} shape={nx0}x{ny0} pix_arcsec={pix_arc:.5f} "
                f"fov_w={fov_w:.5f}deg fov_h={fov_h:.5f}deg diag={diag_deg:.5f}deg "
                f"gaia_r={gaia_r:.5f}deg"
            )
            gaia_df, gaia_src = self._load_or_query_gaia(center_coord, gaia_r)
            if self._stop_requested:
                self.finished.emit({"stopped": True, "total": 0, "ok": 0, "wcs_qc_pass": 0})
                return
            L(
                f"[Gaia] center=({center_coord.ra.deg:.6f},{center_coord.dec.deg:.6f}) "
                f"r={gaia_r:.4f}deg mag_max={gaia_mag_max:.2f} source={gaia_src} N={len(gaia_df)}"
            )
            gaia_ra_vals = np.array([], dtype=float)
            gaia_dec_vals = np.array([], dtype=float)
            if isinstance(gaia_df, pd.DataFrame) and (not gaia_df.empty) and {"ra", "dec"} <= set(gaia_df.columns):
                gaia_ra_vals = pd.to_numeric(gaia_df["ra"], errors="coerce").to_numpy(float)
                gaia_dec_vals = pd.to_numeric(gaia_df["dec"], errors="coerce").to_numpy(float)
                ok_rd = np.isfinite(gaia_ra_vals) & np.isfinite(gaia_dec_vals)
                gaia_ra_vals = gaia_ra_vals[ok_rd]
                gaia_dec_vals = gaia_dec_vals[ok_rd]

            def solve_one(filename):
                if self._stop_requested:
                    return filename, None

                if self.use_cropped:
                    fits_path = step2_cropped_dir(self.result_dir) / filename
                else:
                    fits_path = self.params.get_file_path(filename)
                source_sig = _build_source_signature(Path(fits_path), use_cropped=bool(self.use_cropped))

                status = "fail"
                fail_reason = ""
                pix_fit = np.nan
                wcs_ok = False
                refine_note = ""
                resid_med = np.nan
                resid_max = np.nan
                match_n = 0
                rc = -998
                dt_astap = 0.0
                ok_astap = False
                astap_cmd = []
                astap_stdout = ""
                astap_stderr = ""
                astnet_reason = ""
                det_xy = self._load_detect_xy(filename)
                qc_metrics = self._empty_wcs_qc_metrics(n_detect=len(det_xy))
                qc_metrics["pix_scale_input_arcsec"] = float(pix_arc) if np.isfinite(pix_arc) else np.nan

                with fits.open(fits_path, memmap=False, ignore_missing_simple=True) as hdul:
                    hdr = hdul[0].header
                    data = hdul[0].data
                    if data is None:
                        return filename, {
                            "fname": filename,
                            "ok": False,
                            "status": "data_none",
                            "wcs_ok": False,
                            "pix_fit": None,
                            "elapsed": 0.0,
                            "fail_reason": "data_none",
                            "refine": "",
                            "solver": "astap",
                            "wcs_qc_pass": False,
                            "wcs_qc_reason": "data_none,wcs_fail",
                            **source_sig,
                            **qc_metrics,
                        }
                    ny, nx = data.shape

                if self._stop_requested:
                    return filename, None
                fov_w_deg = (nx * pix_arc) / 3600.0 * astap_fov_fudge
                fov_h_deg = (ny * pix_arc) / 3600.0 * astap_fov_fudge
                fov_deg = float(max(fov_w_deg, fov_h_deg))

                ok_astap, rc, dt_astap, astap_stdout, astap_stderr, astap_cmd = self._run_astap(
                    fits_path, fov_deg=fov_deg, radius_deg=astap_radius, timeout_s=astap_timeout
                )
                if self._stop_requested:
                    return filename, None
                astap_cmd_str = " ".join(str(c) for c in astap_cmd)
                if not ok_astap:
                    if rc == -999 or "timeout" in str(astap_stderr).lower():
                        fail_reason = "astap_timeout"
                    elif rc == -997 or "stopped" in str(astap_stderr).lower():
                        fail_reason = "astap_stopped"
                    else:
                        fail_reason = f"astap_rc_{rc}"
                    log_cmd_failure(
                        "ASTAP",
                        filename,
                        f"{fail_reason}, dt={dt_astap:.1f}s",
                        cmd=astap_cmd,
                        stdout=astap_stdout,
                        stderr=astap_stderr,
                    )
                    if not astnet_local_enable:
                        qc_metrics["pix_scale_fit_arcsec"] = np.nan
                        return filename, {
                            "fname": filename,
                            "ok": False,
                            "status": f"astap_fail rc={rc}",
                            "wcs_ok": False,
                            "pix_fit": pix_fit,
                            "elapsed": float(dt_astap),
                            "fail_reason": fail_reason,
                            "refine": refine_note,
                            "solver": "astap",
                            "astap_cmd": astap_cmd_str,
                            "astap_stdout": _tail_text(astap_stdout, limit=2000, max_lines=12),
                            "astap_stderr": _tail_text(astap_stderr, limit=2000, max_lines=12),
                            "wcs_qc_pass": False,
                            "wcs_qc_reason": "astap_fail,wcs_fail",
                            **source_sig,
                            **qc_metrics,
                        }

                astnet_ok = False
                astnet_dt = np.nan
                astnet_stdout = ""
                astnet_stderr = ""
                astnet_cmd = []
                astnet_cmd_str = ""
                astnet_new_path = None
                solver = "astap"
                used_elapsed = float(dt_astap)

                # WCS를 FITS 파일에 저장 (writeto 사용 - Windows 호환성)
                with fits.open(fits_path, memmap=False, ignore_missing_simple=True) as hdul:
                    hdr = hdul[0].header
                    data = hdul[0].data
                    w_final = None
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
                        astnet_cmd_str = " ".join(str(c) for c in astnet_cmd)
                        if astnet_ok:
                            L(f"[ASTNET_WSL] {filename} success dt={astnet_dt:.1f}s")
                            if astnet_stderr:
                                L(f"{filename}: ASTNET_WSL stderr_tail={_tail_text(astnet_stderr, limit=600, max_lines=6)}")
                        else:
                            if "timeout" in str(astnet_stderr).lower():
                                astnet_reason = "astnet_timeout"
                            elif "stopped" in str(astnet_stderr).lower():
                                astnet_reason = "astnet_stopped"
                            elif (
                                "wsl_path_unavailable" in str(astnet_stderr).lower()
                                or "wsl_not_found" in str(astnet_stderr).lower()
                                or "wsl_input_missing" in str(astnet_stderr).lower()
                                or "input_missing:" in str(astnet_stderr).lower()
                                or (
                                    "cannot open `" in str(astnet_stderr).lower()
                                    and "no such file or directory" in str(astnet_stderr).lower()
                                )
                            ):
                                astnet_reason = "astnet_wsl_path_unavailable"
                            elif "cache_hit" in str(astnet_stdout):
                                astnet_reason = "astnet_cache_miss_solved_marker_missing"
                            else:
                                astnet_reason = "astnet_fail_no_solution"
                            if not fail_reason:
                                fail_reason = astnet_reason
                            log_cmd_failure(
                                "ASTNET_WSL",
                                filename,
                                f"{astnet_reason}, dt={astnet_dt:.1f}s",
                                cmd=astnet_cmd,
                                stdout=astnet_stdout,
                                stderr=astnet_stderr,
                            )

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
                        if refine_enable and gaia_df is not None and len(gaia_df) > 0 and len(det_xy) > 0:
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
                        qc_metrics = self._compute_wcs_qc_metrics(
                            w=w_final,
                            det_xy=det_xy,
                            nx=int(hdr.get("NAXIS1", nx)),
                            ny=int(hdr.get("NAXIS2", ny)),
                            gaia_ra_deg=gaia_ra_vals,
                            gaia_dec_deg=gaia_dec_vals,
                            pix_input_arcsec=float(pix_arc),
                            pix_fit_arcsec=float(pix_fit) if np.isfinite(pix_fit) else np.nan,
                            center_coord=center_coord,
                        )

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
                        if not fail_reason:
                            if not ok_astap:
                                fail_reason = f"astap_rc_{rc}"
                            elif astnet_local_enable and astnet_reason:
                                fail_reason = astnet_reason
                            elif astnet_local_enable:
                                fail_reason = "astnet_ran_but_wcs_missing"
                            else:
                                fail_reason = "wcs_keywords_missing_after_astap"
                        L(
                            f"{filename}: final_wcs_fail status={status} fail_reason={fail_reason} "
                            f"solver={solver} astap_ok={ok_astap} astnet_ok={astnet_ok}"
                        )
                        # Set defaults for failed WCS
                        wcs_rot_deg = np.nan
                        center_ra = np.nan
                        center_dec = np.nan
                        sip_order = 0

                    qc_pass, qc_reasons = self._evaluate_wcs_qc_pass(qc_metrics, wcs_ok=bool(wcs_ok))
                    qc_reason = ",".join(qc_reasons)

                if self._stop_requested:
                    return filename, None
                # writeto로 확실하게 저장 (Windows 호환)
                fits.writeto(fits_path, data, hdr, overwrite=True)
                if self._stop_requested:
                    return filename, None

                meta = {
                    "fname": filename,
                    "ok": bool(wcs_ok),
                    "wcs_ok": bool(wcs_ok),
                    "status": status,
                    "fail_reason": fail_reason,
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
                    "astap_cmd": astap_cmd_str,
                    "astap_stdout": _tail_text(astap_stdout, limit=2000, max_lines=12),
                    "astap_stderr": _tail_text(astap_stderr, limit=2000, max_lines=12),
                    "astnet_wsl_ok": bool(astnet_ok),
                    "astnet_wsl_elapsed": float(astnet_dt) if np.isfinite(astnet_dt) else None,
                    "astnet_wsl_fail_reason": astnet_reason,
                    "astnet_wsl_cmd": astnet_cmd_str,
                    "astnet_wsl_stdout": _tail_text(astnet_stdout, limit=2000, max_lines=12),
                    "astnet_wsl_stderr": _tail_text(astnet_stderr, limit=2000, max_lines=12),
                    "wcs_qc_pass": bool(qc_pass),
                    "wcs_qc_reason": qc_reason,
                    **source_sig,
                }
                meta.update(qc_metrics)
                rms_px_val = qc_metrics.get("rms_px", np.nan)
                try:
                    rms_px_val = float(rms_px_val)
                except Exception:
                    rms_px_val = np.nan
                rms_px_str = f"{rms_px_val:.3f}" if np.isfinite(rms_px_val) else "-"
                L(
                    f"{filename}: {status} pix_fit={pix_fit:.4f} dt={used_elapsed:.1f}s "
                    f"refine={refine_note or '-'} resid_med={resid_med if np.isfinite(resid_med) else '-'} "
                    f"match_n={match_n} wcs_qc={'PASS' if qc_pass else 'FAIL'} "
                    f"n_det={int(qc_metrics.get('n_detect', 0) or 0)} "
                    f"n_match={int(qc_metrics.get('n_match', 0) or 0)} "
                    f"rms_px={rms_px_str} "
                    f"reason={qc_reason or '-'} "
                    f"fail_reason={fail_reason or '-'}"
                )
                if self._stop_requested:
                    return filename, None
                (meta_dir / f"wcs_{filename}.json").write_text(
                    json.dumps(meta, indent=2), encoding="utf-8"
                )
                return filename, meta

            completed = 0
            _default_workers = get_parallel_workers(self.params)
            max_workers = int(getattr(self.params.P, "wcs_max_workers", _default_workers))
            with ThreadPoolExecutor(max_workers=max(1, max_workers)) as ex:
                futures = {ex.submit(solve_one, f): f for f in files}
                for fut in as_completed(futures):
                    if self._stop_requested:
                        for f_cancel in futures:
                            f_cancel.cancel()
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
                        L(f"{fname}: worker_exception={_exc_brief(e)}")
                        self.error.emit(fname, str(e))
                    self.progress.emit(completed, len(files), fname)

            # Save summary CSV
            try:
                df = pd.DataFrame(results)
                step5_out = step5_dir(self.result_dir)
                step5_out.mkdir(parents=True, exist_ok=True)
                df.to_csv(step5_out / "wcs_solve_summary.csv", index=False)
                qc_cols = [
                    "fname",
                    "status",
                    "fail_reason",
                    "ok",
                    "wcs_ok",
                    "solver",
                    "elapsed",
                    "source_path",
                    "source_use_cropped",
                    "source_size",
                    "source_mtime_ns",
                    "n_detect",
                    "n_catalog_in_fov",
                    "n_match",
                    "n_inlier",
                    "match_rate",
                    "match_rate_cat",
                    "match_rate_eff",
                    "match_radius_arcsec",
                    "match_radius_px",
                    "dx_med_px",
                    "dy_med_px",
                    "resid_med_px",
                    "resid_mad_px",
                    "resid_p99_px",
                    "resid_peak_px",
                    "rms_px",
                    "inlier_rate",
                    "resid_vs_radius_slope",
                    "edge_resid_ratio",
                    "pix_scale_input_arcsec",
                    "pix_scale_fit_arcsec",
                    "scale_delta_pct",
                    "wcs_rot_deg",
                    "center_ra_deg",
                    "center_dec_deg",
                    "center_offset_arcsec",
                    "sip_order",
                    "wcs_qc_pass",
                    "wcs_qc_reason",
                ]
                qc_df = df[[c for c in qc_cols if c in df.columns]].copy()
                if "fname" in qc_df.columns and "file" not in qc_df.columns:
                    qc_df = qc_df.rename(columns={"fname": "file"})
                qc_df.to_csv(step5_out / "frame_wcs_qc.csv", index=False)
            except Exception:
                pass

            n_qc_pass = sum(1 for r in results if bool(r.get("wcs_qc_pass", False)))
            summary = {
                "total": len(results),
                "ok": sum(1 for r in results if r.get("ok")),
                "wcs_qc_pass": int(n_qc_pass),
                "stopped": bool(self._stop_requested),
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
            return _coerce_source_id_int64(tab.to_pandas())
        except Exception:
            return None

    def _query_gaia(self, center: SkyCoord, radius_deg: float, mag_max: float):
        if not _HAS_GAIA:
            raise RuntimeError("astroquery.gaia not available")
        if self._stop_requested:
            raise RuntimeError("stopped")
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
        if self._stop_requested:
            raise RuntimeError("stopped")
        try:
            job = Gaia.launch_job_async(adql, dump_to_file=False)
            tab = job.get_results()
        except Exception as e:
            raise RuntimeError(f"Gaia TAP async query failed: {_exc_brief(e)}") from e
        if "phot_g_mean_mag" in tab.colnames and np.isfinite(mag_max):
            tab = tab[np.isfinite(tab["phot_g_mean_mag"]) & (tab["phot_g_mean_mag"] <= mag_max)]
        return _coerce_source_id_int64(tab.to_pandas())

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

        def _cache_mag_max(df_in: pd.DataFrame, meta_in: dict | None) -> float:
            try:
                if isinstance(meta_in, dict) and ("mag_max" in meta_in):
                    v = float(meta_in.get("mag_max"))
                    if np.isfinite(v):
                        return v
            except Exception:
                pass
            try:
                if "phot_g_mean_mag" in df_in.columns:
                    g = pd.to_numeric(df_in["phot_g_mean_mag"], errors="coerce")
                    if g.notna().any():
                        return float(g.max())
            except Exception:
                pass
            return np.nan

        def _filter_cache_by_mag(df_in: pd.DataFrame) -> pd.DataFrame:
            if not np.isfinite(mag_max):
                return df_in
            if "phot_g_mean_mag" not in df_in.columns:
                return df_in
            g = pd.to_numeric(df_in["phot_g_mean_mag"], errors="coerce")
            keep = g.notna() & (g <= float(mag_max))
            return df_in.loc[keep].copy()

        def _cache_covers_field(df_in: pd.DataFrame, ctr: SkyCoord, rad_deg: float) -> bool:
            # Guard against stale/misaligned Gaia caches that cover only part of the requested field.
            try:
                ra = pd.to_numeric(df_in.get("ra"), errors="coerce")
                dec = pd.to_numeric(df_in.get("dec"), errors="coerce")
            except Exception:
                return False
            m = ra.notna() & dec.notna()
            n = int(m.sum())
            if n <= 0:
                return False
            if n < 50:
                return True
            ra_v = ra[m].to_numpy(float)
            dec_v = dec[m].to_numpy(float)
            cos_dec = float(np.cos(np.deg2rad(float(ctr.dec.deg))))
            if not np.isfinite(cos_dec) or cos_dec <= 0:
                cos_dec = 1.0
            dx = (ra_v - float(ctr.ra.deg)) * cos_dec
            dy = dec_v - float(ctr.dec.deg)
            if dx.size == 0 or dy.size == 0:
                return False
            side_frac = 0.60 if n >= 200 else 0.45
            need = float(rad_deg) * side_frac
            min_x = float(np.nanmin(dx))
            max_x = float(np.nanmax(dx))
            min_y = float(np.nanmin(dy))
            max_y = float(np.nanmax(dy))
            return bool(
                np.isfinite(min_x)
                and np.isfinite(max_x)
                and np.isfinite(min_y)
                and np.isfinite(max_y)
                and (min_x <= -need)
                and (max_x >= need)
                and (min_y <= -need)
                and (max_y >= need)
            )

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
                cached_mag_max = _cache_mag_max(df_cache, meta)
                dist_deg = np.hypot(center.ra.deg - cached_ra, center.dec.deg - cached_dec)
                same_field = bool(dist_deg < 0.03 and cached_radius >= radius_deg * 0.9)
                mag_ok = (not np.isfinite(mag_max)) or (not np.isfinite(cached_mag_max)) or (cached_mag_max + 1e-6 >= mag_max)
                coverage_ok = _cache_covers_field(df_cache, center, radius_deg)
                if same_field and mag_ok and coverage_ok:
                    cache_valid = True
                elif same_field and mag_ok and (not coverage_ok):
                    self._log("[Gaia] cache rejected: field coverage mismatch; re-querying Gaia.")
            except Exception:
                pass
        elif df_cache is not None:
            cached_mag_max = _cache_mag_max(df_cache, None)
            mag_ok = (not np.isfinite(mag_max)) or (not np.isfinite(cached_mag_max)) or (cached_mag_max + 1e-6 >= mag_max)
            coverage_ok = _cache_covers_field(df_cache, center, radius_deg)
            if mag_ok and coverage_ok:
                cache_valid = True
            elif mag_ok and (not coverage_ok):
                self._log("[Gaia] legacy cache rejected: field coverage mismatch; re-querying Gaia.")

        if cache_valid and df_cache is not None:
            return _filter_cache_by_mag(df_cache), "cache"
        if not _HAS_GAIA:
            if allow_no_cache:
                return pd.DataFrame(), "no_gaia_module"
            raise RuntimeError("astroquery.gaia not available and no cache")

        last_err = None
        for att in range(1, max(1, retry) + 1):
            if self._stop_requested:
                raise RuntimeError("stopped")
            try:
                df = self._query_gaia(center, radius_deg, mag_max)
                df.columns = [c.lower() for c in df.columns]
                try:
                    df_out = df.copy()
                    if "source_id" in df_out.columns:
                        df_out = _coerce_source_id_int64(df_out)
                    Table.from_pandas(df_out).write(cache_path, format="ascii.ecsv", overwrite=True)
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
                if self._stop_requested:
                    raise RuntimeError("stopped")
                last_err = e
                if att < retry:
                    slept = 0.0
                    while slept < backoff_s:
                        if self._stop_requested:
                            raise RuntimeError("stopped")
                        dt = min(0.25, backoff_s - slept)
                        time.sleep(dt)
                        slept += dt

        df_cache = self._load_gaia_cache_if_ok(cache_path)
        if df_cache is not None:
            cached_mag_max = _cache_mag_max(df_cache, None)
            mag_ok = (not np.isfinite(mag_max)) or (not np.isfinite(cached_mag_max)) or (cached_mag_max + 1e-6 >= mag_max)
            if mag_ok:
                return _filter_cache_by_mag(df_cache), "cache(after_fail)"
            if allow_no_cache:
                if last_err is None:
                    return pd.DataFrame(), f"cache_too_shallow:{cached_mag_max:.2f}<{mag_max:.2f}"
                return pd.DataFrame(), (
                    f"cache_too_shallow:{cached_mag_max:.2f}<{mag_max:.2f};"
                    f"query_failed:{_exc_brief(last_err, limit=140)}"
                )
            if last_err is None:
                raise RuntimeError(
                    f"Gaia cache too shallow (cache_mag_max={cached_mag_max:.2f}, required={mag_max:.2f})"
                )
            raise RuntimeError(
                f"Gaia query failed and cache too shallow "
                f"(cache_mag_max={cached_mag_max:.2f}, required={mag_max:.2f}): {last_err}"
            )
        if allow_no_cache:
            if last_err is None:
                return pd.DataFrame(), "fail_no_cache:unknown"
            return pd.DataFrame(), f"fail_no_cache:{_exc_brief(last_err, limit=180)}"
        raise RuntimeError(f"Gaia query failed: {last_err}")

    def _resolve_source_fits_path(self, fname: str) -> Path | None:
        if self.use_cropped:
            cand = step2_cropped_dir(self.result_dir) / fname
            if cand.exists():
                return cand
            legacy = self.result_dir / "cropped" / fname
            if legacy.exists():
                return legacy
        try:
            orig = Path(self.params.get_file_path(fname))
            if orig.exists():
                return orig
        except Exception:
            pass
        return None

    def _compatible_detect_signature(self, fname: str) -> dict | None:
        src = self._resolve_source_fits_path(fname)
        if src is None or not src.exists():
            return None
        return _build_source_signature(src, use_cropped=bool(self.use_cropped))

    def _detect_meta_matches(self, payload: dict, sig_now: dict, meta_path: Path) -> bool:
        if not isinstance(payload, dict) or not isinstance(sig_now, dict):
            return False
        try:
            schema = int(payload.get("cache_schema", 0) or 0)
        except Exception:
            schema = 0
        if schema < 2:
            return False
        if not _source_signature_detection_compatible(payload, sig_now):
            return False
        return True

    def _legacy_detect_cache_allowed(self, marker_path: Path) -> bool:
        try:
            marker_mtime = int(marker_path.stat().st_mtime_ns)
        except Exception:
            return False
        if self.use_cropped:
            rect_path = crop_rect_path(self.result_dir)
            if rect_path.exists():
                try:
                    rect_mtime = int(rect_path.stat().st_mtime_ns)
                    if marker_mtime < rect_mtime:
                        return False
                except Exception:
                    return False
        return True

    def _load_fwhm_for_frame(self, fname: str):
        sig_now = self._compatible_detect_signature(fname)
        if sig_now is None:
            return float(getattr(self.params.P, "fwhm_seed_px", 6.0)), np.nan
        candidates = [
            self.cache_dir / f"detect_{fname}.json",
            step4_dir(self.result_dir) / f"detect_{fname}.json",
        ]
        candidates = [p for p in candidates if p.exists()]
        candidates.sort(key=lambda p: p.stat().st_mtime_ns if p.exists() else 0, reverse=True)
        for meta_json in candidates:
            try:
                meta = json.loads(meta_json.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not self._detect_meta_matches(meta, sig_now, meta_json):
                continue
            try:
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
                continue
        for meta_json in candidates:
            try:
                meta = json.loads(meta_json.read_text(encoding="utf-8"))
            except Exception:
                continue
            try:
                schema = int(meta.get("cache_schema", 0) or 0)
            except Exception:
                schema = 0
            if schema >= 2:
                continue
            if not self._legacy_detect_cache_allowed(meta_json):
                continue
            try:
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
                continue
        return float(getattr(self.params.P, "fwhm_seed_px", 6.0)), np.nan

    def _load_detect_xy(self, fname: str):
        sig_now = self._compatible_detect_signature(fname)
        if sig_now is None:
            return np.empty((0, 2)), None
        candidates = [
            self.cache_dir / f"detect_{fname}.csv",
            step4_dir(self.result_dir) / f"detect_{fname}.csv",
        ]
        for csv_path in candidates:
            if not csv_path.exists():
                continue
            meta_path = csv_path.with_suffix(".json")
            if not meta_path.exists():
                continue
            try:
                payload = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not self._detect_meta_matches(payload, sig_now, meta_path):
                continue
            try:
                df = pd.read_csv(csv_path)
                if not {"x", "y"} <= set(df.columns):
                    continue
                xy = df[["x", "y"]].to_numpy(float)
                xy = xy[np.isfinite(xy).all(axis=1)]
                return xy, df
            except Exception:
                continue
        for csv_path in candidates:
            if not csv_path.exists():
                continue
            meta_path = csv_path.with_suffix(".json")
            schema = 0
            if meta_path.exists():
                try:
                    payload = json.loads(meta_path.read_text(encoding="utf-8"))
                    schema = int(payload.get("cache_schema", 0) or 0)
                except Exception:
                    continue
            if schema >= 2:
                continue
            marker_path = meta_path if meta_path.exists() else csv_path
            if not self._legacy_detect_cache_allowed(marker_path):
                continue
            try:
                df = pd.read_csv(csv_path)
                if not {"x", "y"} <= set(df.columns):
                    continue
                xy = df[["x", "y"]].to_numpy(float)
                xy = xy[np.isfinite(xy).all(axis=1)]
                return xy, df
            except Exception:
                continue
        return np.empty((0, 2)), None

    def _wcs_rotation_deg(self, w: WCS) -> float:
        try:
            if not w.has_celestial:
                return float("nan")
            if hasattr(w.wcs, "cd") and w.wcs.cd is not None:
                cd = w.wcs.cd
            elif hasattr(w.wcs, "pc") and w.wcs.pc is not None:
                pc = w.wcs.pc
                cdelt = w.wcs.cdelt
                cd = pc * cdelt[:, np.newaxis]
            else:
                return float("nan")
            rot_rad = np.arctan2(-cd[0, 1], cd[1, 1])
            return float(np.degrees(rot_rad))
        except Exception:
            return float("nan")

    def _wcs_center_coords(self, w: WCS, nx: int, ny: int) -> tuple:
        try:
            if not w.has_celestial:
                return (float("nan"), float("nan"))
            cx, cy = nx / 2.0, ny / 2.0
            sky = w.pixel_to_world(cx, cy)
            return (float(sky.ra.deg), float(sky.dec.deg))
        except Exception:
            return (float("nan"), float("nan"))

    def _empty_wcs_qc_metrics(self, n_detect: int = 0) -> dict:
        return {
            "n_detect": int(max(0, n_detect)),
            "n_catalog_in_fov": 0,
            "n_match": 0,
            "n_inlier": 0,
            "match_rate": np.nan,
            "match_rate_cat": np.nan,
            "match_rate_eff": np.nan,
            "match_radius_arcsec": np.nan,
            "match_radius_px": np.nan,
            "dx_med_px": np.nan,
            "dy_med_px": np.nan,
            "resid_med_px": np.nan,
            "resid_mad_px": np.nan,
            "resid_peak_px": np.nan,
            "resid_p99_px": np.nan,
            "rms_px": np.nan,
            "inlier_rate": np.nan,
            "resid_vs_radius_slope": np.nan,
            "edge_resid_ratio": np.nan,
            "center_offset_arcsec": np.nan,
            "pix_scale_input_arcsec": np.nan,
            "pix_scale_fit_arcsec": np.nan,
            "scale_delta_pct": np.nan,
        }

    def _compute_wcs_qc_metrics(
        self,
        *,
        w: WCS | None,
        det_xy: np.ndarray,
        nx: int,
        ny: int,
        gaia_ra_deg: np.ndarray,
        gaia_dec_deg: np.ndarray,
        pix_input_arcsec: float,
        pix_fit_arcsec: float,
        center_coord: SkyCoord | None,
    ) -> dict:
        out = self._empty_wcs_qc_metrics(n_detect=len(det_xy))
        if np.isfinite(pix_input_arcsec):
            out["pix_scale_input_arcsec"] = float(pix_input_arcsec)
        if np.isfinite(pix_fit_arcsec):
            out["pix_scale_fit_arcsec"] = float(pix_fit_arcsec)
        if np.isfinite(pix_input_arcsec) and pix_input_arcsec > 0 and np.isfinite(pix_fit_arcsec):
            out["scale_delta_pct"] = float((pix_fit_arcsec - pix_input_arcsec) / pix_input_arcsec * 100.0)

        if w is not None and w.has_celestial and center_coord is not None:
            try:
                c_ra, c_dec = self._wcs_center_coords(w, nx, ny)
                if np.isfinite(c_ra) and np.isfinite(c_dec):
                    c_sky = SkyCoord(c_ra * u.deg, c_dec * u.deg, frame="icrs")
                    out["center_offset_arcsec"] = float(c_sky.separation(center_coord).arcsec)
            except Exception:
                pass

        if w is None or (not w.has_celestial):
            return out
        if len(det_xy) == 0:
            return out
        if gaia_ra_deg.size == 0 or gaia_dec_deg.size == 0:
            return out

        try:
            xg, yg = w.celestial.all_world2pix(gaia_ra_deg, gaia_dec_deg, 0)
            xg = np.asarray(xg, float)
            yg = np.asarray(yg, float)
        except Exception:
            return out

        ok_g = (
            np.isfinite(xg)
            & np.isfinite(yg)
            & (xg >= 0.0)
            & (xg < float(nx))
            & (yg >= 0.0)
            & (yg < float(ny))
        )
        if not np.any(ok_g):
            return out

        gaia_xy = np.column_stack((xg[ok_g], yg[ok_g]))
        out["n_catalog_in_fov"] = int(len(gaia_xy))

        pix_use = pix_fit_arcsec if np.isfinite(pix_fit_arcsec) and pix_fit_arcsec > 0 else pix_input_arcsec
        match_r_arcsec = float(getattr(self.params.P, "wcs_qc_match_radius_arcsec", 2.0))
        if not np.isfinite(match_r_arcsec) or match_r_arcsec <= 0:
            match_r_arcsec = 2.0
        if np.isfinite(pix_use) and pix_use > 0:
            match_r_px = float(match_r_arcsec / pix_use)
        else:
            match_r_px = float(getattr(self.params.P, "wcs_qc_match_radius_px", 2.5))
        match_r_px = float(np.clip(match_r_px, 1.0, 25.0))
        out["match_radius_arcsec"] = float(match_r_arcsec)
        out["match_radius_px"] = float(match_r_px)

        tree = KDTree(gaia_xy)
        d, j = tree.query(det_xy, k=1)
        d = np.asarray(d, float)
        j = np.asarray(j, int)
        ok = np.isfinite(d) & (d <= match_r_px) & (j >= 0) & (j < len(gaia_xy))
        if not np.any(ok):
            return out

        det_candidates = np.where(ok)[0]
        order = np.argsort(d[det_candidates])
        used_gaia = set()
        keep_det = []
        keep_gaia = []
        for ord_idx in order:
            det_i = int(det_candidates[ord_idx])
            gaia_i = int(j[det_i])
            if gaia_i in used_gaia:
                continue
            used_gaia.add(gaia_i)
            keep_det.append(det_i)
            keep_gaia.append(gaia_i)
        if not keep_det:
            return out

        det_keep = np.asarray(keep_det, dtype=int)
        gaia_keep = np.asarray(keep_gaia, dtype=int)
        dx = det_xy[det_keep, 0] - gaia_xy[gaia_keep, 0]
        dy = det_xy[det_keep, 1] - gaia_xy[gaia_keep, 1]
        r = np.hypot(dx, dy)
        finite_r = np.isfinite(r)
        if not np.any(finite_r):
            return out
        if not np.all(finite_r):
            dx = dx[finite_r]
            dy = dy[finite_r]
            r = r[finite_r]
            gaia_keep = gaia_keep[finite_r]

        n_match = int(len(r))
        out["n_match"] = n_match
        out["match_rate"] = float(n_match / max(int(len(det_xy)), 1))
        out["match_rate_cat"] = float(n_match / max(int(len(gaia_xy)), 1))
        out["match_rate_eff"] = float(max(out["match_rate"], out["match_rate_cat"]))
        if n_match == 0:
            return out

        out["dx_med_px"] = float(np.nanmedian(dx)) if len(dx) else np.nan
        out["dy_med_px"] = float(np.nanmedian(dy)) if len(dy) else np.nan
        resid_med = float(np.nanmedian(r))
        resid_mad = float(1.4826 * np.nanmedian(np.abs(r - resid_med)))
        out["resid_med_px"] = resid_med
        out["resid_mad_px"] = resid_mad
        out["resid_p99_px"] = float(np.nanpercentile(r, 99))
        out["resid_peak_px"] = out["resid_p99_px"]

        clip_sigma = float(getattr(self.params.P, "wcs_qc_clip_sigma", 3.0))
        if not np.isfinite(clip_sigma) or clip_sigma <= 0:
            clip_sigma = 3.0
        if np.isfinite(resid_mad) and resid_mad > 0:
            inlier = np.abs(r - resid_med) <= clip_sigma * resid_mad
        else:
            rstd = float(np.nanstd(r))
            inlier = np.abs(r - float(np.nanmean(r))) <= clip_sigma * rstd if np.isfinite(rstd) and rstd > 0 else np.ones(len(r), dtype=bool)
        n_inlier = int(np.sum(inlier))
        r_in = r[inlier] if n_inlier > 0 else r
        out["n_inlier"] = n_inlier
        out["inlier_rate"] = float(n_inlier / max(n_match, 1))
        out["rms_px"] = float(np.sqrt(np.nanmean(r_in ** 2))) if len(r_in) else np.nan

        if len(det_keep) >= 8:
            cx = float(nx) / 2.0
            cy = float(ny) / 2.0
            rr = np.hypot(gaia_xy[gaia_keep, 0] - cx, gaia_xy[gaia_keep, 1] - cy)
            max_rr = max(float(np.hypot(max(cx, 1.0), max(cy, 1.0))), 1.0)
            rho = rr / max_rr
            if np.isfinite(np.nanstd(rho)) and float(np.nanstd(rho)) > 1e-6:
                try:
                    out["resid_vs_radius_slope"] = float(np.polyfit(rho, r, 1)[0])
                except Exception:
                    out["resid_vs_radius_slope"] = np.nan
            core = r[rho <= 0.4]
            edge = r[rho >= 0.8]
            if len(core) >= 3 and len(edge) >= 3:
                core_med = float(np.nanmedian(core))
                if np.isfinite(core_med) and core_med > 1e-9:
                    out["edge_resid_ratio"] = float(np.nanmedian(edge) / core_med)

        return out

    def _evaluate_wcs_qc_pass(self, metrics: dict, *, wcs_ok: bool) -> tuple[bool, list[str]]:
        def _num(key: str) -> float:
            try:
                return float(metrics.get(key, np.nan))
            except Exception:
                return np.nan

        reasons: list[str] = []
        if bool(getattr(self.params.P, "wcs_qc_require_wcs_ok", True)) and not wcs_ok:
            reasons.append("wcs_fail")
        n_detect = int(metrics.get("n_detect", 0) or 0)
        n_match = int(metrics.get("n_match", 0) or 0)
        if n_detect <= 0:
            reasons.append("no_detect_data")
        if int(getattr(self.params.P, "wcs_qc_min_match_n", 20)) > 0 and n_match < int(getattr(self.params.P, "wcs_qc_min_match_n", 20)):
            reasons.append("low_match_n")
        mrate_det = _num("match_rate")
        mrate_cat = _num("match_rate_cat")
        mrate_eff = _num("match_rate_eff")
        if not np.isfinite(mrate_eff):
            if np.isfinite(mrate_det) or np.isfinite(mrate_cat):
                mrate_eff = float(np.nanmax([mrate_det, mrate_cat]))
        min_rate = float(getattr(self.params.P, "wcs_qc_min_match_rate", 0.20))
        if np.isfinite(min_rate) and min_rate > 0 and ((not np.isfinite(mrate_eff)) or (mrate_eff < min_rate)):
            reasons.append("low_match_rate")
        max_rms = float(getattr(self.params.P, "wcs_qc_max_rms_px", 2.5))
        if n_match > 0 and np.isfinite(max_rms) and max_rms > 0:
            rms_px = _num("rms_px")
            if (not np.isfinite(rms_px)) or (rms_px > max_rms):
                reasons.append("high_rms")
        max_p99 = float(getattr(self.params.P, "wcs_qc_max_p99_px", 5.0))
        if n_match > 0 and np.isfinite(max_p99) and max_p99 > 0:
            p99_px = _num("resid_p99_px")
            if (not np.isfinite(p99_px)) or (p99_px > max_p99):
                reasons.append("high_p99")
        min_inlier = float(getattr(self.params.P, "wcs_qc_min_inlier_rate", 0.50))
        if n_match > 0 and np.isfinite(min_inlier) and min_inlier > 0:
            inlier = _num("inlier_rate")
            if (not np.isfinite(inlier)) or (inlier < min_inlier):
                reasons.append("low_inlier")
        edge_ratio = _num("edge_resid_ratio")
        max_edge = float(getattr(self.params.P, "wcs_qc_max_edge_ratio", 0.0))
        if np.isfinite(max_edge) and max_edge > 0 and np.isfinite(edge_ratio) and edge_ratio > max_edge:
            reasons.append("edge_resid")
        center_off = _num("center_offset_arcsec")
        max_center = float(getattr(self.params.P, "wcs_qc_max_center_offset_arcsec", 0.0))
        if np.isfinite(max_center) and max_center > 0 and ((not np.isfinite(center_off)) or (center_off > max_center)):
            reasons.append("center_offset")
        return len(reasons) == 0, reasons

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
            canon = _canonicalize_windows_path_case(path)
            wp = PureWindowsPath(str(canon))
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
        wcs_path = outdir / f"{stem}.wcs"
        sig_path = outdir / f"{stem}.input.json"
        source_sig = _build_source_signature(fits_path, use_cropped=bool(self.use_cropped))
        if use_cache and new_path.exists() and (solved_path.exists() or wcs_path.exists()):
            cache_ok = False
            try:
                if sig_path.exists():
                    saved_sig = json.loads(sig_path.read_text(encoding="utf-8"))
                    cache_ok = _source_signature_detection_compatible(saved_sig, source_sig)
            except Exception:
                cache_ok = False
            if cache_ok:
                return True, 0.0, "cache_hit", "", [], new_path
        for p in outdir.glob(f"{stem}.*"):
            try:
                p.unlink()
            except Exception:
                pass
        staged_path = fits_path
        if stage_in_outdir and (not use_wsl):
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
        if not staged_path.exists():
            return False, 0.0, "", f"input_missing:{staged_path}", cmd, None
        if use_wsl and cmd and str(cmd[0]).lower() == "wsl":
            try:
                chk = subprocess.run(
                    ["wsl", "test", "-f", fits_arg],
                    capture_output=True,
                    text=True,
                    timeout=5.0,
                )
                if chk.returncode != 0:
                    err_msg = (
                        f"wsl_path_unavailable:{fits_arg} | "
                        "WSL cannot read this file path. Use ASTAP solver or move data to a local "
                        "drive/folder that WSL can access (e.g. C:) and rerun."
                    )
                    return False, 0.0, "", err_msg, cmd, None
            except FileNotFoundError:
                return (
                    False,
                    0.0,
                    "",
                    "wsl_not_found: WSL command not available. Disable WSL solve and use ASTAP.",
                    cmd,
                    None,
                )
            except Exception:
                pass

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
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stdout_s = ""
            stderr_s = ""
            while True:
                if self._stop_requested:
                    try:
                        proc.terminate()
                        proc.wait(timeout=2.0)
                    except Exception:
                        try:
                            proc.kill()
                        except Exception:
                            pass
                    try:
                        out_s, err_s = proc.communicate(timeout=0.5)
                        stdout_s = out_s or ""
                        stderr_s = err_s or ""
                    except Exception:
                        pass
                    if staged_path != fits_path:
                        try:
                            staged_path.unlink()
                        except Exception:
                            pass
                    err_msg = "stopped"
                    err_tail = _tail_text(stderr_s, limit=1000, max_lines=10)
                    if err_tail:
                        err_msg = f"stopped | {err_tail}"
                    return False, time.time() - start, stdout_s, err_msg, cmd, None

                elapsed = time.time() - start
                if elapsed >= timeout_s:
                    try:
                        proc.terminate()
                        proc.wait(timeout=2.0)
                    except Exception:
                        try:
                            proc.kill()
                        except Exception:
                            pass
                    try:
                        out_s, err_s = proc.communicate(timeout=0.5)
                        stdout_s = out_s or ""
                        stderr_s = err_s or ""
                    except Exception:
                        pass
                    if staged_path != fits_path:
                        try:
                            staged_path.unlink()
                        except Exception:
                            pass
                    err_msg = "timeout"
                    err_tail = _tail_text(stderr_s, limit=1000, max_lines=10)
                    if err_tail:
                        err_msg = f"timeout | {err_tail}"
                    return False, timeout_s, stdout_s, err_msg, cmd, None

                try:
                    out_s, err_s = proc.communicate(timeout=0.2)
                    stdout_s = out_s or ""
                    stderr_s = err_s or ""
                    break
                except subprocess.TimeoutExpired:
                    continue

            dt = time.time() - start
            rc = int(proc.returncode if proc.returncode is not None else -998)
            # Some solve-field builds can return non-zero even when solution artifacts are created.
            # Prefer artifact existence over process return code.
            wcs_path = outdir / f"{stem}.wcs"
            ok = bool(new_path.exists() and (solved_path.exists() or wcs_path.exists()))
            if ok:
                try:
                    sig_path.write_text(json.dumps(source_sig, indent=2), encoding="utf-8")
                except Exception:
                    pass
            if staged_path != fits_path:
                try:
                    staged_path.unlink()
                except Exception:
                    pass
            return ok, dt, stdout_s, stderr_s, cmd, new_path
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
        log_path = self.cache_dir / "astnet_solve.log"

        def LOG(msg, emit=False):
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            line = f"{ts} {msg}"
            try:
                with open(log_path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            except Exception:
                pass
            if emit:
                self.log_message.emit(msg)

        LOG("=" * 60)
        LOG(
            f"[ASTNET] start files={len(self.file_list)} use_cropped={self.use_cropped} "
            f"workers={max_workers} outdir={outdir}"
        )
        LOG(
            f"[ASTNET] scale=[{scale_low:.5f},{scale_high:.5f}] downsample={downsample} "
            f"max_objs={max_objs} radius_deg={radius_deg} timeout_s={timeout_s} "
            f"cpulimit_s={cpulimit_s} use_wsl={use_wsl} use_cache={use_cache}"
        )

        self.log_message.emit(f"Starting parallel plate solving with {max_workers} workers...")
        self.log_message.emit(f"  Scale: {scale_low:.4f} - {scale_high:.4f} arcsec/px")
        self.log_message.emit(f"  Downsample: {downsample}, Max objs: {max_objs}")
        self.log_message.emit(f"  Debug log: {log_path}")

        # 내부 함수: 단일 파일 처리 로직 (스레드에서 실행됨)
        def process_single_file(filename):
            if self._stop_requested:
                LOG(f"{filename}: stop_requested_before_start")
                return filename, {"ok": False, "status": "stopped", "fail_reason": "stopped"}

            if self.use_cropped:
                fits_path = step2_cropped_dir(self.result_dir) / filename
            else:
                fits_path = self.params.get_file_path(filename)
            source_sig = _build_source_signature(Path(fits_path), use_cropped=bool(self.use_cropped))

            if not fits_path.exists():
                LOG(f"{filename}: file_not_found path={fits_path}")
                return filename, {"ok": False, "status": "file_not_found", "fail_reason": "file_not_found", **source_sig}

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
            cmd_str = " ".join(str(c) for c in cmd) if cmd else ""
            stdout_tail = _tail_text(out_s, limit=2000, max_lines=12)
            stderr_tail = _tail_text(err_s, limit=2000, max_lines=12)
            fail_reason = ""
            if not ok:
                err_l = str(err_s).lower()
                if "timeout" in err_l:
                    fail_reason = "astnet_timeout"
                elif "stopped" in err_l:
                    fail_reason = "astnet_stopped"
                elif (
                    "wsl_path_unavailable" in err_l
                    or "wsl_not_found" in err_l
                    or "wsl_input_missing" in err_l
                    or "input_missing:" in err_l
                    or ("cannot open `" in err_l and "no such file or directory" in err_l)
                ):
                    fail_reason = "astnet_wsl_path_unavailable"
                elif "cache_hit" in str(out_s):
                    fail_reason = "astnet_cache_hit_without_solution_marker"
                else:
                    fail_reason = "astnet_fail_no_solution"

            result = {
                "fname": filename,
                "file": filename,
                "ok": False,
                "wcs_ok": False,
                "status": "fail",
                "fail_reason": fail_reason,
                "ra": 0.0, "dec": 0.0, "pixscale": 0.0,
                "elapsed_s": float(dt),
                "solver": "astnet_wsl",
                "astnet_wsl_cmd": cmd_str,
                "astnet_wsl_stdout": stdout_tail,
                "astnet_wsl_stderr": stderr_tail,
                **source_sig,
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
                                "fname": filename,
                                "file": filename,
                                "ok": True,
                                "wcs_ok": True,
                                "status": "solved",
                                "ra": float(ra_dec.ra.deg),
                                "dec": float(ra_dec.dec.deg),
                                "pixscale": pix_fit,
                                "elapsed_s": float(dt),
                                "solver": "astnet_wsl",
                                "fail_reason": "",
                                "astnet_wsl_cmd": cmd_str,
                                "astnet_wsl_stdout": stdout_tail,
                                "astnet_wsl_stderr": stderr_tail,
                                "wcs_header": dict(new_hdr),
                                "fits_path": str(fits_path),
                                **source_sig,
                            }
                            LOG(
                                f"{filename}: solved dt={dt:.1f}s RA={result['ra']:.6f} "
                                f"Dec={result['dec']:.6f} pix={pix_fit:.5f}"
                            )
                            if stderr_tail:
                                LOG(f"{filename}: solver_stderr_tail={stderr_tail}")
                        else:
                            fail_reason = "wcs_header_not_celestial"
                            result["status"] = "wcs_not_celestial"
                            result["fail_reason"] = fail_reason
                            LOG(f"{filename}: fail reason={fail_reason} dt={dt:.1f}s")
                            if cmd_str:
                                LOG(f"{filename}: cmd={cmd_str}")
                            if stdout_tail:
                                LOG(f"{filename}: stdout_tail={stdout_tail}")
                            if stderr_tail:
                                LOG(f"{filename}: stderr_tail={stderr_tail}")
                except Exception as e:
                    fail_reason = f"header_update_error:{_exc_brief(e, limit=160)}"
                    result = {
                        "fname": filename,
                        "file": filename,
                        "ok": False,
                        "wcs_ok": False,
                        "status": f"error: {e}",
                        "fail_reason": fail_reason,
                        "elapsed_s": float(dt),
                        "solver": "astnet_wsl",
                        "astnet_wsl_cmd": cmd_str,
                        "astnet_wsl_stdout": stdout_tail,
                        "astnet_wsl_stderr": stderr_tail,
                    }
                    LOG(f"{filename}: {fail_reason}")
                    if cmd_str:
                        LOG(f"{filename}: cmd={cmd_str}")
                    if stdout_tail:
                        LOG(f"{filename}: stdout_tail={stdout_tail}")
                    if stderr_tail:
                        LOG(f"{filename}: stderr_tail={stderr_tail}")
            else:
                if not fail_reason:
                    fail_reason = "astnet_no_solution_file"
                result["fail_reason"] = fail_reason
                LOG(f"{filename}: fail reason={fail_reason} dt={dt:.1f}s")
                if cmd_str:
                    LOG(f"{filename}: cmd={cmd_str}")
                if stdout_tail:
                    LOG(f"{filename}: stdout_tail={stdout_tail}")
                if stderr_tail:
                    LOG(f"{filename}: stderr_tail={stderr_tail}")

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
                    for f_cancel in future_to_file:
                        f_cancel.cancel()
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
                        status_txt = str(res.get("status", "fail"))
                        fail_txt = str(res.get("fail_reason", "")).strip()
                        if fail_txt:
                            self.log_message.emit(f"[FAIL] {filename}: {status_txt} ({fail_txt})")
                        else:
                            self.log_message.emit(f"[FAIL] {filename}: {status_txt}")
                        try:
                            elapsed_val = float(res.get("elapsed_s", np.nan))
                        except Exception:
                            elapsed_val = np.nan
                        LOG(
                            f"{filename}: status={status_txt} fail_reason={fail_txt or '-'} "
                            f"elapsed={elapsed_val:.1f}s"
                        )

                except Exception as e:
                    LOG(f"{fname}: worker_exception={_exc_brief(e)}")
                    self.error.emit(fname, str(e))

                completed_count += 1
                self.progress.emit(completed_count, total, f"Solved {completed_count}/{total}")

        # --- Gaia 쿼리 및 WCS Refine ---
        if self._stop_requested:
            self.finished.emit({
                "total": len(results),
                "ok": sum(1 for r in results if r.get("ok")),
                "wcs_qc_pass": sum(1 for r in results if bool(r.get("wcs_qc_pass", False))),
                "stopped": True,
            })
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
                gaia_mag_max = float(getattr(self.params.P, "gaia_mag_max", 18.0))
                gaia_r = float(0.5 * diag_deg * gaia_fudge)

                gaia_df, gaia_src = self._load_or_query_gaia(center_coord, gaia_r)
                self.log_message.emit(
                    f"[Gaia] center=({center_coord.ra.deg:.6f},{center_coord.dec.deg:.6f}) "
                    f"r={gaia_r:.4f}deg mag_max={gaia_mag_max:.2f} source={gaia_src} N={len(gaia_df)}"
                )
            except Exception as e:
                msg = _exc_brief(e, limit=240)
                self.log_message.emit(f"[Gaia] Query error: {msg}")
                LOG(f"[Gaia] Query error: {msg}")
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

        gaia_ra_vals = np.array([], dtype=float)
        gaia_dec_vals = np.array([], dtype=float)
        if isinstance(gaia_df, pd.DataFrame) and (not gaia_df.empty) and {"ra", "dec"} <= set(gaia_df.columns):
            gaia_ra_vals = pd.to_numeric(gaia_df["ra"], errors="coerce").to_numpy(float)
            gaia_dec_vals = pd.to_numeric(gaia_df["dec"], errors="coerce").to_numpy(float)
            ok_rd = np.isfinite(gaia_ra_vals) & np.isfinite(gaia_dec_vals)
            gaia_ra_vals = gaia_ra_vals[ok_rd]
            gaia_dec_vals = gaia_dec_vals[ok_rd]

        for res in results:
            filename = str(res.get("filename") or res.get("fname") or res.get("file") or "").strip()
            if not filename:
                continue

            det_xy, _ = self._load_detect_xy(filename)
            qc_metrics = self._empty_wcs_qc_metrics(n_detect=len(det_xy))
            qc_metrics["pix_scale_input_arcsec"] = float(pix_arc) if np.isfinite(pix_arc) else np.nan

            fits_path = None
            fp = str(res.get("fits_path", "")).strip()
            if fp:
                fits_path = Path(fp)
            if fits_path is None or not fits_path.exists():
                if self.use_cropped:
                    cp = step2_cropped_dir(self.result_dir) / filename
                    if cp.exists():
                        fits_path = cp
                if (fits_path is None or not fits_path.exists()):
                    try:
                        op = Path(self.params.get_file_path(filename))
                        if op.exists():
                            fits_path = op
                    except Exception:
                        fits_path = None

            w = None
            wcs_ok = False
            nx = 0
            ny = 0
            wcs_rot_deg = np.nan
            center_ra = np.nan
            center_dec = np.nan
            pix_fit = float(res.get("pixscale", np.nan))
            if fits_path is not None and fits_path.exists():
                try:
                    with fits.open(fits_path, memmap=False) as hdul:
                        hdr = hdul[0].header
                        nx = int(hdr.get("NAXIS1", 0))
                        ny = int(hdr.get("NAXIS2", 0))
                        w = WCS(hdr, relax=True)
                    wcs_ok = bool(w is not None and w.has_celestial)
                    if wcs_ok:
                        pix_fit = self._pixscale_from_wcs(w)
                        wcs_rot_deg = self._wcs_rotation_deg(w)
                        center_ra, center_dec = self._wcs_center_coords(w, nx, ny)
                except Exception:
                    w = None
                    wcs_ok = False

            qc_metrics = self._compute_wcs_qc_metrics(
                w=w,
                det_xy=det_xy,
                nx=nx,
                ny=ny,
                gaia_ra_deg=gaia_ra_vals,
                gaia_dec_deg=gaia_dec_vals,
                pix_input_arcsec=float(pix_arc),
                pix_fit_arcsec=float(pix_fit) if np.isfinite(pix_fit) else np.nan,
                center_coord=center_coord,
            )
            qc_pass, qc_reasons = self._evaluate_wcs_qc_pass(qc_metrics, wcs_ok=bool(wcs_ok))

            elapsed_s = float(res.get("elapsed_s", np.nan))
            elapsed = elapsed_s if np.isfinite(elapsed_s) else np.nan
            res.update({
                "fname": filename,
                "file": filename,
                "wcs_ok": bool(wcs_ok),
                "pix_fit": float(pix_fit) if np.isfinite(pix_fit) else None,
                "elapsed": float(elapsed) if np.isfinite(elapsed) else None,
                "wcs_rot_deg": float(wcs_rot_deg) if np.isfinite(wcs_rot_deg) else None,
                "center_ra_deg": float(center_ra) if np.isfinite(center_ra) else None,
                "center_dec_deg": float(center_dec) if np.isfinite(center_dec) else None,
                "wcs_qc_pass": bool(qc_pass),
                "wcs_qc_reason": ",".join(qc_reasons),
            })
            res.update(qc_metrics)
            try:
                mr_eff = float(res.get("match_rate_eff", np.nan))
            except Exception:
                mr_eff = np.nan
            try:
                rms_px = float(res.get("rms_px", np.nan))
            except Exception:
                rms_px = np.nan
            try:
                n_det = int(res.get("n_detect", 0) or 0)
            except Exception:
                n_det = 0
            try:
                n_match = int(res.get("n_match", 0) or 0)
            except Exception:
                n_match = 0
            qc_txt = "PASS" if qc_pass else f"FAIL({','.join(qc_reasons) or '-'})"
            self.log_message.emit(
                f"[WCS-QC] {filename}: n_det={n_det} n_match={n_match} "
                f"rate_eff={mr_eff:.3f} rms_px={rms_px:.3f} qc={qc_txt}"
            )
            # Push finalized metrics so Astrometry table reflects QC columns, not only refine residue.
            self.file_done.emit(filename, res)

        try:
            step5_out = step5_dir(self.result_dir)
            step5_out.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame(results)
            df.to_csv(step5_out / "wcs_solve_summary.csv", index=False)
            qc_cols = [
                "file",
                "status",
                "fail_reason",
                "ok",
                "wcs_ok",
                "solver",
                "elapsed",
                "elapsed_s",
                "source_path",
                "source_use_cropped",
                "source_size",
                "source_mtime_ns",
                "n_detect",
                "n_catalog_in_fov",
                "n_match",
                "n_inlier",
                "match_rate",
                "match_rate_cat",
                "match_rate_eff",
                "match_radius_arcsec",
                "match_radius_px",
                "dx_med_px",
                "dy_med_px",
                "resid_med_px",
                "resid_mad_px",
                "resid_p99_px",
                "resid_peak_px",
                "rms_px",
                "inlier_rate",
                "resid_vs_radius_slope",
                "edge_resid_ratio",
                "pix_scale_input_arcsec",
                "pixscale",
                "pix_fit",
                "scale_delta_pct",
                "wcs_rot_deg",
                "center_ra_deg",
                "center_dec_deg",
                "center_offset_arcsec",
                "wcs_qc_pass",
                "wcs_qc_reason",
            ]
            qc_df = df[[c for c in qc_cols if c in df.columns]].copy()
            qc_df.to_csv(step5_out / "frame_wcs_qc.csv", index=False)
        except Exception as e:
            self.log_message.emit(f"[WCS-QC] Failed to write QC CSV: {e}")

        # --- 마무리 ---
        n_qc_pass = sum(1 for r in results if bool(r.get("wcs_qc_pass", False)))
        summary = {
            "total": len(results),
            "ok": sum(1 for r in results if r.get("ok")),
            "wcs_qc_pass": int(n_qc_pass),
            "stopped": bool(self._stop_requested),
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
        self._load_existing_result_tables()

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
        self.results_table.setColumnCount(13)
        self.results_table.setHorizontalHeaderLabels([
            "File", "Status", "Solver", "RA", "Dec", "PixScale",
            "Refine", "Match N", "Match Rate Eff", "RMS(px)", "WCS QC",
            "Fail Reason", "Elapsed (s)"
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
        self.astrometrynet_results_table.setColumnCount(14)
        self.astrometrynet_results_table.setHorizontalHeaderLabels([
            "File", "Status", "Solver", "RA", "Dec", "PixScale",
            "Refine", "Resid(\")", "Match N", "Match Rate Eff",
            "RMS(px)", "WCS QC", "Fail Reason", "Elapsed (s)"
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
            self.btn_stop_astrometrynet.setEnabled(False)
            self.astrometrynet_status.setText("Stopping...")
            self.log("Astrometry.net stop requested...")
            self.astrometrynet_worker.stop()

    def on_astrometrynet_progress(self, current, total, status):
        pct = int(100 * current / max(1, total))
        self.astrometrynet_progress.setValue(pct)
        self.astrometrynet_status.setText(status)

    def on_astrometrynet_file_done(self, filename, result, emit_log=True):
        def _f(v):
            try:
                return float(v)
            except Exception:
                return float("nan")

        row = -1
        for i in range(self.astrometrynet_results_table.rowCount()):
            item = self.astrometrynet_results_table.item(i, 0)
            if item and item.text() == str(filename):
                row = i
                break
        is_new_row = row < 0
        if is_new_row:
            row = self.astrometrynet_results_table.rowCount()
            self.astrometrynet_results_table.insertRow(row)
        self.astrometrynet_results_table.setItem(row, 0, QTableWidgetItem(str(filename)))
        status = str(result.get("status", ""))
        self.astrometrynet_results_table.setItem(row, 1, QTableWidgetItem(status))
        solver = str(result.get("solver", "") or "astnet_wsl")
        self.astrometrynet_results_table.setItem(row, 2, QTableWidgetItem(solver))

        ra = _f(result.get("ra", result.get("center_ra_deg", np.nan)))
        dec = _f(result.get("dec", result.get("center_dec_deg", np.nan)))
        pixscale = _f(result.get("pixscale", result.get("pix_fit", np.nan)))
        refine = str(result.get("refine", "-") or "-")
        resid_med = _f(result.get("resid_med", np.nan))
        n_match = int(result.get("n_match", result.get("match_n", 0)) or 0)
        match_rate = _f(self._effective_match_rate(result))
        rms_px = _f(result.get("rms_px", np.nan))
        qc_pass = bool(result.get("wcs_qc_pass", False))
        qc_reason = str(result.get("wcs_qc_reason", "") or "").strip()
        fail_reason = self._text_or_dash(result.get("fail_reason", ""))
        if fail_reason == "-" and (not bool(result.get("ok", False))):
            fail_reason = self._text_or_dash(result.get("wcs_qc_reason", ""))
            if fail_reason == "-":
                fail_reason = self._text_or_dash(result.get("status", ""))
        elapsed = _f(result.get("elapsed", result.get("elapsed_s", 0.0)) or 0.0)

        self.astrometrynet_results_table.setItem(row, 3, QTableWidgetItem(f"{ra:.6f}" if np.isfinite(ra) else "-"))
        self.astrometrynet_results_table.setItem(row, 4, QTableWidgetItem(f"{dec:.6f}" if np.isfinite(dec) else "-"))
        self.astrometrynet_results_table.setItem(row, 5, QTableWidgetItem(f"{pixscale:.4f}" if np.isfinite(pixscale) and pixscale > 0 else "-"))
        self.astrometrynet_results_table.setItem(row, 6, QTableWidgetItem(refine))
        self.astrometrynet_results_table.setItem(row, 7, QTableWidgetItem(f"{resid_med:.2f}" if np.isfinite(resid_med) else "-"))
        self.astrometrynet_results_table.setItem(row, 8, QTableWidgetItem(str(n_match)))
        self.astrometrynet_results_table.setItem(row, 9, QTableWidgetItem(f"{match_rate:.3f}" if np.isfinite(match_rate) else "-"))
        self.astrometrynet_results_table.setItem(row, 10, QTableWidgetItem(f"{rms_px:.3f}" if np.isfinite(rms_px) else "-"))
        if qc_pass:
            qc_txt = "PASS"
        elif qc_reason:
            qc_txt = f"FAIL ({qc_reason})"
        else:
            qc_txt = "-"
        self.astrometrynet_results_table.setItem(row, 11, QTableWidgetItem(qc_txt))
        self.astrometrynet_results_table.setItem(row, 12, QTableWidgetItem(fail_reason))
        self.astrometrynet_results_table.setItem(row, 13, QTableWidgetItem(f"{elapsed:.1f}" if np.isfinite(elapsed) and elapsed > 0 else "-"))

        self.results[filename] = result
        if result.get("ok") and emit_log and is_new_row:
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
                self.astrometrynet_results_table.setItem(row, 6, QTableWidgetItem(str(refine) if refine else "-"))
                self.astrometrynet_results_table.setItem(row, 7, QTableWidgetItem(f"{resid_med:.2f}" if np.isfinite(resid_med) else "-"))
                # results에도 업데이트
                if filename in self.results:
                    self.results[filename].update(result)
                break

    def on_astrometrynet_finished(self, summary):
        self.btn_solve_astrometrynet.setEnabled(True)
        self.btn_stop_astrometrynet.setEnabled(False)
        stopped = bool(summary.get("stopped")) if isinstance(summary, dict) else False
        n_ok = summary.get("ok", 0)
        n_qc = summary.get("wcs_qc_pass", 0)
        if not stopped:
            self.astrometrynet_progress.setValue(100)
            self.astrometrynet_status.setText(f"Done: {n_ok}/{summary.get('total', 0)} solved")
        else:
            self.astrometrynet_status.setText(f"Stopped: {n_ok}/{summary.get('total', 0)} solved")
        if n_ok > 0:
            self.log(f"Astrometry.net: {n_ok} frames solved successfully | WCS-QC pass: {n_qc}")
        if stopped:
            self.log("Astrometry.net solve stopped by user")
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

    @staticmethod
    def _csv_bool(v) -> bool:
        if isinstance(v, bool):
            return v
        if v is None:
            return False
        if isinstance(v, (int, np.integer)):
            return bool(v)
        s = str(v).strip().lower()
        return s in ("1", "true", "t", "yes", "y", "on")

    @staticmethod
    def _text_or_dash(v) -> str:
        if v is None:
            return "-"
        try:
            if isinstance(v, (float, np.floating)) and (not np.isfinite(v)):
                return "-"
        except Exception:
            pass
        s = str(v).strip()
        if not s or s.lower() in ("nan", "none", "null"):
            return "-"
        return s

    @staticmethod
    def _effective_match_rate(result: dict) -> float:
        rates = []
        for key in ("match_rate_eff", "match_rate", "match_rate_cat"):
            try:
                v = float(result.get(key, np.nan))
            except Exception:
                v = np.nan
            if np.isfinite(v):
                rates.append(v)
        if not rates:
            return float("nan")
        return float(max(rates))

    def _resolve_source_path_for_file(self, filename: str) -> Path | None:
        if self.use_cropped:
            cand = step2_cropped_dir(self.params.P.result_dir) / filename
            if cand.exists():
                return cand
            legacy = self.params.P.result_dir / "cropped" / filename
            if legacy.exists():
                return legacy
        try:
            orig = Path(self.params.get_file_path(filename))
            if orig.exists():
                return orig
        except Exception:
            pass
        return None

    def _current_source_signature(self, filename: str) -> dict | None:
        src = self._resolve_source_path_for_file(filename)
        if src is None or not src.exists():
            return None
        try:
            return _build_source_signature(src, use_cropped=bool(self.use_cropped))
        except Exception:
            return None

    def _summary_row_compatible(self, filename: str, rec: dict, summary_path: Path) -> bool:
        sig_now = self._current_source_signature(filename)
        if sig_now is None:
            return False
        if not isinstance(rec, dict):
            return False

        # Preferred path: signature-compatible match from new cache schema.
        src_path_val = rec.get("source_path")
        src_path_txt = str(src_path_val).strip().lower() if src_path_val is not None else ""
        if src_path_txt not in ("", "nan", "none", "null"):
            return _source_signature_detection_compatible(rec, sig_now)

        # Legacy path: allow only when saved fits_path matches current source.
        fits_path = _norm_path_key(rec.get("fits_path", ""))
        if not fits_path:
            return False
        if fits_path != _norm_path_key(sig_now.get("source_path", "")):
            return False
        return True

    def _resolve_wcs_summary_path(self) -> Path | None:
        candidates = [
            step5_dir(self.params.P.result_dir) / "wcs_solve_summary.csv",
            legacy_step7_wcs_dir(self.params.P.result_dir) / "wcs_solve_summary.csv",
            self.params.P.result_dir / "wcs_solve_summary.csv",
        ]
        existing = [p for p in candidates if p.exists()]
        existing.sort(key=lambda p: p.stat().st_mtime_ns if p.exists() else 0, reverse=True)
        for p in existing:
            if p.exists():
                return p
        return None

    def _load_existing_result_tables(self):
        summary_path = self._resolve_wcs_summary_path()
        if summary_path is None:
            return
        try:
            df = pd.read_csv(summary_path)
        except Exception:
            return
        if df.empty:
            return

        self.results_table.setRowCount(0)
        self.astrometrynet_results_table.setRowCount(0)
        self.results = {}
        ignored = 0
        loaded = 0

        for _, row in df.iterrows():
            rec = row.to_dict()
            fname = str(rec.get("file", rec.get("fname", "")) or "").strip()
            if not fname:
                continue
            if not self._summary_row_compatible(fname, rec, summary_path):
                ignored += 1
                continue
            rec["file"] = fname
            rec["fname"] = fname
            rec["ok"] = self._csv_bool(rec.get("ok"))
            rec["wcs_qc_pass"] = self._csv_bool(rec.get("wcs_qc_pass"))
            self.results[fname] = rec
            loaded += 1

            # Primary summary on ASTAP tab
            self.on_file_done(fname, rec, emit_log=False)

            # Show astrometry.net rows on its dedicated tab too
            solver = str(rec.get("solver", "") or "").lower()
            if "astnet" in solver:
                self.on_astrometrynet_file_done(fname, rec, emit_log=False)
        if ignored > 0:
            self.log(
                f"[CACHE] ignored {ignored} stale/incompatible WCS rows from {summary_path.name}"
            )
        if loaded <= 0 and ignored > 0:
            self.log("[CACHE] no compatible WCS summary rows loaded; re-run Step5 for current frame set.")

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
        wcs_form.addRow("Gaia Query Mag Max (Step5):", self.param_gaia_mag_max)

        self.param_ref_gaia_match_tol = QDoubleSpinBox()
        self.param_ref_gaia_match_tol.setRange(0.1, 30.0)
        self.param_ref_gaia_match_tol.setDecimals(2)
        self.param_ref_gaia_match_tol.setSingleStep(0.1)
        self.param_ref_gaia_match_tol.setValue(float(getattr(self.params.P, "ref_wcs_match_radius_arcsec", 2.0)))
        wcs_form.addRow("Gaia Match Tol (Ref, arcsec):", self.param_ref_gaia_match_tol)

        self.param_wcs_qc_match_radius = QDoubleSpinBox()
        self.param_wcs_qc_match_radius.setRange(0.5, 15.0)
        self.param_wcs_qc_match_radius.setDecimals(2)
        self.param_wcs_qc_match_radius.setSingleStep(0.1)
        self.param_wcs_qc_match_radius.setValue(float(getattr(self.params.P, "wcs_qc_match_radius_arcsec", 2.0)))
        wcs_form.addRow("WCS-QC Match Radius (arcsec):", self.param_wcs_qc_match_radius)

        self.param_gaia_g_limit = QDoubleSpinBox()
        self.param_gaia_g_limit.setRange(10.0, 25.0)
        self.param_gaia_g_limit.setDecimals(2)
        self.param_gaia_g_limit.setSingleStep(0.5)
        self.param_gaia_g_limit.setValue(
            float(
                getattr(
                    self.params.P,
                    "idmatch_gaia_g_limit",
                    getattr(self.params.P, "gaia_mag_max", 18.0),
                )
            )
        )
        wcs_form.addRow("Gaia G Limit (Hybrid ID, Step6/7):", self.param_gaia_g_limit)

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
        dialog.resize(560, 640)

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

        self.param_astnet_wcs_qc_match_radius = QDoubleSpinBox()
        self.param_astnet_wcs_qc_match_radius.setRange(0.5, 15.0)
        self.param_astnet_wcs_qc_match_radius.setDecimals(2)
        self.param_astnet_wcs_qc_match_radius.setSingleStep(0.1)
        self.param_astnet_wcs_qc_match_radius.setValue(float(getattr(self.params.P, "wcs_qc_match_radius_arcsec", 2.0)))
        form.addRow("WCS-QC Match Radius (arcsec):", self.param_astnet_wcs_qc_match_radius)

        self.param_astnet_gaia_fudge = QDoubleSpinBox()
        self.param_astnet_gaia_fudge.setRange(0.5, 3.0)
        self.param_astnet_gaia_fudge.setSingleStep(0.05)
        self.param_astnet_gaia_fudge.setValue(float(getattr(self.params.P, "gaia_radius_fudge", 1.35)))
        form.addRow("Gaia Radius Fudge:", self.param_astnet_gaia_fudge)

        self.param_astnet_gaia_mag_max = QDoubleSpinBox()
        self.param_astnet_gaia_mag_max.setRange(10.0, 25.0)
        self.param_astnet_gaia_mag_max.setDecimals(2)
        self.param_astnet_gaia_mag_max.setSingleStep(0.5)
        self.param_astnet_gaia_mag_max.setValue(float(getattr(self.params.P, "gaia_mag_max", 18.0)))
        form.addRow("Gaia Query Mag Max (Step5):", self.param_astnet_gaia_mag_max)

        self.param_astnet_ref_gaia_match_tol = QDoubleSpinBox()
        self.param_astnet_ref_gaia_match_tol.setRange(0.1, 30.0)
        self.param_astnet_ref_gaia_match_tol.setDecimals(2)
        self.param_astnet_ref_gaia_match_tol.setSingleStep(0.1)
        self.param_astnet_ref_gaia_match_tol.setValue(
            float(getattr(self.params.P, "ref_wcs_match_radius_arcsec", 2.0))
        )
        form.addRow("Gaia Match Tol (Ref, arcsec):", self.param_astnet_ref_gaia_match_tol)

        self.param_astnet_gaia_g_limit = QDoubleSpinBox()
        self.param_astnet_gaia_g_limit.setRange(10.0, 25.0)
        self.param_astnet_gaia_g_limit.setDecimals(2)
        self.param_astnet_gaia_g_limit.setSingleStep(0.5)
        self.param_astnet_gaia_g_limit.setValue(
            float(
                getattr(
                    self.params.P,
                    "idmatch_gaia_g_limit",
                    getattr(self.params.P, "gaia_mag_max", 18.0),
                )
            )
        )
        form.addRow("Gaia G Limit (Hybrid ID, Step6/7):", self.param_astnet_gaia_g_limit)

        self.param_astnet_gaia_retry = QSpinBox()
        self.param_astnet_gaia_retry.setRange(0, 10)
        self.param_astnet_gaia_retry.setValue(int(getattr(self.params.P, "gaia_retry", 2)))
        form.addRow("Gaia Retry:", self.param_astnet_gaia_retry)

        self.param_astnet_gaia_backoff = QDoubleSpinBox()
        self.param_astnet_gaia_backoff.setRange(0.0, 120.0)
        self.param_astnet_gaia_backoff.setSingleStep(0.5)
        self.param_astnet_gaia_backoff.setValue(float(getattr(self.params.P, "gaia_backoff_s", 6.0)))
        form.addRow("Gaia Backoff (s):", self.param_astnet_gaia_backoff)

        self.param_astnet_gaia_allow_no_cache = QCheckBox("Allow")
        self.param_astnet_gaia_allow_no_cache.setChecked(bool(getattr(self.params.P, "gaia_allow_no_cache", True)))
        form.addRow("Gaia Allow No Cache:", self.param_astnet_gaia_allow_no_cache)

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
        self.params.P.ref_wcs_match_radius_arcsec = self.param_ref_gaia_match_tol.value()
        self.params.P.wcs_qc_match_radius_arcsec = self.param_wcs_qc_match_radius.value()
        self.params.P.idmatch_gaia_g_limit = self.param_gaia_g_limit.value()
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
        self.params.P.wcs_qc_match_radius_arcsec = self.param_astnet_wcs_qc_match_radius.value()
        self.params.P.gaia_radius_fudge = self.param_astnet_gaia_fudge.value()
        self.params.P.gaia_mag_max = self.param_astnet_gaia_mag_max.value()
        self.params.P.ref_wcs_match_radius_arcsec = self.param_astnet_ref_gaia_match_tol.value()
        self.params.P.idmatch_gaia_g_limit = self.param_astnet_gaia_g_limit.value()
        self.params.P.gaia_retry = self.param_astnet_gaia_retry.value()
        self.params.P.gaia_backoff_s = self.param_astnet_gaia_backoff.value()
        self.params.P.gaia_allow_no_cache = self.param_astnet_gaia_allow_no_cache.isChecked()
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
            self.stop_requested = True
            self.btn_stop.setEnabled(False)
            self.progress_label.setText("Stopping...")
            self.log("Stop requested...")
            self.worker.stop()

    def on_progress(self, current, total, filename):
        self.progress_bar.setValue(current)
        self.progress_label.setText(f"{current}/{total} | {filename}")

    def on_file_done(self, filename, result, emit_log=True):
        def _f(v):
            try:
                return float(v)
            except Exception:
                return float("nan")

        self.results[filename] = result
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)
        self.results_table.setItem(row, 0, QTableWidgetItem(filename))
        status = str(result.get("status", ""))
        self.results_table.setItem(row, 1, QTableWidgetItem(status))

        solver = str(result.get("solver", "") or "-")
        self.results_table.setItem(row, 2, QTableWidgetItem(solver))

        ra = _f(result.get("center_ra_deg", result.get("ra", np.nan)))
        dec = _f(result.get("center_dec_deg", result.get("dec", np.nan)))
        self.results_table.setItem(row, 3, QTableWidgetItem(f"{ra:.6f}" if np.isfinite(ra) else "-"))
        self.results_table.setItem(row, 4, QTableWidgetItem(f"{dec:.6f}" if np.isfinite(dec) else "-"))

        pix_fit = _f(result.get("pix_fit", result.get("pixscale", np.nan)))
        self.results_table.setItem(row, 5, QTableWidgetItem(f"{pix_fit:.4f}" if np.isfinite(pix_fit) and pix_fit > 0 else "-"))

        refine = str(result.get("refine", "") or "-")
        self.results_table.setItem(row, 6, QTableWidgetItem(refine))

        n_match = int(result.get("n_match", result.get("match_n", 0)) or 0)
        self.results_table.setItem(row, 7, QTableWidgetItem(str(n_match)))

        match_rate = _f(self._effective_match_rate(result))
        self.results_table.setItem(row, 8, QTableWidgetItem(f"{match_rate:.3f}" if np.isfinite(match_rate) else "-"))

        rms_px = _f(result.get("rms_px", np.nan))
        self.results_table.setItem(row, 9, QTableWidgetItem(f"{rms_px:.3f}" if np.isfinite(rms_px) else "-"))

        qc_pass = bool(result.get("wcs_qc_pass", False))
        qc_reason = str(result.get("wcs_qc_reason", "") or "").strip()
        if qc_pass:
            qc_txt = "PASS"
        elif qc_reason:
            qc_txt = f"FAIL ({qc_reason})"
        else:
            qc_txt = "-"
        self.results_table.setItem(row, 10, QTableWidgetItem(qc_txt))

        fail_reason = self._text_or_dash(result.get("fail_reason", ""))
        if fail_reason == "-" and (not bool(result.get("ok", False))):
            fail_reason = self._text_or_dash(result.get("wcs_qc_reason", ""))
            if fail_reason == "-":
                fail_reason = self._text_or_dash(result.get("status", ""))
        self.results_table.setItem(row, 11, QTableWidgetItem(fail_reason))

        elapsed = _f(result.get("elapsed", result.get("elapsed_s", 0.0)) or 0.0)
        self.results_table.setItem(row, 12, QTableWidgetItem(f"{elapsed:.1f}" if np.isfinite(elapsed) else "-"))

        if emit_log:
            self.log(
                f"{filename}: {status} solver={solver} pix={pix_fit if np.isfinite(pix_fit) else '-'} "
                f"n_match={n_match} rate={match_rate if np.isfinite(match_rate) else '-'} "
                f"rms={rms_px if np.isfinite(rms_px) else '-'} qc={qc_txt} fail={fail_reason}"
            )

    def on_error(self, filename, error):
        self.log(f"ERROR {filename}: {error}")

    def on_finished(self, summary):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.stop_requested = False
        stopped = bool(summary.get("stopped")) if isinstance(summary, dict) else False
        self.progress_label.setText("Stopped" if stopped else "Done")
        if summary:
            self.log(
                f"WCS done: {summary.get('ok', 0)}/{summary.get('total', 0)} OK | "
                f"WCS-QC pass: {summary.get('wcs_qc_pass', 0)}"
            )
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
            "ref_wcs_match_radius_arcsec": getattr(self.params.P, "ref_wcs_match_radius_arcsec", 2.0),
            "idmatch_gaia_g_limit": getattr(self.params.P, "idmatch_gaia_g_limit", getattr(self.params.P, "gaia_mag_max", 18.0)),
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
