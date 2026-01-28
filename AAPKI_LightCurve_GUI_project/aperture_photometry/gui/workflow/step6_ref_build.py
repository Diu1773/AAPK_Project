"""
Step 6: Reference Build (WCS-based ref catalog)

- Select reference frame using detection-based quality metrics
- Build a fixed master star list from the reference frame detections
- Write master catalogs for downstream steps
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

from PyQt5.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QGroupBox, QMessageBox,
    QTextEdit, QFormLayout, QProgressBar, QDoubleSpinBox, QSpinBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, QWidget,
    QDialog, QDialogButtonBox, QTabWidget, QCheckBox, QComboBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from .step_window_base import StepWindowBase
from ...utils.step_paths import (
    step5_dir,
    step6_dir,
    step2_cropped_dir,
    step4_dir,
    crop_is_active,
    legacy_step5_refbuild_dir,
    legacy_step7_refbuild_dir,
    legacy_step7_wcs_dir,
)


_FILTER_RE = re.compile(r"[-_]([ugrizbvUGRIZBV])[-_.]", re.IGNORECASE)
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


def _get_filter_from_filename(filename: str) -> Optional[str]:
    match = _FILTER_RE.search(str(filename))
    return match.group(1).lower() if match else None


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


def _safe_float(val, default=np.nan) -> float:
    try:
        if val is None:
            return default
        return float(val)
    except Exception:
        return default


class RefBuildWorker(QThread):
    progress = pyqtSignal(int, int, str)
    log = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str, str)

    def __init__(
        self,
        params,
        data_dir: Path,
        result_dir: Path,
        cache_dir: Path,
        file_list: List[str],
        ref_filter: str,
        sat_drop_pct: float,
        elong_drop_pct: float,
        ref_cat_max_sources: int,
        ref_cat_min_sources: int,
        ref_cat_max_elong: float,
        ref_cat_max_abs_round: float,
        ref_cat_sharp_min: float,
        ref_cat_sharp_max: float,
        ref_cat_min_peak_adu: float,
        wcs_match_radius_arcsec: float,
        wcs_min_match_rate: float,
        wcs_min_match_n: int,
        wcs_max_sep_med_arcsec: float,
        wcs_max_sep_p90_arcsec: float,
        wcs_max_dup_rate: float,
        ref_per_date: bool,
        ref_build_mode: str = "hybrid",
        gaia_mag_limit: float = 18.0,
    ):
        super().__init__()
        self.params = params
        self.data_dir = Path(data_dir)
        self.result_dir = Path(result_dir)
        self.cache_dir = Path(cache_dir)
        self.file_list = list(file_list)
        self.ref_filter = ref_filter
        self.sat_drop_pct = float(sat_drop_pct)
        self.elong_drop_pct = float(elong_drop_pct)
        self.ref_cat_max_sources = int(ref_cat_max_sources)
        self.ref_cat_min_sources = int(ref_cat_min_sources)
        self.ref_cat_max_elong = float(ref_cat_max_elong)
        self.ref_cat_max_abs_round = float(ref_cat_max_abs_round)
        self.ref_cat_sharp_min = float(ref_cat_sharp_min)
        self.ref_cat_sharp_max = float(ref_cat_sharp_max)
        self.ref_cat_min_peak_adu = float(ref_cat_min_peak_adu)
        self.wcs_match_radius_arcsec = float(wcs_match_radius_arcsec)
        self.wcs_min_match_rate = float(wcs_min_match_rate)
        self.wcs_min_match_n = int(wcs_min_match_n)
        self.wcs_max_sep_med_arcsec = float(wcs_max_sep_med_arcsec)
        self.wcs_max_sep_p90_arcsec = float(wcs_max_sep_p90_arcsec)
        self.wcs_max_dup_rate = float(wcs_max_dup_rate)
        self.ref_per_date = bool(ref_per_date)
        self.ref_build_mode = str(ref_build_mode).lower()
        self.gaia_mag_limit = float(gaia_mag_limit)
        self._stop_requested = False
        # Cache for WCS headers (path -> fits.Header)
        self._wcs_header_cache: Dict[str, fits.Header] = {}

    def stop(self):
        self._stop_requested = True

    def _log(self, msg: str):
        self.log.emit(msg)

    def _load_meta(self, fname: str) -> Optional[dict]:
        meta_path = self.cache_dir / f"detect_{fname}.json"
        if not meta_path.exists():
            fallback = step4_dir(self.result_dir) / f"detect_{fname}.json"
            if fallback.exists():
                meta_path = fallback
            else:
                return None
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _resolve_detect_csv(self, fname: str) -> Optional[Path]:
        cand = self.cache_dir / f"detect_{fname}.csv"
        if cand.exists():
            return cand
        fallback = step4_dir(self.result_dir) / f"detect_{fname}.csv"
        if fallback.exists():
            return fallback
        return None

    def _resolve_fits_path(self, fname: str) -> Optional[Path]:
        step5_out = step5_dir(self.result_dir)
        cand = step5_out / fname
        if cand.exists():
            return cand
        cropped_dir = step2_cropped_dir(self.result_dir)
        if crop_is_active(self.result_dir):
            cand = cropped_dir / fname
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
        # fallback: original file if cropped path lacks WCS
        try:
            orig = Path(self.params.get_file_path(fname))
            if orig.exists() and orig not in candidates:
                candidates.append(orig)
        except Exception:
            pass
        for path in candidates:
            try:
                # Use cached header if available
                path_key = str(path)
                if path_key in self._wcs_header_cache:
                    hdr = self._wcs_header_cache[path_key]
                else:
                    hdr = fits.getheader(path)
                    self._wcs_header_cache[path_key] = hdr
                w = WCS(hdr, relax=True)
                if w.has_celestial:
                    return w
                # fallback: ASTAP .wcs sidecar
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

    def _load_gaia_catalog(self) -> Optional[SkyCoord]:
        try:
            df = self._load_gaia_table()
            if df is None or df.empty:
                return None
            ra = pd.to_numeric(df["ra"], errors="coerce")
            dec = pd.to_numeric(df["dec"], errors="coerce")
            m = ra.notna() & dec.notna()
            if not m.any():
                return None
            return SkyCoord(ra[m].to_numpy(float) * u.deg, dec[m].to_numpy(float) * u.deg, frame="icrs")
        except Exception:
            return None

    def _load_gaia_table(self) -> Optional[pd.DataFrame]:
        gaia_path = step5_dir(self.result_dir) / "gaia_fov.ecsv"
        if not gaia_path.exists():
            gaia_path = legacy_step7_wcs_dir(self.result_dir) / "gaia_fov.ecsv"
        if not gaia_path.exists():
            return None
        try:
            tab = Table.read(str(gaia_path), format="ascii.ecsv")
            cols = [c.lower() for c in tab.colnames]
            if cols != list(tab.colnames):
                tab.rename_columns(tab.colnames, cols)
            df = tab.to_pandas()
            if "ra" not in df.columns or "dec" not in df.columns:
                return None
            return df
        except Exception:
            return None

    def _attach_gaia_photometry(self, master_df: pd.DataFrame, gaia_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        if gaia_df is None or gaia_df.empty:
            return master_df
        if "ra_deg" not in master_df.columns or "dec_deg" not in master_df.columns:
            return master_df

        gaia_ra = pd.to_numeric(gaia_df.get("ra"), errors="coerce")
        gaia_dec = pd.to_numeric(gaia_df.get("dec"), errors="coerce")
        mask = gaia_ra.notna() & gaia_dec.notna()
        if not mask.any():
            return master_df

        gaia_df = gaia_df.loc[mask].copy()
        gaia_ra = gaia_ra[mask].to_numpy(float)
        gaia_dec = gaia_dec[mask].to_numpy(float)
        gaia_sky = SkyCoord(gaia_ra * u.deg, gaia_dec * u.deg, frame="icrs")

        src_ra = pd.to_numeric(master_df["ra_deg"], errors="coerce")
        src_dec = pd.to_numeric(master_df["dec_deg"], errors="coerce")
        src_mask = src_ra.notna() & src_dec.notna()
        if not src_mask.any():
            return master_df

        src_sky = SkyCoord(src_ra[src_mask].to_numpy(float) * u.deg,
                           src_dec[src_mask].to_numpy(float) * u.deg,
                           frame="icrs")
        idx, sep2d, _ = src_sky.match_to_catalog_sky(gaia_sky)
        match_r = max(0.5, float(self.wcs_match_radius_arcsec))
        ok = sep2d.arcsec <= match_r
        if not np.any(ok):
            return master_df

        out = master_df.copy()
        out["gaia_source_id"] = np.nan
        out["gaia_ra_deg"] = np.nan
        out["gaia_dec_deg"] = np.nan
        out["phot_g_mean_mag"] = np.nan
        out["phot_bp_mean_mag"] = np.nan
        out["phot_rp_mean_mag"] = np.nan

        src_idx = np.where(src_mask)[0]
        match_idx = src_idx[ok]
        gaia_idx = idx[ok]

        if "source_id" in gaia_df.columns:
            gaia_sid = pd.to_numeric(gaia_df["source_id"], errors="coerce").to_numpy()
            out.loc[match_idx, "gaia_source_id"] = gaia_sid[gaia_idx]

        out.loc[match_idx, "gaia_ra_deg"] = gaia_ra[gaia_idx]
        out.loc[match_idx, "gaia_dec_deg"] = gaia_dec[gaia_idx]

        if "phot_g_mean_mag" in gaia_df.columns:
            g = pd.to_numeric(gaia_df["phot_g_mean_mag"], errors="coerce").to_numpy()
            out.loc[match_idx, "phot_g_mean_mag"] = g[gaia_idx]
        if "phot_bp_mean_mag" in gaia_df.columns:
            bp = pd.to_numeric(gaia_df["phot_bp_mean_mag"], errors="coerce").to_numpy()
            out.loc[match_idx, "phot_bp_mean_mag"] = bp[gaia_idx]
        if "phot_rp_mean_mag" in gaia_df.columns:
            rp = pd.to_numeric(gaia_df["phot_rp_mean_mag"], errors="coerce").to_numpy()
            out.loc[match_idx, "phot_rp_mean_mag"] = rp[gaia_idx]

        out["gaia_G"] = out["phot_g_mean_mag"]
        out["gaia_BP"] = out["phot_bp_mean_mag"]
        out["gaia_RP"] = out["phot_rp_mean_mag"]
        try:
            out["color_gr"] = pd.to_numeric(out["gaia_BP"], errors="coerce") - pd.to_numeric(out["gaia_RP"], errors="coerce")
        except Exception:
            pass
        return out

    def _apply_hybrid_source_ids(self, df: pd.DataFrame, gaia_mag_limit: float = 18.0) -> pd.DataFrame:
        """Apply hybrid source_id assignment: Gaia ID for matched sources, negative ID for non-Gaia.

        In hybrid mode:
        - Sources matched to Gaia: use gaia_source_id (positive, from Gaia DR3)
        - Sources not matched: assign negative local IDs (-1, -2, ...)

        This ensures consistent source_id across all frames for Gaia-matched sources.
        """
        out = df.copy()

        # Check if gaia_source_id column exists
        if "gaia_source_id" not in out.columns:
            self._log("[REF] No gaia_source_id column; hybrid mode not applied.")
            return out

        # Filter by magnitude limit if gaia_G is available
        if "gaia_G" in out.columns and gaia_mag_limit > 0:
            gaia_g = pd.to_numeric(out["gaia_G"], errors="coerce")
            too_faint = gaia_g > gaia_mag_limit
            # Clear Gaia ID for sources fainter than limit
            out.loc[too_faint, "gaia_source_id"] = np.nan

        # Identify sources with valid Gaia source_id
        gaia_sid = pd.to_numeric(out["gaia_source_id"], errors="coerce")
        has_gaia = gaia_sid.notna() & (gaia_sid > 0)

        # Create new source_id column
        new_source_id = out["source_id"].copy()  # Start with existing
        next_local_id = -1

        for i in out.index:
            if has_gaia[i]:
                # Use Gaia source_id (positive)
                new_source_id[i] = int(gaia_sid[i])
            else:
                # Assign negative local ID
                new_source_id[i] = next_local_id
                next_local_id -= 1

        out["source_id"] = new_source_id
        # Keep ID as sequential for display purposes
        out["ID"] = range(1, len(out) + 1)

        n_gaia = int(has_gaia.sum())
        n_local = len(out) - n_gaia
        self._log(f"[REF] Hybrid IDs assigned: {n_gaia} Gaia, {n_local} local (negative)")

        return out

    def _merge_ref_catalogs(
        self,
        base_df: Optional[pd.DataFrame],
        new_df: pd.DataFrame,
        match_radius_arcsec: float,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if base_df is None or base_df.empty:
            out = new_df.copy()
            out = out.reset_index(drop=True)
            out["source_id"] = np.arange(1, len(out) + 1, dtype=int)
            out["ID"] = out["source_id"]
            return out, out

        base = base_df.copy().reset_index(drop=True)
        new = new_df.copy().reset_index(drop=True)

        base_ra = pd.to_numeric(base.get("ra_deg"), errors="coerce")
        base_dec = pd.to_numeric(base.get("dec_deg"), errors="coerce")
        new_ra = pd.to_numeric(new.get("ra_deg"), errors="coerce")
        new_dec = pd.to_numeric(new.get("dec_deg"), errors="coerce")
        base_mask = base_ra.notna() & base_dec.notna()
        new_mask = new_ra.notna() & new_dec.notna()
        if not base_mask.any() or not new_mask.any():
            return base, new

        base_sky = SkyCoord(base_ra[base_mask].to_numpy(float) * u.deg,
                            base_dec[base_mask].to_numpy(float) * u.deg,
                            frame="icrs")
        new_sky = SkyCoord(new_ra[new_mask].to_numpy(float) * u.deg,
                           new_dec[new_mask].to_numpy(float) * u.deg,
                           frame="icrs")
        idx, sep2d, _ = new_sky.match_to_catalog_sky(base_sky)
        ok = sep2d.arcsec <= match_radius_arcsec

        new["source_id"] = np.nan
        new["ID"] = np.nan

        base_ids = pd.to_numeric(base.loc[base_mask, "source_id"], errors="coerce").to_numpy()
        match_idx = np.where(new_mask)[0]
        ok_idx = match_idx[ok]
        if len(ok_idx):
            new.loc[ok_idx, "source_id"] = base_ids[idx[ok]]
            new.loc[ok_idx, "ID"] = base_ids[idx[ok]]

        next_id = int(pd.to_numeric(base["source_id"], errors="coerce").max() or 0) + 1
        new_rows = []
        for i in match_idx[~ok]:
            sid = next_id
            next_id += 1
            new.loc[i, "source_id"] = sid
            new.loc[i, "ID"] = sid
            new_rows.append(new.loc[i:i])

        if new_rows:
            base = pd.concat([base] + new_rows, ignore_index=True)

        new["source_id"] = pd.to_numeric(new["source_id"], errors="coerce").astype("Int64")
        new["ID"] = pd.to_numeric(new["ID"], errors="coerce").astype("Int64")
        return base, new

    def _load_wcs_meta(self, fname: str) -> dict:
        meta_path = Path(self.cache_dir) / "wcs_solve" / f"wcs_{fname}.json"
        if not meta_path.exists():
            return {}
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _compute_match_stats(self, det_xy: np.ndarray, wcs: WCS, gaia_sky: SkyCoord) -> dict:
        n_det = len(det_xy)
        if n_det == 0:
            return dict(
                n_det=0,
                n_match=0,
                match_rate=0.0,
                sep_med_arcsec=np.nan,
                sep_p90_arcsec=np.nan,
                dup_rate=np.nan,
            )

        try:
            ra, dec = wcs.all_pix2world(det_xy[:, 0], det_xy[:, 1], 0)
            det_sky = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
        except Exception:
            return dict(
                n_det=n_det,
                n_match=0,
                match_rate=0.0,
                sep_med_arcsec=np.nan,
                sep_p90_arcsec=np.nan,
                dup_rate=np.nan,
            )

        idx, sep2d, _ = det_sky.match_to_catalog_sky(gaia_sky)
        sep_arcsec = sep2d.arcsec
        r_match = max(0.5, float(self.wcs_match_radius_arcsec))
        ok = sep_arcsec <= r_match
        n_match = int(np.sum(ok))
        if n_match == 0:
            return dict(
                n_det=n_det,
                n_match=0,
                match_rate=0.0,
                sep_med_arcsec=np.nan,
                sep_p90_arcsec=np.nan,
                dup_rate=np.nan,
            )

        sep_ok = sep_arcsec[ok]
        sep_med = float(np.nanmedian(sep_ok)) if len(sep_ok) else np.nan
        sep_p90 = float(np.nanpercentile(sep_ok, 90)) if len(sep_ok) else np.nan

        # duplicate matches: multiple detections -> same Gaia index
        dup_rate = np.nan
        try:
            counts = pd.Series(idx[ok]).value_counts()
            dup = counts[counts > 1].sum()
            dup_rate = float(dup / max(n_match, 1))
        except Exception:
            dup_rate = np.nan

        return dict(
            n_det=n_det,
            n_match=n_match,
            match_rate=float(n_match / max(n_det, 1)),
            sep_med_arcsec=sep_med,
            sep_p90_arcsec=sep_p90,
            dup_rate=dup_rate,
        )

    def _frame_metrics(self, fname: str) -> Optional[dict]:
        meta = self._load_meta(fname)
        if not meta:
            return None
        filt = str(meta.get("filter", "") or "").strip().lower()
        if not filt:
            filt = _get_filter_from_filename(fname) or "unknown"
        fwhm_px = _safe_float(meta.get("fwhm_px"), np.nan)
        n_sources = int(meta.get("n_sources", 0) or 0)
        sat_count = int(meta.get("sat_star_count", 0) or 0)
        med_elong = _safe_float(meta.get("median_elongation"), np.nan)
        med_round = _safe_float(meta.get("median_roundness"), np.nan)
        shape_metric = med_elong
        if not np.isfinite(shape_metric):
            shape_metric = abs(med_round) if np.isfinite(med_round) else np.nan

        # Sky background statistics (from detection metadata)
        sky_med = _safe_float(meta.get("sky_med"), np.nan)
        sky_sigma = _safe_float(meta.get("sky_sigma"), np.nan)
        # Alternative names that might be used
        if not np.isfinite(sky_med):
            sky_med = _safe_float(meta.get("bkg_median"), np.nan)
        if not np.isfinite(sky_sigma):
            sky_sigma = _safe_float(meta.get("bkg_rms"), np.nan)

        return {
            "file": fname,
            "filter": filt,
            "date_key": _extract_date_key(fname, self.params),
            "fwhm_px": fwhm_px,
            "n_sources": n_sources,
            "sat_star_count": sat_count,
            "shape_metric": shape_metric,
            "sky_med": sky_med,
            "sky_sigma": sky_sigma,
        }

    def _select_reference(self, metrics: pd.DataFrame, ref_filter: str) -> str:
        if metrics.empty:
            raise RuntimeError("No detection metrics available")

        df = metrics.copy()
        filt = ref_filter.strip().lower()
        if filt:
            cand = df[df["filter"] == filt].copy()
            if cand.empty:
                self._log(f"[REF][QC] Filter '{filt}' not found; using all filters.")
                cand = df.copy()
        else:
            cand = df.copy()

        def _drop_top_percent(sub: pd.DataFrame, col: str, pct: float) -> pd.DataFrame:
            if sub.empty or pct <= 0:
                return sub
            vals = pd.to_numeric(sub[col], errors="coerce")
            if vals.notna().sum() == 0:
                return sub
            n = len(sub)
            drop_n = int(np.ceil(n * pct / 100.0))
            drop_n = max(0, min(n - 1, drop_n))
            if drop_n == 0:
                return sub
            sub = sub.copy()
            sub["_metric"] = vals
            sub = sub.sort_values("_metric", ascending=False)
            kept = sub.iloc[drop_n:]
            return kept.drop(columns=["_metric"])

        n_start = len(cand)
        cand = _drop_top_percent(cand, "sat_star_count", self.sat_drop_pct)
        n_sat = len(cand)
        cand = _drop_top_percent(cand, "shape_metric", self.elong_drop_pct)
        n_shape = len(cand)
        if cand.empty:
            self._log("[REF][QC] All candidates dropped by sat/shape filters; using full set.")
            cand = df.copy()

        cand = cand.copy()
        cand["fwhm_px"] = pd.to_numeric(cand["fwhm_px"], errors="coerce")
        cand["shape_metric"] = pd.to_numeric(cand["shape_metric"], errors="coerce")
        cand["sat_star_count"] = pd.to_numeric(cand["sat_star_count"], errors="coerce")
        cand["n_sources"] = pd.to_numeric(cand["n_sources"], errors="coerce")

        # Apply WCS match quality filters when available
        if "match_rate" in cand.columns:
            cand["match_rate"] = pd.to_numeric(cand["match_rate"], errors="coerce")
        if "n_match" in cand.columns:
            cand["n_match"] = pd.to_numeric(cand["n_match"], errors="coerce")
        if "sep_med_arcsec" in cand.columns:
            cand["sep_med_arcsec"] = pd.to_numeric(cand["sep_med_arcsec"], errors="coerce")
        if "sep_p90_arcsec" in cand.columns:
            cand["sep_p90_arcsec"] = pd.to_numeric(cand["sep_p90_arcsec"], errors="coerce")
        if "dup_rate" in cand.columns:
            cand["dup_rate"] = pd.to_numeric(cand["dup_rate"], errors="coerce")

        cand_wcs = cand.copy()
        if "wcs_ok" in cand_wcs.columns:
            cand_wcs = cand_wcs[cand_wcs["wcs_ok"] == True]
        if "match_rate" in cand_wcs.columns and self.wcs_min_match_rate > 0:
            if cand_wcs["match_rate"].notna().any():
                cand_wcs = cand_wcs[cand_wcs["match_rate"] >= self.wcs_min_match_rate]
        if "n_match" in cand_wcs.columns and self.wcs_min_match_n > 0:
            if cand_wcs["n_match"].notna().any():
                cand_wcs = cand_wcs[cand_wcs["n_match"] >= self.wcs_min_match_n]
        if "sep_med_arcsec" in cand_wcs.columns and np.isfinite(self.wcs_max_sep_med_arcsec) and self.wcs_max_sep_med_arcsec > 0:
            if cand_wcs["sep_med_arcsec"].notna().any():
                cand_wcs = cand_wcs[cand_wcs["sep_med_arcsec"] <= self.wcs_max_sep_med_arcsec]
        if "sep_p90_arcsec" in cand_wcs.columns and np.isfinite(self.wcs_max_sep_p90_arcsec) and self.wcs_max_sep_p90_arcsec > 0:
            if cand_wcs["sep_p90_arcsec"].notna().any():
                cand_wcs = cand_wcs[cand_wcs["sep_p90_arcsec"] <= self.wcs_max_sep_p90_arcsec]
        if "dup_rate" in cand_wcs.columns and np.isfinite(self.wcs_max_dup_rate) and self.wcs_max_dup_rate > 0:
            if cand_wcs["dup_rate"].notna().any():
                cand_wcs = cand_wcs[cand_wcs["dup_rate"] <= self.wcs_max_dup_rate]

        n_wcs = len(cand_wcs)
        if not cand_wcs.empty:
            cand = cand_wcs
        self._log(
            "[REF][QC] candidates: start={s} sat_drop={sat:.1f}% -> {n1} "
            "shape_drop={elong:.1f}% -> {n2} wcs_pass -> {n3}".format(
                s=n_start,
                sat=self.sat_drop_pct,
                n1=n_sat,
                elong=self.elong_drop_pct,
                n2=n_shape,
                n3=n_wcs,
            )
        )
        self._log(
            "[REF][QC] wcs thresholds: match_r={r:.2f}\" min_rate={mr:.2f} "
            "min_match={mn} max_sep_med={smed:.2f}\" max_sep_p90={sp90:.2f}\" max_dup={dup:.2f}".format(
                r=self.wcs_match_radius_arcsec,
                mr=self.wcs_min_match_rate,
                mn=self.wcs_min_match_n,
                smed=self.wcs_max_sep_med_arcsec,
                sp90=self.wcs_max_sep_p90_arcsec,
                dup=self.wcs_max_dup_rate,
            )
        )

        sort_cols = []
        sort_asc = []
        if "match_rate" in cand.columns:
            sort_cols.append("match_rate")
            sort_asc.append(False)
        if "sep_med_arcsec" in cand.columns:
            sort_cols.append("sep_med_arcsec")
            sort_asc.append(True)
        sort_cols.extend(["fwhm_px", "shape_metric", "sat_star_count", "n_sources"])
        sort_asc.extend([True, True, True, False])

        cand = cand.sort_values(
            sort_cols,
            ascending=sort_asc,
            na_position="last",
        )
        try:
            top = cand.head(5)
            for _, row in top.iterrows():
                self._log(
                    "[REF][QC] cand {file} f={flt} mr={mr} sep={sep} fwhm={fwhm} "
                    "shape={shape} sat={sat} n={n} wcs_ok={wcs} match_n={mn}".format(
                        file=row.get("file", ""),
                        flt=row.get("filter", ""),
                        mr=_safe_float(row.get("match_rate"), np.nan),
                        sep=_safe_float(row.get("sep_med_arcsec"), np.nan),
                        fwhm=_safe_float(row.get("fwhm_px"), np.nan),
                        shape=_safe_float(row.get("shape_metric"), np.nan),
                        sat=_safe_float(row.get("sat_star_count"), np.nan),
                        n=_safe_float(row.get("n_sources"), np.nan),
                        wcs=bool(row.get("wcs_ok", False)),
                        mn=int(row.get("n_match", 0) or 0),
                    )
                )
        except Exception:
            pass
        return str(cand.iloc[0]["file"])

    def _build_master_catalog(self, ref_fname: str) -> tuple[pd.DataFrame, dict]:
        """Build master catalog from reference frame detections.

        Returns:
            Tuple of (catalog DataFrame, stats dict with n_ref_total/after_cuts/used)
        """
        det_path = self._resolve_detect_csv(ref_fname)
        if det_path is None:
            raise RuntimeError(f"Missing detection file: detect_{ref_fname}.csv")
        df = pd.read_csv(det_path)
        if not {"x", "y"} <= set(df.columns):
            raise RuntimeError(f"Detection file missing x/y: {det_path}")

        df = df.copy()
        for col in (
            "x",
            "y",
            "elongation",
            "roundness",
            "sharpness",
            "dao_flux",
            "dao_peak",
            "peak_adu",
            "fwhm_px",
        ):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        base_all = df[["x", "y"]].rename(columns={"x": "x_ref", "y": "y_ref"})
        base_all = base_all.dropna(subset=["x_ref", "y_ref"]).copy()
        if base_all.empty:
            raise RuntimeError("Reference detection list is empty")

        # Track n_ref_total
        n_ref_total = len(base_all)

        cand = df.dropna(subset=["x", "y"]).copy()
        n_before = len(cand)

        if np.isfinite(self.ref_cat_max_elong) and self.ref_cat_max_elong > 0 and "elongation" in cand.columns:
            cand = cand[cand["elongation"] <= self.ref_cat_max_elong]

        if np.isfinite(self.ref_cat_max_abs_round) and self.ref_cat_max_abs_round > 0 and "roundness" in cand.columns:
            cand = cand[cand["roundness"].abs() <= self.ref_cat_max_abs_round]

        if "sharpness" in cand.columns:
            if np.isfinite(self.ref_cat_sharp_min):
                cand = cand[cand["sharpness"] >= self.ref_cat_sharp_min]
            if np.isfinite(self.ref_cat_sharp_max) and self.ref_cat_sharp_max > 0:
                cand = cand[cand["sharpness"] <= self.ref_cat_sharp_max]

        # Track n_ref_after_qualitycuts (after shape/quality filters, before brightness)
        n_ref_after_qualitycuts = len(cand)

        brightness = None
        for col in ("dao_flux", "dao_peak", "peak_adu"):
            if col in cand.columns:
                brightness = cand[col]
                break
        if brightness is not None:
            cand = cand.copy()
            cand["brightness"] = brightness
            if np.isfinite(self.ref_cat_min_peak_adu) and self.ref_cat_min_peak_adu > 0:
                cand = cand[cand["brightness"] >= self.ref_cat_min_peak_adu]
            if self.ref_cat_max_sources > 0:
                cand = cand.sort_values("brightness", ascending=False)
                cand = cand.head(self.ref_cat_max_sources)

        base = cand[["x", "y"]].rename(columns={"x": "x_ref", "y": "y_ref"})
        base = base.dropna(subset=["x_ref", "y_ref"]).copy()

        used_full_detections = False
        if base.empty or len(base) < max(10, self.ref_cat_min_sources):
            self._log(
                f"[REF] Ref catalog filter too strict "
                f"({len(base)}/{n_before}); using full detections."
            )
            base = base_all.copy()
            used_full_detections = True
        else:
            self._log(
                f"[REF] Ref catalog filtered: {len(base)}/{n_before} sources kept."
            )

        # Track n_ref_used (final count)
        n_ref_used = len(base)

        wcs = self._load_wcs_for_frame(ref_fname)
        if wcs is None:
            raise RuntimeError(f"No WCS for reference frame: {ref_fname}")
        try:
            ra, dec = wcs.all_pix2world(base["x_ref"].to_numpy(float), base["y_ref"].to_numpy(float), 0)
        except Exception as e:
            raise RuntimeError(f"WCS conversion failed for {ref_fname}: {e}")

        base = base.sort_values(["y_ref", "x_ref"]).reset_index(drop=True)
        base["source_id"] = range(1, len(base) + 1)
        base["ID"] = range(1, len(base) + 1)
        base["ra_deg"] = ra
        base["dec_deg"] = dec

        # Log ref catalog stats
        self._log(
            f"[REF][STATS] n_ref_total={n_ref_total} n_ref_after_qualitycuts={n_ref_after_qualitycuts} "
            f"n_ref_used={n_ref_used} used_full={used_full_detections}"
        )

        ref_stats = {
            "n_ref_total": n_ref_total,
            "n_ref_after_qualitycuts": n_ref_after_qualitycuts,
            "n_ref_used": n_ref_used,
            "used_full_detections": used_full_detections,
        }

        return base[["ID", "source_id", "ra_deg", "dec_deg", "x_ref", "y_ref"]].copy(), ref_stats

    def run(self):
        try:
            self._run_impl()
        except Exception as e:
            import traceback
            self._log(f"[ERROR] {e}\n{traceback.format_exc()}")
            self.error.emit("WORKER", str(e))
            self.finished.emit({})

    def _run_impl(self):
        if not self.file_list:
            raise RuntimeError("No frames available")

        metrics_rows = []
        total = len(self.file_list)
        for i, fname in enumerate(self.file_list, 1):
            if self._stop_requested:
                return
            row = self._frame_metrics(fname)
            if row:
                metrics_rows.append(row)
            self.progress.emit(i, total, fname)

        if not metrics_rows:
            raise RuntimeError("No detection metrics found. Run Source Detection first.")

        metrics = pd.DataFrame(metrics_rows)

        gaia_df = self._load_gaia_table()
        gaia_sky = self._load_gaia_catalog()
        if gaia_sky is None:
            self._log("[REF] Gaia catalog not available; WCS match stats will be skipped.")
        else:
            self._log(f"[REF] Gaia catalog loaded: {len(gaia_sky)} sources")

        stats_rows = []
        for row in metrics.to_dict(orient="records"):
            fname = row["file"]
            det_path = self._resolve_detect_csv(fname)
            det_xy = np.zeros((0, 2), float)
            if det_path is not None and det_path.exists():
                try:
                    df_det = pd.read_csv(det_path)
                    if {"x", "y"} <= set(df_det.columns):
                        det_xy = df_det[["x", "y"]].to_numpy(float)
                        det_xy = det_xy[np.isfinite(det_xy).all(axis=1)]
                except Exception:
                    det_xy = np.zeros((0, 2), float)

            wcs = self._load_wcs_for_frame(fname)
            row["wcs_ok"] = bool(wcs is not None)
            wcs_meta = self._load_wcs_meta(fname)
            row["wcs_resid_med"] = _safe_float(wcs_meta.get("resid_med"), np.nan)
            row["wcs_resid_max"] = _safe_float(wcs_meta.get("resid_max"), np.nan)
            row["wcs_match_n"] = int(wcs_meta.get("match_n", 0) or 0)

            if wcs is None or gaia_sky is None:
                row.update(
                    dict(
                        n_match=0,
                        match_rate=np.nan,
                        sep_med_arcsec=np.nan,
                        sep_p90_arcsec=np.nan,
                        dup_rate=np.nan,
                    )
                )
            else:
                row.update(self._compute_match_stats(det_xy, wcs, gaia_sky))

            stats_rows.append(row)

        metrics = pd.DataFrame(stats_rows)

        try:
            self._log(f"[REF][QC] total frames={len(metrics)}")
            if "filter" in metrics.columns:
                counts = metrics["filter"].value_counts(dropna=False)
                parts = [f"{k}:{v}" for k, v in counts.items()]
                self._log(f"[REF][QC] filter counts: {', '.join(parts)}")
            if "wcs_ok" in metrics.columns:
                wcs_ok = int(metrics["wcs_ok"].fillna(False).astype(bool).sum())
                self._log(f"[REF][QC] wcs_ok={wcs_ok}/{len(metrics)}")
            if "match_rate" in metrics.columns and metrics["match_rate"].notna().any():
                mr = metrics["match_rate"].median()
                self._log(f"[REF][QC] match_rate median={mr:.3f}")
            if "sep_med_arcsec" in metrics.columns and metrics["sep_med_arcsec"].notna().any():
                sm = metrics["sep_med_arcsec"].median()
                self._log(f"[REF][QC] sep_med_arcsec median={sm:.3f}")
            if "fwhm_px" in metrics.columns and metrics["fwhm_px"].notna().any():
                fmed = metrics["fwhm_px"].median()
                self._log(f"[REF][QC] fwhm_px median={fmed:.3f}")
        except Exception:
            pass

        ref_frames_by_date: Dict[str, str] = {}
        ref_filters_by_date: Dict[str, str] = {}
        ref_catalogs_by_date: Dict[str, pd.DataFrame] = {}
        master_df: Optional[pd.DataFrame] = None
        ref_catalog_stats: dict = {}  # Track ref catalog build stats

        match_r = max(0.5, float(self.wcs_match_radius_arcsec))
        if self.ref_per_date:
            metrics["date_key"] = metrics.get(
                "date_key",
                metrics["file"].apply(lambda x: _extract_date_key(x, self.params)),
            )
            for date_key, group in metrics.groupby("date_key", dropna=False):
                group = group.copy()
                ref_fname_date = self._select_reference(group, self.ref_filter)
                ref_filter_date = str(group.loc[group["file"] == ref_fname_date, "filter"].iloc[0])
                ref_frames_by_date[str(date_key)] = ref_fname_date
                ref_filters_by_date[str(date_key)] = ref_filter_date
                self._log(f"[REF][QC] date={date_key} ref={ref_fname_date} (filter={ref_filter_date})")

                date_df, date_ref_stats = self._build_master_catalog(ref_fname_date)
                date_df = self._attach_gaia_photometry(date_df, gaia_df)
                if ref_catalogs_by_date:
                    master_df, date_df = self._merge_ref_catalogs(
                        master_df, date_df, match_r
                    )
                else:
                    master_df, date_df = self._merge_ref_catalogs(
                        None, date_df, match_r
                    )
                ref_catalogs_by_date[str(date_key)] = date_df

        ref_fname = self._select_reference(metrics, self.ref_filter)
        ref_filter = str(metrics.loc[metrics["file"] == ref_fname, "filter"].iloc[0])

        self._log("=" * 60)
        self._log(f"[REF] Selected reference frame: {ref_fname} (filter={ref_filter})")

        if not self.ref_per_date:
            master_df, ref_catalog_stats = self._build_master_catalog(ref_fname)
            master_df = self._attach_gaia_photometry(master_df, gaia_df)

        # Apply hybrid source_id assignment if mode is "hybrid"
        if self.ref_build_mode == "hybrid":
            master_df = self._apply_hybrid_source_ids(master_df, self.gaia_mag_limit)
            # Also apply to date catalogs if ref_per_date
            if self.ref_per_date:
                for date_key in ref_catalogs_by_date:
                    ref_catalogs_by_date[date_key] = self._apply_hybrid_source_ids(
                        ref_catalogs_by_date[date_key], self.gaia_mag_limit
                    )

        if "phot_g_mean_mag" in master_df.columns:
            try:
                n_g = int(pd.to_numeric(master_df["phot_g_mean_mag"], errors="coerce").notna().sum())
                self._log(f"[REF] Gaia photometry attached: {n_g}/{len(master_df)}")
            except Exception:
                pass

        out_dir = step6_dir(self.result_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        filters = sorted(metrics["filter"].dropna().astype(str).unique().tolist())
        if not filters:
            filters = [ref_filter]

        for flt in filters:
            out_path = out_dir / f"ref_catalog_{flt}.tsv"
            master_df.to_csv(out_path, sep="\t", index=False, na_rep="NaN", encoding="utf-8-sig")
            map_path = out_dir / f"sourceid_to_ID_{flt}.csv"
            master_df[["source_id", "ID"]].to_csv(map_path, index=False, encoding="utf-8-sig")
            if self.ref_per_date:
                for date_key, date_df in ref_catalogs_by_date.items():
                    date_path = out_dir / f"ref_catalog_{flt}_{date_key}.tsv"
                    date_df.to_csv(date_path, sep="\t", index=False, na_rep="NaN", encoding="utf-8-sig")
                    date_map = out_dir / f"sourceid_to_ID_{flt}_{date_key}.csv"
                    date_df[["source_id", "ID"]].to_csv(date_map, index=False, encoding="utf-8-sig")

        # Default (no-filter) copies for downstream compatibility
        master_df.to_csv(
            out_dir / "ref_catalog.tsv",
            sep="\t",
            index=False,
            na_rep="NaN",
            encoding="utf-8-sig",
        )
        master_df[["source_id", "ID"]].to_csv(
            out_dir / "sourceid_to_ID.csv", index=False, encoding="utf-8-sig"
        )

        metrics["selected"] = metrics["file"] == ref_fname
        if self.ref_per_date:
            metrics["date_key"] = metrics.get(
                "date_key",
                metrics["file"].apply(lambda x: _extract_date_key(x, self.params)),
            )
            metrics["selected_date"] = metrics.apply(
                lambda r: r["file"] == ref_frames_by_date.get(str(r.get("date_key")), ""),
                axis=1
            )
        metrics_path = out_dir / "ref_frame_stats.csv"
        metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")

        # Compute sky statistics summary from frame metrics
        sky_med_median = np.nan
        sky_sigma_median = np.nan
        if "sky_med" in metrics.columns:
            sky_med_vals = pd.to_numeric(metrics["sky_med"], errors="coerce")
            if sky_med_vals.notna().any():
                sky_med_median = float(sky_med_vals.median())
        if "sky_sigma" in metrics.columns:
            sky_sigma_vals = pd.to_numeric(metrics["sky_sigma"], errors="coerce")
            if sky_sigma_vals.notna().any():
                sky_sigma_median = float(sky_sigma_vals.median())

        meta = {
            "ref_frame": ref_fname,
            "ref_filter": ref_filter,
            "ref_per_date": bool(self.ref_per_date),
            "ref_frames_by_date": ref_frames_by_date,
            "ref_filters_by_date": ref_filters_by_date,
            "sat_drop_pct": float(self.sat_drop_pct),
            "elong_drop_pct": float(self.elong_drop_pct),
            "ref_cat_max_sources": int(self.ref_cat_max_sources),
            "ref_cat_min_sources": int(self.ref_cat_min_sources),
            "ref_cat_max_elong": float(self.ref_cat_max_elong),
            "ref_cat_max_abs_round": float(self.ref_cat_max_abs_round),
            "ref_cat_sharp_min": float(self.ref_cat_sharp_min),
            "ref_cat_sharp_max": float(self.ref_cat_sharp_max),
            "ref_cat_min_peak_adu": float(self.ref_cat_min_peak_adu),
            "wcs_match_radius_arcsec": float(self.wcs_match_radius_arcsec),
            "wcs_min_match_rate": float(self.wcs_min_match_rate),
            "wcs_min_match_n": int(self.wcs_min_match_n),
            "wcs_max_sep_med_arcsec": float(self.wcs_max_sep_med_arcsec),
            "wcs_max_sep_p90_arcsec": float(self.wcs_max_sep_p90_arcsec),
            "wcs_max_dup_rate": float(self.wcs_max_dup_rate),
            "filters": filters,
            # Reference catalog statistics
            "n_ref_total": ref_catalog_stats.get("n_ref_total"),
            "n_ref_after_qualitycuts": ref_catalog_stats.get("n_ref_after_qualitycuts"),
            "n_ref_used": ref_catalog_stats.get("n_ref_used"),
            "used_full_detections": ref_catalog_stats.get("used_full_detections"),
            # Sky statistics (median across frames)
            "sky_med_median": sky_med_median if np.isfinite(sky_med_median) else None,
            "sky_sigma_median": sky_sigma_median if np.isfinite(sky_sigma_median) else None,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        (out_dir / "ref_build_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

        self.finished.emit({
            "ref_frame": ref_fname,
            "ref_filter": ref_filter,
            "n_sources": len(master_df),
            "filters": filters,
            "ref_per_date": bool(self.ref_per_date),
            "ref_frames_by_date": ref_frames_by_date,
        })


class RefBuildWindow(StepWindowBase):
    """Step 6: Reference Build (WCS-based)"""

    def __init__(self, params, file_manager, project_state, main_window):
        self.file_manager = file_manager
        self.worker = None
        self.log_window = None
        self.results = {}

        super().__init__(
            step_index=5,
            step_name="Reference Build",
            params=params,
            project_state=project_state,
            main_window=main_window,
        )

        self.setup_step_ui()
        self.restore_state()

    def setup_step_ui(self):
        info = QLabel(
            "Build a fixed reference catalog using WCS-solved frames.\n"
            "Selection prefers good WCS match stats, then saturation/elongation/FWHM."
        )
        info.setStyleSheet("QLabel { background-color: #E3F2FD; padding: 10px; border-radius: 5px; }")
        info.setWordWrap(True)
        self.content_layout.addWidget(info)

        status_group = QGroupBox("WCS/Detection Status")
        status_layout = QVBoxLayout(status_group)
        self.status_label = QLabel("Checking...")
        status_layout.addWidget(self.status_label)
        self.content_layout.addWidget(status_group)

        control_layout = QHBoxLayout()

        btn_params = QPushButton("Reference Parameters")
        btn_params.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; font-weight: bold; padding: 8px 15px; }"
        )
        btn_params.clicked.connect(self.open_parameters_dialog)
        control_layout.addWidget(btn_params)

        self.btn_run = QPushButton("Run Reference Build")
        self.btn_run.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px 20px; }"
        )
        self.btn_run.clicked.connect(self.run_ref_build)
        control_layout.addWidget(self.btn_run)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setStyleSheet(
            "QPushButton { background-color: #f44336; color: white; font-weight: bold; padding: 8px 15px; }"
        )
        self.btn_stop.clicked.connect(self.stop_ref_build)
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

        summary_tab = QWidget()
        summary_layout = QVBoxLayout(summary_tab)
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(6)
        self.results_table.setHorizontalHeaderLabels([
            "Date", "Filter", "Ref Frame", "Sources", "FWHM (px)", "Sat Count"
        ])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.results_table.setMinimumHeight(120)
        summary_layout.addWidget(self.results_table)
        self.tabs.addTab(summary_tab, "Summary")

        stats_tab = QWidget()
        stats_layout = QVBoxLayout(stats_tab)
        self.stats_table = QTableWidget()
        self.stats_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.stats_table.setMinimumHeight(220)
        stats_layout.addWidget(self.stats_table)
        self.tabs.addTab(stats_tab, "Stats")

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
        self.log_window.setWindowTitle("Reference Build Log")
        self.log_window.resize(900, 500)
        log_layout = QVBoxLayout(self.log_window)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("QTextEdit { font-family: monospace; font-size: 9pt; }")
        log_layout.addWidget(self.log_text)

        self.check_detection_status()

    def check_detection_status(self):
        cache_dir = Path(self.params.P.cache_dir)
        metas = list(cache_dir.glob("detect_*.json"))
        if not metas:
            step4_out = step4_dir(self.params.P.result_dir)
            metas = list(step4_out.glob("detect_*.json"))
        if not metas:
            self.status_label.setText("No detection cache found. Run Source Detection first.")
            self.status_label.setStyleSheet("color: red;")
            return
        self.status_label.setText(f"Detection cache found: {len(metas)} frames")
        self.status_label.setStyleSheet("color: green;")

    def open_parameters_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Reference Build Parameters")
        dialog.resize(460, 350)

        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        layout.addLayout(form)

        sat_spin = QDoubleSpinBox()
        sat_spin.setRange(0.0, 100.0)
        sat_spin.setDecimals(1)
        sat_spin.setSuffix("%")
        sat_spin.setValue(float(getattr(self.params.P, "ref_select_sat_pct", 20.0)))
        form.addRow("Drop top saturation frames:", sat_spin)

        elong_spin = QDoubleSpinBox()
        elong_spin.setRange(0.0, 100.0)
        elong_spin.setDecimals(1)
        elong_spin.setSuffix("%")
        elong_spin.setValue(float(getattr(self.params.P, "ref_select_elong_pct", 20.0)))
        form.addRow("Drop top elongation frames:", elong_spin)

        per_date_check = QCheckBox("Per-date reference (default)")
        per_date_check.setChecked(bool(getattr(self.params.P, "ref_per_date", True)))
        form.addRow("Per-date reference:", per_date_check)

        max_src_spin = QSpinBox()
        max_src_spin.setRange(0, 50000)
        max_src_spin.setValue(int(getattr(self.params.P, "ref_cat_max_sources", 0)))
        form.addRow("Ref catalog max sources (0=all):", max_src_spin)

        min_src_spin = QSpinBox()
        min_src_spin.setRange(0, 50000)
        min_src_spin.setValue(int(getattr(self.params.P, "ref_cat_min_sources", 50)))
        form.addRow("Ref catalog min sources:", min_src_spin)

        max_elong_spin = QDoubleSpinBox()
        max_elong_spin.setRange(0.0, 10.0)
        max_elong_spin.setDecimals(2)
        max_elong_spin.setValue(float(getattr(self.params.P, "ref_cat_max_elong", 1.5)))
        form.addRow("Ref max elongation:", max_elong_spin)

        max_round_spin = QDoubleSpinBox()
        max_round_spin.setRange(0.0, 5.0)
        max_round_spin.setDecimals(2)
        max_round_spin.setValue(float(getattr(self.params.P, "ref_cat_max_abs_round", 0.4)))
        form.addRow("Ref max |roundness|:", max_round_spin)

        sharp_min_spin = QDoubleSpinBox()
        sharp_min_spin.setRange(-5.0, 5.0)
        sharp_min_spin.setDecimals(2)
        sharp_min_spin.setValue(float(getattr(self.params.P, "ref_cat_sharp_min", 0.2)))
        form.addRow("Ref sharpness min:", sharp_min_spin)

        sharp_max_spin = QDoubleSpinBox()
        sharp_max_spin.setRange(-5.0, 5.0)
        sharp_max_spin.setDecimals(2)
        sharp_max_spin.setValue(float(getattr(self.params.P, "ref_cat_sharp_max", 1.0)))
        form.addRow("Ref sharpness max:", sharp_max_spin)

        peak_spin = QDoubleSpinBox()
        peak_spin.setRange(0.0, 1e9)
        peak_spin.setDecimals(1)
        peak_spin.setValue(float(getattr(self.params.P, "ref_cat_min_peak_adu", 0.0)))
        form.addRow("Ref min peak/flux (0=off):", peak_spin)

        match_r_spin = QDoubleSpinBox()
        match_r_spin.setRange(0.1, 30.0)
        match_r_spin.setDecimals(2)
        match_r_spin.setValue(float(getattr(self.params.P, "ref_wcs_match_radius_arcsec", 2.0)))
        form.addRow("WCS match radius (arcsec):", match_r_spin)

        min_rate_spin = QDoubleSpinBox()
        min_rate_spin.setRange(0.0, 1.0)
        min_rate_spin.setDecimals(2)
        min_rate_spin.setSingleStep(0.05)
        min_rate_spin.setValue(float(getattr(self.params.P, "ref_wcs_min_match_rate", 0.2)))
        form.addRow("WCS min match rate:", min_rate_spin)

        min_match_spin = QSpinBox()
        min_match_spin.setRange(0, 100000)
        min_match_spin.setValue(int(getattr(self.params.P, "ref_wcs_min_match_n", 50)))
        form.addRow("WCS min match count:", min_match_spin)

        max_sep_med_spin = QDoubleSpinBox()
        max_sep_med_spin.setRange(0.0, 30.0)
        max_sep_med_spin.setDecimals(2)
        max_sep_med_spin.setValue(float(getattr(self.params.P, "ref_wcs_max_sep_med_arcsec", 1.5)))
        form.addRow("WCS max sep median (arcsec):", max_sep_med_spin)

        max_sep_p90_spin = QDoubleSpinBox()
        max_sep_p90_spin.setRange(0.0, 60.0)
        max_sep_p90_spin.setDecimals(2)
        max_sep_p90_spin.setValue(float(getattr(self.params.P, "ref_wcs_max_sep_p90_arcsec", 2.5)))
        form.addRow("WCS max sep p90 (arcsec):", max_sep_p90_spin)

        max_dup_spin = QDoubleSpinBox()
        max_dup_spin.setRange(0.0, 1.0)
        max_dup_spin.setDecimals(2)
        max_dup_spin.setSingleStep(0.05)
        max_dup_spin.setValue(float(getattr(self.params.P, "ref_wcs_max_dup_rate", 0.1)))
        form.addRow("WCS max duplicate rate:", max_dup_spin)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec_() == QDialog.Accepted:
            self.params.P.ref_select_sat_pct = sat_spin.value()
            self.params.P.ref_select_elong_pct = elong_spin.value()
            self.params.P.ref_per_date = per_date_check.isChecked()
            self.params.P.ref_cat_max_sources = max_src_spin.value()
            self.params.P.ref_cat_min_sources = min_src_spin.value()
            self.params.P.ref_cat_max_elong = max_elong_spin.value()
            self.params.P.ref_cat_max_abs_round = max_round_spin.value()
            self.params.P.ref_cat_sharp_min = sharp_min_spin.value()
            self.params.P.ref_cat_sharp_max = sharp_max_spin.value()
            self.params.P.ref_cat_min_peak_adu = peak_spin.value()
            self.params.P.ref_wcs_match_radius_arcsec = match_r_spin.value()
            self.params.P.ref_wcs_min_match_rate = min_rate_spin.value()
            self.params.P.ref_wcs_min_match_n = min_match_spin.value()
            self.params.P.ref_wcs_max_sep_med_arcsec = max_sep_med_spin.value()
            self.params.P.ref_wcs_max_sep_p90_arcsec = max_sep_p90_spin.value()
            self.params.P.ref_wcs_max_dup_rate = max_dup_spin.value()
            self.persist_params()
            QMessageBox.information(dialog, "Success", "Parameters saved!")

    def run_ref_build(self):
        if self.worker and self.worker.isRunning():
            return
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

        ref_filter = str(getattr(self.params.P, "global_ref_filter", "") or "").strip().lower()
        self.params.P.ref_select_sat_pct = float(getattr(self.params.P, "ref_select_sat_pct", 20.0))
        self.params.P.ref_select_elong_pct = float(getattr(self.params.P, "ref_select_elong_pct", 20.0))
        self.params.P.ref_cat_max_sources = int(getattr(self.params.P, "ref_cat_max_sources", 0))
        self.params.P.ref_cat_min_sources = int(getattr(self.params.P, "ref_cat_min_sources", 50))
        self.params.P.ref_cat_max_elong = float(getattr(self.params.P, "ref_cat_max_elong", 1.5))
        self.params.P.ref_cat_max_abs_round = float(getattr(self.params.P, "ref_cat_max_abs_round", 0.4))
        self.params.P.ref_cat_sharp_min = float(getattr(self.params.P, "ref_cat_sharp_min", 0.2))
        self.params.P.ref_cat_sharp_max = float(getattr(self.params.P, "ref_cat_sharp_max", 1.0))
        self.params.P.ref_cat_min_peak_adu = float(getattr(self.params.P, "ref_cat_min_peak_adu", 0.0))
        self.params.P.ref_wcs_match_radius_arcsec = float(getattr(self.params.P, "ref_wcs_match_radius_arcsec", 2.0))
        self.params.P.ref_wcs_min_match_rate = float(getattr(self.params.P, "ref_wcs_min_match_rate", 0.2))
        self.params.P.ref_wcs_min_match_n = int(getattr(self.params.P, "ref_wcs_min_match_n", 50))
        self.params.P.ref_wcs_max_sep_med_arcsec = float(getattr(self.params.P, "ref_wcs_max_sep_med_arcsec", 1.5))
        self.params.P.ref_wcs_max_sep_p90_arcsec = float(getattr(self.params.P, "ref_wcs_max_sep_p90_arcsec", 2.5))
        self.params.P.ref_wcs_max_dup_rate = float(getattr(self.params.P, "ref_wcs_max_dup_rate", 0.1))
        self.params.P.ref_per_date = bool(getattr(self.params.P, "ref_per_date", True))

        self.worker = RefBuildWorker(
            params=self.params,
            data_dir=self.params.P.data_dir,
            result_dir=self.params.P.result_dir,
            cache_dir=self.params.P.cache_dir,
            file_list=files,
            ref_filter=ref_filter,
            sat_drop_pct=self.params.P.ref_select_sat_pct,
            elong_drop_pct=self.params.P.ref_select_elong_pct,
            ref_cat_max_sources=self.params.P.ref_cat_max_sources,
            ref_cat_min_sources=self.params.P.ref_cat_min_sources,
            ref_cat_max_elong=self.params.P.ref_cat_max_elong,
            ref_cat_max_abs_round=self.params.P.ref_cat_max_abs_round,
            ref_cat_sharp_min=self.params.P.ref_cat_sharp_min,
            ref_cat_sharp_max=self.params.P.ref_cat_sharp_max,
            ref_cat_min_peak_adu=self.params.P.ref_cat_min_peak_adu,
            wcs_match_radius_arcsec=self.params.P.ref_wcs_match_radius_arcsec,
            wcs_min_match_rate=self.params.P.ref_wcs_min_match_rate,
            wcs_min_match_n=self.params.P.ref_wcs_min_match_n,
            wcs_max_sep_med_arcsec=self.params.P.ref_wcs_max_sep_med_arcsec,
            wcs_max_sep_p90_arcsec=self.params.P.ref_wcs_max_sep_p90_arcsec,
            wcs_max_dup_rate=self.params.P.ref_wcs_max_dup_rate,
            ref_per_date=self.params.P.ref_per_date,
            ref_build_mode=getattr(self.params.P, "ref_build_mode", "hybrid"),
            gaia_mag_limit=getattr(self.params.P, "gaia_mag_limit", 18.0),
        )
        self.worker.progress.connect(self.on_progress)
        self.worker.log.connect(self.log)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)

        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(len(files))
        self.progress_label.setText(f"0/{len(files)} | Starting...")
        self.worker.start()
        self.show_log_window()

    def stop_ref_build(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()

    def on_progress(self, current, total, filename):
        self.progress_bar.setValue(current)
        self.progress_label.setText(f"{current}/{total} | {filename}")

    def on_finished(self, summary: dict):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_label.setText("Done")
        if summary:
            self.results = summary
            self._update_results_table(summary)
            self._update_stats_table()
            self._update_plot_tab(summary)
            self.save_state(summary)
        self.update_navigation_buttons()

    def on_error(self, filename, error):
        self.log(f"ERROR {filename}: {error}")

    def _update_results_table(self, summary: dict):
        self.results_table.setRowCount(0)
        ref_frame = summary.get("ref_frame", "")
        ref_filter = summary.get("ref_filter", "")
        n_sources = summary.get("n_sources", 0)
        ref_frames_by_date = summary.get("ref_frames_by_date", {}) or {}

        metrics_path = step6_dir(self.params.P.result_dir) / "ref_frame_stats.csv"
        metrics_df = None
        if metrics_path.exists():
            try:
                metrics_df = pd.read_csv(metrics_path)
            except Exception:
                metrics_df = None

        def _row_stats(fname: str):
            fwhm = "-"
            sat = "-"
            if metrics_df is not None and fname:
                row = metrics_df[metrics_df["file"] == fname]
                if not row.empty:
                    fwhm = f"{float(row.iloc[0].get('fwhm_px', np.nan)):.2f}" if np.isfinite(row.iloc[0].get('fwhm_px', np.nan)) else "-"
                    sat = str(int(row.iloc[0].get('sat_star_count', 0) or 0))
            return fwhm, sat

        if ref_frames_by_date:
            for date_key, fname in sorted(ref_frames_by_date.items()):
                fwhm, sat = _row_stats(fname)
                row = self.results_table.rowCount()
                self.results_table.insertRow(row)
                self.results_table.setItem(row, 0, QTableWidgetItem(str(date_key)))
                self.results_table.setItem(row, 1, QTableWidgetItem(str(ref_filter)))
                self.results_table.setItem(row, 2, QTableWidgetItem(str(fname)))
                self.results_table.setItem(row, 3, QTableWidgetItem(str(n_sources)))
                self.results_table.setItem(row, 4, QTableWidgetItem(str(fwhm)))
                self.results_table.setItem(row, 5, QTableWidgetItem(str(sat)))
        else:
            fwhm, sat = _row_stats(ref_frame)
            row = self.results_table.rowCount()
            self.results_table.insertRow(row)
            self.results_table.setItem(row, 0, QTableWidgetItem(_extract_date_key(ref_frame, self.params)))
            self.results_table.setItem(row, 1, QTableWidgetItem(str(ref_filter)))
            self.results_table.setItem(row, 2, QTableWidgetItem(str(ref_frame)))
            self.results_table.setItem(row, 3, QTableWidgetItem(str(n_sources)))
            self.results_table.setItem(row, 4, QTableWidgetItem(str(fwhm)))
            self.results_table.setItem(row, 5, QTableWidgetItem(str(sat)))

    def _update_stats_table(self):
        stats_path = step6_dir(self.params.P.result_dir) / "ref_frame_stats.csv"
        if not stats_path.exists():
            self.stats_table.setRowCount(0)
            self.stats_table.setColumnCount(0)
            return
        try:
            df = pd.read_csv(stats_path)
        except Exception:
            self.stats_table.setRowCount(0)
            self.stats_table.setColumnCount(0)
            return

        preferred_cols = [
            "date_key",
            "file",
            "filter",
            "wcs_ok",
            "match_rate",
            "sep_med_arcsec",
            "sep_p90_arcsec",
            "dup_rate",
            "wcs_resid_med",
            "wcs_match_n",
            "fwhm_px",
            "sat_star_count",
            "n_sources",
            "selected",
            "selected_date",
        ]
        cols = [c for c in preferred_cols if c in df.columns]
        if not cols:
            cols = list(df.columns)
        df = df[cols].copy()

        self.stats_table.setColumnCount(len(cols))
        self.stats_table.setRowCount(len(df))
        self.stats_table.setHorizontalHeaderLabels(cols)
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.stats_table.horizontalHeader().setStretchLastSection(True)

        for r, (_, row) in enumerate(df.iterrows()):
            for c, col in enumerate(cols):
                val = row.get(col, "")
                if isinstance(val, float):
                    if "rate" in col or "sep_" in col:
                        text = f"{val:.3f}" if np.isfinite(val) else ""
                    else:
                        text = f"{val:.2f}" if np.isfinite(val) else ""
                else:
                    text = str(val)
                self.stats_table.setItem(r, c, QTableWidgetItem(text))

    def _update_plot_tab(self, summary: Optional[dict] = None) -> None:
        if not hasattr(self, "plot_canvas") or self.plot_canvas is None:
            return
        fig = self.plot_canvas.figure
        fig.clear()
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        stats_path = step6_dir(self.params.P.result_dir) / "ref_frame_stats.csv"
        if not stats_path.exists():
            ax1.text(0.5, 0.5, "No ref stats available", ha="center", va="center")
            ax2.axis("off")
            self.plot_canvas.draw_idle()
            return
        try:
            df = pd.read_csv(stats_path)
        except Exception:
            ax1.text(0.5, 0.5, "Failed to read ref stats", ha="center", va="center")
            ax2.axis("off")
            self.plot_canvas.draw_idle()
            return
        if df.empty:
            ax1.text(0.5, 0.5, "No ref stats available", ha="center", va="center")
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
        if "fwhm_px" in df.columns:
            df["fwhm_px"] = pd.to_numeric(df["fwhm_px"], errors="coerce")
        else:
            df["fwhm_px"] = np.nan
        if "sat_star_count" in df.columns:
            df["sat_star_count"] = pd.to_numeric(df["sat_star_count"], errors="coerce")
        else:
            df["sat_star_count"] = np.nan

        if "filter" in df.columns:
            filters = sorted(df["filter"].fillna("").astype(str).unique().tolist())
        else:
            filters = [""]
        color_cycle = ["#1E88E5", "#43A047", "#F4511E", "#8E24AA", "#00897B", "#6D4C41"]
        color_map = {f: color_cycle[i % len(color_cycle)] for i, f in enumerate(filters)}

        df_plot = df
        if selected_date and "date_key" in df.columns:
            df_plot = df[df["date_key"].astype(str) == str(selected_date)]
            if df_plot.empty:
                df_plot = df

        for flt in filters:
            sub = df_plot[df_plot["filter"] == flt] if "filter" in df_plot.columns else df_plot
            ax1.scatter(
                sub["sep_med_arcsec"],
                sub["match_rate"],
                s=28,
                alpha=0.75,
                color=color_map.get(flt, "#90A4AE"),
                label=flt or "unknown",
                edgecolors="none",
            )
            ax2.scatter(
                sub["fwhm_px"],
                sub["sat_star_count"],
                s=28,
                alpha=0.75,
                color=color_map.get(flt, "#90A4AE"),
                edgecolors="none",
            )

        selected_frames: Dict[str, str] = {}
        if summary and isinstance(summary, dict):
            ref_by_date = summary.get("ref_frames_by_date", {}) or {}
            if ref_by_date:
                selected_frames.update({str(k): str(v) for k, v in ref_by_date.items()})
            elif summary.get("ref_frame"):
                selected_frames["ref"] = str(summary.get("ref_frame"))
        if not selected_frames:
            meta_path = step6_dir(self.params.P.result_dir) / "ref_build_meta.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    ref_by_date = meta.get("ref_frames_by_date", {}) or {}
                    if ref_by_date:
                        selected_frames.update({str(k): str(v) for k, v in ref_by_date.items()})
                    elif meta.get("ref_frame"):
                        selected_frames["ref"] = str(meta.get("ref_frame"))
                except Exception:
                    pass

        if selected_date and selected_date in selected_frames:
            selected_frames = {str(selected_date): selected_frames[str(selected_date)]}

        for _, fname in selected_frames.items():
            if "file" in df_plot.columns:
                row = df_plot[df_plot["file"] == fname]
            else:
                row = df_plot.iloc[0:0]
            if row.empty:
                continue
            r = row.iloc[0]
            ax1.scatter(
                r["sep_med_arcsec"], r["match_rate"],
                s=140, marker="*", color="#FF5252", edgecolors="#212121", linewidths=0.8, zorder=5
            )
            ax2.scatter(
                r["fwhm_px"], r["sat_star_count"],
                s=140, marker="*", color="#FF5252", edgecolors="#212121", linewidths=0.8, zorder=5
            )

        ax1.set_title("Match Rate vs Sep (arcsec)")
        ax1.set_xlabel("Sep med (arcsec)")
        ax1.set_ylabel("Match rate")
        ax1.grid(True, alpha=0.2)
        ax1.legend(fontsize=7, loc="best")

        ax2.set_title("FWHM vs Saturation")
        ax2.set_xlabel("FWHM (px)")
        ax2.set_ylabel("Sat star count")
        ax2.grid(True, alpha=0.2)

        fig.tight_layout()
        self.plot_canvas.draw_idle()

    def _on_plot_date_changed(self, index: int) -> None:
        if index < 0:
            return
        summary = self.results if isinstance(self.results, dict) else None
        self._update_plot_tab(summary)

    def validate_step(self) -> bool:
        out_dir = step6_dir(self.params.P.result_dir)
        return (out_dir / "ref_build_meta.json").exists()

    def save_state(self, summary: Optional[dict] = None):
        if summary is None:
            summary = self.results if isinstance(self.results, dict) else {}
        summary = summary or {}

        ref_frame = summary.get("ref_frame")
        ref_filter = summary.get("ref_filter")
        n_sources = summary.get("n_sources", 0)

        if not ref_frame or not ref_filter:
            meta_path = step6_dir(self.params.P.result_dir) / "ref_build_meta.json"
            if not meta_path.exists():
                meta_path = legacy_step5_refbuild_dir(self.params.P.result_dir) / "ref_build_meta.json"
            if not meta_path.exists():
                meta_path = legacy_step7_refbuild_dir(self.params.P.result_dir) / "ref_build_meta.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    ref_frame = ref_frame or meta.get("ref_frame")
                    ref_filter = ref_filter or meta.get("ref_filter")
                except Exception:
                    pass

        if (not n_sources) and ref_filter:
            ref_path = step6_dir(self.params.P.result_dir) / f"ref_catalog_{ref_filter}.tsv"
            legacy_path = legacy_step5_refbuild_dir(self.params.P.result_dir) / f"ref_catalog_{ref_filter}.tsv"
            legacy_master = legacy_step7_refbuild_dir(self.params.P.result_dir) / f"master_catalog_{ref_filter}.tsv"
            if not ref_path.exists() and legacy_path.exists():
                ref_path = legacy_path
            if not ref_path.exists() and legacy_master.exists():
                ref_path = legacy_master
            if ref_path.exists():
                try:
                    n_sources = len(pd.read_csv(ref_path, sep="\t"))
                except Exception:
                    n_sources = 0

        state_data = {
            "ref_frame": ref_frame,
            "ref_filter": ref_filter,
            "n_sources": n_sources,
        }
        self.project_state.store_step_data("ref_build", state_data)
        if ref_frame:
            self.file_manager.ref_filename = ref_frame

    def restore_state(self):
        state = self.project_state.get_step_data("ref_build")
        if not state:
            return
        if state.get("ref_frame"):
            self.file_manager.ref_filename = state.get("ref_frame")
        self._update_stats_table()
        self._update_plot_tab(state)

    def log(self, msg: str):
        if self.log_text is not None:
            self.log_text.append(msg)

    def show_log_window(self):
        if self.log_window is not None:
            self.log_window.show()
            self.log_window.raise_()
