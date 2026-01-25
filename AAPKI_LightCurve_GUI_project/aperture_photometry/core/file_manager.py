"""
File management for FITS image processing
Extracted from AAPKI_GUI.ipynb Cell 2-3
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from astropy.io import fits

from ..utils import get_logger
from ..utils.astro_utils import (
    DEFAULT_AIRMASS_FORMULA,
    airmass_from_alt,
    compute_airmass_from_header,
)
from ..utils.step_paths import step1_dir

log = get_logger("aperture_phot.core.file_manager")


class FileManager:
    """
    Manages FITS file discovery, header reading, and reference frame selection
    """

    def __init__(self, params):
        """
        Initialize file manager

        Args:
            params: Parameters object from config module
        """
        self.params = params
        self.filenames: List[str] = []
        self.df_headers: Optional[pd.DataFrame] = None
        self.ref_filename: Optional[str] = None
        self.path_map: Dict[str, Path] = {}
        self.selected_dirs: List[Path] = []
        self.root_dir: Optional[Path] = None

    def set_multi_night_dirs(self, root_dir: Path, night_dirs: List[Path]) -> None:
        """Configure multi-night scanning with selected subdirectories."""
        self.root_dir = Path(root_dir)
        self.selected_dirs = [Path(p) for p in night_dirs if p]

    def clear_multi_night_dirs(self) -> None:
        """Clear multi-night selection and revert to single directory scan."""
        self.selected_dirs = []
        self.root_dir = None

    def _make_file_key(self, rel_path: Path) -> str:
        parts = [p for p in rel_path.parts if p not in (".", "")]
        return "__".join(parts)

    def _ensure_unique_key(self, key: str) -> str:
        if key not in self.path_map:
            return key
        base = key
        idx = 2
        while f"{base}__dup{idx}" in self.path_map:
            idx += 1
        return f"{base}__dup{idx}"

    def scan_files(self) -> List[str]:
        """
        Scan data directory for FITS files matching prefix pattern

        Returns:
            List of matching filenames (sorted)

        Raises:
            RuntimeError: If no matching files found
        """
        data_dir = Path(self.params.P.data_dir)
        prefix = self.params.P.filename_prefix

        self.path_map = {}

        selected_dirs = [Path(p) for p in getattr(self, "selected_dirs", []) if p]
        if selected_dirs:
            root_dir = Path(self.root_dir or data_dir)
            file_items: List[tuple[str, Path]] = []
            collected: List[tuple[str, Path, Path]] = []
            prefix_lower = str(prefix).lower()
            suffixes = (".fit", ".fits", ".fit.fz", ".fits.fz")
            for subdir in selected_dirs:
                if not subdir.exists():
                    continue
                for fpath in sorted(p for p in subdir.iterdir() if p.is_file()):
                    name = fpath.name
                    lower = name.lower()
                    if prefix_lower and not lower.startswith(prefix_lower):
                        continue
                    if not lower.endswith(suffixes):
                        continue
                    try:
                        rel = fpath.relative_to(root_dir)
                    except ValueError:
                        rel = Path(subdir.name) / fpath.name
                    collected.append((fpath.name, fpath, rel))

            name_counts: Dict[str, int] = {}
            for base, _, _ in collected:
                name_counts[base] = name_counts.get(base, 0) + 1

            for base, fpath, rel in collected:
                if name_counts.get(base, 0) == 1:
                    key = base
                else:
                    key = self._make_file_key(rel)
                key = self._ensure_unique_key(key)
                file_items.append((key, fpath))

            self.filenames = sorted([k for k, _ in file_items])
            for key, fpath in file_items:
                self.path_map[key] = fpath
        else:
            # Find all FITS files matching prefix (case-insensitive)
            self.filenames = sorted([
                f for f in os.listdir(data_dir)
                if f.lower().startswith(prefix.lower()) and f.lower().endswith((".fit", ".fits", ".fit.fz", ".fits.fz"))
            ])
            self.path_map = {fn: data_dir / fn for fn in self.filenames}

        if not self.filenames:
            raise RuntimeError(f"No input files found with prefix '{prefix}' in {data_dir}")

        if hasattr(self.params, "P"):
            self.params.P.file_path_map = {k: str(v) for k, v in self.path_map.items()}

        # Save target list
        step1_out = step1_dir(self.params.P.result_dir)
        step1_out.mkdir(parents=True, exist_ok=True)
        target_list_path = step1_out / "target_list.txt"
        target_list_path.write_text("\n".join(self.filenames), encoding="utf-8")

        log.info(f"Found {len(self.filenames)} files -> {target_list_path}")
        return self.filenames

    def read_headers(self) -> pd.DataFrame:
        """
        Read FITS headers from all files and create summary DataFrame

        Returns:
            DataFrame with header information (Filename, DATE-OBS, FILTER, EXPTIME, etc.)
        """
        if not self.filenames:
            self.scan_files()

        rows = []
        lat = float(getattr(self.params.P, "site_lat_deg", 0.0))
        lon = float(getattr(self.params.P, "site_lon_deg", 0.0))
        alt = float(getattr(self.params.P, "site_alt_m", 0.0))
        tz = float(getattr(self.params.P, "site_tz_offset_hours", 0.0))
        for fn in self.filenames:
            try:
                file_path = self.get_file_path(fn)
                with fits.open(file_path) as hdul:
                    h = hdul[0].header

                # Get EXPTIME with proper numeric handling
                exptime_val = h.get("EXPTIME", None)
                if exptime_val is None:
                    # Try alternative keywords
                    for key in ("EXPOSURE", "ITIME", "ELAPTIME"):
                        if key in h:
                            exptime_val = h[key]
                            break
                try:
                    exptime = float(exptime_val) if exptime_val is not None else np.nan
                except (TypeError, ValueError):
                    exptime = np.nan

                # Get AIRMASS with proper numeric handling
                airmass_val = h.get("AIRMASS", None)
                try:
                    airmass = float(airmass_val) if airmass_val is not None else np.nan
                except (TypeError, ValueError):
                    airmass = np.nan

                # JD + RA/Dec (best-effort)
                jd_val = h.get("JD", None)
                try:
                    jd = float(jd_val) if jd_val is not None else np.nan
                except (TypeError, ValueError):
                    jd = np.nan
                if not np.isfinite(jd):
                    date_obs = h.get("DATE-OBS", None) or h.get("DATE", None)
                    time_obs = h.get("TIME-OBS", None) or h.get("UTC", None) or h.get("UT", None)
                    if date_obs:
                        dt_str = str(date_obs).strip()
                        if "T" not in dt_str and time_obs:
                            dt_str = f"{dt_str}T{str(time_obs).strip()}"
                        try:
                            from astropy.time import Time
                            jd = float(Time(dt_str, scale="utc").jd)
                        except Exception:
                            jd = np.nan

                info = compute_airmass_from_header(h, lat, lon, alt, tz)
                ra_deg = float(info.get("ra_deg", np.nan))
                dec_deg = float(info.get("dec_deg", np.nan))

                rows.append({
                    "Filename": fn,
                    "DATE-OBS": h.get("DATE-OBS", "N/A"),
                    "FILTER": h.get("FILTER", "UNKNOWN"),
                    "EXPTIME": exptime,
                    "AIRMASS": airmass,
                    "IMAGETYP": h.get("IMAGETYP", h.get("FRAME", "Unknown")),
                    "JD": jd,
                    "RA_DEG": ra_deg,
                    "DEC_DEG": dec_deg,
                })
            except Exception as e:
                log.warning(f"{fn}: Header read failed - {e}")

        self.df_headers = pd.DataFrame(rows)

        # Save headers CSV
        step1_out = step1_dir(self.params.P.result_dir)
        step1_out.mkdir(parents=True, exist_ok=True)
        headers_path = step1_out / "headers.csv"
        self.df_headers.to_csv(headers_path, index=False, encoding="utf-8")
        log.info(f"Saved: {headers_path} | rows: {len(self.df_headers)}")

        return self.df_headers

    def update_airmass_headers(
        self,
        formula: str | None = None,
        mode: str = "overwrite",
        source: str | None = None,
    ) -> dict:
        """
        Update FITS header AIRMASS based on RA/Dec+time (computed altitude).

        Args:
            formula: Airmass formula label (see astro_utils.AIRMASS_FORMULAS).
            mode: "overwrite" or "if_missing"

        Returns:
            dict with counts: updated, skipped_missing_alt, skipped_has_airmass, failed, skipped_compressed
        """
        if not self.filenames:
            self.scan_files()

        stats = {
            "updated": 0,
            "skipped_missing_alt": 0,
            "skipped_has_airmass": 0,
            "failed": 0,
            "skipped_compressed": 0,
        }

        formula_label = formula or getattr(self.params.P, "airmass_formula", DEFAULT_AIRMASS_FORMULA)
        source_pref = source or getattr(self.params.P, "airmass_update_source", "auto")
        mode = str(mode or "overwrite").strip().lower()

        lat = float(getattr(self.params.P, "site_lat_deg", 0.0))
        lon = float(getattr(self.params.P, "site_lon_deg", 0.0))
        alt = float(getattr(self.params.P, "site_alt_m", 0.0))
        tz = float(getattr(self.params.P, "site_tz_offset_hours", 0.0))

        for fn in self.filenames:
            try:
                file_path = self.get_file_path(fn)
            except Exception:
                stats["failed"] += 1
                continue
            if str(file_path).lower().endswith(".fz"):
                stats["skipped_compressed"] += 1
                continue
            try:
                with fits.open(file_path, mode="update") as hdul:
                    hdr = hdul[0].header

                    if mode == "if_missing":
                        try:
                            val = float(hdr.get("AIRMASS", np.nan))
                            if np.isfinite(val):
                                stats["skipped_has_airmass"] += 1
                                continue
                        except Exception:
                            pass

                    info = compute_airmass_from_header(
                        hdr,
                        lat,
                        lon,
                        alt,
                        tz,
                        formula=formula_label,
                        source=source_pref,
                    )
                    alt_deg = float(info.get("alt_deg", np.nan))
                    if not np.isfinite(alt_deg):
                        # Fallback: use target RA/Dec if available
                        tgt_ra = getattr(self.params.P, "target_ra_deg", None)
                        tgt_dec = getattr(self.params.P, "target_dec_deg", None)
                        if tgt_ra is not None and tgt_dec is not None and source_pref in ("auto", "radec", "ra"):
                            try:
                                hdr_tmp = hdr.copy()
                                hdr_tmp["RA"] = float(tgt_ra)
                                hdr_tmp["DEC"] = float(tgt_dec)
                                info = compute_airmass_from_header(
                                    hdr_tmp,
                                    lat,
                                    lon,
                                    alt,
                                    tz,
                                    formula=formula_label,
                                    source="radec",
                                )
                                alt_deg = float(info.get("alt_deg", np.nan))
                            except Exception:
                                alt_deg = np.nan
                        if not np.isfinite(alt_deg):
                            stats["skipped_missing_alt"] += 1
                            continue

                    airmass = airmass_from_alt(alt_deg, formula_label)
                    if not np.isfinite(airmass):
                        stats["skipped_missing_alt"] += 1
                        continue

                    comment = f"{formula_label} AAPKI_lightcurve"
                    hdr["AIRMASS"] = (float(airmass), comment)
                    stats["updated"] += 1
            except Exception:
                stats["failed"] += 1

        # headers.csv 동기화 (AIRMASS 컬럼 업데이트)
        if stats["updated"] > 0:
            self._sync_headers_csv_airmass()

        return stats

    def _sync_headers_csv_airmass(self):
        """FITS 헤더 변경 후 headers.csv의 AIRMASS 컬럼 동기화"""
        headers_path = step1_dir(self.params.P.result_dir) / "headers.csv"
        if not headers_path.exists():
            return

        try:
            df = pd.read_csv(headers_path)
            if "Filename" not in df.columns:
                return

            updated = 0
            for idx, row in df.iterrows():
                fn = str(row["Filename"])
                try:
                    file_path = self.get_file_path(fn)
                    with fits.open(file_path) as hdul:
                        new_airmass = hdul[0].header.get("AIRMASS", np.nan)
                        try:
                            new_airmass = float(new_airmass)
                        except (TypeError, ValueError):
                            new_airmass = np.nan
                        if np.isfinite(new_airmass):
                            df.at[idx, "AIRMASS"] = new_airmass
                            updated += 1
                except Exception:
                    continue

            if updated > 0:
                df.to_csv(headers_path, index=False, encoding="utf-8")
                log.info(f"headers.csv AIRMASS synced: {updated} rows updated")

        except Exception as e:
            log.warning(f"headers.csv sync failed: {e}")

    def select_reference_frame(self) -> str:
        """
        Select reference frame based on filter and index preferences

        Returns:
            Filename of selected reference frame
        """
        if self.df_headers is None:
            self.read_headers()

        # Start with all candidates
        ref_candidates = self.df_headers

        # Filter by global_ref_filter if specified
        grf = str(getattr(self.params.P, "global_ref_filter", "")).lower()
        if grf in ("g", "r", "i"):
            filtered = self.df_headers[
                self.df_headers["FILTER"].astype(str).str.lower() == grf
            ]
            if not filtered.empty:
                ref_candidates = filtered

        # Select by index
        ref_index = max(
            0,
            min(
                int(getattr(self.params.P, "global_ref_index", 0) or 0),
                len(ref_candidates) - 1,
            ),
        )

        if not ref_candidates.empty:
            self.ref_filename = ref_candidates.iloc[ref_index]["Filename"]
        else:
            self.ref_filename = self.filenames[0]

        log.info(f"Selected reference frame: {self.ref_filename}")
        return self.ref_filename

    def get_reference_header(self) -> fits.Header:
        """
        Get FITS header of reference frame

        Returns:
            FITS header object
        """
        if self.ref_filename is None:
            self.select_reference_frame()

        file_path = self.get_file_path(self.ref_filename)
        with fits.open(file_path) as hdul:
            return hdul[0].header.copy()

    def print_reference_header(self):
        """Print reference frame header for inspection"""
        header = self.get_reference_header()
        log.info(f"Reference Frame Header: {self.ref_filename}")
        log.debug(repr(header))

    def get_file_path(self, filename: str) -> Path:
        """
        Get full path to a file in data directory

        Args:
            filename: Base filename

        Returns:
            Full path to file
        """
        if filename in self.path_map:
            return self.path_map[filename]
        return Path(self.params.P.data_dir) / filename

    def get_all_file_paths(self) -> List[Path]:
        """
        Get full paths to all scanned files

        Returns:
            List of Path objects
        """
        if not self.filenames:
            self.scan_files()
        return [self.get_file_path(fn) for fn in self.filenames]
