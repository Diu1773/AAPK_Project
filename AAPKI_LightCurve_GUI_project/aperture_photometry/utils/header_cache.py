"""
FITS Header Cache Utility

Provides efficient header data access by:
1. Loading headers.csv (from Step 1) as primary source
2. Lazy-loading FITS headers only when CSV data is insufficient
3. Caching FITS headers once loaded to avoid repeated file access
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

from .step_paths import step1_dir


class HeaderCache:
    """Centralized header data cache for workflow steps.

    Uses headers.csv from Step 1 as the primary source of metadata.
    Falls back to FITS header reading only when necessary, with caching.
    """

    def __init__(self, result_dir: Path, data_dir: Optional[Path] = None):
        """Initialize the header cache.

        Args:
            result_dir: Path to the result directory containing step outputs
            data_dir: Optional path to the FITS data directory
        """
        self.result_dir = Path(result_dir)
        self.data_dir = Path(data_dir) if data_dir else None
        self._headers_df: Optional[pd.DataFrame] = None
        self._headers_loaded = False
        self._fits_cache: Dict[str, Any] = {}  # path -> fits.Header or None
        self._filename_map: Dict[str, Dict[str, Any]] = {}  # filename -> row dict

    def _load_headers_csv(self) -> None:
        """Load headers.csv from Step 1 output directory."""
        if self._headers_loaded:
            return

        # Try step1 directory first, then result root
        headers_path = step1_dir(self.result_dir) / "headers.csv"
        if not headers_path.exists():
            headers_path = self.result_dir / "headers.csv"

        if headers_path.exists():
            try:
                self._headers_df = pd.read_csv(headers_path)
                # Build filename -> row mapping for fast lookup
                if "Filename" in self._headers_df.columns:
                    for _, row in self._headers_df.iterrows():
                        fname = str(row["Filename"])
                        # Also index by basename for flexible matching
                        self._filename_map[fname] = row.to_dict()
                        basename = Path(fname).name
                        if basename not in self._filename_map:
                            self._filename_map[basename] = row.to_dict()
            except Exception:
                self._headers_df = pd.DataFrame()
        else:
            self._headers_df = pd.DataFrame()

        self._headers_loaded = True

    def _get_row(self, filename: str) -> Optional[Dict[str, Any]]:
        """Get cached row for a filename."""
        self._load_headers_csv()

        # Try exact match first
        if filename in self._filename_map:
            return self._filename_map[filename]

        # Try basename
        basename = Path(filename).name
        if basename in self._filename_map:
            return self._filename_map[basename]

        return None

    def get_filter(self, filename: str, fits_path: Optional[Path] = None) -> str:
        """Get FILTER value for a file, preferring headers.csv.

        Args:
            filename: The filename to look up
            fits_path: Optional path to FITS file for fallback

        Returns:
            Filter name (lowercase) or "unknown"
        """
        row = self._get_row(filename)

        # Try headers.csv first
        if row:
            for col in ("FILTER", "filter", "Filter"):
                if col in row and pd.notna(row[col]):
                    val = str(row[col]).strip()
                    if val:
                        return val.lower()

        # Fallback to FITS header
        if fits_path and Path(fits_path).exists():
            hdr = self._get_fits_header(str(fits_path))
            if hdr:
                for key in ("FILTER", "FILTER1", "FILTER2"):
                    if key in hdr:
                        val = hdr[key]
                        if val:
                            return str(val).strip().lower()

        return "unknown"

    def get_exptime(self, filename: str, fits_path: Optional[Path] = None,
                    default: float = 1.0) -> float:
        """Get EXPTIME value for a file, preferring headers.csv.

        Args:
            filename: The filename to look up
            fits_path: Optional path to FITS file for fallback
            default: Default value if not found

        Returns:
            Exposure time in seconds
        """
        row = self._get_row(filename)

        # Try headers.csv first
        if row:
            for col in ("EXPTIME", "EXPOSURE", "Exptime", "exptime"):
                if col in row:
                    try:
                        val = float(row[col])
                        if np.isfinite(val) and val > 0:
                            return val
                    except (ValueError, TypeError):
                        continue

        # Fallback to FITS header
        if fits_path and Path(fits_path).exists():
            hdr = self._get_fits_header(str(fits_path))
            if hdr:
                for key in ("EXPTIME", "EXPOSURE", "ITIME", "ELAPTIME"):
                    if key in hdr:
                        try:
                            val = float(hdr[key])
                            if np.isfinite(val) and val > 0:
                                return val
                        except (ValueError, TypeError):
                            continue

        return default

    def get_dateobs(self, filename: str, fits_path: Optional[Path] = None) -> Optional[str]:
        """Get DATE-OBS value for a file.

        Args:
            filename: The filename to look up
            fits_path: Optional path to FITS file for fallback

        Returns:
            DATE-OBS string or None
        """
        row = self._get_row(filename)

        if row:
            for col in ("DATE-OBS", "DATE_OBS", "DATEOBS", "Date-Obs"):
                if col in row and pd.notna(row[col]):
                    return str(row[col])

        if fits_path and Path(fits_path).exists():
            hdr = self._get_fits_header(str(fits_path))
            if hdr:
                for key in ("DATE-OBS", "DATE_OBS", "DATEOBS"):
                    if key in hdr:
                        return str(hdr[key])

        return None

    def get_airmass(self, filename: str, fits_path: Optional[Path] = None,
                    default: float = 1.0) -> float:
        """Get AIRMASS value for a file.

        Args:
            filename: The filename to look up
            fits_path: Optional path to FITS file for fallback
            default: Default value if not found

        Returns:
            Airmass value
        """
        row = self._get_row(filename)

        if row:
            for col in ("AIRMASS", "Airmass", "airmass"):
                if col in row:
                    try:
                        val = float(row[col])
                        if np.isfinite(val) and val > 0:
                            return val
                    except (ValueError, TypeError):
                        continue

        if fits_path and Path(fits_path).exists():
            hdr = self._get_fits_header(str(fits_path))
            if hdr and "AIRMASS" in hdr:
                try:
                    return float(hdr["AIRMASS"])
                except (ValueError, TypeError):
                    pass

        return default

    def get_jd(self, filename: str, fits_path: Optional[Path] = None) -> Optional[float]:
        """Get Julian Date for a file.

        Args:
            filename: The filename to look up
            fits_path: Optional path to FITS file for fallback

        Returns:
            Julian Date or None
        """
        row = self._get_row(filename)

        if row:
            for col in ("JD", "jd", "MJD", "mjd"):
                if col in row:
                    try:
                        val = float(row[col])
                        if np.isfinite(val) and val > 0:
                            return val
                    except (ValueError, TypeError):
                        continue

        if fits_path and Path(fits_path).exists():
            hdr = self._get_fits_header(str(fits_path))
            if hdr:
                for key in ("JD", "MJD-OBS", "JD-OBS"):
                    if key in hdr:
                        try:
                            return float(hdr[key])
                        except (ValueError, TypeError):
                            continue

        return None

    def get_header_value(self, filename: str, key: str,
                         fits_path: Optional[Path] = None,
                         default: Any = None) -> Any:
        """Get any header value for a file.

        Args:
            filename: The filename to look up
            key: Header keyword to retrieve
            fits_path: Optional path to FITS file for fallback
            default: Default value if not found

        Returns:
            Header value or default
        """
        row = self._get_row(filename)

        if row:
            # Try exact key and common variations
            for col in (key, key.upper(), key.lower(), key.replace("-", "_")):
                if col in row and pd.notna(row[col]):
                    return row[col]

        if fits_path and Path(fits_path).exists():
            hdr = self._get_fits_header(str(fits_path))
            if hdr and key in hdr:
                return hdr[key]

        return default

    def _get_fits_header(self, fits_path: str) -> Optional[Any]:
        """Get FITS header with caching.

        Args:
            fits_path: Path to the FITS file

        Returns:
            FITS header or None
        """
        if fits_path in self._fits_cache:
            return self._fits_cache[fits_path]

        try:
            from astropy.io import fits
            hdr = fits.getheader(fits_path)
            self._fits_cache[fits_path] = hdr
            return hdr
        except Exception:
            self._fits_cache[fits_path] = None
            return None

    def clear_fits_cache(self) -> None:
        """Clear FITS header cache (call when switching projects)."""
        self._fits_cache.clear()

    def reload(self) -> None:
        """Force reload headers.csv and clear all caches."""
        self._headers_loaded = False
        self._headers_df = None
        self._filename_map.clear()
        self._fits_cache.clear()
        self._load_headers_csv()

    def has_headers_csv(self) -> bool:
        """Check if headers.csv exists and was loaded.

        Returns:
            True if headers.csv is available
        """
        self._load_headers_csv()
        return self._headers_df is not None and not self._headers_df.empty
