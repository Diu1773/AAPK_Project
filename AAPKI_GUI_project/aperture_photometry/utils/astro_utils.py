"""
Astronomical utility functions
Extracted from AAPKI_GUI.ipynb Cell 0 and Cell 1
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import math
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
import astropy.units as u
import pandas as pd


class HeaderCache:
    """
    Thread-safe cache for FITS header information.

    Replaces the global df_headers variable with a proper class-based approach
    that supports dependency injection and is easier to test.

    Usage:
        # Set headers from file_manager
        header_cache.set_headers(df)

        # Get filter for a file
        filter_name = header_cache.get_filter("image.fits")

        # Or use as context
        with header_cache.use(df):
            filter_name = get_filter_from_fits(path)
    """

    def __init__(self):
        self._df_headers: Optional[pd.DataFrame] = None

    def set_headers(self, df: Optional[pd.DataFrame]) -> None:
        """Set the headers DataFrame."""
        self._df_headers = df

    def get_headers(self) -> Optional[pd.DataFrame]:
        """Get the current headers DataFrame."""
        return self._df_headers

    def clear(self) -> None:
        """Clear the cached headers."""
        self._df_headers = None

    def get_filter(self, filename: str) -> Optional[str]:
        """
        Get filter name for a given filename from cached headers.

        Args:
            filename: The filename to look up (with or without path)

        Returns:
            Filter name (lowercase) or None if not found
        """
        if self._df_headers is None or self._df_headers.empty:
            return None

        # Strip path and common prefixes
        base = Path(filename).name
        for prefix in ("rc_", "r_", "crop_", "Crop_"):
            if base.startswith(prefix):
                base = base[len(prefix):]

        try:
            # Try exact match first
            row = self._df_headers[self._df_headers["Filename"] == base]
            if row.empty:
                # Try with original name
                row = self._df_headers[self._df_headers["Filename"] == Path(filename).name]

            if not row.empty and "FILTER" in row.columns:
                return str(row["FILTER"].values[0]).strip().lower()
        except Exception:
            pass

        return None

    def use(self, df: pd.DataFrame):
        """Context manager for temporary header usage."""
        return _HeaderCacheContext(self, df)


class _HeaderCacheContext:
    """Context manager for HeaderCache."""

    def __init__(self, cache: HeaderCache, df: pd.DataFrame):
        self._cache = cache
        self._new_df = df
        self._old_df: Optional[pd.DataFrame] = None

    def __enter__(self):
        self._old_df = self._cache.get_headers()
        self._cache.set_headers(self._new_df)
        return self._cache

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cache.set_headers(self._old_df)
        return False


# Global singleton instance (for backward compatibility)
header_cache = HeaderCache()

# Legacy global variable (deprecated, use header_cache instead)
# This is kept for backward compatibility but will be removed in future versions
df_headers = None


def _set_headers_compat(df: Optional[pd.DataFrame]) -> None:
    """Set headers for both new and legacy systems."""
    global df_headers
    df_headers = df
    header_cache.set_headers(df)


def _to_plain(a):
    """Convert masked array to plain array with NaN fill"""
    if isinstance(a, np.ma.MaskedArray):
        return a.filled(np.nan)
    return a


def _jsonify(o):
    """Convert numpy types to Python native types for JSON"""
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def _is_up_to_date(target: Path, deps: list[Path]) -> bool:
    """Check if target file is up-to-date relative to dependencies"""
    target = Path(target)
    if not target.exists():
        return False
    try:
        t = target.stat().st_mtime
        return all(Path(d).exists() and t >= Path(d).stat().st_mtime for d in deps)
    except Exception:
        return False


def get_exptime_from_fits(path: Path, default: float = 1.0) -> float:
    """Extract exposure time from FITS header"""
    try:
        with fits.open(path) as hdul:
            return float(hdul[0].header.get("EXPTIME", default))
    except Exception:
        return default


def get_filter_from_fits(path: Path, cache: Optional[HeaderCache] = None) -> str:
    """
    Extract filter name from FITS header.

    Falls back to header cache (headers.csv/tsv) if not in FITS header.
    Returns lowercase filter name.

    Args:
        path: Path to FITS file
        cache: Optional HeaderCache instance (uses global header_cache if None)

    Returns:
        Filter name in lowercase, or "unknown" if not found
    """
    # Try FITS header first
    try:
        with fits.open(path) as hdul:
            f = hdul[0].header.get("FILTER", None)
            if f:
                return str(f).strip().lower()
    except Exception:
        pass

    # Fallback to header cache
    _cache = cache if cache is not None else header_cache
    cached_filter = _cache.get_filter(str(path))
    if cached_filter:
        return cached_filter

    # Legacy fallback (for backward compatibility)
    global df_headers
    if df_headers is not None:
        base = Path(path).name
        for prefix in ("rc_", "r_"):
            if base.startswith(prefix):
                base = base[len(prefix):]
        try:
            row = df_headers[df_headers["Filename"] == base]
            if not row.empty:
                return str(row["FILTER"].values[0]).strip().lower()
        except Exception:
            pass

    return "unknown"


def estimate_limiting_mag(
    zp: float,
    exptime_s: float,
    fwhm_arcsec: float,
    sky_sigma_e_per_px: float,
    pix_scale_arcsec: float,
    snr_limit: float = 5.0,
    rdnoise_e: float = 7.5,
) -> float:
    """
    Estimate limiting magnitude for given observing conditions

    Args:
        zp: Zero point magnitude
        exptime_s: Exposure time in seconds
        fwhm_arcsec: Seeing FWHM in arcseconds
        sky_sigma_e_per_px: Sky background sigma in electrons per pixel
        pix_scale_arcsec: Pixel scale in arcseconds per pixel
        snr_limit: SNR threshold for limiting magnitude (default 5.0)
        rdnoise_e: Read noise in electrons (default 7.5)

    Returns:
        Limiting magnitude
    """
    if pix_scale_arcsec <= 0:
        raise ValueError("pix_scale_arcsec must be > 0")

    r_fwhm_px = (fwhm_arcsec / pix_scale_arcsec) / 2.0
    A_psf = math.pi * max(r_fwhm_px, 0.5) ** 2

    sky_e_per_px = sky_sigma_e_per_px ** 2

    def snr_for_mag(m: float) -> float:
        """Calculate SNR for a given magnitude"""
        rate_e = 10.0 ** (-0.4 * (m - zp))
        N_star = rate_e * exptime_s
        N_sky = sky_e_per_px * A_psf
        N_RN2 = (rdnoise_e ** 2) * A_psf
        var = max(N_star + N_sky + N_RN2, 1e-12)
        return N_star / math.sqrt(var)

    # Binary search for limiting magnitude
    m_min = zp - 5.0
    m_max = zp + 20.0
    s_min = snr_for_mag(m_min)
    s_max = snr_for_mag(m_max)

    if s_min < snr_limit:
        return m_min
    if s_max > snr_limit:
        return m_max

    for _ in range(60):
        m_mid = 0.5 * (m_min + m_max)
        s_mid = snr_for_mag(m_mid)
        if s_mid > snr_limit:
            m_min = m_mid
        else:
            m_max = m_mid
        if abs(m_max - m_min) < 1e-3:
            break

    return 0.5 * (m_min + m_max)


def _parse_obs_time(header: fits.Header) -> Time | None:
    """Parse observation time from FITS header (UTC)."""
    date_obs = header.get("DATE-OBS", None) or header.get("DATE", None)
    time_obs = header.get("TIME-OBS", None) or header.get("UTC", None) or header.get("UT", None)
    if date_obs is None:
        return None
    date_obs = str(date_obs).strip()
    if "T" in date_obs:
        dt_str = date_obs
    elif time_obs:
        dt_str = f"{date_obs}T{str(time_obs).strip()}"
    else:
        dt_str = date_obs
    for fmt in ("isot", "fits"):
        try:
            return Time(dt_str, format=fmt, scale="utc")
        except Exception:
            continue
    try:
        return Time(dt_str, scale="utc")
    except Exception:
        return None


def _parse_ra_dec_from_header(header: fits.Header) -> tuple[float, float] | None:
    """Parse RA/Dec from common FITS header keys."""
    key_pairs = [
        ("OBJCTRA", "OBJCTDEC", True),
        ("OBJRA", "OBJDEC", True),
        ("RA", "DEC", False),
        ("RA_OBJ", "DEC_OBJ", False),
    ]
    for ra_key, dec_key, prefer_hourangle in key_pairs:
        if ra_key in header and dec_key in header:
            ra_val = header.get(ra_key)
            dec_val = header.get(dec_key)
            if ra_val is None or dec_val is None:
                continue
            ra_str = str(ra_val).strip()
            dec_str = str(dec_val).strip()
            try:
                if (":" in ra_str) or ("h" in ra_str.lower()) or prefer_hourangle:
                    sc = SkyCoord(ra_str, dec_str, unit=(u.hourangle, u.deg), frame="icrs")
                else:
                    sc = SkyCoord(float(ra_val) * u.deg, float(dec_val) * u.deg, frame="icrs")
                return float(sc.ra.deg), float(sc.dec.deg)
            except Exception:
                continue
    return None


def _parse_radec_from_wcs(header: fits.Header) -> tuple[float, float] | None:
    """Parse RA/Dec from WCS center if available."""
    try:
        w = WCS(header, relax=True)
        if w.celestial is None:
            return None
        nx = header.get("NAXIS1", None)
        ny = header.get("NAXIS2", None)
        if nx is None or ny is None:
            return None
        x = float(nx) / 2.0
        y = float(ny) / 2.0
        ra, dec = w.celestial.wcs_pix2world([[x, y]], 0)[0]
        return float(ra), float(dec)
    except Exception:
        return None


def kasten_young_airmass(alt_deg: float) -> float:
    """Kasten & Young (1989) airmass approximation with horizon correction."""
    if not np.isfinite(alt_deg):
        return np.nan
    if alt_deg <= 0:
        return np.nan
    z = 90.0 - float(alt_deg)
    return 1.0 / (np.cos(np.deg2rad(z)) + 0.50572 * (96.07995 - z) ** (-1.6364))


def compute_airmass_from_header(
    header: fits.Header,
    site_lat_deg: float,
    site_lon_deg: float,
    site_alt_m: float,
    tz_offset_hours: float = 0.0,
) -> dict:
    """
    Compute airmass and related metadata from FITS header.

    Returns dict with keys:
    - airmass, airmass_source, alt_deg, zenith_deg
    - datetime_utc, datetime_local, ra_deg, dec_deg
    """
    airmass_val = header.get("AIRMASS", None)
    airmass = np.nan
    airmass_source = "computed"
    try:
        if airmass_val is not None:
            airmass = float(airmass_val)
            if np.isfinite(airmass):
                airmass_source = "header"
    except Exception:
        airmass = np.nan

    t = _parse_obs_time(header)
    ra_dec = _parse_ra_dec_from_header(header)
    if ra_dec is None:
        ra_dec = _parse_radec_from_wcs(header)

    alt_deg = np.nan
    zenith_deg = np.nan
    if t is not None and ra_dec is not None:
        try:
            loc = EarthLocation(lat=float(site_lat_deg) * u.deg,
                                lon=float(site_lon_deg) * u.deg,
                                height=float(site_alt_m) * u.m)
            sc = SkyCoord(ra_dec[0] * u.deg, ra_dec[1] * u.deg, frame="icrs")
            altaz = sc.transform_to(AltAz(obstime=t, location=loc))
            alt_deg = float(altaz.alt.deg)
            zenith_deg = 90.0 - alt_deg
            if not np.isfinite(airmass):
                airmass = kasten_young_airmass(alt_deg)
                airmass_source = "computed"
        except Exception:
            pass

    time_utc = t.isot if t is not None else ""
    try:
        if t is not None and np.isfinite(tz_offset_hours):
            t_local = t + float(tz_offset_hours) * u.hour
            time_local = t_local.isot
        else:
            time_local = ""
    except Exception:
        time_local = ""

    ra_deg = float(ra_dec[0]) if ra_dec is not None else np.nan
    dec_deg = float(ra_dec[1]) if ra_dec is not None else np.nan

    return {
        "airmass": float(airmass) if np.isfinite(airmass) else np.nan,
        "airmass_source": airmass_source,
        "alt_deg": float(alt_deg) if np.isfinite(alt_deg) else np.nan,
        "zenith_deg": float(zenith_deg) if np.isfinite(zenith_deg) else np.nan,
        "datetime_utc": time_utc,
        "datetime_local": time_local,
        "ra_deg": ra_deg,
        "dec_deg": dec_deg,
    }
