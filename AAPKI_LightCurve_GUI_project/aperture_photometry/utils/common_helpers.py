"""
Shared helper functions used across multiple step modules.

Consolidates duplicated utilities (_safe_float, _normalize_filter_key, _parse_jd)
that were previously defined independently in 6-8 step files.
"""

from __future__ import annotations

import numpy as np
from astropy.time import Time


def safe_float(value, default: float = np.nan) -> float:
    """Safely convert value to float, returning *default* on failure."""
    try:
        return float(value)
    except Exception:
        return default


def normalize_filter_key(value: str | None) -> str:
    """Normalize a filter name to a lowercase, stripped key.

    Returns empty string for None/empty input.
    """
    if value is None:
        return ""
    return str(value).strip().lower()


def parse_jd(date_obs: str | None) -> float:
    """Parse a DATE-OBS string into Julian Date (float).

    Returns NaN on failure or empty input.
    """
    if not date_obs:
        return np.nan
    try:
        return float(Time(str(date_obs).strip()).jd)
    except Exception:
        return np.nan
