"""Pure helpers for merged ID reconciliation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u


def extract_row_float(row: pd.Series, *cols: str) -> float:
    for col in cols:
        if col in row.index:
            val = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
            if np.isfinite(val):
                return float(val)
    return float("nan")


def row_radec(row: pd.Series) -> tuple[float, float]:
    return (
        extract_row_float(row, "ra_deg", "ra", "RA"),
        extract_row_float(row, "dec_deg", "dec", "DEC"),
    )


def append_folder_tag(existing: str, tag: str) -> str:
    parts = [p for p in str(existing or "").split(",") if p]
    if tag not in parts:
        parts.append(tag)
    return ",".join(parts)


def best_positional_match(row: pd.Series, canonical_df: pd.DataFrame, tol_arcsec: float) -> tuple[int | None, float]:
    ra, dec = row_radec(row)
    if not (np.isfinite(ra) and np.isfinite(dec)):
        return None, float("nan")
    if canonical_df is None or canonical_df.empty or "ra_deg" not in canonical_df.columns or "dec_deg" not in canonical_df.columns:
        return None, float("nan")

    cand = canonical_df.copy()
    cand_ra = pd.to_numeric(cand["ra_deg"], errors="coerce")
    cand_dec = pd.to_numeric(cand["dec_deg"], errors="coerce")
    mask = cand_ra.notna() & cand_dec.notna()
    if not mask.any():
        return None, float("nan")

    sc = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
    csc = SkyCoord(cand_ra[mask].to_numpy(float) * u.deg, cand_dec[mask].to_numpy(float) * u.deg, frame="icrs")
    sep = sc.separation(csc).arcsec
    if len(sep) == 0:
        return None, float("nan")
    best_i = int(np.argmin(sep))
    best_sep = float(sep[best_i])
    if not np.isfinite(best_sep) or best_sep > tol_arcsec:
        return None, best_sep
    best_rows = cand.loc[mask].reset_index(drop=True)
    return int(pd.to_numeric(best_rows.loc[best_i, "source_id"], errors="coerce")), best_sep


def canonicalize_catalog_row(
    row: pd.Series,
    merged_id: int,
    merged_source_id: int,
    folder_tag: str,
) -> dict:
    data = row.to_dict()
    data["ID"] = int(merged_id)
    data["source_id"] = int(merged_source_id)
    data["gaia_id"] = int(merged_source_id) if int(merged_source_id) > 0 else np.nan
    data["match_status"] = "matched" if int(merged_source_id) > 0 else "no_gaia_match"
    data["folder_count"] = 1
    data["folder_tags"] = folder_tag
    return data

