"""Shared aperture-photometry utilities.

Provides circle_mask, refine_local_centroid, phot_one_star (full CCD equation),
and small helper functions used by step9 (forced photometry) and the extinction tool.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from astropy.stats import SigmaClip, sigma_clipped_stats

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAG_ERR_COEFF = 2.5 / np.log(10)   # magnitude error coefficient (approx 1.0857)
MAD_TO_SIGMA = 1.4826               # MAD -> Gaussian sigma conversion factor


# ---------------------------------------------------------------------------
# Aperture helpers
# ---------------------------------------------------------------------------

def circle_mask(shape: tuple[int, int], cx: float, cy: float, r: float) -> np.ndarray:
    """Boolean mask for pixels inside a circle of radius *r* centred at (*cx*, *cy*)."""
    h, w = shape
    y = np.arange(h)[:, None]
    x = np.arange(w)[None, :]
    return (x - cx) ** 2 + (y - cy) ** 2 <= (r * r)


def refine_local_centroid(
    img: np.ndarray,
    x: float,
    y: float,
    fwhm_used: float,
    cbox_scale: float,
) -> tuple[float, float]:
    """Refine star centre using intensity-weighted centroid in a small box."""
    h, w = img.shape
    r = int(max(cbox_scale * max(float(fwhm_used), 2.0), 6.0))
    xi, yi = int(round(x)), int(round(y))
    x0, x1 = max(0, xi - r), min(w, xi + r + 1)
    y0, y1 = max(0, yi - r), min(h, yi + r + 1)
    if (x1 - x0) < 9 or (y1 - y0) < 9:
        return (x, y)

    cut = img[y0:y1, x0:x1]
    _, med, _ = sigma_clipped_stats(cut, sigma=3.0)
    z = cut - med
    z[~np.isfinite(z)] = 0.0
    z[z < 0] = 0.0
    s = np.nansum(z)
    if s <= 0:
        return (x, y)

    yy, xx = np.mgrid[y0:y1, x0:x1]
    xc = float(np.nansum(xx * z) / s)
    yc = float(np.nansum(yy * z) / s)
    return (xc, yc)


def phot_one_star(
    img: np.ndarray,
    xc: float,
    yc: float,
    r_ap: float,
    r_in: float,
    r_out: float,
    sigma_clip_val: float = 3.0,
    maxiters: int = 5,
    gain: float = 1.0,
    rn_param_e: float = 7.5,
    sky_frame_e: float = np.nan,
    sky_sigma_mode: str = "local",
    sky_sigma_includes_rn: bool = True,
    min_n_sky_for_local: int = 50,
    sat_adu: float = 60000.0,
    datamax_adu: float | None = None,
) -> tuple[float, float, float, float, float, float, float, int, bool, bool]:
    """Circular-aperture photometry for one star (full CCD equation).

    Returns
    -------
    (flux_e, sigma_e, snr, ap_sum, bkg_med, bkg_std, ap_area, n_sky,
     is_sat, is_nonlinear)
    """
    ap_mask = circle_mask(img.shape, xc, yc, r_ap)
    ann_out_mask = circle_mask(img.shape, xc, yc, r_out)
    ann_in_mask = circle_mask(img.shape, xc, yc, r_in)
    ann_mask = ann_out_mask & (~ann_in_mask)

    # --- Sky estimation ---
    ann_vals = img[ann_mask]
    ann_vals = ann_vals[np.isfinite(ann_vals)]
    if ann_vals.size:
        sc = SigmaClip(sigma=sigma_clip_val, maxiters=maxiters)
        vv = sc(ann_vals)
        vv = vv.compressed() if np.ma.isMaskedArray(vv) else np.asarray(vv)
        vv = vv[np.isfinite(vv)]
        bkg_med = float(np.nanmedian(vv)) if vv.size else float(np.nanmedian(ann_vals))
        bkg_std = (
            float(np.nanstd(vv, ddof=1))
            if vv.size > 1
            else float(np.nanstd(ann_vals, ddof=1))
            if ann_vals.size > 1
            else 0.0
        )
        n_sky = int(len(vv)) if vv.size else int(len(ann_vals))
    else:
        bkg_med = 0.0
        bkg_std = 0.0
        n_sky = 0

    # --- Aperture flux ---
    ap_sum = float(np.nansum(img[ap_mask]))
    ap_area = float(np.count_nonzero(ap_mask))
    flux_net_adu = ap_sum - bkg_med * ap_area
    flux_e = flux_net_adu * gain

    # --- Noise model (CCD equation) ---
    sigma_local_e2 = (max(bkg_std, 0.0) * gain) ** 2 if np.isfinite(bkg_std) else np.nan
    sigma_frame_e2 = float(sky_frame_e) ** 2 if np.isfinite(sky_frame_e) else np.nan

    mode = str(sky_sigma_mode or "local").strip().lower()
    use_local = (
        np.isfinite(sigma_local_e2)
        and bkg_std > 0
        and (n_sky >= min_n_sky_for_local)
    )

    if mode == "frame":
        sigma_pix_e2 = sigma_frame_e2 if np.isfinite(sigma_frame_e2) else sigma_local_e2
    elif mode == "max":
        sigma_pix_e2 = np.nanmax([sigma_local_e2, sigma_frame_e2])
    else:
        sigma_pix_e2 = sigma_local_e2 if use_local else sigma_frame_e2

    if not np.isfinite(sigma_pix_e2):
        sigma_pix_e2 = 0.0

    if sky_sigma_includes_rn:
        sigma_pix_e2 = max(sigma_pix_e2 - rn_param_e ** 2, 0.0)

    var_source = max(flux_e, 0.0)
    var_bkg_in_ap = ap_area * sigma_pix_e2
    var_bkg_est = (ap_area ** 2 / max(n_sky, 1)) * sigma_pix_e2
    var_readnoise = ap_area * rn_param_e ** 2 if sky_sigma_includes_rn else 0.0

    var_e = var_source + var_bkg_in_ap + var_bkg_est + var_readnoise
    sigma_e = float(np.sqrt(max(var_e, 0.0)))
    snr = float(flux_e / sigma_e) if sigma_e > 0 else np.nan

    # --- Saturation / nonlinearity flags ---
    peak_adu = float(np.nanmax(img[ap_mask])) if ap_area > 0 else np.nan
    is_sat = bool(np.isfinite(peak_adu) and (peak_adu >= float(sat_adu)))
    is_nonlinear = False
    if datamax_adu is not None and np.isfinite(datamax_adu) and float(datamax_adu) > 0:
        is_nonlinear = bool(np.isfinite(peak_adu) and (peak_adu >= float(datamax_adu)))

    return (flux_e, sigma_e, snr, ap_sum, bkg_med, bkg_std, ap_area, n_sky, is_sat, is_nonlinear)


# ---------------------------------------------------------------------------
# DataFrame helpers
# ---------------------------------------------------------------------------

def get_numeric_array(df: pd.DataFrame, col: str, default: float = np.nan) -> np.ndarray:
    """Extract a numeric column as a float64 numpy array, coercing errors."""
    return pd.to_numeric(df[col], errors="coerce").fillna(default).to_numpy(float)
