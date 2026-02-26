from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath

try:
    from scipy.optimize import curve_fit
except Exception:  # pragma: no cover
    curve_fit = None

try:
    from scipy.ndimage import gaussian_filter
except Exception:  # pragma: no cover
    gaussian_filter = None

try:
    from scipy.stats import gaussian_kde
except Exception:  # pragma: no cover
    gaussian_kde = None


@dataclass
class CenterSummary:
    iter_ra: float
    iter_dec: float
    kde_ra: float
    kde_dec: float
    moment_ra: float
    moment_dec: float
    auto_method: str
    auto_ra: float
    auto_dec: float
    iter_kde_sep_arcsec: float
    kde_grid_x: Optional[np.ndarray] = None
    kde_grid_y: Optional[np.ndarray] = None
    kde_grid_z: Optional[np.ndarray] = None


def angular_sep_arcsec(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    dra = ((ra2 - ra1 + 180.0) % 360.0) - 180.0
    ddec = dec2 - dec1
    cd = math.cos(math.radians(0.5 * (dec1 + dec2)))
    return float(math.hypot(dra * cd, ddec) * 3600.0)


def sky_to_local_xy(ra_deg: np.ndarray, dec_deg: np.ndarray, center_ra: float, center_dec: float) -> tuple[np.ndarray, np.ndarray]:
    ra = np.asarray(ra_deg, float)
    dec = np.asarray(dec_deg, float)
    dra = ((ra - center_ra + 180.0) % 360.0) - 180.0
    cosd = math.cos(math.radians(center_dec))
    x = dra * cosd * 3600.0
    y = (dec - center_dec) * 3600.0
    return x, y


def local_xy_to_sky(x_arcsec: np.ndarray, y_arcsec: np.ndarray, center_ra: float, center_dec: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x_arcsec, float)
    y = np.asarray(y_arcsec, float)
    cosd = max(math.cos(math.radians(center_dec)), 1e-8)
    ra = center_ra + (x / 3600.0) / cosd
    dec = center_dec + y / 3600.0
    return ra, dec


def _weighted_mean(ra: np.ndarray, dec: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    ww = np.asarray(w, float)
    ww = np.where(np.isfinite(ww) & (ww > 0), ww, 0.0)
    if np.sum(ww) <= 0:
        return float(np.nanmean(ra)), float(np.nanmean(dec))
    return float(np.sum(ra * ww) / np.sum(ww)), float(np.sum(dec * ww) / np.sum(ww))


def weighted_iterative_center(
    ra: np.ndarray,
    dec: np.ndarray,
    w: np.ndarray,
    radius_arcsec: float = 900.0,
    n_iter: int = 8,
) -> tuple[float, float]:
    cra, cdec = _weighted_mean(ra, dec, w)
    if not np.isfinite(cra) or not np.isfinite(cdec):
        return float(np.nan), float(np.nan)

    for _ in range(max(1, int(n_iter))):
        x, y = sky_to_local_xy(ra, dec, cra, cdec)
        rr = np.hypot(x, y)
        m = np.isfinite(rr) & (rr <= float(radius_arcsec))
        if int(np.sum(m)) < 5:
            break
        cra_new, cdec_new = _weighted_mean(ra[m], dec[m], w[m])
        if not (np.isfinite(cra_new) and np.isfinite(cdec_new)):
            break
        if angular_sep_arcsec(cra, cdec, cra_new, cdec_new) < 0.01:
            cra, cdec = cra_new, cdec_new
            break
        cra, cdec = cra_new, cdec_new
    return float(cra), float(cdec)


def kde_peak_center(
    ra: np.ndarray,
    dec: np.ndarray,
    w: np.ndarray,
    grid_size: int = 140,
) -> tuple[float, float, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    cra, cdec = _weighted_mean(ra, dec, w)
    x, y = sky_to_local_xy(ra, dec, cra, cdec)
    m = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(m)) < 10:
        return float(cra), float(cdec), None, None, None
    xx = x[m]
    yy = y[m]
    ww = np.asarray(w[m], float)
    ww = np.where(np.isfinite(ww) & (ww > 0), ww, 1e-6)

    qx = np.nanpercentile(xx, [1.0, 99.0])
    qy = np.nanpercentile(yy, [1.0, 99.0])
    padx = max(10.0, 0.15 * (qx[1] - qx[0]))
    pady = max(10.0, 0.15 * (qy[1] - qy[0]))
    gx = np.linspace(qx[0] - padx, qx[1] + padx, int(max(40, grid_size)))
    gy = np.linspace(qy[0] - pady, qy[1] + pady, int(max(40, grid_size)))
    X, Y = np.meshgrid(gx, gy)

    Z = None
    if gaussian_kde is not None:
        try:
            kde = gaussian_kde(np.vstack([xx, yy]), weights=ww)
            Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
        except Exception:
            Z = None

    if Z is None:
        bins_x = max(40, int(grid_size // 2))
        bins_y = max(40, int(grid_size // 2))
        hist, ex, ey = np.histogram2d(xx, yy, bins=[bins_x, bins_y], weights=ww)
        if gaussian_filter is not None:
            hist = gaussian_filter(hist, sigma=1.2)
        gx = 0.5 * (ex[:-1] + ex[1:])
        gy = 0.5 * (ey[:-1] + ey[1:])
        X, Y = np.meshgrid(gx, gy)
        Z = hist.T

    iy, ix = np.unravel_index(np.nanargmax(Z), Z.shape)
    x_peak = float(X[iy, ix])
    y_peak = float(Y[iy, ix])
    ra_peak, dec_peak = local_xy_to_sky(np.array([x_peak]), np.array([y_peak]), cra, cdec)
    return float(ra_peak[0]), float(dec_peak[0]), gx, gy, Z


def compute_centers(
    ra: np.ndarray,
    dec: np.ndarray,
    weights: np.ndarray,
    iter_radius_arcsec: float,
    grid_size: int,
) -> CenterSummary:
    moment_ra, moment_dec = _weighted_mean(ra, dec, weights)
    iter_ra, iter_dec = weighted_iterative_center(
        ra,
        dec,
        weights,
        radius_arcsec=float(iter_radius_arcsec),
        n_iter=8,
    )
    kde_ra, kde_dec, gx, gy, gz = kde_peak_center(ra, dec, weights, grid_size=int(grid_size))

    sep_iter_kde = angular_sep_arcsec(iter_ra, iter_dec, kde_ra, kde_dec)
    score_iter = local_density_score(ra, dec, weights, iter_ra, iter_dec, probe_arcsec=60.0)
    score_kde = local_density_score(ra, dec, weights, kde_ra, kde_dec, probe_arcsec=60.0)
    score_iter = float(score_iter if np.isfinite(score_iter) else 0.0)
    score_kde = float(score_kde if np.isfinite(score_kde) else 0.0)

    # Guardrail: when candidates are far apart, prefer iterative center for stability.
    sep_guard_arcsec = max(60.0, 0.05 * float(iter_radius_arcsec))
    if np.isfinite(sep_iter_kde) and (sep_iter_kde > sep_guard_arcsec):
        auto_method = "iter_sep_guard"
        auto_ra, auto_dec = iter_ra, iter_dec
    else:
        # KDE must be meaningfully better to win AUTO; otherwise keep iterative.
        kde_gain = score_kde / max(score_iter, 1e-12)
        if kde_gain >= 1.10:
            auto_method = "kde"
            auto_ra, auto_dec = kde_ra, kde_dec
        elif score_iter > 0:
            auto_method = "iter"
            auto_ra, auto_dec = iter_ra, iter_dec
        else:
            auto_method = "iter_fallback"
            auto_ra, auto_dec = iter_ra, iter_dec

    return CenterSummary(
        iter_ra=float(iter_ra),
        iter_dec=float(iter_dec),
        kde_ra=float(kde_ra),
        kde_dec=float(kde_dec),
        moment_ra=float(moment_ra),
        moment_dec=float(moment_dec),
        auto_method=auto_method,
        auto_ra=float(auto_ra),
        auto_dec=float(auto_dec),
        iter_kde_sep_arcsec=sep_iter_kde,
        kde_grid_x=gx,
        kde_grid_y=gy,
        kde_grid_z=gz,
    )


def local_density_score(ra: np.ndarray, dec: np.ndarray, w: np.ndarray, c_ra: float, c_dec: float, probe_arcsec: float) -> float:
    x, y = sky_to_local_xy(ra, dec, c_ra, c_dec)
    r = np.hypot(x, y)
    m = np.isfinite(r) & (r <= probe_arcsec)
    if not np.any(m):
        return 0.0
    ww = np.asarray(w[m], float)
    ww = np.where(np.isfinite(ww), ww, 0.0)
    area = math.pi * probe_arcsec * probe_arcsec
    return float(np.sum(ww) / max(area, 1e-9))


def weighted_quantile(values: np.ndarray, quantiles: np.ndarray, sample_weight: np.ndarray) -> np.ndarray:
    v = np.asarray(values, float)
    q = np.asarray(quantiles, float)
    w = np.asarray(sample_weight, float)
    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if int(np.sum(m)) == 0:
        return np.full_like(q, np.nan, dtype=float)
    v = v[m]
    w = w[m]
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cdf = np.cumsum(w)
    cdf /= cdf[-1]
    return np.interp(q, cdf, v)


def build_annulus_edges(
    radii_arcsec: np.ndarray,
    weights: np.ndarray,
    mode: str,
    r_min: float,
    r_max: float,
    fixed_width: float,
    n_equal_bins: int,
) -> np.ndarray:
    r = np.asarray(radii_arcsec, float)
    m = np.isfinite(r) & (r >= float(r_min)) & (r <= float(r_max))
    if int(np.sum(m)) < 5:
        return np.array([r_min, r_max], dtype=float)

    mode_key = str(mode).strip().lower()
    if mode_key.startswith("equal"):
        nq = max(3, int(n_equal_bins))
        q = np.linspace(0.0, 1.0, nq + 1)
        edges = weighted_quantile(r[m], q, np.asarray(weights[m], float))
        edges[0] = float(r_min)
        edges[-1] = float(r_max)
        edges = np.unique(np.asarray(edges, float))
        if len(edges) < 3:
            return np.array([r_min, r_max], dtype=float)
        return edges

    dr = max(1e-3, float(fixed_width))
    edges = np.arange(float(r_min), float(r_max) + dr, dr)
    if len(edges) < 3:
        edges = np.array([float(r_min), float(r_max)], dtype=float)
    if edges[-1] < float(r_max):
        edges = np.append(edges, float(r_max))
    return edges


def compute_area_efficiency(edges: np.ndarray, footprint_xy: np.ndarray, mc_samples: int, seed: int) -> pd.DataFrame:
    e = np.asarray(edges, float)
    poly = np.asarray(footprint_xy, float)
    if poly.ndim != 2 or poly.shape[1] != 2:
        raise RuntimeError("Invalid footprint polygon for area correction")

    path = MplPath(poly)
    rng = np.random.default_rng(int(seed))
    rows = []

    for i in range(len(e) - 1):
        r_in = float(e[i])
        r_out = float(e[i + 1])
        ann_area = math.pi * (r_out * r_out - r_in * r_in)
        nmc = max(200, int(mc_samples))

        u = rng.random(nmc)
        th = rng.uniform(0.0, 2.0 * math.pi, nmc)
        rr = np.sqrt(u * (r_out * r_out - r_in * r_in) + r_in * r_in)
        xs = rr * np.cos(th)
        ys = rr * np.sin(th)
        inside = path.contains_points(np.column_stack([xs, ys]))
        frac = float(np.mean(inside)) if len(inside) else 0.0
        area_eff = float(frac * ann_area)

        rows.append(
            {
                "bin": int(i),
                "r_in": r_in,
                "r_out": r_out,
                "A_annulus": float(ann_area),
                "A_eff": float(area_eff),
                "f_area": float(frac),
            }
        )

    return pd.DataFrame(rows)


def compute_profile(
    radii_arcsec: np.ndarray,
    weights: np.ndarray,
    edges: np.ndarray,
    area_df: pd.DataFrame,
) -> pd.DataFrame:
    r = np.asarray(radii_arcsec, float)
    w = np.asarray(weights, float)
    e = np.asarray(edges, float)

    idx = np.digitize(r, e) - 1
    nbin = len(e) - 1
    rows = []
    for b in range(nbin):
        m = idx == b
        ww = w[m]
        ww = np.where(np.isfinite(ww), ww, 0.0)
        nw = float(np.sum(ww))
        nstar = int(np.sum(m))
        sigma_n = float(np.sqrt(np.sum(ww * ww)))

        area_eff = float(area_df.loc[b, "A_eff"]) if b < len(area_df) else np.nan
        f_area = float(area_df.loc[b, "f_area"]) if b < len(area_df) else np.nan

        if np.isfinite(area_eff) and area_eff > 0:
            sigma = nw / area_eff
            sigma_err = sigma_n / area_eff
        else:
            sigma = np.nan
            sigma_err = np.nan

        r_in = float(e[b])
        r_out = float(e[b + 1])
        rows.append(
            {
                "bin": int(b),
                "r_in": r_in,
                "r_out": r_out,
                "r_mid": 0.5 * (r_in + r_out),
                "Nw": nw,
                "A_eff": area_eff,
                "f_area": f_area,
                "Sigma": float(sigma),
                "Sigma_err": float(sigma_err),
                "N_stars": nstar,
            }
        )

    return pd.DataFrame(rows)


def estimate_background_outer(profile_df: pd.DataFrame, n_outer: int = 3) -> float:
    if profile_df is None or profile_df.empty:
        return float("nan")
    s = pd.to_numeric(profile_df["Sigma"], errors="coerce").to_numpy(float)
    m = np.isfinite(s)
    if np.sum(m) == 0:
        return float("nan")
    s = s[m]
    n = max(1, min(int(n_outer), len(s)))
    return float(np.nanmedian(s[-n:]))


def eff_model(r: np.ndarray, sigma0: float, a: float, gamma: float, sigma_bg: float) -> np.ndarray:
    rr = np.asarray(r, float)
    aa = max(float(a), 1e-8)
    gg = max(float(gamma), 1e-8)
    return float(sigma0) * (1.0 + (rr / aa) ** 2) ** (-0.5 * gg) + float(sigma_bg)


def king_model(r: np.ndarray, sigma0: float, rc: float, rt: float, sigma_bg: float) -> np.ndarray:
    rr = np.asarray(r, float)
    rcv = max(float(rc), 1e-8)
    rtv = max(float(rt), rcv + 1e-8)
    term = (1.0 / np.sqrt(1.0 + (rr / rcv) ** 2)) - (1.0 / np.sqrt(1.0 + (rtv / rcv) ** 2))
    term = np.clip(term, 0.0, None)
    out = float(sigma0) * term * term + float(sigma_bg)
    out = np.where(rr >= rtv, float(sigma_bg), out)
    return out


def _fit_curve(
    model_name: str,
    r: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    bg_mode: str,
    bg_outer: float,
) -> dict:
    if curve_fit is None:
        return {"ok": False, "error": "scipy.optimize.curve_fit unavailable"}

    rr = np.asarray(r, float)
    yy = np.asarray(y, float)
    ee = np.asarray(yerr, float)

    m = np.isfinite(rr) & np.isfinite(yy)
    if np.sum(m) < 4:
        return {"ok": False, "error": "too_few_points"}
    rr = rr[m]
    yy = yy[m]
    ee = ee[m]

    e_med = np.nanmedian(ee[np.isfinite(ee) & (ee > 0)])
    if not np.isfinite(e_med) or e_med <= 0:
        e_med = max(np.nanmedian(np.abs(yy)) * 0.05, 1e-6)
    ee = np.where(np.isfinite(ee) & (ee > 0), ee, e_med)

    bg_joint = str(bg_mode).strip().lower().startswith("joint")
    y_min = max(0.0, float(np.nanmin(yy)))
    y_max = max(y_min + 1e-6, float(np.nanmax(yy)))
    r_max = max(float(np.nanmax(rr)), 1.0)

    try:
        if model_name == "eff":
            if bg_joint:
                p0 = [max(y_max - y_min, 1e-6), max(np.nanmedian(rr), 5.0), 3.0, y_min]
                lb = [0.0, 1e-3, 0.2, 0.0]
                ub = [max(1e8, 10.0 * y_max), 20.0 * r_max, 12.0, max(1e8, 10.0 * y_max)]

                popt, pcov = curve_fit(
                    eff_model,
                    rr,
                    yy,
                    sigma=ee,
                    absolute_sigma=True,
                    p0=p0,
                    bounds=(lb, ub),
                    maxfev=30000,
                )
                sigma0, a, gamma, bg = [float(v) for v in popt]
            else:
                bg = float(max(bg_outer, 0.0))

                def _eff_fix_bg(rin, sigma0, a, gamma):
                    return eff_model(rin, sigma0, a, gamma, bg)

                p0 = [max(y_max - bg, 1e-6), max(np.nanmedian(rr), 5.0), 3.0]
                lb = [0.0, 1e-3, 0.2]
                ub = [max(1e8, 10.0 * y_max), 20.0 * r_max, 12.0]
                popt, pcov = curve_fit(
                    _eff_fix_bg,
                    rr,
                    yy,
                    sigma=ee,
                    absolute_sigma=True,
                    p0=p0,
                    bounds=(lb, ub),
                    maxfev=30000,
                )
                sigma0, a, gamma = [float(v) for v in popt]

            y_hat = eff_model(rr, sigma0, a, gamma, bg)
            resid = yy - y_hat
            rss = float(np.sum(resid * resid))
            n = int(len(rr))
            k = 4 if bg_joint else 3
            rms = float(np.sqrt(np.mean(resid * resid)))
            aic = float(n * np.log(max(rss / max(n, 1), 1e-24)) + 2 * k)
            bic = float(n * np.log(max(rss / max(n, 1), 1e-24)) + k * np.log(max(n, 1)))

            params = {
                "Sigma0": sigma0,
                "a": a,
                "gamma": gamma,
                "Sigma_bg": bg,
            }

        elif model_name == "king":
            if bg_joint:
                p0 = [max(y_max - y_min, 1e-6), max(np.nanmedian(rr) * 0.4, 2.0), max(np.nanmax(rr) * 1.2, 20.0), y_min]
                lb = [0.0, 1e-3, 1e-2, 0.0]
                ub = [max(1e8, 10.0 * y_max), 10.0 * r_max, 100.0 * r_max, max(1e8, 10.0 * y_max)]

                popt, pcov = curve_fit(
                    king_model,
                    rr,
                    yy,
                    sigma=ee,
                    absolute_sigma=True,
                    p0=p0,
                    bounds=(lb, ub),
                    maxfev=50000,
                )
                sigma0, rc, rt, bg = [float(v) for v in popt]
            else:
                bg = float(max(bg_outer, 0.0))

                def _king_fix_bg(rin, sigma0, rc, rt):
                    return king_model(rin, sigma0, rc, rt, bg)

                p0 = [max(y_max - bg, 1e-6), max(np.nanmedian(rr) * 0.4, 2.0), max(np.nanmax(rr) * 1.2, 20.0)]
                lb = [0.0, 1e-3, 1e-2]
                ub = [max(1e8, 10.0 * y_max), 10.0 * r_max, 100.0 * r_max]
                popt, pcov = curve_fit(
                    _king_fix_bg,
                    rr,
                    yy,
                    sigma=ee,
                    absolute_sigma=True,
                    p0=p0,
                    bounds=(lb, ub),
                    maxfev=50000,
                )
                sigma0, rc, rt = [float(v) for v in popt]

            if rt <= rc:
                rt = rc * 1.05

            y_hat = king_model(rr, sigma0, rc, rt, bg)
            resid = yy - y_hat
            rss = float(np.sum(resid * resid))
            n = int(len(rr))
            k = 4 if bg_joint else 3
            rms = float(np.sqrt(np.mean(resid * resid)))
            aic = float(n * np.log(max(rss / max(n, 1), 1e-24)) + 2 * k)
            bic = float(n * np.log(max(rss / max(n, 1), 1e-24)) + k * np.log(max(n, 1)))

            params = {
                "Sigma0": sigma0,
                "rc": rc,
                "rt": rt,
                "Sigma_bg": bg,
            }

        else:
            return {"ok": False, "error": f"unknown_model:{model_name}"}

        errs = {}
        if "pcov" in locals() and isinstance(pcov, np.ndarray) and pcov.size > 0:
            d = np.sqrt(np.maximum(np.diag(pcov), 0.0))
            if model_name == "eff":
                names = ["Sigma0", "a", "gamma", "Sigma_bg"] if bg_joint else ["Sigma0", "a", "gamma"]
            else:
                names = ["Sigma0", "rc", "rt", "Sigma_bg"] if bg_joint else ["Sigma0", "rc", "rt"]
            for i, nm in enumerate(names):
                if i < len(d):
                    errs[nm] = float(d[i])
            if not bg_joint:
                errs["Sigma_bg"] = 0.0

        return {
            "ok": True,
            "model": model_name,
            "bg_mode": "joint" if bg_joint else "outer",
            "params": params,
            "errors": errs,
            "r_fit": rr,
            "y_fit": y_hat,
            "residuals": resid,
            "rms": rms,
            "aic": aic,
            "bic": bic,
            "n_points": int(len(rr)),
        }
    except Exception as e:
        return {"ok": False, "error": f"fit_failed:{type(e).__name__}:{e}"}


def fit_profile_models(
    profile_df: pd.DataFrame,
    model_key: str,
    bg_mode: str,
    bootstrap: bool,
    bootstrap_n: int,
    seed: int,
) -> dict:
    if profile_df is None or profile_df.empty:
        return {"eff": {"ok": False, "error": "empty_profile"}, "king": {"ok": False, "error": "empty_profile"}}

    rr = pd.to_numeric(profile_df["r_mid"], errors="coerce").to_numpy(float)
    yy = pd.to_numeric(profile_df["Sigma"], errors="coerce").to_numpy(float)
    ee = pd.to_numeric(profile_df["Sigma_err"], errors="coerce").to_numpy(float)

    bg_outer = estimate_background_outer(profile_df, n_outer=max(2, int(0.2 * len(profile_df))))

    do_eff = model_key in ("eff", "both")
    do_king = model_key in ("king", "both")

    out = {
        "eff": _fit_curve("eff", rr, yy, ee, bg_mode, bg_outer) if do_eff else {"ok": False, "error": "disabled"},
        "king": _fit_curve("king", rr, yy, ee, bg_mode, bg_outer) if do_king else {"ok": False, "error": "disabled"},
    }

    if bootstrap:
        rng = np.random.default_rng(int(seed))
        nboot = max(10, int(bootstrap_n))
        for mk in ("eff", "king"):
            fres = out.get(mk, {})
            if not fres.get("ok", False):
                continue
            keys = list(fres.get("params", {}).keys())
            samples = {k: [] for k in keys}
            for _ in range(nboot):
                yb = yy + rng.normal(0.0, np.where(np.isfinite(ee) & (ee > 0), ee, np.nanmedian(ee[np.isfinite(ee)])))
                fb = _fit_curve(mk, rr, yb, ee, bg_mode, bg_outer)
                if not fb.get("ok", False):
                    continue
                for k in keys:
                    v = fb.get("params", {}).get(k)
                    if v is not None and np.isfinite(v):
                        samples[k].append(float(v))
            bserr = {}
            for k in keys:
                arr = np.asarray(samples[k], float)
                if arr.size > 1 and np.isfinite(arr).any():
                    bserr[k] = float(np.nanstd(arr, ddof=1))
            if bserr:
                fres.setdefault("errors", {})
                for k, v in bserr.items():
                    fres["errors"][k] = float(v)
                fres["bootstrap_n"] = int(nboot)

    return out


def estimate_limiting_radius(profile_df: pd.DataFrame, sigma_bg: float) -> float:
    """Estimate limiting radius where Sigma(r) reaches background."""
    if profile_df is None or profile_df.empty:
        return float("nan")
    if not np.isfinite(float(sigma_bg)):
        return float("nan")

    r = pd.to_numeric(profile_df["r_mid"], errors="coerce").to_numpy(float)
    s = pd.to_numeric(profile_df["Sigma"], errors="coerce").to_numpy(float)
    m = np.isfinite(r) & np.isfinite(s)
    if int(np.sum(m)) < 2:
        return float("nan")
    r = r[m]
    s = s[m]
    order = np.argsort(r)
    r = r[order]
    s = s[order]

    y = s - float(sigma_bg)
    if y[0] <= 0:
        return float(r[0])

    for i in range(1, len(y)):
        if y[i] <= 0 and y[i - 1] > 0:
            dy = y[i] - y[i - 1]
            if abs(dy) < 1e-12:
                return float(r[i])
            t = -y[i - 1] / dy
            t = float(np.clip(t, 0.0, 1.0))
            return float(r[i - 1] + t * (r[i] - r[i - 1]))
    return float("nan")


def estimate_half_number_radius(profile_df: pd.DataFrame) -> float:
    """Estimate half-number radius from cumulative weighted counts."""
    if profile_df is None or profile_df.empty:
        return float("nan")

    rin = pd.to_numeric(profile_df["r_in"], errors="coerce").to_numpy(float)
    rout = pd.to_numeric(profile_df["r_out"], errors="coerce").to_numpy(float)
    nw = pd.to_numeric(profile_df["Nw"], errors="coerce").to_numpy(float)
    m = np.isfinite(rin) & np.isfinite(rout) & np.isfinite(nw)
    if int(np.sum(m)) < 1:
        return float("nan")
    rin = rin[m]
    rout = rout[m]
    nw = np.clip(nw[m], 0.0, None)

    order = np.argsort(rout)
    rin = rin[order]
    rout = rout[order]
    nw = nw[order]
    tot = float(np.sum(nw))
    if not np.isfinite(tot) or tot <= 0:
        return float("nan")

    target = 0.5 * tot
    csum = np.cumsum(nw)
    idx = int(np.searchsorted(csum, target, side="left"))
    if idx <= 0:
        frac = target / max(csum[0], 1e-12)
        frac = float(np.clip(frac, 0.0, 1.0))
        return float(rin[0] + frac * (rout[0] - rin[0]))
    if idx >= len(csum):
        return float(rout[-1])

    prev = float(csum[idx - 1])
    cur = float(csum[idx])
    frac = (target - prev) / max(cur - prev, 1e-12)
    frac = float(np.clip(frac, 0.0, 1.0))
    return float(rin[idx] + frac * (rout[idx] - rin[idx]))


def summarize_fit_choice(fit_out: dict) -> dict:
    eff = fit_out.get("eff", {}) if isinstance(fit_out, dict) else {}
    king = fit_out.get("king", {}) if isinstance(fit_out, dict) else {}

    eff_ok = bool(eff.get("ok", False))
    king_ok = bool(king.get("ok", False))
    if eff_ok and king_ok:
        aic_eff = float(eff.get("aic", np.nan))
        aic_king = float(king.get("aic", np.nan))
        if np.isfinite(aic_eff) and np.isfinite(aic_king):
            daic = aic_king - aic_eff
            if abs(daic) < 2.0:
                pref = "tie"
                note = "EFF/King 동급 (|ΔAIC|<2)"
            elif daic > 0:
                pref = "eff"
                note = "EFF 우세 (AIC)"
            else:
                pref = "king"
                note = "King 우세 (AIC)"
            return {"preferred": pref, "delta_aic_king_minus_eff": float(daic), "note": note}
        return {"preferred": "undetermined", "delta_aic_king_minus_eff": None, "note": "AIC unavailable"}
    if eff_ok:
        return {"preferred": "eff", "delta_aic_king_minus_eff": None, "note": "EFF only"}
    if king_ok:
        return {"preferred": "king", "delta_aic_king_minus_eff": None, "note": "King only"}
    return {"preferred": "none", "delta_aic_king_minus_eff": None, "note": "No valid fit"}


def estimate_shape_moments(x_arcsec: np.ndarray, y_arcsec: np.ndarray, weights: np.ndarray) -> dict:
    """Estimate 2D shape from weighted second moments in local tangent plane."""
    x = np.asarray(x_arcsec, float)
    y = np.asarray(y_arcsec, float)
    w = np.asarray(weights, float)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    if int(np.sum(m)) < 5:
        return {"available": False, "reason": "too_few_points", "n_used": int(np.sum(m))}

    x = x[m]
    y = y[m]
    w = w[m]
    sw = float(np.sum(w))
    if not np.isfinite(sw) or sw <= 0:
        return {"available": False, "reason": "non_positive_weight_sum", "n_used": int(len(w))}

    mx = float(np.sum(w * x) / sw)
    my = float(np.sum(w * y) / sw)
    dx = x - mx
    dy = y - my

    cxx = float(np.sum(w * dx * dx) / sw)
    cyy = float(np.sum(w * dy * dy) / sw)
    cxy = float(np.sum(w * dx * dy) / sw)
    cov = np.array([[cxx, cxy], [cxy, cyy]], dtype=float)

    try:
        evals, evecs = np.linalg.eigh(cov)
    except Exception:
        return {"available": False, "reason": "eigh_failed", "n_used": int(len(w))}
    order = np.argsort(evals)[::-1]
    evals = np.asarray(evals[order], float)
    evecs = np.asarray(evecs[:, order], float)

    l1 = float(max(evals[0], 0.0))
    l2 = float(max(evals[1], 0.0)) if len(evals) > 1 else 0.0
    a_sig = float(math.sqrt(l1)) if l1 > 0 else float("nan")
    b_sig = float(math.sqrt(l2)) if l2 > 0 else float("nan")
    ell = float(1.0 - (b_sig / a_sig)) if np.isfinite(a_sig) and np.isfinite(b_sig) and a_sig > 0 else float("nan")

    vx = float(evecs[0, 0]) if evecs.size >= 2 else float("nan")
    vy = float(evecs[1, 0]) if evecs.size >= 2 else float("nan")
    pa = float(np.degrees(np.arctan2(vy, vx)) % 180.0) if np.isfinite(vx) and np.isfinite(vy) else float("nan")

    return {
        "available": True,
        "n_used": int(len(w)),
        "sum_weight": sw,
        "x_centroid_arcsec": mx,
        "y_centroid_arcsec": my,
        "cov_xx": cxx,
        "cov_yy": cyy,
        "cov_xy": cxy,
        "a_sigma_arcsec": a_sig,
        "b_sigma_arcsec": b_sig,
        "ellipticity": ell,
        "axis_ratio_b_over_a": float(b_sig / a_sig) if np.isfinite(a_sig) and np.isfinite(b_sig) and a_sig > 0 else None,
        "pa_deg_from_x_ccw": pa,
        "pa_definition": "angle of major axis measured from +X(local east) toward +Y(local north), CCW, modulo 180deg",
    }
