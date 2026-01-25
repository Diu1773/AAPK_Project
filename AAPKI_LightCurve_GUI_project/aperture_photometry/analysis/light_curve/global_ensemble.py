"""Global ensemble photometry (inhomogeneous ensemble) solver.

This implements method C: solve per-frame zeropoints (Z_t) and per-star
means (M_i) simultaneously with weighted least squares:
    mag_inst(i,t,f) = M_i,f + Z_t,f + eps

The solution is per-filter by default. It supports iterative sigma-clipping
and comp-star rejection, then produces:
  - zp_df: per-frame zeropoints
  - mean_df: per-star mean instrumental mags
  - lc_df: target light curve corrected by Z_t
  - diagnostics: residual stats, removed comps, outliers
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr

REQUIRED_COLS = {"time_id", "jd", "filter", "star_id", "mag_inst", "err"}


def solve_global_ensemble(
    df: pd.DataFrame,
    target_id: int,
    comp_ids: Iterable[int],
    min_comps: int = 3,
    sigma: float = 3.0,
    n_iter: int = 3,
    gauge: str = "meanZ0",
    per_filter: bool = True,
    robust: bool = True,
    rms_clip_pct: float = 20.0,
    rms_clip_threshold: float | None = None,
    frame_sigma: float = 3.0,
    interp_missing: bool = False,
    normalize_target: bool = False,
    max_dense_params: int = 2000,
    log: Optional[Callable[[str], None]] = None,
) -> Dict[str, pd.DataFrame | dict]:
    """Solve global ensemble photometry (method C).

    Args:
        df: DataFrame with REQUIRED_COLS.
        target_id: Target star ID (excluded from solving).
        comp_ids: Comparison star IDs.
        min_comps: Minimum comps required per frame to solve Z_t.
        sigma: Sigma clip threshold.
        n_iter: Iteration count for comp/outlier rejection.
        gauge: "meanZ0" (default) or "ref".
        per_filter: Solve independently per filter.
        robust: Use MAD-based sigma for clipping.
        rms_clip_pct: Drop top % comps by RMS each iteration.
        rms_clip_threshold: Absolute RMS cutoff for comps (optional).
        frame_sigma: Frame-level outlier sigma threshold.
        interp_missing: Interpolate Z_t for frames with too few comps.
        normalize_target: Subtract median from corrected target curve.
        max_dense_params: Dense covariance threshold.
        log: Optional logger callback.
    """

    _log = log or (lambda _: None)
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty")
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    comp_ids = [int(c) for c in comp_ids if str(c).strip()]
    comp_ids = [c for c in comp_ids if c != int(target_id)]
    if not comp_ids:
        raise ValueError("comp_ids is empty after removing target_id")

    data = df.copy()
    data["time_id"] = data["time_id"].astype(str)
    data["filter"] = data["filter"].astype(str).str.strip().str.lower()
    data["star_id"] = pd.to_numeric(data["star_id"], errors="coerce").astype("Int64")
    data["mag_inst"] = pd.to_numeric(data["mag_inst"], errors="coerce")
    data["err"] = pd.to_numeric(data["err"], errors="coerce")
    data = data.dropna(subset=["time_id", "filter", "star_id", "mag_inst"])
    data = data[data["star_id"].notna()].copy()
    data["star_id"] = data["star_id"].astype(int)
    data["err"] = data["err"].where(data["err"] > 0, np.nan)

    if per_filter:
        filters = sorted(set(data["filter"].unique()))
    else:
        filters = [""]
        data["filter"] = ""

    zp_frames = []
    mean_frames = []
    lc_frames = []
    diagnostics: dict = {"filters": {}, "removed_comps": [], "outliers": []}

    for fkey in filters:
        sub = data[data["filter"] == fkey].copy()
        if sub.empty:
            continue
        _log(f"[GLOBAL] filter={fkey} rows={len(sub)} comps={len(comp_ids)}")

        result = _solve_one_filter(
            sub,
            target_id=target_id,
            comp_ids=comp_ids,
            min_comps=min_comps,
            sigma=sigma,
            n_iter=n_iter,
            gauge=gauge,
            robust=robust,
            rms_clip_pct=rms_clip_pct,
            rms_clip_threshold=rms_clip_threshold,
            frame_sigma=frame_sigma,
            interp_missing=interp_missing,
            normalize_target=normalize_target,
            max_dense_params=max_dense_params,
            log=_log,
        )

        zp = result["zp_df"]
        mean = result["mean_df"]
        lc = result["lc_df"]
        diag = result["diagnostics"]

        zp["filter"] = fkey
        mean["filter"] = fkey
        lc["filter"] = fkey

        zp_frames.append(zp)
        mean_frames.append(mean)
        lc_frames.append(lc)
        diagnostics["filters"][fkey] = diag
        diagnostics["removed_comps"].extend(diag.get("removed_comps", []))
        diagnostics["outliers"].extend(diag.get("outliers", []))

    zp_df = pd.concat(zp_frames, ignore_index=True) if zp_frames else pd.DataFrame()
    mean_df = pd.concat(mean_frames, ignore_index=True) if mean_frames else pd.DataFrame()
    lc_df = pd.concat(lc_frames, ignore_index=True) if lc_frames else pd.DataFrame()

    return {
        "zp_df": zp_df,
        "mean_df": mean_df,
        "lc_df": lc_df,
        "diagnostics": diagnostics,
    }


def _solve_one_filter(
    df: pd.DataFrame,
    target_id: int,
    comp_ids: List[int],
    min_comps: int,
    sigma: float,
    n_iter: int,
    gauge: str,
    robust: bool,
    rms_clip_pct: float,
    rms_clip_threshold: float | None,
    frame_sigma: float,
    interp_missing: bool,
    normalize_target: bool,
    max_dense_params: int,
    log: Callable[[str], None],
) -> Dict[str, pd.DataFrame | dict]:
    comp_active = [c for c in comp_ids if c != int(target_id)]
    removed_comps: List[int] = []
    outliers: List[dict] = []
    diag_iters = []

    target_df = df[df["star_id"] == int(target_id)].copy()
    comp_df_all = df[df["star_id"].isin(comp_active)].copy()

    if comp_df_all.empty:
        raise ValueError("No comparison star measurements found")

    # Track measurement-level removals
    comp_df = comp_df_all.copy()
    comp_df["keep"] = True

    for it in range(max(1, n_iter)):
        comp_df = comp_df[comp_df["star_id"].isin(comp_active)].copy()
        if comp_df.empty:
            break

        comp_df = _drop_frames_with_few_comps(comp_df, min_comps)
        if comp_df.empty:
            break

        fit = _solve_wls(comp_df, gauge=gauge, max_dense_params=max_dense_params, log=log)
        resid = comp_df["mag_inst"].to_numpy(float) - fit["model"]
        comp_df["resid"] = resid

        # Outlier rejection (measurement-level)
        sigma_global = _robust_sigma(resid) if robust else np.nanstd(resid)
        if not np.isfinite(sigma_global) or sigma_global <= 0:
            sigma_global = np.nanstd(resid) if np.isfinite(np.nanstd(resid)) else 0.0

        frame_sigmas = _frame_sigmas(comp_df, sigma_global, robust)
        comp_df["frame_sigma"] = comp_df["time_id"].map(frame_sigmas).fillna(sigma_global)
        bad = comp_df["frame_sigma"] > 0
        comp_df["outlier"] = False
        comp_df.loc[bad, "outlier"] = np.abs(comp_df["resid"]) > (frame_sigma * comp_df["frame_sigma"])
        out_rows = comp_df[comp_df["outlier"]]
        if not out_rows.empty:
            for _, r in out_rows.iterrows():
                outliers.append(
                    dict(
                        time_id=str(r["time_id"]),
                        star_id=int(r["star_id"]),
                        resid=float(r["resid"]),
                    )
                )
            comp_df = comp_df[~comp_df["outlier"]].copy()

        # Comp RMS and rejection
        rms_by_star = comp_df.groupby("star_id")["resid"].apply(lambda x: float(np.sqrt(np.mean(x**2))) if len(x) else np.nan)
        rms_by_star = rms_by_star.dropna()
        drop_ids = []
        if rms_clip_threshold is not None:
            drop_ids.extend(rms_by_star[rms_by_star > float(rms_clip_threshold)].index.tolist())
        if rms_clip_pct and rms_clip_pct > 0 and len(rms_by_star) > min_comps:
            n_drop = int(np.ceil(len(rms_by_star) * rms_clip_pct / 100.0))
            n_drop = max(0, min(len(rms_by_star) - min_comps, n_drop))
            if n_drop > 0:
                drop_ids.extend(rms_by_star.sort_values(ascending=False).head(n_drop).index.tolist())
        drop_ids = sorted(set(int(x) for x in drop_ids))

        if drop_ids:
            removed_comps.extend(drop_ids)
            comp_active = [c for c in comp_active if c not in drop_ids]

        diag_iters.append(
            dict(
                iter=it + 1,
                n_comp=len(comp_active),
                n_obs=len(comp_df),
                n_outliers=int(len(out_rows)),
            )
        )

        if len(comp_active) <= min_comps:
            break

    # Final solve
    comp_df = comp_df[comp_df["star_id"].isin(comp_active)].copy()
    comp_df = _drop_frames_with_few_comps(comp_df, min_comps)
    if comp_df.empty:
        raise ValueError("No usable comp measurements after clipping")

    fit = _solve_wls(comp_df, gauge=gauge, max_dense_params=max_dense_params, log=log)
    comp_df["resid"] = comp_df["mag_inst"].to_numpy(float) - fit["model"]

    # ZP table
    zp_df = _build_zp_df(comp_df, fit, min_comps=min_comps)
    if interp_missing:
        zp_df = _interp_missing_zp(zp_df)

    # Mean table
    mean_df = _build_mean_df(comp_df, fit)

    # Target light curve
    lc_df = _build_target_lc(
        target_df,
        comp_df,
        zp_df,
        min_comps=min_comps,
        normalize_target=normalize_target,
    )

    diagnostics = dict(
        removed_comps=removed_comps,
        outliers=outliers,
        iterations=diag_iters,
        n_comp_final=len(comp_active),
        n_obs_final=len(comp_df),
    )

    return {"zp_df": zp_df, "mean_df": mean_df, "lc_df": lc_df, "diagnostics": diagnostics}


def _solve_wls(
    df: pd.DataFrame,
    gauge: str,
    max_dense_params: int,
    log: Callable[[str], None],
) -> dict:
    # Map IDs
    star_ids = sorted(df["star_id"].unique().tolist())
    time_ids = sorted(df["time_id"].unique().tolist())
    star_map = {sid: i for i, sid in enumerate(star_ids)}
    time_map = {tid: i for i, tid in enumerate(time_ids)}

    ref_time = time_ids[0]
    n_star = len(star_ids)
    n_time = len(time_ids)
    n_time_params = n_time - 1  # ref Z fixed to 0

    n_params = n_star + n_time_params
    n_obs = len(df)

    # Build sparse A
    row = np.arange(n_obs, dtype=int)
    cols = np.empty(n_obs * 2, dtype=int)
    data = np.ones(n_obs * 2, dtype=float)

    star_idx = df["star_id"].map(star_map).to_numpy(int)
    time_idx = df["time_id"].map(time_map).to_numpy(int)
    time_col = np.where(time_idx == 0, -1, time_idx - 1)  # ref time removed

    cols[:n_obs] = star_idx
    cols[n_obs:] = n_star + time_col
    keep = cols[n_obs:] >= n_star
    cols = np.concatenate([cols[:n_obs], cols[n_obs:][keep]])
    data = np.concatenate([data[:n_obs], data[n_obs:][keep]])
    row = np.concatenate([row, row[keep]])

    A = csr_matrix((data, (row, cols)), shape=(n_obs, n_params))

    y = df["mag_inst"].to_numpy(float)
    err = df["err"].to_numpy(float)
    w = np.where(np.isfinite(err) & (err > 0), 1.0 / (err * err), 1.0)

    # Weighted LS via lsqr
    Aw = A.multiply(np.sqrt(w)[:, None])
    yw = y * np.sqrt(w)
    sol = lsqr(Aw, yw, atol=1e-10, btol=1e-10, iter_lim=2000)
    x = sol[0]

    # Build M and Z
    M = x[:n_star].copy()
    Z = np.zeros(n_time, dtype=float)
    Z[0] = 0.0
    if n_time_params > 0:
        Z[1:] = x[n_star:]

    # Gauge normalization
    if gauge.lower() == "meanz0":
        weights = df.groupby("time_id")["mag_inst"].count().reindex(time_ids).fillna(1).to_numpy(float)
        wsum = np.sum(weights)
        if wsum > 0:
            z_mean = float(np.sum(Z * weights) / wsum)
            Z = Z - z_mean
            M = M + z_mean

    model = M[star_idx] + Z[time_idx]

    # Error estimates
    approx = False
    M_err = np.full(n_star, np.nan)
    Z_err = np.full(n_time, np.nan)
    try:
        if n_params <= max_dense_params:
            AtW = A.T.multiply(w)
            AtWA = (AtW @ A).toarray()
            cov = np.linalg.pinv(AtWA)
            diag = np.diag(cov)
            M_err = np.sqrt(np.clip(diag[:n_star], 0, np.inf))
            z_diag = np.zeros(n_time, dtype=float)
            z_diag[0] = 0.0
            if n_time_params > 0:
                z_diag[1:] = diag[n_star:]
            Z_err = np.sqrt(np.clip(z_diag, 0, np.inf))
        else:
            AtW = A.T.multiply(w)
            AtWA = (AtW @ A).diagonal()
            diag = np.where(AtWA > 0, 1.0 / AtWA, np.nan)
            M_err = np.sqrt(np.clip(diag[:n_star], 0, np.inf))
            z_diag = np.zeros(n_time, dtype=float)
            z_diag[0] = 0.0
            if n_time_params > 0:
                z_diag[1:] = diag[n_star:]
            Z_err = np.sqrt(np.clip(z_diag, 0, np.inf))
            approx = True
    except Exception as e:
        approx = True
        log(f"[GLOBAL] Warning: covariance failed ({e})")

    return dict(
        M=M,
        Z=Z,
        M_err=M_err,
        Z_err=Z_err,
        model=model,
        star_ids=star_ids,
        time_ids=time_ids,
        approx_errors=approx,
    )


def _build_zp_df(comp_df: pd.DataFrame, fit: dict, min_comps: int) -> pd.DataFrame:
    time_ids = fit["time_ids"]
    Z = fit["Z"]
    Z_err = fit["Z_err"]

    stats = comp_df.groupby("time_id").apply(_frame_stats, include_groups=False)
    stats = stats.reindex(time_ids)
    n_used = stats.get("n_used", pd.Series([0] * len(time_ids))).to_numpy(int)
    chi2_red = stats.get("chi2_red", pd.Series([np.nan] * len(time_ids))).to_numpy(float)

    zp_df = pd.DataFrame(
        dict(
            time_id=time_ids,
            Z=Z,
            Z_err=Z_err,
            n_used=n_used,
            chi2_red=chi2_red,
        )
    )
    zp_df.loc[zp_df["n_used"] < min_comps, ["Z", "Z_err"]] = np.nan
    return zp_df


def _build_mean_df(comp_df: pd.DataFrame, fit: dict) -> pd.DataFrame:
    star_ids = fit["star_ids"]
    M = fit["M"]
    M_err = fit["M_err"]
    counts = comp_df.groupby("star_id")["mag_inst"].count().reindex(star_ids).fillna(0).to_numpy(int)
    return pd.DataFrame(
        dict(
            star_id=star_ids,
            M=M,
            M_err=M_err,
            n_used=counts,
        )
    )


def _build_target_lc(
    target_df: pd.DataFrame,
    comp_df: pd.DataFrame,
    zp_df: pd.DataFrame,
    min_comps: int,
    normalize_target: bool,
) -> pd.DataFrame:
    if target_df.empty:
        return pd.DataFrame()

    comp_mean = comp_df.groupby("time_id")["mag_inst"].mean()
    comp_n = comp_df.groupby("time_id")["mag_inst"].count()

    lc = target_df.copy()
    lc = lc.rename(columns={"mag_inst": "mag"})
    lc["diff_mag_raw"] = lc["mag"] - lc["time_id"].map(comp_mean)
    lc.loc[lc["time_id"].map(comp_n) < min_comps, "diff_mag_raw"] = np.nan

    zp_map = zp_df.set_index("time_id")["Z"]
    zp_err_map = zp_df.set_index("time_id")["Z_err"]
    lc["diff_mag_corr"] = lc["mag"] - lc["time_id"].map(zp_map)
    lc["diff_err_corr"] = np.sqrt(
        np.square(lc["err"].to_numpy(float)) + np.square(lc["time_id"].map(zp_err_map).to_numpy(float))
    )

    if normalize_target:
        med = np.nanmedian(lc["diff_mag_corr"].to_numpy(float))
        if np.isfinite(med):
            lc["diff_mag_corr"] = lc["diff_mag_corr"] - med

    lc = lc.rename(columns={"err": "diff_err"})
    return lc[
        [
            "jd",
            "time_id",
            "filter",
            "star_id",
            "mag",
            "diff_mag_raw",
            "diff_mag_corr",
            "diff_err",
            "diff_err_corr",
        ]
    ]


def _drop_frames_with_few_comps(df: pd.DataFrame, min_comps: int) -> pd.DataFrame:
    counts = df.groupby("time_id")["star_id"].nunique()
    keep = counts[counts >= min_comps].index
    return df[df["time_id"].isin(keep)].copy()


def _frame_stats(group: pd.DataFrame) -> pd.Series:
    resid = group["resid"].to_numpy(float)
    err = group["err"].to_numpy(float)
    ok = np.isfinite(resid) & np.isfinite(err) & (err > 0)
    chi2 = np.sum((resid[ok] / err[ok]) ** 2) if np.any(ok) else np.nan
    dof = max(int(np.sum(ok)) - 1, 1)
    return pd.Series(
        dict(
            n_used=int(len(group)),
            chi2_red=float(chi2 / dof) if np.isfinite(chi2) else np.nan,
        )
    )


def _frame_sigmas(df: pd.DataFrame, sigma_global: float, robust: bool) -> Dict[str, float]:
    sigmas: Dict[str, float] = {}
    for tid, sub in df.groupby("time_id"):
        vals = sub["resid"].to_numpy(float)
        if len(vals) >= 5:
            sig = _robust_sigma(vals) if robust else np.nanstd(vals)
            sig = sig if np.isfinite(sig) and sig > 0 else sigma_global
        else:
            sig = sigma_global
        sigmas[str(tid)] = float(sig)
    return sigmas


def _robust_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if np.isfinite(mad) and mad > 0:
        return float(1.4826 * mad)
    return float(np.nanstd(x))


def _interp_missing_zp(zp_df: pd.DataFrame) -> pd.DataFrame:
    if zp_df.empty or "Z" not in zp_df.columns:
        return zp_df
    zp_df = zp_df.copy()
    z = pd.to_numeric(zp_df["Z"], errors="coerce")
    zp_df["Z"] = z.interpolate(limit_direction="both")
    return zp_df


def generate_synthetic_data(
    n_frames: int = 200,
    n_comps: int = 20,
    period: float = 0.35,
    noise: float = 0.01,
    seed: int = 42,
) -> Tuple[pd.DataFrame, int, List[int], dict]:
    """Create synthetic dataset for validation."""
    rng = np.random.default_rng(seed)
    time_ids = [f"f{i:04d}.fit" for i in range(n_frames)]
    jd = np.linspace(0.0, 2.0, n_frames) + 2450000.0
    filters = ["g"] * n_frames
    target_id = 1
    comp_ids = list(range(2, 2 + n_comps))

    # True per-frame ZP offsets (nightly jumps)
    Z_true = np.zeros(n_frames)
    for k in range(0, n_frames, 50):
        Z_true[k:k + 50] = rng.normal(0, 0.05)

    # Star means
    M_comp = rng.normal(12.0, 0.5, size=n_comps)
    M_target = 12.3
    target_signal = 0.2 * np.sin(2 * np.pi * (jd - jd.min()) / period)

    rows = []
    for t, tid in enumerate(time_ids):
        for i, sid in enumerate(comp_ids):
            mag = M_comp[i] + Z_true[t] + rng.normal(0, noise)
            rows.append(dict(time_id=tid, jd=jd[t], filter="g", star_id=sid, mag_inst=mag, err=noise))
        mag_t = M_target + Z_true[t] + target_signal[t] + rng.normal(0, noise)
        rows.append(dict(time_id=tid, jd=jd[t], filter="g", star_id=target_id, mag_inst=mag_t, err=noise))

    df = pd.DataFrame(rows)
    truth = dict(Z_true=Z_true, jd=jd)
    return df, target_id, comp_ids, truth


def run_synthetic_test() -> dict:
    """Basic verification: recover injected frame offsets."""
    df, target_id, comp_ids, truth = generate_synthetic_data()
    result = solve_global_ensemble(df, target_id, comp_ids, min_comps=5, n_iter=2)
    zp = result["zp_df"]
    z = pd.to_numeric(zp["Z"], errors="coerce").to_numpy(float)
    ok = np.isfinite(z)
    if np.any(ok):
        corr = np.corrcoef(z[ok], truth["Z_true"][ok])[0, 1]
    else:
        corr = np.nan
    return {"corr_Z": float(corr), "n_points": int(np.sum(ok))}


def example_usage() -> None:
    """Example usage snippet."""
    df, target_id, comp_ids, _ = generate_synthetic_data()
    result = solve_global_ensemble(df, target_id, comp_ids)
    print(result["zp_df"].head())
