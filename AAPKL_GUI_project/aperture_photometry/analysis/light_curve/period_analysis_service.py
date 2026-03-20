"""Reusable period-analysis service functions."""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks


def filter_valid(
    time: np.ndarray,
    mag: np.ndarray,
    mag_err: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    mask = np.isfinite(time) & np.isfinite(mag)
    if mag_err is not None:
        mask &= np.isfinite(mag_err)
    t = time[mask]
    y = mag[mask]
    dy = mag_err[mask] if mag_err is not None else None
    return t, y, dy


def compute_ls(
    time: np.ndarray,
    mag: np.ndarray,
    mag_err: Optional[np.ndarray],
    label: str,
    min_period: float,
    max_period: float,
    samples_per_peak: int,
) -> dict:
    t, y, dy = filter_valid(time, mag, mag_err)

    if len(t) < 10:
        return {
            "label": label, "method": "ls",
            "error": "Not enough data points (< 10)",
            "best_period": np.nan, "best_power": np.nan,
        }

    if dy is not None and np.any(dy > 0):
        ls = LombScargle(t, y, dy)
    else:
        ls = LombScargle(t, y)

    frequency, power = ls.autopower(
        minimum_frequency=1.0 / max_period,
        maximum_frequency=1.0 / min_period,
        samples_per_peak=samples_per_peak,
    )

    best_idx = np.argmax(power)
    best_freq = float(frequency[best_idx])
    best_period = 1.0 / best_freq
    best_power = float(power[best_idx])

    try:
        fap = float(ls.false_alarm_probability(best_power))
    except Exception:
        fap = np.nan

    peak_indices, _ = find_peaks(power, height=0.1 * best_power)
    if len(peak_indices) == 0:
        peak_indices = [best_idx]
    sorted_peaks = sorted(peak_indices, key=lambda i: power[i], reverse=True)[:5]
    top_periods = [1.0 / float(frequency[i]) for i in sorted_peaks]
    top_powers = [float(power[i]) for i in sorted_peaks]

    return {
        "label": label, "method": "ls",
        "frequency": np.array(frequency, dtype=float),
        "power": np.array(power, dtype=float),
        "best_period": best_period, "best_power": best_power, "fap": fap,
        "top_periods": top_periods, "top_powers": top_powers,
        "n_points": len(t), "time": t, "mag": y, "mag_err": dy,
    }


def compute_pdm(
    time: np.ndarray,
    mag: np.ndarray,
    mag_err: Optional[np.ndarray],
    label: str,
    min_period: float,
    max_period: float,
    samples_per_peak: int,
    pdm_n_bins: int,
) -> dict:
    t, y, dy = filter_valid(time, mag, mag_err)

    if len(t) < 10:
        return {
            "label": label, "method": "pdm",
            "error": "Not enough data points (< 10)",
            "best_period": np.nan, "best_power": np.nan,
        }

    baseline = t.max() - t.min()
    n_trials = min(
        50000,
        int(samples_per_peak * baseline / min_period)
    )
    trial_periods = np.linspace(min_period, max_period, n_trials)

    var_total = np.var(y)
    if var_total == 0:
        return {
            "label": label, "method": "pdm",
            "error": "Zero variance in data",
            "best_period": np.nan, "best_power": np.nan,
        }

    theta = np.ones(n_trials)
    n_bins = pdm_n_bins
    t_min = t.min()
    dt = t - t_min
    y2 = y * y
    for i in range(n_trials):
        phase = (dt / trial_periods[i]) % 1.0
        bi = np.clip((phase * n_bins).astype(np.int32), 0, n_bins - 1)
        counts = np.bincount(bi, minlength=n_bins)
        sums = np.bincount(bi, weights=y, minlength=n_bins)
        sum_sq = np.bincount(bi, weights=y2, minlength=n_bins)
        good = counts >= 2
        if good.any():
            c_g = counts[good]
            ss = sum_sq[good] - sums[good] ** 2 / c_g
            dof = c_g - 1
            theta[i] = ss.sum() / dof.sum() / var_total

    power = 1.0 - theta

    best_idx = np.argmax(power)
    best_period = float(trial_periods[best_idx])
    best_power = float(power[best_idx])
    best_theta = float(theta[best_idx])

    peak_indices, _ = find_peaks(power, height=0.1 * best_power)
    if len(peak_indices) == 0:
        peak_indices = [best_idx]
    sorted_peaks = sorted(peak_indices, key=lambda i: power[i], reverse=True)[:5]
    top_periods = [trial_periods[i] for i in sorted_peaks]
    top_powers = [power[i] for i in sorted_peaks]

    return {
        "label": label, "method": "pdm",
        "trial_periods": trial_periods,
        "theta": theta, "power": power,
        "best_period": best_period, "best_power": best_power,
        "best_theta": best_theta, "fap": np.nan,
        "top_periods": top_periods, "top_powers": top_powers,
        "n_points": len(t), "time": t, "mag": y, "mag_err": dy,
    }


def compute_bls(
    time: np.ndarray,
    mag: np.ndarray,
    mag_err: Optional[np.ndarray],
    label: str,
    min_period: float,
    max_period: float,
) -> dict:
    from astropy.timeseries import BoxLeastSquares

    t, y, dy = filter_valid(time, mag, mag_err)

    if len(t) < 10:
        return {
            "label": label, "method": "bls",
            "error": "Not enough data points (< 10)",
            "best_period": np.nan, "best_power": np.nan,
        }

    if dy is not None and np.any(dy > 0):
        bls = BoxLeastSquares(t, y, dy=dy)
    else:
        bls = BoxLeastSquares(t, y)

    baseline = t.max() - t.min()
    max_dur = min(min_period * 0.5, max_period * 0.25, baseline * 0.25)
    min_dur = max(min_period * 0.01, max_dur * 0.05)
    if min_dur >= max_dur:
        min_dur = max_dur * 0.1
    durations = np.linspace(min_dur, max_dur, 10)

    try:
        result = bls.autopower(
            durations,
            minimum_period=min_period,
            maximum_period=max_period,
        )
    except Exception as e:
        return {
            "label": label, "method": "bls",
            "error": f"BLS failed: {e}",
            "best_period": np.nan, "best_power": np.nan,
        }

    power = result.power
    periods = result.period

    best_idx = np.argmax(power)
    best_period = float(periods[best_idx])
    best_power = float(power[best_idx])

    peak_indices, _ = find_peaks(power, height=0.1 * best_power)
    if len(peak_indices) == 0:
        peak_indices = [best_idx]
    sorted_peaks = sorted(peak_indices, key=lambda i: power[i], reverse=True)[:5]
    top_periods = [float(periods[i]) for i in sorted_peaks]
    top_powers = [float(power[i]) for i in sorted_peaks]

    return {
        "label": label, "method": "bls",
        "trial_periods": np.array(periods, dtype=float),
        "power": np.array(power, dtype=float),
        "best_period": best_period, "best_power": best_power,
        "fap": np.nan,
        "top_periods": top_periods, "top_powers": top_powers,
        "n_points": len(t), "time": t, "mag": y, "mag_err": dy,
    }


def run_period_analysis(
    time: np.ndarray,
    mag_raw: np.ndarray,
    mag_corr: Optional[np.ndarray],
    mag_err: Optional[np.ndarray],
    min_period: float,
    max_period: float,
    samples_per_peak: int = 10,
    methods: Optional[List[str]] = None,
    pdm_n_bins: int = 10,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> dict:
    results = {}
    for method in (methods or ["ls"]):
        if progress_cb is not None:
            progress_cb(f"Computing {method.upper()} for raw magnitudes...")
        results[f"raw_{method}"] = _compute_method(
            time, mag_raw, mag_err, "raw", method,
            min_period, max_period, samples_per_peak, pdm_n_bins,
        )
        if mag_corr is not None and np.any(np.isfinite(mag_corr)):
            if progress_cb is not None:
                progress_cb(f"Computing {method.upper()} for corrected magnitudes...")
            results[f"corr_{method}"] = _compute_method(
                time, mag_corr, mag_err, "corr", method,
                min_period, max_period, samples_per_peak, pdm_n_bins,
            )
    return results


def _compute_method(
    time: np.ndarray,
    mag: np.ndarray,
    mag_err: Optional[np.ndarray],
    label: str,
    method: str,
    min_period: float,
    max_period: float,
    samples_per_peak: int,
    pdm_n_bins: int,
) -> dict:
    if method == "ls":
        return compute_ls(time, mag, mag_err, label, min_period, max_period, samples_per_peak)
    if method == "pdm":
        return compute_pdm(time, mag, mag_err, label, min_period, max_period, samples_per_peak, pdm_n_bins)
    if method == "bls":
        return compute_bls(time, mag, mag_err, label, min_period, max_period)
    return {"label": label, "method": method, "error": f"Unknown method: {method}"}
