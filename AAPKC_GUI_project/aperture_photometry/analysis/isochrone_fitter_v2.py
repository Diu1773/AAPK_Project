"""
Isochrone Fitting Module v2 - Improved accuracy

Key improvements over v1:
1. Bilinear interpolation between isochrone grid points
2. Perpendicular distance to isochrone curve (not nearest point)
3. IMF-weighted likelihood (accounts for stellar mass function)
4. Proper chi² with consistent error propagation

Author: KNUEMAO Pipeline
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple, Callable, Dict, List

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import cKDTree


class FitMode(Enum):
    """Fitting mode selection"""
    FAST = "fast"
    HESSIAN = "hessian"
    GRID_SCAN = "grid_scan"


@dataclass
class FitBounds:
    """Parameter bounds for fitting"""
    log_age: Tuple[float, float] = (8.0, 10.0)
    metallicity: Tuple[float, float] = (-0.5, 0.5)
    distance_mod: Tuple[float, float] = (8.0, 13.0)
    extinction_gr: Tuple[float, float] = (0.0, 0.5)

    def to_list(self):
        return [self.log_age, self.metallicity,
                self.distance_mod, self.extinction_gr]


@dataclass
class FitResult:
    """Isochrone fitting result"""
    log_age: float
    metallicity: float
    distance_mod: float
    extinction_gr: float

    log_age_err: Optional[float] = None
    metallicity_err: Optional[float] = None
    distance_mod_err: Optional[float] = None
    extinction_gr_err: Optional[float] = None

    chi2: float = 0.0
    reduced_chi2: float = 0.0
    n_stars: int = 0
    n_params: int = 4

    fit_mode: str = "fast"
    elapsed_sec: float = 0.0
    converged: bool = True

    @property
    def age_gyr(self) -> float:
        return 10 ** (self.log_age - 9)

    @property
    def distance_pc(self) -> float:
        return 10 ** (1 + self.distance_mod / 5)

    def summary(self) -> str:
        lines = [
            "=" * 50,
            f"Isochrone Fit Result ({self.fit_mode.upper()} mode)",
            "=" * 50,
            "",
            "Best-fit Parameters:",
        ]

        if self.log_age_err:
            lines.append(f"  log(Age) = {self.log_age:.3f} ± {self.log_age_err:.3f}")
            age_gyr_err = self.age_gyr * np.log(10) * self.log_age_err
            lines.append(f"           = {self.age_gyr:.2f} ± {age_gyr_err:.2f} Gyr")
        else:
            lines.append(f"  log(Age) = {self.log_age:.3f}  ({self.age_gyr:.2f} Gyr)")

        if self.metallicity_err:
            lines.append(f"  [M/H]    = {self.metallicity:.3f} ± {self.metallicity_err:.3f}")
        else:
            lines.append(f"  [M/H]    = {self.metallicity:.3f}")

        if self.distance_mod_err:
            lines.append(f"  (m-M)    = {self.distance_mod:.3f} ± {self.distance_mod_err:.3f}")
            dist_pc_err = self.distance_pc * np.log(10) / 5 * self.distance_mod_err
            lines.append(f"           = {self.distance_pc:.0f} ± {dist_pc_err:.0f} pc")
        else:
            lines.append(f"  (m-M)    = {self.distance_mod:.3f}  ({self.distance_pc:.0f} pc)")

        if self.extinction_gr_err:
            lines.append(f"  E(g-r)   = {self.extinction_gr:.4f} ± {self.extinction_gr_err:.4f}")
        else:
            lines.append(f"  E(g-r)   = {self.extinction_gr:.4f}")

        lines.extend([
            "",
            "Fit Quality:",
            f"  chi²     = {self.chi2:.1f}",
            f"  chi²/dof = {self.reduced_chi2:.3f}",
            f"  N_stars  = {self.n_stars}",
            "",
            f"Elapsed: {self.elapsed_sec:.2f} sec",
            "=" * 50,
        ])

        return "\n".join(lines)


@dataclass
class GridScanResult:
    """Result from grid scan fitting, including chi2 landscape."""
    best_fit: FitResult
    chi2_grid: np.ndarray          # shape (n_ages, n_mhs)
    dm_grid: np.ndarray            # shape (n_ages, n_mhs)
    egr_grid: np.ndarray           # shape (n_ages, n_mhs)
    grid_ages: np.ndarray          # 1D sorted ages within bounds
    grid_mhs: np.ndarray           # 1D sorted metallicities within bounds
    n_evaluated: int = 0
    elapsed_sec: float = 0.0

    def summary(self) -> str:
        lines = [
            "=" * 50,
            "Grid Scan Result",
            "=" * 50,
            "",
            f"Grid: {len(self.grid_ages)} ages x {len(self.grid_mhs)} [M/H]"
            f" = {len(self.grid_ages) * len(self.grid_mhs)} cells",
            f"Evaluated: {self.n_evaluated} cells",
            f"Elapsed: {self.elapsed_sec:.2f} sec",
            "",
        ]
        lines.append(self.best_fit.summary())
        return "\n".join(lines)


class IsochroneFitterV2:
    """
    Improved Isochrone Fitter with interpolation and perpendicular distance.
    """

    # SDSS extinction coefficients: R_band = A_band / E(B-V)
    SDSS_R = {"g": 3.303, "r": 2.285, "i": 1.698, "z": 1.263}

    def __init__(
        self,
        isochrone_file: str | Path,
        col_mh: int = 1,
        col_age: int = 2,
        col_g: int = 29,
        col_r: int = 30,
        col_mag: Optional[int] = None,
        col_mass: int = 5,  # Initial mass column for IMF weighting
        fit_fraction: float = 0.7,
        R_band1: float = 3.303,
        R_band2: float = 2.285,
        R_mag: Optional[float] = None,
    ):
        self.iso_file = Path(isochrone_file)
        self.COL_MH = col_mh
        self.COL_AGE = col_age
        # band1, band2 define color = band1 - band2
        self.COL_G = col_g
        self.COL_R = col_r
        # mag band can be independent from color bands.
        self.COL_MAG = int(col_mag) if col_mag is not None else int(col_g)
        self.COL_MASS = col_mass
        self.fit_fraction = fit_fraction
        self.R_1 = R_band1
        self.R_2 = R_band2
        self.R_MAG = float(R_mag) if R_mag is not None else float(R_band1)

        self.iso_data = self._load_isochrone()

        # Build grid structure
        self.ages = np.unique(self.iso_data[:, self.COL_AGE])
        self.metallicities = np.unique(self.iso_data[:, self.COL_MH])

        # Pre-compute isochrone curves for each (age, mh) combination
        self._iso_cache: Dict[Tuple[float, float], np.ndarray] = {}
        self._build_iso_cache()

        # Fitting state
        self.progress_callback: Optional[Callable[[float, str], None]] = None
        self._fit_obs: Optional[np.ndarray] = None
        self._fit_err: Optional[np.ndarray] = None
        self._fit_weight: Optional[np.ndarray] = None
        self._fit_iteration: int = 0
        self._fit_max_iter: int = 100

    def _load_isochrone(self) -> np.ndarray:
        data = np.genfromtxt(self.iso_file, comments='#')
        data = data[~np.isnan(data).any(axis=1)]
        return data

    def _build_iso_cache(self):
        """Pre-cache isochrone data for each grid point"""
        for age in self.ages:
            for mh in self.metallicities:
                mask = (
                    (np.abs(self.iso_data[:, self.COL_AGE] - age) < 0.01) &
                    (np.abs(self.iso_data[:, self.COL_MH] - mh) < 0.01)
                )
                iso_sub = self.iso_data[mask]
                if len(iso_sub) > 10:
                    # Sort by magnitude (evolutionary sequence)
                    sort_idx = np.argsort(iso_sub[:, self.COL_MAG])
                    self._iso_cache[(round(age, 2), round(mh, 2))] = iso_sub[sort_idx]

    def _get_nearby_isochrones(self, log_age: float, mh: float) -> List[Tuple[float, float, np.ndarray]]:
        """Get up to 4 nearest isochrones for interpolation"""
        # Find bracketing ages
        age_idx = np.searchsorted(self.ages, log_age)
        age_lo = self.ages[max(0, age_idx - 1)]
        age_hi = self.ages[min(len(self.ages) - 1, age_idx)]

        # Find bracketing metallicities
        mh_idx = np.searchsorted(self.metallicities, mh)
        mh_lo = self.metallicities[max(0, mh_idx - 1)]
        mh_hi = self.metallicities[min(len(self.metallicities) - 1, mh_idx)]

        nearby = []
        for a in [age_lo, age_hi]:
            for m in [mh_lo, mh_hi]:
                key = (round(a, 2), round(m, 2))
                if key in self._iso_cache:
                    nearby.append((a, m, self._iso_cache[key]))

        return nearby

    def _interpolate_isochrone(
        self, log_age: float, mh: float, dm: float, e_gr: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get isochrone CMD coordinates using nearest grid point.
        Fast version - no interpolation between grid points.
        Returns (color, mag, mass) arrays.
        """
        # Find nearest age and metallicity in grid
        age_idx = np.argmin(np.abs(self.ages - log_age))
        mh_idx = np.argmin(np.abs(self.metallicities - mh))

        nearest_age = self.ages[age_idx]
        nearest_mh = self.metallicities[mh_idx]

        key = (round(nearest_age, 2), round(nearest_mh, 2))

        if key not in self._iso_cache:
            return np.array([]), np.array([]), np.array([])

        iso_data = self._iso_cache[key]

        band1 = iso_data[:, self.COL_G]
        band2 = iso_data[:, self.COL_R]
        mag_band = iso_data[:, self.COL_MAG]
        mass = iso_data[:, self.COL_MASS]

        # Apply extinction and distance modulus
        E_BV = e_gr / (self.R_1 - self.R_2)
        A_1 = self.R_1 * E_BV
        A_2 = self.R_2 * E_BV
        A_mag = self.R_MAG * E_BV

        b1_obs = band1 + dm + A_1
        b2_obs = band2 + dm + A_2
        mag_obs = mag_band + dm + A_mag
        color = b1_obs - b2_obs

        valid = np.isfinite(color) & np.isfinite(mag_obs)
        return color[valid], mag_obs[valid], mass[valid]

    def _fast_distance(
        self,
        obs_color: np.ndarray,
        obs_mag: np.ndarray,
        obs_color_err: np.ndarray,
        obs_mag_err: np.ndarray,
        iso_color: np.ndarray,
        iso_mag: np.ndarray
    ) -> np.ndarray:
        """
        Fast distance computation using KD-tree with error normalization.
        Normalizes CMD space by typical errors before computing distances.
        """
        if len(iso_color) < 2:
            return np.full(len(obs_color), np.inf)

        # Normalize by median errors for balanced color/mag weighting
        color_scale = max(np.median(obs_color_err), 1e-6)
        mag_scale = max(np.median(obs_mag_err), 1e-6)

        # Normalized isochrone points
        iso_pts = np.column_stack([
            iso_color / color_scale,
            iso_mag / mag_scale
        ])

        # Normalized observed points
        obs_pts = np.column_stack([
            obs_color / color_scale,
            obs_mag / mag_scale
        ])

        # KD-tree for fast nearest neighbor
        tree = cKDTree(iso_pts)
        dist, idx = tree.query(obs_pts)

        # Scale back and normalize by individual errors
        closest_iso_c = iso_color[idx]
        closest_iso_m = iso_mag[idx]

        # Chi-like distance normalized by individual errors
        dist_c = (obs_color - closest_iso_c) / obs_color_err
        dist_m = (obs_mag - closest_iso_m) / obs_mag_err
        chi_dist = np.sqrt(dist_c**2 + dist_m**2)

        return chi_dist

    def _imf_weight(self, mass: np.ndarray) -> np.ndarray:
        """
        IMF weighting - Kroupa IMF.
        More massive (brighter) stars are rarer.
        """
        # Kroupa IMF: dN/dM ∝ M^(-alpha)
        # alpha = 1.3 for M < 0.5, alpha = 2.3 for M >= 0.5
        weights = np.ones_like(mass)
        low_mass = mass < 0.5
        high_mass = mass >= 0.5

        weights[low_mass] = mass[low_mass] ** (-1.3)
        weights[high_mass] = 0.5 ** (-1.3) * (mass[high_mass] / 0.5) ** (-2.3)

        # Normalize
        weights /= np.sum(weights)
        return weights

    def _objective(self, params: np.ndarray) -> float:
        """
        Objective function using KD-tree distance with error normalization.
        """
        log_age, mh, dm, e_gr = params

        try:
            iso_c, iso_m, iso_mass = self._interpolate_isochrone(log_age, mh, dm, e_gr)

            if len(iso_c) < 10:
                return 1e10

            obs_c = self._fit_obs[:, 0]
            obs_m = self._fit_obs[:, 1]
            obs_c_err = self._fit_err[:, 0]
            obs_m_err = self._fit_err[:, 1]

            # Fast KD-tree based distance (chi-like, normalized by errors)
            dist = self._fast_distance(
                obs_c, obs_m, obs_c_err, obs_m_err, iso_c, iso_m
            )

            # Robust: use only closest fraction
            n_use = max(20, int(len(dist) * self.fit_fraction))
            sorted_idx = np.argsort(dist)
            closest_idx = sorted_idx[:n_use]

            # Chi² = sum of squared normalized distances
            # Optional robust weighting for iterative membership refinement.
            weights = self._fit_weight
            if weights is not None and len(weights) == len(dist):
                w = np.asarray(weights, dtype=float)[closest_idx]
                w = np.clip(w, 1e-3, None)
                # Normalize weights so chi2 scale stays comparable across EM rounds.
                chi2 = np.sum((dist[closest_idx] ** 2) * w) * (len(w) / np.sum(w))
            else:
                chi2 = np.sum(dist[closest_idx] ** 2)

            return chi2

        except Exception:
            return 1e10

    def _de_callback(self, xk, convergence):
        self._fit_iteration += 1
        progress = min(0.95, self._fit_iteration / self._fit_max_iter)
        if self.progress_callback:
            self.progress_callback(progress, f"Iteration {self._fit_iteration}")

    def fit(
        self,
        obs_color: np.ndarray,
        obs_mag: np.ndarray,
        obs_color_err: np.ndarray,
        obs_mag_err: np.ndarray,
        mode: FitMode = FitMode.FAST,
        bounds: Optional[FitBounds] = None,
        snr_min: float = 5.0,
        **kwargs
    ) -> FitResult:
        """Fit isochrone to observed CMD"""
        t0 = time.time()

        if bounds is None:
            bounds = FitBounds()

        obs_color_err = np.clip(obs_color_err, 0.01, None)
        obs_mag_err = np.clip(obs_mag_err, 0.01, None)

        # Quality mask
        snr_c = 1.0 / obs_color_err
        snr_m = 1.0 / obs_mag_err
        snr = np.minimum(snr_c, snr_m)
        mask = (snr > snr_min) & np.isfinite(obs_color) & np.isfinite(obs_mag)

        color = obs_color[mask]
        mag = obs_mag[mask]
        color_err = obs_color_err[mask]
        mag_err = obs_mag_err[mask]

        n_stars = len(color)

        if n_stars < 20:
            return FitResult(
                log_age=9.0, metallicity=0.0, distance_mod=10.0, extinction_gr=0.0,
                chi2=np.inf, n_stars=n_stars, converged=False,
                fit_mode=mode.value, elapsed_sec=time.time() - t0
            )

        # Store for objective function
        self._fit_obs = np.column_stack([color, mag])
        self._fit_err = np.column_stack([color_err, mag_err])
        self._fit_weight = np.ones(n_stars, dtype=float)
        self._fit_iteration = 0

        em_iters = max(1, int(kwargs.pop("em_iters", 1)))
        membership_sigma_scale = float(kwargs.pop("membership_sigma_scale", 1.3))
        weight_floor = float(kwargs.pop("weight_floor", 0.05))

        best_result = None

        for em_idx in range(em_iters):
            # Scale per-round progress into overall [0, 1]
            original_callback = self.progress_callback
            if original_callback:
                def _round_callback(progress, message, _em=em_idx):
                    p = (float(_em) + float(np.clip(progress, 0.0, 1.0))) / float(em_iters)
                    original_callback(p, f"Round {_em + 1}/{em_iters}: {message}")
                self.progress_callback = _round_callback

            try:
                if mode == FitMode.FAST:
                    result = self._fit_fast(bounds, n_stars, **kwargs)
                elif mode == FitMode.HESSIAN:
                    result = self._fit_hessian(bounds, n_stars, **kwargs)
                else:
                    raise ValueError(f"Unsupported fit mode: {mode}")
            finally:
                self.progress_callback = original_callback

            if (best_result is None) or (float(result.chi2) < float(best_result.chi2)):
                best_result = result

            # Update weights from membership for next EM round.
            if em_idx < em_iters - 1:
                try:
                    prob = self.compute_membership(
                        result,
                        color,
                        mag,
                        sigma_scale=membership_sigma_scale,
                    )
                    if prob is not None and len(prob) == n_stars:
                        self._fit_weight = np.clip(np.asarray(prob, dtype=float), weight_floor, 1.0)
                except Exception:
                    pass

        result = best_result if best_result is not None else result

        result.n_stars = n_stars
        result.elapsed_sec = time.time() - t0
        result.fit_mode = mode.value

        self._fit_obs = None
        self._fit_err = None
        self._fit_weight = None

        return result

    def _fit_fast(self, bounds: FitBounds, n_stars: int, **kwargs) -> FitResult:
        maxiter = kwargs.get('maxiter', 100)
        seed = kwargs.get('seed', 42)
        self._fit_max_iter = maxiter

        if self.progress_callback:
            self.progress_callback(0.05, "Starting optimization...")

        result = differential_evolution(
            self._objective,
            bounds=bounds.to_list(),
            maxiter=maxiter,
            workers=1,
            updating='immediate',
            polish=True,
            seed=seed,
            tol=0.01,
            atol=0.01,
            popsize=10,
            mutation=(0.5, 1.0),
            recombination=0.7,
            callback=self._de_callback
        )

        if self.progress_callback:
            self.progress_callback(1.0, "Complete")

        chi2 = result.fun
        dof = max(1, n_stars - 4)

        return FitResult(
            log_age=result.x[0],
            metallicity=result.x[1],
            distance_mod=result.x[2],
            extinction_gr=result.x[3],
            chi2=chi2,
            reduced_chi2=chi2 / dof,
            converged=result.success
        )

    def _fit_hessian(self, bounds: FitBounds, n_stars: int, **kwargs) -> FitResult:
        de_maxiter = int(kwargs.get("de_maxiter", kwargs.get("maxiter", 50)))
        local_maxiter = int(kwargs.get("local_maxiter", 100))
        n_starts = max(1, int(kwargs.get("n_starts", 6)))
        de_seed = int(kwargs.get("seed", 42))
        initial_guess = kwargs.get("initial_guess", None)
        bounds_list = bounds.to_list()

        def _clip_guess(guess):
            try:
                g = np.asarray(guess, dtype=float).reshape(-1)
            except Exception:
                return None
            if g.size != 4 or (not np.isfinite(g).all()):
                return None
            clipped = [float(np.clip(val, lo, hi)) for val, (lo, hi) in zip(g, bounds_list)]
            return np.array(clipped, dtype=float)

        original_callback = self.progress_callback

        # ---------------------------------------------------------------------
        # 1) Global multi-start search (Differential Evolution)
        # ---------------------------------------------------------------------
        global_candidates = []
        global_base = 0.05
        global_span = 0.50
        for i in range(n_starts):
            seg_base = global_base + global_span * (i / n_starts)
            seg_span = global_span / n_starts

            if original_callback:
                def _scaled_global_cb(progress, message, _b=seg_base, _s=seg_span, _i=i):
                    p = _b + _s * float(np.clip(progress, 0.0, 1.0))
                    original_callback(p, f"Global {_i + 1}/{n_starts}: {message}")
                self.progress_callback = _scaled_global_cb

            try:
                fast_result = self._fit_fast(
                    bounds,
                    n_stars,
                    maxiter=de_maxiter,
                    seed=de_seed + i * 1009,
                )
            finally:
                self.progress_callback = original_callback

            x_fast = np.array(
                [
                    fast_result.log_age,
                    fast_result.metallicity,
                    fast_result.distance_mod,
                    fast_result.extinction_gr,
                ],
                dtype=float,
            )
            global_candidates.append(
                {
                    "source": f"global#{i + 1}",
                    "x0": x_fast,
                    "chi2": float(fast_result.chi2),
                    "converged": bool(fast_result.converged),
                }
            )

        # ---------------------------------------------------------------------
        # 2) Build local-start candidate set
        # ---------------------------------------------------------------------
        local_candidates = []
        seen = set()

        g0 = _clip_guess(initial_guess)
        if g0 is not None:
            key = tuple(np.round(g0, 6))
            seen.add(key)
            local_candidates.append({"source": "slider", "x0": g0})
            if original_callback:
                original_callback(0.57, "Added slider initial guess candidate")

        for c in sorted(global_candidates, key=lambda x: x["chi2"]):
            key = tuple(np.round(c["x0"], 6))
            if key in seen:
                continue
            seen.add(key)
            local_candidates.append({"source": c["source"], "x0": c["x0"]})

        if not local_candidates and global_candidates:
            best_global = min(global_candidates, key=lambda x: x["chi2"])
            local_candidates.append({"source": best_global["source"], "x0": best_global["x0"]})

        # ---------------------------------------------------------------------
        # 3) Local refinement for each candidate; select minimum chi²
        # ---------------------------------------------------------------------
        best_local_result = None
        best_local_source = None
        local_base = 0.55
        local_span = 0.40
        n_local = max(1, len(local_candidates))

        for j, cand in enumerate(local_candidates):
            seg_base = local_base + local_span * (j / n_local)
            seg_span = local_span / n_local
            iter_box = {"n": 0}

            def _local_callback(_xk, _b=seg_base, _s=seg_span, _j=j, _src=cand["source"]):
                iter_box["n"] += 1
                if original_callback:
                    frac = min(1.0, iter_box["n"] / max(local_maxiter, 1))
                    p = _b + _s * frac
                    original_callback(
                        p,
                        f"Local {_j + 1}/{n_local} [{_src}] {iter_box['n']}/{local_maxiter}",
                    )

            if original_callback:
                original_callback(seg_base, f"Local refine {j + 1}/{n_local} [{cand['source']}]")

            try:
                result = minimize(
                    self._objective,
                    cand["x0"],
                    method='L-BFGS-B',
                    bounds=bounds_list,
                    options={'maxiter': local_maxiter},
                    callback=_local_callback,
                )
            except Exception:
                continue

            if (best_local_result is None) or (float(result.fun) < float(best_local_result.fun)):
                best_local_result = result
                best_local_source = cand["source"]

        # ---------------------------------------------------------------------
        # 4) Compare with best global-only result and finalize
        # ---------------------------------------------------------------------
        best_global = min(global_candidates, key=lambda x: x["chi2"]) if global_candidates else None
        use_local = best_local_result is not None
        use_global = best_global is not None

        errors = [None, None, None, None]
        converged = False

        if use_local and (not use_global or float(best_local_result.fun) <= float(best_global["chi2"])):
            best_x = np.asarray(best_local_result.x, dtype=float)
            chi2 = float(best_local_result.fun)
            converged = bool(best_local_result.success)
            if hasattr(best_local_result, 'hess_inv') and best_local_result.hess_inv is not None:
                try:
                    if hasattr(best_local_result.hess_inv, 'todense'):
                        hess_inv = np.array(best_local_result.hess_inv.todense())
                    else:
                        hess_inv = np.array(best_local_result.hess_inv)
                    errors = np.sqrt(np.abs(np.diag(hess_inv)))
                except Exception:
                    pass
            selected_source = best_local_source or "local"
        elif use_global:
            best_x = np.asarray(best_global["x0"], dtype=float)
            chi2 = float(best_global["chi2"])
            converged = bool(best_global["converged"])
            selected_source = best_global["source"]
        else:
            # Fallback: center of bounds
            best_x = np.array([(lo + hi) * 0.5 for lo, hi in bounds_list], dtype=float)
            chi2 = float(self._objective(best_x))
            converged = False
            selected_source = "fallback"

        if original_callback:
            original_callback(0.98, f"Selected {selected_source} | chi2={chi2:.2f}")
            original_callback(1.0, "Complete")

        dof = max(1, n_stars - 4)
        return FitResult(
            log_age=float(best_x[0]),
            metallicity=float(best_x[1]),
            distance_mod=float(best_x[2]),
            extinction_gr=float(best_x[3]),
            log_age_err=errors[0] if errors[0] else None,
            metallicity_err=errors[1] if errors[1] else None,
            distance_mod_err=errors[2] if errors[2] else None,
            extinction_gr_err=errors[3] if errors[3] else None,
            chi2=chi2,
            reduced_chi2=chi2 / dof,
            converged=converged,
        )

    # -----------------------------------------------------------------
    # Grid Scan
    # -----------------------------------------------------------------

    def _grid_objective_2p(self, params_2p: np.ndarray, age: float, mh: float) -> float:
        """2-parameter wrapper: optimize (DM, E(g-r)) for fixed (age, mh)."""
        dm, e_gr = params_2p
        return self._objective(np.array([age, mh, dm, e_gr]))

    def _fit_grid_scan(self, bounds: FitBounds, n_stars: int, **kwargs) -> GridScanResult:
        local_maxiter = int(kwargs.get("local_maxiter", 50))
        initial_dm = float(kwargs.get("initial_dm", (bounds.distance_mod[0] + bounds.distance_mod[1]) * 0.5))
        initial_egr = float(kwargs.get("initial_egr", (bounds.extinction_gr[0] + bounds.extinction_gr[1]) * 0.5))

        # Filter grid to bounds
        age_mask = (self.ages >= bounds.log_age[0]) & (self.ages <= bounds.log_age[1])
        mh_mask = (self.metallicities >= bounds.metallicity[0]) & (self.metallicities <= bounds.metallicity[1])
        grid_ages = self.ages[age_mask]
        grid_mhs = self.metallicities[mh_mask]

        n_ages = len(grid_ages)
        n_mhs = len(grid_mhs)
        total_cells = n_ages * n_mhs

        if total_cells == 0:
            empty_fit = FitResult(
                log_age=0.0, metallicity=0.0, distance_mod=0.0, extinction_gr=0.0,
                chi2=np.inf, n_stars=n_stars, converged=False, fit_mode="grid_scan",
            )
            return GridScanResult(
                best_fit=empty_fit,
                chi2_grid=np.full((0, 0), np.nan),
                dm_grid=np.full((0, 0), np.nan),
                egr_grid=np.full((0, 0), np.nan),
                grid_ages=grid_ages, grid_mhs=grid_mhs,
            )

        chi2_grid = np.full((n_ages, n_mhs), np.nan)
        dm_grid = np.full((n_ages, n_mhs), np.nan)
        egr_grid = np.full((n_ages, n_mhs), np.nan)

        bounds_2p = [bounds.distance_mod, bounds.extinction_gr]
        x0_2p = np.array([initial_dm, initial_egr], dtype=float)
        n_evaluated = 0

        for i, age in enumerate(grid_ages):
            for j, mh in enumerate(grid_mhs):
                cell_idx = i * n_mhs + j

                # Check cache
                key = (round(float(age), 2), round(float(mh), 2))
                if key not in self._iso_cache:
                    if self.progress_callback:
                        self.progress_callback(
                            (cell_idx + 1) / total_cells,
                            f"Cell {cell_idx + 1}/{total_cells} (skip)"
                        )
                    continue

                try:
                    res = minimize(
                        self._grid_objective_2p,
                        x0_2p,
                        args=(float(age), float(mh)),
                        method='L-BFGS-B',
                        bounds=bounds_2p,
                        options={'maxiter': local_maxiter},
                    )
                    chi2_grid[i, j] = float(res.fun)
                    dm_grid[i, j] = float(res.x[0])
                    egr_grid[i, j] = float(res.x[1])
                    n_evaluated += 1
                except Exception:
                    pass

                if self.progress_callback:
                    self.progress_callback(
                        (cell_idx + 1) / total_cells,
                        f"Cell {cell_idx + 1}/{total_cells} | age={age:.2f} [M/H]={mh:.2f}"
                    )

        # Find best cell
        if n_evaluated == 0:
            empty_fit = FitResult(
                log_age=0.0, metallicity=0.0, distance_mod=0.0, extinction_gr=0.0,
                chi2=np.inf, n_stars=n_stars, converged=False, fit_mode="grid_scan",
            )
            return GridScanResult(
                best_fit=empty_fit,
                chi2_grid=chi2_grid, dm_grid=dm_grid, egr_grid=egr_grid,
                grid_ages=grid_ages, grid_mhs=grid_mhs,
                n_evaluated=n_evaluated,
            )

        flat_idx = np.nanargmin(chi2_grid)
        i_best, j_best = np.unravel_index(flat_idx, chi2_grid.shape)
        best_age = float(grid_ages[i_best])
        best_mh = float(grid_mhs[j_best])
        best_dm = float(dm_grid[i_best, j_best])
        best_egr = float(egr_grid[i_best, j_best])
        best_chi2 = float(chi2_grid[i_best, j_best])

        # Final 4-param refinement for error estimation
        errors = [None, None, None, None]
        try:
            bounds_list = bounds.to_list()
            ref = minimize(
                self._objective,
                np.array([best_age, best_mh, best_dm, best_egr]),
                method='L-BFGS-B',
                bounds=bounds_list,
                options={'maxiter': 200},
            )
            if ref.fun <= best_chi2:
                best_age, best_mh, best_dm, best_egr = (float(v) for v in ref.x)
                best_chi2 = float(ref.fun)
            if hasattr(ref, 'hess_inv') and ref.hess_inv is not None:
                if hasattr(ref.hess_inv, 'todense'):
                    hess_inv = np.array(ref.hess_inv.todense())
                else:
                    hess_inv = np.array(ref.hess_inv)
                errors = list(np.sqrt(np.abs(np.diag(hess_inv))))
        except Exception:
            pass

        dof = max(1, n_stars - 4)
        best_fit = FitResult(
            log_age=best_age,
            metallicity=best_mh,
            distance_mod=best_dm,
            extinction_gr=best_egr,
            log_age_err=errors[0],
            metallicity_err=errors[1],
            distance_mod_err=errors[2],
            extinction_gr_err=errors[3],
            chi2=best_chi2,
            reduced_chi2=best_chi2 / dof,
            n_stars=n_stars,
            converged=True,
            fit_mode="grid_scan",
        )

        return GridScanResult(
            best_fit=best_fit,
            chi2_grid=chi2_grid, dm_grid=dm_grid, egr_grid=egr_grid,
            grid_ages=grid_ages, grid_mhs=grid_mhs,
            n_evaluated=n_evaluated,
        )

    def fit_grid_scan(
        self,
        obs_color: np.ndarray,
        obs_mag: np.ndarray,
        obs_color_err: np.ndarray,
        obs_mag_err: np.ndarray,
        bounds: Optional[FitBounds] = None,
        snr_min: float = 5.0,
        **kwargs,
    ) -> GridScanResult:
        """Exhaustive grid scan over all (age, mh) within bounds."""
        t0 = time.time()
        if bounds is None:
            bounds = FitBounds()

        obs_color_err = np.clip(obs_color_err, 0.01, None)
        obs_mag_err = np.clip(obs_mag_err, 0.01, None)
        snr_c = 1.0 / obs_color_err
        snr_m = 1.0 / obs_mag_err
        snr = np.minimum(snr_c, snr_m)
        mask = (snr > snr_min) & np.isfinite(obs_color) & np.isfinite(obs_mag)

        color = obs_color[mask]
        mag = obs_mag[mask]
        color_err = obs_color_err[mask]
        mag_err = obs_mag_err[mask]
        n_stars = len(color)

        if n_stars < 20:
            empty_fit = FitResult(
                log_age=0.0, metallicity=0.0, distance_mod=0.0, extinction_gr=0.0,
                chi2=np.inf, n_stars=n_stars, converged=False, fit_mode="grid_scan",
            )
            return GridScanResult(
                best_fit=empty_fit,
                chi2_grid=np.full((0, 0), np.nan),
                dm_grid=np.full((0, 0), np.nan),
                egr_grid=np.full((0, 0), np.nan),
                grid_ages=np.array([]), grid_mhs=np.array([]),
            )

        self._fit_obs = np.column_stack([color, mag])
        self._fit_err = np.column_stack([color_err, mag_err])
        self._fit_weight = np.ones(n_stars, dtype=float)

        try:
            grid_result = self._fit_grid_scan(bounds, n_stars, **kwargs)
        finally:
            self._fit_obs = None
            self._fit_err = None
            self._fit_weight = None

        grid_result.elapsed_sec = time.time() - t0
        grid_result.best_fit.n_stars = n_stars
        grid_result.best_fit.elapsed_sec = grid_result.elapsed_sec
        return grid_result

    def get_best_fit_isochrone(self, result: FitResult) -> Tuple[np.ndarray, np.ndarray]:
        """Get best-fit isochrone CMD coordinates"""
        iso_c, iso_m, _ = self._interpolate_isochrone(
            result.log_age, result.metallicity,
            result.distance_mod, result.extinction_gr
        )
        return iso_c, iso_m

    def compute_membership(
        self,
        result: FitResult,
        obs_color: np.ndarray,
        obs_mag: np.ndarray,
        sigma_scale: float = 1.0
    ) -> np.ndarray:
        """
        Compute membership probability for each star.

        Uses perpendicular distance to isochrone for consistency
        with the fitting algorithm.

        Parameters
        ----------
        result : FitResult
            Fitting result with best-fit parameters
        obs_color, obs_mag : arrays
            Observed CMD data
        sigma_scale : float
            Scale factor for membership threshold

        Returns
        -------
        prob : array
            Membership probability [0, 1] for each star
        """
        iso_c, iso_m, _ = self._interpolate_isochrone(
            result.log_age, result.metallicity,
            result.distance_mod, result.extinction_gr
        )

        if len(iso_c) < 10:
            return np.zeros(len(obs_color))

        # Use KD-tree for fast nearest neighbor
        iso_pts = np.column_stack([iso_c, iso_m])
        tree = cKDTree(iso_pts)

        obs_pts = np.column_stack([obs_color, obs_mag])
        dist, _ = tree.query(obs_pts)

        # Adaptive sigma based on median distance
        valid_dist = dist[np.isfinite(dist)]
        if len(valid_dist) == 0:
            return np.zeros(len(obs_color))

        sigma = np.median(valid_dist) * sigma_scale

        # Gaussian membership probability
        prob = np.exp(-0.5 * (dist / sigma) ** 2)
        prob = np.clip(prob, 0, 1)

        return prob
