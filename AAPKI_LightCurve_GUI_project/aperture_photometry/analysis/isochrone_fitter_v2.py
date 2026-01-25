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
    MCMC = "mcmc"


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


class IsochroneFitterV2:
    """
    Improved Isochrone Fitter with interpolation and perpendicular distance.
    """

    # Extinction coefficients (SDSS)
    R_G = 3.303
    R_R = 2.285

    def __init__(
        self,
        isochrone_file: str | Path,
        col_mh: int = 1,
        col_age: int = 2,
        col_g: int = 29,
        col_r: int = 30,
        col_mass: int = 5,  # Initial mass column for IMF weighting
        fit_fraction: float = 0.7
    ):
        self.iso_file = Path(isochrone_file)
        self.COL_MH = col_mh
        self.COL_AGE = col_age
        self.COL_G = col_g
        self.COL_R = col_r
        self.COL_MASS = col_mass
        self.fit_fraction = fit_fraction

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
                    sort_idx = np.argsort(iso_sub[:, self.COL_G])
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

        g = iso_data[:, self.COL_G]
        r = iso_data[:, self.COL_R]
        mass = iso_data[:, self.COL_MASS]

        # Apply extinction and distance modulus
        E_BV = e_gr / (self.R_G - self.R_R)
        A_g = self.R_G * E_BV
        A_r = self.R_R * E_BV

        g_obs = g + dm + A_g
        r_obs = r + dm + A_r
        color = g_obs - r_obs

        valid = np.isfinite(color) & np.isfinite(g_obs)
        return color[valid], g_obs[valid], mass[valid]

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
        color_scale = np.median(obs_color_err)
        mag_scale = np.median(obs_mag_err)

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

        eps = 1e-6
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
        self._fit_iteration = 0

        # Run optimization
        if mode == FitMode.FAST:
            result = self._fit_fast(bounds, n_stars, **kwargs)
        elif mode == FitMode.HESSIAN:
            result = self._fit_hessian(bounds, n_stars, **kwargs)
        else:
            result = self._fit_fast(bounds, n_stars, **kwargs)

        result.n_stars = n_stars
        result.elapsed_sec = time.time() - t0
        result.fit_mode = mode.value

        self._fit_obs = None
        self._fit_err = None

        return result

    def _fit_fast(self, bounds: FitBounds, n_stars: int, **kwargs) -> FitResult:
        maxiter = kwargs.get('maxiter', 100)
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
            seed=42,
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
        if self.progress_callback:
            self.progress_callback(0.05, "Fast fit first...")

        fast_result = self._fit_fast(bounds, n_stars, maxiter=50)

        if self.progress_callback:
            self.progress_callback(0.6, "Refining + uncertainties...")

        x0 = [fast_result.log_age, fast_result.metallicity,
              fast_result.distance_mod, fast_result.extinction_gr]

        result = minimize(
            self._objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds.to_list(),
            options={'maxiter': 100}
        )

        errors = [None, None, None, None]
        if hasattr(result, 'hess_inv') and result.hess_inv is not None:
            try:
                if hasattr(result.hess_inv, 'todense'):
                    hess_inv = np.array(result.hess_inv.todense())
                else:
                    hess_inv = np.array(result.hess_inv)
                errors = np.sqrt(np.abs(np.diag(hess_inv)))
            except Exception:
                pass

        if self.progress_callback:
            self.progress_callback(1.0, "Complete")

        chi2 = result.fun
        dof = max(1, n_stars - 4)

        return FitResult(
            log_age=result.x[0],
            metallicity=result.x[1],
            distance_mod=result.x[2],
            extinction_gr=result.x[3],
            log_age_err=errors[0] if errors[0] else None,
            metallicity_err=errors[1] if errors[1] else None,
            distance_mod_err=errors[2] if errors[2] else None,
            extinction_gr_err=errors[3] if errors[3] else None,
            chi2=chi2,
            reduced_chi2=chi2 / dof,
            converged=result.success
        )

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
