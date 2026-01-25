"""
Fast Isochrone Fitting Module

Provides three fitting modes:
1. Fast (~1-3s): Differential Evolution
2. Hessian (~5s): Fast + uncertainty estimation
3. MCMC (~60s): Full posterior sampling

Uses normalized CMD space for proper distance calculation.

Author: KNUEMAO Pipeline
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Callable

import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize, differential_evolution
from scipy.interpolate import interp1d


class FitMode(Enum):
    """Fitting mode selection"""
    FAST = "fast"           # ~1-3 seconds
    HESSIAN = "hessian"     # ~5 seconds (with uncertainties)
    MCMC = "mcmc"           # ~60 seconds (full posterior)


@dataclass
class FitBounds:
    """Parameter bounds for fitting"""
    log_age: Tuple[float, float] = (8.0, 10.2)
    metallicity: Tuple[float, float] = (-0.5, 0.5)
    distance_mod: Tuple[float, float] = (8.0, 12.0)
    extinction_gr: Tuple[float, float] = (0.0, 0.5)

    def to_list(self):
        return [self.log_age, self.metallicity,
                self.distance_mod, self.extinction_gr]


@dataclass
class FitResult:
    """Isochrone fitting result"""
    # Best-fit parameters
    log_age: float
    metallicity: float
    distance_mod: float
    extinction_gr: float

    # Uncertainties (None if not computed)
    log_age_err: Optional[float] = None
    metallicity_err: Optional[float] = None
    distance_mod_err: Optional[float] = None
    extinction_gr_err: Optional[float] = None

    # Fit quality
    chi2: float = 0.0
    reduced_chi2: float = 0.0
    n_stars: int = 0
    n_params: int = 4

    # Metadata
    fit_mode: str = "fast"
    elapsed_sec: float = 0.0
    converged: bool = True

    # MCMC specific (optional)
    mcmc_chain: Optional[np.ndarray] = None
    mcmc_log_prob: Optional[np.ndarray] = None

    @property
    def age_gyr(self) -> float:
        """Age in Gyr"""
        return 10**(self.log_age - 9)

    @property
    def age_gyr_err(self) -> Optional[float]:
        """Age uncertainty in Gyr"""
        if self.log_age_err is None:
            return None
        # Error propagation: d(10^x) = 10^x * ln(10) * dx
        return self.age_gyr * np.log(10) * self.log_age_err

    @property
    def distance_pc(self) -> float:
        """Distance in parsecs"""
        return 10**(1 + self.distance_mod / 5)

    @property
    def distance_pc_err(self) -> Optional[float]:
        """Distance uncertainty in parsecs"""
        if self.distance_mod_err is None:
            return None
        return self.distance_pc * np.log(10) / 5 * self.distance_mod_err

    @property
    def A_V(self) -> float:
        """Visual extinction (assuming R_V = 3.1)"""
        # A_V ≈ 3.1 * E(B-V) ≈ 2.5 * E(g-r) approximately
        return 2.5 * self.extinction_gr

    def summary(self) -> str:
        """Return formatted summary string"""
        lines = [
            "=" * 50,
            f"Isochrone Fit Result ({self.fit_mode.upper()} mode)",
            "=" * 50,
            "",
            "Best-fit Parameters:",
        ]

        # Age
        if self.log_age_err:
            lines.append(f"  log(Age) = {self.log_age:.3f} +/- {self.log_age_err:.3f}")
            lines.append(f"           = {self.age_gyr:.2f} +/- {self.age_gyr_err:.2f} Gyr")
        else:
            lines.append(f"  log(Age) = {self.log_age:.3f}  ({self.age_gyr:.2f} Gyr)")

        # Metallicity
        if self.metallicity_err:
            lines.append(f"  [M/H]    = {self.metallicity:.3f} +/- {self.metallicity_err:.3f}")
        else:
            lines.append(f"  [M/H]    = {self.metallicity:.3f}")

        # Distance modulus
        if self.distance_mod_err:
            lines.append(f"  (m-M)    = {self.distance_mod:.3f} +/- {self.distance_mod_err:.3f}")
            lines.append(f"           = {self.distance_pc:.0f} +/- {self.distance_pc_err:.0f} pc")
        else:
            lines.append(f"  (m-M)    = {self.distance_mod:.3f}  ({self.distance_pc:.0f} pc)")

        # Extinction
        if self.extinction_gr_err:
            lines.append(f"  E(g-r)   = {self.extinction_gr:.4f} +/- {self.extinction_gr_err:.4f}")
        else:
            lines.append(f"  E(g-r)   = {self.extinction_gr:.4f}  (A_V ~ {self.A_V:.2f})")

        lines.extend([
            "",
            "Fit Quality:",
            f"  chi2     = {self.chi2:.1f}",
            f"  chi2/dof = {self.reduced_chi2:.3f}",
            f"  N_stars  = {self.n_stars}",
            "",
            f"Elapsed: {self.elapsed_sec:.2f} sec",
            "=" * 50,
        ])

        return "\n".join(lines)


class IsochroneFitter:
    """
    Fast Isochrone Fitter

    Supports three modes:
    - FAST: Quick fitting with Differential Evolution (~1-3s)
    - HESSIAN: Fast + Hessian-based uncertainties (~5s)
    - MCMC: Full Bayesian posterior sampling (~60s)

    Uses normalized CMD space for proper distance calculation.

    Usage:
        fitter = IsochroneFitter(isochrone_file)
        result = fitter.fit(color, mag, color_err, mag_err, mode=FitMode.FAST)
    """

    # Extinction coefficients (SDSS)
    R_G = 3.303     # A_g / E(B-V)
    R_R = 2.285     # A_r / E(B-V)

    def __init__(
        self,
        isochrone_file: str | Path,
        col_mh: int = 1,
        col_age: int = 2,
        col_g: int = 29,
        col_r: int = 30,
        fit_fraction: float = 0.6
    ):
        """
        Initialize fitter with isochrone data

        Parameters
        ----------
        isochrone_file : str or Path
            Path to isochrone data file
        col_mh : int
            Column index for metallicity (default: 1)
        col_age : int
            Column index for log(Age) (default: 2)
        col_g : int
            Column index for g-band magnitude (default: 29)
        col_r : int
            Column index for r-band magnitude (default: 30)
        fit_fraction : float
            Fraction of closest stars to use in robust fitting (default: 0.6)
        """
        self.iso_file = Path(isochrone_file)

        # Configurable column indices
        self.COL_MH = col_mh
        self.COL_AGE = col_age
        self.COL_G = col_g
        self.COL_R = col_r
        self.fit_fraction = fit_fraction

        self.iso_data = self._load_isochrone()

        # Extract unique age/metallicity values (rounded for proper matching)
        self.ages = np.unique(np.round(self.iso_data[:, self.COL_AGE], 2))
        self.metallicities = np.unique(np.round(self.iso_data[:, self.COL_MH], 2))

        # Callback for progress updates
        self.progress_callback: Optional[Callable[[float, str], None]] = None

        # Store current fitting data for objective function (avoid closure/pickle issues)
        self._fit_obs_pts: Optional[np.ndarray] = None
        self._fit_obs_pts_norm: Optional[np.ndarray] = None
        self._fit_weights: Optional[np.ndarray] = None
        self._fit_iteration: int = 0
        self._fit_max_iter: int = 50

        # Normalization parameters (set during fit)
        self._color_mean: float = 0.0
        self._color_std: float = 1.0
        self._mag_mean: float = 0.0
        self._mag_std: float = 1.0

    def _load_isochrone(self) -> np.ndarray:
        """Load isochrone data from file"""
        data = np.genfromtxt(self.iso_file, comments='#')
        # Remove rows with NaN
        data = data[~np.isnan(data).any(axis=1)]
        return data

    def _find_nearest_isochrone(self, log_age: float, mh: float) -> np.ndarray:
        """Find nearest isochrone in grid using rounded values"""
        # Round input values
        log_age_r = round(log_age, 1)  # Ages are in 0.1 steps
        mh_r = round(mh, 1)  # MH typically in 0.1 steps

        # Find nearest age
        age_idx = np.argmin(np.abs(self.ages - log_age_r))
        nearest_age = self.ages[age_idx]

        # Find nearest metallicity
        mh_idx = np.argmin(np.abs(self.metallicities - mh_r))
        nearest_mh = self.metallicities[mh_idx]

        # Filter data with larger tolerance
        mask = (np.abs(self.iso_data[:, self.COL_AGE] - nearest_age) < 0.06) & \
               (np.abs(self.iso_data[:, self.COL_MH] - nearest_mh) < 0.06)

        return self.iso_data[mask]

    def _get_isochrone_cmd(self, log_age: float, mh: float,
                           dm: float, e_gr: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get isochrone CMD with extinction and distance modulus applied

        Returns (color, magnitude) arrays
        """
        iso = self._find_nearest_isochrone(log_age, mh)

        if len(iso) < 10:
            return np.array([]), np.array([])

        # Get magnitudes (absolute)
        g = iso[:, self.COL_G].copy()
        r = iso[:, self.COL_R].copy()

        # Apply extinction: E(g-r) directly affects color
        # A_g = R_g * E(B-V), A_r = R_r * E(B-V)
        # E(g-r) = A_g - A_r = (R_g - R_r) * E(B-V)
        # So E(B-V) = E(g-r) / (R_g - R_r)
        E_BV = e_gr / (self.R_G - self.R_R)
        A_g = self.R_G * E_BV
        A_r = self.R_R * E_BV

        # Apply distance modulus and extinction
        g_obs = g + dm + A_g
        r_obs = r + dm + A_r

        color = g_obs - r_obs  # = (g-r)_intrinsic + E(g-r)
        mag = g_obs

        return color, mag

    def _normalize_point(self, color: np.ndarray, mag: np.ndarray) -> np.ndarray:
        """Normalize CMD points using stored statistics"""
        color_norm = (color - self._color_mean) / self._color_std
        mag_norm = (mag - self._mag_mean) / self._mag_std
        return np.column_stack([color_norm, mag_norm])

    def _objective(self, params: np.ndarray) -> float:
        """
        Robust objective function for optimization.
        Uses trimmed mean - only considers stars closest to isochrone.
        This handles field star contamination naturally.
        """
        log_age, mh, dm, e_gr = params

        try:
            iso_c, iso_m = self._get_isochrone_cmd(log_age, mh, dm, e_gr)

            if len(iso_c) < 10:
                return 1e10

            # Normalize isochrone points using same statistics as observed data
            iso_pts_norm = self._normalize_point(iso_c, iso_m)

            # Build KD-tree in normalized space
            tree = cKDTree(iso_pts_norm)

            # Compute distances in normalized space
            dist, _ = tree.query(self._fit_obs_pts_norm)

            # Robust: use only the closest N% of stars (trimmed mean)
            # This naturally excludes field stars and outliers
            n_use = max(20, int(len(dist) * self.fit_fraction))
            sorted_idx = np.argsort(dist)
            closest_idx = sorted_idx[:n_use]

            # Weighted sum of closest stars only
            chi2 = np.sum(dist[closest_idx]**2 * self._fit_weights[closest_idx])

            return chi2

        except Exception:
            return 1e10

    def _de_callback(self, xk, convergence):
        """Callback for differential_evolution progress updates"""
        self._fit_iteration += 1
        progress = min(0.95, self._fit_iteration / self._fit_max_iter)
        if self.progress_callback:
            self.progress_callback(progress, f"Iteration {self._fit_iteration}/{self._fit_max_iter}")

    def fit(self,
            obs_color: np.ndarray,
            obs_mag: np.ndarray,
            obs_color_err: np.ndarray,
            obs_mag_err: np.ndarray,
            mode: FitMode = FitMode.FAST,
            bounds: Optional[FitBounds] = None,
            snr_min: float = 5.0,
            **kwargs) -> FitResult:
        """
        Fit isochrone to observed CMD

        Parameters
        ----------
        obs_color : array
            Observed color (g-r)
        obs_mag : array
            Observed magnitude (g)
        obs_color_err : array
            Color measurement errors
        obs_mag_err : array
            Magnitude measurement errors
        mode : FitMode
            Fitting mode (FAST, HESSIAN, or MCMC)
        bounds : FitBounds
            Parameter bounds
        snr_min : float
            Minimum SNR for star selection (default: 5.0)
        **kwargs
            Additional arguments for specific modes

        Returns
        -------
        FitResult
            Fitting result with parameters and uncertainties
        """
        t0 = time.time()

        if bounds is None:
            bounds = FitBounds()

        # Add small epsilon to avoid divide by zero
        eps = 1e-6
        obs_color_err = np.clip(obs_color_err, eps, None)
        obs_mag_err = np.clip(obs_mag_err, eps, None)

        # Quality mask - less strict filtering
        total_err = np.sqrt(obs_color_err**2 + obs_mag_err**2)
        snr = 1.0 / (total_err + eps)
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

        # Compute normalization parameters from observed data
        self._color_mean = np.median(color)
        self._color_std = np.std(color) + eps
        self._mag_mean = np.median(mag)
        self._mag_std = np.std(mag) + eps

        # Normalize weights (scale to ~1 average)
        raw_weights = 1.0 / (color_err**2 + mag_err**2 + eps)
        weights = raw_weights / np.median(raw_weights)

        # Store in instance for objective function (avoid closure/pickle issues)
        self._fit_obs_pts = np.column_stack([color, mag])
        self._fit_obs_pts_norm = self._normalize_point(color, mag)
        self._fit_weights = weights
        self._fit_iteration = 0

        # Run fitting based on mode
        if mode == FitMode.FAST:
            result = self._fit_fast(bounds, n_stars, **kwargs)
        elif mode == FitMode.HESSIAN:
            result = self._fit_hessian(bounds, n_stars, **kwargs)
        elif mode == FitMode.MCMC:
            result = self._fit_mcmc(bounds, n_stars, color, mag, weights, **kwargs)
        else:
            raise ValueError(f"Unknown fit mode: {mode}")

        result.n_stars = n_stars
        result.elapsed_sec = time.time() - t0
        result.fit_mode = mode.value

        # Clear temp data
        self._fit_obs_pts = None
        self._fit_obs_pts_norm = None
        self._fit_weights = None

        return result

    def _fit_fast(self, bounds: FitBounds, n_stars: int, **kwargs) -> FitResult:
        """Fast fitting using Differential Evolution"""

        maxiter = kwargs.get('maxiter', 100)
        self._fit_max_iter = maxiter
        self._fit_iteration = 0

        if self.progress_callback:
            self.progress_callback(0.05, "Starting Differential Evolution...")

        result = differential_evolution(
            self._objective,
            bounds=bounds.to_list(),
            maxiter=maxiter,
            workers=1,  # Single thread to avoid pickle issues
            updating='deferred',  # More stable convergence
            polish=True,
            seed=42,
            tol=0.001,  # Tighter tolerance
            atol=0.001,
            popsize=20,  # Larger population for better exploration
            mutation=(0.5, 1.0),
            recombination=0.7,
            callback=self._de_callback
        )

        if self.progress_callback:
            self.progress_callback(1.0, "Fast fit complete")

        chi2 = result.fun
        dof = n_stars - 4

        return FitResult(
            log_age=result.x[0],
            metallicity=result.x[1],
            distance_mod=result.x[2],
            extinction_gr=result.x[3],
            chi2=chi2,
            reduced_chi2=chi2 / dof if dof > 0 else np.inf,
            converged=result.success
        )

    def _fit_hessian(self, bounds: FitBounds, n_stars: int, **kwargs) -> FitResult:
        """Fitting with Hessian-based uncertainty estimation"""

        if self.progress_callback:
            self.progress_callback(0.05, "Running fast fit first...")

        # First run fast fit
        fast_result = self._fit_fast(bounds, n_stars, maxiter=30)

        if self.progress_callback:
            self.progress_callback(0.6, "Estimating uncertainties...")

        # Refine with L-BFGS-B and get Hessian
        x0 = [fast_result.log_age, fast_result.metallicity,
              fast_result.distance_mod, fast_result.extinction_gr]

        result = minimize(
            self._objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds.to_list(),
            options={'maxiter': 100}
        )

        # Estimate uncertainties from inverse Hessian
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
            self.progress_callback(1.0, "Hessian fit complete")

        chi2 = result.fun
        dof = n_stars - 4

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
            reduced_chi2=chi2 / dof if dof > 0 else np.inf,
            converged=result.success
        )

    def _fit_mcmc(self, bounds: FitBounds, n_stars: int,
                  obs_color: np.ndarray, obs_mag: np.ndarray,
                  weights: np.ndarray, **kwargs) -> FitResult:
        """Full MCMC posterior sampling"""

        try:
            import emcee
        except ImportError:
            raise ImportError("MCMC mode requires 'emcee' package. "
                              "Install with: pip install emcee")

        nwalkers = kwargs.get('nwalkers', 16)
        nsteps = kwargs.get('nsteps', 500)
        burn_in = kwargs.get('burn_in', 100)

        if self.progress_callback:
            self.progress_callback(0.05, "Running fast fit for initial guess...")

        # Get initial guess from fast fit
        fast_result = self._fit_fast(bounds, n_stars, maxiter=30)

        # Define log probability (using instance method)
        bounds_list = bounds.to_list()

        def log_prior(theta):
            log_age, mh, dm, e_gr = theta
            if (bounds_list[0][0] < log_age < bounds_list[0][1] and
                bounds_list[1][0] < mh < bounds_list[1][1] and
                bounds_list[2][0] < dm < bounds_list[2][1] and
                bounds_list[3][0] < e_gr < bounds_list[3][1]):
                return 0.0
            return -np.inf

        def log_probability(theta):
            lp = log_prior(theta)
            if not np.isfinite(lp):
                return -np.inf
            chi2 = self._objective(theta)
            if not np.isfinite(chi2):
                return -np.inf
            return lp - 0.5 * chi2

        # Initialize walkers
        ndim = 4
        p0 = [fast_result.log_age, fast_result.metallicity,
              fast_result.distance_mod, fast_result.extinction_gr]

        # Small perturbations around best fit
        pos = np.array(p0) + 1e-3 * np.random.randn(nwalkers, ndim)

        # Ensure within bounds
        for i, (lo, hi) in enumerate(bounds_list):
            pos[:, i] = np.clip(pos[:, i], lo + 1e-6, hi - 1e-6)

        if self.progress_callback:
            self.progress_callback(0.15, "Running MCMC...")

        # Run MCMC
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)

        # Run with progress updates
        for i, _ in enumerate(sampler.sample(pos, iterations=nsteps, progress=False)):
            if self.progress_callback and i % 25 == 0:
                progress = 0.15 + 0.80 * (i / nsteps)
                self.progress_callback(progress, f"MCMC step {i}/{nsteps}")

        if self.progress_callback:
            self.progress_callback(0.95, "Analyzing chains...")

        # Get chains after burn-in
        chain = sampler.get_chain(discard=burn_in, flat=True)
        log_prob = sampler.get_log_prob(discard=burn_in, flat=True)

        # Best fit (maximum likelihood)
        best_idx = np.argmax(log_prob)
        best_params = chain[best_idx]

        # Uncertainties from percentiles
        percentiles = np.percentile(chain, [16, 50, 84], axis=0)
        medians = percentiles[1]
        errors_lo = medians - percentiles[0]
        errors_hi = percentiles[2] - medians
        errors = (errors_lo + errors_hi) / 2  # Symmetric approximation

        if self.progress_callback:
            self.progress_callback(1.0, "MCMC complete")

        chi2 = -2 * log_prob[best_idx]
        dof = n_stars - 4

        return FitResult(
            log_age=best_params[0],
            metallicity=best_params[1],
            distance_mod=best_params[2],
            extinction_gr=best_params[3],
            log_age_err=errors[0],
            metallicity_err=errors[1],
            distance_mod_err=errors[2],
            extinction_gr_err=errors[3],
            chi2=chi2,
            reduced_chi2=chi2 / dof if dof > 0 else np.inf,
            converged=True,
            mcmc_chain=chain,
            mcmc_log_prob=log_prob
        )

    def compute_membership(self, result: FitResult,
                           obs_color: np.ndarray,
                           obs_mag: np.ndarray,
                           sigma_scale: float = 1.0) -> np.ndarray:
        """
        Compute membership probability for each star

        Parameters
        ----------
        result : FitResult
            Fitting result
        obs_color, obs_mag : arrays
            Observed CMD
        sigma_scale : float
            Scale factor for membership threshold

        Returns
        -------
        prob : array
            Membership probability [0, 1] for each star
        """
        iso_c, iso_m = self._get_isochrone_cmd(
            result.log_age, result.metallicity,
            result.distance_mod, result.extinction_gr
        )

        if len(iso_c) < 10:
            return np.zeros(len(obs_color))

        iso_pts = np.column_stack([iso_c, iso_m])
        tree = cKDTree(iso_pts)

        obs_pts = np.column_stack([obs_color, obs_mag])
        dist, _ = tree.query(obs_pts)

        # Adaptive sigma based on median distance
        sigma = np.median(dist[np.isfinite(dist)]) * sigma_scale

        # Gaussian membership probability
        prob = np.exp(-0.5 * (dist / sigma)**2)
        prob = np.clip(prob, 0, 1)

        return prob

    def get_best_fit_isochrone(self, result: FitResult) -> Tuple[np.ndarray, np.ndarray]:
        """Get best-fit isochrone CMD coordinates"""
        return self._get_isochrone_cmd(
            result.log_age, result.metallicity,
            result.distance_mod, result.extinction_gr
        )
