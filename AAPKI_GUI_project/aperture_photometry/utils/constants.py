"""
Constants and configuration values for the aperture photometry pipeline.

This module centralizes magic numbers and configuration constants
that were previously scattered throughout the codebase.
"""

from dataclasses import dataclass
from typing import Dict, Tuple


# =============================================================================
# Photometry Constants
# =============================================================================

@dataclass(frozen=True)
class PhotometryConstants:
    """Constants for aperture photometry calculations."""

    # Magnitude error coefficient: 2.5 / ln(10) ≈ 1.0857
    MAG_ERR_COEFF: float = 1.0857

    # Minimum cutout size for centroid refinement (pixels)
    MIN_CUTOUT_SIZE: int = 9

    # Default centroid box scale (× FWHM)
    DEFAULT_CBOX_SCALE: float = 1.5

    # Minimum SNR for valid magnitude
    DEFAULT_MIN_SNR: float = 3.0

    # Default zero point (ADU/sec)
    DEFAULT_ZP: float = 25.0

    # Default saturation level (ADU)
    DEFAULT_SATURATION_ADU: float = 60000.0

    # Sigma clipping parameters
    DEFAULT_SIGMA_CLIP: float = 3.0
    DEFAULT_CLIP_MAXITER: int = 5

    # MAD to sigma conversion factor
    MAD_TO_SIGMA: float = 1.4826


PHOT = PhotometryConstants()


# =============================================================================
# Detection Constants
# =============================================================================

@dataclass(frozen=True)
class DetectionConstants:
    """Constants for source detection."""

    # FWHM bounds (pixels)
    DEFAULT_FWHM_MIN_PX: float = 3.5
    DEFAULT_FWHM_MAX_PX: float = 12.0

    # Gaussian FWHM to sigma conversion
    FWHM_TO_SIGMA: float = 2.35482  # 2 * sqrt(2 * ln(2))

    # Detection threshold (sigma above background)
    DEFAULT_DETECT_SIGMA: float = 3.2

    # Minimum connected pixels for source
    DEFAULT_NPIXELS: int = 5

    # Deblend parameters
    DEFAULT_DEBLEND_NLEVELS: int = 32
    DEFAULT_DEBLEND_CONTRAST: float = 0.001


DETECT = DetectionConstants()


# =============================================================================
# Aperture Parameters
# =============================================================================

@dataclass(frozen=True)
class ApertureConstants:
    """Constants for aperture configuration."""

    # Aperture scale (× FWHM)
    DEFAULT_APERTURE_SCALE: float = 1.0

    # Annulus inner radius scale (× FWHM)
    DEFAULT_ANNULUS_SCALE: float = 4.0

    # Annulus width scale (× FWHM)
    DEFAULT_DANNULUS_SCALE: float = 2.0

    # Minimum annulus gap (pixels)
    DEFAULT_MIN_GAP_PX: float = 6.0

    # Minimum annulus width (pixels)
    DEFAULT_MIN_WIDTH_PX: float = 12.0

    # Neighbor mask scale for crowding
    DEFAULT_NEIGHBOR_MASK_SCALE: float = 1.3

    # Maximum recenter shift (pixels)
    DEFAULT_MAX_RECENTER_SHIFT: float = 2.0

    # Centroid outlier threshold (pixels)
    DEFAULT_CENTROID_OUTLIER_PX: float = 1.0

    # Minimum sky pixels for local sigma
    DEFAULT_MIN_N_SKY: int = 50


APERTURE = ApertureConstants()


# =============================================================================
# Extinction Coefficients
# =============================================================================

@dataclass(frozen=True)
class ExtinctionCoefficients:
    """SDSS extinction coefficients (Schlafly & Finkbeiner 2011)."""

    # A_X / E(B-V) values
    R_U: float = 4.239
    R_G: float = 3.303
    R_R: float = 2.285
    R_I: float = 1.698
    R_Z: float = 1.263

    # Standard R_V
    DEFAULT_RV: float = 3.1


EXTINCTION = ExtinctionCoefficients()


# Atmospheric extinction coefficients (mag/airmass) - typical values
ATMOSPHERIC_EXTINCTION: Dict[str, float] = {
    'u': 0.50,
    'g': 0.20,
    'r': 0.12,
    'i': 0.08,
    'z': 0.05,
}


# =============================================================================
# Gaia to SDSS Transformations (Jordi et al. 2010)
# =============================================================================

# Polynomial coefficients for G → SDSS transformations
# Format: (a0, a1, a2, a3, ...) where result = sum(a_i * (BP-RP)^i)
GAIA_TO_SDSS_COEFFS: Dict[str, Tuple[float, ...]] = {
    'g': (0.2199, -0.6365, -0.1548, 0.0064),      # G → g
    'r': (-0.09837, 0.08592, 0.1907, -0.1701, 0.02263),  # G → r
    'i': (-0.293, 0.6404, -0.09609, -0.002104),   # G → i
}


# =============================================================================
# Isochrone Fitting
# =============================================================================

@dataclass(frozen=True)
class IsochroneConstants:
    """Constants for isochrone fitting."""

    # Default parameter bounds
    LOG_AGE_MIN: float = 8.0
    LOG_AGE_MAX: float = 10.2

    METALLICITY_MIN: float = -0.5
    METALLICITY_MAX: float = 0.5

    DISTANCE_MOD_MIN: float = 8.0
    DISTANCE_MOD_MAX: float = 12.0

    EXTINCTION_GR_MIN: float = 0.0
    EXTINCTION_GR_MAX: float = 0.5

    # Fitting parameters
    DEFAULT_FIT_FRACTION: float = 0.6  # Trimmed mean fraction
    DEFAULT_SNR_MIN: float = 5.0       # Minimum SNR for fitting


ISOCHRONE = IsochroneConstants()


# =============================================================================
# WCS / Astrometry
# =============================================================================

@dataclass(frozen=True)
class AstrometryConstants:
    """Constants for WCS and astrometric calculations."""

    # Arcsec per radian
    ARCSEC_PER_RAD: float = 206265.0

    # Pixel scale calculation coefficient
    PIXEL_SCALE_COEFF: float = 206.265

    # Default WCS solve timeout (seconds)
    DEFAULT_WCS_TIMEOUT: int = 120

    # ASTAP search radius (degrees)
    DEFAULT_SEARCH_RADIUS: float = 30.0

    # Gaia match radius (arcsec)
    DEFAULT_MATCH_RADIUS_ARCSEC: float = 2.0


ASTROMETRY = AstrometryConstants()


# =============================================================================
# File Patterns
# =============================================================================

# FITS file extensions
FITS_EXTENSIONS = ('.fits', '.fit', '.fts', '.FITS', '.FIT', '.FTS')

# Common FITS header keys for filter
FILTER_HEADER_KEYS = ('FILTER', 'FILT', 'FILTNAM', 'FILTER1')

# Common FITS header keys for exposure time
EXPTIME_HEADER_KEYS = ('EXPTIME', 'EXPOSURE', 'ITIME', 'ELAPTIME')

# Common FITS header keys for date/time
DATETIME_HEADER_KEYS = ('DATE-OBS', 'DATE', 'DATEOBS')
TIME_HEADER_KEYS = ('TIME-OBS', 'UTC', 'UT', 'TIME')


# =============================================================================
# GUI Constants
# =============================================================================

@dataclass(frozen=True)
class GUIConstants:
    """Constants for GUI elements."""

    # Default log window size
    LOG_WINDOW_WIDTH: int = 800
    LOG_WINDOW_HEIGHT: int = 400

    # Progress update interval (ms)
    PROGRESS_UPDATE_INTERVAL: int = 100

    # Table column widths
    DEFAULT_COL_WIDTH_NARROW: int = 80
    DEFAULT_COL_WIDTH_MEDIUM: int = 120
    DEFAULT_COL_WIDTH_WIDE: int = 200


GUI = GUIConstants()


# =============================================================================
# Parallel Processing Constants
# =============================================================================

@dataclass(frozen=True)
class ParallelConstants:
    """Constants for parallel/concurrent processing."""

    # Default number of workers (0 = auto, use CPU count)
    DEFAULT_MAX_WORKERS: int = 0

    # Maximum workers cap (safety limit)
    MAX_WORKERS_CAP: int = 16

    # Minimum workers
    MIN_WORKERS: int = 1

    # Default timeout per task (seconds)
    DEFAULT_TASK_TIMEOUT: float = 300.0

    # Batch size for progress updates
    PROGRESS_BATCH_SIZE: int = 1


PARALLEL = ParallelConstants()


def get_parallel_workers(params=None) -> int:
    """
    Get parallel worker count from params or auto-detect.

    Central function for all steps to use consistent parallel settings.

    Args:
        params: Parameter object with P attribute, or None for auto-detect

    Returns:
        Number of workers to use

    Usage:
        from aperture_photometry.utils.constants import get_parallel_workers
        max_workers = get_parallel_workers(self.params)
    """
    import os

    # Try to get from params
    if params is not None:
        try:
            val = int(getattr(params.P, "parallel_max_workers", 0))
            if val > 0:
                return min(val, PARALLEL.MAX_WORKERS_CAP)
        except (AttributeError, TypeError, ValueError):
            pass

    # Auto-detect: 75% of CPU cores, min 2, max cap
    cpu_count = os.cpu_count() or 4
    optimal = max(2, min(int(cpu_count * 0.75), PARALLEL.MAX_WORKERS_CAP))
    return optimal
