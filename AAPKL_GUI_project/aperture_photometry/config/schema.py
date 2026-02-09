"""
Pydantic-based parameter schema for aperture photometry pipeline.
Provides type-safe configuration with validation.
"""

from __future__ import annotations

import hashlib
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # Python 3.10 and earlier

import tomli_w
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


# =============================================================================
# Enums for constrained string choices
# =============================================================================

class ParallelMode(str, Enum):
    """Parallel processing mode"""
    THREAD = "thread"
    PROCESS = "process"
    AUTO = "auto"
    NONE = "none"


class DetectEngine(str, Enum):
    """Source detection engine"""
    SEGM = "segm"
    PEAK = "peak"
    DAO = "dao"


class BkgEdgeMethod(str, Enum):
    """Background 2D edge handling method"""
    PAD = "pad"
    CROP = "crop"
    WRAP = "wrap"
    EXTEND = "extend"


class BkgMethod(str, Enum):
    """Background estimation method"""
    MEDIAN = "median"
    MEAN = "mean"


class MagFluxMode(str, Enum):
    """Magnitude/flux display mode for 5X viewer"""
    RATE_E = "rate_e"
    FLUX_E = "flux_e"


class ApertureMode(str, Enum):
    """Aperture photometry mode"""
    APCORR = "apcorr"
    FIXED = "fixed"


class CacheStrategy(str, Enum):
    """Detection cache validation strategy"""
    MTIME = "mtime"
    HASH = "hash"
    NONE = "none"


# =============================================================================
# Sub-configuration models
# =============================================================================

class IOConfig(BaseModel):
    """I/O and directory configuration"""
    model_config = ConfigDict(validate_assignment=True)

    data_dir: str = Field(
        default=".",
        description="Input data directory containing FITS files"
    )
    filename_prefix: str = Field(
        default="pp_",
        description="Prefix pattern for matching FITS files"
    )
    result_dir: str = Field(
        default="",
        description="Output directory (empty = data_dir/result)"
    )
    cache_dir: str = Field(
        default="cache",
        description="Cache directory name"
    )


class TargetConfig(BaseModel):
    """Target object configuration"""
    model_config = ConfigDict(validate_assignment=True)

    name: str = Field(
        default="",
        description="Target name (e.g., M38)"
    )
    ra_deg: Optional[float] = Field(
        default=None,
        ge=0.0, le=360.0,
        description="Target RA in degrees (optional, for initial WCS)"
    )
    dec_deg: Optional[float] = Field(
        default=None,
        ge=-90.0, le=90.0,
        description="Target Dec in degrees (optional, for initial WCS)"
    )


class ParallelConfig(BaseModel):
    """Parallel processing configuration"""
    model_config = ConfigDict(validate_assignment=True)

    mode: ParallelMode = Field(
        default=ParallelMode.THREAD,
        description="Parallel processing mode"
    )
    max_workers: int = Field(
        default=0,
        ge=0,
        description="Number of workers (0 = auto)"
    )
    resume_mode: bool = Field(
        default=True,
        description="Skip completed steps on restart"
    )
    force_redetect: bool = Field(
        default=False,
        description="Force re-detection even if cached"
    )
    force_rephot: bool = Field(
        default=False,
        description="Force re-photometry even if cached"
    )
    detect_cache_strategy: CacheStrategy = Field(
        default=CacheStrategy.MTIME,
        description="Cache validation strategy"
    )


class UIConfig(BaseModel):
    """UI-related configuration"""
    model_config = ConfigDict(validate_assignment=True)

    log_tail: int = Field(
        default=300,
        ge=10, le=10000,
        description="Number of log lines to display"
    )
    detect_progress_bar: bool = Field(
        default=True,
        description="Show detection progress bar"
    )
    canvas_px: int = Field(
        default=900,
        ge=400, le=4000,
        description="UI canvas size in pixels"
    )


class InstrumentConfig(BaseModel):
    """Telescope/camera instrument configuration"""
    model_config = ConfigDict(validate_assignment=True)

    telescope_focal_mm: float = Field(
        default=3947.0,
        gt=0,
        description="Telescope focal length in mm"
    )
    camera_pixel_um: float = Field(
        default=3.76,
        gt=0,
        description="Camera pixel size in micrometers"
    )
    binning: int = Field(
        default=2,
        ge=1, le=8,
        description="Camera binning (1, 2, 4, etc.)"
    )
    gain_e_per_adu: float = Field(
        default=0.1,
        gt=0,
        description="Gain in electrons per ADU"
    )
    rdnoise_e: float = Field(
        ...,  # Required field
        gt=0,
        description="Read noise in electrons (REQUIRED)"
    )
    saturation_adu: float = Field(
        default=65000.0,
        gt=0, le=65535,
        description="Saturation level in ADU"
    )
    zp_initial: float = Field(
        default=25.0,
        ge=15.0, le=35.0,
        description="Initial zero point magnitude (ADU/sec)"
    )
    datamin_adu: Optional[float] = Field(
        default=None,
        description="Minimum valid ADU value (optional)"
    )
    datamax_adu: Optional[float] = Field(
        default=None,
        description="Maximum valid ADU value (optional)"
    )

    @property
    def pixel_scale_arcsec(self) -> float:
        """Calculate pixel scale in arcsec/pixel"""
        return 206.265 * self.camera_pixel_um * self.binning / self.telescope_focal_mm


class AlignmentConfig(BaseModel):
    """Frame alignment configuration"""
    model_config = ConfigDict(validate_assignment=True)

    ref_index: int = Field(
        default=0,
        ge=0,
        description="Reference frame index for alignment"
    )
    global_align: bool = Field(
        default=True,
        description="Enable global (cross-filter) alignment"
    )
    global_ref_filter: str = Field(
        default="r",
        description="Reference filter for global alignment (g, r, i)"
    )
    global_ref_index: int = Field(
        default=0,
        ge=0,
        description="Global reference frame index"
    )


class FWHMConfig(BaseModel):
    """FWHM/PSF configuration"""
    model_config = ConfigDict(validate_assignment=True)

    guess_arcsec: Optional[float] = Field(
        default=2.5,
        gt=0, le=20.0,
        description="FWHM seed in arcseconds"
    )
    guess_px: Optional[float] = Field(
        default=None,
        gt=0, le=50.0,
        description="FWHM seed in pixels (overrides arcsec if set)"
    )
    px_min: float = Field(
        default=3.0,
        gt=0, le=20.0,
        description="Minimum FWHM in pixels"
    )
    px_max: float = Field(
        default=12.0,
        gt=0, le=50.0,
        description="Maximum FWHM in pixels"
    )
    arcsec_min: Optional[float] = Field(
        default=None,
        gt=0,
        description="Minimum FWHM in arcseconds"
    )
    arcsec_max: Optional[float] = Field(
        default=None,
        gt=0,
        description="Maximum FWHM in arcseconds"
    )
    elong_max: float = Field(
        default=1.3,
        ge=1.0, le=3.0,
        description="Maximum elongation for isolated sources"
    )
    qc_max_sources: int = Field(
        default=40,
        ge=5, le=200,
        description="Max sources for FWHM QC check"
    )
    iso_min_sep_pix: float = Field(
        default=18.0,
        ge=3.0,
        description="Minimum separation for isolated sources (pixels)"
    )
    measure_max: int = Field(
        default=25,
        ge=5, le=100,
        description="Max radius for radial FWHM measurement"
    )
    dr: float = Field(
        default=0.5,
        gt=0, le=2.0,
        description="Radial step size for FWHM measurement"
    )

    @model_validator(mode='after')
    def validate_fwhm_range(self) -> 'FWHMConfig':
        if self.px_min >= self.px_max:
            raise ValueError("fwhm.px_min must be less than fwhm.px_max")
        return self


class DeblendConfig(BaseModel):
    """Deblending configuration"""
    model_config = ConfigDict(validate_assignment=True)

    enable: bool = Field(
        default=True,
        description="Enable source deblending"
    )
    nthresh: int = Field(
        default=64,
        ge=8, le=256,
        description="Number of threshold levels"
    )
    contrast: float = Field(
        default=0.004,
        gt=0, le=1.0,
        description="Deblend contrast threshold"
    )
    max_labels: int = Field(
        default=4000,
        ge=100,
        description="Maximum deblend labels"
    )
    label_hard_max: int = Field(
        default=7000,
        ge=100,
        description="Hard limit on deblend labels"
    )
    nlevels_soft: int = Field(
        default=32,
        ge=8, le=128,
        description="Soft deblend levels"
    )
    contrast_soft: float = Field(
        default=0.005,
        gt=0, le=1.0,
        description="Soft deblend contrast"
    )


class PeakConfig(BaseModel):
    """Peak detection (supplementary) configuration"""
    model_config = ConfigDict(validate_assignment=True)

    enable: bool = Field(
        default=True,
        description="Enable peak detection pass"
    )
    nsigma: float = Field(
        default=3.2,
        ge=1.0, le=10.0,
        description="Peak detection sigma threshold"
    )
    kernel_scales: List[float] = Field(
        default=[0.9, 1.3],
        description="Peak kernel scales"
    )
    min_sep_px: float = Field(
        default=4.0,
        ge=1.0,
        description="Minimum separation between peaks (pixels)"
    )
    max_add: int = Field(
        default=600,
        ge=0,
        description="Maximum sources to add from peak detection"
    )
    max_elong: float = Field(
        default=1.6,
        ge=1.0, le=5.0,
        description="Maximum elongation for peaks"
    )
    sharp_lo: float = Field(
        default=0.12,
        ge=0, le=1.0,
        description="Minimum sharpness for peaks"
    )
    skip_if_nsrc_ge: int = Field(
        default=4500,
        ge=100,
        description="Skip peak pass if sources >= this value"
    )


class DAOConfig(BaseModel):
    """DAO refinement configuration (optional)"""
    model_config = ConfigDict(validate_assignment=True)

    enable: bool = Field(
        default=False,
        description="Enable DAO refinement"
    )
    fwhm_px: float = Field(
        default=6.0,
        gt=0, le=30.0,
        description="DAO FWHM estimate (pixels)"
    )
    sharp_lo: float = Field(
        default=0.2,
        ge=0, le=1.0,
        description="DAO sharpness lower bound"
    )
    sharp_hi: float = Field(
        default=1.0,
        ge=0, le=2.0,
        description="DAO sharpness upper bound"
    )
    round_lo: float = Field(
        default=-0.5,
        ge=-2.0, le=0.0,
        description="DAO roundness lower bound"
    )
    round_hi: float = Field(
        default=0.5,
        ge=0.0, le=2.0,
        description="DAO roundness upper bound"
    )
    match_tol_px: float = Field(
        default=2.0,
        ge=0.5, le=10.0,
        description="DAO source matching tolerance (pixels)"
    )


class DetectionConfig(BaseModel):
    """Source detection configuration"""
    model_config = ConfigDict(validate_assignment=True)

    engine: DetectEngine = Field(
        default=DetectEngine.DAO,
        description="Detection engine (segm, peak, or dao)"
    )
    sigma: float = Field(
        default=3.2,
        ge=1.0, le=10.0,
        description="Base detection threshold (sigma)"
    )
    sigma_g: Optional[float] = Field(
        default=None,
        ge=1.0, le=10.0,
        description="Detection sigma for g-band"
    )
    sigma_r: Optional[float] = Field(
        default=None,
        ge=1.0, le=10.0,
        description="Detection sigma for r-band"
    )
    sigma_i: Optional[float] = Field(
        default=None,
        ge=1.0, le=10.0,
        description="Detection sigma for i-band"
    )
    minarea_pix: int = Field(
        default=3,
        ge=1, le=100,
        description="Minimum source area in pixels"
    )
    keep_max: int = Field(
        default=6000,
        ge=100,
        description="Maximum sources to keep per frame"
    )
    dilate_radius_px: int = Field(
        default=4,
        ge=0, le=20,
        description="Segmentation dilation radius (pixels)"
    )
    deblend: DeblendConfig = Field(default_factory=DeblendConfig)
    peak: PeakConfig = Field(default_factory=PeakConfig)
    dao: DAOConfig = Field(default_factory=DAOConfig)


class Background2DConfig(BaseModel):
    """2D background estimation configuration"""
    model_config = ConfigDict(validate_assignment=True)

    enable: bool = Field(
        default=True,
        description="Enable 2D background subtraction"
    )
    in_detect: bool = Field(
        default=True,
        description="Use 2D background in detection"
    )
    box: int = Field(
        default=64,
        ge=16, le=512,
        description="Background box size (pixels)"
    )
    filter_size: int = Field(
        default=3,
        ge=1, le=11,
        description="Background filter size"
    )
    edge_method: BkgEdgeMethod = Field(
        default=BkgEdgeMethod.PAD,
        description="Edge handling method"
    )
    method: BkgMethod = Field(
        default=BkgMethod.MEDIAN,
        description="Background estimation method"
    )
    downsample: int = Field(
        default=4,
        ge=1, le=16,
        description="Downsampling factor for optimization"
    )


class ClipConfig(BaseModel):
    """Data clipping configuration"""
    model_config = ConfigDict(validate_assignment=True)

    min_adu: Optional[float] = Field(
        default=None,
        description="Minimum ADU for clipping (optional)"
    )
    max_adu: Optional[float] = Field(
        default=None,
        description="Maximum ADU for clipping (optional)"
    )


class QCConfig(BaseModel):
    """Quality control configuration"""
    model_config = ConfigDict(validate_assignment=True)

    gate_enable: bool = Field(
        default=True,
        description="Enable QC gating"
    )
    sky_sigma_max_e: float = Field(
        default=25.0,
        gt=0,
        description="Maximum sky sigma in electrons"
    )
    nsrc_min: int = Field(
        default=0,
        ge=0,
        description="Minimum number of sources"
    )
    keep_positions_if_fail: bool = Field(
        default=True,
        description="Keep source positions even if QC fails"
    )


class ApertureScalesConfig(BaseModel):
    """Aperture scaling configuration"""
    model_config = ConfigDict(validate_assignment=True)

    aperture_scale: float = Field(
        default=1.0,
        gt=0, le=10.0,
        description="Aperture scale (in FWHM units)"
    )
    annulus_scale: float = Field(
        default=4.0,
        gt=0, le=20.0,
        description="Inner annulus scale (in FWHM units)"
    )
    dannulus_scale: float = Field(
        default=2.0,
        gt=0, le=10.0,
        description="Annulus width scale (in FWHM units)"
    )
    center_cbox_scale: float = Field(
        default=1.5,
        gt=0, le=5.0,
        description="Centering box scale (in FWHM units)"
    )
    annulus_min_gap_px: float = Field(
        default=6.0,
        ge=0,
        description="Minimum gap between aperture and annulus (pixels)"
    )
    annulus_min_width_px: float = Field(
        default=12.0,
        ge=1.0,
        description="Minimum annulus width (pixels)"
    )


class ApertureRadiiConfig(BaseModel):
    """Minimum aperture radii configuration"""
    model_config = ConfigDict(validate_assignment=True)

    min_r_ap_px: float = Field(
        default=4.0,
        ge=1.0,
        description="Minimum aperture radius (pixels)"
    )
    min_r_in_px: float = Field(
        default=12.0,
        ge=1.0,
        description="Minimum inner annulus radius (pixels)"
    )
    min_r_out_px: float = Field(
        default=20.0,
        ge=1.0,
        description="Minimum outer annulus radius (pixels)"
    )
    sigma_clip: float = Field(
        default=3.0,
        ge=1.0, le=10.0,
        description="Sigma clipping for sky estimation"
    )
    max_iter: int = Field(
        default=5,
        ge=1, le=20,
        description="Max iterations for sky fitting"
    )
    neighbor_mask_scale: float = Field(
        default=1.3,
        ge=1.0, le=5.0,
        description="Neighbor masking scale"
    )

    @model_validator(mode='after')
    def validate_radii_order(self) -> 'ApertureRadiiConfig':
        if self.min_r_ap_px >= self.min_r_in_px:
            raise ValueError("min_r_ap_px must be less than min_r_in_px")
        if self.min_r_in_px >= self.min_r_out_px:
            raise ValueError("min_r_in_px must be less than min_r_out_px")
        return self


class ApcorrConfig(BaseModel):
    """Aperture correction configuration"""
    model_config = ConfigDict(validate_assignment=True)

    apply: bool = Field(
        default=True,
        description="Apply aperture correction"
    )
    small_scale: float = Field(
        default=1.0,
        gt=0, le=5.0,
        description="Small aperture scale for correction"
    )
    large_scale: float = Field(
        default=3.0,
        gt=0, le=10.0,
        description="Large aperture scale for correction"
    )
    min_n: int = Field(
        default=20,
        ge=5,
        description="Minimum sources for correction calculation"
    )
    scatter_max: float = Field(
        default=0.05,
        gt=0, le=0.5,
        description="Maximum relative scatter for correction"
    )
    min_snr: float = Field(
        default=40.0,
        ge=5.0,
        description="Minimum SNR for aperture correction sources"
    )


class PhotConfig(BaseModel):
    """Photometry configuration"""
    model_config = ConfigDict(validate_assignment=True)

    mode: ApertureMode = Field(
        default=ApertureMode.APCORR,
        description="Aperture photometry mode"
    )
    recenter: bool = Field(
        default=True,
        description="Recenter apertures on sources"
    )
    use_segm_mask: bool = Field(
        default=True,
        description="Use segmentation mask for background"
    )
    min_snr_for_mag: float = Field(
        default=3.0,
        ge=1.0,
        description="Minimum SNR to calculate magnitude"
    )
    use_qc_pass_only: bool = Field(
        default=False,
        description="Only use QC-passed frames for photometry"
    )
    use_original_frames: bool = Field(
        default=True,
        description="Use original frames for photometry"
    )
    scales: ApertureScalesConfig = Field(default_factory=ApertureScalesConfig)
    radii: ApertureRadiiConfig = Field(default_factory=ApertureRadiiConfig)
    apcorr: ApcorrConfig = Field(default_factory=ApcorrConfig)


class WCSConfig(BaseModel):
    """WCS plate solving configuration"""
    model_config = ConfigDict(validate_assignment=True)

    do_plate_solve: bool = Field(
        default=True,
        description="Enable plate solving"
    )
    solver_priority: str = Field(
        default="astrometry",
        description="Primary solver: 'astrometry' or 'astap'. Other solver used as fallback."
    )
    astap_exe: str = Field(
        default="astap_cli.exe",
        description="Path to ASTAP executable"
    )
    astap_timeout_s: float = Field(
        default=120.0,
        ge=10.0, le=600.0,
        description="ASTAP timeout in seconds"
    )
    astap_search_radius_deg: float = Field(
        default=8.0,
        ge=0.1, le=30.0,
        description="ASTAP search radius in degrees"
    )
    astap_database: str = Field(
        default="D50",
        description="ASTAP star database (e.g., D50, H18)"
    )
    astap_fov_fudge: float = Field(
        default=1.0,
        ge=0.5, le=2.0,
        description="FOV fudge factor for ASTAP"
    )
    astap_downsample: int = Field(
        default=2,
        ge=1, le=8,
        description="ASTAP downsampling factor"
    )
    astap_max_stars: int = Field(
        default=500,
        ge=50, le=2000,
        description="Max stars for ASTAP plate solving"
    )
    astnet_local_enable: bool = Field(
        default=False,
        description="Enable local astrometry.net (solve-field) fallback"
    )
    astnet_local_use_wsl: bool = Field(
        default=True,
        description="Run solve-field via WSL"
    )
    astnet_local_command: str = Field(
        default="solve-field",
        description="solve-field command"
    )
    astnet_local_timeout_s: float = Field(
        default=300.0,
        ge=30.0, le=3600.0,
        description="solve-field timeout in seconds"
    )
    astnet_local_downsample: int = Field(
        default=2,
        ge=1, le=8,
        description="solve-field downsampling factor"
    )
    astnet_local_scale_low: float = Field(
        default=0.0,
        ge=0.0, le=10.0,
        description="solve-field scale low (0=auto)"
    )
    astnet_local_scale_high: float = Field(
        default=0.0,
        ge=0.0, le=10.0,
        description="solve-field scale high (0=auto)"
    )
    astnet_local_radius_deg: float = Field(
        default=8.0,
        ge=0.1, le=30.0,
        description="solve-field search radius in degrees"
    )
    astnet_local_keep_outputs: bool = Field(
        default=True,
        description="Keep solve-field output files"
    )
    astnet_local_use_cache: bool = Field(
        default=True,
        description="Reuse existing solve-field outputs when available"
    )
    astnet_local_max_objs: int = Field(
        default=2000,
        ge=100, le=20000,
        description="solve-field --objs limit"
    )
    astnet_local_cpulimit_s: float = Field(
        default=30.0,
        ge=1.0, le=600.0,
        description="solve-field --cpulimit (seconds)"
    )
    wcs_propagate_max_shift_px: float = Field(
        default=50.0,
        ge=1.0, le=500.0,
        description="Max match distance (px) for WCS propagation"
    )
    wcs_propagate_min_match: int = Field(
        default=12,
        ge=3, le=500,
        description="Minimum matches for WCS propagation"
    )
    wcs_propagate_sigma_clip: float = Field(
        default=3.0,
        ge=1.0, le=10.0,
        description="Sigma clip for WCS propagation offsets"
    )
    max_workers: int = Field(
        default=1,
        ge=1, le=8,
        description="Workers for WCS solving"
    )
    require_qc_pass: bool = Field(
        default=True,
        description="Require QC pass for WCS solving"
    )
    force_solve: bool = Field(
        default=False,
        description="Force re-solve even if WCS exists"
    )


class WCSRefineConfig(BaseModel):
    """WCS refinement configuration"""
    model_config = ConfigDict(validate_assignment=True)

    enable: bool = Field(
        default=True,
        description="Enable WCS refinement"
    )
    max_match: int = Field(
        default=600,
        ge=10, le=2000,
        description="Max matches for WCS refinement"
    )
    match_r_fwhm: float = Field(
        default=1.6,
        ge=0.5, le=5.0,
        description="Match radius in FWHM units"
    )
    min_match: int = Field(
        default=50,
        ge=5, le=500,
        description="Minimum matches for successful refinement"
    )
    max_sep_arcsec: float = Field(
        default=2.5,
        ge=0.1, le=10.0,
        description="Max separation for matching (arcsec)"
    )
    require: bool = Field(
        default=False,
        description="Require successful refinement"
    )


class GAIAConfig(BaseModel):
    """GAIA catalog configuration"""
    model_config = ConfigDict(validate_assignment=True)

    radius_fudge: float = Field(
        default=1.35,
        ge=1.0, le=3.0,
        description="GAIA query radius scale factor"
    )
    mag_max: float = Field(
        default=18.0,
        ge=10.0, le=25.0,
        description="Maximum GAIA magnitude"
    )
    g_limit: float = Field(
        default=18.0,
        ge=10.0, le=25.0,
        description="GAIA G magnitude limit for matching"
    )
    retry: int = Field(
        default=2,
        ge=0, le=10,
        description="Retry attempts for GAIA queries"
    )
    backoff_s: float = Field(
        default=6.0,
        ge=1.0, le=60.0,
        description="Backoff time between retries (seconds)"
    )
    allow_no_cache: bool = Field(
        default=True,
        description="Allow GAIA queries without cache"
    )
    snr_calib_min: float = Field(
        default=20.0,
        ge=1.0,
        description="GAIA SNR minimum for calibration"
    )
    gi_min: float = Field(
        default=-0.5,
        description="GAIA G-I min color for calibration"
    )
    gi_max: float = Field(
        default=4.5,
        description="GAIA G-I max color for calibration"
    )


class IDMatchConfig(BaseModel):
    """Source ID matching configuration"""
    model_config = ConfigDict(validate_assignment=True)

    tol_px: float = Field(
        default=2.0,
        ge=0.1, le=20.0,
        description="Tolerance in pixels for matching"
    )
    tol_arcsec: Optional[float] = Field(
        default=None,
        ge=0.1, le=10.0,
        description="Tolerance in arcseconds (overrides px)"
    )
    force: bool = Field(
        default=False,
        description="Force re-matching of IDs"
    )
    use_qc_pass_only: bool = Field(
        default=True,
        description="Only use QC-passed frames for ID matching"
    )
    match_r_fwhm: float = Field(
        default=0.8,
        gt=0.0, le=5.0,
        description="Match radius in FWHM units"
    )
    init_r_fwhm: float = Field(
        default=5.0,
        gt=0.0, le=20.0,
        description="Initial match radius in FWHM units"
    )
    ratio_max: float = Field(
        default=0.7,
        gt=0.0, le=1.0,
        description="Max nearest/second distance ratio"
    )
    min_pairs: int = Field(
        default=15,
        ge=2, le=2000,
        description="Minimum pairs for transform fit"
    )
    transform_mode: str = Field(
        default="similarity",
        description="Transform model: shift | similarity | affine"
    )
    mutual_nearest: bool = Field(
        default=True,
        description="Require mutual nearest match"
    )
    two_pass_enable: bool = Field(
        default=True,
        description="Enable two-pass matching (tight then loose)"
    )
    tight_radius_arcsec: float = Field(
        default=1.0,
        ge=0.1, le=10.0,
        description="Pass 1 tight match radius (arcsec)"
    )
    loose_radius_arcsec: float = Field(
        default=3.0,
        ge=0.5, le=30.0,
        description="Pass 2 loose match radius (arcsec)"
    )


class RefBuildConfig(BaseModel):
    """Reference-frame selection configuration"""
    model_config = ConfigDict(validate_assignment=True)

    mode: str = Field(
        default="hybrid",
        description="Reference catalog mode: single, stacked, gaia, hybrid"
    )
    gaia_mag_limit: float = Field(
        default=18.0,
        ge=10.0, le=22.0,
        description="Gaia magnitude limit for hybrid/gaia mode"
    )
    per_date: bool = Field(
        default=True,
        description="Build reference per date (YYYYMMDD) and merge"
    )
    sat_drop_pct: float = Field(
        default=20.0,
        ge=0.0, le=100.0,
        description="Drop top saturation frames (%)"
    )
    elong_drop_pct: float = Field(
        default=20.0,
        ge=0.0, le=100.0,
        description="Drop top elongation frames (%)"
    )
    ref_cat_max_sources: int = Field(
        default=0,
        ge=0,
        description="Max sources to keep in reference catalog (0=all)"
    )
    ref_cat_min_sources: int = Field(
        default=50,
        ge=0,
        description="Minimum sources required after filtering"
    )
    ref_cat_max_elong: float = Field(
        default=1.5,
        ge=0.0,
        description="Max elongation for reference catalog"
    )
    ref_cat_max_abs_round: float = Field(
        default=0.4,
        ge=0.0,
        description="Max absolute roundness for reference catalog"
    )
    ref_cat_sharp_min: float = Field(
        default=0.2,
        description="Min sharpness for reference catalog"
    )
    ref_cat_sharp_max: float = Field(
        default=1.0,
        description="Max sharpness for reference catalog"
    )
    ref_cat_min_peak_adu: float = Field(
        default=0.0,
        ge=0.0,
        description="Minimum peak/flux for reference catalog (0=off)"
    )
    wcs_match_radius_arcsec: float = Field(
        default=2.0,
        ge=0.1,
        description="Match radius for WCS stats (arcsec)"
    )
    wcs_min_match_rate: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Minimum WCS match rate for reference selection"
    )
    wcs_min_match_n: int = Field(
        default=50,
        ge=0,
        description="Minimum WCS match count for reference selection"
    )
    wcs_max_sep_med_arcsec: float = Field(
        default=1.5,
        ge=0.0,
        description="Max median separation (arcsec) for reference selection"
    )
    wcs_max_sep_p90_arcsec: float = Field(
        default=2.5,
        ge=0.0,
        description="Max p90 separation (arcsec) for reference selection"
    )
    wcs_max_dup_rate: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Max duplicate match rate for reference selection"
    )


class MasterCatalogConfig(BaseModel):
    """Master catalog configuration"""
    model_config = ConfigDict(validate_assignment=True)

    n_master: int = Field(
        default=1000,
        ge=10,
        description="Target number of master catalog sources"
    )
    min_frames_xy: int = Field(
        default=1,
        ge=1,
        description="Minimum frames for master catalog"
    )
    preserve_ids: bool = Field(
        default=True,
        description="Preserve IDs during master build"
    )
    force_build: bool = Field(
        default=False,
        description="Force rebuild of master catalog"
    )
    iso_min_sep_pix: float = Field(
        default=10.0,
        ge=1.0,
        description="Minimum separation for isolated sources"
    )
    keep_max: int = Field(
        default=12000,
        ge=100,
        description="Maximum sources to keep in master"
    )
    flux_quantile: float = Field(
        default=0.80,
        ge=0.1, le=1.0,
        description="Flux quantile for selection"
    )
    filter_keep: str = Field(
        default="r",
        description="Filter to keep for master catalog"
    )
    ref_frame: Optional[str] = Field(
        default=None,
        description="Reference frame for building master"
    )


class MasterEditorConfig(BaseModel):
    """Master ID editor configuration"""
    model_config = ConfigDict(validate_assignment=True)

    search_radius_px: float = Field(
        default=7.0,
        ge=1.0, le=50.0,
        description="Search radius for ID editing (pixels)"
    )
    bulk_drop_box_px: int = Field(
        default=24,
        ge=5, le=500,
        description="Bulk drop box size (pixels)"
    )
    gaia_add_max_sep_arcsec: float = Field(
        default=2.0,
        ge=0.1, le=10.0,
        description="Max separation for adding GAIA sources (arcsec)"
    )


class OverlayConfig(BaseModel):
    """Aperture overlay display configuration"""
    model_config = ConfigDict(validate_assignment=True)

    max_labels: int = Field(
        default=2000,
        ge=10, le=10000,
        description="Max labels to display in overlay"
    )
    label_fontsize: float = Field(
        default=6.0,
        ge=4.0, le=20.0,
        description="Label font size"
    )
    label_offset_px: float = Field(
        default=3.0,
        ge=0, le=20.0,
        description="Label offset from source (pixels)"
    )
    show_id_when_no_mag: bool = Field(
        default=False,
        description="Show ID if no magnitude available"
    )
    use_phot_centroid: bool = Field(
        default=True,
        description="Use photometry centroid for overlay"
    )
    show_ref_pos: bool = Field(
        default=True,
        description="Show reference positions"
    )
    show_shift_vectors: bool = Field(
        default=False,
        description="Show shift vectors"
    )
    shift_max_vectors: int = Field(
        default=300,
        ge=10, le=2000,
        description="Max shift vectors to display"
    )
    shift_min_px: float = Field(
        default=1.5,
        ge=0,
        description="Minimum shift to display (pixels)"
    )
    inspect_index: int = Field(
        default=0,
        ge=0,
        description="Default frame index for overlay inspection"
    )


class MatchConfig(BaseModel):
    """Catalog matching configuration"""
    model_config = ConfigDict(validate_assignment=True)

    tol_px: float = Field(
        default=1.0,
        ge=0.1, le=10.0,
        description="Tolerance for catalog matching (pixels)"
    )
    wcs_radius_arcsec: float = Field(
        default=1.0,
        ge=0.1, le=10.0,
        description="WCS match radius (arcsec)"
    )
    min_gaia_matches: int = Field(
        default=10,
        ge=1, le=100,
        description="Minimum GAIA matches for calibration"
    )


class ZPConfig(BaseModel):
    """Zero point fitting configuration"""
    model_config = ConfigDict(validate_assignment=True)

    clip_sigma: float = Field(
        default=3.0,
        ge=1.0, le=10.0,
        description="Sigma clipping for zero point fitting"
    )
    fit_iters: int = Field(
        default=5,
        ge=1, le=20,
        description="Iterations for zero point fitting"
    )
    slope_absmax: float = Field(
        default=1.0,
        ge=0.1, le=5.0,
        description="Maximum absolute slope for zero point"
    )
    gaia_slope_absmax: float = Field(
        default=1.0,
        ge=0.1, le=5.0,
        description="Max absolute slope for GAIA zero point"
    )


class ColorConfig(BaseModel):
    """Color calibration configuration"""
    model_config = ConfigDict(validate_assignment=True)

    clip_sigma: float = Field(
        default=3.0,
        ge=1.0, le=10.0,
        description="Sigma clipping for color fitting"
    )
    fit_iters: int = Field(
        default=5,
        ge=1, le=20,
        description="Iterations for color fitting"
    )
    slope_absmax: float = Field(
        default=2.0,
        ge=0.1, le=5.0,
        description="Maximum absolute slope for color"
    )


class CMDConfig(BaseModel):
    """CMD/calibration analysis configuration"""
    model_config = ConfigDict(validate_assignment=True)

    snr_calib_min: float = Field(
        default=20.0,
        ge=1.0,
        description="Minimum SNR for calibration sources"
    )
    max_sources: int = Field(
        default=5000,
        ge=100,
        description="Maximum sources for CMD"
    )
    zp: ZPConfig = Field(default_factory=ZPConfig)
    color: ColorConfig = Field(default_factory=ColorConfig)


class ExtinctionFitConfig(BaseModel):
    """Differential extinction fit configuration"""
    model_config = ConfigDict(validate_assignment=True)

    order: int = Field(
        default=1,
        ge=1, le=2,
        description="Extinction fit order (1 or 2)"
    )
    min_points: int = Field(
        default=5,
        ge=3, le=200,
        description="Minimum points for fit"
    )
    clip_sigma: float = Field(
        default=3.0,
        ge=0.5, le=10.0,
        description="Sigma clipping for extinction fit"
    )
    fit_iters: int = Field(
        default=5,
        ge=1, le=20,
        description="Iterations for extinction fit"
    )
    use_color_terms: bool = Field(
        default=False,
        description="Apply color term corrections before fitting"
    )
    color_index_by_filter: Dict[str, str] = Field(
        default_factory=dict,
        description="Color index mapping by filter (e.g., g:g-r)"
    )
    color_c1_by_filter: Dict[str, float] = Field(
        default_factory=dict,
        description="Color term C1 by filter"
    )
    color_c2_by_filter: Dict[str, float] = Field(
        default_factory=dict,
        description="Color term C2 by filter"
    )


class IsochroneConfig(BaseModel):
    """Isochrone fitting configuration"""
    model_config = ConfigDict(validate_assignment=True)

    file_path: str = Field(
        default="",
        description="Path to isochrone file"
    )
    age_init: float = Field(
        default=9.7,
        ge=6.0, le=10.5,
        description="Initial age guess (log age)"
    )
    mh_init: float = Field(
        default=-0.1,
        ge=-3.0, le=1.0,
        description="Initial metallicity guess [M/H]"
    )
    eg_r_init: float = Field(
        default=0.0033,
        ge=0.0, le=2.0,
        description="Initial extinction E(g-r)"
    )
    dm_init: float = Field(
        default=9.46,
        ge=0.0, le=25.0,
        description="Initial distance modulus"
    )

    # Column indices for isochrone file (CMD 3.x SDSS format)
    col_mh: int = Field(
        default=1,
        ge=0,
        description="Metallicity column index"
    )
    col_age: int = Field(
        default=2,
        ge=0,
        description="log(Age) column index"
    )
    col_g: int = Field(
        default=29,
        ge=0,
        description="g-band magnitude column index"
    )
    col_r: int = Field(
        default=30,
        ge=0,
        description="r-band magnitude column index"
    )

    # Fitting parameters
    fit_fraction: float = Field(
        default=0.6,
        ge=0.1, le=1.0,
        description="Fraction of closest stars to use in robust fitting"
    )


class HUD5XConfig(BaseModel):
    """5X HUD viewer configuration"""
    model_config = ConfigDict(validate_assignment=True)

    aperture_scale: Optional[float] = Field(
        default=1.0,
        gt=0, le=10.0,
        description="5X viewer aperture scale"
    )
    annulus_in_scale: Optional[float] = Field(
        default=4.0,
        gt=0, le=20.0,
        description="5X viewer inner annulus scale"
    )
    annulus_out_scale: Optional[float] = Field(
        default=2.0,
        gt=0, le=10.0,
        description="5X viewer annulus width scale"
    )
    center_cbox_scale: Optional[float] = Field(
        default=1.5,
        gt=0, le=5.0,
        description="5X viewer center box scale"
    )
    neighbor_mask_scale: Optional[float] = Field(
        default=1.3,
        ge=1.0, le=5.0,
        description="5X viewer neighbor mask scale"
    )
    sigma_clip: Optional[float] = Field(
        default=3.0,
        ge=1.0, le=10.0,
        description="5X viewer sigma clipping"
    )
    mag_flux: MagFluxMode = Field(
        default=MagFluxMode.RATE_E,
        description="Display mode (rate_e or flux_e)"
    )
    use_header_exptime: bool = Field(
        default=True,
        description="Use exposure time from FITS header"
    )
    min_r_ap_px: Optional[float] = Field(
        default=12.0,
        ge=1.0,
        description="5X viewer minimum aperture radius"
    )
    min_r_in_px: Optional[float] = Field(
        default=24.0,
        ge=1.0,
        description="5X viewer minimum inner annulus radius"
    )
    min_r_out_px: Optional[float] = Field(
        default=36.0,
        ge=1.0,
        description="5X viewer minimum outer annulus radius"
    )


class TransformConfig(BaseModel):
    """Transformation/alignment output configuration"""
    model_config = ConfigDict(validate_assignment=True)

    save_src2ref: bool = Field(
        default=True,
        description="Save source-to-reference transforms"
    )


# =============================================================================
# Main Parameters class
# =============================================================================

class Parameters(BaseModel):
    """
    Main configuration container for aperture photometry pipeline.
    All parameters are validated on assignment for GUI integration.
    """
    model_config = ConfigDict(validate_assignment=True)

    # I/O and basic settings
    io: IOConfig = Field(default_factory=IOConfig)
    target: TargetConfig = Field(default_factory=TargetConfig)
    parallel: ParallelConfig = Field(default_factory=ParallelConfig)
    ui: UIConfig = Field(default_factory=UIConfig)

    # Instrument configuration
    instrument: InstrumentConfig

    # Processing settings
    alignment: AlignmentConfig = Field(default_factory=AlignmentConfig)
    fwhm: FWHMConfig = Field(default_factory=FWHMConfig)
    clip: ClipConfig = Field(default_factory=ClipConfig)
    qc: QCConfig = Field(default_factory=QCConfig)

    # Detection
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    background: Background2DConfig = Field(default_factory=Background2DConfig)

    # Photometry
    photometry: PhotConfig = Field(default_factory=PhotConfig)

    # WCS and astrometry
    wcs: WCSConfig = Field(default_factory=WCSConfig)
    wcs_refine: WCSRefineConfig = Field(default_factory=WCSRefineConfig)
    gaia: GAIAConfig = Field(default_factory=GAIAConfig)

    # ID matching and catalogs
    refbuild: RefBuildConfig = Field(default_factory=RefBuildConfig)
    idmatch: IDMatchConfig = Field(default_factory=IDMatchConfig)
    master: MasterCatalogConfig = Field(default_factory=MasterCatalogConfig)
    master_editor: MasterEditorConfig = Field(default_factory=MasterEditorConfig)

    # Analysis and calibration
    match: MatchConfig = Field(default_factory=MatchConfig)
    cmd: CMDConfig = Field(default_factory=CMDConfig)
    extinction_fit: ExtinctionFitConfig = Field(default_factory=ExtinctionFitConfig)
    isochrone: IsochroneConfig = Field(default_factory=IsochroneConfig)

    # Visualization
    overlay: OverlayConfig = Field(default_factory=OverlayConfig)
    hud5x: HUD5XConfig = Field(default_factory=HUD5XConfig)

    # Transforms
    transform: TransformConfig = Field(default_factory=TransformConfig)

    # Internal computed values (not saved to TOML)
    _config_hash: str = ""
    _resolved_result_dir: Optional[Path] = None
    _resolved_cache_dir: Optional[Path] = None

    @classmethod
    def from_toml(cls, path: Path | str) -> "Parameters":
        """Load parameters from TOML file"""
        path = Path(path)
        with open(path, "rb") as f:
            data = tomllib.load(f)

        # Handle nested configs
        params = cls._parse_toml_data(data)
        params._config_hash = cls._compute_hash(path)
        params._resolve_paths()
        return params

    @classmethod
    def _parse_toml_data(cls, data: Dict[str, Any]) -> "Parameters":
        """Parse TOML data into Parameters object"""
        # Extract nested configurations
        detection_data = data.get("detection", {})
        detection_data["deblend"] = detection_data.pop("deblend", {})
        detection_data["peak"] = detection_data.pop("peak", {})
        detection_data["dao"] = detection_data.pop("dao", {})

        photometry_data = data.get("photometry", {})
        photometry_data["scales"] = photometry_data.pop("scales", {})
        photometry_data["radii"] = photometry_data.pop("radii", {})
        photometry_data["apcorr"] = photometry_data.pop("apcorr", {})

        cmd_data = data.get("cmd", {})
        cmd_data["zp"] = cmd_data.pop("zp", {})
        cmd_data["color"] = cmd_data.pop("color", {})

        return cls(
            io=IOConfig(**data.get("io", {})),
            target=TargetConfig(**data.get("target", {})),
            parallel=ParallelConfig(**data.get("parallel", {})),
            ui=UIConfig(**data.get("ui", {})),
            instrument=InstrumentConfig(**data.get("instrument", {})),
            alignment=AlignmentConfig(**data.get("alignment", {})),
            fwhm=FWHMConfig(**data.get("fwhm", {})),
            clip=ClipConfig(**data.get("clip", {})),
            qc=QCConfig(**data.get("qc", {})),
            detection=DetectionConfig(**detection_data),
            background=Background2DConfig(**data.get("background", {})),
            photometry=PhotConfig(**photometry_data),
            wcs=WCSConfig(**data.get("wcs", {})),
            wcs_refine=WCSRefineConfig(**data.get("wcs_refine", {})),
            gaia=GAIAConfig(**data.get("gaia", {})),
            refbuild=RefBuildConfig(**data.get("refbuild", {})),
            idmatch=IDMatchConfig(**data.get("idmatch", {})),
            master=MasterCatalogConfig(**data.get("master", {})),
            master_editor=MasterEditorConfig(**data.get("master_editor", {})),
            match=MatchConfig(**data.get("match", {})),
            cmd=CMDConfig(**cmd_data),
            extinction_fit=ExtinctionFitConfig(**data.get("extinction_fit", {})),
            isochrone=IsochroneConfig(**data.get("isochrone", {})),
            overlay=OverlayConfig(**data.get("overlay", {})),
            hud5x=HUD5XConfig(**data.get("hud5x", {})),
            transform=TransformConfig(**data.get("transform", {})),
        )

    def to_toml(self, path: Path | str) -> None:
        """Save parameters to TOML file"""
        path = Path(path)
        data = self._to_toml_dict()

        with open(path, "wb") as f:
            tomli_w.dump(data, f)

    def _to_toml_dict(self) -> Dict[str, Any]:
        """Convert to TOML-compatible dictionary"""
        def convert(obj):
            if isinstance(obj, BaseModel):
                result = {}
                for key, value in obj.model_dump().items():
                    if not key.startswith("_"):
                        result[key] = convert(value)
                return result
            elif isinstance(obj, Enum):
                return obj.value
            elif isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            else:
                return obj

        return convert(self)

    def _resolve_paths(self) -> None:
        """Resolve result and cache directories"""
        data_dir = Path(self.io.data_dir)

        if self.io.result_dir:
            self._resolved_result_dir = Path(self.io.result_dir)
        else:
            self._resolved_result_dir = data_dir / "result"

        self._resolved_result_dir.mkdir(parents=True, exist_ok=True)

        self._resolved_cache_dir = self._resolved_result_dir / self.io.cache_dir
        self._resolved_cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def result_dir(self) -> Path:
        """Get resolved result directory"""
        if self._resolved_result_dir is None:
            self._resolve_paths()
        return self._resolved_result_dir

    @property
    def cache_dir(self) -> Path:
        """Get resolved cache directory"""
        if self._resolved_cache_dir is None:
            self._resolve_paths()
        return self._resolved_cache_dir

    @property
    def data_dir(self) -> Path:
        """Get data directory as Path"""
        return Path(self.io.data_dir)

    @staticmethod
    def _compute_hash(path: Path) -> str:
        """Compute SHA1 hash of config file content"""
        try:
            content = path.read_bytes()
            return hashlib.sha1(content).hexdigest()
        except Exception:
            return "NO_CONFIG"

    def print_summary(self) -> None:
        """Print parameter summary"""
        print("\n==================== PARAM SUMMARY ====================")
        print(f"DATA_DIR      : {self.io.data_dir}")
        print(f"RESULT_DIR    : {self.result_dir}")
        print(f"CACHE_DIR     : {self.cache_dir}")
        print(f"TARGET        : {self.target.name}")
        print(f"parallel_mode : {self.parallel.mode.value} | max_workers={self.parallel.max_workers}")
        print(f"resume_mode   : {self.parallel.resume_mode} | force_redetect={self.parallel.force_redetect}")
        print(f"FWHM range    : {self.fwhm.px_min:.2f} ~ {self.fwhm.px_max:.2f} px")
        print(f"detect_sigma  : base={self.detection.sigma}")
        print(f"camera        : gain={self.instrument.gain_e_per_adu} e-/ADU | rdnoise={self.instrument.rdnoise_e} e-")
        print(f"pixel_scale   : {self.instrument.pixel_scale_arcsec:.4f} arcsec/px")
        print("=======================================================\n")


# =============================================================================
# Factory function
# =============================================================================

def load_parameters(path: Path | str = "parameters.toml") -> Parameters:
    """
    Load parameters from TOML file.

    Args:
        path: Path to TOML config file

    Returns:
        Parameters object with validated configuration
    """
    return Parameters.from_toml(path)
