"""
Parameter management for aperture photometry pipeline
Extracted from AAPKI_GUI.ipynb Cell 0
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Iterable
import hashlib
import types
try:  # Python 3.11+
    import tomllib  # type: ignore
except Exception:  # Python 3.10 and earlier
    import tomli as tomllib  # type: ignore
try:
    import tomli_w  # type: ignore
except Exception:
    tomli_w = None


def _as_bool(v, default=False):
    """Convert value to boolean"""
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def _as_float_or_none(v):
    """Convert value to float or None"""
    try:
        s = str(v).strip()
        return float(s) if s != "" else None
    except:
        return None


def _get_path(data: Dict[str, Any], path: Iterable[str]):
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _set_path(data: Dict[str, Any], path: Iterable[str], value: Any) -> None:
    """Set a value at a nested path in a dictionary, creating intermediate dicts as needed."""
    cur: Any = data
    keys = list(path)
    for key in keys[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[keys[-1]] = value


TOML_KEY_MAP: list[tuple[Iterable[str], str]] = [
    (("io", "data_dir"), "data_dir"),
    (("io", "filename_prefix"), "filename_prefix"),
    (("io", "result_dir"), "result_dir"),
    (("io", "cache_dir"), "cache_dir"),
    (("parallel", "mode"), "parallel_mode"),
    (("parallel", "max_workers"), "parallel_max_workers"),
    (("parallel", "resume_mode"), "resume_mode"),
    (("parallel", "force_redetect"), "force_redetect"),
    (("parallel", "force_rephot"), "force_rephot"),
    (("parallel", "detect_cache_strategy"), "detect_cache_strategy"),
    (("ui", "log_tail"), "ui_log_tail"),
    (("ui", "detect_progress_bar"), "detect_progress_bar"),
    (("alignment", "ref_index"), "align_ref_index"),
    (("alignment", "global_align"), "global_align"),
    (("alignment", "global_ref_filter"), "global_ref_filter"),
    (("alignment", "global_ref_index"), "global_ref_index"),
    (("fwhm", "guess_px"), "fwhm_pix_guess"),
    (("fwhm", "guess_arcsec"), "fwhm_guess_arcsec"),
    (("fwhm", "arcsec_min"), "fwhm_arcsec_min"),
    (("fwhm", "arcsec_max"), "fwhm_arcsec_max"),
    (("fwhm", "px_min"), "fwhm_px_min"),
    (("fwhm", "px_max"), "fwhm_px_max"),
    (("fwhm", "qc_max_sources"), "fwhm_qc_max_sources"),
    (("fwhm", "elong_max"), "fwhm_elong_max"),
    (("fwhm", "iso_min_sep_pix"), "iso_min_sep_pix"),
    (("fwhm", "measure_max"), "fwhm_measure_max"),
    (("fwhm", "dr"), "fwhm_dr"),
    (("clip", "min_adu"), "clip_min_adu"),
    (("clip", "max_adu"), "clip_max_adu"),
    (("detection", "engine"), "detect_engine"),
    (("detection", "sigma"), "detect_sigma"),
    (("detection", "sigma_g"), "detect_sigma_g"),
    (("detection", "sigma_r"), "detect_sigma_r"),
    (("detection", "sigma_i"), "detect_sigma_i"),
    (("detection", "minarea_pix"), "minarea_pix"),
    (("detection", "keep_max"), "detect_keep_max"),
    (("detection", "dilate_radius_px"), "segm_dilate_radius_px"),
    (("detection", "deblend", "enable"), "deblend_enable"),
    (("detection", "deblend", "nthresh"), "deblend_nthresh"),
    (("detection", "deblend", "contrast"), "deblend_cont"),
    (("detection", "deblend", "max_labels"), "deblend_max_labels"),
    (("detection", "deblend", "label_hard_max"), "deblend_label_hard_max"),
    (("detection", "deblend", "nlevels_soft"), "deblend_nlevels_soft"),
    (("detection", "deblend", "contrast_soft"), "deblend_contrast_soft"),
    (("detection", "peak", "enable"), "peak_pass_enable"),
    (("detection", "peak", "nsigma"), "peak_nsigma"),
    (("detection", "peak", "min_sep_px"), "peak_min_sep_px"),
    (("detection", "peak", "max_add"), "peak_max_add"),
    (("detection", "peak", "max_elong"), "peak_max_elong"),
    (("detection", "peak", "sharp_lo"), "peak_sharp_lo"),
    (("detection", "peak", "skip_if_nsrc_ge"), "peak_skip_if_nsrc_ge"),
    (("detection", "dao", "enable"), "dao_refine_enable"),
    (("detection", "dao", "fwhm_px"), "dao_fwhm_px"),
    (("detection", "dao", "sharp_lo"), "dao_sharp_lo"),
    (("detection", "dao", "sharp_hi"), "dao_sharp_hi"),
    (("detection", "dao", "round_lo"), "dao_round_lo"),
    (("detection", "dao", "round_hi"), "dao_round_hi"),
    (("detection", "dao", "match_tol_px"), "dao_match_tol_px"),
    (("background", "enable"), "bkg2d_enable"),
    (("background", "in_detect"), "bkg2d_in_detect"),
    (("background", "box"), "bkg2d_box"),
    (("background", "filter_size"), "bkg2d_filter_size"),
    (("background", "edge_method"), "bkg2d_edge_method"),
    (("background", "method"), "bkg2d_method"),
    (("background", "downsample"), "bkg2d_downsample"),
    (("qc", "gate_enable"), "gate_enable"),
    (("qc", "sky_sigma_max_e"), "gate_sky_sigma_max_e"),
    (("qc", "nsrc_min"), "gate_nsrc_min"),
    (("qc", "keep_positions_if_fail"), "keep_positions_if_qc_fail"),
    (("photometry", "mode"), "aperture_mode"),
    (("photometry", "recenter"), "recenter_aperture"),
    (("photometry", "use_segm_mask"), "bkg_use_segm_mask"),
    (("photometry", "min_snr_for_mag"), "min_snr_for_mag"),
    (("photometry", "use_qc_pass_only"), "phot_use_qc_pass_only"),
    (("photometry", "scales", "aperture_scale"), "phot_aperture_scale"),
    (("photometry", "scales", "annulus_scale"), "fitsky_annulus_scale"),
    (("photometry", "scales", "dannulus_scale"), "fitsky_dannulus_scale"),
    (("photometry", "scales", "center_cbox_scale"), "center_cbox_scale"),
    (("photometry", "scales", "annulus_min_gap_px"), "annulus_min_gap_px"),
    (("photometry", "scales", "annulus_min_width_px"), "annulus_min_width_px"),
    (("photometry", "radii", "min_r_ap_px"), "min_r_ap_px"),
    (("photometry", "radii", "min_r_in_px"), "min_r_in_px"),
    (("photometry", "radii", "min_r_out_px"), "min_r_out_px"),
    (("photometry", "radii", "sigma_clip"), "annulus_sigma_clip"),
    (("photometry", "radii", "max_iter"), "fitsky_max_iter"),
    (("photometry", "radii", "neighbor_mask_scale"), "annulus_neighbor_mask_scale"),
    (("photometry", "apcorr", "apply"), "apcorr_apply"),
    (("photometry", "apcorr", "small_scale"), "apcorr_small_scale"),
    (("photometry", "apcorr", "large_scale"), "apcorr_large_scale"),
    (("photometry", "apcorr", "min_n"), "apcorr_use_min_n"),
    (("photometry", "apcorr", "scatter_max"), "apcorr_scatter_max"),
        (("wcs", "astap_exe"), "astap_exe"),
        (("wcs", "timeout_s"), "astap_timeout_s"),
        (("wcs", "astap_search_radius_deg"), "astap_search_radius_deg"),
        (("wcs", "astap_database"), "astap_database"),
        (("wcs", "astap_fov_fudge"), "astap_fov_fudge"),
    (("wcs", "astap_downsample"), "astap_downsample_z"),
    (("wcs", "astap_max_stars"), "astap_max_stars_s"),
    (("wcs", "require_qc_pass"), "wcs_require_qc_pass"),
    (("wcs", "refine_enable"), "wcs_refine_enable"),
    (("wcs_refine", "enable"), "wcs_refine_enable"),
    (("wcs_refine", "max_match"), "wcs_refine_max_match"),
    (("wcs_refine", "match_r_fwhm"), "wcs_refine_match_r_fwhm"),
    (("wcs_refine", "min_match"), "wcs_refine_min_match"),
    (("gaia", "radius_fudge"), "gaia_radius_fudge"),
    (("gaia", "mag_max"), "gaia_mag_max"),
    (("gaia", "snr_calib_min"), "gaia_snr_calib_min"),
    (("gaia", "gi_min"), "gaia_gi_min"),
    (("gaia", "gi_max"), "gaia_gi_max"),
    (("gaia", "retry"), "gaia_retry"),
    (("gaia", "backoff_s"), "gaia_backoff_s"),
    (("gaia", "allow_no_cache"), "gaia_allow_no_cache"),
    (("gaia", "g_limit"), "idmatch_gaia_g_limit"),
    (("idmatch", "gaia_g_limit"), "idmatch_gaia_g_limit"),
    (("idmatch", "tol_px"), "idmatch_tol_px"),
    (("idmatch", "tol_arcsec"), "idmatch_tol_arcsec"),
    (("idmatch", "force"), "force_idmatch"),
    (("idmatch", "use_qc_pass_only"), "idmatch_use_qc_pass_only"),
    (("master", "n_master"), "N_master"),
    (("master", "iso_min_sep_pix"), "master_iso_min_sep_pix"),
    (("master", "keep_max"), "master_keep_max"),
    (("master", "flux_quantile"), "master_flux_quantile"),
    (("master", "filter_keep"), "master_filter_keep"),
    (("master", "min_frames_xy"), "master_min_frames_xy"),
    (("master", "preserve_ids"), "master_preserve_ids"),
    (("master", "force_build"), "force_master_build"),
    (("master_editor", "search_radius_px"), "search_radius_px"),
    (("master_editor", "bulk_drop_box_px"), "bulk_drop_box_px"),
    (("master_editor", "gaia_add_max_sep_arcsec"), "gaia_add_max_sep_arcsec"),
    (("match", "tol_px"), "match_tol_px"),
    (("match", "wcs_radius_arcsec"), "wcs_match_radius_arcsec"),
    (("match", "min_gaia_matches"), "min_master_gaia_matches"),
    (("cmd", "snr_calib_min"), "cmd_snr_calib_min"),
    (("cmd", "max_sources"), "cmd_max_sources"),
    (("cmd", "apply_extinction"), "cmd_apply_extinction"),
    (("cmd", "extinction_mode"), "cmd_extinction_mode"),
    (("cmd", "frame_zp_min_n"), "frame_zp_min_n"),
    (("cmd", "zp", "clip_sigma"), "zp_clip_sigma"),
    (("cmd", "zp", "fit_iters"), "zp_fit_iters"),
    (("cmd", "zp", "slope_absmax"), "zp_slope_absmax"),
    (("cmd", "color", "clip_sigma"), "color_clip_sigma"),
    (("cmd", "color", "fit_iters"), "color_fit_iters"),
    (("cmd", "color", "slope_absmax"), "color_slope_absmax"),
    (("gaia", "zp_slope_absmax"), "gaia_zp_slope_absmax"),
    (("gaia", "color_slope_absmax"), "gaia_color_slope_absmax"),
    (("overlay", "max_labels"), "overlay_max_labels"),
    (("overlay", "label_fontsize"), "overlay_label_fontsize"),
    (("overlay", "label_offset_px"), "overlay_label_offset_px"),
    (("overlay", "show_id_when_no_mag"), "overlay_show_id_when_no_mag"),
    (("overlay", "use_phot_centroid"), "overlay_use_phot_centroid"),
    (("overlay", "show_ref_pos"), "overlay_show_ref_pos"),
    (("overlay", "show_shift_vectors"), "overlay_show_shift_vectors"),
    (("overlay", "shift_max_vectors"), "overlay_shift_max_vectors"),
    (("overlay", "shift_min_px"), "overlay_shift_min_px"),
    (("overlay", "inspect_index"), "inspect_index"),
    (("ui", "canvas_px"), "ui_canvas_px"),
    (("transform", "save_src2ref"), "save_src2ref_tforms"),
    (("detection", "peak", "kernel_scales"), "peak_kernel_scales"),
    (("target", "name"), "target_name"),
    (("target", "ra_deg"), "target_ra_deg"),
    (("target", "dec_deg"), "target_dec_deg"),
    (("instrument", "gain_e_per_adu"), "gain_e_per_adu"),
    (("instrument", "rdnoise_e"), "rdnoise_e"),
    (("instrument", "saturation_adu"), "saturation_adu"),
    (("instrument", "datamin_adu"), "datamin_adu"),
    (("instrument", "datamax_adu"), "datamax_adu"),
    (("instrument", "binning"), "binning_default"),
    (("instrument", "telescope_focal_mm"), "telescope_focal_mm"),
    (("instrument", "camera_pixel_um"), "camera_pixel_um"),
    (("instrument", "zp_initial"), "zp_initial"),
    (("site", "lat_deg"), "site_lat_deg"),
    (("site", "lon_deg"), "site_lon_deg"),
    (("site", "alt_m"), "site_alt_m"),
    (("site", "tz_offset_hours"), "site_tz_offset_hours"),
]


def _read_toml(path: Path) -> Dict[str, Any]:
    """Read TOML config into a flat key dict compatible with legacy Parameters."""
    if not path.exists():
        return {}

    with path.open("rb") as f:
        data = tomllib.load(f)

    raw: Dict[str, Any] = {}

    def set_if(key: str, value: Any):
        if value is not None:
            raw[key] = value

    # Use the global TOML_KEY_MAP for mapping
    for path_keys, out_key in TOML_KEY_MAP:
        value = _get_path(data, path_keys)
        set_if(out_key, value)

    peak_kernel_scales = _get_path(data, ("detection", "peak", "kernel_scales"))
    if isinstance(peak_kernel_scales, list):
        raw["peak_kernel_scales"] = ",".join(str(v) for v in peak_kernel_scales)

    target_ra = _get_path(data, ("target", "ra_deg"))
    target_dec = _get_path(data, ("target", "dec_deg"))
    target_name = _get_path(data, ("target", "name"))
    set_if("target_name", target_name)
    set_if("target_ra_deg", target_ra)
    set_if("target_dec_deg", target_dec)
    set_if("ra_deg", target_ra)
    set_if("dec_deg", target_dec)

    inst = _get_path(data, ("instrument",)) or {}
    if isinstance(inst, dict):
        set_if("telescope_focal_mm", inst.get("telescope_focal_mm"))
        set_if("camera_pixel_um", inst.get("camera_pixel_um"))
        set_if("camera_binning", inst.get("binning"))
        set_if("binning_default", inst.get("binning"))
        set_if("gain_e_per_adu", inst.get("gain_e_per_adu"))
        set_if("rdnoise_e", inst.get("rdnoise_e"))
        set_if("saturation_adu", inst.get("saturation_adu"))
        set_if("zp_initial", inst.get("zp_initial"))
        set_if("datamin_adu", inst.get("datamin_adu"))
        set_if("datamax_adu", inst.get("datamax_adu"))

    hud5x = _get_path(data, ("hud5x",)) or {}
    if isinstance(hud5x, dict):
        set_if("5x.aperture_scale", hud5x.get("aperture_scale"))
        set_if("5x.center_cbox_scale", hud5x.get("center_cbox_scale"))
        set_if("5x.annulus_in_scale", hud5x.get("annulus_in_scale"))
        set_if("5x.annulus_out_scale", hud5x.get("annulus_out_scale"))
        set_if("5x.sigma_clip", hud5x.get("sigma_clip"))
        set_if("5x.neighbor_mask_scale", hud5x.get("neighbor_mask_scale"))
        set_if("5x.mag_flux", str(hud5x.get("mag_flux")) if hud5x.get("mag_flux") is not None else None)
        set_if("5x.use_header_exptime", hud5x.get("use_header_exptime"))
        set_if("5x.min_r_ap_px", hud5x.get("min_r_ap_px"))
        set_if("5x.min_r_in_px", hud5x.get("min_r_in_px"))
        set_if("5x.min_r_out_px", hud5x.get("min_r_out_px"))

    site = _get_path(data, ("site",)) or {}
    if isinstance(site, dict):
        set_if("site_lat_deg", site.get("lat_deg"))
        set_if("site_lon_deg", site.get("lon_deg"))
        set_if("site_alt_m", site.get("alt_m"))
        set_if("site_tz_offset_hours", site.get("tz_offset_hours"))

    legacy = _get_path(data, ("legacy",)) or {}
    if isinstance(legacy, dict):
        for key, value in legacy.items():
            raw.setdefault(key, value)

    return raw


def _getf(raw, key, default):
    """Get float value from raw dict"""
    s = str(raw.get(key, "")).strip()
    try:
        return default if s == "" else float(s)
    except:
        return default


def _geti(raw, key, default):
    """Get int value from raw dict"""
    s = str(raw.get(key, "")).strip()
    try:
        if s == "":
            return default
        return int(float(s))
    except:
        return default


class Parameters:
    """
    Main parameter container for the photometry pipeline
    All configuration is stored as a SimpleNamespace for easy attribute access
    """

    def __init__(self, param_file: str | Path = "parameters.toml"):
        """Initialize parameters from file"""
        self.param_file = Path(param_file)
        self.P = self._load_from_file(self.param_file)
        self.param_hash = self._compute_hash(self.param_file)

    @staticmethod
    def _load_from_file(path: Path) -> types.SimpleNamespace:
        """Load parameters from text file"""
        raw = _read_toml(path)

        # --- rdnoise is REQUIRED ---
        rdnoise_candidate = (
            _as_float_or_none(raw.get("rdnoise_e", ""))
            or _as_float_or_none(raw.get("datapar.readnoise", ""))
            or _as_float_or_none(raw.get("readnoise_e", ""))
        )
        if rdnoise_candidate is None:
            raise RuntimeError(
                "[parameter.txt] read noise value is required.\n"
                "  Allowed keys: rdnoise_e or datapar.readnoise or readnoise_e\n"
                "  Example: rdnoise_e = 1.39   # electrons"
            )

        P = types.SimpleNamespace(
            # I/O
            data_dir=raw.get("data_dir", "."),
            filename_prefix=raw.get("filename_prefix", "pp_"),
            result_dir=raw.get("result_dir", ""),
            cache_dir=raw.get("cache_dir", "cache"),

            # Parallel processing
            parallel_mode=raw.get("parallel_mode", "thread"),
            parallel_max_workers=_geti(raw, "parallel_max_workers", 0),  # 0 = auto
            ui_log_tail=_geti(raw, "ui_log_tail", 300),
            resume_mode=_as_bool(raw.get("resume_mode", "true"), True),
            force_redetect=_as_bool(raw.get("force_redetect", "false"), False),
            force_rephot=_as_bool(raw.get("force_rephot", "false"), False),
            detect_cache_strategy=raw.get("detect_cache_strategy", "mtime"),
            detect_progress_bar=_as_bool(raw.get("detect_progress_bar", "true"), True),

            # Alignment
            align_ref_index=_geti(raw, "align_ref_index", 0),
            global_align=_as_bool(raw.get("global_align", "true"), True),
            global_ref_filter=raw.get("global_ref_filter", "r"),
            global_ref_index=_geti(raw, "global_ref_index", 0),

            # FWHM/PSF parameters
            fwhm_pix_guess=_as_float_or_none(raw.get("fwhm_pix_guess", "")),
            fwhm_guess_arcsec=_as_float_or_none(raw.get("fwhm_guess_arcsec", "")),
            fwhm_arcsec_min=_as_float_or_none(raw.get("fwhm_arcsec_min", "")),
            fwhm_arcsec_max=_as_float_or_none(raw.get("fwhm_arcsec_max", "")),
            fwhm_px_min=_getf(raw, "fwhm_px_min", 3.5),
            fwhm_px_max=_getf(raw, "fwhm_px_max", 12.0),
            fwhm_qc_max_sources=_geti(raw, "fwhm_qc_max_sources", 40),
            fwhm_elong_max=_getf(raw, "fwhm_elong_max", 1.3),
            iso_min_sep_pix=_getf(raw, "iso_min_sep_pix", 18.0),

            # Detection parameters
            detect_engine=raw.get("detect_engine", "segm"),
            detect_sigma=_getf(raw, "detect_sigma", 3.2),
            detect_sigma_g=_as_float_or_none(raw.get("detect_sigma_g", "")),
            detect_sigma_r=_as_float_or_none(raw.get("detect_sigma_r", "")),
            detect_sigma_i=_as_float_or_none(raw.get("detect_sigma_i", "")),
            minarea_pix=_geti(raw, "minarea_pix", 3),
            detect_keep_max=_geti(raw, "detect_keep_max", 6000),
            deblend_enable=_as_bool(raw.get("deblend_enable", "true"), True),
            deblend_nthresh=_geti(raw, "deblend_nthresh", 64),
            deblend_cont=_getf(raw, "deblend_cont", 0.0025),
            segm_dilate_radius_px=_geti(raw, "segm_dilate_radius_px", 4),
            deblend_max_labels=_geti(raw, "deblend_max_labels", 4000),
            deblend_label_hard_max=_geti(raw, "deblend_label_hard_max", 7000),
            deblend_nlevels_soft=_geti(raw, "deblend_nlevels_soft", 32),
            deblend_contrast_soft=_getf(raw, "deblend_contrast_soft", 0.005),
            dao_refine_enable=_as_bool(raw.get("dao_refine_enable", "false"), False),
            dao_fwhm_px=_getf(raw, "dao_fwhm_px", 6.0),
            dao_sharp_lo=_getf(raw, "dao_sharp_lo", 0.2),
            dao_sharp_hi=_getf(raw, "dao_sharp_hi", 1.0),
            dao_round_lo=_getf(raw, "dao_round_lo", -0.5),
            dao_round_hi=_getf(raw, "dao_round_hi", 0.5),
            dao_match_tol_px=_getf(raw, "dao_match_tol_px", 2.0),

            # Background parameters
            bkg2d_enable=_as_bool(raw.get("bkg2d_enable", "true"), True),
            bkg2d_in_detect=_as_bool(raw.get("bkg2d_in_detect", "true"), True),
            bkg2d_box=_geti(raw, "bkg2d_box", 64),
            bkg2d_filter_size=_geti(raw, "bkg2d_filter_size", 3),
            bkg2d_edge_method=raw.get("bkg2d_edge_method", "pad"),
            bkg2d_method=raw.get("bkg2d_method", "median"),
            bkg2d_downsample=_geti(raw, "bkg2d_downsample", 4),

            # Quality control
            gate_enable=_as_bool(raw.get("gate_enable", "true"), True),

            # Photometry aperture scales
            phot_aperture_scale=_getf(raw, "phot_aperture_scale", 1.0),
            fitsky_annulus_scale=_getf(raw, "fitsky_annulus_scale", 4.0),
            fitsky_dannulus_scale=_getf(raw, "fitsky_dannulus_scale", 2.0),
            center_cbox_scale=_getf(raw, "center_cbox_scale", 1.5),
            annulus_min_gap_px=_getf(raw, "annulus_min_gap_px", 6.0),
            annulus_min_width_px=_getf(raw, "annulus_min_width_px", 12.0),

            # Camera/instrument
            saturation_adu=_getf(raw, "saturation_adu", 60000.0),
            datamin_adu=_getf(raw, "datamin_adu", 0.1),
            datamax_adu=_getf(raw, "datamax_adu", 60000.0),
            gain_e_per_adu=_getf(raw, "gain_e_per_adu", 0.1),
            rdnoise_e=float(rdnoise_candidate),
            zp_initial=_getf(raw, "zp_initial", 25.0),
            binning_default=_geti(raw, "binning_default", 2),
            site_lat_deg=_getf(raw, "site_lat_deg", 0.0),
            site_lon_deg=_getf(raw, "site_lon_deg", 0.0),
            site_alt_m=_getf(raw, "site_alt_m", 0.0),
            site_tz_offset_hours=_getf(raw, "site_tz_offset_hours", 0.0),

            # 5X HUD viewer parameters
            _hud5={
                "5x.aperture_scale": raw.get("5x.aperture_scale", ""),
                "5x.center_cbox_scale": raw.get("5x.center_cbox_scale", ""),
                "5x.annulus_in_scale": raw.get("5x.annulus_in_scale", ""),
                "5x.annulus_out_scale": raw.get("5x.annulus_out_scale", ""),
                "5x.min_r_ap_px": raw.get("5x.min_r_ap_px", ""),
                "5x.min_r_in_px": raw.get("5x.min_r_in_px", ""),
                "5x.min_r_out_px": raw.get("5x.min_r_out_px", ""),
                "5x.sigma_clip": raw.get("5x.sigma_clip", ""),
                "5x.neighbor_mask_scale": raw.get("5x.neighbor_mask_scale", ""),
                "5x.mag_flux": raw.get("5x.mag_flux", "rate_e"),
                "5x.use_header_exptime": raw.get("5x.use_header_exptime", "true"),
            },

            # Photometry execution flags
            aperture_mode=raw.get("aperture_mode", "apcorr"),
            annulus_neighbor_mask_scale=_getf(raw, "annulus_neighbor_mask_scale", 1.3),
            recenter_aperture=_as_bool(raw.get("recenter_aperture", "true"), True),
            bkg_use_segm_mask=_as_bool(raw.get("bkg_use_segm_mask", "true"), True),
            min_snr_for_mag=_getf(raw, "min_snr_for_mag", 3.0),
            max_recenter_shift=_getf(raw, "max_recenter_shift", 2.0),
            centroid_outlier_px=_getf(raw, "centroid_outlier_px", 1.0),
            sky_sigma_mode=raw.get("sky_sigma_mode", "local"),
            sky_sigma_includes_rn=_as_bool(raw.get("sky_sigma_includes_rn", "true"), True),
            sky_sigma_min_n_sky=_geti(raw, "sky_sigma_min_n_sky", 50),

            # ID matching
            idmatch_gaia_g_limit=_getf(raw, "idmatch_gaia_g_limit", 18.0),
            idmatch_tol_arcsec=_as_float_or_none(raw.get("idmatch_tol_arcsec", "")),
            idmatch_tol_px=_getf(raw, "idmatch_tol_px", 2.0),
            force_idmatch=_as_bool(raw.get("force_idmatch", "false"), False),
            platesolve_gaia_radius_scale=_getf(raw, "platesolve_gaia_radius_scale", 1.35),

            # Aperture/annulus config
            min_r_ap_px=_getf(raw, "min_r_ap_px", 4.0),
            min_r_in_px=_getf(raw, "min_r_in_px", 12.0),
            min_r_out_px=_getf(raw, "min_r_out_px", 20.0),
            annulus_sigma_clip=_getf(raw, "annulus_sigma_clip", 3.0),
            fitsky_max_iter=_geti(raw, "fitsky_max_iter", 5),
            apcorr_apply=_as_bool(raw.get("apcorr_apply", "true"), True),
            apcorr_small_scale=_getf(raw, "apcorr_small_scale", 1.0),
            apcorr_large_scale=_getf(raw, "apcorr_large_scale", 3.0),
            apcorr_use_min_n=_geti(raw, "apcorr_use_min_n", 20),
            apcorr_scatter_max=_getf(raw, "apcorr_scatter_max", 0.05),
            phot_use_qc_pass_only=_as_bool(raw.get("phot_use_qc_pass_only", "false"), False),

            # Master ID editor
            search_radius_px=_getf(raw, "search_radius_px", 7.0),
            bulk_drop_box_px=_geti(raw, "bulk_drop_box_px", 200),
            gaia_add_max_sep_arcsec=_getf(raw, "gaia_add_max_sep_arcsec", 2.0),

            # REF build
            force_master_build=_as_bool(raw.get("force_master_build", "false"), False),
            master_min_frames_xy=_geti(raw, "master_min_frames_xy", 1),
            master_preserve_ids=_as_bool(raw.get("master_preserve_ids", "true"), True),
            ref_frame=raw.get("ref_frame", None),

            # Peak assist (detection supplement)
            peak_pass_enable=_as_bool(raw.get("peak_pass_enable", "true"), True),
            peak_nsigma=_getf(raw, "peak_nsigma", 3.2),
            peak_kernel_scales=raw.get("peak_kernel_scales", "0.9,1.3"),
            peak_min_sep_px=_getf(raw, "peak_min_sep_px", 4.0),
            peak_max_add=_geti(raw, "peak_max_add", 600),
            peak_max_elong=_getf(raw, "peak_max_elong", 1.6),
            peak_sharp_lo=_getf(raw, "peak_sharp_lo", 0.12),
            peak_skip_if_nsrc_ge=_geti(raw, "peak_skip_if_nsrc_ge", 4500),

            # Radial FWHM params
            fwhm_measure_max=_geti(raw, "fwhm_measure_max", 25),
            fwhm_dr=_getf(raw, "fwhm_dr", 0.5),

            # WCS solve (ASTAP)
            astap_exe=raw.get("astap_exe", "astap_cli.exe"),
            astap_timeout_s=_getf(raw, "astap_timeout_s", 120.0),
            astap_search_radius_deg=_getf(raw, "astap_search_radius_deg", 8.0),
            astap_database=raw.get("astap_database", "D50"),
            astap_fov_fudge=_getf(raw, "astap_fov_fudge", 1.0),
            astap_downsample_z=_geti(raw, "astap_downsample_z", 2),
            astap_max_stars_s=_geti(raw, "astap_max_stars_s", 500),
            wcs_require_qc_pass=_as_bool(raw.get("wcs_require_qc_pass", "true"), True),
            wcs_refine_enable=_as_bool(raw.get("wcs_refine_enable", "true"), True),
            wcs_refine_max_match=_geti(raw, "wcs_refine_max_match", 600),
            wcs_refine_match_r_fwhm=_getf(raw, "wcs_refine_match_r_fwhm", 1.6),
            wcs_refine_min_match=_geti(raw, "wcs_refine_min_match", 50),
            gaia_radius_fudge=_getf(raw, "gaia_radius_fudge", 1.35),
            gaia_mag_max=_getf(raw, "gaia_mag_max", 18.0),
            gaia_retry=_geti(raw, "gaia_retry", 2),
            gaia_backoff_s=_getf(raw, "gaia_backoff_s", 6.0),
            gaia_allow_no_cache=_as_bool(raw.get("gaia_allow_no_cache", "true"), True),

            # Aperture overlay (Step 12)
            inspect_index=_geti(raw, "inspect_index", 0),
            overlay_max_labels=_geti(raw, "overlay_max_labels", 2000),
            overlay_label_fontsize=_getf(raw, "overlay_label_fontsize", 6.0),
            overlay_label_offset_px=_getf(raw, "overlay_label_offset_px", 3.0),
            overlay_show_id_when_no_mag=_as_bool(raw.get("overlay_show_id_when_no_mag", "false"), False),
            overlay_use_phot_centroid=_as_bool(raw.get("overlay_use_phot_centroid", "true"), True),
            overlay_show_ref_pos=_as_bool(raw.get("overlay_show_ref_pos", "true"), True),
            overlay_show_shift_vectors=_as_bool(raw.get("overlay_show_shift_vectors", "false"), False),
            overlay_shift_max_vectors=_geti(raw, "overlay_shift_max_vectors", 300),
            overlay_shift_min_px=_getf(raw, "overlay_shift_min_px", 1.5),

            # CMD/analysis (Step 13)
            pixel_scale_arcsec=_as_float_or_none(raw.get("pixel_scale_arcsec", "")),
            match_tol_px=_getf(raw, "match_tol_px", 1.0),
            min_master_gaia_matches=_geti(raw, "min_master_gaia_matches", 10),
            cmd_snr_calib_min=_getf(raw, "cmd_snr_calib_min", 20.0),
            zp_clip_sigma=_getf(raw, "zp_clip_sigma", 3.0),
            zp_fit_iters=_geti(raw, "zp_fit_iters", 5),
            zp_slope_absmax=_getf(raw, "zp_slope_absmax", 1.0),
            frame_zp_min_n=_geti(raw, "frame_zp_min_n", 5),
            cmd_apply_extinction=_as_bool(raw.get("cmd_apply_extinction", "false"), False),
            cmd_extinction_mode=raw.get("cmd_extinction_mode", "absorb"),
            gaia_snr_calib_min=_getf(raw, "gaia_snr_calib_min", 20.0),
            gaia_gi_min=_getf(raw, "gaia_gi_min", -0.5),
            gaia_gi_max=_getf(raw, "gaia_gi_max", 4.5),
            gaia_zp_slope_absmax=_getf(raw, "gaia_zp_slope_absmax", 1.0),
            gaia_color_slope_absmax=_getf(raw, "gaia_color_slope_absmax", 2.0),

            # Isochrone (Step 14)
            iso_file_path=raw.get("iso_file_path", ""),
            iso_age_init=_getf(raw, "iso_age_init", 9.7),
            iso_mh_init=_getf(raw, "iso_mh_init", -0.1),
            iso_eg_r_init=_getf(raw, "iso_eg_r_init", 0.0033),
            iso_dm_init=_getf(raw, "iso_dm_init", 9.46),
        )

        # Store raw dict for compatibility
        P._raw = raw

        # Setup directory paths
        P.data_dir = Path(P.data_dir)
        P.result_dir = Path(P.result_dir) if P.result_dir else (P.data_dir / "result")
        P.result_dir.mkdir(parents=True, exist_ok=True)
        P.cache_dir = (P.result_dir / str(P.cache_dir))
        P.cache_dir.mkdir(parents=True, exist_ok=True)

        # FWHM seed (pixel-based legacy)
        pix_legacy = P.fwhm_pix_guess
        P.fwhm_seed_px = float(pix_legacy if pix_legacy is not None else 6.0)
        P._fwhm_seed_from = "pixel-legacy"

        return P

    @staticmethod
    def _compute_hash(path: Path) -> str:
        """Compute SHA1 hash of parameter file content (excluding comments)"""
        try:
            txt = Path(path).read_text(encoding="utf-8", errors="ignore")
            lines = []
            for ln in txt.splitlines():
                s = ln.strip()
                if (not s) or s.startswith("#"):
                    continue
                if "#" in s:
                    s = s.split("#", 1)[0].strip()
                lines.append(s)
            norm = "\n".join(lines).encode("utf-8")
            return hashlib.sha1(norm).hexdigest()
        except Exception:
            return "NO_PARAM"

    def save(self, path: Path):
        """Save current parameters to file"""
        path = Path(path)
        data = {"legacy": {}}

        for key, value in vars(self.P).items():
            if key.startswith("_"):
                continue
            if isinstance(value, Path):
                value = str(value)
            data["legacy"][key] = value

        if tomli_w is None:
            raise RuntimeError("tomli_w is required to write parameters.toml")
        with path.open("wb") as f:
            tomli_w.dump(data, f)

    def get(self, name: str, default=None):
        """Get parameter value with fallback to raw dict"""
        if hasattr(self.P, name):
            val = getattr(self.P, name)
            if not (val is None or (isinstance(val, str) and val.strip() == "")):
                return val
        if hasattr(self.P, "_raw") and name in self.P._raw:
            rawv = self.P._raw[name]
            if isinstance(default, bool):
                return _as_bool(rawv, default)
            if isinstance(default, int):
                fv = _as_float_or_none(rawv)
                return int(fv) if fv is not None else default
            if isinstance(default, float):
                fv = _as_float_or_none(rawv)
                return fv if fv is not None else default
            return rawv
        return default

    def save_toml(self, path: Path | str | None = None) -> bool:
        """Persist current parameters to TOML file while preserving structure."""
        if tomli_w is None:
            return False
        param_path = Path(path or getattr(self, "param_file", "parameters.toml"))
        try:
            with param_path.open("rb") as f:
                data = tomllib.load(f)
        except Exception:
            data = {}

        def to_toml_value(attr: str, value: Any, path_keys: Iterable[str]) -> Any:
            if isinstance(value, Path):
                if tuple(path_keys) == ("io", "cache_dir"):
                    return value.name
                return str(value)
            if attr == "peak_kernel_scales":
                if isinstance(value, str):
                    items = [v.strip() for v in value.split(",") if v.strip()]
                    out: list[Any] = []
                    for v in items:
                        try:
                            out.append(float(v))
                        except Exception:
                            out.append(v)
                    return out
                if isinstance(value, (list, tuple)):
                    return list(value)
            return value

        for path_keys, attr in TOML_KEY_MAP:
            if not hasattr(self.P, attr):
                continue
            val = getattr(self.P, attr)
            if val is None:
                continue
            _set_path(data, path_keys, to_toml_value(attr, val, path_keys))

        # Keep wcs_refine.enable in sync with legacy flag if present.
        if hasattr(self.P, "wcs_refine_enable"):
            _set_path(data, ("wcs_refine", "enable"), bool(getattr(self.P, "wcs_refine_enable")))

        try:
            with param_path.open("wb") as f:
                tomli_w.dump(data, f)
        except Exception:
            return False
        return True

    def print_summary(self):
        """Print parameter summary"""
        P = self.P
        print("\n==================== PARAM SUMMARY ====================")
        print(f"DATA_DIR      : {P.data_dir}")
        print(f"RESULT_DIR    : {P.result_dir}")
        print(f"CACHE_DIR     : {P.cache_dir}")
        print(f"resume_mode   : {P.resume_mode} | force_redetect={P.force_redetect} | force_rephot={P.force_rephot}")
        print(f"parallel_mode : {P.parallel_mode} | parallel_max_workers={P.parallel_max_workers}")
        print(f"FWHM seed     : {P.fwhm_seed_px:.2f} px (from={getattr(P, '_fwhm_seed_from', '?')})")
        print(f"FWHM range    : {P.fwhm_px_min:.2f} ~ {P.fwhm_px_max:.2f} px | elong_max={P.fwhm_elong_max} | iso_min_sep={P.iso_min_sep_pix}px")
        print(f"bkg2d detect  : {P.bkg2d_in_detect} | box={P.bkg2d_box}")
        print(f"detect_sigma  : base={P.detect_sigma} | g={P.detect_sigma_g} r={P.detect_sigma_r} i={P.detect_sigma_i}")
        print(f"deblend       : enable={P.deblend_enable} nthresh={P.deblend_nthresh} cont={P.deblend_cont} dilate={P.segm_dilate_radius_px}")
        print(f"clip          : sat_adu={P.saturation_adu}")
        print(f"camera        : gain={P.gain_e_per_adu} e-/ADU | rdnoise={P.rdnoise_e} e- | zp_init={P.zp_initial}")
        print("=======================================================\n")


def read_params(path: str | Path = "parameters.toml") -> Parameters:
    """
    Load parameters from file

    Args:
        path: Path to parameter file

    Returns:
        Parameters object
    """
    return Parameters(path)
