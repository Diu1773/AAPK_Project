# Technical Reference (AAPKI GUI)

## Scope and Conventions
This document describes the scientific and processing logic for the AAPKI GUI
pipeline (Steps 1-13) and tools (QA Report, Extinction, IRAF/DAOPHOT
Photometry).

Conventions used here:
- Parameter names are TOML keys from `parameters.toml`.
- Units are explicit in each section (px, arcsec, ADU, e-, mag, s, deg).
- FITS filter values are normalized to lowercase by the pipeline unless noted.
- Pixel coordinates in AAPKI outputs follow the photutils convention (0-based).
- If `result/cropped/` exists, the pipeline prefers cropped frames for
  detection/photometry/overlay.

## Global Definitions and Formulas

Pixel scale (arcsec per pixel):
- `pix_scale_arcsec = 206.265 * camera_pixel_um * binning / telescope_focal_mm`

FWHM conversions:
- `fwhm_px = fwhm_arcsec / pix_scale_arcsec`
- `fwhm_arcsec = fwhm_px * pix_scale_arcsec`

Aperture photometry (instrumental magnitude):
- `mag_inst = -2.5 * log10(flux_corr_adu / exptime_s) + zp_initial`
- `mag_err = 1.0857 / SNR` (1.0857 = 2.5 / ln(10))

Sky preview SNR (Step 3):
- `SNR = N_star / sqrt(N_star + N_sky + N_pix * RN^2)`
- `N_star` and `N_sky` are in electrons, `RN` is read noise in e-.

Forced photometry CCD error model (Step 9):
- `sigma_pix_e2` is the background variance per pixel (electrons^2).
- `var_source = max(flux_e, 0)`
- `var_bkg_in_ap = N_ap * sigma_pix_e2`
- `var_bkg_est = (N_ap^2 / n_sky) * sigma_pix_e2`
- `var_readnoise = N_ap * RN^2` (only if sky_sigma_includes_rn is true)
- `sigma_e = sqrt(var_source + var_bkg_in_ap + var_bkg_est + var_readnoise)`
- `SNR = flux_e / sigma_e`

Airmass (when not present in FITS header):
- Compute altitude from RA/Dec/time/site, then Kasten-Young approximation:
- `X = 1 / (cos(z) + 0.50572 * (96.07995 - z)^(-1.6364))`
- `z = 90 - altitude_deg`

Gaia to SDSS transforms (Jordi 2010, implemented coefficients):
- `g_sdss = G - (0.2199 - 0.6365*c - 0.1548*c^2 + 0.0064*c^3)`
- `r_sdss = G - (-0.09837 + 0.08592*c + 0.1907*c^2 - 0.1701*c^3 + 0.02263*c^4)`
- `i_sdss = G - (-0.293 + 0.6404*c - 0.09609*c^2 - 0.002104*c^3)`
- `c = BP - RP` with validity ranges:
  - g: 0.3 <= c <= 3.0
  - r: 0.0 <= c <= 3.0
  - i: 0.5 <= c <= 2.0

Calibration model (Step 11):
- `delta = sdss_ref - mag_inst`
- Fit `delta = ZP + CT * color` with sigma clipping.
- `mag_cal = mag_inst + zp_frame + color_term (+ k * airmass if two_step)`

Aperture correction (Step 9):
- `apcorr = median(flux_large / flux_small)`
- Apply if `n_used >= min_n` and `rel_scatter <= scatter_max`.

## Workflow Steps

### Step 1 - File Selection and Header Scan
Purpose: scan FITS files, verify headers, and initialize project state.

Method:
- Scan `io.data_dir` for FITS files matching `io.filename_prefix`.
- Build header summary and store state in `result/project_state.json`.

Outputs:
- `result/headers.csv`
- `result/project_state.json`

Parameters:
- `io.data_dir` (path)
- `io.filename_prefix` (string)

### Step 2 - Image Crop
Purpose: define a crop region and apply it to all frames.

Method:
- User draws a rectangle on a reference image.
- Crop is applied to all frames and saved into `result/cropped/` with the same
  filenames (no added prefix).
- FITS headers are updated with `CROPPED`, `CROP_X0`, `CROP_Y0`, `CROP_X1`,
  `CROP_Y1`.

Outputs:
- `result/cropped/*.fit*`
- `result/crop_rect.json`

Parameters:
- None (interactive only)

### Step 3 - Sky Preview and QC
Purpose: interactive per-frame inspection with imexamine-style tools.

Method:
- Cursor measurement uses DAOStarFinder to refine centroid and radial profile
  to estimate FWHM (for display).
- Aperture/annulus sizes use the FWHM seed, not the measured FWHM.
- SNR and magnitude are computed using the preview equation in the
  Global Definitions section.

Outputs:
- None (interactive only)

Parameters:
- `hud5x.aperture_scale` (x FWHM)
- `hud5x.center_cbox_scale` (x FWHM)
- `hud5x.annulus_in_scale` (x FWHM)
- `hud5x.annulus_out_scale` (x FWHM, width)
- `hud5x.min_r_ap_px` (px)
- `hud5x.min_r_in_px` (px)
- `hud5x.min_r_out_px` (px)
- `hud5x.sigma_clip` (sigma)
- `hud5x.mag_flux` (rate_e or flux_e)
- `hud5x.use_header_exptime` (bool)
- `fwhm.guess_arcsec` (arcsec)
- `instrument.gain_e_per_adu` (e-/ADU)
- `instrument.rdnoise_e` (e-)
- `instrument.zp_initial` (mag)

### Step 4 - Source Detection
Purpose: detect star candidates and estimate per-frame FWHM.

Method summary:
- Optional 2D background subtraction (downsampled median filter).
- Gaussian smoothing with sigma = fwhm_seed / 2.355.
- Threshold = median + nsigma * std from sigma-clipped stats.
- Segmentation via `detect_sources`, optional deblending with
  `deblend_sources`.
- Filter by elongation, saturation, and isolation distance.
- Optional DAOStarFinder refinement and peak-assist pass.
- FWHM per frame is the median of radial-profile FWHM estimates from bright
  sources.

Outputs:
- `result/cache/detect_<frame>.csv` (x,y)
- `result/cache/detect_<frame>.json` (metadata)
- `result/cache/detect_peak_<frame>.csv` (peak-assist additions)

Parameters:
- `detection.engine` (segm or peak)
- `detection.sigma` (sigma threshold)
- `detection.sigma_g`, `detection.sigma_r`, `detection.sigma_i` (per filter)
- `detection.minarea_pix` (px)
- `detection.keep_max` (count)
- `detection.dilate_radius_px` (px)
- `detection.deblend.*` (see Parameter Reference)
- `detection.peak.*` (see Parameter Reference)
- `detection.dao.*` (see Parameter Reference)
- `background.in_detect` (bool)
- `background.box`, `background.downsample` (px)
- `fwhm.elong_max`, `fwhm.iso_min_sep_pix` (ratio, px)
- `photometry.radii.*` (annulus settings used for FWHM)
- `instrument.saturation_adu` (ADU)
- `parallel.max_workers` (threads)

### Step 5 - WCS Plate Solving
Purpose: solve WCS per frame and refine CRPIX alignment with Gaia.

Method summary:
- Run ASTAP CLI with estimated FOV and search radius.
- Load WCS from FITS header or `.wcs` sidecar if present.
- Optional refinement: match detection XY to Gaia-projected XY, compute median
  dx/dy, shift CRPIX1/2, and report residuals in arcsec.

Outputs:
- `result/wcs_solve_summary.csv`
- `result/cache/wcs_solve/wcs_<frame>.json`
- `.wcs` sidecar files from ASTAP (if generated)

Parameters:
- `wcs.do_plate_solve` (bool)
- `wcs.astap_exe` (path)
- `wcs.astap_timeout_s` (s)
- `wcs.astap_search_radius_deg` (deg)
- `wcs.astap_fov_fudge` (scale)
- `wcs.astap_downsample` (int)
- `wcs.astap_max_stars` (int)
- `wcs.max_workers` (int)
- `wcs.require_qc_pass` (bool)
- `wcs.force_solve` (bool)
- `wcs_refine.enable` (bool)
- `wcs_refine.max_match` (int)
- `wcs_refine.match_r_fwhm` (x FWHM)
- `wcs_refine.min_match` (int)
- `gaia.radius_fudge`, `gaia.mag_max`, `gaia.retry`, `gaia.backoff_s`
- `legacy.platesolve_gaia_radius_scale` (scale)

### Step 6 - Star ID Matching
Purpose: assign Gaia source_id to detections using WCS.

Method summary:
- Use `gaia_fov.ecsv` from Step 5.
- Convert detection XY to sky coordinates using WCS.
- Match to Gaia with `match_to_catalog_sky` within tolerance.
- Write per-frame idmatch CSV and summary.

Outputs:
- `result/cache/idmatch/idmatch_<frame>.csv`
- `result/cache/idmatch/idmatch_<frame>.json`
- `result/idmatch_summary.csv`
- `result/master_star_ids.csv`

Parameters:
- `idmatch.tol_px` (px)
- `idmatch.tol_arcsec` (arcsec, overrides px)
- `idmatch.gaia_g_limit` or `gaia.g_limit` (mag)
- `idmatch.force` (bool)
- `parallel.resume_mode` (bool)
- `legacy.platesolve_gaia_radius_scale` (scale)
- `target.ra_deg`, `target.dec_deg` (deg, for Gaia requery if needed)

### Step 7 - Reference Build
Purpose: build the master catalog and ID mapping.

Method summary:
- Merge idmatch results across frames.
- Select reference frame (forced or best match count).
- Generate `master_catalog.tsv`, `sourceid_to_ID.csv`, and
  `frame_sourceid_to_ID.tsv`.

Outputs:
- `result/master_catalog.tsv`
- `result/sourceid_to_ID.csv`
- `result/frame_sourceid_to_ID.tsv`
- `result/master_ref_build_debug.json`
- `result/master_missing_in_ref.csv`
- `result/master_star_ids.orig.csv`

Parameters:
- `master.min_frames_xy` (count)
- `master.preserve_ids` (bool)
- `master.force_build` (bool)
- `legacy.ref_frame` (string or index)

### Step 8 - Master ID Editor
Purpose: manually add/remove master IDs on the reference frame.

Method summary:
- Load master catalog and GAIA overlay.
- Add/remove IDs interactively and save a curated list.

Outputs:
- `result/master_star_ids.csv`

Parameters:
- `master_editor.search_radius_px` (px)
- `master_editor.bulk_drop_box_px` (px)
- `master_editor.gaia_add_max_sep_arcsec` (arcsec)

### Step 9 - Forced Photometry
Purpose: compute per-frame photometry at fixed master IDs.

Method summary:
- Build per-frame apertures from FWHM (from detection cache).
- Optional aperture correction from bright sources.
- Forced photometry at each ID, with optional centroid refinement.
- Error propagation follows the CCD equation in Global Definitions.

Aperture sizing:
- `r_ap = max(aperture_scale * fwhm_used, min_r_ap_px)`
- `r_in = max(annulus_scale * fwhm_used, min_r_in_px, r_ap + annulus_min_gap_px)`
- `r_out = max(r_in + dannulus_scale * fwhm_used, min_r_out_px, r_in + annulus_min_width_px)`
- `cbox = max(center_cbox_scale * fwhm_used, 5 px)`

Outputs:
- `result/aperture_by_frame.csv`
- `result/apcorr_summary.csv`
- `result/<frame>_photometry.tsv`
- `result/photometry_index.csv`
- `result/phot_forced_debug.json`
- `result/phot_forced_fail.tsv`

Parameters:
- `photometry.mode` (apcorr or fixed)
- `photometry.recenter` (bool)
- `photometry.min_snr_for_mag` (SNR)
- `photometry.use_qc_pass_only` (bool)
- `photometry.scales.*` (x FWHM, px)
- `photometry.radii.*` (px, sigma)
- `photometry.apcorr.*` (apcorr settings)
- `instrument.gain_e_per_adu` (e-/ADU)
- `instrument.rdnoise_e` (e-)
- `instrument.saturation_adu` (ADU)
- `parallel.resume_mode`, `parallel.force_rephot` (bool)
- `legacy.max_recenter_shift` (px)
- `legacy.centroid_outlier_px` (px)
- `legacy.sky_sigma_mode` (local, frame, max)
- `legacy.sky_sigma_includes_rn` (bool)
- `legacy.sky_sigma_min_n_sky` (count)

### Step 10 - Aperture Overlay
Purpose: verify apertures and centroids visually on a frame.

Method summary:
- Load `aperture_by_frame.csv` and `master_catalog.tsv`.
- Overlay apertures/annuli and optional shift vectors.

Outputs:
- None (interactive only)

Parameters:
- `overlay.max_labels` (count)
- `overlay.label_fontsize` (pt)
- `overlay.label_offset_px` (px)
- `overlay.show_id_when_no_mag` (bool)
- `overlay.use_phot_centroid` (bool)
- `overlay.show_ref_pos` (bool)
- `overlay.show_shift_vectors` (bool)
- `overlay.shift_max_vectors` (count)
- `overlay.shift_min_px` (px)
- `overlay.inspect_index` (int)

### Step 11 - Zeropoint Calibration
Purpose: calibrate to SDSS, compute per-frame zeropoints, and build CMD tables.

Method summary:
- Read `*_photometry.tsv` and compute per-ID medians.
- Transform Gaia G/BP/RP to SDSS g/r/i using Jordi coefficients.
- Fit `delta = ZP + CT * color` with sigma clipping and slope limits.
- Compute per-frame ZP from median delta values.
- Optionally include extinction term from the Extinction tool.

Outputs:
- `result/median_by_ID_filter_raw.csv`
- `result/median_by_ID_filter_wide_raw.csv`
- `result/gaia_sdss_calibrator_by_ID.csv`
- `result/frame_zeropoint_cut_summary.csv`
- `result/frame_zeropoint.csv`
- `result/frame_zeropoint_rejects.csv`
- `result/median_by_ID_filter.csv`
- `result/median_by_ID_filter_wide.csv`
- `result/median_by_ID_filter_wide_cmd.csv`
- `result/frame_airmass.csv`

Parameters:
- `cmd.snr_calib_min` (SNR)
- `cmd.max_sources` (count)
- `cmd.zp.clip_sigma` (sigma)
- `cmd.zp.fit_iters` (iterations)
- `cmd.zp.slope_absmax` (abs slope limit)
- `cmd.color.clip_sigma` (sigma, reserved)
- `cmd.color.fit_iters` (reserved)
- `cmd.color.slope_absmax` (reserved)
- `match.min_gaia_matches` (count)
- `gaia.gi_min`, `gaia.gi_max` (color range, reserved in UI)
- `legacy.frame_zp_min_n` (count)
- `legacy.cmd_apply_extinction` (bool)
- `legacy.cmd_extinction_mode` (absorb or two_step)

### Step 12 - CMD Plot
Purpose: open the CMD viewer using calibrated tables.

Method summary:
- Load `median_by_ID_filter_wide_cmd.csv` (or fallback to wide table).
- The CMD viewer is embedded automatically when Step 12 is opened.

Outputs:
- None (interactive only)

Parameters:
- None

### Step 13 - Isochrone Model
Purpose: fit isochrone parameters to the CMD and compute memberships.

Method summary:
- Uses IsochroneFitterV2 with differential evolution (fast) and optional
  Hessian mode for uncertainties.
- Nearest isochrone grid is used (no bilinear interpolation).
- Distance is computed via KD-tree with per-point error normalization.
- A robust fraction of closest points (fit_fraction) is used in chi2.
- Extinction: `E(B-V) = E(g-r) / (R_G - R_R)` with R_G=3.303, R_R=2.285.

Outputs:
- `result/isochrone_fit_result.txt`
- `result/isochrone_fit_result.json`
- `result/cmd_with_membership.csv` (after membership computation)

Parameters:
- `isochrone.file_path` (path)
- `isochrone.age_init` (log(age) years)
- `isochrone.mh_init` ([M/H])
- `isochrone.eg_r_init` (mag)
- `isochrone.dm_init` (mag)
- `legacy.iso_col_mh`, `legacy.iso_col_age`, `legacy.iso_col_g`, `legacy.iso_col_r`
- `legacy.iso_fit_fraction` (0-1)

## Tools

### QA Report Tool
Purpose: validate photometry quality and error model performance.

Method summary:
- Error model: per-star RMS vs predicted mag_err, chi2/nu, per-filter summary.
- Centroid QA: delta_x, delta_y, delta_r per measurement.
- Frame quality: SNR, good-mag fraction, optional FWHM-based flagging.
- Background QA: median, std, n_sky per frame and per filter.
- Optional ZP-based error model uses `mag_err_eff = sqrt(mag_err^2 + zp_sigma^2)`.

Outputs (under `result/qa_report/`):
- `qa_error_model_by_star*.csv`, `qa_error_model_by_filter*.csv`
- `qa_error_model_summary*.json`
- `qa_centroid_shift.csv`, `qa_centroid_summary.json`
- `qa_frame_quality.csv`
- `qa_background.csv`
- `qa_error_model_plot*.png`, `qa_centroid_plot.png`, `qa_background_plot.png`

Parameters (`tools.qa_report`):
- `min_n_frames` (count)
- `min_snr` (SNR)
- `max_chi2_nu` (unitless)
- `max_delta_r` (px)
- `exclude_saturated` (bool)
- `error_model_source` (raw or zp)
- `frame_flag_mode` (absolute or percentile)
- `frame_snr_min` (SNR)
- `frame_goodmag_min` (fraction)
- `frame_fwhm_mode` (scale or absolute)
- `frame_fwhm_scale` (x median)
- `frame_fwhm_abs` (px)
- `enabled_filters_all` (bool)
- `enabled_filters` (list of filter strings)

### Extinction (Airmass Fit) Tool
Purpose: fit per-filter extinction coefficients using instrumental mags.

Method summary:
- Build `frame_airmass.csv` from FITS header or computed airmass.
- Compute Gaia->SDSS reference mags and color terms (same as Step 11).
- Fit `delta = ref_mag - (mag_inst + color_term)` vs airmass using robust
  sigma-clipped linear regression per filter.

Outputs (under `result/extinction/`):
- `extinction_fit_filter_stats.csv`
- `extinction_fit_by_filter.csv`
- `extinction_fit_points.csv`
- `extinction_fit_plot.png`

Parameters:
- `cmd.zp.clip_sigma` (sigma)
- `cmd.zp.fit_iters` (iterations)
- `cmd.snr_calib_min` (SNR)
- `match.min_gaia_matches` (count)
- `site.*` (lat/lon/alt/tz for computed airmass)

### IRAF/DAOPHOT Photometry Tool
Purpose: run IRAF DAOPHOT photometry using PyRAF in WSL.

Method summary:
- Generates a per-run PyRAF script and executes it via WSL.
- FWHM is read from FITS headers (`FWHMARC`, `SEEING`, `FWHM_AS`, `FWHMPSF`,
  `FWHM_PIX`) or from filter-specific seeing defaults.
- If auto-sigma is enabled, sigma is estimated by `iraf.imstat`.
- Threshold scaling: `threshold = base_threshold * clip(sigma / sigma_ref, 0.8, 1.6)`.
- Aperture/annulus/cbox sizes are FWHM multipliers.
- Script `_pyraf_photometry.py` is auto-removed after completion.

Outputs (default `result/iraf_phot/`):
- `<frame>.coo`, `<frame>.mag`, `<frame>.txt`
- `_pyraf_photometry.py` (temporary, auto-cleaned)

Parameters:
- `tools.iraf.params` (auto-saved IRAF parameter set)
- `tools.iraf.filters.<filter>` (per-filter overrides)
- `tools.iraf.filter_aliases` (header FILTER normalization)

## Parameter Reference (TOML)

### [io]
- `data_dir` (path) root directory of FITS files.
- `filename_prefix` (string) prefix filter for input files.
- `result_dir` (path) output directory; empty means `data_dir/result`.
- `cache_dir` (string) cache folder under result_dir.

### [target]
- `name` (string) target label.
- `ra_deg` (deg) target RA (optional).
- `dec_deg` (deg) target Dec (optional).

### [parallel]
- `mode` (string) thread, process, auto, none.
- `max_workers` (int) 0 means auto.
- `resume_mode` (bool) skip cached outputs.
- `force_redetect` (bool) force detection re-run.
- `force_rephot` (bool) force photometry re-run.
- `detect_cache_strategy` (string) mtime, hash, none.

### [ui]
- `log_tail` (int) log lines to keep.
- `detect_progress_bar` (bool) show detection progress.
- `canvas_px` (int) canvas size.

### [instrument]
- `telescope_focal_mm` (mm) telescope focal length.
- `camera_pixel_um` (um) pixel size.
- `binning` (int) binning factor.
- `gain_e_per_adu` (e-/ADU) gain.
- `rdnoise_e` (e-) read noise.
- `saturation_adu` (ADU) saturation threshold.
- `zp_initial` (mag) initial ZP for inst magnitudes.
- `datamin_adu` (ADU) min valid value (optional).
- `datamax_adu` (ADU) max valid value (optional).

### [alignment] (reserved)
- `ref_index` (int) reference frame index.
- `global_align` (bool) cross-filter alignment.
- `global_ref_filter` (string) filter label.
- `global_ref_index` (int) frame index.

### [fwhm]
- `guess_arcsec` (arcsec) FWHM seed.
- `guess_px` (px) FWHM seed override.
- `px_min`, `px_max` (px) FWHM bounds.
- `arcsec_min`, `arcsec_max` (arcsec) FWHM bounds.
- `elong_max` (ratio) max elongation.
- `qc_max_sources` (count) max sources used for FWHM QC.
- `iso_min_sep_pix` (px) min separation for isolated sources.
- `measure_max` (px) max radius for radial profile.
- `dr` (px) radial bin step.

### [clip] (reserved)
- `min_adu` (ADU) lower clip.
- `max_adu` (ADU) upper clip.

### [detection]
- `engine` (string) segm or peak.
- `sigma` (float) detection threshold in sigma.
- `sigma_g`, `sigma_r`, `sigma_i` (float) per-filter thresholds.
- `minarea_pix` (px) minimum region area.
- `keep_max` (count) maximum detections kept.
- `dilate_radius_px` (px) segmentation dilation.

### [detection.deblend]
- `enable` (bool) enable deblending.
- `nthresh` (int) deblend levels.
- `contrast` (float) deblend contrast.
- `max_labels` (count) soft deblend limit.
- `label_hard_max` (count) hard deblend limit.
- `nlevels_soft` (int) soft levels for crowded fields.
- `contrast_soft` (float) soft contrast for crowded fields.

### [detection.peak]
- `enable` (bool) peak assist on/off.
- `nsigma` (float) peak threshold in sigma.
- `kernel_scales` (list) kernel scales relative to FWHM.
- `min_sep_px` (px) min separation.
- `max_add` (count) max added peaks.
- `max_elong` (ratio) max elongation.
- `sharp_lo` (float) min sharpness.
- `skip_if_nsrc_ge` (count) skip peak pass if detections exceed this.

### [detection.dao]
- `enable` (bool) DAO refine on/off.
- `fwhm_px` (px) DAO FWHM.
- `sharp_lo`, `sharp_hi` (float) sharpness bounds.
- `round_lo`, `round_hi` (float) roundness bounds.
- `match_tol_px` (px) DAO match tolerance.

### [background]
- `enable` (bool) 2D background enable.
- `in_detect` (bool) apply background in detection.
- `box` (px) background box size.
- `filter_size` (px) filter size.
- `edge_method` (string) pad, crop, wrap, extend.
- `method` (string) median or mean.
- `downsample` (int) downsample factor.

### [qc] (reserved)
- `gate_enable` (bool) QC gating enable.
- `sky_sigma_max_e` (e-) max sky sigma for QC.
- `nsrc_min` (count) minimum sources for QC.
- `keep_positions_if_fail` (bool) keep positions if QC fails.

### [photometry]
- `mode` (string) apcorr or fixed.
- `recenter` (bool) centroid refinement.
- `use_segm_mask` (bool) reserved; segmentation masking is not applied yet.
- `min_snr_for_mag` (SNR) minimum for mag output.
- `use_qc_pass_only` (bool) restrict to QC-passed frames.

### [photometry.scales]
- `aperture_scale` (x FWHM).
- `annulus_scale` (x FWHM).
- `dannulus_scale` (x FWHM).
- `center_cbox_scale` (x FWHM).
- `annulus_min_gap_px` (px).
- `annulus_min_width_px` (px).

### [photometry.radii]
- `min_r_ap_px` (px).
- `min_r_in_px` (px).
- `min_r_out_px` (px).
- `sigma_clip` (sigma) sky clip.
- `max_iter` (int) clip iterations.
- `neighbor_mask_scale` (x FWHM) reserved for masking.

### [photometry.apcorr]
- `apply` (bool).
- `small_scale` (x FWHM).
- `large_scale` (x FWHM).
- `min_n` (count).
- `scatter_max` (fraction).
- `min_snr` (SNR, reserved).

### [hud5x]
- `aperture_scale` (x FWHM).
- `center_cbox_scale` (x FWHM).
- `annulus_in_scale` (x FWHM).
- `annulus_out_scale` (x FWHM width).
- `sigma_clip` (sigma).
- `neighbor_mask_scale` (x FWHM).
- `mag_flux` (rate_e or flux_e).
- `use_header_exptime` (bool).
- `min_r_ap_px`, `min_r_in_px`, `min_r_out_px` (px).

### [wcs]
- `do_plate_solve` (bool).
- `astap_exe` (path).
- `astap_timeout_s` (s).
- `astap_search_radius_deg` (deg).
- `astap_fov_fudge` (scale).
- `astap_downsample` (int).
- `astap_max_stars` (int).
- `max_workers` (int).
- `require_qc_pass` (bool).
- `force_solve` (bool).

### [wcs_refine]
- `enable` (bool).
- `max_match` (count).
- `match_r_fwhm` (x FWHM).
- `min_match` (count).
- `max_sep_arcsec` (arcsec, reserved).
- `require` (bool, reserved).

### [gaia]
- `radius_fudge` (scale) query radius multiplier.
- `mag_max` (mag) Gaia magnitude limit.
- `g_limit` (mag) alternate g limit.
- `retry` (int) query attempts.
- `backoff_s` (s) delay between retries.
- `allow_no_cache` (bool).
- `snr_calib_min` (SNR) calibration SNR.
- `gi_min`, `gi_max` (mag) color limits for Gaia-based filtering (UI only).

### [idmatch]
- `tol_px` (px) match tolerance.
- `tol_arcsec` (arcsec) match tolerance override.
- `force` (bool) force re-match.
- `use_qc_pass_only` (bool, reserved).
- `gaia_g_limit` (mag) Gaia g limit (optional).

### [master]
- `n_master` (count, reserved).
- `min_frames_xy` (count).
- `preserve_ids` (bool).
- `force_build` (bool).
- `iso_min_sep_pix` (px, reserved).
- `keep_max` (count, reserved).
- `flux_quantile` (fraction, reserved).
- `filter_keep` (string, reserved).
- `ref_frame` (string or index, reserved; use legacy.ref_frame for now).

### [master_editor]
- `search_radius_px` (px).
- `bulk_drop_box_px` (px).
- `gaia_add_max_sep_arcsec` (arcsec).

### [match]
- `tol_px` (px) generic match tolerance.
- `wcs_radius_arcsec` (arcsec, reserved).
- `min_gaia_matches` (count).

### [cmd]
- `snr_calib_min` (SNR).
- `max_sources` (count).

### [cmd.zp]
- `clip_sigma` (sigma).
- `fit_iters` (int).
- `slope_absmax` (abs slope limit).
- `gaia_slope_absmax` (reserved).

### [cmd.color]
- `clip_sigma` (sigma, reserved).
- `fit_iters` (int, reserved).
- `slope_absmax` (abs slope limit, reserved).

### [isochrone]
- `file_path` (path) isochrone file.
- `age_init` (log years).
- `mh_init` ([M/H]).
- `eg_r_init` (mag).
- `dm_init` (mag).
- `col_mh`, `col_age`, `col_g`, `col_r` (int) column indices (not wired; use legacy keys).
- `fit_fraction` (0-1, not wired; use legacy keys).

### [overlay]
- `max_labels` (count).
- `label_fontsize` (pt).
- `label_offset_px` (px).
- `show_id_when_no_mag` (bool).
- `use_phot_centroid` (bool).
- `show_ref_pos` (bool).
- `show_shift_vectors` (bool).
- `shift_max_vectors` (count).
- `shift_min_px` (px).
- `inspect_index` (int).

### [transform]
- `save_src2ref` (bool, reserved).

### [site]
- `lat_deg` (deg).
- `lon_deg` (deg).
- `alt_m` (m).
- `tz_offset_hours` (hours).

### [tools.qa_report]
- `min_n_frames` (count).
- `min_snr` (SNR).
- `max_chi2_nu` (unitless).
- `max_delta_r` (px).
- `exclude_saturated` (bool).
- `error_model_source` (raw or zp).
- `frame_flag_mode` (absolute or percentile).
- `frame_snr_min` (SNR).
- `frame_goodmag_min` (fraction).
- `frame_fwhm_mode` (scale or absolute).
- `frame_fwhm_scale` (x median).
- `frame_fwhm_abs` (px).
- `enabled_filters_all` (bool).
- `enabled_filters` (list).

### [tools.iraf]
- `filter_aliases` (dict) map header FILTER values to normalized keys.
- `filters.<filter>` (dict) per-filter overrides; any key listed in the
  `tools.iraf.params` section can be overridden.
- `params` (dict) auto-saved IRAF parameter set from the GUI.

### [tools.iraf.params] (auto-saved)
- `scale` (units per pixel, IRAF DATAPARS scale)
- `emission` (bool)
- `datamax` (ADU)
- `noise` (string: poisson, constant, file)
- `readnoise` (e-)
- `epadu` (e-/ADU)
- `exposure` (header keyword string)
- `itime` (s, fallback if header missing)
- `seeing_g`, `seeing_r`, `seeing_i`, `seeing_default` (arcsec)
- `sigma_g`, `sigma_r`, `sigma_i`, `sigma_default` (ADU)
- `threshold_g`, `threshold_r`, `threshold_i`, `threshold_default` (sigma)
- `nsigma` (sigma)
- `ratio` (float)
- `theta` (deg)
- `sharplo_g`, `sharplo_r`, `sharplo_i`, `sharplo_default` (float)
- `sharphi` (float)
- `roundlo`, `roundhi` (float)
- `datamin_g`, `datamin_r`, `datamin_i`, `datamin_default` (ADU)
- `calgorithm` (string)
- `cbox_mult` (x FWHM)
- `cthreshold` (sigma)
- `minsnratio` (float)
- `cmaxiter` (int)
- `maxshift` (scale units)
- `clean` (bool)
- `rclean`, `rclip` (scale units)
- `kclean` (sigma)
- `salgorithm` (string)
- `annulus_mult` (x FWHM)
- `dannulus_mult` (x FWHM)
- `skyvalue` (ADU)
- `smaxiter` (int)
- `sloclip`, `shiclip` (sigma)
- `snreject` (int)
- `sloreject`, `shireject` (sigma)
- `khist` (sigma)
- `binsize` (sigma)
- `smooth` (bool)
- `rgrow` (scale units)
- `aperture_mult` (x FWHM)
- `zmag` (mag)
- `mkapert` (bool)
- `pix_scale` (arcsec/px)
- `sigma_ref` (ADU, used for threshold scaling)

### [legacy] (advanced/compat)
- `max_recenter_shift` (px) recenter cap for Step 9.
- `centroid_outlier_px` (px) centroid outlier flag.
- `sky_sigma_mode` (local, frame, max) background sigma source.
- `sky_sigma_includes_rn` (bool) treat sky sigma as including read noise.
- `sky_sigma_min_n_sky` (count) min annulus pixels for local sigma.
- `platesolve_gaia_radius_scale` (scale) Gaia query radius multiplier.
- `apcorr_min_snr` (SNR, reserved).
- `frame_zp_min_n` (count) min refs per frame for ZP.
- `cmd_apply_extinction` (bool) apply extinction in Step 11.
- `cmd_extinction_mode` (absorb or two_step).
- `gaia_zp_slope_absmax` (float, UI only).
- `gaia_color_slope_absmax` (float, UI only).
- `ref_frame` (string or index) force reference frame.
- `iso_col_mh`, `iso_col_age`, `iso_col_g`, `iso_col_r` (int) isochrone columns.
- `iso_fit_fraction` (0-1) isochrone fit fraction.
