# User Guide

## Overview
AAPKI Light Curve Toolkit is a GUI scaffold for building target light curves while
reusing the AAPKI detection/photometry pipeline. The light curve GUI exposes three
modes: variable stars, eclipsing targets, and asteroids. The underlying photometry
workflow remains available for generating per-frame results used by the light curve
modules. For full algorithm details and a complete parameter reference, see
TECHNICAL_REFERENCE.md.

## Installation
1) Install Python dependencies:
   `pip install -r requirements.txt`
2) Install ASTAP if you plan to use WCS solving later.
   - Download from: https://www.hnsky.org/astap.htm

## Configuration
Create `parameters.toml` in the repo root:

- Copy example:
  - Windows CMD: `copy parameters_example.toml parameters.toml`
  - PowerShell/Git Bash: `cp parameters_example.toml parameters.toml`

- Minimum required fields:
  - `data_dir`: path to FITS data directory.
  - `filename_prefix`: FITS filename prefix.
  - `rdnoise_e`: camera read noise in electrons.
  - `gain_e_per_adu`: camera gain in e-/ADU.

Tool parameters (QA report and IRAF) are persisted back to `parameters.toml`.

## Run the GUI
From the project root:
- `python main.py`

## Light Curve Tabs
- Variable Star: light curves for periodic or irregular variability.
- Eclipsing: eclipse light curves with phase folding support.
- Asteroid: light curves with motion-aware extraction.

## Workflow Steps

### Step 1 - File Selection
- Choose `data_dir` and `filename_prefix`.
- If the root contains subfolders, enable multi-night and pick a subfolder.
- Rescan headers, inspect FITS metadata, set a reference frame if needed.
- Outputs: `result/headers.csv`, `result/project_state.json`.

### Step 2 - Image Crop
- Draw a crop box and apply it to all frames.
- Outputs: `result/crop_rect.json`, cropped FITS in `result/cropped/`.

### Step 3 - Sky Preview and QC
- Interactive per-frame inspection with cursor tools.
- Shortcuts: `m` (measure), `h` (histogram), `g` (radial profile), `.` (next filter), `[`/`]` (prev/next frame).
- FWHM displayed as arcsec with pixel equivalent.

### Step 4 - Source Detection
- Multi-threaded segmentation + peak finding.
- Outputs per-frame source lists and detection summary.
- Key parameters: `detect_sigma`, `minarea_pix`, `dao_fwhm_px`, deblend settings.

### Step 5 - WCS Plate Solving
- Solves WCS, refines astrometry, records match quality.
- Outputs: `result/wcs_solve_summary.csv` and WCS metadata per frame.

### Step 6 - Star ID Matching
- Cross-matches detections to Gaia sources and produces per-frame ID maps.
- Outputs under `result/cache/idmatch/` and summary tables.

### Step 7 - Reference Build
- Builds `master_catalog.tsv` and ID maps from matched frames.
- Ref frame can be auto-selected or forced.
- Outputs: `result/master_catalog.tsv`, `result/sourceid_to_ID.csv`, debug logs.

### Step 8 - Target/Comparison Selection
- REF 프레임에서 대상 1개와 비교성 여러 개를 선택.
- SIMBAD/Gaia 기반 자동 추천 지원.
- Outputs: `result/master_star_ids.csv`, `result/target_selection.json`.

### Step 9 - Forced Photometry
- 선택된 대상/비교성만 강제측광 수행.
- Aperture/annulus scales and sky clipping parameters 적용.
- Outputs: `*_photometry.tsv`, `result/photometry_index.csv`, `result/phot_forced_debug.json`.

### Step 10 - Aperture Overlay
- Visual overlay of apertures/annuli on frames for verification.
- Loads `aperture_by_frame.csv` and master catalog.

### Step 11 - Zeropoint Calibration
- Computes Gaia -> SDSS transforms, color terms, and per-frame ZP.
- Outputs: `gaia_sdss_calibrator_by_ID.csv`, `frame_zeropoint.csv`,
  `median_by_ID_filter(_raw|_wide|_wide_cmd).csv`.

### Step 12 - Light Curve Builder
- 대상/비교성 라이트커브(차등/절대) 생성.
- 멀티나잇은 날짜별 result_dir을 추가해 병합.
- Outputs: `result/lightcurve_ID*_diff.csv`, `result/lightcurve_ID*_abs.csv`.

## Tools (Top Menu)

### QA Reports
- Independent from the main workflow; reads photometry outputs.
- Generates background, centroid, error-model, and frame-quality summaries.
- Can compare raw vs ZP error model validation.
- Outputs under `result/qa_report/`.

### Extinction (Airmass Fit)
- Diagnostic fit of extinction coefficient k per filter using instrumental mags.
- Uses airmass from FITS headers or computed from DATE-OBS + site/RA/DEC.
- Outputs under `result/extinction/` and `result/frame_airmass.csv`.

### IRAF/DAOPHOT Photometry
- Runs PyRAF in WSL using IRAF DAOPHOT tasks.
- Outputs under `result/iraf_phot/` (per-frame `.coo`, `.mag`, `.txt`).

## Key Calculation Notes

For detailed formulas and parameter definitions, see TECHNICAL_REFERENCE.md.

### Airmass (when missing in FITS header)
- Uses DATE-OBS/TIME-OBS + site location + RA/DEC.
- Kasten & Young (1989) style approximation (not simple sec(z)).

### Zeropoint Calibration
- Per-frame ZP computed from Gaia-calibrated reference stars.
- Default mode absorbs transparency/extinction changes into ZP.
- Optional extinction mode applies k*X if provided.

### Error Model QA
- Raw (instrumental) error model validation is the physical baseline.
- ZP-corrected QA compares reduced scatter after frame-level offsets are removed.
- ZP QA uses: `sigma_after^2 = sigma_meas^2 + sigma_ZP^2` where `sigma_ZP ~= zp_scatter / sqrt(n_ref)`.

### Systematic Floor
- Estimated from bright-star RMS in QA and reported; not forced into `mag_err`.
- Add as `sqrt(sigma_meas^2 + sigma_sys^2)` only if required by analysis.

## Output Structure
Under `<data_dir>/result/`:
- `headers.csv`: FITS header summary.
- `project_state.json`: workflow state.
- `crop_rect.json`: crop metadata.
- `photometry_index.csv`: per-frame photometry index.
- `frame_zeropoint.csv`: per-frame ZP and scatter.
- `frame_airmass.csv`: airmass + metadata per frame.
- `cache/`: detection, WCS, and ID match caches.
- `qa_report/`: QA tables, plots, summaries.
- `extinction/`: extinction fit tables and plots.
- `iraf_phot/`: IRAF DAOPHOT output files.

## Troubleshooting
- `No module named 'PyQt5'`: `pip install PyQt5`
- `No module named 'astropy'`: `pip install astropy photutils`
- `parameters.toml not found`: create or copy from example.
- FITS files not shown: verify `data_dir`, `filename_prefix`, and file extensions.
