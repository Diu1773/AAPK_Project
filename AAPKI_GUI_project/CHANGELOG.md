# Changelog

All notable changes are documented here. Dates follow the existing project notes.

## 2024-12-29

### Configuration and Tools
- Switched to TOML-only configuration defaults and updated GUI dialogs to prefer `parameters.toml`.
- IRAF/DAOPHOT tool now loads/saves parameters in `parameters.toml` and auto-cleans `_pyraf_photometry.py`.
- Extinction tool outputs are stored under `result/extinction/`.

### QA and Visualization
- QA Report now auto-saves parameters to `parameters.toml` and adds per-filter summaries and plots.
- Step 12 CMD viewer is embedded and opens when the step loads.
- Step 3 zoom/measurement state persists across frame navigation.

### Docs
- Added `TECHNICAL_REFERENCE.md` and refreshed docs for current pipeline behavior.

## 2024-12-25

### Step 3 (Sky Preview and QC)
- Added imexamine-style keyboard shortcuts:
  - `m` measure at cursor.
  - `h` histogram around cursor.
  - `g` radial profile around cursor.
  - `.` cycle filter.
  - `[` and `]` previous/next frame.
- Added histogram and radial profile dialogs.
- Added photometry parameter dialog.
- Added enhanced auto-stretch (PixInsight-style) for ZScale.

### Bug Fixes and Behavior Changes
- Removed automatic measurement on left-click; measurements are keyboard-driven.
- Removed cursor overlay preview to match AAPKI Cell 6 behavior.
- Fixed scalar indexing errors from `ApertureStats`.
- Suppressed `NoDetectionsWarning` from `DAOStarFinder`.
- Added safer patch removal for overlays.

## 2024-12-27

### Workflow and Tools
- Updated workflow to 13 steps (forced photometry -> ZP calibration -> CMD -> isochrone).
- QA Reports moved under Tools and expanded (raw/ZP error model, frame quality).
- Added Extinction (Airmass Fit) tool using instrumental magnitudes.

### Calibration and QA
- Per-frame zeropoint calibration with frame-level ZP outputs.
- Airmass computation from header or DATE-OBS + site/RA/DEC.
- ZP QA uses `sigma_after^2 = sigma_meas^2 + sigma_ZP^2`.

### Data Paths
- Cropped FITS are stored under `result/cropped/`.

## 2024-12-24

### Step 1 Improvements
- Case-insensitive filename prefix matching.
- Removed auto-scan on directory change.
- Removed auto-reference selection; Step 1 only validates file loading.

### Instrument Settings Dialog
- Instrument settings moved to a dedicated dialog.
- Gain, read noise, and saturation are now editable.
- Changes persist to `parameters.toml`.

### Step 2 (Image Crop)
- Implemented interactive crop selection with matplotlib.
- Applied crop to all FITS files and saved to `cropped/`.
- Saved crop metadata to `result/crop_rect.json` and project state.

### Main Window and Navigation
- Step buttons are disabled when locked.
- Added lock icon for locked steps.
- Navigation buttons updated with consistent colors and enabled states.
- Fixed crash when resuming next step without completing prior steps.
