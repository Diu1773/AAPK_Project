# CLI Guide (AI Reference)

This document is the primary reference for running and editing this repository via CLI or automation.

## Purpose
- Provide a single source of truth for commands, file layout, and workflow assumptions.
- Summarize conventions for safe edits and consistent structure.
- Point to technical details in TECHNICAL_REFERENCE.md.

## Quick Commands
- Install deps: `pip install -r requirements.txt`
- Run GUI: `python main.py`
- Build exe (Windows):
  `pyinstaller --onefile --windowed --add-data "aperture_photometry;aperture_photometry" --name "AAPKI-LightCurve" main.py`

## Required Files
- `parameters.toml` must exist in repo root for normal execution.
  - Copy from `parameters_example.toml` and edit paths and camera values.
- Outputs are written under `<data_dir>/result/`.
- Full parameter definitions and formulas: `TECHNICAL_REFERENCE.md`.

## Project Layout
- `main.py` is the GUI entry point.
- Core package: `aperture_photometry/`
  - `config/`: parameter parsing and validation.
  - `core/`: file and instrument handling and pipeline state.
  - `gui/`: PyQt5 windows and workflow steps.
  - `utils/`: shared I/O and astronomy helpers.
  - `analysis/`: scientific analysis modules.
  - `tests/`: placeholder for future tests.

## Workflow Steps (GUI)
- Step 1: File selection and FITS header scan.
- Step 2: Crop selection and batch cropping to `result/cropped/`.
- Step 3: Sky preview and QC (imexamine-style tools).
- Step 4: Source detection.
- Step 5: WCS plate solving.
- Step 6: Star ID matching.
- Step 7: Reference build.
- Step 8: Master ID editor.
- Step 9: Forced photometry.
- Step 10: Aperture overlay.
- Step 11: Zeropoint calibration.
- Step 12: CMD plot.
- Step 13: Isochrone model.

## Tools (Top Menu)
- QA Report: photometry validation and QA tables/plots.
- Extinction (Airmass Fit): per-filter k fit and diagnostics.
- IRAF/DAOPHOT Photometry: run PyRAF in WSL and save outputs.

## Data and State Files
- `parameters.toml`: user config.
- `result/project_state.json`: workflow state and per-step data.
- `result/crop_rect.json`: crop selection metadata.
- `result/cropped/`: cropped FITS files created by Step 2.
- `result/cache/`: detection, WCS, and ID match caches.
- `result/photometry_index.csv`: per-frame photometry index.
- `result/frame_zeropoint.csv`: per-frame ZP results.
- `result/qa_report/`: QA outputs.
- `result/extinction/`: extinction fit outputs.
- `result/iraf_phot/`: IRAF photometry outputs (tool).

## External Dependencies
- ASTAP is required only for WCS solving steps (future). Install separately when needed.

## Conventions for Edits
- Python: 4-space indentation, PEP 8 naming.
- GUI classes stay in `aperture_photometry/gui/`.
- Processing logic stays in `aperture_photometry/core/` or `analysis/`.
- Step modules use `stepN_*` naming.

## Testing
- No automated tests configured. If adding tests later, use `python -m pytest` and place tests in `aperture_photometry/tests/` as `test_*.py`.

## Troubleshooting (Common)
- Missing `parameters.toml`: copy from `parameters_example.toml`.
- Import errors: reinstall dependencies with `pip install -r requirements.txt`.
- No FITS files found: verify `data_dir` and `filename_prefix` in `parameters.toml`.
