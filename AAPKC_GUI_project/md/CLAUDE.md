# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

**AAPKC** (Auto Aperture Photometry KNU EMAO CMD) — PyQt5 GUI for step-by-step aperture
photometry of FITS images, producing Color-Magnitude Diagrams and Isochrone fitting.
Sister project **AAPKL** (same parent dir) produces light curves instead.

## Commands

```bash
# Run GUI (WSL/Linux environment)
python3 main.py

# Syntax check a file
python3 -c "import ast; ast.parse(open('path/to/file.py').read()); print('OK')"

# Build Windows executable
pyinstaller --onefile --windowed --add-data "aperture_photometry;aperture_photometry" --name "AAPKC" main.py
```

> **Always use `python3`**, not `python` (WSL environment).

## Pipeline: 13-Step WCS-First Architecture

| Step | UI Index | File | Class | Output Dir |
|------|----------|------|-------|-----------|
| 1 File Selection | 0 | step1_file_selection_window.py | FileSelectionWindow | — |
| 2 Image Crop | 1 | step2_crop_selector.py | CropSelectorWindow | step2_cropped/ |
| 3 Sky Preview | 2 | step3_sky_preview.py | SkyPreviewWindow | — |
| 4 Source Detection | 3 | step4_source_detection.py | SourceDetectionWindow | step4_detection/ |
| 5 Aperture Photometry | 4 | step5_aperture_photometry.py | AperturePhotometryWindow | step5_aperture/ |
| 6 PSF Photometry | 5 | step6_psf_photometry.py | PSFPhotometryWindow | step6_psf/ |
| 7 WCS Plate Solving | 6 | step7_wcs_plate_solving.py | WcsPlateSolvingWindow | step5_wcs/ |
| 8 Reference Catalog Build | 7 | step8_ref_build.py | RefBuildWindow | step6_refbuild/ |
| 9 Star ID Matching | 8 | step9_star_id_matching.py | StarIdMatchingWindow | step7_idmatch/ |
| 10 Master ID Editor | 9 | step10_master_id_editor.py | MasterIdEditorWindow | step8_selection/ |
| 11 Zeropoint Calibration | 10 | step10_zeropoint_calibration.py | ZeropointCalibrationWindow | step10_zeropoint/ |
| 12 CMD Plot | 11 | step11_cmd_plot.py | CmdPlotWindow | step11_cmd/ |
| 13 Isochrone Model | 12 | step12_isochrone_model.py | IsochroneModelWindow | step12_isochrone/ |

**Tools (non-step):** `extinction_fit_window.py`, `iraf_comparison_window.py`, `aperture_overlay_panel.py`

> **Step index is 0-based.** `step_index=6` = Step 7 (WCS). Always verify when creating/modifying steps.

## step_paths.py — Critical Mapping

Output directory functions. **Legacy names ≠ semantic names** — don't confuse:

```python
step5_dir()          → step5_wcs/        # WCS outputs (Step 7)
step6_dir()          → step6_refbuild/   # RefBuild outputs (Step 8)
step7_dir()          → step7_idmatch/    # IDMatch outputs (Step 9)
step8_dir()          → step8_selection/  # MasterID outputs (Step 10)
step5_aperture_dir() → step5_aperture/   # Aperture photometry (Step 5)
step6_psf_dir()      → step6_psf/        # PSF photometry (Step 6)
step4_dir()          → step4_detection/  # Detection (Step 4)
```

Legacy fallback functions (`legacy_step5_refbuild_dir` etc.) exist for backward compat — don't remove.

## Data Flow

```
Step 4 (detect_{fname}.csv, det_uid)
    ↓
Step 5 (photometry_{fname}.tsv, xcenter/ycenter)
    ↓
Step 6 (photometry_{fname}.tsv, x_fit/y_fit, iter_found, det_uid<0 = new sources)
    ↓
Step 7 (wcs_*.fits + gaia_fov.ecsv → per-frame WCS)
    ↓
Step 8 (ref_catalog.tsv, master_catalog.tsv, sourceid_to_ID.csv)
    PSF positions used: step6_psf/ → step5_aperture/ → step4_detection/ (fallback)
    ↓
Step 9 (idmatch_{fname}.csv — RA/Dec KDTree matching)
    ↓
Step 10 (master_star_ids.csv — PSF iter2 new sources included)
    ↓
Step 11 → Step 12 → Step 13
```

## Common Pitfalls

### Gaia source_id precision
Gaia `source_id` is int64. **Never store as float64** — causes ~128 rounding error.
Use `parse_int64_series()` from `utils/io_utils.py`.

### Gaia ROW_LIMIT
`astroquery.gaia` default is 2000 rows. Always set before query:
```python
Gaia.ROW_LIMIT = -1
job = Gaia.launch_job_async(adql, dump_to_file=False)
```

### WCS hint coordinates
`OBJCTRA/OBJCTDEC` in FITS header can be wrong (e.g., different night's target).
If hint-based solve fails → blind retry (center_coord=None) is implemented
(`astnet_blind_retry_on_fail=True` default).

### PSF iter2 sources
`iter_found > 1` sources have `det_uid < 0`. These are new detections from residual images.
- Step 8 RefBuild: use `iter_found == 1` only (stable positions)
- Step 10 MasterID: explicitly loads iter2 sources via `load_psf_new_sources()`

## Architecture

### Package Structure
```
main.py                          # Entry point
aperture_photometry/
  config/                        # parameters.py (Parameters/ParamSet), schema.py
  core/                          # FileManager, InstrumentConfig, ProjectState
  gui/
    main_window_workflow.py      # Main window, step button management
    workflow/                    # All step window files
    tools/                       # Non-step tool windows
  utils/
    step_paths.py                # Output directory functions
    io_utils.py                  # parse_int64_series, read_ecsv_int64_source_id
    photometry_utils.py          # Shared photometry helpers (step9 + extinction)
    qc_utils.py                  # filter_files_by_qc
```

### StepWindowBase Pattern

```python
from .step_window_base import StepWindowBase

class MyStepWindow(StepWindowBase):
    def __init__(self, params, file_manager, project_state, main_window):
        super().__init__(
            step_index=N,           # 0-based: Step N+1 in UI
            step_name="My Step",
            params=params,
            project_state=project_state,
            main_window=main_window,
        )
        self.setup_step_ui()
        self.restore_state()

    def setup_step_ui(self): ...
    def validate_step(self) -> bool: return True
```

Register in `main_window_workflow.py`:
1. Add name to `self.step_names` at correct index
2. Add `elif step_index == N:` block in `open_step()`

### Parameters

- Read via `getattr(self.params.P, "param_name", default)`
- Add new params in `config/parameters.py` (`_getf`, `_geti`, `_as_bool`)
- Optional schema validation in `config/schema.py`
- Persist: `self.persist_params()` after updating `self.params.P.*`

## Coding Conventions

- PEP 8: 4-space indent, snake_case functions, CapWords classes
- `_safe_float(x, default)` inlined in each file (no common helpers in AAPKC)
- Worker threads via `QThread` + `pyqtSignal`
- No `python` — always `python3` in CLI

## External Dependencies

- **astrometry.net** (`solve-field`) — WCS plate solving, called via WSL
- **astroquery** — Gaia DR3 TAP queries
- **photutils** — PSF photometry (EPSFBuilder, PSFPhotometry)
- **astropy** — FITS, WCS, SkyCoord
- **PyQt5** — GUI framework
