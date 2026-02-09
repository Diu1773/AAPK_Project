# Developer Guide

## Architecture
- GUI entry point: `main.py`.
- Core package: `aperture_photometry/`.
  - `config/`: parameters and validation.
  - `core/`: file manager, instrument config, pipeline utilities.
  - `gui/`: main window and workflow steps.
  - `analysis/`: scientific analysis utilities.
  - `utils/`: shared helpers.
 - Technical reference: `TECHNICAL_REFERENCE.md`.

## Workflow
- Main window shows a 13-step progress list.
- Each step opens a separate popup window.
- State is persisted in `result/project_state.json`.
- Step buttons are gated by completion status.

### Step Window Base
`aperture_photometry/gui/workflow/step_window_base.py` provides:
- Previous / Complete / Next buttons.
- `validate_step()` override for gating completion.
- `save_state()` and `restore_state()` hooks.

## State Format
`result/project_state.json` stores:
- `current_step`
- `completed_steps`
- `step_data` keyed by step ID

Example:
```json
{
  "current_step": 2,
  "completed_steps": [0, 1],
  "step_data": {
    "file_selection": {
      "data_dir": "...",
      "file_count": 45,
      "reference_frame": "pp_001.fits"
    }
  }
}
```

## Adding a New Step
1) Create a window class under `aperture_photometry/gui/workflow/` using `StepWindowBase`.
2) Implement `setup_step_ui()` and `validate_step()`.
3) Register the step in `main_window_workflow.py`.
4) Store any step outputs via `project_state.store_step_data()`.

Minimal template:
```python
from .step_window_base import StepWindowBase

class MyStepWindow(StepWindowBase):
    def __init__(self, params, file_manager, project_state, main_window):
        super().__init__(
            step_index=3,
            step_name="My Step",
            params=params,
            project_state=project_state,
            main_window=main_window
        )
        self.setup_step_ui()
        self.restore_state()

    def setup_step_ui(self):
        pass

    def validate_step(self) -> bool:
        return True
```

## UI Conventions
- Step buttons:
  - Completed: green.
  - Accessible: blue.
  - Locked: gray and disabled.
- Navigation buttons:
  - Previous is always enabled (except Step 1).
  - Next is enabled only after completion.

## Implementation Status (Summary)
- Implemented: Steps 1-13 (File Selection -> Isochrone Model).
- Tools: QA Reports, Extinction (Airmass Fit), and IRAF/DAOPHOT Photometry.

## Step List (Current)
1. File Selection
2. Image Crop
3. Sky Preview & QC
4. Source Detection
5. WCS Plate Solving
6. Star ID Matching
7. Reference Build
8. Master ID Editor
9. Forced Photometry
10. Aperture Overlay
11. Zeropoint Calibration
12. CMD Plot
13. Isochrone Model

## Key Outputs
- `result/photometry_index.csv`: per-frame photometry index.
- `result/gaia_sdss_calibrator_by_ID.csv`: Gaia->SDSS reference table.
- `result/frame_zeropoint.csv`: per-frame ZP, scatter, reference counts.
- `result/median_by_ID_filter_wide_cmd.csv`: CMD-ready calibrated table.
- `result/qa_report/`: QA plots and summary tables.
- `result/extinction/`: extinction fit outputs.
- `result/iraf_phot/`: IRAF DAOPHOT outputs (tool).

## QA Tools
- QA Reports compute background/centroid/error-model/frame quality from photometry TSVs.
- Error model can be evaluated on raw or ZP-corrected magnitudes.
- ZP QA uses `sigma_after^2 = sigma_meas^2 + sigma_ZP^2`, where `sigma_ZP ~= zp_scatter / sqrt(n_ref)`.

See TECHNICAL_REFERENCE.md for formulas, parameter definitions, and method details.

## Extinction Fit
- Uses instrumental magnitudes and airmass for robust linear fit.
- Airmass is read from FITS headers or computed from DATE-OBS + site/RA/DEC.
- Core modules for detection/photometry/analysis are placeholders.

## Build
- `pyinstaller --onefile --windowed --add-data "aperture_photometry;aperture_photometry" --name "AAPKI-Photometry" main.py`

## Testing
- No formal tests yet. If added:
  - Use `aperture_photometry/tests/test_*.py`.
  - Run `python -m pytest`.

## Design Notes
- PyQt5 chosen for native performance and PyInstaller compatibility.
- Step-based workflow enables clear, recoverable processing.
- Step data is persisted to allow resume after restart.
