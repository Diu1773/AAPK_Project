# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Aperture Photometry Toolkit (AAPKI) - a PyQt5 GUI application for step-by-step aperture photometry of FITS astronomical images with state persistence. Designed for analyzing variable stars, eclipsing binaries, and asteroids from multi-frame astronomical surveys.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the GUI
python main.py

# Build Windows executable
pyinstaller --onefile --windowed --add-data "aperture_photometry;aperture_photometry" --name "AAPKI-Photometry" main.py

# Run tests (when added)
python -m pytest aperture_photometry/tests/
```

## Configuration

Copy `parameters_example.toml` to `parameters.toml` and set:
- `data_dir`: path to FITS data
- `filename_prefix`: FITS filename prefix (e.g., `pp_`)
- `rdnoise_e`: camera read noise (electrons)
- `gain_e_per_adu`: camera gain (e-/ADU)

Full parameter definitions and formulas are documented in `TECHNICAL_REFERENCE.md`.

## Architecture

### Package Structure
- `main.py` - Entry point, creates QApplication and MainWindowWorkflow
- `aperture_photometry/config/` - TOML parameter parsing with Pydantic validation (`schema.py`)
- `aperture_photometry/core/` - FileManager (FITS discovery), InstrumentConfig (telescope/camera), ProjectState (workflow persistence)
- `aperture_photometry/gui/` - PyQt5 windows and widgets
- `aperture_photometry/gui/workflow/` - Step window implementations (step1-12)
- `aperture_photometry/gui/light_curve/` - Light curve tabs (variable, eclipsing, asteroid)
- `aperture_photometry/analysis/` - Scientific analysis modules (isochrone fitting, light curve builders)
- `aperture_photometry/utils/` - I/O helpers, astronomy utilities, constants, step output paths

### Data Flow

```
main.py → MainWindowWorkflow
           ├─ Parameters (config/parameters.py)
           ├─ FileManager (core/file_manager.py)
           ├─ InstrumentConfig (core/instrument.py)
           └─ ProjectState (core/project_state.py) → project_state.json
               └─ StepWindows (stepN_*.py) ← StepWindowBase
```

### Workflow Pattern

The main window (`main_window_workflow.py`) displays 13 sequential step buttons:
1. File Selection → 2. Image Crop → 3. Sky Preview & QC → 4. Source Detection
5. WCS Plate Solving → 6. Star ID Matching → 7. Reference Build → 8. Master ID Editor
9. Forced Photometry → 10. Aperture Overlay → 11. Zeropoint Calibration → 12. CMD Plot → 13. Isochrone Model

Button states: Locked (gray) → Accessible (blue) → Completed (green)
- Steps unlock linearly; completing step N enables step N+1
- State persisted to `<data_dir>/result/project_state.json`

Each step window extends `StepWindowBase` which provides:
- Previous/Complete/Next navigation buttons
- `validate_step()` override for gating completion
- `save_state()` and `restore_state()` hooks

### Adding a New Step

1. Create class in `aperture_photometry/gui/workflow/stepN_*.py` extending `StepWindowBase`
2. Implement `setup_step_ui()` and `validate_step()`
3. Register in `main_window_workflow.py`
4. Store outputs via `project_state.store_step_data()`

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

### Key Output Files

All outputs go to `<data_dir>/result/`:
- `project_state.json` - Workflow progress and step data
- `photometry_index.csv` - Per-frame photometry index
- `gaia_sdss_calibrator_by_ID.csv` - Reference stars with Gaia→SDSS transforms
- `frame_zeropoint.csv` - Per-frame zero-point calibration
- `median_by_ID_filter_wide_cmd.csv` - CMD-ready calibrated table

### Threading Pattern

Long operations use `QThread` workers (e.g., `AperturePhotometryWorker` in step9). Progress signals update the UI without blocking. Parallelism is controlled by `parallel.mode` (thread/process/auto/none) and `parallel.max_workers` in parameters.toml.

## Coding Conventions

- PEP 8 style: 4-space indentation, snake_case functions, CapWords classes
- GUI classes in `gui/`, processing logic in `core/` or `analysis/`
- Step files named `stepN_*.py` (e.g., `step2_crop_selector.py`)
- Step classes named descriptively (e.g., `CropSelectorWindow`)
- Tests in `aperture_photometry/tests/test_*.py`
- Pixel coordinates follow photutils convention (0-based)

## External Dependencies

- **ASTAP** is required for WCS solving (Steps 5+): https://www.hnsky.org/astap.htm
- **PyRAF/IRAF** (optional) for IRAF/DAOPHOT tool (runs via WSL on Windows)
