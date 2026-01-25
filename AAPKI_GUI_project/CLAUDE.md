# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Aperture Photometry Toolkit (AAPKI) - a PyQt5 GUI application for step-by-step aperture photometry of FITS astronomical images with state persistence.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the GUI
python main.py

# Build Windows executable
pyinstaller --onefile --windowed --add-data "aperture_photometry;aperture_photometry" --name "AAPKI-Photometry" main.py

# Run tests (when added)
python -m pytest
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
- `aperture_photometry/config/` - Parameter parsing from parameters.toml
- `aperture_photometry/core/` - FileManager, InstrumentConfig, ProjectState
- `aperture_photometry/gui/` - PyQt5 windows and widgets
- `aperture_photometry/gui/workflow/` - Step window implementations
- `aperture_photometry/utils/` - I/O and astronomy helpers

### Workflow Pattern

The main window (`main_window_workflow.py`) displays 13 sequential step buttons:
- Locked (gray) -> Accessible (blue) -> Completed (green)
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

### Implementation Status
- Implemented: Steps 1-13 (File Selection -> Isochrone Model)
- Tools: QA Reports and Extinction (Airmass Fit)

## Coding Conventions

- PEP 8 style: 4-space indentation, snake_case functions, CapWords classes
- GUI classes in `gui/`, processing logic in `core/` or `analysis/`
- Step files named `stepN_*.py`
- Tests in `aperture_photometry/tests/test_*.py`

## External Dependencies

ASTAP is required for WCS solving (Steps 4+): https://www.hnsky.org/astap.htm
