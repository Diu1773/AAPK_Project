# Repository Guidelines

## Project Structure & Module Organization
- `main.py` is the GUI entry point.
- Core package lives in `aperture_photometry/` with submodules:
  - `config/` parameter handling (see `parameters_example.toml`).
  - `core/` pipeline state and file/instrument management.
  - `gui/` PyQt5 windows, workflow steps, and widgets.
  - `gui/light_curve/` light curve tabs (variable, eclipsing, asteroid).
  - `utils/` shared helpers (I/O, astronomy utilities).
  - `analysis/light_curve/` light curve loaders and builders.
  - `tests/` placeholder for future tests.
- User configuration is loaded from `parameters.toml` in the repo root.
- Docs and status notes: `README.md`, `USER_GUIDE.md`, `TECHNICAL_REFERENCE.md`.

## Build, Test, and Development Commands
- `pip install -r requirements.txt` installs runtime dependencies.
- `python main.py` runs the GUI locally.
- `pyinstaller --onefile --windowed --add-data "aperture_photometry;aperture_photometry" --name "AAPKI-LightCurve" main.py` builds a Windows executable.
- If you add tests later, prefer `python -m pytest` (not currently configured).

## Coding Style & Naming Conventions
- Python style: 4-space indentation, PEP 8 naming (snake_case for functions/vars, CapWords for classes).
- Keep GUI classes in `aperture_photometry/gui/` and processing logic in `aperture_photometry/core/` or `analysis/`.
- Match existing module naming: `stepN_*` for workflow steps (e.g., `step2_crop_selector.py`).

## Testing Guidelines
- No formal test framework is configured yet; `aperture_photometry/tests/` is empty.
- If you add tests, place them under `aperture_photometry/tests/` and name files `test_*.py`.

## Commit & Pull Request Guidelines
- Git history is not available in this workspace, so no established commit convention was detected.
- Suggested standard: Conventional Commits (e.g., `feat: add crop preview`, `fix: handle missing param file`).
- PRs should include: purpose summary, steps to verify, and screenshots for UI changes.

## Configuration & Data Notes
- `parameters.toml` is required for running the GUI; copy from `parameters_example.toml` and edit paths (e.g., `data_dir`, `filename_prefix`).
- The pipeline writes outputs under `<data_dir>/result/` (see `USER_GUIDE.md`).
- ASTAP is an external dependency for WCS solving; install separately when enabling those steps.
