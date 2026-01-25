"""Load light curve CSV outputs."""

from pathlib import Path
from typing import List

import pandas as pd

from ...utils.step_paths import step12_dir, legacy_step12_isochrone_dir


def find_lightcurve_files(result_dir: Path) -> List[Path]:
    """Return light curve CSV files in a result directory."""
    if not isinstance(result_dir, Path):
        result_dir = Path(result_dir)
    step12_out = step12_dir(result_dir)
    if step12_out.exists():
        files = sorted(step12_out.glob("lightcurve_*.csv"))
        if files:
            return files
    legacy_out = legacy_step12_isochrone_dir(result_dir)
    if legacy_out.exists():
        files = sorted(legacy_out.glob("lightcurve_*.csv"))
        if files:
            return files
    return sorted(result_dir.glob("lightcurve_*.csv"))


def load_lightcurve_csv(path: Path) -> pd.DataFrame:
    """Load a light curve CSV into a DataFrame."""
    if not isinstance(path, Path):
        path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Light curve file not found: {path}")
    return pd.read_csv(path)
