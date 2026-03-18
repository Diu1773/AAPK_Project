"""Load light curve CSV outputs."""

from pathlib import Path
from typing import List

import pandas as pd

from ...utils.step_paths import step10_dir, step11_dir


def find_lightcurve_files(result_dir: Path) -> List[Path]:
    """Return light curve CSV files in a result directory."""
    if not isinstance(result_dir, Path):
        result_dir = Path(result_dir)
    search_dirs = [
        step11_dir(result_dir),
        step10_dir(result_dir),
        result_dir / "step12_isochrone",
        result_dir,
    ]
    for out_dir in search_dirs:
        if not out_dir.exists():
            continue
        files = sorted(out_dir.glob("lightcurve_*.csv"))
        if files:
            return files
    return []


def load_lightcurve_csv(path: Path) -> pd.DataFrame:
    """Load a light curve CSV into a DataFrame."""
    if not isinstance(path, Path):
        path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Light curve file not found: {path}")
    return pd.read_csv(path)
