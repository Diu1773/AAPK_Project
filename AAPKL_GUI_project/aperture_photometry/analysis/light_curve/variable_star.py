"""Variable star light curve utilities."""

from pathlib import Path
import pandas as pd

from .loader import load_lightcurve_csv


def build_variable_star_lightcurve(path: Path) -> pd.DataFrame:
    """Load a variable star light curve from CSV."""
    return load_lightcurve_csv(path)
