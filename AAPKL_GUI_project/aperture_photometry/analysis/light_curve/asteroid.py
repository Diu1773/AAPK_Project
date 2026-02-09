"""Asteroid light curve utilities."""

from pathlib import Path
import pandas as pd

from .loader import load_lightcurve_csv


def build_asteroid_lightcurve(path: Path) -> pd.DataFrame:
    """Load an asteroid light curve from CSV."""
    return load_lightcurve_csv(path)
