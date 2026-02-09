"""Eclipsing light curve utilities."""

from pathlib import Path
import pandas as pd

from .loader import load_lightcurve_csv


def build_eclipse_lightcurve(path: Path) -> pd.DataFrame:
    """Load an eclipsing light curve from CSV."""
    return load_lightcurve_csv(path)
