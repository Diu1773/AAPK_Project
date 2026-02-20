"""
I/O utility functions
Extracted from AAPKI_GUI.ipynb Cell 0
"""

from __future__ import annotations
from pathlib import Path
import time
from collections import deque
from typing import Union
from decimal import Decimal, InvalidOperation

import pandas as pd

INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1
NULL_STRINGS = {"", "nan", "none", "null", "<na>", "na"}


def parse_int64_scalar(value):
    """Parse a scalar into signed int64 safely without float64 round-trip."""
    if pd.isna(value):
        return pd.NA

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, int):
        return value if INT64_MIN <= value <= INT64_MAX else pd.NA

    if isinstance(value, float):
        if not pd.notna(value):
            return pd.NA
        if not float(value).is_integer():
            return pd.NA
        iv = int(value)
        return iv if INT64_MIN <= iv <= INT64_MAX else pd.NA

    s = str(value).strip()
    if s.lower() in NULL_STRINGS:
        return pd.NA

    try:
        d = Decimal(s)
    except (InvalidOperation, ValueError):
        return pd.NA

    if d != d.to_integral_value():
        return pd.NA

    try:
        iv = int(d)
    except (ValueError, OverflowError):
        return pd.NA

    return iv if INT64_MIN <= iv <= INT64_MAX else pd.NA


def parse_int64_series(series: pd.Series) -> pd.Series:
    """Convert a series to pandas nullable Int64 without float precision loss."""
    arr = pd.array([parse_int64_scalar(v) for v in series], dtype="Int64")
    return pd.Series(arr, index=series.index, dtype="Int64")


def _parse_int64_col(series: pd.Series) -> pd.array:
    """Convert a string/object series of source_ids to pandas Int64.

    - Exact integer strings: "2823345641878527872" → preserved with full precision
    - Float strings (old format): "2823345641878528000.0" → int (already-rounded)
    - Blank / nan / NA → pd.NA
    """
    return pd.array(parse_int64_series(series), dtype="Int64")


def read_csv_int64_source_id(path: Union[str, Path], sep: str = ",", **kwargs) -> pd.DataFrame:
    """Read a CSV/TSV file preserving 19-digit Gaia source_id precision.

    pandas default read_csv promotes a column with mixed integer/NaN to float64,
    silently rounding the last 3-4 digits of 19-digit Gaia source_ids.
    This function reads source_id as string then converts to Int64.
    """
    df = pd.read_csv(path, sep=sep, dtype={"source_id": str}, **kwargs)
    if "source_id" in df.columns:
        df["source_id"] = _parse_int64_col(df["source_id"])
    return df


def read_ecsv_int64_source_id(path: Union[str, Path]) -> pd.DataFrame:
    """Read ECSV and preserve 64-bit source_id precision."""
    from astropy.table import Table

    tab = Table.read(str(path), format="ascii.ecsv")
    cols = list(tab.colnames)
    lower = [c.lower() for c in cols]
    if cols != lower:
        tab.rename_columns(cols, lower)
    df = tab.to_pandas()
    if "source_id" in df.columns:
        df["source_id"] = parse_int64_series(df["source_id"])
    return df


class TailLogger:
    """
    Logger that maintains a tail buffer of recent messages
    Useful for displaying recent activity in GUI
    """

    def __init__(self, log_path: Path, tail: int = 5, enable_console: bool = True):
        """
        Initialize tail logger

        Args:
            log_path: Path to log file
            tail: Number of recent messages to keep in buffer
            enable_console: Whether to print to console
        """
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.log_path, "a", encoding="utf-8")
        self.buf = deque(maxlen=max(1, tail))
        self.enable_console = enable_console

        # Try to import IPython clear_output for Jupyter support
        try:
            from IPython.display import clear_output
            self._clear = lambda: clear_output(wait=True)
        except Exception:
            self._clear = lambda: None

    def write(self, msg: str):
        """Write message to log file and buffer"""
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        self.fh.write(line + "\n")
        self.fh.flush()

        if self.enable_console:
            self.buf.append(line)
            self._clear()
            print("\n".join(self.buf))

    def get_recent(self) -> list[str]:
        """Get recent messages from buffer"""
        return list(self.buf)

    def close(self):
        """Close log file"""
        try:
            self.fh.close()
        except Exception:
            pass

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
