"""
I/O utility functions
Extracted from AAPKI_GUI.ipynb Cell 0
"""

from __future__ import annotations
from pathlib import Path
import json
import time
from collections import deque
from typing import Union
from decimal import Decimal, InvalidOperation

import pandas as pd


def _parse_int64_col(series: pd.Series) -> pd.array:
    """Convert a string/object series of source_ids to pandas Int64.

    - Exact integer strings: "2823345641878527872" → preserved with full precision
    - Float strings (old format): "2823345641878528000.0" → int (already-rounded)
    - Blank / nan / NA → pd.NA
    """
    def _parse(s):
        if pd.isna(s):
            return pd.NA
        sval = str(s).strip()
        if sval in ("", "nan", "NaN", "<NA>", "None"):
            return pd.NA
        try:
            # Keep exact precision for integer-like decimal/exponent strings.
            if "." in sval or "e" in sval.lower():
                d = Decimal(sval)
                if not d.is_finite():
                    return pd.NA
                if d != d.to_integral_value():
                    return pd.NA
                return int(d)
            return int(sval)
        except (ValueError, OverflowError, InvalidOperation):
            return pd.NA
    return pd.array([_parse(v) for v in series], dtype="Int64")


def coerce_int64_source_id(series: pd.Series) -> pd.Series:
    """Coerce arbitrary source_id values to nullable Int64 without float round-trip."""
    if series is None:
        return pd.Series(pd.array([], dtype="Int64"))
    parsed = _parse_int64_col(series)
    return pd.Series(parsed, index=series.index, dtype="Int64")


# Alias for compatibility with AAPKC code that uses parse_int64_series
def parse_int64_series(series: pd.Series) -> pd.Series:
    """Convert a series to pandas nullable Int64 without float precision loss.

    Alias for coerce_int64_source_id for cross-project compatibility.
    """
    return coerce_int64_source_id(series)


def read_csv_int64_source_id(path: Union[str, Path], sep: str = ",", **kwargs) -> pd.DataFrame:
    """Read a CSV/TSV file preserving 19-digit Gaia source_id precision.

    pandas default read_csv promotes a column with mixed integer/NaN to float64,
    silently rounding the last 3-4 digits of 19-digit Gaia source_ids.
    This function reads source_id as string then converts to Int64.
    """
    df = pd.read_csv(path, sep=sep, dtype={"source_id": str}, **kwargs)
    if "source_id" in df.columns:
        df["source_id"] = coerce_int64_source_id(df["source_id"])
    return df


def load_night_assignments(result_dir: Path) -> dict[str, int]:
    """Load filename → night_id from step1/night_assignments.json."""
    from .step_paths import step1_dir
    na_path = step1_dir(result_dir) / "night_assignments.json"
    if not na_path.exists():
        na_path = result_dir / "night_assignments.json"
    if not na_path.exists():
        return {}
    try:
        data = json.loads(na_path.read_text(encoding="utf-8"))
        return {k: int(v) for k, v in data.get("night_assignments", {}).items()}
    except Exception:
        return {}


def load_headers_table(result_dir: Path) -> pd.DataFrame:
    """Read headers.csv from step1 dir (fallback: result_dir)."""
    from .step_paths import step1_dir
    headers_path = step1_dir(result_dir) / "headers.csv"
    if not headers_path.exists():
        headers_path = result_dir / "headers.csv"
    if not headers_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(headers_path)
    except Exception:
        return pd.DataFrame()


def load_file_path_map(result_dir: Path) -> dict[str, str]:
    """Load filename → original FITS path mapping from step1/file_path_map.json."""
    from .step_paths import step1_dir
    path = step1_dir(result_dir) / "file_path_map.json"
    if not path.exists():
        path = result_dir / "file_path_map.json"
    if not path.exists():
        project_state_path = Path(result_dir) / "project_state.json"
        if not project_state_path.exists():
            return {}
        try:
            state = json.loads(project_state_path.read_text(encoding="utf-8"))
            step_data = state.get("step_data", {})
            file_sel = step_data.get("file_selection", {}) if isinstance(step_data, dict) else {}
            data = file_sel.get("file_path_map", {}) if isinstance(file_sel, dict) else {}
        except Exception:
            return {}
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, str] = {}
    for key, value in data.items():
        if not key or not value:
            continue
        out[str(key)] = str(value)
    return out


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
        except:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
