"""
I/O utility functions
Extracted from AAPKI_GUI.ipynb Cell 0
"""

from __future__ import annotations
from pathlib import Path
import time
from collections import deque


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
