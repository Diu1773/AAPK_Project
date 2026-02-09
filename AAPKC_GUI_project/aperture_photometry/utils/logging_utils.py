"""
Centralized logging configuration for the aperture photometry pipeline.
"""

import logging
import sys
import traceback
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "aperture_phot",
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    console: bool = True
) -> logging.Logger:
    """
    Setup and return a configured logger.

    Parameters
    ----------
    name : str
        Logger name
    level : int
        Logging level (DEBUG, INFO, WARNING, ERROR)
    log_file : Path, optional
        Path to log file. If None, only console output.
    console : bool
        Whether to output to console

    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Format
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S"
    )

    # Console handler
    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    # File handler
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def get_logger(name: str = "aperture_phot") -> logging.Logger:
    """
    Get existing logger or create with defaults.

    Parameters
    ----------
    name : str
        Logger name (use module name like "aperture_phot.core")

    Returns
    -------
    logging.Logger
        Logger instance
    """
    logger = logging.getLogger(name)

    # If no handlers, setup with defaults
    if not logger.handlers and not logger.parent.handlers:
        setup_logger(name)

    return logger


# Convenience: module-level logger
log = get_logger()


def format_exception(e: Exception, include_traceback: bool = True) -> str:
    """
    Format an exception for logging/display.

    Parameters
    ----------
    e : Exception
        The exception to format
    include_traceback : bool
        Whether to include full traceback (default: True)

    Returns
    -------
    str
        Formatted error message with optional traceback
    """
    if include_traceback:
        return f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
    return f"{type(e).__name__}: {str(e)}"


def log_exception(
    logger: logging.Logger,
    e: Exception,
    context: str = "",
    level: int = logging.ERROR
) -> str:
    """
    Log an exception with full traceback and return formatted message.

    Parameters
    ----------
    logger : logging.Logger
        Logger instance to use
    e : Exception
        The exception to log
    context : str
        Additional context (e.g., "processing file X")
    level : int
        Logging level (default: ERROR)

    Returns
    -------
    str
        Formatted error message (for UI display)
    """
    msg = format_exception(e, include_traceback=True)
    if context:
        full_msg = f"[{context}] {msg}"
    else:
        full_msg = msg

    logger.log(level, full_msg)
    return full_msg


class WorkerLogger:
    """
    Logger adapter for QThread workers with signal emission.

    Provides consistent logging that works with both standard logging
    and Qt signal-based log display.

    Usage:
        class MyWorker(QThread):
            log_signal = pyqtSignal(str)

            def __init__(self):
                self._logger = WorkerLogger("MyWorker", self.log_signal.emit)

            def run(self):
                self._logger.info("Starting...")
                try:
                    # work
                except Exception as e:
                    self._logger.exception(e, "processing data")
    """

    def __init__(self, name: str, emit_func=None):
        """
        Initialize worker logger.

        Parameters
        ----------
        name : str
            Logger name (e.g., "ForcedPhotometryWorker")
        emit_func : callable, optional
            Signal emit function for GUI updates (e.g., self.log.emit)
        """
        self._logger = get_logger(f"aperture_phot.worker.{name}")
        self._emit = emit_func

    def _output(self, msg: str, level: int = logging.INFO):
        """Log and optionally emit to GUI."""
        self._logger.log(level, msg)
        if self._emit:
            self._emit(msg)

    def debug(self, msg: str):
        """Log debug message."""
        self._output(msg, logging.DEBUG)

    def info(self, msg: str):
        """Log info message."""
        self._output(msg, logging.INFO)

    def warning(self, msg: str):
        """Log warning message."""
        self._output(f"WARNING: {msg}", logging.WARNING)

    def error(self, msg: str):
        """Log error message."""
        self._output(f"ERROR: {msg}", logging.ERROR)

    def exception(self, e: Exception, context: str = "") -> str:
        """
        Log exception with full traceback.

        Returns formatted message for error signals.
        """
        formatted = format_exception(e, include_traceback=True)
        if context:
            msg = f"[{context}] {formatted}"
        else:
            msg = formatted

        self._logger.error(msg)
        if self._emit:
            # For GUI, show shorter version
            short_msg = f"ERROR: {context}: {type(e).__name__}: {str(e)}" if context else f"ERROR: {type(e).__name__}: {str(e)}"
            self._emit(short_msg)

        return msg
