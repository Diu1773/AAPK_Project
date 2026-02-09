"""
Centralized logging configuration for the aperture photometry pipeline.
"""

import logging
import sys
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
