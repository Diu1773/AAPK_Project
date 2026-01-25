"""
File management for FITS image processing
Extracted from AAPKI_GUI.ipynb Cell 2-3
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
from astropy.io import fits

from ..utils import get_logger
from ..utils.step_paths import step1_dir

log = get_logger("aperture_phot.core.file_manager")


class FileManager:
    """
    Manages FITS file discovery, header reading, and reference frame selection
    """

    def __init__(self, params):
        """
        Initialize file manager

        Args:
            params: Parameters object from config module
        """
        self.params = params
        self.filenames: List[str] = []
        self.df_headers: Optional[pd.DataFrame] = None
        self.ref_filename: Optional[str] = None

    def scan_files(self) -> List[str]:
        """
        Scan data directory for FITS files matching prefix pattern

        Returns:
            List of matching filenames (sorted)

        Raises:
            RuntimeError: If no matching files found
        """
        data_dir = self.params.P.data_dir
        prefix = self.params.P.filename_prefix

        # Find all FITS files matching prefix (case-insensitive)
        self.filenames = sorted([
            f for f in os.listdir(data_dir)
            if f.lower().startswith(prefix.lower()) and f.lower().endswith((".fit", ".fits", ".fit.fz", ".fits.fz"))
        ])

        if not self.filenames:
            raise RuntimeError(f"No input files found with prefix '{prefix}' in {data_dir}")

        # Save target list
        output_dir = step1_dir(self.params.P.result_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        target_list_path = output_dir / "target_list.txt"
        target_list_path.write_text("\n".join(self.filenames), encoding="utf-8")

        log.info(f"Found {len(self.filenames)} files -> {target_list_path}")
        return self.filenames

    def read_headers(self) -> pd.DataFrame:
        """
        Read FITS headers from all files and create summary DataFrame

        Returns:
            DataFrame with header information (Filename, DATE-OBS, FILTER, EXPTIME, etc.)
        """
        if not self.filenames:
            self.scan_files()

        rows = []
        for fn in self.filenames:
            try:
                file_path = self.params.P.data_dir / fn
                with fits.open(file_path) as hdul:
                    h = hdul[0].header

                # Get EXPTIME with proper numeric handling
                exptime_val = h.get("EXPTIME", None)
                if exptime_val is None:
                    # Try alternative keywords
                    for key in ("EXPOSURE", "ITIME", "ELAPTIME"):
                        if key in h:
                            exptime_val = h[key]
                            break
                try:
                    exptime = float(exptime_val) if exptime_val is not None else np.nan
                except (TypeError, ValueError):
                    exptime = np.nan

                # Get AIRMASS with proper numeric handling
                airmass_val = h.get("AIRMASS", None)
                try:
                    airmass = float(airmass_val) if airmass_val is not None else np.nan
                except (TypeError, ValueError):
                    airmass = np.nan

                rows.append({
                    "Filename": fn,
                    "DATE-OBS": h.get("DATE-OBS", "N/A"),
                    "FILTER": h.get("FILTER", "UNKNOWN"),
                    "EXPTIME": exptime,
                    "AIRMASS": airmass,
                    "IMAGETYP": h.get("IMAGETYP", h.get("FRAME", "Unknown")),
                })
            except Exception as e:
                log.warning(f"{fn}: Header read failed - {e}")

        self.df_headers = pd.DataFrame(rows)

        # Save headers CSV
        output_dir = step1_dir(self.params.P.result_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        headers_path = output_dir / "headers.csv"
        self.df_headers.to_csv(headers_path, index=False, encoding="utf-8")
        log.info(f"Saved: {headers_path} | rows: {len(self.df_headers)}")

        return self.df_headers

    def select_reference_frame(self) -> str:
        """
        Select reference frame based on filter and index preferences

        Returns:
            Filename of selected reference frame
        """
        if self.df_headers is None:
            self.read_headers()

        # Start with all candidates
        ref_candidates = self.df_headers

        # Filter by global_ref_filter if specified
        grf = str(getattr(self.params.P, "global_ref_filter", "")).lower()
        if grf in ("g", "r", "i"):
            filtered = self.df_headers[
                self.df_headers["FILTER"].astype(str).str.lower() == grf
            ]
            if not filtered.empty:
                ref_candidates = filtered

        # Select by index
        ref_index = max(
            0,
            min(
                int(getattr(self.params.P, "global_ref_index", 0) or 0),
                len(ref_candidates) - 1,
            ),
        )

        if not ref_candidates.empty:
            self.ref_filename = ref_candidates.iloc[ref_index]["Filename"]
        else:
            self.ref_filename = self.filenames[0]

        log.info(f"Selected reference frame: {self.ref_filename}")
        return self.ref_filename

    def get_reference_header(self) -> fits.Header:
        """
        Get FITS header of reference frame

        Returns:
            FITS header object
        """
        if self.ref_filename is None:
            self.select_reference_frame()

        file_path = self.params.P.data_dir / self.ref_filename
        with fits.open(file_path) as hdul:
            return hdul[0].header.copy()

    def print_reference_header(self):
        """Print reference frame header for inspection"""
        header = self.get_reference_header()
        log.info(f"Reference Frame Header: {self.ref_filename}")
        log.debug(repr(header))

    def get_file_path(self, filename: str) -> Path:
        """
        Get full path to a file in data directory

        Args:
            filename: Base filename

        Returns:
            Full path to file
        """
        return self.params.P.data_dir / filename

    def get_all_file_paths(self) -> List[Path]:
        """
        Get full paths to all scanned files

        Returns:
            List of Path objects
        """
        if not self.filenames:
            self.scan_files()
        return [self.get_file_path(fn) for fn in self.filenames]
