"""
Instrument configuration (Telescope + Camera)
Extracted from AAPKI_GUI.ipynb Cell 1
"""

from __future__ import annotations
import math
import numpy as np
import pandas as pd
from pathlib import Path
from astropy.coordinates import SkyCoord
import astropy.units as u
from typing import Optional
from ..utils.step_paths import step1_dir

# Try to import SIMBAD
try:
    from astroquery.simbad import Simbad
    _HAS_SIMBAD = True
except Exception as e:
    print(f"[SIMBAD] astroquery.simbad import failed → SIMBAD unavailable: {e}")
    _HAS_SIMBAD = False


class InstrumentConfig:
    """
    Telescope and camera configuration
    Handles pixel scale, FOV calculation, and target resolution
    """

    def __init__(self, params, binning: Optional[int] = None):
        """
        Initialize instrument configuration

        Args:
            params: Parameters object from config module
            binning: Binning mode override (default: from params)
        """
        self.params = params

        # Telescope specifications (CDK500)
        self.telescope_name = "Planewave CDK500"
        self.aperture_mm = 500.0
        self.focal_length_mm = 3947.0
        self.focal_ratio = self.focal_length_mm / self.aperture_mm

        # Camera specifications (Moravian C3-61000)
        self.camera_name = "Moravian C3-61000"
        self.pix_size_um = 3.76
        self.sensor_w_mm = 36.01
        self.sensor_h_mm = 24.02
        self.sensor_nx_1x = 9576
        self.sensor_ny_1x = 6388

        # Binning mode
        if binning is not None:
            self.binning = int(binning)
        else:
            try:
                self.binning = int(float(getattr(params.P, "binning_default", 2) or 2))
            except:
                self.binning = 2

        # Calculate pixel scale and FOV
        self.pix_scale_1x = self._pixel_scale_arcsec(1)
        self.pix_scale_bin = self._pixel_scale_arcsec(self.binning)
        self.fov_w_deg = self._fov_deg(self.sensor_w_mm)
        self.fov_h_deg = self._fov_deg(self.sensor_h_mm)

        # Update parameter object with calculated values
        self.params.P.pixel_scale_arcsec = float(self.pix_scale_bin)
        self.params.P.telescope_focal_mm = float(self.focal_length_mm)
        self.params.P.camera_pixel_um = float(self.pix_size_um)
        self.params.P.binning_default = int(self.binning)

        # Apply arcsec-based FWHM conversions if specified
        self._apply_fwhm_conversions()

        # Target resolution
        self.targets_resolved = []
        self.primary_target = None
        self.primary_coord = None

    def _pixel_scale_arcsec(self, binning: int = 1) -> float:
        """Calculate pixel scale in arcseconds per pixel"""
        return 206.265 * self.pix_size_um * float(binning) / float(self.focal_length_mm)

    def _fov_deg(self, sensor_mm: float) -> float:
        """Calculate field of view in degrees"""
        return 57.2957795 * sensor_mm / float(self.focal_length_mm)

    def _apply_fwhm_conversions(self):
        """Convert FWHM parameters from arcsec to pixels if specified"""
        P = self.params.P

        # Convert fwhm_guess_arcsec to pixels
        arc = getattr(P, "fwhm_guess_arcsec", None)
        if arc is not None and np.isfinite(arc) and arc > 0:
            P.fwhm_seed_px = max(2.0, float(arc) / self.pix_scale_bin)
            P._fwhm_seed_from = "arcsec"

        # Convert fwhm_arcsec_min
        arcmin = getattr(P, "fwhm_arcsec_min", None)
        if arcmin is not None and np.isfinite(arcmin):
            P.fwhm_px_min = max(float(P.fwhm_px_min), float(arcmin) / self.pix_scale_bin)

        # Convert fwhm_arcsec_max
        arcmax = getattr(P, "fwhm_arcsec_max", None)
        if arcmax is not None and np.isfinite(arcmax):
            P.fwhm_px_max = min(float(P.fwhm_px_max), float(arcmax) / self.pix_scale_bin)

    def resolve_targets(self, target_names: Optional[list[str]] = None):
        """
        Resolve target coordinates using SIMBAD

        Args:
            target_names: List of target names to resolve (default: from params/targets.txt)
        """
        if not _HAS_SIMBAD:
            print("[SIMBAD] astroquery not available, skipping target resolution")
            return

        # Get target names from various sources
        raw_targets = []
        if target_names:
            raw_targets.extend(target_names)

        # From parameter file
        if hasattr(self.params.P, "_raw"):
            tn_param = self.params.P._raw.get("target_name", None)
            if tn_param and str(tn_param).strip():
                raw_targets.append(str(tn_param).strip())

        # From targets.txt
        target_list_path = self.params.P.data_dir / "targets.txt"
        if target_list_path.exists():
            for line in target_list_path.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if (not s) or s.startswith("#"):
                    continue
                raw_targets.append(s)

        # Default target
        if not raw_targets:
            raw_targets = ["M31"]

        # Remove duplicates
        seen = set()
        targets = []
        for name in raw_targets:
            if name not in seen:
                seen.add(name)
                targets.append(name)

        # Query SIMBAD
        Simbad.reset_votable_fields()
        Simbad.add_votable_fields("ra(d)", "dec(d)", "flux(V)", "otype")

        print("\n=== SIMBAD Target Resolution ===")
        for name in targets:
            try:
                res = Simbad.query_object(name)
                if res is None or len(res) == 0:
                    print(f"[SIMBAD] Not found: {name}")
                    continue

                ra_deg = float(res["RA_d"][0])
                dec_deg = float(res["DEC_d"][0])
                ra_str = str(res["RA"][0])
                dec_str = str(res["DEC"][0])
                vmag = float(res["FLUX_V"][0]) if "FLUX_V" in res.colnames else np.nan
                otype = str(res["OTYPE"][0]) if "OTYPE" in res.colnames else ""

                rec = dict(
                    name=name,
                    ra_deg=ra_deg,
                    dec_deg=dec_deg,
                    ra_str=ra_str,
                    dec_str=dec_str,
                    vmag=vmag,
                    otype=otype,
                )
                self.targets_resolved.append(rec)

                print(
                    f"[SIMBAD] {name:20s} → RA={ra_str}  DEC={dec_str}  "
                    f"({ra_deg:9.5f}°, {dec_deg:9.5f}°)  V~{vmag if np.isfinite(vmag) else 'n/a'}  [{otype}]"
                )

            except Exception as e:
                print(f"[SIMBAD] ERROR for '{name}': {e}")

        # Set primary target
        if self.targets_resolved:
            self.primary_target = self.targets_resolved[0]["name"]
            self.primary_coord = SkyCoord(
                self.targets_resolved[0]["ra_deg"],
                self.targets_resolved[0]["dec_deg"],
                unit="deg",
            )

            # Save results
            output_dir = step1_dir(self.params.P.result_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            out = output_dir / "targets_simbad.tsv"
            pd.DataFrame(self.targets_resolved).to_csv(out, sep="\t", index=False)
            print(f"→ SIMBAD results saved: {out}")
        else:
            print("No targets successfully resolved via SIMBAD")

    def print_summary(self):
        """Print instrument configuration summary"""
        print("\n=== Instrument Summary (CDK500 + C3-61000) ===")
        print(f"Telescope : {self.telescope_name}")
        print(f"  Aperture D = {self.aperture_mm:.1f} mm")
        print(f"  Focal length F = {self.focal_length_mm:.1f} mm  (f/{self.focal_ratio:.2f})")
        print(f"Camera    : {self.camera_name}")
        print(f"  Pixel size (1x1) = {self.pix_size_um:.3f} μm")
        print(
            f"  Sensor = {self.sensor_w_mm:.2f} × {self.sensor_h_mm:.2f} mm  "
            f"({self.sensor_nx_1x} × {self.sensor_ny_1x} px @1x1)"
        )
        print(f"Default binning = {self.binning}×{self.binning}")
        print(f"Pixel scale (1x1)  ≈ {self.pix_scale_1x:.6f} \" / px")
        print(
            f"Pixel scale ({self.binning}x{self.binning}) ≈ {self.pix_scale_bin:.6f} \" / px  "
            f"(pipeline pixel_scale_arcsec={self.params.P.pixel_scale_arcsec:.6f} \" / px)"
        )
        print(f"Full FOV (sensor, any binning) ≈ {self.fov_w_deg:.3f} × {self.fov_h_deg:.3f} deg")

        if self.primary_target is not None:
            if self.primary_coord is not None:
                ra_hms = self.primary_coord.ra.to_string(unit=u.hour, sep=":", precision=2)
                dec_dms = self.primary_coord.dec.to_string(
                    unit=u.deg, sep=":", precision=1, alwayssign=True
                )
                print(f"\nPRIMARY TARGET: {self.primary_target}  (RA={ra_hms}, DEC={dec_dms})")
            else:
                print(f"\nPRIMARY TARGET (name only): {self.primary_target}")
