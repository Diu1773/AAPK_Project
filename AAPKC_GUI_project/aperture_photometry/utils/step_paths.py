"""
Step-specific output path helpers.

Each step writes to a dedicated subdirectory under result_dir,
matching AAPKL's directory layout for consistency.

Current UI workflow order:
  Step 4: Source Detection
  Step 5: Aperture Photometry
  Step 6: PSF Photometry (optional)
  Step 7: WCS Plate Solving
  Step 8: Reference Catalog Build
  Step 9: Star ID Matching
  Step10: Master ID Editor
  Step11: Zeropoint Calibration
  Step12: CMD Plot
  Step13: Isochrone Model

On-disk directory names (some are legacy names kept for compatibility):
  step4_detection/   – Step 4 Detection
  step5_aperture/    – Step 5 Aperture Photometry
  step6_psf/         – Step 6 PSF Photometry
  step7_wcs/         – Step 7 WCS Plate Solving
  step6_refbuild/    – Step 8 Reference Catalog Build  (legacy dirname)
  step7_idmatch/     – Step 9 Star ID Matching         (legacy dirname)
  step8_selection/   – Step10 Master ID Editor         (legacy dirname)
  step9_photometry/  – legacy forced-phot outputs
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, Path]

# ── On-disk directory names ───────────────────────────────────────────────────
# Steps 4-7: named to match UI step number
STEP4_DIRNAME          = "step4_detection"
STEP5_APERTURE_DIRNAME = "step5_aperture"
STEP6_PSF_DIRNAME      = "step6_psf"
STEP7_WCS_DIRNAME      = "step7_wcs"
# Steps 8-10: on-disk names are legacy (from old pipeline numbering)
LEGACY_STEP6_REFBUILD_DIRNAME  = "step6_refbuild"   # Step 8 RefBuild
LEGACY_STEP7_IDMATCH_DIRNAME   = "step7_idmatch"    # Step 9 IDMatch
LEGACY_STEP8_SELECTION_DIRNAME = "step8_selection"  # Step10 MasterID
# Steps 11-13
STEP11_ZP_DIRNAME      = "step11_zeropoint"
STEP12_CMD_DIRNAME     = "step12_cmd"
STEP13_ISO_DIRNAME     = "step13_isochrone"
# Old legacy fallback directories
LEGACY_STEP5_WCS_DIRNAME      = "step5_wcs"         # pre-step7_wcs WCS output
LEGACY_STEP7_WCS_DIRNAME      = STEP7_WCS_DIRNAME   # compatibility alias during renumbering
LEGACY_STEP9_PHOT_DIRNAME     = "step9_photometry"  # old forced-phot
LEGACY_STEP5_REFBUILD_DIRNAME = "step5_refbuild"     # very old refbuild
LEGACY_STEP7_REFBUILD_DIRNAME = "step7_refbuild"     # old refbuild fallback


def _as_path(result_dir: Optional[PathLike]) -> Path:
    if isinstance(result_dir, Path):
        return result_dir
    if result_dir is None:
        return Path.cwd()
    s = str(result_dir).strip()
    return Path(s) if s else Path.cwd()


def step1_dir(result_dir: Optional[PathLike]) -> Path:
    return _as_path(result_dir)


def step2_dir(result_dir: Optional[PathLike]) -> Path:
    return _as_path(result_dir)


def step2_cropped_dir(result_dir: Optional[PathLike]) -> Path:
    return _as_path(result_dir) / "cropped"

def crop_rect_path(result_dir: Optional[PathLike]) -> Path:
    return _as_path(result_dir) / "crop_rect.json"


def crop_is_active(result_dir: Optional[PathLike]) -> bool:
    return crop_rect_path(result_dir).exists()


def step4_dir(result_dir: Optional[PathLike]) -> Path:
    """Step 4: Source Detection outputs."""
    return _as_path(result_dir) / STEP4_DIRNAME


# ── New pipeline step directories ───────────────────────────────────────────

def step5_aperture_dir(result_dir: Optional[PathLike]) -> Path:
    """Step 5: Aperture photometry outputs (det_uid based)."""
    return _as_path(result_dir) / STEP5_APERTURE_DIRNAME


def step6_psf_dir(result_dir: Optional[PathLike]) -> Path:
    """Step 6: PSF photometry outputs (optional, skippable)."""
    return _as_path(result_dir) / STEP6_PSF_DIRNAME


def step8_refbuild_dir(result_dir: Optional[PathLike]) -> Path:
    """Step 8 RefBuild outputs → step6_refbuild/ (on-disk legacy dirname)."""
    return step6_dir(result_dir)


def step9_idmatch_dir(result_dir: Optional[PathLike]) -> Path:
    """Step 9 IDMatch outputs → step7_idmatch/ (on-disk legacy dirname)."""
    return step7_dir(result_dir)


def step10_selection_dir(result_dir: Optional[PathLike]) -> Path:
    """Step 10 Master ID Editor outputs → step8_selection/ (on-disk legacy dirname)."""
    return step8_dir(result_dir)


# ── Deprecated aliases (wrong step numbers; kept for one-version compat) ─────
def step7_refbuild_dir(result_dir: Optional[PathLike]) -> Path:
    """Deprecated: use step8_refbuild_dir."""
    return step8_refbuild_dir(result_dir)


def step8_idmatch_dir(result_dir: Optional[PathLike]) -> Path:
    """Deprecated: use step9_idmatch_dir."""
    return step9_idmatch_dir(result_dir)


def step9_selection_dir(result_dir: Optional[PathLike]) -> Path:
    """Deprecated: use step10_selection_dir."""
    return step10_selection_dir(result_dir)


def step10_wcs_gaia_dir(result_dir: Optional[PathLike]) -> Path:
    """Deprecated: use step7_wcs_dir."""
    return step7_wcs_dir(result_dir)


def step7_wcs_dir(result_dir: Optional[PathLike]) -> Path:
    """Step 7: WCS Plate Solving outputs (UI-matched naming)."""
    return _as_path(result_dir) / STEP7_WCS_DIRNAME


def step11_zp_dir(result_dir: Optional[PathLike]) -> Path:
    """Step 11: Zeropoint calibration outputs."""
    root = _as_path(result_dir)
    new = root / STEP11_ZP_DIRNAME
    # fallback: if new dir doesn't exist but old (root) has files, keep root
    if not new.exists() and (root / "zp_fit_coefficients.csv").exists():
        return root
    return new


def step12_cmd_dir(result_dir: Optional[PathLike]) -> Path:
    """Step 12: CMD plot outputs."""
    return _as_path(result_dir) / STEP12_CMD_DIRNAME


def step13_iso_dir(result_dir: Optional[PathLike]) -> Path:
    """Step 13: Isochrone outputs."""
    return _as_path(result_dir) / STEP13_ISO_DIRNAME


# ── Legacy functions (old pipeline, preserved for 1-version compat) ─────────

def step5_dir(result_dir: Optional[PathLike]) -> Path:
    """WCS outputs: returns step7_wcs/ if it exists, else falls back to legacy step5_wcs/."""
    root = _as_path(result_dir)
    new = root / STEP7_WCS_DIRNAME
    if new.exists():
        return new
    return root / LEGACY_STEP5_WCS_DIRNAME


def step6_dir(result_dir: Optional[PathLike]) -> Path:
    """Step 8 RefBuild on-disk directory → step6_refbuild/."""
    return _as_path(result_dir) / LEGACY_STEP6_REFBUILD_DIRNAME


def step7_dir(result_dir: Optional[PathLike]) -> Path:
    """Step 9 IDMatch on-disk directory → step7_idmatch/."""
    return _as_path(result_dir) / LEGACY_STEP7_IDMATCH_DIRNAME


def step8_dir(result_dir: Optional[PathLike]) -> Path:
    """Step 10 Master ID Editor on-disk directory → step8_selection/."""
    return _as_path(result_dir) / LEGACY_STEP8_SELECTION_DIRNAME


def step9_dir(result_dir: Optional[PathLike]) -> Path:
    """Legacy forced photometry directory → step9_photometry/."""
    return _as_path(result_dir) / LEGACY_STEP9_PHOT_DIRNAME


def legacy_step5_wcs_dir(result_dir: Optional[PathLike]) -> Path:
    """Alias: old Step5 WCS directory."""
    root = _as_path(result_dir)
    candidate = root / LEGACY_STEP5_WCS_DIRNAME
    return candidate if candidate.exists() else root


def legacy_step6_refbuild_dir(result_dir: Optional[PathLike]) -> Path:
    """Alias: old Step6 RefBuild directory."""
    root = _as_path(result_dir)
    candidate = root / LEGACY_STEP6_REFBUILD_DIRNAME
    return candidate if candidate.exists() else root


def legacy_step7_idmatch_dir(result_dir: Optional[PathLike]) -> Path:
    """Alias: old Step7 IDMatch directory."""
    root = _as_path(result_dir)
    candidate = root / LEGACY_STEP7_IDMATCH_DIRNAME
    return candidate if candidate.exists() else root


def legacy_step8_selection_dir(result_dir: Optional[PathLike]) -> Path:
    """Alias: old Step8 MasterID directory."""
    root = _as_path(result_dir)
    candidate = root / LEGACY_STEP8_SELECTION_DIRNAME
    return candidate if candidate.exists() else root


def legacy_step9_phot_dir(result_dir: Optional[PathLike]) -> Path:
    """Alias: old Step9 photometry directory."""
    root = _as_path(result_dir)
    candidate = root / LEGACY_STEP9_PHOT_DIRNAME
    return candidate if candidate.exists() else root


def legacy_step5_refbuild_dir(result_dir: Optional[PathLike]) -> Path:
    root = _as_path(result_dir)
    candidate = root / LEGACY_STEP5_REFBUILD_DIRNAME
    return candidate if candidate.exists() else root


def legacy_step7_wcs_dir(result_dir: Optional[PathLike]) -> Path:
    root = _as_path(result_dir)
    candidate = root / LEGACY_STEP7_WCS_DIRNAME
    return candidate if candidate.exists() else root


def legacy_step7_refbuild_dir(result_dir: Optional[PathLike]) -> Path:
    root = _as_path(result_dir)
    candidate = root / LEGACY_STEP7_REFBUILD_DIRNAME
    return candidate if candidate.exists() else root


# ── Downstream step directories ──────────────────────────────────────────────

def step11_dir(result_dir: Optional[PathLike]) -> Path:
    """Step 11 ZP calibration — delegates to step11_zp_dir (with legacy fallback)."""
    return step11_zp_dir(result_dir)


def step11_extinction_dir(result_dir: Optional[PathLike]) -> Path:
    return _as_path(result_dir) / "tool_extinction"


def step12_dir(result_dir: Optional[PathLike]) -> Path:
    """Step 12 CMD plot — delegates to step12_cmd_dir."""
    return step12_cmd_dir(result_dir)


def step13_dir(result_dir: Optional[PathLike]) -> Path:
    """Step 13 Isochrone — delegates to step13_iso_dir."""
    return step13_iso_dir(result_dir)


__all__ = [
    # Steps 1–4
    "step1_dir",
    "step2_dir",
    "step2_cropped_dir",
    "crop_rect_path",
    "crop_is_active",
    "step4_dir",
    # New pipeline (UI step number matched)
    "step5_aperture_dir",
    "step6_psf_dir",
    "step7_wcs_dir",
    "step8_refbuild_dir",
    "step9_idmatch_dir",
    "step10_selection_dir",
    "step11_zp_dir",
    "step12_cmd_dir",
    "step13_iso_dir",
    # Deprecated aliases (wrong step numbers)
    "step7_refbuild_dir",
    "step8_idmatch_dir",
    "step9_selection_dir",
    "step10_wcs_gaia_dir",
    # Legacy (old pipeline, 1-version compat)
    "step5_dir",
    "step6_dir",
    "step7_dir",
    "step8_dir",
    "step9_dir",
    "legacy_step5_wcs_dir",
    "legacy_step6_refbuild_dir",
    "legacy_step7_idmatch_dir",
    "legacy_step8_selection_dir",
    "legacy_step9_phot_dir",
    "legacy_step5_refbuild_dir",
    "legacy_step7_wcs_dir",
    "legacy_step7_refbuild_dir",
    # Downstream
    "step11_dir",
    "step11_extinction_dir",
    "step12_dir",
    "step13_dir",
]
