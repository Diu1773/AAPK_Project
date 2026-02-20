"""
Step-specific output path helpers.

Each step writes to a dedicated subdirectory under result_dir,
matching AAPKL's directory layout for consistency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, Path]

# Step directory names
STEP4_DIRNAME = "step4_detection"
STEP5_DIRNAME = "step5_wcs"
STEP6_DIRNAME = "step6_refbuild"
STEP7_DIRNAME = "step7_idmatch"
STEP8_DIRNAME = "step8_selection"
STEP9_DIRNAME = "step9_photometry"

# Legacy directory names (pre-refactor)
LEGACY_STEP5_REFBUILD_DIRNAME = "step5_refbuild"
LEGACY_STEP7_WCS_DIRNAME = "step7_wcs"
LEGACY_STEP7_REFBUILD_DIRNAME = "step7_refbuild"


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
    """Step 4 output directory with compatibility fallback."""
    root = _as_path(result_dir)
    candidate = root / STEP4_DIRNAME
    return candidate if candidate.exists() else root


def step5_dir(result_dir: Optional[PathLike]) -> Path:
    """Step 5: WCS Plate Solving outputs."""
    return _as_path(result_dir) / STEP5_DIRNAME


def step6_dir(result_dir: Optional[PathLike]) -> Path:
    """Step 6: Reference Build outputs."""
    return _as_path(result_dir) / STEP6_DIRNAME


def step7_dir(result_dir: Optional[PathLike]) -> Path:
    """Step 7: Star ID Matching outputs."""
    return _as_path(result_dir) / STEP7_DIRNAME


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


def step8_dir(result_dir: Optional[PathLike]) -> Path:
    """Step 8: Master ID Editor outputs."""
    return _as_path(result_dir) / STEP8_DIRNAME


def step9_dir(result_dir: Optional[PathLike]) -> Path:
    """Step 9: Forced Photometry outputs."""
    return _as_path(result_dir) / STEP9_DIRNAME


def step11_dir(result_dir: Optional[PathLike]) -> Path:
    return _as_path(result_dir)


def step11_extinction_dir(result_dir: Optional[PathLike]) -> Path:
    return _as_path(result_dir) / "extinction"


def step12_dir(result_dir: Optional[PathLike]) -> Path:
    return _as_path(result_dir)


def step13_dir(result_dir: Optional[PathLike]) -> Path:
    return _as_path(result_dir)


__all__ = [
    "step1_dir",
    "step2_dir",
    "step2_cropped_dir",
    "crop_rect_path",
    "crop_is_active",
    "step4_dir",
    "step5_dir",
    "step6_dir",
    "step7_dir",
    "legacy_step5_refbuild_dir",
    "legacy_step7_wcs_dir",
    "legacy_step7_refbuild_dir",
    "step8_dir",
    "step9_dir",
    "step11_dir",
    "step11_extinction_dir",
    "step12_dir",
    "step13_dir",
]
