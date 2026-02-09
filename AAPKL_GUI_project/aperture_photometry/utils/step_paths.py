"""Common step output paths for the workflow."""

from __future__ import annotations

from pathlib import Path


STEP1_DIRNAME = "step1_file_selection"
STEP2_DIRNAME = "step2_crop"
STEP4_DIRNAME = "step4_detection"
STEP5_DIRNAME = "step5_wcs"
STEP6_DIRNAME = "step6_refbuild"
STEP7_DIRNAME = "step7_idmatch"
STEP8_DIRNAME = "step8_selection"
STEP9_DIRNAME = "step9_photometry"
STEP10_DIRNAME = "step10_lightcurve"
STEP11_DIRNAME = "step11_detrend_merge"
STEP12_PERIOD_DIRNAME = "step12_period_analysis"
# Legacy directory names (pre-refactor)
LEGACY_STEP5_REFBUILD_DIRNAME = "step5_refbuild"
LEGACY_STEP6_IDMATCH_DIRNAME = "step6_idmatch"
LEGACY_STEP7_WCS_DIRNAME = "step7_wcs"
LEGACY_STEP7_REFBUILD_DIRNAME = "step7_refbuild"
# Legacy directory names for step renumbering (step11→10, step12→11, step13→12)
LEGACY_STEP11_LIGHTCURVE_DIRNAME = "step11_lightcurve"
LEGACY_STEP12_DETREND_DIRNAME = "step12_detrend_merge"
LEGACY_STEP13_PERIOD_DIRNAME = "step13_period_analysis"
# Legacy directory names for backward compatibility
LEGACY_STEP11_ZEROPOINT_DIRNAME = "step11_zeropoint"  # Old step11 (now tool)
# Tool directories
TOOL_EXTINCTION_DIRNAME = "tool_extinction"


def step_dir(result_dir: Path, dirname: str) -> Path:
    """Return a step directory path under the result directory."""
    return Path(result_dir) / dirname


def step1_dir(result_dir: Path) -> Path:
    return step_dir(result_dir, STEP1_DIRNAME)


def step2_dir(result_dir: Path) -> Path:
    return step_dir(result_dir, STEP2_DIRNAME)


def step2_cropped_dir(result_dir: Path) -> Path:
    return step2_dir(result_dir) / "cropped"


def crop_rect_path(result_dir: Path) -> Path:
    return step2_dir(result_dir) / "crop_rect.json"


def crop_is_active(result_dir: Path) -> bool:
    return crop_rect_path(result_dir).exists()


def step4_dir(result_dir: Path) -> Path:
    return step_dir(result_dir, STEP4_DIRNAME)


def step5_dir(result_dir: Path) -> Path:
    return step_dir(result_dir, STEP5_DIRNAME)


def step6_dir(result_dir: Path) -> Path:
    return step_dir(result_dir, STEP6_DIRNAME)


def step7_dir(result_dir: Path) -> Path:
    return step_dir(result_dir, STEP7_DIRNAME)


def legacy_step5_refbuild_dir(result_dir: Path) -> Path:
    return step_dir(result_dir, LEGACY_STEP5_REFBUILD_DIRNAME)


def legacy_step6_idmatch_dir(result_dir: Path) -> Path:
    return step_dir(result_dir, LEGACY_STEP6_IDMATCH_DIRNAME)


def legacy_step7_wcs_dir(result_dir: Path) -> Path:
    return step_dir(result_dir, LEGACY_STEP7_WCS_DIRNAME)


def legacy_step7_refbuild_dir(result_dir: Path) -> Path:
    return step_dir(result_dir, LEGACY_STEP7_REFBUILD_DIRNAME)


def step8_dir(result_dir: Path) -> Path:
    return step_dir(result_dir, STEP8_DIRNAME)


def step9_dir(result_dir: Path) -> Path:
    return step_dir(result_dir, STEP9_DIRNAME)


def step10_dir(result_dir: Path) -> Path:
    """Step 10: Light Curve Builder (was step 11)"""
    p = step_dir(result_dir, STEP10_DIRNAME)
    if not p.exists():
        legacy = step_dir(result_dir, LEGACY_STEP11_LIGHTCURVE_DIRNAME)
        if legacy.exists():
            return legacy
    return p


def step11_dir(result_dir: Path) -> Path:
    """Step 11: Detrend & Night Merge (was step 12)"""
    p = step_dir(result_dir, STEP11_DIRNAME)
    if not p.exists():
        legacy = step_dir(result_dir, LEGACY_STEP12_DETREND_DIRNAME)
        if legacy.exists():
            return legacy
    return p


def step12_period_dir(result_dir: Path) -> Path:
    """Step 12: Period Analysis (was step 13)"""
    p = step_dir(result_dir, STEP12_PERIOD_DIRNAME)
    if not p.exists():
        legacy = step_dir(result_dir, LEGACY_STEP13_PERIOD_DIRNAME)
        if legacy.exists():
            return legacy
    return p


# Tool directory functions
def tool_extinction_dir(result_dir: Path) -> Path:
    """Extinction & Zeropoint Tool output directory"""
    return step_dir(result_dir, TOOL_EXTINCTION_DIRNAME)


# Legacy functions for backward compatibility
def legacy_step11_zeropoint_dir(result_dir: Path) -> Path:
    """Legacy: Old step11 zeropoint directory (now tool_extinction)"""
    return step_dir(result_dir, LEGACY_STEP11_ZEROPOINT_DIRNAME)


def legacy_step11_extinction_dir(result_dir: Path) -> Path:
    """Legacy: Old step11 extinction subdirectory. Now checks tool_extinction first."""
    # First check new tool directory
    new_path = tool_extinction_dir(result_dir)
    if new_path.exists():
        return new_path
    # Fall back to legacy path
    legacy_path = legacy_step11_zeropoint_dir(result_dir) / "step11_extinction"
    return legacy_path


