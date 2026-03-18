"""Common step output paths for the workflow.

Pipeline order (after refactoring):
  Step 1: File Selection
  Step 2: Image Crop
  Step 3: Sky Preview & QC
  Step 4: Source Detection
  Step 5: Aperture Photometry   (was step 9, forced photometry)
  Step 6: WCS Plate Solving     (was step 5)
  Step 7: Reference Build       (was step 6)
  Step 8: Star ID Matching      (was step 7)
  Step 9: Target/Comp Selection (was step 8, master ID editor)
  Step 10: Light Curve Builder
  Step 11: Detrend & Night Merge
  Step 12: Period Analysis
"""

from __future__ import annotations

from pathlib import Path


# ── Current directory names ──────────────────────────────────────────
STEP1_DIRNAME = "step1_file_selection"
STEP2_DIRNAME = "step2_crop"
STEP4_DIRNAME = "step4_detection"
STEP5_DIRNAME = "step5_photometry"      # Full aperture photometry (was step9_photometry)
STEP6_DIRNAME = "step6_wcs"             # WCS plate solving (was step5_wcs)
STEP7_DIRNAME = "step7_refbuild"        # Reference catalog build (was step6_refbuild)
STEP8_DIRNAME = "step8_idmatch"         # Star ID matching (was step7_idmatch)
STEP9_DIRNAME = "step9_selection"       # Target/comp selection (was step8_selection)
STEP10_DIRNAME = "step10_lightcurve"
STEP11_DIRNAME = "step11_detrend_merge"
STEP12_PERIOD_DIRNAME = "step12_period_analysis"

# ── Legacy directory names (v2 reorder: phot moved before WCS) ───────
LEGACY_STEP5_WCS_DIRNAME = "step5_wcs"
LEGACY_STEP6_REFBUILD_DIRNAME_V2 = "step6_refbuild"
LEGACY_STEP7_IDMATCH_DIRNAME_V2 = "step7_idmatch"
LEGACY_STEP8_SELECTION_DIRNAME = "step8_selection"
LEGACY_STEP9_PHOTOMETRY_DIRNAME = "step9_photometry"

# ── Legacy directory names (v1 reorder: ref↔idmatch swap) ───────────
LEGACY_STEP5_REFBUILD_DIRNAME = "step5_refbuild"
LEGACY_STEP6_IDMATCH_DIRNAME = "step6_idmatch"
LEGACY_STEP7_WCS_DIRNAME = "step7_wcs"
LEGACY_STEP7_REFBUILD_DIRNAME = "step7_refbuild"

# ── Legacy directory names (step renumbering: step11→10 etc.) ────────
LEGACY_STEP11_LIGHTCURVE_DIRNAME = "step11_lightcurve"
LEGACY_STEP12_DETREND_DIRNAME = "step12_detrend_merge"
LEGACY_STEP13_PERIOD_DIRNAME = "step13_period_analysis"
LEGACY_STEP11_ZEROPOINT_DIRNAME = "step11_zeropoint"  # Old step11 (now tool)

# ── Tool directories ─────────────────────────────────────────────────
TOOL_EXTINCTION_DIRNAME = "tool_extinction"


# ── Generic helper ───────────────────────────────────────────────────

def step_dir(result_dir: Path, dirname: str) -> Path:
    """Return a step directory path under the result directory."""
    return Path(result_dir) / dirname


# ── Step 1–4 (unchanged) ────────────────────────────────────────────

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


# ── Step 5: Aperture Photometry (was step 9) ────────────────────────

def step5_dir(result_dir: Path) -> Path:
    """Step 5: Full Aperture Photometry output."""
    return step_dir(result_dir, STEP5_DIRNAME)


def step5_photometry_dir(result_dir: Path) -> Path:
    """Preferred explicit alias for Step 5 photometry outputs."""
    return step5_dir(result_dir)


def step5_aperture_dir(result_dir: Path) -> Path:
    """Compatibility alias for Step 5 aperture photometry outputs."""
    return step5_photometry_dir(result_dir)


# ── Step 6: WCS Plate Solving (was step 5) ──────────────────────────

def step6_dir(result_dir: Path) -> Path:
    """Step 6: WCS Plate Solving output."""
    return step_dir(result_dir, STEP6_DIRNAME)


def step6_wcs_dir(result_dir: Path) -> Path:
    """Preferred explicit alias for Step 6 WCS outputs."""
    return step6_dir(result_dir)


# ── Step 7: Reference Build (was step 6) ────────────────────────────

def step7_dir(result_dir: Path) -> Path:
    """Step 7: Reference Catalog Build output."""
    return step_dir(result_dir, STEP7_DIRNAME)


def step7_refbuild_dir(result_dir: Path) -> Path:
    """Preferred explicit alias for Step 7 reference-build outputs."""
    return step7_dir(result_dir)


# ── Step 8: Star ID Matching (was step 7) ───────────────────────────

def step8_dir(result_dir: Path) -> Path:
    """Step 8: Star ID Matching output."""
    return step_dir(result_dir, STEP8_DIRNAME)


def step8_idmatch_dir(result_dir: Path) -> Path:
    """Preferred explicit alias for Step 8 star-ID matching outputs."""
    return step8_dir(result_dir)


# ── Step 9: Target/Comp Selection (was step 8) ─────────────────────

def step9_dir(result_dir: Path) -> Path:
    """Step 9: Target/Comparison Selection (Master ID Editor) output."""
    return step_dir(result_dir, STEP9_DIRNAME)


def step9_selection_dir(result_dir: Path) -> Path:
    """Preferred explicit alias for Step 9 target/comparison selection outputs."""
    return step9_dir(result_dir)


# ── Step 10–12 ───────────────────────────────────────────────────────

def step10_dir(result_dir: Path) -> Path:
    """Step 10: Light Curve Builder (was step 11)"""
    return step_dir(result_dir, STEP10_DIRNAME)


def step11_dir(result_dir: Path) -> Path:
    """Step 11: Detrend & Night Merge (was step 12)"""
    return step_dir(result_dir, STEP11_DIRNAME)


def step12_period_dir(result_dir: Path) -> Path:
    """Step 12: Period Analysis (was step 13)"""
    return step_dir(result_dir, STEP12_PERIOD_DIRNAME)


# ── Legacy convenience functions (for very old data) ─────────────────

def legacy_step5_refbuild_dir(result_dir: Path) -> Path:
    return step_dir(result_dir, LEGACY_STEP5_REFBUILD_DIRNAME)


def legacy_step6_idmatch_dir(result_dir: Path) -> Path:
    return step_dir(result_dir, LEGACY_STEP6_IDMATCH_DIRNAME)


def legacy_step7_wcs_dir(result_dir: Path) -> Path:
    return step_dir(result_dir, LEGACY_STEP7_WCS_DIRNAME)


def legacy_step7_refbuild_dir(result_dir: Path) -> Path:
    return step_dir(result_dir, LEGACY_STEP7_REFBUILD_DIRNAME)


def legacy_step5_wcs_dir(result_dir: Path) -> Path:
    """Legacy: step5_wcs/ (WCS was step 5 before v2 reorder)"""
    return step_dir(result_dir, LEGACY_STEP5_WCS_DIRNAME)


def legacy_step9_photometry_dir(result_dir: Path) -> Path:
    """Legacy: step9_photometry/ (photometry was step 9 before v2 reorder)"""
    return step_dir(result_dir, LEGACY_STEP9_PHOTOMETRY_DIRNAME)


# ── Step 11 helper paths (unchanged) ─────────────────────────────────

def step11_current_lc_path(result_dir: Path, target_id: int) -> Path:
    return step11_dir(result_dir) / f"lightcurve_ID{int(target_id)}_current.csv"


def step11_current_params_path(result_dir: Path, target_id: int) -> Path:
    return step11_dir(result_dir) / f"params_ID{int(target_id)}_current.csv"


def step11_current_summary_path(result_dir: Path, target_id: int) -> Path:
    return step11_dir(result_dir) / f"summary_ID{int(target_id)}_current.txt"


def step11_current_plot_path(result_dir: Path, target_id: int) -> Path:
    return step11_dir(result_dir) / f"plot_ID{int(target_id)}_current.png"


def step11_current_meta_path(result_dir: Path, target_id: int) -> Path:
    return step11_dir(result_dir) / f"result_ID{int(target_id)}_current.json"


def step11_current_global_zp_path(result_dir: Path, target_id: int) -> Path:
    return step11_dir(result_dir) / f"global_zp_ID{int(target_id)}_current.csv"


def step11_current_global_mean_path(result_dir: Path, target_id: int) -> Path:
    return step11_dir(result_dir) / f"global_mean_ID{int(target_id)}_current.csv"


def step11_current_global_diag_path(result_dir: Path, target_id: int) -> Path:
    return step11_dir(result_dir) / f"global_diagnostics_ID{int(target_id)}_current.json"


def step11_history_dir(result_dir: Path) -> Path:
    return step11_dir(result_dir) / "_history"


def load_detrend_preference(result_dir: Path, target_id: int | None = None) -> str | None:
    """Read the adopted correction mode from step11 meta JSON.

    Returns mode string ('global', 'color', 'offset') or None.
    """
    import json as _json
    if target_id is not None:
        meta = step11_current_meta_path(result_dir, target_id)
        if meta.exists():
            try:
                data = _json.loads(meta.read_text(encoding="utf-8"))
                return data.get("mode", "").lower() or None
            except Exception:
                pass
    # Fallback: scan for any meta file
    s11 = step11_dir(result_dir)
    if s11.exists():
        for mp in sorted(s11.glob("result_ID*_current.json")):
            try:
                data = _json.loads(mp.read_text(encoding="utf-8"))
                return data.get("mode", "").lower() or None
            except Exception:
                continue
    return None


def find_best_lightcurve_csv(result_dir: Path, target_id: int | None = None) -> Path | None:
    """Resolve the preferred light curve CSV for downstream analysis.

    Preference order:
    1. Step 11 current result
    2. Legacy Step 11 mode-specific result (global > color > offset)
    3. Step 10 raw combined/raw
    """
    step10_out = step10_dir(result_dir)
    step11_out = step11_dir(result_dir)
    candidates: list[Path] = []

    if target_id is not None:
        candidates.append(step11_current_lc_path(result_dir, target_id))
        for mode in ("global", "color", "offset"):
            candidates.append(step11_out / f"lightcurve_ID{int(target_id)}_{mode}.csv")
        candidates.append(step10_out / f"lightcurve_combined_ID{int(target_id)}_raw.csv")
        candidates.append(step10_out / f"lightcurve_ID{int(target_id)}_raw.csv")
    else:
        if step11_out.exists():
            candidates.extend(sorted(step11_out.glob("lightcurve_ID*_current.csv"), reverse=True))
            for mode in ("global", "color", "offset"):
                candidates.extend(sorted(step11_out.glob(f"lightcurve_ID*_{mode}.csv"), reverse=True))
        if step10_out.exists():
            candidates.extend(sorted(step10_out.glob("lightcurve_combined_ID*_raw.csv"), reverse=True))
            candidates.extend(sorted(step10_out.glob("lightcurve_ID*_raw.csv"), reverse=True))

    for d in (step11_out, step10_out):
        if d.exists():
            for f in sorted(d.glob("lightcurve_*.csv"), reverse=True):
                if f not in candidates:
                    candidates.append(f)

    for cand in candidates:
        if cand.exists():
            return cand
    return None


# ── Tool directory functions ──────────────────────────────────────────

def tool_extinction_dir(result_dir: Path) -> Path:
    """Extinction & Zeropoint Tool output directory"""
    return step_dir(result_dir, TOOL_EXTINCTION_DIRNAME)


def legacy_step11_zeropoint_dir(result_dir: Path) -> Path:
    """Legacy: Old step11 zeropoint directory (now tool_extinction)"""
    return step_dir(result_dir, LEGACY_STEP11_ZEROPOINT_DIRNAME)


def legacy_step11_extinction_dir(result_dir: Path) -> Path:
    """Legacy: Old step11 extinction subdirectory. Now checks tool_extinction first."""
    new_path = tool_extinction_dir(result_dir)
    if new_path.exists():
        return new_path
    legacy_path = legacy_step11_zeropoint_dir(result_dir) / "step11_extinction"
    return legacy_path
