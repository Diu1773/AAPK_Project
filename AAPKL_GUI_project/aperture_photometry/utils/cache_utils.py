"""Shared file-signature and WCS-cache helpers used by workflow steps."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List


def norm_path_key(path_value) -> str:
    if path_value is None:
        return ""
    try:
        s = str(path_value).strip().replace("\\", "/")
    except Exception:
        s = str(path_value).replace("\\", "/")
    if len(s) >= 3 and s[1] == ":" and s[2] == "/" and s[0].isalpha():
        s = f"/mnt/{s[0].lower()}/{s[3:]}"
    while "//" in s:
        s = s.replace("//", "/")
    if len(s) > 1 and s.endswith("/"):
        s = s[:-1]
    return s.lower()


def build_file_signature(path: Path, *, use_cropped: bool) -> dict:
    sig = {
        "source_path": norm_path_key(path),
        "source_use_cropped": bool(use_cropped),
        "source_size": None,
        "source_mtime_ns": None,
    }
    try:
        st = path.stat()
        sig["source_size"] = int(st.st_size)
        sig["source_mtime_ns"] = int(st.st_mtime_ns)
    except Exception:
        pass
    return sig


def file_signature_matches(saved_sig: dict, current_sig: dict) -> bool:
    if not isinstance(saved_sig, dict) or not isinstance(current_sig, dict):
        return False
    if bool(saved_sig.get("source_use_cropped", None)) != bool(current_sig.get("source_use_cropped", None)):
        return False
    if norm_path_key(saved_sig.get("source_path", "")) != norm_path_key(current_sig.get("source_path", "")):
        return False
    try:
        saved_mtime_ns = int(saved_sig.get("source_mtime_ns"))
        curr_mtime_ns = int(current_sig.get("source_mtime_ns"))
    except Exception:
        return False
    return saved_mtime_ns == curr_mtime_ns


def file_signature_matches_relaxed(saved_sig: dict, current_sig: dict) -> bool:
    """Accept mtime drift when path/crop/size are stable.

    Step5 can update FITS headers in-place after solving, which changes mtime
    without invalidating detection XY. In that case, accept cache when path,
    crop mode, and file size still match.
    """
    if file_signature_matches(saved_sig, current_sig):
        return True
    if not isinstance(saved_sig, dict) or not isinstance(current_sig, dict):
        return False
    if bool(saved_sig.get("source_use_cropped", None)) != bool(current_sig.get("source_use_cropped", None)):
        return False
    if norm_path_key(saved_sig.get("source_path", "")) != norm_path_key(current_sig.get("source_path", "")):
        return False
    try:
        saved_size = int(saved_sig.get("source_size"))
        curr_size = int(current_sig.get("source_size"))
    except Exception:
        return False
    if saved_size <= 0 or curr_size <= 0:
        return False
    return saved_size == curr_size


def astap_wcs_candidates(fits_path: Path) -> List[Path]:
    return [
        fits_path.with_suffix(".wcs"),
        Path(str(fits_path) + ".wcs"),
        fits_path.parent / (fits_path.stem + ".wcs"),
        fits_path.parent / (fits_path.name + ".wcs"),
    ]


def parse_astap_wcs_file(wcs_path: Path) -> dict:
    d: Dict[str, object] = {}
    try:
        lines = wcs_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return d
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        if "/" in s:
            s = s.split("/", 1)[0].strip()
        if "=" not in s:
            continue
        key, val = [t.strip() for t in s.split("=", 1)]
        if not key:
            continue
        if val.startswith("'") and val.endswith("'"):
            d[key] = val.strip("'")
            continue
        try:
            if "." in val or "E" in val.upper():
                d[key] = float(val)
            else:
                d[key] = int(val)
        except Exception:
            d[key] = val
    return d
