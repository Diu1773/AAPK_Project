"""
Step-specific output path helpers.

These helpers keep output locations consistent with documentation:
most steps write directly under result_dir, with a few dedicated subfolders.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, Path]


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


def step5_dir(result_dir: Optional[PathLike]) -> Path:
    return _as_path(result_dir)


def step6_dir(result_dir: Optional[PathLike]) -> Path:
    return _as_path(result_dir)


def step7_dir(result_dir: Optional[PathLike]) -> Path:
    return _as_path(result_dir)


def step8_dir(result_dir: Optional[PathLike]) -> Path:
    return _as_path(result_dir)


def step9_dir(result_dir: Optional[PathLike]) -> Path:
    return _as_path(result_dir)


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
    "step5_dir",
    "step6_dir",
    "step7_dir",
    "step8_dir",
    "step9_dir",
    "step11_dir",
    "step11_extinction_dir",
    "step12_dir",
    "step13_dir",
]
