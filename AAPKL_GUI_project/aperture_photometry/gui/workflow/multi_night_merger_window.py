"""Compatibility wrapper for the tools-native multi-night merger window."""

from __future__ import annotations

from ..tools.multi_night_merger_tool import (
    MultiNightMergerWindow,
    _MergedFileManagerProxy,
    _MergedParamsProxy,
)

__all__ = [
    "MultiNightMergerWindow",
    "_MergedFileManagerProxy",
    "_MergedParamsProxy",
]
