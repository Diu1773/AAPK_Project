"""Scientific analysis modules"""

from .isochrone_fitter import (
    IsochroneFitter,
    FitMode,
    FitResult,
    FitBounds,
)

from .isochrone_fitter_v2 import (
    IsochroneFitterV2,
    GridScanResult,
)

__all__ = [
    'IsochroneFitter',
    'IsochroneFitterV2',
    'FitMode',
    'FitResult',
    'FitBounds',
    'GridScanResult',
]
