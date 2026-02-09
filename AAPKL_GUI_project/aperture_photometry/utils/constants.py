from dataclasses import dataclass


@dataclass(frozen=True)
class GUIConstants:
    """Constants for GUI elements."""

    # Default log window size
    LOG_WINDOW_WIDTH: int = 800
    LOG_WINDOW_HEIGHT: int = 400

    # Progress update interval (ms)
    PROGRESS_UPDATE_INTERVAL: int = 100

    # Table column widths
    DEFAULT_COL_WIDTH_NARROW: int = 80
    DEFAULT_COL_WIDTH_MEDIUM: int = 120
    DEFAULT_COL_WIDTH_WIDE: int = 200


GUI = GUIConstants()


# =============================================================================
# Parallel Processing Constants
# =============================================================================


@dataclass(frozen=True)
class ParallelConstants:
    """Constants for parallel/concurrent processing."""

    # Default number of workers (0 = auto, use CPU count)
    DEFAULT_MAX_WORKERS: int = 0

    # Maximum workers cap (safety limit)
    MAX_WORKERS_CAP: int = 16

    # Minimum workers
    MIN_WORKERS: int = 1

    # Default timeout per task (seconds)
    DEFAULT_TASK_TIMEOUT: float = 300.0

    # Batch size for progress updates
    PROGRESS_BATCH_SIZE: int = 1


PARALLEL = ParallelConstants()


def get_parallel_workers(params=None) -> int:
    """
    Get parallel worker count from params or auto-detect.

    Central function for all steps to use consistent parallel settings.

    Args:
        params: Parameter object with P attribute, or None for auto-detect

    Returns:
        Number of workers to use

    Usage:
        from aperture_photometry.utils.constants import get_parallel_workers
        max_workers = get_parallel_workers(self.params)
    """
    import os

    # Try to get from params
    if params is not None:
        try:
            val = int(getattr(params.P, "max_workers", 0))
            if val > 0:
                return min(val, PARALLEL.MAX_WORKERS_CAP)
        except (AttributeError, TypeError, ValueError):
            pass

    # Auto-detect: 75% of CPU cores, min 2, max cap
    cpu_count = os.cpu_count() or 4
    optimal = max(2, min(int(cpu_count * 0.75), PARALLEL.MAX_WORKERS_CAP))
    return optimal
