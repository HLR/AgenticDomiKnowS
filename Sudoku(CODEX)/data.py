"""Utilities for creating a toy 6x6 Sudoku grid."""
from __future__ import annotations

from typing import Tuple

import numpy as np

GRID_SIZE: int = 6


def create_dummy_grid(seed: int | None = 13) -> np.ndarray:
    """Return a deterministic 6x6 grid with numbers between 0 and 3.

    The pattern simply tiles the numbers 0, 1, 2, and 3 so the example can
    demonstrate how a learner can observe partially-filled cells.
    """
    rng = np.random.default_rng(seed)
    grid = rng.integers(low=0, high=4, size=(GRID_SIZE, GRID_SIZE), dtype=np.int64)
    return grid


__all__ = ["GRID_SIZE", "create_dummy_grid"]
