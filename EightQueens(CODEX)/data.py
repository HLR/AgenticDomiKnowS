"""Synthetic board generator for the Eight Queens example."""
from __future__ import annotations

import numpy as np
from typing import Dict, List

BOARD_SIZE: int = 8


def generate_partial_board(min_queens: int = 2, max_queens: int = 5) -> np.ndarray:
    """Create a board with a random number of pre-placed queens."""
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    num_queens = np.random.randint(min_queens, max_queens + 1)

    positions = set()
    while len(positions) < num_queens:
        row = np.random.randint(0, BOARD_SIZE)
        col = np.random.randint(0, BOARD_SIZE)
        positions.add((row, col))

    for row, col in positions:
        board[row, col] = 1.0
    return board


def build_dataset(num_samples: int = 128) -> List[Dict[str, np.ndarray]]:
    return [{"board": generate_partial_board()} for _ in range(num_samples)]


__all__ = ["BOARD_SIZE", "build_dataset", "generate_partial_board"]
