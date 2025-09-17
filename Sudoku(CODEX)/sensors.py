"""Neural components for the Sudoku example."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .graph import GRID_SIZE, NUM_DIGITS


class SudokuCNN(nn.Module):
    """A compact CNN that emits logits for each cell-digit pair."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * GRID_SIZE * GRID_SIZE, 256)
        self.fc2 = nn.Linear(256, GRID_SIZE * GRID_SIZE * NUM_DIGITS)

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """Return logits shaped (batch, 36, NUM_DIGITS)."""
        x = F.relu(self.conv1(grid))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.view(-1, GRID_SIZE * GRID_SIZE, NUM_DIGITS)


def build_classifier() -> SudokuCNN:
    """Instantiate the shared CNN used by the example."""
    return SudokuCNN()


def predict_probabilities(model: SudokuCNN, grid_batch: torch.Tensor) -> torch.Tensor:
    """Helper that converts raw logits into per-cell distributions."""
    logits = model(grid_batch)
    return torch.softmax(logits, dim=-1)


__all__ = ["SudokuCNN", "build_classifier", "predict_probabilities"]
