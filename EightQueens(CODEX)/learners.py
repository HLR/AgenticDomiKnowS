"""CNN classifier that predicts queen placement."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .data import BOARD_SIZE


class QueenCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * BOARD_SIZE * BOARD_SIZE, 256)
        self.fc2 = nn.Linear(256, BOARD_SIZE * BOARD_SIZE)

    def forward(self, board: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(board))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.view(-1, BOARD_SIZE, BOARD_SIZE)


def build_model() -> QueenCNN:
    return QueenCNN()


__all__ = ["QueenCNN", "build_model"]
