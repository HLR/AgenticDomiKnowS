"""Learners for digit classification."""
from __future__ import annotations

import torch
from torch import nn

from domiknows.sensor.pytorch.learners import ModuleLearner


class DigitClassifier(nn.Module):
    """A compact MLP for MNIST digits."""

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.dim() == 3:
            image = image.unsqueeze(0)
        return self.network(image)


class DigitLearner(ModuleLearner):
    """Wrap the shared digit classifier as a ModuleLearner."""

    def __init__(self, *pres, module: nn.Module, **kwargs):
        super().__init__(*pres, module=module, **kwargs)


__all__ = ["DigitClassifier", "DigitLearner"]
