"""Torch learners that score emails for spam."""
from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

from domiknows.sensor.pytorch.learners import ModuleLearner

_SPAM_TRIGGER_WORDS: Iterable[str] = (
    "free",
    "win",
    "winner",
    "prize",
    "bonus",
    "offer",
    "urgent",
    "money",
    "guaranteed",
    "click",
)


class _SpamScoringModule(nn.Module):
    def __init__(self, hidden_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, header_text: str, body_text: str) -> torch.Tensor:
        device = next(self.parameters()).device
        features = self._string_features(header_text, body_text, device)
        return self.net(features)

    @staticmethod
    def _string_features(header_text: str, body_text: str, device: torch.device) -> torch.Tensor:
        header_lower = header_text.lower()
        body_lower = body_text.lower()

        spam_keyword_hits = sum(word in header_lower for word in _SPAM_TRIGGER_WORDS)
        spam_keyword_hits += sum(word in body_lower for word in _SPAM_TRIGGER_WORDS)

        features = torch.tensor(
            [
                len(header_text),
                header_text.count("!"),
                len(body_text),
                body_text.count("!"),
                body_text.count("$"),
                float(spam_keyword_hits),
            ],
            dtype=torch.float32,
            device=device,
        )
        features = features.unsqueeze(0)
        return features


class SpamLearner(ModuleLearner):
    def __init__(self, *pres, hidden_dim: int = 8, **kwargs):
        super().__init__(*pres, module=_SpamScoringModule(hidden_dim=hidden_dim), **kwargs)
