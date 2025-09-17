"""Data utilities for the MNIST digit-pair sum example."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
from torchvision import datasets, transforms


@dataclass(frozen=True)
class DigitPair:
    """Container holding two MNIST digits and their sum."""

    img_a: torch.Tensor
    img_b: torch.Tensor
    digit_a: int
    digit_b: int

    @property
    def sum_label(self) -> int:
        return self.digit_a + self.digit_b

    def to_record(self) -> Dict[str, object]:
        return {
            "img_a": self.img_a,
            "img_b": self.img_b,
            "digit_a": self.digit_a,
            "digit_b": self.digit_b,
            "sum_label": self.sum_label,
        }


def _load_mnist(root: str, train: bool) -> Sequence[torch.Tensor]:
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root=root, train=train, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    images, labels = next(iter(loader))
    return images, labels


def build_dataset(
    *,
    num_pairs: int = 512,
    mnist_root: str = "./data",
    train_split: bool = True,
    rng: np.random.Generator | None = None,
) -> List[Dict[str, object]]:
    """Create a small paired dataset of MNIST digits with their sums."""
    images, labels = _load_mnist(mnist_root, train_split)
    generator = rng or np.random.default_rng(seed=13)

    records: List[Dict[str, object]] = []
    num_items = images.shape[0]

    for _ in range(num_pairs):
        idx_a = generator.integers(0, num_items)
        idx_b = generator.integers(0, num_items)

        img_a = images[idx_a]
        img_b = images[idx_b]
        digit_a = int(labels[idx_a].item())
        digit_b = int(labels[idx_b].item())

        pair = DigitPair(img_a=img_a, img_b=img_b, digit_a=digit_a, digit_b=digit_b)
        records.append(pair.to_record())

    return records


__all__ = ["DigitPair", "build_dataset"]
