"""Sensors for mapping raw boards to the graph."""
from __future__ import annotations

import torch
from domiknows.sensor.pytorch.sensors import ReaderSensor
from domiknows.sensor.pytorch.learners import ModuleLearner
from domiknows.sensor.pytorch.sensors import FunctionalSensor

from .data import BOARD_SIZE
from .learners import QueenCNN
from .graph import queen


class BoardReader(ReaderSensor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, keyword="board", label=False, **kwargs)


def build_sensors() -> tuple[ReaderSensor, ModuleLearner]:
    reader = BoardReader()
    model = QueenCNN()
    learner = ModuleLearner(reader, module=model)
    return reader, learner


def attach_predictions(learner: ModuleLearner) -> None:
    def cell_forward(logits, row: int, col: int):
        return logits[:, row, col]

    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            queen[row][col]["logits"] = FunctionalSensor(
                learner,
                forward=lambda logits, r=row, c=col: cell_forward(logits, r, c),
            )


__all__ = ["BoardReader", "build_sensors", "attach_predictions"]
