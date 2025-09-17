"""Driver script demonstrating the Eight Queens(CODEX) components."""
from __future__ import annotations

import numpy as np
import torch

from .data import BOARD_SIZE, build_dataset
from .graph import graph
from .learners import build_model
from .program import build_program
from .sensors import attach_predictions, build_sensors


def _prepare_input(board: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(board.astype(np.float32)).unsqueeze(0).unsqueeze(0)


def main() -> None:
    dataset = build_dataset(num_samples=4)
    sample = dataset[0]["board"]
    print("Sample partially filled board:")
    print(sample)

    reader, learner = build_sensors()
    attach_predictions(learner)

    model = build_model()
    inputs = _prepare_input(sample)
    logits = model(inputs)
    print("\nRaw logits from untrained CNN:")
    print(logits.view(BOARD_SIZE, BOARD_SIZE).detach().numpy())

    program = build_program()
    print("\nConstructed SolverPOIProgram covering queen placements.")
    print("Infer types:", program.model.inferTypes)
    print("NOTE: Constraint-driven training requires integrating the CNN outputs with program training loops.")


if __name__ == "__main__":
    main()
