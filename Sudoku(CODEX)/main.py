"""Minimal driver for the Sudoku(CODEX) example."""
from __future__ import annotations

import numpy as np
import torch

from .data import GRID_SIZE, create_dummy_grid
from .graph import NUM_DIGITS, graph
from .program import build_program
from .sensors import build_classifier, predict_probabilities


def _prepare_input(grid: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(grid.astype(np.float32))
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # (batch, channel, rows, cols)
    return tensor / max(1.0, float(grid.max()))


def _to_prediction_grid(probabilities: torch.Tensor) -> np.ndarray:
    digits = torch.argmax(probabilities, dim=-1) + 1
    return digits.view(GRID_SIZE, GRID_SIZE).cpu().numpy()


def _print_uniqueness_report(predicted_grid: np.ndarray) -> None:
    def unique(values: np.ndarray) -> bool:
        filtered = values.tolist()
        return len(set(filtered)) == len(filtered)

    for row in range(GRID_SIZE):
        print(f"Row {row} unique: {unique(predicted_grid[row, :])}")
    for col in range(GRID_SIZE):
        print(f"Column {col} unique: {unique(predicted_grid[:, col])}")


def main() -> None:
    dummy_grid = create_dummy_grid()
    print("Dummy 6x6 grid (values limited to 0-3):")
    print(dummy_grid)

    model = build_classifier()
    input_tensor = _prepare_input(dummy_grid)
    probabilities = predict_probabilities(model, input_tensor)

    predicted_grid = _to_prediction_grid(probabilities)
    print("\nRaw argmax predictions from the untrained CNN:")
    print(predicted_grid)

    print("\nUniqueness diagnostics (prior to any training or constraint enforcement):")
    _print_uniqueness_report(predicted_grid)

    program = build_program()
    print("\nConstructed SolverPOIProgram with graph-defined Sudoku constraints.")
    print("Program infer types:", program.model.inferTypes)
    print("NOTE: Training the CNN under these constraints requires integrating the sensor with a learning loop.")


if __name__ == "__main__":
    main()
