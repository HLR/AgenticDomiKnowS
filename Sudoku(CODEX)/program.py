"""Program assembly for the Sudoku example."""
from __future__ import annotations

from domiknows.program import SolverPOIProgram

from .graph import graph, has_digit


def build_program() -> SolverPOIProgram:
    """Return a solver program that operates over the Sudoku graph.

    The example is purely constraint-driven, so no supervised loss or
    metrics are supplied. Users can plug in their own optimisation loop
    if desired.
    """
    return SolverPOIProgram(
        graph,
        poi=[has_digit],
        inferTypes=["local/argmax"],
        loss=None,
        metric=None,
    )


__all__ = ["build_program"]
