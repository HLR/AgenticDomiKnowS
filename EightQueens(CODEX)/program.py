"""Program wrapper for the Eight Queens example."""
from __future__ import annotations

from domiknows.program import SolverPOIProgram

from .graph import graph, queen


def build_program() -> SolverPOIProgram:
    return SolverPOIProgram(
        graph,
        poi=queen,
        inferTypes=["local/argmax"],
        loss=None,
        metric=None,
    )


__all__ = ["build_program"]
