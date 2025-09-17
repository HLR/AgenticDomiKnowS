"""Knowledge graph and logical constraints for Eight Queens."""
from __future__ import annotations

from domiknows.graph import Concept, Graph
from domiknows.graph.logicalConstrain import LogicalConstraint, exactL, atMostL

BOARD_SIZE: int = 8

Graph.clear()
Concept.clear()

with Graph("eight_queens_codex") as graph:
    board = Concept(name="board")
    cell = Concept(name="cell")

    board.contains(cell)
    cell.has_attributes(row=int, col=int)

    queen = cell(name="queen")

    # Row constraints: exactly one queen per row.
    for row in range(BOARD_SIZE):
        row_cells = [queen(row=row, col=col) for col in range(BOARD_SIZE)]
        LogicalConstraint(exactL(*row_cells))

    # Column constraints: exactly one queen per column.
    for col in range(BOARD_SIZE):
        col_cells = [queen(row=row, col=col) for row in range(BOARD_SIZE)]
        LogicalConstraint(exactL(*col_cells))

    # Diagonal constraints: at most one queen along each diagonal.
    for difference in range(-BOARD_SIZE + 1, BOARD_SIZE):
        diagonal_cells = [
            queen(row=row, col=col)
            for row in range(BOARD_SIZE)
            for col in range(BOARD_SIZE)
            if row - col == difference
        ]
        if len(diagonal_cells) > 1:
            LogicalConstraint(atMostL(*diagonal_cells, 1))

    for total in range(2 * BOARD_SIZE - 1):
        diagonal_cells = [
            queen(row=row, col=col)
            for row in range(BOARD_SIZE)
            for col in range(BOARD_SIZE)
            if row + col == total
        ]
        if len(diagonal_cells) > 1:
            LogicalConstraint(atMostL(*diagonal_cells, 1))

__all__ = ["graph", "board", "cell", "queen", "BOARD_SIZE"]
