"""Knowledge graph declaration and Sudoku constraints."""
from __future__ import annotations

from domiknows.graph import Concept, Graph, Relation
from domiknows.graph.logicalConstrain import LogicalConstraint, exactL, nandL

GRID_SIZE: int = 6
BLOCK_ROWS: int = 2
BLOCK_COLS: int = 3
NUM_DIGITS: int = 6

Graph.clear()
Concept.clear()
Relation.clear()

with Graph("sudoku_codex") as graph:
    board = Concept(name="board")
    cell = Concept(name="cell")
    digit = Concept(name="digit")

    board.contains(cell)

    cell.has_attributes(row=int, col=int)
    digit.has_attributes(value=int)

    has_digit = Relation(cell, digit, name="has_digit")

    # Each cell must pick exactly one digit.
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            cell_instance = cell(row=row, col=col)
            predicates = [
                has_digit(cell_instance, digit(value=value))
                for value in range(1, NUM_DIGITS + 1)
            ]
            LogicalConstraint(exactL(*predicates))

    # Row uniqueness: a digit may appear at most once per row.
    for row in range(GRID_SIZE):
        for value in range(1, NUM_DIGITS + 1):
            digit_instance = digit(value=value)
            for left in range(GRID_SIZE):
                for right in range(left + 1, GRID_SIZE):
                    cell_left = cell(row=row, col=left)
                    cell_right = cell(row=row, col=right)
                    LogicalConstraint(
                        nandL(
                            has_digit(cell_left, digit_instance),
                            has_digit(cell_right, digit_instance),
                        )
                    )

    # Column uniqueness: a digit may appear at most once per column.
    for col in range(GRID_SIZE):
        for value in range(1, NUM_DIGITS + 1):
            digit_instance = digit(value=value)
            for top in range(GRID_SIZE):
                for bottom in range(top + 1, GRID_SIZE):
                    cell_top = cell(row=top, col=col)
                    cell_bottom = cell(row=bottom, col=col)
                    LogicalConstraint(
                        nandL(
                            has_digit(cell_top, digit_instance),
                            has_digit(cell_bottom, digit_instance),
                        )
                    )

    # Block uniqueness: digits must differ within each 3x2 sub-grid.
    for block_row in range(GRID_SIZE // BLOCK_ROWS):
        for block_col in range(GRID_SIZE // BLOCK_COLS):
            block_cells = []
            for r_offset in range(BLOCK_ROWS):
                for c_offset in range(BLOCK_COLS):
                    row = block_row * BLOCK_ROWS + r_offset
                    col = block_col * BLOCK_COLS + c_offset
                    block_cells.append(cell(row=row, col=col))

            for value in range(1, NUM_DIGITS + 1):
                digit_instance = digit(value=value)
                for idx in range(len(block_cells)):
                    for jdx in range(idx + 1, len(block_cells)):
                        first_cell = block_cells[idx]
                        second_cell = block_cells[jdx]
                        LogicalConstraint(
                            nandL(
                                has_digit(first_cell, digit_instance),
                                has_digit(second_cell, digit_instance),
                            )
                        )

__all__ = [
    "graph",
    "board",
    "cell",
    "digit",
    "has_digit",
    "GRID_SIZE",
    "BLOCK_ROWS",
    "BLOCK_COLS",
    "NUM_DIGITS",
]
