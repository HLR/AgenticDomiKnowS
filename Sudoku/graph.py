from domiknows.graph import Graph, Concept, Relation, Predicate
from domiknows.graph.logicalConstrain import LogicalConstraint
from domiknows.graph.logicalConstrain import nandL, exactL

# Define the dimensions of the Sudoku grid
GRID_SIZE = 6
BLOCK_ROW_SIZE = 2
BLOCK_COL_SIZE = 3
NUM_VALUES = 3 # Numbers 1, 2, 3

Graph.clear()
with Graph("sudoku") as graph:
    # Concepts
    # Cell represents a position in the 6x6 grid
    cell = Concept(name="cell")
    # Number represents the possible values (1, 2, 3)
    number = Concept(name="number")

    # Attributes for Cell to store its row and column index
    cell.has_attributes(row=int, col=int)
    # Attribute for Number to store its value
    number.has_attributes(value=int)

    # Relation: HasNumber(cell, number) - indicates a cell has a specific number
    has_number = Relation(cell, number, name="has_number")

    # --- Logical Constraints ---

    # 1. Each cell must have exactly one number
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            cell_r_c = cell(row=r, col=c)
            # Create predicates for each possible number for the current cell
            predicates = [has_number(cell_r_c, number(value=v + 1)) for v in range(NUM_VALUES)]
            # Apply exactL constraint: exactly one of these predicates must be true
            LogicalConstraint(exactL(*predicates))

    # 2. Row uniqueness: In each row, all numbers must be different
    # For each row 'r', and for each number 'v', no two distinct cells (r, c1) and (r, c2) can have 'v'
    for r in range(GRID_SIZE):
        for v in range(NUM_VALUES):
            num_v = number(value=v + 1)
            # Iterate over all unique pairs of columns in the current row
            for c1 in range(GRID_SIZE):
                for c2 in range(c1 + 1, GRID_SIZE):
                    cell_r_c1 = cell(row=r, col=c1)
                    cell_r_c2 = cell(row=r, col=c2)
                    # Constraint: It's not allowed that both cell_r_c1 and cell_r_c2 have num_v
                    LogicalConstraint(nandL(has_number(cell_r_c1, num_v), has_number(cell_r_c2, num_v)))

    # 3. Column uniqueness: In each column, all numbers must be different
    # For each column 'c', and for each number 'v', no two distinct cells (r1, c) and (r2, c) can have 'v'
    for c in range(GRID_SIZE):
        for v in range(NUM_VALUES):
            num_v = number(value=v + 1)
            # Iterate over all unique pairs of rows in the current column
            for r1 in range(GRID_SIZE):
                for r2 in range(r1 + 1, GRID_SIZE):
                    cell_r1_c = cell(row=r1, col=c)
                    cell_r2_c = cell(row=r2, col=c)
                    # Constraint: It's not allowed that both cell_r1_c and cell_r2_c have num_v
                    LogicalConstraint(nandL(has_number(cell_r1_c, num_v), has_number(cell_r2_c, num_v)))

    # 4. Block uniqueness: In each 3x2 block, all numbers must be different
    # The grid is 6x6, blocks are 3x2. There are (6/2) * (6/3) = 3 * 2 = 6 blocks.
    # Block (br, bc) refers to the block at block_row_idx 'br' and block_col_idx 'bc'
    for br in range(GRID_SIZE // BLOCK_ROW_SIZE): # 0, 1, 2
        for bc in range(GRID_SIZE // BLOCK_COL_SIZE): # 0, 1
            # Collect all cells within the current block
            block_cells = []
            for r_offset in range(BLOCK_ROW_SIZE):
                for c_offset in range(BLOCK_COL_SIZE):
                    r = br * BLOCK_ROW_SIZE + r_offset
                    c = bc * BLOCK_COL_SIZE + c_offset
                    block_cells.append(cell(row=r, col=c))

            # Apply uniqueness constraint for each number within this block
            for v in range(NUM_VALUES):
                num_v = number(value=v + 1)
                # Iterate over all unique pairs of cells within the current block
                for i in range(len(block_cells)):
                    for j in range(i + 1, len(block_cells)):
                        cell1 = block_cells[i]
                        cell2 = block_cells[j]
                        # Constraint: It's not allowed that both cell1 and cell2 have num_v
                        LogicalConstraint(nandL(has_number(cell1, num_v), has_number(cell2, num_v)))

# You can access the graph object via graph.g
# For example, to print all concepts:
# for c in graph.g.concepts:
#     print(c.name)
