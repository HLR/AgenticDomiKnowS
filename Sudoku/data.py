
import numpy as np

def generate_dummy_sudoku_grid():
    """
    Generates a dummy 6x6 Sudoku grid with numbers 1-3.
    0 represents an empty cell.
    """
    grid = np.zeros((6, 6), dtype=int)

    # Pre-fill some cells with numbers 1-3
    grid[0, 0] = 1
    grid[0, 1] = 2
    grid[1, 0] = 3
    grid[2, 2] = 1
    grid[2, 3] = 2
    grid[3, 2] = 3
    grid[4, 4] = 1
    grid[4, 5] = 2
    grid[5, 4] = 3

    return grid

if __name__ == "__main__":
    grid = generate_dummy_sudoku_grid()
    print("Dummy 6x6 Sudoku Grid:")
    print(grid)
