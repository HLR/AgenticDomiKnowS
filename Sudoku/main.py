import torch
import numpy as np
from domiknows.graph import Graph, Concept, Relation, DataNode
from domiknows.program import LearningBasedProgram
from domiknows.program.model_program import SolverPOIProgram

from Sudoku.data import generate_dummy_sudoku_grid
from Sudoku.graph import graph, GRID_SIZE, NUM_VALUES, BLOCK_ROW_SIZE, BLOCK_COL_SIZE
from Sudoku.sensors import create_sudoku_sensor

# 1. Create Sensor and Input Concept
sudoku_sensor, grid_input_concept = create_sudoku_sensor(graph)

# 2. Generate Dummy Data
dummy_grid = generate_dummy_sudoku_grid()
print("Initial Dummy Sudoku Grid (0 for empty cells):")
print(dummy_grid)

# 3. Create DataNodes
# Create a DataNode for the entire grid input
grid_dn = DataNode(name="grid_data_node")
# Set the grid_data attribute for the grid_input_concept instance within this DataNode
grid_dn[grid_input_concept].set(grid_data=torch.from_numpy(dummy_grid).unsqueeze(0)) # Add batch dimension

# Create DataNodes for cells and numbers within the scope of grid_dn
for r in range(GRID_SIZE):
    for c in range(GRID_SIZE):
        # Create an instance of the 'cell' concept with its attributes
        grid_dn[graph.get_concept("cell")](row=r, col=c)

for v in range(NUM_VALUES):
    # Create an instance of the 'number' concept with its attribute
    grid_dn[graph.get_concept("number")](value=v + 1)

# 4. Initialize LearningBasedProgram
program = SolverPOIProgram(
    graph=graph,
    Model={
        'sudoku_sensor': sudoku_sensor
    },
    # No uniform_loss for has_number as there are no labels.
    # The learning will be driven purely by ConstraintLoss.
    uniform_loss=None,
    metrics={}
)

# 5. Training Loop
optimizer = torch.optim.Adam(program.model.parameters(), lr=0.01)
num_epochs = 2000 # Increased epochs for better convergence

print("\nStarting training...")
for epoch in range(num_epochs):
    optimizer.zero_grad()

    # Forward pass: run the sensor to get initial predictions
    program.model(grid_dn)

    # Calculate total loss (including constraint loss)
    loss = program.compute_loss(grid_dn)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 200 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

print("Training finished.")

# 6. Inference and Display Results
print("\nInference Results:")
# Get the predicted probabilities for has_number relation
has_number_predictions = grid_dn[graph.get_relation("has_number")].probs

# Reshape predictions to a 6x6 grid of numbers
predicted_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

# Iterate through cells and find the most probable number
for r in range(GRID_SIZE):
    for c in range(GRID_SIZE):
        cell_idx = r * GRID_SIZE + c
        # Get probabilities for the current cell across all numbers
        # [0] is for the batch dimension (since we have batch_size=1)
        cell_probs = has_number_predictions[0, cell_idx, :].detach().numpy()
        predicted_number_value = np.argmax(cell_probs) + 1 # +1 because numbers are 1, 2, 3

        predicted_grid[r, c] = predicted_number_value

print("Predicted Sudoku Grid:")
print(predicted_grid)

# 7. Verification of Sudoku Rules
def verify_sudoku_rules(grid):
    # Check if all cells are filled with numbers 1, 2, or 3
    if not np.all((grid >= 1) & (grid <= NUM_VALUES)):
        print("Error: Not all cells are filled with numbers 1-3.")
        return False

    # Helper to check uniqueness of numbers 1-3 in a given array (row, col, or block)
    def check_unique_numbers_1_to_3(arr):
        seen_numbers = set()
        for x in arr:
            if x in [1, 2, 3]: # Only consider numbers 1, 2, 3 for uniqueness
                if x in seen_numbers:
                    return False # Number already seen
                seen_numbers.add(x)
        return True

    # Check row uniqueness
    for r in range(GRID_SIZE):
        if not check_unique_numbers_1_to_3(grid[r, :]):
            print(f"Row {r} violates uniqueness: {grid[r, :]}")
            return False

    # Check column uniqueness
    for c in range(GRID_SIZE):
        if not check_unique_numbers_1_to_3(grid[:, c]):
            print(f"Column {c} violates uniqueness: {grid[:, c]}")
            return False

    # Check block uniqueness
    for br in range(GRID_SIZE // BLOCK_ROW_SIZE):
        for bc in range(GRID_SIZE // BLOCK_COL_SIZE):
            block_values = []
            for r_offset in range(BLOCK_ROW_SIZE):
                for c_offset in range(BLOCK_COL_SIZE):
                    r = br * BLOCK_ROW_SIZE + r_offset
                    c = bc * BLOCK_COL_SIZE + c_offset
                    block_values.append(grid[r, c])
            if not check_unique_numbers_1_to_3(block_values):
                print(f"Block ({br}, {bc}) violates uniqueness: {block_values}")
                return False
    return True

if verify_sudoku_rules(predicted_grid):
    print("\nPredicted grid satisfies all Sudoku rules (1-3 uniqueness in rows, columns, and blocks).")
else:
    print("\nPredicted grid DOES NOT satisfy all Sudoku rules.")
