import torch
import torch.nn as nn
import torch.nn.functional as F
from domiknows.sensor.pytorch import PyTorchSensor
from domiknows.graph import Concept, Relation

# Define the dimensions of the Sudoku grid
GRID_SIZE = 6
NUM_VALUES = 3 # Numbers 1, 2, 3

# Define the CNN model
class SudokuCNN(nn.Module):
    def __init__(self):
        super(SudokuCNN, self).__init__()
        # Input: 1 channel (6x6 grid), Output: 32 channels
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # Output: 64 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Flatten and connect to 256 neurons
        self.fc1 = nn.Linear(64 * GRID_SIZE * GRID_SIZE, 256)
        # Output: 36 cells * 3 numbers (logits)
        self.fc2 = nn.Linear(256, GRID_SIZE * GRID_SIZE * NUM_VALUES)

    def forward(self, x):
        # x shape: (batch_size, 1, GRID_SIZE, GRID_SIZE)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 64 * GRID_SIZE * GRID_SIZE) # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # Reshape to (batch_size, 36 cells, 3 numbers)
        x = x.view(-1, GRID_SIZE * GRID_SIZE, NUM_VALUES)
        return x

# Define the sensor
def create_sudoku_sensor(graph):
    cell = graph.get_concept("cell")
    number = graph.get_concept("number")
    has_number = graph.get_relation("has_number")

    # Define a concept for the input grid itself
    # This concept will hold the entire 6x6 grid data
    grid_input_concept = Concept(name="grid_input")
    grid_input_concept.has_attributes(grid_data=torch.Tensor)

    class SudokuSensor(PyTorchSensor):
        def __init__(self, name, model, graph, input_concept, output_relation):
            super().__init__(name, model, graph,
                             input=[(input_concept, 'grid_data')],
                             output=[(output_relation, 'probs')])

        def forward(self, grid_data):
            # grid_data shape: (batch_size, GRID_SIZE, GRID_SIZE)
            # Add a channel dimension for CNN: (batch_size, 1, GRID_SIZE, GRID_SIZE)
            # Normalize input values (0-3) to a range like 0-1
            x = grid_data.unsqueeze(1).float() / NUM_VALUES
            logits = self.model(x) # (batch_size, 36, 3)
            probs = F.softmax(logits, dim=-1)
            return probs

    # Instantiate the CNN model
    model = SudokuCNN()

    # Create the sensor
    sensor = SudokuSensor(name="sudoku_sensor", model=model, graph=graph,
                          input_concept=grid_input_concept, output_relation=has_number)

    return sensor, grid_input_concept
