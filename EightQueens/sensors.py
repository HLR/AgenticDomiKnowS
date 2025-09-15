
import torch
from torch import nn
from domiknows.sensor.pytorch.sensors import ReaderSensor

class BoardSensor(ReaderSensor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, data_item):
        return torch.from_numpy(data_item[self.keyword]).unsqueeze(0)

class QueenLearner(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 1, kernel_size=3, padding=1)

    def forward(self, board):
        x = self.conv1(board)
        x = self.relu(x)
        x = self.conv2(x)
        return x.squeeze(0)
