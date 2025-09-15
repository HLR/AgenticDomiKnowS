
import torch
from torch import nn
from domiknows.sensor.pytorch.sensors import TorchSensor, ReaderSensor
from domiknows.sensor.pytorch.learners import TorchLearner
from domiknows.sensor.pytorch.relation_sensors import EdgeSensor

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 1024)
        x = self.fc1(x)
        return x

def get_sensors(graph, device):
    # --- Learners ---
    digit_model = SimpleCNN().to(device)
    digit_learner = TorchLearner(
        graph['digit_a'],
        model=digit_model,
        optimizer=torch.optim.Adam, 
        lr=0.001
    )

    # --- Sensors ---
    def to_device(x):
        return x.to(device)

    img_a_sensor = ReaderSensor('img_a', preprocess=to_device)
    img_b_sensor = ReaderSensor('img_b', preprocess=to_device)
    sum_sensor = ReaderSensor('sum')
    
    # Ground truth labels for pre-training
    digit_a_label_sensor = ReaderSensor('digit_a')
    digit_b_label_sensor = ReaderSensor('digit_b')

    return {
        'img_a': img_a_sensor,
        'img_b': img_b_sensor,
        'sum': sum_sensor,
        'digit_a_learner': digit_learner,
        'digit_b_learner': digit_learner, # Shared learner
        'digit_a_label': digit_a_label_sensor,
        'digit_b_label': digit_b_label_sensor
    }
