
import torch
from domiknows.sensor.pytorch.sensors import TorchSensor, ReaderSensor
from domiknows.sensor.pytorch.learners import TorchLearner

class HeaderSensor(ReaderSensor):
    def __init__(self, *pres, header, label=False, device='auto'):
        super().__init__(*pres, keyword=header, label=label, device=device)

class BodySensor(ReaderSensor):
    def __init__(self, *pres, body, label=False, device='auto'):
        super().__init__(*pres, keyword=body, label=label, device=device)

class SpamLabelSensor(ReaderSensor):
    def __init__(self, *pres, spam, label=True, device='auto'):
        super().__init__(*pres, keyword=spam, label=label, device=device)

class SpamModel1(TorchLearner):
    def __init__(self, *pres):
        super().__init__(*pres)
        self.spam_model = torch.nn.Linear(2, 1) # 2 features (header, body)

    def forward(self, header, body):
        # This is a simplified example. In a real scenario, you'd use embeddings.
        header_len = torch.tensor([len(h) for h in header], dtype=torch.float32).unsqueeze(1)
        body_len = torch.tensor([len(b) for b in body], dtype=torch.float32).unsqueeze(1)
        features = torch.cat([header_len, body_len], dim=1)
        return self.spam_model(features)

    def parameters(self):
        return self.spam_model.parameters()

class SpamModel2(TorchLearner):
    def __init__(self, *pres):
        super().__init__(*pres)
        self.spam_model = torch.nn.Linear(2, 1)

    def forward(self, header, body):
        header_len = torch.tensor([len(h) for h in header], dtype=torch.float32).unsqueeze(1)
        body_len = torch.tensor([len(b) for b in body], dtype=torch.float32).unsqueeze(1)
        features = torch.cat([header_len, body_len], dim=1)
        return self.spam_model(features)

    def parameters(self):
        return self.spam_model.parameters()
