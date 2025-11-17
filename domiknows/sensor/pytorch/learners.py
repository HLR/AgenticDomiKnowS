import abc
from typing import Any
import os.path
import warnings

import torch

from .. import Learner
from .sensors import TorchSensor, FunctionalSensor, ModuleSensor, ReaderSensor
from .learnerModels import PyTorchFC, LSTMModel, PyTorchFCRelu


class TorchLearner(Learner, FunctionalSensor):
    __metaclass__ = abc.ABCMeta

    def __init__(self, *pre, edges=None, loss=None, metric=None, label=False, device='auto',**kwargs):
        self.updated = False
        super(TorchLearner, self).__init__(*pre, edges=edges, label=label, device=device,**kwargs)
        self._loss = loss
        self._metric = metric

    @property
    def model(self):
        return None

    @property
    @abc.abstractmethod
    def parameters(self) -> Any:
        # self.update_parameters()
        if self.model is not None:
            return self.model.parameters()

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        if self.model is not None:
            self.parameters.to(device)
        self._device = device

    def update_parameters(self):
        if not self.updated:
            for pre in self.pres:
                for learner in self.sup.sup[pre].find(TorchLearner):
                    self.model.add_module(learner.name, module=learner.model)
            self.updated = True

    @property
    def sanitized_name(self):
        return self.fullname.replace('/', '_').replace("<","").replace(">","")

    def save(self, filepath):
        save_path = os.path.join(filepath, self.sanitized_name)
        torch.save(self.model.state_dict(), save_path)

    def load(self, filepath):
        save_path = os.path.join(filepath, self.sanitized_name)
        try:
            self.model.load_state_dict(torch.load(save_path))
            self.model.eval()
            self.model.train()
        except FileNotFoundError:
            message = f'Failed to load {self} from {save_path}. Continue not loaded.'
            warnings.warn(message)

    def loss(self, data_item, target):
        if self._loss is not None:
            pred = self(data_item)
            label = target(data_item)
            return self._loss(pred, label)

    def metric(self, data_item, target):
        if self._metric:
            pred = self(data_item)
            label = target(data_item)
            return self._metric(pred, label)


class ModuleLearner(ModuleSensor, TorchLearner):
    def __init__(self, *pres, module, edges=None, loss=None, metric=None, label=False, **kwargs):
        super().__init__(*pres, module=module, edges=edges, label=label, **kwargs)
        self._loss = loss
        self._metric = metric
        self.updated = True  # no need to update

    def update_parameters(self):
        pass


class LSTMLearner(TorchLearner):
    def __init__(self, *pres, input_dim, hidden_dim, num_layers=1, bidirectional=False, device='auto'):
        super(LSTMLearner, self).__init__(*pres, device=device)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.input_dim = input_dim
        self.model = LSTMModel(input_dim=self.input_dim, hidden_dim=self.hidden_dim, num_layers=self.num_layers,
                               batch_size=1, bidirectional=self.bidirectional)
        is_cuda = torch.cuda.is_available()
        if is_cuda:
            self.model.cuda()

    def forward(
            self,
    ) -> Any:
        value = self.inputs[0]
        if not torch.is_tensor(value):
            value = torch.stack(self.inputs[0])
        output = self.model(value)
        return output


class FullyConnectedLearner(TorchLearner):
    def __init__(self, *pres, input_dim, output_dim, device='auto'):
        super(FullyConnectedLearner, self).__init__(*pres, device=device)
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.model = PyTorchFC(input_dim=self.input_dim, output_dim=self.output_dim)
        is_cuda = torch.cuda.is_available()
        if is_cuda:
            self.model.cuda()

    def forward(
            self,
    ) -> Any:
        _tensor = self.inputs[0]
        output = self.model(_tensor)
        return output


class FullyConnectedLearnerRelu(TorchLearner):
    def __init__(self, *pres, input_dim, output_dim, device='auto'):
        super(FullyConnectedLearnerRelu, self).__init__(*pres, device=device)
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.model = PyTorchFCRelu(input_dim=self.input_dim, output_dim=self.output_dim)
        is_cuda = torch.cuda.is_available()
        if is_cuda:
            self.model.cuda()

    def forward(
            self,
    ) -> Any:
        _tensor = self.inputs[0]
        output = self.model(_tensor)
        return output

class DummyLearner(TorchLearner):

    def __init__(self, *pre, output_size=2):
        super(DummyLearner, self).__init__(*pre)
        self.output_size = output_size

    def forward(self, x):
        result = torch.zeros(len(x), self.output_size)
        random_indices = torch.randint(0, self.output_size, (len(x),))
        result[torch.arange(len(x)), random_indices] = 1
        return result

class LLMLearner(ModuleSensor, TorchLearner):

    def __init__(self, *pres, prompt, classes,rel=None, **kwargs):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        model_name = "distilgpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name)
        self.prompt = prompt
        self.classes = [str(c) for c in classes]
        self.tokenizer = tokenizer
        self.rel = rel
        super().__init__(*pres, module=model, edges=None, label=False, **kwargs)
        self.updated = True

    def fill_data_(self, data_item):
        if self.rel:
            self.rel_data = data_item[self.rel]

    def _build_prompt(self, inputs) -> str:
        options = ", ".join(self.classes)
        features = ", ".join(inputs)
        return (f"{self.prompt}\n\nfeatures: {features}\n\nAnswer with exactly one label from: {options} and do not say anything else.")

    def _logprob_continuation(self, prompt: str, continuation: str) -> float:
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False, return_tensors=None)["input_ids"]
        cont_ids = self.tokenizer(continuation, add_special_tokens=False, return_tensors=None)["input_ids"]
        input_ids = torch.tensor([prompt_ids + cont_ids], dtype=torch.long)

        outputs = self.module(input_ids=input_ids)
        logits = outputs.logits

        seq_len = input_ids.size(1)
        prompt_len = len(prompt_ids)
        cont_len = len(cont_ids)
        logits_ctx = logits[0, prompt_len - 1: prompt_len + cont_len - 1, :]
        logprobs = torch.log_softmax(logits_ctx, dim=-1)
        target_ids = torch.tensor(cont_ids, dtype=torch.long, device=logprobs.device)
        token_logps = logprobs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
        return token_logps.sum()

    def forward(self, *inputs):
        max_size = max([len(i) for i in inputs])

        if not self.rel:
            batch = [[] for i in range(max_size)]
            for i in inputs:
                if len(i) == 1:
                    for j in batch:
                        j.append(str(i[0]))
                else:
                    for j, k in zip(batch, i):
                        j.append(str(k))
        else:
            relations = self.rel_data[0]
            batch = []
            if len(relations[0]) == 2:
                for i, j in relations:
                    batch.append([str(inputs[0][i]), str(inputs[1][j])])
            elif len(relations[0]) == 3:
                for i, j, k in relations:
                    batch.append([str(inputs[0][i]), str(inputs[1][j]), str(inputs[2][k])])

        outputs = []
        for instance in batch:
            prompt = self._build_prompt(instance)
            scores = []
            for c in self.classes:
                continuation = " " + str(c)
                logp = self._logprob_continuation(prompt, continuation)
                scores.append(logp)
            probs = torch.tensor(scores, dtype=torch.float32)
            probs = torch.softmax(probs, dim=0)
            outputs.append(probs)

        return outputs
