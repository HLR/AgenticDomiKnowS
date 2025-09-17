"""Entry point for the MNIST digit-pair sum example."""
from __future__ import annotations

import itertools
from typing import Iterable, List

import numpy as np
import torch
from domiknows.sensor.pytorch.sensors import FunctionalSensor

from .data import build_dataset
from .graph import (
    digit_a_logits,
    digit_a_value,
    digit_b_logits,
    digit_b_value,
    graph,
    img_a,
    img_b,
    pair,
    sum_label,
)
from .learners import DigitClassifier, DigitLearner
from .program import build_program
from .sensors import DigitLabelSensor, ImageSensor, SumLabelSensor


def _argmax_digit(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits, dim=-1)


def _attach_components() -> DigitClassifier:
    image_a_sensor = ImageSensor(keyword="img_a")
    image_b_sensor = ImageSensor(keyword="img_b")

    digit_a_label_sensor = DigitLabelSensor(keyword="digit_a")
    digit_b_label_sensor = DigitLabelSensor(keyword="digit_b")
    sum_label_sensor = SumLabelSensor()

    classifier = DigitClassifier()
    learner_img_a = DigitLearner(image_a_sensor, module=classifier)
    learner_img_b = DigitLearner(image_b_sensor, module=classifier)

    pair[img_a] = image_a_sensor
    pair[img_b] = image_b_sensor

    pair[digit_a_logits] = digit_a_label_sensor
    pair[digit_a_logits] = learner_img_a

    pair[digit_b_logits] = digit_b_label_sensor
    pair[digit_b_logits] = learner_img_b

    pair[sum_label] = sum_label_sensor

    pair[digit_a_value] = FunctionalSensor(learner_img_a, forward=_argmax_digit)
    pair[digit_b_value] = FunctionalSensor(learner_img_b, forward=_argmax_digit)

    return classifier


def _train_shared_classifier(
    module: DigitClassifier,
    dataset: List[dict],
    *,
    device: torch.device,
    epochs: int = 5,
    batch_size: int = 64,
) -> None:
    module.to(device)

    optimizer = torch.optim.Adam(module.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    num_examples = len(dataset)
    index_array = np.arange(num_examples)

    for epoch in range(epochs):
        np.random.shuffle(index_array)
        epoch_loss = 0.0
        batches = 0

        for start in range(0, num_examples, batch_size):
            batch_indices = index_array[start : start + batch_size]
            batch = [dataset[idx] for idx in batch_indices]

            imgs_a = torch.stack([item["img_a"] for item in batch]).to(device)
            imgs_b = torch.stack([item["img_b"] for item in batch]).to(device)
            labels_a = torch.tensor([item["digit_a"] for item in batch], dtype=torch.long, device=device)
            labels_b = torch.tensor([item["digit_b"] for item in batch], dtype=torch.long, device=device)

            optimizer.zero_grad(set_to_none=True)
            logits_a = module(imgs_a)
            logits_b = module(imgs_b)

            loss = criterion(logits_a, labels_a) + criterion(logits_b, labels_b)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            batches += 1

        avg_loss = epoch_loss / max(batches, 1)
        print(f"Epoch {epoch + 1}: loss={avg_loss:.4f}")

    module.to("cpu")


def _evaluate_locally(module: DigitClassifier, dataset: Iterable[dict]) -> None:
    module.eval()
    with torch.no_grad():
        for i, item in enumerate(itertools.islice(dataset, 3)):
            img_a_tensor = item["img_a"].unsqueeze(0)
            img_b_tensor = item["img_b"].unsqueeze(0)
            pred_a = torch.argmax(module(img_a_tensor), dim=-1).item()
            pred_b = torch.argmax(module(img_b_tensor), dim=-1).item()
            print(
                f"Example {i + 1}: digits=({item['digit_a']}, {item['digit_b']}), "
                f"pred=({pred_a}, {pred_b}), sum_label={item['sum_label']}"
            )


def main() -> None:
    dataset = build_dataset(num_pairs=512)
    classifier = _attach_components()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training digit classifier on {device} ...")
    _train_shared_classifier(classifier, dataset, device=device)

    _evaluate_locally(classifier, dataset)

    program = build_program()
    sample = dataset[:2]

    try:
        datanode = next(program.populate(sample, device="cpu"))
        datanode.inferILPResults()
        pred_a = int(datanode.getAttribute(digit_a_value, "ILP"))
        pred_b = int(datanode.getAttribute(digit_b_value, "ILP"))
        total = int(datanode.getAttribute(sum_label, "ILP"))
        print("\nILP-enforced prediction:")
        print(f"  pred_a={pred_a}, pred_b={pred_b}, sum_label={total}")
    except Exception as exc:  # pragma: no cover - depends on optional solvers
        print("\nUnable to run ILP inference inside DomiKnowS:", exc)
        print("The shared classifier is still trained; install optional solvers for full inference.")


if __name__ == "__main__":
    main()
