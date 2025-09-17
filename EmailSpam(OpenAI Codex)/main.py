"""Run the email spam consistency example."""
from __future__ import annotations

import pathlib
import sys
from typing import TYPE_CHECKING

import torch

from domiknows.sensor.pytorch.sensors import FunctionalSensor

if TYPE_CHECKING:
    from .data import build_dataset  # type: ignore[attr-defined]
    from .graph import (
        body,
        email,
        graph,
        header,
        model1_logits,
        model1_not_spam,
        model1_spam,
        model2_logits,
        model2_not_spam,
        model2_spam,
    )
    from .learners import SpamLearner
    from .program import build_program
    from .sensors import BodySensor, HeaderSensor, SpamLabelSensor


if __package__ is None or __package__ == "":
    package_root = pathlib.Path(__file__).resolve().parent
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from data import build_dataset  # type: ignore
    from graph import (
        body,
        email,
        graph,
        header,
        model1_logits,
        model1_not_spam,
        model1_spam,
        model2_logits,
        model2_not_spam,
        model2_spam,
    )  # type: ignore
    from learners import SpamLearner  # type: ignore
    from program import build_program  # type: ignore
    from sensors import BodySensor, HeaderSensor, SpamLabelSensor  # type: ignore
else:  # pragma: no cover - handled in TYPE_CHECKING block
    from .data import build_dataset
    from .graph import (
        body,
        email,
        graph,
        header,
        model1_logits,
        model1_not_spam,
        model1_spam,
        model2_logits,
        model2_not_spam,
        model2_spam,
    )
    from .learners import SpamLearner
    from .program import build_program
    from .sensors import BodySensor, HeaderSensor, SpamLabelSensor


def _spam_indicator(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)[..., 1:2]
    return (probs >= 0.5).to(dtype=logits.dtype)


def _not_spam_indicator(logits: torch.Tensor) -> torch.Tensor:
    return 1.0 - _spam_indicator(logits)


def _attach_sensors() -> None:
    header_sensor = HeaderSensor()
    body_sensor = BodySensor()

    model1_label = SpamLabelSensor()
    model2_label = SpamLabelSensor()

    model1 = SpamLearner(header_sensor, body_sensor)
    model2 = SpamLearner(header_sensor, body_sensor)

    email[header] = header_sensor
    email[body] = body_sensor

    email[model1_logits] = model1_label
    email[model1_logits] = model1

    email[model2_logits] = model2_label
    email[model2_logits] = model2

    email[model1_spam] = FunctionalSensor(model1, forward=_spam_indicator)
    email[model1_not_spam] = FunctionalSensor(model1, forward=_not_spam_indicator)
    email[model2_spam] = FunctionalSensor(model2, forward=_spam_indicator)
    email[model2_not_spam] = FunctionalSensor(model2, forward=_not_spam_indicator)


def _describe_predictions(datanode) -> None:
    model1_logits_value = datanode.getAttribute(model1_logits)
    model2_logits_value = datanode.getAttribute(model2_logits)

    model1_probs = torch.softmax(model1_logits_value, dim=-1)
    model2_probs = torch.softmax(model2_logits_value, dim=-1)

    print("Model 1 spam probability:", model1_probs[..., 1].item())
    print("Model 2 spam probability:", model2_probs[..., 1].item())

    print("Consistency attributes (1 == spam, 0 == not spam):")
    print("  Model 1 spam flag:", datanode.getAttribute(model1_spam).item())
    print("  Model 2 spam flag:", datanode.getAttribute(model2_spam).item())
    print("  Model 1 not spam flag:", datanode.getAttribute(model1_not_spam).item())
    print("  Model 2 not spam flag:", datanode.getAttribute(model2_not_spam).item())


def main() -> None:
    _attach_sensors()
    dataset = build_dataset()

    program = build_program()
    optimizer_factory = lambda params: torch.optim.Adam(params, lr=0.01)

    program.train(
        dataset["train"],
        valid_set=dataset["dev"],
        train_epoch_num=25,
        Optim=optimizer_factory,
        device="cpu",
    )

    program.test(dataset["test"], device="cpu")

    sample = dataset["test"][0]
    datanode = next(program.populate([sample], device="cpu"))
    _describe_predictions(datanode)


if __name__ == "__main__":
    main()
