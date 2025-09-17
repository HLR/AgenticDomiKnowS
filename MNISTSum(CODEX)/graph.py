"""Knowledge graph and constraints for digit-pair addition."""
from __future__ import annotations

from domiknows.graph import Concept, Graph
from domiknows.graph.logicalConstrain import andL, ifL

Graph.clear()
Concept.clear()

with Graph("mnist_sum_codex") as graph:
    pair = Concept(name="pair")

    img_a = pair(name="img_a")
    img_b = pair(name="img_b")

    digit_a_logits = pair(name="digit_a_logits")
    digit_b_logits = pair(name="digit_b_logits")

    digit_a_value = pair(name="digit_a_value")
    digit_b_value = pair(name="digit_b_value")

    sum_label = pair(name="sum_label")

    # Enforce the arithmetic constraint: predicted digit_a + digit_b equals sum_label.
    for left_digit in range(10):
        for right_digit in range(10):
            expected_sum = left_digit + right_digit
            ifL(
                andL(
                    digit_a_value.eq(left_digit),
                    digit_b_value.eq(right_digit),
                ),
                sum_label.eq(expected_sum),
            )

__all__ = [
    "graph",
    "pair",
    "img_a",
    "img_b",
    "digit_a_logits",
    "digit_b_logits",
    "digit_a_value",
    "digit_b_value",
    "sum_label",
]
