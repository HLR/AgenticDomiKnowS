"""Program loader for the MNIST sum example."""
from domiknows.program import SolverPOIProgram
from domiknows.program.loss import NBCrossEntropyLoss
from domiknows.program.metric import DatanodeCMMetric, MacroAverageTracker, PRF1Tracker

from .graph import digit_a_logits, digit_b_logits, graph


def build_program() -> SolverPOIProgram:
    poi = [digit_a_logits, digit_b_logits]
    loss = MacroAverageTracker(NBCrossEntropyLoss())
    metric = {"argmax": PRF1Tracker(DatanodeCMMetric("local/argmax"))}

    return SolverPOIProgram(
        graph,
        poi=poi,
        inferTypes=["local/argmax"],
        loss=loss,
        metric=metric,
    )


__all__ = ["build_program"]
