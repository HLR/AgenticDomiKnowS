"""Program assembly for the email spam example."""
from domiknows.program import SolverPOIProgram
from domiknows.program.loss import NBCrossEntropyLoss
from domiknows.program.metric import DatanodeCMMetric, MacroAverageTracker, PRF1Tracker

from .graph import graph, model1_logits, model2_logits


def build_program() -> SolverPOIProgram:
    poi = [model1_logits, model2_logits]
    loss = MacroAverageTracker(NBCrossEntropyLoss())
    metric = {"argmax": PRF1Tracker(DatanodeCMMetric("local/argmax"))}

    return SolverPOIProgram(
        graph,
        poi=poi,
        inferTypes=["local/argmax"],
        loss=loss,
        metric=metric,
    )
