
from domiknows.program import SolverPOIProgram
from domiknows.program.metric import PRF1Tracker, DatanodeCMMetric
from domiknows.program.loss import NBCrossEntropyLoss
from graph import spam, model1_spam, model2_spam

def build_program(graph, sensors, learners, constraints):
    # Points of Interest
    poi = [spam, model1_spam, model2_spam]

    # Program
    program = SolverPOIProgram(
        graph,
        poi=poi,
        inferTypes=['local/argmax'],
        loss=NBCrossEntropyLoss(),
        metric=PRF1Tracker(DatanodeCMMetric('local/argmax'))
    )

    return program
