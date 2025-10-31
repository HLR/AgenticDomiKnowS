import sys
sys.path.append('../../../')
sys.path.append('../../')
sys.path.append('./')
sys.path.append('../')
import torch, random
from domiknows.program.loss import *
from domiknows.program.metric import *
from domiknows.sensor.pytorch.learners import *
from domiknows.sensor.pytorch.sensors import *
from domiknows.sensor.pytorch.relation_sensors import *
from domiknows.graph.logicalConstrain import *
from domiknows.graph import *
from domiknows.sensor.pytorch.relation_sensors import *
from domiknows.program import *

with Graph('IMDB') as graph:
    review = Concept(name='review')
    positive = review(name='positive')
    negative = review(name='negative')

    xorL(positive, negative)

def random_IMDB_instance():
    reviews = [1]
    positive = [random.randint(0,1) for _ in reviews]
    negative = [random.randint(0,1) for _ in reviews]
    data = {
        "review_id": list(range(len(reviews))),
        "positive": positive,
        "negative": negative,
    }
    return data

dataset = [random_IMDB_instance() for _ in range(1)]

review['review_id'] = ReaderSensor(keyword='review_id')
review[positive] = LabelReaderSensor(keyword='positive')
review[negative] = LabelReaderSensor(keyword='negative')

review[positive] = DummyLearner('review_id', output_size=2)
review[negative] = DummyLearner('review_id', output_size=2)

program = SolverPOIProgram(graph, poi=[review, positive, negative], inferTypes=['local/argmax'], loss=MacroAverageTracker(NBCrossEntropyLoss()), metric=PRF1Tracker())
for datanode in program.populate(dataset=dataset):
    datanode.inferILPResults()
    print("positive :", datanode.getResult(positive, "local", "argmax"))
    print("positive ILP:", datanode.getResult(positive, "ILP"))
    print("negative :", datanode.getResult(negative, "local", "argmax"))
    print("negative ILP:", datanode.getResult(negative, "ILP"))