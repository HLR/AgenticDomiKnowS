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
    review_id = [0]
    positive_labels = [random.randint(0,1)]
    negative_labels = [1 - positive_labels[0]]
    data = {
        "review_id": review_id,
        "review_text": ["This is a sample review text."],
        "positive": positive_labels,
        "negative": negative_labels,
    }
    return data

dataset = [random_IMDB_instance() for _ in range(1)]

review['review_id'] = ReaderSensor(keyword='review_id')
review['review_text'] = ReaderSensor(keyword='review_text')

review[positive] = LabelReaderSensor(keyword='positive')
review[negative] = LabelReaderSensor(keyword='negative')

review[positive] = LLMLearner(review["review_text"], prompt="Determine if the  movie review is positive.", classes=["false", "true"])
review[negative] = LLMLearner(review["review_text"], prompt="Determine if the  movie review is negative.", classes=["false", "true"])

program = SolverPOIProgram(graph, poi=[review], inferTypes=['local/argmax'], loss=MacroAverageTracker(NBCrossEntropyLoss()), metric=PRF1Tracker())
for datanode in program.populate(dataset=dataset):
    datanode.inferILPResults()
    print("positive :", datanode.getResult(positive, "local", "argmax"))
    print("positive ILP:", datanode.getResult(positive, "ILP"))
    print("negative :", datanode.getResult(negative, "local", "argmax"))
    print("negative ILP:", datanode.getResult(negative, "ILP"))
