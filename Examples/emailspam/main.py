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

with Graph('email_spam_consistency') as graph:
    email = Concept(name='email')

    m1 = email(name='model1_pred', ConceptClass=EnumConcept, values=['spam', 'not_spam'])
    m2 = email(name='model2_pred', ConceptClass=EnumConcept, values=['spam', 'not_spam'])

    xorL(m1.spam, m2.not_spam)

def random_emailspam_instance():
    email_id = [0]
    m1 = [random.randint(0,1)]
    m2 = [random.randint(0, 1)]

    data = {
        "email_id": email_id,
        "m1_id":  [i for i in range(len(m1))],
        "m2_id":  [i for i in range(len(m2))],
        "email": email,
        "m1": m1,
        "m2": m2,
    }
    return data

dataset = [random_emailspam_instance() for _ in range(1)]

email['email_id'] = ReaderSensor(keyword='email_id')
m1['m1_id'] = ReaderSensor(keyword='m1_id')
m2['m2_id'] = ReaderSensor(keyword='m2_id')

m1[m1] = LabelReaderSensor(keyword='m1')
m2[m2] = LabelReaderSensor(keyword='m2')

email[m1] = DummyLearner('email_id', output_size=2)
email[m2] = DummyLearner('email_id', output_size=2)

program = SolverPOIProgram(graph,poi=[email,m1,m2],inferTypes=['local/argmax'],loss=MacroAverageTracker(NBCrossEntropyLoss()),metric=PRF1Tracker())
for datanode in program.populate(dataset=dataset):
    datanode.inferILPResults()

    print(f"m1 :",datanode.getResult(m1,"local","argmax"))
    print(f"m1 ILP:", datanode.getResult(m1, "ILP"))

    print(f"m2 :",datanode.getResult(m2,"local","argmax"))
    print(f"m2 ILP:", datanode.getResult(m2, "ILP"))