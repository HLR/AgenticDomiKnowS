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

with Graph('belief_bank') as graph:

    subject = Concept(name='subject')
    facts = Concept(name='facts')
    subject_facts_contains, = subject.contains(facts)

    fact_check = facts(name='fact_check')
    implication = Concept(name='implication')
    i_arg1, i_arg2 = implication.has_a(iarg1=facts, iarg2=facts)

    nimplication = Concept(name='nimplication')
    ni_arg1, ni_arg2 = nimplication.has_a(narg1=facts, narg2=facts)

    ifL(andL(fact_check('x'), existsL(implication('s', path=('x', i_arg1.reversed)))), fact_check(path=('s', i_arg2)))
    ifL(andL(fact_check('x'), existsL(nimplication('s', path=('x', ni_arg1.reversed)))), notL(fact_check(path=('s', ni_arg2))))

def random_belief_bank_instance():
    subject_ids = [0]
    facts_ids = list(range(6))
    fact_check_labels = [random.randint(0, 1) for _ in facts_ids]

    contains_pairs = [(subject_ids[0], f) for f in facts_ids]

    implication_pairs = []
    for _ in range(5):
        a, b = random.sample(facts_ids, 2)
        implication_pairs.append((a, b))

    nimplication_pairs = []
    for _ in range(5):
        a, b = random.sample(facts_ids, 2)
        nimplication_pairs.append((a, b))

    data = {
        "subject_id": subject_ids,
        "facts_id": facts_ids,
        "fact_check": fact_check_labels,
        "subject_facts_contains": [contains_pairs],
        "implication": [implication_pairs],
        "nimplication": [nimplication_pairs],
    }
    return data

dataset = [random_belief_bank_instance() for _ in range(1)]

subject['subject_id'] = ReaderSensor(keyword='subject_id')
facts['facts_id'] = ReaderSensor(keyword='facts_id')
facts[fact_check] = LabelReaderSensor(keyword='fact_check')

facts[subject_facts_contains] = EdgeReaderSensor(subject['subject_id'], facts['facts_id'], keyword='subject_facts_contains', relation=subject_facts_contains)
implication[i_arg1.reversed, i_arg2.reversed] = ManyToManyReaderSensor(facts['facts_id'], facts['facts_id'], keyword='implication')
nimplication[ni_arg1.reversed, ni_arg2.reversed] = ManyToManyReaderSensor(facts['facts_id'], facts['facts_id'], keyword='nimplication')

facts[fact_check] = DummyLearner('facts_id', output_size=2)

program = SolverPOIProgram(graph, poi=[subject, facts, fact_check, implication, nimplication], inferTypes=['local/argmax'], loss=MacroAverageTracker(NBCrossEntropyLoss()), metric=PRF1Tracker())
for datanode in program.populate(dataset=dataset):
    datanode.inferILPResults()
    for idx, fnode in enumerate(datanode.getChildDataNodes()):
        print("fact", idx, "fact_check :", fnode.getResult(fact_check, "local", "argmax"))
        print("fact", idx, "fact_check ILP:", fnode.getResult(fact_check, "ILP"))
