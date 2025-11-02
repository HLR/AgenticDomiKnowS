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

with Graph('ruletaker_bank') as graph:

    context = Concept(name='context')
    question = Concept(name='question')
    context_question_contains, = context.contains(question)

    qlabel = question(name='qlabel')
    implication = Concept(name='implication')
    i_arg1, i_arg2 = implication.has_a(arg1=question, arg2=question)

    ifL(andL(qlabel('x'), existsL(implication('s', path=('x', i_arg1.reversed)))), qlabel(path=('s', i_arg2)))


def random_ruletaker_bank_instance():
    context_id = [0]
    num_questions = 6
    question_ids = list(range(num_questions))
    qlabel = [random.randint(0, 1) for _ in question_ids]

    contains_pairs = [(0, qid) for qid in question_ids]

    all_pairs = [(i, j) for i in question_ids for j in question_ids if i != j]
    random.shuffle(all_pairs)
    implication_pairs = all_pairs[:max(5, len(all_pairs)//2)]

    data = {
        "context_id": context_id,
        "question_id": question_ids,
        "qlabel": qlabel,
        "context_question_contains": [contains_pairs],
        "implication": [implication_pairs],
    }
    return data

dataset = [random_ruletaker_bank_instance() for _ in range(1)]

context['context_id'] = ReaderSensor(keyword='context_id')
question['question_id'] = ReaderSensor(keyword='question_id')

question[context_question_contains] = EdgeReaderSensor(context['context_id'], question['question_id'], keyword='context_question_contains', relation=context_question_contains)
implication[i_arg1.reversed, i_arg2.reversed] = ManyToManyReaderSensor(question['question_id'], question['question_id'], keyword='implication')

question[qlabel] = LabelReaderSensor(keyword='qlabel')
question[qlabel] = DummyLearner('question_id', output_size=2)

program = SolverPOIProgram(graph, poi=[context, question, qlabel, implication], inferTypes=['local/argmax'], loss=MacroAverageTracker(NBCrossEntropyLoss()), metric=PRF1Tracker())
for datanode in program.populate(dataset=dataset):
    datanode.inferILPResults()

    for idx, qnode in enumerate(datanode.getChildDataNodes()):
        print("relations", qnode.impactLinks)
        print(f"question {idx} qlabel :", qnode.getResult(qlabel, "local", "argmax"))
        print(f"question {idx} qlabel ILP:", qnode.getResult(qlabel, "ILP"))