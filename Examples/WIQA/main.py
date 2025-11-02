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

print("Graph Declaration:")
Graph.clear()
Concept.clear()
Relation.clear()

with Graph('WIQA_graph') as graph:
    paragraph = Concept(name='paragraph')
    question = Concept(name='question')
    para_quest_contains, = paragraph.contains(question)

    is_more = question(name='is_more')
    is_less = question(name='is_less')
    no_effect = question(name='no_effect')

    symmetric = Concept(name='symmetric')
    s_arg1, s_arg2 = symmetric.has_a(s_arg1=question, s_arg2=question)

    transitive = Concept(name='transitive')
    t_arg1, t_arg2, t_arg3 = transitive.has_a(t_arg1=question, t_arg2=question, t_arg3=question)

    ifL(question("x"),exactL(is_more(path=("x")), is_less(path=("x")), no_effect(path=("x")), 1))

    ifL(symmetric('x'), ifL(is_more(path=('x', s_arg1)), is_less(path=('x', s_arg2))))
    ifL(symmetric('x'), ifL(is_less(path=('x', s_arg1)), is_less(path=('x', s_arg2))))

    ifL(transitive('x'), ifL(andL(is_more(path=('x', t_arg1)), is_more(path=('x', t_arg2))), is_more(path=('x', t_arg3))))
    ifL(transitive('x'), ifL(andL(is_less(path=('x', t_arg1)), is_less(path=('x', t_arg2))), is_less(path=('x', t_arg3))))

def random_wiqa_instance():
    paragraph_id = [0]
    questions_id = [i for i in range(5)]
    is_more, is_less, no_effect = [], [], []
    for _ in range(5):
        lbl = random.choice([0, 1, 2])
        is_more.append(1 if lbl == 0 else 0)
        is_less.append(1 if lbl == 1 else 0)
        no_effect.append(1 if lbl == 2 else 0)

    data = {
        "paragraph_id": paragraph_id,
        "question_id":  questions_id,
        "is_more_id":  [i for i in range(len(is_more))],
        "is_less_id":  [i for i in range(len(is_less))],
        "no_effect_id":  [i for i in range(len(no_effect))],

        "is_more": is_more,
        "is_less": is_less,
        "no_effect": no_effect,
    }

    para_quest_contains = list()
    for para in data["paragraph_id"]:
        for quest in data["question_id"]:
            para_quest_contains.append((para, quest))
    data["para_quest_contains"] = [para_quest_contains]

    symmetric = list()
    for s_arg1 in data["question_id"]:
        for s_arg2 in data["question_id"]:
            if s_arg1 != s_arg2 and random.random() < 0.2:
                symmetric.append((s_arg1, s_arg2))
    data["symmetric"] = [symmetric]

    transitive = []
    q_ids = data["question_id"]
    n = len(q_ids)
    for i in q_ids:
        for j in q_ids:
            for k in q_ids:
                if i == j or k == i or k == j:
                    continue
                if random.random() < 0.15:
                    transitive.append((i, j, k))

    data["transitive"] = [transitive]
    return data

dataset = [random_wiqa_instance() for _ in range(1)]


paragraph['paragraph_id'] = ReaderSensor(keyword='paragraph_id')
paragraph['symmetric'] = ReaderSensor(keyword='symmetric')
paragraph['transitive'] = ReaderSensor(keyword='transitive')

question['question_id'] = ReaderSensor(keyword='question_id')
question['is_more_id'] = ReaderSensor(keyword='is_more_id')
question['is_less_id'] = ReaderSensor(keyword='is_less_id')
question['no_effect_id'] = ReaderSensor(keyword='no_effect_id')

question[para_quest_contains] = EdgeReaderSensor(paragraph['paragraph_id'], question['question_id'],keyword='para_quest_contains', relation=para_quest_contains)
symmetric[s_arg1.reversed, s_arg2.reversed] = ManyToManyReaderSensor(question['question_id'], question['question_id'],keyword='symmetric')
transitive[t_arg1.reversed, t_arg2.reversed, t_arg3.reversed] = ManyToManyReaderSensor(question['question_id'], question['question_id'],question['question_id'],keyword='transitive')

question[is_more] = LabelReaderSensor(keyword='is_more')
question[is_less] = LabelReaderSensor(keyword='is_less')
question[no_effect] = LabelReaderSensor(keyword='no_effect')
question[is_more] = DummyLearner('is_more_id')
question[is_less] = DummyLearner('is_less_id')
question[no_effect] = DummyLearner('no_effect_id')

program = SolverPOIProgram(graph,poi=[paragraph,question,is_less, is_more, no_effect, symmetric, transitive],inferTypes=['local/argmax'],loss=MacroAverageTracker(NBCrossEntropyLoss()),metric=PRF1Tracker())
for datanode in program.populate(dataset=dataset):
    datanode.inferILPResults()
    for num, question in enumerate(datanode.getChildDataNodes()):
        print("question relations", question.impactLinks)

        print(f"question {num} is_more:",question.getResult(is_more, 'local',"argmax"))
        print(f"question {num} is_less:",question.getResult(is_less, 'local',"argmax"))
        print(f"question {num} no_effect:",question.getResult(no_effect, 'local', "argmax"))

        print(f"question {num} is_more ILP:",question.getResult(is_more,"ILP"))
        print(f"question {num} is_less ILP:",question.getResult(is_less,"ILP"))
        print(f"question {num} no_effect ILP:",question.getResult(no_effect,"ILP"))

