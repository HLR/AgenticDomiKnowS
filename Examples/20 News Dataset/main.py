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

with Graph('20news') as graph:
    news_group = Concept(name="news_group")

    news = Concept(name='news')
    news_group_contains, = news_group.contains(news)

    level1_list = ["comp", "rec", "sci", "misc", "talk", "alt", "soc", "None"]
    level2_list = ["os", "sys", "windows", "graphics", "motorcycles", "sport", "autos", "religion", "electronics", "med", "space", "forsale", "politics", "crypt", "None"]
    level3_list = ["guns", "ibm", "mac", "baseball", "hockey", "mideast", "None"]

    level1 = news(name="level1", ConceptClass=EnumConcept, values=level1_list)
    level2 = news(name="level2", ConceptClass=EnumConcept, values=level2_list)
    level3 = news(name="level3", ConceptClass=EnumConcept, values=level3_list)

    hierarchy_1 = {
        "comp": {"graphics", "os", "sys", "windows"},
        "rec": {"autos", "motorcycles", "sport"},
        "sci": {"crypt", "electronics", "med", "space"},
        "misc": {"forsale"},
        "talk": {"politics", "religion"},
        "alt": {},
        "soc": {},
        "None": {},
    }
    hierarchy_2 = {
        "windows": {},
        "os": {},
        "religion": {},
        "politics": {"guns", "mideast"},
        "sys": {"ibm", "mac"},
        "sport": {"hockey", "baseball"}
    }

    for parent in hierarchy_1.keys():
        if hierarchy_1[parent]:
            ifL(existsL(*[level2.__getattr__(child) for child in hierarchy_1[parent]]), level1.__getattr__(parent))
        else:
            ifL(level1.__getattr__(parent), andL(level2.__getattr__("None"), level3.__getattr__("None")))

    for parent in hierarchy_2.keys():
        if hierarchy_2[parent]:
            ifL(existsL(*[level3.__getattr__(child) for child in hierarchy_2[parent]]), level2.__getattr__(parent))
        else:
            ifL(level2.__getattr__(parent), level3.__getattr__("None"))

def random_20news_instance():
    news_group = [0]
    num_news = 6
    news_items = list(range(num_news))
    level1_vals = [random.randint(0, 7) for _ in news_items]
    level2_vals = [random.randint(0, 14) for _ in news_items]
    level3_vals = [random.randint(0, 6) for _ in news_items]

    data = {
        "news_group_id": news_group,
        "news_id": [i for i in range(len(news_items))],
        "level1_id": [i for i in range(len(news_items))],
        "level2_id": [i for i in range(len(news_items))],
        "level3_id": [i for i in range(len(news_items))],
        "level1": level1_vals,
        "level2": level2_vals,
        "level3": level3_vals,
    }

    ng_news_contains = []
    for ng in data["news_group_id"]:
        for n in data["news_id"]:
            ng_news_contains.append((ng, n))
    data["news_group_news_contains"] = [ng_news_contains]
    return data

dataset = [random_20news_instance() for _ in range(1)]

news_group['news_group_id'] = ReaderSensor(keyword='news_group_id')
news['news_id'] = ReaderSensor(keyword='news_id')
news[news_group_contains] = EdgeReaderSensor(news_group['news_group_id'], news['news_id'], keyword='news_group_news_contains', relation=news_group_contains)

level1['level1_id'] = ReaderSensor(keyword='level1_id')
level2['level2_id'] = ReaderSensor(keyword='level2_id')
level3['level3_id'] = ReaderSensor(keyword='level3_id')

level1[level1] = LabelReaderSensor(keyword='level1')
level2[level2] = LabelReaderSensor(keyword='level2')
level3[level3] = LabelReaderSensor(keyword='level3')

news[level1] = DummyLearner('news_id', output_size=8)
news[level2] = DummyLearner('news_id', output_size=15)
news[level3] = DummyLearner('news_id', output_size=7)

program = SolverPOIProgram(graph, poi=[news_group, news, level1, level2, level3], inferTypes=['local/argmax'], loss=MacroAverageTracker(NBCrossEntropyLoss()), metric=PRF1Tracker())
for datanode in program.populate(dataset=dataset):
    datanode.inferILPResults()
    for i, child in enumerate(datanode.getChildDataNodes()):
        print("news", i, "level1 :", child.getResult(level1, "local", "argmax"))
        print("news", i, "level1 ILP:", child.getResult(level1, "ILP"))
        print("news", i, "level2 :", child.getResult(level2, "local", "argmax"))
        print("news", i, "level2 ILP:", child.getResult(level2, "ILP"))
        print("news", i, "level3 :", child.getResult(level3, "local", "argmax"))
        print("news", i, "level3 ILP:", child.getResult(level3, "ILP"))