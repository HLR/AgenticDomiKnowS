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

with Graph('CIFAR10_hier') as graph:
    image = Concept(name='image')

    animals = ['bird','cat','deer','dog','frog','horse']
    vehicles = ['airplane','automobile','ship','truck']
    coarse = image(name='coarse', ConceptClass=EnumConcept, values=['animal', 'vehicle'])
    fine = image(name='fine', ConceptClass=EnumConcept, values=animals + vehicles)

    ifL(orL(*[fine.__getattr__(l) for l in animals]), coarse.__getattr__('animal'))
    ifL(orL(*[fine.__getattr__(l) for l in vehicles]), coarse.__getattr__('vehicle'))

def random_cifar10_instance():
    image_id = [0]
    coarse = [random.randint(0,1)]
    fine = [random.randint(0, 9)]

    data = {
        "image_id": image_id,
        "coarse_id":  [i for i in range(len(coarse))],
        "fine_id":  [i for i in range(len(fine))],
        "image": image,
        "coarse": coarse,
        "fine": fine,
    }
    return data

dataset = [random_cifar10_instance() for _ in range(1)]

image['image_id'] = ReaderSensor(keyword='image_id')
coarse['coarse_id'] = ReaderSensor(keyword='coarse_id')
fine['fine_id'] = ReaderSensor(keyword='fine_id')

image[coarse] = LabelReaderSensor(keyword='coarse')
image[fine] = LabelReaderSensor(keyword='fine')
image[coarse] = DummyLearner('image_id', output_size=2)
image[fine] = DummyLearner('image_id', output_size=10)

program = SolverPOIProgram(graph,poi=[image,coarse,fine],inferTypes=['local/argmax'],loss=MacroAverageTracker(NBCrossEntropyLoss()),metric=PRF1Tracker())
for datanode in program.populate(dataset=dataset):
    datanode.inferILPResults()

    print(f"coarse :",datanode.getResult(coarse,"local","argmax"))
    print(f"coarse ILP:", datanode.getResult(coarse, "ILP"))

    print(f"fine :",datanode.getResult(fine,"local","argmax"))
    print(f"fine ILP:", datanode.getResult(fine, "ILP"))



