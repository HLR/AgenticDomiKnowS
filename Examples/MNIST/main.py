import sys
sys.path.append('../../../../')
sys.path.append('../../../')
sys.path.append('../')
sys.path.append('../../')
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

with Graph(name='MNIST_sum') as graph:
    image_batch = Concept(name='image_batch')
    image = Concept(name='image')
    digit = image(name='digits', ConceptClass=EnumConcept, values=['n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9'])
    image_contains, = image_batch.contains(image)

    image_pair = Concept(name='pair')
    s = image_pair(name='summations', ConceptClass=EnumConcept, values=['s0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18'])
    pair_d0, pair_d1 = image_pair.has_a(digit0=image, digit1=image)

    for sum_val in range(19):
        sum_combinations = []

        for d0_val in range(sum_val + 1):
            d1_val = sum_val - d0_val
            if d0_val >= 10 or d1_val >= 10:
                continue
            sum_combinations.append(andL(getattr(digit, "n"+str(d0_val))(path=('x', pair_d0)),getattr(digit, "n"+str(d1_val))(path=('x', pair_d1))))

        if len(sum_combinations) == 1:
            ifL(getattr(s, "s"+str(sum_val))('x'),sum_combinations[0])
        else:
            ifL(getattr(s, "s"+str(sum_val))('x'),orL(*sum_combinations))

def random_mnist_sum_instance():
    n_images = 6
    image_batch_id = [0]
    image_id = list(range(n_images))
    digits = [random.randint(0, 9) for _ in range(n_images)]
    image_batch_image_contains = [(0, i) for i in image_id]
    pair_tuples = [(a,b) for a in image_id for b in image_id if a != b]
    pair_id = list(range(len(pair_tuples)))
    summations = [min(digits[a] + digits[b], 18) for a, b in pair_tuples]

    data = {
        "image_batch_id": image_batch_id,
        "image_id": image_id,
        "pair_id": pair_id,
        "digits": digits,
        "summations": summations,
        "image_batch_image_contains": [image_batch_image_contains],
        "pair_has_a": [pair_tuples],
    }
    return data

dataset = [random_mnist_sum_instance() for _ in range(1)]

image_batch['image_batch_id'] = ReaderSensor(keyword='image_batch_id')
image['image_id'] = ReaderSensor(keyword='image_id')
image_pair['pair_id'] = ReaderSensor(keyword='pair_id')

image[digit] = LabelReaderSensor(keyword='digits')
image_pair[s] = LabelReaderSensor(keyword='summations')

image[image_contains] = EdgeReaderSensor(image_batch['image_batch_id'], image['image_id'], keyword='image_batch_image_contains', relation=image_contains)
image_pair[pair_d0.reversed, pair_d1.reversed] = ManyToManyReaderSensor(image['image_id'], image['image_id'], keyword='pair_has_a')

image[digit] = DummyLearner('image_id', output_size=10)
image_pair[s] = DummyLearner('pair_id', output_size=19)

program = SolverPOIProgram(graph, poi=[image, digit, image_pair, s], inferTypes=['local/argmax'], loss=MacroAverageTracker(NBCrossEntropyLoss()), metric=PRF1Tracker())
for datanode in program.populate(dataset=dataset):
    datanode.inferILPResults()
    for idx, img_node in enumerate(datanode.getChildDataNodes()):
        print("image relations", img_node.impactLinks)
        print("image", idx, "digits :", img_node.getResult(digit, "local", "argmax"))
        print("image", idx, "digits ILP:", img_node.getResult(digit, "ILP"))