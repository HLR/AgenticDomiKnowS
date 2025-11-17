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

with Graph('AnimalAndFlower') as graph:
    image_group = Concept(name='image_group')
    image = Concept(name='image')
    image_group_contains, = image_group.contains(image)

    animal = image(name='animal')
    cat = animal(name='cat')
    dog = animal(name='dog')
    monkey = animal(name='monkey')
    squirrel = animal(name='squirrel')

    flower = image(name='flower')
    daisy = flower(name='daisy')
    dandelion = flower(name='dandelion')
    rose = flower(name='rose')
    sunflower = flower(name='sunflower')
    tulip = flower(name='tulip')

    ifL(flower, orL(daisy, dandelion, rose, sunflower, tulip))
    ifL(animal, orL(cat, dog, monkey, squirrel))

    ifL( image, exactL(cat, dog, monkey, squirrel, daisy, dandelion, rose, sunflower, tulip, 1))
    ifL(image, xorL(animal, flower, 1))

def random_animalandflower_instance():
    n_images = 6
    image_group_ids = [0]
    image_ids = list(range(n_images))

    animals = ['cat','dog','monkey','squirrel']
    flowers = ['daisy','dandelion','rose','sunflower','tulip']
    leaves = animals + flowers

    labels = {name: [0]*n_images for name in ['animal','flower'] + leaves}
    for i in range(n_images):
        k = random.randint(0, len(leaves)-1)
        for name in leaves:
            labels[name][i] = 0
        labels[leaves[k]][i] = 1
        labels['animal'][i] = 1 if k < len(animals) else 0
        labels['flower'][i] = 1 if k >= len(animals) else 0

    contains_pairs = [(image_group_ids[0], img_id) for img_id in image_ids]

    data = {
        "image_group_id": image_group_ids,
        "image_id": image_ids,
        "image_pixels": image_ids,
        "image_group_image_contains": [contains_pairs],
    }
    data.update(labels)
    return data

dataset = [random_animalandflower_instance() for _ in range(1)]

image_group['image_group_id'] = ReaderSensor(keyword='image_group_id')
image['image_id'] = ReaderSensor(keyword='image_id')
image['image_pixels'] = ReaderSensor(keyword='image_pixels')

image[image_group_contains] = EdgeReaderSensor(image_group['image_group_id'], image['image_id'], keyword='image_group_image_contains', relation=image_group_contains)

image[animal] = LabelReaderSensor(keyword='animal')
image[flower] = LabelReaderSensor(keyword='flower')
image[cat] = LabelReaderSensor(keyword='cat')
image[dog] = LabelReaderSensor(keyword='dog')
image[monkey] = LabelReaderSensor(keyword='monkey')
image[squirrel] = LabelReaderSensor(keyword='squirrel')
image[daisy] = LabelReaderSensor(keyword='daisy')
image[dandelion] = LabelReaderSensor(keyword='dandelion')
image[rose] = LabelReaderSensor(keyword='rose')
image[sunflower] = LabelReaderSensor(keyword='sunflower')
image[tulip] = LabelReaderSensor(keyword='tulip')

image[animal] = LLMLearner(image["image_pixels"], prompt="Classify the image and predict whether it is an animal or not.",classes=["false","true"])
image[flower] = LLMLearner(image["image_pixels"], prompt="Classify the image and predict whether it is an flower or not.",classes=["false","true"])
image[cat] = LLMLearner(image["image_pixels"], prompt="Classify the image and predict whether it is an cat or not.",classes=["false","true"])
image[dog] = LLMLearner(image["image_pixels"], prompt="Classify the image and predict whether it is an dog or not.",classes=["false","true"])
image[monkey] = LLMLearner(image["image_pixels"], prompt="Classify the image and predict whether it is an monkey or not.",classes=["false","true"])
image[squirrel] = LLMLearner(image["image_pixels"], prompt="Classify the image and predict whether it is an squirrel or not.",classes=["false","true"])
image[daisy] = LLMLearner(image["image_pixels"], prompt="Classify the image and predict whether it is an daisy or not.",classes=["false","true"])
image[dandelion] = LLMLearner(image["image_pixels"], prompt="Classify the image and predict whether it is an dandelion or not.",classes=["false","true"])
image[rose] = LLMLearner(image["image_pixels"], prompt="Classify the image and predict whether it is an rose or not.",classes=["false","true"])
image[sunflower] = LLMLearner(image["image_pixels"], prompt="Classify the image and predict whether it is an sunflower or not.",classes=["false","true"])
image[tulip] = LLMLearner(image["image_pixels"], prompt="Classify the image and predict whether it is an tulip or not.",classes=["false","true"])

program = SolverPOIProgram(
    graph,
    poi=[image_group, image, animal, flower, cat, dog, monkey, squirrel, daisy, dandelion, rose, sunflower, tulip],
    inferTypes=['local/argmax'],
    loss=MacroAverageTracker(NBCrossEntropyLoss()),
    metric=PRF1Tracker()
)

for datanode in program.populate(dataset=dataset):
    datanode.inferILPResults()
    for child in datanode.getChildDataNodes():
        print("animal :", child.getResult(animal, "local", "argmax"))
        print("animal ILP:", child.getResult(animal, "ILP"))
        print("flower :", child.getResult(flower, "local", "argmax"))
        print("flower ILP:", child.getResult(flower, "ILP"))

        print("cat :", child.getResult(cat, "local", "argmax"))
        print("cat ILP:", child.getResult(cat, "ILP"))
        print("dog :", child.getResult(dog, "local", "argmax"))
        print("dog ILP:", child.getResult(dog, "ILP"))
        print("monkey :", child.getResult(monkey, "local", "argmax"))
        print("monkey ILP:", child.getResult(monkey, "ILP"))
        print("squirrel :", child.getResult(squirrel, "local", "argmax"))
        print("squirrel ILP:", child.getResult(squirrel, "ILP"))

        print("daisy :", child.getResult(daisy, "local", "argmax"))
        print("daisy ILP:", child.getResult(daisy, "ILP"))
        print("dandelion :", child.getResult(dandelion, "local", "argmax"))
        print("dandelion ILP:", child.getResult(dandelion, "ILP"))
        print("rose :", child.getResult(rose, "local", "argmax"))
        print("rose ILP:", child.getResult(rose, "ILP"))
        print("sunflower :", child.getResult(sunflower, "local", "argmax"))
        print("sunflower ILP:", child.getResult(sunflower, "ILP"))
        print("tulip :", child.getResult(tulip, "local", "argmax"))
        print("tulip ILP:", child.getResult(tulip, "ILP"))