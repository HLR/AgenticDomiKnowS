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

with Graph('propara') as graph:

    procedure = Concept(name="procedure")
    step = Concept(name="step")
    (procedure_contain_step, ) = procedure.contains(step)

    non_existence = step(name="non_existence")
    unknown_loc = step(name="unknown_location")
    known_loc = step(name="known_location")

    action = Concept(name="action")
    action_arg1, action_arg2 = action.has_a(arg1=step, arg2=step)

    create = action(name="create")
    destroy = action(name="destroy")
    other = action(name="other")

    ifL(step("x"), exactL(known_loc(path=("x")), unknown_loc(path=("x")), non_existence(path=("x")), 1))
    ifL(action("x"), exactL(create(path=("x")), destroy(path=("x")), other(path=("x")), 1))

    ifL(create("x"),
      andL(
          non_existence("y1",path=("x", action_arg1)),
          orL(
              unknown_loc(path=("x", action_arg2)),
              known_loc(path=("x", action_arg2))
          )
      )
    )

    ifL(destroy("x"),
        andL(
            orL(
                    known_loc(path=("x", action_arg1)),
                    unknown_loc(path=("x", action_arg1))
                ),
                non_existence(path=("x", action_arg2))
        )
    )

def random_propara_instance():
    procedure_ids = [0]
    num_steps = 6
    step_ids = list(range(num_steps))

    non_existence_labels = [random.randint(0, 1) for _ in step_ids]
    unknown_location_labels = [random.randint(0, 1) for _ in step_ids]
    known_location_labels = [random.randint(0, 1) for _ in step_ids]

    proc_step_pairs = [(procedure_ids[0], sid) for sid in step_ids]

    action_edges = []
    for i in range(5):
        action_edges.append((i, i+1))
    action_ids = list(range(len(action_edges)))

    create_labels = []
    destroy_labels = []
    other_labels = []
    for _ in action_ids:
        t = random.choice([0, 1, 2])
        create_labels.append(1 if t == 0 else 0)
        destroy_labels.append(1 if t == 1 else 0)
        other_labels.append(1 if t == 2 else 0)

    data = {
        "procedure_id": procedure_ids,
        "step_id": step_ids,

        "non_existence": non_existence_labels,
        "unknown_location": unknown_location_labels,
        "known_location": known_location_labels,

        "procedure_step_contains": [proc_step_pairs],

        "action": [action_edges],
        "action_id": action_ids,
        "create": create_labels,
        "destroy": destroy_labels,
        "other": other_labels,
    }
    return data

dataset = [random_propara_instance() for _ in range(1)]

procedure["procedure_id"] = ReaderSensor(keyword="procedure_id")
step["step_id"] = ReaderSensor(keyword="step_id")

step[procedure_contain_step] = EdgeReaderSensor(procedure["procedure_id"], step["step_id"], keyword="procedure_step_contains", relation=procedure_contain_step)

action[action_arg1.reversed, action_arg2.reversed] = ManyToManyReaderSensor(step["step_id"], step["step_id"], keyword="action")

action["action_id"] = ReaderSensor(keyword="action_id")

step[non_existence] = LabelReaderSensor(keyword="non_existence")
step[unknown_loc] = LabelReaderSensor(keyword="unknown_location")
step[known_loc] = LabelReaderSensor(keyword="known_location")

action[create] = LabelReaderSensor(keyword="create")
action[destroy] = LabelReaderSensor(keyword="destroy")
action[other] = LabelReaderSensor(keyword="other")

step[non_existence] = DummyLearner("step_id", output_size=2)
step[unknown_loc] = DummyLearner("step_id", output_size=2)
step[known_loc] = DummyLearner("step_id", output_size=2)

action[create] = DummyLearner("action_id", output_size=2)
action[destroy] = DummyLearner("action_id", output_size=2)
action[other] = DummyLearner("action_id", output_size=2)

program = SolverPOIProgram(graph, poi=[procedure, step, non_existence, unknown_loc, known_loc, action, create, destroy, other], inferTypes=["local/argmax"], loss=MacroAverageTracker(NBCrossEntropyLoss()), metric=PRF1Tracker())

for datanode in program.populate(dataset=dataset):
    datanode.inferILPResults()

    for idx, st in enumerate(datanode.getChildDataNodes()):
        print("step", idx, "non_existence :", st.getResult(non_existence, "local", "argmax"))
        print("step", idx, "non_existence ILP:", st.getResult(non_existence, "ILP"))
        print("step", idx, "unknown_location :", st.getResult(unknown_loc, "local", "argmax"))
        print("step", idx, "unknown_location ILP:", st.getResult(unknown_loc, "ILP"))
        print("step", idx, "known_location :", st.getResult(known_loc, "local", "argmax"))
        print("step", idx, "known_location ILP:", st.getResult(known_loc, "ILP"))

        if not idx == len(datanode.getChildDataNodes())-1:
            current_action = st.impactLinks["arg1"][0]
            print("action", current_action, "create :", current_action.getResult(create, "local", "argmax"))
            print("action", current_action, "create ILP:", current_action.getResult(create, "ILP"))
            print("action", current_action, "destroy :", current_action.getResult(destroy, "local", "argmax"))
            print("action", current_action, "destroy ILP:", current_action.getResult(destroy, "ILP"))
            print("action", current_action, "other :", current_action.getResult(other, "local", "argmax"))
            print("action", current_action, "other ILP:", current_action.getResult(other, "ILP"))
