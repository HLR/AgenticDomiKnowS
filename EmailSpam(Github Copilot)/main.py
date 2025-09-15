import json
from graph import graph
from sensors import header_sensor, body_sensor
from learners import model1_learner, model2_learner
from constraints import constraint
from program import build_program

def load_dataset(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

def main():
    # Load dataset
    dataset = load_dataset("EmailSpam/data/emails.jsonl")

    # Build graph
    sensors = {"header": header_sensor, "body": body_sensor}
    learners = {"model1": model1_learner, "model2": model2_learner}
    constraints = {"consistency": constraint}
    program = build_program(graph, sensors, learners, constraints)

    # Train and evaluate
    program.train(dataset["train"])
    program.evaluate(dataset["dev"])

    # Test inference
    test_example = {"header": "Special Offer!", "body": "Get 50% off now."}
    prediction = program.infer(test_example)
    print("Prediction:", prediction)

if __name__ == "__main__":
    main()