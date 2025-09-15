
import torch
from domiknows.program import POIProgram
from domiknows.program.metric import MacroAverageTracker, MetricTracker
from domiknows.program.model.pytorch import PoiModel
from domiknows.graph import DataNode

from data import get_data
from graph import graph
from sensors import get_sensors

import logging
logging.basicConfig(level=logging.INFO)

def main():
    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Data ---
    dataset = get_data(size=2000) # Increased size for better training
    datanode = Datanode(dataset, name='mnist_sum_dataset')

    # --- Graph and Sensors ---
    sensors = get_sensors(graph, device)
    img_a_s = sensors['img_a']
    img_b_s = sensors['img_b']
    sum_s = sensors['sum']
    digit_a_l = sensors['digit_a_learner']
    digit_b_l = sensors['digit_b_learner'] # This is the same instance as digit_a_l
    digit_a_label_s = sensors['digit_a_label']
    digit_b_label_s = sensors['digit_b_label']

    # --- Program Definition ---
    # Define the model
    model = PoiModel(graph, metric=MacroAverageTracker(MetricTracker(graph)))
    program = POIProgram(graph, model, device=device)


    # --- Connect Data to Graph ---
    # Learners for digit classification
    program.add_learner(digit_a_l, 'digit_a', [img_a_s])
    program.add_learner(digit_b_l, 'digit_b', [img_b_s])

    # Sensor for the sum label
    program.add_sensor(sum_s, 'sum_label')

    # Sensors for pre-training labels
    program.add_sensor(digit_a_label_s, 'digit_a', label=True)
    program.add_sensor(digit_b_label_s, 'digit_b', label=True)

    # --- Training ---
    print("--- Starting Pre-training ---")
    # Pre-train the digit classifier without the sum constraint
    # This helps the model learn the basics of digit recognition first
    for _ in range(5): # 5 epochs of pre-training
        program.train(datanode, train_epoch_num=1, Optim=torch.optim.Adam, device=device)

    print("--- Starting Training with Constraints ---")
    # Activate the constraints and train the full model
    # The framework automatically adds the logical constraint loss (LC Loss)
    for _ in range(10): # 10 epochs of full training
        program.train(datanode, train_epoch_num=1, Optim=torch.optim.Adam, device=device)

    # --- Inference and Evaluation ---
    print("--- Running Inference ---")
    
    # Get a few examples to test
    test_examples = dataset[:5]
    for i, item in enumerate(test_examples):
        datanode_item = Datanode([item], name=f'item_{i}')
        
        # Run inference with ILP
        result = program.infer_ir(datanode_item)
        
        # Extract predictions
        pred_a = int(result[0]['digit_a'].argmax())
        pred_b = int(result[0]['digit_b'].argmax())
        
        true_a = item['digit_a']
        true_b = item['digit_b']
        true_sum = item['sum']
        
        print(f"\n--- Example {i+1} ---")
        print(f"True Digits: {true_a} + {true_b} = {true_sum}")
        print(f"Predicted Digits: {pred_a} + {pred_b} = {pred_a + pred_b}")
        
        if (pred_a + pred_b) == true_sum:
            print("Constraint SATISFIED")
        else:
            print("Constraint VIOLATED")

if __name__ == '__main__':
    main()
