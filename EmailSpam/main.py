
import torch
from data import get_data
from graph import graph
from sensors import HeaderSensor, BodySensor, SpamLabelSensor, SpamModel1, SpamModel2
from program import build_program

def main():
    # Get data
    data = get_data()

    # Define sensors
    header_sensor = HeaderSensor(header='header')
    body_sensor = BodySensor(body='body')
    spam_label_sensor = SpamLabelSensor(spam='spam')

    # Define learners
    model1 = SpamModel1(header_sensor, body_sensor)
    model2 = SpamModel2(header_sensor, body_sensor)

    # Assign sensors and learners to the graph
    graph['email']['header'] = header_sensor
    graph['email']['body'] = body_sensor
    graph['email']['spam'] = spam_label_sensor
    graph['email']['model1_spam'] = model1
    graph['email']['model2_spam'] = model2

    # Build program
    program = build_program(graph, None, None, None) # sensors, learners, and constraints are already in the graph

    # Manually train the model
    optimizer = torch.optim.Adam(list(model1.parameters()) + list(model2.parameters()))
    for epoch in range(10):
        for email_data in data:
            optimizer.zero_grad()

            # Forward pass
            output1 = model1.forward([email_data['header']], [email_data['body']])
            output2 = model2.forward([email_data['header']], [email_data['body']])
            
            # Calculate loss
            loss1 = torch.nn.BCEWithLogitsLoss()(output1, torch.tensor([[float(email_data['spam'])]]))
            loss2 = torch.nn.BCEWithLogitsLoss()(output2, torch.tensor([[float(email_data['spam'])]]))
            loss = loss1 + loss2

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

    # Inference example
    test_email = {'header': 'Free money!', 'body': 'Click here to get rich quick!'}
    output1 = model1.forward([test_email['header']], [test_email['body']])
    output2 = model2.forward([test_email['header']], [test_email['body']])
    print(f'Inference on: {test_email}')
    print(f'Spam prediction (Model 1): {torch.sigmoid(output1)}')
    print(f'Spam prediction (Model 2): {torch.sigmoid(output2)}')

if __name__ == '__main__':
    main()
