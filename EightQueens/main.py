import torch
from domiknows.program.lossprogram import PrimalDualProgram
from domiknows.program.model.pytorch import SolverModel
from domiknows.program.loss import NBCrossEntropyLoss
from domiknows.program.metric import MacroAverageTracker, PRF1Tracker, DatanodeCMMetric
from domiknows.sensor.pytorch.sensors import FunctionalSensor
from domiknows.sensor.pytorch.learners import ModuleLearner
from graph import graph, queens
from sensors import BoardSensor, QueenLearner
from data import create_data

def main():
    # Load data
    dataset = create_data(num_samples=500)

    # Define sensors and learners
    board_sensor = BoardSensor(keyword='board')
    queen_learner_model = QueenLearner()
    queen_learner = ModuleLearner(board_sensor, module=queen_learner_model)

    # Assign sensors to the graph
    for i in range(8):
        for j in range(8):
            def forward_fn(logits, i=i, j=j):
                return logits[:, i, j]
            queens[i][j]['pred'] = FunctionalSensor(queen_learner, forward=forward_fn)

    # Define the program
    program = PrimalDualProgram(
        graph,
        SolverModel,
        poi=queens,
        inferTypes=['local/argmax'],
        loss=MacroAverageTracker(NBCrossEntropyLoss()),
        metric={'argmax': PRF1Tracker(DatanodeCMMetric('local/argmax'))}
    )

    # Train the model
    program.train(dataset, train_epoch_num=10, Optim=torch.optim.Adam, lr=0.001)

    # Get a prediction for one board
    board_data = dataset[0]
    datanode = program.populate(board_data)
    
    print("Input board:")
    print(board_data['board'])
    
    print("\nPredicted queens (logits):")
    predicted_board = torch.zeros(8, 8)
    for i in range(8):
        for j in range(8):
            predicted_board[i, j] = datanode.getAttribute(queens[i][j]['pred'])
    print(predicted_board.detach().numpy())

    # Get the final output
    datanode.inferILPResults()
    print("\nFinal output after ILP:")
    ilp_board = torch.zeros(8, 8)
    for i in range(8):
        for j in range(8):
            ilp_board[i, j] = datanode.getAttribute(queens[i][j], 'ILP')
    print(ilp_board.detach().numpy())

if __name__ == '__main__':
    main()