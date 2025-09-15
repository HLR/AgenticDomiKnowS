
import numpy as np
import random

def create_data(num_samples=100):
    dataset = []
    for _ in range(num_samples):
        board = np.zeros((8, 8), dtype=np.float32)
        k = random.randint(2, 5)
        for _ in range(k):
            while True:
                row, col = random.randint(0, 7), random.randint(0, 7)
                if board[row, col] == 0:
                    board[row, col] = 1
                    break
        dataset.append({'board': board})
    return dataset
