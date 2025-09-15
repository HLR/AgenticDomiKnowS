from domiknows.graph import Graph, Concept
from domiknows.graph.logicalConstrain import exactL, atMostL

Graph.clear()
Concept.clear()

with Graph('eight_queens') as graph:
    # Create an 8x8 grid of concepts
    queens = [[Concept(name=f'queen_{i}_{j}') for j in range(8)] for i in range(8)]

    # Row constraints
    for i in range(8):
        exactL(*queens[i])

    # Column constraints
    for j in range(8):
        exactL(*[queens[i][j] for i in range(8)])

    # Main diagonal constraints
    for k in range(-7, 8):
        diagonal = []
        for i in range(8):
            for j in range(8):
                if (i - j) == k:
                    diagonal.append(queens[i][j])
        if len(diagonal) > 1:
            atMostL(*diagonal, 1)

    # Anti-diagonal constraints
    for k in range(0, 15):
        diagonal = []
        for i in range(8):
            for j in range(8):
                if (i + j) == k:
                    diagonal.append(queens[i][j])
        if len(diagonal) > 1:
            atMostL(*diagonal, 1)