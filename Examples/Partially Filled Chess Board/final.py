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

with Graph('eight_queens') as graph:
    board = Concept(name='board')
    cell = Concept(name='cell')
    (board_contains_cell,) = board.contains(cell)

    cell_state = cell(name='cell_state', ConceptClass=EnumConcept, values=['Q', 'E'])

    same_row = Concept(name='same_row')
    (sr_arg1, sr_arg2) = same_row.has_a(rarg1=cell, rarg2=cell)

    same_col = Concept(name='same_col')
    (sc_arg1, sc_arg2) = same_col.has_a(carg1=cell, carg2=cell)

    same_diag = Concept(name='same_diag')
    (sd_arg1, sd_arg2) = same_diag.has_a(darg1=cell, darg2=cell)

    ifL(same_row('r'), notL(existsL(andL(cell_state.Q(path=('r', sr_arg1)), cell_state.Q(path=('r', sr_arg2))))))
    ifL(same_col('c'), notL(existsL(andL(cell_state.Q(path=('c', sc_arg1)), cell_state.Q(path=('c', sc_arg2))))))
    ifL(same_diag('d'), notL(existsL(andL(cell_state.Q(path=('d', sd_arg1)), cell_state.Q(path=('d', sd_arg2))))))

    ifL(board('b'), exactL(cell_state.Q('x', path=('b', board_contains_cell)), 8))

def random_chess_graph_instance():
    size = 8
    board_ids = [0]
    cell_ids = list(range(size * size))
    queen_labels = [random.randint(0, 1) for _ in cell_ids]

    def rc(cell_id):
        return divmod(cell_id, size)

    rows = {r: [] for r in range(size)}
    cols = {c: [] for c in range(size)}
    diag_main = {}
    diag_anti = {}

    for cid in cell_ids:
        r, c = rc(cid)
        rows[r].append(cid)
        cols[c].append(cid)
        diag_main.setdefault(r - c, []).append(cid)
        diag_anti.setdefault(r + c, []).append(cid)

    def all_pairs(lst):
        pairs = []
        n = len(lst)
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((lst[i], lst[j]))
        return pairs

    same_row_pairs = []
    for r in rows:
        same_row_pairs.extend(all_pairs(rows[r]))

    same_col_pairs = []
    for c in cols:
        same_col_pairs.extend(all_pairs(cols[c]))

    same_diag_pairs = []
    for key in diag_main:
        same_diag_pairs.extend(all_pairs(diag_main[key]))
    for key in diag_anti:
        same_diag_pairs.extend(all_pairs(diag_anti[key]))

    board_grid = []
    for r in range(size):
        row_chars = []
        for c in range(size):
            cid = r * size + c
            row_chars.append('Q' if queen_labels[cid] == 1 else '.')
        board_grid.append(''.join(row_chars))
    chess_board_feature = '\n'.join(board_grid)

    data = {
        "chess_board_id": board_ids,
        "chess_board": [chess_board_feature],
        "queen_placed_id": cell_ids,
        "queen_placed": queen_labels,
        "same_row": [same_row_pairs],
        "same_column": [same_col_pairs],
        "same_diagonal": [same_diag_pairs],
    }

    board_contains_cell = []
    for b in data["chess_board_id"]:
        for cell_id in data["queen_placed_id"]:
            board_contains_cell.append((b, cell_id))
    data["board_contains_cell"] = [board_contains_cell]

    return data

dataset = [random_chess_graph_instance() for _ in range(1)]

board['chess_board_id'] = ReaderSensor(keyword='chess_board_id')
board['chess_board'] = ReaderSensor(keyword='chess_board')

cell['queen_placed_id'] = ReaderSensor(keyword='queen_placed_id')

cell[board_contains_cell] = EdgeReaderSensor(board['chess_board_id'], cell['queen_placed_id'], keyword='board_contains_cell', relation=board_contains_cell)

same_row[sr_arg1.reversed, sr_arg2.reversed] = ManyToManyReaderSensor(cell['queen_placed_id'], cell['queen_placed_id'], keyword='same_row')
same_col[sc_arg1.reversed, sc_arg2.reversed] = ManyToManyReaderSensor(cell['queen_placed_id'], cell['queen_placed_id'], keyword='same_column')
same_diag[sd_arg1.reversed, sd_arg2.reversed] = ManyToManyReaderSensor(cell['queen_placed_id'], cell['queen_placed_id'], keyword='same_diagonal')

cell_state[cell_state] = LabelReaderSensor(keyword='queen_placed')

cell[cell_state] = LLMLearner(board["chess_board"], cell["queen_placed_id"], prompt="Given the 8x8 chess board with fixed queens (Q) and empty cells (.), and a zero-based cell index (row-major order), predict if the cell contains a Queen (Q) or is Empty (E). ", classes=['Q', 'E'])

program = SolverPOIProgram(graph, poi=[board, cell, same_row, same_col, same_diag], inferTypes=['local/argmax'], loss=MacroAverageTracker(NBCrossEntropyLoss()), metric=PRF1Tracker())
for datanode in program.populate(dataset=dataset):
    datanode.inferILPResults()
    local_labels = []
    ilp_labels = []
    for idx, cnode in enumerate(datanode.getChildDataNodes()):
        local_idx = cnode.getResult(cell_state, "local", "argmax")
        ilp_idx = cnode.getResult(cell_state, "ILP")

        local_labels.append('Q' if local_idx == 0 else 'E')
        ilp_labels.append('Q' if ilp_idx == 0 else 'E')

    size = 8
    print("Local argmax (Q/E) grid:")
    for r in range(size):
        row = local_labels[r*size:(r+1)*size]
        print(' '.join(row))

    print("ILP (Q/E) grid:")
    for r in range(size):
        row = ilp_labels[r*size:(r+1)*size]
        print(' '.join(row))