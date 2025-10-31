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

with Graph('sudoko') as graph:

    sudoku = Concept('sodoku')

    empty_entry = Concept(name='empty_entry')
    (empty_rel,) = sudoku.contains(empty_entry)

    same_row = Concept(name='same_row')
    (same_row_arg1, same_row_arg2) = same_row.has_a(row1=empty_entry, row2=empty_entry)

    same_col = Concept(name='same_col')
    (same_col_arg1, same_col_arg2) = same_col.has_a(col1=empty_entry, col2=empty_entry)

    same_table = Concept(name='same_table')
    (same_table_arg1, same_table_arg2) = same_table.has_a(entry1=empty_entry, entry2=empty_entry)

    empty_entry_label = empty_entry(name='empty_entry_label', ConceptClass=EnumConcept, values=[f'n{i}' for i in range(9)])

    for val in [f'{i}' for i in range(9)]:
        ifL(getattr(empty_entry_label, f'n{val}')('x'), notL(existsL(andL(same_row('z', path=('x', same_row_arg1.reversed)), getattr(empty_entry_label, f'n{val}')('y', path=('z', same_row_arg2))))))
        ifL(getattr(empty_entry_label, f'n{val}')('x'), notL(existsL(andL(same_col('z', path=('x', same_col_arg1.reversed)),getattr(empty_entry_label, f'n{val}')('y', path=('z', same_col_arg2))))))
        ifL(getattr(empty_entry_label, f'n{val}')('x'), notL(existsL(andL(same_table('z', path=('x', same_table_arg1.reversed)),getattr(empty_entry_label, f'n{val}')('y', path=('z', same_table_arg2))))))


def random_sudoku_instance():
    sudoku_id = [0]
    empty_entry_id = [i for i in range(9*9)]

    fixed = [False] * (9*9)
    labels = [None] * (9*9)
    k_fixed = 9*2-3
    fixed_idxs = random.sample(empty_entry_id, k_fixed)
    for idx in fixed_idxs:
        fixed[idx] = True
        labels[idx] = random.randint(0, 8)

    for i in range(9*9):
        if labels[i] is None:
            labels[i] = random.randint(0, 8)

    same_row_pairs = []
    for r in range(9):
        cells = [r * 9 + c for c in range(9)]
        for i in cells:
            for j in cells:
                if i != j:
                    same_row_pairs.append((i, j))

    same_col_pairs = []
    for c in range(9):
        cells = [r * 9 + c for r in range(9)]
        for i in cells:
            for j in cells:
                if i != j:
                    same_col_pairs.append((i, j))

    same_table_pairs = []
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            cells = []
            for dr in range(3):
                for dc in range(3):
                    cells.append((br + dr) * 9 + (bc + dc))
            for i in cells:
                for j in cells:
                    if i != j:
                        same_table_pairs.append((i, j))

    data = {
        'sudoku_id': sudoku_id,
        'empty_entry_id': empty_entry_id,
        'empty_entry_label': labels,
        'same_row': [same_row_pairs],
        'same_col': [same_col_pairs],
        'same_table': [same_table_pairs],
    }

    empty_rel = list()
    for table in data["sudoku_id"]:
        for cell in data["empty_entry_id"]:
            empty_rel.append((table, cell))
    data["empty_rel"] = [empty_rel]
    return data

dataset = [random_sudoku_instance() for _ in range(1)]

sudoku['sudoku_id'] = ReaderSensor(keyword='sudoku_id')
empty_entry['empty_entry_id'] = ReaderSensor(keyword='empty_entry_id')
empty_entry[empty_entry_label] = LabelReaderSensor(keyword='empty_entry_label')

empty_entry[empty_rel] = EdgeReaderSensor(sudoku['sudoku_id'], empty_entry['empty_entry_id'],keyword='empty_rel', relation=empty_rel)

same_row[same_row_arg1.reversed, same_row_arg2.reversed] = ManyToManyReaderSensor(empty_entry['empty_entry_id'], empty_entry['empty_entry_id'], keyword='same_row')
same_col[same_col_arg1.reversed, same_col_arg2.reversed] = ManyToManyReaderSensor(empty_entry['empty_entry_id'], empty_entry['empty_entry_id'], keyword='same_col')
same_table[same_table_arg1.reversed, same_table_arg2.reversed] = ManyToManyReaderSensor(empty_entry['empty_entry_id'], empty_entry['empty_entry_id'], keyword='same_table')

empty_entry[empty_entry_label] = DummyLearner('empty_entry_id', output_size=9)

program = SolverPOIProgram(
    graph,
    poi=[sudoku, empty_entry, empty_entry_label, same_row, same_col, same_table],
    inferTypes=['local/argmax'],
    loss=MacroAverageTracker(NBCrossEntropyLoss()),
    metric=PRF1Tracker()
)

for datanode in program.populate(dataset=dataset):
    datanode.inferILPResults()

    cells = datanode.getChildDataNodes()
    n = 9
    def _print_grid(title, grid):
        print(title)
        for r in range(n):
            row = []
            for c in range(n):
                val = grid[r][c]
                row.append('.' if val is None or val == '' else str(val))
            print(' '.join(f"{x:>2}" for x in row))
        print()

    local_grid = [[None for _ in range(n)] for _ in range(n)]
    ilp_grid = [[None for _ in range(n)] for _ in range(n)]

    for idx, cell in enumerate(cells):
        r, c = divmod(idx, n)
        local_grid[r][c] = cell.getResult(empty_entry_label, 'local', 'argmax')
        ilp_grid[r][c] = cell.getResult(empty_entry_label, 'ILP')

    _print_grid("Table with dummy predictions:", local_grid)
    _print_grid("Table after ILP:", ilp_grid)