import nbformat as nbf
import os
import tempfile
import unittest


def create_notebook(code: str, filename: str) -> None:
    """
    Create a Jupyter notebook from a monolithic code string by splitting it into
    logical sections and writing them into separate cells.

    Expected sections in the resulting notebook:
      1) Intro (markdown): explain that the notebook was generated automatically.
      2) Git (code): clone AgenticDomiKnowS and checkout the "execution" branch.
      3) Imports (code)
      4) Graph declaration (code)
      5) Dataset note (markdown): dataset generator is a placeholder; users can replace it or
         import from domiknows.datasets.
      6) Dataset creation (code)
      7) Sensors / remaining code (code)

    Heuristics used to split the input code:
      - Imports: everything before the first line that contains "with Graph".
      - Graph: from the first "with Graph" line up to (but not including) the first
        dataset-related region.
      - Dataset: from either the first line starting with "def random_" (if present)
        or the first line containing "dataset =" (whichever comes first and exists),
        up to and including the first "dataset =" line.
      - Sensors/Rest: everything after the first "dataset =" line.

    The function is robust to missing sections; any missing part will simply be
    skipped or merged into the "Sensors/Rest" cell as appropriate.
    
    Note:
    - If the provided filename includes directories that do not exist, they
      will be created automatically.
    """

    if not isinstance(code, str) or code.strip() == "":
        raise ValueError("'code' must be a non-empty string.")

    # Ensure the filename ends with .ipynb
    if not isinstance(filename, str) or filename.strip() == "":
        raise ValueError("'filename' must be a non-empty string path.")
    out_path = filename if filename.endswith(".ipynb") else f"{filename}.ipynb"

    lines = code.splitlines()

    def find_index(predicate):
        for i, ln in enumerate(lines):
            if predicate(ln):
                return i
        return -1

    # Find boundaries
    idx_graph = find_index(lambda s: "with Graph" in s)
    idx_random_def = find_index(lambda s: s.strip().startswith("def random_"))
    idx_dataset_assign = find_index(lambda s: "dataset =" in s)

    # Determine dataset start: prefer the earlier of random-def or dataset assignment
    candidates = [i for i in (idx_random_def, idx_dataset_assign) if i != -1]
    idx_dataset_start = min(candidates) if candidates else -1

    # Slice sections with safety
    imports_block = []
    graph_block = []
    dataset_block = []
    sensors_block = []

    if idx_graph == -1:
        # No graph marker; treat all as imports/sensors
        imports_block = lines
    else:
        imports_block = lines[:idx_graph]
        if idx_dataset_start == -1:
            # No dataset markers, so place everything after graph into sensors
            graph_block = lines[idx_graph:]
        else:
            graph_block = lines[idx_graph:idx_dataset_start]
            if idx_dataset_assign == -1:
                # Unexpected (start exists but no assign). Treat rest as dataset.
                dataset_block = lines[idx_dataset_start:]
            else:
                dataset_block = lines[idx_dataset_start:idx_dataset_assign + 1]
                sensors_block = lines[idx_dataset_assign + 1:]

    def join(block):
        # Avoid trailing blank-only cells
        return "\n".join(block).strip("\n") if block else ""

    nb = nbf.v4.new_notebook()
    nb.cells = []

    # 1) Intro markdown
    intro_md = (
        "This notebook was generated automatically to execute DomiKnowS code.\n\n"
        "- You can edit any cell and re-run.\n"
        "- The dataset generation provided here is a placeholder; please replace it with your actual data pipeline."
    )
    nb.cells.append(nbf.v4.new_markdown_cell(intro_md))

    # 2) Git clone + checkout
    git_code = (
        "!git clone https://github.com/HLR/AgenticDomiKnowS.git\n"
        "%cd AgenticDomiKnowS\n"
        "!git checkout execution\n"
        "!pip install -r domiknows-requirements.txt\n"
    )
    nb.cells.append(nbf.v4.new_code_cell(git_code))

    # 3) Imports
    imports_src = join(imports_block)
    if imports_src:
        nb.cells.append(nbf.v4.new_code_cell(imports_src))

    # 4) Graph declaration
    graph_src = join(graph_block)
    if graph_src:
        nb.cells.append(nbf.v4.new_code_cell(graph_src))

    # 5) Dataset note
    dataset_md = (
        "The following dataset code is a simple/random generator intended as a placeholder.\n\n"
        "- Replace it with your actual dataset creation code, or\n"
        "- Import available sample datasets from `domiknows.datasets`."
    )
    nb.cells.append(nbf.v4.new_markdown_cell(dataset_md))

    # 6) Dataset creation
    dataset_src = join(dataset_block)
    if dataset_src:
        nb.cells.append(nbf.v4.new_code_cell(dataset_src))

    # 7) Sensors / remaining code
    sensors_src = join(sensors_block)
    if sensors_src:
        nb.cells.append(nbf.v4.new_code_cell(sensors_src))

    # Ensure parent directories exist (if any were provided)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Write notebook to file
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

if __name__ == "__main__":
    code = """
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
from Examples.emailspam.data import email_data

with Graph('email_spam_consistency') as graph:
    email = Concept(name='email')

    m1 = email(name='model1_pred', ConceptClass=EnumConcept, values=['spam', 'not_spam'])
    m2 = email(name='model2_pred', ConceptClass=EnumConcept, values=['spam', 'not_spam'])

    xorL(m1.spam, m2.not_spam)

def random_emailspam_instance():
    email_id = [0]
    m1_labels = [random.randint(0,1)]
    m2_labels = [random.randint(0, 1)]

    data = {
        "email_id": email_id,
        "email_header_text": email_id,
        "email_body_text": email_id,
        "m1_id":  [i for i in range(len(m1))],
        "m2_id":  [i for i in range(len(m2))],
        "m1": m1_labels,
        "m2": m2_labels,
    }
    return data

dataset = [random_emailspam_instance() for _ in range(1)]

email['email_id'] = ReaderSensor(keyword='email_id')
email['email_header_text'] = ReaderSensor(keyword='email_header_text')
email['email_body_text'] = ReaderSensor(keyword='email_body_text')
m1['m1_id'] = ReaderSensor(keyword='m1_id')
m2['m2_id'] = ReaderSensor(keyword='m2_id')

m1[m1] = LabelReaderSensor(keyword='m1')
m2[m2] = LabelReaderSensor(keyword='m2')

email[m1] = LLMLearner(email["email_header_text"],email["email_body_text"], prompt="Classify the emails as spam or not_spam.",classes=['spam', 'not_spam'])
email[m2] = LLMLearner(email["email_header_text"],email["email_body_text"], prompt="Classify the emails as spam or not_spam.",classes=['spam', 'not_spam'])

program = SolverPOIProgram(graph,poi=[email],inferTypes=['local/argmax'],loss=MacroAverageTracker(NBCrossEntropyLoss()),metric=PRF1Tracker())
for datanode in program.populate(dataset=dataset):
    datanode.inferILPResults(email)

    print(f"m1 :",datanode.getResult(m1,"local","argmax"))
    print(f"m1 ILP:", datanode.getResult(m1, "ILP"))

    print(f"m2 :",datanode.getResult(m2,"local","argmax"))
    print(f"m2 ILP:", datanode.getResult(m2, "ILP"))
"""
    create_notebook(code, "collab.ipynb")