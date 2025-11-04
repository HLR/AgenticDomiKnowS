from __future__ import annotations
import traceback
import io
from contextlib import redirect_stdout, redirect_stderr
import re
import textwrap
import pandas as pd
import os
from typing import List, Tuple
from langchain_chroma import Chroma
import hashlib
from dotenv import load_dotenv
load_dotenv()

_PY_FENCE = re.compile(r"```(?:python|py|python3)(?:[^\n]*)\r?\n(.*?)(?:\r?\n)?```", flags=re.IGNORECASE | re.DOTALL,)

_ANY_FENCE = re.compile(r"```\s*\r?\n(.*?)(?:\r?\n)?```", flags=re.DOTALL,)

def extract_python_code(text: str) -> str:
    m = _PY_FENCE.search(text) or _ANY_FENCE.search(text)
    code = m.group(1) if m else text
    return textwrap.dedent(code).strip("\n")

def exec_code(code):
    _stdout_buf = io.StringIO()
    _stderr_buf = io.StringIO()
    captured_prints = ""
    captured_stderr = ""
    captured_error = ""
    try:
        ns = {"__package__": None, "__builtins__": __builtins__}
        with redirect_stdout(_stdout_buf), redirect_stderr(_stderr_buf):
            exec(compile(code, "<code_exec>", "exec"), ns, ns)
        captured_prints = _stdout_buf.getvalue()
        captured_stderr = _stderr_buf.getvalue()
    except Exception:
        captured_error = traceback.format_exc()
        captured_stderr = _stderr_buf.getvalue()
    finally:
        _stdout_buf.close()
        _stderr_buf.close()

    return captured_prints, captured_stderr, captured_error

code_prefix = """
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

Graph.clear()
Concept.clear()
Relation.clear()

"""

def load_all_examples_info(address="",exclude_graph=None):
    if address:
        data = pd.read_csv(address+'lang_to_code_test.csv')
    else:
        data = pd.read_csv(os.path.dirname(os.path.abspath(__file__))+'/datasets/lang_to_code_test.csv')
    example_graphs = []
    for i in data.index:
        row = data.loc[i]
        desc = (row.get("description") or "").strip()
        constr = (row.get("description_constraint") or "").strip()
        gold_graph = (row.get("graph") or "") + "\n" + (row.get("constraints") or "").strip()
        if gold_graph == exclude_graph: continue
        task_text = (desc + ("\n\n" + constr if constr else "")) if desc or constr else ""
        sensor_code = (row.get("dummysensor") or "")
        example_graphs.append({"task_text":task_text, "gold_graph":gold_graph,"sensor_code":sensor_code})
    return example_graphs

def upsert_examples(llm, examples: List[str]):

    DB = Chroma(embedding_function=llm.embedder)
    texts, metas, ids = [], [], []
    for example in examples:
        desc, gold_graph, sensor_code = example.get("task_text"), example.get("gold_graph"), example.get("sensor_code")
        _id = hashlib.sha1(desc.encode("utf-8")).hexdigest()
        texts.append(desc)
        metas.append({"desc": desc, "gold_graph": gold_graph, "sensor_code" : sensor_code})
        ids.append(_id)
    DB.add_texts(texts=texts, metadatas=metas, ids=ids)
    return DB

def select_graph_examples(DB: Chroma, task_desc: str, k: int) -> List[str]:
    if not k or k<=0:
        return [], []
    results = DB.similarity_search(task_desc or "", k=k)
    graph_out,sensor_out = [], []
    for d in results:
        md = d.metadata or {}
        graph_out.extend([md.get("desc", d.page_content)] + ([md["gold_graph"]] if md.get("gold_graph") else []))
        sensor_out.extend([md.get("desc", d.page_content) + "\n" + md["gold_graph"], md["sensor_code"]])
    return graph_out, sensor_out
