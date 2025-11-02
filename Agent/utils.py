from __future__ import annotations
import traceback
import io
from contextlib import redirect_stdout, redirect_stderr
import re
import textwrap

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