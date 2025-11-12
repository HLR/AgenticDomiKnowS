from __future__ import annotations
from Agent.LLM.llm import LLM
from Agent.Sensor.prompts import sensor_instructions, examples
from Agent.utils import extract_python_code, code_prefix, exec_code
from typing import Tuple, List, Dict

def _build_base_messages(rag_selected) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = []
    msgs.append({"role": "system", "content": sensor_instructions})
    for example in examples:
        msgs.append({"role": "user", "content": example[0]})
        msgs.append({"role": "assistant", "content": example[1]})
    for num in range(0,len(rag_selected),2):
        msgs.append({"role": "user", "content": rag_selected[num]})
        msgs.append({"role": "assistant", "content": rag_selected[num+1]})
    return msgs

def sensor_agent(llm: LLM, test_description: str, input_graph: str, rag_selected, prev_code, outputs):
    msgs = list(_build_base_messages(rag_selected)) + [{"role": "user", "content": test_description+"\n"+input_graph}]
    if prev_code and outputs:
        msgs.append({"role": "assistant", "content": prev_code})
        msgs.append({"role": "user", "content": "This is the error that I received after executing the code with its graph. Fix it and only return the fixed code as instructed before: \n\n"+outputs})
    sensor_code = extract_python_code(llm(msgs))
    code = code_prefix + "\n" + input_graph + "\n" + sensor_code
    captured_prints, captured_stderr, captured_error = exec_code(code)
    return code, sensor_code, captured_prints, captured_stderr, captured_error



