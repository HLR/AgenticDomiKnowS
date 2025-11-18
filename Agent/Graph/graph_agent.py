from .graph_prompt import get_graph_prompt, get_graph_reviewer_prompt
from Agent.utils import extract_python_code, exec_code

def graph_swe_agent(llm, Task_definition, rag_selected, graph_code_draft, graph_review_notes, graph_exe_notes, human_note):
    instructions, all_examples = get_graph_prompt()
    msgs = [{"role": "system", "content": instructions}]
    if rag_selected: all_examples.extend(rag_selected)
    i = 0
    while i < len(all_examples):
        user_msg = all_examples[i]
        msgs.append({"role": "user", "content": user_msg})
        if i + 1 < len(all_examples):
            assistant_msg = all_examples[i + 1]
            msgs.append({"role": "assistant", "content": assistant_msg})
        i += 2
    msgs.append({"role": "user", "content": Task_definition})
    if not human_note:
        for draft, review, syntax_error in zip(graph_code_draft, graph_review_notes, graph_exe_notes):
            msgs.append({"role": "assistant", "content": draft})
            msgs.append(
                {
                    "role": "user",
                    "content": ("Review of the graph: " + review if review else "") + ("Syntax Error that i got: " + syntax_error if syntax_error else ""),
                }
            )
    else:
        msgs.append({"role": "assistant", "content": graph_code_draft[-1]})
        msgs.append({"role": "user", "content": human_note,})

    code = llm(msgs)
    return code

def graph_exe_agent(graph_code,task_id,len_code_list):
    code = extract_python_code(graph_code)
    captured_prints, captured_stderr, captured_error, graph = exec_code(code,return_graph=True)
    if graph:
        graph.visualize(f"graph_images/{task_id}_{len_code_list}.png")
    return captured_error

def graph_reviewer_agent(llm, Task_definition, code, rag_selected):
    instructions, all_examples = get_graph_reviewer_prompt()
    msgs = [{"role": "system", "content": instructions}]
    if rag_selected:
        for i in range(0, len(rag_selected) - 1, 2):
            all_examples.extend([rag_selected[i] + "\n" + rag_selected[i+1], "approve"])
    i = 0
    while i < len(all_examples):
        user_msg = all_examples[i]
        msgs.append({"role": "user", "content": user_msg})
        if i + 1 < len(all_examples):
            assistant_msg = all_examples[i + 1]
            msgs.append({"role": "assistant", "content": assistant_msg})
        i += 2
    msgs.append({"role": "user", "content": Task_definition + code})
    review_text = llm.review(msgs)
    return review_text, "approve" in review_text.lower()