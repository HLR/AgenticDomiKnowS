from __future__ import annotations
import argparse
import sys; sys.path.append("../")
from Agent.LLM.llm import LLM
from Agent.Graph.graph_prompt import get_graph_prompt, get_graph_reviewer_prompt
from Agent.Graph.graph_agent import graph_swe_agent, graph_exe_agent, graph_reviewer_agent
from typing import Any, Callable, Dict, List, Optional, TypedDict
from langgraph.graph import StateGraph, END, START
from Agent.utils import extract_python_code, code_prefix, load_all_examples_info, upsert_examples, select_graph_examples
from langgraph.checkpoint.memory import InMemorySaver
from Agent.Sensor.sensor_agent import sensor_agent
from langgraph.types import interrupt, Command

class BuildState(TypedDict):
    Task_ID: str
    Task_definition: str

    graph_rag_examples: List[str]
    graph_max_attempts: int
    graph_attempt: int
    graph_code_draft: List[str]

    graph_review_notes: List[str]
    graph_reviewer_agent_approved: bool
    graph_exe_notes: List[str]
    graph_exe_agent_approved: bool
    graph_human_approved: bool
    graph_human_notes: str

    sensor_attempt: int
    sensor_codes: List[str]
    sensor_human_changed: bool
    entire_sensor_codes: List[str]
    sensor_code_outputs: List[str]
    sensor_rag_examples: List[str]

    property_human_text: str
    property_rag_examples: List[str]
    final_code_text: str

def build_graph(
    llm: Optional[Callable[[Any], str]] = None,
    graph_DB=None,
    graph_rag_k: int = 0,
):
    def graph_swe_agent_node(state: BuildState) -> BuildState:
        code = graph_swe_agent(llm, state.get("Task_definition", ""), list(state.get("graph_rag_examples", [])), list(state.get("graph_code_draft", [])), list(state.get("graph_review_notes", [])) , list(state.get("graph_exe_notes", [])), state.get("graph_human_notes", ""))
        return {"graph_attempt": int(state.get("graph_attempt", 0)) + 1, "graph_code_draft": list(state.get("graph_code_draft", [])) + [code], "graph_reviewer_agent_approved": False, "graph_exe_agent_approved": False, "graph_human_notes": ""}

    def graph_exe_agent_node(state: BuildState) -> BuildState:
        captureed_error = graph_exe_agent(code_prefix+extract_python_code(state.get("graph_code_draft", [""])[-1]))
        return {"graph_exe_notes": state.get("graph_exe_notes", []) + [captureed_error] ,"graph_exe_agent_approved": captureed_error == ""}

    def graph_reviewer_agent_node(state: BuildState) -> BuildState:
        review_text, approved = graph_reviewer_agent(llm, state.get("Task_definition", ""), list(state.get("graph_code_draft", []))[-1], list(state.get("graph_rag_examples", [])))
        return {"graph_review_notes": list(state.get("graph_review_notes", [])) + [review_text], "graph_reviewer_agent_approved": approved,}

    def join_review_exe(state: BuildState) -> BuildState:
        return {}

    def graph_human_agent(state: BuildState) -> BuildState:
        human_response = interrupt("Did human approve?")
        graph_human_approved, graph_human_notes = human_response.get("graph_human_approved", ""), human_response.get("graph_human_notes", "")
        if graph_human_approved: return {"graph_human_approved" : graph_human_approved, "graph_human_notes": graph_human_notes}
        return {"graph_human_approved" : graph_human_approved, "graph_human_notes": graph_human_notes, "graph_review_notes": [], "graph_reviewer_agent_approved": False, "graph_exe_notes": [], "graph_exe_agent_approved": False, "graph_attempt": 0}

    def route_after_review(state: BuildState) -> str:
        both_approved = bool(state.get("graph_reviewer_agent_approved")) and bool(state.get("graph_exe_agent_approved"))
        attempts_reached = int(state.get("graph_attempt", 0)) >= int(state.get("graph_max_attempts", 1))
        return "to_human" if (both_approved or attempts_reached) else "revise"

    def route_after_human(state: BuildState) -> str:
        if bool(state.get("graph_human_approved")): return "approved"
        return "reform"

    def graph_rag_selector(state: BuildState) -> BuildState:
        graph_out, sensor_out, property_out = select_graph_examples(graph_DB, state.get("Task_definition", "") or "", graph_rag_k)
        return {"graph_rag_examples": graph_out or [], "sensor_rag_examples": sensor_out or [], "property_rag_examples": property_out or []}

    def sensor_agent_node(state: BuildState) -> BuildState:
        print("Sensor agent node")
        prev_code, outputs = "", ""
        sensor_codes_list, entire_sensor_codes_list, sensor_code_outputs = [], [], []
        attempts = int(state.get("sensor_attempt", 0))
        while attempts:
            code, sensor_code, captured_prints, captured_stderr, captured_error = sensor_agent(llm, state.get("Task_definition", ""), state.get("graph_code_draft")[-1], state.get("sensor_rag_examples"), prev_code, outputs)
            attempts -= 1
            outputs = captured_prints+captured_stderr+captured_error
            prev_code = sensor_code
            sensor_codes_list.append(sensor_code)
            entire_sensor_codes_list.append(code)
            sensor_code_outputs.append(outputs)
            if not "Traceback" in outputs: break
        return {"sensor_codes": sensor_codes_list, "entire_sensor_codes":entire_sensor_codes_list , "sensor_code_outputs": sensor_code_outputs}

    def property_agent_node(state: BuildState) -> BuildState:
        human_response = interrupt("Did human set up the properties")
        property_human_text = human_response.get("property_human_text", "")
        # TODO
        return {"final_code_text" : "Please refer to this link TODO"}

    builder = StateGraph(BuildState)
    builder.add_node("graph_swe_agent_node", graph_swe_agent_node)
    builder.add_node("graph_reviewer_agent_node", graph_reviewer_agent_node)
    builder.add_node("graph_exe_agent_node", graph_exe_agent_node)
    builder.add_node("join_review_exe", join_review_exe)
    builder.add_node("graph_human_agent", graph_human_agent)
    builder.add_node("graph_rag_selector", graph_rag_selector)

    builder.add_node("sensor_agent_node", sensor_agent_node)

    builder.add_node("property_agent_node", property_agent_node)

    builder.add_edge(START, "graph_rag_selector")
    builder.add_edge("graph_rag_selector", "graph_swe_agent_node")
    builder.add_edge("graph_swe_agent_node", "graph_reviewer_agent_node")
    builder.add_edge("graph_swe_agent_node", "graph_exe_agent_node")
    builder.add_edge(["graph_reviewer_agent_node", "graph_exe_agent_node"], "join_review_exe")

    builder.add_conditional_edges("join_review_exe", route_after_review, {"to_human": "graph_human_agent", "revise": "graph_swe_agent_node",},)
    builder.add_conditional_edges("graph_human_agent", route_after_human, {"approved": "sensor_agent_node", "reform": "graph_swe_agent_node",},)

    builder.add_edge("sensor_agent_node", "property_agent_node")
    builder.add_edge("property_agent_node", END)

    checkpointer = InMemorySaver()
    graph = builder.compile(checkpointer=checkpointer, interrupt_before=["join_review_exe","graph_human_agent","sensor_agent_node","property_agent_node"])
    return graph

def pre_process_graph(reasoning_effort = "medium", task_id=0, task_description="", graph_examples=load_all_examples_info(), graph_rag_k=3, max_graphs_check=3):
    initial_state = {
        "Task_ID": str(task_id),
        "Task_definition": task_description,
        "graph_rag_examples": [],
        "graph_max_attempts": int(max_graphs_check),
        "graph_attempt": 0,
        "graph_code_draft": [],
        "graph_review_notes": [],
        "graph_reviewer_agent_approved": False,
        "graph_exe_notes": [],
        "graph_exe_agent_approved": False,
        "graph_human_approved": False,
        "graph_human_notes": "",
        "sensor_attempt": 3,
        "sensor_codes": [],
        "sensor_human_changed": False,
        "entire_sensor_codes": [],
        "sensor_code_outputs": [],
        "sensor_rag_examples": [],
        "property_human_text": "",
        "final_code_text": ""
    }

    
    print("=== INITIAL STATE CREATED ===")
    llm = LLM(reasoning_effort=reasoning_effort)
    graph_DB = upsert_examples(llm, examples=graph_examples or [])
    print("RAG DB created.")
    print("Building graph...")
    graph = build_graph(llm=llm, graph_DB=graph_DB, graph_rag_k=int(graph_rag_k or 0))
    return initial_state, graph

def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Coding LangGraph pipeline")
    parser.add_argument("--task-id", type=int, default=0, help="Task ID")
    parser.add_argument("--task-description",type=str,default="Create a graph related to agriculture and its intricacies",help="Description of the graph to build",)
    parser.add_argument("--graph-examples",nargs="+",type=str,default=load_all_examples_info(),help="List of other examples (paths or text) for RAG",)
    parser.add_argument("--graph-rag-k", type=int, default=5, help="Number of relevant examples to retrieve with RAG (0 to disable) for the graph")
    parser.add_argument("--max-graphs-check",type=int,default=3 ,help="Maximum revision attempts before triggering human final approval",)
    parser.add_argument("--reasoning-effort", default="minimal", choices=["minimal","low","medium","high"], help="Set the LLM reasoning effort level")
    args = parser.parse_args(argv)

    initial_state, graph = pre_process_graph(
        reasoning_effort=args.reasoning_effort,
        task_id=args.task_id,
        task_description=args.task_description,
        graph_examples=args.graph_examples,
        graph_rag_k=args.graph_rag_k,
        max_graphs_check=args.max_graphs_check
    )

    print("Start")
    config = {"configurable": {"thread_id": "ID: "+str(args.task_id)}}
    graph.invoke(initial_state, config=config, stream_mode="updates")
    while True:
        snap = graph.get_state(config=config)
        print(snap.next)
        if not snap.next:
            break
        graph.invoke(Command(resume={"graph_human_approved":True}), config=config)
    print("End")
    return 0, snap.values

if __name__ == "__main__":
    main()
