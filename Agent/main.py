from __future__ import annotations
import traceback, argparse, os
import sys; sys.path.append("../")
from Agent.llm import LLM
from Agent.graph_prompt import get_graph_prompt, load_all_graphs, get_graph_reviewer_prompt
from typing import Any, Callable, Dict, List, Optional, TypedDict
from langgraph.graph import StateGraph, END, START
from Agent.rag import upsert_examples, select_graph_examples
from domiknows.graph import *
from Agent.utils import extract_python_code
from langgraph.checkpoint.memory import InMemorySaver

class BuildState(TypedDict):
    Task_ID: str
    Task_definition: str

    graph_rag_examples: List[str]
    graph_max_attempts: int
    graph_attempt: int
    graph_code_draft: List[str]
    graph_visual_tools = Dict[Any, Any]

    graph_review_notes: List[str]
    graph_reviewer_agent_approved: bool
    graph_exe_notes: List[str]
    graph_exe_agent_approved: bool
    human_approved: bool
    human_notes: str

def build_graph(
    llm: Optional[Callable[[Any], str]] = None,
    graph_DB=None,
    rag_k: int = 0,
):
    def graph_swe_agent(state: BuildState) -> BuildState:
        attempt = int(state.get("graph_attempt", 0))
        instructions, examples = get_graph_prompt()
        msgs = [{"role": "system", "content": instructions}]
        all_examples = list(examples or [])
        task = state.get("Task_definition", "")
        rag_selected = list(state.get("graph_rag_examples", []))
        if rag_selected:
            all_examples.extend(rag_selected)
        i = 0
        while i < len(all_examples):
            user_msg = all_examples[i]
            msgs.append({"role": "user", "content": user_msg})
            if i + 1 < len(all_examples):
                assistant_msg = all_examples[i + 1]
                msgs.append({"role": "assistant", "content": assistant_msg})
            i += 2
        msgs.append({"role": "user", "content": task})

        for draft, review, syntax_error in zip(
            state.get("graph_code_draft", []),
            state.get("graph_review_notes", []),
            state.get("graph_exe_notes", []),
        ):
            msgs.append({"role": "assistant", "content": draft})
            msgs.append(
                {
                    "role": "user",
                    "content": ("Review of the graph: " + review if review else "")
                    + ("Syntax Error that i got: " + syntax_error if syntax_error else ""),
                }
            )

        code = llm(msgs)
        drafts = list(state.get("graph_code_draft", []))
        drafts.append(code)
        return {
            "graph_rag_examples":rag_selected,
            "graph_attempt": attempt + 1,
            "graph_code_draft": drafts,
            "graph_reviewer_agent_approved": False,
            "graph_exe_agent_approved": False,
        }

    def graph_exe_agent(state: BuildState) -> BuildState:
        try:
            lcls = locals()
            Graph.clear()
            Concept.clear()
            Relation.clear()
            code_list = state.get("graph_code_draft", [])
            latest = code_list[-1] if code_list else ""
            if latest:
                exec(extract_python_code(latest), globals(), lcls)
            graph = lcls["graph"]
            graph.visualize(f"graph_images/{state['Task_ID']}_{len(code_list)}")
            error_msg = ""
        except Exception as e:
            error_msg = traceback.format_exc()
        exe_notes = list(state.get("graph_exe_notes", []))
        if error_msg:
            exe_notes.append("Syntax Error: " + error_msg)
            return {
                "graph_exe_notes": exe_notes,
                "graph_exe_agent_approved": False,
            }
        exe_notes.append("")
        return {"graph_exe_notes": exe_notes,"graph_exe_agent_approved": True}

    def graph_reviewer(state: BuildState) -> BuildState:
        instructions, examples = get_graph_reviewer_prompt()
        msgs = [{"role": "system", "content": instructions}]
        all_examples = list(examples or [])
        task = state.get("Task_definition", "") or ""
        drafts = list(state.get("graph_code_draft", []))
        latest_code = drafts[-1] if drafts else ""
        rag_selected = select_graph_examples(graph_DB, task, rag_k, review=True)
        if rag_selected:
            all_examples.extend(rag_selected)
        i = 0
        while i < len(all_examples):
            user_msg = all_examples[i]
            msgs.append({"role": "user", "content": user_msg})
            if i + 1 < len(all_examples):
                assistant_msg = all_examples[i + 1]
                msgs.append({"role": "assistant", "content": assistant_msg})
            i += 2
        msgs.append({"role": "user", "content": task + latest_code})
        try:
            raw = llm.review(msgs)
            graph_review_notes = list(state.get("graph_review_notes", []))
            graph_review_notes.append(raw)
            graph_reviewer_agent_approved = False
            if "approve" in raw:
                graph_reviewer_agent_approved = True
            return {
                "graph_review_notes": graph_review_notes,
                "graph_reviewer_agent_approved": graph_reviewer_agent_approved,
            }
        except Exception as e:
            graph_review_notes = list(state.get("graph_review_notes", []))
            graph_review_notes.append(f"Reviewer error: {e}")
            return {
                "graph_review_notes": graph_review_notes,
                "graph_reviewer_agent_approved": True,
            }

    def join_review_exe(state: BuildState) -> BuildState:
        return {}

    def graph_human_agent(state: BuildState) -> BuildState:
        human_approved = state.get("human_approved", "")
        print(f"🤖 === GRAPH_HUMAN_AGENT CALLED ===")
        print(f"🤖 Current human_approved: {human_approved}")
        print(f"🤖 Type of human_approved: {type(human_approved)}")
        print(f"🤖 Boolean evaluation: {bool(human_approved)}")
        
        if human_approved:
            print(f"✅ Human already approved - returning state unchanged")
            return state
        else:
            print(f"❌ Human not approved - processing rejection/revision")
            graph_review_notes = list(state.get("graph_review_notes", []))
            graph_review_notes.append(state.get("human_notes", ""))
            graph_exe_notes = list(state.get("graph_exe_notes", []))
            graph_exe_notes.append("")
            return {
                "graph_review_notes": graph_review_notes,
                "graph_reviewer_agent_approved": False,
                "graph_exe_notes": graph_exe_notes,
                "graph_exe_agent_approved": False,
            }

    def route_after_review(state: BuildState) -> str:
        both_approved = bool(state.get("graph_reviewer_agent_approved")) and bool(
            state.get("graph_exe_agent_approved")
        )
        attempts_reached = int(state.get("graph_attempt", 0)) >= int(
            state.get("graph_max_attempts", 1)
        )
        return "to_human" if (both_approved or attempts_reached) else "revise"

    def route_after_human(state: BuildState) -> str:
        if state.get("human_approved"):
            return "approved"
        return "reform"

    def graph_rag_selector(state: BuildState) -> BuildState:
        task = state.get("Task_definition", "") or ""
        rag_selected = select_graph_examples(graph_DB, task, rag_k)
        return {
            "graph_rag_examples": rag_selected or [],
        }

    builder = StateGraph(BuildState)
    builder.add_node("graph_swe_agent", graph_swe_agent)
    builder.add_node("graph_reviewer", graph_reviewer)
    builder.add_node("graph_exe_agent", graph_exe_agent)
    builder.add_node("join_review_exe", join_review_exe)
    builder.add_node("graph_human", graph_human_agent)
    builder.add_node("graph_rag_selector", graph_rag_selector)

    builder.add_edge(START, "graph_rag_selector")
    builder.add_edge("graph_rag_selector", "graph_swe_agent")
    builder.add_edge("graph_swe_agent", "graph_reviewer")
    builder.add_edge("graph_swe_agent", "graph_exe_agent")
    builder.add_edge(["graph_reviewer", "graph_exe_agent"], "join_review_exe")
    builder.add_conditional_edges(
        "join_review_exe",
        route_after_review,
        {
            "to_human": "graph_human",
            "revise": "graph_swe_agent",
        },
    )

    builder.add_conditional_edges(
        "graph_human",
        route_after_human,
        {
            "approved": END,
            "reform": "graph_swe_agent",
        },
    )
    checkpointer = InMemorySaver()
    graph = builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["join_review_exe","graph_human"],
        #interrupt_after=["graph_swe_agent"],
    )
    return graph

def pre_process_graph(test_run = False, task_id=0, task_description="", graph_examples=load_all_graphs(), rag_k=3, max_graphs_check=3):
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
        "human_approved": False,
        "human_notes": "",
    }
    
    print(f"🏁 === INITIAL STATE CREATED ===")
    print(f"🏁 human_approved set to: {initial_state['human_approved']}")
    print(f"🏁 Type: {type(initial_state['human_approved'])}")
    print(f"🏁 Full initial_state: {initial_state}")
    
    llm = LLM(test_run=test_run)
    graph_DB = upsert_examples(task_id=task_id, examples=graph_examples or [], forced=True)
    print("RAG DB created.")
    print("Building graph...")
    graph = build_graph(
        llm=llm,
        graph_DB=graph_DB,
        rag_k=int(rag_k or 0),
    )
    return initial_state, graph

def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="SWE + Reviewer + Exec (parallel) + Human LangGraph pipeline")
    parser.add_argument("--task-id", type=int, default=0, help="Task ID")
    parser.add_argument("--task-description",type=str,default="Create an email spam graph",help="Description of the graph to build",)
    parser.add_argument("--graph-examples",nargs="+",type=str,default=load_all_graphs(),help="List of other examples (paths or text) for RAG",)
    parser.add_argument("--rag-k", type=int, default=3, help="Number of relevant examples to retrieve with RAG (0 to disable)")
    parser.add_argument("--max-graphs-check",type=int,default=3,help="Maximum revision attempts before triggering human final approval",)
    parser.add_argument("--test-run", default=False, action="store_true", help="Use gpt-4o-mini instead of gpt5")
    args = parser.parse_args(argv)

    initial_state, graph = pre_process_graph(
        test_run=args.test_run,
        task_id=args.task_id,
        task_description=args.task_description,
        graph_examples=args.graph_examples,
        rag_k=args.rag_k,
        max_graphs_check=args.max_graphs_check
    )


    snap = None
    print("Start")
    config = {"configurable": {"thread_id": "ID: "+str(args.task_id)}}
    snap = graph.invoke(initial_state, config=config, stream_mode="updates")
    while True:
        snap = graph.get_state(config=config)
        print(snap.next)
        if not snap.next:
            break
        graph.update_state(config,{"human_approved":True},as_node="graph_human")
        graph.invoke(None, config=config)
    print("End")

    return 0, snap.values

if __name__ == "__main__":
    main()
