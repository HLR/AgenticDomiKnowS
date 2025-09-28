from __future__ import annotations
import traceback, argparse, os
from Agent.llm import LLM
from Agent.graph_prompt import get_graph_prompt, load_all_graphs, get_graph_reviewer_prompt
from typing import Any, Callable, Dict, List, Optional, TypedDict
from langgraph.graph import StateGraph, END, START
from Agent.rag import upsert_examples, select_graph_examples
from domiknows.graph import *
from Agent.utils import extract_python_code

class BuildState(TypedDict):
    Task_definition: str

    graph_rag_examples: List[str]
    graph_max_attempts: int
    graph_attempt: int
    graph_code_draft: List[str]
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
        rag_selected = select_graph_examples(graph_DB, task, rag_k)
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
            Graph.clear()
            Concept.clear()
            Relation.clear()
            code_list = state.get("graph_code_draft", [])
            latest = code_list[-1] if code_list else ""
            if latest:
                exec(extract_python_code(latest))
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
        return {"graph_exe_notes": exe_notes+[""],"graph_exe_agent_approved": True}

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
        return state

    def graph_human_agent(state: BuildState) -> BuildState:
        human_approved = state.get("human_approved", "")
        if human_approved:
            return state
        else:
            graph_review_notes = state.get("graph_review_notes", "")
            graph_review_notes.append(state.get("human_notes", ""))
            return {
                "graph_review_notes": graph_review_notes,
                "graph_reviewer_agent_approved": False,
                "graph_exe_notes": state.get("graph_exe_notes", []) +[""],
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

    builder = StateGraph(BuildState)
    builder.add_node("graph_swe_agent", graph_swe_agent)
    builder.add_node("graph_reviewer", graph_reviewer)
    builder.add_node("graph_exe_agent", graph_exe_agent)
    builder.add_node("join_review_exe", join_review_exe)
    builder.add_node("graph_human", graph_human_agent)

    builder.add_edge(START, "graph_swe_agent")
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

    graph = builder.compile()
    return graph

def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(
        description="SWE + Reviewer + Exec (parallel) + Human LangGraph pipeline"
    )
    parser.add_argument("--task-id", type=int, default=0, help="Task ID")
    parser.add_argument(
        "--task-description",
        type=str,
        default="Create an email spam graph",
        help="Description of the graph to build",
    )
    parser.add_argument(
        "--graph-examples",
        nargs="+",
        type=str,
        default=load_all_graphs(),
        help="List of other examples (paths or text) for RAG",
    )
    parser.add_argument(
        "--rag-k", type=int, default=3, help="Number of relevant examples to retrieve with RAG (0 to disable)"
    )
    parser.add_argument(
        "--max-graphs-check",
        type=int,
        default=3,
        help="Maximum revision attempts before triggering human final approval",
    )
    parser.add_argument(
        "--test-run", action="store_true", help="Use gpt-4o-mini instead of gpt5"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="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",
        help="OpenAI API key",
    )
    args = parser.parse_args(argv)

    initial_state: BuildState = {
        "Task_definition": args.task_description,
        "graph_rag_examples":[],
        "graph_max_attempts": int(args.max_graphs_check),
        "graph_attempt": 0,
        "graph_code_draft": [],
        "graph_review_notes": [],
        "graph_reviewer_agent_approved": False,
        "graph_exe_notes": [],
        "graph_exe_agent_approved": False,
        "human_approved": True,
        "human_notes": "",
    }

    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    llm = LLM(test_run=args.test_run, api_key=args.api_key)
    graph_DB = upsert_examples(
        task_id=args.task_id, examples=args.graph_examples or [], api_key=args.api_key, forced=True
    )
    print("RAG DB created.")
    print("Building graph...")
    graph = build_graph(
        llm=llm,
        graph_DB=graph_DB,
        rag_k=int(args.rag_k or 0),
    )

    final_state = None
    print("Start")
    for mode, event in graph.stream(initial_state, stream_mode=["updates", "values"]):
        if mode == "values":
            final_state = event
        else:
            for node, delta in event.items():
                print(f"[{node}] -> keys updated: {list(delta.keys())}")
                if node == "graph_swe_agent" and "graph_code_draft" in delta:
                    print("\n--- Drafted code (truncated) ---")
                    code = delta["graph_code_draft"]
                    last_graph_code = code[-1] if code else None
                    print(last_graph_code)
                    print("--- end ---\n")
                if node == "graph_reviewer":
                    rev_ok = delta.get("graph_reviewer_agent_approved")
                    if rev_ok is not None:
                        print(f"Reviewer approved? {rev_ok}")
                    if delta.get("graph_review_notes"):
                        print("Reviewer notes:")
                        for c in delta.get("graph_review_notes", []) or []:
                            print(f" - {c}")
                if node == "graph_exe_agent":
                    exe_ok = delta.get("graph_exe_agent_approved")
                    if exe_ok is not None:
                        print(f"Executor approved? {exe_ok}")
                    if delta.get("graph_exe_notes"):
                        print("Executor notes:")
                        for c in delta.get("graph_exe_notes", []) or []:
                            print(f" - {c}")
                if node == "graph_human":
                    happr = delta.get("human_approved")
                    hnotes = delta.get("human_notes")
                    print(f"Human approved? {happr}")
                    if hnotes:
                        print("Human notes:")
                        for c in hnotes:
                            print(f" - {c}")
    print("End")

    return 0, final_state

if __name__ == "__main__":
    main()
