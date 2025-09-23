from __future__ import annotations

from Agent.llm import LLM
from Agent.graph_prompt import get_graph_prompt
import argparse
from typing import Any, Callable, Dict, List, Optional, TypedDict
from langgraph.graph import StateGraph, END, START

class BuildState(TypedDict):

    Task_definition: str

    graph_max_attempts: int
    graph_attempt: int
    graph_code_draft: List[str]
    graph_review_notes: List[str]
    graph_reviewer_agent_approved: bool


def build_graph(llm: Optional[Callable[[str], str]] = None):

    def graph_swe_agent(state: BuildState) -> BuildState:
        attempt = int(state.get("graph_attempt", 0))
        prompt = get_graph_prompt(state["Task_definition"])
        code = llm(prompt)
        drafts = list(state.get("graph_code_draft", []))
        drafts.append(code)
        return {**state, "graph_attempt": attempt + 1, "graph_code_draft": drafts}

    def graph_reviewer(state: BuildState) -> BuildState:
        return {**state, "graph_review_notes": [], "graph_reviewer_agent_approved": True, }

    def route_after_review(state: BuildState) -> str:
        if state.get("graph_reviewer_agent_approved"):
            return "approve"
        if int(state.get("graph_attempt", 0)) >= int(state.get("graph_max_attempts", 1)):
            return "giveup"
        return "revise"

    builder = StateGraph(BuildState)
    builder.add_node("graph_swe_agent", graph_swe_agent)
    builder.add_node("graph_reviewer", graph_reviewer)

    builder.add_edge(START, "graph_swe_agent")
    builder.add_edge("graph_swe_agent", "graph_reviewer")
    builder.add_conditional_edges(
        "graph_reviewer",
        route_after_review,
        {
            "approve": END,
            "revise": "graph_swe_agent",
            "giveup": END,
        },
    )
    graph = builder.compile()
    return graph


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="SWE + Reviewer LangGraph pipeline")
    parser.add_argument("--task-description", type=str, default="Create an email spam graph", help="Description of the graph to build")
    parser.add_argument("--max-graphs-check", type=int, default=3, help="Maximum revision attempts before giving up on the graphs")
    parser.add_argument("--test-run", action="store_true", help="Use gpt-4o-mini instead of gpt5")
    parser.add_argument("--api-key", type=str, default="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", help="OpenAI API key")
    args = parser.parse_args(argv)

    initial_state: BuildState = {
        "Task_definition": args.task_description,
        "max_graphs_check": int(args.max_graphs_check),
        "attempt": 0,
        "graph_approved": False,
    }
    llm=LLM(test_run=args.test_run,api_key=args.api_key)
    graph = build_graph(llm=llm)

    last_graph_code: Optional[str] = None
    try:
        print("Start")
        for event in graph.stream(initial_state):
            for node, delta in event.items():
                print(f"[{node}] -> keys updated: {list(delta.keys())}")
                if node == "graph_swe_agent" and "graph_code_draft" in delta:
                    print("\n--- Drafted code (truncated) ---")
                    code = delta["graph_code_draft"]
                    last_graph_code = code[-1] if code else None
                    print(last_graph_code)
                    print("--- end ---\n")
                if node == "graph_reviewer":
                    approved = delta.get("graph_reviewer_agent_approved")
                    concerns = delta.get("graph_review_notes")
                    print(f"Approved? {approved}")
                    if concerns:
                        print("Concerns:")
                        for c in concerns:
                            print(f" - {c}")
        print("End")
    except Exception as e:
        print("Execution error:", e)
        return 1, None
    return 0, last_graph_code

if __name__ == "__main__":
    main()
