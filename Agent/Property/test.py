from __future__ import annotations
import argparse, csv, io, os, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Dict, List, Any

from Agent.main import pre_process_graph
from Agent.utils import load_all_examples_info
from langgraph.types import Command


def _run_single(task_id: str, task_name: str, task_text: str, gold_graph: str, property_text: str, reasoning_effort: str) -> Dict[str, str]:
    """Run the full pipeline and return the final_code_text produced by the property agent.

    Flow requirements per issue:
    - Provide Task_definition (handled by pre_process_graph)
    - Interrupt: approve graph via graph_human_approved
    - Interrupt: provide property_human_text read from CSV 'property' field
    """
    buf_out = io.StringIO()
    buf_err = io.StringIO()

    try:
        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            try:
                numeric_task_id: Any = int(task_id)
            except Exception:
                numeric_task_id = task_id

            initial_state, graph = pre_process_graph(
                reasoning_effort=reasoning_effort,
                task_id=numeric_task_id,
                task_description=task_text,
                graph_examples=list(load_all_examples_info("", gold_graph) or []),
                graph_rag_k=5,
                max_graphs_check=3,
            )

            config = {"configurable": {"thread_id": "ID: " + str(task_id)}}
            graph.invoke(initial_state, config=config, stream_mode="updates")

            # Drive the interrupts in order: join_review_exe -> graph_human_agent -> sensor_agent_node -> property_agent_node
            while True:
                snap = graph.get_state(config=config)
                if not snap.next:
                    break
                nxt = (snap.next or [None])[0]
                if nxt == "graph_human_agent":
                    graph.invoke(Command(resume={"graph_human_approved": True}), config=config)
                elif nxt == "property_agent_node":
                    graph.invoke(Command(resume={"property_human_text": property_text}), config=config)
                else:
                    # For join_review_exe and sensor_agent_node we just continue
                    graph.invoke(Command(resume={}), config=config)

            state: Dict[str, Any] = dict(snap.values or {})
            final_code_text = state.get("final_code_text", "") or ""
            final_code_output = state.get("final_code_output", "") or ""

            return {
                "ID": str(task_id),
                "name": str(task_name),
                "final_code_text": final_code_text,
                "final_code_output": final_code_output,
            }

    except Exception:
        err_text = traceback.format_exc()
        return {
            "ID": str(task_id),
            "name": str(task_name),
            "final_code_text": err_text,
            "final_code_output": err_text,
        }
    finally:
        buf_out.close()
        buf_err.close()


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Agent Property over CSV tasks in parallel")
    parser.add_argument("--csv-path", type=str, default="../datasets/lang_to_code_test.csv")
    parser.add_argument("--output-path", type=str, default="../datasets/")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--reasoning-effort", default=["low","medium","medium"], choices=["minimal", "low", "medium", "high"], help="Set the LLM reasoning effort level")
    parser.add_argument("--samples", type=int, default=5, help="Number of repeated runs to execute and save (files will be suffixed 0..N-1 if N>1)")

    args = parser.parse_args(argv)

    csv_path = Path(args.csv_path)

    rows: List[Dict[str, str]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    tasks = []
    for row in rows:
        task_id = str(row.get("ID", "")).strip()
        task_name = str(row.get("name", "")).strip()
        desc = (row.get("description") or "").strip()
        constr = (row.get("description_constraint") or "").strip()
        gold_graph = (row.get("graph") or "") + "\n" + (row.get("constraints") or "").strip()
        task_text = (desc + ("\n\n" + constr if constr else "")) if desc or constr else ""
        property_text = (row.get("property") or "").strip()
        tasks.append((task_id, task_name, task_text, gold_graph, property_text))

    # Normalize samples value
    total_samples = args.samples if isinstance(args.samples, int) else 1
    if total_samples <= 0:
        print(f"[warn] --samples was {args.samples}, defaulting to 1")
        total_samples = 1

    for sample_idx in range(total_samples):
        # Decide output filename per sample
        if total_samples > 1:
            out_path = Path(args.output_path + f"/{args.reasoning_effort}_" + f"property_test_results_{sample_idx}.csv")
        else:
            out_path = Path(args.output_path + f"/{args.reasoning_effort}_" + "property_test_results.csv")

        print(
            (
                f"Starting sample {sample_idx + 1}/{total_samples}: "
                f"{len(tasks)} tasks with {args.workers} workers. Output -> {out_path}"
            )
        )

        results: List[Dict[str, str]] = []
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            fut_to_idx = {
                ex.submit(
                    _run_single,
                    task_id,
                    task_name,
                    task_text,
                    gold_graph,
                    property_text,
                    args.reasoning_effort,
                ): i
                for i, (task_id, task_name, task_text, gold_graph, property_text) in enumerate(tasks)
            }
            for fut in as_completed(fut_to_idx):
                res_row = fut.result()
                results.append(res_row)

        # Sort results by numeric ID if possible, otherwise lexicographically
        def _sort_key(r: Dict[str, str]):
            sid = r.get("ID", "")
            try:
                return (0, int(str(sid).strip()))
            except Exception:
                return (1, str(sid))

        results.sort(key=_sort_key)

        fieldnames = [
            "ID",
            "name",
            "final_code_text",
            "final_code_output",
        ]

        out_path.parent.mkdir(parents=True, exist_ok=True)

        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for res in results:
                row = {
                    "ID": res.get("ID", ""),
                    "name": res.get("name", ""),
                    "final_code_text": res.get("final_code_text", ""),
                    "final_code_output": res.get("final_code_output", ""),
                }
                writer.writerow(row)

        print(f"Completed sample {sample_idx + 1}/{total_samples}. Wrote {len(results)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
