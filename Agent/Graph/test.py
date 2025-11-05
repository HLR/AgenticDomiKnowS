from __future__ import annotations
import argparse, csv, io, os, json, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from Agent.main import pre_process_graph
from Agent.utils import load_all_examples_info
from langgraph.types import Command

def _run_single(task_id: str, task_name: str, task_text: str, graph_examples: List[str], reasoning_effort: bool) -> List[Dict[str, Any]]:
    """
    Run a single task end-to-end without calling Agent.main.main.
    Returns a list of CSV row dicts — one row per graph attempt with its reviews.
    Also captures a raw JSON snapshot of the agent state at each attempt.
    """
    buf_out = io.StringIO()
    buf_err = io.StringIO()
    rows: List[Dict[str, Any]] = []

    try:
        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            # Prepare graph and initial state
            try:
                numeric_task_id: Any = int(task_id)
            except Exception:
                numeric_task_id = task_id
            initial_state, graph = pre_process_graph(
                reasoning_effort=reasoning_effort,
                task_id=numeric_task_id,
                task_description=task_text,
                graph_examples=list(graph_examples or []),
                graph_rag_k=5,
                max_graphs_check=2,
            )

            # Drive the graph loop similarly to Agent.main.main
            config = {"configurable": {"thread_id": "ID: " + str(task_id)}}
            graph.invoke(initial_state, config=config, stream_mode="updates")

            # Collect per-iteration snapshots so we can attach them to attempts later
            iteration_snapshots: List[Dict[str, Any]] = []

            while True:
                snap = graph.get_state(config=config)
                # Deep-copy to plain JSON-serializable dict to avoid later mutation
                try:
                    snapshot_copy = json.loads(json.dumps(dict(snap.values or {})))
                except Exception:
                    # Fallback to shallow copy if something is not JSON-serializable
                    snapshot_copy = dict(snap.values or {})
                iteration_snapshots.append(snapshot_copy)

                if not snap.next or snap.next[0]=='sensor_agent_node':
                    break
                graph.invoke(Command(resume={"graph_human_approved": True}), config=config)

            state: Dict[str, Any] = dict(snap.values or {})

            # Build a lookup from attempt number -> snapshot dict
            attempt_to_snapshot: Dict[int, Dict[str, Any]] = {}
            for s in iteration_snapshots:
                try:
                    a = int(s.get("graph_attempt", 0))
                    if a > 0 and a not in attempt_to_snapshot:
                        attempt_to_snapshot[a] = s
                except Exception:
                    continue

            drafts: List[str] = list(state.get("graph_code_draft", []) or [])
            reviews: List[str] = list(state.get("graph_review_notes", []) or [])
            exe_notes: List[str] = list(state.get("graph_exe_notes", []) or [])

            attempts = max(len(drafts), len(reviews), len(exe_notes))
            for i in range(attempts):
                code_i = drafts[i] if i < len(drafts) else ""
                review_i = reviews[i] if i < len(reviews) else ""
                exec_i = exe_notes[i] if i < len(exe_notes) else ""
                reviewer_approved = ("approve" in (review_i or "").lower())
                exe_approved = (exec_i or "").strip() == ""

                # Attach the raw JSON snapshot for this attempt (if available)
                snapshot_dict = attempt_to_snapshot.get(i + 1)
                try:
                    snapshot_json = json.dumps(snapshot_dict, ensure_ascii=False)
                except Exception:
                    snapshot_json = "" if snapshot_dict is None else str(snapshot_dict)

                rows.append({
                    "id": task_id,
                    "name": task_name,
                    "task_description": task_text,
                    "attempt": i + 1,
                    "graph_code": code_i,
                    "reviewer_review": review_i,
                    "exec_notes": exec_i,
                    "reviewer_approved": reviewer_approved,
                    "exe_approved": exe_approved,
                    "agent_snapshot": snapshot_json,
                })

    except Exception:
        # If anything fails, return a single row with the error captured in exec_notes
        err_text = traceback.format_exc()
        rows.append({
            "id": task_id,
            "name": task_name,
            "task_description": task_text,
            "attempt": 1,
            "graph_code": "",
            "reviewer_review": "",
            "exec_notes": err_text,
            "reviewer_approved": False,
            "exe_approved": False,
            "agent_snapshot": "",
        })
    finally:
        stdout = buf_out.getvalue()
        stderr = buf_err.getvalue()
        buf_out.close()
        buf_err.close()
        # Attach stdout/stderr to the last row for debugging context
        if rows:
            rows[-1]["stdout"] = stdout
            rows[-1]["stderr"] = stderr

    return rows

def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Agent.main over CSV tasks in parallel")
    parser.add_argument("--csv-path", type=str, default="../datasets/lang_to_code_test.csv")
    parser.add_argument("--output-path", type=str, default="../datasets/")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--reasoning-effort", default="minimal", choices=["minimal", "low", "medium", "high"],help="Set the LLM reasoning effort level")

    args = parser.parse_args(argv)

    csv_path = Path(args.csv_path)
    out_path = Path(args.output_path+f"/{args.reasoning_effort}_"+"graph_test_results.csv")

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
        gold_graph = (row.get("graph") or "")+"\n"+(row.get("constraints") or "").strip()
        task_text = (desc + ("\n\n" + constr if constr else "")) if desc or constr else ""
        tasks.append((task_id, task_name, task_text, gold_graph))

    print(f"Starting {len(tasks)} tasks with {args.workers} workers. Output -> {out_path}")
    # Collect rows from all tasks
    results: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        fut_to_idx = {
            ex.submit(
                _run_single,
                task_id,
                task_name,
                task_text,
                load_all_examples_info("", gold_graph),
                args.reasoning_effort,
            ): i
            for i, (task_id, task_name, task_text, gold_graph) in enumerate(tasks)
        }
        for fut in as_completed(fut_to_idx):
            res_rows = fut.result()  # List[Dict]
            results.extend(res_rows)

    # Sort rows by task id then attempt for readability
    try:
        results.sort(key=lambda r: (str(r.get("id", "")), int(r.get("attempt", 0)) if str(r.get("attempt", "")).isdigit() else 0))
    except Exception:
        pass

    fieldnames = [
        "id",
        "name",
        "task_description",
        "attempt",
        "graph_code",
        "agent_snapshot",
        "reviewer_review",
        "exec_notes",
        "reviewer_approved",
        "exe_approved",
        "stdout",
        "stderr",
        "generated_at",
        "reasoning_effort",
    ]
    generated_at = datetime.now().isoformat(timespec="seconds")
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for res in results:
            row = {
                "id": res.get("id", ""),
                "name": res.get("name", ""),
                "task_description": res.get("task_description", ""),
                "attempt": res.get("attempt", ""),
                "graph_code": res.get("graph_code", ""),
                "agent_snapshot": res.get("agent_snapshot", ""),
                "reviewer_review": res.get("reviewer_review", ""),
                "exec_notes": res.get("exec_notes", ""),
                "reviewer_approved": res.get("reviewer_approved", False),
                "exe_approved": res.get("exe_approved", False),
                "stdout": res.get("stdout", ""),
                "stderr": res.get("stderr", ""),
                "generated_at": generated_at,
                "reasoning_effort": args.reasoning_effort,
            }
            writer.writerow(row)

    print(f"Completed. Wrote {len(results)} rows to {out_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())