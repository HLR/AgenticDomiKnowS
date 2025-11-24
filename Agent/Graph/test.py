from __future__ import annotations
import argparse, csv, io, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from Agent.main import pre_process_graph
from Agent.utils import load_all_examples_info
from langgraph.types import Command

def _run_single(task_id: str, task_name: str, task_text: str, graph_examples: List[str], reasoning_effort: str) -> List[Dict[str, Any]]:
    """
    Run a single task end-to-end without calling Agent.main.main.
    Returns a list of CSV row dicts — one row per graph attempt with its reviews.
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
                max_graphs_check=5,
            )

            # Drive the graph loop similarly to Agent.main.main
            config = {"configurable": {"thread_id": "ID: " + str(task_id)}}
            graph.invoke(initial_state, config=config, stream_mode="updates")

            while True:
                snap = graph.get_state(config=config)

                if not snap.next or snap.next[0]=='sensor_agent_node':
                    break
                graph.invoke(Command(resume={"graph_human_approved": True}), config=config)

            state: Dict[str, Any] = dict(snap.values or {})

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
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--reasoning-effort", default="low", choices=["minimal", "low", "medium", "high"],help="Set the LLM reasoning effort level")
    parser.add_argument("--samples", type=int, default=1, help="Number of repeated runs to execute and save (files will be suffixed 0..N-1 if N>1)")

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
        gold_graph = (row.get("graph") or "")+"\n"+(row.get("constraints") or "").strip()
        task_text = (desc + ("\n\n" + constr if constr else "")) if desc or constr else ""
        tasks.append((task_id, task_name, task_text, gold_graph))

    # Normalize samples value
    total_samples = args.samples if isinstance(args.samples, int) else 1
    if total_samples <= 0:
        print(f"[warn] --samples was {args.samples}, defaulting to 1")
        total_samples = 1

    for sample_idx in range(total_samples):
        # Decide output filename per sample
        output_dir = Path(args.output_path)
        if total_samples > 1:
            out_path = output_dir / f"{args.reasoning_effort}_graph_test_results_{sample_idx}.csv"
        else:
            out_path = output_dir / f"{args.reasoning_effort}_graph_test_results.csv"

        print(
            (
                f"Starting sample {sample_idx + 1}/{total_samples}: "
                f"{len(tasks)} tasks with {args.workers} workers. Output -> {out_path}"
            )
        )

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

        # Sort rows by numeric id when possible (fallback to lexicographic), then by attempt
        try:
            def _sort_key(r: Dict[str, Any]):
                sid = r.get("id", "")
                try:
                    id_key = (0, int(str(sid).strip()))
                except Exception:
                    id_key = (1, str(sid))
                try:
                    att = int(str(r.get("attempt", 0)).strip())
                except Exception:
                    att = 0
                return (id_key[0], id_key[1], att)

            results.sort(key=_sort_key)
        except Exception:
            pass

        fieldnames = [
            "id",
            "name",
            "task_description",
            "attempt",
            "graph_code",
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
        # Ensure the output directory exists before writing
        out_path.parent.mkdir(parents=True, exist_ok=True)
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

        print(f"Completed sample {sample_idx + 1}/{total_samples}. Wrote {len(results)} rows to {out_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())