from __future__ import annotations
import argparse, csv, io, os, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from typing import Dict, List, Any

from Agent.main import pre_process_graph
from Agent.utils import load_all_examples_info
from langgraph.types import Command

def _run_single(task_id: str, task_name: str, task_text: str, graph_examples: List[str], reasoning_effort: str) -> Dict[str, Any]:
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
                graph_examples=list(graph_examples or []),
                graph_rag_k=5,
                max_graphs_check=3,
            )

            config = {"configurable": {"thread_id": "ID: " + str(task_id)}}
            graph.invoke(initial_state, config=config, stream_mode="updates")

            while True:
                snap = graph.get_state(config=config)
                if not snap.next or snap.next[0]=='property_agent_node':
                    break
                graph.invoke(Command(resume={"graph_human_approved":True}), config=config)

            state: Dict[str, Any] = dict(snap.values or {})
            sensor_codes = state.get("sensor_codes", []) or []
            entire_sensor_codes = state.get("entire_sensor_codes", []) or []
            sensor_code_outputs = state.get("sensor_code_outputs", []) or []

            return {
                "sensor_codes": list(sensor_codes),
                "entire_sensor_codes": list(entire_sensor_codes),
                "sensor_code_outputs": list(sensor_code_outputs),
            }

    except Exception:
        err_text = traceback.format_exc()
        # On failure, return lists consistent with the new schema
        return {
            "sensor_codes": [],
            "entire_sensor_codes": [],
            # Store the error text as an output entry so it appears in CSV
            "sensor_code_outputs": [err_text],
        }
    finally:
        buf_out.close()
        buf_err.close()


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Agent Sensor over CSV tasks in parallel")
    parser.add_argument("--csv-path", type=str, default="../datasets/lang_to_code_test.csv")
    parser.add_argument("--output-path", type=str, default="../datasets/")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--reasoning-effort", default="minimal", choices=["minimal", "low", "medium", "high"], help="Set the LLM reasoning effort level",)

    args = parser.parse_args(argv)

    csv_path = Path(args.csv_path)
    out_path = Path(args.output_path + f"/{args.reasoning_effort}_" + "sensor_test_results.csv")

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
        tasks.append((task_id, task_name, task_text, gold_graph))

    print(f"Starting {len(tasks)} tasks with {args.workers} workers. Output -> {out_path}")

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
            res_row = fut.result()
            results.append(res_row)

    # Determine dynamic column names based on the maximum lengths of returned lists
    max_sensor_codes = 0
    max_entire_codes = 0
    max_outputs = 0
    for res in results:
        max_sensor_codes = max(max_sensor_codes, len(res.get("sensor_codes", []) or []))
        max_entire_codes = max(max_entire_codes, len(res.get("entire_sensor_codes", []) or []))
        max_outputs = max(max_outputs, len(res.get("sensor_code_outputs", []) or []))

    fieldnames: List[str] = []
    fieldnames += [f"sensor_code_{i+1}" for i in range(max_sensor_codes)]
    fieldnames += [f"entire_sensor_code_{i+1}" for i in range(max_entire_codes)]
    fieldnames += [f"sensor_code_output_{i+1}" for i in range(max_outputs)]

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for res in results:
            row: Dict[str, Any] = {}
            sensor_codes = list(res.get("sensor_codes", []) or [])
            entire_sensor_codes = list(res.get("entire_sensor_codes", []) or [])
            sensor_code_outputs = list(res.get("sensor_code_outputs", []) or [])

            # Populate per-indexed columns, padding with empty strings as needed
            for i in range(max_sensor_codes):
                row[f"sensor_code_{i+1}"] = sensor_codes[i] if i < len(sensor_codes) else ""
            for i in range(max_entire_codes):
                row[f"entire_sensor_code_{i+1}"] = (
                    entire_sensor_codes[i] if i < len(entire_sensor_codes) else ""
                )
            for i in range(max_outputs):
                row[f"sensor_code_output_{i+1}"] = (
                    sensor_code_outputs[i] if i < len(sensor_code_outputs) else ""
                )

            writer.writerow(row)

    print(f"Completed. Wrote {len(results)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
