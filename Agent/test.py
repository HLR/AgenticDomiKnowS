from __future__ import annotations
import argparse, csv, io, os
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from Agent.main import main as run_main

def _run_single(task_id: str, task_name: str, task_text: str,graph_examples: List[str], test_run: bool) -> Dict[str, Any]:
    argv: List[str] = ["--task-id", task_id ,"--task-description", task_text, "--graph-examples", *graph_examples]
    if test_run:
        argv.append("--test-run")

    buf_out = io.StringIO()
    buf_err = io.StringIO()
    rc: int = 1
    graph_code: Optional[str] = None
    exc: Optional[str] = None

    try:
        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            result = run_main(argv)
        if isinstance(result, tuple) and len(result) == 2:
            rc, graph_code = int(result[0]), result[1]
        else:
            rc = int(result)
    except Exception as e:
        exc = f"{type(e).__name__}: {e}"
    finally:
        stdout = buf_out.getvalue()
        stderr = buf_err.getvalue()
        buf_out.close()
        buf_err.close()

    return {
        "id": task_id,
        "name": task_name,
        "task_description": task_text,
        "return_code": rc,
        "graph_code": graph_code,
        "stdout": stdout,
        "stderr": stderr,
        "exception": exc,
    }

def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Agent.main over CSV tasks in parallel")
    parser.add_argument("--csv-path", type=str, default="lang_to_code_test.csv")
    parser.add_argument("--output-path", type=str, default="test_results.csv")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--test-run", default=True, action="store_true", help="Use gpt-4o-mini instead of gpt5")
    args = parser.parse_args(argv)

    csv_path = Path(args.csv_path)
    out_path = Path(args.output_path)

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
    results: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        fut_to_idx = {
            ex.submit(_run_single, task_id, task_name, task_text,[x for j in tasks if j != gold_graph for x in (j[2], j[3])], args.test_run): i
            for i, (task_id, task_name, task_text, gold_graph) in enumerate(tasks)
        }
        for fut in as_completed(fut_to_idx):
            res = fut.result()
            results.append(res)

    fieldnames = [
        "id",
        "name",
        "task_description",
        "graph_code",
        "generated_at",
        "test_run",
    ]
    generated_at = datetime.now().isoformat(timespec="seconds")
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for res in results:
            row = dict()
            row["id"] = res["id"]
            row["name"] = res["name"]
            row["task_description"] = res["task_description"]
            row["graph_code"] = res["graph_code"]
            row["generated_at"] = generated_at
            row["test_run"] = bool(args.test_run)
            writer.writerow(row)

    print(f"Completed. Wrote {len(results)} results to {out_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())