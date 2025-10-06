from __future__ import annotations
from fastapi import FastAPI, Request, Response, Depends
from fastapi.middleware.cors import CORSMiddleware
import sys; sys.path.append("../")
from Agent.main import pre_process_graph
from server.model import typed_dict_to_model, model_to_typed_dict, BuildStateModel, typed_dict_changes
from server.session import *
from Agent.graph_prompt import load_all_graphs

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

initial_state, graph = pre_process_graph(
        test_run=False,
        task_id="Deploy",
        task_description="Create a Graph",
        graph_examples=load_all_graphs("static/"),
        rag_k=3,
        max_graphs_check=3
    )
@app.get("/whoami")
def whoami(ctx = Depends(current_session)):
    return {
        "session_id": ctx["sid"],
        "data": ctx["session"]["data"]
    }

@app.post("/logout")
def logout(response: Response, ctx = Depends(current_session)):
    sid = ctx["sid"]
    SESSIONS.pop(sid, None)
    response.delete_cookie(SESSION_COOKIE)
    return {"ok": True}
@app.get("/UI")
def init_graph(task_description: str, ctx = Depends(current_session)):
    sess = ctx["session"]
    sess.setdefault("data", {})
    session_id = ctx["sid"]

    new_state = initial_state.copy()
    new_state["Task_definition"] = task_description
    config = {"configurable": {"thread_id": "ID: " + str(session_id)}}
    sess["data"]["config"] = config
    graph.invoke(initial_state, config=config)
    return typed_dict_to_model(graph.get_state(config=config).values, BuildStateModel)

@app.post("/UI")
def step_graph(buildstate: BuildStateModel, ctx = Depends(current_session)):
    config = ctx["session"]["data"]["config"]

    state = model_to_typed_dict(buildstate)
    prev_state = graph.get_state(config=config).values
    new_changes = typed_dict_changes(prev_state, state)
    if new_changes:
        graph.update_state(ctx["session"]["data"]["config"], state, as_node="graph_human")
    graph.invoke(None, config=ctx["session"]["data"]["config"])
    return typed_dict_to_model(graph.get_state(config=config).values, BuildStateModel)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
