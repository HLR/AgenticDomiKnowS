from __future__ import annotations
from fastapi import FastAPI, Response, Depends, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import sys; sys.path.append("../")
import os
from Agent.main import pre_process_graph
from server.model import typed_dict_to_model, model_to_typed_dict, BuildStateModel, typed_dict_changes
from server.session import *
from Agent.utils import load_all_examples_info

app = FastAPI()
origins = [
    "http://localhost:3000",
    "http://localhost:49790",
    "https://hlr-demo.egr.msu.edu",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

initial_state, graph = pre_process_graph(
        reasoning_effort="minimal",
        task_id="Deploy",
        task_description="Create a Graph",
        graph_examples=load_all_examples_info("static/"),
        graph_rag_k=3,
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
@app.get("/initialize-graph")
def init_graph(task_description: str, ctx = Depends(current_session)):
    sess = ctx["session"]
    sess.setdefault("data", {})
    session_id = ctx["sid"]

    new_state = initial_state.copy()
    new_state["Task_definition"] = task_description
    new_state["Task_ID"] = str(session_id)
    config = {"configurable": {"thread_id": "ID: " + str(session_id)}}
    sess["data"]["config"] = config
    graph.invoke(new_state, config=config)  # FIXED: Use new_state instead of initial_state
    return typed_dict_to_model(graph.get_state(config=config).values, BuildStateModel)

@app.post("/continue-graph")
def step_graph(buildstate: dict, ctx = Depends(current_session)):
    config = ctx["session"]["data"]["config"]

    # buildstate will be received as a plain dict from the request body.
    # Convert or use directly as a mapping for update operations.
    state = buildstate
    prev_state = graph.get_state(config=config).values
    new_changes = typed_dict_changes(prev_state, state)
    if new_changes:
        # Use the node name defined in Agent.main (graph_human_agent)
        graph.update_state(ctx["session"]["data"]["config"], state, as_node="graph_human_agent")
    graph.invoke(None, config=ctx["session"]["data"]["config"])
    return typed_dict_to_model(graph.get_state(config=config).values, BuildStateModel)

@app.get("/graph-image/{task_id}/{attempt}")
def get_graph_image(task_id: str, attempt: int):
    """
    Retrieve the graph visualization image for a specific task and attempt.
    Images are stored in graph_images/{task_id}_{attempt}.png
    """
    # Construct the file path
    image_filename = f"{task_id}_{attempt}.png"
    image_path = os.path.join("graph_images", image_filename)
    
    # Check if file exists
    if not os.path.exists(image_path):
        raise HTTPException(
            status_code=404, 
            detail=f"Graph image not found for task {task_id} at attempt {attempt}"
        )
    
    # Return the image file
    return FileResponse(
        image_path,
        media_type="image/png",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
