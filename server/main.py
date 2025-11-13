from __future__ import annotations
import asyncio
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Response, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import sys; sys.path.append("../")
import os
from Agent.main import pre_process_graph
from server.model import typed_dict_to_model, BuildStateModel, typed_dict_changes
from server.session import *
from Agent.utils import load_all_examples_info
from langgraph.types import Command
from server.mongodb import MongoLoggingMiddleware, mongodb_lifespan
from server.auth import (
    UserCreate,
    LoginRequest,
    UserPublic,
    Token,
    CurrentUser,
    register_user,
    authenticate_and_create_token,
    revoke_access_token,
    get_current_token,
    get_current_user,
)

app = FastAPI(lifespan=mongodb_lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(MongoLoggingMiddleware)

initial_state, graph = pre_process_graph(
        reasoning_effort="minimal",
        task_id="Deploy",
        task_description="Create a Graph",
        graph_examples=load_all_examples_info("static/"),
        graph_rag_k=3,
        max_graphs_check=3
    )


async def log_graph_state(
    event: str,
    session_id: str,
    username: Optional[str],
    state_payload: dict,
    meta: Optional[dict] = None,
) -> None:
    collection = getattr(app.state, "mongo_graph_collection", None)
    if collection is None:
        return
    doc = {
        "event": event,
        "session_id": session_id,
        "username": username,
        "timestamp": datetime.utcnow(),
        "state": state_payload,
    }
    if meta:
        doc["meta"] = meta
    await asyncio.to_thread(collection.insert_one, doc)


@app.post("/auth/register", response_model=UserPublic, status_code=201)
async def register(user: UserCreate, request: Request):
    return await register_user(request, user)


@app.post("/auth/login", response_model=Token)
async def login(
    credentials: LoginRequest,
    request: Request,
    ctx = Depends(current_session),
):
    token = await authenticate_and_create_token(request, credentials, ctx["sid"])
    sess = ctx["session"]
    sess["user_id"] = credentials.username
    sess.setdefault("data", {})
    return token


@app.get("/whoami")
async def whoami(
    ctx = Depends(current_session),
    user: CurrentUser = Depends(get_current_user),
):
    sess = ctx["session"]
    sess.setdefault("data", {})
    return {
        "session_id": ctx["sid"],
        "user": user.model_dump(),
        "data": sess["data"],
    }


@app.post("/logout")
async def logout(
    request: Request,
    response: Response,
    ctx = Depends(current_session),
    token: Optional[str] = Depends(get_current_token),
    user: CurrentUser = Depends(get_current_user),
):
    sid = ctx["sid"]
    if token:
        await revoke_access_token(request, token)
    SESSIONS.pop(sid, None)
    response.delete_cookie(SESSION_COOKIE)
    return {"ok": True, "user": user.username}

@app.post("/reset-session")
async def reset_session(
    request: Request,
    response: Response,
    ctx = Depends(current_session),
    token: Optional[str] = Depends(get_current_token),
    user: CurrentUser = Depends(get_current_user),
):
    sid = ctx["sid"]
    SESSIONS.pop(sid, None)
    response.delete_cookie(SESSION_COOKIE)
    return {"ok": True, "user": user.username}


@app.get("/initialize-graph")
async def init_graph(
    task_description: str,
    ctx = Depends(current_session),
    user: CurrentUser = Depends(get_current_user),
):
    sess = ctx["session"]
    sess.setdefault("data", {})
    sess.setdefault("user_id", user.username)
    session_id = ctx["sid"]

    new_state = initial_state.copy()
    new_state["Task_definition"] = task_description
    new_state["Task_ID"] = str(session_id)
    config = {"configurable": {"thread_id": "ID: " + str(session_id)}}
    sess["data"]["config"] = config
    graph.invoke(new_state, config=config)
    build_state = typed_dict_to_model(graph.get_state(config=config).values, BuildStateModel)
    await log_graph_state(
        event="init",
        session_id=str(session_id),
        username=user.username,
        state_payload=build_state.model_dump(),
        meta={"task_description": task_description},
    )
    return build_state


@app.post("/continue-graph")
async def step_graph(
    buildstate: dict,
    ctx = Depends(current_session),
    user: CurrentUser = Depends(get_current_user),
):
    session_store = ctx["session"]
    session_store.setdefault("data", {})
    config = session_store["data"].get("config")
    if config is None:
        raise HTTPException(status_code=400, detail="Graph session not initialized")
    session_store.setdefault("user_id", user.username)

    state = buildstate
    prev_state = graph.get_state(config=config).values
    new_changes = typed_dict_changes(prev_state, state)
    if new_changes:
        graph.invoke(Command(resume=new_changes), config=ctx["session"]["data"]["config"])
    graph.invoke(None, config=ctx["session"]["data"]["config"])
    build_state = typed_dict_to_model(graph.get_state(config=config).values, BuildStateModel)
    await log_graph_state(
        event="step",
        session_id=str(ctx["sid"]),
        username=user.username,
        state_payload=build_state.model_dump(),
        meta={"changes": new_changes},
    )
    return build_state

@app.get("/graph-image/{task_id}/{attempt}")
async def get_graph_image(
    task_id: str,
    attempt: int,
    _user: CurrentUser = Depends(get_current_user),
):
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
            "Expires": "0",
        },
    )

if __name__ == "__main__":
    import uvicorn
    import argparse
    parser = argparse.ArgumentParser(description="Run the AgenticDomiKnowS server.")
    parser.add_argument("--port", type=int, default=8001, help="Port to run the server on.")
    args = parser.parse_args()
    uvicorn.run(app, host="0.0.0.0", port=args.port)