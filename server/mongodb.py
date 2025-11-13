from __future__ import annotations

import asyncio
import datetime
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from pymongo import MongoClient
from starlette.middleware.base import BaseHTTPMiddleware

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB", "agentic_server")
MONGODB_COLLECTION_NAME = os.getenv("MONGODB_COLLECTION", "interactions")
MONGODB_USER_COLLECTION_NAME = os.getenv("MONGODB_USER_COLLECTION", "users")
MONGODB_JWT_COLLECTION_NAME = os.getenv("MONGODB_JWT_COLLECTION", "jwt_tokens")
MONGODB_GRAPH_COLLECTION_NAME = os.getenv("MONGODB_GRAPH_COLLECTION", "graph_states")

logger = logging.getLogger(__name__)


class MongoLoggingMiddleware(BaseHTTPMiddleware):
	async def dispatch(self, request, call_next):
		start_time = time.time()
		try:
			request_body = await request.body()
		except Exception:
			request_body = b""
		try:
			response = await call_next(request)
		except Exception as exc:
			await self._insert_log(request, 500, start_time, request_body, None, repr(exc))
			raise
		await self._insert_log(request, response.status_code, start_time, request_body, response.headers, None)
		return response

	async def _insert_log(self, request, status_code, start_time, request_body, response_headers, error_detail):
		collection = getattr(request.app.state, "mongo_collection", None)
		if collection is None:
			return
		duration_ms = round((time.time() - start_time) * 1000, 2)
		body_text = None
		if request_body:
			body_text = request_body.decode("utf-8", errors="replace")
			if len(body_text) > 4096:
				body_text = body_text[:4096] + "...(truncated)"
		log_doc = {
			"timestamp": datetime.datetime.utcnow(),
			"method": request.method,
			"url": str(request.url),
			"path": request.url.path,
			"query_params": list(request.query_params.multi_items()),
			"client": request.client.host if request.client else None,
			"status_code": status_code,
			"duration_ms": duration_ms,
			"headers": dict(request.headers),
		}
		if body_text:
			log_doc["body"] = body_text
		if response_headers:
			log_doc["response_headers"] = dict(response_headers.items())
		if error_detail:
			log_doc["error"] = error_detail
		try:
			await asyncio.to_thread(collection.insert_one, log_doc)
		except Exception:
			logger.exception("Failed to write interaction log to MongoDB")


@asynccontextmanager
async def mongodb_lifespan(app: FastAPI) -> AsyncIterator[None]:
	try:
		app.state.mongo_client = MongoClient(MONGODB_URI)
		mongo_db = app.state.mongo_client[MONGODB_DB_NAME]
		app.state.mongo_db = mongo_db
		app.state.mongo_collection = mongo_db[MONGODB_COLLECTION_NAME]
		app.state.mongo_user_collection = mongo_db[MONGODB_USER_COLLECTION_NAME]
		app.state.mongo_jwt_collection = mongo_db[MONGODB_JWT_COLLECTION_NAME]
		app.state.mongo_graph_collection = mongo_db[MONGODB_GRAPH_COLLECTION_NAME]
	except Exception:
		logger.exception("Unable to connect to MongoDB")
		app.state.mongo_client = None
		app.state.mongo_db = None
		app.state.mongo_collection = None
		app.state.mongo_user_collection = None
		app.state.mongo_jwt_collection = None
		app.state.mongo_graph_collection = None
	try:
		yield
	finally:
		mongo_client = getattr(app.state, "mongo_client", None)
		if mongo_client:
			mongo_client.close()
