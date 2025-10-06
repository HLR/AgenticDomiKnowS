from __future__ import annotations
from fastapi import Request, Response
from uuid import uuid4
from datetime import datetime, timedelta

SESSIONS = {}
SESSION_COOKIE = "sid"
SESSION_TTL = timedelta(days=7)


def get_or_create_session(request: Request, response: Response):
    sid = request.cookies.get(SESSION_COOKIE)
    now = datetime.utcnow()

    if sid and (sess := SESSIONS.get(sid)):
        if sess["expires_at"] > now:
            sess["expires_at"] = now + SESSION_TTL
            return sid, sess

    sid = uuid4().hex
    sess = {"user_id": None, "data": {}, "expires_at": now + SESSION_TTL}
    SESSIONS[sid] = sess

    response.set_cookie(
        key=SESSION_COOKIE,
        value=sid,
        httponly=True,
        samesite="Lax",
        secure=False
    )
    return sid, sess


def current_session(request: Request, response: Response):
    sid, sess = get_or_create_session(request, response)
    return {"sid": sid, "session": sess}


__all__ = [
    "SESSIONS",
    "SESSION_COOKIE",
    "SESSION_TTL",
    "get_or_create_session",
    "current_session",
]
