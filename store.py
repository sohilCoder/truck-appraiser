"""In-memory session store — identical pattern to v1."""

import threading
import time
import uuid

_LOCK     = threading.RLock()
_SESSIONS = {}
MAX       = 200


def create():
    with _LOCK:
        _evict()
        sid = "s_" + uuid.uuid4().hex[:10]
        _SESSIONS[sid] = {
            "id": sid, "created_at": time.time(), "updated_at": time.time(),
            "photos": [], "typed_specs": {}, "truck_class": None,
            "valuation": None, "coverage": None, "comparables": [],
            "merged": None, "findings": [], "recommendation": None,
            "next_priorities": [], "error": None,
        }
        return _SESSIONS[sid]


def get(sid):
    with _LOCK:
        return _SESSIONS.get(sid)


def lock():
    return _LOCK


def next_photo_id(session):
    return f"photo_{len(session['photos']) + 1}"


def _evict():
    if len(_SESSIONS) <= MAX:
        return
    oldest = sorted(_SESSIONS.values(), key=lambda s: s["updated_at"])
    for s in oldest[:len(_SESSIONS) - MAX + 1]:
        _SESSIONS.pop(s["id"], None)
