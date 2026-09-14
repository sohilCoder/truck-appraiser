#!/usr/bin/env python3
"""
TruckVal v2 — server.

Run:  python3 server.py
Runs on port 8001 by default so v1 (port 8000) can stay up alongside it.
No pip install beyond scikit-learn and numpy.
"""

import json
import mimetypes
import os
import re
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
WEB  = ROOT / "web"


def load_env():
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


load_env()

import pipeline
import store

MAX_BODY   = 14 * 1024 * 1024
MAX_PHOTOS = 20

SESSION_RE = re.compile(r"^/api/sessions/([A-Za-z0-9_]+)$")
PHOTOS_RE  = re.compile(r"^/api/sessions/([A-Za-z0-9_]+)/photos$")
PHOTO_RE   = re.compile(r"^/api/sessions/([A-Za-z0-9_]+)/photos/([A-Za-z0-9_]+)$")
SPECS_RE   = re.compile(r"^/api/sessions/([A-Za-z0-9_]+)/specs$")

_UPLOAD_LOCKS = {}
_UL_GUARD     = threading.Lock()


def session_lock(sid):
    with _UL_GUARD:
        if sid not in _UPLOAD_LOCKS:
            _UPLOAD_LOCKS[sid] = threading.Lock()
        return _UPLOAD_LOCKS[sid]


# -------------------------------------------------------------------------
# Serialisation
# -------------------------------------------------------------------------

PHOTO_FIELDS = (
    "id", "filename", "accepted", "subject", "view", "view_quality",
    "quality_issues", "rejection_reason", "guidance", "findings",
    "assessment_notes", "mileage", "mileage_units",
    "truck_class", "make_guess", "model_guess", "year_guess",
    "identity_confidence", "overall_condition",
    "rust_severity", "body_damage_severity",
    "tire_wear_severity", "interior_wear_severity",
    "frame_concern", "mechanical_warning", "error",
)


def ser_photo(p):
    return {k: p.get(k) for k in PHOTO_FIELDS}


def ser_session(s):
    return {
        "session_id":     s["id"],
        "photos":         [ser_photo(p) for p in s["photos"]],
        "accepted_count": sum(1 for p in s["photos"] if p.get("accepted")),
        "rejected_count": sum(1 for p in s["photos"] if not p.get("accepted")),
        "typed_specs":    s.get("typed_specs") or {},
        "truck_class":    s.get("truck_class"),
        "coverage":       s.get("coverage"),
        "findings":       s.get("findings") or [],
        "valuation":      s.get("valuation"),
        "comparables":    s.get("comparables") or [],
        "merged_features": s.get("merged"),
        "recommendation": s.get("recommendation"),
        "next_priorities": s.get("next_priorities") or [],
        "error":          s.get("error"),
        "mock_mode":      pipeline.MOCK,
        "version":        "2",
    }


# -------------------------------------------------------------------------
# Request handler
# -------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version   = "TruckValV2/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("  " + (fmt % args) + "\n")

    def _send(self, status, payload, ct="application/json"):
        if isinstance(payload, (dict, list)):
            body = json.dumps(payload).encode()
        elif isinstance(payload, str):
            body = payload.encode()
        else:
            body = payload
        self.send_response(status)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _err(self, status, msg):
        self._send(status, {"error": msg})

    def _read_json(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:   return {}
        if n > MAX_BODY:
            raise ValueError("Request body too large.")
        return json.loads(self.rfile.read(n).decode())

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/health":
            from llm_client import PROVIDER, MODEL
            return self._send(200, {
                "ok": True, "version": 2, "mock_mode": pipeline.MOCK,
                "provider": PROVIDER, "model": MODEL,
                "gp_ready": pipeline._GP is not None,
            })
        m = SESSION_RE.match(path)
        if m:
            s = store.get(m.group(1))
            return self._send(200, ser_session(s)) if s else self._err(404, "Session not found.")
        return self._serve_static(path)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            if path == "/api/sessions":
                s = store.create()
                pipeline.build_session_state(s)
                return self._send(201, ser_session(s))
            m = PHOTOS_RE.match(path)
            if m:
                return self._handle_upload(m.group(1))
            return self._err(404, "Unknown endpoint.")
        except ValueError as exc:
            return self._err(400, str(exc))
        except Exception:
            traceback.print_exc()
            return self._err(500, "Server error — check terminal output.")

    def do_PATCH(self):
        path = urlparse(self.path).path
        try:
            m = SPECS_RE.match(path)
            if m:
                return self._handle_specs(m.group(1))
            return self._err(404, "Unknown endpoint.")
        except ValueError as exc:
            return self._err(400, str(exc))
        except Exception:
            traceback.print_exc()
            return self._err(500, "Server error.")

    def do_DELETE(self):
        path = urlparse(self.path).path
        m = PHOTO_RE.match(path)
        if not m:
            return self._err(404, "Unknown endpoint.")
        s = store.get(m.group(1))
        if not s:
            return self._err(404, "Session not found.")
        pid = m.group(2)
        with session_lock(s["id"]):
            before = len(s["photos"])
            s["photos"] = [p for p in s["photos"] if p["id"] != pid]
            if len(s["photos"]) == before:
                return self._err(404, "Photo not found.")
            pipeline.build_session_state(s)
        return self._send(200, ser_session(s))

    def _handle_upload(self, sid):
        s = store.get(sid)
        if not s:
            return self._err(404, "Session not found.")
        payload  = self._read_json()
        data_url = payload.get("data_url") or ""
        filename = str(payload.get("filename") or "photo.jpg")[:120]
        if not data_url.startswith("data:image/"):
            return self._err(400, "Not an image. Upload a JPEG, PNG, WebP or HEIC photo.")
        if len(s["photos"]) >= MAX_PHOTOS:
            return self._err(400, f"Session already has {MAX_PHOTOS} photos.")

        with session_lock(sid):
            pid   = store.next_photo_id(s)
            photo = pipeline.process_photo(pid, data_url, filename)
            photo["_data_url"] = data_url   # in-memory only, never serialised
            s["photos"].append(photo)
            s["updated_at"] = __import__("time").time()
            pipeline.build_session_state(s)

        resp = ser_session(s)
        resp["uploaded_photo"] = ser_photo(photo)
        return self._send(200, resp)

    def _handle_specs(self, sid):
        s = store.get(sid)
        if not s:
            return self._err(404, "Session not found.")
        payload = self._read_json()
        specs   = {}

        year = payload.get("year")
        if year not in (None, ""):
            try:
                y = int(year)
                if 1950 <= y <= 2027: specs["year"] = y
                else: raise ValueError
            except (TypeError, ValueError):
                raise ValueError("Year must be between 1950 and 2027.")

        for field in ("make", "model", "trim"):
            v = payload.get(field)
            if v not in (None, ""):
                specs[field] = str(v).strip()[:40]

        mileage = payload.get("mileage")
        if mileage not in (None, ""):
            try:
                m = int(str(mileage).replace(",", "").strip())
                if 0 < m < 2_000_000: specs["mileage"] = m
                else: raise ValueError
            except (TypeError, ValueError):
                raise ValueError("Mileage must be a number under 2,000,000.")

        specs["mileage_units"] = "km" if str(payload.get("mileage_units") or "").lower() == "km" else "miles"

        with session_lock(sid):
            s["typed_specs"] = specs
            pipeline.build_session_state(s, force_voi=True)

        return self._send(200, ser_session(s))

    def _serve_static(self, path):
        if path == "/": path = "/index.html"
        target = (WEB / path.lstrip("/")).resolve()
        if not str(target).startswith(str(WEB.resolve())) or not target.is_file():
            return self._err(404, "Not found.")
        ct, _ = mimetypes.guess_type(str(target))
        self._send(200, target.read_bytes(), ct or "application/octet-stream")


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def main():
    port = int(os.environ.get("PORT", "8001"))

    # Validate web folder
    missing = [f for f in ("index.html", "styles.css", "app.js") if not (WEB / f).exists()]
    if not WEB.exists() or missing:
        print(f"\n  ERROR: web/ folder or files missing: {missing or 'folder itself'}")
        print("  Make sure the full truck-valuation-ai-v2 folder was downloaded.\n")
        sys.exit(1)

    # Pre-load GP (catches missing comps.csv early)
    print("\n  Loading GP pricing model…", end=" ", flush=True)
    try:
        pipeline.get_gp()
        n = len(pipeline._GP.comps)
        print(f"OK — {n} comps loaded.")
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n\n  ERROR: {exc}\n")
        sys.exit(1)

    from llm_client import PROVIDER, MODEL
    print(f"\n  TruckVal v2")
    print(f"  http://localhost:{port}")
    print(f"  Root    : {ROOT}")
    print(f"  Provider: {PROVIDER}  |  Model: {MODEL}")
    if pipeline.MOCK:
        print("  Mode    : OFFLINE MOCK (set TRUCKVAL_MOCK=1)")
    print()

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    server.daemon_threads = True
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
