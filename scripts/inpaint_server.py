"""Keeps TRELLIS.2 loaded and runs inpainting jobs sent by the Blender add-on (blender/inpaint_addon.py).

Blender (Flatpak) can't start programs of the conda environment, but it can talk to this server over
the local network. TRELLIS.2 is loaded once at start (about a minute), so every job starts right away.

Usage (TRELLIS.2 environment, leave it running in its own terminal):
    source activate.sh
    python scripts/inpaint_server.py

Talks JSON on http://127.0.0.1:8765 (only this computer can reach it):
    POST /inpaint   {"model": .glb, "picture": image, "points": .npy (N x 3, model coordinates),
                     "out": .glb, "seed": 42, "grow": 2, "steps": 12}   starts a job
    GET  /status    {"busy": bool, "message": str, "result": .glb or null, "error": str or null}
"""
import json
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import generate
import numpy as np

from inpaint import inpaint

PORT = 8765
state = {"busy": False, "message": "loading TRELLIS.2...", "result": None, "error": None}
lock = threading.Lock()


def report(message):
    state["message"] = message
    print(message, flush=True)


def run_job(pipeline, job):
    try:
        out = inpaint(pipeline, job["model"], job["picture"], np.load(job["points"]), job["out"],
                      grow=int(job.get("grow", 2)), seed=int(job.get("seed", 42)),
                      steps=int(job.get("steps", generate.DEFAULT_STEPS)), report=report)
        state.update(result=out, message="done")
    except Exception as e:
        traceback.print_exc()
        state.update(error=f"{type(e).__name__}: {e}", message="failed")
    finally:
        state["busy"] = False
        lock.release()


def make_handler(pipeline):
    class Handler(BaseHTTPRequestHandler):
        def answer(self, code, data):
            body = json.dumps(data).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/status":
                self.answer(200, state)
            else:
                self.answer(404, {"error": "unknown path"})

        def do_POST(self):
            if self.path != "/inpaint":
                return self.answer(404, {"error": "unknown path"})
            job = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if not lock.acquire(blocking=False):
                return self.answer(409, {"error": "busy with another job"})
            state.update(busy=True, message="starting", result=None, error=None)
            print(f"Job: {job}", flush=True)
            threading.Thread(target=run_job, args=(pipeline, job), daemon=True).start()
            self.answer(202, {"started": True})

        def log_message(self, *args):
            pass  # Blender asks for the status every second; don't print each request

    return Handler


def main():
    pipeline = generate.load_pipeline()
    state["message"] = "ready"
    server = ThreadingHTTPServer(("127.0.0.1", PORT), make_handler(pipeline))
    print(f"Ready: waiting for Blender on http://127.0.0.1:{PORT} (stop with Ctrl+C)", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
