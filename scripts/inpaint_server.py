"""Keeps TRELLIS.2 loaded and runs inpainting jobs sent by the Blender add-on (blender/inpaint_addon.py).

Blender (Flatpak) can't start programs of the conda environment, but it can talk to this server over
the local network. TRELLIS.2 is loaded once at start (about a minute), so every job starts right away.

Usage (TRELLIS.2 environment, leave it running in its own terminal):
    source activate.sh
    python scripts/inpaint_server.py

Talks JSON on http://127.0.0.1:8765 (only this computer can reach it):
    POST /picture   {"model": .glb, "points": .npy (N x 3, model coordinates), "direction": [x, y, z],
                     "prompt": str, "reference": image or null, "out": .png, "seed": 1}
                    paints the prompt into a picture of the model (edit_picture.py, uses ComfyUI)
    POST /inpaint   {"model": .glb, "picture": image, "points": .npy, "out": .glb, "seed": 42, "grow": 2,
                     "steps": 12}   regenerates the area with TRELLIS.2 (inpaint.py)
    GET  /status    {"busy": bool, "kind": "picture" or "inpaint", "message": str, "result": path or null,
                     "error": str or null}
"""
import json
import os
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import generate
import numpy as np

from edit_picture import make_picture, view_from_direction
from inpaint import inpaint

PORT = int(os.environ.get("INPAINT_PORT", 8765))
state = {"busy": False, "kind": None, "message": "loading TRELLIS.2...", "result": None, "error": None}
lock = threading.Lock()


def report(message):
    state["message"] = message
    print(message, flush=True)


def run_inpaint(pipeline, job):
    return inpaint(pipeline, job["model"], job["picture"], np.load(job["points"]), job["out"],
                   grow=int(job.get("grow", 2)), seed=int(job.get("seed", 42)),
                   steps=int(job.get("steps", generate.DEFAULT_STEPS)), report=report)


def run_picture(pipeline, job):
    yaw, pitch = view_from_direction(job["direction"])
    return make_picture(job["model"], np.load(job["points"]), job["out"], job["prompt"], yaw, pitch,
                        seed=int(job.get("seed", 1)), reference_path=job.get("reference") or None, report=report)


JOBS = {"/inpaint": run_inpaint, "/picture": run_picture}


def run_job(pipeline, work, job):
    try:
        state.update(result=work(pipeline, job), message="done")
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
            if self.path not in JOBS:
                return self.answer(404, {"error": "unknown path"})
            job = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if not lock.acquire(blocking=False):
                return self.answer(409, {"error": "busy with another job"})
            state.update(busy=True, kind=self.path[1:], message="starting", result=None, error=None)
            print(f"Job {self.path}: {job}", flush=True)
            threading.Thread(target=run_job, args=(pipeline, JOBS[self.path], job), daemon=True).start()
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
