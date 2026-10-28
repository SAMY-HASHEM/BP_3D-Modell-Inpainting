"""Keeps TRELLIS.2 loaded and runs inpainting jobs sent by the Blender add-on (blender/inpaint_addon.py).

Two ways to use it:
  * on this computer (default): Blender on the same PC talks to http://127.0.0.1:8765
  * inside the uni network: start it on the GPU server with --host 0.0.0.0; Blender on any PC in the
    network talks to http://<server>:8765 (e.g. nc.hcigroup.de:8765). From outside the network, use the
    cloud route instead (cloud.py / cloud_worker.py).

Every request needs the server's token (header X-Token). It is made once and kept in
~/.config/inpainting3d/server_token (readable only by you); the server prints it at the start, and it
goes into Blender's add-on preferences. TRELLIS.2 is loaded once (about a minute), so jobs start right away.

Usage (TRELLIS.2 environment, leave it running in its own terminal):
    source activate.sh
    python scripts/inpaint_server.py                         # only this computer
    python scripts/inpaint_server.py --host 0.0.0.0          # the uni network (on the GPU server)
    python scripts/inpaint_server.py --host 0.0.0.0 --port 8770 --gpu 1

Talks JSON on http://<host>:<port>:
    POST /job       body: a job .zip (the add-on's job package)        regenerates the area (inpaint.py)
    GET  /result?job=<id>                                             the finished .glb
    GET  /status    {"busy", "kind", "job", "message", "result", "error"}
  only when it runs for this computer alone (they pass file paths of this computer):
    POST /inpaint   {"model", "picture", "points", "out", "seed", "grow", "steps"}
    POST /picture   {"model", "points", "direction", "prompt", "reference", "out", "seed"}
"""
import argparse
import hmac
import json
import os
import secrets
import threading
import time
import traceback
import urllib.parse
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import generate
import numpy as np

from edit_picture import make_picture, view_from_direction
from inpaint import inpaint
from run_job import freest_gpu

PORT = int(os.environ.get("INPAINT_PORT", 8765))
TOKEN_FILE = os.path.expanduser("~/.config/inpainting3d/server_token")
JOB_DIR = os.path.join(generate.OUTPUT_DIR, "server_jobs")
MAX_JOB_SIZE = 500 * 2**20  # a job .zip is ~40 MB
state = {"busy": False, "kind": None, "job": None, "message": "loading TRELLIS.2...", "result": None, "error": None}
lock = threading.Lock()


def report(message):
    state["message"] = message
    print(message, flush=True)


def server_token():
    """The token, made once and kept readable only by you."""
    if not os.path.isfile(TOKEN_FILE):
        os.makedirs(os.path.dirname(TOKEN_FILE), exist_ok=True)
        with open(os.open(TOKEN_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f:
            f.write(secrets.token_urlsafe(24))
    with open(TOKEN_FILE) as f:
        return f.read().strip()


# ---------------------------------------------------------------- the jobs

def run_inpaint(pipeline, job):
    return inpaint(pipeline, job["model"], job["picture"], np.load(job["points"]), job["out"],
                   grow=int(job.get("grow", 2)), seed=int(job.get("seed", 42)),
                   steps=int(job.get("steps", generate.DEFAULT_STEPS)), report=report)


def run_picture(pipeline, job):
    yaw, pitch = view_from_direction(job["direction"])
    return make_picture(job["model"], np.load(job["points"]), job["out"], job["prompt"], yaw, pitch,
                        seed=int(job.get("seed", 1)), reference_path=job.get("reference") or None, report=report)


def run_package(pipeline, folder):
    """A job .zip from the add-on, unpacked in `folder`: checked, then regenerated."""
    with open(os.path.join(folder, "job.json")) as f:
        job = json.load(f)
    if job.get("kind", "inpaint") != "inpaint":
        raise ValueError("this server runs inpainting jobs only")
    names = {k: os.path.basename(job[k]) for k in ("model", "points", "picture")}  # only files of the job
    paths = {k: os.path.join(folder, v) for k, v in names.items()}
    for k, p in paths.items():
        if not os.path.isfile(p):
            raise ValueError(f"the job has no {k} file")
    seed, grow, steps = int(job.get("seed", 42)), int(job.get("grow", 2)), int(job.get("steps", generate.DEFAULT_STEPS))
    if not (0 <= grow <= 8 and 1 <= steps <= 100):
        raise ValueError("grow or steps out of range")
    method = job.get("method", "repaint")
    if method not in ("repaint", "inversion"):
        raise ValueError(f"unknown method: {method}")
    extra = {k: os.path.join(folder, os.path.basename(job[k])) for k in ("source", "picture_mask") if job.get(k)}
    for p in extra.values():
        if not os.path.isfile(p):
            raise ValueError(f"the job has no {os.path.basename(p)}")
    out = os.path.join(folder, f"{os.path.basename(folder)}_seed{seed}.glb")
    return inpaint(pipeline, paths["model"], paths["picture"], np.load(paths["points"]), out,
                   grow=grow, seed=seed, steps=steps, report=report, method=method,
                   source_path=extra.get("source"), picture_mask_path=extra.get("picture_mask"))


LOCAL_JOBS = {"/inpaint": run_inpaint, "/picture": run_picture}


def run_job(work, *args):
    try:
        state.update(result=work(*args), message="done")
    except Exception as e:
        traceback.print_exc()
        state.update(error=f"{type(e).__name__}: {e}", message="failed")
    finally:
        state["busy"] = False
        lock.release()


# ---------------------------------------------------------------- the web server

def make_handler(pipeline, token, local_only):
    class Handler(BaseHTTPRequestHandler):
        def answer(self, code, data):
            body = json.dumps(data).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def allowed(self):
            if hmac.compare_digest(self.headers.get("X-Token", ""), token):
                return True
            self.answer(401, {"error": "wrong or missing token"})
            return False

        def do_GET(self):
            if not self.allowed():
                return
            url = urllib.parse.urlparse(self.path)
            if url.path == "/status":
                return self.answer(200, state)
            if url.path == "/result":
                job = urllib.parse.parse_qs(url.query).get("job", [""])[0]
                if state["busy"] or job != state["job"] or not state["result"] or not os.path.isfile(state["result"]):
                    return self.answer(404, {"error": "no finished result for this job"})
                with open(state["result"], "rb") as f:
                    body = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "model/gltf-binary")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.answer(404, {"error": "unknown path"})

        def do_POST(self):
            if not self.allowed():
                return
            size = int(self.headers.get("Content-Length", 0))
            if self.path == "/job":
                if not 0 < size <= MAX_JOB_SIZE:
                    return self.answer(413, {"error": "job too big or empty"})
                body = self.rfile.read(size)
                if not lock.acquire(blocking=False):
                    return self.answer(409, {"error": "busy with another job"})
                job_id = time.strftime("job_%Y%m%d-%H%M%S")
                folder = os.path.join(JOB_DIR, job_id)
                os.makedirs(folder, exist_ok=True)
                with open(os.path.join(folder, "job.zip"), "wb") as f:
                    f.write(body)
                try:
                    with zipfile.ZipFile(os.path.join(folder, "job.zip")) as z:
                        z.extractall(folder)  # zipfile drops '..' and absolute paths
                except zipfile.BadZipFile:
                    lock.release()
                    return self.answer(400, {"error": "not a job .zip"})
                state.update(busy=True, kind="job", job=job_id, message="starting", result=None, error=None)
                print(f"Job {job_id} from {self.client_address[0]} ({size / 2**20:.0f} MB)", flush=True)
                threading.Thread(target=run_job, args=(run_package, pipeline, folder), daemon=True).start()
                return self.answer(202, {"started": True, "job": job_id})
            if self.path in LOCAL_JOBS:
                if not local_only:
                    return self.answer(403, {"error": "jobs with file paths only work on the server's own computer"})
                job = json.loads(self.rfile.read(size))
                if not lock.acquire(blocking=False):
                    return self.answer(409, {"error": "busy with another job"})
                state.update(busy=True, kind=self.path[1:], job=None, message="starting", result=None, error=None)
                print(f"Job {self.path}: {job}", flush=True)
                threading.Thread(target=run_job, args=(LOCAL_JOBS[self.path], pipeline, job), daemon=True).start()
                return self.answer(202, {"started": True})
            self.answer(404, {"error": "unknown path"})

        def log_message(self, *args):
            pass  # Blender asks for the status every second; don't print each request

    return Handler


def main():
    parser = argparse.ArgumentParser(description="TRELLIS.2 inpainting server for the Blender add-on")
    parser.add_argument("--host", default="127.0.0.1",
                        help="127.0.0.1 = only this computer (default); 0.0.0.0 = reachable in the network")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--gpu", type=int, help="GPU to use (default: the one with the least memory in use)")
    args = parser.parse_args()

    gpu = args.gpu if args.gpu is not None else freest_gpu()
    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)  # before TRELLIS.2 touches the GPU
    token = server_token()
    local_only = args.host in ("127.0.0.1", "localhost")
    pipeline = generate.load_pipeline()
    state["message"] = "ready"
    server = ThreadingHTTPServer((args.host, args.port), make_handler(pipeline, token, local_only))
    where = f"http://127.0.0.1:{args.port} (this computer only)" if local_only else \
        f"port {args.port} of this computer, from the network (e.g. {os.uname().nodename}:{args.port})"
    print(f"Ready on GPU {gpu}: waiting for Blender on {where}", flush=True)
    print(f"Token for Blender (add-on preferences): {token}", flush=True)
    print("Stop with Ctrl+C", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
