"""Running the jobs of scripts/pipeline/nodes_core.py from ComfyUI.

The nodes run inside ComfyUI's Python, but TRELLIS.2, the vision AI and our scripts need the project's
conda environment. So every step runs as its own process: `source activate.sh && <command>`. ComfyUI's
environment stays untouched, and the GPU memory is free again after each step.
"""
import importlib.util
import os
import subprocess
import time

import numpy as np
import torch
from PIL import Image

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
_spec = importlib.util.spec_from_file_location("bp_nodes_core", os.path.join(PROJECT, "scripts", "pipeline", "nodes_core.py"))
core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(core)  # standard library only: safe inside ComfyUI


def run_step(command, folder, label, done=None):
    """One command in the project environment, its output in folder/log.txt. Skipped when `done` exists.
    ComfyUI's Cancel stops it."""
    if done and os.path.exists(done):
        print(f"[BP] {label}: done before ({os.path.relpath(folder, PROJECT)})", flush=True)
        return
    import comfy.model_management as mm
    quoted = " ".join("'" + str(c).replace("'", "'\\''") + "'" for c in command)
    shell = f"source '{PROJECT}/activate.sh' >/dev/null && exec {quoted}"
    env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME")}
    env["PYTHONUNBUFFERED"] = "1"
    print(f"[BP] {label}: running ({os.path.relpath(folder, PROJECT)})", flush=True)
    start = time.time()
    log_path = os.path.join(folder, "log.txt" if not label.endswith("preview") else "log_preview.txt")
    with open(log_path, "w") as log:
        process = subprocess.Popen(["bash", "-c", shell], stdout=log, stderr=subprocess.STDOUT, cwd=PROJECT, env=env)
        try:
            while process.poll() is None:
                time.sleep(1)
                mm.throw_exception_if_processing_interrupted()
        except BaseException:
            process.kill()
            raise
    if process.returncode != 0 or (done and not os.path.exists(done)):
        tail = open(log_path, errors="replace").read()[-1500:]
        raise RuntimeError(f"{label} failed (see {log_path}):\n{tail}")
    print(f"[BP] {label}: done in {time.time() - start:.0f} s", flush=True)


def run_job(job):
    for command, label, done in job["steps"]:
        run_step(command, job["folder"], label, done)
    return job["outputs"]


def to_image(picture):
    """A picture (path or PIL) as a ComfyUI IMAGE (1, H, W, 3, values 0..1)."""
    if isinstance(picture, str):
        picture = Image.open(picture)
    return torch.from_numpy(np.asarray(picture.convert("RGB"), np.float32) / 255)[None]


def grid(paths, cell=256, cols=6):
    """Pictures small in one picture (videos are left out)."""
    pics = [p for p in paths if p.lower().endswith(core.PICTURE_TYPES)][:36]
    rows = max(1, -(-len(pics) // cols))
    picture = Image.new("RGB", (min(len(pics), cols) * cell or cell, rows * cell), (40, 40, 44))
    for k, p in enumerate(pics):
        image = Image.open(p).convert("RGB")
        image.thumbnail((cell, cell))
        picture.paste(image, ((k % cols) * cell + (cell - image.width) // 2, (k // cols) * cell + (cell - image.height) // 2))
    return to_image(picture)


def file3d(path):
    """A .glb as ComfyUI's 3D file type, for the Preview 3D node."""
    from comfy_api.latest import Types
    return Types.File3D(path)
