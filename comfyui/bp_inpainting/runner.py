"""Helpers for the BP Inpainting nodes: run our scripts in the project's own environment, cache results,
render previews, turn pictures into ComfyUI images.

The nodes run inside ComfyUI's Python, but TRELLIS.2, the vision AI and our scripts need the project's
conda environment. So every heavy step runs as its own process: `source activate.sh && python <script>`.
ComfyUI's environment stays untouched, and the GPU memory is free again after each step.
"""
import hashlib
import json
import os
import subprocess
import time

import numpy as np
import torch
from PIL import Image, ImageDraw

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
SCRIPTS = os.path.join(PROJECT, "scripts")
OUTPUTS = os.path.join(PROJECT, "outputs", "pipeline")
BLENDER = os.environ.get("BLENDER", "blender")


def project_path(path):
    """A path as the user typed it: relative paths are relative to the project folder."""
    path = os.path.expanduser(path.strip())
    return path if os.path.isabs(path) else os.path.join(PROJECT, path)


def node_folder(name, node, settings):
    """The folder of one node's result. Its name comes from the settings and the inputs (with their
    modification times), so the same settings are never computed twice, even after a restart."""
    key = json.dumps(settings, sort_keys=True, default=str)
    digest = hashlib.sha1(key.encode()).hexdigest()[:8]
    folder = os.path.join(OUTPUTS, name, node, digest)
    os.makedirs(folder, exist_ok=True)
    json.dump(settings, open(os.path.join(folder, "settings.json"), "w"), indent=1, default=str)
    return folder


def stamp(path):
    """A file or folder as part of a cache key: the path and when it last changed."""
    try:
        return [path, round(os.path.getmtime(path), 1)]
    except OSError:
        return [path, None]


def run(command, folder, label, done=None):
    """Runs one of our scripts in the project environment, writing log.txt into `folder`. Skips the run when
    the file `done` is already there (cached). Stops the process when ComfyUI's Cancel is pressed."""
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
    with open(os.path.join(folder, "log.txt"), "w") as log:
        process = subprocess.Popen(["bash", "-c", shell], stdout=log, stderr=subprocess.STDOUT, cwd=PROJECT, env=env)
        try:
            while process.poll() is None:
                time.sleep(1)
                mm.throw_exception_if_processing_interrupted()
        except BaseException:
            process.kill()
            raise
    if process.returncode != 0 or (done and not os.path.exists(done)):
        tail = open(os.path.join(folder, "log.txt"), errors="replace").read()[-1500:]
        raise RuntimeError(f"{label} failed (see {folder}/log.txt):\n{tail}")
    print(f"[BP] {label}: done in {time.time() - start:.0f} s", flush=True)


def render(model, out, focus=None, size=384, clay=False):
    """Preview renders with Blender on the CPU (scripts/tools/render_preview.py), cached next to `out`."""
    extra = ["--focus", *[f"{v:.5f}" for v in focus]] if focus is not None else []
    extra += ["--clay"] if clay else []
    names = ["front", "left", "right", "back"] if focus is not None else ["0", "1", "2", "3"]
    paths = [f"{out}_{n}.png" for n in names]
    if not all(os.path.exists(p) for p in paths):
        subprocess.call([BLENDER, "-b", "--python", os.path.join(SCRIPTS, "tools", "render_preview.py"), "--",
                         os.path.abspath(model), os.path.abspath(out), "--size", str(size), *extra],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return [p for p in paths if os.path.exists(p)]


def sheet(rows, cell=320):
    """A comparison picture: rows of (label, [pictures]), every picture fitted into one cell."""
    rows = [(label, pics) for label, pics in rows if pics]
    width = max(len(pics) for _, pics in rows) * cell
    picture = Image.new("RGB", (width, len(rows) * (cell + 26)), (40, 40, 44))
    draw = ImageDraw.Draw(picture)
    for r, (label, pics) in enumerate(rows):
        y = r * (cell + 26)
        draw.text((6, y + 6), label, fill=(255, 255, 255))
        for c, p in enumerate(pics):
            image = Image.open(p).convert("RGB") if isinstance(p, str) else p.convert("RGB")
            image.thumbnail((cell, cell))
            picture.paste(image, (c * cell + (cell - image.width) // 2, y + 26 + (cell - image.height) // 2))
    return picture


def to_image(picture):
    """A PIL picture as a ComfyUI IMAGE (1, H, W, 3, values 0..1)."""
    return torch.from_numpy(np.asarray(picture.convert("RGB"), np.float32) / 255)[None]


def file3d(path):
    """A .glb as ComfyUI's 3D file type, for the Preview 3D node."""
    from comfy_api.latest import Types
    return Types.File3D(path)
