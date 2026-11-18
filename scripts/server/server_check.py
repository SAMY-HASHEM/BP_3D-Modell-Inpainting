"""What this computer has for the pipeline: asked through the cloud (cloud.py check), so nobody has to log in on the
uni server to find out why a step can't run there.

A flow that runs completely on the server (cloud.py flow, the web page's "everything on the uni server") needs more
than TRELLIS.2: Blender for the pictures of a model, a vision AI for the data sheet and the judges, the picture AI
for a part from words, the mask AI and the skeleton model. This lists each of them: there or not, and what to do.

    python scripts/server/server_check.py --out some_folder        # prints the list, writes check.json and check.zip
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run(*command, timeout=60):
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        return (done.stdout + done.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"({type(e).__name__})"


def reachable(address, timeout=8):
    try:
        urllib.request.urlopen(urllib.request.Request(address, method="HEAD"), timeout=timeout)
        return True
    except urllib.error.HTTPError:
        return True  # it answered
    except Exception:  # noqa: BLE001
        return False


def blender():
    for path in (os.environ.get("BLENDER"), os.path.expanduser("~/Applications/blender/blender"), os.path.join(PROJECT, "blender", "blender"),
                 shutil.which("blender")):
        if path and os.path.isfile(path):
            return path, run(path, "--version").splitlines()[0][:60]
    return None, ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    found = {"code": run("git", "-C", PROJECT, "log", "-1", "--format=%h %s")[:100], "python": sys.version.split()[0]}
    found["gpus"] = run("nvidia-smi", "--query-gpu=index,name,memory.used,memory.total", "--format=csv,noheader").splitlines()
    found["disk_free_gb"] = round(shutil.disk_usage(PROJECT).free / 1e9)
    try:
        import torch
        found["torch"] = f"{torch.__version__}, CUDA {'yes' if torch.cuda.is_available() else 'NO'}"
    except ImportError:
        found["torch"] = None
    parts = []  # (name, there, what it is for, what to do when it is missing)

    def part(name, there, used_for, missing, detail=""):
        parts.append({"name": name, "there": bool(there), "for": used_for, "do": "" if there else missing, "detail": str(detail)[:200]})

    try:
        import o_voxel  # noqa: F401
        trellis = True
    except ImportError:
        trellis = False
    part("TRELLIS.2 environment", trellis, "the base model and every inpainting", "run ./install.sh, then source activate.sh")
    path, version = blender()
    part("Blender", path, "the pictures of a model: the edit, the judges, placing a part",
         "bash install_server_extras.sh blender   (a portable Blender into ~/Applications/blender)", f"{path} {version}")
    ollama = shutil.which("ollama") or (os.path.expanduser("~/Applications/ollama/bin/ollama") if os.path.isfile(os.path.expanduser("~/Applications/ollama/bin/ollama")) else None)
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    try:
        models = [m["name"] for m in json.load(urllib.request.urlopen(host + "/api/tags", timeout=5)).get("models", [])]
    except Exception:  # noqa: BLE001
        models = None
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pipeline"))
    from judge import MODEL
    part(f"Ollama with {MODEL}", models is not None and any(m.startswith(MODEL) for m in models),
         "the vision AI: the data sheet's sides, the judges' look, understanding a request",
         "bash install_server_extras.sh ollama   (without it the measured numbers decide alone, and all pictures count as front views)",
         f"program: {ollama}, answering: {models}")
    comfy = os.environ.get("COMFYUI_DIR", os.path.expanduser("~/ai-models/tools/ComfyUI"))
    qwen = os.path.join(PROJECT, "local_ai_models", "comfyui")
    part("ComfyUI with Qwen-Image-Edit", os.path.isfile(os.path.join(comfy, "main.py")) and os.path.isdir(qwen),
         "the picture AI: a part from words, a part placed where words say",
         "install ComfyUI and run ./download_models.sh qwen (about 30 GB); without it a part needs its own pictures and goes on the top of the object",
         f"ComfyUI: {comfy} {'there' if os.path.isfile(os.path.join(comfy, 'main.py')) else 'missing'}; models: {'there' if os.path.isdir(qwen) else 'missing'}")
    part("mask AI environment", os.path.isfile(os.path.join(PROJECT, "conda/envs/mask-ai/bin/python")),
         "finding a part by its name (place_part.py --old)", "./install_mask.sh")
    part("skeleton model (UniRig)", os.path.isfile(os.path.join(PROJECT, "conda/envs/unirig/bin/python"))
         and os.path.isdir(os.path.join(PROJECT, "local_ai_models/rigging/UniRig")),
         "a model's bones (scripts/tools/rig.sh)", "bash install_server_extras.sh unirig")
    part("ffmpeg", shutil.which("ffmpeg"), "frames of a video", "conda install -c conda-forge ffmpeg (in the TRELLIS.2 environment)")
    found["internet"] = {name: reachable(address) for name, address in (
        ("github.com", "https://github.com"), ("huggingface.co", "https://huggingface.co"), ("blender.org", "https://download.blender.org"),
        ("ollama.com", "https://ollama.com"), ("pypi.org", "https://pypi.org"))}
    found["parts"] = parts
    json.dump(found, open(os.path.join(args.out, "check.json"), "w"), indent=1)
    print(f"code {found['code']} | Python {found['python']} | torch {found['torch']} | {found['disk_free_gb']} GB free")
    for gpu in found["gpus"]:
        print("  GPU " + gpu)
    for p in parts:
        print(f"  {'yes' if p['there'] else 'NO '}  {p['name']}: {p['for']}" + (f"\n        -> {p['do']}" if p["do"] else ""))
    print("  reachable from here: " + ", ".join(f"{k} {'yes' if v else 'no'}" for k, v in found["internet"].items()))
    with zipfile.ZipFile(os.path.join(args.out, "check.zip"), "w") as z:
        z.write(os.path.join(args.out, "check.json"), "check.json")
    print(f"Result: {os.path.join(args.out, 'check.zip')}", flush=True)


if __name__ == "__main__":
    main()
