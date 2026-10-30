"""Make the picture for inpainting from a text prompt: picture the model, paint the prompt into the area.

TRELLIS.2 only understands pictures, and a picture always stands for the whole model. So to add or
change something, the model's picture has to show the change first. This script makes that picture:
    1. render the model from the given direction (the Blender view)
    2. find the area to change in that picture (the selection and the space boxes, projected)
    3. let Qwen-Image-Edit-2511 (through ComfyUI) paint the prompt into that area, optionally after a
       reference picture of the thing to add
    4. keep the render outside the area, so only the area changes
The result is the picture inpaint.py then uses.

Usage (TRELLIS.2 environment; ComfyUI is started if it isn't running):
    source activate.sh
    python scripts/regenerate/edit_picture.py outputs/tavern_1536_seed42.glb points.npy out.png \\
        --prompt "a stone statue of a dancing woman" --yaw 30 --pitch 15 [--reference statue.jpg]
"""
import _paths  # noqa: F401  (the scripts of the other folders)
import argparse
import io
import json
import os
import subprocess
import time
import urllib.parse
import urllib.request
import uuid

import cv2
import numpy as np
import nvdiffrast.torch as dr
import torch
from PIL import Image

from project_texture import to_camera, to_pixels
from render_views import load, render_view

COMFYUI = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")
COMFYUI_DIR = os.environ.get("COMFYUI_DIR", os.path.expanduser("~/ai-models/tools/ComfyUI"))
MODEL_PATHS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "comfyui_paths.yaml")  # local_ai_models/comfyui
SIZE = 1024
BACKGROUND = 0.75  # light grey, like a product shot
FRAME = (255, 0, 0)

PROMPT = ("In image 1, add {prompt} inside the red rectangle{reference}. Remove the red rectangle. "
          "Keep everything else in image 1 exactly as it is: same object, same camera angle, same lighting, "
          "same plain grey background. The new part fits the scale, material and style of the object.")


# ---------------------------------------------------------------- picture and area

def view_from_direction(direction):
    """Yaw and pitch (degrees) of a camera looking at the model from `direction` (glTF axes,
    pointing from the model towards the camera), in render_views' convention."""
    d = np.asarray(direction, float)
    d /= np.linalg.norm(d)
    return float(np.degrees(np.arctan2(-d[0], d[2]))), float(np.degrees(np.arcsin(np.clip(d[1], -1, 1))))


def area_mask(points, cam, size, grow=0.02):
    """Picture pixels covered by the points (model coordinates), closed and grown a little."""
    p = torch.tensor(np.asarray(points), dtype=torch.float32, device=cam["center"].device)
    u, v, depth = to_pixels(to_camera(p, cam), cam, size, size)
    mask = np.zeros((size, size), np.uint8)
    keep = (depth > 0).cpu().numpy()
    u, v = u.cpu().numpy()[keep].astype(int), v.cpu().numpy()[keep].astype(int)
    inside = (u >= 0) & (u < size) & (v >= 0) & (v < size)
    mask[v[inside], u[inside]] = 1
    k = max(3, int(grow * size)) | 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    return cv2.dilate(mask, np.ones((k, k), np.uint8)) > 0


# ---------------------------------------------------------------- ComfyUI

def comfy(path, data=None, raw=None, headers=None, timeout=30):
    body = raw if raw is not None else (json.dumps(data).encode() if data is not None else None)
    request = urllib.request.Request(COMFYUI + path, data=body, headers=headers or {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content = response.read()
    return json.loads(content) if content[:1] in (b"{", b"[") else content


def start_comfyui(report=print):
    """Starts ComfyUI if it isn't running; returns the process if this function started it."""
    try:
        comfy("/system_stats", timeout=3)
        return None
    except OSError:
        pass
    report("Starting ComfyUI...")
    log = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "comfyui.log"), "w")
    process = subprocess.Popen([os.path.join(COMFYUI_DIR, ".venv", "bin", "python"), "main.py", "--listen", "127.0.0.1",
                                "--port", COMFYUI.rsplit(":", 1)[1], "--extra-model-paths-config", MODEL_PATHS],
                               cwd=COMFYUI_DIR, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(120):
        time.sleep(2)
        try:
            comfy("/system_stats", timeout=3)
            return process
        except OSError:
            if process.poll() is not None:
                raise RuntimeError("ComfyUI didn't start, see outputs/comfyui.log")
    raise RuntimeError("ComfyUI didn't answer within 4 minutes")


def upload(image, name):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    boundary = uuid.uuid4().hex
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
            f"Content-Type: image/png\r\n\r\n").encode() + buffer.getvalue() + \
           (f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n--{boundary}--\r\n").encode()
    return comfy("/upload/image", raw=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})["name"]


def qwen_edit_workflow(image_name, prompt, seed, reference_name=None):
    """The Qwen-Image-Edit-2511 chain of the 'Outfit swap - pro compare' workflow, in ComfyUI's API format."""
    w = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_edit_2511_fp8mixed.safetensors", "weight_dtype": "default"}},
        "shift": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["unet", 0], "shift": 3.1}},
        "norm": {"class_type": "CFGNorm", "inputs": {"model": ["shift", 0], "strength": 1.0}},
        "lora": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["norm", 0], "lora_name": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors", "strength_model": 1.0}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "image": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "scaled": {"class_type": "FluxKontextImageScale", "inputs": {"image": ["image", 0]}},
        "positive": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"clip": ["clip", 0], "prompt": prompt, "vae": ["vae", 0], "image1": ["scaled", 0]}},
        "negative": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"clip": ["clip", 0], "prompt": "", "vae": ["vae", 0], "image1": ["scaled", 0]}},
        "pos_ref": {"class_type": "FluxKontextMultiReferenceLatentMethod", "inputs": {"conditioning": ["positive", 0], "reference_latents_method": "index_timestep_zero"}},
        "neg_ref": {"class_type": "FluxKontextMultiReferenceLatentMethod", "inputs": {"conditioning": ["negative", 0], "reference_latents_method": "index_timestep_zero"}},
        "latent": {"class_type": "VAEEncode", "inputs": {"pixels": ["scaled", 0], "vae": ["vae", 0]}},
        "sample": {"class_type": "KSampler", "inputs": {"model": ["lora", 0], "seed": seed, "steps": 4, "cfg": 1.0, "sampler_name": "euler",
                                                        "scheduler": "simple", "positive": ["pos_ref", 0], "negative": ["neg_ref", 0],
                                                        "latent_image": ["latent", 0], "denoise": 1.0}},
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": "inpaint3d/edit"}},
    }
    if reference_name:
        w["reference"] = {"class_type": "LoadImage", "inputs": {"image": reference_name}}
        w["ref_scaled"] = {"class_type": "ImageScaleToTotalPixels", "inputs": {"image": ["reference", 0], "upscale_method": "lanczos", "megapixels": 1.0, "resolution_steps": 1}}
        for node in ("positive", "negative"):
            w[node]["inputs"]["image2"] = ["ref_scaled", 0]
    return w


def comfy_edit(image, prompt, seed, reference=None, report=print):
    """Runs the Qwen edit in ComfyUI and frees ComfyUI's GPU memory afterwards (TRELLIS.2 needs it)."""
    started = start_comfyui(report)
    try:
        report("Painting the prompt into the picture (Qwen-Image-Edit, first time ~1 min to load)...")
        name = upload(image, "inpaint3d_view.png")
        ref = upload(reference.convert("RGB"), "inpaint3d_reference.png") if reference is not None else None
        text = PROMPT.format(prompt=prompt, reference=", exactly like the one shown in image 2" if ref else "")
        job = comfy("/prompt", {"prompt": qwen_edit_workflow(name, text, seed, ref), "client_id": "inpaint3d"})
        if job.get("node_errors"):
            raise RuntimeError(f"ComfyUI: {job['node_errors']}")
        while True:
            time.sleep(2)
            history = comfy(f"/history/{job['prompt_id']}")
            if job["prompt_id"] in history:
                entry = history[job["prompt_id"]]
                if entry.get("status", {}).get("status_str") == "error":
                    raise RuntimeError(f"ComfyUI failed: {entry['status'].get('messages', '')}"[:500])
                files = [i for out in entry["outputs"].values() for i in out.get("images", [])]
                if files:
                    break
        f = files[0]
        query = urllib.parse.urlencode({"filename": f["filename"], "subfolder": f["subfolder"], "type": f["type"]})
        return Image.open(io.BytesIO(comfy(f"/view?{query}", timeout=60))).convert("RGB")
    finally:
        if started is not None:
            started.terminate()  # we started it: stop it, so all of its GPU memory is free again
            started.wait(timeout=60)
        else:
            try:
                comfy("/free", {"unload_models": True, "free_memory": True})
            except OSError:
                pass


# ---------------------------------------------------------------- all together

def make_picture(model_path, points, out, prompt, yaw, pitch, seed=1, reference_path=None, report=print):
    """Writes the edited picture to `out`, a before/after preview, the render before the edit
    (<out>_before.png) and the 2D mask of the edit (<out>_mask.png) next to it; returns `out`."""
    report("Picturing the model from the Blender view...")
    ctx = dr.RasterizeCudaContext()
    meshes = load(model_path)
    picture, _, cam = render_view(ctx, meshes, yaw, pitch, SIZE, background=BACKGROUND)
    mask = area_mask(points, cam, SIZE)
    if not mask.any():
        raise ValueError("the selected area isn't visible from this view")
    del ctx
    torch.cuda.empty_cache()

    ys, xs = np.nonzero(mask)
    pad = int(0.02 * SIZE)
    x0, y0, x1, y1 = max(0, xs.min() - pad), max(0, ys.min() - pad), min(SIZE - 1, xs.max() + pad), min(SIZE - 1, ys.max() + pad)
    marked = picture.copy()
    cv2.rectangle(marked, (int(x0), int(y0)), (int(x1), int(y1)), FRAME, max(3, SIZE // 200))
    reference = Image.open(reference_path) if reference_path else None
    edited = np.asarray(comfy_edit(Image.fromarray(marked), prompt, seed, reference, report).resize((SIZE, SIZE), Image.LANCZOS))

    # only the framed area changes: the rest stays the exact render (soft edge)
    region = np.zeros((SIZE, SIZE), np.float32)
    grow = max(4, SIZE // 100)
    region[max(0, y0 - grow):y1 + grow + 1, max(0, x0 - grow):x1 + grow + 1] = 1
    region = cv2.GaussianBlur(region, (0, 0), grow)[..., None]
    result = (edited * region + picture * (1 - region)).astype(np.uint8)
    Image.fromarray(result).save(out)
    base = os.path.splitext(out)[0]  # for the inversion method: the original's render and the 2D mask
    Image.fromarray(picture).save(base + "_before.png")
    Image.fromarray(((region[..., 0] > 0.5) * 255).astype(np.uint8)).save(base + "_mask.png")
    preview = np.concatenate([marked, result], 1)
    Image.fromarray(preview).resize((SIZE, SIZE // 2)).save(os.path.splitext(out)[0] + "_preview.jpg")
    report("Picture ready")
    return out


def main():
    parser = argparse.ArgumentParser(description="Paint a prompt into a picture of the model, for inpainting")
    parser.add_argument("model")
    parser.add_argument("points", help=".npy with points of the area (model coordinates, N x 3)")
    parser.add_argument("out", help="picture to write (.png)")
    parser.add_argument("--prompt", required=True, help="what the area should show, e.g. 'a stone statue'")
    parser.add_argument("--yaw", type=float, default=30)
    parser.add_argument("--pitch", type=float, default=15)
    parser.add_argument("--reference", help="optional picture of the thing to add")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    make_picture(args.model, np.load(args.points), args.out, args.prompt, args.yaw, args.pitch, args.seed, args.reference)


if __name__ == "__main__":
    main()
