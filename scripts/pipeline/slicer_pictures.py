"""The Slicer's pictures: an edit asked for in words, made locally like Gemini does it.

1. the picture goes on a square plain grey canvas (the model keeps the canvas: room for straight arms and legs);
2. Qwen-Image-Edit 2511 (ComfyUI, scripts/inpainting/picture_from_prompt.py) paints the instruction, always on a plain
   grey background: image models cannot make a picture transparent, asked for "no background" they paint a checkerboard;
3. BiRefNet (RMBG-2.0) cuts the figure out for real: a transparent PNG, and the same on flat #808080;
4. both brought to 2048 x 2048 (the locked size; Lanczos for now).

The instruction is written by Qwen3-VL from what you typed (slicer_worker.py): the edit model follows long, exact
sentences much better than a short wish (Lou, Nov 24: "make only the character in pose A with no background" kept her
pose and the gun and painted a checkerboard; the exact instruction gave her standing, without the gun, on grey).
"""
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "inpainting"))
sys.path.insert(0, os.path.join(PROJECT, "TRELLIS.2"))

SIZE = 2048
GREY = (128, 128, 128)  # #808080
ALWAYS = (" Plain flat medium grey background (#808080), even soft studio light, no shadows on the floor, no text, "
          "no checkerboard.")

_remover = None
_comfy = None  # the ComfyUI this worker started (kept running while you work, stopped when quiet)


def square_on_grey(image):
    image = image.convert("RGBA")
    side = max(image.size)
    canvas = Image.new("RGB", (side, side), GREY)
    canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2), image)
    return canvas


def remover():
    global _remover
    if _remover is None:
        import json
        from huggingface_hub import hf_hub_download
        from trellis2.pipelines import rembg
        os.environ.setdefault("HF_HUB_CACHE", os.path.join(PROJECT, "local_ai_models"))
        config = json.load(open(hf_hub_download("microsoft/TRELLIS.2-4B", "pipeline.json")))["args"]["rembg_model"]
        _remover = getattr(rembg, config["name"])(**config["args"])
        _remover.cuda()
    return _remover


def cut_out(image):
    """The figure alone: (transparent RGBA, the same on #808080), both SIZE x SIZE, the figure centred with a margin."""
    import torch
    with torch.no_grad():
        rgba = remover()(image.convert("RGB"))
    alpha = np.asarray(rgba.getchannel("A"))
    ys, xs = np.nonzero(alpha > 20)
    if len(xs) == 0:
        raise ValueError("nothing was left after cutting the background out")
    box = rgba.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    scale = 0.9 * SIZE / max(box.size)  # 5 % margin on every side
    box = box.resize((max(1, round(box.width * scale)), max(1, round(box.height * scale))), Image.LANCZOS)
    clear = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    clear.paste(box, ((SIZE - box.width) // 2, (SIZE - box.height) // 2), box)
    grey = Image.new("RGB", (SIZE, SIZE), GREY)
    grey.paste(clear, (0, 0), clear)
    return clear, grey


def edit(image, instruction, seed=1):
    """Qwen-Image-Edit on a square grey canvas; returns its picture (about 1024 x 1024)."""
    global _comfy
    from picture_from_prompt import comfy_edit, start_comfyui
    if _comfy is None or _comfy.poll() is not None:
        _comfy = start_comfyui(lambda *a: None) or _comfy  # ours stays running: the next edit doesn't wait for it
    text = instruction.strip()
    if "background" not in text.lower():
        text += ALWAYS
    return comfy_edit(square_on_grey(image), text, seed, report=lambda *a: None, template="{prompt}")


def stop():
    """Lets the GPU go: ComfyUI stopped, BiRefNet dropped."""
    global _comfy, _remover
    if _comfy is not None and _comfy.poll() is None:
        _comfy.terminate()
        try:
            _comfy.wait(timeout=60)
        except Exception:
            _comfy.kill()
    _comfy = None
    _remover = None
