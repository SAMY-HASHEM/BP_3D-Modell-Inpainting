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


def clean_alpha(alpha):
    """Only the figure: faint pixels (BiRefNet's half-sure ghosts of what the edit removed: a body around a lone
    bodysuit, Nov 24) go, and so do pieces much smaller than the main one, unless they are solid and near it."""
    from scipy import ndimage
    alpha = alpha.copy()
    alpha[alpha < 96] = 0
    labels, n = ndimage.label(alpha > 128)
    if n > 1:
        sizes = ndimage.sum(np.ones_like(labels), labels, range(1, n + 1))
        main = sizes.max()
        keep = np.zeros(n + 1, bool)
        keep[1:] = sizes >= 0.02 * main  # small specks go; real separate pieces (two boots, two gloves) stay
        solid = keep[labels] | (labels == 0)
        alpha[~solid] = 0
        alpha[(labels == 0) & ~ndimage.binary_dilation(keep[labels], iterations=6)] = 0  # soft edges only next to kept pieces
    return alpha


MASK_PYTHON = os.path.join(PROJECT, "conda", "envs", "mask-ai", "bin", "python")
FIND_PART = os.path.join(PROJECT, "scripts", "inpainting", "find_part.py")


def find(image, word):
    """Where `word` is in the picture (Grounding DINO + SAM 2.1, find_part.py in its own environment): a yes/no
    array, or None if it isn't found (or the finder isn't there)."""
    import json
    import subprocess
    import tempfile
    if not word or not os.path.isfile(MASK_PYTHON):
        return None
    with tempfile.TemporaryDirectory() as folder:
        image.convert("RGB").save(os.path.join(folder, "picture.png"))
        json.dump({"word": word, "views": [{"picture": "picture.png", "clicks": []}]},
                  open(os.path.join(folder, "views.json"), "w"))
        run = subprocess.run([MASK_PYTHON, FIND_PART, folder], capture_output=True, text=True)
        masks = os.path.join(folder, "masks.npz")
        if run.returncode != 0 or not os.path.isfile(masks):
            return None
        found = np.load(masks)
        return found["picture"].astype(bool) if "picture" in found.files and found["picture"].any() else None


def cut_out(image, only=None):
    """The figure alone: (transparent RGBA, the same on #808080), both SIZE x SIZE, the figure centred with a margin.
    only: a word ("bodysuit"); then only that thing stays (the edit model leaves other pieces floating around a
    piece asked for alone: the ears and hair strands around Lou's bodysuit, Nov 24)."""
    import torch
    with torch.no_grad():
        rgba = remover()(image.convert("RGB"))
    alpha = clean_alpha(np.asarray(rgba.getchannel("A")))
    if only:
        from scipy import ndimage
        mask = find(image, only)
        if mask is not None:
            mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(rgba.size)) > 127
            alpha = np.where(ndimage.binary_dilation(mask, iterations=4), alpha, 0).astype(np.uint8)
    rgba.putalpha(Image.fromarray(alpha))
    ys, xs = np.nonzero(alpha > 128)
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
    try:
        return comfy_edit(square_on_grey(image), text, seed, report=lambda *a: None, template="{prompt}")
    except RuntimeError as e:  # mostly out of memory right after another model left the GPU: once more, fresh
        if "execution_error" not in str(e) and "OutOfMemory" not in str(e):
            raise
        import time
        stop()
        time.sleep(10)
        _comfy = start_comfyui(lambda *a: None)
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
    import subprocess  # also a ComfyUI of ours this worker didn't start (left from before a restart): our port only
    from picture_from_prompt import COMFYUI
    port = COMFYUI.rsplit(":", 1)[1]
    found = subprocess.run(["pgrep", "-u", str(os.getuid()), "-f", f"main.py --listen 127.0.0.1 --port {port}"],
                           capture_output=True, text=True).stdout.split()
    for pid in found:
        subprocess.run(["kill", pid])
    if found:
        import time
        time.sleep(3)
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()
