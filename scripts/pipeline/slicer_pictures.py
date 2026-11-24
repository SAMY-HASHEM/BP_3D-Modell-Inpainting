"""The Slicer's pictures: an edit asked for in words, made locally like Gemini does it.

1. the picture goes on a square plain grey canvas (the model keeps the canvas: room for straight arms and legs);
2. Qwen-Image-Edit 2511 (ComfyUI, scripts/inpainting/picture_from_prompt.py) paints the instruction, always on a plain
   grey background: image models cannot make a picture transparent, asked for "no background" they paint a checkerboard;
3. BiRefNet (RMBG-2.0) cuts the figure out for real: a transparent PNG, and the same on flat #808080;
4. both brought to 2048 x 2048 (the locked size) by Real-ESRGAN x4 (upscale).

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


def find(image, word, boxes=None):
    """Where `word` is in the picture (Grounding DINO + SAM 2.1, find_part.py in its own environment): a yes/no
    array, or None if it isn't found (or the finder isn't there)."""
    import json
    import subprocess
    import tempfile
    if not (word or boxes) or not os.path.isfile(MASK_PYTHON):
        return None
    with tempfile.TemporaryDirectory() as folder:
        image.convert("RGB").save(os.path.join(folder, "picture.png"))
        json.dump({"word": word or "", "views": [{"picture": "picture.png", "clicks": [],
                                                   **({"boxes": [list(map(float, b)) for b in boxes]} if boxes else {})}]},
                  open(os.path.join(folder, "views.json"), "w"))
        run = subprocess.run([MASK_PYTHON, FIND_PART, folder], capture_output=True, text=True)
        masks = os.path.join(folder, "masks.npz")
        if run.returncode != 0 or not os.path.isfile(masks):
            return None
        found = np.load(masks)
        return found["picture"].astype(bool) if "picture" in found.files and found["picture"].any() else None


def cut_out(image, only=None, keep_frame=False):
    """The figure alone: (transparent RGBA, the same on #808080), both SIZE x SIZE, the figure centred with a margin.
    keep_frame: not centred, where it is in the picture (the layers made from the A-pose keep its frame, so the base
    body and every piece line up with it, Nov 24).
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
    if keep_frame:
        clear = rgba.resize((SIZE, SIZE), Image.LANCZOS)
        grey = Image.new("RGB", (SIZE, SIZE), GREY)
        grey.paste(clear, (0, 0), clear)
        return clear, grey
    box = rgba.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    scale = 0.9 * SIZE / max(box.size)  # 5 % margin on every side
    box = box.resize((max(1, round(box.width * scale)), max(1, round(box.height * scale))), Image.LANCZOS)
    clear = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    clear.paste(box, ((SIZE - box.width) // 2, (SIZE - box.height) // 2), box)
    grey = Image.new("RGB", (SIZE, SIZE), GREY)
    grey.paste(clear, (0, 0), clear)
    return clear, grey


_upscaler = None
UPSCALER = os.path.join(PROJECT, "local_ai_models", "upscale", "RealESRGAN_x4plus.pth")


def upscale(image, size=SIZE):
    """Real-ESRGAN x4 (spandrel; the weights MV-Adapter uses too), in tiles, then down to `size` on its long side:
    the edit model paints about 1024, the locked size is 2048 (the user, Nov 24: "the resolution and quality we
    discussed"); plain resizing only blew the blur up."""
    global _upscaler
    import torch
    if not os.path.isfile(UPSCALER):
        return image.resize((size, round(size * image.height / image.width)), Image.LANCZOS)
    if _upscaler is None:
        from spandrel import ModelLoader
        _upscaler = ModelLoader().load_from_file(UPSCALER).cuda().eval()
    x = torch.from_numpy(np.asarray(image.convert("RGB"))).permute(2, 0, 1).float().div(255)[None].cuda()
    _, _, h, w = x.shape
    tile, pad, k = 256, 16, 4
    out = torch.zeros(1, 3, h * k, w * k, device="cuda")
    with torch.no_grad():
        for y in range(0, h, tile):
            for x0 in range(0, w, tile):
                ya, yb, xa, xb = max(y - pad, 0), min(y + tile + pad, h), max(x0 - pad, 0), min(x0 + tile + pad, w)
                piece = _upscaler(x[:, :, ya:yb, xa:xb])
                oy, ox = (y - ya) * k, (x0 - xa) * k
                th, tw = min(tile, h - y) * k, min(tile, w - x0) * k
                out[:, :, y * k:y * k + th, x0 * k:x0 * k + tw] = piece[:, :, oy:oy + th, ox:ox + tw]
    big = Image.fromarray((out[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8))
    return big.resize((size, round(size * big.height / big.width)), Image.LANCZOS)


SKIN = ("Give this smooth figure realistic human skin: natural subtle skin texture, soft realistic shading, the same "
        "skin colour. Keep the shape, the outline, the face, the pose, the size and position exactly as they are. It "
        "stays a smooth featureless body: the chest smooth and rounded without nipples, the crotch smooth without "
        "genitals, no hair, no clothes, nothing sexual. Plain flat #808080 grey background.")


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
        return upscale(comfy_edit(square_on_grey(image), text, seed, report=lambda *a: None, template="{prompt}"))
    except RuntimeError as e:  # mostly out of memory right after another model left the GPU: once more, fresh
        if "execution_error" not in str(e) and "OutOfMemory" not in str(e):
            raise
        import time
        stop()
        time.sleep(10)
        _comfy = start_comfyui(lambda *a: None)
        return upscale(comfy_edit(square_on_grey(image), text, seed, report=lambda *a: None, template="{prompt}"))


def stop():
    """Lets the GPU go: ComfyUI stopped, BiRefNet dropped."""
    global _comfy, _remover
    if _comfy is not None and _comfy.poll() is None:
        _comfy.terminate()
        try:
            _comfy.wait(timeout=60)
        except Exception:
            _comfy.kill()
    global _upscaler
    _comfy = None
    _remover = None
    _upscaler = None
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


# ---------------------------------------------------------------- a piece: cut out of the A-pose, then completed

COMPLETE = ("This picture shows {piece} cut out from a character: the parts of it that were hidden behind other things "
            "are missing. Complete it into the whole {piece}, empty, nobody wearing it: fill in only the missing parts "
            "in the same style, colours and material, and keep every visible part exactly as it is. Openings, holes and "
            "cut-outs in it stay open and empty (where skin showed through, it is now empty). Nothing else in "
            "the picture: no body, no skin, no head, no other clothes. Plain flat #808080 grey background, soft even "
            "studio light, no shadow, no text.")


def _alpha_rgba(image):
    """BiRefNet's cut-out of a picture at its own size (cleaned), as RGBA."""
    import torch
    with torch.no_grad():
        rgba = remover()(image.convert("RGB"))
    rgba.putalpha(Image.fromarray(clean_alpha(np.asarray(rgba.getchannel("A")))))
    return rgba


def cut(apose, boxes):
    """What is in the boxes (from Qwen3-VL), cut out of the A-pose picture by SAM: the real pixels at their place
    (RGBA, SIZE x SIZE), or None."""
    apose = apose.convert("RGBA").resize((SIZE, SIZE), Image.LANCZOS)
    grey = Image.new("RGB", apose.size, GREY)
    grey.paste(apose, (0, 0), apose)
    mask = find(grey, None, boxes=boxes)
    if mask is None:
        return None
    mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((SIZE, SIZE))) > 127
    mask &= np.asarray(apose.getchannel("A")) > 128
    if mask.sum() < 50:
        return None
    out = Image.new("RGBA", apose.size, (0, 0, 0, 0))
    out.paste(apose, (0, 0), Image.fromarray(mask.astype(np.uint8) * 255))
    return out


def redo(layer, instruction, seed=1, match="colour", room=1.7, report=print):
    """An edit of one layer that stays where the layer is in the A-pose's frame (the edit model draws a small thing
    large and centred): a square crop around it (room: for what it may grow by), the edit, its cut-out, and back on
    the frame by a small search over size and offset that lines it up with the layer: by its colours on the layer's
    pixels (match="colour": a completed piece) or by its outline (match="shape": the same piece in grey clay).
    Returns (transparent RGBA, on #808080), SIZE x SIZE."""
    import cv2
    layer = layer.convert("RGBA").resize((SIZE, SIZE), Image.LANCZOS)
    alpha = np.asarray(layer.getchannel("A")) > 128
    ys, xs = np.nonzero(alpha)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    side = int(max(x1 - x0, y1 - y0) * room) + 8
    left, top = (x0 + x1) // 2 - side // 2, (y0 + y1) // 2 - side // 2
    crop = Image.new("RGB", (side, side), GREY)
    crop.paste(layer, (-left, -top), layer)
    raw = edit(crop, instruction, seed)
    done = _alpha_rgba(raw).resize((side, side), Image.LANCZOS)
    n = 160
    small_done = np.asarray(done.resize((n, n), Image.BILINEAR)).astype(np.float32) / 255
    want_rgba = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    want_rgba.paste(layer, (-left, -top), layer)
    want = np.asarray(want_rgba.resize((n, n), Image.BILINEAR)).astype(np.float32) / 255
    want_mask = want[..., 3] > 0.5
    best = (1e9, 1.0, 0.0, 0.0)
    for sc in np.arange(0.8, 1.201, 0.025):
        for dx in np.arange(-0.1, 0.101, 0.0125):
            for dy in np.arange(-0.1, 0.101, 0.0125):
                m = np.float32([[sc, 0, n / 2 * (1 - sc) + dx * n], [0, sc, n / 2 * (1 - sc) + dy * n]])
                w = cv2.warpAffine(small_done, m, (n, n), flags=cv2.INTER_LINEAR, borderValue=0)
                a = w[..., 3]
                if match == "shape":
                    got = a > 0.5
                    cost = 1 - (got & want_mask).sum() / max((got | want_mask).sum(), 1)
                else:
                    cover = a[want_mask]
                    diff = np.abs(w[..., :3] - want[..., :3]).mean(-1)[want_mask]
                    cost = (diff * cover).sum() / max(cover.sum(), 1e-6) + 0.6 * (1 - cover).mean()
                if cost < best[0]:
                    best = (cost, sc, dx, dy)
    _, sc, dx, dy = best
    m = np.float32([[sc, 0, side / 2 * (1 - sc) + dx * side + left], [0, sc, side / 2 * (1 - sc) + dy * side + top]])
    clear = Image.fromarray(cv2.warpAffine(np.asarray(done), m, (SIZE, SIZE), flags=cv2.INTER_LANCZOS4,
                                           borderValue=(0, 0, 0, 0)))
    on_grey = Image.new("RGB", (SIZE, SIZE), GREY)
    on_grey.paste(clear, (0, 0), clear)
    report(f"lined up ({match}): size {sc:.3f}, offset {dx:+.3f} {dy:+.3f}, cost {best[0]:.3f}")
    return clear, on_grey


def piece(apose, boxes, description, seed=1, report=print):
    """A piece of the A-pose made whole, in the A-pose's frame (Nov 24: asked to "keep only" a piece, the edit model
    kept the whole outfit; cut out first, it only has to complete what was hidden): (clear, on grey, the cut)."""
    part = cut(apose, boxes)
    if part is None:
        return None
    clear, on_grey = redo(part, COMPLETE.format(piece=description), seed, match="colour", report=report)
    return clear, on_grey, part


def skin_tone(apose):
    """The skin's colour in the A-pose picture: measured on the face first (always bare skin; "bare skin" alone found
    demon's purple leggings, Nov 24), then on bare arms; only a colour that looks like skin (red >= green >= blue,
    not grey) counts. The lit part: not shadows, highlights, eyes or lips."""
    apose = apose.convert("RGBA").resize((SIZE, SIZE), Image.LANCZOS)
    grey = Image.new("RGB", apose.size, GREY)
    grey.paste(apose, (0, 0), apose)
    solid = np.asarray(apose.getchannel("A")) > 200
    for word in ("face", "bare arm", "bare skin"):
        mask = find(grey, word)
        if mask is None:
            continue
        mask = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize((SIZE, SIZE))) > 127
        px = np.asarray(apose)[..., :3][mask & solid].astype(np.float32)
        if len(px) < 100:
            continue
        lum = px.mean(1)
        px = px[(lum > np.percentile(lum, 35)) & (lum < np.percentile(lum, 90))]
        r, g, b = np.median(px, 0)
        if r >= g >= b and r - b > 12:
            return np.array([r, g, b])
    return None


def tint(mannequin, tone):
    """The grey mannequin in the skin's colour: its light and shade kept, nothing new drawn (asked for skin, the edit
    model drew nipples whatever the words said: the colour is put on instead)."""
    a = np.asarray(mannequin.convert("RGBA")).astype(np.float32)
    light = a[..., :3].mean(-1, keepdims=True) / 255
    body = a[..., 3] > 200
    shade = light / max(float(np.median(light[body])), 1e-3)
    out = np.clip(np.asarray(tone, np.float32) * shade ** 1.15, 0, 255)
    clear = Image.fromarray(np.dstack([out, a[..., 3:]]).astype(np.uint8), "RGBA")
    on_grey = Image.new("RGB", clear.size, GREY)
    on_grey.paste(clear, (0, 0), clear)
    return clear, on_grey
