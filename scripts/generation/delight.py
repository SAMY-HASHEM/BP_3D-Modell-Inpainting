"""De-lighting with a clay render: the colours without the light and shadows baked into them (pipeline stage 3).

A grey ("clay") render with the same lights and camera as the coloured one shows only the light at each point: the
clay is one colour everywhere. So, per pixel, in linear light (not sRGB):

    true colour  ~  coloured render / (grey render / the clay's own brightness)

The clay's brightness is taken as the 90th percentile of the grey render on the object (where it is fully lit). The
light is clipped from below (deep shadow: the colour there is unknown, so it is only brightened up to 1/LIGHT_MIN), and
the result from above. Glowing parts (lamps) are lit in both and come out as they are.

Why (Nov 23): the motorcycle girl's model came out a quarter darker than her video (base colour 0.235 vs 0.317
without the grey views): the video's shadows had gone into the colours.

    python scripts/generation/delight.py colour.png grey.png out.png          # two pictures of the same view
    python scripts/generation/delight.py --video painted.mp4 grey.mp4 --frames 108 199 58 --out views/ [--names a b c]

The two must line up (the same view, the same framing; checked: the masks' overlap is printed, under 0.95 it warns).
Video mode cuts the object out (RMBG-2.0) and crops it like views_from_video.py --cutout; writes delit_<name>.png.
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

LIGHT_MIN = 0.25      # shadows are brightened at most 4 x
CLAY_PERCENTILE = 90  # the clay's own brightness: where it is fully lit


def to_linear(srgb):
    return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)


def to_srgb(linear):
    linear = np.clip(linear, 0, 1)
    return np.where(linear <= 0.0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - 0.055)


def delight(colour, grey, mask):
    """colour, grey: float RGB 0..1 (sRGB), the same size; mask: bool (the object). The de-lit colour (sRGB)."""
    c = to_linear(colour)
    g = to_linear(grey).mean(2)
    clay = np.percentile(g[mask], CLAY_PERCENTILE)
    light = np.clip(g / max(clay, 1e-6), LIGHT_MIN, None)[..., None]
    out = to_srgb(c / light)
    out[~mask] = colour[~mask]
    return out


def overlap(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))


def from_video(colour_video, grey_video, frames, out, names=None):
    import cv2
    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from views_from_video import read_frame
    project = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, os.path.join(project, "TRELLIS.2"))
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(project, "local_ai_models"))
    from trellis2.pipelines.rembg.BiRefNet import BiRefNet
    remover = BiRefNet("briaai/RMBG-2.0")
    remover.cuda()
    torch.set_grad_enabled(False)
    os.makedirs(out, exist_ok=True)
    cv, gv = cv2.VideoCapture(colour_video), cv2.VideoCapture(grey_video)
    for k, f in enumerate(frames):
        colour = read_frame(cv, f)
        grey = read_frame(gv, f).resize(colour.size, Image.LANCZOS)
        small = (1024, 1024)
        c_alpha = np.asarray(remover(colour.convert("RGB").resize(small)))[..., 3]
        ga = np.asarray(remover(grey.convert("RGB").resize(small)))[..., 3] > 127
        o = overlap(c_alpha > 127, ga)
        full_alpha = np.asarray(Image.fromarray(c_alpha).resize(colour.size, Image.LANCZOS))
        alpha = full_alpha > 127
        lit = delight(np.asarray(colour.convert("RGB"), np.float32) / 255,
                      np.asarray(grey.convert("RGB"), np.float32) / 255, alpha)
        name = names[k] if names else f"{f:04d}"
        picture = Image.fromarray(np.dstack([(lit * 255).astype(np.uint8), full_alpha]))
        ys, xs = np.nonzero(alpha)
        margin = int(0.03 * max(colour.size))  # cropped like views_from_video.py --cutout
        box = (max(0, xs.min() - margin), max(0, ys.min() - margin),
               min(colour.width, xs.max() + margin), min(colour.height, ys.max() + margin))
        picture.crop(box).save(os.path.join(out, f"delit_{name}.png"))
        before = np.asarray(colour.convert("L"), np.float32)[alpha].mean() / 255
        after = np.asarray(Image.fromarray((lit * 255).astype(np.uint8)).convert("L"), np.float32)[alpha].mean() / 255
        print(f"frame {f} ({name}): grey/colour overlap {o:.3f}{'  WARNING: not lined up' if o < 0.95 else ''}, "
              f"brightness {before:.3f} -> {after:.3f}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("pictures", nargs="*", help="colour.png grey.png out.png")
    p.add_argument("--video", nargs=2, metavar=("COLOUR", "GREY"))
    p.add_argument("--frames", type=int, nargs="+")
    p.add_argument("--names", nargs="+")
    p.add_argument("--out")
    a = p.parse_args()
    if a.video:
        from_video(a.video[0], a.video[1], a.frames, a.out, a.names)
        return
    colour, grey, out = a.pictures
    c = Image.open(colour)
    g = Image.open(grey).resize(c.size, Image.LANCZOS)
    alpha = np.asarray(c)[..., 3] > 127 if c.mode == "RGBA" else np.ones(c.size[::-1], bool)
    lit = delight(np.asarray(c.convert("RGB"), np.float32) / 255, np.asarray(g.convert("RGB"), np.float32) / 255,
                  alpha)
    result = Image.fromarray((lit * 255).astype(np.uint8))
    if c.mode == "RGBA":
        result.putalpha(c.getchannel("A"))
    result.save(out)


if __name__ == "__main__":
    main()
