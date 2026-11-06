"""Measured judge for seeds: which model's outline fits the pictures best.

Every model is rendered straight from 8 sides (orthographic, render_views.py does 4; the 4 in between come from a
second run turned by 45 degrees). Every picture's outline (background AI) is compared with every render's outline,
both cropped to the object and scaled to 128 x 128, mirrored too: the picture's score is its best overlap (IoU).
A model's score is the mean over the pictures. Nothing is chosen by hand.

    python scripts/pipeline/pick_seed.py --pictures a.png b.png c.png --models seed42.glb seed1.glb seed2.glb --out folder
"""
import _paths  # noqa: F401  (the scripts of the other folders)

import argparse
import json
import os
import subprocess

import numpy as np
from PIL import Image

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BLENDER = os.environ.get("BLENDER", os.path.expanduser("~/Applications/blender/blender"))
SIZE = 128


def outline(mask):
    """The object's outline, cropped to it and scaled to SIZE x SIZE with its own aspect kept (padded)."""
    ys, xs = np.nonzero(mask)
    crop = Image.fromarray((mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1] * 255).astype(np.uint8))
    crop.thumbnail((SIZE, SIZE))
    plate = Image.new("L", (SIZE, SIZE), 0)
    plate.paste(crop, ((SIZE - crop.width) // 2, (SIZE - crop.height) // 2))
    return np.asarray(plate) > 127


def iou(a, b):
    return float((a & b).sum() / max(1, (a | b).sum()))


def main():
    parser = argparse.ArgumentParser(description="The seed whose outline fits the pictures best")
    parser.add_argument("--pictures", nargs="+", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import generate  # noqa: F401  (TRELLIS.2's environment, before the background AI)
    from make_edit import background_remover
    remover = background_remover()
    pictures = [outline(np.asarray(remover(Image.open(p).convert("RGB")))[..., 3] > 128) for p in args.pictures]

    report = {}
    for model in args.models:
        folder = os.path.join(args.out, os.path.splitext(os.path.basename(model))[0])
        renders = []
        if not os.path.isfile(os.path.join(folder, "view_3.png")):
            subprocess.run([BLENDER, "-b", "--python", os.path.join(PROJECT, "scripts/tools/render_views.py"), "--",
                            os.path.abspath(model), folder, "--size", "512", "--samples", "4", "--turn", "45"],
                           check=True, stdout=subprocess.DEVNULL)
        for name in sorted(os.listdir(folder)):
            if name.startswith("view_") and name.endswith(".png"):
                mask = np.asarray(Image.open(os.path.join(folder, name)).convert("RGBA"))[..., 3] > 128
                renders += [outline(mask), outline(mask[:, ::-1])]
        scores = [max(iou(p, r) for r in renders) for p in pictures]
        report[model] = {"score": round(float(np.mean(scores)), 4), "per_picture": [round(s, 4) for s in scores]}
        print(f"{os.path.basename(model)}: {report[model]['score']:.3f}  {report[model]['per_picture']}", flush=True)
    best = max(report, key=lambda m: report[m]["score"])
    json.dump({"best": best, "models": report}, open(os.path.join(args.out, "pick.json"), "w"), indent=1)
    print(f"Best: {best}")


if __name__ == "__main__":
    main()
