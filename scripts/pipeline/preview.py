"""The pictures every node shows: renders (Blender on the CPU, render_preview.py) put together into one
compare.jpg per node. Used by both node editors (ComfyUI and Blender) through nodes_core.py.

Usage (TRELLIS.2 environment, Pillow):
    python scripts/pipeline/preview.py base   <folder> <base.glb> [pictures...]
    python scripts/pipeline/preview.py refine <folder> <before.glb> <after.glb>
    python scripts/pipeline/preview.py combine <folder> <base.glb> <combined.glb> [labels...]
    python scripts/pipeline/preview.py compare <out.jpg> [--clay] <a.glb> <b.glb> ...
"""
import json
import os
import subprocess
import sys

from PIL import Image, ImageDraw

SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLENDER = os.environ.get("BLENDER", "blender")


def render(model, out, focus=None, size=384, clay=False):
    """Front, side, back, side (or, with a focus box, front, 45 degrees left / right, behind); cached."""
    extra = ["--focus", *[f"{v:.5f}" for v in focus]] if focus is not None else []
    extra += ["--clay"] if clay else []
    names = ["front", "left", "right", "back"] if focus is not None else ["0", "1", "2", "3"]
    paths = [f"{out}_{n}.png" for n in names]
    if not all(os.path.exists(p) for p in paths):
        subprocess.call([BLENDER, "-b", "--python", os.path.join(SCRIPTS, "tools", "render_preview.py"), "--",
                         os.path.abspath(model), os.path.abspath(out), "--size", str(size), *extra],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return [p for p in paths if os.path.exists(p)]


def sheet(rows, out, cell=320):
    """rows of (label, [pictures]); every picture fitted into one cell."""
    rows = [(label, pics) for label, pics in rows if pics]
    width = max(len(pics) for _, pics in rows) * cell
    picture = Image.new("RGB", (width, len(rows) * (cell + 26)), (40, 40, 44))
    draw = ImageDraw.Draw(picture)
    for r, (label, pics) in enumerate(rows):
        y = r * (cell + 26)
        draw.text((6, y + 6), label, fill=(255, 255, 255))
        for c, p in enumerate(pics):
            image = Image.open(p).convert("RGB")
            image.thumbnail((cell, cell))
            picture.paste(image, (c * cell + (cell - image.width) // 2, y + 26 + (cell - image.height) // 2))
    picture.save(out, quality=90)
    print(f"Result: {out}", flush=True)


def main():
    kind, args = sys.argv[1], sys.argv[2:]
    if kind == "base":
        folder, glb, pictures = args[0], args[1], args[2:]
        sheet([("pictures (shape only, both, colour only)", pictures),
               ("base model", render(glb, os.path.join(folder, "render"))),
               ("shape alone (clay)", render(glb, os.path.join(folder, "clay"), clay=True))],
              os.path.join(folder, "compare.jpg"))
    elif kind == "refine":
        folder, before, after = args
        work = os.path.splitext(after)[0] + "_work"
        info = json.load(open(os.path.join(work, "refine.json")))
        focus = [*info["inner"][0], *info["inner"][1]]
        pictures = [os.path.join(work, os.path.basename(p)) for p, _ in info["pictures"]]
        what = "colours only" if info.get("keep_shape") else "shape + colours"
        sheet([("pictures: " + ", ".join(role for _, role in info["pictures"]), pictures[:6]),
               ("before", render(before, os.path.join(folder, "before"), focus)),
               (f"after ({what})", render(after, os.path.join(folder, "after"), focus))],
              os.path.join(folder, "compare.jpg"))
    elif kind == "combine":
        folder, base, combined, labels = args[0], args[1], args[2], args[3:]
        sheet([("base", render(base, os.path.join(folder, "base"))),
               ("combined: " + ", ".join(labels), render(combined, os.path.join(folder, "combined")))],
              os.path.join(folder, "compare.jpg"))
    elif kind == "compare":
        out, rest = args[0], args[1:]
        clay = "--clay" in rest
        models = [m for m in rest if m != "--clay"]
        rows = [(os.path.basename(os.path.dirname(m)) + "/" + os.path.basename(m),
                 render(m, os.path.splitext(m)[0] + ("_compare_clay" if clay else "_compare"), clay=clay)) for m in models]
        sheet(rows, out, 384)
    else:
        raise SystemExit(f"unknown kind {kind}")


if __name__ == "__main__":
    main()
