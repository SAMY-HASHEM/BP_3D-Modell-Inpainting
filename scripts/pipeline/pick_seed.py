"""Judge for seeds: which model fits the pictures best. Measured (the outline), and seen (the local vision AI).

Every model is rendered straight from 8 sides (orthographic, render_views.py does 4; the 4 in between come from a
second run turned by 45 degrees). Every picture's outline (background AI) is compared with every render's outline,
both cropped to the object and scaled to 128 x 128, mirrored too: the picture's score is its best overlap (IoU).
A model's outline score is the mean over the pictures. Nothing is chosen by hand.

The outline can't see what is wrong inside it (a melted face, a hole, a doubled arm). So the local vision AI
(Qwen3.6-27B, Ollama, as judge.py) also looks at every model: the pictures in the top row, the model's eight renders
below, and lists its defects and gives 1 to 10. The model's total is half the outline score and half the AI's; without
an answer (Ollama not running, --no-ai) the outline alone decides.

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


QUESTION = """Top row: the artist's pictures of an object. Bottom rows: renders of ONE 3D model made from those pictures,
turned in eight steps all around.

How good is the 3D model? Look for defects first: missing or broken parts (hands, fingers, thin bands, hair), holes,
melted or doubled shapes, a face that is wrong or flat, a back that doesn't fit the front, parts that float.
Then give a score: 10 = like the pictures from every side, no defects; 1 = unusable.
Answer only with JSON, no other text:
{"defects": ["at most 4 short points, the worst first"], "score": a number from 1 to 10, "why": "a few words"}"""


def judged_by_ai(pictures, folder):
    """The local AI's answer about one model: its pictures above, its eight renders below (on grey)."""
    from judge import ask, encoded
    top = [Image.open(p).convert("RGB") for p in pictures[:4]]
    top = [p.resize((max(1, round(p.width * 420 / p.height)), 420), Image.LANCZOS) for p in top]
    names = sorted(n for n in os.listdir(folder) if n.startswith("view_") and n.endswith(".png"))[:8]
    cell = 330
    sheet = Image.new("RGB", (4 * cell, 440 + 2 * cell), (190, 190, 192))
    x = (sheet.width - sum(p.width + 10 for p in top)) // 2
    for picture in top:
        sheet.paste(picture, (max(0, x), 10))
        x += picture.width + 10
    for k, name in enumerate(names):
        view = Image.open(os.path.join(folder, name)).convert("RGBA").resize((cell, cell), Image.LANCZOS)
        sheet.paste(view, ((k % 4) * cell, 440 + (k // 4) * cell), view)
    path = os.path.join(folder, "judged.jpg")
    sheet.save(path, quality=90)
    answer, _ = ask([encoded(path, 1536)], QUESTION)
    return {"score": max(1, min(10, float(answer["score"]))), "defects": [str(d)[:120] for d in answer.get("defects", [])][:4],
            "why": str(answer.get("why", ""))[:160]}


def renders_folder(out, model):
    """Where one model's renders go: named after its own folder too. Two seeds made on this PC are both called
    base.glb, so the second one was judged on the first one's renders (found Nov 14: the same score for both)."""
    return os.path.join(out, f"{os.path.basename(os.path.dirname(os.path.abspath(model)))}_{os.path.splitext(os.path.basename(model))[0]}")


def iou(a, b):
    return float((a & b).sum() / max(1, (a | b).sum()))


def main():
    parser = argparse.ArgumentParser(description="The seed whose outline fits the pictures best")
    parser.add_argument("--pictures", nargs="+", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-ai", action="store_true", help="only the measured outline")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import generate  # noqa: F401  (TRELLIS.2's environment, before the background AI)
    from make_edit import background_remover
    remover = background_remover()
    pictures = [outline(np.asarray(remover(Image.open(p).convert("RGB")))[..., 3] > 128) for p in args.pictures]

    report = {}
    for model in args.models:
        folder = renders_folder(args.out, model)
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
    del remover
    import torch
    torch.cuda.empty_cache()  # the GPU is the vision AI's now
    for model, one in report.items():
        one["total"] = one["score"]
        if args.no_ai:
            continue
        try:
            one["ai"] = judged_by_ai(args.pictures, renders_folder(args.out, model))
            one["total"] = round(0.5 * one["score"] + 0.5 * one["ai"]["score"] / 10, 4)
            print(f"{os.path.basename(model)}: the local AI {one['ai']['score']:g}/10, {one['ai']['defects']}", flush=True)
        except Exception as e:  # noqa: BLE001 (Ollama not running: the outline decides)
            one["ai_error"] = f"{type(e).__name__}: {e}"[:200]
            print(f"{os.path.basename(model)}: the local AI didn't answer ({one['ai_error']})", flush=True)
    if any("ai" not in one for one in report.values()):  # not every model was seen: only the outline is fair
        for one in report.values():
            one["total"] = one["score"]
    best = max(report, key=lambda m: report[m]["total"])
    seen = report[best].get("ai")
    result = {"best": best, "models": report,
              "verdict": f"{os.path.basename(os.path.dirname(best)) or os.path.basename(best)}: outline {report[best]['score']:.2f}"
                         + (f", local AI {seen['score']:g}/10" if seen else ""),
              "why": (seen["defects"][:2] if seen else [])}
    json.dump(result, open(os.path.join(args.out, "pick.json"), "w"), indent=1)
    print(f"Best: {best}")


if __name__ == "__main__":
    main()
