"""Node 01 of the inpainting flow: the data sheet of one input (pictures 1..n, videos, or both).

An input is either the OBJECT (what TRELLIS.2 builds first, e.g. the girl) or the PART that is inpainted onto it
later (e.g. the crown). TRELLIS.2 reads one picture as one object, so the sheet turns whatever comes in into
single views and says what each one is good for.

  1. every picture   a sheet with several views side by side is cut into its views (measured: separate blobs of
                     object pixels, sheet_views.py); every video gives frames a part of a turn apart
  2. every view      measured: size, sharpness, grey sculpt or colour, cut off at which border, how symmetric
  3. every view      the local vision AI (Qwen3.6-27B, Ollama) says: what it shows, which side (front,
                     three-quarter, side, back), what is at its very top
  4. rules           the lists the next nodes use:
                       base   the views a model is generated from: the most frontal one first, then one per
                              other side (three-quarter, side, back), the sharpest of each
                       sides  a part's four sides in turning order (from a video: a quarter turn apart)

Writes into --out: sheet.json, views/*.png, contact.jpg (all views with their tags, to check at a glance).

    python scripts/pipeline/data_sheet.py pictures/girl.jpg --role object --out outputs/pipeline/girl/01_sheet
    python scripts/pipeline/data_sheet.py pictures/crown.mp4 --role part --out outputs/pipeline/girl/01_sheet_part
"""
import _paths  # noqa: F401  (the scripts of the other folders)

import argparse
import json
import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw

PICTURE_TYPES = (".png", ".jpg", ".jpeg", ".webp")
VIDEO_TYPES = (".mp4", ".mov", ".webm", ".avi", ".mkv", ".m4v")
SIDES = ("front", "three-quarter", "side", "back", "top", "unknown")
ORDER = ("front", "three-quarter", "side", "back")  # the order of the base views after the main one

QUESTION = """This picture shows one view of one object, cut out of a larger picture or a video.
Answer only with JSON, no other text:
{"what": "what the object is, at most 8 words",
 "side": "front" (the object faces the camera) or "three-quarter" (turned about 45 degrees) or "side" (turned
         90 degrees, a profile) or "back" (seen from behind) or "top" or "unknown",
 "top": "what is at the very top of the object, at most 5 words"}"""


def frame_count(video):
    answer = subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0", "-show_entries",
                             "stream=nb_read_packets", "-of", "json", video], capture_output=True, text=True, check=True)
    return int(json.loads(answer.stdout)["streams"][0]["nb_read_packets"])


def video_views(video, folder, name, count):
    """`count` frames evenly over the video (it shows one full turn), as pictures; the first is frame 0."""
    frames = frame_count(video)
    paths = []
    for k in range(count):
        paths.append(os.path.join(folder, f"{name}_turn{k * 360 // count:03d}.png"))
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", video, "-vf", f"select=eq(n\\,{k * frames // count})",
                        "-frames:v", "1", paths[-1]], check=True)
    return paths


def measure(path, remover):
    """What can be measured about a view, on the object's pixels (background removed as TRELLIS.2 does it)."""
    import cv2
    image = Image.open(path).convert("RGB")
    alpha = np.asarray(remover(image.copy()))[..., 3] > 128  # a copy: the background AI writes into its picture
    rgb = np.asarray(image, np.float32) / 255
    on = rgb[alpha] if alpha.sum() > 200 else rgb.reshape(-1, 3)
    saturation = float((on.max(1) - on.min(1)).mean())  # how far from grey, absolute: dark folds count little
    grey = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    sharp = float(cv2.Laplacian(grey, cv2.CV_64F)[alpha].var()) if alpha.any() else 0.0
    edges = {"top": alpha[0], "bottom": alpha[-1], "left": alpha[:, 0], "right": alpha[:, -1]}
    ys, xs = np.nonzero(alpha)
    box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if len(xs) else [0, 0, *image.size]
    crop = alpha[box[1]:box[3], box[0]:box[2]]
    mirrored = float((crop & crop[:, ::-1]).sum() / max(1, (crop | crop[:, ::-1]).sum()))  # 1 = the same left and right
    return {"size": list(image.size), "object_box": box, "colour": "colour" if saturation > 0.12 else "grey",
            "saturation": round(saturation, 3), "sharpness": round(sharp, 1), "symmetry": round(mirrored, 3),
            "cut_off": [side for side, row in edges.items() if row.mean() > 0.02]}


def choose_lists(views, role):
    """The rules: which views the next nodes take. Nothing here knows what the pictures show."""
    stills = [v for v in views if v["from_video"] is None]
    turning = [v for v in views if v["from_video"] is not None]
    lists = {"base": [], "sides": [], "why": []}
    if turning:  # a video of one full turn: its frames are in turning order, a quarter turn apart
        quarter = [v for v in turning if v["turn"] % 90 == 0]
        lists["sides"] = [v["name"] for v in quarter]
        lists["why"].append("sides: the video's frames at 0, 90, 180, 270 degrees of its turn")
    by_side = {}
    for view in stills:
        by_side.setdefault(view["side"], []).append(view)
    # the main view: the most symmetric of the front views; without a front view the most frontal side there is
    fronts = next((by_side[side] for side in ORDER if by_side.get(side)), stills[:1])
    if fronts:
        main = max(fronts, key=lambda v: (v["symmetry"], v["sharpness"]))
        lists["base"].append(main["name"])
        lists["why"].append(f"base main view: {main['name']} ({main['side']}, symmetry {main['symmetry']})")
        for side in ORDER[1:]:
            others = [v for v in by_side.get(side, []) if v["name"] not in lists["base"]]
            if others and side != main["side"]:
                best = max(others, key=lambda v: v["sharpness"])
                lists["base"].append(best["name"])
                lists["why"].append(f"base {side}: {best['name']} (the sharpest of {len(by_side[side])})")
    elif turning:  # only a video: the four quarter-turn frames build the model
        lists["base"] = list(lists["sides"])
        lists["why"].append("base: the video's four sides (no single pictures)")
    if role == "part" and not lists["sides"]:  # single pictures of the part: front first
        lists["sides"] = lists["base"][:1] + [v["name"] for v in stills if v["name"] not in lists["base"][:1]][:3]
        lists["why"].append("sides: the part's pictures as given, the front first")
    return lists


def contact(views, lists, path, cell=300):
    columns = min(len(views), 6)
    rows = (len(views) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell, rows * (cell + 58)), (40, 40, 44))
    draw = ImageDraw.Draw(sheet)
    for k, view in enumerate(views):
        x, y = (k % columns) * cell, (k // columns) * (cell + 58)
        image = Image.open(view["path"]).convert("RGB")
        image.thumbnail((cell - 8, cell - 8))
        sheet.paste(image, (x + (cell - image.width) // 2, y + 56 + (cell - image.height) // 2))
        used = [name for name in ("base", "sides") if view["name"] in lists[name]]
        place = "".join(f"  {name} {lists[name].index(view['name']) + 1}" for name in used) or "  not used"
        draw.text((x + 6, y + 4), f"{view['name']}: {view['side']}, {view['colour']}", fill=(255, 255, 255))
        draw.text((x + 6, y + 20), f"{view['what']}"[:46], fill=(200, 200, 200))
        draw.text((x + 6, y + 36), place.strip(), fill=(120, 220, 140) if used else (230, 120, 110))
    sheet.save(path, quality=90)


def main():
    parser = argparse.ArgumentParser(description="The data sheet of one input: pictures, videos, or both")
    parser.add_argument("files", nargs="+", help="pictures and videos (a folder: everything in it)")
    parser.add_argument("--role", choices=["object", "part"], default="object")
    parser.add_argument("--out", required=True)
    parser.add_argument("--frames", type=int, default=8, help="frames taken from a video's turn (a multiple of 4)")
    parser.add_argument("--no-ai", action="store_true", help="only the measured facts (every single picture counts as front)")
    args = parser.parse_args()
    views_dir = os.path.join(args.out, "views")
    os.makedirs(views_dir, exist_ok=True)
    files = []
    for path in args.files:
        files += sorted(os.path.join(path, f) for f in os.listdir(path)) if os.path.isdir(path) else [path]
    files = [f for f in files if f.lower().endswith(PICTURE_TYPES + VIDEO_TYPES)]
    if not files:
        raise SystemExit("no pictures or videos given")

    import types
    import sheet_views
    from make_edit import background_remover
    remover = background_remover()
    pipeline = types.SimpleNamespace(low_vram=False, rembg_model=remover, device="cuda")
    views = []
    for number, path in enumerate(files):
        name = f"in{number}"
        if path.lower().endswith(VIDEO_TYPES):
            for k, frame in enumerate(video_views(path, views_dir, name, args.frames)):
                views.append({"name": os.path.basename(frame)[:-4], "path": frame, "from": path, "from_video": path,
                              "turn": k * 360 // args.frames})
            continue
        cut = sheet_views.split(pipeline, path, views_dir, name=f"{name}_view")
        if not cut:  # one view: the picture as it is
            cut = [os.path.join(views_dir, f"{name}_view1.png")]
            Image.open(path).convert("RGB").save(cut[0])
        for frame in cut:
            views.append({"name": os.path.basename(frame)[:-4], "path": frame, "from": path, "from_video": None, "turn": None})
    for view in views:
        view.update(measure(view["path"], remover))
    del remover, pipeline
    import torch
    torch.cuda.empty_cache()  # the GPU is the vision AI's now

    if args.no_ai:
        for view in views:
            view.update(what="", side="front" if view["from_video"] is None else "unknown", top="")
    else:
        from judge import MODEL, ask, encoded
        for view in views:
            if view["from_video"] is not None and view["turn"] != 0:  # a video: its turn says the side
                view.update(what=views_first(views, view)["what"], top="", ai=None,
                            side={90: "side", 180: "back", 270: "side"}.get(view["turn"], "three-quarter"))
                continue
            try:
                answer, raw = ask([encoded(view["path"], 768)], QUESTION)
            except Exception as e:  # noqa: BLE001 (Ollama not running: the sheet still has what was measured)
                answer, raw = {}, f"{type(e).__name__}: {e}"
            side = answer.get("side") if answer.get("side") in SIDES else "unknown"
            if view["from_video"] is not None:
                side = "front"  # a turning video starts at the front
            view.update(what=str(answer.get("what", ""))[:80], side=side, top=str(answer.get("top", ""))[:60], ai=raw[:400])
            print(f"  {view['name']}: {view['side']}, {view['colour']}, {view['what']}", flush=True)
    lists = choose_lists(views, args.role)
    what = next((v["what"] for v in views if v["name"] in (lists["base"] + lists["sides"])[:1] and v["what"]), "")
    sheet = {"role": args.role, "what": what, "files": files, "ai": None if args.no_ai else MODEL, "views": views,
             "lists": lists}
    contact(views, lists, os.path.join(args.out, "contact.jpg"))
    json.dump(sheet, open(os.path.join(args.out, "sheet.json"), "w"), indent=1)
    print("\n".join(lists["why"]), flush=True)
    print(f"Result: {os.path.join(args.out, 'sheet.json')}", flush=True)


def views_first(views, view):
    """The first frame of the same video (the one the AI looked at)."""
    return next(v for v in views if v["from_video"] == view["from_video"] and v["turn"] == 0)


if __name__ == "__main__":
    main()
