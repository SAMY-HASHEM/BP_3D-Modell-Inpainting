"""The correction loop: one consistent model first, then every picture corrects the part of it that it shows.

    1 BASE     two candidates: the main view alone (generate.py), and all sides mixed (multi_image.py,
               multidiffusion). A real turnaround (front, side, back of one pose) wants the mix: no side has to be
               invented. Free renders that don't fit together doubled parts when mixed: there one view alone is
               cleaner. Both are fitted and judged (steps 2, 3); the better one is the base
    2 FIT      every other view, close-ups too: where its camera stood on that model (fit_camera). A view whose
               outline doesn't fit (another pose, another object) is left out and said
    3 JUDGE    every fitted view is cut into cells (and, if most of them are bad, taken as one whole side first:
               patching a side that is wrong as a whole cell by cell gave box-shaped pieces, Nov 15); per cell the picture's crop next to the model's render from
               the same camera:
                 measured  outline (edge cells), colour difference, how much fine detail the render has of the picture's
                 seen      the local vision AI (Qwen3.6-27B, Ollama): shape, colour, detail, 0 to 100
               a cell is bad only if BOTH are below the threshold (the outline alone kept a worse block, Nov 15)
    4 CORRECT  only that part is regenerated (refine_region.py --box3d: zoomed in, inversion), from that view's
               crop at the picture's full resolution, with the camera of step 2. The model is turned so that the
               view's camera looks at its front: TRELLIS.2 reads every picture as the front view
    5 KEEP     the local AI sees the picture's crop, the old render and the new one (asked twice, the order
               swapped): the correction stays only if both answers say the new one is closer, and the whole
               outline didn't get worse. The kept ones are put into ONE mesh (combine_parts.py): the next round's model

Every heavy step is its own process, so the GPU is free for the next one. Results in --out (default
outputs/<name>/work/correct): 00_views ... 05_model, final.glb, summary.md.

    python scripts/pipeline/correct.py pictures/robot/bee-untitled-3.jpg --name robot_bee --sides "front, side, back"
    python scripts/pipeline/correct.py "pictures/robot hand" --name robot_hand_1 --base picked.glb --main in3_view1
"""
import _paths  # noqa: F401  (the scripts of the other folders)

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
PROJECT = os.path.dirname(SCRIPTS)
MIN_FIT = 0.80      # a view judges only if its outline overlaps the model's this well
MIN_PIXELS = 3000   # a cell needs this many object pixels

QUESTION = """Picture 1 is a crop of the artist's picture of an object. Picture 2 shows the same spot of a 3D model of it,
from the same camera. Compare them, for this spot only. Lighting and background may differ; that does not count.
 - shape: are the forms the same? Missing, doubled, melted or broken parts, holes, lumps?
 - colour: are the colours and patterns in the same places? Smears, black or grey spots?
 - detail: does the model show the small things of the picture (lines, bolts, panels, edges) as sharply?
Be strict: 90 and above only if you could hardly tell them apart; one clear defect is at most 70.
Answer only with JSON, no other text:
{"shape": 0-100, "colour": 0-100, "detail": 0-100, "defects": ["at most 3 short points"]}"""

CLOSER = """Picture 1 is a crop of the artist's picture of an object. Pictures 2 and 3 show the same spot of two 3D models
of it (model A = picture 2, model B = picture 3), from the same camera. Lighting and background do not count.
Which model is closer to the artist's picture in shape, colours and small details? Look also at the edges of the
crop: a model with a seam, a step or a hole there is worse.
Answer only with JSON, no other text: {"closer": "A" or "B" or "same", "why": "a few words"}"""


# ---------------------------------------------------------------- the steps that need the GPU (own processes)

def open_view(view):
    return Image.open(os.path.join(PROJECT, view["path"])).convert("RGB")


def step_fit(task):
    """2 FIT: the camera of every view on the model -> cameras.json, fit.json, fit.jpg (picture and render mixed)."""
    import nvdiffrast.torch as dr
    from grid_judge import save_cameras
    from project_texture import fit_camera, load_meshes, object_masks, render_meshes
    _, meshes = load_meshes(task["model"])
    ctx = dr.RasterizeCudaContext()
    pictures = [open_view(v) for v in task["views"]]
    masks = object_masks(pictures)
    cams, fits, tiles = {}, {}, []
    for view, picture, mask in zip(task["views"], pictures, masks):
        cam, overlap = fit_camera(ctx, meshes, picture, mask)
        fits[view["name"]] = round(float(overlap), 3)
        ok = overlap >= MIN_FIT
        print(f"  {view['name']} ({view.get('side')}): outline overlap {overlap:.2f}" + ("" if ok else "  -> left out"), flush=True)
        if ok:
            cams[view["name"]] = cam
        render, _ = render_meshes(ctx, meshes, cam, *picture.size)
        tile = Image.blend(picture, Image.fromarray((render.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)), 0.5)
        ImageDraw.Draw(tile).text((8, 8), f"{view['name']} {overlap:.2f}" + ("" if ok else " LEFT OUT"), fill=(255, 255, 0))
        tile.thumbnail((520, 900))
        tiles.append(tile)
    save_cameras(os.path.join(task["out"], "cameras.json"), cams)
    row(tiles).save(os.path.join(task["out"], "fit.jpg"), quality=90)
    json.dump(fits, open(os.path.join(task["out"], "fit.json"), "w"), indent=1)


def row(tiles, gap=6):
    sheet = Image.new("RGB", (sum(t.width + gap for t in tiles) + gap, max(t.height for t in tiles) + 2 * gap), (40, 40, 44))
    x = gap
    for t in tiles:
        sheet.paste(t, (x, gap))
        x += t.width + gap
    return sheet


def lab(a):
    linear = np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
    m = np.array([[.4124, .3576, .1805], [.2126, .7152, .0722], [.0193, .1192, .9505]])
    xyz = linear @ m.T / np.array([.9505, 1, 1.089])
    f = np.where(xyz > .00886, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def measured(picture, render, object_mask, model_mask):
    """The three numbers that need no AI, 0 to 100, on one cell (crops of the same size):
      outline  which share of the object has no model next to it or the other way round (None: no edge in the cell)
      colour   100 - 1.5 x the difference of the mean colours (CIE Lab) where both have the object
      detail   the fine structure (Laplacian) of the render as a share of the picture's: a smooth, washed-out
               render of a spot full of lines gets a low number"""
    import cv2
    from grid_judge import measure_shape
    result = {}
    edge = 1 - (object_mask & model_mask).mean() > 0.03  # more than 3 % of the cell is not object in both
    if edge:
        result["outline"] = measure_shape(object_mask, model_mask)["outline"]
    both = cv2.erode((object_mask & model_mask).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    if both.sum() > 200:
        a = np.asarray(picture, np.float32) / 255
        b = np.asarray(render, np.float32) / 255
        result["colour"] = round(float(np.clip(100 - 1.5 * np.linalg.norm(lab(a[both]).mean(0) - lab(b[both]).mean(0)), 0, 100)))
        fine = lambda image: float(np.abs(cv2.Laplacian(cv2.GaussianBlur(np.asarray(image.convert("L"), np.float32), (0, 0), 1), cv2.CV_32F))[both].mean())
        result["detail"] = round(float(np.clip(100 * fine(render) / max(fine(picture), 1e-3), 0, 100)))
    return result


def renders_of(ctx, meshes, cam, size):
    """The textured render at the picture's size, where the model is, and which face every pixel shows."""
    from grid_judge import face_ids
    from project_texture import render_meshes
    colour, covered = render_meshes(ctx, meshes, cam, *size)
    image = Image.fromarray((colour.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
    return image, covered.cpu().numpy(), face_ids(ctx, meshes, cam, *size)


def lowest(numbers):
    numbers = [x for x in numbers.values() if isinstance(x, (int, float))]
    return min(numbers) if numbers else None


def step_judge(task):
    """3 JUDGE: every fitted view in cells, measured and seen -> judge.json, judge.jpg."""
    import nvdiffrast.torch as dr
    import torch
    import grid_judge
    from project_texture import load_meshes, object_masks
    grid_judge.BACKEND["kind"] = task["ai"] if isinstance(task["ai"], str) and task["ai"] else "ollama"
    _, meshes = load_meshes(task["model"])
    verts = torch.cat([m.verts for m in meshes])
    corners, offset = [], 0
    for m in meshes:  # every face's three corners, over all meshes
        corners.append(m.verts[m.faces])
        offset += len(m.verts)
    corners = torch.cat(corners)
    ctx = dr.RasterizeCudaContext()
    cams = grid_judge.load_cameras(task["cameras"])
    views = [v for v in task["views"] if v["name"] in cams]
    pictures = [open_view(v) for v in views]
    masks = [m > 0.5 for m in object_masks(pictures)]
    threshold, cells, tiles = task["threshold"], [], []
    from refine_region import turn_to_camera
    all_points = verts.cpu().numpy()

    def boxes(points, turn, near_half=False):
        """A region's box in the model turned to the view's camera (what refine_region.py --face-camera works in)
        and in the model's own position (to see which regions overlap). near_half: only the camera's half."""
        import trimesh
        points = points.cpu().numpy() if hasattr(points, "cpu") else points
        turned = trimesh.transform_points(points, turn)
        if near_half:  # the camera looks from +Z of the turned model
            middle = (turned[:, 2].min() + turned[:, 2].max()) / 2
            points, turned = points[turned[:, 2] >= middle], turned[turned[:, 2] >= middle]
        box = lambda p: [np.quantile(p, 0.02, axis=0).tolist(), np.quantile(p, 0.98, axis=0).tolist()]
        return box(turned), box(points)

    for view, picture, mask in zip(views, pictures, masks):
        name = view["name"]
        turn = turn_to_camera(cams[name], all_points)
        first = len(cells)
        render, covered, ids = renders_of(ctx, meshes, cams[name], picture.size)
        render.save(os.path.join(task["out"], f"{name}_render.jpg"), quality=92)
        ys, xs = np.nonzero(mask | covered)
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        side = max(x1 - x0, y1 - y0) / task["cells"]
        nx, ny = max(1, round((x1 - x0) / side)), max(1, round((y1 - y0) / side))
        tile = Image.blend(picture, render, 0.5)
        draw = ImageDraw.Draw(tile)
        for j in range(ny):
            for i in range(nx):
                cx0, cx1 = int(x0 + i * (x1 - x0) / nx), int(x0 + (i + 1) * (x1 - x0) / nx)
                cy0, cy1 = int(y0 + j * (y1 - y0) / ny), int(y0 + (j + 1) * (y1 - y0) / ny)
                if (mask | covered)[cy0:cy1, cx0:cx1].sum() < max(MIN_PIXELS, 0.08 * (cx1 - cx0) * (cy1 - cy0)):
                    continue
                pad = int(0.12 * max(cx1 - cx0, cy1 - cy0))
                crop = (max(0, cx0 - pad), max(0, cy0 - pad), min(picture.width, cx1 + pad), min(picture.height, cy1 + pad))
                area = (slice(crop[1], crop[3]), slice(crop[0], crop[2]))
                ref, mine = picture.crop(crop), render.crop(crop)
                numbers = measured(ref, mine, mask[area], covered[area])
                seen_faces = torch.unique(ids[cy0:cy1, cx0:cx1])
                seen_faces = seen_faces[seen_faces >= 0]
                if len(seen_faces) < 20:
                    continue
                box, box_model = boxes(corners[seen_faces].reshape(-1, 3), turn)
                stem = f"{name}_c{j}{i}"
                ref.save(os.path.join(task["out"], stem + "_picture.jpg"), quality=92)
                mine.save(os.path.join(task["out"], stem + "_model.jpg"), quality=92)
                seen = {}
                if task["ai"]:
                    answer = asked([ref, mine], QUESTION)
                    seen = {k: answer.get(k) for k in ("shape", "colour", "detail")}
                    seen["defects"] = answer.get("defects", [])
                low_measured, low_seen = lowest(numbers), lowest(seen)
                bad = (low_measured is not None and low_measured < threshold
                       and (low_seen < threshold if low_seen is not None else True))
                cells.append({"view": name, "cell": [j, i], "rect": [cx0, cy0, cx1, cy1], "crop": list(crop), "box": box,
                              "box_model": box_model, "measured": numbers, "seen": seen, "score": low_seen if low_seen is not None else low_measured,
                              "bad": bool(bad)})
                print(f"  {stem}: measured {numbers}, seen {({k: v for k, v in seen.items() if k != 'defects'})}"
                      + ("  BAD" if bad else ""), flush=True)
                colour = (230, 60, 50) if bad else (60, 200, 80)
                draw.rectangle((cx0, cy0, cx1, cy1), outline=colour, width=4)
                draw.text((cx0 + 8, cy0 + 6), f"{low_measured} / {low_seen}", fill=colour)
        mine = cells[first:]
        if len(mine) >= 3 and sum(c["bad"] for c in mine) >= 0.6 * len(mine):  # the side is wrong as a whole
            pad = int(0.04 * max(x1 - x0, y1 - y0))
            box, box_model = boxes(all_points, turn, near_half=True)
            numbers = [c["score"] for c in mine if c["score"] is not None]
            cells.append({"view": name, "cell": "side", "rect": [int(x0), int(y0), int(x1), int(y1)],
                          "crop": [max(0, int(x0) - pad), max(0, int(y0) - pad), min(picture.width, int(x1) + pad),
                                   min(picture.height, int(y1) + pad)], "box": box, "box_model": box_model, "measured": {},
                          "seen": {}, "score": round(float(np.mean(numbers)), 1), "bad": True, "whole": True})
            print(f"  {name}: {sum(c['bad'] for c in mine)} of {len(mine)} cells bad -> the whole side first", flush=True)
            draw.rectangle((x0 - 6, y0 - 6, x1 + 6, y1 + 6), outline=(255, 200, 0), width=6)
        tile.thumbnail((640, 1000))
        tiles.append(tile)
    grid_judge.unload()
    json.dump({"model": task["model"], "threshold": threshold, "ai": task["ai"], "cells": cells},
              open(os.path.join(task["out"], "judge.json"), "w"), indent=1)
    if tiles:
        row(tiles).save(os.path.join(task["out"], "judge.jpg"), quality=90)


def which_ai(wanted):
    """The vision AI that judges: "ollama" (Qwen3.6-27B, the strong one) if it answers here, else "qwen3vl"
    (Qwen3-VL-8B through transformers: what the uni server has without Ollama). "" = none."""
    if wanted in ("none", "ollama", "qwen3vl"):
        return "" if wanted == "none" else wanted
    import urllib.request
    import grid_judge
    try:
        tags = json.load(urllib.request.urlopen(f"{grid_judge.OLLAMA}/api/tags", timeout=5))
        if any(m["name"].startswith(grid_judge.JUDGE) for m in tags.get("models", [])):
            return "ollama"
    except (OSError, ValueError):
        pass
    return "qwen3vl"


def asked(images, question):
    """The local AI's answer; once more after a pause if it fails (it had no room on the GPU: Nov 15, error 500)."""
    import grid_judge
    try:
        return grid_judge.ask(images, question, 768)
    except OSError as e:
        print(f"  the local AI failed ({e}), once more", flush=True)
        time.sleep(20)
        return grid_judge.ask(images, question, 768)


def step_keep(task):
    """5 KEEP: per correction, the picture's crop, the old render and the new one -> keep.json, <part>_compare.jpg.
    First everything is rendered, then the models leave the GPU, then the local AI is asked: both don't fit at once."""
    import gc
    import nvdiffrast.torch as dr
    import torch
    import grid_judge
    from project_texture import load_meshes, object_masks
    grid_judge.BACKEND["kind"] = task["ai"] if isinstance(task["ai"], str) and task["ai"] else "ollama"
    ctx = dr.RasterizeCudaContext()
    cams = grid_judge.load_cameras(task["cameras"])
    _, old_meshes = load_meshes(task["model"])
    views = {v["name"]: v for v in task["views"]}
    cache, shown = {}, {}
    for key, part in task["parts"].items():
        cell = part["cell"]
        name = cell["view"]
        if name not in cache:
            picture = open_view(views[name])
            cache[name] = (picture, object_masks([picture])[0] > 0.5, *renders_of(ctx, old_meshes, cams[name], picture.size)[:2])
        picture, mask, old, old_covered = cache[name]
        _, new_meshes = load_meshes(part["model"])
        new, new_covered, _ = renders_of(ctx, new_meshes, cams[name], picture.size)
        del new_meshes
        crop = cell["crop"]
        area = (slice(crop[1], crop[3]), slice(crop[0], crop[2]))
        ref, a, b = picture.crop(crop), old.crop(crop), new.crop(crop)
        iou = lambda c: float((mask & c).sum() / max(1, (mask | c).sum()))
        shown[key] = {"pictures": (ref, a, b), "before": measured(ref, a, mask[area], old_covered[area]),
                      "after": measured(ref, b, mask[area], new_covered[area]),
                      "whole": [round(iou(old_covered), 4), round(iou(new_covered), 4)]}
    del old_meshes, ctx
    gc.collect()
    torch.cuda.empty_cache()

    results = {}
    rng = random.Random(0)
    for key, one in shown.items():
        ref, a, b = one["pictures"]
        before, after, whole = one["before"], one["after"], one["whole"]
        answers, why = [], []
        if task["ai"]:
            for new_is in rng.sample(["A", "B"], 2):  # asked twice, the new one once as A and once as B
                answer = asked([ref, b, a] if new_is == "A" else [ref, a, b], CLOSER)
                said = str(answer.get("closer", "")).strip().upper()[:1]
                answers.append("new" if said == new_is else "old" if said in ("A", "B") else "same")
                why.append(str(answer.get("why", "")))
            keep = answers == ["new", "new"]
        else:  # no AI wanted: the lowest measured number clearly better
            keep = (lowest(after) or 0) >= (lowest(before) or 0) + 3
        if whole[1] < whole[0] - 0.01:  # the whole outline got worse: something outside the cell broke
            keep = False
        results[key] = {"keep": bool(keep), "answers": answers, "why": why, "before": before, "after": after, "outline_whole": whole}
        print(f"  {key}: AI {answers or 'not asked'}, measured {before} -> {after}, whole outline {whole[0]} -> {whole[1]}"
              + ("  KEPT" if keep else "  undone"), flush=True)
        tiles = []
        for image, label in ((ref, "picture"), (a, "before"), (b, "after: " + ("KEPT" if keep else "undone"))):
            image = image.resize((512, round(512 * image.height / image.width)), Image.LANCZOS)
            ImageDraw.Draw(image).text((8, 8), label, fill=(255, 255, 0))
            tiles.append(image)
        row(tiles).save(os.path.join(task["out"], f"{key}_compare.jpg"), quality=92)
    grid_judge.unload()
    json.dump(results, open(os.path.join(task["out"], "keep.json"), "w"), indent=1)


STEPS = {"fit": step_fit, "judge": step_judge, "keep": step_keep}


# ---------------------------------------------------------------- the loop

def run(command, log, done=None):
    """One step as its own process; skipped when `done` is there (a stopped run goes on where it was)."""
    if done and os.path.exists(done):
        print(f"  (done before) {os.path.relpath(done, PROJECT)}", flush=True)
        return True
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, "w") as f:
        code = subprocess.call([sys.executable, "-u", *map(str, command)], stdout=f, stderr=subprocess.STDOUT, cwd=PROJECT)
    ok = code == 0 and (not done or os.path.exists(done))
    if not ok:
        print(f"  FAILED (see {log}):\n" + open(log, errors="replace").read()[-1500:], flush=True)
    return ok


def step(name, task, done):
    """A GPU step of this file as its own process; what it printed is shown."""
    os.makedirs(task["out"], exist_ok=True)
    task_file = os.path.join(task["out"], "task.json")
    json.dump(task, open(task_file, "w"), indent=1)
    log = os.path.join(task["out"], "log.txt")
    was_there = os.path.exists(done)
    ok = run([os.path.abspath(__file__), "--step", name, "--task", task_file], log, done)
    if ok and not was_there:
        print("".join(line for line in open(log, errors="replace") if line.startswith("  ")), end="", flush=True)
    return ok


def overlap_3d(a, b):
    """How much of the smaller box lies in the other (0 to 1)."""
    a, b = np.array(a), np.array(b)
    common = np.clip(np.minimum(a[1], b[1]) - np.maximum(a[0], b[0]), 0, None).prod()
    return float(common / max(1e-12, min((a[1] - a[0]).prod(), (b[1] - b[0]).prod())))


def main():
    parser = argparse.ArgumentParser(description="The correction loop: one consistent base, corrected picture by picture")
    parser.add_argument("pictures", nargs="*", help="pictures or a folder of them (sheets are cut into views)")
    parser.add_argument("--name", help="the object's name (its folder in outputs)")
    parser.add_argument("--out", help="default: outputs/<name>/work/correct")
    parser.add_argument("--sides", default="", help="a sheet's views from left to right, e.g. \"front, side, back\"")
    parser.add_argument("--main", help="the view the base is made from (default: the data sheet's main view)")
    parser.add_argument("--base", help="a base model that is there already (.glb): step 1 is left out")
    parser.add_argument("--base-from", choices=["best", "one", "all"], default="best",
                        help="the base model: from the main view alone, from all sides mixed (multidiffusion), or "
                             "(default) both are made and judged and the better one is taken")
    parser.add_argument("--resolution", default="1536", help="the base model's resolution (default 1536)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--cells", type=int, default=3, help="cells along a view's longer side (default 3)")
    parser.add_argument("--threshold", type=float, default=80)
    parser.add_argument("--max-parts", type=int, default=4, help="parts corrected per round, the worst first")
    parser.add_argument("--ai", choices=["auto", "ollama", "qwen3vl", "none"], default="auto",
                        help="the vision AI that judges: ollama (Qwen3.6-27B), qwen3vl (Qwen3-VL-8B, no Ollama needed), "
                             "none (only measured); auto (default): ollama if it answers here, else qwen3vl")
    parser.add_argument("--zip", action="store_true", help="also a small zip of the results (for the cloud worker)")
    parser.add_argument("--live", help="the cloud job's name: send every in-between result at once")
    parser.add_argument("--step", choices=STEPS, help=argparse.SUPPRESS)
    parser.add_argument("--task", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.step:
        return STEPS[args.step](json.load(open(args.task)))
    if not args.pictures or not args.name:
        parser.error("the pictures and --name are needed")
    out = os.path.abspath(args.out or os.path.join(PROJECT, "outputs", args.name, "work", "correct"))
    start = time.time()
    script = lambda path: os.path.join(SCRIPTS, *path.split("/"))
    ai = which_ai(args.ai)
    print(f"the judging AI: {ai or 'none (only measured)'}", flush=True)
    summary = [f"# {args.name}: the correction loop (threshold {args.threshold}, judging AI: {ai or 'none'})", ""]

    def live(path, label):
        """An in-between result into the cloud at once (--live), so it can be looked at while the job runs."""
        if not args.live or not os.path.exists(path):
            return
        try:
            sys.path.insert(0, os.path.join(SCRIPTS, "server"))
            from cloud import Cloud
            Cloud().upload(path, f"{args.live}__live_{label}")
        except Exception as e:  # noqa: BLE001 (only the live copy is lost; the job goes on)
            print(f"  not sent: {label} ({type(e).__name__}: {e})", flush=True)

    def show(path, title):
        """The model into outputs/<name> at once, as the next version."""
        subprocess.call([sys.executable, script("tools/arrange_outputs.py"), "--do", "--more", args.name, title, path],
                        stdout=subprocess.DEVNULL)
        live(path, title + ".glb")

    print("0 VIEWS", flush=True)
    sheet_path = os.path.join(out, "00_views", "sheet.json")
    command = [script("pipeline/data_sheet.py"), *args.pictures, "--role", "object", "--out", os.path.dirname(sheet_path)]
    if ai != "ollama":  # the data sheet's AI is Ollama's: without it the sides come from --sides or count as front
        command.append("--no-ai")
    if not run(command + (["--sides", args.sides] if args.sides else []), os.path.join(out, "00_views", "log.txt"), sheet_path):
        return 1
    sheet = json.load(open(sheet_path))
    views = [{"name": v["name"], "side": v.get("side"),  # the file next to the sheet: the folder may have been moved
              "path": os.path.relpath(os.path.join(os.path.dirname(sheet_path), "views", os.path.basename(v["path"])), PROJECT)}
             for v in sheet["views"]]
    main_view = args.main or sheet["lists"]["base"][0]
    print(f"  {len(views)} views, the base from {main_view}", flush=True)

    def judge(path, name, suffix, cameras):
        folder = os.path.join(out, "03_judge" + suffix, name)
        task = {"model": path, "views": views, "cameras": cameras, "out": folder, "cells": args.cells,
                "threshold": args.threshold, "ai": ai}
        if not step("judge", task, os.path.join(folder, "judge.json")):
            return None
        live(os.path.join(folder, "judge.jpg"), f"judge{suffix}_{name}.jpg")
        return json.load(open(os.path.join(folder, "judge.json")))

    def mean(result):
        return float(np.mean([c["score"] for c in result["cells"] if not c.get("whole") and c["score"] is not None]))

    print("1 BASE", flush=True)
    # two candidates: the main view alone (free renders that don't fit together doubled parts when mixed), and all
    # sides mixed (multidiffusion: right for a real turnaround, no side has to be invented). Both are fitted and
    # judged against every view; the better one is the base. Folders of the mixed one end in _all.
    path_of = lambda name: os.path.join(PROJECT, next(v["path"] for v in views if v["name"] == name))
    sides = [n for n in sheet["lists"]["base"] if n != main_view]
    candidates = {"": ("one_view_" + main_view, [script("generation/generate.py"), path_of(main_view)])}
    if sides and not args.base and args.base_from != "one":
        candidates["_all"] = (f"{len(sides) + 1}_views_mixed", [script("generation/multi_image.py"), path_of(main_view),
                                                               *map(path_of, sides), "--mode", "multidiffusion"])
    if args.base_from == "all" and "_all" in candidates:
        del candidates[""]
    judged = {}
    for suffix, (title, command) in candidates.items():
        model = os.path.join(out, "01_base" + suffix, "base.glb")
        if args.base and not os.path.isfile(model):
            os.makedirs(os.path.dirname(model), exist_ok=True)
            shutil.copy(args.base, model)
        if not run([*command, "--resolution", args.resolution, "--seed", args.seed, "--faces", "4000000", "--out", model],
                   os.path.join(out, "01_base" + suffix, "log.txt"), model):
            continue
        show(model, "base_given" if args.base else f"base_{title}_{args.resolution}_seed{args.seed}")
        print(f"2 FIT, the base from {title}", flush=True)
        fit = os.path.join(out, "02_fit" + suffix)
        if not step("fit", {"model": model, "views": views, "out": fit}, os.path.join(fit, "cameras.json")):
            continue
        print(f"3 JUDGE, the base from {title}", flush=True)
        result = judge(model, "r1", suffix, os.path.join(fit, "cameras.json"))
        if result:
            judged[suffix] = (mean(result), title, model)
            summary.append(f"- base from {title}: mean {mean(result):.1f}, {sum(c['bad'] for c in result['cells'] if not c.get('whole'))} bad cells")
    if not judged:
        return 1
    suffix = max(judged, key=lambda k: judged[k][0])
    _, title, model = judged[suffix]
    base = model
    if len(judged) > 1:
        print(f"  the base is the one from {title}", flush=True)
        summary.append(f"  -> the base is the one from {title}")
    cameras = os.path.join(out, "02_fit" + suffix, "cameras.json")
    fits = json.load(open(os.path.join(out, "02_fit" + suffix, "fit.json")))
    summary += ["cameras (outline overlap): " + ", ".join(f"{n} {o}" + ("" if o >= MIN_FIT else " (left out)") for n, o in fits.items()), ""]

    def line(result, label):
        cells = [c for c in result["cells"] if not c.get("whole") and c["score"] is not None]
        scores = [c["score"] for c in cells]
        return (f"- {label}: {len(cells)} cells, mean {np.mean(scores):.1f}, lowest {min(scores)}, "
                f"{sum(c['bad'] for c in cells)} bad")

    for r in range(1, args.rounds + 1):
        print(f"3 JUDGE, round {r}", flush=True)
        result = judge(model, f"r{r}", suffix, cameras)
        if result is None:
            return 1
        summary.append(line(result, f"round {r}, before"))
        chosen = []  # whole sides first, then the worst cells that don't lie in a region chosen before
        for cell in sorted((c for c in result["cells"] if c["bad"]), key=lambda c: (not c.get("whole"), c["score"])):
            if len(chosen) < args.max_parts and all(overlap_3d(cell["box_model"], other["box_model"]) < 0.3 for other in chosen):
                chosen.append(cell)
        print(f"  {sum(c['bad'] for c in result['cells'])} bad cells, {len(chosen)} are corrected", flush=True)
        if not chosen:
            break

        print(f"4 CORRECT, round {r}", flush=True)
        parts = {}
        for cell in chosen:
            key = f"{cell['view']}_side" if cell.get("whole") else f"{cell['view']}_c{cell['cell'][0]}{cell['cell'][1]}"
            folder = os.path.join(out, "04_parts" + suffix, f"r{r}", key)
            part = os.path.join(folder, "part.glb")
            view = next(v for v in views if v["name"] == cell["view"])
            print(f"  {key} (seen {cell['seen']}, measured {cell['measured']})", flush=True)
            command = [script("inpainting/refine_region.py"), model, os.path.join(PROJECT, view["path"]), "--box3d",
                       *[f"{x:.5f}" for x in cell["box"][0] + cell["box"][1]], "--camera", cameras, cell["view"],
                       "--face-camera", "--margin", "0.1" if cell.get("whole") else "0.35", "--method", "inversion", "--start", "random", "--soft", "2", "--steps", "25", "--seed", args.seed,
                       "--texture-guidance", "3", "--faces", "1500000", "--grow", "1", "--out", part]
            if run(command, os.path.join(folder, "log.txt"), part):
                parts[key] = {"model": part, "cell": cell}
        if not parts:
            continue

        print(f"5 KEEP, round {r}", flush=True)
        keep_dir = os.path.join(out, "05_keep" + suffix, f"r{r}")
        if not step("keep", {"model": model, "views": views, "cameras": cameras, "parts": parts, "out": keep_dir,
                             "ai": ai}, os.path.join(keep_dir, "keep.json")):
            return 1
        kept = {k: v for k, v in json.load(open(os.path.join(keep_dir, "keep.json"))).items() if v["keep"]}
        for k in parts:
            live(os.path.join(keep_dir, f"{k}_compare.jpg"), f"r{r}_{k}_compare.jpg")
        summary.append(f"  corrected {', '.join(parts)}; kept: {', '.join(kept) or 'none'}")
        if not kept:
            print("  no correction was closer to its picture: the model stays", flush=True)
            continue
        new_model = os.path.join(out, "05_model" + suffix, f"r{r}.glb")
        if not run([script("pipeline/combine_parts.py"), model, *[parts[k]["model"] for k in kept], "--out", new_model,
                    "--single"], os.path.join(out, "05_model" + suffix, f"r{r}_log.txt"), new_model):
            return 1
        model = new_model
        show(model, f"corrected_round{r}_kept_{len(kept)}_of_{len(parts)}_parts")

    if model != base:
        print("3 JUDGE, the final model", flush=True)
        result = judge(model, "final", suffix, cameras)
        if result:
            summary.append(line(result, "final"))
    shutil.copy(model, os.path.join(out, "final.glb"))
    summary.append(f"\nfinal: {os.path.relpath(model, out)}, {time.time() - start:.0f} s")
    open(os.path.join(out, "summary.md"), "w").write("\n".join(summary) + "\n")
    print("\n".join(summary), flush=True)
    if args.zip:  # the small results: the models, the judge and compare pictures, the scores
        import zipfile
        keep = lambda rel: (rel.endswith((".json", ".md", ".jpg")) and "_picture.jpg" not in rel and "_model.jpg" not in rel
                            or rel == "final.glb" or (rel.startswith(("01_base", "05_model")) and rel.endswith(".glb")))
        with zipfile.ZipFile(out + ".zip", "w", zipfile.ZIP_DEFLATED) as z:
            for root, _, files in os.walk(out):
                for f in files:
                    rel = os.path.relpath(os.path.join(root, f), out)
                    if keep(rel) and not rel.startswith(("00_", "04_")):
                        z.write(os.path.join(root, f), os.path.join(args.name, rel))
        print(f"Result: {out}.zip", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
