"""Node 01 of the inpainting flow: the data sheet of one input (pictures 1..n, videos, or both).

An input is either the OBJECT (what TRELLIS.2 builds first, e.g. the girl) or the PART that is inpainted onto it
later (e.g. the crown). TRELLIS.2 reads one picture as one object, so the sheet turns whatever comes in into
single views and says what each one is good for.

  1. every picture   a sheet with several views side by side is cut into its views (measured: separate blobs of
                     object pixels, split() below); every video gives frames a part of a turn apart
  2. every view      measured: size, sharpness, grey sculpt or colour, cut off at which border, how symmetric
  3. every view      the local vision AI (Qwen3.6-27B, Ollama) says: what it shows, which side (front,
                     three-quarter, side, back), what is at its very top; the views of one sheet are shown to it
                     together in one picture, so it can tell front from back by comparing them
  4. rules           the lists the next nodes use:
                       base   the views a model is generated from: the most frontal complete one first, then
                              one per other side (three-quarter, side, back), from the same sheet if it has one
                       sides  a part's four sides in turning order (from a video: a quarter turn apart)

Writes into --out: sheet.json, views/*.png, contact.jpg (all views with their tags, to check at a glance).

    python scripts/pipeline/data_sheet.py pictures/girl.jpg --role object --out outputs/girl/work/01_sheet
    python scripts/pipeline/data_sheet.py pictures/crown.mp4 --role part --out outputs/girl/work/01_sheet_part
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


# ---- a sheet with several views side by side, split into single views (also used by inpaint.py)
# TRELLIS.2 reads one picture as one object: four helmets in a row gave a mix of all four (Nov 4). The background AI
# (the same one TRELLIS.2 uses) finds the object pixels; every separate blob of them is one view. Measured, no AI
# that names anything.

def object_mask(pipeline, image):
    if pipeline.low_vram:
        pipeline.rembg_model.to(pipeline.device)
    rgba = np.asarray(pipeline.rembg_model(image))
    if pipeline.low_vram:
        pipeline.rembg_model.cpu()
    return rgba[..., 3] > 128


def boxes(mask, shrink=3, share=0.25):
    """The views' boxes: separate blobs of object pixels. The mask is first shrunk a little, so that a thin cable or
    antenna touching the next view doesn't join two views; a blob counts if it has at least `share` of the biggest
    one's pixels (logos, text and loose small bits drop out)."""
    from scipy import ndimage
    core = ndimage.binary_erosion(mask, iterations=shrink)
    labels, count = ndimage.label(core)
    if count < 2:
        return []
    sizes = ndimage.sum(core, labels, range(1, count + 1))
    keep = [k for k in np.argsort(-sizes) if sizes[k] >= share * sizes.max()]  # the biggest first
    found = []
    for k, (rows, cols) in enumerate(ndimage.find_objects(labels)):
        if k in keep:
            found.append((keep.index(k), (cols.start - shrink, rows.start - shrink, cols.stop + shrink, rows.stop + shrink)))
    return [box for _, box in sorted(found)]


def split(pipeline, path, out_dir, name="inpaint_view", left_to_right=False):
    """The single views of a sheet, saved as <name><k>.png (biggest first, or as they stand on the sheet from left to
    right); [] when it shows one view."""
    image = Image.open(path).convert("RGB")
    scale = min(1.0, 1024 / max(image.size))
    small = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
    found = boxes(object_mask(pipeline, small))
    if len(found) < 2:
        return []
    if left_to_right:
        found.sort(key=lambda box: box[0])
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for k, (x0, y0, x1, y1) in enumerate(found):
        pad = 0.03 * max(x1 - x0, y1 - y0)
        box = [max(0, round((x0 - pad) / scale)), max(0, round((y0 - pad) / scale)),
               min(image.width, round((x1 + pad) / scale)), min(image.height, round((y1 + pad) / scale))]
        paths.append(os.path.join(out_dir, f"{name}{k + 1}.png"))
        image.crop(box).save(paths[-1])  # small views are upscaled to 1024 later (generate.crop_to_object)
    return paths


TOGETHER = """This picture shows {n} views of the same object next to each other, numbered 1 to {n} from left to right.
They show it from different sides. Compare them with each other before you answer: only a view where the object faces
the camera (its face, eyes or chest are seen) is the front; in a view from behind the back of the head and the back are
seen and no face; a profile is a side.
Answer only with JSON, no other text:
{{"what": "what the object is, at most 8 words",
 "views": [{{"number": 1, "side": "front" or "three-quarter" or "side" or "back" or "top" or "unknown",
            "top": "what is at the very top of the object, at most 5 words"}}, ...one entry per view...]}}"""


def sides_together(group, path, ask, encoded, height=640):
    """The views of one sheet shown to the local AI in one picture, so it can tell them apart: looked at alone, it
    called the bee robot's back a front (Nov 14). Returns {view name: (answer, raw)}, or {} when the answer is unusable
    (the views are then asked one by one)."""
    pictures = []
    for view in group:
        image = Image.open(view["path"]).convert("RGB")
        pictures.append(image.resize((max(1, round(image.width * height / image.height)), height), Image.LANCZOS))
    strip = Image.new("RGB", (sum(p.width for p in pictures) + 20 * (len(pictures) + 1), height + 60), (255, 255, 255))
    draw, x = ImageDraw.Draw(strip), 20
    for number, picture in enumerate(pictures, 1):
        draw.text((x + picture.width // 2 - 4, 8), str(number), fill=(0, 0, 0), font_size=36)
        strip.paste(picture, (x, 54))
        x += picture.width + 20
    strip.save(path, quality=92)
    try:
        answer, raw = ask([encoded(path, 1536)], TOGETHER.format(n=len(group)))
        said = {int(one["number"]): one for one in answer["views"]}
        if sorted(said) != list(range(1, len(group) + 1)):
            return {}
        return {view["name"]: ({"what": answer.get("what", ""), "side": said[k]["side"], "top": said[k].get("top", "")},
                               "together: " + raw) for k, view in enumerate(group, 1)}
    except Exception as e:  # noqa: BLE001 (a wrong shape of answer, or Ollama not running)
        print(f"  the views together gave no answer ({type(e).__name__}): asked one by one", flush=True)
        return {}


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
        # complete views first (a close-up cut off at a border is no view of the whole object); of those about as
        # symmetric as the most symmetric one, the one that shows the object biggest
        whole = [v for v in fronts if not v["cut_off"]] or fronts
        most = max(v["symmetry"] for v in whole)
        big = lambda v: (v["object_box"][2] - v["object_box"][0]) * (v["object_box"][3] - v["object_box"][1])
        main = max([v for v in whole if v["symmetry"] >= most - 0.05], key=lambda v: (big(v), v["sharpness"]))
        lists["base"].append(main["name"])
        left_out = len(fronts) - len(whole)
        lists["why"].append(f"base main view: {main['name']} ({main['side']}, symmetry {main['symmetry']}"
                            + (f"; {left_out} cut-off view(s) left out" if left_out else "") + ")")
        for side in ORDER[1:]:
            others = [v for v in by_side.get(side, []) if v["name"] not in lists["base"]]
            if others and side != main["side"]:
                # the same sheet as the main view first: another picture can show another version of the object (a
                # statue with its base, a colour version without it), and mixed they give a mix
                best = max(others, key=lambda v: (v["from"] == main["from"], not v["cut_off"], v["sharpness"]))
                lists["base"].append(best["name"])
                lists["why"].append(f"base {side}: {best['name']} (of {len(by_side[side])}: "
                                    + ("the main view's own sheet" if best["from"] == main["from"] else "the sharpest complete one") + ")")
        # free views (renders from any angle, e.g. three whole pictures of a robot arm that are all "three-quarter"):
        # the other complete views of the same thing count too, up to 4 in all. Only single pictures and the main
        # view's own sheet (another sheet can be another version), and only views the AI calls by a word of the main
        # view's name (a picture of a loose part alone is no view of the whole).
        per_source = {}
        for view in stills:
            per_source[view["from"]] = per_source.get(view["from"], 0) + 1
        words = lambda v: {w for w in v["what"].lower().replace("-", " ").split() if len(w) > 3 and w not in ("model", "with", "rendered")}
        same = lambda v: not words(main) or not words(v) or bool(words(main) & words(v))
        if main["what"]:  # the per-side picks must be the same thing as well
            dropped = [n for n in lists["base"][1:] if not same(next(v for v in stills if v["name"] == n))]
            lists["base"] = [n for n in lists["base"] if n not in dropped]
            lists["why"] += [f"base: {n} left out (not called as the main view: another thing)" for n in dropped]
        more = [v for v in stills if v["name"] not in lists["base"] and not v["cut_off"] and same(v)
                and (per_source[v["from"]] == 1 or v["from"] == main["from"])]
        for view in sorted(more, key=lambda v: -v["sharpness"])[:max(0, 4 - len(lists["base"]))]:
            lists["base"].append(view["name"])
            lists["why"].append(f"base one more view: {view['name']} ({view['side']}, complete, the same thing)")
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
    parser.add_argument("--what", default="", help="what the input shows, in the user's own words: taken instead of the local "
                        "AI's guess (which reads e.g. a logo on the picture as the object's name)")
    parser.add_argument("--sides", default="", help="the sides of a sheet's views from left to right, in the user's own words, "
                        "e.g. \"front, side, back\": taken instead of the local AI's guess (which called a robot's back its front)")
    args = parser.parse_args()
    said_sides = [s.strip().lower() for s in args.sides.split(",") if s.strip()]
    if any(s not in SIDES for s in said_sides):
        raise SystemExit(f"--sides: only {', '.join(SIDES)}")
    views_dir = os.path.join(args.out, "views")
    os.makedirs(views_dir, exist_ok=True)
    files = []
    for path in args.files:
        files += sorted(os.path.join(path, f) for f in os.listdir(path)) if os.path.isdir(path) else [path]
    files = [f for f in files if f.lower().endswith(PICTURE_TYPES + VIDEO_TYPES)]
    if not files:
        raise SystemExit("no pictures or videos given")

    import types
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
        cut = split(pipeline, path, views_dir, name=f"{name}_view", left_to_right=bool(said_sides))
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
        together = {}
        for number, source in enumerate(dict.fromkeys(v["from"] for v in views if v["from_video"] is None)):
            group = [v for v in views if v["from"] == source and v["from_video"] is None]
            if len(group) > 1:  # a sheet: its views in one question
                together.update(sides_together(group, os.path.join(args.out, f"together_{number}.jpg"), ask, encoded))
        for view in views:
            if view["from_video"] is not None and view["turn"] != 0:  # a video: its turn says the side
                view.update(what=views_first(views, view)["what"], top="", ai=None,
                            side={90: "side", 180: "back", 270: "side"}.get(view["turn"], "three-quarter"))
                continue
            try:
                answer, raw = together.get(view["name"]) or ask([encoded(view["path"], 768)], QUESTION)
            except Exception as e:  # noqa: BLE001 (Ollama not running: the sheet still has what was measured)
                answer, raw = {}, f"{type(e).__name__}: {e}"
            side = answer.get("side") if answer.get("side") in SIDES else "unknown"
            if view["from_video"] is not None:
                side = "front"  # a turning video starts at the front
            view.update(what=str(answer.get("what", ""))[:80], side=side, top=str(answer.get("top", ""))[:60], ai=raw[:400])
            print(f"  {view['name']}: {view['side']}, {view['colour']}, {view['what']}", flush=True)
    stills = [v for v in views if v["from_video"] is None]
    if said_sides:  # the user's words win over the AI's guess
        if len(said_sides) != len(stills):
            raise SystemExit(f"--sides names {len(said_sides)} views, the pictures show {len(stills)}")
        for view, side in zip(stills, said_sides):
            view.update(side=side, **({"side_ai": view["side"]} if view["side"] != side else {}))
    lists = choose_lists(views, args.role)
    what = next((v["what"] for v in views if v["name"] in (lists["base"] + lists["sides"])[:1] and v["what"]), "")
    said = " ".join(args.what.split())
    sheet = {"role": args.role, "what": said or what, **({"what_ai": what} if said else {}), "files": files, "ai": None if args.no_ai else MODEL, "views": views,
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
