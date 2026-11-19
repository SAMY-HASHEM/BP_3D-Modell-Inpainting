"""What every pipeline node does: which commands it runs, in which folder, with which result files.

Shared by both node editors, ComfyUI (comfyui/bp_inpainting) and Blender (blender/pipeline_nodes.py), so a
node does exactly the same in both. Only Python's standard library is used, so it also loads inside Blender
and inside ComfyUI's own Python. The commands themselves run in the project environment (activate.sh).

A node returns a job: {"folder", "steps": [(command, label, done_file)], "outputs": {...}}. A step whose
done_file is already there is skipped: a node's folder is named after its settings and inputs (with their
modification times), so the same settings are never computed twice, even after a restart.
"""
import hashlib
import json
import os

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
SCRIPTS = os.path.join(PROJECT, "scripts")
OUTPUTS = os.path.join(PROJECT, "outputs")  # outputs/<object>/: its models as versions, and work/ (what the programs need)
PICTURE_TYPES = (".png", ".jpg", ".jpeg", ".webp")
VIDEO_TYPES = (".mp4", ".mov", ".webm", ".avi", ".mkv", ".m4v")
# the same parts as scripts/pipeline/sort_pictures.py (PARTS)
PARTS = ["head", "hair", "bust", "belly", "back", "butt", "thighs", "lower_legs", "feet"]
VISION_MODELS = ["Qwen/Qwen3-VL-8B-Instruct", "Qwen/Qwen3-VL-2B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct"]


def project_path(path):
    path = os.path.expanduser(path.strip())
    return path if os.path.isabs(path) else os.path.join(PROJECT, path)


def stamp(path):
    try:
        return [path, round(os.path.getmtime(path), 1)]
    except OSError:
        return [path, None]


def flow_folder(name):
    """Where a flow works: outputs/<name>/work (its models are put next to it as versions, arrange_outputs.py)."""
    return os.path.join(OUTPUTS, name, "work")


def node_folder(name, node, settings):
    digest = hashlib.sha1(json.dumps(settings, sort_keys=True, default=str).encode()).hexdigest()[:8]
    folder = os.path.join(flow_folder(name), node, digest)
    os.makedirs(folder, exist_ok=True)
    json.dump(settings, open(os.path.join(folder, "settings.json"), "w"), indent=1, default=str)
    return folder


def py(script, *args):
    return ["python", os.path.join(SCRIPTS, *script.split("/")), *[str(a) for a in args]]


def preview(*args):
    return py("pipeline/preview.py", *args)


# ------------------------------------------------------------------ the nodes

def pictures(name, folder):
    path = project_path(folder)
    if not os.path.isdir(path):
        raise ValueError(f"no folder {path}")
    files = sorted(os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(PICTURE_TYPES + VIDEO_TYPES))
    return {"name": name.strip().replace(" ", "_"), "files": files}


def video(pics, path, frames):
    folder = node_folder(pics["name"], "00_video", {"video": stamp(path), "frames": frames})
    frames_dir = os.path.join(folder, "frames")
    return {"folder": folder, "steps": [(py("pipeline/sort_pictures.py", path, "--frames", frames, "--out", folder),
                                         "video frames", frames_dir)],
            "outputs": {"frames": frames_dir}}


def video_result(pics, job):
    new = sorted(os.path.join(job["outputs"]["frames"], f) for f in os.listdir(job["outputs"]["frames"]))
    return {**pics, "files": pics["files"] + new}, new


def sort(pics, model):
    files = [f for f in pics["files"] if f.lower().endswith(PICTURE_TYPES)]
    folder = node_folder(pics["name"], "01_sort", {"files": [stamp(f) for f in files], "model": model})
    inbox = os.path.join(folder, "in")
    os.makedirs(inbox, exist_ok=True)
    for k, f in enumerate(files):  # one folder with all pictures, as sort_pictures.py wants it
        link = os.path.join(inbox, os.path.basename(f))
        if os.path.lexists(link) and os.path.realpath(link) != os.path.realpath(f):
            link = os.path.join(inbox, f"{k:02d}_{os.path.basename(f)}")
        if not os.path.lexists(link):
            os.symlink(os.path.realpath(f), link)
    out = os.path.join(folder, "sorted.json")
    return {"folder": folder, "steps": [(py("pipeline/sort_pictures.py", inbox, "--out", folder, "--model", model), "01 sort", out)],
            "outputs": {"sorted": {"path": out, "name": pics["name"]}, "sheet": os.path.join(folder, "contact.jpg")}}


def lists_text(sorted_):
    result = json.load(open(sorted_["path"]))
    short = lambda n: n.replace("deyvson-william-", "")
    lines = [f"{key}: " + "; ".join(f"{role} {', '.join(short(n) for n in names)}"
                                     for role, names in value.items() if isinstance(names, list) and names)
             for key, value in result["lists"].items() if isinstance(value, dict)]
    lines.append(f"hair colour: {(result.get('hair') or {}).get('hair_colour')}")
    return "\n".join(lines)


def base(sorted_, mode="multidiffusion", resolution="1024", steps=12, faces=4_000_000, seed=42):
    result = json.load(open(sorted_["path"]))
    views = {v["name"]: v for v in result["views"]}
    lists = result["lists"]["base"]
    path = lambda names: [os.path.join(os.path.dirname(sorted_["path"]), views[n]["view"]) for n in names]
    folder = node_folder(sorted_["name"], "02_base", {"sorted": stamp(sorted_["path"]), "mode": mode, "resolution": resolution,
                                                      "steps": steps, "faces": faces, "seed": seed, "min_side": 1024})
    out = os.path.join(folder, "base.glb")
    command = py("generation/multi_image.py", *path(lists["both"]), "--mode", mode, "--resolution", resolution,
                 "--steps", steps, "--faces", faces, "--seed", seed, "--out", out)
    if lists["shape_only"]:
        command += ["--shape-only", *path(lists["shape_only"])]
    if lists["colour_only"]:
        command += ["--colour-only", *path(lists["colour_only"])]
    sheet = os.path.join(folder, "compare.jpg")
    refs = path(lists["shape_only"] + lists["both"] + lists["colour_only"])
    return {"folder": folder, "steps": [(command, "02 base", out), (preview("base", folder, out, *refs), "02 base preview", sheet)],
            "outputs": {"model": {"glb": out, "name": sorted_["name"], "kind": "base", "label": "base"}, "sheet": sheet}}


def refine(base_model, sorted_, part, shape="auto", colour_pictures=1, shape_pictures="auto", texture_guidance=3.0,
           steps=25, seed=42):
    """auto = the lessons: the head also regenerates its shape (margin 0.6 of the head) from ONE picture in total;
    every other part only its colours (shape kept, margin 0.1 of the model); one sharp complete colour picture."""
    lists = json.load(open(sorted_["path"]))["lists"].get(part, {})
    if not any(isinstance(v, list) and v for v in lists.values()):  # no picture shows this part: skipped
        return {"folder": None, "steps": [], "outputs": {"model": None, "skipped": f"no picture shows the {part}"}}
    keep = (part != "head") if shape == "auto" else shape.startswith("keep")
    no_shape_views = shape_pictures == "none" or (shape_pictures == "auto" and part == "head")
    settings = {"base": stamp(base_model["glb"]), "sorted": stamp(sorted_["path"]), "part": part, "keep_shape": keep,
                "colour_pictures": colour_pictures, "texture_guidance": texture_guidance, "steps": steps, "seed": seed,
                "min_side": 1024}
    if no_shape_views:
        settings["shape_pictures"] = 0
    folder = node_folder(sorted_["name"], f"04_refine_{part}", settings)
    out = os.path.join(folder, f"{part}.glb")
    command = py("inpainting/refine_region.py", base_model["glb"], "--sorted", sorted_["path"], "--part", part,
                 "--method", "repaint", "--margin", "0.6", "--steps", steps, "--seed", seed, "--mode", "stochastic",
                 "--grow", "1", "--texture-guidance", texture_guidance, "--faces", "4000000", "--out", out)
    if part != "head":
        command += ["--margin-of-model", "0.1"]
    if keep:
        command += ["--keep-shape"]
    if colour_pictures:
        command += ["--best-colour", str(colour_pictures)]
    if no_shape_views:
        command += ["--shape-pictures", "0"]
    sheet = os.path.join(folder, "compare.jpg")
    label = f"{part} ({'colours' if keep else 'shape+colours'})"
    return {"folder": folder, "steps": [(command, f"04 refine {part}", out),
                                        (preview("refine", folder, base_model["glb"], out), f"04 {part} preview", sheet)],
            "outputs": {"model": {"glb": out, "name": sorted_["name"], "kind": "part", "part": part, "label": label},
                        "sheet": sheet}}


def combine(base_model, parts):
    folder = node_folder(base_model["name"], "05_combine", {"base": stamp(base_model["glb"]), "parts": [stamp(p["glb"]) for p in parts]})
    out = os.path.join(folder, "combined.glb")
    sheet = os.path.join(folder, "compare.jpg")
    labels = [p.get("part", "?") for p in parts]
    return {"folder": folder, "steps": [(py("pipeline/combine_parts.py", base_model["glb"], *[p["glb"] for p in parts], "--out", out),
                                         "05 combine", out),
                                        (preview("combine", folder, base_model["glb"], out, *labels), "05 preview", sheet)],
            "outputs": {"model": {"glb": out, "name": base_model["name"], "kind": "combined", "label": "combined"}, "sheet": sheet}}


def focus_of(model):
    """The box of a refined part (refine.json), e.g. the head for the hair node's cameras."""
    work = os.path.splitext(model["glb"])[0] + "_work"
    try:
        info = json.load(open(os.path.join(work, "refine.json")))
    except OSError:
        return None
    return [*info["inner"][0], *info["inner"][1]]


def hair(model, sorted_, count=35000, length=0.10, clump=0.35, head=None, blender="blender"):
    result = json.load(open(sorted_["path"]))
    colour = (result.get("hair") or {}).get("hair_colour")
    if not colour:
        raise ValueError("the Sort node found no hair colour")
    front = next(v for v in result["views"] if v["name"] == result["lists"]["base"]["front"])
    box = front["part_boxes"].get("hair") or front.get("part_boxes_seen", {}).get("hair")
    share = (box[3] - front["box"][1]) / (front["box"][3] - front["box"][1]) + 0.05 if box else 1.0
    focus = focus_of(head) if head else None
    folder = node_folder(model["name"], "06_hair", {"model": stamp(model["glb"]), "colour": colour, "share": round(share, 3),
                                                    "count": count, "length": length, "clump": clump, "focus": focus})
    out = os.path.join(folder, "hair")
    command = [blender, "-b", "--python", os.path.join(SCRIPTS, "pipeline", "hair_strands.py"), "--", model["glb"], out,
               "--hair-colour", colour, "--hair-share", f"{share:.3f}", "--count", str(count), "--length", str(length),
               "--clump", str(clump), "--glb"]
    if focus:
        command += ["--focus", *[f"{v:.5f}" for v in focus]]
    renders = [out + f"_{n}.png" for n in ("front", "left", "right", "back")]
    return {"folder": folder, "steps": [(command, "06 hair", out + ".glb")],
            "outputs": {"glb": out + ".glb", "blend": out + ".blend", "renders": renders,
                        "model": {"glb": out + ".glb", "name": model["name"], "kind": "hair", "label": "with hair"}}}


def compare(models, clay=False):
    name = models[0]["name"]
    folder = node_folder(name, "99_compare", {"models": [stamp(m["glb"]) for m in models], "clay": clay})
    out = os.path.join(folder, "compare.jpg")
    return {"folder": folder, "steps": [(preview("compare", out, *(["--clay"] if clay else []), *[m["glb"] for m in models]),
                                         "99 compare", out)],
            "outputs": {"sheet": out}}


def pick_best(candidates):
    candidates = [c for c in candidates if c]
    """04b: the local vision AI (Qwen3.6-27B, Ollama) picks the candidate closest to the artist's picture.
    The chosen model is only known after the run: resolve() reads it."""
    folder = node_folder(candidates[0]["name"], f"04b_pick_{candidates[0].get('part', 'part')}",
                         {"candidates": [stamp(c["glb"]) for c in candidates], "judge": "one candidate at a time"})
    out = os.path.join(folder, "judge.json")
    return {"folder": folder,
            "steps": [(py("pipeline/judge.py", *[os.path.dirname(c["glb"]) for c in candidates], "--out", folder), "04b pick", out)],
            "outputs": {"model_from": out, "candidates": candidates, "sheet": os.path.join(folder, "compare.jpg")}}


def resolve(outputs):
    """Outputs that are only known after the run (pick_best: which candidate won; a server job: its files)."""
    if "models_from" in outputs:  # a server job (cloud.py --wait): the models it sent back
        like = outputs["model_like"]
        files = sorted(f for f in json.load(open(outputs["models_from"]))["results"] if f.endswith(".glb"))
        models = [{**like, "glb": f, "label": f"{like['label']} {os.path.basename(f)[-14:-4].split('_')[-1]}"} for f in files]
        outputs = {**outputs, "models": models, "model": models[0]}
    if "model_from" in outputs:
        answer = json.load(open(outputs["model_from"]))
        best = os.path.abspath(answer["best"])
        outputs = {**outputs, "model": next(c for c in outputs["candidates"] if os.path.abspath(c["glb"]) == best)}
        if "verdict" in answer:
            outputs["verdict"] = answer["verdict"] + (f" ({'; '.join(answer['why'])})" if answer.get("why") else "") + \
                (f"; warning: {'; '.join(answer['warnings'])}" if answer.get("warnings") else "")
    return outputs


# ------------------------------------------------------------------ the inpainting flow (a new part onto a model)
#
#   00 Input (object) -> 01 Data sheet -> 02 Base model (x seeds) -> 02b Pick seed --+
#                                                                                    +-> 03 Edit -> 04 Inpaint: shape
#   00 Input (part)   -> 01 Data sheet ---------------------------------------------+      -> 05 Judge -> 04 details
#                                                                                           -> 05 Judge -> 04 colours -> 05 Judge
#
# An input is pictures (1..n), videos, or both. "where" = "server" sends the TRELLIS.2 work to the uni GPU server
# through the cloud and waits for the result (cloud.py --wait), "pc" runs it on this computer.

def inputs(name, files):
    """00: pictures and videos of one thing (files, or folders with everything in them)."""
    found = []
    for path in files if isinstance(files, (list, tuple)) else [f for f in files.split(";") if f.strip()]:
        path = project_path(path)
        if os.path.isdir(path):
            found += sorted(os.path.join(path, f) for f in os.listdir(path))
        elif os.path.isfile(path):
            found.append(path)
        else:
            raise ValueError(f"no file or folder {path}")
    found = [f for f in found if f.lower().endswith(PICTURE_TYPES + VIDEO_TYPES)]
    if not found:
        raise ValueError("no pictures or videos in the input")
    return {"name": name.strip().replace(" ", "_"), "files": found}


def data_sheet(pics, role="object", frames=8, ai=True, what="", sides=""):
    """01: the local AI's data sheet of one input: its single views, what each is, and the lists for TRELLIS.2.
    what = the user's own words for what the input shows: the judge gets them instead of the local AI's guess.
    sides = the user's own words for the sides of a sheet's views, left to right ("front, side, back")."""
    what = " ".join(str(what or "").split())
    sides = ", ".join(s.strip() for s in str(sides or "").split(",") if s.strip())
    folder = node_folder(pics["name"], f"01_sheet_{role}", {"files": [stamp(f) for f in pics["files"]], "role": role,
                                                             "frames": frames, "ai": ai, **({"what": what} if what else {}),
                                                             **({"sides": sides} if sides else {})})
    out = os.path.join(folder, "sheet.json")
    command = py("pipeline/data_sheet.py", *pics["files"], "--role", role, "--frames", frames, "--out", folder,
                 *(["--what", what] if what else []), *(["--sides", sides] if sides else []))
    return {"folder": folder, "steps": [(command + ([] if ai else ["--no-ai"]), f"01 data sheet ({role})", out)],
            "outputs": {"data": {"path": out, "name": pics["name"], "role": role}, "sheet": os.path.join(folder, "contact.jpg")}}


def sheet_views(data, which):
    sheet = json.load(open(data["path"]))
    paths = {v["name"]: v["path"] for v in sheet["views"]}
    return [project_path(paths[name]) for name in sheet["lists"][which]]


def part_views(data, way="angles", seed=1):
    """01b: a part with one picture only gets its missing sides from the picture AI (views_from_picture.py): from one
    picture TRELLIS.2 guesses the back, and the red book of Nov 19 got a grey-blue painting as its back cover. Writes a
    data sheet of its own (the four views in turning order, the front first), which the part's model and the edit read."""
    front = sheet_views(data, "base")[0]
    what = json.load(open(data["path"])).get("what") or ""
    folder = node_folder(data["name"], "01b_part_views", {"front": stamp(front), "way": way, "seed": seed})
    out = os.path.join(folder, "sheet.json")
    command = py("generation/views_from_picture.py", front, "--way", way, "--seed", seed, "--out", folder, "--sheet", what)
    return {"folder": folder, "steps": [(command, "01b the part's missing sides (picture AI)", out)],
            "outputs": {"data": {**data, "path": out}, "sheet": os.path.join(folder, "overview.jpg")}}


def base_model(data, where="server", mode="multidiffusion", resolution="1024", steps=25, seed=42, colours="all", view=None,
               given=None):
    """02: the object's model from the views its data sheet chose (the main view first).
    colours = "main": only the main view gives the colours, the other views the shape only (every extra view makes
    the colours softer, multidiffusion averages them); "all": every view gives both.
    given: the person's own model (a .glb, from the project folder): it is the base, nothing is generated."""
    if given:
        model = project_path(given)
        if not os.path.isfile(model):
            raise ValueError(f"the given model is not there: {given}")
        folder = node_folder(data["name"], "02_base", {"given": stamp(model)})
        return {"folder": folder, "steps": [], "outputs": {"model": {"name": data["name"], "kind": "base", "label": "your model", "glb": model}}}
    views = sheet_views(data, "base")
    if view is not None:  # only this one of the sheet's views (1 = the main view): free renders from any angle don't
        views = views[int(view) - 1:int(view)]  # fit together, mixed they doubled a robot arm's tool head (Nov 15)
        if not views:
            raise ValueError(f"the data sheet has no view {view}")
    main_only = colours == "main" and len(views) > 1
    folder = node_folder(data["name"], "02_base", {"views": [stamp(v) for v in views], "where": where, "mode": mode,
                                                   "resolution": resolution, "steps": steps, "seed": seed,
                                                   **({"colours": "main"} if main_only else {})})
    like = {"name": data["name"], "kind": "base", "label": f"base seed {seed}"}
    if where == "server":
        done = os.path.join(folder, "server_job.json")
        command = py("server/cloud.py", "generate", *(views[:1] + ["--shape-only", *views[1:]] if main_only else views),
                     "--resolution", resolution, "--steps", steps, "--seed", seed,
                     *(["--mode", mode] if len(views) > 1 else []), "--wait", folder)
        return {"folder": folder, "steps": [(command, "02 base (uni server)", done)],
                "outputs": {"models_from": done, "model_like": like}}
    out = os.path.join(folder, "base.glb")
    script = "generation/multi_image.py" if len(views) > 1 else "generation/generate.py"
    command = py(script, *(views[:1] + ["--shape-only", *views[1:]] if main_only else views),
                 "--resolution", resolution, "--steps", steps, "--seed", seed, "--out", out,
                 *(["--mode", mode] if len(views) > 1 else []))
    return {"folder": folder, "steps": [(command, "02 base (this PC)", out)], "outputs": {"model": {**like, "glb": out}}}


def pick_seed(data, candidates):
    """02b: the measured judge of seeds: the model whose outline fits the input's views best."""
    candidates = [c for c in candidates if c]
    views = sheet_views(data, "base")
    folder = node_folder(data["name"], "02b_pick_seed", {"views": [stamp(v) for v in views],
                                                          "candidates": [stamp(c["glb"]) for c in candidates]})
    out = os.path.join(folder, "pick.json")
    command = py("pipeline/pick_seed.py", "--pictures", *views, "--models", *[c["glb"] for c in candidates], "--out", folder)
    return {"folder": folder, "steps": [(command, "02b pick seed", out)],
            "outputs": {"model_from": out, "candidates": candidates}}


def edit(model, part, obj, replace=False, width=0.75, sink=0.15, head=0.12, fit="between", place=None, side=None, hand=None,
         after=None, prompt=None, prompt_seed=1, at=None):
    """03: where the part goes. The three pictures per side (before, after = a collage, 2D mask) and the 3D area with
    a job for every small cube (keep / delete / edit). The place is measured (make_edit.py), or given by hand (the web
    page): place = the part's left, top, width in the picture of `side`, hand = a folder with the painted corrections
    of the 2D mask (add_<view>.png, erase_<view>.png), after = a ready AFTER picture of `side` (made from a prompt).
    No picture of the part (part = None): prompt = what should be added; the local picture AI paints it onto the
    model's picture and what differs is the part (make_edit.py --prompt).
    at: for a part with pictures, where it goes in words ("in the left hand", "on the back"): the picture AI paints
    the part's own picture there, instead of the measured place on the top. A prompt beside a part's pictures is not used."""
    prompt = " ".join(str(prompt or "").split())
    if not part and not prompt:
        raise ValueError("the edit needs a part: its pictures (the input `part`), or a prompt that says what to add")
    sides, reference = sheet_views(part, "sides") if part else [], sheet_views(obj, "base")[0]
    words = {"prompt": prompt, "prompt_seed": int(prompt_seed)} if prompt and not part else {}
    at = " ".join(str(at or "").split())
    if part and at and not place:  # the part's own picture, placed by the picture AI
        called = json.load(open(part["path"])).get("what") or "new part"
        called = called if called.lower().startswith(("the ", "a ", "an ")) else f"the {called}"
        words = {"at": at, "prompt": f"{called} {at}", "prompt_seed": int(prompt_seed)}
    hand = project_path(hand) if hand else None
    painted = sorted(os.path.join(hand, f) for f in os.listdir(hand) if f.endswith(".png")) if hand and os.path.isdir(hand) else []
    by_hand = {**({"place": [int(v) for v in place]} if place else {}), **({"side": int(side)} if side is not None else {}),
               **({"hand": [stamp(f) for f in painted if os.path.basename(f).startswith(("add_", "erase_"))]} if painted else {}),
               **({"after": stamp(project_path(after))} if after else {})}  # nothing by hand: the folder's name stays as it was
    folder = node_folder(model["name"], "03_edit", {"model": stamp(model["glb"]), "sides": [stamp(s) for s in sides],
                                                    "reference": stamp(reference), "replace": replace, "width": width, "sink": sink,
                                                    **({"head": head} if head != 0.12 else {}), "fit": fit, **by_hand, **{k: v for k, v in words.items() if k != "prompt" or "at" not in words}})
    turning = os.path.join(folder, "part_sides")  # the part's sides in turning order, as make_edit.py wants a folder
    os.makedirs(turning, exist_ok=True)
    for k, side in enumerate(sides[:4]):
        link = os.path.join(turning, f"{k}{os.path.splitext(side)[1]}")
        if not os.path.lexists(link):
            os.symlink(os.path.realpath(side), link)
    out = os.path.join(folder, "edit.json")
    command = py("inpainting/make_edit.py", model["glb"], *([turning] if part else []), *(["--prompt", words["prompt"], "--prompt-seed", words["prompt_seed"]] if words else []),
                 "--reference", reference, "--out", folder,
                 "--width", width, "--sink", sink, "--head", head, "--fit", fit, *(["--replace"] if replace else []),
                 *(["--place", *by_hand["place"]] if place else []), *(["--side", by_hand["side"]] if "side" in by_hand else []),
                 *(["--hand", hand] if painted else []), *(["--after", project_path(after)] if after else []))
    return {"folder": folder, "steps": [(command, "03 edit (before, after, mask, area)", out)],
            "outputs": {"edit": {"folder": folder, "name": model["name"], "original": model,
                                 "object": json.load(open(obj["path"])).get("what") or "an object",
                                 "part": (json.load(open(part["path"])).get("what") if part else prompt) or "a new part"},
                        "sheet": os.path.join(folder, "three.jpg"), "area_map": os.path.join(folder, "area_map.jpg")}}


def place(part_model, edit_, zoom=0.25, size=0.8, sides=4, behind=4, min_voxels=5):
    """03b: the part's own 3D model put where the old part is (place_part.py, docs/FIRST-SUCCESS.md): the edit's front
    mask says where and how big, the old part's plane how it lies. Only a box around the two parts goes on (zoom), and
    the four sides' pictures all show the same part. Its folder is the edit for every step after it; the model the
    inpainting starts from is placed.glb (the details step keeps its voxels), the judges compare with before.glb."""
    model = edit_["original"]
    folder = node_folder(edit_["name"], "03b_place", {"model": stamp(model["glb"]), "part": stamp(part_model["glb"]),
                                                      "edit": stamp(os.path.join(edit_["folder"], "edit.json")),
                                                      "zoom": zoom, "size": size, "sides": sides, "behind": behind,
                                                      "facing": "by colour", "min_voxels": min_voxels})  # Nov 19: measured side, thick enough
    out = os.path.join(folder, "placed.glb")
    done = os.path.join(folder, "edit.json")  # place_part.py writes it last: a run stopped halfway is not taken as done
    command = py("inpainting/place_part.py", model["glb"], part_model["glb"], "--edit", edit_["folder"], "--out", folder,
                 "--zoom", zoom, "--size", size, "--sides", sides, "--behind", behind, "--min-voxels", min_voxels)
    placed = {"name": edit_["name"], "kind": "placed", "label": "the part placed"}
    return {"folder": folder, "steps": [(command, "03b place the part in 3D", done)],
            "outputs": {"edit": {**edit_, "folder": folder, "original": {**placed, "label": "before", "glb": os.path.join(folder, "before.glb")},
                                 "whole": model},
                        "model": {**placed, "glb": out}, "sheet": os.path.join(folder, "three.jpg"),
                        "area_map": os.path.join(folder, "area_map.jpg")}}


def plan_check(edit_, ai=True):
    """03c: the judge before the inpainting (edit_check.py): are the pictures and the 3D area fit to be sent? A plan
    judged redo stops the flow here, before it costs GPU time; passed.json is only written on pass."""
    folder = node_folder(edit_["name"], "03c_check", {"edit": stamp(os.path.join(edit_["folder"], "edit.json")), "ai": ai})
    out = os.path.join(folder, "passed.json")
    command = py("pipeline/edit_check.py", edit_["folder"], "--object", edit_["object"], "--part", edit_["part"],
                 "--out", folder, *([] if ai else ["--no-ai"]))
    return {"folder": folder, "steps": [(command, "03c judge the plan", out)],
            "outputs": {"edit": edit_, "verdict": "pass", "sheet": os.path.join(folder, "check.jpg")}}


def colours_from_part(result, edit_):
    """05b: the details step's result, its new part coloured from the part's own model (place_part.py --part-colours):
    instead of TRELLIS.2's colours step, which guessed the book's back cover grey. No GPU, seconds."""
    folder = node_folder(edit_["name"], "05b_part_colours", {"result": stamp(result["glb"]), "edit": stamp(os.path.join(edit_["folder"], "edit.json"))})
    out = os.path.join(folder, "coloured.glb")
    command = py("inpainting/place_part.py", "--part-colours", result["glb"], "--out", edit_["folder"], "--final", out)
    return {"folder": folder, "steps": [(command, "05b colours from the part's own model", out)],
            "outputs": {"model": {"name": edit_["name"], "kind": "colours", "label": "colours from the part", "glb": out}}}


def finish(result, edit_):
    """06: the inpainted part back into the whole, untouched model (place_part.py --finish): only the box went through
    TRELLIS.2, everything else is the model's own mesh."""
    folder = node_folder(edit_["name"], "06_finish", {"result": stamp(result["glb"]), "edit": stamp(os.path.join(edit_["folder"], "edit.json"))})
    out = os.path.join(folder, f"{edit_['name']}_finished.glb")
    command = py("inpainting/place_part.py", "--finish", result["glb"], "--out", edit_["folder"], "--final", out)
    return {"folder": folder, "steps": [(command, "06 back into the whole model", out)],
            "outputs": {"model": {"name": edit_["name"], "kind": "finished", "label": "finished", "glb": out}}}


def inpaint_step(model, edit_, step="shape", seeds=(1,), where="server"):
    """04: one step of the inpainting, always in the order shape -> details -> colours, each from the result of the
    one before. Several seeds give several candidates in one server job (TRELLIS.2 is loaded once)."""
    here = where == "pc"  # the same job on this PC's GPU (cloud.py edit --here)
    seeds = [int(s) for s in seeds]
    folder = node_folder(edit_["name"], f"04_inpaint_{step}", {"model": stamp(model["glb"]), "seeds": seeds,
                                                                "edit": stamp(os.path.join(edit_["folder"], "edit.json")),
                                                                **({"where": "pc"} if here else {})})
    done = os.path.join(folder, "server_job.json")
    command = py("server/cloud.py", "edit", edit_["folder"], model["glb"], "--seeds", *seeds, "--wait", folder,
                 *({"shape": [], "details": ["--details"], "colours": ["--colours"]}[step]), *(["--here"] if here else []))
    return {"folder": folder, "steps": [(command, f"04 inpaint: {step} ({'this PC' if here else 'uni server'})", done)],
            "outputs": {"models_from": done, "model_like": {"name": edit_["name"], "kind": step, "label": step}}}


def edit_judge(candidates, edit_, step="shape", ai=True, keep_min=85, part_min=70):
    """05: the local judge: measured (is the rest kept, is the part there, the colours) and the local vision AI;
    the best candidate goes on, with the verdict pass / redo."""
    candidates = [c for c in candidates if c]
    folder = node_folder(edit_["name"], f"05_judge_{step}", {"candidates": [stamp(c["glb"]) for c in candidates], "ai": ai,
                                                              "edit": stamp(os.path.join(edit_["folder"], "edit.json")),
                                                              "keep_min": keep_min, "part_min": part_min})
    out = os.path.join(folder, "judge.json")
    command = py("pipeline/edit_judge.py", "--edit", edit_["folder"], "--original", edit_["original"]["glb"], "--candidates",
                 *[c["glb"] for c in candidates], "--out", folder, "--step", step, "--object", edit_["object"],
                 "--part", edit_["part"], "--keep-min", keep_min, "--part-min", part_min, *([] if ai else ["--no-ai"]))
    return {"folder": folder, "steps": [(command, f"05 judge: {step}", out)],
            "outputs": {"model_from": out, "candidates": candidates, "sheet": os.path.join(folder, "compare.jpg")}}
