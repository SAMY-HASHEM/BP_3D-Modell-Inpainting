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
OUTPUTS = os.path.join(PROJECT, "outputs", "pipeline")
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


def node_folder(name, node, settings):
    digest = hashlib.sha1(json.dumps(settings, sort_keys=True, default=str).encode()).hexdigest()[:8]
    folder = os.path.join(OUTPUTS, name, node, digest)
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
    return {"folder": folder, "steps": [(py("pipeline/video_frames.py", path, "--frames", frames, "--out", folder),
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
    """Outputs that are only known after the run (pick_best: which candidate won)."""
    if "model_from" in outputs:
        best = json.load(open(outputs["model_from"]))["best"]
        outputs = {**outputs, "model": next(c for c in outputs["candidates"] if c["glb"] == best)}
    return outputs
