"""BP Inpainting: the node pipeline (scripts/pipeline/run_pipeline.py) as ComfyUI nodes.

Pictures / video  ->  Sort (data sheet)  ->  Base model  ->  Refine part (x N)  ->  Combine  ->  Hair  ->  Compare
Every node shows what it made (pictures, comparison sheets) and gives its model to ComfyUI's
"Preview 3D" node. ComfyUI only re-runs a node when its inputs or settings change; on top of that each
result is kept in outputs/pipeline/<name>/<node>/<settings hash>/, so it survives a restart.

The heavy work runs in the project's own environment (runner.run), never inside ComfyUI's Python.
Install: link this folder into ComfyUI/custom_nodes (see comfyui/README.md).
"""
import json
import os

from PIL import Image

from . import runner
from .runner import PROJECT, SCRIPTS, file3d, node_folder, project_path, render, run, sheet, stamp, to_image

CATEGORY = "BP Inpainting"
PICTURE_TYPES = (".png", ".jpg", ".jpeg", ".webp")
VIDEO_TYPES = (".mp4", ".mov", ".webm", ".avi", ".mkv", ".m4v")
# the same parts as scripts/pipeline/sort_pictures.py (PARTS)
PARTS = ["head", "hair", "bust", "belly", "back", "butt", "thighs", "lower_legs", "feet"]
MODELS = ["Qwen/Qwen3-VL-8B-Instruct", "Qwen/Qwen3-VL-2B-Instruct", "Qwen/Qwen2.5-VL-7B-Instruct"]


def grid(paths, cell=256, cols=6):
    """All pictures small in one picture (videos are left out)."""
    pics = [p for p in paths if p.lower().endswith(PICTURE_TYPES)][:36] or []
    if not pics:
        return to_image(Image.new("RGB", (cell, cell), (40, 40, 44)))
    rows = [("", pics[i:i + cols]) for i in range(0, len(pics), cols)]
    return to_image(sheet(rows, cell))


def input_files(kind):
    import folder_paths
    folder = folder_paths.get_input_directory()
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    return sorted(folder_paths.filter_files_content_types(files, [kind]))


# ------------------------------------------------------------------ pictures and videos

class Pictures:
    """All pictures (and videos) of one character from a folder."""
    CATEGORY = CATEGORY
    RETURN_TYPES = ("BP_PICTURES", "IMAGE")
    RETURN_NAMES = ("pictures", "preview")
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "name": ("STRING", {"default": "morrigan", "tooltip": "the character's name: outputs/pipeline/<name>"}),
            "folder": ("STRING", {"default": "pictures/local model", "tooltip": "relative to the project folder"}),
        }}

    @classmethod
    def IS_CHANGED(cls, name, folder):
        path = project_path(folder)
        return json.dumps(sorted(stamp(os.path.join(path, f)) for f in os.listdir(path))) if os.path.isdir(path) else ""

    def load(self, name, folder):
        path = project_path(folder)
        if not os.path.isdir(path):
            raise ValueError(f"no folder {path}")
        files = sorted(os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(PICTURE_TYPES + VIDEO_TYPES))
        return ({"name": name.strip().replace(" ", "_"), "files": files}, grid(files))


class AddPicture:
    """One more picture (uploaded into ComfyUI) to the character's pictures."""
    CATEGORY = CATEGORY
    RETURN_TYPES = ("BP_PICTURES", "IMAGE")
    RETURN_NAMES = ("pictures", "preview")
    FUNCTION = "add"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"pictures": ("BP_PICTURES",),
                             "image": (input_files("image"), {"image_upload": True})}}

    def add(self, pictures, image):
        import folder_paths
        path = folder_paths.get_annotated_filepath(image)
        files = pictures["files"] + [path]
        return ({**pictures, "files": files}, grid(files))


class AddVideo:
    """A video (uploaded into ComfyUI): evenly spaced frames become pictures. Which frames are useful views
    is decided later by the vision AI in the Sort node."""
    CATEGORY = CATEGORY
    RETURN_TYPES = ("BP_PICTURES", "IMAGE")
    RETURN_NAMES = ("pictures", "frames")
    FUNCTION = "add"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"pictures": ("BP_PICTURES",),
                             "video": (input_files("video"), {"video_upload": True}),
                             "frames": ("INT", {"default": 12, "min": 2, "max": 64})}}

    def add(self, pictures, video, frames):
        import folder_paths
        path = folder_paths.get_annotated_filepath(video)
        folder = node_folder(pictures["name"], "00_video", {"video": stamp(path), "frames": frames})
        run(["python", os.path.join(SCRIPTS, "pipeline", "video_frames.py"), path, "--frames", str(frames),
             "--out", folder], folder, "video frames", done=os.path.join(folder, "frames"))
        new = sorted(os.path.join(folder, "frames", f) for f in os.listdir(os.path.join(folder, "frames")))
        return ({**pictures, "files": pictures["files"] + new}, grid(new))


# ------------------------------------------------------------------ node 01: the data sheet

class Sort:
    """Node 01: the vision AI sorts the pictures into lists (which picture builds the shape, which only
    gives colours, which part each close-up shows). Shows the data sheet with every view and its tags."""
    CATEGORY = CATEGORY
    RETURN_TYPES = ("BP_SORTED", "IMAGE", "STRING")
    RETURN_NAMES = ("sorted", "data_sheet", "lists")
    FUNCTION = "sort"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"pictures": ("BP_PICTURES",), "model": (MODELS,)}}

    def sort(self, pictures, model):
        files = [f for f in pictures["files"] if f.lower().endswith(PICTURE_TYPES)]
        folder = node_folder(pictures["name"], "01_sort", {"files": [stamp(f) for f in files], "model": model})
        inbox = os.path.join(folder, "in")
        os.makedirs(inbox, exist_ok=True)
        for k, f in enumerate(files):  # one folder with all pictures, as sort_pictures.py wants it
            link = os.path.join(inbox, os.path.basename(f))
            if os.path.lexists(link) and os.path.realpath(link) != os.path.realpath(f):
                link = os.path.join(inbox, f"{k:02d}_{os.path.basename(f)}")
            if not os.path.lexists(link):
                os.symlink(os.path.realpath(f), link)
        out = os.path.join(folder, "sorted.json")
        run(["python", os.path.join(SCRIPTS, "pipeline", "sort_pictures.py"), inbox, "--out", folder, "--model", model],
            folder, "01 sort", done=out)
        result = json.load(open(out))
        short = lambda n: n.replace("deyvson-william-", "")
        lines = [f"{key}: " + "; ".join(f"{role} {', '.join(short(n) for n in names)}"
                                         for role, names in value.items() if isinstance(names, list) and names)
                 for key, value in result["lists"].items() if isinstance(value, dict)]
        lines.append(f"hair colour: {(result.get('hair') or {}).get('hair_colour')}")
        return ({"path": out, "name": pictures["name"]}, to_image(Image.open(os.path.join(folder, "contact.jpg"))),
                "\n".join(lines))


# ------------------------------------------------------------------ node 02: the base model

class BaseModel:
    """Node 02: the whole figure. Shape from the views of one version (e.g. a grey turnaround), colours
    from the whole colour view only (the lessons of rounds 1-3)."""
    CATEGORY = CATEGORY
    RETURN_TYPES = ("BP_MODEL", "IMAGE", "FILE_3D_GLB")
    RETURN_NAMES = ("model", "renders", "model_3d")
    FUNCTION = "make"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "sorted": ("BP_SORTED",),
            "mode": (["stochastic", "multidiffusion"],),
            "resolution": (["1024", "1536", "512"],),
            "steps": ("INT", {"default": 12, "min": 4, "max": 100}),
            "faces": ("INT", {"default": 4_000_000, "min": 100_000, "max": 16_000_000, "step": 100_000}),
            "seed": ("INT", {"default": 42, "min": 0, "max": 2**31, "control_after_generate": True}),
        }}

    def make(self, sorted, mode, resolution, steps, faces, seed):
        result = json.load(open(sorted["path"]))
        views = {v["name"]: v for v in result["views"]}
        lists = result["lists"]["base"]
        path = lambda names: [os.path.join(os.path.dirname(sorted["path"]), views[n]["view"]) for n in names]
        settings = {"sorted": stamp(sorted["path"]), "mode": mode, "resolution": resolution, "steps": steps,
                    "faces": faces, "seed": seed}
        folder = node_folder(sorted["name"], "02_base", settings)
        out = os.path.join(folder, "base.glb")
        command = ["python", os.path.join(SCRIPTS, "generation", "multi_image.py"), *path(lists["both"]),
                   "--mode", mode, "--resolution", resolution, "--steps", str(steps), "--faces", str(faces),
                   "--seed", str(seed), "--out", out]
        if lists["shape_only"]:
            command += ["--shape-only", *path(lists["shape_only"])]
        if lists["colour_only"]:
            command += ["--colour-only", *path(lists["colour_only"])]
        run(command, folder, "02 base", done=out)
        refs = path(lists["shape_only"] + lists["both"] + lists["colour_only"])
        picture = sheet([("pictures (shape only, both, colour only)", refs),
                         (f"base: {mode}, {resolution}, {steps} steps", render(out, os.path.join(folder, "render"))),
                         ("shape alone (clay)", render(out, os.path.join(folder, "clay"), clay=True))])
        picture.save(os.path.join(folder, "compare.jpg"), quality=90)
        model = {"glb": out, "name": sorted["name"], "kind": "base", "label": "base"}
        return (model, to_image(picture), file3d(out))


# ------------------------------------------------------------------ node 04: one part

class RefinePart:
    """Node 04: one part zoomed in and repainted from its pictures. auto = the lessons: the head also gets its
    shape regenerated (margin 0.6 of the head), every other part only its colours (shape kept, margin 0.1 of
    the model); one sharp complete colour picture, only pictures from the part's side."""
    CATEGORY = CATEGORY
    RETURN_TYPES = ("BP_MODEL", "IMAGE", "FILE_3D_GLB")
    RETURN_NAMES = ("part", "compare", "model_3d")
    FUNCTION = "refine"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "base": ("BP_MODEL",),
            "sorted": ("BP_SORTED",),
            "part": (PARTS,),
            "shape": (["auto", "keep (colours only)", "regenerate"],),
            "colour_pictures": ("INT", {"default": 1, "min": 0, "max": 8, "tooltip": "0 = all"}),
            "texture_guidance": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10.0, "step": 0.5}),
            "steps": ("INT", {"default": 25, "min": 4, "max": 100}),
            "seed": ("INT", {"default": 42, "min": 0, "max": 2**31, "control_after_generate": True}),
        }}

    def refine(self, base, sorted, part, shape, colour_pictures, texture_guidance, steps, seed):
        keep = (part != "head") if shape == "auto" else shape.startswith("keep")
        settings = {"base": stamp(base["glb"]), "sorted": stamp(sorted["path"]), "part": part, "keep_shape": keep,
                    "colour_pictures": colour_pictures, "texture_guidance": texture_guidance, "steps": steps, "seed": seed}
        folder = node_folder(sorted["name"], f"04_refine_{part}", settings)
        out = os.path.join(folder, f"{part}.glb")
        command = ["python", os.path.join(SCRIPTS, "inpainting", "refine_region.py"), base["glb"], "--sorted", sorted["path"],
                   "--part", part, "--method", "repaint", "--margin", "0.6", "--steps", str(steps), "--seed", str(seed),
                   "--mode", "stochastic", "--grow", "1", "--texture-guidance", str(texture_guidance),
                   "--faces", "4000000", "--out", out]
        if part != "head":
            command += ["--margin-of-model", "0.1"]
        if keep:
            command += ["--keep-shape"]
        if colour_pictures:
            command += ["--best-colour", str(colour_pictures)]
        run(command, folder, f"04 refine {part}", done=out)
        work = os.path.splitext(out)[0] + "_work"
        info = json.load(open(os.path.join(work, "refine.json")))
        focus = [*info["inner"][0], *info["inner"][1]]
        pictures = [os.path.join(work, os.path.basename(x[0])) for x in info["pictures"]]
        picture = sheet([(f"{part}: pictures ({', '.join(x[1] for x in info['pictures'])})", pictures[:6]),
                         ("before", render(base["glb"], os.path.join(folder, "before"), focus)),
                         (f"after ({'colours only' if keep else 'shape + colours'})", render(out, os.path.join(folder, "after"), focus))])
        picture.save(os.path.join(folder, "compare.jpg"), quality=90)
        model = {"glb": out, "name": sorted["name"], "kind": "part", "part": part, "base": base["glb"], "focus": focus,
                 "label": f"{part} ({'colours' if keep else 'shape+colours'})"}
        return (model, to_image(picture), file3d(out))


# ------------------------------------------------------------------ node 05: all parts together

class Combine:
    """Node 05: the refined parts put into the base model. The first part wins where two overlap."""
    CATEGORY = CATEGORY
    RETURN_TYPES = ("BP_MODEL", "IMAGE", "FILE_3D_GLB")
    RETURN_NAMES = ("model", "renders", "model_3d")
    FUNCTION = "combine"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"base": ("BP_MODEL",)},
                "optional": {f"part_{k}": ("BP_MODEL",) for k in range(1, 9)}}

    def combine(self, base, **parts):
        parts = [parts[f"part_{k}"] for k in range(1, 9) if parts.get(f"part_{k}")]
        settings = {"base": stamp(base["glb"]), "parts": [stamp(p["glb"]) for p in parts]}
        folder = node_folder(base["name"], "05_combine", settings)
        out = os.path.join(folder, "combined.glb")
        run(["python", os.path.join(SCRIPTS, "pipeline", "combine_parts.py"), base["glb"], *[p["glb"] for p in parts],
             "--out", out], folder, "05 combine", done=out)
        picture = sheet([("base", render(base["glb"], os.path.join(folder, "base"))),
                         ("combined: " + ", ".join(p.get("part", "?") for p in parts), render(out, os.path.join(folder, "combined")))])
        picture.save(os.path.join(folder, "compare.jpg"), quality=90)
        model = {"glb": out, "name": base["name"], "kind": "combined", "label": "combined"}
        return (model, to_image(picture), file3d(out))


# ------------------------------------------------------------------ node 06: hair

class Hair:
    """Node 06: real hair strands grown along the hair shell (Blender, Geometry Nodes: Clump, Noise, Profile).
    The hair colour and how far down the hair reaches come from the Sort node."""
    CATEGORY = CATEGORY
    RETURN_TYPES = ("IMAGE", "FILE_3D_GLB", "STRING")
    RETURN_NAMES = ("renders", "model_3d", "blend_file")
    FUNCTION = "hair"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("BP_MODEL",),
            "sorted": ("BP_SORTED",),
            "count": ("INT", {"default": 35000, "min": 1000, "max": 200000, "step": 1000}),
            "length": ("FLOAT", {"default": 0.10, "min": 0.02, "max": 0.5, "step": 0.01}),
            "clump": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05}),
        }, "optional": {"head": ("BP_MODEL", {"tooltip": "the head part: its box is the camera's focus"})}}

    def hair(self, model, sorted, count, length, clump, head=None):
        result = json.load(open(sorted["path"]))
        colour = (result.get("hair") or {}).get("hair_colour")
        if not colour:
            raise ValueError("the Sort node found no hair colour")
        front = next(v for v in result["views"] if v["name"] == result["lists"]["base"]["front"])
        box = front["part_boxes"].get("hair") or front.get("part_boxes_seen", {}).get("hair")
        share = (box[3] - front["box"][1]) / (front["box"][3] - front["box"][1]) + 0.05 if box else 1.0
        settings = {"model": stamp(model["glb"]), "colour": colour, "share": round(share, 3), "count": count,
                    "length": length, "clump": clump, "focus": head.get("focus") if head else None}
        folder = node_folder(model["name"], "06_hair", settings)
        out = os.path.join(folder, "hair")
        command = [runner.BLENDER, "-b", "--python", os.path.join(SCRIPTS, "pipeline", "hair_strands.py"), "--",
                   model["glb"], out, "--hair-colour", colour, "--hair-share", f"{share:.3f}", "--count", str(count),
                   "--length", str(length), "--clump", str(clump), "--glb"]
        if head and head.get("focus"):
            command += ["--focus", *[f"{v:.5f}" for v in head["focus"]]]
        run(command, folder, "06 hair", done=out + ".glb")
        picture = sheet([(f"hair: {count} strands, clump {clump}", [out + f"_{n}.png" for n in ("front", "left", "right", "back")])])
        picture.save(os.path.join(folder, "compare.jpg"), quality=90)
        return (to_image(picture), file3d(out + ".glb"), out + ".blend")


# ------------------------------------------------------------------ 99: side by side

class Compare:
    """Up to 4 models side by side, always the same 4 cameras (front, side, back, side); with clay on, the
    shape alone, without colours (like a marble statue)."""
    CATEGORY = CATEGORY
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("compare",)
    FUNCTION = "compare"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model_1": ("BP_MODEL",), "clay": ("BOOLEAN", {"default": False})},
                "optional": {"model_2": ("BP_MODEL",), "model_3": ("BP_MODEL",), "model_4": ("BP_MODEL",),
                             "reference": ("IMAGE",)}}

    def compare(self, model_1, clay, model_2=None, model_3=None, model_4=None, reference=None):
        rows = []
        for model in (model_1, model_2, model_3, model_4):
            if model:
                base = os.path.splitext(model["glb"])[0] + ("_compare_clay" if clay else "_compare")
                rows.append((model.get("label", model["kind"]), render(model["glb"], base, clay=clay)))
        if reference is not None:
            import numpy as np
            rows.insert(0, ("reference", [Image.fromarray((reference[0].cpu().numpy() * 255).astype(np.uint8))]))
        return (to_image(sheet(rows, 384)),)


NODE_CLASS_MAPPINGS = {
    "BP_Pictures": Pictures, "BP_AddPicture": AddPicture, "BP_AddVideo": AddVideo, "BP_Sort": Sort,
    "BP_BaseModel": BaseModel, "BP_RefinePart": RefinePart, "BP_Combine": Combine, "BP_Hair": Hair,
    "BP_Compare": Compare,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BP_Pictures": "BP 00 Pictures (folder)", "BP_AddPicture": "BP 00 Add picture", "BP_AddVideo": "BP 00 Add video",
    "BP_Sort": "BP 01 Sort pictures (data sheet)", "BP_BaseModel": "BP 02 Base model (TRELLIS.2)",
    "BP_RefinePart": "BP 04 Refine part", "BP_Combine": "BP 05 Combine parts", "BP_Hair": "BP 06 Hair strands",
    "BP_Compare": "BP 99 Compare models",
}
