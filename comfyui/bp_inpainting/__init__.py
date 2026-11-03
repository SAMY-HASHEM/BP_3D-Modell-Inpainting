"""BP Inpainting: the node pipeline as ComfyUI nodes.

Pictures / video  ->  Sort (data sheet)  ->  Base model  ->  Refine part (x N)  ->  Combine  ->  Hair  ->  Compare
What each node does is in scripts/pipeline/nodes_core.py, shared with the Blender node editor
(blender/pipeline_nodes.py), so both do exactly the same. Here only the ComfyUI side: inputs, settings,
pictures to show, and the model for ComfyUI's "Preview 3D" node. ComfyUI only re-runs a node when its inputs
or settings change; on top of that each result stays in outputs/pipeline/<name>/<node>/<settings hash>/.
Install: link this folder into ComfyUI/custom_nodes (see comfyui/README.md).
"""
import json
import os

from PIL import Image

from .runner import core, file3d, grid, run_job, to_image

CATEGORY = "BP Inpainting"


def input_files(kind):
    import folder_paths
    folder = folder_paths.get_input_directory()
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    return sorted(folder_paths.filter_files_content_types(files, [kind]))


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
        path = core.project_path(folder)
        return json.dumps(sorted(core.stamp(os.path.join(path, f)) for f in os.listdir(path))) if os.path.isdir(path) else ""

    def load(self, name, folder):
        pics = core.pictures(name, folder)
        return (pics, grid(pics["files"]))


class AddPicture:
    """One more picture (uploaded into ComfyUI) to the character's pictures."""
    CATEGORY = CATEGORY
    RETURN_TYPES = ("BP_PICTURES", "IMAGE")
    RETURN_NAMES = ("pictures", "preview")
    FUNCTION = "add"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"pictures": ("BP_PICTURES",), "image": (input_files("image"), {"image_upload": True})}}

    def add(self, pictures, image):
        import folder_paths
        files = pictures["files"] + [folder_paths.get_annotated_filepath(image)]
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
        return {"required": {"pictures": ("BP_PICTURES",), "video": (input_files("video"), {"video_upload": True}),
                             "frames": ("INT", {"default": 12, "min": 2, "max": 64})}}

    def add(self, pictures, video, frames):
        import folder_paths
        job = core.video(pictures, folder_paths.get_annotated_filepath(video), frames)
        run_job(job)
        pics, new = core.video_result(pictures, job)
        return (pics, grid(new))


class Sort:
    """Node 01: the vision AI sorts the pictures into lists (which picture builds the shape, which only gives
    colours, which part each close-up shows). Shows the data sheet with every view and its tags."""
    CATEGORY = CATEGORY
    RETURN_TYPES = ("BP_SORTED", "IMAGE", "STRING")
    RETURN_NAMES = ("sorted", "data_sheet", "lists")
    FUNCTION = "sort"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"pictures": ("BP_PICTURES",), "model": (core.VISION_MODELS,)}}

    def sort(self, pictures, model):
        out = run_job(core.sort(pictures, model))
        return (out["sorted"], to_image(out["sheet"]), core.lists_text(out["sorted"]))


class BaseModel:
    """Node 02: the whole figure (TRELLIS.2). Shape from the views of one version (e.g. a grey turnaround),
    colours from the whole colour view only."""
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
        out = run_job(core.base(sorted, mode, resolution, steps, faces, seed))
        return (out["model"], to_image(out["sheet"]), file3d(out["model"]["glb"]))


class RefinePart:
    """Node 04: one part zoomed in and repainted. auto = the lessons: the head also regenerates its shape from ONE
    picture in total; every other part only its colours (shape kept); one sharp complete colour picture, only
    pictures from the part's side."""
    CATEGORY = CATEGORY
    RETURN_TYPES = ("BP_MODEL", "IMAGE", "FILE_3D_GLB")
    RETURN_NAMES = ("part", "compare", "model_3d")
    FUNCTION = "refine"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "base": ("BP_MODEL",),
            "sorted": ("BP_SORTED",),
            "part": (core.PARTS,),
            "shape": (["auto", "keep (colours only)", "regenerate"],),
            "colour_pictures": ("INT", {"default": 1, "min": 0, "max": 8, "tooltip": "0 = all"}),
            "shape_pictures": (["auto", "none", "all"], {"tooltip": "extra grey / shape-only views; auto: none for the "
                                                                    "head (one picture in total gave the clean face)"}),
            "texture_guidance": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10.0, "step": 0.5}),
            "steps": ("INT", {"default": 25, "min": 4, "max": 100}),
            "seed": ("INT", {"default": 42, "min": 0, "max": 2**31, "control_after_generate": True}),
        }}

    def refine(self, base, sorted, part, shape, colour_pictures, shape_pictures, texture_guidance, steps, seed):
        out = run_job(core.refine(base, sorted, part, shape, colour_pictures, shape_pictures, texture_guidance, steps, seed))
        return (out["model"], to_image(out["sheet"]), file3d(out["model"]["glb"]))


class PickBest:
    """Node 04b: the local vision AI (Qwen3.6-27B) compares candidates of one part (e.g. the head with several
    seeds) with the artist's picture and passes on the best. The face changes a lot with small changes of the
    random start, so one seed is a lottery."""
    CATEGORY = CATEGORY
    RETURN_TYPES = ("BP_MODEL", "IMAGE", "FILE_3D_GLB")
    RETURN_NAMES = ("best", "compare", "model_3d")
    FUNCTION = "pick"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"candidate_1": ("BP_MODEL",), "candidate_2": ("BP_MODEL",)},
                "optional": {f"candidate_{k}": ("BP_MODEL",) for k in range(3, 7)}}

    def pick(self, **candidates):
        found = [candidates[f"candidate_{k}"] for k in range(1, 7) if candidates.get(f"candidate_{k}")]
        out = run_job(core.pick_best(found))
        return (out["model"], to_image(out["sheet"]), file3d(out["model"]["glb"]))


class Combine:
    """Node 05: the refined parts put into the base model. The first part wins where two overlap."""
    CATEGORY = CATEGORY
    RETURN_TYPES = ("BP_MODEL", "IMAGE", "FILE_3D_GLB")
    RETURN_NAMES = ("model", "renders", "model_3d")
    FUNCTION = "combine"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"base": ("BP_MODEL",)}, "optional": {f"part_{k}": ("BP_MODEL",) for k in range(1, 9)}}

    def combine(self, base, **parts):
        parts = [parts[f"part_{k}"] for k in range(1, 9) if parts.get(f"part_{k}")]
        out = run_job(core.combine(base, parts))
        return (out["model"], to_image(out["sheet"]), file3d(out["model"]["glb"]))


class Hair:
    """Node 06: real hair strands grown along the hair shell (Blender Geometry Nodes: Clump, Noise, Profile).
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
        }, "optional": {"head": ("BP_MODEL", {"tooltip": "the head part: its box is the cameras' focus"})}}

    def hair(self, model, sorted, count, length, clump, head=None):
        out = run_job(core.hair(model, sorted, count, length, clump, head))
        row = Image.new("RGB", (4 * 512, 512), (40, 40, 44))
        for k, p in enumerate(out["renders"]):
            if os.path.exists(p):
                row.paste(Image.open(p).convert("RGB").resize((512, 512)), (k * 512, 0))
        return (to_image(row), file3d(out["glb"]), out["blend"])


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
        out = run_job(core.compare([m for m in (model_1, model_2, model_3, model_4) if m], clay))
        picture = Image.open(out["sheet"]).convert("RGB")
        if reference is not None:  # the reference picture on top
            import numpy as np
            ref = Image.fromarray((reference[0].cpu().numpy() * 255).astype(np.uint8))
            ref.thumbnail((picture.width, 512))
            both = Image.new("RGB", (picture.width, picture.height + ref.height), (40, 40, 44))
            both.paste(ref, (0, 0))
            both.paste(picture, (0, ref.height))
            picture = both
        return (to_image(picture),)


NODE_CLASS_MAPPINGS = {
    "BP_Pictures": Pictures, "BP_AddPicture": AddPicture, "BP_AddVideo": AddVideo, "BP_Sort": Sort,
    "BP_BaseModel": BaseModel, "BP_RefinePart": RefinePart, "BP_PickBest": PickBest, "BP_Combine": Combine, "BP_Hair": Hair,
    "BP_Compare": Compare,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "BP_Pictures": "BP 00 Pictures (folder)", "BP_AddPicture": "BP 00 Add picture", "BP_AddVideo": "BP 00 Add video",
    "BP_Sort": "BP 01 Sort pictures (data sheet)", "BP_BaseModel": "BP 02 Base model (TRELLIS.2)",
    "BP_RefinePart": "BP 04 Refine part", "BP_PickBest": "BP 04b Pick best (local AI)", "BP_Combine": "BP 05 Combine parts", "BP_Hair": "BP 06 Hair strands",
    "BP_Compare": "BP 99 Compare models",
}
