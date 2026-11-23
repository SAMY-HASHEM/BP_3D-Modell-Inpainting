"""Scores for a model against its pictures (the pipeline's check, at every stage): numbers instead of "looks good".

For each picture: the camera found by the outline (paint_views.py's fit), the model drawn from it with its own
colours (no light: its base colour), and then
  outline     intersection over union of the model's outline and the picture's (the shape)
  colour      the median CIELAB distance between the two where both are (the colours; under ~10 hard to see)
  sharpness   the model's fine detail against the picture's (high frequencies of both, 1.0 = as sharp)
and the mean over the pictures. --face face: also how alike that part is (on the first picture: found by
Grounding DINO + SAM, the same box of the model's picture, compared by DINOv3; 1.0 = the same): the seed search
picked Erik's cleanest bust, not his likest face (Nov 23).

    python scripts/tools/score_model.py model.glb [more.glb ...] --pictures front.png side.png ... \\
        --views 0 side [...] [--out scores.json]

The pictures cut out (RGBA), ideally de-lit (the model's base colour has no light either). Prints a table.
Fair only for views the colours were not painted from: the motorcycle girl's photo painter (paint_views.py) scored
best on colour (16.3 against 23-27) on the very pictures it had projected, and ghosted from every other side.

The motorcycle girl, Nov 23 (front, side, back): N4 (de-lit colours) outline 0.841, colour 23.3, sharpness 0.74;
the merge with MV-Adapter 0.841, 23.2, 1.45: its colours as faithful, twice as sharp.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import trimesh
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "generation"))
from paint_views import Silhouettes, find_camera  # noqa: E402
from merge_textures import to_lab  # noqa: E402


def draw(mesh, texture, cam, size, ctx):
    import nvdiffrast.torch as dr
    H, W = size
    xy, depth = cam.project(mesh.vertices)
    near, far = depth.min() * 0.5, depth.max() * 2
    z = (depth - near) / (far - near) * 2 - 1
    clip = np.stack([(xy[:, 0] / W * 2 - 1) * depth, (1 - xy[:, 1] / H * 2) * depth, z * depth, depth], 1)
    faces = torch.as_tensor(np.ascontiguousarray(mesh.faces), dtype=torch.int32, device="cuda")
    rast, _ = dr.rasterize(ctx, torch.as_tensor(clip, dtype=torch.float32, device="cuda")[None], faces, (H, W))
    uv = mesh.visual.uv
    tuv, _ = dr.interpolate(torch.as_tensor(np.stack([uv[:, 0], 1 - uv[:, 1]], 1), dtype=torch.float32,
                                            device="cuda"), rast, faces)
    colour = dr.texture(torch.as_tensor(texture[None], dtype=torch.float32, device="cuda"), tuv,
                        filter_mode="linear")[0].flip(0).cpu().numpy()
    return colour, (rast[0, ..., 3] > 0).flip(0).cpu().numpy()


def detail(gray, mask):
    import cv2
    high = gray - cv2.GaussianBlur(gray, (0, 0), 2)
    return float(np.abs(high[mask]).mean()) if mask.any() else 0.0


BOXES = {}
CAMERAS = {}  # (the shape's point count, size, picture) -> its camera: models that differ only in colour share it


PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_dino = {}


def dino_embed(image):
    """DINOv3's summary of a picture (its CLS token), for comparing two pictures of the same thing."""
    import glob
    from transformers import AutoImageProcessor, AutoModel
    if not _dino:
        path = glob.glob(os.path.join(PROJECT, "local_ai_models", "models--facebook--dinov3-vitl16-pretrain-lvd1689m",
                                      "snapshots", "*"))[0]
        _dino["proc"] = AutoImageProcessor.from_pretrained(path)
        _dino["model"] = AutoModel.from_pretrained(path).cuda().eval()
    inputs = _dino["proc"](images=image, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = _dino["model"](**inputs).last_hidden_state[:, 0]
    return torch.nn.functional.normalize(out, dim=-1)[0]


def find_box(picture_path, word):
    """The box around `word` in a picture (find_part.py: Grounding DINO + SAM 2.1, its own environment)."""
    import json as js
    import shutil
    import subprocess
    import tempfile
    folder = tempfile.mkdtemp()
    shutil.copy(picture_path, os.path.join(folder, "view.png"))
    js.dump({"word": word, "views": [{"picture": "view.png", "clicks": []}]}, open(os.path.join(folder, "views.json"), "w"))
    subprocess.run([os.path.join(PROJECT, "conda", "envs", "mask-ai", "bin", "python"),
                    os.path.join(PROJECT, "scripts", "inpainting", "find_part.py"), folder], capture_output=True)
    try:
        m = np.load(os.path.join(folder, "masks.npz"))["view"]
    except Exception:
        return None
    ys, xs = np.nonzero(m)
    return (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1) if len(xs) else None


def score(path, pictures, views, ctx, face=None):
    import cv2
    scene = trimesh.load(path)
    mesh = list(scene.geometry.values())[0]
    to_world = scene.graph.get(scene.graph.nodes_geometry[0])[0]
    mesh.vertices = trimesh.transform_points(mesh.vertices, to_world)
    texture = np.asarray(mesh.visual.material.baseColorTexture.convert("RGB")).astype(np.float32) / 255
    lo, hi = np.percentile(mesh.vertices, 0.5, axis=0), np.percentile(mesh.vertices, 99.5, axis=0)
    sil = Silhouettes(mesh.vertices, mesh.faces)
    rows = []
    for pic, view in zip(pictures, views):
        rgba = np.asarray(Image.open(pic).convert("RGBA"))
        mask = rgba[..., 3] > 127
        key = (len(mesh.faces), tuple(np.round(hi - lo, 4)), pic)
        if key not in CAMERAS:
            CAMERAS[key] = find_camera(sil, mask, view, float(np.max(hi - lo)) / 2, (lo + hi) / 2)
        cam, iou = CAMERAS[key]
        colour, seen = draw(mesh, texture, cam, mask.shape, ctx)
        both = mask & seen
        inner = cv2.erode(both.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0  # off the outline's edge
        photo = rgba[..., :3].astype(np.float32) / 255
        de = np.linalg.norm(to_lab(colour) - to_lab(photo), axis=-1)
        g_model, g_photo = colour.mean(-1), photo.mean(-1)
        row = {"picture": os.path.basename(pic), "outline": round(float(iou), 3),
               "colour": round(float(np.median(de[inner])), 1) if inner.any() else None,
               "sharpness": round(detail(g_model, inner) / max(detail(g_photo, inner), 1e-6), 2)}
        if face and pic == pictures[0]:  # the face (or another part) alone: how alike, by DINOv3's eyes
            key = (pic, face)
            if key not in BOXES:
                BOXES[key] = find_box(pic, face)
            box = BOXES[key]
            if box is not None:
                x0, y0, x1, y1 = box
                grey = np.full_like(photo, 0.5)
                a_photo = np.where(mask[..., None], photo, grey)[y0:y1, x0:x1]
                a_model = np.where(seen[..., None], colour, grey)[y0:y1, x0:x1]
                to_pil = lambda x: Image.fromarray((np.clip(x, 0, 1) * 255).astype(np.uint8))
                row["face"] = round(float((dino_embed(to_pil(a_photo)) * dino_embed(to_pil(a_model))).sum()), 3)
        rows.append(row)
    mean = {k: round(float(np.mean([r[k] for r in rows if r.get(k) is not None])), 3) for k in
            ("outline", "colour", "sharpness") + (("face",) if any("face" in r for r in rows) else ())}
    return {"model": path, "mean": mean, "pictures": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("models", nargs="+")
    p.add_argument("--pictures", nargs="+", required=True)
    p.add_argument("--views", nargs="+", required=True)
    p.add_argument("--out")
    p.add_argument("--face", help="also how alike one part is (e.g. face): found on the first picture, compared by "
                   "DINOv3 (1.0 = the same)")
    a = p.parse_args()
    import nvdiffrast.torch as dr
    ctx = dr.RasterizeCudaContext()
    results = [score(m, a.pictures, a.views, ctx, a.face) for m in a.models]
    print(f"{'model':48s} {'outline':>8s} {'colour dE':>10s} {'sharpness':>10s}" + (f" {a.face:>8s}" if a.face else ""))
    for r in results:
        m = r["mean"]
        print(f"{os.path.basename(r['model'])[:48]:48s} {m['outline']:8.3f} {m['colour']:10.1f} {m['sharpness']:10.2f}"
              + (f" {m.get('face', float('nan')):8.3f}" if a.face else ""))
    if a.out:
        json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
