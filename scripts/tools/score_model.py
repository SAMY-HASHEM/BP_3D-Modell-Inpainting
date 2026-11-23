"""Scores for a model against its pictures (the pipeline's check, at every stage): numbers instead of "looks good".

For each picture: the camera found by the outline (paint_views.py's fit), the model drawn from it with its own
colours (no light: its base colour), and then
  outline     intersection over union of the model's outline and the picture's (the shape)
  colour      the median CIELAB distance between the two where both are (the colours; under ~10 hard to see)
  sharpness   the model's fine detail against the picture's (high frequencies of both, 1.0 = as sharp)
and the mean over the pictures.

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


CAMERAS = {}  # (the shape's point count, size, picture) -> its camera: models that differ only in colour share it


def score(path, pictures, views, ctx):
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
        rows.append({"picture": os.path.basename(pic), "outline": round(float(iou), 3),
                     "colour": round(float(np.median(de[inner])), 1) if inner.any() else None,
                     "sharpness": round(detail(g_model, inner) / max(detail(g_photo, inner), 1e-6), 2)})
    mean = {k: round(float(np.mean([r[k] for r in rows if r[k] is not None])), 3) for k in
            ("outline", "colour", "sharpness")}
    return {"model": path, "mean": mean, "pictures": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("models", nargs="+")
    p.add_argument("--pictures", nargs="+", required=True)
    p.add_argument("--views", nargs="+", required=True)
    p.add_argument("--out")
    a = p.parse_args()
    import nvdiffrast.torch as dr
    ctx = dr.RasterizeCudaContext()
    results = [score(m, a.pictures, a.views, ctx) for m in a.models]
    print(f"{'model':48s} {'outline':>8s} {'colour dE':>10s} {'sharpness':>10s}")
    for r in results:
        m = r["mean"]
        print(f"{os.path.basename(r['model'])[:48]:48s} {m['outline']:8.3f} {m['colour']:10.1f} {m['sharpness']:10.2f}")
    if a.out:
        json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
