"""The picture's own fine surface detail onto the model (pipeline stage 8b): Marigold (ETH Zurich, OpenRAIL++) reads
the surface direction at every pixel of the picture; where the model is seen from that picture's camera, the
picture's FINE detail (its normals minus their blur: creases, engravings, folds) is added to the model's normals.
The big shape stays the model's own: a camera a few pixels off would otherwise bend whole areas.

Unlike stage 8 (detail_normals.py: a material's generic pattern), this is the detail of THIS object, where the
picture sees it (the front of the model it was made from: there the camera fits best).

    conda/envs/mvadapter-venv/bin/python scripts/generation/picture_normals.py model.glb --picture front.png \\
        --out detailed.glb [--view 0] [--strength 0.8] [--fine 3]

Runs in MV-Adapter's environment (diffusers); the model in local_ai_models/marigold_normals (2.6 GB, fp16).
The model's existing normal map (stage 8) is kept and the picture's detail added on top. Writes detailed.glb and
detailed_picture_normals.png (what Marigold saw).
"""
import argparse
import math
import os
import sys

import numpy as np
import torch
import trimesh
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(HERE))
MARIGOLD = os.path.join(PROJECT, "local_ai_models", "marigold_normals")
sys.path.insert(0, HERE)
os.environ["HF_HUB_OFFLINE"] = "1"


def marigold_normals(picture):
    """Per pixel the surface direction, camera axes (x right, y up, z towards the viewer), as Marigold gives it."""
    from diffusers import MarigoldNormalsPipeline
    pipe = MarigoldNormalsPipeline.from_pretrained(MARIGOLD, variant="fp16", torch_dtype=torch.float16).to("cuda")
    rgb = Image.new("RGB", picture.size, (127, 127, 127))
    rgb.paste(picture, mask=picture.getchannel("A"))
    out = pipe(rgb, num_inference_steps=4, ensemble_size=3, processing_resolution=768,
               match_input_resolution=True)
    n = np.asarray(out.prediction[0], np.float32)  # H x W x 3 in [-1, 1]
    del pipe
    torch.cuda.empty_cache()
    return n / np.linalg.norm(n, axis=-1, keepdims=True).clip(1e-6)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("model")
    p.add_argument("--picture", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--view", default="0", help="the picture's rough angle (paint_views.py)")
    p.add_argument("--strength", type=float, default=0.8)
    p.add_argument("--fine", type=float, default=3, help="how fine the detail kept is (blur sigma, px)")
    a = p.parse_args()
    import cv2
    import nvdiffrast.torch as dr
    from paint_views import Silhouettes, find_camera, line_up_on

    picture = Image.open(a.picture).convert("RGBA")
    rgba = np.asarray(picture)
    mask = rgba[..., 3] > 127
    H, W = mask.shape
    normals = marigold_normals(picture)
    Image.fromarray(((normals * 0.5 + 0.5) * 255).astype(np.uint8)).save(
        os.path.splitext(a.out)[0] + "_picture_normals.png")
    detail = normals - np.stack([cv2.GaussianBlur(normals[..., k], (0, 0), a.fine) for k in range(3)], -1)

    scene = trimesh.load(a.model)
    mesh = list(scene.geometry.values())[0]
    to_world = scene.graph.get(scene.graph.nodes_geometry[0])[0]
    verts = trimesh.transform_points(mesh.vertices, to_world)
    vnorm = mesh.vertex_normals @ to_world[:3, :3].T
    lo, hi = np.percentile(verts, 0.5, axis=0), np.percentile(verts, 99.5, axis=0)
    size, centre = float(np.max(hi - lo)) / 2, (lo + hi) / 2
    cam, score = find_camera(Silhouettes(verts, mesh.faces), mask, a.view, size, centre)
    print(f"camera: azimuth {cam.azimuth:.1f}, elevation {cam.elevation:.1f}: outline overlap {score:.1%}", flush=True)

    ctx = dr.RasterizeCudaContext()
    cuda = lambda x, t=torch.float32: torch.as_tensor(np.ascontiguousarray(x), dtype=t, device="cuda")
    faces = cuda(mesh.faces, torch.int32)
    material = mesh.visual.material
    texture = np.asarray(material.baseColorTexture.convert("RGB")).astype(np.float32) / 255
    th, tw = texture.shape[:2]
    uv = mesh.visual.uv

    # the picture lined up on the model's own colours (a few pixels: the detail is fine)
    xy, depth = cam.project(verts)
    near, far = depth.min() * 0.5, depth.max() * 2
    zc = (depth - near) / (far - near) * 2 - 1
    clip = np.stack([(xy[:, 0] / W * 2 - 1) * depth, (1 - xy[:, 1] / H * 2) * depth, zc * depth, depth], 1)
    rast, _ = dr.rasterize(ctx, cuda(clip)[None], faces, (H, W))
    depth_map, _ = dr.interpolate(cuda(depth[:, None]), rast, faces)
    depth_map = torch.where(rast[..., 3:] > 0, depth_map, torch.full_like(depth_map, 1e9))[0, ..., 0].flip(0)
    depth_map = depth_map.cpu().numpy()
    tuv, _ = dr.interpolate(cuda(np.stack([uv[:, 0], 1 - uv[:, 1]], 1)), rast, faces)
    own = dr.texture(cuda(texture[None]), tuv, filter_mode="linear")[0].flip(0).cpu().numpy()
    own_mask = (rast[0, ..., 3] > 0).flip(0).cpu().numpy()
    warp, match = line_up_on(own, own_mask & mask, rgba[..., :3].astype(np.float32) / 255, shift_only=True,
                             limit=0.03 * max(H, W))
    if warp is not None:
        detail = np.stack([cv2.warpAffine(detail[..., k], warp, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
                           for k in range(3)], -1)
    print(f"lined up: {None if warp is None else np.round(warp[:, 2], 1).tolist()} px", flush=True)

    # every texel: place, normal, tangent frame; its existing normal map (stage 8) decoded
    clip_uv = np.stack([uv[:, 0] * 2 - 1, uv[:, 1] * 2 - 1, np.zeros(len(uv)), np.ones(len(uv))], 1)
    rast_uv, _ = dr.rasterize(ctx, cuda(clip_uv)[None], faces, (th, tw))
    place, _ = dr.interpolate(cuda(verts), rast_uv, faces)
    nrm, _ = dr.interpolate(cuda(vnorm), rast_uv, faces)
    rast_uv = rast_uv[0].flip(0).cpu().numpy()
    covered = rast_uv[..., 3] > 0
    face = rast_uv[..., 3][covered].astype(np.int64) - 1
    P = place[0].flip(0).cpu().numpy()[covered]
    N = nrm[0].flip(0).cpu().numpy()[covered]
    N /= np.linalg.norm(N, axis=1, keepdims=True).clip(1e-9)
    tri, tuvf = verts[mesh.faces], uv[mesh.faces]
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    d1, d2 = tuvf[:, 1] - tuvf[:, 0], tuvf[:, 2] - tuvf[:, 0]
    det = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    r = 1.0 / np.where(np.abs(det) < 1e-12, 1e-12, det)
    Tf = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) * r[:, None]
    Bf = (e2 * d1[:, 0:1] - e1 * d2[:, 0:1]) * r[:, None]
    T = Tf[face] - N * (Tf[face] * N).sum(1, keepdims=True)
    T /= np.linalg.norm(T, axis=1, keepdims=True).clip(1e-9)
    hand = np.sign((np.cross(N, T) * Bf[face]).sum(1, keepdims=True))
    hand[hand == 0] = 1
    B = np.cross(N, T) * hand
    current = N.copy()
    if getattr(material, "normalTexture", None) is not None:
        nm = np.asarray(material.normalTexture.convert("RGB").resize((tw, th)), np.float32)[covered] / 127.5 - 1
        current = nm[:, 0:1] * T + nm[:, 1:2] * B + nm[:, 2:3] * N
        current /= np.linalg.norm(current, axis=1, keepdims=True).clip(1e-9)

    # seen from the picture's camera? how straight?
    pxy, pdepth = cam.project(P)
    inside = (pxy[:, 0] >= 0) & (pxy[:, 1] >= 0) & (pxy[:, 0] < W - 1) & (pxy[:, 1] < H - 1)
    px = np.clip(pxy.astype(int), 0, [W - 1, H - 1])
    seen = inside & (pdepth <= depth_map[px[:, 1], px[:, 0]] + 0.01 * size) & mask[px[:, 1], px[:, 0]]
    eye = cam.centre + cam.R.T @ np.array([0, 0, cam.distance])
    look = eye - P
    look /= np.linalg.norm(look, axis=1, keepdims=True)
    straight = np.clip(((N * look).sum(1) - math.cos(math.radians(70))) / 0.4, 0, 1) * seen
    d_cam = detail[px[:, 1], px[:, 0]]  # camera axes: x right, y up, z to the viewer
    d_world = d_cam @ cam.R  # camera -> model (R maps model to camera; its inverse is its transpose)
    d_world -= (d_world * current).sum(1, keepdims=True) * current
    new = current + a.strength * straight[:, None] * d_world
    new /= np.linalg.norm(new, axis=1, keepdims=True).clip(1e-9)
    print(f"picture detail on {(straight > 0.5).mean():.1%} of the texels (match {match:.2f})", flush=True)

    ts = np.stack([(new * T).sum(1), (new * B).sum(1), (new * N).sum(1)], 1)
    out_map = np.full((th, tw, 3), (0.5, 0.5, 1.0), np.float32)
    out_map[covered] = ts * 0.5 + 0.5
    material.normalTexture = Image.fromarray((np.clip(out_map, 0, 1) * 255).astype(np.uint8))
    scene.export(a.out)
    print("detailed:", a.out)


if __name__ == "__main__":
    main()
