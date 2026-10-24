"""Render one or more .glb models from 4 sides into one comparison picture (one row per model).

Usage (TRELLIS.2 environment):
    source activate.sh
    python scripts/render_views.py compare.jpg outputs/a.glb outputs/b.glb
    python scripts/render_views.py parts.jpg outputs/yeonhee_parts18.glb --by-part   # one colour per part

Lit with a simple light from the front-top; textures are shown if the meshes have them.
"""
import argparse
import math

import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image

from project_texture import DEVICE, make_camera, rotation, to_clip

PALETTE = np.array([[230, 90, 90], [90, 160, 230], [90, 220, 140], [220, 140, 230], [240, 200, 80],
                    [120, 220, 230], [200, 180, 120], [250, 150, 90], [150, 120, 230], [160, 230, 90]]) / 255


def load(path):
    """All meshes of a .glb as (vertices, faces, uvs or None, texture or None, vertex colours or None)."""
    scene = trimesh.load(path)
    meshes = []
    for node in scene.graph.nodes_geometry:
        transform, name = scene.graph[node]
        g = scene.geometry[name]
        verts = trimesh.transform_points(g.vertices, transform)
        uv = texture = colors = None
        material = getattr(g.visual, "material", None)
        if getattr(g.visual, "uv", None) is not None and getattr(material, "baseColorTexture", None) is not None:
            uv = np.asarray(g.visual.uv, np.float32).copy()
            uv[:, 1] = 1 - uv[:, 1]
            texture = np.asarray(material.baseColorTexture.convert("RGB"), np.float32) / 255
        elif hasattr(g.visual, "vertex_colors"):
            colors = np.asarray(g.visual.vertex_colors)[:, :3] / 255
        meshes.append((verts, np.asarray(g.faces), uv, texture, colors))
    return meshes


def render(ctx, meshes, yaw, size, by_part):
    return render_view(ctx, meshes, yaw, 10, size, by_part)[0]


def render_view(ctx, meshes, yaw, pitch, size, by_part=False, background=0.25):
    """One view (angles in degrees): the picture (uint8), where the model covers it, and the camera."""
    verts_all = np.concatenate([m[0] for m in meshes])
    center = torch.tensor((verts_all.min(0) + verts_all.max(0)) / 2, device=DEVICE, dtype=torch.float32)
    radius = float(np.linalg.norm(np.ptp(verts_all, 0)) / 2)
    cam = make_camera(center, radius, math.radians(yaw), math.radians(pitch), focal=2.2)
    light = F.normalize(torch.tensor([0.4, 0.6, 1.0], device=DEVICE), dim=0)
    image = torch.full((size, size, 3), background, device=DEVICE)
    best = torch.full((size, size), float("inf"), device=DEVICE)
    for k, (v, f, uv, texture, colors) in enumerate(meshes):
        verts = torch.tensor(v, device=DEVICE, dtype=torch.float32)
        faces = torch.tensor(f, device=DEVICE, dtype=torch.int32)
        clip = to_clip(verts, cam, size, size)[None]
        rast, _ = dr.rasterize(ctx, clip, faces, (size, size))
        if by_part:
            color = torch.tensor(PALETTE[k % len(PALETTE)], device=DEVICE, dtype=torch.float32).expand(size, size, 3)
        elif texture is not None:
            uvs, _ = dr.interpolate(torch.tensor(uv, device=DEVICE)[None], rast, faces)
            color = dr.texture(torch.tensor(texture, device=DEVICE)[None], uvs, filter_mode="linear")[0]
        elif colors is not None:
            color = dr.interpolate(torch.tensor(colors, device=DEVICE, dtype=torch.float32)[None].contiguous(), rast, faces)[0][0]
        else:
            color = torch.full((size, size, 3), 0.8, device=DEVICE)
        tri = verts[faces.long()]
        normal = F.normalize(torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=1), dim=1)
        normal = normal @ rotation(cam["yaw"], cam["pitch"], cam["roll"]).T
        face = rast[0, ..., 3].long() - 1
        shade = 0.4 + 0.6 * (normal[face.clamp(min=0)] @ light).abs()
        z = torch.where(rast[0, ..., 3] > 0, rast[0, ..., 2], torch.full_like(best, float("inf")))
        nearer = z < best
        image = torch.where(nearer[..., None], color * shade[..., None], image)
        best = torch.where(nearer, z, best)
    picture = (image.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    return picture, torch.isfinite(best).cpu().numpy(), cam


def main():
    parser = argparse.ArgumentParser(description="Render .glb models from 4 sides")
    parser.add_argument("out", help="picture to write (.jpg or .png)")
    parser.add_argument("models", nargs="+")
    parser.add_argument("--by-part", action="store_true", help="one colour per mesh instead of the texture")
    parser.add_argument("--size", type=int, default=512, help="size of each view in pixels")
    args = parser.parse_args()
    ctx = dr.RasterizeCudaContext()
    rows = []
    for path in args.models:
        meshes = load(path)
        rows.append(np.concatenate([render(ctx, meshes, yaw, args.size, args.by_part) for yaw in (0, 90, 180, 270)], 1))
        print(f"    {path}: {len(meshes)} mesh(es), {sum(len(m[1]) for m in meshes):,} faces")
    Image.fromarray(np.concatenate(rows, 0)).save(args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
