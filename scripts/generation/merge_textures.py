"""Two textures of one mesh combined (pipeline stage 7): the colours of the faithful one (e.g. the model's own colours
from de-lit views, N4) with the crisp detail of the vivid one (e.g. MV-Adapter's, which paints sharp but guesses
colours it can't see: the motorcycle girl's helmet came out green, Nov 23).

Per texel of the vivid one's texture (both meshes the same, face for face; MV-Adapter only gives it new UVs):
  low      both textures' large-scale colours: averaged in 3D cells (--cell, 1.5 % of the model) with their
           neighbours (not blurred on the image: a texture's UV islands lie anywhere)
  agree    how close the two large-scale colours are (CIELAB distance; full up to --agree, none from 2 x --agree)
  result   faithful_low + vivid_detail + agree x (vivid_low - faithful_low)
           = the vivid texture where they agree, the faithful colours with the vivid one's detail where they don't

    python scripts/generation/merge_textures.py faithful.glb vivid.glb --out merged.glb [--cell 0.015] [--agree 14]

Writes merged.glb (the vivid one's mesh and UVs) and merged_agree.png (white: the vivid colours kept).
"""
import argparse
import os

import numpy as np
import torch
import trimesh
from PIL import Image


def load(path):
    scene = trimesh.load(path)
    mesh = list(scene.geometry.values())[0]
    tex = np.asarray(mesh.visual.material.baseColorTexture.convert("RGB")).astype(np.float32) / 255
    return scene, mesh, tex


def to_lab(rgb):
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = lin @ np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]).T
    xyz /= np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def blur(image, sigma):
    import cv2
    return cv2.GaussianBlur(image, (0, 0), sigma)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("faithful")
    p.add_argument("vivid")
    p.add_argument("--out", required=True)
    p.add_argument("--cell", type=float, default=0.015, help="the 3D cell for the large-scale colours, of the "
                   "model's size")
    p.add_argument("--agree", type=float, default=14, help="CIELAB distance up to which the vivid colours are kept")
    p.add_argument("--check", action="store_true", help="write only the faithful colours on the vivid mesh (a test "
                   "of the mapping: it must look like the faithful model)")
    a = p.parse_args()
    import nvdiffrast.torch as dr

    _, mesh_f, tex_f = load(a.faithful)
    scene_v, mesh_v, tex_v = load(a.vivid)
    assert len(mesh_f.faces) == len(mesh_v.faces), "not the same mesh (face counts differ)"
    size = tex_v.shape[0]

    # every texel of the vivid texture: its face and place in it (rasterised in its UV space)
    ctx = dr.RasterizeCudaContext()
    uv = mesh_v.visual.uv
    clip = np.stack([uv[:, 0] * 2 - 1, uv[:, 1] * 2 - 1, np.zeros(len(uv)), np.ones(len(uv))], 1)
    faces_v = torch.as_tensor(np.ascontiguousarray(mesh_v.faces), dtype=torch.int32, device="cuda")
    rast, _ = dr.rasterize(ctx, torch.as_tensor(clip, dtype=torch.float32, device="cuda")[None], faces_v,
                           (size, size))
    rast = rast[0].flip(0).cpu().numpy()  # texture rows top down
    covered = rast[..., 3] > 0
    face = rast[..., 3][covered].astype(np.int64) - 1
    b0, b1 = rast[..., 0][covered], rast[..., 1][covered]  # nvdiffrast: u, v weigh the face's corners 0 and 1
    b2 = 1 - b0 - b1

    corners = mesh_v.vertices[mesh_v.faces[face]]  # (n, 3, 3): every texel's place on the mesh
    place = b0[:, None] * corners[:, 0] + b1[:, None] * corners[:, 1] + b2[:, None] * corners[:, 2]

    # the same face, the same place, in the faithful mesh's UVs -> its colour there
    uv_f = mesh_f.visual.uv[mesh_f.faces[face]]  # (n, 3, 2)
    at = b0[:, None] * uv_f[:, 0] + b1[:, None] * uv_f[:, 1] + b2[:, None] * uv_f[:, 2]
    h, w = tex_f.shape[:2]
    x = np.clip(at[:, 0] * (w - 1), 0, w - 1)
    y = np.clip((1 - at[:, 1]) * (h - 1), 0, h - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    faithful = np.zeros_like(tex_v)
    faithful[covered] = ((tex_f[y0, x0] * (1 - fx) + tex_f[y0, x1] * fx) * (1 - fy)
                         + (tex_f[y1, x0] * (1 - fx) + tex_f[y1, x1] * fx) * fy)
    faithful[~covered] = tex_v[~covered]

    if a.check:
        mesh_v.visual.material.baseColorTexture = Image.fromarray((np.clip(faithful, 0, 1) * 255).astype(np.uint8))
        scene_v.export(a.out)
        print("mapping check:", a.out)
        return
    # the large-scale colours: averaged in 3D cells (--cell of the model's size) and their neighbours, not by
    # blurring the texture image (MV-Adapter's UV layout is thousands of small islands: a blur there mixed far-apart
    # parts of the model and the helmet stayed green, Nov 23)
    cell = a.cell * float(np.ptp(mesh_v.vertices, axis=0).max())
    key = np.floor(place / cell).astype(np.int64)
    key -= key.min(0)
    dims = key.max(0) + 3
    def cell_mean(colours):
        flat = (key[:, 0] + 1) * dims[1] * dims[2] + (key[:, 1] + 1) * dims[2] + (key[:, 2] + 1)
        total = np.zeros((dims.prod(), 3)), np.zeros(dims.prod())
        sums, counts = total
        for dx in (-1, 0, 1):  # each cell with its 26 neighbours (a box of 3 x 3 x 3 cells)
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    shifted = flat + dx * dims[1] * dims[2] + dy * dims[2] + dz
                    counts += np.bincount(shifted, minlength=len(counts))
                    for ch in range(3):
                        sums[:, ch] += np.bincount(shifted, weights=colours[:, ch], minlength=len(counts))
        return sums[flat] / np.maximum(counts[flat], 1)[:, None]
    f_c, v_c = faithful[covered], tex_v[covered]
    low_f, low_v = cell_mean(f_c), cell_mean(v_c)
    dist = np.linalg.norm(to_lab(np.clip(low_f, 0, 1)) - to_lab(np.clip(low_v, 0, 1)), axis=-1)
    agree = np.clip((2 * a.agree - dist) / a.agree, 0, 1)
    merged = tex_v.copy()
    merged[covered] = low_f + (v_c - low_v) + agree[:, None] * (low_v - low_f)
    agree_image = np.ones(tex_v.shape[:2], np.float32)
    agree_image[covered] = agree
    print(f"vivid colours kept on {agree.mean():.0%} of the surface (CIELAB distance median "
          f"{np.median(dist):.1f})", flush=True)

    mesh_v.visual.material.baseColorTexture = Image.fromarray((np.clip(merged, 0, 1) * 255).astype(np.uint8))
    scene_v.export(a.out)
    Image.fromarray((agree_image * 255).astype(np.uint8)).save(os.path.splitext(a.out)[0] + "_agree.png")
    print("merged:", a.out)


if __name__ == "__main__":
    main()
