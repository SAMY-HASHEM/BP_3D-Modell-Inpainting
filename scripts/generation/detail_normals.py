"""Surface detail by material (pipeline stage 8): fine detail normals laid onto a model part by part, as artists do it
(an alpha and noise for the skin, a fabric's weave for cloth, a detail normal map per material: Marmoset, Substance).

  1 parts     the model drawn from 8 sides (each pixel remembers its texel); each material word ("skin", "cloth",
              ...) found on the pictures (find_part.py: Grounding DINO + SAM 2.1, its own environment); every texel
              takes the material most of the pictures that see it agree on (none: no detail)
  2 detail    each material's detail normal map (Poly Haven, CC0, local_ai_models/detail_maps; or made here:
              "noise" for skin and stone) laid on from the three axes in 3D (triplanar: the pattern does not follow
              the UV islands, which lie anywhere), --scale of the model's size per tile, its strength per material
  3 bake      the bumps added to the surface's own normals, written as the glTF normal map (tangent space)

    python scripts/generation/detail_normals.py model.glb --out detailed.glb \\
        --parts "skin=noise:0.35:0.02" "cloth=cotton_jersey:0.6:0.04" "metal=metal_plate_02:0.4:0.08"
    (word=map:strength:tile size, a share of the model's size)

Writes detailed.glb, detailed_parts.jpg (the views, each material in its colour) and detailed_normal.png.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch
import trimesh
from PIL import Image

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MAPS = os.path.join(PROJECT, "local_ai_models", "detail_maps")
MASK_PYTHON = os.path.join(PROJECT, "conda", "envs", "mask-ai", "bin", "python")
FIND_PART = os.path.join(PROJECT, "scripts", "inpainting", "find_part.py")
COLOURS = [(230, 60, 60), (60, 170, 230), (240, 200, 40), (90, 200, 90), (200, 90, 220), (240, 140, 40)]


def noise_map(size=1024, grain=6.0, seed=0):
    """A tileable normal map of fine random bumps (skin pores, stone grain): filtered noise as a height."""
    rng = np.random.default_rng(seed)
    f = np.fft.fftfreq(size)
    r = np.sqrt(f[:, None] ** 2 + f[None, :] ** 2)
    spectrum = np.fft.fft2(rng.standard_normal((size, size))) * np.exp(-(r * size / grain) ** 2)
    height = np.real(np.fft.ifft2(spectrum))
    height = (height - height.mean()) / (height.std() + 1e-9)
    dx = (np.roll(height, -1, 1) - np.roll(height, 1, 1)) * 0.5
    dy = (np.roll(height, -1, 0) - np.roll(height, 1, 0)) * 0.5
    n = np.stack([-dx, -dy, np.ones_like(height) * 2.0], -1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def load_map(name):
    """A detail normal map as unit vectors (x right, y up: OpenGL), size x size x 3."""
    if name == "noise":
        return noise_map()
    index = json.load(open(os.path.join(MAPS, "index.json")))
    for items in index.values():
        for it in items:
            if it["name"] == name:
                n = np.asarray(Image.open(os.path.join(MAPS, it["maps"]["nor_gl"])).convert("RGB"), np.float32)
                n = n / 127.5 - 1
                return n / np.linalg.norm(n, axis=-1, keepdims=True).clip(1e-6)
    raise SystemExit(f"no detail map called {name} (see {MAPS}/index.json, or 'noise')")


def sample(tex, u, v):
    """Bilinear, wrapping (the maps tile)."""
    h, w = tex.shape[:2]
    x, y = (u % 1.0) * w, (v % 1.0) * h
    x0, y0 = np.floor(x).astype(int) % w, np.floor(y).astype(int) % h
    x1, y1 = (x0 + 1) % w, (y0 + 1) % h
    fx, fy = (x - np.floor(x))[:, None], (y - np.floor(y))[:, None]
    return (tex[y0, x0] * (1 - fx) + tex[y0, x1] * fx) * (1 - fy) + (tex[y1, x0] * (1 - fx) + tex[y1, x1] * fx) * fy


def views_of(mesh, texture, folder, size=768):
    """The model from 8 sides (around, a little from above) and from above, orthographic: the pictures (for the
    part finder) and, per pixel, which texel it shows (row, column) or -1."""
    import nvdiffrast.torch as dr
    ctx = dr.RasterizeCudaContext()
    v = mesh.vertices - (mesh.vertices.max(0) + mesh.vertices.min(0)) / 2
    span = np.abs(v).max() * 1.05
    faces = torch.as_tensor(np.ascontiguousarray(mesh.faces), dtype=torch.int32, device="cuda")
    uv = mesh.visual.uv
    uvt = torch.as_tensor(np.stack([uv[:, 0], 1 - uv[:, 1]], 1), dtype=torch.float32, device="cuda")
    tex = torch.as_tensor(texture[None, ..., :3], dtype=torch.float32, device="cuda")
    th, tw = texture.shape[:2]
    views, lookups = [], []
    angles = [(a, 15) for a in range(0, 360, 45)] + [(0, 80)]
    os.makedirs(folder, exist_ok=True)
    for k, (az, el) in enumerate(angles):
        a, e = np.radians(az), np.radians(el)
        turn = np.array([[np.cos(a), 0, -np.sin(a)], [0, 1, 0], [np.sin(a), 0, np.cos(a)]])
        tilt = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
        p = v @ (tilt @ turn).T
        clip = np.stack([p[:, 0] / span, p[:, 1] / span, -p[:, 2] / span * 0.5, np.ones(len(p))], 1)
        rast, _ = dr.rasterize(ctx, torch.as_tensor(clip, dtype=torch.float32, device="cuda")[None], faces,
                               (size, size))
        tuv, _ = dr.interpolate(uvt, rast, faces)
        colour = dr.texture(tex, tuv, filter_mode="linear")[0].flip(0).cpu().numpy()
        hit = (rast[0, ..., 3] > 0).flip(0).cpu().numpy()
        tuv = tuv[0].flip(0).cpu().numpy()
        picture = np.where(hit[..., None], colour, 0.5)
        name = f"view_{k:02d}.png"
        Image.fromarray((np.clip(picture, 0, 1) * 255).astype(np.uint8)).save(os.path.join(folder, name))
        rows = np.where(hit, np.clip((tuv[..., 1] * th).astype(int), 0, th - 1), -1)
        cols = np.where(hit, np.clip((tuv[..., 0] * tw).astype(int), 0, tw - 1), -1)
        views.append(name)
        lookups.append((rows, cols))
    return views, lookups


def find(folder, views, word):
    json.dump({"word": word, "views": [{"picture": n, "clicks": []} for n in views]},
              open(os.path.join(folder, "views.json"), "w"))
    subprocess.run([MASK_PYTHON, FIND_PART, folder], check=True, capture_output=True)
    masks = dict(np.load(os.path.join(folder, "masks.npz")))
    return [masks.get(os.path.splitext(n)[0]) for n in views]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("model")
    p.add_argument("--out", required=True)
    p.add_argument("--parts", nargs="+", required=True, help="word=map:strength:tile")
    a = p.parse_args()
    import nvdiffrast.torch as dr

    scene = trimesh.load(a.model)
    mesh = list(scene.geometry.values())[0]
    material = mesh.visual.material
    texture = np.asarray(material.baseColorTexture.convert("RGB")).astype(np.float32) / 255
    th, tw = texture.shape[:2]
    work = os.path.splitext(os.path.abspath(a.out))[0] + "_work"
    parts = []
    for spec in a.parts:
        word, rest = spec.split("=", 1)
        name, strength, tile = (rest.split(":") + ["0.5", "0.04"])[:3]
        parts.append((word, name, float(strength), float(tile)))

    # 1 parts: votes per texel
    views, lookups = views_of(mesh, texture, work)
    votes = np.zeros((len(parts), th, tw), np.float32)
    seen = np.zeros((th, tw), np.float32)
    for rows, cols in lookups:
        ok = rows >= 0
        np.add.at(seen, (rows[ok], cols[ok]), 1)
    for i, (word, *_rest) in enumerate(parts):
        masks = find(work, views, word)
        found = 0
        for (rows, cols), m in zip(lookups, masks):
            if m is None:
                continue
            ok = (rows >= 0) & np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize(rows.shape[::-1])) > 0
            np.add.at(votes[i], (rows[ok], cols[ok]), 1)
            found += 1
        print(f"{word}: found on {found} of {len(views)} pictures", flush=True)
    share = votes / np.maximum(seen, 1)[None]
    label = np.where((seen > 0) & (share.max(0) >= 0.5), share.argmax(0), -1)
    # texels no picture saw (folds, the inside): the material of the nearest texel a picture saw
    from scipy.ndimage import distance_transform_edt
    if (seen > 0).any():
        _, (iy, ix) = distance_transform_edt(seen == 0, return_indices=True)
        label = np.where(seen > 0, label, label[iy, ix])

    # the views with the parts in colour (a check)
    sheet = []
    for name, (rows, cols) in zip(views, lookups):
        im = np.asarray(Image.open(os.path.join(work, name)).convert("RGB")).astype(np.float32)
        lab = np.where(rows >= 0, label[np.clip(rows, 0, th - 1), np.clip(cols, 0, tw - 1)], -1)
        for i in range(len(parts)):
            im[lab == i] = im[lab == i] * 0.45 + np.array(COLOURS[i % len(COLOURS)]) * 0.55
        sheet.append(Image.fromarray(im.astype(np.uint8)).resize((384, 384)))
    out = Image.new("RGB", (384 * len(sheet), 384))
    for k, im in enumerate(sheet):
        out.paste(im, (384 * k, 0))
    out.save(os.path.splitext(a.out)[0] + "_parts.jpg", quality=85)
    print("parts (share of the texture):", {w: f"{(label == i).mean():.1%}" for i, (w, *_r) in enumerate(parts)},
          flush=True)

    # 2-3 detail: per texel its place, normal and tangent frame (rasterised in UV space)
    ctx = dr.RasterizeCudaContext()
    uv = mesh.visual.uv
    clip = np.stack([uv[:, 0] * 2 - 1, uv[:, 1] * 2 - 1, np.zeros(len(uv)), np.ones(len(uv))], 1)
    faces_t = torch.as_tensor(np.ascontiguousarray(mesh.faces), dtype=torch.int32, device="cuda")
    rast, _ = dr.rasterize(ctx, torch.as_tensor(clip, dtype=torch.float32, device="cuda")[None], faces_t, (th, tw))
    place, _ = dr.interpolate(torch.as_tensor(mesh.vertices, dtype=torch.float32, device="cuda"), rast, faces_t)
    normal, _ = dr.interpolate(torch.as_tensor(mesh.vertex_normals, dtype=torch.float32, device="cuda"), rast,
                               faces_t)
    rast = rast[0].flip(0).cpu().numpy()
    covered = rast[..., 3] > 0
    face = rast[..., 3][covered].astype(np.int64) - 1
    P = place[0].flip(0).cpu().numpy()[covered]
    N = normal[0].flip(0).cpu().numpy()[covered]
    N /= np.linalg.norm(N, axis=1, keepdims=True).clip(1e-9)
    # each face's tangent and bitangent from its UVs (the directions u and v grow along)
    tri, tuv = mesh.vertices[mesh.faces], uv[mesh.faces]
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    d1, d2 = tuv[:, 1] - tuv[:, 0], tuv[:, 2] - tuv[:, 0]
    r = 1.0 / np.where(np.abs(d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]) < 1e-12, 1e-12,
                       d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1])
    T_face = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) * r[:, None]
    B_face = (e2 * d1[:, 0:1] - e1 * d2[:, 0:1]) * r[:, None]
    T = T_face[face] - N * (T_face[face] * N).sum(1, keepdims=True)
    T /= np.linalg.norm(T, axis=1, keepdims=True).clip(1e-9)
    handed = np.sign((np.cross(N, T) * B_face[face]).sum(1, keepdims=True))
    handed[handed == 0] = 1
    B = np.cross(N, T) * handed

    size = float(np.ptp(mesh.vertices, axis=0).max())
    lab = label[covered]
    detailed = N.copy()
    axes = np.eye(3)
    for i, (word, name, strength, tile) in enumerate(parts):
        sel = lab == i
        if not sel.any():
            continue
        dmap = load_map(name)
        p, n = P[sel] / (tile * size), N[sel]
        w = np.abs(n) ** 4
        w /= w.sum(1, keepdims=True)
        bump = np.zeros_like(n)
        for ax, (ia, ib) in enumerate(((1, 2), (0, 2), (0, 1))):  # the planes seen along x, y, z
            d = sample(dmap, p[:, ia], p[:, ib])
            bump += w[:, ax:ax + 1] * (d[:, 0:1] * axes[ia] + d[:, 1:2] * axes[ib])
        bump -= (bump * n).sum(1, keepdims=True) * n
        nn = n + strength * bump
        detailed[sel] = nn / np.linalg.norm(nn, axis=1, keepdims=True)
        print(f"{word}: {name}, strength {strength}, tile {tile} of the model: {sel.mean():.1%} of the texels",
              flush=True)

    # tangent space, as the glTF normal map wants it
    ts = np.stack([(detailed * T).sum(1), (detailed * B).sum(1), (detailed * N).sum(1)], 1)
    normal_map = np.full((th, tw, 3), (0.5, 0.5, 1.0), np.float32)
    normal_map[covered] = ts * 0.5 + 0.5
    img = Image.fromarray((np.clip(normal_map, 0, 1) * 255).astype(np.uint8))
    img.save(os.path.splitext(a.out)[0] + "_normal.png")
    material.normalTexture = img
    scene.export(a.out)
    print("detailed:", a.out)


if __name__ == "__main__":
    main()
