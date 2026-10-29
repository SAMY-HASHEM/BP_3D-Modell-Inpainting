"""Replace one part of a model that is split into parts with a newly generated version of that part.

This is part-level inpainting: everything except the chosen part stays exactly as it is.
The model comes from transfer_parts.py (one mesh per part). The new part is generated alone from a
cut-out of the same picture (cut_regions.py / cut_parts.py, then generate.py), so it gets TRELLIS.2's
whole 3D grid, e.g. a head at 1536 instead of the few voxels it had in the full model.

Usage (TRELLIS.2 environment):
    source activate.sh
    python scripts/regenerate/replace_part.py outputs/yeonhee_parts18.glb pictures/yeonhee/parts-front-clean.png \\
        --with outputs/head_1536_seed42.glb pictures/yeonhee/head.png

The region is what the cut-out covers in the picture (front and back, in whichever parts it lies); of
it, only old surfaces close to the fitted new part are removed. If the new part was cut from another
picture (e.g. a closer shot), --region gives a cut-out from the model's own picture for the region.
Writes <model>_<new part name>.glb (the rest unchanged, the new part under the cut-out's name).

Steps:
    1. find the camera of the whole model in the picture, ignoring the region that gets replaced
    2. place the new part roughly with its own camera in the cut-out (as insert_part.py does)
    3. fine-align it onto the old region with ICP (scale, rotation, shift), so it sits exactly there
    4. remove the old surfaces close to the new part (its old version); farther ones stay (neck, throne)
    5. match its colours to the removed old surfaces' colours, add the new part; the rest is untouched
"""
import _paths  # noqa: F401  (the scripts of the other folders)
import argparse
import math
import os

import cv2
import numpy as np
import nvdiffrast.torch as dr
import torch
import trimesh
from PIL import Image

from insert_part import crop_camera_to_picture, locate, similarity
from project_texture import fit_camera, load_meshes, object_masks, to_camera, to_pixels
from scipy.spatial import cKDTree

from transfer_parts import icp, rough_align, similarity_fit


class Part:
    """A single-mesh model (e.g. from generate.py) with the fields fit_camera needs."""

    def __init__(self, path):
        scene, meshes = load_meshes(path)
        assert len(meshes) == 1, f"{path}: expected one mesh"
        self.scene, self.mesh = scene, meshes[0]
        self.verts, self.faces, self.uvs_tex, self.texture = (self.mesh.verts, self.mesh.faces,
                                                              self.mesh.uvs_tex, self.mesh.texture)


def face_colors(geometry, faces=None, limit=200000):
    """Texture colour at the centre of (some of) a mesh's faces, as RGB 0..1."""
    faces = np.arange(len(geometry.faces)) if faces is None else faces
    if len(faces) > limit:
        faces = np.random.default_rng(0).choice(faces, limit, replace=False)
    texture = np.asarray(geometry.visual.material.baseColorTexture.convert("RGB"), np.float32) / 255
    uv = geometry.visual.uv[geometry.faces[faces]].mean(1)
    h, w = texture.shape[:2]
    x = np.clip((uv[:, 0] * w).astype(int), 0, w - 1)
    y = np.clip(((1 - uv[:, 1]) * h).astype(int), 0, h - 1)
    return texture[y, x]


def match_colors(geometry, reference):
    """Shift the texture's colours towards `reference` (N x 3, RGB 0..1): same mean and (within limits)
    the same spread per channel, in Lab colour space, so hues don't drift (RGB stretching gave blue streaks)."""
    to_lab = lambda rgb: cv2.cvtColor(rgb.reshape(-1, 1, 3).astype(np.float32), cv2.COLOR_RGB2LAB).reshape(-1, 3)
    own, ref = to_lab(face_colors(geometry)), to_lab(reference)
    ratio = np.clip(ref.std(0) / (own.std(0) + 1e-6), 0.7, 1.4)
    texture = np.asarray(geometry.visual.material.baseColorTexture.convert("RGB"), np.float32) / 255
    lab = cv2.cvtColor(texture, cv2.COLOR_RGB2LAB)
    lab = (lab - own.mean(0)) * ratio + ref.mean(0)
    rgb = cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2RGB)
    geometry.visual.material.baseColorTexture = Image.fromarray((rgb.clip(0, 1) * 255).astype(np.uint8))


def main():
    parser = argparse.ArgumentParser(description="Replace one region of a model with a newly generated part")
    parser.add_argument("model", help="model to change (one mesh, or split into parts by transfer_parts.py)")
    parser.add_argument("picture", help="the picture the model was generated from")
    parser.add_argument("--with", dest="new", nargs=2, required=True, metavar=("MODEL", "CUTOUT"),
                        help="the new part's model and the RGBA cut-out it was generated from")
    parser.add_argument("--region", help="RGBA cut-out from the model's picture marking the region to replace; "
                                         "needed when the new part was cut from another picture")
    parser.add_argument("--near", type=float, default=0.05,
                        help="old surfaces closer than this share of the new part's size to it are removed "
                             "(default 0.05); farther ones stay, e.g. the neck below a new head")
    parser.add_argument("--keep-colors", action="store_true",
                        help="keep the new part's own colours (default: match them to the old region's)")
    args = parser.parse_args()
    ctx = dr.RasterizeCudaContext()

    scene, meshes = load_meshes(args.model)
    names = [scene.graph[node][1] for node in scene.graph.nodes_geometry]
    picture = Image.open(args.picture)
    width, height = picture.size
    region_cutout = Image.open(args.region or args.new[1]).convert("RGBA")
    x0, y0 = locate(region_cutout, picture)
    region_alpha = np.asarray(region_cutout)[..., 3].astype(np.float32) / 255
    region = np.zeros((height, width), np.uint8)
    region[y0:y0 + region_alpha.shape[0], x0:x0 + region_alpha.shape[1]] = region_alpha > 0.5
    grow = max(3, int(0.01 * max(width, height)))
    region = cv2.dilate(region, np.ones((grow, grow), np.uint8)) > 0

    print("1. Camera of the whole model (ignoring the region to replace)...")
    mask = object_masks([picture])[0]
    full_cam, _ = fit_camera(ctx, meshes, picture, mask, ignore=region)

    # The old region: every face of every part that the picture shows inside the region, front and back
    in_region = []
    for m in meshes:
        centers = m.verts[m.faces.long()].mean(1)
        u, v, _ = to_pixels(to_camera(centers, full_cam), full_cam, width, height)
        u = u.long().clamp(0, width - 1).cpu().numpy()
        v = v.long().clamp(0, height - 1).cpu().numpy()
        in_region.append(np.nonzero(region[v, u])[0])
    centers = [m.verts[m.faces.long()].mean(1).cpu().numpy() for m in meshes]
    old_points = np.concatenate([c[f] for c, f in zip(centers, in_region) if len(f)])
    print(f"    the region covers {len(old_points):,} faces")

    new = Part(args.new[0])
    target = old_points[np.random.default_rng(0).choice(len(old_points), min(20000, len(old_points)), replace=False)]
    if args.region is None:
        print("2. Rough placement of the new part from its own camera...")
        cutout = Image.open(args.new[1]).convert("RGBA")
        alpha = np.asarray(cutout)[..., 3].astype(np.float32) / 255
        crop_cam, _ = fit_camera(ctx, [new], cutout, alpha)
        part_cam = crop_camera_to_picture(crop_cam, cutout.size, (x0, y0), picture.size)
        placed = similarity(part_cam, full_cam, picture.size)(new.verts).cpu().numpy()
        starts = [placed + (old_points.mean(0) - placed.mean(0))]  # depth: centre on the old region
    else:
        # cut from another picture: no shared camera, so start from the region's centre and height,
        # turned to each of 4 directions, and let ICP pick the best fit
        print("2. Rough placement of the new part by size, from 4 directions...")
        verts = new.verts.cpu().numpy()
        starts = [rough_align(verts, old_points, math.radians(yaw)) for yaw in (0, 90, 180, 270)]

    print("3. Fine alignment onto the old region (ICP)...")
    best = None
    for start in starts:
        source, _ = trimesh.sample.sample_surface(trimesh.Trimesh(start, new.mesh.geometry.faces, process=False), 20000, seed=1)
        moved, error = icp(source, target)
        if best is None or error < best[0]:
            scale, rotation, shift = similarity_fit(source, moved)
            best = (error, scale * start @ rotation.T + shift)
    error, final = best
    print(f"    remaining mean distance to the old region: {error / np.linalg.norm(np.ptp(target, 0)):.1%} of its size")

    piece = new.mesh.geometry.copy()
    piece.vertices = final
    # Remove only old surfaces lying close to the new part: the old version of the part itself. Things
    # farther away stay, like the neck below a new head or a throne behind it.
    surface, _ = trimesh.sample.sample_surface(trimesh.Trimesh(final, piece.faces, process=False), 300000, seed=2)
    tree = cKDTree(surface)
    limit = args.near * np.linalg.norm(np.ptp(final, 0))
    old_faces = []
    for c, f in zip(centers, in_region):
        old_faces.append(f[tree.query(c[f])[0] < limit] if len(f) else f)
    total = sum(len(f) for f in old_faces)
    if not args.keep_colors and total:
        reference = np.concatenate([face_colors(scene.geometry[n], f) for n, f in zip(names, old_faces) if len(f)])
        match_colors(piece, reference)

    out = trimesh.Scene()
    for name, faces in zip(names, old_faces):
        part = scene.geometry[name].copy()
        if len(faces):
            keep = np.ones(len(part.faces), bool)
            keep[faces] = False
            if not keep.any():
                continue
            part.update_faces(keep)
            part.remove_unreferenced_vertices()
        out.add_geometry(part, geom_name=name)
    new_name = os.path.splitext(os.path.basename(args.new[1]))[0]
    out.add_geometry(piece, geom_name=new_name)
    base = os.path.splitext(args.model)[0] + f"_{new_name}"
    out.export(base + ".glb")
    print(f"Done: {base}.glb  ({total:,} old faces -> {new_name}, {len(piece.faces):,} faces)")


if __name__ == "__main__":
    main()
