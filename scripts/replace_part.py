"""Replace one part of a model that is split into parts with a newly generated version of that part.

This is part-level inpainting: everything except the chosen part stays exactly as it is.
The model comes from transfer_parts.py (one mesh per part). The new part is generated alone from a
cut-out of the same picture (cut_regions.py / cut_parts.py, then generate.py), so it gets TRELLIS.2's
whole 3D grid, e.g. a head at 1536 instead of the few voxels it had in the full model.

Usage (TRELLIS.2 environment):
    source activate.sh
    python scripts/replace_part.py outputs/yeonhee_parts18.glb pictures/yeonhee/parts-front-clean.png \\
        --with outputs/head_1536_seed42.glb pictures/yeonhee/head.png

The region to replace is what the cut-out covers in the picture, front and back, in whichever parts
it lies. Writes <model>_<new part name>.glb (the rest unchanged, the new part under the cut-out's name).

Steps:
    1. find the camera of the whole model in the picture, ignoring the region that gets replaced
    2. place the new part roughly with its own camera in the cut-out (as insert_part.py does)
    3. fine-align it onto the old region with ICP (scale, rotation, shift), so it sits exactly there
    4. match its colours to the old region's (the part alone often comes out in other colours)
    5. drop the old region's faces from every part, add the new part; everything else is untouched
"""
import argparse
import os

import cv2
import numpy as np
import nvdiffrast.torch as dr
import torch
import trimesh
from PIL import Image

from insert_part import crop_camera_to_picture, locate, similarity
from project_texture import fit_camera, load_meshes, object_masks, to_camera, to_pixels
from transfer_parts import icp, similarity_fit


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
    parser = argparse.ArgumentParser(description="Replace one region of a part-split model with a new part")
    parser.add_argument("model", help="model split into parts (transfer_parts.py output), or any model")
    parser.add_argument("picture", help="the picture the model was generated from")
    parser.add_argument("--with", dest="new", nargs=2, required=True, metavar=("MODEL", "CUTOUT"),
                        help="the new part's model and the RGBA cut-out it was generated from")
    parser.add_argument("--overlap", type=float, default=0.12,
                        help="how far (share of the new part's height) the old model reaches up into the "
                             "new part's lower end, so no gap opens (default 0.12)")
    parser.add_argument("--keep-colors", action="store_true",
                        help="keep the new part's own colours (default: match them to the old region's)")
    args = parser.parse_args()
    ctx = dr.RasterizeCudaContext()

    scene, meshes = load_meshes(args.model)
    names = [scene.graph[node][1] for node in scene.graph.nodes_geometry]
    picture = Image.open(args.picture)
    width, height = picture.size
    cutout = Image.open(args.new[1]).convert("RGBA")
    x0, y0 = locate(cutout, picture)
    alpha = np.asarray(cutout)[..., 3].astype(np.float32) / 255
    region = np.zeros((height, width), np.uint8)
    region[y0:y0 + alpha.shape[0], x0:x0 + alpha.shape[1]] = alpha > 0.5
    grow = max(3, int(0.01 * max(width, height)))
    region = cv2.dilate(region, np.ones((grow, grow), np.uint8)) > 0

    print("1. Camera of the whole model (ignoring the region to replace)...")
    mask = object_masks([picture])[0]
    full_cam, _ = fit_camera(ctx, meshes, picture, mask, ignore=region)

    # The old region: every face of every part that the picture shows inside the cut-out, front and
    # back (a part's own boundaries rarely match the cut-out, e.g. a "head" part with only the hair).
    old_faces = []
    for name, m in zip(names, meshes):
        centers = m.verts[m.faces.long()].mean(1)
        u, v, _ = to_pixels(to_camera(centers, full_cam), full_cam, width, height)
        u = u.long().clamp(0, width - 1).cpu().numpy()
        v = v.long().clamp(0, height - 1).cpu().numpy()
        old_faces.append(np.nonzero(region[v, u])[0])
    total = sum(len(f) for f in old_faces)
    print(f"    the region covers {total:,} faces in {sum(len(f) > 0 for f in old_faces)} parts")
    old_points = np.concatenate([m.verts[m.faces.long()[f]].mean(1).cpu().numpy() for m, f in zip(meshes, old_faces) if len(f)])

    print("2. Rough placement of the new part from its own camera...")
    new = Part(args.new[0])
    crop_cam, _ = fit_camera(ctx, [new], cutout, alpha)
    part_cam = crop_camera_to_picture(crop_cam, cutout.size, (x0, y0), picture.size)
    placed = similarity(part_cam, full_cam, picture.size)(new.verts).cpu().numpy()
    placed = placed + (old_points.mean(0) - placed.mean(0))  # depth: centre on the old region

    print("3. Fine alignment onto the old region (ICP)...")
    new_mesh = trimesh.Trimesh(placed, new.mesh.geometry.faces, process=False)
    source, _ = trimesh.sample.sample_surface(new_mesh, 20000, seed=1)
    target = old_points[np.random.default_rng(0).choice(len(old_points), min(20000, len(old_points)), replace=False)]
    moved, error = icp(source, target)
    scale, rotation, shift = similarity_fit(source, moved)
    final = scale * new_mesh.vertices @ rotation.T + shift
    print(f"    remaining mean distance to the old region: {error / np.linalg.norm(np.ptp(target, 0)):.1%} of its size")

    piece = new.mesh.geometry.copy()
    piece.vertices = final
    # keep the old model below the new part's lower end (e.g. the neck under a new head), reaching a
    # bit into it (--overlap), so no gap opens where the new part ends
    # The lowest point is measured near the part's vertical middle line: for a head that is the end
    # of the neck, not the tips of hair hanging down at the sides.
    up = 1  # glTF: +Y is up
    middle = np.median(final[:, [0, 2]], axis=0)
    width = np.ptp(final[:, [0, 2]], axis=0).max()
    central = np.linalg.norm(final[:, [0, 2]] - middle, axis=1) < 0.2 * width
    lowest = final[central, up].min() + args.overlap * np.ptp(final[:, up])
    for k, (m, faces) in enumerate(zip(meshes, old_faces)):
        if len(faces):
            heights = m.verts[m.faces.long()[faces]].mean(1)[:, up].cpu().numpy()
            old_faces[k] = faces[heights >= lowest]
    total = sum(len(f) for f in old_faces)
    if not args.keep_colors:
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
