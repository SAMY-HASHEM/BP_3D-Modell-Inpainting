"""Split a detailed TRELLIS.2 model into the parts that OmniPart found for the same picture.

OmniPart knows which part is where, but its meshes come from a 256^3 grid. TRELLIS.2 gives far more
detail (up to 1536^3) but one single mesh. This script aligns both models and gives every face of
the TRELLIS.2 model the part of the nearest OmniPart surface, so the result has TRELLIS.2's detail
and OmniPart's parts.

Usage (in the TRELLIS.2 environment):
    source activate.sh
    python scripts/transfer_parts.py outputs/front_1536_seed42.glb outputs/omnipart/front_rgba/mesh_segment.glb

Writes <TRELLIS.2 model>_parts.glb (one mesh per part, named part-1, part-2, ...) and
<TRELLIS.2 model>_parts_check.png (the TRELLIS.2 model coloured by part, from 4 sides).

Steps:
    1. load both; OmniPart's tiny leftover mesh (voxels outside all part boxes) is ignored
    2. rough alignment: same centre and height; front and back are both tried
    3. fine alignment with ICP: rotate, scale and move until the surfaces lie on each other
    4. every TRELLIS.2 face gets the part of its nearest OmniPart point, then neighbours vote
       a few times to smooth ragged borders between parts
    5. split the TRELLIS.2 mesh by part (UVs and texture are kept)
"""
import argparse
import math
import os

import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import cKDTree

MIN_PART_FACES = 500  # smaller OmniPart meshes are leftovers, not parts
ICP_POINTS = 30000
ICP_ITERATIONS = 40
SMOOTHING_ROUNDS = 3


def load_single(path):
    scene = trimesh.load(path)
    meshes = [trimesh.Trimesh(trimesh.transform_points(scene.geometry[name].vertices, transform),
                              scene.geometry[name].faces, process=False)
              for node in scene.graph.nodes_geometry for transform, name in [scene.graph[node]]]
    return scene, meshes


def similarity_fit(source, target):
    """Scale, rotation and shift that move points `source` onto `target` (Umeyama)."""
    mu_s, mu_t = source.mean(0), target.mean(0)
    s, t = source - mu_s, target - mu_t
    u, d, vt = np.linalg.svd(t.T @ s / len(s))
    flip = np.eye(3)
    flip[2, 2] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ flip @ vt
    scale = (d * np.diag(flip)).sum() / (s ** 2).sum(1).mean()
    return scale, rotation, mu_t - scale * rotation @ mu_s


def icp(source, target):
    """Align point set `source` to `target`; returns the moved points and the mean remaining distance."""
    tree = cKDTree(target)
    moved = source.copy()
    for _ in range(ICP_ITERATIONS):
        distance, index = tree.query(moved)
        keep = distance < np.percentile(distance, 90)  # ignore the worst matches (parts that differ)
        scale, rotation, shift = similarity_fit(moved[keep], target[index[keep]])
        moved = scale * moved @ rotation.T + shift
    return moved, tree.query(moved)[0].mean()


def rough_align(points, reference, yaw):
    """Same centre and height as `reference`, turned by `yaw` around the up axis (glTF +Y)."""
    c, r = (points.min(0) + points.max(0)) / 2, (reference.min(0) + reference.max(0)) / 2
    scale = np.ptp(reference[:, 1]) / np.ptp(points[:, 1])
    cos, sin = math.cos(yaw), math.sin(yaw)
    turn = np.array([[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]])
    return (points - c) @ turn.T * scale + r


def main():
    parser = argparse.ArgumentParser(description="Split a TRELLIS.2 model into OmniPart's parts")
    parser.add_argument("model", help="detailed model from generate.py (one mesh)")
    parser.add_argument("parts", help="OmniPart's mesh_segment.glb for the same picture (one mesh per part)")
    parser.add_argument("--out", help="output name without .glb (default: <model>_parts)")
    args = parser.parse_args()

    scene, detailed = load_single(args.model)
    assert len(detailed) == 1, "expected one mesh in the TRELLIS.2 model"
    detailed = detailed[0]
    geometry = next(iter(scene.geometry.values()))
    _, parts = load_single(args.parts)
    parts = [p for p in parts if len(p.faces) >= MIN_PART_FACES]
    print(f"TRELLIS.2 model: {len(detailed.faces):,} faces; OmniPart: {len(parts)} parts")

    # points on the OmniPart surface, each with its part number
    rng = np.random.default_rng(0)
    part_points, part_labels = [], []
    for k, p in enumerate(parts):
        points, _ = trimesh.sample.sample_surface(p, max(2000, ICP_POINTS * len(p.faces) // sum(len(q.faces) for q in parts)), seed=k)
        part_points.append(points)
        part_labels.append(np.full(len(points), k))
    part_points, part_labels = np.concatenate(part_points), np.concatenate(part_labels)
    detail_points, _ = trimesh.sample.sample_surface(detailed, ICP_POINTS, seed=1)

    print("Aligning the OmniPart parts to the TRELLIS.2 model...")
    best = None
    for yaw in (0, math.pi):
        start = rough_align(part_points, detail_points, yaw)
        moved, error = icp(start, detail_points)
        print(f"    starting turned by {math.degrees(yaw):3.0f}°: mean distance {error:.4f}")
        if best is None or error < best[1]:
            best = (yaw, error)
    yaw, error = best
    size = np.linalg.norm(np.ptp(detail_points, 0))
    print(f"    remaining mean distance: {error / size:.2%} of the model size")

    # the transform found on the samples, applied to all OmniPart points again for the labels
    start = rough_align(part_points, detail_points, yaw)
    moved, _ = icp(start, detail_points)

    print("Giving every face of the TRELLIS.2 model its part...")
    centers = detailed.triangles_center
    _, nearest = cKDTree(moved).query(centers)
    labels = part_labels[nearest]
    adjacency = detailed.face_adjacency
    for _ in range(SMOOTHING_ROUNDS):
        votes = np.zeros((len(labels), len(parts)), np.int32)
        for a, b in ((adjacency[:, 0], adjacency[:, 1]), (adjacency[:, 1], adjacency[:, 0])):
            np.add.at(votes, (a, labels[b]), 1)
        votes[np.arange(len(labels)), labels] += 1  # a face's own label counts too
        labels = votes.argmax(1)

    out = trimesh.Scene()
    for k in range(len(parts)):
        keep = labels == k
        if not keep.any():
            continue
        piece = geometry.copy()
        piece.update_faces(keep)
        piece.remove_unreferenced_vertices()
        out.add_geometry(piece, geom_name=f"part-{k + 1}")
        print(f"    part-{k + 1}: {keep.sum():,} faces")
    base = args.out or os.path.splitext(args.model)[0] + "_parts"
    out.export(base + ".glb")

    # check picture: the detailed model coloured by part, from 4 sides (simple orthographic splat)
    palette = np.array([[230, 90, 90], [90, 160, 230], [90, 220, 140], [220, 140, 230], [240, 200, 80],
                        [120, 220, 230], [200, 180, 120], [250, 150, 90], [150, 120, 230], [160, 230, 90]])
    tiles = []
    v = detailed.vertices - detailed.vertices.mean(0)
    scale = 480 / np.ptp(v, 0).max()
    for yaw in (0, 90, 180, 270):
        c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        p = centers - detailed.vertices.mean(0)
        x, y, z = p[:, 0] * c + p[:, 2] * s, p[:, 1], -p[:, 0] * s + p[:, 2] * c
        img = np.full((512, 512, 3), 60, np.uint8)
        depth = np.full((512, 512), -np.inf)
        u = (256 + x * scale).astype(int).clip(0, 511)
        w = (256 - y * scale).astype(int).clip(0, 511)
        order = np.argsort(z)  # far to near, so nearer faces are written last and stay visible
        img[w[order], u[order]] = palette[labels[order] % len(palette)]
        tiles.append(img)
    Image.fromarray(np.concatenate(tiles, 1)).save(base + "_check.png")
    print(f"Done: {base}.glb  (check: {base}_check.png)")


if __name__ == "__main__":
    main()
