"""Remove pieces that float away from a model, e.g. a ghost copy of hair from disagreeing views.

The biggest connected part is the model itself. Every other part is kept if it touches it or lies
close to it (accessories, a held object, hair strands), and removed if it floats further away.

Usage (TRELLIS.2 environment):
    source activate.sh
    python scripts/models/remove_floaters.py outputs/model.glb                 # writes outputs/model_clean.glb
    python scripts/models/remove_floaters.py outputs/model.glb --gap 0.03      # keep parts up to 3 % away
"""
import argparse
import os

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def remove_floaters(mesh, gap=0.02):
    """Removes parts further than `gap` (share of the model size) from the main part, in place.
    Returns (parts removed, faces removed)."""
    # parts are found on positions: the texture seams cut the mesh into pieces that belong together
    _, spot = np.unique(np.round(mesh.vertices, 6), axis=0, return_inverse=True)
    spot = spot.ravel()
    joined = trimesh.Trimesh(np.zeros((spot.max() + 1, 3)), spot[mesh.faces], process=False)
    labels = trimesh.graph.connected_component_labels(joined.face_adjacency, node_count=len(mesh.faces))
    sizes = np.bincount(labels)
    main = int(np.argmax(sizes))
    centers = mesh.triangles_center
    tree = cKDTree(centers[labels == main])
    limit = gap * np.linalg.norm(np.ptp(mesh.vertices, 0))
    keep_part = np.zeros(len(sizes), bool)
    keep_part[main] = True
    for part in np.nonzero(sizes)[0]:
        if part != main:
            faces = np.nonzero(labels == part)[0]
            sample = faces[:: max(1, len(faces) // 200)]  # a few hundred faces are enough to measure
            keep_part[part] = tree.query(centers[sample])[0].min() <= limit
    keep = keep_part[labels]
    removed_parts = int((~keep_part[np.nonzero(sizes)[0]]).sum())
    removed_faces = int((~keep).sum())
    if removed_faces:
        mesh.update_faces(keep)
        mesh.remove_unreferenced_vertices()
    return removed_parts, removed_faces


def main():
    parser = argparse.ArgumentParser(description="Remove pieces floating away from a model")
    parser.add_argument("model", help=".glb with one mesh")
    parser.add_argument("--gap", type=float, default=0.02,
                        help="keep parts at most this share of the model size away from the main part (default 0.02)")
    parser.add_argument("--out", help="output file (default: <model>_clean.glb)")
    args = parser.parse_args()
    scene = trimesh.load(args.model)
    for mesh in scene.geometry.values():
        parts, faces = remove_floaters(mesh, args.gap)
        print(f"    removed {parts} floating parts ({faces:,} of {len(mesh.faces) + faces:,} faces)")
    out = args.out or os.path.splitext(args.model)[0] + "_clean.glb"
    scene.export(out, extension_webp=True)
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
