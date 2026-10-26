"""Turn every face of a model so that it points outward, as Blender expects.

The 1536 export (and inpaint.py's) keeps TRELLIS.2's own mesh instead of remeshing it. That mesh
is made of many small patches whose faces point in random directions, in or out. Our renders
don't care, but Blender lights inward faces from behind: patches with dark edges all over the
model, e.g. across a face.

How: look at the model from 64 directions around it. Every face that is seen gets a vote: +1 when
it points towards the viewer, -1 when it points away. The patches are consistent inside (their
faces agree with their neighbours), so a whole patch is flipped when its faces vote "away".

Usage (TRELLIS.2 environment):
    source activate.sh
    python scripts/fix_normals.py outputs/model.glb                  # writes outputs/model_fixed.glb
    python scripts/fix_normals.py outputs/model.glb --out fixed.glb
generate.py and inpaint.py do this by themselves when they save (since this script exists).
"""
import argparse
import os

import numpy as np
import nvdiffrast.torch as dr
import torch
import trimesh

VIEWS = 64
SIZE = 2048


def directions(n):
    """n directions spread evenly over a sphere (Fibonacci sphere)."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.cos(phi), np.sin(theta) * np.sin(phi)], 1)


def outward_votes(vertices, faces, views=VIEWS, size=SIZE):
    """Per face: how many pixels show it facing the viewer, minus how many show it facing away."""
    ctx = dr.RasterizeCudaContext()
    v = torch.tensor(vertices, dtype=torch.float32, device="cuda")
    f = torch.tensor(faces, dtype=torch.int32, device="cuda")
    center = (v.min(0)[0] + v.max(0)[0]) / 2
    radius = float(torch.linalg.norm(v.max(0)[0] - v.min(0)[0]) / 2) * 1.05
    tri = v[f.long()]
    normals = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=1)
    votes = torch.zeros(len(faces), device="cuda")
    for d in directions(views):
        d = torch.tensor(d, dtype=torch.float32, device="cuda")
        up = torch.tensor([0.0, 1.0, 0.0], device="cuda") if abs(d[1]) < 0.9 else torch.tensor([1.0, 0.0, 0.0], device="cuda")
        right = torch.nn.functional.normalize(torch.cross(up, d, dim=0), dim=0)
        up = torch.cross(d, right, dim=0)
        p = v - center
        # orthographic view from direction d: nearer to the viewer = smaller depth
        clip = torch.stack([p @ right / radius, p @ up / radius, -(p @ d) / radius * 0.5, torch.ones(len(p), device="cuda")], 1)
        rast, _ = dr.rasterize(ctx, clip[None].contiguous(), f, (size, size))
        seen = rast[0, ..., 3].long().flatten() - 1
        seen = seen[seen >= 0]
        sign = torch.sign(normals[seen] @ d)
        votes.index_add_(0, seen, sign)
    return votes.cpu().numpy()


def fix_face_directions(mesh):
    """Flips the inward patches of a trimesh.Trimesh in place; returns the share of faces flipped."""
    trimesh.repair.fix_winding(mesh)  # every patch consistent inside
    labels = trimesh.graph.connected_component_labels(mesh.face_adjacency, node_count=len(mesh.faces))
    votes = outward_votes(mesh.vertices, mesh.faces)
    flip = np.bincount(labels, weights=votes)[labels] < 0
    faces = np.array(mesh.faces)
    faces[flip] = faces[flip][:, ::-1]
    mesh.faces = faces
    mesh.vertex_normals = smooth_normals(mesh)  # changing the faces dropped them; without, Blender shades flat
    torch.cuda.empty_cache()
    return flip.mean()


def smooth_normals(mesh):
    """Vertex normals averaged over all faces at the same position, also across texture seams (where
    the mesh is cut into separate vertices), so the seams don't show as lines in the shading."""
    _, spot = np.unique(np.round(mesh.vertices, 6), axis=0, return_inverse=True)
    spot = spot.ravel()
    tri = mesh.vertices[mesh.faces]
    face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])  # longer for bigger faces
    total = np.zeros((spot.max() + 1, 3))
    for corner in range(3):
        np.add.at(total, spot[mesh.faces[:, corner]], face_normals)
    normals = total[spot]
    return normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)


def main():
    parser = argparse.ArgumentParser(description="Turn all faces of a model outward (for Blender)")
    parser.add_argument("model", help=".glb with one mesh")
    parser.add_argument("--out", help="output file (default: <model>_fixed.glb)")
    args = parser.parse_args()
    scene = trimesh.load(args.model)
    for mesh in scene.geometry.values():
        share = fix_face_directions(mesh)
        print(f"    flipped {share:.0%} of {len(mesh.faces):,} faces")
    out = args.out or os.path.splitext(args.model)[0] + "_fixed.glb"
    scene.export(out, extension_webp=True)
    print(f"Done: {out}")


if __name__ == "__main__":
    main()
