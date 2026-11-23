"""A closed outer skin around any mesh (for retopo.py): points spread densely over the surface mark the voxels they
lie in, the marks grown by --grow voxels, everything enclosed filled, the outside surface taken (marching cubes).
Needs no inside / outside from the mesh: TRELLIS.2's 1536 export is open patches facing any way, and Blender's voxel
remesh made lace of it (the knight made in one go, Nov 23). Thin open sheets become thin closed slabs.

    python scripts/generation/skin_surface.py in.ply out.ply --voxel 0.004 [--grow 1]

(--voxel in the mesh's own units.) Runs where scikit-image is (MV-Adapter's environment).
"""
import argparse

import numpy as np
import trimesh
from scipy import ndimage
from skimage import measure


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("mesh")
    p.add_argument("out")
    p.add_argument("--voxel", type=float, required=True)
    p.add_argument("--grow", type=int, default=1)
    a = p.parse_args()
    mesh = trimesh.load(a.mesh, force="mesh", process=False)
    area = mesh.area
    count = int(min(30_000_000, max(2_000_000, 8 * area / a.voxel ** 2)))  # several points per voxel face
    points, _ = trimesh.sample.sample_surface(mesh, count)
    lo = points.min(0) - (a.grow + 3) * a.voxel
    idx = np.floor((points - lo) / a.voxel).astype(np.int64)
    shape = idx.max(0) + a.grow + 4
    grid = np.zeros(shape, bool)
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    if a.grow:
        grid = ndimage.binary_dilation(grid, iterations=a.grow)
    grid = ndimage.binary_fill_holes(grid)
    verts, faces, _, _ = measure.marching_cubes(grid.astype(np.float32), 0.5)
    verts = verts * a.voxel + lo
    skin = trimesh.Trimesh(verts, faces, process=True)
    parts = skin.split(only_watertight=False)
    skin = max(parts, key=lambda m: len(m.faces)) if len(parts) > 1 else skin
    keep = [m for m in parts if len(m.faces) >= 0.01 * len(skin.faces)] if len(parts) > 1 else [skin]
    skin = trimesh.util.concatenate(keep)
    skin.export(a.out)
    print(f"skin: {len(skin.faces)} faces, {len(keep)} parts, grid {tuple(shape)}", flush=True)


if __name__ == "__main__":
    main()
