"""Node 08: put the refined parts into the base model.

Every refine node (refine_region.py) regenerated one part from the base model on its own, so they can be
re-run one by one. This node takes, for every part, its new faces inside its seam (half way through its
margin, as refine_region.py does it) and the base model's faces outside all seams. Where two seams overlap,
the part named first wins (e.g. the head before the chest).

Usage:
    python scripts/pipeline/combine_parts.py base.glb 04_refine_head/head.glb 04_refine_chest/chest.glb --out final.glb
"""
import _paths  # noqa: F401  (the scripts of the other folders)

import argparse
import json
import os

import numpy as np
import trimesh

from inpaint import load_model


def grow_box(low, high, share):
    pad = (high - low) * share
    return low - pad, high + pad


def inside(points, box):
    return np.all((points >= box[0]) & (points <= box[1]), axis=1)


def part_of(refined):
    """The new region of a refine_region.py result: (its name, its model, its seam box)."""
    work = os.path.splitext(refined)[0] + "_work"
    info = json.load(open(os.path.join(work, "refine.json")))
    low, high = (np.array(c) for c in info["inner"])
    name = os.path.basename(os.path.splitext(refined)[0])
    return name, load_model(os.path.join(work, "zoom_out.glb")), grow_box(low, high, info["margin"] / 2)


def main():
    parser = argparse.ArgumentParser(description="Node 08: the refined parts put into the base model")
    parser.add_argument("base", help="the base model (.glb with one mesh)")
    parser.add_argument("parts", nargs="*", help="refine_region.py results, the most important first")
    parser.add_argument("--out", required=True)
    parser.add_argument("--single", action="store_true",
                        help="one mesh with one texture (packed), so the next round can inpaint the result again")
    args = parser.parse_args()
    base = load_model(args.base)
    centers = base.triangles_center
    taken = np.zeros(len(base.faces), bool)
    scene, report, seams = trimesh.Scene(), {}, []
    for refined in args.parts:
        name, mesh, seam = part_of(refined)
        mine = inside(mesh.triangles_center, seam)
        for other in seams:  # a part named earlier keeps its region
            mine &= ~inside(mesh.triangles_center, other)
        seams.append(seam)
        taken |= inside(centers, seam)
        faces = np.nonzero(mine)[0]
        scene.add_geometry(mesh.submesh([faces], append=True), geom_name=name)
        report[name] = int(len(faces))
        print(f"  {name}: {len(faces):,} new faces", flush=True)
    kept = np.nonzero(~taken)[0]
    scene.add_geometry(base.submesh([kept], append=True), geom_name="base")
    report["base"] = int(len(kept))
    if args.single:  # all meshes as one, their textures packed into one atlas (at most 8192 px)
        from PIL import Image
        one = trimesh.util.concatenate(list(scene.geometry.values()))
        texture = one.visual.material.baseColorTexture
        if texture is not None and max(texture.size) > 8192:
            scale = 8192 / max(texture.size)
            one.visual.material.baseColorTexture = texture.resize(
                (round(texture.width * scale), round(texture.height * scale)), Image.LANCZOS)
        scene = trimesh.Scene(one)
    scene.export(args.out)
    json.dump({"base": args.base, "parts": args.parts, "faces": report},
              open(os.path.splitext(args.out)[0] + ".json", "w"), indent=1)
    print(f"  base: {len(kept):,} faces kept", flush=True)
    print(f"Result: {args.out}", flush=True)


if __name__ == "__main__":
    main()
