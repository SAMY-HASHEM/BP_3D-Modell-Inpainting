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


def pack(meshes):
    """All meshes as ONE mesh with one texture atlas, so the next round can inpaint it. Our own atlas: trimesh's
    turned a whole model white. The last mesh (what was kept of the model) gets the big tile, so it loses little
    per round: 12288 x 8192, kept part 8192 x 8192, the new blocks in a column of 4096 x 8192 beside it."""
    from PIL import Image
    width, height, big = 12288, 8192, 8192
    blocks = meshes[:-1]
    cols = 1 if len(blocks) <= 2 else 2 if len(blocks) <= 8 else 4
    rows = max(1, -(-len(blocks) // cols))
    tile = (4096 // cols, height // rows)
    places = [((big + (k % cols) * tile[0], (k // cols) * tile[1]), tile) for k in range(len(blocks))] + [((0, 0), (big, big))]
    colour = Image.new("RGB", (width, height), (128, 128, 128))
    surface = Image.new("RGB", (width, height), (0, 128, 0))  # glTF: green = roughness, blue = metallic
    vertices, faces, uvs, offset = [], [], [], 0
    for mesh, ((x, y), (w, h)) in zip(meshes, places):
        material = mesh.visual.material
        colour.paste(material.baseColorTexture.convert("RGB").resize((w, h), Image.LANCZOS), (x, y))
        if getattr(material, "metallicRoughnessTexture", None) is not None:
            surface.paste(material.metallicRoughnessTexture.convert("RGB").resize((w, h), Image.LANCZOS), (x, y))
        uv = np.mod(np.asarray(mesh.visual.uv, float), 1.0)
        uvs.append(np.stack([(x + uv[:, 0] * w) / width, 1 - (y + (1 - uv[:, 1]) * h) / height], 1))
        vertices.append(np.asarray(mesh.vertices))
        faces.append(np.asarray(mesh.faces) + offset)
        offset += len(mesh.vertices)
    material = trimesh.visual.material.PBRMaterial(baseColorTexture=colour, metallicRoughnessTexture=surface,
                                                   metallicFactor=1.0, roughnessFactor=1.0)
    return trimesh.Trimesh(np.concatenate(vertices), np.concatenate(faces), process=False,
                           visual=trimesh.visual.TextureVisuals(uv=np.concatenate(uvs), material=material))


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
    if args.single:
        scene = trimesh.Scene(pack(list(scene.geometry.values())))
    scene.export(args.out, extension_webp=True)
    json.dump({"base": args.base, "parts": args.parts, "faces": report},
              open(os.path.splitext(args.out)[0] + ".json", "w"), indent=1)
    print(f"  base: {len(kept):,} faces kept", flush=True)
    print(f"Result: {args.out}", flush=True)


if __name__ == "__main__":
    main()
