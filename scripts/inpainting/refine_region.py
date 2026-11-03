"""Regenerate one region of a model (e.g. the head) with more detail: zoom in, inpaint, zoom back out.

TRELLIS.2 sees every picture at 1024 x 1024 and builds every model into one cube of the same grid, so a small
region of a big object (a face on a full figure) gets few pixels and few voxels. Here the region gets both:

  1. find      the vision AI (Qwen3-VL, as in organise_pictures.py) puts a box around the part in the picture
               (or --picture-box gives it)
  2. 3D region the model's camera in the picture (fit_camera, as in project_texture.py) turns the box into the
               part of the model it covers, through the whole depth: the inner box. Around it a margin (e.g.
               neck and shoulders around a head): the outer box
  3. zoom      only the model inside the outer box goes to TRELLIS.2, so it fills the whole cube; the picture
               is cut to the same region at full resolution, so the part fills the whole 1024 px
  4. inpaint   inpaint.py with inversion (VoxHammer's method): the margin stays on the original's own path and
               its keys and values are injected, the inner box is regenerated from the sharp cut-out, so the new
               part grows into the margin without a seam
  5. zoom out  the original model outside the seam (half way through the margin) + the new part inside it,
               as two meshes in one .glb

Usage (TRELLIS.2 environment):
    python scripts/inpainting/refine_region.py outputs/ninja.glb pictures/ninja/front.png --part head
    python scripts/inpainting/refine_region.py model.glb picture.png --picture-box 1200 300 1900 1100 --start random
"""
import _paths  # noqa: F401  (the scripts of the other folders)

import argparse
import json
import os
import time

import numpy as np
import torch
import trimesh
from PIL import Image

import generate  # sets the environment variables before TRELLIS.2 loads
from inpaint import box_points, inpaint, load_model

FIND = """Put a box around the %s of the figure in this picture: the whole %s, nothing else.
Answer only with JSON, no other text: {"box": [x0, y0, x1, y1]}   // coordinates from 0 to 1000 of the picture"""


def find_in_picture(picture_path, part):
    """The vision AI's box around the part, in pixels of the picture (then the AI is unloaded again)."""
    import organise_pictures as ai
    model, processor = ai.load(ai.MODELS[0], cpu=False)
    image = Image.open(picture_path).convert("RGB")
    shown = ai.shrink(image, 1024)
    answer, raw = ai.ask(model, processor, [shown], FIND % (part, part), 200)
    box = ai.to_pixels(answer.get("box"), image.width, image.height, shown.size)
    del model
    torch.cuda.empty_cache()
    if box is None:
        raise SystemExit(f"The vision AI found no {part}: {raw[:200]}")
    print(f"    the vision AI's box around the {part}: {box}", flush=True)
    return box


def region_in_model(model_path, picture_path, box):
    """Model vertices whose spot in the picture lies inside the box (through the whole depth), with the camera
    that shows the model like the picture. Returns (vertices of the model, which are inside, camera, picture size)."""
    import nvdiffrast.torch as dr
    from project_texture import fit_camera, load_meshes, object_masks, to_camera, to_pixels
    _, meshes = load_meshes(model_path)
    picture = Image.open(picture_path)
    mask = object_masks([picture])[0]
    cam, overlap = fit_camera(dr.RasterizeCudaContext(), meshes, picture, mask)
    print(f"    camera found (outline overlap {float(overlap):.2f})", flush=True)
    verts = load_model(model_path).vertices
    u, v, _ = to_pixels(to_camera(torch.tensor(verts, dtype=torch.float32, device=meshes[0].verts.device), cam),
                        cam, *picture.size)
    u, v = u.cpu().numpy(), v.cpu().numpy()
    x0, y0, x1, y1 = box
    inside = (u >= x0) & (u <= x1) & (v >= y0) & (v <= y1)
    return verts, inside, cam, picture.size, (u, v)


def grow_box(low, high, share):
    pad = (high - low) * share
    return low - pad, high + pad


def main():
    parser = argparse.ArgumentParser(description="Regenerate one region of a model with more detail (zoom in, inpaint)")
    parser.add_argument("model", help="the model (.glb with one textured mesh, e.g. from generate.py)")
    parser.add_argument("picture", help="the picture the model was made from, at full resolution")
    parser.add_argument("--part", default="head", help="what to refine, found by the vision AI (default: head)")
    parser.add_argument("--picture-box", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                        help="or the region as a box in the picture's pixels (no vision AI)")
    parser.add_argument("--margin", type=float, default=0.35,
                        help="the kept margin around the region, as a share of its size (default 0.35)")
    parser.add_argument("--method", choices=["inversion", "repaint"], default="inversion")
    parser.add_argument("--start", choices=["inverted", "random"], default="inverted",
                        help="inversion: the region starts from the original's noise or from new noise")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grow", type=int, default=1, help="grow the regenerated region by voxels (default 1)")
    parser.add_argument("--out", help="result (default: <model>_refined_<part>.glb)")
    args = parser.parse_args()
    start = time.time()
    name = os.path.splitext(args.model)[0]
    out = args.out or f"{name}_refined_{args.part.replace(' ', '_')}_{args.method}_{args.start}.glb"
    work = os.path.splitext(out)[0] + "_work"
    os.makedirs(work, exist_ok=True)

    print(f"1. Finding the {args.part} in the picture", flush=True)
    box = args.picture_box or find_in_picture(args.picture, args.part)

    print("2. The region in the model", flush=True)
    verts, inside, cam, (width, height), (u, v) = region_in_model(args.model, args.picture, box)
    if inside.sum() < 50:
        raise SystemExit(f"Only {inside.sum()} vertices of the model lie in that box: is it the right picture?")
    inner = (verts[inside].min(0), verts[inside].max(0))
    outer = grow_box(*inner, args.margin)
    seam = grow_box(*inner, args.margin / 2)
    print(f"    {inside.sum():,} of {len(verts):,} vertices; inner box {np.round(inner, 3).tolist()}", flush=True)

    print("3. Zoom: the model and the picture cut to the region with its margin", flush=True)
    mesh = load_model(args.model)
    centers = mesh.triangles_center
    in_box = lambda points, b: np.all((points >= b[0]) & (points <= b[1]), axis=1)
    near = np.nonzero(in_box(centers, outer))[0]
    zoomed = mesh.submesh([near], append=True)
    zoom_in = os.path.join(work, "zoom_in.glb")
    zoomed.export(zoom_in)
    # the picture region of everything in the outer box, with a little air around it
    shown = in_box(verts, outer)
    px0, px1 = u[shown].min(), u[shown].max()
    py0, py1 = v[shown].min(), v[shown].max()
    air = 0.05 * max(px1 - px0, py1 - py0)
    crop_box = [int(max(0, px0 - air)), int(max(0, py0 - air)), int(min(width, px1 + air)), int(min(height, py1 + air))]
    crop = os.path.join(work, "zoom_picture.png")
    Image.open(args.picture).crop(crop_box).save(crop)
    print(f"    {len(near):,} faces (of {len(mesh.faces):,}); picture cut {crop_box} "
          f"({crop_box[2] - crop_box[0]} x {crop_box[3] - crop_box[1]} px)", flush=True)

    print(f"4. Inpainting the region ({args.method}, start {args.start}, {args.steps} steps)", flush=True)
    pipeline = generate.load_pipeline()
    size = np.ptp(zoomed.vertices, 0).max()
    points = box_points(np.concatenate(inner), size)
    zoom_out = os.path.join(work, "zoom_out.glb")
    inpaint(pipeline, zoom_in, crop, points, zoom_out, grow=args.grow, seed=args.seed, steps=args.steps,
            method=args.method, start=args.start)

    print("5. Zoom out: the original outside the seam, the new region inside it", flush=True)
    new = load_model(zoom_out)
    new_faces = np.nonzero(in_box(new.triangles_center, seam))[0]
    kept_faces = np.nonzero(~in_box(centers, seam))[0]
    scene = trimesh.Scene()
    scene.add_geometry(mesh.submesh([kept_faces], append=True), geom_name="kept")
    scene.add_geometry(new.submesh([new_faces], append=True), geom_name=f"refined_{args.part}")
    scene.export(out)
    json.dump({"picture_box": [float(x) for x in box], "crop": crop_box, "inner": np.array(inner).tolist(),
               "margin": args.margin, "method": args.method, "start": args.start, "steps": args.steps,
               "faces_kept": len(kept_faces), "faces_new": len(new_faces)},
              open(os.path.join(work, "refine.json"), "w"), indent=1)
    print(f"    {len(kept_faces):,} faces kept + {len(new_faces):,} new faces", flush=True)
    print(f"Result: {out}", flush=True)
    print(f"Done in {time.time() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
