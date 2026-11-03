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

With --sorted (node 01's sorted.json, scripts/pipeline/sort_pictures.py) the region uses ALL pictures that
show the part: the main view (the one the model's camera is fitted to) and every other view or close-up, each
cut to the AI's box of the part, grown by the same margin. Each one has its role from node 01: shape and
colours, shape only (grey sculpt) or colours only (another version).

Usage (TRELLIS.2 environment):
    python scripts/inpainting/refine_region.py outputs/ninja.glb pictures/ninja/front.png --part head
    python scripts/inpainting/refine_region.py model.glb picture.png --picture-box 1200 300 1900 1100 --start random
    python scripts/inpainting/refine_region.py base.glb --sorted 01_sort/sorted.json --part head --method repaint
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
    parser.add_argument("picture", nargs="?", help="the picture the model was made from, at full resolution "
                                                   "(with --sorted: node 01's main view)")
    parser.add_argument("--sorted", help="node 01's sorted.json: use all pictures of the part, each with its role")
    parser.add_argument("--only", nargs="+", help="with --sorted: use only these pictures (names from sorted.json) "
                                                  "besides the main view, e.g. one sharp front picture")
    parser.add_argument("--best-colour", type=int,
                        help="with --sorted: of the colour pictures, only this many, the ones showing the part the "
                             "largest (most pixels): one sharp picture gave a clean face, six mixed ones smeared it")
    parser.add_argument("--mode", choices=["stochastic", "multidiffusion"], default="stochastic",
                        help="how several pictures are combined (multi_image.py)")
    parser.add_argument("--part", default="head", help="what to refine, found by the vision AI (default: head)")
    parser.add_argument("--picture-box", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                        help="or the region as a box in the picture's pixels (no vision AI)")
    parser.add_argument("--margin", type=float, default=0.35,
                        help="the kept margin around the region, as a share of its size (default 0.35)")
    parser.add_argument("--margin-of-model", type=float,
                        help="or the margin as a share of the whole model's height (the same width for big and small "
                             "parts, so big parts still get zoomed), e.g. 0.1")
    parser.add_argument("--method", choices=["inversion", "repaint"], default="inversion")
    parser.add_argument("--start", choices=["inverted", "random"], default="inverted",
                        help="inversion: the region starts from the original's noise or from new noise")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--texture-guidance", type=float, help="how strictly the colours follow the pictures (default 1 = off)")
    parser.add_argument("--faces", type=int, default=1_000_000, help="faces of the new region (default 1,000,000)")
    parser.add_argument("--keep-shape", action="store_true",
                        help="only the colours are regenerated, the shape stays exactly as it is (e.g. clothes)")
    parser.add_argument("--grow", type=int, default=1, help="grow the regenerated region by voxels (default 1)")
    parser.add_argument("--out", help="result (default: <model>_refined_<part>.glb)")
    args = parser.parse_args()
    start = time.time()
    name = os.path.splitext(args.model)[0]
    out = args.out or f"{name}_refined_{args.part.replace(' ', '_')}_{args.method}_{args.start}.glb"
    work = os.path.splitext(out)[0] + "_work"
    os.makedirs(work, exist_ok=True)

    sorted_views = more = None
    if args.sorted:
        result = json.load(open(args.sorted))
        sorted_views = {v["name"]: v for v in result["views"]}
        lists = result["lists"]
        base = lists["base"]
        # the main view (its camera is fitted to the model): of the views the base model was made from, the one
        # closest to the side the part is seen from (node 01: the back and the butt from behind) that shows it
        angle = {"front": 0, "front-right": 45, "right": 90, "back-right": 135, "back": 180, "back-left": 225,
                 "left": 270, "front-left": 315}
        apart = lambda a, b: min(abs(a - b), 360 - abs(a - b)) if a is not None and b is not None else 360
        wanted = angle.get(lists[args.part].get("side", "front"), 0)
        candidates = [base["front"]] + [n for n in base["shape_only"] + base["both"] if n != base["front"]]
        candidates.sort(key=lambda n: apart(angle.get(sorted_views[n]["side"]), wanted))  # stable: front first on ties
        main = next((sorted_views[n] for n in candidates if args.part in sorted_views[n]["part_boxes"]),
                    sorted_views[base["front"]])
        args.picture = os.path.join(os.path.dirname(args.sorted), main["view"])
        part = lists[args.part]
        role_of = {n: r for r, key in (("both", "both"), ("colour", "colour_only"), ("shape", "shape_only")) for n in part[key]}
        main_role = role_of.get(main["name"], "shape" if main["colours"] != "colour" else "both")
        # only pictures from the main view's side (within 90 degrees): TRELLIS.2 has no camera, so a picture from
        # behind of a front part (e.g. the back for the chest) turns the region around
        def same_side(view):
            a, b = angle.get(view["side"]), angle.get(main["side"])
            return a is None or b is None or min(abs(a - b), 360 - abs(a - b)) <= 90
        more = [(sorted_views[n], role_of[n]) for n in role_of if n != main["name"] and (not args.only or n in args.only)]
        for view, _ in more:
            if not same_side(view):
                print(f"    left out: {view['name']} ({view['side']}, the main view is {main['side']})", flush=True)
        more = [(view, role) for view, role in more if same_side(view)]
        if args.best_colour is not None:
            def score(view):
                """Complete first (the part's box touches the picture's border on at most one side: lower legs r9
                took a close-up where the leg filled the whole picture, cut off, seen from below), then largest."""
                b = view["part_boxes"][args.part]
                w, h = Image.open(view["picture"]).size
                touches = (b[0] <= 0.01 * w) + (b[1] <= 0.01 * h) + (b[2] >= 0.99 * w) + (b[3] >= 0.99 * h)
                return (touches > 1, -(b[2] - b[0]) * (b[3] - b[1]))
            coloured = sorted([m for m in more if m[1] != "shape"], key=lambda m: score(m[0]))
            for view, _ in coloured[args.best_colour:]:
                print(f"    left out: {view['name']} (only the {args.best_colour} sharpest colour picture(s))", flush=True)
            more = [m for m in more if m[1] == "shape"] + coloured[:args.best_colour]
        print(f"1. The {args.part} in the main view {main['name']} ({main_role}) + {len(more)} more pictures", flush=True)
        if args.part not in main["part_boxes"]:
            raise SystemExit(f"node 01 found no {args.part} in the main view")
        b = main["part_boxes"][args.part]
        box = [b[0] - main["box"][0], b[1] - main["box"][1], b[2] - main["box"][0], b[3] - main["box"][1]]
    else:
        main_role = "both"
        print(f"1. Finding the {args.part} in the picture", flush=True)
        box = args.picture_box or find_in_picture(args.picture, args.part)

    print("2. The region in the model", flush=True)
    verts, inside, cam, (width, height), (u, v) = region_in_model(args.model, args.picture, box)
    if inside.sum() < 50:
        raise SystemExit(f"Only {inside.sum()} vertices of the model lie in that box: is it the right picture?")
    inner = (verts[inside].min(0), verts[inside].max(0))
    if args.margin_of_model:
        args.margin = round(args.margin_of_model * np.ptp(verts, 0).max() / (inner[1] - inner[0]).max(), 3)
        print(f"    margin {args.margin} of the part's size ({args.margin_of_model} of the model's)", flush=True)
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

    more_pictures = []
    for k, (view, role) in enumerate(more or []):
        # the AI's box of the part in that picture, grown like the 3D box (margin) plus the same air
        x0, y0, x1, y1 = view["part_boxes"][args.part]
        grow = args.margin + 0.05
        w, h = x1 - x0, y1 - y0
        picture = Image.open(view["picture"])
        cut = [int(max(0, x0 - grow * w)), int(max(0, y0 - grow * h)), int(min(picture.width, x1 + grow * w)),
               int(min(picture.height, y1 + grow * h))]
        path = os.path.join(work, f"zoom_picture_{k + 1}_{role}.png")
        picture.crop(cut).save(path)
        more_pictures.append((path, role))
        print(f"    + {view['name']} ({role}): cut {cut}", flush=True)

    print(f"4. Inpainting the region ({args.method}, start {args.start}, {args.steps} steps)", flush=True)
    pipeline = generate.load_pipeline()
    size = np.ptp(zoomed.vertices, 0).max()
    points = box_points(np.concatenate(inner), size)
    zoom_out = os.path.join(work, "zoom_out.glb")
    inpaint(pipeline, zoom_in, crop, points, zoom_out, grow=args.grow, seed=args.seed, steps=args.steps,
            method=args.method, start=args.start, more_pictures=more_pictures, main_role=main_role, mode=args.mode,
            texture_guidance=args.texture_guidance, faces=args.faces, keep_shape=args.keep_shape)

    print("5. Zoom out: the original outside the seam, the new region inside it", flush=True)
    new = load_model(zoom_out)
    new_faces = np.nonzero(in_box(new.triangles_center, seam))[0]
    kept_faces = np.nonzero(~in_box(centers, seam))[0]
    scene = trimesh.Scene()
    scene.add_geometry(mesh.submesh([kept_faces], append=True), geom_name="kept")
    scene.add_geometry(new.submesh([new_faces], append=True), geom_name=f"refined_{args.part}")
    scene.export(out)
    json.dump({"picture_box": [float(x) for x in box], "crop": crop_box, "inner": np.array(inner).tolist(),
               "margin": args.margin, "method": args.method, "pictures": [[crop, main_role]] + more_pictures, "mode": args.mode,
               "texture_guidance": args.texture_guidance, "faces": args.faces, "keep_shape": args.keep_shape, "start": args.start, "steps": args.steps,
               "faces_kept": len(kept_faces), "faces_new": len(new_faces)},
              open(os.path.join(work, "refine.json"), "w"), indent=1)
    print(f"    {len(kept_faces):,} faces kept + {len(new_faces):,} new faces", flush=True)
    print(f"Result: {out}", flush=True)
    print(f"Done in {time.time() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
