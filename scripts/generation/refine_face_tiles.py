"""(An experiment, Nov 22, stopped after 2 of 4 tiles: each tile came out as a block with a face on its front;
the merge was not tried yet.) More detail in a Pixal3D face: the face cut into tiles (2 x 2, overlapping), each tile made alone with Pixal3D
(a quarter of the face fills the whole cube: ~2x the detail), and the face model's surface pulled onto the tiles'.

The head itself (F4 of compare_pixal3d.py) keeps its mesh, its texture and its outline; only its visible front moves,
each vertex along its camera ray (so it stays where the picture has it) to the depth the tiles have there, blended
to nothing towards each tile's edge (the tiles overlap, so their edges are inside a neighbour). Tile and head are tied
together like swap_head.py: each tile pixel is a pixel of the head's picture, so the two models give pairs of 3D
points, and scale + turn + shift (Umeyama, the worst 30 % left out) put the tile onto the head.

    python scripts/generation/refine_face_tiles.py head.glb --camera head.json --picture head_input_rgba.png \
        --source colour_front_rgba.png --tile 380 480 720 800 --tile 640 480 980 800 \
        --tile 380 720 720 1030 --tile 640 720 980 1030 --out outputs/x/tiles

Step 1 makes each tile's model (compare_pixal3d.py at --resolution, ~12 min each at 1536; one after another, a
made tile is kept); step 2 writes <out>/refined.glb and <out>/moves.jpg (how far the face moved, per pixel).
--source: the picture the head was made from (before Pixal3D's own cut-out); --tile: x0 y0 x1 y1 in it.
"""
import argparse
import json
import math
import os
import subprocess
import sys

import numpy as np
import trimesh
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paint_front as paint  # noqa: E402
from swap_head import positions, own_colours, similarity  # noqa: E402


def square_of(rgba):
    """Pixal3D's own cut-out (compare_pixal3d.cut_out): the corner and side of its square in this picture."""
    a = np.asarray(rgba)[..., 3]
    ys, xs = np.nonzero(a > 0.8 * 255)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    size = int(max(xs.max() - xs.min(), ys.max() - ys.min()) * 1.1)
    return np.array([cx - size // 2, cy - size // 2]), size


def make_tiles(args):
    source = Image.open(args.source).convert("RGBA")
    for k, (x0, y0, x1, y1) in enumerate(args.tile):
        name = f"tile{k + 1}"
        if os.path.exists(os.path.join(args.out, name + ".glb")):
            continue
        tile = source.crop((x0, y0, x1, y1))
        a = np.asarray(tile).copy()
        a[..., 3] = 255
        a[:2, :, 3] = a[-2:, :, 3] = a[:, :2, 3] = a[:, -2:, 3] = 0  # an outline: the whole tile is the object
        Image.fromarray(a).save(os.path.join(args.out, name + "_picture.png"))
        print(f"=== {name}: {x1 - x0} x {y1 - y0} px of the face", flush=True)
        subprocess.run([sys.executable, os.path.join(HERE, "compare_pixal3d.py"),
                        os.path.join(args.out, name + "_picture.png"), "--name", name, "--out", args.out,
                        "--resolution", args.resolution], check=True)


def merge(args):
    head = trimesh.load(args.model)
    node = head.graph.nodes_geometry[0]
    mesh = next(iter(head.geometry.values()))
    to_world = head.graph.get(node)[0]
    world = trimesh.transform_points(mesh.vertices, to_world)  # glTF axes
    cam = json.load(open(args.camera))["camera"]
    fov, distance = math.radians(cam["fov_degrees"]), cam["distance"]
    picture = np.asarray(Image.open(args.picture).convert("RGBA")).astype(np.float32) / 255
    res = picture.shape[0]
    flat = trimesh.Trimesh(world, mesh.faces, process=False)
    flat.visual = mesh.visual

    # source px -> head picture px (Pixal3D's square) -> head model px (the picture lined up on its colours)
    corner, size = square_of(Image.open(args.source).convert("RGBA"))
    own, own_mask = own_colours(flat, cam, res)
    whole, _ = paint.line_up_on(own, own_mask & (picture[..., 3] > 0.5), picture[..., :3])
    whole = np.eye(2, 3) if whole is None else whole
    to_model = np.linalg.inv(np.vstack([whole, [0, 0, 1]]))[:2]
    source_to_model = lambda p: ((p - corner) * (res / size)) @ to_model[:, :2].T + to_model[:, 2]

    head_pos, head_mask = positions(flat, cam, res)
    cube = world @ np.linalg.inv(paint.TO_GLTF).T
    uv, depth = paint.project(cube, paint.CAMERA_TURN, fov, distance)
    eye = np.linalg.inv(paint.CAMERA_TURN) @ np.array([0, -distance, 0])  # the camera, cube frame
    head_depth = np.full((res, res), np.inf)
    hy, hx = np.nonzero(head_mask)
    head_depth[hy, hx] = paint.project(head_pos[hy, hx] @ np.linalg.inv(paint.TO_GLTF).T, paint.CAMERA_TURN, fov,
                                       distance)[1]
    px = np.clip((uv * res).astype(int), 0, res - 1)
    visible = depth <= head_depth[px[:, 1], px[:, 0]] + 0.004

    move = np.zeros(len(world))
    weight_sum = np.zeros(len(world))
    for k, (x0, y0, x1, y1) in enumerate(args.tile):
        name = f"tile{k + 1}"
        tile = trimesh.load(os.path.join(args.out, name + ".glb"), force="mesh")
        tcam = json.load(open(os.path.join(args.out, name + ".json")))["camera"]
        tpic = Image.open(os.path.join(args.out, name + "_input_rgba.png"))
        tcorner, tsize = square_of(Image.open(os.path.join(args.out, name + "_picture.png")))
        rt = tpic.width
        tile_pos, tile_mask = positions(tile, tcam, rt)
        ys, xs = np.nonzero(tile_mask)
        p_src = (np.stack([xs, ys], 1) + 0.5) * (tsize / rt) + tcorner + np.array([x0, y0])
        inner = ((p_src[:, 0] > x0 + 0.12 * (x1 - x0)) & (p_src[:, 0] < x1 - 0.12 * (x1 - x0)) &
                 (p_src[:, 1] > y0 + 0.12 * (y1 - y0)) & (p_src[:, 1] < y1 - 0.12 * (y1 - y0)))
        ys, xs, p_src = ys[inner][::5], xs[inner][::5], p_src[inner][::5]
        q = np.round(source_to_model(p_src) - 0.5).astype(int)
        ok = (q.min(1) >= 0) & (q.max(1) < res)
        ok[ok] &= head_mask[q[ok, 1], q[ok, 0]]
        a, b = tile_pos[ys[ok], xs[ok]], head_pos[q[ok, 1], q[ok, 0]]
        use = np.ones(len(a), bool)
        for _ in range(3):
            s, R, t = similarity(a[use], b[use])
            err = np.linalg.norm(s * a @ R.T + t - b, axis=1)
            use = err <= np.quantile(err, 0.7)
        height = flat.bounds[1, 1] - flat.bounds[0, 1]
        print(f"{name}: {len(a)} pairs, scale {s:.3f}, error {np.median(err) / height * 100:.2f} % of the head",
              flush=True)
        tile.vertices = s * np.asarray(tile.vertices) @ R.T + t
        # the tile seen from the head's camera: its depth per pixel
        tpos, tmask = positions(tile, cam, res)
        tdepth = np.full((res, res), np.nan)
        ty, tx = np.nonzero(tmask)
        tdepth[ty, tx] = paint.project(tpos[ty, tx] @ np.linalg.inv(paint.TO_GLTF).T, paint.CAMERA_TURN, fov,
                                       distance)[1]
        # the tile's box in the head model's pixels, and a weight: 1 inside, to 0 at its edge
        corners = source_to_model(np.array([[x0, y0], [x1, y1]], float))
        (bx0, by0), (bx1, by1) = corners
        fx = np.clip(np.minimum(uv[:, 0] * res - bx0, bx1 - uv[:, 0] * res) / (0.22 * (bx1 - bx0)), 0, 1)
        fy = np.clip(np.minimum(uv[:, 1] * res - by0, by1 - uv[:, 1] * res) / (0.22 * (by1 - by0)), 0, 1)
        w = (fx * fy) ** 1.5 * visible
        target = tdepth[px[:, 1], px[:, 0]]
        diff = target - depth
        good = (w > 0) & np.isfinite(target) & (np.abs(diff) < args.max_move)
        move[good] += w[good] * diff[good]
        weight_sum[good] += w[good]
        print(f"  moves {good.sum()} vertices, median {np.median(np.abs(diff[good])) / height * 100:.2f} % "
              f"of the head", flush=True)

    moved = weight_sum > 0
    step = np.zeros(len(world))
    step[moved] = move[moved] / np.maximum(weight_sum[moved], 1) * np.minimum(weight_sum[moved], 1)
    ray = cube - eye
    new_cube = cube + ray / np.linalg.norm(ray, axis=1, keepdims=True) * (step / np.maximum(
        (ray / np.linalg.norm(ray, axis=1, keepdims=True)) @ (np.linalg.inv(paint.CAMERA_TURN) @ [0, 1, 0]), 0.2))[:, None]
    new_world = new_cube @ paint.TO_GLTF.T
    mesh.vertices = trimesh.transform_points(new_world, np.linalg.inv(to_world))
    head.export(os.path.join(args.out, "refined.glb"), extension_webp=True)
    print(f"refined: {moved.sum()} of {len(world)} vertices moved -> {args.out}/refined.glb")
    # a picture of the moves
    img = np.zeros((res, res), np.float32)
    img[px[moved, 1], px[moved, 0]] = np.abs(step[moved])
    img = (img / max(img.max(), 1e-9) * 255).astype(np.uint8)
    Image.fromarray(img).save(os.path.join(args.out, "moves.jpg"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("model")
    parser.add_argument("--camera", required=True)
    parser.add_argument("--picture", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--tile", type=int, nargs=4, action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--resolution", default="1536")
    parser.add_argument("--max-move", type=float, default=0.03, help="a vertex moves at most this far (cube units)")
    parser.add_argument("--only-merge", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if not args.only_merge:
        make_tiles(args)
    merge(args)


if __name__ == "__main__":
    main()
