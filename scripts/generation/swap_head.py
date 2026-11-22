"""Puts a head made alone (more detail: it fills TRELLIS.2's / Pixal3D's whole cube) onto a whole-body model, in
place of the body's own head (a face is only ~40 of 1024 steps tall in a whole-body model: flat, blobby).

Both models are Pixal3D's, so both cameras are known, and their pictures are tied together: the head's picture is a
cut-out of the close-up, the close-up is matched to the body's front picture by its features (SIFT), and that
picture is lined up on the body model's own colours (paint_front.line_up_on). So each pixel of the head's face
gives a pair of 3D points: where the head model has it and where the body model has it (both drawn from their
cameras). The best scale + turn + shift for those pairs (Umeyama, the worst 30 % left out twice) puts the head in.
Then the body's head is cut away (its faces inside the new head, above the neck cut) and the new head, from the neck
up, is added as its own mesh (own texture). The halo and the armour stay the body's.

    python scripts/generation/swap_head.py body.glb --body-camera body.json --body-picture body_colour_input_rgba.png \
        head.glb --head-camera head.json --head-picture head_input_rgba.png \
        --closeup "close-up.jpg" --closeup-box 1100 0 2359 1573 --head-crop 1500 260 --out swapped.glb

--head-crop: where the head's picture (before Pixal3D's own cut-out) was cut from the close-up picture.
--neck: where the new head ends, as a share of the head picture's height from the top (default 0.70: inside the
collar, so the seam is hidden). --corners: the picture's lower corners are left out (the crop's cut armour).
"""
import argparse
import json
import math

import numpy as np
import torch
import trimesh
from PIL import Image

import paint_front as paint


def load(path):
    scene = trimesh.load(path)
    mesh = next(iter(scene.geometry.values()))
    to_world = scene.graph.get(scene.graph.nodes_geometry[0])[0]
    mesh = mesh.copy()
    mesh.apply_transform(to_world)
    return mesh  # glTF axes


def positions(mesh, camera, res):
    """The model drawn from its Pixal3D camera: each pixel's 3D point (glTF axes) and whether there is one."""
    import nvdiffrast.torch as dr
    fov, distance = math.radians(camera["fov_degrees"]), camera["distance"]
    cube = np.asarray(mesh.vertices) @ np.linalg.inv(paint.TO_GLTF).T
    uv, depth = paint.project(cube, paint.CAMERA_TURN, fov, distance)
    near, far = depth.min() * 0.5, depth.max() * 2
    z = (depth - near) / (far - near) * 2 - 1
    clip = np.stack([(uv[:, 0] * 2 - 1) * depth, (1 - uv[:, 1] * 2) * depth, z * depth, depth], 1)
    cuda = lambda a, t=torch.float32: torch.as_tensor(np.ascontiguousarray(a), dtype=t, device="cuda")
    glctx = dr.RasterizeCudaContext()
    faces = cuda(mesh.faces, torch.int32)
    rast, _ = dr.rasterize(glctx, cuda(clip)[None], faces, (res, res))
    pos, _ = dr.interpolate(cuda(np.asarray(mesh.vertices)), rast, faces)
    mask = (rast[0, ..., 3] > 0).flip(0).cpu().numpy()
    return pos[0].flip(0).cpu().numpy(), mask


def own_colours(mesh, camera, res):
    """The model's own colours drawn from its camera (to line its picture up on)."""
    import nvdiffrast.torch as dr
    fov, distance = math.radians(camera["fov_degrees"]), camera["distance"]
    cube = np.asarray(mesh.vertices) @ np.linalg.inv(paint.TO_GLTF).T
    uv_cam, depth = paint.project(cube, paint.CAMERA_TURN, fov, distance)
    near, far = depth.min() * 0.5, depth.max() * 2
    z = (depth - near) / (far - near) * 2 - 1
    clip = np.stack([(uv_cam[:, 0] * 2 - 1) * depth, (1 - uv_cam[:, 1] * 2) * depth, z * depth, depth], 1)
    cuda = lambda a, t=torch.float32: torch.as_tensor(np.ascontiguousarray(a), dtype=t, device="cuda")
    glctx = dr.RasterizeCudaContext()
    faces = cuda(mesh.faces, torch.int32)
    rast, _ = dr.rasterize(glctx, cuda(clip)[None], faces, (res, res))
    uv = mesh.visual.uv
    tex_uv, _ = dr.interpolate(cuda(np.stack([uv[:, 0], 1 - uv[:, 1]], 1)), rast, faces)
    texture = np.asarray(mesh.visual.material.baseColorTexture.convert("RGB")).astype(np.float32) / 255
    own = dr.texture(cuda(texture[None]), tex_uv, filter_mode="linear")[0].flip(0).cpu().numpy()
    return own, (rast[0, ..., 3] > 0).flip(0).cpu().numpy()


def similarity(a, b):
    """Scale s, turn R, shift t with s R a + t ~ b (Umeyama)."""
    ma, mb = a.mean(0), b.mean(0)
    A, B = a - ma, b - mb
    U, S, Vt = np.linalg.svd(B.T @ A / len(a))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / (A ** 2).sum(1).mean()
    return s, R, mb - s * R @ ma


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("body")
    parser.add_argument("--body-camera", required=True)
    parser.add_argument("--body-picture", required=True)
    parser.add_argument("head")
    parser.add_argument("--head-camera", required=True)
    parser.add_argument("--head-picture", required=True, help="the head's picture as Pixal3D saw it (_input_rgba)")
    parser.add_argument("--head-source", required=True, help="the cut-out the head was made from (before Pixal3D)")
    parser.add_argument("--closeup", required=True)
    parser.add_argument("--closeup-box", type=int, nargs=4, required=True)
    parser.add_argument("--head-crop", type=int, nargs=2, required=True, help="x y of the head's cut in the close-up")
    parser.add_argument("--neck", type=float, default=0.61,
                        help="where the new head ends: share of its picture's height from the top (above its collar)")
    parser.add_argument("--body-cut", type=float, default=0.57,
                        help="the body's own head goes above this (in the same shares): higher than --neck, so the "
                             "two necks overlap inside and leave no gap")
    parser.add_argument("--fit-above", type=float, default=0.57, help="only the face and hair are fitted")
    parser.add_argument("--corners", type=float, default=0.55,
                        help="below this share of its picture's height, its left and right quarters are left out")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    body, head = load(args.body), load(args.head)
    body_cam = json.load(open(args.body_camera))["camera"]
    head_cam = json.load(open(args.head_camera))["camera"]
    body_pic = np.asarray(Image.open(args.body_picture).convert("RGBA")).astype(np.float32) / 255
    head_pic = Image.open(args.head_picture).convert("RGBA")
    rb, rh = body_pic.shape[0], head_pic.width

    # head picture px -> close-up px: Pixal3D's own cut-out of the source (its square around the object, x 1.1)
    src = np.asarray(Image.open(args.head_source).convert("RGBA"))
    ys, xs = np.nonzero(src[..., 3] > 0.8 * 255)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    size = int(max(xs.max() - xs.min(), ys.max() - ys.min()) * 1.1)
    corner = np.array([cx - size // 2, cy - size // 2]) + np.array(args.head_crop) - np.array(args.closeup_box[:2])
    head_to_close = lambda p: p * (size / rh) + corner
    # close-up px -> body picture px (features), then body picture px -> body model px (lined up on its colours)
    close = np.asarray(Image.open(args.closeup).convert("RGB")).astype(np.float32) / 255
    x0, y0, x1, y1 = args.closeup_box
    to_body_pic = paint.match_features(close[y0:y1, x0:x1], body_pic[..., :3], body_pic[..., 3])
    own, own_mask = own_colours(body, body_cam, rb)
    whole, score = paint.line_up_on(own, own_mask & (body_pic[..., 3] > 0.5), body_pic[..., :3])
    whole = np.eye(2, 3) if whole is None else whole
    print(f"body picture lined up on the body: shift {whole[0, 2]:+.1f} {whole[1, 2]:+.1f} px ({score:.2f})")
    inverse = np.linalg.inv(np.vstack([whole, [0, 0, 1]]))[:2]  # picture px -> model px

    # pairs of 3D points: every head-picture pixel of the face and hair (above the neck)
    head_pos, head_mask = positions(head, head_cam, rh)
    body_pos, body_mask = positions(body, body_cam, rb)
    ys, xs = np.nonzero(head_mask)
    keep = ys < args.fit_above * rh
    ys, xs = ys[keep][::7], xs[keep][::7]
    p = head_to_close(np.stack([xs, ys], 1).astype(np.float64) + 0.5)
    p = p @ to_body_pic[:, :2].T + to_body_pic[:, 2]
    p = p @ inverse[:, :2].T + inverse[:, 2]
    bx, by = np.round(p[:, 0] - 0.5).astype(int), np.round(p[:, 1] - 0.5).astype(int)
    ok = (bx >= 0) & (by >= 0) & (bx < rb) & (by < rb)
    ok[ok] &= body_mask[by[ok], bx[ok]]
    a, b = head_pos[ys[ok], xs[ok]], body_pos[by[ok], bx[ok]]
    print(f"{len(a)} pairs of points")
    use = np.ones(len(a), bool)
    for _ in range(3):  # fit, leave out the worst 30 %, fit again
        s, R, t = similarity(a[use], b[use])
        err = np.linalg.norm((s * a @ R.T + t) - b, axis=1)
        use = err <= np.quantile(err, 0.7)
    print(f"head into the body: scale {s:.4f}, turn {math.degrees(math.acos(np.clip((np.trace(R) - 1) / 2, -1, 1))):.1f}"
          f" deg, error {np.median(err) / (body.bounds[1, 1] - body.bounds[0, 1]) * 100:.2f} % of her height")

    # the new head in the body's place, from the neck up
    original_head_vertices = np.asarray(head.vertices).copy()
    head.vertices = s * original_head_vertices @ R.T + t
    def height(share):  # the height (glTF y, in the body) of the head picture's row at `share`
        row = int(share * rh)
        points = head_pos[row][head_mask[row]]
        return float(np.median((s * points @ R.T + t)[:, 1])) if len(points) else None
    neck_y, cut_y = height(args.neck), height(args.body_cut)
    print(f"the new head ends at {neck_y:.4f}, the body's head goes above {cut_y:.4f}")
    head_faces = head.vertices[head.faces][:, :, 1].min(1) >= neck_y
    # and nothing from the head picture's lower corners (its crop cut through the shoulder armour there)
    cube = original_head_vertices @ np.linalg.inv(paint.TO_GLTF).T
    uv, _ = paint.project(cube, paint.CAMERA_TURN, math.radians(head_cam["fov_degrees"]), head_cam["distance"])
    corner = (uv[:, 1] > args.corners) & ((uv[:, 0] < 0.25) | (uv[:, 0] > 0.75))
    head_faces &= ~corner[head.faces].any(1)
    head.update_faces(head_faces)
    head.remove_unreferenced_vertices()
    # loose pieces away: only the parts joined to the head's biggest piece, or hair strands above the chin
    labels = trimesh.graph.connected_component_labels(head.face_adjacency, node_count=len(head.faces))
    biggest = np.bincount(labels).argmax()
    centres_h = head.vertices[head.faces].mean(1)
    chin = height(args.fit_above)
    keep_piece = np.zeros(labels.max() + 1, bool)
    keep_piece[biggest] = True
    for k in np.unique(labels):
        if k != biggest and np.median(centres_h[labels == k][:, 1]) > chin:
            keep_piece[k] = True
    head.update_faces(keep_piece[labels])
    head.remove_unreferenced_vertices()
    # the body's own head away: its faces inside the new head (or within a voxel of it), above the neck
    voxel = (body.bounds[1] - body.bounds[0]).max() / 64
    from scipy.spatial import cKDTree
    centres = body.vertices[body.faces].mean(1)
    near = cKDTree(head.vertices).query(centres)[0] < voxel
    inside = np.zeros(len(centres), bool)
    above = centres[:, 1] > cut_y
    from scipy.spatial import Delaunay
    hull = Delaunay(trimesh.convex.convex_hull(head.vertices).vertices)  # the new head's hull (its corners)
    inside[above] = hull.find_simplex(centres[above]) >= 0
    gone = above & (near | inside)
    body.update_faces(~gone)
    body.remove_unreferenced_vertices()
    print(f"body: {gone.sum()} faces of its head taken away; the new head: {len(head.faces)} faces")

    scene = trimesh.Scene()
    scene.add_geometry(body, node_name="body")
    scene.add_geometry(head, node_name="head")
    scene.export(args.out, extension_webp=True)
    print("->", args.out)


if __name__ == "__main__":
    main()
