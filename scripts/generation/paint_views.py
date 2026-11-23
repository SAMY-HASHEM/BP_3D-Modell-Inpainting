"""(An experiment, Nov 23: on the motorcycle girl worse than the model's own colours. The cameras fitted to 79-92 %
outline overlap, but the generated shape is not the real one: each view lands a little beside its place, the 6 views
ghost over each other, and the green background bleeds in at the edges. A painter that makes its views from the
model's own shape (Hunyuan3D-Paint 2.1) is the next thing to try.)

Paints several pictures of an object onto its model (pipeline stage 7): every part of the texture takes its colour
from the pictures that see it, most from the one that sees it most straight on; what no picture sees keeps the model's
own colours. For views whose cameras nobody wrote down (frames of a turntable video, a character sheet):

  1 cameras   per picture, the camera is found by the outline: turned around the model (azimuth), raised (elevation)
              and with a field of view, the model's outline is drawn (by the GPU) and laid on the picture's; a coarse
              search, then all values together (with a roll, size and place) to the best overlap. --views gives each picture's rough angle (front 0, the model's left 90,
              back 180, its right -90; "side": either side); the search stays within 60 degrees of it
  2 line-up   the picture shifted onto the model's own colours there (paint_front.py's line_up_on, shift only)
  3 paint     each texel: seen by which pictures (depth test), how straight (full colour to 50 degrees, none from
              72); the pictures blended by that weight squared; texels no picture sees keep the model's colours

    python scripts/generation/paint_views.py model.glb --pictures front.png back.png side_a.png ... \\
        --views 0 180 side side 135 45 --out painted.glb

The model in glTF axes, front +Z (as generate.py / multi_image.py write them; the first picture was the front).
The pictures cut out (RGBA), ideally de-lit (delight.py). Writes painted.glb and painted_cameras.json, and prints each
camera's outline overlap.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import trimesh
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paint_front import bleed, fill_holes, line_up_on  # noqa: E402


def rotation(azimuth, elevation):
    """Model (glTF: x right, y up, front +z) -> camera axes (x right, y up, z towards the camera) for a camera at
    `azimuth` degrees around the model (0: in front, 90: at the model's left = +x) raised by `elevation`."""
    a, e = math.radians(azimuth), math.radians(elevation)
    turn = np.array([[math.cos(a), 0, -math.sin(a)], [0, 1, 0], [math.sin(a), 0, math.cos(a)]])  # about y
    tilt = np.array([[1, 0, 0], [0, math.cos(e), -math.sin(e)], [0, math.sin(e), math.cos(e)]])  # about x
    return tilt @ turn


class Camera:
    """A camera around the model: rotation, field of view; the model's centre at distance 2.5 x its size; then a
    2D fit (scale, centre) onto the picture."""

    def __init__(self, azimuth, elevation, fov, size, centre):
        self.azimuth, self.elevation, self.fov = azimuth, elevation, fov
        self.R = rotation(azimuth, elevation)
        self.distance = 2.5 * size / math.tan(math.radians(fov) / 2) * math.tan(math.radians(30) / 2)
        self.centre = centre
        self.scale, self.shift = 1.0, np.zeros(2)

    def camera_space(self, points):
        p = (points - self.centre) @ self.R.T
        return p, self.distance - p[:, 2]  # depth along the view

    def raw(self, points):
        p, depth = self.camera_space(points)
        focal = 1 / math.tan(math.radians(self.fov) / 2)
        return np.stack([focal * p[:, 0] / depth, -focal * p[:, 1] / depth], 1), depth  # y down

    def project(self, points):
        """-> picture pixels (x right, y down) and depth."""
        xy, depth = self.raw(points)
        return xy * self.scale + self.shift, depth

    def fit(self, points, mask):
        """Scale and place so that the outline's box lies on the mask's box."""
        xy, _ = self.raw(points)
        ys, xs = np.nonzero(mask)
        lo, hi = np.percentile(xy, 0.5, axis=0), np.percentile(xy, 99.5, axis=0)
        box_lo, box_hi = np.array([xs.min(), ys.min()]), np.array([xs.max(), ys.max()])
        self.scale = float(np.mean((box_hi - box_lo) / np.maximum(hi - lo, 1e-9)))
        self.shift = (box_lo + box_hi) / 2 - (lo + hi) / 2 * self.scale

    def to_json(self):
        return {"azimuth": self.azimuth, "elevation": self.elevation, "fov": self.fov, "scale": self.scale,
                "shift": self.shift.tolist()}


class Silhouettes:
    """The model's outline from a camera, drawn by the GPU (nvdiffrast) at a small size: exact, unlike splatted points
    (their outline had holes: the first fit stopped at 66-83 % overlap, the paint landed beside its place, Nov 23)."""

    def __init__(self, vertices, faces, size=200):
        import nvdiffrast.torch as dr
        self.dr, self.ctx = dr, dr.RasterizeCudaContext()
        self.v, self.size = vertices, size
        self.faces = torch.as_tensor(np.ascontiguousarray(faces), dtype=torch.int32, device="cuda")

    def overlap(self, cam, mask_small, k):
        xy, depth = cam.project(self.v)
        h, w = mask_small.shape
        x, y = xy[:, 0] * k, xy[:, 1] * k
        near, far = max(depth.min() * 0.5, 1e-3), depth.max() * 2
        z = (depth - near) / (far - near) * 2 - 1
        clip = np.stack([(x / w * 2 - 1) * depth, (1 - y / h * 2) * depth, z * depth, depth], 1)
        clip = torch.as_tensor(clip, dtype=torch.float32, device="cuda")
        rast, _ = self.dr.rasterize(self.ctx, clip[None], self.faces, (h, w))
        seen = (rast[0, ..., 3] > 0).flip(0).cpu().numpy()
        return (seen & mask_small).sum() / max((seen | mask_small).sum(), 1)


class FreeCamera(Camera):
    """A Camera with a roll and its 2D scale and place as free values (fitted, not from the box)."""

    def __init__(self, values, size, centre):
        az, el, roll, fov, scale, sx, sy = values
        super().__init__(float(az), float(el), float(np.clip(fov, 3, 60)), size, centre)
        r = math.radians(roll)
        self.R = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]]) @ self.R
        self.roll = float(roll)
        self.scale, self.shift = float(scale), np.array([sx, sy], float)

    def to_json(self):
        return {**super().to_json(), "roll": self.roll}


def find_camera(sil, mask, view, size, centre):
    """The camera whose outline fits the picture best, near the rough angle `view` ("side": either side): a coarse
    search (azimuth, elevation, field of view; size and place from the outline's box), then all seven values
    together (Nelder-Mead on the overlap)."""
    from scipy.optimize import minimize
    h, w = mask.shape
    k = sil.size / max(h, w)
    small = np.asarray(Image.fromarray(mask).resize((max(1, round(w * k)), max(1, round(h * k))))) > 0
    starts = [90, -90] if view == "side" else [float(view)]
    best = (-1, None)
    sample = sil.v[np.random.default_rng(0).choice(len(sil.v), min(len(sil.v), 30000), replace=False)]
    for start in starts:
        for az in np.arange(start - 50, start + 51, 10):
            for el in (-5, 5, 15, 25, 35):
                for fov in (10, 25, 40):
                    cam = Camera(float(az), float(el), fov, size, centre)
                    cam.fit(sample, mask)
                    score = sil.overlap(cam, small, k)
                    if score > best[0]:
                        best = (score, cam)
    score, cam = best
    x0 = np.array([cam.azimuth, cam.elevation, 0.0, cam.fov, cam.scale, *cam.shift])
    steps = np.array([4, 4, 3, 6, cam.scale * 0.04, 0.02 * w, 0.02 * h])
    simplex = np.vstack([x0] + [x0 + np.eye(7)[i] * steps[i] for i in range(7)])
    loss = lambda x: -sil.overlap(FreeCamera(x, size, centre), small, k)
    res = minimize(loss, x0, method="Nelder-Mead",
                   options={"initial_simplex": simplex, "maxiter": 400, "xatol": 0.05, "fatol": 1e-4})
    cam = FreeCamera(res.x, size, centre)
    return cam, -res.fun


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("model")
    parser.add_argument("--pictures", nargs="+", required=True)
    parser.add_argument("--views", nargs="+", required=True, help="rough angle per picture (degrees) or 'side'")
    parser.add_argument("--out", required=True)
    parser.add_argument("--from-angle", type=float, default=50)
    parser.add_argument("--to-angle", type=float, default=72)
    args = parser.parse_args()
    assert len(args.views) == len(args.pictures), "one --views entry per picture"
    import nvdiffrast.torch as dr

    scene = trimesh.load(args.model)
    name, mesh = next(iter(scene.geometry.items()))
    material = mesh.visual.material
    texture = np.asarray(material.baseColorTexture.convert("RGBA")).astype(np.float32) / 255
    tsize = texture.shape[0]
    to_world = scene.graph.get(scene.graph.nodes_geometry[0])[0]
    vertices = trimesh.transform_points(mesh.vertices, to_world)
    normals = mesh.vertex_normals @ to_world[:3, :3].T
    normals /= np.linalg.norm(normals, axis=1, keepdims=True).clip(1e-9)
    lo, hi = np.percentile(vertices, 0.5, axis=0), np.percentile(vertices, 99.5, axis=0)
    centre, size = (lo + hi) / 2, float(np.max(hi - lo)) / 2

    glctx = dr.RasterizeCudaContext()
    cuda = lambda a, t=torch.float32: torch.as_tensor(np.ascontiguousarray(a), dtype=t, device="cuda")
    faces = cuda(mesh.faces, torch.int32)
    sil = Silhouettes(vertices, mesh.faces)
    uv = mesh.visual.uv
    clip_uv = np.stack([uv[:, 0] * 2 - 1, uv[:, 1] * 2 - 1, np.zeros(len(uv)), np.ones(len(uv))], 1)
    rast_uv, _ = dr.rasterize(glctx, cuda(clip_uv)[None], faces, (tsize, tsize))
    covered = (rast_uv[0, ..., 3] > 0).flip(0).cpu().numpy()
    place, _ = dr.interpolate(cuda(vertices), rast_uv, faces)
    facing, _ = dr.interpolate(cuda(normals), rast_uv, faces)
    place, facing = place[0].flip(0).cpu().numpy()[covered], facing[0].flip(0).cpu().numpy()[covered]
    facing /= np.linalg.norm(facing, axis=1, keepdims=True).clip(1e-9)

    total = np.zeros((len(place), 3), np.float64)
    weight_sum = np.zeros(len(place), np.float64)
    best_weight = np.zeros(len(place), np.float64)
    cameras = []
    for path, view in zip(args.pictures, args.views):
        picture = Image.open(path).convert("RGBA")
        rgba = np.asarray(picture)
        mask = rgba[..., 3] > 127
        cam, score = find_camera(sil, mask, view, size, centre)
        cameras.append({"picture": path, **cam.to_json(), "overlap": round(float(score), 3)})
        print(f"{os.path.basename(path)}: azimuth {cam.azimuth:.1f}, elevation {cam.elevation:.1f}, roll "
              f"{getattr(cam, 'roll', 0):.1f}, fov {cam.fov:.0f}: outline overlap {score:.1%}", flush=True)
        H, W = mask.shape

        # the model from this camera: depth per pixel and its own colours (to line the picture up on)
        xy, depth = cam.project(vertices)
        near, far = depth.min() * 0.5, depth.max() * 2
        z = (depth - near) / (far - near) * 2 - 1
        clip = np.stack([(xy[:, 0] / W * 2 - 1) * depth, (1 - xy[:, 1] / H * 2) * depth, z * depth, depth], 1)
        rast, _ = dr.rasterize(glctx, cuda(clip)[None], faces, (H, W))
        depth_map, _ = dr.interpolate(cuda(depth[:, None]), rast, faces)
        depth_map = torch.where(rast[..., 3:] > 0, depth_map, torch.full_like(depth_map, 1e9))[0, ..., 0].flip(0)
        depth_map = depth_map.cpu().numpy()
        tex_uv, _ = dr.interpolate(cuda(np.stack([uv[:, 0], 1 - uv[:, 1]], 1)), rast, faces)
        own = dr.texture(cuda(texture[None, ..., :3]), tex_uv, filter_mode="linear")[0].flip(0).cpu().numpy()
        own_mask = (rast[0, ..., 3] > 0).flip(0).cpu().numpy()

        rgb = rgba[..., :3].astype(np.float32) / 255
        alpha = np.asarray(Image.fromarray(rgba[..., 3]).filter(ImageFilter.MinFilter(5))).astype(np.float32) / 255
        import cv2
        warp, match = line_up_on(own, own_mask & (alpha > 0.5), rgb, shift_only=True, limit=0.03 * max(H, W))
        if warp is not None:
            rgb = cv2.warpAffine(rgb, warp, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                 borderMode=cv2.BORDER_REPLICATE)
            alpha = cv2.warpAffine(alpha, warp, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        print(f"    lined up: shift {np.round(warp[:, 2], 1).tolist() if warp is not None else None} px "
              f"(match {match if match is None else round(float(match), 2)})", flush=True)

        # the texels: seen? how straight?
        pxy, pdepth = cam.project(place)
        inside = (pxy[:, 0] >= 0) & (pxy[:, 1] >= 0) & (pxy[:, 0] < W - 1) & (pxy[:, 1] < H - 1)
        px = np.clip(pxy.astype(int), 0, [W - 1, H - 1])
        seen = inside & (pdepth <= depth_map[px[:, 1], px[:, 0]] + 0.01 * size)
        eye = cam.centre + cam.R.T @ np.array([0, 0, cam.distance])
        look = eye - place
        look /= np.linalg.norm(look, axis=1, keepdims=True)
        cos = (facing * look).sum(1)
        lo_c, hi_c = math.cos(math.radians(args.to_angle)), math.cos(math.radians(args.from_angle))
        t = np.clip((cos - lo_c) / (hi_c - lo_c), 0, 1)
        w = (t * t * (3 - 2 * t)) * seen * alpha[px[:, 1], px[:, 0]]
        total += (w ** 2)[:, None] * rgb[px[:, 1], px[:, 0]]
        weight_sum += w ** 2
        best_weight = np.maximum(best_weight, w)
        print(f"    paints {(w > 0.5).mean():.1%} of the texels", flush=True)

    colour = total / np.maximum(weight_sum, 1e-9)[:, None]
    painted = np.zeros((tsize, tsize, 3), np.float32)
    weights = np.zeros((tsize, tsize), np.float32)
    painted[covered], weights[covered] = colour, best_weight
    weights, painted = fill_holes(weights, painted, covered)
    texture[..., :3] = texture[..., :3] * (1 - weights[..., None]) + painted * weights[..., None]
    texture = bleed(texture, covered)
    material.baseColorTexture = Image.fromarray((texture * 255).round().clip(0, 255).astype(np.uint8))
    scene.export(args.out, extension_webp=True)
    json.dump(cameras, open(os.path.splitext(args.out)[0] + "_cameras.json", "w"), indent=1)
    print(f"painted {(weights[covered] > 0.5).mean():.1%} of the texels from the pictures -> {args.out}")


if __name__ == "__main__":
    main()
