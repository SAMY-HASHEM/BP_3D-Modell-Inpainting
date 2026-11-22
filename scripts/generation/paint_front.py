"""Paints the picture itself onto a Pixal3D model where its camera saw it: the front then has exactly the picture's
colours (face, lips, emblems), the model's own colours stay where the picture can't see (sides, back).

Pixal3D (compare_pixal3d.py) makes the model in its camera's frame and writes the camera (field of view, distance) to
<name>.json, so each texel of the model's texture can be looked up in the picture:
  1. the texel's place and facing on the model (the texture's triangles drawn in UV space, nvdiffrast)
  2. seen? its depth against the depth the camera sees there (the model drawn from the camera)
  3. its colour = the picture's, blended by how straight the surface faces the camera (none from 70 degrees off on)

    python scripts/generation/paint_front.py outputs/x/F_pixal3d.glb --picture outputs/x/F_pixal3d_input_rgba.png \
        --camera outputs/x/F_pixal3d.json --out outputs/x/F_pixal3d_painted.glb

The camera: Pixal3D's, from the front (CAMERA_TURN). Printed as a check: how well the model's outline through it lies
on the picture's (about 75 % here: the picture's own outline is soft and the model not exact). (Nov 22: finding the
axes by the outline alone chose the camera behind the model, mirrored, and painted the face onto the back.)
"""
import argparse
import json
import math

import numpy as np
import torch
import trimesh
from PIL import Image, ImageFilter

TO_GLTF = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)  # compare_pixal3d.py's turn of the model


def project(points, turn, fov, distance):
    """Model points (cube frame) -> picture coordinates 0..1 (x right, y down) and depth, Pixal3D's camera:
    world = turn @ p, the camera on -y at `distance`, looking along +y, z up (inference.distance_from_fov)."""
    w = points @ turn.T
    depth = distance + w[:, 1]
    focal = 0.5 / math.tan(fov / 2)  # in picture widths
    return np.stack([0.5 + focal * w[:, 0] / depth, 0.5 - focal * w[:, 2] / depth], 1), depth


# Pixal3D's camera looks at the model's front from (0, 0, +distance) with Y up (render_utils.
# proj_camera_to_render_params), the front being glTF's +Z after compare_pixal3d.py's turn. In project()'s terms:
# world x = glTF x, world y (away from the camera) = -glTF z, world z (up) = glTF y; from the cube frame:
CAMERA_TURN = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float)


def outline_overlap(points, alpha, turn, fov, distance, size=160):
    """How well the model's outline through the camera lies on the picture's (a check that the camera is right;
    a front and a back view have nearly the same outline, so it can't choose between them)."""
    from PIL import ImageFilter
    target = np.asarray(Image.fromarray(alpha).resize((size, size))) > 127
    sample = points[np.random.default_rng(0).choice(len(points), min(len(points), 200_000), replace=False)]
    uv, depth = project(sample, turn, fov, distance)
    ok = (depth > 0) & (uv.min(1) >= 0) & (uv.max(1) < 1)
    seen = np.zeros((size, size), bool)
    seen[(uv[ok, 1] * size).astype(int), (uv[ok, 0] * size).astype(int)] = True
    seen = np.asarray(Image.fromarray(seen).filter(ImageFilter.MaxFilter(3)))
    return (seen & target).sum() / max((seen | target).sum(), 1)


def gray(image):
    import cv2
    g = cv2.cvtColor((np.clip(image, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return cv2.equalizeHist(g).astype(np.float32) / 255


def cv2_warp(image, warp, size, inverse=True):
    """image moved by a 2x3 warp to a size x size picture (inverse: warp maps the result's pixels into image)."""
    import cv2
    flags = cv2.INTER_LINEAR | (cv2.WARP_INVERSE_MAP if inverse else 0)
    return cv2.warpAffine(np.ascontiguousarray(image, np.float32), np.asarray(warp, np.float32), (size, size),
                          flags=flags, borderMode=cv2.BORDER_CONSTANT)


def line_up_on(target, mask, image, window=False, start=None, limit=None, shift_only=False):
    """The small affine move of `image` that fits the model's own colours (`target`) best where `mask`
    (OpenCV's ECC on blurred, equalised greys: picture lighting vs texture colours differ, their edges agree).
    window: only the box around the mask is compared (a part); start: the move to begin from. Rejected (None)
    if it would go more than `limit` pixels from the start (default 4 % of the picture) or scale by over 8 %."""
    import cv2
    full = target.shape[0]
    limit = limit or 0.04 * full
    start = np.eye(2, 3, dtype=np.float32) if start is None else np.asarray(start, np.float32)
    t, i, m = gray(target), gray(image), mask.astype(np.uint8)
    t, i = cv2.GaussianBlur(t, (0, 0), 1.5), cv2.GaussianBlur(i, (0, 0), 1.5)
    offset = np.zeros(2, np.float32)
    if window:  # both cut to the box around the part (with a margin for the move)
        ys, xs = np.nonzero(m)
        pad = int(0.03 * full)
        y0, y1 = max(ys.min() - pad, 0), min(ys.max() + 1 + pad, full)
        x0, x1 = max(xs.min() - pad, 0), min(xs.max() + 1 + pad, full)
        t, i, m, offset = t[y0:y1, x0:x1], i[y0:y1, x0:x1], m[y0:y1, x0:x1], np.float32([x0, y0])
    warp = start.copy()
    warp[:, 2] = start[:, :2] @ offset + start[:, 2] - offset  # the start move in the box's pixels
    try:
        motion = cv2.MOTION_TRANSLATION if shift_only else cv2.MOTION_AFFINE
        score, warp = cv2.findTransformECC(t, i, warp, motion,
                                           (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-6), m, 5)
    except cv2.error as e:
        print("    (no fit:", str(e).strip().splitlines()[-1][:90] + ")", flush=True)
        return None, 0.0
    warp[:, 2] = warp[:, 2] - warp[:, :2] @ offset + offset  # back to the whole picture's pixels
    scale = np.sqrt(abs(np.linalg.det(warp[:, :2])))
    if np.abs(warp[:, 2] - start[:, 2]).max() > limit or abs(scale - 1) > 0.08:
        print(f"    (fit thrown away: moves {warp[0, 2] - start[0, 2]:+.1f} {warp[1, 2] - start[1, 2]:+.1f} px "
              f"from the start, scale {scale:.3f})", flush=True)
        return None, 0.0
    return warp, score


HEAD_REGIONS = {  # the same for a head / bust: the face in the middle, the hair above on each side
    "face": ((0.28, 0.80), (0.22, 0.78)),
    "hair left": ((0.0, 0.45), (0.0, 0.56)),
    "hair right": ((0.0, 0.45), (0.44, 1.0)),
    "lower left": ((0.60, 1.0), (0.0, 0.56)),
    "lower right": ((0.60, 1.0), (0.44, 1.0)),
}
REGIONS = {  # (top, bottom) as shares of the figure's height, (left, right) as shares of its width; overlapping
    "face": ((0.10, 0.26), (0.36, 0.64)),
    "upper body left": ((0.18, 0.58), (0.0, 0.56)),
    "upper body right": ((0.18, 0.58), (0.44, 1.0)),
    "lower body left": ((0.52, 1.0), (0.0, 0.56)),
    "lower body right": ((0.52, 1.0), (0.44, 1.0)),
}


def line_up_by_regions(own, mask, rgb, alpha, regions=None):
    """The picture lined up on the model part by part (REGIONS: face, upper and lower body, each side): a
    generated model is not shifted the same everywhere (its face a little lower, a leg a little wider), so each
    part gets its own small move; the parts are blended softly where they overlap. Picture's left = screen left."""
    import cv2
    res = rgb.shape[0]
    whole, score = line_up_on(own, mask, rgb)
    whole = np.eye(2, 3, dtype=np.float32) if whole is None else whole
    print(f"  whole picture: shift {whole[0, 2]:+.1f} {whole[1, 2]:+.1f} px (match {score:.2f})", flush=True)
    ys, xs = np.nonzero(mask)
    y0, h, x0, w = ys.min(), ys.max() - ys.min(), xs.min(), xs.max() - xs.min()
    base_rgb, base_alpha = cv2_warp(rgb, whole, res), cv2_warp(alpha, whole, res)
    total = np.full((res, res), 0.05, np.float32)  # the whole picture's move where no part reaches
    sum_rgb, sum_alpha = base_rgb * 0.05, base_alpha * 0.05
    for name, ((top, bottom), (left, right)) in (regions or REGIONS).items():
        box = np.zeros((res, res), np.float32)
        r0, r1 = int(y0 + top * h), int(y0 + bottom * h)
        c0, c1 = int(x0 + left * w), int(x0 + right * w)
        box[r0:r1, c0:c1] = 1
        part = mask & (box > 0)
        # each part: only a shift of its own, on top of the whole picture's move (an affine fit runs away in a
        # small window of mostly black armour)
        step, score = (line_up_on(own, part, base_rgb, window=True, limit=0.02 * res, shift_only=True)
                       if part.sum() > 500 else (None, 0.0))
        d = np.zeros(2, np.float32) if step is None else step[:, 2]
        warp = whole.copy()
        warp[:, 2] += whole[:, :2] @ d
        print(f"  {name}: its own shift {d[0]:+.1f} {d[1]:+.1f} px (match {score:.2f})", flush=True)
        soft = cv2.GaussianBlur(box, (0, 0), max(4, 0.03 * h))
        soft *= 3 if name == "face" else 1  # the face wins where it overlaps
        total += soft
        sum_rgb += cv2_warp(rgb, warp, res) * soft[..., None]
        sum_alpha += cv2_warp(alpha, warp, res) * soft
    return sum_rgb / total[..., None], sum_alpha / total


def bend_onto(target, mask, image, alpha, strength=1.0):
    """Every pixel of `image` moved a little on its own (optical flow, OpenCV's DIS) so that it lands on the same
    feature of `target` (the model's own colours, which sit exactly on its shape): a picture whose face is shaped a
    little differently (rounder, wider eyes) then puts its lips on the 3D lips, its eyes in the 3D eyes, instead of
    beside them (the doubled, fuzzy look). The flow is smoothed and only used inside the mask."""
    import cv2
    t = (gray(target) * 255).astype(np.uint8)
    i = (gray(image) * 255).astype(np.uint8)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flow = dis.calc(t, i, None)  # for each target pixel: where its feature is in the image
    flow = cv2.GaussianBlur(flow, (0, 0), 3)
    limit = 0.03 * target.shape[0]
    flow = np.clip(flow, -limit, limit) * strength
    m = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 6)[..., None]
    flow *= m
    h, w = t.shape
    grid = np.stack(np.meshgrid(np.arange(w), np.arange(h)), -1).astype(np.float32) + flow
    moved = cv2.remap(np.ascontiguousarray(image, np.float32), grid[..., 0], grid[..., 1], cv2.INTER_LINEAR)
    moved_alpha = cv2.remap(np.ascontiguousarray(alpha, np.float32), grid[..., 0], grid[..., 1], cv2.INTER_LINEAR)
    print(f"  picture bent onto the model: mean move {np.linalg.norm(flow, axis=2)[mask].mean():.1f} px, "
          f"largest {np.linalg.norm(flow, axis=2)[mask].max():.1f} px", flush=True)
    return moved, moved_alpha


def match_features(close, picture, alpha):
    """2x3 map from the close-up's pixels to the picture's (SIFT features, RANSAC; scale + turn + shift)."""
    import cv2
    sift = cv2.SIFT_create(8000)
    k1, d1 = sift.detectAndCompute((gray(close) * 255).astype(np.uint8), None)
    k2, d2 = sift.detectAndCompute((gray(picture) * 255).astype(np.uint8), (alpha > 0.1).astype(np.uint8))
    pairs = [m for m, n in cv2.BFMatcher().knnMatch(d1, d2, k=2) if m.distance < 0.75 * n.distance]
    a = np.float32([k1[m.queryIdx].pt for m in pairs])
    b = np.float32([k2[m.trainIdx].pt for m in pairs])
    warp, inliers = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=3)
    print(f"close-up matched to the picture: {int(inliers.sum())} of {len(pairs)} features agree, "
          f"scale {np.sqrt(abs(np.linalg.det(warp[:, :2]))):.3f}", flush=True)
    return warp


def fill_holes(weights, painted, covered):
    """Texels inside a painted area that got nothing (a failed depth test at a crease, a texel between two
    triangles) take the mean of their painted neighbours: the thin dark cracks of the first paintings."""
    import cv2
    w = cv2.blur(weights, (5, 5))
    p = cv2.blur(painted * weights[..., None], (5, 5))
    hole = covered & (weights < 0.2) & (w > 0.4)
    painted[hole] = p[hole] / w[hole, None]
    weights[hole] = w[hole]
    return weights, painted


def bleed(texture, covered, pixels=8):
    """Texels outside every piece of the texture (around its seams, which a renderer samples too) take the
    colour of the nearest texel inside: no dark lines along the seams."""
    from scipy.ndimage import distance_transform_edt
    dist, (iy, ix) = distance_transform_edt(~covered, return_indices=True)
    near = (dist > 0) & (dist <= pixels)
    texture[near] = texture[iy[near], ix[near]]
    return texture


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("model")
    parser.add_argument("--picture", required=True, help="the picture as Pixal3D saw it, with its outline "
                        "(<name>_input_rgba.png; or <name>_colour_input_rgba.png for the colour picture)")
    parser.add_argument("--camera", required=True, help="<name>.json of compare_pixal3d.py")
    parser.add_argument("--out", required=True)
    parser.add_argument("--view-angle", type=float, default=0,
                        help="the picture's camera around the model (degrees; 180 = a back view of multi-view Pixal3D)")
    parser.add_argument("--closeup", help="a nearer picture of the same view (e.g. the face): painted over, sharper")
    parser.add_argument("--closeup-box", type=int, nargs=4, help="x0 y0 x1 y1: the close-up's part of that picture")
    parser.add_argument("--parts", choices=["body", "head"], default="body",
                        help="the 5 parts lined up one by one: a whole body's or a head's (a bust)")
    parser.add_argument("--flow", action="store_true",
                        help="after lining up, bend the picture pixel by pixel onto the model's own colours "
                             "(optical flow). An experiment: on the girl's face (Nov 22) it followed the shading, "
                             "not the features, and painted skin over the eyes; better left off")
    parser.add_argument("--no-align", action="store_true", help="don't line the picture up on the model first")
    parser.add_argument("--from-angle", type=float, default=50, help="full picture colour up to this angle (deg)")
    parser.add_argument("--to-angle", type=float, default=72, help="none of it from this angle on (deg)")
    args = parser.parse_args()
    import nvdiffrast.torch as dr

    camera = json.load(open(args.camera))["camera"]
    fov, distance = math.radians(camera["fov_degrees"]), camera["distance"]
    picture = Image.open(args.picture)
    if picture.mode != "RGBA":  # no outline kept (older runs): the background is black
        alpha = Image.fromarray(((np.asarray(picture.convert("RGB")).max(2) > 6) * 255).astype(np.uint8))
        picture = picture.convert("RGB")
        picture.putalpha(alpha)
    rgba = np.asarray(picture)
    # the picture's edge is half background: only what lies well inside its outline
    alpha = np.asarray(Image.fromarray(rgba[..., 3]).filter(ImageFilter.MinFilter(5))).astype(np.float32) / 255

    scene = trimesh.load(args.model)
    name, mesh = next(iter(scene.geometry.items()))
    material = mesh.visual.material
    texture = np.asarray(material.baseColorTexture.convert("RGBA")).astype(np.float32) / 255
    size = texture.shape[0]
    to_world = scene.graph.get(scene.graph.nodes_geometry[0])[0]  # the mesh's place in the scene
    vertices = trimesh.transform_points(mesh.vertices, to_world) @ np.linalg.inv(TO_GLTF).T  # back to Pixal3D's cube
    normals = mesh.vertex_normals @ to_world[:3, :3].T @ np.linalg.inv(TO_GLTF).T
    normals /= np.linalg.norm(normals, axis=1, keepdims=True).clip(1e-9)

    a = math.radians(-args.view_angle)  # the camera at view_angle = the model turned the other way, seen from the front
    turn = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]]) @ CAMERA_TURN
    print(f"camera check: outline overlap {outline_overlap(vertices, rgba[..., 3], turn, fov, distance):.1%}",
          flush=True)

    glctx = dr.RasterizeCudaContext()
    cuda = lambda a, t=torch.float32: torch.as_tensor(np.ascontiguousarray(a), dtype=t, device="cuda")
    faces = cuda(mesh.faces, torch.int32)

    # 1. the model from the camera: its depth per pixel, and its own colours there (to line the picture up on)
    uv_cam, depth = project(vertices, turn, fov, distance)
    near, far = depth.min() * 0.5, depth.max() * 2
    z = (depth - near) / (far - near) * 2 - 1
    clip = np.stack([(uv_cam[:, 0] * 2 - 1) * depth, (1 - uv_cam[:, 1] * 2) * depth, z * depth, depth], 1)
    res = picture.width
    rast, _ = dr.rasterize(glctx, cuda(clip)[None], faces, (res, res))
    depth_map, _ = dr.interpolate(cuda(depth[:, None]), rast, faces)
    depth_map = torch.where(rast[..., 3:] > 0, depth_map, torch.full_like(depth_map, 1e9))[0, ..., 0].flip(0)
    uv = mesh.visual.uv
    tex_uv, _ = dr.interpolate(cuda(np.stack([uv[:, 0], 1 - uv[:, 1]], 1)), rast, faces)  # texture rows top down
    own = dr.texture(cuda(texture[None, ..., :3]), tex_uv, filter_mode="linear")[0].flip(0).cpu().numpy()
    own_mask = (rast[0, ..., 3] > 0).flip(0).cpu().numpy()

    # 2. every texel: its place and facing (the triangles drawn in UV space)
    clip_uv = np.stack([uv[:, 0] * 2 - 1, uv[:, 1] * 2 - 1, np.zeros(len(uv)), np.ones(len(uv))], 1)
    rast_uv, _ = dr.rasterize(glctx, cuda(clip_uv)[None], faces, (size, size))
    covered = (rast_uv[0, ..., 3] > 0).flip(0).cpu().numpy()  # texture rows go top to bottom, UV v bottom to top
    place, _ = dr.interpolate(cuda(vertices), rast_uv, faces)
    facing, _ = dr.interpolate(cuda(normals), rast_uv, faces)
    place, facing = place[0].flip(0).cpu().numpy()[covered], facing[0].flip(0).cpu().numpy()[covered]

    # 3. seen, and how straight
    uv_tex, depth_tex = project(place, turn, fov, distance)
    inside = (uv_tex.min(1) >= 0) & (uv_tex.max(1) < 1)
    px = np.clip((uv_tex * res).astype(int), 0, res - 1)
    seen = inside & (depth_tex <= depth_map.cpu().numpy()[px[:, 1], px[:, 0]] + 0.015)
    eye = (np.linalg.inv(turn) @ np.array([0, -distance, 0]))  # the camera's place in the cube frame
    view = eye - place
    view /= np.linalg.norm(view, axis=1, keepdims=True)
    facing /= np.linalg.norm(facing, axis=1, keepdims=True).clip(1e-9)
    cos = (facing * view).sum(1)
    lo, hi = math.cos(math.radians(args.to_angle)), math.cos(math.radians(args.from_angle))
    t = np.clip((cos - lo) / (hi - lo), 0, 1)
    straight = t * t * (3 - 2 * t) * seen

    # 4. the picture lined up on the model's own colours (B), then painted on
    rgb = rgba[..., :3].astype(np.float32) / 255
    if args.no_align:
        aligned, aligned_alpha = rgb, alpha
    else:
        aligned, aligned_alpha = line_up_by_regions(own, own_mask & (alpha > 0.5), rgb, alpha,
                                                    HEAD_REGIONS if args.parts == "head" else REGIONS)
    if args.flow:
        aligned, aligned_alpha = bend_onto(own, own_mask & (aligned_alpha > 0.5), aligned, aligned_alpha)
    weight = straight * aligned_alpha[px[:, 1], px[:, 0]]
    colour = aligned[px[:, 1], px[:, 0]]
    painted = np.zeros((size, size, 3), np.float32)
    weights = np.zeros((size, size), np.float32)
    painted[covered], weights[covered] = colour, weight

    # 5. the close-up (A): the same view nearer, matched to the picture by its features, lined up again on the
    #    model around the head, painted over with K times the pixels
    if args.closeup:
        K = 4
        close = np.asarray(Image.open(args.closeup).convert("RGB")).astype(np.float32) / 255
        if args.closeup_box:
            x0, y0, x1, y1 = args.closeup_box
            close = close[y0:y1, x0:x1]
        to_picture = match_features(close, rgb, alpha)
        to_canvas = np.diag([K, K, 1.0])[:2] @ np.vstack([to_picture, [0, 0, 1]])
        canvas = cv2_warp(close, to_canvas, K * res, inverse=False)
        cover = cv2_warp(np.ones(close.shape[:2], np.float32), to_canvas, K * res, inverse=False)
        cover = np.asarray(Image.fromarray((cover * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(9))) / 255
        own_big = np.asarray(Image.fromarray((own * 255).astype(np.uint8)).resize((K * res, K * res))) / 255
        mask_big = np.asarray(Image.fromarray(own_mask).resize((K * res, K * res))) & (cover > 0.5)
        warp, score = line_up_on(own_big, mask_big, canvas, window=True, limit=0.02 * K * res, shift_only=True)
        warp = np.eye(2, 3, dtype=np.float32) if warp is None else warp
        canvas, cover = cv2_warp(canvas, warp, K * res), cv2_warp(cover, warp, K * res)
        if args.flow:  # the close-up bent onto the model as well, in the box it covers
            ys, xs = np.nonzero(mask_big)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            bent, bent_cover = bend_onto(own_big[y0:y1, x0:x1], mask_big[y0:y1, x0:x1], canvas[y0:y1, x0:x1],
                                         cover[y0:y1, x0:x1])
            canvas[y0:y1, x0:x1], cover[y0:y1, x0:x1] = bent, bent_cover
        alpha_big = np.asarray(Image.fromarray((alpha * 255).astype(np.uint8)).resize((K * res, K * res))) / 255
        qx = np.clip((uv_tex * K * res).astype(int), 0, K * res - 1)
        w_close = straight * cover[qx[:, 1], qx[:, 0]] * alpha_big[qx[:, 1], qx[:, 0]]
        sel = w_close > 0
        flat_p, flat_w = painted[covered], weights[covered]
        flat_p[sel] = canvas[qx[sel, 1], qx[sel, 0]]
        flat_w[sel] = np.maximum(flat_w[sel], w_close[sel])
        painted[covered], weights[covered] = flat_p, flat_w
        print(f"close-up: {sel.mean():.1%} of the texels from it (lined up on the head: match {score:.2f})",
              flush=True)

    # 6. (C) pinholes inside painted areas filled from their painted neighbours; then blended into the texture
    weights, painted = fill_holes(weights, painted, covered)
    texture[..., :3] = texture[..., :3] * (1 - weights[..., None]) + painted * weights[..., None]
    texture = bleed(texture, covered)  # the seams: texels just outside each piece take its nearest painted colour
    material.baseColorTexture = Image.fromarray((texture * 255).round().clip(0, 255).astype(np.uint8))
    print(f"painted {(weights[covered] > 0.5).mean():.1%} of the texels from the picture -> {args.out}")
    scene.export(args.out, extension_webp=True)


if __name__ == "__main__":
    main()
