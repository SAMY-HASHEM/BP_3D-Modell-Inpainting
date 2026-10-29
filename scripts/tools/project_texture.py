"""Sharpen a TRELLIS.2 model's texture by projecting the original picture(s) onto it.

TRELLIS.2 only sees the picture at 1024 x 1024 (as 64 x 64 tiles), so fine details come out blurry.
This script paints the colours of the full-size pictures straight onto the parts of the model
that they show. The shape stays as it is. The angle of each picture is found automatically.

Usage:
    source activate.sh
    python scripts/tools/project_texture.py outputs/full_1536_seed42.glb pictures/4/full.jpg
    python scripts/tools/project_texture.py outputs/front_multi4_multidiffusion_1536_seed42.glb \\
        pictures/yeonhee/front.png pictures/yeonhee/front-left.png pictures/yeonhee/side-right.png
    python scripts/tools/project_texture.py outputs/full_1536_seed42.glb pictures/4/full.jpg \\
        --closeups pictures/4/closeup-upper.jpg
    # a model from insert_part.py (body + separately generated parts) works the same way:
    python scripts/tools/project_texture.py outputs/head_1536_seed42_parts.glb pictures/head/head.png

All pictures must show the object in the same pose. Where several pictures see the same spot,
the one that looks at it most directly wins. Close-ups must be crops (or zoomed renders) of the
first picture: they are found inside it by matching image features, and replace its colours.

Steps:
    1. cut the object out of each picture (RMBG-2.0, the same as TRELLIS.2 uses)
    2. find each picture's camera: try many directions, keep the best outline match, fine-tune it
    3. for every texel of the texture that a camera sees, take the colour from that picture
    4. save <model>_projected.glb and <model>_projected_check.png (per picture: picture | before | after)
"""
import _paths  # noqa: F401  (the scripts of the other folders)
import argparse
import math
import os
import time

import generate  # noqa: F401  (sets up the models folder and environment like the other scripts)

import cv2
import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image
from trellis2.pipelines.rembg.BiRefNet import BiRefNet

DEVICE = "cuda"
FIT_SIZE = 512  # longest side of the picture while searching the camera
BAKE_SIZE = 2048  # longest side of the depth image used to decide what the camera sees
SHARPNESS = 16  # how strongly the best-facing picture wins where several see the same spot
AGREEMENT = 0.35  # blurred colour difference (0..1) to TRELLIS.2's texture at which trust drops to 37 %


# ---------- camera ----------

def rotation(yaw, pitch, roll):
    """Rotation matrix; yaw turns around the model's up axis (glTF +Y)."""
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cr, sr = torch.cos(roll), torch.sin(roll)
    one, zero = torch.ones_like(cy), torch.zeros_like(cy)
    r_yaw = torch.stack([cy, zero, sy, zero, one, zero, -sy, zero, cy]).view(3, 3)
    r_pitch = torch.stack([one, zero, zero, zero, cp, -sp, zero, sp, cp]).view(3, 3)
    r_roll = torch.stack([cr, -sr, zero, sr, cr, zero, zero, zero, one]).view(3, 3)
    return r_roll @ r_pitch @ r_yaw


def to_camera(points, cam):
    """Model coordinates -> camera coordinates (camera at the origin, looking down -Z)."""
    p = (points - cam["center"]) @ rotation(cam["yaw"], cam["pitch"], cam["roll"]).T
    return torch.cat([p[:, :2], p[:, 2:] - cam["dist"]], dim=1)


def to_pixels(p_cam, cam, width, height):
    """Camera coordinates -> picture pixels (u to the right, v down) and depth."""
    depth = -p_cam[:, 2]
    focal = cam["focal"] * max(width, height)
    u = cam["cx"] * width + focal * p_cam[:, 0] / depth
    v = cam["cy"] * height - focal * p_cam[:, 1] / depth
    return u, v, depth


def to_clip(points, cam, width, height, near=0.01, far=100.0):
    """Clip coordinates for nvdiffrast, so that image row 0 is the picture's top row."""
    u, v, depth = to_pixels(to_camera(points, cam), cam, width, height)
    z = (far + near) / (far - near) * depth - 2 * far * near / (far - near)
    return torch.stack([(2 * u / width - 1) * depth, (2 * v / height - 1) * depth, z, depth], dim=1)


def make_camera(center, radius, yaw, pitch, roll=0.0, focal=1.5, cx=0.5, cy=0.5, dist=4.0):
    t = lambda x: torch.tensor(float(x), device=DEVICE)
    return {"center": center, "yaw": t(yaw), "pitch": t(pitch), "roll": t(roll),
            "focal": t(focal), "cx": t(cx), "cy": t(cy), "dist": t(dist * radius)}


# ---------- rendering ----------

def render_silhouette(ctx, verts, faces, cam, width, height):
    clip = to_clip(verts, cam, width, height)[None]
    rast, _ = dr.rasterize(ctx, clip, faces, (height, width))
    mask = dr.antialias((rast[..., 3:] > 0).float(), rast, clip, faces)
    return mask[0, ..., 0]


def combine(meshes):
    """All meshes as one vertex and face list (for outlines and depth, where textures don't matter)."""
    verts = torch.cat([m.verts for m in meshes])
    offsets = np.cumsum([0] + [len(m.verts) for m in meshes[:-1]])
    faces = torch.cat([m.faces + int(o) for m, o in zip(meshes, offsets)]).contiguous()
    return verts, faces


def render_meshes(ctx, meshes, cam, width, height, textures=None):
    """Textured render of several meshes together (the nearest surface wins) and where any is seen."""
    image = torch.zeros((height, width, 3), device=DEVICE)
    best = torch.full((height, width), float("inf"), device=DEVICE)
    for i, m in enumerate(meshes):
        texture = m.texture if textures is None else textures[i]
        clip = to_clip(m.verts, cam, width, height)[None]
        rast, _ = dr.rasterize(ctx, clip, m.faces, (height, width))
        uv, _ = dr.interpolate(m.uvs_tex[None], rast, m.faces)
        color = dr.texture(texture[None], uv, filter_mode="linear")[0]
        z = torch.where(rast[0, ..., 3] > 0, rast[0, ..., 2], torch.full_like(best, float("inf")))
        nearer = z < best
        image = torch.where(nearer[..., None], color, image)
        best = torch.where(nearer, z, best)
    return image, best < float("inf")


def bbox(mask):
    ys, xs = torch.nonzero(mask > 0.5, as_tuple=True)
    if len(xs) == 0:
        return None
    return xs.min().item(), xs.max().item() + 1, ys.min().item(), ys.max().item() + 1


# ---------- step 1: cut out the object ----------

def object_masks(pictures):
    """Alpha mask of the object in each picture, 0..1, at the picture's size (RMBG-2.0)."""
    remover = BiRefNet(model_name="briaai/RMBG-2.0")
    remover.cuda()
    masks = [np.asarray(remover(p.convert("RGB")))[..., 3].astype(np.float32) / 255 for p in pictures]
    del remover
    torch.cuda.empty_cache()
    return masks


# ---------- step 2: find the camera ----------

def fit_camera(ctx, meshes, picture, mask, ignore=None):
    """Camera that shows the model (a list of meshes) like the picture. `ignore` (optional,
    picture-sized bool array) marks areas whose outline must not steer the camera, e.g. a part
    that will be replaced. Each mesh needs .verts, .faces, .uvs_tex and .texture."""
    verts, faces = combine(meshes)
    scale = FIT_SIZE / max(picture.size)
    width, height = (round(s * scale / 8) * 8 for s in picture.size)
    target = torch.tensor(cv2.resize(mask, (width, height), interpolation=cv2.INTER_AREA), device=DEVICE)
    valid = torch.ones_like(target, dtype=torch.bool)
    if ignore is not None:
        small = cv2.resize(ignore.astype(np.float32), (width, height), interpolation=cv2.INTER_AREA)
        valid = torch.tensor(small < 0.5, device=DEVICE)
    overlap = lambda sil: (((sil > 0.5) & (target > 0.5) & valid).sum() / (((sil > 0.5) | (target > 0.5)) & valid).sum())
    colors = torch.tensor(np.asarray(picture.convert("RGB").resize((width, height), Image.LANCZOS)),
                          device=DEVICE, dtype=torch.float32) / 255
    box_t = bbox(target)
    center = (verts.min(0).values + verts.max(0).values) / 2
    radius = ((verts.max(0).values - verts.min(0).values).norm() / 2).item()

    # a) try many directions; scale and shift each so its outline box fits the picture's
    candidates = []
    for yaw in range(0, 360, 15):
        for pitch in (-30, -15, 0, 15, 30):
            cam = make_camera(center, radius, math.radians(yaw), math.radians(pitch))
            box_s = bbox(render_silhouette(ctx, verts, faces, cam, width, height))
            if box_s is None:
                continue
            s = math.sqrt((box_t[1] - box_t[0]) * (box_t[3] - box_t[2]) / ((box_s[1] - box_s[0]) * (box_s[3] - box_s[2])))
            px, py = cam["cx"].item() * width, cam["cy"].item() * height
            sx, sy = (box_s[0] + box_s[1]) / 2, (box_s[2] + box_s[3]) / 2
            tx, ty = (box_t[0] + box_t[1]) / 2, (box_t[2] + box_t[3]) / 2
            cam["focal"] = cam["focal"] * s
            cam["cx"] = torch.tensor((px + tx - (px + s * (sx - px))) / width, device=DEVICE)
            cam["cy"] = torch.tensor((py + ty - (py + s * (sy - py))) / height, device=DEVICE)
            sil = render_silhouette(ctx, verts, faces, cam, width, height)
            candidates.append((overlap(sil).item(), cam))

    # b) outlines alone can mislead (front and back are mirror images; a bust cut off by the picture
    #    edge fits a camera from below about as well as one from the front), so the colours decide
    #    among all directions whose outline is nearly as good as the best one
    candidates.sort(key=lambda c: -c[0])
    best_iou = candidates[0][0]
    fg = target > 0.5
    scored = []
    for iou, cam in candidates[:40]:
        if iou < best_iou - 0.15:
            break
        rendered, covered = render_meshes(ctx, meshes, cam, width, height)
        both = fg & covered & valid
        error = (rendered[both] - colors[both]).abs().mean().item() if both.any() else 1.0
        scored.append((error + 0.5 * (best_iou - iou), iou, cam))
    error, iou, cam = min(scored, key=lambda c: c[0])
    print(f"    best start: yaw {math.degrees(cam['yaw']):.0f}°, pitch {math.degrees(cam['pitch']):.0f}°, "
          f"outline overlap {iou:.0%}")

    # c) fine-tune all camera values so the outlines match as closely as possible
    params = {k: cam[k].clone().requires_grad_(True) for k in ("yaw", "pitch", "roll", "focal", "cx", "cy", "dist")}
    optimizer = torch.optim.Adam([
        {"params": [params["yaw"], params["pitch"], params["roll"]], "lr": 0.005},
        {"params": [params["focal"], params["cx"], params["cy"]], "lr": 0.002},
        {"params": [params["dist"]], "lr": 0.01 * radius},
    ])
    blur = lambda m: F.avg_pool2d(m[None, None], 9, stride=1, padding=4)[0, 0]
    for _ in range(300):
        sil = render_silhouette(ctx, verts, faces, {"center": center, **params}, width, height)
        sil, goal = sil * valid, target * valid
        loss = F.mse_loss(sil, goal) + F.mse_loss(blur(sil), blur(goal))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    cam = {"center": center, **{k: v.detach() for k, v in params.items()}}
    iou = overlap(render_silhouette(ctx, verts, faces, cam, width, height)).item()
    print(f"    after fine-tuning: outline overlap {iou:.0%}")
    return cam, iou


# ---------- step 3: take the colours ----------

def sample(image, u, v, mode="bilinear"):
    """Sample an image (H, W, C) at pixel positions u, v (any shape)."""
    height, width = image.shape[:2]
    grid = torch.stack([2 * u / width - 1, 2 * v / height - 1], dim=-1).view(1, -1, 1, 2)
    out = F.grid_sample(image.permute(2, 0, 1)[None], grid, mode=mode, align_corners=False)
    return out[0, :, :, 0].T.reshape(*u.shape, image.shape[2])


def texel_geometry(ctx, verts, normals, faces, uvs, size):
    """3D position and normal of every texel, by drawing the mesh into its own texture space."""
    # trimesh's v points up, but texture row 0 is the top row, and nvdiffrast puts y = -1 in row 0
    clip = torch.cat([uvs[:, :1] * 2 - 1, 1 - uvs[:, 1:] * 2,
                      torch.zeros_like(uvs[:, :1]), torch.ones_like(uvs[:, :1])], dim=1)
    rast, _ = dr.rasterize(ctx, clip[None], faces, (size, size))
    position, _ = dr.interpolate(verts[None], rast, faces)
    normal, _ = dr.interpolate(normals[None], rast, faces)
    return position[0], F.normalize(normal[0], dim=-1), rast[0, ..., 3] > 0


def picture_view(ctx, meshes, cam, picture, mask, check_colors=False):
    """Everything about one picture and its camera that all meshes share: the picture itself, the
    depth of the nearest surface per pixel (what the camera sees), where the object is, and with
    --check-colors how far the picture's colours are from the model's own."""
    image = torch.tensor(np.asarray(picture.convert("RGB")), device=DEVICE, dtype=torch.float32) / 255
    height, width = image.shape[:2]
    scale = min(1.0, BAKE_SIZE / max(width, height))
    dw, dh = round(width * scale / 8) * 8, round(height * scale / 8) * 8
    verts, faces = combine(meshes)
    clip = to_clip(verts, cam, dw, dh)[None]
    rast, _ = dr.rasterize(ctx, clip, faces, (dh, dw))
    depth, _ = dr.interpolate(clip[..., 3:].contiguous(), rast, faces)
    depth = torch.where(rast[..., 3:] > 0, depth, torch.full_like(depth, 1e9))[0]

    # fade out towards the outline, so no background colour lands on the model's edges
    inside = cv2.erode((mask > 0.5).astype(np.uint8), np.ones((5, 5), np.uint8), iterations=max(1, width // 1024))
    inside = cv2.GaussianBlur(inside.astype(np.float32), (0, 0), max(1.0, width / 1024))

    # Where the 3D shape differs from the picture (e.g. a hand held a bit differently), the picture
    # paints the hand onto whatever is behind it. TRELLIS.2's own texture is blurry but has roughly
    # the right colours, so --check-colors compares both blurred (small details like eyes vanish,
    # a hand in front of a black top does not) and trusts the picture less where they disagree.
    # Off by default: it also rejects good detail where TRELLIS.2's shape is slightly off (faces).
    difference = None
    if check_colors:
        rendered, _ = render_meshes(ctx, meshes, cam, dw, dh)
        sigma = max(dw, dh) / 150
        small = cv2.resize(image.cpu().numpy(), (dw, dh), interpolation=cv2.INTER_AREA)
        difference = np.abs(cv2.GaussianBlur(small, (0, 0), sigma) - cv2.GaussianBlur(rendered.cpu().numpy(), (0, 0), sigma))
        difference = torch.tensor(difference.mean(-1), device=DEVICE)[..., None]
    return {"image": image, "depth": depth, "depth_size": (dw, dh),
            "inside": torch.tensor(inside, device=DEVICE)[..., None], "difference": difference}


def project(view, texels, cam, radius):
    """Colour from the picture for every texel of one mesh the camera sees, and how much to trust it."""
    position, normal, covered = texels
    image = view["image"]
    height, width = image.shape[:2]
    dw, dh = view["depth_size"]

    p_cam = to_camera(position.view(-1, 3), cam)
    u, v, texel_depth = to_pixels(p_cam, cam, width, height)
    surface_depth = sample(view["depth"], u * dw / width, v * dh / height, mode="nearest")[:, 0]
    visible = texel_depth <= surface_depth + 0.01 * radius

    # trust surfaces that face the camera; fade out towards the outline and at grazing angles
    normal_cam = normal.view(-1, 3) @ rotation(cam["yaw"], cam["pitch"], cam["roll"]).T
    facing = (normal_cam * F.normalize(-p_cam, dim=1)).sum(1).abs()
    trust = ((facing - 0.15) / 0.35).clamp(0, 1)
    trust = trust * trust * (3 - 2 * trust)
    trust = trust * sample(view["inside"], u, v)[:, 0]
    if view["difference"] is not None:
        trust = trust * torch.exp(-(sample(view["difference"], u * dw / width, v * dh / height)[:, 0] / AGREEMENT) ** 2)

    weight = (covered.view(-1) & visible).float() * trust
    color = sample(image, u, v)
    size = covered.shape[0]
    return color.view(size, size, 3), weight.view(size, size), (u, v)


def find_in_picture(closeup, picture):
    """Homography that maps close-up pixels to picture pixels (SIFT features + RANSAC)."""
    sift = cv2.SIFT_create(nfeatures=8000)
    gray = lambda im: cv2.cvtColor(np.asarray(im.convert("RGB")), cv2.COLOR_RGB2GRAY)
    k1, d1 = sift.detectAndCompute(gray(closeup), None)
    k2, d2 = sift.detectAndCompute(gray(picture), None)
    matches = [m for m, n in cv2.BFMatcher().knnMatch(d1, d2, k=2) if m.distance < 0.75 * n.distance]
    if len(matches) < 20:
        return None, len(matches)
    src = np.float32([k1[m.queryIdx].pt for m in matches])
    dst = np.float32([k2[m.trainIdx].pt for m in matches])
    homography, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    return homography, int(inliers.sum()) if inliers is not None else 0


def project_closeup(closeup, homography, picture_uv, picture_weight, size):
    """Colours from a close-up, for the texels the main picture already trusts."""
    image = torch.tensor(np.asarray(closeup.convert("RGB")), device=DEVICE, dtype=torch.float32) / 255
    height, width = image.shape[:2]
    to_closeup = torch.tensor(np.linalg.inv(homography), device=DEVICE, dtype=torch.float32)
    u, v = picture_uv
    p = to_closeup @ torch.stack([u, v, torch.ones_like(u)])
    cu, cv = p[0] / p[2], p[1] / p[2]
    # fade out over the last 3 % towards the close-up's border
    edge = torch.minimum(torch.minimum(cu, width - cu) / width, torch.minimum(cv, height - cv) / height)
    border = (edge / 0.03).clamp(0, 1)
    color = sample(image, cu, cv)
    return color.view(size, size, 3), picture_weight * border.view(size, size)


def spread_into_gaps(values, weight, covered, steps=8):
    """Extend the new colours a few texels past the edges of the UV islands, against seams."""
    filled = covered.float()
    for _ in range(steps):
        num = F.avg_pool2d((values * filled[..., None]).permute(2, 0, 1)[None], 3, 1, 1)[0].permute(1, 2, 0)
        den = F.avg_pool2d(filled[None, None], 3, 1, 1)[0, 0]
        grow = (filled == 0) & (den > 0)
        values = torch.where(grow[..., None], num / den.clamp(min=1e-6)[..., None], values)
        weight_num = F.avg_pool2d((weight * filled)[None, None], 3, 1, 1)[0, 0]
        weight = torch.where(grow, weight_num / den.clamp(min=1e-6), weight)
        filled = torch.where(grow, torch.ones_like(filled), filled)
    return values, weight


# ---------- main ----------

class Mesh:
    """One textured mesh of a .glb, as GPU tensors (positions with its scene transform applied)."""

    def __init__(self, geometry, transform):
        self.geometry = geometry
        self.material = geometry.visual.material
        points = trimesh.transform_points(geometry.vertices, transform)
        normals = trimesh.transform_points(geometry.vertex_normals, transform, translate=False)
        self.verts = torch.tensor(points, device=DEVICE, dtype=torch.float32)
        self.faces = torch.tensor(geometry.faces, device=DEVICE, dtype=torch.int32)
        self.normals = torch.tensor(normals, device=DEVICE, dtype=torch.float32)
        self.uvs = torch.tensor(geometry.visual.uv, device=DEVICE, dtype=torch.float32)
        self.uvs_tex = torch.stack([self.uvs[:, 0], 1 - self.uvs[:, 1]], dim=1)  # trimesh's v points up
        old = self.material.baseColorTexture.convert("RGB")
        self.texture = torch.tensor(np.asarray(old), device=DEVICE, dtype=torch.float32) / 255
        self.size = old.size[0]


def load_meshes(path):
    """The scene and all its textured meshes (a TRELLIS.2 model has one; insert_part.py adds more)."""
    scene = trimesh.load(path)
    meshes = []
    for node in scene.graph.nodes_geometry:
        transform, name = scene.graph[node]
        meshes.append(Mesh(scene.geometry[name], transform))
    return scene, meshes


def main():
    parser = argparse.ArgumentParser(description="Project the original picture(s) onto a TRELLIS.2 model")
    parser.add_argument("model", help="the .glb from generate.py, multi_image.py or insert_part.py")
    parser.add_argument("pictures", nargs="+", help="full-size pictures of the object, all in the same pose")
    parser.add_argument("--closeups", nargs="*", default=[], help="close-ups of the first picture, for more detail")
    parser.add_argument("--check-colors", action="store_true",
                        help="trust the pictures less where their colours differ strongly from the model's "
                             "(against e.g. a hand painted onto the chest; can cost detail in faces)")
    args = parser.parse_args()
    start = time.time()

    scene, meshes = load_meshes(args.model)
    all_verts, all_faces = combine(meshes)
    radius = ((all_verts.max(0).values - all_verts.min(0).values).norm() / 2).item()
    ctx = dr.RasterizeCudaContext()
    print(f"Model: {len(meshes)} mesh(es), {len(all_faces):,} faces")

    pictures = [Image.open(path) for path in args.pictures]
    names = [os.path.basename(path) for path in args.pictures]
    print("1. Cutting the object out of the pictures...")
    masks = object_masks(pictures)

    texels = [texel_geometry(ctx, m.verts, m.normals, m.faces, m.uvs, m.size) for m in meshes]
    colors = [[] for _ in meshes]
    weights = [[] for _ in meshes]
    cameras = []
    for i, (name, picture, mask) in enumerate(zip(names, pictures, masks)):
        print(f"2. Finding the camera of {name}...")
        cam, iou = fit_camera(ctx, meshes, picture, mask)
        if iou < 0.8:
            print("    Warning: the outlines match poorly, so colours may land in the wrong places.")
        cameras.append(cam)
        print(f"3. Taking the colours from {name}...")
        view = picture_view(ctx, meshes, cam, picture, mask, args.check_colors)
        for k, m in enumerate(meshes):
            color, weight, picture_uv = project(view, texels[k], cam, radius)
            if i == 0:
                for path in args.closeups:
                    closeup = Image.open(path)
                    homography, inliers = find_in_picture(closeup, picture)
                    if homography is None or inliers < 30:
                        print(f"    {os.path.basename(path)}: not found in the picture ({inliers} matches), skipped")
                        continue
                    c_color, c_weight = project_closeup(closeup, homography, picture_uv, weight, m.size)
                    color = torch.lerp(color, c_color, c_weight[..., None])
                    print(f"    {os.path.basename(path)}: found ({inliers} matching points)")
            colors[k].append(color)
            weights[k].append(weight)
        del view

    # Where several pictures see the same texel, the one facing it most directly wins. Averaging
    # them would show double edges ("ghosting"), because no two pictures line up perfectly.
    # (weight / best weight) ** SHARPNESS keeps the best picture at 1 and lets the others drop off
    # quickly, so they only blend in where two pictures are almost equally good (soft seams).
    new_textures = []
    for k, m in enumerate(meshes):
        w = torch.stack(weights[k])
        weight_max = w.max(0).values
        strength = (w / weight_max.clamp(min=1e-6)) ** SHARPNESS
        color = (torch.stack(colors[k]) * strength[..., None]).sum(0) / strength.sum(0).clamp(min=1e-6)[..., None]
        color, weight = spread_into_gaps(color, weight_max, texels[k][2])
        new_textures.append(torch.lerp(m.texture, color, weight[..., None]))
        print(f"    mesh {k + 1}: {(weight > 0.5).float().mean().item():.0%} of its texture now comes from the pictures")
    del colors, weights

    print("4. Saving...")
    base = os.path.splitext(args.model)[0] + "_projected"
    for m, texture in zip(meshes, new_textures):
        m.material.baseColorTexture = Image.fromarray((texture.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
    scene.export(base + ".glb")

    # check picture: one row per picture: picture | before | after, from that picture's camera
    rows = []
    for picture, cam in zip(pictures, cameras):
        view_h = 768
        view_w = round(picture.size[0] * view_h / picture.size[1] / 8) * 8
        before, seen = render_meshes(ctx, meshes, cam, view_w, view_h)
        after, _ = render_meshes(ctx, meshes, cam, view_w, view_h, new_textures)
        gray = torch.full_like(before, 0.25)
        panels = [np.asarray(picture.convert("RGB").resize((view_w, view_h), Image.LANCZOS))]
        for img in (before, after):
            panels.append((torch.where(seen[..., None], img, gray).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
        rows.append(np.concatenate(panels, axis=1))
    width = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
    Image.fromarray(np.concatenate(rows, axis=0)).save(base + "_check.png")
    print(f"Done in {time.time() - start:.0f} s: {base}.glb")
    print(f"Check picture (per picture: original | before | after): {base}_check.png")


if __name__ == "__main__":
    main()
