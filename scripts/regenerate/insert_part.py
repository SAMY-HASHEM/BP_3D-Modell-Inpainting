"""Replace a region of a TRELLIS.2 model with a separately generated, more detailed part.

The whole model shares one 3D grid, so small parts (boots, gloves, ...) get few voxels. Generating a
part alone from a cut-out of the same picture gives it the whole grid. This script puts such a part
model into the full model: it finds where the part sits, removes the full model's own (blurry)
version of that region and inserts the part in its place.

Usage:
    source activate.sh
    python scripts/regenerate/insert_part.py outputs/parts-front_1536_seed42.glb pictures/yeonhee/parts-front.png \\
        --part outputs/boot-right_1536_seed42.glb pictures/yeonhee/parts/boot-right.png \\
        --part outputs/boot-left_1536_seed42.glb pictures/yeonhee/parts/boot-left.png
    # all cut-outs of a folder (from cut_parts.py), each with its newest model from outputs/:
    python scripts/regenerate/insert_part.py outputs/parts-front_1536_seed42.glb pictures/yeonhee/parts-front.png \\
        --parts-dir pictures/yeonhee/parts-auto

Each part picture must be a cut-out (RGBA, transparent background) taken pixel for pixel from the
full picture; its position in the full picture is found automatically.

Steps:
    1. find the camera of the full model in the full picture (as in project_texture.py)
    2. for each part: find its camera in its cut-out, and from both cameras where it sits in the model
    3. move the part in depth so its front surface lies on the full model's surface
    4. delete the full model's faces in that region (inside the part's outline and depth range)
    5. save <model>_parts.glb (one mesh per part) and <model>_parts_check.png (picture | before | after)
"""
import _paths  # noqa: F401  (the scripts of the other folders)
import argparse
import glob
import os
import time

import cv2
import numpy as np
import nvdiffrast.torch as dr
import torch
import trimesh
from PIL import Image

from generate import OUTPUT_DIR
from project_texture import DEVICE, fit_camera, object_masks, rotation, to_camera, to_clip

REMOVE_MARGIN = 0.02  # extra depth (fraction of the model size) removed in front of and behind a part
VIEW_SIZE = 1024  # longest side of the pictures used for depth and for the check picture


class Model:
    """A TRELLIS.2 .glb as GPU tensors (one mesh with one base colour texture)."""

    def __init__(self, path):
        scene = trimesh.load(path)
        assert len(scene.geometry) == 1, f"{path}: expected a single mesh, as TRELLIS.2 exports it"
        self.mesh = next(iter(scene.geometry.values()))
        self.verts = torch.tensor(self.mesh.vertices, device=DEVICE, dtype=torch.float32)
        self.faces = torch.tensor(self.mesh.faces, device=DEVICE, dtype=torch.int32)
        uvs = torch.tensor(self.mesh.visual.uv, device=DEVICE, dtype=torch.float32)
        self.uvs_tex = torch.stack([uvs[:, 0], 1 - uvs[:, 1]], dim=1)  # trimesh's v points up
        texture = self.mesh.visual.material.baseColorTexture.convert("RGB")
        self.texture = torch.tensor(np.asarray(texture), device=DEVICE, dtype=torch.float32) / 255
        self.radius = ((self.verts.max(0).values - self.verts.min(0).values).norm() / 2).item()


def camera_point(points, cam):
    """Model coordinates -> camera coordinates before the camera's distance is subtracted."""
    return (points - cam["center"]) @ rotation(cam["yaw"], cam["pitch"], cam["roll"]).T


def locate(cutout, picture):
    """Top-left pixel of a cut-out inside the picture it was taken from."""
    full = np.asarray(picture.convert("RGB")).astype(np.float32)
    part = np.asarray(cutout.convert("RGB")).astype(np.float32)
    scores = cv2.matchTemplate(full, part, cv2.TM_SQDIFF)
    _, _, (x, y), _ = cv2.minMaxLoc(scores)
    return x, y


def crop_camera_to_picture(cam, crop_size, offset, picture_size):
    """A camera fitted inside a cut-out, expressed for the whole picture (same pixels, other frame)."""
    (cw, ch), (x0, y0), (pw, ph) = crop_size, offset, picture_size
    focal_px = cam["focal"] * max(cw, ch)
    return {**cam, "focal": focal_px / max(pw, ph),
            "cx": (x0 + cam["cx"] * cw) / pw, "cy": (y0 + cam["cy"] * ch) / ph}


def similarity(part_cam, full_cam, picture_size):
    """Scale, rotation and shift that move part coordinates into full-model coordinates.

    Seen from far away (the cameras are about 4 model sizes away), a camera maps a point to
    pixel = centre + k * (rotated point), with k = focal / distance. Matching the part's camera to
    the full model's camera in the same picture gives everything except the depth (set later).
    """
    width, height = picture_size
    k_part = part_cam["focal"] * max(width, height) / part_cam["dist"]
    k_full = full_cam["focal"] * max(width, height) / full_cam["dist"]
    scale = k_part / k_full
    shift = torch.stack([(part_cam["cx"] - full_cam["cx"]) * width / k_full,
                         -(part_cam["cy"] - full_cam["cy"]) * height / k_full,
                         torch.zeros((), device=DEVICE)])
    r_part = rotation(part_cam["yaw"], part_cam["pitch"], part_cam["roll"])
    r_full = rotation(full_cam["yaw"], full_cam["pitch"], full_cam["roll"])

    def apply(points, depth_shift=0.0):
        q = scale * (points - part_cam["center"]) @ r_part.T + shift
        q = q + torch.tensor([0.0, 0.0, depth_shift], device=DEVICE)
        return q @ r_full + full_cam["center"]

    return apply


def render_depth(ctx, verts, faces, cam, width, height):
    """Camera-space depth (bigger = closer) of the nearest surface for every pixel, NaN where empty."""
    clip = to_clip(verts, cam, width, height)[None]
    rast, _ = dr.rasterize(ctx, clip, faces, (height, width))
    q_z = camera_point(verts, cam)[:, 2:].contiguous()
    depth, _ = dr.interpolate(q_z[None], rast, faces)
    depth = depth[0, ..., 0]
    return torch.where(rast[0, ..., 3] > 0, depth, torch.full_like(depth, float("nan")))


def render_scene(ctx, models, cam, width, height):
    """Textured render of several meshes together (the nearest surface wins), background grey."""
    image = torch.full((height, width, 3), 0.25, device=DEVICE)
    best = torch.full((height, width), float("inf"), device=DEVICE)
    for verts, faces, uvs_tex, texture in models:
        clip = to_clip(verts, cam, width, height)[None]
        rast, _ = dr.rasterize(ctx, clip, faces, (height, width))
        uv, _ = dr.interpolate(uvs_tex[None], rast, faces)
        color = dr.texture(texture[None], uv, filter_mode="linear")[0]
        z = torch.where(rast[0, ..., 3] > 0, rast[0, ..., 2], torch.full_like(best, float("inf")))
        nearer = z < best
        image = torch.where(nearer[..., None], color, image)
        best = torch.where(nearer, z, best)
    return image


def main():
    parser = argparse.ArgumentParser(description="Insert detailed part models into a full TRELLIS.2 model")
    parser.add_argument("model", help="the full model (.glb)")
    parser.add_argument("picture", help="the picture the full model was generated from")
    parser.add_argument("--part", nargs=2, action="append", default=[], metavar=("MODEL", "CUTOUT"),
                        help="a part model and the RGBA cut-out it was generated from (repeat for more parts)")
    parser.add_argument("--parts-dir", help="folder of RGBA cut-outs; each is paired with its newest model in outputs/")
    args = parser.parse_args()
    if args.parts_dir:
        for cutout in sorted(glob.glob(os.path.join(args.parts_dir, "*.png"))):
            stem = os.path.splitext(os.path.basename(cutout))[0]
            models = [m for m in glob.glob(os.path.join(OUTPUT_DIR, f"{stem}_*.glb"))
                      if not m.endswith(("_parts.glb", "_projected.glb"))]
            if not models:
                print(f"No model for {cutout} in {OUTPUT_DIR} (generate it with generate.py), skipped")
                continue
            args.part.append([max(models, key=os.path.getmtime), cutout])
    if not args.part:
        parser.error("give parts with --part MODEL CUTOUT or --parts-dir FOLDER")
    start = time.time()
    ctx = dr.RasterizeCudaContext()

    full = Model(args.model)
    picture = Image.open(args.picture)
    width, height = picture.size
    print(f"Full model: {len(full.faces):,} faces")

    print("1. Finding the camera of the full model...")
    full_mask = object_masks([picture])[0]
    # The parts are replaced anyway, and the full model's own version of them is often what went
    # wrong (e.g. hair that fell apart), so their outlines must not steer the full model's camera.
    replaced = np.zeros((height, width), bool)
    for _, cutout_path in args.part:
        cutout = Image.open(cutout_path).convert("RGBA")
        x0, y0 = locate(cutout, picture)
        replaced[y0:y0 + cutout.size[1], x0:x0 + cutout.size[0]] |= np.asarray(cutout)[..., 3] > 127
    replaced = cv2.dilate(replaced.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    full_cam, _ = fit_camera(ctx, [full], picture, full_mask, ignore=replaced)

    scale = VIEW_SIZE / max(width, height)
    vw, vh = round(width * scale / 8) * 8, round(height * scale / 8) * 8
    full_depth = render_depth(ctx, full.verts, full.faces, full_cam, vw, vh)
    full_q = camera_point(full.verts, full_cam)
    face_q = full_q[full.faces.long()].mean(1)
    face_clip = to_clip(full.verts[full.faces.long()].mean(1), full_cam, vw, vh)
    face_u = ((face_clip[:, 0] / face_clip[:, 3] + 1) / 2 * vw).long().clamp(0, vw - 1)
    face_v = ((face_clip[:, 1] / face_clip[:, 3] + 1) / 2 * vh).long().clamp(0, vh - 1)
    remove = torch.zeros(len(full.faces), dtype=torch.bool, device=DEVICE)

    inserted = []
    for part_path, cutout_path in args.part:
        name = os.path.splitext(os.path.basename(cutout_path))[0]
        part = Model(part_path)
        cutout = Image.open(cutout_path).convert("RGBA")
        x0, y0 = locate(cutout, picture)
        print(f"2. {name}: found at x={x0}, y={y0} in the picture; finding its camera...")
        alpha = np.asarray(cutout)[..., 3].astype(np.float32) / 255
        crop_cam, _ = fit_camera(ctx, [part], cutout, alpha)
        part_cam = crop_camera_to_picture(crop_cam, cutout.size, (x0, y0), picture.size)
        move = similarity(part_cam, full_cam, picture.size)

        # 3. depth: put the part's front surface onto the full model's front surface
        part_mask = np.zeros((height, width), np.float32)
        part_mask[y0:y0 + cutout.size[1], x0:x0 + cutout.size[0]] = alpha
        part_mask = torch.tensor(cv2.resize(part_mask, (vw, vh), interpolation=cv2.INTER_AREA) > 0.5, device=DEVICE)
        placed = move(part.verts)
        part_depth = render_depth(ctx, placed, part.faces, full_cam, vw, vh)
        both = part_mask & ~full_depth.isnan() & ~part_depth.isnan()
        depth_shift = (full_depth[both] - part_depth[both]).median().item()
        placed = move(part.verts, depth_shift)
        print(f"3. {name}: scale {placed.std() / part.verts.std():.3f}, moved {depth_shift:+.3f} in depth "
              f"({both.sum().item():,} pixels compared)")

        # 4. remove the full model's faces inside the part's outline and depth range
        part_q = camera_point(placed, full_cam)
        margin = REMOVE_MARGIN * full.radius
        grown = cv2.dilate(part_mask.cpu().numpy().astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        inside = torch.tensor(grown, device=DEVICE)[face_v, face_u]
        in_depth = (face_q[:, 2] > part_q[:, 2].min() - margin) & (face_q[:, 2] < part_q[:, 2].max() + margin)
        region = inside & in_depth
        remove |= region
        print(f"4. {name}: removing {region.sum().item():,} faces of the full model")
        inserted.append((name, part, placed))

    print("5. Saving...")
    body = full.mesh.copy()
    body.update_faces((~remove).cpu().numpy())
    body.remove_unreferenced_vertices()
    scene = trimesh.Scene()
    scene.add_geometry(body, geom_name="body")
    for name, part, placed in inserted:
        mesh = part.mesh.copy()
        mesh.vertices = placed.cpu().numpy()
        scene.add_geometry(mesh, geom_name=name)
    base = os.path.splitext(args.model)[0] + "_parts"
    scene.export(base + ".glb")

    before = render_scene(ctx, [(full.verts, full.faces, full.uvs_tex, full.texture)], full_cam, vw, vh)
    keep = ~remove
    after_models = [(full.verts, full.faces[keep].contiguous(), full.uvs_tex, full.texture)]
    after_models += [(placed, part.faces, part.uvs_tex, part.texture) for _, part, placed in inserted]
    after = render_scene(ctx, after_models, full_cam, vw, vh)
    panels = [np.asarray(picture.convert("RGB").resize((vw, vh), Image.LANCZOS))]
    panels += [(img.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8) for img in (before, after)]
    Image.fromarray(np.concatenate(panels, axis=1)).save(base + "_check.png")
    print(f"Done in {time.time() - start:.0f} s: {base}.glb")
    print(f"Check picture (picture | before | after): {base}_check.png")


if __name__ == "__main__":
    main()
