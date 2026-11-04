"""The three pictures of inpainting by picture, as in VoxHammer (the crown on the bee): BEFORE (a render of the
whole model), AFTER (the same render with the new part on it, a plain collage) and the 2D MASK of the change.
TRELLIS.2 needs the whole object in the picture: a picture of only the new part is read as the whole object.

Nothing is placed by hand. The new part's picture is cut out, the side of the model that fits the model's own
picture is measured, the part is put on the top of the object at a size measured from the object, and the 3D area
is the room the part takes: the empty space around it, without the model's own surface (protected, "Teil + leerer
Raum" of week 2), so that the rest stays exact.

    python scripts/inpainting/make_edit.py model.glb crown.png --reference robot.png --out outputs/edits/robot_crown

Writes before.png, after.png, mask.png, area.npy, edit.json, then send with cloud.py inpaint (edit.json has the line).
"""
import _paths  # noqa: F401  (the scripts of the other folders)

import argparse
import json
import os
import subprocess

import numpy as np
import trimesh
from PIL import Image, ImageFilter
from scipy import ndimage
from scipy.spatial import cKDTree

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BLENDER = os.environ.get("BLENDER", os.path.expanduser("~/Applications/blender/blender"))
GREY = (190, 190, 192)
GRID = 64  # TRELLIS.2's voxels per side


def background_remover():
    from huggingface_hub import hf_hub_download
    from trellis2.pipelines import rembg
    config = json.load(open(hf_hub_download("microsoft/TRELLIS.2-4B", "pipeline.json")))["args"]["rembg_model"]
    model = getattr(rembg, config["name"])(**config["args"])
    model.cuda()
    return model


def cut_out(remover, path):
    """The object of a picture, on transparent, cropped to it."""
    picture = remover(Image.open(path).convert("RGB")).convert("RGBA")
    ys, xs = np.nonzero(np.asarray(picture)[..., 3] > 128)
    return picture.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))


def on_grey(view):
    plate = Image.new("RGBA", view.size, GREY + (255,))
    plate.alpha_composite(view)
    return plate.convert("RGB")


def lab(image):
    import cv2
    return cv2.cvtColor(np.asarray(image.convert("RGB"), np.float32) / 255, cv2.COLOR_RGB2LAB)


def front_view(renders, reference):
    """The render that looks most like the model's own picture (colours on the object pixels, 128 x 128)."""
    size = (128, 128)
    wanted = reference.resize(size, Image.LANCZOS)
    wanted_on, wanted_lab = np.asarray(wanted)[..., 3] > 128, lab(on_grey(wanted))
    scores = []
    for render in renders:
        ys, xs = np.nonzero(np.asarray(render)[..., 3] > 128)
        crop = render.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)).resize(size, Image.LANCZOS)
        both = wanted_on & (np.asarray(crop)[..., 3] > 128)
        scores.append(float(np.linalg.norm(lab(on_grey(crop))[both] - wanted_lab[both], axis=1).mean()))
    print("    difference to the model's picture, per side:", [round(s, 1) for s in scores], flush=True)
    return int(np.argmin(scores))


def main():
    parser = argparse.ArgumentParser(description="BEFORE, AFTER and the 2D mask for inpainting by picture")
    parser.add_argument("model")
    parser.add_argument("part", help="picture of the new part")
    parser.add_argument("--reference", help="the model's own picture: picks the side the collage is made on")
    parser.add_argument("--side", type=int, choices=range(4), help="or the side by hand")
    parser.add_argument("--out", required=True)
    parser.add_argument("--width", type=float, default=0.75, help="the part's width, share of the top of the object")
    parser.add_argument("--sink", type=float, default=0.15, help="share of the part's height pushed down into the object")
    parser.add_argument("--head", type=float, default=0.12, help="what counts as the top: share of the object's height")
    parser.add_argument("--no-protect", action="store_true", help="also regenerate the model's surface inside the area")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if not os.path.isfile(os.path.join(args.out, "views.json")):
        subprocess.run([BLENDER, "-b", "--python", os.path.join(PROJECT, "scripts/tools/render_views.py"), "--",
                        os.path.abspath(args.model), os.path.abspath(args.out)], check=True, stdout=subprocess.DEVNULL)
    info = json.load(open(os.path.join(args.out, "views.json")))
    renders = [Image.open(os.path.join(args.out, f"view_{k}.png")).convert("RGBA") for k in range(4)]
    remover = background_remover()
    side = args.side if args.side is not None else front_view(renders, cut_out(remover, args.reference)) if args.reference else 0
    render = renders[side]
    size, span = info["size"], info["span"]
    per_pixel = span / size
    print(f"    the side that fits: view {side}", flush=True)

    # the top of the object, measured in the picture
    alpha = np.asarray(render)[..., 3] > 128
    rows = np.nonzero(alpha.any(1))[0]
    top, height = int(rows[0]), int(rows[-1] - rows[0])
    head = alpha[top:top + int(args.head * height)]
    labels, count = ndimage.label(head)  # only the blob at the very top (arms at the same height don't count)
    first = labels[np.nonzero(head.any(1))[0][0]]
    blob = np.isin(labels, first[first > 0])
    columns = np.nonzero(blob.any(0))[0]
    left, right = int(columns[0]), int(columns[-1])

    part = cut_out(remover, args.part)
    width = int(args.width * (right - left))
    part_height = int(part.height * width / part.width)
    part = part.resize((width, part_height), Image.LANCZOS)
    x = (left + right - width) // 2
    y = top - part_height + int(args.sink * part_height)
    assert y > 0, "the part doesn't fit into the picture"

    before = on_grey(render)
    after = before.copy()
    after.paste(part, (x, y), part)
    mask = Image.new("L", before.size, 0)
    mask.paste(part.getchannel("A").point(lambda v: 255 if v > 30 else 0), (x, y))
    mask = mask.filter(ImageFilter.MaxFilter(9))
    for picture, name in ((before, "before"), (after, "after")):
        picture.save(os.path.join(args.out, f"{name}.png"))
    mask.save(os.path.join(args.out, "mask.png"))

    # the same place in 3D: Blender's axes (Z up) back to the model's glTF axes (Y up)
    center, view = np.array(info["center"]), info["views"][side]
    to_gltf = lambda p: np.stack([p[..., 0], p[..., 2], -p[..., 1]], -1)
    mesh = trimesh.load(args.model, force="mesh")
    vertices = np.asarray(mesh.vertices)
    model_size = np.ptp(vertices, 0).max()
    voxel = model_size / GRID
    up = (size / 2 - np.array([y + part_height, y])) * per_pixel + to_gltf(center)[1]  # bottom, top of the part
    across = (np.array([x, x + width]) - size / 2) * per_pixel
    right_axis = to_gltf(np.array(view["right"]))
    near_top = vertices[vertices[:, 1] > vertices[:, 1].max() - args.head * np.ptp(vertices[:, 1])]
    # the part is round: as deep as it is wide, around the middle of the head
    head_middle = near_top.mean(0)
    middle = to_gltf(center) + right_axis * across.mean()
    depth_axis = to_gltf(np.array(view["toward"]))
    middle = middle + depth_axis * ((head_middle - to_gltf(center)) @ depth_axis)
    half = (across[1] - across[0]) / 2 + 2 * voxel
    steps = np.arange(-half, half + voxel / 2, voxel / 2)
    heights = np.arange(up[0] - voxel, up[1] + 2 * voxel, voxel / 2)
    a, b, c = np.meshgrid(steps, steps, heights, indexing="ij")
    points = middle + a[..., None] * right_axis + b[..., None] * depth_axis
    points = points.reshape(-1, 3)
    points[:, 1] = c.reshape(-1)
    points = points[np.hypot(a.reshape(-1), b.reshape(-1)) <= half]  # a round area for a round part

    kept = len(points)
    if not args.no_protect:
        surface = trimesh.sample.sample_surface(mesh, 2_000_000, seed=0)[0]
        surface = surface[surface[:, 1] > up[0] - 4 * voxel]
        distance = cKDTree(surface).query(points)[0] if len(surface) else np.full(len(points), np.inf)
        points = points[distance > 1.5 * voxel]
    np.save(os.path.join(args.out, "area.npy"), points)

    above = max(0.0, up[1] + 2 * voxel - vertices[:, 1].max())
    room = round(float(min(0.5, 1.05 * above / (above + model_size))), 3)
    edit = {"side": side, "room": room, "area_points": len(points), "before_protection": kept,
            "box_y": [float(up[0]), float(up[1])], "part_pixels": [x, y, width, part_height]}
    json.dump(edit, open(os.path.join(args.out, "edit.json"), "w"), indent=1)
    sheet = Image.new("RGB", (1500, 500), "white")
    for k, picture in enumerate((before, after, mask.convert("RGB"))):
        sheet.paste(picture.resize((500, 500), Image.LANCZOS), (k * 500, 0))
    sheet.save(os.path.join(args.out, "three.jpg"), quality=92)
    print(json.dumps(edit), flush=True)


if __name__ == "__main__":
    main()
