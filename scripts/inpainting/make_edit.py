"""The three pictures of inpainting by picture, as in VoxHammer (the crown on the bee): BEFORE (a render of the
whole model), AFTER (the same render with the new part on it, a plain collage) and the 2D MASK of the change.
TRELLIS.2 needs the whole object in the picture: a picture of only the new part is read as the whole object.

Nothing is placed by hand. The new part's picture is cut out, the side of the model that fits the model's own
picture is measured (its outline first, then its colours), the part is put on the top of the object at a size
measured from the object, and the 3D area is the room the part takes. Every small cube of that area gets one job
(area_map.jpg shows them), decided on TRELLIS.2's own 64^3 voxels: KEEP a voxel the model's surface runs through and
the model's inside (protected, "Teil + leerer Raum" of week 2), DELETE the old surface under the 2D mask but not
under its rim (--replace), EDIT the empty room the new part grows into. Only DELETE + EDIT are regenerated, so the
rest stays exact.

    python scripts/inpainting/make_edit.py model.glb crown.png --reference robot.png --out outputs/edits/robot_crown

By hand (the web page's "Edit by hand", scripts/pipeline/flow_page.py): --place X Y WIDTH puts the part there instead
of the measured place (pixels of the main side's picture), and --hand FOLDER has the painted corrections of the 2D
mask, per side: add_<view>.png (white = this changes too) and erase_<view>.png (white = this stays).
--after PICTURE is a ready AFTER picture of the main side (picture_from_prompt.py: a prompt painted into the painted area):
what differs from BEFORE inside add_<side>.png is the new part, no part picture is used.

No picture of the part at all: --prompt "a wide pointed wizard hat on the head" instead of the part. The local picture
AI (Qwen-Image-Edit, as picture_from_prompt.py) paints the words onto BEFORE of the main side; what differs from BEFORE is
the new part and its 2D mask. Such a part can be anywhere on the model, so the 3D area is put where the model's
surface is under the part (tried on the wanderer, Nov 11: a hat, a shield in the hand, wings on the back).

    python scripts/inpainting/make_edit.py model.glb --prompt "a round wooden shield in the free hand" --out outputs/edits/shield

A picture of the part AND --prompt: the picture AI paints that part where the words say, at the object's scale and
angle, instead of the measured place on the top (the edit node's `at`; tried Nov 12: the crown on the wanderer).

    python scripts/inpainting/make_edit.py model.glb shield.png --prompt "the shield in the left hand" --out outputs/edits/shield

Writes before.png, after.png, mask.png, area.npy, area_map.jpg, edit.json, then send with cloud.py inpaint (edit.json has the line).
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

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BLENDER = os.environ.get("BLENDER", os.path.expanduser("~/Applications/blender/blender"))
GREY = (190, 190, 192)
GRID = 64  # TRELLIS.2's voxels per side
# the sentence around --prompt: nothing is marked in the picture, the picture AI finds the place from the words
WORDS = ("In image 1, add {prompt} to the object{reference}. Keep everything else in image 1 exactly as it is: "
         "same object, same pose, same camera angle, same lighting, same plain grey background. "
         "The new part fits the scale, material and style of the object.")


def background_remover():
    from huggingface_hub import hf_hub_download
    from trellis2.pipelines import rembg
    config = json.load(open(hf_hub_download("microsoft/TRELLIS.2-4B", "pipeline.json")))["args"]["rembg_model"]
    model = getattr(rembg, config["name"])(**config["args"])
    model.cuda()
    return model


def cut_out(remover, path):
    """The object of a picture, on transparent, cropped to it."""
    picture = np.array(remover(Image.open(path).convert("RGB")).convert("RGBA"))
    labels, count = ndimage.label(picture[..., 3] > 128)
    biggest = 1 + int(np.argmax(ndimage.sum(np.ones(labels.shape), labels, range(1, count + 1))))
    picture[..., 3][ndimage.binary_dilation(labels == biggest, iterations=3) == 0] = 0  # loose bits of a neighbour go
    ys, xs = np.nonzero(picture[..., 3] > 128)
    return Image.fromarray(picture).crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))


def on_grey(view):
    plate = Image.new("RGBA", view.size, GREY + (255,))
    plate.alpha_composite(view)
    return plate.convert("RGB")


def lab(image):
    import cv2
    return cv2.cvtColor(np.asarray(image.convert("RGB"), np.float32) / 255, cv2.COLOR_RGB2LAB)


def front_view(renders, reference):
    """The render that shows the model from the side of its own picture. First the outline: how much the two shapes
    cover each other (128 x 128, both cropped to the object). A dark figure has nearly the same colours from every
    side, so colours alone took Nefertiti's side for her front (Nov 11). Front and back have the same outline: among
    the sides whose outline fits about as well as the best one, the colours on the object decide."""
    size = (128, 128)
    wanted = reference.resize(size, Image.LANCZOS)
    wanted_on, wanted_lab = np.asarray(wanted)[..., 3] > 128, lab(on_grey(wanted))
    outline, colours = [], []
    for render in renders:
        ys, xs = np.nonzero(np.asarray(render)[..., 3] > 128)
        crop = render.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)).resize(size, Image.LANCZOS)
        on = np.asarray(crop)[..., 3] > 128
        both = wanted_on & on
        outline.append(float(both.sum() / max(1, (wanted_on | on).sum())))
        colours.append(float(np.linalg.norm(lab(on_grey(crop))[both] - wanted_lab[both], axis=1).mean()) if both.any() else 1e9)
    print("    fit to the model's picture, per side: outline", [round(o, 2) for o in outline],
          "colour difference", [round(c, 1) for c in colours], flush=True)
    close = [k for k, o in enumerate(outline) if o >= max(outline) - 0.05]
    return min(close, key=lambda k: colours[k])


def turning_frames(video, folder):
    """Four frames of a video of the part turning once: its four sides, a quarter turn apart."""
    frames = json.loads(subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0", "-show_entries",
                                        "stream=nb_read_packets", "-of", "json", video], capture_output=True, text=True,
                                       check=True).stdout)["streams"][0]["nb_read_packets"]
    paths = []
    for k in range(4):
        paths.append(os.path.join(folder, f"part_{k}.png"))
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", video, "-vf", f"select=eq(n\\,{k * int(frames) // 4})", "-frames:v", "1",
                        paths[-1]], check=True)
    return paths


def collage(render, part, x, y):
    """AFTER and the 2D MASK of one side."""
    before = on_grey(render)
    after = before.copy()
    after.paste(part, (x, y), part)
    mask = Image.new("L", before.size, 0)
    mask.paste(part.getchannel("A").point(lambda v: 255 if v > 30 else 0), (x, y))
    return before, after, mask.filter(ImageFilter.MaxFilter(9))


def changed_part(args, render, side):
    """A ready AFTER picture instead of a part's picture: the new part is what differs from BEFORE inside the painted
    area. Sets --place to where it is; returns it cut out, as a part's picture would be."""
    before = np.asarray(on_grey(render), int)
    wanted = Image.open(args.after).convert("RGB").resize(render.size, Image.LANCZOS)
    painted = Image.open(os.path.join(args.hand or "", f"add_{side}.png"))
    painted = painted.getchannel("A") if painted.mode == "RGBA" else painted.convert("L")
    area = np.asarray(painted.resize(render.size, Image.NEAREST)) > 127
    differs = (np.abs(np.asarray(wanted, int) - before).sum(-1) > 40) & area
    differs = ndimage.binary_fill_holes(ndimage.binary_closing(differs, iterations=3))
    if differs.sum() < 50:  # nothing to see of the change: the whole painted area
        differs = area
    ys, xs = np.nonzero(differs)
    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    part = wanted.convert("RGBA")
    part.putalpha(Image.fromarray((differs * 255).astype(np.uint8)))
    args.place = [box[0], box[1], box[2] - box[0]]
    return part.crop(box)


def part_from_words(args, render, reference=None):
    """No picture of the part: the picture AI paints --prompt onto BEFORE, the new part is what differs from BEFORE
    (only the big changes: the picture AI's small noise elsewhere is not the part). Sets --place to where it is;
    returns it cut out, as a part's picture would be. Writes words.png (what the picture AI painted).
    reference: the part's own picture (cut out), when there is one: the picture AI paints that part, where the words say."""
    import torch
    from picture_from_prompt import comfy_edit
    torch.cuda.empty_cache()  # the picture AI needs this PC's whole GPU
    plate = on_grey(render)
    painted = os.path.join(args.out, "words.png")
    if not os.path.isfile(painted):  # the same words and seed give the same picture: not painted twice
        comfy_edit(plate, args.prompt, args.prompt_seed, reference=on_grey(reference) if reference is not None else None,
                   report=lambda text: print(f"    {text}", flush=True), template=WORDS).resize(render.size, Image.LANCZOS).save(painted)
    wanted = Image.open(painted).convert("RGB")
    differs = np.abs(np.asarray(wanted, int) - np.asarray(plate, int)).sum(-1) > 60
    differs = ndimage.binary_closing(ndimage.binary_opening(differs, iterations=2), iterations=4)
    labels, count = ndimage.label(differs)
    if not count:
        raise SystemExit(f"the picture AI changed nothing to see for “{args.prompt}”: other words, or another --prompt-seed")
    sizes = ndimage.sum(np.ones(labels.shape), labels, range(1, count + 1))
    big = [k + 1 for k, pixels_ in enumerate(sizes) if pixels_ > max(400, 0.25 * sizes.max())]
    differs = ndimage.binary_fill_holes(np.isin(labels, big))
    share = differs.mean()
    print(f"    the words changed {100 * share:.1f} % of the picture, in {len(big)} piece(s)", flush=True)
    if share > 0.5:
        raise SystemExit("the picture AI changed more than half of the picture: that is a new object, not a new part")
    ys, xs = np.nonzero(differs)
    box = (int(xs.min()), max(1, int(ys.min())), int(xs.max()) + 1, min(render.size[1] - 2, int(ys.max()) + 1))
    part = wanted.convert("RGBA")
    part.putalpha(Image.fromarray((differs * 255).astype(np.uint8)))
    if not args.place:
        args.place = [box[0], box[1], box[2] - box[0]]
    return part.crop(box)


def pixels(points, origin, right, size, per_pixel):
    """Where points of the model fall in a straight view (views.json): column and row."""
    u = ((points - origin) @ right / per_pixel + size / 2).astype(int)
    v = (size / 2 - (points[:, 1] - origin[1]) / per_pixel).astype(int)
    return u, v


def show_area(front, side_view, info, side, points, keep, delete, edit, origin, per_pixel, path):
    """The area's cubes on the model from two sides: green = KEEP, red = DELETE, blue = EDIT."""
    size = info["size"]
    sheet = Image.new("RGB", (2 * size, size), GREY)
    for k, (view, index) in enumerate(((front, side), (side_view, (side + 1) % 4))):
        picture = on_grey(view)
        right = np.array(info["views"][index]["right"])
        right = np.array([right[0], right[2], -right[1]])
        tint = picture
        for chosen, colour in ((edit, (70, 120, 255)), (delete, (235, 50, 40)), (keep, (40, 200, 90))):
            u, v = pixels(points[chosen], origin, right, size, per_pixel)
            good = (u >= 0) & (u < size) & (v >= 0) & (v < size)
            spots = np.zeros((size, size), np.uint8)
            spots[v[good], u[good]] = 150
            spots = Image.fromarray(spots).filter(ImageFilter.MaxFilter(7))
            tint = Image.composite(Image.new("RGB", tint.size, colour), tint, spots)
        sheet.paste(tint, (k * size, 0))
    sheet.resize((1400, 700), Image.LANCZOS).save(path, quality=92)


def main():
    parser = argparse.ArgumentParser(description="BEFORE, AFTER and the 2D mask for inpainting by picture")
    parser.add_argument("model")
    parser.add_argument("part", nargs="?", help="picture of the new part; a video of it turning once, or a folder of its sides in "
                        "turning order: all four sides are made. Left out with --prompt")
    parser.add_argument("--prompt", help="no picture of the part: the local picture AI paints these words onto the model's "
                        "picture of the main side, what differs from BEFORE is the new part. With a picture of the part: "
                        "the picture AI paints that part where the words say (e.g. \"the shield in the left hand\"), "
                        "instead of the measured place on the top")
    parser.add_argument("--prompt-seed", type=int, default=1, help="another seed gives another picture for the same words")
    parser.add_argument("--reference", help="the model's own picture: picks the side the collage is made on")
    parser.add_argument("--side", type=int, choices=range(4), help="or the side by hand")
    parser.add_argument("--out", required=True)
    parser.add_argument("--width", type=float, default=0.75, help="the part's width, share of the top of the object")
    parser.add_argument("--sink", type=float, default=0.15, help="share of the part's height pushed down into the object")
    parser.add_argument("--head", type=float, default=0.12, help="what counts as the top: share of the object's height")
    parser.add_argument("--fit", choices=["between", "over"], default="between",
                        help="when things stand up beside the middle (hair buns, horns): between = as wide as the free gap; "
                        "over = the full width, over them (use it with --replace: what the part covers is regenerated)")
    parser.add_argument("--place", type=int, nargs=3, metavar=("X", "Y", "WIDTH"),
                        help="by hand: the part's left, top and width in the main side's picture (pixels), nothing is measured")
    parser.add_argument("--hand", metavar="FOLDER", help="by hand: painted corrections of the 2D mask, "
                        "add_<view>.png (white = changes too) and erase_<view>.png (white = stays) per side")
    parser.add_argument("--after", metavar="PICTURE", help="by hand: a ready AFTER picture of the main side (--side), e.g. "
                        "from a prompt; the new part is what differs from BEFORE inside the painted area (--hand, add_<side>.png)")
    parser.add_argument("--replace", action="store_true", help="the model's surface under the 2D mask goes too (DELETE); "
                        "without it the whole surface stays and the part grows only into the empty room")
    args = parser.parse_args()
    if not (args.part or args.prompt or args.after):
        parser.error("give the part's picture, or --prompt with what should be added")
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

    given = bool(args.place)  # a place from the command line (the web page's edit by hand)
    if args.after:
        parts = [changed_part(args, render, side)]
    elif args.part and args.prompt and not args.place:  # the part's own picture, placed by the picture AI where the words say
        first = turning_frames(args.part, args.out)[0] if args.part.lower().endswith((".mp4", ".mov", ".webm", ".mkv")) else \
            sorted(os.path.join(args.part, n) for n in os.listdir(args.part) if n.lower().endswith((".png", ".jpg", ".jpeg")))[0] \
            if os.path.isdir(args.part) else args.part
        own = cut_out(remover, first)
        del remover
        parts = [part_from_words(args, render, own)]
    elif args.part:
        video = args.part.lower().endswith((".mp4", ".mov", ".webm", ".mkv"))
        if os.path.isdir(args.part):  # a folder: its pictures in turning order
            paths = [os.path.join(args.part, n) for n in sorted(os.listdir(args.part)) if n.lower().endswith((".png", ".jpg", ".jpeg"))][:4]
        else:
            paths = turning_frames(args.part, args.out) if video else [args.part]
        parts = [cut_out(remover, p) for p in paths]
    else:
        del remover
        parts = [part_from_words(args, render)]
    by_words = bool(args.prompt and (not args.part or not given))  # the picture AI painted it
    anywhere = bool(args.after or by_words)  # such a part is where the picture shows it, not only on the top
    for k, one in enumerate(parts):  # the cut-out parts: the web page shows them to place the part by hand
        one.save(os.path.join(args.out, f"cut_{k}.png"))
    part = parts[0]
    width = int(args.width * (right - left))
    part_height = int(part.height * width / part.width)
    # things standing up beside the middle (hair buns, horns, ears): the part sits on the object's middle line, between
    # them, as wide as the free gap there (a crown as wide as both buns did not fit into the picture, girl, Nov 10)
    centre = (left + right) // 2
    profile = np.where(alpha.any(0), alpha.argmax(0), alpha.shape[0])  # the object's top row in every column
    near = slice(centre - max(4, (right - left) // 20), centre + max(4, (right - left) // 20) + 1)
    middle_top = int(profile[near].max())  # the floor of the valley in the middle (a parting counts as its floor)
    fitted = middle_top - top > 0.1 * part_height and not args.place
    if fitted:
        level = (middle_top + top) / 2  # the gap is measured halfway up the things beside it
        free = profile > level
        to_left = centre - int(np.argmax(~free[centre::-1])) if not free[:centre + 1].all() else 0
        to_right = centre + int(np.argmax(~free[centre:])) if not free[centre:].all() else alpha.shape[1]
        gap = 2 * min(centre - to_left, to_right - centre)
        if args.fit == "between":
            width = max(8, min(width, gap))
        part_height = int(part.height * width / part.width)
        print(f"    the top stands {middle_top - top} px above the middle: the part sits on the middle, {width} px wide "
              f"(the free gap is {gap} px, fit: {args.fit})", flush=True)
        top = middle_top
    x = (left + right - width) // 2
    y = top - part_height + int(args.sink * part_height)
    if args.place:  # by hand
        x, y, width = args.place[0], args.place[1], max(8, args.place[2])
        part_height = max(2, int(part.height * width / part.width))
        print(f"    {'where the picture shows it' if anywhere and not given else 'placed by hand'}: left {x}, top {y}, {width} x {part_height} px", flush=True)
    part = part.resize((width, part_height), Image.LANCZOS)
    assert y > 0 and y + part_height < size and 0 <= x and x + width <= size, "the part doesn't fit into the picture"

    def by_hand(one, view_index):
        """The 2D mask of a side with the painted corrections: what was added changes too, what was erased stays."""
        for name, lighter in (("add", True), ("erase", False)):
            path = os.path.join(args.hand or "", f"{name}_{view_index}.png")
            if args.hand and os.path.isfile(path):
                painted = Image.open(path).convert("L").resize(one.size, Image.NEAREST).point(lambda v: 255 if v > 127 else 0)
                one = Image.composite(Image.new("L", one.size, 255 if lighter else 0), one, painted)
        return one

    before, after, mask = collage(render, part, x, y)
    mask = by_hand(mask, side)
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
    surface = trimesh.sample.sample_surface(mesh, 2_000_000, seed=0)[0]
    behind = None  # how far behind the area's middle it still reaches (None: all around, as a crown)
    if anywhere:
        # where the model is under the part, seen from the main side. A part that tops the model there (a hat) goes
        # all around the model's middle; any other (a shield, wings) sits on the side that is seen: the area starts
        # at the surface there and takes only a little of what lies behind it, so the far side stays out of it
        u, v = pixels(surface, to_gltf(center), right_axis, size, per_pixel)
        in_columns = (u >= x) & (u < x + width)
        under = in_columns & (v >= y) & (v < y + part_height)
        if not under.any():  # the part is beside the model: the nearest of the model counts
            reach = np.hypot(u - (x + width / 2), v - (y + part_height / 2))
            under = reach <= np.partition(reach, 2000)[2000]
        depth = (surface - to_gltf(center)) @ depth_axis
        tops = in_columns.any() and y <= v[in_columns].min() + 2
        low, high = np.percentile(depth[under], [2, 98])
        if tops:
            at = (low + high) / 2
        else:
            at, behind = high, max(3 * voxel, 0.35 * half)
        middle = to_gltf(center) + right_axis * across.mean() + depth_axis * at
        print(f"    the part {'tops the model: the area goes all around it' if tops else 'sits on the side that is seen: the area starts at the surface there'}", flush=True)
    steps = np.arange(-half, half + voxel / 2, voxel / 2)
    heights = np.arange(up[0] - voxel, up[1] + 2 * voxel, voxel / 2)
    a, b, c = np.meshgrid(steps, steps, heights, indexing="ij")
    points = middle + a[..., None] * right_axis + b[..., None] * depth_axis
    points = points.reshape(-1, 3)
    points[:, 1] = c.reshape(-1)
    round_ = np.hypot(a.reshape(-1), b.reshape(-1)) <= half  # a round area for a round part
    points = points[round_ & (b.reshape(-1) >= -behind)] if behind is not None else points[round_]

    sides = []
    masks = [mask]
    origin = to_gltf(center)
    for k in range(1, len(parts)):  # the other sides: the same place, each with its own side of the part
        index = (side + k) % 4
        right_k = to_gltf(np.array(info["views"][index]["right"]))
        middle_x = pixels(middle[None], origin, right_k, size, per_pixel)[0][0]
        one = parts[k].resize((max(2, int(parts[k].width * part_height / parts[k].height)), part_height), Image.LANCZOS)
        pictures = collage(renders[index], one, int(middle_x - one.width / 2), y)
        pictures = (pictures[0], pictures[1], by_hand(pictures[2], index))
        names = [os.path.join(args.out, f"{what}_{k}.png") for what in ("after", "before", "mask")]
        pictures[1].save(names[0])
        pictures[0].save(names[1])
        pictures[2].save(names[2])
        masks.append(pictures[2])
        sides += ["--more-edit", *names]

    # every small cube of the area: KEEP (model surface that stays), DELETE (old surface under the 2D mask), EDIT (empty).
    # Decided on TRELLIS.2's own 64^3 voxels, placed as inpaint.py places the model (Placement, with the room above):
    # a voxel the model's surface runs through is KEEP, the empty voxel right next to it is EDIT already. Before, all
    # within 1.5 voxels of the surface was kept, also the empty room there: a new part could not reach the model
    above = max(0.0, up[1] + 2 * voxel - vertices[:, 1].max())
    room = round(float(min(0.5, 1.05 * above / (above + model_size))), 3)
    low_, high_ = vertices.min(0), vertices.max(0)
    scale = 0.99999 * (1 - room) / (high_ - low_).max()

    def cells(p):
        p = (np.asarray(p) - (low_ + high_) / 2) * scale
        cube = np.stack([p[..., 0], -p[..., 2], p[..., 1] - room / 2], -1)
        return np.clip(((cube + 0.5) * GRID).astype(int), 0, GRID - 1)

    occupied = np.zeros((GRID,) * 3, bool)
    triangles = vertices[np.asarray(mesh.faces)]
    one, other = cells(triangles.min(1)), cells(triangles.max(1))
    first, last = np.minimum(one, other), np.maximum(one, other)
    for corner in np.ndindex(2, 2, 2):  # the corners of every face's box of voxels (a face is smaller than a voxel)
        index = np.where(np.array(corner, bool), last, first)
        occupied[index[:, 0], index[:, 1], index[:, 2]] = True
    index = cells(surface)  # and what large faces cover between their corners
    occupied[index[:, 0], index[:, 1], index[:, 2]] = True
    # inside the model is no empty room (new parts grew out through the head): a voxel with the model's surface on at
    # least five of its six sides (five: a bust is open at the bottom). The room under an arm or a chin stays free
    blocked = np.zeros((GRID,) * 3, int)
    for axis in range(3):
        seen = np.maximum.accumulate(occupied, axis)
        blocked += seen
        blocked += np.flip(np.maximum.accumulate(np.flip(occupied, axis), axis), axis)
    solid = occupied | (blocked >= 5)
    index = cells(points)
    on_surface = solid[index[:, 0], index[:, 1], index[:, 2]]
    changed = np.ones(len(points), bool)  # under the new part from every side, and not at its rim: what is deleted has
    rim = 2 * int(1.5 * voxel / per_pixel / 2) + 1  # to be covered by the part again, or a hole stays (crown, Nov 10)
    for k, one in enumerate(masks):
        right_k = to_gltf(np.array(info["views"][(side + k) % 4]["right"]))
        u, v = pixels(points, origin, right_k, size, per_pixel)
        changed &= np.asarray(one.filter(ImageFilter.MinFilter(rim)))[np.clip(v, 0, size - 1), np.clip(u, 0, size - 1)] > 127
    delete = on_surface & changed & args.replace
    keep = on_surface & ~delete
    edit = ~on_surface
    np.save(os.path.join(args.out, "area.npy"), points[edit | delete])  # all the rest stays the model's own
    show_area(renders[side], renders[(side + 1) % 4], info, side, points, keep, delete, edit, origin, per_pixel,
              os.path.join(args.out, "area_map.jpg"))

    report = {"side": side, "room": room, "keep": int(keep.sum()), "delete": int(delete.sum()), "edit": int(edit.sum()),
              "box_y": [float(up[0]), float(up[1])], "part_pixels": [x, y, width, part_height],
              "fitted_between": bool(fitted), "by_hand": bool(args.place or args.hand)}
    if by_words:
        report.update(prompt=args.prompt, prompt_seed=args.prompt_seed, by_hand=bool(args.hand))
    report["more_edit"] = sides
    json.dump(report, open(os.path.join(args.out, "edit.json"), "w"), indent=1)
    sheet = Image.new("RGB", (1500, 500), "white")
    for k, picture in enumerate((before, after, mask.convert("RGB"))):
        sheet.paste(picture.resize((500, 500), Image.LANCZOS), (k * 500, 0))
    sheet.save(os.path.join(args.out, "three.jpg"), quality=92)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
