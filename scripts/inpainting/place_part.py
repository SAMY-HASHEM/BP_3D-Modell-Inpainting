"""A new part in place of an old one, placed in 3D: the edit's pictures from all four sides, and a 3D area that is
only the new part and the old one.

make_edit.py with words gives one picture: the picture AI paints the new part onto the front. Its depth, its sides
and its back are then TRELLIS.2's guess, and the 3D area is a wide column in front of the model (the toaster robot's
toast -> book, Nov 15: a thin torn slab, loose pieces in the empty room). Here the new part is its own small 3D
model (generate.py from its picture), put where the old part is:

    1 the old part   the pieces of the model that the front picture shows inside the 2D mask (a model made of
                     several pieces: whole pieces; one fused mesh: its faces under the mask, --behind voxels deep)
    2 the place      the new part's middle and size from the front mask (what the picture AI painted), its depth and
                     its tilt from the old part: a flat part lies in the old part's plane
    3 the pictures   the model without the old part + the new part, rendered from the four straight sides: BEFORE,
                     AFTER and the 2D MASK (what differs) per side. All sides show the same part, because it is one
    4 the area       the small cubes within 1.5 voxels of the new part and those of the old part, without the voxels
                     the rest of the model runs through. Nothing else is regenerated
    5 zoom           (--zoom 0.5) only the model in a box around the two parts goes to TRELLIS.2, so the part fills
                     half of its cube instead of a few voxels (a book in a small robot's hand: 3 voxels thick in the
                     whole model, 9 zoomed). The box is the cube: tiny pegs at its corners make the model as big as
                     the box, so a new part that is bigger than the old one is not cut off at the model's edge
                     (the book reached 3 voxels past it, Nov 16). --finish puts the result's new part back into
                     the whole, untouched model

    python scripts/inpainting/place_part.py model.glb book.glb --edit outputs/x/work/edit --out outputs/x/work/edit_3d --zoom 0.5
    python scripts/server/cloud.py edit outputs/x/work/edit_3d outputs/x/work/edit_3d/before.glb --wait outputs/x/work/shape
    python scripts/inpainting/place_part.py --finish outputs/x/work/shape/result.glb --out outputs/x/work/edit_3d --final outputs/x/book.glb

The model that goes to the server is <out>/before.glb (one mesh, cut to the box with --zoom).

--edit is the folder make_edit.py wrote for the same change with --prompt (its views, its front mask, its side).
Writes the same files as make_edit.py (before.png, after.png, mask.png, after_1.png ..., area.npy, edit.json), so
cloud.py edit, edit_check.py and edit_judge.py take the folder as it is; and after.glb (the placed part, to look at).
"""
import _paths  # noqa: F401  (the scripts of the other folders)

import argparse
import json
import os
import shutil
import subprocess

import numpy as np
import trimesh
from PIL import Image, ImageFilter
from scipy import ndimage
from scipy.spatial import cKDTree

from make_edit import BLENDER, GRID, PROJECT, on_grey, pixels, show_area

to_gltf = lambda p: np.stack([p[..., 0], p[..., 2], -p[..., 1]], -1)  # Blender's axes (Z up) -> the model's (Y up)


def one_mesh(path):
    mesh = trimesh.load(path, force="mesh")
    assert isinstance(mesh, trimesh.Trimesh) and len(mesh.faces), f"no mesh in {path}"
    return mesh


def pieces_of(mesh):
    """Which piece every face belongs to. Corners at the same place count as one: a textured mesh has them doubled
    along every seam of its texture, and each patch of the texture would be a piece of its own."""
    _, same = np.unique(np.round(mesh.vertices, 6), axis=0, return_inverse=True)
    faces = same.reshape(-1)[mesh.faces]
    return trimesh.graph.connected_component_labels(trimesh.graph.face_adjacency(faces=faces), node_count=len(faces))


def old_part(mesh, mask, origin, right, toward, size, per_pixel, voxel, behind):
    """Which faces are the part that goes: what the front picture shows inside the 2D mask."""
    points, face = trimesh.sample.sample_surface(mesh, 1_500_000, seed=0)
    u, v = pixels(points, origin, right, size, per_pixel)
    inside = (u >= 0) & (u < size) & (v >= 0) & (v < size)
    points, face, u, v = points[inside], face[inside], u[inside], v[inside]
    depth = (points - origin) @ toward
    nearest = np.full((size, size), -np.inf)
    order = np.argsort(depth)  # the nearest point of every pixel is written last
    nearest[v[order], u[order]] = depth[order]
    seen = depth >= nearest[v, u] - 0.5 * voxel
    in_mask = mask[v, u]
    labels = pieces_of(mesh)
    pieces = labels.max() + 1
    shown = np.bincount(labels[face[seen]], minlength=pieces)
    shown_in = np.bincount(labels[face[seen & in_mask]], minlength=pieces)
    # the new part may be bigger than the old one: the mask then also shows the model behind it (the chest above a
    # toast). The old part is what stands in front: the nearer of the two depths seen in the mask (Otsu's limit), and
    # only pieces of some size (a fingertip around the toast's edge is not the toast)
    values = depth[seen & in_mask]
    counts, edges = np.histogram(values, 128)
    middles = (edges[:-1] + edges[1:]) / 2
    below, above = np.cumsum(counts), np.cumsum(counts[::-1])[::-1]
    mean_below = np.cumsum(counts * middles) / np.maximum(below, 1)
    mean_above = (np.cumsum((counts * middles)[::-1]) / np.maximum(np.cumsum(counts[::-1]), 1))[::-1]
    between = below[:-1] * above[1:] * (mean_below[:-1] - mean_above[1:]) ** 2
    limit = edges[1 + int(np.argmax(between))]
    in_front = np.bincount(labels[face[seen & in_mask & (depth >= limit)]], minlength=pieces)
    old = (shown_in >= 0.8 * shown) & (in_front >= 0.8 * np.maximum(shown_in, 1))
    old &= shown_in >= 0.03 * max(1, shown_in[old].max() if old.any() else 1)
    if pieces > 1 and old.any() and not old.all():
        print(f"    the old part: {int(old.sum())} of the model's {pieces} pieces", flush=True)
        return np.isin(labels, np.nonzero(old)[0])
    # one fused mesh (as TRELLIS.2 makes them): its faces under the mask, from the nearest surface --behind voxels deep
    front = np.percentile(depth[seen & in_mask], 98)
    chosen = np.zeros(len(mesh.faces), bool)
    chosen[np.unique(face[in_mask & (depth >= front - behind * voxel)])] = True
    print(f"    the old part: {int(chosen.sum())} faces under the mask, {behind:g} voxels deep (the model is one piece)", flush=True)
    return chosen


def axes_of(points):
    """A point cloud's own three directions, the longest first, and how far it reaches along each."""
    middle = points.mean(0)
    _, _, directions = np.linalg.svd(points - middle, full_matrices=False)
    reach = np.ptp((points - middle) @ directions.T, 0)
    order = np.argsort(-reach)
    return directions[order], reach[order]


def placed(part, old_points, mask, origin, right, toward, size, per_pixel, factor=1.0):
    """The new part moved, turned and scaled to its place: the 4 x 4 matrix. factor: bigger or smaller than painted."""
    up = np.array([0.0, 1.0, 0.0])
    cloud = trimesh.sample.sample_surface(part, 200_000, seed=0)[0]
    directions, reach = axes_of(cloud)
    flat = reach[2] < 0.35 * reach[1]  # a book, a shield, a plate: its thin direction decides how it lies
    if flat:
        thin = directions[2] * (1 if directions[2][2] >= 0 else -1)  # the side the part's picture shows looks at the camera
        long = max(directions[:2], key=lambda d: abs(d[1]))
        long = long * (1 if long[1] >= 0 else -1)
    else:  # TRELLIS.2's own pose: Y up, the front towards +Z
        thin, long = np.array([0.0, 0.0, 1.0]), up
    across = np.cross(long, thin)
    old_directions, old_reach = axes_of(old_points)
    normal = toward
    if flat and old_reach[2] < 0.5 * old_reach[1]:  # the old part is flat too: the new one lies in its plane
        normal = old_directions[2] * (1 if old_directions[2] @ toward >= 0 else -1)
    stand = up - (up @ normal) * normal
    if np.linalg.norm(stand) < 0.3:  # the old part lies flat on something: along its own longest direction
        stand = old_directions[0] - (old_directions[0] @ normal) * normal
    stand /= np.linalg.norm(stand)
    turn = np.stack([np.cross(stand, normal), stand, normal], 1) @ np.stack([across, long, thin], 1).T
    # the size and the middle in the picture: the mask without its thin arms (where only the old part went)
    solid = ndimage.binary_opening(mask, iterations=max(2, int(0.12 * np.sqrt(mask.sum()))))
    solid = solid if solid.sum() > 0.3 * mask.sum() else mask
    ys, xs = np.nonzero(solid)
    middle = cloud.mean(0)
    turned = (cloud - middle) @ turn.T
    # the picture AI painted the part upright: its height there is the part's own length, also when it then leans
    # (sized by its leaning height, the book came out a third too long and went through the hip)
    height = reach[0] if flat else np.ptp(turned[:, 1])
    scale = factor * (ys.max() - ys.min()) * per_pixel / height
    at = origin + right * ((xs.min() + xs.max()) / 2 - size / 2) * per_pixel
    at[1] = origin[1] + (size / 2 - (ys.min() + ys.max()) / 2) * per_pixel
    at = at + toward * ((old_points.mean(0) - origin) @ toward)
    matrix = np.eye(4)
    matrix[:3, :3] = scale * turn
    matrix[:3, 3] = at - scale * turn @ middle
    lean = np.degrees(np.arccos(np.clip(normal @ toward, -1, 1)))
    print(f"    the new part: {'flat, in the old part’s plane' if flat else 'upright'}, leaning {lean:.0f}° from the "
          f"camera, {scale * height:.2f} high (the model is {np.ptp(old_points[:, 1]):.2f} where the old part was)", flush=True)
    return matrix


def with_pegs(mesh, low, high, size):
    """The mesh with a tiny peg in every corner of the box: its bounds are then the box, and TRELLIS.2's cube with it."""
    corners = np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])])
    inward = np.sign((low + high) / 2 - corners) * size
    vertices, faces = [np.asarray(mesh.vertices)], [np.asarray(mesh.faces)]
    count = len(mesh.vertices)
    for corner, step in zip(corners, inward):
        vertices.append(corner + np.array([[0, 0, 0], [step[0], 0, 0], [0, step[1], 0], [0, 0, step[2]]]))
        faces.append(count + np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]))
        count += 4
    uv = np.concatenate([np.asarray(mesh.visual.uv), np.zeros((32, 2))])
    return trimesh.Trimesh(np.concatenate(vertices), np.concatenate(faces), process=False,
                           visual=trimesh.visual.TextureVisuals(uv=uv, material=mesh.visual.material))


def finish(args):
    """The result's new part put back into the whole model: everything else is the model's own mesh, untouched."""
    made = json.load(open(os.path.join(args.out, "edit.json")))
    rest, result = one_mesh(os.path.join(args.out, "rest_whole.glb")), one_mesh(args.finish)
    area = np.load(os.path.join(args.out, "area.npy"))
    near = cKDTree(area).query(result.triangles_center, distance_upper_bound=0.6 * made["voxel"])[0] < np.inf
    new = result.submesh([np.nonzero(near)[0]], append=True)
    scene = trimesh.Scene()
    scene.add_geometry(rest, geom_name="kept")
    scene.add_geometry(new, geom_name="new_part")
    scene.export(args.final)
    print(f"    {args.final}: the model's own {len(rest.faces):,} faces + {len(new.faces):,} new ones", flush=True)


def main():
    parser = argparse.ArgumentParser(description="a new part as a 3D model in place of an old one: pictures from all sides")
    parser.add_argument("model", nargs="?")
    parser.add_argument("part", nargs="?", help="the new part's own 3D model (.glb, e.g. from generate.py)")
    parser.add_argument("--edit", help="the folder make_edit.py --prompt wrote for this change (views, front mask)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--behind", type=float, default=4, help="one fused mesh: how deep the old part is, in voxels")
    parser.add_argument("--reach", type=float, default=1.5, help="the area around the new part, in voxels")
    parser.add_argument("--zoom", type=float, default=0.0, metavar="MARGIN", help="only a box around the two parts goes to "
                        "TRELLIS.2: the margin around them, as a share of their size (0.5: they fill half of the cube; 0: the whole model)")
    parser.add_argument("--size", type=float, default=1.0, help="the new part bigger or smaller than the picture AI painted it")
    parser.add_argument("--sides", type=int, choices=[4, 8], default=4, help="8: also the four sides in between (45 degrees)")
    parser.add_argument("--finish", metavar="RESULT", help="after the inpainting: its result's new part into the whole model")
    parser.add_argument("--final", help="--finish: the finished model's file")
    args = parser.parse_args()
    if args.finish:
        return finish(args)
    os.makedirs(args.out, exist_ok=True)
    made = json.load(open(os.path.join(args.edit, "edit.json")))
    info = json.load(open(os.path.join(args.edit, "views.json")))
    side, size = made["side"], info["size"]
    per_pixel = info["span"] / size
    origin = to_gltf(np.array(info["center"]))
    right, toward = (to_gltf(np.array(info["views"][side][k])) for k in ("right", "toward"))

    mesh, part = one_mesh(args.model), one_mesh(args.part)
    labels = pieces_of(part)
    sizes = np.bincount(labels, weights=part.area_faces)
    if (sizes < 0.05 * sizes.max()).any():  # loose bits beside the part (a piece of its shadow in the picture)
        part = part.submesh([np.nonzero(sizes[labels] >= 0.05 * sizes.max())[0]], append=True)
        print(f"    the new part's model: {int((sizes < 0.05 * sizes.max()).sum())} loose piece(s) left out", flush=True)
    front_mask = np.asarray(Image.open(os.path.join(args.edit, "mask.png")).convert("L")) > 127
    goes = old_part(mesh, front_mask, origin, right, toward, size, per_pixel, np.ptp(mesh.vertices, 0).max() / GRID, args.behind)
    old, rest = mesh.submesh([np.nonzero(goes)[0]], append=True), mesh.submesh([np.nonzero(~goes)[0]], append=True)
    old_points = trimesh.sample.sample_surface(old, 200_000, seed=0)[0]
    part.apply_transform(placed(part, old_points, front_mask, origin, right, toward, size, per_pixel, args.size))
    new_points = trimesh.sample.sample_surface(part, 300_000, seed=0)[0]
    both = np.concatenate([new_points, old_points])
    rest.export(os.path.join(args.out, "rest_whole.glb"))

    # the box that becomes TRELLIS.2's cube: around the two parts (--zoom), or the whole model and the new part
    if args.zoom:
        middle, half = (both.min(0) + both.max(0)) / 2, np.ptp(both, 0).max() / 2 * (1 + 2 * args.zoom)
        low_, high_ = middle - half, middle + half
        cut = lambda m: m.submesh([np.nonzero(np.all((m.triangles_center >= low_) & (m.triangles_center <= high_), 1))[0]], append=True)
        mesh, rest = cut(mesh), cut(rest)
        print(f"    zoom: a box of {2 * half:.2f} around the parts (the model is {np.ptp(one_mesh(args.model).vertices, 0).max():.2f}), "
              f"{len(mesh.faces):,} faces of the model in it", flush=True)
    else:
        low_, high_ = np.minimum(mesh.vertices.min(0), both.min(0)), np.maximum(mesh.vertices.max(0), both.max(0))
    voxel = (high_ - low_).max() / GRID
    low_, high_ = low_ - 0.5 * voxel, high_ + 0.5 * voxel
    voxel = (high_ - low_).max() / GRID
    mesh, rest = with_pegs(mesh, low_, high_, 0.4 * voxel), with_pegs(rest, low_, high_, 0.4 * voxel)
    trimesh.Scene([mesh]).export(os.path.join(args.out, "before.glb"))
    trimesh.Scene([rest, part]).export(os.path.join(args.out, "after.glb"))

    # the pictures: the same frame and light for BEFORE and AFTER, and both models saved the same way (a model saved
    # again by trimesh shines differently than its file did: the whole robot counted as changed), so only the change differs
    folder = os.path.join(args.out, "after_views")
    render = [BLENDER, "-b", "--python", os.path.join(PROJECT, "scripts/tools/render_views.py"), "--"]
    turn = ["--turn", "45"] if args.sides == 8 else []
    subprocess.run([*render, os.path.abspath(os.path.join(args.out, "before.glb")), os.path.abspath(args.out), *turn],
                   check=True, stdout=subprocess.DEVNULL)
    subprocess.run([*render, os.path.abspath(os.path.join(args.out, "after.glb")), os.path.abspath(folder), "--frame",
                    os.path.abspath(os.path.join(args.out, "views.json")), *turn], check=True, stdout=subprocess.DEVNULL)
    info = json.load(open(os.path.join(args.out, "views.json")))
    size, per_pixel, origin = info["size"], info["span"] / info["size"], to_gltf(np.array(info["center"]))
    view_of = [(side + k) % 4 for k in range(4)] + ([4 + (side + k) % 4 for k in range(4)] if args.sides == 8 else [])
    more, masks, tiles = [], [], []
    for k, index in enumerate(view_of):
        before = on_grey(Image.open(os.path.join(args.out, f"view_{index}.png")).convert("RGBA"))
        after = on_grey(Image.open(os.path.join(folder, f"view_{index}.png")).convert("RGBA"))
        differs = np.abs(np.asarray(after, int) - np.asarray(before, int)).sum(-1) > 45
        differs = ndimage.binary_closing(ndimage.binary_opening(differs, iterations=1), iterations=6)
        labels, count = ndimage.label(differs)
        if count:  # specks of the render's noise are not a change
            sizes = ndimage.sum(np.ones(labels.shape), labels, range(1, count + 1))
            differs = ndimage.binary_fill_holes(np.isin(labels, 1 + np.nonzero(sizes >= 150)[0]))
        mask = Image.fromarray((differs * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(9))
        names = [os.path.join(args.out, f"{what}.png" if k == 0 else f"{what}_{k}.png") for what in ("after", "before", "mask")]
        for picture, name in zip((after, before, mask), names):
            picture.save(name)
        if k:
            more += ["--more-edit", *names]
        masks.append(np.asarray(mask) > 127)
        tiles.append((before, after, mask.convert("RGB")))
        print(f"    view {index}: the change is {100 * masks[-1].mean():.1f} % of the picture", flush=True)
    sheet = Image.new("RGB", (len(tiles) * 300, 3 * 300), "white")
    for column, pictures in enumerate(tiles):
        for row, picture in enumerate(pictures):
            sheet.paste(picture.resize((300, 300), Image.LANCZOS), (column * 300, row * 300))
    sheet.save(os.path.join(args.out, "three.jpg"), quality=92)

    # the area: around the new part and where the old one was, on a grid of half voxels
    low, high = both.min(0) - 2 * voxel, both.max(0) + 2 * voxel
    grid = np.stack(np.meshgrid(*[np.arange(a, b, voxel / 2) for a, b in zip(low, high)], indexing="ij"), -1).reshape(-1, 3)
    near_new = cKDTree(new_points).query(grid, distance_upper_bound=args.reach * voxel)[0] < np.inf
    near_old = cKDTree(old_points).query(grid, distance_upper_bound=voxel)[0] < np.inf
    # the voxels the rest of the model runs through stay its own (as make_edit.py: TRELLIS.2's 64^3, placed as inpaint.py does)
    scale = 0.99999 / (high_ - low_).max()

    def cells(p):
        p = (np.asarray(p) - (low_ + high_) / 2) * scale
        return np.clip(((np.stack([p[..., 0], -p[..., 2], p[..., 1]], -1) + 0.5) * GRID).astype(int), 0, GRID - 1)

    occupied = np.zeros((GRID,) * 3, bool)
    index = cells(trimesh.sample.sample_surface(rest, 2_000_000, seed=0)[0])
    occupied[index[:, 0], index[:, 1], index[:, 2]] = True
    index = cells(grid)
    kept = occupied[index[:, 0], index[:, 1], index[:, 2]] & (near_new | near_old)
    delete, edit = near_old & ~kept, near_new & ~near_old & ~kept
    np.save(os.path.join(args.out, "area.npy"), grid[delete | edit])
    show_area(Image.open(os.path.join(args.out, f"view_{side}.png")).convert("RGBA"),
              Image.open(os.path.join(args.out, f"view_{(side + 1) % 4}.png")).convert("RGBA"), info, side, grid,
              kept, delete, edit, origin, per_pixel, os.path.join(args.out, "area_map.jpg"))
    ys, xs = np.nonzero(masks[0])
    report = {"side": side, "room": 0.0, "keep": int(kept.sum()), "delete": int(delete.sum()), "edit": int(edit.sum()),
              "box_y": [float(low[1]), float(high[1])],
              "part_pixels": [int(xs.min()), int(ys.min()), int(np.ptp(xs)) + 1, int(np.ptp(ys)) + 1], "fitted_between": False,
              "by_hand": False, "placed_in_3d": True, "zoom": args.zoom, "voxel": float(voxel), "view_of": view_of,
              "cube": [low_.tolist(), high_.tolist()], "part_voxels": [round(float(r / voxel), 1) for r in axes_of(new_points)[1]],
              "more_edit": more, **({"prompt": made["prompt"]} if "prompt" in made else {})}
    json.dump(report, open(os.path.join(args.out, "edit.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in report.items() if k not in ("more_edit", "cube")}), flush=True)


if __name__ == "__main__":
    main()
