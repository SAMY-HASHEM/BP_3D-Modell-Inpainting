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
    python scripts/pipeline/edit_check.py outputs/x/work/edit_3d --part "the toast in the hand is replaced by a book"
    python scripts/server/cloud.py edit outputs/x/work/edit_3d outputs/x/work/edit_3d/placed.glb --details --wait outputs/x/work/details
    python scripts/inpainting/place_part.py --finish outputs/x/work/details/result.glb --out outputs/x/work/edit_3d --final outputs/x/book.glb

The model that goes to the server is <out>/placed.glb: the model (cut to the box with --zoom) and the placed part as one
mesh; the details step keeps its voxels (docs/FIRST-SUCCESS.md). <out>/before.glb is the model without the new part,
for the shape step from noise (cloud.py edit <out> <out>/before.glb), which did not build a thin part.

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


def old_part_by_word(mesh, word, folder):
    """Which faces are the part that goes, found by its name ("axe") on pictures from eight sides: the mask AI
    (find_part.py: Grounding DINO + SAM 2.1, its own environment) marks it in every picture, and every face is voted on
    by the pictures that see it. For one fused mesh (what TRELLIS.2 makes), where the part is no piece of its own."""
    os.makedirs(folder, exist_ok=True)
    whole = os.path.join(folder, "model.glb")
    trimesh.Scene([mesh]).export(whole)
    if not os.path.isfile(os.path.join(folder, "render", "views.json")):
        subprocess.run([BLENDER, "-b", "--python", os.path.join(PROJECT, "scripts/tools/render_views.py"), "--", os.path.abspath(whole),
                        os.path.abspath(os.path.join(folder, "render")), "--turn", "45", "--samples", "16"], check=True, stdout=subprocess.DEVNULL)
    info = json.load(open(os.path.join(folder, "render", "views.json")))
    for k in range(8):
        on_grey(Image.open(os.path.join(folder, "render", f"view_{k}.png")).convert("RGBA")).save(os.path.join(folder, f"view_{k}.png"))
    # several names for the same thing, with commas ("axe, battle axe, weapon"): the mask AI often boxes the whole
    # figure that holds the thing (7 of 8 pictures of the warrior for "axe"). A mask over more than 40 % of the object in
    # its picture is the whole thing and does not vote; per picture the first name with a usable mask counts
    masks, names = {}, [w.strip() for w in word.split(",") if w.strip()]
    for n, name in enumerate(names):
        sub = os.path.join(folder, f"word_{n}")
        os.makedirs(sub, exist_ok=True)
        for k in range(8):
            if not os.path.exists(os.path.join(sub, f"view_{k}.png")):
                os.symlink(os.path.abspath(os.path.join(folder, f"view_{k}.png")), os.path.join(sub, f"view_{k}.png"))
        json.dump({"word": name, "views": [{"picture": f"view_{k}.png", "clicks": []} for k in range(8)]}, open(os.path.join(sub, "views.json"), "w"))
        if not os.path.isfile(os.path.join(sub, "masks.npz")):
            subprocess.run([os.path.join(PROJECT, "conda/envs/mask-ai/bin/python"), os.path.join(PROJECT, "scripts/inpainting/find_part.py"),
                            os.path.abspath(sub)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        found = np.load(os.path.join(sub, "masks.npz"))
        for k in range(8):
            if f"view_{k}" in found.files and f"view_{k}" not in masks:
                there = np.asarray(Image.open(os.path.join(folder, "render", f"view_{k}.png")).convert("RGBA").resize(found[f"view_{k}"].shape[::-1]))[..., 3] > 128
                if 0.005 < found[f"view_{k}"].sum() / max(1, there.sum()) < 0.4:
                    masks[f"view_{k}"] = found[f"view_{k}"]
    size, per_pixel, origin = info["size"], info["span"] / info["size"], to_gltf(np.array(info["center"]))
    voxel = np.ptp(mesh.vertices, 0).max() / GRID
    centres = mesh.triangles_center
    yes, seen_by = np.zeros(len(centres)), np.zeros(len(centres))
    for k in range(8):
        if f"view_{k}" not in masks:
            continue
        mask = np.asarray(Image.fromarray(masks[f"view_{k}"].astype(np.uint8) * 255).resize((size, size), Image.NEAREST)) > 127
        right, toward = (to_gltf(np.array(info["views"][k][key])) for key in ("right", "toward"))
        u, v = pixels(centres, origin, right, size, per_pixel)
        inside = (u >= 0) & (u < size) & (v >= 0) & (v < size)
        depth = (centres - origin) @ toward
        nearest = np.full((size, size), -np.inf)
        order = np.argsort(depth[inside])
        nearest[v[inside][order], u[inside][order]] = depth[inside][order]
        seen = inside.copy()
        seen[inside] = depth[inside] >= nearest[v[inside], u[inside]] - 0.75 * voxel
        seen_by += seen
        yes[seen] += mask[v[seen], u[seen]]
    print(f"    “{word}” was found in {len(masks)} of 8 pictures: {sorted(masks)}", flush=True)
    assert masks, f"the mask AI found no {word} that is not the whole object: other words"
    voted = seen_by > 0
    chosen = np.zeros(len(centres), bool)
    chosen[voted] = yes[voted] / seen_by[voted] >= 0.5
    sure = voted & (yes / np.maximum(seen_by, 1) >= 0.8)  # marked in nearly every picture that sees it
    if (~voted).any() and voted.any():  # a face no picture sees (in a groove): as the nearest face that was seen
        chosen[~voted] = chosen[voted][cKDTree(centres[voted]).query(centres[~voted])[1]]
    # decided per small cube of space, not per face: a cube whose faces were mostly chosen goes as a whole, and only the
    # big connected groups of such cubes count. Face by face the part stayed speckled: every second face of the axe was
    # left (a face seen from two sides and marked in one of them, faces of the inner side no picture sees)
    from scipy import ndimage
    cells = 160
    low, span = centres.min(0), np.ptp(centres, 0).max() * 1.0001
    cell = np.minimum(((centres - low) / span * cells).astype(int), cells - 1)
    flat = (cell[:, 0] * cells + cell[:, 1]) * cells + cell[:, 2]
    area = mesh.area_faces
    share = np.bincount(flat, weights=area * chosen, minlength=cells ** 3) / np.maximum(np.bincount(flat, weights=area, minlength=cells ** 3), 1e-12)
    cube = ndimage.binary_closing((share > 0.3).reshape(cells, cells, cells), iterations=1)
    groups, count = ndimage.label(cube, structure=np.ones((3, 3, 3)))
    if count:
        sizes = ndimage.sum(cube, groups, range(1, count + 1))
        cube = np.isin(groups, 1 + np.nonzero(sizes >= 0.1 * sizes.max())[0])
    cube = ndimage.binary_dilation(cube, structure=np.ones((3, 3, 3)))  # and its rim: thin slivers stayed at the part's edges
    chosen = cube.reshape(-1)[flat]
    # by colour too: the cubes around an axe's shaft also hold the fingers that close around it, and cutting them out
    # left a torn hole in the fist that the inpainting did not close (the warrior, Nov 16). The part's own colours are
    # those of the faces marked in nearly every picture; a chosen face with none of them stays with the model
    texture = getattr(mesh.visual.material, "baseColorTexture", None)
    if texture is not None and (sure & chosen).sum() > 500:
        image = np.asarray(texture.convert("RGB"), float)
        uv = np.asarray(mesh.visual.uv)[mesh.faces].mean(1)
        colour = image[np.clip(((1 - uv[:, 1]) * image.shape[0]).astype(int), 0, image.shape[0] - 1),
                       np.clip((uv[:, 0] * image.shape[1]).astype(int), 0, image.shape[1] - 1)]
        own = colour[sure & chosen]
        own = own[np.random.default_rng(0).choice(len(own), min(20000, len(own)), replace=False)]
        centres_ = own[np.linspace(0, len(own) - 1, 3).astype(int)].copy()
        for _ in range(12):  # three main colours of the part
            nearest_ = np.linalg.norm(own[:, None] - centres_[None], axis=2).argmin(1)
            centres_ = np.array([own[nearest_ == k].mean(0) if (nearest_ == k).any() else centres_[k] for k in range(3)])
        centres_ = centres_[np.bincount(nearest_, minlength=3) >= 0.08 * len(own)]
        away = np.linalg.norm(colour[:, None] - centres_[None], axis=2).min(1)
        stays = chosen & (away > 55)
        print(f"    by colour: {int(stays.sum()):,} chosen faces are not the part's colour (the hand around it): they stay", flush=True)
        chosen &= ~stays
    print(f"    the old part: {int(chosen.sum()):,} of the model's {len(centres):,} faces", flush=True)
    return chosen


def profile(points, axis, slices=60):
    """Along a long thing: per slice how far it reaches from its middle line (None where it is empty)."""
    middle = points.mean(0)
    along = (points - middle) @ axis
    away = np.linalg.norm((points - middle) - along[:, None] * axis, axis=1)
    edges = np.linspace(along.min(), along.max(), slices + 1)
    index = np.clip(np.digitize(along, edges) - 1, 0, slices - 1)
    reach = np.array([np.percentile(away[index == k], 90) if (index == k).sum() > 20 else np.nan for k in range(slices)])
    return (edges[:-1] + edges[1:]) / 2, reach, middle


def placed_in_hand(part, old_points, rest_points, hand, voxel, factor=1.0):
    """A held thing (a sword) put into the hand that held the old one (an axe): the 4 x 4 matrix.
    The grip lies on the old thing's shaft where the hand closes around it, so the fist that held the shaft holds the
    grip; the blade points to where the old thing's head was; the grip is as thick as the shaft was there (x factor).
    hand: the hand's joint from the skeleton (rig_parts.py), in the model's coordinates."""
    cloud = trimesh.sample.sample_surface(part, 200_000, seed=0)[0]
    (long, wide, thin), _ = axes_of(cloud)
    at, reach, middle = profile(cloud, long)
    inner = np.arange(len(at))[4:-4]
    # the grip: the thinnest place that has something thicker on both sides (the pommel and the guard); a blade's tip has not
    grips = [k for k in inner if np.isfinite(reach[k]) and np.nanmax(reach[:k]) > 1.3 * reach[k] < np.nanmax(reach[k + 1:]) / 1.3]
    grip = min(grips, key=lambda k: reach[k]) if grips else int(inner[np.nanargmin(reach[inner])])
    blade = 1.0 if (at[-1] - at[grip]) > (at[grip] - at[0]) else -1.0  # the long end from the grip is the blade
    grip_point, grip_radius = middle + at[grip] * long, float(reach[grip])
    # where the hand closes around the old thing: the model's own surface right next to it, near the hand's joint.
    # The old thing as it was marked is often only its head (an axe's blade; the shaft hides in the fist), so the
    # direction is from the fist to the old thing's middle, not the old thing's own longest side
    hand = np.asarray(hand, float)
    touching = (cKDTree(old_points).query(rest_points, distance_upper_bound=2.5 * voxel)[0] < np.inf) & \
               (np.linalg.norm(rest_points - hand, axis=1) < 10 * voxel)
    fist = rest_points[touching].mean(0) if touching.sum() > 30 else hand
    out = old_points.mean(0) - fist
    out /= np.linalg.norm(out)
    flat = axes_of(old_points)[0][2]
    flat = flat - (flat @ out) * out
    flat /= np.linalg.norm(flat)
    far = float(np.percentile((old_points - fist) @ out, 99))  # the blade's tip comes where the old thing ended
    scale = factor * far / abs((at[-1] if blade > 0 else at[0]) - at[grip])
    target = np.stack([out, np.cross(flat, out), flat], 1)
    own = np.stack([blade * long, np.cross(thin, blade * long), thin], 1)
    turn = target @ own.T
    matrix = np.eye(4)
    matrix[:3, :3] = scale * turn
    matrix[:3, 3] = fist - scale * turn @ grip_point
    shaft_radius = float("nan")
    print(f"    the new part in the hand: its grip ({2 * grip_radius * scale / voxel:.1f} voxels thick, the shaft was "
          f"{2 * shaft_radius / voxel:.1f}) where the hand held the old one, {np.ptp(cloud @ long) * scale / voxel:.0f} voxels long", flush=True)
    return matrix


def axes_of(points):
    """A point cloud's own three directions, the longest first, and how far it reaches along each."""
    middle = points.mean(0)
    _, _, directions = np.linalg.svd(points - middle, full_matrices=False)
    reach = np.ptp((points - middle) @ directions.T, 0)
    order = np.argsort(-reach)
    return directions[order], reach[order]


def side_colours(part, thin, count=60_000):
    """The mean colour (RGB 0-255) of a flat part's two big sides: the one thin points to, and the other one."""
    try:
        coloured = part.copy()
        coloured.visual.material = coloured.visual.material.to_simple()  # a PBR material: its base colour texture
        points, _, colours = trimesh.sample.sample_surface(coloured, count, seed=0, sample_color=True)
    except Exception:  # noqa: BLE001 (no texture: the side is not known by colour)
        return None
    depth = (points - points.mean(0)) @ thin
    front, back = depth > 0.3 * depth.max(), depth < 0.3 * depth.min()
    if front.sum() < 100 or back.sum() < 100:
        return None
    return colours[front, :3].mean(0), colours[back, :3].mean(0)


def placed(part, old_points, mask, origin, right, toward, size, per_pixel, factor=1.0, wanted=None):
    """The new part moved, turned and scaled to its place: the 4 x 4 matrix. factor: bigger or smaller than painted.
    wanted: the colour the AFTER picture shows inside the mask (RGB): a flat part turns the side of that colour to the camera."""
    up = np.array([0.0, 1.0, 0.0])
    cloud = trimesh.sample.sample_surface(part, 200_000, seed=0)[0]
    directions, reach = axes_of(cloud)
    flat = reach[2] < 0.35 * reach[1]  # a book, a shield, a plate: its thin direction decides how it lies
    if flat:
        thin = directions[2] * (1 if directions[2][2] >= 0 else -1)  # the side the part's picture shows looks at the camera
        # which side that is, is measured when the colours tell: TRELLIS.2 does not always turn the picture's side to +Z.
        # The red book of Nov 19 had its red cover at -Z and a guessed grey-blue back at +Z, and was placed back to front.
        sides = side_colours(part, thin) if wanted is not None else None
        if sides is not None:
            near, far = (float(np.linalg.norm(c - np.asarray(wanted, float))) for c in sides)
            if far < 0.8 * near:
                thin = -thin
            print(f"    the side that looks at the camera: by colour (wanted {np.round(wanted).astype(int).tolist()}, "
                  f"this side {np.round(sides[0 if far >= 0.8 * near else 1]).astype(int).tolist()}, "
                  f"{'turned round' if far < 0.8 * near else 'as it was'})", flush=True)
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


def thickened(part, new_points, both, mesh, zoom, least):
    """The 4 x 4 matrix that makes a flat part `least` voxels thick in TRELLIS.2's cube, or None when it is thick enough.
    The toaster robot's book, Nov 19: 3.6 voxels thick, the details step built only its frame and left the covers hollow
    and speckled; the book of Nov 16 was 5 voxels thick and came out whole. Only the thin direction is stretched, around
    the part's middle: a book a little thicker still looks like the same book in the hand."""
    directions, reach = axes_of(new_points)
    if reach[2] >= 0.35 * reach[1]:  # not flat
        return None
    span = np.ptp(both, 0).max() * (1 + 2 * zoom) if zoom else np.maximum(np.ptp(mesh.vertices, 0), np.ptp(both, 0)).max()
    voxel = span / GRID
    if reach[2] >= least * voxel:
        return None
    factor = least * voxel / reach[2]
    thin, middle = directions[2], new_points.mean(0)
    stretch = np.eye(4)
    stretch[:3, :3] += (factor - 1) * np.outer(thin, thin)
    stretch[:3, 3] = middle - stretch[:3, :3] @ middle
    print(f"    the new part is {reach[2] / voxel:.1f} voxels thick: made {least:.0f} (x {factor:.2f} in its thin direction)", flush=True)
    return stretch


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
    # the model's own surface that was made again (--adjust: the fist around a new grip) goes: the result has it anew
    again = cKDTree(area).query(rest.triangles_center, distance_upper_bound=0.6 * made["voxel"])[0] < np.inf
    if again.any():
        rest = rest.submesh([np.nonzero(~again)[0]], append=True)
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
    parser.add_argument("--min-voxels", type=float, default=5.0, help="a flat part thinner than this in TRELLIS.2's cube is "
                        "made this thick (a book 3.6 voxels thick came out hollow, 5 whole); 0: as it is")
    parser.add_argument("--old", metavar="WORD", help="without --edit: the old part by its name (\"axe\"), found on pictures "
                        "from eight sides by the mask AI. For one fused mesh, where the part is no piece of its own")
    parser.add_argument("--hand", nargs=2, metavar=("RIG_JSON", "PART"), help="with --old: the new part is a held thing; its grip "
                        "goes into this hand of the skeleton (rig_parts.py's json, e.g. hand_right), where it held the old one")
    parser.add_argument("--adjust", type=float, default=0.0, metavar="VOXELS", help="the model's own surface this near to the new "
                        "part is made again too (a fist around a new grip), and the cut edges where the old part was")
    parser.add_argument("--front", type=int, default=0, help="without --edit: which straight view is the model's front (0: as TRELLIS.2 puts it)")
    parser.add_argument("--finish", metavar="RESULT", help="after the inpainting: its result's new part into the whole model")
    parser.add_argument("--final", help="--finish: the finished model's file")
    args = parser.parse_args()
    if args.finish:
        return finish(args)
    os.makedirs(args.out, exist_ok=True)
    made, side = {}, args.front
    if args.edit:
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
    if args.edit:
        front_mask = np.asarray(Image.open(os.path.join(args.edit, "mask.png")).convert("L")) > 127
        goes = old_part(mesh, front_mask, origin, right, toward, size, per_pixel, np.ptp(mesh.vertices, 0).max() / GRID, args.behind)
    else:
        goes = old_part_by_word(mesh, args.old, os.path.join(args.out, "find"))
    old, rest = mesh.submesh([np.nonzero(goes)[0]], append=True), mesh.submesh([np.nonzero(~goes)[0]], append=True)
    old_points = trimesh.sample.sample_surface(old, 200_000, seed=0)[0]
    if args.hand:
        rig = json.load(open(args.hand[0]))  # a part (hand_right) or one bone (arm_right_4: the wrist)
        joint = np.array((rig["parts"].get(args.hand[1]) or rig["bones"][args.hand[1]])["joint"])
        part.apply_transform(placed_in_hand(part, old_points, trimesh.sample.sample_surface(rest, 400_000, seed=0)[0], joint,
                                            np.ptp(mesh.vertices, 0).max() / GRID, args.size))
    else:
        after = np.asarray(Image.open(os.path.join(args.edit, "after.png")).convert("RGB"), float)
        inside = ndimage.binary_erosion(front_mask, iterations=3) if front_mask.shape == after.shape[:2] else None
        wanted = after[inside].mean(0) if inside is not None and inside.sum() > 50 else None  # the painted part's colour
        part.apply_transform(placed(part, old_points, front_mask, origin, right, toward, size, per_pixel, args.size, wanted))
    new_points = trimesh.sample.sample_surface(part, 300_000, seed=0)[0]
    both = np.concatenate([new_points, old_points])
    if args.min_voxels:  # a flat part too thin for TRELLIS.2's cube is made a little thicker (it would come out hollow)
        thicker = thickened(part, new_points, both, mesh, args.zoom, args.min_voxels)
        if thicker is not None:
            part.apply_transform(thicker)
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
    # the same as ONE mesh with one texture: the model for the details step, which keeps these voxels and makes only the
    # fine shape and the join again (cloud.py edit <out> <out>/placed.glb --details). From noise a thin part came out
    # as shards or a small plate; on the placed part's voxels it came out whole (the toaster robot's book, Nov 16)
    from combine_parts import pack
    trimesh.Scene([pack([part, rest])]).export(os.path.join(args.out, "placed.glb"))

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
    if args.adjust:  # the model's own surface next to the new part and at the cut edges: made again, so it joins
        again = (cKDTree(new_points).query(grid, distance_upper_bound=args.adjust * voxel)[0] < np.inf) | \
                (cKDTree(old_points).query(grid, distance_upper_bound=voxel)[0] < np.inf)
        near_old |= kept & again
        kept &= ~again
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
