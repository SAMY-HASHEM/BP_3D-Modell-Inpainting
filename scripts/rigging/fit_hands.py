"""The hands of a rigged character fitted to its mesh: the finger bones moved into the mesh's own fingers and the
fingers' skin made from the mesh, not from the rig head.

Why (Nov 23): the rig head sees the 64^3 voxel grid; a chibi finger is ~1 voxel thick. On the Plague Knight its 15
finger bones of each hand pointed straight down into the sleeve while his (3-fingered) hands point out sideways,
and of the right hand's fingers the middle finger's 3 bones moved 0 points of the mesh, the thumb 11 800 (sleeve
included): closing the fingers bent bones that moved nothing; the hand stayed open around the axe.

On the mesh, for each hand:
  1 the hand    the points the rig gives mostly to the hand and its fingers, welded (a UV seam splits a point in
                two), split into connected pieces; the hand is the piece that reaches furthest on along the forearm
                (the others: sleeve the voxel skin gave to the hand)
  2 fingers     distance from the wrist along the surface (Dijkstra); the points of each 1/40 of it in rings, a ring
                split into pieces where fingers are apart. Followed from each fingertip down, a finger's rings go on
                until they meet another finger or the palm (the knuckle; a ring broken in two is still one
                finger). The thumb is the finger that leaves the palm nearest to the wrist; the others ordered from the thumb: index, middle, ring, little (a hand of 3
                fingers: the little finger's bones on the ring finger's, moving nothing; hold_item.py needs them)
  3 bones       each finger's line: its rings' middles, from a little inside the palm (the knuckle joint lies
                under the skin, before the finger parts from the next one) to its tip; its 3 joints along it at
                0 / 0.45 / 0.75 of it (thumb 0 / 0.4 / 0.72)
  4 skin        a finger's points to its 3 bones by where they lie along its line (blended across each joint), the
                palm's to the hand, smoothed along the surface a little; the hand's share of a sleeve piece goes to
                the forearm. Points of a finger belong to that finger only (nothing of the sleeve or the body)

    python scripts/rigging/fit_hands.py rigged.blend out.blend [--glb]      (the project's Python, with scipy)

Runs Blender twice (read the mesh, write the bones and weights); prints for each finger its length and points.
"""
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(HERE))
BLENDER = next((b for b in (os.path.join(PROJECT_DIR, "blender", "blender"),) if os.path.exists(b)), "blender")
FINGERS = ("thumb", "index", "middle", "ring", "little")
JOINTS = {"thumb": (0.0, 0.40, 0.72), "other": (0.0, 0.45, 0.75)}
INTO_PALM = {"thumb": 0.5, "other": 0.3}  # how far the first joint lies before where the finger parts, of its length
LEVELS = 40


def hand_side(name):
    """'r' / 'l' for the hand and finger bones (hand_r, index2_l ...), else None."""
    if name[-2:] not in ("_r", "_l"):
        return None
    base = name[:-2]
    if base == "hand" or (base[:-1] in FINGERS and base[-1:] in "123"):
        return name[-1]
    return None


# ---------------------------------------------------------------- Blender: read and write

def blender_read(src, out):
    import bpy
    bpy.ops.wm.open_mainfile(filepath=os.path.abspath(src))
    s = bpy.context.scene
    rig = next(o for o in s.objects if o.type == "ARMATURE")
    bodies = [o for o in s.objects if o.type == "MESH" and o.parent == rig and o.parent_type != "BONE"]
    pts, faces, hand_w, owner, offset = [], [], {"r": [], "l": []}, [], 0
    for k, obj in enumerate(bodies):
        m = obj.matrix_world
        pts.append(np.array([tuple(m @ v.co) for v in obj.data.vertices]))
        obj.data.calc_loop_triangles()
        faces.append(np.array([t.vertices[:] for t in obj.data.loop_triangles]) + offset)
        side_of = {g.index: hand_side(g.name) for g in obj.vertex_groups}
        for side in "rl":
            w = np.zeros(len(obj.data.vertices))
            for v in obj.data.vertices:
                w[v.index] = sum(g.weight for g in v.groups if side_of.get(g.group) == side)
            hand_w[side].append(w)
        owner.append(np.full(len(obj.data.vertices), k))
        offset += len(obj.data.vertices)
    bones = {b.name: (tuple(rig.matrix_world @ b.head_local), tuple(rig.matrix_world @ b.tail_local))
             for b in rig.data.bones}
    np.savez(out, points=np.concatenate(pts), faces=np.concatenate(faces), owner=np.concatenate(owner),
             hand_r=np.concatenate(hand_w["r"]), hand_l=np.concatenate(hand_w["l"]), bones=json.dumps(bones))


def blender_write(src, fit_path, out, glb):
    import bpy
    from mathutils import Vector
    bpy.ops.wm.open_mainfile(filepath=os.path.abspath(src))
    s = bpy.context.scene
    rig = next(o for o in s.objects if o.type == "ARMATURE")
    bodies = [o for o in s.objects if o.type == "MESH" and o.parent == rig and o.parent_type != "BONE"]
    fit = np.load(fit_path)
    bones = json.loads(str(fit["bones"]))
    # bones (in rest): heads and tails in world -> the armature's space
    inv = rig.matrix_world.inverted()
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    for name, (head, tail) in bones.items():
        eb = rig.data.edit_bones.get(name)
        if eb is None:
            continue
        eb.use_connect = False
        eb.head, eb.tail = inv @ Vector(head), inv @ Vector(tail)
    bpy.ops.object.mode_set(mode="OBJECT")
    # weights: vertex (over all body meshes), what to clear, what to add
    vid, clear_all, names, add_v, add_b, add_w = (fit[k] for k in ("vid", "clear_all", "names", "add_v", "add_b",
                                                                     "add_w"))
    names = [str(n) for n in names]
    offset = 0
    for obj in bodies:
        n = len(obj.data.vertices)
        mine = (vid >= offset) & (vid < offset + n)
        local = (vid[mine] - offset).tolist()
        everything = (vid[mine & clear_all] - offset).tolist()
        for g in list(obj.vertex_groups):
            if hand_side(g.name):
                g.remove(local)
            elif everything:
                g.remove(everything)
        groups = {g.name: g for g in obj.vertex_groups}
        sel = (add_v >= offset) & (add_v < offset + n)
        for v, b, w in zip((add_v[sel] - offset).tolist(), add_b[sel].tolist(), add_w[sel].tolist()):
            name = names[b]
            if name not in groups:
                groups[name] = obj.vertex_groups.new(name=name)
            groups[name].add([v], float(w), "ADD")
        offset += n
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(out))
    if glb:
        bpy.ops.export_scene.gltf(filepath=os.path.splitext(os.path.abspath(out))[0] + ".glb", export_format="GLB",
                                  export_skins=True)
    print("hands fitted:", out, flush=True)


# ---------------------------------------------------------------- the fit (scipy)

def weld(points, faces):
    _, index = np.unique(np.round(points / 1e-5).astype(np.int64), axis=0, return_inverse=True)
    index = index.ravel()
    p = np.zeros((index.max() + 1, 3))
    p[index] = points
    e = index[np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])]
    return p, index, e[e[:, 0] != e[:, 1]]


def graph(p, edges, nodes):
    import scipy.sparse as sp
    sel = np.full(len(p), -1)
    sel[nodes] = np.arange(len(nodes))
    e = sel[edges]
    e = e[(e >= 0).all(1)]
    length = np.linalg.norm(p[nodes[e[:, 0]]] - p[nodes[e[:, 1]]], axis=1) + 1e-9
    a = sp.coo_matrix((length, (e[:, 0], e[:, 1])), shape=(len(nodes),) * 2).tocsr()
    return a.maximum(a.T)


def pieces(points, radius):
    """Pieces of a point set: points within radius of each other are one piece."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    pairs = cKDTree(points).query_pairs(radius, output_type="ndarray")
    a = sp.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(points),) * 2)
    return connected_components(a, directed=False)[1]


def along(line, q):
    """Where points q lie along a polyline, 0..1 of its length (by the nearest point on it), and how far off it."""
    seg = np.diff(line, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    start = np.concatenate([[0], np.cumsum(seg_len)])
    best_s, best_d = np.zeros(len(q)), np.full(len(q), np.inf)
    for k in range(len(seg)):
        t = np.clip(((q - line[k]) @ seg[k]) / max(seg_len[k] ** 2, 1e-12), 0, 1)
        d = np.linalg.norm(q - (line[k] + t[:, None] * seg[k]), axis=1)
        closer = d < best_d
        best_d[closer], best_s[closer] = d[closer], start[k] + t[closer] * seg_len[k]
    return best_s / start[-1], best_d


def at(line, s):
    seg_len = np.linalg.norm(np.diff(line, axis=0), axis=1)
    start = np.concatenate([[0], np.cumsum(seg_len)])
    x = s * start[-1]
    k = min(np.searchsorted(start, x, side="right") - 1, len(seg_len) - 1)
    t = (x - start[k]) / max(seg_len[k], 1e-12)
    return line[k] + t * (line[k + 1] - line[k])


def find_fingers(q, g, step, edge=0.0):
    """Fingers on the hand's points q (g: their distance from the wrist along the surface): rings of `step`, each in
    pieces; followed from the tips down as branches: a piece touching one branch above goes on with it (several pieces
    of one branch: its ring broke in two, still the one finger), touching none starts one (a tip), touching two
    or more that are each 3 rings long or more is where they join (their knuckles; a shorter one is noise and
    goes on in the other). Returns [(point ids, ring middles from knuckle to tip, tip point, knuckle distance)]."""
    from scipy.spatial import cKDTree
    radius = max(step * 0.9, edge * 2.0)  # a coarse mesh: the ring's points further apart than the ring is thick
    levels = []
    for lv in range(int(g.max() / step) + 1):
        ids = np.nonzero((g >= lv * step) & (g < (lv + 1) * step))[0]
        lab = pieces(q[ids], radius) if len(ids) > 1 else np.zeros(len(ids), int)
        levels.append([ids[lab == c] for c in range(lab.max() + 1 if len(ids) else 0) if (lab == c).sum() >= 6])
    branches = []  # each: {"rings": [[ids of pieces] per level, top first], "top": level, "open": bool, "merged": bool}
    above = []     # (piece ids, branch) of the level above
    for lv in range(len(levels) - 1, -1, -1):
        here = []
        for piece in levels[lv]:
            tree = cKDTree(q[piece])
            touch = sorted({b for ids, b in above
                            if np.isfinite(tree.query(q[ids], distance_upper_bound=radius * 1.3)[0]).any()})
            real = [b for b in touch if len(branches[b]["rings"]) >= 3 or branches[b]["merged"]]
            if not touch:
                branches.append({"rings": [], "top": lv, "open": True, "merged": False})
                b = len(branches) - 1
            elif len(real) >= 2:
                for t in touch:
                    branches[t]["open"] = False
                branches.append({"rings": [], "top": lv, "open": True, "merged": True})
                b = len(branches) - 1
            else:
                b = real[0] if real else max(touch, key=lambda t: len(branches[t]["rings"]))
                for t in touch:
                    if t != b and branches[t]["open"] and not real:
                        branches[t]["open"] = False
                if not branches[b]["open"]:  # it was closed by another piece of this level: a new merged one
                    branches.append({"rings": [], "top": lv, "open": True, "merged": True})
                    b = len(branches) - 1
            here.append((piece, b))
        for piece, b in here:
            rings = branches[b]["rings"]
            if rings and branches[b].get("last") == lv:
                rings[-1] = np.concatenate([rings[-1], piece])
            else:
                rings.append(piece)
                branches[b]["last"] = lv
        above = here
    fingers = []
    for br in branches:
        if br["merged"] or br["open"] or len(br["rings"]) < 4:
            continue  # the palm (what fingers joined into, or what reaches the wrist) or noise
        rings = br["rings"]
        ids = np.concatenate(rings)
        middles = np.array([q[c].mean(0) for c in rings[::-1]])
        tip = q[rings[0][np.argmax(g[rings[0]])]]
        fingers.append((ids, middles, tip, (br["top"] - len(rings) + 1) * step))
    fingers.sort(key=lambda f: -len(f[1]))
    if fingers:
        longest = len(fingers[0][1])
        fingers = [f for f in fingers if len(f[1]) >= 0.35 * longest][:5]
    return fingers


def fit_side(side, p, edges, hand_w, bones):
    """New bones (name -> head, tail) and new hand-side weights for one hand."""
    from scipy.sparse.csgraph import connected_components, dijkstra
    wrist = np.array(bones[f"hand_{side}"][0])
    elbow = np.array(bones[f"lower_arm_{side}"][0])
    arm = (wrist - elbow) / np.linalg.norm(wrist - elbow)
    nodes = np.nonzero(hand_w > 0.5)[0]
    a = graph(p, edges, nodes)
    _, lab = connected_components(a, directed=False)
    sizes = np.bincount(lab)
    big = np.nonzero(sizes > 0.1 * sizes.max())[0]
    best = max(big, key=lambda c: (p[nodes[lab == c]].mean(0) - wrist) @ arm)
    hand = nodes[lab == best]
    q = p[hand]
    a = graph(p, edges, hand)
    start = np.argsort(np.linalg.norm(q - wrist, axis=1))[:max(10, len(hand) // 50)]
    g = dijkstra(a, directed=False, indices=start, min_only=True)
    g[~np.isfinite(g)] = 0
    edge = float(np.mean(a.data))
    step = max(g.max() / LEVELS, edge * 1.5)  # rings at least 1.5 edges thick (the Sister's hand: 2.5 mm edges)
    found = find_fingers(q, g, step, edge)
    report = {"hand points": int(len(hand)), "fingers found": len(found)}
    if len(found) < 2:
        print(f"hand {side}: only {len(found)} fingers found; left as it was", flush=True)
        return None, None, report
    # the thumb: the finger that parts from the palm nearest to the wrist (when it is clearly the earliest)
    by_knuckle = sorted(range(len(found)), key=lambda i: found[i][3])
    thumb = by_knuckle[0] if found[by_knuckle[0]][3] < 0.8 * found[by_knuckle[1]][3] else None
    size = [len(f[0]) for f in found]
    if len(found) == 2 and max(size) >= 3 * min(size):  # a mitten and a thumb: the small one is the thumb
        thumb = int(np.argmin(size))
    rest = [i for i in range(len(found)) if i != thumb]
    if thumb is not None:
        t0 = found[thumb][1][0]
        rest.sort(key=lambda i: np.linalg.norm(found[i][1][0] - t0))
    else:  # no thumb: across the hand, from one side
        across = np.cross(arm, [0, 0, 1])
        rest.sort(key=lambda i: found[i][1][0] @ across)
    # fewer than 4 fingers apart: 1 = a mitten (the fingers fused into one block: the Sister of War's), it is the
    # middle finger; 2 = index, middle; 3 = index, middle, ring. The others' bones on the nearest one, moving nothing
    slots = {1: ["middle"], 2: ["index", "middle"], 3: ["index", "middle", "ring"]}.get(
        len(rest), ["index", "middle", "ring", "little"])
    assign = {"thumb": thumb} if thumb is not None else {}
    for nm, i in zip(slots, rest):
        assign[nm] = i
    lines, new_bones = {}, {}
    for nm in FINGERS:
        i = assign.get(nm)
        if i is None:
            continue
        ids, middles, tip, _ = found[i]
        kind = "thumb" if nm == "thumb" else "other"
        if len(middles) >= 3:  # a little smoothed
            middles = np.vstack([middles[:1], (middles[:-2] + middles[1:-1] + middles[2:]) / 3, middles[-1:]])
        line = np.vstack([middles, tip])
        length = np.linalg.norm(np.diff(line, axis=0), axis=1).sum()
        d = line[min(2, len(line) - 1)] - line[0]
        d /= max(np.linalg.norm(d), 1e-9)
        line = np.vstack([line[0] - d * length * INTO_PALM[kind], line])
        lines[nm] = (i, line, kind)
        j = JOINTS[kind]
        pts = [at(line, s) for s in j] + [line[-1]]
        for k in range(3):
            new_bones[f"{nm}{k + 1}_{side}"] = (pts[k].tolist(), pts[k + 1].tolist())
        report[nm] = {"length": round(float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum()), 4),
                      "points": int(len(ids))}
    order_across = ["index", "middle", "ring", "little"]
    for nm in order_across:  # a hand of fewer fingers: these on the nearest one found
        if nm not in lines and lines:
            have = [x for x in order_across if x in lines]
            if not have:
                break
            src = min(have, key=lambda x: abs(order_across.index(x) - order_across.index(nm)))
            for k in range(3):
                new_bones[f"{nm}{k + 1}_{side}"] = new_bones[f"{src}{k + 1}_{side}"]
            report[nm] = f"none (on the {src} finger, moving nothing)"
    if "middle" in lines:
        new_bones[f"hand_{side}"] = (wrist.tolist(), new_bones[f"middle1_{side}"][0])
    # weights over the hand: bones hand, then the fingers' 3 each
    order = [f"hand_{side}"] + [f"{nm}{k}_{side}" for nm in FINGERS for k in (1, 2, 3)]
    w = np.zeros((len(hand), len(order)))
    w[:, 0] = 1
    finger_point = np.zeros(len(hand), bool)
    taken = {}
    for nm, (i, line, kind) in lines.items():
        ids = found[i][0]
        s, _ = along(line, q[ids])
        if i in taken:
            continue
        taken[i] = nm
        j0, j1 = JOINTS[kind][1], JOINTS[kind][2]  # where the 2nd and 3rd bones begin
        base = INTO_PALM[kind] / (1 + INTO_PALM[kind])  # where the finger parts from the palm, along its line
        blend = 0.06
        ramp = lambda x, c: np.clip((x - c) / blend + 0.5, 0, 1)
        f1 = 1 - ramp(s, j0)
        f2 = ramp(s, j0) * (1 - ramp(s, j1))
        f3 = ramp(s, j1)
        into = np.clip((s - base) / 0.08, 0, 1)  # the hand still holds the finger's very base a little
        col = order.index(f"{nm}1_{side}")
        w[ids] = 0
        w[ids, 0] = 1 - into
        w[ids, col], w[ids, col + 1], w[ids, col + 2] = f1 * into, f2 * into, f3 * into
        finger_point[ids] = into > 0.5
    # a little smoothing along the surface (neighbours' mean, 4 rounds)
    adj = (a > 0).astype(float)
    deg = np.asarray(adj.sum(1)).ravel().clip(1)
    for _ in range(4):
        w = 0.5 * w + 0.5 * (adj @ w) / deg[:, None]
    w /= w.sum(1, keepdims=True).clip(1e-9)
    report["fingers"] = {nm: taken.get(i) for nm, (i, _, _) in lines.items()}
    return new_bones, (hand, order, w, finger_point), report


def fit(src, out, glb=False):
    tmp = os.path.splitext(os.path.abspath(out))[0] + "_hands"
    subprocess.run([BLENDER, "-b", "--python", os.path.abspath(__file__), "--", "read", src, tmp + "_mesh.npz"],
                   check=True, stdout=subprocess.DEVNULL)
    d = np.load(tmp + "_mesh.npz")
    bones = json.loads(str(d["bones"]))
    p, index, edges = weld(d["points"], d["faces"])
    first = np.full(len(p), -1)
    first[index[::-1]] = np.arange(len(index))[::-1]
    all_bones, names = {}, []
    add_v, add_b, add_w, vid, clear_all = [], [], [], [], []
    report = {}
    name_id = lambda nm: names.index(nm) if nm in names else (names.append(nm) or len(names) - 1)
    for side in "rl":
        hw_v = d[f"hand_{side}"]
        hw = np.zeros(len(p))
        np.maximum.at(hw, index, hw_v)
        new_bones, weights, report[side] = fit_side(side, p, edges, hw, bones)
        print(f"hand {side}:", json.dumps(report[side]), flush=True)
        if new_bones is None:
            continue
        all_bones.update(new_bones)
        hand, order, w, finger_point = weights
        from scipy.spatial import cKDTree
        node_row = np.full(len(p), -1)
        node_row[hand] = np.arange(len(hand))
        # every vertex with any hand-side weight: on the hand (or within a touch of it) -> new weights; else its
        # hand share to the forearm (a sleeve piece)
        touched = np.nonzero(hw_v > 0.01)[0]
        row = node_row[index[touched]]
        near_d, near = cKDTree(p[hand]).query(d["points"][touched])
        close = (row < 0) & (near_d < 0.004)
        row[close] = near[close]
        for v, r, total in zip(touched.tolist(), row.tolist(), hw_v[touched].tolist()):
            vid.append(v)
            if r >= 0 and finger_point[r]:
                clear_all.append(True)
                total = 1.0
            else:
                clear_all.append(False)
            if r >= 0:
                for b in np.nonzero(w[r] > 0.01)[0]:
                    add_v.append(v), add_b.append(name_id(order[b])), add_w.append(total * w[r, b])
            else:
                add_v.append(v), add_b.append(name_id(f"lower_arm_{side}")), add_w.append(total)
        # finger points the rig gave mostly to other bones (not in the hand piece): none here, by construction
    np.savez(tmp + "_fit.npz", bones=json.dumps(all_bones), vid=np.array(vid, np.int64),
             clear_all=np.array(clear_all, bool), names=np.array(names), add_v=np.array(add_v, np.int64),
             add_b=np.array(add_b, np.int64), add_w=np.array(add_w, np.float32))
    json.dump({"bones": all_bones, "report": report}, open(tmp + "_fit.json", "w"), indent=1)
    subprocess.run([BLENDER, "-b", "--python", os.path.abspath(__file__), "--", "write", src, tmp + "_fit.npz", out,
                    "glb" if glb else "-"], check=True, stdout=subprocess.DEVNULL)
    print("hands fitted:", out, flush=True)
    return report


if __name__ == "__main__":
    if "--" in sys.argv:  # inside Blender
        a = sys.argv[sys.argv.index("--") + 1:]
        if a[0] == "read":
            blender_read(a[1], a[2])
        else:
            blender_write(a[1], a[2], a[3], a[4] == "glb")
    else:
        fit(sys.argv[1], sys.argv[2], "--glb" in sys.argv)
