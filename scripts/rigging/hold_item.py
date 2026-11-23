"""A rigged character holding an item (a weapon, a tool) in its hand, the fingers closed around the item's handle.

    blender -b --python scripts/rigging/hold_item.py -- rigged.blend item.glb <out> [--hand r] [--size 0.45]
        [--grip 0.18] [--swing]

  the handle    the item's long axis (its points' main direction); its head is the wider end (an axe's blade, a
                hammer's head), the grip on the other end, --grip of the length from it (0.18: a hand's width up
                from the end); its thickness there measured. --size: the item's length in the character's heights
  the palm      from the rig (rig_from_glb.py): the fingers' direction (hand -> middle1), the knuckles' line
                (little1 -> index1), and the palm's side (where the thumb sits); the palm's surface found with a ray
  the hold      the handle across the palm, along the knuckles, the head out on the thumb's side, the blade
                pointing where the fingers point; the item a child of the hand bone (it goes wherever the hand goes)
  the fingers   each finger's three joints bent together, a few degrees at a time, until its skin meets the
                handle (its bones within the handle's radius + the finger's own thickness, measured on the mesh)
                or a joint reaches its limit; the thumb: the pose of it (swung, bent) whose skin rests on the handle
  the check     on the posed mesh: each finger's skin gap to the handle and how much of it is inside: PASS / FAIL
                (<out>_grip.json). The hands must fit the mesh (fit_hands.py; every rig since Nov 23 does)
  --swing       an animation to check the hold: the arm down, raised, a chop (48 frames)

Writes <out>.blend (press Space with --swing), <out>.glb (the character, its armature and the item on the hand) and
<out>_hand_<k>.png, <out>_body_<k>.png (the grip close up from 3 sides, the whole body), <out>_grip.json.
The rigged .blend keeps the twist bones' constraints and preserve volume (rig_from_glb.py); a .glb is set up again.
"""
import json
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
FINGER_NAMES = ("index", "middle", "ring", "little")


def args():
    a = sys.argv[sys.argv.index("--") + 1:]
    opts = {"hand": "r", "size": 0.45, "grip": 0.18, "swing": False, "pose": "down"}
    rest = []
    i = 0
    while i < len(a):
        if a[i] == "--swing":
            opts["swing"] = True
        elif a[i].startswith("--"):
            key = a[i][2:]
            opts[key] = type(opts[key])(a[i + 1]) if key in opts and not isinstance(opts[key], bool) else a[i + 1]
            i += 1
        else:
            rest.append(a[i])
        i += 1
    return rest, opts


def load_character(path):
    if path.endswith(".blend"):
        bpy.ops.wm.open_mainfile(filepath=path)
    else:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.gltf(filepath=path)
    rig = next(o for o in bpy.context.scene.objects if o.type == "ARMATURE")
    if not path.endswith(".blend"):
        sys.path.insert(0, HERE)
        from rig_from_glb import better_skinning
        better_skinning(rig)
    body = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    return rig, body


def world(rig, name, end="head"):
    pb = rig.pose.bones[name]
    return rig.matrix_world @ (pb.head if end == "head" else pb.tail)


def turn_bone(rig, name, axis, degrees, pivot=None):
    """Turns a pose bone about a world axis through its head (or pivot), on top of its present pose."""
    pb = rig.pose.bones[name]
    inv = rig.matrix_world.inverted()
    axis_a = (inv.to_3x3() @ axis).normalized()
    pivot_a = inv @ pivot if pivot is not None else pb.head.copy()
    r = Matrix.Translation(pivot_a) @ Quaternion(axis_a, math.radians(degrees)).to_matrix().to_4x4() @ \
        Matrix.Translation(-pivot_a)
    pb.matrix = r @ pb.matrix
    bpy.context.view_layer.update()


def surface_points(obj, n=40000):
    """Points spread evenly over the mesh's surface (world space): a mesh may have its vertices only at the ends of
    a long handle (a cylinder), so the vertices alone can't say how thick the handle is in its middle."""
    obj.data.calc_loop_triangles()
    tris = obj.data.loop_triangles
    v = np.array([obj.matrix_world @ x.co for x in obj.data.vertices])
    idx = np.zeros((len(tris), 3), int)
    tris.foreach_get("vertices", idx.ravel())
    a, b, c = v[idx[:, 0]], v[idx[:, 1]], v[idx[:, 2]]
    area = np.linalg.norm(np.cross(b - a, c - a), axis=1) / 2
    pick = np.random.default_rng(0).choice(len(tris), n, p=area / area.sum())
    r1, r2 = np.random.default_rng(1).random((2, n))
    flip = r1 + r2 > 1
    r1[flip], r2[flip] = 1 - r1[flip], 1 - r2[flip]
    return a[pick] + (b[pick] - a[pick]) * r1[:, None] + (c[pick] - a[pick]) * r2[:, None]


def item_axes(item):
    """The item's long axis, its head end and its blade direction, its length; all in world space."""
    pts = surface_points(item)
    centre = pts.mean(0)
    u, s, vt = np.linalg.svd(pts - centre, full_matrices=False)
    axis = vt[0]
    t = (pts - centre) @ axis
    lo, hi = t.min(), t.max()
    # the head: the end whose last fifth spreads widest away from the axis
    radial = np.linalg.norm((pts - centre) - np.outer(t, axis), axis=1)
    end_lo = radial[t < lo + 0.2 * (hi - lo)].max()
    end_hi = radial[t > hi - 0.2 * (hi - lo)].max()
    if end_lo > end_hi:
        axis, t, lo, hi = -axis, -t, -hi, -lo
    head_pts = (pts - centre)[t > hi - 0.25 * (hi - lo)]
    off = head_pts - np.outer(head_pts @ axis, axis)
    blade = off[np.argmax(np.linalg.norm(off, axis=1))]
    blade = blade / max(np.linalg.norm(blade), 1e-9)
    return pts, centre, axis, t, lo, hi, blade


def handle_at(pts, centre, axis, t, at):
    """The handle's middle and radius at the axis position `at`."""
    near = np.abs(t - at) < 0.02 * (t.max() - t.min())
    sl = pts[near]
    mid = sl.mean(0)
    mid = centre + axis * at + ((mid - centre) - axis * ((mid - centre) @ axis))  # on the axis' line, centred
    r = np.median(np.linalg.norm((sl - mid) - np.outer((sl - mid) @ axis, axis), axis=1))
    return mid, r


def place_item(rig, body, item, hand, size, grip):
    z = np.concatenate([[(o.matrix_world @ v.co).z for v in o.data.vertices] for o in body])
    height = float(np.percentile(z, 99.5) - np.percentile(z, 0.5))  # the mesh's own points (the Sister's glb's
    # objects' boxes said 2.0 for a model 1 tall: her axe came out as long as she is)
    pts, centre, axis, t, lo, hi, blade = item_axes(item)
    scale = size * height / (hi - lo)
    item.scale = item.scale * scale
    bpy.context.view_layer.update()
    pts, centre, axis, t, lo, hi, blade = item_axes(item)
    at = lo + grip * (hi - lo)
    grip_mid, radius = handle_at(pts, centre, axis, t, at)

    H = world(rig, "hand_" + hand)
    M = world(rig, "middle1_" + hand)
    I, L = world(rig, "index1_" + hand), world(rig, "little1_" + hand)
    T = world(rig, "thumb1_" + hand)
    f = (M - H).normalized()
    a = (I - L)
    a = (a - f * a.dot(f)).normalized()
    n = (T - H) - f * (T - H).dot(f) - a * (T - H).dot(a)
    n = n.normalized() if n.length > 1e-6 else f.cross(a).normalized()
    # where a held bar lies: across the base of the fingers, just in front of the knuckles' line, on the palm side.
    # From the hand's own joints, not a ray (the knight's ray hit his sleeve: the handle 4 cm off, the fingers short)
    K = (I + L) / 2
    seg = (world(rig, "middle2_" + hand) - M).length
    # ...moved back into the palm towards the thumb when the thumb grows far back (a chibi's short thumb near the
    # wrist could not reach a handle at the finger bases: its base 5.3 cm off the handle's axis, it 3.8 cm long)
    out_along = seg * 0.35
    if "thumb2_" + hand in rig.pose.bones:
        thumb_at = (world(rig, "thumb2_" + hand) - K).dot(f)
        out_along = max(min(out_along, (out_along + thumb_at) / 2), -seg * 1.2)
    target = K + f * out_along + n * (radius + seg * 0.28)
    dist = (target - K).length
    # turn the item: its axis along the knuckles (head on the index side), its blade where the fingers point
    src = Matrix((Vector(axis), Vector(blade), Vector(np.cross(axis, blade)))).transposed()
    b = (f - a * f.dot(a)).normalized()
    dst = Matrix((a, b, a.cross(b))).transposed()
    turn = (dst @ src.inverted()).to_4x4()
    item.matrix_world = Matrix.Translation(target) @ turn @ Matrix.Translation(-Vector(grip_mid)) @ item.matrix_world
    bpy.context.view_layer.update()
    # the item on the hand bone
    world_m = item.matrix_world.copy()
    item.parent, item.parent_type, item.parent_bone = rig, "BONE", "hand_" + hand
    bpy.context.view_layer.update()
    item.matrix_world = world_m
    bpy.context.view_layer.update()
    print(f"item: {(hi - lo):.3f} long ({size:.2f} of {height:.3f}), handle radius {radius:.4f}, its axis "
          f"{dist:.4f} from the knuckles", flush=True)
    return target, Vector(a), radius, n


CURL = (1.0, 1.1, 0.8)        # how much each joint bends for one step of the whole finger (base, middle, tip)
LIMIT = (95, 110, 85)         # and at most (degrees); the thumb: close_thumb


def open_hand(rig, hand):
    """The fingers straightened before the item goes in: each finger's first segment pointing from the hand bone's
    head through its own base, the next two in line with it (a generated hand is often half curled at rest: the
    Sister of War's was, and the handle landed between her fingers, which then could not close)."""
    H = world(rig, "hand_" + hand)
    for finger in FINGER_NAMES:
        names = [f"{finger}{k}_{hand}" for k in (1, 2, 3)]
        if any(nm not in rig.pose.bones for nm in names):
            continue
        d = (world(rig, names[0]) - H).normalized()
        for nm in names:
            point_bone(rig, nm, d)


def finger_thickness(rig, body, hand):
    """Each finger's own thickness (half of it): its points' (its bones' weight over 0.5) median distance to its bones,
    at rest. A finger stops when its skin, not its bone, meets the handle (a guess from the bones' lengths, as before
    Nov 23, stopped the knight's fingers 7 mm short once the handle lay deeper in his palm)."""
    m = np.array(rig.matrix_world)
    ends = lambda b: [m[:3, :3] @ np.array(v) + m[:3, 3] for v in (b.head_local, b.tail_local)]
    out = {}
    for finger in ("thumb",) + FINGER_NAMES:
        names = [f"{finger}{k}_{hand}" for k in (1, 2, 3)]
        if any(nm not in rig.data.bones for nm in names):
            continue
        segs = [ends(rig.data.bones[nm]) for nm in names]
        dists = []
        for obj in body:
            ids = {obj.vertex_groups[nm].index for nm in names if nm in obj.vertex_groups}
            if not ids:
                continue
            mw = obj.matrix_world
            pts = np.array([tuple(mw @ v.co) for v in obj.data.vertices
                            if sum(g.weight for g in v.groups if g.group in ids) > 0.5])
            if len(pts) == 0:
                continue
            best = np.full(len(pts), np.inf)
            for h, t in segs:
                ab = t - h
                u = np.clip(((pts - h) @ ab) / max(ab @ ab, 1e-12), 0, 1)
                best = np.minimum(best, np.linalg.norm(pts - (h + u[:, None] * ab), axis=1))
            dists.append(best)
        if dists:
            out[finger] = float(np.median(np.concatenate(dists)))
    print("finger thickness (half):", {k: round(v, 4) for k, v in out.items()}, flush=True)
    return out


def close_fingers(rig, hand, grip_point, axis, radius, palm, thick=None):
    """Each finger curls all its joints together (CURL), two degrees a step; a segment that touches the handle (its
    middle or its end within the handle's radius + a finger's thickness of the handle's axis) stops its own joint and
    those before it, the joints after it go on, so the finger wraps the handle; each joint stops at its LIMIT.
    The way to bend: the one that brings the fingertip nearer the handle."""
    def to_axis(p):
        d = p - grip_point
        return (d - axis * d.dot(axis)).length
    for finger in FINGER_NAMES:
        names = [f"{finger}{k}_{hand}" for k in (1, 2, 3)]
        if any(nm not in rig.pose.bones for nm in names):
            continue
        curl, limit = CURL, LIMIT
        seg = (world(rig, names[0]) - world(rig, names[1])).length
        skin = (thick or {}).get(finger, max(seg * 0.3, radius * 0.2))  # the finger's half thickness
        inside_margin = skin * 0.7  # its skin may press in a little, its bone not
        def inside_handle(chain):  # any later segment's middle or end inside the handle (with a finger's half
            for nm in chain:           # thickness): the finger would go through it
                h, e = world(rig, nm), world(rig, nm, "tail")
                if min(to_axis(e), to_axis((h + e) / 2)) < radius + inside_margin:
                    return True
            return False
        touch = radius + skin * 1.1
        tip = lambda: world(rig, names[2], "tail")

        before = to_axis(tip())
        turn_bone(rig, names[0], axis, 4)
        sign = 1 if to_axis(tip()) < before else -1
        turn_bone(rig, names[0], axis, -4)
        bent = [0.0, 0.0, 0.0]
        active = [True, True, True]
        for _ in range(80):
            if not any(active):
                break
            for k in range(3):
                if active[k]:
                    step = 2 * curl[k]
                    turn_bone(rig, names[k], axis, sign * step)
                    if inside_handle(names[k:]):  # into the handle: undo, this joint is done
                        turn_bone(rig, names[k], axis, -sign * step)
                        active[k] = False
                        continue
                    bent[k] += step
                    if bent[k] >= limit[k]:
                        active[k] = False
            for k in range(3):  # a segment touching the handle: it and the joints before it stop
                head, end = world(rig, names[k]), world(rig, names[k], "tail")
                if min(to_axis(end), to_axis((head + end) / 2)) <= touch:
                    for j in range(k + 1):
                        active[j] = False
        print(f"  {finger}: bent {[round(x) for x in bent]} degrees, tip {to_axis(tip()) / radius:.2f} handle radii "
              "from its axis", flush=True)


def close_thumb(rig, hand, grip_point, axis, radius, closer=0.0, thick=None):
    """The thumb onto the handle: of its poses (swung across the palm, its base bent, its two outer joints bent
    together; each way), the one whose tip rests on the handle's surface (its radius + a thumb's thickness from its
    axis), near the fist along it, no segment inside the handle. (Curling it about the handle's axis like a finger,
    as before Nov 23, swung it up and away once its first bone sat inside the palm where it belongs: 4.9 radii.)"""
    names = [f"thumb{k}_{hand}" for k in (1, 2, 3)]
    if any(nm not in rig.pose.bones for nm in names):
        return
    def to_axis(p):
        d = p - grip_point
        return (d - axis * d.dot(axis)).length
    rest = [rig.pose.bones[nm].matrix_basis.copy() for nm in names]
    along = (world(rig, "middle1_" + hand) - world(rig, "hand_" + hand)).normalized()
    seg = (world(rig, names[2], "tail") - world(rig, names[1])).length / 2
    skin = (thick or {}).get("thumb", seg * 0.35)
    goal = max(radius * 1.05, radius + skin - closer)  # closer: what the skin check found still between

    def pose(swing, c1, c23):
        for nm, m in zip(names, rest):
            rig.pose.bones[nm].matrix_basis = m.copy()
        bpy.context.view_layer.update()
        turn_bone(rig, names[0], along, swing)
        turn_bone(rig, names[0], axis, c1)
        turn_bone(rig, names[1], axis, c23)
        turn_bone(rig, names[2], axis, c23)

    def cost():
        for nm in names:
            h, e = world(rig, nm), world(rig, nm, "tail")
            if min(to_axis(e), to_axis((h + e) / 2)) < radius + skin * 0.7:
                return np.inf  # through the handle
        tip = world(rig, names[2], "tail")
        off = abs((tip - grip_point).dot(axis))
        return abs(to_axis(tip) - goal) + 0.3 * max(0.0, off - 2 * radius)

    best = (np.inf, 0, 0, 0)
    for swing in range(-60, 61, 15):
        for c1 in range(-45, 46, 15):
            for c23 in range(-80, 81, 20):
                pose(swing, c1, c23)
                best = min(best, (cost(), swing, c1, c23))
    _, s0, a0, b0 = best
    for swing in range(s0 - 10, s0 + 11, 5):  # finer around it
        for c1 in range(a0 - 10, a0 + 11, 5):
            for c23 in range(b0 - 15, b0 + 16, 5):
                if abs(c23) <= 90:
                    pose(swing, c1, c23)
                    best = min(best, (cost(), swing, c1, c23))
    pose(*best[1:])
    print(f"  thumb: swung {best[1]}, base {best[2]}, joints {best[3]} degrees, tip "
          f"{to_axis(world(rig, names[2], 'tail')) / radius:.2f} handle radii from its axis", flush=True)


def mesh_follows(body, on):
    """The body's armature modifiers on or off: off while only bones are posed and measured (each step otherwise
    deforms the whole mesh again: 675 000 points of the knight, the thumb's search took minutes)."""
    for obj in body:
        for m in obj.modifiers:
            if m.type == "ARMATURE":
                m.show_viewport = on
    bpy.context.view_layer.update()


def grip_check(rig, body, hand, grip_point, axis, radius, out):
    """Does the skin really hold the handle? (Nov 23: the bones said 'bent', the fingers of the mesh did not move:
    0 points on the knight's middle finger.) The mesh as posed (after the armature), each finger's own points (its
    bones' weight over 0.5): the nearest one's gap to the handle's surface, in handle radii, and how many lie inside
    it. A finger holds when its gap is under 0.35 radii and under 2 % of its points are inside. Written to
    <out>_grip.json, printed as PASS / FAIL."""
    dg = bpy.context.evaluated_depsgraph_get()
    g = np.array(grip_point)
    ax = np.array(axis)
    result = {}
    names = ("thumb",) + FINGER_NAMES
    pts = {f: [] for f in names}
    for obj in body:
        groups = {gr.index: gr.name for gr in obj.vertex_groups}
        weight = {f: np.zeros(len(obj.data.vertices)) for f in names}
        for v in obj.data.vertices:
            for gr in v.groups:
                n = groups[gr.group]
                if n.endswith("_" + hand) and n[:-3] in names and n[-3] in "123":
                    weight[n[:-3]][v.index] += gr.weight
        ev = obj.evaluated_get(dg)
        me = ev.to_mesh()
        co = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get("co", co)
        m = np.array(ev.matrix_world)
        co = co.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3]
        ev.to_mesh_clear()
        if len(co) != len(obj.data.vertices):
            continue
        for f in names:
            pts[f].append(co[weight[f] > 0.5])
    ok_all = True
    for f in names:
        p = np.concatenate(pts[f]) if pts[f] else np.zeros((0, 3))
        if len(p) == 0:
            result[f] = {"points": 0, "holds": None}
            continue
        d = p - g
        r = np.linalg.norm(d - np.outer(d @ ax, ax), axis=1)
        near = np.abs(d @ ax) < radius * 6  # the part of the handle the hand is at
        gap = float((r[near].min() - radius) / radius) if near.any() else float("inf")
        inside = float((r[near] < radius * 0.9).mean()) if near.any() else 0.0
        holds = gap < 0.35 and inside < 0.02
        result[f] = {"points": int(len(p)), "gap_radii": round(gap, 2), "inside": round(inside, 3), "holds": holds}
        ok_all &= holds if f != "little" or result[f]["points"] else True
        print(f"  check {f}: {len(p)} points, skin {gap:+.2f} handle radii off the handle, {inside:.1%} inside: "
              f"{'PASS' if holds else 'FAIL'}", flush=True)
    result["all_hold"] = bool(ok_all)
    print(f"grip check: {'PASS' if ok_all else 'FAIL'}", flush=True)
    json.dump(result, open(out + "_grip.json", "w"), indent=1)
    return result


def keyframe_all(rig, frame):
    for pb in rig.pose.bones:
        pb.rotation_mode = "QUATERNION"
        pb.keyframe_insert("rotation_quaternion", frame=frame)
        pb.keyframe_insert("location", frame=frame)


def point_bone(rig, name, direction):
    """Turns a bone (at its head) so that it points in a world direction. Targets, not turns by so many degrees: the
    knight's rest pose is an A pose (arms already 50 degrees down), so 'lower the arm by 55' swung it across his belly."""
    head, tail = world(rig, name), world(rig, name, "tail")
    q = (tail - head).normalized().rotation_difference(Vector(direction).normalized())
    if q.angle > 1e-4:
        turn_bone(rig, name, q.axis, math.degrees(q.angle))


# arm directions (Blender: x = the character's left, -y = its front, z = up); x is flipped for the right arm
POSES = {
    "hold": {"upper_arm": (0.30, -0.25, -0.92), "lower_arm": (0.15, -0.75, -0.65)},   # down at the side, forearm forward
    "raise": {"upper_arm": (0.20, -0.55, 0.81), "lower_arm": (0.10, -0.30, 0.95)},     # up above the head, ready to chop
    "chop": {"upper_arm": (0.20, -0.90, -0.40), "lower_arm": (0.10, -0.95, -0.30)},    # struck down in front
}


def pose_arm(rig, hand, pose):
    side = 1 if hand == "l" else -1
    for bone, (x, y, z) in POSES[pose].items():
        point_bone(rig, f"{bone}_{hand}", (side * x, y, z))


def arm_down(rig, hand):
    pose_arm(rig, hand, "hold")
    roll_hand_upright(rig, hand)


def roll_hand_upright(rig, hand):
    """The wrist turned (about the forearm) so the knuckles' line points up as far as it can: the handle, which lies
    along it, then stands upright with the head up, as one holds an axe at one's side (without it the knight held his
    axe flat across his belly). The forearm's twist bone takes half of that turn (rig_from_glb.py)."""
    along = (world(rig, "hand_" + hand) - world(rig, "lower_arm_" + hand)).normalized()
    pivot = world(rig, "hand_" + hand)
    def up():
        return (world(rig, "index1_" + hand) - world(rig, "little1_" + hand)).normalized().z
    best, best_angle = up(), 0
    for angle in range(-170, 180, 10):
        turn_bone(rig, "hand_" + hand, along, angle, pivot)
        if up() > best:
            best, best_angle = up(), angle
        turn_bone(rig, "hand_" + hand, along, -angle, pivot)
    best_angle = max(-90, min(90, best_angle))  # a wrist turns about 90 degrees either way, not more
    turn_bone(rig, "hand_" + hand, along, best_angle, pivot)
    print(f"wrist turned {best_angle} degrees: knuckles {up():+.2f} up", flush=True)


def render(path, centre, span, direction, res=700, cut=None):
    """cut: only what lies within `cut` of the centre (along the view) is drawn: a close-up through a wide sleeve."""
    s = bpy.context.scene
    s.render.engine = "BLENDER_EEVEE_NEXT"
    s.render.resolution_x = s.render.resolution_y = res
    if not s.world:
        s.world = bpy.data.worlds.new("w")
    s.world.use_nodes = True
    s.world.node_tree.nodes["Background"].inputs[1].default_value = 1.0
    if "sun" not in bpy.data.objects:
        sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
        sun.data.energy = 3
        s.collection.objects.link(sun)
        sun.rotation_euler = (math.radians(50), 0, math.radians(-30))
    cam = bpy.data.objects.get("cam") or bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    if cam.name not in s.collection.objects:
        s.collection.objects.link(cam)
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = span
    d = direction.normalized()
    cam.location = centre + d * span * 4
    cam.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()
    cam.data.clip_start, cam.data.clip_end = (span * 4 - cut, span * 4 + cut) if cut else (0.01, span * 20)
    s.camera = cam
    s.render.filepath = path
    bpy.ops.render.render(write_still=True)


def main():
    (character, item_path, out), o = args()
    rig, body = load_character(os.path.abspath(character))
    before = set(bpy.context.scene.objects)
    bpy.ops.import_scene.gltf(filepath=os.path.abspath(item_path))
    new = [ob for ob in bpy.context.scene.objects if ob not in before]
    parts = [ob for ob in new if ob.type == "MESH"]
    bpy.ops.object.select_all(action="DESELECT")
    for ob in parts:
        ob.select_set(True)
    bpy.context.view_layer.objects.active = parts[0]
    if len(parts) > 1:
        bpy.ops.object.join()
    item = bpy.context.view_layer.objects.active
    for ob in new:
        if ob.type != "MESH" and ob.name in bpy.context.scene.objects:
            bpy.data.objects.remove(ob)
    item.parent = None
    item.name = "item"
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    hand = o["hand"]

    if o["pose"] == "down":
        arm_down(rig, hand)
    mesh_follows(body, False)
    open_hand(rig, hand)
    grip_point, axis, radius, palm = place_item(rig, body, item, hand, o["size"], o["grip"])
    thick = finger_thickness(rig, body, hand)
    close_fingers(rig, hand, grip_point, axis, radius, palm, thick)
    close_thumb(rig, hand, grip_point, axis, radius, thick=thick)
    closer = 0.0
    for _ in range(3):  # the thumb's skin onto the handle: its bones' goal moved by the gap the skin still shows
        mesh_follows(body, True)
        check = grip_check(rig, body, hand, grip_point, axis, radius, out)
        t = check["thumb"]
        if not t.get("points") or t["holds"] or t["inside"] >= 0.02:
            break
        closer += t["gap_radii"] * radius
        mesh_follows(body, False)
        close_thumb(rig, hand, grip_point, axis, radius, closer, thick)
    else:
        mesh_follows(body, True)
        grip_check(rig, body, hand, grip_point, axis, radius, out)
    mesh_follows(body, True)
    bpy.context.scene.frame_set(1)
    keyframe_all(rig, 1)

    # pictures (in clay, to see the shapes): the grip close up from 3 sides, the whole body
    clay = bpy.data.materials.new("clay")
    clay.use_nodes = True
    looks = {ob.name: list(ob.data.materials) for ob in body + [item]}
    for ob in body + [item]:
        ob.data.materials.clear()
        ob.data.materials.append(clay)
    hand_at = item.matrix_world.translation.copy() if False else grip_point
    span = (world(rig, "middle1_" + hand) - world(rig, "hand_" + hand)).length * 4.5
    rig.hide_render = True
    side = Vector((1 if hand == "l" else -1, 0, 0))
    for k, d in enumerate((Vector((0, -1, 0)), side + Vector((0, -0.3, 0.1)), side * 0.3 + Vector((0, 0, -1)))):
        render(f"{out}_hand_{k}.png", hand_at, span, d, cut=span * 0.45)
    pts = np.concatenate([[tuple(ob.matrix_world @ v.co) for v in ob.data.vertices] for ob in body + [item]])
    lo, hi = np.percentile(pts[:, 2], 0.5), np.percentile(pts[:, 2], 99.5)  # the points, not the objects' boxes
    mid = Vector((float(np.median(pts[:, 0])), float(np.median(pts[:, 1])), (lo + hi) / 2))
    for k, d in enumerate((Vector((0, -1, 0)), Vector((-0.7, -0.7, 0.2)))):
        render(f"{out}_body_{k}.png", mid, (hi - lo) * 1.25, d)

    if o["swing"]:  # raise and chop: the shoulder lifts the arm forward and up, then down fast
        s = bpy.context.scene
        s.frame_start, s.frame_end = 1, 48
        keyframe_all(rig, 1)
        s.frame_set(16)
        pose_arm(rig, hand, "raise")
        keyframe_all(rig, 16)
        s.frame_set(26)
        pose_arm(rig, hand, "chop")
        keyframe_all(rig, 26)
        s.frame_set(16)
        render(f"{out}_swing_up.png", mid, (hi - lo) * 2.2, Vector((-1 if hand == "r" else 1, 0, 0)))
        s.frame_set(26)
        render(f"{out}_swing_down.png", mid, (hi - lo) * 2.2, Vector((-1 if hand == "r" else 1, 0, 0)))
        s.frame_set(1)
    rig.hide_render = False
    for ob in body + [item]:  # their own colours back
        ob.data.materials.clear()
        for m in looks[ob.name]:
            ob.data.materials.append(m)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(out + ".blend"))
    bpy.ops.export_scene.gltf(filepath=os.path.abspath(out + ".glb"), export_format="GLB", export_skins=True,
                              export_animations=bool(o["swing"]))
    print("held:", out + ".blend", flush=True)


main()
