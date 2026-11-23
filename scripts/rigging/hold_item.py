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
  the fingers   each finger's three joints bent in turn, a few degrees at a time, until that segment touches the
                handle (its tip within the handle's radius + a finger's thickness of the handle's axis) or reaches
                its limit; the thumb wraps from the other side
  --swing       an animation to check the hold: the arm down, raised, a chop (48 frames)

Writes <out>.blend (press Space with --swing), <out>.glb (the character, its armature and the item on the hand) and
<out>_hand_<k>.png, <out>_body_<k>.png (the grip close up from 3 sides, the whole body).
The rigged .blend keeps the twist bones' constraints and preserve volume (rig_from_glb.py); a .glb is set up again.
"""
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector
from mathutils.bvhtree import BVHTree

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
    target = K + f * seg * 0.35 + n * (radius + seg * 0.28)
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
LIMIT = (95, 110, 85)         # and at most (degrees); the thumb: its own
THUMB_CURL, THUMB_LIMIT = (0.6, 1.0, 0.9), (50, 70, 80)


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


def close_fingers(rig, hand, grip_point, axis, radius, palm):
    """Each finger curls all its joints together (CURL), two degrees a step; a segment that touches the handle (its
    middle or its end within the handle's radius + a finger's thickness of the handle's axis) stops its own joint and
    those before it, the joints after it go on, so the finger wraps the handle; each joint stops at its LIMIT.
    The way to bend: the one that brings the fingertip nearer the handle."""
    def to_axis(p):
        d = p - grip_point
        return (d - axis * d.dot(axis)).length
    for finger in FINGER_NAMES + ("thumb",):
        names = [f"{finger}{k}_{hand}" for k in (1, 2, 3)]
        if any(nm not in rig.pose.bones for nm in names):
            continue
        curl, limit = (THUMB_CURL, THUMB_LIMIT) if finger == "thumb" else (CURL, LIMIT)
        seg = (world(rig, names[0]) - world(rig, names[1])).length
        def inside_handle(chain):  # any later segment's middle or end inside the handle (with a finger's half
            for nm in chain:           # thickness): the finger would go through it
                h, e = world(rig, nm), world(rig, nm, "tail")
                if min(to_axis(e), to_axis((h + e) / 2)) < radius + max(seg * 0.12, radius * 0.1):
                    return True
            return False
        if finger == "thumb":  # first across the palm (about the hand's length), towards the handle, at most 60 deg
            along = (world(rig, "middle1_" + hand) - world(rig, "hand_" + hand)).normalized()
            base = world(rig, "hand_" + hand)
            best, way = to_axis(world(rig, names[2], "tail")), 0
            for sgn in (1, -1):
                turn_bone(rig, names[0], along, sgn * 10, pivot=world(rig, names[0]))
                d = to_axis(world(rig, names[2], "tail"))
                turn_bone(rig, names[0], along, -sgn * 10, pivot=world(rig, names[0]))
                if d < best:
                    best, way = d, sgn
            swung = 0
            while way and swung < 60:
                d0 = to_axis(world(rig, names[2], "tail"))
                turn_bone(rig, names[0], along, way * 3, pivot=world(rig, names[0]))
                swung += 3
                if to_axis(world(rig, names[2], "tail")) > d0 or inside_handle(names):
                    turn_bone(rig, names[0], along, -way * 3, pivot=world(rig, names[0]))
                    swung -= 3
                    break
            print(f"  thumb: across the palm {swung} degrees", flush=True)
        touch = radius + max(seg * 0.3, radius * 0.2)
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
    open_hand(rig, hand)
    grip_point, axis, radius, palm = place_item(rig, body, item, hand, o["size"], o["grip"])
    close_fingers(rig, hand, grip_point, axis, radius, palm)
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
