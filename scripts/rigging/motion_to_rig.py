"""A tracked motion (motion_track.py: GEM-X's 77 joints per frame) put onto a rigged model (rig_from_glb.py: 22 body
bones and 30 finger bones), as an animation.

The two skeletons differ in size, proportions and rest pose, so no rotations are copied. Every bone is turned so that it
points where the same bone of the tracked person points, frame by frame, parents first:
  hips, upper chest, head, hands   their whole turn from two directions (up the bone, and across: hip to hip, shoulder
                                   to shoulder, eye to eye, index finger to little finger), so the body can face away
                                   and the wrists can twist
  every other bone                 the smallest turn that points it the same way, from where its parent has put it
The hips also move: the tracked hips' way through the room, scaled to the model (by the legs' length), on its floor.

    blender -b --python scripts/rigging/motion_to_rig.py -- rigged.glb joints.npz out_folder [--start 0] [--frames 300]
        [--keep-colours] [--no-frames]

The model gets one even skin colour (its own colours are often dark: the moving body is hard to see); --keep-colours
keeps them. The pictures look from where the video's camera stood, when motion_track.py knew it.

Writes <out>/animated.blend (press Space), <out>/animated.glb (any program that reads glTF) and <out>/frames/ (a
picture of every frame from the front, for motion_from_video.py's side-by-side video).
"""
import math
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

args = sys.argv[sys.argv.index("--") + 1:]
glb, joints_path, out = args[:3]
option = lambda name, default: type(default)(args[args.index(name) + 1]) if name in args else default
first, count = option("--start", 0), option("--frames", 0)
os.makedirs(out, exist_ok=True)

data = np.load(joints_path)
names = [str(n) for n in data["names"]]
J = {n: k for k, n in enumerate(names)}
P = data["joints"].astype(np.float64)
P = P[first:first + count] if count else P[first:]
P = np.stack([P[..., 0], -P[..., 2], P[..., 1]], -1)  # GEM-X's world (Y up) -> Blender's (Z up)
fps = float(data["fps"])

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=os.path.abspath(glb))
rig = next(o for o in bpy.context.scene.objects if o.type == "ARMATURE")
if any(b.name.endswith(("twist_l", "twist_r")) for b in rig.data.bones):  # rig_from_glb.py's twist bones (Nov 22):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))       # their constraints and preserve volume
    from rig_from_glb import better_skinning                               # are not kept in a glb
    better_skinning(rig)
for o in list(bpy.context.scene.objects):  # the importer's bone shape (an icosphere) is not part of the model
    if o.type == "MESH" and o.parent is None and len(o.data.vertices) < 100:
        bpy.data.objects.remove(o)
if "--keep-colours" not in args:  # one even skin colour: the moving body is easier to see than in its own dark colours
    skin = bpy.data.materials.new("skin")
    linear = tuple(((c / 255 + .055) / 1.055) ** 2.4 for c in (230, 180, 150))  # a warm light skin tone, sRGB -> linear
    skin.diffuse_color = (*linear, 1)  # what Blender's solid view and Workbench show
    skin.roughness = .6
    skin.use_nodes = True
    bsdf = next(n for n in skin.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = (*linear, 1)
    bsdf.inputs["Roughness"].default_value = .55
    if "Subsurface Weight" in bsdf.inputs:  # light a little under the skin (Eevee, Cycles)
        bsdf.inputs["Subsurface Weight"].default_value = .1
    for o in bpy.context.scene.objects:
        if o.type == "MESH":
            o.data.materials.clear()
            o.data.materials.append(skin)
to_rig = np.array(rig.matrix_world.inverted())
P = P @ to_rig[:3, :3].T + to_rig[:3, 3]
bones = rig.data.bones
rest = {b.name: np.array(b.matrix_local) for b in bones}
head = {b.name: np.array(b.head_local) for b in bones}
tail = {b.name: np.array(b.tail_local) for b in bones}

# ---------------------------------------------------------------- which tracked joints each bone follows
FOLLOW = {}  # bone: (tracked from, tracked to)  or  ("two", (up from, up to), (across a, across b), (rest across a, rest across b))
FOLLOW["hips"] = ("two", ("Hips", "Spine1"), ("LeftLeg", "RightLeg"), ("upper_leg_l", "upper_leg_r"))
FOLLOW["spine"] = ("Spine1", "Spine2")
FOLLOW["chest"] = ("Spine2", "Chest")
FOLLOW["upper_chest"] = ("two", ("Chest", "Neck1"), ("LeftShoulder", "RightShoulder"), ("shoulder_l", "shoulder_r"))
FOLLOW["neck"] = ("Neck1", "Head")
FOLLOW["head"] = ("two", ("Head", "HeadEnd"), ("LeftEye", "RightEye"), None)  # across: the model's own left-right
for s, side in (("l", "Left"), ("r", "Right")):
    FOLLOW[f"shoulder_{s}"] = (f"{side}Shoulder", f"{side}Arm")
    FOLLOW[f"upper_arm_{s}"] = (f"{side}Arm", f"{side}ForeArm")
    FOLLOW[f"lower_arm_{s}"] = (f"{side}ForeArm", f"{side}Hand")
    FOLLOW[f"hand_{s}"] = ("two", (f"{side}Hand", f"{side}HandMiddle2"), (f"{side}HandIndex2", f"{side}HandPinky2"),
                           (f"index1_{s}", f"little1_{s}"))
    FOLLOW[f"upper_leg_{s}"] = (f"{side}Leg", f"{side}Shin")
    FOLLOW[f"lower_leg_{s}"] = (f"{side}Shin", f"{side}Foot")
    FOLLOW[f"foot_{s}"] = (f"{side}Foot", f"{side}ToeBase")
    FOLLOW[f"toe_{s}"] = (f"{side}ToeBase", f"{side}ToeEnd")
    for ours, theirs, offset in (("thumb", "Thumb", 0), ("index", "Index", 1), ("middle", "Middle", 1), ("ring", "Ring", 1),
                                 ("little", "Pinky", 1)):  # SOMA's Index1 is the bone in the palm: our index1 is its Index2
        chain = [f"{side}Hand{theirs}{k + offset}" for k in (1, 2, 3)] + [f"{side}Hand{theirs}End"]
        for k in (1, 2, 3):
            FOLLOW[f"{ours}{k}_{s}"] = (chain[k - 1], chain[k])
missing = sorted({n for f in FOLLOW.values() for part in (f if f[0] != "two" else (*f[1], *f[2])) for n in
                  (part if isinstance(part, tuple) else (part,)) if n not in J})
assert not missing, f"tracked joints not found: {missing}"
order = []  # parents first
def visit(b):
    order.append(b.name)
    for c in b.children:
        visit(c)
for b in bones:
    if b.parent is None:
        visit(b)


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def frame(up, across):
    e1 = unit(up)
    e2 = unit(across - (across @ e1) * e1)
    return np.stack([e1, e2, np.cross(e1, e2)], 1)


def swing(a, b):
    """The smallest turn that takes direction a to direction b."""
    a, b = unit(a), unit(b)
    axis, c = np.cross(a, b), float(np.clip(a @ b, -1, 1))
    s = np.linalg.norm(axis)
    if s < 1e-9:  # the same way, or the opposite: half a turn about any axis across a
        if c > 0:
            return np.eye(3)
        k = unit(np.cross(a, [1.0, 0, 0]) if abs(a[0]) < .9 else np.cross(a, [0, 1.0, 0]))
        return 2 * np.outer(k, k) - np.eye(3)
    k = axis / s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * K + (1 - c) * K @ K


# ---------------------------------------------------------------- size and floor
legs = lambda points, a, b, c: np.linalg.norm(points[..., a, :] - points[..., b, :], axis=-1) + np.linalg.norm(points[..., b, :] - points[..., c, :], axis=-1)
theirs = np.median(np.concatenate([legs(P, J["LeftLeg"], J["LeftShin"], J["LeftFoot"]), legs(P, J["RightLeg"], J["RightShin"], J["RightFoot"])]))
ours = np.mean([np.linalg.norm(head["upper_leg_l"] - head["lower_leg_l"]) + np.linalg.norm(head["lower_leg_l"] - head["foot_l"]),
                np.linalg.norm(head["upper_leg_r"] - head["lower_leg_r"]) + np.linalg.norm(head["lower_leg_r"] - head["foot_r"])])
scale = ours / theirs
# the floor: where the lowest of the feet, toes, knees and hands is in most frames (motion_track.py made it level), put
# where the model's lowest foot joint (its toe tips) is at rest
contact = [k for k, n in enumerate(names) if n.endswith(("Foot", "ToeBase", "ToeEnd", "Shin", "Hand"))]
their_floor = np.percentile(P[:, contact, 2].min(1), 3)
our_floor = min(min(head[n][2], tail[n][2]) for n in ("foot_l", "foot_r", "toe_l", "toe_r"))
start_xy = P[0, J["Hips"], :2].copy()
print(f"{len(P)} frames at {fps:g} fps; the model is {scale:.3f} times the tracked person's size", flush=True)

# ---------------------------------------------------------------- the pose of every frame
scene = bpy.context.scene
scene.render.fps = int(round(fps))
scene.frame_start, scene.frame_end = 1, len(P)
for pb in rig.pose.bones:
    pb.rotation_mode = "QUATERNION"
before = {}  # each bone's last quaternion: q and -q are the same turn, the one nearer the last frame's is kept (no flips)
for f, Q in enumerate(P, 1):
    posed = {}
    for name in order:
        b = bones[name]
        M = rest[name]
        if b.parent is None:
            where = Q[J["Hips"]]
            spot = np.array([(where[0] - start_xy[0]) * scale + head[name][0], (where[1] - start_xy[1]) * scale + head[name][1],
                             (where[2] - their_floor) * scale + our_floor])
            parent_delta, parent_basis = np.eye(3), None
        else:
            Pp, Rp = posed[b.parent.name], rest[b.parent.name]
            spot = (Pp @ np.linalg.inv(Rp) @ np.append(head[name], 1))[:3]
            parent_delta = (Pp @ np.linalg.inv(Rp))[:3, :3]
        follow = FOLLOW.get(name)
        if follow is None:  # a bone the tracker has nothing for: it goes along with its parent
            delta = parent_delta
        elif follow[0] == "two":
            _, (u0, u1), (a0, a1), rest_across = follow
            up_rest = tail[name] - head[name]
            across_rest = (head[rest_across[0]] - head[rest_across[1]]) if rest_across else np.array([1.0, 0, 0])
            delta = frame(Q[J[u1]] - Q[J[u0]], Q[J[a0]] - Q[J[a1]]) @ frame(up_rest, across_rest).T
        else:
            now = parent_delta @ (tail[name] - head[name])
            delta = swing(now, Q[J[follow[1]]] - Q[J[follow[0]]]) @ parent_delta
        T = np.eye(4)
        T[:3, :3] = delta @ M[:3, :3]
        T[:3, 3] = spot
        posed[name] = T
        local = (np.linalg.inv(M) @ T) if b.parent is None else np.linalg.inv(np.linalg.inv(rest[b.parent.name]) @ M) @ np.linalg.inv(posed[b.parent.name]) @ T
        pb = rig.pose.bones[name]
        q = Matrix(local[:3, :3].tolist()).to_quaternion()
        if name in before and q.dot(before[name]) < 0:
            q.negate()
        before[name] = q.copy()
        pb.rotation_quaternion = q
        pb.keyframe_insert("rotation_quaternion", frame=f)
        if b.parent is None:
            pb.location = Vector(local[:3, 3].tolist())
            pb.keyframe_insert("location", frame=f)
    if f % 60 == 0:
        print(f"  frame {f} of {len(P)}", flush=True)

rig.show_in_front = True
rig.data.display_type = "STICK"
scene.frame_set(1)
bpy.ops.export_scene.gltf(filepath=os.path.join(os.path.abspath(out), "animated.glb"), export_format="GLB", export_animations=True)

# ---------------------------------------------------------------- a floor, a light and the video's camera (in the .blend too)
heights = []  # the model at rest (bound_box would be the posed one)
for o in scene.objects:
    if o.type == "MESH":
        co = np.empty(len(o.data.vertices) * 3)
        o.data.vertices.foreach_get("co", co)
        heights.append((co.reshape(-1, 3) @ np.array(o.matrix_world)[:3, :3].T + np.array(o.matrix_world)[:3, 3])[:, 2])
heights = np.concatenate(heights)
tall = float(heights.max() - heights.min())
floor_z = float(heights.min())  # the soles at rest
look = np.array([0, 1.0, 0])  # from the front
if "view" in data.files:  # the way the video's camera looked (motion_track.py), only its turn about the up axis
    v = np.array(data["view"], dtype=np.float64)
    v = to_rig[:3, :3] @ np.array([v[0], -v[2], v[1]])
    if np.linalg.norm(v[:2]) > .1:
        look = np.array([v[0], v[1], 0]) / np.linalg.norm(v[:2])
right = np.array([look[1], -look[0]])
root = order[0]
every = ((P - np.array([*start_xy, their_floor])) * scale + np.array([head[root][0], head[root][1], our_floor])).reshape(-1, 3)
across, depth = every[:, :2] @ right, every[:, :2] @ look[:2]
left_end, right_end = np.percentile(across, [.5, 99.5])
top = float(np.percentile(every[:, 2], 99.5))
margin = .12 * tall  # the joints are inside the body: room for its flesh, the hair, the hands
W, H = 960, 540  # as wide as the video
wide = max(right_end - left_end + 2 * margin, (top - floor_z + 2 * margin) * W / H)
middle_xy = right[:2] * (left_end + right_end) / 2 + look[:2] * float(np.median(depth))
middle = Vector((float(middle_xy[0]), float(middle_xy[1]), floor_z - margin + wide * H / W / 2))  # the floor near the bottom
floor_mesh = bpy.data.meshes.new("floor")  # a mat: as wide as the picture, a bit deeper than the body
x, y = Vector((*right[:2], 0)) * wide, Vector((*look[:2], 0)) * (.5 * tall)
base = Vector((middle.x, middle.y, floor_z))
floor_mesh.from_pydata([base - x - y, base + x - y, base + x + y, base - x + y], [], [(0, 1, 2, 3)])
floor = bpy.data.objects.new("floor", floor_mesh)
ground = bpy.data.materials.new("floor")
ground.diffuse_color = (.50, .40, .30, 1)  # light wood: a floor seen almost flat gets little light
floor_mesh.materials.append(ground)
scene.collection.objects.link(floor)
cam = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
cam.data.type = "ORTHO"
cam.data.ortho_scale = wide
scene.collection.objects.link(cam)
scene.camera = cam
tilt = math.radians(20)  # a little from above, so the floor shows
aim = Vector((float(look[0]) * math.cos(tilt), float(look[1]) * math.cos(tilt), -math.sin(tilt)))
cam.location = middle - aim * wide * 3
cam.rotation_euler = aim.to_track_quat("-Z", "Y").to_euler()
sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
sun.data.energy = 3
sun.rotation_euler = (math.radians(50), 0, math.radians(25))
scene.collection.objects.link(sun)
world = bpy.data.worlds.new("w")
world.color = (.03, .033, .04)
scene.world = world
rig.hide_render = True
scene.render.resolution_x, scene.render.resolution_y = W, H
scene.render.engine = "BLENDER_WORKBENCH"
scene.display.shading.light = "STUDIO"
scene.display.shading.color_type = "TEXTURE" if "--keep-colours" in args else "MATERIAL"  # its own colours: its texture
scene.display.shading.show_shadows = True  # where the hands and feet touch the floor
scene.display.shading.shadow_intensity = .45
light = np.array([-look[0] * .6 + right[0] * .3, -look[1] * .6 + right[1] * .3, 1])  # from above, on the camera's side
scene.display.light_direction = tuple(float(v) for v in light / np.linalg.norm(light))
scene.render.filepath = os.path.join(os.path.abspath(out), "frames", "f_")
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(os.path.abspath(out), "animated.blend"))

# ---------------------------------------------------------------- a picture of every frame (for the side-by-side video)
if "--no-frames" not in args:
    try:
        bpy.ops.render.render(animation=True)
    except RuntimeError as e:  # no OpenGL here: Cycles on the CPU, a few samples
        print(f"  Workbench can't render here ({e}): Cycles", flush=True)
        scene.render.engine = "CYCLES"
        scene.cycles.device = "CPU"
        scene.cycles.samples = 8
        scene.cycles.use_denoising = True
        bpy.ops.render.render(animation=True)
print(f"Result: {os.path.join(out, 'animated.blend')}", flush=True)
