"""A humanoid model's joints found on pictures of it: the model rendered from 12 sides, GEM-X's 77-point finder (the
same that reads the person in a video for motion_track.py) on every picture, each joint put where the pictures agree.

For models in a pose the rig head (rig_head.py) has not learnt: it learnt from characters standing in a T or A pose.
Mai (Nov 22, a downloaded model, hands on her hips, elbows back) got elbows and a shoulder outside her body, and her
arms did not move with the bones. rig_from_glb.py --joints views uses this.

    blender -b --python scripts/rigging/rig_views.py -- render model.glb <folder>     # 12 pictures and views.json
    conda/envs/gemx/bin/python scripts/rigging/rig_views.py detect <folder>          # points.npz and joints.npz

joints.npz: gltf_joints (22, 3) and gltf_fingers (30, 3) in glTF axes (Y up, front +Z), in rig_data.py's order, and
how far the pictures disagree per joint (spread, in model heights). check.jpg: the points on the pictures.
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(HERE))
GEMX = os.path.join(PROJECT, "local_ai_models", "motion", "GEM-X")
SIZE = 1024

# our bones' heads (rig_data.py's BONES, FINGERS) <- GEM-X's 77 points (SOMA's joints). SOMA's Index1 lies in the palm:
# our index1 is its Index2 (as motion_to_rig.py's FOLLOW).
BODY = {"hips": "Hips", "spine": "Spine1", "chest": "Spine2", "upper_chest": "Chest", "neck": "Neck1", "head": "Head"}
for _s, _side in (("l", "Left"), ("r", "Right")):
    BODY.update({f"shoulder_{_s}": f"{_side}Shoulder", f"upper_arm_{_s}": f"{_side}Arm", f"lower_arm_{_s}": f"{_side}ForeArm",
                 f"hand_{_s}": f"{_side}Hand", f"upper_leg_{_s}": f"{_side}Leg", f"lower_leg_{_s}": f"{_side}Shin",
                 f"foot_{_s}": f"{_side}Foot", f"toe_{_s}": f"{_side}ToeBase"})
FINGER = {}
for _s, _side in (("l", "Left"), ("r", "Right")):
    for _ours, _theirs, _offset in (("thumb", "Thumb", 0), ("index", "Index", 1), ("middle", "Middle", 1),
                                    ("ring", "Ring", 1), ("little", "Pinky", 1)):
        for _k in (1, 2, 3):
            FINGER[f"{_ours}{_k}_{_s}"] = f"{_side}Hand{_theirs}{_k + _offset}"


def render(model, folder):
    """In Blender: the model from 8 sides around it and 4 from above, orthographic, in its own colours."""
    import bpy
    from mathutils import Vector
    os.makedirs(folder, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=os.path.abspath(model))
    scene = bpy.context.scene
    meshes = [o for o in scene.objects if o.type == "MESH" and len(o.data.vertices) >= 100]
    for o in scene.objects:
        if o.type == "MESH" and o not in meshes:  # the importer's bone shape
            o.hide_render = True
    corners = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
    low = Vector([min(c[i] for c in corners) for i in range(3)])
    high = Vector([max(c[i] for c in corners) for i in range(3)])
    centre, span = (low + high) / 2, max(high - low)
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "TEXTURE"
    scene.render.film_transparent = True
    scene.render.resolution_x = scene.render.resolution_y = SIZE
    camera = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = span * 1.15
    camera.data.clip_end = span * 20
    scene.collection.objects.link(camera)
    scene.camera = camera
    views = []
    for elevation, step in ((0, 45), (30, 90)):
        for azimuth in range(0, 360, step):
            a, e = math.radians(azimuth), math.radians(elevation)
            d = Vector((math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e)))  # 0: from the front (-Y)
            camera.location = centre + d * span * 4
            camera.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()
            bpy.context.view_layer.update()
            name = f"view_{len(views):02d}_az{azimuth:03d}_el{elevation:02d}.png"
            scene.render.filepath = os.path.join(folder, name)
            bpy.ops.render.render(write_still=True)
            m = camera.matrix_world
            views.append({"file": name, "location": list(camera.location), "right": list(m.col[0].xyz),
                          "up": list(m.col[1].xyz), "scale": span * 1.15, "size": SIZE, "azimuth": azimuth,
                          "elevation": elevation})
    json.dump({"views": views, "height": float(high.z - low.z)}, open(os.path.join(folder, "views.json"), "w"), indent=1)
    print(f"rendered {len(views)} views: {folder}", flush=True)


def detect(folder):
    """In conda/envs/gemx: the 77 points on every picture, then each joint where the pictures agree."""
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    sys.path.insert(0, GEMX)
    from gem.common.pose2d.native77 import load_native77_detector
    from gem.motion.frontend.joint_sets import SOMA77_JOINT_NAMES
    info = json.load(open(os.path.join(folder, "views.json")))
    frames, boxes = [], []
    for v in info["views"]:
        rgba = np.asarray(Image.open(os.path.join(folder, v["file"])).convert("RGBA")).astype(np.float32)
        alpha = rgba[..., 3:] / 255
        frames.append((rgba[..., :3] * alpha + 128 * (1 - alpha)).astype(np.uint8))  # on grey
        ys, xs = np.nonzero(alpha[..., 0] > .5)
        w, h = xs.max() - xs.min(), ys.max() - ys.min()
        boxes.append([(xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2, max(h, w / .75) * 1.2])  # it reads 3:4 crops
    detector = load_native77_detector(os.path.join(GEMX, "weights", "vitpose.pth"),
                                      release_id="vitpose77-dinov3-huge-metrosim-256x192-v1",
                                      device="cuda" if torch.cuda.is_available() else "cpu", batch_size=4)
    points = detector.predict(frames, torch.tensor(boxes, dtype=torch.float32)).numpy()  # (views, 77, x y confidence)
    np.savez(os.path.join(folder, "points.npz"), points=points)

    names = list(SOMA77_JOINT_NAMES)
    height = info["height"]
    found, spread = {}, {}
    for j, name in enumerate(names):
        rows, values, weights, which = [], [], [], []
        for k, v in enumerate(info["views"]):
            px, py, sure = points[k, j]
            loc, right, up = (np.array(v[key]) for key in ("location", "right", "up"))
            x = (px - v["size"] / 2) / v["size"] * v["scale"]
            y = (v["size"] / 2 - py) / v["size"] * v["scale"]
            rows += [right, up]
            values += [x + right @ loc, y + up @ loc]
            weights += [max(sure, 0)] * 2
            which += [k, k]
        A, b, w, which = np.array(rows), np.array(values), np.array(weights), np.array(which)
        use = w > .25
        for _ in range(3):  # the pictures that disagree (a hidden hand, left and right mixed up from behind) dropped
            if len(set(which[use])) < 2:
                use = w > 0
            P = np.linalg.lstsq(A[use] * np.sqrt(w[use])[:, None], b[use] * np.sqrt(w[use]), rcond=None)[0]
            off = np.abs(A @ P - b)
            per_view = np.array([np.hypot(*off[which == k]) for k in range(len(info["views"]))])
            limit = max(.015 * height, 2.5 * np.median(per_view[np.unique(which[use])]))
            keep = np.isin(which, np.nonzero(per_view < limit)[0]) & (w > .15)
            if len(set(which[keep])) < 2 or np.array_equal(keep, use):
                break
            use = keep
        found[name] = P
        spread[name] = float(np.median(per_view[np.unique(which[use])]) / height)
    to_gltf = lambda p: [p[0], p[2], -p[1]]  # Blender (Z up) -> glTF (Y up)
    sys.path.insert(0, HERE)
    from rig_data import BONES, FINGERS
    np.savez(os.path.join(folder, "joints.npz"),
             gltf_joints=np.array([to_gltf(found[BODY[n]]) for n in BONES]),
             gltf_fingers=np.array([to_gltf(found[FINGER[n]]) for n in FINGERS]),
             spread=np.array([spread[BODY[n]] for n in BONES] + [spread[FINGER[n]] for n in FINGERS]))
    worst = sorted(((spread[BODY[n]], n) for n in BONES), reverse=True)[:4]
    print("joints from the pictures: " + ", ".join(f"{n} {s * 100:.1f} %" for s, n in worst)
          + " of the height apart (the worst four)", flush=True)

    # check.jpg: the body's points on four of the pictures
    sheet = Image.new("RGB", (4 * 512, 512), (40, 40, 44))
    for i, k in enumerate((0, 2, 4, 6)):
        picture = Image.fromarray(frames[k]).resize((512, 512))
        draw = ImageDraw.Draw(picture)
        for n in list(BODY.values()):
            px, py, sure = points[k, names.index(n)] / [2, 2, 1]
            colour = (255, 220, 0) if "Left" in n else (0, 220, 255) if "Right" in n else (255, 80, 80)
            if sure > .25:
                draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=colour)
        sheet.paste(picture, (i * 512, 0))
    sheet.save(os.path.join(folder, "check.jpg"), quality=90)


if __name__ == "__main__":
    if "--" in sys.argv:  # in Blender
        what, *rest = sys.argv[sys.argv.index("--") + 1:]
        render(*rest)
    elif sys.argv[1] == "detect":
        detect(sys.argv[2])
