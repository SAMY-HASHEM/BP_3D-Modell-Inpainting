"""Four straight views of a model (orthographic, transparent background), for inpainting by picture: because the
camera has no perspective, every pixel of these pictures can be turned back into a place on the model.

    blender -b --python scripts/tools/render_views.py -- model.glb out_folder [--size 1024] [--samples 32] [--frame views.json] [--turn 45]

Writes view_0.png ... view_3.png and views.json (see make_edit.py).
"""
import json
import math
import os
import sys

import bpy
from mathutils import Vector

args = sys.argv[sys.argv.index("--") + 1:]
model, out = args[0], args[1]
size = int(args[args.index("--size") + 1]) if "--size" in args else 1024
samples = int(args[args.index("--samples") + 1]) if "--samples" in args else 32
os.makedirs(out, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=model)
meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
corners = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
low = Vector((min(c.x for c in corners), min(c.y for c in corners), min(c.z for c in corners)))
high = Vector((max(c.x for c in corners), max(c.y for c in corners), max(c.z for c in corners)))
center = (low + high) / 2
span = max(high - low) * 1.4  # free space around the model: something new may be put on it
if "--frame" in args:  # the frame of an earlier views.json: pictures of a changed model stay comparable
    frame = json.load(open(args[args.index("--frame") + 1]))
    center, span = Vector(frame["center"]), frame["span"]

scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"  # never the GPU
scene.cycles.samples = samples
scene.cycles.use_denoising = True
scene.render.resolution_x = scene.render.resolution_y = size
scene.render.film_transparent = True
world = bpy.data.worlds.new("light")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)
world.node_tree.nodes["Background"].inputs[1].default_value = 0.9
scene.world = world
sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
sun.data.energy = 2.0
sun.rotation_euler = (math.radians(40), 0, math.radians(30))
scene.collection.objects.link(sun)
camera = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
camera.data.type = "ORTHO"
camera.data.ortho_scale = span
camera.data.clip_end = span * 10
scene.collection.objects.link(camera)
scene.camera = camera

turn = float(args[args.index("--turn") + 1]) if "--turn" in args else None  # also the sides in between
views = []
for k in range(4 if turn is None else 8):
    a = math.radians(90 * (k % 4) + (turn if k >= 4 else 0))
    toward = Vector((math.sin(a), -math.cos(a), 0))  # from the model towards the camera
    camera.location = center + toward * span * 2
    camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
    sun.rotation_euler = (math.radians(40), 0, a + math.radians(30))  # the same light on every side
    scene.render.filepath = os.path.join(out, f"view_{k}.png")
    bpy.ops.render.render(write_still=True)
    right = Vector((math.cos(a), math.sin(a), 0))
    views.append({"right": list(right), "toward": list(toward)})
json.dump({"center": list(center), "span": span, "size": size, "low": list(low), "high": list(high), "views": views},
          open(os.path.join(out, "views.json"), "w"), indent=1)
