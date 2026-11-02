"""Render a .glb model from 4 sides (0, 90, 180, 270 degrees) with Blender, on the CPU only.

For checking results while the GPU is busy or must stay free (e.g. the night tests): Cycles on the CPU,
a few samples, grey background, soft light. Writes <out>_0.png ... <out>_3.png; run_tests.py puts them
side by side.

Usage:
    blender -b --python scripts/tools/render_preview.py -- outputs/model.glb /tmp/model_preview [--size 384]
"""
import math
import sys

import bpy
from mathutils import Vector

args = sys.argv[sys.argv.index("--") + 1:]
model, out = args[0], args[1]
size = int(args[args.index("--size") + 1]) if "--size" in args else 384

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=model)
meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
corners = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
low = Vector((min(c.x for c in corners), min(c.y for c in corners), min(c.z for c in corners)))
high = Vector((max(c.x for c in corners), max(c.y for c in corners), max(c.z for c in corners)))
center, radius = (low + high) / 2, (high - low).length / 2

scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"  # never the GPU
scene.cycles.samples = 16
scene.cycles.use_denoising = True
scene.render.resolution_x = scene.render.resolution_y = size
scene.render.film_transparent = False
world = bpy.data.worlds.new("grey")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.25, 0.25, 0.27, 1)
world.node_tree.nodes["Background"].inputs[1].default_value = 0.8
scene.world = world
sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
sun.data.energy = 3.0
sun.rotation_euler = (math.radians(40), 0, math.radians(30))
scene.collection.objects.link(sun)

camera = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
camera.data.lens = 50
scene.collection.objects.link(camera)
scene.camera = camera
distance = radius / math.tan(camera.data.angle / 2) * 1.05
for k, angle in enumerate((0, 90, 180, 270)):
    a = math.radians(angle)
    camera.location = center + Vector((math.sin(a) * distance, -math.cos(a) * distance, radius * 0.25))
    camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
    scene.render.filepath = f"{out}_{k}.png"
    bpy.ops.render.render(write_still=True)
print(f"Rendered {out}_0..3.png", flush=True)
