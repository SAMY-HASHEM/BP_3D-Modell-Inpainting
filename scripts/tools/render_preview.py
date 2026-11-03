"""Render a .glb model from 4 sides (0, 90, 180, 270 degrees) with Blender, on the CPU only.

For checking results while the GPU is busy or must stay free (e.g. the night tests): Cycles on the CPU,
a few samples, grey background, soft light. Writes <out>_0.png ... <out>_3.png; run_tests.py puts them
side by side.

Usage:
    blender -b --python scripts/tools/render_preview.py -- outputs/model.glb /tmp/model_preview [--size 384]
    blender -b --python scripts/tools/render_preview.py -- outputs/model.glb /tmp/model_preview --closeup   # upper part, front
    # a box of the model (glTF coordinates, Y up, as in refine.json) from the front, 45° left / right and behind:
    blender -b --python scripts/tools/render_preview.py -- model.glb /tmp/head --focus X0 Y0 Z0 X1 Y1 Z1 [--samples 32]
    blender -b --python scripts/tools/render_preview.py -- model.glb /tmp/model --clay   # shape only, no colours
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
if "--clay" in args:  # without colours, like a marble statue: for judging the shape alone
    clay = bpy.data.materials.new("clay")
    clay.diffuse_color = (0.72, 0.72, 0.70, 1)
    for o in meshes:
        o.data.materials.clear()
        o.data.materials.append(clay)
corners = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
low = Vector((min(c.x for c in corners), min(c.y for c in corners), min(c.z for c in corners)))
high = Vector((max(c.x for c in corners), max(c.y for c in corners), max(c.z for c in corners)))
center, radius = (low + high) / 2, (high - low).length / 2

scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"  # never the GPU
scene.cycles.samples = int(args[args.index("--samples") + 1]) if "--samples" in args else 16
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
if "--focus" in args:  # one box of the model, the same camera for every model: before / after comparisons
    k = args.index("--focus")
    a, b = Vector([float(v) for v in args[k + 1:k + 4]]), Vector([float(v) for v in args[k + 4:k + 7]])
    a, b = Vector((a.x, -a.z, a.y)), Vector((b.x, -b.z, b.y))  # glTF (Y up) -> Blender (Z up), as the importer does
    target, span = (a + b) / 2, max((b - a).x.__abs__(), (b - a).z.__abs__()) * 0.62
    scene.render.resolution_x = scene.render.resolution_y = 2 * size
    for name, angle in (("front", 0), ("left", -45), ("right", 45), ("back", 180)):
        turn = math.radians(angle)
        direction = Vector((math.sin(turn), -math.cos(turn), 0.08))
        camera.location = target + direction.normalized() * span / math.tan(camera.data.angle / 2)
        camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
        scene.render.filepath = f"{out}_{name}.png"
        bpy.ops.render.render(write_still=True)
    print(f"Rendered {out}_front / _left / _right / _back.png", flush=True)
    sys.exit(0)
if "--closeup" in args:  # only a close-up of the upper part from the front, for rating fine detail
    height = high.z - low.z
    target = Vector((center.x, center.y, high.z - 0.2 * height))
    span = 0.38 * max(height, 1e-6)
    camera.location = target + Vector((0, -span / math.tan(camera.data.angle / 2), 0))
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
    scene.render.resolution_x = scene.render.resolution_y = 2 * size
    scene.render.filepath = f"{out}_closeup.png"
    bpy.ops.render.render(write_still=True)
    print(f"Rendered {out}_closeup.png", flush=True)
    sys.exit(0)
for k, angle in enumerate((0, 90, 180, 270)):
    a = math.radians(angle)
    camera.location = center + Vector((math.sin(a) * distance, -math.cos(a) * distance, radius * 0.25))
    camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
    scene.render.filepath = f"{out}_{k}.png"
    bpy.ops.render.render(write_still=True)
print(f"Rendered {out}_0..3.png", flush=True)
