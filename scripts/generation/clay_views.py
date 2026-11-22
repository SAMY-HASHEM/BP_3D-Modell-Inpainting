"""Grey ("clay") pictures of models, to compare their shapes without colours: front, side (the character's left) and
back of each model, every model in a row of one sheet.

    blender -b --python scripts/generation/clay_views.py -- out.jpg A.glb B.glb      # renders out_<k>_<view>.png
    python scripts/generation/clay_views.py out.jpg "TRELLIS.2" "TRELLIS.2 + Hi3D"   # the sheet out.jpg from them

(Blender's Python has no PIL, so the sheet is a second call.) CLAY_COLOURS=1 keeps the models' own colours,
CLAY_SIZE sets the pixels per view. The models are in glTF axes (Y up, front +Z) as
generate.py and compare_hi3d.py write them; each is fitted to its own frame. Cycles on the CPU, orthographic.
"""
import os
import sys

SIZE = int(os.environ.get("CLAY_SIZE", 700))  # pixels per view
VIEWS = ["front", "side", "back"]


def render(model, out, tag):
    import bpy
    from mathutils import Vector
    directions = {"front": Vector((0, -1, 0)), "side": Vector((1, 0, 0)), "back": Vector((0, 1, 0))}  # Blender: front -Y
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    bpy.ops.import_scene.gltf(filepath=model)
    clay = bpy.data.materials.new("clay")
    clay.use_nodes = True
    bsdf = clay.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.62, 0.62, 0.64, 1)
    bsdf.inputs["Roughness"].default_value = 0.6
    meshes = [o for o in scene.objects if o.type == "MESH"]
    for o in meshes if not os.environ.get("CLAY_COLOURS") else []:  # CLAY_COLOURS=1: the models' own colours
        o.data.materials.clear()
        o.data.materials.append(clay)
    corners = [o.matrix_world @ Vector(c) for o in meshes for c in o.bound_box]
    low = Vector([min(c[i] for c in corners) for i in range(3)])
    high = Vector([max(c[i] for c in corners) for i in range(3)])
    centre, span = (low + high) / 2, max(high - low) * 1.08

    world = bpy.data.worlds.new("w")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.16, 0.16, 0.18, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.6
    scene.world = world
    for direction, energy in ((Vector((-0.6, -1, 0.9)), 3.5), (Vector((1, 0.4, 0.5)), 1.2), (Vector((0, 1, 0.3)), 1.0)):
        light = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN"))
        light.data.energy = energy
        light.rotation_euler = (-direction).to_track_quat("-Z", "Y").to_euler()
        scene.collection.objects.link(light)
    camera = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = span
    camera.data.clip_end = span * 10
    scene.collection.objects.link(camera)
    scene.camera = camera
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 24
    scene.render.resolution_x = scene.render.resolution_y = SIZE
    for name in VIEWS:
        d = directions[name]
        camera.location = centre + d * span * 3
        camera.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()
        scene.render.filepath = f"{out}_{tag}_{name}.png"
        bpy.ops.render.render(write_still=True)


def sheet(out, labels):
    from PIL import Image, ImageDraw
    rows = len(labels)
    page = Image.new("RGB", (SIZE * 3 + 240, SIZE * rows + 30), (40, 40, 44))
    draw = ImageDraw.Draw(page)
    for c, name in enumerate(VIEWS):
        draw.text((240 + c * SIZE + 10, 8), name, fill=(220, 220, 220))
    for r, label in enumerate(labels):
        draw.text((12, 30 + r * SIZE + SIZE // 2), label, fill=(255, 230, 120))
        for c, name in enumerate(VIEWS):
            f = f"{out}_{r}_{name}.png"
            page.paste(Image.open(f).convert("RGB"), (240 + c * SIZE, 30 + r * SIZE))
            os.remove(f)
    page.save(out, quality=90)
    print("sheet:", out)


if "--" in sys.argv:  # in Blender
    args = sys.argv[sys.argv.index("--") + 1:]
    for k, model in enumerate(args[1:]):
        render(model, args[0], k)
else:
    sheet(sys.argv[1], sys.argv[2:])
