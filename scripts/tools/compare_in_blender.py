"""Put all .glb models of one batch into a single Blender scene, as a grid for comparing them.

Rows are resolutions, columns are step counts (or seeds, if all models have the same steps).

Usage (runs Blender for you, see compare_in_blender.sh):
    ./scripts/tools/compare_in_blender.sh outputs/<picture>
    ./scripts/tools/compare_in_blender.sh outputs/<picture> --render
    ./scripts/tools/compare_in_blender.sh outputs/<picture> --render --back   # the backs of the models

Saves outputs/<picture>/comparison.blend (open it in Blender) and, with --render, comparison.png
(comparison_back.* with --back).
"""
import argparse
import glob
import math
import os
import re
import sys

import bpy
from mathutils import Vector

DEFAULT_STEPS = 12  # same as in generate.py
SPACING = 1.25  # models are about 1 x 1 x 1


def parse_name(path):
    match = re.search(r"_(\d+)_seed(\d+)(?:_steps(\d+))?\.glb$", path)
    if not match:
        return None
    resolution, seed, steps = match.groups()
    return int(resolution), int(seed), int(steps) if steps else DEFAULT_STEPS


def import_glb(path):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    imported = [o for o in bpy.data.objects if o not in before]
    holder = bpy.data.objects.new(os.path.basename(path), None)
    bpy.context.scene.collection.objects.link(holder)
    for obj in imported:
        if obj.parent is None:
            obj.parent = holder
    return holder, imported


def add_label(text, location, size=0.14):
    curve = bpy.data.curves.new(text, "FONT")
    curve.body = text
    curve.size = size
    curve.align_x = "CENTER"
    curve.align_y = "CENTER"
    label = bpy.data.objects.new(text, curve)
    label.location = location
    label.rotation_euler = (math.radians(90), 0, 0)  # stand upright, facing the front camera
    bpy.context.scene.collection.objects.link(label)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Compare a batch of .glb models in Blender")
    parser.add_argument("folder", help="folder with the .glb files, e.g. outputs/Kennen_Armored_Render")
    parser.add_argument("--render", action="store_true", help="also render the grid to comparison.png")
    parser.add_argument("--back", action="store_true", help="turn the models around to compare their backs")
    args = parser.parse_args(argv)

    models = sorted((parse_name(p), p) for p in glob.glob(os.path.join(args.folder, "*.glb")) if parse_name(p))
    if not models:
        sys.exit(f"No generated .glb files found in {args.folder}")
    resolutions = sorted({info[0] for info, _ in models})
    # Columns show whatever was varied: steps if there are several, otherwise seeds.
    by_steps = len({info[2] for info, _ in models}) > 1
    columns = sorted({info[2] if by_steps else info[1] for info, _ in models})
    column_label = (lambda c: f"{c} steps") if by_steps else (lambda c: f"seed {c}")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    for (resolution, seed, steps), path in models:
        print(f"Importing {os.path.basename(path)}")
        holder, _ = import_glb(path)
        col = columns.index(steps if by_steps else seed)
        row = resolutions.index(resolution)
        holder.location = (col * SPACING, 0, -row * SPACING)
        if args.back:
            holder.rotation_euler.z = math.pi

    top = 0.5 + 0.25
    for col, value in enumerate(columns):
        add_label(column_label(value), (col * SPACING, 0, top))
    for row, resolution in enumerate(resolutions):
        add_label(str(resolution), (-0.95, 0, -row * SPACING))

    # Front camera (models face -Y after import), orthographic so all models have the same scale.
    # The picture is sized to the grid: models plus the row labels on the left and column labels on top.
    width = (len(columns) - 1) * SPACING + 1.85
    height = (len(resolutions) - 1) * SPACING + 1.4
    center = Vector((((len(columns) - 1) * SPACING - 0.65) / 2, 0, (0.3 - (len(resolutions) - 1) * SPACING) / 2))
    camera = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = max(width, height)
    camera.location = center + Vector((0, -10, 0))
    camera.rotation_euler = (math.radians(90), 0, 0)
    scene.collection.objects.link(camera)
    scene.camera = camera

    sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", "SUN"))
    sun.data.energy = 5.0
    sun.rotation_euler = (math.radians(60), 0, math.radians(-30))
    scene.collection.objects.link(sun)
    scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.8

    scene.render.resolution_x = round(2400 * width / max(width, height))
    scene.render.resolution_y = round(2400 * height / max(width, height))
    scene.render.film_transparent = False

    out_name = "comparison_back" if args.back else "comparison"
    blend_path = os.path.abspath(os.path.join(args.folder, f"{out_name}.blend"))
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)
    print(f"Saved scene: {blend_path}")

    if args.render:
        scene.render.engine = "BLENDER_EEVEE"
        scene.render.filepath = os.path.abspath(os.path.join(args.folder, f"{out_name}.png"))
        bpy.ops.render.render(write_still=True)
        print(f"Saved render: {scene.render.filepath}")


main()
