"""Geometry Nodes "Clean up" without the Blender window: deletes small loose pieces of generated models.

The node graph itself is in the add-on (inpaint_addon.py: cleanup_group), where the button
"Clean up model" (Inpaint tab, section E) uses it. This runs the same graph on .glb files:
loads each, puts the modifier on it, prints how many faces it removed and, with --out, saves the result.

Usage:
    flatpak run --filesystem=home org.blender.Blender -b --python $PWD/blender/cleanup_nodes.py -- \
        $PWD/outputs/a.glb [$PWD/outputs/b.glb ...] [--out $PWD/outputs/cleaned] [--min-share 0.002]
    or in Blender: Scripting tab, open this file, click the model, Run Script (no file arguments)
"""
import argparse
import importlib.util
import os
import sys

import bpy

# the add-on file next to this one (not an older copy installed in Blender's add-on folder)
_spec = importlib.util.spec_from_file_location(
    "inpaint_addon_project", os.path.join(os.path.dirname(os.path.abspath(__file__)), "inpaint_addon.py"))
addon = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(addon)
add_cleanup, cleanup_input, faces_after = addon.add_cleanup, addon.cleanup_input, addon.faces_after


def clean(obj, min_share=None):
    """Puts the modifier on obj; returns (faces before, faces after)."""
    modifier = add_cleanup(obj)
    if min_share is not None:
        data, prop = cleanup_input(modifier, "Min share")
        if prop == "value":
            data.value = min_share
        else:
            data[prop[2:-2]] = min_share
    return len(obj.data.polygons), faces_after(obj)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(prog="cleanup_nodes.py")
    parser.add_argument("models", nargs="*", help=".glb files (none: the selected objects in Blender)")
    parser.add_argument("--out", help="folder to save the cleaned .glb files into")
    parser.add_argument("--min-share", type=float, help="pieces smaller than this share of all faces go (0.002)")
    args = parser.parse_args(argv)

    if not args.models:
        for obj in [o for o in bpy.context.selected_objects if o.type == "MESH"]:
            before, after = clean(obj, args.min_share)
            print(f"{obj.name}: {before} faces -> {after} after Clean up")
        return
    for path in args.models:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.gltf(filepath=path)
        for obj in [o for o in bpy.context.scene.objects if o.type == "MESH"]:
            before, after = clean(obj, args.min_share)
            print(f"{os.path.basename(path)}: {before} faces -> {after} after Clean up "
                  f"({100 * (before - after) / before:.1f} % removed)", flush=True)
        if args.out:
            os.makedirs(args.out, exist_ok=True)
            bpy.ops.export_scene.gltf(filepath=os.path.join(args.out, os.path.basename(path)), export_apply=True)


main()
