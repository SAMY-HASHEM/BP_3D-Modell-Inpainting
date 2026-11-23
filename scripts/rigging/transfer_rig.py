"""A rig carried from the dense model onto its retopologised one (pipeline stages 6 + 9 in the right order): the
skeleton and the hands are fitted on the dense mesh (fit_hands.py needs its fingers: on 39 k quads only one finger
per hand was found, the grip failed, Nov 23), then the clean mesh takes over its skin weights, each point from the
nearest place on the dense surface (Blender's Data Transfer: vertex groups, nearest face, interpolated).

    blender -b --python scripts/rigging/transfer_rig.py -- dense_rigged.blend retopo.glb out.blend

Writes out.blend (the armature with the clean mesh, twist bones and preserve volume as before) and out.glb.
"""
import os
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    a = sys.argv[sys.argv.index("--") + 1:]
    rigged, clean_path, out = a[0], a[1], a[2]
    bpy.ops.wm.open_mainfile(filepath=os.path.abspath(rigged))
    s = bpy.context.scene
    rig = next(o for o in s.objects if o.type == "ARMATURE")
    dense = [o for o in s.objects if o.type == "MESH" and o.parent == rig and o.parent_type != "BONE"]
    before = set(s.objects)
    bpy.ops.import_scene.gltf(filepath=os.path.abspath(clean_path))
    clean = [o for o in s.objects if o not in before and o.type == "MESH"]
    for o in clean:
        o.data.transform(o.matrix_world)
        o.parent = None
        o.matrix_world.identity()
    bpy.ops.object.select_all(action="DESELECT")
    for o in clean:
        o.select_set(True)
    bpy.context.view_layer.objects.active = clean[0]
    if len(clean) > 1:
        bpy.ops.object.join()
    clean = bpy.context.view_layer.objects.active
    if len(dense) > 1:  # one source
        bpy.ops.object.select_all(action="DESELECT")
        for o in dense:
            o.select_set(True)
        bpy.context.view_layer.objects.active = dense[0]
        bpy.ops.object.join()
    source = dense[0]
    for g in source.vertex_groups:
        clean.vertex_groups.new(name=g.name)
    dt = clean.modifiers.new("weights", "DATA_TRANSFER")
    dt.object = source
    dt.use_vert_data = True
    dt.data_types_verts = {"VGROUP_WEIGHTS"}
    dt.vert_mapping = "POLYINTERP_NEAREST"
    dt.layers_vgroup_select_src = "ALL"
    dt.layers_vgroup_select_dst = "NAME"
    bpy.context.view_layer.objects.active = clean
    bpy.ops.object.modifier_apply(modifier="weights")
    bpy.ops.object.vertex_group_normalize_all(lock_active=False)
    clean.parent = rig
    arm = clean.modifiers.new("rig", "ARMATURE")
    arm.object = rig
    for m in source.modifiers:  # preserve volume as the dense one had it
        if m.type == "ARMATURE":
            arm.use_deform_preserve_volume = m.use_deform_preserve_volume
    bpy.data.objects.remove(source)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(out))
    bpy.ops.export_scene.gltf(filepath=os.path.splitext(os.path.abspath(out))[0] + ".glb", export_format="GLB",
                              export_skins=True)
    print(f"rig carried over: {len(clean.data.polygons)} faces, {len(clean.vertex_groups)} groups -> {out}", flush=True)


main()
