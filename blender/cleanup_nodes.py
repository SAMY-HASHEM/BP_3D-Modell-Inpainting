"""Geometry Nodes "Clean up": deletes small loose pieces of a generated model and closes tiny gaps.

TRELLIS.2 results (and inpainting results) often have small floating pieces, e.g. a leftover under a
new part. This builds a node graph that removes every loose piece with less than `Min share` of the
model's faces. It is a modifier: the model itself is not changed, the modifier can be switched off or
applied (Ctrl+A) at any time.

The graph (open the "Geometry Nodes" workspace to see it):
    Group Input -> Merge by Distance -> Delete Geometry -> Group Output
                                          ^ Selection = "piece too small":
    Mesh Island (which piece a face belongs to)
      -> Accumulate Field, grouped by piece: faces per piece
      -> divided by all faces (Accumulate Field without group)
      -> Compare: share < Min share

Usage:
    in Blender: Scripting tab, open this file, click the model, Run Script
    or without window:
    flatpak run org.blender.Blender -b --python blender/cleanup_nodes.py -- model.glb [out.glb]
"""
import sys

import bpy

NAME = "Clean up"


def link(tree, a, b):
    tree.links.new(a, b)


def build_group():
    """The node graph, made once per .blend file."""
    if NAME in bpy.data.node_groups:
        return bpy.data.node_groups[NAME]
    tree = bpy.data.node_groups.new(NAME, "GeometryNodeTree")
    tree.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    share = tree.interface.new_socket("Min share", in_out="INPUT", socket_type="NodeSocketFloat")
    share.default_value, share.min_value, share.max_value = 0.002, 0.0, 1.0  # 0.2 % of all faces
    gap = tree.interface.new_socket("Merge distance", in_out="INPUT", socket_type="NodeSocketFloat")
    gap.default_value, gap.min_value = 0.0001, 0.0
    tree.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

    nodes = tree.nodes
    inp = nodes.new("NodeGroupInput")
    out = nodes.new("NodeGroupOutput")
    merge = nodes.new("GeometryNodeMergeByDistance")
    island = nodes.new("GeometryNodeInputMeshIsland")
    per_piece = nodes.new("GeometryNodeAccumulateField")
    per_piece.data_type, per_piece.domain = "INT", "FACE"
    all_faces = nodes.new("GeometryNodeAccumulateField")
    all_faces.data_type, all_faces.domain = "INT", "FACE"
    divide = nodes.new("ShaderNodeMath")
    divide.operation = "DIVIDE"
    small = nodes.new("FunctionNodeCompare")
    small.data_type, small.operation = "FLOAT", "LESS_THAN"
    delete = nodes.new("GeometryNodeDeleteGeometry")
    delete.domain = "FACE"

    for node, x, y in ((inp, -900, 0), (merge, -600, 100), (island, -900, -250), (per_piece, -600, -200),
                       (all_faces, -600, -450), (divide, -350, -250), (small, -150, -150),
                       (delete, 100, 100), (out, 350, 100)):
        node.location = (x, y)

    link(tree, inp.outputs["Geometry"], merge.inputs["Geometry"])
    link(tree, inp.outputs["Merge distance"], merge.inputs["Distance"])
    per_piece.inputs[0].default_value = 1  # count one per face ...
    link(tree, island.outputs["Island Index"], per_piece.inputs[1])  # ... grouped by piece
    all_faces.inputs[0].default_value = 1
    link(tree, per_piece.outputs["Total"], divide.inputs[0])
    link(tree, all_faces.outputs["Total"], divide.inputs[1])
    link(tree, divide.outputs[0], small.inputs[0])
    link(tree, inp.outputs["Min share"], small.inputs[1])
    link(tree, merge.outputs["Geometry"], delete.inputs["Geometry"])
    link(tree, small.outputs["Result"], delete.inputs["Selection"])
    link(tree, delete.outputs["Geometry"], out.inputs["Geometry"])
    return tree


def add_cleanup(obj):
    """Puts the "Clean up" modifier on a mesh object (once)."""
    modifier = obj.modifiers.get(NAME) or obj.modifiers.new(NAME, "NODES")
    modifier.node_group = build_group()
    return modifier


def faces_after(obj):
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    return len(evaluated.data.polygons)


if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if args:  # without window: load, clean, report, save
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.gltf(filepath=args[0])
        targets = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    else:
        targets = [o for o in bpy.context.selected_objects if o.type == "MESH"]
    for obj in targets:
        before = len(obj.data.polygons)
        add_cleanup(obj)
        print(f"{obj.name}: {before} faces -> {faces_after(obj)} after Clean up")
    if len(args) > 1:
        bpy.ops.export_scene.gltf(filepath=args[1], export_apply=True)
