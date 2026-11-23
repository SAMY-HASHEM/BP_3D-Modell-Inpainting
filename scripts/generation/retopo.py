"""(An experiment, Nov 23: on the motorcycle girl worse than the original. QuadriFlow wants a closed mesh, so the
model is voxel-remeshed first; that turns thin open parts (panels, cloth, a visor) into double walls, which break
when reduced: dark blotches, softer edges. QuadriFlow itself took 97 s once and hung > 25 min the next time on
the same mesh (one CPU core): it runs in its own Blender with a time limit. Maybe for closed bodies (a character
to rig); not for hard surfaces.)

Retopology (pipeline stage 6): a generated model's dense triangles -> clean quads with a fresh UV layout, its colours
baked over from the original. Runs in Blender (QuadriFlow is built in; nothing to download).

  1 clean     a closed copy: voxel remesh (--voxel thousandths of the model's size, default 6), small loose bits
              dropped, reduced to --pre faces (QuadriFlow is slow on big meshes), degenerate edges removed
  2 quads     QuadriFlow: quad loops following the shape, about --faces faces; in its own Blender with a limit
              (--seconds, 300): over it, or refused, an even triangle mesh (--faces x 2) instead
  3 UVs       Smart UV Project (islands by angle, packed with a margin)
  4 colour    the original's colours (base colour, and metal/roughness as plain values) baked onto the new UVs
              ("selected to active", rays from a little outside), --texture px

Why: a generated mesh is triangle soup (~1 M faces, seams everywhere); quads bend cleaner at the joints (rigging),
weigh less, and a fresh UV layout is what stage 7 paints the views onto (painting first and remeshing after would
throw the paint away).

    blender -b --python scripts/generation/retopo.py -- model.glb out.glb [--faces 60000] [--pre 100000] [--voxel 6] [--seconds 300] [--texture 4096]

Writes out.glb and out_wire.png (the new quads, front and side).
"""
import math
import os
import sys

import bpy


def args():
    a = sys.argv[sys.argv.index("--") + 1:]
    opt = {"faces": 60000, "pre": 100000, "texture": 4096, "voxel": 6, "seconds": 300}
    pos = []
    i = 0
    while i < len(a):
        if a[i].startswith("--"):
            opt[a[i][2:]] = int(a[i + 1])
            i += 2
        else:
            pos.append(a[i])
            i += 1
    return pos, opt


def select_only(objs, active):
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = active


def quadriflow_only(src, out, faces):
    """The child Blender: QuadriFlow on a .ply, the result to a .ply (nothing written when it refuses)."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.ply_import(filepath=src)
    obj = bpy.context.view_layer.objects.active
    if bpy.ops.object.quadriflow_remesh(target_faces=int(faces), seed=0) == {"FINISHED"}:
        bpy.ops.wm.ply_export(filepath=out, export_selected_objects=True, export_normals=False, export_uv=False,
                              export_colors="NONE")


def main():
    if "--quadriflow" in sys.argv:
        a = sys.argv[sys.argv.index("--quadriflow") + 1:]
        return quadriflow_only(*a)
    (src, out), opt = args()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=os.path.abspath(src))
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    for o in meshes:  # their places into the meshes
        o.data.transform(o.matrix_world)
        o.parent = None
        o.matrix_world.identity()
    select_only(meshes, meshes[0])
    if len(meshes) > 1:
        bpy.ops.object.join()
    original = bpy.context.view_layer.objects.active
    original.name = "original"

    # 1 clean: a closed surface QuadriFlow accepts (it wants a manifold, consistently facing mesh with no edge shorter
    # than 1e-4 units, a fixed number: our models are 1 unit tall, so the work is done 100 x bigger). A plain copy
    # (points and faces only), a voxel remesh (closed), the small loose bits out, reduced (QuadriFlow is slow on
    # big meshes: 418 k faces ran > 28 min, 132 k 97 s), degenerate edges out (one edge of 600 k blocked it)
    import bmesh
    scale = 100.0
    size = max(original.dimensions)
    mesh = bpy.data.meshes.new("retopo")
    mesh.from_pydata([v.co * scale for v in original.data.vertices], [], [p.vertices[:] for p in original.data.polygons])
    work = bpy.data.objects.new("retopo", mesh)
    bpy.context.scene.collection.objects.link(work)
    select_only([work], work)
    work.data.remesh_voxel_size = opt["voxel"] / 1000 * size * scale
    bpy.ops.object.voxel_remesh()
    bm = bmesh.new()
    bm.from_mesh(work.data)
    seen, small = set(), []
    for f in bm.faces:
        if f.index in seen:
            continue
        part, stack = [f], [f]
        seen.add(f.index)
        while stack:
            for e in stack.pop().edges:
                for h in e.link_faces:
                    if h.index not in seen:
                        seen.add(h.index)
                        stack.append(h)
                        part.append(h)
        if len(part) < 0.002 * len(bm.faces):
            small += part
    bmesh.ops.delete(bm, geom=small, context="FACES")
    bm.to_mesh(work.data)
    bm.free()
    if len(work.data.polygons) > opt["pre"]:
        dec = work.modifiers.new("pre", "DECIMATE")
        dec.ratio = opt["pre"] / len(work.data.polygons)
        bpy.ops.object.modifier_apply(modifier="pre")
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.dissolve_degenerate(threshold=1e-3)
    bpy.ops.object.mode_set(mode="OBJECT")
    print(f"clean: {len(work.data.polygons)} faces", flush=True)

    # 2 quads: QuadriFlow in its own Blender, with a time limit (on one CPU core it took 97 s for this mesh once and
    # hung > 25 min the next time, Nov 23); over the limit or refused: an even triangle mesh instead (the voxel
    # surface reduced to --faces x 2 triangles), always done in seconds
    import subprocess
    import tempfile
    tmp = tempfile.mkdtemp()
    src_ply, out_ply = os.path.join(tmp, "in.ply"), os.path.join(tmp, "out.ply")
    select_only([work], work)
    bpy.ops.wm.ply_export(filepath=src_ply, export_selected_objects=True, apply_modifiers=False,
                          export_normals=False, export_uv=False, export_colors="NONE")
    try:
        done = subprocess.run([bpy.app.binary_path, "-b", "--python", os.path.abspath(__file__), "--", "--quadriflow",
                               src_ply, out_ply, str(opt["faces"])], timeout=opt["seconds"],
                              capture_output=True, text=True)
        ok = os.path.isfile(out_ply)
        if not ok:
            print("QuadriFlow refused:", [l for l in done.stdout.splitlines() if "QuadriFlow" in l][:1], flush=True)
    except subprocess.TimeoutExpired:
        ok = False
        print(f"QuadriFlow over {opt['seconds']} s: stopped", flush=True)
    if ok:
        bpy.ops.wm.ply_import(filepath=out_ply)
        quads = bpy.context.view_layer.objects.active
        work.data = quads.data
        bpy.data.objects.remove(quads)
        kind = "quads (QuadriFlow)"
    else:
        dec = work.modifiers.new("even", "DECIMATE")
        dec.ratio = min(1.0, 2 * opt["faces"] / max(len(work.data.polygons), 1))
        select_only([work], work)
        bpy.ops.object.modifier_apply(modifier="even")
        kind = "even triangles (QuadriFlow's fallback)"
    work.data.transform(__import__("mathutils").Matrix.Scale(1 / scale, 4))
    print(f"retopology: {len(work.data.polygons)} faces, {kind}", flush=True)

    # 3 UVs
    select_only([work], work)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=math.radians(60), island_margin=0.003)
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.shade_smooth()

    # 4 colour: baked from the original onto the new UVs
    size = opt["texture"]
    image = bpy.data.images.new("base_color", size, size)
    mat = bpy.data.materials.new("retopo")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = image
    nodes.active = tex
    bsdf = nodes["Principled BSDF"]
    mat.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    work.data.materials.append(mat)
    s = bpy.context.scene
    s.render.engine = "CYCLES"
    s.cycles.device = "CPU"
    s.cycles.samples = 1
    s.render.bake.use_selected_to_active = True
    s.render.bake.use_cage = False
    s.render.bake.cage_extrusion = 0.01
    s.render.bake.max_ray_distance = 0.03
    s.render.bake.margin = 8
    s.render.bake.use_pass_direct = False
    s.render.bake.use_pass_indirect = False
    s.render.bake.use_pass_color = True
    select_only([original, work], work)
    bpy.ops.object.bake(type="DIFFUSE", pass_filter={"COLOR"})
    image.filepath_raw = os.path.splitext(os.path.abspath(out))[0] + "_base_color.png"
    image.file_format = "PNG"
    image.save()
    image.pack()
    bsdf.inputs["Metallic"].default_value = 0.0  # painted surfaces; stage 7 decides metal where it really is
    bsdf.inputs["Roughness"].default_value = 0.6
    bpy.data.objects.remove(original)
    select_only([work], work)
    bpy.ops.export_scene.gltf(filepath=os.path.abspath(out), export_format="GLB", use_selection=True)
    print("retopology:", out, flush=True)
    wire(work, os.path.splitext(os.path.abspath(out))[0] + "_wire.png")


def wire(obj, path):
    """The new quads as lines over the model, from the front and the side."""
    from mathutils import Vector
    s = bpy.context.scene
    s.render.engine = "BLENDER_WORKBENCH"
    s.display.shading.light = "STUDIO"
    s.display.shading.color_type = "TEXTURE"
    s.display.shading.show_object_outline = False
    s.render.resolution_x, s.render.resolution_y = 1400, 700
    mod = obj.modifiers.new("w", "WIREFRAME")
    mod.thickness = 0.0008
    mod.use_replace = False
    mod.material_offset = 0
    box = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    centre = sum(box, Vector()) / 8
    span = max((max(p[i] for p in box) - min(p[i] for p in box)) for i in range(3))
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    s.collection.objects.link(cam)
    s.camera = cam
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = span * 1.1
    s.render.resolution_x = s.render.resolution_y = 700
    shots = []
    for k, d in enumerate((Vector((0, -1, 0)), Vector((1, 0, 0)))):
        cam.location = centre + d * span * 3
        cam.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()
        s.render.filepath = path.replace(".png", f"_{k}.png")
        bpy.ops.render.render(write_still=True)
        shots.append(s.render.filepath)
    print("wire pictures:", shots, flush=True)


main()
