"""Retopology (pipeline stage 6): a generated model's dense triangles -> clean quads with a fresh UV layout, its colours
baked over from the original. For closed bodies (characters): the knight 950 k triangles -> 39 k quads in 14 s, the
same shape and colours (Nov 23). Not for hard surfaces with thin open parts (the motorcycle girl's panels): the voxel
step makes them double-walled. Not before rigging (Nov 23): the knight at 39 k quads (voxel 4) kept 1 finger per hand,
at 144 k (voxel 2) 3 without the thumb; the grip check failed where the dense mesh passed. Rig the dense mesh, then
retopologise and carry the weights over (rigging/transfer_rig.py: the 39 k quads then pass the grip check). Runs in
Blender; the quads by Instant Meshes (local_ai_models/instant_meshes, BSD).

  1 clean     a closed outer skin (skin_surface.py: surface points -> voxels of --voxel thousandths of the model's
              size, default 6 -> filled -> the outside; no inside needed, TRELLIS.2's 1536 export is open patches),
              reduced to --pre faces, degenerate edges removed
  2 quads     Instant Meshes: quad loops following the shape, about --faces faces (--method im); without it, or
              failing, QuadriFlow in its own Blender with a limit (--seconds, 300; --method quadriflow), and last an
              even triangle mesh (--faces x 2)
  3 UVs       Smart UV Project (islands by angle, packed with a margin)
  4 colour    the original's colours (base colour, and metal/roughness as plain values) baked onto the new UVs
              ("selected to active", rays from a little outside), --texture px

Why: a generated mesh is triangle soup (~1 M faces, seams everywhere); quads bend cleaner at the joints (rigging),
weigh less, and a fresh UV layout is what stage 7 paints the views onto (painting first and remeshing after would
throw the paint away).

    blender -b --python scripts/generation/retopo.py -- model.glb out.glb [--faces 60000] [--pre 100000] [--voxel 6] [--seconds 300] [--texture 4096]
        [--shape shape.glb]      (the body from this plain shape, the colours from model.glb)

Writes out.glb and out_wire.png (the new quads, front and side).
"""
import math
import os
import sys

import bpy

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INSTANT_MESHES = os.path.join(PROJECT, "local_ai_models", "instant_meshes", "Instant Meshes")


def args():
    a = sys.argv[sys.argv.index("--") + 1:]
    opt = {"faces": 60000, "pre": 100000, "texture": 4096, "voxel": 6, "seconds": 300, "method": "auto"}
    pos = []
    i = 0
    while i < len(a):
        if a[i].startswith("--"):
            opt[a[i][2:]] = a[i + 1] if a[i][2:] in ("method", "shape") else int(a[i + 1])
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
    geometry = original
    if opt.get("shape"):  # the body from the plain shape (a model whose UVs are a mosaic, MV-Adapter's, shredded in the
        # voxel step even welded: the knight made in one go); the colours still baked from the model itself
        before = set(bpy.context.scene.objects)
        bpy.ops.import_scene.gltf(filepath=os.path.abspath(opt["shape"]))
        parts = [o for o in bpy.context.scene.objects if o not in before and o.type == "MESH"]
        for o in parts:
            o.data.transform(o.matrix_world)
            o.parent = None
            o.matrix_world.identity()
        select_only(parts, parts[0])
        if len(parts) > 1:
            bpy.ops.object.join()
        geometry = bpy.context.view_layer.objects.active
        geometry.name = "shape"

    # 1 clean: a closed surface QuadriFlow accepts (it wants a manifold, consistently facing mesh with no edge shorter
    # than 1e-4 units, a fixed number: our models are 1 unit tall, so the work is done 100 x bigger). A plain copy
    # (points and faces only), a voxel remesh (closed), the small loose bits out, reduced (QuadriFlow is slow on
    # big meshes: 418 k faces ran > 28 min, 132 k 97 s), degenerate edges out (one edge of 600 k blocked it)
    import bmesh
    scale = 100.0
    size = max(geometry.dimensions)
    mesh = bpy.data.meshes.new("retopo")
    mesh.from_pydata([v.co * scale for v in geometry.data.vertices], [], [p.vertices[:] for p in geometry.data.polygons])
    work = bpy.data.objects.new("retopo", mesh)
    bpy.context.scene.collection.objects.link(work)
    if geometry is not original:
        bpy.data.objects.remove(geometry)
    select_only([work], work)
    # the points welded first: a model whose UVs are a mosaic (MV-Adapter's) is split along thousands of seams, and the
    # voxel step can't tell inside from outside on such loose pieces (the knight made in one go came out shredded)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=1e-4 * size * scale)
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    # a closed outer skin without needing an inside (skin_surface.py: points -> voxels -> filled -> outer surface).
    # Blender's voxel remesh needs to know inside from outside; TRELLIS.2's 1536 export is open patches facing any
    # way (fix_normals.py), and it came out as lace (the knight made in one go, Nov 23)
    import subprocess
    import tempfile
    skin_dir = tempfile.mkdtemp()
    raw, skin_path = os.path.join(skin_dir, "raw.ply"), os.path.join(skin_dir, "skin.ply")
    select_only([work], work)
    bpy.ops.wm.ply_export(filepath=raw, export_selected_objects=True, apply_modifiers=False, export_normals=False,
                          export_uv=False, export_colors="NONE", export_triangulated_mesh=True)
    venv = os.path.join(PROJECT, "conda", "envs", "mvadapter-venv", "bin", "python")
    done = subprocess.run([venv, os.path.join(PROJECT, "scripts", "generation", "skin_surface.py"), raw, skin_path,
                           "--voxel", str(opt["voxel"] / 1000 * size * scale)], capture_output=True, text=True)
    print((done.stdout + done.stderr).strip().splitlines()[-1:], flush=True)
    bpy.ops.wm.ply_import(filepath=skin_path)
    skin = bpy.context.view_layer.objects.active
    old_data = work.data
    work.data = skin.data
    bpy.data.objects.remove(skin)
    bpy.data.meshes.remove(old_data)
    select_only([work], work)
    print(f"skin: {len(work.data.polygons)} faces", flush=True)
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

    # 2 quads: Instant Meshes (its own program, local_ai_models/instant_meshes; BSD) when it is there: on the closed
    # voxel surface a clean quad mesh in seconds (the knight: 3 s, Nov 23; on the raw generated mesh it cracked into
    # pieces). Else QuadriFlow in its own Blender with a time limit (97 s once, > 25 min the next time on the same
    # mesh). Either failing: an even triangle mesh (the voxel surface reduced to --faces x 2 triangles)
    import subprocess
    import tempfile
    tmp = tempfile.mkdtemp()
    src_ply, out_ply, out_obj = (os.path.join(tmp, n) for n in ("in.ply", "out.ply", "out.obj"))
    select_only([work], work)
    bpy.ops.wm.ply_export(filepath=src_ply, export_selected_objects=True, apply_modifiers=False,
                          export_normals=False, export_uv=False, export_colors="NONE", export_triangulated_mesh=True)
    kind = None
    if os.path.isfile(INSTANT_MESHES) and opt["method"] != "quadriflow":
        # its -v gives about 4 x as many points as asked (calibrated on the knight): -v faces / 4 ~ faces quads.
        # It sometimes stops ("Internal error in extraction": the knight made in one go, Nov 23): then again,
        # quad-dominant (-D), then a little denser
        v = max(500, opt["faces"] // 4)
        faces = lambda path: sum(1 for line in open(path) if line.startswith("f ")) if os.path.isfile(path) else 0
        for k, extra in enumerate(([], ["-D"], [], ["-D"])):
            vk = v * (1, 1, 2, 2)[k]
            command = [INSTANT_MESHES, "-o", out_obj, "-v", str(vk), "-t", "16"] + extra + [src_ply]
            if os.path.isfile(out_obj):
                os.remove(out_obj)
            try:
                done = subprocess.run(command, timeout=opt["seconds"], capture_output=True, text=True)
            except subprocess.TimeoutExpired:
                print(f"Instant Meshes over {opt['seconds']} s: stopped", flush=True)
                continue
            got = faces(out_obj)
            if got >= 0.5 * opt["faces"]:  # its -v means more or fewer faces from mesh to mesh (3.5 k for 40 k once)
                break
            print(f"Instant Meshes -v {vk} {' '.join(extra)}: {got} faces" +
                  ("" if got else f" ({(done.stdout + done.stderr).strip().splitlines()[-1:]})"), flush=True)
            if os.path.isfile(out_obj):
                os.remove(out_obj)
        if os.path.isfile(out_obj):
            bpy.ops.wm.obj_import(filepath=out_obj, forward_axis="Y", up_axis="Z")
            quads = bpy.context.view_layer.objects.active
            work.data = quads.data
            bpy.data.objects.remove(quads)
            select_only([work], work)
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.remove_doubles(threshold=1e-4 * scale)  # its OBJ repeats the corners of each quad
            bpy.ops.object.mode_set(mode="OBJECT")
            kind = "quads (Instant Meshes)"
    if kind is None and opt["method"] != "im":
        try:
            done = subprocess.run([bpy.app.binary_path, "-b", "--python", os.path.abspath(__file__), "--",
                                   "--quadriflow", src_ply, out_ply, str(opt["faces"])], timeout=opt["seconds"],
                                  capture_output=True, text=True)
            if os.path.isfile(out_ply):
                bpy.ops.wm.ply_import(filepath=out_ply)
                quads = bpy.context.view_layer.objects.active
                work.data = quads.data
                bpy.data.objects.remove(quads)
                kind = "quads (QuadriFlow)"
            else:
                print("QuadriFlow refused:", [l for l in done.stdout.splitlines() if "QuadriFlow" in l][:1], flush=True)
        except subprocess.TimeoutExpired:
            print(f"QuadriFlow over {opt['seconds']} s: stopped", flush=True)
    if kind is None:
        dec = work.modifiers.new("even", "DECIMATE")
        dec.ratio = min(1.0, 2 * opt["faces"] / max(len(work.data.polygons), 1))
        select_only([work], work)
        bpy.ops.object.modifier_apply(modifier="even")
        bpy.ops.object.mode_set(mode="EDIT")  # all faces outward (reduced, some turned: dark and light at random)
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.mode_set(mode="OBJECT")
        kind = "even triangles (fallback)"
    select_only([work], work)  # all faces outward: Instant Meshes gave every model inside out (volume < 0: the knight,
    bpy.ops.object.mode_set(mode="EDIT")  # the motorcycle girl, Malenia, Nov 24), so the bake's cage went inward and
    bpy.ops.mesh.select_all(action="SELECT")  # thin parts got their other side's colour (Malenia's red cape black)
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    work.data.transform(__import__("mathutils").Matrix.Scale(1 / scale, 4))
    # the corners back onto the original surface (the voxel surface has terraces of its voxels: they showed on the
    # knight's sleeves, Nov 23); the quads' layout stays, their shape is the original's
    if kind.startswith("quads"):
        select_only([work], work)
        wrap = work.modifiers.new("onto", "SHRINKWRAP")
        wrap.target = original
        wrap.wrap_method = "NEAREST_SURFACEPOINT"
        bpy.ops.object.modifier_apply(modifier="onto")
        smooth = work.modifiers.new("even", "CORRECTIVE_SMOOTH")  # evens out the corners the wrap bunched
        smooth.iterations, smooth.factor, smooth.use_only_smooth = 3, 0.5, False
        bpy.ops.object.modifier_apply(modifier="even")
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
    s.render.bake.cage_extrusion = 0.002  # the quads lie on the original surface (shrinkwrap): short rays, no neighbour hit
    s.render.bake.max_ray_distance = 0.006  # (0.01 / 0.03 smeared thin parts: the motorcycle girl, Nov 23)
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
