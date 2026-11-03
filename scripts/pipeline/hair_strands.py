"""Node 05: real hair strands on a TRELLIS.2 model (Blender, Geometry Nodes).

TRELLIS.2 builds surfaces in a voxel grid; a strand thinner than a voxel can't exist, so its hair is a closed
shell (with cracks where strands would separate). This node keeps the shell as the hair's volume and grows
real strands on it:

  1. find the hair    the faces whose texture colour is the hair colour (node 01's vision AI names it)
  2. guide strands    roots spread over the hair shell; each strand walks step by step ALONG the shell, downwards
                      (gravity, pushed along the surface), snapped back onto the shell at every step, so the
                      strands follow the shell's flow, parting and fringe
  3. Geometry Nodes   Blender's own hair nodes on the strands: Clump Hair Curves (locks), Hair Curves Noise,
                      Set Hair Curve Profile (thin tips)
  4. colour           every strand takes the shell's texture colour at its root; Principled Hair BSDF
  5. out              <out>.blend (real hair curves), <out>_*.png renders, <out>.glb (strands as thin strips,
                      .glb can't store hair curves) with --glb

The same node makes fur: shorter --length, more --count.

Usage:
    blender -b --python scripts/pipeline/hair_strands.py -- model.glb out/hair --hair-colour "#2e9c7a" \\
        [--count 30000] [--length 0.10] [--focus X0 Y0 Z0 X1 Y1 Z1] [--glb]
"""
import math
import os
import random
import sys

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

args = sys.argv[sys.argv.index("--") + 1:]
MODEL, OUT = args[0], args[1]


def option(name, default, kind=str, count=1):
    if f"--{name}" not in args:
        return default
    k = args.index(f"--{name}")
    values = [kind(v) for v in args[k + 1:k + 1 + count]]
    return values[0] if count == 1 else values


HAIR_COLOUR = option("hair-colour", None)
COUNT = option("count", 30000, int)
LENGTH = option("length", 0.10, float)  # strand length, share of the hair's height
POINTS = option("points", 16, int)
HUE_TOLERANCE = option("hue-tolerance", 0.07, float)  # 0.07 of the colour circle = 25 degrees
HAIR_SHARE = option("hair-share", 1.0, float)  # the hair only in the top share of the model (node 01's hair box)
FOCUS = option("focus", None, float, 6)
SIZE = option("size", 384, int)
SAMPLES = option("samples", 32, int)
SEED = option("seed", 1, int)
random.seed(SEED)
rng = np.random.default_rng(SEED)


def hsv(rgb):
    """rgb (n, 3) in 0..1 -> hue, saturation, value (n,) each, hue 0..1."""
    mx, mn = rgb.max(1), rgb.min(1)
    delta = mx - mn + 1e-8
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    hue = np.where(mx == r, ((g - b) / delta) % 6, np.where(mx == g, (b - r) / delta + 2, (r - g) / delta + 4)) / 6
    return hue, (mx - mn) / (mx + 1e-8), mx


def texture_of(obj):
    """The base colour image of an object's (glTF) material, as a float array (h, w, 4)."""
    for slot in obj.material_slots:
        nodes = slot.material.node_tree.nodes if slot.material and slot.material.node_tree else []
        for node in nodes:
            if node.type == "TEX_IMAGE" and any(l.to_socket.name == "Base Color" for out in node.outputs for l in out.links):
                image = node.image
                pixels = np.empty(image.size[0] * image.size[1] * 4, np.float32)
                image.pixels.foreach_get(pixels)
                return pixels.reshape(image.size[1], image.size[0], 4)
    return None


def face_data(obj):
    """World positions of the vertices, the triangles, and every face's centre, normal, area and texture colour."""
    mesh = obj.data
    mesh.calc_loop_triangles()
    world = np.array(obj.matrix_world)
    verts = np.empty(len(mesh.vertices) * 3, np.float32)
    mesh.vertices.foreach_get("co", verts)
    verts = verts.reshape(-1, 3) @ world[:3, :3].T + world[:3, 3]
    tris = np.empty(len(mesh.loop_triangles) * 3, np.int32)
    mesh.loop_triangles.foreach_get("vertices", tris)
    tris = tris.reshape(-1, 3)
    loops = np.empty(len(mesh.loop_triangles) * 3, np.int32)
    mesh.loop_triangles.foreach_get("loops", loops)
    corners = verts[tris]
    cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    area = np.linalg.norm(cross, axis=1) / 2
    normal = cross / (2 * area[:, None] + 1e-12)
    colour = np.full((len(tris), 3), 0.5, np.float32)
    texture = texture_of(obj)
    if texture is not None and mesh.uv_layers.active:
        uv = np.empty(len(mesh.loops) * 2, np.float32)
        mesh.uv_layers.active.data.foreach_get("uv", uv)
        uv = uv.reshape(-1, 2)[loops].reshape(-1, 3, 2).mean(1)
        h, w = texture.shape[:2]
        px = (np.mod(uv[:, 0], 1) * (w - 1)).astype(int)
        py = (np.mod(uv[:, 1], 1) * (h - 1)).astype(int)
        colour = texture[py, px, :3]
    return verts, tris, corners.mean(1), normal, area, colour


# ------------------------------------------------------------------ scene

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=os.path.abspath(MODEL))
meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]

# 1. the hair: faces of the hair colour (in every mesh of the model)
target = np.array([int(HAIR_COLOUR.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)], np.float32)[None]
target_hue, target_sat, target_val = hsv(target)
corners_all = [np.array(o.matrix_world) @ np.array([*c, 1]) for o in meshes for c in o.bound_box]
model_top = max(c[2] for c in corners_all)
model_height = model_top - min(c[2] for c in corners_all)
hair_verts, hair_tris, hair_colours, roots = [], [], [], []
offset = 0
for obj in meshes:
    verts, tris, centre, normal, area, colour = face_data(obj)
    hue, sat, val = hsv(colour)
    near = np.minimum(np.abs(hue - target_hue), 1 - np.abs(hue - target_hue)) < HUE_TOLERANCE
    hair = near & (sat > 0.5 * target_sat[0]) & (val > 0.35 * target_val[0])  # dark faces with a tint are no hair
    hair &= centre[:, 2] > model_top - HAIR_SHARE * model_height
    print(f"  {obj.name}: {hair.sum():,} of {len(tris):,} faces are hair", flush=True)
    hair_verts.append(verts)
    hair_tris.append(tris[hair] + offset)
    hair_colours.append((centre[hair], normal[hair], area[hair], colour[hair], verts[tris[hair]]))
    offset += len(verts)
verts = np.concatenate(hair_verts)
tris = np.concatenate(hair_tris)
centre, normal, area, colour, corners = (np.concatenate([h[i] for h in hair_colours]) for i in range(5))
if len(tris) < 100:
    raise SystemExit(f"only {len(tris)} faces have the hair colour {HAIR_COLOUR}: is it the right colour?")
tree = BVHTree.FromPolygons([Vector(v) for v in verts], tris.tolist(), all_triangles=True)
low, high = corners.reshape(-1, 3).min(0), corners.reshape(-1, 3).max(0)
height = high[2] - low[2]
crown = Vector(((low[0] + high[0]) / 2, (low[1] + high[1]) / 2, high[2]))
print(f"  hair: {len(tris):,} faces, {height:.3f} high", flush=True)

# 2. guide strands: roots spread by area, each walks down along the shell
pick = rng.choice(len(tris), COUNT, p=area / area.sum())
bary = rng.dirichlet((1, 1, 1), COUNT)
starts = (corners[pick] * bary[:, :, None]).sum(1)
gravity = Vector((0, 0, -1))
strands, strand_colours = [], []
for k in range(COUNT):
    p = Vector(starts[k])
    n = Vector(normal[pick[k]])
    length = LENGTH * height * (0.5 + random.random())
    step = length / (POINTS - 1)
    lift = height * 0.002 * random.random()  # strands lie on the shell, a little above it
    points, d = [p + n * lift], None
    for _ in range(POINTS - 1):
        along = gravity - n * gravity.dot(n)
        if along.length < 0.3:  # the top of the head: away from the crown, along the surface
            away = p - crown
            along = away - n * away.dot(n)
        along = along.normalized() if along.length > 1e-6 else Vector((0, 0, -1))
        d = along if d is None else (0.6 * d + 0.4 * along).normalized()
        q, n_new, _, _ = tree.find_nearest(p + d * step)
        if q is None or (q - p).length < 0.3 * step:  # the shell ends here (a tip)
            break
        p, n = q, n_new
        points.append(p + n * lift)
    if len(points) >= 3:
        strands.append(points)
        strand_colours.append(colour[pick[k]])
print(f"  {len(strands):,} strands grown", flush=True)

# the strands as hair curves
curves = bpy.data.hair_curves.new("hair")
curves.add_curves([len(s) for s in strands])
curves.position_data.foreach_set("vector", np.array([c for s in strands for v in s for c in v], np.float32))
attribute = curves.attributes.new("hair_colour", "FLOAT_COLOR", "CURVE")
attribute.data.foreach_set("color", np.concatenate([np.array(strand_colours), np.ones((len(strands), 1))], 1).astype(np.float32).ravel())
hair = bpy.data.objects.new("hair", curves)
bpy.context.scene.collection.objects.link(hair)

# 3. Geometry Nodes: Blender's own hair nodes
library = os.path.join(os.path.dirname(bpy.app.binary_path), f"{bpy.app.version[0]}.{bpy.app.version[1]}",
                       "datafiles", "assets", "nodes", "procedural_hair_node_assets.blend")
STACK = [  # (node group, {input: value}) -- the sizes are shares of the hair's height
    ("Clump Hair Curves", {"Factor": 0.7, "Shape": 0.4, "Tip Spread": 0.002 * height, "Guide Distance": 0.025 * height,
                           "Existing Guide Map": False, "Preserve Length": True}),
    ("Hair Curves Noise", {"Factor": 0.5, "Distance": 0.004 * height, "Scale": 8.0}),
    ("Set Hair Curve Profile", {"Radius": 0.0012 * height, "Shape": 0.5}),
]
with bpy.data.libraries.load(library) as (source, target_lib):
    target_lib.node_groups = [name for name, _ in STACK]
for name, values in STACK:
    group = bpy.data.node_groups[name]
    modifier = hair.modifiers.new(name, "NODES")
    modifier.node_group = group
    for item in group.interface.items_tree:
        if item.item_type == "SOCKET" and item.in_out == "INPUT" and item.name in values:
            getattr(modifier.properties.inputs, item.identifier).value = values[item.name]  # Blender 5 API

# 4. colour: Principled Hair BSDF with every strand's own colour
material = bpy.data.materials.new("hair")
material.use_nodes = True
nodes, links = material.node_tree.nodes, material.node_tree.links
nodes.clear()
bsdf = nodes.new("ShaderNodeBsdfHairPrincipled")
bsdf.parametrization = "COLOR"
bsdf.inputs["Roughness"].default_value = 0.35
attr = nodes.new("ShaderNodeAttribute")
attr.attribute_name = "hair_colour"
links.new(attr.outputs["Color"], bsdf.inputs["Color"])
output = nodes.new("ShaderNodeOutputMaterial")
links.new(bsdf.outputs[0], output.inputs["Surface"])
curves.materials.append(material)

os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(OUT + ".blend"))
print(f"Saved {OUT}.blend", flush=True)

# 5. renders (the same cameras as render_preview.py)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = SAMPLES
scene.cycles.use_denoising = True
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
if FOCUS:
    a, b = Vector(FOCUS[:3]), Vector(FOCUS[3:])
    a, b = Vector((a.x, -a.z, a.y)), Vector((b.x, -b.z, b.y))  # glTF (Y up) -> Blender (Z up)
    aim, span = (a + b) / 2, max(abs((b - a).x), abs((b - a).z)) * 0.62
else:
    aim, span = Vector((crown.x, crown.y, crown.z - 0.35 * height)), 0.6 * height
scene.render.resolution_x = scene.render.resolution_y = 2 * SIZE
for name, angle in (("front", 0), ("left", -45), ("right", 45), ("back", 180)):
    turn = math.radians(angle)
    direction = Vector((math.sin(turn), -math.cos(turn), 0.08)).normalized()
    camera.location = aim + direction * span / math.tan(camera.data.angle / 2)
    camera.rotation_euler = (aim - camera.location).to_track_quat("-Z", "Y").to_euler()
    scene.render.filepath = os.path.abspath(f"{OUT}_{name}.png")
    bpy.ops.render.render(write_still=True)
print(f"Rendered {OUT}_front / _left / _right / _back.png", flush=True)

if "--glb" in args:  # the strands as thin strips (two triangles per segment), next to the model
    evaluated = hair.evaluated_get(bpy.context.evaluated_depsgraph_get()).data
    positions = np.empty(len(evaluated.points) * 3, np.float32)
    evaluated.position_data.foreach_get("vector", positions)
    positions = positions.reshape(-1, 3)
    offsets = np.empty(len(evaluated.curves) + 1, np.int32)
    evaluated.curve_offset_data.foreach_get("value", offsets)
    width = 0.0015 * height
    strip_verts, strip_faces, strip_colours, base = [], [], [], 0
    for c in range(len(evaluated.curves)):
        line = positions[offsets[c]:offsets[c + 1]]
        tangent = np.gradient(line, axis=0)
        side = np.cross(tangent, line - np.array(crown))
        side /= np.linalg.norm(side, axis=1, keepdims=True) + 1e-9
        taper = np.linspace(1, 0.2, len(line))[:, None]
        strip_verts.append(np.concatenate([line - side * width * taper, line + side * width * taper]))
        n = len(line)
        strip_faces += [(base + i, base + i + 1, base + n + i + 1, base + n + i) for i in range(n - 1)]
        strip_colours.append(np.repeat([strand_colours[c] if c < len(strand_colours) else (0.5, 0.5, 0.5)], 2 * n, 0))
        base += 2 * n
    mesh = bpy.data.meshes.new("hair_strips")
    mesh.from_pydata(np.concatenate(strip_verts).tolist(), [], strip_faces)
    colours = mesh.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
    colours.data.foreach_set("color", np.concatenate([np.concatenate(strip_colours), np.ones((len(mesh.vertices), 1))], 1)
                             .astype(np.float32).ravel())
    strips = bpy.data.objects.new("hair_strips", mesh)
    scene.collection.objects.link(strips)
    hair.hide_set(True)
    for o in scene.objects:
        o.select_set(o.type == "MESH")
    bpy.ops.export_scene.gltf(filepath=os.path.abspath(OUT + ".glb"), use_selection=True, export_vertex_color="ACTIVE")
    print(f"Saved {OUT}.glb", flush=True)
