"""Blender add-on: mark an area of a model, press a button, and TRELLIS.2 regenerates that area.

The area is the selected faces (Edit Mode) and/or "space boxes": boxes that mark empty space where
something new may grow (e.g. a statue on the ground). The boxes are drawn see-through blue, and the
selected one has an arrow on each side (drag it to move that side) and a dot on each corner (drag it
to pull that corner anywhere, so a box can take any shape). "Find part in box" selects only one part
inside the box (e.g. the helmet, by its name and/or a click on it) with a mask AI on this computer
(scripts/mask/mask_ai.py, installed by install_mask.sh); Box = "Selected faces + empty space" then regenerates
that part and the empty space in the box and keeps every other surface exactly. Optionally a prompt (and a reference picture)
says what should be there: the server then first paints it into a picture of the model, seen from the
current view, and TRELLIS.2 builds that picture's change in 3D.

The work is done by scripts/server/inpaint_server.py, which must be running in the TRELLIS.2 environment.

Install: Edit > Preferences > Add-ons > (arrow at the top right) > Install from Disk... > this file.
Use:     3D view sidebar (N key) > Inpaint tab.
"""
import base64
import json
import os
import subprocess
import threading
import time
import urllib.parse
import urllib.request
import zipfile

import bpy
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader
from mathutils import Matrix, Vector

bl_info = {
    "name": "TRELLIS.2 Inpainting",
    "author": "Samy Hashem",
    "version": (1, 10, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > Inpaint",
    "description": "Regenerate an area of a model with TRELLIS.2, optionally guided by a prompt",
    "category": "3D View",
}

LOCAL_TOKEN = os.path.expanduser("~/.config/inpainting3d/server_token")  # made by inpaint_server.py
BOX = "InpaintBox"
GRID = 64  # cubes per side of TRELLIS.2's grid, as in inpaint.py
BOX_COLOR = (0.25, 0.65, 1.0)  # blue, so a box doesn't look like Blender's orange selection
# a box's 8 corners in Blender's bound_box order: which side of each axis they are on (-1/+1), their
# 12 triangles and 12 edges, and the 4 edges along each axis
CORNER_SIGNS = np.array([(-1, -1, -1), (-1, -1, 1), (-1, 1, 1), (-1, 1, -1), (1, -1, -1), (1, -1, 1), (1, 1, 1), (1, 1, -1)])
BOX_TRIANGLES = [(0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6), (0, 4, 5), (0, 5, 1),
                 (3, 2, 6), (3, 6, 7), (0, 3, 7), (0, 7, 4), (1, 5, 6), (1, 6, 2)]
BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
EDGES_ALONG = [[(a, b) for a, b in BOX_EDGES if CORNER_SIGNS[a][axis] != CORNER_SIGNS[b][axis]] for axis in range(3)]
CUBE_CORNERS = [0, 1, 3, 2, 4, 5, 7, 6]  # Blender's cube (Add > Cube) has its vertices in this order
status = {"server": "not checked", "message": "", "job": None}
draw_handler = None


def server():
    """The inpaint server's address and request headers (with its token), from the add-on preferences."""
    prefs = bpy.context.preferences.addons[__name__].preferences
    address = prefs.server.strip().removeprefix("http://").rstrip("/") or "127.0.0.1:8765"
    token = prefs.token.strip()
    if not token and address.split(":")[0] in ("127.0.0.1", "localhost") and os.path.isfile(LOCAL_TOKEN):
        with open(LOCAL_TOKEN) as f:  # a server on this computer: its token is right here
            token = f.read().strip()
    return f"http://{address}", {"X-Token": token}


def is_local():
    return server()[0].split("//")[1].split(":")[0] in ("127.0.0.1", "localhost")


def ask(path, data=None, timeout=1.0):
    """JSON request to the inpaint server; None if it isn't running."""
    url, headers = server()
    request = urllib.request.Request(url + path, data=json.dumps(data).encode() if data is not None else None,
                                     headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as e:
        return json.loads(e.read())
    except OSError:
        return None


def redraw():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


def to_gltf(points):
    """Blender's axes (Z up) to glTF's (Y up), as Blender's glTF exporter converts them."""
    points = np.asarray(points, np.float64).reshape(-1, 3)
    return np.stack([points[:, 0], points[:, 2], -points[:, 1]], 1)


def is_box(obj):
    return obj is not None and obj.name.startswith(BOX)


def target_model(context):
    """The model to change: the active object, or the last model used when a box is active. If that one
    was deleted meanwhile (it is no longer in the scene), the biggest visible model is taken."""
    obj, settings = context.active_object, context.scene.inpaint
    if obj is not None and obj.type == "MESH" and not is_box(obj):
        settings.model = obj
    model = settings.model
    if model is None or context.view_layer.objects.get(model.name) is not model:
        models = [o for o in context.view_layer.objects if o.type == "MESH" and not is_box(o) and o.visible_get()]
        model = max(models, key=lambda o: len(o.data.polygons), default=None)
        settings.model = model
    return model


def selected_points(obj):
    """World positions of the selected faces' corners and centres (call in Object Mode)."""
    mesh = obj.data
    n = len(mesh.polygons)
    faces = np.zeros(n, bool)
    mesh.polygons.foreach_get("select", faces)
    centers = np.zeros(n * 3, np.float32)
    mesh.polygons.foreach_get("center", centers)
    # corners of the selected faces only (vertex selection flags can be stale, e.g. all on after an import)
    totals = np.zeros(n, np.int32)
    mesh.polygons.foreach_get("loop_total", totals)
    corners = np.zeros(len(mesh.loops), np.int32)
    mesh.loops.foreach_get("vertex_index", corners)
    co = np.zeros(len(mesh.vertices) * 3, np.float32)
    mesh.vertices.foreach_get("co", co)
    corners = corners[np.repeat(faces, totals)]  # loops are stored face after face
    if faces.all():  # everything selected (as after an import) means nothing was chosen
        return np.zeros((0, 3)), 0
    local = np.concatenate([centers.reshape(-1, 3)[faces], co.reshape(-1, 3)[corners]])
    m = np.array(obj.matrix_world)
    return local @ m[:3, :3].T + m[:3, 3], int(faces.sum())


def box_corners(box):
    """The box's 8 corners in world space, in bound_box order. Its corners can be pulled anywhere, so
    the box needn't be rectangular. Uses matrix_basis, which follows location/rotation/scale at once
    (space boxes have no parent)."""
    m = box.matrix_basis
    vertices = box.data.vertices
    if len(vertices) != 8:  # changed by hand in Edit Mode: its bounding box
        return [m @ Vector(c) for c in box.bound_box]
    return [m @ vertices[i].co for i in CUBE_CORNERS]


def box_points(model):
    """Points filling every visible space box (world), about 2 per voxel of TRELLIS.2's 64 grid. They are
    blended between the 8 corners (trilinear), so a box with pulled corners is filled in its own shape."""
    size = max(model.dimensions)
    points = []
    for box in bpy.data.objects:
        if not is_box(box) or not box.visible_get():
            continue
        corners = np.array(box_corners(box))
        grid = np.zeros((2, 2, 2, 3))  # corners by side: grid[x][y][z], 0 = low side, 1 = high side
        for corner, signs in zip(corners, CORNER_SIGNS):
            grid[tuple((signs + 1) // 2)] = corner
        steps = [max(2, int(max(np.linalg.norm(corners[a] - corners[b]) for a, b in edges) / size * 128))
                 for edges in EDGES_ALONG]
        weights = [np.stack([1 - t, t], 1) for t in (np.linspace(0, 1, k) for k in steps)]
        points.append(np.einsum("ia,jb,kc,abcd->ijkd", *weights, grid).reshape(-1, 3))
    return np.concatenate(points) if points else np.zeros((0, 3))


def keep_the_rest(model, part, boxes, grow, room=0.0):
    """For Box = "Selected faces + empty space": the cubes of TRELLIS.2's 64 grid to regenerate, as their
    centres (glTF coordinates, sent with grow 0): the box and the selected faces (world points), grown by
    `grow` cubes, without the cubes that hold only other surfaces of the model, which so stay exactly as
    they are. A cube holding some of the part is regenerated, so the part goes completely. room: as the
    job's Room above (the model placed smaller and lower in the cube, as inpaint.py's Placement)."""
    mesh = model.data
    m = np.array(model.matrix_world)
    world = lambda local: local @ m[:3, :3].T + m[:3, 3]
    co = np.zeros(len(mesh.vertices) * 3, np.float32)
    mesh.vertices.foreach_get("co", co)
    vertices = to_gltf(world(co.reshape(-1, 3)))
    low, high = vertices.min(0), vertices.max(0)
    centre, scale = (low + high) / 2, 0.99999 * (1 - room) / (high - low).max()  # as inpaint.py places the model
    lift = np.array([0, -room / 2, 0])  # glTF Y is the cube's up

    def cubes(points):
        return np.clip((((to_gltf(points) - centre) * scale + lift + 0.5) * GRID).astype(int), 0, GRID - 1)

    mask = np.zeros((GRID,) * 3, bool)
    for points in (part, boxes):
        mask[tuple(cubes(points).T)] = True
    for _ in range(grow):  # one cube in every direction per round, as inpaint.py grows
        padded = np.pad(mask, 1)
        mask = np.any([padded[x:x + GRID, y:y + GRID, z:z + GRID] for x in range(3) for y in range(3) for z in range(3)], 0)
    selected = np.zeros(len(mesh.polygons), bool)
    mesh.polygons.foreach_get("select", selected)
    centers = np.zeros(len(mesh.polygons) * 3, np.float32)
    mesh.polygons.foreach_get("center", centers)
    found = cubes(world(centers.reshape(-1, 3)))
    other, own = np.zeros_like(mask), np.zeros_like(mask)
    other[tuple(found[~selected].T)] = True
    own[tuple(found[selected].T)] = True
    mask &= ~(other & ~own)
    return ((np.argwhere(mask) + 0.5) / GRID - 0.5 - lift) / scale + centre


def area_points(operator, context):
    """The model, the marked area (glTF coordinates), how many faces are selected, whether boxes count, and
    the grow the job must use; or None after reporting what's missing."""
    model = target_model(context)
    if model is None:
        operator.report({"ERROR"}, "Click the model first")
        return None
    if context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")  # also writes the Edit Mode selection into the mesh
    settings = context.scene.inpaint
    faces, count = selected_points(model)
    boxes = np.zeros((0, 3)) if settings.box_use == "part" else box_points(model)
    if settings.box_use != "all" and count == 0:
        operator.report({"ERROR"}, "No faces selected: find the part first (or select faces), "
                                   "or set Box to 'Everything in the box'")
        return None
    if count == 0 and len(boxes) == 0:
        operator.report({"ERROR"}, "Mark an area: select faces (Edit Mode) or add a space box")
        return None
    if settings.box_use == "part_space":  # grown here already, around the protected surfaces: the job grows 0
        return model, keep_the_rest(model, faces, boxes, settings.grow, settings.room), count, True, 0
    return model, to_gltf(np.concatenate([faces, boxes])), count, len(boxes) > 0, settings.grow


def export_model(context, model, folder):
    """The model as .glb (only the model, not the boxes) and its file name."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{bpy.path.clean_name(model.name)}.glb")
    for other in context.view_layer.objects:
        other.select_set(False)
    model.select_set(True)
    context.view_layer.objects.active = model
    bpy.ops.export_scene.gltf(filepath=path, export_format="GLB", use_selection=True,
                              export_image_format="AUTO", export_materials="EXPORT")
    return path


def server_ready(operator):
    server = ask("/status")
    if server is None:
        operator.report({"ERROR"}, "The inpaint server isn't running: python scripts/server/inpaint_server.py")
        return False
    if server["busy"]:
        operator.report({"ERROR"}, "The server is busy with another job")
        return False
    return True


def start(operator, path, job, info):
    answer = ask(path, job, timeout=5)
    if not answer or not answer.get("started"):
        operator.report({"ERROR"}, f"The server didn't start the job: {answer}")
        return {"CANCELLED"}
    status["job"] = info
    status["message"] = "started"
    bpy.app.timers.register(follow_job, first_interval=1.0)
    return {"FINISHED"}


class InpaintClick(bpy.types.PropertyGroup):
    point: bpy.props.FloatVectorProperty(name="Point", size=3, subtype="XYZ")


class InpaintSettings(bpy.types.PropertyGroup):
    picture: bpy.props.StringProperty(name="Picture", subtype="FILE_PATH",
                                      description="Picture that guides the new area: the model's own picture, "
                                                  "or the one made from the prompt")
    prompt: bpy.props.StringProperty(name="Prompt", description="What the marked area should show, "
                                                                "e.g. 'a stone statue of a kneeling woman'")
    reference: bpy.props.StringProperty(name="Reference", subtype="FILE_PATH",
                                        description="Optional: a picture of the thing to add")
    seed: bpy.props.IntProperty(name="Seed", default=1, min=0, description="Another seed gives another result")
    grow: bpy.props.IntProperty(name="Grow", default=2, min=0, max=8,
                                description="Grow the marked area by this many voxels (of 64 per side)")
    steps: bpy.props.IntProperty(name="Steps", default=12, min=4, max=50, description="Sampling steps per stage")
    room: bpy.props.FloatProperty(name="Room above", default=0.0, min=0.0, max=0.5, step=5,
                                  description="Share of TRELLIS.2's space kept free above the model, for a new part "
                                              "on top of it (e.g. 0.2 for a crown on a head that touches the top)")
    method: bpy.props.EnumProperty(
        name="Method",
        items=[("repaint", "Repaint", "The rest is kept with fresh noise at every step (fast)"),
               ("inversion", "Inversion", "The rest follows the original's own path and the new area attends to "
                                          "exactly the original around it: joins better, about 3x slower")],
        default="repaint")
    model: bpy.props.PointerProperty(name="Model", type=bpy.types.Object, description="The model being changed")
    connection: bpy.props.EnumProperty(
        name="Connection",
        items=[("DIRECT", "Inside the network", "Directly to the inpaint server: on this computer, or on the uni "
                                                "GPU server from inside the uni network (address in the add-on preferences)"),
               ("CLOUD", "Outside (cloud)", "Through the Sciebo cloud to the uni GPU server, from anywhere "
                                            "(cloud_worker.py must run there)")],
        default="DIRECT")
    part: bpy.props.StringProperty(name="Part", description="What to find in the space box, in English "
                                                            "(e.g. helmet, sword, left wing)")
    clicks: bpy.props.CollectionProperty(type=InpaintClick, description="Spots on the part, for Find part")
    find_cpu: bpy.props.BoolProperty(name="CPU only", default=False,
                                     description="Run the mask AI without the GPU (slower: about 1-2 minutes)")
    box_use: bpy.props.EnumProperty(
        name="Box",
        items=[("all", "Everything in the box", "Everything inside the space box is regenerated"),
               ("part", "Only the selected faces", "The box only frames the part: just the selected faces (e.g. "
                                                   "the part the AI found) are regenerated, in their own place"),
               ("part_space", "Selected faces + empty space",
                "The selected faces and the empty space in the box are regenerated; every other surface in the "
                "box stays exactly as it is (no colour bleeding onto it). For replacing a part with something new")],
        default="all")
    folder: bpy.props.StringProperty(name="Work folder", subtype="DIR_PATH",
                                     default=os.path.expanduser("~/inpainting-3D/outputs/blender"),
                                     description="Where the model, the area, the pictures and the results are saved")


class INPAINT_OT_add_box(bpy.types.Operator):
    bl_idname = "inpaint.add_box"
    bl_label = "Add space box"
    bl_description = ("Add a box that marks space where something new may grow "
                      "(a side: drag its arrow; a corner: drag its dot; move: G, size: S, turn: R)")

    def execute(self, context):
        model = target_model(context)
        size = max(model.dimensions) if model is not None else 1.0
        if context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        center = context.scene.cursor.location.copy()
        if model is not None:
            points, count = selected_points(model)
            if count:  # put it on the selected faces
                center = Vector(points.mean(0))
        bpy.ops.mesh.primitive_cube_add(size=2, location=center)
        box = context.active_object
        box.name = BOX
        box.scale = (0.08 * size,) * 3
        box.display_type = "WIRE"
        box.show_in_front = True
        box.hide_render = True
        self.report({"INFO"}, "Space box added: drag an arrow to move a side, a dot to move a corner; "
                              "G moves it, R turns it")
        return {"FINISHED"}


def box_side(box, axis, side):
    """One side of a space box (axis 0/1/2 = its X/Y/Z, side +1/-1), in world space: the middle of the
    opposite side (which stays put while this one is dragged), the outward direction and their distance."""
    corners = box_corners(box)
    this = sum((c for c, s in zip(corners, CORNER_SIGNS) if s[axis] == side), Vector()) / 4
    other = sum((c for c, s in zip(corners, CORNER_SIGNS) if s[axis] != side), Vector()) / 4
    if (this - other).length < 1e-9:  # squashed flat: push out along the box's own axis
        return other, box.matrix_basis.to_3x3().col[axis].normalized() * side, 0.0
    return other, (this - other).normalized(), (this - other).length


def set_box_length(box, axis, side, length):
    """Moves one side of a space box (its 4 corners) to `length` from the opposite side, which stays put."""
    _, outward, old = box_side(box, axis, side)
    move = box.matrix_basis.inverted().to_3x3() @ (outward * (max(length, 1e-4) - old))
    for i, signs in zip(CUBE_CORNERS, CORNER_SIGNS):
        if signs[axis] == side:
            box.data.vertices[i].co += move
    box.data.update()


def set_box_corner(box, corner, position):
    """Puts one corner of a space box (bound_box order) at `position` (world)."""
    box.data.vertices[CUBE_CORNERS[corner]].co = box.matrix_basis.inverted() @ Vector(position)
    box.data.update()


def draw_boxes():
    """Every visible space box as see-through blue with bold edges, so its space can be seen without
    turning the view: stronger where the box is in front of the model, fainter behind it. And the point
    the user clicked on for "Find part", as red dots."""
    settings = bpy.context.scene.inpaint
    boxes = [o for o in bpy.context.view_layer.objects if is_box(o) and o.visible_get()]
    model = bpy.data.objects.get(finder["part_model"] or "")
    part = model is not None and bpy.context.mode == "OBJECT" and model.visible_get() and len(finder["part"]) > 0
    if not boxes and not settings.clicks and not part:
        return
    before = gpu.state.blend_get(), gpu.state.depth_test_get(), gpu.state.depth_mask_get()
    gpu.state.blend_set("ALPHA")
    gpu.state.depth_mask_set(False)
    fill = gpu.shader.from_builtin("UNIFORM_COLOR")
    if part:  # the found part, red on the model
        if finder["part_batch"] is None:
            finder["part_batch"] = batch_for_shader(fill, "TRIS", {"pos": finder["part"]})
        gpu.state.depth_test_set("LESS_EQUAL")
        fill.uniform_float("color", (1.0, 0.15, 0.1, 0.6))
        finder["part_batch"].draw(fill)
    edge = gpu.shader.from_builtin("POLYLINE_UNIFORM_COLOR")
    edge.uniform_float("viewportSize", gpu.state.viewport_get()[2:])
    edge.uniform_float("lineWidth", 2.0)
    for box in boxes:
        corners = box_corners(box)
        strength = 1.5 if box.select_get() else 1.0
        sides = batch_for_shader(fill, "TRIS", {"pos": corners}, indices=BOX_TRIANGLES)
        for depth, alpha in (("NONE", 0.07), ("LESS_EQUAL", 0.15)):  # behind the model / in front of it
            gpu.state.depth_test_set(depth)
            fill.uniform_float("color", (*BOX_COLOR, alpha * strength))
            sides.draw(fill)
        gpu.state.depth_test_set("NONE")
        edge.uniform_float("color", (*BOX_COLOR, 0.9))
        batch_for_shader(edge, "LINES", {"pos": corners}, indices=BOX_EDGES).draw(edge)
    if settings.clicks:
        gpu.state.depth_test_set("NONE")
        dot = gpu.shader.from_builtin("POINT_UNIFORM_COLOR")
        gpu.state.point_size_set(12.0)
        dot.uniform_float("color", (1.0, 0.15, 0.1, 1.0))
        batch_for_shader(dot, "POINTS", {"pos": [tuple(c.point) for c in settings.clicks]}).draw(dot)
        gpu.state.point_size_set(1.0)
    gpu.state.blend_set(before[0])
    gpu.state.depth_test_set(before[1])
    gpu.state.depth_mask_set(before[2])


class INPAINT_GGT_box_sides(bpy.types.GizmoGroup):
    """An arrow on each of the six sides of the selected space box: dragging one moves that side only,
    like the edge of a window (the opposite side stays where it is). And a dot on each of its eight
    corners: dragging one moves that corner anywhere (across the view), so the box can take any shape."""
    bl_idname = "INPAINT_GGT_box_sides"
    bl_label = "Space box sides"
    bl_space_type = "VIEW_3D"
    bl_region_type = "WINDOW"
    bl_options = {"3D", "PERSISTENT"}

    @classmethod
    def poll(cls, context):
        box = context.active_object
        return (context.mode == "OBJECT" and is_box(box) and box.select_get() and box.visible_get()
                and len(box.data.vertices) == 8)

    def setup(self, context):
        self.sides = []
        for axis in range(3):
            for side in (1, -1):
                arrow = self.gizmos.new("GIZMO_GT_arrow_3d")
                arrow.target_set_handler("offset", get=self.length_of(axis, side), set=self.resize(axis, side))
                arrow.color, arrow.alpha = BOX_COLOR, 0.9
                arrow.color_highlight, arrow.alpha_highlight = (1.0, 1.0, 1.0), 1.0
                arrow.scale_basis = 0.7
                self.sides.append((arrow, axis, side))
        for corner in range(8):
            dot = self.gizmos.new("GIZMO_GT_move_3d")  # drawn at its offset: the corner's world position
            dot.target_set_handler("offset", get=self.position_of(corner), set=self.move_corner(corner))
            dot.draw_options = {"FILL", "ALIGN_VIEW"}
            dot.color, dot.alpha = BOX_COLOR, 0.9
            dot.color_highlight, dot.alpha_highlight = (1.0, 1.0, 1.0), 1.0
            dot.scale_basis = 0.12

    @staticmethod
    def position_of(corner):
        def get():
            box = bpy.context.active_object
            return tuple(box_corners(box)[corner]) if is_box(box) else (0.0, 0.0, 0.0)
        return get

    @staticmethod
    def move_corner(corner):
        def set(position):
            box = bpy.context.active_object
            if is_box(box):
                set_box_corner(box, corner, position)
        return set

    @staticmethod
    def length_of(axis, side):
        # the arrow sits at the opposite side, pushed out by the box's length: so it shows on its own side
        def get():
            box = bpy.context.active_object
            return box_side(box, axis, side)[2] if is_box(box) else 0.0
        return get

    @staticmethod
    def resize(axis, side):
        def set(length):
            box = bpy.context.active_object
            if is_box(box):
                set_box_length(box, axis, side, length)
        return set

    def draw_prepare(self, context):
        box = context.active_object
        for arrow, axis, side in self.sides:
            fixed, outward, _ = box_side(box, axis, side)
            arrow.matrix_basis = Matrix.Translation(fixed) @ outward.to_track_quat("Z", "Y").to_matrix().to_4x4()


# ---------------------------------------------------------------- finding a part in the box (the mask AI)
# Blender takes pictures of the space box from the directions that show it best, scripts/mask/mask_ai.py (in its
# own environment, made by install_mask.sh) finds the part in each, and every face in the box is voted on
# by the pictures that see it.

VIEW_COUNT = 16      # pictures per search ...
CANDIDATES = 64      # ... the best out of this many directions all around the box
PICTURE_SIZE = 1024  # pixels of the pictures for the mask AI
CHECK_SIZE = 256     # pixels of the depth-only pictures that find the best directions
finder = {"process": None, "message": "", "model": None, "folder": None, "faces": None, "centers": None,
          "views": None, "log": [], "part_model": None, "part": None, "part_batch": None, "mode_before": None}


def mask_ai():
    """The mask AI's Python and script, in the project folder from the add-on preferences."""
    project = os.path.expanduser(bpy.context.preferences.addons[__name__].preferences.project)
    return (os.path.join(project, "conda", "envs", "mask-ai", "bin", "python"),
            os.path.join(project, "scripts", "mask", "mask_ai.py"))


def face_centers(model):
    mesh = model.data
    centers = np.zeros(len(mesh.polygons) * 3, np.float32)
    mesh.polygons.foreach_get("center", centers)
    m = np.array(model.matrix_world)
    return centers.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3]


def inside_box(points, corners):
    """Which points lie inside a box given by its 8 corners (bound_box order; any convex shape)."""
    middle = corners.mean(0)
    inside = np.ones(len(points), bool)
    for a, b, c in BOX_TRIANGLES:
        normal = np.cross(corners[b] - corners[a], corners[c] - corners[a])
        if normal @ (middle - corners[a]) > 0:  # make it point outward
            normal = -normal
        inside &= (points - corners[a]) @ normal <= 1e-9
    return inside


def project(points, view, size=PICTURE_SIZE):
    """Pixel x, y (from the top left) and depth of world points in one picture (orthographic camera)."""
    m = view["world_to_camera"]
    p = points @ m[:3, :3].T + m[:3, 3]
    return (p[:, 0] / view["scale"] + 0.5) * size, (0.5 - p[:, 1] / view["scale"]) * size, -p[:, 2]


def front_depth(x, y, depth, size=PICTURE_SIZE):
    """Depth of the nearest surface in every pixel (from the face centres; gaps closed by the neighbours)."""
    inside = (x >= 0) & (x < size) & (y >= 0) & (y < size)
    z = np.full(size * size, np.inf)
    np.minimum.at(z, y[inside].astype(np.int64) * size + x[inside].astype(np.int64), depth[inside])
    padded = np.pad(z.reshape(size, size), 1, constant_values=np.inf)
    return np.min([padded[dy:dy + size, dx:dx + size] for dy in range(3) for dx in range(3)], axis=0)


def seen(x, y, depth, front, tolerance, size=PICTURE_SIZE):
    """Which points are seen in the picture (not hidden behind something nearer), and their pixels."""
    xi = np.clip(x.astype(np.int64), 0, size - 1)
    yi = np.clip(y.astype(np.int64), 0, size - 1)
    inside = (x >= 0) & (x < size) & (y >= 0) & (y < size)
    return inside & (depth <= front[yi, xi] + tolerance), xi, yi


def directions_around(count):
    """`count` directions spread evenly over the sphere (a Fibonacci spiral from top to bottom)."""
    i = np.arange(count) + 0.5
    z = 1 - 2 * i / count
    angle = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.sqrt(1 - z ** 2) * np.sin(angle), -np.sqrt(1 - z ** 2) * np.cos(angle), z], 1)


def camera_for(direction, corners, distance):
    """An orthographic camera looking at the box from `direction`, framing it tightly: its location, its
    rotation and the view (world_to_camera, scale) for project()."""
    toward = Vector(direction).normalized()
    location = Vector(corners.mean(0)) + toward * distance
    rotation = (-toward).to_track_quat("-Z", "Y")
    world_to_camera = np.array((Matrix.Translation(location) @ rotation.to_matrix().to_4x4()).inverted())
    seen_corners = corners @ world_to_camera[:3, :3].T
    scale = 1.1 * 2 * float(np.abs(seen_corners[:, :2] - seen_corners[:, :2].mean(0)).max())
    return location, rotation, {"world_to_camera": world_to_camera, "scale": scale}


def best_directions(centers, faces, corners, distance):
    """The VIEW_COUNT directions, out of CANDIDATES all around, that show the faces in the box best: one
    after the other, each the one that sees the most of what the chosen ones saw least. So the pictures
    show the part from all its sides, past whatever else of the model stands in front of it."""
    rng = np.random.default_rng(0)
    occluders = centers[::max(1, len(centers) // 1_000_000)]  # enough for depths at CHECK_SIZE
    sample = centers[rng.choice(faces, min(len(faces), 50_000), replace=False)]
    candidates = directions_around(CANDIDATES)
    seen_by = np.zeros((len(candidates), len(sample)), np.float32)
    for i, direction in enumerate(candidates):
        view = camera_for(direction, corners, distance)[2]
        front = front_depth(*project(occluders, view, CHECK_SIZE), CHECK_SIZE)
        seen_by[i] = seen(*project(sample, view, CHECK_SIZE), front, 3 * view["scale"] / CHECK_SIZE, CHECK_SIZE)[0]
    chosen, times = [], np.zeros(len(sample), np.float32)
    for _ in range(VIEW_COUNT):
        score = seen_by @ (1 / (1 + times))  # faces seen less often so far count more
        score[chosen] = -1
        chosen.append(int(score.argmax()))
        times += seen_by[chosen[-1]]
    return candidates[chosen]


def take_pictures(context, model, corners, folder, directions, distance):
    """Renders the model around the box corners from every direction (Workbench, with its colours,
    orthographic, transparent background), restoring the scene's render settings afterwards."""
    scene = context.scene
    camera = bpy.data.objects.new("InpaintFindCamera", bpy.data.cameras.new("InpaintFindCamera"))
    camera.data.type = "ORTHO"
    camera.data.clip_start, camera.data.clip_end = distance * 1e-3, distance * 3
    scene.collection.objects.link(camera)
    render, shading = scene.render, scene.display.shading
    changes = [(render, "engine", "BLENDER_WORKBENCH"), (render, "resolution_x", PICTURE_SIZE),
               (render, "resolution_y", PICTURE_SIZE), (render, "resolution_percentage", 100),
               (render, "pixel_aspect_x", 1.0), (render, "pixel_aspect_y", 1.0), (render, "film_transparent", True),
               (render, "use_border", False), (render, "use_compositing", False), (render, "use_sequencer", False),
               (render, "filepath", render.filepath), (render.image_settings, "file_format", "PNG"),
               (render.image_settings, "color_mode", "RGBA"), (render.image_settings, "color_depth", "8"),
               (scene.display, "render_aa", "8"), (shading, "light", "STUDIO"), (shading, "color_type", "TEXTURE"),
               (shading, "show_cavity", False), (shading, "show_shadows", False),
               (scene.view_settings, "view_transform", "Standard"), (scene.view_settings, "look", "None"),
               (scene.view_settings, "exposure", 0.0), (scene.view_settings, "gamma", 1.0),
               (scene, "camera", camera)]
    changes += [(o, "hide_render", True) for o in scene.objects if o not in (model, camera)]
    before = [(thing, name, getattr(thing, name)) for thing, name, _ in changes]
    views = []
    try:
        for thing, name, value in changes:
            setattr(thing, name, value)
        for i, direction in enumerate(directions):
            camera.location, rotation, view = camera_for(direction, corners, distance)
            camera.rotation_euler = rotation.to_euler()
            camera.data.ortho_scale = view["scale"]
            render.filepath = os.path.join(folder, f"view_{i:02d}.png")
            bpy.ops.render.render(write_still=True)
            views.append({"name": f"view_{i:02d}", **view})
    finally:
        for thing, name, value in reversed(before):
            setattr(thing, name, value)
        bpy.data.objects.remove(camera)
    return views


def select_faces(model, faces):
    """Makes exactly these faces (a yes/no per face) the model's selection, with their edges and vertices."""
    mesh = model.data
    totals = np.zeros(len(mesh.polygons), np.int32)
    mesh.polygons.foreach_get("loop_total", totals)
    loop_vertices = np.zeros(len(mesh.loops), np.int32)
    mesh.loops.foreach_get("vertex_index", loop_vertices)
    loop_edges = np.zeros(len(mesh.loops), np.int32)
    mesh.loops.foreach_get("edge_index", loop_edges)
    chosen = np.repeat(faces, totals)  # loops are stored face after face
    vertices = np.zeros(len(mesh.vertices), bool)
    vertices[loop_vertices[chosen]] = True
    edges = np.zeros(len(mesh.edges), bool)
    edges[loop_edges[chosen]] = True
    mesh.polygons.foreach_set("select", faces)
    mesh.edges.foreach_set("select", edges)
    mesh.vertices.foreach_set("select", vertices)
    mesh.update()


class INPAINT_OT_pick_point(bpy.types.Operator):
    bl_idname = "inpaint.pick_point"
    bl_label = "Click on it"
    bl_description = ("Then click on the part in the 3D view (e.g. on the helmet), on as many spots as you like: "
                      "everything clicked counts as one part (e.g. the foot and the arm of a leg). Enter: done")

    def invoke(self, context, event):
        if target_model(context) is None:
            self.report({"ERROR"}, "Import the model first")
            return {"CANCELLED"}
        context.window.cursor_modal_set("EYEDROPPER")
        context.workspace.status_text_set("Click on the part you mean, on as many spots as you like "
                                          "(turning the view works as usual). Enter or right-click: done")
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def finish(self, context):
        context.window.cursor_modal_restore()
        context.workspace.status_text_set(None)
        redraw()

    def modal(self, context, event):
        if event.type in {"RET", "NUMPAD_ENTER", "ESC", "RIGHTMOUSE"} and event.value == "PRESS":
            self.finish(context)
            return {"FINISHED"}
        if event.type != "LEFTMOUSE" or event.value != "PRESS":
            return {"PASS_THROUGH"}
        for area in context.window.screen.areas:
            if area.type != "VIEW_3D":
                continue
            for region in area.regions:
                if (region.type == "WINDOW" and region.x <= event.mouse_x < region.x + region.width
                        and region.y <= event.mouse_y < region.y + region.height):
                    return self.pick(context, region, (event.mouse_x - region.x, event.mouse_y - region.y))
        return {"PASS_THROUGH"}  # e.g. a click in the sidebar

    def pick(self, context, region, mouse):
        from bpy_extras import view3d_utils
        model = target_model(context)
        origin = view3d_utils.region_2d_to_origin_3d(region, region.data, mouse)
        direction = view3d_utils.region_2d_to_vector_3d(region, region.data, mouse)
        to_model = model.matrix_world.inverted()
        hit, location, _, _ = model.ray_cast(to_model @ origin, to_model.to_3x3() @ direction)
        if not hit:
            self.report({"WARNING"}, "That wasn't on the model: click on the part itself")
            return {"RUNNING_MODAL"}
        context.scene.inpaint.clicks.add().point = model.matrix_world @ location
        redraw()
        return {"RUNNING_MODAL"}  # more spots, until Enter


class INPAINT_OT_clear_click(bpy.types.Operator):
    bl_idname = "inpaint.clear_click"
    bl_label = "Forget the clicks"
    bl_description = "Find the part by its word only"

    def execute(self, context):
        context.scene.inpaint.clicks.clear()
        redraw()
        return {"FINISHED"}


class INPAINT_OT_find_part(bpy.types.Operator):
    bl_idname = "inpaint.find_part"
    bl_label = "Find part in box"
    bl_description = ("Take pictures of the space box from the 16 sides that show it best, let the mask AI "
                      "find the part in each "
                      "(the word, and what you clicked on), and select the faces in the box most pictures agree on")

    @classmethod
    def poll(cls, context):
        return finder["process"] is None

    def execute(self, context):
        settings = context.scene.inpaint
        python, script = mask_ai()
        if not os.path.isfile(python):
            self.report({"ERROR"}, "The mask AI isn't installed: run ./install_mask.sh in the project folder "
                                   "(add-on preferences: Project folder)")
            return {"CANCELLED"}
        model = target_model(context)
        boxes = [o for o in context.view_layer.objects if is_box(o) and o.visible_get()]
        word = settings.part.strip()
        if model is None or not boxes:
            self.report({"ERROR"}, "Import the model and put a space box around the part first")
            return {"CANCELLED"}
        if not word and not settings.clicks:
            self.report({"ERROR"}, "Type what to find (e.g. helmet), or click on it")
            return {"CANCELLED"}
        if context.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        corners = [np.array([tuple(c) for c in box_corners(b)]) for b in boxes]
        centers = face_centers(model)
        in_box = np.zeros(len(centers), bool)
        for c in corners:
            in_box |= inside_box(centers, c)
        if not in_box.any():
            self.report({"ERROR"}, "No part of the model is inside the space box")
            return {"CANCELLED"}
        folder = os.path.join(bpy.path.abspath(settings.folder), f"{bpy.path.clean_name(model.name)}_find")
        os.makedirs(folder, exist_ok=True)
        for name in os.listdir(folder):
            os.remove(os.path.join(folder, name))
        all_corners = np.concatenate(corners)
        distance = float(np.linalg.norm(all_corners - all_corners.mean(0), axis=1).max()) + 2 * max(model.dimensions)
        faces = np.flatnonzero(in_box)
        directions = best_directions(centers, faces, all_corners, distance)
        views = take_pictures(context, model, all_corners, folder, directions, distance)
        clicked = np.array([tuple(c.point) for c in settings.clicks]).reshape(-1, 3)
        asked = []
        for view in views:
            x, y, depth = project(centers, view)
            front = front_depth(x, y, depth)
            tolerance = 6 * view["scale"] / PICTURE_SIZE  # a few pixels: the surface itself, slanted
            view["seen"], view["x"], view["y"] = seen(x[faces], y[faces], depth[faces], front, tolerance)
            cx, cy, cdepth = project(clicked, view)
            visible = seen(cx, cy, cdepth, front, tolerance)[0]
            clicks = [[float(x_), float(y_)] for x_, y_ in zip(cx[visible], cy[visible])]
            if word or clicks:  # a picture without the word and without a click can't tell anything
                asked.append({"picture": view["name"] + ".png", "clicks": clicks})
            else:
                os.remove(os.path.join(folder, view["name"] + ".png"))
        if not asked:
            self.report({"ERROR"}, "The clicks can't be seen from around the box: click on the part again")
            return {"CANCELLED"}
        with open(os.path.join(folder, "views.json"), "w") as f:
            json.dump({"word": word, "views": asked}, f, indent=2)
        command = [python, "-u", script, folder] + (["--cpu"] if settings.find_cpu else [])
        environment = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
        finder.update(model=model.name, folder=folder, faces=faces, centers=centers[faces], views=views, log=[],
                      message=f"mask AI: looking at {len(asked)} pictures...",
                      process=subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                               text=True, env=environment))

        def read_progress(process):
            for line in process.stdout:
                finder["log"].append(line.rstrip())
                finder["message"] = "mask AI: " + line.strip()[:100]
        threading.Thread(target=read_progress, args=(finder["process"],), daemon=True).start()
        bpy.app.timers.register(follow_find, first_interval=1.0)
        return {"FINISHED"}


def follow_find():
    """Timer: wait for the mask AI; then let the pictures vote on every face in the box and select the part."""
    process = finder["process"]
    if process.poll() is None:
        redraw()
        return 1.0
    finder["process"] = None
    masks_file = os.path.join(finder["folder"], "masks.npz")
    model = bpy.data.objects.get(finder["model"])
    if process.returncode != 0 or not os.path.isfile(masks_file) or model is None:
        finder["message"] = "mask AI failed: " + " / ".join(finder["log"][-2:])[:150]
        redraw()
        return None
    view = view3d_area()
    if view is not None and bpy.context.mode != "OBJECT":
        with bpy.context.temp_override(**view):  # leaving Edit Mode would overwrite the new selection
            bpy.ops.object.mode_set(mode="OBJECT")
    found = vote(model, np.load(masks_file))
    bpy.context.scene.inpaint.box_use = "part_space"
    finder["message"] = f"found {found:,} faces: red (Tab to fix them by hand)"
    for o in bpy.context.view_layer.objects:  # the model active, so Tab opens it (not the box)
        o.select_set(False)
    model.select_set(True)
    bpy.context.view_layer.objects.active = model
    show_part(model)
    redraw()
    return None


def show_part(model):
    """Remembers the model's selected faces as red triangles, drawn by draw_boxes() in Object Mode (in Edit
    Mode, a model with millions of faces is all black edges). Slightly lifted off the surface, so they
    don't flicker with it."""
    mesh = model.data
    mesh.calc_loop_triangles()
    selected = np.zeros(len(mesh.polygons), bool)
    mesh.polygons.foreach_get("select", selected)
    owner = np.zeros(len(mesh.loop_triangles), np.int32)
    mesh.loop_triangles.foreach_get("polygon_index", owner)
    corners = np.zeros(len(mesh.loop_triangles) * 3, np.int32)
    mesh.loop_triangles.foreach_get("vertices", corners)
    corners = corners.reshape(-1, 3)[selected[owner]]
    co = np.zeros(len(mesh.vertices) * 3, np.float32)
    mesh.vertices.foreach_get("co", co)
    m = np.array(model.matrix_world, np.float32)
    triangles = (co.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3])[corners]  # (n, 3 corners, xyz)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    triangles += normals[:, None] * (2e-4 * max(model.dimensions))
    finder.update(part_model=model.name, part=triangles.reshape(-1, 3), part_batch=None)


@bpy.app.handlers.persistent
def after_change(scene, depsgraph):
    """Back in Object Mode after fixing the found part by hand: show the new selection in red."""
    model = bpy.data.objects.get(finder["part_model"] or "")
    mode = bpy.context.mode
    if model is not None and mode == "OBJECT" and finder["mode_before"] == "EDIT_MESH":
        show_part(model)
    finder["mode_before"] = mode


def view3d_area():
    """A window, 3D view and its main region, for operators started from a timer (None without a window)."""
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                region = next(r for r in area.regions if r.type == "WINDOW")
                return {"window": window, "area": area, "region": region}
    return None


def vote(model, masks):
    """Selects the faces in the box that most pictures show as the part. The votes are pooled in 3D: every
    face counts the yes and no votes of everything around it, so faces no picture sees well (inner sides,
    crevices) follow their neighbours, and a single wrong picture is outvoted. Then every face follows the
    majority around it (no speckles, no small holes)."""
    faces, centers = finder["faces"], finder["centers"]
    yes = np.zeros(len(faces), np.float32)
    seen_by = np.zeros(len(faces), np.float32)
    for view in finder["views"]:
        if view["name"] in masks:  # pictures where nothing was found don't vote
            seen_by += view["seen"]
            yes += view["seen"] & masks[view["name"]][view["y"], view["x"]]
    cell = np.ptp(centers, 0).max() / 80
    part = np.zeros(len(faces), bool)
    open_ = np.ones(len(faces), bool)  # not decided yet: no votes around at this size
    for size in (cell, 2 * cell, 4 * cell, 8 * cell):
        yes_around, seen_around = sum_around(centers, [yes, seen_by], size)
        decided = open_ & (seen_around > 0)
        part[decided] = 2 * yes_around[decided] > seen_around[decided]
        open_ &= ~decided
    total_around, part_around = sum_around(centers, [np.ones(len(faces), np.float32), part], cell)
    part = 2 * part_around >= total_around
    chosen = np.zeros(len(model.data.polygons), bool)
    chosen[faces[part]] = True
    select_faces(model, chosen)
    return int(chosen.sum())


def sum_around(points, values, cell):
    """For every point, the sums of each of `values` over the points in its 3x3x3 neighbourhood of cells."""
    keys = np.floor((points - points.min(0)) / cell).astype(np.int64) + 1
    shape = keys.max(0) + 2
    index = np.ravel_multi_index(keys.T, shape)
    inner = tuple(slice(1, n - 1) for n in shape)
    around = [tuple(slice(1 + d, n - 1 + d) for d, n in zip(offset, shape))
              for offset in np.ndindex(3, 3, 3) for offset in [np.array(offset) - 1]]
    sums = []
    for value in values:
        grid = np.bincount(index, weights=value, minlength=int(shape.prod())).reshape(shape)
        total = np.zeros_like(grid)
        total[inner] = sum(grid[s] for s in around)
        sums.append(total.ravel()[index])
    return sums


class INPAINT_OT_show_found(bpy.types.Operator):
    bl_idname = "inpaint.show_found"
    bl_label = "Show what the AI saw"
    bl_description = "Open the pictures the mask AI looked at, with what it found in red"

    def execute(self, context):
        path = os.path.join(finder["folder"] or "", "overview.jpg")
        if not os.path.isfile(path):
            self.report({"ERROR"}, "Find a part first")
            return {"CANCELLED"}
        open_image_window(path)
        return {"FINISHED"}


def open_image_window(path):
    image = bpy.data.images.get(os.path.basename(path))
    if image is None:
        image = bpy.data.images.load(path)
    else:
        image.reload()
    bpy.ops.wm.window_new()
    area = bpy.context.window_manager.windows[-1].screen.areas[0]
    area.type = "IMAGE_EDITOR"
    area.spaces.active.image = image


class INPAINT_OT_make_picture(bpy.types.Operator):
    bl_idname = "inpaint.make_picture"
    bl_label = "1. Make picture from prompt"
    bl_description = ("Picture the model from the current view and paint the prompt into the marked area "
                      "(Qwen-Image-Edit in ComfyUI); the result becomes the Picture")

    @classmethod
    def poll(cls, context):
        return status["job"] is None and context.space_data is not None and context.space_data.type == "VIEW_3D"

    def execute(self, context):
        settings = context.scene.inpaint
        if not settings.prompt.strip():
            self.report({"ERROR"}, "Write a prompt first")
            return {"CANCELLED"}
        if settings.connection != "DIRECT" or not is_local():
            self.report({"ERROR"}, "Making a picture needs an inpaint server on this computer (with ComfyUI)")
            return {"CANCELLED"}
        reference = bpy.path.abspath(settings.reference) if settings.reference else ""
        if reference and not os.path.isfile(reference):
            self.report({"ERROR"}, "The reference picture doesn't exist")
            return {"CANCELLED"}
        if not server_ready(self):
            return {"CANCELLED"}
        area = area_points(self, context)
        if area is None:
            return {"CANCELLED"}
        model, points = area[:2]
        # direction from the model towards the viewer, as the view shows it now
        direction = context.space_data.region_3d.view_rotation @ Vector((0, 0, 1))
        folder = bpy.path.abspath(settings.folder)
        path = export_model(context, model, folder)
        name = bpy.path.clean_name(model.name)
        np.save(os.path.join(folder, f"{name}_area.npy"), points)
        out = os.path.join(folder, f"{name}_picture_seed{settings.seed}.png")
        return start(self, "/picture", {"model": path, "points": os.path.join(folder, f"{name}_area.npy"), "out": out,
                                        "direction": to_gltf(direction)[0].tolist(), "prompt": settings.prompt,
                                        "reference": reference or None, "seed": settings.seed},
                     {"kind": "picture"})


class INPAINT_OT_regenerate(bpy.types.Operator):
    bl_idname = "inpaint.regenerate"
    bl_label = "Regenerate area"
    bl_description = ("Pack the model, the marked area and the Picture, send them the chosen way "
                      "(inside the network: directly; outside: through the cloud) and load the result")

    @classmethod
    def poll(cls, context):
        return status["job"] is None and remote["job"] is None

    def execute(self, context):
        if context.scene.inpaint.connection == "CLOUD":
            return bpy.ops.inpaint.send_job()
        if not server_ready(self):
            return {"CANCELLED"}
        out = pack_job(self, context)
        if out is None:
            return {"CANCELLED"}
        url, headers = server()
        remote.update(job="?", message="sending the job to the server...", error=None, results=None,
                      object=target_model(context).name, seed=context.scene.inpaint.seed)

        def upload():
            with open(out, "rb") as f:
                request = urllib.request.Request(url + "/job", data=f.read(), method="POST",
                                                 headers={"Content-Type": "application/zip", **headers})
            with urllib.request.urlopen(request, timeout=600) as response:
                remote["job"] = json.loads(response.read())["job"]
        in_background(upload)
        bpy.app.timers.register(follow_direct, first_interval=2.0)
        return {"FINISHED"}


def follow_direct():
    """Timer for a job sent directly: progress from /status; when it's done, fetch and load the result."""
    if remote["busy"] is not None:
        return 2.0  # still sending or fetching
    if remote["error"]:
        remote.update(message="failed: " + remote["error"], job=None)
        redraw()
        return None
    if remote["results"] is not None:
        load_result({"object": remote["object"], "seed": remote["seed"]}, remote["results"][0])
        remote.update(message=f"done: {os.path.basename(remote['results'][0])}", job=None)
        redraw()
        return None
    state = ask("/status", timeout=10)  # saving the model keeps the server busy: answers can be slow
    if state is None:
        remote["silent"] = remote.get("silent", 0) + 1
        if remote["silent"] < 6:
            return 10.0
        remote.update(message="lost the server", job=None)
        redraw()
        return None
    remote["silent"] = 0
    remote["message"] = state["message"][:120]
    if state["job"] != remote["job"] or state["busy"]:
        redraw()
        return 2.0
    if state["error"]:
        remote.update(message="failed: " + state["error"], job=None)
        redraw()
        return None
    url, headers = server()
    folder = bpy.path.abspath(bpy.context.scene.inpaint.folder)
    path = os.path.join(folder, f"{remote['object']}_{remote['job']}.glb")
    remote["message"] = "fetching the result..."

    def fetch():
        request = urllib.request.Request(f"{url}/result?job={remote['job']}", headers=headers)
        with urllib.request.urlopen(request, timeout=600) as response, open(path, "wb") as f:
            f.write(response.read())
        remote["results"] = [path]
    in_background(fetch)
    redraw()
    return 2.0


def follow_job():
    """Timer: show the server's progress; take over the result when it is done."""
    job = status["job"]
    state = ask("/status", timeout=10)  # saving the model keeps the server busy: answers can be slow
    if state is None:
        job["silent"] = job.get("silent", 0) + 1
        if job["silent"] < 6:
            return 10.0
        status.update(message="lost the server", job=None)
        redraw()
        return None
    job["silent"] = 0
    status["message"] = state["message"]
    if state["busy"]:
        redraw()
        return 1.0
    status["job"] = None
    if state["error"]:
        status["message"] = "failed: " + state["error"]
    elif state["result"] and job["kind"] == "picture":
        show_picture(state["result"])
    elif state["result"]:
        load_result(job, state["result"])
    redraw()
    return None


def show_picture(path):
    """The new picture becomes the Picture; its before/after preview is shown in the panel."""
    bpy.context.scene.inpaint.picture = path
    preview = os.path.splitext(path)[0] + "_preview.jpg"
    for p in (path, preview):
        image = bpy.data.images.get(os.path.basename(p))
        if image is None:
            image = bpy.data.images.load(p)
        else:
            image.reload()
        image.preview_ensure()
        if image.preview is not None:
            image.preview.reload()
    status["message"] = "picture ready: check it, then 2. Regenerate"


def load_result(job, path):
    before = set(bpy.data.objects)
    window = bpy.context.window_manager.windows[0]
    with bpy.context.temp_override(window=window):
        bpy.ops.import_scene.gltf(filepath=path)
    new = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
    original = bpy.data.objects.get(job["object"])
    for o in new:
        o.name = f"{job['object']}_seed{job['seed']}"
    if original is not None and new:
        original.hide_set(True)  # still there: the eye icon in the Outliner brings it back
    for box in bpy.data.objects:
        if is_box(box):
            box.hide_set(True)  # done with; un-hide in the Outliner to use them again
    status["message"] = f"done: {new[0].name if new else path}"


def pack_job(operator, context):
    """The model, the marked area, the Picture and the settings as one .zip; its path, or None."""
    settings = context.scene.inpaint
    picture = bpy.path.abspath(settings.picture)
    if not os.path.isfile(picture):
        operator.report({"ERROR"}, "Choose the picture first (or make one from a prompt)")
        return None
    area = area_points(operator, context)
    if area is None:
        return None
    model, points, _, _, grow = area
    folder = bpy.path.abspath(settings.folder)
    path = export_model(context, model, folder)
    name = bpy.path.clean_name(model.name)
    area_file = os.path.join(folder, f"{name}_area.npy")
    np.save(area_file, points)
    picture_name = "picture" + os.path.splitext(picture)[1].lower()
    job = {"model": "model.glb", "points": "area.npy", "picture": picture_name, "object": model.name,
           "seed": settings.seed, "grow": grow, "steps": settings.steps, "method": settings.method}
    if settings.room > 0:
        job["room"] = round(settings.room, 3)
    # a picture made from a prompt comes with the render before the edit and the edit's 2D mask
    before, picture_mask = (os.path.splitext(picture)[0] + end for end in ("_before.png", "_mask.png"))
    if os.path.isfile(before) and os.path.isfile(picture_mask):
        job.update(source="source.png", picture_mask="picture_mask.png")
    out = os.path.join(folder, f"{name}_job.zip")
    with zipfile.ZipFile(out, "w") as z:  # stored, not compressed: the model is compressed already
        z.write(path, "model.glb")
        z.write(area_file, "area.npy")
        z.write(picture, picture_name)
        if "source" in job:
            z.write(before, "source.png")
            z.write(picture_mask, "picture_mask.png")
        z.writestr("job.json", json.dumps(job, indent=2))
    return out


class INPAINT_OT_save_job(bpy.types.Operator):
    bl_idname = "inpaint.save_job"
    bl_label = "Save job for server"
    bl_description = ("Pack the model, the marked area, the Picture and the settings into one .zip, "
                      "to run on the uni GPU server with scripts/server/run_job.py")

    def execute(self, context):
        out = pack_job(self, context)
        if out is None:
            return {"CANCELLED"}
        status["message"] = f"job saved: {os.path.basename(out)}"
        self.report({"INFO"}, f"Saved {out}: upload it to the server and run scripts/server/run_job.py")
        return {"FINISHED"}


# ---------------------------------------------------------------- clean up (Geometry Nodes)

CLEANUP = "Clean up"


def cleanup_group():
    """The Geometry Nodes graph "Clean up", made once per .blend file: closes tiny gaps and deletes every
    loose piece with less than `Min share` of the model's faces (floating leftovers of TRELLIS.2).

        Group Input -> Merge by Distance -> Delete Geometry -> Group Output
                                              ^ Selection = "piece too small":
        Mesh Island -> Accumulate Field (faces per piece) / Accumulate Field (all faces) < Min share
    """
    if CLEANUP in bpy.data.node_groups:
        return bpy.data.node_groups[CLEANUP]
    tree = bpy.data.node_groups.new(CLEANUP, "GeometryNodeTree")
    tree.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    share = tree.interface.new_socket("Min share", in_out="INPUT", socket_type="NodeSocketFloat")
    share.default_value, share.min_value, share.max_value = 0.002, 0.0, 1.0  # 0.2 % of all faces
    gap = tree.interface.new_socket("Merge distance", in_out="INPUT", socket_type="NodeSocketFloat")
    gap.default_value, gap.min_value = 0.0001, 0.0
    tree.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

    nodes, link = tree.nodes, tree.links.new
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

    link(inp.outputs["Geometry"], merge.inputs["Geometry"])
    link(inp.outputs["Merge distance"], merge.inputs["Distance"])
    per_piece.inputs[0].default_value = 1  # count one per face ...
    link(island.outputs["Island Index"], per_piece.inputs[1])  # ... grouped by piece
    all_faces.inputs[0].default_value = 1
    link(per_piece.outputs["Total"], divide.inputs[0])
    link(all_faces.outputs["Total"], divide.inputs[1])
    link(divide.outputs[0], small.inputs[0])
    link(inp.outputs["Min share"], small.inputs[1])
    link(merge.outputs["Geometry"], delete.inputs["Geometry"])
    link(small.outputs["Result"], delete.inputs["Selection"])
    link(delete.outputs["Geometry"], out.inputs["Geometry"])
    return tree


def add_cleanup(obj):
    """Puts the "Clean up" modifier on a mesh object (once) and returns it. The model itself isn't
    changed: the modifier can be switched off, or applied (Ctrl+A over it) to keep the result."""
    modifier = obj.modifiers.get(CLEANUP) or obj.modifiers.new(CLEANUP, "NODES")
    modifier.node_group = cleanup_group()
    return modifier


def cleanup_input(modifier, name):
    """Where the modifier keeps its own value of one graph input, as (data, property) for layout.prop:
    Blender 5.x in modifier.properties.inputs, Blender 4.x as an ID property of the modifier."""
    ident = modifier.node_group.interface.items_tree[name].identifier
    inputs = getattr(getattr(modifier, "properties", None), "inputs", None)
    if inputs is not None and hasattr(inputs, ident):
        return getattr(inputs, ident), "value"
    return modifier, f'["{ident}"]'


def faces_after(obj):
    return len(obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).data.polygons)


class INPAINT_OT_clean_up(bpy.types.Operator):
    bl_idname = "inpaint.clean_up"
    bl_label = "Clean up model"
    bl_description = ("Geometry Nodes: delete small loose pieces (floating leftovers) of the model. "
                      "A modifier, so nothing is lost: switch it off to compare, Ctrl+A over it to keep it")

    def execute(self, context):
        model = target_model(context)
        if model is None:
            self.report({"ERROR"}, "Click the model first")
            return {"CANCELLED"}
        before = len(model.data.polygons)
        add_cleanup(model)
        after = faces_after(model)
        self.report({"INFO"}, f"{model.name}: {before - after} of {before} faces removed")
        return {"FINISHED"}


COMPARE = "Compare with original"
KEEP = "Keep original outside"
HEATMAP = "Change heatmap"


def proximity(tree, target, element="FACES"):
    """A Geometry Proximity node with `target` linked (its first input: "Geometry" in 5.x, "Target" in 4.x)."""
    node = tree.nodes.new("GeometryNodeProximity")
    node.target_element = element
    tree.links.new(target, node.inputs[0])
    return node


def original_info(tree):
    """Object Info of the "Original" input, in the space of the object the modifier is on."""
    node = tree.nodes.new("GeometryNodeObjectInfo")
    node.transform_space = "RELATIVE"
    return node


def heatmap_material():
    """Material that shows the "change" attribute: blue = like the original, red = new or moved."""
    material = bpy.data.materials.get(HEATMAP)
    if material is not None:
        return material
    material = bpy.data.materials.new(HEATMAP)
    material.use_nodes = True
    nodes, link = material.node_tree.nodes, material.node_tree.links.new
    bsdf = nodes.get("Principled BSDF")
    attribute = nodes.new("ShaderNodeAttribute")
    attribute.attribute_name, attribute.attribute_type = "change", "GEOMETRY"
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.1, 0.25, 0.9, 1)
    ramp.color_ramp.elements[1].color = (0.95, 0.1, 0.05, 1)
    attribute.location, ramp.location = (-600, 0), (-350, 0)
    link(attribute.outputs["Fac"], ramp.inputs["Fac"])
    link(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    return material


def compare_group():
    """Graph "Compare with original": stores on every point of the result how far it is from the original
    model ("change": 0 = on the original's surface, 1 = Threshold x 4 or further away) and, with Show,
    colours the model by it. The 3D version of comparing the pictures before and after.

        Group Input -> Store Named Attribute "change" -> Switch (Show) -> Group Output
                         ^ Geometry Proximity (to Original) / (Threshold x 4), clamped
                                                   Set Material (heatmap) ^
    """
    if COMPARE in bpy.data.node_groups:
        return bpy.data.node_groups[COMPARE]
    tree = bpy.data.node_groups.new(COMPARE, "GeometryNodeTree")
    tree.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    tree.interface.new_socket("Original", in_out="INPUT", socket_type="NodeSocketObject")
    threshold = tree.interface.new_socket("Threshold", in_out="INPUT", socket_type="NodeSocketFloat")
    threshold.default_value, threshold.min_value = 0.005, 0.0
    show = tree.interface.new_socket("Show", in_out="INPUT", socket_type="NodeSocketBool")
    show.default_value = True
    tree.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

    nodes, link = tree.nodes, tree.links.new
    inp, out = nodes.new("NodeGroupInput"), nodes.new("NodeGroupOutput")
    info = original_info(tree)
    link(inp.outputs["Original"], info.inputs["Object"])
    near = proximity(tree, info.outputs["Geometry"])
    scale = nodes.new("ShaderNodeMath")
    scale.operation = "MULTIPLY"
    scale.inputs[1].default_value = 4
    link(inp.outputs["Threshold"], scale.inputs[0])
    share = nodes.new("ShaderNodeMath")
    share.operation, share.use_clamp = "DIVIDE", True
    link(near.outputs["Distance"], share.inputs[0])
    link(scale.outputs[0], share.inputs[1])
    store = nodes.new("GeometryNodeStoreNamedAttribute")
    store.data_type, store.domain = "FLOAT", "POINT"
    store.inputs["Name"].default_value = "change"
    link(inp.outputs["Geometry"], store.inputs["Geometry"])
    link(share.outputs[0], store.inputs["Value"])
    paint = nodes.new("GeometryNodeSetMaterial")
    paint.inputs["Material"].default_value = heatmap_material()
    link(store.outputs["Geometry"], paint.inputs["Geometry"])
    switch = nodes.new("GeometryNodeSwitch")
    switch.input_type = "GEOMETRY"
    link(inp.outputs["Show"], switch.inputs["Switch"])
    link(store.outputs["Geometry"], switch.inputs["False"])
    link(paint.outputs["Geometry"], switch.inputs["True"])
    link(switch.outputs[0], out.inputs["Geometry"])
    for node, x, y in ((inp, -900, 0), (info, -900, -250), (near, -650, -250), (scale, -650, -450), (share, -400, -300),
                       (store, -150, 0), (paint, 100, -150), (switch, 350, 0), (out, 600, 0)):
        node.location = (x, y)
    return tree


def keep_group():
    """Graph "Keep original outside": the new part from the result, everything else from the original
    model itself, so outside the change the model is exactly the original (TRELLIS.2 reads the model in
    and writes it out again, which changes the kept surfaces a little).

        result faces further than Threshold from the original      (the new part)  --+
        original faces at most Threshold from the result      (what is still there)  --+-> Join -> Merge by Distance
    """
    if KEEP in bpy.data.node_groups:
        return bpy.data.node_groups[KEEP]
    tree = bpy.data.node_groups.new(KEEP, "GeometryNodeTree")
    tree.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    tree.interface.new_socket("Original", in_out="INPUT", socket_type="NodeSocketObject")
    threshold = tree.interface.new_socket("Threshold", in_out="INPUT", socket_type="NodeSocketFloat")
    threshold.default_value, threshold.min_value = 0.005, 0.0
    tree.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")

    nodes, link = tree.nodes, tree.links.new
    inp, out = nodes.new("NodeGroupInput"), nodes.new("NodeGroupOutput")
    info = original_info(tree)
    link(inp.outputs["Original"], info.inputs["Object"])

    def delete_where(geometry, target, far):
        """geometry without the faces that are (far=True: further, False: not further) than Threshold from target."""
        near = proximity(tree, target)
        compare = nodes.new("FunctionNodeCompare")
        compare.data_type, compare.operation = "FLOAT", "GREATER_THAN" if far else "LESS_EQUAL"
        link(near.outputs["Distance"], compare.inputs[0])
        link(inp.outputs["Threshold"], compare.inputs[1])
        delete = nodes.new("GeometryNodeDeleteGeometry")
        delete.domain = "FACE"
        link(geometry, delete.inputs["Geometry"])
        link(compare.outputs["Result"], delete.inputs["Selection"])
        return delete

    new_part = delete_where(inp.outputs["Geometry"], info.outputs["Geometry"], far=False)
    still_there = delete_where(info.outputs["Geometry"], inp.outputs["Geometry"], far=True)
    join = nodes.new("GeometryNodeJoinGeometry")
    link(still_there.outputs["Geometry"], join.inputs["Geometry"])
    link(new_part.outputs["Geometry"], join.inputs["Geometry"])
    merge = nodes.new("GeometryNodeMergeByDistance")
    merge.inputs["Distance"].default_value = 0.0001  # as Clean up: only closes the seams
    link(join.outputs["Geometry"], merge.inputs["Geometry"])
    link(merge.outputs["Geometry"], out.inputs["Geometry"])
    x = -900
    for node in tree.nodes:
        node.location = (x, 0 if node in (inp, join, merge, out) else -250)
        x += 180
    return tree


def find_original(context, result):
    """The original of a result: the other selected mesh, or the object the result's name came from
    (load_result names it <original>_seed<N>)."""
    others = [o for o in context.selected_objects if o.type == "MESH" and o is not result and not is_box(o)]
    if len(others) == 1:
        return others[0]
    if "_seed" in result.name:
        return bpy.data.objects.get(result.name.rsplit("_seed", 1)[0])
    return None


def add_graph(obj, name, group, original):
    modifier = obj.modifiers.get(name) or obj.modifiers.new(name, "NODES")
    modifier.node_group = group
    data, prop = cleanup_input(modifier, "Original")
    if prop == "value":
        data.value = original
    else:
        data[prop[2:-2]] = original
    return modifier


def changed_share(obj):
    """Share of the result's points that moved away from the original (change > 0.25, i.e. > Threshold)."""
    mesh = obj.evaluated_get(bpy.context.evaluated_depsgraph_get()).data
    attribute = mesh.attributes.get("change")
    if attribute is None or not len(attribute.data):
        return 0.0
    values = np.zeros(len(attribute.data), np.float32)
    attribute.data.foreach_get("value", values)
    return float((values > 0.25).mean())


class INPAINT_OT_compare(bpy.types.Operator):
    bl_idname = "inpaint.compare"
    bl_label = "Compare with original"
    bl_description = ("Geometry Nodes: colour the result by how far it is from the original model "
                      "(blue = unchanged, red = new). Click the result; the original is found by its name, "
                      "or select the original too")

    def execute(self, context):
        result = context.active_object
        original = find_original(context, result) if result is not None else None
        if result is None or original is None:
            self.report({"ERROR"}, "Click the result (and select its original too)")
            return {"CANCELLED"}
        add_graph(result, COMPARE, compare_group(), original)
        self.report({"INFO"}, f"{100 * changed_share(result):.1f} % of {result.name} differs from {original.name}")
        return {"FINISHED"}


class INPAINT_OT_keep_original(bpy.types.Operator):
    bl_idname = "inpaint.keep_original"
    bl_label = "Keep original outside"
    bl_description = ("Geometry Nodes: take only the new part from the result and everything else from the "
                      "original model, so the rest stays exactly the original")

    def execute(self, context):
        result = context.active_object
        original = find_original(context, result) if result is not None else None
        if result is None or original is None:
            self.report({"ERROR"}, "Click the result (and select its original too)")
            return {"CANCELLED"}
        if result.modifiers.get(COMPARE):
            result.modifiers.remove(result.modifiers[COMPARE])
        add_graph(result, KEEP, keep_group(), original)
        original.hide_set(True)
        self.report({"INFO"}, f"{result.name}: {faces_after(result)} faces (new part + original)")
        return {"FINISHED"}


# ---------------------------------------------------------------- the uni server through the Sciebo cloud
# scripts/server/cloud.py setup saves the cloud login; scripts/server/cloud_worker.py on the server runs the jobs.

CLOUD_CONFIG = os.path.expanduser("~/.config/inpainting3d/cloud.json")
CLOUD_FOLDER = "inpainting-jobs"
remote = {"job": None, "message": "", "busy": None, "error": None, "results": None, "object": None, "seed": None}


def cloud(method, name, data=None, timeout=600, config=None):
    """One WebDAV request to the cloud folder (runs in a background thread, never in Blender's own)."""
    if config is None:
        with open(CLOUD_CONFIG) as f:
            config = json.load(f)
    token = base64.b64encode(f"{config['user']}:{config['password']}".encode()).decode()
    path = f"{config.get('folder', CLOUD_FOLDER)}/{name}"
    url = f"{config['url'].rstrip('/')}/{urllib.parse.quote(path)}"
    headers = {"Authorization": f"Basic {token}", **({"Depth": "0"} if method == "PROPFIND" else {})}
    request = urllib.request.Request(url, data=data, method=method, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


class InpaintPreferences(bpy.types.AddonPreferences):
    """The cloud login, filled in once (Edit > Preferences > Add-ons > TRELLIS.2 Inpainting). The password
    is only kept in ~/.config/inpainting3d/cloud.json (readable only by you), not in Blender's settings."""
    bl_idname = __name__

    url: bpy.props.StringProperty(name="Cloud address",
                                  default="https://uni-duisburg-essen.sciebo.de/remote.php/webdav")
    user: bpy.props.StringProperty(name="Login name", description="From Sciebo: Sicherheit > App-Passwort")
    password: bpy.props.StringProperty(name="App password", subtype="PASSWORD",
                                       description="A Sciebo app password, not your uni password")
    folder: bpy.props.StringProperty(name="Job folder", default=CLOUD_FOLDER,
                                     description="The cloud folder the uni server watches (e.g. one shared with you)")
    server: bpy.props.StringProperty(name="Server address", default="127.0.0.1:8765",
                                     description="Inside the network: the inpaint server, e.g. nc.hcigroup.de:8765; "
                                                 "127.0.0.1:8765 = this computer")
    token: bpy.props.StringProperty(name="Server token", subtype="PASSWORD",
                                    description="Printed by inpaint_server.py at its start (not needed on this computer)")
    project: bpy.props.StringProperty(name="Project folder", subtype="DIR_PATH", default="~/inpainting-3D",
                                      description="This project on this computer (for the mask AI of 'Find part')")

    def draw(self, context):
        layout = self.layout
        box = layout.box()
        box.label(text="Find part: the mask AI on this computer", icon="VIEWZOOM")
        box.prop(self, "project")
        installed = os.path.isfile(mask_ai()[0])
        box.label(text="Mask AI installed" if installed else "Not installed yet: run ./install_mask.sh in the project folder",
                  icon="CHECKMARK" if installed else "ERROR")
        box = layout.box()
        box.label(text="Inside the uni network: the inpaint server directly", icon="LINKED")
        box.prop(self, "server")
        box.prop(self, "token")
        layout = layout.box()
        layout.label(text="Outside the network: the Sciebo cloud", icon="WORLD")
        if os.path.isfile(CLOUD_CONFIG):
            with open(CLOUD_CONFIG) as f:
                saved = json.load(f)
            layout.label(text=f"Saved: {saved['user']}, folder {saved.get('folder', CLOUD_FOLDER)}", icon="CHECKMARK")
        layout.label(text="Sciebo: avatar > Persönliche Einstellungen > Sicherheit > Neues App-Passwort erstellen")
        for name in ("url", "user", "password", "folder"):
            layout.prop(self, name)
        layout.operator(INPAINT_OT_cloud_login.bl_idname, icon="LOCKED")


class INPAINT_OT_cloud_login(bpy.types.Operator):
    bl_idname = "inpaint.cloud_login"
    bl_label = "Save and test the cloud login"
    bl_description = "Test the login on the cloud folder and save it (readable only by you)"

    def execute(self, context):
        prefs = context.preferences.addons[__name__].preferences
        config = {"url": prefs.url.strip(), "user": prefs.user.strip(), "password": prefs.password.strip(),
                  "folder": prefs.folder.strip().strip("/") or CLOUD_FOLDER}
        try:
            cloud("PROPFIND", "", config=config, timeout=20)  # the folder exists and the login works
        except urllib.error.HTTPError as e:
            self.report({"ERROR"}, "Wrong login name or app password" if e.code == 401 else
                        f"The cloud folder '{config['folder']}' isn't there (HTTP {e.code})")
            return {"CANCELLED"}
        except OSError as e:
            self.report({"ERROR"}, f"The cloud can't be reached: {e}")
            return {"CANCELLED"}
        os.makedirs(os.path.dirname(CLOUD_CONFIG), exist_ok=True)
        with open(os.open(CLOUD_CONFIG, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f:
            json.dump(config, f)
        prefs.password = ""  # don't keep it in Blender's settings file
        self.report({"INFO"}, f"Cloud login works and is saved for {config['user']}")
        return {"FINISHED"}


def in_background(work):
    """Runs work() in a thread; follow_remote() sees it finish through remote["busy"]."""
    def wrapper():
        try:
            work()
        except Exception as e:
            remote["error"] = f"{type(e).__name__}: {e}"
        finally:
            remote["busy"] = None
    remote["busy"] = threading.Thread(target=wrapper, daemon=True)
    remote["busy"].start()


class INPAINT_OT_send_job(bpy.types.Operator):
    bl_idname = "inpaint.send_job"
    bl_label = "Send to uni server"
    bl_description = ("Pack the job and send it to the uni GPU server through the Sciebo cloud; "
                      "the result is loaded here when it is done")

    @classmethod
    def poll(cls, context):
        return remote["job"] is None

    def execute(self, context):
        if not os.path.isfile(CLOUD_CONFIG):
            self.report({"ERROR"}, "No cloud login yet: Edit > Preferences > Add-ons > TRELLIS.2 Inpainting")
            return {"CANCELLED"}
        out = pack_job(self, context)
        if out is None:
            return {"CANCELLED"}
        job = f"{os.path.splitext(os.path.basename(out))[0]}_{time.strftime('%Y%m%d-%H%M%S')}"
        remote.update(job=job, message="uploading the job...", error=None, results=None,
                      object=target_model(context).name)

        def upload():
            with open(out, "rb") as f:
                cloud("PUT", job + ".zip", f.read())
            cloud("PUT", job + ".status", json.dumps({"state": "queued", "message": "waiting for the uni server",
                                                       "results": []}).encode())
        in_background(upload)
        bpy.app.timers.register(follow_remote, first_interval=2.0)
        return {"FINISHED"}


def follow_remote():
    """Timer: show how far the job is on the uni server; download and load the result when it's done."""
    if remote["busy"] is not None:
        return 2.0  # still uploading or downloading
    if remote["error"]:
        remote.update(message="failed: " + remote["error"], job=None)
        redraw()
        return None
    if remote["results"] is not None:  # downloaded: load it
        folder = bpy.path.abspath(bpy.context.scene.inpaint.folder)
        path = os.path.join(folder, remote["results"][0])
        seed = path.rsplit("_seed", 1)[1].split(".")[0] if "_seed" in path else "server"
        load_result({"object": remote["object"], "seed": seed}, path)
        remote.update(message=f"done: {os.path.basename(path)}", job=None)
        redraw()
        return None
    job = remote["job"]
    try:
        state = json.loads(cloud("GET", job + ".status", timeout=20))
    except Exception as e:
        remote["message"] = f"can't read the status yet ({type(e).__name__})"
        redraw()
        return 15.0
    remote["message"] = f"{state['state']}: {state.get('message', '')}"[:120]
    if state["state"] == "failed":
        remote["job"] = None
        redraw()
        return None
    if state["state"] == "done" and state.get("results"):
        names = state["results"]
        folder = bpy.path.abspath(bpy.context.scene.inpaint.folder)
        remote["message"] = f"downloading {len(names)} result(s)..."

        def download():
            for name in names:
                data = cloud("GET", name)
                with open(os.path.join(folder, name), "wb") as f:
                    f.write(data)
            remote["results"] = names
        in_background(download)
    redraw()
    return 15.0


class INPAINT_OT_load_result(bpy.types.Operator):
    bl_idname = "inpaint.load_result"
    bl_label = "Load result"
    bl_description = "Load a result .glb (e.g. downloaded from the server) next to the hidden original"

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    filter_glob: bpy.props.StringProperty(default="*.glb", options={"HIDDEN"})

    def invoke(self, context, event):
        self.filepath = bpy.path.abspath(context.scene.inpaint.folder)
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context):
        model = target_model(context)
        name = os.path.splitext(os.path.basename(self.filepath))[0]
        seed = name.rsplit("_seed", 1)[1] if "_seed" in name else "server"
        load_result({"object": model.name if model else name, "seed": seed}, self.filepath)
        return {"FINISHED"}


class INPAINT_OT_show_picture(bpy.types.Operator):
    bl_idname = "inpaint.show_picture"
    bl_label = "Show big"
    bl_description = "Open the picture's before/after preview in its own window"

    def execute(self, context):
        path = os.path.splitext(bpy.path.abspath(context.scene.inpaint.picture))[0] + "_preview.jpg"
        if not os.path.isfile(path):
            self.report({"ERROR"}, "No preview for this picture")
            return {"CANCELLED"}
        open_image_window(path)
        return {"FINISHED"}


class INPAINT_OT_check(bpy.types.Operator):
    bl_idname = "inpaint.check_server"
    bl_label = "Check server"
    bl_description = "Ask whether the inpaint server is running"

    def execute(self, context):
        if context.scene.inpaint.connection == "CLOUD":
            status["server"] = "cloud login saved" if os.path.isfile(CLOUD_CONFIG) else "no cloud login (preferences)"
            return {"FINISHED"}
        state = ask("/status", timeout=5)
        if state is not None and "error" in state and "busy" not in state:
            status["server"] = state["error"]  # e.g. wrong token
        else:
            status["server"] = "not running" if state is None else ("busy" if state["busy"] else "ready")
        return {"FINISHED"}


class INPAINT_PT_panel(bpy.types.Panel):
    bl_label = "TRELLIS.2 Inpainting"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Inpaint"

    def draw(self, context):
        settings = context.scene.inpaint
        layout = self.layout
        row = layout.row()
        row.label(text=f"Server: {status['server']}")
        row.operator(INPAINT_OT_check.bl_idname, text="", icon="FILE_REFRESH")
        if status["message"]:
            layout.label(text=status["message"])

        box = layout.box()
        box.label(text="A. Mark the area", icon="SELECT_SET")
        box.label(text="Faces: Edit Mode (Tab), select")
        box.operator(INPAINT_OT_add_box.bl_idname, icon="CUBE")
        if is_box(context.active_object):
            box.label(text="Drag: arrow = a side, dot = a corner")
        if settings.model is not None:
            box.label(text=f"Model: {settings.model.name}")
        box.label(text="Find a part in the box (AI):")
        box.prop(settings, "part")
        row = box.row(align=True)
        row.operator(INPAINT_OT_pick_point.bl_idname, icon="EYEDROPPER",
                     text=f"Clicked {len(settings.clicks)} spot(s): add more" if settings.clicks else "Click on it")
        if settings.clicks:
            row.operator(INPAINT_OT_clear_click.bl_idname, text="", icon="X")
        box.operator(INPAINT_OT_find_part.bl_idname, icon="VIEWZOOM")
        box.prop(settings, "find_cpu")
        if finder["message"]:
            box.label(text=finder["message"])
        if finder["folder"] and os.path.isfile(os.path.join(finder["folder"], "overview.jpg")):
            box.operator(INPAINT_OT_show_found.bl_idname, icon="IMAGE_DATA")
        box.prop(settings, "box_use")

        box = layout.box()
        box.label(text="B. What should be there (optional)", icon="TEXT")
        box.prop(settings, "prompt", text="")
        box.prop(settings, "reference")
        box.label(text="Turn the view to show the area, then:")
        box.operator(INPAINT_OT_make_picture.bl_idname, icon="IMAGE_DATA")

        box = layout.box()
        box.label(text="C. Regenerate in 3D", icon="MOD_REMESH")
        box.prop(settings, "picture")
        preview = bpy.data.images.get(os.path.splitext(os.path.basename(bpy.path.abspath(settings.picture)))[0] + "_preview.jpg")
        if preview is not None and preview.preview is not None:
            box.template_icon(icon_value=preview.preview.icon_id, scale=8)
            box.operator(INPAINT_OT_show_picture.bl_idname, icon="WINDOW")
        row = box.row()
        row.prop(settings, "seed")
        row.prop(settings, "grow")
        box.prop(settings, "room")
        box.prop(settings, "steps")
        box.prop(settings, "method")
        box.prop(settings, "connection", expand=True)
        if settings.connection == "DIRECT":
            box.label(text=f"Server: {server()[0].split('//')[1]} (add-on preferences)")
        box.operator(INPAINT_OT_regenerate.bl_idname, text="2. Regenerate area", icon="PLAY")
        if remote["message"]:
            box.label(text=remote["message"])

        box = layout.box()
        box.label(text="D. Job files", icon="PACKAGE")
        row = box.row()
        row.operator(INPAINT_OT_save_job.bl_idname, text="Save job only", icon="PACKAGE")
        row.operator(INPAINT_OT_load_result.bl_idname, icon="IMPORT")

        box = layout.box()
        box.label(text="E. Geometry Nodes", icon="GEOMETRY_NODES")
        box.operator(INPAINT_OT_clean_up.bl_idname, icon="BRUSH_DATA")
        row = box.row()
        row.operator(INPAINT_OT_compare.bl_idname, icon="MOD_MASK")
        row.operator(INPAINT_OT_keep_original.bl_idname, icon="MOD_BOOLEAN")
        obj = context.active_object
        for name, value, label in ((CLEANUP, "Min share", "Min share"), (COMPARE, "Threshold", "Threshold"),
                                   (KEEP, "Threshold", "Threshold")):
            modifier = obj.modifiers.get(name) if obj is not None and obj.type == "MESH" else None
            if modifier is not None and modifier.node_group is not None:
                row = box.row()
                row.label(text=name)
                row.prop(*cleanup_input(modifier, value), text=label)
                row.prop(modifier, "show_viewport", text="", icon="HIDE_OFF")

        layout.prop(settings, "folder")


classes = (InpaintClick, InpaintSettings, INPAINT_OT_add_box, INPAINT_GGT_box_sides, INPAINT_OT_pick_point,
           INPAINT_OT_clear_click, INPAINT_OT_find_part, INPAINT_OT_show_found, INPAINT_OT_make_picture,
           INPAINT_OT_regenerate, INPAINT_OT_save_job, INPAINT_OT_send_job, INPAINT_OT_load_result,
           INPAINT_OT_clean_up, INPAINT_OT_compare, INPAINT_OT_keep_original,
           InpaintPreferences, INPAINT_OT_cloud_login,
           INPAINT_OT_show_picture, INPAINT_OT_check, INPAINT_PT_panel)


def register():
    global draw_handler
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.inpaint = bpy.props.PointerProperty(type=InpaintSettings)
    draw_handler = bpy.types.SpaceView3D.draw_handler_add(draw_boxes, (), "WINDOW", "POST_VIEW")
    bpy.app.handlers.depsgraph_update_post.append(after_change)


def unregister():
    global draw_handler
    if after_change in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(after_change)
    if draw_handler is not None:
        bpy.types.SpaceView3D.draw_handler_remove(draw_handler, "WINDOW")
        draw_handler = None
    del bpy.types.Scene.inpaint
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
