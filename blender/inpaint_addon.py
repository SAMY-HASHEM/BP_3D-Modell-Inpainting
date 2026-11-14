"""Blender add-on: mark an area of a model, press a button, and TRELLIS.2 regenerates that area.

The area is the selected faces (Edit Mode) and/or "space boxes": boxes that mark empty space where
something new may grow (e.g. a statue on the ground). The boxes are drawn see-through blue, and the
selected one has an arrow on each side (drag it to move that side) and a dot on each corner (drag it
to pull that corner anywhere, so a box can take any shape). "Find part in box" selects only one part
inside the box (e.g. the helmet, by its name and/or a click on it) with a mask AI on this computer
(scripts/inpainting/find_part.py, installed by install_mask.sh); Box = "Selected faces + empty space" then regenerates
that part and the empty space in the box and keeps every other surface exactly. Every face has one of three colours or none:
red goes completely, yellow may be changed by the AI, green is never touched; Q starts the brush that paints them (E yellow,
R green, W erase, Shift + click the whole piece under the mouse, Ctrl + Z undo). Optionally a prompt (and a reference picture)
says what should be there: the server then first paints it into a picture of the model, seen from the
current view, and TRELLIS.2 builds that picture's change in 3D.

Where the AI runs ("Where does the AI run?" at the top of the panel):
  Uni server   cloud_worker.py on the uni GPU server, reached through the Sciebo cloud: the same from home and at uni
  This PC      scripts/server/inpaint_server.py on this computer's GPU; the add-on starts and stops it

Install: Edit > Preferences > Add-ons > (arrow at the top right) > Install from Disk... > this file.
Use:     3D view sidebar (N key) > Inpaint tab.
"""
import atexit
import base64
import io
import json
import os
import signal
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
    "version": (1, 20, 0),
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
local_server = {"process": None, "log": None}  # the inpaint server this add-on started on this PC
draw_handler = None


LOCAL_PORT = 8766  # not the server's usual 8765: another program on this PC had that port (a subtitle tool, Nov 14)
LOCAL_ADDRESS = f"127.0.0.1:{LOCAL_PORT}"


def server():
    """The inpaint server on this computer: its address and request headers (its token is in a file here)."""
    token = ""
    if os.path.isfile(LOCAL_TOKEN):
        with open(LOCAL_TOKEN) as f:
            token = f.read().strip()
    return f"http://{LOCAL_ADDRESS}", {"X-Token": token}


def is_local():
    return bpy.context.scene.inpaint.connection == "LOCAL"


def local_running():
    process = local_server["process"]
    return process is not None and process.poll() is None


def ask(path, data=None, timeout=1.0):
    """JSON request to the inpaint server; None if it isn't running."""
    url, headers = server()
    request = urllib.request.Request(url + path, data=json.dumps(data).encode() if data is not None else None,
                                     headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as e:
        try:
            return json.loads(e.read())
        except ValueError:  # another program answers on this port, not the inpaint server
            return None
    except (OSError, ValueError):
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


def only_empty_space(model, points):
    """The points of a box that are empty room: not on the model's surface and not inside it (below its surface). The
    model's surface and inside stay KEEP, so nothing old grows again (a second layer on a head)."""
    vertices = np.empty(len(model.data.vertices) * 3)
    model.data.vertices.foreach_get("co", vertices)
    matrix = np.array(model.matrix_world)
    vertices = vertices.reshape(-1, 3) @ matrix[:3, :3].T + matrix[:3, 3]
    low = vertices.min(0)
    cell = np.ptp(vertices, 0).max() / GRID
    cells = ((vertices - low) / cell).astype(int)
    full = np.zeros((GRID + 3, GRID + 3, GRID + 3), bool)
    full[cells[:, 0] + 1, cells[:, 1] + 1, cells[:, 2] + 1] = True
    for axis in range(3):  # one cube wider: the surface stays exact
        full = full | np.roll(full, 1, axis) | np.roll(full, -1, axis)
    roof = np.full((GRID + 3, GRID + 3), -1)
    np.maximum.at(roof, (cells[:, 0] + 1, cells[:, 1] + 1), cells[:, 2] + 1)
    where = ((points - low) / cell).astype(int) + 1
    inside = np.all((where >= 0) & (where < GRID + 3), axis=1)
    keep = np.zeros(len(points), bool)
    a, b, c = where[inside].T
    keep[inside] = full[a, b, c] | (c <= roof[a, b])
    return points[~keep]


def area_points(operator, context):
    """The model, the marked area (glTF coordinates), how many faces are selected, whether boxes count, and
    the grow the job must use; or None after reporting what's missing. Green faces are taken out of the area."""
    area = marked_area(operator, context)
    if area is None:
        return None
    model, points, count, boxes, grow = area
    if (zones_of(model) == GREEN).any():
        points, both = protect_green(model, points, grow, context.scene.inpaint.room)
        if not len(points):
            operator.report({"ERROR"}, "Everything marked is green (never touched): paint the part red (Q) first")
            return None
        if both:
            operator.report({"WARNING"}, f"{both} small cube(s) hold red and green faces: green wins, they stay")
        grow = 0  # grown here already, around the green
    return model, points, count, boxes, grow


def marked_area(operator, context):
    """As area_points, before the green faces are taken out. Red and yellow are the selected faces."""
    model = target_model(context)
    if model is None:
        operator.report({"ERROR"}, "Click the model first")
        return None
    if context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")  # also writes the Edit Mode selection into the mesh
    zones_to_selection(model)
    settings = context.scene.inpaint
    faces, count = selected_points(model)
    boxes = np.zeros((0, 3)) if settings.box_use == "part" else box_points(model)
    if settings.box_use not in ("all", "space", "replace") and count == 0:
        operator.report({"ERROR"}, "Nothing is red or yellow: find the part or paint it (Q), "
                                   "or set Box to 'Everything in the box'")
        return None
    if count == 0 and len(boxes) == 0:
        operator.report({"ERROR"}, "Mark an area: paint the part red (Q) or add a space box")
        return None
    if settings.step != "shape" and settings.box_use in ("space", "replace"):  # the part is there now: the new part in the box gets its colours
        return model, to_gltf(boxes), count, True, 0
    if settings.box_use in ("space", "replace"):  # the surfaces stay (replace: pack_job adds what goes): the job grows 0
        return model, to_gltf(only_empty_space(model, boxes)), count, True, 0
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
        operator.report({"ERROR"}, "The server isn't running: press 'Start server on this PC'")
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
                                        description="A picture of the new part, as seen from this view, or a video of it "
                                                    "turning once (its four sides are taken from the video). The "
                                                    "background is cut away on this PC")
    sides: bpy.props.StringProperty(name="More sides", subtype="DIR_PATH",
                                    description="Optional: a folder with up to 3 pictures of the new part, in turning "
                                                "order (right, back, left). All sides together give the best colours. "
                                                "Not needed with a video as the Reference")
    step: bpy.props.EnumProperty(
        name="Step",
        items=[("shape", "1 Shape", "The new part grows in the marked area"),
               ("details", "2 Details", "The part stays where it is, only its fine shape gets sharper"),
               ("colours", "3 Colours", "The shape stays exactly, only the colours change, on the same mesh (fast). "
                                        "Always the last step")],
        default="shape")
    seed: bpy.props.IntProperty(name="Seed", default=1, min=0, description="Another seed gives another result")
    grow: bpy.props.IntProperty(name="Grow", default=2, min=0, max=8,
                                description="Grow the marked area by this many voxels (of 64 per side)")
    steps: bpy.props.IntProperty(name="Steps", default=25, min=4, max=50, description="Sampling steps per stage")
    room: bpy.props.FloatProperty(name="Room above", default=0.0, min=0.0, max=0.5, step=5,
                                  description="Share of TRELLIS.2's space kept free above the model, for a new part "
                                              "on top of it (e.g. 0.2 for a crown on a head that touches the top)")
    method: bpy.props.EnumProperty(
        name="Method",
        items=[("inversion", "Inversion", "As VoxHammer: the rest follows the original's own path and the new area "
                                          "attends to exactly the original around it")],
        default="inversion")
    model: bpy.props.PointerProperty(name="Model", type=bpy.types.Object, description="The model being changed")
    connection: bpy.props.EnumProperty(
        name="Connection",
        items=[("CLOUD", "Uni server", "Through the Sciebo cloud to the uni GPU server, the same from home and at uni "
                                       "(cloud_worker.py runs there; login in the preferences)"),
               ("LOCAL", "This PC", "The GPU of this computer (e.g. an RTX 3090): the add-on starts the server here")],
        default="CLOUD")
    part: bpy.props.StringProperty(name="Part", description="What to find in the space box, in English "
                                                            "(e.g. helmet, sword, left wing)")
    clicks: bpy.props.CollectionProperty(type=InpaintClick, description="Spots on the part, for Find part")
    show_steps: bpy.props.BoolProperty(name="Server steps (JupyterLab)", default=False,
                                       description="What to type on the uni GPU server before the first job")
    find_cpu: bpy.props.BoolProperty(name="CPU only", default=False,
                                     description="Run the mask AI without the GPU (slower: about 1-2 minutes)")
    box_use: bpy.props.EnumProperty(
        name="Box",
        items=[("space", "Only the empty space", "The model stays exactly (its surface and inside): the new part grows "
                                                 "only into the empty room in the box. Best for adding something"),
               ("replace", "Replace what the new part covers", "The old surface goes where the new part covers it from "
                                                               "every side of 1. Make picture; the rest of the model stays. "
                                                               "For a helmet in place of hair"),
               ("all", "Everything in the box", "Everything inside the space box is regenerated, the old surface too"),
               ("part", "Only the selected faces", "The box only frames the part: just the selected faces (e.g. "
                                                   "the part the AI found) are regenerated, in their own place"),
               ("part_space", "Selected faces + empty space",
                "The selected faces and the empty space in the box are regenerated; every other surface in the "
                "box stays exactly as it is (no colour bleeding onto it). For replacing a part with something new")],
        default="space")
    improve: bpy.props.StringProperty(name="Make it better", description="How the picture of the change should be better, in "
                                      "your words, e.g. 'make the head smaller and join it to the neck with a metal collar'. "
                                      "The local picture AI reworks the AFTER pictures so")
    piece_angle: bpy.props.FloatProperty(name="Piece ends at edges over", default=40.0, min=5.0, max=180.0,
                                         description="Shift + click takes the whole piece under the mouse: it ends at edges "
                                                     "sharper than this many degrees (smaller: smaller pieces; 180: only a "
                                                     "green line or the box's wall ends it)")
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
    if not boxes and not settings.clicks and settings.model is None and painter["tool"] is None:
        return
    before = gpu.state.blend_get(), gpu.state.depth_test_get(), gpu.state.depth_mask_get()
    gpu.state.blend_set("ALPHA")
    gpu.state.depth_mask_set(False)
    fill = gpu.shader.from_builtin("UNIFORM_COLOR")
    draw_zones(fill)  # red, yellow and green on the model
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
# Blender takes pictures of the space box from the directions that show it best, scripts/inpainting/find_part.py (in its
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
            os.path.join(project, "scripts", "inpainting", "find_part.py"))


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
    finder["message"] = f"found {found:,} faces: red (fix them with the brush: Q)"
    for o in bpy.context.view_layer.objects:  # the model active, so Tab opens it (not the box)
        o.select_set(False)
    model.select_set(True)
    bpy.context.view_layer.objects.active = model
    show_part(model)
    redraw()
    return None


def show_part(model):
    """The model's selected faces become red (drawn by draw_zones() in Object Mode: in Edit Mode, a model with
    millions of faces is all black edges). Yellow and green stay as painted."""
    selection_to_zones(model)
    finder.update(part_model=model.name)


@bpy.app.handlers.persistent
def after_change(scene, depsgraph):
    """Back in Object Mode after fixing the found part by hand: show the new selection in red."""
    model = bpy.data.objects.get(finder["part_model"] or "") or scene.inpaint.model
    mode = bpy.context.mode
    if model is not None and model.type == "MESH" and mode == "OBJECT" and finder["mode_before"] == "EDIT_MESH":
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


# ---------------------------------------------------------------- the picture of the change, without any AI

GREY = (0.745, 0.745, 0.753)


def read_image(path):
    """A picture as rows x columns x 4, top row first, as stored in the file."""
    image = bpy.data.images.load(path, check_existing=False)
    width, height = image.size
    pixels = np.empty(width * height * 4, np.float32)
    image.pixels.foreach_get(pixels)
    bpy.data.images.remove(image)
    return pixels.reshape(height, width, 4)[::-1]


def write_image(pixels, path):
    height, width = pixels.shape[:2]
    image = bpy.data.images.new("inpaint_picture", width, height, alpha=True)
    image.colorspace_settings.name = "sRGB"
    image.pixels.foreach_set(pixels[::-1].astype(np.float32).ravel())
    image.filepath_raw = path
    image.file_format = "JPEG" if path.lower().endswith((".jpg", ".jpeg")) else "PNG"
    image.save()
    bpy.data.images.remove(image)


def cut_out(pixels):
    """The object of a picture, cropped: its alpha, or else everything that isn't the plain background around it."""
    if pixels[..., 3].min() > 0.9:
        border = np.concatenate([pixels[0, :, :3], pixels[-1, :, :3], pixels[:, 0, :3], pixels[:, -1, :3]])
        plain = np.linalg.norm(pixels[..., :3] - np.median(border, 0), axis=2) < 0.12
        reach = lambda a, axis: np.flip(np.logical_and.accumulate(np.flip(a, axis), axis), axis)
        outside = (np.logical_and.accumulate(plain, 0) | reach(plain, 0)
                   | np.logical_and.accumulate(plain, 1) | reach(plain, 1))  # only background the edge can reach
        pixels = np.dstack([pixels[..., :3], (~outside).astype(np.float32)])
    rows, cols = np.nonzero(pixels[..., 3] > 0.5)
    if not len(rows):
        return None
    return pixels[rows.min():rows.max() + 1, cols.min():cols.max() + 1]


def resized(pixels, width, height):
    ys = np.linspace(0, pixels.shape[0] - 1, height)
    xs = np.linspace(0, pixels.shape[1] - 1, width)
    y0, x0 = ys.astype(int), xs.astype(int)
    y1, x1 = np.minimum(y0 + 1, pixels.shape[0] - 1), np.minimum(x0 + 1, pixels.shape[1] - 1)
    fy, fx = (ys - y0)[:, None, None], (xs - x0)[None, :, None]
    top = pixels[y0][:, x0] * (1 - fx) + pixels[y0][:, x1] * fx
    return top * (1 - fy) + (pixels[y1][:, x0] * (1 - fx) + pixels[y1][:, x1] * fx) * fy


WITHOUT = "Inpaint without red"


def without_group():
    """The Geometry Nodes graph that takes the red faces (the part that goes) off a model, for the picture of what is
    behind and under it:   Named Attribute (inpaint_zone) == red -> Delete Geometry (faces)."""
    if WITHOUT in bpy.data.node_groups:
        return bpy.data.node_groups[WITHOUT]
    tree = bpy.data.node_groups.new(WITHOUT, "GeometryNodeTree")
    tree.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    tree.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    tree.is_modifier = True
    nodes, link = tree.nodes, tree.links.new
    inp, out = nodes.new("NodeGroupInput"), nodes.new("NodeGroupOutput")
    zone = nodes.new("GeometryNodeInputNamedAttribute")
    zone.data_type = "INT"
    zone.inputs["Name"].default_value = ZONE
    red = nodes.new("FunctionNodeCompare")
    red.data_type, red.operation = "INT", "EQUAL"
    delete = nodes.new("GeometryNodeDeleteGeometry")
    delete.domain = "FACE"
    for node, x in ((inp, -600), (zone, -600), (red, -350), (delete, -100), (out, 150)):
        node.location = (x, -200 if node in (zone, red) else 0)
    whole = [s for s in red.inputs if s.enabled and s.type == "INT"]  # the node's two whole-number inputs
    link(next(s for s in zone.outputs if s.enabled and s.type == "INT"), whole[0])
    whole[1].default_value = RED
    link(red.outputs["Result"], delete.inputs["Selection"])
    link(inp.outputs["Geometry"], delete.inputs["Geometry"])
    link(delete.outputs["Geometry"], out.inputs["Geometry"])
    return tree


def render_view(model, rotation, path, size=1024, without_red=False):
    """The model alone from the view's direction, without perspective, so every pixel is a known place.
    Returns (centre, right, up, span) of the camera. without_red: the model without the part that goes (the camera
    stays where it is for the whole model)."""
    scene = bpy.data.scenes.new("InpaintView")
    shown = model
    if without_red:  # a stand-in that shares the model's mesh, with the red faces taken off by Geometry Nodes
        shown = model.copy()
        shown.modifiers.new(WITHOUT, "NODES").node_group = without_group()
    scene.collection.objects.link(shown)
    corners = [model.matrix_world @ Vector(c) for c in model.bound_box]
    center = sum(corners, Vector()) / 8
    span = max((a - b).length for a in corners for b in corners) * 1.3  # room for a new part
    world = bpy.data.worlds.new("InpaintView")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)
    world.node_tree.nodes["Background"].inputs[1].default_value = 0.9
    scene.world = world
    sun = bpy.data.objects.new("InpaintSun", bpy.data.lights.new("InpaintSun", "SUN"))
    sun.data.energy = 2.0
    sun.rotation_euler = rotation.to_euler()
    scene.collection.objects.link(sun)
    camera = bpy.data.objects.new("InpaintCamera", bpy.data.cameras.new("InpaintCamera"))
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = span
    camera.data.clip_end = span * 10
    camera.location = center + rotation @ Vector((0, 0, span * 2))
    camera.rotation_euler = rotation.to_euler()
    scene.collection.objects.link(camera)
    scene.camera = camera
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 16
    scene.cycles.use_denoising = True
    scene.render.resolution_x = scene.render.resolution_y = size
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True, scene=scene.name)
    right, up = rotation @ Vector((1, 0, 0)), rotation @ Vector((0, 1, 0))
    for thing, where in ((camera.data, bpy.data.cameras), (sun.data, bpy.data.lights)):
        where.remove(thing)
    scene.collection.objects.unlink(shown)
    if shown is not model:
        bpy.data.objects.remove(shown)
    bpy.data.scenes.remove(scene)
    bpy.data.worlds.remove(world)
    return center, right, up, span


def nearest_faces(model, center, right, up, span, size, faces=None):
    """For every pixel of a side's picture: the colour (0 none, red, yellow, green; -1 = no surface) and the depth (in the
    scene's units, smaller = nearer the viewer) of the nearest face there. Per 2 x 2 pixels the nearest face centre
    counts; small gaps between the centres are closed from their neighbours. faces: only these (a yes/no per face)."""
    cache, zones = mesh_cache(model), zones_of(model)
    use = np.arange(len(zones)) if faces is None else np.flatnonzero(faces)
    away = cache["centers"][use] - np.array(center, np.float32)
    right, up = np.array(right, np.float32), np.array(up, np.float32)
    cell = 2
    side = size // cell
    x = ((away @ right / span + 0.5) * side).astype(np.int64)
    y = ((0.5 - away @ up / span) * side).astype(np.int64)
    depth = -(away @ np.cross(right, up))
    inside = np.flatnonzero((x >= 0) & (x < side) & (y >= 0) & (y < side))
    index = y[inside] * side + x[inside]
    order = np.lexsort((depth[inside], index))  # per cell the nearest face first
    first = order[np.flatnonzero(np.diff(index[order], prepend=-1))]
    zone = np.full(side * side, -1, np.int8)
    near = np.full(side * side, np.inf, np.float32)
    zone[index[first]] = zones[use[inside[first]]]
    near[index[first]] = depth[inside[first]]
    zone, near = zone.reshape(side, side), near.reshape(side, side)
    for _ in range(2):  # empty cells between face centres take the nearest of their neighbours
        empty = zone < 0
        padded_zone, padded_near = np.pad(zone, 1, constant_values=-1), np.pad(near, 1, constant_values=np.inf)
        best_zone, best_near = zone.copy(), near.copy()
        for dy in range(3):
            for dx in range(3):
                other_zone, other_near = padded_zone[dy:dy + side, dx:dx + side], padded_near[dy:dy + side, dx:dx + side]
                better = empty & (other_near < best_near)
                best_zone[better], best_near[better] = other_zone[better], other_near[better]
        zone, near = best_zone, best_near
    grow = lambda a: np.repeat(np.repeat(a, cell, 0), cell, 1)
    return grow(zone), grow(near)


def picture_of_side(model, rotation, points, part, stem):
    """One side: BEFORE, AFTER and the 2D MASK of the whole model from this direction. AFTER is the model without the
    part that goes (the red faces: what is behind and under them shows), with the new part put into the marked area by
    depth: the new part is taken as round (as wide front to back as it is from side to side, row by row), and surface
    that stays and is nearer than it stays in front. So a neck goes into a collar instead of lying on it. The 2D mask
    is everything that changed: where the old part showed and where the new one shows. Writes <stem>_before.png,
    <stem>.png, <stem>_mask.png and <stem>_old.png (where the old part showed, for the judge). Returns which of
    `points` the new part covers, or the text of what went wrong."""
    render = stem + "_render.png"
    center, right, up, span = render_view(model, rotation, render)
    view = read_image(render)
    size = view.shape[0]
    grey = np.array(GREY, np.float32)
    before = view[..., :3] * view[..., 3:4] + grey * (1 - view[..., 3:4])
    zones = zones_of(model)
    goes = zones == RED
    old = np.zeros((size, size), bool)
    base, stays_depth = before, None
    if goes.any():
        old = (nearest_faces(model, center, right, up, span, size)[0] == RED) & (view[..., 3] > 0.5)
        render_view(model, rotation, render, without_red=True)
        rest = read_image(render)
        base = rest[..., :3] * rest[..., 3:4] + grey * (1 - rest[..., 3:4])
        stays_depth = nearest_faces(model, center, right, up, span, size, ~goes)[1]
        stays_depth[rest[..., 3] < 0.5] = np.inf  # no surface there
    os.remove(render)
    if os.path.isfile(stem + "_old.png"):
        os.remove(stem + "_old.png")
    opaque = np.ones((size, size, 1), np.float32)
    if old.any():
        write_image(np.dstack([np.repeat(old[..., None], 3, 2).astype(np.float32), opaque]), stem + "_old.png")
    world = np.stack([points[:, 0], -points[:, 2], points[:, 1]], 1) - np.array(center)  # glTF back to Blender's axes
    xs = (world @ np.array(right) / span + 0.5) * size
    ys = (0.5 - world @ np.array(up) / span) * size
    left, top, wide, tall = xs.min(), ys.min(), np.ptp(xs), np.ptp(ys)
    scale = min(wide / part.shape[1], tall / part.shape[0])
    width, height = max(2, int(part.shape[1] * scale)), max(2, int(part.shape[0] * scale))
    x, y = int(left + (wide - width) / 2), int(top + tall - height)  # middle of the area, standing on its floor
    if x < 0 or y < 0 or x + width > size or y + height > size:
        return "The marked area is outside the view"
    part = resized(part, width, height)
    share = part[..., 3:4].copy()
    if stays_depth is not None:  # what stays and is nearer than the new part stays in front of it
        solid = part[..., 3] > 0.5
        columns = np.arange(width)[None]
        first = np.where(solid.any(1), solid.argmax(1), 0)[:, None]
        last = np.where(solid.any(1), width - 1 - solid[:, ::-1].argmax(1), 0)[:, None]
        radius = (last - first + 1) / 2
        bulge = np.sqrt(np.maximum(radius ** 2 - (columns - (first + last) / 2) ** 2, 0)) * span / size
        middle = float(np.mean(-(world @ np.cross(np.array(right), np.array(up)))))  # the marked area's middle, front to back
        share[stays_depth[y:y + height, x:x + width] < middle - bulge] = 0
    after = base.copy()
    after[y:y + height, x:x + width] = part[..., :3] * share + after[y:y + height, x:x + width] * (1 - share)
    covers = np.zeros((size, size), bool)
    covers[y:y + height, x:x + width] = share[..., 0] > 0.1
    mask = covers | old
    for _ in range(4):  # a little wider, as an outline
        mask = mask | np.roll(mask, 1, 0) | np.roll(mask, -1, 0) | np.roll(mask, 1, 1) | np.roll(mask, -1, 1)
        covers = covers | np.roll(covers, 1, 0) | np.roll(covers, -1, 0) | np.roll(covers, 1, 1) | np.roll(covers, -1, 1)
    write_image(np.dstack([before, opaque]), stem + "_before.png")
    write_image(np.dstack([after, opaque]), stem + ".png")
    write_image(np.dstack([np.repeat(mask[..., None], 3, 2).astype(np.float32), opaque]), stem + "_mask.png")
    return covers[np.clip(ys.astype(int), 0, size - 1), np.clip(xs.astype(int), 0, size - 1)]


def change_preview(stem, cell=512):
    """The pictures of the change side by side, to check before regenerating: the top row before, the bottom row after,
    a column per side (as seen now, then a quarter turn further each). Writes <stem>_preview.jpg, shown in the panel."""
    stems = [stem] + [f"{stem}_side{k}" for k in range(1, 4) if os.path.isfile(f"{stem}_side{k}.png")]
    sheet = np.ones((2 * cell, len(stems) * cell, 4), np.float32)
    for column, one in enumerate(stems):
        for row, end in enumerate(("_before.png", ".png")):
            sheet[row * cell:(row + 1) * cell, column * cell:(column + 1) * cell] = resized(read_image(one + end), cell, cell)
    sheet[..., 3] = 1
    sheet[cell - 1:cell + 1] = (0.2, 0.2, 0.2, 1)  # a line between before and after
    write_image(sheet, stem + "_preview.jpg")
    return stem + "_preview.jpg"


PICTURE_ENDS = (".png", ".jpg", ".jpeg", ".webp")
VIDEO_ENDS = (".mp4", ".mov", ".webm", ".avi", ".mkv", ".m4v")


def project_command(script, *arguments):
    """The command that runs a script of scripts/inpainting with the project's own Python (outside Blender, also from a
    Flatpak Blender); None without the project on this PC."""
    project = project_folder()
    path = os.path.join(project, "scripts", "inpainting", script)
    if not os.path.isfile(path) or not os.path.isfile(os.path.join(project, "activate.sh")):
        return None
    command = ["bash", "-c", 'source activate.sh >/dev/null && exec python -u "$@"', "project", path, *map(str, arguments)]
    if os.path.exists("/.flatpak-info"):  # Blender as a Flatpak: the project's Python is outside its sandbox
        command = ["flatpak-spawn", "--host", f"--directory={project}"] + command
    return command


def part_pictures(reference, sides, folder):
    """The new part's pictures cut out on transparent by the project's background AI (scripts/inpainting/picture_check.py parts,
    outside Blender): a video gives its four sides, a picture with a background loses it. Returns (the pictures, "")
    or (None, why not): without the project on this PC the pictures are taken as they are."""
    project = project_folder()
    command = project_command("picture_check.py", "parts", reference, "--out", folder, *(["--sides", sides] if sides else []))
    if command is None:
        return None, "no project folder on this PC (add-on preferences)"
    try:
        done = subprocess.run(command, cwd=project, capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired) as e:
        return None, str(e)
    record = os.path.join(folder, "part.json")
    if done.returncode != 0 or not os.path.isfile(record):
        return None, (done.stdout + done.stderr).strip().splitlines()[-1][:200] if (done.stdout + done.stderr).strip() else "it gave nothing"
    with open(record) as f:
        return json.load(f)["pictures"], ""


def picture_of_change(model, rotation, points, reference, stem, sides="", pictures=None):
    """The change from this view and, with a folder of `sides`, from the sides a quarter turn further each (as the crown
    on the simple robot: every side its own before, after and mask). The sides are <stem>_side1.png and so on.
    pictures = the part's sides ready cut out (part_pictures), instead of the reference and the folder as they are."""
    if pictures is None:
        pictures = [reference]
        if sides:
            pictures += [os.path.join(sides, n) for n in sorted(os.listdir(sides)) if n.lower().endswith(PICTURE_ENDS)][:3]
    for k in range(1, 4):  # nothing left of an earlier picture with more sides
        for end in (".png", "_before.png", "_mask.png"):
            if os.path.isfile(f"{stem}_side{k}{end}"):
                os.remove(f"{stem}_side{k}{end}")
    covered = np.ones(len(points), bool)
    for k, picture in enumerate(pictures):
        part = cut_out(read_image(picture))
        if part is None:
            return f"No object in {os.path.basename(picture)}"
        turn = Matrix.Rotation(k * 1.5707963267948966, 4, "Z").to_quaternion()
        side = picture_of_side(model, turn @ rotation, points, part, stem if k == 0 else f"{stem}_side{k}")
        if isinstance(side, str):
            return side
        covered &= side
    np.save(stem + "_covered.npy", points[covered])  # what the part covers from every side (Box: Replace)
    return None


class INPAINT_OT_make_picture(bpy.types.Operator):
    bl_idname = "inpaint.make_picture"
    bl_label = "1. Make picture"
    bl_description = ("Picture the whole model from the current view, with the Reference put into the marked area: "
                      "the picture before, after and the 2D mask. The Reference is a picture or a video of the part "
                      "turning once; its background is cut away first (this PC's project folder). Without a Reference: "
                      "the prompt is painted in by Qwen-Image-Edit in ComfyUI (only on This PC)")

    @classmethod
    def poll(cls, context):
        return status["job"] is None and context.space_data is not None and context.space_data.type == "VIEW_3D"

    def execute(self, context):
        settings = context.scene.inpaint
        reference = bpy.path.abspath(settings.reference) if settings.reference else ""
        if reference and not os.path.isfile(reference):
            self.report({"ERROR"}, "The reference picture doesn't exist")
            return {"CANCELLED"}
        if reference:  # the collage: the whole model before, after and the 2D mask
            area = area_points(self, context)
            if area is None:
                return {"CANCELLED"}
            model, points = area[:2]
            folder = bpy.path.abspath(settings.folder)
            os.makedirs(folder, exist_ok=True)
            stem = os.path.join(folder, f"{bpy.path.clean_name(model.name)}_picture_seed{settings.seed}")
            sides = bpy.path.abspath(settings.sides) if settings.sides else ""
            if sides and not os.path.isdir(sides):
                self.report({"ERROR"}, "The folder of More sides doesn't exist")
                return {"CANCELLED"}
            whole_box = to_gltf(box_points(model))  # also the old surface in it: the part may cover it
            context.window.cursor_set("WAIT")
            pictures, why = part_pictures(reference, sides, stem + "_part")  # cut out, a video's four sides
            context.window.cursor_set("DEFAULT")
            if pictures is None and reference.lower().endswith(VIDEO_ENDS):
                self.report({"ERROR"}, f"The video's sides couldn't be taken: {why}")
                return {"CANCELLED"}
            problem = picture_of_change(model, context.space_data.region_3d.view_rotation.copy(),
                                        whole_box if len(whole_box) else points, reference, stem, sides, pictures)
            if problem:
                self.report({"ERROR"}, problem)
                return {"CANCELLED"}
            preview = change_preview(stem)
            show_picture(stem + ".png")  # it is the Picture now, its before / after sheet shows in the panel
            open_image_window(preview)  # and big in its own window: top row before, bottom row after
            self.report({"INFO"} if pictures else {"WARNING"},
                        f"Picture ready (before, after, 2D mask) from {len(pictures)} side(s) of the part. Press 2. Regenerate area"
                        if pictures else f"Picture ready, but the part's pictures were taken as they are ({why})")
            return {"FINISHED"}
        if not is_local():  # the picture AI (ComfyUI) runs only on this PC
            self.report({"ERROR"}, "Uni server: choose a Reference picture of the new part")
            return {"CANCELLED"}
        if not settings.prompt.strip():
            self.report({"ERROR"}, "Choose a Reference picture, or write a prompt")
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


# ---------------------------------------------------------------- the picture judged and made better, before regenerating
# scripts/inpainting/picture_check.py (outside Blender): "judge" measures every side's pictures and asks the local vision
# AI, "improve" lets the local picture AI rework the AFTER pictures as the user's words say, "restore" brings the first
# pictures back. One of them at a time, in the background; the panel shows its last line.

checker = {"process": None, "kind": None, "stem": None, "message": "", "log": None, "read": None}


def picture_stem(context):
    """The Picture's path without .png, when it is a picture of a change (it has its BEFORE and 2D mask); else None."""
    path = bpy.path.abspath(context.scene.inpaint.picture) if context.scene.inpaint.picture else ""
    stem = os.path.splitext(path)[0]
    return stem if path.lower().endswith(".png") and all(os.path.isfile(stem + end) for end in (".png", "_before.png", "_mask.png")) else None


def judgement(stem):
    """What the judge wrote about this picture (read again when its file changed); None before the first judging."""
    path = stem + "_judge.json"
    if not os.path.isfile(path):
        return None
    changed = os.path.getmtime(path)
    if checker["read"] is None or checker["read"][:2] != (path, changed):
        try:
            with open(path) as f:
                checker["read"] = (path, changed, json.load(f))
        except ValueError:  # being written this moment
            return checker["read"][2] if checker["read"] and checker["read"][0] == path else None
    return checker["read"][2]


def start_check(operator, kind, stem, *arguments):
    command = project_command("picture_check.py", kind, stem, *arguments)
    if command is None:
        operator.report({"ERROR"}, "This needs the project on this PC: set the Project folder in the add-on preferences")
        return {"CANCELLED"}
    log = stem + f"_{kind}.log"
    with open(log, "w") as f:
        process = subprocess.Popen(command, cwd=project_folder(), stdout=f, stderr=subprocess.STDOUT)
    starting = {"judge": "judging: measuring the pictures...", "improve": "the picture AI starts (first time about a minute)...",
                "enhance": "enhancing: judging first...",
                "restore": "bringing the first pictures back..."}[kind]
    checker.update(process=process, kind=kind, stem=stem, log=log, message=starting)
    bpy.app.timers.register(follow_check, first_interval=1.0)
    return {"FINISHED"}


def follow_check():
    """Timer: the last line of the script's log as the message; when it is done, the new pictures are shown."""
    process, kind, stem = checker["process"], checker["kind"], checker["stem"]
    try:
        with open(checker["log"], errors="replace") as f:
            lines = [line.strip() for line in f.read().replace("\r", "\n").splitlines() if line.strip()]
    except OSError:
        lines = []
    if process.poll() is None:
        if lines:
            checker["message"] = f"{kind}: {lines[-1][:90]}"
        redraw()
        return 1.0
    checker["process"] = None
    if process.returncode != 0:
        checker["message"] = f"{kind} failed: {lines[-1][:140] if lines else 'no answer'}"
    elif kind == "judge":
        checker["message"] = "judged: see below"
    else:
        change_preview(stem)
        show_picture(stem + ".png")
        checker["message"] = {"improve": "the pictures are reworked: look, judge again, or go back",
                              "enhance": f"enhanced: {lines[-1][lines[-1].rfind('(') + 1:].rstrip(')') if lines else ''}",
                              "restore": "the first pictures are back"}[kind]
    redraw()
    return None


class INPAINT_OT_judge_picture(bpy.types.Operator):
    bl_idname = "inpaint.judge_picture"
    bl_label = "Judge the picture"
    bl_description = ("Before regenerating: measure the pictures of the change (is the old part covered, does the new part "
                      "lie on things that stay, does it float, how big is it) and let the local vision AI look at them. "
                      "Says what is wrong and what to do about it")

    @classmethod
    def poll(cls, context):
        return checker["process"] is None and picture_stem(context) is not None

    def execute(self, context):
        settings = context.scene.inpaint
        busy = status["job"] is not None or remote["job"] is not None  # the GPU is taken: only what is measured
        return start_check(self, "judge", picture_stem(context), "--what", settings.prompt.strip() or "a new part",
                           *(["--no-ai"] if busy else []))


class INPAINT_OT_improve_picture(bpy.types.Operator):
    bl_idname = "inpaint.improve_picture"
    bl_label = "Improve the picture (AI)"
    bl_description = ("The local picture AI (Qwen-Image-Edit) reworks the AFTER picture of every side as your words say; "
                      "outside the change the picture stays exactly. 'Back to the first picture' undoes it")

    @classmethod
    def poll(cls, context):
        return (checker["process"] is None and picture_stem(context) is not None and status["job"] is None
                and remote["job"] is None and bool(context.scene.inpaint.improve.strip()))

    def execute(self, context):
        settings = context.scene.inpaint
        return start_check(self, "improve", picture_stem(context), "--prompt", settings.improve.strip(), "--seed", settings.seed)


class INPAINT_OT_enhance_picture(bpy.types.Operator):
    bl_idname = "inpaint.enhance_picture"
    bl_label = "Enhance by itself (AI)"
    bl_description = ("The local AIs work on the picture by themselves: judge it, rework it with the judge's own sentence "
                      "(and your words in 'Make it better', if any), judge again. A round stays only if the picture got "
                      "better; at most two rounds, a few minutes each. 'Back to the first picture' undoes it")

    @classmethod
    def poll(cls, context):
        return (checker["process"] is None and picture_stem(context) is not None and status["job"] is None
                and remote["job"] is None)

    def execute(self, context):
        settings = context.scene.inpaint
        return start_check(self, "enhance", picture_stem(context), "--what", settings.prompt.strip() or "a new part",
                           "--seed", settings.seed, *(["--prompt", settings.improve.strip()] if settings.improve.strip() else []))


class INPAINT_OT_restore_picture(bpy.types.Operator):
    bl_idname = "inpaint.restore_picture"
    bl_label = "Back to the first picture"
    bl_description = "Bring back the pictures as 1. Make picture made them, before the picture AI reworked them"

    @classmethod
    def poll(cls, context):
        stem = picture_stem(context)
        return checker["process"] is None and stem is not None and os.path.isfile(stem + "_first.png")

    def execute(self, context):
        return start_check(self, "restore", picture_stem(context))


class INPAINT_OT_use_idea(bpy.types.Operator):
    bl_idname = "inpaint.use_idea"
    bl_label = "Use the AI's idea"
    bl_description = "Put the local AI's own sentence for a better picture into 'Make it better' (change it as you like)"

    @classmethod
    def poll(cls, context):
        stem = picture_stem(context)
        found = judgement(stem) if stem else None
        return bool(found and found.get("ai") and found["ai"].get("prompt"))

    def execute(self, context):
        context.scene.inpaint.improve = judgement(picture_stem(context))["ai"]["prompt"]
        return {"FINISHED"}


def draw_judgement(box, context):
    """Section B's lower half: the picture, its judge and the words that make it better."""
    settings = context.scene.inpaint
    stem = picture_stem(context)
    if stem is None:
        return
    preview = bpy.data.images.get(os.path.basename(stem) + "_preview.jpg")
    if preview is not None and preview.preview is not None:
        box.label(text="Top: before.  Bottom: after.", icon="IMAGE_DATA")
        box.template_icon(icon_value=preview.preview.icon_id, scale=10)
        big(box, 1.3).operator(INPAINT_OT_show_picture.bl_idname, text="Show the picture big", icon="WINDOW")
    check = box.box()
    check.label(text="Is the picture good? Ask before regenerating", icon="VIEWZOOM")
    big(check).operator(INPAINT_OT_judge_picture.bl_idname, icon="VIEWZOOM")
    if checker["message"]:
        row = check.row()
        row.alert = "failed" in checker["message"]
        row.label(text=checker["message"], icon="ERROR" if row.alert else "INFO")
    found = judgement(stem)
    if found is not None:
        ai = found.get("ai")
        row = check.row()
        row.scale_y = 1.3
        row.label(text=f"Measured: {found['score_measured']} / 10" + (f"      Local AI: {ai['score']} / 10" if ai else ""),
                  icon="CHECKMARK" if found["score_measured"] >= 8 and (not ai or ai["score"] >= 7) else "ERROR")
        if not found["findings"]:
            check.label(text="Nothing wrong was measured", icon="CHECKMARK")
        for finding in found["findings"]:
            lines = check.column(align=True)
            lines.alert = finding["bad"] >= 2
            for k, line in enumerate(wrapped(finding["what"], context)):
                lines.label(text=line, icon="ERROR" if k == 0 else "BLANK1")
            lines = check.column(align=True)
            for k, line in enumerate(wrapped(finding["do"], context)):
                lines.label(text=line, icon="FORWARD" if k == 0 else "BLANK1")
        if ai:
            check.label(text="The local AI says:", icon="INFO")
            for key, icon in (("good", "CHECKMARK"), ("problems", "ERROR"), ("better", "FORWARD")):
                for text in ai.get(key, []):
                    lines = check.column(align=True)
                    for k, line in enumerate(wrapped(text, context)):
                        lines.label(text=line, icon=icon if k == 0 else "BLANK1")
        for one in found.get("rounds") or []:
            check.label(text=f"Round {one['round']}: {one['before']} to {one['after']} of 20, " + ("kept" if one["kept"] else "taken back"),
                        icon="CHECKMARK" if one["kept"] else "LOOP_BACK")
        if ai:
            pass
        elif found.get("ai_error"):
            for k, line in enumerate(wrapped("The local AI didn't answer (is Ollama running, is the GPU free?): only what was measured", context)):
                check.label(text=line, icon="INFO" if k == 0 else "BLANK1")
    better = box.box()
    better.label(text="Make it better: say how", icon="BRUSH_DATA")
    better.prop(settings, "improve", text="")
    better.operator(INPAINT_OT_use_idea.bl_idname, icon="FORWARD")
    big(better).operator(INPAINT_OT_improve_picture.bl_idname, icon="BRUSH_DATA")
    big(better).operator(INPAINT_OT_enhance_picture.bl_idname, icon="SHADERFX")
    if status["job"] is not None or remote["job"] is not None:
        better.label(text="Waits until the regenerating is done", icon="INFO")
    better.operator(INPAINT_OT_restore_picture.bl_idname, icon="LOOP_BACK")


class INPAINT_OT_regenerate(bpy.types.Operator):
    bl_idname = "inpaint.regenerate"
    bl_label = "Regenerate area"
    bl_description = ("Pack the model, the marked area and the Picture, send them the chosen way "
                      "(the uni server through the cloud, or this PC) and load the result")

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
    covered = os.path.splitext(picture)[0] + "_covered.npy"
    if settings.box_use == "replace" and settings.step == "shape" and os.path.isfile(covered):
        points = np.concatenate([points, np.load(covered)])
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
    # the settings that worked best: new noise in the area, soft edge, colours that follow the pictures
    version = {"method": settings.method, "start": "random", "soft": 2, "texture_guidance": 3}
    if settings.step == "colours":
        version["keep_shape"] = True
    elif settings.step == "details":
        version["keep_structure"] = True
    job["variants"] = [version]
    stem = os.path.splitext(picture)[0]
    sides = [k for k in range(1, 4) if all(os.path.isfile(f"{stem}_side{k}{end}") for end in (".png", "_before.png", "_mask.png"))]
    if "source" in job and sides:
        job["more_edits"] = [[f"side{k}_{what}.png" for what in ("after", "before", "mask")] for k in sides]
    out = os.path.join(folder, f"{name}_job.zip")
    with zipfile.ZipFile(out, "w") as z:  # stored, not compressed: the model is compressed already
        z.write(path, "model.glb")
        z.write(area_file, "area.npy")
        z.write(picture, picture_name)
        if "source" in job:
            z.write(before, "source.png")
            z.write(picture_mask, "picture_mask.png")
        for k in sides if "more_edits" in job else ():
            for what, end in (("after", ".png"), ("before", "_before.png"), ("mask", "_mask.png")):
                z.write(f"{stem}_side{k}{end}", f"side{k}_{what}.png")
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
    project: bpy.props.StringProperty(name="Project folder", subtype="DIR_PATH", default="~/inpainting-3D",
                                      description="This project on this computer (for the mask AI of 'Find part')")

    def draw(self, context):
        layout = self.layout
        box = layout.box()
        box.label(text="This PC: the project folder (server and mask AI)", icon="DESKTOP")
        box.prop(self, "project")
        installed = os.path.isfile(mask_ai()[0])
        box.label(text="Mask AI installed" if installed else "Not installed yet: run ./install_mask.sh in the project folder",
                  icon="CHECKMARK" if installed else "ERROR")
        layout = layout.box()
        layout.label(text="Uni server: the Sciebo cloud", icon="WORLD")
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


# What to type on the uni GPU server (JupyterLab terminal) for each connection, shown in the panel
SERVER_STEPS = {
    "CLOUD": {
        "commands": ["cd ~/projects/bp_samy_hashem && git pull",
                     "pkill -u $USER -f cloud_worker.py",
                     "source activate.sh",
                     "nohup python -u scripts/server/cloud_worker.py > outputs/cloud_worker.log 2>&1 &",
                     "sleep 5; tail -n 5 outputs/cloud_worker.log"],
        "explain": ["",
                    "stop an old worker, if one runs",
                    "turn on the project's Python",
                    "start the worker; it keeps running after the VDI is closed",
                    "show its first lines"],
        "expect": ["You should see: 'Watching the cloud folder", "inpainting-jobs/ every 30 s'. Then close the VDI."]},
}


class INPAINT_OT_copy_steps(bpy.types.Operator):
    bl_idname = "inpaint.copy_steps"
    bl_label = "Copy commands"
    bl_description = "Copy the commands for the uni server; paste them in the JupyterLab terminal (Ctrl+Shift+V)"

    first: bpy.props.BoolProperty(default=False, options={"SKIP_SAVE"})  # only the git pull (it asks for the login)

    def execute(self, context):
        commands = SERVER_STEPS[context.scene.inpaint.connection]["commands"]
        commands = commands[:1] if self.first else commands[1:]
        context.window_manager.clipboard = "\n".join(commands) + "\n"
        self.report({"INFO"}, "Copied: paste them in the JupyterLab terminal with Ctrl+Shift+V")
        return {"FINISHED"}


def draw_steps(layout, settings):
    """The fold-out with the server steps for the chosen uni connection."""
    steps = SERVER_STEPS[settings.connection]
    layout.prop(settings, "show_steps", icon="TRIA_DOWN" if settings.show_steps else "TRIA_RIGHT", emboss=False)
    if not settings.show_steps:
        return
    column = layout.box().column(align=True)
    column.label(text="1. VDI > browser > nc.hcigroup.de > JupyterLab")
    column.label(text="    File > New > Terminal")
    column.label(text="    Paste there with Ctrl+Shift+V")
    pairs = list(zip(steps["commands"], steps["explain"]))
    column.separator()
    column.label(text="2. Get the newest code, then type your uni login:")
    column.operator(INPAINT_OT_copy_steps.bl_idname, text="Copy command 2", icon="COPYDOWN").first = True
    column.label(text=f"   {pairs[0][0]}", icon="CONSOLE")
    column.separator()
    column.label(text="3. Then the rest:")
    column.operator(INPAINT_OT_copy_steps.bl_idname, text="Copy commands 3", icon="COPYDOWN").first = False
    for command, explain in pairs[1:]:
        column.label(text=f"   {command}", icon="CONSOLE")
        column.label(text=f"      {explain}")
    column.separator()
    column.label(text="4. " + steps["expect"][0])
    for line in steps["expect"][1:]:
        column.label(text="    " + line)


def project_folder():
    return os.path.expanduser(bpy.context.preferences.addons[__name__].preferences.project)


class INPAINT_OT_start_local(bpy.types.Operator):
    bl_idname = "inpaint.start_local"
    bl_label = "Start server on this PC"
    bl_description = "Start inpaint_server.py on this computer's GPU (it loads TRELLIS.2: about a minute)"

    @classmethod
    def poll(cls, context):
        return not local_running()

    def execute(self, context):
        if os.name == "nt":
            self.report({"ERROR"}, "'This PC' needs the Linux setup: on Windows choose 'Uni server'")
            return {"CANCELLED"}
        project = project_folder()
        if not os.path.isfile(os.path.join(project, "activate.sh")):
            self.report({"ERROR"}, f"No project in {project}: set the project folder in the add-on preferences")
            return {"CANCELLED"}
        if ask("/status") is not None:
            status["server"] = "already running"
            return {"FINISHED"}
        log = os.path.join(project, "outputs", "local_server.log")
        os.makedirs(os.path.dirname(log), exist_ok=True)
        local_server["log"] = log
        command = ["bash", "-c", f"source activate.sh && INPAINT_PORT={LOCAL_PORT} exec python -u scripts/server/inpaint_server.py"]
        if os.path.exists("/.flatpak-info"):  # Blender as a Flatpak: its sandbox can't see the NVIDIA GPU
            command = ["flatpak-spawn", "--host", f"--directory={project}"] + command
        with open(log, "w") as f:
            local_server["process"] = subprocess.Popen(command, cwd=project, stdout=f, stderr=subprocess.STDOUT,
                                                       start_new_session=True)
        status["server"] = "starting (about a minute)..."
        bpy.app.timers.register(follow_local_start, first_interval=3.0)
        return {"FINISHED"}


def follow_local_start():
    """Shows the local server's start until it answers (or stops)."""
    if not local_running():
        with open(local_server["log"]) as f:
            text = f.read()
        status["server"] = ("Blender may not start it (Flatpak): see docs/HOW-TO.md"
                            if "org.freedesktop.Flatpak" in text or "Portal" in text else
                            f"didn't start: see {local_server['log']}")
    elif ask("/status") is not None:
        status["server"] = "ready"
    else:
        with open(local_server["log"]) as f:
            lines = [line.strip() for line in f if line.strip()]
        status["server"] = "starting: " + (lines[-1][:60] if lines else "...")
        redraw()
        return 3.0
    redraw()
    return None


def redraw():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()


class INPAINT_OT_stop_local(bpy.types.Operator):
    bl_idname = "inpaint.stop_local"
    bl_label = "Stop server"
    bl_description = "Stop the server this add-on started on this PC (frees the GPU memory)"

    @classmethod
    def poll(cls, context):
        return local_running()

    def execute(self, context):
        stop_local()
        status["server"] = "stopped"
        return {"FINISHED"}


def stop_local():
    """Stops the server started by this add-on, with everything it started (its own process group)."""
    if local_running():
        os.killpg(local_server["process"].pid, signal.SIGTERM)
    local_server["process"] = None


atexit.register(stop_local)  # closing Blender frees the GPU too


ping = {"job": None, "sent": 0.0, "error": None}  # the running connection test through the cloud
PING_WAIT = 150  # seconds: the worker looks every 30 s, and may be starting a job first


class INPAINT_OT_test_connection(bpy.types.Operator):
    bl_idname = "inpaint.test_connection"
    bl_label = "Test connection"
    bl_description = ("Send a tiny test job and wait for the answer: This PC asks the server here; "
                      "Uni server goes through Sciebo to the worker on the server and back (up to a minute)")

    @classmethod
    def poll(cls, context):
        return ping["job"] is None

    def execute(self, context):
        if context.scene.inpaint.connection != "CLOUD":
            bpy.ops.inpaint.check_server()
            self.report({"INFO"}, f"Server: {status['server']}")
            return {"FINISHED"}
        if not os.path.isfile(CLOUD_CONFIG):
            status["server"] = "no cloud login (Login / address)"
            return {"CANCELLED"}
        job = f"ping_{time.strftime('%Y%m%d-%H%M%S')}"
        ping.update(job=job, sent=time.time(), error=None)
        status["server"] = "test: sending to Sciebo..."

        def send():
            try:
                data = io.BytesIO()
                with zipfile.ZipFile(data, "w") as z:
                    z.writestr("job.json", json.dumps({"kind": "ping"}))
                cloud("PUT", job + ".zip", data.getvalue(), timeout=20)
                cloud("PUT", job + ".status", json.dumps({"state": "queued", "message": "connection test",
                                                          "results": []}).encode(), timeout=20)
            except Exception as e:
                ping["error"] = f"Sciebo not reachable: {type(e).__name__}: {e}"[:120]
        threading.Thread(target=send, daemon=True).start()
        bpy.app.timers.register(follow_ping, first_interval=3.0)
        return {"FINISHED"}


def follow_ping():
    """Timer: waits for the worker's answer to the connection test, then removes the test files."""
    job = ping["job"]
    waited = time.time() - ping["sent"]
    answer = None
    if ping["error"]:
        status["server"] = ping["error"]
    else:
        try:
            answer = json.loads(cloud("GET", job + ".status", timeout=20))
        except Exception:
            answer = None
        if answer is not None and answer["state"] == "done":
            status["server"] = f"works ({waited:.0f} s): {answer['message']}"
        elif answer is not None and answer["state"] == "failed":
            status["server"] = "the worker runs, but with old code: do the Server steps again"
        elif waited > PING_WAIT:
            status["server"] = "no answer: the worker isn't running (see Server steps)"
        else:
            status["server"] = f"test: waiting for the uni server... {waited:.0f} s"
            redraw()
            return 5.0
    ping["job"] = None

    def clean():
        for name in (job + ".zip", job + ".status"):
            try:
                cloud("DELETE", name, timeout=20)
            except Exception:
                pass
    threading.Thread(target=clean, daemon=True).start()
    redraw()
    return None


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


# ---------------------------------------------------------------- the three colours (what may happen to every face)
# Every face of the model has one of three colours, or none. They are kept in the mesh (a number per face), so they are
# saved with the .blend file and Blender's own undo knows them.
#   red     goes completely: regenerated from nothing (the old head)
#   yellow  may be changed, removed or kept: the AI decides (the neck under it)
#   green   GREEN     is never touched: its small cubes are taken out of the area, whatever else is in them
# Red and yellow are the "selected faces" of the Box menu. The paint tool (Q) paints them with a brush.

ZONE = "inpaint_zone"  # the mesh attribute: a whole number per face
NO_COLOUR, RED, YELLOW, GREEN = 0, 1, 2, 3

def colour_icon(number):
    """Blender's small colour square (1 red, 3 yellow, 4 green); its name changed between versions."""
    icons = bpy.types.UILayout.bl_rna.functions["label"].parameters["icon"].enum_items.keys()
    return next((name for name in (f"STRIP_COLOR_0{number}", f"SEQUENCE_COLOR_0{number}") if name in icons), "DOT")


ZONES = {  # name, colour word, colour, key, icon, what happens to it
    RED: ("Replace", "red", (1.0, 0.13, 0.1), "Q", colour_icon(1), "goes completely, made new"),
    YELLOW: ("Maybe", "yellow", (1.0, 0.82, 0.1), "E", colour_icon(3), "the AI may change, remove or keep it"),
    GREEN: ("Keep", "green", (0.2, 0.85, 0.3), "R", colour_icon(4), "never touched"),
}
ERASER = ("Erase", "no colour", (0.85, 0.85, 0.85), "W", "X", "takes the colour off")
painter = {"cache": None, "batches": None, "counts": None, "show": True, "tool": None, "undo": [], "redo": []}
UNDO_STEPS = 40


def zones_of(model):
    """The colour of every face (0 = none). Only in Object Mode: in Edit Mode the mesh's own data is empty."""
    mesh = model.data
    values = np.zeros(len(mesh.polygons), np.int32)
    attribute = mesh.attributes.get(ZONE)
    if attribute is not None and len(attribute.data) == len(values):
        attribute.data.foreach_get("value", values)
    return values


def set_zones(model, values):
    mesh = model.data
    attribute = mesh.attributes.get(ZONE)
    if attribute is None or attribute.domain != "FACE" or attribute.data_type != "INT":
        if attribute is not None:
            mesh.attributes.remove(attribute)
        attribute = mesh.attributes.new(ZONE, "INT", "FACE")
    if len(attribute.data) != len(values):
        return False  # Edit Mode: nothing can be written
    attribute.data.foreach_set("value", np.ascontiguousarray(values, np.int32))
    forget_colours()
    return True


def forget_colours(*_):
    """The colours changed (painted, undone, another file): what is drawn and counted is made again."""
    painter["batches"] = painter["counts"] = None
    redraw()


def zone_counts(model):
    """How many faces have each colour, counted once per change (the panel asks at every redraw)."""
    if model is None or model.type != "MESH" or bpy.context.mode != "OBJECT":
        return painter["counts"] or (model.name if model else "", [0, 0, 0, 0])
    if painter["counts"] is None or painter["counts"][0] != model.name:
        painter["counts"] = (model.name, np.bincount(np.clip(zones_of(model), 0, 3), minlength=4).tolist())
    return painter["counts"]


def mesh_cache(model):
    """The model's triangles, face centres and face normals in world space, read once per model and place."""
    mesh = model.data
    key = (model.name, len(mesh.polygons), len(mesh.vertices), tuple(round(v, 6) for row in model.matrix_world for v in row))
    if painter["cache"] is not None and painter["cache"]["key"] == key:
        return painter["cache"]
    mesh.calc_loop_triangles()
    m = np.array(model.matrix_world, np.float32)
    co = np.zeros(len(mesh.vertices) * 3, np.float32)
    mesh.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3]
    owner = np.zeros(len(mesh.loop_triangles), np.int32)
    mesh.loop_triangles.foreach_get("polygon_index", owner)
    corners = np.zeros(len(mesh.loop_triangles) * 3, np.int32)
    mesh.loop_triangles.foreach_get("vertices", corners)
    corners = corners.reshape(-1, 3)
    centers = np.zeros(len(mesh.polygons) * 3, np.float32)
    mesh.polygons.foreach_get("center", centers)
    centers = centers.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3]
    normals = np.zeros(len(mesh.polygons) * 3, np.float32)
    mesh.polygons.foreach_get("normal", normals)
    normals = normals.reshape(-1, 3) @ np.linalg.inv(m[:3, :3])  # as normals turn: with the inverse, transposed
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    painter["cache"] = {"key": key, "co": co, "owner": owner, "corners": corners, "centers": centers, "normals": normals,
                        "size": float(np.ptp(co, 0).max()) if len(co) else 1.0, "next": None, "view": None}
    painter["batches"] = None
    return painter["cache"]


def draw_zones(fill):
    """The three colours on the model, see-through, a little lifted off the surface so they don't flicker with it."""
    tool = painter["tool"]
    model = bpy.data.objects.get(tool["model"]) if tool else bpy.context.scene.inpaint.model
    if (not painter["show"] or model is None or model.type != "MESH" or bpy.context.mode != "OBJECT"
            or bpy.context.view_layer.objects.get(model.name) is not model or not model.visible_get()):
        return
    if painter["batches"] is None or painter["batches"][0] != model.name:
        zones = zones_of(model)
        batches = {}
        if zones.any():
            cache = mesh_cache(model)
            of_triangle = zones[cache["owner"]]
            for zone in ZONES:
                corners = cache["corners"][of_triangle == zone]
                if len(corners):
                    triangles = cache["co"][corners]
                    lift = cache["normals"][cache["owner"][of_triangle == zone]] * (2e-4 * cache["size"])
                    batches[zone] = batch_for_shader(fill, "TRIS", {"pos": (triangles + lift[:, None]).reshape(-1, 3)})
        painter["batches"] = (model.name, batches)
    gpu.state.depth_test_set("LESS_EQUAL")
    for zone, batch in painter["batches"][1].items():
        fill.uniform_float("color", (*ZONES[zone][2], 0.55))
        batch.draw(fill)


def selection_to_zones(model):
    """After Find part, or after selecting by hand in Edit Mode: the selected faces without a colour become red, and red
    faces that are no longer selected lose it. Yellow and green stay as painted."""
    mesh = model.data
    selected = np.zeros(len(mesh.polygons), bool)
    mesh.polygons.foreach_get("select", selected)
    if selected.all():  # everything selected (as after an import) means nothing was chosen
        return
    zones = zones_of(model)
    new = zones.copy()
    new[selected & (zones == NO_COLOUR)] = RED
    new[~selected & (zones == RED)] = NO_COLOUR
    if (new != zones).any():
        set_zones(model, new)


def zones_to_selection(model):
    """Red and yellow are the model's selected faces (what the Box menu calls so), if any colour is painted."""
    zones = zones_of(model)
    if zones.any():
        select_faces(model, (zones == RED) | (zones == YELLOW))
    return zones


def protect_green(model, points, grow, room):
    """The area as the centres of its small cubes (glTF coordinates, to send with grow 0), without the cubes that hold
    green faces: green is never regenerated, also not by growing the area or by red in the same cube. Returns the
    centres and how many cubes held both red and green (they stay)."""
    mesh = model.data
    m = np.array(model.matrix_world)
    co = np.zeros(len(mesh.vertices) * 3, np.float32)
    mesh.vertices.foreach_get("co", co)
    vertices = to_gltf(co.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3])
    low, high = vertices.min(0), vertices.max(0)
    centre, scale = (low + high) / 2, 0.99999 * (1 - room) / (high - low).max()  # as keep_the_rest and inpaint.py
    lift = np.array([0, -room / 2, 0])
    cubes = lambda p: np.clip((((p - centre) * scale + lift + 0.5) * GRID).astype(int), 0, GRID - 1)
    mask = np.zeros((GRID,) * 3, bool)
    mask[tuple(cubes(np.asarray(points, np.float64).reshape(-1, 3)).T)] = True
    for _ in range(grow):
        padded = np.pad(mask, 1)
        mask = np.any([padded[x:x + GRID, y:y + GRID, z:z + GRID] for x in range(3) for y in range(3) for z in range(3)], 0)
    zones = zones_of(model)
    centers = to_gltf(mesh_cache(model)["centers"])
    green, red = np.zeros_like(mask), np.zeros_like(mask)
    green[tuple(cubes(centers[zones == GREEN]).T)] = True
    red[tuple(cubes(centers[zones == RED]).T)] = True
    mask &= ~green
    return ((np.argwhere(mask) + 0.5) / GRID - 0.5 - lift) / scale + centre, int((green & red).sum())


# ---------------------------------------------------------------- the paint tool

def faces_on_screen(cache, region, rv3d):
    """Where every face's centre is in the 3D view (pixels) and which faces can be seen: not hidden behind others (the
    nearest surface per 6 x 6 pixels counts, with a little depth to spare). Made again when the view has turned."""
    stamp = (region.width, region.height, tuple(round(v, 5) for row in rv3d.perspective_matrix for v in row))
    if cache["view"] is not None and cache["view"]["stamp"] == stamp:
        return cache["view"]
    centers = cache["centers"]
    p = np.array(rv3d.perspective_matrix, np.float32)
    w = centers @ p[3, :3] + p[3, 3]
    w = np.where(np.abs(w) < 1e-9, 1e-9, w)
    x = ((centers @ p[0, :3] + p[0, 3]) / w * 0.5 + 0.5) * region.width
    y = ((centers @ p[1, :3] + p[1, 3]) / w * 0.5 + 0.5) * region.height
    v = np.array(rv3d.view_matrix, np.float32)
    depth = -(centers @ v[2, :3] + v[2, 3])  # distance in front of the viewer, in the scene's units
    cell = 6
    columns, rows = region.width // cell + 1, region.height // cell + 1
    inside = (x >= 0) & (x < region.width) & (y >= 0) & (y < region.height) & (w > 0)
    index = (y[inside] // cell).astype(np.int64) * columns + (x[inside] // cell).astype(np.int64)
    nearest = np.full(columns * rows, np.inf, np.float32)
    order = np.argsort(index, kind="stable")
    starts = np.flatnonzero(np.diff(index[order], prepend=-1))
    if len(order):
        nearest[index[order][starts]] = np.minimum.reduceat(depth[inside][order], starts)
    seen = np.zeros(len(centers), bool)
    seen[inside] = depth[inside] <= nearest[index] + 0.012 * cache["size"]
    cache["view"] = {"stamp": stamp, "x": x, "y": y, "seen": seen}
    return cache["view"]


def faces_next_to(model, cache):
    """For every face up to four faces that share an edge with it (-1 = none), and the angle between them as the cosine.
    An edge is shared when its two ends are at the same places: a generated model's mesh is cut into thousands of
    small patches (one per piece of its texture), which share no edge in the mesh itself."""
    if cache["next"] is not None:
        return cache["next"]
    mesh = model.data
    count = len(mesh.polygons)
    totals = np.zeros(count, np.int32)
    mesh.polygons.foreach_get("loop_total", totals)
    starts = np.zeros(count, np.int32)
    mesh.polygons.foreach_get("loop_start", starts)
    corner = np.zeros(len(mesh.loops), np.int32)
    mesh.loops.foreach_get("vertex_index", corner)
    co = cache["co"].astype(np.float64)
    steps = np.round((co - co.min(0)) / max(cache["size"], 1e-12) * (2 ** 20 - 1)).astype(np.int64)  # a millionth of the model
    place = np.unique(steps[:, 0] << 40 | steps[:, 1] << 20 | steps[:, 2], return_inverse=True)[1].astype(np.int64)
    face = np.repeat(np.arange(count, dtype=np.int32), totals)
    after = np.arange(len(corner)) + 1  # the next corner of the same face
    last = starts + totals - 1
    after[last] = starts
    a, b = place[corner], place[corner[after]]
    edges = np.minimum(a, b) << 32 | np.maximum(a, b)
    order = np.argsort(edges, kind="stable")
    edges, face = edges[order], face[order]
    same = np.flatnonzero(edges[1:] == edges[:-1])
    a, b = face[same], face[same + 1]
    source, target = np.concatenate([a, b]), np.concatenate([b, a])
    order = np.argsort(source, kind="stable")
    source, target = source[order], target[order]
    place = np.arange(len(source)) - np.searchsorted(source, source)  # the how-manieth neighbour of its face
    table = np.full((count, 4), -1, np.int32)
    fits = place < 4
    table[source[fits], place[fits]] = target[fits]
    cosine = np.ones((count, 4), np.float32)
    has = table >= 0
    cosine[has] = np.einsum("ij,ij->i", np.repeat(cache["normals"], 4, 0).reshape(count, 4, 3)[has], cache["normals"][table[has]])
    cache["next"] = (table, cosine)
    return cache["next"]


def whole_piece(model, cache, seed, zones, angle):
    """The piece of surface around one face: everything reached from it without crossing an edge sharper than `angle`
    (degrees), a green face (a green line is a fence), or the wall of a space box the face is in."""
    table, cosine = faces_next_to(model, cache)
    allowed = zones != GREEN if zones[seed] != GREEN else np.ones(len(zones), bool)
    boxes = [np.array([tuple(c) for c in box_corners(o)]) for o in bpy.context.view_layer.objects if is_box(o) and o.visible_get()]
    around = [c for c in boxes if inside_box(cache["centers"][seed:seed + 1], c)[0]]
    if around:
        within = np.zeros(len(zones), bool)
        for corners in around:
            within |= inside_box(cache["centers"], corners)
        allowed &= within
    limit = float(np.cos(np.radians(angle)))
    reached = np.zeros(len(zones), bool)
    reached[seed] = True
    front = np.array([seed])
    while len(front):
        near, smooth = table[front].ravel(), cosine[front].ravel() >= limit
        near = near[smooth & (near >= 0)]
        near = np.unique(near[allowed[near] & ~reached[near]])
        reached[near] = True
        front = near
    return reached


def view_under(context, event):
    """The 3D view's main region under the mouse and the mouse's place in it; None over the sidebar, a header or
    another editor."""
    for area in context.window.screen.areas:
        if area.type != "VIEW_3D":
            continue
        over = lambda r: r.x <= event.mouse_x < r.x + r.width and r.y <= event.mouse_y < r.y + r.height
        if any(over(r) for r in area.regions if r.type != "WINDOW" and r.width > 1 and r.height > 1):
            return None
        for region in area.regions:
            if region.type == "WINDOW" and over(region):
                return region, area.spaces.active.region_3d, (event.mouse_x - region.x, event.mouse_y - region.y)
    return None


def draw_brush():
    """The brush as a circle in its colour at the mouse, and the keys at the top of the view (drawn in pixels)."""
    import blf
    tool = painter["tool"]
    if tool is None or bpy.context.region.as_pointer() != tool.get("region"):
        return
    name, word, colour, key = (ZONES.get(tool["zone"]) or ERASER)[:4]
    gpu.state.blend_set("ALPHA")
    line = gpu.shader.from_builtin("POLYLINE_UNIFORM_COLOR")
    line.uniform_float("viewportSize", gpu.state.viewport_get()[2:])
    turn = np.linspace(0, 2 * np.pi, 49)
    x, y = tool["mouse"]
    circle = [(x + tool["radius"] * np.cos(t), y + tool["radius"] * np.sin(t), 0) for t in turn]
    for width, tint in ((4.0, (0, 0, 0, 0.55)), (2.0, (*colour, 1.0))):  # a dark rim, so it shows on every colour
        line.uniform_float("lineWidth", width)
        line.uniform_float("color", tint)
        batch_for_shader(line, "LINE_STRIP", {"pos": circle}).draw(line)
    gpu.state.blend_set("NONE")
    scale = bpy.context.preferences.system.ui_scale
    top = bpy.context.region.height - 34 * scale
    lines = [(18, (*colour, 1.0), f"PAINTING: {name.upper()} ({word})   {tool['message']}"),
             (13, (1, 1, 1, 0.95), "Q Replace (red)    E Maybe (yellow)    R Keep (green)    W Erase"),
             (13, (1, 1, 1, 0.95), "Drag: paint    Shift + click: the whole piece    [ ]: brush size    Tab: hide / show the colours"),
             (13, (1, 1, 1, 0.95), "Ctrl + Z: undo    Ctrl + Shift + Z: redo    Enter or Esc: done")]
    for size, tint, text in lines:
        blf.size(0, size * scale)
        blf.enable(0, blf.SHADOW)
        blf.shadow(0, 3, 0, 0, 0, 0.9)
        blf.color(0, *tint)
        blf.position(0, 60 * scale, top, 0)
        blf.draw(0, text)
        blf.disable(0, blf.SHADOW)
        top -= (size + 9) * scale


class INPAINT_OT_paint(bpy.types.Operator):
    bl_idname = "inpaint.paint"
    bl_label = "Paint the area"
    bl_description = ("Paint on the model what may happen to it. Q red: goes completely. E yellow: the AI may change it. "
                      "R green: never touched. W: erase. Drag paints, Shift + click takes the whole piece under the mouse, "
                      "[ and ] change the brush, Ctrl + Z undoes, Enter or Esc ends")
    bl_options = {"REGISTER", "UNDO"}

    zone: bpy.props.IntProperty(default=RED, min=NO_COLOUR, max=GREEN, options={"SKIP_SAVE"})

    @classmethod
    def poll(cls, context):
        return (context.mode == "OBJECT" and painter["tool"] is None and context.area is not None
                and context.area.type == "VIEW_3D" and paint_model(context) is not None)

    def invoke(self, context, event):
        model = target_model(context)
        if model is None:
            self.report({"ERROR"}, "Click the model first")
            return {"CANCELLED"}
        for o in context.view_layer.objects:  # the model active and no box selected: the box's arrows would catch the clicks
            o.select_set(False)
        model.select_set(True)
        context.view_layer.objects.active = model
        self.model, self.cache = model, mesh_cache(model)
        self.zones = zones_of(model)
        self.stroke = None
        painter.update(show=True, undo=[], redo=[],
                       tool={"model": model.name, "zone": self.zone, "radius": 28.0, "mouse": (-999, -999), "region": None, "message": ""})
        global brush_handler
        brush_handler = bpy.types.SpaceView3D.draw_handler_add(draw_brush, (), "WINDOW", "POST_PIXEL")
        context.window.cursor_modal_set("PAINT_CROSS")
        context.workspace.status_text_set("Paint:  Q red (replace)   E yellow (maybe)   R green (keep)   W erase   |   drag: paint   "
                                          "Shift + click: whole piece   [ ]: brush size   Tab: hide / show   Ctrl + Z: undo   Enter / Esc: done")
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def finish(self, context):
        global brush_handler
        if brush_handler is not None:
            bpy.types.SpaceView3D.draw_handler_remove(brush_handler, "WINDOW")
            brush_handler = None
        painter["tool"] = None
        painter["show"] = True
        context.window.cursor_modal_restore()
        context.workspace.status_text_set(None)
        if context.mode == "OBJECT" and self.zones.any():
            select_faces(self.model, (self.zones == RED) | (self.zones == YELLOW))  # Tab shows the same faces
            if context.scene.inpaint.box_use in ("space", "all"):
                context.scene.inpaint.box_use = "part_space"  # the colours count: the red and yellow faces + the empty space
        counts = np.bincount(self.zones, minlength=4)
        self.report({"INFO"}, f"Painted: {counts[1]:,} red, {counts[2]:,} yellow, {counts[3]:,} green faces")
        redraw()
        return {"FINISHED"}

    def change(self, faces, zone):
        """Gives the faces (a yes/no per face) the colour, and shows it."""
        if not faces.any():
            return 0
        self.zones[faces] = zone
        set_zones(self.model, self.zones)
        return int(faces.sum())

    def remember(self):
        painter["undo"].append(self.zones.astype(np.uint8))
        del painter["undo"][:-UNDO_STEPS]
        painter["redo"].clear()

    def step_back(self, forward):
        take, give = (painter["redo"], painter["undo"]) if forward else (painter["undo"], painter["redo"])
        if not take:
            return "nothing to redo" if forward else "nothing to undo"
        give.append(self.zones.astype(np.uint8))
        self.zones = take.pop().astype(np.int32)
        set_zones(self.model, self.zones)
        return "redone" if forward else "undone"

    def dab(self, view, mouse, before):
        """The faces seen under the brush on its way from `before` to `mouse`."""
        radius = painter["tool"]["radius"]
        a, b = np.array(before, np.float32), np.array(mouse, np.float32)
        low, high = np.minimum(a, b) - radius, np.maximum(a, b) + radius
        near = np.flatnonzero(view["seen"] & (view["x"] >= low[0]) & (view["x"] <= high[0]) & (view["y"] >= low[1]) & (view["y"] <= high[1]))
        point = np.stack([view["x"][near], view["y"][near]], 1)
        along = b - a
        share = np.clip((point - a) @ along / max(float(along @ along), 1e-9), 0, 1)
        close = np.linalg.norm(point - (a + share[:, None] * along), axis=1) <= radius
        faces = np.zeros(len(self.zones), bool)
        faces[near[close]] = True
        return faces

    def modal(self, context, event):
        tool = painter["tool"]
        if context.mode != "OBJECT" or context.view_layer.objects.get(self.model.name) is not self.model:
            return self.finish(context)
        if event.type in {"RET", "NUMPAD_ENTER", "ESC", "RIGHTMOUSE"} and event.value == "PRESS":
            return self.finish(context)
        if event.value == "PRESS" and event.type in {"Q", "E", "R", "W"} and not event.ctrl and not event.alt:
            tool["zone"] = {"Q": RED, "E": YELLOW, "R": GREEN, "W": NO_COLOUR}[event.type]
            tool["message"] = ""
            redraw()
            return {"RUNNING_MODAL"}
        if event.value == "PRESS" and event.type in {"LEFT_BRACKET", "RIGHT_BRACKET"}:
            tool["radius"] = float(np.clip(tool["radius"] * (1.25 if event.type == "RIGHT_BRACKET" else 0.8), 4, 400))
            redraw()
            return {"RUNNING_MODAL"}
        if event.value == "PRESS" and event.type == "TAB":
            painter["show"] = not painter["show"]
            tool["message"] = "" if painter["show"] else "(colours hidden: Tab shows them)"
            redraw()
            return {"RUNNING_MODAL"}
        if event.value == "PRESS" and event.type == "Z" and event.ctrl:
            tool["message"] = self.step_back(event.shift)
            redraw()
            return {"RUNNING_MODAL"}
        under = view_under(context, event)
        if under is None:
            if self.stroke is not None and event.type == "LEFTMOUSE" and event.value == "RELEASE":
                self.stroke = None
            return {"PASS_THROUGH"}  # the sidebar, a header, another editor: they work as always
        region, rv3d, mouse = under
        tool["mouse"], tool["region"] = mouse, region.as_pointer()
        if event.type == "LEFTMOUSE" and event.value == "PRESS" and not event.ctrl and not event.alt:
            view = faces_on_screen(self.cache, region, rv3d)
            self.remember()
            painter["show"] = True
            if event.shift:  # the whole piece under the mouse
                seen = np.flatnonzero(view["seen"])
                away = (view["x"][seen] - mouse[0]) ** 2 + (view["y"][seen] - mouse[1]) ** 2
                if not len(seen) or away.min() > 20 ** 2:
                    painter["undo"].pop()
                    tool["message"] = "that wasn't on the model"
                else:
                    piece = whole_piece(self.model, self.cache, int(seen[away.argmin()]), self.zones, context.scene.inpaint.piece_angle)
                    tool["message"] = f"{self.change(piece, tool['zone']):,} faces: the whole piece (too much? Ctrl + Z, then a green line as a fence)"
            else:
                self.stroke = mouse
                tool["message"] = f"{self.change(self.dab(view, mouse, mouse), tool['zone']):,} faces"
            redraw()
            return {"RUNNING_MODAL"}
        if event.type == "MOUSEMOVE":
            if self.stroke is not None:
                self.change(self.dab(faces_on_screen(self.cache, region, rv3d), mouse, self.stroke), tool["zone"])
                self.stroke = mouse
            redraw()
            return {"RUNNING_MODAL"} if self.stroke is not None else {"PASS_THROUGH"}
        if event.type == "LEFTMOUSE" and event.value == "RELEASE" and self.stroke is not None:
            self.stroke = None
            return {"RUNNING_MODAL"}
        return {"PASS_THROUGH"}  # turning and zooming the view work while painting


brush_handler = None


def paint_model(context):
    """The model the paint keys work on, without changing anything (asked while the panel is drawn): only when this
    add-on is in use in the scene, so Q and W stay Blender's own everywhere else."""
    settings = context.scene.inpaint
    model = context.active_object if context.active_object is not None and context.active_object.type == "MESH" \
        and not is_box(context.active_object) else settings.model
    if model is None or model.type != "MESH" or context.view_layer.objects.get(model.name) is not model:
        return None
    in_use = settings.model is not None or model.data.attributes.get(ZONE) is not None \
        or any(is_box(o) for o in context.view_layer.objects)
    return model if in_use else None


class INPAINT_OT_clear_zones(bpy.types.Operator):
    bl_idname = "inpaint.clear_zones"
    bl_label = "Take all colours off"
    bl_description = "Remove the red, yellow and green from the whole model (Ctrl + Z brings them back)"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT" and painter["tool"] is None

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def execute(self, context):
        model = target_model(context)
        if model is None:
            return {"CANCELLED"}
        set_zones(model, np.zeros(len(model.data.polygons), np.int32))
        select_faces(model, np.zeros(len(model.data.polygons), bool))
        self.report({"INFO"}, "All colours taken off (Ctrl + Z brings them back)")
        return {"FINISHED"}


class INPAINT_OT_fill_box(bpy.types.Operator):
    bl_idname = "inpaint.fill_box"
    bl_label = "Colour everything in the box"
    bl_description = "Give every face inside the space box this colour (faces that are green stay green)"
    bl_options = {"REGISTER", "UNDO"}

    zone: bpy.props.IntProperty(default=RED, min=NO_COLOUR, max=GREEN, options={"SKIP_SAVE"})

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT" and painter["tool"] is None

    def execute(self, context):
        model = target_model(context)
        boxes = [o for o in context.view_layer.objects if is_box(o) and o.visible_get()]
        if model is None or not boxes:
            self.report({"ERROR"}, "Put a space box around the part first")
            return {"CANCELLED"}
        centers = mesh_cache(model)["centers"]
        within = np.zeros(len(centers), bool)
        for box in boxes:
            within |= inside_box(centers, np.array([tuple(c) for c in box_corners(box)]))
        zones = zones_of(model)
        within &= (zones != GREEN) | (self.zone == GREEN)
        zones[within] = self.zone
        set_zones(model, zones)
        select_faces(model, (zones == RED) | (zones == YELLOW))
        self.report({"INFO"}, f"{int(within.sum()):,} faces in the box are {(ZONES.get(self.zone) or ERASER)[1]} now")
        return {"FINISHED"}


class INPAINT_OT_fit_box(bpy.types.Operator):
    bl_idname = "inpaint.fit_box"
    bl_label = "Fit box to the colours"
    bl_description = ("Make the space box exactly as big as the red and yellow faces, one small cube of room around them, "
                      "its sides on the lines of TRELLIS.2's 64 grid (a new box if there is none)")
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT" and painter["tool"] is None

    def execute(self, context):
        model = target_model(context)
        if model is None:
            self.report({"ERROR"}, "Click the model first")
            return {"CANCELLED"}
        cache = mesh_cache(model)
        zones = zones_of(model)
        marked = (zones == RED) | (zones == YELLOW)
        if not marked.any():
            self.report({"ERROR"}, "Nothing is red or yellow yet: paint the part (Q) or find it first")
            return {"CANCELLED"}
        points = cache["co"][cache["corners"][marked[cache["owner"]]].ravel()]
        low, high = cache["co"].min(0), cache["co"].max(0)
        middle, cell = (low + high) / 2, float((high - low).max()) / GRID
        first = np.floor((points.min(0) - middle) / cell - 1e-6) - 1  # one cube of room, on the grid's lines
        last = np.ceil((points.max(0) - middle) / cell + 1e-6) + 1
        first, last = np.clip(first, -GRID / 2, GRID / 2), np.clip(last, -GRID / 2, GRID / 2)
        box = context.active_object if is_box(context.active_object) else None
        boxes = [o for o in context.view_layer.objects if is_box(o) and o.visible_get()]
        if box is None and len(boxes) == 1:
            box = boxes[0]
        if box is None or len(box.data.vertices) != 8:
            bpy.ops.mesh.primitive_cube_add(size=2)
            box = context.active_object
            box.name = BOX
            box.display_type, box.show_in_front, box.hide_render = "WIRE", True, True
        for i, signs in zip(CUBE_CORNERS, CORNER_SIGNS):
            box.data.vertices[i].co = Vector([float(s) for s in signs])
        box.data.update()
        box.rotation_euler = (0, 0, 0)
        box.rotation_quaternion = (1, 0, 0, 0)
        box.location = Vector((middle + (first + last) / 2 * cell).tolist())
        box.scale = Vector(((last - first) / 2 * cell).tolist())
        cubes = (last - first).astype(int)
        self.report({"INFO"}, f"The box fits the red and yellow faces: {cubes[0]} x {cubes[1]} x {cubes[2]} small cubes of 64")
        return {"FINISHED"}


def checklist(context):
    """The four steps of a change and whether each is done: (done, name, what to do next). For the panel's top."""
    settings = context.scene.inpaint
    model = paint_model(context)
    counts = zone_counts(model)[1] if model is not None else [0, 0, 0, 0]
    boxes = any(is_box(o) and o.visible_get() for o in context.view_layer.objects)
    marked = counts[RED] + counts[YELLOW] > 0 or (boxes and settings.box_use in ("space", "replace", "all"))
    picture = bool(settings.picture) and os.path.isfile(bpy.path.abspath(settings.picture))
    ready = status["server"] in ("ready", "already running") or settings.connection == "CLOUD"
    return [(ready, "AI ready", "Start the server on this PC (top of this panel)"),
            (marked, "Area marked", "Mark the area: add a space box, then find the part or paint it red (Q)"),
            (picture, "Picture of the change", "Choose a Reference (picture or video of the new part), then 1. Make picture"),
            (bool(remote["results"]), "Regenerated", "Press 2. Regenerate area")]


def section(layout, title, icon, done=None):
    """A part of the panel with a bold head; done = True / False shows whether its step is finished."""
    box = layout.box()
    row = box.row()
    row.scale_y = 1.2
    row.label(text=title, icon=icon)
    if done is not None:
        row.label(text="", icon="CHECKMARK" if done else "RADIOBUT_OFF")
    return box


def big(layout, scale=1.5):
    row = layout.row(align=True)
    row.scale_y = scale
    return row


class INPAINT_PT_panel(bpy.types.Panel):
    bl_label = "TRELLIS.2 Inpainting"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Inpaint"

    def draw(self, context):
        settings = context.scene.inpaint
        layout = self.layout
        steps = checklist(context)
        painting = painter["tool"] is not None

        # where things stand: the four steps, and the one to do next
        box = layout.box()
        row = box.row()
        row.scale_y = 1.2
        row.label(text="Where you are", icon="INFO")
        for number, (done, name, _) in enumerate(steps, 1):
            box.label(text=f"{number}. {name}", icon="CHECKMARK" if done else "RADIOBUT_OFF")
        todo = next((hint for done, _, hint in steps if not done), "Done. Compare it with the original (E), or change more")
        hint = box.box()
        hint.label(text="Painting now: Enter or Esc ends it" if painting else "Next:", icon="BRUSH_DATA" if painting else "FORWARD")
        if not painting:
            for line in wrapped(todo, context):
                hint.label(text=line)

        box = section(layout, "Where does the AI run?", "NETWORK_DRIVE", steps[0][0])
        big(box, 1.2).prop(settings, "connection", expand=True)
        if settings.connection == "LOCAL":
            row = big(box)
            row.operator(INPAINT_OT_start_local.bl_idname, icon="PLAY")
            row.operator(INPAINT_OT_stop_local.bl_idname, text="", icon="SNAP_FACE")
        else:
            box.label(text="Uni GPU server through Sciebo")
            box.label(text="First start the worker on the server (once):")
            draw_steps(box, settings)
        row = box.row()
        row.alert = status["server"] not in ("ready", "already running", "not checked", "busy")
        row.label(text=f"Status: {status['server']}")
        row.operator(INPAINT_OT_check.bl_idname, text="", icon="FILE_REFRESH")
        row = box.row()
        row.operator(INPAINT_OT_test_connection.bl_idname, icon="LINKED")
        if settings.connection != "LOCAL":
            row.operator("preferences.addon_show", text="Login", icon="PREFERENCES").module = __name__
        if status["message"]:
            row = layout.row()
            row.alert = status["message"].startswith("failed")
            row.label(text=status["message"], icon="ERROR" if row.alert else "INFO")

        box = section(layout, "A. Mark the area", "SELECT_SET", steps[1][0])
        if settings.model is not None:
            box.label(text=f"Model: {settings.model.name}", icon="MESH_DATA")
        row = big(box)
        row.operator(INPAINT_OT_add_box.bl_idname, icon="CUBE")
        row.operator(INPAINT_OT_fit_box.bl_idname, text="Fit to colours", icon="PIVOT_BOUNDBOX")
        if is_box(context.active_object):
            box.label(text="Drag: arrow = a side, dot = a corner")

        part = box.box()
        part.label(text="Find a part in the box (AI)", icon="VIEWZOOM")
        part.prop(settings, "part")
        row = part.row(align=True)
        row.operator(INPAINT_OT_pick_point.bl_idname, icon="EYEDROPPER",
                     text=f"Clicked {len(settings.clicks)} spot(s): add more" if settings.clicks else "Click on it")
        if settings.clicks:
            row.operator(INPAINT_OT_clear_click.bl_idname, text="", icon="X")
        big(part).operator(INPAINT_OT_find_part.bl_idname, icon="VIEWZOOM")
        part.prop(settings, "find_cpu")
        if finder["message"]:
            row = part.row()
            row.alert = finder["message"].startswith("mask AI failed")
            row.label(text=finder["message"])
        if finder["folder"] and os.path.isfile(os.path.join(finder["folder"], "overview.jpg")):
            part.operator(INPAINT_OT_show_found.bl_idname, icon="IMAGE_DATA")

        # the three colours: the same names, colours and keys as in the brush and on the model
        paint = box.box()
        paint.label(text="Paint it by hand: three colours", icon="BRUSH_DATA")
        model = paint_model(context)
        counts = zone_counts(model)[1] if model is not None else [0, 0, 0, 0]
        for zone, (name, word, _, key, icon, what) in ZONES.items():
            row = paint.row(align=True)
            row.scale_y = 1.4
            start = row.operator(INPAINT_OT_paint.bl_idname, text=f"{key}    {word.capitalize()}: {name.lower()}", icon=icon,
                                 depress=painting and painter["tool"]["zone"] == zone)
            start.zone = zone
            row = paint.row()
            row.active = counts[zone] > 0
            row.label(text=f"        {what}: {counts[zone]:,} faces")
        row = paint.row(align=True)
        row.scale_y = 1.4
        row.operator(INPAINT_OT_paint.bl_idname, text=f"{ERASER[3]}    {ERASER[0]}", icon=ERASER[4],
                     depress=painting and painter["tool"]["zone"] == NO_COLOUR).zone = NO_COLOUR
        keys = paint.column(align=True)
        keys.scale_y = 0.85
        for line in ("Drag: paint.  [ ]: brush size.", "Shift + click: the whole piece.", "Tab: hide / show.  Enter or Esc: done."):
            keys.label(text=line)
        paint.prop(settings, "piece_angle")
        row = paint.row(align=True)
        row.operator(INPAINT_OT_fill_box.bl_idname, text="Box red", icon=ZONES[RED][4]).zone = RED
        row.operator(INPAINT_OT_fill_box.bl_idname, text="Box yellow", icon=ZONES[YELLOW][4]).zone = YELLOW
        row = paint.row(align=True)
        row.operator("ed.undo", text="Undo", icon="LOOP_BACK")
        row.operator("ed.redo", text="Redo", icon="LOOP_FORWARDS")
        row.operator(INPAINT_OT_clear_zones.bl_idname, text="", icon="TRASH")
        if counts[GREEN] and not counts[RED] and not counts[YELLOW]:
            row = paint.row()
            row.alert = True
            row.label(text="Only green: nothing would change", icon="ERROR")
        box.prop(settings, "box_use")

        box = section(layout, "B. What should be there", "TEXT", steps[2][0])
        box.prop(settings, "prompt", text="")
        box.prop(settings, "reference")
        if not settings.reference.lower().endswith(VIDEO_ENDS):
            box.prop(settings, "sides")
        else:
            box.label(text="A video: its four sides are taken", icon="CHECKMARK")
        box.label(text="Turn the view to show the area, then:")
        big(box, 1.6).operator(INPAINT_OT_make_picture.bl_idname, icon="IMAGE_DATA")
        draw_judgement(box, context)

        box = section(layout, "C. Regenerate in 3D", "MOD_REMESH", steps[3][0])
        box.prop(settings, "picture")
        preview = bpy.data.images.get(os.path.splitext(os.path.basename(bpy.path.abspath(settings.picture)))[0] + "_preview.jpg")
        if preview is not None and preview.preview is not None and picture_stem(context) is None:  # else it is in B
            box.template_icon(icon_value=preview.preview.icon_id, scale=10)
            big(box, 1.3).operator(INPAINT_OT_show_picture.bl_idname, text="Show the picture big", icon="WINDOW")
        row = box.row()
        row.prop(settings, "seed")
        row.prop(settings, "grow")
        box.prop(settings, "room")
        box.prop(settings, "steps")
        box.prop(settings, "step")
        missing = [hint for done, _, hint in steps[1:3] if not done]  # no area or no picture: the button can't work yet
        row = big(box, 1.9)
        row.enabled = not missing
        row.operator(INPAINT_OT_regenerate.bl_idname, text="2. Regenerate area", icon="PLAY")
        for hint in missing:
            warn = box.column(align=True)
            warn.alert = True
            for k, line in enumerate(wrapped("First: " + hint, context)):
                warn.label(text=line, icon="ERROR" if k == 0 else "BLANK1")
        if remote["message"]:
            row = box.row()
            row.alert = bool(remote["error"])
            row.label(text=remote["message"], icon="ERROR" if remote["error"] else "INFO")

        box = section(layout, "D. Job files", "PACKAGE")
        row = box.row()
        row.operator(INPAINT_OT_save_job.bl_idname, text="Save job only", icon="PACKAGE")
        row.operator(INPAINT_OT_load_result.bl_idname, icon="IMPORT")

        box = section(layout, "E. Geometry Nodes", "GEOMETRY_NODES")
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


def wrapped(text, context):
    """A sentence as lines that fit the sidebar (labels don't wrap by themselves)."""
    import textwrap
    width = context.region.width if context.region is not None else 300
    return textwrap.wrap(text, max(18, int(width / (7.0 * max(context.preferences.system.ui_scale, 0.5))))) or [""]


classes = (InpaintClick, InpaintSettings, INPAINT_OT_add_box, INPAINT_GGT_box_sides, INPAINT_OT_pick_point,
           INPAINT_OT_clear_click, INPAINT_OT_find_part, INPAINT_OT_paint, INPAINT_OT_clear_zones,
           INPAINT_OT_fill_box, INPAINT_OT_fit_box, INPAINT_OT_judge_picture, INPAINT_OT_improve_picture,
           INPAINT_OT_enhance_picture, INPAINT_OT_restore_picture, INPAINT_OT_use_idea, INPAINT_OT_show_found, INPAINT_OT_make_picture,
           INPAINT_OT_regenerate, INPAINT_OT_save_job, INPAINT_OT_send_job, INPAINT_OT_load_result,
           INPAINT_OT_clean_up, INPAINT_OT_compare, INPAINT_OT_keep_original,
           InpaintPreferences, INPAINT_OT_cloud_login,
           INPAINT_OT_show_picture, INPAINT_OT_start_local, INPAINT_OT_stop_local, INPAINT_OT_copy_steps, INPAINT_OT_test_connection, INPAINT_OT_check, INPAINT_PT_panel)


paint_keys = []


@bpy.app.handlers.persistent
def colours_changed(*_):
    """After Blender's undo or redo, or another file: the colours are read from the mesh again."""
    painter.update(cache=None, batches=None, counts=None)


def register():
    global draw_handler
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.inpaint = bpy.props.PointerProperty(type=InpaintSettings)
    draw_handler = bpy.types.SpaceView3D.draw_handler_add(draw_boxes, (), "WINDOW", "POST_VIEW")
    bpy.app.handlers.depsgraph_update_post.append(after_change)
    for handlers in (bpy.app.handlers.undo_post, bpy.app.handlers.redo_post, bpy.app.handlers.load_post):
        handlers.append(colours_changed)
    config = bpy.context.window_manager.keyconfigs.addon
    if config is not None:  # none when Blender runs without a window
        keymap = config.keymaps.new(name="3D View", space_type="VIEW_3D")
        for key, zone in (("Q", RED), ("W", NO_COLOUR)):  # Q paints red, W erases (E and R: yellow, green, once painting)
            item = keymap.keymap_items.new(INPAINT_OT_paint.bl_idname, key, "PRESS")
            item.properties.zone = zone
            paint_keys.append((keymap, item))


def unregister():
    global draw_handler, brush_handler
    stop_local()
    for keymap, item in paint_keys:
        keymap.keymap_items.remove(item)
    paint_keys.clear()
    for handlers in (bpy.app.handlers.undo_post, bpy.app.handlers.redo_post, bpy.app.handlers.load_post):
        if colours_changed in handlers:
            handlers.remove(colours_changed)
    if brush_handler is not None:
        bpy.types.SpaceView3D.draw_handler_remove(brush_handler, "WINDOW")
        brush_handler = None
    painter.update(cache=None, batches=None, counts=None, tool=None)
    if after_change in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(after_change)
    if draw_handler is not None:
        bpy.types.SpaceView3D.draw_handler_remove(draw_handler, "WINDOW")
        draw_handler = None
    del bpy.types.Scene.inpaint
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
