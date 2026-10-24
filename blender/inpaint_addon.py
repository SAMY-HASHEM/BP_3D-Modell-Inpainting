"""Blender add-on: mark an area of a model, press a button, and TRELLIS.2 regenerates that area.

The area is the selected faces (Edit Mode) and/or "space boxes": boxes that mark empty space where
something new may grow (e.g. a statue on the ground). Optionally a prompt (and a reference picture)
says what should be there: the server then first paints it into a picture of the model, seen from the
current view, and TRELLIS.2 builds that picture's change in 3D.

The work is done by scripts/inpaint_server.py, which must be running in the TRELLIS.2 environment.

Install: Edit > Preferences > Add-ons > (arrow at the top right) > Install from Disk... > this file.
Use:     3D view sidebar (N key) > Inpaint tab.
"""
import json
import os
import urllib.request

import bpy
import numpy as np
from mathutils import Vector

bl_info = {
    "name": "TRELLIS.2 Inpainting",
    "author": "Samy Hashem",
    "version": (1, 1, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > Inpaint",
    "description": "Regenerate an area of a model with TRELLIS.2, optionally guided by a prompt",
    "category": "3D View",
}

SERVER = "http://127.0.0.1:8765"
BOX = "InpaintBox"
status = {"server": "not checked", "message": "", "job": None}


def ask(path, data=None, timeout=1.0):
    """JSON request to the inpaint server; None if it isn't running."""
    request = urllib.request.Request(SERVER + path, data=json.dumps(data).encode() if data is not None else None,
                                     headers={"Content-Type": "application/json"})
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
    """The model to change: the active object, or the last model used when a box is active."""
    obj, settings = context.active_object, context.scene.inpaint
    if obj is not None and obj.type == "MESH" and not is_box(obj):
        settings.model = obj
    return settings.model


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


def box_points(model):
    """Points filling every visible space box (world), about 2 per voxel of TRELLIS.2's 64 grid."""
    size = max(model.dimensions)
    points = []
    for box in bpy.data.objects:
        if not is_box(box) or not box.visible_get():
            continue
        m = np.array(box.matrix_world)
        steps = [max(2, int(np.linalg.norm(m[:3, i]) * 2 / size * 128)) for i in range(3)]
        grid = np.stack(np.meshgrid(*[np.linspace(-1, 1, k) for k in steps], indexing="ij"), -1).reshape(-1, 3)
        points.append(grid @ m[:3, :3].T + m[:3, 3])
    return np.concatenate(points) if points else np.zeros((0, 3))


def area_points(operator, context):
    """The model and the marked area (glTF coordinates), or None after reporting what's missing."""
    model = target_model(context)
    if model is None:
        operator.report({"ERROR"}, "Click the model first")
        return None
    if context.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")  # also writes the Edit Mode selection into the mesh
    faces, count = selected_points(model)
    boxes = box_points(model)
    if count == 0 and len(boxes) == 0:
        operator.report({"ERROR"}, "Mark an area: select faces (Edit Mode) or add a space box")
        return None
    return model, to_gltf(np.concatenate([faces, boxes])), count, len(boxes) > 0


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
        operator.report({"ERROR"}, "The inpaint server isn't running: python scripts/inpaint_server.py")
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
    model: bpy.props.PointerProperty(name="Model", type=bpy.types.Object, description="The model being changed")
    folder: bpy.props.StringProperty(name="Work folder", subtype="DIR_PATH",
                                     default=os.path.expanduser("~/inpainting-3D/outputs/blender"),
                                     description="Where the model, the area, the pictures and the results are saved")


class INPAINT_OT_add_box(bpy.types.Operator):
    bl_idname = "inpaint.add_box"
    bl_label = "Add space box"
    bl_description = "Add a box that marks empty space where something new may grow (move: G, size: S, turn: R)"

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
        self.report({"INFO"}, "Space box added: move it with G, size it with S (X/Y/Z for one direction)")
        return {"FINISHED"}


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
        reference = bpy.path.abspath(settings.reference) if settings.reference else ""
        if reference and not os.path.isfile(reference):
            self.report({"ERROR"}, "The reference picture doesn't exist")
            return {"CANCELLED"}
        if not server_ready(self):
            return {"CANCELLED"}
        area = area_points(self, context)
        if area is None:
            return {"CANCELLED"}
        model, points, _, _ = area
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
    bl_description = "Send the model and the marked area to TRELLIS.2 and regenerate that area after the Picture"

    @classmethod
    def poll(cls, context):
        return status["job"] is None

    def execute(self, context):
        settings = context.scene.inpaint
        picture = bpy.path.abspath(settings.picture)
        if not os.path.isfile(picture):
            self.report({"ERROR"}, "Choose the picture first (or make one from a prompt)")
            return {"CANCELLED"}
        if not server_ready(self):
            return {"CANCELLED"}
        area = area_points(self, context)
        if area is None:
            return {"CANCELLED"}
        model, points, count, has_boxes = area
        folder = bpy.path.abspath(settings.folder)
        self.report({"INFO"}, f"Exporting the model ({len(model.data.polygons):,} faces)...")
        path = export_model(context, model, folder)
        name = bpy.path.clean_name(model.name)
        np.save(os.path.join(folder, f"{name}_area.npy"), points)
        out = os.path.join(folder, f"{name}_inpaint_seed{settings.seed}.glb")
        return start(self, "/inpaint", {"model": path, "picture": picture, "points": os.path.join(folder, f"{name}_area.npy"),
                                        "out": out, "seed": settings.seed, "grow": settings.grow, "steps": settings.steps},
                     {"kind": "inpaint", "object": model.name, "seed": settings.seed})


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


class INPAINT_OT_show_picture(bpy.types.Operator):
    bl_idname = "inpaint.show_picture"
    bl_label = "Show big"
    bl_description = "Open the picture's before/after preview in its own window"

    def execute(self, context):
        path = os.path.splitext(bpy.path.abspath(context.scene.inpaint.picture))[0] + "_preview.jpg"
        image = bpy.data.images.get(os.path.basename(path))
        if image is None:
            if not os.path.isfile(path):
                self.report({"ERROR"}, "No preview for this picture")
                return {"CANCELLED"}
            image = bpy.data.images.load(path)
        bpy.ops.wm.window_new()
        area = context.window_manager.windows[-1].screen.areas[0]
        area.type = "IMAGE_EDITOR"
        area.spaces.active.image = image
        return {"FINISHED"}


class INPAINT_OT_check(bpy.types.Operator):
    bl_idname = "inpaint.check_server"
    bl_label = "Check server"
    bl_description = "Ask whether the inpaint server is running"

    def execute(self, context):
        state = ask("/status")
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
        if settings.model is not None:
            box.label(text=f"Model: {settings.model.name}")

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
        box.prop(settings, "steps")
        box.operator(INPAINT_OT_regenerate.bl_idname, text="2. Regenerate area", icon="PLAY")

        layout.prop(settings, "folder")


classes = (InpaintSettings, INPAINT_OT_add_box, INPAINT_OT_make_picture, INPAINT_OT_regenerate,
           INPAINT_OT_show_picture, INPAINT_OT_check, INPAINT_PT_panel)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.inpaint = bpy.props.PointerProperty(type=InpaintSettings)


def unregister():
    del bpy.types.Scene.inpaint
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
