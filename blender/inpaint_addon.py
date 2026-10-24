"""Blender add-on: select faces of a model, press a button, and TRELLIS.2 regenerates that area.

The work is done by scripts/inpaint_server.py, which must be running in the TRELLIS.2 environment.
This add-on only sends the job (the model, the selected area, the picture) and loads the result.

Install: Edit > Preferences > Add-ons > (arrow at the top right) > Install from Disk... > this file.
Use:     3D view sidebar (N key) > Inpaint tab.
"""
import json
import os
import urllib.request

import bpy
import numpy as np

bl_info = {
    "name": "TRELLIS.2 Inpainting",
    "author": "Samy Hashem",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > Inpaint",
    "description": "Regenerate the selected faces of a model with TRELLIS.2",
    "category": "3D View",
}

SERVER = "http://127.0.0.1:8765"
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
    return np.stack([points[:, 0], points[:, 2], -points[:, 1]], 1)


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
    local = np.concatenate([centers.reshape(-1, 3)[faces], co.reshape(-1, 3)[corners]])
    m = np.array(obj.matrix_world)
    return local @ m[:3, :3].T + m[:3, 3], int(faces.sum())


class InpaintSettings(bpy.types.PropertyGroup):
    picture: bpy.props.StringProperty(name="Picture", subtype="FILE_PATH",
                                      description="Picture that guides the new area (usually the model's own picture)")
    seed: bpy.props.IntProperty(name="Seed", default=1, min=0, description="Another seed gives another result")
    grow: bpy.props.IntProperty(name="Grow", default=2, min=0, max=8,
                                description="Grow the selected area by this many voxels (of 64 per side)")
    steps: bpy.props.IntProperty(name="Steps", default=12, min=4, max=50, description="Sampling steps per stage")
    folder: bpy.props.StringProperty(name="Work folder", subtype="DIR_PATH",
                                     default=os.path.expanduser("~/inpainting-3D/outputs/blender"),
                                     description="Where the model, the selection and the results are saved")


class INPAINT_OT_regenerate(bpy.types.Operator):
    bl_idname = "inpaint.regenerate"
    bl_label = "Regenerate selection"
    bl_description = "Send the model and its selected faces to TRELLIS.2 and regenerate that area"

    @classmethod
    def poll(cls, context):
        return context.active_object is not None and context.active_object.type == "MESH" and status["job"] is None

    def execute(self, context):
        settings = context.scene.inpaint
        obj = context.active_object
        picture = bpy.path.abspath(settings.picture)
        if not os.path.isfile(picture):
            self.report({"ERROR"}, "Choose the picture first")
            return {"CANCELLED"}
        server = ask("/status")
        if server is None:
            self.report({"ERROR"}, "The inpaint server isn't running: python scripts/inpaint_server.py")
            return {"CANCELLED"}
        if server["busy"]:
            self.report({"ERROR"}, "The server is busy with another job")
            return {"CANCELLED"}

        if obj.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")  # also writes the Edit Mode selection into the mesh
        points, count = selected_points(obj)
        if count == 0:
            self.report({"ERROR"}, "Select the faces to regenerate first (Edit Mode)")
            return {"CANCELLED"}

        folder = bpy.path.abspath(settings.folder)
        os.makedirs(folder, exist_ok=True)
        name = bpy.path.clean_name(obj.name)
        model = os.path.join(folder, f"{name}.glb")
        selection = os.path.join(folder, f"{name}_selection.npy")
        out = os.path.join(folder, f"{name}_inpaint_seed{settings.seed}.glb")
        self.report({"INFO"}, f"Exporting the model ({len(obj.data.polygons):,} faces)...")
        for other in context.selected_objects:
            other.select_set(False)
        obj.select_set(True)
        bpy.ops.export_scene.gltf(filepath=model, export_format="GLB", use_selection=True,
                                  export_image_format="AUTO", export_materials="EXPORT")
        np.save(selection, to_gltf(points))

        answer = ask("/inpaint", {"model": model, "picture": picture, "points": selection, "out": out,
                                  "seed": settings.seed, "grow": settings.grow, "steps": settings.steps}, timeout=5)
        if not answer or not answer.get("started"):
            self.report({"ERROR"}, f"The server didn't start the job: {answer}")
            return {"CANCELLED"}
        status["job"] = {"object": obj.name, "out": out, "seed": settings.seed}
        status["message"] = f"started: {count:,} faces selected"
        bpy.app.timers.register(follow_job, first_interval=1.0)
        return {"FINISHED"}


def follow_job():
    """Timer: show the server's progress; load the result when it is done."""
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
    elif state["result"]:
        load_result(job, state["result"])
    redraw()
    return None


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
    status["message"] = f"done: {new[0].name if new else path}"


class INPAINT_PT_panel(bpy.types.Panel):
    bl_label = "TRELLIS.2 Inpainting"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Inpaint"

    def draw(self, context):
        settings = context.scene.inpaint
        layout = self.layout
        layout.prop(settings, "picture")
        row = layout.row()
        row.prop(settings, "seed")
        row.prop(settings, "grow")
        layout.prop(settings, "steps")
        layout.label(text="1. Edit Mode (Tab), select faces")
        layout.operator(INPAINT_OT_regenerate.bl_idname, icon="MOD_REMESH")
        box = layout.box()
        row = box.row()
        row.label(text=f"Server: {status['server']}")
        row.operator(INPAINT_OT_check.bl_idname, text="", icon="FILE_REFRESH")
        if status["message"]:
            box.label(text=status["message"])
        layout.prop(settings, "folder")


class INPAINT_OT_check(bpy.types.Operator):
    bl_idname = "inpaint.check_server"
    bl_label = "Check server"
    bl_description = "Ask whether the inpaint server is running"

    def execute(self, context):
        state = ask("/status")
        status["server"] = "not running" if state is None else ("busy" if state["busy"] else state["message"])
        return {"FINISHED"}


classes = (InpaintSettings, INPAINT_OT_regenerate, INPAINT_OT_check, INPAINT_PT_panel)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.inpaint = bpy.props.PointerProperty(type=InpaintSettings)


def unregister():
    del bpy.types.Scene.inpaint
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
