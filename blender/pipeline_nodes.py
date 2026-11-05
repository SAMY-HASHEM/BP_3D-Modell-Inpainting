"""BP Pipeline Nodes: the node pipeline as a node editor inside Blender.

A node tree of its own ("BP Pipeline", in the node editor's tree menu), with the same nodes as in ComfyUI:

    Pictures -> Sort (data sheet) -> Base model -> Refine part (x N) -> Combine -> Hair -> Show in scene

What each node does is in scripts/pipeline/nodes_core.py, shared with ComfyUI (comfyui/bp_inpainting), so
both do exactly the same. Blender's Geometry Nodes can't run outside programs (TRELLIS.2, the vision AI), so
this is its own node editor; only the hair node uses real Geometry Nodes. Every node shows its picture
(data sheet, before / after) and can put its model into the scene. Run = every node up to the selected one,
in order; a node whose result is already there (same settings) is done at once.

On a computer without the project's environment (e.g. a Windows PC at the uni) the heavy work runs on the uni
server: 3D view > sidebar (N) > BP > BP Server sends a picture folder, fetches the results and opens them.

Install: Edit > Preferences > Add-ons > arrow at the top right > Install from Disk > this one file, enable it. Then
3D view > sidebar (N) > BP > BP Setup: the rest (the project, the Inpaint add-on, the login, the pictures) is buttons.
"""
import importlib.util
import json
import os
import subprocess
import threading
import time

import bpy
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, StringProperty

bl_info = {
    "name": "BP Pipeline Nodes",
    "author": "Samy Hashem",
    "version": (0, 3, 0),
    "blender": (4, 2, 0),
    "location": "Node Editor > BP Pipeline",
    "description": "Pictures -> data sheet -> base model -> refined parts -> final model, as nodes (TRELLIS.2)",
    "category": "Node",
}

PROJECT_NOTE = os.path.expanduser("~/.config/inpainting3d/project_path")  # written by `cloud.py device`


def find_project():
    """The project folder: next to this file when it is linked (Linux), or as noted by `cloud.py device` (on
    Windows "Install from Disk" copies this file away from the project), or a usual place."""
    here = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    candidates = [here]
    if os.path.isfile(PROJECT_NOTE):
        candidates.append(open(PROJECT_NOTE).read().strip())
    home = os.path.expanduser("~")
    candidates += [os.path.join(home, n) for n in ("inpainting-3D", "BP_3D-Modell-Inpainting", "bp_samy_hashem")]
    for path in candidates:
        if os.path.isfile(os.path.join(path, "scripts", "pipeline", "nodes_core.py")):
            return path
    return None


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class core:  # noqa: N801 (until the project is there; the BP Setup box gets it)
    PARTS = ["head", "hair", "bust", "belly", "back", "butt", "thighs", "lower_legs", "feet"]
    VISION_MODELS = ["Qwen/Qwen3-VL-8B-Instruct"]


PROJECT, CAN_RUN = None, False


def use_project():
    """Loads what the add-on needs from the project folder, if there is one."""
    global PROJECT, core, CAN_RUN
    PROJECT = find_project()
    if PROJECT:
        core = load_module("bp_nodes_core", os.path.join(PROJECT, "scripts", "pipeline", "nodes_core.py"))  # standard library only
    # the nodes run here only where the project's environment is (Linux, with conda); else through the server
    CAN_RUN = bool(PROJECT) and os.name != "nt" and os.path.isdir(os.path.join(PROJECT, "conda", "envs"))


use_project()

TREE = "BPPipelineTree"
SEEDS = (42, 7, 123, 2024)  # the head's candidates for the Pick best node
state = {"queue": [], "process": None, "thread": None, "error": None, "tree": None, "log": ""}


# ------------------------------------------------------------------ tree and sockets

class BPPipelineTree(bpy.types.NodeTree):
    """Pictures to a 3D model with TRELLIS.2, step by step"""
    bl_idname = TREE
    bl_label = "BP Pipeline"
    bl_icon = "NODETREE"


def socket(name, idname, colour):
    def draw(self, context, layout, node, text):
        layout.label(text=text)
    return type(idname, (bpy.types.NodeSocket,), {"bl_idname": idname, "bl_label": name, "draw": draw,
                                                    "draw_color": lambda self, context, node: colour})


PicturesSocket = socket("Pictures", "BPPicturesSocket", (0.25, 0.47, 0.62, 1))
SortedSocket = socket("Data sheet", "BPSortedSocket", (0.53, 0.53, 0.67, 1))
ModelSocket = socket("Model", "BPModelSocket", (0.71, 0.55, 0.16, 1))


class BPNode:
    """What all BP nodes share: their result (as JSON), status, picture, and the buttons."""
    result: StringProperty(default="")
    status: StringProperty(default="")
    picture: StringProperty(default="")  # the node's picture (data sheet, before / after)

    @classmethod
    def poll(cls, tree):
        return tree.bl_idname == TREE

    def value(self, key):
        return json.loads(self.result).get(key) if self.result else None

    def upstream(self, name):
        sock = self.inputs.get(name)
        if not sock or not sock.is_linked:
            return None
        if sock.is_multi_input:  # in the order they were connected (Blender lists the newest first)
            return [link.from_node for link in sorted(sock.links, key=lambda l: getattr(l, "multi_input_sort_id", 0))]
        return sock.links[0].from_node

    def draw_result(self, layout):
        if self.status:
            layout.label(text=self.status, icon="ERROR" if self.status.startswith("failed") else "INFO")
        image = bpy.data.images.get(self.picture) if self.picture else None
        if image:
            image.preview_ensure()
            layout.template_icon(icon_value=image.preview.icon_id, scale=9)
            row = layout.row(align=True)
            row.operator("bp.view_picture", text="Picture", icon="IMAGE_DATA").node = self.name
            if self.value("model"):
                row.operator("bp.show_model", text="In scene", icon="IMPORT").node = self.name
        elif self.value("model"):
            layout.operator("bp.show_model", text="In scene", icon="IMPORT").node = self.name
        layout.operator("bp.run", text="Run up to here", icon="PLAY").node = self.name

    def draw_buttons(self, context, layout):
        for prop in getattr(self, "settings", ()):
            layout.prop(self, prop)
        self.draw_result(layout)


# ------------------------------------------------------------------ the nodes (settings as in ComfyUI)

class PicturesNode(BPNode, bpy.types.Node):
    """All pictures (and videos) of one character from a folder"""
    bl_idname, bl_label, bl_width_default = "BPPictures", "00 Pictures", 260
    settings = ("character", "folder")
    character: StringProperty(name="Name", default="morrigan")
    folder: StringProperty(name="Folder", default="pictures/local model", subtype="DIR_PATH")

    def init(self, context):
        self.outputs.new("BPPicturesSocket", "pictures")

    def job(self):
        return None, {"pictures": core.pictures(self.character, bpy.path.abspath(self.folder))}


class SortNode(BPNode, bpy.types.Node):
    """01: the vision AI sorts the pictures (data sheet)"""
    bl_idname, bl_label, bl_width_default = "BPSort", "01 Sort pictures (data sheet)", 300
    settings = ("model",)
    model: EnumProperty(name="Vision AI", items=[(m, m.split("/")[1], "") for m in core.VISION_MODELS])

    def init(self, context):
        self.inputs.new("BPPicturesSocket", "pictures")
        self.outputs.new("BPSortedSocket", "sorted")

    def job(self):
        job = core.sort(self.upstream("pictures").value("pictures"), self.model)
        return job, None


class BaseNode(BPNode, bpy.types.Node):
    """02: the whole figure (TRELLIS.2)"""
    bl_idname, bl_label, bl_width_default = "BPBase", "02 Base model", 280
    settings = ("mode", "resolution", "steps", "faces", "seed")
    # multidiffusion: every step looks at ALL pictures (stochastic took turns: with 8 pictures in 12 steps the back
    # view got 1-2 steps, and the witch's skin at the butt was lost)
    mode: EnumProperty(name="Mode", items=[(m, m, "") for m in ("multidiffusion", "stochastic")])
    resolution: EnumProperty(name="Resolution", items=[(r, r, "") for r in ("1024", "1536", "512")])
    steps: IntProperty(name="Steps", default=12, min=4, max=100)
    faces: IntProperty(name="Faces", default=4_000_000, min=100_000, max=16_000_000, step=100_000)
    seed: IntProperty(name="Seed", default=42, min=0)

    def init(self, context):
        self.inputs.new("BPSortedSocket", "sorted")
        self.outputs.new("BPModelSocket", "model")

    def job(self):
        return core.base(self.upstream("sorted").value("sorted"), self.mode, self.resolution, self.steps, self.faces,
                         self.seed), None


class RefineNode(BPNode, bpy.types.Node):
    """04: one part zoomed in and repainted (auto = the lessons of rounds 1-10)"""
    bl_idname, bl_label, bl_width_default = "BPRefine", "04 Refine part", 280
    settings = ("part", "shape", "colour_pictures", "shape_pictures", "texture_guidance", "steps", "seed")
    part: EnumProperty(name="Part", items=[(p, p, "") for p in core.PARTS])
    shape: EnumProperty(name="Shape", items=[(s, s, "") for s in ("auto", "keep (colours only)", "regenerate")])
    colour_pictures: IntProperty(name="Colour pictures", default=1, min=0, max=8, description="0 = all")
    shape_pictures: EnumProperty(name="Shape pictures", items=[(s, s, "") for s in ("auto", "none", "all")])
    texture_guidance: FloatProperty(name="Texture guidance", default=3.0, min=1.0, max=10.0)
    steps: IntProperty(name="Steps", default=25, min=4, max=100)
    seed: IntProperty(name="Seed", default=42, min=0)

    def init(self, context):
        self.inputs.new("BPModelSocket", "base")
        self.inputs.new("BPSortedSocket", "sorted")
        self.outputs.new("BPModelSocket", "part")

    def job(self):
        return core.refine(self.upstream("base").value("model"), self.upstream("sorted").value("sorted"), self.part,
                           self.shape, self.colour_pictures, self.shape_pictures, round(self.texture_guidance, 2),
                           self.steps, self.seed), None


class PickBestNode(BPNode, bpy.types.Node):
    """04b: the local vision AI picks the candidate closest to the artist's picture (e.g. the head with several seeds)"""
    bl_idname, bl_label, bl_width_default = "BPPickBest", "04b Pick best (local AI)", 300

    def init(self, context):
        self.inputs.new("BPModelSocket", "candidates", use_multi_input=True)
        self.outputs.new("BPModelSocket", "best")

    def job(self):
        return core.pick_best([n.value("model") for n in (self.upstream("candidates") or []) if n.value("model")]), None


class CombineNode(BPNode, bpy.types.Node):
    """05: the parts put into the base (the first connected part wins where two overlap)"""
    bl_idname, bl_label, bl_width_default = "BPCombine", "05 Combine parts", 280

    def init(self, context):
        self.inputs.new("BPModelSocket", "base")
        self.inputs.new("BPModelSocket", "parts", use_multi_input=True)
        self.outputs.new("BPModelSocket", "model")

    def draw_buttons(self, context, layout):
        parts = self.upstream("parts") or []
        layout.label(text="order (first wins): " + ", ".join(getattr(n, "part", n.name) for n in parts))
        self.draw_result(layout)

    def job(self):
        parts = [n.value("model") for n in (self.upstream("parts") or [])]
        return core.combine(self.upstream("base").value("model"), [p for p in parts if p]), None


class HairNode(BPNode, bpy.types.Node):
    """06: real hair strands (Blender Geometry Nodes)"""
    bl_idname, bl_label, bl_width_default = "BPHair", "06 Hair strands", 260
    settings = ("count", "length", "clump")
    count: IntProperty(name="Strands", default=35000, min=1000, max=200000, step=1000)
    length: FloatProperty(name="Length", default=0.10, min=0.02, max=0.5)
    clump: FloatProperty(name="Clump", default=0.35, min=0.0, max=1.0)

    def init(self, context):
        self.inputs.new("BPModelSocket", "model")
        self.inputs.new("BPSortedSocket", "sorted")
        self.inputs.new("BPModelSocket", "head")
        self.outputs.new("BPModelSocket", "model")

    def job(self):
        head = self.upstream("head")
        return core.hair(self.upstream("model").value("model"), self.upstream("sorted").value("sorted"), self.count,
                         round(self.length, 3), round(self.clump, 3), head.value("model") if head else None,
                         blender=bpy.app.binary_path), None


class CompareNode(BPNode, bpy.types.Node):
    """99: models side by side, the same 4 cameras (clay = the shape alone)"""
    bl_idname, bl_label, bl_width_default = "BPCompare", "99 Compare", 260
    settings = ("clay",)
    clay: BoolProperty(name="Clay (shape only)", default=False)

    def init(self, context):
        self.inputs.new("BPModelSocket", "models", use_multi_input=True)

    def job(self):
        return core.compare([n.value("model") for n in (self.upstream("models") or []) if n.value("model")], self.clay), None


NODES = (PicturesNode, SortNode, BaseNode, RefineNode, PickBestNode, CombineNode, HairNode, CompareNode)


# ------------------------------------------------------------------ running

def order_up_to(node):
    """The node and every node it needs, each after the ones it needs."""
    done, out = set(), []

    def visit(n):
        if n.name in done:
            return
        done.add(n.name)
        for sock in n.inputs:
            for link in sock.links:
                visit(link.from_node)
        out.append(n.name)
    visit(node)
    return out


def run_steps(steps, folder):
    """In a thread: the job's commands one after another in the project environment."""
    try:
        for command, label, done in steps:
            if done and os.path.exists(done):
                continue
            state["log"] = f"{label}: running"
            quoted = " ".join("'" + str(c).replace("'", "'\\''") + "'" for c in command)
            shell = f"source '{PROJECT}/activate.sh' >/dev/null && exec {quoted}"
            env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
            env["PYTHONUNBUFFERED"] = "1"
            log = open(os.path.join(folder, "log_preview.txt" if label.endswith("preview") else "log.txt"), "w")
            state["process"] = subprocess.Popen(["bash", "-c", shell], stdout=log, stderr=subprocess.STDOUT, cwd=PROJECT, env=env)
            code = state["process"].wait()
            log.close()
            if code != 0 or (done and not os.path.exists(done)):
                state["error"] = f"{label} failed, see {folder}/log.txt"
                return
    except Exception as e:  # noqa: BLE001 (shown on the node)
        state["error"] = f"{type(e).__name__}: {e}"
    finally:
        state["process"] = None


def load_picture(node, path):
    if not path or not os.path.exists(path):
        return
    name = f"BP {node.name}"
    image = bpy.data.images.get(name)
    if image and bpy.path.abspath(image.filepath) == path:
        image.reload()
    else:
        if image:
            bpy.data.images.remove(image)
        image = bpy.data.images.load(path)
        image.name = name
    node.picture = image.name


def finish(node, outputs, job):
    outputs = core.resolve(dict(outputs))
    node.result = json.dumps(outputs)
    node.status = f"done: {os.path.relpath(job['folder'], PROJECT)}" if job else "done"
    load_picture(node, outputs.get("sheet") or (outputs.get("renders") or [None])[0])


def step():
    """Timer: starts the next node, or waits for the running one. Returns the seconds until it runs again."""
    tree = bpy.data.node_groups.get(state["tree"] or "")
    if tree is None:
        state["queue"] = []
        return None
    if state["thread"] and state["thread"].is_alive():
        node = tree.nodes.get(state["queue"][0])
        if node:
            node.status = state["log"]
        return 1.0
    if state["thread"]:  # a node's steps just finished
        state["thread"] = None
        node = tree.nodes.get(state["queue"].pop(0))
        if state["error"]:
            node.status = "failed: " + state["error"]
            state["queue"] = []
            redraw()
            return None
        finish(node, state["current"][1], state["current"][0])
    while state["queue"]:
        node = tree.nodes.get(state["queue"][0])
        try:
            job, outputs = node.job()
        except Exception as e:  # noqa: BLE001 (e.g. an input is not connected or not run yet)
            node.status = f"failed: {type(e).__name__}: {e}"[:200]
            state["queue"] = []
            break
        if job is not None and job["folder"] is None:  # skipped (e.g. no picture shows this part)
            job, outputs = None, job["outputs"]
        if job is None:  # nothing to compute (Pictures, or skipped)
            finish(node, outputs, None)
            if outputs.get("skipped"):
                node.status = "skipped: " + outputs["skipped"]
            state["queue"].pop(0)
            continue
        if all(done and os.path.exists(done) for _, _, done in job["steps"]):  # computed before
            finish(node, job["outputs"], job)
            state["queue"].pop(0)
            continue
        node.status = "running..."
        state["current"], state["error"] = (job, job["outputs"]), None
        state["thread"] = threading.Thread(target=run_steps, args=(job["steps"], job["folder"]), daemon=True)
        state["thread"].start()
        redraw()
        return 1.0
    redraw()
    return None


def redraw():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "NODE_EDITOR":
                area.tag_redraw()


def run_blocking(tree, node_name):
    """Without the user interface (blender -b, for tests): the same as the Run button, waiting."""
    state.update(tree=tree.name, queue=order_up_to(tree.nodes[node_name]), thread=None, error=None)
    while True:
        delay = step()
        if delay is None:
            return
        time.sleep(delay)


# ------------------------------------------------------------------ operators and panel

class BP_OT_run(bpy.types.Operator):
    """Run every node up to this one (nodes whose result is already there are done at once)"""
    bl_idname, bl_label = "bp.run", "Run up to here"
    node: StringProperty()

    def execute(self, context):
        tree = context.space_data.edit_tree
        if not CAN_RUN:
            self.report({"ERROR"}, "No project environment on this computer: use 3D view > sidebar > BP Server")
            return {"CANCELLED"}
        if state["queue"]:
            self.report({"WARNING"}, "already running")
            return {"CANCELLED"}
        target = tree.nodes.get(self.node) or tree.nodes.active
        if target is None:
            self.report({"WARNING"}, "select a node")
            return {"CANCELLED"}
        state.update(tree=tree.name, queue=order_up_to(target), thread=None, error=None)
        bpy.app.timers.register(step, first_interval=0.1)
        return {"FINISHED"}


class BP_OT_cancel(bpy.types.Operator):
    """Stop the running node"""
    bl_idname, bl_label = "bp.cancel", "Cancel"

    def execute(self, context):
        state["queue"] = state["queue"][:1]
        if state["process"]:
            state["process"].kill()
        return {"FINISHED"}


class BP_OT_view_picture(bpy.types.Operator):
    """Show the node's picture big, in an Image Editor"""
    bl_idname, bl_label = "bp.view_picture", "Show picture"
    node: StringProperty()

    def execute(self, context):
        node = context.space_data.edit_tree.nodes[self.node]
        image = bpy.data.images.get(node.picture)
        areas = [a for w in context.window_manager.windows for a in w.screen.areas if a.type == "IMAGE_EDITOR"]
        if not areas:
            bpy.ops.wm.window_new()
            area = context.window_manager.windows[-1].screen.areas[0]
            area.type = "IMAGE_EDITOR"
            areas = [area]
        areas[0].spaces.active.image = image
        return {"FINISHED"}


class BP_OT_show_model(bpy.types.Operator):
    """Put the node's model into the scene (in its own collection, replacing an earlier one)"""
    bl_idname, bl_label = "bp.show_model", "Show model in the scene"
    node: StringProperty()

    def execute(self, context):
        node = context.space_data.edit_tree.nodes[self.node]
        model = node.value("model")
        name = f"BP {node.label or node.bl_label} ({node.name})"
        old = bpy.data.collections.get(name)
        if old:
            for obj in list(old.objects):
                bpy.data.objects.remove(obj, do_unlink=True)
            bpy.data.collections.remove(old)
        collection = bpy.data.collections.new(name)
        context.scene.collection.children.link(collection)
        before = set(bpy.data.objects)
        bpy.ops.import_scene.gltf(filepath=model["glb"])
        for obj in set(bpy.data.objects) - before:
            for c in obj.users_collection:
                c.objects.unlink(obj)
            collection.objects.link(obj)
        self.report({"INFO"}, f"{name}: {model['glb']}")
        return {"FINISHED"}


class BP_OT_new_pipeline(bpy.types.Operator):
    """A new BP Pipeline tree with all nodes connected (as the ComfyUI workflow)"""
    bl_idname, bl_label = "bp.new_pipeline", "New pipeline"
    character: StringProperty(name="Name", default="morrigan")
    folder: StringProperty(name="Pictures folder", default="pictures/local model", subtype="DIR_PATH")

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        tree = bpy.data.node_groups.new(f"BP Pipeline {self.character}", TREE)
        add = lambda kind, x, y, **props: _node(tree, kind, x, y, props)
        pics = add("BPPictures", 0, 0, character=self.character, folder=self.folder)
        sort = add("BPSort", 320, 0)
        link = lambda a, out, b, inp: tree.links.new(a.outputs[out], b.inputs[inp])
        link(pics, "pictures", sort, "pictures")
        base = add("BPBase", 680, 0)
        link(sort, "sorted", base, "sorted")
        parts = {}
        for k, part in enumerate(["lower_legs", "thighs", "feet", "belly", "back", "butt", "bust"]):
            parts[part] = add("BPRefine", 1040, 400 - k * 420, part=part)
            parts[part].label = f"04 {part}"
            link(base, "model", parts[part], "base")
            link(sort, "sorted", parts[part], "sorted")
        pick = add("BPPickBest", 1420, -1700)  # the face is a lottery: 4 seeds, the local AI picks
        for k, seed in enumerate(SEEDS):
            head = add("BPRefine", 1040, -1700 - k * 420, part="head", seed=seed)
            head.label = f"04 head seed {seed}"
            link(base, "model", head, "base")
            link(sort, "sorted", head, "sorted")
            link(head, "part", pick, "candidates")
        parts["head"] = pick
        combine = add("BPCombine", 1420, 0)
        link(base, "model", combine, "base")
        for part in ["head", "bust", "belly", "butt", "back", "thighs", "lower_legs", "feet"]:  # the first wins
            link(parts[part], parts[part].outputs[0].name, combine, "parts")
        hair = add("BPHair", 1780, 0)
        link(combine, "model", hair, "model")
        link(sort, "sorted", hair, "sorted")
        link(parts["head"], "best", hair, "head")
        compare = add("BPCompare", 1780, -520)
        link(base, "model", compare, "models")
        link(combine, "model", compare, "models")
        if context.space_data and context.space_data.type == "NODE_EDITOR":
            context.space_data.node_tree = tree
        return {"FINISHED"}


def _node(tree, kind, x, y, props):
    node = tree.nodes.new(kind)
    node.location = (x, y)
    for key, value in props.items():
        setattr(node, key, value)
    return node


class BP_PT_pipeline(bpy.types.Panel):
    bl_space_type, bl_region_type, bl_category, bl_label = "NODE_EDITOR", "UI", "BP", "BP Pipeline"

    @classmethod
    def poll(cls, context):
        return context.space_data.tree_type == TREE

    def draw(self, context):
        layout = self.layout
        layout.operator("bp.new_pipeline", icon="ADD")
        tree = context.space_data.edit_tree
        if tree and tree.nodes.active:
            layout.operator("bp.run", text=f"Run up to: {tree.nodes.active.label or tree.nodes.active.bl_label}",
                            icon="PLAY").node = tree.nodes.active.name
        if state["queue"]:
            layout.label(text=f"running: {state['queue'][0]} ({len(state['queue'])} left)", icon="TIME")
            layout.label(text=state["log"])
            layout.operator("bp.cancel", icon="CANCEL")


# ------------------------------------------------------------------ the server (any computer, also Windows)

server = {"message": "", "auto": False, "busy": False}


def cloud_module():
    return load_module("bp_cloud", os.path.join(PROJECT, "scripts", "server", "cloud.py"))


def in_background(work):
    """Network work in a thread, so that Blender stays usable; the panel shows its message."""
    def run():
        server["busy"] = True
        try:
            work()
        except Exception as e:  # noqa: BLE001 (shown in the panel)
            server["message"] = f"{type(e).__name__}: {e}"[:120]
        server["busy"] = False
    threading.Thread(target=run, daemon=True).start()


def fetch_now():
    def work():
        module = cloud_module()
        new = module.fetch_results(module.Cloud())
        server["message"] = f"{time.strftime('%H:%M')}: {new} new file(s)"
    in_background(work)


def auto_fetch():
    """Timer: every 10 minutes, while the switch is on."""
    if not server["auto"]:
        return None
    if not server["busy"]:
        fetch_now()
    return 600.0


def runs():
    folder = os.path.join(PROJECT, "outputs", "server_runs") if PROJECT else ""
    return sorted(os.listdir(folder), reverse=True) if folder and os.path.isdir(folder) else []


def run_files(run, *types):
    folder = os.path.join(PROJECT, "outputs", "server_runs", run)
    found = []
    for root, _, files in os.walk(folder):
        found += [os.path.join(root, f) for f in files if f.lower().endswith(types)]
    return sorted(found)


# ------------------------------------------------------------------ setup without a terminal

CODE_ZIP = "https://github.com/SAMY-HASHEM/BP_3D-Modell-Inpainting/archive/refs/heads/main.zip"
CONFIG = os.path.expanduser("~/.config/inpainting3d/cloud.json")  # the login, shared with the Inpaint add-on and cloud.py
WEBDAV = "https://uni-duisburg-essen.sciebo.de/remote.php/webdav"
setup = {"message": "", "busy": False, "enable_inpaint": False}


def inpaint_ready():
    import addon_utils
    return "inpaint_addon" in bpy.context.preferences.addons or any(
        m.__name__ == "inpaint_addon" for m in addon_utils.modules())


def has_pictures():
    folder = os.path.join(PROJECT or "", "pictures")
    return bool(PROJECT) and os.path.isdir(folder) and any(os.scandir(folder))


def get_code():
    """The project from GitHub, into the home folder (no Git needed). A Git checkout is left to Git. Replaces only the
    code: pictures/ and outputs/ stay."""
    import io
    import shutil
    import urllib.request
    import zipfile
    target = PROJECT or os.path.join(os.path.expanduser("~"), "BP_3D-Modell-Inpainting")
    if os.path.isdir(os.path.join(target, ".git")):
        subprocess.run(["git", "-C", target, "pull", "-q"], check=True)
    else:
        setup["message"] = "downloading the project..."
        data = urllib.request.urlopen(CODE_ZIP, timeout=300).read()
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for item in z.infolist():
                name = item.filename.split("/", 1)[1] if "/" in item.filename else ""
                if not name or item.is_dir():
                    continue
                path = os.path.join(target, *name.split("/"))
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with z.open(item) as source, open(path, "wb") as out:
                    shutil.copyfileobj(source, out)
    for folder in ("pictures", "outputs", os.path.join("outputs", "server_runs"), os.path.join("outputs", "cloud")):
        os.makedirs(os.path.join(target, folder), exist_ok=True)
    os.makedirs(os.path.dirname(PROJECT_NOTE), exist_ok=True)
    with open(PROJECT_NOTE, "w") as f:
        f.write(target)


def finish_setup():
    """Timer, in Blender's main thread: the Inpaint add-on (installed from the project) and its project folder."""
    if setup["busy"]:
        return 0.5
    if not setup["enable_inpaint"]:
        return None
    setup["enable_inpaint"] = False
    use_project()
    try:
        installed = os.path.join(bpy.utils.user_resource("SCRIPTS", path="addons"), "inpaint_addon.py")
        updated = inpaint_ready() and not os.path.islink(installed)  # a link to the project (Linux) is new already
        if not inpaint_ready() or updated:  # every update also brings the new add-on
            bpy.ops.preferences.addon_install(filepath=os.path.join(PROJECT, "blender", "inpaint_addon.py"), overwrite=True)
        bpy.ops.preferences.addon_enable(module="inpaint_addon")
        bpy.context.preferences.addons["inpaint_addon"].preferences.project = PROJECT
        bpy.ops.wm.save_userpref()
        setup["message"] = ("updated: restart Blender for the new Inpaint add-on" if updated else
                            "set up: now the Sciebo login" if not os.path.isfile(CONFIG) else "everything is set up")
    except Exception as e:  # noqa: BLE001 (shown in the panel)
        setup["message"] = f"{type(e).__name__}: {e}"[:120]
    for area in bpy.context.screen.areas if bpy.context.screen else []:
        area.tag_redraw()
    return None


def setup_work(work):
    def run():
        setup["busy"] = True
        try:
            work()
        except Exception as e:  # noqa: BLE001 (shown in the panel)
            setup["message"] = f"{type(e).__name__}: {e}"[:120]
        setup["busy"] = False
    threading.Thread(target=run, daemon=True).start()
    bpy.app.timers.register(finish_setup, first_interval=0.5)


class BP_OT_setup(bpy.types.Operator):
    """Get or update the project and set up the Inpaint add-on: nothing else to install, no terminal"""
    bl_idname, bl_label = "bp.setup", "Set up everything"

    def execute(self, context):
        def work():
            get_code()
            setup["enable_inpaint"] = True
            setup["message"] = "project ready"
        setup_work(work)
        return {"FINISHED"}


class BP_OT_login(bpy.types.Operator):
    """Test the Sciebo login and save it for both add-ons"""
    bl_idname, bl_label = "bp.login", "Save and test"

    def execute(self, context):
        wm = context.window_manager
        user, password = wm.bp_user.strip(), wm.bp_password.strip()
        wm.bp_password = ""  # never kept in a field

        def work():
            config = {"url": WEBDAV, "user": user, "password": password, "folder": "inpainting-jobs"}
            module = cloud_module()
            cloud = module.Cloud(config)
            try:
                cloud.make_folder()
                cloud.list()
            except Exception as e:  # noqa: BLE001
                setup["message"] = f"Sciebo refused the login: {e}"[:120]
                return
            os.makedirs(os.path.dirname(CONFIG), exist_ok=True)
            with open(os.open(CONFIG, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f:
                json.dump(config, f)
            setup["message"] = "login works"
        setup_work(work)
        return {"FINISHED"}


class BP_OT_pictures(bpy.types.Operator):
    """Fetch the picture folders from Sciebo"""
    bl_idname, bl_label = "bp.pictures", "Get pictures"

    def execute(self, context):
        def work():
            import types
            cloud_module().pictures_sync(types.SimpleNamespace(direction="down"))
            setup["message"] = "pictures ready"
        setup_work(work)
        return {"FINISHED"}


class BP_PT_setup(bpy.types.Panel):
    """A checklist: only what is still missing needs a click"""
    bl_space_type, bl_region_type, bl_category, bl_label = "VIEW_3D", "UI", "BP", "BP Setup"

    def draw(self, context):
        layout = self.layout
        check = lambda done, text: layout.label(text=text, icon="CHECKMARK" if done else "CHECKBOX_DEHLT")
        login = os.path.isfile(CONFIG)
        check(bool(PROJECT), "Project")
        check(inpaint_ready(), "Inpaint add-on")
        check(login, "Sciebo login")
        check(has_pictures(), "Pictures")
        layout.operator("bp.setup", text="Update project" if PROJECT and inpaint_ready() else "Set up everything", icon="IMPORT")
        if PROJECT and not login:
            box = layout.box()
            box.label(text="Sciebo > Einstellungen > Sicherheit > App-Passwort")
            box.prop(context.window_manager, "bp_user", text="Login name")
            box.prop(context.window_manager, "bp_password", text="App password")
            box.operator("bp.login", icon="CHECKMARK")
        if PROJECT and login and not has_pictures():
            layout.operator("bp.pictures", icon="IMAGE_DATA")
        if setup["busy"]:
            layout.label(text="working...", icon="TIME")
        if setup["message"]:
            layout.label(text=setup["message"])


class BP_OT_server_send(bpy.types.Operator):
    """Send a picture folder to the uni server: the whole pipeline (one job at a time, one GPU)"""
    bl_idname, bl_label = "bp.server_send", "Send to the server"

    def execute(self, context):
        folder = core.project_path(bpy.path.abspath(context.scene.bp_pictures))
        if not os.path.isdir(folder):
            self.report({"ERROR"}, f"no folder {folder}")
            return {"CANCELLED"}

        def work():
            import types
            module = cloud_module()
            module.pipeline(types.SimpleNamespace(folder=folder, name=None, rounds=3, threshold=85.0, code="main"))
            server["message"] = "sent: it starts within a minute if the worker runs"
        in_background(work)
        return {"FINISHED"}


class BP_OT_server_fetch(bpy.types.Operator):
    """Fetch everything new from the server now (Sciebo)"""
    bl_idname, bl_label = "bp.server_fetch", "Fetch new results"

    def execute(self, context):
        fetch_now()
        return {"FINISHED"}


class BP_OT_server_auto(bpy.types.Operator):
    """Fetch every 10 minutes, on or off"""
    bl_idname, bl_label = "bp.server_auto", "Fetch every 10 minutes"

    def execute(self, context):
        server["auto"] = not server["auto"]
        if server["auto"]:
            bpy.app.timers.register(auto_fetch, first_interval=0.1)
        return {"FINISHED"}


class BP_OT_server_open(bpy.types.Operator):
    """Load this result: a model into the scene, a picture into an Image Editor"""
    bl_idname, bl_label = "bp.server_open", "Open"
    path: StringProperty()

    def execute(self, context):
        if self.path.lower().endswith(".glb"):
            name = "BP " + os.path.basename(os.path.dirname(self.path))[-22:] + " " + os.path.basename(self.path)[:-4]
            collection = bpy.data.collections.new(name)
            context.scene.collection.children.link(collection)
            before = set(bpy.data.objects)
            bpy.ops.import_scene.gltf(filepath=self.path)
            for obj in set(bpy.data.objects) - before:
                for c in obj.users_collection:
                    c.objects.unlink(obj)
                collection.objects.link(obj)
        else:
            image = bpy.data.images.load(self.path, check_existing=True)
            areas = [a for w in context.window_manager.windows for a in w.screen.areas if a.type == "IMAGE_EDITOR"]
            if areas:
                areas[0].spaces.active.image = image
            else:
                self.report({"INFO"}, f"Loaded {image.name}: open it in an Image Editor")
        return {"FINISHED"}


class BP_PT_server(bpy.types.Panel):
    """The same on every computer, also on Windows: the heavy work runs on the uni server"""
    bl_space_type, bl_region_type, bl_category, bl_label = "VIEW_3D", "UI", "BP", "BP Server"

    def draw(self, context):
        layout = self.layout
        if not PROJECT:
            layout.label(text="First: BP Setup above", icon="INFO")
            return
        layout.prop(context.scene, "bp_pictures", text="Pictures")
        layout.operator("bp.server_send", icon="EXPORT")
        row = layout.row(align=True)
        row.operator("bp.server_fetch", icon="FILE_REFRESH")
        row.operator("bp.server_auto", text="Auto 10 min", icon="CHECKBOX_HLT" if server["auto"] else "CHECKBOX_DEHLT")
        if server["message"]:
            layout.label(text=server["message"])
        for run in runs()[:4]:
            box = layout.box()
            box.label(text=run[-34:], icon="FILE_FOLDER")
            for path in run_files(run, ".glb", "heatmap_r1.jpg", "heatmap_r2.jpg", "heatmap_r3.jpg", "contact.jpg", "01_data_sheet.jpg"):
                label = os.path.relpath(path, os.path.join(PROJECT, "outputs", "server_runs", run))
                box.operator("bp.server_open", text=label[-36:], icon="MESH_DATA" if path.endswith(".glb") else "IMAGE_DATA").path = path


class BP_MT_add(bpy.types.Menu):
    bl_idname, bl_label = "BP_MT_add", "BP Pipeline"

    def draw(self, context):
        for cls in NODES:
            op = self.layout.operator("node.add_node", text=cls.bl_label)
            op.type, op.use_transform = cls.bl_idname, True


def add_menu(self, context):
    if context.space_data.tree_type == TREE:
        self.layout.menu("BP_MT_add")


CLASSES = (BPPipelineTree, PicturesSocket, SortedSocket, ModelSocket, *NODES, BP_OT_run, BP_OT_cancel,
           BP_OT_view_picture, BP_OT_show_model, BP_OT_new_pipeline, BP_PT_pipeline, BP_MT_add,
           BP_OT_setup, BP_OT_login, BP_OT_pictures, BP_PT_setup,
           BP_OT_server_send, BP_OT_server_fetch, BP_OT_server_auto, BP_OT_server_open, BP_PT_server)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.NODE_MT_add.append(add_menu)
    bpy.types.Scene.bp_pictures = StringProperty(name="Pictures", default="pictures/monster", subtype="DIR_PATH")
    bpy.types.WindowManager.bp_user = StringProperty(name="Login name")
    bpy.types.WindowManager.bp_password = StringProperty(name="App password", subtype="PASSWORD")


def unregister():
    del bpy.types.WindowManager.bp_user
    del bpy.types.WindowManager.bp_password
    del bpy.types.Scene.bp_pictures
    bpy.types.NODE_MT_add.remove(add_menu)
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
