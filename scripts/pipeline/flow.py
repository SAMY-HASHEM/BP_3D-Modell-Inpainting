"""Runs a node flow from a file, without Blender or ComfyUI: the same nodes (nodes_core.py), connected in a .json.

A flow file is a node tree written down: every node has an id, a type, its settings, and "in": which node's
result goes into which of its inputs (as the links between sockets in a node editor).

    {"name": "girl_crown", "nodes": [
        {"id": "object", "type": "input", "files": ["pictures/girl.jpg"]},
        {"id": "object_sheet", "type": "data_sheet", "role": "object", "in": {"pictures": "object"}}, ...]}

    source activate.sh
    python scripts/pipeline/flow.py flows/girl_crown.json                 # every node, in order
    python scripts/pipeline/flow.py flows/girl_crown.json --upto edit     # only up to this node
    python scripts/pipeline/flow.py flows/girl_crown.json --structure     # only write the structure picture

A node whose result is already there (the same settings and inputs) is done at once, so a flow can be stopped and
started again. Nodes that don't depend on each other run at the same time: the server's work side by side, this
PC's GPU work (data sheet, edit, judge) one after another.

Writes outputs/<name>/work/flow.html (the structure with every node's state and picture, refreshed while it
runs) and flow_state.json (every node's result).
"""
import argparse
import html
import json
import os
import subprocess
import sys
import threading
import time

import nodes_core as core

BLENDER = os.environ.get("BLENDER") or next(  # the usual place, or a portable Blender unpacked into the project's blender/
    (p for p in (os.path.expanduser("~/Applications/blender/blender"), os.path.join(core.PROJECT, "blender", "blender")) if os.path.isfile(p)),
    os.path.expanduser("~/Applications/blender/blender"))


def models_of(results):
    """The models a candidates input gets: all of a node that made several (seeds), else its one model."""
    found = []
    for result in results:
        found += result.get("models") or [result.get("model")]
    return [m for m in found if m]


# type: (title, the node's sockets in, what runs where, how its job is made from the inputs i and its settings s)
TYPES = {
    "input": ("00 Input", [], "", lambda i, s, name: (None, {"pictures": core.inputs(name, s["files"])})),
    "data_sheet": ("01 Data sheet", ["pictures"], "this PC: local AI",
                   lambda i, s, name: (core.data_sheet(i["pictures"]["pictures"], s.get("role", "object"), s.get("frames", 8),
                                                       s.get("ai", True), s.get("what", ""), s.get("sides", "")), None)),
    "cut_part": ("03a The painted part", ["edit"], "this PC: measured",
                 lambda i, s, name: (core.cut_part(i["edit"]["edit"], s.get("change", 60)), None)),
    "part_views": ("01b The part's sides", ["data"], "this PC: picture AI",
                   lambda i, s, name: (core.part_views(i["data"]["data"], s.get("way", "angles"), s.get("seed", 1)), None)),
    "base": ("02 Base model", ["data"], "TRELLIS.2",
             lambda i, s, name: (core.base_model(i["data"]["data"], s.get("where", "server"), s.get("mode", "multidiffusion"),
                                                 str(s.get("resolution", "1024")), s.get("steps", 25), s.get("seed", 42), s.get("colours", "all"), s.get("view"),
                                                 s.get("given")), None)),
    "pick_seed": ("02b Pick seed", ["data", "candidates"], "this PC: measured",
                  lambda i, s, name: (core.pick_seed(i["data"]["data"], models_of(i["candidates"])), None)),
    "edit": ("03 Edit", ["model", "part", "object"], "this PC: measured",
             lambda i, s, name: (core.edit(i["model"]["model"], (i.get("part") or {}).get("data"), i["object"]["data"], s.get("replace", False),
                                           s.get("width", 0.75), s.get("sink", 0.15), s.get("head", 0.12), s.get("fit", "between"),
                                           s.get("place"), s.get("side"), s.get("hand"), s.get("after"), s.get("prompt"), s.get("prompt_seed", 1), s.get("at")), None)),
    "place": ("03b Place the part", ["part_model", "edit"], "this PC: measured",
              lambda i, s, name: (core.place(i["part_model"]["model"], i["edit"]["edit"], s.get("zoom", 0.25), s.get("size", 0.8),
                                             s.get("sides", 4), s.get("behind", 4), s.get("min_voxels", 5)), None)),
    "check": ("03c Judge the plan", ["edit"], "this PC: measured + local AI",
              lambda i, s, name: (core.plan_check(i["edit"]["edit"], s.get("ai", True)), None)),
    "part_colours": ("05b Colours from the part", ["model", "edit"], "this PC: measured",
                     lambda i, s, name: (core.colours_from_part(i["model"]["model"], i["edit"]["edit"]), None)),
    "finish": ("06 Into the whole", ["model", "edit"], "this PC: measured",
               lambda i, s, name: (core.finish(i["model"]["model"], i["edit"]["edit"]), None)),
    "inpaint": ("04 Inpaint", ["model", "edit"], "TRELLIS.2",
                lambda i, s, name: (core.inpaint_step(i["model"]["model"], i["edit"]["edit"], s.get("step", "shape"),
                                                      s.get("seeds", [1]), s.get("where", "server")), None)),
    "judge": ("05 Judge", ["candidates", "edit"], "this PC: measured + local AI",
              lambda i, s, name: (core.edit_judge(models_of(i["candidates"]), i["edit"]["edit"], s.get("step", "shape"),
                                                  s.get("ai", True), s.get("keep_min", 85), s.get("part_min", 70)), None)),
}
LOCAL_GPU = {"data_sheet", "part_views", "pick_seed", "edit", "place", "check", "judge", "part_colours", "finish", "cut_part"}  # one after another on this PC's GPU
COLOURS = {"input": "#4a7a9c", "data_sheet": "#8a8ab0", "base": "#b58c2a", "pick_seed": "#5f9c6a", "edit": "#a8623c",
           "inpaint": "#b58c2a", "judge": "#5f9c6a", "place": "#a8623c", "check": "#5f9c6a", "finish": "#4a7a9c", "part_views": "#a8623c", "part_colours": "#a8623c", "cut_part": "#a8623c"}


def links(node):
    """{socket: [node ids]} of a node."""
    return {socket: ([source] if isinstance(source, str) else list(source)) for socket, source in node.get("in", {}).items()}


def needed(nodes, target):
    """The target and every node it needs."""
    keep, todo = set(), [target]
    while todo:
        one = todo.pop()
        if one not in keep:
            keep.add(one)
            todo += [source for sources in links(nodes[one]).values() for source in sources]
    return keep


def with_preview(outputs, folder, render=True):
    """A node that made or chose a model and has no picture of its own gets one: the model from four sides.
    render=False only takes a picture that is there already (the web page: it must not start Blender)."""
    model = (outputs.get("model") or {}).get("glb")
    if outputs.get("sheet") or not model or not folder or not os.path.exists(model):
        return outputs
    sheet = os.path.join(folder, "preview.jpg")
    if not os.path.exists(sheet) and render:
        subprocess.run([sys.executable, os.path.join(core.SCRIPTS, "pipeline", "preview.py"), "compare", sheet, model],
                       env={**os.environ, "BLENDER": BLENDER}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return {**outputs, "sheet": sheet} if os.path.exists(sheet) else outputs


def caches_off_a_full_home(least=500 << 20):
    """The settings that keep TRELLIS.2's caches off the home folder while its disk is (nearly) full: flex_gemm saves its
    autotune cache to ~/.flex_gemm after a run and Triton compiles into ~/.triton; on the uni server /home had 0 bytes
    left (Nov 19) and the part's model failed in that last save. Both are only for speed: then not saved / kept in /tmp."""
    import shutil
    import tempfile
    try:
        if shutil.disk_usage(os.path.expanduser("~")).free >= least:
            return {}
    except OSError:
        return {}
    return {"FLEX_GEMM_AUTOSAVE_AUTOTUNE_CACHE": "0",
            "TRITON_CACHE_DIR": os.environ.get("TRITON_CACHE_DIR") or os.path.join(tempfile.gettempdir(), f"triton-{os.getuid()}")}


class Flow:
    def __init__(self, path):
        self.file = json.load(open(path))
        self.name = self.file["name"]
        self.nodes = {n["id"]: n for n in self.file["nodes"]}
        self.folder = core.flow_folder(self.name)
        os.makedirs(self.folder, exist_ok=True)
        self.state = {key: {"status": "waiting", "outputs": None, "folder": None, "note": ""} for key in self.nodes}
        self.lock, self.gpu = threading.Lock(), threading.Lock()

    def settings(self, node):
        return {k: v for k, v in node.items() if k not in ("id", "type", "in", "label")}

    def ready(self, key):
        return all(self.state[s]["status"] == "done" for sources in links(self.nodes[key]).values() for s in sources)

    def run_node(self, key):
        node, state = self.nodes[key], self.state[key]
        try:
            given = {}
            for socket, sources in links(node).items():
                results = [self.state[s]["outputs"] for s in sources]
                given[socket] = results if socket == "candidates" else results[0]
            job, outputs = TYPES[node["type"]][3](given, self.settings(node), self.name)
            if job is not None:
                outputs, state["folder"] = job["outputs"], job["folder"]
                todo = [step for step in job["steps"] if not (step[2] and os.path.exists(step[2]))]
                if todo:
                    # this PC has one GPU: its own work one after another (also TRELLIS.2 with where = pc, two seeds at once
                    # would not fit); the server's work side by side
                    on_this_pc = node["type"] in LOCAL_GPU or self.settings(node).get("where") == "pc"
                    with self.gpu if on_this_pc else threading.Lock():
                        self.set(key, "running", todo[0][1])
                        start = time.time()
                        for command, label, done in todo:
                            self.run_step(key, command, label, done, job["folder"])
                        state["minutes"] = round((time.time() - start) / 60, 1)
                        json.dump({"minutes": state["minutes"]}, open(os.path.join(job["folder"], "time.json"), "w"))
                elif os.path.isfile(os.path.join(job["folder"], "time.json")):
                    state["minutes"] = json.load(open(os.path.join(job["folder"], "time.json")))["minutes"]
            state["outputs"] = with_preview(core.resolve(dict(outputs)), state["folder"])
            if self.rejected(key):
                return self.set(key, "failed", self.rejected(key))
            self.set(key, "done", state["outputs"].get("verdict", ""))
        except Exception as e:  # noqa: BLE001 (shown on the node; the nodes after it stay waiting)
            self.set(key, "failed", f"{type(e).__name__}: {e}"[:300])

    def peek(self, previews=True):
        """What is there already, without running anything: a node whose results exist is done."""
        for key, node in self.nodes.items():  # a flow file lists a node after the ones it needs
            if not self.ready(key):
                continue
            try:
                given = {}
                for socket, sources in links(node).items():
                    results = [self.state[s]["outputs"] for s in sources]
                    given[socket] = results if socket == "candidates" else results[0]
                job, outputs = TYPES[node["type"]][3](given, self.settings(node), self.name)
                if job is not None:
                    self.state[key]["folder"] = job["folder"]
                    if not all(done and os.path.exists(done) for _, _, done in job["steps"]):
                        continue
                    outputs = job["outputs"]
                    if os.path.isfile(os.path.join(job["folder"], "time.json")):
                        self.state[key]["minutes"] = json.load(open(os.path.join(job["folder"], "time.json")))["minutes"]
                self.state[key]["outputs"] = with_preview(core.resolve(dict(outputs)), self.state[key]["folder"], previews)
                if self.rejected(key):
                    self.state[key].update(status="failed", note=self.rejected(key))
                    continue
                self.state[key].update(status="done", note=self.state[key]["outputs"].get("verdict", ""))
            except Exception as e:  # noqa: BLE001 (it shows when the node runs)
                self.state[key]["note"] = f"{type(e).__name__}: {e}"[:200]

    def rejected(self, key):
        """A judge that says redo stops the flow there: the steps after it would build on a model it rejected (the
        toaster's book, Nov 19: details judged redo, part 46, and the colours step started on it anyway). A node
        with "stop_on_redo": false lets it go on, to look at what comes of it."""
        verdict = str((self.state[key]["outputs"] or {}).get("verdict") or "")
        if self.nodes[key]["type"] == "judge" and verdict.startswith("redo") and self.settings(self.nodes[key]).get("stop_on_redo", True):
            return f"the judge says {verdict}; the steps after it are not run on a rejected model"
        return ""

    def run_step(self, key, command, label, done, folder):
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "BLENDER": BLENDER, **caches_off_a_full_home()}
        log = os.path.join(folder, "log.txt")
        with open(log, "a") as out:
            out.write(f"\n$ {' '.join(map(str, command))}\n")
            out.flush()
            process = subprocess.Popen([sys.executable if command[0] == "python" else command[0], *map(str, command[1:])],
                                       cwd=core.PROJECT, env=env, stdout=out, stderr=subprocess.STDOUT)
            while process.poll() is None:
                time.sleep(2)
                try:  # the last line of the log is the node's progress (read often: the web page follows it live)
                    tail = open(log, errors="replace").read()[-400:].replace("\r", "\n").strip().splitlines()[-1]
                except IndexError:
                    tail = ""
                self.set(key, "running", f"{label}: {tail[:110]}", quiet=True)
        if process.returncode != 0 or (done and not os.path.exists(done)):
            tail = open(log, errors="replace").read()[-600:].strip().splitlines()[-3:]
            raise RuntimeError(f"{label} failed: {' | '.join(tail)[-240:]} (see {os.path.relpath(log, core.PROJECT)})")

    def set(self, key, status, note="", quiet=False):
        with self.lock:
            self.state[key].update(status=status, note=note)
            if not quiet:
                print(f"{time.strftime('%H:%M:%S')} {key}: {status} {note}", flush=True)
            self.write()

    def run(self, upto=None):
        keys = needed(self.nodes, upto) if upto else set(self.nodes)
        threads = {}
        while True:
            for key in self.nodes:  # the file's order
                if key in keys and key not in threads and self.state[key]["status"] == "waiting" and self.ready(key):
                    threads[key] = threading.Thread(target=self.run_node, args=(key,), daemon=True)
                    self.state[key]["status"] = "starting"
                    threads[key].start()
            if not any(t.is_alive() for t in threads.values()):
                if not any(self.state[k]["status"] == "waiting" and self.ready(k) for k in keys):
                    break
            time.sleep(1)
        self.write()
        failed = [k for k in keys if self.state[k]["status"] == "failed"]
        left = [k for k in keys if self.state[k]["status"] == "waiting"]
        print(f"flow {self.name}: {sum(self.state[k]['status'] == 'done' for k in keys)} of {len(keys)} nodes done"
              + (f", failed: {failed}" if failed else "") + (f", not reached: {left}" if left else ""), flush=True)
        print(f"Result: {os.path.join(self.folder, 'flow.html')}", flush=True)
        # the flow's models as versions with a title in outputs/<object>, next to its work folder
        subprocess.run([sys.executable, os.path.join(core.SCRIPTS, "tools", "arrange_outputs.py"), "--do"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return not failed

    # ---------------------------------------------------------------- the structure picture

    def write(self):
        json.dump({"name": self.name, "nodes": self.state}, open(os.path.join(self.folder, "flow_state.json"), "w"),
                  indent=1, default=str)
        with open(os.path.join(self.folder, "flow.html.part"), "w") as f:
            f.write(self.page())
        os.replace(os.path.join(self.folder, "flow.html.part"), os.path.join(self.folder, "flow.html"))

    def layout(self):
        """Column = how many nodes come before it at most; rows in the file's order."""
        depth = {}
        for key in self.nodes:  # a flow file lists a node after the ones it needs
            sources = [s for sources in links(self.nodes[key]).values() for s in sources]
            depth[key] = 1 + max((depth[s] for s in sources), default=-1)
        rows, place = {}, {}
        for key in self.nodes:
            place[key] = (depth[key], rows.get(depth[key], 0))
            rows[depth[key]] = rows.get(depth[key], 0) + 1
        return place, max(rows.values())

    def picture(self, key):
        outputs = self.state[key]["outputs"] or {}
        path = outputs.get("sheet")
        return os.path.relpath(path, self.folder) if path and os.path.exists(path) else None

    def page(self):
        place, tallest = self.layout()
        w, h, gap_x, gap_y = 250, 236, 70, 26
        width = (max(c for c, _ in place.values()) + 1) * (w + gap_x)
        height = tallest * (h + gap_y)
        at = lambda key: (place[key][0] * (w + gap_x), place[key][1] * (h + gap_y))
        wires, cards, details = [], [], []
        for key, node in self.nodes.items():
            x, y = at(key)
            for number, (socket, sources) in enumerate(links(node).items()):
                for source in sources:
                    sx, sy = at(source)
                    x0, y0, x1, y1 = sx + w, sy + 22, x, y + 22 + 16 * number
                    wires.append(f'<path d="M{x0},{y0} C{x0 + 40},{y0} {x1 - 40},{y1} {x1},{y1}" class="wire"/>')
            state = self.state[key]
            title, sockets, where, _ = TYPES[node["type"]]
            label = node.get("label") or (f"{title}: {node['step']}" if "step" in node else title)
            shown = {k: v for k, v in self.settings(node).items() if k != "files"}
            if "files" in node:
                shown["files"] = ", ".join(os.path.basename(f) for f in node["files"])
            lines = "".join(f"<div class='set'><b>{html.escape(k)}</b> {html.escape(str(v))}</div>" for k, v in shown.items())
            picture = self.picture(key)
            image = f"<a href='#{key}'><img src='{html.escape(picture)}?{int(time.time())}'></a>" if picture else ""
            minutes = f" · {state['minutes']} min" if state.get("minutes") else ""
            ins = "".join(f"<span class='socket'>&#9679; {html.escape(s)}</span>" for s in links(node))
            cards.append(
                f"<div class='card {state['status']}' style='left:{x}px;top:{y}px;width:{w}px;height:{h}px'>"
                f"<div class='head' style='background:{COLOURS[node['type']]}'>{html.escape(label)}<span class='id'>{html.escape(key)}</span></div>"
                f"<div class='body'><div class='where'>{html.escape(where)}</div><div class='ins'>{ins}</div>{lines}"
                f"<div class='status'>{state['status']}{minutes}</div><div class='note'>{html.escape(state['note'][:120])}</div>{image}</div></div>")
            if picture or state["outputs"]:
                outputs = state["outputs"] or {}
                model = (outputs.get("model") or {}).get("glb")
                facts = [f"folder: {os.path.relpath(state['folder'], core.PROJECT)}" if state["folder"] else "",
                         f"model: {os.path.relpath(model, core.PROJECT)}" if model else "",
                         f"verdict: {outputs['verdict']}" if outputs.get("verdict") else ""]
                big = f"<img src='{html.escape(picture)}?{int(time.time())}'>" if picture else ""
                extra = outputs.get("area_map")
                if extra and os.path.exists(extra):
                    big += f"<img src='{html.escape(os.path.relpath(extra, self.folder))}'>"
                details.append(f"<section id='{key}'><h2>{html.escape(label)} <small>{html.escape(key)}</small></h2>"
                               f"<p>{'<br>'.join(html.escape(x) for x in facts if x)}</p>{big}</section>")
        running = any(s["status"] in ("running", "starting") for s in self.state.values())
        refresh = "<meta http-equiv='refresh' content='20'>" if running else ""
        return f"""<!doctype html><html><head><meta charset='utf-8'>{refresh}<title>Flow {html.escape(self.name)}</title><style>
:root{{--bg:#1d1d20;--card:#2b2b30;--text:#e8e8ea;--dim:#a0a0a8;--wire:#8a8a95}}
@media (prefers-color-scheme: light){{:root{{--bg:#f3f3f1;--card:#fff;--text:#1c1c1f;--dim:#666;--wire:#888}}}}
body{{background:var(--bg);color:var(--text);font:14px system-ui,sans-serif;margin:24px}}
h1{{font-size:20px;margin:0 0 4px}} .sub{{color:var(--dim);margin-bottom:18px}}
.graph{{position:relative;overflow-x:auto;padding-bottom:12px}} .inner{{position:relative;width:{width}px;height:{height}px}}
svg{{position:absolute;left:0;top:0}} .wire{{fill:none;stroke:var(--wire);stroke-width:2}}
.card{{position:absolute;background:var(--card);border-radius:8px;overflow:hidden;box-shadow:0 1px 4px #0005;border:2px solid transparent}}
.card.running,.card.starting{{border-color:#e0a030}} .card.failed{{border-color:#e05040}} .card.done{{border-color:#4f9c62}}
.head{{color:#fff;font-weight:600;padding:5px 9px;font-size:13px}} .id{{float:right;font-weight:400;opacity:.8;font-size:11px}}
.body{{padding:6px 9px;font-size:12px}} .where{{color:var(--dim);font-size:11px}} .socket{{margin-right:8px;color:var(--dim);font-size:11px}}
.set b{{font-weight:600;color:var(--dim)}} .status{{margin-top:4px;font-weight:600}} .note{{color:var(--dim);font-size:11px;max-height:28px;overflow:hidden}}
.card img{{max-width:100%;max-height:78px;margin-top:4px;border-radius:4px;display:block}}
section{{margin-top:34px}} section img{{max-width:100%;border-radius:6px;display:block;margin-top:8px}} small{{color:var(--dim);font-weight:400}}
</style></head><body><h1>Flow: {html.escape(self.name)}</h1>
<div class='sub'>{time.strftime('%d %b %Y, %H:%M:%S')} · green = done, yellow = running, red = failed, no border = waiting ·
gold nodes are TRELLIS.2, green nodes judge, the others prepare</div>
<div class='graph'><div class='inner'><svg width='{width}' height='{height}'>{''.join(wires)}</svg>{''.join(cards)}</div></div>
{''.join(details)}</body></html>"""


def main():
    parser = argparse.ArgumentParser(description="Run a node flow from a .json file")
    parser.add_argument("flow")
    parser.add_argument("--upto", help="only this node and what it needs")
    parser.add_argument("--structure", action="store_true", help="run nothing, only write the structure picture")
    args = parser.parse_args()
    flow = Flow(args.flow)
    if args.structure:
        flow.peek()
        flow.write()
        print(f"Result: {os.path.join(flow.folder, 'flow.html')}")
        return
    if not flow.run(args.upto):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
